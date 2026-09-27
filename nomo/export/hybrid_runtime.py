"""Backend-agnostic float runtime for an optimised hybrid design (SPEC §6.4).

SELF-CONTAINED ON PURPOSE: this file is embedded verbatim into generated `deploy_model.py`
scripts, so it may import nothing but the standard library. All tensor math goes through an
`ops` object (NumpyOps here; TorchOps inside generated scripts), so the numpy path tested in
this repository and the PyTorch path users run share every line of model logic.

Plan format (JSON-compatible dict; arrays are numpy/torch tensors after loading):
    {"input_shape": [...], "n_aux": int, "layers": [LayerPlan...], "guards": [GuardPlan...]}
    LayerPlan: name, op (dense|conv2d), domain (ANN|SNN|SYM), activation (relu|linear), W, b,
               stride, padding, pool ({type, kernel, stride} | None), flatten_input,
               act_bits (ANN activation fake-quant; 0 = off), act_amax,
               coding (rate|ttfs), T, lam_in, lam_out          (SNN only)
    GuardPlan: after (layer index), terms: [{kind: box|thrust|rotational_rate, ...}]

Semantics
    ANN     y = act(op(x)); pool; optional symmetric fake-quantisation of activations
    SYM     y = op(x) with the exact physics matrix (linear)
    SNN     a maximal run of SNN layers shares (coding, T). Input values are encoded to spikes,
            integrate-and-fire neurons with threshold lambda_out propagate spikes layer to layer
            (each spike of the previous layer carries lambda_in), and the segment output is decoded.
              rate: signed sigma-delta encoder; reset by subtraction; decode count * lambda / T
              ttfs: two-phase time-to-first-spike; each value quantised to T spike-time bins
    guards  exact box projections on the outputs (see nomo/symbolic/constraints.py)
"""
import math


class NumpyOps:
    name = "numpy"

    def __init__(self):
        import numpy as np
        self.np = np

    def asarray(self, x):
        return self.np.asarray(x, dtype=self.np.float64)

    def zeros_like(self, x):
        return self.np.zeros_like(x)

    def maximum(self, a, b):
        return self.np.maximum(a, b)

    def minimum(self, a, b):
        return self.np.minimum(a, b)

    def where(self, c, a, b):
        return self.np.where(c, a, b)

    def clip(self, x, lo, hi):
        return self.np.minimum(self.np.maximum(x, lo), hi)

    def rint(self, x):
        return self.np.floor(x + 0.5)

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def linear(self, x, W, b):
        return x @ W.T + b

    def conv2d(self, x, W, b, stride, pad):
        # process two samples at a time: keeps the unfolded patches small (memory-bounded servers)
        if x.shape[0] > 2:
            return self.np.concatenate([self._conv2d(x[i:i + 2], W, b, stride, pad) for i in range(0, x.shape[0], 2)])
        return self._conv2d(x, W, b, stride, pad)

    def _conv2d(self, x, W, b, stride, pad):
        np = self.np
        B, C, H, Wd = x.shape
        O, _, k, _ = W.shape
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        oh, ow = (H + 2 * pad - k) // stride + 1, (Wd + 2 * pad - k) // stride + 1
        cols = np.lib.stride_tricks.sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
        cols = cols[:, :, :oh, :ow]                                   # B, C, oh, ow, k, k
        out = np.einsum("bchwij,ocij->bohw", cols, W, optimize=True)
        return out + b.reshape(1, -1, 1, 1)

    def pool(self, x, kind, k, s):
        np = self.np
        if kind == "global_avg":
            return x.mean(axis=(2, 3), keepdims=True)
        win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
        oh, ow = (x.shape[2] - k) // s + 1, (x.shape[3] - k) // s + 1
        win = win[:, :, :oh, :ow]
        return win.max(axis=(4, 5)) if kind == "max" else win.mean(axis=(4, 5))

    def stack_cols(self, cols):
        return self.np.stack(cols, axis=1)

    def col(self, x, i):
        return x[:, i]


# ---------------------------------------------------------------------------

def _op(ops, L, x):
    if L["op"] == "conv2d":
        return ops.conv2d(x, L["W"], L["b"], int(L["stride"]), int(L["padding"]))
    if L.get("flatten_input") and len(x.shape) > 2:
        x = ops.flatten(x)
    return ops.linear(x, L["W"], L["b"])


def _pool(ops, L, x):
    p = L.get("pool")
    if not p:
        return x
    return ops.pool(x, p["type"], int(p.get("kernel", 0) or 0), int(p.get("stride", p.get("kernel", 0)) or 0))


def _fake_quant(ops, y, bits, amax):
    if not bits or bits >= 16 or amax <= 0:
        return y
    q = float(2 ** (bits - 1) - 1)
    s = amax / q
    return ops.clip(ops.rint(y / s), -q, q) * s


def _continuous(ops, L, x):
    y = _op(ops, L, x)
    if L["activation"] == "relu":
        y = ops.maximum(y, 0.0)
    y = _pool(ops, L, y)
    if L["domain"] == "ANN":
        y = _fake_quant(ops, y, int(L.get("act_bits", 0)), float(L.get("act_amax", 0.0)))
    return y


def _ttfs_quant(ops, x, lam, T):
    """Time-to-first-spike code: value in [0, lam] -> spike at bin t = rint((1 - x/lam)(T-1)),
    decoded back as lam (1 - t/(T-1)). Earlier spike = larger value; sign rides on the spike."""
    mag = ops.clip(ops.where(x >= 0, x, -x) / lam, 0.0, 1.0)
    t = ops.rint((1.0 - mag) * (T - 1))
    live = ops.where(mag > 0, 1.0, 0.0)
    sign = ops.where(x >= 0, 1.0, -1.0)
    return sign * live * lam * (1.0 - t / (T - 1))


def _snn_segment(ops, seg, x):
    """Simulate a run of SNN layers sharing (coding, T). Returns decoded values of the last layer.

    rate: explicit timestep simulation (signed sigma-delta encoder, integrate-and-fire neurons with
          reset by subtraction, decode = spike count * lambda / T).
    ttfs: two-phase TTFS (integrate during one window, then a ramp makes neurons with larger
          membrane fire earlier in the next). In discrete time this is exactly: each layer's ReLU
          output quantised to T spike-time bins, which is what is computed here in closed form.
    """
    T = int(seg[0]["T"])
    lam = float(seg[0]["lam_in"])
    if seg[0]["coding"] == "ttfs":
        v = _ttfs_quant(ops, x, lam, T)
        for L in seg:
            y = _op(ops, L, v)
            y = _pool(ops, L, ops.maximum(y, 0.0))
            v = _ttfs_quant(ops, y, float(L["lam_out"]), T)
        return v
    acc = ops.zeros_like(x)
    spikes_in = []
    for _ in range(T):                                   # signed sigma-delta encoder
        acc = acc + x / lam
        s = ops.where(acc >= 1.0, 1.0, 0.0) - ops.where(acc <= -1.0, 1.0, 0.0)
        acc = acc - s
        spikes_in.append(s)
    for L in seg:
        thr = float(L["lam_out"])
        v, out = None, []
        for t in range(T):
            cur = _op(ops, L, spikes_in[t] * lam)
            v = cur if v is None else v + cur
            z = ops.where(v >= thr, 1.0, 0.0)
            v = v - thr * z                              # reset by subtraction
            out.append(_pool(ops, L, z))
        spikes_in, lam = out, thr
    total = spikes_in[0]
    for s in spikes_in[1:]:
        total = total + s
    return total * (lam / T)


def _guard(ops, G, y, aux):
    cols = [ops.col(y, i) for i in range(y.shape[1])]
    for term in G["terms"]:
        k = term["kind"]
        if k == "box":
            for c, lo, hi in zip(term["channels"], term["lo"], term["hi"]):
                cols[c] = ops.clip(cols[c], float(lo), float(hi))
        elif k == "thrust":
            cap = float(term["t_max_n"]) * ops.col(aux, int(term["aux_density_ratio"]))
            cols[term["channel"]] = ops.minimum(ops.maximum(cols[term["channel"]], 0.0), cap)
        elif k == "rotational_rate":
            I = [float(v) for v in term["inertia"]]
            w = [ops.col(aux, int(i)) for i in term["aux_omega"]]
            g = [(I[2] - I[1]) * w[1] * w[2], (I[0] - I[2]) * w[2] * w[0], (I[1] - I[0]) * w[0] * w[1]]
            wmax, tmax, dt = float(term["w_max"]), float(term["tau_max"]), float(term["dt"])
            for ax, c in enumerate(term["channels"]):
                r_lo = I[ax] * (-wmax - w[ax]) / dt + g[ax]
                r_hi = I[ax] * (wmax - w[ax]) / dt + g[ax]
                inner = ops.minimum(ops.maximum(cols[c], ops.maximum(r_lo, -tmax)), ops.minimum(r_hi, tmax))
                cols[c] = ops.where(r_lo > tmax, tmax + 0.0 * inner, ops.where(r_hi < -tmax, -tmax + 0.0 * inner, inner))
        else:
            raise ValueError("unknown guard term " + str(k))
    return ops.stack_cols(cols)


def forward(ops, plan, x, aux=None):
    """Run the hybrid design on a batch. x: [B, *input_shape]; aux: [B, n_aux] or None."""
    x = ops.asarray(x)
    layers = plan["layers"]
    guards = {int(G["after"]): G for G in plan.get("guards", [])}
    i = 0
    while i < len(layers):
        L = layers[i]
        if L["domain"] == "SNN":
            j = i
            while j + 1 < len(layers) and layers[j + 1]["domain"] == "SNN" and (j not in guards):
                j += 1
            x = _snn_segment(ops, layers[i:j + 1], x)
            i = j
        else:
            x = _continuous(ops, L, x)
        if i in guards:
            if aux is None:
                raise ValueError("this design has safety guards; pass aux (e.g. body rates, air density ratio)")
            x = _guard(ops, guards[i], ops.flatten(x) if len(x.shape) > 2 else x, ops.asarray(aux))
        i += 1
    return x


def max_rel_error(a, b):
    num = max(abs(float(u) - float(v)) for u, v in zip(a, b)) if len(a) else 0.0
    den = max(max(abs(float(v)) for v in b), 1e-12) if len(b) else 1.0
    return num / den

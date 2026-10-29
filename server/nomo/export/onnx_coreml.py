"""ONNX and CoreML exporters (SPEC §7.5).

Both formats describe continuous computation, so they export the design's *continuous sections*:
maximal runs of ANN and SYM layers (SYM layers are exact linear maps). Spiking sections are not
representable in ONNX/CoreML and stay in the NIR / PyTorch exports; a design that is continuous
end to end therefore exports as one complete model, safety guards included.

A section is first lowered to a tiny op list (flatten, dense, conv, relu, pool, fake-quant, guard),
then emitted by one of two small backends, so ONNX and CoreML graphs are structurally identical.

Activation precision is emitted as explicit fake-quantisation (x/s -> floor(+0.5) -> clip -> *s),
which reproduces the reference runtime bit for bit in float32 up to rounding-boundary ties.
Standard QDQ nodes were not used because their round-half-to-even rule differs from the compiler's
round-half-up rule, which would change results.
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Section = Tuple[int, int]


def continuous_sections(plan: Dict[str, Any]) -> List[Section]:
    out, i, L = [], 0, plan["layers"]
    while i < len(L):
        if L[i]["domain"] == "SNN":
            i += 1
            continue
        j = i
        while j + 1 < len(L) and L[j + 1]["domain"] != "SNN":
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def section_ops(plan: Dict[str, Any], s: int, e: int) -> Tuple[List[Dict[str, Any]], List[int], int, List[int]]:
    """Lower layers s..e to an op list. Returns (ops, input_shape, n_aux_needed, output_shape)."""
    L = plan["layers"]
    in_shape = list(plan["input_shape"]) if s == 0 else list(L[s - 1]["out_shape"])
    guards = {int(G["after"]): G for G in plan.get("guards", [])}
    ops: List[Dict[str, Any]] = []
    n_aux = 0
    rank = len(in_shape)
    for i in range(s, e + 1):
        l = L[i]
        if l["op"] == "dense":
            if rank > 1:
                ops.append({"op": "flatten"})
                rank = 1
            ops.append({"op": "dense", "W": np.asarray(l["W"], np.float32), "b": np.asarray(l["b"], np.float32)})
        else:
            W = np.asarray(l["W"], np.float32)
            ops.append({"op": "conv", "W": W, "b": np.asarray(l["b"], np.float32),
                        "stride": int(l["stride"]), "pad": int(l["padding"]), "k": int(W.shape[2])})
        if l["activation"] == "relu":
            ops.append({"op": "relu"})
        if l.get("pool"):
            p = l["pool"]
            ops.append({"op": "pool", "kind": p["type"], "k": int(p.get("kernel", 0) or 0),
                        "s": int(p.get("stride", p.get("kernel", 0)) or 0)})
        if l["domain"] == "ANN" and l.get("act_bits") and int(l["act_bits"]) < 16 and float(l["act_amax"]) > 0:
            q = float(2 ** (int(l["act_bits"]) - 1) - 1)
            ops.append({"op": "fq", "scale": float(l["act_amax"]) / q, "q": q})
        rank = len(l["out_shape"])
        if i in guards:
            if rank > 1:
                ops.append({"op": "flatten"})
                rank = 1
            ops.append({"op": "guard", "terms": guards[i]["terms"], "n": int(np.prod(l["out_shape"]))})
            n_aux = max(n_aux, int(guards[i]["n_aux"]))
    out_shape = [int(np.prod(L[e]["out_shape"]))] if rank == 1 else [int(v) for v in L[e]["out_shape"]]
    return ops, in_shape, n_aux, out_shape


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------

def to_onnx(plan: Dict[str, Any], s: int, e: int, name: str):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ops, in_shape, n_aux, out_shape = section_ops(plan, s, e)
    nodes, inits = [], []
    cnt = [0]

    def nm(p: str) -> str:
        cnt[0] += 1
        return f"{p}_{cnt[0]}"

    def const(v, dtype=np.float32) -> str:
        n = nm("c")
        inits.append(numpy_helper.from_array(np.asarray(v, dtype=dtype), n))
        return n

    def node(op: str, inputs: List[str], **attrs) -> str:
        out = nm(op.lower())
        nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    x = "input"
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N"] + in_shape)]
    if n_aux:
        inputs.append(helper.make_tensor_value_info("aux", TensorProto.FLOAT, ["N", n_aux]))
    for o in ops:
        k = o["op"]
        if k == "flatten":
            x = node("Flatten", [x], axis=1)
        elif k == "dense":
            x = node("Gemm", [x, const(o["W"]), const(o["b"])], transB=1)
        elif k == "conv":
            x = node("Conv", [x, const(o["W"]), const(o["b"])], strides=[o["stride"]] * 2,
                     pads=[o["pad"]] * 4, kernel_shape=[o["k"]] * 2)
        elif k == "relu":
            x = node("Relu", [x])
        elif k == "pool":
            if o["kind"] == "global_avg":
                x = node("GlobalAveragePool", [x])
            else:
                x = node("MaxPool" if o["kind"] == "max" else "AveragePool", [x],
                         kernel_shape=[o["k"]] * 2, strides=[o["s"]] * 2)
        elif k == "fq":
            t = node("Div", [x, const(o["scale"])])
            t = node("Floor", [node("Add", [t, const(0.5)])])
            t = node("Clip", [t, const(-o["q"]), const(o["q"])])
            x = node("Mul", [t, const(o["scale"])])
        elif k == "guard":
            x = _onnx_guard(o["terms"], o["n"], x, "aux", node, const)
    nodes.append(helper.make_node("Identity", [x], ["output"], name="output"))
    graph = helper.make_graph(nodes, name, inputs, [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N"] + out_shape)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="nomo")
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def _onnx_guard(terms, n: int, y, aux, node, const) -> str:
    cols: Dict[int, str] = {}

    def col(t: str, i: int) -> str:
        return node("Gather", [t, const(np.int64(i), np.int64)], axis=1)

    def get(c: int) -> str:
        if c not in cols:
            cols[c] = col(y, c)
        return cols[c]

    for term in terms:
        k = term["kind"]
        if k == "box":
            for c, lo, hi in zip(term["channels"], term["lo"], term["hi"]):
                cols[c] = node("Clip", [get(c), const(float(lo)), const(float(hi))])
        elif k == "thrust":
            c = int(term["channel"])
            cap = node("Mul", [col(aux, int(term["aux_density_ratio"])), const(float(term["t_max_n"]))])
            cols[c] = node("Min", [node("Max", [get(c), const(0.0)]), cap])
        elif k == "rotational_rate":
            I = [float(v) for v in term["inertia"]]
            w = [col(aux, int(i)) for i in term["aux_omega"]]
            g = [node("Mul", [node("Mul", [w[1], w[2]]), const(I[2] - I[1])]),
                 node("Mul", [node("Mul", [w[2], w[0]]), const(I[0] - I[2])]),
                 node("Mul", [node("Mul", [w[0], w[1]]), const(I[1] - I[0])])]
            wmax, tmax, dt = float(term["w_max"]), float(term["tau_max"]), float(term["dt"])
            for ax, c in enumerate(term["channels"]):
                r_lo = node("Add", [node("Mul", [node("Sub", [const(-wmax), w[ax]]), const(I[ax] / dt)]), g[ax]])
                r_hi = node("Add", [node("Mul", [node("Sub", [const(wmax), w[ax]]), const(I[ax] / dt)]), g[ax]])
                lo = node("Max", [r_lo, const(-tmax)])
                hi = node("Min", [r_hi, const(tmax)])
                inner = node("Min", [node("Max", [get(int(c)), lo]), hi])
                out = node("Where", [node("Less", [r_hi, const(-tmax)]), node("Add", [node("Mul", [inner, const(0.0)]), const(-tmax)]), inner])
                cols[int(c)] = node("Where", [node("Greater", [r_lo, const(tmax)]),
                                              node("Add", [node("Mul", [inner, const(0.0)]), const(tmax)]), out])
        else:
            raise ValueError(f"unknown guard term {k}")
    # untouched channels pass through unchanged; rebuild [N, n] in channel order
    return node("Concat", [node("Unsqueeze", [cols.get(c) or col(y, c), const(np.array([1], np.int64), np.int64)])
                           for c in range(n)], axis=1)


def run_onnx(model, x: np.ndarray, aux: Optional[np.ndarray] = None) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feed = {"input": np.asarray(x, np.float32)}
    if aux is not None and any(i.name == "aux" for i in sess.get_inputs()):
        feed["aux"] = np.asarray(aux, np.float32)
    return sess.run(["output"], feed)[0]


# ---------------------------------------------------------------------------
# CoreML (ML Program via the MIL builder; no PyTorch required)
# ---------------------------------------------------------------------------

def to_coreml_package(plan: Dict[str, Any], s: int, e: int, name: str) -> bytes:
    """Build an .mlpackage for section s..e and return it zipped. Building works on any OS;
    running it requires macOS/iOS (Core ML runtime)."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    ops, in_shape, n_aux, _ = section_ops(plan, s, e)
    specs = [mb.TensorSpec(shape=(1, *in_shape))]
    if n_aux:
        specs.append(mb.TensorSpec(shape=(1, n_aux)))

    def body(x, aux=None):
        for o in ops:
            k = o["op"]
            if k == "flatten":
                x = mb.reshape(x=x, shape=[1, -1])
            elif k == "dense":
                x = mb.linear(x=x, weight=o["W"], bias=o["b"])
            elif k == "conv":
                x = mb.conv(x=x, weight=o["W"], bias=o["b"], strides=[o["stride"]] * 2,
                            pad_type="custom", pad=[o["pad"]] * 4)
            elif k == "relu":
                x = mb.relu(x=x)
            elif k == "pool":
                if o["kind"] == "global_avg":
                    x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True)
                elif o["kind"] == "max":
                    x = mb.max_pool(x=x, kernel_sizes=[o["k"]] * 2, strides=[o["s"]] * 2, pad_type="valid")
                else:
                    x = mb.avg_pool(x=x, kernel_sizes=[o["k"]] * 2, strides=[o["s"]] * 2, pad_type="valid")
            elif k == "fq":
                t = mb.floor(x=mb.add(x=mb.real_div(x=x, y=np.float32(o["scale"])), y=np.float32(0.5)))
                x = mb.mul(x=mb.clip(x=t, alpha=np.float32(-o["q"]), beta=np.float32(o["q"])), y=np.float32(o["scale"]))
            elif k == "guard":
                x = _mil_guard(mb, o["terms"], o["n"], x, aux)
        return mb.identity(x=x, name="output")

    if n_aux:
        @mb.program(input_specs=specs)
        def prog(input, aux):  # noqa: A002 - MIL uses parameter names as model input names
            return body(input, aux)
    else:
        @mb.program(input_specs=specs)
        def prog(input):  # noqa: A002
            return body(input)

    model = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                       minimum_deployment_target=ct.target.iOS16, skip_model_load=True)
    model.short_description = f"Nomo continuous section {s}-{e} of {name}"
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, f"{name}.mlpackage")
        model.save(path)
        ct.models.MLModel(path, skip_model_load=True)          # structural validation (reload)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(path):
                for f in files:
                    full = os.path.join(root, f)
                    zf.write(full, os.path.relpath(full, tmp))
        return buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _mil_guard(mb, terms, n, y, aux):
    cols = {c: mb.slice_by_index(x=y, begin=[0, c], end=[1, c + 1]) for c in range(n)}

    def a(i):
        return mb.slice_by_index(x=aux, begin=[0, int(i)], end=[1, int(i) + 1])

    f = np.float32
    for term in terms:
        k = term["kind"]
        if k == "box":
            for c, lo, hi in zip(term["channels"], term["lo"], term["hi"]):
                cols[c] = mb.clip(x=cols[c], alpha=f(lo), beta=f(hi))
        elif k == "thrust":
            c = int(term["channel"])
            cap = mb.mul(x=a(term["aux_density_ratio"]), y=f(term["t_max_n"]))
            cols[c] = mb.minimum(x=mb.maximum(x=cols[c], y=f(0.0)), y=cap)
        elif k == "rotational_rate":
            I = [float(v) for v in term["inertia"]]
            w = [a(i) for i in term["aux_omega"]]
            g = [mb.mul(x=mb.mul(x=w[1], y=w[2]), y=f(I[2] - I[1])),
                 mb.mul(x=mb.mul(x=w[2], y=w[0]), y=f(I[0] - I[2])),
                 mb.mul(x=mb.mul(x=w[0], y=w[1]), y=f(I[1] - I[0]))]
            wmax, tmax, dt = float(term["w_max"]), float(term["tau_max"]), float(term["dt"])
            for ax, c in enumerate(term["channels"]):
                r_lo = mb.add(x=mb.mul(x=mb.sub(x=f(-wmax), y=w[ax]), y=f(I[ax] / dt)), y=g[ax])
                r_hi = mb.add(x=mb.mul(x=mb.sub(x=f(wmax), y=w[ax]), y=f(I[ax] / dt)), y=g[ax])
                inner = mb.minimum(x=mb.maximum(x=cols[c], y=mb.maximum(x=r_lo, y=f(-tmax))),
                                   y=mb.minimum(x=r_hi, y=f(tmax)))
                neg = mb.add(x=mb.mul(x=inner, y=f(0.0)), y=f(-tmax))
                pos = mb.add(x=mb.mul(x=inner, y=f(0.0)), y=f(tmax))
                cols[c] = mb.select(cond=mb.greater(x=r_lo, y=f(tmax)), a=pos,
                                    b=mb.select(cond=mb.less(x=r_hi, y=f(-tmax)), a=neg, b=inner))
    return mb.concat(values=[cols[c] for c in range(n)], axis=1)

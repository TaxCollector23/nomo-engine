"""Generate `deploy_model.py`: a standalone script that rebuilds the optimised hybrid design in
PyTorch (nn.Module) and runs it.

The model logic is `hybrid_runtime.py`, embedded verbatim, so the PyTorch path executes exactly the
code that this repository tests with numpy. Only `TorchOps` (a thin adapter over standard
torch.nn.functional calls) is PyTorch-specific. The script ships reference inputs/outputs and a
`--check` mode that compares its own output against them, so the first run on any machine verifies
the PyTorch path end to end. Without PyTorch it falls back to numpy (`--backend numpy`).
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import hybrid_runtime

RUNTIME_SOURCE = Path(hybrid_runtime.__file__).read_text()

TORCH_ADAPTER = r'''
class TorchOps:
    """PyTorch backend for the runtime above (float64 to match the reference outputs)."""
    name = "torch"

    def __init__(self):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F

    def _t(self, v, like):
        if isinstance(v, self.torch.Tensor):
            return v
        return self.torch.as_tensor(v, dtype=self.torch.float64, device=like.device)

    def asarray(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.to(self.torch.float64)
        import numpy as np
        return self.torch.as_tensor(np.asarray(x, dtype=np.float64))

    def zeros_like(self, x):
        return self.torch.zeros_like(x)

    def maximum(self, a, b):
        like = a if isinstance(a, self.torch.Tensor) else b
        return self.torch.maximum(self._t(a, like), self._t(b, like))

    def minimum(self, a, b):
        like = a if isinstance(a, self.torch.Tensor) else b
        return self.torch.minimum(self._t(a, like), self._t(b, like))

    def where(self, c, a, b):
        return self.torch.where(c, self._t(a, c.to(self.torch.float64)), self._t(b, c.to(self.torch.float64)))

    def clip(self, x, lo, hi):
        return self.minimum(self.maximum(x, lo), hi)

    def rint(self, x):
        return self.torch.floor(x + 0.5)

    def flatten(self, x):
        return x.reshape(x.shape[0], -1)

    def linear(self, x, W, b):
        return x @ W.T + b

    def conv2d(self, x, W, b, stride, pad):
        return self.F.conv2d(x, W, b, stride=stride, padding=pad)

    def pool(self, x, kind, k, s):
        if kind == "global_avg":
            return x.mean(dim=(2, 3), keepdim=True)
        if kind == "max":
            return self.F.max_pool2d(x, k, s)
        return self.F.avg_pool2d(x, k, s)

    def stack_cols(self, cols):
        return self.torch.stack(cols, dim=1)

    def col(self, x, i):
        return x[:, i]
'''

MAIN = r'''
# ---------------------------------------------------------------------------
# loader, nn.Module wrapper and command line
# ---------------------------------------------------------------------------
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "deploy_weights.npz")


def load_plan(backend):
    import numpy as np
    z = np.load(WEIGHTS, allow_pickle=False)
    plan = json.loads(str(z["plan_json"]))
    for i, L in enumerate(plan["layers"]):
        W, b = z[f"W{i}"], z[f"b{i}"]
        if backend == "torch":
            import torch
            W, b = torch.as_tensor(W, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
        L["W"], L["b"] = W, b
    ref = {k: z[k] for k in ("ref_x", "ref_aux", "ref_y") if k in z.files}
    return plan, ref


def build_torch_model():
    """Return the optimised hybrid design as a torch.nn.Module (inference; weights are buffers)."""
    import torch

    class HybridNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.plan, _ = load_plan("torch")
            self.ops = TorchOps()
            for i, L in enumerate(self.plan["layers"]):
                self.register_buffer(f"W{i}", L["W"])
                self.register_buffer(f"b{i}", L["b"])

        def forward(self, x, aux=None):
            for i, L in enumerate(self.plan["layers"]):         # keep plan tensors on the module's device
                L["W"], L["b"] = getattr(self, f"W{i}"), getattr(self, f"b{i}")
            return forward(self.ops, self.plan, x, aux)

    return HybridNet()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify this machine reproduces the reference outputs")
    ap.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--input", help=".npy file with a batch of inputs, shape [N, " + ", ".join(map(str, INPUT_SHAPE)) + "]")
    ap.add_argument("--aux", help=".npy file with guard inputs, shape [N, %d]" % N_AUX)
    ap.add_argument("--output", help="where to save predictions (.npy)")
    a = ap.parse_args(argv)
    backend = a.backend
    if backend == "auto":
        try:
            import torch  # noqa: F401
            backend = "torch"
        except ImportError:
            backend = "numpy"
            print("PyTorch not found: using the numpy backend (pip install torch for the nn.Module).")
    import numpy as np
    if backend == "torch":
        import torch
        model = build_torch_model().eval()
        run = lambda x, aux: model(torch.as_tensor(x, dtype=torch.float64),
                                   None if aux is None else torch.as_tensor(aux, dtype=torch.float64)).detach().cpu().numpy()
    else:
        plan, _ = load_plan("numpy")
        run = lambda x, aux: np.asarray(forward(NumpyOps(), plan, x, aux))
    if a.check or not a.input:
        _, ref = load_plan("numpy")
        y = run(ref["ref_x"], ref.get("ref_aux"))
        err = max_rel_error(np.asarray(y).ravel().tolist(), ref["ref_y"].ravel().tolist())
        ok = err < 1e-6
        print(f"[{backend}] reference check: max relative error {err:.2e} -> {'PASS' if ok else 'FAIL'}")
        if not a.input:
            return 0 if ok else 1
    x = np.load(a.input)
    aux = np.load(a.aux) if a.aux else None
    y = run(x, aux)
    if a.output:
        np.save(a.output, y)
        print(f"saved {y.shape} predictions to {a.output}")
    else:
        print(y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def generate(plan: Dict[str, Any], ref_x: np.ndarray, ref_aux: Optional[np.ndarray], ref_y: np.ndarray,
             title: str, notes: str) -> Tuple[str, bytes]:
    """Return (deploy_model.py source, deploy_weights.npz bytes)."""
    meta_layers = []
    arrays: Dict[str, np.ndarray] = {}
    for i, L in enumerate(plan["layers"]):
        arrays[f"W{i}"] = np.asarray(L["W"], np.float64)
        arrays[f"b{i}"] = np.asarray(L["b"], np.float64)
        meta_layers.append({k: v for k, v in L.items() if k not in ("W", "b")})
    meta = {**{k: v for k, v in plan.items() if k != "layers"}, "layers": meta_layers}
    arrays["plan_json"] = np.array(json.dumps(meta, default=_json_default))
    arrays["ref_x"] = np.asarray(ref_x, np.float64)
    arrays["ref_y"] = np.asarray(ref_y, np.float64)
    if ref_aux is not None:
        arrays["ref_aux"] = np.asarray(ref_aux, np.float64)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)

    layer_lines = "\n".join(
        f"#   {L['name']:<14} {L['domain']:<4} {L['op']:<7}"
        + (f" {L['coding']} T={L['T']}" if L["domain"] == "SNN" else "")
        + (f" w{L.get('w_bits')}" if L["domain"] != "SYM" else " exact physics matrix")
        for L in plan["layers"])
    header = f'''#!/usr/bin/env python3
"""{title}

Rebuilds the Nomo-optimised hybrid design as a PyTorch nn.Module and runs it.

    python deploy_model.py --check                     # verify this machine reproduces the reference outputs
    python deploy_model.py --input x.npy [--aux a.npy] --output y.npy
    python deploy_model.py --backend numpy --check     # no PyTorch needed

    from deploy_model import build_torch_model
    net = build_torch_model()                          # torch.nn.Module
    y = net(x, aux)                                    # x: [N, {", ".join(map(str, plan["input_shape"]))}]

Requires numpy; PyTorch optional (>= 2.0). Weights and the design are in deploy_weights.npz.
{notes}
"""
# Design: {plan["genome"]}
{layer_lines}
import json

INPUT_SHAPE = {tuple(plan["input_shape"])!r}
N_AUX = {int(plan.get("n_aux", 0))}

# ---------------------------------------------------------------------------
# shared runtime (identical to nomo/export/hybrid_runtime.py)
# ---------------------------------------------------------------------------
'''
    body = RUNTIME_SOURCE.split('"""', 2)[2]          # drop the module docstring, keep the code
    return header + body + TORCH_ADAPTER + MAIN, buf.getvalue()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))

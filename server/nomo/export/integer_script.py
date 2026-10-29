"""Generate a standalone integer-only Python driver from a QGraph.

The generated driver is intentionally separate from the float reference
export.  It consumes int8/Q16.16 arrays, uses integer accumulators and checks
bit-for-bit against embedded golden vectors.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, Tuple

import numpy as np

from ..runtime.qgraph import (QDecoder, QDense, QEncoder, QFromQ16, QGraph, QGuard, QLIF,
                              QSymLinear, QToQ16)
from ..symbolic.constraints import BoxBound, RotationalRateBound, ThrustLimit, rot_params_q16


def _meta(qg: QGraph) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    stages, arrays = [], {}
    for i, st in enumerate(qg.stages):
        d: Dict[str, Any] = {"kind": st.kind, "name": st.name}
        if isinstance(st, QDense):
            d.update({"W": f"s{i}_W", "b": f"s{i}_b", "m0": st.m0, "shift": st.shift,
                      "relu": st.relu})
            arrays[f"s{i}_W"], arrays[f"s{i}_b"] = np.asarray(st.W, np.int8), np.asarray(st.b, np.int32)
        elif isinstance(st, QEncoder):
            d.update({"theta": st.theta, "T": st.T, "n": st.n})
        elif isinstance(st, QLIF):
            d.update({"W": f"s{i}_W", "b": f"s{i}_b", "theta": st.theta,
                      "leak_shift": st.leak_shift, "v_bits": st.v_bits, "T": st.T})
            arrays[f"s{i}_W"], arrays[f"s{i}_b"] = np.asarray(st.W, np.int8), np.asarray(st.b, np.int32)
        elif isinstance(st, QDecoder):
            d.update({"T": st.T, "n": st.n, "out": st.out, "m0": st.m0, "shift": st.shift, "k_q16": st.k_q16})
        elif isinstance(st, QToQ16):
            d.update({"k_q16": st.k_q16, "n": st.n})
        elif isinstance(st, QFromQ16):
            d.update({"m0": st.m0, "shift": st.shift, "n": st.n})
        elif isinstance(st, QSymLinear):
            d.update({"Phi": f"s{i}_Phi"})
            arrays[f"s{i}_Phi"] = np.asarray(st.Phi, np.int64)
        elif isinstance(st, QGuard):
            terms = []
            for term in st.constraint.terms:
                if isinstance(term, BoxBound):
                    terms.append({"kind": "box", "channels": list(term.channels),
                                  "lo": [int(round(v * 65536)) for v in term.lo],
                                  "hi": [int(round(v * 65536)) for v in term.hi]})
                elif isinstance(term, ThrustLimit):
                    terms.append({"kind": "thrust", "channel": term.channel,
                                  "t_max": int(round(term.t_max_n * 65536)),
                                  "aux_density_ratio": term.aux_density_ratio})
                elif isinstance(term, RotationalRateBound):
                    p = rot_params_q16(term)
                    terms.append({"kind": "rotational_rate", "channels": list(term.channels),
                                  "inertia": p["I"], "inv_dt": p["inv_dt"], "w_max": p["w_max"],
                                  "tau_max": p["tau_max"], "aux_omega": list(term.aux_omega)})
            d.update({"n": st.n, "n_aux": st.n_aux, "terms": terms})
        stages.append(d)
    return {"name": qg.name, "in_size": qg.in_size, "out_size": qg.out_size,
            "n_aux": qg.n_aux, "in_scale": qg.in_scale, "stages": stages}, arrays


SOURCE = r'''#!/usr/bin/env python3
"""Nomo native integer deployment driver.

Input arrays are int8 and auxiliary/output arrays are signed Q16.16 int32.
No floating-point arithmetic is used by the inference path.
"""
import argparse
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "deploy_integer_weights.npz")

def asr(v, n):
    v = int(v)
    if n <= 0: return v << (-n)
    return v >> n

def rsr(v, n):
    return int(v) << (-n) if n <= 0 else asr(int(v) + (1 << (n - 1)), n)

def sat(v, lo, hi): return min(max(int(v), int(lo)), int(hi))
def sat8(v): return sat(v, -128, 127)
def sat32(v): return sat(v, -(1 << 31), (1 << 31) - 1)
def q16mul(a, b): return sat32(rsr(int(a) * int(b), 16))

def load():
    z = np.load(WEIGHTS, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    arrays = {k: z[k] for k in z.files if k != "meta_json" and not k.startswith("ref_")}
    refs = {k: z[k] for k in z.files if k.startswith("ref_")}
    return meta, arrays, refs

def run(meta, arrays, x, aux):
    kind, value = "i8", np.asarray(x, dtype=np.int64).reshape(-1)
    if value.size != meta["in_size"] or value.min(initial=0) < -128 or value.max(initial=0) > 127:
        raise ValueError("input must be a flat int8 vector with NOMO_IN_SIZE values")
    aux = [] if aux is None else [int(v) for v in np.asarray(aux, dtype=np.int64).reshape(-1)]
    for st in meta["stages"]:
        k = st["kind"]
        if k == "dense":
            W, b = arrays[st["W"]].astype(np.int64), arrays[st["b"]].astype(np.int64)
            acc = W @ value + b
            value = np.array([sat8(rsr(int(v) * st["m0"], 31 + st["shift"])) for v in acc], dtype=np.int64)
            if st["relu"]: value = np.maximum(value, 0)
            kind = "i8"
        elif k == "encoder":
            acc = np.zeros(st["n"], dtype=np.int64); raster = []
            for _ in range(st["T"]):
                acc += value
                pos, neg = acc >= st["theta"], acc <= -st["theta"]
                acc = acc - st["theta"] * pos + st["theta"] * neg
                raster.append(pos.astype(np.int64) - neg.astype(np.int64))
            value, kind = np.stack(raster), "spk"
        elif k == "lif":
            W, b = arrays[st["W"]].astype(np.int64), arrays[st["b"]].astype(np.int64)
            v = np.zeros(W.shape[0], dtype=np.int64); raster = []
            lo, hi = -(1 << (st["v_bits"] - 1)), (1 << (st["v_bits"] - 1)) - 1
            for t in range(st["T"]):
                if st["leak_shift"] > 0: v = v - np.array([asr(q, st["leak_shift"]) for q in v], dtype=np.int64)
                inc = W @ value[t].astype(np.int64) + b
                v = np.clip(v + inc, lo, hi)
                z = v >= st["theta"]
                v = v - st["theta"] * z
                raster.append(z.astype(np.int64))
            value, kind = np.stack(raster), "spk"
        elif k == "decoder":
            count = value.sum(axis=0)
            if st["out"] == "i8":
                value, kind = np.array([sat8(rsr(int(v) * st["m0"], 31 + st["shift"])) for v in count], dtype=np.int64), "i8"
            else:
                value, kind = np.array([sat32(int(v) * st["k_q16"]) for v in count], dtype=np.int64), "q16"
        elif k == "to_q16":
            value, kind = np.array([sat32(int(v) * st["k_q16"]) for v in value], dtype=np.int64), "q16"
        elif k == "from_q16":
            value, kind = np.array([sat8(rsr(int(v) * st["m0"], 31 + st["shift"])) for v in value], dtype=np.int64), "i8"
        elif k == "sym_linear":
            P = arrays[st["Phi"]].astype(np.int64); out = []
            for row in P:
                out.append(sat32(rsr(sum(int(a) * int(b) for a, b in zip(row, value)), 16)))
            value, kind = np.array(out, dtype=np.int64), "q16"
        elif k == "guard":
            value = [int(v) for v in value]
            for term in st["terms"]:
                if term["kind"] == "box":
                    for c, lo, hi in zip(term["channels"], term["lo"], term["hi"]): value[c] = sat(value[c], lo, hi)
                elif term["kind"] == "thrust":
                    cap = q16mul(term["t_max"], aux[term["aux_density_ratio"]]); value[term["channel"]] = sat(value[term["channel"]], 0, cap)
                elif term["kind"] == "rotational_rate":
                    w = [aux[i] for i in term["aux_omega"]]; I = term["inertia"]
                    g = [q16mul(q16mul(I[2]-I[1], w[1]), w[2]), q16mul(q16mul(I[0]-I[2], w[2]), w[0]), q16mul(q16mul(I[1]-I[0], w[0]), w[1])]
                    for ax, c in enumerate(term["channels"]):
                        lo = sat32(q16mul(q16mul(I[ax], -term["w_max"]-w[ax]), term["inv_dt"]) + g[ax])
                        hi = sat32(q16mul(q16mul(I[ax], term["w_max"]-w[ax]), term["inv_dt"]) + g[ax])
                        if lo > term["tau_max"]: value[c] = term["tau_max"]
                        elif hi < -term["tau_max"]: value[c] = -term["tau_max"]
                        else: value[c] = sat(value[c], max(lo, -term["tau_max"]), min(hi, term["tau_max"]))
            kind = "q16"
        else: raise ValueError("unknown integer stage " + k)
    if kind != "q16": raise ValueError("integer graph did not terminate in Q16.16")
    return np.asarray(value, dtype=np.int32)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--input", help=".npy batch of flat int8 vectors")
    ap.add_argument("--aux", help=".npy batch of Q16.16 auxiliary vectors")
    ap.add_argument("--output")
    a = ap.parse_args(argv)
    meta, arrays, refs = load()
    if a.check or not a.input:
        x, aux, y = refs["ref_x"], refs.get("ref_aux"), refs["ref_y"]
        got = np.stack([run(meta, arrays, row, None if aux is None else aux[i]) for i, row in enumerate(x)])
        ok = np.array_equal(got, y)
        print("[integer] reference check: {} -> {}".format("bit-exact" if ok else "mismatch", "PASS" if ok else "FAIL"))
        if not a.input: return 0 if ok else 1
    x = np.load(a.input)
    aux = np.load(a.aux) if a.aux else None
    y = np.stack([run(meta, arrays, row, None if aux is None else aux[i]) for i, row in enumerate(x)])
    if a.output: np.save(a.output, y); print("saved {} predictions to {}".format(y.shape, a.output))
    else: print(y)
    return 0

if __name__ == "__main__": sys.exit(main())
'''


def generate(qg: QGraph, ref_x: np.ndarray, ref_aux: np.ndarray | None, ref_y: np.ndarray) -> Tuple[str, bytes]:
    meta, arrays = _meta(qg)
    arrays["meta_json"] = np.array(json.dumps(meta))
    arrays["ref_x"] = np.asarray(ref_x, dtype=np.int8)
    arrays["ref_y"] = np.asarray(ref_y, dtype=np.int32)
    if ref_aux is not None:
        arrays["ref_aux"] = np.asarray(ref_aux, dtype=np.int32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return SOURCE, buf.getvalue()

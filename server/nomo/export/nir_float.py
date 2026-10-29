"""Float (architecture-level) NIR export for any design, including convolutional and TTFS ones
(SPEC §7.5). The integer NIR from nir_export.export_all remains the bit-exact artefact for designs
the C11 compiler supports; this graph carries the float weights (with the searched precision
applied), spiking thresholds from calibration, and all Nomo semantics in metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .nir_export import NomoGraph, NomoNode, _json


def plan_to_nomo_graph(plan: Dict[str, Any], dt: float = 1e-3) -> NomoGraph:
    nodes: Dict[str, NomoNode] = {}
    edges: List[Tuple[str, str]] = []
    nodes["input"] = NomoNode("Input", {"shape": np.array(plan["input_shape"])}, {})
    prev = "input"
    shape = list(plan["input_shape"])
    guards = {int(G["after"]): G for G in plan.get("guards", [])}
    layers = plan["layers"]

    def add(name: str, node: NomoNode) -> None:
        nonlocal prev
        nodes[name] = node
        edges.append((prev, name))
        prev = name

    for i, L in enumerate(layers):
        base = f"{i:02d}_{L['name']}"
        snn = L["domain"] == "SNN"
        seg_start = snn and (i == 0 or layers[i - 1]["domain"] != "SNN" or (i - 1) in guards)
        seg_end = snn and (i == len(layers) - 1 or layers[i + 1]["domain"] != "SNN" or i in guards)
        if seg_start:
            add(f"{base}_enc", NomoNode("nomo.SpikeEncoder", {
                "theta": np.int32(0), "timesteps": np.int32(L["T"]), "shape": np.array(shape),
                "lam": np.float32(L["lam_in"]), "in_scale": np.float32(1.0)},
                {"nomo.coding": L["coding"], "nomo.signed": 1, "nomo.reset": "subtract"}))
        if L["op"] == "dense" and len(shape) > 1:
            add(f"{base}_flatten", NomoNode("Flatten", {"input_shape": np.array(shape)}, {}))
            shape = [int(np.prod(shape))]
        md: Dict[str, Any] = {"nomo.domain": L["domain"], "nomo.activation": L["activation"],
                              "nomo.q.bits": int(L.get("w_bits") or 32)}
        if L["domain"] == "SYM":
            md.update({"nomo.op": "SymbolicSubstitute", "nomo.q.format": "exact physics matrix"})
        if L["domain"] == "ANN" and L.get("act_bits"):
            md.update({"nomo.q.act_bits": int(L["act_bits"]), "nomo.q.act_amax": float(L["act_amax"])})
        W = np.asarray(L["W"], np.float32)
        b = np.asarray(L["b"], np.float32)
        if L["op"] == "conv2d":
            add(base, NomoNode("Conv2d", {"input_shape": np.array(shape[1:]), "weight": W, "bias": b,
                                          "stride": np.array([L["stride"]] * 2), "padding": np.array([L["padding"]] * 2)}, md))
            k, st, pd = W.shape[2], int(L["stride"]), int(L["padding"])
            shape = [W.shape[0], (shape[1] + 2 * pd - k) // st + 1, (shape[2] + 2 * pd - k) // st + 1]
        else:
            add(base, NomoNode("Affine", {"weight": W, "bias": b}, md))
            shape = [W.shape[0]]
        if snn:
            n = tuple(shape)
            add(f"{base}_neuron", NomoNode("IF", {"r": np.full(n, 1.0 / dt, np.float32),
                                                   "v_threshold": np.full(n, L["lam_out"], np.float32),
                                                   "v_reset": np.zeros(n, np.float32)},
                                           {"nomo.domain": "SNN", "nomo.coding": L["coding"], "nomo.timesteps": int(L["T"]),
                                            "nomo.reset": "subtract" if L["coding"] == "rate" else "single_spike_ttfs",
                                            "nomo.plastic": int(bool(L.get("plastic"))), "nomo.dt": float(dt)}))
        p = L.get("pool")
        if p:
            if p["type"] == "max":
                add(f"{base}_pool", NomoNode("nomo.MaxPool2d", {"kernel_size": np.array([p["kernel"]] * 2),
                                                                 "stride": np.array([p.get("stride", p["kernel"])] * 2)}, {}))
            else:
                k = [shape[1], shape[2]] if p["type"] == "global_avg" else [p["kernel"]] * 2
                st = k if p["type"] == "global_avg" else [p.get("stride", p["kernel"])] * 2
                add(f"{base}_pool", NomoNode("AvgPool2d", {"kernel_size": np.array(k), "stride": np.array(st)}, {}))
            shape = list(L["out_shape"])
        if seg_end:
            add(f"{base}_dec", NomoNode("nomo.SpikeDecoder", {
                "timesteps": np.int32(L["T"]), "shape": np.array(shape), "lam": np.float32(L["lam_out"]),
                "out_format": "float", "m0": np.int64(0), "shift": np.int32(0), "k_q16": np.int64(0)},
                {"nomo.coding": L["coding"]}))
        if i in guards:
            G = guards[i]
            if len(shape) > 1:
                add(f"{base}_flatten_g", NomoNode("Flatten", {"input_shape": np.array(shape)}, {}))
                shape = [int(np.prod(shape))]
            nodes[f"{base}_guard_aux"] = NomoNode("Input", {"shape": np.array([max(1, G["n_aux"])])},
                                                  {"nomo.role": "symbolic_aux"})
            add(f"{base}_guard", NomoNode("nomo.SymbolicConstraint", {
                "shape": np.array(shape), "n_aux": np.int32(G["n_aux"]),
                "constraint_json": _json({"id": G["id"], "terms": G["terms"]})}, {"nomo.domain": "SYM"}))
            edges.append((f"{base}_guard_aux", f"{base}_guard"))
    nodes["output"] = NomoNode("Output", {"shape": np.array(shape)}, {})
    edges.append((prev, "output"))
    return NomoGraph(nodes, edges, {"nomo.version": "0.4.0", "nomo.graph": plan["name"], "nomo.genome": plan["genome"],
                                    "nomo.dt": float(dt), "nomo.precision": "float weights, searched precision applied"})


def has_max_pool(plan: Dict[str, Any]) -> bool:
    return any((L.get("pool") or {}).get("type") == "max" for L in plan["layers"])

"""Build the float execution plan of a design (SPEC §6.4): the single description that every
exporter (PyTorch script, ONNX, CoreML, float NIR, PDF brief) is generated from and checked against.

Precision choices are applied as fake quantisation (weights per tensor, symmetric; ANN activations
per edge), so the exported float model reflects the bit widths the search selected. Spiking
thresholds come from data-based normalisation on calibration inputs (99.9th percentile of each
layer's continuous activation), exactly as the integer compiler does.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir import ModelGraph
from ..search.genome import Coding, Domain, Genome
from . import hybrid_runtime as hr

Weights = Dict[str, Tuple[np.ndarray, np.ndarray]]


def fake_quant_weights(W: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 16:
        return W.astype(np.float64)
    if bits == 1:
        return np.where(W >= 0, 1.0, -1.0) * float(np.abs(W).mean())
    q = 2 ** (bits - 1) - 1
    s = max(float(np.abs(W).max()), 1e-12) / q
    return np.clip(np.round(W / s), -q, q) * s


def calibration_inputs(model: ModelGraph, n: int = 16, seed: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Representative inputs. Built-in models use their own generators; uploads get N(0, 1)
    (reported as an assumption, since real calibration data is not available to the server)."""
    if model.name == "attitude_policy":
        from ..models.zoo import attitude_calibration
        return attitude_calibration(n, seed)
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n,) + tuple(model.input_shape))
    if model.name == "perception_cnn":
        X = np.abs(X)                                   # event-camera counts are non-negative
    n_aux = max([model.constraints[s.constraint_id].n_aux for s in model.guard_sites] or [0])
    return X, (np.abs(rng.normal(0, 1, size=(n, n_aux))) if n_aux else None)


def _layer_dict(model: ModelGraph, i: int, W: np.ndarray, b: np.ndarray, domain: str) -> Dict[str, Any]:
    spec = model.layers[i]
    a = spec.attrs
    return {"name": spec.name, "op": spec.op, "domain": domain,
            "activation": "linear" if domain == "SYM" else spec.activation,
            "W": W, "b": b, "stride": a.get("stride", 1), "padding": a.get("padding", 0),
            "pool": a.get("pool"), "flatten_input": bool(a.get("flatten_input", False)),
            "architecture_family": a.get("operator_family"),
            "preserve_spatial": bool(a.get("preserve_spatial", spec.op == "conv2d")),
            "act_bits": 0, "act_amax": 0.0, "coding": None, "T": 0, "lam_in": 0.0, "lam_out": 0.0,
            "w_bits": 32 if domain == "SYM" else 0}


def build_plan(model: ModelGraph, weights: Weights, genome: Genome, calib_X: np.ndarray,
               calib_aux: Optional[np.ndarray] = None, pct: float = 99.9) -> Dict[str, Any]:
    ops = hr.NumpyOps()
    # 1. continuous full-precision pass (substitutes applied) for activation statistics
    base: List[Dict[str, Any]] = []
    for i, (spec, gene) in enumerate(zip(model.layers, genome.layers)):
        if gene.domain == Domain.SYM:
            Phi = model.substitutes[spec.symbolic_substitute].phi()
            base.append(_layer_dict(model, i, Phi, np.zeros(Phi.shape[0]), "SYM"))
        else:
            W, b = weights[spec.name]
            base.append(_layer_dict(model, i, np.asarray(W, float), np.asarray(b, float), "ANN"))
    edge_in, edge_out = [], []
    x = ops.asarray(calib_X)
    for L in base:
        edge_in.append(x)
        y = hr._op(ops, L, x)
        y = ops.maximum(y, 0.0) if L["activation"] == "relu" else y
        edge_out.append(y)                               # pre-pool activation: what a spike rate encodes
        x = hr._pool(ops, L, y)

    def amax(a: np.ndarray) -> float:
        return float(max(np.percentile(np.abs(a), pct), 1e-8))

    # 2. apply the genome
    layers = []
    for i, (spec, gene, L) in enumerate(zip(model.layers, genome.layers, base)):
        L = dict(L)
        if gene.domain == Domain.ANN:
            L["W"] = fake_quant_weights(L["W"], gene.w_bits)
            L["w_bits"] = gene.w_bits
            post = hr._pool(ops, L, edge_out[i]) if L.get("pool") else edge_out[i]
            L["act_bits"], L["act_amax"] = gene.a_bits, amax(post)
        elif gene.domain == Domain.SNN:
            L["domain"] = "SNN"
            L["W"] = fake_quant_weights(L["W"], gene.w_bits)
            L["w_bits"] = gene.w_bits
            L["coding"] = "ttfs" if gene.coding == Coding.TTFS else "rate"
            L["T"] = gene.timesteps
            L["lam_in"], L["lam_out"] = amax(edge_in[i]), amax(edge_out[i])
            L["plastic"] = bool(gene.plastic)
        L["in_shape"] = list(np.shape(edge_in[i])[1:])
        L["out_shape"] = list(np.shape(hr._pool(ops, L, edge_out[i]) if L.get("pool") else edge_out[i])[1:])
        layers.append(L)
    del edge_in, edge_out, base                  # calibration activations are no longer needed
    for L in layers:                             # weights are already rounded to <= 16 bits: float32 is lossless
        L["W"] = np.asarray(L["W"], np.float32)
        L["b"] = np.asarray(L["b"], np.float32)
    # chain lambdas inside a spiking segment: each layer's input quantum is the previous threshold
    for i in range(1, len(layers)):
        if layers[i]["domain"] == "SNN" and layers[i - 1]["domain"] == "SNN" and model.guard_at(i - 1) is None:
            layers[i]["lam_in"] = layers[i - 1]["lam_out"]
    guards = []
    for site in model.guard_sites:
        con = model.constraints[site.constraint_id]
        terms = []
        for t in con.terms:
            d = dict(t.__dict__)
            terms.append({**d, "channels": list(d["channels"]) if "channels" in d else d.get("channels")})
        guards.append({"after": site.after_layer, "id": con.id, "terms": terms, "n_aux": con.n_aux})
    return {"format": "nomo.plan/1", "name": model.name, "input_shape": list(model.input_shape),
            "n_aux": max([g["n_aux"] for g in guards] or [0]), "genome": genome.key,
            "layers": layers, "guards": guards}


def run_plan(plan: Dict[str, Any], X: np.ndarray, aux: Optional[np.ndarray] = None) -> np.ndarray:
    return np.asarray(hr.forward(hr.NumpyOps(), plan, X, aux))

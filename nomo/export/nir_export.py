"""NIR export (SPEC §7).

Two HDF5 serialisations of the same graph, plus a lowering manifest:

strict    `<name>.nir`       Only standard NIR primitives. Loads with stock `nir.read` (verified
                             in tests). Nomo semantics are lowered and annotated in node metadata:
                               nomo.SpikeEncoder     -> Scale(1/lambda) -> IF(r=1/dt, thr=1)
                               nomo.SpikeDecoder     -> I(r=1/dt) -> Scale(lambda/T)
                               nomo.SymbolicConstraint -> graph cut: Output(<id>.pre) + Input(<id>.post)
                                                         (+ Input(<id>.aux)), constraint JSON in metadata;
                                                         the runtime binds the pair to the symbolic engine
                               integer LIF (leak k)  -> Affine -> LIF(tau=dt 2^k, r=2^k, v_leak=0)
                                                        (k = 0 -> IF(r=1/dt)); reset-by-subtraction flagged
                               ANN layer             -> Affine, activation + int quantisation in metadata
                             NIR has no ReLU / clamp / ternary-spike primitives; consumers that ignore
                             metadata see the linear skeleton only. This is a NIR limitation, recorded in
                             graph metadata as nomo.strict_lossy = 1.

extended  `<name>.nomo.nir`  Same HDF5 layout, custom node types `nomo.SpikeEncoder`, `nomo.SpikeDecoder`,
                             `nomo.SymbolicConstraint` kept intact. Stock `nir.read` rejects unknown types
                             by design; use `read_extended`.

manifest  `<name>.qgraph.json`  The complete integer program (weights, thresholds, m0/shift) with static
                             shapes: the hand-off format for MLIR / TOSA lowering (SPEC §7.4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from ..runtime.qgraph import (QDecoder, QDense, QEncoder, QFromQ16, QGraph, QGuard, QLIF, QSymLinear, QToQ16)

NOMO_NIR_EXT_VERSION = "nomo-ext-1"


@dataclass
class NomoNode:
    type: str
    params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NomoGraph:
    nodes: Dict[str, NomoNode]
    edges: List[Tuple[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _json(o: Any) -> str:
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, tuple):
            return list(x)
        raise TypeError(type(x))
    return json.dumps(o, default=conv, sort_keys=True)


# ---------------------------------------------------------------------------
# QGraph -> extended NomoGraph
# ---------------------------------------------------------------------------

def build_graph(qg: QGraph, dt: float = 1e-3, genome_wire: Optional[dict] = None) -> NomoGraph:
    nodes: Dict[str, NomoNode] = {}
    edges: List[Tuple[str, str]] = []
    nodes["input"] = NomoNode("Input", {"shape": np.array([qg.in_size])}, {"nomo.q.scale": float(qg.in_scale)})
    prev = "input"

    def add(name: str, node: NomoNode) -> None:
        nonlocal prev
        nodes[name] = node
        edges.append((prev, name))
        prev = name

    for k, st in enumerate(qg.stages):
        base = f"{k:02d}_{st.name.replace(':', '_').replace('.', '_')}"
        if isinstance(st, QDense):
            add(base, NomoNode("Affine", {"weight": _f32(st.W * st.w_scale), "bias": _f32(st.b * st.in_scale * st.w_scale)}, {
                "nomo.domain": "ANN", "nomo.activation": "relu" if st.relu else "linear",
                "nomo.q.bits": st.w_bits, "nomo.q.w_int": st.W.astype(np.int8), "nomo.q.b_int": st.b.astype(np.int32),
                "nomo.q.w_scale": float(st.w_scale), "nomo.q.in_scale": float(st.in_scale),
                "nomo.q.out_scale": float(st.out_scale), "nomo.q.m0": int(st.m0), "nomo.q.shift": int(st.shift)}))
        elif isinstance(st, QEncoder):
            add(base, NomoNode("nomo.SpikeEncoder", {
                "theta": np.int32(st.theta), "timesteps": np.int32(st.T), "shape": np.array([st.n]),
                "lam": np.float32(st.lam), "in_scale": np.float32(st.in_scale)},
                {"nomo.coding": "rate", "nomo.signed": 1, "nomo.reset": "subtract"}))
        elif isinstance(st, QLIF):
            no = st.W.shape[0]
            add(base + "_syn", NomoNode("Affine", {"weight": _f32(st.W * st.v_scale), "bias": _f32(st.b * st.v_scale)}, {
                "nomo.domain": "SNN", "nomo.q.bits": st.w_bits, "nomo.q.w_int": st.W.astype(np.int8),
                "nomo.q.b_int": st.b.astype(np.int32), "nomo.q.v_scale": float(st.v_scale),
                "nomo.note": "bias is injected every timestep"}))
            meta = {"nomo.domain": "SNN", "nomo.reset": "subtract", "nomo.dt": float(dt), "nomo.timesteps": st.T,
                    "nomo.q.theta_int": int(st.theta), "nomo.q.v_bits": int(st.v_bits),
                    "nomo.q.leak_shift": int(st.leak_shift), "nomo.q.v_scale": float(st.v_scale),
                    "nomo.lambda_out": float(st.lam_out), "nomo.plastic": int(st.plastic)}
            thr = np.full(no, st.theta * st.v_scale, np.float32)
            if st.leak_shift > 0:
                k_ = float(1 << st.leak_shift)
                add(base, NomoNode("LIF", {"tau": np.full(no, dt * k_, np.float32), "r": np.full(no, k_, np.float32),
                                           "v_leak": np.zeros(no, np.float32), "v_threshold": thr,
                                           "v_reset": np.zeros(no, np.float32)}, meta))
            else:
                add(base, NomoNode("IF", {"r": np.full(no, 1.0 / dt, np.float32), "v_threshold": thr,
                                          "v_reset": np.zeros(no, np.float32)}, meta))
        elif isinstance(st, QDecoder):
            add(base, NomoNode("nomo.SpikeDecoder", {
                "timesteps": np.int32(st.T), "shape": np.array([st.n]), "lam": np.float32(st.lam),
                "out_format": st.out, "m0": np.int64(st.m0), "shift": np.int32(st.shift), "k_q16": np.int64(st.k_q16)},
                {"nomo.coding": "rate"}))
        elif isinstance(st, (QToQ16, QFromQ16)):
            continue          # representation change only; real-valued graph is unaffected (kept in manifest)
        elif isinstance(st, QSymLinear):
            no, ni = st.Phi.shape
            add(base, NomoNode("Affine", {"weight": _f32(st.Phi / 65536.0), "bias": np.zeros(no, np.float32)}, {
                "nomo.domain": "SYM", "nomo.op": "SymbolicSubstitute", "nomo.substitute": st.substitute_id,
                "nomo.q.phi_q16": st.Phi.astype(np.int32), "nomo.q.format": "Q16.16"}))
        elif isinstance(st, QGuard):
            nodes[f"{base}_aux"] = NomoNode("Input", {"shape": np.array([max(1, st.n_aux)])},
                                            {"nomo.role": "symbolic_aux", "nomo.q.format": "Q16.16"})
            add(base, NomoNode("nomo.SymbolicConstraint", {
                "shape": np.array([st.n]), "n_aux": np.int32(st.n_aux),
                "constraint_json": _json(st.constraint.to_meta())}, {"nomo.domain": "SYM", "nomo.q.format": "Q16.16"}))
            edges.append((f"{base}_aux", base))
        else:  # pragma: no cover
            raise TypeError(st)
    nodes["output"] = NomoNode("Output", {"shape": np.array([qg.out_size])}, {"nomo.q.format": "Q16.16"})
    edges.append((prev, "output"))
    meta = {"nomo.version": "0.2.0", "nomo.ext": NOMO_NIR_EXT_VERSION, "nomo.graph": qg.name,
            "nomo.genome": str(qg.meta.get("genome", "")), "nomo.dt": float(dt)}
    if genome_wire is not None:
        meta["nomo.genome_json"] = _json(genome_wire)
    return NomoGraph(nodes, edges, meta)


# ---------------------------------------------------------------------------
# strict lowering -> nir.NIRGraph
# ---------------------------------------------------------------------------

def to_nir_strict(g: NomoGraph):
    import nir

    nodes: Dict[str, Any] = {}
    rename: Dict[str, Tuple[str, str]] = {}          # nomo node -> (entry, exit) nir node names
    extra_edges: List[Tuple[str, str]] = []
    dt = float(g.metadata.get("nomo.dt", 1e-3))

    for name, n in g.nodes.items():
        md = dict(n.metadata)
        p = n.params
        if n.type == "Input":
            nodes[name] = nir.Input(input_type={"input": np.asarray(p["shape"])}, metadata=md)
            rename[name] = (name, name)
        elif n.type == "Output":
            nodes[name] = nir.Output(output_type={"output": np.asarray(p["shape"])}, metadata=md)
            rename[name] = (name, name)
        elif n.type == "Affine":
            nodes[name] = nir.Affine(weight=p["weight"], bias=p["bias"], metadata=md)
            rename[name] = (name, name)
        elif n.type == "LIF":
            nodes[name] = nir.LIF(tau=p["tau"], r=p["r"], v_leak=p["v_leak"], v_threshold=p["v_threshold"],
                                  v_reset=p["v_reset"], metadata=md)
            rename[name] = (name, name)
        elif n.type == "IF":
            nodes[name] = nir.IF(r=p["r"], v_threshold=p["v_threshold"], v_reset=p["v_reset"], metadata=md)
            rename[name] = (name, name)
        elif n.type == "nomo.SpikeEncoder":
            N = int(np.asarray(p["shape"])[0])
            md.update({"nomo.op": "SpikeEncoder", "nomo.timesteps": int(p["timesteps"]),
                       "nomo.q.theta_int": int(p["theta"]), "nomo.q.in_scale": float(p["in_scale"])})
            nodes[f"{name}_scale"] = nir.Scale(scale=np.full(N, 1.0 / float(p["lam"]), np.float32), metadata=md)
            nodes[f"{name}_if"] = nir.IF(r=np.full(N, 1.0 / dt, np.float32), v_threshold=np.ones(N, np.float32),
                                         v_reset=np.zeros(N, np.float32), metadata=md)
            extra_edges.append((f"{name}_scale", f"{name}_if"))
            rename[name] = (f"{name}_scale", f"{name}_if")
        elif n.type == "nomo.SpikeDecoder":
            N = int(np.asarray(p["shape"])[0])
            md.update({"nomo.op": "SpikeDecoder", "nomo.timesteps": int(p["timesteps"]), "nomo.out_format": str(p["out_format"]),
                       "nomo.q.m0": int(p["m0"]), "nomo.q.shift": int(p["shift"]), "nomo.q.k_q16": int(p["k_q16"])})
            nodes[f"{name}_int"] = nir.I(r=np.full(N, 1.0 / dt, np.float32), metadata=md)
            nodes[f"{name}_scale"] = nir.Scale(scale=np.full(N, float(p["lam"]) / int(p["timesteps"]), np.float32), metadata=md)
            extra_edges.append((f"{name}_int", f"{name}_scale"))
            rename[name] = (f"{name}_int", f"{name}_scale")
        elif n.type == "nomo.SymbolicConstraint":
            N = int(np.asarray(p["shape"])[0])
            md.update({"nomo.op": "SymbolicConstraint", "nomo.constraint_json": str(p["constraint_json"]),
                       "nomo.n_aux": int(p["n_aux"])})
            nodes[f"{name}_pre"] = nir.Output(output_type={"output": np.array([N])}, metadata={**md, "nomo.pair": f"{name}_post"})
            nodes[f"{name}_post"] = nir.Input(input_type={"input": np.array([N])}, metadata={**md, "nomo.pair": f"{name}_pre"})
            rename[name] = (f"{name}_pre", f"{name}_post")
        else:
            raise TypeError(f"no strict lowering for {n.type}")

    edges: List[Tuple[str, str]] = []
    for a, b in g.edges:
        if g.nodes[a].metadata.get("nomo.role") == "symbolic_aux":
            continue                       # aux is bound by the runtime at the cut, not a dataflow edge
        edges.append((rename[a][1], rename[b][0]))
    edges += extra_edges
    # aux inputs stay declared as graph inputs so the port set is complete
    meta = {**g.metadata, "nomo.strict_lossy": 1}
    return nir.NIRGraph(nodes=nodes, edges=edges, metadata=meta)


# ---------------------------------------------------------------------------
# extended HDF5 (same layout as nir.write, custom types preserved)
# ---------------------------------------------------------------------------

def _write_value(group: h5py.Group, k: str, v: Any) -> None:
    if isinstance(v, str):
        group.create_dataset(k, data=v, dtype=h5py.string_dtype())
    elif isinstance(v, dict):
        sub = group.create_group(k)
        for kk, vv in v.items():
            _write_value(sub, kk, vv)
    elif isinstance(v, np.ndarray) and v.ndim > 0:
        group.create_dataset(k, data=v, dtype=v.dtype, compression="gzip")
    else:
        group.create_dataset(k, data=v)


def write_extended(path: str | Path, g: NomoGraph) -> None:
    import nir
    with h5py.File(path, "w") as f:
        f.create_dataset("version", data=nir.version, dtype=h5py.string_dtype())
        f.create_dataset("nomo_ext", data=NOMO_NIR_EXT_VERSION, dtype=h5py.string_dtype())
        root = f.create_group("node")
        root.create_dataset("type", data="NIRGraph", dtype=h5py.string_dtype())
        nodes = root.create_group("nodes")
        for name, n in g.nodes.items():
            grp = nodes.create_group(name)
            grp.create_dataset("type", data=n.type, dtype=h5py.string_dtype())
            for k, v in n.params.items():
                _write_value(grp, k, v)
            if n.metadata:
                _write_value(grp, "metadata", n.metadata)
        root.create_dataset("edges", data=np.array(g.edges, dtype=object), dtype=h5py.string_dtype())
        if g.metadata:
            _write_value(root, "metadata", g.metadata)


def _read_value(item) -> Any:
    if isinstance(item, h5py.Group):
        return {k: _read_value(v) for k, v in item.items()}
    v = item[()]
    if isinstance(v, bytes):
        return v.decode("utf8")
    return v


def read_extended(path: str | Path) -> NomoGraph:
    with h5py.File(path, "r") as f:
        root = f["node"]
        nodes = {}
        for name, grp in root["nodes"].items():
            d = {k: _read_value(v) for k, v in grp.items()}
            t = d.pop("type")
            md = d.pop("metadata", {})
            nodes[name] = NomoNode(t, d, md)
        edges = [(a.decode() if isinstance(a, bytes) else a, b.decode() if isinstance(b, bytes) else b)
                 for a, b in root["edges"][()]]
        meta = _read_value(root["metadata"]) if "metadata" in root else {}
    return NomoGraph(nodes, edges, meta)


# ---------------------------------------------------------------------------
# lowering manifest
# ---------------------------------------------------------------------------

def qgraph_manifest(qg: QGraph) -> Dict[str, Any]:
    st_out = []
    for st in qg.stages:
        d: Dict[str, Any] = {"kind": st.kind, "name": st.name}
        for k, v in st.__dict__.items():
            if k in ("name", "kind"):
                continue
            if k == "constraint":
                d[k] = v.to_meta()
            elif isinstance(v, np.ndarray):
                d[k] = {"dtype": str(v.dtype), "shape": list(v.shape), "data": v.ravel().tolist()}
            elif isinstance(v, (np.integer, np.floating)):
                d[k] = v.item()
            else:
                d[k] = v
        st_out.append(d)
    return {"format": "nomo.qgraph/1", "name": qg.name, "in_size": qg.in_size, "in_scale": qg.in_scale,
            "out_size": qg.out_size, "out_format": "Q16.16", "n_aux": qg.n_aux, "stages": st_out,
            "semantics": "see SPEC §6 and nomo/runtime/qgraph.py docstring"}


def export_all(qg: QGraph, out_dir: str | Path, dt: float = 1e-3, genome_wire: Optional[dict] = None) -> Dict[str, Path]:
    import nir
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = build_graph(qg, dt, genome_wire)
    paths = {"strict": out / f"{qg.name}.nir", "extended": out / f"{qg.name}.nomo.nir",
             "manifest": out / f"{qg.name}.qgraph.json"}
    nir.write(paths["strict"], to_nir_strict(g))
    write_extended(paths["extended"], g)
    paths["manifest"].write_text(_json(qgraph_manifest(qg)))
    return paths

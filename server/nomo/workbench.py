"""Serializable state for the six-level Nomo Workbench canvas."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .architecture import classify, layer_contract
from .modes import mode_constraints


LEVELS = (
    (1, "system_topology", "System topology", "model task DAG"),
    (2, "partitioning", "Partitioning", "ANN / SNN / symbolic boundaries"),
    (3, "hardware_graph", "Hardware graph", "PE arrays, memory, and links"),
    (4, "cycle_emulation", "Cycle-accurate emulation", "waveform and register state"),
    (5, "rtl", "RTL / Verilog", "structural datapath modules"),
    (6, "silicon_floorplan", "Silicon floorplan", "power and thermal density"),
)


def schema() -> Dict[str, Any]:
    return {"format": "nomo.workbench/1", "levels": [
        {"level": n, "id": ident, "title": title, "description": desc,
         "state": {"nodes": "array", "edges": "array", "selection": "string|null", "revision": "integer"}}
        for n, ident, title, desc in LEVELS
    ], "state_rules": [
        "selection is a layer or design key and is stable across levels",
        "each edit increments revision and preserves provenance",
        "target weights are normalized before selecting a Pareto candidate",
    ]}


def _red_green(value: float) -> str:
    v = max(0.0, min(1.0, float(value)))
    r = int(34 + 185 * v)
    g = int(153 - 108 * v)
    b = int(84 - 66 * v)
    return f"#{r:02x}{g:02x}{b:02x}"


def _norm(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(float(v) - lo) / (hi - lo) for v in values]


def select_candidate(evaluations: Iterable[Any], weights: Sequence[float] = (1.0, 1.0, 1.0)) -> Optional[Any]:
    items = list(evaluations)
    if not items:
        return None
    w = [max(float(x), 1e-9) for x in list(weights)[:3]]
    while len(w) < 3:
        w.append(1.0)
    e = _norm([float(x.cost.energy_j) for x in items])
    l = _norm([float(x.cost.latency_s) for x in items])
    # Accuracy is maximised, so its normalized loss is inverted here.
    a = _norm([float(x.accuracy) for x in items])
    scored = zip(items, e, l, a)
    return min(scored, key=lambda item: w[0] * item[1] + w[1] * item[2] - w[2] * item[3])[0]


def build_state(model: Any, ev: Optional[Any] = None, *, plan: Optional[Dict[str, Any]] = None,
                ptq_report: Optional[Dict[str, Any]] = None, hitl: Optional[Dict[str, Any]] = None,
                target_weights: Sequence[float] = (1.0, 1.0, 1.0), mode: Optional[str] = None,
                hardware: Optional[Any] = None) -> Dict[str, Any]:
    layers = list(getattr(model, "layers", []))
    costs = []
    memory = []
    sensitivity = []
    for layer in layers:
        stage = next((s for s in getattr(ev.cost, "stages", []) if s.name == layer.name), None) if ev else None
        costs.append(float(getattr(stage, "energy_j", 0.0)))
        memory.append(float(getattr(stage, "mem_bytes", 0.0)))
        sensitivity.append(float(getattr(getattr(layer, "sensitivity", None), "q_w", 0.0)))
    cn, mn, sn = _norm(costs), _norm(memory), _norm(sensitivity)
    layer_nodes = []
    for i, layer in enumerate(layers):
        gene = ev.genome.layers[i] if ev else None
        domain = getattr(getattr(gene, "domain", None), "name", "ANN")
        layer_nodes.append({
            "id": layer.name, "label": layer.name, "op": layer.op, "index": i,
            "domain": domain, "params": layer.params, "macs": layer.macs,
            "contract": layer_contract(layer), "cost_score": cn[i] if cn else 0.0,
            "memory_score": mn[i] if mn else 0.0, "quantization_sensitivity": sn[i] if sn else 0.0,
            "color": _red_green(max(cn[i] if cn else 0.0, mn[i] if mn else 0.0, sn[i] if sn else 0.0)),
            "precision_lock": None,
        })
    edges = [{"source": layers[i].name, "target": layers[i + 1].name} for i in range(max(0, len(layers) - 1))]
    domains = {n["id"]: n["domain"] for n in layer_nodes}
    topology = {"nodes": layer_nodes, "edges": edges, "selection": ev.key if ev else None}
    partition = {"nodes": [{"id": k, "layer": k, "domain": v, "boundary": i > 0 and v != layer_nodes[i - 1]["domain"]}
                             for i, (k, v) in enumerate(domains.items())], "edges": edges}
    n_cores = int(getattr(hardware, "n_cores", 0) or 0)
    hardware = {"nodes": [{"id": "pe-array", "kind": "processing_elements", "count": n_cores},
                           {"id": "sram", "kind": "memory", "capacity_bytes": getattr(getattr(ev, "cost", None), "memory_bytes", 0)},
                           {"id": "interconnect", "kind": "bus"}],
                "edges": [{"source": "pe-array", "target": "sram"}, {"source": "sram", "target": "interconnect"}],
                "mapping": [{"layer": n["id"], "pe": i % max(n_cores, 1)} for i, n in enumerate(layer_nodes)]}
    emulation = {"nodes": [{"id": "cycle-0", "cycle": 0, "event": "frame_start"}], "edges": [],
                 "metrics": {"cycle_count": None, "cache_hit_ratio": None, "bus_contention": None},
                 "source": "not measured"}
    if ev:
        emulation["metrics"].update({"estimated_latency_s": float(ev.cost.latency_s), "estimated_energy_j": float(ev.cost.energy_j)})
    rtl = {"nodes": [{"id": "custom_npu_core", "kind": "systemverilog", "path": "silicon_eda/rtl/custom_npu_core.sv"},
                      {"id": "pe_array", "kind": "systemverilog", "path": "silicon_eda/rtl/pe_array.sv"}],
           "edges": [{"source": "custom_npu_core", "target": "pe_array"}],
           "active_datapath": [n["id"] for n in layer_nodes if n["domain"] in ("ANN", "SNN")]}
    floorplan = {"nodes": [{"id": n["id"], "x": (i % 4) * 0.24, "y": (i // 4) * 0.22,
                             "w": 0.2, "h": 0.18, "power_density": n["cost_score"],
                             "thermal_color": _red_green(n["cost_score"])} for i, n in enumerate(layer_nodes)],
                 "edges": [], "thermal_model": "proxy until HITL telemetry is attached"}
    metrics = {"energy_j": float(ev.cost.energy_j), "latency_s": float(ev.cost.latency_s),
               "accuracy_pct": float(ev.accuracy), "accuracy_source": ev.accuracy_source,
               "area_um2": None, "ppa_source": "proxy"} if ev else {}
    return {"format": "nomo.workbench/1", "revision": 1, "model": getattr(model, "name", "model"),
            "architecture": classify(model).to_dict(), "mode": mode_constraints(mode),
            "target_weights": list(target_weights), "cost_function": {
                "formula": "L_unified = alpha*L_task + beta*Latency + gamma*Energy + delta*Area_silicon",
                "coefficients": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0}},
            "metrics": metrics, "ptq": ptq_report, "hitl": hitl,
            "levels": {ident: {"level": n, "title": title, "description": desc,
                               "nodes": (topology if n == 1 else partition if n == 2 else hardware if n == 3 else
                                          emulation if n == 4 else rtl if n == 5 else floorplan)["nodes"],
                               "edges": (topology if n == 1 else partition if n == 2 else hardware if n == 3 else
                                         emulation if n == 4 else rtl if n == 5 else floorplan)["edges"],
                               "state": (topology if n == 1 else partition if n == 2 else hardware if n == 3 else
                                         emulation if n == 4 else rtl if n == 5 else floorplan)}
                      for n, ident, title, desc in LEVELS}}

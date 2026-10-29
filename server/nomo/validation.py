"""Evidence reports for export packages: proxies, calibration measurements, and spike efficiency."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _ttfs_efficiency(plan: Dict[str, Any], inputs: np.ndarray) -> Dict[str, Any]:
    """Count deterministic TTFS event work against a dense T-step baseline.

    For a dense layer the baseline performs ``MACs*T`` operations.  TTFS emits
    at most one event per active input value, so the event work is the exact
    non-zero-input count multiplied by each row's non-zero weight fan-out.
    """
    rows = []
    x = np.asarray(inputs, dtype=np.float64)
    for i, layer in enumerate(plan.get("layers", [])):
        if layer.get("domain") != "SNN":
            continue
        if layer.get("coding") != "ttfs":
            continue
        W = np.asarray(layer.get("W"), dtype=np.float64)
        T = max(1, int(layer.get("T", 1)))
        if W.ndim != 2:
            rows.append({"layer": layer.get("name"), "available": False, "reason": "non-matrix operator"})
            continue
        active = int(np.count_nonzero(np.abs(x) > 0.0))
        fanout = np.count_nonzero(W, axis=0)
        flat = np.asarray(x).reshape(x.shape[0], -1)
        if flat.shape[1] != len(fanout):
            active_ops = active * float(np.count_nonzero(W) / max(1, W.shape[1]))
        else:
            active_ops = float(np.sum((np.abs(flat) > 0.0) @ fanout))
        dense_ops = float(W.shape[0] * W.shape[1] * T * max(1, x.shape[0]))
        saved = max(0.0, dense_ops - active_ops)
        rows.append({"layer": layer.get("name"), "available": True, "timesteps": T,
                     "calibration_samples": int(x.shape[0]), "spike_events": active,
                     "dense_fp32_ops": dense_ops, "event_ops": active_ops,
                     "spike_sparsity_pct": 100.0 * max(0.0, 1.0 - active_ops / max(dense_ops, 1.0)),
                     "ops_saved": saved, "definition": "dense MACs*T minus non-zero TTFS synaptic events"})
        # Preserve the layer shape without inventing a spatial flattening step.
        x = np.zeros((x.shape[0], W.shape[0]), dtype=np.float64)
    if not rows:
        return {"available": False, "reason": "no TTFS layer in the selected design", "layers": []}
    available = [r for r in rows if r.get("available")]
    return {"available": bool(available), "layers": rows,
            "aggregate_ops_saved": float(sum(r.get("ops_saved", 0.0) for r in available)),
            "aggregate_spike_sparsity_pct": float(np.average(
                [r["spike_sparsity_pct"] for r in available],
                weights=[max(r["dense_fp32_ops"], 1.0) for r in available])) if available else None}


def build_sensitivity_report(ctx: Any, ev: Any) -> Dict[str, Any]:
    from .architecture import classify
    from .export.plan import run_plan
    from .runtime.quantize import float_forward

    X, A = ctx.calib()
    plan = ctx.plan(ev)
    proxy = {"status": "estimated", "accuracy_pct": float(ev.accuracy),
             "accuracy_source": ev.accuracy_source, "terms": dict(ev.accuracy_terms),
             "note": "search sensitivity coefficients are zero-shot proxies unless an oracle result is attached"}
    empirical: Dict[str, Any] = {"status": "not_available", "source": "none", "sample_count": int(X.shape[0])}
    try:
        # Compare the generated plan with the unquantised reference on the same
        # calibration tensors. This is a measured execution-fidelity check, not
        # a claim about task accuracy.
        ref = float_forward(ctx.model, ctx.weights, ev.genome, X, A)[-1]
        out = run_plan(plan, X, A)
        diff = np.asarray(out, float) - np.asarray(ref, float)
        empirical = {"status": "measured", "source": "calibration_data", "sample_count": int(X.shape[0]),
                     "runtime_rmse": float(np.sqrt(np.mean(diff * diff))),
                     "runtime_max_abs_error": float(np.max(np.abs(diff))),
                     "task_accuracy_pct": float(ev.accuracy) if ev.accuracy_source == "oracle" else None,
                     "task_accuracy_status": "oracle" if ev.accuracy_source == "oracle" else "not measured"}
    except Exception as exc:  # backend limitations should be visible in the report
        empirical["reason"] = f"integer/reference fidelity check unavailable: {type(exc).__name__}: {exc}"
    return {
        "format": "nomo.validation/1",
        "model": ctx.model.name,
        "architecture": classify(ctx.model).to_dict(),
        "zero_shot_sensitivity_proxies": proxy,
        "empirical_validation_results": empirical,
        "weights_provenance": {"source": ctx.weights_source,
                               "synthetic_or_demo": str(ctx.weights_source).lower().startswith("synthetic")},
        "calibration": getattr(ctx, "calibration_report", None) or {
            "source": "generated representative tensors", "sample_count": int(X.shape[0]), "is_uploaded": False},
        "neuromorphic_efficiency": _ttfs_efficiency(plan, X),
        "unified_cost_function": {
            "formula": "L_unified = alpha*L_task + beta*Latency + gamma*Energy + delta*Area_silicon",
            "coefficients": (ctx.settings.get("unified_loss") if isinstance(ctx.settings, dict) else None)
            or {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0},
            "metrics": {"latency_s": float(ev.cost.latency_s), "energy_j": float(ev.cost.energy_j),
                        "area_source": "proxy"},
        },
        "hitl": getattr(ctx, "hitl_measurement", None),
    }

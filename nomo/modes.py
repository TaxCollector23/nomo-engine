"""Operational modes and their explicit deployment guardrails.

The optimizer remains mode-agnostic at its core.  A mode is a small, serialisable
contract applied at the API/CLI boundary so a run records the requirements that
motivated its budgets and export package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OperationalMode:
    id: str
    title: str
    summary: str
    requirements: tuple[str, ...]
    defaults: Dict[str, Any]
    export_tags: tuple[str, ...]


MODES: Dict[str, OperationalMode] = {
    "low_power_neuromorphic": OperationalMode(
        "low_power_neuromorphic",
        "Sub-mW ultra-low-power neuromorphic",
        "Event-driven sparse execution for always-on sensors and wearables.",
        ("clock_gating", "event_driven", "power_budget"),
        {"budgets": {"energy_j": 1e-3}, "search": {"allow_spiking": True}},
        ("power_gating", "dvs_input", "sparsity_telemetry"),
    ),
    "hard_realtime": OperationalMode(
        "hard_realtime",
        "Hard real-time physical AI",
        "Deterministic execution for robotics, automotive, and industrial control.",
        ("wcet_bound", "deterministic_memory", "fail_safe"),
        {"budgets": {"period_s": 1e-3}, "search": {"crossing_penalty": 0.25}},
        ("wcet", "safety_guard", "hitl_required"),
    ),
    "radiation_hardened": OperationalMode(
        "radiation_hardened",
        "Extreme environment / radiation hardened",
        "Fault-tolerant execution for aerospace and high-altitude systems.",
        ("tmr_logic", "seu_ecc", "thermal_throttle"),
        {},
        ("tmr", "ecc", "thermal_profile"),
    ),
    "on_chip_learning": OperationalMode(
        "on_chip_learning",
        "Agentic on-chip continuous learning",
        "Bounded local adaptation with plasticity and replay protection.",
        ("local_updates", "plasticity", "replay_buffer"),
        {"budgets": {"min_plastic_params": 1}, "search": {"allow_spiking": True}},
        ("plasticity", "replay_guard", "online_calibration"),
    ),
}

_ALIASES = {
    "mode_i": "low_power_neuromorphic",
    "mode_ii": "hard_realtime",
    "mode_iii": "radiation_hardened",
    "mode_iv": "on_chip_learning",
    "low_power": "low_power_neuromorphic",
    "real_time": "hard_realtime",
    "radiation": "radiation_hardened",
    "continual_learning": "on_chip_learning",
}


def get_mode(mode: Optional[str]) -> Optional[OperationalMode]:
    if not mode:
        return None
    key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in MODES:
        raise ValueError(f"unknown operational mode '{mode}'; choose from {', '.join(MODES)}")
    return MODES[key]


def mode_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        k: {"id": v.id, "title": v.title, "summary": v.summary,
            "requirements": list(v.requirements), "defaults": v.defaults, "export_tags": list(v.export_tags)}
        for k, v in MODES.items()
    }


def validate_mode(mode: Optional[str], *, period_s: Optional[float] = None,
                  min_plastic_params: int = 0, allow_spiking: bool = True) -> List[str]:
    """Return user-facing warnings/errors without silently changing a run.

    Callers can turn returned strings into HTTP 422s when they are hard
    requirements.  Keeping this function pure makes the same contract usable by
    the CLI, API, and package manifest.
    """
    m = get_mode(mode)
    if m is None:
        return []
    warnings: List[str] = []
    if m.id == "low_power_neuromorphic" and not allow_spiking:
        warnings.append("low-power neuromorphic mode is selected without spiking execution; event-driven savings may be unavailable")
    if m.id == "hard_realtime" and period_s is None:
        warnings.append("hard real-time mode has no explicit period budget; the package will not claim a WCET bound")
    if m.id == "on_chip_learning" and min_plastic_params <= 0:
        warnings.append("on-chip learning mode has no minimum plastic-parameter budget; continual learning is not enforced")
    return warnings


def mode_constraints(mode: Optional[str]) -> Dict[str, Any]:
    """Return machine-readable constraints used in run/package manifests."""
    m = get_mode(mode)
    if m is None:
        return {"id": None, "requirements": [], "export_tags": []}
    return {"id": m.id, "title": m.title, "requirements": list(m.requirements),
            "defaults": m.defaults, "export_tags": list(m.export_tags)}

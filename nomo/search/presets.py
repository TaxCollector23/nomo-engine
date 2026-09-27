"""One-click optimisation presets (SPEC §10.3). Each preset is plain run settings, so the UI can
show exactly what it changed and the user can edit anything afterwards.

Presets steer the search through budgets, toggles and the recommendation weights. They never
invent hardware numbers: a preset that would need different silicon (e.g. a lower clock) says so in
`notes` instead of silently changing the chip model.
"""
from __future__ import annotations

from typing import Any, Dict

PRESETS: Dict[str, Dict[str, Any]] = {
    "battery_saver": {
        "label": "Battery Saver",
        "summary": "Lowest energy per decision. Spiking layers wherever they pay off; accepts a larger accuracy loss.",
        "settings": {
            "budgets": {"accuracy_drop_max": 5.0},
            "search": {"allow_continuous": True, "allow_spiking": True, "allow_symbolic": True,
                       "codings": ["rate", "ttfs"], "crossing_penalty": 0.0, "crossing_min_saving_pct": 0.0,
                       "asf_weights": [4.0, 1.0, 1.0]},
        },
        "notes": "Lowering the clock is not applied: with fixed static power a slower clock raises energy per "
                 "inference in this model. Set a custom clock under Hardware if your chip has DVFS data.",
    },
    "ultra_low_latency": {
        "label": "Ultra-Low Latency",
        "summary": "Fastest response. Keeps layers continuous (no spike timesteps to wait for) and discourages domain switches.",
        "settings": {
            "budgets": {"accuracy_drop_max": 2.0},
            "search": {"allow_continuous": True, "allow_spiking": False, "allow_symbolic": True,
                       "codings": ["rate", "ttfs"], "crossing_penalty": 0.2, "crossing_min_saving_pct": 0.0,
                       "asf_weights": [1.0, 4.0, 1.0]},
        },
        "notes": "",
    },
    "balanced_edge": {
        "label": "Balanced Edge",
        "summary": "Best all-round trade-off between energy, speed and accuracy.",
        "settings": {
            "budgets": {"accuracy_drop_max": 3.0},
            "search": {"allow_continuous": True, "allow_spiking": True, "allow_symbolic": True,
                       "codings": ["rate", "ttfs"], "crossing_penalty": 0.05, "crossing_min_saving_pct": 5.0,
                       "asf_weights": [1.0, 1.0, 1.0]},
        },
        "notes": "",
    },
    "strict_safety": {
        "label": "Strict Safety",
        "summary": "Accuracy first, physics-based layers locked in wherever available, safety guards always on.",
        "settings": {
            "budgets": {"accuracy_drop_max": 1.0},
            "search": {"allow_continuous": True, "allow_spiking": True, "allow_symbolic": True,
                       "codings": ["rate"], "crossing_penalty": 0.1, "crossing_min_saving_pct": 10.0,
                       "asf_weights": [1.0, 1.0, 4.0]},
            "lock_symbolic": True,
        },
        "notes": "Safety guards are mandatory in every preset; this one also locks every layer that has a "
                 "physics formula to that formula.",
    },
}

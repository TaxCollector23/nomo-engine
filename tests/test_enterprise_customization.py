"""Focused tests for the bounded enterprise customization layer."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nomo.enterprise import (
    ProfileValidationError,
    canonical_json,
    capability_evidence_report,
    load_profile,
    merge_with_base_config,
    profile_fingerprint,
)
from nomo.enterprise.evidence import EvidenceValidationError


EXAMPLE = Path(__file__).parents[1] / "nomo" / "enterprise" / "examples" / "hard_realtime_robotics.yaml"


def test_hard_realtime_example_loads_and_has_an_explicit_contract():
    profile = load_profile(EXAMPLE)

    assert profile.id == "hard-realtime-robotics"
    assert profile.hardware["target"] == "akd1500"
    assert profile.safety["claim_policy"] == "measured_only"
    assert profile.precision["allowed_weight_bits"] == [8]
    assert {item["name"] for item in profile.required_evidence} == {
        "task_accuracy",
        "end_to_end_latency",
        "energy_per_inference",
        "memory_footprint",
    }


def test_json_profile_loading_and_clear_validation_errors(tmp_path):
    profile = load_profile(EXAMPLE)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")

    loaded = load_profile(path)
    assert loaded.canonical_json() == profile.canonical_json()

    invalid = profile.to_dict()
    invalid["objectives"][0]["direction"] = "sideways"
    with pytest.raises(ProfileValidationError) as exc:
        load_profile(invalid)
    message = str(exc.value)
    assert "objectives[0].direction" in message
    assert "minimize" in message


def test_profile_config_overrides_base_recursively_without_mutating_it():
    profile = load_profile(EXAMPLE)
    profile_data = profile.to_dict()
    profile_data["config"] = {
        "budgets": {"accuracy_drop_max": 0.5},
        "search": {"population": 96, "asf_weights": [3.0, 1.0, 0.5]},
        "export": {"formats": ["enterprise", "c11"]},
    }
    profile = load_profile(profile_data)
    base = {
        "model": "base-model",
        "budgets": {"accuracy_drop_max": 4.0, "period_s": 0.02},
        "search": {"population": 16, "seed": 3, "asf_weights": [1.0, 1.0, 1.0]},
        "export": {"archive": "zip", "formats": ["pdf"]},
    }
    original = copy.deepcopy(base)

    merged = merge_with_base_config(base, profile)

    assert base == original
    assert merged["model"] == "base-model"
    assert merged["budgets"] == {"accuracy_drop_max": 0.5, "period_s": 0.02}
    assert merged["search"] == {"population": 96, "seed": 3, "asf_weights": [3.0, 1.0, 0.5]}
    assert merged["export"] == {"archive": "zip", "formats": ["enterprise", "c11"]}
    assert merged["enterprise"]["profile"]["id"] == profile.id
    assert merged["enterprise"]["profile"]["schema"] == "nomo.enterprise/design-profile"


def test_profile_merge_reads_a_base_nomo_yaml(tmp_path):
    profile = load_profile(EXAMPLE)
    base_path = tmp_path / "nomo.yaml"
    base_path.write_text(
        "model: base-model\n"
        "budgets:\n"
        "  accuracy_drop_max: 4.0\n"
        "  period_s: 0.02\n"
        "search:\n"
        "  population: 16\n",
        encoding="utf-8",
    )

    merged = merge_with_base_config(base_path, profile)

    assert merged["model"] == "attitude_policy"
    assert merged["budgets"] == {"accuracy_drop_max": 1.0, "period_s": 0.01}
    assert merged["search"] == {"population": 64, "rounds": 20, "seed": 17}


def test_canonical_serialization_and_fingerprint_ignore_mapping_order():
    profile = load_profile(EXAMPLE)
    data = profile.to_dict()
    reordered = dict(reversed(list(data.items())))
    reordered["constraints"] = dict(reversed(list(data["constraints"].items())))
    reordered["organization"] = dict(reversed(list(data["organization"].items())))
    other = load_profile(reordered)

    assert canonical_json(profile) == canonical_json(other)
    assert profile_fingerprint(profile) == profile_fingerprint(other)


def test_capability_evidence_report_distinguishes_measured_simulated_proxy_and_missing():
    profile = load_profile(EXAMPLE)
    report = capability_evidence_report(
        profile,
        evidence={
            "task_accuracy": {"evidence_level": "measured", "source": "holdout-v3"},
            "end_to_end_latency": {"evidence_level": "simulated", "source": "timing-model"},
            "energy_per_inference": {"evidence_level": "proxy", "source": "analytic-profile"},
            "memory_footprint": {"evidence_level": "simulated", "source": "static-analysis"},
        },
    )

    assert report["claims"]["task_accuracy"]["evidence_level"] == "measured"
    assert report["claims"]["end_to_end_latency"]["status"] == "insufficient"
    assert report["claims"]["energy_per_inference"]["evidence_level"] == "proxy"
    assert report["claims"]["memory_footprint"]["satisfies_requirement"] is True
    assert report["summary"]["release_ready"] is False
    assert "end_to_end_latency" in report["summary"]["missing_or_insufficient"]
    assert report["claims"]["hardware_assumption.clock_hz"]["evidence_level"] == "proxy"


def test_capability_evidence_rejects_unknown_evidence_level():
    profile = load_profile(EXAMPLE)
    with pytest.raises(EvidenceValidationError) as exc:
        capability_evidence_report(profile, evidence={"task_accuracy": {"evidence_level": "guessed"}})
    assert "evidence.task_accuracy.evidence_level" in str(exc.value)

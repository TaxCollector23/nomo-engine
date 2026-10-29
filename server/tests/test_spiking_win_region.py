"""The spiking win-region study must run, stay honest, and find the documented tradeoffs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studies.spiking_win_region import study_profile  # noqa: E402


def test_energy_saving_exists_on_neuromorphic_profiles():
    for profile in ("akd1500", "loihi2"):
        for model in ("attitude_policy", "perception_cnn"):
            r = study_profile(model, profile)
            assert r["best_spiking_energy_j"] is not None
            assert r["best_spiking_energy_j"] < r["ann_reference"]["energy_j"], (
                f"spiking should offer an energy saving over the profile's own ANN engine for {model}@{profile}")


def test_attitude_policy_win_region_is_documented_not_assumed():
    r = study_profile("attitude_policy", "akd1500")
    nonempty = {slack: v for slack, v in r["win_regions_by_accuracy_budget"].items() if v["nonempty"]}
    assert nonempty, "attitude_policy@akd1500 should have a documented win region within 20 proxy points"
    for v in nonempty.values():
        assert v["best_energy_saving_pct"] is not None and v["best_energy_saving_pct"] > 0


def test_gpu_profile_honest_negative():
    """The dense GPU engine should NOT lose to its own SNN mapping: a negative result must be reported."""
    r = study_profile("attitude_policy", "gpu")
    assert r["win_region_nonempty"] is False or r["win_regions_by_accuracy_budget"]["0.0"]["nonempty"] is False


def test_report_labels_accuracy_as_proxy_and_energy_as_modeled():
    r = study_profile("perception_cnn", "akd1500")
    assert "not measured task accuracy" in r["criteria"]["note"]
    assert "not hardware validation data" in r["provenance"]
    assert all(row["accuracy_source"] in ("proxy", "oracle") for row in r["rows"])


def test_serializable_report(tmp_path):
    from studies.spiking_win_region import main as run_main
    out = tmp_path / "win.json"
    sys_argv = sys.argv
    try:
        sys.argv = ["spiking_win_region.py", "--out", str(out)]
        run_main()
    finally:
        sys.argv = sys_argv
    data = json.loads(out.read_text())
    assert data["schema"] == "nomo-spiking-win-region/1"
    assert "not silicon validation" in data["disclaimer"]
    assert len(data["studies"]) == 6

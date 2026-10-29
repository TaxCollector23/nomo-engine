"""The serving-sim parity goldens must regenerate byte-identically.

Guards against silent drift between the Python reference simulator and the
checked-in scripts/serving-sim-golden.json that the browser verify step
(scripts/verify-engine.mjs) replays through the TypeScript port.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # nomo-planner/
GOLDEN = REPO.parent / "scripts" / "serving-sim-golden.json"


def test_serving_sim_golden_regenerates_identically(tmp_path):
    out = tmp_path / "serving-sim-golden.json"
    result = subprocess.run(
        [sys.executable, str(REPO / "studies" / "export_serving_sim_golden.py"), str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    fresh = json.loads(out.read_text())
    assert GOLDEN.exists(), "checked-in golden is missing; run studies/export_serving_sim_golden.py"
    checked_in = json.loads(GOLDEN.read_text())
    assert fresh == checked_in, (
        "serving-sim golden drifted from the Python reference — regenerate with "
        "`cd nomo-planner && PYTHONPATH=. python3 studies/export_serving_sim_golden.py ../scripts/serving-sim-golden.json`"
    )


def test_golden_cases_cover_the_interesting_paths():
    checked_in = json.loads(GOLDEN.read_text())
    names = {case["name"] for case in checked_in["cases"]}
    assert {"poisson_fixed", "lognormal_bursty", "saturation_preemption", "prefix_cache", "tp_replicas", "replay_rows"} <= names
    by_name = {case["name"]: case for case in checked_in["cases"]}
    # the saturation case must actually exercise preemption, or the parity claim is hollow
    assert by_name["saturation_preemption"]["metrics"]["preemptions"] > 0
    # the prefix-cache case must produce at least one hit
    assert by_name["prefix_cache"]["metrics"]["prefix_cache_hits"] > 0

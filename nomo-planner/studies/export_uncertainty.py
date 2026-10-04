"""Fit and export the deterministic bootstrap artifact consumed by the Lab."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nomo_planner.uncertainty import fit_bootstrap, load_observations, sample_manifest  # noqa: E402


def main() -> None:
    rows = load_observations(ROOT / "data" / "training_observations.csv")
    samples = fit_bootstrap(rows)
    manifest = sample_manifest(
        samples,
        cluster="a100_nvlink_ib",
        seed=20260929,
        source="Narayanan et al. 2021 (arXiv:2104.04473), Tables 1-2",
    )
    output = ROOT.parent / "src" / "planner" / "uncertainty.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(samples)} bootstrap samples from {len(rows)} rows)")


if __name__ == "__main__":
    main()

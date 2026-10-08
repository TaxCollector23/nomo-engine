"""Leave-one-out predictive coverage study for A2."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nomo_planner.uncertainty import (  # noqa: E402
    Observation,
    fit_bootstrap,
    load_observations,
    measured_step,
    predictive_interval,
)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return centre - radius, centre + radius


def _predictive_interval(row: Observation, samples, seed: int):
    # Keep the study independent of the browser: predictive_interval draws a
    # held-out measurement distribution from the same stored residual contract.
    return predictive_interval(row, samples, seed=seed)


def main() -> None:
    rows = load_observations(ROOT / "data" / "training_observations.csv")
    records = []
    for index, row in enumerate(rows):
        train = rows[:index] + rows[index + 1:]
        samples = fit_bootstrap(train, replicates=128, seed=20260929 + index)
        interval = _predictive_interval(row, samples, 20260929 + 10_000 + index)
        measured = measured_step(row)
        records.append({
            "row_id": row.row_id,
            "strategy": row.strategy,
            "measured": measured,
            "median": interval["median"],
            "low": interval["low"],
            "high": interval["high"],
            "covered": interval["low"] <= measured <= interval["high"],
        })

    by_strategy = {}
    for strategy in sorted({r["strategy"] for r in records}):
        subset = [r for r in records if r["strategy"] == strategy]
        covered = sum(bool(r["covered"]) for r in subset)
        by_strategy[strategy] = {"n": len(subset), "covered": covered, "coverage": covered / len(subset),
                                 "wilson_95": _wilson(covered, len(subset))}
    covered = sum(bool(r["covered"]) for r in records)
    report = {
        "method": "leave-one-out held-out coverage; stratified bootstrap refit; central predictive 90% interval",
        "source": "Narayanan et al. 2021 (arXiv:2104.04473), Tables 1-2",
        "nominal_coverage": 0.90,
        "n": len(records),
        "covered": covered,
        "coverage": covered / len(records),
        "wilson_95": _wilson(covered, len(records)),
        "by_strategy": by_strategy,
        "rows": records,
        "caveat": "22 published rows make the interval around empirical coverage wide; this is validation evidence, not a guarantee of future coverage.",
    }
    output = ROOT / "docs" / "UNCERTAINTY_COVERAGE.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = ROOT / "docs" / "UNCERTAINTY_COVERAGE.md"
    markdown.write_text(
        "# A2 held-out uncertainty coverage\n\n"
        f"Nominal interval: 90%. Leave-one-out coverage: **{covered}/{len(records)} ({covered / len(records):.1%})**.\n\n"
        f"95% Wilson interval: {report['wilson_95'][0]:.1%}–{report['wilson_95'][1]:.1%}.\n\n"
        + "\n".join(f"- `{key}`: {value['covered']}/{value['n']} ({value['coverage']:.1%}); Wilson {value['wilson_95'][0]:.1%}–{value['wilson_95'][1]:.1%}"
                    for key, value in by_strategy.items())
        + "\n\nThis is held-out evidence for the published sample, not a guarantee of coverage on a different hardware or workload distribution.\n",
        encoding="utf-8",
    )
    print(json.dumps({"coverage": report["coverage"], "by_strategy": by_strategy}, indent=2))


if __name__ == "__main__":
    main()

"""Small benchmark designer for recommendation-sensitive measurements."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ExperimentCandidate:
    name: str
    config: Mapping[str, Any]
    expected_step_s: float
    uncertainty_s: float
    benchmark_hours: float


@dataclass(frozen=True)
class RankedExperiment:
    name: str
    config: Mapping[str, Any]
    probability_changes_recommendation: float
    value_per_benchmark_hour: float
    expected_step_s: float
    uncertainty_s: float


def rank_experiments(
    candidates: Iterable[ExperimentCandidate],
    *,
    current_step_s: float,
    alternate_step_s: float,
) -> list[RankedExperiment]:
    if current_step_s <= 0 or alternate_step_s <= 0:
        raise ValueError("current and alternate predictions must be positive")
    margin = abs(current_step_s - alternate_step_s)
    result: list[RankedExperiment] = []
    for candidate in candidates:
        if candidate.expected_step_s <= 0 or candidate.uncertainty_s < 0 or candidate.benchmark_hours <= 0:
            raise ValueError("experiment predictions, uncertainty, and duration must be valid")
        probability = min(0.99, max(0.01, candidate.uncertainty_s / (candidate.uncertainty_s + margin)))
        result.append(RankedExperiment(
            name=candidate.name, config=dict(candidate.config),
            probability_changes_recommendation=probability,
            value_per_benchmark_hour=probability / candidate.benchmark_hours,
            expected_step_s=candidate.expected_step_s, uncertainty_s=candidate.uncertainty_s,
        ))
    return sorted(result, key=lambda item: (-item.value_per_benchmark_hour, item.name))


def csv_template(ranked: Iterable[RankedExperiment]) -> str:
    rows = list(ranked)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["experiment", "config_json", "observed_step_s", "observed_tokens_per_s", "notes"])
    for row in rows:
        writer.writerow([row.name, _json(row.config), "", "", "Fill after running this benchmark; values stay local"])
    return output.getvalue()


def _json(value: Mapping[str, Any]) -> str:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

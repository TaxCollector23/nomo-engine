"""Public serving validation fixtures and comparison utilities.

The checked-in Sarathi-Serve rows are measured values copied from a cited
paper, never simulator output.  This module keeps the boundary explicit:
``compare_serving_predictions`` only reports an error after a caller supplies
predictions from a Nomo run, and ``run_sarathi_table4_replay`` labels its
request-length reconstruction as a derived replay because the paper exposes
summary statistics rather than the original request trace.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .serving_sim import DistributionSpec, ServingAssumptions, SimulationResult, simulate_serving
from .simcalibration import load_evidence


DEFAULT_SARATHI_PATH = Path(__file__).resolve().parent.parent / "data" / "serving_sarathi_table4.json"


@dataclass(frozen=True)
class PublicServingObservation:
    row_id: str
    workload: str
    scheduler: str
    metric: str
    value_s: float
    source_id: str
    source_url: str
    source_citation: str
    model: str
    hardware: str
    request_count: int
    token_budget: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "workload": self.workload,
            "scheduler": self.scheduler,
            "metric": self.metric,
            "value_s": self.value_s,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "source_citation": self.source_citation,
            "model": self.model,
            "hardware": self.hardware,
            "request_count": self.request_count,
            "token_budget": self.token_budget,
        }


@dataclass(frozen=True)
class ServingValidationReport:
    source_id: str
    source_url: str
    observation_count: int
    matched_count: int
    missing_row_ids: tuple[str, ...]
    predictions: Mapping[str, float]
    mean_absolute_error_s: float | None
    mean_absolute_percentage_error_pct: float | None
    max_absolute_percentage_error_pct: float | None
    status: str
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "observation_count": self.observation_count,
            "matched_count": self.matched_count,
            "missing_row_ids": list(self.missing_row_ids),
            "predictions": dict(self.predictions),
            "mean_absolute_error_s": self.mean_absolute_error_s,
            "mean_absolute_percentage_error_pct": self.mean_absolute_percentage_error_pct,
            "max_absolute_percentage_error_pct": self.max_absolute_percentage_error_pct,
            "status": self.status,
            "notes": list(self.notes),
        }


def load_sarathi_table4(path: str | Path = DEFAULT_SARATHI_PATH) -> tuple[PublicServingObservation, ...]:
    """Load cited measured rows with source and workload metadata intact."""

    bundle = load_evidence(path)
    observations: list[PublicServingObservation] = []
    for record in bundle.records:
        if record.kind != "serving":
            continue
        features = record.features
        workload = str(features.get("workload", ""))
        if not workload:
            raise ValueError(f"public serving row {record.row_id!r} lacks workload metadata")
        observations.append(
            PublicServingObservation(
                row_id=record.row_id,
                workload=workload,
                scheduler=record.strategy,
                metric=record.metric,
                value_s=float(record.value),
                source_id=record.provenance.source_id,
                source_url=record.provenance.url,
                source_citation=record.provenance.citation,
                model=str(features.get("model", "unknown")),
                hardware=str(features.get("hardware", "unknown")),
                request_count=int(float(features.get("request_count", 0))),
                token_budget=int(float(features.get("token_budget", 0))),
            )
        )
    if not observations:
        raise ValueError("Sarathi-Serve fixture contains no serving observations")
    return tuple(observations)


def compare_serving_predictions(
    observations: Sequence[PublicServingObservation],
    predictions: Mapping[str, float],
) -> ServingValidationReport:
    """Score supplied simulator outputs against published measured rows.

    A missing prediction is an incomplete validation, not a zero error.  The
    report therefore keeps missing IDs and returns ``status='incomplete'``.
    """

    if not observations:
        raise ValueError("at least one public serving observation is required")
    source_ids = {row.source_id for row in observations}
    source_urls = {row.source_url for row in observations}
    if len(source_ids) != 1 or len(source_urls) != 1:
        raise ValueError("comparison currently requires one coherent public source")
    matched: list[tuple[float, float]] = []
    missing: list[str] = []
    for row in observations:
        value = predictions.get(row.row_id)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            missing.append(row.row_id)
            continue
        matched.append((row.value_s, float(value)))
    errors = [abs(predicted - observed) for observed, predicted in matched]
    percentage_errors = [error / observed * 100.0 for (observed, _), error in zip(matched, errors) if observed > 0]
    return ServingValidationReport(
        source_id=next(iter(source_ids)),
        source_url=next(iter(source_urls)),
        observation_count=len(observations),
        matched_count=len(matched),
        missing_row_ids=tuple(missing),
        predictions={key: float(value) for key, value in predictions.items() if math.isfinite(float(value))},
        mean_absolute_error_s=(sum(errors) / len(errors)) if errors else None,
        mean_absolute_percentage_error_pct=(sum(percentage_errors) / len(percentage_errors)) if percentage_errors else None,
        max_absolute_percentage_error_pct=max(percentage_errors, default=None),
        status="validated" if len(matched) == len(observations) else "incomplete",
        notes=(
            "Observed values are measured rows from the cited Sarathi-Serve paper.",
            "Prediction values must come from a Nomo serving simulation; this function never fills them in.",
        ),
    )


_SARATHI_WORKLOAD_STATS: Mapping[str, Mapping[str, float]] = {
    "openchat_sharegpt4": {
        "prompt_median": 1730.0,
        "prompt_p90": 5696.0,
        "answer_median": 415.0,
        "answer_p90": 834.0,
    },
    "arxiv_summarization": {
        "prompt_median": 7059.0,
        "prompt_p90": 12985.0,
        "answer_median": 208.0,
        "answer_p90": 371.0,
    },
}


def _lognormal_from_median_p90(median: float, p90: float) -> DistributionSpec:
    sigma = math.log(max(p90, median) / median) / 1.2815515655446004
    return DistributionSpec("lognormal", mean=math.log(median), stddev=max(0.0, sigma))


def run_sarathi_table4_replay(
    observations: Sequence[PublicServingObservation] | None = None,
    *,
    seed: int = 20261005,
) -> ServingValidationReport:
    """Run Nomo against both workloads represented by Sarathi Table 4.

    Sarathi publishes workload summary statistics rather than its raw trace.
    The replay derives log-normal request-length distributions from those
    published median/P90 values and labels the result accordingly.  It is a
    reproducible validation comparison, not a claim that the original trace
    was recovered.
    """

    rows = tuple(observations or load_sarathi_table4())
    predictions: dict[str, float] = {}
    for workload in sorted({row.workload for row in rows}):
        stats = _SARATHI_WORKLOAD_STATS.get(workload)
        if stats is None:
            raise ValueError(f"no published workload summary is registered for {workload!r}")
        for scheduler in sorted({row.scheduler for row in rows if row.workload == workload}):
            chunked = 1024 if "chunked" in scheduler.lower() or "combined" in scheduler.lower() else 0
            assumptions = ServingAssumptions(
                seed=seed + sum(ord(char) for char in f"{workload}:{scheduler}"),
                arrival_process="poisson",
                request_count=128,
                arrival_rate_per_s=1.0,
                prompt_distribution=_lognormal_from_median_p90(stats["prompt_median"], stats["prompt_p90"]),
                answer_distribution=_lognormal_from_median_p90(stats["answer_median"], stats["answer_p90"]),
                replicas=1,
                tensor_parallel=2,
                max_batch_size=128,
                chunked_prefill_tokens=chunked,
                prefill_tokens_per_s=15_000.0,
                decode_tokens_per_s=220.0,
                prefill_batch_gain=0.20,
                decode_batch_gain=0.15,
                slo_ttft_s=None,
                slo_itl_s=None,
                slo_e2e_s=None,
                metadata={
                    "source": "Sarathi-Serve Table 2 summary statistics",
                    "source_url": "https://arxiv.org/abs/2403.02310",
                    "derived_replay": True,
                    "model": "Yi-34B",
                    "hardware": "2x NVIDIA A100",
                    "scheduler": scheduler,
                },
            )
            result: SimulationResult = simulate_serving(assumptions=assumptions)
            metrics = result.metrics
            for row in rows:
                if row.workload != workload or row.scheduler != scheduler:
                    continue
                if row.metric == "ttft_p50":
                    prediction = metrics.get("ttft_p50_s")
                elif row.metric == "tbt_p99":
                    prediction = metrics.get("inter_token_latency_p99_s")
                else:
                    raise ValueError(f"unsupported Sarathi metric: {row.metric!r}")
                if not isinstance(prediction, (int, float)) or not math.isfinite(float(prediction)):
                    raise ValueError(f"simulation did not produce finite {row.metric} for {row.row_id}")
                predictions[row.row_id] = float(prediction)
    report = compare_serving_predictions(rows, predictions)
    return ServingValidationReport(
        **{
            **report.__dict__,
            "notes": report.notes
            + (
                "Nomo prediction replay uses published median/P90 lengths and explicit modeled rates; raw Sarathi request traces were not available.",
            ),
        }
    )


__all__ = [
    "DEFAULT_SARATHI_PATH",
    "PublicServingObservation",
    "ServingValidationReport",
    "compare_serving_predictions",
    "load_sarathi_table4",
    "run_sarathi_table4_replay",
]

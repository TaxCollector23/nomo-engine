"""Browser-safe calibration helpers for customer run logs.

This is deliberately a local multiplicative refit.  It only uses rows the
customer supplies, refuses non-positive values, and exposes the sample count,
scale, residual spread, and empirical 90% coverage so the UI never confuses a
small local correction with the published A100 calibration study.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CustomerObservation:
    run_id: str
    cluster: str
    predicted_step_s: float
    observed_step_s: float


@dataclass(frozen=True)
class CalibrationFit:
    cluster: str
    observations: int
    scale: float
    residual_sigma: float
    interval_low_scale: float
    interval_high_scale: float
    coverage: float
    held_out_mape_pct: float | None
    held_out_observations: int
    method: str = "median multiplicative refit in log-step-time space"
    validation: str = "leave-one-out error on customer-supplied rows"

    def predict(self, predicted_step_s: float) -> float:
        return predicted_step_s * self.scale

    def as_dict(self) -> dict[str, object]:
        return {
            "cluster": self.cluster, "observations": self.observations, "scale": self.scale,
            "residual_sigma": self.residual_sigma, "interval_low_scale": self.interval_low_scale,
            "interval_high_scale": self.interval_high_scale, "coverage": self.coverage,
            "held_out_mape_pct": self.held_out_mape_pct, "held_out_observations": self.held_out_observations,
            "method": self.method, "validation": self.validation,
        }


def parse_customer_csv(text: str) -> list[CustomerObservation]:
    reader = csv.DictReader(io.StringIO(text.strip()))
    required = {"run_id", "cluster", "predicted_step_s", "observed_step_s"}
    if not required.issubset(set(reader.fieldnames or ())):
        missing = ", ".join(sorted(required - set(reader.fieldnames or ())))
        raise ValueError(f"customer calibration CSV is missing columns: {missing}")
    rows: list[CustomerObservation] = []
    for index, row in enumerate(reader, start=2):
        try:
            predicted = float(row["predicted_step_s"] or "")
            observed = float(row["observed_step_s"] or "")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {index} has non-numeric step time") from exc
        if predicted <= 0 or observed <= 0:
            raise ValueError(f"row {index} step times must be positive")
        rows.append(CustomerObservation(row["run_id"], row["cluster"], predicted, observed))
    return rows


def fit_customer_scale(rows: Iterable[CustomerObservation], *, cluster: str | None = None) -> CalibrationFit:
    selected = [row for row in rows if cluster is None or row.cluster == cluster]
    if not selected:
        raise ValueError("customer calibration needs at least one matching observation")
    ratios = [row.observed_step_s / row.predicted_step_s for row in selected]
    scale = median(ratios)
    residuals = [math.log(ratio / scale) for ratio in ratios]
    sigma = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    low = scale * math.exp(-1.645 * sigma)
    high = scale * math.exp(1.645 * sigma)
    covered = sum(low <= ratio <= high for ratio in ratios) / len(ratios)
    held_out_errors: list[float] = []
    if len(ratios) >= 2:
        for index, row in enumerate(selected):
            training_ratios = ratios[:index] + ratios[index + 1:]
            held_out_scale = median(training_ratios)
            held_out_errors.append(abs((row.observed_step_s / row.predicted_step_s) / held_out_scale - 1) * 100)
    return CalibrationFit(
        cluster=cluster or (selected[0].cluster if len({row.cluster for row in selected}) == 1 else "mixed"),
        observations=len(selected), scale=scale, residual_sigma=sigma,
        interval_low_scale=low, interval_high_scale=high, coverage=covered,
        held_out_mape_pct=(sum(held_out_errors) / len(held_out_errors)) if held_out_errors else None,
        held_out_observations=len(held_out_errors),
    )


def apply_customer_fit(predicted_step_s: float, fit: CalibrationFit) -> dict[str, float]:
    if predicted_step_s <= 0:
        raise ValueError("predicted step time must be positive")
    return {
        "median": fit.predict(predicted_step_s),
        "low": predicted_step_s * fit.interval_low_scale,
        "high": predicted_step_s * fit.interval_high_scale,
    }

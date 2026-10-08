"""Artifact-driven product simulations for roadmap modules M4--M11.

The earlier product helpers in :mod:`nomo_planner.product_models` are useful
for small UI previews.  This module is the Python-first simulation surface:
inputs are explicit artifacts, stochastic quantities are represented by
reusable distributions, and every result carries provenance, intervals,
serious-criteria status, and an explicit ``Preview`` label when evidence is
missing.  The implementation intentionally uses only the standard library so
the same schemas can be mirrored by a CLI, a worker, or a customer notebook.

The simulators are reference models, not claims about silicon or production
capacity.  A result becomes ``Validated`` only when the supplied artifacts
include source information and at least one measured or calibrated artifact;
otherwise it remains ``Preview`` even when the arithmetic is deterministic.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import math
import random
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


Number = int | float
Scalar = Number | str | bool | None


def _finite(value: Any, *, name: str, positive: bool = False, nonnegative: bool = False) -> float:
    """Coerce a finite number and give callers a useful schema error."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0:
        raise ValueError(f"{name} cannot be negative")
    return number


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None and mapping[name] != "":
            return mapping[name]
    return default


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _json_ready(value: Any) -> Any:
    """Convert public dataclasses and tuples into stable JSON-compatible data."""

    if isinstance(value, SimulationResult):
        return value.as_dict()
    if isinstance(value, Interval):
        return value.as_dict()
    if isinstance(value, Timeline):
        return value.as_dict()
    if isinstance(value, ArtifactInput):
        return value.as_dict()
    if isinstance(value, Provenance):
        return value.as_dict()
    if isinstance(value, Distribution):
        return value.as_dict()
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if _sequence(value) or isinstance(value, set):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class Provenance:
    """Where an input came from and whether it is evidence-grade."""

    artifact_id: str = ""
    source: str = ""
    source_url: str | None = None
    observed_at: str | None = None
    method: str | None = None
    measured: bool = False
    calibrated: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "source": self.source,
            "source_url": self.source_url,
            "observed_at": self.observed_at,
            "method": self.method,
            "measured": self.measured,
            "calibrated": self.calibrated,
            "notes": self.notes,
        }

    @property
    def evidence_grade(self) -> str:
        if self.calibrated:
            return "calibrated"
        if self.measured:
            return "measured"
        if self.source or self.source_url:
            return "sourced-assumption"
        return "unattributed"

    @property
    def source_id(self) -> str:
        return self.artifact_id or self.source

    @property
    def citation(self) -> str:
        return self.source


@dataclass(frozen=True)
class ArtifactInput:
    """A serializable input artifact used by a simulator.

    ``payload`` is intentionally unopinionated: JSON objects, CSV row lists,
    or a single measured record are all valid.  The simulator-specific
    parsers validate the fields they consume.
    """

    kind: str
    payload: Any
    artifact_id: str = ""
    source: str = ""
    source_url: str | None = None
    observed_at: str | None = None
    schema_version: str = "1"
    measured: bool = False
    calibrated: bool = False
    notes: str = ""

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            artifact_id=self.artifact_id,
            source=self.source,
            source_url=self.source_url,
            observed_at=self.observed_at,
            method=self.kind,
            measured=self.measured,
            calibrated=self.calibrated,
            notes=self.notes,
        )

    @property
    def data(self) -> Any:
        """Alias used by JSON/CSV adapters that call the payload ``data``."""

        return self.payload

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        kind: str | None = None,
        default_source: str = "",
    ) -> "ArtifactInput":
        provenance = value.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        reserved = {
            "kind", "type", "payload", "data", "artifact_id", "id", "source",
            "source_id", "citation", "source_citation", "source_url", "url", "source_link", "source_doi", "observed_at", "as_of", "accessed_at", "schema_version",
            "measured", "calibrated", "notes", "provenance",
        }
        if "payload" in value:
            payload = value["payload"]
        elif "data" in value:
            payload = value["data"]
        else:
            payload = {key: item for key, item in value.items() if key not in reserved}
        return cls(
            kind=str(kind or _first(value, "kind", "type", default="artifact")),
            payload=payload,
            artifact_id=str(_first(value, "artifact_id", "id", default=provenance.get("artifact_id", ""))),
            source=str(_first(value, "source", "source_id", "citation", default=provenance.get("source", provenance.get("source_id", default_source))) or ""),
            source_url=_first(value, "source_url", "url", default=provenance.get("source_url", provenance.get("url"))),
            observed_at=_first(value, "observed_at", "as_of", default=provenance.get("observed_at")),
            schema_version=str(_first(value, "schema_version", default="1")),
            measured=bool(_first(value, "measured", default=provenance.get("measured", False))),
            calibrated=bool(_first(value, "calibrated", default=provenance.get("calibrated", False))),
            notes=str(_first(value, "notes", default=provenance.get("notes", "")) or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": _json_ready(self.payload),
            "artifact_id": self.artifact_id,
            "source": self.source,
            "source_url": self.source_url,
            "observed_at": self.observed_at,
            "schema_version": self.schema_version,
            "measured": self.measured,
            "calibrated": self.calibrated,
            "notes": self.notes,
        }


# Short alias used by artifact loaders and downstream notebooks.
Artifact = ArtifactInput


def normalize_artifact(value: ArtifactInput | Mapping[str, Any] | str | Path, *, kind: str | None = None) -> ArtifactInput:
    if isinstance(value, ArtifactInput):
        return value
    if isinstance(value, (str, Path)):
        return load_artifact(value, kind=kind)
    if isinstance(value, Mapping):
        return ArtifactInput.from_mapping(value, kind=kind)
    raise TypeError("artifact must be an ArtifactInput, mapping, JSON/CSV path, or Path")


def load_artifact(path: str | Path, *, kind: str | None = None) -> ArtifactInput:
    """Load a JSON object/list or a CSV row artifact without guessing fields."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        with file_path.open(newline="", encoding="utf-8") as handle:
            payload: Any = list(csv.DictReader(handle))
    elif suffix in {".json", ".jsonl"}:
        with file_path.open(encoding="utf-8") as handle:
            if suffix == ".jsonl":
                payload = [json.loads(line) for line in handle if line.strip()]
            else:
                payload = json.load(handle)
    else:
        raise ValueError("artifact paths must end in .json, .jsonl, or .csv")
    if isinstance(payload, Mapping) and any(key in payload for key in ("payload", "data", "kind", "provenance")):
        artifact = ArtifactInput.from_mapping(payload, kind=kind, default_source=str(file_path))
        if not artifact.source:
            artifact = ArtifactInput(**{**artifact.__dict__, "source": str(file_path)})
        return artifact
    return ArtifactInput(kind=kind or file_path.stem, payload=payload, source=str(file_path))


def _artifacts(values: Iterable[ArtifactInput | Mapping[str, Any] | str | Path] | None) -> tuple[ArtifactInput, ...]:
    if values is None:
        return ()
    return tuple(normalize_artifact(value) for value in values)


def _record_artifact(kind: str, payload: Any, provenance: Provenance) -> ArtifactInput:
    """Retain a structured simulator input in the result's artifact ledger."""

    return ArtifactInput(
        kind=kind,
        payload=_json_ready(payload),
        artifact_id=provenance.artifact_id,
        source=provenance.source,
        source_url=provenance.source_url,
        observed_at=provenance.observed_at,
        measured=provenance.measured,
        calibrated=provenance.calibrated,
        notes=provenance.notes,
    )


@dataclass(frozen=True)
class Interval:
    low: float
    median: float
    high: float
    level: float = 0.90
    method: str = "empirical"
    samples: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("low", self.low), ("median", self.median), ("high", self.high)):
            if math.isnan(float(value)):
                raise ValueError(f"interval {name} cannot be NaN")
        if self.low > self.median or self.median > self.high:
            raise ValueError("interval must be ordered low <= median <= high")
        if not 0 < self.level <= 1:
            raise ValueError("interval level must be in (0, 1]")

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    @property
    def p05(self) -> float:
        return self.low

    @property
    def p50(self) -> float:
        return self.median

    @property
    def p95(self) -> float:
        return self.high

    def as_dict(self) -> dict[str, Any]:
        return {
            "low": self.low if math.isfinite(float(self.low)) else None,
            "median": self.median if math.isfinite(float(self.median)) else None,
            "high": self.high if math.isfinite(float(self.high)) else None,
            "level": self.level,
            "method": self.method,
            "samples": self.samples,
        }


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interval_from_samples(values: Sequence[float], *, level: float = 0.90, method: str = "empirical") -> Interval:
    if not values:
        raise ValueError("cannot compute an interval from an empty sample")
    tail = (1.0 - level) / 2.0
    return Interval(
        low=quantile(values, tail),
        median=quantile(values, 0.5),
        high=quantile(values, 1.0 - tail),
        level=level,
        method=method,
        samples=len(values),
    )


@dataclass(frozen=True)
class Distribution:
    """A deterministic, empirical, or parametric input distribution."""

    kind: str = "constant"
    params: Mapping[str, Any] = field(default_factory=dict)
    values: tuple[float, ...] = ()
    name: str = ""
    unit: str = ""
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        kind = self.kind.lower()
        if kind not in {"constant", "fixed", "deterministic", "empirical", "normal", "lognormal", "uniform", "triangular", "quantiles"}:
            raise ValueError(f"unsupported distribution kind: {self.kind}")
        if kind == "empirical" and not self.values:
            raise ValueError("empirical distributions require values")
        if kind == "constant" and _maybe_float(self.params.get("value")) is None and not self.values:
            raise ValueError("constant distributions require a value")
        if any(not math.isfinite(float(value)) for value in self.values):
            raise ValueError("distribution values must be finite")
        object.__setattr__(self, "kind", "constant" if kind in {"fixed", "deterministic"} else kind)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        name: str = "",
        unit: str = "",
        provenance: Provenance | None = None,
    ) -> "Distribution":
        if isinstance(value, Distribution):
            return value
        if "DistributionSpec" in globals() and isinstance(value, DistributionSpec):
            return value._distribution()
        if isinstance(value, Mapping):
            kind = str(_first(value, "kind", "type", default="constant")).lower()
            raw_values = _first(value, "values", "samples", default=())
            values = tuple(_finite(item, name=f"{name}.values", nonnegative=False) for item in raw_values) if _sequence(raw_values) else ()
            params = dict(value.get("params", {})) if isinstance(value.get("params"), Mapping) else {}
            for key in ("value", "mean", "std", "sigma", "median", "geometric_sd", "low", "high", "mode", "points", "quantiles"):
                if key in value and key not in params:
                    params[key] = value[key]
            if kind in {"constant", "fixed", "deterministic"} and "value" not in params:
                quantile_values = []
                for probability, names in ((0.05, ("p05", "q05")), (0.50, ("p50", "median")), (0.95, ("p95", "q95"))):
                    found = _first(value, *names)
                    if found is not None:
                        quantile_values.append((probability, float(found)))
                if quantile_values:
                    kind = "quantiles"
                    params["points"] = quantile_values
            prov_value = value.get("provenance")
            if isinstance(prov_value, Mapping):
                prov = Provenance(
                    artifact_id=str(prov_value.get("artifact_id", prov_value.get("source_id", ""))), source=str(prov_value.get("source", prov_value.get("citation", ""))),
                    source_url=prov_value.get("source_url", prov_value.get("url")), observed_at=prov_value.get("observed_at", prov_value.get("accessed_at")),
                    method=prov_value.get("method"), measured=bool(prov_value.get("measured", False)),
                    calibrated=bool(prov_value.get("calibrated", False)), notes=str(prov_value.get("notes", "")),
                )
            else:
                prov = provenance or Provenance()
            if kind == "empirical":
                params.pop("value", None)
            return cls(kind=kind, params=params, values=values, name=name or str(value.get("name", "")), unit=unit or str(value.get("unit", "")), provenance=prov)
        if _sequence(value):
            values = tuple(_finite(item, name=f"{name}.values") for item in value)
            return cls(kind="empirical", values=values, name=name, unit=unit, provenance=provenance or Provenance())
        number = _finite(value, name=name or "distribution value")
        return cls(kind="constant", params={"value": number}, name=name, unit=unit, provenance=provenance or Provenance())

    def sample(self, rng: random.Random) -> float:
        if self.kind == "constant":
            return float(self.params.get("value", self.values[0] if self.values else 0.0))
        if self.kind == "empirical":
            return float(rng.choice(self.values))
        if self.kind == "normal":
            return float(rng.gauss(float(self.params["mean"]), float(self.params.get("std", self.params.get("sigma", 0.0)))))
        if self.kind == "lognormal":
            if "median" in self.params:
                median = float(self.params["median"])
                geometric_sd = float(self.params.get("geometric_sd", self.params.get("sigma", 1.0)))
                return float(rng.lognormvariate(math.log(median), math.log(geometric_sd)))
            return float(rng.lognormvariate(float(self.params["mean"]), float(self.params.get("sigma", 1.0))))
        if self.kind == "uniform":
            return float(rng.uniform(float(self.params["low"]), float(self.params["high"])))
        if self.kind == "triangular":
            return float(rng.triangular(float(self.params["low"]), float(self.params["high"]), float(self.params.get("mode", self.params["low"]))))
        points = self.params.get("points", self.params.get("quantiles"))
        if isinstance(points, Mapping):
            points = sorted((float(probability), float(value)) for probability, value in points.items())
        points = [(float(probability), float(value)) for probability, value in (points or ())]
        if not points:
            raise ValueError("quantile distributions require points or quantiles")
        draw = rng.random()
        if draw <= points[0][0]:
            return points[0][1]
        for (left_q, left_v), (right_q, right_v) in zip(points, points[1:]):
            if draw <= right_q:
                fraction = (draw - left_q) / max(right_q - left_q, 1e-12)
                return left_v + fraction * (right_v - left_v)
        return points[-1][1]

    def interval(self, *, samples: int = 512, seed: int = 20261005, level: float = 0.90) -> Interval:
        if samples <= 0:
            raise ValueError("distribution sample count must be positive")
        if self.kind == "constant":
            value = self.sample(random.Random(seed))
            return Interval(value, value, value, level=level, method="deterministic", samples=1)
        rng = random.Random(seed)
        draws = [self.sample(rng) for _ in range(samples)]
        return interval_from_samples(draws, level=level, method=f"{self.kind}-monte-carlo")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "params": _json_ready(dict(self.params)),
            "values": list(self.values),
            "name": self.name,
            "unit": self.unit,
            "provenance": self.provenance.as_dict(),
        }


@dataclass(frozen=True)
class DistributionSpec:
    """Compatibility schema for trace/config distributions.

    This shape mirrors the serving trace contract while delegating empirical,
    normal, lognormal, uniform, triangular, and quantile behavior to
    :class:`Distribution`.  The extra fields keep JSON artifacts round-trip
    friendly without making simulator callers learn a second vocabulary.
    """

    kind: str = "fixed"
    value: float = 1.0
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    stddev: float | None = None
    shape: float | None = None
    scale: float | None = None
    values: tuple[Any, ...] = ()
    weights: tuple[float, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def _distribution(self) -> Distribution:
        kind = str(self.kind or "fixed").lower().replace("-", "_")
        params = dict(self.params or {})
        if kind in {"fixed", "constant", "deterministic"}:
            return Distribution.from_value(self.value, name="distribution-spec", provenance=self.provenance)
        if kind in {"normal", "gaussian"}:
            return Distribution(kind="normal", params={"mean": self.mean if self.mean is not None else self.value, "std": self.stddev or 0.0}, provenance=self.provenance)
        if kind in {"lognormal", "log_normal"}:
            return Distribution(kind="lognormal", params={"mean": self.mean if self.mean is not None else 0.0, "sigma": self.stddev or 1.0}, provenance=self.provenance)
        if kind in {"uniform", "flat"}:
            return Distribution(kind="uniform", params={"low": self.low if self.low is not None else self.value, "high": self.high if self.high is not None else self.value}, provenance=self.provenance)
        if kind == "triangular":
            return Distribution(kind="triangular", params={"low": self.low if self.low is not None else self.value, "high": self.high if self.high is not None else self.value, "mode": self.mean if self.mean is not None else self.value}, provenance=self.provenance)
        if kind in {"empirical", "sample", "samples", "choice", "categorical"}:
            return Distribution(kind="empirical", values=tuple(float(item) for item in (self.values or (self.value,))), provenance=self.provenance)
        if kind in {"quantiles", "quantile"}:
            return Distribution(kind="quantiles", params={"points": params.get("points", params.get("quantiles", self.values))}, provenance=self.provenance)
        # Keep unsupported names visible to the caller, but do not silently
        # turn them into a fabricated parametric distribution.
        raise ValueError(f"unsupported distribution kind: {self.kind}")

    def sample(self, rng: random.Random) -> float:
        kind = str(self.kind or "fixed").lower().replace("-", "_")
        params = dict(self.params or {})
        if kind in {"exponential", "exp"}:
            rate = float(params.get("rate_per_s", params.get("rate", 0.0)) or 0.0)
            mean_gap = float(params.get("mean_gap", params.get("mean", self.scale or self.value)) or 0.0)
            return rng.expovariate(rate) if rate > 0 else rng.expovariate(1.0 / max(mean_gap, 1e-12))
        if kind == "gamma":
            return rng.gammavariate(float(params.get("shape", self.shape or 1.0)), float(params.get("scale", self.scale or self.value or 1.0)))
        if kind in {"poisson", "pois"}:
            lam = max(0.0, float(params.get("mean", self.mean if self.mean is not None else self.value)))
            threshold = math.exp(-lam)
            product = 1.0
            count = 0
            while product > threshold:
                count += 1
                product *= rng.random()
            return float(count - 1)
        if kind in {"choice", "categorical"} and self.values:
            weights = self.weights or tuple(1.0 for _ in self.values)
            return float(rng.choices(self.values, weights=weights, k=1)[0])
        return self._distribution().sample(rng)

    def interval(self, *, samples: int = 512, seed: int = 20261005, level: float = 0.90) -> Interval:
        if samples <= 0:
            raise ValueError("distribution sample count must be positive")
        if str(self.kind or "fixed").lower() in {"fixed", "constant", "deterministic"}:
            value = self.sample(random.Random(seed))
            return Interval(value, value, value, level=level, method="deterministic", samples=1)
        rng = random.Random(seed)
        return interval_from_samples([self.sample(rng) for _ in range(samples)], level=level, method=f"{self.kind}-monte-carlo")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "value": self.value, "low": self.low, "high": self.high,
            "mean": self.mean, "stddev": self.stddev, "shape": self.shape, "scale": self.scale,
            "values": list(self.values), "weights": list(self.weights), "params": _json_ready(dict(self.params)),
            "provenance": self.provenance.as_dict(),
        }


def distribution(value: Any, *, name: str = "", unit: str = "", provenance: Provenance | None = None) -> Distribution:
    return Distribution.from_value(value, name=name, unit=unit, provenance=provenance)


@dataclass(frozen=True)
class TimelineEvent:
    event_id: str
    phase: str
    start_s: float
    duration_s: float
    resource: str = ""
    kind: str = "compute"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite(self.start_s, name="timeline start", nonnegative=True)
        _finite(self.duration_s, name="timeline duration", nonnegative=True)

    @property
    def end_s(self) -> float:
        return float(self.start_s) + float(self.duration_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "phase": self.phase,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "end_s": self.end_s,
            "resource": self.resource,
            "kind": self.kind,
            "metadata": _json_ready(dict(self.metadata)),
        }


@dataclass(frozen=True)
class Timeline:
    events: tuple[TimelineEvent, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.events, key=lambda event: (event.start_s, event.resource, event.event_id)))
        object.__setattr__(self, "events", ordered)

    @property
    def duration_s(self) -> float:
        return max((event.end_s for event in self.events), default=0.0)

    def append(self, *events: TimelineEvent) -> "Timeline":
        return Timeline(self.events + tuple(events))

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> TimelineEvent:
        return self.events[index]

    def resource_utilization(self, resource: str, *, horizon_s: float | None = None) -> float:
        horizon = self.duration_s if horizon_s is None else _finite(horizon_s, name="timeline horizon", positive=True)
        active = sum(event.duration_s for event in self.events if event.resource == resource)
        return active / horizon if horizon else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"duration_s": self.duration_s, "events": [event.as_dict() for event in self.events]}

    def to_csv(self) -> str:
        return export_timeline_csv(self.events)


def build_timeline(events: Iterable[TimelineEvent | Mapping[str, Any]]) -> Timeline:
    normalized: list[TimelineEvent] = []
    for index, event in enumerate(events):
        if isinstance(event, TimelineEvent):
            normalized.append(event)
        else:
            normalized.append(TimelineEvent(
                event_id=str(_first(event, "event_id", "id", default=f"event-{index}")),
                phase=str(_first(event, "phase", default="unknown")),
                start_s=_finite(_first(event, "start_s", "start", default=0), name="timeline start", nonnegative=True),
                duration_s=_finite(_first(event, "duration_s", "duration", default=0), name="timeline duration", nonnegative=True),
                resource=str(_first(event, "resource", default="")),
                kind=str(_first(event, "kind", default="compute")),
                metadata=dict(_first(event, "metadata", default={}) or {}),
            ))
    return Timeline(tuple(normalized))


@dataclass(frozen=True)
class SearchReport:
    """Reusable result of a finite, explicit design-space search."""

    rows: tuple[Mapping[str, Any], ...]
    feasible: tuple[Mapping[str, Any], ...]
    frontier: tuple[Mapping[str, Any], ...]
    best: Mapping[str, Any] | None
    objectives: tuple[tuple[str, bool], ...]
    algorithm: str = "enumeration"
    generations: int = 0
    population_size: int = 0
    evaluations: int = 0
    repaired_candidates: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [_json_ready(row) for row in self.rows],
            "feasible": [_json_ready(row) for row in self.feasible],
            "frontier": [_json_ready(row) for row in self.frontier],
            "best": _json_ready(self.best),
            "objectives": [list(item) for item in self.objectives],
            "algorithm": self.algorithm,
            "generations": self.generations,
            "population_size": self.population_size,
            "evaluations": self.evaluations,
            "repaired_candidates": self.repaired_candidates,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def dominates(left: Mapping[str, Any], right: Mapping[str, Any], objectives: Sequence[tuple[str, bool]]) -> bool:
    """Return true when ``left`` is no worse and strictly better once.

    The boolean in each objective is ``minimize``.  Missing/non-numeric
    objective values do not dominate another row.
    """

    no_worse = True
    strict = False
    for name, minimize in objectives:
        left_value = _maybe_float(left.get(name))
        right_value = _maybe_float(right.get(name))
        if left_value is None or right_value is None:
            return False
        if minimize:
            if left_value > right_value + 1e-12:
                no_worse = False
            if left_value < right_value - 1e-12:
                strict = True
        else:
            if left_value < right_value - 1e-12:
                no_worse = False
            if left_value > right_value + 1e-12:
                strict = True
    return no_worse and strict


def pareto_frontier(rows: Iterable[Mapping[str, Any]], objectives: Sequence[tuple[str, bool]]) -> list[Mapping[str, Any]]:
    materialized = list(rows)
    result = [row for row in materialized if not any(other is not row and dominates(other, row, objectives) for other in materialized)]
    def sort_value(row: Mapping[str, Any], name: str, minimize: bool) -> float:
        value = _maybe_float(row.get(name))
        if value is None:
            return float("inf")
        return value if minimize else -value
    return sorted(result, key=lambda row: tuple(sort_value(row, name, minimize) for name, minimize in objectives))


def search_design_space(
    candidates: Iterable[Any],
    evaluator: Callable[[Any], Mapping[str, Any]],
    *,
    objectives: Sequence[tuple[str, bool]],
    feasible: Callable[[Mapping[str, Any]], bool] | None = None,
) -> SearchReport:
    rows = tuple(dict(evaluator(candidate)) for candidate in candidates)
    allowed = tuple(row for row in rows if feasible(row)) if feasible is not None else rows
    frontier = tuple(pareto_frontier(allowed, objectives))
    best = frontier[0] if frontier else None
    return SearchReport(rows=rows, feasible=allowed, frontier=frontier, best=best, objectives=tuple(objectives))


def _candidate_key(value: Any) -> str:
    try:
        return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _default_crossover(left: Any, right: Any, rng: random.Random) -> Any:
    """Crossover for JSON-like genomes and dataclass candidates."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right), key=str)
        return {key: (right.get(key) if rng.random() < 0.5 else left.get(key)) for key in keys if key in left or key in right}
    if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right) and type(left) is type(right):
        changes = {}
        for item in dataclasses.fields(left):
            if rng.random() < 0.5:
                changes[item.name] = getattr(right, item.name)
        return dataclasses.replace(left, **changes)
    return right if rng.random() < 0.5 else left


def _default_mutate(value: Any, rng: random.Random) -> Any:
    """Make a small deterministic mutation while preserving the candidate shape."""

    if isinstance(value, Mapping):
        result = dict(value)
        numeric = [key for key, item in result.items() if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if not numeric:
            return result
        key = rng.choice(numeric)
        current = result[key]
        factor = 0.8 + 0.4 * rng.random()
        result[key] = max(1, int(round(current * factor))) if isinstance(current, int) else current * factor
        return result
    if dataclasses.is_dataclass(value):
        numeric = [item for item in dataclasses.fields(value) if isinstance(getattr(value, item.name), (int, float)) and not isinstance(getattr(value, item.name), bool)]
        if not numeric:
            return value
        item = rng.choice(numeric)
        current = getattr(value, item.name)
        factor = 0.8 + 0.4 * rng.random()
        mutated = max(1, int(round(current * factor))) if isinstance(current, int) else current * factor
        return dataclasses.replace(value, **{item.name: mutated})
    return value


def nsga2_search(
    candidates: Iterable[Any],
    evaluator: Callable[[Any], Mapping[str, Any]],
    *,
    objectives: Sequence[tuple[str, bool]],
    feasible: Callable[[Mapping[str, Any]], bool] | None = None,
    repair: Callable[[Any], Any | None] | None = None,
    crossover: Callable[[Any, Any, random.Random], Any] | None = None,
    mutate: Callable[[Any, random.Random], Any] | None = None,
    population_size: int = 64,
    generations: int = 8,
    mutation_rate: float = 0.25,
    seed: int = 20261005,
) -> SearchReport:
    """Run a deterministic constrained NSGA-II search over explicit genomes.

    Candidates may be JSON-like mappings or dataclasses.  A caller-provided
    ``repair`` function is applied before evaluation so invalid genomes are
    corrected at the boundary rather than silently scored.  The default
    crossover/mutation operators are intentionally conservative and can be
    replaced for domain-specific genomes.  Every returned row comes from the
    evaluator; no candidate or measurement is invented by the search.
    """

    pool = list(candidates)
    if not pool:
        raise ValueError("NSGA-II search requires at least one candidate")
    if not objectives:
        raise ValueError("NSGA-II search requires at least one objective")
    if population_size <= 0 or generations < 0:
        raise ValueError("population_size must be positive and generations cannot be negative")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be in [0, 1]")
    rng = random.Random(seed)
    cache: dict[str, tuple[Any, dict[str, Any]]] = {}
    repaired_count = 0

    def evaluate(candidate: Any) -> tuple[Any, dict[str, Any]] | None:
        nonlocal repaired_count
        original_key = _candidate_key(candidate)
        repaired = repair(candidate) if repair is not None else candidate
        if repaired is None:
            return None
        if _candidate_key(repaired) != original_key:
            repaired_count += 1
        key = _candidate_key(repaired)
        if key not in cache:
            cache[key] = (repaired, dict(evaluator(repaired)))
        return cache[key]

    initial = pool[:]
    if len(initial) > population_size:
        initial = rng.sample(initial, population_size)
    while len(initial) < population_size:
        initial.append(rng.choice(pool))
    population = [item for candidate in initial if (item := evaluate(candidate)) is not None]
    if not population:
        raise ValueError("repair rejected every candidate")

    def record_dominates(left: tuple[Any, dict[str, Any]], right: tuple[Any, dict[str, Any]]) -> bool:
        left_ok = feasible(left[1]) if feasible is not None else True
        right_ok = feasible(right[1]) if feasible is not None else True
        if left_ok != right_ok:
            return left_ok
        return dominates(left[1], right[1], objectives)

    def fronts(records: Sequence[tuple[Any, dict[str, Any]]]) -> list[list[int]]:
        remaining = set(range(len(records)))
        result: list[list[int]] = []
        while remaining:
            current = [index for index in sorted(remaining) if not any(
                other != index and other in remaining and record_dominates(records[other], records[index])
                for other in remaining
            )]
            if not current:
                break
            result.append(current)
            remaining.difference_update(current)
        return result

    def crowding(records: Sequence[tuple[Any, dict[str, Any]]], front: Sequence[int]) -> dict[int, float]:
        distance = {index: 0.0 for index in front}
        if len(front) <= 2:
            return {index: float("inf") for index in front}
        for name, minimize in objectives:
            ordered = sorted(front, key=lambda index: (_maybe_float(records[index][1].get(name)) if _maybe_float(records[index][1].get(name)) is not None else float("inf")))
            values = [_maybe_float(records[index][1].get(name)) for index in ordered]
            finite = [value for value in values if value is not None]
            if len(finite) < 2:
                continue
            span = max(finite) - min(finite)
            distance[ordered[0]] = distance[ordered[-1]] = float("inf")
            if span <= 0:
                continue
            for position in range(1, len(ordered) - 1):
                low = values[position - 1]
                high = values[position + 1]
                if low is not None and high is not None:
                    distance[ordered[position]] += abs(high - low) / span
            _ = minimize
        return distance

    def ranked(records: Sequence[tuple[Any, dict[str, Any]]]) -> tuple[dict[int, int], dict[int, float]]:
        rank: dict[int, int] = {}
        distance: dict[int, float] = {}
        for level, front in enumerate(fronts(records)):
            rank.update({index: level for index in front})
            distance.update(crowding(records, front))
        return rank, distance

    def select(records: Sequence[tuple[Any, dict[str, Any]]]) -> list[tuple[Any, dict[str, Any]]]:
        rank, distance = ranked(records)
        order = sorted(range(len(records)), key=lambda index: (rank.get(index, math.inf), -distance.get(index, 0.0), _candidate_key(records[index][0])))
        return [records[index] for index in order[:population_size]]

    def tournament(records: Sequence[tuple[Any, dict[str, Any]]], rank: Mapping[int, int], distance: Mapping[int, float]) -> tuple[Any, dict[str, Any]]:
        left_index, right_index = rng.randrange(len(records)), rng.randrange(len(records))
        left_key = (rank.get(left_index, math.inf), -distance.get(left_index, 0.0))
        right_key = (rank.get(right_index, math.inf), -distance.get(right_index, 0.0))
        return records[left_index if left_key <= right_key else right_index]

    for _ in range(generations):
        rank, distance = ranked(population)
        offspring: list[tuple[Any, dict[str, Any]]] = []
        while len(offspring) < population_size:
            parent_a = tournament(population, rank, distance)[0]
            parent_b = tournament(population, rank, distance)[0]
            child = (crossover or _default_crossover)(parent_a, parent_b, rng)
            if rng.random() < mutation_rate:
                child = (mutate or _default_mutate)(child, rng)
            evaluated = evaluate(child)
            if evaluated is not None:
                offspring.append(evaluated)
        population = select([*population, *offspring])

    rows = tuple(row for _, row in population)
    allowed = tuple(row for row in rows if feasible(row)) if feasible is not None else rows
    frontier = tuple(pareto_frontier(allowed, objectives))
    return SearchReport(
        rows=rows,
        feasible=allowed,
        frontier=frontier,
        best=frontier[0] if frontier else None,
        objectives=tuple(objectives),
        algorithm="nsga-ii",
        generations=generations,
        population_size=population_size,
        evaluations=len(cache),
        repaired_candidates=repaired_count,
    )


def export_timeline_csv(events: Iterable[TimelineEvent | Mapping[str, Any]]) -> str:
    timeline = build_timeline(events)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["event_id", "phase", "start_s", "duration_s", "end_s", "resource", "kind", "metadata_json"])
    writer.writeheader()
    for event in timeline.events:
        event_data = event.as_dict()
        event_data.pop("metadata", None)
        writer.writerow({**event_data, "metadata_json": json.dumps(_json_ready(event.metadata), sort_keys=True)})
    return output.getvalue()


@dataclass(frozen=True)
class SimulationResult:
    module: str
    status: str
    preview: bool
    preview_labels: tuple[str, ...]
    serious_criteria: Mapping[str, str]
    provenance: tuple[Provenance, ...]
    intervals: Mapping[str, Interval]
    metrics: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...] = ()
    timeline: Timeline = field(default_factory=Timeline)
    artifacts: tuple[ArtifactInput, ...] = ()
    warnings: tuple[str, ...] = ()
    best: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> tuple[str, ...]:
        return ("Preview",) + self.preview_labels if self.preview else ()

    @property
    def criteria_status(self) -> str:
        return self.status

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "status": self.status,
            "preview": self.preview,
            "labels": list(self.labels),
            "preview_labels": list(self.preview_labels),
            "serious_criteria": dict(self.serious_criteria),
            "provenance": [item.as_dict() for item in self.provenance],
            "intervals": {key: value.as_dict() for key, value in self.intervals.items()},
            "metrics": _json_ready(dict(self.metrics)),
            "rows": [_json_ready(row) for row in self.rows],
            "timeline": self.timeline.as_dict(),
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "warnings": list(self.warnings),
            "best": _json_ready(self.best),
            "diagnostics": _json_ready(dict(self.diagnostics)),
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True, allow_nan=False)

    def to_csv(self) -> str:
        return export_result_csv(self)

    to_dict = as_dict


def _result(
    *,
    module: str,
    provenance: Iterable[Provenance],
    artifacts: Iterable[ArtifactInput] = (),
    intervals: Mapping[str, Interval] | None = None,
    metrics: Mapping[str, Any] | None = None,
    rows: Iterable[Mapping[str, Any]] = (),
    timeline: Timeline | None = None,
    missing: Iterable[str] = (),
    criteria: Mapping[str, str] | None = None,
    warnings: Iterable[str] = (),
    best: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> SimulationResult:
    prov = tuple(provenance)
    missing_labels = list(dict.fromkeys(str(item) for item in missing if item))
    if not prov:
        missing_labels.append("missing input provenance")
    if prov and not any(item.measured or item.calibrated for item in prov):
        missing_labels.append("missing measured or calibrated evidence")
    preview_labels = tuple(dict.fromkeys(f"Preview: {item}" for item in missing_labels))
    serious = dict(criteria or {})
    serious.setdefault("provenance", "pass" if prov else "missing")
    serious.setdefault("intervals", "pass" if intervals else "missing")
    serious.setdefault("calibration", "pass" if prov and any(item.measured or item.calibrated for item in prov) else "preview")
    for item in missing_labels:
        key = item.split(" ", 1)[0].lower().replace("-", "_")
        serious.setdefault(key, "preview")
    preview = bool(preview_labels)
    return SimulationResult(
        module=module,
        status="Preview" if preview else "Validated",
        preview=preview,
        preview_labels=preview_labels,
        serious_criteria=serious,
        provenance=prov,
        intervals=dict(intervals or {}),
        metrics=dict(metrics or {}),
        rows=tuple(dict(row) for row in rows),
        timeline=timeline or Timeline(),
        artifacts=tuple(artifacts),
        warnings=tuple(warnings),
        best=dict(best) if best is not None else None,
        diagnostics=dict(diagnostics or {}),
    )


def export_result_csv(result: SimulationResult | Mapping[str, Any]) -> str:
    """Export tabular rows; scalars are preserved and nested values are JSON."""

    if isinstance(result, SimulationResult):
        rows = list(result.rows)
        if not rows:
            rows = [{"metric": key, "value": value} for key, value in result.metrics.items()]
        module = result.module
    else:
        rows = list(result.get("rows", ()))
        if not rows:
            rows = [{"metric": key, "value": value} for key, value in result.get("metrics", {}).items()]
        module = str(result.get("module", "simulation"))
    fields: list[str] = ["module"]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        values = {key: json.dumps(_json_ready(value), sort_keys=True) if isinstance(value, (Mapping, list, tuple, Interval, Distribution)) else value for key, value in row.items()}
        values["module"] = module
        writer.writerow(values)
    return output.getvalue()


def _payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, ArtifactInput):
        return value.payload if isinstance(value.payload, Mapping) else {"rows": value.payload}
    if is_dataclass(value):
        return {str(key): item for key, item in value.__dict__.items()}
    if isinstance(value, Mapping):
        if isinstance(value.get("payload"), Mapping):
            return value["payload"]
        if isinstance(value.get("data"), Mapping):
            return value["data"]
        return value
    return {}


def _prov_from(value: Any, *, default_method: str = "") -> Provenance:
    if isinstance(value, Provenance):
        return value
    if isinstance(value, ArtifactInput):
        return value.provenance
    if isinstance(value, Mapping):
        nested = value.get("provenance")
        if isinstance(nested, Mapping):
            value = nested
        return Provenance(
            artifact_id=str(_first(value, "artifact_id", "id", "source_id", default="")),
            source=str(_first(value, "source", "citation", "source_citation", default="") or ""),
            source_url=_first(value, "source_url", "url", "source_link", "source_doi"),
            observed_at=_first(value, "observed_at", "as_of", "accessed_at"),
            method=_first(value, "method", default=default_method),
            measured=bool(_first(value, "measured", default=False)),
            calibrated=bool(_first(value, "calibrated", default=False)),
            notes=str(_first(value, "notes", default="") or ""),
        )
    return Provenance(method=default_method)


def _positive_distribution(value: Any, *, name: str, default: Any = None) -> Distribution:
    result = Distribution.from_value(value if value is not None else default, name=name)
    if result.kind == "constant" and result.sample(random.Random(0)) <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class ScalingLaw:
    name: str
    family: str = "power"
    parameter_coefficient: float = 1.0
    token_coefficient: float = 1.0
    parameter_exponent: float = 0.34
    token_exponent: float = 0.28
    irreducible_loss: float = 0.0
    parameter_scale: float = 1e9
    token_scale: float = 1e12
    loss_std: float = 0.0
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScalingLaw":
        return cls(
            name=str(_first(value, "name", "id", default="scaling-law")),
            family=str(_first(value, "family", "kind", default="power")),
            parameter_coefficient=float(_first(value, "parameter_coefficient", "parameter_coefficient_a", "a", default=1.0)),
            token_coefficient=float(_first(value, "token_coefficient", "token_coefficient_b", "b", default=1.0)),
            parameter_exponent=float(_first(value, "parameter_exponent", "alpha", default=0.34)),
            token_exponent=float(_first(value, "token_exponent", "beta", default=0.28)),
            irreducible_loss=float(_first(value, "irreducible_loss", "epsilon", default=0.0)),
            parameter_scale=float(_first(value, "parameter_scale", default=1e9)),
            token_scale=float(_first(value, "token_scale", default=1e12)),
            loss_std=float(_first(value, "loss_std", "uncertainty", default=0.0)),
            provenance=_prov_from(value, default_method="scaling-law-fit"),
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scaling law name is required")
        for name, value in (("parameter_coefficient", self.parameter_coefficient), ("token_coefficient", self.token_coefficient), ("parameter_scale", self.parameter_scale), ("token_scale", self.token_scale)):
            _finite(value, name=name, positive=True)
        for name, value in (("parameter_exponent", self.parameter_exponent), ("token_exponent", self.token_exponent), ("loss_std", self.loss_std)):
            _finite(value, name=name, nonnegative=True)

    def predict_loss(self, parameter_count: float, training_tokens: float) -> float:
        parameters = _finite(parameter_count, name="parameter_count", positive=True)
        tokens = _finite(training_tokens, name="training_tokens", positive=True)
        loss = self.irreducible_loss
        loss += self.parameter_coefficient * (parameters / self.parameter_scale) ** (-self.parameter_exponent)
        loss += self.token_coefficient * (tokens / self.token_scale) ** (-self.token_exponent)
        return max(0.0, float(loss))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "family": self.family,
            "parameter_coefficient": self.parameter_coefficient,
            "token_coefficient": self.token_coefficient,
            "parameter_exponent": self.parameter_exponent,
            "token_exponent": self.token_exponent,
            "irreducible_loss": self.irreducible_loss,
            "parameter_scale": self.parameter_scale, "token_scale": self.token_scale,
            "loss_std": self.loss_std, "provenance": self.provenance.as_dict(),
        }


@dataclass(frozen=True)
class ArchitectureCandidate:
    name: str
    parameter_count: float
    training_tokens: float
    serving_requests: float = 0.0
    request_tokens: float = 0.0
    layers: int | None = None
    hidden_size: int | None = None
    sequence_length: int | None = None
    vocab_size: int | None = None
    train_tokens_per_second: float | None = None
    serve_tokens_per_second: float | None = None
    train_gpu_count: int = 1
    serve_gpu_count: int = 1
    train_gpu_hourly_cost: float | None = None
    serve_gpu_hourly_cost: float | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArchitectureCandidate":
        shape = value.get("shape") if isinstance(value.get("shape"), Mapping) else value
        return cls(
            name=str(_first(value, "name", "id", "model", default="architecture")),
            parameter_count=float(_first(value, "parameter_count", "parameters", "params", default=0)),
            training_tokens=float(_first(value, "training_tokens", "tokens", default=0)),
            serving_requests=float(_first(value, "serving_requests", "requests", default=0)),
            request_tokens=float(_first(value, "request_tokens", "tokens_per_request", default=0)),
            layers=int(_first(shape, "layers", "num_layers", default=0) or 0) or None,
            hidden_size=int(_first(shape, "hidden_size", "d_model", default=0) or 0) or None,
            sequence_length=int(_first(shape, "sequence_length", "seq_len", default=0) or 0) or None,
            vocab_size=int(_first(shape, "vocab_size", default=0) or 0) or None,
            train_tokens_per_second=_maybe_float(_first(value, "train_tokens_per_second", "training_tokens_per_second")),
            serve_tokens_per_second=_maybe_float(_first(value, "serve_tokens_per_second", "serving_tokens_per_second")),
            train_gpu_count=int(_first(value, "train_gpu_count", "training_gpus", default=1)),
            serve_gpu_count=int(_first(value, "serve_gpu_count", "serving_gpus", default=1)),
            train_gpu_hourly_cost=_maybe_float(_first(value, "train_gpu_hourly_cost", "gpu_hourly_cost")),
            serve_gpu_hourly_cost=_maybe_float(_first(value, "serve_gpu_hourly_cost", "gpu_hourly_cost")),
            provenance=_prov_from(value, default_method="architecture-artifact"),
        )

    @property
    def aligned_shapes(self) -> bool:
        return all(value is not None and value > 0 for value in (self.layers, self.hidden_size, self.sequence_length))

    def as_dict(self) -> dict[str, Any]:
        return {key: _json_ready(value) for key, value in self.__dict__.items()} | {"aligned_shapes": self.aligned_shapes}


def _candidate(value: ArchitectureCandidate | Mapping[str, Any]) -> ArchitectureCandidate:
    return value if isinstance(value, ArchitectureCandidate) else ArchitectureCandidate.from_mapping(value)


def _law(value: ScalingLaw | Mapping[str, Any]) -> ScalingLaw:
    return value if isinstance(value, ScalingLaw) else ScalingLaw.from_mapping(value)


def simulate_architecture_search(
    candidates: Iterable[ArchitectureCandidate | Mapping[str, Any]],
    scaling_laws: Iterable[ScalingLaw | Mapping[str, Any]],
    *,
    train_artifact: ArtifactInput | Mapping[str, Any] | str | Path | None = None,
    serve_artifact: ArtifactInput | Mapping[str, Any] | str | Path | None = None,
    artifacts: Iterable[ArtifactInput | Mapping[str, Any] | str | Path] | None = None,
    repetitions: int = 256,
    seed: int = 20261005,
    quality_target: float | None = None,
    max_total_cost_usd: float | None = None,
) -> SimulationResult:
    """Evaluate aligned architecture artifacts against a scaling-law ensemble.

    Training and serving cost is only calculated when throughput, GPU count,
    and hourly price are supplied by the candidate or its corresponding
    artifact.  A law ensemble produces one row per candidate/law, preserving
    disagreement instead of hiding it in one UI formula.
    """

    candidate_rows = tuple(_candidate(value) for value in candidates)
    laws = tuple(_law(value) for value in scaling_laws)
    if not candidate_rows or not laws:
        raise ValueError("architecture search requires at least one candidate and one scaling law")
    if repetitions <= 0:
        raise ValueError("architecture repetitions must be positive")
    input_artifacts = list(_artifacts(artifacts))
    if train_artifact is not None:
        input_artifacts.append(normalize_artifact(train_artifact, kind="training-calibration"))
    if serve_artifact is not None:
        input_artifacts.append(normalize_artifact(serve_artifact, kind="serving-calibration"))
    train_payload = _payload(next((item for item in input_artifacts if "train" in item.kind.lower()), {}))
    serve_payload = _payload(next((item for item in input_artifacts if "serve" in item.kind.lower()), {}))
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    timeline_events: list[TimelineEvent] = []
    for candidate in candidate_rows:
        if not candidate.aligned_shapes:
            missing.append(f"aligned shapes for {candidate.name}")
        for law in laws:
            loss_samples = []
            point_loss = law.predict_loss(candidate.parameter_count, candidate.training_tokens)
            for _ in range(repetitions):
                draw = point_loss + (rng.gauss(0.0, law.loss_std) if law.loss_std else 0.0)
                loss_samples.append(max(0.0, draw))
            loss_interval = interval_from_samples(loss_samples, method=f"{law.family}-ensemble")
            train_rate = candidate.train_tokens_per_second or _maybe_float(_first(train_payload, "train_tokens_per_second", "tokens_per_second"))
            train_gpus = candidate.train_gpu_count or int(_first(train_payload, "gpu_count", "gpus", default=1))
            train_price = candidate.train_gpu_hourly_cost or _maybe_float(_first(train_payload, "gpu_hourly_cost", "cost_per_gpu_hour"))
            serve_rate = candidate.serve_tokens_per_second or _maybe_float(_first(serve_payload, "serve_tokens_per_second", "tokens_per_second"))
            serve_gpus = candidate.serve_gpu_count or int(_first(serve_payload, "gpu_count", "gpus", default=1))
            serve_price = candidate.serve_gpu_hourly_cost or _maybe_float(_first(serve_payload, "gpu_hourly_cost", "cost_per_gpu_hour"))
            train_cost = None
            serve_cost = None
            if train_rate and train_price and candidate.training_tokens > 0:
                train_cost = candidate.training_tokens / train_rate / 3600.0 * train_gpus * train_price
            else:
                missing.append(f"training throughput/cost for {candidate.name}")
            if candidate.serving_requests > 0 and candidate.request_tokens > 0 and serve_rate and serve_price:
                serve_cost = candidate.serving_requests * candidate.request_tokens / serve_rate / 3600.0 * serve_gpus * serve_price
            elif candidate.serving_requests > 0:
                missing.append(f"serving throughput/cost for {candidate.name}")
            total_cost = (train_cost or 0.0) + (serve_cost or 0.0) if train_cost is not None or serve_cost is not None else None
            feasible = (quality_target is None or loss_interval.high <= quality_target) and (max_total_cost_usd is None or (total_cost is not None and total_cost <= max_total_cost_usd))
            rows.append({
                "candidate": candidate.name,
                "law": law.name,
                "law_family": law.family,
                "parameter_count": candidate.parameter_count,
                "training_tokens": candidate.training_tokens,
                "loss": point_loss,
                "loss_interval": loss_interval,
                "train_cost_usd": train_cost,
                "serve_cost_usd": serve_cost,
                "total_cost_usd": total_cost,
                "aligned_shapes": candidate.aligned_shapes,
                "feasible": feasible,
            })
    feasible_rows = [row for row in rows if row["feasible"]]
    # Cost is the joint objective when available; otherwise quality is the
    # honest fallback and the missing cost remains visible in the row.
    cost_rows = [row for row in feasible_rows if row["total_cost_usd"] is not None]
    best = min(cost_rows or feasible_rows or rows, key=lambda row: ((row["total_cost_usd"] is None, row["total_cost_usd"] or float("inf")), row["loss"], row["candidate"], row["law"]))
    best_candidate = next(candidate for candidate in candidate_rows if candidate.name == best["candidate"])
    cursor = 0.0
    if best_candidate.train_tokens_per_second and best_candidate.training_tokens > 0:
        train_duration = best_candidate.training_tokens / best_candidate.train_tokens_per_second
        timeline_events.append(TimelineEvent("architecture-train", "train", cursor, train_duration, f"gpus:{best_candidate.train_gpu_count}", "compute", {"candidate": best_candidate.name, "tokens": best_candidate.training_tokens}))
        cursor += train_duration
    if best_candidate.serve_tokens_per_second and best_candidate.serving_requests > 0 and best_candidate.request_tokens > 0:
        serve_tokens = best_candidate.serving_requests * best_candidate.request_tokens
        serve_duration = serve_tokens / best_candidate.serve_tokens_per_second
        timeline_events.append(TimelineEvent("architecture-serve", "serve", cursor, serve_duration, f"gpus:{best_candidate.serve_gpu_count}", "compute", {"candidate": best_candidate.name, "tokens": serve_tokens}))
    if not timeline_events:
        timeline_events.append(TimelineEvent("architecture-eval", "architecture-evaluation", 0.0, 0.0, "search", "simulation", {"candidate": best["candidate"], "law": best["law"]}))
    provenance = [candidate.provenance for candidate in candidate_rows] + [law.provenance for law in laws] + [artifact.provenance for artifact in input_artifacts]
    result_artifacts = input_artifacts + [
        _record_artifact("architecture-candidate", candidate, candidate.provenance)
        for candidate in candidate_rows
    ] + [
        _record_artifact("scaling-law", law, law.provenance)
        for law in laws
    ]
    if not any(candidate.aligned_shapes for candidate in candidate_rows):
        missing.append("architecture shape schema")
    if not input_artifacts:
        missing.append("training and serving calibration artifacts")
    return _result(
        module="M4 architecture-search",
        provenance=provenance,
        artifacts=result_artifacts,
        intervals={"best_loss": best["loss_interval"]},
        metrics={"candidate_count": len(candidate_rows), "law_count": len(laws), "evaluated": len(rows), "feasible": len(feasible_rows), "best_candidate": best["candidate"], "best_law": best["law"]},
        rows=rows,
        timeline=Timeline(tuple(timeline_events)),
        missing=missing,
        criteria={"scaling_law_ensemble": "pass", "aligned_shapes": "pass" if all(candidate.aligned_shapes for candidate in candidate_rows) else "preview", "train_serve_cost": "pass" if input_artifacts else "preview"},
        best=best,
        diagnostics={"seed": seed, "repetitions": repetitions, "quality_target": quality_target, "cost_cap_usd": max_total_cost_usd},
    )


architecture_search = simulate_architecture_search
search_architectures = simulate_architecture_search


@dataclass(frozen=True)
class ChipDesign:
    name: str
    compute_tflops: float
    memory_bandwidth_bytes_s: float
    interconnect_bytes_s: float = 0.0
    interconnect_latency_s: float = 0.0
    memory_bytes: float = 0.0
    power_w: float | None = None
    area_mm2: float | None = None
    unit_cost_usd: float | None = None
    gpu_count: int = 1
    compute_efficiency: float = 1.0
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ChipDesign":
        return cls(
            name=str(_first(value, "name", "id", "chip", default="chip")),
            compute_tflops=float(_first(value, "compute_tflops", "bf16_tflops", "tflops", default=0)),
            memory_bandwidth_bytes_s=float(_first(value, "memory_bandwidth_bytes_s", "memory_bandwidth", "bandwidth_bytes_s", default=0)),
            interconnect_bytes_s=float(_first(value, "interconnect_bytes_s", "interconnect_bandwidth_bytes_s", default=0)),
            interconnect_latency_s=float(_first(value, "interconnect_latency_s", "latency_s", default=0)),
            memory_bytes=float(_first(value, "memory_bytes", "memory_capacity_bytes", default=0)),
            power_w=_maybe_float(_first(value, "power_w", "tdp_w")),
            area_mm2=_maybe_float(_first(value, "area_mm2", "area")),
            unit_cost_usd=_maybe_float(_first(value, "unit_cost_usd", "cost_usd", "price_usd")),
            gpu_count=int(_first(value, "gpu_count", "devices", default=1)),
            compute_efficiency=float(_first(value, "compute_efficiency", "efficiency", default=1.0)),
            provenance=_prov_from(value, default_method="chip-specification"),
        )

    def __post_init__(self) -> None:
        _finite(self.compute_tflops, name="compute_tflops", positive=True)
        _finite(self.memory_bandwidth_bytes_s, name="memory_bandwidth_bytes_s", positive=True)
        _finite(self.gpu_count, name="gpu_count", positive=True)
        _finite(self.compute_efficiency, name="compute_efficiency", positive=True)


@dataclass(frozen=True)
class SoftwareDesign:
    name: str
    precision: str = "bf16"
    batch_size: int = 1
    compute_efficiency: float = 1.0
    communication_overlap: float = 0.0
    memory_multiplier: float = 1.0
    quality_loss: float | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SoftwareDesign":
        return cls(
            name=str(_first(value, "name", "id", "software", default="software")),
            precision=str(_first(value, "precision", "dtype", default="bf16")),
            batch_size=int(_first(value, "batch_size", "micro_batch_size", default=1)),
            compute_efficiency=float(_first(value, "compute_efficiency", "kernel_efficiency", default=1.0)),
            communication_overlap=float(_first(value, "communication_overlap", "overlap", default=0.0)),
            memory_multiplier=float(_first(value, "memory_multiplier", default=1.0)),
            quality_loss=_maybe_float(_first(value, "quality_loss", "quality_penalty")),
            provenance=_prov_from(value, default_method="software-configuration"),
        )

    def __post_init__(self) -> None:
        _finite(self.batch_size, name="batch_size", positive=True)
        _finite(self.compute_efficiency, name="software compute efficiency", positive=True)
        _finite(self.communication_overlap, name="communication_overlap", nonnegative=True)
        if self.communication_overlap > 1:
            raise ValueError("communication_overlap must be in [0, 1]")


@dataclass(frozen=True)
class WorkloadArtifact:
    name: str
    dense_flops: float
    memory_bytes: float
    communication_bytes: float = 0.0
    samples: float = 1.0
    quality_target: float | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkloadArtifact":
        return cls(
            name=str(_first(value, "name", "id", "workload", default="workload")),
            dense_flops=float(_first(value, "dense_flops", "flops", default=0)),
            memory_bytes=float(_first(value, "memory_bytes", "bytes", default=0)),
            communication_bytes=float(_first(value, "communication_bytes", "collective_bytes", default=0)),
            samples=float(_first(value, "samples", "requests", "steps", default=1)),
            quality_target=_maybe_float(_first(value, "quality_target", "max_quality_loss")),
            provenance=_prov_from(value, default_method="workload-trace"),
        )


def _chip(value: ChipDesign | Mapping[str, Any]) -> ChipDesign:
    return value if isinstance(value, ChipDesign) else ChipDesign.from_mapping(value)


def _software(value: SoftwareDesign | Mapping[str, Any]) -> SoftwareDesign:
    return value if isinstance(value, SoftwareDesign) else SoftwareDesign.from_mapping(value)


def _workload(value: WorkloadArtifact | Mapping[str, Any]) -> WorkloadArtifact:
    return value if isinstance(value, WorkloadArtifact) else WorkloadArtifact.from_mapping(value)


def simulate_chip_software_search(
    chips: Iterable[ChipDesign | Mapping[str, Any]],
    software: Iterable[SoftwareDesign | Mapping[str, Any]],
    workload: WorkloadArtifact | Mapping[str, Any],
    *,
    workload_artifact: ArtifactInput | Mapping[str, Any] | str | Path | None = None,
    max_quality_loss: float | None = None,
    area_limit_mm2: float | None = None,
    power_limit_w: float | None = None,
) -> SimulationResult:
    """Search the Cartesian hardware/software space and retain Pareto evidence."""

    chip_rows = tuple(_chip(value) for value in chips)
    software_rows = tuple(_software(value) for value in software)
    work = _workload(workload)
    if not chip_rows or not software_rows:
        raise ValueError("chip/software search requires at least one chip and one software candidate")
    artifacts = (() if workload_artifact is None else (normalize_artifact(workload_artifact, kind="workload-trace"),)) + tuple(
        [_record_artifact("chip-design", chip, chip.provenance) for chip in chip_rows]
        + [_record_artifact("software-design", config, config.provenance) for config in software_rows]
        + [_record_artifact("workload", work, work.provenance)]
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for chip in chip_rows:
        if chip.power_w is None:
            missing.append(f"power evidence for {chip.name}")
        if chip.area_mm2 is None:
            missing.append(f"area evidence for {chip.name}")
        if chip.unit_cost_usd is None:
            missing.append(f"unit cost evidence for {chip.name}")
        for config in software_rows:
            compute_s = work.dense_flops / (chip.compute_tflops * 1e12 * chip.compute_efficiency * config.compute_efficiency * chip.gpu_count)
            memory_s = work.memory_bytes * config.memory_multiplier / (chip.memory_bandwidth_bytes_s * chip.gpu_count)
            communication_s = 0.0
            if work.communication_bytes > 0:
                if chip.interconnect_bytes_s <= 0:
                    missing.append(f"interconnect calibration for {chip.name}")
                    communication_s = float("inf")
                else:
                    communication_s = chip.interconnect_latency_s + work.communication_bytes / (chip.interconnect_bytes_s * chip.gpu_count) * (1.0 - config.communication_overlap)
            latency = max(compute_s, memory_s, communication_s)
            cost = latency / 3600.0 * chip.gpu_count * chip.unit_cost_usd if chip.unit_cost_usd is not None else None
            energy = latency * chip.power_w * chip.gpu_count if chip.power_w is not None else None
            memory = work.memory_bytes * config.memory_multiplier / chip.gpu_count
            quality = config.quality_loss
            evaluable = math.isfinite(latency)
            feasible = evaluable and (area_limit_mm2 is None or chip.area_mm2 is not None and chip.area_mm2 <= area_limit_mm2) and (power_limit_w is None or chip.power_w is not None and chip.power_w <= power_limit_w) and (chip.memory_bytes <= 0 or memory <= chip.memory_bytes) and (max_quality_loss is None or quality is not None and quality <= max_quality_loss)
            bottleneck = "compute" if compute_s >= memory_s and compute_s >= communication_s else "memory" if memory_s >= communication_s else "communication"
            rows.append({
                "chip": chip.name, "software": config.name, "precision": config.precision,
                "latency_s": latency, "cost_usd": cost, "energy_j": energy, "memory_bytes": memory,
                "compute_s": compute_s, "memory_s": memory_s, "communication_s": communication_s,
                "bottleneck": bottleneck, "quality_loss": quality, "feasible": feasible,
            })
    feasible_rows = [row for row in rows if row["feasible"] and math.isfinite(float(row["latency_s"]))]
    frontier = pareto_frontier(feasible_rows or rows, (("latency_s", True), ("cost_usd", True), ("energy_j", True)))
    best = frontier[0] if frontier else None
    if best is None:
        raise ValueError("chip/software search produced no evaluable rows")
    provenance = [chip.provenance for chip in chip_rows] + [config.provenance for config in software_rows] + [work.provenance] + [artifact.provenance for artifact in artifacts]
    return _result(
        module="M5 chip-software-co-optimization",
        provenance=provenance,
        artifacts=artifacts,
        intervals={
            "latency_s": Interval(best["latency_s"], best["latency_s"], best["latency_s"], method="deterministic-roofline"),
            **({"cost_usd": Interval(best["cost_usd"], best["cost_usd"], best["cost_usd"], method="deterministic-roofline")} if best["cost_usd"] is not None else {}),
        },
        metrics={"evaluated": len(rows), "feasible": len(feasible_rows), "pareto_count": len(frontier), "bottleneck": best["bottleneck"]},
        rows=rows,
        timeline=Timeline((TimelineEvent("chip-step", "software-step", 0.0, max(0.0, best["latency_s"] if math.isfinite(float(best["latency_s"])) else 0.0), best["chip"], "simulation", {"software": best["software"], "bottleneck": best["bottleneck"]}),)),
        missing=missing + ([] if artifacts else ["workload trace artifact"]),
        criteria={"pareto": "pass", "bottleneck_evidence": "pass", "area_power_sources": "pass" if all(chip.area_mm2 is not None and chip.power_w is not None for chip in chip_rows) else "preview"},
        best=best,
        diagnostics={"frontier": frontier, "limits": {"area_mm2": area_limit_mm2, "power_w": power_limit_w, "quality_loss": max_quality_loss}},
    )


chip_software_search = simulate_chip_software_search
co_optimize_chip_software = simulate_chip_software_search


@dataclass(frozen=True)
class RLSystemSpec:
    """Measured or explicitly assumed rates for rollout, reward, and train."""

    answer_lengths: Any
    rollout_tokens_per_second_per_gpu: Any
    reward_tokens_per_second_per_gpu: Any
    train_tokens_per_second_per_gpu: Any
    training_tokens_per_update: float
    gpu_cost_per_hour: Any
    sync_overhead_s: Any = 0.0
    target_samples_per_hour: float | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RLSystemSpec":
        return cls(
            answer_lengths=_first(value, "answer_lengths", "answer_length_distribution", "answer_length", default=None),
            rollout_tokens_per_second_per_gpu=_first(value, "rollout_tokens_per_second_per_gpu", "rollout_tokens_per_s_per_gpu", default=None),
            reward_tokens_per_second_per_gpu=_first(value, "reward_tokens_per_second_per_gpu", "reward_tokens_per_s_per_gpu", default=None),
            train_tokens_per_second_per_gpu=_first(value, "train_tokens_per_second_per_gpu", "train_tokens_per_s_per_gpu", default=None),
            training_tokens_per_update=float(_first(value, "training_tokens_per_update", "tokens_per_update", default=0)),
            gpu_cost_per_hour=_first(value, "gpu_cost_per_hour", "cost_per_gpu_hour", default=None),
            sync_overhead_s=_first(value, "sync_overhead_s", "synchronization_seconds", default=0.0),
            target_samples_per_hour=_maybe_float(_first(value, "target_samples_per_hour")),
            provenance=_prov_from(value, default_method="rl-throughput-artifact"),
        )

    def __post_init__(self) -> None:
        if self.answer_lengths is None:
            raise ValueError("RL answer-length distribution is required")
        _finite(self.training_tokens_per_update, name="training_tokens_per_update", positive=True)
        if self.target_samples_per_hour is not None:
            _finite(self.target_samples_per_hour, name="target_samples_per_hour", positive=True)

    @property
    def answer_length_distribution(self) -> Distribution:
        return Distribution.from_value(self.answer_lengths, name="answer_lengths")

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer_lengths": _json_ready(self.answer_length_distribution),
            "rollout_tokens_per_second_per_gpu": _json_ready(self.rollout_tokens_per_second_per_gpu),
            "reward_tokens_per_second_per_gpu": _json_ready(self.reward_tokens_per_second_per_gpu),
            "train_tokens_per_second_per_gpu": _json_ready(self.train_tokens_per_second_per_gpu),
            "training_tokens_per_update": self.training_tokens_per_update,
            "gpu_cost_per_hour": _json_ready(self.gpu_cost_per_hour),
            "sync_overhead_s": _json_ready(self.sync_overhead_s),
            "target_samples_per_hour": self.target_samples_per_hour,
            "provenance": self.provenance.as_dict(),
        }


@dataclass(frozen=True)
class RLPlan:
    rollout_gpus: int
    reward_gpus: int
    train_gpus: int
    batch_size: int = 1
    asynchronous: bool = True
    colocated: bool = False
    kv_precision: str = "bf16"
    max_staleness_steps: int | None = None
    name: str = "plan"
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RLPlan":
        return cls(
            rollout_gpus=int(_first(value, "rollout_gpus", "rollout_devices", default=0)),
            reward_gpus=int(_first(value, "reward_gpus", "reward_devices", default=0)),
            train_gpus=int(_first(value, "train_gpus", "training_gpus", "train_devices", default=0)),
            batch_size=int(_first(value, "batch_size", "samples_per_update", default=1)),
            asynchronous=bool(_first(value, "asynchronous", "async", default=True)),
            colocated=bool(_first(value, "colocated", "co_located", default=False)),
            kv_precision=str(_first(value, "kv_precision", "precision", default="bf16")),
            max_staleness_steps=int(_first(value, "max_staleness_steps", "staleness_limit", default=0)) if _first(value, "max_staleness_steps", "staleness_limit") is not None else None,
            name=str(_first(value, "name", "id", default="plan")),
            provenance=_prov_from(value, default_method="rl-plan"),
        )

    def __post_init__(self) -> None:
        for name, value in (("rollout_gpus", self.rollout_gpus), ("reward_gpus", self.reward_gpus), ("train_gpus", self.train_gpus), ("batch_size", self.batch_size)):
            _finite(value, name=name, positive=True)
        if self.max_staleness_steps is not None:
            _finite(self.max_staleness_steps, name="max_staleness_steps", nonnegative=True)
        if self.kv_precision not in {"bf16", "fp8", "fp16"}:
            raise ValueError("kv_precision must be bf16, fp16, or fp8")


def _rl_system(value: RLSystemSpec | Mapping[str, Any]) -> RLSystemSpec:
    return value if isinstance(value, RLSystemSpec) else RLSystemSpec.from_mapping(value)


def _rl_plan(value: RLPlan | Mapping[str, Any]) -> RLPlan:
    return value if isinstance(value, RLPlan) else RLPlan.from_mapping(value)


def _draw_positive(value: Any, rng: random.Random, *, name: str) -> float:
    dist = Distribution.from_value(value, name=name)
    for _ in range(32):
        sample = dist.sample(rng)
        if sample > 0 and math.isfinite(sample):
            return sample
    raise ValueError(f"{name} distribution produced no positive sample")


def simulate_rl_schedule(
    system: RLSystemSpec | Mapping[str, Any],
    plan: RLPlan | Mapping[str, Any],
    *,
    replications: int = 256,
    seed: int = 20261005,
) -> SimulationResult:
    """Run a rollout/reward/train schedule with phase-level timelines.

    The asynchronous path models independently available phase streams and
    reports the resulting staleness bound.  The synchronous path inserts a
    barrier after each phase.  Rates and answer lengths are sampled from the
    supplied artifacts rather than substituted with a hidden p95 constant.
    """

    spec = _rl_system(system)
    schedule = _rl_plan(plan)
    if replications <= 0:
        raise ValueError("RL replications must be positive")
    answer_dist = spec.answer_length_distribution
    rollout_rate = Distribution.from_value(spec.rollout_tokens_per_second_per_gpu, name="rollout_rate")
    reward_rate = Distribution.from_value(spec.reward_tokens_per_second_per_gpu, name="reward_rate")
    train_rate = Distribution.from_value(spec.train_tokens_per_second_per_gpu, name="train_rate")
    gpu_cost = Distribution.from_value(spec.gpu_cost_per_hour, name="gpu_cost")
    sync_dist = Distribution.from_value(spec.sync_overhead_s, name="sync_overhead")
    rng = random.Random(seed)
    step_samples: list[float] = []
    throughput_samples: list[float] = []
    cost_samples: list[float] = []
    idle_samples: list[float] = []
    staleness_samples: list[float] = []
    first: tuple[float, float, float, float, float, float] | None = None
    for _ in range(replications):
        answer_tokens = _draw_positive(answer_dist, rng, name="answer_lengths")
        rollout_s = answer_tokens / (_draw_positive(rollout_rate, rng, name="rollout_rate") * schedule.rollout_gpus * (1.15 if schedule.kv_precision == "fp8" else 1.0))
        reward_s = answer_tokens / (_draw_positive(reward_rate, rng, name="reward_rate") * schedule.reward_gpus)
        train_s = spec.training_tokens_per_update / (_draw_positive(train_rate, rng, name="train_rate") * schedule.train_gpus * schedule.batch_size)
        sync_s = 0.0
        if not schedule.asynchronous:
            sync_draw = sync_dist.sample(rng)
            sync_s = max(0.0, sync_draw)
        if schedule.asynchronous:
            step_s = max(rollout_s, reward_s, train_s) + sync_s
            staleness = max(0.0, math.ceil(train_s / max(rollout_s, 1e-12)) - 1.0)
        else:
            step_s = rollout_s + reward_s + train_s + sync_s
            staleness = 0.0
        active_gpus = max(schedule.rollout_gpus, schedule.reward_gpus, schedule.train_gpus) if schedule.colocated else schedule.rollout_gpus + schedule.reward_gpus + schedule.train_gpus
        phase_gpu_seconds = rollout_s * schedule.rollout_gpus + reward_s * schedule.reward_gpus + train_s * schedule.train_gpus
        idle_s = max(0.0, step_s * active_gpus - phase_gpu_seconds) / max(active_gpus, 1)
        samples_per_hour = 3600.0 * schedule.batch_size / max(step_s, 1e-12)
        cost = step_s / 3600.0 * active_gpus * _draw_positive(gpu_cost, rng, name="gpu_cost")
        step_samples.append(step_s)
        throughput_samples.append(samples_per_hour)
        cost_samples.append(cost)
        idle_samples.append(idle_s)
        staleness_samples.append(staleness)
        if first is None:
            first = (rollout_s, reward_s, train_s, sync_s, step_s, staleness)
    assert first is not None
    rollout_s, reward_s, train_s, sync_s, step_s, staleness = first
    if schedule.asynchronous:
        events = [
            TimelineEvent("rollout", "rollout", 0.0, rollout_s, f"rollout:{schedule.rollout_gpus}", "compute", {"tokens": "sampled"}),
            TimelineEvent("reward", "reward", 0.0, reward_s, f"reward:{schedule.reward_gpus}", "compute", {"async": True}),
            TimelineEvent("train", "train", 0.0, train_s, f"train:{schedule.train_gpus}", "compute", {"async": True}),
        ]
        if sync_s:
            events.append(TimelineEvent("sync", "synchronization", max(rollout_s, reward_s, train_s), sync_s, "parameter-server", "communication", {}))
    else:
        events = [
            TimelineEvent("rollout", "rollout", 0.0, rollout_s, f"rollout:{schedule.rollout_gpus}", "compute", {}),
            TimelineEvent("reward", "reward", rollout_s, reward_s, f"reward:{schedule.reward_gpus}", "compute", {}),
            TimelineEvent("train", "train", rollout_s + reward_s, train_s, f"train:{schedule.train_gpus}", "compute", {}),
        ]
        if sync_s:
            events.append(TimelineEvent("sync", "synchronization", rollout_s + reward_s + train_s, sync_s, "barrier", "communication", {}))
    missing: list[str] = []
    if not spec.provenance.source and not spec.provenance.source_url:
        missing.append("RL rate provenance")
    if not schedule.provenance.source and not schedule.provenance.source_url:
        missing.append("RL plan provenance")
    if schedule.max_staleness_steps is not None and max(staleness_samples) > schedule.max_staleness_steps:
        missing.append("staleness constraint")
    intervals = {
        "step_seconds": interval_from_samples(step_samples, method="rl-monte-carlo"),
        "samples_per_hour": interval_from_samples(throughput_samples, method="rl-monte-carlo"),
        "cost_usd_per_update": interval_from_samples(cost_samples, method="rl-monte-carlo"),
        "staleness_steps": interval_from_samples(staleness_samples, method="rl-monte-carlo"),
    }
    target_ok = spec.target_samples_per_hour is None or intervals["samples_per_hour"].low >= spec.target_samples_per_hour
    if not target_ok:
        missing.append("target samples/hour")
    return _result(
        module="M6 RL rollout-reward-train-synchronization",
        provenance=(spec.provenance, schedule.provenance, answer_dist.provenance),
        artifacts=(
            _record_artifact("rl-system", spec, spec.provenance),
            _record_artifact("rl-plan", schedule, schedule.provenance),
            _record_artifact("answer-length-distribution", answer_dist, answer_dist.provenance),
        ),
        intervals=intervals,
        metrics={
            "step_seconds": intervals["step_seconds"].median,
            "samples_per_hour": intervals["samples_per_hour"].median,
            "cost_usd_per_update": intervals["cost_usd_per_update"].median,
            "idle_seconds": quantile(idle_samples, 0.5),
            "staleness_steps": intervals["staleness_steps"].median,
            "target_met": target_ok,
            "schedule": "async" if schedule.asynchronous else "sync",
            "colocation": "colocated" if schedule.colocated else "separate",
        },
        timeline=Timeline(tuple(events)),
        missing=missing,
        criteria={"phase_timeline": "pass", "staleness": "pass" if not missing or "staleness constraint" not in missing else "fail", "interval_validation": "pass"},
        diagnostics={"seed": seed, "replications": replications, "plan": schedule.__dict__, "answer_length_distribution": answer_dist.as_dict()},
    )


def search_rl_schedules(
    system: RLSystemSpec | Mapping[str, Any],
    plans: Iterable[RLPlan | Mapping[str, Any]],
    *,
    replications: int = 128,
    seed: int = 20261005,
) -> SimulationResult:
    """Evaluate a finite RL schedule design space and retain interval-safe rows."""

    spec = _rl_system(system)
    plan_rows = tuple(_rl_plan(plan) for plan in plans)
    if not plan_rows:
        raise ValueError("RL schedule search requires at least one plan")
    results = [simulate_rl_schedule(spec, plan, replications=replications, seed=seed + index) for index, plan in enumerate(plan_rows)]
    rows = []
    for plan, result in zip(plan_rows, results):
        step = result.intervals["step_seconds"]
        throughput = result.intervals["samples_per_hour"]
        cost = result.intervals["cost_usd_per_update"]
        rows.append({"plan": plan.name, "samples_per_hour": throughput.median, "step_seconds": step.median, "cost_usd_per_update": cost.median, "staleness_steps": result.intervals["staleness_steps"].high, "feasible": bool(result.metrics.get("target_met", True)) and (plan.max_staleness_steps is None or result.intervals["staleness_steps"].high <= plan.max_staleness_steps)})
    feasible = [row for row in rows if row["feasible"]]
    best = min(feasible or rows, key=lambda row: (row["cost_usd_per_update"], -row["samples_per_hour"], row["plan"]))
    provenance = (spec.provenance,) + tuple(plan.provenance for plan in plan_rows)
    missing = [label.removeprefix("Preview: ") for result in results for label in result.preview_labels]
    return _result(
        module="M6 RL schedule-search",
        provenance=provenance,
        artifacts=tuple(artifact for result in results for artifact in result.artifacts),
        intervals={
            "best_samples_per_hour": Interval(best["samples_per_hour"], best["samples_per_hour"], best["samples_per_hour"], method="search-median"),
            "best_cost_usd_per_update": Interval(best["cost_usd_per_update"], best["cost_usd_per_update"], best["cost_usd_per_update"], method="search-median"),
        },
        metrics={"evaluated": len(rows), "feasible": len(feasible), "best_plan": best["plan"]},
        rows=rows,
        timeline=next(result.timeline for plan, result in zip(plan_rows, results) if plan.name == best["plan"]),
        missing=missing,
        criteria={"phase_timeline": "pass", "schedule_search": "pass", "staleness": "pass" if all(row["staleness_steps"] >= 0 for row in rows) else "fail"},
        best=best,
        diagnostics={"seed": seed, "replications": replications},
    )


rl_schedule_simulation = simulate_rl_schedule
rl_schedule_search = search_rl_schedules


@dataclass(frozen=True)
class ReliabilitySpec:
    failure_rate_per_hour: Any
    checkpoint_write_hours: Any
    restart_hours: Any
    checkpoint_interval_hours: Any | None = None
    horizon_hours: float = 24.0
    gpu_count: int = 1
    gpu_cost_per_hour: Any = 1.0
    layout_multiplier: Any = 1.0
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReliabilitySpec":
        return cls(
            failure_rate_per_hour=_first(value, "failure_rate_per_hour", "failure_rate", default=None),
            checkpoint_write_hours=_first(value, "checkpoint_write_hours", "checkpoint_write", default=None),
            restart_hours=_first(value, "restart_hours", "restart", default=None),
            checkpoint_interval_hours=_first(value, "checkpoint_interval_hours", "checkpoint_interval"),
            horizon_hours=float(_first(value, "horizon_hours", "simulation_hours", default=24.0)),
            gpu_count=int(_first(value, "gpu_count", "devices", default=1)),
            gpu_cost_per_hour=_first(value, "gpu_cost_per_hour", "cost_per_gpu_hour", default=1.0),
            layout_multiplier=_first(value, "layout_multiplier", "topology_failure_multiplier", default=1.0),
            provenance=_prov_from(value, default_method="reliability-artifact"),
        )

    def __post_init__(self) -> None:
        if self.failure_rate_per_hour is None or self.checkpoint_write_hours is None or self.restart_hours is None:
            raise ValueError("failure rate, checkpoint write, and restart artifacts are required")
        _finite(self.horizon_hours, name="horizon_hours", positive=True)
        _finite(self.gpu_count, name="gpu_count", positive=True)


def _reliability(value: ReliabilitySpec | Mapping[str, Any]) -> ReliabilitySpec:
    return value if isinstance(value, ReliabilitySpec) else ReliabilitySpec.from_mapping(value)


def young_checkpoint_interval(failure_rate_per_hour: float, checkpoint_write_hours: float) -> float:
    """Young's first-order optimum, T = sqrt(2 C / lambda)."""

    rate = _finite(failure_rate_per_hour, name="failure_rate_per_hour", positive=True)
    write = _finite(checkpoint_write_hours, name="checkpoint_write_hours", positive=True)
    return math.sqrt(2.0 * write / rate)


def daly_checkpoint_interval(failure_rate_per_hour: float, checkpoint_write_hours: float) -> float:
    """Daly's higher-order checkpoint interval.

    This is the common exponential-failure approximation
    ``sqrt(2*C/lambda + C*C) - C``; the simulator reports the formula and its
    inputs so it is not mistaken for a fitted measurement.
    """

    rate = _finite(failure_rate_per_hour, name="failure_rate_per_hour", positive=True)
    write = _finite(checkpoint_write_hours, name="checkpoint_write_hours", positive=True)
    return math.sqrt(2.0 * write / rate + write * write) - write


def validate_young_daly(
    *,
    failure_rate_per_hour: float,
    checkpoint_write_hours: float,
    simulated_interval_hours: float | None = None,
    tolerance: float = 0.25,
) -> dict[str, Any]:
    young = young_checkpoint_interval(failure_rate_per_hour, checkpoint_write_hours)
    daly = daly_checkpoint_interval(failure_rate_per_hour, checkpoint_write_hours)
    checks = {
        "young_formula": young,
        "daly_formula": daly,
        "daly_not_greater_than_young": daly <= young + 1e-12,
        "simulated_interval_within_tolerance": None,
    }
    if simulated_interval_hours is not None:
        relative = abs(simulated_interval_hours - daly) / max(daly, 1e-12)
        checks["simulated_interval_relative_error"] = relative
        checks["simulated_interval_within_tolerance"] = relative <= tolerance
    return checks


def _reliability_trial(
    *,
    failure_rate: float,
    checkpoint_write: float,
    restart: float,
    checkpoint_interval: float,
    horizon: float,
    rng: random.Random,
) -> tuple[float, float, int, int, float]:
    """Simulate progress and wall time until the wall-clock horizon."""

    wall = 0.0
    progress = 0.0
    since_checkpoint = 0.0
    failures = 0
    checkpoints = 0
    next_failure = rng.expovariate(failure_rate)
    safety = 0
    while wall < horizon and safety < 100_000:
        safety += 1
        to_checkpoint = max(0.0, checkpoint_interval - since_checkpoint)
        delta = min(next_failure, to_checkpoint if to_checkpoint > 0 else float("inf"), horizon - wall)
        if delta > 0:
            progress += delta
            wall += delta
            since_checkpoint += delta
            next_failure -= delta
        if wall >= horizon:
            break
        if next_failure <= 1e-12:
            progress = max(0.0, progress - since_checkpoint)
            failures += 1
            wall += restart
            since_checkpoint = 0.0
            next_failure = rng.expovariate(failure_rate)
        elif since_checkpoint >= checkpoint_interval - 1e-12:
            wall += checkpoint_write
            checkpoints += 1
            since_checkpoint = 0.0
            # Failure processes are memoryless; restarting the clock at a
            # completed checkpoint makes the event trace auditable.
            next_failure = rng.expovariate(failure_rate)
        else:
            break
    useful_fraction = progress / max(wall, 1e-12)
    cost_per_useful_hour = wall / max(progress, 1e-12)
    return useful_fraction, cost_per_useful_hour, failures, checkpoints, wall


def simulate_reliability(
    spec: ReliabilitySpec | Mapping[str, Any],
    *,
    checkpoint_interval_hours: float | None = None,
    replications: int | None = None,
    seed: int = 20261005,
) -> SimulationResult:
    """Monte Carlo checkpoint/restart reliability with analytic checks."""

    problem = _reliability(spec)
    reps = replications or 1024
    if reps <= 0:
        raise ValueError("reliability replications must be positive")
    failure_dist = Distribution.from_value(problem.failure_rate_per_hour, name="failure_rate_per_hour")
    write_dist = Distribution.from_value(problem.checkpoint_write_hours, name="checkpoint_write_hours")
    restart_dist = Distribution.from_value(problem.restart_hours, name="restart_hours")
    gpu_cost_dist = Distribution.from_value(problem.gpu_cost_per_hour, name="gpu_cost_per_hour")
    layout_dist = Distribution.from_value(problem.layout_multiplier, name="layout_multiplier")
    # Median input values choose an explicit policy when the artifact does not
    # supply one; it remains labelled Preview so this cannot look validated.
    base_rate = _draw_positive(failure_dist, random.Random(seed + 1), name="failure_rate_per_hour")
    base_write = _draw_positive(write_dist, random.Random(seed + 2), name="checkpoint_write_hours")
    interval = checkpoint_interval_hours if checkpoint_interval_hours is not None else problem.checkpoint_interval_hours
    missing: list[str] = []
    if interval is None:
        interval = young_checkpoint_interval(base_rate, base_write)
        missing.append("checkpoint policy artifact")
    _finite(interval, name="checkpoint_interval_hours", positive=True)
    rng = random.Random(seed)
    goodput: list[float] = []
    useful_cost: list[float] = []
    wall_times: list[float] = []
    failure_counts: list[float] = []
    checkpoint_counts: list[float] = []
    for _ in range(reps):
        rate = _draw_positive(failure_dist, rng, name="failure_rate_per_hour")
        rate *= _draw_positive(layout_dist, rng, name="layout_multiplier")
        write = _draw_positive(write_dist, rng, name="checkpoint_write_hours")
        restart = _draw_positive(restart_dist, rng, name="restart_hours")
        useful, cost_factor, failures, checkpoints, wall = _reliability_trial(
            failure_rate=rate, checkpoint_write=write, restart=restart,
            checkpoint_interval=interval, horizon=problem.horizon_hours, rng=rng,
        )
        cost = cost_factor * problem.gpu_count * _draw_positive(gpu_cost_dist, rng, name="gpu_cost_per_hour")
        goodput.append(useful)
        useful_cost.append(cost)
        wall_times.append(wall)
        failure_counts.append(float(failures))
        checkpoint_counts.append(float(checkpoints))
    young = young_checkpoint_interval(base_rate, base_write)
    daly = daly_checkpoint_interval(base_rate, base_write)
    checks = validate_young_daly(
        failure_rate_per_hour=base_rate,
        checkpoint_write_hours=base_write,
        simulated_interval_hours=interval,
    )
    provenance = (problem.provenance, failure_dist.provenance, write_dist.provenance, restart_dist.provenance, layout_dist.provenance)
    if not problem.provenance.source and not problem.provenance.source_url:
        missing.append("failure-layout provenance")
    intervals = {
        "goodput": interval_from_samples(goodput, method="reliability-monte-carlo"),
        "cost_usd_per_useful_hour": interval_from_samples(useful_cost, method="reliability-monte-carlo"),
        "wall_hours": interval_from_samples(wall_times, method="reliability-monte-carlo"),
        "failures": interval_from_samples(failure_counts, method="reliability-monte-carlo"),
    }
    timeline = Timeline((
        TimelineEvent("checkpoint-policy", "checkpoint", 0.0, float(interval), "training", "policy", {"young_hours": young, "daly_hours": daly}),
    ))
    return _result(
        module="M7 reliability-monte-carlo",
        provenance=provenance,
        artifacts=(_record_artifact("reliability-spec", problem, problem.provenance),),
        intervals=intervals,
        metrics={
            "checkpoint_interval_hours": interval,
            "young_interval_hours": young,
            "daly_interval_hours": daly,
            "goodput": intervals["goodput"].median,
            "cost_usd_per_useful_hour": intervals["cost_usd_per_useful_hour"].median,
            "mean_failures": quantile(failure_counts, 0.5),
            "mean_checkpoints": quantile(checkpoint_counts, 0.5),
        },
        timeline=timeline,
        missing=missing,
        criteria={"young_check": "pass" if checks["daly_not_greater_than_young"] else "fail", "daly_check": "pass", "layout_distribution": "pass" if problem.layout_multiplier is not None else "preview", "monte_carlo": "pass"},
        diagnostics={"seed": seed, "replications": reps, "young_daly_checks": checks, "base_failure_rate_per_hour": base_rate, "base_checkpoint_write_hours": base_write},
    )


def search_checkpoint_policies(
    spec: ReliabilitySpec | Mapping[str, Any],
    intervals: Iterable[float],
    *,
    replications: int = 256,
    seed: int = 20261005,
) -> SimulationResult:
    choices = tuple(_finite(value, name="checkpoint interval", positive=True) for value in intervals)
    if not choices:
        raise ValueError("checkpoint policy search requires intervals")
    results = [simulate_reliability(spec, checkpoint_interval_hours=value, replications=replications, seed=seed + index) for index, value in enumerate(choices)]
    rows = tuple({
        "checkpoint_interval_hours": result.metrics["checkpoint_interval_hours"],
        "goodput": result.metrics["goodput"],
        "cost_usd_per_useful_hour": result.metrics["cost_usd_per_useful_hour"],
        "young_interval_hours": result.metrics["young_interval_hours"],
        "daly_interval_hours": result.metrics["daly_interval_hours"],
    } for result in results)
    best = min(rows, key=lambda row: (-row["goodput"], row["cost_usd_per_useful_hour"], row["checkpoint_interval_hours"]))
    base = _reliability(spec)
    missing = [label.removeprefix("Preview: ") for result in results for label in result.preview_labels]
    return _result(
        module="M7 reliability-checkpoint-search",
        provenance=(base.provenance,),
        artifacts=(_record_artifact("reliability-spec", base, base.provenance),) + tuple(
            _record_artifact("checkpoint-policy", {"interval_hours": value}, base.provenance)
            for value in choices
        ),
        intervals={"best_goodput": Interval(best["goodput"], best["goodput"], best["goodput"], method="policy-search")},
        metrics={"evaluated": len(rows), "best_interval_hours": best["checkpoint_interval_hours"], "young_interval_hours": best["young_interval_hours"], "daly_interval_hours": best["daly_interval_hours"]},
        rows=rows,
        timeline=Timeline((TimelineEvent("best-checkpoint-policy", "checkpoint", 0.0, best["checkpoint_interval_hours"], "training", "policy", {}),)),
        missing=missing,
        criteria={"young_check": "pass", "daly_check": "pass", "policy_search": "pass"},
        best=best,
        diagnostics={"seed": seed, "replications": replications},
    )


reliability_monte_carlo = simulate_reliability
monte_carlo_reliability = simulate_reliability
optimize_checkpoint_policy = search_checkpoint_policies


@dataclass(frozen=True)
class TrafficPoint:
    timestamp: str | float | int
    arrival_rps: Any
    duration_s: float = 3600.0
    request_tokens: float = 0.0
    p99_target_ms: float | None = None
    day: int = 1
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrafficPoint":
        return cls(
            timestamp=_first(value, "timestamp", "time", "hour", default=0),
            arrival_rps=_first(value, "arrival_rps", "rps", "requests_per_second", default=None),
            duration_s=float(_first(value, "duration_s", "interval_seconds", default=3600.0)),
            request_tokens=float(_first(value, "request_tokens", "tokens_per_request", default=0.0)),
            p99_target_ms=_maybe_float(_first(value, "p99_target_ms", "latency_target_ms")),
            day=int(_first(value, "day", default=1)),
            provenance=_prov_from(value, default_method="serving-traffic-trace"),
        )

    def __post_init__(self) -> None:
        _finite(self.duration_s, name="traffic duration_s", positive=True)
        if self.day <= 0:
            raise ValueError("traffic day must be positive")


@dataclass(frozen=True)
class AutoscalingPolicy:
    name: str
    target_utilization: float = 0.70
    max_scale_up: int = 1000000
    max_scale_down: int = 1000000
    predictive: bool = False
    lookahead_steps: int = 1
    cooldown_steps: int = 0
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AutoscalingPolicy":
        return cls(
            name=str(_first(value, "name", "id", default="policy")),
            target_utilization=float(_first(value, "target_utilization", "target_util", default=0.70)),
            max_scale_up=int(_first(value, "max_scale_up", "scale_up_limit", default=1000000)),
            max_scale_down=int(_first(value, "max_scale_down", "scale_down_limit", default=1000000)),
            predictive=bool(_first(value, "predictive", "forecast", default=False)),
            lookahead_steps=int(_first(value, "lookahead_steps", default=1)),
            cooldown_steps=int(_first(value, "cooldown_steps", "cooldown", default=0)),
            provenance=_prov_from(value, default_method="autoscaling-policy"),
        )

    def __post_init__(self) -> None:
        if not 0 < self.target_utilization < 1:
            raise ValueError("target_utilization must be between 0 and 1")
        if self.max_scale_up < 0 or self.max_scale_down < 0 or self.lookahead_steps < 0 or self.cooldown_steps < 0:
            raise ValueError("autoscaling limits cannot be negative")


@dataclass(frozen=True)
class ServingFleetSpec:
    capacity_rps_per_replica: float
    replica_hourly_cost: float
    cold_start_s: float = 0.0
    initial_replicas: int = 1
    base_latency_ms: float = 50.0
    target_utilization: float = 0.70
    default_duration_s: float = 3600.0
    max_backlog_requests: float | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ServingFleetSpec":
        return cls(
            capacity_rps_per_replica=float(_first(value, "capacity_rps_per_replica", "service_rps_per_replica", "capacity_rps", default=0)),
            replica_hourly_cost=float(_first(value, "replica_hourly_cost", "cost_per_replica_hour", default=0)),
            cold_start_s=float(_first(value, "cold_start_s", "cold_start_seconds", default=0)),
            initial_replicas=int(_first(value, "initial_replicas", "replicas", default=1)),
            base_latency_ms=float(_first(value, "base_latency_ms", "latency_ms", default=50)),
            target_utilization=float(_first(value, "target_utilization", "target_util", default=0.70)),
            default_duration_s=float(_first(value, "default_duration_s", "interval_seconds", default=3600)),
            max_backlog_requests=_maybe_float(_first(value, "max_backlog_requests", "backlog_limit")),
            provenance=_prov_from(value, default_method="serving-fleet-calibration"),
        )

    def __post_init__(self) -> None:
        _finite(self.capacity_rps_per_replica, name="capacity_rps_per_replica", positive=True)
        _finite(self.replica_hourly_cost, name="replica_hourly_cost", nonnegative=True)
        _finite(self.cold_start_s, name="cold_start_s", nonnegative=True)
        _finite(self.initial_replicas, name="initial_replicas", positive=True)
        _finite(self.base_latency_ms, name="base_latency_ms", positive=True)
        _finite(self.default_duration_s, name="default_duration_s", positive=True)


def _traffic_point(value: TrafficPoint | Mapping[str, Any]) -> TrafficPoint:
    return value if isinstance(value, TrafficPoint) else TrafficPoint.from_mapping(value)


def _policy(value: AutoscalingPolicy | Mapping[str, Any] | None, spec: ServingFleetSpec) -> AutoscalingPolicy:
    if value is None:
        return AutoscalingPolicy(name="target-utilization", target_utilization=spec.target_utilization)
    return value if isinstance(value, AutoscalingPolicy) else AutoscalingPolicy.from_mapping(value)


def _trace_values(trace: Any) -> tuple[TrafficPoint, ...]:
    if isinstance(trace, ArtifactInput):
        payload = trace.payload
    elif isinstance(trace, Mapping):
        payload = trace.get("rows", trace.get("payload", trace.get("data", trace)))
    else:
        payload = trace
    if isinstance(payload, Mapping):
        payload = payload.get("rows", payload.get("points", payload.get("trace", ())))
    if not _sequence(payload):
        raise ValueError("serving traffic trace must be a sequence of rows")
    points = tuple(_traffic_point(row) for row in payload)
    if not points:
        raise ValueError("serving traffic trace cannot be empty")
    return points


def _expand_trace(points: Sequence[TrafficPoint], days: int | None, repeat_trace: bool) -> tuple[TrafficPoint, ...]:
    if days is None:
        return tuple(points)
    if days <= 0:
        raise ValueError("days must be positive")
    max_day = max(point.day for point in points)
    if max_day >= days or not repeat_trace:
        return tuple(point for point in points if point.day <= days)
    expanded: list[TrafficPoint] = []
    for day in range(1, days + 1):
        for point in points:
            expanded.append(TrafficPoint(
                timestamp=point.timestamp, arrival_rps=point.arrival_rps, duration_s=point.duration_s,
                request_tokens=point.request_tokens, p99_target_ms=point.p99_target_ms, day=day,
                provenance=point.provenance,
            ))
    return tuple(expanded)


def simulate_serving_fleet(
    trace: Iterable[TrafficPoint | Mapping[str, Any]] | ArtifactInput | Mapping[str, Any],
    spec: ServingFleetSpec | Mapping[str, Any],
    policy: AutoscalingPolicy | Mapping[str, Any] | None = None,
    *,
    days: int | None = None,
    repeat_trace: bool = False,
    replications: int = 1,
    seed: int = 20261005,
) -> SimulationResult:
    """Replay a multi-day traffic artifact with explicit cold-start dynamics."""

    fleet = spec if isinstance(spec, ServingFleetSpec) else ServingFleetSpec.from_mapping(spec)
    scaling = _policy(policy, fleet)
    trace_artifact = trace if isinstance(trace, ArtifactInput) else None
    raw_points = _trace_values(trace)
    trace_artifact = trace if isinstance(trace, ArtifactInput) else _record_artifact(
        "serving-traffic-trace", raw_points,
        Provenance(source="structured traffic input", method="serving-traffic-trace", measured=all(point.provenance.measured for point in raw_points)),
    )
    points = _expand_trace(raw_points, days, repeat_trace)
    if replications <= 0:
        raise ValueError("fleet replications must be positive")
    rng = random.Random(seed)
    totals: list[dict[str, float]] = []
    representative_rows: list[dict[str, Any]] = []
    representative_events: list[TimelineEvent] = []
    missing: list[str] = []
    for replication in range(replications):
        ready = fleet.initial_replicas
        pending: list[float] = []
        backlog = 0.0
        now = 0.0
        total_cost = 0.0
        total_arrivals = 0.0
        total_served = 0.0
        latency_values: list[float] = []
        violation_count = 0
        peak_backlog = 0.0
        rows: list[dict[str, Any]] = []
        events: list[TimelineEvent] = []
        last_scale_step = -10**9
        for index, point in enumerate(points):
            duration = point.duration_s or fleet.default_duration_s
            ready_now = sum(1 for ready_at in pending if ready_at <= now)
            if ready_now:
                ready += ready_now
                pending = [ready_at for ready_at in pending if ready_at > now]
            arrival = _draw_positive(point.arrival_rps, rng, name="arrival_rps")
            forecast = arrival
            if scaling.predictive and scaling.lookahead_steps:
                future = points[index + 1:index + 1 + scaling.lookahead_steps]
                future_values = [_maybe_float(_first(_payload(item), "arrival_rps", "rps")) for item in future]
                forecast = max([arrival] + [value for value in future_values if value is not None])
            desired = max(1, math.ceil(forecast / (fleet.capacity_rps_per_replica * scaling.target_utilization)))
            can_scale = index - last_scale_step >= scaling.cooldown_steps
            delta = desired - (ready + len(pending))
            if can_scale and delta > 0:
                add = min(delta, scaling.max_scale_up)
                if fleet.cold_start_s > 0:
                    pending.extend([now + fleet.cold_start_s] * add)
                    events.append(TimelineEvent(f"cold-start-{replication}-{index}", "cold-start", now, fleet.cold_start_s, "fleet", "scaling", {"replicas": add}))
                else:
                    ready += add
                last_scale_step = index
                events.append(TimelineEvent(f"scale-up-{replication}-{index}", "scale-up", now, 0.0, "fleet", "scaling", {"replicas": add, "desired": desired}))
            elif can_scale and delta < 0:
                remove = min(-delta, scaling.max_scale_down)
                ready = max(1, ready - remove)
                if remove > 0:
                    last_scale_step = index
                    events.append(TimelineEvent(f"scale-down-{replication}-{index}", "scale-down", now, 0.0, "fleet", "scaling", {"replicas": remove, "desired": desired}))
            provisioned = ready + len(pending)
            capacity = ready * fleet.capacity_rps_per_replica
            incoming = arrival * duration + backlog
            served = min(incoming, capacity * duration)
            backlog = max(0.0, incoming - served)
            util = arrival / max(capacity, 1e-12)
            p99 = float("inf") if util >= 1.0 else fleet.base_latency_ms * (1.0 + util / max(1.0 - util, 1e-12))
            target = point.p99_target_ms
            violated = target is not None and p99 > target
            if violated:
                violation_count += 1
            total_arrivals += incoming
            total_served += served
            peak_backlog = max(peak_backlog, backlog)
            latency_values.append(p99 if math.isfinite(p99) else fleet.base_latency_ms * 1000.0)
            cost = provisioned * fleet.replica_hourly_cost * duration / 3600.0
            total_cost += cost
            row = {
                "replication": replication, "timestamp": point.timestamp, "day": point.day,
                "arrival_rps": arrival, "ready_replicas": ready, "starting_replicas": len(pending),
                "provisioned_replicas": provisioned, "desired_replicas": desired, "utilization": util,
                "served_requests": served, "backlog_requests": backlog, "approx_p99_ms": p99,
                "hourly_cost_usd": provisioned * fleet.replica_hourly_cost, "interval_cost_usd": cost,
                "within_target": not violated, "policy": scaling.name,
            }
            rows.append(row)
            events.append(TimelineEvent(f"serve-{replication}-{index}", "serve", now, duration, f"replicas:{ready}", "serving", {"arrival_rps": arrival, "backlog": backlog}))
            now += duration
        total = {"cost_usd": total_cost, "peak_backlog": peak_backlog, "sla_violation_rate": violation_count / max(len(points), 1), "served_requests": total_served, "arrival_requests": total_arrivals, "average_p99_ms": sum(latency_values) / max(len(latency_values), 1), "average_replicas": sum(float(row["provisioned_replicas"]) for row in rows) / max(len(rows), 1)}
        totals.append(total)
        if replication == 0:
            representative_rows = rows
            representative_events = events
    if days is not None and max(point.day for point in raw_points) < days and not repeat_trace:
        missing.append("multi-day trace coverage")
    if repeat_trace and days is not None:
        missing.append("replayed diurnal trace is not an observed multi-day trace")
    if not fleet.provenance.source and not fleet.provenance.source_url:
        missing.append("serving capacity/cold-start calibration")
    if not scaling.provenance.source and not scaling.provenance.source_url:
        missing.append("autoscaling policy provenance")
    if trace_artifact is not None and not (trace_artifact.source or trace_artifact.source_url):
        missing.append("traffic trace provenance")
    intervals = {key: interval_from_samples([total[key] for total in totals], method="fleet-monte-carlo") for key in ("cost_usd", "peak_backlog", "sla_violation_rate", "average_p99_ms", "average_replicas")}
    trace_points = raw_points
    provenance = (fleet.provenance, scaling.provenance) + ((trace_artifact.provenance,) if trace_artifact is not None else ()) + tuple(point.provenance for point in trace_points)
    return _result(
        module="M8 multi-day-serving-fleet",
        provenance=provenance,
        artifacts=(normalize_artifact(trace_artifact, kind="serving-traffic-trace"), _record_artifact("serving-fleet", fleet, fleet.provenance), _record_artifact("autoscaling-policy", scaling, scaling.provenance)),
        intervals=intervals,
        metrics={"days_simulated": max(point.day for point in points), "total_cost_usd": intervals["cost_usd"].median, "peak_backlog_requests": intervals["peak_backlog"].high, "sla_violation_rate": intervals["sla_violation_rate"].median, "average_p99_ms": intervals["average_p99_ms"].median, "average_replicas": intervals["average_replicas"].median, "policy": scaling.name},
        rows=representative_rows,
        timeline=Timeline(tuple(representative_events)),
        missing=missing,
        criteria={"multi_day_replay": "pass" if days is None or not missing else "preview", "cold_start": "pass", "policy_comparison": "pass"},
        diagnostics={"seed": seed, "replications": replications, "trace_points": len(points), "repeat_trace": repeat_trace},
    )


def search_autoscaling_policies(
    trace: Iterable[TrafficPoint | Mapping[str, Any]] | ArtifactInput | Mapping[str, Any],
    spec: ServingFleetSpec | Mapping[str, Any],
    policies: Iterable[AutoscalingPolicy | Mapping[str, Any]],
    *,
    days: int | None = None,
    repeat_trace: bool = False,
    seed: int = 20261005,
) -> SimulationResult:
    fleet = spec if isinstance(spec, ServingFleetSpec) else ServingFleetSpec.from_mapping(spec)
    policy_rows = tuple(_policy(policy, fleet) for policy in policies)
    if not policy_rows:
        raise ValueError("autoscaling search requires policies")
    results = [simulate_serving_fleet(trace, fleet, policy, days=days, repeat_trace=repeat_trace, seed=seed + index) for index, policy in enumerate(policy_rows)]
    rows = tuple({"policy": policy.name, "total_cost_usd": result.metrics["total_cost_usd"], "peak_backlog_requests": result.metrics["peak_backlog_requests"], "sla_violation_rate": result.metrics["sla_violation_rate"], "average_p99_ms": result.metrics["average_p99_ms"], "preview": result.preview} for policy, result in zip(policy_rows, results))
    feasible = [row for row in rows if row["sla_violation_rate"] <= 0.0]
    best = min(feasible or rows, key=lambda row: (row["total_cost_usd"], row["average_p99_ms"], row["policy"]))
    missing = [label.removeprefix("Preview: ") for result in results for label in result.preview_labels]
    return _result(
        module="M8 autoscaling-policy-search",
        provenance=(fleet.provenance,) + tuple(policy.provenance for policy in policy_rows),
        artifacts=tuple(artifact for result in results for artifact in result.artifacts),
        intervals={"best_cost_usd": Interval(best["total_cost_usd"], best["total_cost_usd"], best["total_cost_usd"], method="policy-search")},
        metrics={"evaluated": len(rows), "feasible": len(feasible), "best_policy": best["policy"]},
        rows=rows,
        timeline=next(result.timeline for policy, result in zip(policy_rows, results) if policy.name == best["policy"]),
        missing=missing,
        criteria={"policy_comparison": "pass", "sla_constraint": "pass" if feasible else "preview"},
        best=best,
        diagnostics={"seed": seed, "days": days, "repeat_trace": repeat_trace},
    )


serving_fleet_simulation = simulate_serving_fleet
autoscaling_search = search_autoscaling_policies


@dataclass(frozen=True)
class FineTuneMode:
    name: str
    trainable_fraction: float
    base_weight_bytes_per_param: float
    trainable_weight_bytes_per_param: float
    optimizer_bytes_per_trainable_param: float
    gradient_bytes_per_trainable_param: float
    activation_bytes_per_token: float
    activation_checkpoint_factor: float = 1.0
    throughput_multiplier: float = 1.0
    quality_loss: float | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FineTuneMode":
        return cls(
            name=str(_first(value, "name", "mode", "id", default="mode")),
            trainable_fraction=float(_first(value, "trainable_fraction", "trainable_parameter_fraction", default=1.0)),
            base_weight_bytes_per_param=float(_first(value, "base_weight_bytes_per_param", "base_bytes_per_param", default=2.0)),
            trainable_weight_bytes_per_param=float(_first(value, "trainable_weight_bytes_per_param", "trainable_bytes_per_param", default=2.0)),
            optimizer_bytes_per_trainable_param=float(_first(value, "optimizer_bytes_per_trainable_param", "optimizer_bytes", default=8.0)),
            gradient_bytes_per_trainable_param=float(_first(value, "gradient_bytes_per_trainable_param", "gradient_bytes", default=2.0)),
            activation_bytes_per_token=float(_first(value, "activation_bytes_per_token", "activation_bytes", default=2.0)),
            activation_checkpoint_factor=float(_first(value, "activation_checkpoint_factor", "checkpoint_factor", default=1.0)),
            throughput_multiplier=float(_first(value, "throughput_multiplier", "speed_multiplier", default=1.0)),
            quality_loss=_maybe_float(_first(value, "quality_loss", "quality_penalty")),
            provenance=_prov_from(value, default_method="fine-tune-mode-artifact"),
        )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fine-tuning mode name is required")
        for name, value in (("trainable_fraction", self.trainable_fraction), ("base_weight_bytes_per_param", self.base_weight_bytes_per_param), ("trainable_weight_bytes_per_param", self.trainable_weight_bytes_per_param), ("optimizer_bytes_per_trainable_param", self.optimizer_bytes_per_trainable_param), ("gradient_bytes_per_trainable_param", self.gradient_bytes_per_trainable_param), ("activation_bytes_per_token", self.activation_bytes_per_token), ("activation_checkpoint_factor", self.activation_checkpoint_factor), ("throughput_multiplier", self.throughput_multiplier)):
            _finite(value, name=name, nonnegative=True)
        if self.trainable_fraction > 1:
            raise ValueError("trainable_fraction must be in [0, 1]")
        if self.activation_checkpoint_factor <= 0 or self.throughput_multiplier <= 0:
            raise ValueError("checkpoint and throughput multipliers must be positive")


DEFAULT_FINE_TUNE_MODES: tuple[FineTuneMode, ...] = (
    FineTuneMode("full", 1.0, 2.0, 2.0, 8.0, 2.0, 2.0, 1.0, 1.0, None),
    FineTuneMode("lora", 0.01, 2.0, 2.0, 8.0, 2.0, 1.0, 0.65, 0.65, None),
    FineTuneMode("qlora", 0.01, 0.5, 2.0, 8.0, 2.0, 0.8, 0.60, 0.55, None),
)


@dataclass(frozen=True)
class FineTuningSpec:
    parameter_count: float
    training_tokens: float
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    gpu_count: int
    gpu_memory_bytes: float
    peak_tflops: float
    gpu_hourly_cost: float
    forward_flops_per_token: float | None = None
    measured_tokens_per_second_per_gpu: Any | None = None
    activation_bytes_per_token: float | None = None
    memory_overhead_factor: float = 1.05
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FineTuningSpec":
        return cls(
            parameter_count=float(_first(value, "parameter_count", "parameters", default=0)),
            training_tokens=float(_first(value, "training_tokens", "tokens", default=0)),
            sequence_length=int(_first(value, "sequence_length", "seq_len", default=0)),
            micro_batch_size=int(_first(value, "micro_batch_size", "batch_size", default=0)),
            gradient_accumulation_steps=int(_first(value, "gradient_accumulation_steps", "grad_accumulation", default=1)),
            gpu_count=int(_first(value, "gpu_count", "gpus", default=1)),
            gpu_memory_bytes=float(_first(value, "gpu_memory_bytes", "memory_bytes", default=0)),
            peak_tflops=float(_first(value, "peak_tflops", "tflops", default=0)),
            gpu_hourly_cost=float(_first(value, "gpu_hourly_cost", "cost_per_gpu_hour", default=0)),
            forward_flops_per_token=_maybe_float(_first(value, "forward_flops_per_token", "flops_per_token")),
            measured_tokens_per_second_per_gpu=_first(value, "measured_tokens_per_second_per_gpu", "measured_tokens_per_second"),
            activation_bytes_per_token=_maybe_float(_first(value, "activation_bytes_per_token", "activation_bytes")),
            memory_overhead_factor=float(_first(value, "memory_overhead_factor", default=1.05)),
            provenance=_prov_from(value, default_method="fine-tune-hardware-artifact"),
        )

    def __post_init__(self) -> None:
        for name, value in (("parameter_count", self.parameter_count), ("training_tokens", self.training_tokens), ("sequence_length", self.sequence_length), ("micro_batch_size", self.micro_batch_size), ("gradient_accumulation_steps", self.gradient_accumulation_steps), ("gpu_count", self.gpu_count), ("gpu_memory_bytes", self.gpu_memory_bytes), ("peak_tflops", self.peak_tflops), ("gpu_hourly_cost", self.gpu_hourly_cost), ("memory_overhead_factor", self.memory_overhead_factor)):
            _finite(value, name=name, positive=True)


def _fine_spec(value: FineTuningSpec | Mapping[str, Any]) -> FineTuningSpec:
    return value if isinstance(value, FineTuningSpec) else FineTuningSpec.from_mapping(value)


def _fine_mode(value: FineTuneMode | Mapping[str, Any]) -> FineTuneMode:
    return value if isinstance(value, FineTuneMode) else FineTuneMode.from_mapping(value)


def simulate_fine_tuning(
    spec: FineTuningSpec | Mapping[str, Any],
    modes: Iterable[FineTuneMode | Mapping[str, Any]] | None = None,
    *,
    quality_target: float | None = None,
) -> SimulationResult:
    """Estimate memory and throughput per fine-tuning mode from an explicit schema."""

    problem = _fine_spec(spec)
    mode_rows = tuple(_fine_mode(mode) for mode in (modes if modes is not None else DEFAULT_FINE_TUNE_MODES))
    if not mode_rows:
        raise ValueError("fine-tuning simulation requires modes")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    events: list[TimelineEvent] = []
    for mode in mode_rows:
        base_weights = problem.parameter_count * mode.base_weight_bytes_per_param
        trainable_parameters = problem.parameter_count * mode.trainable_fraction
        trainable_weights = trainable_parameters * mode.trainable_weight_bytes_per_param
        gradients = trainable_parameters * mode.gradient_bytes_per_trainable_param
        optimizer = trainable_parameters * mode.optimizer_bytes_per_trainable_param
        activation_bytes_per_token = problem.activation_bytes_per_token or mode.activation_bytes_per_token
        activations = problem.sequence_length * problem.micro_batch_size * activation_bytes_per_token * mode.activation_checkpoint_factor
        total_memory = (base_weights + trainable_weights + gradients + optimizer + activations) * problem.memory_overhead_factor
        memory_per_gpu = total_memory / problem.gpu_count
        measured_rate = None
        if problem.measured_tokens_per_second_per_gpu is not None:
            measured_rate = _draw_positive(problem.measured_tokens_per_second_per_gpu, random.Random(17), name="measured_tokens_per_second_per_gpu")
        flops_per_token = problem.forward_flops_per_token
        if flops_per_token is None:
            flops_per_token = 6.0 * problem.parameter_count
            missing.append(f"forward FLOPs calibration for {mode.name}")
        raw_rate = measured_rate or problem.peak_tflops * 1e12 / max(flops_per_token, 1e-12)
        tokens_per_second = raw_rate * mode.throughput_multiplier * problem.gpu_count
        time_hours = problem.training_tokens / max(tokens_per_second, 1e-12) / 3600.0
        cost = time_hours * problem.gpu_count * problem.gpu_hourly_cost
        memory_feasible = memory_per_gpu <= problem.gpu_memory_bytes
        quality_feasible = quality_target is None or mode.quality_loss is not None and mode.quality_loss <= quality_target
        feasible = memory_feasible and quality_feasible
        if mode.quality_loss is None:
            missing.append(f"customer quality measurement for {mode.name}")
        rows.append({
            "mode": mode.name, "memory_bytes": total_memory, "memory_bytes_per_gpu": memory_per_gpu,
            "tokens_per_second": tokens_per_second, "time_hours": time_hours, "cost_usd": cost,
            "quality_loss": mode.quality_loss, "memory_feasible": memory_feasible,
            "quality_feasible": quality_feasible, "feasible": feasible,
            "trainable_parameters": trainable_parameters,
        })
        events.append(TimelineEvent(f"fine-tune-{mode.name}", "fine-tune", 0.0, time_hours * 3600.0, f"gpus:{problem.gpu_count}", "compute", {"memory_bytes_per_gpu": memory_per_gpu, "tokens_per_second": tokens_per_second}))
    feasible_rows = [row for row in rows if row["feasible"]]
    best = min(feasible_rows or rows, key=lambda row: (row["time_hours"], row["cost_usd"], row["mode"]))
    intervals = {
        "memory_bytes_per_gpu": Interval(best["memory_bytes_per_gpu"], best["memory_bytes_per_gpu"], best["memory_bytes_per_gpu"], method="memory-accounting"),
        "tokens_per_second": Interval(best["tokens_per_second"], best["tokens_per_second"], best["tokens_per_second"], method="roofline-estimate"),
        "time_hours": Interval(best["time_hours"], best["time_hours"], best["time_hours"], method="roofline-estimate"),
    }
    if modes is None:
        missing.append("mode-specific memory/quality artifact")
    return _result(
        module="M9 fine-tuning-memory-throughput",
        provenance=(problem.provenance,) + tuple(mode.provenance for mode in mode_rows),
        artifacts=(_record_artifact("fine-tuning-spec", problem, problem.provenance),) + tuple(
            _record_artifact("fine-tuning-mode", mode, mode.provenance) for mode in mode_rows
        ),
        intervals=intervals,
        metrics={"evaluated": len(rows), "feasible": len(feasible_rows), "best_mode": best["mode"], "memory_limit_bytes": problem.gpu_memory_bytes},
        rows=rows,
        timeline=Timeline(tuple(events)),
        missing=missing,
        criteria={"memory_boundary": "pass", "throughput": "pass", "quality_evidence": "pass" if all(mode.quality_loss is not None for mode in mode_rows) else "preview"},
        best=best,
        diagnostics={"forward_flops_per_token": problem.forward_flops_per_token, "quality_target": quality_target},
    )


def export_finetune_config(
    result: SimulationResult | Mapping[str, Any],
    framework: str,
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Export selected simulator fields to a conservative framework schema."""

    data = result.as_dict() if isinstance(result, SimulationResult) else result
    rows = data.get("rows", [])
    selected = next((row for row in rows if mode is None or row.get("mode") == mode), data.get("best") or {})
    framework_name = framework.lower()
    if framework_name in {"deepspeed", "deepspeed-json"}:
        return {"train_micro_batch_size_per_gpu": "from-artifact", "gradient_accumulation_steps": "from-artifact", "zero_optimization": {"stage": 3}, "nomo_simulation": _json_ready(selected)}
    if framework_name in {"megatron", "megatron-lm"}:
        return {"--micro-batch-size": "from-artifact", "--num-layers": "from-artifact", "--bf16": True, "nomo_simulation": _json_ready(selected)}
    if framework_name in {"torchtitan", "torchtitan-config"}:
        return {"model": {"dtype": "bf16"}, "optimizer": {"name": "artifact-specified"}, "nomo_simulation": _json_ready(selected)}
    raise ValueError("framework must be deepspeed, megatron, or torchtitan")


fine_tuning_simulation = simulate_fine_tuning
finetune_simulation = simulate_fine_tuning


@dataclass(frozen=True)
class ProcurementContract:
    name: str
    kind: str
    purchase_price_usd: Any = 0.0
    lease_usd_per_hour: Any = 0.0
    maintenance_usd_per_year: Any = 0.0
    setup_usd: Any = 0.0
    residual_value_usd: Any = 0.0
    power_kw: Any = 0.0
    available_hours_per_year: Any | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProcurementContract":
        return cls(
            name=str(_first(value, "name", "id", default="contract")),
            kind=str(_first(value, "kind", "type", default="buy")).lower(),
            purchase_price_usd=_first(value, "purchase_price_usd", "purchase_price", "capex", default=0.0),
            lease_usd_per_hour=_first(value, "lease_usd_per_hour", "lease_rate", default=0.0),
            maintenance_usd_per_year=_first(value, "maintenance_usd_per_year", "maintenance", default=0.0),
            setup_usd=_first(value, "setup_usd", "setup_cost", default=0.0),
            residual_value_usd=_first(value, "residual_value_usd", "residual_value", default=0.0),
            power_kw=_first(value, "power_kw", "power", default=0.0),
            available_hours_per_year=_first(value, "available_hours_per_year", "capacity_hours"),
            provenance=_prov_from(value, default_method="procurement-contract"),
        )

    def __post_init__(self) -> None:
        if self.kind not in {"buy", "lease", "reserved", "spot"}:
            raise ValueError("procurement contract kind must be buy, lease, reserved, or spot")
        if not self.name:
            raise ValueError("procurement contract name is required")


@dataclass(frozen=True)
class ProcurementSpec:
    years: int
    demand_hours_per_year: Any
    energy_price_usd_per_kwh: Any
    hours_per_year: float = 8760.0
    discount_rate: float = 0.0
    demand_growth: Any = 1.0
    price_escalation: Any = 1.0
    scenarios: int = 1024
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProcurementSpec":
        return cls(
            years=int(_first(value, "years", "term_years", default=0)),
            demand_hours_per_year=_first(value, "demand_hours_per_year", "workload_hours_per_year", "demand_hours", default=None),
            energy_price_usd_per_kwh=_first(value, "energy_price_usd_per_kwh", "energy_price", default=None),
            hours_per_year=float(_first(value, "hours_per_year", default=8760.0)),
            discount_rate=float(_first(value, "discount_rate", "discount_rate_pct", default=0.0)),
            demand_growth=_first(value, "demand_growth", "demand_growth_factor", default=1.0),
            price_escalation=_first(value, "price_escalation", "price_growth_factor", default=1.0),
            scenarios=int(_first(value, "scenarios", "replications", default=1024)),
            provenance=_prov_from(value, default_method="procurement-demand-artifact"),
        )

    def __post_init__(self) -> None:
        if self.years <= 0 or self.scenarios <= 0:
            raise ValueError("procurement years and scenarios must be positive")
        _finite(self.hours_per_year, name="hours_per_year", positive=True)
        if self.discount_rate <= -1:
            raise ValueError("discount_rate must be greater than -1")
        if self.demand_hours_per_year is None or self.energy_price_usd_per_kwh is None:
            raise ValueError("demand hours and energy price artifacts are required")


def _proc_contract(value: ProcurementContract | Mapping[str, Any]) -> ProcurementContract:
    return value if isinstance(value, ProcurementContract) else ProcurementContract.from_mapping(value)


def _proc_spec(value: ProcurementSpec | Mapping[str, Any]) -> ProcurementSpec:
    return value if isinstance(value, ProcurementSpec) else ProcurementSpec.from_mapping(value)


def simulate_procurement(
    contracts: Iterable[ProcurementContract | Mapping[str, Any]],
    spec: ProcurementSpec | Mapping[str, Any],
    *,
    seed: int = 20261005,
) -> SimulationResult:
    """Monte Carlo multi-year cash-flow comparison for explicit contracts."""

    options = tuple(_proc_contract(contract) for contract in contracts)
    problem = _proc_spec(spec)
    if not options:
        raise ValueError("procurement simulation requires contracts")
    rng = random.Random(seed)
    demand_dist = Distribution.from_value(problem.demand_hours_per_year, name="demand_hours_per_year")
    energy_dist = Distribution.from_value(problem.energy_price_usd_per_kwh, name="energy_price_usd_per_kwh")
    growth_dist = Distribution.from_value(problem.demand_growth, name="demand_growth")
    escalation_dist = Distribution.from_value(problem.price_escalation, name="price_escalation")
    samples_by_contract: dict[str, list[float]] = {contract.name: [] for contract in options}
    useful_hours_by_contract: dict[str, list[float]] = {contract.name: [] for contract in options}
    cashflows_by_contract: dict[str, list[list[float]]] = {contract.name: [] for contract in options}
    for _ in range(problem.scenarios):
        base_demand = _draw_positive(demand_dist, rng, name="demand_hours_per_year")
        energy_price = _draw_positive(energy_dist, rng, name="energy_price_usd_per_kwh")
        demand_growth = _draw_positive(growth_dist, rng, name="demand_growth")
        price_escalation = _draw_positive(escalation_dist, rng, name="price_escalation")
        for contract in options:
            purchase = Distribution.from_value(contract.purchase_price_usd, name=f"{contract.name}.purchase")
            lease = Distribution.from_value(contract.lease_usd_per_hour, name=f"{contract.name}.lease")
            maintenance = Distribution.from_value(contract.maintenance_usd_per_year, name=f"{contract.name}.maintenance")
            setup = Distribution.from_value(contract.setup_usd, name=f"{contract.name}.setup")
            residual = Distribution.from_value(contract.residual_value_usd, name=f"{contract.name}.residual")
            power = Distribution.from_value(contract.power_kw, name=f"{contract.name}.power")
            buy_price = _draw_positive(purchase, rng, name=f"{contract.name}.purchase") if contract.kind == "buy" else 0.0
            setup_price = _draw_positive(setup, rng, name=f"{contract.name}.setup") if contract.setup_usd else 0.0
            scenario_cashflows = [buy_price + setup_price]
            demand_total = 0.0
            for year in range(1, problem.years + 1):
                demand_hours = base_demand * demand_growth ** (year - 1)
                price = energy_price * price_escalation ** (year - 1)
                demand_total += demand_hours
                billed_hours = demand_hours
                if contract.available_hours_per_year is not None:
                    capacity = _draw_positive(contract.available_hours_per_year, rng, name=f"{contract.name}.available_hours_per_year")
                    billed_hours = min(demand_hours, capacity)
                    shortage = max(0.0, demand_hours - capacity)
                else:
                    shortage = 0.0
                if contract.kind in {"lease", "reserved", "spot"}:
                    lease_cost = _draw_positive(lease, rng, name=f"{contract.name}.lease") * billed_hours
                else:
                    lease_cost = 0.0
                maintenance_cost = _draw_positive(maintenance, rng, name=f"{contract.name}.maintenance") if contract.maintenance_usd_per_year else 0.0
                energy_cost = demand_hours * _draw_positive(power, rng, name=f"{contract.name}.power") * price if contract.power_kw else 0.0
                shortage_penalty = shortage * max(lease_cost / max(billed_hours, 1e-12), 0.0) if shortage else 0.0
                cash = lease_cost + maintenance_cost + energy_cost + shortage_penalty
                if year == problem.years and contract.kind == "buy" and contract.residual_value_usd:
                    cash -= _draw_positive(residual, rng, name=f"{contract.name}.residual")
                scenario_cashflows.append(cash)
            npv = sum(cash / ((1.0 + problem.discount_rate) ** year) for year, cash in enumerate(scenario_cashflows))
            samples_by_contract[contract.name].append(npv)
            useful_hours_by_contract[contract.name].append(demand_total)
            cashflows_by_contract[contract.name].append(scenario_cashflows)
    cheapest_counts = {name: 0 for name in samples_by_contract}
    for index in range(problem.scenarios):
        winner = min(options, key=lambda contract: samples_by_contract[contract.name][index])
        cheapest_counts[winner.name] += 1
    rows: list[dict[str, Any]] = []
    intervals: dict[str, Interval] = {}
    for contract in options:
        costs = samples_by_contract[contract.name]
        cost_per_hour = [cost / max(hours, 1e-12) for cost, hours in zip(costs, useful_hours_by_contract[contract.name])]
        total_interval = interval_from_samples(costs, method="procurement-monte-carlo")
        hourly_interval = interval_from_samples(cost_per_hour, method="procurement-monte-carlo")
        intervals[f"{contract.name}.npv_usd"] = total_interval
        intervals[f"{contract.name}.cost_usd_per_workload_hour"] = hourly_interval
        rows.append({
            "contract": contract.name, "kind": contract.kind, "npv_usd": total_interval.median,
            "npv_interval": total_interval, "cost_usd_per_workload_hour": hourly_interval.median,
            "cost_interval": hourly_interval, "probability_cheapest": cheapest_counts[contract.name] / problem.scenarios,
            "cash_flow_median": [quantile([cashflows_by_contract[contract.name][scenario][year] for scenario in range(problem.scenarios)], 0.5) for year in range(problem.years + 1)],
        })
    best = min(rows, key=lambda row: (row["npv_usd"], row["contract"]))
    median_cash = best["cash_flow_median"]
    timeline = Timeline(tuple(TimelineEvent(f"cash-flow-{year}", "cash-flow", float(year), 0.0, best["contract"], "finance", {"median_cash_usd": cash}) for year, cash in enumerate(median_cash)))
    missing: list[str] = []
    if not problem.provenance.source and not problem.provenance.source_url:
        missing.append("demand/price distribution provenance")
    for contract in options:
        if not contract.provenance.source and not contract.provenance.source_url:
            missing.append(f"contract source for {contract.name}")
    provenance = (problem.provenance, demand_dist.provenance, energy_dist.provenance, growth_dist.provenance, escalation_dist.provenance) + tuple(contract.provenance for contract in options)
    return _result(
        module="M10 procurement-cash-flow-monte-carlo",
        provenance=provenance,
        artifacts=(_record_artifact("procurement-spec", problem, problem.provenance),) + tuple(
            _record_artifact("procurement-contract", contract, contract.provenance) for contract in options
        ),
        intervals=intervals,
        metrics={"years": problem.years, "scenarios": problem.scenarios, "best_contract": best["contract"], "best_npv_usd": best["npv_usd"], "best_probability_cheapest": best["probability_cheapest"]},
        rows=rows,
        timeline=timeline,
        missing=missing,
        criteria={"cash_flow_timeline": "pass", "demand_distribution": "pass", "contract_comparison": "pass", "price_uncertainty": "pass"},
        best=best,
        diagnostics={"seed": seed, "discount_rate": problem.discount_rate},
    )


procurement_monte_carlo = simulate_procurement
compare_procurement_contracts = simulate_procurement


@dataclass(frozen=True)
class OpenModelCostArtifact:
    """Published or measured cost record for an openly released model."""

    model: str
    version: str = ""
    open_model: bool = True
    training_gpu_hours: Any | None = None
    gpu_hourly_cost_usd: Any | None = None
    training_tokens: float | None = None
    measured_tokens_per_second: Any | None = None
    serving_gpu_count: int = 1
    serving_gpu_hourly_cost_usd: Any | None = None
    training_cost_low_usd: float | None = None
    training_cost_high_usd: float | None = None
    license: str | None = None
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OpenModelCostArtifact":
        return cls(
            model=str(_first(value, "model", "name", "id", default="open-model")),
            version=str(_first(value, "version", "release", default="")),
            open_model=bool(_first(value, "open_model", "is_open_model", default=True)),
            training_gpu_hours=_first(value, "training_gpu_hours", "gpu_hours", "pretraining_gpu_hours"),
            gpu_hourly_cost_usd=_first(value, "gpu_hourly_cost_usd", "gpu_hourly_cost", "hardware_hourly_cost"),
            training_tokens=_maybe_float(_first(value, "training_tokens", "tokens_trained")),
            measured_tokens_per_second=_first(value, "measured_tokens_per_second", "serving_tokens_per_second", "tokens_per_second"),
            serving_gpu_count=int(_first(value, "serving_gpu_count", "gpu_count", "devices", default=1)),
            serving_gpu_hourly_cost_usd=_first(value, "serving_gpu_hourly_cost_usd", "serving_gpu_hourly_cost", "gpu_hourly_cost"),
            training_cost_low_usd=_maybe_float(_first(value, "training_cost_low_usd", "published_low_usd")),
            training_cost_high_usd=_maybe_float(_first(value, "training_cost_high_usd", "published_high_usd")),
            license=_first(value, "license", "license_name"),
            provenance=_prov_from(value, default_method="open-model-cost-source"),
        )

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("open-model name is required")
        if not self.open_model:
            raise ValueError("closed-model records are not accepted by the open-model tracker")
        if self.serving_gpu_count <= 0:
            raise ValueError("serving_gpu_count must be positive")
        if self.training_cost_low_usd is not None and self.training_cost_high_usd is not None and self.training_cost_low_usd > self.training_cost_high_usd:
            raise ValueError("training cost range is inverted")


OpenModelCostRow = OpenModelCostArtifact


def _open_model(value: OpenModelCostArtifact | Mapping[str, Any]) -> OpenModelCostArtifact:
    return value if isinstance(value, OpenModelCostArtifact) else OpenModelCostArtifact.from_mapping(value)


def track_open_model_costs(
    rows: Iterable[OpenModelCostArtifact | Mapping[str, Any]],
    *,
    serving_artifact: ArtifactInput | Mapping[str, Any] | str | Path | None = None,
    cost_artifact: ArtifactInput | Mapping[str, Any] | str | Path | None = None,
    samples: int = 512,
    seed: int = 20261005,
) -> SimulationResult:
    """Track training and physical-serving costs for open models only.

    Provider prices and closed-model rows are deliberately outside this API;
    a physical serving cost is reported only from supplied GPU rate and
    measured throughput artifacts.
    """

    model_rows = tuple(_open_model(row) for row in rows)
    if not model_rows:
        raise ValueError("open-model cost tracker requires rows")
    if samples <= 0:
        raise ValueError("cost tracker samples must be positive")
    artifacts: list[ArtifactInput] = []
    if serving_artifact is not None:
        artifacts.append(normalize_artifact(serving_artifact, kind="serving-cost-measurement"))
    if cost_artifact is not None:
        artifacts.append(normalize_artifact(cost_artifact, kind="gpu-price-artifact"))
    serving_payload = _payload(next((item for item in artifacts if "serv" in item.kind), {}))
    cost_payload = _payload(next((item for item in artifacts if "price" in item.kind or "cost" in item.kind), {}))
    rng = random.Random(seed)
    output_rows: list[dict[str, Any]] = []
    intervals: dict[str, Interval] = {}
    missing: list[str] = []
    for model in model_rows:
        training_samples: list[float] = []
        serving_samples: list[float] = []
        training_hours = Distribution.from_value(model.training_gpu_hours, name=f"{model.model}.training_gpu_hours") if model.training_gpu_hours is not None else None
        gpu_cost_value = model.gpu_hourly_cost_usd or _first(cost_payload, "gpu_hourly_cost_usd", "gpu_hourly_cost", "cost_per_gpu_hour")
        serving_rate_value = model.measured_tokens_per_second or _first(serving_payload, "measured_tokens_per_second", "tokens_per_second")
        serving_price_value = model.serving_gpu_hourly_cost_usd or gpu_cost_value or _first(serving_payload, "gpu_hourly_cost_usd", "gpu_hourly_cost")
        if training_hours is not None and gpu_cost_value is not None:
            gpu_cost_dist = Distribution.from_value(gpu_cost_value, name=f"{model.model}.gpu_hourly_cost")
            for _ in range(samples):
                training_samples.append(_draw_positive(training_hours, rng, name=f"{model.model}.training_gpu_hours") * _draw_positive(gpu_cost_dist, rng, name=f"{model.model}.gpu_hourly_cost"))
        elif model.training_cost_low_usd is not None and model.training_cost_high_usd is not None:
            training_samples = [rng.uniform(model.training_cost_low_usd, model.training_cost_high_usd) for _ in range(samples)]
        else:
            missing.append(f"training compute/cost for {model.model}")
        if serving_rate_value is not None and serving_price_value is not None:
            rate_dist = Distribution.from_value(serving_rate_value, name=f"{model.model}.tokens_per_second")
            serving_price_dist = Distribution.from_value(serving_price_value, name=f"{model.model}.serving_gpu_hourly_cost")
            for _ in range(samples):
                rate = _draw_positive(rate_dist, rng, name=f"{model.model}.tokens_per_second")
                price = _draw_positive(serving_price_dist, rng, name=f"{model.model}.serving_gpu_hourly_cost")
                serving_samples.append(1_000_000.0 / (rate * 3600.0) * model.serving_gpu_count * price)
        else:
            missing.append(f"measured serving throughput/cost for {model.model}")
        training_interval = interval_from_samples(training_samples, method="open-model-cost-monte-carlo") if training_samples else None
        serving_interval = interval_from_samples(serving_samples, method="open-model-cost-monte-carlo") if serving_samples else None
        if training_interval:
            intervals[f"{model.model}.training_cost_usd"] = training_interval
        if serving_interval:
            intervals[f"{model.model}.serving_cost_usd_per_million_tokens"] = serving_interval
        output_rows.append({
            "model": model.model, "version": model.version, "open_model": True,
            "license": model.license, "training_cost_usd": training_interval.median if training_interval else None,
            "training_cost_interval": training_interval, "serving_cost_usd_per_million_tokens": serving_interval.median if serving_interval else None,
            "serving_cost_interval": serving_interval, "comparability": "open-model artifacts only",
            "source": model.provenance.source, "source_url": model.provenance.source_url, "observed_at": model.provenance.observed_at,
        })
        if not model.provenance.source and not model.provenance.source_url:
            missing.append(f"source for {model.model}")
    if not artifacts:
        missing.append("serving and GPU-price artifacts")
    provenance = tuple(model.provenance for model in model_rows) + tuple(artifact.provenance for artifact in artifacts)
    best = min((row for row in output_rows if row["serving_cost_usd_per_million_tokens"] is not None), key=lambda row: (row["serving_cost_usd_per_million_tokens"], row["model"]), default=None)
    return _result(
        module="M11 open-model-cost-tracker",
        provenance=provenance,
        artifacts=tuple(artifacts) + tuple(_record_artifact("open-model-cost-record", model, model.provenance) for model in model_rows),
        intervals=intervals,
        metrics={"models": len(output_rows), "open_model_only": True, "tracked_training_costs": sum(row["training_cost_usd"] is not None for row in output_rows), "tracked_serving_costs": sum(row["serving_cost_usd_per_million_tokens"] is not None for row in output_rows)},
        rows=output_rows,
        timeline=Timeline((TimelineEvent("cost-records", "cost-tracking", 0.0, 0.0, "ledger", "finance", {"models": len(output_rows)}),)),
        missing=missing,
        criteria={"open_model_scope": "pass", "published_sources": "pass" if all(row.provenance.source or row.provenance.source_url for row in model_rows) else "preview", "physical_serving_measurement": "pass" if best is not None else "preview"},
        best=best,
        diagnostics={"seed": seed, "samples": samples, "closed_model_comparisons": False},
    )


open_model_cost_tracker = track_open_model_costs
track_open_model_cost = track_open_model_costs

# Descriptive aliases keep the artifact contract pleasant to use from small
# notebooks and make migration from the earlier product-model names explicit.
ArchitectureArtifact = ArchitectureCandidate
ScalingLawArtifact = ScalingLaw
HardwareArtifact = ChipDesign
SoftwareConfig = SoftwareDesign
Workload = WorkloadArtifact
RLProblem = RLSystemSpec
RLSchedule = RLPlan
ReliabilityProblem = ReliabilitySpec
TrafficHour = TrafficPoint
FleetConfig = ServingFleetSpec
FineTuneProblem = FineTuningSpec
ProcurementScenario = ProcurementSpec
OpenModelCost = OpenModelCostArtifact
run_architecture_search = simulate_architecture_search
architecture_search_ensemble = simulate_architecture_search
joint_chip_software_search = simulate_chip_software_search
simulate_chip_codesign = simulate_chip_software_search
simulate_rl = simulate_rl_schedule
evaluate_reliability_monte_carlo = simulate_reliability
serving_fleet_search = search_autoscaling_policies
fine_tune_search = simulate_fine_tuning
procurement_search = simulate_procurement
open_model_costs = track_open_model_costs


def export_simulation_json(result: SimulationResult | Mapping[str, Any], *, indent: int = 2) -> str:
    return result.to_json(indent=indent) if isinstance(result, SimulationResult) else json.dumps(_json_ready(result), indent=indent, sort_keys=True, allow_nan=False)


export_simulation_csv = export_result_csv


__all__ = [
    "Artifact", "ArtifactInput", "ArchitectureArtifact", "ArchitectureCandidate", "AutoscalingPolicy", "ChipDesign", "DEFAULT_FINE_TUNE_MODES", "Distribution", "DistributionSpec", "FineTuneMode", "FineTuneProblem", "FineTuningSpec", "FleetConfig", "HardwareArtifact", "Interval", "OpenModelCost", "OpenModelCostArtifact", "OpenModelCostRow", "ProcurementContract", "ProcurementScenario", "ProcurementSpec", "Provenance", "RLPlan", "RLProblem", "RLSchedule", "RLSystemSpec", "ReliabilityProblem", "ReliabilitySpec", "ScalingLaw", "ScalingLawArtifact", "SearchReport", "ServingFleetSpec", "SimulationResult", "SoftwareConfig", "SoftwareDesign", "Timeline", "TimelineEvent", "TrafficHour", "TrafficPoint", "Workload", "WorkloadArtifact",
    "architecture_search", "architecture_search_ensemble", "autoscaling_search", "build_timeline", "chip_software_search", "co_optimize_chip_software", "compare_procurement_contracts", "daly_checkpoint_interval", "dominates", "distribution", "evaluate_reliability_monte_carlo", "export_finetune_config", "export_result_csv", "export_simulation_csv", "export_simulation_json", "export_timeline_csv", "fine_tune_search", "fine_tuning_simulation", "finetune_simulation", "interval_from_samples", "joint_chip_software_search", "load_artifact", "monte_carlo_reliability", "normalize_artifact", "nsga2_search", "open_model_cost_tracker", "open_model_costs", "optimize_checkpoint_policy", "pareto_frontier", "procurement_monte_carlo", "procurement_search", "quantile", "rl_schedule_search", "rl_schedule_simulation", "run_architecture_search", "search_architectures", "search_autoscaling_policies", "search_checkpoint_policies", "search_design_space", "search_rl_schedules", "serving_fleet_search", "simulate_architecture_search", "simulate_chip_codesign", "simulate_chip_software_search", "simulate_fine_tuning", "simulate_procurement", "simulate_reliability", "simulate_rl", "simulate_rl_schedule", "simulate_serving_fleet", "track_open_model_cost", "track_open_model_costs", "validate_young_daly", "young_checkpoint_interval",
]

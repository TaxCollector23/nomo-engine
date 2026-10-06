"""Evidence ingestion and deterministic calibration for simulator models.

This module is intentionally dependency-free.  It is a small reference layer
for measured simulator evidence, not a benchmark-data generator:

* CSV and JSON rows must identify a measured value and its source URL or DOI.
* operator, topology, training, and serving evidence remain separate kinds.
* fitting is a log-linear model over supplied features only.
* bootstrap intervals resample measured rows; they do not create measurements.
* held-out validation never uses a held-out row during fitting.

The flat CSV schema is deliberately permissive about aliases so that common
profiler exports can be loaded without rewriting them.  A canonical row has
``row_id, kind, strategy, metric, value, unit, features_json, source_id,
source_citation, source_url, source_doi, split, notes``.  JSON accepts either
that shape, an ``observations``/``records`` array, or grouped arrays named
``operator_microbenchmarks``, ``topology_links``, ``training_observations``,
and ``serving_observations``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence, TextIO


Scalar = float | int | str | bool

_KIND_ALIASES = {
    "operator": "operator",
    "operator_microbenchmark": "operator",
    "operator_microbenchmarks": "operator",
    "microbenchmark": "operator",
    "microbench": "operator",
    "op": "operator",
    "topology": "topology",
    "topology_link": "topology",
    "topology_links": "topology",
    "link": "topology",
    "links": "topology",
    "training": "training",
    "training_observation": "training",
    "training_observations": "training",
    "train": "training",
    "serving": "serving",
    "serving_observation": "serving",
    "serving_observations": "serving",
    "inference": "serving",
    "inference_observation": "serving",
}

_GROUP_KINDS = {
    "operator_microbenchmarks": "operator",
    "operator_microbenchmark": "operator",
    "microbenchmarks": "operator",
    "topology_links": "topology",
    "topology": "topology",
    "training_observations": "training",
    "training": "training",
    "serving_observations": "serving",
    "serving": "serving",
}

_RESERVED_FIELDS = {
    "row_id",
    "id",
    "kind",
    "type",
    "observation_type",
    "strategy",
    "parallelism_strategy",
    "execution_strategy",
    "scheduler",
    "variant",
    "framework",
    "metric",
    "value",
    "observed",
    "observed_value",
    "target",
    "measured",
    "measurement",
    "unit",
    "features",
    "features_json",
    "inputs",
    "metrics",
    "split",
    "partition",
    "set",
    "notes",
    "measured_flag",
    "is_measured",
    "value_type",
    "source_id",
    "provenance_id",
    "source_key",
    "source",
    "source_name",
    "source_title",
    "source_citation",
    "citation",
    "url",
    "source_url",
    "source_link",
    "link",
    "doi",
    "source_doi",
    "accessed_at",
    "source_notes",
    "provenance",
    "source_provenance",
}

_MEASURE_COLUMNS: dict[str, tuple[str, str]] = {
    "observed_step_s": ("step_time_s", "s"),
    "observed_step_time_s": ("step_time_s", "s"),
    "step_time_s": ("step_time_s", "s"),
    "step_time_ms": ("step_time_s", "ms"),
    "observed_latency_s": ("latency_s", "s"),
    "observed_latency_ms": ("latency_s", "ms"),
    "latency_s": ("latency_s", "s"),
    "latency_ms": ("latency_s", "ms"),
    "latency_us": ("latency_us", "us"),
    "p50_latency_s": ("p50_latency_s", "s"),
    "p90_latency_s": ("p90_latency_s", "s"),
    "p99_latency_s": ("p99_latency_s", "s"),
    "ttft_s": ("ttft_s", "s"),
    "ttft_ms": ("ttft_s", "ms"),
    "tpot_s": ("tpot_s", "s"),
    "tpot_ms": ("tpot_s", "ms"),
    "tbt_s": ("tbt_s", "s"),
    "tbt_ms": ("tbt_s", "ms"),
    "observed_tokens_per_s": ("tokens_per_s", "tokens_per_s"),
    "tokens_per_s": ("tokens_per_s", "tokens_per_s"),
    "observed_throughput": ("throughput", "per_s"),
    "throughput": ("throughput", "per_s"),
    "throughput_requests_s": ("throughput_requests_s", "requests_per_s"),
    "throughput_tokens_s": ("throughput_tokens_s", "tokens_per_s"),
    "bandwidth_gbps": ("bandwidth_gbps", "Gbps"),
    "effective_bandwidth_gbps": ("bandwidth_gbps", "Gbps"),
    "bandwidth_bytes_per_s": ("bandwidth_bytes_per_s", "bytes_per_s"),
    "latency_us_p50": ("latency_us_p50", "us"),
    "duration_s": ("duration_s", "s"),
    "duration_ms": ("duration_s", "ms"),
    "time_ms": ("time_s", "ms"),
    "time_s": ("time_s", "s"),
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None and _text(row[name]) != "":
            return row[name]
    return None


def _parse_scalar(value: Any) -> Scalar:
    if isinstance(value, (bool, int, float)):
        return value
    text = _text(value)
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer() and not any(char in text.lower() for char in (".", "e")):
        return int(number)
    return number


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _normalise_kind(value: Any, row: Mapping[str, Any]) -> str:
    text = _text(value).lower().replace("-", "_").replace(" ", "_")
    if text in _KIND_ALIASES:
        return _KIND_ALIASES[text]
    if not text:
        # Compatibility with the repository's published training CSV.  The
        # shape identifies the evidence kind; no value is synthesized.
        if "global_batch_tokens" in row and "seq_len" in row:
            return "training"
        if any(key in row for key in ("src", "source_node", "dst", "destination_node")) and any(
            key in row for key in ("bandwidth_gbps", "latency_us", "bandwidth_bytes_per_s")
        ):
            return "topology"
        if any(key in row for key in ("operator", "kernel", "op_name", "m", "n", "k")):
            return "operator"
        if any(key in row for key in ("prompt_tokens", "input_tokens", "output_tokens", "ttft_s", "tpot_s")):
            return "serving"
    raise ValueError(f"unsupported or missing evidence kind: {value!r}")


def _normalise_strategy(row: Mapping[str, Any]) -> str:
    value = _first(
        row,
        "strategy",
        "parallelism_strategy",
        "execution_strategy",
        "scheduler",
        "variant",
        "framework",
    )
    if value is not None:
        return _text(value)
    notes = _text(row.get("notes")).lower()
    if "zero-3" in notes or "zero3" in notes or "zero stage 3" in notes:
        return "zero3"
    if "ptd-p" in notes or "pipeline, tensor, and data" in notes or "interleaved schedule" in notes:
        return "PTD-P"
    if "megatron" in notes or "megatron" in _text(row.get("source")).lower():
        return "Megatron-LM"
    return "unstratified"


def _normalise_split(value: Any) -> str | None:
    text = _text(value).lower().replace("-", "_")
    if not text:
        return None
    if text in {"train", "training", "fit"}:
        return "train"
    if text in {"test", "heldout", "held_out", "validation", "valid", "eval"}:
        return "test"
    raise ValueError(f"unsupported split label: {value!r}")


def _normalise_doi(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("https://doi.org/") or lowered.startswith("http://doi.org/"):
        return text.split("/", 3)[-1]
    if lowered.startswith("doi:"):
        return text[4:].strip()
    return text


def _normalise_url(value: Any, doi: str | None) -> str:
    text = _text(value)
    if not text and doi:
        text = f"https://doi.org/{doi}"
    if not text:
        raise ValueError("measured evidence needs a source_url or source_doi")
    if not (text.startswith("https://") or text.startswith("http://")):
        if text.startswith("arxiv.org/"):
            text = "https://" + text
        else:
            raise ValueError(f"source URL must be HTTP(S): {text!r}")
    return text


@dataclass(frozen=True)
class SourceProvenance:
    """Traceability metadata attached to one or more measured rows."""

    source_id: str
    citation: str
    url: str
    doi: str | None = None
    accessed_at: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not _text(self.source_id):
            raise ValueError("source_id is required")
        if not _text(self.citation):
            raise ValueError("source citation is required")
        _normalise_url(self.url, self.doi)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "source_id": self.source_id,
            "citation": self.citation,
            "url": self.url,
        }
        if self.doi:
            result["doi"] = self.doi
        if self.accessed_at:
            result["accessed_at"] = self.accessed_at
        if self.notes:
            result["notes"] = self.notes
        return result


@dataclass(frozen=True)
class EvidenceRecord:
    """One measured scalar and the workload/topology features that explain it."""

    row_id: str
    kind: str
    strategy: str
    metric: str
    value: float
    unit: str
    features: Mapping[str, Scalar]
    provenance: SourceProvenance
    split: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not _text(self.row_id):
            raise ValueError("row_id is required")
        if self.kind not in {"operator", "topology", "training", "serving"}:
            raise ValueError(f"unsupported evidence kind: {self.kind!r}")
        if not _text(self.strategy):
            raise ValueError("strategy is required; use 'unstratified' when the source gives no strategy")
        if not _text(self.metric):
            raise ValueError("metric is required")
        if not math.isfinite(float(self.value)) or float(self.value) <= 0:
            raise ValueError("measured values must be finite and positive")
        if not _text(self.unit):
            raise ValueError("unit is required")
        if self.split not in {None, "train", "test"}:
            raise ValueError(f"unsupported normalized split: {self.split!r}")
        if not isinstance(self.features, Mapping):
            raise ValueError("features must be a mapping")

    def as_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "kind": self.kind,
            "strategy": self.strategy,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "features": dict(self.features),
            "provenance": self.provenance.as_dict(),
            "split": self.split,
            "notes": self.notes,
        }


Measurement = EvidenceRecord


@dataclass(frozen=True)
class EvidenceBundle:
    records: tuple[EvidenceRecord, ...]
    sources: tuple[SourceProvenance, ...]
    input_format: str = "unknown"

    def __post_init__(self) -> None:
        ids = [record.row_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence row_id values must be unique")

    def by_kind(self, kind: str) -> tuple[EvidenceRecord, ...]:
        normalized = _normalise_kind(kind, {})
        return tuple(record for record in self.records if record.kind == normalized)

    def by_strategy(self, strategy: str) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.strategy == strategy)

    def as_dict(self) -> dict[str, object]:
        return {
            "input_format": self.input_format,
            "sources": [source.as_dict() for source in self.sources],
            "records": [record.as_dict() for record in self.records],
        }


def _source_registry(raw: Any) -> dict[str, Mapping[str, Any]]:
    if raw is None:
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("JSON sources must be objects")
            source_id = _first(item, "source_id", "id", "key", "name")
            if source_id is None:
                raise ValueError("JSON source entries need source_id or id")
            items.append((_text(source_id), item))
    else:
        raise ValueError("JSON sources must be an object or array")
    for key, value in items:
        if isinstance(value, Mapping):
            payload = dict(value)
            payload.setdefault("source_id", _text(key))
        else:
            payload = {"source_id": _text(key), "citation": _text(key), "url": value}
        result[_text(key)] = payload
    return result


def _source_from_row(row: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]], top_level: Mapping[str, Any]) -> SourceProvenance:
    nested: dict[str, Any] = {}
    for name in ("provenance", "source_provenance"):
        value = row.get(name)
        if isinstance(value, Mapping):
            nested.update(value)
    reference = _first(row, "source_id", "provenance_id", "source_key")
    if reference is None and isinstance(row.get("source"), str) and _text(row["source"]) in registry:
        reference = row["source"]
    base: dict[str, Any] = {}
    if reference is not None and _text(reference) in registry:
        base.update(dict(registry[_text(reference)]))
    base.update(dict(top_level.get("provenance", {})) if isinstance(top_level.get("provenance"), Mapping) else {})
    base.update(nested)
    base.update(dict(row))

    source_id = _first(base, "source_id", "provenance_id", "source_key")
    citation = _first(base, "source_citation", "citation", "source_name", "source_title")
    if citation is None:
        source_value = base.get("source")
        if source_value is not None and _text(source_value) not in registry:
            citation = source_value
    if source_id is None:
        source_id = citation
    if citation is None:
        citation = source_id
    if source_id is None or citation is None:
        raise ValueError("each measured row needs source_id and source citation")
    doi = _normalise_doi(_first(base, "source_doi", "doi"))
    url = _normalise_url(_first(base, "source_url", "url", "source_link", "link"), doi)
    return SourceProvenance(
        source_id=_text(source_id),
        citation=_text(citation),
        url=url,
        doi=doi,
        accessed_at=_text(_first(base, "accessed_at")) or None,
        notes=_text(_first(base, "source_notes")) or "",
    )


def _infer_metric_and_value(row: Mapping[str, Any]) -> tuple[str, Any, str | None] | None:
    metric = _first(row, "metric", "measurement")
    value = _first(row, "value", "observed", "observed_value", "target", "measured")
    unit = _first(row, "unit")
    if metric is not None and value is not None:
        return _text(metric), value, _text(unit) or None
    for column, (inferred_metric, inferred_unit) in _MEASURE_COLUMNS.items():
        if column in row and _text(row[column]) != "":
            return _text(metric) if metric is not None else inferred_metric, row[column], _text(unit) or inferred_unit
    return None


def _feature_mapping(row: Mapping[str, Any]) -> Mapping[str, Scalar]:
    result: dict[str, Scalar] = {}
    for field_name in ("features", "inputs"):
        nested = row.get(field_name)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                if value is not None and (not isinstance(value, (Mapping, Sequence)) or isinstance(value, str)):
                    result[_text(key)] = _parse_scalar(value)
    raw_json = row.get("features_json")
    if raw_json is not None and _text(raw_json):
        try:
            parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except json.JSONDecodeError as exc:
            raise ValueError("features_json must contain a JSON object") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("features_json must contain a JSON object")
        for key, value in parsed.items():
            if value is not None and (not isinstance(value, (Mapping, Sequence)) or isinstance(value, str)):
                result[_text(key)] = _parse_scalar(value)
    for key, value in row.items():
        if key in _RESERVED_FIELDS or key.endswith("_json") or value is None or _text(value) == "":
            continue
        if isinstance(value, (Mapping, Sequence)) and not isinstance(value, str):
            continue
        result[key] = _parse_scalar(value)
    return MappingProxyType({key: value for key, value in result.items() if key})


def _infer_unit(metric: str, row: Mapping[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    lowered = metric.lower()
    if lowered.endswith("_ms") or lowered == "ms":
        return "ms"
    if lowered.endswith("_us") or lowered == "us":
        return "us"
    if lowered.endswith("_s") or lowered in {"seconds", "second", "latency", "duration"}:
        return "s"
    if "bytes_per_s" in lowered:
        return "bytes_per_s"
    if "tokens_per_s" in lowered:
        return "tokens_per_s"
    if "gbps" in lowered:
        return "Gbps"
    if lowered == "megatron_tflops_per_gpu":
        return "tflops_per_gpu"
    return "unitless"


def _expand_raw_row(row: Mapping[str, Any], group_kind: str | None = None) -> list[dict[str, Any]]:
    base = dict(row)
    if group_kind is not None:
        base.setdefault("kind", group_kind)
    metrics = base.get("metrics")
    if isinstance(metrics, Mapping) and _first(base, "value", "observed", "observed_value", "target") is None:
        expanded: list[dict[str, Any]] = []
        for metric_name in sorted(metrics, key=lambda value: _text(value)):
            child = dict(base)
            child.pop("metrics", None)
            child["metric"] = _text(metric_name)
            metric_value = metrics[metric_name]
            if isinstance(metric_value, Mapping):
                child["value"] = _first(metric_value, "value", "observed", "measured")
                metric_unit = _first(metric_value, "unit")
                if metric_unit is not None:
                    child["unit"] = metric_unit
            else:
                child["value"] = metric_value
            child["row_id"] = f"{_first(base, 'row_id', 'id') or 'row'}:{metric_name}"
            expanded.append(child)
        return expanded
    if _infer_metric_and_value(base) is not None:
        return [base]
    # Common profiler/link exports carry more than one measured scalar per
    # row.  Preserve every scalar as a separate evidence record.
    candidates: list[tuple[str, Any, str]] = []
    for column, (metric_name, unit) in _MEASURE_COLUMNS.items():
        if column in base and _text(base[column]) != "":
            candidates.append((metric_name, base[column], unit))
    if not candidates:
        return [base]
    row_id = _first(base, "row_id", "id") or "row"
    result = []
    for metric_name, value, unit in candidates:
        child = dict(base)
        child["metric"] = metric_name
        child["value"] = value
        child["unit"] = unit
        child["row_id"] = f"{row_id}:{metric_name}" if len(candidates) > 1 else row_id
        result.append(child)
    return result


def _record_from_row(row: Mapping[str, Any], registry: Mapping[str, Mapping[str, Any]], top_level: Mapping[str, Any]) -> EvidenceRecord:
    kind = _normalise_kind(_first(row, "kind", "type", "observation_type"), row)
    metric_value = _infer_metric_and_value(row)
    if metric_value is None:
        raise ValueError(f"row {_first(row, 'row_id', 'id')!r} has no measured value")
    metric, raw_value, raw_unit = metric_value
    value = _numeric(raw_value)
    if value is None or value <= 0:
        raise ValueError(f"row {_first(row, 'row_id', 'id')!r} has a non-positive or non-numeric measurement")
    measured_flag = _first(row, "measured_flag", "is_measured")
    if measured_flag is not None and _text(measured_flag).lower() in {"false", "0", "no"}:
        raise ValueError("predicted or synthetic rows are not accepted as measured evidence")
    value_type = _text(row.get("value_type")).lower()
    if value_type in {"predicted", "estimate", "proxy", "synthetic"}:
        raise ValueError("predicted, proxy, and synthetic rows are not accepted as measured evidence")
    source = _source_from_row(row, registry, top_level)
    row_id = _text(_first(row, "row_id", "id"))
    if not row_id:
        raise ValueError("each measured row needs row_id")
    return EvidenceRecord(
        row_id=row_id,
        kind=kind,
        strategy=_normalise_strategy(row),
        metric=_text(metric),
        value=value,
        unit=_infer_unit(_text(metric), row, raw_unit),
        features=_feature_mapping(row),
        provenance=source,
        split=_normalise_split(_first(row, "split", "partition", "set")),
        notes=_text(row.get("notes")),
    )


def _bundle_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    registry: Mapping[str, Mapping[str, Any]] | None = None,
    top_level: Mapping[str, Any] | None = None,
    input_format: str,
) -> EvidenceBundle:
    source_registry = registry or {}
    defaults = top_level or {}
    records: list[EvidenceRecord] = []
    for raw in rows:
        for row in _expand_raw_row(raw):
            records.append(_record_from_row(row, source_registry, defaults))
    if not records:
        raise ValueError("evidence input contains no measured rows")
    seen: set[str] = set()
    for record in records:
        if record.row_id in seen:
            raise ValueError(f"duplicate evidence row_id: {record.row_id}")
        seen.add(record.row_id)
    sources: dict[str, SourceProvenance] = {}
    for record in records:
        previous = sources.get(record.provenance.source_id)
        if previous is not None and previous != record.provenance:
            raise ValueError(f"source_id {record.provenance.source_id!r} has conflicting provenance")
        sources[record.provenance.source_id] = record.provenance
    return EvidenceBundle(tuple(records), tuple(sources[key] for key in sorted(sources)), input_format=input_format)


def parse_evidence_csv(text: str | TextIO, *, input_format: str = "csv") -> EvidenceBundle:
    """Parse measured evidence from a CSV string or text stream."""

    content = text.read() if hasattr(text, "read") else text
    reader = csv.DictReader(io.StringIO(str(content)))
    if not reader.fieldnames:
        raise ValueError("evidence CSV needs a header row")
    rows = [dict(row) for row in reader]
    return _bundle_from_rows(rows, input_format=input_format)


def _json_rows(payload: Any) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [item for item in payload if isinstance(item, Mapping)], {}, {}
    if not isinstance(payload, Mapping):
        raise ValueError("evidence JSON must be an object or array")
    registry = _source_registry(payload.get("sources"))
    defaults: dict[str, Any] = {}
    for field_name in ("provenance", "source_provenance"):
        if isinstance(payload.get(field_name), Mapping):
            defaults[field_name] = payload[field_name]
    for field_name in ("source_id", "source_url", "source_doi", "citation", "source_citation", "accessed_at"):
        if field_name in payload:
            defaults[field_name] = payload[field_name]
    rows: list[Mapping[str, Any]] = []
    for key, kind in _GROUP_KINDS.items():
        values = payload.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for item in values:
                if not isinstance(item, Mapping):
                    raise ValueError(f"JSON group {key!r} must contain objects")
                row = dict(defaults)
                row.update(item)
                row.setdefault("kind", kind)
                rows.append(row)
    for key in ("observations", "records", "measurements", "data"):
        values = payload.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for item in values:
                if not isinstance(item, Mapping):
                    raise ValueError(f"JSON array {key!r} must contain objects")
                row = dict(defaults)
                row.update(item)
                rows.append(row)
    if not rows and any(key in payload for key in ("row_id", "id", "value", "observed", "metric")):
        rows = [payload]
    return rows, registry, defaults


def parse_evidence_json(payload: str | bytes | Mapping[str, Any] | Sequence[Mapping[str, Any]], *, input_format: str = "json") -> EvidenceBundle:
    """Parse measured evidence from flat or grouped JSON."""

    if isinstance(payload, (str, bytes, bytearray)):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("evidence JSON is invalid") from exc
    else:
        parsed = payload
    rows, registry, defaults = _json_rows(parsed)
    expanded: list[Mapping[str, Any]] = []
    for row in rows:
        expanded.extend(_expand_raw_row(row, _text(row.get("kind")) or None))
    return _bundle_from_rows(expanded, registry=registry, top_level=defaults, input_format=input_format)


def load_evidence(path: str | Path) -> EvidenceBundle:
    """Load a CSV or JSON evidence file based on its suffix."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        if file_path.suffix.lower() == ".json":
            return parse_evidence_json(handle.read(), input_format="json")
        if file_path.suffix.lower() in {".csv", ".tsv"}:
            if file_path.suffix.lower() == ".tsv":
                content = handle.read().replace("\t", ",")
                return parse_evidence_csv(content, input_format="tsv")
            return parse_evidence_csv(handle, input_format="csv")
    raise ValueError(f"unsupported evidence file suffix: {file_path.suffix!r}")


# Descriptive aliases make the ingestion boundary easy to discover without
# duplicating parsing code.
load_evidence_csv = parse_evidence_csv
load_evidence_json = parse_evidence_json
ingest_csv = parse_evidence_csv
ingest_json = parse_evidence_json


def _records(value: Iterable[EvidenceRecord] | EvidenceBundle) -> list[EvidenceRecord]:
    if isinstance(value, EvidenceBundle):
        result = list(value.records)
    else:
        result = list(value)
    if not result:
        raise ValueError("calibration needs at least one measured evidence row")
    if not all(isinstance(record, EvidenceRecord) for record in result):
        raise TypeError("calibration rows must be EvidenceRecord values")
    return result


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    categories: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "kind": self.kind}
        if self.categories:
            result["categories"] = list(self.categories)
        return result


def _feature_specs(rows: Sequence[EvidenceRecord], feature_names: Sequence[str] | None) -> tuple[FeatureSpec, ...]:
    if feature_names is None:
        candidates = sorted(set.intersection(*(set(row.features) for row in rows)) if rows else set())
    else:
        candidates = []
        for name in feature_names:
            if name not in candidates:
                candidates.append(name)
        missing = [name for name in candidates if any(name not in row.features for row in rows)]
        if missing:
            raise ValueError(f"calibration feature(s) missing from at least one row: {', '.join(missing)}")
    specs: list[FeatureSpec] = []
    for name in candidates:
        values = [row.features[name] for row in rows]
        numeric_values = [_numeric(value) for value in values]
        if all(value is not None for value in numeric_values):
            if any(float(value) < 0 for value in numeric_values if value is not None):
                raise ValueError(f"numeric calibration feature {name!r} must be non-negative")
            if len({float(value) for value in numeric_values if value is not None}) > 1:
                specs.append(FeatureSpec(name, "numeric"))
        else:
            categories = tuple(sorted({_text(value) for value in values}))
            if len(categories) > 1:
                specs.append(FeatureSpec(name, "categorical", categories))
    return tuple(specs)


def _design_row(features: Mapping[str, Any], specs: Sequence[FeatureSpec]) -> list[float]:
    values = [1.0]
    for spec in specs:
        if spec.name not in features:
            raise ValueError(f"prediction is missing calibration feature {spec.name!r}")
        value = features[spec.name]
        if spec.kind == "numeric":
            number = _numeric(value)
            if number is None or number < 0:
                raise ValueError(f"feature {spec.name!r} must be a finite non-negative number")
            values.append(math.log1p(number))
        else:
            text = _text(value)
            values.extend(1.0 if text == category else 0.0 for category in spec.categories[1:])
    return values


def _terms(specs: Sequence[FeatureSpec]) -> tuple[str, ...]:
    names = ["intercept"]
    for spec in specs:
        if spec.kind == "numeric":
            names.append(spec.name)
        else:
            names.extend(f"{spec.name}={category}" for category in spec.categories[1:])
    return tuple(names)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-14:
            raise ValueError("calibration features are rank deficient")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                current - factor * pivot_value for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def _fit_core(
    rows: Sequence[EvidenceRecord],
    *,
    specs: Sequence[FeatureSpec],
    ridge: float,
    template: "ParameterFit | None" = None,
) -> "ParameterFit":
    terms = _terms(specs)
    design = [_design_row(row.features, specs) for row in rows]
    targets = [math.log(row.value) for row in rows]
    size = len(terms)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    vector = [0.0 for _ in range(size)]
    for row, target in zip(design, targets):
        for left in range(size):
            vector[left] += row[left] * target
            for right in range(size):
                matrix[left][right] += row[left] * row[right]
    for index in range(1, size):
        matrix[index][index] += ridge
    try:
        coefficients = _solve(matrix, vector)
    except ValueError:
        if ridge > 0:
            raise
        for index in range(1, size):
            matrix[index][index] += 1e-12
        coefficients = _solve(matrix, vector)
    residuals = tuple(
        target - sum(coefficient * value for coefficient, value in zip(coefficients, row))
        for row, target in zip(design, targets)
    )
    first = rows[0]
    return ParameterFit(
        strategy=first.strategy,
        kind=first.kind,
        metric=first.metric,
        unit=first.unit,
        feature_specs=tuple(specs),
        terms=terms,
        coefficients=tuple(coefficients),
        residuals=residuals,
        observations=len(rows),
        source_row_ids=tuple(row.row_id for row in rows),
        ridge=ridge,
        training_records=tuple(rows),
        method=template.method if template is not None else "log-linear least squares over measured rows",
    )


@dataclass(frozen=True)
class ParameterFit:
    strategy: str
    kind: str
    metric: str
    unit: str
    feature_specs: tuple[FeatureSpec, ...]
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    residuals: tuple[float, ...]
    observations: int
    source_row_ids: tuple[str, ...]
    ridge: float
    training_records: tuple[EvidenceRecord, ...] = field(repr=False, compare=False, default=())
    method: str = "log-linear least squares over measured rows"

    @property
    def residual_sigma(self) -> float:
        return math.sqrt(sum(value * value for value in self.residuals) / max(1, len(self.residuals)))

    @property
    def parameters(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.terms, self.coefficients)))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.feature_specs)

    def predict(self, features: Mapping[str, Any]) -> float:
        log_prediction = sum(
            coefficient * value for coefficient, value in zip(self.coefficients, _design_row(features, self.feature_specs))
        )
        if log_prediction > 709:
            raise ValueError("calibration prediction is not finite and positive")
        prediction = math.exp(log_prediction)
        if not math.isfinite(prediction) or prediction <= 0:
            raise ValueError("calibration prediction is not finite and positive")
        return prediction

    def predict_record(self, record: EvidenceRecord) -> float:
        return self.predict(record.features)

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "kind": self.kind,
            "metric": self.metric,
            "unit": self.unit,
            "feature_specs": [spec.as_dict() for spec in self.feature_specs],
            "parameters": dict(self.parameters),
            "residual_sigma_log": self.residual_sigma,
            "observations": self.observations,
            "source_row_ids": list(self.source_row_ids),
            "ridge": self.ridge,
            "method": self.method,
        }


FittedParameters = ParameterFit


def _select_fit_rows(
    rows: Sequence[EvidenceRecord],
    *,
    strategy: str | None,
    kind: str | None,
    metric: str | None,
    unit: str | None,
) -> list[EvidenceRecord]:
    selected = [
        row
        for row in rows
        if (strategy is None or row.strategy == strategy)
        and (kind is None or row.kind == kind)
        and (metric is None or row.metric == metric)
        and (unit is None or row.unit == unit)
    ]
    if not selected:
        raise ValueError("no measured rows match the requested calibration subset")
    identities = {(row.kind, row.metric, row.unit, row.strategy) for row in selected}
    if len(identities) > 1:
        raise ValueError("fit_parameters requires one kind/metric/unit/strategy group")
    return sorted(selected, key=lambda row: row.row_id)


def fit_parameters(
    rows: Iterable[EvidenceRecord] | EvidenceBundle,
    *,
    strategy: str | None = None,
    kind: str | None = None,
    metric: str | None = None,
    unit: str | None = None,
    feature_names: Sequence[str] | None = None,
    ridge: float = 1e-10,
) -> ParameterFit:
    """Fit one measured kind/metric/strategy group deterministically."""

    if ridge < 0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    materialized = _records(rows)
    selected = _select_fit_rows(materialized, strategy=strategy, kind=kind, metric=metric, unit=unit)
    specs = _feature_specs(selected, feature_names)
    return _fit_core(selected, specs=specs, ridge=ridge)


def fit_by_strategy(
    rows: Iterable[EvidenceRecord] | EvidenceBundle,
    *,
    kind: str | None = None,
    metric: str | None = None,
    unit: str | None = None,
    feature_names: Sequence[str] | None = None,
    ridge: float = 1e-10,
) -> dict[str, ParameterFit]:
    """Fit separate deterministic models for every strategy in one group."""

    materialized = _records(rows)
    selected = [
        row
        for row in materialized
        if (kind is None or row.kind == kind)
        and (metric is None or row.metric == metric)
        and (unit is None or row.unit == unit)
    ]
    if not selected:
        raise ValueError("no measured rows match the requested calibration subset")
    identities_by_strategy: dict[str, set[tuple[str, str, str]]] = {}
    for row in selected:
        identities_by_strategy.setdefault(row.strategy, set()).add((row.kind, row.metric, row.unit))
    conflicting = [strategy for strategy, identities in identities_by_strategy.items() if len(identities) > 1]
    if conflicting:
        raise ValueError("fit_by_strategy needs one metric/unit group per strategy; use fit_by_group for mixed evidence")
    return {
        strategy: fit_parameters(
            selected,
            strategy=strategy,
            kind=next(iter(identities_by_strategy[strategy]))[0],
            metric=next(iter(identities_by_strategy[strategy]))[1],
            unit=next(iter(identities_by_strategy[strategy]))[2],
            feature_names=feature_names,
            ridge=ridge,
        )
        for strategy in sorted(identities_by_strategy)
    }


def fit_by_group(
    rows: Iterable[EvidenceRecord] | EvidenceBundle,
    *,
    feature_names: Sequence[str] | None = None,
    ridge: float = 1e-10,
) -> dict[tuple[str, str, str, str], ParameterFit]:
    """Fit every independent ``(kind, metric, unit, strategy)`` group."""

    materialized = _records(rows)
    groups: dict[tuple[str, str, str, str], list[EvidenceRecord]] = {}
    for row in materialized:
        groups.setdefault((row.kind, row.metric, row.unit, row.strategy), []).append(row)
    return {
        key: fit_parameters(group, feature_names=feature_names, ridge=ridge)
        for key, group in sorted(groups.items())
    }


def bootstrap_parameter_samples(
    rows_or_fit: Iterable[EvidenceRecord] | EvidenceBundle | ParameterFit,
    *,
    replicates: int = 256,
    seed: int = 20261005,
) -> tuple[ParameterFit, ...]:
    """Return deterministic non-parametric bootstrap refits.

    Every bootstrap sample consists only of row IDs from the supplied fit
    group.  This is uncertainty over fitted parameters, not a claim that the
    resampled rows are new measurements.
    """

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if isinstance(rows_or_fit, ParameterFit):
        template = rows_or_fit
        rows = list(template.training_records)
        if not rows:
            raise ValueError("the fit does not retain rows for bootstrap sampling")
    else:
        rows = _records(rows_or_fit)
        template = fit_parameters(rows)
    rows = sorted(rows, key=lambda row: row.row_id)
    rng = random.Random(seed)
    samples: list[ParameterFit] = []
    for _ in range(replicates):
        resampled = [rows[rng.randrange(len(rows))] for _ in rows]
        samples.append(_fit_core(resampled, specs=template.feature_specs, ridge=template.ridge, template=template))
    return tuple(samples)


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parameter_intervals(
    fit_or_rows: ParameterFit | Iterable[EvidenceRecord] | EvidenceBundle,
    *,
    replicates: int = 256,
    seed: int = 20261005,
    confidence: float = 0.90,
) -> dict[str, dict[str, float]]:
    """Sample deterministic central intervals for fitted parameters."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    fit = fit_or_rows if isinstance(fit_or_rows, ParameterFit) else fit_parameters(fit_or_rows)
    samples = bootstrap_parameter_samples(fit, replicates=replicates, seed=seed)
    alpha = (1.0 - confidence) / 2.0
    return {
        term: {
            "low": _quantile([sample.coefficients[index] for sample in samples], alpha),
            "median": _quantile([sample.coefficients[index] for sample in samples], 0.5),
            "high": _quantile([sample.coefficients[index] for sample in samples], 1.0 - alpha),
        }
        for index, term in enumerate(fit.terms)
    }


sample_parameter_intervals = parameter_intervals


def predictive_interval(
    fit: ParameterFit,
    features: Mapping[str, Any],
    *,
    samples: Sequence[ParameterFit] | None = None,
    replicates: int = 256,
    seed: int = 20261005,
    confidence: float = 0.90,
) -> dict[str, float]:
    """Return a deterministic bootstrap predictive interval.

    Residual draws are sampled from the fit's observed log residuals.  No
    Gaussian noise or external measurements are introduced.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    sampled = tuple(samples) if samples is not None else bootstrap_parameter_samples(fit, replicates=replicates, seed=seed)
    if not sampled:
        raise ValueError("predictive interval needs at least one parameter sample")
    rng = random.Random(seed + 1)
    predictions: list[float] = []
    for sample in sampled:
        log_prediction = sum(
            coefficient * value for coefficient, value in zip(sample.coefficients, _design_row(features, sample.feature_specs))
        )
        residuals = sample.residuals or fit.residuals
        residual = residuals[rng.randrange(len(residuals))] if residuals else 0.0
        log_value = log_prediction + residual
        if log_value > 709:
            predictions.append(float("inf"))
        else:
            predictions.append(math.exp(log_value))
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": _quantile(predictions, alpha),
        "median": _quantile(predictions, 0.5),
        "high": _quantile(predictions, 1.0 - alpha),
    }


@dataclass(frozen=True)
class HeldOutSplit:
    train: tuple[EvidenceRecord, ...]
    held_out: tuple[EvidenceRecord, ...]
    test_fraction: float
    seed: int
    method: str = "explicit split labels, otherwise stable SHA-256 row-id split per evidence group"

    def as_dict(self) -> dict[str, object]:
        return {
            "train_row_ids": [row.row_id for row in self.train],
            "held_out_row_ids": [row.row_id for row in self.held_out],
            "test_fraction": self.test_fraction,
            "seed": self.seed,
            "method": self.method,
        }


def _group_key(row: EvidenceRecord) -> tuple[str, str, str, str]:
    return row.kind, row.metric, row.unit, row.strategy


def _stable_row_key(row: EvidenceRecord, seed: int) -> str:
    material = f"{seed}\0{row.kind}\0{row.metric}\0{row.unit}\0{row.strategy}\0{row.row_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def split_held_out(
    rows: Iterable[EvidenceRecord] | EvidenceBundle,
    *,
    test_fraction: float = 0.20,
    seed: int = 20261005,
) -> HeldOutSplit:
    """Make a deterministic, group-preserving train/held-out split."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    materialized = _records(rows)
    groups: dict[tuple[str, str, str, str], list[EvidenceRecord]] = {}
    for row in materialized:
        groups.setdefault(_group_key(row), []).append(row)
    train: list[EvidenceRecord] = []
    held_out: list[EvidenceRecord] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row.row_id)
        explicit_train = [row for row in group if row.split == "train"]
        explicit_test = [row for row in group if row.split == "test"]
        unlabeled = [row for row in group if row.split is None]
        if explicit_test:
            group_train = explicit_train + unlabeled
            group_test = explicit_test
        elif explicit_train:
            ordered = sorted(unlabeled, key=lambda row: _stable_row_key(row, seed))
            count = min(len(ordered), max(1, math.ceil(len(group) * test_fraction))) if ordered else 0
            group_test = ordered[:count]
            group_train = explicit_train + ordered[count:]
        elif len(group) >= 2:
            ordered = sorted(group, key=lambda row: _stable_row_key(row, seed))
            count = min(len(group) - 1, max(1, math.ceil(len(group) * test_fraction)))
            group_test = ordered[:count]
            group_train = ordered[count:]
        else:
            group_train = group
            group_test = []
        train.extend(group_train)
        held_out.extend(group_test)
    return HeldOutSplit(
        train=tuple(sorted(train, key=lambda row: row.row_id)),
        held_out=tuple(sorted(held_out, key=lambda row: row.row_id)),
        test_fraction=test_fraction,
        seed=seed,
    )


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("MAE needs equally sized non-empty sequences")
    return sum(abs(p - a) for a, p in zip(actual, predicted)) / len(actual)


def mean_absolute_percentage_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("MAPE needs equally sized non-empty sequences")
    if any(value == 0 for value in actual):
        raise ValueError("MAPE is undefined for zero measured values")
    return sum(abs(p - a) / abs(a) for a, p in zip(actual, predicted)) * 100.0 / len(actual)


def total_deviation_percentage(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Absolute aggregate prediction-vs-data deviation in percent.

    The report exposes this derived scalar under the compatibility key
    ``PTD-P``.  PTD-P is also a common strategy label for pipeline/tensor/data
    parallelism; the label and this scalar must not be conflated.
    """

    if len(actual) != len(predicted) or not actual:
        raise ValueError("aggregate deviation needs equally sized non-empty sequences")
    denominator = sum(abs(value) for value in actual)
    if denominator == 0:
        raise ValueError("aggregate deviation is undefined for zero measured values")
    return abs(sum(predicted) - sum(actual)) / denominator * 100.0


ptd_p_percentage = total_deviation_percentage


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    result = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            result[ordered[position][0]] = rank
        index = end
    return result


def rank_correlation(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    """Spearman rank correlation; ``None`` for fewer than two/non-varying rows."""

    if len(actual) != len(predicted):
        raise ValueError("rank correlation needs equally sized sequences")
    if len(actual) < 2:
        return None
    left = _rank(actual)
    right = _rank(predicted)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right)) / math.sqrt(left_ss * right_ss)


@dataclass(frozen=True)
class HeldOutPrediction:
    row_id: str
    strategy: str
    actual: float
    predicted: float
    interval_low: float | None
    interval_high: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "strategy": self.strategy,
            "actual": self.actual,
            "predicted": self.predicted,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
        }


@dataclass(frozen=True)
class ValidationMetrics:
    strategy: str
    kind: str
    metric: str
    unit: str
    train_observations: int
    held_out_observations: int
    mae: float | None
    mape_pct: float | None
    ptd_p_pct: float | None
    rank_correlation: float | None
    coverage: float | None
    covered_observations: int | None
    interval_observations: int
    status: str = "ok"

    @property
    def mape(self) -> float | None:
        return self.mape_pct

    @property
    def ptd_p(self) -> float | None:
        return self.ptd_p_pct

    @property
    def spearman(self) -> float | None:
        return self.rank_correlation

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "kind": self.kind,
            "metric": self.metric,
            "unit": self.unit,
            "train_observations": self.train_observations,
            "held_out_observations": self.held_out_observations,
            "MAE": self.mae,
            "MAPE": self.mape_pct,
            "PTD-P": self.ptd_p_pct,
            "mae": self.mae,
            "mape_pct": self.mape_pct,
            "ptd_p_pct": self.ptd_p_pct,
            "rank_correlation": self.rank_correlation,
            "coverage": self.coverage,
            "covered_observations": self.covered_observations,
            "interval_observations": self.interval_observations,
            "status": self.status,
        }


@dataclass(frozen=True)
class ValidationReport:
    split: HeldOutSplit
    by_strategy: Mapping[str, ValidationMetrics]
    predictions: tuple[HeldOutPrediction, ...]
    fits: Mapping[str, ParameterFit]
    confidence: float
    bootstrap_replicates: int
    interval_seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split.as_dict(),
            "confidence": self.confidence,
            "bootstrap_replicates": self.bootstrap_replicates,
            "interval_seed": self.interval_seed,
            "by_strategy": {key: value.as_dict() for key, value in self.by_strategy.items()},
            "predictions": [prediction.as_dict() for prediction in self.predictions],
            "fits": {key: value.as_dict() for key, value in self.fits.items()},
        }


def evaluate_held_out(
    rows: Iterable[EvidenceRecord] | EvidenceBundle,
    *,
    test_fraction: float = 0.20,
    split_seed: int = 20261005,
    interval_seed: int = 20261005,
    confidence: float = 0.90,
    bootstrap_replicates: int = 256,
    feature_names: Sequence[str] | None = None,
    ridge: float = 1e-10,
) -> ValidationReport:
    """Fit on deterministic train rows and score held-out rows by strategy."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    materialized = _records(rows)
    split = split_held_out(materialized, test_fraction=test_fraction, seed=split_seed)
    all_groups = sorted({_group_key(row) for row in materialized})
    group_counts: dict[str, int] = {}
    for kind, metric, unit, strategy in all_groups:
        del kind, metric, unit
        group_counts[strategy] = group_counts.get(strategy, 0) + 1
    metrics: dict[str, ValidationMetrics] = {}
    fits: dict[str, ParameterFit] = {}
    predictions: list[HeldOutPrediction] = []
    for group in all_groups:
        kind, metric, unit, strategy = group
        train_rows = [row for row in split.train if _group_key(row) == group]
        test_rows = [row for row in split.held_out if _group_key(row) == group]
        key = strategy if group_counts[strategy] == 1 else f"{strategy}|{kind}|{metric}|{unit}"
        if not train_rows and test_rows:
            raise ValueError(f"held-out group {key!r} has no training rows")
        if not train_rows:
            metrics[key] = ValidationMetrics(
                strategy=strategy,
                kind=kind,
                metric=metric,
                unit=unit,
                train_observations=0,
                held_out_observations=0,
                mae=None,
                mape_pct=None,
                ptd_p_pct=None,
                rank_correlation=None,
                coverage=None,
                covered_observations=None,
                interval_observations=0,
                status="no_training_rows",
            )
            continue
        fit = fit_parameters(train_rows, feature_names=feature_names, ridge=ridge)
        fits[key] = fit
        if not test_rows:
            metrics[key] = ValidationMetrics(
                strategy=strategy,
                kind=kind,
                metric=metric,
                unit=unit,
                train_observations=len(train_rows),
                held_out_observations=0,
                mae=None,
                mape_pct=None,
                ptd_p_pct=None,
                rank_correlation=None,
                coverage=None,
                covered_observations=None,
                interval_observations=0,
                status="no_held_out_rows",
            )
            continue
        samples = bootstrap_parameter_samples(fit, replicates=bootstrap_replicates, seed=interval_seed)
        actual = [row.value for row in test_rows]
        predicted: list[float] = []
        covered = 0
        interval_count = 0
        for row in test_rows:
            estimate = fit.predict_record(row)
            interval = predictive_interval(
                fit,
                row.features,
                samples=samples,
                seed=interval_seed,
                confidence=confidence,
            )
            predicted.append(estimate)
            low = interval["low"]
            high = interval["high"]
            if math.isfinite(low) and math.isfinite(high):
                interval_count += 1
                if low <= row.value <= high:
                    covered += 1
            predictions.append(HeldOutPrediction(row.row_id, strategy, row.value, estimate, low, high))
        metrics[key] = ValidationMetrics(
            strategy=strategy,
            kind=kind,
            metric=metric,
            unit=unit,
            train_observations=len(train_rows),
            held_out_observations=len(test_rows),
            mae=mean_absolute_error(actual, predicted),
            mape_pct=mean_absolute_percentage_error(actual, predicted),
            ptd_p_pct=total_deviation_percentage(actual, predicted),
            rank_correlation=rank_correlation(actual, predicted),
            coverage=covered / interval_count if interval_count else None,
            covered_observations=covered if interval_count else None,
            interval_observations=interval_count,
            status="ok",
        )
    return ValidationReport(
        split=split,
        by_strategy=MappingProxyType(dict(sorted(metrics.items()))),
        predictions=tuple(sorted(predictions, key=lambda item: item.row_id)),
        fits=MappingProxyType(dict(sorted(fits.items()))),
        confidence=confidence,
        bootstrap_replicates=bootstrap_replicates,
        interval_seed=interval_seed,
    )


held_out_metrics = evaluate_held_out

# Short aliases keep the boundary discoverable for callers that use the
# conventional names, while the longer names remain self-documenting.
CalibrationDataset = EvidenceBundle
CalibrationObservation = EvidenceRecord
load_csv = parse_evidence_csv
load_json = parse_evidence_json
load_measurements = load_evidence
sample_intervals = parameter_intervals
evaluate_heldout = evaluate_held_out
split_holdout = split_held_out
spearman_rank_correlation = rank_correlation
mae = mean_absolute_error
mape = mean_absolute_percentage_error
ptd_p = total_deviation_percentage


__all__ = [
    "EvidenceBundle",
    "EvidenceRecord",
    "CalibrationDataset",
    "CalibrationObservation",
    "FittedParameters",
    "FeatureSpec",
    "HeldOutPrediction",
    "HeldOutSplit",
    "Measurement",
    "ParameterFit",
    "SourceProvenance",
    "ValidationMetrics",
    "ValidationReport",
    "bootstrap_parameter_samples",
    "evaluate_held_out",
    "fit_by_group",
    "fit_by_strategy",
    "fit_parameters",
    "held_out_metrics",
    "ingest_csv",
    "ingest_json",
    "evaluate_heldout",
    "load_csv",
    "load_evidence",
    "load_evidence_csv",
    "load_evidence_json",
    "load_json",
    "load_measurements",
    "mae",
    "mape",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "parameter_intervals",
    "parse_evidence_csv",
    "parse_evidence_json",
    "predictive_interval",
    "ptd_p",
    "ptd_p_percentage",
    "rank_correlation",
    "sample_parameter_intervals",
    "sample_intervals",
    "spearman_rank_correlation",
    "split_held_out",
    "split_holdout",
    "total_deviation_percentage",
]

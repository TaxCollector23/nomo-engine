"""Safe ingestion boundaries for model, state-dict, topology, and metrics artifacts.

The product accepts real files without silently guessing missing architecture
facts. Hugging Face JSON configs and Nomo graph contracts are immediately
usable by the Python reference. ONNX and state-dict files are inspected with
safe, dependency-aware boundaries: an installed ONNX reader may provide graph
nodes, while binary weights are never unpickled implicitly. Prometheus text is
parsed into timestamped, labelled samples with source provenance.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


_HF_KEYS = {
    "model_type",
    "architectures",
    "hidden_size",
    "d_model",
    "n_embd",
    "num_hidden_layers",
    "n_layer",
    "num_attention_heads",
    "n_head",
    "vocab_size",
}


@dataclass(frozen=True)
class PrometheusSample:
    """One parsed Prometheus exposition sample."""

    metric: str
    labels: Mapping[str, str]
    value: float
    timestamp_ms: int | None = None
    source: str = "prometheus-text"

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "labels": dict(self.labels),
            "value": self.value,
            "timestamp_ms": self.timestamp_ms,
            "source": self.source,
        }


@dataclass(frozen=True)
class ModelArtifact:
    """Inspection result for an uploaded model artifact."""

    format: str
    source: str
    status: str
    config: Mapping[str, Any] | None = None
    graph_nodes: tuple[Mapping[str, Any], ...] = ()
    tensors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "source": self.source,
            "status": self.status,
            "config": dict(self.config) if self.config is not None else None,
            "graph_nodes": [dict(node) for node in self.graph_nodes],
            "tensors": {key: dict(value) for key, value in self.tensors.items()},
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


def _source_bytes(source: str | Path | bytes | bytearray | Mapping[str, Any]) -> tuple[str, str, bytes | None, Mapping[str, Any] | None]:
    if isinstance(source, Mapping):
        return "inline-json", "json", None, dict(source)
    if isinstance(source, (bytes, bytearray)):
        return "inline-bytes", "binary", bytes(source), None
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path), path.suffix.lower().lstrip("."), path.read_bytes(), None


def _provenance(source: str, fmt: str, *, measured: bool = False) -> dict[str, Any]:
    return {
        "source": source,
        "format": fmt,
        "measured": measured,
        "classification": "uploaded artifact metadata",
    }


def _looks_like_hf_config(payload: Mapping[str, Any]) -> bool:
    return bool(_HF_KEYS.intersection(payload)) and not isinstance(payload.get("nodes"), list)


def _parse_safetensors(raw: bytes, source: str) -> ModelArtifact:
    if len(raw) < 8:
        raise ValueError("safetensors header is truncated")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length <= 2 or header_length > len(raw) - 8:
        raise ValueError("safetensors header length is invalid")
    try:
        header = json.loads(raw[8 : 8 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("safetensors header is not valid JSON") from exc
    if not isinstance(header, Mapping):
        raise ValueError("safetensors header must be an object")
    tensors: dict[str, Mapping[str, Any]] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, Mapping):
            raise ValueError(f"safetensors tensor {name!r} has invalid metadata")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not isinstance(offsets, Sequence):
            raise ValueError(f"safetensors tensor {name!r} is missing shape/data_offsets")
        tensors[str(name)] = {
            "dtype": str(metadata.get("dtype", "unknown")),
            "shape": [int(value) for value in shape],
            "data_offsets": [int(value) for value in offsets],
        }
    return ModelArtifact(
        format="safetensors",
        source=source,
        status="metadata-only",
        tensors=tensors,
        warnings=(
            "Weights were not loaded into memory; architecture is not inferred from tensor names.",
            "Supply a Hugging Face config.json alongside the state dict for operator-graph construction.",
        ),
        provenance=_provenance(source, "safetensors"),
    )


def load_model_artifact(source: str | Path | bytes | bytearray | Mapping[str, Any]) -> ModelArtifact:
    """Inspect a model artifact without unsafe deserialization or guessing."""

    source_name, suffix, raw, inline = _source_bytes(source)
    if inline is not None:
        payload: Any = inline
        fmt = "json"
    else:
        assert raw is not None
        fmt = suffix
        if suffix in {"json", "jsonl"}:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("JSON model artifact is invalid") from exc
        elif suffix == "safetensors":
            return _parse_safetensors(raw, source_name)
        elif suffix in {"onnx"}:
            try:
                import onnx  # type: ignore[import-not-found]
            except ImportError:
                return ModelArtifact(
                    format="onnx",
                    source=source_name,
                    status="preview",
                    warnings=("Install the optional 'onnx' package to inspect graph nodes.",),
                    provenance=_provenance(source_name, "onnx"),
                )
            model = onnx.load_model_from_string(raw)
            nodes = tuple(
                {"op_type": node.op_type, "name": node.name, "inputs": list(node.input), "outputs": list(node.output)}
                for node in model.graph.node
            )
            return ModelArtifact(
                format="onnx",
                source=source_name,
                status="graph-inspected",
                graph_nodes=nodes,
                warnings=("ONNX nodes are inspected; framework-specific shape lowering still needs an adapter.",),
                provenance=_provenance(source_name, "onnx"),
            )
        elif suffix in {"pt", "pth", "bin", "ckpt"}:
            return ModelArtifact(
                format="state-dict",
                source=source_name,
                status="preview",
                warnings=(
                    "Pickle-backed weights are not deserialized automatically because loading them can execute code.",
                    "Export a safetensors file or provide a trusted framework adapter explicitly.",
                ),
                provenance=_provenance(source_name, "state-dict"),
            )
        else:
            raise ValueError(f"unsupported model artifact suffix: .{suffix}")

    if not isinstance(payload, Mapping):
        raise ValueError("JSON model artifact must be an object")
    if _looks_like_hf_config(payload):
        return ModelArtifact(
            format="huggingface-config",
            source=source_name,
            status="ready",
            config=dict(payload),
            warnings=("Operator accounting uses the fields present in config.json; missing fields are not invented.",),
            provenance=_provenance(source_name, "huggingface-config"),
        )
    if isinstance(payload.get("nodes"), list) or isinstance(payload.get("graph"), Mapping):
        nodes = payload.get("nodes")
        if isinstance(payload.get("graph"), Mapping):
            nodes = payload["graph"].get("nodes")
        return ModelArtifact(
            format="nomo-graph-json",
            source=source_name,
            status="ready",
            config=dict(payload),
            graph_nodes=tuple(node for node in (nodes or ()) if isinstance(node, Mapping)),
            warnings=("Imported graph accounting is used as supplied; omitted architecture metadata is not inferred.",),
            provenance=_provenance(source_name, "nomo-graph-json"),
        )
    raise ValueError("JSON artifact is neither a Hugging Face config nor a Nomo graph contract")


_PROMETHEUS_LINE = re.compile(
    r"^(?P<metric>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+(?P<timestamp>-?[0-9]+))?\s*$"
)
_PROMETHEUS_LABEL = re.compile(r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\"(?P<value>(?:\\.|[^\"])*)\"")


def parse_prometheus_text(text: str, *, source: str = "prometheus-text") -> tuple[PrometheusSample, ...]:
    """Parse Prometheus/OpenMetrics sample lines and ignore comments."""

    samples: list[PrometheusSample] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_LINE.match(line)
        if not match:
            raise ValueError(f"invalid Prometheus sample at line {line_number}")
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise ValueError(f"invalid Prometheus value at line {line_number}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite Prometheus value at line {line_number}")
        labels: dict[str, str] = {}
        label_text = match.group("labels") or ""
        consumed = ""
        for label in _PROMETHEUS_LABEL.finditer(label_text):
            labels[label.group("name")] = json.loads('"' + label.group("value") + '"')
            consumed += label.group(0)
        if label_text.replace(",", "").replace(" ", "") != consumed.replace(",", "").replace(" ", ""):
            raise ValueError(f"invalid Prometheus label set at line {line_number}")
        timestamp = match.group("timestamp")
        samples.append(
            PrometheusSample(
                metric=match.group("metric"),
                labels=labels,
                value=value,
                timestamp_ms=None if timestamp is None else int(timestamp),
                source=source,
            )
        )
    if not samples:
        raise ValueError("Prometheus input contains no samples")
    return tuple(samples)


def load_prometheus_text(path: str | Path) -> tuple[PrometheusSample, ...]:
    """Read and parse a Prometheus text exposition file."""

    file_path = Path(path)
    return parse_prometheus_text(file_path.read_text(encoding="utf-8"), source=str(file_path))


__all__ = [
    "ModelArtifact",
    "PrometheusSample",
    "load_model_artifact",
    "load_prometheus_text",
    "parse_prometheus_text",
]

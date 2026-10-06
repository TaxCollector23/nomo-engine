"""Safe ingestion boundaries for model, state-dict, topology, and metrics artifacts.

The product accepts real files without silently guessing missing architecture
facts. Hugging Face JSON configs and Nomo graph contracts are validated at a
bounded structural boundary. Complete Hugging Face configs are lowered to a
canonical transformer skeleton using only fields present in the config; this
does not invent costs, timings, tensor shapes, or measurements. ONNX and
state-dict files are inspected with safe, dependency-aware boundaries: an
installed ONNX reader may validate graph structure, while binary weights are
never unpickled implicitly. Prometheus text is parsed into timestamped,
labelled samples with source provenance.
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

_HF_ALIASES = {
    "layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "hidden_size": ("hidden_size", "d_model", "n_embd"),
    "attention_heads": ("num_attention_heads", "n_head", "num_heads"),
    "vocab_size": ("vocab_size", "n_vocab"),
    "kv_heads": ("num_key_value_heads", "n_kv_heads"),
    "intermediate_size": ("intermediate_size", "n_inner"),
    "experts": ("num_local_experts", "num_experts"),
    "experts_per_token": ("num_experts_per_tok", "num_experts_per_token", "moe_top_k"),
}

_MAX_GRAPH_NODES = 8192


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
    validation: Mapping[str, Any] = field(default_factory=dict)
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
            "validation": dict(self.validation),
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


def _first_present(payload: Mapping[str, Any], *keys: str) -> tuple[str | None, Any]:
    for key in keys:
        if key in payload and payload[key] is not None:
            return key, payload[key]
    return None, None


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _validate_hf_config(payload: Mapping[str, Any]) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...]]:
    """Validate only architecture facts needed for structural lowering.

    The returned warnings describe fields that are optional or omitted. They
    are deliberately not filled with estimates; the caller can still expose
    an incomplete config as a preview artifact.
    """

    canonical: dict[str, int] = {}
    missing: list[str] = []
    for name, aliases in _HF_ALIASES.items():
        key, value = _first_present(payload, *aliases)
        if value is None:
            if name in {"layers", "hidden_size", "attention_heads", "vocab_size"}:
                missing.append(name)
            continue
        canonical[name] = _positive_int(value, key or name)

    if "hidden_size" in canonical and "attention_heads" in canonical:
        if canonical["hidden_size"] % canonical["attention_heads"]:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
    if "kv_heads" in canonical and "attention_heads" in canonical:
        if canonical["attention_heads"] % canonical["kv_heads"]:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
    if "experts_per_token" in canonical and "experts" in canonical:
        if canonical["experts_per_token"] > canonical["experts"]:
            raise ValueError("num_experts_per_tok cannot exceed num_local_experts")
    if ("experts" in canonical) != ("experts_per_token" in canonical):
        raise ValueError("MoE config must provide both expert count and experts per token")

    warnings = tuple(
        [f"Missing structural fields: {', '.join(missing)}"] if missing else []
    )
    return canonical, tuple(missing), warnings


def _lower_hf_config(payload: Mapping[str, Any], canonical: Mapping[str, int]) -> tuple[Mapping[str, Any], ...]:
    """Lower a complete HF config to a bounded, cost-free transformer skeleton."""

    layer_count = canonical["layers"]
    if layer_count * 2 + 2 > _MAX_GRAPH_NODES:
        raise ValueError(
            f"Hugging Face config would lower to more than {_MAX_GRAPH_NODES} graph nodes"
        )

    model_type = str(payload.get("model_type", "unknown"))
    hidden_size = canonical["hidden_size"]
    attention_heads = canonical["attention_heads"]
    vocab_size = canonical["vocab_size"]
    gated = str(payload.get("hidden_act", "")).lower() in {"silu", "swish", "geglu"} or bool(payload.get("gated_mlp", False))

    nodes: list[Mapping[str, Any]] = [{
        "id": "embedding",
        "kind": "embedding",
        "operator": "Embedding",
        "layer_index": None,
        "inputs": [],
        "outputs": ["embedding.output"],
        "metadata": {
            "model_type": model_type,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
        },
    }]
    previous_output = "embedding.output"
    for index in range(layer_count):
        attention_id = f"block.{index}.attention"
        attention_output = f"block.{index}.attention.output"
        nodes.append({
            "id": attention_id,
            "kind": "attention",
            "operator": "SelfAttention",
            "layer_index": index,
            "inputs": [previous_output],
            "outputs": [attention_output],
            "metadata": {
                "hidden_size": hidden_size,
                "attention_heads": attention_heads,
                **({"kv_heads": canonical["kv_heads"]} if "kv_heads" in canonical else {}),
            },
        })
        mlp_id = f"block.{index}.mlp"
        mlp_output = f"block.{index}.mlp.output"
        mlp_metadata: dict[str, Any] = {"gated": gated}
        if "intermediate_size" in canonical:
            mlp_metadata["intermediate_size"] = canonical["intermediate_size"]
        if "experts" in canonical:
            mlp_metadata["experts"] = canonical["experts"]
            mlp_metadata["experts_per_token"] = canonical["experts_per_token"]
        nodes.append({
            "id": mlp_id,
            "kind": "mlp",
            "operator": "MLP",
            "layer_index": index,
            "inputs": [attention_output],
            "outputs": [mlp_output],
            "metadata": mlp_metadata,
        })
        previous_output = mlp_output
    nodes.append({
        "id": "output",
        "kind": "output",
        "operator": "LMHead",
        "layer_index": None,
        "inputs": [previous_output],
        "outputs": ["logits"],
        "metadata": {"vocab_size": vocab_size, "tied_embeddings": bool(payload.get("tie_word_embeddings", payload.get("tie_embeddings", False)))},
    })
    return tuple(nodes)


def _validate_nomo_graph_nodes(nodes: Any) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Nomo graph JSON must contain a non-empty nodes list")
    if len(nodes) > _MAX_GRAPH_NODES:
        raise ValueError(f"Nomo graph JSON exceeds the {_MAX_GRAPH_NODES}-node ingestion limit")
    normalized: list[Mapping[str, Any]] = []
    ids: set[str] = set()
    for index, node in enumerate(nodes, start=1):
        if not isinstance(node, Mapping):
            raise ValueError(f"Nomo graph node {index} is not an object")
        node_id = str(node.get("id", ""))
        kind = str(node.get("kind", node.get("type", "")))
        if not node_id:
            raise ValueError(f"Nomo graph node {index} is missing id")
        if node_id in ids:
            raise ValueError(f"Nomo graph contains duplicate node id: {node_id}")
        if not kind:
            raise ValueError(f"Nomo graph node {index} is missing kind")
        for field_name in ("inputs", "outputs"):
            if field_name in node and (not isinstance(node[field_name], list) or not all(isinstance(item, str) for item in node[field_name])):
                raise ValueError(f"Nomo graph node {index} has invalid {field_name}")
        ids.add(node_id)
        normalized.append(dict(node))
    return tuple(normalized), {
        "kind": "structural",
        "valid": True,
        "lowering": "pass-through-contract",
        "node_count": len(normalized),
        "checks": ["non-empty nodes", "unique ids", "string inputs/outputs when supplied"],
    }


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
        validation={
            "kind": "header",
            "valid": True,
            "tensor_count": len(tensors),
            "checks": ["valid header length", "valid tensor metadata", "non-negative tensor shapes are not inferred"],
        },
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
                    validation={"kind": "onnx", "valid": None, "lowering": "not-run"},
                    warnings=("Install the optional 'onnx' package to inspect graph nodes.",),
                    provenance=_provenance(source_name, "onnx"),
                )
            model = onnx.load_model_from_string(raw)
            try:
                onnx.checker.check_model(model)
            except Exception as exc:  # onnx exposes version-specific checker errors
                raise ValueError(f"ONNX graph validation failed: {exc}") from exc
            if len(model.graph.node) > _MAX_GRAPH_NODES:
                return ModelArtifact(
                    format="onnx",
                    source=source_name,
                    status="preview",
                    validation={
                        "kind": "onnx-checker",
                        "valid": True,
                        "lowering": "bounded-export-skipped",
                        "node_count": len(model.graph.node),
                    },
                    warnings=(
                        f"ONNX graph passed structural validation but exceeds the {_MAX_GRAPH_NODES}-node export limit; nodes were not copied.",
                    ),
                    provenance=_provenance(source_name, "onnx"),
                )
            nodes = tuple(
                {
                    "id": node.name or f"{node.op_type}.{index}",
                    "kind": node.op_type,
                    "op_type": node.op_type,
                    "name": node.name,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                }
                for index, node in enumerate(model.graph.node)
            )
            return ModelArtifact(
                format="onnx",
                source=source_name,
                status="graph-inspected",
                graph_nodes=nodes,
                validation={
                    "kind": "onnx-checker",
                    "valid": True,
                    "lowering": "node-metadata-only",
                    "node_count": len(nodes),
                    "checks": ["onnx.checker.check_model"],
                },
                warnings=(
                    "ONNX nodes were structurally validated; framework-specific lowering and performance accounting still need an adapter.",
                ),
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
        canonical, missing, validation_warnings = _validate_hf_config(payload)
        if missing:
            return ModelArtifact(
                format="huggingface-config",
                source=source_name,
                status="preview",
                config=dict(payload),
                validation={
                    "kind": "structural",
                    "valid": False,
                    "lowering": "not-run",
                    "missing": list(missing),
                },
                warnings=validation_warnings + (
                    "Provide the missing architecture fields before config-only graph lowering; no values were inferred.",
                ),
                provenance=_provenance(source_name, "huggingface-config"),
            )
        graph_nodes = _lower_hf_config(payload, canonical)
        return ModelArtifact(
            format="huggingface-config",
            source=source_name,
            status="ready",
            config=dict(payload),
            graph_nodes=graph_nodes,
            validation={
                "kind": "structural",
                "valid": True,
                "lowering": "transformer-skeleton-v1",
                "node_count": len(graph_nodes),
                "checks": [
                    "required architecture fields present",
                    "positive integer dimensions",
                    "hidden_size divisible by attention_heads",
                    "bounded node count",
                ],
            },
            warnings=(
                "Graph lowering is a config-only transformer skeleton; costs, timings, tensor shapes, and measurements are not inferred.",
                "Optional fields absent from config.json remain absent from the lowered metadata.",
            ),
            provenance={
                **_provenance(source_name, "huggingface-config"),
                "graph_lowering": "structural-config-only",
            },
        )
    if isinstance(payload.get("nodes"), list) or isinstance(payload.get("graph"), Mapping):
        nodes = payload.get("nodes")
        if isinstance(payload.get("graph"), Mapping):
            nodes = payload["graph"].get("nodes")
        normalized_nodes, validation = _validate_nomo_graph_nodes(nodes)
        return ModelArtifact(
            format="nomo-graph-json",
            source=source_name,
            status="ready",
            config=dict(payload),
            graph_nodes=normalized_nodes,
            validation=validation,
            warnings=(
                "Imported graph accounting is used as supplied; omitted architecture metadata is not inferred.",
                "Structural validation does not establish hardware cost or runtime fidelity.",
            ),
            provenance={
                **_provenance(source_name, "nomo-graph-json"),
                "graph_lowering": "pass-through-contract",
            },
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

"""Safe ingestion boundaries for model, state-dict, topology, and metrics artifacts.

The product accepts real files without silently guessing missing architecture
facts. Hugging Face JSON configs and Nomo graph contracts are validated at a
bounded structural boundary. Complete Hugging Face configs are lowered to a
canonical transformer skeleton using only fields present in the config; this
does not invent costs, timings, tensor shapes, or measurements. ONNX and
state-dict files are inspected with safe, dependency-aware boundaries: an
installed ONNX reader may validate and structurally lower graph boundaries,
operators, attributes, and explicit tensor metadata, while binary weights are
never unpickled implicitly. Prometheus text is parsed into timestamped,
labelled samples with source provenance.
"""

from __future__ import annotations

import base64
import binascii
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
_MAX_BINARY_ARTIFACT_BYTES = 10 * 1024 * 1024
_MAX_BASE64_ARTIFACT_CHARS = ((_MAX_BINARY_ARTIFACT_BYTES + 2) // 3) * 4 + 4


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
        if "base64" in source or source.get("encoding") == "base64":
            filename = source.get("filename")
            encoded = source.get("base64")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError("binary artifact envelope requires a filename")
            if source.get("encoding", "base64") != "base64":
                raise ValueError("binary artifact envelope encoding must be base64")
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("binary artifact envelope requires a base64 string")
            if len(encoded) > _MAX_BASE64_ARTIFACT_CHARS:
                raise ValueError("binary artifact envelope exceeds the 10 MB limit")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("binary artifact envelope contains invalid base64") from exc
            if len(decoded) > _MAX_BINARY_ARTIFACT_BYTES:
                raise ValueError("binary artifact envelope exceeds the 10 MB limit")
            safe_name = Path(filename).name
            suffix = Path(safe_name).suffix.lower().lstrip(".")
            if not suffix:
                raise ValueError("binary artifact envelope filename must have a supported suffix")
            return f"inline-binary:{safe_name}", suffix, decoded, None
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


def _onnx_sequence(value: Any) -> tuple[Any, ...]:
    """Read protobuf repeated fields without depending on protobuf classes."""

    if value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _onnx_string(value: Any) -> str | Mapping[str, str]:
    """Keep ONNX string attributes JSON-safe without silently decoding bytes."""

    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return {"encoding": "base64", "data": base64.b64encode(bytes(value)).decode("ascii")}
    return str(value)


def _onnx_value_metadata(value_info: Any) -> dict[str, Any]:
    """Extract only type/shape facts explicitly present in a ValueInfoProto."""

    metadata: dict[str, Any] = {}
    type_proto = getattr(value_info, "type", None)
    which_oneof = getattr(type_proto, "WhichOneof", None)
    if callable(which_oneof):
        try:
            active_type = which_oneof("value")
        except (KeyError, ValueError):
            active_type = None
        if active_type not in {None, "tensor_type"}:
            return metadata
    tensor_type = getattr(type_proto, "tensor_type", None)
    if tensor_type is None:
        return metadata

    elem_type = getattr(tensor_type, "elem_type", None)
    if isinstance(elem_type, int) and elem_type > 0:
        metadata["data_type"] = elem_type

    shape_proto = getattr(tensor_type, "shape", None)
    if shape_proto is not None and hasattr(shape_proto, "dim"):
        shape: list[int | str | None] = []
        for dimension in _onnx_sequence(getattr(shape_proto, "dim", ())):
            parameter = getattr(dimension, "dim_param", "")
            if parameter:
                shape.append(str(parameter))
                continue
            value = getattr(dimension, "dim_value", None)
            try:
                integer = int(value) if value is not None else 0
            except (TypeError, ValueError):
                integer = 0
            # None records an unknown dimension position; it is never a guessed size.
            shape.append(integer if integer > 0 else None)
        metadata["shape"] = shape
    return metadata


def _onnx_tensor_name(value: Any) -> str:
    return str(getattr(value, "name", "") or "")


def _onnx_tensor_metadata(value: Any, *, initializer: bool = False) -> dict[str, Any]:
    metadata = _onnx_value_metadata(value)
    if initializer:
        data_type = getattr(value, "data_type", None)
        if isinstance(data_type, int) and data_type > 0:
            metadata["data_type"] = data_type
        if hasattr(value, "dims"):
            dimensions: list[int] = []
            for dimension in _onnx_sequence(getattr(value, "dims", ())):
                try:
                    dimensions.append(int(dimension))
                except (TypeError, ValueError) as exc:
                    raise ValueError("ONNX initializer contains a non-integer dimension") from exc
            metadata["shape"] = dimensions
        metadata["initializer"] = True
    return metadata


def _merge_onnx_tensor_metadata(target: dict[str, dict[str, Any]], name: str, metadata: Mapping[str, Any]) -> None:
    if not name or not metadata:
        return
    existing = target.setdefault(name, {})
    for key, value in metadata.items():
        # The checker has already validated the graph. Preserve the first
        # explicit value rather than resolving a disagreement by guessing.
        existing.setdefault(key, value)


def _onnx_tensor_references(names: Sequence[str], tensor_metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for name in names:
        reference: dict[str, Any] = {"name": name}
        metadata = tensor_metadata.get(name)
        if metadata:
            reference["metadata"] = dict(metadata)
        references.append(reference)
    return references


def _onnx_attribute_metadata(attribute: Any) -> dict[str, Any]:
    """Serialize scalar/list ONNX attributes while omitting tensor payload bytes."""

    result: dict[str, Any] = {"name": str(getattr(attribute, "name", "") or "")}
    attribute_type = getattr(attribute, "type", None)
    if isinstance(attribute_type, int):
        result["type"] = attribute_type

    scalar_fields = {1: "f", 2: "i", 3: "s"}
    repeated_fields = {6: "floats", 7: "ints", 8: "strings"}
    if attribute_type in scalar_fields:
        field_name = scalar_fields[attribute_type]
        value = getattr(attribute, field_name, None)
        if field_name == "s":
            result["value"] = _onnx_string(value or b"")
        elif value is not None:
            result["value"] = value
    elif attribute_type in repeated_fields:
        field_name = repeated_fields[attribute_type]
        values = _onnx_sequence(getattr(attribute, field_name, ()))
        result["value"] = [_onnx_string(value) if field_name == "strings" else value for value in values]
    elif attribute_type in {4, 9, 11, 12}:
        tensor = getattr(attribute, "t", None)
        if tensor is not None:
            result["tensor"] = _onnx_tensor_metadata(tensor, initializer=True)
        if attribute_type in {9, 12}:
            result["tensor_count"] = len(_onnx_sequence(getattr(attribute, "tensors", ())))
    elif attribute_type in {5, 10}:
        graph = getattr(attribute, "g", None)
        if graph is not None and hasattr(graph, "node"):
            result["graph_node_count"] = len(_onnx_sequence(getattr(graph, "node", ())))
        if attribute_type == 10:
            result["graph_count"] = len(_onnx_sequence(getattr(attribute, "graphs", ())))
    return result


def _unique_onnx_id(preferred: str, used: set[str]) -> str:
    candidate = preferred or "node"
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 1
    while f"{candidate}.{index}" in used:
        index += 1
    result = f"{candidate}.{index}"
    used.add(result)
    return result


def _lower_onnx_graph(graph: Any) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any], tuple[str, ...]]:
    """Lower an ONNX GraphProto to a cost-free structural graph contract.

    This deliberately copies only facts represented by ONNX protobuf fields:
    graph boundaries, initializer metadata, operator connectivity, attributes,
    and explicit tensor type/shape metadata. It does not calculate FLOPs,
    bytes, memory, latency, or hardware placement.
    """

    graph_inputs = _onnx_sequence(getattr(graph, "input", ()))
    graph_outputs = _onnx_sequence(getattr(graph, "output", ()))
    initializers = _onnx_sequence(getattr(graph, "initializer", ()))
    value_infos = _onnx_sequence(getattr(graph, "value_info", ()))
    operator_nodes = _onnx_sequence(getattr(graph, "node", ()))
    initializer_names = {_onnx_tensor_name(value) for value in initializers if _onnx_tensor_name(value)}

    tensor_metadata: dict[str, dict[str, Any]] = {}
    for value in (*graph_inputs, *graph_outputs, *value_infos):
        _merge_onnx_tensor_metadata(tensor_metadata, _onnx_tensor_name(value), _onnx_value_metadata(value))
    for initializer in initializers:
        _merge_onnx_tensor_metadata(
            tensor_metadata,
            _onnx_tensor_name(initializer),
            _onnx_tensor_metadata(initializer, initializer=True),
        )

    lowered: list[Mapping[str, Any]] = []
    used_ids: set[str] = set()

    for value in graph_inputs:
        name = _onnx_tensor_name(value)
        if not name or name in initializer_names:
            continue
        metadata: dict[str, Any] = {"source": "graph.input"}
        if tensor_metadata.get(name):
            metadata["tensor"] = dict(tensor_metadata[name])
        lowered.append({
            "id": _unique_onnx_id(f"input.{name}", used_ids),
            "kind": "input",
            "operator": "GraphInput",
            "inputs": [],
            "outputs": [name],
            "metadata": metadata,
        })

    for initializer in initializers:
        name = _onnx_tensor_name(initializer)
        if not name:
            continue
        metadata = {"source": "graph.initializer", "tensor": dict(tensor_metadata.get(name, {}))}
        lowered.append({
            "id": _unique_onnx_id(f"initializer.{name}", used_ids),
            "kind": "parameter",
            "operator": "Initializer",
            "inputs": [],
            "outputs": [name],
            "metadata": metadata,
        })

    operator_types: set[str] = set()
    attribute_count = 0
    for index, node in enumerate(operator_nodes):
        op_type = str(getattr(node, "op_type", "Unknown") or "Unknown")
        name = str(getattr(node, "name", "") or "")
        domain = str(getattr(node, "domain", "") or "")
        inputs = [str(value) for value in _onnx_sequence(getattr(node, "input", ()))]
        outputs = [str(value) for value in _onnx_sequence(getattr(node, "output", ()))]
        attributes = [
            _onnx_attribute_metadata(attribute)
            for attribute in _onnx_sequence(getattr(node, "attribute", ()))
        ]
        operator_types.add(op_type)
        attribute_count += len(attributes)
        metadata: dict[str, Any] = {
            "onnx_index": index,
            "input_tensors": _onnx_tensor_references(inputs, tensor_metadata),
            "output_tensors": _onnx_tensor_references(outputs, tensor_metadata),
        }
        if attributes:
            metadata["attributes"] = attributes
        lowered_node: dict[str, Any] = {
            "id": _unique_onnx_id(name or f"{op_type}.{index}", used_ids),
            "kind": op_type,
            "operator": op_type,
            "op_type": op_type,
            "name": name,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata,
        }
        if domain:
            lowered_node["domain"] = domain
            metadata["domain"] = domain
        lowered.append(lowered_node)

    for value in graph_outputs:
        name = _onnx_tensor_name(value)
        if not name:
            continue
        metadata = {"source": "graph.output"}
        if tensor_metadata.get(name):
            metadata["tensor"] = dict(tensor_metadata[name])
        lowered.append({
            "id": _unique_onnx_id(f"output.{name}", used_ids),
            "kind": "output",
            "operator": "GraphOutput",
            "inputs": [name],
            "outputs": [],
            "metadata": metadata,
        })

    source_tensors = {
        _onnx_tensor_name(value)
        for value in (*graph_inputs, *initializers)
        if _onnx_tensor_name(value)
    }
    produced_tensors = {
        output
        for node in operator_nodes
        for output in (str(value) for value in _onnx_sequence(getattr(node, "output", ())))
        if output
    }
    unresolved_inputs = sorted({
        input_name
        for node in operator_nodes
        for input_name in (str(value) for value in _onnx_sequence(getattr(node, "input", ())))
        if input_name and input_name not in source_tensors and input_name not in produced_tensors
    })
    unresolved_outputs = sorted({
        _onnx_tensor_name(value)
        for value in graph_outputs
        if _onnx_tensor_name(value) and _onnx_tensor_name(value) not in source_tensors | produced_tensors
    })
    structurally_valid = not unresolved_inputs and not unresolved_outputs
    checks = [
        "onnx.checker.check_model",
        "graph inputs, initializers, operators, and outputs lowered in source order",
        "explicit tensor type/shape metadata copied without derived values",
        "unique lowered node ids",
        "operator connectivity references checked",
    ]
    validation: dict[str, Any] = {
        "kind": "onnx-checker",
        "valid": structurally_valid,
        "lowering": "structural-graph-v1",
        "node_count": len(operator_nodes),
        "lowered_node_count": len(lowered),
        "graph_input_count": len(graph_inputs),
        "graph_output_count": len(graph_outputs),
        "initializer_count": len(initializers),
        "value_info_count": len(value_infos),
        "tensor_metadata_count": len(tensor_metadata),
        "operator_types": sorted(operator_types),
        "attribute_count": attribute_count,
        "checks": checks,
    }
    warnings = (
        "ONNX nodes, graph boundaries, attributes, and explicit tensor metadata were structurally lowered; no FLOPs, bytes, timings, or measurements were inferred.",
        "Framework-specific operator semantics, tensor byte accounting, and hardware performance still require an adapter and measured inputs.",
    )
    if unresolved_inputs or unresolved_outputs:
        validation["unresolved_inputs"] = unresolved_inputs
        validation["unresolved_outputs"] = unresolved_outputs
        warnings += ("The checker-backed graph contains unresolved tensor references; the result remains Preview.",)
    return tuple(lowered), validation, warnings


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
            nodes, validation, lowering_warnings = _lower_onnx_graph(model.graph)
            return ModelArtifact(
                format="onnx",
                source=source_name,
                status="graph-inspected" if validation["valid"] else "preview",
                graph_nodes=nodes,
                validation=validation,
                warnings=lowering_warnings,
                provenance={
                    **_provenance(source_name, "onnx"),
                    "graph_lowering": "onnx-structural-graph-v1",
                },
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

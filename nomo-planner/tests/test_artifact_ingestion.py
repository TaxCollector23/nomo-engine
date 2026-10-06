import json
import sys
import types

import pytest

from nomo_planner.artifact_ingestion import (
    load_model_artifact,
    parse_prometheus_text,
)


def test_huggingface_config_is_a_ready_real_artifact_boundary():
    artifact = load_model_artifact(
        {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }
    )
    assert artifact.format == "huggingface-config"
    assert artifact.status == "ready"
    assert artifact.config["model_type"] == "llama"
    assert artifact.provenance["measured"] is False


def test_complete_huggingface_config_lowers_to_bounded_structural_nodes_only():
    artifact = load_model_artifact({
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "vocab_size": 32,
    })

    assert [node["id"] for node in artifact.graph_nodes] == [
        "embedding", "block.0.attention", "block.0.mlp",
        "block.1.attention", "block.1.mlp", "output",
    ]
    assert artifact.validation["lowering"] == "transformer-skeleton-v1"
    assert artifact.validation["valid"] is True
    assert all("flops" not in node and "bytes" not in node for node in artifact.graph_nodes)
    assert artifact.provenance["graph_lowering"] == "structural-config-only"


def test_incomplete_huggingface_config_stays_preview_without_inferred_fields():
    artifact = load_model_artifact({"model_type": "llama", "hidden_size": 8})

    assert artifact.status == "preview"
    assert artifact.graph_nodes == ()
    assert artifact.validation["valid"] is False
    assert set(artifact.validation["missing"]) == {"layers", "attention_heads", "vocab_size"}
    assert any("no values were inferred" in warning for warning in artifact.warnings)


def test_huggingface_structural_validation_rejects_incompatible_heads():
    with pytest.raises(ValueError, match="divisible"):
        load_model_artifact({
            "model_type": "llama",
            "hidden_size": 10,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "vocab_size": 32,
        })


def test_nomo_graph_json_validates_ids_and_contract_fields_without_rewriting_the_contract():
    artifact = load_model_artifact({
        "format": "nomo.graph/1",
        "nodes": [
            {"id": "input", "kind": "embedding", "outputs": ["x"]},
            {"id": "output", "kind": "custom", "inputs": ["x"]},
        ],
    })

    assert artifact.validation["lowering"] == "pass-through-contract"
    assert artifact.graph_nodes[1]["kind"] == "custom"

    with pytest.raises(ValueError, match="duplicate node id"):
        load_model_artifact({"nodes": [{"id": "same", "kind": "a"}, {"id": "same", "kind": "b"}]})


def test_onnx_without_optional_reader_is_an_explicit_preview_boundary(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx bytes are not parsed without the optional reader")
    monkeypatch.setitem(sys.modules, "onnx", None)

    artifact = load_model_artifact(path)

    assert artifact.format == "onnx"
    assert artifact.status == "preview"
    assert artifact.validation["valid"] is None
    assert artifact.graph_nodes == ()


def test_onnx_reader_path_runs_structural_checker_before_exposing_nodes(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"safe test fixture")
    calls = []

    node = types.SimpleNamespace(op_type="Add", name="add", input=["x", "y"], output=["z"])
    model = types.SimpleNamespace(graph=types.SimpleNamespace(node=[node]))
    fake_onnx = types.ModuleType("onnx")
    fake_onnx.load_model_from_string = lambda raw: model
    fake_onnx.checker = types.SimpleNamespace(check_model=lambda checked: calls.append(checked))
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    artifact = load_model_artifact(path)

    assert calls == [model]
    assert artifact.status == "graph-inspected"
    assert artifact.validation["lowering"] == "node-metadata-only"
    assert artifact.graph_nodes == ({
        "id": "add", "kind": "Add", "op_type": "Add", "name": "add",
        "inputs": ["x", "y"], "outputs": ["z"],
    },)


def test_safetensors_header_is_inspected_without_loading_weights(tmp_path):
    header = {"layer.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "weights.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0" * 12)
    artifact = load_model_artifact(path)
    assert artifact.status == "metadata-only"
    assert artifact.tensors["layer.weight"]["shape"] == [2, 3]
    assert any("not loaded" in warning for warning in artifact.warnings)


def test_pickle_state_dict_has_an_explicit_safe_preview_boundary(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"not deserialized")
    artifact = load_model_artifact(path)
    assert artifact.format == "state-dict"
    assert artifact.status == "preview"
    assert any("not deserialized" in warning for warning in artifact.warnings)


def test_prometheus_samples_preserve_labels_and_timestamps():
    samples = parse_prometheus_text(
        '# HELP gpu_memory_bytes memory\n'
        'gpu_memory_bytes{gpu="0",host="node-a"} 123.5 1700000000123\n'
        'request_latency_seconds{route="generate"} 0.25\n'
    )
    assert len(samples) == 2
    assert samples[0].labels["host"] == "node-a"
    assert samples[0].timestamp_ms == 1700000000123
    assert samples[1].value == 0.25


def test_prometheus_rejects_invalid_lines():
    with pytest.raises(ValueError, match="invalid Prometheus"):
        parse_prometheus_text("bad line")

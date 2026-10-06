import json

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

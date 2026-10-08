import json

import numpy as np
import pytest

from nomo.emulation import (
    CONFIG_SCHEMA,
    RESULT_SCHEMA,
    EmulationConfig,
    EmulationLimitError,
    EmulationLimits,
    EmulationTrace,
    UnsupportedGraphOperation,
    artifact_from_dict,
    artifact_to_dict,
    emulate,
    qgraph_to_dict,
)
from nomo.emulation.cli import main
from nomo.runtime.qgraph import QDecoder, QDense, QEncoder, QGraph, QLIF, QSymLinear, QToQ16


def dense_graph():
    return QGraph(
        "tiny-dense",
        [
            QDense("dense", np.array([[1, 2], [3, 4]], dtype=np.int8), np.array([0, 1], dtype=np.int32), 1, 0, True),
            QToQ16("output", 65536, 2),
        ],
        in_size=2,
        in_scale=1.0,
        out_size=2,
    )


def rate_lif_graph():
    return QGraph(
        "tiny-rate-lif",
        [
            QEncoder("encode", theta=2, T=3, n=2),
            QLIF(
                "lif",
                np.array([[2, 1], [1, 2]], dtype=np.int8),
                np.zeros(2, dtype=np.int32),
                theta=2,
                leak_shift=0,
                v_bits=8,
                T=3,
            ),
            QDecoder("decode", T=3, n=2, out="q16", k_q16=65536),
        ],
        in_size=2,
        in_scale=1.0,
        out_size=2,
    )


def test_emulation_is_deterministic_and_explicitly_simulated():
    qgraph = dense_graph()
    first = emulate(qgraph, [[1, 2], [-1, 3]])
    second = emulate(qgraph, [[1, 2], [-1, 3]])

    assert first.to_json() == second.to_json()
    assert first.schema == RESULT_SCHEMA
    assert first.backend["capability"] == "simulated"
    assert first.backend["physical_measurement"] is False
    assert first.summary["measurement"]["physical_measurement"] is False
    assert first.summary["metrics"]["energy_uj"]["capability"] == "proxy"
    assert first.summary["metrics"]["energy_uj"]["value"] is None


def test_cycle_and_memory_accounting_balances_trace():
    result = emulate(dense_graph(), [[1, 2], [3, -2]])

    assert result.summary["cycle_accounting"]["balanced"] is True
    assert result.summary["cycles"] == sum(layer["cycles"] for layer in result.layers)
    assert result.summary["memory_accesses"] == result.summary["cache_hits"] + result.summary["cache_misses"]
    assert result.summary["memory_accesses"] == sum(layer["memory_accesses"] for layer in result.layers)
    assert result.summary["cycles"] == sum(event["duration_cycles"] for event in result.trace)

    for vector in result.vectors:
        events = [event for event in result.trace if event["vector"] == vector["index"]]
        assert events[0]["cycle_start"] == 0
        assert events[-1]["cycle_end"] == vector["cycles"]
        assert all(left["cycle_end"] == right["cycle_start"] for left, right in zip(events, events[1:]))
        assert sum(event["duration_cycles"] for event in events) == vector["cycles"]

    for layer in result.layers:
        assert layer["memory_accesses"] == layer["cache_hits"] + layer["cache_misses"]
        assert layer["capability"] == "simulated"


def test_rate_lif_reports_spikes_and_bounded_timestep_trace():
    config = EmulationConfig(trace=EmulationTrace("timestep"))
    result = emulate(rate_lif_graph(), [[2, 1], [-2, 3]], config=config)

    lif = next(layer for layer in result.layers if layer["kind"] == "lif")
    encoder = next(layer for layer in result.layers if layer["kind"] == "encoder")
    decoder = next(layer for layer in result.layers if layer["kind"] == "decoder")
    assert encoder["spike_count"] > 0
    assert lif["input_event_count"] == encoder["spike_count"]
    assert lif["spike_count"] == lif["output_event_count"]
    assert decoder["spike_count"] > 0
    assert any(event["event_type"] == "timestep" for event in result.trace)
    assert result.summary["cycles"] == result.summary["cycle_accounting"]["layer_cycles"]


def test_symbolic_integer_stage_is_profiled():
    graph = QGraph(
        "tiny-symbolic",
        [
            QToQ16("to_q16", 65536, 2),
            QSymLinear("symbolic", np.eye(2, dtype=np.int64) * 65536, "identity"),
        ],
        in_size=2,
        in_scale=1.0,
        out_size=2,
    )
    result = emulate(graph, [[1, -2]])
    assert [layer["kind"] for layer in result.layers] == ["to_q16", "sym_linear"]
    assert result.summary["cycles"] > 0


def test_unsupported_graph_operation_is_rejected_before_execution():
    class UnsupportedStage:
        name = "conv"
        kind = "conv2d"

    graph = QGraph("unsupported", [UnsupportedStage()], 2, 1.0, 2)
    with pytest.raises(UnsupportedGraphOperation, match="unsupported operation"):
        emulate(graph, [[1, 2]])


def test_input_and_execution_limits_are_enforced():
    with pytest.raises(ValueError, match="signed 8-bit"):
        emulate(dense_graph(), [[128, 0]])
    config = EmulationConfig(limits=EmulationLimits(max_vectors=1))
    with pytest.raises(EmulationLimitError, match="vectors"):
        emulate(dense_graph(), [[1, 2], [3, 4]], config=config)


def test_artifact_round_trip_and_cli(tmp_path, capsys):
    artifact_path = tmp_path / "artifact.json"
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    artifact_path.write_text(json.dumps(artifact_to_dict(dense_graph(), [[1, 2]])))
    config_path.write_text(json.dumps({
        "schema": CONFIG_SCHEMA,
        "hardware": {"macs_per_cycle": 4, "vector_lanes": 2},
        "trace": {"granularity": "stage"},
    }))

    assert main(["--artifact", str(artifact_path), "--config", str(config_path), "--out", str(result_path)]) == 0
    payload = json.loads(result_path.read_text())
    assert payload["schema"] == RESULT_SCHEMA
    assert payload["backend"]["capability"] == "simulated"
    assert payload["configuration"]["hardware"]["macs_per_cycle"] == 4
    assert "wrote" in capsys.readouterr().out

    restored = artifact_from_dict(json.loads(artifact_path.read_text()))
    assert qgraph_to_dict(restored.graph) == qgraph_to_dict(dense_graph())

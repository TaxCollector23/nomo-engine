from nomo_planner.model_configs import LLAMA_3_8B, MIXTRAL_8X7B
from nomo_planner.simcore import (
    EfficiencyPoint, Link, Parallelism, RooflineHardware, Topology, TrainingOptions,
    build_operator_graph, collective_time, roofline_time, simulate_training_step,
)


def tiny_config(**changes):
    value = {
        "_name_or_path": "tiny-fixture", "num_hidden_layers": 2, "hidden_size": 8,
        "intermediate_size": 16, "num_attention_heads": 2,
        "num_key_value_heads": 1, "vocab_size": 32,
        "tie_word_embeddings": False, "hidden_act": "gelu",
    }
    value.update(changes)
    return value


def test_operator_graph_has_hand_checkable_shapes_flops_bytes_and_lifetimes():
    graph = build_operator_graph(tiny_config(num_hidden_layers=1), sequence_length=2,
                                 batch_size=1, precision="bf16")
    ops = graph.by_id
    q = ops["layer.0.q_proj"]
    assert q.shape == (1, 2, 8)
    assert q.flops == 2 * 2 * 8 * 8
    assert q.read_bytes == 2 * 8 * 2 + 8 * 8 * 2
    assert q.write_bytes == 2 * 8 * 2
    assert q.live_until > q.live_from
    assert "loss.backward" in ops and "optimizer" in ops
    assert ops["layer.0.q_proj.backward"].phase == "backward"
    assert all(op.flops >= 0 and op.bytes_moved >= 0 for op in graph.operators)


def test_real_llama_and_mixtral_configs_build_dense_and_moe_operator_graphs():
    llama = build_operator_graph(LLAMA_3_8B, sequence_length=4, batch_size=1)
    mixtral = build_operator_graph(MIXTRAL_8X7B, sequence_length=4, batch_size=1)
    assert len(llama.operators) > 32 * 10
    assert llama.by_id["layer.0.q_proj"].metadata["weight_shape"] == (4096, 4096)
    assert mixtral.by_id["layer.0.router"].metadata["experts"] == 8
    assert mixtral.by_id["layer.0.experts"].metadata["active_experts"] == 2
    assert mixtral.by_id["layer.0.experts"].flops < 8 * 2 * 4 * 3 * 4096 * 14336 * 2


def test_roofline_respects_compute_memory_bounds_shape_curve_and_provenance():
    graph = build_operator_graph(tiny_config(num_hidden_layers=1), sequence_length=2)
    op = graph.by_id["layer.0.q_proj"]
    base = roofline_time(op, RooflineHardware(1000, 100), precision="bf16")
    assert base.seconds == max(base.compute_seconds, base.memory_seconds)
    assert base.arithmetic_intensity == op.flops / op.bytes_moved
    assert "not hardware validation" in base.provenance[0]
    calibrated = roofline_time(op, RooflineHardware(
        1000, 100, curves=(EfficiencyPoint("gemm", "bf16", 1, 1, 1, 0.25,
                                            "customer microbenchmark run-17"),)), precision="bf16")
    assert calibrated.compute_efficiency == 0.25
    assert calibrated.provenance[0] == "customer microbenchmark run-17"


def test_collective_hand_calculation_and_hierarchical_selection():
    topology = Topology(4, gpus_per_node=4,
                       intra_node=Link(100, 0.01, "fixture NVLink"),
                       inter_node=Link(10, 0.1, "fixture fabric"))
    ring = collective_time(topology, 4, 120, "all_reduce", "ring")
    assert ring.bytes_transferred == 2 * 3 / 4 * 120
    assert ring.seconds == 3 * 0.01 * 2 + ring.bytes_transferred / 100
    assert collective_time(Topology(16), 16, 1024).algorithm == "hierarchical"


def test_simulator_is_deterministic_emits_gantt_memory_and_parallel_collectives():
    graph = build_operator_graph(tiny_config(num_hidden_layers=2), sequence_length=2)
    hardware = RooflineHardware(1e9, 1e9)
    topology = Topology(2)
    options = TrainingOptions(micro_batches=2, schedule="gpipe",
                              parallelism=Parallelism(data=2))
    first = simulate_training_step(graph, hardware, topology, options)
    second = simulate_training_step(graph, hardware, topology, options)
    assert first == second
    assert first.step_time_s == max(event.end_s for event in first.events)
    assert any(event.stream == "communication" and event.kind == "collective" for event in first.events)
    assert {event.gpu for event in first.events} == {0, 1}
    assert first.memory_timeline and all(point.allocated_bytes >= 0 for point in first.memory_timeline)
    assert len(first.peak_memory_by_gpu) == 2


def test_recompute_and_offload_flags_change_event_schedule_and_gantt_exposes_dependencies():
    graph = build_operator_graph(tiny_config(num_hidden_layers=1), sequence_length=2)
    hardware = RooflineHardware(1e9, 1e9)
    topology = Topology(1)
    plain = simulate_training_step(graph, hardware, topology,
                                   TrainingOptions(micro_batches=1, schedule="gpipe"))
    changed = simulate_training_step(graph, hardware, topology,
                                     TrainingOptions(micro_batches=1, schedule="gpipe",
                                                     recompute=True, offload=True))
    assert changed.step_time_s > plain.step_time_s
    assert any(event.kind == "offload" for event in changed.events)
    assert all(event.end_s >= event.start_s for event in changed.events)
    assert any(event.dependencies for event in plain.events)


def test_pipeline_transfer_is_explicit_and_downstream_compute_waits_for_it():
    graph = build_operator_graph(tiny_config(num_hidden_layers=2), sequence_length=2)
    result = simulate_training_step(
        graph, RooflineHardware(1e9, 1e9), Topology(2),
        TrainingOptions(micro_batches=2, schedule="1f1b",
                        parallelism=Parallelism(pipeline=2)),
    )
    transfers = [event for event in result.events if event.kind == "pipeline_transfer"]
    assert len(transfers) == 2
    by_id = {event.id: event for event in result.events}
    for transfer in transfers:
        downstream = next(event for event in result.events
                          if event.id.startswith(f"fwd.mb{transfer.micro_batch}.layer.1."))
        assert transfer.id in downstream.dependencies
        assert downstream.start_s >= transfer.end_s


def test_stage_aware_schedules_keep_embedding_dependency_and_support_interleaved_zero_bubble():
    graph = build_operator_graph(tiny_config(num_hidden_layers=2), sequence_length=2)
    for schedule in ("1f1b", "interleaved", "zero-bubble"):
        result = simulate_training_step(
            graph, RooflineHardware(1e9, 1e9), Topology(2),
            TrainingOptions(micro_batches=2, schedule=schedule,
                            parallelism=Parallelism(pipeline=2),
                            sequence_parallel=1, context_parallel=1, expert_parallel=1),
        )
        first_layer = next(event for event in result.events if event.id == "fwd.mb0.layer.0.norm1")
        embedding = next(event for event in result.events if event.id == "fwd.mb0.embedding")
        assert embedding.id in first_layer.dependencies
        assert first_layer.start_s >= embedding.end_s
        assert any("Schedule policy" in assumption for assumption in result.assumptions)


def test_sequence_context_expert_parallel_and_fsdp_boundaries_are_explicit():
    graph = build_operator_graph(tiny_config(num_hidden_layers=1, num_local_experts=2, num_experts_per_tok=1), sequence_length=2)
    result = simulate_training_step(
        graph, RooflineHardware(1e9, 1e9), Topology(4),
        TrainingOptions(micro_batches=1, schedule="1f1b", parallelism=Parallelism(data=1),
                        sequence_parallel=2, context_parallel=2, expert_parallel=2,
                        zero_stage=3, fsdp=True),
    )
    kinds = {event.kind for event in result.events}
    assert {"sequence_parallel", "context_parallel", "expert_parallel"}.issubset(kinds)
    assert any("ZeRO stage 3" in assumption for assumption in result.assumptions)

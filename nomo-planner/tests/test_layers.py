from nomo_planner.layers import (
    LayerTrainingPlan,
    TrainingHardware,
    TrainingProblem,
    balanced_stages,
    build_graph,
    repair_plan,
    search,
)


def config(**overrides):
    value = {
        "_name_or_path": "tiny-test",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
    }
    value.update(overrides)
    return value


def test_huggingface_config_expands_ordered_graph_and_totals():
    graph = build_graph(config(), seq_len=16, batch_size=2)
    assert [node.id for node in graph.nodes] == [
        "embedding", "block.0.attention", "block.0.mlp", "block.1.attention", "block.1.mlp", "output"
    ]
    assert graph.nodes[-1].parameter_count == 0
    assert graph.nodes[1].kv_cache_bytes > 0
    assert graph.parameter_count == sum(node.parameter_count for node in graph.nodes)


def test_repair_makes_stages_contiguous_and_locks_are_absolute():
    graph = build_graph(config())
    raw = LayerTrainingPlan((4, 0, 4, 9, 2, 9), ("fp8",) * 6, ("full",) * 6, (True,) * 6)
    repaired = repair_plan(graph, raw, locks={"block.0.mlp": {"stage": 1, "precision": "fp8"}})
    assert repaired.stages == tuple(sorted(repaired.stages))
    assert repaired.stages[1] <= repaired.stages[2]
    assert repaired.precision[0] == "bf16"
    assert repaired.precision[-1] == "bf16"
    assert repaired.precision[2] == "fp8"


def test_layer_search_reports_comparison_and_respects_memory():
    graph = build_graph(config(), seq_len=32, batch_size=1)
    problem = TrainingProblem(graph, TrainingHardware(devices=4, memory_bytes=1e9, usable_memory=0.95), max_candidates=3000)
    result = search(problem)
    assert result.evaluated > 0
    assert result.best is not None
    assert result.best_metrics is not None
    assert result.baseline[1].objectives["step_time_s"] > 0
    assert result.best_metrics.objectives["step_time_s"] > 0


def test_balanced_stages_are_non_empty():
    stages = balanced_stages(7, 3)
    assert stages == (0, 0, 0, 1, 1, 2, 2)

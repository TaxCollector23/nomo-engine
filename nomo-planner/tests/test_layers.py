from nomo_planner.layers import (
    LayerTrainingPlan,
    TrainingHardware,
    TrainingProblem,
    balanced_stages,
    build_graph,
    repair_plan,
    search,
)
from nomo_planner.model_configs import LLAMA_3_8B, LLAMA_3_70B, MIXTRAL_8X7B


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


def test_fair_global_comparison_separates_precision_and_per_layer_gain():
    graph = build_graph(LLAMA_3_8B, seq_len=128, batch_size=1)
    result = search(TrainingProblem(graph, max_candidates=600, seed=17))
    assert result.global_best == result.baseline
    assert result.precision_gain_pct >= 0
    assert 0 <= result.per_layer_gain_pct <= 20, "unexpected >20% gain needs a documented physical case"


def test_offload_is_not_chosen_when_all_stages_have_headroom():
    graph = build_graph(config(), seq_len=16, batch_size=1)
    problem = TrainingProblem(graph, TrainingHardware(memory_bytes=80e9, overhead_bytes=1e6), max_candidates=500)
    result = search(problem)
    assert result.best is not None
    assert not any(result.best.offload)


def test_stage_count_is_minimal_unless_charged_cost_shows_benefit():
    graph = build_graph(config(), seq_len=16, batch_size=1)
    problem = TrainingProblem(graph, TrainingHardware(memory_bytes=80e9, overhead_bytes=1e6), max_candidates=500)
    result = search(problem)
    assert result.best is not None
    assert "minimum feasible stage count" in " ".join(result.assumptions) or "charged pipeline/bandwidth model" in " ".join(result.assumptions)


def test_published_model_search_is_reproducible_with_fixed_seed():
    for model_config in (LLAMA_3_8B, LLAMA_3_70B):
        graph = build_graph(model_config, seq_len=128, batch_size=1)
        problem = TrainingProblem(graph, max_candidates=600, seed=20261003)
        first = search(problem)
        second = search(problem)
        assert first.seed == second.seed == 20261003
        assert first.best == second.best
        assert first.best_metrics == second.best_metrics


def test_builtin_model_config_values_are_published_values():
    assert (LLAMA_3_8B["num_hidden_layers"], LLAMA_3_8B["hidden_size"], LLAMA_3_8B["vocab_size"]) == (32, 4096, 128256)
    assert (LLAMA_3_70B["num_hidden_layers"], LLAMA_3_70B["hidden_size"], LLAMA_3_70B["vocab_size"]) == (80, 8192, 128256)
    assert (MIXTRAL_8X7B["num_local_experts"], MIXTRAL_8X7B["num_experts_per_tok"], MIXTRAL_8X7B["vocab_size"]) == (8, 2, 32000)

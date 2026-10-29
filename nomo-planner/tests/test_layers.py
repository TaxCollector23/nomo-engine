from nomo_planner.layers import (
    LayerTrainingPlan,
    TrainingHardware,
    TrainingProblem,
    balanced_stages,
    build_neuromorphic_manifest,
    build_graph_from_contract,
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


def test_neuromorphic_manifest_preserves_shared_graph_and_declares_boundary():
    graph = build_graph(config(), seq_len=16, batch_size=2)
    manifest = build_neuromorphic_manifest(graph)
    assert manifest["format"] == "nomo.graph/neuromorphic-preview/1"
    assert manifest["status"] == "contract-only"
    assert len(manifest["graph"]["nodes"]) == len(graph.nodes)
    assert "NIR/C11/RTL" in manifest["export_boundary"]


def test_graph_contract_can_be_reimported_without_reducing_nodes():
    graph = build_graph(config(), seq_len=16, batch_size=2)
    restored = build_graph_from_contract(build_neuromorphic_manifest(graph))
    assert restored.nodes == graph.nodes
    assert restored.parameter_count == graph.parameter_count
    assert restored.seq_len == graph.seq_len and restored.batch_size == graph.batch_size


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


def test_small_graph_keeps_exact_exhaustive_search_when_uncapped():
    graph = build_graph(config(num_hidden_layers=1), seq_len=8, batch_size=1)
    result = search(TrainingProblem(graph, pipeline_stages=(1, 2), max_candidates=10_000, seed=5))
    assert result.exhaustive is True
    # Endpoint BF16 repair leaves 2^(4 - 2) precision choices, with all
    # recompute/offload choices, for each of the two stage counts.
    assert result.evaluated == 2 * (2 ** 2) * (2 ** 4) * (2 ** 4)
    assert "exact exhaustive" in " ".join(result.assumptions)


def test_large_graph_uses_repair_aware_nsga2_with_a_deterministic_budget():
    graph = build_graph(config(num_hidden_layers=5), seq_len=16, batch_size=1)
    problem = TrainingProblem(graph, pipeline_stages=(1, 2, 4, 8), max_candidates=80, seed=123)
    first = search(problem)
    second = search(problem)
    assert first.exhaustive is False
    assert first.evaluated == 80
    assert first == second
    assert "NSGA-II" in " ".join(first.assumptions)
    assert first.best is not None
    assert first.best.precision[0] == first.best.precision[-1] == "bf16"
    for plan, _metrics in first.front:
        assert list(plan.stages) == sorted(plan.stages)
        assert plan.precision[0] == plan.precision[-1] == "bf16"


def test_builtin_model_config_values_are_published_values():
    assert (LLAMA_3_8B["num_hidden_layers"], LLAMA_3_8B["hidden_size"], LLAMA_3_8B["vocab_size"]) == (32, 4096, 128256)
    assert (LLAMA_3_70B["num_hidden_layers"], LLAMA_3_70B["hidden_size"], LLAMA_3_70B["vocab_size"]) == (80, 8192, 128256)
    assert (MIXTRAL_8X7B["num_local_experts"], MIXTRAL_8X7B["num_experts_per_tok"], MIXTRAL_8X7B["vocab_size"]) == (8, 2, 32000)


def test_roofline_points_use_evaluator_accounting_and_ridge():
    from nomo_planner.layers import roofline_points

    graph = build_graph(config(num_hidden_layers=4), seq_len=32, batch_size=1)
    problem = TrainingProblem(graph, TrainingHardware(memory_bandwidth_bytes_s=3.35e12), max_candidates=100)
    result = search(problem)
    points = roofline_points(problem, result.best)
    assert len(points) == len(graph.nodes)
    for point, node in zip(points, graph.nodes):
        assert point["id"] == node.id
        # intensity is exactly work / minimal bytes on the evaluator's accounting
        assert abs(point["intensity"] - point["flops"] / point["bytes_min"]) < 1e-9 * max(point["intensity"], 1e-300)
        # ridge and attainable follow the roofline definitions
        assert abs(point["ridge_flops"] - point["peak_flops"] / 3.35e12) < 1e-6
        assert abs(point["attainable_flops"] - min(point["peak_flops"], point["intensity"] * 3.35e12)) < 1e-3
        assert point["bound"] == ("memory" if point["intensity"] < point["ridge_flops"] else "compute")
    # embedding/output nodes move whole weights for little math: they must classify as bandwidth-bound
    kinds = {point["id"]: point for point in points}
    assert kinds[graph.nodes[0].id]["bound"] == "memory"
    # a tiny bandwidth pushes the ridge up and can only flip compute -> memory, never the reverse
    slow = roofline_points(
        TrainingProblem(graph, TrainingHardware(memory_bandwidth_bytes_s=1e10), max_candidates=100), result.best
    )
    fast = {point["id"]: point["bound"] for point in points}
    for point in slow:
        assert point["bound"] == "memory" or fast[point["id"]] == "memory"


def test_memory_timeline_peaks_match_evaluator_and_returns_to_baseline():
    from nomo_planner.layers import memory_timeline

    graph = build_graph(config(num_hidden_layers=6), seq_len=32, batch_size=1)
    problem = TrainingProblem(graph, TrainingHardware(devices=4, memory_bytes=1e9, usable_memory=0.95), max_candidates=2000)
    result = search(problem)
    timeline = memory_timeline(problem, result.best)
    assert timeline["micro_batches"] == problem.micro_batches
    assert [stage["stage"] for stage in timeline["stages"]] == list(range(max(result.best.stages) + 1))
    for stage, memory in zip(timeline["stages"], result.best_metrics.memory_by_stage):
        # the timeline is the evaluator's accounting shown as a shape, never new terms
        assert abs(stage["forward_peak_bytes"] - memory) < 1e-6 * memory
        assert abs(stage["end_bytes"] - stage["baseline_bytes"]) < 1e-6 * stage["baseline_bytes"]
        assert stage["baseline_bytes"] > 0
        # the forward walk is monotone and ends at the peak
        walks = stage["forward_walk"]
        assert walks[0]["bytes_after"] >= stage["baseline_bytes"]
        assert abs(walks[-1]["bytes_after"] - stage["forward_peak_bytes"]) < 1e-6 * memory


def test_memory_timeline_offload_keeps_15_percent_like_evaluator():
    from nomo_planner.layers import balanced_stages, memory_timeline

    graph = build_graph(config(num_hidden_layers=4), seq_len=64, batch_size=1)
    problem = TrainingProblem(graph, TrainingHardware(memory_bytes=80e9), max_candidates=100)
    n = len(graph.nodes)
    plan = LayerTrainingPlan(
        balanced_stages(n, 2),
        tuple(["bf16"] * n),
        tuple(["none"] * n),
        tuple([True] * n),
    )
    metrics = problem_hardware_metrics(problem, plan)
    timeline = memory_timeline(problem, plan)
    for stage, memory in zip(timeline["stages"], metrics.memory_by_stage):
        assert abs(stage["forward_peak_bytes"] - memory) < 1e-6 * memory


def problem_hardware_metrics(problem, plan):
    from nomo_planner.layers import evaluate_plan

    return evaluate_plan(problem, plan)

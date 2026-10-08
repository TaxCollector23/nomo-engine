from nomo_planner.layers import build_graph
from nomo_planner.serving import ServingPlan, ServingProblem, evaluate_serving_plan, search_serving


def config(**overrides):
    value = {
        "_name_or_path": "tiny-serving-test",
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


def test_serving_search_returns_per_node_weight_and_kv_decisions():
    graph = build_graph(config(), seq_len=32, batch_size=1)
    result = search_serving(ServingProblem(graph, max_quality_penalty_pct=2.0, max_candidates=500))
    assert result.best is not None
    assert result.best_metrics is not None
    assert len(result.best.weight_precision) == len(graph.nodes)
    assert len(result.best.kv_precision) == len(graph.nodes)
    assert result.best_metrics.objectives["latency_s"] > 0
    assert result.best_metrics.objectives["memory_bytes_per_device"] > 0


def test_serving_locks_are_absolute_and_non_attention_kv_is_bf16():
    graph = build_graph(config(), seq_len=16, batch_size=1)
    result = search_serving(ServingProblem(graph, max_quality_penalty_pct=2.0), {
        "block.0.attention": {"weight_precision": "bf16", "kv_precision": "fp8"},
        "block.0.mlp": {"precision": "fp8"},
    })
    assert result.best is not None
    attention = graph.decision_units.index("block.0.attention")
    mlp = graph.decision_units.index("block.0.mlp")
    assert result.best.weight_precision[attention] == "bf16"
    assert result.best.kv_precision[attention] == "fp8"
    assert result.best.weight_precision[mlp] == "fp8"
    assert result.best.kv_precision[mlp] == "bf16"


def test_serving_rejects_quality_or_memory_violations():
    graph = build_graph(config(), seq_len=64, batch_size=1)
    problem = ServingProblem(graph, max_quality_penalty_pct=0.01, max_candidates=200)
    plan = ServingPlan(tuple("fp8" for _ in graph.nodes), tuple("fp8" for _ in graph.nodes))
    metrics = evaluate_serving_plan(problem, plan)
    assert metrics.constraints["quality"] > 0
    assert metrics.constraints["memory"] <= 0


def test_serving_search_is_reproducible():
    graph = build_graph(config(), seq_len=32, batch_size=1)
    problem = ServingProblem(graph, max_quality_penalty_pct=2.0, seed=17, max_candidates=500)
    first = search_serving(problem)
    second = search_serving(problem)
    assert first == second

from nomo_planner.product_models import (
    ChipSpec, FineTuneProblem, ReliabilityProblem, TCOOption, TrafficHour, Workload,
    evaluate_chip, evaluate_reliability, evaluate_rl_plan, evaluate_tco, fine_tune_options,
    optimize_reliability, pareto_chips, size_fleet, RLProblem,
)


def test_chip_pareto_and_bottleneck_are_physical():
    chip = ChipSpec("test", 1000, 80, 100, 200, 800, 128, 16, 10000, 300, 2)
    workload = Workload("llm", 1e15, 1e12)
    result = evaluate_chip(chip, workload, precision="fp8")
    assert result["feasible"] and result["seconds"] > 0
    assert pareto_chips([chip], [workload])[0]["chip"] == "test"


def test_reliability_young_interval_and_goodput_sanity():
    problem = ReliabilityProblem(0.1, 0.05, 0.1, 8, 2)
    expected = (2 * 0.05 / 0.1) ** 0.5
    assert abs(optimize_reliability(problem)["young_interval_hours"] - expected) < 1e-12
    assert evaluate_reliability(problem, expected)["goodput"] < 1


def test_fleet_size_meets_capacity_target_and_finetune_modes_are_labeled():
    fleet = size_fleet([TrafficHour(9, 100, 80, 200), TrafficHour(10, 200, 80, 200)])
    assert fleet[1]["replicas"] >= fleet[0]["replicas"]
    options = fine_tune_options(FineTuneProblem(1e9, 1e10, 16e9, 100, 80e9, 2, {"full": 0, "lora": 0.5, "qlora": 1}))
    assert {row["mode"] for row in options} == {"full", "lora", "qlora"}


def test_rl_requires_answer_length_distribution_and_returns_plan_metrics():
    problem = RLProblem(128, 512, 1000, 5000, 2000, (2,), (4,), (1,), 2, 1_000_000, 100)
    result = evaluate_rl_plan(problem, rollout_gpus=2, train_gpus=4, reward_gpus=1, colocated=False, asynchronous=True, batch_size=8, kv_precision="fp8")
    assert result["samples_per_hour"] > 0 and result["cost_usd_per_update"] > 0


def test_tco_keeps_price_range_and_buy_rent_comparison_explicit():
    result = evaluate_tco(TCOOption("test", 1000, 1, 0.5, 3, 1000, 0.1, 0.1, 800, 1200))
    assert result["range_low"] < result["range_high"]
    assert result["recommended"] in {"buy", "rent"}

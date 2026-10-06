from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nomo_planner.product_simulations import (  # noqa: E402
    ArtifactInput,
    AutoscalingPolicy,
    ArchitectureCandidate,
    ChipDesign,
    Distribution,
    FineTuneMode,
    FineTuningSpec,
    Interval,
    OpenModelCostArtifact,
    ProcurementContract,
    ProcurementSpec,
    Provenance,
    RLPlan,
    RLSystemSpec,
    ReliabilitySpec,
    ScalingLaw,
    ServingFleetSpec,
    SoftwareDesign,
    Timeline,
    TimelineEvent,
    TrafficPoint,
    WorkloadArtifact,
    build_timeline,
    daly_checkpoint_interval,
    export_result_csv,
    export_timeline_csv,
    interval_from_samples,
    pareto_frontier,
    nsga2_search,
    search_design_space,
    search_rl_schedules,
    simulate_architecture_search,
    simulate_chip_software_search,
    simulate_fine_tuning,
    simulate_procurement,
    simulate_reliability,
    simulate_rl_schedule,
    simulate_serving_fleet,
    track_open_model_costs,
    validate_young_daly,
    young_checkpoint_interval,
)


EVIDENCE = Provenance(
    artifact_id="fixture-2026-10-05",
    source="customer-fixture",
    source_url="https://example.invalid/fixture",
    observed_at="2026-10-05",
    measured=True,
)


def test_distribution_timeline_and_search_are_reusable_and_exportable():
    dist = Distribution.from_value({"kind": "empirical", "values": [1, 2, 3]}, name="latency", provenance=EVIDENCE)
    assert dist.interval(samples=12).low == 1
    assert interval_from_samples([1, 2, 3]).median == 2

    timeline = build_timeline([
        TimelineEvent("b", "train", 2, 3, "gpu"),
        TimelineEvent("a", "rollout", 0, 2, "gpu"),
    ])
    assert isinstance(timeline, Timeline)
    assert timeline.duration_s == 5
    assert timeline.resource_utilization("gpu") == 1
    assert "event_id" in export_timeline_csv(timeline.events)

    rows = [{"name": "fast", "latency": 1, "cost": 5}, {"name": "cheap", "latency": 2, "cost": 1}, {"name": "dominated", "latency": 3, "cost": 6}]
    frontier = pareto_frontier(rows, (("latency", True), ("cost", True)))
    assert {row["name"] for row in frontier} == {"fast", "cheap"}
    report = search_design_space([1, 2], lambda value: {"name": value, "latency": value}, objectives=(("latency", True),))
    assert report.best["name"] == 1


def test_nsga2_search_repairs_genomes_and_returns_constrained_frontier():
    def repair(candidate):
        value = int(candidate["x"])
        return {"x": max(1, min(18, value))}

    report = nsga2_search(
        [{"x": value} for value in range(0, 21)],
        lambda candidate: {"latency": candidate["x"], "cost": 20 - candidate["x"], "valid": candidate["x"] <= 18},
        objectives=(("latency", True), ("cost", True)),
        feasible=lambda row: bool(row["valid"]),
        repair=repair,
        population_size=10,
        generations=4,
        seed=7,
    )
    assert report.algorithm == "nsga-ii"
    assert report.evaluations > 0
    assert report.repaired_candidates > 0
    assert report.feasible and report.frontier and report.best
    assert all(row["valid"] for row in report.feasible)
    assert report.as_dict()["algorithm"] == "nsga-ii"


def test_artifact_result_exports_are_structured():
    artifact = ArtifactInput("trace", {"value": 3}, artifact_id="a1", source="fixture", measured=True)
    assert artifact.provenance.evidence_grade == "measured"
    result = simulate_architecture_search(
        [ArchitectureCandidate("tiny", 1e9, 1e12, layers=12, hidden_size=768, sequence_length=2048, provenance=EVIDENCE)],
        [ScalingLaw("law", parameter_coefficient=0.5, token_coefficient=0.5, provenance=EVIDENCE)],
        train_artifact=ArtifactInput("training-calibration", {"train_tokens_per_second": 1e5, "gpu_count": 8, "gpu_hourly_cost": 2}, source="train.csv", measured=True),
        serve_artifact=ArtifactInput("serving-calibration", {"serve_tokens_per_second": 2e4, "gpu_count": 2, "gpu_hourly_cost": 2}, source="serve.csv", measured=True),
        repetitions=8,
    )
    assert result.status == "Validated"
    assert result.intervals["best_loss"].low <= result.intervals["best_loss"].median <= result.intervals["best_loss"].high
    assert result.best["total_cost_usd"] >= 0
    assert "candidate" in export_result_csv(result)
    assert result.to_json().startswith("{")
    assert {artifact.kind for artifact in result.artifacts} >= {"architecture-candidate", "scaling-law", "training-calibration", "serving-calibration"}


def test_missing_architecture_evidence_is_preview():
    result = simulate_architecture_search(
        [{"name": "unshaped", "parameter_count": 1e9, "training_tokens": 1e12}],
        [{"name": "law", "parameter_coefficient": 1, "token_coefficient": 1}],
        repetitions=4,
    )
    assert result.preview
    assert result.status == "Preview"
    assert "Preview" in result.labels
    assert result.preview_labels


def test_chip_software_search_reports_bottleneck_and_frontier():
    result = simulate_chip_software_search(
        [ChipDesign("chip-a", 100, 1e12, interconnect_bytes_s=2e11, memory_bytes=80e9, power_w=300, area_mm2=700, unit_cost_usd=2, provenance=EVIDENCE)],
        [SoftwareDesign("kernel-a", precision="bf16", compute_efficiency=0.8, quality_loss=0.1, provenance=EVIDENCE)],
        WorkloadArtifact("train-step", 1e15, 1e12, communication_bytes=1e11, provenance=EVIDENCE),
        workload_artifact=ArtifactInput("workload-trace", {"step": 1}, source="trace", measured=True),
    )
    assert result.metrics["pareto_count"] == 1
    assert result.best["bottleneck"] in {"compute", "memory", "communication"}
    assert result.intervals["latency_s"].median > 0
    assert result.status == "Validated"
    assert {artifact.kind for artifact in result.artifacts} >= {"workload-trace", "chip-design", "software-design", "workload"}


def test_rl_simulation_has_phase_timeline_staleness_and_search():
    system = RLSystemSpec(
        answer_lengths=Distribution("empirical", values=(128, 256, 512), provenance=EVIDENCE),
        rollout_tokens_per_second_per_gpu=1000,
        reward_tokens_per_second_per_gpu=2000,
        train_tokens_per_second_per_gpu=5000,
        training_tokens_per_update=1_000_000,
        gpu_cost_per_hour=2,
        target_samples_per_hour=1,
        provenance=EVIDENCE,
    )
    plan = RLPlan(rollout_gpus=2, reward_gpus=1, train_gpus=4, batch_size=8, asynchronous=True, provenance=EVIDENCE)
    result = simulate_rl_schedule(system, plan, replications=16)
    assert result.metrics["samples_per_hour"] > 0
    assert {event.phase for event in result.timeline.events} >= {"rollout", "reward", "train"}
    assert result.intervals["staleness_steps"].low >= 0
    assert {artifact.kind for artifact in result.artifacts} >= {"rl-system", "rl-plan", "answer-length-distribution"}
    assert result.status == "Validated"
    searched = search_rl_schedules(system, [plan, RLPlan(1, 1, 1, batch_size=4, asynchronous=False, name="sync", provenance=EVIDENCE)], replications=8)
    assert searched.metrics["evaluated"] == 2
    assert len(searched.rows) == 2 and searched.timeline.events and searched.artifacts


def test_reliability_monte_carlo_checks_young_and_daly():
    young = young_checkpoint_interval(0.1, 0.05)
    daly = daly_checkpoint_interval(0.1, 0.05)
    assert daly < young
    checks = validate_young_daly(failure_rate_per_hour=0.1, checkpoint_write_hours=0.05, simulated_interval_hours=daly)
    assert checks["daly_not_greater_than_young"]
    result = simulate_reliability(ReliabilitySpec(0.1, 0.05, 0.1, checkpoint_interval_hours=daly, horizon_hours=8, gpu_count=4, gpu_cost_per_hour=2, provenance=EVIDENCE), replications=64)
    assert result.metrics["young_interval_hours"] == young
    assert result.metrics["daly_interval_hours"] == daly
    assert 0 <= result.intervals["goodput"].median <= 1
    assert any(artifact.kind == "reliability-spec" for artifact in result.artifacts)
    assert result.status == "Validated"


def test_serving_fleet_replays_cold_starts_and_compares_policies():
    trace = [
        TrafficPoint(0, 10, duration_s=60, p99_target_ms=500, day=1, provenance=EVIDENCE),
        TrafficPoint(1, 100, duration_s=60, p99_target_ms=500, day=1, provenance=EVIDENCE),
        TrafficPoint(2, 20, duration_s=60, p99_target_ms=500, day=2, provenance=EVIDENCE),
    ]
    fleet = ServingFleetSpec(50, 1.0, cold_start_s=30, initial_replicas=1, provenance=EVIDENCE)
    policy = AutoscalingPolicy("reactive", target_utilization=0.7, provenance=EVIDENCE)
    result = simulate_serving_fleet(trace, fleet, policy, days=2)
    assert result.metrics["days_simulated"] == 2
    assert result.metrics["total_cost_usd"] > 0
    assert any(event.phase == "cold-start" for event in result.timeline.events)
    assert {artifact.kind for artifact in result.artifacts} >= {"serving-traffic-trace", "serving-fleet", "autoscaling-policy"}
    compared = __import__("nomo_planner.product_simulations", fromlist=["search_autoscaling_policies"]).search_autoscaling_policies(trace, fleet, [policy, AutoscalingPolicy("predictive", predictive=True, provenance=EVIDENCE)], days=2)
    assert compared.metrics["evaluated"] == 2
    assert len(compared.rows) == 2 and compared.timeline.events and compared.artifacts


def test_fine_tuning_reports_memory_boundary_and_framework_export():
    spec = FineTuningSpec(1e9, 1e8, 2048, 2, 4, 4, 80e9, 100, 2, forward_flops_per_token=6e9, provenance=EVIDENCE)
    modes = [
        FineTuneMode("full", 1.0, 2, 2, 8, 2, 2, quality_loss=0, provenance=EVIDENCE),
        FineTuneMode("lora", 0.01, 2, 2, 8, 2, 1, quality_loss=0.5, provenance=EVIDENCE),
    ]
    result = simulate_fine_tuning(spec, modes, quality_target=1)
    assert {row["mode"] for row in result.rows} == {"full", "lora"}
    assert result.best["memory_bytes_per_gpu"] > 0
    assert {artifact.kind for artifact in result.artifacts} >= {"fine-tuning-spec", "fine-tuning-mode"}
    assert result.status == "Validated"
    exported = __import__("nomo_planner.product_simulations", fromlist=["export_finetune_config"]).export_finetune_config(result, "deepspeed")
    assert exported["zero_optimization"]["stage"] == 3


def test_procurement_monte_carlo_tracks_cash_flow_intervals():
    spec = ProcurementSpec(3, [1000, 1200, 1400], [0.1, 0.12], scenarios=64, provenance=EVIDENCE)
    contracts = [
        ProcurementContract("buy", "buy", purchase_price_usd=10_000, maintenance_usd_per_year=500, residual_value_usd=1000, power_kw=1, provenance=EVIDENCE),
        ProcurementContract("lease", "lease", lease_usd_per_hour=1.5, power_kw=1, provenance=EVIDENCE),
    ]
    result = simulate_procurement(contracts, spec)
    assert result.metrics["scenarios"] == 64
    assert len(result.rows) == 2
    assert result.best["contract"] in {"buy", "lease"}
    assert result.intervals["buy.npv_usd"].low <= result.intervals["buy.npv_usd"].high
    assert {artifact.kind for artifact in result.artifacts} >= {"procurement-spec", "procurement-contract"}
    assert result.status == "Validated"


def test_open_model_cost_tracker_is_scoped_and_uses_measured_serving_inputs():
    row = OpenModelCostArtifact("open-7b", version="1", training_gpu_hours=[100, 110], gpu_hourly_cost_usd=[2, 3], measured_tokens_per_second=[1000, 1200], serving_gpu_count=2, license="Apache-2.0", provenance=EVIDENCE)
    result = __import__("nomo_planner.product_simulations", fromlist=["track_open_model_costs"]).track_open_model_costs(
        [row],
        serving_artifact=ArtifactInput("serving-cost-measurement", {"measured_tokens_per_second": [1000, 1200]}, source="serving.csv", measured=True),
        cost_artifact=ArtifactInput("gpu-price-artifact", {"gpu_hourly_cost_usd": [2, 3]}, source="price.csv", measured=True),
        samples=32,
    )
    assert result.metrics["open_model_only"] is True
    assert result.rows[0]["comparability"] == "open-model artifacts only"
    assert result.rows[0]["serving_cost_usd_per_million_tokens"] > 0
    assert any(artifact.kind == "open-model-cost-record" for artifact in result.artifacts)


def test_closed_model_is_rejected_in_open_model_tracker():
    with pytest.raises(ValueError, match="closed-model"):
        OpenModelCostArtifact("closed", open_model=False)


def test_architecture_loss_and_cost_match_hand_calculation():
    law = ScalingLaw("hand", parameter_coefficient=2, token_coefficient=3,
                     parameter_exponent=1, token_exponent=1,
                     irreducible_loss=0.5, parameter_scale=2, token_scale=3,
                     loss_std=0,
                     provenance=EVIDENCE)
    candidate = ArchitectureCandidate(
        "tiny", parameter_count=2, training_tokens=3, serving_requests=6,
        request_tokens=10, layers=1, hidden_size=2, sequence_length=4,
        train_tokens_per_second=10, serve_tokens_per_second=5,
        train_gpu_count=2, serve_gpu_count=1,
        train_gpu_hourly_cost=3, serve_gpu_hourly_cost=2,
        provenance=EVIDENCE,
    )
    result = simulate_architecture_search(
        [candidate], [law],
        train_artifact=ArtifactInput("training-calibration", {"train_tokens_per_second": 10, "gpu_count": 2, "gpu_hourly_cost": 3}, source="train", measured=True),
        serve_artifact=ArtifactInput("serving-calibration", {"serve_tokens_per_second": 5, "gpu_count": 1, "gpu_hourly_cost": 2}, source="serve", measured=True),
        repetitions=4,
    )
    row = result.best
    assert row["loss"] == pytest.approx(5.5)
    assert row["train_cost_usd"] == pytest.approx(3 / 10 / 3600 * 2 * 3)
    assert row["serve_cost_usd"] == pytest.approx(60 / 5 / 3600 * 2)
    assert row["loss_interval"].low == row["loss_interval"].high == 5.5
    assert [event.phase for event in result.timeline.events] == ["train", "serve"]
    assert result.timeline.duration_s == pytest.approx(3 / 10 + 60 / 5)
    assert result.status == "Validated"


def test_chip_roofline_is_compute_memory_and_energy_consistent():
    chip = ChipDesign("one", compute_tflops=1, memory_bandwidth_bytes_s=1e12,
                      interconnect_bytes_s=1e12, memory_bytes=2e12, power_w=100,
                      area_mm2=100, unit_cost_usd=1, gpu_count=1,
                      provenance=EVIDENCE)
    software = SoftwareDesign("plain", compute_efficiency=1,
                              provenance=EVIDENCE)
    workload = WorkloadArtifact("step", dense_flops=2e12,
                                memory_bytes=1e12, communication_bytes=0,
                                provenance=EVIDENCE)
    result = simulate_chip_software_search([chip], [software], workload,
        workload_artifact=ArtifactInput("trace", {"steps": 1}, source="trace", measured=True))
    row = result.best
    assert row["compute_s"] == pytest.approx(2)
    assert row["memory_s"] == pytest.approx(1)
    assert row["latency_s"] == pytest.approx(2)
    assert row["energy_j"] == pytest.approx(200)
    assert row["bottleneck"] == "compute"
    assert result.rows == (row,)
    assert result.timeline.duration_s == pytest.approx(2)


def test_rl_synchronous_phases_and_cost_are_hand_checkable():
    system = RLSystemSpec(
        answer_lengths=100,
        rollout_tokens_per_second_per_gpu=10,
        reward_tokens_per_second_per_gpu=20,
        train_tokens_per_second_per_gpu=100,
        training_tokens_per_update=1000,
        gpu_cost_per_hour=1,
        sync_overhead_s=1,
        provenance=EVIDENCE,
    )
    plan = RLPlan(2, 1, 2, batch_size=2, asynchronous=False, provenance=EVIDENCE)
    result = simulate_rl_schedule(system, plan, replications=3)
    assert result.metrics["step_seconds"] == pytest.approx(13.5)
    assert result.metrics["samples_per_hour"] == pytest.approx(3600 * 2 / 13.5)
    assert result.metrics["cost_usd_per_update"] == pytest.approx(13.5 * 5 / 3600)
    assert [(event.phase, event.duration_s) for event in result.timeline.events] == [
        ("rollout", 5), ("reward", 5), ("train", 2.5), ("synchronization", 1)
    ]
    assert result.status == "Validated"


def test_reliability_seeded_trials_and_checkpoint_search_are_repeatable():
    spec = ReliabilitySpec(0.1, 0.05, 0.1, checkpoint_interval_hours=1,
                           horizon_hours=8, gpu_count=2, gpu_cost_per_hour=3,
                           provenance=EVIDENCE)
    left = simulate_reliability(spec, replications=64, seed=11)
    right = simulate_reliability(spec, replications=64, seed=11)
    assert left.metrics == right.metrics
    assert left.intervals == right.intervals
    assert left.intervals["goodput"].low >= 0
    assert left.intervals["goodput"].high <= 1
    assert left.metrics["daly_interval_hours"] <= left.metrics["young_interval_hours"]
    search = __import__("nomo_planner.product_simulations", fromlist=["search_checkpoint_policies"]).search_checkpoint_policies(
        spec, [0.5, 1, 2], replications=16, seed=11)
    assert search.metrics["evaluated"] == 3
    assert len(search.rows) == 3
    assert search.timeline.events


def test_serving_queue_load_latency_and_cost_match_hand_calculation():
    trace = [TrafficPoint(0, 2, duration_s=3600, p99_target_ms=100,
                         provenance=EVIDENCE)]
    fleet = ServingFleetSpec(10, 3, initial_replicas=1, base_latency_ms=50,
                             provenance=EVIDENCE)
    result = simulate_serving_fleet(trace, fleet,
        AutoscalingPolicy("steady", target_utilization=0.5, provenance=EVIDENCE), days=1)
    assert result.rows[0]["desired_replicas"] == 1
    assert result.rows[0]["utilization"] == pytest.approx(0.2)
    assert result.rows[0]["approx_p99_ms"] == pytest.approx(62.5)
    assert result.metrics["total_cost_usd"] == pytest.approx(3)
    assert result.rows[0]["served_requests"] == pytest.approx(7200)
    assert result.rows[0]["backlog_requests"] == 0
    assert result.status == "Validated"


def test_finetuning_memory_throughput_and_time_match_accounting():
    spec = FineTuningSpec(100, 72000, 10, 2, 1, 2, 1000, 1, 3,
                          forward_flops_per_token=100,
                          activation_bytes_per_token=2,
                          memory_overhead_factor=1,
                          provenance=EVIDENCE)
    mode = FineTuneMode("full", 1, 2, 2, 8, 2, 2, quality_loss=0,
                        provenance=EVIDENCE)
    result = simulate_fine_tuning(spec, [mode], quality_target=0)
    row = result.best
    assert row["memory_bytes"] == pytest.approx(1440)
    assert row["memory_bytes_per_gpu"] == pytest.approx(720)
    assert row["tokens_per_second"] == pytest.approx(2e10)
    assert row["time_hours"] == pytest.approx(1e-9)
    assert result.metrics["evaluated"] == 1
    assert result.timeline.events[0].duration_s == pytest.approx(72000 / 2e10)


def test_procurement_constant_scenarios_match_cash_flow_and_contract_search():
    spec = ProcurementSpec(2, 100, 0.1, discount_rate=0, scenarios=4,
                           provenance=EVIDENCE)
    buy = ProcurementContract("buy", "buy", purchase_price_usd=1000,
        maintenance_usd_per_year=10, residual_value_usd=100, power_kw=2,
        provenance=EVIDENCE)
    lease = ProcurementContract("lease", "lease", lease_usd_per_hour=2,
        power_kw=2, provenance=EVIDENCE)
    result = simulate_procurement([buy, lease], spec, seed=9)
    assert result.intervals["buy.npv_usd"].median == pytest.approx(960)
    assert result.intervals["lease.npv_usd"].median == pytest.approx(440)
    assert result.best["contract"] == "lease"
    assert result.best["probability_cheapest"] == 1
    assert [event.metadata["median_cash_usd"] for event in result.timeline.events] == [0, 220, 220]
    assert result.status == "Validated"


def test_open_model_training_and_serving_costs_match_physical_units():
    result = track_open_model_costs([
        OpenModelCostArtifact("open-a", training_gpu_hours=10,
            gpu_hourly_cost_usd=2, measured_tokens_per_second=1000,
            serving_gpu_count=2, serving_gpu_hourly_cost_usd=2,
            provenance=EVIDENCE),
        OpenModelCostArtifact("open-b", training_gpu_hours=20,
            gpu_hourly_cost_usd=2, measured_tokens_per_second=500,
            serving_gpu_count=1, serving_gpu_hourly_cost_usd=2,
            provenance=EVIDENCE),
    ], serving_artifact=ArtifactInput("serving", {"measured_tokens_per_second": 1000}, source="bench", measured=True),
       cost_artifact=ArtifactInput("price", {"gpu_hourly_cost_usd": 2}, source="price", measured=True),
       samples=8, seed=4)
    assert result.rows[0]["training_cost_usd"] == pytest.approx(20)
    assert result.rows[0]["serving_cost_usd_per_million_tokens"] == pytest.approx(1_000_000 / 1000 / 3600 * 2 * 2)
    assert result.best["model"] == "open-a"
    assert result.metrics["models"] == 2
    assert result.timeline.events
    assert result.status == "Validated"


def test_unattributed_artifacts_keep_preview_visible_with_outputs():
    result = simulate_architecture_search(
        [{"name": "preview", "parameter_count": 100, "training_tokens": 1000,
          "layers": 2, "hidden_size": 8, "sequence_length": 16}],
        [{"name": "assumed", "parameter_coefficient": 1,
          "token_coefficient": 1, "loss_std": 0}], repetitions=4)
    assert result.status == "Preview"
    assert result.labels[0] == "Preview"
    assert result.artifacts
    assert result.rows and result.timeline.events and result.intervals

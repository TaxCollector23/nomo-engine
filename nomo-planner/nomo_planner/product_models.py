"""Reference models for the infrastructure and chip-maker product packs.

These are deliberately small, auditable estimators.  Inputs such as prices,
failure rates, traffic, answer lengths, and quality effects are required from
the user; no market or customer data is silently invented.

Formula references:
* Young (1961), checkpoint interval: doi:10.1145/1460765.1460782
* Daly (1990), improved checkpoint/restart analysis: doi:10.1145/93542.93546
* Erlang (1917), queueing approximation: https://archive.org/details/solutionofsomepr00erla
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ChipSpec:
    name: str
    memory_bandwidth_gbps: float
    sram_gb: float
    bf16_tops: float
    fp8_tops: float
    interconnect_gbps: float
    neuromorphic_cores: int
    neuromorphic_memory_gb: float
    cost_usd: float
    power_w: float
    hourly_cost_usd: float


@dataclass(frozen=True)
class Workload:
    name: str
    dense_flops: float
    memory_bytes: float
    spiking_ops: float = 0.0
    requires_neuromorphic: bool = False


def evaluate_chip(spec: ChipSpec, workload: Workload, *, precision: str = "bf16") -> dict[str, object]:
    if min(spec.memory_bandwidth_gbps, spec.sram_gb, spec.bf16_tops, spec.power_w, spec.hourly_cost_usd) <= 0:
        raise ValueError("chip specs must be positive")
    if workload.dense_flops < 0 or workload.memory_bytes < 0 or workload.spiking_ops < 0:
        raise ValueError("workload terms cannot be negative")
    tops = spec.fp8_tops if precision == "fp8" else spec.bf16_tops
    if precision not in {"bf16", "fp8"} or tops <= 0:
        raise ValueError("precision must have a positive chip throughput")
    if workload.requires_neuromorphic and (spec.neuromorphic_cores <= 0 or spec.neuromorphic_memory_gb <= 0):
        return {"feasible": False, "reason": "workload requires neuromorphic cores and memory"}
    compute_s = workload.dense_flops / (tops * 1e12)
    memory_s = workload.memory_bytes / (spec.memory_bandwidth_gbps * 1e9)
    spike_s = workload.spiking_ops / (spec.neuromorphic_cores * 1e9) if workload.spiking_ops else 0.0
    seconds = max(compute_s, memory_s, spike_s)
    return {
        "feasible": True, "seconds": seconds, "energy_j": seconds * spec.power_w,
        "cost_usd": seconds / 3600 * spec.hourly_cost_usd,
        "mapping": {"dense_precision": precision, "bottleneck": "compute" if compute_s >= memory_s and compute_s >= spike_s else "memory" if memory_s >= spike_s else "neuromorphic", "neuromorphic_cores": spec.neuromorphic_cores if workload.spiking_ops else 0},
        "notes": ["chip bandwidth, throughput, power, and price are supplied estimates or customer inputs"],
    }


def pareto_chips(specs: Iterable[ChipSpec], workloads: Sequence[Workload], *, precision: str = "bf16") -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        evaluations = [evaluate_chip(spec, workload, precision=precision) for workload in workloads]
        if all(bool(item.get("feasible")) for item in evaluations):
            rows.append({"chip": spec.name, "spec": spec, "workloads": evaluations, "total_seconds": sum(float(item["seconds"]) for item in evaluations), "total_cost_usd": sum(float(item["cost_usd"]) for item in evaluations), "total_energy_j": sum(float(item["energy_j"]) for item in evaluations)})
    frontier = []
    for row in rows:
        dominated = any(other is not row and other["total_seconds"] <= row["total_seconds"] and other["total_cost_usd"] <= row["total_cost_usd"] and (other["total_seconds"] < row["total_seconds"] or other["total_cost_usd"] < row["total_cost_usd"]) for other in rows)
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (float(row["total_cost_usd"]), float(row["total_seconds"])))


@dataclass(frozen=True)
class RLProblem:
    answer_length_p50: int
    answer_length_p95: int
    rollout_tokens_per_s_per_gpu: float
    train_tokens_per_s_per_gpu: float
    reward_tokens_per_s_per_gpu: float
    rollout_gpus: tuple[int, ...]
    train_gpus: tuple[int, ...]
    reward_gpus: tuple[int, ...]
    gpu_cost_per_hour: float
    training_tokens_per_update: int
    target_samples_per_hour: float


def evaluate_rl_plan(problem: RLProblem, *, rollout_gpus: int, train_gpus: int, reward_gpus: int, colocated: bool, asynchronous: bool, batch_size: int, kv_precision: str) -> dict[str, float | str]:
    if problem.answer_length_p50 <= 0 or problem.answer_length_p95 < problem.answer_length_p50:
        raise ValueError("answer-length p50/p95 are required customer inputs")
    if min(rollout_gpus, train_gpus, reward_gpus, batch_size) <= 0:
        raise ValueError("RL plan counts must be positive")
    precision_factor = 1.15 if kv_precision == "fp8" else 1.0
    rollout_s = problem.answer_length_p95 / (problem.rollout_tokens_per_s_per_gpu * rollout_gpus * precision_factor)
    reward_s = problem.answer_length_p95 / (problem.reward_tokens_per_s_per_gpu * reward_gpus)
    train_s = problem.training_tokens_per_update / (problem.train_tokens_per_s_per_gpu * train_gpus * batch_size)
    sync_s = 0.0 if asynchronous else 0.05 * max(rollout_s, train_s)
    if colocated:
        idle_s = max(0.0, rollout_s + train_s + reward_s - max(rollout_s, train_s, reward_s))
        active_gpus = max(rollout_gpus, train_gpus, reward_gpus)
    else:
        idle_s = max(0.0, rollout_s - train_s) + max(0.0, train_s - reward_s)
        active_gpus = rollout_gpus + train_gpus + reward_gpus
    step_s = rollout_s + train_s + reward_s + sync_s
    samples_per_hour = 3600 * batch_size / max(step_s, 1e-12)
    return {"samples_per_hour": samples_per_hour, "cost_usd_per_update": step_s / 3600 * active_gpus * problem.gpu_cost_per_hour, "idle_seconds": idle_s, "step_seconds": step_s, "kv_precision": kv_precision, "schedule": "async" if asynchronous else "sync", "colocation": "colocated" if colocated else "separate", "feasible": samples_per_hour >= problem.target_samples_per_hour}


@dataclass(frozen=True)
class ReliabilityProblem:
    failure_rate_per_hour: float
    checkpoint_write_hours: float
    restart_hours: float
    gpu_count: int
    gpu_cost_per_hour: float


def evaluate_reliability(problem: ReliabilityProblem, checkpoint_interval_hours: float) -> dict[str, float]:
    if min(problem.failure_rate_per_hour, problem.checkpoint_write_hours, problem.restart_hours, checkpoint_interval_hours) <= 0:
        raise ValueError("failure, checkpoint, restart, and interval inputs must be positive")
    checkpoint_overhead = problem.checkpoint_write_hours / checkpoint_interval_hours
    lost_work = problem.failure_rate_per_hour * checkpoint_interval_hours / 2
    restart_overhead = problem.failure_rate_per_hour * problem.restart_hours
    goodput = 1 / (1 + checkpoint_overhead + lost_work + restart_overhead)
    useful_cost = problem.gpu_count * problem.gpu_cost_per_hour / goodput
    return {"checkpoint_interval_hours": checkpoint_interval_hours, "goodput": goodput, "useful_training_hours_per_hour": goodput, "cost_usd_per_useful_hour": useful_cost}


def optimize_reliability(problem: ReliabilityProblem, intervals: Iterable[float] | None = None) -> dict[str, float]:
    choices = tuple(intervals or (0.25, 0.5, 1, 2, 4, 8, 12, 24))
    optimal_young = math.sqrt(2 * problem.checkpoint_write_hours / problem.failure_rate_per_hour)
    best = min((evaluate_reliability(problem, interval) for interval in choices), key=lambda row: -row["goodput"])
    return {**best, "young_interval_hours": optimal_young}


@dataclass(frozen=True)
class TrafficHour:
    hour: int
    arrival_rps: float
    service_rps_per_replica: float
    p99_target_ms: float


def size_fleet(traffic: Iterable[TrafficHour], *, target_utilization: float = 0.70, replica_hourly_cost: float = 1.0,
               service_cv: float = 1.0) -> list[dict[str, float]]:
    """Fleet sizing with a real queueing tail.

    Response-time p99 uses the exact M/M/1 sojourn-time tail P(T > t) = exp(-(mu - lambda) t),
    which is what the old 1000*ln(100)/spare formula approximated - but the old denominator
    divided the fleet-wide spare capacity by nothing and understated p99 by a factor of the
    replica count. With a service-time coefficient of variation cv > 1, the Pollaczek-Khinchine
    (Kingman) mean-wait excess is added on top; cv = 1 reproduces M/M/1 exactly.
    """
    if not 0 < target_utilization < 1:
        raise ValueError("target utilization must be between 0 and 1")
    if service_cv <= 0:
        raise ValueError("service coefficient of variation must be positive")
    result = []
    for row in traffic:
        if min(row.arrival_rps, row.service_rps_per_replica, row.p99_target_ms) <= 0:
            raise ValueError("traffic and latency inputs must be positive")
        replicas = max(1, math.ceil(row.arrival_rps / (row.service_rps_per_replica * target_utilization)))
        per_replica_rps = row.arrival_rps / replicas
        mu = row.service_rps_per_replica
        rho = per_replica_rps / mu
        if rho < 1.0:
            mm1_p99_s = math.log(100.0) / (mu - per_replica_rps)
            mean_wait_mg1 = rho * (1.0 + service_cv ** 2) / (2.0 * mu * (1.0 - rho))
            mean_wait_mm1 = rho / (mu - per_replica_rps)
            p99 = 1000.0 * (mm1_p99_s + max(0.0, mean_wait_mg1 - mean_wait_mm1))
            queue_model = "M/M/1 sojourn tail" if service_cv == 1.0 else "M/G/1 tail (Pollaczek-Khinchine excess)"
        else:
            p99 = math.inf
            queue_model = "saturated (rho >= 1)"
        result.append({"hour": row.hour, "replicas": replicas, "approx_p99_ms": p99, "hourly_cost_usd": replicas * replica_hourly_cost,
                       "within_target": p99 <= row.p99_target_ms, "arrival_rps": row.arrival_rps,
                       "per_replica_utilization": rho, "queue_model": queue_model})
    return result


@dataclass(frozen=True)
class FineTuneProblem:
    parameter_count: int
    training_tokens: int
    memory_bytes: float
    peak_tflops: float
    gpu_memory_bytes: float
    gpu_cost_per_hour: float
    quality_loss: Mapping[str, float]
    adapter_param_fraction: float = 0.005  # share of weight matrices adapted (LoRA/QLoRA)


def fine_tune_options(problem: FineTuneProblem) -> list[dict[str, object]]:
    """Fine-tuning cost model from first principles, replacing invented factors.

    Memory (per-parameter convention, matching nomo_planner.layers): full fine-tuning trains
    everything: 2 B weights + 2 B grads + 4 B fp32 master + 8 B Adam states = 16 B/param plus
    activations. LoRA freezes the base in bf16 (2 B/param) and trains adapters only
    (adapter share f of params, each with the same 16 B/param training state) - Hu et al. 2021,
    arXiv:2106.09685. QLoRA loads the base in NF4 (about 0.55 B/param including quantization
    constants) - Dettmers et al. 2023, arXiv:2305.14314. Activations are shared by all modes
    and are recovered from the user-supplied full fine-tune footprint.

    Compute: full training costs about 6ND FLOPs (2 forward + 4 backward). LoRA skips the
    frozen-weight gradient term: about 4ND FLOPs. QLoRA additionally pays an assumed ~10%
    NF4 de-quantization overhead (4ND / 0.9). Quality remains a customer-supplied
    assumption; the planner makes no quality claim.
    """
    if min(problem.parameter_count, problem.training_tokens, problem.memory_bytes, problem.peak_tflops, problem.gpu_memory_bytes, problem.gpu_cost_per_hour) <= 0:
        raise ValueError("fine-tuning inputs must be positive")
    if not 0 < problem.adapter_param_fraction <= 1:
        raise ValueError("adapter_param_fraction must be in (0, 1]")
    n = problem.parameter_count
    full_states = 16.0 * n
    activation_bytes = max(0.0, problem.memory_bytes - full_states)
    adapter_states = 16.0 * n * problem.adapter_param_fraction
    # FLOPs per token per parameter: full training 6ND (2 fwd + 4 bwd); LoRA skips the
    # frozen-weight gradient term -> 4ND; QLoRA adds an assumed ~10% NF4 de-quantization
    # overhead -> 4ND / 0.9.
    flops_per_token = {"full": 6.0, "lora": 4.0, "qlora": 4.0 / 0.9}
    memory_models = {
        "full": (problem.memory_bytes, "16 B/param training state (2 weights + 2 grads + 4 master + 8 Adam) plus activations"),
        "lora": (2.0 * n + adapter_states + activation_bytes,
                 f"frozen bf16 base (2 B/param) + adapters ({problem.adapter_param_fraction:.3g} of params, 16 B/param state) + activations; compute ~4ND/6ND of full (Hu et al. 2021, arXiv:2106.09685)"),
        "qlora": (0.55 * n + adapter_states + activation_bytes,
                  f"NF4 base ~0.55 B/param incl. quantization constants (Dettmers et al. 2023, arXiv:2305.14314) + adapter states + activations; ~10% de-quantization overhead assumed"),
    }
    rows = []
    for mode in ("full", "lora", "qlora"):
        memory, model_note = memory_models[mode]
        seconds = flops_per_token[mode] * n * problem.training_tokens / (problem.peak_tflops * 1e12)
        rows.append({"mode": mode, "memory_bytes": memory, "time_hours": seconds / 3600,
                     "cost_usd": seconds / 3600 * problem.gpu_cost_per_hour,
                     "quality_loss": problem.quality_loss.get(mode, float("nan")),
                     "feasible": memory <= problem.gpu_memory_bytes,
                     "notes": f"quality is a customer-supplied assumption; {model_note}"})
    return rows


@dataclass(frozen=True)
class TCOOption:
    name: str
    purchase_usd: float
    lease_usd_per_hour: float
    utilization: float
    years: float
    hours_per_year: float
    energy_usd_per_hour: float
    maintenance_pct: float
    price_low_usd: float | None = None
    price_high_usd: float | None = None


def evaluate_tco(option: TCOOption) -> dict[str, float | str]:
    if min(option.purchase_usd, option.lease_usd_per_hour, option.utilization, option.years, option.hours_per_year) <= 0 or not 0 < option.utilization <= 1:
        raise ValueError("TCO inputs must be positive and utilization must be in (0, 1]")
    effective_hours = option.years * option.hours_per_year * option.utilization
    capex = option.purchase_usd / effective_hours
    maintenance = option.purchase_usd * option.maintenance_pct / effective_hours
    buy = capex + maintenance + option.energy_usd_per_hour / option.utilization
    rent = option.lease_usd_per_hour / option.utilization + option.energy_usd_per_hour / option.utilization
    low = ((option.price_low_usd or option.purchase_usd) / effective_hours) + maintenance + option.energy_usd_per_hour / option.utilization
    high = ((option.price_high_usd or option.purchase_usd) / effective_hours) + maintenance + option.energy_usd_per_hour / option.utilization
    return {"name": option.name, "buy_usd_per_useful_hour": buy, "rent_usd_per_useful_hour": rent, "range_low": min(low, high), "range_high": max(low, high), "recommended": "buy" if buy <= rent else "rent", "notes": "prices, utilization, depreciation, energy, and maintenance are user-entered inputs"}

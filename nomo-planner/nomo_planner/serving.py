"""Layer-aware serving precision planner.

This module consumes the same :class:`ModelGraph` used by the training
planner.  It searches per-node weight precision and per-attention-node KV
precision, while charging memory, latency, and serving cost explicitly.
The numbers are estimates until a customer supplies calibrated GPU
measurements; the assumptions are returned with every result.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping

from .layers import BYTES, ModelGraph


SERVE_SPEEDUP = {"bf16": 1.0, "fp8": 1.35}
SERVE_QUALITY_PENALTY = {"bf16": 0.0, "fp8": 0.35}


@dataclass(frozen=True)
class ServingHardware:
    devices: int = 1
    memory_bytes: float = 80e9
    peak_flops: float = 989e12
    memory_bandwidth_bytes_s: float = 2e12
    cost_per_device_hour: float = 3.50
    usable_memory: float = 0.90


@dataclass(frozen=True)
class ServingProblem:
    graph: ModelGraph
    hardware: ServingHardware = ServingHardware()
    max_quality_penalty_pct: float = 1.0
    max_candidates: int = 20_000
    seed: int = 20261003


@dataclass(frozen=True)
class ServingPlan:
    weight_precision: tuple[str, ...]
    kv_precision: tuple[str, ...]


@dataclass(frozen=True)
class ServingMetrics:
    objectives: dict[str, float]
    constraints: dict[str, float]
    memory_by_device: tuple[float, ...]
    quality_penalty_pct: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServingSearchResult:
    graph: ModelGraph
    best: ServingPlan | None
    best_metrics: ServingMetrics | None
    baseline: tuple[ServingPlan, ServingMetrics]
    global_best: tuple[ServingPlan, ServingMetrics]
    per_layer_gain_pct: float
    evaluated: int
    exhaustive: bool
    seed: int
    assumptions: tuple[str, ...]


def _lock_value(lock: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in lock and lock[key] is not None:
            value = str(lock[key])
            if value in BYTES:
                return value
    return None


def repair_serving_plan(
    graph: ModelGraph,
    plan: ServingPlan,
    *,
    locks: Mapping[str, Mapping[str, Any]] | None = None,
) -> ServingPlan:
    """Normalize precision arrays and apply absolute node locks."""

    n = len(graph.nodes)
    if len(plan.weight_precision) != n or len(plan.kv_precision) != n:
        raise ValueError("serving plan arrays must match graph node count")
    weights = [value if value in BYTES else "bf16" for value in plan.weight_precision]
    kv = [value if value in BYTES else "bf16" for value in plan.kv_precision]
    # KV cache only exists on attention nodes.  Keeping other entries BF16
    # makes the serialized plan total and keeps browser/Python parity simple.
    for index, node in enumerate(graph.nodes):
        if node.kind != "attention":
            kv[index] = "bf16"
        lock = (locks or {}).get(node.id, {})
        weight = _lock_value(lock, "weight_precision", "precision")
        cache = _lock_value(lock, "kv_precision", "kv")
        if weight is not None:
            weights[index] = weight
        if cache is not None and node.kind == "attention":
            kv[index] = cache
    return ServingPlan(tuple(weights), tuple(kv))


def evaluate_serving_plan(
    problem: ServingProblem,
    raw_plan: ServingPlan,
    locks: Mapping[str, Mapping[str, Any]] | None = None,
) -> ServingMetrics:
    graph = problem.graph
    plan = repair_serving_plan(graph, raw_plan, locks=locks)
    hardware = problem.hardware
    devices = max(1, int(hardware.devices))
    capacity = hardware.memory_bytes * hardware.usable_memory
    weights_bytes = sum(node.parameter_count * BYTES[precision] for node, precision in zip(graph.nodes, plan.weight_precision))
    kv_bytes = sum(node.kv_cache_bytes * BYTES[precision] for node, precision in zip(graph.nodes, plan.kv_precision))
    # Weights are tensor-parallel sharded; KV is replicated on each serving
    # device.  This is conservative for memory and transparent in the report.
    replicated = kv_bytes + hardware.memory_bytes * 0.02
    memory = weights_bytes / devices + replicated
    memory_by_device = tuple(memory for _ in range(devices))
    compute_seconds = sum(
        node.forward_flops / (hardware.peak_flops * SERVE_SPEEDUP[precision] * devices)
        for node, precision in zip(graph.nodes, plan.weight_precision)
    )
    transfer_seconds = (weights_bytes / devices + kv_bytes) / hardware.memory_bandwidth_bytes_s
    latency = compute_seconds + transfer_seconds
    cost = latency / 3600.0 * devices * hardware.cost_per_device_hour
    weighted_fp8 = sum(node.parameter_count for node, precision in zip(graph.nodes, plan.weight_precision) if precision == "fp8")
    total_parameters = max(1, graph.parameter_count)
    weight_penalty = weighted_fp8 / total_parameters * SERVE_QUALITY_PENALTY["fp8"]
    attention_nodes = [index for index, node in enumerate(graph.nodes) if node.kind == "attention"]
    kv_fp8 = sum(1 for index in attention_nodes if plan.kv_precision[index] == "fp8")
    kv_penalty = (kv_fp8 / max(1, len(attention_nodes))) * 0.20
    quality_penalty = weight_penalty + kv_penalty
    notes = [
        "FP8 throughput and quality penalty are assumptions until customer evaluation",
        "weights are tensor-parallel sharded; KV cache is conservatively replicated",
        "latency uses a roofline-style compute plus memory-transfer estimate",
    ]
    return ServingMetrics(
        objectives={
            "latency_s": latency,
            "cost_usd_per_request": cost,
            "memory_bytes_per_device": memory,
            "throughput_requests_s": 1.0 / latency if latency > 0 else 0.0,
            "weight_bytes": weights_bytes,
            "kv_bytes": kv_bytes,
        },
        constraints={
            "memory": memory / capacity - 1.0,
            "quality": quality_penalty / max(problem.max_quality_penalty_pct, 1e-12) - 1.0,
        },
        memory_by_device=memory_by_device,
        quality_penalty_pct=quality_penalty,
        notes=tuple(notes),
    )


def _feasible(metrics: ServingMetrics) -> bool:
    return all(value <= 0.0 for value in metrics.constraints.values())


def _dedup(plans: Iterable[ServingPlan]) -> list[ServingPlan]:
    seen: set[ServingPlan] = set()
    result: list[ServingPlan] = []
    for plan in plans:
        if plan not in seen:
            seen.add(plan)
            result.append(plan)
    return result


def _candidate_plans(problem: ServingProblem, locks: Mapping[str, Mapping[str, Any]] | None) -> tuple[list[ServingPlan], bool]:
    n = len(problem.graph.nodes)
    attention = [index for index, node in enumerate(problem.graph.nodes) if node.kind == "attention"]
    # Full enumeration is useful for the tiny examples and is the reference
    # path used by tests.  Published models use deterministic bounded moves.
    exhaustive = n <= 6
    plans: list[ServingPlan] = []
    if exhaustive:
        for weights in product(("bf16", "fp8"), repeat=n):
            for kv_values in product(("bf16", "fp8"), repeat=len(attention)):
                kv = ["bf16"] * n
                for index, value in zip(attention, kv_values):
                    kv[index] = value
                plans.append(repair_serving_plan(problem.graph, ServingPlan(weights, tuple(kv)), locks=locks))
                if len(plans) >= problem.max_candidates:
                    return _dedup(plans), False
    else:
        for weight in ("bf16", "fp8"):
            for cache in ("bf16", "fp8"):
                weights = tuple(weight for _ in range(n))
                kv = tuple(cache if index in attention else "bf16" for index in range(n))
                plans.append(repair_serving_plan(problem.graph, ServingPlan(weights, kv), locks=locks))
        # Refine one node at a time around each global policy.  This keeps
        # search bounded while exposing the layer-level trade-off to users.
        for base in tuple(plans):
            for index in range(n):
                weights = list(base.weight_precision)
                weights[index] = "fp8" if weights[index] == "bf16" else "bf16"
                plans.append(repair_serving_plan(problem.graph, ServingPlan(tuple(weights), base.kv_precision), locks=locks))
                if index in attention:
                    kv = list(base.kv_precision)
                    kv[index] = "fp8" if kv[index] == "bf16" else "bf16"
                    plans.append(repair_serving_plan(problem.graph, ServingPlan(base.weight_precision, tuple(kv)), locks=locks))
                if len(plans) >= problem.max_candidates:
                    return _dedup(plans), False
    return _dedup(plans), exhaustive


def _choose_global(problem: ServingProblem, precision: str, kv_precision: str, locks: Mapping[str, Mapping[str, Any]] | None) -> tuple[ServingPlan, ServingMetrics]:
    n = len(problem.graph.nodes)
    attention = {index for index, node in enumerate(problem.graph.nodes) if node.kind == "attention"}
    plan = repair_serving_plan(problem.graph, ServingPlan(
        tuple(precision for _ in range(n)),
        tuple(kv_precision if index in attention else "bf16" for index in range(n)),
    ), locks=locks)
    return plan, evaluate_serving_plan(problem, plan, locks=locks)


def search_serving(problem: ServingProblem, locks: Mapping[str, Mapping[str, Any]] | None = None) -> ServingSearchResult:
    assumptions = (
        "Serving recommendations use the uploaded shared model graph",
        "FP8 quality penalty is a bounded assumption, not a customer measurement",
        "Provide calibrated GPU bandwidth/throughput before production capacity commitments",
    )
    baseline = _choose_global(problem, "bf16", "bf16", locks)
    global_candidates = [_choose_global(problem, weight, kv, locks) for weight in ("bf16", "fp8") for kv in ("bf16", "fp8")]
    feasible_global = [item for item in global_candidates if _feasible(item[1])]
    global_best = min(feasible_global or global_candidates, key=lambda item: (item[1].objectives["latency_s"], item[1].objectives["cost_usd_per_request"]))
    candidates, exhaustive = _candidate_plans(problem, locks)
    evaluated = 0
    feasible: list[tuple[ServingPlan, ServingMetrics]] = []
    for plan in candidates:
        metrics = evaluate_serving_plan(problem, plan, locks=locks)
        evaluated += 1
        if _feasible(metrics):
            feasible.append((plan, metrics))
    best = min(feasible, key=lambda item: (item[1].objectives["latency_s"], item[1].objectives["cost_usd_per_request"])) if feasible else None
    per_layer_gain = 0.0
    if best is not None:
        per_layer_gain = max(0.0, (global_best[1].objectives["latency_s"] / best[1].objectives["latency_s"] - 1.0) * 100.0)
    return ServingSearchResult(
        graph=problem.graph, best=best[0] if best else None, best_metrics=best[1] if best else None,
        baseline=baseline, global_best=global_best, per_layer_gain_pct=per_layer_gain,
        evaluated=evaluated, exhaustive=exhaustive, seed=problem.seed, assumptions=assumptions,
    )

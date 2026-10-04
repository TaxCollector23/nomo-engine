"""Layer-aware model graph and training planner reference implementation.

The module deliberately has no framework dependency. It accepts the stable
subset of a Hugging Face ``config.json`` needed for transformer accounting and
produces a deterministic bounded search result that the browser port can
mirror. Numbers are engineering estimates, not hardware measurements, unless
the caller supplies calibrated hardware parameters.

Formula sources:
* Chowdhery et al. (2022), PaLM, https://arxiv.org/abs/2203.15556
* Korthikanti et al. (2022), activation memory,
  https://arxiv.org/abs/2205.05198
* Chen et al. (2016), recomputation,
  https://arxiv.org/abs/1604.06174
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from math import inf
from typing import Any, Iterable, Mapping, Sequence


BYTES = {"bf16": 2.0, "fp8": 1.0}
PRECISION_SPEEDUP = {"bf16": 1.0, "fp8": 1.35}
RECOMPUTE_FLOP_MULTIPLIER = {"none": 1.0, "selective": 1.12, "full": 1.28}


def _first(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _required_int(config: Mapping[str, Any], *keys: str) -> int:
    value = _first(config, *keys)
    if value is None:
        raise ValueError(f"config.json is missing one of: {', '.join(keys)}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"config.json field {keys[0]} must be positive")
    return value


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    layer_index: int | None
    parameter_count: int
    forward_flops: float
    activation_bytes: float
    kv_cache_bytes: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelGraph:
    name: str
    source: str
    seq_len: int
    batch_size: int
    hidden_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    intermediate_size: int
    vocab_size: int
    tied_embeddings: bool
    gated_mlp: bool
    experts: int | None
    experts_per_token: int | None
    nodes: tuple[GraphNode, ...]
    assumptions: tuple[str, ...] = ()

    @property
    def parameter_count(self) -> int:
        return sum(node.parameter_count for node in self.nodes)

    @property
    def decision_units(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["nodes"] = [asdict(node) for node in self.nodes]
        data["parameter_count"] = self.parameter_count
        data["decision_units"] = list(self.decision_units)
        return data


def build_graph(config: Mapping[str, Any], *, seq_len: int = 2048, batch_size: int = 1,
                source: str = "huggingface-config") -> ModelGraph:
    """Expand a transformer config into one ordered, decision-addressable graph."""

    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    layers = _required_int(config, "num_hidden_layers", "n_layer", "num_layers")
    hidden = _required_int(config, "hidden_size", "n_embd", "d_model")
    heads = _required_int(config, "num_attention_heads", "n_head", "num_heads")
    if hidden % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    kv_heads = int(_first(config, "num_key_value_heads", "n_kv_heads", default=heads))
    if kv_heads <= 0 or heads % kv_heads:
        raise ValueError("num_key_value_heads must be a positive divisor of num_attention_heads")
    ffn = int(_first(config, "intermediate_size", "n_inner", default=4 * hidden))
    vocab = _required_int(config, "vocab_size", "n_vocab")
    tied = bool(_first(config, "tie_word_embeddings", "tie_embeddings", default=False))
    gated = bool(_first(config, "gated_mlp", default=False)) or str(_first(config, "hidden_act", default="")).lower() in {"silu", "swish", "geglu"}
    experts_raw = _first(config, "num_local_experts", "num_experts")
    experts = int(experts_raw) if experts_raw is not None else None
    topk_raw = _first(config, "num_experts_per_tok", "num_experts_per_token", "moe_top_k")
    experts_per_token = int(topk_raw) if topk_raw is not None else None
    if experts is not None and (experts <= 0 or experts_per_token is None or not 0 < experts_per_token <= experts):
        raise ValueError("MoE config must provide valid num_experts and num_experts_per_tok")
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    token_count = seq_len * batch_size
    assumptions: list[str] = []
    if _first(config, "intermediate_size", "n_inner") is None:
        assumptions.append("intermediate_size inferred as 4 * hidden_size")
    if _first(config, "num_key_value_heads", "n_kv_heads") is None:
        assumptions.append("num_key_value_heads inferred as num_attention_heads")
    if experts is not None:
        assumptions.append("MoE parameter memory includes resident experts; FLOPs use top-k active experts")

    nodes: list[GraphNode] = [GraphNode(
        id="embedding", kind="embedding", layer_index=None,
        parameter_count=vocab * hidden, forward_flops=0.0,
        activation_bytes=token_count * hidden * 2.0, kv_cache_bytes=0.0,
        metadata={"shared_with": "output" if tied else None},
    )]
    for index in range(layers):
        attention_params = hidden * hidden + 2 * hidden * kv_dim + hidden * hidden
        attention_flops = 2.0 * token_count * (hidden * hidden + 2 * hidden * kv_dim + hidden * hidden)
        attention_flops += 4.0 * batch_size * seq_len * seq_len * hidden
        nodes.append(GraphNode(
            id=f"block.{index}.attention", kind="attention", layer_index=index,
            parameter_count=attention_params, forward_flops=attention_flops,
            activation_bytes=token_count * hidden * 2.0,
            kv_cache_bytes=batch_size * seq_len * 2.0 * kv_heads * head_dim * 2.0,
            metadata={"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim},
        ))
        expert_multiplier = experts if experts is not None else 1
        active_multiplier = experts_per_token if experts_per_token is not None else 1
        mlp_matrices = 3 if gated else 2
        mlp_params = mlp_matrices * hidden * ffn * expert_multiplier
        mlp_flops = 2.0 * token_count * mlp_matrices * hidden * ffn * active_multiplier
        nodes.append(GraphNode(
            id=f"block.{index}.mlp", kind="mlp", layer_index=index,
            parameter_count=mlp_params, forward_flops=mlp_flops,
            activation_bytes=token_count * ffn * 2.0,
            kv_cache_bytes=0.0,
            metadata={"gated": gated, "experts": experts, "active_experts": experts_per_token},
        ))
    nodes.append(GraphNode(
        id="output", kind="output", layer_index=None,
        parameter_count=0 if tied else vocab * hidden,
        forward_flops=2.0 * token_count * hidden * vocab,
        activation_bytes=token_count * vocab * 2.0,
        kv_cache_bytes=0.0,
        metadata={"shared_with": "embedding" if tied else None},
    ))
    return ModelGraph(
        name=str(_first(config, "_name_or_path", "model_type", default="uploaded-transformer")),
        source=source, seq_len=seq_len, batch_size=batch_size, hidden_size=hidden,
        layers=layers, attention_heads=heads, kv_heads=kv_heads,
        intermediate_size=ffn, vocab_size=vocab, tied_embeddings=tied,
        gated_mlp=gated, experts=experts, experts_per_token=experts_per_token,
        nodes=tuple(nodes), assumptions=tuple(assumptions),
    )


@dataclass(frozen=True)
class TrainingHardware:
    devices: int = 8
    memory_bytes: float = 80e9
    peak_flops: float = 989e12
    cost_per_device_hour: float = 3.50
    interconnect_bytes_s: float = 600e9
    cpu_offload_bytes_s: float = 32e9
    overhead_bytes: float = 3e9
    usable_memory: float = 0.90


@dataclass(frozen=True)
class TrainingProblem:
    graph: ModelGraph
    hardware: TrainingHardware = TrainingHardware()
    micro_batches: int = 8
    pipeline_stages: tuple[int, ...] = (1, 2, 4)
    fp8_enabled: bool = True
    max_candidates: int = 20_000


@dataclass(frozen=True)
class LayerTrainingPlan:
    stages: tuple[int, ...]
    precision: tuple[str, ...]
    recompute: tuple[str, ...]
    offload: tuple[bool, ...]


@dataclass(frozen=True)
class LayerMetrics:
    objectives: dict[str, float]
    constraints: dict[str, float]
    memory_by_stage: tuple[float, ...]
    stage_times: tuple[float, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerSearchResult:
    graph: ModelGraph
    best: LayerTrainingPlan | None
    best_metrics: LayerMetrics | None
    front: tuple[tuple[LayerTrainingPlan, LayerMetrics], ...]
    baseline: tuple[LayerTrainingPlan, LayerMetrics]
    evaluated: int
    exhaustive: bool
    assumptions: tuple[str, ...]


def balanced_stages(node_count: int, stages: int) -> tuple[int, ...]:
    stages = max(1, min(stages, node_count))
    return tuple(min(stages - 1, (index * stages) // node_count) for index in range(node_count))


def repair_plan(graph: ModelGraph, plan: LayerTrainingPlan, *, locks: Mapping[str, Mapping[str, Any]] | None = None) -> LayerTrainingPlan:
    """Repair to contiguous non-empty stages and apply absolute per-node locks."""

    n = len(graph.nodes)
    if not (len(plan.stages) == len(plan.precision) == len(plan.recompute) == len(plan.offload) == n):
        raise ValueError("layer plan arrays must match graph node count")
    raw_stages = [max(0, int(value)) for value in plan.stages]
    transitions = [i for i in range(1, n) if raw_stages[i] != raw_stages[i - 1]]
    stage_count = max(1, len(transitions) + 1)
    stages = list(balanced_stages(n, stage_count))
    precision = [value if value in BYTES else "bf16" for value in plan.precision]
    recompute = [value if value in RECOMPUTE_FLOP_MULTIPLIER else "none" for value in plan.recompute]
    offload = [bool(value) for value in plan.offload]
    # Endpoint precision is conservative by default; an explicit lock can override it.
    precision[0] = "bf16"
    precision[-1] = "bf16"
    for index, node in enumerate(graph.nodes):
        lock = (locks or {}).get(node.id, {})
        if "stage" in lock:
            stages[index] = int(lock["stage"])
        if "precision" in lock:
            precision[index] = str(lock["precision"])
        if "recompute" in lock:
            recompute[index] = str(lock["recompute"])
        if "offload" in lock:
            offload[index] = bool(lock["offload"])
    # Locks may introduce gaps, so normalize stage labels once more while preserving order.
    labels: dict[int, int] = {}
    normalized: list[int] = []
    for value in stages:
        if value not in labels:
            labels[value] = len(labels)
        normalized.append(labels[value])
    if locks:
        # A locked stage boundary is absolute for its node; preserve boundaries
        # by using the locked label order, while still removing empty labels.
        normalized = list(stages)
        labels = {}
        normalized = [labels.setdefault(value, len(labels)) for value in normalized]
    return LayerTrainingPlan(tuple(normalized), tuple(precision), tuple(recompute), tuple(offload))


def evaluate_plan(problem: TrainingProblem, plan: LayerTrainingPlan, *, locks: Mapping[str, Mapping[str, Any]] | None = None) -> LayerMetrics:
    graph = problem.graph
    plan = repair_plan(graph, plan, locks=locks)
    hardware = problem.hardware
    stage_count = max(plan.stages) + 1
    stage_times = [0.0] * stage_count
    memory = [hardware.overhead_bytes] * stage_count
    notes: list[str] = []
    for node, precision, recompute, offload, stage in zip(graph.nodes, plan.precision, plan.recompute, plan.offload, plan.stages):
        speed = PRECISION_SPEEDUP[precision]
        stage_times[stage] += node.forward_flops * RECOMPUTE_FLOP_MULTIPLIER[recompute] / (hardware.peak_flops * speed)
        state_bytes = node.parameter_count * (1.0 if precision == "fp8" else 2.0 + 2.0 + 12.0)
        activation_bytes = node.activation_bytes
        if offload:
            memory[stage] += state_bytes + activation_bytes * 0.15
            transfer = activation_bytes * 0.85 / hardware.cpu_offload_bytes_s
            stage_times[stage] += transfer
            notes.append("CPU activation offload uses the supplied/default transfer bandwidth")
        else:
            memory[stage] += state_bytes + activation_bytes
        if precision == "fp8":
            notes.append("FP8 speed and training quality are assumptions until customer evaluation")
    if stage_count > 1:
        boundary_bytes = sum(node.activation_bytes for node in graph.nodes[:-1]) / max(1, stage_count)
        stage_times = [value + 2.0 * boundary_bytes / hardware.interconnect_bytes_s for value in stage_times]
    max_stage = max(stage_times, default=0.0)
    micro_batches = max(1, problem.micro_batches)
    step_time = max_stage * (micro_batches + stage_count - 1) / micro_batches
    cost = step_time / 3600.0 * hardware.devices * hardware.cost_per_device_hour
    capacity = hardware.memory_bytes * hardware.usable_memory
    constraints = {f"memory_stage_{i}": value / capacity - 1.0 for i, value in enumerate(memory)}
    constraints["stage_count"] = -1.0 if stage_count in problem.pipeline_stages or not problem.pipeline_stages else 1.0
    headroom = min((capacity - value) / capacity for value in memory)
    return LayerMetrics(
        objectives={"step_time_s": step_time, "cost_usd_per_step": cost, "memory_headroom": headroom},
        constraints=constraints, memory_by_stage=tuple(memory), stage_times=tuple(stage_times), notes=tuple(sorted(set(notes))),
    )


def _feasible(metrics: LayerMetrics) -> bool:
    return all(value <= 0.0 for value in metrics.constraints.values())


def _vector(metrics: LayerMetrics) -> tuple[float, float, float]:
    return (metrics.objectives["step_time_s"], metrics.objectives["cost_usd_per_step"], -metrics.objectives["memory_headroom"])


def _dominates(left: LayerMetrics, right: LayerMetrics) -> bool:
    a, b = _vector(left), _vector(right)
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _candidate_plans(problem: TrainingProblem, locks: Mapping[str, Mapping[str, Any]] | None) -> tuple[list[LayerTrainingPlan], bool]:
    n = len(problem.graph.nodes)
    precisions = ["bf16", "fp8"] if problem.fp8_enabled else ["bf16"]
    plans: list[LayerTrainingPlan] = []
    # Small graphs are exhaustive over the user-facing per-node switches.
    exhaustive = n <= 4
    if exhaustive:
        for stage_count in range(1, min(n, max(problem.pipeline_stages or (1,))) + 1):
            for precision in product(precisions, repeat=n):
                for recompute in product(("none", "selective"), repeat=n):
                    for offload in product((False, True), repeat=n):
                        plans.append(repair_plan(problem.graph, LayerTrainingPlan(
                            balanced_stages(n, stage_count), precision, recompute, offload), locks=locks))
                        if len(plans) >= problem.max_candidates:
                            return _dedup(plans), False
    else:
        for stage_count in sorted(set((1, *problem.pipeline_stages))):
            if stage_count <= n:
                stages = balanced_stages(n, stage_count)
                for precision in ("bf16", "fp8"):
                    for recompute in ("none", "selective", "full"):
                        for offload in (False, True):
                            plans.append(repair_plan(problem.graph, LayerTrainingPlan(
                                stages, tuple(precision for _ in range(n)), tuple(recompute for _ in range(n)), tuple(offload for _ in range(n))), locks=locks))
                # selective middle-node variants cover the common activation-heavy case
                middle = tuple(index not in (0, n - 1) for index in range(n))
                plans.append(repair_plan(problem.graph, LayerTrainingPlan(
                    stages, tuple("fp8" if middle[i] else "bf16" for i in range(n)),
                    tuple("selective" if middle[i] else "none" for i in range(n)), middle), locks=locks))
        # Coordinate mutations make the bounded search responsive for larger graphs.
        seed = list(plans)
        for base in seed:
            for index in range(n):
                for precision in precisions:
                    values = list(base.precision); values[index] = precision
                    plans.append(repair_plan(problem.graph, LayerTrainingPlan(base.stages, tuple(values), base.recompute, base.offload), locks=locks))
                values = list(base.recompute); values[index] = "none" if values[index] != "none" else "selective"
                plans.append(repair_plan(problem.graph, LayerTrainingPlan(base.stages, base.precision, tuple(values), base.offload), locks=locks))
                values_offload = list(base.offload); values_offload[index] = not values_offload[index]
                plans.append(repair_plan(problem.graph, LayerTrainingPlan(base.stages, base.precision, base.recompute, tuple(values_offload)), locks=locks))
                if len(plans) >= problem.max_candidates:
                    return _dedup(plans), False
    return _dedup(plans), exhaustive


def _dedup(plans: Iterable[LayerTrainingPlan]) -> list[LayerTrainingPlan]:
    seen: set[LayerTrainingPlan] = set()
    result: list[LayerTrainingPlan] = []
    for plan in plans:
        if plan not in seen:
            seen.add(plan)
            result.append(plan)
    return result


def global_baseline(problem: TrainingProblem, *, precision: str = "bf16", recompute: str = "none", offload: bool = False,
                    stages: int = 1, locks: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[LayerTrainingPlan, LayerMetrics]:
    n = len(problem.graph.nodes)
    plan = repair_plan(problem.graph, LayerTrainingPlan(
        balanced_stages(n, stages), tuple(precision for _ in range(n)),
        tuple(recompute for _ in range(n)), tuple(offload for _ in range(n))), locks=locks)
    return plan, evaluate_plan(problem, plan, locks=locks)


def search(problem: TrainingProblem, *, locks: Mapping[str, Mapping[str, Any]] | None = None) -> LayerSearchResult:
    candidates, exhaustive = _candidate_plans(problem, locks)
    evaluated = [(plan, evaluate_plan(problem, plan, locks=locks)) for plan in candidates]
    feasible = [(plan, metrics) for plan, metrics in evaluated if _feasible(metrics)]
    pool = feasible or sorted(evaluated, key=lambda item: sum(max(0.0, value) for value in item[1].constraints.values()))[:1]
    front = tuple(item for item in pool if not any(_dominates(other[1], item[1]) for other in pool if other != item))
    if not front:
        front = tuple(pool)
    ranges = []
    for index in range(3):
        values = [_vector(metrics)[index] for _plan, metrics in front]
        ranges.append((min(values, default=0.0), max(values, default=1.0)))
    def score(item: tuple[LayerTrainingPlan, LayerMetrics]) -> float:
        values = _vector(item[1])
        return sum((value - low) / (high - low if high > low else 1.0) for value, (low, high) in zip(values, ranges))
    best = min(front, key=score) if front else None
    baseline = global_baseline(problem, locks=locks)
    return LayerSearchResult(
        graph=problem.graph, best=best[0] if best else None, best_metrics=best[1] if best else None,
        front=front, baseline=baseline, evaluated=len(evaluated), exhaustive=exhaustive,
        assumptions=tuple(sorted(set(problem.graph.assumptions + ("FP8 quality and CPU offload transfer are assumptions",)))),
    )


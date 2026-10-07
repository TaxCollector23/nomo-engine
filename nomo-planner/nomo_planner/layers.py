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
import json
from math import inf
from typing import Any, Iterable, Mapping, Sequence


BYTES = {"bf16": 2.0, "fp8": 1.0}
PRECISION_SPEEDUP = {"bf16": 1.0, "fp8": 1.35}
RECOMPUTE_FLOP_MULTIPLIER = {"none": 1.0, "selective": 1.12, "full": 1.28}
EXHAUSTIVE_NODE_LIMIT = 4
NSGA_MAX_POPULATION = 64
NSGA_CROSSOVER_RATE = 0.90


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


def build_neuromorphic_manifest(graph: ModelGraph) -> dict[str, Any]:
    """Serialize the shared graph for the existing compiler boundary.

    This is deliberately a contract/export artifact, not a neuromorphic
    placement recommendation. NIR, C11, RTL, and chip validation remain owned
    by the established compiler dashboard.
    """
    return {
        "format": "nomo.graph/neuromorphic-preview/1",
        "status": "contract-only",
        "source": graph.source,
        "graph": {
            "name": graph.name,
            "seq_len": graph.seq_len,
            "batch_size": graph.batch_size,
            "hidden_size": graph.hidden_size,
            "layers": graph.layers,
            "attention_heads": graph.attention_heads,
            "kv_heads": graph.kv_heads,
            "intermediate_size": graph.intermediate_size,
            "vocab_size": graph.vocab_size,
            "tied_embeddings": graph.tied_embeddings,
            "gated_mlp": graph.gated_mlp,
            "experts": graph.experts,
            "experts_per_token": graph.experts_per_token,
            "parameter_count": graph.parameter_count,
            "nodes": [asdict(node) for node in graph.nodes],
        },
        "export_boundary": "NIR/C11/RTL exports remain owned by the validated compiler dashboard",
        "assumptions": list(graph.assumptions),
    }


def build_graph_from_contract(raw: Mapping[str, Any], *, seq_len: int | None = None,
                              batch_size: int | None = None, source: str | None = None) -> ModelGraph:
    """Re-import a Nomo graph contract without inferring omitted architecture facts."""
    candidate = raw.get("graph") if isinstance(raw.get("graph"), Mapping) else raw
    nodes_raw = candidate.get("nodes") if isinstance(candidate, Mapping) else None
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ValueError("graph JSON must contain a non-empty nodes list")
    nodes: list[GraphNode] = []
    for index, value in enumerate(nodes_raw, start=1):
        if not isinstance(value, Mapping):
            raise ValueError(f"graph JSON node {index} is not an object")
        kind = str(value.get("kind", ""))
        if kind not in {"embedding", "attention", "mlp", "output"}:
            raise ValueError(f"graph JSON node {index} has an unsupported kind")
        def number(*keys: str, default: float = 0.0) -> float:
            parsed = _first(value, *keys, default=default)
            try:
                result = float(parsed)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"graph JSON node {index} has invalid numeric accounting") from exc
            if result < 0:
                raise ValueError(f"graph JSON node {index} has negative accounting")
            return result
        layer = _first(value, "layer_index", "layerIndex")
        nodes.append(GraphNode(
            id=str(value.get("id", "")), kind=kind,
            layer_index=None if layer is None else int(layer),
            parameter_count=int(number("parameter_count", "parameterCount")),
            forward_flops=number("forward_flops", "forwardFlops"),
            activation_bytes=number("activation_bytes", "activationBytes"),
            kv_cache_bytes=number("kv_cache_bytes", "kvCacheBytes"),
            metadata=dict(value.get("metadata", {})) if isinstance(value.get("metadata"), Mapping) else {},
        ))
    def positive(*keys: str, default: int) -> int:
        value = _first(candidate, *keys, default=default)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("graph JSON has invalid architecture metadata") from exc
        return parsed if parsed > 0 else default
    assumptions = list(raw.get("assumptions", [])) if isinstance(raw.get("assumptions"), list) else []
    assumptions.append("Imported from a Nomo graph contract; missing architecture metadata is not inferred")
    return ModelGraph(
        name=str(candidate.get("name", "uploaded-graph")), source=str(raw.get("source", source or "graph-json")),
        seq_len=positive("seq_len", "seqLen", default=seq_len or 2048),
        batch_size=positive("batch_size", "batchSize", default=batch_size or 1),
        hidden_size=positive("hidden_size", "hiddenSize", default=1),
        layers=positive("layers", default=max(1, max((node.layer_index or -1) + 1 for node in nodes))),
        attention_heads=positive("attention_heads", "attentionHeads", default=1),
        kv_heads=positive("kv_heads", "kvHeads", default=1),
        intermediate_size=positive("intermediate_size", "intermediateSize", default=1),
        vocab_size=positive("vocab_size", "vocabSize", default=1),
        tied_embeddings=bool(_first(candidate, "tied_embeddings", "tiedEmbeddings", default=False)),
        gated_mlp=bool(_first(candidate, "gated_mlp", "gatedMlp", default=False)),
        experts=_first(candidate, "experts", default=None),
        experts_per_token=_first(candidate, "experts_per_token", "expertsPerToken", default=None),
        nodes=tuple(nodes), assumptions=tuple(sorted(set(map(str, assumptions)))),
    )


def build_graph(config: Mapping[str, Any], *, seq_len: int = 2048, batch_size: int = 1,
                source: str = "huggingface-config") -> ModelGraph:
    """Expand a transformer config into one ordered, decision-addressable graph."""

    if isinstance(config.get("nodes"), list) or isinstance(config.get("graph"), Mapping):
        return build_graph_from_contract(config, seq_len=seq_len, batch_size=batch_size, source=source)

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
    pipeline_stages: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    fp8_enabled: bool = True
    max_candidates: int = 20_000
    total_steps: int = 1_000
    seed: int = 20261003


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
    global_bf16: tuple[LayerTrainingPlan, LayerMetrics]
    global_best: tuple[LayerTrainingPlan, LayerMetrics]
    precision_gain_pct: float
    per_layer_gain_pct: float
    seed: int
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
    offload_transfer = 0.0
    communication_by_stage = [0.0] * stage_count
    for node, precision, recompute, offload, stage in zip(graph.nodes, plan.precision, plan.recompute, plan.offload, plan.stages):
        speed = PRECISION_SPEEDUP[precision]
        stage_times[stage] += node.forward_flops * RECOMPUTE_FLOP_MULTIPLIER[recompute] / (hardware.peak_flops * speed)
        state_bytes = node.parameter_count * (1.0 if precision == "fp8" else 2.0 + 2.0 + 12.0)
        activation_bytes = node.activation_bytes
        if offload:
            memory[stage] += state_bytes + activation_bytes * 0.15
            transfer = activation_bytes * 0.85 / hardware.cpu_offload_bytes_s
            stage_times[stage] += transfer
            offload_transfer += transfer
            notes.append("CPU activation offload uses the supplied/default transfer bandwidth")
        else:
            memory[stage] += state_bytes + activation_bytes
        if precision == "fp8":
            notes.append("FP8 speed and training quality are assumptions until customer evaluation")
    communication_bytes = 0.0
    if stage_count > 1:
        for index in range(1, len(graph.nodes)):
            left, right = plan.stages[index - 1], plan.stages[index]
            if left != right:
                bytes_at_boundary = graph.nodes[index - 1].activation_bytes
                communication_bytes += bytes_at_boundary
                transfer = bytes_at_boundary / hardware.interconnect_bytes_s
                communication_by_stage[left] += transfer
                communication_by_stage[right] += transfer
        stage_times = [value + communication_by_stage[index] for index, value in enumerate(stage_times)]
        notes.append("Inter-stage communication charges both adjacent stages at the supplied/default interconnect bandwidth")
    max_stage = max(stage_times, default=0.0)
    micro_batches = max(1, problem.micro_batches)
    pipeline_bubble = max_stage * max(0, stage_count - 1) / micro_batches
    step_time = max_stage + pipeline_bubble
    cost = step_time / 3600.0 * hardware.devices * hardware.cost_per_device_hour
    capacity = hardware.memory_bytes * hardware.usable_memory
    constraints = {f"memory_stage_{i}": value / capacity - 1.0 for i, value in enumerate(memory)}
    constraints["stage_count"] = -1.0 if stage_count in problem.pipeline_stages or not problem.pipeline_stages else 1.0
    headroom = min((capacity - value) / capacity for value in memory)
    return LayerMetrics(
        objectives={
            "step_time_s": step_time,
            "cost_usd_per_step": cost,
            "memory_headroom": headroom,
            "communication_s": sum(communication_by_stage),
            "pipeline_bubble_s": pipeline_bubble,
            "offload_transfer_s": offload_transfer,
            "whole_run_time_s": step_time * max(1, problem.total_steps),
            "whole_run_cost_usd": cost * max(1, problem.total_steps),
        },
        constraints=constraints, memory_by_stage=tuple(memory), stage_times=tuple(stage_times), notes=tuple(sorted(set(notes))),
    )


def _feasible(metrics: LayerMetrics) -> bool:
    return all(value <= 0.0 for value in metrics.constraints.values())


def _vector(metrics: LayerMetrics) -> tuple[float, float, float]:
    return (metrics.objectives["step_time_s"], metrics.objectives["cost_usd_per_step"], -metrics.objectives["memory_headroom"])


def _dominates(left: LayerMetrics, right: LayerMetrics) -> bool:
    a, b = _vector(left), _vector(right)
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _constraint_violation(metrics: LayerMetrics) -> float:
    return sum(max(0.0, value) for value in metrics.constraints.values())


def _constrained_dominates(left: LayerMetrics, right: LayerMetrics) -> bool:
    """NSGA-II comparison: feasible plans dominate infeasible plans first."""

    left_violation = _constraint_violation(left)
    right_violation = _constraint_violation(right)
    if left_violation <= 0.0 and right_violation > 0.0:
        return True
    if left_violation > 0.0 and right_violation <= 0.0:
        return False
    if left_violation > 0.0 or right_violation > 0.0:
        return left_violation < right_violation
    return _dominates(left, right)


def _plan_key(plan: LayerTrainingPlan) -> str:
    """JSON key shared with the TypeScript port for deterministic tie breaks."""

    return json.dumps(
        [list(plan.stages), list(plan.precision), list(plan.recompute), list(plan.offload)],
        separators=(",", ":"),
    )


class _DeterministicRng:
    """Small unsigned LCG whose arithmetic is mirrored in TypeScript."""

    def __init__(self, seed: int):
        self.state = int(seed) & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0

    def index(self, length: int) -> int:
        return min(length - 1, int(self.random() * length)) if length > 1 else 0


def _nondominated_sort(
    evaluated: Sequence[tuple[LayerTrainingPlan, LayerMetrics]],
) -> tuple[list[int], list[list[int]]]:
    """Return NSGA-II ranks/fronts with stable input-order tie handling."""

    count = len(evaluated)
    dominates_indices: list[list[int]] = [[] for _ in range(count)]
    dominated_by = [0] * count
    for left in range(count):
        for right in range(left + 1, count):
            if _constrained_dominates(evaluated[left][1], evaluated[right][1]):
                dominates_indices[left].append(right)
                dominated_by[right] += 1
            elif _constrained_dominates(evaluated[right][1], evaluated[left][1]):
                dominates_indices[right].append(left)
                dominated_by[left] += 1
    first = [index for index, count_below in enumerate(dominated_by) if count_below == 0]
    fronts: list[list[int]] = [first]
    rank = [-1] * count
    for index in first:
        rank[index] = 0
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[int] = []
        for left in fronts[front_index]:
            for right in dominates_indices[left]:
                dominated_by[right] -= 1
                if dominated_by[right] == 0:
                    rank[right] = front_index + 1
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return rank, fronts


def _crowding_distances(
    evaluated: Sequence[tuple[LayerTrainingPlan, LayerMetrics]],
    fronts: Sequence[Sequence[int]],
) -> list[float]:
    distances = [0.0] * len(evaluated)
    for front in fronts:
        if len(front) <= 2:
            for index in front:
                distances[index] = inf
            continue
        for objective in range(3):
            ordered = sorted(front, key=lambda index: (_vector(evaluated[index][1])[objective], index))
            distances[ordered[0]] = inf
            distances[ordered[-1]] = inf
            low = _vector(evaluated[ordered[0]][1])[objective]
            high = _vector(evaluated[ordered[-1]][1])[objective]
            if high <= low:
                continue
            for position in range(1, len(ordered) - 1):
                index = ordered[position]
                if distances[index] == inf:
                    continue
                previous_value = _vector(evaluated[ordered[position - 1]][1])[objective]
                next_value = _vector(evaluated[ordered[position + 1]][1])[objective]
                distances[index] += (next_value - previous_value) / (high - low)
    return distances


def _select_nsga(
    evaluated: Sequence[tuple[LayerTrainingPlan, LayerMetrics]],
    limit: int,
) -> list[tuple[LayerTrainingPlan, LayerMetrics]]:
    rank, fronts = _nondominated_sort(evaluated)
    del rank  # The fronts carry the same information for environmental selection.
    distances = _crowding_distances(evaluated, fronts)
    selected: list[tuple[LayerTrainingPlan, LayerMetrics]] = []
    for front in fronts:
        if len(selected) + len(front) <= limit:
            selected.extend(evaluated[index] for index in front)
            continue
        ordered = sorted(
            front,
            key=lambda index: (-distances[index], _plan_key(evaluated[index][0])),
        )
        selected.extend(evaluated[index] for index in ordered[: max(0, limit - len(selected))])
        break
    return selected


def _tournament(
    population: Sequence[tuple[LayerTrainingPlan, LayerMetrics]],
    rng: _DeterministicRng,
    ranks: Sequence[int],
    distances: Sequence[float],
) -> tuple[LayerTrainingPlan, LayerMetrics]:
    left = rng.index(len(population))
    right = rng.index(len(population))
    if ranks[left] < ranks[right]:
        winner = left
    elif ranks[right] < ranks[left]:
        winner = right
    elif distances[left] > distances[right]:
        winner = left
    elif distances[right] > distances[left]:
        winner = right
    else:
        winner = left if _plan_key(population[left][0]) <= _plan_key(population[right][0]) else right
    return population[winner]


def _structured_seed_plans(
    problem: TrainingProblem,
    locks: Mapping[str, Mapping[str, Any]] | None,
) -> list[LayerTrainingPlan]:
    """Return deterministic, physically meaningful anchors for the large search."""

    n = len(problem.graph.nodes)
    precisions = ["bf16", "fp8"] if problem.fp8_enabled else ["bf16"]
    plans: list[LayerTrainingPlan] = []
    stage_counts = sorted(set((1, *problem.pipeline_stages)))
    for stage_count in stage_counts:
        if stage_count > n:
            continue
        stages = balanced_stages(n, stage_count)
        for precision in precisions:
            for recompute in ("none", "selective", "full"):
                for offload in (False, True):
                    plans.append(repair_plan(problem.graph, LayerTrainingPlan(
                        stages, tuple(precision for _ in range(n)),
                        tuple(recompute for _ in range(n)), tuple(offload for _ in range(n)),
                    ), locks=locks))
        middle = tuple(index not in (0, n - 1) for index in range(n))
        plans.append(repair_plan(problem.graph, LayerTrainingPlan(
            stages, tuple("fp8" if middle[index] else "bf16" for index in range(n)),
            tuple("selective" if middle[index] else "none" for index in range(n)), middle,
        ), locks=locks))
    return _dedup(plans)


def _random_plan(
    problem: TrainingProblem,
    rng: _DeterministicRng,
    locks: Mapping[str, Mapping[str, Any]] | None,
) -> LayerTrainingPlan:
    n = len(problem.graph.nodes)
    precisions = ["bf16", "fp8"] if problem.fp8_enabled else ["bf16"]
    stage_counts = sorted(set((1, *problem.pipeline_stages))) or [1]
    stage_count = stage_counts[rng.index(len(stage_counts))]
    return repair_plan(problem.graph, LayerTrainingPlan(
        balanced_stages(n, stage_count),
        tuple(precisions[rng.index(len(precisions))] for _ in range(n)),
        tuple(("none", "selective", "full")[rng.index(3)] for _ in range(n)),
        tuple(rng.random() < 0.5 for _ in range(n)),
    ), locks=locks)


def _crossover_mutate(
    left: LayerTrainingPlan,
    right: LayerTrainingPlan,
    problem: TrainingProblem,
    rng: _DeterministicRng,
    locks: Mapping[str, Mapping[str, Any]] | None,
) -> LayerTrainingPlan:
    n = len(problem.graph.nodes)
    if n > 1 and rng.random() < NSGA_CROSSOVER_RATE:
        cut = 1 + rng.index(n - 1)
        stages = left.stages[:cut] + right.stages[cut:]
        precision = left.precision[:cut] + right.precision[cut:]
        recompute = left.recompute[:cut] + right.recompute[cut:]
        offload = left.offload[:cut] + right.offload[cut:]
    else:
        stages, precision, recompute, offload = left.stages, left.precision, left.recompute, left.offload
    precisions = ["bf16", "fp8"] if problem.fp8_enabled else ["bf16"]
    stage_limit = max(sorted(set((1, *problem.pipeline_stages))) or [1])
    stages = list(stages)
    precision = list(precision)
    recompute = list(recompute)
    offload = list(offload)
    changed = False
    mutation_rate = max(1.0 / max(1, n), 0.02)
    for index in range(n):
        if rng.random() < mutation_rate:
            stages[index] = rng.index(stage_limit)
            changed = True
        if rng.random() < mutation_rate:
            precision[index] = precisions[rng.index(len(precisions))]
            changed = True
        if rng.random() < mutation_rate:
            recompute[index] = ("none", "selective", "full")[rng.index(3)]
            changed = True
        if rng.random() < mutation_rate:
            offload[index] = rng.random() < 0.5
            changed = True
    if not changed:
        gene = rng.index(max(1, 4 * n))
        index, field = divmod(gene, 4)
        if field == 0:
            stages[index] = rng.index(stage_limit)
        elif field == 1:
            precision[index] = precisions[rng.index(len(precisions))]
        elif field == 2:
            recompute[index] = ("none", "selective", "full")[rng.index(3)]
        else:
            offload[index] = not offload[index]
    return repair_plan(problem.graph, LayerTrainingPlan(
        tuple(stages), tuple(precision), tuple(recompute), tuple(offload),
    ), locks=locks)


def _nsga_candidate_plans(
    problem: TrainingProblem,
    locks: Mapping[str, Mapping[str, Any]] | None,
) -> list[LayerTrainingPlan]:
    """Deterministic constrained NSGA-II candidate generation for larger graphs."""

    budget = max(1, int(problem.max_candidates))
    n = len(problem.graph.nodes)
    population_size = min(budget, max(8, min(NSGA_MAX_POPULATION, 2 * n)))
    rng = _DeterministicRng(problem.seed)
    evaluated: list[tuple[LayerTrainingPlan, LayerMetrics]] = []
    seen: set[str] = set()

    def add(plan: LayerTrainingPlan) -> bool:
        repaired = repair_plan(problem.graph, plan, locks=locks)
        key = _plan_key(repaired)
        if key in seen or len(evaluated) >= budget:
            return False
        seen.add(key)
        evaluated.append((repaired, evaluate_plan(problem, repaired, locks=locks)))
        return True

    for plan in _structured_seed_plans(problem, locks):
        if len(evaluated) >= population_size:
            break
        add(plan)
    attempts = 0
    while len(evaluated) < population_size and attempts < population_size * 20:
        add(_random_plan(problem, rng, locks))
        attempts += 1
    population = evaluated[:population_size]
    while population and len(evaluated) < budget:
        ranks, fronts = _nondominated_sort(population)
        distances = _crowding_distances(population, fronts)
        target = min(population_size, budget - len(evaluated))
        offspring: list[tuple[LayerTrainingPlan, LayerMetrics]] = []
        attempts = 0
        while len(offspring) < target and attempts < target * 30:
            left = _tournament(population, rng, ranks, distances)
            right = _tournament(population, rng, ranks, distances)
            child = _crossover_mutate(left[0], right[0], problem, rng, locks)
            key = _plan_key(child)
            if key not in seen:
                seen.add(key)
                record = (child, evaluate_plan(problem, child, locks=locks))
                evaluated.append(record)
                offspring.append(record)
            attempts += 1
        if not offspring:
            break
        population = _select_nsga([*population, *offspring], population_size)
    return [plan for plan, _metrics in evaluated]


def _candidate_plans(problem: TrainingProblem, locks: Mapping[str, Mapping[str, Any]] | None) -> tuple[list[LayerTrainingPlan], bool]:
    n = len(problem.graph.nodes)
    precisions = ["bf16", "fp8"] if problem.fp8_enabled else ["bf16"]
    plans: list[LayerTrainingPlan] = []
    # Small graphs are exhaustive over the user-facing per-node switches.
    exhaustive = n <= EXHAUSTIVE_NODE_LIMIT
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
        return _nsga_candidate_plans(problem, locks), False
    return _dedup(plans), exhaustive


def _dedup(plans: Iterable[LayerTrainingPlan]) -> list[LayerTrainingPlan]:
    seen: set[LayerTrainingPlan] = set()
    result: list[LayerTrainingPlan] = []
    for plan in plans:
        if plan not in seen:
            seen.add(plan)
            result.append(plan)
    return result


def _choose_global(problem: TrainingProblem, *, precision: str | None = None,
                   locks: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[LayerTrainingPlan, LayerMetrics]:
    n = len(problem.graph.nodes)
    choices: list[tuple[LayerTrainingPlan, LayerMetrics]] = []
    precisions = [precision] if precision is not None else (["bf16", "fp8"] if problem.fp8_enabled else ["bf16"])
    for stages in sorted(set((1, *problem.pipeline_stages))):
        if stages > n:
            continue
        for selected_precision in precisions:
            for recompute in RECOMPUTE_FLOP_MULTIPLIER:
                for offload in (False, True):
                    plan = repair_plan(problem.graph, LayerTrainingPlan(
                        balanced_stages(n, stages), tuple(selected_precision for _ in range(n)),
                        tuple(recompute for _ in range(n)), tuple(offload for _ in range(n))), locks=locks)
                    metrics = evaluate_plan(problem, plan, locks=locks)
                    if _feasible(metrics):
                        choices.append((plan, metrics))
    if not choices:
        # Preserve a diagnostic comparison for deliberately undersized test
        # hardware instead of hiding the infeasibility behind an exception.
        fallback: list[tuple[LayerTrainingPlan, LayerMetrics]] = []
        for stages in sorted(set((1, *problem.pipeline_stages))):
            if stages > n:
                continue
            for selected_precision in precisions:
                plan = repair_plan(problem.graph, LayerTrainingPlan(
                    balanced_stages(n, stages), tuple(selected_precision for _ in range(n)),
                    tuple("none" for _ in range(n)), tuple(False for _ in range(n))), locks=locks)
                fallback.append((plan, evaluate_plan(problem, plan, locks=locks)))
        return min(fallback, key=lambda item: sum(max(0.0, value) for value in item[1].constraints.values()))
    # Prefer the least expensive step time, then fewer stages. Offload has no
    # objective benefit when memory is already ample because its transfer time
    # is explicitly charged above.
    return min(choices, key=lambda item: (item[1].objectives["step_time_s"], len(set(item[0].stages)), item[1].objectives["cost_usd_per_step"]))


def global_baseline(problem: TrainingProblem, *, precision: str = "bf16", recompute: str = "none", offload: bool = False,
                    stages: int = 1, locks: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[LayerTrainingPlan, LayerMetrics]:
    """Compatibility helper for callers that need one explicit global plan."""
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
    if feasible:
        fastest = min(feasible, key=lambda item: (item[1].objectives["step_time_s"], item[1].objectives["cost_usd_per_step"]))
        minimum_stage = min(feasible, key=lambda item: (len(set(item[0].stages)), item[1].objectives["step_time_s"]))
        if (len(set(fastest[0].stages)) > len(set(minimum_stage[0].stages))
                and fastest[1].objectives["step_time_s"] < minimum_stage[1].objectives["step_time_s"] * 0.95):
            best = fastest
            stage_note = (f"{len(set(best[0].stages))} stages selected because the charged pipeline/bandwidth model "
                          f"improves step time by more than 5% over the minimum-memory-feasible stage count")
        else:
            best = minimum_stage
            stage_note = "minimum feasible stage count selected; extra stages did not earn a documented cost benefit"
    else:
        best = min(front, key=score) if front else None
        stage_note = "no feasible plan under the supplied hardware limits; showing the least-violating diagnostic"
    global_bf16 = _choose_global(problem, precision="bf16", locks=locks)
    global_best = _choose_global(problem, locks=locks)
    baseline = global_best
    precision_gain = (global_bf16[1].objectives["step_time_s"] - global_best[1].objectives["step_time_s"]) / global_bf16[1].objectives["step_time_s"] * 100.0
    per_layer_gain = (global_best[1].objectives["step_time_s"] - best[1].objectives["step_time_s"]) / global_best[1].objectives["step_time_s"] * 100.0 if best else 0.0
    search_note = ("Small graph search is exact exhaustive enumeration when the candidate cap is not reached"
                   if exhaustive else
                   "Large graph search uses deterministic repair-aware NSGA-II; the bounded Pareto set is not globally exhaustive")
    return LayerSearchResult(
        graph=problem.graph, best=best[0] if best else None, best_metrics=best[1] if best else None,
        front=front, baseline=baseline, global_bf16=global_bf16, global_best=global_best,
        precision_gain_pct=precision_gain, per_layer_gain_pct=per_layer_gain, seed=problem.seed,
        evaluated=len(evaluated), exhaustive=exhaustive,
        assumptions=tuple(sorted(set(problem.graph.assumptions + (
            "FP8 quality is an assumption until customer evaluation",
            "CPU activation offload uses the supplied/default PCIe/host bandwidth",
            "Pipeline bubble and inter-stage communication are charged per stage",
            search_note,
            stage_note,
        )))),
    )


"""Deterministic, dependency-free reference model for transformer training.

This is a transparent analytical simulator, not a benchmark harness.  Default
efficiencies are documented modeled assumptions; caller supplied measurements
can be attached to an efficiency curve with explicit provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, log2
from typing import Any, Mapping, Sequence

from .layers import build_graph


PRECISION_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}


@dataclass(frozen=True)
class Operator:
    id: str
    kind: str
    layer: int | None
    phase: str
    inputs: tuple[str, ...]
    shape: tuple[int, ...]
    flops: float
    read_bytes: float
    write_bytes: float
    live_from: int
    live_until: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def bytes_moved(self) -> float:
        return self.read_bytes + self.write_bytes


@dataclass(frozen=True)
class OperatorGraph:
    name: str
    source: str
    batch_size: int
    sequence_length: int
    hidden_size: int
    precision: str
    operators: tuple[Operator, ...]
    assumptions: tuple[str, ...] = ()

    @property
    def by_id(self) -> dict[str, Operator]:
        return {op.id: op for op in self.operators}


def build_operator_graph(config: Mapping[str, Any], *, sequence_length: int = 2048,
                         batch_size: int = 1, precision: str = "bf16",
                         source: str | None = None) -> OperatorGraph:
    """Build per-operation forward/backward graph using HF config fields."""
    if precision not in PRECISION_BYTES:
        raise ValueError(f"unsupported precision: {precision}")
    model = build_graph(config, seq_len=sequence_length, batch_size=batch_size,
                        source=source or str(config.get("source", "huggingface-config")))
    b, s, h, d = batch_size, sequence_length, model.hidden_size, PRECISION_BYTES[precision]
    t = b * s
    ops: list[Operator] = []

    def add(ident: str, kind: str, layer: int | None, phase: str, inputs: Sequence[str],
            shape: Sequence[int], flops: float, read: float, write: float,
            metadata: Mapping[str, Any] | None = None) -> str:
        index = len(ops)
        ops.append(Operator(ident, kind, layer, phase, tuple(inputs), tuple(map(int, shape)),
                            float(flops), float(read), float(write), index, index,
                            dict(metadata or {})))
        return ident

    current = add("embedding", "embedding", None, "forward", (), (b, s, h), 0,
                  t * h * d, t * h * d, {"vocab_size": model.vocab_size})
    for layer in range(model.layers):
        prefix = f"layer.{layer}"
        norm = add(f"{prefix}.norm1", "layer_norm", layer, "forward", (current,), (b, s, h),
                   5 * t * h, t * h * d, t * h * d)
        kv = model.kv_heads * (h // model.attention_heads)
        q = add(f"{prefix}.q_proj", "gemm", layer, "forward", (norm,), (b, s, h),
                2 * t * h * h, t * h * d + h * h * d, t * h * d,
                {"weight_shape": (h, h), "heads": model.attention_heads})
        k = add(f"{prefix}.k_proj", "gemm", layer, "forward", (norm,), (b, s, kv),
                2 * t * h * kv, t * h * d + h * kv * d, t * kv * d,
                {"weight_shape": (h, kv), "kv_heads": model.kv_heads})
        v = add(f"{prefix}.v_proj", "gemm", layer, "forward", (norm,), (b, s, kv),
                2 * t * h * kv, t * h * d + h * kv * d, t * kv * d,
                {"weight_shape": (h, kv), "kv_heads": model.kv_heads})
        attn = add(f"{prefix}.attention", "attention", layer, "forward", (q, k, v), (b, s, h),
                   4 * b * s * s * h, 2 * b * model.attention_heads * s * s * d,
                   t * h * d, {"heads": model.attention_heads, "causal": True})
        proj = add(f"{prefix}.o_proj", "gemm", layer, "forward", (attn,), (b, s, h),
                   2 * t * h * h, t * h * d + h * h * d, t * h * d,
                   {"weight_shape": (h, h)})
        norm2 = add(f"{prefix}.norm2", "layer_norm", layer, "forward", (proj,), (b, s, h),
                    5 * t * h, t * h * d, t * h * d)
        ffn = model.intermediate_size
        if model.experts:
            topk = int(model.experts_per_token or 1)
            router = add(f"{prefix}.router", "moe_router", layer, "forward", (norm2,),
                         (b, s, model.experts), 2 * t * h * model.experts,
                         t * h * d + h * model.experts * d, t * model.experts * d,
                         {"experts": model.experts, "top_k": topk})
            mlp = add(f"{prefix}.experts", "moe_experts", layer, "forward", (norm2, router),
                      (b, s, h), 2 * t * 3 * h * ffn * topk,
                      t * h * d + 3 * h * ffn * model.experts * d,
                      t * h * d, {"experts": model.experts, "active_experts": topk,
                                  "gated": model.gated_mlp})
        else:
            matrices = 3 if model.gated_mlp else 2
            mlp = add(f"{prefix}.mlp", "gated_mlp" if model.gated_mlp else "mlp",
                      layer, "forward", (norm2,), (b, s, ffn),
                      2 * t * matrices * h * ffn, t * h * d + matrices * h * ffn * d,
                      t * ffn * d, {"intermediate_size": ffn, "gated": model.gated_mlp})
        current = add(f"{prefix}.residual", "residual_add", layer, "forward", (proj, mlp),
                      (b, s, h), t * h, 2 * t * h * d, t * h * d)
    logits = add("lm_head", "gemm", None, "forward", (current,), (b, s, model.vocab_size),
                 2 * t * h * model.vocab_size,
                 t * h * d + h * model.vocab_size * d, t * model.vocab_size * d,
                 {"weight_shape": (h, model.vocab_size)})
    loss = add("cross_entropy", "loss", None, "forward", (logits,), (b, s),
               6 * t * model.vocab_size, t * model.vocab_size * d, t * d)

    # Reverse-mode operators are explicit counterparts; their FLOP multipliers
    # follow common dense-matmul accounting (backward input + weight gradients).
    forward = tuple(ops)
    grad = add("loss.backward", "loss_backward", None, "backward", (loss,), (b, s),
               6 * t * model.vocab_size, t * model.vocab_size * d, t * model.vocab_size * d)
    consumers: dict[str, list[str]] = {}
    for op in forward:
        for input_id in op.inputs:
            consumers.setdefault(input_id, []).append(op.id)
    for op in reversed(forward[:-1]):
        downstream = tuple(consumers.get(op.id, ()))
        # One explicit backward node for every forward operator preserves
        # dependency/lifetime visibility without pretending to choose kernels.
        mult = 2.0 if op.kind in {"gemm", "mlp", "gated_mlp", "moe_experts"} else 1.0
        grad = add(f"{op.id}.backward", f"{op.kind}_backward", op.layer, "backward",
                   (grad,) + downstream, op.shape, op.flops * mult,
                   op.write_bytes + op.read_bytes, op.write_bytes,
                   {"forward_operator": op.id})
    add("optimizer", "optimizer", None, "optimizer", (grad,), (model.hidden_size,),
        2 * sum(n.parameter_count for n in model.nodes),
        2 * sum(n.parameter_count for n in model.nodes) * d,
        2 * sum(n.parameter_count for n in model.nodes) * d,
        {"parameter_count": sum(n.parameter_count for n in model.nodes)})

    # Set last-consumer points as inclusive indices for activation lifetime.
    last_use = {op.id: op.live_from for op in ops}
    for index, op in enumerate(ops):
        for input_id in op.inputs:
            if input_id in last_use:
                last_use[input_id] = max(last_use[input_id], index)
    ops = [Operator(op.id, op.kind, op.layer, op.phase, op.inputs, op.shape,
                    op.flops, op.read_bytes, op.write_bytes, op.live_from,
                    max(op.live_from, last_use[op.id]), op.metadata) for op in ops]
    assumptions = list(model.assumptions) + [
        "Operator FLOPs and tensor traffic are analytical estimates from config dimensions",
        "Backward operators use explicit analytical multipliers; kernel fusion is not modeled",
    ]
    return OperatorGraph(model.name, model.source, b, s, h, precision, tuple(ops), tuple(assumptions))


@dataclass(frozen=True)
class EfficiencyPoint:
    operation: str
    precision: str
    min_m: int
    min_n: int
    min_k: int
    efficiency: float
    provenance: str


@dataclass(frozen=True)
class RooflineHardware:
    peak_flops: float
    memory_bandwidth: float
    compute_efficiency: float = 0.55
    memory_efficiency: float = 0.70
    curves: tuple[EfficiencyPoint, ...] = ()
    name: str = "user-specified hardware"


@dataclass(frozen=True)
class KernelEstimate:
    seconds: float
    compute_seconds: float
    memory_seconds: float
    arithmetic_intensity: float
    compute_efficiency: float
    memory_efficiency: float
    provenance: tuple[str, ...]


def roofline_time(op: Operator, hardware: RooflineHardware, *, precision: str = "bf16") -> KernelEstimate:
    if hardware.peak_flops <= 0 or hardware.memory_bandwidth <= 0:
        raise ValueError("hardware peak_flops and memory_bandwidth must be positive")
    if precision not in PRECISION_BYTES:
        raise ValueError(f"unsupported precision: {precision}")
    if not 0 < hardware.compute_efficiency <= 1 or not 0 < hardware.memory_efficiency <= 1:
        raise ValueError("efficiencies must be in (0, 1]")
    dims = tuple(op.metadata.get("weight_shape", (0, 0)))
    m, n = (op.shape[-2:] if len(op.shape) >= 2 else (0, 0))
    k = dims[0] if len(dims) == 2 else 0
    ce, me = hardware.compute_efficiency, hardware.memory_efficiency
    provenance = ["modeled default efficiency; not hardware validation data"]
    for point in hardware.curves:
        if (point.operation in {op.kind, "*"} and point.precision in {precision, "*"}
                and m >= point.min_m and n >= point.min_n and k >= point.min_k):
            if not 0 < point.efficiency <= 1:
                raise ValueError("efficiency curve values must be in (0, 1]")
            if op.kind in {"gemm", "mlp", "gated_mlp", "moe_experts"}:
                ce = point.efficiency
            else:
                me = point.efficiency
            provenance = [point.provenance]
    compute_s = op.flops / (hardware.peak_flops * ce)
    memory_s = op.bytes_moved / (hardware.memory_bandwidth * me)
    return KernelEstimate(max(compute_s, memory_s), compute_s, memory_s,
                          op.flops / max(op.bytes_moved, 1.0), ce, me,
                          tuple(provenance + [f"hardware={hardware.name}"]))


@dataclass(frozen=True)
class Link:
    bandwidth_bytes_s: float
    latency_s: float
    kind: str = "fabric"


@dataclass(frozen=True)
class Topology:
    devices: int
    gpus_per_node: int = 8
    intra_node: Link = Link(600e9, 2e-6, "NVLink/NVSwitch modeled")
    inter_node: Link = Link(50e9, 8e-6, "NIC/fabric modeled")
    cross_node_bandwidth: float | None = None

    def __post_init__(self) -> None:
        if self.devices <= 0 or self.gpus_per_node <= 0:
            raise ValueError("device counts must be positive")
        if self.intra_node.bandwidth_bytes_s <= 0 or self.inter_node.bandwidth_bytes_s <= 0:
            raise ValueError("link bandwidths must be positive")

    @property
    def nodes(self) -> int:
        return ceil(self.devices / self.gpus_per_node)


@dataclass(frozen=True)
class CollectiveEstimate:
    algorithm: str
    seconds: float
    bytes_transferred: float
    participants: int
    provenance: tuple[str, ...]


def collective_time(topology: Topology, participants: int, message_bytes: float,
                    collective: str = "all_reduce", algorithm: str = "auto") -> CollectiveEstimate:
    if participants <= 0 or participants > topology.devices or message_bytes < 0:
        raise ValueError("collective participants/message size are invalid")
    if collective not in {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}:
        raise ValueError("unsupported collective")
    chosen = algorithm
    if chosen == "auto":
        chosen = "hierarchical" if participants > topology.gpus_per_node else ("ring" if participants <= 8 else "tree")
    if chosen not in {"ring", "tree", "hierarchical"}:
        raise ValueError("algorithm must be auto, ring, tree, or hierarchical")
    levels = max(0, ceil(log2(participants)))
    inter = topology.inter_node if participants > topology.gpus_per_node else topology.intra_node
    if chosen == "ring":
        rounds = max(0, participants - 1)
        factor = 2 if collective == "all_reduce" else 1
        amount = factor * rounds / max(1, participants) * message_bytes
        seconds = rounds * inter.latency_s * factor + amount / inter.bandwidth_bytes_s
    elif chosen == "tree":
        rounds = levels
        amount = (2 if collective == "all_reduce" else 1) * message_bytes
        seconds = rounds * inter.latency_s * (2 if collective == "all_reduce" else 1) + amount / inter.bandwidth_bytes_s
    else:
        local_n = min(participants, topology.gpus_per_node)
        groups = ceil(participants / topology.gpus_per_node)
        intra_amount = 2 * (local_n - 1) / max(1, local_n) * message_bytes
        inter_amount = (2 if collective == "all_reduce" else 1) * message_bytes
        seconds = (2 * max(0, local_n - 1) * topology.intra_node.latency_s
                   + 2 * max(0, groups - 1) * topology.inter_node.latency_s
                   + intra_amount / topology.intra_node.bandwidth_bytes_s
                   + inter_amount / topology.inter_node.bandwidth_bytes_s)
        amount = intra_amount + inter_amount
    return CollectiveEstimate(chosen, seconds, amount, participants,
                              ("alpha-beta analytical model", "link rates are supplied or modeled defaults; not measured"))


@dataclass(frozen=True)
class Parallelism:
    data: int = 1
    tensor: int = 1
    pipeline: int = 1

    @property
    def world_size(self) -> int:
        return self.data * self.tensor * self.pipeline


@dataclass(frozen=True)
class TrainingOptions:
    micro_batches: int = 4
    schedule: str = "1f1b"
    precision: str = "bf16"
    recompute: bool = False
    offload: bool = False
    parallelism: Parallelism = Parallelism()
    data_collective: str = "all_reduce"
    tensor_collective: str = "all_reduce"
    sequence_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1
    zero_stage: int = 0
    fsdp: bool = False


@dataclass(frozen=True)
class GanttEvent:
    id: str
    gpu: int
    stream: str
    start_s: float
    end_s: float
    kind: str
    layer: int | None
    micro_batch: int
    dependencies: tuple[str, ...]
    bytes_moved: float = 0


@dataclass(frozen=True)
class MemoryPoint:
    time_s: float
    gpu: int
    allocated_bytes: float
    event: str


@dataclass(frozen=True)
class TrainingSimulation:
    step_time_s: float
    events: tuple[GanttEvent, ...]
    memory_timeline: tuple[MemoryPoint, ...]
    peak_memory_by_gpu: tuple[float, ...]
    assumptions: tuple[str, ...]
    provenance: tuple[str, ...]


def simulate_training_step(graph: OperatorGraph, hardware: RooflineHardware, topology: Topology,
                           options: TrainingOptions = TrainingOptions()) -> TrainingSimulation:
    """Schedule deterministic microbatch forward/backward work and collectives.

    Per-operation compute runs on a compute stream; collectives/offload run on
    a separate communication stream. Dependencies constrain overlap. The
    supported pipeline schedules are GPipe and a deterministic 1F1B ordering.
    """
    p = options.parallelism
    if options.micro_batches <= 0 or p.world_size > topology.devices:
        raise ValueError("micro_batches must be positive and parallelism fit the topology")
    schedule = options.schedule.lower().replace("-", "_")
    if schedule not in {"gpipe", "1f1b", "interleaved", "zero_bubble"}:
        raise ValueError("schedule must be gpipe, 1f1b, interleaved, or zero-bubble")
    if options.precision not in PRECISION_BYTES:
        raise ValueError("unsupported precision")
    for name in ("sequence_parallel", "context_parallel", "expert_parallel"):
        if getattr(options, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if options.zero_stage not in {0, 1, 2, 3}:
        raise ValueError("zero_stage must be 0, 1, 2, or 3")
    # Layer index determines pipeline stage; each stage's local GPU group is
    # replicated over data parallel ranks and tensor-parallel ranks.
    stage_count = p.pipeline
    layer_count = max(1, max((o.layer for o in graph.operators if o.layer is not None), default=-1) + 1)
    stage_of = {layer: min(stage_count - 1, layer * stage_count // layer_count)
                for layer in range(layer_count)}
    gpu_count = p.world_size
    compute_ready = [0.0] * gpu_count
    comm_ready = [0.0] * gpu_count
    events: list[GanttEvent] = []
    finish: dict[str, float] = {}
    event_ids: dict[str, str] = {}
    live_memory = [0.0] * gpu_count
    peak_memory = [0.0] * gpu_count
    memory_timeline: list[MemoryPoint] = []
    op_map = graph.by_id
    fwd = [o for o in graph.operators if o.phase == "forward"]
    bwd = [o for o in graph.operators if o.phase == "backward"]
    layers = max(1, max((o.layer for o in graph.operators if o.layer is not None), default=-1) + 1)

    def owner(layer: int | None) -> int:
        stage = 0 if layer is None else stage_of.get(layer, 0)
        return stage * p.tensor

    def launch(key: str, op: Operator | None, gpu: int, stream: str, deps: Sequence[str],
               duration: float, kind: str, layer: int | None, mb: int, moved: float = 0) -> str:
        start = max([compute_ready[gpu] if stream == "compute" else comm_ready[gpu]]
                    + [finish.get(dep, 0.0) for dep in deps])
        end = start + duration
        event_id = key
        events.append(GanttEvent(event_id, gpu, stream, start, end, kind, layer, mb,
                                 tuple(deps), moved))
        if stream == "compute":
            compute_ready[gpu] = end
        else:
            comm_ready[gpu] = end
        finish[key] = end
        event_ids[key] = event_id
        if op is not None:
            alloc = op.write_bytes
            live_memory[gpu] += alloc
            peak_memory[gpu] = max(peak_memory[gpu], live_memory[gpu])
            memory_timeline.append(MemoryPoint(end, gpu, live_memory[gpu], event_id))
            # Release temporary outputs after their final graph consumer, plus
            # backward use. This is a conservative operator-level live set.
            for source_op in graph.operators:
                if source_op.id == op.id:
                    continue
                if op.id in source_op.inputs and source_op.phase == "backward":
                    live_memory[gpu] = max(0.0, live_memory[gpu] - source_op.write_bytes)
        return key

    generated_fwd: dict[tuple[int, str], str] = {}
    generated_bwd: dict[tuple[int, str], str] = {}
    # Forward pass order differs for GPipe and 1F1B to express scheduling
    # policy while retaining exact graph dependencies within each microbatch.
    fwd_order = [(mb, op) for mb in range(options.micro_batches) for op in fwd]
    if schedule in {"1f1b", "interleaved", "zero_bubble"}:
        # Input/embedding work must precede the first layer.  The old
        # layer-first ordering silently dropped that dependency for 1F1B and
        # made the rendered timeline look plausible while starting layer 0 at
        # time zero.  Keep the prelude, stage work, and tail explicit.
        prelude = [op for op in fwd if op.id == "embedding" or (op.layer is None and not op.inputs)]
        tail = [op for op in fwd if op.layer is None and op not in prelude]
        fwd_order = [(mb, op) for mb in range(options.micro_batches) for op in prelude]
        fwd_order += [(mb, op) for layer in range(layers) for mb in range(options.micro_batches)
                      for op in fwd if op.layer == layer]
        fwd_order += [(mb, op) for mb in range(options.micro_batches) for op in tail]
    for mb, op in fwd_order:
        gpu = owner(op.layer)
        deps = [generated_fwd[(mb, source)] for source in op.inputs if (mb, source) in generated_fwd]
        if op.layer is not None and op.layer > 0:
            boundary = f"layer.{op.layer - 1}.residual"
            if (mb, boundary) in generated_fwd:
                boundary_key = generated_fwd[(mb, boundary)]
                if stage_of[op.layer - 1] != stage_of[op.layer]:
                    transfer_key = f"pipeline.mb{mb}.layer{op.layer - 1}.to{op.layer}"
                    if transfer_key not in finish:
                        payload = graph.batch_size * graph.sequence_length * graph.hidden_size * PRECISION_BYTES[options.precision]
                        link = topology.intra_node if owner(op.layer - 1) // topology.gpus_per_node == gpu // topology.gpus_per_node else topology.inter_node
                        launch(transfer_key, None, owner(op.layer - 1), "communication",
                               [boundary_key], link.latency_s + payload / link.bandwidth_bytes_s,
                               "pipeline_transfer", op.layer - 1, mb, payload)
                    boundary_key = transfer_key
                deps = [dep for dep in deps if dep != generated_fwd[(mb, boundary)]]
                if boundary_key not in deps:
                    deps.append(boundary_key)
        estimate = roofline_time(op, hardware, precision=options.precision)
        duration = estimate.seconds / p.tensor
        if options.recompute and op.kind not in {"embedding", "loss", "optimizer"}:
            duration *= 1.12
        key = f"fwd.mb{mb}.{op.id}"
        launch(key, op, gpu, "compute", deps, duration, op.kind, op.layer, mb, op.bytes_moved)
        generated_fwd[(mb, op.id)] = key
    # Backward depends on its matching forward and reverse consumers. Keeping
    # a per-microbatch backward stream order makes the simulation deterministic.
    if schedule == "gpipe":
        bwd_order = [(mb, op) for mb in range(options.micro_batches) for op in bwd]
    else:
        # Reverse-stage priority models the drain/ramp direction of 1F1B.
        bwd_order = sorted(((mb, op) for mb in range(options.micro_batches) for op in bwd),
                           key=lambda item: (-(stage_of.get(item[1].layer, 0)), item[0],
                                             item[1].id))
    for mb, op in bwd_order:
        forward_id = str(op.metadata.get("forward_operator", ""))
        deps = [generated_fwd[(mb, forward_id)]] if (mb, forward_id) in generated_fwd else []
        # Chain reverse pass by graph order, within the same microbatch.
        previous = next((key for (m, _), key in reversed(tuple(generated_bwd.items())) if m == mb), None)
        if previous:
            deps.append(previous)
        gpu = owner(op.layer)
        duration = roofline_time(op, hardware, precision=options.precision).seconds / p.tensor
        if options.recompute:
            duration *= 1.12
        key = f"bwd.mb{mb}.{op.id}"
        launch(key, op, gpu, "compute", deps, duration, op.kind, op.layer, mb, op.bytes_moved)
        generated_bwd[(mb, op.id)] = key
    # Gradient synchronization and optional activation transfers are explicit
    # communication events and therefore may overlap independent computation.
    end_compute = max(compute_ready, default=0.0)
    comm_keys: list[str] = []
    parameter_bytes = sum(int(op.metadata.get("parameter_count", 0)) for op in graph.operators) * PRECISION_BYTES[options.precision]
    if p.data > 1:
        estimate = collective_time(topology, p.data, parameter_bytes, options.data_collective)
        for rank in range(p.data):
            gpu = rank * p.tensor
            key = f"data_collective.rank{rank}"
            deps = [generated_bwd[(options.micro_batches - 1, bwd[-1].id)]] if bwd else []
            launch(key, None, gpu, "communication", deps, estimate.seconds, "collective", None,
                   options.micro_batches - 1, estimate.bytes_transferred)
            comm_keys.append(key)
    if options.sequence_parallel > 1:
        estimate = collective_time(topology, options.sequence_parallel,
                                   graph.sequence_length * graph.hidden_size * PRECISION_BYTES[options.precision],
                                   "all_gather", "ring")
        key = "sequence_parallel.collective"
        dependency = generated_fwd.get((options.micro_batches - 1, fwd[-1].id)) if fwd else None
        launch(key, None, owner(None), "communication", [dependency] if dependency else [], estimate.seconds,
               "sequence_parallel", None, options.micro_batches - 1, estimate.bytes_transferred)
        comm_keys.append(key)
    if options.context_parallel > 1:
        estimate = collective_time(topology, options.context_parallel,
                                   graph.sequence_length * graph.hidden_size * PRECISION_BYTES[options.precision],
                                   "all_gather", "ring")
        key = "context_parallel.collective"
        dependency = generated_fwd.get((options.micro_batches - 1, fwd[-1].id)) if fwd else None
        launch(key, None, owner(None), "communication", [dependency] if dependency else [], estimate.seconds,
               "context_parallel", None, options.micro_batches - 1, estimate.bytes_transferred)
        comm_keys.append(key)
    if options.expert_parallel > 1:
        estimate = collective_time(topology, options.expert_parallel,
                                   graph.sequence_length * graph.hidden_size * PRECISION_BYTES[options.precision],
                                   "all_to_all", "ring")
        key = "expert_parallel.collective"
        dependency = generated_fwd.get((options.micro_batches - 1, fwd[-1].id)) if fwd else None
        launch(key, None, owner(None), "communication", [dependency] if dependency else [], estimate.seconds,
               "expert_parallel", None, options.micro_batches - 1, estimate.bytes_transferred)
        comm_keys.append(key)
    if p.tensor > 1:
        for layer in range(layers):
            estimate = collective_time(topology, p.tensor, graph.sequence_length * graph.hidden_size * PRECISION_BYTES[options.precision],
                                       options.tensor_collective, "ring")
            for mb in range(options.micro_batches):
                gpu = owner(layer)
                key = f"tensor_collective.layer{layer}.mb{mb}"
                dep = generated_fwd.get((mb, next((o.id for o in fwd if o.layer == layer), "")))
                launch(key, None, gpu, "communication", [dep] if dep else [], estimate.seconds,
                       "collective", layer, mb, estimate.bytes_transferred)
                comm_keys.append(key)
    if options.offload:
        for mb in range(options.micro_batches):
            for layer in range(layers):
                gpu = owner(layer)
                byte_count = graph.batch_size * graph.sequence_length * graph.hidden_size * PRECISION_BYTES[options.precision]
                duration = byte_count / max(hardware.memory_bandwidth, 1.0)
                key = f"offload.layer{layer}.mb{mb}"
                dep = generated_fwd.get((mb, f"layer.{layer}.residual"))
                launch(key, None, gpu, "communication", [dep] if dep else [], duration,
                       "offload", layer, mb, byte_count)
                comm_keys.append(key)
    events.sort(key=lambda e: (e.start_s, e.gpu, e.stream, e.id))
    step_time = max((e.end_s for e in events), default=end_compute)
    if options.zero_stage or options.fsdp:
        shard_factor = max(1, p.data)
        parameter_bytes = parameter_bytes / shard_factor if options.zero_stage >= 1 or options.fsdp else parameter_bytes
    assumptions = list(graph.assumptions) + [
        "Compute kernels use max(compute roofline, memory roofline) with supplied peak rates",
        "Tensor/data/pipeline partitioning and communication are analytical approximations",
        "Recompute applies a 12% modeled compute multiplier; offload uses memory-bandwidth transfer rate",
        "Per-GPU memory timeline tracks operator output allocations and is a lower-level estimate, not allocator tracing",
        f"Schedule policy: {schedule}; interleaved and zero-bubble use deterministic stage-aware event ordering, not a runtime scheduler",
        f"Memory sharding: ZeRO stage {options.zero_stage}{' / FSDP' if options.fsdp else ''}; shard factors are analytical",
        f"Sequence/context/expert parallel factors: {options.sequence_parallel}/{options.context_parallel}/{options.expert_parallel}",
    ]
    provenance = sorted({item for op in graph.operators
                         for item in roofline_time(op, hardware, precision=options.precision).provenance})
    return TrainingSimulation(step_time, tuple(events), tuple(sorted(memory_timeline,
                             key=lambda m: (m.time_s, m.gpu, m.event))), tuple(peak_memory),
                              tuple(sorted(set(assumptions))), tuple(provenance))

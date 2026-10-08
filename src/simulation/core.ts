/**
 * Python ``nomo_planner.simcore`` parity surface.
 *
 * The types deliberately stay close to the dependency-free reference module:
 * operators carry shape/FLOP/traffic/lifetime data, topology uses alpha-beta
 * links, and the training scheduler emits deterministic Gantt events.  This
 * is an analytic model, not a hardware measurement or a server calibration.
 */

export const PRECISION_BYTES = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
} as const;

export type CorePrecision = keyof typeof PRECISION_BYTES;

export interface Operator {
  id: string;
  kind: string;
  layer: number | null;
  phase: string;
  inputs: readonly string[];
  shape: readonly number[];
  flops: number;
  readBytes: number;
  writeBytes: number;
  liveFrom: number;
  liveUntil: number;
  metadata: Readonly<Record<string, unknown>>;
  bytesMoved: number;
}

export interface OperatorGraph {
  name: string;
  source: string;
  batchSize: number;
  sequenceLength: number;
  hiddenSize: number;
  precision: CorePrecision;
  operators: readonly Operator[];
  assumptions: readonly string[];
  byId: Readonly<Record<string, Operator>>;
}

interface RecordLike {
  [key: string]: unknown;
}

function record(value: unknown): RecordLike {
  return typeof value === "object" && value !== null ? value as RecordLike : {};
}

function first(value: RecordLike, ...keys: string[]): unknown {
  for (const key of keys) if (value[key] !== undefined && value[key] !== null && value[key] !== "") return value[key];
  return undefined;
}

function positiveInt(value: unknown, label: string, fallback?: number): number {
  const parsed = Number(value ?? fallback);
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`${label} must be positive`);
  return Math.floor(parsed);
}

function corePrecision(value: unknown): CorePrecision {
  const candidate = String(value ?? "bf16").toLowerCase() as CorePrecision;
  if (!(candidate in PRECISION_BYTES)) throw new Error(`unsupported precision: ${candidate}`);
  return candidate;
}

function makeOperator(
  id: string,
  kind: string,
  layer: number | null,
  phase: string,
  inputs: readonly string[],
  shape: readonly number[],
  flops: number,
  readBytes: number,
  writeBytes: number,
  liveFrom: number,
  metadata: Readonly<Record<string, unknown>> = {},
): Operator {
  return {
    id,
    kind,
    layer,
    phase,
    inputs: [...inputs],
    shape: shape.map((dimension) => Math.floor(dimension)),
    flops,
    readBytes,
    writeBytes,
    liveFrom,
    liveUntil: liveFrom,
    metadata,
    bytesMoved: readBytes + writeBytes,
  };
}

function modelConfig(config: RecordLike): {
  name: string;
  layers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  ffn: number;
  vocab: number;
  gated: boolean;
  experts: number | null;
  expertsPerToken: number | null;
  tied: boolean;
  assumptions: string[];
} {
  const layers = positiveInt(first(config, "num_hidden_layers", "n_layer", "num_layers"), "layers");
  const hidden = positiveInt(first(config, "hidden_size", "n_embd", "d_model"), "hidden size");
  const heads = positiveInt(first(config, "num_attention_heads", "n_head", "num_heads"), "attention heads");
  if (hidden % heads) throw new Error("hidden_size must be divisible by num_attention_heads");
  const kvHeads = positiveInt(first(config, "num_key_value_heads", "n_kv_heads") ?? heads, "KV heads");
  if (heads % kvHeads) throw new Error("num_key_value_heads must divide num_attention_heads");
  const ffn = positiveInt(first(config, "intermediate_size", "n_inner") ?? 4 * hidden, "intermediate size");
  const vocab = positiveInt(first(config, "vocab_size", "n_vocab"), "vocabulary size");
  const tied = Boolean(first(config, "tie_word_embeddings", "tie_embeddings") ?? false);
  const gated = Boolean(first(config, "gated_mlp", "gatedMlp") ?? false) || ["silu", "swish", "geglu"].includes(String(first(config, "hidden_act", "hiddenAct") ?? "").toLowerCase());
  const expertsRaw = first(config, "num_local_experts", "num_experts", "experts");
  const experts = expertsRaw === undefined ? null : positiveInt(expertsRaw, "experts");
  const topKRaw = first(config, "num_experts_per_tok", "num_experts_per_token", "moe_top_k", "experts_per_token");
  const expertsPerToken = topKRaw === undefined ? null : positiveInt(topKRaw, "experts per token");
  if (experts !== null && (expertsPerToken === null || expertsPerToken > experts)) throw new Error("experts per token must be <= experts");
  const assumptions: string[] = [];
  if (first(config, "intermediate_size", "n_inner") === undefined) assumptions.push("intermediate_size inferred as 4 * hidden_size");
  if (first(config, "num_key_value_heads", "n_kv_heads") === undefined) assumptions.push("num_key_value_heads inferred as num_attention_heads");
  if (experts !== null) assumptions.push("MoE parameter memory includes resident experts; FLOPs use top-k active experts");
  return { name: String(first(config, "_name_or_path", "model_type") ?? "uploaded-transformer"), layers, hidden, heads, kvHeads, ffn, vocab, gated, experts, expertsPerToken, tied, assumptions };
}

export interface BuildCoreGraphOptions {
  sequenceLength?: number;
  batchSize?: number;
  precision?: CorePrecision;
  source?: string;
}

export function buildOperatorGraph(configValue: unknown, options: BuildCoreGraphOptions = {}): OperatorGraph {
  const config = record(configValue);
  const precision = corePrecision(options.precision ?? first(config, "precision", "dtype") ?? "bf16");
  const sequenceLength = positiveInt(options.sequenceLength ?? first(config, "sequence_length", "sequenceLength", "seq_len", "seqLen") ?? 2048, "sequence length");
  const batchSize = positiveInt(options.batchSize ?? first(config, "batch_size", "batchSize") ?? 1, "batch size");
  const model = modelConfig(config);
  const bytes = PRECISION_BYTES[precision];
  const tokens = batchSize * sequenceLength;
  const ops: Operator[] = [];
  const add = (id: string, kind: string, layer: number | null, phase: string, inputs: readonly string[], shape: readonly number[], flops: number, readBytes: number, writeBytes: number, metadata: Readonly<Record<string, unknown>> = {}): string => {
    ops.push(makeOperator(id, kind, layer, phase, inputs, shape, flops, readBytes, writeBytes, ops.length, metadata));
    return id;
  };
  let current = add("embedding", "embedding", null, "forward", [], [batchSize, sequenceLength, model.hidden], 0, tokens * model.hidden * bytes, tokens * model.hidden * bytes, { vocabSize: model.vocab });
  for (let layer = 0; layer < model.layers; layer += 1) {
    const prefix = `layer.${layer}`;
    const norm = add(`${prefix}.norm1`, "layer_norm", layer, "forward", [current], [batchSize, sequenceLength, model.hidden], 5 * tokens * model.hidden, tokens * model.hidden * bytes, tokens * model.hidden * bytes);
    const kv = model.kvHeads * (model.hidden / model.heads);
    const q = add(`${prefix}.q_proj`, "gemm", layer, "forward", [norm], [batchSize, sequenceLength, model.hidden], 2 * tokens * model.hidden * model.hidden, tokens * model.hidden * bytes + model.hidden * model.hidden * bytes, tokens * model.hidden * bytes, { weight_shape: [model.hidden, model.hidden], heads: model.heads });
    const k = add(`${prefix}.k_proj`, "gemm", layer, "forward", [norm], [batchSize, sequenceLength, kv], 2 * tokens * model.hidden * kv, tokens * model.hidden * bytes + model.hidden * kv * bytes, tokens * kv * bytes, { weight_shape: [model.hidden, kv], kv_heads: model.kvHeads });
    const v = add(`${prefix}.v_proj`, "gemm", layer, "forward", [norm], [batchSize, sequenceLength, kv], 2 * tokens * model.hidden * kv, tokens * model.hidden * bytes + model.hidden * kv * bytes, tokens * kv * bytes, { weight_shape: [model.hidden, kv], kv_heads: model.kvHeads });
    const attention = add(`${prefix}.attention`, "attention", layer, "forward", [q, k, v], [batchSize, sequenceLength, model.hidden], 4 * batchSize * sequenceLength * sequenceLength * model.hidden, 2 * batchSize * model.heads * sequenceLength * sequenceLength * bytes, tokens * model.hidden * bytes, { heads: model.heads, causal: true });
    const projection = add(`${prefix}.o_proj`, "gemm", layer, "forward", [attention], [batchSize, sequenceLength, model.hidden], 2 * tokens * model.hidden * model.hidden, tokens * model.hidden * bytes + model.hidden * model.hidden * bytes, tokens * model.hidden * bytes, { weight_shape: [model.hidden, model.hidden] });
    const norm2 = add(`${prefix}.norm2`, "layer_norm", layer, "forward", [projection], [batchSize, sequenceLength, model.hidden], 5 * tokens * model.hidden, tokens * model.hidden * bytes, tokens * model.hidden * bytes);
    let mlp: string;
    if (model.experts !== null) {
      const topK = model.expertsPerToken ?? 1;
      const router = add(`${prefix}.router`, "moe_router", layer, "forward", [norm2], [batchSize, sequenceLength, model.experts], 2 * tokens * model.hidden * model.experts, tokens * model.hidden * bytes + model.hidden * model.experts * bytes, tokens * model.experts * bytes, { experts: model.experts, top_k: topK });
      mlp = add(`${prefix}.experts`, "moe_experts", layer, "forward", [norm2, router], [batchSize, sequenceLength, model.hidden], 2 * tokens * 3 * model.hidden * model.ffn * topK, tokens * model.hidden * bytes + 3 * model.hidden * model.ffn * model.experts * bytes, tokens * model.hidden * bytes, { experts: model.experts, active_experts: topK, gated: model.gated });
    } else {
      const matrices = model.gated ? 3 : 2;
      mlp = add(`${prefix}.mlp`, model.gated ? "gated_mlp" : "mlp", layer, "forward", [norm2], [batchSize, sequenceLength, model.ffn], 2 * tokens * matrices * model.hidden * model.ffn, tokens * model.hidden * bytes + matrices * model.hidden * model.ffn * bytes, tokens * model.ffn * bytes, { intermediate_size: model.ffn, gated: model.gated });
    }
    current = add(`${prefix}.residual`, "residual_add", layer, "forward", [projection, mlp], [batchSize, sequenceLength, model.hidden], tokens * model.hidden, 2 * tokens * model.hidden * bytes, tokens * model.hidden * bytes);
  }
  const logits = add("lm_head", "gemm", null, "forward", [current], [batchSize, sequenceLength, model.vocab], 2 * tokens * model.hidden * model.vocab, tokens * model.hidden * bytes + model.hidden * model.vocab * bytes, tokens * model.vocab * bytes, { weight_shape: [model.hidden, model.vocab] });
  const loss = add("cross_entropy", "loss", null, "forward", [logits], [batchSize, sequenceLength], 6 * tokens * model.vocab, tokens * model.vocab * bytes, tokens * bytes);
  const forward = [...ops];
  let gradient = add("loss.backward", "loss_backward", null, "backward", [loss], [batchSize, sequenceLength], 6 * tokens * model.vocab, tokens * model.vocab * bytes, tokens * model.vocab * bytes);
  const consumers: Record<string, string[]> = {};
  for (const op of forward) for (const input of op.inputs) (consumers[input] ??= []).push(op.id);
  for (const op of [...forward.slice(0, -1)].reverse()) {
    const downstream = consumers[op.id] ?? [];
    const multiplier = ["gemm", "mlp", "gated_mlp", "moe_experts"].includes(op.kind) ? 2 : 1;
    gradient = add(`${op.id}.backward`, `${op.kind}_backward`, op.layer, "backward", [gradient, ...downstream], op.shape, op.flops * multiplier, op.writeBytes + op.readBytes, op.writeBytes, { forward_operator: op.id });
  }
  const parameterCount = model.layers * (model.hidden * model.hidden + 2 * model.hidden * (model.kvHeads * (model.hidden / model.heads)) + model.hidden * model.hidden + (model.gated ? 3 : 2) * model.hidden * model.ffn * (model.experts ?? 1)) + model.vocab * model.hidden * (model.tied ? 1 : 2);
  add("optimizer", "optimizer", null, "optimizer", [gradient], [model.hidden], 2 * parameterCount, 2 * parameterCount * bytes, 2 * parameterCount * bytes, { parameter_count: parameterCount });
  const lastUse = new Map(ops.map((op) => [op.id, op.liveFrom]));
  ops.forEach((op, index) => op.inputs.forEach((input) => {
    if (lastUse.has(input)) lastUse.set(input, Math.max(lastUse.get(input) ?? index, index));
  }));
  const materialized = ops.map((op) => ({ ...op, liveUntil: Math.max(op.liveFrom, lastUse.get(op.id) ?? op.liveFrom) }));
  const byId = Object.fromEntries(materialized.map((op) => [op.id, op]));
  const assumptions = [
    ...model.assumptions,
    "Operator FLOPs and tensor traffic are analytical estimates from config dimensions",
    "Backward operators use explicit analytical multipliers; kernel fusion is not modeled",
  ];
  return { name: model.name, source: options.source ?? String(config.source ?? "huggingface-config"), batchSize, sequenceLength, hiddenSize: model.hidden, precision, operators: materialized, assumptions, byId };
}

export interface EfficiencyPoint {
  operation: string;
  precision: string;
  minM: number;
  minN: number;
  minK: number;
  efficiency: number;
  provenance: string;
}

export interface RooflineHardware {
  peakFlops: number;
  memoryBandwidth: number;
  computeEfficiency?: number;
  memoryEfficiency?: number;
  curves?: readonly EfficiencyPoint[];
  name?: string;
}

export interface KernelEstimate {
  seconds: number;
  computeSeconds: number;
  memorySeconds: number;
  arithmeticIntensity: number;
  computeEfficiency: number;
  memoryEfficiency: number;
  provenance: readonly string[];
}

export function rooflineTime(op: Operator, hardware: RooflineHardware, options: { precision?: CorePrecision } = {}): KernelEstimate {
  if (hardware.peakFlops <= 0 || hardware.memoryBandwidth <= 0) throw new Error("hardware peakFlops and memoryBandwidth must be positive");
  const precision = corePrecision(options.precision ?? "bf16");
  const computeEfficiency = hardware.computeEfficiency ?? 0.55;
  const memoryEfficiency = hardware.memoryEfficiency ?? 0.70;
  if (!(computeEfficiency > 0 && computeEfficiency <= 1) || !(memoryEfficiency > 0 && memoryEfficiency <= 1)) throw new Error("efficiencies must be in (0, 1]");
  const dims = Array.isArray(op.metadata.weight_shape) ? op.metadata.weight_shape.map(Number) : [];
  const shape = op.shape;
  const m = shape.length >= 2 ? shape[shape.length - 2] : 0;
  const n = shape.length >= 1 ? shape[shape.length - 1] : 0;
  const k = dims.length === 2 ? dims[0] : 0;
  let ce = computeEfficiency;
  let me = memoryEfficiency;
  let provenance: string[] = ["modeled default efficiency; not hardware validation data"];
  for (const point of hardware.curves ?? []) {
    if ([op.kind, "*"].includes(point.operation) && [precision, "*"].includes(point.precision) && m >= point.minM && n >= point.minN && k >= point.minK) {
      if (!(point.efficiency > 0 && point.efficiency <= 1)) throw new Error("efficiency curve values must be in (0, 1]");
      if (["gemm", "mlp", "gated_mlp", "moe_experts"].includes(op.kind)) ce = point.efficiency;
      else me = point.efficiency;
      provenance = [point.provenance];
    }
  }
  const computeSeconds = op.flops / (hardware.peakFlops * ce);
  const memorySeconds = op.bytesMoved / (hardware.memoryBandwidth * me);
  return { seconds: Math.max(computeSeconds, memorySeconds), computeSeconds, memorySeconds, arithmeticIntensity: op.flops / Math.max(op.bytesMoved, 1), computeEfficiency: ce, memoryEfficiency: me, provenance: [...provenance, `hardware=${hardware.name ?? "user-specified hardware"}`] };
}

export class Link {
  public constructor(public readonly bandwidthBytesS: number, public readonly latencyS: number, public readonly kind = "fabric") {
    if (bandwidthBytesS <= 0 || latencyS < 0) throw new Error("link bandwidth must be positive and latency non-negative");
  }

  public get bandwidth_bytes_s(): number { return this.bandwidthBytesS; }
  public get latency_s(): number { return this.latencyS; }
}

export class Topology {
  public constructor(
    public readonly devices: number,
    public readonly gpusPerNode = 8,
    public readonly intraNode: Link = new Link(600e9, 2e-6, "NVLink/NVSwitch modeled"),
    public readonly interNode: Link = new Link(50e9, 8e-6, "NIC/fabric modeled"),
    public readonly crossNodeBandwidth: number | null = null,
  ) {
    if (devices <= 0 || gpusPerNode <= 0) throw new Error("device counts must be positive");
  }

  public get nodes(): number {
    return Math.ceil(this.devices / this.gpusPerNode);
  }

  public get gpus_per_node(): number { return this.gpusPerNode; }
}

export interface CollectiveEstimate {
  algorithm: string;
  seconds: number;
  bytesTransferred: number;
  participants: number;
  provenance: readonly string[];
}

export function collectiveTime(topology: Topology, participants: number, messageBytes: number, collective = "all_reduce", algorithm = "auto"): CollectiveEstimate {
  if (participants <= 0 || participants > topology.devices || messageBytes < 0) throw new Error("collective participants/message size are invalid");
  if (!["all_reduce", "all_gather", "reduce_scatter", "all_to_all"].includes(collective)) throw new Error("unsupported collective");
  let chosen = algorithm;
  if (chosen === "auto") chosen = participants > topology.gpusPerNode ? "hierarchical" : participants <= 8 ? "ring" : "tree";
  if (!["ring", "tree", "hierarchical"].includes(chosen)) throw new Error("algorithm must be auto, ring, tree, or hierarchical");
  const levels = Math.max(0, Math.ceil(Math.log2(participants)));
  const link = participants > topology.gpusPerNode ? topology.interNode : topology.intraNode;
  let amount: number;
  let seconds: number;
  if (chosen === "ring") {
    const rounds = Math.max(0, participants - 1);
    const factor = collective === "all_reduce" ? 2 : 1;
    amount = factor * rounds / Math.max(1, participants) * messageBytes;
    seconds = rounds * link.latencyS * factor + amount / link.bandwidthBytesS;
  } else if (chosen === "tree") {
    const amountFactor = collective === "all_reduce" ? 2 : 1;
    amount = amountFactor * messageBytes;
    seconds = levels * link.latencyS * (collective === "all_reduce" ? 2 : 1) + amount / link.bandwidthBytesS;
  } else {
    const localN = Math.min(participants, topology.gpusPerNode);
    const groups = Math.ceil(participants / topology.gpusPerNode);
    const intraAmount = 2 * (localN - 1) / Math.max(1, localN) * messageBytes;
    const interAmount = (collective === "all_reduce" ? 2 : 1) * messageBytes;
    seconds = 2 * Math.max(0, localN - 1) * topology.intraNode.latencyS + 2 * Math.max(0, groups - 1) * topology.interNode.latencyS + intraAmount / topology.intraNode.bandwidthBytesS + interAmount / topology.interNode.bandwidthBytesS;
    amount = intraAmount + interAmount;
  }
  return { algorithm: chosen, seconds, bytesTransferred: amount, participants, provenance: ["alpha-beta analytical model", "link rates are supplied or modeled defaults; not measured"] };
}

export class Parallelism {
  public constructor(public readonly data = 1, public readonly tensor = 1, public readonly pipeline = 1) {
    if (data <= 0 || tensor <= 0 || pipeline <= 0) throw new Error("parallelism dimensions must be positive");
  }

  public get worldSize(): number {
    return this.data * this.tensor * this.pipeline;
  }

  public get world_size(): number { return this.worldSize; }
}

export type CoreSchedule = "gpipe" | "1f1b" | "interleaved" | "zero_bubble";

export interface TrainingOptionsInit {
  microBatches?: number;
  schedule?: CoreSchedule | "zero-bubble";
  precision?: CorePrecision;
  recompute?: boolean;
  offload?: boolean;
  parallelism?: Parallelism | { data?: number; tensor?: number; pipeline?: number };
  dataCollective?: string;
  tensorCollective?: string;
  sequenceParallel?: number;
  contextParallel?: number;
  expertParallel?: number;
  zeroStage?: number;
  fsdp?: boolean;
}

export class TrainingOptions {
  public readonly microBatches: number;
  public readonly schedule: CoreSchedule;
  public readonly precision: CorePrecision;
  public readonly recompute: boolean;
  public readonly offload: boolean;
  public readonly parallelism: Parallelism;
  public readonly dataCollective: string;
  public readonly tensorCollective: string;
  public readonly sequenceParallel: number;
  public readonly contextParallel: number;
  public readonly expertParallel: number;
  public readonly zeroStage: number;
  public readonly fsdp: boolean;

  public constructor(options: TrainingOptionsInit = {}) {
    this.microBatches = positiveInt(options.microBatches ?? 4, "micro batches");
    const schedule = String(options.schedule ?? "1f1b").toLowerCase().replace(/-/g, "_");
    const supportedSchedules: readonly string[] = ["gpipe", "1f1b", "interleaved", "zero_bubble"];
    if (!supportedSchedules.includes(schedule)) {
      throw new Error("schedule must be gpipe, 1f1b, interleaved, or zero-bubble");
    }
    this.schedule = schedule as CoreSchedule;
    this.precision = corePrecision(options.precision ?? "bf16");
    this.recompute = options.recompute ?? false;
    this.offload = options.offload ?? false;
    const p = options.parallelism;
    this.parallelism = p instanceof Parallelism ? p : new Parallelism(p?.data ?? 1, p?.tensor ?? 1, p?.pipeline ?? 1);
    this.dataCollective = options.dataCollective ?? "all_reduce";
    this.tensorCollective = options.tensorCollective ?? "all_reduce";
    this.sequenceParallel = positiveInt(options.sequenceParallel ?? 1, "sequence parallel");
    this.contextParallel = positiveInt(options.contextParallel ?? 1, "context parallel");
    this.expertParallel = positiveInt(options.expertParallel ?? 1, "expert parallel");
    this.zeroStage = Math.floor(Number(options.zeroStage ?? 0));
    if (![0, 1, 2, 3].includes(this.zeroStage)) throw new Error("zero stage must be 0, 1, 2, or 3");
    this.fsdp = options.fsdp ?? false;
  }
}

export interface GanttEvent {
  id: string;
  gpu: number;
  stream: string;
  startS: number;
  endS: number;
  kind: string;
  layer: number | null;
  microBatch: number;
  dependencies: readonly string[];
  bytesMoved: number;
}

export interface MemoryPoint {
  timeS: number;
  gpu: number;
  allocatedBytes: number;
  event: string;
}

export interface TrainingSimulation {
  stepTimeS: number;
  events: readonly GanttEvent[];
  memoryTimeline: readonly MemoryPoint[];
  peakMemoryByGpu: readonly number[];
  assumptions: readonly string[];
  provenance: readonly string[];
}

export function simulateTrainingStep(graph: OperatorGraph, hardware: RooflineHardware, topology: Topology, options: TrainingOptions = new TrainingOptions()): TrainingSimulation {
  const parallelism = options.parallelism;
  if (options.microBatches <= 0 || parallelism.worldSize > topology.devices) throw new Error("micro_batches must be positive and parallelism fit the topology");
  const layerCount = Math.max(1, Math.max(-1, ...graph.operators.filter((op) => op.layer !== null).map((op) => op.layer as number)) + 1);
  const stageOf = new Map(Array.from({ length: layerCount }, (_, layer) => [layer, Math.min(parallelism.pipeline - 1, Math.floor(layer * parallelism.pipeline / layerCount))]));
  const gpuCount = parallelism.worldSize;
  const computeReady = Array.from({ length: gpuCount }, () => 0);
  const commReady = Array.from({ length: gpuCount }, () => 0);
  const events: GanttEvent[] = [];
  const finish = new Map<string, number>();
  const liveMemory = Array.from({ length: gpuCount }, () => 0);
  const peakMemory = Array.from({ length: gpuCount }, () => 0);
  const memoryTimeline: MemoryPoint[] = [];
  const forward = graph.operators.filter((op) => op.phase === "forward");
  const backward = graph.operators.filter((op) => op.phase === "backward");
  const owner = (layer: number | null): number => {
    const stage = layer === null ? 0 : (stageOf.get(layer) ?? 0);
    return stage * parallelism.tensor;
  };
  const launch = (key: string, op: Operator | null, gpu: number, stream: string, dependencies: readonly string[], duration: number, kind: string, layer: number | null, microBatch: number, moved = 0): string => {
    const start = Math.max(stream === "compute" ? computeReady[gpu] : commReady[gpu], ...dependencies.map((dependency) => finish.get(dependency) ?? 0));
    const end = start + duration;
    events.push({ id: key, gpu, stream, startS: start, endS: end, kind, layer, microBatch, dependencies: [...dependencies], bytesMoved: moved });
    if (stream === "compute") computeReady[gpu] = end; else commReady[gpu] = end;
    finish.set(key, end);
    if (op) {
      liveMemory[gpu] += op.writeBytes;
      peakMemory[gpu] = Math.max(peakMemory[gpu], liveMemory[gpu]);
      memoryTimeline.push({ timeS: end, gpu, allocatedBytes: liveMemory[gpu], event: key });
      for (const source of graph.operators) if (source.id !== op.id && source.phase === "backward" && source.inputs.includes(op.id)) liveMemory[gpu] = Math.max(0, liveMemory[gpu] - source.writeBytes);
    }
    return key;
  };
  const generatedForward = new Map<string, string>();
  const generatedBackward = new Map<string, string>();
  let forwardOrder: Array<{ microBatch: number; operator: Operator }> = Array.from({ length: options.microBatches }, (_, microBatch) => forward.map((operator) => ({ microBatch, operator }))).flat();
  if (["1f1b", "interleaved", "zero_bubble"].includes(options.schedule)) {
    // Keep input/embedding work ahead of the first layer.  A layer-first
    // order makes a plausible-looking timeline while violating the graph's
    // actual dependency on the embedding output.
    const prelude = forward.filter((operator) => operator.id === "embedding" || (operator.layer === null && operator.inputs.length === 0));
    const tail = forward.filter((operator) => operator.layer === null && !prelude.includes(operator));
    forwardOrder = [];
    for (let microBatch = 0; microBatch < options.microBatches; microBatch += 1) for (const operator of prelude) forwardOrder.push({ microBatch, operator });
    for (let layer = 0; layer < layerCount; layer += 1) for (let microBatch = 0; microBatch < options.microBatches; microBatch += 1) for (const operator of forward) if (operator.layer === layer) forwardOrder.push({ microBatch, operator });
    for (let microBatch = 0; microBatch < options.microBatches; microBatch += 1) for (const operator of tail) forwardOrder.push({ microBatch, operator });
  }
  for (const item of forwardOrder) {
    const { microBatch, operator } = item;
    const gpu = owner(operator.layer);
    const dependencies = operator.inputs.map((source) => generatedForward.get(`${microBatch}:${source}`)).filter((value): value is string => value !== undefined);
    if (operator.layer !== null && operator.layer > 0) {
      const boundary = `layer.${operator.layer - 1}.residual`;
      const boundaryKey = generatedForward.get(`${microBatch}:${boundary}`);
      if (boundaryKey && (stageOf.get(operator.layer - 1) ?? 0) !== (stageOf.get(operator.layer) ?? 0)) {
        const transferKey = `pipeline.mb${microBatch}.layer${operator.layer - 1}.to${operator.layer}`;
        const payload = graph.batchSize * graph.sequenceLength * graph.hiddenSize * PRECISION_BYTES[options.precision];
        const link = Math.floor(owner(operator.layer - 1) / topology.gpusPerNode) === Math.floor(gpu / topology.gpusPerNode) ? topology.intraNode : topology.interNode;
        if (!finish.has(transferKey)) launch(transferKey, null, owner(operator.layer - 1), "communication", [boundaryKey], link.latencyS + payload / link.bandwidthBytesS, "pipeline_transfer", operator.layer - 1, microBatch, payload);
        dependencies.splice(0, dependencies.length, ...dependencies.filter((dependency) => dependency !== boundaryKey), transferKey);
      }
    }
    const estimate = rooflineTime(operator, hardware, { precision: options.precision });
    const duration = estimate.seconds / parallelism.tensor * (options.recompute && !["embedding", "loss", "optimizer"].includes(operator.kind) ? 1.12 : 1);
    const key = `fwd.mb${microBatch}.${operator.id}`;
    launch(key, operator, gpu, "compute", dependencies, duration, operator.kind, operator.layer, microBatch, operator.bytesMoved);
    generatedForward.set(`${microBatch}:${operator.id}`, key);
  }
  const backwardOrder = options.schedule === "gpipe" ? Array.from({ length: options.microBatches }, (_, microBatch) => backward.map((operator) => ({ microBatch, operator }))).flat() : Array.from({ length: options.microBatches }, (_, microBatch) => backward.map((operator) => ({ microBatch, operator }))).flat().sort((left, right) => -((stageOf.get(right.operator.layer ?? 0) ?? 0) - (stageOf.get(left.operator.layer ?? 0) ?? 0)) || left.microBatch - right.microBatch || left.operator.id.localeCompare(right.operator.id));
  for (const item of backwardOrder) {
    const forwardOperator = String(item.operator.metadata.forward_operator ?? "");
    const dependencies: string[] = [];
    const matchingForward = generatedForward.get(`${item.microBatch}:${forwardOperator}`);
    if (matchingForward) dependencies.push(matchingForward);
    const previous = [...generatedBackward.entries()].reverse().find(([key]) => key.startsWith(`${item.microBatch}:`))?.[1];
    if (previous) dependencies.push(previous);
    const gpu = owner(item.operator.layer);
    const estimate = rooflineTime(item.operator, hardware, { precision: options.precision });
    const duration = estimate.seconds / parallelism.tensor * (options.recompute ? 1.12 : 1);
    const key = `bwd.mb${item.microBatch}.${item.operator.id}`;
    launch(key, item.operator, gpu, "compute", dependencies, duration, item.operator.kind, item.operator.layer, item.microBatch, item.operator.bytesMoved);
    generatedBackward.set(`${item.microBatch}:${item.operator.id}`, key);
  }
  const parameterBytes = graph.operators.reduce((sum, operator) => sum + Number(operator.metadata.parameter_count ?? 0), 0) * PRECISION_BYTES[options.precision];
  if (parallelism.data > 1) {
    const estimate = collectiveTime(topology, parallelism.data, parameterBytes, options.dataCollective);
    for (let rank = 0; rank < parallelism.data; rank += 1) {
      const dependency = backward.length ? generatedBackward.get(`${options.microBatches - 1}:${backward[backward.length - 1].id}`) : undefined;
      launch(`data_collective.rank${rank}`, null, rank * parallelism.tensor, "communication", dependency ? [dependency] : [], estimate.seconds, "collective", null, options.microBatches - 1, estimate.bytesTransferred);
    }
  }
  if (options.sequenceParallel > 1) {
    const estimate = collectiveTime(topology, options.sequenceParallel, graph.sequenceLength * graph.hiddenSize * PRECISION_BYTES[options.precision], "all_gather", "ring");
    const dependency = forward.length ? generatedForward.get(`${options.microBatches - 1}:${forward[forward.length - 1].id}`) : undefined;
    launch("sequence_parallel.collective", null, owner(null), "communication", dependency ? [dependency] : [], estimate.seconds, "sequence_parallel", null, options.microBatches - 1, estimate.bytesTransferred);
  }
  if (options.contextParallel > 1) {
    const estimate = collectiveTime(topology, options.contextParallel, graph.sequenceLength * graph.hiddenSize * PRECISION_BYTES[options.precision], "all_gather", "ring");
    const dependency = forward.length ? generatedForward.get(`${options.microBatches - 1}:${forward[forward.length - 1].id}`) : undefined;
    launch("context_parallel.collective", null, owner(null), "communication", dependency ? [dependency] : [], estimate.seconds, "context_parallel", null, options.microBatches - 1, estimate.bytesTransferred);
  }
  if (options.expertParallel > 1) {
    const estimate = collectiveTime(topology, options.expertParallel, graph.sequenceLength * graph.hiddenSize * PRECISION_BYTES[options.precision], "all_to_all", "ring");
    const dependency = forward.length ? generatedForward.get(`${options.microBatches - 1}:${forward[forward.length - 1].id}`) : undefined;
    launch("expert_parallel.collective", null, owner(null), "communication", dependency ? [dependency] : [], estimate.seconds, "expert_parallel", null, options.microBatches - 1, estimate.bytesTransferred);
  }
  if (parallelism.tensor > 1) for (let layer = 0; layer < layerCount; layer += 1) {
    const estimate = collectiveTime(topology, parallelism.tensor, graph.sequenceLength * graph.hiddenSize * PRECISION_BYTES[options.precision], options.tensorCollective, "ring");
    for (let microBatch = 0; microBatch < options.microBatches; microBatch += 1) {
      const firstOperator = forward.find((operator) => operator.layer === layer);
      const dependency = firstOperator ? generatedForward.get(`${microBatch}:${firstOperator.id}`) : undefined;
      launch(`tensor_collective.layer${layer}.mb${microBatch}`, null, owner(layer), "communication", dependency ? [dependency] : [], estimate.seconds, "collective", layer, microBatch, estimate.bytesTransferred);
    }
  }
  if (options.offload) for (let microBatch = 0; microBatch < options.microBatches; microBatch += 1) for (let layer = 0; layer < layerCount; layer += 1) {
    const gpu = owner(layer);
    const payload = graph.batchSize * graph.sequenceLength * graph.hiddenSize * PRECISION_BYTES[options.precision];
    const dependency = generatedForward.get(`${microBatch}:layer.${layer}.residual`);
    launch(`offload.layer${layer}.mb${microBatch}`, null, gpu, "communication", dependency ? [dependency] : [], payload / Math.max(hardware.memoryBandwidth, 1), "offload", layer, microBatch, payload);
  }
  events.sort((left, right) => left.startS - right.startS || left.gpu - right.gpu || left.stream.localeCompare(right.stream) || left.id.localeCompare(right.id));
  if (options.zeroStage || options.fsdp) {
    // This affects the modeled parameter residency boundary.  The current
    // operator-level memory timeline remains deliberately conservative.
    const shardFactor = Math.max(1, parallelism.data);
    void (parameterBytes / (options.zeroStage >= 1 || options.fsdp ? shardFactor : 1));
  }
  const assumptions = [
    ...graph.assumptions,
    "Compute kernels use max(compute roofline, memory roofline) with supplied peak rates",
    "Tensor/data/pipeline partitioning and communication are analytical approximations",
    "Recompute applies a 12% modeled compute multiplier; offload uses memory-bandwidth transfer rate",
    "Per-GPU memory timeline tracks operator output allocations and is a lower-level estimate, not allocator tracing",
    `Schedule policy: ${options.schedule}; interleaved and zero-bubble use deterministic stage-aware event ordering, not a runtime scheduler`,
    `Memory sharding: ZeRO stage ${options.zeroStage}${options.fsdp ? " / FSDP" : ""}; shard factors are analytical`,
    `Sequence/context/expert parallel factors: ${options.sequenceParallel}/${options.contextParallel}/${options.expertParallel}`,
  ];
  const provenance = [...new Set(graph.operators.flatMap((operator) => rooflineTime(operator, hardware, { precision: options.precision }).provenance))].sort();
  return { stepTimeS: Math.max(...events.map((event) => event.endS), 0), events, memoryTimeline: memoryTimeline.sort((left, right) => left.timeS - right.timeS || left.gpu - right.gpu || left.event.localeCompare(right.event)), peakMemoryByGpu: peakMemory, assumptions: [...new Set(assumptions)].sort(), provenance };
}

export function operatorGraphToJSON(graph: OperatorGraph): Record<string, unknown> {
  return {
    name: graph.name,
    source: graph.source,
    batch_size: graph.batchSize,
    sequence_length: graph.sequenceLength,
    hidden_size: graph.hiddenSize,
    precision: graph.precision,
    operators: graph.operators.map((operator) => ({ id: operator.id, kind: operator.kind, layer: operator.layer, phase: operator.phase, inputs: [...operator.inputs], shape: [...operator.shape], flops: operator.flops, read_bytes: operator.readBytes, write_bytes: operator.writeBytes, live_from: operator.liveFrom, live_until: operator.liveUntil, metadata: operator.metadata })),
    assumptions: [...graph.assumptions],
  };
}

export function trainingSimulationToJSON(simulation: TrainingSimulation): Record<string, unknown> {
  return {
    step_time_s: simulation.stepTimeS,
    events: simulation.events.map((event) => ({ id: event.id, gpu: event.gpu, stream: event.stream, start_s: event.startS, end_s: event.endS, kind: event.kind, layer: event.layer, micro_batch: event.microBatch, dependencies: [...event.dependencies], bytes_moved: event.bytesMoved })),
    memory_timeline: simulation.memoryTimeline.map((point) => ({ time_s: point.timeS, gpu: point.gpu, allocated_bytes: point.allocatedBytes, event: point.event })),
    peak_memory_by_gpu: [...simulation.peakMemoryByGpu],
    assumptions: [...simulation.assumptions],
    provenance: [...simulation.provenance],
  };
}

// Direct-name aliases make fixture adapters mechanically comparable with the
// Python reference while browser callers can use the normal camelCase API.
export const build_operator_graph = buildOperatorGraph;
export const roofline_time = rooflineTime;
export const collective_time = collectiveTime;
export const simulate_training_step = simulateTrainingStep;

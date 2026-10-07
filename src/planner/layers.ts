/**
 * Layer-aware model graph and bounded training planner.
 *
 * This is the browser port of nomo_planner/layers.py. The accounting follows
 * PaLM (https://arxiv.org/abs/2203.15556), activation-memory terms follow
 * Korthikanti et al. (https://arxiv.org/abs/2205.05198), and recomputation is
 * treated as the trade-off described by Chen et al.
 * (https://arxiv.org/abs/1604.06174). Hardware transfer and FP8 quality values
 * are assumptions until customer measurements replace them.
 */

export type LayerKind = "embedding" | "attention" | "mlp" | "output";
export type LayerPrecision = "bf16" | "fp8";
export type RecomputeMode = "none" | "selective" | "full";

export interface LayerNode {
  id: string;
  kind: LayerKind;
  layerIndex: number | null;
  parameterCount: number;
  forwardFlops: number;
  activationBytes: number;
  kvCacheBytes: number;
  metadata: Record<string, unknown>;
}

export interface LayerGraph {
  name: string;
  source: string;
  seqLen: number;
  batchSize: number;
  hiddenSize: number;
  layers: number;
  attentionHeads: number;
  kvHeads: number;
  intermediateSize: number;
  vocabSize: number;
  tiedEmbeddings: boolean;
  gatedMlp: boolean;
  experts: number | null;
  expertsPerToken: number | null;
  nodes: LayerNode[];
  assumptions: string[];
  parameterCount: number;
}

export interface LayerTrainingPlan {
  stages: number[];
  precision: LayerPrecision[];
  recompute: RecomputeMode[];
  offload: boolean[];
}

export interface LayerHardware {
  devices: number;
  memoryBytes: number;
  peakFlops: number;
  costPerDeviceHour: number;
  interconnectBytesS: number;
  cpuOffloadBytesS: number;
  overheadBytes: number;
  usableMemory: number;
}

export interface LayerTrainingProblem {
  graph: LayerGraph;
  hardware?: Partial<LayerHardware>;
  microBatches?: number;
  pipelineStages?: number[];
  fp8Enabled?: boolean;
  maxCandidates?: number;
  totalSteps?: number;
  seed?: number;
}

export interface LayerMetrics {
  objectives: {
    stepTimeS: number;
    costUsdPerStep: number;
    memoryHeadroom: number;
    communicationS: number;
    pipelineBubbleS: number;
    offloadTransferS: number;
    wholeRunTimeS: number;
    wholeRunCostUsd: number;
  };
  constraints: Record<string, number>;
  memoryByStage: number[];
  stageTimes: number[];
  notes: string[];
}

export interface LayerSearchResult {
  graph: LayerGraph;
  best: LayerTrainingPlan | null;
  bestMetrics: LayerMetrics | null;
  front: Array<{ plan: LayerTrainingPlan; metrics: LayerMetrics }>;
  baseline: { plan: LayerTrainingPlan; metrics: LayerMetrics };
  globalBf16: { plan: LayerTrainingPlan; metrics: LayerMetrics };
  globalBest: { plan: LayerTrainingPlan; metrics: LayerMetrics };
  precisionGainPct: number;
  perLayerGainPct: number;
  seed: number;
  evaluated: number;
  exhaustive: boolean;
  assumptions: string[];
}

/** Planner-facing fields copied from the public Hugging Face model configs. */
export const BUILTIN_MODEL_CONFIGS = {
  llama3_8b: {
    _name_or_path: "meta-llama/Meta-Llama-3-8B", model_type: "llama", num_hidden_layers: 32,
    hidden_size: 4096, intermediate_size: 14336, num_attention_heads: 32, num_key_value_heads: 8,
    vocab_size: 128256, max_position_embeddings: 8192, rope_theta: 500000, tie_word_embeddings: false, hidden_act: "silu",
    source: "https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json",
  },
  llama3_70b: {
    _name_or_path: "meta-llama/Meta-Llama-3-70B", model_type: "llama", num_hidden_layers: 80,
    hidden_size: 8192, intermediate_size: 28672, num_attention_heads: 64, num_key_value_heads: 8,
    vocab_size: 128256, max_position_embeddings: 8192, rope_theta: 500000, tie_word_embeddings: false, hidden_act: "silu",
    source: "https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json",
  },
  mixtral_8x7b: {
    _name_or_path: "mistralai/Mixtral-8x7B-v0.1", model_type: "mixtral", num_hidden_layers: 32,
    hidden_size: 4096, intermediate_size: 14336, num_attention_heads: 32, num_key_value_heads: 8,
    num_local_experts: 8, num_experts_per_tok: 2, vocab_size: 32000, max_position_embeddings: 32768,
    rope_theta: 1000000, tie_word_embeddings: false, hidden_act: "silu",
    source: "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json",
  },
} as const;

const PRECISION_SPEEDUP: Record<LayerPrecision, number> = { bf16: 1, fp8: 1.35 };
const RECOMPUTE_MULTIPLIER: Record<RecomputeMode, number> = { none: 1, selective: 1.12, full: 1.28 };
const EXHAUSTIVE_NODE_LIMIT = 4;
const NSGA_MAX_POPULATION = 64;
const NSGA_CROSSOVER_RATE = 0.90;
const DEFAULT_PIPELINE_STAGES = [1, 2, 4, 8, 16, 32];

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
}

function first(config: Record<string, unknown>, ...keys: string[]): unknown {
  return keys.map((key) => config[key]).find((value) => value !== undefined && value !== null);
}

function requiredInt(config: Record<string, unknown>, ...keys: string[]): number {
  const value = Number(first(config, ...keys));
  if (!Number.isFinite(value) || value <= 0) throw new Error(`config.json is missing one of: ${keys.join(", ")}`);
  return Math.floor(value);
}

function finitePositive(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

/** Re-import the graph contract emitted by the shared graph and neuromorphic tab. */
function buildGraphFromContract(raw: unknown, options: { seqLen?: number; batchSize?: number; source?: string }): LayerGraph | null {
  const outer = asRecord(raw);
  const candidate = asRecord(outer.graph ?? raw);
  if (!Array.isArray(candidate.nodes)) return null;
  if (!candidate.nodes.length) throw new Error("graph JSON must contain at least one node");
  const nodes = candidate.nodes.map((value, index) => {
    const node = asRecord(value);
    const kind = String(node.kind ?? "");
    if (!["embedding", "attention", "mlp", "output"].includes(kind)) throw new Error(`graph JSON node ${index + 1} has an unsupported kind`);
    const parameterCount = Number(node.parameterCount ?? node.parameter_count);
    const forwardFlops = Number(node.forwardFlops ?? node.forward_flops ?? 0);
    const activationBytes = Number(node.activationBytes ?? node.activation_bytes ?? 0);
    const kvCacheBytes = Number(node.kvCacheBytes ?? node.kv_cache_bytes ?? 0);
    if (!String(node.id ?? "") || ![parameterCount, forwardFlops, activationBytes, kvCacheBytes].every((v) => Number.isFinite(v) && v >= 0)) {
      throw new Error(`graph JSON node ${index + 1} is missing numeric accounting fields`);
    }
    const layerIndexRaw = node.layerIndex ?? node.layer_index;
    const layerIndex = layerIndexRaw === null || layerIndexRaw === undefined ? null : Number(layerIndexRaw);
    return {
      id: String(node.id), kind: kind as LayerKind, layerIndex: layerIndex === null || !Number.isFinite(layerIndex) ? null : Math.floor(layerIndex),
      parameterCount, forwardFlops, activationBytes, kvCacheBytes, metadata: asRecord(node.metadata),
    };
  });
  const seqLen = finitePositive(candidate.seqLen ?? candidate.seq_len ?? options.seqLen, 2048);
  const batchSize = finitePositive(candidate.batchSize ?? candidate.batch_size ?? options.batchSize, 1);
  const layers = finitePositive(candidate.layers, Math.max(1, ...nodes.map((node) => (node.layerIndex ?? -1) + 1)));
  const attentionHeads = finitePositive(candidate.attentionHeads ?? candidate.attention_heads, 1);
  const kvHeads = finitePositive(candidate.kvHeads ?? candidate.kv_heads, attentionHeads);
  const assumptions = Array.isArray(outer.assumptions) ? outer.assumptions.map(String) : [];
  assumptions.push("Imported from a Nomo graph contract; missing architecture metadata is not inferred");
  return {
    name: String(candidate.name ?? "uploaded-graph"), source: String(outer.source ?? options.source ?? "graph-json"),
    seqLen, batchSize, hiddenSize: finitePositive(candidate.hiddenSize ?? candidate.hidden_size, 1), layers,
    attentionHeads, kvHeads, intermediateSize: finitePositive(candidate.intermediateSize ?? candidate.intermediate_size, 1),
    vocabSize: finitePositive(candidate.vocabSize ?? candidate.vocab_size, 1),
    tiedEmbeddings: Boolean(candidate.tiedEmbeddings ?? candidate.tied_embeddings), gatedMlp: Boolean(candidate.gatedMlp ?? candidate.gated_mlp),
    experts: candidate.experts === null || candidate.experts === undefined ? null : finitePositive(candidate.experts, 1),
    expertsPerToken: candidate.expertsPerToken === null || candidate.expertsPerToken === undefined ? null : finitePositive(candidate.expertsPerToken, 1),
    nodes, assumptions: [...new Set(assumptions)], parameterCount: nodes.reduce((total, node) => total + node.parameterCount, 0),
  };
}

/** Expand a Hugging Face config.json object into the shared ordered graph. */
export function buildLayerGraph(raw: unknown, options: { seqLen?: number; batchSize?: number; source?: string } = {}): LayerGraph {
  const config = asRecord(raw);
  const imported = buildGraphFromContract(raw, options);
  if (imported) return imported;
  const seqLen = options.seqLen ?? 2048;
  const batchSize = options.batchSize ?? 1;
  if (seqLen <= 0 || batchSize <= 0) throw new Error("sequence length and batch size must be positive");
  const layers = requiredInt(config, "num_hidden_layers", "n_layer", "num_layers");
  const hiddenSize = requiredInt(config, "hidden_size", "n_embd", "d_model");
  const attentionHeads = requiredInt(config, "num_attention_heads", "n_head", "num_heads");
  if (hiddenSize % attentionHeads !== 0) throw new Error("hidden_size must be divisible by num_attention_heads");
  const kvHeads = Math.floor(Number(first(config, "num_key_value_heads", "n_kv_heads") ?? attentionHeads));
  if (kvHeads <= 0 || attentionHeads % kvHeads !== 0) throw new Error("num_key_value_heads must divide num_attention_heads");
  const intermediateSize = Math.floor(Number(first(config, "intermediate_size", "n_inner") ?? hiddenSize * 4));
  const vocabSize = requiredInt(config, "vocab_size", "n_vocab");
  const tiedEmbeddings = Boolean(first(config, "tie_word_embeddings", "tie_embeddings") ?? false);
  const hiddenAct = String(first(config, "hidden_act") ?? "").toLowerCase();
  const gatedMlp = Boolean(first(config, "gated_mlp") ?? false) || ["silu", "swish", "geglu"].includes(hiddenAct);
  const expertsRaw = first(config, "num_local_experts", "num_experts");
  const experts = expertsRaw === undefined ? null : Math.floor(Number(expertsRaw));
  const topKRaw = first(config, "num_experts_per_tok", "num_experts_per_token", "moe_top_k");
  const expertsPerToken = topKRaw === undefined ? null : Math.floor(Number(topKRaw));
  if (experts !== null && (!Number.isFinite(experts) || experts <= 0 || expertsPerToken === null || expertsPerToken <= 0 || expertsPerToken > experts)) {
    throw new Error("MoE config must provide valid num_experts and num_experts_per_tok");
  }
  const headDim = hiddenSize / attentionHeads;
  const kvDim = kvHeads * headDim;
  const tokenCount = seqLen * batchSize;
  const assumptions: string[] = [];
  if (first(config, "intermediate_size", "n_inner") === undefined) assumptions.push("intermediate_size inferred as 4 × hidden_size");
  if (first(config, "num_key_value_heads", "n_kv_heads") === undefined) assumptions.push("num_key_value_heads inferred as num_attention_heads");
  if (experts !== null) assumptions.push("MoE memory includes resident experts; FLOPs use top-k active experts");
  const nodes: LayerNode[] = [{
    id: "embedding", kind: "embedding", layerIndex: null, parameterCount: vocabSize * hiddenSize,
    forwardFlops: 0, activationBytes: tokenCount * hiddenSize * 2, kvCacheBytes: 0,
    metadata: { sharedWith: tiedEmbeddings ? "output" : null },
  }];
  for (let index = 0; index < layers; index += 1) {
    const attentionParams = hiddenSize * hiddenSize + 2 * hiddenSize * kvDim + hiddenSize * hiddenSize;
    const attentionFlops = 2 * tokenCount * (hiddenSize * hiddenSize + 2 * hiddenSize * kvDim + hiddenSize * hiddenSize) + 4 * batchSize * seqLen * seqLen * hiddenSize;
    nodes.push({
      id: `block.${index}.attention`, kind: "attention", layerIndex: index, parameterCount: attentionParams,
      forwardFlops: attentionFlops, activationBytes: tokenCount * hiddenSize * 2,
      kvCacheBytes: batchSize * seqLen * 2 * kvHeads * headDim * 2,
      metadata: { heads: attentionHeads, kvHeads, headDim },
    });
    const expertMultiplier = experts ?? 1;
    const activeMultiplier = expertsPerToken ?? 1;
    const matrices = gatedMlp ? 3 : 2;
    nodes.push({
      id: `block.${index}.mlp`, kind: "mlp", layerIndex: index, parameterCount: matrices * hiddenSize * intermediateSize * expertMultiplier,
      forwardFlops: 2 * tokenCount * matrices * hiddenSize * intermediateSize * activeMultiplier,
      activationBytes: tokenCount * intermediateSize * 2, kvCacheBytes: 0,
      metadata: { gated: gatedMlp, experts, activeExperts: expertsPerToken },
    });
  }
  nodes.push({
    id: "output", kind: "output", layerIndex: null, parameterCount: tiedEmbeddings ? 0 : vocabSize * hiddenSize,
    forwardFlops: 2 * tokenCount * hiddenSize * vocabSize, activationBytes: tokenCount * vocabSize * 2, kvCacheBytes: 0,
    metadata: { sharedWith: tiedEmbeddings ? "embedding" : null },
  });
  return {
    name: String(first(config, "_name_or_path", "model_type") ?? "uploaded-transformer"), source: options.source ?? "huggingface-config",
    seqLen, batchSize, hiddenSize, layers, attentionHeads, kvHeads, intermediateSize, vocabSize,
    tiedEmbeddings, gatedMlp, experts, expertsPerToken, nodes, assumptions,
    parameterCount: nodes.reduce((total, node) => total + node.parameterCount, 0),
  };
}

/** Contract/export artifact for the established neuromorphic compiler boundary. */
export function buildNeuromorphicManifest(graph: LayerGraph) {
  return {
    format: "nomo.graph/neuromorphic-preview/1",
    status: "contract-only",
    source: graph.source,
    graph: {
      name: graph.name,
      seqLen: graph.seqLen,
      batchSize: graph.batchSize,
      hiddenSize: graph.hiddenSize,
      layers: graph.layers,
      attentionHeads: graph.attentionHeads,
      kvHeads: graph.kvHeads,
      intermediateSize: graph.intermediateSize,
      vocabSize: graph.vocabSize,
      tiedEmbeddings: graph.tiedEmbeddings,
      gatedMlp: graph.gatedMlp,
      experts: graph.experts,
      expertsPerToken: graph.expertsPerToken,
      parameterCount: graph.parameterCount,
      nodes: graph.nodes,
    },
    exportBoundary: "NIR/C11/RTL exports remain owned by the validated compiler dashboard",
    assumptions: graph.assumptions,
  };
}

export function balancedLayerStages(nodeCount: number, stages: number): number[] {
  const count = Math.max(1, Math.min(stages, nodeCount));
  return Array.from({ length: nodeCount }, (_, index) => Math.min(count - 1, Math.floor((index * count) / nodeCount)));
}

function planKey(plan: LayerTrainingPlan): string {
  return JSON.stringify([plan.stages, plan.precision, plan.recompute, plan.offload]);
}

/** Repair a plan while preserving absolute per-node locks. */
export function repairLayerPlan(graph: LayerGraph, input: LayerTrainingPlan, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>> = {}): LayerTrainingPlan {
  const n = graph.nodes.length;
  if ([input.stages.length, input.precision.length, input.recompute.length, input.offload.length].some((length) => length !== n)) throw new Error("layer plan arrays must match graph node count");
  const transitions = input.stages.slice(1).filter((value, index) => value !== input.stages[index]).length;
  const stages = balancedLayerStages(n, transitions + 1);
  const precision: LayerPrecision[] = input.precision.map((value) => value === "fp8" ? "fp8" : "bf16");
  const recompute: RecomputeMode[] = input.recompute.map((value) => value in RECOMPUTE_MULTIPLIER ? value : "none");
  const offload = input.offload.map(Boolean);
  precision[0] = "bf16";
  precision[n - 1] = "bf16";
  graph.nodes.forEach((node, index) => {
    const lock = locks[node.id] ?? {};
    if (lock.stage !== undefined) stages[index] = Math.max(0, Math.floor(lock.stage));
    if (lock.precision !== undefined) precision[index] = lock.precision;
    if (lock.recompute !== undefined) recompute[index] = lock.recompute;
    if (lock.offload !== undefined) offload[index] = lock.offload;
  });
  const labels = new Map<number, number>();
  const normalized = stages.map((stage) => {
    if (!labels.has(stage)) labels.set(stage, labels.size);
    return labels.get(stage)!;
  });
  return { stages: normalized, precision, recompute, offload };
}

function hardware(problem: LayerTrainingProblem): LayerHardware {
  return {
    devices: 8, memoryBytes: 80e9, peakFlops: 989e12, costPerDeviceHour: 3.5,
    interconnectBytesS: 600e9, cpuOffloadBytesS: 32e9, overheadBytes: 3e9, usableMemory: 0.9,
    ...(problem.hardware ?? {}),
  };
}

export function evaluateLayerPlan(problem: LayerTrainingProblem, rawPlan: LayerTrainingPlan, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>> = {}): LayerMetrics {
  const graph = problem.graph;
  const plan = repairLayerPlan(graph, rawPlan, locks);
  const hw = hardware(problem);
  const stageCount = Math.max(...plan.stages) + 1;
  const stageTimes = Array.from({ length: stageCount }, () => 0);
  const memoryByStage = Array.from({ length: stageCount }, () => hw.overheadBytes);
  const communicationByStage = Array.from({ length: stageCount }, () => 0);
  const notes = new Set<string>();
  let offloadTransferS = 0;
  graph.nodes.forEach((node, index) => {
    const p = plan.precision[index]!;
    const r = plan.recompute[index]!;
    const s = plan.stages[index]!;
    stageTimes[s]! += node.forwardFlops * RECOMPUTE_MULTIPLIER[r]! / (hw.peakFlops * PRECISION_SPEEDUP[p]!);
    const stateBytes = node.parameterCount * (p === "fp8" ? 1 : 16);
    if (plan.offload[index]) {
      memoryByStage[s]! += stateBytes + node.activationBytes * 0.15;
      const transfer = node.activationBytes * 0.85 / hw.cpuOffloadBytesS;
      stageTimes[s]! += transfer;
      offloadTransferS += transfer;
      notes.add("CPU activation offload uses the supplied/default PCIe/host bandwidth");
    } else {
      memoryByStage[s]! += stateBytes + node.activationBytes;
    }
    if (p === "fp8") notes.add("FP8 speed and training quality are assumptions until customer evaluation");
  });
  let communicationS = 0;
  if (stageCount > 1) {
    for (let index = 1; index < graph.nodes.length; index += 1) {
      const left = plan.stages[index - 1]!;
      const right = plan.stages[index]!;
      if (left !== right) {
        const bytesAtBoundary = graph.nodes[index - 1]!.activationBytes;
        const transfer = bytesAtBoundary / hw.interconnectBytesS;
        communicationS += bytesAtBoundary * 2 / hw.interconnectBytesS;
        communicationByStage[left]! += transfer;
        communicationByStage[right]! += transfer;
      }
    }
    for (let index = 0; index < stageTimes.length; index += 1) stageTimes[index]! += communicationByStage[index]!;
    notes.add("Inter-stage communication charges both adjacent stages at the supplied/default interconnect bandwidth");
  }
  const maxStage = Math.max(...stageTimes, 0);
  const microBatches = Math.max(1, problem.microBatches ?? 8);
  const pipelineBubbleS = maxStage * Math.max(0, stageCount - 1) / microBatches;
  const stepTimeS = maxStage + pipelineBubbleS;
  const costUsdPerStep = stepTimeS / 3600 * hw.devices * hw.costPerDeviceHour;
  const capacity = hw.memoryBytes * hw.usableMemory;
  const constraints: Record<string, number> = {};
  memoryByStage.forEach((value, index) => { constraints[`memory_stage_${index}`] = value / capacity - 1; });
  const allowedStages = problem.pipelineStages ?? [1, 2, 4];
  constraints.stageCount = allowedStages.includes(stageCount) ? -1 : 1;
  const memoryHeadroom = Math.min(...memoryByStage.map((value) => (capacity - value) / capacity));
  const totalSteps = Math.max(1, problem.totalSteps ?? 1000);
  return {
    objectives: {
      stepTimeS, costUsdPerStep, memoryHeadroom, communicationS, pipelineBubbleS, offloadTransferS,
      wholeRunTimeS: stepTimeS * totalSteps, wholeRunCostUsd: costUsdPerStep * totalSteps,
    },
    constraints, memoryByStage, stageTimes, notes: [...notes].sort(),
  };
}

function feasible(metrics: LayerMetrics): boolean {
  return Object.values(metrics.constraints).every((value) => value <= 0);
}

function objectiveVector(metrics: LayerMetrics): [number, number, number] {
  return [metrics.objectives.stepTimeS, metrics.objectives.costUsdPerStep, -metrics.objectives.memoryHeadroom];
}

function dominates(left: LayerMetrics, right: LayerMetrics): boolean {
  const a = objectiveVector(left); const b = objectiveVector(right);
  return a.every((value, index) => value <= b[index]!) && a.some((value, index) => value < b[index]!);
}

function constraintViolation(metrics: LayerMetrics): number {
  return Object.values(metrics.constraints).reduce((sum, value) => sum + Math.max(0, value), 0);
}

function constrainedDominates(left: LayerMetrics, right: LayerMetrics): boolean {
  const leftViolation = constraintViolation(left); const rightViolation = constraintViolation(right);
  if (leftViolation <= 0 && rightViolation > 0) return true;
  if (leftViolation > 0 && rightViolation <= 0) return false;
  if (leftViolation > 0 || rightViolation > 0) return leftViolation < rightViolation;
  return dominates(left, right);
}

type EvaluatedLayerPlan = { plan: LayerTrainingPlan; metrics: LayerMetrics };

class DeterministicRng {
  private state: number;

  constructor(seed: number) { this.state = Math.trunc(seed) >>> 0; }

  random(): number {
    this.state = (Math.imul(1664525, this.state) + 1013904223) >>> 0;
    return this.state / 4294967296;
  }

  index(length: number): number { return length > 1 ? Math.min(length - 1, Math.floor(this.random() * length)) : 0; }
}

function nondominatedSort(evaluated: EvaluatedLayerPlan[]): { ranks: number[]; fronts: number[][] } {
  const dominatedBy = Array(evaluated.length).fill(0) as number[];
  const dominatesIndices = evaluated.map(() => [] as number[]);
  for (let left = 0; left < evaluated.length; left += 1) {
    for (let right = left + 1; right < evaluated.length; right += 1) {
      if (constrainedDominates(evaluated[left]!.metrics, evaluated[right]!.metrics)) {
        dominatesIndices[left]!.push(right); dominatedBy[right]! += 1;
      } else if (constrainedDominates(evaluated[right]!.metrics, evaluated[left]!.metrics)) {
        dominatesIndices[right]!.push(left); dominatedBy[left]! += 1;
      }
    }
  }
  const first = dominatedBy.flatMap((count, index) => count === 0 ? [index] : []);
  const fronts: number[][] = [first]; const ranks = Array(evaluated.length).fill(-1) as number[];
  first.forEach((index) => { ranks[index] = 0; });
  for (let frontIndex = 0; frontIndex < fronts.length && fronts[frontIndex]!.length; frontIndex += 1) {
    const next: number[] = [];
    fronts[frontIndex]!.forEach((left) => {
      dominatesIndices[left]!.forEach((right) => {
        dominatedBy[right]! -= 1;
        if (dominatedBy[right] === 0) { ranks[right] = frontIndex + 1; next.push(right); }
      });
    });
    if (next.length) fronts.push(next);
  }
  return { ranks, fronts };
}

function crowdingDistances(evaluated: EvaluatedLayerPlan[], fronts: number[][]): number[] {
  const distances = Array(evaluated.length).fill(0) as number[];
  fronts.forEach((front) => {
    if (front.length <= 2) { front.forEach((index) => { distances[index] = Number.POSITIVE_INFINITY; }); return; }
    for (let objective = 0; objective < 3; objective += 1) {
      const ordered = [...front].sort((left, right) => objectiveVector(evaluated[left]!.metrics)[objective]! - objectiveVector(evaluated[right]!.metrics)[objective]! || left - right);
      distances[ordered[0]!] = Number.POSITIVE_INFINITY; distances[ordered[ordered.length - 1]!] = Number.POSITIVE_INFINITY;
      const low = objectiveVector(evaluated[ordered[0]!]!.metrics)[objective]!;
      const high = objectiveVector(evaluated[ordered[ordered.length - 1]!]!.metrics)[objective]!;
      if (high <= low) continue;
      for (let position = 1; position < ordered.length - 1; position += 1) {
        const index = ordered[position]!;
        if (distances[index] === Number.POSITIVE_INFINITY) continue;
        const previous = objectiveVector(evaluated[ordered[position - 1]!]!.metrics)[objective]!;
        const next = objectiveVector(evaluated[ordered[position + 1]!]!.metrics)[objective]!;
        distances[index]! += (next - previous) / (high - low);
      }
    }
  });
  return distances;
}

function selectNsga(evaluated: EvaluatedLayerPlan[], limit: number): EvaluatedLayerPlan[] {
  const { fronts } = nondominatedSort(evaluated); const distances = crowdingDistances(evaluated, fronts);
  const selected: EvaluatedLayerPlan[] = [];
  for (const front of fronts) {
    if (selected.length + front.length <= limit) { selected.push(...front.map((index) => evaluated[index]!)); continue; }
    const ordered = [...front].sort((left, right) => distances[right]! - distances[left]!
      || comparePlanKeys(planKey(evaluated[left]!.plan), planKey(evaluated[right]!.plan)));
    selected.push(...ordered.slice(0, Math.max(0, limit - selected.length)).map((index) => evaluated[index]!));
    break;
  }
  return selected;
}

function tournament(population: EvaluatedLayerPlan[], rng: DeterministicRng, ranks: number[], distances: number[]): EvaluatedLayerPlan {
  const left = rng.index(population.length); const right = rng.index(population.length);
  let winner = left;
  if (ranks[right]! < ranks[left]!) winner = right;
  else if (ranks[right] === ranks[left] && distances[right]! > distances[left]!) winner = right;
  else if (ranks[right] === ranks[left] && distances[right] === distances[left]
    && comparePlanKeys(planKey(population[right]!.plan), planKey(population[left]!.plan)) < 0) winner = right;
  return population[winner]!;
}

function comparePlanKeys(left: string, right: string): number { return left < right ? -1 : left > right ? 1 : 0; }

function addUnique(plans: LayerTrainingPlan[], seen: Set<string>, plan: LayerTrainingPlan, graph: LayerGraph, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>): void {
  const repaired = repairLayerPlan(graph, plan, locks); const key = planKey(repaired);
  if (!seen.has(key)) { seen.add(key); plans.push(repaired); }
}

function productVectors<T>(values: T[], length: number): T[][] {
  const result: T[][] = [];
  const visit = (prefix: T[]): void => {
    if (prefix.length === length) { result.push(prefix); return; }
    values.forEach((value) => visit([...prefix, value]));
  };
  visit([]);
  return result;
}

function dedupPlans(plans: LayerTrainingPlan[]): LayerTrainingPlan[] {
  const result: LayerTrainingPlan[] = []; const seen = new Set<string>();
  plans.forEach((plan) => { const key = planKey(plan); if (!seen.has(key)) { seen.add(key); result.push(plan); } });
  return result;
}

function structuredSeedPlans(problem: LayerTrainingProblem, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>): LayerTrainingPlan[] {
  const n = problem.graph.nodes.length; const plans: LayerTrainingPlan[] = []; const seen = new Set<string>();
  const stageChoices = [...new Set([1, ...(problem.pipelineStages ?? DEFAULT_PIPELINE_STAGES)])].sort((left, right) => left - right);
  const precisionChoices: LayerPrecision[] = problem.fp8Enabled === false ? ["bf16"] : ["bf16", "fp8"];
  stageChoices.forEach((stageCount) => {
    if (stageCount > n) return;
    const stages = balancedLayerStages(n, stageCount);
    precisionChoices.forEach((precision) => (["none", "selective", "full"] as RecomputeMode[]).forEach((recompute) => [false, true].forEach((offload) => {
      addUnique(plans, seen, { stages, precision: Array(n).fill(precision), recompute: Array(n).fill(recompute), offload: Array(n).fill(offload) }, problem.graph, locks);
    })));
    const middle = Array.from({ length: n }, (_, index) => index !== 0 && index !== n - 1);
    addUnique(plans, seen, { stages, precision: middle.map((value) => value ? "fp8" : "bf16"), recompute: middle.map((value) => value ? "selective" : "none"), offload: middle }, problem.graph, locks);
  });
  return plans;
}

function randomPlan(problem: LayerTrainingProblem, rng: DeterministicRng, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>): LayerTrainingPlan {
  const n = problem.graph.nodes.length;
  const stageChoices = [...new Set([1, ...(problem.pipelineStages ?? DEFAULT_PIPELINE_STAGES)])].sort((left, right) => left - right);
  const precisionChoices: LayerPrecision[] = problem.fp8Enabled === false ? ["bf16"] : ["bf16", "fp8"];
  const stageCount = stageChoices[rng.index(stageChoices.length)] ?? 1;
  return repairLayerPlan(problem.graph, {
    stages: balancedLayerStages(n, stageCount),
    precision: Array.from({ length: n }, () => precisionChoices[rng.index(precisionChoices.length)]!),
    recompute: Array.from({ length: n }, () => (["none", "selective", "full"] as RecomputeMode[])[rng.index(3)]!),
    offload: Array.from({ length: n }, () => rng.random() < 0.5),
  }, locks);
}

function crossoverMutate(left: LayerTrainingPlan, right: LayerTrainingPlan, problem: LayerTrainingProblem, rng: DeterministicRng, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>): LayerTrainingPlan {
  const n = problem.graph.nodes.length;
  let stages = [...left.stages]; let precision = [...left.precision]; let recompute = [...left.recompute]; let offload = [...left.offload];
  if (n > 1 && rng.random() < NSGA_CROSSOVER_RATE) {
    const cut = 1 + rng.index(n - 1);
    stages = [...left.stages.slice(0, cut), ...right.stages.slice(cut)];
    precision = [...left.precision.slice(0, cut), ...right.precision.slice(cut)];
    recompute = [...left.recompute.slice(0, cut), ...right.recompute.slice(cut)];
    offload = [...left.offload.slice(0, cut), ...right.offload.slice(cut)];
  }
  const precisionChoices: LayerPrecision[] = problem.fp8Enabled === false ? ["bf16"] : ["bf16", "fp8"];
  const stageLimit = Math.max(...([...new Set([1, ...(problem.pipelineStages ?? DEFAULT_PIPELINE_STAGES)])]), 1);
  let changed = false; const mutationRate = Math.max(1 / Math.max(1, n), 0.02);
  for (let index = 0; index < n; index += 1) {
    if (rng.random() < mutationRate) { stages[index] = rng.index(stageLimit); changed = true; }
    if (rng.random() < mutationRate) { precision[index] = precisionChoices[rng.index(precisionChoices.length)]!; changed = true; }
    if (rng.random() < mutationRate) { recompute[index] = (["none", "selective", "full"] as RecomputeMode[])[rng.index(3)]!; changed = true; }
    if (rng.random() < mutationRate) { offload[index] = rng.random() < 0.5; changed = true; }
  }
  if (!changed) {
    const gene = rng.index(Math.max(1, 4 * n)); const index = Math.floor(gene / 4); const field = gene % 4;
    if (field === 0) stages[index] = rng.index(stageLimit);
    else if (field === 1) precision[index] = precisionChoices[rng.index(precisionChoices.length)]!;
    else if (field === 2) recompute[index] = (["none", "selective", "full"] as RecomputeMode[])[rng.index(3)]!;
    else offload[index] = !offload[index];
  }
  return repairLayerPlan(problem.graph, { stages, precision, recompute, offload }, locks);
}

function nsgaCandidates(problem: LayerTrainingProblem, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>): LayerTrainingPlan[] {
  const budget = Math.max(1, Math.trunc(problem.maxCandidates ?? 20000)); const n = problem.graph.nodes.length;
  const populationSize = Math.min(budget, Math.max(8, Math.min(NSGA_MAX_POPULATION, 2 * n)));
  const rng = new DeterministicRng(problem.seed ?? 20261003); const evaluated: EvaluatedLayerPlan[] = []; const seen = new Set<string>();
  const add = (raw: LayerTrainingPlan): boolean => {
    const plan = repairLayerPlan(problem.graph, raw, locks); const key = planKey(plan);
    if (seen.has(key) || evaluated.length >= budget) return false;
    seen.add(key); evaluated.push({ plan, metrics: evaluateLayerPlan(problem, plan, locks) }); return true;
  };
  for (const plan of structuredSeedPlans(problem, locks)) {
    if (evaluated.length >= populationSize) break;
    add(plan);
  }
  let attempts = 0;
  while (evaluated.length < populationSize && attempts < populationSize * 20) { add(randomPlan(problem, rng, locks)); attempts += 1; }
  let population = evaluated.slice(0, populationSize);
  while (population.length && evaluated.length < budget) {
    const { ranks, fronts } = nondominatedSort(population); const distances = crowdingDistances(population, fronts);
    const target = Math.min(populationSize, budget - evaluated.length); const offspring: EvaluatedLayerPlan[] = []; let generationAttempts = 0;
    while (offspring.length < target && generationAttempts < target * 30) {
      const left = tournament(population, rng, ranks, distances); const right = tournament(population, rng, ranks, distances);
      const child = crossoverMutate(left.plan, right.plan, problem, rng, locks); const key = planKey(child);
      if (!seen.has(key)) { seen.add(key); const record = { plan: child, metrics: evaluateLayerPlan(problem, child, locks) }; evaluated.push(record); offspring.push(record); }
      generationAttempts += 1;
    }
    if (!offspring.length) break;
    population = selectNsga([...population, ...offspring], populationSize);
  }
  return evaluated.map((item) => item.plan);
}

function candidates(problem: LayerTrainingProblem, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>): { plans: LayerTrainingPlan[]; exhaustive: boolean } {
  const n = problem.graph.nodes.length; const maxCandidates = problem.maxCandidates ?? 20000;
  const plans: LayerTrainingPlan[] = [];
  if (n <= EXHAUSTIVE_NODE_LIMIT) {
    const precisionChoices: LayerPrecision[] = problem.fp8Enabled === false ? ["bf16"] : ["bf16", "fp8"];
    const maxStageCount = Math.min(n, Math.max(1, ...(problem.pipelineStages ?? DEFAULT_PIPELINE_STAGES)));
    for (let stageCount = 1; stageCount <= maxStageCount; stageCount += 1) {
      const stages = balancedLayerStages(n, stageCount);
      for (const precision of productVectors(precisionChoices, n)) for (const recompute of productVectors(["none", "selective"] as RecomputeMode[], n)) for (const offload of productVectors([false, true], n)) {
        plans.push(repairLayerPlan(problem.graph, { stages, precision, recompute, offload }, locks));
        if (plans.length >= maxCandidates) return { plans: dedupPlans(plans), exhaustive: false };
      }
    }
    return { plans: dedupPlans(plans), exhaustive: true };
  }
  return { plans: nsgaCandidates(problem, locks), exhaustive: false };
}

function chooseGlobal(problem: LayerTrainingProblem, precision?: LayerPrecision, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>> = {}): { plan: LayerTrainingPlan; metrics: LayerMetrics } {
  const n = problem.graph.nodes.length;
  const plans: Array<{ plan: LayerTrainingPlan; metrics: LayerMetrics }> = [];
  const precisionChoices: LayerPrecision[] = precision ? [precision] : (problem.fp8Enabled === false ? ["bf16"] : ["bf16", "fp8"]);
  const stageChoices = [...new Set([1, ...(problem.pipelineStages ?? [1, 2, 4, 8, 16, 32])])].filter((count) => count <= n);
  for (const stages of stageChoices) {
    for (const selectedPrecision of precisionChoices) for (const recompute of ["none", "selective", "full"] as RecomputeMode[]) for (const offload of [false, true]) {
      const plan = repairLayerPlan(problem.graph, {
        stages: balancedLayerStages(n, stages), precision: Array(n).fill(selectedPrecision),
        recompute: Array(n).fill(recompute), offload: Array(n).fill(offload),
      }, locks);
      const metrics = evaluateLayerPlan(problem, plan, locks);
      if (feasible(metrics)) plans.push({ plan, metrics });
    }
  }
  if (!plans.length) {
    const fallback: Array<{ plan: LayerTrainingPlan; metrics: LayerMetrics }> = [];
    for (const stages of stageChoices) {
      for (const selectedPrecision of precisionChoices) {
        const plan = repairLayerPlan(problem.graph, {
          stages: balancedLayerStages(n, stages), precision: Array(n).fill(selectedPrecision),
          recompute: Array(n).fill("none"), offload: Array(n).fill(false),
        }, locks);
        fallback.push({ plan, metrics: evaluateLayerPlan(problem, plan, locks) });
      }
    }
    return fallback.sort((left, right) => Object.values(left.metrics.constraints).reduce((sum, value) => sum + Math.max(0, value), 0)
      - Object.values(right.metrics.constraints).reduce((sum, value) => sum + Math.max(0, value), 0))[0]!;
  }
  return plans.sort((left, right) => left.metrics.objectives.stepTimeS - right.metrics.objectives.stepTimeS
    || new Set(left.plan.stages).size - new Set(right.plan.stages).size
    || left.metrics.objectives.costUsdPerStep - right.metrics.objectives.costUsdPerStep)[0]!;
}

export function searchLayerTraining(problem: LayerTrainingProblem, locks: Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>> = {}): LayerSearchResult {
  const generated = candidates(problem, locks);
  const evaluated = generated.plans.map((plan) => ({ plan, metrics: evaluateLayerPlan(problem, plan, locks) }));
  const feasiblePlans = evaluated.filter((item) => feasible(item.metrics));
  const pool = feasiblePlans.length ? feasiblePlans : evaluated.slice().sort((a, b) => Object.values(a.metrics.constraints).reduce((x, y) => x + Math.max(0, y), 0) - Object.values(b.metrics.constraints).reduce((x, y) => x + Math.max(0, y), 0)).slice(0, 1);
  const front = pool.filter((item) => !pool.some((other) => other !== item && dominates(other.metrics, item.metrics)));
  let stageNote = "no feasible plan under the supplied hardware limits; showing the least-violating diagnostic";
  let best: { plan: LayerTrainingPlan; metrics: LayerMetrics } | null = null;
  if (feasiblePlans.length) {
    const fastest = [...feasiblePlans].sort((a, b) => a.metrics.objectives.stepTimeS - b.metrics.objectives.stepTimeS || a.metrics.objectives.costUsdPerStep - b.metrics.objectives.costUsdPerStep)[0]!;
    const minimumStage = [...feasiblePlans].sort((a, b) => new Set(a.plan.stages).size - new Set(b.plan.stages).size || a.metrics.objectives.stepTimeS - b.metrics.objectives.stepTimeS)[0]!;
    if (new Set(fastest.plan.stages).size > new Set(minimumStage.plan.stages).size && fastest.metrics.objectives.stepTimeS < minimumStage.metrics.objectives.stepTimeS * 0.95) {
      best = fastest;
      stageNote = `${new Set(best.plan.stages).size} stages selected because the charged pipeline/bandwidth model improves step time by more than 5% over the minimum-memory-feasible stage count`;
    } else {
      best = minimumStage;
      stageNote = "minimum feasible stage count selected; extra stages did not earn a documented cost benefit";
    }
  } else if (front.length) {
    best = front[0]!;
  }
  const globalBf16 = chooseGlobal(problem, "bf16", locks);
  const globalBest = chooseGlobal(problem, undefined, locks);
  const precisionGainPct = (globalBf16.metrics.objectives.stepTimeS - globalBest.metrics.objectives.stepTimeS) / globalBf16.metrics.objectives.stepTimeS * 100;
  const perLayerGainPct = best ? (globalBest.metrics.objectives.stepTimeS - best.metrics.objectives.stepTimeS) / globalBest.metrics.objectives.stepTimeS * 100 : 0;
  const searchNote = generated.exhaustive
    ? "Small graph search is exact exhaustive enumeration when the candidate cap is not reached"
    : "Large graph search uses deterministic repair-aware NSGA-II; the bounded Pareto set is not globally exhaustive";
  return {
    graph: problem.graph, best: best?.plan ?? null, bestMetrics: best?.metrics ?? null,
    front, baseline: globalBest, globalBf16, globalBest, precisionGainPct, perLayerGainPct,
    seed: problem.seed ?? 20261003, evaluated: evaluated.length, exhaustive: generated.exhaustive,
    assumptions: [...new Set([...problem.graph.assumptions,
      "FP8 quality is an assumption until customer evaluation",
      "CPU activation offload uses the supplied/default PCIe/host bandwidth",
      "Pipeline bubble and inter-stage communication are charged per stage",
      searchNote,
      stageNote,
    ])],
  };
}

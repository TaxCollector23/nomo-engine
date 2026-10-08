import { DTYPE_BYTES, SIMULATION_CONTRACT_VERSION, type DType, type JsonValue, type Shape, type TensorSpec } from "./contracts";
import { deriveProvenance, makeProvenance, normalizeProvenance, type Provenance } from "./provenance";

export type OperatorKind =
  | "input"
  | "embedding"
  | "gemm"
  | "attention"
  | "norm"
  | "activation"
  | "residual"
  | "mlp"
  | "moe"
  | "loss"
  | "backward"
  | "optimizer"
  | "communication"
  | "output"
  | "custom";

export type OperatorPhase = "forward" | "backward" | "update";
export type OperatorGraphMode = "inference" | "training";

export interface OperatorCost {
  flops: number;
  bytesRead: number;
  bytesWritten: number;
  parameterBytes: number;
  activationBytes: number;
  workspaceBytes: number;
}

export interface OperatorLifetime {
  firstOperatorIndex: number;
  lastOperatorIndex: number;
}

export interface OperatorNode {
  id: string;
  name: string;
  kind: OperatorKind;
  phase: OperatorPhase;
  inputs: readonly string[];
  outputs: readonly string[];
  cost: OperatorCost;
  trainable: boolean;
  lifetime?: OperatorLifetime;
  attributes: Readonly<Record<string, unknown>>;
  provenance: Provenance;
}

export interface OperatorEdge {
  id: string;
  source: string;
  target: string;
  tensorId?: string;
  bytes: number;
  kind: "data" | "control" | "gradient";
  provenance: Provenance;
}

export interface OperatorGraphAccounting {
  forwardFlops: number;
  backwardFlops: number;
  updateFlops: number;
  totalFlops: number;
  bytesRead: number;
  bytesWritten: number;
  parameterBytes: number;
  activationBytes: number;
  workspaceBytes: number;
  peakLiveBytes: number;
}

export interface OperatorGraph {
  contractVersion: typeof SIMULATION_CONTRACT_VERSION;
  id: string;
  name: string;
  source: string;
  mode: OperatorGraphMode;
  dtype: DType;
  batchSize: number;
  sequenceLength: number;
  hiddenSize: number;
  layers: number;
  attentionHeads: number;
  kvHeads: number;
  intermediateSize: number;
  vocabSize: number;
  experts: number | null;
  expertsPerToken: number | null;
  operators: readonly OperatorNode[];
  edges: readonly OperatorEdge[];
  tensors: readonly TensorSpec[];
  parameterCount: number;
  accounting: OperatorGraphAccounting;
  assumptions: readonly string[];
  provenance: Provenance;
}

export interface BuildOperatorGraphOptions {
  id?: string;
  source?: string;
  mode?: OperatorGraphMode;
  dtype?: DType;
  sequenceLength?: number;
  batchSize?: number;
  includeLoss?: boolean;
  includeBackward?: boolean;
  includeOptimizer?: boolean;
}

interface RecordLike {
  [key: string]: unknown;
}

interface MutableTensor {
  id: string;
  shape: Shape;
  dtype: DType;
  bytes: number;
  producerId?: string;
  consumerIds: string[];
  persistent?: boolean;
  metadata?: Readonly<Record<string, JsonValue>>;
}

interface MutableGraphParts {
  operators: OperatorNode[];
  edges: OperatorEdge[];
  tensors: MutableTensor[];
  tensorById: Map<string, MutableTensor>;
  operatorById: Map<string, OperatorNode>;
}

const OPERATOR_KINDS: readonly OperatorKind[] = [
  "input",
  "embedding",
  "gemm",
  "attention",
  "norm",
  "activation",
  "residual",
  "mlp",
  "moe",
  "loss",
  "backward",
  "optimizer",
  "communication",
  "output",
  "custom",
] as const;

function asRecord(value: unknown): RecordLike {
  return typeof value === "object" && value !== null ? value as RecordLike : {};
}

function first(record: RecordLike, ...keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key];
  }
  return undefined;
}

function positiveInteger(value: unknown, label: string, fallback?: number): number {
  const number = Number(value ?? fallback);
  if (!Number.isFinite(number) || number <= 0) throw new Error(`${label} must be a positive number`);
  return Math.floor(number);
}

function nonNegative(value: unknown, label: string, fallback = 0): number {
  const number = Number(value ?? fallback);
  if (!Number.isFinite(number) || number < 0) throw new Error(`${label} must be a non-negative number`);
  return number;
}

function dtype(value: unknown, fallback: DType): DType {
  const candidate = String(value ?? fallback).toLowerCase();
  if (candidate === "float32") return "fp32";
  if (candidate === "float16") return "fp16";
  if (candidate === "bfloat16") return "bf16";
  if (candidate === "float8") return "fp8";
  if (candidate === "int4" || candidate === "int8" || candidate === "fp32" || candidate === "fp16" || candidate === "bf16" || candidate === "fp8") {
    return candidate;
  }
  throw new Error(`unsupported graph dtype: ${candidate}`);
}

function shapeSize(shape: Shape): number {
  if (!shape.length) return 1;
  return shape.reduce((product, dimension) => product * positiveInteger(dimension, "tensor dimension"), 1);
}

function tensorBytes(shape: Shape, tensorDtype: DType): number {
  return shapeSize(shape) * DTYPE_BYTES[tensorDtype];
}

function makeCost(
  flops: number,
  inputBytes: number,
  outputBytes: number,
  parameterBytes = 0,
  workspaceBytes = 0,
): OperatorCost {
  [flops, inputBytes, outputBytes, parameterBytes, workspaceBytes].forEach((value) => {
    if (!Number.isFinite(value) || value < 0) throw new Error("operator costs must be finite and non-negative");
  });
  return {
    flops,
    bytesRead: inputBytes + parameterBytes,
    bytesWritten: outputBytes,
    parameterBytes,
    activationBytes: outputBytes,
    workspaceBytes,
  };
}

function newParts(): MutableGraphParts {
  return { operators: [], edges: [], tensors: [], tensorById: new Map(), operatorById: new Map() };
}

function addTensor(parts: MutableGraphParts, id: string, shape: Shape, tensorDtype: DType, persistent = false, metadata?: Readonly<Record<string, JsonValue>>): MutableTensor {
  if (parts.tensorById.has(id)) throw new Error(`duplicate tensor id: ${id}`);
  const tensor: MutableTensor = {
    id,
    shape: [...shape],
    dtype: tensorDtype,
    bytes: tensorBytes(shape, tensorDtype),
    consumerIds: [],
    persistent,
    metadata,
  };
  parts.tensors.push(tensor);
  parts.tensorById.set(id, tensor);
  return tensor;
}

function addOperator(
  parts: MutableGraphParts,
  spec: Omit<OperatorNode, "inputs" | "outputs"> & { inputs?: readonly string[]; output?: { id: string; shape: Shape; dtype: DType; persistent?: boolean } },
): OperatorNode {
  if (parts.operatorById.has(spec.id)) throw new Error(`duplicate operator id: ${spec.id}`);
  const inputs = [...(spec.inputs ?? [])];
  for (const tensorId of inputs) {
    const tensor = parts.tensorById.get(tensorId);
    if (!tensor) throw new Error(`operator ${spec.id} references unknown tensor ${tensorId}`);
    tensor.consumerIds.push(spec.id);
  }
  const outputs: string[] = [];
  if (spec.output) {
    const output = parts.tensorById.get(spec.output.id) ?? addTensor(parts, spec.output.id, spec.output.shape, spec.output.dtype, spec.output.persistent);
    if (output.producerId !== undefined) throw new Error(`tensor ${spec.output.id} already has a producer`);
    output.producerId = spec.id;
    outputs.push(output.id);
  }
  const node: OperatorNode = { ...spec, inputs, outputs };
  parts.operators.push(node);
  parts.operatorById.set(node.id, node);
  for (const tensorId of inputs) {
    const tensor = parts.tensorById.get(tensorId);
    if (!tensor?.producerId) continue;
    parts.edges.push({
      id: `${tensor.producerId}->${node.id}:${tensorId}`,
      source: tensor.producerId,
      target: node.id,
      tensorId,
      bytes: tensor.bytes,
      kind: node.phase === "backward" ? "gradient" : "data",
      provenance: deriveProvenance([tensorProvenance(tensor)], "operator data dependency"),
    });
  }
  return node;
}

function tensorProvenance(tensor: MutableTensor): Provenance {
  return makeProvenance(tensor.persistent ? "spec" : "derived", tensor.persistent ? "model graph parameters" : "operator output");
}

function nodeProvenance(source: string, kind: "spec" | "assumption" | "derived" = "derived"): Provenance {
  return makeProvenance(kind, source);
}

function graphShapeForConfig(record: RecordLike, options: BuildOperatorGraphOptions): {
  name: string;
  source: string;
  mode: OperatorGraphMode;
  tensorDtype: DType;
  batchSize: number;
  sequenceLength: number;
  hiddenSize: number;
  layers: number;
  attentionHeads: number;
  kvHeads: number;
  intermediateSize: number;
  vocabSize: number;
  experts: number | null;
  expertsPerToken: number | null;
  assumptions: string[];
} {
  const sequenceLength = positiveInteger(first(record, "sequenceLength", "sequence_length", "seqLen", "seq_len") ?? options.sequenceLength ?? 2048, "sequence length");
  const batchSize = positiveInteger(first(record, "batchSize", "batch_size") ?? options.batchSize ?? 1, "batch size");
  const layers = positiveInteger(first(record, "layers", "num_hidden_layers", "num_layers", "n_layer"), "layers");
  const hiddenSize = positiveInteger(first(record, "hiddenSize", "hidden_size", "d_model", "n_embd"), "hidden size");
  const attentionHeads = positiveInteger(first(record, "attentionHeads", "attention_heads", "num_attention_heads", "num_heads", "n_head"), "attention heads");
  if (hiddenSize % attentionHeads !== 0) throw new Error("hidden size must be divisible by attention heads");
  const kvHeads = positiveInteger(first(record, "kvHeads", "kv_heads", "num_key_value_heads", "n_kv_heads") ?? attentionHeads, "KV heads");
  if (attentionHeads % kvHeads !== 0) throw new Error("KV heads must divide attention heads");
  const intermediateValue = first(record, "intermediateSize", "intermediate_size", "n_inner");
  const intermediateSize = positiveInteger(intermediateValue ?? hiddenSize * 4, "intermediate size");
  const vocabSize = positiveInteger(first(record, "vocabSize", "vocab_size", "n_vocab"), "vocabulary size");
  const expertsValue = first(record, "experts", "num_local_experts", "num_experts");
  const experts = expertsValue === undefined ? null : positiveInteger(expertsValue, "experts");
  const expertsPerTokenValue = first(record, "expertsPerToken", "experts_per_token", "num_experts_per_tok", "moe_top_k");
  const expertsPerToken = expertsPerTokenValue === undefined ? null : positiveInteger(expertsPerTokenValue, "experts per token");
  if (experts !== null && (expertsPerToken === null || expertsPerToken > experts)) throw new Error("experts per token must be between 1 and experts");
  const assumptions: string[] = [];
  if (intermediateValue === undefined) assumptions.push("intermediate size inferred as 4 * hidden size");
  if (first(record, "kvHeads", "kv_heads", "num_key_value_heads", "n_kv_heads") === undefined) assumptions.push("KV heads inferred as attention heads");
  if (experts !== null) assumptions.push("MoE parameter memory includes resident experts; FLOPs use active experts");
  const mode = (options.mode ?? String(first(record, "mode") ?? "training")) as OperatorGraphMode;
  if (mode !== "training" && mode !== "inference") throw new Error(`unsupported graph mode: ${mode}`);
  return {
    name: String(first(record, "name", "_name_or_path", "model_type") ?? "uploaded-transformer"),
    source: String(first(record, "source") ?? options.source ?? "browser-config"),
    mode,
    tensorDtype: dtype(first(record, "dtype", "precision") ?? options.dtype, options.dtype ?? "bf16"),
    batchSize,
    sequenceLength,
    hiddenSize,
    layers,
    attentionHeads,
    kvHeads,
    intermediateSize,
    vocabSize,
    experts,
    expertsPerToken,
    assumptions,
  };
}

function buildFromConfig(record: RecordLike, options: BuildOperatorGraphOptions): OperatorGraph {
  const shape = graphShapeForConfig(record, options);
  const parts = newParts();
  const tokens = shape.batchSize * shape.sequenceLength;
  const hidden = shape.hiddenSize;
  const headDim = hidden / shape.attentionHeads;
  const kvDim = shape.kvHeads * headDim;
  const elementBytes = DTYPE_BYTES[shape.tensorDtype];
  const activationShape: Shape = [shape.batchSize, shape.sequenceLength, hidden];
  const activationBytes = tensorBytes(activationShape, shape.tensorDtype);
  const input = addOperator(parts, {
    id: "input.tokens",
    name: "Token input",
    kind: "input",
    phase: "forward",
    cost: makeCost(0, 0, tokens * DTYPE_BYTES.int8 /* token ids are represented as compact input bytes */),
    trainable: false,
    attributes: { shape: [shape.batchSize, shape.sequenceLength], role: "token_ids" },
    provenance: makeProvenance("user", "input contract"),
    output: { id: "tensor.input.tokens", shape: [shape.batchSize, shape.sequenceLength], dtype: "int8" },
  });
  let current = input.outputs[0];
  const trainableOperators: OperatorNode[] = [];
  const addForward = (
    spec: Omit<OperatorNode, "inputs" | "outputs">,
    outputShape: Shape,
    outputId: string,
  ): OperatorNode => {
    const node = addOperator(parts, {
      ...spec,
      inputs: current ? [current] : [],
      output: { id: outputId, shape: outputShape, dtype: shape.tensorDtype },
    });
    current = node.outputs[0];
    if (node.trainable) trainableOperators.push(node);
    return node;
  };

  const embedding = addForward({
    id: "embedding",
    name: "Token embedding",
    kind: "embedding",
    phase: "forward",
    cost: makeCost(0, tensorBytes([shape.batchSize, shape.sequenceLength], "int8"), activationBytes, shape.vocabSize * hidden * elementBytes),
    trainable: true,
    attributes: { vocabSize: shape.vocabSize, hiddenSize: hidden },
    provenance: nodeProvenance("model config", "spec"),
  }, activationShape, "tensor.embedding");
  current = embedding.outputs[0];

  for (let layer = 0; layer < shape.layers; layer += 1) {
    const prefix = `block.${layer}`;
    addForward({
      id: `${prefix}.norm1`,
      name: `${prefix} pre-attention norm`,
      kind: "norm",
      phase: "forward",
      cost: makeCost(tokens * hidden * 5, activationBytes, activationBytes, 2 * hidden * elementBytes),
      trainable: true,
      attributes: { norm: String(first(record, "norm", "normalization") ?? "rms_norm") },
      provenance: nodeProvenance("model config", "assumption"),
    }, activationShape, `tensor.${prefix}.norm1`);
    addForward({
      id: `${prefix}.qkv`,
      name: `${prefix} QKV projection`,
      kind: "gemm",
      phase: "forward",
      cost: makeCost(2 * tokens * hidden * (hidden + 2 * kvDim), activationBytes, activationBytes, hidden * (hidden + 2 * kvDim) * elementBytes),
      trainable: true,
      attributes: { inputSize: hidden, outputSize: hidden + 2 * kvDim, heads: shape.attentionHeads, kvHeads: shape.kvHeads },
      provenance: nodeProvenance("model config", "spec"),
    }, activationShape, `tensor.${prefix}.qkv`);
    addForward({
      id: `${prefix}.attention`,
      name: `${prefix} scaled dot-product attention`,
      kind: "attention",
      phase: "forward",
      cost: makeCost(4 * shape.batchSize * shape.sequenceLength * shape.sequenceLength * hidden, activationBytes, activationBytes, 0, activationBytes),
      trainable: false,
      attributes: { heads: shape.attentionHeads, kvHeads: shape.kvHeads, headDim, causal: true },
      provenance: nodeProvenance("attention accounting", "assumption"),
    }, activationShape, `tensor.${prefix}.attention`);
    addForward({
      id: `${prefix}.attn-proj`,
      name: `${prefix} attention output projection`,
      kind: "gemm",
      phase: "forward",
      cost: makeCost(2 * tokens * hidden * hidden, activationBytes, activationBytes, hidden * hidden * elementBytes),
      trainable: true,
      attributes: { inputSize: hidden, outputSize: hidden },
      provenance: nodeProvenance("model config", "spec"),
    }, activationShape, `tensor.${prefix}.attn-proj`);
    addForward({
      id: `${prefix}.residual1`,
      name: `${prefix} attention residual`,
      kind: "residual",
      phase: "forward",
      cost: makeCost(tokens * hidden, activationBytes * 2, activationBytes),
      trainable: false,
      attributes: { branch: "attention" },
      provenance: nodeProvenance("residual accounting", "assumption"),
    }, activationShape, `tensor.${prefix}.residual1`);
    addForward({
      id: `${prefix}.norm2`,
      name: `${prefix} pre-MLP norm`,
      kind: "norm",
      phase: "forward",
      cost: makeCost(tokens * hidden * 5, activationBytes, activationBytes, 2 * hidden * elementBytes),
      trainable: true,
      attributes: { norm: String(first(record, "norm", "normalization") ?? "rms_norm") },
      provenance: nodeProvenance("model config", "assumption"),
    }, activationShape, `tensor.${prefix}.norm2`);
    const expertMultiplier = shape.experts ?? 1;
    const activeMultiplier = shape.expertsPerToken ?? 1;
    const matrices = Boolean(first(record, "gatedMlp", "gated_mlp")) || ["silu", "swish", "geglu"].includes(String(first(record, "hiddenAct", "hidden_act") ?? "").toLowerCase()) ? 3 : 2;
    const mlpKind: OperatorKind = shape.experts === null ? "mlp" : "moe";
    addForward({
      id: `${prefix}.mlp`,
      name: `${prefix} ${mlpKind === "moe" ? "mixture-of-experts" : "MLP"}`,
      kind: mlpKind,
      phase: "forward",
      cost: makeCost(2 * tokens * matrices * hidden * shape.intermediateSize * activeMultiplier, activationBytes, activationBytes, matrices * hidden * shape.intermediateSize * expertMultiplier * elementBytes, activationBytes),
      trainable: true,
      attributes: { matrices, gated: matrices === 3, experts: shape.experts, activeExperts: shape.expertsPerToken, residentExpertMultiplier: expertMultiplier },
      provenance: nodeProvenance(shape.experts === null ? "MLP accounting" : "MoE accounting", shape.experts === null ? "spec" : "assumption"),
    }, activationShape, `tensor.${prefix}.mlp`);
    addForward({
      id: `${prefix}.residual2`,
      name: `${prefix} MLP residual`,
      kind: "residual",
      phase: "forward",
      cost: makeCost(tokens * hidden, activationBytes * 2, activationBytes),
      trainable: false,
      attributes: { branch: "mlp" },
      provenance: nodeProvenance("residual accounting", "assumption"),
    }, activationShape, `tensor.${prefix}.residual2`);
  }

  addForward({
    id: "final.norm",
    name: "Final normalization",
    kind: "norm",
    phase: "forward",
    cost: makeCost(tokens * hidden * 5, activationBytes, activationBytes, 2 * hidden * elementBytes),
    trainable: true,
    attributes: { norm: String(first(record, "norm", "normalization") ?? "rms_norm") },
    provenance: nodeProvenance("model config", "assumption"),
  }, activationShape, "tensor.final.norm");
  const logitsShape: Shape = [shape.batchSize, shape.sequenceLength, shape.vocabSize];
  const logitsBytes = tensorBytes(logitsShape, shape.tensorDtype);
  addForward({
    id: "output.logits",
    name: "Output projection",
    kind: "output",
    phase: "forward",
    cost: makeCost(2 * tokens * hidden * shape.vocabSize, activationBytes, logitsBytes, shape.vocabSize * hidden * elementBytes),
    trainable: true,
    attributes: { vocabSize: shape.vocabSize, tiedEmbeddings: Boolean(first(record, "tiedEmbeddings", "tie_word_embeddings", "tie_embeddings") ?? false) },
    provenance: nodeProvenance("model config", "spec"),
  }, logitsShape, "tensor.output.logits");

  const includeLoss = options.includeLoss ?? shape.mode === "training";
  const includeBackward = options.includeBackward ?? shape.mode === "training";
  const includeOptimizer = options.includeOptimizer ?? shape.mode === "training";
  let lossOutput: string | undefined;
  if (includeLoss) {
    const loss = addForward({
      id: "loss.cross-entropy",
      name: "Token cross-entropy loss",
      kind: "loss",
      phase: "forward",
      cost: makeCost(2 * tokens * shape.vocabSize, logitsBytes, DTYPE_BYTES.fp32, 0, DTYPE_BYTES.fp32),
      trainable: false,
      attributes: { reduction: "mean", labels: "token_ids" },
      provenance: nodeProvenance("loss accounting", "assumption"),
    }, [1], "tensor.loss");
    lossOutput = loss.outputs[0];
  }

  if (includeBackward && lossOutput) {
    let gradient = lossOutput;
    const gradientOutputs: string[] = [];
    for (const forward of [...trainableOperators].reverse()) {
      const inputTensor = forward.inputs[0];
      const inputShape = inputTensor ? parts.tensorById.get(inputTensor)?.shape ?? [1] : [1];
      const backward = addOperator(parts, {
        id: `backward.${forward.id}`,
        name: `Gradient for ${forward.name}`,
        kind: "backward",
        phase: "backward",
        inputs: [gradient],
        cost: makeCost(Math.max(forward.cost.flops * 2, 0), parts.tensorById.get(gradient)?.bytes ?? DTYPE_BYTES.fp32, tensorBytes(inputShape, shape.tensorDtype), forward.cost.parameterBytes),
        trainable: false,
        attributes: { forwardOperatorId: forward.id },
        provenance: deriveProvenance([forward.provenance], "backward operator derived from forward operator"),
        output: { id: `tensor.backward.${forward.id}`, shape: inputShape, dtype: shape.tensorDtype },
      });
      gradient = backward.outputs[0];
      gradientOutputs.push(gradient);
    }
    if (includeOptimizer) {
      addOperator(parts, {
        id: "optimizer.update",
        name: "Optimizer parameter update",
        kind: "optimizer",
        phase: "update",
        inputs: gradientOutputs,
        cost: makeCost(Math.max(1, gradientOutputs.length) * hidden, gradientOutputs.reduce((sum, id) => sum + (parts.tensorById.get(id)?.bytes ?? 0), 0), 0, 0, Math.max(1, graphParameterBytes(parts)) * 2),
        trainable: false,
        attributes: { optimizer: String(first(record, "optimizer") ?? "adamw"), serverOnly: false },
        provenance: nodeProvenance("optimizer accounting", "assumption"),
      });
    }
  }

  return finalizeGraph({
    id: String(first(record, "id") ?? options.id ?? slug(shape.name)),
    name: shape.name,
    source: shape.source,
    mode: shape.mode,
    dtype: shape.tensorDtype,
    batchSize: shape.batchSize,
    sequenceLength: shape.sequenceLength,
    hiddenSize: shape.hiddenSize,
    layers: shape.layers,
    attentionHeads: shape.attentionHeads,
    kvHeads: shape.kvHeads,
    intermediateSize: shape.intermediateSize,
    vocabSize: shape.vocabSize,
    experts: shape.experts,
    expertsPerToken: shape.expertsPerToken,
    parts,
    assumptions: shape.assumptions,
    provenance: makeProvenance("spec", shape.source),
  });
}

function slug(value: string): string {
  const result = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return result || "operator-graph";
}

function graphParameterBytes(parts: MutableGraphParts): number {
  return parts.operators.reduce((sum, operator) => sum + operator.cost.parameterBytes, 0);
}

function normalizeCost(value: unknown, fallback: Partial<OperatorCost> = {}): OperatorCost {
  const record = asRecord(value);
  const flops = nonNegative(first(record, "flops", "forwardFlops", "forward_flops") ?? fallback.flops, "operator flops");
  const bytesRead = nonNegative(first(record, "bytesRead", "bytes_read") ?? fallback.bytesRead, "operator bytesRead");
  const bytesWritten = nonNegative(first(record, "bytesWritten", "bytes_written") ?? fallback.bytesWritten, "operator bytesWritten");
  const parameterBytes = nonNegative(first(record, "parameterBytes", "parameter_bytes") ?? fallback.parameterBytes, "operator parameterBytes");
  const activationBytes = nonNegative(first(record, "activationBytes", "activation_bytes") ?? fallback.activationBytes ?? bytesWritten, "operator activationBytes");
  const workspaceBytes = nonNegative(first(record, "workspaceBytes", "workspace_bytes") ?? fallback.workspaceBytes, "operator workspaceBytes");
  return { flops, bytesRead, bytesWritten, parameterBytes, activationBytes, workspaceBytes };
}

function normalizeKind(value: unknown): OperatorKind {
  const candidate = String(value ?? "custom");
  return OPERATOR_KINDS.includes(candidate as OperatorKind) ? candidate as OperatorKind : "custom";
}

function normalizePhase(value: unknown): OperatorPhase {
  const candidate = String(value ?? "forward");
  if (candidate === "forward" || candidate === "backward" || candidate === "update") return candidate;
  throw new Error(`unsupported operator phase: ${candidate}`);
}

function parseOperatorGraph(raw: RecordLike): OperatorGraph {
  const rawOperators = Array.isArray(raw.operators) ? raw.operators : Array.isArray(raw.nodes) ? raw.nodes : [];
  if (!rawOperators.length) throw new Error("operator graph contract must contain operators");
  const parts = newParts();
  const rawTensors = Array.isArray(raw.tensors) ? raw.tensors : [];
  for (const rawTensor of rawTensors) {
    const tensor = asRecord(rawTensor);
    const id = String(first(tensor, "id", "tensorId", "tensor_id") ?? "");
    if (!id) throw new Error("tensor contract is missing id");
    const rawShape = Array.isArray(first(tensor, "shape")) ? first(tensor, "shape") as unknown[] : [1];
    const tensorDtype = dtype(first(tensor, "dtype") ?? raw.dtype, "bf16");
    addTensor(parts, id, rawShape.map((value) => positiveInteger(value, "tensor dimension")), tensorDtype, tensor.persistent === true);
  }
  for (const rawOperator of rawOperators) {
    const operator = asRecord(rawOperator);
    const id = String(first(operator, "id") ?? "");
    if (!id) throw new Error("operator contract is missing id");
    const inputIds = Array.isArray(first(operator, "inputs", "input_ids", "inputTensorIds", "input_tensor_ids"))
      ? (first(operator, "inputs", "input_ids", "inputTensorIds", "input_tensor_ids") as unknown[]).map(String)
      : [];
    const outputIds = Array.isArray(first(operator, "outputs", "output_ids", "outputTensorIds", "output_tensor_ids"))
      ? (first(operator, "outputs", "output_ids", "outputTensorIds", "output_tensor_ids") as unknown[]).map(String)
      : [];
    for (const inputId of inputIds) {
      if (!parts.tensorById.has(inputId)) addTensor(parts, inputId, [1], "bf16");
    }
    const output = outputIds[0] ? parts.tensorById.get(outputIds[0]) ?? addTensor(parts, outputIds[0], [1], "bf16") : undefined;
    addOperator(parts, {
      id,
      name: String(first(operator, "name") ?? id),
      kind: normalizeKind(first(operator, "kind", "type")),
      phase: normalizePhase(first(operator, "phase")),
      inputs: inputIds,
      cost: normalizeCost(first(operator, "cost") ?? operator, { activationBytes: output?.bytes }),
      trainable: operator.trainable === true,
      attributes: asRecord(first(operator, "attributes", "metadata")),
      provenance: normalizeProvenance(operator.provenance, makeProvenance("assumption", String(raw.source ?? "operator graph contract"))),
      output: output ? { id: output.id, shape: output.shape, dtype: output.dtype, persistent: output.persistent } : undefined,
    });
  }
  // Explicit contract edges are retained when present; derived tensor edges are
  // deterministic and are the source of truth for browser scheduling.
  if (Array.isArray(raw.edges)) {
    const explicit = raw.edges.map((value, index) => {
      const edge = asRecord(value);
      return {
        id: String(first(edge, "id") ?? `edge.${index}`),
        source: String(first(edge, "source", "from") ?? ""),
        target: String(first(edge, "target", "to") ?? ""),
        tensorId: first(edge, "tensorId", "tensor_id") === undefined ? undefined : String(first(edge, "tensorId", "tensor_id")),
        bytes: nonNegative(first(edge, "bytes") ?? 0, "edge bytes"),
        kind: String(first(edge, "kind") ?? "data") === "gradient" ? "gradient" as const : String(first(edge, "kind") ?? "data") === "control" ? "control" as const : "data" as const,
        provenance: normalizeProvenance(edge.provenance, makeProvenance("derived", "operator graph contract")),
      };
    });
    const derivedIds = new Set(parts.edges.map((edge) => edge.id));
    parts.edges.push(...explicit.filter((edge) => !derivedIds.has(edge.id)));
  }
  const tensorDtype = dtype(raw.dtype, "bf16");
  return finalizeGraph({
    id: String(first(raw, "id") ?? "operator-graph"),
    name: String(first(raw, "name") ?? "operator-graph"),
    source: String(first(raw, "source") ?? "operator-graph-contract"),
    mode: (String(first(raw, "mode") ?? "training") === "inference" ? "inference" : "training"),
    dtype: tensorDtype,
    batchSize: positiveInteger(first(raw, "batchSize", "batch_size") ?? 1, "batch size"),
    sequenceLength: positiveInteger(first(raw, "sequenceLength", "sequence_length", "seqLen", "seq_len") ?? 1, "sequence length"),
    hiddenSize: positiveInteger(first(raw, "hiddenSize", "hidden_size") ?? 1, "hidden size"),
    layers: positiveInteger(first(raw, "layers") ?? 1, "layers"),
    attentionHeads: positiveInteger(first(raw, "attentionHeads", "attention_heads") ?? 1, "attention heads"),
    kvHeads: positiveInteger(first(raw, "kvHeads", "kv_heads") ?? 1, "KV heads"),
    intermediateSize: positiveInteger(first(raw, "intermediateSize", "intermediate_size") ?? 1, "intermediate size"),
    vocabSize: positiveInteger(first(raw, "vocabSize", "vocab_size") ?? 1, "vocabulary size"),
    experts: first(raw, "experts") === null || first(raw, "experts") === undefined ? null : positiveInteger(first(raw, "experts"), "experts"),
    expertsPerToken: first(raw, "expertsPerToken", "experts_per_token") === null || first(raw, "expertsPerToken", "experts_per_token") === undefined ? null : positiveInteger(first(raw, "expertsPerToken", "experts_per_token"), "experts per token"),
    parts,
    assumptions: Array.isArray(raw.assumptions) ? raw.assumptions.map(String) : [],
    provenance: normalizeProvenance(raw.provenance, makeProvenance("assumption", String(raw.source ?? "operator graph contract"))),
  });
}

function finalizeGraph(input: {
  id: string;
  name: string;
  source: string;
  mode: OperatorGraphMode;
  dtype: DType;
  batchSize: number;
  sequenceLength: number;
  hiddenSize: number;
  layers: number;
  attentionHeads: number;
  kvHeads: number;
  intermediateSize: number;
  vocabSize: number;
  experts: number | null;
  expertsPerToken: number | null;
  parts: MutableGraphParts;
  assumptions: readonly string[];
  provenance: Provenance;
}): OperatorGraph {
  const operatorIndex = new Map(input.parts.operators.map((operator, index) => [operator.id, index]));
  const tensors: TensorSpec[] = input.parts.tensors.map((tensor) => {
    const producerIndex = tensor.producerId === undefined ? 0 : operatorIndex.get(tensor.producerId) ?? 0;
    const lastConsumerIndex = tensor.consumerIds.reduce((last, consumer) => Math.max(last, operatorIndex.get(consumer) ?? producerIndex), producerIndex);
    return {
      id: tensor.id,
      shape: tensor.shape,
      dtype: tensor.dtype,
      bytes: tensor.bytes,
      producerId: tensor.producerId,
      consumerIds: tensor.consumerIds,
      lifetime: { firstOperatorIndex: producerIndex, lastOperatorIndex: lastConsumerIndex },
      persistent: tensor.persistent,
      metadata: tensor.metadata,
    };
  });
  const tensorById = new Map(tensors.map((tensor) => [tensor.id, tensor]));
  const operators = input.parts.operators.map((operator, index) => {
    const outputBytes = operator.outputs.reduce((sum, tensorId) => sum + (tensorById.get(tensorId)?.bytes ?? 0), 0);
    const lifetimeValues = operator.outputs.map((tensorId) => tensorById.get(tensorId)?.lifetime).filter((value): value is NonNullable<typeof value> => value !== undefined);
    return {
      ...operator,
      lifetime: lifetimeValues.length ? {
        firstOperatorIndex: Math.min(...lifetimeValues.map((value) => value.firstOperatorIndex)),
        lastOperatorIndex: Math.max(...lifetimeValues.map((value) => value.lastOperatorIndex)),
      } : { firstOperatorIndex: index, lastOperatorIndex: index },
      cost: { ...operator.cost, activationBytes: operator.cost.activationBytes || outputBytes },
    };
  });
  const accounting = calculateGraphAccounting(operators, tensors);
  const parameterCount = Math.round(operators.reduce((sum, operator) => sum + operator.cost.parameterBytes, 0) / DTYPE_BYTES[input.dtype]);
  const graph: OperatorGraph = {
    contractVersion: SIMULATION_CONTRACT_VERSION,
    id: input.id,
    name: input.name,
    source: input.source,
    mode: input.mode,
    dtype: input.dtype,
    batchSize: input.batchSize,
    sequenceLength: input.sequenceLength,
    hiddenSize: input.hiddenSize,
    layers: input.layers,
    attentionHeads: input.attentionHeads,
    kvHeads: input.kvHeads,
    intermediateSize: input.intermediateSize,
    vocabSize: input.vocabSize,
    experts: input.experts,
    expertsPerToken: input.expertsPerToken,
    operators,
    edges: input.parts.edges,
    tensors,
    parameterCount,
    accounting,
    assumptions: [...new Set(input.assumptions.map(String))],
    provenance: input.provenance,
  };
  validateOperatorGraph(graph);
  return graph;
}

export function calculateGraphAccounting(operators: readonly OperatorNode[], tensors: readonly TensorSpec[]): OperatorGraphAccounting {
  const byPhase = (phase: OperatorPhase) => operators.filter((operator) => operator.phase === phase).reduce((sum, operator) => sum + operator.cost.flops, 0);
  const activationBytes = operators.reduce((sum, operator) => sum + operator.cost.activationBytes, 0);
  const workspaceBytes = operators.reduce((sum, operator) => sum + operator.cost.workspaceBytes, 0);
  let peakLiveBytes = 0;
  const maxIndex = Math.max(0, operators.length - 1);
  for (let index = 0; index <= maxIndex; index += 1) {
    const live = tensors.reduce((sum, tensor) => {
      const lifetime = tensor.lifetime;
      return lifetime && !tensor.persistent && lifetime.firstOperatorIndex <= index && index <= lifetime.lastOperatorIndex ? sum + tensor.bytes : sum;
    }, 0);
    peakLiveBytes = Math.max(peakLiveBytes, live);
  }
  const forwardFlops = byPhase("forward");
  const backwardFlops = byPhase("backward");
  const updateFlops = byPhase("update");
  return {
    forwardFlops,
    backwardFlops,
    updateFlops,
    totalFlops: forwardFlops + backwardFlops + updateFlops,
    bytesRead: operators.reduce((sum, operator) => sum + operator.cost.bytesRead, 0),
    bytesWritten: operators.reduce((sum, operator) => sum + operator.cost.bytesWritten, 0),
    parameterBytes: operators.reduce((sum, operator) => sum + operator.cost.parameterBytes, 0),
    activationBytes,
    workspaceBytes,
    peakLiveBytes,
  };
}

export function topologicalOrder(graph: OperatorGraph): OperatorNode[] {
  const byId = new Map(graph.operators.map((operator) => [operator.id, operator]));
  const indegree = new Map(graph.operators.map((operator) => [operator.id, 0]));
  const outgoing = new Map(graph.operators.map((operator) => [operator.id, [] as string[]]));
  for (const edge of graph.edges) {
    if (!byId.has(edge.source) || !byId.has(edge.target)) throw new Error(`edge ${edge.id} references an unknown operator`);
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1);
    outgoing.get(edge.source)?.push(edge.target);
  }
  const originalOrder = new Map(graph.operators.map((operator, index) => [operator.id, index]));
  const ready = graph.operators.filter((operator) => indegree.get(operator.id) === 0).map((operator) => operator.id);
  const result: OperatorNode[] = [];
  while (ready.length) {
    ready.sort((left, right) => (originalOrder.get(left) ?? 0) - (originalOrder.get(right) ?? 0) || left.localeCompare(right));
    const id = ready.shift() as string;
    result.push(byId.get(id) as OperatorNode);
    for (const target of outgoing.get(id) ?? []) {
      const next = (indegree.get(target) ?? 0) - 1;
      indegree.set(target, next);
      if (next === 0) ready.push(target);
    }
  }
  if (result.length !== graph.operators.length) throw new Error("operator graph contains a cycle");
  return result;
}

export function validateOperatorGraph(graph: OperatorGraph): void {
  if (graph.contractVersion !== SIMULATION_CONTRACT_VERSION) throw new Error("unsupported operator graph contract version");
  const operatorIds = new Set<string>();
  for (const operator of graph.operators) {
    if (operatorIds.has(operator.id)) throw new Error(`duplicate operator id: ${operator.id}`);
    operatorIds.add(operator.id);
    if (!operator.id || !operator.name) throw new Error("operators need ids and names");
    Object.values(operator.cost).forEach((value) => {
      if (!Number.isFinite(value) || value < 0) throw new Error(`operator ${operator.id} has invalid cost`);
    });
  }
  const tensorIds = new Set<string>();
  for (const tensor of graph.tensors) {
    if (tensorIds.has(tensor.id)) throw new Error(`duplicate tensor id: ${tensor.id}`);
    tensorIds.add(tensor.id);
    if (!Number.isFinite(tensor.bytes) || tensor.bytes < 0) throw new Error(`tensor ${tensor.id} has invalid bytes`);
  }
  const edgeIds = new Set<string>();
  for (const edge of graph.edges) {
    if (edgeIds.has(edge.id)) throw new Error(`duplicate edge id: ${edge.id}`);
    edgeIds.add(edge.id);
    if (!operatorIds.has(edge.source) || !operatorIds.has(edge.target)) throw new Error(`edge ${edge.id} references an unknown operator`);
    if (edge.tensorId !== undefined && !tensorIds.has(edge.tensorId)) throw new Error(`edge ${edge.id} references an unknown tensor`);
  }
  topologicalOrder(graph);
}

export function buildOperatorGraph(raw: unknown, options: BuildOperatorGraphOptions = {}): OperatorGraph {
  const record = asRecord(raw);
  if (Array.isArray(record.operators) || Array.isArray(record.nodes)) return parseOperatorGraph(record);
  return buildFromConfig(record, options);
}

export function buildGraph(raw: unknown, options: BuildOperatorGraphOptions = {}): OperatorGraph {
  return buildOperatorGraph(raw, options);
}

export function fromOperatorGraphContract(raw: unknown): OperatorGraph {
  return parseOperatorGraph(asRecord(raw));
}

export function operatorGraphToJSON(graph: OperatorGraph): Record<string, unknown> {
  validateOperatorGraph(graph);
  return {
    contract_version: graph.contractVersion,
    id: graph.id,
    name: graph.name,
    source: graph.source,
    mode: graph.mode,
    dtype: graph.dtype,
    batch_size: graph.batchSize,
    sequence_length: graph.sequenceLength,
    hidden_size: graph.hiddenSize,
    layers: graph.layers,
    attention_heads: graph.attentionHeads,
    kv_heads: graph.kvHeads,
    intermediate_size: graph.intermediateSize,
    vocab_size: graph.vocabSize,
    experts: graph.experts,
    experts_per_token: graph.expertsPerToken,
    operators: graph.operators.map((operator) => ({
      id: operator.id,
      name: operator.name,
      kind: operator.kind,
      phase: operator.phase,
      inputs: [...operator.inputs],
      outputs: [...operator.outputs],
      cost: {
        flops: operator.cost.flops,
        bytes_read: operator.cost.bytesRead,
        bytes_written: operator.cost.bytesWritten,
        parameter_bytes: operator.cost.parameterBytes,
        activation_bytes: operator.cost.activationBytes,
        workspace_bytes: operator.cost.workspaceBytes,
      },
      trainable: operator.trainable,
      lifetime: operator.lifetime,
      attributes: operator.attributes,
      provenance: operator.provenance,
    })),
    edges: graph.edges.map((edge) => ({ ...edge, tensor_id: edge.tensorId })),
    tensors: graph.tensors.map((tensor) => ({
      id: tensor.id,
      shape: [...tensor.shape],
      dtype: tensor.dtype,
      bytes: tensor.bytes,
      producer_id: tensor.producerId,
      consumer_ids: [...tensor.consumerIds],
      lifetime: tensor.lifetime,
      persistent: tensor.persistent,
      metadata: tensor.metadata,
    })),
    parameter_count: graph.parameterCount,
    accounting: {
      forward_flops: graph.accounting.forwardFlops,
      backward_flops: graph.accounting.backwardFlops,
      update_flops: graph.accounting.updateFlops,
      total_flops: graph.accounting.totalFlops,
      bytes_read: graph.accounting.bytesRead,
      bytes_written: graph.accounting.bytesWritten,
      parameter_bytes: graph.accounting.parameterBytes,
      activation_bytes: graph.accounting.activationBytes,
      workspace_bytes: graph.accounting.workspaceBytes,
      peak_live_bytes: graph.accounting.peakLiveBytes,
    },
    assumptions: [...graph.assumptions],
    provenance: graph.provenance,
  };
}

export function serializeOperatorGraph(graph: OperatorGraph): Record<string, unknown> {
  return operatorGraphToJSON(graph);
}

/** Rich browser IR names used internally once the Python-core parity names
 * are exported from core.ts. */
export type GraphIR = OperatorGraph;
export type GraphIROperator = OperatorNode;
export type BuildGraphIROptions = BuildOperatorGraphOptions;
export const buildGraphIR = buildOperatorGraph;
export const fromGraphIRContract = fromOperatorGraphContract;
export const graphIRToJSON = operatorGraphToJSON;

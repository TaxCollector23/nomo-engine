/** Browser port of nomo_planner/auditor.py. */

export interface AuditRun {
  sourceFormat: "megatron" | "deepspeed-json" | "vllm" | "log";
  model: string | null;
  sequenceLength: number | null;
  microBatchSize: number | null;
  tensorParallel: number;
  pipelineParallel: number;
  dataParallel: number | null;
  precision: string;
  zeroStage: number | null;
  command: string | null;
  observedStepTimeS: number | null;
  observedTokensPerS: number | null;
  observedMemoryBytes: number | null;
  warnings: string[];
  unrecognizedOptions: string[];
}

/** Match only the published presets shipped with the shared graph. Unknown model names stay unknown. */
export function modelConfigKey(model: string | null): "llama3_8b" | "llama3_70b" | "mixtral_8x7b" | null {
  const value = (model ?? "").toLowerCase();
  if (value.includes("llama-3-8b") || value.includes("llama3-8b")) return "llama3_8b";
  if (value.includes("llama-3-70b") || value.includes("llama3-70b")) return "llama3_70b";
  if (value.includes("mixtral-8x7b") || value.includes("mixtral_8x7b")) return "mixtral_8x7b";
  return null;
}

function base(sourceFormat: AuditRun["sourceFormat"], command: string | null = null): AuditRun {
  return { sourceFormat, model: null, sequenceLength: null, microBatchSize: null, tensorParallel: 1, pipelineParallel: 1, dataParallel: null, precision: "unknown", zeroStage: null, command, observedStepTimeS: null, observedTokensPerS: null, observedMemoryBytes: null, warnings: [], unrecognizedOptions: [] };
}

function unknownOptions(values: string[], known: string[]): string[] {
  const allowed = new Set(known);
  return [...new Set(values.filter((value) => value.startsWith("--")).map((value) => value.split("=", 1)[0]).filter((value) => !allowed.has(value)))].sort();
}

function tokens(command: string): string[] {
  return command.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g)?.map((value) => value.replace(/^['"]|['"]$/g, "")) ?? [];
}

function option(values: string[], ...names: string[]): string | null {
  for (let index = 0; index < values.length; index += 1) {
    for (const name of names) {
      if (values[index] === name) return values[index + 1] ?? null;
      if (values[index]?.startsWith(`${name}=`)) return values[index]!.slice(name.length + 1);
    }
  }
  return null;
}

function positive(value: string | null, fallback: number | null = null): number | null {
  const result = value === null ? NaN : Number(value);
  return Number.isFinite(result) && result > 0 ? result : fallback;
}

export function toMegatronArgs(run: AuditRun): string {
  const args = ["torchrun", "pretrain.py", "--tensor-model-parallel-size", String(run.tensorParallel), "--pipeline-model-parallel-size", String(run.pipelineParallel)];
  if (run.model) args.push("--model", run.model);
  if (run.sequenceLength) args.push("--seq-length", String(run.sequenceLength));
  if (run.microBatchSize) args.push("--micro-batch-size", String(run.microBatchSize));
  if (run.dataParallel) args.push("--data-parallel-size", String(run.dataParallel));
  if (run.precision === "bf16") args.push("--bf16");
  else if (run.precision === "fp8") args.push("--fp8");
  else if (run.precision === "fp16") args.push("--fp16");
  return args.join(" ");
}

export function toVllmCommand(run: AuditRun): string {
  const args = ["vllm", "serve", run.model ?? "<model>"];
  if (run.tensorParallel > 1) args.push("--tensor-parallel-size", String(run.tensorParallel));
  if (run.sequenceLength) args.push("--max-model-len", String(run.sequenceLength));
  if (run.microBatchSize) args.push("--max-num-seqs", String(run.microBatchSize));
  if (run.precision === "bf16") args.push("--dtype", "bfloat16");
  else if (run.precision === "fp16") args.push("--dtype", "float16");
  else if (run.precision !== "unknown") args.push("--dtype", run.precision);
  return args.join(" ");
}

export function toDeepSpeedConfig(run: AuditRun): Record<string, unknown> {
  const config: Record<string, unknown> = {
    train_micro_batch_size_per_gpu: run.microBatchSize ?? "auto",
    tensor_parallel: { tp_size: run.tensorParallel },
    pipeline_parallel: { stages: run.pipelineParallel },
  };
  if (run.zeroStage !== null) config.zero_optimization = { stage: run.zeroStage };
  if (run.precision === "bf16") config.bf16 = { enabled: true };
  else if (run.precision === "fp16") config.fp16 = { enabled: true };
  if (run.model) config.model_name_or_path = run.model;
  return config;
}

export function exportSameFormat(run: AuditRun): string {
  if (run.sourceFormat === "vllm") return toVllmCommand(run);
  if (run.sourceFormat === "deepspeed-json") return JSON.stringify(toDeepSpeedConfig(run), null, 2);
  return toMegatronArgs(run);
}

export function parseMegatronCommand(command: string): AuditRun {
  const values = tokens(command);
  const result = base("megatron", command);
  result.model = option(values, "--model", "--model-name", "--model-type");
  result.sequenceLength = positive(option(values, "--seq-length", "--max-position-embeddings"));
  result.microBatchSize = positive(option(values, "--micro-batch-size", "--micro-batch-size-per-gpu"));
  result.tensorParallel = positive(option(values, "--tensor-model-parallel-size", "--tensor-parallel-size"), 1)!;
  result.pipelineParallel = positive(option(values, "--pipeline-model-parallel-size", "--pipeline-parallel-size"), 1)!;
  result.dataParallel = positive(option(values, "--data-parallel-size"));
  result.precision = values.includes("--fp8") ? "fp8" : values.includes("--bf16") ? "bf16" : values.includes("--fp16") ? "fp16" : "unknown";
  if (!result.model) result.warnings.push("model name was not present in the command");
  if (result.precision === "unknown") result.warnings.push("precision flag was not present; quality impact is unknown");
  result.unrecognizedOptions = unknownOptions(values, ["--model", "--model-name", "--model-type", "--seq-length", "--max-position-embeddings", "--micro-batch-size", "--micro-batch-size-per-gpu", "--tensor-model-parallel-size", "--tensor-parallel-size", "--pipeline-model-parallel-size", "--pipeline-parallel-size", "--data-parallel-size", "--fp8", "--bf16", "--fp16"]);
  if (result.unrecognizedOptions.length) result.warnings.push(`options were not interpreted: ${result.unrecognizedOptions.join(", ")}`);
  return result;
}

export function parseVllmCommand(command: string): AuditRun {
  const values = tokens(command);
  const result = base("vllm", command);
  result.model = values.slice(1).find((value) => !value.startsWith("-") && value.includes("/")) ?? null;
  result.sequenceLength = positive(option(values, "--max-model-len", "--max-seq-len"));
  result.microBatchSize = positive(option(values, "--max-num-seqs"));
  result.tensorParallel = positive(option(values, "--tensor-parallel-size"), 1)!;
  const dtype = (option(values, "--dtype") ?? "unknown").toLowerCase();
  result.precision = dtype === "bfloat16" ? "bf16" : dtype === "half" || dtype === "float16" ? "fp16" : dtype;
  if (!result.model) result.warnings.push("model path was not detected; pass a Hugging Face model path");
  result.unrecognizedOptions = unknownOptions(values, ["--max-model-len", "--max-seq-len", "--max-num-seqs", "--tensor-parallel-size", "--dtype"]);
  if (result.unrecognizedOptions.length) result.warnings.push(`options were not interpreted: ${result.unrecognizedOptions.join(", ")}`);
  return result;
}

export function parseDeepSpeedConfig(config: Record<string, unknown>, command: string | null = null): AuditRun {
  const result = base("deepspeed-json", command);
  const zero = (config.zero_optimization && typeof config.zero_optimization === "object" ? config.zero_optimization : {}) as Record<string, unknown>;
  const tensor = (config.tensor_parallel && typeof config.tensor_parallel === "object" ? config.tensor_parallel : {}) as Record<string, unknown>;
  const pipeline = (config.pipeline_parallel && typeof config.pipeline_parallel === "object" ? config.pipeline_parallel : config.pipeline_parallel) as Record<string, unknown> | number | undefined;
  const bf16 = (config.bf16 && typeof config.bf16 === "object" ? config.bf16 : {}) as Record<string, unknown>;
  const fp16 = (config.fp16 && typeof config.fp16 === "object" ? config.fp16 : {}) as Record<string, unknown>;
  result.model = typeof config.model_name_or_path === "string" ? config.model_name_or_path : null;
  result.microBatchSize = positive(String(config.train_micro_batch_size_per_gpu ?? ""));
  result.tensorParallel = positive(String(tensor.tp_size ?? tensor.tp ?? 1), 1)!;
  result.pipelineParallel = positive(String(pipeline && typeof pipeline === "object" ? pipeline.stages ?? 1 : pipeline ?? 1), 1)!;
  result.precision = bf16.enabled ? "bf16" : fp16.enabled ? "fp16" : "unknown";
  result.zeroStage = positive(String(zero.stage ?? ""));
  if (result.precision === "unknown") result.warnings.push("DeepSpeed JSON did not enable bf16 or fp16");
  result.unrecognizedOptions = Object.keys(config).filter((key) => !["model_name_or_path", "train_micro_batch_size_per_gpu", "tensor_parallel", "pipeline_parallel", "zero_optimization", "bf16", "fp16"].includes(key)).sort();
  if (result.unrecognizedOptions.length) result.warnings.push(`DeepSpeed fields were not interpreted: ${result.unrecognizedOptions.join(", ")}`);
  return result;
}

export function parseLogMetrics(log: string, run: AuditRun = base("log")): AuditRun {
  const step = log.match(/(?:step time|iteration time|elapsed)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)/i);
  const tok = log.match(/(?:tokens?\s*\/\s*s|tokens per second)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)/i);
  const memory = log.match(/(?:memory|allocated)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB)/i);
  const factors: Record<string, number> = { gb: 1e9, gib: 2 ** 30, mb: 1e6, mib: 2 ** 20 };
  return { ...run, observedStepTimeS: step ? Number(step[1]) / (step[2]!.toLowerCase() === "ms" ? 1000 : 1) : null, observedTokensPerS: tok ? Number(tok[1]) : null, observedMemoryBytes: memory ? Number(memory[1]) * factors[memory[2]!.toLowerCase()]! : null };
}

export function auditText(value: string, log = ""): AuditRun {
  let result: AuditRun;
  try {
    const parsed: unknown = JSON.parse(value);
    result = parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parseDeepSpeedConfig(parsed as Record<string, unknown>) : parseMegatronCommand(value);
  } catch {
    result = value.toLowerCase().includes("vllm") ? parseVllmCommand(value) : parseMegatronCommand(value);
  }
  return parseLogMetrics(log, result);
}

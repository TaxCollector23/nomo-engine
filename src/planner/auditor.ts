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
}

function base(sourceFormat: AuditRun["sourceFormat"], command: string | null = null): AuditRun {
  return { sourceFormat, model: null, sequenceLength: null, microBatchSize: null, tensorParallel: 1, pipelineParallel: 1, dataParallel: null, precision: "unknown", zeroStage: null, command, observedStepTimeS: null, observedTokensPerS: null, observedMemoryBytes: null, warnings: [] };
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
  return result;
}

export function parseDeepSpeedConfig(config: Record<string, unknown>, command: string | null = null): AuditRun {
  const result = base("deepspeed-json", command);
  const zero = (config.zero_optimization && typeof config.zero_optimization === "object" ? config.zero_optimization : {}) as Record<string, unknown>;
  const tensor = (config.tensor_parallel && typeof config.tensor_parallel === "object" ? config.tensor_parallel : {}) as Record<string, unknown>;
  const pipeline = (config.pipeline_parallel && typeof config.pipeline_parallel === "object" ? config.pipeline_parallel : config.pipeline_parallel) as Record<string, unknown> | number | undefined;
  const bf16 = (config.bf16 && typeof config.bf16 === "object" ? config.bf16 : {}) as Record<string, unknown>;
  const fp16 = (config.fp16 && typeof config.fp16 === "object" ? config.fp16 : {}) as Record<string, unknown>;
  result.microBatchSize = positive(String(config.train_micro_batch_size_per_gpu ?? ""));
  result.tensorParallel = positive(String(tensor.tp_size ?? tensor.tp ?? 1), 1)!;
  result.pipelineParallel = positive(String(typeof pipeline === "object" ? pipeline.stages ?? 1 : pipeline ?? 1), 1)!;
  result.precision = bf16.enabled ? "bf16" : fp16.enabled ? "fp16" : "unknown";
  result.zeroStage = positive(String(zero.stage ?? ""));
  if (result.precision === "unknown") result.warnings.push("DeepSpeed JSON did not enable bf16 or fp16");
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

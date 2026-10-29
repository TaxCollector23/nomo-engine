/** Browser port of nomo_planner/auditor.py. */
import type { CalibrationFit } from "./customerCalibration";
import { CLUSTERS } from "./hardware";
import type { ExperimentCandidate } from "./experiments";
import type { Choice, Metrics, Pack, Plan } from "./packs";
import { feasible } from "./packs";
import { build, DEFAULT_SETTINGS, withLocks, type Settings } from "./registry";
import { Planner } from "./search";

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

// ================================================================== experiment candidates from real re-evaluation
// The auditor UI used to pass three hand-entered candidates into
// rankExperiments, so its "change risk" percentages came from invented
// numbers. Candidates are now derived by real re-evaluation through the same
// TrainingPack machinery the Lab uses: the parsed topology is mapped onto
// registry.build("llm_training", ...), the current plan is fitted to the
// pack's memory model, an exhaustive Planner search finds the recommended
// alternative, and each design variant (recompute / parallelism / ZeRO
// sharding) is evaluated with the same pack. uncertaintyS comes from the
// customer-calibration residual sigma when a fit exists, else a disclosed
// default. When the audited run cannot be mapped onto a pack, the previous
// assumption-based candidates are kept behind a visible tag.

export const DEFAULT_UNCERTAINTY_S = 0.15;
const ASSUMED_CLUSTER = "h100_nvlink_ib";
const TP_CHOICES = [1, 2, 4, 8];
const PP_CHOICES = [1, 2, 4, 8, 16];
const MICRO_BATCH_CHOICES = [1, 2, 4, 8];
const ZERO_CHOICES = [0, 1, 2, 3];
const PLAN_VARIABLES = ["devices", "tp", "pp", "zero", "recompute", "micro_batch", "matmul_dtype"];
// Only modelConfigKey presets with an entry in hardware.ts MODELS can build a
// TrainingPack; llama3_70b / mixtral_8x7b have no published pack preset here.
const REGISTRY_MODEL_BY_CONFIG: Record<string, string> = { llama3_8b: "llama3-8b" };

export interface DerivedExperimentCandidate extends ExperimentCandidate {
  /** null for assumption-based fallback candidates that never touched a pack. */
  feasible: boolean | null;
  evaluatedPlan: string | null;
  uncertaintyBasis: "customer-calibration" | "default" | "assumption";
}

export interface DerivedExperiments {
  basis: "re-evaluated" | "assumption-based";
  note: string;
  uncertaintyNote: string;
  candidates: DerivedExperimentCandidate[];
  currentStepS: number;
  alternateStepS: number;
  currentPlan: string | null;
  recommendedPlan: string | null;
  clusterKey: string | null;
}

function snapChoice(value: number, choices: number[], direction: "up" | "down"): number {
  if (choices.includes(value)) return value;
  const sorted = [...choices].sort((a, b) => a - b);
  if (direction === "up") return sorted.find((choice) => choice >= value) ?? sorted[sorted.length - 1]!;
  return [...sorted].reverse().find((choice) => choice <= value) ?? sorted[0]!;
}

function numericChoices(pack: Pack, name: string): number[] {
  const variable = pack.variables().find((v) => v.name === name);
  return variable ? variable.choices.filter((choice): choice is number => typeof choice === "number") : [];
}

function planVariables(plan: Plan): Plan {
  return Object.fromEntries(Object.entries(plan).filter(([name]) => PLAN_VARIABLES.includes(name))) as Plan;
}

function evaluatePlan(pack: Pack, plan: Plan): Metrics | null {
  for (const variable of pack.variables()) {
    const value = plan[variable.name];
    if (value === undefined || !variable.choices.includes(value as Choice)) return null;
  }
  const result = new Planner(withLocks(pack, planVariables(plan))).run();
  return result.all[0]?.metrics ?? null;
}

function stepSeconds(metrics: Metrics): number | null {
  const step = metrics.breakdown.step_s;
  return typeof step === "number" && Number.isFinite(step) && step > 0 ? step : null;
}

/** Evaluate a plan, raising the device count until the pack memory model is satisfied. */
function fitDevicesUp(pack: Pack, plan: Plan): { plan: Plan; metrics: Metrics } | null {
  let current = plan;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const metrics = evaluatePlan(pack, current);
    if (!metrics) return null;
    if (feasible(metrics)) return { plan: current, metrics };
    const next = numericChoices(pack, "devices").slice().sort((a, b) => a - b)
      .find((count) => count > Number(current.devices) && count % (Number(current.tp) * Number(current.pp)) === 0);
    if (!next) return null;
    current = { ...current, devices: next };
  }
  return null;
}

function samePlan(a: Plan, b: Plan): boolean {
  return JSON.stringify(planVariables(a)) === JSON.stringify(planVariables(b));
}

function assumptionBased(run: AuditRun, reason: string): DerivedExperiments {
  const seq = run.sequenceLength ?? 4096;
  return {
    basis: "assumption-based",
    note: `${reason} Falling back to the previous hand-entered candidates; treat every number here as an assumption, not a re-evaluation.`,
    uncertaintyNote: "assumption-based candidates keep the previous hand-entered uncertainties; no pack re-evaluation backed them",
    candidates: [
      { name: "tensor-parallel-2", config: { tensor_parallel: 2, sequence_length: seq }, expectedStepS: 1.1, uncertaintyS: 0.25, benchmarkHours: 0.5, feasible: null, evaluatedPlan: null, uncertaintyBasis: "assumption" },
      { name: "tensor-parallel-4", config: { tensor_parallel: 4, sequence_length: seq }, expectedStepS: 0.9, uncertaintyS: 0.45, benchmarkHours: 1, feasible: null, evaluatedPlan: null, uncertaintyBasis: "assumption" },
      { name: "bf16-baseline", config: { tensor_parallel: run.tensorParallel, precision: "bf16" }, expectedStepS: 1.25, uncertaintyS: 0.12, benchmarkHours: 0.25, feasible: null, evaluatedPlan: null, uncertaintyBasis: "assumption" },
    ],
    currentStepS: 1,
    alternateStepS: 1.2,
    currentPlan: null,
    recommendedPlan: null,
    clusterKey: null,
  };
}

export function deriveExperimentCandidates(run: AuditRun, calibration?: CalibrationFit | null): DerivedExperiments {
  const fallback = (reason: string) => assumptionBased(run, reason);
  const modelKey = modelConfigKey(run.model);
  const registryModel = modelKey ? REGISTRY_MODEL_BY_CONFIG[modelKey] : undefined;
  if (!registryModel) return fallback(`the parsed model "${run.model ?? "unknown"}" does not map to a published training-pack preset`);
  try {
    const settings: Settings = { ...DEFAULT_SETTINGS.llm_training, model: registryModel };
    if (run.sequenceLength) settings.seq_len = run.sequenceLength;
    const pack = build("llm_training", settings);
    const clusterKey = ASSUMED_CLUSTER;
    const clusterLabel = CLUSTERS[clusterKey]?.label ?? clusterKey;
    const seq = Number(settings.seq_len);
    const assumptions: string[] = [];
    const tp0 = snapChoice(Math.max(1, Math.round(run.tensorParallel) || 1), TP_CHOICES, "up");
    const pp0 = snapChoice(Math.max(1, Math.round(run.pipelineParallel) || 1), PP_CHOICES, "up");
    const dp0 = run.dataParallel && Number.isFinite(run.dataParallel) ? Math.max(1, Math.round(run.dataParallel)) : 8;
    if (!run.dataParallel) assumptions.push(`data parallel not recorded; ${dp0} assumed`);
    if (tp0 !== run.tensorParallel) assumptions.push(`tensor parallel snapped ${run.tensorParallel} to the pack choice ${tp0}`);
    if (pp0 !== run.pipelineParallel) assumptions.push(`pipeline parallel snapped ${run.pipelineParallel} to the pack choice ${pp0}`);
    const zero0 = run.zeroStage !== null && ZERO_CHOICES.includes(Math.round(run.zeroStage)) ? Math.round(run.zeroStage) : 1;
    if (zero0 === 1 && run.zeroStage === null) assumptions.push("ZeRO stage not recorded; 1 assumed");
    const mb0 = run.microBatchSize ? snapChoice(Math.round(run.microBatchSize), MICRO_BATCH_CHOICES, "down") : 1;
    if (!run.microBatchSize) assumptions.push("micro-batch not recorded; 1 assumed");
    const dtypeChoices = pack.variables().find((v) => v.name === "matmul_dtype")?.choices.map(String) ?? [];
    const dtype0 = run.precision === "fp8" && dtypeChoices.includes("fp8") ? "fp8" : "bf16";
    if (run.precision === "unknown") assumptions.push("precision not recorded; bf16 assumed");
    assumptions.push('recompute is never recorded in commands; "none" assumed (the training pack has no CPU-offload lever, so the ZeRO-stage toggle below is its memory-sharding lever)');
    assumptions.push(`cluster not parsed from logs; ${clusterLabel} assumed (the Lab's default) with registry-default global batch`);
    const wanted = Math.max(1, tp0 * pp0 * dp0);
    const devices0 = numericChoices(pack, "devices").slice().sort((a, b) => a - b)
      .find((count) => count >= wanted && count % (tp0 * pp0) === 0);
    if (!devices0) return fallback("the audited topology has no device count that fits the pack choice set");
    const basePlan: Plan = { devices: devices0, tp: tp0, pp: pp0, zero: zero0, recompute: "none", micro_batch: mb0, matmul_dtype: dtype0 };
    const fitted = fitDevicesUp(pack, basePlan);
    if (!fitted) return fallback("the audited topology has no feasible plan on the assumed cluster even at the largest device count");
    assumptions.push(`devices fitted to the smallest count (${fitted.plan.devices}) that satisfies the pack memory model`);
    const currentStepS = stepSeconds(fitted.metrics);
    if (currentStepS === null) return fallback("the pack could not produce a finite step time for the audited topology");
    const search = new Planner(pack).run();
    if (!search.recommended) return fallback("the pack search found no feasible alternative under the assumed hardware");
    const alternateStepS = stepSeconds(search.recommended.metrics) ?? currentStepS;

    const tpBump = tp0 < 8 ? { tp: tp0 * 2, name: `tensor-parallel-${tp0 * 2}` } : { tp: 4, name: "tensor-parallel-halve" };
    const zeroToggle = zero0 === 3 ? 1 : 3;
    const mbUp = mb0 < 8 ? mb0 * 2 : null;
    const ppUp = pp0 < 16 ? pp0 * 2 : null;
    const variants: Array<{ name: string; hours: number; change: Record<string, Choice> }> = [
      { name: "recompute-selective", hours: 0.5, change: { recompute: "selective" } },
      { name: "recompute-full", hours: 0.5, change: { recompute: "full" } },
      { name: tpBump.name, hours: 1, change: { tp: tpBump.tp } },
      { name: `zero-stage-${zeroToggle}`, hours: 0.5, change: { zero: zeroToggle } },
      ...(mbUp ? [{ name: `micro-batch-${mbUp}`, hours: 0.5, change: { micro_batch: mbUp } }] : []),
      ...(ppUp ? [{ name: `pipeline-stages-${ppUp}`, hours: 1, change: { pp: ppUp } }] : []),
    ];
    const sigmaFrac = calibration ? calibration.residualSigma : null;
    const candidates: DerivedExperimentCandidate[] = [];
    for (const variant of variants) {
      if (candidates.length >= 4) break;
      const variantPlan = { ...fitted.plan, ...variant.change };
      if (samePlan(variantPlan, fitted.plan)) continue;
      const fit = fitDevicesUp(pack, variantPlan);
      if (!fit) continue;
      const step = stepSeconds(fit.metrics);
      if (step === null) continue;
      candidates.push({
        name: variant.name,
        config: { ...planVariables(fit.plan), seq_len: seq },
        expectedStepS: step,
        uncertaintyS: sigmaFrac !== null ? sigmaFrac * step : DEFAULT_UNCERTAINTY_S,
        benchmarkHours: variant.hours,
        feasible: feasible(fit.metrics),
        evaluatedPlan: pack.describe(fit.plan),
        uncertaintyBasis: sigmaFrac !== null ? "customer-calibration" : "default",
      });
    }
    if (!candidates.length) return fallback("no design variant stayed feasible under the pack constraints");
    return {
      basis: "re-evaluated",
      note: `Parsed topology (TP${tp0}×PP${pp0}, ${dtype0}, ${registryModel}, seq ${seq}) re-evaluated on the ${clusterLabel} training pack. ${assumptions.join("; ")}.`,
      uncertaintyNote: calibration
        ? `uncertainty = customer-calibration residual sigma (${(sigmaFrac! * 100).toFixed(1)}% log-space over ${calibration.observations} rows, cluster label "${calibration.cluster}") × predicted step time`
        : `no customer calibration supplied; using the disclosed default ±${DEFAULT_UNCERTAINTY_S} s per candidate`,
      candidates,
      currentStepS,
      alternateStepS,
      currentPlan: pack.describe(fitted.plan),
      recommendedPlan: pack.describe(search.recommended.plan),
      clusterKey,
    };
  } catch (error) {
    return fallback(`pack re-evaluation failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}

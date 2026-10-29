// Preregistration scaffold + in-browser run execution.
// The research lab keeps user measurements and planner-computed rows strictly
// apart: computed rows carry origin "computed" and never overwrite user data.
import { build, DEFAULT_SETTINGS, innerPack, type DomainId, type Settings } from "./registry";
import { CLUSTERS, MODELS } from "./hardware";
import type { Metrics, Pack } from "./packs";
import { Planner, type Evaluated, type Result } from "./search";

export const RESEARCH_SCHEMA_VERSION = "nomo-research-study/1" as const;

export type StudyStatus = "draft" | "preregistered" | "running" | "complete" | "archived";
export type MetricDirection = "minimize" | "maximize";
export type RunCondition = "baseline" | "candidate";
export type RunStatus = "planned" | "measured" | "excluded";

export interface ResearchProtocol {
  title: string;
  question: string;
  hypothesis: string;
  primaryMetric: string;
  direction: MetricDirection;
  successThreshold: number | null;
  baseline: string;
  candidate: string;
  model: string;
  workload: string;
  hardware: string;
  method: string;
  seed: number;
  repetitions: number;
  author: string;
  notes: string;
  limitations: string;
}

export interface ResearchRun {
  id: string;
  condition: RunCondition;
  repetition: number;
  seed: number;
  status: RunStatus;
  metricValue: number | null;
  energyJ: number | null;
  latencyMs: number | null;
  source: string;
  notes: string;
  /** "computed" marks rows filled by executeRunMatrix (planner output, not hardware measurements). Absent = user-entered. */
  origin?: "computed";
}

export interface ResearchStudy {
  schemaVersion: typeof RESEARCH_SCHEMA_VERSION;
  id: string;
  createdAt: string;
  updatedAt: string;
  status: StudyStatus;
  protocol: ResearchProtocol;
  checklist: Record<string, boolean>;
  runs: ResearchRun[];
}

export interface ChecklistDefinition {
  id: string;
  label: string;
  description: string;
}

export interface ProtocolReadiness {
  checked: number;
  total: number;
  percent: number;
  blocking: string[];
  ready: boolean;
}

export interface RunAggregate {
  count: number;
  sourceCount: number;
  mean: number | null;
  median: number | null;
  values: number[];
}

export interface RunSummary {
  plannedCount: number;
  measuredCount: number;
  excludedCount: number;
  computedCount: number;
  pairedCount: number;
  baseline: RunAggregate;
  candidate: RunAggregate;
  deltaPct: number | null;
  improvementPct: number | null;
  thresholdMet: boolean | null;
  conclusion: string;
  evidence: "preview" | "measured" | "computed";
}

export const CHECKLIST: ChecklistDefinition[] = [
  { id: "question", label: "Question is specific", description: "A reviewer can tell what decision the study is meant to answer." },
  { id: "hypothesis", label: "Hypothesis is written", description: "State the expected direction before collecting results." },
  { id: "conditions", label: "Baseline and candidate are fixed", description: "Name both sides of the comparison; do not compare moving targets." },
  { id: "workload", label: "Workload and model are fixed", description: "Record the model, inputs, dataset or trace, and workload shape." },
  { id: "hardware", label: "Hardware environment is recorded", description: "Record the device, topology, driver, runtime, and relevant software versions." },
  { id: "method", label: "Measurement method is fixed", description: "Describe warmup, sampling, timers, and any exclusion rule." },
  { id: "randomness", label: "Seed and repetitions are set", description: "Use paired seeds where possible and record the planned repetition count." },
  { id: "decision", label: "Decision rule is acknowledged", description: "Set a threshold when possible, and explain why a threshold is unavailable otherwise." },
  { id: "limits", label: "Limitations are recorded", description: "Say what this study cannot establish yet." },
];

const STATUS_VALUES: StudyStatus[] = ["draft", "preregistered", "running", "complete", "archived"];
const RUN_STATUS_VALUES: RunStatus[] = ["planned", "measured", "excluded"];

const DEFAULT_PROTOCOL: ResearchProtocol = {
  title: "Untitled study",
  question: "",
  hypothesis: "",
  primaryMetric: "",
  direction: "minimize",
  successThreshold: null,
  baseline: "",
  candidate: "",
  model: "",
  workload: "",
  hardware: "",
  method: "",
  seed: 20261014,
  repetitions: 3,
  author: "",
  notes: "",
  limitations: "",
};

function makeId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function finiteOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function integerOr(value: unknown, fallback: number, min: number, max: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.round(n)));
}

function textOr(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function recordOr(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
}

export function createResearchStudy(now = new Date().toISOString()): ResearchStudy {
  return {
    schemaVersion: RESEARCH_SCHEMA_VERSION,
    id: makeId("study"),
    createdAt: now,
    updatedAt: now,
    status: "draft",
    protocol: { ...DEFAULT_PROTOCOL },
    checklist: Object.fromEntries(CHECKLIST.map((item) => [item.id, false])),
    runs: [],
  };
}

export function buildRunMatrix(study: ResearchStudy): ResearchRun[] {
  const existing = new Map(study.runs.map((run) => [`${run.condition}-${run.repetition}`, run]));
  const repetitions = integerOr(study.protocol.repetitions, 1, 1, 100);
  const rows: ResearchRun[] = [];
  for (let repetition = 1; repetition <= repetitions; repetition += 1) {
    for (const condition of ["baseline", "candidate"] as const) {
      const key = `${condition}-${repetition}`;
      const old = existing.get(key);
      rows.push(old ? { ...old, repetition, seed: study.protocol.seed + repetition - 1 } : {
        id: makeId(`run-${condition}-${repetition}`),
        condition,
        repetition,
        seed: study.protocol.seed + repetition - 1,
        status: "planned",
        metricValue: null,
        energyJ: null,
        latencyMs: null,
        source: "",
        notes: "",
      });
    }
  }
  return rows;
}

export function normalizeResearchStudy(value: unknown): ResearchStudy | null {
  const raw = recordOr(value);
  const protocolRaw = recordOr(raw.protocol);
  const checklistRaw = recordOr(raw.checklist);
  const status = STATUS_VALUES.includes(raw.status as StudyStatus) ? raw.status as StudyStatus : "draft";
  const direction = protocolRaw.direction === "maximize" ? "maximize" : "minimize";
  const now = new Date().toISOString();
  const createdAt = textOr(raw.createdAt, now);
  const study: ResearchStudy = {
    schemaVersion: RESEARCH_SCHEMA_VERSION,
    id: textOr(raw.id, makeId("study")),
    createdAt,
    updatedAt: textOr(raw.updatedAt, createdAt),
    status,
    protocol: {
      ...DEFAULT_PROTOCOL,
      title: textOr(protocolRaw.title, textOr(raw.title, DEFAULT_PROTOCOL.title)),
      question: textOr(protocolRaw.question),
      hypothesis: textOr(protocolRaw.hypothesis),
      primaryMetric: textOr(protocolRaw.primaryMetric),
      direction,
      successThreshold: finiteOrNull(protocolRaw.successThreshold),
      baseline: textOr(protocolRaw.baseline),
      candidate: textOr(protocolRaw.candidate),
      model: textOr(protocolRaw.model),
      workload: textOr(protocolRaw.workload),
      hardware: textOr(protocolRaw.hardware),
      method: textOr(protocolRaw.method),
      seed: integerOr(protocolRaw.seed, DEFAULT_PROTOCOL.seed, 0, 2147483647),
      repetitions: integerOr(protocolRaw.repetitions, DEFAULT_PROTOCOL.repetitions, 1, 100),
      author: textOr(protocolRaw.author),
      notes: textOr(protocolRaw.notes),
      limitations: textOr(protocolRaw.limitations),
    },
    checklist: Object.fromEntries(CHECKLIST.map((item) => [item.id, checklistRaw[item.id] === true])),
    runs: [],
  };
  const rawRuns = Array.isArray(raw.runs) ? raw.runs : [];
  study.runs = rawRuns.slice(0, 200).flatMap((item, index) => {
    const run = recordOr(item);
    const condition = run.condition === "candidate" ? "candidate" : run.condition === "baseline" ? "baseline" : null;
    if (!condition) return [];
    const statusValue = RUN_STATUS_VALUES.includes(run.status as RunStatus) ? run.status as RunStatus : "planned";
    return [{
      id: textOr(run.id, `imported-run-${index + 1}`),
      condition,
      repetition: integerOr(run.repetition, index + 1, 1, 100),
      seed: integerOr(run.seed, study.protocol.seed, 0, 2147483647),
      status: statusValue,
      metricValue: finiteOrNull(run.metricValue),
      energyJ: finiteOrNull(run.energyJ),
      latencyMs: finiteOrNull(run.latencyMs),
      source: textOr(run.source),
      notes: textOr(run.notes),
      origin: run.origin === "computed" ? "computed" as const : undefined,
    }];
  });
  return study;
}

export function protocolReadiness(study: ResearchStudy): ProtocolReadiness {
  const blocking = validateResearchStudy(study);
  const checked = CHECKLIST.filter((item) => study.checklist[item.id] === true).length;
  for (const item of CHECKLIST) {
    if (!study.checklist[item.id]) blocking.push(`Checklist: ${item.label}`);
  }
  const total = CHECKLIST.length;
  return { checked, total, percent: Math.round((checked / total) * 100), blocking, ready: blocking.length === 0 };
}

export function validateResearchStudy(study: ResearchStudy): string[] {
  const p = study.protocol;
  const missing: string[] = [];
  if (!p.title.trim() || p.title === "Untitled study") missing.push("Add a study title.");
  if (!p.question.trim()) missing.push("Write the research question.");
  if (!p.hypothesis.trim()) missing.push("Write the hypothesis before collecting results.");
  if (!p.primaryMetric.trim()) missing.push("Name the primary metric and its unit.");
  if (!p.baseline.trim()) missing.push("Name the baseline.");
  if (!p.candidate.trim()) missing.push("Name the candidate.");
  if (!p.model.trim()) missing.push("Record the model or graph.");
  if (!p.workload.trim()) missing.push("Record the workload, trace, or dataset.");
  if (!p.hardware.trim()) missing.push("Record the hardware and runtime environment.");
  if (!p.method.trim()) missing.push("Describe how measurements will be taken.");
  if (!Number.isInteger(p.seed) || p.seed < 0) missing.push("Use a non-negative integer seed.");
  if (!Number.isInteger(p.repetitions) || p.repetitions < 1) missing.push("Plan at least one repetition.");
  return missing;
}

function aggregate(runs: ResearchRun[]): RunAggregate {
  const values = runs.flatMap((run) => (run.status === "measured" || run.origin === "computed") && run.metricValue !== null ? [run.metricValue] : []);
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length ? (sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2) : null;
  return {
    count: values.length,
    sourceCount: runs.filter((run) => (run.status === "measured" || run.origin === "computed") && run.metricValue !== null && run.source.trim()).length,
    mean,
    median,
    values,
  };
}

export function summarizeResearchRuns(study: ResearchStudy): RunSummary {
  const baselineRuns = study.runs.filter((run) => run.condition === "baseline");
  const candidateRuns = study.runs.filter((run) => run.condition === "candidate");
  const baseline = aggregate(baselineRuns);
  const candidate = aggregate(candidateRuns);
  const measuredCount = study.runs.filter((run) => run.status === "measured").length;
  const plannedCount = study.runs.filter((run) => run.status === "planned").length;
  const excludedCount = study.runs.filter((run) => run.status === "excluded").length;
  const computedCount = study.runs.filter((run) => run.origin === "computed").length;
  const pairedCount = Math.min(baseline.count, candidate.count);
  const deltaPct = baseline.mean !== null && candidate.mean !== null && baseline.mean !== 0
    ? ((candidate.mean - baseline.mean) / Math.abs(baseline.mean)) * 100 : null;
  const improvementPct = deltaPct === null ? null : study.protocol.direction === "minimize" ? -deltaPct : deltaPct;
  const thresholdMet = improvementPct === null || study.protocol.successThreshold === null
    ? null : improvementPct >= study.protocol.successThreshold;
  const userMeasured = (runs: ResearchRun[]) => runs.filter((run) => run.status === "measured" && run.origin !== "computed" && run.metricValue !== null).length;
  const evidence: RunSummary["evidence"] = userMeasured(baselineRuns) > 0 && userMeasured(candidateRuns) > 0
    ? "measured"
    : baseline.count > 0 && candidate.count > 0 ? "computed" : "preview";
  let conclusion = "Waiting for at least one measured baseline and candidate result.";
  if (improvementPct !== null) {
    const direction = improvementPct >= 0 ? "better" : "worse";
    const thresholdNote = thresholdMet === null
      ? " Set a success threshold to preregister a decision rule."
      : thresholdMet ? " The candidate meets the preregistered threshold." : " The candidate does not meet the preregistered threshold.";
    conclusion = `The candidate is ${Math.abs(improvementPct).toFixed(1)}% ${direction} on ${study.protocol.primaryMetric || "the primary metric"}.${thresholdNote}`;
    if (evidence === "computed") conclusion += " Rows are planner-computed under the study assumptions — not hardware measurements.";
  }
  return {
    plannedCount,
    measuredCount,
    excludedCount,
    computedCount,
    pairedCount,
    baseline,
    candidate,
    deltaPct,
    improvementPct,
    thresholdMet,
    conclusion,
    evidence,
  };
}

export function studyManifest(study: ResearchStudy): Record<string, unknown> {
  const summary = summarizeResearchRuns(study);
  return {
    schemaVersion: RESEARCH_SCHEMA_VERSION,
    exportedAt: new Date().toISOString(),
    study,
    derived: {
      protocolReadiness: protocolReadiness(study),
      summary,
      note: "Derived values are recomputable from the study protocol and run rows; they are not independent evidence.",
    },
  };
}

function markdownCell(value: string): string {
  return value.replaceAll("|", "\\|").replaceAll("\n", " ").trim();
}

export function studyMarkdown(study: ResearchStudy): string {
  const p = study.protocol;
  const readiness = protocolReadiness(study);
  const summary = summarizeResearchRuns(study);
  const checklist = CHECKLIST.map((item) => `- [${study.checklist[item.id] ? "x" : " "}] ${item.label}`).join("\n");
  const rows = study.runs.map((run) => `| ${run.condition} | ${run.repetition} | ${run.seed} | ${run.status} | ${run.metricValue ?? ""} | ${run.energyJ ?? ""} | ${run.latencyMs ?? ""} | ${markdownCell(run.source)} |`).join("\n");
  return `# ${p.title}\n\n- Status: ${study.status}\n- Study ID: \`${study.id}\`\n- Created: ${study.createdAt}\n- Protocol readiness: ${readiness.checked}/${readiness.total} checks (${readiness.percent}%)\n\n> This is a local Nomo research record. It is not a signed preregistration, a peer-reviewed result, or proof of silicon performance.\n\n## Research question\n\n${p.question || "_Not recorded._"}\n\n## Hypothesis\n\n${p.hypothesis || "_Not recorded._"}\n\n## Protocol\n\n| Field | Value |\n| --- | --- |\n| Baseline | ${markdownCell(p.baseline)} |\n| Candidate | ${markdownCell(p.candidate)} |\n| Primary metric | ${markdownCell(p.primaryMetric)} |\n| Direction | ${p.direction} |\n| Success threshold | ${p.successThreshold ?? "Not set"} |\n| Model / graph | ${markdownCell(p.model)} |\n| Workload / trace | ${markdownCell(p.workload)} |\n| Hardware / runtime | ${markdownCell(p.hardware)} |\n| Measurement method | ${markdownCell(p.method)} |\n| Seed | ${p.seed} |\n| Repetitions | ${p.repetitions} |\n| Author | ${markdownCell(p.author)} |\n\n### Protocol checks\n\n${checklist}\n\n## Run matrix\n\n| Condition | Repetition | Seed | Status | Primary metric | Energy (J) | Latency (ms) | Source |\n| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |\n${rows || "| _No runs planned_ | | | | | | | |"}\n\n## Current result summary\n\n${summary.conclusion}\n\n- Baseline mean: ${summary.baseline.mean ?? "not measured"}\n- Candidate mean: ${summary.candidate.mean ?? "not measured"}\n- Improvement: ${summary.improvementPct === null ? "not available" : `${summary.improvementPct.toFixed(2)}%`}\n- Paired measured repetitions: ${summary.pairedCount}\n\n## Notes and limitations\n\n${p.notes || "_Not recorded._"}\n\n${p.limitations || "_Not recorded._"}\n`;
}

function csvCell(value: string | number | null): string {
  const text = value === null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function studyCsv(study: ResearchStudy): string {
  const header = ["study_id", "condition", "repetition", "seed", "status", "primary_metric", "energy_j", "latency_ms", "source", "notes"];
  const rows = study.runs.map((run) => [study.id, run.condition, run.repetition, run.seed, run.status, run.metricValue, run.energyJ, run.latencyMs, run.source, run.notes]
    .map((value) => csvCell(value)).join(","));
  return [header.join(","), ...rows].join("\n") + "\n";
}

// ================================================================== run-matrix execution
// The research lab can now actually run its planned rows through the same
// in-browser planner the Lab uses (registry.build + search.Planner). The
// planner is deterministic, so the study seed does not perturb the physics:
// it is mixed with the repetition index using the same LCG constants as
// src/planner/layers.ts (DeterministicRng) and used to select reproducibly
// among tied plans. Computed rows are labeled origin "computed", never
// overwrite user-entered values, and are never presented as measurements.

export const RUN_MATRIX_LCG_MULTIPLIER = 1664525;
export const RUN_MATRIX_LCG_INCREMENT = 1013904223;

export const PLANNER_RUN_NOTE = "Computed locally by the in-browser planner under study assumptions — not hardware measurements.";
export const COMPARISON_NOTE = "Computed baseline rows evaluate the planner's balanced recommendation; computed candidate rows evaluate the Pareto-front plan that minimizes the pack's final objective (ties broken by the LCG-derived run seed).";

/** Mix a study seed and repetition index with the layers.ts LCG constants. */
export function deriveRunSeed(baseSeed: number, repetition: number): number {
  const state = (Math.trunc(baseSeed) + Math.max(0, Math.trunc(repetition) - 1)) >>> 0;
  return ((Math.imul(RUN_MATRIX_LCG_MULTIPLIER, state) + RUN_MATRIX_LCG_INCREMENT) >>> 0);
}

export interface RunExecutorContext { study: ResearchStudy; condition: RunCondition; repetition: number; seed: number }
export interface RunExecutorPatch { metricValue?: number | null; energyJ?: number | null; latencyMs?: number | null; planNote?: string }
export type RunExecutor = (context: RunExecutorContext) => RunExecutorPatch | null;

/**
 * Rebuild the run matrix and fill planned, still-empty rows through `execute`.
 * Rows with user-entered metric/energy/latency data are never touched, and
 * computed rows keep status "planned" (they are not measurements) with
 * origin "computed" for the aggregators and UI.
 */
export function runMatrix(study: ResearchStudy, execute: RunExecutor): ResearchStudy {
  const rows = buildRunMatrix(study);
  const next = rows.map((row) => {
    if (row.status !== "planned" || row.origin === "computed") return row;
    if (row.metricValue !== null || row.energyJ !== null || row.latencyMs !== null) return row;
    const patch = execute({ study, condition: row.condition, repetition: row.repetition, seed: row.seed });
    if (!patch) return row;
    const metricValue = finiteOrNull(patch.metricValue);
    if (metricValue === null) return row;
    const planNote = typeof patch.planNote === "string" && patch.planNote.trim() ? ` Plan: ${patch.planNote.trim()}.` : "";
    return {
      ...row,
      metricValue,
      energyJ: finiteOrNull(patch.energyJ),
      latencyMs: finiteOrNull(patch.latencyMs),
      origin: "computed" as const,
      source: row.source.trim() ? row.source : "nomo-planner (in-browser)",
      notes: row.notes.trim() ? row.notes : `${PLANNER_RUN_NOTE}${planNote}`,
    };
  });
  return { ...study, runs: next };
}

interface PlannerComputation {
  ok: boolean;
  error?: string;
  domain?: DomainId;
  pack?: Pack;
  result?: Result;
  assumed?: string;
}

function normalizedKey(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9.]/g, "");
}

function matchKey(text: string, entries: Array<[string, string]>): string | null {
  const t = normalizedKey(text);
  if (!t) return null;
  for (const [key, label] of entries) {
    const keyNorm = normalizedKey(key);
    if (keyNorm && t.includes(keyNorm)) return key;
    const labelNorm = normalizedKey(label);
    if (labelNorm.length >= 4 && t.includes(labelNorm)) return key;
    const head = key.split(/[^a-z0-9.]+/)[0]!;
    if (head.length >= 3 && t.includes(normalizedKey(head))) return key;
  }
  return null;
}

function protocolDomain(p: ResearchProtocol): DomainId {
  const text = `${p.title} ${p.question} ${p.primaryMetric} ${p.workload} ${p.hardware}`.toLowerCase();
  if (/(serve|serving|inference|decode|prompt|tokens per second|token throughput)/.test(text)) return "llm_inference";
  if (/(design|architecture|scaling law|budget|build)/.test(text)) return "arch_codesign";
  return "llm_training";
}

/** Map the free-text protocol onto a registry pack; every assumption is disclosed. */
function protocolSettings(p: ResearchProtocol): { domain: DomainId; settings: Settings; assumed: string } {
  const domain = protocolDomain(p);
  if (domain === "arch_codesign") {
    return { domain, settings: { ...DEFAULT_SETTINGS.arch_codesign }, assumed: "arch_codesign pack from registry defaults (besiroglu2024 scaling law, train_mfu 0.4)" };
  }
  const settings: Settings = { ...DEFAULT_SETTINGS[domain] };
  const assumed: string[] = [`seq_len / batch / total_tokens left at registry defaults`];
  const modelKey = matchKey(p.model, Object.entries(MODELS).map(([key, m]) => [key, m.name] as [string, string]));
  if (modelKey) settings.model = modelKey;
  else assumed.push(`model text "${p.model || "empty"}" matched no published model preset; registry default ${String(DEFAULT_SETTINGS[domain].model)} assumed`);
  const clusterKey = matchKey(p.hardware, Object.entries(CLUSTERS).map(([key, c]) => [key, c.label] as [string, string]));
  if (clusterKey) settings.cluster = clusterKey;
  else assumed.push(`hardware text "${p.hardware || "empty"}" matched no published cluster; registry default ${String(DEFAULT_SETTINGS[domain].cluster)} assumed`);
  return { domain, settings, assumed: assumed.join("; ") };
}

function computePlannerRun(p: ResearchProtocol): PlannerComputation {
  try {
    const { domain, settings, assumed } = protocolSettings(p);
    const pack = innerPack(build(domain, settings));
    const planner = new Planner(pack);
    const result = planner.run();
    if (!result.front.length) return { ok: false, error: "the pack search found no feasible plan under the study assumptions; rows were left untouched", domain };
    return { ok: true, domain, pack, result, assumed };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

function pickPlan(result: Result, condition: RunCondition, seed: number): Evaluated | null {
  if (condition === "baseline") return result.recommended ?? result.front[0] ?? null;
  if (!result.front.length) return null;
  const last = result.front[0]!.F.length - 1;
  let minVal = Infinity;
  for (const entry of result.front) minVal = Math.min(minVal, entry.F[last]!);
  const tied = result.front.filter((entry) => entry.F[last] === minVal);
  if (!tied.length) return null;
  return tied[seed % tied.length] ?? null;
}

function primaryMetricValue(p: ResearchProtocol, pack: Pack, metrics: Metrics): { value: number | null; label: string } {
  const t = p.primaryMetric.toLowerCase();
  const bd = metrics.breakdown;
  const obj = metrics.objectives;
  const firstFiniteObjective = (): number | null => {
    for (const objective of pack.objectives()) {
      const value = obj[objective.name];
      if (typeof value === "number" && Number.isFinite(value)) return value;
    }
    return null;
  };
  if (/(latency|\bms\b|step time|per token)/.test(t)) {
    if (typeof obj.ms_per_token === "number" && Number.isFinite(obj.ms_per_token)) return { value: obj.ms_per_token, label: "ms_per_token" };
    if (typeof bd.step_s === "number") return { value: bd.step_s * 1000, label: "step time (ms)" };
  }
  if (/energy/.test(t) && typeof bd.energy_mwh === "number") return { value: bd.energy_mwh * 3.6e9, label: "energy (J)" };
  if (/cost|usd|\$/.test(t)) {
    for (const name of ["cost_usd", "usd_per_mtok", "total_cost_usd"]) {
      const value = obj[name];
      if (typeof value === "number" && Number.isFinite(value)) return { value, label: name };
    }
  }
  if (/(time|day|duration|wall)/.test(t) && typeof obj.days === "number" && Number.isFinite(obj.days)) return { value: obj.days, label: "days" };
  if (/loss/.test(t) && typeof obj.loss === "number" && Number.isFinite(obj.loss)) return { value: obj.loss, label: "loss" };
  if (/quality/.test(t) && typeof obj.quality_loss === "number" && Number.isFinite(obj.quality_loss)) return { value: obj.quality_loss, label: "quality_loss" };
  const value = firstFiniteObjective();
  const first = pack.objectives()[0]?.name;
  return { value, label: first ?? "first finite objective" };
}

/**
 * Execute the planned run matrix through the in-browser planner.
 * The planner runs once per study; baseline rows evaluate the balanced
 * recommendation and candidate rows evaluate the front plan that minimizes
 * the pack's final objective. Pack assumptions derived from the free-text
 * protocol are disclosed in each computed row's notes. Throws when the study
 * assumptions cannot be turned into a feasible pack search, leaving all rows
 * untouched.
 */
export function executeRunMatrix(study: ResearchStudy): ResearchStudy {
  let cached: PlannerComputation | null = null;
  const ensureComputed = (): PlannerComputation => {
    if (!cached) cached = computePlannerRun(study.protocol);
    return cached;
  };
  const executor: RunExecutor = (context) => {
    const state = ensureComputed();
    if (!state.ok || !state.pack || !state.result) throw new Error(state.error ?? "the planner could not run under the study assumptions");
    const seed = deriveRunSeed(study.protocol.seed, context.repetition);
    const picked = pickPlan(state.result, context.condition, seed);
    if (!picked) return null;
    const metrics = picked.metrics;
    const primary = primaryMetricValue(study.protocol, state.pack, metrics);
    if (primary.value === null) return null;
    const stepS = typeof metrics.breakdown.step_s === "number" ? metrics.breakdown.step_s : null;
    const energyJ = typeof metrics.breakdown.energy_mwh === "number" ? metrics.breakdown.energy_mwh * 3.6e9 : null;
    const msPerToken = metrics.objectives.ms_per_token;
    const latencyMs = typeof msPerToken === "number" && Number.isFinite(msPerToken) ? msPerToken : stepS !== null ? stepS * 1000 : null;
    const planNote = `${state.pack.describe(picked.plan)} Metric: ${primary.label}.${state.assumed ? ` Assumptions: ${state.assumed}.` : ""} Computed seed ${seed}.`;
    return { metricValue: primary.value, energyJ, latencyMs, planNote };
  };
  return runMatrix(study, executor);
}

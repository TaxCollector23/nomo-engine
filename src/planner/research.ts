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
  pairedCount: number;
  baseline: RunAggregate;
  candidate: RunAggregate;
  deltaPct: number | null;
  improvementPct: number | null;
  thresholdMet: boolean | null;
  conclusion: string;
  evidence: "preview" | "measured";
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
  const values = runs.flatMap((run) => run.status === "measured" && run.metricValue !== null ? [run.metricValue] : []);
  const sorted = [...values].sort((a, b) => a - b);
  const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length ? (sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2) : null;
  return {
    count: values.length,
    sourceCount: runs.filter((run) => run.status === "measured" && run.metricValue !== null && run.source.trim()).length,
    mean,
    median,
    values,
  };
}

export function summarizeResearchRuns(study: ResearchStudy): RunSummary {
  const baseline = aggregate(study.runs.filter((run) => run.condition === "baseline"));
  const candidate = aggregate(study.runs.filter((run) => run.condition === "candidate"));
  const measuredCount = study.runs.filter((run) => run.status === "measured").length;
  const plannedCount = study.runs.filter((run) => run.status === "planned").length;
  const excludedCount = study.runs.filter((run) => run.status === "excluded").length;
  const pairedCount = Math.min(baseline.count, candidate.count);
  const deltaPct = baseline.mean !== null && candidate.mean !== null && baseline.mean !== 0
    ? ((candidate.mean - baseline.mean) / Math.abs(baseline.mean)) * 100 : null;
  const improvementPct = deltaPct === null ? null : study.protocol.direction === "minimize" ? -deltaPct : deltaPct;
  const thresholdMet = improvementPct === null || study.protocol.successThreshold === null
    ? null : improvementPct >= study.protocol.successThreshold;
  let conclusion = "Waiting for at least one measured baseline and candidate result.";
  if (improvementPct !== null) {
    const direction = improvementPct >= 0 ? "better" : "worse";
    const thresholdNote = thresholdMet === null
      ? " Set a success threshold to preregister a decision rule."
      : thresholdMet ? " The candidate meets the preregistered threshold." : " The candidate does not meet the preregistered threshold.";
    conclusion = `The candidate is ${Math.abs(improvementPct).toFixed(1)}% ${direction} on ${study.protocol.primaryMetric || "the primary metric"}.${thresholdNote}`;
  }
  return {
    plannedCount,
    measuredCount,
    excludedCount,
    pairedCount,
    baseline,
    candidate,
    deltaPct,
    improvementPct,
    thresholdMet,
    conclusion,
    evidence: baseline.count > 0 && candidate.count > 0 ? "measured" : "preview",
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

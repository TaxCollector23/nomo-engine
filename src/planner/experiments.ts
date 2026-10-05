import type { CustomerObservation } from "./customerCalibration";

export interface ExperimentCandidate {
  name: string;
  config: Record<string, unknown>;
  expectedStepS: number;
  uncertaintyS: number;
  benchmarkHours: number;
}

export interface RankedExperiment extends ExperimentCandidate {
  probabilityChangesRecommendation: number;
  valuePerBenchmarkHour: number;
}

export function rankExperiments(candidates: ExperimentCandidate[], currentStepS: number, alternateStepS: number): RankedExperiment[] {
  if (!(currentStepS > 0) || !(alternateStepS > 0)) throw new Error("current and alternate predictions must be positive");
  const margin = Math.abs(currentStepS - alternateStepS);
  return candidates.map((candidate) => {
    if (!(candidate.expectedStepS > 0) || candidate.uncertaintyS < 0 || !(candidate.benchmarkHours > 0)) throw new Error("experiment predictions, uncertainty, and duration must be valid");
    const probabilityChangesRecommendation = Math.min(0.99, Math.max(0.01, candidate.uncertaintyS / (candidate.uncertaintyS + margin)));
    return { ...candidate, probabilityChangesRecommendation, valuePerBenchmarkHour: probabilityChangesRecommendation / candidate.benchmarkHours };
  }).sort((left, right) => right.valuePerBenchmarkHour - left.valuePerBenchmarkHour || left.name.localeCompare(right.name));
}

export function csvTemplate(ranked: RankedExperiment[]): string {
  const escape = (value: string): string => /[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
  return ["experiment,config_json,observed_step_s,observed_tokens_per_s,notes", ...ranked.map((row) => [
    row.name, JSON.stringify(row.config), "", "", "Fill after running this benchmark; values stay local",
  ].map(escape).join(","))].join("\n");
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], cell = "", quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]!;
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') { cell += '"'; index += 1; continue; }
    if (char === '"') { quoted = !quoted; continue; }
    if (char === "," && !quoted) { row.push(cell.trim()); cell = ""; continue; }
    if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell.trim()); cell = "";
      if (row.some((value) => value.length)) rows.push(row);
      row = [];
      continue;
    }
    cell += char;
  }
  row.push(cell.trim());
  if (row.some((value) => value.length)) rows.push(row);
  return rows;
}

export function parseExperimentResultsCsv(
  text: string,
  candidates: Array<ExperimentCandidate | RankedExperiment>,
  cluster = "experiment-upload",
): CustomerObservation[] {
  const rows = parseCsv(text);
  if (!rows.length) throw new Error("experiment results CSV is empty");
  const header = new Map(rows[0]!.map((name, index) => [name, index]));
  for (const required of ["experiment", "observed_step_s"]) {
    if (!header.has(required)) throw new Error(`experiment results CSV is missing columns: ${required}`);
  }
  const byName = new Map(candidates.map((candidate) => [candidate.name, candidate]));
  const observations: CustomerObservation[] = [];
  for (const [offset, row] of rows.slice(1).entries()) {
    const name = row[header.get("experiment")!] ?? "";
    const observedText = row[header.get("observed_step_s")!] ?? "";
    if (!name && !observedText) continue;
    const candidate = byName.get(name);
    if (!candidate) throw new Error(`row ${offset + 2} names an experiment that was not in the current designer: ${name}`);
    const observedStepS = Number(observedText);
    if (!(observedStepS > 0)) throw new Error(`row ${offset + 2} observed_step_s must be positive`);
    observations.push({ runId: `experiment:${name}`, cluster, predictedStepS: candidate.expectedStepS, observedStepS });
  }
  if (!observations.length) throw new Error("experiment results CSV contains no completed observations");
  return observations;
}

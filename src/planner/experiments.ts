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
  return ["experiment,config_json,observed_step_s,observed_tokens_per_s,notes", ...ranked.map((row) => `${row.name},${JSON.stringify(JSON.stringify(row.config))},,,Fill after running this benchmark; values stay local`)].join("\n");
}

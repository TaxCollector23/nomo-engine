export interface CustomerObservation {
  runId: string;
  cluster: string;
  predictedStepS: number;
  observedStepS: number;
}

export interface CalibrationFit {
  cluster: string;
  observations: number;
  scale: number;
  residualSigma: number;
  intervalLowScale: number;
  intervalHighScale: number;
  coverage: number;
  method: string;
}

export function parseCustomerCsv(text: string): CustomerObservation[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = lines[0]!.split(",").map((value) => value.trim());
  const required = ["run_id", "cluster", "predicted_step_s", "observed_step_s"];
  const missing = required.filter((value) => !headers.includes(value));
  if (missing.length) throw new Error(`customer calibration CSV is missing columns: ${missing.join(", ")}`);
  return lines.slice(1).map((line, row) => {
    const values = line.split(",").map((value) => value.trim());
    const value = (name: string) => values[headers.indexOf(name)] ?? "";
    const predicted = Number(value("predicted_step_s"));
    const observed = Number(value("observed_step_s"));
    if (!(predicted > 0) || !(observed > 0)) throw new Error(`customer calibration row ${row + 2} step times must be positive`);
    return { runId: value("run_id"), cluster: value("cluster"), predictedStepS: predicted, observedStepS: observed };
  });
}

function median(values: number[]): number {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = (ordered.length - 1) / 2;
  return ordered[Math.floor(middle)]! + (ordered[Math.ceil(middle)]! - ordered[Math.floor(middle)]!) * (middle - Math.floor(middle));
}

export function fitCustomerScale(rows: CustomerObservation[], cluster?: string): CalibrationFit {
  const selected = rows.filter((row) => cluster === undefined || row.cluster === cluster);
  if (!selected.length) throw new Error("customer calibration needs at least one matching observation");
  const ratios = selected.map((row) => row.observedStepS / row.predictedStepS);
  const scale = median(ratios);
  const residuals = ratios.map((ratio) => Math.log(ratio / scale));
  const residualSigma = Math.sqrt(residuals.reduce((sum, value) => sum + value * value, 0) / residuals.length);
  const intervalLowScale = scale * Math.exp(-1.645 * residualSigma);
  const intervalHighScale = scale * Math.exp(1.645 * residualSigma);
  const coverage = ratios.filter((ratio) => ratio >= intervalLowScale && ratio <= intervalHighScale).length / ratios.length;
  return { cluster: cluster ?? (new Set(selected.map((row) => row.cluster)).size === 1 ? selected[0]!.cluster : "mixed"), observations: selected.length, scale, residualSigma, intervalLowScale, intervalHighScale, coverage, method: "median multiplicative refit in log-step-time space" };
}

export function applyCustomerFit(predictedStepS: number, fit: CalibrationFit): { median: number; low: number; high: number } {
  if (!(predictedStepS > 0)) throw new Error("predicted step time must be positive");
  return { median: predictedStepS * fit.scale, low: predictedStepS * fit.intervalLowScale, high: predictedStepS * fit.intervalHighScale };
}

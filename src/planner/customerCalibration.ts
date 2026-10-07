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
  heldOutMapePct: number | null;
  heldOutObservations: number;
  method: string;
  validation: string;
  intervalLevel: number;
  intervalScope: string;
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!;
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { field += '"'; index++; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"' && field.length === 0) quoted = true;
    else if (char === ",") { row.push(field.trim()); field = ""; }
    else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[index + 1] === "\n") index++;
      row.push(field.trim()); field = "";
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
    } else field += char;
  }
  if (quoted) throw new Error("customer calibration CSV has an unterminated quoted field");
  if (field.length || row.length) { row.push(field.trim()); if (row.some((value) => value !== "")) rows.push(row); }
  return rows;
}

export function parseCustomerCsv(text: string): CustomerObservation[] {
  const rows = parseCsvRows(text);
  if (!rows.length) return [];
  const headers = rows[0]!.map((value) => value.trim());
  const required = ["run_id", "cluster", "predicted_step_s", "observed_step_s"];
  const missing = required.filter((value) => !headers.includes(value));
  if (missing.length) throw new Error(`customer calibration CSV is missing columns: ${missing.join(", ")}`);
  if (new Set(headers).size !== headers.length) throw new Error("customer calibration CSV has duplicate columns");
  return rows.slice(1).map((values, row) => {
    const value = (name: string) => values[headers.indexOf(name)] ?? "";
    const predicted = Number(value("predicted_step_s"));
    const observed = Number(value("observed_step_s"));
    if (!value("run_id") || !value("cluster")) throw new Error(`customer calibration row ${row + 2} needs non-empty run_id and cluster`);
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
  const heldOutErrors = selected.length >= 2 ? ratios.map((_, index) => {
    const train = ratios.filter((_ratio, trainIndex) => trainIndex !== index);
    const heldOutScale = median(train);
    return Math.abs(ratios[index]! / heldOutScale - 1) * 100;
  }) : [];
  return {
    cluster: cluster ?? (new Set(selected.map((row) => row.cluster)).size === 1 ? selected[0]!.cluster : "mixed"),
    observations: selected.length, scale, residualSigma, intervalLowScale, intervalHighScale, coverage,
    heldOutMapePct: heldOutErrors.length ? heldOutErrors.reduce((sum, value) => sum + value, 0) / heldOutErrors.length : null,
    heldOutObservations: heldOutErrors.length,
    method: "median multiplicative refit in log-step-time space",
    validation: "leave-one-out error on customer-supplied rows",
    intervalLevel: 0.90,
    intervalScope: "empirical in-sample central interval; not a predictive guarantee",
  };
}

export function applyCustomerFit(predictedStepS: number, fit: CalibrationFit): { median: number; low: number; high: number } {
  if (!(predictedStepS > 0)) throw new Error("predicted step time must be positive");
  return { median: predictedStepS * fit.scale, low: predictedStepS * fit.intervalLowScale, high: predictedStepS * fit.intervalHighScale };
}

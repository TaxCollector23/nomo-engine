/**
 * A reviewable early-stage chip design model.
 *
 * This is deliberately not a silicon simulator.  It turns a design brief,
 * explicit architecture inputs, and measured calibration rows into a small
 * decision record with visible constraints and provenance.  The model never
 * turns an unmeasured number into a validation claim.
 */

export type ChipEvidenceKind = "assumption" | "datasheet" | "measured" | "calibrated";
export type ChipCalibrationKind = "compute" | "memory" | "interconnect" | "end_to_end";

export interface ChipDesignSpec {
  name: string;
  processNm: number;
  dieAreaMm2: number;
  tileCount: number;
  bf16Tops: number;
  fp8Tops: number;
  memoryBandwidthGbps: number;
  sramGb: number;
  hbmGb: number;
  interconnectGbps: number;
  interconnectLatencyNs: number;
  neuromorphicCores: number;
  powerW: number;
  voltageV: number;
  frequencyGHz: number;
  yieldPct: number;
  unitCostUsd: number;
  hourlyCostUsd: number;
}

export interface ChipWorkloadSpec {
  name: string;
  denseFlops: number;
  memoryBytes: number;
  interconnectBytes: number;
  spikingOps: number;
  batchSize: number;
  precision: "bf16" | "fp8";
}

export interface ChipConstraints {
  maxPowerW: number;
  maxAreaMm2: number;
  maxLatencyMs: number;
  minThroughputPerSecond: number;
  maxCostUsdPerBatch: number;
  minYieldPct: number;
}

export interface ChipCalibrationObservation {
  id: string;
  name: string;
  kind: ChipCalibrationKind;
  baselineSeconds: number;
  observedSeconds: number;
  observedPowerW: number | null;
  source: string;
  observedAt: string;
}

export interface ChipCalibrationFit {
  status: "assumption" | "partial" | "calibrated";
  computeEfficiency: number;
  memoryEfficiency: number;
  interconnectEfficiency: number;
  powerScale: number;
  sampleCount: number;
  fittedKinds: ChipCalibrationKind[];
  residualP90Pct: number | null;
  note: string;
}

export interface ChipConstraintResult {
  label: string;
  value: number;
  limit: number;
  unit: string;
  direction: "max" | "min";
  pass: boolean;
  headroom: number;
}

export interface ChipDesignEvaluation {
  feasible: boolean;
  latencySeconds: number;
  throughputPerSecond: number;
  energyJPerBatch: number;
  costUsdPerBatch: number;
  powerW: number;
  bottleneck: "compute" | "memory" | "interconnect" | "neuromorphic";
  componentSeconds: Record<"compute" | "memory" | "interconnect" | "neuromorphic", number>;
  constraints: Record<string, ChipConstraintResult>;
  calibration: ChipCalibrationFit;
  evidence: ChipEvidenceKind;
  notes: string[];
}

export interface ChipDesignVariant {
  id: string;
  label: string;
  rationale: string;
  changes: string[];
  spec: ChipDesignSpec;
}

export const DEFAULT_CHIP_DESIGN: ChipDesignSpec = {
  name: "Reference edge accelerator",
  processNm: 5,
  dieAreaMm2: 420,
  tileCount: 16,
  bf16Tops: 1_000,
  fp8Tops: 2_000,
  memoryBandwidthGbps: 1_600,
  sramGb: 80,
  hbmGb: 64,
  interconnectGbps: 800,
  interconnectLatencyNs: 250,
  neuromorphicCores: 128,
  powerW: 300,
  voltageV: 0.8,
  frequencyGHz: 1.5,
  yieldPct: 90,
  unitCostUsd: 10_000,
  hourlyCostUsd: 2,
};

export const DEFAULT_CHIP_WORKLOAD: ChipWorkloadSpec = {
  name: "Transformer serving batch",
  denseFlops: 1e15,
  memoryBytes: 1e12,
  interconnectBytes: 2e11,
  spikingOps: 0,
  batchSize: 1,
  precision: "fp8",
};

export const DEFAULT_CHIP_CONSTRAINTS: ChipConstraints = {
  maxPowerW: 350,
  maxAreaMm2: 500,
  maxLatencyMs: 25,
  minThroughputPerSecond: 30,
  maxCostUsdPerBatch: 0.01,
  minYieldPct: 85,
};

const DEFAULT_COMPUTE_EFFICIENCY = 0.70;
const DEFAULT_MEMORY_EFFICIENCY = 0.78;
const DEFAULT_INTERCONNECT_EFFICIENCY = 0.82;

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function positive(value: number): boolean {
  return finite(value) && value > 0;
}

function median(values: number[]): number {
  if (!values.length) return 1;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function percentile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[index]!;
}

function fitEfficiency(rows: ChipCalibrationObservation[], kind: ChipCalibrationKind, fallback: number): number {
  const fits = rows
    .filter((row) => row.kind === kind && positive(row.baselineSeconds) && positive(row.observedSeconds))
    .map((row) => row.baselineSeconds / row.observedSeconds)
    .filter((value) => finite(value) && value > 0)
    .map((value) => Math.min(1.5, Math.max(0.05, value)));
  return fits.length ? median(fits) : fallback;
}

export function validateChipDesign(design: ChipDesignSpec, workload: ChipWorkloadSpec, constraints: ChipConstraints): string[] {
  const errors: string[] = [];
  const designFields: Array<[keyof ChipDesignSpec, string]> = [
    ["processNm", "process node"], ["dieAreaMm2", "die area"], ["tileCount", "tile count"],
    ["bf16Tops", "BF16 throughput"], ["fp8Tops", "FP8 throughput"], ["memoryBandwidthGbps", "memory bandwidth"],
    ["sramGb", "SRAM capacity"], ["hbmGb", "HBM capacity"], ["interconnectGbps", "interconnect bandwidth"],
    ["interconnectLatencyNs", "interconnect latency"], ["powerW", "power"], ["voltageV", "voltage"],
    ["frequencyGHz", "frequency"], ["unitCostUsd", "unit cost"], ["hourlyCostUsd", "hourly cost"],
  ];
  for (const [key, label] of designFields) if (!positive(Number(design[key]))) errors.push(`${label} must be positive`);
  if (!finite(design.yieldPct) || design.yieldPct <= 0 || design.yieldPct > 100) errors.push("yield must be between 0 and 100% ");
  const workloadFields: Array<[keyof ChipWorkloadSpec, string]> = [
    ["denseFlops", "dense operations"], ["memoryBytes", "memory traffic"], ["interconnectBytes", "interconnect traffic"], ["batchSize", "batch size"],
  ];
  for (const [key, label] of workloadFields) if (!finite(workload[key]) || workload[key] < 0) errors.push(`${label} cannot be negative`);
  if (!finite(workload.spikingOps) || workload.spikingOps < 0) errors.push("spiking operations cannot be negative");
  const constraintFields: Array<[keyof ChipConstraints, string]> = [
    ["maxPowerW", "power limit"], ["maxAreaMm2", "area limit"], ["maxLatencyMs", "latency limit"],
    ["minThroughputPerSecond", "throughput target"], ["maxCostUsdPerBatch", "cost limit"], ["minYieldPct", "yield target"],
  ];
  for (const [key, label] of constraintFields) if (!positive(constraints[key])) errors.push(`${label} must be positive`);
  if (workload.spikingOps > 0 && design.neuromorphicCores <= 0) errors.push("spiking workload needs at least one neuromorphic core");
  return errors;
}

export function fitChipCalibration(rows: ChipCalibrationObservation[], nominalPowerW: number): ChipCalibrationFit {
  const valid = rows.filter((row) => positive(row.baselineSeconds) && positive(row.observedSeconds));
  const fittedKinds = (["compute", "memory", "interconnect", "end_to_end"] as ChipCalibrationKind[]).filter((kind) => valid.some((row) => row.kind === kind));
  const residuals = valid.map((row) => Math.abs(1 - row.observedSeconds / row.baselineSeconds)).filter(finite);
  const powerRatios = valid.map((row) => row.observedPowerW && nominalPowerW > 0 ? row.observedPowerW / nominalPowerW : null).filter((value): value is number => finite(value) && value > 0);
  const status = fittedKinds.length >= 3 || (fittedKinds.includes("end_to_end") && valid.length >= 3) ? "calibrated" : fittedKinds.length ? "partial" : "assumption";
  return {
    status,
    computeEfficiency: fitEfficiency(valid, "compute", DEFAULT_COMPUTE_EFFICIENCY),
    memoryEfficiency: fitEfficiency(valid, "memory", DEFAULT_MEMORY_EFFICIENCY),
    interconnectEfficiency: fitEfficiency(valid, "interconnect", DEFAULT_INTERCONNECT_EFFICIENCY),
    powerScale: powerRatios.length ? Math.min(2, Math.max(0.25, median(powerRatios))) : 1,
    sampleCount: valid.length,
    fittedKinds,
    residualP90Pct: percentile(residuals, 0.9) === null ? null : percentile(residuals, 0.9)! * 100,
    note: status === "assumption"
      ? "No valid measurement rows. Efficiencies are explicit planning assumptions."
      : status === "partial"
        ? "Some paths are fitted; remaining paths still use planning assumptions."
        : "All primary paths have measurement rows; residual error is reported, not hidden.",
  };
}

function constraint(label: string, value: number, limit: number, unit: string, direction: "max" | "min"): ChipConstraintResult {
  const pass = direction === "max" ? value <= limit : value >= limit;
  return { label, value, limit, unit, direction, pass, headroom: direction === "max" ? limit - value : value - limit };
}

export function evaluateChipDesign(
  design: ChipDesignSpec,
  workload: ChipWorkloadSpec,
  constraints: ChipConstraints,
  observations: ChipCalibrationObservation[] = [],
): ChipDesignEvaluation {
  const errors = validateChipDesign(design, workload, constraints);
  if (errors.length) throw new Error(errors[0]);
  const calibration = fitChipCalibration(observations, design.powerW);
  const computeSeconds = workload.denseFlops / (design[workload.precision === "fp8" ? "fp8Tops" : "bf16Tops"] * 1e12 * calibration.computeEfficiency);
  const memorySeconds = workload.memoryBytes / (design.memoryBandwidthGbps * 1e9 * calibration.memoryEfficiency);
  const interconnectSeconds = workload.interconnectBytes / (design.interconnectGbps * 1e9 * calibration.interconnectEfficiency) + design.interconnectLatencyNs * 1e-9;
  const neuromorphicSeconds = workload.spikingOps > 0 ? workload.spikingOps / (Math.max(1, design.neuromorphicCores) * 1e9 * calibration.computeEfficiency) : 0;
  const componentSeconds = { compute: computeSeconds, memory: memorySeconds, interconnect: interconnectSeconds, neuromorphic: neuromorphicSeconds } as const;
  const bottleneck = (Object.entries(componentSeconds) as Array<[keyof typeof componentSeconds, number]>).sort((a, b) => b[1] - a[1])[0]![0];
  const latencySeconds = Math.max(...Object.values(componentSeconds));
  const throughputPerSecond = workload.batchSize / latencySeconds;
  const powerW = design.powerW * calibration.powerScale;
  const energyJPerBatch = latencySeconds * powerW;
  const costUsdPerBatch = latencySeconds / 3600 * design.hourlyCostUsd;
  const checks = {
    power: constraint("Power", powerW, constraints.maxPowerW, "W", "max"),
    area: constraint("Die area", design.dieAreaMm2, constraints.maxAreaMm2, "mm²", "max"),
    latency: constraint("Latency", latencySeconds * 1000, constraints.maxLatencyMs, "ms", "max"),
    throughput: constraint("Throughput", throughputPerSecond, constraints.minThroughputPerSecond, "req/s", "min"),
    cost: constraint("Cost per batch", costUsdPerBatch, constraints.maxCostUsdPerBatch, "USD", "max"),
    yield: constraint("Yield", design.yieldPct, constraints.minYieldPct, "%", "min"),
  };
  const notes = [
    calibration.note,
    `Critical path is ${bottleneck}; inspect the variant table before increasing peak compute.`,
    "Area, power, timing closure, yield, and cost remain planning constraints until backed by PDK, synthesis, packaging, or hardware data.",
  ];
  return {
    feasible: Object.values(checks).every((check) => check.pass),
    latencySeconds,
    throughputPerSecond,
    energyJPerBatch,
    costUsdPerBatch,
    powerW,
    bottleneck,
    componentSeconds,
    constraints: checks,
    calibration,
    evidence: calibration.status === "calibrated" ? "calibrated" : calibration.status === "partial" ? "measured" : "assumption",
    notes,
  };
}

export function buildChipDesignVariants(design: ChipDesignSpec): ChipDesignVariant[] {
  return [
    { id: "baseline", label: "Baseline", rationale: "Your entered architecture.", changes: ["No scenario changes"], spec: design },
    {
      id: "compute-forward", label: "Compute-forward", rationale: "Spend area and power to relieve a compute-bound path.",
      changes: ["+25% BF16/FP8 throughput", "+12% area", "+12% power", "+12% unit cost"],
      spec: { ...design, bf16Tops: design.bf16Tops * 1.25, fp8Tops: design.fp8Tops * 1.25, dieAreaMm2: design.dieAreaMm2 * 1.12, powerW: design.powerW * 1.12, unitCostUsd: design.unitCostUsd * 1.12 },
    },
    {
      id: "memory-forward", label: "Memory-forward", rationale: "Spend area and power to relieve a bandwidth-bound path.",
      changes: ["+25% memory bandwidth", "+15% SRAM", "+10% area", "+8% power"],
      spec: { ...design, memoryBandwidthGbps: design.memoryBandwidthGbps * 1.25, sramGb: design.sramGb * 1.15, dieAreaMm2: design.dieAreaMm2 * 1.10, powerW: design.powerW * 1.08, unitCostUsd: design.unitCostUsd * 1.10 },
    },
    {
      id: "edge-efficient", label: "Edge-efficient", rationale: "Trade peak speed for a smaller thermal and cost envelope.",
      changes: ["−20% peak compute", "−20% power", "−15% area", "−15% unit cost"],
      spec: { ...design, bf16Tops: design.bf16Tops * 0.80, fp8Tops: design.fp8Tops * 0.80, memoryBandwidthGbps: design.memoryBandwidthGbps * 0.85, dieAreaMm2: design.dieAreaMm2 * 0.85, powerW: design.powerW * 0.80, unitCostUsd: design.unitCostUsd * 0.85 },
    },
  ];
}

function csvLine(values: unknown[]): string {
  return values.map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(",");
}

export function chipCalibrationCsvTemplate(): string {
  return [
    csvLine(["name", "kind", "baseline_seconds", "observed_seconds", "observed_power_w", "source", "observed_at"]),
    csvLine(["gemm-benchmark", "compute", "0.001", "0.0014", "220", "lab-run-001", "2026-10-14"]),
  ].join("\n");
}

function parseCsv(input: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < input.length; i += 1) {
    const char = input[i]!;
    const next = input[i + 1];
    if (char === '"' && quoted && next === '"') { cell += '"'; i += 1; continue; }
    if (char === '"') { quoted = !quoted; continue; }
    if (char === "," && !quoted) { row.push(cell.trim()); cell = ""; continue; }
    if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
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

export function parseChipCalibrationCsv(input: string): ChipCalibrationObservation[] {
  const rows = parseCsv(input);
  if (rows.length < 2) throw new Error("calibration CSV needs a header and at least one row");
  const header = rows[0]!.map((value) => value.toLowerCase());
  const index = (name: string) => header.indexOf(name);
  for (const required of ["kind", "baseline_seconds", "observed_seconds"]) if (index(required) < 0) throw new Error(`calibration CSV is missing ${required}`);
  return rows.slice(1).map((row, rowIndex) => {
    const value = (name: string) => row[index(name)] ?? "";
    const kind = value("kind") as ChipCalibrationKind;
    if (!["compute", "memory", "interconnect", "end_to_end"].includes(kind)) throw new Error(`row ${rowIndex + 2}: unknown calibration kind`);
    const baselineSeconds = Number(value("baseline_seconds"));
    const observedSeconds = Number(value("observed_seconds"));
    if (!positive(baselineSeconds) || !positive(observedSeconds)) throw new Error(`row ${rowIndex + 2}: baseline and observed seconds must be positive`);
    const power = value("observed_power_w");
    return {
      id: `csv-${rowIndex + 1}`,
      name: value("name") || `Imported row ${rowIndex + 1}`,
      kind,
      baselineSeconds,
      observedSeconds,
      observedPowerW: power ? Number(power) : null,
      source: value("source") || "Imported measurement",
      observedAt: value("observed_at") || new Date().toISOString().slice(0, 10),
    };
  });
}

export function chipCalibrationCsv(rows: ChipCalibrationObservation[]): string {
  const header = ["name", "kind", "baseline_seconds", "observed_seconds", "observed_power_w", "source", "observed_at"];
  return [csvLine(header), ...rows.map((row) => csvLine([row.name, row.kind, row.baselineSeconds, row.observedSeconds, row.observedPowerW ?? "", row.source, row.observedAt]))].join("\n");
}

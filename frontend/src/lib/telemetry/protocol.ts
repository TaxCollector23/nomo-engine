/**
 * Telemetry wire protocol v1 — mirror of nomo/telemetry/schema.py (SPEC §9).
 * Any change here must be made there too; `isEnvelope` guards the boundary.
 */

export const PROTOCOL_VERSION = 1 as const;

export enum Domain { ANN = 0, SNN = 1, SYM = 2 }
export enum Coding { NONE = 0, RATE = 1, TTFS = 2 }
export enum GuardImpl { HOST = 0, FUSED = 1 }

/** [domain, w_bits, a_bits, coding, T, plastic] */
export type LayerWire = [Domain, number, number, Coding, number, 0 | 1];
/** [site (after_layer), impl] */
export type GuardWire = [number, GuardImpl];

export interface GenomeWire {
  layers: LayerWire[];
  guards: GuardWire[];
}

export interface EvalItem {
  key: string;
  /** [energy_j, latency_s, accuracy_pct] */
  f: [number, number, number];
  cv: number;
  feasible: boolean;
  rank: number;
  acc_src: "proxy" | "oracle";
  genome: GenomeWire;
  crossings: number;
  cores: number;
  parent?: string;
  changed?: number[];
  ops?: [string | null, string];
}

export interface RunStarted {
  model: string;
  hardware: string;
  layers: string[];
  guard_sites: number[];
  pop_size: number;
  generations: number;
  objectives: string[];
}

export interface EvalBatch { gen: number; items: EvalItem[] }

export interface GenCompleted {
  gen: number;
  hv: number;
  evaluations: number;
  unique: number;
  feasible_fraction: number;
  front: string[];
  population: string[];
  recommended: EvalItem | null;
  operators: Record<string, number>;
}

export interface RunCompleted {
  recommended: EvalItem | null;
  front: EvalItem[];
  generations: number;
  evaluations: number;
  unique: number;
  hv: number;
  wall_s: number;
}

export interface RunFailed { error: string }

export type RunStatus = "pending" | "running" | "completed" | "stopped" | "failed";

export interface Snapshot {
  status: RunStatus;
  run: RunStarted | null;
  items: EvalItem[];
  last_gen: GenCompleted | null;
}

interface EnvelopeBase<T extends string, D> {
  v: typeof PROTOCOL_VERSION;
  run_id: string;
  seq: number;
  ts: number;
  type: T;
  data: D;
}

export type Envelope =
  | EnvelopeBase<"run.started", RunStarted>
  | EnvelopeBase<"eval.batch", EvalBatch>
  | EnvelopeBase<"gen.completed", GenCompleted>
  | EnvelopeBase<"run.completed", RunCompleted>
  | EnvelopeBase<"run.failed", RunFailed>
  | EnvelopeBase<"snapshot", Snapshot>;

export type EnvelopeType = Envelope["type"];

const TYPES: ReadonlySet<string> = new Set<EnvelopeType>([
  "run.started", "eval.batch", "gen.completed", "run.completed", "run.failed", "snapshot",
]);

export function isEnvelope(x: unknown): x is Envelope {
  if (typeof x !== "object" || x === null) return false;
  const o = x as Record<string, unknown>;
  return o.v === PROTOCOL_VERSION && typeof o.seq === "number" && typeof o.run_id === "string"
    && typeof o.type === "string" && TYPES.has(o.type) && typeof o.data === "object" && o.data !== null;
}

/** REST payloads (POST /runs). */
export interface BudgetIn {
  energy_j?: number | null;
  latency_s?: number | null;
  accuracy_min?: number | null;
  accuracy_drop_max?: number | null;
  period_s?: number | null;
  min_plastic_params?: number;
}

export interface SearchIn {
  allow_continuous: boolean;
  allow_spiking: boolean;
  allow_symbolic: boolean;
  codings: ("rate" | "ttfs")[];
  crossing_penalty: number;
  crossing_min_saving_pct: number;
  p_crossover: number;
  p_mutation: number;
  archive_capacity: number;
  patience: number;
  asf_weights: [number, number, number] | null;
}

export type Precision = "INT16" | "INT8" | "INT4" | "INT2" | "BINARY";
export interface PinIn { domain?: "ANN" | "SNN" | "SYM" | null; w_bits?: Precision | number | null; a_bits?: Precision | number | null }

export interface HardwareIn {
  mac_energy_pj?: number | null;
  sop_energy_pj?: number | null;
  neuron_energy_pj?: number | null;
  sram_kb_per_core?: number | null;
  neurons_per_core?: number | null;
  n_cores?: number | null;
  bus_bandwidth_gbs?: number | null;
  routing_latency_us?: number | null;
  timestep_us?: number | null;
  static_power_mw?: number | null;
  clock_mhz?: number | null;
}

export interface RunIn {
  model: string;
  hardware: string;
  budgets: BudgetIn;
  pop_size: number;
  generations: number;
  seed: number;
  search?: SearchIn;
  pins?: Record<string, PinIn>;
  lock_symbolic?: boolean;
  hardware_overrides?: HardwareIn | null;
  preset?: string | null;
  mode?: string | null;
}

export interface LayerRow {
  name: string;
  op: string;
  activation: string;
  params: number;
  macs: number;
  fan_in: number;
  fan_out: number;
  weight_shape: number[];
  weight_kb_int8: number;
  can_be: ("ANN" | "SNN" | "SYM")[];
  in_shape: number[] | null;
  out_shape: number[] | null;
}

export interface Catalog {
  models: Record<string, { layers: string[]; base_accuracy: number; layer_table: LayerRow[] }>;
  hardware: Record<string, { name: string; provenance: Record<string, string>; defaults: Record<string, number>;
    bits: { continuous: number[]; spiking: number[] } }>;
  modes?: Record<string, { title: string; summary: string; requirements: string[]; export_tags: string[] }>;
  limits: { max_pop: number; max_generations: number };
}

export interface Preset {
  label: string;
  summary: string;
  notes: string;
  settings: { budgets: BudgetIn; search: Partial<SearchIn>; lock_symbolic?: boolean };
}

export interface UploadReport {
  source_format: string; layers: number; params: number; macs: number;
  weights_source: string; assumptions: string[]; dropped_ops: string[];
}

export interface UploadedModel {
  model_id: string; name: string; base_accuracy: number; input_shape: number[];
  report: UploadReport; layer_table: LayerRow[]; calibration?: CalibrationSummary | null;
}

export interface CalibrationSummary { sample_count: number; input_shape: number[]; aux_shape: number[] | null; source: string }

export interface Capability { label: string; available: boolean; note: string; reason?: string }

export interface DesignLayer {
  name: string; op: string; domain: "ANN" | "SNN" | "SYM"; w_bits: number; a_bits: number;
  coding: string | null; timesteps: number | null; plastic: boolean;
  energy_j: number | null; memory_bytes: number | null; cores: number | null; params: number; macs: number;
}

export interface DesignDetail {
  design: {
    design_key: string; model: string; hardware: { id: string; name: string; provenance: Record<string, string> };
    metrics: { energy_j: number; latency_s: number; accuracy_pct: number; accuracy_source: string; feasible: boolean;
      domain_crossings: number; cores_used: number; memory_bytes: number };
    baseline_all_continuous: { energy_j: number; latency_s: number; accuracy: number };
    layers: DesignLayer[]; weights_source: string; assumptions: string[];
  };
  summary: CopilotAnswer;
  capabilities: Record<string, Capability>;
}

export type CopilotAction =
  | { type: "select"; key: string; label: string }
  | { type: "rerun"; settings: Partial<RunIn>; label: string }
  | { type: "apply_preset"; preset: string; label: string }
  | { type: "ask"; text: string; label: string };

export interface CopilotAnswer { text: string; facts: Record<string, unknown>; actions: CopilotAction[]; source: "rules" | "llm" }

/** Hosted backend used when NEXT_PUBLIC_NOMO_API is not set at build time. */
export const DEFAULT_API = "https://nomo-engine.onrender.com";

export function apiBase(): string {
  return (process.env.NEXT_PUBLIC_NOMO_API || DEFAULT_API).replace(/\/$/, "");
}

export function wsUrl(runId: string, since: number, client?: string): string {
  const base = apiBase().replace(/^http/, "ws");
  const q = new URLSearchParams();
  if (since > 0) q.set("since", String(since));
  if (client) q.set("client", client);
  const qs = q.toString();
  return `${base}/ws/runs/${encodeURIComponent(runId)}${qs ? `?${qs}` : ""}`;
}

export function geneCode(l: LayerWire): string {
  const [d, w, a, c, T, p] = l;
  if (d === Domain.SNN) return `S${w}.${a}${c === Coding.TTFS ? "T" : "R"}${T}${p ? "p" : ""}`;
  return `${d === Domain.ANN ? "A" : "Y"}${w}.${a}`;
}

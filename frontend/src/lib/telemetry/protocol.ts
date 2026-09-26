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

export interface RunIn {
  model: string;
  hardware: string;
  budgets: BudgetIn;
  pop_size: number;
  generations: number;
  seed: number;
}

export interface Catalog {
  models: Record<string, { layers: string[]; base_accuracy: number }>;
  hardware: Record<string, { name: string; provenance: Record<string, string> }>;
}

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

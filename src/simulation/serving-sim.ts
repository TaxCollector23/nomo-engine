/**
 * Browser port of nomo_planner.serving_sim — the deterministic reference
 * simulator for autoregressive serving (non-disaggregated path).
 *
 * The Python reference is the source of truth; this file mirrors it line for
 * line so a run with the same seed reproduces Python's request-level results
 * exactly. Parity is enforced by scripts/serving-sim-golden.json (generated
 * by nomo-planner/studies/export_serving_sim_golden.py) inside
 * `npm run verify`.
 *
 * Not ported (kept server-only, like the module docs say): disaggregated
 * prefill/decode pools, CSV ingestion and text-length heuristics. The port
 * throws an explicit error if asked for them rather than guessing.
 */

import { PyRandom, samplePoisson } from "./pyrandom";
import { SIMULATED_LABEL } from "./serving-contracts";

// --------------------------------------------------------------------- utils

export function pyRound(x: number): number {
  // Python round(): banker's rounding (half to even).
  if (!Number.isFinite(x)) return x;
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

const finite = (value: unknown, fallback = 0): number => {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
};

const ceilDiv = (value: number, divisor: number) => (value <= 0 ? 0 : Math.floor((value + divisor - 1) / divisor));

// -------------------------------------------------------------- distributions

export type SimDistribution =
  | { kind: "fixed" | "constant" | "deterministic"; value?: number }
  | { kind: "uniform" | "flat"; low?: number; high?: number }
  | { kind: "normal" | "gaussian"; mean?: number; stddev?: number }
  | { kind: "lognormal" | "log_normal"; mean?: number; stddev?: number } // log-space mean/stddev
  | { kind: "exponential" | "exp"; ratePerS?: number; mean?: number; scale?: number }
  | { kind: "gamma"; shape?: number; scale?: number }
  | { kind: "poisson"; mean?: number }
  | { kind: "choice" | "categorical"; values?: number[]; weights?: number[]; value?: number }
  | { kind: "empirical" | "sample" | "samples"; values?: number[]; value?: number }
  | { kind: string; [key: string]: unknown };

/** serving_sim._coerce_distribution: accept dicts, numbers, or arrays. */
export function coerceDistribution(value: unknown, fallback: SimDistribution = { kind: "fixed", value: 1.0 }): SimDistribution {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "number" && Number.isFinite(value)) return { kind: "fixed", value };
  if (Array.isArray(value)) return { kind: "empirical", values: value.map(Number) };
  if (typeof value === "object") {
    const raw = { ...(value as Record<string, unknown>) };
    const kind = String(raw.kind ?? raw.type ?? raw.distribution ?? "fixed");
    return { kind, ...raw } as SimDistribution;
  }
  return fallback;
}

/** serving_sim.DistributionSpec.sample — identical RNG call sequence. */
export function sampleDistribution(spec: SimDistribution, rng: PyRandom): number {
  const kind = String(spec.kind || "fixed").toLowerCase().replace(/-/g, "_");
  const params = (spec as Record<string, unknown>) ?? {};
  let value = finite((spec as { value?: number }).value, 0.0);
  if ("value" in params && params.value !== undefined && params.value !== null) value = finite(params.value, value);
  let mean = finite((spec as { mean?: number }).mean, finite(value, 0));
  if ("mean" in params && params.mean !== undefined && params.mean !== null) mean = finite(params.mean, mean);
  let low = finite(params.low ?? (spec as { low?: number }).low, 0.0);
  let high = finite(params.high ?? (spec as { high?: number }).high, low);
  if (high < low) [low, high] = [high, low];
  const stddev = Math.max(0, finite(params.stddev ?? params.std ?? (spec as { stddev?: number }).stddev, 0.0));
  const shape = Math.max(1e-12, finite(params.shape ?? (spec as { shape?: number }).shape, 1.0));
  const scale = Math.max(1e-12, finite(params.scale ?? (spec as { scale?: number }).scale, mean > 0 ? mean : 1.0));
  const values = ((params.values ?? (spec as { values?: number[] }).values) as unknown[] | undefined)?.map((v) => finite(v, 0)) ?? [];
  const weights = ((params.weights ?? (spec as { weights?: number[] }).weights) as unknown[] | undefined)?.map((w) => finite(w, 0)) ?? [];

  if (kind === "fixed" || kind === "constant" || kind === "deterministic") return finite(value, mean);
  if (kind === "uniform" || kind === "flat") return rng.uniform(low, high);
  if (kind === "normal" || kind === "gaussian") return rng.gauss(mean, stddev);
  if (kind === "lognormal" || kind === "log_normal") return rng.lognormvariate(mean, stddev);
  if (kind === "exponential" || kind === "exp") {
    const rate = finite(params.rate_per_s ?? params.ratePerS ?? params.rate, 0.0);
    const meanGap = finite(params.mean_gap ?? params.meanGap ?? params.mean, 0.0);
    const scaleValue = rate > 0 ? 1 / rate : Math.max(meanGap, scale, 1e-12);
    return rng.expovariate(1.0 / scaleValue);
  }
  if (kind === "gamma") return rng.gammavariate(shape, scale);
  if (kind === "poisson" || kind === "pois") return samplePoisson(rng, Math.max(0, mean));
  if (kind === "choice" || kind === "categorical") {
    if (!values.length) return finite(value, mean);
    return weightedChoice(rng, values, weights);
  }
  if (kind === "empirical" || kind === "sample" || kind === "samples") {
    if (!values.length) return finite(value, mean);
    return rng.choice(values);
  }
  return finite(value, mean);
}

function weightedChoice<T>(rng: PyRandom, values: readonly T[], weights: readonly number[]): T {
  if (!weights.length || weights.length !== values.length || weights.reduce((a, w) => a + Math.max(0, w), 0) <= 0) {
    return rng.choice(values);
  }
  const total = weights.reduce((a, w) => a + Math.max(0, w), 0);
  const draw = rng.random() * total;
  let cumulative = 0;
  for (let i = 0; i < values.length; i++) {
    cumulative += Math.max(0, weights[i]!);
    if (draw <= cumulative) return values[i]!;
  }
  return values[values.length - 1]!;
}

// ---------------------------------------------------------------- assumptions

export interface ServingSimAssumptions {
  seed?: number;
  arrivalProcess: "poisson" | "bursty" | "replay" | string;
  requestCount?: number;
  durationS?: number | null;
  arrivalRatePerS?: number;
  burstRatePerS?: number;
  burstDurationS?: number;
  quietDurationS?: number;
  replayRows?: ServingSimInputRow[];
  promptDistribution?: SimDistribution;
  answerDistribution?: SimDistribution;
  replicas?: number;
  tensorParallel?: number;
  tpEfficiency?: number;
  prefillTokensPerS?: number;
  decodeTokensPerS?: number;
  draftTokensPerS?: number;
  prefillBatchOverheadS?: number;
  decodeBatchOverheadS?: number;
  prefillBatchGain?: number;
  decodeBatchGain?: number;
  speculativeVerificationFactor?: number;
  maxBatchSize?: number;
  chunkedPrefillTokens?: number;
  maxDecodeStepsBeforePrefill?: number;
  continuousBatching?: boolean;
  kvBlockTokens?: number;
  kvCapacityBlocks?: number | null;
  kvCapacityTokens?: number;
  evictionPolicy?: string;
  preemptionEnabled?: boolean;
  prefixCacheEnabled?: boolean;
  prefixCacheCapacityBlocks?: number | null;
  prefixCacheMinTokens?: number;
  speculativeEnabled?: boolean;
  speculativeDraftTokens?: number;
  speculativeAcceptance?: SimDistribution | number;
  sloTtftS?: number | null;
  sloItlS?: number | null;
  sloE2eS?: number | null;
}

export interface ServingSimInputRow {
  requestId?: string | null;
  arrivalS?: number | null;
  arrivalMs?: number | null;
  promptTokens?: number | null;
  answerTokens?: number | null;
  prefixId?: string | null;
  prefixTokens?: number | null;
}

interface ResolvedAssumptions {
  seed: number;
  arrivalProcess: string;
  requestCount: number;
  durationS: number | null;
  arrivalRatePerS: number;
  burstRatePerS: number;
  burstDurationS: number;
  quietDurationS: number;
  replayRows: ServingSimInputRow[];
  promptDistribution: SimDistribution;
  answerDistribution: SimDistribution;
  replicas: number;
  tensorParallel: number;
  tpEfficiency: number;
  prefillTokensPerS: number;
  decodeTokensPerS: number;
  draftTokensPerS: number;
  prefillBatchOverheadS: number;
  decodeBatchOverheadS: number;
  prefillBatchGain: number;
  decodeBatchGain: number;
  speculativeVerificationFactor: number;
  maxBatchSize: number;
  chunkedPrefillTokens: number;
  maxDecodeStepsBeforePrefill: number;
  continuousBatching: boolean;
  kvBlockTokens: number;
  kvCapacityBlocks: number | null;
  kvCapacityTokens: number;
  evictionPolicy: string;
  preemptionEnabled: boolean;
  prefixCacheEnabled: boolean;
  prefixCacheCapacityBlocks: number | null;
  prefixCacheMinTokens: number;
  speculativeEnabled: boolean;
  speculativeDraftTokens: number;
  speculativeAcceptance: SimDistribution;
  sloTtftS: number | null;
  sloItlS: number | null;
  sloE2eS: number | null;
}

const posInt = (value: unknown, fallback: number, minimum = 1) => {
  if (value === undefined || value === null || value === "") return fallback;
  return Math.max(minimum, pyRound(finite(value, fallback)));
};
const nonNegInt = (value: unknown, fallback = 0) => Math.max(0, pyRound(finite(value, fallback)));
const num0 = (value: unknown, fallback: number) => (value === undefined || value === null || value === "" ? fallback : finite(value, fallback));
const bool0 = (value: unknown, fallback: boolean) => (value === undefined || value === null ? fallback : Boolean(value));

/** Resolve user input onto the Python dataclass defaults (snake_case or camelCase). */
export function resolveAssumptions(raw?: ServingSimAssumptions | Record<string, unknown> | null): ResolvedAssumptions {
  const r: Record<string, unknown> = { ...(raw ?? {}) };
  // accept the nested "arrival" section like the Python mapping coercion
  if (r.arrival && typeof r.arrival === "object") {
    const arrival = { ...(r.arrival as Record<string, unknown>) };
    for (const [key, value] of Object.entries(arrival)) if (r[key] === undefined) r[key] = value;
    if (r.arrivalProcess === undefined) r.arrivalProcess = arrival.kind ?? arrival.type ?? "replay";
  }
  const promptFallback: SimDistribution = { kind: "fixed", value: 256 };
  const answerFallback: SimDistribution = { kind: "fixed", value: 64 };
  const promptRaw = r.prompt_distribution ?? r.promptDistribution ?? r.prompt_tokens ?? r.promptTokens;
  const answerRaw = r.answer_distribution ?? r.answerDistribution ?? r.answer_tokens ?? r.answerTokens;
  const kvCapacityBlocksRaw = r.kv_capacity_blocks ?? r.kvCapacityBlocks;
  return {
    seed: Math.trunc(finite(r.seed, 0)),
    arrivalProcess: String(r.arrival_process ?? r.arrivalProcess ?? (r.replay_rows || r.replayRows ? "replay" : "replay")),
    requestCount: nonNegInt(r.request_count ?? r.requestCount, 32),
    durationS: (r.duration_s ?? r.durationS ?? null) === null ? null : finite(r.duration_s ?? r.durationS, 0),
    arrivalRatePerS: num0(r.arrival_rate_per_s ?? r.arrivalRatePerS, 1.0),
    burstRatePerS: num0(r.burst_rate_per_s ?? r.burstRatePerS, 5.0),
    burstDurationS: num0(r.burst_duration_s ?? r.burstDurationS, 1.0),
    quietDurationS: num0(r.quiet_duration_s ?? r.quietDurationS, 1.0),
    replayRows: ((r.replay_rows ?? r.replayRows ?? []) as ServingSimInputRow[]).slice(),
    promptDistribution: coerceDistribution(promptRaw, promptFallback),
    answerDistribution: coerceDistribution(answerRaw, answerFallback),
    replicas: posInt(r.replicas, 1),
    tensorParallel: posInt(r.tensor_parallel ?? r.tensorParallel, 1),
    tpEfficiency: num0(r.tp_efficiency ?? r.tpEfficiency, 0.9),
    prefillTokensPerS: num0(r.prefill_tokens_per_s ?? r.prefillTokensPerS, 10_000.0),
    decodeTokensPerS: num0(r.decode_tokens_per_s ?? r.decodeTokensPerS, 100.0),
    draftTokensPerS: num0(r.draft_tokens_per_s ?? r.draftTokensPerS, 400.0),
    prefillBatchOverheadS: num0(r.prefill_batch_overhead_s ?? r.prefillBatchOverheadS, 0.002),
    decodeBatchOverheadS: num0(r.decode_batch_overhead_s ?? r.decodeBatchOverheadS, 0.001),
    prefillBatchGain: num0(r.prefill_batch_gain ?? r.prefillBatchGain, 0.2),
    decodeBatchGain: num0(r.decode_batch_gain ?? r.decodeBatchGain, 0.15),
    speculativeVerificationFactor: num0(r.speculative_verification_factor ?? r.speculativeVerificationFactor, 0.15),
    maxBatchSize: posInt(r.max_batch_size ?? r.maxBatchSize, 8),
    chunkedPrefillTokens: nonNegInt(r.chunked_prefill_tokens ?? r.chunkedPrefillTokens, 0),
    maxDecodeStepsBeforePrefill: posInt(r.max_decode_steps_before_prefill ?? r.maxDecodeStepsBeforePrefill, 1),
    continuousBatching: bool0(r.continuous_batching ?? r.continuousBatching, true),
    kvBlockTokens: posInt(r.kv_block_tokens ?? r.kvBlockTokens, 16),
    kvCapacityBlocks: kvCapacityBlocksRaw === undefined || kvCapacityBlocksRaw === null ? null : posInt(kvCapacityBlocksRaw, 1),
    kvCapacityTokens: posInt(r.kv_capacity_tokens ?? r.kvCapacityTokens, 65_536),
    evictionPolicy: String(r.eviction_policy ?? r.evictionPolicy ?? "lru").toLowerCase(),
    preemptionEnabled: bool0(r.preemption_enabled ?? r.preemptionEnabled, true),
    prefixCacheEnabled: bool0(r.prefix_cache_enabled ?? r.prefixCacheEnabled, true),
    prefixCacheCapacityBlocks: r.prefix_cache_capacity_blocks ?? r.prefixCacheCapacityBlocks ?? null,
    prefixCacheMinTokens: posInt(r.prefix_cache_min_tokens ?? r.prefixCacheMinTokens, 1),
    speculativeEnabled: bool0(r.speculative_enabled ?? r.speculativeEnabled, false),
    speculativeDraftTokens: posInt(r.speculative_draft_tokens ?? r.speculativeDraftTokens, 4),
    speculativeAcceptance: coerceDistribution(r.speculative_acceptance ?? r.speculativeAcceptance, { kind: "fixed", value: 0.8 }),
    sloTtftS: r.slo_ttft_s ?? r.sloTtftS ?? null,
    sloItlS: r.slo_itl_s ?? r.sloItlS ?? null,
    sloE2eS: r.slo_e2e_s ?? r.sloE2eS ?? null,
  } as ResolvedAssumptions;
}

// -------------------------------------------------------------------- requests

interface TraceRequest {
  requestId: string;
  arrivalS: number;
  promptTokens: number;
  answerTokens: number;
  prefixId: string | null;
  prefixTokens: number;
}

function rowToTrace(row: ServingSimInputRow, index: number, a: ResolvedAssumptions, rng: PyRandom): TraceRequest {
  const requestId = String(row.requestId ?? `request-${index}`);
  let arrival = 0;
  const arrivalRaw = row.arrivalS ?? row.arrivalMs ?? null;
  if (arrivalRaw !== null && arrivalRaw !== undefined) arrival = finite(arrivalRaw, 0);
  let prompt = row.promptTokens ?? null;
  let answer = row.answerTokens ?? null;
  if (prompt === null || prompt === undefined) prompt = sampleDistribution(a.promptDistribution, rng);
  if (answer === null || answer === undefined) answer = sampleDistribution(a.answerDistribution, rng);
  const promptTokens = Math.max(1, pyRound(finite(prompt, 1.0)));
  const answerTokens = Math.max(0, pyRound(finite(answer, 0.0)));
  let prefixId = row.prefixId ?? null;
  let prefixTokens = row.prefixTokens ?? null;
  if (prefixId === null || prefixId === undefined) prefixTokens = 0;
  else if (prefixTokens === null || prefixTokens === undefined) prefixTokens = promptTokens;
  prefixTokens = Math.min(promptTokens, Math.max(0, pyRound(finite(prefixTokens ?? 0, 0))));
  return { requestId, arrivalS: Math.max(0, finite(arrival, 0)), promptTokens, answerTokens, prefixId, prefixTokens };
}

function generatedTrace(a: ResolvedAssumptions, rng: PyRandom): TraceRequest[] {
  const kind = String(a.arrivalProcess || "replay").toLowerCase().replace(/-/g, "_");
  const rate = Math.max(0, a.arrivalRatePerS);
  const burstRate = Math.max(0, a.burstRatePerS);
  const onDuration = Math.max(1e-12, a.burstDurationS);
  const offDuration = Math.max(0, a.quietDurationS);
  const horizon = a.durationS === null || a.durationS === undefined ? null : Math.max(0, finite(a.durationS, 0));
  const target = nonNegInt(a.requestCount, 0);
  if (kind === "replay" || kind === "trace") return a.replayRows.map((row, index) => rowToTrace(row, index, a, rng));
  const safetyLimit = Math.max(1, target > 0 ? target : 10_000);
  const arrivals: number[] = [];
  let timeS = 0;
  if (kind === "poisson" || kind === "exp" || kind === "exponential") {
    while (arrivals.length < safetyLimit && (horizon === null || timeS <= horizon)) {
      if (rate <= 0) break;
      timeS += rng.expovariate(rate);
      if (horizon !== null && timeS > horizon) break;
      arrivals.push(timeS);
    }
  } else if (kind === "bursty" || kind === "burst" || kind === "on_off" || kind === "onoff") {
    let cycleStart = 0;
    while (arrivals.length < safetyLimit && (horizon === null || cycleStart <= horizon)) {
      let cursor = cycleStart;
      while (burstRate > 0 && arrivals.length < safetyLimit) {
        cursor += rng.expovariate(burstRate);
        if (cursor >= cycleStart + onDuration) break;
        if (horizon !== null && cursor > horizon) break;
        arrivals.push(cursor);
      }
      cycleStart += onDuration + offDuration;
      if (onDuration + offDuration <= 1e-12) break;
    }
  } else {
    throw new Error(`unsupported arrival process: ${String(a.arrivalProcess)}`);
  }
  const rows = arrivals.map((arrival, index) => ({ requestId: `request-${index}`, arrivalS: arrival, promptTokens: null, answerTokens: null }));
  return rows.map((row, index) => rowToTrace(row, index, a, rng));
}

function normalizeRows(rows: TraceRequest[]): TraceRequest[] {
  // serving_sim.normalize_trace_rows: shift to zero origin, dedup ids, stable order.
  if (!rows.length) return [];
  const origin = Math.min(...rows.map((row) => row.arrivalS));
  const shifted = rows.map((row) => ({ ...row, arrivalS: Math.max(0, row.arrivalS - origin) }));
  const seen = new Map<string, number>();
  const indexed = shifted.map((row, index) => ({ row, index }));
  indexed.sort((x, y) => x.row.arrivalS - y.row.arrivalS || x.index - y.index);
  const unique: TraceRequest[] = [];
  for (const { row } of indexed) {
    const count = seen.get(row.requestId) ?? 0;
    seen.set(row.requestId, count + 1);
    unique.push({ ...row, requestId: count === 0 ? row.requestId : `${row.requestId}-${count + 1}` });
  }
  return unique;
}

// ---------------------------------------------------------------------- state

interface MutableRequest {
  trace: TraceRequest;
  promptTokens: number;
  answerTokens: number;
  arrivalS: number;
  prefixKey: string | null;
  prefixTokensRequested: number;
  status: string;
  firstTokenS: number | null;
  completionS: number | null;
  error: string | null;
  prefillReplica: string | null;
  decodeReplica: string | null;
  prefixCacheHit: boolean;
  prefixCacheRef: boolean;
  prefixCacheStored: boolean;
  prefixMissSeen: boolean;
  cachedPrefixTokens: number;
  preemptions: number;
  contextTokens: number;
  generatedTokens: number;
  prefillRemaining: number;
  prefillInitialized: boolean;
  privateBlocks: number;
  kvPeakBlocks: number;
  lastServiceTime: number;
  tokenTimes: number[];
}

interface PrefixEntry {
  key: string;
  prefixTokens: number;
  blocks: number;
  lastUsed: number;
  refs: number;
}

class EventCounter {
  maxEventTime = 0;
  kvEvictions = 0;
  prefixMissCandidates = 0; // requests with a prefix_cache_miss event
  speculativeAttempted = 0;
  speculativeAccepted = 0;
  note(timeS: number): void {
    if (timeS > this.maxEventTime) this.maxEventTime = timeS;
  }
}


class KVState {
  blockTokens: number;
  capacityBlocks: number;
  prefixCapacityBlocks: number;
  enabled: boolean;
  prefixMinTokens: number;
  evictionPolicy: string;
  usedBlocks = 0;
  prefixBlocks = 0;
  private activeBlocks = new Map<string, number>();
  private prefix = new Map<string, PrefixEntry>();
  private clock = 0;
  evictions = 0;
  samples: Array<[number, number]> = [[0, 0]];
  peakBlocks = 0;
  counter: EventCounter;

  constructor(a: ResolvedAssumptions, counter: EventCounter) {
    this.counter = counter;
    this.blockTokens = Math.max(1, a.kvBlockTokens);
    const configured = a.kvCapacityBlocks ?? ceilDiv(Math.max(0, a.kvCapacityTokens), this.blockTokens);
    this.capacityBlocks = Math.max(1, configured);
    const prefixConfigured = a.prefixCacheCapacityBlocks ?? Math.max(0, Math.floor(this.capacityBlocks / 4));
    this.prefixCapacityBlocks = Math.max(0, Math.min(this.capacityBlocks, prefixConfigured));
    this.enabled = a.prefixCacheEnabled && this.prefixCapacityBlocks > 0;
    this.prefixMinTokens = Math.max(1, a.prefixCacheMinTokens);
    this.evictionPolicy = a.evictionPolicy;
  }

  get utilization(): number {
    return this.capacityBlocks ? this.usedBlocks / this.capacityBlocks : 0;
  }

  sample(timeS: number): void {
    this.samples.push([Math.max(0, timeS), this.usedBlocks]);
    this.peakBlocks = Math.max(this.peakBlocks, this.usedBlocks);
  }

  private touch(entry: PrefixEntry): void {
    this.clock += 1;
    entry.lastUsed = this.clock;
  }

  lookup(request: MutableRequest, timeS: number): number {
    if (!this.enabled || !request.prefixKey || request.prefixTokensRequested < this.prefixMinTokens) return 0;
    const entry = this.prefix.get(request.prefixKey);
    if (entry === undefined || entry.prefixTokens < request.prefixTokensRequested) {
      request.prefixMissSeen = true;
      this.counter.prefixMissCandidates += 1;
      this.counter.note(timeS);
      return 0;
    }
    this.touch(entry);
    entry.refs += 1;
    request.prefixCacheRef = true;
    request.prefixCacheHit = true;
    request.cachedPrefixTokens = Math.min(entry.prefixTokens, request.prefixTokensRequested);
    return request.cachedPrefixTokens;
  }

  releasePrefix(request: MutableRequest): void {
    if (request.prefixKey && request.prefixCacheRef) {
      const entry = this.prefix.get(request.prefixKey);
      if (entry && entry.refs > 0) entry.refs -= 1;
      request.prefixCacheRef = false;
    }
  }

  evictInactive(requiredExtra: number): number {
    const needed = Math.max(0, requiredExtra);
    while (this.usedBlocks + needed > this.capacityBlocks) {
      const candidates = [...this.prefix.values()].filter((entry) => entry.refs === 0);
      if (!candidates.length) break;
      let victim = candidates[0]!;
      for (const entry of candidates) {
        if (this.evictionPolicy === "largest") {
          // max by (blocks, -last_used, key)
          if (entry.blocks > victim.blocks) victim = entry;
          else if (entry.blocks === victim.blocks) {
            if (-entry.lastUsed > -victim.lastUsed) victim = entry;
            else if (entry.lastUsed === victim.lastUsed && entry.key > victim.key) victim = entry;
          }
        } else {
          // min by (last_used, key)
          if (entry.lastUsed < victim.lastUsed || (entry.lastUsed === victim.lastUsed && entry.key < victim.key)) victim = entry;
        }
      }
      this.prefix.delete(victim.key);
      this.prefixBlocks -= victim.blocks;
      this.usedBlocks -= victim.blocks;
      this.evictions += 1;
      this.counter.kvEvictions += 1;
    }
    return Math.max(0, this.capacityBlocks - this.usedBlocks);
  }

  addActiveBlocks(request: MutableRequest, desired: number): number {
    desired = Math.max(0, desired);
    const additional = Math.max(0, desired - request.privateBlocks);
    request.privateBlocks = desired;
    this.activeBlocks.set(request.trace.requestId, desired);
    this.usedBlocks += additional;
    this.peakBlocks = Math.max(this.peakBlocks, this.usedBlocks);
    return additional;
  }

  freeActive(request: MutableRequest): void {
    const old = this.activeBlocks.get(request.trace.requestId) ?? request.privateBlocks;
    this.activeBlocks.delete(request.trace.requestId);
    this.usedBlocks = Math.max(0, this.usedBlocks - old);
    request.privateBlocks = 0;
  }

  promotePrefix(request: MutableRequest): boolean {
    if (!this.enabled || !request.prefixKey || request.prefixTokensRequested < this.prefixMinTokens || request.generatedTokens > 0 || request.prefixCacheHit) {
      return false;
    }
    const prefixTokens = Math.min(request.prefixTokensRequested, request.contextTokens);
    if (prefixTokens <= 0) return false;
    const blocks = ceilDiv(prefixTokens, this.blockTokens);
    if (blocks > this.prefixCapacityBlocks) return false;
    const existing = this.prefix.get(request.prefixKey!);
    if (existing && existing.prefixTokens >= prefixTokens) {
      existing.refs += 1;
      request.prefixCacheRef = true;
      request.prefixCacheStored = true;
      request.prefixCacheHit = true;
      request.cachedPrefixTokens = existing.prefixTokens;
      const shared = Math.min(existing.blocks, request.privateBlocks);
      if (shared) {
        request.privateBlocks -= shared;
        this.activeBlocks.set(request.trace.requestId, request.privateBlocks);
        this.usedBlocks = Math.max(0, this.usedBlocks - shared);
      }
      return true;
    }
    while (this.prefixBlocks + blocks > this.prefixCapacityBlocks) {
      const candidates = [...this.prefix.values()].filter((entry) => entry.refs === 0);
      if (!candidates.length) return false;
      let victim = candidates[0]!;
      for (const entry of candidates) {
        if (entry.lastUsed < victim.lastUsed || (entry.lastUsed === victim.lastUsed && entry.key < victim.key)) victim = entry;
      }
      this.prefix.delete(victim.key);
      this.prefixBlocks -= victim.blocks;
      this.usedBlocks -= victim.blocks;
      this.evictions += 1;
      this.counter.kvEvictions += 1;
    }
    if (request.privateBlocks < blocks) return false;
    request.privateBlocks -= blocks;
    this.activeBlocks.set(request.trace.requestId, request.privateBlocks);
    this.prefix.set(request.prefixKey!, { key: request.prefixKey!, prefixTokens, blocks, lastUsed: this.clock + 1, refs: 1 });
    this.clock += 1;
    this.prefixBlocks += blocks;
    request.prefixCacheRef = true;
    request.prefixCacheStored = true;
    request.cachedPrefixTokens = prefixTokens;
    return true;
  }
}

// --------------------------------------------------------------------- worker

type DebugHook = (timeS: number, event: string, requestId: string, details: Record<string, number | string>) => void;
export let debugHook: DebugHook | null = null;
export function setDebugHook(hook: DebugHook | null): void { debugHook = hook; }

class Worker {
  private requests: MutableRequest[];
  private a: ResolvedAssumptions;
  private counter: EventCounter;
  private rng: PyRandom;
  readonly replicaId: string;
  readonly kv: KVState;
  now = 0;
  private cursor = 0;
  private waiting: MutableRequest[] = [];
  private active: MutableRequest[] = [];
  private decodeStepsSincePrefill = 0;
  endTime = 0;

  constructor(requests: MutableRequest[], a: ResolvedAssumptions, counter: EventCounter, rng: PyRandom, replicaId: string) {
    // Python: sorted(requests, key=lambda item: (item.arrival_s, item.trace.request_id))
    this.requests = [...requests].sort((x, y) => {
      if (x.arrivalS !== y.arrivalS) return x.arrivalS - y.arrivalS;
      if (x.trace.requestId < y.trace.requestId) return -1;
      if (x.trace.requestId > y.trace.requestId) return 1;
      return 0;
    });
    this.a = a;
    this.counter = counter;
    this.rng = rng;
    this.replicaId = replicaId;
    this.kv = new KVState(a, counter);
  }

  private get tpFactor(): number {
    return Math.max(1, this.a.tensorParallel) * Math.max(0.01, this.a.tpEfficiency);
  }

  private addReady(): void {
    while (this.cursor < this.requests.length && this.requests[this.cursor]!.arrivalS <= this.now + 1e-12) {
      const request = this.requests[this.cursor]!;
      this.cursor += 1;
      if (request.status === "completed" || request.status === "failed") continue;
      request.status = "waiting_prefill";
      this.waiting.push(request);
      this.counter.note(this.now);
      debugHook?.(this.now, "queued", request.trace.requestId, { queue_wait_s: Math.max(0, this.now - request.arrivalS) });
    }
  }

  private pending(): boolean {
    return this.cursor < this.requests.length || this.waiting.length > 0 || this.active.length > 0;
  }

  private advanceIfIdle(): boolean {
    if (this.waiting.length || this.active.length) return false;
    if (this.cursor >= this.requests.length) return false;
    this.now = Math.max(this.now, this.requests[this.cursor]!.arrivalS);
    this.addReady();
    return true;
  }

  private batchScale(batchSize: number, gain: number): number {
    return Math.max(1, 1 + Math.max(0, gain) * Math.max(0, batchSize - 1));
  }

  private prefillDuration(totalTokens: number, batchSize: number): number {
    const rate = Math.max(1e-12, this.a.prefillTokensPerS) * this.tpFactor;
    return Math.max(0, this.a.prefillBatchOverheadS) + Math.max(0, totalTokens) / (rate * this.batchScale(batchSize, this.a.prefillBatchGain));
  }

  private decodeDuration(work: number, batchSize: number): number {
    const rate = Math.max(1e-12, this.a.decodeTokensPerS) * this.tpFactor;
    return Math.max(0, this.a.decodeBatchOverheadS) + Math.max(0, work) / (rate * this.batchScale(batchSize, this.a.decodeBatchGain));
  }

  private preempt(request: MutableRequest, timeS: number): void {
    if (!this.active.includes(request)) return;
    this.active = this.active.filter((item) => item !== request);
    this.kv.freeActive(request);
    request.preemptions += 1;
    request.status = "waiting_prefill";
    request.prefillInitialized = true;
    const recomputeTokens = Math.max(0, request.contextTokens - request.cachedPrefixTokens);
    request.contextTokens = request.cachedPrefixTokens;
    request.prefillRemaining = recomputeTokens;
    request.lastServiceTime = timeS;
    this.waiting.push(request);
    this.counter.note(timeS);
  }

  private ensureCapacity(request: MutableRequest, desiredPrivateBlocks: number, timeS: number, protectedIds: Set<string>): boolean {
    desiredPrivateBlocks = Math.max(0, desiredPrivateBlocks);
    const additional = Math.max(0, desiredPrivateBlocks - request.privateBlocks);
    if (this.kv.usedBlocks + additional <= this.kv.capacityBlocks) {
      this.kv.addActiveBlocks(request, desiredPrivateBlocks);
      request.kvPeakBlocks = Math.max(request.kvPeakBlocks, this.kv.usedBlocks);
      this.kv.sample(timeS);
      return true;
    }
    this.kv.evictInactive(additional);
    while (this.kv.usedBlocks + additional > this.kv.capacityBlocks) {
      if (!this.a.preemptionEnabled) return false;
      const victims = this.active.filter(
        (item) => !protectedIds.has(item.trace.requestId) && item.status === "decoding" && item.privateBlocks > 0,
      );
      if (!victims.length) return false;
      // min by (last_service_time, -private_blocks, request_id); first minimum wins
      let victim = victims[0]!;
      for (const item of victims.slice(1)) {
        if (
          item.lastServiceTime < victim.lastServiceTime ||
          (item.lastServiceTime === victim.lastServiceTime &&
            (-item.privateBlocks > -victim.privateBlocks ||
              (-item.privateBlocks === -victim.privateBlocks && item.trace.requestId < victim.trace.requestId)))
        ) {
          victim = item;
        }
      }
      this.preempt(victim, timeS);
      this.kv.evictInactive(additional);
    }
    this.kv.addActiveBlocks(request, desiredPrivateBlocks);
    request.kvPeakBlocks = Math.max(request.kvPeakBlocks, this.kv.usedBlocks);
    this.kv.sample(timeS);
    return true;
  }

  private preparePrefill(request: MutableRequest, timeS: number): void {
    if (request.prefillInitialized) return;
    const hit = this.kv.lookup(request, timeS);
    request.prefillInitialized = true;
    request.contextTokens = hit;
    request.prefillRemaining = Math.max(0, request.promptTokens - hit);
    request.cachedPrefixTokens = hit;
    this.counter.note(timeS);
  }

  private finishPrefill(request: MutableRequest, timeS: number): void {
    request.prefillRemaining = 0;
    request.lastServiceTime = timeS;
    if (request.generatedTokens === 0) this.kv.promotePrefix(request);
    if (request.answerTokens <= 0) {
      this.finish(request, timeS);
      return;
    }
    request.status = "decoding";
    this.active.push(request);
    this.counter.note(timeS);
  }

  private runPrefillBatch(): void {
    if (!this.waiting.length) return;
    const take = Math.max(1, this.a.maxBatchSize);
    const selected = this.waiting.slice(0, take);
    this.waiting = this.waiting.slice(selected.length);
    for (const request of selected) this.preparePrefill(request, this.now);

    let chunkLimit = this.a.chunkedPrefillTokens;
    if (chunkLimit <= 0) chunkLimit = Math.max(1, Math.max(1, ...selected.map((request) => request.prefillRemaining)));
    const chunks: Array<[MutableRequest, number]> = [];
    for (const request of selected) {
      if (request.status === "completed" || request.status === "failed") continue;
      const chunk = Math.min(Math.max(0, request.prefillRemaining), chunkLimit);
      if (chunk === 0) {
        this.finishPrefill(request, this.now);
        continue;
      }
      const desiredContext = request.contextTokens + chunk;
      const desiredPrivate = Math.max(0, desiredContext - request.cachedPrefixTokens);
      const desiredBlocks = ceilDiv(desiredPrivate, this.kv.blockTokens);
      const protectedIds = new Set<string>(selected.map((item) => item.trace.requestId));
      for (const [done] of chunks) protectedIds.add(done.trace.requestId);
      if (!this.ensureCapacity(request, desiredBlocks, this.now, protectedIds)) {
        request.status = "failed";
        request.error = "kv_capacity_exhausted";
        this.counter.note(this.now);
        continue;
      }
      request.contextTokens = desiredContext;
      request.prefillRemaining -= chunk;
      chunks.push([request, chunk]);
    }

    const batchSize = Math.max(1, chunks.length);
    const totalTokens = chunks.reduce((sum, [, chunk]) => sum + chunk, 0);
    const duration = this.prefillDuration(totalTokens, batchSize);
    const end = this.now + duration;
    debugHook?.(this.now, "prefill_batch_start", "batch", { batch_size: batchSize, total_batch_tokens: totalTokens });
    this.now = end;
    this.decodeStepsSincePrefill = 0;
    for (const [request] of chunks) {
      request.lastServiceTime = end;
      this.counter.note(end);
      if (request.prefillRemaining > 0) {
        request.status = "waiting_prefill";
        this.waiting.push(request);
      } else {
        this.finishPrefill(request, end);
      }
    }
    this.addReady();
  }

  private sampleDecodeWork(request: MutableRequest): [number, number, number, number] {
    const remaining = Math.max(0, request.answerTokens - request.generatedTokens);
    if (remaining <= 0) return [0, 0, 0, 0];
    if (!this.a.speculativeEnabled) return [1, 0, 0, 1.0];
    const attempted = Math.min(Math.max(1, this.a.speculativeDraftTokens), remaining);
    const probability = Math.min(1, Math.max(0, sampleDistribution(this.a.speculativeAcceptance, this.rng)));
    let accepted = 0;
    for (let i = 0; i < attempted; i++) {
      if (this.rng.random() < probability) accepted += 1;
      else break;
    }
    const emitted = Math.min(remaining, accepted + 1);
    const rejected = Math.max(0, attempted - accepted);
    const verificationWork = 1 + Math.max(0, emitted - 1) * Math.max(0, this.a.speculativeVerificationFactor);
    const draftWork = attempted * ((1 / Math.max(1e-12, this.a.draftTokensPerS)) * Math.max(1e-12, this.a.decodeTokensPerS));
    return [emitted, accepted, rejected, verificationWork + draftWork];
  }

  private finish(request: MutableRequest, timeS: number): void {
    if (request.status === "completed") return;
    request.status = "completed";
    request.completionS = timeS;
    this.kv.freeActive(request);
    this.kv.releasePrefix(request);
    this.active = this.active.filter((item) => item !== request);
    this.kv.sample(timeS);
    this.counter.note(timeS);
  }

  private runDecodeBatch(): void {
    if (!this.active.length) return;
    const active = this.active.filter((item) => item.status === "decoding");
    if (!active.length) {
      this.active = [];
      return;
    }
    const batch = active.slice(0, Math.max(1, this.a.maxBatchSize));
    const works: Array<[MutableRequest, number, number, number, number]> = [];
    const protectedIds = new Set(batch.map((item) => item.trace.requestId));
    for (const request of batch) {
      const [emitted, accepted, rejected, work] = this.sampleDecodeWork(request);
      if (emitted <= 0) continue;
      const desiredContext = request.contextTokens + emitted;
      const desiredPrivate = Math.max(0, desiredContext - request.cachedPrefixTokens);
      const desiredBlocks = ceilDiv(desiredPrivate, this.kv.blockTokens);
      if (!this.ensureCapacity(request, desiredBlocks, this.now, protectedIds)) {
        request.status = "failed";
        request.error = "kv_capacity_exhausted";
        this.kv.freeActive(request);
        this.active = this.active.filter((item) => item !== request);
        this.counter.note(this.now);
        continue;
      }
      works.push([request, emitted, accepted, rejected, work]);
    }
    if (!works.length) {
      this.now += Math.max(0, this.a.decodeBatchOverheadS);
      this.addReady();
      return;
    }
    const batchSize = works.length;
    const totalWork = works.reduce((sum, [, , , , work]) => sum + work, 0);
    debugHook?.(this.now, "decode_batch_start", "batch", { batch_size: batchSize, total_work: totalWork });
    const duration = this.decodeDuration(totalWork, batchSize);
    const start = this.now;
    const end = start + duration;
    for (const [request, emitted, accepted, rejected] of works) {
      if (this.a.speculativeEnabled) {
        this.counter.speculativeAttempted += accepted + rejected;
        this.counter.speculativeAccepted += accepted;
      }
      for (let index = 0; index < emitted; index++) {
        const tokenTime = start + duration * ((index + 1) / emitted);
        request.generatedTokens += 1;
        request.contextTokens += 1;
        request.tokenTimes.push(tokenTime);
        if (request.firstTokenS === null) request.firstTokenS = tokenTime;
      }
      request.lastServiceTime = end;
      if (request.generatedTokens >= request.answerTokens) this.finish(request, end);
    }
    this.counter.note(end);
    this.now = end;
    this.decodeStepsSincePrefill += 1;
    this.addReady();
  }

  /** Python mixed-mode run loop, mirrored exactly. */
  run(): void {
    while (this.pending()) {
      this.addReady();
      if (!this.waiting.length && !this.active.length && this.cursor < this.requests.length) {
        this.advanceIfIdle();
        continue;
      }
      if (
        this.waiting.length &&
        (!this.active.length || !this.a.continuousBatching || this.decodeStepsSincePrefill >= Math.max(1, this.a.maxDecodeStepsBeforePrefill))
      ) {
        this.runPrefillBatch();
      } else if (this.active.length) {
        this.runDecodeBatch();
      } else if (this.waiting.length) {
        this.runPrefillBatch();
      } else {
        this.advanceIfIdle();
      }
    }
    this.endTime = this.now;
  }
}

// ------------------------------------------------------------------ pipeline

export interface ServingSimEvent {
  timeS: number;
  requestId: string;
  event: string;
  phase: string;
  replicaId: string | null;
  details: Record<string, unknown>;
  label: typeof SIMULATED_LABEL;
  sequence: number;
}

export interface ServingSimRequestResult {
  requestId: string;
  arrivalS: number;
  promptTokens: number;
  answerTokens: number;
  status: string;
  firstTokenS: number | null;
  completionS: number | null;
  ttftS: number | null;
  e2eS: number | null;
  interTokenLatencyS: number[];
  averageItlS: number | null;
  maxItlS: number | null;
  outputTokens: number;
  prefillReplica: string | null;
  decodeReplica: string | null;
  prefixCacheHit: boolean;
  prefixTokens: number;
  preemptions: number;
  kvPeakBlocks: number;
  error: string | null;
  label: typeof SIMULATED_LABEL;
}

export interface ServingSimDistributionSummary {
  name: string;
  values: number[];
  p50: number | null;
  p90: number | null;
  p99: number | null;
  label: typeof SIMULATED_LABEL;
}

export interface ServingSimResult {
  seed: number;
  simulated: true;
  requests: ServingSimRequestResult[];
  distributions: Record<string, ServingSimDistributionSummary>;
  metrics: Record<string, number | null>;
  replicaMetrics: Record<string, Record<string, number | null>>;
  assumptions: Record<string, unknown>;
  notes: string[];
}

function makeRequestStates(trace: TraceRequest[]): MutableRequest[] {
  return trace.map((item) => ({
    trace: item,
    promptTokens: Math.max(1, item.promptTokens || 1),
    answerTokens: Math.max(0, item.answerTokens || 0),
    arrivalS: Math.max(0, finite(item.arrivalS, 0)),
    prefixKey: item.prefixId,
    prefixTokensRequested: Math.max(0, item.prefixTokens || 0),
    status: "new",
    firstTokenS: null,
    completionS: null,
    error: null,
    prefillReplica: null,
    decodeReplica: null,
    prefixCacheHit: false,
    prefixCacheRef: false,
    prefixCacheStored: false,
    prefixMissSeen: false,
    cachedPrefixTokens: 0,
    preemptions: 0,
    contextTokens: 0,
    generatedTokens: 0,
    prefillRemaining: 0,
    prefillInitialized: false,
    privateBlocks: 0,
    kvPeakBlocks: 0,
    lastServiceTime: 0,
    tokenTimes: [],
  }));
}

/** serving_sim._assign_requests: least-loaded pool by prompt/decode work estimate. */
function assignRequests(requests: MutableRequest[], count: number, a: ResolvedAssumptions): MutableRequest[][] {
  const pools: MutableRequest[][] = Array.from({ length: Math.max(1, count) }, () => []);
  const loads = pools.map(() => 0);
  const prefillRate = Math.max(1e-12, a.prefillTokensPerS);
  const decodeRate = Math.max(1e-12, a.decodeTokensPerS);
  const ordered = [...requests].sort((x, y) => {
    if (x.arrivalS !== y.arrivalS) return x.arrivalS - y.arrivalS;
    if (x.trace.requestId < y.trace.requestId) return -1;
    if (x.trace.requestId > y.trace.requestId) return 1;
    return 0;
  });
  for (const request of ordered) {
    const estimate = request.promptTokens / prefillRate + request.answerTokens / decodeRate;
    let selected = 0;
    for (let index = 1; index < loads.length; index++) {
      if (loads[index]! < loads[selected]!) selected = index;
    }
    pools[selected]!.push(request);
    loads[selected]! += Math.max(0.000001, estimate);
  }
  return pools;
}

/** serving_sim._percentile: linear interpolation on the (n-1) rank. */
export function simPercentile(values: readonly number[], percentile: number): number | null {
  if (!values.length) return null;
  const ordered = [...values].sort((a, b) => a - b);
  if (ordered.length === 1) return ordered[0]!;
  const rank = ((ordered.length - 1) * percentile) / 100;
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  if (lower === upper) return ordered[lower]!;
  const fraction = rank - lower;
  return ordered[lower]! + (ordered[upper]! - ordered[lower]!) * fraction;
}

function summary(name: string, values: readonly number[]): ServingSimDistributionSummary {
  const data = values.filter((value) => Number.isFinite(value));
  return { name, values: [...data], p50: simPercentile(data, 50), p90: simPercentile(data, 90), p99: simPercentile(data, 99), label: SIMULATED_LABEL };
}

function timeWeightedUtilization(workers: readonly Worker[]): { average: number; peak: number; samples: number[] } {
  const averages: number[] = [];
  const peaks: number[] = [];
  const samples: number[] = [];
  for (const worker of workers) {
    const kv = worker.kv;
    if (!kv.samples.length) {
      averages.push(0);
      peaks.push(0);
      continue;
    }
    const ordered = [...kv.samples].sort((a, b) => a[0] - b[0]);
    const localSamples = ordered.map(([, used]) => used / kv.capacityBlocks);
    const end = Math.max(worker.endTime, ordered[ordered.length - 1]![0]);
    let integral = 0;
    let previousTime = 0;
    let previousUsed = ordered[0]![1];
    for (const [timeS, used] of ordered) {
      if (timeS > previousTime) integral += (timeS - previousTime) * (previousUsed / kv.capacityBlocks);
      previousTime = Math.max(previousTime, timeS);
      previousUsed = used;
      samples.push(used / kv.capacityBlocks);
    }
    if (end > previousTime) integral += (end - previousTime) * (previousUsed / kv.capacityBlocks);
    averages.push(end > 0 ? integral / end : previousUsed / kv.capacityBlocks);
    let localPeak = 0;
    for (const sample of localSamples) if (sample > localPeak) localPeak = sample;
    peaks.push(localPeak);
  }
  return {
    average: averages.length ? averages.reduce((a, b) => a + b, 0) / averages.length : 0,
    peak: peaks.length ? Math.max(...peaks) : 0,
    samples,
  };
}

function requestResult(request: MutableRequest): ServingSimRequestResult {
  const itl: number[] = [];
  for (let i = 1; i < request.tokenTimes.length; i++) {
    itl.push(Math.max(0, request.tokenTimes[i]! - request.tokenTimes[i - 1]!));
  }
  const ttft = request.firstTokenS === null ? null : Math.max(0, request.firstTokenS - request.trace.arrivalS);
  const e2e = request.completionS === null ? null : Math.max(0, request.completionS - request.trace.arrivalS);
  return {
    requestId: request.trace.requestId,
    arrivalS: request.trace.arrivalS,
    promptTokens: request.promptTokens,
    answerTokens: request.answerTokens,
    status: request.status,
    firstTokenS: request.firstTokenS,
    completionS: request.completionS,
    ttftS: ttft,
    e2eS: e2e,
    interTokenLatencyS: itl,
    averageItlS: itl.length ? itl.reduce((a, b) => a + b, 0) / itl.length : null,
    maxItlS: itl.length ? Math.max(...itl) : null,
    outputTokens: request.generatedTokens,
    prefillReplica: request.prefillReplica,
    decodeReplica: request.decodeReplica,
    prefixCacheHit: request.prefixCacheHit,
    prefixTokens: request.prefixTokensRequested,
    preemptions: request.preemptions,
    kvPeakBlocks: request.kvPeakBlocks,
    error: request.error,
    label: SIMULATED_LABEL,
  };
}

export interface ServingSimOptions {
  /** Explicit trace rows (replay mode). Lengths are sampled for missing fields with the seed. */
  traceRows?: ServingSimInputRow[] | null;
}

/**
 * simulate_serving: deterministic request-level simulation of continuous
 * batching, chunked prefill, paged KV with LRU prefix cache and preemption.
 * Same seed -> same numbers, byte-identical to the Python reference on the
 * supported (non-disaggregated) path.
 */
export function simulateServing(
  assumptions?: ServingSimAssumptions | Record<string, unknown> | null,
  options: ServingSimOptions = {},
): ServingSimResult {
  const a = resolveAssumptions(assumptions);
  if (Boolean((assumptions as Record<string, unknown> | null)?.disaggregate)) {
    throw new Error("disaggregated prefill/decode simulation stays server-only (nomo_planner.serving_sim)");
  }
  const actualSeed = Math.trunc(finite(a.seed, 0));
  const counter = new EventCounter();

  // Python rng contract: the replay path samples row lengths from a fresh
  // Random(seed) inside normalize_trace_rows while workers consume a separate
  // Random(seed); the generated path draws arrivals, lengths, and worker
  // randomness from ONE stream in order.
  let trace: TraceRequest[];
  let simRng: PyRandom;
  if (options.traceRows != null) {
    const rowRng = new PyRandom(actualSeed);
    trace = normalizeRows(options.traceRows.map((row, index) => rowToTrace(row, index, a, rowRng)));
    simRng = new PyRandom(actualSeed);
  } else {
    simRng = new PyRandom(actualSeed);
    trace = generatedTrace(a, simRng);
  }

  const requests = makeRequestStates(trace);
  const pools = assignRequests(requests, a.replicas, a);
  const workers: Worker[] = [];
  pools.forEach((pool, index) => {
    const replicaId = `replica-${index}`;
    for (const request of pool) {
      request.prefillReplica = replicaId;
      request.decodeReplica = replicaId;
    }
    workers.push(new Worker(pool, a, counter, simRng, replicaId));
  });
  for (const worker of workers) worker.run();

  const results = [...requests]
    .sort((x, y) => {
      if (x.trace.arrivalS !== y.trace.arrivalS) return x.trace.arrivalS - y.trace.arrivalS;
      if (x.trace.requestId < y.trace.requestId) return -1;
      if (x.trace.requestId > y.trace.requestId) return 1;
      return 0;
    })
    .map(requestResult);

  const ttftValues = results.flatMap((request) => (request.ttftS === null ? [] : [request.ttftS]));
  const itlValues = results.flatMap((request) => request.interTokenLatencyS);
  const e2eValues = results.flatMap((request) => (request.e2eS === null ? [] : [request.e2eS]));
  const completed = results.filter((request) => request.status === "completed");
  const elapsedStart = results.length ? Math.min(...results.map((request) => request.arrivalS)) : 0;
  const elapsedEnd = Math.max(
    results.length ? Math.max(...results.map((request) => request.completionS ?? request.firstTokenS ?? request.arrivalS)) : elapsedStart,
    counter.maxEventTime,
  );
  const elapsed = Math.max(0, elapsedEnd - elapsedStart);

  const sloGood: ServingSimRequestResult[] = [];
  for (const request of completed) {
    let good = true;
    if (a.sloTtftS !== null && a.sloTtftS !== undefined) good = good && request.ttftS !== null && request.ttftS <= finite(a.sloTtftS, 0);
    if (a.sloItlS !== null && a.sloItlS !== undefined && request.interTokenLatencyS.length) {
      good = good && request.maxItlS !== null && request.maxItlS <= finite(a.sloItlS, 0);
    }
    if (a.sloE2eS !== null && a.sloE2eS !== undefined) good = good && request.e2eS !== null && request.e2eS <= finite(a.sloE2eS, 0);
    if (good) sloGood.push(request);
  }

  const utilization = timeWeightedUtilization(workers);
  const prefixCandidates = requests.filter((request) => request.prefixTokensRequested > 0 || request.prefixCacheHit || request.prefixMissSeen).length;
  const prefixHits = results.filter((request) => request.prefixCacheHit).length;
  const preemptions = results.reduce((sum, request) => sum + request.preemptions, 0);
  const outputTokens = completed.reduce((sum, request) => sum + request.outputTokens, 0);

  const metrics: Record<string, number | null> = {
    requests: results.length,
    completed_requests: completed.length,
    failed_requests: results.filter((request) => request.status === "failed").length,
    wall_time_s: elapsed,
    throughput_requests_per_s: elapsed > 0 ? completed.length / elapsed : 0,
    throughput_output_tokens_per_s: elapsed > 0 ? outputTokens / elapsed : 0,
    output_tokens: outputTokens,
    ttft_p50_s: simPercentile(ttftValues, 50),
    ttft_p90_s: simPercentile(ttftValues, 90),
    ttft_p99_s: simPercentile(ttftValues, 99),
    itl_p50_s: simPercentile(itlValues, 50),
    itl_p90_s: simPercentile(itlValues, 90),
    itl_p99_s: simPercentile(itlValues, 99),
    e2e_p50_s: simPercentile(e2eValues, 50),
    e2e_p90_s: simPercentile(e2eValues, 90),
    e2e_p99_s: simPercentile(e2eValues, 99),
    slo_goodput_requests: sloGood.length,
    slo_goodput_ratio: results.length ? sloGood.length / results.length : 0,
    slo_goodput_requests_per_s: elapsed > 0 ? sloGood.length / elapsed : 0,
    kv_utilization_avg: utilization.average,
    kv_utilization_peak: utilization.peak,
    kv_utilization_p50: simPercentile(utilization.samples, 50),
    prefix_cache_hits: prefixHits,
    prefix_cache_hit_rate: prefixCandidates ? prefixHits / prefixCandidates : 0,
    prefix_cache_evictions: counter.kvEvictions,
    preemptions,
    speculative_attempted_tokens: counter.speculativeAttempted,
    speculative_accepted_tokens: counter.speculativeAccepted,
    speculative_acceptance_rate: counter.speculativeAttempted ? counter.speculativeAccepted / counter.speculativeAttempted : null,
    tensor_parallel: a.tensorParallel,
    replicas: a.replicas,
  };

  const replicaMetrics: Record<string, Record<string, number | null>> = {};
  for (const worker of workers) {
    const workerRequests = results.filter((request) => request.prefillReplica === worker.replicaId || request.decodeReplica === worker.replicaId);
    const workerCompleted = workerRequests.filter((request) => request.status === "completed");
    const workerUtil = timeWeightedUtilization([worker]);
    replicaMetrics[worker.replicaId] = {
      requests: workerRequests.length,
      completed_requests: workerCompleted.length,
      capacity_blocks: worker.kv.capacityBlocks,
      kv_utilization_avg: workerUtil.average,
      kv_utilization_peak: workerUtil.peak,
      kv_evictions: worker.kv.evictions,
      end_time_s: worker.endTime,
      label: SIMULATED_LABEL as unknown as number,
    };
  }

  const distributions: Record<string, ServingSimDistributionSummary> = {
    ttft_s: summary("ttft_s", ttftValues),
    inter_token_latency_s: summary("inter_token_latency_s", itlValues),
    e2e_s: summary("e2e_s", e2eValues),
    prompt_tokens: summary("prompt_tokens", results.map((request) => request.promptTokens)),
    answer_tokens: summary("answer_tokens", results.map((request) => request.answerTokens)),
    kv_utilization: summary("kv_utilization", utilization.samples),
  };

  return {
    seed: actualSeed,
    simulated: true,
    requests: results,
    distributions,
    metrics,
    replicaMetrics,
    assumptions: assumptionsForExport(a, actualSeed),
    notes: [
      "All latency, throughput, goodput, cache, and utilization values are simulated quantities.",
      "Prefill/decode rates, batching gains, KV capacity, cache policy, and SLO thresholds are explicit assumptions.",
      "Browser port of nomo_planner.serving_sim (non-disaggregated); parity-checked against the Python reference per seed.",
    ],
  };
}

function assumptionsForExport(a: ResolvedAssumptions, seed: number): Record<string, unknown> {
  return {
    label: "assumption",
    simulator: "nomo_lab browser port of nomo_planner.serving_sim",
    seed,
    arrival_process: a.arrivalProcess,
    request_count: a.requestCount,
    duration_s: a.durationS,
    arrival_rate_per_s: a.arrivalRatePerS,
    burst_rate_per_s: a.burstRatePerS,
    burst_duration_s: a.burstDurationS,
    quiet_duration_s: a.quietDurationS,
    prompt_distribution: a.promptDistribution,
    answer_distribution: a.answerDistribution,
    replicas: a.replicas,
    tensor_parallel: a.tensorParallel,
    tp_efficiency: a.tpEfficiency,
    disaggregate: false,
    prefill_tokens_per_s: a.prefillTokensPerS,
    decode_tokens_per_s: a.decodeTokensPerS,
    prefill_batch_overhead_s: a.prefillBatchOverheadS,
    decode_batch_overhead_s: a.decodeBatchOverheadS,
    prefill_batch_gain: a.prefillBatchGain,
    decode_batch_gain: a.decodeBatchGain,
    max_batch_size: a.maxBatchSize,
    chunked_prefill_tokens: a.chunkedPrefillTokens,
    max_decode_steps_before_prefill: a.maxDecodeStepsBeforePrefill,
    continuous_batching: a.continuousBatching,
    kv_block_tokens: a.kvBlockTokens,
    kv_capacity_blocks: a.kvCapacityBlocks,
    kv_capacity_tokens: a.kvCapacityTokens,
    eviction_policy: a.evictionPolicy,
    preemption_enabled: a.preemptionEnabled,
    prefix_cache_enabled: a.prefixCacheEnabled,
    prefix_cache_capacity_blocks: a.prefixCacheCapacityBlocks,
    prefix_cache_min_tokens: a.prefixCacheMinTokens,
    speculative_enabled: a.speculativeEnabled,
    slo_ttft_s: a.sloTtftS,
    slo_itl_s: a.sloItlS,
    slo_e2e_s: a.sloE2eS,
  };
}

import type { JsonValue } from "./contracts";
import { makeProvenance, normalizeProvenance, Provenance, type Provenance as ProvenanceType } from "./provenance";

export interface ConstantDistribution {
  kind: "constant";
  value: number;
  provenance?: ProvenanceType;
}

export interface UniformDistribution {
  kind: "uniform";
  min: number;
  max: number;
  provenance?: ProvenanceType;
}

export interface NormalDistribution {
  kind: "normal";
  mean: number;
  stddev: number;
  provenance?: ProvenanceType;
}

export interface LogNormalDistribution {
  kind: "lognormal";
  logMean: number;
  logStddev: number;
  provenance?: ProvenanceType;
}

export interface EmpiricalDistribution {
  kind: "empirical";
  values: readonly number[];
  weights?: readonly number[];
  provenance?: ProvenanceType;
}

export interface CategoricalDistribution {
  kind: "categorical";
  values: readonly string[];
  probabilities: readonly number[];
  provenance?: ProvenanceType;
}

export type NumericDistribution =
  | ConstantDistribution
  | UniformDistribution
  | NormalDistribution
  | LogNormalDistribution
  | EmpiricalDistribution;

/** Browser-local numeric/categorical contract.  The Python product Distribution
 * class is exported below under the canonical ``Distribution`` name. */
export type BrowserDistribution = NumericDistribution | CategoricalDistribution;
export type DistributionSample = number | string;

export interface DistributionSummary {
  count: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  p05: number;
  p95: number;
  provenance: ProvenanceType;
}

export interface RandomSource {
  next(): number;
  nextUint32(): number;
  normal(): number;
}

/**
 * Small, explicit PRNG used by the Worker and Node golden runner.  It is not
 * intended for cryptography; its value is that a seed has identical draws in
 * browser and server-side JavaScript without relying on Math.random().
 */
export class DeterministicRng implements RandomSource {
  private state: number;

  public constructor(seed: number | string = 0) {
    this.state = seedToUint32(seed) || 0x6d2b79f5;
  }

  public nextUint32(): number {
    let x = this.state >>> 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }

  public next(): number {
    return this.nextUint32() / 0x1_0000_0000;
  }

  public normal(): number {
    const u1 = Math.max(this.next(), Number.MIN_VALUE);
    const u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  public fork(label: string): DeterministicRng {
    return new DeterministicRng(`${this.state}:${label}`);
  }
}

export function seedToUint32(seed: number | string): number {
  if (typeof seed === "number") {
    if (!Number.isFinite(seed)) throw new Error("seed must be finite");
    let value = Math.trunc(seed) >>> 0;
    value ^= value >>> 16;
    value = Math.imul(value, 0x45d9f3b) >>> 0;
    value ^= value >>> 16;
    return value >>> 0;
  }
  let hash = 2166136261;
  for (const character of seed) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function assertFinite(value: number, label: string): void {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`);
}

function validateNumericDistribution(distribution: NumericDistribution): void {
  switch (distribution.kind) {
    case "constant":
      assertFinite(distribution.value, "constant value");
      return;
    case "uniform":
      assertFinite(distribution.min, "uniform min");
      assertFinite(distribution.max, "uniform max");
      if (distribution.max < distribution.min) throw new Error("uniform max must be >= min");
      return;
    case "normal":
      assertFinite(distribution.mean, "normal mean");
      assertFinite(distribution.stddev, "normal stddev");
      if (distribution.stddev < 0) throw new Error("normal stddev must be >= 0");
      return;
    case "lognormal":
      assertFinite(distribution.logMean, "lognormal logMean");
      assertFinite(distribution.logStddev, "lognormal logStddev");
      if (distribution.logStddev < 0) throw new Error("lognormal logStddev must be >= 0");
      return;
    case "empirical":
      if (!distribution.values.length) throw new Error("empirical distribution needs at least one value");
      distribution.values.forEach((value) => assertFinite(value, "empirical value"));
      if (distribution.weights !== undefined) {
        if (distribution.weights.length !== distribution.values.length) throw new Error("empirical weights must match values");
        distribution.weights.forEach((weight) => {
          assertFinite(weight, "empirical weight");
          if (weight < 0) throw new Error("empirical weights must be non-negative");
        });
        if (distribution.weights.every((weight) => weight === 0)) throw new Error("empirical weights must have positive mass");
      }
      return;
  }
}

function validateCategoricalDistribution(distribution: CategoricalDistribution): void {
  if (!distribution.values.length || distribution.values.length !== distribution.probabilities.length) {
    throw new Error("categorical values and probabilities must be non-empty and have equal length");
  }
  distribution.probabilities.forEach((probability) => {
    assertFinite(probability, "categorical probability");
    if (probability < 0) throw new Error("categorical probabilities must be non-negative");
  });
  if (distribution.probabilities.every((probability) => probability === 0)) throw new Error("categorical probabilities must have positive mass");
}

export function validateDistribution(distribution: BrowserDistribution): void {
  if (distribution.kind === "categorical") validateCategoricalDistribution(distribution);
  else validateNumericDistribution(distribution);
}

function weightedIndex(weights: readonly number[] | undefined, draw: number, count: number): number {
  if (!weights) return Math.min(count - 1, Math.floor(draw * count));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let cursor = draw * total;
  for (let index = 0; index < weights.length; index += 1) {
    cursor -= weights[index];
    if (cursor < 0) return index;
  }
  return weights.length - 1;
}

export function sampleDistribution(distribution: BrowserDistribution, rng: RandomSource): DistributionSample {
  validateDistribution(distribution);
  switch (distribution.kind) {
    case "constant":
      return distribution.value;
    case "uniform":
      return distribution.min + (distribution.max - distribution.min) * rng.next();
    case "normal":
      return distribution.mean + distribution.stddev * rng.normal();
    case "lognormal":
      return Math.exp(distribution.logMean + distribution.logStddev * rng.normal());
    case "empirical":
      return distribution.values[weightedIndex(distribution.weights, rng.next(), distribution.values.length)];
    case "categorical":
      return distribution.values[weightedIndex(distribution.probabilities, rng.next(), distribution.values.length)];
  }
}

export function sampleNumeric(distribution: NumericDistribution, rng: RandomSource): number {
  const value = sampleDistribution(distribution, rng);
  if (typeof value !== "number") throw new Error("expected a numeric distribution");
  return value;
}

export function sampleMany(distribution: NumericDistribution, count: number, seed: number | string = 0): number[] {
  if (!Number.isInteger(count) || count < 0) throw new Error("sample count must be a non-negative integer");
  const rng = new DeterministicRng(seed);
  return Array.from({ length: count }, () => sampleNumeric(distribution, rng));
}

export function quantile(values: readonly number[], probability: number): number {
  if (!values.length) throw new Error("quantile needs at least one value");
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error("quantile probability must be in [0, 1]");
  const ordered = [...values].sort((left, right) => left - right);
  const position = (ordered.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower];
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower);
}

export function summarizeSamples(values: readonly number[], provenance: ProvenanceType = makeProvenance("derived", "simulation samples")): DistributionSummary {
  if (!values.length) throw new Error("cannot summarize empty samples");
  values.forEach((value) => assertFinite(value, "sample"));
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    count: values.length,
    min: Math.min(...values),
    max: Math.max(...values),
    mean,
    median: quantile(values, 0.5),
    p05: quantile(values, 0.05),
    p95: quantile(values, 0.95),
    provenance,
  };
}

export function distributionToJSON(distribution: BrowserDistribution): JsonValue {
  validateDistribution(distribution);
  return JSON.parse(JSON.stringify(distribution)) as JsonValue;
}

export function parseDistribution(value: unknown): BrowserDistribution {
  if (typeof value !== "object" || value === null) throw new Error("distribution must be an object");
  const raw = value as Record<string, unknown>;
  const kind = String(raw.kind ?? raw.type);
  const provenance = raw.provenance === undefined ? undefined : normalizeProvenance(raw.provenance);
  switch (kind) {
    case "constant":
      return { kind, value: Number(raw.value), provenance };
    case "uniform":
      return { kind, min: Number(raw.min), max: Number(raw.max), provenance };
    case "normal":
      return { kind, mean: Number(raw.mean), stddev: Number(raw.stddev ?? raw.standardDeviation), provenance };
    case "lognormal":
    case "log-normal":
      return { kind: "lognormal", logMean: Number(raw.logMean ?? raw.log_mean), logStddev: Number(raw.logStddev ?? raw.log_stddev), provenance };
    case "empirical":
      return { kind, values: Array.isArray(raw.values) ? raw.values.map(Number) : [], weights: Array.isArray(raw.weights) ? raw.weights.map(Number) : undefined, provenance };
    case "categorical":
      return { kind, values: Array.isArray(raw.values) ? raw.values.map(String) : [], probabilities: Array.isArray(raw.probabilities) ? raw.probabilities.map(Number) : [], provenance };
    default:
      throw new Error(`unsupported distribution kind: ${kind}`);
  }
}

export interface Interval {
  low: number;
  median: number;
  high: number;
  level: number;
  method: string;
  samples: number | null;
}

/** Python product_simulations.Distribution parity contract. */
export class Distribution {
  public readonly kind: string;
  public readonly params: Readonly<Record<string, unknown>>;
  public readonly values: readonly number[];
  public readonly name: string;
  public readonly unit: string;
  public readonly provenance: ProvenanceType;

  public constructor(
    kind = "constant",
    params: Readonly<Record<string, unknown>> = {},
    values: readonly number[] = [],
    name = "",
    unit = "",
    provenance: ProvenanceType = new Provenance(),
  ) {
    const normalizedKind = kind.toLowerCase().replaceAll("-", "");
    if (!["constant", "empirical", "normal", "lognormal", "uniform", "triangular", "quantiles"].includes(normalizedKind)) {
      throw new Error(`unsupported distribution kind: ${kind}`);
    }
    if (normalizedKind === "empirical" && !values.length) throw new Error("empirical distributions require values");
    if (normalizedKind === "constant" && params.value === undefined && !values.length) throw new Error("constant distributions require a value");
    values.forEach((value) => {
      if (!Number.isFinite(value)) throw new Error("distribution values must be finite");
    });
    this.kind = normalizedKind;
    this.params = { ...params };
    this.values = [...values];
    this.name = name;
    this.unit = unit;
    this.provenance = provenance;
  }

  public static fromValue(value: unknown, options: { name?: string; unit?: string; provenance?: ProvenanceType } = {}): Distribution {
    if (value instanceof Distribution) return value;
    const name = options.name ?? "";
    const unit = options.unit ?? "";
    const provenance = options.provenance ?? new Provenance();
    if (Array.isArray(value)) return new Distribution("empirical", {}, value.map(Number), name, unit, provenance);
    if (typeof value === "object" && value !== null) {
      const raw = value as Record<string, unknown>;
      const kind = String(raw.kind ?? raw.type ?? "constant").toLowerCase();
      const rawValues = raw.values ?? raw.samples;
      const values = Array.isArray(rawValues) ? rawValues.map(Number) : [];
      const params: Record<string, unknown> = typeof raw.params === "object" && raw.params !== null ? { ...(raw.params as Record<string, unknown>) } : {};
      for (const key of ["value", "mean", "std", "sigma", "median", "geometric_sd", "low", "high", "mode", "points", "quantiles"]) {
        if (raw[key] !== undefined && params[key] === undefined) params[key] = raw[key];
      }
      const rawProvenance = raw.provenance === undefined ? provenance : normalizeProvenance(raw.provenance, provenance);
      return new Distribution(kind, params, values, name || String(raw.name ?? ""), unit || String(raw.unit ?? ""), rawProvenance);
    }
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`${name || "distribution value"} must be finite`);
    return new Distribution("constant", { value: number }, [], name, unit, provenance);
  }

  public sample(rng: RandomSource): number {
    const params = this.params;
    const number = (key: string, fallback = 0): number => Number(params[key] ?? fallback);
    switch (this.kind) {
      case "constant":
        return Number(params.value ?? this.values[0] ?? 0);
      case "empirical":
        return this.values[Math.min(this.values.length - 1, Math.floor(rng.next() * this.values.length))];
      case "normal":
        return number("mean") + rng.normal() * Number(params.std ?? params.sigma ?? 0);
      case "lognormal": {
        if (params.median !== undefined) return Math.exp(Math.log(Number(params.median)) + rng.normal() * Math.log(Number(params.geometric_sd ?? params.sigma ?? 1)));
        return Math.exp(number("mean") + rng.normal() * Number(params.sigma ?? 1));
      }
      case "uniform":
        return number("low") + rng.next() * (number("high") - number("low"));
      case "triangular": {
        const low = number("low");
        const high = number("high");
        const mode = number("mode", low);
        const draw = rng.next();
        const split = (mode - low) / Math.max(high - low, Number.MIN_VALUE);
        return draw < split ? low + Math.sqrt(draw * (high - low) * (mode - low)) : high - Math.sqrt((1 - draw) * (high - low) * (high - mode));
      }
      case "quantiles": {
        const pointsValue = params.points ?? params.quantiles;
        const points: Array<[number, number]> = Array.isArray(pointsValue)
          ? pointsValue.map((point) => Array.isArray(point) ? [Number(point[0]), Number(point[1])] : [0, Number(point)])
          : typeof pointsValue === "object" && pointsValue !== null
            ? Object.entries(pointsValue as Record<string, unknown>).map(([probability, point]) => [Number(probability), Number(point)])
            : [];
        points.sort((left, right) => left[0] - right[0]);
        if (!points.length) throw new Error("quantile distributions require points or quantiles");
        const draw = rng.next();
        if (draw <= points[0][0]) return points[0][1];
        for (let index = 1; index < points.length; index += 1) {
          if (draw <= points[index][0]) {
            const [leftQ, leftV] = points[index - 1];
            const [rightQ, rightV] = points[index];
            return leftV + ((draw - leftQ) / Math.max(rightQ - leftQ, Number.MIN_VALUE)) * (rightV - leftV);
          }
        }
        return points[points.length - 1][1];
      }
      default:
        return Number(params.value ?? 0);
    }
  }

  public interval(options: { samples?: number; seed?: number | string; level?: number } = {}): Interval {
    const sampleCount = options.samples ?? 512;
    const level = options.level ?? 0.9;
    if (!Number.isInteger(sampleCount) || sampleCount <= 0) throw new Error("distribution sample count must be positive");
    if (this.kind === "constant") {
      const value = this.sample(new DeterministicRng(options.seed ?? 20261005));
      return { low: value, median: value, high: value, level, method: "deterministic", samples: 1 };
    }
    const rng = new DeterministicRng(options.seed ?? 20261005);
    const draws = Array.from({ length: sampleCount }, () => this.sample(rng));
    return intervalFromSamples(draws, { level, method: `${this.kind}-monte-carlo` });
  }

  public asDict(): Record<string, unknown> {
    return { kind: this.kind, params: { ...this.params }, values: [...this.values], name: this.name, unit: this.unit, provenance: this.provenance.asDict() };
  }

  public as_dict(): Record<string, unknown> { return this.asDict(); }
}

export function distribution(value: unknown, options: { name?: string; unit?: string; provenance?: ProvenanceType } = {}): Distribution {
  return Distribution.fromValue(value, options);
}

export interface DistributionSpecInit {
  kind?: string;
  value?: number;
  low?: number | null;
  high?: number | null;
  mean?: number | null;
  stddev?: number | null;
  shape?: number | null;
  scale?: number | null;
  values?: readonly unknown[];
  weights?: readonly number[];
  params?: Readonly<Record<string, unknown>>;
}

/** Serving-side distribution description mirrored from serving_sim.py. */
export class DistributionSpec {
  public readonly kind: string;
  public readonly value: number;
  public readonly low: number | null;
  public readonly high: number | null;
  public readonly mean: number | null;
  public readonly stddev: number | null;
  public readonly shape: number | null;
  public readonly scale: number | null;
  public readonly values: readonly unknown[];
  public readonly weights: readonly number[];
  public readonly params: Readonly<Record<string, unknown>>;

  public constructor(init: DistributionSpecInit = {}) {
    this.kind = init.kind ?? "fixed";
    this.value = init.value ?? 1;
    this.low = init.low ?? null;
    this.high = init.high ?? null;
    this.mean = init.mean ?? null;
    this.stddev = init.stddev ?? null;
    this.shape = init.shape ?? null;
    this.scale = init.scale ?? null;
    this.values = [...(init.values ?? [])];
    this.weights = [...(init.weights ?? [])];
    this.params = { ...(init.params ?? {}) };
  }

  public sample(rng: RandomSource): number {
    const kind = this.kind.toLowerCase().replaceAll("-", "_");
    const params = this.params;
    const value = Number(params.value ?? this.value);
    const mean = Number(params.mean ?? this.mean ?? value);
    const low = Number(params.low ?? this.low ?? 0);
    const high = Number(params.high ?? this.high ?? low);
    const stddev = Math.max(0, Number(params.stddev ?? params.std ?? this.stddev ?? 0));
    const shape = Math.max(Number.MIN_VALUE, Number(params.shape ?? this.shape ?? 1));
    const scale = Math.max(Number.MIN_VALUE, Number(params.scale ?? this.scale ?? (mean > 0 ? mean : 1)));
    const values = (params.values as readonly unknown[] | undefined) ?? this.values;
    switch (kind) {
      case "fixed":
      case "constant":
      case "deterministic":
        return value;
      case "uniform":
      case "flat":
        return low + rng.next() * (high - low);
      case "normal":
      case "gaussian":
        return mean + rng.normal() * stddev;
      case "lognormal":
      case "log_normal":
        return Math.exp(mean + rng.normal() * stddev);
      case "exponential":
      case "exp":
        return -Math.log(Math.max(Number.MIN_VALUE, 1 - rng.next())) * (1 / Math.max(Number.MIN_VALUE, Number(params.rate_per_s ?? params.rate ?? 0) || 1 / Math.max(mean, scale)));
      case "gamma": {
        // Marsaglia-Tsang is deterministic and sufficient for the positive
        // serving-input draws used by the Worker contract.
        if (shape < 1) return new DistributionSpec({ kind: "gamma", shape: shape + 1, scale }).sample(rng) * Math.pow(rng.next(), 1 / shape);
        const d = shape - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);
        for (;;) {
          const normal = rng.normal();
          const candidate = 1 + c * normal;
          if (candidate <= 0) continue;
          const cube = candidate * candidate * candidate;
          const uniform = rng.next();
          if (uniform < 1 - 0.0331 * normal ** 4 || Math.log(uniform) < 0.5 * normal * normal + d * (1 - cube + Math.log(cube))) return d * cube * scale;
        }
      }
      case "poisson": {
        let product = 1;
        let count = 0;
        const lambda = Math.max(0, mean);
        while (product > Math.exp(-lambda) && count < 100000) { product *= rng.next(); count += 1; }
        return Math.max(0, count - 1);
      }
      case "choice":
      case "categorical":
      case "empirical":
      case "sample":
      case "samples": {
        if (!values.length) return value;
        const index = Math.min(values.length - 1, Math.floor(rng.next() * values.length));
        return Number(values[index]) || mean;
      }
      default:
        return value;
    }
  }

  public asDict(): Record<string, unknown> {
    return { kind: this.kind, value: this.value, low: this.low, high: this.high, mean: this.mean, stddev: this.stddev, shape: this.shape, scale: this.scale, values: [...this.values], weights: [...this.weights], params: { ...this.params } };
  }

  public as_dict(): Record<string, unknown> { return this.asDict(); }
}

export function intervalFromSamples(values: readonly number[], options: { level?: number; method?: string } = {}): Interval {
  if (!values.length) throw new Error("cannot compute an interval from an empty sample");
  const level = options.level ?? 0.9;
  const tail = (1 - level) / 2;
  return { low: quantile(values, tail), median: quantile(values, 0.5), high: quantile(values, 1 - tail), level, method: options.method ?? "empirical", samples: values.length };
}

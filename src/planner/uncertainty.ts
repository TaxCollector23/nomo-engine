import posterior from "./uncertainty.json";
import coverage from "./uncertaintyCoverage.json";
import { TrainingPack, type Pack, type Plan } from "./packs";
import type { Evaluated } from "./search";

export interface Interval {
  median: number;
  low: number;
  high: number;
}

export interface PlanUncertainty {
  objectives: Record<string, Interval>;
  probabilityBest: Record<string, number>;
  validSamples: number;
}

export interface UncertaintyResult {
  available: boolean;
  method: string;
  source: string;
  sampleCount: number;
  validation: { nominal: number; actual: number | null; n: number | null };
  byPlan: Record<string, PlanUncertainty>;
}

export const UNCERTAINTY_VALIDATION = {
  nominal: coverage.nominal_coverage,
  actual: coverage.coverage,
  n: coverage.n,
  covered: coverage.covered,
};

type PosteriorSample = {
  params: Record<string, number>;
  residual_sigma: number;
  fit_error: number;
  source_rows: number;
};

const SAMPLES = (posterior.samples as PosteriorSample[]).filter((s) =>
  Number.isFinite(s.residual_sigma) && s.residual_sigma >= 0 && Object.values(s.params).every(Number.isFinite));

function unwrap(pack: Pack): Pack {
  const inner = (pack as Pack & { inner?: Pack }).inner;
  return inner ? unwrap(inner) : pack;
}

function planKey(plan: Plan): string {
  return JSON.stringify(Object.keys(plan).sort().map((k) => [k, plan[k]]));
}

function quantile(values: number[], q: number): number {
  const ordered = [...values].sort((a, b) => a - b);
  if (!ordered.length) return NaN;
  const p = (ordered.length - 1) * q;
  const lo = Math.floor(p), hi = Math.ceil(p);
  return lo === hi ? ordered[lo]! : ordered[lo]! + (ordered[hi]! - ordered[lo]!) * (p - lo);
}

function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

function normal(seed: number): number {
  let x = (seed >>> 0) || 1;
  x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
  const u1 = Math.max(x / 4294967296, 1e-12);
  x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
  const u2 = x / 4294967296;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function interval(values: number[]): Interval | null {
  if (!values.length || values.some((v) => !Number.isFinite(v) || v <= 0)) return null;
  return { median: quantile(values, 0.5), low: quantile(values, 0.05), high: quantile(values, 0.95) };
}

/**
 * Evaluate the empirical bootstrap artifact produced by the Python reference.
 * The A100 training pack is the only calibrated uncertainty source today;
 * uncalibrated serving/co-design packs deliberately return null.
 */
export function uncertaintyForPlans(pack: Pack, plans: Evaluated[]): UncertaintyResult | null {
  return uncertaintyForPlansWithRanking(pack, plans, plans);
}

export function uncertaintyForPlansWithRanking(pack: Pack, plans: Evaluated[], rankingPlans: Evaluated[]): UncertaintyResult | null {
  const concrete = unwrap(pack);
  if (!(concrete instanceof TrainingPack) || !concrete.calibration || concrete.p.cluster.key !== "a100_nvlink_ib" || !SAMPLES.length) return null;
  const objectives = concrete.objectives();
  const byPlan: Record<string, PlanUncertainty> = {};

  for (const evaluated of plans) {
    const drawsByObjective: Record<string, number[]> = {};
    objectives.forEach((o) => { drawsByObjective[o.name] = []; });
    for (let i = 0; i < SAMPLES.length; i++) {
      const sample = SAMPLES[i]!;
      const metrics = new TrainingPack(concrete.p, sample.params).evaluate(evaluated.plan);
      for (const objective of objectives) {
        const point = metrics.objectives[objective.name]!;
        if (!Number.isFinite(point) || point <= 0) continue;
        const noise = normal(hash(`${planKey(evaluated.plan)}:${objective.name}:${i}:20260929`)) * sample.residual_sigma;
        drawsByObjective[objective.name]!.push(Math.exp(Math.log(point) + noise));
      }
    }
    const uncertain: PlanUncertainty = { objectives: {}, probabilityBest: {}, validSamples: 0 };
    for (const objective of objectives) {
      const values = drawsByObjective[objective.name]!;
      const q = interval(values);
      if (q) uncertain.objectives[objective.name] = q;
      uncertain.validSamples = Math.max(uncertain.validSamples, values.length);
    }
    byPlan[planKey(evaluated.plan)] = uncertain;
  }

  for (const objective of objectives) {
    const winners = new Map<string, number>();
    const objectiveValues = rankingPlans.map((e) => {
      const key = planKey(e.plan);
      const values = byPlan[key]!.objectives[objective.name];
      return { key, value: values?.median ?? Infinity };
    });
    // Probability of being best is computed from the posterior parameter draw,
    // not from overlapping quantile intervals.
    for (let i = 0; i < SAMPLES.length; i++) {
      let best = Infinity;
      const tied: string[] = [];
      for (const evaluated of rankingPlans) {
        const key = planKey(evaluated.plan);
        const point = new TrainingPack(concrete.p, SAMPLES[i]!.params).evaluate(evaluated.plan).objectives[objective.name]!;
        const score = objective.maximize ? -point : point;
        if (score < best - Math.max(Math.abs(best), 1) * 1e-12) { best = score; tied.length = 0; tied.push(key); }
        else if (Math.abs(score - best) <= Math.max(Math.abs(best), 1) * 1e-12) tied.push(key);
      }
      for (const key of tied) winners.set(key, (winners.get(key) ?? 0) + 1 / Math.max(1, tied.length));
    }
    for (const row of objectiveValues) byPlan[row.key]!.probabilityBest[objective.name] = (winners.get(row.key) ?? 0) / SAMPLES.length;
  }

  return {
    available: true,
    method: String(posterior.method),
    source: String(posterior.source),
    sampleCount: SAMPLES.length,
    validation: UNCERTAINTY_VALIDATION,
    byPlan,
  };
}

export function uncertaintyForPlan(result: UncertaintyResult | null | undefined, plan: Plan): PlanUncertainty | null {
  return result?.byPlan[planKey(plan)] ?? null;
}

export { planKey };

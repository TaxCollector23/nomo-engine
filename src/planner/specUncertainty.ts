import type { Pack, Plan } from "./packs";
import { InferencePack, TrainingPack } from "./packs";

/** Spec-range Monte Carlo for uncalibrated hardware.
 *
 * The calibrated A100 pack ships posterior intervals (uncertainty.ts). Every
 * other cluster — H100 and the AMD Instinct fleet — would otherwise show bare
 * point estimates built from datasheet peaks. This module puts honest error
 * bars on those numbers: it samples the disclosed spec-uncertainty ranges
 * (achievable MFU, overlap efficiency, datasheet derating) and re-evaluates
 * the SAME plan through the SAME evaluator, then reports P10/P50/P90 plus a
 * tornado attribution of which input drives the spread.
 *
 * The sampling is a mulberry32 LCG mirrored byte-for-byte in
 * nomo-planner/studies/export_spec_mc.py; scripts/spec-mc-golden.json pins
 * the percentiles so the browser and Python cannot drift apart. */

export interface SpecInput { name: string; label: string; low: number; high: number; default: number }
export interface SpecInterval { p10: number; p50: number; p90: number }
export interface SpecMcResult {
  available: true;
  method: "spec-range Monte Carlo";
  samples: number;
  inputs: SpecInput[];
  intervals: Record<string, SpecInterval>;
  tornado: Record<string, Array<{ name: string; label: string; swingPct: number }>>;
}

export const SPEC_MC_SAMPLES = 400;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function specMcSeed(text: string): number {
  let h = 1779033703;
  for (let i = 0; i < text.length; i++) {
    h = Math.imul(h ^ text.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return h >>> 0;
}

function unwrap(pack: Pack): Pack {
  const inner = (pack as Pack & { inner?: Pack }).inner;
  return inner ? unwrap(inner) : pack;
}

function quantile(values: number[], q: number): number {
  const ordered = [...values].sort((a, b) => a - b);
  const p = (ordered.length - 1) * q;
  const lo = Math.floor(p);
  const hi = Math.ceil(p);
  return lo === hi ? ordered[lo]! : ordered[lo]! + (ordered[hi]! - ordered[lo]!) * (p - lo);
}

const TRAIN_INPUTS: SpecInput[] = [
  { name: "e_matmul", label: "achievable MFU", low: 0.35, high: 0.65, default: 0.55 },
  { name: "dp_overlap", label: "gradient overlap", low: 0.5, high: 0.85, default: 0.7 },
  { name: "bubble_scale", label: "pipeline bubble factor", low: 0.85, high: 1.15, default: 1.0 },
];

const INFER_INPUTS: SpecInput[] = [
  { name: "compute_eff", label: "achieved share of datasheet flops", low: 0.85, high: 1.0, default: 0.95 },
  { name: "mem_eff", label: "achieved share of datasheet HBM bandwidth", low: 0.8, high: 1.0, default: 0.92 },
];

function lerp(low: number, high: number, u: number): number {
  return low + (high - low) * u;
}

/** Monte Carlo + tornado for one plan on uncalibrated hardware. Returns null
 * for calibrated packs (they carry a posterior instead) or unknown packs. */
export function specMcForPlan(pack: Pack, plan: Plan): SpecMcResult | null {
  const concrete = unwrap(pack);
  const rng = mulberry32(specMcSeed(`${concrete.name}:${JSON.stringify(Object.keys(plan).sort().map((k) => [k, plan[k]]))}`));

  if (concrete instanceof TrainingPack) {
    const objectives = concrete.objectives().map((o) => o.name);
    const drawParams = (u: number[]) => ({
      e_matmul: lerp(TRAIN_INPUTS[0]!.low, TRAIN_INPUTS[0]!.high, u[0]!),
      dp_overlap: lerp(TRAIN_INPUTS[1]!.low, TRAIN_INPUTS[1]!.high, u[1]!),
      bubble_scale: lerp(TRAIN_INPUTS[2]!.low, TRAIN_INPUTS[2]!.high, u[2]!),
    });
    const draws: Record<string, number[]> = {};
    for (const name of objectives) draws[name] = [];
    for (let i = 0; i < SPEC_MC_SAMPLES; i++) {
      const params = drawParams([rng(), rng(), rng()]);
      const metrics = new TrainingPack(concrete.p, params).evaluate(plan);
      for (const name of objectives) {
        const value = metrics.objectives[name]!;
        if (Number.isFinite(value) && value > 0) draws[name]!.push(value);
      }
    }
    const intervals: Record<string, SpecInterval> = {};
    for (const name of objectives) {
      if (!draws[name]!.length) return null;
      intervals[name] = { p10: quantile(draws[name]!, 0.1), p50: quantile(draws[name]!, 0.5), p90: quantile(draws[name]!, 0.9) };
    }
    const tornado: Record<string, Array<{ name: string; label: string; swingPct: number }>> = {};
    for (const name of objectives) {
      const base = new TrainingPack(concrete.p).evaluate(plan).objectives[name]!;
      const rows = TRAIN_INPUTS.map((input) => {
        const lowParams = Object.fromEntries(TRAIN_INPUTS.map((other) => [other.name, other === input ? input.low : other.default]));
        const highParams = Object.fromEntries(TRAIN_INPUTS.map((other) => [other.name, other === input ? input.high : other.default]));
        const lowValue = new TrainingPack(concrete.p, lowParams).evaluate(plan).objectives[name]!;
        const highValue = new TrainingPack(concrete.p, highParams).evaluate(plan).objectives[name]!;
        return { name: input.name, label: input.label, swingPct: (Math.abs(highValue - lowValue) / Math.max(Math.abs(base), 1e-12)) * 100 };
      });
      tornado[name] = rows.sort((a, b) => b.swingPct - a.swingPct);
    }
    return { available: true, method: "spec-range Monte Carlo", samples: SPEC_MC_SAMPLES, inputs: TRAIN_INPUTS, intervals, tornado };
  }

  if (concrete instanceof InferencePack) {
    const objectives = concrete.objectives().map((o) => o.name);
    const withEff = (computeEff: number, memEff: number) => {
      const device = { ...concrete.p.cluster.device };
      device.peakFlops = Object.fromEntries(Object.entries(device.peakFlops).map(([key, value]) => [key, value * computeEff]));
      device.memBw = device.memBw * memEff;
      const cluster = { ...concrete.p.cluster, device };
      return new InferencePack({ ...concrete.p, cluster });
    };
    const draws: Record<string, number[]> = {};
    for (const name of objectives) draws[name] = [];
    for (let i = 0; i < SPEC_MC_SAMPLES; i++) {
      const computeEff = lerp(INFER_INPUTS[0]!.low, INFER_INPUTS[0]!.high, rng());
      const memEff = lerp(INFER_INPUTS[1]!.low, INFER_INPUTS[1]!.high, rng());
      const metrics = withEff(computeEff, memEff).evaluate(plan);
      for (const name of objectives) {
        const value = metrics.objectives[name]!;
        if (Number.isFinite(value) && value > 0) draws[name]!.push(value);
      }
    }
    const intervals: Record<string, SpecInterval> = {};
    for (const name of objectives) {
      if (!draws[name]!.length) return null;
      intervals[name] = { p10: quantile(draws[name]!, 0.1), p50: quantile(draws[name]!, 0.5), p90: quantile(draws[name]!, 0.9) };
    }
    const tornado: Record<string, Array<{ name: string; label: string; swingPct: number }>> = {};
    for (const name of objectives) {
      const base = withEff(INFER_INPUTS[0]!.default, INFER_INPUTS[1]!.default).evaluate(plan).objectives[name]!;
      const rows = INFER_INPUTS.map((input) => {
        const lowValue = withEff(
          input.name === "compute_eff" ? input.low : INFER_INPUTS[0]!.default,
          input.name === "mem_eff" ? input.low : INFER_INPUTS[1]!.default,
        ).evaluate(plan).objectives[name]!;
        const highValue = withEff(
          input.name === "compute_eff" ? input.high : INFER_INPUTS[0]!.default,
          input.name === "mem_eff" ? input.high : INFER_INPUTS[1]!.default,
        ).evaluate(plan).objectives[name]!;
        return { name: input.name, label: input.label, swingPct: (Math.abs(highValue - lowValue) / Math.max(Math.abs(base), 1e-12)) * 100 };
      });
      tornado[name] = rows.sort((a, b) => b.swingPct - a.swingPct);
    }
    return { available: true, method: "spec-range Monte Carlo", samples: SPEC_MC_SAMPLES, inputs: INFER_INPUTS, intervals, tornado };
  }

  return null;
}

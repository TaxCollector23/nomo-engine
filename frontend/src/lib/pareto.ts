import type { EvalItem } from "./telemetry/protocol";

export interface Bounds { lo: [number, number, number]; hi: [number, number, number] }

/** Axis transform: energy and latency span decades (log10), accuracy is linear. */
export function axisValue(it: EvalItem, j: 0 | 1 | 2): number {
  const v = it.f[j];
  return j === 2 ? v : Math.log10(Math.max(v, 1e-30));
}

function quantile(sorted: number[], q: number): number {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i]!;
}

/**
 * Robust plot bounds: the archive contains badly-dominated early samples that would squash the
 * front into a corner, so energy/latency use the [0, 98th] percentile and accuracy the [5th, max]
 * range (the good end of every axis is always kept). Outliers are clamped onto the cube faces.
 */
export function bounds(items: Iterable<EvalItem>): Bounds | null {
  const cols: [number[], number[], number[]] = [[], [], []];
  for (const it of items) for (const j of [0, 1, 2] as const) cols[j].push(axisValue(it, j));
  if (!cols[0].length) return null;
  const lo: [number, number, number] = [0, 0, 0];
  const hi: [number, number, number] = [0, 0, 0];
  for (const j of [0, 1, 2] as const) {
    const s = cols[j].sort((a, b) => a - b);
    lo[j] = j === 2 ? quantile(s, 0.05) : s[0]!;
    hi[j] = j === 2 ? s[s.length - 1]! : quantile(s, 0.98);
    if (hi[j] - lo[j] < 1e-9) {
      lo[j] -= 0.5;
      hi[j] += 0.5;
    }
  }
  return { lo, hi };
}

/** Map to the unit cube [-1, 1]^3. Accuracy is flipped so "better" is up/towards the origin corner. */
export function toScene(it: EvalItem, b: Bounds): [number, number, number] {
  const n = (j: 0 | 1 | 2) => Math.min(1, Math.max(-1, ((axisValue(it, j) - b.lo[j]) / (b.hi[j] - b.lo[j])) * 2 - 1));
  return [n(0), n(2), n(1)];
}

export function formatSI(v: number, unit: string): string {
  const p: [number, string][] = [[1, ""], [1e-3, "m"], [1e-6, "µ"], [1e-9, "n"], [1e-12, "p"]];
  for (const [s, pre] of p) if (Math.abs(v) >= s) return `${(v / s).toPrecision(3)} ${pre}${unit}`;
  return `${v.toExponential(2)} ${unit}`;
}

/** Axis tick labels in physical units for the log axes. */
export function ticks(lo: number, hi: number, isLog: boolean, n = 4): { at: number; label: number }[] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = lo + ((hi - lo) * i) / n;
    return { at: (i / n) * 2 - 1, label: isLog ? 10 ** t : t };
  });
}

/** True when a design is inside the user's trade-off filters. */
export function passesFilter(it: EvalItem, f: { eMax: number | null; lMax: number | null; accMin: number | null }): boolean {
  return (f.eMax === null || it.f[0] <= f.eMax) && (f.lMax === null || it.f[1] <= f.lMax) && (f.accMin === null || it.f[2] >= f.accMin);
}

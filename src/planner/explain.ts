// Explanations computed by re-evaluating plans. Port of nomo_planner/explain.py.
import { feasible, type Plan } from "./packs";
import type { Evaluated, Planner } from "./search";

export interface Counterfactual { variable: string; value: string | number; deltas_pct: Record<string, number>; feasible: boolean; broken: string[] }

const pct = (n: number, o: number) => (o ? (100 * (n - o)) / Math.abs(o) : 0);

export function counterfactuals(pl: Planner, plan: Plan): Counterfactual[] {
  const base = pl.evaluate(plan);
  const objs = pl.pack.objectives();
  const out: Counterfactual[] = [];
  for (const v of pl.pack.variables()) {
    for (const c of v.choices) {
      if (c === plan[v.name]) continue;
      const alt = pl.evaluate({ ...plan, [v.name]: c });
      out.push({
        variable: v.name, value: c,
        deltas_pct: Object.fromEntries(objs.map((o) => [o.name, pct(alt.metrics.objectives[o.name]!, base.metrics.objectives[o.name]!)])),
        feasible: feasible(alt.metrics),
        broken: Object.entries(alt.metrics.constraints).filter(([, x]) => x > 0).map(([k]) => k).sort(),
      });
    }
  }
  return out;
}

export function sensitivity(pl: Planner, plan: Plan): Record<string, number> {
  const res: Record<string, number> = Object.fromEntries(pl.pack.variables().map((v) => [v.name, 0]));
  for (const cf of counterfactuals(pl, plan)) {
    if (cf.feasible) res[cf.variable] = Math.max(res[cf.variable]!, Math.max(...Object.values(cf.deltas_pct).map(Math.abs)));
  }
  return Object.fromEntries(Object.entries(res).sort((a, b) => b[1] - a[1]));
}

export function binding(e: Evaluated, top = 3): string[] {
  return Object.entries(e.metrics.constraints).sort((a, b) => b[1] - a[1]).slice(0, top).map(([k]) => k);
}

export function bestImprovement(pl: Planner, plan: Plan, objective: string): Counterfactual | null {
  const c = counterfactuals(pl, plan).filter((cf) => cf.feasible && cf.deltas_pct[objective]! < -0.5);
  return c.length ? c.reduce((a, b) => (b.deltas_pct[objective]! < a.deltas_pct[objective]! ? b : a)) : null;
}

export function summary(pl: Planner, plan: Plan): string {
  const ev = pl.evaluate(plan);
  const pack = pl.pack;
  const parts = [`Plan: ${pack.describe(plan)}.`];
  if (!feasible(ev.metrics)) {
    parts.push(`It is not feasible: it breaks ${Object.entries(ev.metrics.constraints).filter(([, x]) => x > 0).map(([k]) => k).join(", ")}.`);
    return parts.join(" ");
  }
  const labels = Object.fromEntries(pack.variables().map((v) => [v.name, v.label || v.name]));
  const sens = Object.entries(sensitivity(pl, plan)).filter(([, s]) => s > 1.0).slice(0, 3).map(([k]) => k);
  if (sens.length) parts.push(`The decisions that matter most here are ${sens.map((k) => labels[k]).join(", ")}.`);
  const tight = Object.entries(ev.metrics.constraints).filter(([, x]) => x > -0.1).map(([k]) => k.replace(/_/g, " "));
  if (tight.length) parts.push(`It is close to its limit on ${tight.join(", ")}.`);
  const olab = Object.fromEntries(pack.objectives().map((o) => [o.name, o.label || o.name]));
  for (const o of pack.objectives()) {
    if (o.maximize) continue;
    const imp = bestImprovement(pl, plan, o.name);
    if (imp) {
      const others = Object.entries(imp.deltas_pct).filter(([k]) => k !== o.name).map(([k, d]) => `${olab[k]} ${d >= 0 ? "+" : ""}${d.toFixed(0)}%`).join(", ");
      parts.push(`To lower ${o.label}, setting ${labels[imp.variable]} to ${imp.value} changes it by ${imp.deltas_pct[o.name]! >= 0 ? "+" : ""}${imp.deltas_pct[o.name]!.toFixed(0)}%${others ? ` (${others}).` : "."}`);
      break;
    }
  }
  return parts.join(" ");
}

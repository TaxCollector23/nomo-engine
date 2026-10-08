// Exhaustive search, Pareto front, recommendation (ASF). Port of nomo_planner/search.py.
import { feasible, violation, type Metrics, type Pack, type Plan } from "./packs";
import { uncertaintyForPlansWithRanking, type UncertaintyResult } from "./uncertainty";

export interface Evaluated { plan: Plan; metrics: Metrics; F: number[] }
export interface Result {
  front: Evaluated[];
  closest: Evaluated[];
  all: Evaluated[];
  evaluated: number;
  exhaustive: boolean;
  recommended: Evaluated | null;
  uncertainty: UncertaintyResult | null;
}

export function objectiveVector(pack: Pack, m: Metrics): number[] {
  return pack.objectives().map((o) => (o.maximize ? -m.objectives[o.name]! : m.objectives[o.name]!));
}

export function* enumeratePlans(pack: Pack): Generator<Plan> {
  const vars = pack.variables();
  const idx = vars.map(() => 0);
  if (vars.some((v) => v.choices.length === 0)) return;
  while (true) {
    const plan: Plan = {};
    vars.forEach((v, i) => (plan[v.name] = v.choices[idx[i]!]!));
    yield plan;
    let k = vars.length - 1;                     // last variable changes fastest (itertools.product order)
    while (k >= 0) {
      idx[k]!++;
      if (idx[k]! < vars[k]!.choices.length) break;
      idx[k] = 0;
      k--;
    }
    if (k < 0) return;
  }
}

export function spaceSize(pack: Pack): number {
  return pack.variables().reduce((n, v) => n * v.choices.length, 1);
}

export function nondominated(F: number[][]): boolean[] {
  const n = F.length;
  const keep = new Array<boolean>(n).fill(true);
  for (let i = 0; i < n; i++) {
    const fi = F[i]!;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const fj = F[j]!;
      let le = true, lt = false;
      for (let k = 0; k < fi.length; k++) {
        if (fj[k]! > fi[k]!) { le = false; break; }
        if (fj[k]! < fi[k]!) lt = true;
      }
      if (le && lt) { keep[i] = false; break; }
    }
  }
  return keep;
}

function lexCmp(a: number[], b: number[]): number {
  for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return a[k]! < b[k]! ? -1 : 1;
  return 0;
}

export function recommend(front: Evaluated[], weights?: number[] | null): Evaluated | null {
  if (!front.length) return null;
  const G = front.map((e) => e.F.map((f) => (f > 0 ? Math.log10(Math.max(f, 1e-300)) : f)));
  const m = G[0]!.length;
  const lo = Array.from({ length: m }, (_, k) => Math.min(...G.map((g) => g[k]!)));
  const hi = Array.from({ length: m }, (_, k) => Math.max(...G.map((g) => g[k]!)));
  const w = weights ?? new Array(m).fill(1);
  let best = 0, bestScore = Infinity;
  G.forEach((g, i) => {
    const N = g.map((v, k) => ((v - lo[k]!) / (hi[k]! > lo[k]! ? hi[k]! - lo[k]! : 1)) * w[k]!);
    const score = Math.max(...N) + 1e-3 * N.reduce((a, b) => a + b, 0);
    if (score < bestScore) { bestScore = score; best = i; }
  });
  return front[best]!;
}

export class Planner {
  cache = new Map<string, Evaluated>();
  constructor(public pack: Pack) {}
  key(plan: Plan) { return JSON.stringify(Object.keys(plan).sort().map((k) => [k, plan[k]])); }
  evaluate(plan: Plan): Evaluated {
    const k = this.key(plan);
    let e = this.cache.get(k);
    if (!e) {
      const m = this.pack.evaluate(plan);
      e = { plan: { ...plan }, metrics: m, F: objectiveVector(this.pack, m) };
      this.cache.set(k, e);
    }
    return e;
  }
  run(weights?: number[] | null): Result {
    for (const plan of enumeratePlans(this.pack)) this.evaluate(plan);
    return this.result(weights);
  }
  result(weights?: number[] | null): Result {
    const all = [...this.cache.values()];
    const feas = all.filter((e) => feasible(e.metrics));
    let front: Evaluated[] = [];
    if (feas.length) {
      const keep = nondominated(feas.map((e) => e.F));
      const nd = feas.filter((_, i) => keep[i]);
      const best = new Map<string, Evaluated>();
      const slack = (e: Evaluated) => Math.max(...Object.values(e.metrics.constraints), -1.0);
      for (const e of nd) {
        const k = e.F.map((f) => f.toPrecision(15)).join(",");
        const cur = best.get(k);
        if (!cur || slack(e) < slack(cur)) best.set(k, e);
      }
      front = [...best.values()].sort((a, b) => lexCmp(a.F, b.F));
    }
    const closest = feas.length ? [] : [...all].sort((a, b) => violation(a.metrics) - violation(b.metrics)).slice(0, 5);
    return {
      front, closest, all, evaluated: all.length, exhaustive: true, recommended: recommend(front, weights),
      uncertainty: uncertaintyForPlansWithRanking(this.pack, all, front),
    };
  }
}

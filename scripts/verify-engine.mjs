// Verifies the browser engine against golden outputs from the Python engine (nomo-planner/studies/export_golden.py).
// Run: node scripts/verify-engine.mjs   (bundles src/planner with esbuild, then compares every number)
import { build as esbuild } from "esbuild";
import { readFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const dir = mkdtempSync(join(tmpdir(), "nomo-verify-"));
const entry = join(dir, "entry.ts");
const src = new URL("../src/planner/", import.meta.url).pathname;
await import("node:fs").then((fs) => fs.writeFileSync(entry, `
export * from "${src}registry"; export * from "${src}search"; export * from "${src}explain"; export * from "${src}calibration";`));
await esbuild({ entryPoints: [entry], bundle: true, format: "esm", platform: "node", outfile: join(dir, "engine.mjs"), logLevel: "error" });
const E = await import(pathToFileURL(join(dir, "engine.mjs")).href);
// Python's json writes bare Infinity (unservable designs have infinite cost); JSON.parse needs it quoted
const golden = JSON.parse(readFileSync(new URL("./golden.json", import.meta.url), "utf8").replace(/: (-?)Infinity/g, ': "$1Infinity"'),
  (_k, v) => (v === "Infinity" ? Infinity : v === "-Infinity" ? -Infinity : v));

let checks = 0, worst = 0;
const fail = (msg) => { console.error("FAIL:", msg); process.exit(1); };
const close = (a, b, what) => {
  checks++;
  if (a === null || b === null || a === undefined || b === undefined) { if (a !== b && !(a == null && b == null)) fail(`${what}: ${a} vs ${b}`); return; }
  if (!Number.isFinite(b)) { if (a !== b) fail(`${what}: ${a} vs ${b}`); return; }
  const rel = Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
  worst = Math.max(worst, rel);
  if (rel > 1e-10) fail(`${what}: TS ${a} vs Python ${b} (rel ${rel})`);
};
const samePlan = (a, b) => JSON.stringify(Object.keys(a).sort().map((k) => [k, String(a[k])])) === JSON.stringify(Object.keys(b).sort().map((k) => [k, String(b[k])]));

for (const c of golden.cases) {
  const pack = E.build(c.domain, E.fromPythonProblem(c.domain, c.problem));
  if (E.spaceSize(pack) !== c.space_size) fail(`${c.domain} space ${E.spaceSize(pack)} vs ${c.space_size}`);
  for (const ev of c.evals) {
    const plan = Object.fromEntries(Object.entries(ev.plan).map(([k, v]) => [k, typeof v === "string" && !isNaN(Number(v)) && k !== "recompute" && k !== "matmul_dtype" && k !== "weights" && k !== "kv" && k !== "attention" ? Number(v) : v]));
    const m = pack.evaluate(plan);
    for (const [k, v] of Object.entries(ev.objectives)) close(m.objectives[k], v, `${c.domain} objective ${k}`);
    for (const [k, v] of Object.entries(ev.constraints)) close(m.constraints[k], v, `${c.domain} constraint ${k}`);
    for (const [k, v] of Object.entries(ev.breakdown)) close(m.breakdown[k], v, `${c.domain} breakdown ${k}`);
  }
  const pl = new E.Planner(pack);
  const r = pl.run();
  if (r.front.length !== c.front.length) fail(`${c.domain} front size ${r.front.length} vs ${c.front.length}`);
  r.front.forEach((e, i) => { if (!samePlan(e.plan, c.front[i])) fail(`${c.domain} front[${i}] differs`); checks++; });
  if (c.recommended && !samePlan(r.recommended.plan, c.recommended)) fail(`${c.domain} recommended differs`);
  if (c.weighted_recommended && !samePlan(E.recommend(r.front, c.weights).plan, c.weighted_recommended)) fail(`${c.domain} weighted recommendation differs`);
  if (c.counterfactuals) {
    const cfs = E.counterfactuals(pl, r.recommended.plan);
    if (cfs.length !== c.counterfactuals.length) fail(`${c.domain} counterfactual count`);
    cfs.forEach((cf, i) => {
      const g = c.counterfactuals[i];
      if (cf.variable !== g.variable || String(cf.value) !== String(g.value) || cf.feasible !== g.feasible) fail(`${c.domain} counterfactual ${i}`);
      for (const [k, v] of Object.entries(g.deltas_pct)) close(cf.deltas_pct[k], v, `${c.domain} cf delta`);
    });
  }
  console.log(`ok  ${c.domain.padEnd(14)} ${JSON.stringify(c.problem).slice(0, 70)}  front ${r.front.length}`);
}
const pts = E.calibrationPoints();
golden.calibration.forEach((g, i) => { close(pts[i].measured, g.measured_step_s, "calibration measured"); close(pts[i].predicted, g.predicted_step_s, "calibration predicted"); close(pts[i].uncalibrated, g.predicted_uncalibrated, "calibration uncalibrated"); });
console.log(`ok  calibration: 22 published runs reproduced`);
console.log(`\nALL MATCH: ${checks} checks, worst relative difference ${worst.toExponential(2)}`);

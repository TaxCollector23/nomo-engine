// Verifies the browser engine against golden outputs from the Python engine (nomo-planner/studies/export_golden.py).
// Run: node scripts/verify-engine.mjs   (bundles src/planner with esbuild, then compares every number)
import { build as esbuild } from "esbuild";
import { readFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const dir = mkdtempSync(join(tmpdir(), "nomo-verify-"));
const entry = join(dir, "entry.ts");
const src = fileURLToPath(new URL("../src/planner/", import.meta.url)).replaceAll("\\", "/");
await import("node:fs").then((fs) => fs.writeFileSync(entry, `
export * from "${src}registry"; export * from "${src}search"; export * from "${src}explain"; export * from "${src}calibration"; export * from "${src}layers"; export * from "${src}productModels";`));
await esbuild({ entryPoints: [entry], bundle: true, format: "esm", platform: "node", outfile: join(dir, "engine.mjs"), logLevel: "error" });
const E = await import(pathToFileURL(join(dir, "engine.mjs")).href);
// Python's json writes bare Infinity (unservable designs have infinite cost); JSON.parse needs it quoted
const golden = JSON.parse(readFileSync(new URL("./golden.json", import.meta.url), "utf8").replace(/: (-?)Infinity/g, ': "$1Infinity"'),
  (_k, v) => (v === "Infinity" ? Infinity : v === "-Infinity" ? -Infinity : v));
const layerGolden = JSON.parse(readFileSync(new URL("./layer-golden.json", import.meta.url), "utf8"));
const productGolden = JSON.parse(readFileSync(new URL("./product-golden.json", import.meta.url), "utf8"));

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
const productClose = (a, b, what) => close(a, b, "products " + what);
const productExact = (a, b, what) => { checks++; if (a !== b) fail("products " + what + ": " + a + " vs " + b); };
const chip = productGolden.chip;
const chipResult = E.evaluateChip(chip.spec, chip.workload, chip.precision);
productClose(chipResult.seconds, chip.expected.seconds, "chip seconds");
productClose(chipResult.energyJ, chip.expected.energy_j, "chip energy");
productClose(chipResult.costUsd, chip.expected.cost_usd, "chip cost");
productExact(chipResult.bottleneck, chip.expected.bottleneck, "chip bottleneck");
productExact(chipResult.mapping.densePrecision, chip.expected.dense_precision, "chip precision");
productExact(chipResult.mapping.neuromorphicCores, chip.expected.neuromorphic_cores, "chip neuromorphic cores");
productExact(E.paretoChips([chip.spec], [chip.workload], chip.precision)[0].chip, "golden", "chip pareto");

const reliability = productGolden.reliability;
const reliabilityResult = E.evaluateReliability(reliability.problem, reliability.interval);
productClose(reliabilityResult.goodput, reliability.expected.goodput, "reliability goodput");
productClose(reliabilityResult.costUsdPerUsefulHour, reliability.expected.cost_usd_per_useful_hour, "reliability cost");

const fleet = productGolden.fleet;
const fleetResult = E.sizeFleet(fleet.traffic);
fleetResult.forEach((row, i) => {
  const expected = fleet.expected[i];
  productExact(row.replicas, expected.replicas, "fleet " + i + " replicas");
  productClose(row.approxP99Ms, expected.approx_p99_ms, "fleet " + i + " p99");
  productExact(row.withinTarget, expected.within_target, "fleet " + i + " target");
});

const fineTune = productGolden.finetune;
const fineTuneResult = E.fineTuneOptions(fineTune.problem);
fineTuneResult.forEach((row, i) => {
  const expected = fineTune.expected[i];
  productExact(row.mode, expected.mode, "fine-tune " + i + " mode");
  productClose(row.memoryBytes, expected.memory_bytes, "fine-tune " + i + " memory");
  productClose(row.timeHours, expected.time_hours, "fine-tune " + i + " time");
  productClose(row.costUsd, expected.cost_usd, "fine-tune " + i + " cost");
  productExact(row.feasible, expected.feasible, "fine-tune " + i + " feasibility");
});

const tco = productGolden.tco;
const tcoResult = E.evaluateTco(tco.option);
productClose(tcoResult.buyUsdPerUsefulHour, tco.expected.buy_usd_per_useful_hour, "TCO buy");
productClose(tcoResult.rentUsdPerUsefulHour, tco.expected.rent_usd_per_useful_hour, "TCO rent");
productClose(tcoResult.rangeLow, tco.expected.range_low, "TCO low");
productClose(tcoResult.rangeHigh, tco.expected.range_high, "TCO high");
productExact(tcoResult.recommended, tco.expected.recommended, "TCO recommendation");

const rl = productGolden.rl;
const rlResult = E.evaluateRlPlan(rl.problem, rl.options);
productClose(rlResult.samplesPerHour, rl.expected.samples_per_hour, "RL samples");
productClose(rlResult.costUsdPerUpdate, rl.expected.cost_usd_per_update, "RL cost");
productClose(rlResult.stepSeconds, rl.expected.step_seconds, "RL step");
productExact(rlResult.schedule, rl.expected.schedule, "RL schedule");
productExact(rlResult.colocation, rl.expected.colocation, "RL colocation");
productExact(rlResult.feasible, rl.expected.feasible, "RL feasibility");
console.log("ok  products: chip, reliability, fleet, fine-tune, TCO, RL goldens");

const pts = E.calibrationPoints();
golden.calibration.forEach((g, i) => { close(pts[i].measured, g.measured_step_s, "calibration measured"); close(pts[i].predicted, g.predicted_step_s, "calibration predicted"); close(pts[i].uncalibrated, g.predicted_uncalibrated, "calibration uncalibrated"); });
console.log(`ok  calibration: 22 published runs reproduced`);

const sameLayerPlan = (a, b) => JSON.stringify(a) === JSON.stringify(b);
for (const c of layerGolden.cases) {
  const graph = E.buildLayerGraph(c.config, { seqLen: c.seq_len, batchSize: c.batch_size, source: "golden" });
  const result = E.searchLayerTraining({ graph, pipelineStages: c.pipeline_stages, maxCandidates: c.max_candidates, totalSteps: c.total_steps, seed: c.seed });
  if (!result.best || !sameLayerPlan(result.best, c.best)) fail(`layers ${c.key} best plan differs`);
  if (!sameLayerPlan(result.globalBf16.plan, c.global_bf16)) fail(`layers ${c.key} global BF16 plan differs`);
  if (!sameLayerPlan(result.globalBest.plan, c.global_best)) fail(`layers ${c.key} global plan differs`);
  const objectiveKey = (key) => ({ step_time_s: "stepTimeS", cost_usd_per_step: "costUsdPerStep", memory_headroom: "memoryHeadroom", communication_s: "communicationS", pipeline_bubble_s: "pipelineBubbleS", offload_transfer_s: "offloadTransferS", whole_run_time_s: "wholeRunTimeS", whole_run_cost_usd: "wholeRunCostUsd" })[key] ?? key;
  for (const [key, value] of Object.entries(c.best_objectives)) close(result.bestMetrics.objectives[objectiveKey(key)], value, `layers ${c.key} best ${key}`);
  for (const [key, value] of Object.entries(c.global_objectives)) close(result.globalBest.metrics.objectives[objectiveKey(key)], value, `layers ${c.key} global ${key}`);
  close(result.precisionGainPct, c.precision_gain_pct, `layers ${c.key} precision gain`);
  close(result.perLayerGainPct, c.per_layer_gain_pct, `layers ${c.key} per-layer gain`);
  console.log(`ok  layers ${c.key.padEnd(14)} seed ${c.seed}  best ${new Set(result.best.stages).size} stages`);
}
console.log(`\nALL MATCH: ${checks} checks, worst relative difference ${worst.toExponential(2)}`);

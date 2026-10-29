// Verifies the browser engine against golden outputs from the Python engine (nomo-planner/studies/export_golden.py).
// Run: node scripts/verify-engine.mjs   (bundles src/planner with esbuild, then compares every number)
import { build as esbuild } from "esbuild";
import { readFileSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const dir = mkdtempSync(join(tmpdir(), "nomo-verify-"));
const entry = join(dir, "entry.ts");
const src = fileURLToPath(new URL("../src/planner/", import.meta.url)).replaceAll("\\", "/");
await import("node:fs").then((fs) => fs.writeFileSync(entry, `
  export * from "${src}registry"; export * from "${src}search"; export * from "${src}explain"; export * from "${src}calibration"; export * from "${src}layers"; export * from "${src}productModels"; export * from "${src}chipDesign"; export * from "${src}auditor"; export * from "${src}costTracker"; export * from "${src}customerCalibration"; export * from "${src}research"; export * from "${src}specUncertainty";`));
await esbuild({ entryPoints: [entry], bundle: true, format: "esm", platform: "node", outfile: join(dir, "engine.mjs"), logLevel: "error" });
const E = await import(pathToFileURL(join(dir, "engine.mjs")).href);
await esbuild({ entryPoints: [fileURLToPath(new URL("../src/simulation/serving-sim.ts", import.meta.url))], bundle: true, format: "esm", platform: "node", outfile: join(dir, "serving-sim.mjs"), logLevel: "error" });
const S = await import(pathToFileURL(join(dir, "serving-sim.mjs")).href);
// Python's json writes bare Infinity (unservable designs have infinite cost); JSON.parse needs it quoted
const golden = JSON.parse(readFileSync(new URL("./golden.json", import.meta.url), "utf8").replace(/: (-?)Infinity/g, ': "$1Infinity"'),
  (_k, v) => (v === "Infinity" ? Infinity : v === "-Infinity" ? -Infinity : v));
const layerGolden = JSON.parse(readFileSync(new URL("./layer-golden.json", import.meta.url), "utf8"));
const productGolden = JSON.parse(readFileSync(new URL("./product-golden.json", import.meta.url), "utf8"));
const auditorGolden = JSON.parse(readFileSync(new URL("./auditor-golden.json", import.meta.url), "utf8"));
const servingGolden = JSON.parse(readFileSync(new URL("./serving-sim-golden.json", import.meta.url), "utf8"));
const specMcGolden = JSON.parse(readFileSync(new URL("./spec-mc-golden.json", import.meta.url), "utf8"));

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

const designPreview = E.evaluateChipDesign(E.DEFAULT_CHIP_DESIGN, E.DEFAULT_CHIP_WORKLOAD, E.DEFAULT_CHIP_CONSTRAINTS);
productExact(designPreview.calibration.status, "assumption", "chip workbench defaults remain preview");
const calibratedRows = [
  { id: "compute", name: "compute", kind: "compute", baselineSeconds: 1, observedSeconds: 2, observedPowerW: 240, source: "fixture", observedAt: "2026-10-14" },
  { id: "memory", name: "memory", kind: "memory", baselineSeconds: 1, observedSeconds: 2, observedPowerW: 240, source: "fixture", observedAt: "2026-10-14" },
  { id: "link", name: "link", kind: "interconnect", baselineSeconds: 1, observedSeconds: 2, observedPowerW: 240, source: "fixture", observedAt: "2026-10-14" },
];
const designFit = E.evaluateChipDesign(E.DEFAULT_CHIP_DESIGN, E.DEFAULT_CHIP_WORKLOAD, E.DEFAULT_CHIP_CONSTRAINTS, calibratedRows);
productExact(designFit.calibration.status, "calibrated", "chip workbench measurement fit");
productClose(designFit.calibration.computeEfficiency, 0.5, "chip workbench compute efficiency");
if (E.buildChipDesignVariants(E.DEFAULT_CHIP_DESIGN).length !== 4) fail("chip workbench scenario count");
checks += 1;
console.log("ok  chip workbench: constraint gate, calibration fit, and scenario set");

const researchStudy = E.createResearchStudy("2026-10-14T00:00:00.000Z");
researchStudy.protocol = {
  ...researchStudy.protocol,
  title: "Paired latency pilot",
  question: "Does the candidate lower end-to-end latency on the fixed trace?",
  hypothesis: "The candidate will lower latency by at least ten percent.",
  primaryMetric: "latency (ms)",
  successThreshold: 10,
  baseline: "control",
  candidate: "candidate",
  model: "fixture-model@abc123",
  workload: "fixed-trace-v1",
  hardware: "fixture-device / runtime 1",
  method: "five warmup runs, then paired measurements",
  seed: 7,
  repetitions: 2,
  limitations: "Synthetic verification rows; not a hardware claim.",
};
researchStudy.checklist = Object.fromEntries(E.CHECKLIST.map((item) => [item.id, true]));
researchStudy.runs = E.buildRunMatrix(researchStudy);
if (researchStudy.runs.length !== 4 || researchStudy.runs[0].seed !== researchStudy.runs[1].seed) fail("research paired run matrix");
const researchPreview = E.summarizeResearchRuns(researchStudy);
if (researchPreview.evidence !== "preview" || researchPreview.improvementPct !== null) fail("research empty rows remain preview");
researchStudy.runs = researchStudy.runs.map((run, index) => ({
  ...run,
  status: "measured",
  metricValue: run.condition === "baseline" ? (index === 0 ? 10 : 12) : (index === 1 ? 8 : 9),
  source: "fixture-run",
}));
const researchMeasured = E.summarizeResearchRuns(researchStudy);
if (researchMeasured.pairedCount !== 2 || researchMeasured.thresholdMet !== true) fail("research measured comparison");
if (!E.studyMarkdown(researchStudy).includes("# Paired latency pilot") || !E.studyCsv(researchStudy).startsWith("study_id,")) fail("research exports");
if (E.studyManifest(researchStudy).schemaVersion !== "nomo-research-study/1") fail("research manifest schema");
checks += 8;
console.log("ok  research lab: paired protocol, preview boundary, measured comparison, and exports");

for (const fixture of auditorGolden) {
  const parsed = fixture.format === "deepspeed-json" ? E.parseDeepSpeedConfig(fixture.input) : fixture.format === "vllm" ? E.parseVllmCommand(fixture.input) : E.parseMegatronCommand(fixture.input);
  for (const [key, value] of Object.entries(fixture.expected)) {
    checks++;
    if (parsed[key] !== value) fail(`auditor ${fixture.name} ${key}: ${parsed[key]} vs ${value}`);
  }
  const exported = E.exportSameFormat(parsed);
  const roundTrip = fixture.format === "deepspeed-json" ? E.parseDeepSpeedConfig(JSON.parse(exported)) : fixture.format === "vllm" ? E.parseVllmCommand(exported) : E.parseMegatronCommand(exported);
  for (const key of ["model", "sequenceLength", "microBatchSize", "tensorParallel", "pipelineParallel", "dataParallel", "precision", "zeroStage"]) {
    checks++;
    if (roundTrip[key] !== parsed[key]) fail(`auditor ${fixture.name} round-trip ${key}: ${roundTrip[key]} vs ${parsed[key]}`);
  }
  console.log(`ok  auditor ${fixture.name}`);
}

const pts = E.calibrationPoints();
golden.calibration.forEach((g, i) => { close(pts[i].measured, g.measured_step_s, "calibration measured"); close(pts[i].predicted, g.predicted_step_s, "calibration predicted"); close(pts[i].uncalibrated, g.predicted_uncalibrated, "calibration uncalibrated"); });
console.log(`ok  calibration: 22 published runs reproduced`);

const customerRows = E.parseCustomerCsv('run_id,cluster,predicted_step_s,observed_step_s\n"run,1",h100,1,1.2\nrun-2,h100,2,2.4');
const customerFit = E.fitCustomerScale(customerRows, "h100");
if (customerFit.intervalLevel !== 0.9 || !customerFit.intervalScope.includes("not a predictive guarantee")) fail("customer calibration interval label");
checks += 2;
try { E.parseCustomerCsv("run_id,cluster,predicted_step_s,observed_step_s\n,h100,1,1"); fail("customer calibration accepted missing run id"); } catch (error) { if (!String(error).includes("run_id")) fail("customer calibration missing-id error"); checks++; }
if (E.PUBLIC_COST_COVERAGE.openModelTrainingRows !== 2 || E.PUBLIC_COST_COVERAGE.openModelServingRows !== 0) fail("cost coverage boundary");
// Real published row (GPT-4o list price, checked 2024-08-02) must honestly read "stale" today.
if (E.costSourceFreshness(E.PUBLIC_COST_ROWS.find((row) => row.item === "GPT-4o"), new Date("2026-10-06T00:00:00Z")).status !== "stale") fail("cost freshness check");
checks += 2;
console.log("ok  evidence boundaries: customer intervals, quoted CSV, and cost freshness/coverage");

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
  // roofline + memory-timeline visualisation helpers must match Python exactly
  const roof = E.rooflinePoints({ graph, pipelineStages: c.pipeline_stages, maxCandidates: c.max_candidates, totalSteps: c.total_steps, seed: c.seed }, result.best);
  if (roof.length !== c.roofline.length) fail(`layers ${c.key} roofline count ${roof.length} vs ${c.roofline.length}`);
  c.roofline.forEach((g, i) => {
    const p = roof[i];
    if (p.id !== g.id || p.kind !== g.kind || p.stage !== g.stage || p.precision !== g.precision) fail(`layers ${c.key} roofline[${i}] identity`);
    close(p.flops, g.flops, `layers ${c.key} roofline flops`);
    close(p.bytesMin, g.bytes_min, `layers ${c.key} roofline bytes`);
    close(p.intensity, g.intensity, `layers ${c.key} roofline intensity`);
    close(p.ridgeFlops, g.ridge_flops, `layers ${c.key} roofline ridge`);
    close(p.attainableFlops, g.attainable_flops, `layers ${c.key} roofline attainable`);
    if (p.bound !== g.bound) fail(`layers ${c.key} roofline[${i}] bound ${p.bound} vs ${g.bound}`);
  });
  const tl = E.memoryTimeline({ graph, pipelineStages: c.pipeline_stages, maxCandidates: c.max_candidates, totalSteps: c.total_steps, seed: c.seed }, result.best);
  if (tl.stages.length !== c.timeline.stages.length) fail(`layers ${c.key} timeline stage count`);
  c.timeline.stages.forEach((g, i) => {
    const s = tl.stages[i];
    close(s.baselineBytes, g.baseline_bytes, `layers ${c.key} timeline baseline`);
    close(s.forwardPeakBytes, g.forward_peak_bytes, `layers ${c.key} timeline peak`);
    close(s.endBytes, g.end_bytes, `layers ${c.key} timeline end`);
    close(s.evaluatorBytes, g.evaluator_bytes, `layers ${c.key} timeline evaluator`);
    if (Math.abs(s.forwardPeakBytes - s.evaluatorBytes) > 1e-6 * s.evaluatorBytes) fail(`layers ${c.key} timeline peak must equal evaluator memory`);
  });
  console.log(`ok  layers ${c.key.padEnd(14)} seed ${c.seed}  best ${new Set(result.best.stages).size} stages`);
}
// serving-sim: the browser simulator must reproduce the Python reference request-for-request
if (servingGolden.contract !== "serving-sim-golden-v1") fail("serving-sim golden contract mismatch");
for (const c of servingGolden.cases) {
  const assumptions = { ...c.assumptions };
  const traceRows = c.trace_rows
    ? c.trace_rows.map((row) => ({
        requestId: row.request_id, arrivalS: row.arrival_s, promptTokens: row.prompt_tokens,
        answerTokens: row.answer_tokens, prefixId: row.prefix_id ?? null, prefixTokens: row.prefix_tokens ?? null,
      }))
    : null;
  const result = S.simulateServing(assumptions, traceRows ? { traceRows } : {});
  for (const [key, value] of Object.entries(c.metrics)) close(result.metrics[key], value, `serving ${c.name} metric ${key}`);
  if (result.requests.length !== c.requests.length) fail(`serving ${c.name}: request count ${result.requests.length} vs ${c.requests.length}`);
  c.requests.forEach((g, i) => {
    const r = result.requests[i];
    close(r.ttftS, g.ttft_s, `serving ${c.name} req ${g.request_id} ttft`);
    close(r.e2eS, g.e2e_s, `serving ${c.name} req ${g.request_id} e2e`);
    close(r.outputTokens, g.output_tokens, `serving ${c.name} req ${g.request_id} output_tokens`);
    close(r.preemptions, g.preemptions, `serving ${c.name} req ${g.request_id} preemptions`);
    close(r.maxItlS, g.max_itl_s, `serving ${c.name} req ${g.request_id} max_itl`);
  });
  for (const [name, g] of Object.entries(c.distributions)) {
    const d = result.distributions[name];
    if (!d) fail(`serving ${c.name}: missing distribution ${name}`);
    close(d.p50, g.p50, `serving ${c.name} dist ${name} p50`);
    close(d.p90, g.p90, `serving ${c.name} dist ${name} p90`);
    close(d.p99, g.p99, `serving ${c.name} dist ${name} p99`);
    if (d.values.length < g.values.length) fail(`serving ${c.name} dist ${name}: values ${d.values.length} vs ${g.values.length}`);
    g.values.forEach((v, i) => close(d.values[i], v, `serving ${c.name} dist ${name} value[${i}]`));
  }
  console.log(`ok  serving ${c.name.padEnd(24)} seed ${c.seed}  ${c.metrics.requests} requests, ttft_p99 ${c.metrics.ttft_p99_s === null ? "n/a" : c.metrics.ttft_p99_s.toFixed(4)}s`);
}
// spec-range Monte Carlo: uncalibrated hardware error bars must match Python exactly
if (specMcGolden.contract !== "spec-mc-golden-v1") fail("spec-mc golden contract mismatch");
for (const c of specMcGolden.cases) {
  // rebuild the pack the way the UI does, then run specMcForPlan on the golden plan
  const { build, specMcForPlan } = E;
  const settings = { ...c.case_settings };
  if (Array.isArray(settings.device_counts)) settings.device_counts = settings.device_counts.map(Number);
  if (Array.isArray(settings.batch_sizes)) settings.batch_sizes = settings.batch_sizes.map(Number);
  const pack = build(c.name.startsWith("train") ? "llm_training" : "llm_inference", settings);
  const plan = Object.fromEntries(Object.entries(c.plan).map(([k, v]) => [k, typeof v === "number" ? (Number.isInteger(v) ? v : Number(v)) : v]));
  const result = specMcForPlan(pack, plan);
  if (!result || !result.available) fail(`spec-mc ${c.name}: TS returned no result`);
  for (const [objName, interval] of Object.entries(c.intervals)) {
    close(result.intervals[objName].p10, interval.p10, `spec-mc ${c.name} ${objName} p10`);
    close(result.intervals[objName].p50, interval.p50, `spec-mc ${c.name} ${objName} p50`);
    close(result.intervals[objName].p90, interval.p90, `spec-mc ${c.name} ${objName} p90`);
    const rows = result.tornado[objName];
    c.tornado[objName].forEach((g, i) => {
      if (rows[i].name !== g.name) fail(`spec-mc ${c.name} ${objName} tornado order ${rows[i].name} vs ${g.name}`);
      close(rows[i].swingPct, g.swingPct, `spec-mc ${c.name} ${objName} tornado ${g.name}`);
    });
  }
  console.log(`ok  spec-mc ${c.name.padEnd(26)} ${c.samples} samples`);
}
console.log(`\nALL MATCH: ${checks} checks, worst relative difference ${worst.toExponential(2)}`);
// Optional machine-readable summary for scripts/generate-stats.mjs (npm run stats); no effect on the check logic.
if (process.env.NOMO_STATS_JSON) {
  writeFileSync(process.env.NOMO_STATS_JSON, `${JSON.stringify({ verify: { checks, worstRelativeDiff: worst.toExponential(2) } }, null, 2)}\n`);
}

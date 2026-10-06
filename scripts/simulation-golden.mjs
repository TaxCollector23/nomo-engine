// Deterministic browser-simulation golden runner.
// Usage: node scripts/simulation-golden.mjs [fixture.json] [--print]
// A fixture may contain only expected fields; extra generated fields are ignored.
import { build as esbuild } from "esbuild";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDirectory = fileURLToPath(new URL(".", import.meta.url));
const cliArguments = process.argv.slice(2);
const fixturePath = resolve(cliArguments.find((argument) => !argument.startsWith("--")) ?? join(scriptDirectory, "../src/simulation/fixtures/simulation-golden.json"));
const printOnly = cliArguments.includes("--print");
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));
if (!Array.isArray(fixture.cases) || fixture.cases.length === 0) throw new Error("simulation fixture needs a non-empty cases array");

const tempDirectory = mkdtempSync(join(tmpdir(), "nomo-simulation-golden-"));
const entryPath = join(tempDirectory, "entry.ts");
const bundlePath = join(tempDirectory, "simulation.mjs");
const sourceDirectory = fileURLToPath(new URL("../src/simulation/", import.meta.url)).replaceAll("\\", "/");
writeFileSync(entryPath, `export * from "${sourceDirectory}index";\n`, "utf8");
try {
  await esbuild({ entryPoints: [entryPath], bundle: true, format: "esm", platform: "node", outfile: bundlePath, logLevel: "error" });
  const engine = await import(pathToFileURL(bundlePath).href);
  let checks = 0;
  let worstRelativeError = 0;

  const fail = (message) => {
    throw new Error(`FAIL: ${message}`);
  };
  const compare = (expected, actual, path) => {
    checks += 1;
    if (typeof expected === "number") {
      if (typeof actual !== "number") fail(`${path}: expected number, received ${typeof actual}`);
      const relative = Math.abs(actual - expected) / Math.max(Math.abs(expected), 1e-300);
      worstRelativeError = Math.max(worstRelativeError, relative);
      if (relative > 1e-10) fail(`${path}: ${actual} vs ${expected} (relative ${relative})`);
      return;
    }
    if (expected === null || typeof expected !== "object") {
      if (actual !== expected) fail(`${path}: ${JSON.stringify(actual)} vs ${JSON.stringify(expected)}`);
      return;
    }
    if (Array.isArray(expected)) {
      if (!Array.isArray(actual) || actual.length !== expected.length) fail(`${path}: array length differs`);
      expected.forEach((value, index) => compare(value, actual[index], `${path}[${index}]`));
      return;
    }
    if (typeof actual !== "object" || actual === null) fail(`${path}: expected object`);
    for (const [key, value] of Object.entries(expected)) compare(value, actual[key], `${path}.${key}`);
  };

  for (const testCase of fixture.cases) {
    let actual;
    if (testCase.core === true) {
      const graph = engine.buildCoreOperatorGraph(testCase.graph, testCase.coreOptions ?? {});
      const topologyConfig = testCase.coreTopology ?? {};
      const topology = new engine.CoreTopology(
        topologyConfig.devices ?? 1,
        topologyConfig.gpus_per_node ?? 8,
        new engine.CoreLink(topologyConfig.intra_bandwidth ?? 600e9, topologyConfig.intra_latency ?? 2e-6, "golden intra"),
        new engine.CoreLink(topologyConfig.inter_bandwidth ?? 50e9, topologyConfig.inter_latency ?? 8e-6, "golden inter"),
      );
      const hardware = { peakFlops: testCase.hardware?.peak_flops ?? 1000, memoryBandwidth: testCase.hardware?.memory_bandwidth ?? 100 };
      const q = graph.byId["layer.0.q_proj"];
      const roofline = engine.coreRooflineTime(q, hardware, { precision: graph.precision });
      const collective = engine.coreCollectiveTime(topology, testCase.collective_participants ?? 2, testCase.collective_bytes ?? 120, "all_reduce", "ring");
      const simulation = engine.coreSimulateTrainingStep(graph, hardware, topology, new engine.CoreTrainingOptions(testCase.trainingOptions ?? {}));
      const embeddingEvent = simulation.events.find((event) => event.id === "fwd.mb0.embedding");
      const firstLayerEvent = simulation.events.find((event) => event.id === "fwd.mb0.layer.0.norm1");
      actual = {
        graph: {
          operator_count: graph.operators.length,
          operator_ids: graph.operators.map((operator) => operator.id),
          q_shape: q.shape,
          q_flops: q.flops,
          q_read_bytes: q.readBytes,
          q_write_bytes: q.writeBytes,
          q_live_until: q.liveUntil,
        },
        roofline: { seconds: roofline.seconds, compute_seconds: roofline.computeSeconds, memory_seconds: roofline.memorySeconds, arithmetic_intensity: roofline.arithmeticIntensity },
        collective: { algorithm: collective.algorithm, seconds: collective.seconds, bytes_transferred: collective.bytesTransferred },
        training: {
          step_time_s: simulation.stepTimeS,
          event_count: simulation.events.length,
          gpu_ids: [...new Set(simulation.events.map((event) => event.gpu))],
          has_memory: simulation.memoryTimeline.length > 0,
          embedding_end_s: embeddingEvent?.endS ?? null,
          first_layer_start_s: firstLayerEvent?.startS ?? null,
          collective_kinds: [...new Set(simulation.events.filter((event) => event.stream === "communication").map((event) => event.kind))].sort(),
        },
      };
    } else {
      const graph = engine.buildOperatorGraph(testCase.graph, testCase.graphOptions ?? {});
      const topology = engine.buildTopology(testCase.topology);
      const result = engine.runSimulation({ graph, topology }, testCase.simulationOptions ?? {});
      actual = {
        graph: {
          id: graph.id,
          operator_count: graph.operators.length,
          operator_ids: graph.operators.map((operator) => operator.id),
          forward_flops: graph.accounting.forwardFlops,
          parameter_bytes: graph.accounting.parameterBytes,
          peak_live_bytes: graph.accounting.peakLiveBytes,
        },
        simulation: {
          metrics: result.metrics,
          event_ids: result.timeline.events.map((event) => event.id),
          communication_event_count: result.timeline.events.filter((event) => event.type === "communication").length,
        },
        collective: testCase.collective === undefined ? undefined : (() => {
          const plan = engine.planCollective(topology, testCase.collective);
          return { algorithm: plan.algorithm, steps: plan.steps, duration_seconds: plan.durationSeconds };
        })(),
      };
    }
    if (printOnly) console.log(JSON.stringify({ name: testCase.name, actual }, null, 2));
    else compare(testCase.expected, actual, testCase.name);
    if (!printOnly) console.log(`ok  ${testCase.name}`);
  }
  if (!printOnly) console.log(`ALL MATCH: ${checks} checks, worst relative difference ${worstRelativeError.toExponential(2)}`);
} finally {
  rmSync(tempDirectory, { recursive: true, force: true });
}

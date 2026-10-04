import { build as esbuild } from "esbuild";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const src = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "planner");
const work = mkdtempSync(join(tmpdir(), "nomo-bench-"));
const entry = join(work, "bentry.ts");
const bundle = join(work, "bengine.mjs");
writeFileSync(entry, `export * from ${JSON.stringify(join(src, "registry"))}; export * from ${JSON.stringify(join(src, "search"))}; export * from ${JSON.stringify(join(src, "explain"))};`);
await esbuild({ entryPoints: [entry], bundle: true, format: "esm", platform: "node", outfile: bundle, logLevel: "error" });
const E = await import(pathToFileURL(bundle).href);
for (const [d, s] of [["llm_training", {}], ["llm_training", { device_counts: [64,128,256,512,1024,2048,4096] }], ["llm_inference", {}], ["arch_codesign", { budget_usd: 5e6 }]]) {
  const t = performance.now(); const pl = new E.Planner(E.build(d, s)); const r = pl.run();
  const t2 = performance.now(); E.summary(pl, r.recommended.plan); E.sensitivity(pl, r.recommended.plan);
  console.log(d.padEnd(14), String(r.evaluated).padStart(6), "plans", (t2 - t).toFixed(0).padStart(5), "ms search,", (performance.now() - t2).toFixed(0), "ms explain");
}

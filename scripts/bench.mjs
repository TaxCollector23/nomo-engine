import { build as esbuild } from "esbuild";
const src = "/home/claude/nomo-v5/src/planner/";
import { writeFileSync } from "node:fs";
writeFileSync("/tmp/bentry.ts", `export * from "${src}registry"; export * from "${src}search"; export * from "${src}explain";`);
await esbuild({ entryPoints: ["/tmp/bentry.ts"], bundle: true, format: "esm", platform: "node", outfile: "/tmp/bengine.mjs", logLevel: "error" });
const E = await import("/tmp/bengine.mjs");
for (const [d, s] of [["llm_training", {}], ["llm_training", { device_counts: [64,128,256,512,1024,2048,4096] }], ["llm_inference", {}], ["arch_codesign", { budget_usd: 5e6 }]]) {
  const t = performance.now(); const pl = new E.Planner(E.build(d, s)); const r = pl.run();
  const t2 = performance.now(); E.summary(pl, r.recommended.plan); E.sensitivity(pl, r.recommended.plan);
  console.log(d.padEnd(14), String(r.evaluated).padStart(6), "plans", (t2 - t).toFixed(0).padStart(5), "ms search,", (performance.now() - t2).toFixed(0), "ms explain");
}

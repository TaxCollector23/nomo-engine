import { useMemo, useState } from "react";

import {
  evaluateChip,
  evaluateRlPlan,
  evaluateTco,
  fineTuneOptions,
  optimizeReliability,
  sizeFleet,
  type ChipSpec,
  type TrafficHour,
} from "../planner/productModels";

type Product = "chip" | "rl" | "reliability" | "fleet" | "finetune" | "tco";
type Mode = "guided" | "explore" | "rigor";

const INITIAL_CHIP: ChipSpec = {
  name: "Customer chip",
  memoryBandwidthGbps: 1600,
  sramGb: 80,
  bf16Tops: 1000,
  fp8Tops: 2000,
  interconnectGbps: 800,
  neuromorphicCores: 128,
  neuromorphicMemoryGb: 16,
  costUsd: 10000,
  powerW: 300,
  hourlyCostUsd: 2,
};

const TITLES: Record<Product, string> = {
  chip: "Chip design", rl: "RL post-training", reliability: "Reliability and goodput", fleet: "Serving fleets", finetune: "Fine-tuning", tco: "TCO and procurement",
};

function NumberField({ label, value, onChange, step = "any", min = 0 }: { label: string; value: number; onChange: (value: number) => void; step?: number | "any"; min?: number }) {
  return <label className="product-field">{label}<input type="number" min={min} step={step} value={Number.isFinite(value) ? value : ""} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function safe<T>(run: () => T): { value: T | null; error: string | null } {
  try { return { value: run(), error: null }; } catch (error) { return { value: null, error: error instanceof Error ? error.message : String(error) }; }
}

function downloadJson(product: Product, value: unknown) {
  const blob = new Blob([JSON.stringify({ product, generatedAt: new Date().toISOString(), value }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `nomo-${product}-preview.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function csvCell(value: unknown): string {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function downloadCsv(product: Product, value: unknown) {
  const rows = Array.isArray(value) ? value : [value];
  const objects = rows.filter((row): row is Record<string, unknown> => typeof row === "object" && row !== null && !Array.isArray(row));
  const keys = [...new Set(objects.flatMap((row) => Object.keys(row)))];
  const content = objects.length && keys.length
    ? [keys.map(csvCell).join(","), ...objects.map((row) => keys.map((key) => csvCell(row[key])).join(","))].join("\n")
    : `field,value\nvalue,${csvCell(value)}`;
  const url = URL.createObjectURL(new Blob([content], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `nomo-${product}-preview.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

function parseTraffic(text: string): TrafficHour[] {
  const parsed: unknown = JSON.parse(text);
  if (!Array.isArray(parsed)) throw new Error("traffic input must be a JSON array");
  return parsed.map((row, index) => {
    if (!row || typeof row !== "object") throw new Error(`traffic row ${index + 1} must be an object`);
    const value = row as Record<string, unknown>;
    return { hour: Number(value.hour), arrivalRps: Number(value.arrivalRps), serviceRpsPerReplica: Number(value.serviceRpsPerReplica), p99TargetMs: Number(value.p99TargetMs) };
  });
}

export default function ProductStudio({ mode = "guided" }: { mode?: Mode }) {
  const [product, setProduct] = useState<Product>("chip");
  const [chipSpec, setChipSpec] = useState<ChipSpec>(INITIAL_CHIP);
  const [chipWorkload, setChipWorkload] = useState({ name: "LLM serving workload", denseFlops: 1e15, memoryBytes: 1e12, spikingOps: 2e12, requiresNeuromorphic: false });
  const [chipPrecision, setChipPrecision] = useState<"bf16" | "fp8">("fp8");
  const [rlProblem, setRlProblem] = useState({ answerLengthP50: 128, answerLengthP95: 512, rolloutTokensPerSPerGpu: 1000, trainTokensPerSPerGpu: 5000, rewardTokensPerSPerGpu: 2000, gpuCostPerHour: 2, trainingTokensPerUpdate: 1e6, targetSamplesPerHour: 100 });
  const [rlOptions, setRlOptions] = useState({ rolloutGpus: 2, trainGpus: 4, rewardGpus: 1, colocated: false, asynchronous: true, batchSize: 8, kvPrecision: "fp8" as "bf16" | "fp8" });
  const [reliabilityProblem, setReliabilityProblem] = useState({ failureRatePerHour: 0.1, checkpointWriteHours: 0.05, restartHours: 0.1, gpuCount: 8, gpuCostPerHour: 2 });
  const [trafficText, setTrafficText] = useState('[{"hour":9,"arrivalRps":100,"serviceRpsPerReplica":80,"p99TargetMs":200},{"hour":10,"arrivalRps":200,"serviceRpsPerReplica":80,"p99TargetMs":200}]');
  const [fineTuneProblem, setFineTuneProblem] = useState({ parameterCount: 1e9, trainingTokens: 1e10, memoryBytes: 16e9, peakTflops: 100, gpuMemoryBytes: 80e9, gpuCostPerHour: 2, qualityLoss: { full: 0, lora: 0.5, qlora: 1 } });
  const [tcoOption, setTcoOption] = useState({ name: "Customer GPU", purchaseUsd: 10000, leaseUsdPerHour: 1.2, utilization: 0.5, years: 3, hoursPerYear: 1000, energyUsdPerHour: 0.1, maintenancePct: 0.1, priceLowUsd: 8000, priceHighUsd: 12000 });

  const evaluation = useMemo(() => {
    if (product === "chip") return safe(() => ({ inputs: { spec: chipSpec, workload: chipWorkload, precision: chipPrecision }, output: evaluateChip(chipSpec, chipWorkload, chipPrecision) }));
    if (product === "rl") return safe(() => ({ inputs: { problem: rlProblem, options: rlOptions }, output: evaluateRlPlan(rlProblem, rlOptions) }));
    if (product === "reliability") return safe(() => ({ inputs: reliabilityProblem, output: optimizeReliability(reliabilityProblem) }));
    if (product === "fleet") return safe(() => { const traffic = parseTraffic(trafficText); return { inputs: traffic, output: sizeFleet(traffic) }; });
    if (product === "finetune") return safe(() => ({ inputs: fineTuneProblem, output: fineTuneOptions(fineTuneProblem) }));
    return safe(() => ({ inputs: tcoOption, output: evaluateTco(tcoOption) }));
  }, [product, chipSpec, chipWorkload, chipPrecision, rlProblem, rlOptions, reliabilityProblem, trafficText, fineTuneProblem, tcoOption]);

  const setChip = (key: keyof ChipSpec, value: string | number) => setChipSpec((current) => ({ ...current, [key]: value } as ChipSpec));
  const setWorkload = (key: keyof typeof chipWorkload, value: string | number | boolean) => setChipWorkload((current) => ({ ...current, [key]: value }));
  const setRl = (key: keyof typeof rlProblem, value: number) => setRlProblem((current) => ({ ...current, [key]: value }));
  const setRlo = (key: keyof typeof rlOptions, value: number | boolean | "bf16" | "fp8") => setRlOptions((current) => ({ ...current, [key]: value }));
  const setReliability = (key: keyof typeof reliabilityProblem, value: number) => setReliabilityProblem((current) => ({ ...current, [key]: value }));
  const setFineTune = (key: keyof Omit<typeof fineTuneProblem, "qualityLoss">, value: number) => setFineTuneProblem((current) => ({ ...current, [key]: value }));
  const setQuality = (key: "full" | "lora" | "qlora", value: number) => setFineTuneProblem((current) => ({ ...current, qualityLoss: { ...current.qualityLoss, [key]: value } }));
  const setTco = (key: keyof typeof tcoOption, value: string | number) => setTcoOption((current) => ({ ...current, [key]: value } as typeof current));

  const value = evaluation.value;
  return <div className="lab-neuro">
    <p className="section-kicker">Phase 4 product studio</p>
    <h2>Decision packs for infrastructure teams and chip makers.</h2>
    <p className="lab-lede">Each pack is a physics/queueing estimate with editable customer inputs. Nothing here is a market quote, silicon measurement, quality evaluation, or production capacity guarantee.</p>
    <div className="layer-tabs" role="tablist" aria-label="Product packs">
      {(Object.keys(TITLES) as Product[]).map((item) => <button key={item} type="button" role="tab" aria-selected={product === item} className={product === item ? "is-on" : ""} onClick={() => setProduct(item)}>{TITLES[item]}</button>)}
    </div>
    <section className="lab-card">
      <div className="lab-card-head"><h3>{TITLES[product]}</h3><div><span className="lab-rel warn">Customer inputs</span>{value && <><button className="ui-button ui-button--outline ui-button--compact" type="button" onClick={() => downloadJson(product, value)}>Download JSON</button>{" "}<button className="ui-button ui-button--outline ui-button--compact" type="button" onClick={() => downloadCsv(product, value)}>Download CSV</button></>}</div></div>
      {evaluation.error && <p className="layer-error" role="alert">{evaluation.error}</p>}
      {product === "chip" && <div className="product-input-grid"><div><h4>Chip specification</h4><div className="lab-fields"><NumberField label="Memory bandwidth (GB/s)" value={chipSpec.memoryBandwidthGbps} onChange={(v) => setChip("memoryBandwidthGbps", v)} /><NumberField label="On-chip SRAM (GB)" value={chipSpec.sramGb} onChange={(v) => setChip("sramGb", v)} /><NumberField label="BF16 TOPS" value={chipSpec.bf16Tops} onChange={(v) => setChip("bf16Tops", v)} /><NumberField label="FP8 TOPS" value={chipSpec.fp8Tops} onChange={(v) => setChip("fp8Tops", v)} /><NumberField label="Interconnect (Gb/s)" value={chipSpec.interconnectGbps} onChange={(v) => setChip("interconnectGbps", v)} /><NumberField label="Neuromorphic cores" value={chipSpec.neuromorphicCores} onChange={(v) => setChip("neuromorphicCores", v)} /><NumberField label="Neuromorphic memory (GB)" value={chipSpec.neuromorphicMemoryGb} onChange={(v) => setChip("neuromorphicMemoryGb", v)} /><NumberField label="Power (W)" value={chipSpec.powerW} onChange={(v) => setChip("powerW", v)} /><NumberField label="Hourly cost (USD)" value={chipSpec.hourlyCostUsd} onChange={(v) => setChip("hourlyCostUsd", v)} /></div></div><div><h4>Workload and mapping</h4><div className="lab-fields"><NumberField label="Dense FLOPs" value={chipWorkload.denseFlops} onChange={(v) => setWorkload("denseFlops", v)} /><NumberField label="Memory bytes" value={chipWorkload.memoryBytes} onChange={(v) => setWorkload("memoryBytes", v)} /><NumberField label="Spiking operations" value={chipWorkload.spikingOps} onChange={(v) => setWorkload("spikingOps", v)} /><label className="product-field">Dense precision<select value={chipPrecision} onChange={(event) => setChipPrecision(event.target.value as "bf16" | "fp8")}><option value="bf16">BF16</option><option value="fp8">FP8</option></select></label><label className="product-field product-check"><input type="checkbox" checked={chipWorkload.requiresNeuromorphic} onChange={(event) => setWorkload("requiresNeuromorphic", event.target.checked)} /> Requires neuromorphic resources</label></div></div></div>}
      {product === "rl" && <div><h4>Customer RL distribution and rates</h4><div className="lab-fields"><NumberField label="Answer length p50" value={rlProblem.answerLengthP50} onChange={(v) => setRl("answerLengthP50", v)} /><NumberField label="Answer length p95" value={rlProblem.answerLengthP95} onChange={(v) => setRl("answerLengthP95", v)} /><NumberField label="Rollout tokens/s/GPU" value={rlProblem.rolloutTokensPerSPerGpu} onChange={(v) => setRl("rolloutTokensPerSPerGpu", v)} /><NumberField label="Training tokens/s/GPU" value={rlProblem.trainTokensPerSPerGpu} onChange={(v) => setRl("trainTokensPerSPerGpu", v)} /><NumberField label="Reward tokens/s/GPU" value={rlProblem.rewardTokensPerSPerGpu} onChange={(v) => setRl("rewardTokensPerSPerGpu", v)} /><NumberField label="GPU cost / hour" value={rlProblem.gpuCostPerHour} onChange={(v) => setRl("gpuCostPerHour", v)} /><NumberField label="Tokens / update" value={rlProblem.trainingTokensPerUpdate} onChange={(v) => setRl("trainingTokensPerUpdate", v)} /><NumberField label="Target samples / hour" value={rlProblem.targetSamplesPerHour} onChange={(v) => setRl("targetSamplesPerHour", v)} /><NumberField label="Rollout GPUs" value={rlOptions.rolloutGpus} onChange={(v) => setRlo("rolloutGpus", v)} /><NumberField label="Training GPUs" value={rlOptions.trainGpus} onChange={(v) => setRlo("trainGpus", v)} /><NumberField label="Reward GPUs" value={rlOptions.rewardGpus} onChange={(v) => setRlo("rewardGpus", v)} /><NumberField label="Batch size" value={rlOptions.batchSize} onChange={(v) => setRlo("batchSize", v)} /><label className="product-field">KV precision<select value={rlOptions.kvPrecision} onChange={(event) => setRlo("kvPrecision", event.target.value as "bf16" | "fp8")}><option value="bf16">BF16</option><option value="fp8">FP8</option></select></label><label className="product-field product-check"><input type="checkbox" checked={rlOptions.colocated} onChange={(event) => setRlo("colocated", event.target.checked)} /> Colocate phases</label><label className="product-field product-check"><input type="checkbox" checked={rlOptions.asynchronous} onChange={(event) => setRlo("asynchronous", event.target.checked)} /> Asynchronous schedule</label></div></div>}
      {product === "reliability" && <div><h4>Failure and checkpoint inputs</h4><div className="lab-fields"><NumberField label="Failures / hour" value={reliabilityProblem.failureRatePerHour} onChange={(v) => setReliability("failureRatePerHour", v)} step={0.01} /><NumberField label="Checkpoint write (hours)" value={reliabilityProblem.checkpointWriteHours} onChange={(v) => setReliability("checkpointWriteHours", v)} step={0.01} /><NumberField label="Restart (hours)" value={reliabilityProblem.restartHours} onChange={(v) => setReliability("restartHours", v)} step={0.01} /><NumberField label="GPU count" value={reliabilityProblem.gpuCount} onChange={(v) => setReliability("gpuCount", v)} /><NumberField label="GPU cost / hour" value={reliabilityProblem.gpuCostPerHour} onChange={(v) => setReliability("gpuCostPerHour", v)} /></div></div>}
      {product === "fleet" && <div><h4>Hourly traffic profile</h4><label className="product-field">Traffic JSON<textarea className="layer-config" aria-label="Hourly traffic JSON" value={trafficText} onChange={(event) => setTrafficText(event.target.value)} spellCheck={false} /></label><p className="lab-note">Each row needs hour, arrivalRps, serviceRpsPerReplica, and p99TargetMs. Continuous batching is represented by the documented queueing approximation.</p></div>}
      {product === "finetune" && <div><h4>Model, hardware, and customer quality inputs</h4><div className="lab-fields"><NumberField label="Parameters" value={fineTuneProblem.parameterCount} onChange={(v) => setFineTune("parameterCount", v)} /><NumberField label="Training tokens" value={fineTuneProblem.trainingTokens} onChange={(v) => setFineTune("trainingTokens", v)} /><NumberField label="Baseline memory (bytes)" value={fineTuneProblem.memoryBytes} onChange={(v) => setFineTune("memoryBytes", v)} /><NumberField label="Peak TFLOPS" value={fineTuneProblem.peakTflops} onChange={(v) => setFineTune("peakTflops", v)} /><NumberField label="GPU memory (bytes)" value={fineTuneProblem.gpuMemoryBytes} onChange={(v) => setFineTune("gpuMemoryBytes", v)} /><NumberField label="GPU cost / hour" value={fineTuneProblem.gpuCostPerHour} onChange={(v) => setFineTune("gpuCostPerHour", v)} /><NumberField label="Full quality loss" value={fineTuneProblem.qualityLoss.full} onChange={(v) => setQuality("full", v)} step={0.1} /><NumberField label="LoRA quality loss" value={fineTuneProblem.qualityLoss.lora} onChange={(v) => setQuality("lora", v)} step={0.1} /><NumberField label="QLoRA quality loss" value={fineTuneProblem.qualityLoss.qlora} onChange={(v) => setQuality("qlora", v)} step={0.1} /></div></div>}
      {product === "tco" && <div><h4>Buy/rent and uncertainty inputs</h4><div className="lab-fields"><NumberField label="Purchase price (USD)" value={tcoOption.purchaseUsd} onChange={(v) => setTco("purchaseUsd", v)} /><NumberField label="Lease price / hour" value={tcoOption.leaseUsdPerHour} onChange={(v) => setTco("leaseUsdPerHour", v)} step={0.01} /><NumberField label="Utilization (0–1)" value={tcoOption.utilization} onChange={(v) => setTco("utilization", v)} step={0.01} min={0.01} /><NumberField label="Years" value={tcoOption.years} onChange={(v) => setTco("years", v)} /><NumberField label="Hours / year" value={tcoOption.hoursPerYear} onChange={(v) => setTco("hoursPerYear", v)} /><NumberField label="Energy / hour" value={tcoOption.energyUsdPerHour} onChange={(v) => setTco("energyUsdPerHour", v)} step={0.01} /><NumberField label="Maintenance fraction" value={tcoOption.maintenancePct} onChange={(v) => setTco("maintenancePct", v)} step={0.01} /><NumberField label="Low purchase range" value={tcoOption.priceLowUsd} onChange={(v) => setTco("priceLowUsd", v)} /><NumberField label="High purchase range" value={tcoOption.priceHighUsd} onChange={(v) => setTco("priceHighUsd", v)} /></div></div>}
      {value && <div className="product-result">
        {product === "chip" && (value.output as ReturnType<typeof evaluateChip>).feasible && <p>Best mapping for the entered workload: <b>{(value.output as ReturnType<typeof evaluateChip> & { feasible: true }).bottleneck}</b> bottleneck, {(value.output as ReturnType<typeof evaluateChip> & { feasible: true }).seconds.toFixed(4)} s, ${(value.output as ReturnType<typeof evaluateChip> & { feasible: true }).costUsd.toFixed(4)} per workload.</p>}
        {product === "rl" && <p>{Number((value.output as { samplesPerHour: number }).samplesPerHour).toFixed(1)} samples/hour · ${Number((value.output as { costUsdPerUpdate: number }).costUsdPerUpdate).toFixed(3)} per update · {(value.output as { schedule: string }).schedule} / {(value.output as { colocation: string }).colocation}.</p>}
        {product === "reliability" && <><p>Recommended tested checkpoint interval: {Number((value.output as { checkpointIntervalHours: number }).checkpointIntervalHours).toFixed(2)} h; Young estimate: {Number((value.output as { youngIntervalHours: number }).youngIntervalHours).toFixed(2)} h.</p><p>{(Number((value.output as { goodput: number }).goodput) * 100).toFixed(1)}% useful training time per wall-clock hour.</p></>}
        {product === "fleet" && <div className="lab-prov"><table><thead><tr><th>Hour</th><th>Replicas</th><th>Approx p99</th><th>Within target</th></tr></thead><tbody>{(value.output as Array<{ hour: number; replicas: number; approxP99Ms: number; withinTarget: boolean }>).map((row) => <tr key={row.hour}><td>{row.hour}:00</td><td>{row.replicas}</td><td>{Number(row.approxP99Ms).toFixed(0)} ms</td><td>{row.withinTarget ? "yes" : "no"}</td></tr>)}</tbody></table></div>}
        {product === "finetune" && <div className="lab-prov"><table><thead><tr><th>Mode</th><th>Memory</th><th>Time</th><th>Cost</th><th>Feasible</th></tr></thead><tbody>{(value.output as Array<{ mode: string; memoryBytes: number; timeHours: number; costUsd: number; feasible: boolean }>).map((row) => <tr key={row.mode}><td>{row.mode}</td><td>{(row.memoryBytes / 1e9).toFixed(1)} GB</td><td>{Number(row.timeHours).toFixed(1)} h</td><td>${Number(row.costUsd).toFixed(2)}</td><td>{row.feasible ? "yes" : "no"}</td></tr>)}</tbody></table></div>}
        {product === "tco" && <><p>Recommendation under these entered numbers: <b>{(value.output as { recommended: string }).recommended}</b>.</p><p>${Number((value.output as { rangeLow: number }).rangeLow).toFixed(2)}–${Number((value.output as { rangeHigh: number }).rangeHigh).toFixed(2)} per useful hour across the entered purchase-price range.</p></>}
      </div>}
      <p className="lab-note">All fields above are {product === "fleet" ? "customer traffic inputs" : "customer inputs or labelled assumptions"}. {mode === "rigor" ? "Rigor view: inspect the Methods module for the cited equations and replace assumptions with measurements before making a production decision." : "Switch to Rigor mode for equations, provenance, and assumption boundaries."}</p>
    </section>
  </div>;
}

import { useMemo, useState } from "react";

import { evaluateChip, evaluateRlPlan, evaluateTco, fineTuneOptions, optimizeReliability, sizeFleet, type ChipSpec } from "../planner/productModels";

type Product = "chip" | "rl" | "reliability" | "fleet" | "finetune" | "tco";

const CHIP: ChipSpec = {
  name: "Customer chip (example inputs)",
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

function downloadJson(product: Product, value: unknown) {
  const blob = new Blob([JSON.stringify({ product, generatedAt: new Date().toISOString(), value }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "nomo-" + product + "-preview.json";
  link.click();
  URL.revokeObjectURL(url);
}

export default function ProductStudio() {
  const [product, setProduct] = useState<Product>("chip");
  const chipWorkload = { name: "LLM serving", denseFlops: 1e15, memoryBytes: 1e12, spikingOps: 2e12 };
  const chip = useMemo(() => evaluateChip(CHIP, chipWorkload, "fp8"), []);
  const rl = useMemo(() => evaluateRlPlan({
    answerLengthP50: 128,
    answerLengthP95: 512,
    rolloutTokensPerSPerGpu: 1000,
    trainTokensPerSPerGpu: 5000,
    rewardTokensPerSPerGpu: 2000,
    gpuCostPerHour: 2,
    trainingTokensPerUpdate: 1e6,
    targetSamplesPerHour: 100,
  }, { rolloutGpus: 2, trainGpus: 4, rewardGpus: 1, colocated: false, asynchronous: true, batchSize: 8, kvPrecision: "fp8" }), []);
  const reliability = useMemo(() => optimizeReliability({ failureRatePerHour: 0.1, checkpointWriteHours: 0.05, restartHours: 0.1, gpuCount: 8, gpuCostPerHour: 2 }), []);
  const fleet = useMemo(() => sizeFleet([{ hour: 9, arrivalRps: 100, serviceRpsPerReplica: 80, p99TargetMs: 200 }, { hour: 10, arrivalRps: 200, serviceRpsPerReplica: 80, p99TargetMs: 200 }]), []);
  const fineTune = useMemo(() => fineTuneOptions({ parameterCount: 1e9, trainingTokens: 1e10, memoryBytes: 16e9, peakTflops: 100, gpuMemoryBytes: 80e9, gpuCostPerHour: 2, qualityLoss: { full: 0, lora: 0.5, qlora: 1 } }), []);
  const tco = useMemo(() => evaluateTco({ name: "Example user-entered GPU", purchaseUsd: 10000, leaseUsdPerHour: 1.2, utilization: 0.5, years: 3, hoursPerYear: 1000, energyUsdPerHour: 0.1, maintenancePct: 0.1, priceLowUsd: 8000, priceHighUsd: 12000 }), []);
  const title: Record<Product, string> = { chip: "Chip design", rl: "RL post-training", reliability: "Reliability and goodput", fleet: "Serving fleets", finetune: "Fine-tuning", tco: "TCO and procurement" };
  const exportValue: Record<Product, unknown> = {
    chip: { inputs: CHIP, workload: chipWorkload, output: chip },
    rl: { inputs: "example RL schedule", output: rl },
    reliability: { inputs: "example failure/checkpoint rates", output: reliability },
    fleet: { inputs: "example hourly traffic", output: fleet },
    finetune: { inputs: "example model/training inputs", output: fineTune },
    tco: { inputs: "example purchase/lease inputs", output: tco },
  };

  return (
    <div className="lab-neuro">
      <p className="section-kicker">Phase 4 product studio</p>
      <h2>Decision packs for infrastructure teams and chip makers.</h2>
      <p className="lab-lede">Each pack is a physics/queueing estimate with explicit user inputs. Example values are marked as inputs or assumptions and are not market quotes.</p>
      <div className="layer-tabs" role="tablist" aria-label="Product packs">
        {(Object.keys(title) as Product[]).map((value) => <button key={value} type="button" role="tab" aria-selected={product === value} className={product === value ? "is-on" : ""} onClick={() => setProduct(value)}>{title[value]}</button>)}
      </div>
      <section className="lab-card">
        <div className="lab-card-head">
          <h3>{title[product]}</h3>
          <div><span className="lab-rel warn">Example inputs</span>{" "}<button className="ui-button ui-button--outline ui-button--compact" type="button" onClick={() => downloadJson(product, exportValue[product])}>Download JSON</button></div>
        </div>
        {product === "chip" && (chip.feasible ? <>
          <p>Best mapping for the example LLM workload: <b>{String(chip.bottleneck)}</b> bottleneck, {Number(chip.seconds).toFixed(4)} s, ${Number(chip.costUsd).toFixed(4)} per workload.</p>
          <p className="lab-note">Chip bandwidth, throughput, power, price, and neuromorphic capacity are customer-supplied fields in the reference pack.</p>
        </> : <p className="lab-note">{chip.reason}</p>)}
        {product === "rl" && <p>{Number(rl.samplesPerHour).toFixed(1)} samples/hour · ${Number(rl.costUsdPerUpdate).toFixed(3)} per update · {rl.schedule} / {rl.colocation}.</p>}
        {product === "reliability" && <>
          <p>Recommended tested checkpoint interval: {Number(reliability.checkpointIntervalHours).toFixed(2)} h; Young estimate: {Number(reliability.youngIntervalHours).toFixed(2)} h.</p>
          <p>{(Number(reliability.goodput) * 100).toFixed(1)}% useful training time per wall-clock hour.</p>
        </>}
        {product === "fleet" && <div className="lab-prov">
          <table><thead><tr><th>Hour</th><th>Replicas</th><th>Approx p99</th><th>Within target</th></tr></thead>
            <tbody>{fleet.map((row) => <tr key={row.hour}><td>{row.hour}:00</td><td>{row.replicas}</td><td>{Number(row.approxP99Ms).toFixed(0)} ms</td><td>{row.withinTarget ? "yes" : "no"}</td></tr>)}</tbody>
          </table>
          <p className="lab-note">Continuous batching is approximated by a simple capacity/queueing bound; calibrate with serving metrics.</p>
        </div>}
        {product === "finetune" && <div className="lab-prov">
          <table><thead><tr><th>Mode</th><th>Memory</th><th>Time</th><th>Cost</th><th>Feasible</th></tr></thead>
            <tbody>{fineTune.map((row) => <tr key={row.mode}><td>{row.mode}</td><td>{(row.memoryBytes / 1e9).toFixed(1)} GB</td><td>{Number(row.timeHours).toFixed(1)} h</td><td>${Number(row.costUsd).toFixed(2)}</td><td>{row.feasible ? "yes" : "no"}</td></tr>)}</tbody>
          </table>
          <p className="lab-note">Quality loss is a customer-supplied assumption.</p>
        </div>}
        {product === "tco" && <>
          <p>Recommendation under these user-entered numbers: <b>{tco.recommended}</b>.</p>
          <p>${Number(tco.rangeLow).toFixed(2)}–${Number(tco.rangeHigh).toFixed(2)} per useful hour across the entered purchase-price range.</p>
          <p className="lab-note">No provider price was scraped; purchase, lease, utilization, energy, maintenance, and depreciation inputs are editable in the reference API.</p>
        </>}
      </section>
    </div>
  );
}

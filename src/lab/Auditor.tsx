import { useMemo, useState } from "react";

import { auditText, exportSameFormat, modelConfigKey, type AuditRun } from "../planner/auditor";
import { applyCustomerFit, fitCustomerScale, parseCustomerCsv, type CalibrationFit } from "../planner/customerCalibration";
import { csvTemplate, rankExperiments } from "../planner/experiments";
import { BUILTIN_MODEL_CONFIGS, buildLayerGraph, searchLayerTraining } from "../planner/layers";

const SAMPLE = "torchrun pretrain_gpt.py --model meta-llama/Meta-Llama-3-8B --seq-length 4096 --micro-batch-size 2 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --bf16";

export default function Auditor() {
  const [source, setSource] = useState(SAMPLE);
  const [log, setLog] = useState("step time: 125 ms, tokens/s: 9876, allocated: 12 GiB");
  const [calibrationCsv, setCalibrationCsv] = useState("");
  const result = useMemo<AuditRun>(() => auditText(source, log), [source, log]);
  const referenceRecommendation = useMemo(() => {
    const modelKey = modelConfigKey(result.model);
    if (!modelKey) return null;
    try {
      const graph = buildLayerGraph(BUILTIN_MODEL_CONFIGS[modelKey], { seqLen: result.sequenceLength ?? 2048, batchSize: result.microBatchSize ?? 1, source: "auditor model preset" });
      const recommendation = searchLayerTraining({ graph, hardware: { devices: 8, memoryBytes: 80e9, usableMemory: 0.9, costPerDeviceHour: 3.5 }, pipelineStages: [1, 2, 4, 8, 16, 32], microBatches: 8, maxCandidates: 5000, totalSteps: 1000, seed: 20261003 });
      return { modelKey, recommendation };
    } catch { return null; }
  }, [result.model, result.sequenceLength, result.microBatchSize]);
  const calibration = useMemo<CalibrationFit | null>(() => {
    if (!calibrationCsv.trim()) return null;
    try { return fitCustomerScale(parseCustomerCsv(calibrationCsv)); } catch { return null; }
  }, [calibrationCsv]);
  const experiments = useMemo(() => rankExperiments([
    { name: "tensor-parallel-2", config: { tensor_parallel: 2, sequence_length: result.sequenceLength ?? 4096 }, expectedStepS: 1.1, uncertaintyS: 0.25, benchmarkHours: 0.5 },
    { name: "tensor-parallel-4", config: { tensor_parallel: 4, sequence_length: result.sequenceLength ?? 4096 }, expectedStepS: 0.9, uncertaintyS: 0.45, benchmarkHours: 1 },
    { name: "bf16-baseline", config: { tensor_parallel: result.tensorParallel, precision: "bf16" }, expectedStepS: 1.25, uncertaintyS: 0.12, benchmarkHours: 0.25 },
  ], 1, 1.2), [result.sequenceLength, result.tensorParallel]);
  const fields: Array<[string, string | number | null]> = [
    ["Source", result.sourceFormat], ["Model", result.model ?? "not supplied"], ["Sequence length", result.sequenceLength ?? "not supplied"],
    ["Micro-batch", result.microBatchSize ?? "not supplied"], ["Tensor parallel", result.tensorParallel], ["Pipeline parallel", result.pipelineParallel],
    ["Precision", result.precision], ["ZeRO stage", result.zeroStage ?? "not supplied"], ["Observed step", result.observedStepTimeS === null ? "not supplied" : `${result.observedStepTimeS.toFixed(3)} s`],
    ["Observed throughput", result.observedTokensPerS === null ? "not supplied" : `${result.observedTokensPerS.toLocaleString("en-US")} tokens/s`],
    ["Unrecognized options", result.unrecognizedOptions.length ? result.unrecognizedOptions.join(", ") : "none"],
  ];
  return <div className="lab-neuro">
    <p className="section-kicker">Phase 2 run auditor</p>
    <h2>Turn an existing run into a canonical plan.</h2>
    <p className="lab-lede">Paste a Megatron command, DeepSpeed JSON, or vLLM command. Nomo extracts topology and precision fields, attaches observed log metrics, and keeps anything missing visible.</p>
    <div className="lab-card">
      <label className="layer-model-select">Command or DeepSpeed JSON<textarea className="layer-config" aria-label="Command or DeepSpeed JSON" value={source} onChange={(event) => setSource(event.target.value)} spellCheck={false} /></label>
      <label className="layer-model-select">Optional run log<textarea className="layer-config" aria-label="Optional run log" value={log} onChange={(event) => setLog(event.target.value)} spellCheck={false} /></label>
    </div>
    <div className="lab-card"><div className="lab-card-head"><h3>Canonical audit</h3><span className="lab-rel ok">Parsed locally</span></div><div className="lab-prov"><table><tbody>{fields.map(([label, value]) => <tr key={label}><td>{label}</td><td>{String(value)}</td></tr>)}</tbody></table></div>{result.warnings.map((warning) => <p className="lab-note" key={warning}>Assumption: {warning}</p>)}<p className="lab-note">Observed metrics are attached as evidence only; the auditor does not claim customer calibration until the logs are matched to a known hardware and model configuration.</p></div>
    <div className="lab-card"><div className="lab-card-head"><h3>Corrected same-format export</h3><span className="lab-rel ok">Canonical only</span></div><pre className="lab-config-output">{exportSameFormat(result)}</pre><p className="lab-note">This is a loss-aware re-serialization of fields the parser saw. It is not a performance recommendation; missing fields and unsupported framework options are not invented.</p></div>
    <div className="lab-card"><div className="lab-card-head"><h3>Reference recommendation</h3><span className={`lab-rel ${referenceRecommendation ? "warn" : "warn"}`}>{referenceRecommendation ? "Published preset · assumed hardware" : "Model preset needed"}</span></div>{referenceRecommendation?.recommendation.best && referenceRecommendation.recommendation.bestMetrics ? <><p>For {referenceRecommendation.modelKey.replaceAll("_", " ")} under the default 8 × 80 GB / ${3.5}/GPU-hour assumption, the bounded layer planner recommends <b>{referenceRecommendation.recommendation.bestMetrics.objectives.stepTimeS.toFixed(3)} s/step</b> and <b>${referenceRecommendation.recommendation.bestMetrics.objectives.costUsdPerStep.toFixed(4)}/step</b>.</p><p className="lab-note">This connects the parsed topology to the shared graph, but it is not a customer-specific recommendation until hardware, quality, and framework measurements are supplied. Search evaluated {referenceRecommendation.recommendation.evaluated.toLocaleString("en-US")} candidates; precision gain {referenceRecommendation.recommendation.precisionGainPct.toFixed(1)}%, additional per-layer gain {referenceRecommendation.recommendation.perLayerGainPct.toFixed(1)}%.</p></> : <p className="lab-note">The auditor can parse this run, but it cannot safely map an unknown model to the shared graph. No recommendation was invented.</p>}</div>
    <div className="lab-card"><div className="lab-card-head"><h3>Calibrate on your cluster</h3><span className={`lab-rel ${calibration ? "ok" : "warn"}`}>{calibration ? `Calibrated on your cluster (${calibration.observations} runs)` : "Customer input needed"}</span></div><label className="layer-model-select">Local calibration CSV<textarea className="layer-config" aria-label="Local calibration CSV" placeholder="run_id,cluster,predicted_step_s,observed_step_s" value={calibrationCsv} onChange={(event) => setCalibrationCsv(event.target.value)} spellCheck={false} /></label>{calibration ? <><p className="lab-note">Multiplicative refit scale {calibration.scale.toFixed(3)} · residual spread {(calibration.residualSigma * 100).toFixed(1)}% · in-sample 90% coverage {(calibration.coverage * 100).toFixed(0)}%</p><p className="lab-note">{calibration.heldOutMapePct === null ? "Held-out error needs at least two rows." : `Leave-one-out held-out error ${calibration.heldOutMapePct.toFixed(1)}% across ${calibration.heldOutObservations} rows.`} Data stays in this browser; the fit is not merged into the published study.</p><p className="lab-note">For a 1.00 s predicted step, calibrated range is {applyCustomerFit(1, calibration).low.toFixed(2)}–{applyCustomerFit(1, calibration).high.toFixed(2)} s.</p></> : <p className="lab-note">Nothing is uploaded. Provide at least two rows to preview a local fit with held-out error; three or more runs are recommended before using it.</p>}</div>
    <div className="lab-card"><div className="lab-card-head"><h3>Recommendation-sensitive experiments</h3><span className="lab-rel ok">Ranked locally</span></div><p className="lab-note">Short benchmarks are ranked by estimated probability of changing the current decision divided by benchmark hours. These are design candidates, not measurements.</p><div className="lab-prov"><table><thead><tr><th>Experiment</th><th>Change risk</th><th>Value / hour</th><th>Config</th></tr></thead><tbody>{experiments.map((experiment) => <tr key={experiment.name}><td>{experiment.name}</td><td>{(experiment.probabilityChangesRecommendation * 100).toFixed(0)}%</td><td>{experiment.valuePerBenchmarkHour.toFixed(2)}</td><td>{JSON.stringify(experiment.config)}</td></tr>)}</tbody></table></div><details><summary>Download CSV template</summary><pre className="lab-note">{csvTemplate(experiments)}</pre></details></div>
  </div>;
}

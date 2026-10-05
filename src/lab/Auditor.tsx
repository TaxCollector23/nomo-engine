import { useMemo, useState } from "react";

import { auditText, exportSameFormat, type AuditRun } from "../planner/auditor";
import { applyCustomerFit, fitCustomerScale, parseCustomerCsv, type CalibrationFit } from "../planner/customerCalibration";
import { csvTemplate, rankExperiments } from "../planner/experiments";

const SAMPLE = "torchrun pretrain_gpt.py --model meta-llama/Meta-Llama-3-8B --seq-length 4096 --micro-batch-size 2 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --bf16";

export default function Auditor() {
  const [source, setSource] = useState(SAMPLE);
  const [log, setLog] = useState("step time: 125 ms, tokens/s: 9876, allocated: 12 GiB");
  const [calibrationCsv, setCalibrationCsv] = useState("");
  const result = useMemo<AuditRun>(() => auditText(source, log), [source, log]);
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
    <div className="lab-card"><div className="lab-card-head"><h3>Calibrate on your cluster</h3><span className={`lab-rel ${calibration ? "ok" : "warn"}`}>{calibration ? `Calibrated on your cluster (${calibration.observations} runs)` : "Customer input needed"}</span></div><label className="layer-model-select">Local calibration CSV<textarea className="layer-config" aria-label="Local calibration CSV" placeholder="run_id,cluster,predicted_step_s,observed_step_s" value={calibrationCsv} onChange={(event) => setCalibrationCsv(event.target.value)} spellCheck={false} /></label>{calibration ? <><p className="lab-note">Multiplicative refit scale {calibration.scale.toFixed(3)} · residual spread {(calibration.residualSigma * 100).toFixed(1)}% · empirical 90% coverage {(calibration.coverage * 100).toFixed(0)}%</p><p className="lab-note">For a 1.00 s predicted step, calibrated range is {applyCustomerFit(1, calibration).low.toFixed(2)}–{applyCustomerFit(1, calibration).high.toFixed(2)} s. Data stays in this browser.</p></> : <p className="lab-note">Nothing is uploaded. Provide at least one row to preview a local fit; three or more runs are recommended before using it.</p>}</div>
    <div className="lab-card"><div className="lab-card-head"><h3>Recommendation-sensitive experiments</h3><span className="lab-rel ok">Ranked locally</span></div><p className="lab-note">Short benchmarks are ranked by estimated probability of changing the current decision divided by benchmark hours. These are design candidates, not measurements.</p><div className="lab-prov"><table><thead><tr><th>Experiment</th><th>Change risk</th><th>Value / hour</th><th>Config</th></tr></thead><tbody>{experiments.map((experiment) => <tr key={experiment.name}><td>{experiment.name}</td><td>{(experiment.probabilityChangesRecommendation * 100).toFixed(0)}%</td><td>{experiment.valuePerBenchmarkHour.toFixed(2)}</td><td>{JSON.stringify(experiment.config)}</td></tr>)}</tbody></table></div><details><summary>Download CSV template</summary><pre className="lab-note">{csvTemplate(experiments)}</pre></details></div>
  </div>;
}

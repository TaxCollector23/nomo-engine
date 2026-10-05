import { useMemo, useState } from "react";

import { auditText, type AuditRun } from "../planner/auditor";

const SAMPLE = "torchrun pretrain_gpt.py --model meta-llama/Meta-Llama-3-8B --seq-length 4096 --micro-batch-size 2 --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 --bf16";

export default function Auditor() {
  const [source, setSource] = useState(SAMPLE);
  const [log, setLog] = useState("step time: 125 ms, tokens/s: 9876, allocated: 12 GiB");
  const result = useMemo<AuditRun>(() => auditText(source, log), [source, log]);
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
  </div>;
}

import { useEffect, useMemo, useState } from "react";

import {
  buildOperatorGraph,
  createDefaultTopology,
  planCollective,
  runSimulation,
  SIMULATION_CONTRACT_VERSION,
  type SimulationWorkerResponse,
  type DType,
  type OperatorGraph,
  type SimulationResult,
  type TimelineEvent,
} from "../simulation";
import { provenanceLabel } from "../simulation/provenance";
import "./simulation-workbench.css";

type WorkloadMode = "training" | "serving";

const GRAPH_DEFAULTS = {
  name: "Transformer reference workload",
  hiddenSize: 4096,
  layers: 32,
  attentionHeads: 32,
  kvHeads: 8,
  intermediateSize: 11008,
  vocabSize: 32000,
  sequenceLength: 2048,
  batchSize: 1,
};

function compactNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1e12) return `${(value / 1e12).toFixed(digits)}T`;
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(digits)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(digits)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(digits)}K`;
  return value.toFixed(digits);
}

function bytes(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TB`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} MB`;
  return `${(value / 1e3).toFixed(1)} KB`;
}

function milliseconds(value: number): string {
  if (value < 1e-3) return `${(value * 1e6).toFixed(1)} µs`;
  if (value < 1) return `${(value * 1e3).toFixed(2)} ms`;
  return `${value.toFixed(2)} s`;
}

function phaseClass(event: TimelineEvent): string {
  if (event.type === "communication") return "is-communication";
  if (event.phase === "backward") return "is-backward";
  if (event.phase === "update") return "is-update";
  return "is-forward";
}

function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="sim-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function EvidenceLabel({ graph, result }: { graph: OperatorGraph; result: SimulationResult }) {
  return (
    <div className="sim-evidence" aria-label="Simulation evidence boundary">
      <span className="sim-status-dot" aria-hidden="true" />
      <strong>Preview · deterministic simulation</strong>
      <span>
        Graph: {provenanceLabel(graph.provenance)} · results: {provenanceLabel(result.provenance)}. Replace the default topology with measured
        device and link artifacts before treating absolute time as a hardware claim.
      </span>
    </div>
  );
}

function TimelineView({ events, makespan }: { events: readonly TimelineEvent[]; makespan: number }) {
  const visible = events.filter((event) => event.durationSeconds > 0).slice(0, 28);
  return (
    <div className="sim-timeline" aria-label="Discrete-event timeline">
      <div className="sim-timeline-scale"><span>0 ms</span><span>{milliseconds(makespan)}</span></div>
      {visible.map((event) => {
        const left = makespan > 0 ? (event.startTime / makespan) * 100 : 0;
        const width = makespan > 0 ? Math.max(0.8, (event.durationSeconds / makespan) * 100) : 100;
        return (
          <div className="sim-event-row" key={event.id} title={`${event.id}: ${milliseconds(event.durationSeconds)} on ${event.deviceId}`}>
            <span className="sim-event-label">{event.operatorId ?? event.phase}</span>
            <div className="sim-event-track">
              <i className={`sim-event-bar ${phaseClass(event)}`} style={{ left: `${left}%`, width: `${Math.min(100 - left, width)}%` }} />
            </div>
            <small>{event.deviceId.replace("device.", "GPU ")}</small>
          </div>
        );
      })}
      {events.length > visible.length && <p className="sim-note">Showing the first {visible.length} timed events of {events.length}; export the run for the complete timeline.</p>}
    </div>
  );
}

export default function SimulationWorkbench() {
  const [mode, setMode] = useState<WorkloadMode>("training");
  const [dtype, setDtype] = useState<DType>("bf16");
  const [devices, setDevices] = useState(4);
  const [sequenceLength, setSequenceLength] = useState(GRAPH_DEFAULTS.sequenceLength);
  const [seed, setSeed] = useState(20261005);

  const graphMode = mode === "training" ? "training" as const : "inference" as const;
  const graph = useMemo(() => buildOperatorGraph({
    ...GRAPH_DEFAULTS,
    mode: graphMode,
    dtype,
    sequenceLength,
    source: "nomo-lab-browser-workbench",
  }, {
    mode: graphMode,
    dtype,
    sequenceLength,
    includeLoss: mode === "training",
    includeBackward: mode === "training",
    includeOptimizer: mode === "training",
  }), [dtype, graphMode, sequenceLength]);

  const topology = useMemo(() => createDefaultTopology(devices, {
    id: `browser-default-${devices}`,
    name: `${devices}-device browser reference topology`,
    source: "nomo-lab-browser-workbench",
  }), [devices]);

  const fallbackResult = useMemo(() => runSimulation({ graph, topology }, { seed, deviceAssignment: "round-robin" }), [graph, seed, topology]);
  const [workerResult, setWorkerResult] = useState<SimulationResult | null>(null);
  useEffect(() => {
    setWorkerResult(fallbackResult);
    if (typeof Worker === "undefined") return undefined;
    const worker = new Worker(new URL("../simulation/worker.ts", import.meta.url), { type: "module" });
    const requestId = `workbench-${Date.now()}-${seed}`;
    const handleMessage = (event: MessageEvent<SimulationWorkerResponse>) => {
      const response = event.data;
      if (response.type === "result" && response.requestId === requestId) setWorkerResult(response.result);
    };
    worker.addEventListener("message", handleMessage);
    worker.postMessage({
      type: "run",
      requestId,
      contractVersion: SIMULATION_CONTRACT_VERSION,
      input: { graph, topology },
      seed,
      options: { deviceAssignment: "round-robin" },
    });
    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
    };
  }, [fallbackResult, graph, seed, topology]);
  const result = workerResult ?? fallbackResult;
  const collective = useMemo(() => devices > 1 ? planCollective(topology, {
    kind: mode === "training" ? "all-reduce" : "all-gather",
    participants: topology.devices.map((device) => device.id),
    bytes: Math.max(1, graph.accounting.parameterBytes / devices),
  }) : null, [devices, graph.accounting.parameterBytes, mode, topology]);
  const [showAll, setShowAll] = useState(false);
  const timelineEvents = showAll ? result.timeline.events : result.timeline.events.slice(0, 28);

  function downloadRun() {
    const payload = {
      graph: { id: graph.id, accounting: graph.accounting, operator_count: graph.operators.length, provenance: graph.provenance },
      topology: { id: topology.id, devices: topology.devices.length, links: topology.links.length, provenance: topology.provenance },
      result: { metrics: result.metrics, events: result.timeline.events.map((event) => typeof event.asDict === "function" ? event.asDict() : event), provenance: result.provenance },
      evidence: "Deterministic analytic browser simulation; default topology is a placeholder.",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `nomo-simulation-${mode}-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }

  return (
    <section className="simulation-workbench" aria-labelledby="simulation-workbench-title">
      <div className="simulation-workbench-heading">
        <div>
          <p className="section-kicker">Shared simulation core</p>
          <h2 id="simulation-workbench-title">See the work before it runs.</h2>
          <p className="simulation-workbench-lede">One operator graph accounts for forward, backward, communication, and memory events. The same contract feeds training and serving studies; large calibration searches stay on the Python reference API.</p>
        </div>
        <button className="ui-button ui-button--outline ui-button--compact" type="button" onClick={downloadRun}>Export run JSON</button>
      </div>

      <div className="sim-controls" aria-label="Simulation controls">
        <div className="sim-control-group" role="group" aria-label="Workload mode">
          <span>Workload</span>
          <button type="button" className={mode === "training" ? "is-selected" : ""} onClick={() => setMode("training")}>Training</button>
          <button type="button" className={mode === "serving" ? "is-selected" : ""} onClick={() => setMode("serving")}>Serving</button>
        </div>
        <label>Precision<select value={dtype} onChange={(event) => setDtype(event.target.value as DType)}><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp8">FP8</option><option value="int8">INT8</option></select></label>
        <label>Devices<select value={devices} onChange={(event) => setDevices(Number(event.target.value))}>{[1, 2, 4, 8].map((count) => <option key={count} value={count}>{count} GPU{count > 1 ? "s" : ""}</option>)}</select></label>
        <label>Sequence<select value={sequenceLength} onChange={(event) => setSequenceLength(Number(event.target.value))}>{[512, 1024, 2048, 4096].map((length) => <option key={length} value={length}>{length.toLocaleString()} tokens</option>)}</select></label>
        <label>Seed<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value) || 0)} /></label>
      </div>

      <EvidenceLabel graph={graph} result={result} />

      <div className="sim-metrics-grid">
        <Metric label="Operators" value={graph.operators.length.toLocaleString()} detail={`${graph.tensors.length.toLocaleString()} tensors · ${graph.edges.length.toLocaleString()} edges`} />
        <Metric label="Total FLOPs" value={compactNumber(graph.accounting.totalFlops)} detail={`${compactNumber(graph.accounting.forwardFlops)} forward · ${compactNumber(graph.accounting.backwardFlops)} backward`} />
        <Metric label="Makespan" value={milliseconds(result.metrics.makespanSeconds)} detail={`${result.metrics.eventCount} events · ${Math.round(result.metrics.computeUtilization * 100)}% compute busy`} />
        <Metric label="Peak live memory" value={bytes(result.metrics.peakMemoryBytes)} detail={`${bytes(graph.accounting.parameterBytes)} parameters`} />
        <Metric label="Communication" value={milliseconds(result.metrics.communicationSeconds)} detail={`${bytes(result.metrics.totalBytes)} scheduled bytes`} />
      </div>

      <div className="sim-workbench-grid">
        <div className="sim-panel sim-panel--timeline">
          <div className="sim-panel-heading"><div><h3>{mode === "training" ? "Training step timeline" : "Serving prefill timeline"}</h3><p>Dependencies, device assignment, and cross-device transfers are explicit.</p></div><button type="button" className="sim-text-button" onClick={() => setShowAll((value) => !value)}>{showAll ? "Show less" : `Show all ${result.timeline.events.length}`}</button></div>
          <TimelineView events={timelineEvents} makespan={result.timeline.makespanSeconds} />
          <div className="sim-legend"><span className="is-forward">Forward / prefill</span><span className="is-backward">Backward</span><span className="is-update">Optimizer</span><span className="is-communication">Communication</span></div>
        </div>
        <aside className="sim-panel sim-panel--evidence">
          <h3>Graph accounting</h3>
          <dl className="sim-definition-list">
            <div><dt>Model</dt><dd>{graph.name}</dd></div>
            <div><dt>Shape</dt><dd>{graph.layers} layers · {graph.hiddenSize} hidden · {graph.attentionHeads}/{graph.kvHeads} heads</dd></div>
            <div><dt>Mode</dt><dd>{graph.mode} · {graph.dtype}</dd></div>
            <div><dt>Topology</dt><dd>{topology.name}</dd></div>
            <div><dt>Collective</dt><dd>{collective ? `${collective.algorithm} · ${milliseconds(collective.durationSeconds)}` : "single device"}</dd></div>
          </dl>
          <h3>Server boundary</h3>
          <ul className="sim-boundary-list"><li>Python calibration and held-out fitting</li><li>CSV, trace, Prometheus, and customer-log ingestion</li><li>Large constrained searches and Monte Carlo studies</li></ul>
          <p className="sim-note">These stay server-only and are surfaced through the platform API; the browser keeps this run deterministic and responsive.</p>
        </aside>
      </div>
      <SimulationEvidencePanel />
    </section>
  );
}

/** Evidence registry shared by the workbench and the full Evidence module. */
export function SimulationEvidencePanel() {
  const rows = [
    { source: "Narayanan et al. 2021 · Tables 1–2", url: "https://arxiv.org/abs/2104.04473", kind: "Training", observations: "22", error: "5.9% PTD-P", rank: "0.94 Spearman", coverage: "18/22 · 81.8%", status: "Measured / cited" },
    { source: "Sarathi-Serve 2024 · Table 4", url: "https://arxiv.org/abs/2403.02310", kind: "Serving", observations: "12", error: "72.98% MAPE", rank: "—", coverage: "12/12 matched", status: "Measured / replayed" },
    { source: "Customer telemetry", kind: "Training + serving", observations: "Not supplied", error: "—", rank: "—", coverage: "Preview", status: "Upload CSV / trace through API" },
  ];
  return (
    <div className="sim-evidence-panel" aria-labelledby="sim-evidence-title">
      <div className="sim-panel-heading"><div><p className="section-kicker">Evidence registry</p><h3 id="sim-evidence-title">What is measured, and what is still a Preview.</h3></div><span className="sim-inline-status">No values hidden</span></div>
      <div className="sim-evidence-table-wrap">
        <table className="sim-evidence-table"><thead><tr><th>Source</th><th>Kind</th><th>Rows</th><th>Held-out error</th><th>Rank</th><th>Coverage</th><th>Status</th></tr></thead><tbody>{rows.map((row) => <tr key={row.source}><td>{row.url ? <a href={row.url} target="_blank" rel="noreferrer">{row.source}</a> : row.source}</td><td>{row.kind}</td><td>{row.observations}</td><td>{row.error}</td><td>{row.rank}</td><td>{row.coverage}</td><td><span className={row.status.startsWith("Measured") ? "sim-table-badge is-measured" : "sim-table-badge"}>{row.status}</span></td></tr>)}</tbody></table>
      </div>
      <p className="sim-note">Training rows are the checked-in Study 1 calibration evidence. The Sarathi-Serve rows are measured public observations compared with a reproducible replay derived from published length summaries; the 72.98% error and missing raw trace keep serving fidelity in Preview. Customer telemetry is still not supplied.</p>
    </div>
  );
}

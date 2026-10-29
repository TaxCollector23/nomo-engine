import { useEffect, useMemo, useRef, useState } from "react";

import {
  buildChipDesignVariants,
  chipCalibrationCsv,
  chipCalibrationCsvTemplate,
  DEFAULT_CHIP_CONSTRAINTS,
  DEFAULT_CHIP_DESIGN,
  DEFAULT_CHIP_WORKLOAD,
  evaluateChipDesign,
  parseChipCalibrationCsv,
  type ChipCalibrationKind,
  type ChipCalibrationObservation,
  type ChipConstraints,
  type ChipDesignSpec,
  type ChipEvidenceKind,
  type ChipWorkloadSpec,
} from "../planner/chipDesign";

type Mode = "guided" | "explore" | "rigor";
type Tab = "brief" | "architecture" | "calibration" | "review";

const STORE = "nomo-chip-design-workbench-v1";
const EVIDENCE_OPTIONS: Array<{ value: ChipEvidenceKind; label: string }> = [
  { value: "assumption", label: "Assumption" },
  { value: "datasheet", label: "Datasheet" },
  { value: "measured", label: "Measured" },
  { value: "calibrated", label: "Calibrated" },
];
const CALIBRATION_KINDS: Array<{ value: ChipCalibrationKind; label: string }> = [
  { value: "compute", label: "Compute" },
  { value: "memory", label: "Memory" },
  { value: "interconnect", label: "Interconnect" },
  { value: "end_to_end", label: "End to end" },
];

interface ChipEvidenceMap {
  brief: ChipEvidenceKind;
  fabric: ChipEvidenceKind;
  power: ChipEvidenceKind;
  economics: ChipEvidenceKind;
}

interface ChipWorkbenchState {
  design: ChipDesignSpec;
  workload: ChipWorkloadSpec;
  constraints: ChipConstraints;
  observations: ChipCalibrationObservation[];
  evidence: ChipEvidenceMap;
}

const DEFAULT_STATE: ChipWorkbenchState = {
  design: DEFAULT_CHIP_DESIGN,
  workload: DEFAULT_CHIP_WORKLOAD,
  constraints: DEFAULT_CHIP_CONSTRAINTS,
  observations: [],
  evidence: { brief: "assumption", fabric: "datasheet", power: "datasheet", economics: "assumption" },
};

function loadState(): ChipWorkbenchState {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE) ?? "null") as Partial<ChipWorkbenchState> | null;
    if (!raw || typeof raw !== "object") return DEFAULT_STATE;
    return {
      design: { ...DEFAULT_CHIP_DESIGN, ...(raw.design ?? {}) },
      workload: { ...DEFAULT_CHIP_WORKLOAD, ...(raw.workload ?? {}) },
      constraints: { ...DEFAULT_CHIP_CONSTRAINTS, ...(raw.constraints ?? {}) },
      observations: Array.isArray(raw.observations) ? raw.observations.filter((row): row is ChipCalibrationObservation => Boolean(row && typeof row === "object" && typeof row.id === "string" && typeof row.kind === "string")) : [],
      evidence: { ...DEFAULT_STATE.evidence, ...(raw.evidence ?? {}) },
    };
  } catch {
    return DEFAULT_STATE;
  }
}

function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1e12) return `${(value / 1e12).toFixed(digits)}T`;
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(digits)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(digits)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(digits)}K`;
  return value.toFixed(digits);
}

function seconds(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value < 1e-3) return `${(value * 1e6).toFixed(1)} µs`;
  if (value < 1) return `${(value * 1e3).toFixed(2)} ms`;
  return `${value.toFixed(2)} s`;
}

function download(name: string, content: string, type: string): void {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function NumberInput({ label, value, unit, min = 0, step = "any", onChange }: { label: string; value: number; unit?: string; min?: number; step?: number | "any"; onChange: (value: number) => void }) {
  // Display-value guard: drop binary-float noise (1e15 -> 999999986991104,
  // 0.01 -> 0.009999999776482582) so engineers see the number they entered.
  // The stored state keeps full precision; only the rendered text is cleaned.
  const shown = Number.isFinite(value) ? Number(value.toPrecision(12)) : "";
  return (
    <label className="chip-field">
      <span><b>{label}</b>{unit && <small>{unit}</small>}</span>
      <input type="number" min={min} step={step} value={shown} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function TextInput({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label className="chip-field"><span><b>{label}</b></span><input type="text" value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}

function EvidenceSelect({ value, onChange, label }: { value: ChipEvidenceKind; onChange: (value: ChipEvidenceKind) => void; label: string }) {
  return <label className="chip-evidence-select"><span>{label}</span><select aria-label={label} value={value} onChange={(event) => onChange(event.target.value as ChipEvidenceKind)}>{EVIDENCE_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>;
}

function Metric({ label, value, detail, status }: { label: string; value: string; detail?: string; status?: "good" | "warn" }) {
  return <div className={`chip-metric${status ? ` is-${status}` : ""}`}><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>;
}

function TabButton({ tab, current, label, onChange }: { tab: Tab; current: Tab; label: string; onChange: (tab: Tab) => void }) {
  return <button type="button" role="tab" aria-selected={tab === current} className={tab === current ? "is-on" : ""} onClick={() => onChange(tab)}>{label}</button>;
}

function chipStateForExport(state: ChipWorkbenchState, selectedVariant: string) {
  return {
    schemaVersion: "nomo-chip-design/1",
    generatedAt: new Date().toISOString(),
    selectedVariant,
    designBrief: state.workload,
    architecture: state.design,
    constraints: state.constraints,
    calibration: state.observations,
    evidence: state.evidence,
    boundary: "Early-stage design-space estimate. Replace assumptions with PDK, synthesis, package, thermal, and hardware measurements before tape-out decisions.",
  };
}

export default function ChipDesignWorkbench({ mode }: { mode: Mode }) {
  const [state, setState] = useState<ChipWorkbenchState>(() => loadState());
  const [tab, setTab] = useState<Tab>("brief");
  const [selectedVariant, setSelectedVariant] = useState("baseline");
  const [selectedTile, setSelectedTile] = useState(0);
  const [notice, setNotice] = useState("");
  const importRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    try { localStorage.setItem(STORE, JSON.stringify(state)); } catch { setNotice("Browser storage is unavailable; export the design record before closing this tab."); }
  }, [state]);

  const variants = useMemo(() => buildChipDesignVariants(state.design), [state.design]);
  const activeVariant = variants.find((variant) => variant.id === selectedVariant) ?? variants[0]!;
  const evaluation = useMemo(() => {
    try { return { value: evaluateChipDesign(activeVariant.spec, state.workload, state.constraints, state.observations), error: "" }; }
    catch (error) { return { value: null, error: error instanceof Error ? error.message : String(error) }; }
  }, [activeVariant, state.constraints, state.observations, state.workload]);
  const variantRows = useMemo(() => variants.map((variant) => {
    try { return { variant, value: evaluateChipDesign(variant.spec, state.workload, state.constraints, state.observations), error: "" }; }
    catch (error) { return { variant, value: null, error: error instanceof Error ? error.message : String(error) }; }
  }), [state.constraints, state.observations, state.workload, variants]);

  const setDesign = <K extends keyof ChipDesignSpec>(key: K, value: ChipDesignSpec[K]) => setState((current) => ({ ...current, design: { ...current.design, [key]: value } }));
  const setWorkload = <K extends keyof ChipWorkloadSpec>(key: K, value: ChipWorkloadSpec[K]) => setState((current) => ({ ...current, workload: { ...current.workload, [key]: value } }));
  const setConstraints = <K extends keyof ChipConstraints>(key: K, value: ChipConstraints[K]) => setState((current) => ({ ...current, constraints: { ...current.constraints, [key]: value } }));
  const setEvidence = <K extends keyof ChipEvidenceMap>(key: K, value: ChipEvidenceKind) => setState((current) => ({ ...current, evidence: { ...current.evidence, [key]: value } }));

  function addObservation() {
    const row: ChipCalibrationObservation = {
      id: `measurement-${Date.now()}`,
      name: `Measurement ${state.observations.length + 1}`,
      kind: "end_to_end",
      baselineSeconds: 0.001,
      observedSeconds: 0.001,
      observedPowerW: null,
      source: "",
      observedAt: new Date().toISOString().slice(0, 10),
    };
    setState((current) => ({ ...current, observations: [...current.observations, row].slice(-50) }));
    setNotice("Added an empty measurement row. Enter the observed result and its source before relying on it.");
  }

  function updateObservation(id: string, patch: Partial<ChipCalibrationObservation>) {
    setState((current) => ({ ...current, observations: current.observations.map((row) => row.id === id ? { ...row, ...patch } : row) }));
  }

  function removeObservation(id: string) {
    setState((current) => ({ ...current, observations: current.observations.filter((row) => row.id !== id) }));
  }

  async function importCalibration(file: File) {
    try {
      const imported = parseChipCalibrationCsv(await file.text());
      setState((current) => ({ ...current, observations: [...current.observations, ...imported].slice(-50) }));
      setNotice(`Imported ${imported.length} calibration row${imported.length === 1 ? "" : "s"}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    }
  }

  function reset() {
    setState(DEFAULT_STATE);
    setSelectedVariant("baseline");
    setSelectedTile(0);
    setNotice("Reset the browser-local design record.");
  }

  const result = evaluation.value;
  const calibration = result?.calibration;
  const issue = evaluation.error;
  const tileCount = Math.min(64, Math.max(1, Math.round(state.design.tileCount || 1)));

  return (
    <div className="chip-workbench" aria-label="Chip design workbench">
      <div className="chip-workbench-head">
        <div>
          <p className="section-kicker">Chip design workbench</p>
          <h3>Make an architecture decision you can defend.</h3>
          <p className="chip-workbench-lede">Start with a workload and hard limits, expose the architecture that drives the result, fit the model to measurements, then compare transparent design moves. The browser record is local until you export or sync it.</p>
        </div>
        <div className="chip-workbench-actions">
          <span className={`chip-status chip-status--${calibration?.status ?? "assumption"}`}>{calibration?.status === "calibrated" ? "Calibrated" : calibration?.status === "partial" ? "Partially calibrated" : "Preview assumptions"}</span>
          <button type="button" className="ui-button ui-button--outline ui-button--compact" onClick={() => download(`nomo-chip-design-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(chipStateForExport(state, selectedVariant), null, 2), "application/json")}>Export record</button>
          <button type="button" className="ui-button ui-button--text ui-button--compact" onClick={reset}>Reset</button>
        </div>
      </div>

      <div className="chip-tabs" role="tablist" aria-label="Chip design workflow">
        <TabButton tab="brief" current={tab} label="1 · Brief" onChange={setTab} />
        <TabButton tab="architecture" current={tab} label="2 · Architecture" onChange={setTab} />
        <TabButton tab="calibration" current={tab} label="3 · Calibration" onChange={setTab} />
        <TabButton tab="review" current={tab} label="4 · Review" onChange={setTab} />
      </div>

      {notice && <p className="chip-notice" role="status">{notice}</p>}
      {mode === "guided" && <p className="chip-guided-note">Guided view: the important decisions are visible first. Explore and Rigor modes add the full evidence ledger and scenario details.</p>}

      {tab === "brief" && (
        <div className="chip-section-grid">
          <section className="chip-panel">
            <div className="chip-panel-head"><div><p className="section-kicker">Design brief</p><h4>What must this chip do?</h4></div><EvidenceSelect label="Brief evidence" value={state.evidence.brief} onChange={(value) => setEvidence("brief", value)} /></div>
            <div className="chip-fields">
              <TextInput label="Workload name" value={state.workload.name} onChange={(value) => setWorkload("name", value)} />
              <NumberInput label="Dense work" unit="FLOPs per batch" value={state.workload.denseFlops} onChange={(value) => setWorkload("denseFlops", value)} />
              <NumberInput label="Memory traffic" unit="bytes per batch" value={state.workload.memoryBytes} onChange={(value) => setWorkload("memoryBytes", value)} />
              <NumberInput label="Interconnect traffic" unit="bytes per batch" value={state.workload.interconnectBytes} onChange={(value) => setWorkload("interconnectBytes", value)} />
              <NumberInput label="Event work" unit="spiking operations" value={state.workload.spikingOps} onChange={(value) => setWorkload("spikingOps", value)} />
              <NumberInput label="Batch size" unit="requests" min={1} step={1} value={state.workload.batchSize} onChange={(value) => setWorkload("batchSize", value)} />
              <label className="chip-field"><span><b>Dense precision</b></span><select value={state.workload.precision} onChange={(event) => setWorkload("precision", event.target.value as ChipWorkloadSpec["precision"])}><option value="bf16">BF16</option><option value="fp8">FP8</option></select></label>
            </div>
          </section>

          <section className="chip-panel">
            <div className="chip-panel-head"><div><p className="section-kicker">Acceptance limits</p><h4>What would make it unacceptable?</h4></div><span className="chip-panel-code">hard limits</span></div>
            <div className="chip-fields">
              <NumberInput label="Maximum power" unit="watts" value={state.constraints.maxPowerW} onChange={(value) => setConstraints("maxPowerW", value)} />
              <NumberInput label="Maximum die area" unit="mm²" value={state.constraints.maxAreaMm2} onChange={(value) => setConstraints("maxAreaMm2", value)} />
              <NumberInput label="Maximum latency" unit="milliseconds" value={state.constraints.maxLatencyMs} onChange={(value) => setConstraints("maxLatencyMs", value)} />
              <NumberInput label="Minimum throughput" unit="batches per second" value={state.constraints.minThroughputPerSecond} onChange={(value) => setConstraints("minThroughputPerSecond", value)} />
              <NumberInput label="Maximum cost" unit="USD per batch" value={state.constraints.maxCostUsdPerBatch} onChange={(value) => setConstraints("maxCostUsdPerBatch", value)} step={0.0001} />
              <NumberInput label="Minimum yield" unit="percent" value={state.constraints.minYieldPct} onChange={(value) => setConstraints("minYieldPct", value)} />
            </div>
            <p className="chip-field-note">A limit is a decision rule, not a measurement. Keep the source in the project review journal when it comes from a product or thermal requirement.</p>
          </section>
        </div>
      )}

      {tab === "architecture" && (
        <div className="chip-section-grid chip-section-grid--wide">
          <section className="chip-panel">
            <div className="chip-panel-head"><div><p className="section-kicker">Architecture</p><h4>Describe the silicon you are considering.</h4></div><EvidenceSelect label="Architecture evidence" value={state.evidence.fabric} onChange={(value) => setEvidence("fabric", value)} /></div>
            <div className="chip-fields">
              <TextInput label="Design name" value={state.design.name} onChange={(value) => setDesign("name", value)} />
              <div className="chip-field-pair"><NumberInput label="Process node" unit="nm" value={state.design.processNm} onChange={(value) => setDesign("processNm", value)} /><NumberInput label="Die area" unit="mm²" value={state.design.dieAreaMm2} onChange={(value) => setDesign("dieAreaMm2", value)} /></div>
              <div className="chip-field-pair"><NumberInput label="Tiles" unit="count" min={1} step={1} value={state.design.tileCount} onChange={(value) => setDesign("tileCount", value)} /><NumberInput label="Frequency" unit="GHz" value={state.design.frequencyGHz} onChange={(value) => setDesign("frequencyGHz", value)} /></div>
              <div className="chip-field-pair"><NumberInput label="BF16 peak" unit="TOPS" value={state.design.bf16Tops} onChange={(value) => setDesign("bf16Tops", value)} /><NumberInput label="FP8 peak" unit="TOPS" value={state.design.fp8Tops} onChange={(value) => setDesign("fp8Tops", value)} /></div>
              <div className="chip-field-pair"><NumberInput label="Memory bandwidth" unit="GB/s" value={state.design.memoryBandwidthGbps} onChange={(value) => setDesign("memoryBandwidthGbps", value)} /><NumberInput label="SRAM" unit="GB" value={state.design.sramGb} onChange={(value) => setDesign("sramGb", value)} /></div>
              <div className="chip-field-pair"><NumberInput label="HBM" unit="GB" value={state.design.hbmGb} onChange={(value) => setDesign("hbmGb", value)} /><NumberInput label="Interconnect" unit="Gb/s" value={state.design.interconnectGbps} onChange={(value) => setDesign("interconnectGbps", value)} /></div>
              <div className="chip-field-pair"><NumberInput label="Link latency" unit="ns" value={state.design.interconnectLatencyNs} onChange={(value) => setDesign("interconnectLatencyNs", value)} /><NumberInput label="Event cores" unit="count" min={0} step={1} value={state.design.neuromorphicCores} onChange={(value) => setDesign("neuromorphicCores", value)} /></div>
            </div>
          </section>

          <div className="chip-architecture-side">
            <section className="chip-panel">
              <div className="chip-panel-head"><div><p className="section-kicker">Physical envelope</p><h4>Power, supply, yield, and economics.</h4></div><EvidenceSelect label="Envelope evidence" value={state.evidence.power} onChange={(value) => setEvidence("power", value)} /></div>
              <div className="chip-fields">
                <div className="chip-field-pair"><NumberInput label="Power" unit="watts" value={state.design.powerW} onChange={(value) => setDesign("powerW", value)} /><NumberInput label="Voltage" unit="volts" value={state.design.voltageV} onChange={(value) => setDesign("voltageV", value)} /></div>
                <div className="chip-field-pair"><NumberInput label="Yield" unit="percent" value={state.design.yieldPct} onChange={(value) => setDesign("yieldPct", value)} /><NumberInput label="Unit cost" unit="USD" value={state.design.unitCostUsd} onChange={(value) => setDesign("unitCostUsd", value)} /></div>
                <NumberInput label="Hourly equivalent" unit="USD per hour" value={state.design.hourlyCostUsd} onChange={(value) => setDesign("hourlyCostUsd", value)} />
              </div>
              <p className="chip-field-note">These values are never promoted to silicon facts by the workbench. Attach the PDK, package, thermal, and cost artifacts in the enterprise workspace.</p>
            </section>

            <section className="chip-panel chip-tile-panel">
              <div className="chip-panel-head"><div><p className="section-kicker">Topology sketch</p><h4>{tileCount} logical tiles</h4></div><span className="chip-panel-code">interactive map</span></div>
              <div className="chip-tile-map" role="grid" aria-label="Logical tile map">
                {Array.from({ length: tileCount }, (_, index) => <button key={index} type="button" role="gridcell" aria-label={`Tile ${index + 1}`} className={selectedTile === index ? "is-selected" : ""} onClick={() => setSelectedTile(index)}><b>T{String(index + 1).padStart(2, "0")}</b><small>{index === selectedTile ? "selected" : "compute"}</small></button>)}
              </div>
              <p className="chip-tile-detail"><b>Tile {selectedTile + 1}</b> · logical inspection target. The map is an architecture sketch; placement, routing, floorplan, and timing still require EDA artifacts.</p>
            </section>
          </div>
        </div>
      )}

      {tab === "calibration" && (
        <section className="chip-panel chip-calibration-panel">
          <div className="chip-panel-head"><div><p className="section-kicker">Measurement loop</p><h4>Replace peak assumptions with observed behavior.</h4><p className="chip-panel-copy">Each row compares a declared peak-model time with a measured time. Nomo fits only the path named in the row and leaves missing paths at their explicit assumption.</p></div><div className="chip-calibration-actions"><button type="button" className="ui-button ui-button--outline ui-button--compact" onClick={addObservation}>Add measurement</button><button type="button" className="ui-button ui-button--outline ui-button--compact" onClick={() => download("nomo-chip-calibration-template.csv", chipCalibrationCsvTemplate(), "text/csv")}>CSV template</button><button type="button" className="ui-button ui-button--text ui-button--compact" onClick={() => download("nomo-chip-calibration.csv", chipCalibrationCsv(state.observations), "text/csv")}>Export CSV</button><button type="button" className="ui-button ui-button--text ui-button--compact" onClick={() => importRef.current?.click()}>Import CSV</button><input ref={importRef} className="chip-hidden-file" type="file" accept=".csv,text/csv" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importCalibration(file); event.target.value = ""; }} /></div></div>
          <div className="chip-calibration-summary">
            <Metric label="Fit status" value={calibration?.status === "calibrated" ? "Calibrated" : calibration?.status === "partial" ? "Partial" : "Assumptions"} detail={calibration?.note} status={calibration?.status === "calibrated" ? "good" : "warn"} />
            <Metric label="Valid rows" value={String(calibration?.sampleCount ?? 0)} detail="Rows with finite positive times" />
            <Metric label="Fitted paths" value={calibration?.fittedKinds.join(", ") || "none"} detail="Missing paths stay conservative" />
            <Metric label="P90 residual" value={calibration?.residualP90Pct === null || calibration?.residualP90Pct === undefined ? "n/a" : `${calibration.residualP90Pct.toFixed(1)}%`} detail="Absolute fit residual; not a guarantee" />
          </div>
          {state.observations.length === 0 ? <div className="chip-empty"><b>No measurements loaded.</b><span>Results currently use the declared planning efficiencies: compute 70%, memory 78%, interconnect 82%. Add a row or import the CSV template format.</span></div> : (
            <div className="chip-table-wrap"><table className="chip-table"><thead><tr><th>Name</th><th>Path</th><th>Baseline s</th><th>Observed s</th><th>Power W</th><th>Source</th><th>Observed</th><th aria-label="Actions" /></tr></thead><tbody>{state.observations.map((row) => <tr key={row.id}>
              <td><input aria-label="Measurement name" value={row.name} onChange={(event) => updateObservation(row.id, { name: event.target.value })} /></td>
              <td><select aria-label="Calibration path" value={row.kind} onChange={(event) => updateObservation(row.id, { kind: event.target.value as ChipCalibrationKind })}>{CALIBRATION_KINDS.map((kind) => <option key={kind.value} value={kind.value}>{kind.label}</option>)}</select></td>
              <td><input aria-label="Baseline seconds" type="number" min="0" step="any" value={Number.isFinite(row.baselineSeconds) ? row.baselineSeconds : ""} onChange={(event) => updateObservation(row.id, { baselineSeconds: Number(event.target.value) })} /></td>
              <td><input aria-label="Observed seconds" type="number" min="0" step="any" value={Number.isFinite(row.observedSeconds) ? row.observedSeconds : ""} onChange={(event) => updateObservation(row.id, { observedSeconds: Number(event.target.value) })} /></td>
              <td><input aria-label="Observed power watts" type="number" min="0" step="any" value={row.observedPowerW ?? ""} onChange={(event) => updateObservation(row.id, { observedPowerW: event.target.value === "" ? null : Number(event.target.value) })} /></td>
              <td><input aria-label="Measurement source" value={row.source} placeholder="lab run / URL" onChange={(event) => updateObservation(row.id, { source: event.target.value })} /></td>
              <td><input aria-label="Observed date" type="date" value={row.observedAt} onChange={(event) => updateObservation(row.id, { observedAt: event.target.value })} /></td>
              <td><button type="button" className="chip-remove" aria-label={`Remove ${row.name}`} onClick={() => removeObservation(row.id)}>×</button></td>
            </tr>)}</tbody></table></div>
          )}
          <p className="chip-field-note">Calibration changes model efficiency and power scaling only. It does not prove task accuracy, PPA, thermal closure, yield, or manufacturability. Source strings are recorded for traceability but are not independently verified by this browser.</p>
        </section>
      )}

      {tab === "review" && (
        <div className="chip-review">
          {issue && <div className="chip-error" role="alert">Cannot evaluate this design: {issue}</div>}
          {result && <>
            <section className="chip-review-hero">
              <div><p className="section-kicker">Decision review · {activeVariant.label}</p><h4>{activeVariant.rationale}</h4><p className="chip-panel-copy">{result.evidence === "calibrated" ? "This scenario uses the fitted measurement rows shown in Calibration." : "This scenario is still a planning estimate; add measurements before treating the absolute values as hardware behavior."}</p></div>
              <div className={`chip-feasibility ${result.feasible ? "is-pass" : "is-fail"}`}><b>{result.feasible ? "Passes entered limits" : "Misses entered limits"}</b><span>{result.bottleneck} is the current critical path</span></div>
            </section>
            <div className="chip-metrics-grid"><Metric label="Latency / batch" value={seconds(result.latencySeconds)} detail="max component time" status={result.constraints.latency?.pass ? "good" : "warn"} /><Metric label="Throughput" value={`${formatNumber(result.throughputPerSecond)} /s`} detail="against entered target" status={result.constraints.throughput?.pass ? "good" : "warn"} /><Metric label="Power" value={`${formatNumber(result.powerW)} W`} detail="including fitted power scale" status={result.constraints.power?.pass ? "good" : "warn"} /><Metric label="Energy / batch" value={`${formatNumber(result.energyJPerBatch)} J`} detail="latency × power" /><Metric label="Cost / batch" value={`$${result.costUsdPerBatch.toFixed(5)}`} detail="entered hourly equivalent" status={result.constraints.cost?.pass ? "good" : "warn"} /></div>
            <div className="chip-review-grid">
              <section className="chip-panel"><div className="chip-panel-head"><div><p className="section-kicker">Constraint gate</p><h4>Why this passes or fails.</h4></div><span className={`chip-status chip-status--${result.feasible ? "calibrated" : "partial"}`}>{result.feasible ? "within limits" : "action required"}</span></div><div className="chip-constraint-list">{Object.entries(result.constraints).map(([key, check]) => <div className="chip-constraint" key={key}><div><b>{check.label}</b><span>{check.direction === "max" ? "≤" : "≥"} {formatNumber(check.limit)} {check.unit}</span></div><strong className={check.pass ? "is-pass" : "is-fail"}>{formatNumber(check.value)} {check.unit}</strong><small>{check.pass ? `${formatNumber(Math.abs(check.headroom))} ${check.unit} headroom` : `${formatNumber(Math.abs(check.headroom))} ${check.unit} over limit`}</small></div>)}</div></section>
              <section className="chip-panel"><div className="chip-panel-head"><div><p className="section-kicker">Critical path</p><h4>What to change first.</h4></div><span className="chip-panel-code">{result.bottleneck}</span></div><div className="chip-bar-list">{Object.entries(result.componentSeconds).map(([key, value]) => <div className="chip-bar-row" key={key}><span>{key}</span><div><i style={{ width: `${Math.max(2, Math.min(100, (value / result.latencySeconds) * 100))}%` }} /></div><b>{seconds(value)}</b></div>)}</div><ul className="chip-notes">{result.notes.map((note) => <li key={note}>{note}</li>)}</ul></section>
            </div>
          </>}
          <section className="chip-panel chip-variant-panel"><div className="chip-panel-head"><div><p className="section-kicker">Transparent scenarios</p><h4>Compare architecture moves, not magic recommendations.</h4></div><span className="chip-panel-code">{variantRows.length} variants</span></div><div className="chip-table-wrap"><table className="chip-table chip-variant-table"><thead><tr><th>Scenario</th><th>Latency</th><th>Throughput</th><th>Power</th><th>Cost / batch</th><th>Critical path</th><th>Gate</th></tr></thead><tbody>{variantRows.map((row) => <tr key={row.variant.id} className={row.variant.id === activeVariant.id ? "is-selected" : ""} tabIndex={0} onClick={() => setSelectedVariant(row.variant.id)} onKeyDown={(event) => { if (event.key === "Enter") setSelectedVariant(row.variant.id); }}>{row.value ? <><td><b>{row.variant.label}</b><small>{row.variant.changes.join(" · ")}</small></td><td>{seconds(row.value.latencySeconds)}</td><td>{formatNumber(row.value.throughputPerSecond)} /s</td><td>{formatNumber(row.value.powerW)} W</td><td>${row.value.costUsdPerBatch.toFixed(5)}</td><td>{row.value.bottleneck}</td><td><span className={row.value.feasible ? "chip-pass" : "chip-fail"}>{row.value.feasible ? "pass" : "fail"}</span></td></> : <td colSpan={7}>{row.error}</td>}</tr>)}</tbody></table></div><p className="chip-field-note">Scenario deltas are transparent planning assumptions, not vendor projections. Select a row to inspect its gate and metrics above.</p></section>
        </div>
      )}
    </div>
  );
}

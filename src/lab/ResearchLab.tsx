import { useEffect, useMemo, useState, type ChangeEvent } from "react";

import {
  buildRunMatrix,
  CHECKLIST,
  COMPARISON_NOTE,
  createResearchStudy,
  executeRunMatrix,
  normalizeResearchStudy,
  PLANNER_RUN_NOTE,
  protocolReadiness,
  studyCsv,
  studyManifest,
  studyMarkdown,
  summarizeResearchRuns,
  type ResearchProtocol,
  type ResearchRun,
  type ResearchStudy,
  type StudyStatus,
} from "../planner/research";

type ResearchTab = "protocol" | "runs" | "results";

const STORE = "nomo-research-registry-v1";
const MAX_IMPORT_BYTES = 5 * 1024 * 1024;
const MAX_STUDIES = 20;
const STATUS_LABELS: Record<StudyStatus, string> = {
  draft: "Draft",
  preregistered: "Preregistered",
  running: "Running",
  complete: "Complete",
  archived: "Archived",
};

function asObject(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? value as Record<string, unknown> : {};
}

function loadStudies(): ResearchStudy[] {
  if (typeof window === "undefined") return [];
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(STORE) ?? "[]");
    const values = Array.isArray(parsed) ? parsed : Array.isArray(asObject(parsed).studies) ? asObject(parsed).studies as unknown[] : [];
    return values.slice(0, MAX_STUDIES).flatMap((value) => {
      const study = normalizeResearchStudy(value);
      return study ? [{ ...study, runs: buildRunMatrix(study) }] : [];
    });
  } catch {
    return [];
  }
}

function numericValue(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function download(filename: string, body: string, type: string): void {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fileStem(study: ResearchStudy): string {
  const stem = study.protocol.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return stem || "nomo-study";
}

function formatNumber(value: number | null, digits = 2): string {
  return value === null ? "n/a" : value.toLocaleString("en-US", { maximumFractionDigits: digits });
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return <label className="research-field"><span>{label}{hint && <small>{hint}</small>}</span>{children}</label>;
}

function Metric({ label, value, detail, tone }: { label: string; value: string; detail?: string; tone?: "good" | "warn" }) {
  return <div className={`research-metric${tone ? ` is-${tone}` : ""}`}><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>;
}

function statusClass(status: StudyStatus): string {
  return `research-status research-status--${status}`;
}

export default function ResearchLab() {
  const [studies, setStudies] = useState<ResearchStudy[]>(loadStudies);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [tab, setTab] = useState<ResearchTab>("protocol");
  const [notice, setNotice] = useState<string | null>(null);
  const [executing, setExecuting] = useState(false);
  const selectedId = activeId ?? studies[0]?.id ?? null;
  const active = studies.find((study) => study.id === selectedId) ?? null;
  const readiness = useMemo(() => active ? protocolReadiness(active) : null, [active]);
  const summary = useMemo(() => active ? summarizeResearchRuns(active) : null, [active]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORE, JSON.stringify(studies));
    } catch {
      setNotice("The browser could not persist this registry. Export the study to keep a portable copy.");
    }
  }, [studies]);

  const updateActive = (update: (study: ResearchStudy) => ResearchStudy) => {
    if (!selectedId) return;
    setStudies((current) => current.map((study) => study.id === selectedId
      ? { ...update(study), updatedAt: new Date().toISOString() }
      : study));
  };

  const addStudy = () => {
    const study = createResearchStudy();
    study.runs = buildRunMatrix(study);
    setStudies((current) => [study, ...current].slice(0, MAX_STUDIES));
    setActiveId(study.id);
    setTab("protocol");
    setNotice("New study created locally. Fill the protocol before recording results.");
  };

  const setProtocol = <K extends keyof ResearchProtocol>(key: K, value: ResearchProtocol[K]) => {
    if (active && active.status !== "draft") {
      setNotice("Protocol is locked while this study is preregistered, running, complete, or archived. Return it to Draft before changing the plan.");
      return;
    }
    updateActive((study) => {
      const protocol = { ...study.protocol, [key]: value };
      const next = { ...study, protocol };
      return key === "seed" || key === "repetitions" ? { ...next, runs: buildRunMatrix(next) } : next;
    });
  };

  const setChecklist = (id: string, checked: boolean) => updateActive((study) => ({ ...study, checklist: { ...study.checklist, [id]: checked } }));

  const setRun = (id: string, patch: Partial<ResearchRun>) => updateActive((study) => ({
    ...study,
    runs: study.runs.map((run) => run.id === id ? { ...run, ...patch } : run),
  }));

  const setStatus = (status: StudyStatus) => {
    if (status === "preregistered" && readiness && !readiness.ready) {
      setNotice("Finish the protocol fields and checklist before marking this study Preregistered.");
      return;
    }
    updateActive((study) => ({ ...study, status }));
  };

  const removeActive = () => {
    if (!active || !window.confirm(`Delete the local study “${active.protocol.title}”? Export it first if you need it.`)) return;
    setStudies((current) => current.filter((study) => study.id !== active.id));
    setActiveId(null);
    setNotice("Study removed from this browser. An exported copy is not affected.");
  };

  // Runs the planned (still-empty) rows through the in-browser planner. The
  // computation is synchronous, so a short timeout lets the progress state
  // paint first. Computed rows are labeled separately and never overwrite
  // user-entered measurements.
  const executeMatrix = () => {
    if (!active || executing) return;
    setExecuting(true);
    window.setTimeout(() => {
      try {
        const before = active.runs.filter((run) => run.origin === "computed").length;
        const next = executeRunMatrix(active);
        const after = next.runs.filter((run) => run.origin === "computed").length;
        if (next !== active) updateActive(() => next);
        setNotice(after > before
          ? `Run matrix executed: ${after - before} planned row${after - before === 1 ? "" : "s"} now hold planner-computed values. ${PLANNER_RUN_NOTE}`
          : "No planned rows were computable. Rows with user-entered values are never overwritten; check that the study assumptions map to a known pack.");
      } catch (error) {
        setNotice(`Run matrix could not execute: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        setExecuting(false);
      }
    }, 30);
  };

  const importFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (file.size > MAX_IMPORT_BYTES) {
      setNotice("Import refused: files are limited to 5 MB.");
      return;
    }
    try {
      const parsed: unknown = JSON.parse(await file.text());
      const root = asObject(parsed);
      const values = Array.isArray(parsed) ? parsed : Array.isArray(root.studies) ? root.studies as unknown[] : [root.study ?? parsed];
      const imported = values.flatMap((value) => {
        const study = normalizeResearchStudy(value);
        return study ? [{ ...study, runs: buildRunMatrix(study) }] : [];
      });
      if (!imported.length) throw new Error("No valid study found");
      setStudies((current) => [...imported, ...current.filter((study) => !imported.some((item) => item.id === study.id))].slice(0, MAX_STUDIES));
      setActiveId(imported[0]!.id);
      setTab("protocol");
      setNotice(`Imported ${imported.length} study${imported.length === 1 ? "" : "ies"} into this browser.`);
    } catch (error) {
      setNotice(`Import failed: ${error instanceof Error ? error.message : "the file is not valid JSON"}.`);
    }
  };

  const exportSelected = (kind: "manifest" | "markdown" | "csv") => {
    if (!active) return;
    const stem = fileStem(active);
    if (kind === "manifest") download(`${stem}.json`, JSON.stringify(studyManifest(active), null, 2), "application/json");
    if (kind === "markdown") download(`${stem}-preregistration.md`, studyMarkdown(active), "text/markdown;charset=utf-8");
    if (kind === "csv") download(`${stem}-runs.csv`, studyCsv(active), "text/csv;charset=utf-8");
    setNotice("Export created from the current local study state.");
  };

  return (
    <div className="research-lab">
      <header className="research-head">
        <div>
          <p className="section-kicker">Research lab</p>
          <h2>Turn a question into evidence.</h2>
          <p className="lab-lede">Write the protocol first, pair a baseline with a candidate, record what was measured, and export a study another engineer can rerun.</p>
        </div>
        <div className="research-head-actions">
          <button type="button" className="ui-button ui-button--primary ui-button--default" onClick={addStudy}>New study</button>
          <label className="ui-button ui-button--secondary ui-button--default">
            Import JSON
            <input className="research-file" type="file" accept="application/json,.json" onChange={importFile} />
          </label>
        </div>
      </header>

      <div className="research-boundary">
        <strong>Local research registry</strong>
        <span>Studies stay in this browser until you export them. Nomo labels measured rows separately from estimates and does not turn a recorded comparison into proof of silicon performance.</span>
      </div>

      {notice && <div className="research-notice" role="status"><span>{notice}</span><button type="button" aria-label="Dismiss notice" onClick={() => setNotice(null)}>×</button></div>}

      <div className="research-layout">
        <aside className="research-index" aria-label="Study registry">
          <div className="research-index-head"><div><p className="section-kicker">Registry</p><h3>{studies.length} {studies.length === 1 ? "study" : "studies"}</h3></div><button type="button" className="research-plus" onClick={addStudy} aria-label="Create a new study">+</button></div>
          {!studies.length && <p className="research-empty-copy">No studies yet. Start with a question and a decision you want the evidence to support.</p>}
          <div className="research-study-list">
            {studies.map((study) => {
              const studyReadiness = protocolReadiness(study);
              const studySummary = summarizeResearchRuns(study);
              return <button key={study.id} type="button" className={`research-study-item${study.id === selectedId ? " is-on" : ""}`} onClick={() => { setActiveId(study.id); setTab("protocol"); }}>
                <span className="research-study-item-top"><b>{study.protocol.title}</b><i className={statusClass(study.status)}>{STATUS_LABELS[study.status]}</i></span>
                <span className="research-study-item-meta">{studyReadiness.checked}/{studyReadiness.total} protocol checks · {studySummary.measuredCount}/{study.runs.length} measured rows</span>
              </button>;
            })}
          </div>
        </aside>

        <main className="research-workspace">
          {!active || !readiness || !summary ? <div className="research-empty"><p className="section-kicker">Start here</p><h3>Build the study record before you run the study.</h3><p>Create a local study to define the question, conditions, metric, hardware, and measurement plan. The run table will then be generated from its repetition count.</p><button type="button" className="ui-button ui-button--primary ui-button--default" onClick={addStudy}>Create the first study</button></div> : <>
            <div className="research-study-head">
              <div><p className="section-kicker">Active study · <code>{active.id}</code></p><h3>{active.protocol.title}</h3><p className="research-updated">Last edited {formatDate(active.updatedAt)}</p></div>
              <div className="research-study-actions"><label className="research-status-select"><span>Status</span><select value={active.status} onChange={(event) => setStatus(event.target.value as StudyStatus)}>{Object.entries(STATUS_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><button type="button" className="research-delete" onClick={removeActive}>Delete</button></div>
            </div>

            <div className="research-summary-strip" aria-label="Study summary">
              <Metric label="Protocol" value={`${readiness.checked}/${readiness.total}`} detail={readiness.ready ? "ready to preregister" : `${readiness.blocking.length} blocker${readiness.blocking.length === 1 ? "" : "s"}`} tone={readiness.ready ? "good" : "warn"} />
              <Metric label="Run coverage" value={`${summary.measuredCount}/${active.runs.length}`} detail={summary.computedCount ? `${summary.computedCount} planner-computed · not measurements` : "rows with recorded status"} />
              <Metric label="Baseline mean" value={formatNumber(summary.baseline.mean)} detail={active.protocol.primaryMetric || "primary metric"} />
              <Metric label="Candidate mean" value={formatNumber(summary.candidate.mean)} detail={summary.evidence === "measured" ? "measured comparison" : summary.evidence === "computed" ? "planner-computed · not measurements" : "awaiting paired data"} />
            </div>

            <div className="research-tabs" role="tablist" aria-label="Study workspace">
              {(["protocol", "runs", "results"] as const).map((item) => <button key={item} type="button" role="tab" aria-selected={tab === item} className={tab === item ? "is-on" : ""} onClick={() => setTab(item)}>{item === "protocol" ? "01 Protocol" : item === "runs" ? "02 Run matrix" : "03 Results & export"}</button>)}
            </div>

            {tab === "protocol" && <ProtocolView study={active} readiness={readiness} setProtocol={setProtocol} setChecklist={setChecklist} />}
            {tab === "runs" && <RunsView study={active} summary={summary} setRun={setRun} onExecute={executeMatrix} executing={executing} />}
            {tab === "results" && <ResultsView study={active} readiness={readiness} summary={summary} exportSelected={exportSelected} />}
          </>}
        </main>
      </div>
    </div>
  );
}

function ProtocolView({
  study,
  readiness,
  setProtocol,
  setChecklist,
}: {
  study: ResearchStudy;
  readiness: ReturnType<typeof protocolReadiness>;
  setProtocol: <K extends keyof ResearchProtocol>(key: K, value: ResearchProtocol[K]) => void;
  setChecklist: (id: string, checked: boolean) => void;
}) {
  const p = study.protocol;
  return <div className="research-view research-protocol-view">
    {study.status !== "draft" && <div className="research-measurement-note"><strong>Protocol locked</strong><span>This plan is {STATUS_LABELS[study.status].toLowerCase()}. Return the study to Draft before changing its question, conditions, metric, environment, or repetitions.</span></div>}
    <div className="research-two-col">
      <section className="research-panel">
        <div className="research-panel-head"><div><p className="section-kicker">01 / question</p><h4>What are you trying to learn?</h4></div><span className="research-code">PRE-REGISTRATION</span></div>
        <div className="research-fields">
          <Field label="Study title"><input value={p.title} onChange={(event) => setProtocol("title", event.target.value)} /></Field>
          <Field label="Author / team" hint="Optional"><input value={p.author} onChange={(event) => setProtocol("author", event.target.value)} /></Field>
          <Field label="Research question"><textarea rows={3} value={p.question} onChange={(event) => setProtocol("question", event.target.value)} placeholder="Which change should improve which measurable outcome, on what workload?" /></Field>
          <Field label="Hypothesis"><textarea rows={3} value={p.hypothesis} onChange={(event) => setProtocol("hypothesis", event.target.value)} placeholder="Before measuring: I expect ... because ..." /></Field>
        </div>
      </section>

      <section className="research-panel">
        <div className="research-panel-head"><div><p className="section-kicker">02 / decision rule</p><h4>Define the comparison</h4></div><span className="research-code">NO HIDDEN DEFAULTS</span></div>
        <div className="research-fields">
          <Field label="Baseline"><input value={p.baseline} onChange={(event) => setProtocol("baseline", event.target.value)} placeholder="Existing implementation or control" /></Field>
          <Field label="Candidate"><input value={p.candidate} onChange={(event) => setProtocol("candidate", event.target.value)} placeholder="Nomo design or proposed change" /></Field>
          <div className="research-field-pair">
            <Field label="Primary metric" hint="Include a unit"><input value={p.primaryMetric} onChange={(event) => setProtocol("primaryMetric", event.target.value)} placeholder="latency (ms), energy (J), accuracy (%)" /></Field>
            <Field label="Metric direction"><select value={p.direction} onChange={(event) => setProtocol("direction", event.target.value as ResearchProtocol["direction"])}><option value="minimize">Lower is better</option><option value="maximize">Higher is better</option></select></Field>
          </div>
          <Field label="Success threshold" hint="Percentage improvement; optional"><input type="number" min="0" step="0.1" value={p.successThreshold ?? ""} onChange={(event) => setProtocol("successThreshold", numericValue(event.target.value))} placeholder="e.g. 10" /></Field>
          <p className="research-help">If you leave the threshold blank, Nomo will show the observed direction but will not call the decision a pass or fail.</p>
        </div>
      </section>
    </div>

    <section className="research-panel">
      <div className="research-panel-head"><div><p className="section-kicker">03 / environment</p><h4>Make the run repeatable</h4></div><span className="research-code">CONTROL THE INPUTS</span></div>
      <div className="research-fields research-fields--three">
        <Field label="Model / graph"><input value={p.model} onChange={(event) => setProtocol("model", event.target.value)} placeholder="Model name, commit, or graph hash" /></Field>
        <Field label="Workload / trace"><input value={p.workload} onChange={(event) => setProtocol("workload", event.target.value)} placeholder="Dataset, prompt set, trace, or batch" /></Field>
        <Field label="Hardware / runtime"><input value={p.hardware} onChange={(event) => setProtocol("hardware", event.target.value)} placeholder="Device, topology, driver, framework" /></Field>
        <Field label="Measurement method"><textarea rows={3} value={p.method} onChange={(event) => setProtocol("method", event.target.value)} placeholder="Warmup, timer, sampling window, exclusions" /></Field>
        <div className="research-field-pair research-field-pair--compact">
          <Field label="Seed"><input type="number" min="0" step="1" value={p.seed} onChange={(event) => setProtocol("seed", Math.max(0, Math.round(Number(event.target.value) || 0)))} /></Field>
          <Field label="Repetitions"><input type="number" min="1" max="100" step="1" value={p.repetitions} onChange={(event) => setProtocol("repetitions", Math.min(100, Math.max(1, Math.round(Number(event.target.value) || 1))))} /></Field>
        </div>
        <Field label="Notes" hint="Optional"><textarea rows={3} value={p.notes} onChange={(event) => setProtocol("notes", event.target.value)} placeholder="Context another researcher should know" /></Field>
        <Field label="Limitations"><textarea rows={3} value={p.limitations} onChange={(event) => setProtocol("limitations", event.target.value)} placeholder="What this study cannot establish yet" /></Field>
      </div>
    </section>

    <section className="research-panel research-checklist-panel">
      <div className="research-panel-head"><div><p className="section-kicker">04 / readiness</p><h4>Protocol checklist</h4></div><strong className={readiness.ready ? "research-pass" : "research-warn"}>{readiness.checked}/{readiness.total} complete</strong></div>
      <p className="research-help">These are deliberate acknowledgements, not a decorative score. Complete them before selecting Preregistered.</p>
      <div className="research-checklist">
        {CHECKLIST.map((item) => <label key={item.id} className={`research-check${study.checklist[item.id] ? " is-checked" : ""}`}><input type="checkbox" checked={study.checklist[item.id] === true} onChange={(event) => setChecklist(item.id, event.target.checked)} /><span><b>{item.label}</b><small>{item.description}</small></span></label>)}
      </div>
      {!readiness.ready && <div className="research-blockers"><b>Still needed</b><ul>{readiness.blocking.slice(0, 6).map((item) => <li key={item}>{item}</li>)}{readiness.blocking.length > 6 && <li>+ {readiness.blocking.length - 6} more before this is ready</li>}</ul></div>}
    </section>
  </div>;
}

function RunsView({ study, summary, setRun, onExecute, executing }: { study: ResearchStudy; summary: ReturnType<typeof summarizeResearchRuns>; setRun: (id: string, patch: Partial<ResearchRun>) => void; onExecute: () => void; executing: boolean }) {
  const baselineLabel = study.protocol.baseline || "Baseline";
  const candidateLabel = study.protocol.candidate || "Candidate";
  return <div className="research-view research-runs-view">
    <div className="research-run-toolbar"><div><p className="section-kicker">Paired run matrix</p><h4>{study.protocol.repetitions} repetitions · {study.runs.length} planned rows</h4><p className="research-help">Rows share a seed by repetition so the baseline and candidate can be compared fairly. Empty fields mean unavailable, not zero.</p></div><div className="research-run-counts"><span><b>{summary.measuredCount}</b> measured</span><span><b>{summary.computedCount}</b> computed</span><span><b>{summary.excludedCount}</b> excluded</span><span><b>{summary.plannedCount}</b> planned</span></div><button type="button" className="ui-button ui-button--primary ui-button--default" onClick={onExecute} disabled={executing}>{executing ? "Computing…" : "Execute run matrix"}</button></div>
    <div className="research-table-wrap">
      <table className="research-table">
        <thead><tr><th>Condition</th><th>Rep.</th><th>Seed</th><th>Status</th><th>{study.protocol.primaryMetric || "Primary metric"}</th><th>Energy (J)</th><th>Latency (ms)</th><th>Origin</th><th>Source / artifact</th><th>Notes</th></tr></thead>
        <tbody>{study.runs.map((run) => <tr key={run.id} className={run.condition === "candidate" ? "is-candidate" : ""}>
          <td><strong>{run.condition === "baseline" ? baselineLabel : candidateLabel}</strong><small>{run.condition}</small></td>
          <td>{run.repetition}</td><td>{run.seed}</td>
          <td><select value={run.status} onChange={(event) => setRun(run.id, { status: event.target.value as ResearchRun["status"] })}><option value="planned">Planned</option><option value="measured">Measured</option><option value="excluded">Excluded</option></select></td>
          <td><input aria-label={`${run.condition} repetition ${run.repetition} primary metric`} type="number" step="any" value={run.metricValue ?? ""} onChange={(event) => setRun(run.id, { metricValue: numericValue(event.target.value) })} /></td>
          <td><input aria-label={`${run.condition} repetition ${run.repetition} energy`} type="number" step="any" value={run.energyJ ?? ""} onChange={(event) => setRun(run.id, { energyJ: numericValue(event.target.value) })} /></td>
          <td><input aria-label={`${run.condition} repetition ${run.repetition} latency`} type="number" step="any" value={run.latencyMs ?? ""} onChange={(event) => setRun(run.id, { latencyMs: numericValue(event.target.value) })} /></td>
          <td>{run.origin === "computed" ? <span className="research-code" title={run.notes}>planner</span> : <span className="research-code">user</span>}</td>
          <td><input aria-label={`${run.condition} repetition ${run.repetition} source`} value={run.source} onChange={(event) => setRun(run.id, { source: event.target.value })} placeholder="trace, file, URL" /></td>
          <td><input aria-label={`${run.condition} repetition ${run.repetition} notes`} value={run.notes} onChange={(event) => setRun(run.id, { notes: event.target.value })} placeholder="optional" /></td>
        </tr>)}</tbody>
      </table>
    </div>
    <div className="research-measurement-note"><strong>Computed rows</strong><span>{PLANNER_RUN_NOTE} {COMPARISON_NOTE} They are rendered next to — never instead of — user-measured columns, and entering your own value always wins over a computed one.</span></div>
    <div className="research-measurement-note"><strong>Measurement boundary</strong><span>Entering a value records a local observation. It does not calibrate the planner, certify the hardware, or establish task accuracy. Keep the source path or artifact identifier with every measured row.</span></div>
  </div>;
}

function ResultsView({
  study,
  readiness,
  summary,
  exportSelected,
}: {
  study: ResearchStudy;
  readiness: ReturnType<typeof protocolReadiness>;
  summary: ReturnType<typeof summarizeResearchRuns>;
  exportSelected: (kind: "manifest" | "markdown" | "csv") => void;
}) {
  const maxValue = Math.max(summary.baseline.mean ?? 0, summary.candidate.mean ?? 0, 1);
  const baselineWidth = summary.baseline.mean === null ? 0 : Math.max(4, (summary.baseline.mean / maxValue) * 100);
  const candidateWidth = summary.candidate.mean === null ? 0 : Math.max(4, (summary.candidate.mean / maxValue) * 100);
  const resultTone = summary.thresholdMet === true ? "research-result--pass" : summary.thresholdMet === false ? "research-result--fail" : "research-result--neutral";
  return <div className="research-view research-results-view">
    <section className={`research-result ${resultTone}`}><div><p className="section-kicker">Current interpretation</p><h4>{summary.conclusion}</h4><p>{summary.evidence === "measured" ? "Both sides have at least one measured primary-metric row. Inspect the raw rows and sources before making a broader claim." : summary.evidence === "computed" ? `${PLANNER_RUN_NOTE} The comparison is between planner-selected operating points, not hardware measurements. Record measured rows to make the comparison evidence.` : "This is still a preview. Record measured baseline and candidate rows to calculate a comparison."}</p></div><span className="research-evidence-label">{summary.evidence === "measured" ? "Measured rows present" : summary.evidence === "computed" ? "Computed rows · not measurements" : "Preview"}</span></section>

    <div className="research-results-grid">
      <section className="research-panel"><div className="research-panel-head"><div><p className="section-kicker">Comparison</p><h4>Baseline versus candidate</h4></div><span className="research-code">MEAN OF RECORDED ROWS</span></div><div className="research-bar-list"><div className="research-bar-row"><span>{study.protocol.baseline || "Baseline"}</span><div><i style={{ width: `${baselineWidth}%` }} /></div><b>{formatNumber(summary.baseline.mean)}</b></div><div className="research-bar-row"><span>{study.protocol.candidate || "Candidate"}</span><div><i className="is-candidate" style={{ width: `${candidateWidth}%` }} /></div><b>{formatNumber(summary.candidate.mean)}</b></div></div><div className="research-result-facts"><Metric label="Improvement" value={summary.improvementPct === null ? "n/a" : `${summary.improvementPct.toFixed(1)}%`} detail={study.protocol.direction === "minimize" ? "lower metric is better" : "higher metric is better"} tone={summary.improvementPct !== null && summary.improvementPct >= 0 ? "good" : summary.improvementPct !== null ? "warn" : undefined} /><Metric label="Paired rows" value={String(summary.pairedCount)} detail="minimum measured count" /><Metric label="Threshold" value={study.protocol.successThreshold === null ? "not set" : `${study.protocol.successThreshold}%`} detail={summary.thresholdMet === null ? "no pass/fail yet" : summary.thresholdMet ? "met" : "not met"} /></div></section>
      <section className="research-panel"><div className="research-panel-head"><div><p className="section-kicker">Coverage</p><h4>What is actually recorded?</h4></div></div><dl className="research-definition-list"><div><dt>Baseline rows</dt><dd>{summary.baseline.count} measured · {summary.baseline.sourceCount} with sources</dd></div><div><dt>Candidate rows</dt><dd>{summary.candidate.count} measured · {summary.candidate.sourceCount} with sources</dd></div><div><dt>Baseline median</dt><dd>{formatNumber(summary.baseline.median)}</dd></div><div><dt>Candidate median</dt><dd>{formatNumber(summary.candidate.median)}</dd></div><div><dt>Protocol</dt><dd className={readiness.ready ? "research-pass" : "research-warn"}>{readiness.ready ? "ready" : `${readiness.blocking.length} blockers`}</dd></div></dl></section>
    </div>

    <section className="research-panel research-export-panel"><div className="research-panel-head"><div><p className="section-kicker">Handoff</p><h4>Export a portable study package</h4></div><span className="research-code">VERSIONED ARTIFACTS</span></div><p className="research-help">Use the markdown file for review, the JSON manifest for a future Nomo importer, and the CSV for analysis in another tool. The exports preserve the distinction between planned, measured, and excluded rows.</p><div className="research-export-actions"><button type="button" className="ui-button ui-button--primary ui-button--default" onClick={() => exportSelected("markdown")}>Download preregistration.md</button><button type="button" className="ui-button ui-button--secondary ui-button--default" onClick={() => exportSelected("manifest")}>Download manifest.json</button><button type="button" className="ui-button ui-button--secondary ui-button--default" onClick={() => exportSelected("csv")}>Download runs.csv</button></div><p className="research-help">A download is a working research artifact, not a signed registration. Publish it with the code, environment, raw traces, and a clear statement of what remains unmeasured.</p></section>
  </div>;
}

import { useEffect, useMemo, useRef, useState } from "react";

const STORE = "nomo-enterprise-workspace-v1";
const VERSION = 1;
const MAX_PROJECTS = 12;
const MAX_RUNS_PER_PROJECT = 60;

type JsonRecord = Record<string, unknown>;

export interface WorkspaceRun {
  id: string;
  createdAt: string;
  module: string;
  moduleTitle: string;
  mode: string;
  label: string;
  planKey: string;
  plan: JsonRecord;
  settings: JsonRecord;
  locks: JsonRecord;
  objectives: Record<string, number>;
  constraints: Record<string, number>;
  evidence: string;
  evaluated: number;
  durationMs: number;
}

interface WorkspaceProject {
  id: string;
  name: string;
  purpose: string;
  createdAt: string;
  updatedAt: string;
  runs: WorkspaceRun[];
}

interface WorkspaceState {
  version: number;
  projects: WorkspaceProject[];
}

export interface CurrentWorkspaceRun {
  module: string;
  moduleTitle: string;
  mode: string;
  label: string;
  planKey: string;
  plan: JsonRecord;
  settings: JsonRecord;
  locks: JsonRecord;
  objectives: Record<string, number>;
  constraints: Record<string, number>;
  evidence: string;
  evaluated: number;
  durationMs: number;
}

interface WorkspacePanelProps {
  current: CurrentWorkspaceRun | null;
  onClose: () => void;
  onRestore: (run: WorkspaceRun) => void;
}

const EMPTY_STATE: WorkspaceState = { version: VERSION, projects: [] };

function id(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return `${prefix}-${crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function asRecord(value: unknown): JsonRecord | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : null;
}

function loadState(): WorkspaceState {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE) ?? "null") as Partial<WorkspaceState> | null;
    if (!raw || !Array.isArray(raw.projects)) return EMPTY_STATE;
    const projects = raw.projects.filter((project): project is WorkspaceProject => {
      return Boolean(project && typeof project === "object" && typeof project.id === "string" && typeof project.name === "string" && Array.isArray(project.runs));
    }).slice(0, MAX_PROJECTS).map((project) => ({
      id: project.id,
      name: project.name.slice(0, 120) || "Untitled workspace",
      purpose: typeof project.purpose === "string" ? project.purpose.slice(0, 240) : "",
      createdAt: typeof project.createdAt === "string" ? project.createdAt : new Date().toISOString(),
      updatedAt: typeof project.updatedAt === "string" ? project.updatedAt : new Date().toISOString(),
      runs: project.runs.filter((run): run is WorkspaceRun => {
        return Boolean(run && typeof run === "object" && typeof run.id === "string" && typeof run.module === "string" && typeof run.label === "string" && typeof run.planKey === "string" && asRecord(run.plan) && asRecord(run.settings) && asRecord(run.locks) && run.objectives && typeof run.objectives === "object");
      }).slice(0, MAX_RUNS_PER_PROJECT),
    }));
    return { version: VERSION, projects };
  } catch {
    return EMPTY_STATE;
  }
}

function timestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "Unknown time" : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function downloadJson(filename: string, value: unknown): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

function metricEntries(run: WorkspaceRun): Array<[string, number]> {
  return Object.entries(run.objectives).filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]));
}

function metricValue(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1_000_000) return value.toExponential(2);
  if (Math.abs(value) >= 100) return value.toFixed(1);
  return value.toFixed(3);
}

export default function WorkspacePanel({ current, onClose, onRestore }: WorkspacePanelProps) {
  const [state, setState] = useState<WorkspaceState>(() => loadState());
  const [activeId, setActiveId] = useState(() => loadState().projects[0]?.id ?? "");
  const [creating, setCreating] = useState(() => loadState().projects.length === 0);
  const [newName, setNewName] = useState("");
  const [newPurpose, setNewPurpose] = useState("");
  const [notice, setNotice] = useState("");
  const importRef = useRef<HTMLInputElement>(null);

  const project = creating ? null : state.projects.find((candidate) => candidate.id === activeId) ?? state.projects[0] ?? null;
  const runs = project?.runs ?? [];
  const comparison = useMemo(() => runs.length >= 2 ? [runs[1], runs[0]] as const : null, [runs]);

  useEffect(() => {
    try {
      localStorage.setItem(STORE, JSON.stringify(state));
    } catch {
      setNotice("Storage is full or blocked; export the workspace before closing this tab.");
    }
  }, [state]);

  useEffect(() => {
    if (!creating && !activeId && state.projects[0]) setActiveId(state.projects[0].id);
    if (!creating && activeId && !state.projects.some((candidate) => candidate.id === activeId)) setActiveId(state.projects[0]?.id ?? "");
  }, [activeId, creating, state.projects]);

  function createProject() {
    const name = newName.trim() || "Untitled workspace";
    const now = new Date().toISOString();
    const next: WorkspaceProject = { id: id("project"), name, purpose: newPurpose.trim(), createdAt: now, updatedAt: now, runs: [] };
    setState((previous) => ({ ...previous, projects: [next, ...previous.projects].slice(0, MAX_PROJECTS) }));
    setActiveId(next.id);
    setCreating(false);
    setNewName("");
    setNewPurpose("");
    setNotice(`Created ${name}.`);
  }

  function updateProject(patch: Partial<Pick<WorkspaceProject, "name" | "purpose">>) {
    if (!project) return;
    setState((previous) => ({
      ...previous,
      projects: previous.projects.map((candidate) => candidate.id === project.id
        ? { ...candidate, ...patch, updatedAt: new Date().toISOString() }
        : candidate),
    }));
  }

  function saveCurrent() {
    if (!project || !current) return;
    const run: WorkspaceRun = { ...current, id: id("run"), createdAt: new Date().toISOString() };
    setState((previous) => ({
      ...previous,
      projects: previous.projects.map((candidate) => candidate.id === project.id
        ? { ...candidate, updatedAt: run.createdAt, runs: [run, ...candidate.runs].slice(0, MAX_RUNS_PER_PROJECT) }
        : candidate),
    }));
    setNotice(`Saved “${current.label}” to ${project.name}.`);
  }

  function removeRun(runId: string) {
    if (!project) return;
    setState((previous) => ({
      ...previous,
      projects: previous.projects.map((candidate) => candidate.id === project.id
        ? { ...candidate, updatedAt: new Date().toISOString(), runs: candidate.runs.filter((run) => run.id !== runId) }
        : candidate),
    }));
  }

  async function importWorkspace(file: File) {
    try {
      const parsed = JSON.parse(await file.text()) as Partial<WorkspaceState>;
      if (!Array.isArray(parsed.projects)) throw new Error("The file has no workspace projects.");
      const imported = loadImportedProjects(parsed.projects);
      setState({ version: VERSION, projects: imported });
      setActiveId(imported[0]?.id ?? "");
      setCreating(imported.length === 0);
      setNotice(`Imported ${imported.length} workspace${imported.length === 1 ? "" : "s"}.`);
    } catch (error) {
      setNotice(`Import failed: ${error instanceof Error ? error.message : "invalid JSON"}`);
    }
  }

  return (
    <section className="workspace-panel lab-card" aria-labelledby="workspace-title">
      <div className="workspace-head">
        <div>
          <p className="section-kicker">Enterprise workspace</p>
          <h2 id="workspace-title">Keep decisions, not just calculations.</h2>
          <p className="workspace-lede">Name a project, save alternative runs, compare them later, and hand a portable decision record to your team.</p>
        </div>
        <button type="button" className="lab-close workspace-close" onClick={onClose} aria-label="Close workspace">×</button>
      </div>

      <div className="workspace-boundary"><strong>Browser-local workspace</strong><span>Nothing is uploaded or shared yet. Export the record before switching devices.</span></div>

      {!project ? (
        <div className="workspace-create">
          <h3>Start a project workspace</h3>
          <p className="lab-muted">Use one workspace for a model launch, a hardware evaluation, or a cost review.</p>
          <div className="workspace-create-fields">
            <label>Project name<input value={newName} onChange={(event) => setNewName(event.target.value)} placeholder="e.g. Q4 inference refresh" /></label>
            <label>Purpose <span className="lab-muted">(optional)</span><input value={newPurpose} onChange={(event) => setNewPurpose(event.target.value)} placeholder="What decision is this workspace for?" /></label>
          </div>
          <button type="button" className="ui-button ui-button--primary ui-button--compact" onClick={createProject}>Create workspace</button>
        </div>
      ) : (
        <>
          <div className="workspace-toolbar">
            <label className="workspace-select-label">Workspace<select value={project.id} onChange={(event) => { setCreating(false); setActiveId(event.target.value); }}>{state.projects.map((candidate) => <option key={candidate.id} value={candidate.id}>{candidate.name}</option>)}</select></label>
            <button type="button" className="lab-link" onClick={() => { setNewName(""); setNewPurpose(""); setCreating(true); }}>New workspace</button>
          </div>

          <div className="workspace-project-fields">
            <label>Project name<input value={project.name} onChange={(event) => updateProject({ name: event.target.value.slice(0, 120) })} /></label>
            <label>Purpose<input value={project.purpose} onChange={(event) => updateProject({ purpose: event.target.value.slice(0, 240) })} placeholder="The decision this workspace supports" /></label>
          </div>

          <div className="workspace-actions">
            <button type="button" className="ui-button ui-button--primary ui-button--compact" disabled={!current} onClick={saveCurrent}>Save current run</button>
            <button type="button" className="ui-button ui-button--outline ui-button--compact" onClick={() => downloadJson(`${project.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "nomo-workspace"}.json`, state)}>Export workspace</button>
            <button type="button" className="ui-button ui-button--outline ui-button--compact" onClick={() => importRef.current?.click()}>Import workspace</button>
            <input ref={importRef} className="workspace-file" type="file" accept="application/json,.json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importWorkspace(file); event.target.value = ""; }} />
          </div>

          {current ? <div className="workspace-current"><span className="workspace-current-mark" aria-hidden="true">●</span><div><strong>Current Lab result ready to save</strong><span>{current.moduleTitle} · {current.label} · {current.evaluated.toLocaleString("en-US")} plans checked</span></div><span className="workspace-evidence">{current.evidence}</span></div> : <p className="workspace-empty">Open a planner module and choose a result to save it here. Simulation runs remain downloadable from the Simulation core.</p>}

          <div className="workspace-runs-head"><div><h3>Decision history</h3><p className="lab-muted">{runs.length ? `${runs.length} saved run${runs.length === 1 ? "" : "s"}; newest first.` : "No saved runs yet."}</p></div>{runs.length > 0 && <span className="workspace-updated">Updated {timestamp(project.updatedAt)}</span>}</div>
          {runs.length > 0 && (
            <div className="workspace-runs">
              {runs.map((run) => (
                <article className="workspace-run" key={run.id}>
                  <div className="workspace-run-main"><div className="workspace-run-title"><strong>{run.label}</strong><span>{run.moduleTitle} · {run.mode}</span></div><time dateTime={run.createdAt}>{timestamp(run.createdAt)}</time></div>
                  <div className="workspace-run-metrics">{metricEntries(run).map(([name, value]) => <span key={name}><b>{metricValue(value)}</b>{name.replace(/_/g, " ")}</span>)}</div>
                  <div className="workspace-run-foot"><span className="workspace-evidence">{run.evidence}</span><span className="workspace-run-buttons"><button type="button" className="lab-link" onClick={() => onRestore(run)}>Load into Lab</button><button type="button" className="lab-link" onClick={() => removeRun(run.id)}>Remove</button></span></div>
                </article>
              ))}
            </div>
          )}

          {comparison && <div className="workspace-compare"><div className="workspace-runs-head"><div><h3>Latest comparison</h3><p className="lab-muted">Saved values side by side; objective direction depends on the module.</p></div></div><div className="workspace-compare-scroll"><table><thead><tr><th>Objective</th><th>{comparison[0].label}</th><th>{comparison[1].label}</th></tr></thead><tbody>{[...new Set([...metricEntries(comparison[0]).map(([name]) => name), ...metricEntries(comparison[1]).map(([name]) => name)])].map((name) => <tr key={name}><th>{name.replace(/_/g, " ")}</th><td>{comparison[0].objectives[name] === undefined ? "—" : metricValue(comparison[0].objectives[name]!)}</td><td>{comparison[1].objectives[name] === undefined ? "—" : metricValue(comparison[1].objectives[name]!)}</td></tr>)}</tbody></table></div></div>}
        </>
      )}

      {notice && <p className="workspace-notice" role="status">{notice}</p>}
    </section>
  );
}

function loadImportedProjects(value: unknown[]): WorkspaceProject[] {
  return value.filter((project): project is WorkspaceProject => {
    return Boolean(project && typeof project === "object" && typeof (project as WorkspaceProject).id === "string" && typeof (project as WorkspaceProject).name === "string" && Array.isArray((project as WorkspaceProject).runs));
  }).slice(0, MAX_PROJECTS).map((project) => ({
    id: project.id,
    name: project.name.slice(0, 120) || "Untitled workspace",
    purpose: typeof project.purpose === "string" ? project.purpose.slice(0, 240) : "",
    createdAt: typeof project.createdAt === "string" ? project.createdAt : new Date().toISOString(),
    updatedAt: typeof project.updatedAt === "string" ? project.updatedAt : new Date().toISOString(),
    runs: project.runs.filter((run): run is WorkspaceRun => Boolean(run && typeof run === "object" && typeof run.id === "string" && typeof run.module === "string" && typeof run.label === "string" && typeof run.planKey === "string" && asRecord(run.plan) && asRecord(run.settings) && asRecord(run.locks) && run.objectives && typeof run.objectives === "object")).slice(0, MAX_RUNS_PER_PROJECT),
  }));
}

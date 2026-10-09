import { useEffect, useMemo, useRef, useState } from "react";

const STORE = "nomo-enterprise-workspace-v1";
const REMOTE_STORE = "nomo-enterprise-remote-v1";
const REMOTE_TOKEN_STORE = "nomo-enterprise-remote-token-v1";
const VERSION = 1;
const MAX_PROJECTS = 12;
const MAX_RUNS_PER_PROJECT = 60;
const MAX_NOTES_PER_PROJECT = 100;
type ReviewState = "draft" | "review" | "approved";

type JsonRecord = Record<string, unknown>;

export interface WorkspaceRun {
  id: string;
  createdAt: string;
  module: string;
  moduleTitle: string;
  mode: string;
  label: string;
  planKey: string;
  review: ReviewState;
  remoteId?: string;
  remoteReview?: ReviewState;
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
  remoteId?: string;
  runs: WorkspaceRun[];
  notes: WorkspaceNote[];
}

interface WorkspaceNote {
  id: string;
  createdAt: string;
  module: string;
  moduleTitle: string;
  text: string;
  remoteId?: string;
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
  context: { module: string; moduleTitle: string };
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
      remoteId: typeof project.remoteId === "string" ? project.remoteId : undefined,
      runs: project.runs.filter((run): run is WorkspaceRun => {
        return Boolean(run && typeof run === "object" && typeof run.id === "string" && typeof run.module === "string" && typeof run.label === "string" && typeof run.planKey === "string" && asRecord(run.plan) && asRecord(run.settings) && asRecord(run.locks) && run.objectives && typeof run.objectives === "object");
      }).slice(0, MAX_RUNS_PER_PROJECT).map((run) => ({ ...run, review: validReview(run.review), remoteId: typeof run.remoteId === "string" ? run.remoteId : undefined, remoteReview: run.remoteReview ? validReview(run.remoteReview) : undefined })),
      notes: loadNotes(project.notes),
    }));
    return { version: VERSION, projects };
  } catch {
    return EMPTY_STATE;
  }
}

function loadRemoteConfig(): { baseUrl: string; token: string } {
  try {
    return {
      baseUrl: localStorage.getItem(REMOTE_STORE) ?? "",
      token: sessionStorage.getItem(REMOTE_TOKEN_STORE) ?? "",
    };
  } catch {
    return { baseUrl: "", token: "" };
  }
}

function loadNotes(value: unknown): WorkspaceNote[] {
  if (!Array.isArray(value)) return [];
  return value.filter((note): note is WorkspaceNote => {
    return Boolean(note && typeof note === "object" && typeof (note as WorkspaceNote).id === "string" && typeof (note as WorkspaceNote).createdAt === "string" && typeof (note as WorkspaceNote).module === "string" && typeof (note as WorkspaceNote).moduleTitle === "string" && typeof (note as WorkspaceNote).text === "string");
  }).slice(0, MAX_NOTES_PER_PROJECT).map((note) => ({
    id: note.id,
    createdAt: note.createdAt,
    module: note.module,
    moduleTitle: note.moduleTitle,
    text: note.text.slice(0, 2000),
    remoteId: typeof note.remoteId === "string" ? note.remoteId : undefined,
  }));
}

function saveRemoteConfig(baseUrl: string, token: string): void {
  try {
    localStorage.setItem(REMOTE_STORE, baseUrl);
    if (token) sessionStorage.setItem(REMOTE_TOKEN_STORE, token);
    else sessionStorage.removeItem(REMOTE_TOKEN_STORE);
  } catch {
    // Browser storage is optional; the connection still works for this tab.
  }
}

async function remoteRequest(baseUrl: string, token: string, path: string, init: RequestInit = {}): Promise<unknown> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${baseUrl.replace(/\/+$/, "")}${path}`, { ...init, headers });
  const text = await response.text();
  let value: unknown = null;
  try { value = text ? JSON.parse(text) : null; } catch { value = text; }
  if (!response.ok) {
    const detail = value && typeof value === "object" && "error" in value ? String((value as { error: unknown }).error) : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  return value;
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

function validReview(value: unknown): ReviewState {
  return value === "review" || value === "approved" ? value : "draft";
}

export default function WorkspacePanel({ current, context, onClose, onRestore }: WorkspacePanelProps) {
  const [state, setState] = useState<WorkspaceState>(() => loadState());
  const [activeId, setActiveId] = useState(() => loadState().projects[0]?.id ?? "");
  const [creating, setCreating] = useState(() => loadState().projects.length === 0);
  const [remoteUrl, setRemoteUrl] = useState(() => loadRemoteConfig().baseUrl);
  const [remoteToken, setRemoteToken] = useState(() => loadRemoteConfig().token);
  const [remoteStatus, setRemoteStatus] = useState<"idle" | "checking" | "connected" | "error">("idle");
  const [remoteDescription, setRemoteDescription] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [newName, setNewName] = useState("");
  const [newPurpose, setNewPurpose] = useState("");
  const [noteText, setNoteText] = useState("");
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
    const next: WorkspaceProject = { id: id("project"), name, purpose: newPurpose.trim(), createdAt: now, updatedAt: now, runs: [], notes: [] };
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
    const run: WorkspaceRun = { ...current, id: id("run"), createdAt: new Date().toISOString(), review: "draft" };
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

  function addNote() {
    if (!project) return;
    const text = noteText.trim().slice(0, 2000);
    if (!text) return;
    const note: WorkspaceNote = { id: id("note"), createdAt: new Date().toISOString(), module: context.module, moduleTitle: context.moduleTitle, text };
    setState((previous) => ({
      ...previous,
      projects: previous.projects.map((candidate) => candidate.id === project.id
        ? { ...candidate, updatedAt: note.createdAt, notes: [note, ...candidate.notes].slice(0, MAX_NOTES_PER_PROJECT) }
        : candidate),
    }));
    setNoteText("");
    setNotice(`Added a note from ${context.moduleTitle}.`);
  }

  function removeNote(noteId: string) {
    if (!project) return;
    setState((previous) => ({
      ...previous,
      projects: previous.projects.map((candidate) => candidate.id === project.id
        ? { ...candidate, updatedAt: new Date().toISOString(), notes: candidate.notes.filter((note) => note.id !== noteId) }
        : candidate),
    }));
    setNotice("Note removed from this browser workspace. A previously synced copy is not deleted remotely.");
  }

  function updateReview(runId: string, review: ReviewState) {
    if (!project) return;
    setState((previous) => ({
      ...previous,
      projects: previous.projects.map((candidate) => candidate.id === project.id
        ? { ...candidate, updatedAt: new Date().toISOString(), runs: candidate.runs.map((run) => run.id === runId ? { ...run, review } : run) }
        : candidate),
    }));
    setNotice("Review flag updated locally. It is not a signed approval.");
  }

  function exportWorkspace() {
    return {
      schema_version: VERSION,
      exported_at: new Date().toISOString(),
      source: "nomo-browser-workspace",
      version: state.version,
      projects: state.projects,
    };
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

  async function connectRemote() {
    const baseUrl = remoteUrl.trim().replace(/\/+$/, "");
    if (!baseUrl) {
      setRemoteStatus("error");
      setRemoteDescription("Enter the platform API URL first.");
      return;
    }
    setRemoteStatus("checking");
    setRemoteDescription("Checking the platform capabilities…");
    try {
      const capabilities = await remoteRequest(baseUrl, remoteToken.trim(), "/capabilities") as { service?: string; version?: string; operations?: unknown };
      if (capabilities.service !== "nomo-platform" || !Array.isArray(capabilities.operations)) throw new Error("Endpoint is reachable but is not the Nomo platform API.");
      saveRemoteConfig(baseUrl, remoteToken.trim());
      setRemoteUrl(baseUrl);
      setRemoteStatus("connected");
      setRemoteDescription(`Connected to Nomo platform ${capabilities.version ?? "unknown"}.`);
    } catch (error) {
      setRemoteStatus("error");
      setRemoteDescription(error instanceof Error ? error.message : "Could not connect to the platform API.");
    }
  }

  async function syncRemote() {
    if (!project || remoteStatus !== "connected") return;
    const baseUrl = remoteUrl.trim().replace(/\/+$/, "");
    if (!baseUrl) return;
    setSyncing(true);
    try {
      let remoteProjectId = project.remoteId;
      if (!remoteProjectId) {
        const created = await remoteRequest(baseUrl, remoteToken.trim(), "/projects", {
          method: "POST",
          body: JSON.stringify({
            name: project.name,
            description: project.purpose,
            metadata: { source: "nomo-browser-workspace", local_project_id: project.id },
          }),
        }) as { id?: string };
        if (!created.id) throw new Error("Platform did not return a project id.");
        remoteProjectId = created.id;
      }

      const unsyncedRuns = runs.filter((run) => !run.remoteId);
      const changedRuns = runs.filter((run) => run.remoteId && run.remoteReview !== run.review);
      const unsyncedNotes = project.notes.filter((note) => !note.remoteId);
      const remoteRunIds = new Map<string, string>();
      const remoteNoteIds = new Map<string, string>();
      for (const run of unsyncedRuns) {
        const created = await remoteRequest(baseUrl, remoteToken.trim(), `/projects/${encodeURIComponent(remoteProjectId)}/runs`, {
          method: "POST",
          body: JSON.stringify({
            name: run.label,
            config: { module: run.module, mode: run.mode, plan_key: run.planKey, plan: run.plan, settings: run.settings, locks: run.locks },
            result: { objectives: run.objectives, constraints: run.constraints, evidence: run.evidence, evaluated: run.evaluated, duration_ms: run.durationMs },
            metadata: { source: "nomo-browser-workspace", local_run_id: run.id, review: run.review, schema_version: 1 },
          }),
        }) as { id?: string };
        if (!created.id) throw new Error(`Platform did not return a run id for ${run.label}.`);
        remoteRunIds.set(run.id, created.id);
      }
      for (const run of changedRuns) {
        await remoteRequest(baseUrl, remoteToken.trim(), `/runs/${encodeURIComponent(run.remoteId!)}`, {
          method: "PATCH",
          body: JSON.stringify({
            metadata: { source: "nomo-browser-workspace", local_run_id: run.id, review: run.review, schema_version: 1 },
          }),
        });
      }
      for (const note of unsyncedNotes) {
        const created = await remoteRequest(baseUrl, remoteToken.trim(), `/projects/${encodeURIComponent(remoteProjectId)}/artifacts`, {
          method: "POST",
          body: JSON.stringify({
            name: `workspace-note-${note.id}.json`,
            content: { schema_version: 1, source: "nomo-browser-workspace", note },
            media_type: "application/json",
            metadata: { kind: "workspace-note", local_note_id: note.id, module: note.module },
          }),
        }) as { id?: string };
        if (!created.id) throw new Error(`Platform did not return an artifact id for the note from ${note.moduleTitle}.`);
        remoteNoteIds.set(note.id, created.id);
      }

      const syncedAt = new Date().toISOString();
      setState((previous) => ({
        ...previous,
        projects: previous.projects.map((candidate) => candidate.id === project.id ? {
          ...candidate,
          remoteId: remoteProjectId,
          updatedAt: syncedAt,
          runs: candidate.runs.map((run) => remoteRunIds.has(run.id) ? { ...run, remoteId: remoteRunIds.get(run.id), remoteReview: run.review } : changedRuns.some((changed) => changed.id === run.id) ? { ...run, remoteReview: run.review } : run),
          notes: candidate.notes.map((note) => remoteNoteIds.has(note.id) ? { ...note, remoteId: remoteNoteIds.get(note.id) } : note),
        } : candidate),
      }));
      setRemoteDescription(`Synced ${unsyncedRuns.length} new and ${changedRuns.length} updated run${unsyncedRuns.length + changedRuns.length === 1 ? "" : "s"}, plus ${unsyncedNotes.length} note${unsyncedNotes.length === 1 ? "" : "s"}.`);
      setNotice(`Workspace “${project.name}” is synced to the platform.`);
    } catch (error) {
      setRemoteStatus("error");
      setRemoteDescription(error instanceof Error ? error.message : "Workspace sync failed.");
    } finally {
      setSyncing(false);
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

      <div className="workspace-boundary"><strong>{project?.remoteId ? "Local + shared copy" : "Browser-local workspace"}</strong><span>{project?.remoteId ? "This project has a platform copy; new runs still sync only when you choose Sync." : "Nothing is uploaded or shared yet. Export the record before switching devices."}</span></div>

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
            <button type="button" className="ui-button ui-button--outline ui-button--compact" onClick={() => downloadJson(`${project.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "nomo-workspace"}.json`, exportWorkspace())}>Export workspace</button>
            <button type="button" className="ui-button ui-button--outline ui-button--compact" onClick={() => importRef.current?.click()}>Import workspace</button>
            <input ref={importRef} className="workspace-file" type="file" accept="application/json,.json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importWorkspace(file); event.target.value = ""; }} />
          </div>

          <div className="workspace-remote">
            <div className="workspace-runs-head"><div><h3>Optional shared platform</h3><p className="lab-muted">Connect an authenticated Nomo platform API when this decision needs team persistence. The token stays in this browser session and is never exported.</p></div><span className={`workspace-remote-status is-${remoteStatus}`}>{remoteStatus === "connected" ? "Connected" : remoteStatus === "checking" ? "Checking…" : remoteStatus === "error" ? "Needs attention" : "Not connected"}</span></div>
            <div className="workspace-remote-fields"><label>Platform URL<input value={remoteUrl} onChange={(event) => { setRemoteUrl(event.target.value); setRemoteStatus("idle"); }} placeholder="https://platform.example" /></label><label>Bearer token <span className="lab-muted">(optional if server is local)</span><input type="password" value={remoteToken} onChange={(event) => { setRemoteToken(event.target.value); setRemoteStatus("idle"); }} placeholder="Stored for this session only" autoComplete="off" /></label></div>
            <div className="workspace-remote-actions"><button type="button" className="ui-button ui-button--outline ui-button--compact" disabled={remoteStatus === "checking"} onClick={() => void connectRemote()}>{remoteStatus === "checking" ? "Connecting…" : "Check connection"}</button><button type="button" className="ui-button ui-button--primary ui-button--compact" disabled={remoteStatus !== "connected" || syncing} onClick={() => void syncRemote()}>{syncing ? "Syncing…" : project.remoteId ? "Sync new runs" : "Create shared project"}</button></div>
            {remoteDescription && <p className={`workspace-remote-note is-${remoteStatus}`} role="status">{remoteDescription}</p>}
          </div>

          {current ? <div className="workspace-current"><span className="workspace-current-mark" aria-hidden="true">●</span><div><strong>Current Lab result ready to save</strong><span>{current.moduleTitle} · {current.label} · {current.evaluated.toLocaleString("en-US")} plans checked</span></div><span className="workspace-evidence">{current.evidence}</span></div> : <p className="workspace-empty">Open a planner or Simulation module and choose a result to save it here.</p>}

          <div className="workspace-journal"><div className="workspace-runs-head"><div><h3>Review journal</h3><p className="lab-muted">Capture a decision, risk, measurement request, or follow-up from {context.moduleTitle}. Notes are local until you sync them.</p></div><span className="workspace-updated">{project.notes.length} note{project.notes.length === 1 ? "" : "s"}</span></div><div className="workspace-note-compose"><textarea value={noteText} onChange={(event) => setNoteText(event.target.value.slice(0, 2000))} placeholder="What should the team remember about this module?" rows={3} /><div className="workspace-note-compose-foot"><span className="lab-muted">{noteText.length}/2,000 · context: {context.moduleTitle}</span><button type="button" className="ui-button ui-button--outline ui-button--compact" disabled={!noteText.trim()} onClick={addNote}>Add note</button></div></div>{project.notes.length > 0 && <div className="workspace-notes">{project.notes.map((note) => <article className="workspace-note" key={note.id}><div className="workspace-note-head"><span>{note.moduleTitle}</span><time dateTime={note.createdAt}>{timestamp(note.createdAt)}</time></div><p>{note.text}</p><button type="button" className="lab-link" onClick={() => removeNote(note.id)}>Remove local note</button></article>)}</div>}</div>

          <div className="workspace-runs-head"><div><h3>Decision history</h3><p className="lab-muted">{runs.length ? `${runs.length} saved run${runs.length === 1 ? "" : "s"}; newest first.` : "No saved runs yet."}</p></div>{runs.length > 0 && <span className="workspace-updated">Updated {timestamp(project.updatedAt)}</span>}</div>
          {runs.length > 0 && (
            <div className="workspace-runs">
              {runs.map((run) => (
                <article className="workspace-run" key={run.id}>
                  <div className="workspace-run-main"><div className="workspace-run-title"><strong>{run.label}</strong><span>{run.moduleTitle} · {run.mode}</span></div><time dateTime={run.createdAt}>{timestamp(run.createdAt)}</time></div>
                  <div className="workspace-run-metrics">{metricEntries(run).map(([name, value]) => <span key={name}><b>{metricValue(value)}</b>{name.replace(/_/g, " ")}</span>)}</div>
                  <div className="workspace-run-foot"><span className="workspace-evidence">{run.evidence}</span><span className="workspace-run-buttons"><label className="workspace-review">Review<select aria-label={`Review state for ${run.label}`} value={run.review} onChange={(event) => updateReview(run.id, event.target.value as ReviewState)}><option value="draft">Draft</option><option value="review">Needs review</option><option value="approved">Approved locally</option></select></label><button type="button" className="lab-link" onClick={() => onRestore(run)}>{run.module === "simulation" ? "Open simulation" : "Load into Lab"}</button><button type="button" className="lab-link" onClick={() => removeRun(run.id)}>Remove</button></span></div>
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
    remoteId: typeof project.remoteId === "string" ? project.remoteId : undefined,
    runs: project.runs.filter((run): run is WorkspaceRun => Boolean(run && typeof run === "object" && typeof run.id === "string" && typeof run.module === "string" && typeof run.label === "string" && typeof run.planKey === "string" && asRecord(run.plan) && asRecord(run.settings) && asRecord(run.locks) && run.objectives && typeof run.objectives === "object")).slice(0, MAX_RUNS_PER_PROJECT).map((run) => ({ ...run, review: validReview(run.review), remoteId: typeof run.remoteId === "string" ? run.remoteId : undefined, remoteReview: run.remoteReview ? validReview(run.remoteReview) : undefined })),
    notes: loadNotes(project.notes),
  }));
}

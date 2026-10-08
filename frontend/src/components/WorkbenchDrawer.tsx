"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { formatSI } from "@/lib/pareto";
import { mergeConfig, readError, startRun } from "@/lib/runConfig";
import type { DesignDetail, EvalItem, HardwareIn } from "@/lib/telemetry/protocol";
import type { RunMeta } from "@/lib/useRunMeta";
import { useRunStore } from "@/lib/telemetry/store";

import { Button, Disclosure, DomainChip, NumberField } from "./ui";
import EnterpriseValidation from "./EnterpriseValidation";

type WorkbenchNode = {
  id: string;
  label?: string;
  kind?: string;
  domain?: string;
  color?: string;
  thermal_color?: string;
  op?: string;
  params?: number;
  macs?: number;
  contract?: string;
  cost_score?: number;
  memory_score?: number;
  quantization_sensitivity?: number;
  cycle?: number;
  event?: string;
  path?: string;
  power_density?: number;
  [key: string]: unknown;
};

type WorkbenchLevel = { level: number; title: string; description: string; nodes: WorkbenchNode[]; edges: unknown[] };
type WorkbenchState = {
  format?: string;
  revision?: number;
  model?: string;
  levels: Record<string, WorkbenchLevel>;
  architecture?: { title?: string; family?: string; [key: string]: unknown };
  mode?: { title?: string; summary?: string; requirements?: string[]; export_tags?: string[] } | null;
  target_weights?: number[];
  cost_function?: { formula?: string; coefficients?: Record<string, number> };
  metrics?: {
    energy_j?: number;
    latency_s?: number;
    accuracy_pct?: number;
    accuracy_source?: string;
    area_um2?: number | null;
    ppa_source?: string;
  };
  ptq?: Record<string, unknown> | null;
  hitl?: { target_id?: string; latency_ms?: number; energy_uj?: number | null; samples?: number; energy_source?: string } | null;
  selected_design_key?: string;
  error?: string;
};

type WeightTuple = [number, number, number];
type EvidenceKind = "proxy" | "simulated" | "measured";
type SelectionSource = "default" | "priority" | "manual";

const LEVELS = [
  ["system_topology", "01"], ["partitioning", "02"], ["hardware_graph", "03"],
  ["cycle_emulation", "04"], ["rtl", "05"], ["silicon_floorplan", "06"],
] as const;

const PRIORITIES = [
  { key: 0 as const, label: "Energy", help: "Prefer lower modeled energy." },
  { key: 1 as const, label: "Response time", help: "Prefer lower modeled latency." },
  { key: 2 as const, label: "Accuracy", help: "Prefer higher task accuracy." },
];

const HARDWARE_FIELDS: { key: keyof HardwareIn; label: string; unit: string; step: number }[] = [
  { key: "mac_energy_pj", label: "MAC energy", unit: "pJ", step: 0.01 },
  { key: "sop_energy_pj", label: "Synaptic op energy", unit: "pJ", step: 0.01 },
  { key: "neuron_energy_pj", label: "Neuron update", unit: "pJ", step: 0.01 },
  { key: "sram_kb_per_core", label: "SRAM per core", unit: "KB", step: 1 },
  { key: "neurons_per_core", label: "Neurons per core", unit: "count", step: 1 },
  { key: "n_cores", label: "Processing cores", unit: "count", step: 1 },
  { key: "bus_bandwidth_gbs", label: "Bus bandwidth", unit: "GB/s", step: 0.01 },
  { key: "routing_latency_us", label: "Routing latency", unit: "μs", step: 0.01 },
  { key: "timestep_us", label: "Minimum timestep", unit: "μs", step: 0.01 },
  { key: "static_power_mw", label: "Static power", unit: "mW", step: 0.01 },
  { key: "clock_mhz", label: "Clock", unit: "MHz", step: 1 },
];

const LABELS: Record<string, string> = {
  proxy: "Proxy",
  simulated: "Simulated",
  measured: "Measured",
};

function evidenceClass(kind: EvidenceKind): string {
  if (kind === "measured") return "border-sym bg-sym-tint text-sym-ink";
  if (kind === "simulated") return "border-ann bg-ann-tint text-ann-ink";
  return "border-line-strong bg-paper text-ink-muted";
}

function EvidenceTag({ kind, detail }: { kind: EvidenceKind; detail?: string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-2xs font-bold ${evidenceClass(kind)}`}>
      <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" />
      {LABELS[kind]}{detail ? ` · ${detail}` : ""}
    </span>
  );
}

function weightLabel(weights: WeightTuple): string {
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((value, index) => `${PRIORITIES[index]!.label} ${Math.round((value / total) * 100)}%`).join(" · ");
}

function candidateId(key: string): string {
  return key.length > 16 ? `${key.slice(0, 8)}…${key.slice(-5)}` : key;
}

function hardwareSnapshot(meta: RunMeta): Record<string, unknown> {
  const config = meta.config;
  const profile = config && meta.catalog ? meta.catalog.hardware[config.hardware] : undefined;
  return {
    id: config?.hardware ?? null,
    name: profile?.name ?? null,
    provenance: profile?.provenance ?? {},
    defaults: profile?.defaults ?? {},
    overrides: config?.hardware_overrides ?? null,
  };
}

function formatNodeValue(node: WorkbenchNode, key: string): string | null {
  const value = node[key];
  if (value === undefined || value === null || value === "") return null;
  if (typeof value === "number") return value.toFixed(2);
  return String(value);
}

function evidenceForAccuracy(detail: DesignDetail | null, state: WorkbenchState | null): EvidenceKind {
  const source = detail?.design.metrics.accuracy_source ?? state?.metrics?.accuracy_source;
  return source === "oracle" || source === "measured" ? "measured" : "proxy";
}

function evidenceForRuntime(state: WorkbenchState | null): EvidenceKind {
  return state?.hitl?.latency_ms !== undefined || state?.hitl?.energy_uj !== undefined ? "measured" : "simulated";
}

function evidenceForCalibration(state: WorkbenchState | null): EvidenceKind {
  if (!state?.ptq) return "proxy";
  return state.ptq.status === "unavailable" ? "proxy" : "measured";
}

function EvidencePanel({ state, detail }: { state: WorkbenchState | null; detail: DesignDetail | null }) {
  const accuracy = evidenceForAccuracy(detail, state);
  const runtime = evidenceForRuntime(state);
  const calibration = evidenceForCalibration(state);
  const hasHitl = Boolean(state?.hitl);
  return (
    <section aria-label="Validation and evidence" className="rounded-lg border border-line bg-paper p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-bold">Evidence ledger</h3>
          <p className="mt-0.5 text-2xs text-ink-muted">Every number is labelled by how it was obtained.</p>
        </div>
        <span className="text-2xs font-mono text-ink-muted">rev {state?.revision ?? "—"}</span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <div className="rounded-md border border-line bg-panel p-2"><p className="text-2xs text-ink-muted">Task accuracy</p><EvidenceTag kind={accuracy} detail={accuracy === "proxy" ? "sensitivity" : "oracle"} /></div>
        <div className="rounded-md border border-line bg-panel p-2"><p className="text-2xs text-ink-muted">Energy / latency</p><EvidenceTag kind={runtime} detail={hasHitl ? state?.hitl?.target_id : "hardware model"} /></div>
        <div className="rounded-md border border-line bg-panel p-2"><p className="text-2xs text-ink-muted">Calibration fidelity</p><EvidenceTag kind={calibration} detail={state?.ptq ? "calibration tensors" : "not attached"} /></div>
        <div className="rounded-md border border-line bg-panel p-2"><p className="text-2xs text-ink-muted">Silicon PPA / thermal</p><EvidenceTag kind="proxy" detail="synthesis required" /></div>
      </div>
      <p className="mt-3 text-2xs leading-relaxed text-ink-muted">
        {hasHitl ? "Hardware-in-the-loop telemetry is attached for this run." : "No hardware-in-the-loop telemetry is attached; energy, timing, and floorplan values are planning estimates."}
      </p>
    </section>
  );
}

function WhySelected({ item, detail, weights, recommended, source }: {
  item: EvalItem | undefined;
  detail: DesignDetail | null;
  weights: WeightTuple;
  recommended: boolean;
  source: SelectionSource;
}) {
  if (!item) return <p className="text-sm text-ink-muted">Select a candidate to see the decision record.</p>;
  const strongest = weights.indexOf(Math.max(...weights));
  const priority = PRIORITIES[strongest]!.label.toLowerCase();
  const lead = source === "priority"
    ? "Workbench selected this candidate from the feasible frontier after you applied new priorities"
    : source === "manual"
      ? "You selected this candidate for inspection"
      : recommended
        ? "Nomo's baseline recommendation is this candidate"
        : "This candidate is active for inspection";
  return (
    <div className="space-y-2 text-sm">
      <p className="leading-relaxed text-ink-soft">
        {lead}. {item.feasible ? "It stays within the run limits" : "It exceeds at least one run limit"}, while the strongest current preference is {priority}.
      </p>
      <div className="flex flex-wrap gap-1.5"><span className="rounded-full border border-line px-2 py-0.5 text-2xs font-bold">{weightLabel(weights)}</span><span className="rounded-full border border-line px-2 py-0.5 text-2xs">{item.feasible ? "Within limits" : "Limit exceeded"}</span></div>
      {detail?.summary.text && <p className="border-l-2 border-ann pl-3 text-2xs leading-relaxed text-ink-muted">{detail.summary.text}</p>}
    </div>
  );
}

export default function WorkbenchDrawer({ runId, meta, detail, onClose }: {
  runId: string;
  meta: RunMeta;
  detail: DesignDetail | null;
  onClose: () => void;
}) {
  const router = useRouter();
  const [state, setState] = useState<WorkbenchState | null>(null);
  const [active, setActive] = useState<string>(LEVELS[0][0]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectionSource, setSelectionSource] = useState<SelectionSource>("default");
  const [weights, setWeights] = useState<WeightTuple>([1, 1, 1]);
  const [compareKeys, setCompareKeys] = useState<string[]>([]);
  const [hardwareValues, setHardwareValues] = useState<HardwareIn>({});
  const [hardwareReady, setHardwareReady] = useState(false);
  const [hardwareBusy, setHardwareBusy] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const weightsRef = useRef<WeightTuple>(weights);
  const compareInitialized = useRef(false);
  const selectedKey = useRunStore((s) => s.selectedKey);
  const recommendedKey = useRunStore((s) => s.recommendedKey);
  const items = useRunStore((s) => s.items);
  const front = useRunStore((s) => s.front);
  const version = useRunStore((s) => s.version);
  const select = useRunStore((s) => s.select);
  const setInspect = useRunStore((s) => s.setInspect);

  useEffect(() => { weightsRef.current = weights; }, [weights]);

  const candidates = useMemo(() => {
    const keys = new Set<string>(front);
    if (recommendedKey) keys.add(recommendedKey);
    if (selectedKey) keys.add(selectedKey);
    return Array.from(keys).map((key) => items.get(key)).filter((item): item is NonNullable<typeof item> => Boolean(item))
      .sort((a, b) => (a.rank - b.rank) || (a.f[0] - b.f[0]));
  }, [front, items, recommendedKey, selectedKey, version]);

  const level = useMemo(() => state?.levels[active] ?? null, [active, state]);
  const selectedItem = selectedKey ? items.get(selectedKey) : undefined;
  const compareItems = useMemo(() => compareKeys.map((key) => items.get(key)).filter((item): item is NonNullable<typeof item> => Boolean(item)), [compareKeys, items, version]);
  const profile = meta.config && meta.catalog ? meta.catalog.hardware[meta.config.hardware] : undefined;
  const unsupported = Object.entries(detail?.capabilities ?? {}).filter(([, capability]) => !capability.available);
  const selectedStateKey = state?.selected_design_key ?? selectedKey ?? recommendedKey;
  const hardwareDirty = useMemo(() => {
    if (!profile) return false;
    return HARDWARE_FIELDS.some(({ key }) => hardwareValues[key] !== profile.defaults[key as string]);
  }, [hardwareValues, profile]);

  useEffect(() => {
    if (compareInitialized.current || candidates.length === 0) return;
    const first = selectedKey ?? recommendedKey ?? candidates[0]?.key;
    setCompareKeys(Array.from(new Set([first, ...candidates.map((candidate) => candidate.key)])).filter((key): key is string => Boolean(key)).slice(0, 3));
    compareInitialized.current = true;
  }, [candidates, recommendedKey, selectedKey]);

  useEffect(() => {
    if (hardwareReady || !meta.config || !profile) return;
    setHardwareValues({ ...profile.defaults, ...(meta.config.hardware_overrides ?? {}) });
    setHardwareReady(true);
  }, [hardwareReady, meta.config, profile]);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(`nomo.workbench.${runId}`);
      if (saved) setSavedAt((JSON.parse(saved) as { saved_at?: string }).saved_at ?? null);
    } catch { /* local persistence is optional */ }
  }, [runId]);

  useEffect(() => {
    let alive = true;
    const query = new URLSearchParams({
      energy_weight: String(weightsRef.current[0]),
      latency_weight: String(weightsRef.current[1]),
      accuracy_weight: String(weightsRef.current[2]),
    });
    if (selectedKey) query.set("key", selectedKey);
    setLoading(true);
    setError(null);
    api(`/runs/${runId}/workbench?${query.toString()}`).then(async (response) => {
      if (!response.ok) throw new Error(await readError(response));
      return response.json() as Promise<WorkbenchState>;
    }).then((value) => {
      if (!alive) return;
      setState(value);
      setLoading(false);
    }).catch((reason) => {
      if (!alive) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      setLoading(false);
    });
    return () => { alive = false; };
  }, [runId, selectedKey]);

  const applyPriorities = async () => {
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const response = await api(`/runs/${runId}/workbench/targets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ energy: weights[0], latency: weights[1], accuracy: weights[2] }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const next = await response.json() as WorkbenchState;
      setState(next);
      const key = next.selected_design_key;
      if (key) {
        setSelectionSource("priority");
        select(key);
      }
      setNotice(`Candidate ${key ? candidateId(key) : "selection"} selected using the new priorities.`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  };

  const activateCandidate = (key: string) => {
    setSelectionSource("manual");
    select(key);
    setCompareKeys((current) => current.includes(key) ? current : [key, ...current].slice(0, 3));
    setNotice(null);
  };

  const toggleCompare = (key: string) => {
    setCompareKeys((current) => {
      if (current.includes(key)) return current.filter((candidate) => candidate !== key);
      if (current.length >= 3) {
        setNotice("Compare up to three candidates at a time.");
        return current;
      }
      return [...current, key];
    });
  };

  const snapshot = useMemo(() => ({
    format: "nomo.workbench-config/1",
    run_id: runId,
    saved_at: new Date().toISOString(),
    selected_design_key: selectedStateKey ?? null,
    target_weights: weights,
    run: meta.config,
    hardware_profile: hardwareSnapshot(meta),
    hardware_values: hardwareValues,
    evidence: {
      accuracy: evidenceForAccuracy(detail, state),
      runtime: evidenceForRuntime(state),
      calibration: evidenceForCalibration(state),
      ppa: "proxy",
    },
  }), [detail, meta, runId, selectedStateKey, state, weights]);

  const saveSnapshot = () => {
    try {
      const next = { ...snapshot, saved_at: new Date().toISOString() };
      window.localStorage.setItem(`nomo.workbench.${runId}`, JSON.stringify(next, null, 2));
      setSavedAt(next.saved_at);
      setNotice("Run configuration saved in this browser.");
    } catch {
      setError("Could not save locally. Browser storage may be disabled.");
    }
  };

  const downloadSnapshot = () => {
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `nomo-${runId}-workbench.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Run configuration JSON downloaded.");
  };

  const updateHardware = (key: keyof HardwareIn, value: number | null) => {
    setHardwareValues((current) => {
      const next = { ...current };
      if (value === null) delete next[key];
      else next[key] = value;
      return next;
    });
  };

  const startHardwareVariant = async () => {
    if (!meta.config) return;
    setHardwareBusy(true);
    setError(null);
    setNotice(null);
    try {
      const id = await startRun(mergeConfig(meta.config, { hardware_overrides: hardwareValues }));
      router.push(`/runs/${id}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      setHardwareBusy(false);
    }
  };

  return (
    <aside aria-label="Nomo Workbench" className="nomo-drawer flex h-full flex-col bg-panel">
      <div className="flex items-center justify-between border-b border-line px-5 py-4">
        <div><p className="text-2xs font-bold uppercase tracking-widest text-ink-muted">Nomo OS canvas</p><h2 className="text-lg font-bold">Workbench</h2></div>
        <button onClick={onClose} aria-label="Close workbench" className="text-xl leading-none text-ink-muted hover:text-ink">×</button>
      </div>

      <div className="border-b border-line px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <div><p className="text-2xs font-bold uppercase tracking-widest text-ink-muted">Engineering decision record</p><p className="mt-1 text-sm text-ink-soft">Compare, constrain, and preserve the selected design.</p></div>
          {loading && <span className="text-2xs text-ink-muted" role="status">Updating…</span>}
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-4">
        {error && <p role="alert" className="rounded-md border border-cross bg-cross-tint p-3 text-sm">{error}</p>}
        {notice && <p role="status" className="rounded-md border border-sym bg-sym-tint p-3 text-sm text-sym-ink">{notice}</p>}
        {!state && !error && <p className="text-sm text-ink-muted">Reading the six-level canvas…</p>}

        <EvidencePanel state={state} detail={detail} />

        <EnterpriseValidation runId={runId} selectedKey={selectedKey} />

        <Disclosure title="Candidate comparison" summary={`${candidates.length} frontier candidates`} defaultOpen>
          <div className="space-y-2">
            <p className="text-2xs leading-relaxed text-ink-muted">Select a candidate to make it active. Use the checkboxes to compare up to three candidates in the table below.</p>
            {candidates.length === 0 && <p className="rounded-md border border-line p-3 text-sm text-ink-muted">No completed candidates are available yet.</p>}
            {candidates.map((candidate) => {
              const activeCandidate = candidate.key === selectedKey;
              const accuracyKind: EvidenceKind = candidate.acc_src === "oracle" ? "measured" : "proxy";
              return (
                <article key={candidate.key} className={`rounded-md border p-3 ${activeCandidate ? "border-ink bg-paper" : "border-line bg-panel"}`}>
                  <div className="flex items-start gap-2">
                    <input type="checkbox" aria-label={`Compare candidate ${candidateId(candidate.key)}`} checked={compareKeys.includes(candidate.key)} onChange={() => toggleCompare(candidate.key)} className="mt-1 h-4 w-4 accent-[#2F5BEA]" />
                    <button type="button" onClick={() => activateCandidate(candidate.key)} className="min-w-0 flex-1 text-left">
                      <span className="flex flex-wrap items-center gap-1.5"><span className="font-mono text-2xs text-ink-muted">{candidateId(candidate.key)}</span>{candidate.key === recommendedKey && <span className="rounded-full bg-ann-tint px-2 py-0.5 text-2xs font-bold text-ann-ink">recommended</span>}{activeCandidate && <span className="rounded-full bg-ink px-2 py-0.5 text-2xs font-bold text-white">active</span>}</span>
                      <span className="mt-1 block text-sm font-bold">{candidate.feasible ? "Feasible frontier design" : "Constraint-breaking design"}</span>
                    </button>
                    <span className="text-2xs text-ink-muted">rank {candidate.rank}</span>
                  </div>
                  <div className="mt-3 grid grid-cols-3 gap-2 text-2xs">
                    <span><span className="block text-ink-muted">Energy</span><strong>{formatSI(candidate.f[0], "J")}</strong></span>
                    <span><span className="block text-ink-muted">Latency</span><strong>{formatSI(candidate.f[1], "s")}</strong></span>
                    <span><span className="block text-ink-muted">Accuracy</span><strong>{candidate.f[2].toFixed(1)}%</strong><span className="block mt-0.5"><EvidenceTag kind={accuracyKind} /></span></span>
                  </div>
                </article>
              );
            })}
            {compareItems.length > 0 && (
              <div className="mt-3 overflow-x-auto rounded-md border border-line">
                <table className="w-full min-w-[380px] text-left text-2xs">
                  <caption className="border-b border-line bg-paper px-3 py-2 text-left font-bold text-ink">Selected comparison</caption>
                  <thead><tr className="border-b border-line text-ink-muted"><th className="px-3 py-2 font-normal">Candidate</th><th className="px-2 py-2 font-normal">Energy</th><th className="px-2 py-2 font-normal">Latency</th><th className="px-2 py-2 font-normal">Accuracy</th></tr></thead>
                  <tbody>{compareItems.map((candidate) => <tr key={candidate.key} className="border-b border-line last:border-0"><th scope="row" className="px-3 py-2 font-mono font-normal">{candidateId(candidate.key)}{candidate.key === selectedKey ? " · active" : ""}</th><td className="px-2 py-2">{formatSI(candidate.f[0], "J")}</td><td className="px-2 py-2">{formatSI(candidate.f[1], "s")}</td><td className="px-2 py-2">{candidate.f[2].toFixed(1)}%</td></tr>)}</tbody>
                </table>
              </div>
            )}
          </div>
        </Disclosure>

        <Disclosure title="Selection priorities" summary={weightLabel(weights)} defaultOpen>
          <div className="space-y-4">
            <p className="text-2xs leading-relaxed text-ink-muted">These weights change which feasible candidate Nomo selects from the completed frontier. They do not change the underlying model or hardware estimates.</p>
            {PRIORITIES.map((priority) => <label key={priority.label} className="block"><span className="flex justify-between gap-3 text-sm"><span className="font-bold">{priority.label}</span><span className="font-mono text-ink-muted">{weights[priority.key]}/5</span></span><input aria-label={`${priority.label} priority`} type="range" min={1} max={5} step={1} value={weights[priority.key]} onChange={(event) => setWeights((current) => { const next: WeightTuple = [...current]; next[priority.key] = Number(event.target.value); return next; })} className="mt-1 w-full" /><span className="block text-2xs text-ink-muted">{priority.help}</span></label>)}
            <div className="flex flex-wrap gap-2"><Button onClick={applyPriorities} disabled={loading}>Apply priorities</Button><Button kind="quiet" onClick={() => setWeights([1, 1, 1])}>Reset</Button></div>
          </div>
        </Disclosure>

        <Disclosure title="Hardware profile assumptions" summary={profile?.name ?? meta.config?.hardware ?? "Loading profile"}>
          {!profile && <p className="text-sm text-ink-muted">Hardware profile details will appear when the run metadata is available.</p>}
          {profile && (
            <div className="space-y-4">
              <div className="rounded-md border border-line bg-paper p-3 text-sm"><p className="font-bold">{profile.name}</p><p className="mt-1 text-2xs text-ink-muted">Profile id: <span className="font-mono">{meta.config?.hardware}</span></p><p className="mt-2 text-2xs leading-relaxed text-ink-muted">{Object.entries(profile.provenance).map(([key, value]) => `${key}: ${value}`).join(" · ")}</p></div>
              <p className="text-2xs leading-relaxed text-ink-muted">Edit the assumptions below to start a separate run. The completed run above remains unchanged.</p>
              <div className="grid grid-cols-2 gap-3">{HARDWARE_FIELDS.map((field) => <NumberField key={field.key} label={field.label} value={hardwareValues[field.key] ?? null} onChange={(value) => updateHardware(field.key, value)} unit={field.unit} step={field.step} min={0} />)}</div>
              <div className="flex flex-wrap items-center gap-2"><Button onClick={startHardwareVariant} disabled={hardwareBusy || !meta.config}>{hardwareBusy ? "Starting…" : "Start run with assumptions"}</Button><Button kind="quiet" onClick={() => setHardwareValues({ ...profile.defaults, ...(meta.config?.hardware_overrides ?? {}) })}>Reset values</Button>{hardwareDirty && <span className="text-2xs font-bold text-ann-ink">Unsaved hardware changes</span>}</div>
            </div>
          )}
        </Disclosure>

        <Disclosure title="Why this candidate?" summary={selectedItem ? candidateId(selectedItem.key) : "No active candidate"} defaultOpen>
          <WhySelected item={selectedItem} detail={detail} weights={weights} recommended={selectedItem?.key === recommendedKey} source={selectionSource} />
        </Disclosure>

        <Disclosure title="Per-layer detail" summary={detail?.design.layers.length ? `${detail.design.layers.length} layers` : "Select a candidate"} defaultOpen>
          {!detail && <p className="text-sm text-ink-muted">Select a candidate and its layer-level costs will appear here.</p>}
          {detail && <div className="overflow-x-auto rounded-md border border-line"><table className="w-full min-w-[460px] text-left text-2xs"><caption className="border-b border-line bg-paper px-3 py-2 text-left text-ink-muted">Click a row to open the existing layer inspector.</caption><thead><tr className="border-b border-line text-ink-muted"><th className="px-3 py-2 font-normal">Layer</th><th className="px-2 py-2 font-normal">Style</th><th className="px-2 py-2 font-normal">Precision</th><th className="px-2 py-2 font-normal">Energy</th><th className="px-2 py-2 font-normal">Memory</th></tr></thead><tbody>{detail.design.layers.map((layer, index) => <tr key={layer.name} className="border-b border-line last:border-0"><td className="px-3 py-2"><button type="button" onClick={() => setInspect(index)} className="text-left font-bold text-ann-ink hover:underline">{layer.name}</button><span className="block text-ink-muted">{layer.op}</span></td><td className="px-2 py-2"><DomainChip d={layer.domain} small /></td><td className="px-2 py-2">{layer.domain === "SYM" ? "fixed" : `${layer.w_bits}/${layer.a_bits}`}</td><td className="px-2 py-2">{layer.energy_j == null ? "—" : formatSI(layer.energy_j, "J")}</td><td className="px-2 py-2">{layer.memory_bytes == null ? "—" : `${(layer.memory_bytes / 1024).toFixed(1)} KB`}</td></tr>)}</tbody></table></div>}
        </Disclosure>

        <Disclosure title="Six-level canvas" summary={level?.title ?? "System topology"}>
          <div className="grid grid-cols-3 gap-1.5">
            {LEVELS.map(([id, number]) => <button key={id} type="button" onClick={() => setActive(id)} aria-pressed={active === id} className={`rounded-md border px-2 py-2 text-left text-2xs font-bold ${active === id ? "border-ink bg-ink text-white" : "border-line hover:border-line-strong"}`}><span className="block font-mono">{number}</span><span className="mt-1 block leading-tight">{id.replaceAll("_", " ")}</span></button>)}
          </div>
          {level && <><p className="mt-4 text-2xs font-bold uppercase tracking-widest text-ink-muted">Level {level.level}</p><h3 className="mt-1 text-xl font-bold">{level.title}</h3><p className="mt-1 text-sm text-ink-muted">{level.description}</p><div className="mt-4 grid gap-2">{level.nodes.map((node) => <article key={node.id} className="rounded-md border border-line bg-paper p-3" style={{ borderLeftColor: node.color ?? node.thermal_color ?? "#1d2433", borderLeftWidth: 4 }}><div className="flex items-start justify-between gap-3"><h4 className="font-bold">{node.label ?? node.id}</h4>{node.domain && <span className="text-2xs text-ink-muted">{node.domain}</span>}</div><p className="mt-1 text-2xs text-ink-muted">{node.kind ? String(node.kind) : node.op ? String(node.op) : node.event ? String(node.event) : "canvas node"}</p><div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-2xs text-ink-muted">{([["cost", "cost_score"], ["memory", "memory_score"], ["sensitivity", "quantization_sensitivity"], ["path", "path"], ["cycle", "cycle"]] as const).map(([label, key]) => { const value = formatNodeValue(node, key); return value ? <span key={key}><span className="capitalize">{label}</span>: <strong className="text-ink">{value}</strong></span> : null; })}</div></article>)}</div>{state?.architecture && <p className="mt-5 border-t border-line pt-3 text-2xs text-ink-muted">Architecture adapter: {state.architecture.title ?? state.architecture.family ?? "unknown"}. Workbench metrics remain labelled by evidence status above.</p>}</>}
        </Disclosure>

        <Disclosure title="Support boundaries" summary={unsupported.length ? `${unsupported.length} output${unsupported.length === 1 ? "" : "s"} unavailable` : "Selected outputs available"}>
          {unsupported.length === 0 ? <p className="text-sm text-sym-ink">All advertised output formats for this candidate are available.</p> : <div className="space-y-2">{unsupported.map(([key, capability]) => <div key={key} className="rounded-md border border-dashed border-line-strong bg-paper p-3"><p className="text-sm font-bold">{capability.label}</p><p className="mt-1 text-2xs leading-relaxed text-cross">Unavailable: {capability.reason ?? "this backend path is not implemented for the selected design"}.</p></div>)}</div>}
        </Disclosure>

        <Disclosure title="Run configuration" summary={savedAt ? `Saved ${new Date(savedAt).toLocaleString()}` : "Local snapshot and JSON export"}>
          <div className="space-y-3"><p className="text-2xs leading-relaxed text-ink-muted">Save a browser-local snapshot or download a portable JSON decision record. This does not upload model data.</p><div className="flex flex-wrap gap-2"><Button onClick={saveSnapshot} disabled={!meta.config}>Save locally</Button><Button kind="secondary" onClick={downloadSnapshot} disabled={!meta.config}>Download JSON</Button></div><details><summary className="cursor-pointer text-2xs font-bold text-ann-ink">Preview decision record</summary><pre className="mt-2 max-h-48 overflow-auto rounded-md bg-ink p-3 text-2xs text-white">{JSON.stringify(snapshot, null, 2)}</pre></details></div>
        </Disclosure>
      </div>
    </aside>
  );
}

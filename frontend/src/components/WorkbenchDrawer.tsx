"use client";

import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";

type WorkbenchNode = { id: string; label?: string; kind?: string; domain?: string; color?: string; thermal_color?: string; [key: string]: unknown };
type WorkbenchLevel = { level: number; title: string; description: string; nodes: WorkbenchNode[]; edges: unknown[] };
type WorkbenchState = { levels: Record<string, WorkbenchLevel>; architecture: { title: string }; metrics: Record<string, unknown>; error?: string };

const LEVELS = [
  ["system_topology", "01"], ["partitioning", "02"], ["hardware_graph", "03"],
  ["cycle_emulation", "04"], ["rtl", "05"], ["silicon_floorplan", "06"],
] as const;

export default function WorkbenchDrawer({ runId, onClose }: { runId: string; onClose: () => void }) {
  const [state, setState] = useState<WorkbenchState | null>(null);
  const [active, setActive] = useState<string>(LEVELS[0][0]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api(`/runs/${runId}/workbench`).then(async (r) => {
      if (!r.ok) throw new Error(await r.text());
      return r.json() as Promise<WorkbenchState>;
    }).then((value) => { if (alive) setState(value); }).catch((x) => { if (alive) setError(x instanceof Error ? x.message : String(x)); });
    return () => { alive = false; };
  }, [runId]);

  const level = useMemo(() => state?.levels[active] ?? null, [active, state]);
  return (
    <aside aria-label="Nomo Workbench" className="nomo-drawer flex h-full flex-col bg-panel">
      <div className="flex items-center justify-between border-b border-line px-5 py-4">
        <div><p className="text-2xs font-bold uppercase tracking-widest text-ink-muted">Nomo OS canvas</p><h2 className="text-lg font-bold">Workbench</h2></div>
        <button onClick={onClose} aria-label="Close workbench" className="text-xl leading-none text-ink-muted hover:text-ink">×</button>
      </div>
      <div className="border-b border-line px-4 py-3">
        <div className="grid grid-cols-3 gap-1.5">
          {LEVELS.map(([id, number]) => <button key={id} type="button" onClick={() => setActive(id)} aria-pressed={active === id}
            className={`rounded-md border px-2 py-2 text-left text-2xs font-bold ${active === id ? "border-ink bg-ink text-white" : "border-line hover:border-line-strong"}`}>
            <span className="block font-mono">{number}</span><span className="mt-1 block leading-tight">{id.replaceAll("_", " ")}</span>
          </button>)}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-5 py-4">
        {!state && !error && <p className="text-sm text-ink-muted">Reading the six-level canvas…</p>}
        {error && <p role="alert" className="rounded-md border border-cross bg-cross-tint p-3 text-sm">{error}</p>}
        {level && (
          <>
            <p className="text-2xs font-bold uppercase tracking-widest text-ink-muted">Level {level.level}</p>
            <h3 className="mt-1 text-xl font-bold">{level.title}</h3>
            <p className="mt-1 text-sm text-ink-muted">{level.description}</p>
            <div className="mt-4 grid gap-2">
              {level.nodes.map((node) => <article key={node.id} className="rounded-md border border-line bg-paper p-3" style={{ borderLeftColor: node.color ?? node.thermal_color ?? "#1d2433", borderLeftWidth: 4 }}>
                <div className="flex items-start justify-between gap-3"><h4 className="font-bold">{node.label ?? node.id}</h4>{node.domain && <span className="text-2xs text-ink-muted">{node.domain}</span>}</div>
                <p className="mt-1 text-2xs text-ink-muted">{node.kind ? String(node.kind) : "canvas node"}</p>
              </article>)}
            </div>
            {state?.architecture && <p className="mt-5 border-t border-line pt-3 text-2xs text-ink-muted">Architecture adapter: {state.architecture.title}. Metrics stay marked proxy until calibration or HITL telemetry is attached.</p>}
          </>
        )}
      </div>
    </aside>
  );
}

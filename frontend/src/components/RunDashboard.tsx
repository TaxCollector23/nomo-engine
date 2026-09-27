"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useState } from "react";

import { api } from "@/lib/api";
import { modelTitle } from "@/lib/models";
import { useDesign, useRunMeta } from "@/lib/useRunMeta";
import { apiBase } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";
import { useRunTelemetry } from "@/lib/telemetry/useRunTelemetry";

import CopilotDrawer from "./CopilotDrawer";
import DesignPanel from "./DesignPanel";
import ExportDrawer from "./ExportDrawer";
import FilterBar from "./FilterBar";
import LayerInspector from "./LayerInspector";
import PartitionGraph from "./PartitionGraph";
import RunHUD from "./RunHUD";
import WorkbenchDrawer from "./WorkbenchDrawer";
import { DomainChip } from "./ui";

// WebGL must not render on the server
const ParetoFront3D = dynamic(() => import("./ParetoFront3D"), { ssr: false });

export default function RunDashboard({ runId }: { runId: string }) {
  useRunTelemetry(runId);
  const status = useRunStore((s) => s.status);
  const conn = useRunStore((s) => s.connection);
  const run = useRunStore((s) => s.run);
  const key = useRunStore((s) => s.selectedKey);
  const drawer = useRunStore((s) => s.drawer);
  const setDrawer = useRunStore((s) => s.setDrawer);
  const meta = useRunMeta(runId);
  const { detail } = useDesign(runId, key, status);
  const [pendingQ, setPendingQ] = useState<string | null>(null);
  const done = status === "completed" || status === "stopped";

  const stop = () => api(`/runs/${runId}/stop`, { method: "POST" });
  const askCopilot = (q: string) => { setPendingQ(`${q}\u200b${Date.now()}`); setDrawer("copilot"); };

  if (conn === "not_found") {
    return (
      <main className="mx-auto max-w-xl p-10">
        <h1 className="text-xl font-bold">This search no longer exists</h1>
        <p className="mt-2 text-ink-muted">The server at {apiBase()} does not have run {runId}. Free hosting forgets searches when it restarts.</p>
        <Link href="/" className="mt-4 inline-block font-bold text-ann-ink hover:underline">Start a new search</Link>
      </main>
    );
  }

  const title = `${modelTitle(run?.model ?? meta.config?.model ?? "", meta.modelName ?? undefined)} on ${meta.catalog?.hardware[run?.hardware ?? ""]?.name ?? run?.hardware ?? "…"}`;

  return (
    <div className="flex h-screen flex-col">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-panel px-5 py-3">
        <div className="flex min-w-0 items-center gap-4">
          <Link href="/" className="text-sm font-bold text-ann-ink hover:underline">Nomo</Link>
          <h1 className="truncate font-bold">{title}</h1>
          <span className="hidden gap-1.5 md:flex"><DomainChip d="ANN" small /><DomainChip d="SNN" small /><DomainChip d="SYM" small /></span>
        </div>
        <div className="flex items-center gap-2">
          {status === "running" && (
            <button onClick={stop} className="rounded-md border border-line-strong px-3 py-1.5 text-sm font-bold hover:border-ink">Stop search</button>
          )}
          <button onClick={() => setDrawer(drawer === "copilot" ? null : "copilot")} aria-pressed={drawer === "copilot"}
            className="rounded-md border border-line-strong px-3 py-1.5 text-sm font-bold hover:border-ink">Ask Copilot</button>
          <button onClick={() => setDrawer(drawer === "workbench" ? null : "workbench")} aria-pressed={drawer === "workbench"} disabled={!done}
            title={done ? undefined : "Available when the search finishes"}
            className="rounded-md border border-line-strong px-3 py-1.5 text-sm font-bold hover:border-ink disabled:opacity-40">Workbench</button>
          <button onClick={() => setDrawer(drawer === "export" ? null : "export")} aria-pressed={drawer === "export"} disabled={!done}
            title={done ? undefined : "Available when the search finishes"}
            className="rounded-md bg-ink px-3 py-1.5 text-sm font-bold text-white hover:bg-ink-soft disabled:opacity-40">Export</button>
        </div>
      </header>

      {meta.warnings.length > 0 && (
        <p className="border-b border-line bg-snn-tint px-5 py-2 text-sm">{meta.warnings.join(" ")}</p>
      )}

      <div className="flex min-h-0 flex-1">
        <main className="grid min-h-0 min-w-0 flex-1 grid-rows-[minmax(0,1fr)_auto]">
          <section className="relative min-h-0 min-w-0 overflow-hidden border-b border-line">
            <ParetoFront3D />
            <div className="absolute inset-x-3 top-3"><FilterBar /></div>
          </section>
          <section className="relative z-10 min-w-0 bg-panel px-4 pb-3 pt-2">
            <p className="mb-1 text-sm text-ink-muted">
              How the selected design runs, layer by layer. Click a layer to inspect or lock it. Red badges mark where data changes style.
            </p>
            <PartitionGraph pins={meta.config?.pins ?? {}} />
          </section>
        </main>

        <aside className="relative z-10 w-[340px] shrink-0 space-y-6 overflow-y-auto border-l border-line bg-panel p-5">
          <LayerInspector meta={meta} detail={detail} onAsk={askCopilot} />
          <DesignPanel detail={detail} />
          <div className="border-t border-line pt-5"><RunHUD /></div>
        </aside>

        {drawer && (
          <div className="relative z-20 w-[380px] shrink-0 border-l border-line shadow-[-8px_0_24px_-12px_rgba(29,36,51,0.25)]">
            {drawer === "copilot" && (
              <CopilotDrawer runId={runId} status={status} config={meta.config} pending={pendingQ}
                onClose={() => setDrawer(null)} />
            )}
            {drawer === "export" && <ExportDrawer runId={runId} detail={detail} onClose={() => setDrawer(null)} />}
            {drawer === "workbench" && <WorkbenchDrawer runId={runId} onClose={() => setDrawer(null)} />}
          </div>
        )}
      </div>
    </div>
  );
}

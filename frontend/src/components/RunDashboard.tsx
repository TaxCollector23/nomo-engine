"use client";

import dynamic from "next/dynamic";
import Link from "next/link";

import { api } from "@/lib/api";
import { apiBase } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";
import { useRunTelemetry } from "@/lib/telemetry/useRunTelemetry";

import CandidatePanel from "./CandidatePanel";
import PartitionGraph from "./PartitionGraph";
import RunHUD from "./RunHUD";

// WebGL must not render on the server
const ParetoFront3D = dynamic(() => import("./ParetoFront3D"), { ssr: false });

export default function RunDashboard({ runId }: { runId: string }) {
  useRunTelemetry(runId);
  const status = useRunStore((s) => s.status);
  const conn = useRunStore((s) => s.connection);

  const stop = () => api(`/runs/${runId}/stop`, { method: "POST" });

  if (conn === "not_found") {
    return (
      <main className="p-8 font-mono text-sm text-neutral-300">
        Run <b>{runId}</b> not found on {apiBase()}. <Link href="/" className="underline">Start a new run</Link>.
      </main>
    );
  }

  return (
    <main className="grid h-screen grid-cols-[1fr_340px] grid-rows-[minmax(0,1fr)_230px] bg-black text-neutral-100">
      <section className="relative border-b border-r border-neutral-900">
        <ParetoFront3D />
      </section>
      <aside className="row-span-2 space-y-6 overflow-y-auto p-5">
        <div className="flex items-center justify-between font-mono text-[10px] text-neutral-500">
          <Link href="/" className="hover:text-neutral-200">← nomo</Link>
          <span>run {runId}</span>
          {status === "running" && (
            <button onClick={stop} className="border border-neutral-700 px-2 py-0.5 hover:border-neutral-300 hover:text-neutral-200">stop</button>
          )}
        </div>
        <RunHUD />
        <CandidatePanel />
      </aside>
      <section className="border-r border-neutral-900 p-3">
        <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-neutral-500">
          tri-domain partition · white = ANN · dashed grey = SNN · black = symbolic · bright edges = domain crossings
        </div>
        <div className="h-[calc(100%-18px)]"><PartitionGraph /></div>
      </section>
    </main>
  );
}

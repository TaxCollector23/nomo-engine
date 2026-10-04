"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { applyPreset, mergeConfig, readError, startRun } from "@/lib/runConfig";
import type { CopilotAction, CopilotAnswer, Preset, RunIn, RunStatus } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";

interface Msg { role: "user" | "copilot"; text: string; actions?: CopilotAction[]; source?: string }

const STARTERS = [
  "Explain this design in plain English",
  "Explain the trade-offs in plain English",
  "How do I cut energy by another 20%?",
  "Make it 30% faster",
];

export default function CopilotDrawer({ runId, status, config, pending, onClose }: {
  runId: string; status: RunStatus; config: RunIn | null; pending: string | null; onClose: () => void;
}) {
  const router = useRouter();
  const key = useRunStore((s) => s.selectedKey);
  const select = useRunStore((s) => s.select);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  const done = status === "completed" || status === "stopped";
  const sent = useRef<string | null>(null);

  const ask = async (question: string) => {
    if (!question.trim() || busy) return;
    setMsgs((m) => [...m, { role: "user", text: question }]);
    setQ("");
    if (!done) {
      setMsgs((m) => [...m, { role: "copilot", text: "I can answer once the search has finished. You can press Stop to finish early." }]);
      return;
    }
    setBusy(true);
    try {
      const r = await api(`/runs/${runId}/copilot`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, key }) });
      if (!r.ok) throw new Error(await readError(r));
      const a = (await r.json()) as CopilotAnswer;
      setMsgs((m) => [...m, { role: "copilot", text: a.text, actions: a.actions, source: a.source }]);
    } catch (x) {
      setMsgs((m) => [...m, { role: "copilot", text: `Something went wrong: ${x instanceof Error ? x.message : String(x)}` }]);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (pending && pending !== sent.current) { sent.current = pending; void ask(pending.split("\u200b")[0]!); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [msgs]);

  const act = async (a: CopilotAction) => {
    if (a.type === "select") { select(a.key); return; }
    if (a.type === "ask") { void ask(a.text); return; }
    if (!config) return;
    try {
      let next = config;
      if (a.type === "rerun") next = mergeConfig(config, a.settings);
      if (a.type === "apply_preset") {
        const presets = (await (await api("/presets")).json()) as Record<string, Preset>;
        if (!presets[a.preset]) throw new Error(`unknown preset ${a.preset}`);
        next = applyPreset(config, a.preset, presets[a.preset]!);
      }
      setMsgs((m) => [...m, { role: "copilot", text: "Starting a new search with those settings…" }]);
      router.push(`/runs/${await startRun({ ...next, preset: a.type === "apply_preset" ? a.preset : null })}`);
    } catch (x) {
      setMsgs((m) => [...m, { role: "copilot", text: `Could not start the search: ${x instanceof Error ? x.message : String(x)}` }]);
    }
  };

  return (
    <aside aria-label="Nomo Copilot" className="nomo-drawer flex h-full flex-col bg-panel">
      <div className="flex items-center justify-between border-b border-line px-5 py-4">
        <div>
          <h2 className="text-lg font-bold">Nomo Copilot</h2>
          <p className="text-2xs text-ink-muted">Answers are computed from this search, not guessed.</p>
        </div>
        <button onClick={onClose} aria-label="Close Copilot" className="text-xl leading-none text-ink-muted hover:text-ink">×</button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4" aria-live="polite">
        {msgs.length === 0 && (
          <div className="space-y-2">
            <p className="text-sm text-ink-muted">Ask about the selected design, or try one of these:</p>
            {STARTERS.map((s) => (
              <button key={s} onClick={() => ask(s)} className="block w-full rounded-lg border border-line px-3 py-2 text-left text-sm hover:border-ink">{s}</button>
            ))}
            <p className="pt-2 text-2xs text-ink-muted">Tip: click a layer in the graph and ask &quot;why is it spiking?&quot;</p>
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "ml-8 rounded-lg bg-ink px-3 py-2 text-sm text-white" : "mr-4 text-sm leading-relaxed"}>
            <p>{m.text}</p>
            {m.actions && m.actions.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {m.actions.map((a, j) => (
                  <button key={j} onClick={() => act(a)}
                    className={`rounded-md px-2.5 py-1 text-sm font-bold ${a.type === "rerun" || a.type === "apply_preset" ? "bg-ann text-white hover:bg-ann-ink" : "border border-line-strong hover:border-ink"}`}>
                    {a.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {busy && <p className="text-sm text-ink-muted">Working it out…</p>}
        <div ref={end} />
      </div>
      <form onSubmit={(e) => { e.preventDefault(); void ask(q); }} className="flex gap-2 border-t border-line px-5 py-4">
        <label htmlFor="copilot-q" className="sr-only">Ask Copilot</label>
        <input id="copilot-q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Why is fc1 spiking?"
          className="min-w-0 flex-1 rounded-md border border-line px-3 py-2 text-sm outline-none focus:border-ann" maxLength={1000} />
        <button type="submit" disabled={busy || !q.trim()} className="rounded-md bg-ink px-3 py-2 text-sm font-bold text-white disabled:opacity-40">Ask</button>
      </form>
    </aside>
  );
}

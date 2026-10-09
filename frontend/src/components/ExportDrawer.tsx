"use client";

import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import { readError } from "@/lib/runConfig";
import type { DesignDetail } from "@/lib/telemetry/protocol";

import { Button } from "./ui";

const ORDER = ["pdf", "pytorch", "onnx", "nir", "c11", "coreml", "design"] as const;
const WHAT: Record<string, string> = {
  pdf: "A short report for decision makers: savings, trade-offs, chip usage.",
  pytorch: "A Python script that rebuilds this design and checks itself on first run.",
  onnx: "Standard model files for ONNX Runtime or TensorRT.",
  nir: "For neuromorphic tools such as Lava, snnTorch, Rockpool or Nengo.",
  c11: "One C header for microcontrollers; no libraries needed.",
  coreml: "For Apple devices (open in Xcode on a Mac).",
  design: "Every setting and number, machine-readable. Always included.",
};

export default function ExportDrawer({ runId, detail, onClose }: { runId: string; detail: DesignDetail | null; onClose: () => void }) {
  const caps = detail?.capabilities;
  const unavailable = caps ? Object.entries(caps).filter(([, capability]) => !capability.available) : [];
  const selectionKey = caps ? ORDER.map((k) => `${k}:${caps[k]?.available ? "yes" : "no"}`).join("|") : "none";
  const available = useMemo(() => new Set(ORDER.filter((k) => caps?.[k]?.available)), [caps]);
  const [selection, setSelection] = useState<{ key: string; values: Set<string> }>({ key: "", values: new Set() });
  const pick = selection.key === selectionKey ? selection.values : available;
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  const download = async () => {
    if (!detail) return;
    setBusy(true);
    setMsg(null);
    try {
      const formats = Array.from(new Set([...pick, "design"]));
      const r = await api(`/runs/${runId}/export`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: detail.design.design_key, formats }) });
      if (!r.ok) throw new Error(await readError(r));
      const blob = await r.blob();
      const name = /filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") ?? "")?.[1] ?? "nomo_export.zip";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      let skipped: Record<string, string> = {};
      try { skipped = (JSON.parse(r.headers.get("X-Nomo-Manifest") ?? "{}") as { skipped?: Record<string, string> }).skipped ?? {}; } catch { /* header optional */ }
      const n = Object.keys(skipped).length;
      setMsg({ kind: "ok", text: `Downloaded ${name}.${n ? ` ${n} format${n > 1 ? "s" : ""} could not be produced; the README inside explains why.` : ""}` });
    } catch (x) {
      setMsg({ kind: "err", text: x instanceof Error ? x.message : String(x) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside aria-label="Export" className="nomo-drawer flex h-full flex-col bg-panel">
      <div className="flex items-center justify-between border-b border-line px-5 py-4">
        <h2 className="text-lg font-bold">Export this design</h2>
        <button onClick={onClose} aria-label="Close export" className="text-xl leading-none text-ink-muted hover:text-ink">×</button>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto px-5 py-4">
        {!detail && <p className="text-sm text-ink-muted">Exports are available once the search has finished.</p>}
        {unavailable.length > 0 && (
          <section aria-label="Unavailable export formats" className="rounded-lg border border-dashed border-line-strong bg-paper p-3">
            <p className="text-sm font-bold">Some outputs are unavailable</p>
            <p className="mt-1 text-2xs leading-relaxed text-ink-muted">Nomo will not pretend to export an unsupported backend. The reason for each gap is recorded below.</p>
            <ul className="mt-2 space-y-1.5">
              {unavailable.map(([key, capability]) => <li key={key} className="text-2xs leading-relaxed"><span className="font-bold">{capability.label}:</span> <span className="text-cross">{capability.reason ?? "not supported for this design"}.</span></li>)}
            </ul>
          </section>
        )}
        {caps && ORDER.map((k) => {
          const c = caps[k];
          if (!c) return null;
          const on = pick.has(k) || k === "design";
          return (
            <label key={k} className={`flex gap-3 rounded-lg border p-3 ${c.available ? "border-line hover:border-line-strong" : "border-dashed border-line opacity-70"}`}>
              <input type="checkbox" className="mt-1 h-4 w-4 accent-[#2F5BEA]" disabled={!c.available || k === "design"} checked={on && c.available}
                onChange={(e) => setSelection({ key: selectionKey, values: new Set((() => { const n = new Set(pick); if (e.target.checked) n.add(k); else n.delete(k); return n; })())})} />
              <span className="text-sm">
                <span className="font-bold">{c.label}</span>
                <span className="block text-ink-muted">{WHAT[k]}</span>
                {!c.available && c.reason && <span className="mt-1 block text-cross">Not available: {c.reason}.</span>}
                {c.available && c.note && <span className="mt-1 block text-2xs text-ink-muted">Note: {c.note}.</span>}
              </span>
            </label>
          );
        })}
      </div>
      <div className="border-t border-line px-5 py-4">
        {msg && <p role={msg.kind === "err" ? "alert" : "status"} className={`mb-3 text-sm ${msg.kind === "err" ? "text-cross" : "text-sym-ink"}`}>{msg.text}</p>}
        <Button onClick={download} disabled={!detail || busy}>{busy ? "Preparing files…" : "Download zip"}</Button>
        <p className="mt-2 text-2xs text-ink-muted">Large vision models can take up to a minute on free hosting.</p>
      </div>
    </aside>
  );
}

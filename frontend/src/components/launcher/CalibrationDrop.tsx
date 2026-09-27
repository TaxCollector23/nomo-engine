"use client";

import { useRef, useState } from "react";

import { api } from "@/lib/api";
import { readError } from "@/lib/runConfig";
import type { CalibrationSummary } from "@/lib/telemetry/protocol";

export default function CalibrationDrop({ modelId, current, onAttached }: {
  modelId: string; current?: CalibrationSummary | null; onAttached: (summary: CalibrationSummary) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const send = async (file: File) => {
    setBusy(true); setErr(null); setStatus(null);
    try {
      const r = await api(`/models/${encodeURIComponent(modelId)}/calibration?filename=${encodeURIComponent(file.name)}`, {
        method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" },
      });
      if (!r.ok) throw new Error(await readError(r));
      const body = await r.json() as { calibration: CalibrationSummary };
      onAttached(body.calibration);
      setStatus(`${body.calibration.sample_count} samples attached. PTQ will use MSE/KL ranges at export time.`);
    } catch (x) { setErr(x instanceof Error ? x.message : String(x)); }
    finally { setBusy(false); }
  };

  return <div className="mt-4 rounded-lg border border-line bg-panel p-4">
    <div className="flex items-start justify-between gap-3">
      <div><p className="font-bold">Attach calibration data</p><p className="mt-1 text-sm text-ink-muted">100–500 input tensors for data-driven PTQ. Use .npz (inputs/X plus optional aux), .npy, or JSON.</p></div>
      {current && <span className="shrink-0 rounded-full bg-sym-tint px-2 py-1 text-2xs font-bold text-sym-ink">{current.sample_count} attached</span>}
    </div>
    <button type="button" disabled={busy} onClick={() => input.current?.click()} className="mt-3 rounded-md border border-line-strong px-3 py-1.5 text-sm font-bold hover:border-ink disabled:opacity-40">
      {busy ? "Attaching…" : "Choose calibration file"}
    </button>
    <input ref={input} type="file" accept=".npz,.npy,.json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) void send(f); e.target.value = ""; }} />
    {status && <p role="status" className="mt-2 text-sm text-sym-ink">{status}</p>}
    {err && <p role="alert" className="mt-2 text-sm text-cross">{err}</p>}
  </div>;
}

"use client";

import { useRef, useState } from "react";

import { api } from "@/lib/api";
import { readError } from "@/lib/runConfig";
import type { UploadedModel } from "@/lib/telemetry/protocol";

import { NumberField } from "../ui";

const ACCEPT = ".onnx,.pt,.pth,.json";

/** Drag-and-drop model upload. The file goes to the server once; the server returns its layer table. */
export default function UploadDrop({ onUploaded }: { onUploaded: (m: UploadedModel) => void }) {
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [acc, setAcc] = useState<number | null>(90);
  const [shape, setShape] = useState("");
  const input = useRef<HTMLInputElement>(null);

  const send = async (file: File) => {
    setErr(null);
    const ext = file.name.toLowerCase().split(".").pop() ?? "";
    if (!["onnx", "pt", "pth", "json"].includes(ext)) {
      setErr(`"${file.name}" is not a supported file. Use .onnx, .pt / .pth (weights only) or a Nomo .json graph.`);
      return;
    }
    setBusy(true);
    try {
      const q = new URLSearchParams({ filename: file.name, base_accuracy: String(acc ?? 90) });
      if (shape.trim()) q.set("input_shape", shape.trim());
      const r = await api(`/models/upload?${q}`, { method: "POST", body: file,
        headers: { "Content-Type": "application/octet-stream" } });
      if (!r.ok) throw new Error(await readError(r));
      onUploaded((await r.json()) as UploadedModel);
    } catch (x) {
      setErr(x instanceof Error ? x.message : String(x));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) void send(f); }}
        className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-8 text-center transition-colors ${over ? "border-ann bg-ann-tint" : "border-line-strong bg-panel"}`}>
        <p className="font-bold">{busy ? "Reading your model…" : "Drop your model file here"}</p>
        <p className="mt-1 text-sm text-ink-muted">ONNX, PyTorch weights (.pt / .pth) or a Nomo JSON graph, up to 50 MB</p>
        <button type="button" onClick={() => input.current?.click()} disabled={busy}
          className="mt-3 rounded-md border border-line-strong px-3 py-1.5 text-sm font-bold hover:border-ink disabled:opacity-40">
          Choose a file
        </button>
        <input ref={input} type="file" accept={ACCEPT} className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) void send(f); e.target.value = ""; }} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Your model's accuracy today" unit="%" value={acc} onChange={setAcc} min={1} max={100}
          help="Used as the starting point for the accuracy budget" />
        <label className="block">
          <span className="mb-1 block text-sm text-ink-soft">Input shape (only if asked)</span>
          <input value={shape} onChange={(e) => setShape(e.target.value)} placeholder="e.g. 3,32,32"
            className="w-full rounded-md border border-line bg-panel px-2.5 py-1.5 text-sm outline-none focus:border-ann" />
          <span className="mt-0.5 block text-2xs text-ink-muted">Needed for PyTorch conv weights and dynamic ONNX inputs</span>
        </label>
      </div>
      {err && <p role="alert" className="rounded-md border border-cross bg-cross-tint p-3 text-sm text-ink">{err}</p>}
    </div>
  );
}

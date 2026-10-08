"use client";

import { useMemo, useState } from "react";

import { runCoDesign, runEmulation, type CoDesignRequest, type CoDesignResponse, type EmulationResponse } from "@/lib/enterprise";

import { Button, Disclosure } from "./ui";

const DEFAULTS: Record<string, string> = {
  pe_rows: "1,2,4,8",
  pe_cols: "1,2,4,8",
  sram_bytes: "32768,131072,524288",
  memory_bandwidth_gbs: "1,4,16",
  precision_bits: "4,8",
};

function parseNumbers(value: string, label: string, integer = true): number[] {
  const values = value.split(",").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item));
  if (values.length === 0 || values.some((item) => item <= 0 || (integer && !Number.isInteger(item)))) {
    throw new Error(`${label} must be a comma-separated list of positive ${integer ? "integers" : "numbers"}.`);
  }
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function compactKey(key: string): string {
  return key.length > 18 ? `${key.slice(0, 9)}…${key.slice(-6)}` : key;
}

function metric(value: number, unit: string): string {
  if (unit === "J") return value >= 1e-3 ? `${(value * 1e3).toFixed(2)} mJ` : `${(value * 1e6).toFixed(2)} µJ`;
  if (unit === "s") return value >= 1 ? `${value.toFixed(2)} s` : `${(value * 1e3).toFixed(2)} ms`;
  return `${value.toFixed(2)} ${unit}`;
}

function Field({ label, value, onChange, help }: { label: string; value: string; onChange: (value: string) => void; help: string }) {
  return <label className="block"><span className="mb-1 block text-2xs font-bold text-ink-soft">{label}</span><input value={value} onChange={(event) => onChange(event.target.value)} className="w-full rounded-md border border-line bg-paper px-2.5 py-1.5 font-mono text-2xs outline-none focus:border-ann" /><span className="mt-0.5 block text-2xs text-ink-muted">{help}</span></label>;
}

export default function EnterpriseValidation({ runId, selectedKey }: { runId: string; selectedKey: string | null }) {
  const [values, setValues] = useState(DEFAULTS);
  const [coDesign, setCoDesign] = useState<CoDesignResponse | null>(null);
  const [emulation, setEmulation] = useState<EmulationResponse | null>(null);
  const [busy, setBusy] = useState<"co-design" | "emulation" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const update = (key: string, value: string) => setValues((current) => ({ ...current, [key]: value }));
  const value = (key: string): string => values[key] ?? "";
  const selected = coDesign?.recommended ?? null;
  const architecture = selected?.architecture;
  const request = useMemo<CoDesignRequest>(() => ({
    deployment_keys: selectedKey ? [selectedKey] : [],
    pe_rows: [], pe_cols: [], sram_bytes: [], memory_bandwidth_bytes_s: [], precision_bits: [],
  }), [selectedKey]);

  const search = async () => {
    setBusy("co-design");
    setError(null);
    setEmulation(null);
    try {
      const next: CoDesignRequest = {
        ...request,
        pe_rows: parseNumbers(value("pe_rows"), "PE rows"),
        pe_cols: parseNumbers(value("pe_cols"), "PE columns"),
        sram_bytes: parseNumbers(value("sram_bytes"), "SRAM sizes"),
        memory_bandwidth_bytes_s: parseNumbers(value("memory_bandwidth_gbs"), "Bandwidth", false).map((item) => item * 1e9),
        precision_bits: parseNumbers(value("precision_bits"), "Precision"),
      };
      setCoDesign(await runCoDesign(runId, next));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(null);
    }
  };

  const emulate = async () => {
    setBusy("emulation");
    setError(null);
    try {
      setEmulation(await runEmulation(runId, selectedKey ?? undefined));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(null);
    }
  };

  return <Disclosure title="Deployment validation" summary={selected ? "co-search complete" : "architecture co-search and cycle evidence"}>
    <div className="space-y-4">
      <div className="rounded-md border border-ann bg-ann-tint p-3 text-2xs leading-relaxed text-ann-ink">
        This is the enterprise handoff boundary: explore a bounded accelerator space, then run the selected integer design through Nomo&apos;s deterministic simulator. Results are analytic or simulated until target hardware telemetry is attached.
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="PE rows" value={value("pe_rows")} onChange={(next) => update("pe_rows", next)} help="Example: 1,2,4,8" />
        <Field label="PE columns" value={value("pe_cols")} onChange={(next) => update("pe_cols", next)} help="Example: 1,2,4,8" />
        <Field label="SRAM sizes (bytes)" value={value("sram_bytes")} onChange={(next) => update("sram_bytes", next)} help="Example: 32768,131072" />
        <Field label="Bandwidth (GB/s)" value={value("memory_bandwidth_gbs")} onChange={(next) => update("memory_bandwidth_gbs", next)} help="Converted to bytes/s for the API" />
        <Field label="Datapath precision (bits)" value={value("precision_bits")} onChange={(next) => update("precision_bits", next)} help="Example: 4,8,16" />
      </div>
      <div className="flex flex-wrap gap-2"><Button onClick={search} disabled={busy !== null}>{busy === "co-design" ? "Searching…" : "Run architecture co-search"}</Button><Button kind="secondary" onClick={emulate} disabled={busy !== null || !selectedKey}>{busy === "emulation" ? "Simulating…" : "Emulate active design"}</Button></div>
      {error && <p role="alert" className="rounded-md border border-cross bg-cross-tint p-3 text-2xs text-cross">{error}</p>}
      {selected && architecture && <section className="rounded-md border border-line bg-paper p-3">
        <div className="flex items-start justify-between gap-3"><div><p className="text-2xs font-bold uppercase tracking-widest text-ink-muted">Recommended architecture</p><p className="mt-1 font-mono text-2xs">{compactKey(selected.key)}</p></div><span className="rounded-full border border-ann bg-ann-tint px-2 py-0.5 text-2xs font-bold text-ann-ink">rank {selected.rank}</span></div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-2xs sm:grid-cols-4"><span><span className="block text-ink-muted">Array</span><strong>{architecture.pe_rows} × {architecture.pe_cols}</strong></span><span><span className="block text-ink-muted">SRAM</span><strong>{(architecture.sram_bytes / 1024).toFixed(0)} KB</strong></span><span><span className="block text-ink-muted">Precision</span><strong>{architecture.precision_bits}-bit</strong></span><span><span className="block text-ink-muted">PEs</span><strong>{architecture.pe_count}</strong></span></div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-2xs"><span><span className="block text-ink-muted">Modeled energy</span><strong>{metric(selected.metrics.estimated_energy_j, "J")}</strong></span><span><span className="block text-ink-muted">Modeled latency</span><strong>{metric(selected.metrics.estimated_latency_s, "s")}</strong></span><span><span className="block text-ink-muted">Accuracy loss prior</span><strong>{selected.metrics.accuracy_loss_pp.toFixed(2)} pp</strong></span><span><span className="block text-ink-muted">Evidence</span><strong>analytic prior</strong></span></div>
        <details className="mt-3"><summary className="cursor-pointer text-2xs font-bold text-ann-ink">Why this was selected</summary><ul className="mt-2 list-disc space-y-1 pl-5 text-2xs text-ink-muted">{selected.selection_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></details>
      </section>}
      {coDesign && <p className="text-2xs text-ink-muted">Evaluated {coDesign.evaluations.length} architecture/deployment combinations; {coDesign.front.length} are on the constrained Pareto frontier.</p>}
      {emulation && <section className="rounded-md border border-ann bg-ann-tint p-3 text-2xs text-ann-ink"><div className="flex items-center justify-between gap-3"><p className="font-bold">Bounded cycle evidence</p><span className="rounded-full border border-ann px-2 py-0.5 font-bold">SIMULATED</span></div><div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4"><span><span className="block opacity-70">Cycles</span><strong>{emulation.summary.cycles.toLocaleString()}</strong></span><span><span className="block opacity-70">Memory accesses</span><strong>{emulation.summary.memory_accesses.toLocaleString()}</strong></span><span><span className="block opacity-70">Cache hit ratio</span><strong>{(emulation.summary.cache_hit_ratio * 100).toFixed(1)}%</strong></span><span><span className="block opacity-70">Vectors</span><strong>{emulation.summary.vector_count}</strong></span></div><p className="mt-3 leading-relaxed opacity-80">No physical device, SystemC, Verilator, or Gem5 execution occurred. Use the HITL path before making latency or energy claims.</p></section>}
    </div>
  </Disclosure>;
}

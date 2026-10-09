"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { weightMemory } from "./launcher/LayerLockTable";
import { formatSI } from "@/lib/pareto";
import { mergeConfig, startRun } from "@/lib/runConfig";
import type { RunMeta } from "@/lib/useRunMeta";
import { Coding, Domain, type DesignDetail, type PinIn } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";

import { Button, DOMAIN_META, DomainChip, Term } from "./ui";

const DOM = ["ANN", "SNN", "SYM"] as const;
const PREC: Record<number, string> = { 16: "INT16", 8: "INT8", 4: "INT4", 2: "INT2", 1: "BINARY" };
interface EditorState { key: string; dom: string; bits: string; err: string | null }

/** Opened by clicking a layer: what it costs, and a one-click re-run with it locked. */
export default function LayerInspector({ meta, detail, onAsk }: { meta: RunMeta; detail: DesignDetail | null; onAsk: (q: string) => void }) {
  const router = useRouter();
  const idx = useRunStore((s) => s.inspect);
  const setInspect = useRunStore((s) => s.setInspect);
  const item = useRunStore((s) => (s.selectedKey ? s.items.get(s.selectedKey) : undefined));
  const names = useRunStore((s) => s.run?.layers ?? []);
  const [busy, setBusy] = useState(false);
  const name = idx !== null ? names[idx] : undefined;
  const pin = name ? meta.config?.pins?.[name] : undefined;
  const editorKey = `${name ?? ""}|${pin?.domain ?? ""}|${pin?.w_bits ?? ""}`;
  const [editor, setEditor] = useState<EditorState | null>(null);
  const current = editor?.key === editorKey ? editor : {
    key: editorKey,
    dom: pin?.domain ?? "",
    bits: pin?.w_bits ? String(pin.w_bits) : "",
    err: null,
  };
  const dom = current.dom;
  const bits = current.bits;
  const err = current.err;
  if (idx === null || !item || !name) return null;

  const gene = item.genome.layers[idx]!;
  const d = DOM[gene[0] as Domain];
  const row = meta.layers.find((l) => l.name === name);
  const dl = detail?.design.design_key === item.key ? detail.design.layers[idx] : undefined;
  const totalE = detail?.design.metrics.energy_j;
  const hwBits = meta.catalog && meta.config ? meta.catalog.hardware[meta.config.hardware]?.bits : undefined;
  const bitOptions = Array.from(new Set([...(hwBits?.continuous ?? []), ...(hwBits?.spiking ?? [])])).sort((a, b) => b - a);

  const rerun = async () => {
    if (!meta.config) return;
    setBusy(true);
    setEditor({ ...current, err: null });
    const pin: PinIn = { domain: (dom || null) as PinIn["domain"], w_bits: bits ? (PREC[Number(bits)] as PinIn["w_bits"]) ?? Number(bits) : null };
    const pins = { ...(meta.config.pins ?? {}) };
    if (pin.domain || pin.w_bits) pins[name] = pin; else delete pins[name];
    try {
      const id = await startRun({ ...mergeConfig(meta.config, {}), pins, preset: null });
      router.push(`/runs/${id}`);
    } catch (x) {
      setEditor({ ...current, err: x instanceof Error ? x.message : String(x) });
      setBusy(false);
    }
  };

  return (
    <section aria-label={`Layer ${name}`} className="nomo-drawer rounded-lg border border-ink bg-panel p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-bold">{name}</h3>
          <p className="text-sm text-ink-muted">{row?.op === "conv2d" ? "Convolution" : "Fully connected"} layer {idx + 1} of {names.length}</p>
        </div>
        <button onClick={() => setInspect(null)} aria-label="Close layer details" className="text-lg leading-none text-ink-muted hover:text-ink">×</button>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
        <span>Runs as</span><DomainChip d={d} />
        {d === "SNN" && <span className="text-ink-muted">{gene[3] === Coding.TTFS ? "time-to-first-spike" : "rate"} code, {gene[4]} steps</span>}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <dt className="text-ink-muted"><Term k="precision">Precision</Term></dt>
        <dd>{d === "SYM" ? "exact (32-bit fixed point)" : `${gene[1]}-bit weights`}</dd>
        {row && <><dt className="text-ink-muted">Weights</dt><dd>{row.params.toLocaleString()} ({weightMemory(row.params, d === "SYM" ? 32 : gene[1])})</dd></>}
        {dl?.memory_bytes != null && <><dt className="text-ink-muted"><Term k="sram">On-chip memory</Term></dt><dd>{(dl.memory_bytes / 1024).toFixed(1)} KB</dd></>}
        {dl?.cores ? <><dt className="text-ink-muted"><Term k="cores">Cores</Term></dt><dd>{dl.cores}</dd></> : null}
        {dl?.energy_j != null && totalE ? <><dt className="text-ink-muted">Energy</dt><dd>{formatSI(dl.energy_j, "J")} ({(100 * dl.energy_j) / totalE < 0.1 ? "under 0.1" : ((100 * dl.energy_j) / totalE).toFixed(1)}% of total)</dd></> : null}
        {row && <><dt className="text-ink-muted">Can run as</dt><dd>{row.can_be.map((x) => DOMAIN_META[x].word).join(", ")}</dd></>}
      </dl>
      <button onClick={() => onAsk(`Why is ${name} ${DOMAIN_META[d].word.toLowerCase()}?`)} className="mt-3 text-sm font-bold text-ann-ink hover:underline">
        Why this style? Ask Copilot
      </button>
      <div className="mt-4 border-t border-line pt-4">
        <p className="mb-2 text-sm font-bold">Lock this layer and search again</p>
        <div className="grid grid-cols-2 gap-2">
          <select aria-label="Lock style" value={dom} onChange={(e) => setEditor({ ...current, dom: e.target.value })} className="rounded-md border border-line bg-panel px-2 py-1.5 text-sm">
            <option value="">Any style</option>
            {(row?.can_be ?? [d]).map((x) => <option key={x} value={x}>{DOMAIN_META[x].word}</option>)}
          </select>
          <select aria-label="Lock precision" value={bits} onChange={(e) => setEditor({ ...current, bits: e.target.value })} className="rounded-md border border-line bg-panel px-2 py-1.5 text-sm">
            <option value="">Any precision</option>
            {bitOptions.map((b) => <option key={b} value={b}>{PREC[b] ?? `${b}-bit`}</option>)}
          </select>
        </div>
        <div className="mt-3"><Button onClick={rerun} disabled={busy || !meta.config}>{busy ? "Starting…" : "Search again with this lock"}</Button></div>
        {err && <p role="alert" className="mt-2 text-sm text-cross">{err}</p>}
      </div>
    </section>
  );
}

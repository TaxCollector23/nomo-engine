"use client";

import type { LayerRow, PinIn, Precision } from "@/lib/telemetry/protocol";

import { DOMAIN_META, Term } from "../ui";

const BITS_TO_NAME: Record<number, Precision> = { 16: "INT16", 8: "INT8", 4: "INT4", 2: "INT2", 1: "BINARY" };
const NAME_TO_BITS: Record<Precision, number> = { INT16: 16, INT8: 8, INT4: 4, INT2: 2, BINARY: 1 };

function compact(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} k`;
  return String(n);
}

export function weightMemory(params: number, bits: number): string {
  const kb = (params * bits) / 8 / 1024;
  return kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(kb < 10 ? 1 : 0)} KB`;
}

/** One row per layer: what it is, and optional locks on its style and weight precision. */
export default function LayerLockTable({ layers, pins, onChange, bits }: {
  layers: LayerRow[]; pins: Record<string, PinIn>; onChange: (p: Record<string, PinIn>) => void;
  bits: { continuous: number[]; spiking: number[] };
}) {
  const set = (name: string, patch: Partial<PinIn>) => {
    const next = { ...pins, [name]: { ...pins[name], ...patch } };
    const p = next[name]!;
    if (!p.domain && !p.w_bits) delete next[name];
    onChange(next);
  };
  const allBits = Array.from(new Set([...bits.continuous, ...bits.spiking])).sort((a, b) => b - a);
  const lockedCount = Object.keys(pins).length;
  return (
    <div>
      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="bg-paper text-left text-ink-muted">
            <tr>
              <th className="px-3 py-2 font-normal">Layer</th>
              <th className="px-3 py-2 font-normal">Size</th>
              <th className="px-3 py-2 font-normal">Work per input</th>
              <th className="px-3 py-2 font-normal">Can run as</th>
              <th className="px-3 py-2 font-normal"><Term k="lock">Lock style</Term></th>
              <th className="px-3 py-2 font-normal"><Term k="precision">Lock precision</Term></th>
            </tr>
          </thead>
          <tbody>
            {layers.map((l) => {
              const pin = pins[l.name] ?? {};
              const pb = typeof pin.w_bits === "string" ? NAME_TO_BITS[pin.w_bits] : pin.w_bits ?? null;
              const locked = Boolean(pin.domain || pin.w_bits);
              return (
                <tr key={l.name} className={`border-t border-line ${locked ? "bg-ann-tint/40" : "bg-panel"}`}>
                  <td className="px-3 py-2">
                    <span className="font-bold">{l.name}</span>
                    <span className="ml-2 text-2xs text-ink-muted">{l.op === "conv2d" ? "convolution" : "fully connected"}</span>
                  </td>
                  <td className="px-3 py-2 text-ink-soft" title={`${l.params.toLocaleString()} weights`}>
                    {compact(l.params)} weights
                    <span className="block text-2xs text-ink-muted">{weightMemory(l.params, pb ?? 8)} at {pb ?? 8}-bit</span>
                  </td>
                  <td className="px-3 py-2 text-ink-soft" title={`${l.macs.toLocaleString()} multiply-adds`}>{compact(l.macs)} ops</td>
                  <td className="px-3 py-2">
                    <span className="flex flex-wrap gap-1">
                      {l.can_be.map((d) => (
                        <span key={d} className={`h-2 w-2 rounded-full ${DOMAIN_META[d].dot}`} title={DOMAIN_META[d].word} />
                      ))}
                      <span className="sr-only">{l.can_be.map((d) => DOMAIN_META[d].word).join(", ")}</span>
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <select aria-label={`Lock style of ${l.name}`} value={pin.domain ?? ""}
                      onChange={(e) => set(l.name, { domain: (e.target.value || null) as PinIn["domain"] })}
                      className="rounded-md border border-line bg-panel px-2 py-1 text-sm">
                      <option value="">Let Nomo choose</option>
                      {l.can_be.map((d) => <option key={d} value={d}>{DOMAIN_META[d].word}</option>)}
                    </select>
                  </td>
                  <td className="px-3 py-2">
                    <select aria-label={`Lock precision of ${l.name}`} value={pb ?? ""}
                      onChange={(e) => set(l.name, { w_bits: e.target.value ? BITS_TO_NAME[Number(e.target.value)] ?? Number(e.target.value) : null })}
                      className="rounded-md border border-line bg-panel px-2 py-1 text-sm">
                      <option value="">Let Nomo choose</option>
                      {allBits.map((b) => <option key={b} value={b}>{BITS_TO_NAME[b] ?? `${b}-bit`}</option>)}
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-2xs text-ink-muted">
        {lockedCount ? `${lockedCount} layer${lockedCount > 1 ? "s" : ""} locked. ` : ""}
        Precisions offered are the ones this chip supports. Floating-point (FP16) is not offered: Nomo&apos;s chip models are integer-only.
      </p>
    </div>
  );
}

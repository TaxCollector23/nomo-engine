"use client";

import { useMemo } from "react";

import { formatSI, passesFilter } from "@/lib/pareto";
import { useRunStore } from "@/lib/telemetry/store";

/** Range filters for the trade-off plot. Sliders span the feasible designs found so far (log scale for energy/latency). */
export default function FilterBar() {
  const items = useRunStore((s) => s.items);
  const version = useRunStore((s) => s.version);
  const filter = useRunStore((s) => s.filter);
  const setFilter = useRunStore((s) => s.setFilter);
  const front = useRunStore((s) => s.front);

  const r = useMemo(() => {
    const f = Array.from(items.values()).filter((it) => it.feasible);
    if (f.length < 2) return null;
    const col = (j: 0 | 1 | 2) => f.map((it) => it.f[j]);
    return { e: [Math.min(...col(0)), Math.max(...col(0))], l: [Math.min(...col(1)), Math.max(...col(1))],
      a: [Math.min(...col(2)), Math.max(...col(2))] };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, version]);
  if (!r) return null;
  const shown = Array.from(front).filter((k) => { const it = items.get(k); return it && passesFilter(it, filter); }).length;

  const logSlider = (lo: number, hi: number, v: number | null, set: (x: number | null) => void, unit: string, label: string) => {
    const L = Math.log10(lo), H = Math.log10(hi);
    const pos = v === null ? 1000 : Math.round((1000 * (Math.log10(v) - L)) / (H - L || 1));
    return (
      <label className="block min-w-0 flex-1">
        <span className="flex justify-between text-2xs text-ink-muted"><span>{label}</span><span className="font-bold text-ink">{v === null ? "any" : `≤ ${formatSI(v, unit)}`}</span></span>
        <input type="range" min={0} max={1000} value={pos} className="w-full"
          onChange={(e) => { const p = Number(e.target.value); set(p >= 1000 ? null : 10 ** (L + ((H - L) * p) / 1000)); }} />
      </label>
    );
  };
  return (
    <div className="flex flex-wrap items-end gap-4 rounded-lg border border-line bg-panel/95 px-3 py-2 backdrop-blur">
      {logSlider(r.e[0]!, r.e[1]!, filter.eMax, (x) => setFilter({ eMax: x }), "J", "Max energy")}
      {logSlider(r.l[0]!, r.l[1]!, filter.lMax, (x) => setFilter({ lMax: x }), "s", "Max response time")}
      <label className="block min-w-0 flex-1">
        <span className="flex justify-between text-2xs text-ink-muted"><span>Min accuracy</span><span className="font-bold text-ink">{filter.accMin === null ? "any" : `≥ ${filter.accMin.toFixed(1)}%`}</span></span>
        <input type="range" min={0} max={1000} className="w-full"
          value={filter.accMin === null ? 0 : Math.round((1000 * (filter.accMin - r.a[0]!)) / (r.a[1]! - r.a[0]! || 1))}
          onChange={(e) => { const p = Number(e.target.value); setFilter({ accMin: p <= 0 ? null : r.a[0]! + ((r.a[1]! - r.a[0]!) * p) / 1000 }); }} />
      </label>
      <div className="flex items-center gap-3 pb-1 text-2xs text-ink-muted">
        <span>{shown} of {front.size} best trade-offs shown</span>
        {(filter.eMax !== null || filter.lMax !== null || filter.accMin !== null) && (
          <button onClick={() => setFilter({ eMax: null, lMax: null, accMin: null })} className="font-bold text-ann-ink hover:underline">Clear</button>
        )}
      </div>
    </div>
  );
}

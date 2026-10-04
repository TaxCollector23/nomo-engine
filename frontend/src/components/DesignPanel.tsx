"use client";

import { formatSI } from "@/lib/pareto";
import { Domain, type DesignDetail } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";

import { DomainChip, Term, formatPct } from "./ui";

const DOM = ["ANN", "SNN", "SYM"] as const;

function Figure({ label, value, delta, better }: { label: React.ReactNode; value: string; delta?: number; better?: "lower" | "higher" }) {
  const good = delta === undefined ? null : better === "lower" ? delta < -0.5 : delta > 0.05;
  const bad = delta === undefined ? null : better === "lower" ? delta > 0.5 : delta < -0.05;
  return (
    <div>
      <div className="text-sm text-ink-muted">{label}</div>
      <div className="text-xl font-bold">{value}</div>
      {delta !== undefined && (
        <div className={`text-sm ${good ? "text-sym-ink" : bad ? "text-cross" : "text-ink-muted"}`}>
          {better === "lower" ? formatPct(delta) : `${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(1)} pts`} vs all-continuous
        </div>
      )}
    </div>
  );
}

/** The selected design at a glance: headline numbers, layer styles, and what the plain-English summary says. */
export default function DesignPanel({ detail }: { detail: DesignDetail | null }) {
  const key = useRunStore((s) => s.selectedKey);
  const it = useRunStore((s) => (s.selectedKey ? s.items.get(s.selectedKey) : undefined));
  const rec = useRunStore((s) => s.recommendedKey);
  const select = useRunStore((s) => s.select);
  const names = useRunStore((s) => s.run?.layers ?? []);
  const setInspect = useRunStore((s) => s.setInspect);
  useRunStore((s) => s.version);
  if (!it || !key) return <p className="text-sm text-ink-muted">Waiting for the first designs…</p>;
  const base = detail?.design.design_key === key ? detail.design.baseline_all_continuous : null;
  const dE = base ? (100 * (it.f[0] - base.energy_j)) / base.energy_j : undefined;
  const dL = base ? (100 * (it.f[1] - base.latency_s)) / base.latency_s : undefined;
  const dA = base ? it.f[2] - base.accuracy : undefined;
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-bold">{key === rec ? "Recommended design" : "Selected design"}</h2>
        {rec && key !== rec && (
          <button onClick={() => select(rec, false)} className="text-sm font-bold text-ann-ink hover:underline">Back to recommended</button>
        )}
      </div>
      <div className="grid grid-cols-3 gap-3">
        <Figure label="Energy" value={formatSI(it.f[0], "J")} delta={dE} better="lower" />
        <Figure label="Response time" value={formatSI(it.f[1], "s")} delta={dL} better="lower" />
        <Figure label={<Term k="estimated">Accuracy</Term>} value={`${it.f[2].toFixed(1)}%`} delta={dA} better="higher" />
      </div>
      {!it.feasible && <p className="rounded-md bg-cross-tint p-2 text-sm">This design breaks one of your limits.</p>}
      {detail?.design.design_key === key && <p className="text-sm leading-relaxed text-ink-soft">{detail.summary.text}</p>}
      <ol className="space-y-1">
        {it.genome.layers.map((l, i) => (
          <li key={i}>
            <button onClick={() => setInspect(i)} className="flex w-full items-center justify-between rounded-md px-2 py-1 text-left text-sm hover:bg-paper">
              <span>{names[i] ?? `layer ${i}`}</span>
              <DomainChip d={DOM[l[0] as Domain]} small />
            </button>
          </li>
        ))}
      </ol>
      <p className="text-2xs text-ink-muted">
        {it.crossings} <Term k="crossing">style change{it.crossings === 1 ? "" : "s"}</Term>, {it.cores} <Term k="cores">neuromorphic cores</Term> used.
      </p>
    </div>
  );
}

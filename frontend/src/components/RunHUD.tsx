"use client";

import { useRunStore } from "@/lib/telemetry/store";

function Sparkline({ values, height = 44 }: { values: number[]; height?: number }) {
  if (values.length < 2) return <div style={{ height }} className="font-mono text-[10px] text-neutral-600">—</div>;
  const w = 220;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${height - 2 - ((v - lo) / span) * (height - 4)}`);
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className="block">
      <polyline points={pts.join(" ")} fill="none" stroke="#fafafa" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-neutral-500">{k}</div>
      <div className="text-sm text-neutral-100">{v}</div>
    </div>
  );
}

export default function RunHUD() {
  const run = useRunStore((s) => s.run);
  const status = useRunStore((s) => s.status);
  const conn = useRunStore((s) => s.connection);
  const detail = useRunStore((s) => s.connectionDetail);
  const lastGen = useRunStore((s) => s.lastGen);
  const hv = useRunStore((s) => s.hv);
  const error = useRunStore((s) => s.error);
  const nFront = useRunStore((s) => s.front.size);
  const CROSSOVERS = new Set(["segment_aligned", "precision_uniform"]);
  const all = Object.entries(lastGen?.operators ?? {}).sort((a, b) => b[1] - a[1]);
  const groups: [string, [string, number][]][] = [
    ["crossover", all.filter(([n]) => CROSSOVERS.has(n))],
    ["mutation", all.filter(([n]) => !CROSSOVERS.has(n))],
  ];

  return (
    <div className="space-y-5 font-mono">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs text-neutral-400">{run ? `${run.model} → ${run.hardware}` : "…"}</div>
          <div className="text-lg text-neutral-50">{status}</div>
        </div>
        <span className={`rounded border px-2 py-0.5 text-[10px] ${conn === "open" ? "border-neutral-300 text-neutral-200" : "border-neutral-700 text-neutral-500"}`}
          title={detail}>
          ws {conn}
        </span>
      </div>
      {error && <div className="border border-neutral-500 p-2 text-xs text-neutral-200">{error}</div>}
      <div className="grid grid-cols-2 gap-3">
        <Stat k="generation" v={`${lastGen?.gen ?? 0} / ${run?.generations ?? "?"}`} />
        <Stat k="front" v={String(nFront)} />
        <Stat k="unique evals" v={(lastGen?.unique ?? 0).toLocaleString()} />
        <Stat k="feasible" v={`${((lastGen?.feasible_fraction ?? 0) * 100).toFixed(0)}%`} />
      </div>
      <div>
        <div className="mb-1 flex justify-between text-[10px] uppercase tracking-wider text-neutral-500">
          <span>hypervolume (archive)</span><span className="text-neutral-300">{(lastGen?.hv ?? 0).toFixed(4)}</span>
        </div>
        <Sparkline values={hv.map((p) => p.hv)} />
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-neutral-500">operator selection (adaptive pursuit, %)</div>
        {groups.map(([gname, ops]) => (
          <div key={gname} className="mb-2 space-y-1">
            <div className="text-[9px] uppercase text-neutral-600">{gname}</div>
            {ops.map(([name, p]) => (
              <div key={name} className="flex items-center gap-2 text-[10px] text-neutral-400">
                <span className="w-28 truncate">{name}</span>
                <div className="h-1.5 flex-1 bg-neutral-900">
                  <div className="h-full bg-neutral-200 transition-[width] duration-300" style={{ width: `${Math.min(100, p * 100)}%` }} />
                </div>
                <span className="w-8 text-right">{(p * 100).toFixed(0)}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

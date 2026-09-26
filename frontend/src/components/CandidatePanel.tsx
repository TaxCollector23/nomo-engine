"use client";

import { formatSI } from "@/lib/pareto";
import { geneCode } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";

export default function CandidatePanel() {
  const key = useRunStore((s) => s.selectedKey);
  const it = useRunStore((s) => (s.selectedKey ? s.items.get(s.selectedKey) : undefined));
  const rec = useRunStore((s) => s.recommendedKey);
  const inFront = useRunStore((s) => (s.selectedKey ? s.front.has(s.selectedKey) : false));
  const select = useRunStore((s) => s.select);
  useRunStore((s) => s.version);
  if (!it || !key) return null;

  const rows: [string, string][] = [
    ["energy / inf", formatSI(it.f[0], "J")],
    ["latency", formatSI(it.f[1], "s")],
    ["accuracy", `${it.f[2].toFixed(2)} % (${it.acc_src})`],
    ["constraint viol.", it.cv === 0 ? "feasible" : it.cv.toFixed(3)],
    ["domain crossings", String(it.crossings)],
    ["neuro cores", String(it.cores)],
  ];
  return (
    <div className="space-y-3 font-mono">
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-wider text-neutral-500">
          selected {key === rec ? "· recommended" : ""} {inFront ? "· on front" : ""}
        </div>
        {rec && key !== rec && (
          <button onClick={() => select(rec, false)} className="border border-neutral-600 px-2 py-0.5 text-[10px] text-neutral-300 hover:border-neutral-200">
            follow recommended
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {it.genome.layers.map((l, i) => (
          <span key={i} className={`px-1.5 py-0.5 text-[10px] ${it.changed?.includes(i) ? "bg-neutral-200 text-neutral-900" : "bg-neutral-900 text-neutral-300"}`}>
            {geneCode(l)}
          </span>
        ))}
      </div>
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b border-neutral-900">
              <td className="py-1 text-neutral-500">{k}</td>
              <td className="py-1 text-right text-neutral-100">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {it.ops && (
        <div className="text-[10px] text-neutral-500">
          origin: {it.ops[0] ?? "clone"} + {it.ops[1]}{it.changed?.length ? ` · changed layers ${it.changed.join(", ")}` : ""}
        </div>
      )}
    </div>
  );
}

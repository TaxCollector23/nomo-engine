"use client";

import { useMemo } from "react";

import { Coding, Domain, GuardImpl, geneCode, type EvalItem } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";

const NODE_W = 92;
const NODE_H = 54;
const GAP = 46;
const PAD = 24;
const IO_W = 44;

const DOMAIN_STYLE: Record<Domain, { fill: string; stroke: string; text: string; dash?: string; label: string }> = {
  [Domain.ANN]: { fill: "#fafafa", stroke: "#fafafa", text: "#0a0a0a", label: "ANN" },
  [Domain.SNN]: { fill: "#5c5c5c", stroke: "#d4d4d4", text: "#fafafa", dash: "4 3", label: "SNN" },
  [Domain.SYM]: { fill: "#0a0a0a", stroke: "#fafafa", text: "#fafafa", label: "SYM" },
};

interface Stage { kind: "in" | "layer" | "guard" | "out"; domain: Domain; x: number; y: number; w: number; label: string; sub: string; idx: number }

/** Layered left-to-right layout of the execution stream (layers, then guards offset below their producer). */
function layout(item: EvalItem, names: string[]): { stages: Stage[]; width: number; height: number } {
  const stages: Stage[] = [];
  let x = PAD;
  const yMain = PAD + 14;
  stages.push({ kind: "in", domain: Domain.ANN, x, y: yMain, w: IO_W, label: "in", sub: "", idx: -1 });
  x += IO_W + GAP;
  item.genome.layers.forEach((l, i) => {
    const [d, , , c, T, p] = l;
    const sub = d === Domain.SNN ? `${c === Coding.TTFS ? "TTFS" : "rate"} T=${T}${p ? " · plastic" : ""}` : geneCode(l);
    stages.push({ kind: "layer", domain: d, x, y: yMain, w: NODE_W, label: names[i] ?? `L${i}`, sub, idx: i });
    const guard = item.genome.guards.find(([site]) => site === i);
    if (guard) {
      const fused = guard[1] === GuardImpl.FUSED;
      stages.push({ kind: "guard", domain: fused ? d : Domain.SYM, x, y: yMain + NODE_H + 30, w: NODE_W,
        label: "guard", sub: fused ? "fused" : "host", idx: i });
    }
    x += NODE_W + GAP;
  });
  stages.push({ kind: "out", domain: Domain.ANN, x, y: yMain, w: IO_W, label: "out", sub: "", idx: -1 });
  return { stages, width: x + IO_W + PAD, height: yMain + 2 * NODE_H + 30 + PAD + 8 };
}

function crossingTag(a: Domain, b: Domain): string {
  return `${DOMAIN_STYLE[a].label}→${DOMAIN_STYLE[b].label}`;
}

export default function PartitionGraph() {
  const key = useRunStore((s) => s.selectedKey);
  const item = useRunStore((s) => (s.selectedKey ? s.items.get(s.selectedKey) : undefined));
  const names = useRunStore((s) => s.run?.layers ?? []);
  useRunStore((s) => s.version);

  const g = useMemo(() => (item ? layout(item, names) : null), [item, names]);
  if (!item || !g) {
    return <div className="flex h-full items-center justify-center font-mono text-xs text-neutral-500">waiting for candidates…</div>;
  }

  // edges follow the execution stream: in -> layers -> (guard) -> ... -> out
  const stream = g.stages.filter((s) => s.kind !== "guard");
  const edges: { x1: number; y1: number; x2: number; y2: number; cross: boolean; tag: string }[] = [];
  let prev: Stage = stream[0]!;
  let prevDomain = item.genome.layers[0]?.[0] === Domain.SNN ? Domain.ANN : prev.domain;
  for (let i = 1; i < stream.length; i++) {
    const cur = stream[i]!;
    const guard = g.stages.find((s) => s.kind === "guard" && s.idx === prev.idx && prev.kind === "layer");
    if (guard) {
      edges.push({ x1: prev.x + prev.w / 2, y1: prev.y + NODE_H, x2: guard.x + guard.w / 2, y2: guard.y,
        cross: prev.domain !== guard.domain, tag: crossingTag(prev.domain, guard.domain) });
      const nextDom = cur.kind === "out" ? (guard.domain === Domain.SNN ? Domain.ANN : guard.domain) : cur.domain;
      edges.push({ x1: guard.x + guard.w, y1: guard.y + NODE_H / 2, x2: cur.x, y2: cur.y + NODE_H / 2,
        cross: guard.domain !== nextDom, tag: crossingTag(guard.domain, nextDom) });
      prevDomain = guard.domain;
    } else {
      const src = prev.kind === "in" ? prevDomain : prev.domain;
      const dst = cur.kind === "out" ? (src === Domain.SNN ? Domain.ANN : src) : cur.domain;
      edges.push({ x1: prev.x + prev.w, y1: prev.y + NODE_H / 2, x2: cur.x, y2: cur.y + NODE_H / 2,
        cross: src !== dst, tag: crossingTag(src, dst) });
      prevDomain = cur.domain;
    }
    prev = cur;
  }

  return (
    <div className="h-full w-full overflow-x-auto">
      <svg width={g.width} height={g.height} className="block font-mono" role="img" aria-label={`partition of ${key}`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#a3a3a3" />
          </marker>
        </defs>
        {edges.map((e, i) => (
          <g key={i}>
            <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={e.cross ? "#fafafa" : "#525252"}
              strokeWidth={e.cross ? 2.2 : 1.2} markerEnd="url(#arrow)" />
            {e.cross && (e.x1 === e.x2 ? (
              <text x={e.x1 + 6} y={(e.y1 + e.y2) / 2 + 3} fontSize={9} fill="#d4d4d4">{e.tag}</text>
            ) : (
              <text x={(e.x1 + e.x2) / 2} y={Math.min(e.y1, e.y2) - 6} textAnchor="middle" fontSize={9} fill="#d4d4d4">{e.tag}</text>
            ))}
          </g>
        ))}
        {g.stages.map((s) => {
          const st = DOMAIN_STYLE[s.domain];
          const io = s.kind === "in" || s.kind === "out";
          return (
            <g key={`${s.kind}${s.idx}`} transform={`translate(${s.x},${s.y})`} style={{ transition: "transform 300ms ease" }}>
              <rect width={s.w} height={NODE_H} rx={io ? 27 : 4}
                fill={io ? "transparent" : st.fill} stroke={io ? "#737373" : st.stroke}
                strokeDasharray={io ? undefined : st.dash} strokeWidth={s.kind === "guard" ? 2 : 1.2}
                style={{ transition: "fill 300ms ease, stroke 300ms ease" }} />
              <text x={s.w / 2} y={io ? NODE_H / 2 + 4 : 20} textAnchor="middle" fontSize={11}
                fill={io ? "#a3a3a3" : st.text}>{s.label}</text>
              {!io && (
                <>
                  <text x={s.w / 2} y={34} textAnchor="middle" fontSize={9} fill={st.text} opacity={0.8}>{s.sub}</text>
                  <text x={s.w / 2} y={47} textAnchor="middle" fontSize={8} fill={st.text} opacity={0.55}>
                    {s.kind === "guard" ? "symbolic constraint" : st.label}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

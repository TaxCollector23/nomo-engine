"use client";

import { useMemo } from "react";

import { Coding, Domain, GuardImpl, geneCode, type EvalItem, type PinIn } from "@/lib/telemetry/protocol";
import { useRunStore } from "@/lib/telemetry/store";

const NODE_W = 112;
const NODE_H = 62;
const GAP = 84;
const PAD = 20;
const IO_W = 48;

const STYLE: Record<Domain, { fill: string; stroke: string; text: string; short: string; word: string }> = {
  [Domain.ANN]: { fill: "#E5ECFD", stroke: "#2F5BEA", text: "#1C3FB0", short: "ANN", word: "Continuous" },
  [Domain.SNN]: { fill: "#FDF1DA", stroke: "#E39B17", text: "#8A5600", short: "SNN", word: "Spiking" },
  [Domain.SYM]: { fill: "#DDF3EC", stroke: "#0F8A6C", text: "#0A6650", short: "SYM", word: "Physics formula" },
};

interface Stage { kind: "in" | "layer" | "guard" | "out"; domain: Domain; x: number; y: number; w: number; label: string; sub: string; idx: number }

function layout(item: EvalItem, names: string[]): { stages: Stage[]; width: number; height: number } {
  const stages: Stage[] = [];
  let x = PAD;
  const yMain = PAD + 26;
  stages.push({ kind: "in", domain: Domain.ANN, x, y: yMain, w: IO_W, label: "Input", sub: "", idx: -1 });
  x += IO_W + GAP;
  item.genome.layers.forEach((l, i) => {
    const [d, w, a, c, T, p] = l;
    const sub = d === Domain.SNN ? `${c === Coding.TTFS ? "TTFS" : "rate"}, ${T} steps${p ? ", learns" : ""}`
      : d === Domain.SYM ? "exact formula" : `${w}-bit weights, ${a}-bit`;
    stages.push({ kind: "layer", domain: d, x, y: yMain, w: NODE_W, label: names[i] ?? `layer ${i}`, sub, idx: i });
    const guard = item.genome.guards.find(([site]) => site === i);
    if (guard) {
      const fused = guard[1] === GuardImpl.FUSED;
      stages.push({ kind: "guard", domain: fused ? d : Domain.SYM, x, y: yMain + NODE_H + 44, w: NODE_W,
        label: "Safety guard", sub: fused ? "inside the layer" : "on the host", idx: i });
    }
    x += NODE_W + GAP;
  });
  stages.push({ kind: "out", domain: Domain.ANN, x, y: yMain, w: IO_W, label: "Output", sub: "", idx: -1 });
  const hasGuard = stages.some((s) => s.kind === "guard");
  return { stages, width: x + IO_W + PAD, height: yMain + NODE_H + (hasGuard ? NODE_H + 44 : 0) + PAD + 6 };
}

interface Edge { x1: number; y1: number; x2: number; y2: number; src: Domain; dst: Domain }

export default function PartitionGraph({ pins = {} }: { pins?: Record<string, PinIn> }) {
  const key = useRunStore((s) => s.selectedKey);
  const item = useRunStore((s) => (s.selectedKey ? s.items.get(s.selectedKey) : undefined));
  const names = useRunStore((s) => s.run?.layers ?? []);
  const inspect = useRunStore((s) => s.inspect);
  const setInspect = useRunStore((s) => s.setInspect);
  useRunStore((s) => s.version);

  const g = useMemo(() => (item ? layout(item, names) : null), [item, names]);
  if (!item || !g) {
    return <div className="flex h-full items-center justify-center text-sm text-ink-muted">The design graph appears as soon as the first designs are tried.</div>;
  }

  const stream = g.stages.filter((s) => s.kind !== "guard");
  const edges: Edge[] = [];
  let prev: Stage = stream[0]!;
  let prevDomain: Domain = Domain.ANN;
  for (let i = 1; i < stream.length; i++) {
    const cur = stream[i]!;
    const guard = g.stages.find((s) => s.kind === "guard" && prev.kind === "layer" && s.idx === prev.idx);
    const dst = cur.kind === "out" ? Domain.ANN : cur.domain;
    if (guard) {
      edges.push({ x1: prev.x + prev.w / 2, y1: prev.y + NODE_H, x2: guard.x + guard.w / 2, y2: guard.y, src: prev.domain, dst: guard.domain });
      edges.push({ x1: guard.x + guard.w, y1: guard.y + NODE_H / 2, x2: cur.x, y2: cur.y + NODE_H / 2, src: guard.domain, dst });
      prevDomain = guard.domain;
    } else {
      const src = prev.kind === "in" ? Domain.ANN : prev.domain;
      edges.push({ x1: prev.x + prev.w, y1: prev.y + NODE_H / 2, x2: cur.x, y2: cur.y + NODE_H / 2, src, dst });
      prevDomain = cur.domain;
    }
    prev = cur;
  }
  void prevDomain;

  return (
    <div className="h-full w-full overflow-x-auto">
      <svg width={g.width} height={g.height} className="block" role="img" aria-label={`Design ${key}`}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#8B93A3" />
          </marker>
        </defs>
        {edges.map((e, i) => {
          const cross = e.src !== e.dst;
          const path = `M${e.x1},${e.y1} L${e.x2},${e.y2}`;
          const mx = (e.x1 + e.x2) / 2;
          const my = (e.y1 + e.y2) / 2;
          const spiking = e.src === Domain.SNN;
          return (
            <g key={i}>
              <path d={path} stroke={cross ? "#C8385A" : "#AEB7C4"} strokeWidth={cross ? 2 : 1.4} fill="none" markerEnd="url(#arrow)" />
              {spiking && [0, 0.33, 0.66].map((delay) => (
                <circle key={delay} r={2.6} fill="#E39B17" className="nomo-spike">
                  <animateMotion dur="1.4s" begin={`${delay * 1.4}s`} repeatCount="indefinite" path={path} />
                </circle>
              ))}
              {cross && (
                <g transform={`translate(${mx},${e.x1 === e.x2 ? my : my - 16})`}>
                  <title>{`${STYLE[e.src].word} to ${STYLE[e.dst].word}: data is converted here, which costs a little energy and time`}</title>
                  <rect x={-31} y={-9} width={62} height={18} rx={9} fill="#FBE4EA" stroke="#C8385A" />
                  <text textAnchor="middle" y={4} fontSize={10.5} fontWeight={700} fill="#8E1F3B">
                    {`${STYLE[e.src].short} → ${STYLE[e.dst].short}`}
                  </text>
                </g>
              )}
            </g>
          );
        })}
        {g.stages.map((s) => {
          const st = STYLE[s.domain];
          const io = s.kind === "in" || s.kind === "out";
          if (io) {
            return (
              <g key={`${s.kind}`} transform={`translate(${s.x},${s.y})`}>
                <rect width={s.w} height={NODE_H} rx={NODE_H / 2} fill="#FFFFFF" stroke="#AEB7C4" />
                <text x={s.w / 2} y={NODE_H / 2 + 4} textAnchor="middle" fontSize={11} fill="#5B6475">{s.label}</text>
              </g>
            );
          }
          const layer = s.kind === "layer";
          const name = layer ? (names[s.idx] ?? "") : "";
          const locked = layer && Boolean(pins[name]?.domain || pins[name]?.w_bits);
          const active = layer && inspect === s.idx;
          const tag = s.kind === "guard" ? "Safety rule" : st.word;
          return (
            <g key={`${s.kind}${s.idx}`} transform={`translate(${s.x},${s.y})`} style={{ transition: "transform 300ms ease" }}
              className={layer ? "cursor-pointer" : undefined}
              role={layer ? "button" : undefined} tabIndex={layer ? 0 : undefined}
              aria-label={layer ? `${s.label}: ${st.word}, ${s.sub}. Open layer details` : undefined}
              onClick={layer ? () => setInspect(active ? null : s.idx) : undefined}
              onKeyDown={layer ? (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setInspect(active ? null : s.idx); } } : undefined}>
              <rect width={s.w} height={NODE_H} rx={8} fill={st.fill} stroke={active ? "#1D2433" : st.stroke}
                strokeWidth={active ? 2.5 : s.kind === "guard" ? 2 : 1.4} strokeDasharray={s.kind === "guard" ? "5 3" : undefined}
                style={{ transition: "fill 300ms ease, stroke 300ms ease" }} />
              <text x={10} y={20} fontSize={12} fontWeight={700} fill="#1D2433">{s.label.length > 14 ? `${s.label.slice(0, 13)}…` : s.label}</text>
              <text x={10} y={37} fontSize={10.5} fill={st.text}>{tag}</text>
              <text x={10} y={52} fontSize={10} fill="#5B6475">{s.sub}</text>
              {locked && (
                <g transform={`translate(${s.w - 18},8)`}>
                  <title>Locked by you</title>
                  <rect width={11} height={8} y={4} rx={1.5} fill="#1D2433" />
                  <path d="M2.5,4 V2.5 a3,3 0 0 1 6,0 V4" stroke="#1D2433" strokeWidth={1.5} fill="none" />
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export { geneCode };

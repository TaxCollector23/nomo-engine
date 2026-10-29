import { useMemo, useState } from "react";

import {
  memoryTimeline,
  rooflinePoints,
  type LayerPrecision,
  type LayerTrainingPlan,
  type LayerTrainingProblem,
  type RecomputeMode,
} from "../planner/layers";

/** Shared helpers ---------------------------------------------------------------- */

function logTicks(min: number, max: number): number[] {
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    const v = 10 ** e;
    if (v >= min && v <= max) ticks.push(v);
  }
  return ticks;
}

function fmtBytes(value: number): string {
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TB`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} MB`;
  return `${(value / 1e3).toFixed(0)} KB`;
}

function fmtRate(value: number): string {
  if (value >= 1e15) return `${(value / 1e15).toFixed(0)} P`;
  if (value >= 1e12) return `${(value / 1e12).toFixed(0)} T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} G`;
  return value.toExponential(1);
}

const W = 560;
const PAD = { l: 62, r: 16, t: 14, b: 40 };
export type LayerLocks = Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>;

/** Pipeline-schedule Gantt -------------------------------------------------------- */

export interface GanttOp { kind: "f" | "b"; j: number; start: number; end: number; stage: number }

/**
 * Simulates the standard 1F1B (PipeDream-Flush) and GPipe schedules on the evaluator's
 * own per-stage times. Forward slot = `stageTimes[s]` exactly as charged by
 * evaluateLayerPlan (transfer included); backward slot = 2x forward, the usual
 * backward/forward FLOP ratio for dense transformers (disclosed assumption; the layer
 * evaluator itself charges forward work only).
 */
export function simulateSchedule(stageTimesS: number[], microBatches: number, schedule: "1f1b" | "gpipe", bwdRatio = 2): { ops: GanttOp[]; makespanS: number } {
  const S = stageTimesS.length;
  const m = Math.max(1, microBatches);
  const f = stageTimesS.map((t) => Math.max(t, 1e-9));
  const b = f.map((v) => v * bwdRatio);
  const order: Array<Array<{ kind: "f" | "b"; j: number }>> = [];
  for (let s = 0; s < S; s++) {
    const ops: Array<{ kind: "f" | "b"; j: number }> = [];
    if (schedule === "gpipe") {
      for (let j = 0; j < m; j++) ops.push({ kind: "f", j });
      for (let j = 0; j < m; j++) ops.push({ kind: "b", j });
    } else {
      const warmup = Math.min(m, S - s);
      for (let j = 0; j < warmup; j++) ops.push({ kind: "f", j });
      for (let j = 0; j < m - warmup; j++) { ops.push({ kind: "b", j }); ops.push({ kind: "f", j: j + warmup }); }
      for (let j = m - warmup; j < m; j++) ops.push({ kind: "b", j });
    }
    order.push(ops);
  }
  // Relaxation: schedule each stage's next op once its dependency end time is known.
  const clock = new Array<number>(S).fill(0);
  const ptr = new Array<number>(S).fill(0);
  const fwdEnd: Array<Array<number | null>> = Array.from({ length: S }, () => Array<number | null>(m).fill(null));
  const bwdEnd: Array<Array<number | null>> = Array.from({ length: S }, () => Array<number | null>(m).fill(null));
  const ops: GanttOp[] = [];
  let progress = true;
  while (progress) {
    progress = false;
    for (let s = 0; s < S; s++) {
      const op = order[s]![ptr[s]!];
      if (!op) continue;
      const ready = op.kind === "f"
        ? (s === 0 ? 0 : fwdEnd[s - 1]![op.j])
        : (s === S - 1 ? fwdEnd[s]![op.j] : bwdEnd[s + 1]![op.j]);
      if (ready === null || ready === undefined) continue;
      const dur = op.kind === "f" ? f[s]! : b[s]!;
      const start = Math.max(clock[s]!, ready);
      const end = start + dur;
      clock[s] = end;
      (op.kind === "f" ? fwdEnd : bwdEnd)[s]![op.j] = end;
      ops.push({ ...op, start, end, stage: s });
      ptr[s]!++;
      progress = true;
    }
  }
  const makespanS = ops.length ? Math.max(...ops.map((o) => o.end)) : 0;
  return { ops, makespanS };
}

export function PipelineGantt({ stageTimes, microBatches }: { stageTimes: number[]; microBatches: number }) {
  const [schedule, setSchedule] = useState<"1f1b" | "gpipe">("1f1b");
  const S = stageTimes.length;
  const { ops, makespanS } = useMemo(
    () => simulateSchedule(stageTimes, microBatches, schedule),
    [stageTimes, microBatches, schedule],
  );
  const busy = useMemo(() => ops.reduce((sum, o) => sum + (o.end - o.start), 0), [ops]);
  const bubblePct = makespanS > 0 ? (100 * (makespanS * S - busy)) / (makespanS * S) : 0;
  // The evaluator's analytic bubble: max_stage x (S-1)/m added on top of max_stage.
  const maxStage = Math.max(...stageTimes);
  const analyticStep = maxStage * (1 + (S - 1) / Math.max(1, microBatches));
  const H = 30 + S * 34;
  const x = (t: number) => PAD.l + (t / Math.max(makespanS, 1e-9)) * (W - PAD.l - PAD.r);
  return (
    <div>
      <div className="lab-axes" role="group" aria-label="Schedule">
        {(["1f1b", "gpipe"] as const).map((s) => (
          <button key={s} type="button" className={schedule === s ? "is-on" : ""} onClick={() => setSchedule(s)}>{s === "1f1b" ? "1F1B" : "GPipe"}</button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Pipeline schedule, ${S} stages, ${microBatches} micro-batches`}>
        {Array.from({ length: S }, (_, s) => (
          <g key={s}>
            <rect x={PAD.l} y={22 + s * 34} width={W - PAD.l - PAD.r} height={26} fill="#f4f4f2" rx={4} />
            <text x={PAD.l - 8} y={22 + s * 34 + 17} textAnchor="end" fontSize={11} fill="#666">S{s + 1}</text>
          </g>
        ))}
        {ops.map((o, i) => (
          <rect key={i} x={x(o.start)} y={24 + o.stage * 34} width={Math.max(0.7, x(o.end) - x(o.start))} height={22} rx={3}
            fill={o.kind === "f" ? "#2f6f4f" : "#b8860b"} opacity={0.88}>
            <title>{`Stage ${o.stage + 1} ${o.kind === "f" ? "forward" : "backward"} micro-batch ${o.j + 1}: ${o.start.toFixed(3)}–${o.end.toFixed(3)} s`}</title>
          </rect>
        ))}
        <text x={PAD.l} y={H - 8} fontSize={11} fill="#666">0 s</text>
        <text x={W - PAD.r} y={H - 8} fontSize={11} fill="#666" textAnchor="end">{makespanS.toFixed(3)} s</text>
      </svg>
      <p className="lab-note">
        One micro-batch forward (green) / backward (amber) slot per stage; grey is bubble — {bubblePct.toFixed(0)}% of device time in
        this schedule. Simulated step {makespanS.toFixed(3)} s vs the evaluator's analytic step {analyticStep.toFixed(3)} s
        (max stage × (1 + (S−1)/m)); they differ because the simulation keeps per-stage times heterogeneous instead of charging every
        stage the max. Backward slots assume 2× forward FLOPs; transfer is folded into stage time exactly as the evaluator charges it.
      </p>
    </div>
  );
}

/** Roofline scatter ---------------------------------------------------------------- */

const KIND_COLOR: Record<string, string> = {
  embedding: "#7a5195", attention: "#2f6f4f", mlp: "#31688e", output: "#b8860b",
};

export function RooflineScatter({ problem, plan, locks }: { problem: LayerTrainingProblem; plan: LayerTrainingPlan; locks: LayerLocks }) {
  const points = useMemo(() => rooflinePoints(problem, plan, locks), [problem, plan, locks]);
  if (!points.length) return null;
  const intensities = points.map((p) => p.intensity).filter((v) => v > 0);
  const minX = Math.min(...(intensities.length ? intensities : [0.1]), 0.1);
  const maxX = Math.max(...intensities, 1) * 1.6;
  const peakMax = Math.max(...points.map((p) => p.peakFlops));
  const H = 320;
  const lx = (v: number) => PAD.l + ((Math.log10(Math.max(v, 1e-2)) - Math.log10(minX)) / Math.max(1e-9, Math.log10(maxX) - Math.log10(minX))) * (W - PAD.l - PAD.r);
  const ly = (v: number) => H - PAD.b - ((Math.log10(Math.max(v, 1e6)) - 6) / Math.max(1e-9, Math.log10(peakMax) - 6)) * (H - PAD.t - PAD.b);
  const bw = problem.hardware?.memoryBandwidthBytesS ?? 3.35e12;
  const bf16Peak = points.find((p) => p.precision === "bf16")?.peakFlops ?? peakMax;
  const fp8Peak = points.find((p) => p.precision === "fp8")?.peakFlops ?? null;
  const rooflinePath = (peak: number) => {
    const ridge = peak / bw;
    const startI = Math.max(minX, 1e-2);
    return `M ${lx(startI)} ${ly(startI * bw)} L ${lx(ridge)} ${ly(peak)} L ${lx(maxX)} ${ly(peak)}`;
  };
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-layer roofline, arithmetic intensity versus attainable FLOPs">
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#ccc" />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="#ccc" />
        {logTicks(minX, maxX).map((t) => (
          <g key={t}>
            <text x={lx(t)} y={H - PAD.b + 16} fontSize={10} fill="#888" textAnchor="middle">{t >= 1 ? t : t.toFixed(1)}</text>
            <line x1={lx(t)} y1={H - PAD.b} x2={lx(t)} y2={H - PAD.b + 4} stroke="#888" />
          </g>
        ))}
        {[1e9, 1e12].map((t) => (
          <g key={t}>
            <text x={PAD.l - 8} y={ly(t) + 3} fontSize={10} fill="#888" textAnchor="end">{fmtRate(t)}</text>
            <line x1={PAD.l - 4} y1={ly(t)} x2={PAD.l} y2={ly(t)} stroke="#888" />
          </g>
        ))}
        <text x={(W + PAD.l) / 2} y={H - 6} fontSize={11} fill="#666" textAnchor="middle">arithmetic intensity (FLOP per byte)</text>
        <path d={rooflinePath(bf16Peak)} fill="none" stroke="#999" strokeDasharray="4 3" />
        {fp8Peak !== null && <path d={rooflinePath(fp8Peak)} fill="none" stroke="#bbb" strokeDasharray="2 3" />}
        {points.map((p) => (
          <circle key={p.id} cx={lx(Math.max(p.intensity, 1e-2))} cy={ly(p.attainableFlops)} r={p.bound === "memory" ? 3.4 : 2.6}
            fill={KIND_COLOR[p.kind] ?? "#555"} opacity={0.85} stroke={p.precision === "fp8" ? "#d1495b" : "none"} strokeWidth={1.2}>
            <title>{`${p.id} (${p.precision}, stage ${p.stage + 1})\nintensity ${p.intensity.toFixed(1)} FLOP/B → ${p.bound}-bound\nattainable ${fmtRate(p.attainableFlops)} FLOP/s`}</title>
          </circle>
        ))}
      </svg>
      <p className="lab-note">
        Roofline (Williams et al. 2009) on the evaluator's own accounting: work is forward FLOPs × recompute multiplier; bytes move the
        training states once plus one activation pass. Dashed lines: bf16 and fp8 rooflines at {(bw / 1e12).toFixed(1)} TB/s device HBM.
        Left of the ridge = bandwidth-bound (larger dots). Red ring = fp8 node. The work and state sizes are exactly the ones behind the
        step-time and memory numbers above.
      </p>
    </div>
  );
}

/** Memory waterfall ---------------------------------------------------------------- */

export function MemoryWaterfall({ problem, plan, locks, memoryCapBytes }: { problem: LayerTrainingProblem; plan: LayerTrainingPlan; locks: LayerLocks; memoryCapBytes: number | null }) {
  const timeline = useMemo(() => memoryTimeline(problem, plan, locks), [problem, plan, locks]);
  const stages = timeline.stages;
  if (!stages.length) return null;
  const H = 30 + stages.length * 34;
  const maxBytes = Math.max(...stages.map((s) => s.forwardPeakBytes), memoryCapBytes ?? 0, 1);
  const x = (v: number) => PAD.l + (v / maxBytes) * (W - PAD.l - PAD.r);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-stage memory across one micro-batch">
        {stages.map((s) => {
          const y = 22 + s.stage * 34;
          return (
            <g key={s.stage}>
              <rect x={PAD.l} y={y} width={Math.max(0.5, x(s.baselineBytes) - PAD.l)} height={24} fill="#31688e" opacity={0.9}>
                <title>{`Stage ${s.stage + 1} persistent states (weights + grads + master + Adam) + overhead: ${fmtBytes(s.baselineBytes)}`}</title>
              </rect>
              <rect x={x(s.baselineBytes)} y={y} width={Math.max(0.5, x(s.forwardPeakBytes) - x(s.baselineBytes))} height={24} fill="#7a5195" opacity={0.8}>
                <title>{`Stage ${s.stage + 1} activations during forward (offload keeps 15%): ${fmtBytes(s.forwardPeakBytes - s.baselineBytes)}`}</title>
              </rect>
              <text x={PAD.l - 8} y={y + 16} textAnchor="end" fontSize={11} fill="#666">S{s.stage + 1}</text>
              <text x={x(s.forwardPeakBytes) + 6} y={y + 16} fontSize={10.5} fill="#555">{fmtBytes(s.forwardPeakBytes)}</text>
            </g>
          );
        })}
        {memoryCapBytes !== null && memoryCapBytes > 0 && memoryCapBytes <= maxBytes && (
          <g>
            <line x1={x(memoryCapBytes)} y1={PAD.t} x2={x(memoryCapBytes)} y2={H - 26} stroke="#d1495b" strokeDasharray="5 4" />
            <text x={x(memoryCapBytes) + 4} y={PAD.t + 10} fontSize={10.5} fill="#d1495b">device limit {fmtBytes(memoryCapBytes)}</text>
          </g>
        )}
      </svg>
      <p className="lab-note">
        Per stage: solid = persistent training states + overhead; purple = activations at the forward peak, released after backward so
        the step ends back at the baseline. Peaks are the evaluator's own memory numbers (states include grads, the fp32 master copy and
        Adam moments for the whole step; offloaded activations keep 15% resident). The timeline covers one micro-batch; this plan runs{" "}
        {timeline.microBatches} per step.
      </p>
    </div>
  );
}

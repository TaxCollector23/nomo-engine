import { useMemo, useState } from "react";

import { feasible, type InferencePack, type Metrics, type Plan } from "../planner/packs";
import type { Evaluated } from "../planner/search";

/** Concurrency sweep for module 04: every batch size x precision x TP combination,
 * evaluated through the same InferencePack instance the recommendation uses. */

const W = 560;
const H = 190;
const PAD = { l: 56, r: 14, t: 12, b: 34 };

const SERIES_COLOR: Record<string, string> = { bf16: "#2f6f4f", fp8: "#31688e", int4: "#7a5195" };

function logTicks(min: number, max: number): number[] {
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    const v = 10 ** e;
    if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v);
  }
  return ticks;
}

interface SweepPoint { batch: number; weights: string; tp: number; kv: string; m: Metrics }

export function ServingSweep({ pack, selected }: { pack: InferencePack; selected: Evaluated | null }) {
  const batches = pack.variables().find((v) => v.name === "batch")!.choices.map(Number);
  const tps = pack.variables().find((v) => v.name === "tp")!.choices.map(Number);
  const precisions = pack.variables().find((v) => v.name === "weights")!.choices.map(String);
  const kvs = pack.variables().find((v) => v.name === "kv")!.choices.map(String);
  const selKv = selected ? String(selected.plan.kv) : "bf16";
  const [kv, setKv] = useState(selKv);
  const [tp, setTp] = useState<number>(selected ? Number(selected.plan.tp) : tps[0]);

  const sweep = useMemo<SweepPoint[]>(() => {
    const rows: SweepPoint[] = [];
    for (const w of precisions) for (const t of tps) for (const b of batches) for (const k of kvs) {
      rows.push({ batch: b, weights: w, tp: t, kv: k, m: pack.evaluate({ weights: w, kv: k, tp: t, batch: b }) });
    }
    return rows;
  }, [pack, batches, tps, precisions, kvs]);

  const rows = useMemo(
    () => sweep.filter((r) => r.kv === kv && r.tp === tp),
    [sweep, kv, tp],
  );
  const series = precisions.map((w) => ({
    weights: w,
    points: batches.map((b) => rows.find((r) => r.weights === w && r.batch === b)).filter(Boolean) as SweepPoint[],
  }));

  const selPoint = selected
    ? sweep.find((r) => r.weights === String(selected.plan.weights) && r.kv === String(selected.plan.kv)
      && r.tp === Number(selected.plan.tp) && r.batch === Number(selected.plan.batch)) ?? null
    : null;

  const charts = [
    { key: "tokens_per_s_per_gpu", label: "output tokens / s / GPU", fmt: (v: number) => v.toFixed(0), logY: true },
    { key: "tbt_static_ms", label: "time between tokens (ms)", fmt: (v: number) => v.toFixed(1), logY: true },
    { key: "usd_per_mtok", label: "USD per 1M output tokens", fmt: (v: number) => v.toFixed(2), logY: true },
  ] as const;

  const renderChart = (chart: (typeof charts)[number]) => {
    const vals = rows.flatMap((r) => [Number(r.m.breakdown[chart.key])]).filter((v) => Number.isFinite(v) && v > 0);
    if (!vals.length) return null;
    const minY = Math.min(...vals) / 1.4, maxY = Math.max(...vals) * 1.4;
    const lx = (b: number) => {
      const lo = Math.log10(batches[0]!), hi = Math.log10(batches[batches.length - 1]!);
      return PAD.l + ((Math.log10(b) - lo) / Math.max(1e-9, hi - lo)) * (W - PAD.l - PAD.r);
    };
    const ly = (v: number) => H - PAD.b - ((Math.log10(Math.max(v, minY)) - Math.log10(minY)) / Math.max(1e-9, Math.log10(maxY) - Math.log10(minY))) * (H - PAD.t - PAD.b);
    return (
      <svg key={chart.key} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${chart.label} versus batch size`}>
        <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#ccc" />
        <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="#ccc" />
        {logTicks(batches[0]!, batches[batches.length - 1]!).map((b) => (
          <g key={b}>
            <text x={lx(b)} y={H - PAD.b + 15} fontSize={10} fill="#888" textAnchor="middle">{b}</text>
            <line x1={lx(b)} y1={H - PAD.b} x2={lx(b)} y2={H - PAD.b + 4} stroke="#888" />
          </g>
        ))}
        <text x={PAD.l - 6} y={ly(maxY) + 10} fontSize={10} fill="#888" textAnchor="end">{chart.fmt(maxY)}</text>
        <text x={PAD.l - 6} y={ly(minY)} fontSize={10} fill="#888" textAnchor="end">{chart.fmt(minY)}</text>
        <text x={(W + PAD.l) / 2} y={H - 4} fontSize={10.5} fill="#666" textAnchor="middle">concurrent sequences (batch)</text>
        {series.map((s) => {
          const good = s.points.filter((p) => feasible(p.m));
          const line = good.map((p) => `${lx(p.batch).toFixed(1)},${ly(Number(p.m.breakdown[chart.key])).toFixed(1)}`).join(" ");
          return (
            <g key={s.weights}>
              {line && <polyline points={line} fill="none" stroke={SERIES_COLOR[s.weights] ?? "#555"} strokeWidth={1.8} />}
              {s.points.map((p) => (
                <circle key={`${s.weights}-${p.batch}`} cx={lx(p.batch)} cy={ly(Number(p.m.breakdown[chart.key]))}
                  r={feasible(p.m) ? 2.6 : 1.8} fill={feasible(p.m) ? (SERIES_COLOR[s.weights] ?? "#555") : "none"}
                  stroke={SERIES_COLOR[s.weights] ?? "#555"} strokeWidth={1}>
                  <title>{`${s.weights} weights, ${tp} GPU${tp === 1 ? "" : "s"}, batch ${p.batch}${feasible(p.m) ? "" : " — infeasible (memory or quality budget)"}\n${chart.label}: ${chart.fmt(Number(p.m.breakdown[chart.key]))}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
        {selPoint && Number.isFinite(Number(selPoint.m.breakdown[chart.key])) && (
          <g>
            <line x1={lx(Number(selPoint.batch))} y1={PAD.t} x2={lx(Number(selPoint.batch))} y2={H - PAD.b} stroke="#d1495b" strokeDasharray="4 3" opacity={0.7} />
            <circle cx={lx(Number(selPoint.batch))} cy={ly(Number(selPoint.m.breakdown[chart.key]))} r={5}
              fill="none" stroke="#d1495b" strokeWidth={2}>
              <title>{`selected plan: ${selPoint.weights} weights, ${selPoint.kv} KV, ${selPoint.tp} GPU${selPoint.tp === 1 ? "" : "s"}, batch ${selPoint.batch}`}</title>
            </circle>
          </g>
        )}
      </svg>
    );
  };

  return (
    <div>
      <div className="lab-axes" role="group" aria-label="Sweep axes">
        <label className="lab-muted">KV precision
          <select value={kv} onChange={(e) => setKv(e.target.value)} aria-label="KV cache precision">
            {kvs.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="lab-muted">GPUs per copy
          <select value={String(tp)} onChange={(e) => setTp(Number(e.target.value))} aria-label="GPUs per model copy">
            {tps.map((t) => <option key={t} value={String(t)}>{t}</option>)}
          </select>
        </label>
      </div>
      <div className="lab-bar-legend" style={{ marginBottom: 6 }}>
        {precisions.map((w) => <span key={w}><i style={{ background: SERIES_COLOR[w] ?? "#555", width: 10, height: 10, display: "inline-block", marginRight: 4, borderRadius: 2 }} />{w} weights</span>)}
        <span><i style={{ border: "2px solid #d1495b", width: 8, height: 8, display: "inline-block", marginRight: 4, borderRadius: 8 }} />selected plan</span>
      </div>
      {charts.map((c) => (
        <div key={c.key}>
          {renderChart(c)}
          <p className="lab-muted" style={{ fontSize: 12, margin: "0 0 8px" }}>{c.label}</p>
        </div>
      ))}
      <p className="lab-note">
        Hollow points are infeasible under the memory or quality budget. TBT and TTFT are the pack's static-batching first-order
        decomposition (a decode step waits for the whole batch; TTFT is the prefill), and quality-loss values remain assumptions until
        you enter your own. Sweep evaluated {sweep.length.toLocaleString("en-US")} configurations through the same pack as the
        recommendation.
      </p>
    </div>
  );
}

export type { Plan };

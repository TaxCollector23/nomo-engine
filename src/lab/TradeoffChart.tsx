import { useEffect, useMemo, useRef, useState } from "react";

import { feasible, type Objective } from "../planner/packs";
import type { Evaluated, Result } from "../planner/search";
import { uncertaintyForPlan, type UncertaintyResult } from "../planner/uncertainty";
import { fmtNum } from "./controls";

interface Props {
  res: Result;
  uncertainty?: UncertaintyResult | null;
  objectives: Objective[];
  xAxis: string;
  yAxis: string;
  selected: Evaluated | null;
  recommended: Evaluated | null;
  ghost?: Evaluated | null;
  onSelect: (e: Evaluated) => void;
  animate: boolean;
  describe: (e: Evaluated) => string;
  compact?: boolean;
}

/** Every evaluated plan on log axes. Grey = feasible, pale = breaks a limit, ink = best trade-offs.
 *  With animation on, the exhaustive search is replayed: plans appear in evaluation order. */
export default function TradeoffChart({ res, uncertainty, objectives, xAxis, yAxis, selected, recommended, ghost, onSelect, animate, describe, compact }: Props) {
  const W = 760, H = compact ? 320 : 420, pl = 78, pb = 46, pr = 18, pt = 16;
  const ox = objectives.find((o) => o.name === xAxis) ?? objectives[0]!;
  const oy = objectives.find((o) => o.name === yAxis && o.name !== ox.name) ?? objectives.find((o) => o !== ox)!;
  xAxis = ox.name; yAxis = oy.name;
  const [shown, setShown] = useState(res.all.length);
  const [hover, setHover] = useState<Evaluated | null>(null);
  const raf = useRef(0);

  useEffect(() => {
    cancelAnimationFrame(raf.current);
    const reduce = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!animate || reduce) { setShown(res.all.length); return; }
    const start = performance.now(), dur = Math.min(1400, 300 + res.all.length / 8);
    const tick = (t: number) => {
      const f = Math.min(1, (t - start) / dur);
      setShown(Math.round(res.all.length * (1 - (1 - f) ** 2)));
      if (f < 1) raf.current = requestAnimationFrame(tick);
    };
    setShown(0);
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [res, animate]);

  const frontKeys = useMemo(() => new Set(res.front), [res]);
  const geo = useMemo(() => {
    const val = (e: Evaluated, n: string) => e.metrics.objectives[n]!;
    const ok = res.all.filter((e) => Number.isFinite(val(e, xAxis)) && Number.isFinite(val(e, yAxis)) && val(e, xAxis) > 0 && val(e, yAxis) > 0 && val(e, xAxis) < 1e29);
    const base = ok.filter((e) => feasible(e.metrics));
    const span = base.length >= 2 ? base : ok;
    const lg = (v: number) => Math.log10(v);
    let x0 = Math.min(...span.map((e) => lg(val(e, xAxis)))), x1 = Math.max(...span.map((e) => lg(val(e, xAxis))));
    let y0 = Math.min(...span.map((e) => lg(val(e, yAxis)))), y1 = Math.max(...span.map((e) => lg(val(e, yAxis))));
    if (x1 - x0 < 1e-6) { x0 -= 0.1; x1 += 0.1; }
    if (y1 - y0 < 1e-6) { y0 -= 0.1; y1 += 0.1; }
    const px = (x1 - x0) * 0.04, py = (y1 - y0) * 0.06;
    x0 -= px; x1 += px; y0 -= py; y1 += py;
    const X = (v: number) => pl + ((lg(v) - x0) / (x1 - x0)) * (W - pl - pr);
    const Y = (v: number) => H - pb - ((lg(v) - y0) / (y1 - y0)) * (H - pb - pt);
    const inBox = (e: Evaluated) => { const x = X(val(e, xAxis)), y = Y(val(e, yAxis)); return x >= pl - 2 && x <= W - pr + 2 && y >= pt - 2 && y <= H - pb + 2; };
    return { ok, X, Y, x0, x1, y0, y1, val, inBox };
  }, [res, xAxis, yAxis, H]);

  const visible = geo.ok.slice(0, shown);
  const ticks = (a: number, b: number) => [0, 1 / 3, 2 / 3, 1].map((f) => 10 ** (a + f * (b - a)));
  const pt2 = (e: Evaluated) => [geo.X(geo.val(e, xAxis)), geo.Y(geo.val(e, yAxis))] as const;
  const frontSorted = res.front.filter(geo.inBox).slice().sort((a, b) => geo.val(a, xAxis) - geo.val(b, xAxis));
  const showLine = objectives.length === 2;
  const progress = shown < res.all.length;

  return (
    <div className="lab-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${res.all.length} plans; ${res.front.length} best trade-offs on ${ox.label} versus ${oy.label}`}>
        <rect x={pl} y={pt} width={W - pl - pr} height={H - pb - pt} className="lab-chart-frame" />
        {ticks(geo.x0, geo.x1).map((v, i) => (
          <g key={`x${i}`}>
            <line x1={geo.X(v)} x2={geo.X(v)} y1={pt} y2={H - pb} className="lab-chart-grid" />
            <text x={geo.X(v)} y={H - pb + 16} textAnchor={i === 0 ? "start" : i === 3 ? "end" : "middle"} className="lab-chart-tick">{fmtNum(v, ox.unit)}</text>
          </g>
        ))}
        {ticks(geo.y0, geo.y1).map((v, i) => (
          <g key={`y${i}`}>
            <line x1={pl} x2={W - pr} y1={geo.Y(v)} y2={geo.Y(v)} className="lab-chart-grid" />
            <text x={pl - 8} y={geo.Y(v) + 4} textAnchor="end" className="lab-chart-tick">{fmtNum(v, oy.unit)}</text>
          </g>
        ))}
        <text x={(W + pl) / 2} y={H - 8} textAnchor="middle" className="lab-chart-axis">{ox.label} →</text>
        <text x={14} y={(H - pb + pt) / 2} textAnchor="middle" className="lab-chart-axis" transform={`rotate(-90 14 ${(H - pb + pt) / 2})`}>{oy.label} →</text>
        <g clipPath="url(#lab-clip)">
          <clipPath id="lab-clip"><rect x={pl} y={pt} width={W - pl - pr} height={H - pb - pt} /></clipPath>
          {visible.map((e, i) => {
            const [x, y] = pt2(e);
            const feas = feasible(e.metrics);
            if (frontKeys.has(e)) return null;
            return <circle key={i} cx={x} cy={y} r={feas ? 2.2 : 1.6} className={feas ? "lab-pt" : "lab-pt-bad"} />;
          })}
          {!progress && showLine && frontSorted.length > 1 && (
            <polyline className="lab-front-line" points={frontSorted.map((e) => pt2(e).join(",")).join(" ")} />
          )}
          {!progress && res.front.map((e, i) => {
            const [x, y] = pt2(e);
            const intervals = uncertaintyForPlan(uncertainty, e.plan);
            const xi = intervals?.objectives[xAxis], yi = intervals?.objectives[yAxis];
            return (
              <g key={`f${i}`}>
                {xi && <line x1={geo.X(xi.low)} x2={geo.X(xi.high)} y1={y} y2={y} className="lab-pt-uncertainty" />}
                {yi && <line x1={x} x2={x} y1={geo.Y(yi.low)} y2={geo.Y(yi.high)} className="lab-pt-uncertainty" />}
                <circle cx={x} cy={y} r={selected === e ? 6.5 : 4.5} className={`lab-pt-front${selected === e ? " is-selected" : ""}`}
                  tabIndex={0} role="button" aria-label={`${describe(e)}${xi ? `; ${ox.label} 90% interval ${xi.low.toPrecision(3)} to ${xi.high.toPrecision(3)}` : ""}`}
                  onMouseEnter={() => setHover(e)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(e)} onBlur={() => setHover(null)}
                  onClick={() => onSelect(e)} onKeyDown={(k) => { if (k.key === "Enter" || k.key === " ") { k.preventDefault(); onSelect(e); } }} />
              </g>
            );
          })}
          {!progress && recommended && (() => { const [x, y] = pt2(recommended); return <circle cx={x} cy={y} r={11} className="lab-pt-rec" />; })()}
          {!progress && ghost && geo.inBox(ghost) && (() => { const [x, y] = pt2(ghost); return <circle cx={x} cy={y} r={7} className="lab-pt-ghost" />; })()}
        </g>
        {hover && (() => {
          const [x, y] = pt2(hover);
          const left = x > W * 0.6;
          return (
            <g className="lab-chart-tip" transform={`translate(${left ? x - 12 : x + 12},${Math.max(pt + 14, y - 8)})`}>
              <text textAnchor={left ? "end" : "start"}>
                <tspan x={0} dy={0} className="lab-chart-tip-strong">{fmtNum(geo.val(hover, xAxis), ox.unit)}, {fmtNum(geo.val(hover, yAxis), oy.unit)}</tspan>
                <tspan x={0} dy={15}>{describe(hover).slice(0, 78)}{describe(hover).length > 78 ? "…" : ""}</tspan>
              </text>
            </g>
          );
        })()}
      </svg>
      <div className="lab-chart-legend" aria-hidden="true">
        <span><i className="lg-front" />best trade-offs</span>
        {uncertainty?.available && <span><i className="lg-uncertainty" />90% interval</span>}
        <span><i className="lg-pt" />other feasible plans</span>
        <span><i className="lg-bad" />breaks a limit</span>
        <span><i className="lg-rec" />recommended</span>
        {ghost && <span><i className="lg-ghost" />what-if</span>}
        <span className="lab-chart-count">{progress ? `checking… ${shown.toLocaleString("en-US")} / ${res.all.length.toLocaleString("en-US")}` : `${res.all.length.toLocaleString("en-US")} plans checked`}</span>
      </div>
    </div>
  );
}

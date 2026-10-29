import type { SpecMcResult } from "../planner/specUncertainty";

/** Tornado attribution: which spec input drives the remaining error.
 * Rendered under the headline for uncalibrated hardware (explore/rigor modes). */
export function SpecTornado({ result, objective, label }: { result: SpecMcResult; objective: string; label: string }) {
  const rows = result.tornado[objective] ?? [];
  if (!rows.length || rows.every((row) => row.swingPct < 0.05)) return null;
  const max = Math.max(...rows.map((row) => row.swingPct));
  return (
    <div className="spec-tornado" aria-label={`What drives the error in ${label}`}>
      <p className="lab-note">What drives the error in <b>{label}</b> — swing when each spec input moves across its disclosed range:</p>
      {rows.map((row) => (
        <div key={row.name} className="spec-tornado-row" title={`${row.label}: ${row.swingPct.toFixed(1)}% swing`}>
          <span className="spec-tornado-label">{row.label}</span>
          <span className="spec-tornado-track">
            <i style={{ width: `${Math.max(2, (row.swingPct / max) * 100)}%` }} />
          </span>
          <span className="spec-tornado-value">±{row.swingPct < 10 ? row.swingPct.toFixed(1) : row.swingPct.toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

/** The headline band: "90%: x–y · spec-range Monte Carlo (400 samples)". */
export function specBandLine(result: SpecMcResult, objective: string): string | null {
  const interval = result.intervals[objective];
  if (!interval) return null;
  const fmt = (v: number) => (v >= 1000 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3));
  return `90%: ${fmt(interval.p10)}–${fmt(interval.p90)} · spec-range MC, ${result.samples} samples`;
}

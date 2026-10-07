import { useMemo, useState } from "react";

import { CALIBRATED, CALIBRATION_RESULTS, calibrationPoints } from "../planner/calibration";
import { UNCERTAINTY_VALIDATION } from "../planner/uncertainty";
import { Segmented } from "./controls";
import VerificationBenchmarks from "./VerificationBenchmarks";

/** Predicted vs measured step time for 22 published runs. Toggling calibration moves every point. */
export default function Evidence() {
  const pts = useMemo(() => calibrationPoints(), []);
  const [which, setWhich] = useState<"calibrated" | "uncalibrated">("uncalibrated");
  const [hover, setHover] = useState<number | null>(null);
  const W = 560, H = 440, p = 58;
  const all = pts.flatMap((q) => [q.measured, q.predicted, q.uncalibrated]);
  const lo = Math.log10(Math.min(...all)) - 0.08, hi = Math.log10(Math.max(...all)) + 0.08;
  const S = (v: number) => p + ((Math.log10(v) - lo) / (hi - lo)) * (W - 2 * p);
  const Sy = (v: number) => H - p - ((Math.log10(v) - lo) / (hi - lo)) * (H - 2 * p);
  const err = (q: (typeof pts)[number]) => (100 * Math.abs((which === "calibrated" ? q.predicted : q.uncalibrated) - q.measured)) / q.measured;
  const mean = pts.reduce((a, q) => a + err(q), 0) / pts.length;
  const ticks = [1, 3, 10, 30, 100, 300].filter((v) => Math.log10(v) > lo && Math.log10(v) < hi);
  const cal = CALIBRATED.a100_nvlink_ib!;

  return (
    <div className="lab-evidence">
      <div className="lab-evidence-head">
        <div>
          <p className="section-kicker">Evidence</p>
          <h2>Does the planner predict reality?</h2>
          <p className="lab-lede">
            We transcribed 22 measured training runs from NVIDIA&apos;s Megatron-LM paper (models from 1.7 billion to 1 trillion
            parameters on up to 3,072 A100 GPUs) and asked the planner to predict each one. Each dot is a real run: on the
            diagonal means a perfect prediction. Switch between the placeholder model and the calibrated one.
          </p>
        </div>
        <div className="lab-stat-row">
          <div className="lab-stat"><b>{CALIBRATION_RESULTS.heldOutPtdMape}%</b><span>average error on runs the model never saw</span></div>
          <div className="lab-stat"><b>{CALIBRATION_RESULTS.heldOutSpearman}</b><span>rank agreement (Spearman) on unseen runs</span></div>
          <div className="lab-stat"><b>{cal.looMapePct}%</b><span>leave-one-out error, all 22 runs</span></div>
          <div className="lab-stat"><b>{UNCERTAINTY_VALIDATION.covered}/{UNCERTAINTY_VALIDATION.n}</b><span>held-out rows inside the nominal 90% interval</span></div>
        </div>
      </div>
      <div className="lab-evidence-body">
        <div className="lab-card">
          <div className="lab-card-bar">
            <Segmented label="Prediction model" value={which} onChange={setWhich} options={[
              { value: "uncalibrated", label: "Placeholder numbers" },
              { value: "calibrated", label: "Calibrated on published data" },
            ]} />
            <span className="lab-muted">mean error <b>{mean.toFixed(1)}%</b></span>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="lab-evidence-plot" role="img"
            aria-label={`Predicted versus measured step time, ${which}; mean error ${mean.toFixed(1)} percent`}>
            <rect x={p} y={p} width={W - 2 * p} height={H - 2 * p} className="lab-chart-frame" />
            {ticks.map((v) => (
              <g key={v}>
                <line x1={S(v)} x2={S(v)} y1={p} y2={H - p} className="lab-chart-grid" />
                <line x1={p} x2={W - p} y1={Sy(v)} y2={Sy(v)} className="lab-chart-grid" />
                <text x={S(v)} y={H - p + 16} textAnchor="middle" className="lab-chart-tick">{v} s</text>
                <text x={p - 8} y={Sy(v) + 4} textAnchor="end" className="lab-chart-tick">{v} s</text>
              </g>
            ))}
            <line x1={S(10 ** lo)} y1={Sy(10 ** lo)} x2={S(10 ** hi)} y2={Sy(10 ** hi)} className="lab-diag" />
            <text x={W - p - 6} y={p + 16} textAnchor="end" className="lab-chart-tick">perfect prediction</text>
            <text x={W / 2} y={H - 12} textAnchor="middle" className="lab-chart-axis">measured time per training step (published)</text>
            <text x={14} y={H / 2} textAnchor="middle" className="lab-chart-axis" transform={`rotate(-90 14 ${H / 2})`}>predicted by Nomo</text>
            {pts.map((q, i) => {
              const pred = which === "calibrated" ? q.predicted : q.uncalibrated;
              return (
                <g key={i} className="lab-ev-pt" style={{ transform: `translate(${S(q.measured)}px, ${Sy(pred)}px)` }}
                  onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                  <circle r={hover === i ? 7 : 5} className={q.strategy === "ZeRO-3" ? "z3" : "ptd"} />
                </g>
              );
            })}
          </svg>
          <div className="lab-chart-legend">
            <span><i className="lg-front" />tensor + pipeline parallel (16 runs)</span>
            <span><i className="lg-hollow" />ZeRO-3 (6 runs)</span>
            <span className="lab-chart-count">{hover !== null ? `${pts[hover]!.label}: measured ${pts[hover]!.measured.toFixed(2)} s, predicted ${(which === "calibrated" ? pts[hover]!.predicted : pts[hover]!.uncalibrated).toFixed(2)} s` : "hover a point for details"}</span>
          </div>
        </div>
        <div className="lab-card lab-evidence-notes">
          <h3>How the test was run</h3>
          <ol>
            <li>Fit five physical parameters on Table 1 only (10 runs).</li>
            <li>Predict the 12 runs of Table 2, which the fit never saw: {CALIBRATION_RESULTS.heldOutPtdMape}% error on the 6 runs of the same strategy, correct ranking of all 12 (Kendall {CALIBRATION_RESULTS.heldOutKendall}).</li>
            <li>ZeRO-3, a strategy absent from Table 1, needs its own data: fitted on the 175B ZeRO-3 runs, the model predicts the 530B ones within {CALIBRATION_RESULTS.zero3TransferMape}%.</li>
          </ol>
          <h3>What we report honestly</h3>
          <ul>
            <li>Only A100 training is calibrated so far; other hardware is labelled uncalibrated everywhere in the Lab.</li>
            <li>With six ZeRO-3 runs, two of its parameters cannot be told apart; we report the simpler model, not the more flattering fit.</li>
            <li>Measured step times are derived from the paper&apos;s own FLOP formula; throughput values are copied exactly.</li>
            <li>The nominal 90% interval covered {UNCERTAINTY_VALIDATION.covered}/{UNCERTAINTY_VALIDATION.n} rows ({(UNCERTAINTY_VALIDATION.actual * 100).toFixed(1)}%); this is checked-in validation evidence, not a future guarantee.</li>
          </ul>
          <p className="lab-muted">Source: Narayanan et al., <i>Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM</i>, SC21 (<a href="https://arxiv.org/abs/2104.04473" target="_blank" rel="noreferrer">arXiv:2104.04473</a>), Tables 1–2.</p>
        </div>
      </div>
      <details className="lab-card lab-table-card">
        <summary>All 22 runs</summary>
        <div className="lab-table-scroll">
          <table className="lab-table">
            <thead><tr><th>Run</th><th>Table</th><th>Measured</th><th>Placeholder</th><th>Calibrated</th><th>Error</th></tr></thead>
            <tbody>
              {pts.map((q, i) => (
                <tr key={i}>
                  <td>{q.label}</td><td>{q.source.split(" ").slice(-1)[0]}</td><td>{q.measured.toFixed(2)} s</td>
                  <td>{q.uncalibrated.toFixed(2)} s</td><td>{q.predicted.toFixed(2)} s</td>
                  <td>{((100 * Math.abs(q.predicted - q.measured)) / q.measured).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <VerificationBenchmarks />
    </div>
  );
}

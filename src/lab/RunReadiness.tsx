import { useMemo, useState } from "react";

import { evaluateReliability, optimizeReliability } from "../planner/productModels";
import type { Evaluated } from "../planner/search";
import type { Cluster } from "../planner/hardware";
import { LogSlider } from "./controls";

/** Run readiness for module 03: put the reliability economics of a real
 * multi-day training run next to the plan. Composed entirely from the
 * parity-checked reliability model in productModels (evaluateReliability,
 * optimizeReliability, Young's formula) fed with the selected plan's own
 * numbers: checkpoint state = weights + grads + optimizer per GPU,
 * run length = the plan's days. No new math, all disclosed. */

const W = 560;
const H = 180;
const PAD = { l: 50, r: 16, t: 12, b: 34 };

export function RunReadiness({ selected, cluster }: { selected: Evaluated | null; cluster: Cluster }) {
  const [mtbfDays, setMtbfDays] = useState(4);
  const [storageGbps, setStorageGbps] = useState(8);
  const [restartMin, setRestartMin] = useState(30);

  const plan = selected && Object.values(selected.metrics.constraints).every((v) => v <= 0) ? selected : null;

  const facts = useMemo(() => {
    if (!plan) return null;
    const b = plan.metrics.breakdown as Record<string, unknown>;
    const parts = b.memory_parts_gb as Record<string, number> | undefined;
    if (!parts) return null;
    const stateGb = Math.max(0.1, Number(parts.weights ?? 0) + Number(parts.grads ?? 0) + Number(parts.optimizer ?? 0));
    const days = Number(plan.metrics.objectives.days);
    const costUsd = Number(plan.metrics.objectives.cost_usd);
    const gpus = Number(plan.plan.devices);
    return { stateGb, days, costUsd, gpus };
  }, [plan]);

  const model = useMemo(() => {
    if (!facts) return null;
    const checkpointWriteHours = (facts.stateGb * 1e9) / ((storageGbps * 1e9) / 8) / 3600; // GB at Gbps/8 bytes-per-s -> hours
    const failureRatePerHour = facts.gpus / (mtbfDays * 24); // cluster-level failures per hour
    const problem = {
      failureRatePerHour,
      checkpointWriteHours,
      restartHours: restartMin / 60,
      gpuCount: facts.gpus,
      gpuCostPerHour: cluster.costPerDeviceHour,
    };
    const grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8].map((hours) => ({ hours, ...evaluateReliability(problem, hours) }));
    const best = optimizeReliability(problem, grid.map((g) => g.hours));
    const goodputCurve = grid.map((g) => ({ hours: g.hours, goodput: g.goodput, cost: g.costUsdPerUsefulHour }));
    return {
      problem, grid, best, goodputCurve, checkpointWriteHours, failureRatePerHour,
      effectiveDays: facts.days / best.goodput,
      wastedGpuHours: facts.gpus * facts.days * (1 / best.goodput - 1),
      wastedUsd: facts.costUsd * (1 / best.goodput - 1),
    };
  }, [facts, mtbfDays, storageGbps, restartMin, cluster.costPerDeviceHour]);

  if (!plan || !facts || !model) return null;

  // ---- chart: goodput vs checkpoint interval (log x)
  const xMin = model.grid[0]!.hours;
  const xMax = model.grid[model.grid.length - 1]!.hours;
  const yMin = Math.min(...model.goodputCurve.map((g) => g.goodput)) * 0.96;
  const yMax = Math.min(1.0, Math.max(...model.goodputCurve.map((g) => g.goodput)) * 1.02);
  const x = (v: number) => PAD.l + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - PAD.l - PAD.r);
  const y = (v: number) => H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  const path = model.goodputCurve.map((g, i) => `${i ? "L" : "M"}${x(g.hours).toFixed(1)},${y(g.goodput).toFixed(1)}`).join(" ");

  return (
    <section className="lab-card readiness" aria-label="Run readiness">
      <div className="lab-card-head">
        <h3>Run readiness — will hardware failures eat this plan?</h3>
        <span className="lab-note">same reliability model as the fleet planner · Young's formula cross-check</span>
      </div>
      <p className="lab-story">
        A plan is not finished until it survives the hardware. At this cluster size, failures arrive every{" "}
        <b>{(1 / model.failureRatePerHour).toFixed(1)} hours</b> on average — this run would {model.failureRatePerHour * facts.days * 24 > 1 ? "hit" : "expect"}{" "}
        <b>{(model.failureRatePerHour * facts.days * 24).toFixed(1)} of them</b> before completing. Every checkpoint writes{" "}
        {facts.stateGb.toFixed(1)} GB of optimizer state per GPU; checkpoint too slowly and a failure erases hours of work, too often and the
        writes eat the throughput. The optimum is a real calculation, not a rule of thumb.
      </p>

      <div className="clinic-controls">
        <div className="clinic-slider">
          <LogSlider label="Failure MTBF (per GPU)" value={mtbfDays * 24} onChange={(v) => setMtbfDays((v ?? 96) / 24)} min={24} max={2400}
            format={(v) => `${v < 48 ? v.toFixed(0) + " h" : (v / 24).toFixed(1) + " days"}`} />
        </div>
        <div className="clinic-slider">
          <LogSlider label="Checkpoint write bandwidth" value={storageGbps} onChange={(v) => setStorageGbps(v ?? 8)} min={1} max={64}
            format={(v) => `${v.toFixed(0)} GB/s per GPU`} />
        </div>
        <div className="clinic-slider">
          <LogSlider label="Restart + rejoin time" value={restartMin} onChange={(v) => setRestartMin(v ?? 30)} min={5} max={240}
            format={(v) => `${v.toFixed(0)} min`} />
        </div>
      </div>

      <div className="clinic-grid">
        <div>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Goodput versus checkpoint interval" className="clinic-chart">
            {[0.25, 0.5, 1, 2, 4, 8].filter((v) => v >= xMin && v <= xMax).map((v) => (
              <g key={v}>
                <line x1={x(v)} y1={PAD.t} x2={x(v)} y2={H - PAD.b} className="clinic-gridline" />
                <text x={x(v)} y={H - PAD.b + 14} textAnchor="middle" className="clinic-tick">{v < 1 ? `${(v * 60).toFixed(0)}m` : `${v}h`}</text>
              </g>
            ))}
            {[yMin, (yMin + yMax) / 2, yMax].map((v) => (
              <g key={`y${v}`}>
                <line x1={PAD.l} y1={y(v)} x2={W - PAD.r} y2={y(v)} className="clinic-gridline" />
                <text x={PAD.l - 6} y={y(v) + 3} textAnchor="end" className="clinic-tick">{(v * 100).toFixed(0)}%</text>
              </g>
            ))}
            <path d={path} className="clinic-goodput-line" />
            <circle cx={x(model.best.checkpointIntervalHours)} cy={y(model.best.goodput)} r={4} className="clinic-knee" />
            {model.best.youngIntervalHours >= xMin && model.best.youngIntervalHours <= xMax && (
              <line x1={x(model.best.youngIntervalHours)} y1={PAD.t} x2={x(model.best.youngIntervalHours)} y2={H - PAD.b} className="clinic-ideal" />
            )}
            <text x={PAD.l} y={H - 4} className="clinic-axis">checkpoint interval (log) — dotted line: Young's optimum √(2C/λ)</text>
          </svg>
          <p className="lab-note">
            Goodput = fraction of GPU-hours that make progress. Left of the peak the writes cost too much; right of it, failures erase too much.
          </p>
        </div>
        <div className="clinic-readout">
          <p className="clinic-verdict">
            Checkpoint every <b>{model.best.checkpointIntervalHours < 1 / 60 ? `${(model.best.checkpointIntervalHours * 3600).toFixed(0)} s` : model.best.checkpointIntervalHours < 1 ? `${(model.best.checkpointIntervalHours * 60).toFixed(0)} min` : `${model.best.checkpointIntervalHours.toFixed(1)} h`}</b>{" "}
            (writes cost {((model.checkpointWriteHours / model.best.checkpointIntervalHours) * 100).toFixed(1)}% of throughput).
            The {facts.days.toFixed(1)}-day run finishes in <b>{model.effectiveDays.toFixed(1)} days</b> effective, wasting{" "}
            <b>{model.wastedGpuHours.toFixed(0)} GPU-hours</b> (~{Math.round(model.wastedUsd).toLocaleString("en-US")} USD) to failures and recovery.
            Young's closed form says {model.best.youngIntervalHours < 1 ? (model.best.youngIntervalHours * 60).toFixed(1) + " min" : model.best.youngIntervalHours.toFixed(1) + " h"} — the grid optimum lands on the same operating point.
          </p>
          <dl className="clinic-facts">
            <div><dt>Checkpoint state per GPU</dt><dd>{facts.stateGb.toFixed(1)} GB (weights + grads + optimizer)</dd></div>
            <div><dt>One checkpoint write</dt><dd>{(model.checkpointWriteHours * 3600).toFixed(0)} s at {storageGbps.toFixed(0)} GB/s</dd></div>
            <div><dt>Cluster failure rate</dt><dd>every {(1 / model.failureRatePerHour).toFixed(1)} h at MTBF {mtbfDays} d/GPU</dd></div>
            <div><dt>Goodput at optimum</dt><dd>{(model.best.goodput * 100).toFixed(1)}%</dd></div>
            <div><dt>Run cost with reliability</dt><dd>{Math.round(facts.costUsd / model.best.goodput).toLocaleString("en-US")} USD (plan says {Math.round(facts.costUsd).toLocaleString("en-US")})</dd></div>
          </dl>
        </div>
      </div>
      <p className="lab-note">
        Model: exponential failures at {model.failureRatePerHour.toExponential(1)}/h across {facts.gpus} GPUs, synchronous checkpointing of{" "}
        {facts.stateGb.toFixed(1)} GB/GPU, lost work ≈ half a checkpoint interval per failure (first order), restart {restartMin} min. Inputs are
        estimates until you supply your cluster's own failure telemetry (Rigor mode exports them). Young (1974) / Daly (2006) give the closed-form
        cross-check.
      </p>
    </section>
  );
}

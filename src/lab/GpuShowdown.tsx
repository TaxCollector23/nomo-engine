// GPU Showdown — the same workload, the same exhaustive search, on every cluster.
// Pure composition over the parity-checked build()/Planner math: no new formulas live here.
import { useDeferredValue, useMemo } from "react";

import { CALIBRATED } from "../planner/calibration";
import { CLUSTERS } from "../planner/hardware";
import { feasible, type Choice } from "../planner/packs";
import { build, withLocks, type Settings } from "../planner/registry";
import { Planner, type Evaluated } from "../planner/search";
import { fmtNum } from "./controls";

export type ShowdownDomain = "llm_training" | "llm_inference";

interface Row {
  key: string;
  clusterLabel: string;
  deviceName: string;
  memGb: number;
  bwTbs: number;
  calibrated: boolean;
  feasible: boolean;
  best: boolean;
  headline: string;
  rankValue: number;
  days: number | null;
  costUsd: number | null;
  usdPerMtok: number | null;
  msPerToken: number | null;
  planCount: number;
}

export default function GpuShowdown({ domain, settings, locks, weights, activeCluster, onPick }: {
  domain: ShowdownDomain;
  settings: Settings;
  locks: Record<string, Choice>;
  weights: number[] | null;
  activeCluster: string;
  onPick: (key: string) => void;
}) {
  const sig = JSON.stringify([domain, settings, locks, weights]);
  const deferred = useDeferredValue(sig);
  const stale = sig !== deferred;

  const rows = useMemo<Row[]>(() => {
    const [d, st, lk, w] = JSON.parse(deferred) as [ShowdownDomain, Settings, Record<string, Choice>, number[] | null];
    const out: Row[] = [];
    for (const c of Object.values(CLUSTERS)) {
      try {
        const pack = withLocks(build(d, { ...st, cluster: c.key }), lk);
        const res = new Planner(pack).run(w);
        const e: Evaluated | null = res.recommended ?? res.front[0] ?? res.closest[0] ?? null;
        if (!e) continue;
        const ok = res.front.length > 0 && feasible(e.metrics);
        const train = d === "llm_training";
        out.push({
          key: c.key,
          clusterLabel: c.label,
          deviceName: c.device.name,
          memGb: Math.round(c.device.memBytes / 1e9),
          bwTbs: Math.round((c.device.memBw / 1e12) * 10) / 10,
          calibrated: Boolean(CALIBRATED[c.key]),
          feasible: ok,
          best: false,
          headline: pack.describe(e.plan),
          rankValue: train ? e.metrics.objectives.days! : e.metrics.objectives.usd_per_mtok!,
          days: e.metrics.objectives.days ?? null,
          costUsd: e.metrics.objectives.cost_usd ?? null,
          usdPerMtok: e.metrics.objectives.usd_per_mtok ?? null,
          msPerToken: e.metrics.objectives.ms_per_token ?? null,
          planCount: res.evaluated,
        });
      } catch { /* this cluster cannot run the workload at all — leave it out rather than fake a row */ }
    }
    const feas = out.filter((r) => r.feasible).sort((a, b) => a.rankValue - b.rankValue);
    const rest = out.filter((r) => !r.feasible).sort((a, b) => a.rankValue - b.rankValue);
    if (feas[0]) feas[0].best = true;
    return [...feas, ...rest];
  }, [deferred]);

  const train = domain === "llm_training";
  const winner = rows.find((r) => r.best);
  const calibratedNames = rows.filter((r) => r.calibrated).map((r) => r.deviceName).join(", ");
  const uncalibratedNames = rows.filter((r) => !r.calibrated).map((r) => r.deviceName).join(", ");
  const active = rows.find((r) => r.key === activeCluster);
  const verdict = useMemo(() => {
    if (!winner || !active || active === winner) return null;
    const ratio = active.rankValue / winner.rankValue;
    if (ratio <= 1.02) return null;
    const metric = train ? "slower" : "costlier per token";
    return `${active.deviceName}'s best plan is ${ratio.toFixed(2)}× ${metric} than ${winner.deviceName} on this exact workload.`;
  }, [winner, active, train]);

  return (
    <section className="lab-card lab-showdown" aria-label="Compare every GPU on this exact workload">
      <div className="lab-card-head">
        <h3>Showdown — this exact workload on all {rows.length} clusters</h3>
        <span className={`lab-rel ${stale ? "warn" : "ok"}`}>{stale ? "re-running every search…" : "same search as the headline, per cluster"}</span>
      </div>
      <div className="lab-table-scroll showdown-scroll">
        <table className="lab-table showdown-table">
          <thead>
            <tr>
              <th>Cluster</th>
              <th>Best plan found</th>
              <th>{train ? "Training time" : "Cost / 1M output tokens"}</th>
              <th>{train ? "Training cost" : "Speed / token"}</th>
              <th>vs best</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className={r.key === activeCluster ? "sd-active" : r.feasible ? undefined : "sd-infeasible"}>
                <td>
                  <b>{r.deviceName}</b>
                  <small className="sd-sub">{r.memGb} GB · {r.bwTbs} TB/s HBM · {r.calibrated ? "calibrated" : "datasheet, uncalibrated"}</small>
                  {r.best && <span className="sd-chip">best {train ? "time" : "price"}</span>}
                </td>
                <td className="sd-plan">{r.headline}{r.feasible ? null : <small className="sd-sub"> — closest attempt; nothing meets your limits</small>}</td>
                <td className="sd-num"><b>{train ? fmtNum(r.days!, "days") : fmtNum(r.usdPerMtok!, "USD / 1M output tokens")}</b></td>
                <td className="sd-num">{train ? fmtNum(r.costUsd!, "USD") : fmtNum(r.msPerToken!, "ms")}</td>
                <td className="sd-num">{r.feasible ? (r.best ? <span className="sd-best">—</span> : `${(r.rankValue / winner!.rankValue).toFixed(2)}×`) : "—"}</td>
                <td>
                  {r.key === activeCluster
                    ? <span className="sd-inuse">in use</span>
                    : <button type="button" className="ui-button ui-button--compact" onClick={() => onPick(r.key)} disabled={!r.feasible}>Use this GPU</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {verdict && <p className="lab-note sd-verdict">{verdict}</p>}
      <p className="lab-note">
        Every row is the same exhaustive search as the headline ({rows[0]?.planCount.toLocaleString("en-US")} plans per cluster), re-run per cluster — nothing here is a lookup table.
        Efficiency is calibrated on {calibratedNames || "no cluster yet"}; {uncalibratedNames} use datasheet peak rates with placeholder efficiency and market-estimate prices, so compare rows against each other and treat absolute numbers as indicative.
      </p>
    </section>
  );
}

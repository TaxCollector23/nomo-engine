import type { CodesignPack, Pack } from "../planner/packs";
import type { Evaluated } from "../planner/search";
import { fmtNum } from "./controls";

function Bar({ parts, total, unit, capLabel }: { parts: [string, number][]; total?: number; unit: string; capLabel?: string }) {
  const sum = parts.reduce((a, [, v]) => a + v, 0);
  const cap = total ?? sum;
  const over = total !== undefined && sum > total;
  return (
    <div className="lab-bar-wrap">
      <div className={`lab-bar${over ? " is-over" : ""}`}>
        {parts.filter(([, v]) => v > 0).map(([k, v], i) => (
          <span key={k} className={`lab-bar-seg s${i}`} style={{ width: `${Math.min(100, (100 * v) / Math.max(cap, sum))}%` }}
            title={`${k}: ${v.toFixed(v < 1 ? 3 : 1)} ${unit}`} />
        ))}
        {total !== undefined && <span className="lab-bar-cap" style={{ left: `${Math.min(100, (100 * total) / Math.max(cap, sum))}%` }} title={capLabel} />}
      </div>
      <div className="lab-bar-legend">
        {parts.filter(([, v]) => v > 0).map(([k, v], i) => (
          <span key={k}><i className={`s${i}`} />{k.replace(/_/g, " ")} {v < 0.01 ? v.toExponential(1) : v.toFixed(v < 10 ? 2 : 0)} {unit}</span>
        ))}
        {total !== undefined && <span className="lab-bar-captext">{capLabel}: {total.toFixed(0)} {unit}{over ? " (exceeded)" : ""}</span>}
      </div>
    </div>
  );
}

function GpuGrid({ G, tp, pp }: { G: number; tp: number; pp: number }) {
  const dp = Math.max(1, Math.floor(G / (tp * pp)));
  const replicasShown = Math.min(dp, Math.max(1, Math.floor(256 / (tp * pp))));
  const shown = replicasShown * tp * pp;
  const shade = (s: number) => `color-mix(in srgb, var(--ink) ${pp > 1 ? 30 + (60 * s) / (pp - 1) : 70}%, var(--paper))`;
  return (
    <div>
      <div className="lab-gpugrid" aria-label={`${G} GPUs: ${dp} copies of ${pp} pipeline stages of ${tp} GPUs`}>
        {Array.from({ length: replicasShown }, (_, r) => (
          <div key={r} className="lab-replica" title={`data-parallel copy ${r + 1} of ${dp}`}>
            {Array.from({ length: pp }, (_, st) => (
              <div key={st} className="lab-stage" title={`pipeline stage ${st + 1} of ${pp}`}>
                {Array.from({ length: tp }, (_, t) => (
                  <span key={t} className="lab-gpu" style={{ background: shade(st), animationDelay: `${Math.min(600, (r * pp * tp + st * tp + t) * 2)}ms` }}
                    title={`copy ${r + 1}, stage ${st + 1} of ${pp}, tensor shard ${t + 1} of ${tp}`} />
                ))}
              </div>
            ))}
          </div>
        ))}
      </div>
      <p className="lab-note">
        Each square is one GPU. Darker squares hold later layers (pipeline stage 1 → {pp}); groups of {tp} split every layer's
        matrices (tensor parallel). The whole pattern repeats {dp} times as data-parallel copies
        {shown < G ? `; showing ${replicasShown} of ${dp} identical copies` : ""}.
      </p>
    </div>
  );
}

export default function Anatomy({ pack, e }: { pack: Pack; e: Evaluated }) {
  const b = e.metrics.breakdown as Record<string, unknown>;
  if (pack.name === "llm_training") {
    const mem = b.memory_parts_gb as Record<string, number>;
    const tim = b.time_parts_s as Record<string, number>;
    return (
      <div className="lab-anatomy">
        <h4>How the GPUs are used</h4>
        <GpuGrid G={Number(e.plan.devices)} tp={Number(e.plan.tp)} pp={Number(e.plan.pp)} />
        <h4>Memory on the busiest GPU</h4>
        <Bar parts={Object.entries(mem)} total={Number(b.memory_cap_gb)} unit="GB" capLabel="usable memory" />
        <h4>One training step ({fmtNum(Number(b.step_s), "s")}, {(Number(b.mfu) * 100).toFixed(0)}% of peak used)</h4>
        <Bar parts={Object.entries(tim)} unit="s" />
      </div>
    );
  }
  if (pack.name === "llm_inference") {
    const mem = b.memory_parts_gb as Record<string, number>;
    return (
      <div className="lab-anatomy">
        <h4>Memory on each GPU</h4>
        <Bar parts={Object.entries(mem)} total={Number(b.memory_cap_gb)} unit="GB" capLabel="usable memory" />
        <h4>Throughput</h4>
        <p className="lab-big-line">
          {fmtNum(Number(b.tokens_per_s_per_gpu), "tokens/s")} per GPU, generating one token every {fmtNum(e.metrics.objectives.ms_per_token!, "ms")}
          {" "}for each of {String(e.plan.batch)} conversations at once.
        </p>
        <p className="lab-note">
          Each new token must read every weight and the whole conversation cache from memory, so this plan is
          <b> {String(b.decode_bound)}-bound</b>: {b.decode_bound === "memory"
            ? "a faster chip helps less than smaller numbers (lower precision) or more conversations per read."
            : "arithmetic, not memory, limits speed; lower precision compute helps most."}
        </p>
      </div>
    );
  }
  const cp = pack as CodesignPack;
  const M = cp.model(e.plan);
  const serve = b.serving_plan as { tp: number; batch: number; ms: number } | null;
  const parts = b.cost_parts_musd as Record<string, number>;
  const layersShown = Math.min(M.layers, 40);
  return (
    <div className="lab-anatomy">
      <h4>The model&apos;s shape</h4>
      <div className="lab-model">
        <div className="lab-model-stack" aria-hidden="true">
          {Array.from({ length: layersShown }, (_, i) => (
            <span key={i} style={{ width: `${Math.min(100, (M.hidden / 16384) * 100)}%` }} />
          ))}
        </div>
        <div className="lab-model-facts">
          <p><b>{(Number(b.params) / 1e9).toFixed(1)}B parameters</b>: {M.layers} layers{M.layers > layersShown ? ` (${layersShown} drawn)` : ""}, each {M.hidden} wide</p>
          <div className="lab-heads" aria-label={`${M.heads} query heads share ${M.kvHeads} key-value heads`}>
            {Array.from({ length: Math.min(M.heads, 64) }, (_, i) => (
              <span key={i} className="q" style={{ borderColor: `hsl(${(Math.floor(i / Math.max(1, M.heads / M.kvHeads)) * 47) % 360} 40% 45%)` }} />
            ))}
          </div>
          <p className="lab-note">{M.heads} attention heads share {M.kvHeads} key-value head{M.kvHeads > 1 ? "s" : ""} ({String(e.plan.attention)}).
            Fewer key-value heads means a smaller memory cache per conversation, so cheaper serving.</p>
          <p className="lab-note">Trained on {Number(e.plan.tokens_per_param)} tokens per parameter
            ({(Number(b.train_tokens) / 1e12).toFixed(2)} trillion tokens){serve ? `; served on ${serve.tp} GPU${serve.tp > 1 ? "s" : ""} per copy, ${serve.batch} conversations at once, ${serve.ms.toFixed(1)} ms per token` : ""}.</p>
        </div>
      </div>
      <h4>Where the money goes (lifetime)</h4>
      <Bar parts={Object.entries(parts)} unit="$M" />
    </div>
  );
}

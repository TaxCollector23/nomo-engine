import { useMemo, useState } from "react";

import {
  buildLayerGraph,
  searchLayerTraining,
  type LayerGraph,
  type LayerPrecision,
  type LayerTrainingPlan,
  type RecomputeMode,
} from "../planner/layers";

type LayerLocks = Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>;

const DEMO_CONFIG = {
  _name_or_path: "demo-transformer",
  model_type: "llama",
  num_hidden_layers: 12,
  hidden_size: 768,
  intermediate_size: 2048,
  num_attention_heads: 12,
  num_key_value_heads: 4,
  vocab_size: 32000,
  tie_word_embeddings: true,
  hidden_act: "silu",
};

function formatBytes(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} MB`;
  return `${(value / 1e3).toFixed(0)} KB`;
}

function formatParams(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return `${(value / 1e3).toFixed(0)}K`;
}

function percentChange(value: number, baseline: number): string {
  const change = (value / baseline - 1) * 100;
  return `${change <= 0 ? "" : "+"}${change.toFixed(1)}%`;
}

function planSummary(graph: LayerGraph, plan: LayerTrainingPlan): string {
  const stages = Math.max(...plan.stages) + 1;
  const fp8 = plan.precision.filter((value) => value === "fp8").length;
  const recompute = plan.recompute.filter((value) => value !== "none").length;
  const offload = plan.offload.filter(Boolean).length;
  return `${stages} contiguous stage${stages === 1 ? "" : "s"}; ${fp8}/${graph.nodes.length} nodes FP8; ${recompute} recomputed; ${offload} offloaded`;
}

function nodeLabel(id: string): string {
  if (id === "embedding" || id === "output") return id;
  const [block, index, kind] = id.split(".");
  return `${block} ${Number(index) + 1} · ${kind}`;
}

export default function LayerPlanner({ mode }: { mode: "guided" | "explore" | "rigor" }) {
  const [configText, setConfigText] = useState(() => JSON.stringify(DEMO_CONFIG, null, 2));
  const [graph, setGraph] = useState<LayerGraph>(() => buildLayerGraph(DEMO_CONFIG, { seqLen: 2048, batchSize: 1, source: "demo config" }));
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"train" | "serve" | "neuromorphic">("train");
  const [selectedId, setSelectedId] = useState("block.5.mlp");
  const [locks, setLocks] = useState<LayerLocks>({});

  const result = useMemo(() => searchLayerTraining({
    graph,
    hardware: { devices: 8, memoryBytes: 80e9, usableMemory: 0.9, costPerDeviceHour: 3.5 },
    pipelineStages: [1, 2, 4, 8],
    microBatches: 8,
    maxCandidates: 20000,
  }, locks), [graph, locks]);
  const selected = graph.nodes.find((node) => node.id === selectedId) ?? graph.nodes[0]!;
  const selectedLock = locks[selected.id] ?? {};
  const best = result.best;
  const metrics = result.bestMetrics;
  const baseline = result.baseline.metrics;

  function parseConfig(nextText: string, source = "pasted config") {
    setConfigText(nextText);
    try {
      const parsed: unknown = JSON.parse(nextText);
      const nextGraph = buildLayerGraph(parsed, { seqLen: graph.seqLen, batchSize: graph.batchSize, source });
      setGraph(nextGraph);
      setSelectedId(nextGraph.nodes[Math.min(1, nextGraph.nodes.length - 1)]!.id);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  function setLock(field: "precision" | "recompute" | "offload", value: LayerPrecision | RecomputeMode | boolean | null) {
    setLocks((current) => {
      const next = { ...current, [selected.id]: { ...current[selected.id] } };
      if (value === null) delete next[selected.id]![field];
      else next[selected.id]![field] = value as never;
      if (Object.keys(next[selected.id]!).length === 0) delete next[selected.id];
      return next;
    });
  }

  return (
    <div className="layer-planner">
      <header className="lab-q">
        <p className="section-kicker">Shared model graph</p>
        <h2>Plan the same model layer by layer.</h2>
        <p className="lab-lede">Upload a Hugging Face <code>config.json</code>. The graph is shared across training, serving, and neuromorphic tabs; this first slice searches training decisions and keeps every assumption visible.</p>
      </header>

      <div className="layer-tabs" role="tablist" aria-label="Shared model graph tabs">
        {(["train", "serve", "neuromorphic"] as const).map((value) => (
          <button key={value} type="button" role="tab" aria-selected={tab === value} className={tab === value ? "is-on" : ""} onClick={() => setTab(value)}>
            {value === "train" ? "Train on a cluster" : value === "serve" ? "Serve on GPUs" : "Deploy on neuromorphic chip"}
          </button>
        ))}
      </div>

      <div className="layer-input-grid">
        <section className="lab-card">
          <div className="lab-card-head"><h3>Model config</h3><label className="ui-button ui-button--outline ui-button--compact" htmlFor="layer-config-file">Choose JSON</label></div>
          <input id="layer-config-file" className="layer-file" type="file" accept="application/json,.json" onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            void file.text().then((text) => parseConfig(text, file.name));
          }} />
          <textarea aria-label="Hugging Face config JSON" className="layer-config" value={configText} onChange={(event) => setConfigText(event.target.value)} onBlur={() => parseConfig(configText)} spellCheck={false} />
          {error && <p className="layer-error" role="alert">{error}</p>}
          <p className="lab-note">Required: layers, hidden size, attention heads, intermediate size, and vocabulary. Aliases such as <code>n_layer</code> and <code>n_embd</code> are accepted.</p>
        </section>

        <section className="lab-card layer-summary">
          <h3>Graph summary</h3>
          <div className="layer-summary-facts">
            <div><b>{formatParams(graph.parameterCount)}</b><span>resident parameters</span></div>
            <div><b>{graph.nodes.length}</b><span>decision nodes</span></div>
            <div><b>{graph.kvHeads}</b><span>KV heads</span></div>
          </div>
          <p>{graph.name} · {graph.layers} transformer blocks · {graph.hiddenSize} hidden · {graph.attentionHeads} attention heads</p>
          {graph.assumptions.map((assumption) => <p className="lab-note" key={assumption}>Assumption: {assumption}</p>)}
        </section>
      </div>

      <section className="lab-card" aria-labelledby="layer-map-title">
        <div className="lab-card-head"><h3 id="layer-map-title">Layer map</h3><span className="lab-muted">Select a node to inspect or lock it</span></div>
        <div className="layer-map" role="list" aria-label="Model graph nodes">
          {graph.nodes.map((node, index) => {
            const locked = locks[node.id] !== undefined;
            const stage = best?.stages[index] ?? 0;
            return <button key={node.id} type="button" role="listitem" aria-pressed={selected.id === node.id} className={`layer-node kind-${node.kind}${selected.id === node.id ? " is-selected" : ""}${locked ? " is-locked" : ""}`} onClick={() => setSelectedId(node.id)}>
              <span className="layer-node-stage">S{stage + 1}</span>
              <b>{node.kind === "attention" ? "ATTN" : node.kind === "mlp" ? "MLP" : node.kind.toUpperCase()}</b>
              <small>{node.layerIndex === null ? "" : `L${node.layerIndex + 1}`}</small>
              <i aria-hidden="true">{locked ? "●" : "○"}</i>
            </button>;
          })}
        </div>
        <div className="layer-node-detail">
          <div><p className="section-kicker">Selected node</p><h4>{nodeLabel(selected.id)}</h4><p className="lab-note">{formatParams(selected.parameterCount)} parameters · {formatBytes(selected.activationBytes)} activation · {formatBytes(selected.kvCacheBytes)} KV cache/token window</p></div>
          <div className="layer-lock-fields">
            <label>Precision<select aria-label={`Precision for ${selected.id}`} value={selectedLock.precision ?? ""} onChange={(event) => setLock("precision", event.target.value === "" ? null : event.target.value as LayerPrecision)}><option value="">Nomo decides</option><option value="bf16">BF16</option><option value="fp8">FP8</option></select></label>
            <label>Recompute<select aria-label={`Recompute for ${selected.id}`} value={selectedLock.recompute ?? ""} onChange={(event) => setLock("recompute", event.target.value === "" ? null : event.target.value as RecomputeMode)}><option value="">Nomo decides</option><option value="none">None</option><option value="selective">Selective</option><option value="full">Full</option></select></label>
            <label className="layer-checkbox"><input type="checkbox" checked={selectedLock.offload ?? false} onChange={(event) => setLock("offload", event.target.checked ? true : null)} /> Allow CPU activation offload</label>
          </div>
        </div>
      </section>

      {tab !== "train" ? (
        <section className="lab-card layer-coming-soon">
          <p className="section-kicker">{tab === "serve" ? "Serve on GPUs" : "Neuromorphic deployment"}</p>
          <h3>The shared graph is ready for this tab.</h3>
          <p>{tab === "serve" ? "Per-layer weight and KV-cache precision will consume these same nodes next. No serving recommendation is fabricated in this slice." : "The existing neuromorphic compiler remains the source of truth for NIR and chip exports. This tab will consume the shared graph without changing those exports."}</p>
          <span className="lab-rel warn">Not yet a recommendation</span>
        </section>
      ) : (
        <div className="layer-results-grid">
          <section className="lab-card layer-answer">
            <div className="lab-card-head"><h3>Layer-aware training plan</h3><span className={`lab-rel ${result.exhaustive ? "ok" : "warn"}`}>{result.exhaustive ? "Exhaustive" : "Bounded search"}</span></div>
            {best && metrics ? <>
              <p className="layer-plan-summary">{planSummary(graph, best)}</p>
              <div className="lab-metrics"><div className="lab-metric"><b>{metrics.objectives.stepTimeS.toFixed(3)} s</b><span>estimated step time</span><small>global baseline {baseline.objectives.stepTimeS.toFixed(3)} s · {percentChange(metrics.objectives.stepTimeS, baseline.objectives.stepTimeS)}</small></div><div className="lab-metric"><b>${metrics.objectives.costUsdPerStep.toFixed(4)}</b><span>estimated cost / step</span><small>global baseline ${baseline.objectives.costUsdPerStep.toFixed(4)} · {percentChange(metrics.objectives.costUsdPerStep, baseline.objectives.costUsdPerStep)}</small></div><div className="lab-metric"><b>{(metrics.objectives.memoryHeadroom * 100).toFixed(1)}%</b><span>memory headroom</span></div></div>
              <p className="lab-note">Compared with the best global-only baseline: the percentages are model estimates, not measurements. Search evaluated {result.evaluated.toLocaleString("en-US")} repaired candidates.</p>
              {mode === "rigor" && <div className="layer-rigor"><p><b>Validity:</b> stages are contiguous and non-empty; endpoint nodes stay BF16 unless explicitly locked; every stage must fit the 80 GB × 90% assumed capacity.</p><p><b>Customer inputs still needed:</b> FP8 quality evaluation, CPU offload bandwidth, framework overhead, and cluster-specific utilization.</p></div>}
            </> : <p role="alert">No feasible layer-aware plan under the current assumptions.</p>}
          </section>
          <section className="lab-card"><h3>Stage assignment</h3>{best && <div className="layer-stage-list">{Array.from(new Set(best.stages)).map((stage) => { const nodes = graph.nodes.filter((_node, index) => best.stages[index] === stage); const memory = metrics?.memoryByStage[stage] ?? 0; return <div key={stage} className="layer-stage"><span>S{stage + 1}</span><div><b>{nodes[0]?.id} → {nodes[nodes.length - 1]?.id}</b><small>{nodes.length} nodes · {formatBytes(memory)} assumed memory</small></div></div>; })}</div>}<p className="lab-note">{result.assumptions.join(" · ")}</p></section>
        </div>
      )}
    </div>
  );
}

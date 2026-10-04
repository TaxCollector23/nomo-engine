import { useMemo, useState } from "react";

import {
  BUILTIN_MODEL_CONFIGS,
  buildLayerGraph,
  searchLayerTraining,
  type LayerGraph,
  type LayerPrecision,
  type LayerTrainingPlan,
  type RecomputeMode,
} from "../planner/layers";

type LayerLocks = Record<string, Partial<{ stage: number; precision: LayerPrecision; recompute: RecomputeMode; offload: boolean }>>;

const TINY_DEMO_CONFIG = {
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

const MODEL_OPTIONS = [
  ["llama3_8b", "Llama 3 8B"],
  ["llama3_70b", "Llama 3 70B"],
  ["mixtral_8x7b", "Mixtral 8x7B"],
  ["tiny_demo", "Tiny demo (example only)"],
] as const;

function formatBytes(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} MB`;
  return `${(value / 1e3).toFixed(0)} KB`;
}

function formatSeconds(value: number): string {
  if (value >= 86400) return `${(value / 86400).toFixed(1)} d`;
  if (value >= 3600) return `${(value / 3600).toFixed(1)} h`;
  if (value >= 60) return `${(value / 60).toFixed(1)} min`;
  if (value >= 1) return `${value.toFixed(2)} s`;
  if (value >= 0.001) return `${(value * 1000).toFixed(1)} ms`;
  return `${(value * 1e6).toFixed(1)} µs`;
}

function formatUsd(value: number): string {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(2)}k`;
  if (value >= 1) return `$${value.toFixed(2)}`;
  if (value >= 0.01) return `$${value.toFixed(3)}`;
  return "<$0.01";
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
  const [modelKey, setModelKey] = useState<"llama3_8b" | "llama3_70b" | "mixtral_8x7b" | "tiny_demo">("llama3_8b");
  const defaultConfig = BUILTIN_MODEL_CONFIGS.llama3_8b;
  const [configText, setConfigText] = useState(() => JSON.stringify(defaultConfig, null, 2));
  const [graph, setGraph] = useState<LayerGraph>(() => buildLayerGraph(defaultConfig, { seqLen: 2048, batchSize: 1, source: "published model config" }));
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"train" | "serve" | "neuromorphic">("train");
  const [selectedId, setSelectedId] = useState("block.5.mlp");
  const [locks, setLocks] = useState<LayerLocks>({});

  const result = useMemo(() => searchLayerTraining({
    graph,
    hardware: { devices: 8, memoryBytes: 80e9, usableMemory: 0.9, costPerDeviceHour: 3.5 },
    pipelineStages: [1, 2, 4, 8, 16, 32],
    microBatches: 8,
    maxCandidates: 20000,
    totalSteps: 1000,
    seed: 20261003,
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

  function selectModel(nextKey: typeof modelKey) {
    setModelKey(nextKey);
    const nextConfig = nextKey === "tiny_demo" ? TINY_DEMO_CONFIG : BUILTIN_MODEL_CONFIGS[nextKey];
    setConfigText(JSON.stringify(nextConfig, null, 2));
    setGraph(buildLayerGraph(nextConfig, { seqLen: graph.seqLen, batchSize: graph.batchSize, source: nextKey === "tiny_demo" ? "tiny example" : "published model config" }));
    setSelectedId("block.0.mlp");
    setLocks({});
    setError(null);
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
          <label className="layer-model-select">Built-in model<select aria-label="Built-in model config" value={modelKey} onChange={(event) => selectModel(event.target.value as typeof modelKey)}>{MODEL_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <input id="layer-config-file" className="layer-file" type="file" accept="application/json,.json" onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            void file.text().then((text) => parseConfig(text, file.name));
          }} />
          <textarea aria-label="Hugging Face config JSON" className="layer-config" value={configText} onChange={(event) => setConfigText(event.target.value)} onBlur={() => parseConfig(configText)} spellCheck={false} />
          {error && <p className="layer-error" role="alert">{error}</p>}
          <p className="lab-note">Default is the published Llama 3 8B config. Tiny demo is an example only. Uploaded fields are parsed from the config; aliases such as <code>n_layer</code> and <code>n_embd</code> are accepted.</p>
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
               <div className="lab-metrics"><div className="lab-metric"><b>{formatSeconds(metrics.objectives.stepTimeS)}</b><span>estimated step time</span><small>best global {formatSeconds(baseline.objectives.stepTimeS)} · {percentChange(metrics.objectives.stepTimeS, baseline.objectives.stepTimeS)}</small></div><div className="lab-metric"><b>{formatUsd(metrics.objectives.costUsdPerStep)}</b><span>estimated cost / step</span><small>best global {formatUsd(baseline.objectives.costUsdPerStep)} · {percentChange(metrics.objectives.costUsdPerStep, baseline.objectives.costUsdPerStep)}</small></div><div className="lab-metric"><b>{formatSeconds(metrics.objectives.wholeRunTimeS)}</b><span>whole run · 1,000 steps</span><small>estimated total time</small></div><div className="lab-metric"><b>{formatUsd(metrics.objectives.wholeRunCostUsd)}</b><span>whole run cost</span><small>estimated total cost</small></div><div className="lab-metric"><b>{(metrics.objectives.memoryHeadroom * 100).toFixed(1)}%</b><span>memory headroom</span></div></div>
               <div className="lab-metrics"><div className="lab-metric"><b>{result.precisionGainPct.toFixed(1)}%</b><span>gain from global precision choice</span><small>BF16 global → best global</small></div><div className="lab-metric"><b>{result.perLayerGainPct.toFixed(1)}%</b><span>gain from per-layer decisions</span><small>best global → per-layer plan</small></div><div className="lab-metric"><b>{formatSeconds(metrics.objectives.pipelineBubbleS)}</b><span>pipeline bubble / step</span><small>charged estimate</small></div><div className="lab-metric"><b>{formatSeconds(metrics.objectives.communicationS)}</b><span>inter-stage transfer / step</span><small>charged estimate</small></div><div className="lab-metric"><b>{formatSeconds(metrics.objectives.offloadTransferS)}</b><span>CPU offload transfer / step</span><small>PCIe/host bandwidth assumption</small></div></div>
               <p className="lab-note">The fair comparison is against the best global-only plan allowed the same precision, recompute, offload, and stage choices. All values are model estimates; hardware measurements and bandwidth remain customer inputs. Search evaluated {result.evaluated.toLocaleString("en-US")} repaired candidates with fixed seed {result.seed}.</p>
              {mode === "rigor" && <div className="layer-rigor"><p><b>Validity:</b> stages are contiguous and non-empty; endpoint nodes stay BF16 unless explicitly locked; every stage must fit the 80 GB × 90% assumed capacity.</p><p><b>Customer inputs still needed:</b> FP8 quality evaluation, CPU offload bandwidth, framework overhead, and cluster-specific utilization.</p></div>}
            </> : <p role="alert">No feasible layer-aware plan under the current assumptions.</p>}
          </section>
          <section className="lab-card"><h3>Stage assignment</h3>{best && <div className="layer-stage-list">{Array.from(new Set(best.stages)).map((stage) => { const nodes = graph.nodes.filter((_node, index) => best.stages[index] === stage); const memory = metrics?.memoryByStage[stage] ?? 0; return <div key={stage} className="layer-stage"><span>S{stage + 1}</span><div><b>{nodes[0]?.id} → {nodes[nodes.length - 1]?.id}</b><small>{nodes.length} nodes · {formatBytes(memory)} assumed memory</small></div></div>; })}</div>}<p className="lab-note">{result.assumptions.join(" · ")}</p></section>
        </div>
      )}
    </div>
  );
}

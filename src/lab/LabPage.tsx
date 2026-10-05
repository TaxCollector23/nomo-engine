import { useCallback, useDeferredValue, useEffect, useMemo, useState, type ReactNode } from "react";

import { CALIBRATED, CALIBRATION_RESULTS } from "../planner/calibration";
import { counterfactuals, sensitivity, summary, type Counterfactual } from "../planner/explain";
import { buildExport } from "../planner/exporters";
import { CLUSTERS, MODELS } from "../planner/hardware";
import { DEFAULT_ATTENTION_PENALTY, DEFAULT_QUALITY_LOSS, SCALING_LAWS, feasible, type Choice } from "../planner/packs";
import { DEFAULT_SETTINGS, build, innerPack, withLocks, type DomainId, type Settings } from "../planner/registry";
import { Planner, recommend, type Evaluated } from "../planner/search";
import { uncertaintyForPlan, UNCERTAINTY_VALIDATION } from "../planner/uncertainty";
import Anatomy from "./Anatomy";
import Auditor from "./Auditor";
import { Chips, Field, Info, LinSlider, LogSlider, Segmented, Select, Toggle, fmtNum, fmtTokens } from "./controls";
import Evidence from "./Evidence";
import ExportDialog from "./ExportDialog";
import LayerPlanner from "./LayerPlanner";
import Methods, { Equations } from "./Methods";
import ProductStudio from "./ProductStudio";
import CostTracker from "./CostTracker";
import TradeoffChart from "./TradeoffChart";
import "./lab.css";

type ModuleId = DomainId | "layers" | "auditor" | "products" | "costs" | "evidence" | "methods" | "neuromorphic";
type Mode = "guided" | "explore" | "rigor";
interface Prefs { animate: boolean; anatomy: boolean; whatif: boolean; table: boolean; equations: boolean; provenance: boolean }

const MODE_PREFS: Record<Mode, Prefs> = {
  guided: { animate: true, anatomy: true, whatif: false, table: false, equations: false, provenance: false },
  explore: { animate: true, anatomy: true, whatif: true, table: true, equations: false, provenance: false },
  rigor: { animate: false, anatomy: true, whatif: true, table: true, equations: true, provenance: true },
};

const MODULES: { id: ModuleId; n: string; title: string; blurb: string }[] = [
  { id: "layers", n: "01", title: "Shared model graph", blurb: "Train, serve and deploy the same layers" },
  { id: "llm_training", n: "02", title: "Train a model", blurb: "Split a training run across GPUs" },
  { id: "llm_inference", n: "03", title: "Serve a model", blurb: "Cheapest tokens at your speed limit" },
  { id: "arch_codesign", n: "04", title: "Design a model", blurb: "What to build for your budget" },
  { id: "neuromorphic", n: "05", title: "Neuromorphic chips", blurb: "Spiking and physics-based layers" },
  { id: "auditor", n: "06", title: "Audit a run", blurb: "Parse configs and observed logs" },
  { id: "products", n: "07", title: "Infrastructure products", blurb: "RL, reliability, fleets, TCO" },
  { id: "costs", n: "08", title: "Cost tracker", blurb: "Cited compute and token prices" },
  { id: "evidence", n: "09", title: "Evidence", blurb: "22 published runs, predicted" },
  { id: "methods", n: "10", title: "Methods", blurb: "Equations, verification, references" },
];

const QUESTIONS: Record<DomainId, string> = {
  llm_training: "How should this model be trained on these GPUs?",
  llm_inference: "How should this model be served?",
  arch_codesign: "Which model should we build?",
};

const STORE = "nomo-lab-v1";
const load = (): { mode: Mode; prefs: Prefs } => {
  try {
    const s = JSON.parse(localStorage.getItem(STORE) ?? "null");
    if (s && s.mode in MODE_PREFS) return { mode: s.mode, prefs: { ...MODE_PREFS[s.mode as Mode], ...s.prefs } };
  } catch { /* first visit or storage blocked */ }
  return { mode: "guided", prefs: MODE_PREFS.guided };
};

const modelOptions = Object.entries(MODELS).map(([k, m]) => ({ value: k, label: m.name }));
const hwOptions = Object.values(CLUSTERS).map((c) => ({ value: c.key, label: `${c.device.name}${CALIBRATED[c.key] ? " (calibrated)" : ""}` }));
const GPU_CHOICES = [64, 128, 256, 512, 1024, 2048, 4096];
const moduleFromUrl = (): ModuleId => {
  const h = window.location.hash.replace("#", "");
  return (MODULES.find((m) => m.id === h)?.id ?? "llm_training") as ModuleId;
};

function Card({ title, children, className, aside }: { title?: ReactNode; children: ReactNode; className?: string; aside?: ReactNode }) {
  return (
    <section className={`lab-card ${className ?? ""}`}>
      {(title || aside) && <div className="lab-card-head">{title && <h3>{title}</h3>}{aside}</div>}
      {children}
    </section>
  );
}

export default function LabPage({ engineUrl }: { engineUrl: string }) {
  const init = useMemo(load, []);
  const [module, setModule] = useState<ModuleId>(moduleFromUrl);
  const [mode, setModeRaw] = useState<Mode>(init.mode);
  const [prefs, setPrefs] = useState<Prefs>(init.prefs);
  const [custOpen, setCustOpen] = useState(false);
  const [settings, setSettings] = useState<Record<DomainId, Settings>>({ ...DEFAULT_SETTINGS });
  const [locks, setLocks] = useState<Record<DomainId, Record<string, Choice>>>({ llm_training: {}, llm_inference: {}, arch_codesign: {} });
  const [weights, setWeights] = useState<Record<DomainId, number[] | null>>({ llm_training: null, llm_inference: null, arch_codesign: null });
  const [selKey, setSelKey] = useState<string | null>(null);
  const [ghost, setGhost] = useState<Evaluated | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [axes, setAxes] = useState<Record<DomainId, [string, string]>>({
    llm_training: ["days", "cost_usd"], llm_inference: ["usd_per_mtok", "ms_per_token"], arch_codesign: ["total_cost_usd", "loss"] });

  useEffect(() => { localStorage.setItem(STORE, JSON.stringify({ mode, prefs })); }, [mode, prefs]);
  useEffect(() => {
    const on = () => setModule(moduleFromUrl());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = (m: ModuleId) => { window.history.replaceState(null, "", `#${m}`); setModule(m); setSelKey(null); setGhost(null); };
  const setMode = (m: Mode) => { setModeRaw(m); setPrefs(MODE_PREFS[m]); };

  const domain = (["llm_training", "llm_inference", "arch_codesign"] as string[]).includes(module) ? (module as DomainId) : null;
  const key = domain ? JSON.stringify([domain, settings[domain], locks[domain]]) : "";
  const deferredKey = useDeferredValue(key);
  const stale = key !== deferredKey;

  const computed = useMemo(() => {
    if (!deferredKey) return null;
    const [d, s, l] = JSON.parse(deferredKey) as [DomainId, Settings, Record<string, Choice>];
    try {
      const pack = withLocks(build(d, s), l);
      const pl = new Planner(pack);
      const t0 = performance.now();
      const res = pl.run();
      return { d, pack, pl, res, ms: performance.now() - t0, error: null as string | null };
    } catch (e) {
      return { d, pack: null, pl: null, res: null, ms: 0, error: e instanceof Error ? e.message : String(e) };
    }
  }, [deferredKey]);

  const w = domain ? weights[domain] : null;
  const rec = useMemo(() => (computed?.res && computed.d === domain ? recommend(computed.res.front, w) : null), [computed, w, domain]);
  const selected = useMemo(() => {
    if (!computed?.pl || !computed.res || computed.d !== domain) return null;
    if (selKey) { const e = computed.pl.cache.get(selKey); if (e && feasible(e.metrics)) return e; }
    return rec;
  }, [computed, selKey, rec, domain]);
  useEffect(() => { setGhost(null); }, [computed]);

  const set = useCallback((k: string, v: unknown) => {
    if (!domain) return;
    setSettings((s) => ({ ...s, [domain]: { ...s[domain], [k]: v } }));
    setSelKey(null);
  }, [domain]);
  const S = domain ? settings[domain] : {};

  // the headline already states the plan, so the story starts with the reasoning
  const story = useMemo(() => (computed?.pl && selected ? summary(computed.pl, selected.plan).replace(/^Plan: .*?\.(?= )/, "").trim() : ""), [computed, selected]);
  const sens = useMemo(() => (computed?.pl && selected ? sensitivity(computed.pl, selected.plan) : {}), [computed, selected]);
  const whatifs = useMemo(() => {
    if (!computed?.pl || !selected) return [] as Counterfactual[];
    const first = computed.pack!.objectives()[0]!.name, second = computed.pack!.objectives()[1]!.name;
    const cfs = counterfactuals(computed.pl, selected.plan)
      .filter((c) => c.feasible && Object.values(c.deltas_pct).some((d) => Math.abs(d) >= 1));   // only changes that matter
    const a = [...cfs].sort((x, y) => x.deltas_pct[first]! - y.deltas_pct[first]!).slice(0, 3);
    const b = [...cfs].sort((x, y) => x.deltas_pct[second]! - y.deltas_pct[second]!).slice(0, 3);
    const seen = new Set<string>();
    return [...a, ...b].filter((c) => { const k = `${c.variable}=${c.value}`; if (seen.has(k)) return false; seen.add(k); return true; });
  }, [computed, selected]);

  const exportFiles = useMemo(() => (exportOpen && computed?.pl && computed.res && selected && domain
    ? buildExport(domain, settings[domain], locks[domain], computed.pl, computed.res, selected) : []), [exportOpen, computed, selected, domain, settings, locks]);

  const planKey = (e: Evaluated) => computed!.pl!.key(e.plan);
  const label = (n: string) => computed?.pack?.variables().find((v) => v.name === n)?.label ?? n;

  // ------------------------------------------------------------------ inputs per domain
  const inputs = () => {
    if (domain === "llm_training") {
      return (
        <>
          <Field label="Model"><Select label="Model" value={String(S.model)} onChange={(v) => set("model", v)} options={modelOptions} /></Field>
          <Field label="GPUs"><Select label="GPUs" value={String(S.cluster)} onChange={(v) => set("cluster", v)} options={hwOptions} /></Field>
          <Field label={<Info text="How much text the model learns from. Chinchilla-style models use roughly 20 tokens per parameter.">Training tokens</Info>}>
            <LogSlider label="Training tokens" value={Number(S.total_tokens)} onChange={(v) => set("total_tokens", v)} min={1e10} max={2e13} format={fmtTokens} />
          </Field>
          <Field label="GPU counts to consider">
            <Chips label="GPU counts" options={GPU_CHOICES} value={(S.device_counts as number[])} onChange={(v) => set("device_counts", v)} />
          </Field>
          {mode !== "guided" && (
            <>
              <Field label="Sequence length"><Select label="Sequence length" value={String(S.seq_len)} onChange={(v) => set("seq_len", Number(v))}
                options={[1024, 2048, 4096, 8192].map((n) => ({ value: String(n), label: `${n} tokens` }))} /></Field>
              <Field label={<Info text="Tokens processed per optimizer step. Larger batches allow more parallelism but can hurt learning if too large.">Batch size</Info>}>
                <Select label="Batch size" value={String(S.global_batch_tokens)} onChange={(v) => set("global_batch_tokens", Number(v))}
                  options={[1, 2, 4, 8, 16].map((n) => ({ value: String(n * 1048576), label: `${n}M tokens` }))} /></Field>
            </>
          )}
          {mode === "rigor" && (
            <Field label="Usable memory per GPU" hint="Headroom for fragmentation and framework buffers">
              <LinSlider label="Usable memory" value={Number(S.usable_memory)} onChange={(v) => set("usable_memory", v)} min={0.7} max={1} step={0.01} format={(v) => `${Math.round(v * 100)}%`} />
            </Field>
          )}
        </>
      );
    }
    if (domain === "llm_inference") {
      const q = (S.quality_loss as typeof DEFAULT_QUALITY_LOSS) ?? DEFAULT_QUALITY_LOSS;
      return (
        <>
          <Field label="Model"><Select label="Model" value={String(S.model)} onChange={(v) => set("model", v)} options={modelOptions.slice(0, 4)} /></Field>
          <Field label="GPUs"><Select label="GPUs" value={String(S.cluster)} onChange={(v) => set("cluster", v)} options={hwOptions} /></Field>
          <Field label={<Info text="The longest acceptable gap between generated tokens. People read at roughly 50-100 ms per token.">Speed limit</Info>}>
            <LogSlider label="Speed limit" allowOff value={S.max_ms_per_token === null ? null : Number(S.max_ms_per_token)}
              onChange={(v) => set("max_ms_per_token", v)} min={5} max={500} format={(v) => `${v.toFixed(0)} ms per token`} />
          </Field>
          <Field label={<Info text="How much benchmark accuracy you accept losing to lower precision. Values are assumptions until you enter your own measurements (Rigor mode).">Quality budget</Info>}>
            <LinSlider label="Quality budget" value={Number(S.max_quality_loss)} onChange={(v) => set("max_quality_loss", v)} min={0} max={3} step={0.1} format={(v) => `${v.toFixed(1)} points`} />
          </Field>
          {mode !== "guided" && (
            <>
              <Field label="Prompt length"><Select label="Prompt length" value={String(S.prompt_tokens)} onChange={(v) => set("prompt_tokens", Number(v))}
                options={[256, 1024, 2048, 8192, 32768].map((n) => ({ value: String(n), label: `${n.toLocaleString("en-US")} tokens` }))} /></Field>
              <Field label="Answer length"><Select label="Answer length" value={String(S.output_tokens)} onChange={(v) => set("output_tokens", Number(v))}
                options={[64, 256, 1024, 4096].map((n) => ({ value: String(n), label: `${n.toLocaleString("en-US")} tokens` }))} /></Field>
            </>
          )}
          {mode === "rigor" && (
            <Field label="Assumed quality loss (points)" hint="Replace with your own evaluation results">
              <table className="lab-mini-table">
                <thead><tr><th /><th>bf16</th><th>fp8</th><th>int4</th></tr></thead>
                <tbody>
                  {(["weights", "kv"] as const).map((row) => (
                    <tr key={row}><th>{row === "kv" ? "KV cache" : "weights"}</th>
                      {(["bf16", "fp8", "int4"] as const).map((c) => (
                        <td key={c}><input type="number" step={0.1} min={0} aria-label={`${row} ${c} quality loss`} value={q[row]![c]}
                          onChange={(e) => set("quality_loss", { ...q, [row]: { ...q[row], [c]: Number(e.target.value) } })} /></td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Field>
          )}
        </>
      );
    }
    if (domain === "arch_codesign") {
      const pen = (S.attention_loss_penalty as Record<string, number>) ?? DEFAULT_ATTENTION_PENALTY;
      return (
        <>
          <Field label="Total budget (training + serving)">
            <LogSlider label="Budget" allowOff value={S.budget_usd === null ? null : Number(S.budget_usd)} onChange={(v) => set("budget_usd", v)}
              min={1e5} max={1e8} format={(v) => fmtNum(v, "USD")} />
          </Field>
          <Field label={<Info text="How many tokens the model will generate over its whole life. More usage makes serving cost matter more than training cost.">Lifetime usage</Info>}>
            <LogSlider label="Lifetime usage" allowOff value={Number(S.tokens_served) || null} onChange={(v) => set("tokens_served", v ?? 0)}
              min={1e9} max={1e15} format={(v) => `${fmtTokens(v)} tokens served`} />
          </Field>
          <Field label="Serving GPUs"><Select label="Serving GPUs" value={String(S.serve_cluster)} onChange={(v) => set("serve_cluster", v)} options={hwOptions} /></Field>
          {mode !== "guided" && (
            <>
              <Field label="Training GPUs"><Select label="Training GPUs" value={String(S.train_cluster)} onChange={(v) => set("train_cluster", v)} options={hwOptions} /></Field>
              <Field label="Speed limit when serving">
                <LogSlider label="Serving speed limit" allowOff value={S.max_ms_per_token === null ? null : Number(S.max_ms_per_token)}
                  onChange={(v) => set("max_ms_per_token", v)} min={5} max={500} format={(v) => `${v.toFixed(0)} ms per token`} />
              </Field>
              <Field label={<Info text="The published formula predicting quality from model size and data. The 2024 replication corrected errors in the original fit.">Scaling law</Info>}>
                <Select label="Scaling law" value={String(S.scaling_law)} onChange={(v) => set("scaling_law", v)}
                  options={Object.entries(SCALING_LAWS).map(([k, l]) => ({ value: k, label: l.label }))} />
              </Field>
            </>
          )}
          {mode === "rigor" && (
            <>
              <Field label="Assumed training utilisation" hint="Fraction of peak compute actually achieved">
                <LinSlider label="Training utilisation" value={Number(S.train_mfu)} onChange={(v) => set("train_mfu", v)} min={0.2} max={0.7} step={0.01} format={(v) => `${Math.round(v * 100)}%`} />
              </Field>
              <Field label="Assumed quality cost of attention types (loss)" hint="Replace with your own measurements">
                <table className="lab-mini-table"><tbody><tr>
                  {(["MHA", "GQA-8", "MQA"] as const).map((k) => (
                    <td key={k}><span>{k}</span><input type="number" step={0.001} min={0} aria-label={`${k} penalty`} value={pen[k]}
                      onChange={(e) => set("attention_loss_penalty", { ...pen, [k]: Number(e.target.value) })} /></td>
                  ))}
                </tr></tbody></table>
              </Field>
            </>
          )}
        </>
      );
    }
    return null;
  };

  // ------------------------------------------------------------------ render
  // a result computed for a different module (deferred update in flight) is never rendered
  const fresh = computed && computed.d === domain ? computed : null;
  const pack = fresh?.pack ?? null;
  const res = fresh?.res ?? null;
  const objs = pack?.objectives() ?? [];
  const calib = pack?.calibration ?? null;
  const selectedUncertainty = selected && res ? uncertaintyForPlan(res.uncertainty, selected.plan) : null;
  const reliability = domain === "llm_training"
    ? (calib ? (res?.uncertainty ? { cls: "warn", text: `Hardware calibrated on ${calib.observations} runs; 90% intervals covered ${UNCERTAINTY_VALIDATION.actual === null ? "n/a" : `${Math.round(UNCERTAINTY_VALIDATION.actual * UNCERTAINTY_VALIDATION.n!)} / ${UNCERTAINTY_VALIDATION.n}`} held-out runs` }
      : { cls: "ok", text: `Hardware calibrated on ${calib.observations} published runs (held-out error ${CALIBRATION_RESULTS.heldOutPtdMape}%)` })
      : { cls: "warn", text: "Uncalibrated hardware numbers: compare plans, treat absolute values as indicative" })
    : domain === "llm_inference" ? { cls: "warn", text: "Uncalibrated upper bound; quality effects are assumptions" }
      : { cls: "warn", text: "Published scaling law; prices and attention quality effects are assumptions" };

  return (
    <div className="lab">
      <div className="lab-top">
        <div>
          <p className="page-kicker">Nomo Lab</p>
          <h1 className="lab-title">Plan how AI runs on hardware.</h1>
          <p className="lab-lede">
            Pick a question. Nomo checks every possible plan in your browser, shows the best trade-offs, explains its choice,
            and shows the evidence behind its numbers.
          </p>
        </div>
        <div className="lab-top-stats" aria-label="Lab facts">
          <div className="lab-stat"><b>14,289</b><span>checks against the reference engine</span></div>
          <div className="lab-stat"><b>22</b><span>published runs used for calibration</span></div>
          <div className="lab-stat"><b>0</b><span>servers: everything runs in this page</span></div>
        </div>
      </div>

      <div className="lab-layout">
        <nav className="lab-rail" aria-label="Lab modules">
          {MODULES.map((m) => (
            <button key={m.id} type="button" className={module === m.id ? "is-on" : ""} aria-current={module === m.id ? "page" : undefined} onClick={() => go(m.id)}>
              <span className="lab-rail-n">{m.n}</span>
              <span><b>{m.title}</b><small>{m.blurb}</small></span>
            </button>
          ))}
          <div className="lab-rail-foot">
            <p className="lab-muted">View</p>
            <Segmented label="Lab mode" value={mode} onChange={setMode} options={[
              { value: "guided", label: "Guided", hint: "Plain answers, key controls only" },
              { value: "explore", label: "Explore", hint: "All controls, charts, what-ifs" },
              { value: "rigor", label: "Rigor", hint: "Equations, provenance, assumptions" },
            ]} />
            <p className="lab-muted lab-mode-hint">{mode === "guided" ? "Plain answers, key controls only." : mode === "explore" ? "Every control, what-ifs and the full plan table." : "Adds equations, where every number comes from, and editable assumptions."}</p>
            <button type="button" className="lab-link" aria-expanded={custOpen} onClick={() => setCustOpen((o) => !o)}>
              {custOpen ? "Hide panel settings" : "Customize panels"}
            </button>
            {custOpen && (
              <div className="lab-cust">
                <Toggle label="Animate the search" checked={prefs.animate} onChange={(v) => setPrefs({ ...prefs, animate: v })} />
                <Toggle label="Plan anatomy diagrams" checked={prefs.anatomy} onChange={(v) => setPrefs({ ...prefs, anatomy: v })} />
                <Toggle label="What-if suggestions" checked={prefs.whatif} onChange={(v) => setPrefs({ ...prefs, whatif: v })} />
                <Toggle label="Table of best trade-offs" checked={prefs.table} onChange={(v) => setPrefs({ ...prefs, table: v })} />
                <Toggle label="Equations" checked={prefs.equations} onChange={(v) => setPrefs({ ...prefs, equations: v })} />
                <Toggle label="Provenance of numbers" checked={prefs.provenance} onChange={(v) => setPrefs({ ...prefs, provenance: v })} />
              </div>
            )}
          </div>
        </nav>

        <main className="lab-main" id="lab-main">
          {module === "layers" && <LayerPlanner mode={mode} />}
          {module === "auditor" && <Auditor />}
          {module === "products" && <ProductStudio />}
          {module === "costs" && <CostTracker />}
          {module === "evidence" && <Evidence />}
          {module === "methods" && <Methods />}
          {module === "neuromorphic" && (
            <div className="lab-neuro">
              <p className="section-kicker">Neuromorphic chips</p>
              <h2>Spiking, continuous and physics-based layers</h2>
              <p className="lab-lede">
                The neuromorphic planner decides, layer by layer, whether a network runs as standard computation, as brain-style
                spiking layers, or as exact physics formulas, then compiles the result to NIR, C, ONNX and PyTorch. It runs on the
                Nomo engine because it needs the full compiler.
              </p>
              <a className="ui-button ui-button--primary ui-button--default" href={engineUrl} target="_blank" rel="noreferrer">Open the neuromorphic engine ↗</a>
              <p className="lab-muted">The same planning engine powers both: exhaustive search, best trade-offs, computed explanations.</p>
            </div>
          )}

          {domain && (
            <div className={`lab-domain${stale ? " is-stale" : ""}`}>
              <header className="lab-q">
                <p className="section-kicker">{MODULES.find((m) => m.id === domain)!.title}</p>
                <h2>{QUESTIONS[domain]}</h2>
              </header>

              <div className={`lab-grid mode-${mode}`}>
                <Card title="Your situation" className="lab-inputs">
                  <div className="lab-fields">{inputs()}</div>
                  {mode !== "guided" && pack && (
                    <details className="lab-locks">
                      <summary>Lock decisions {Object.keys(locks[domain]).length ? `(${Object.keys(locks[domain]).length} locked)` : ""}</summary>
                      <div className="lab-fields">
                        {innerPack(pack).variables().map((v) => (
                          <Field key={v.name} label={v.help ? <Info text={v.help}>{v.label}</Info> : v.label}>
                            <Select label={`Lock ${v.label}`} value={locks[domain][v.name] === undefined ? "" : String(locks[domain][v.name])}
                              onChange={(val) => {
                                setLocks((L) => {
                                  const n = { ...L[domain] };
                                  if (val === "") delete n[v.name]; else n[v.name] = v.choices.find((c) => String(c) === val)!;
                                  return { ...L, [domain]: n };
                                });
                                setSelKey(null);
                              }}
                              options={[{ value: "", label: "Nomo decides" }, ...v.choices.map((c) => ({ value: String(c), label: String(c) }))]} />
                          </Field>
                        ))}
                      </div>
                    </details>
                  )}
                  {mode !== "guided" && objs.length > 0 && (
                    <details className="lab-locks">
                      <summary>What matters most</summary>
                      {objs.map((o, i) => (
                        <Field key={o.name} label={o.label}>
                          <LinSlider label={`Weight of ${o.label}`} value={(w ?? objs.map(() => 1))[i]!} min={0.25} max={4} step={0.25}
                            format={(v) => `×${v}`} onChange={(v) => setWeights((W) => {
                              const cur = [...(W[domain] ?? objs.map(() => 1))]; cur[i] = v; return { ...W, [domain]: cur };
                            })} />
                        </Field>
                      ))}
                    </details>
                  )}
                </Card>

                <div className="lab-results">
                  {fresh?.error && <Card><p role="alert">{fresh.error}</p></Card>}
                  {!fresh && <Card><p className="lab-muted">Checking every plan…</p></Card>}
                  {res && pack && (
                    <>
                      <Card className="lab-answer">
                        <div className="lab-answer-meta">
                          <span>{res.evaluated.toLocaleString("en-US")} plans checked in {Math.max(1, Math.round(computed!.ms))} ms</span>
                          <span className={`lab-rel ${reliability.cls}`}>{reliability.text}</span>
                        </div>
                        {selected ? (
                          <>
                            <p className="lab-answer-label">{selected === rec ? "Recommended plan" : "Selected plan"}</p>
                            <p className="lab-answer-plan">{pack.describe(selected.plan)}</p>
                            <div className="lab-metrics">
                              {objs.map((o) => (
                                <div key={o.name} className="lab-metric">
                                  <b>{fmtNum(selected.metrics.objectives[o.name]!, o.unit)}</b><span>{o.label}</span>
                                  {selectedUncertainty?.objectives[o.name] ? (
                                    <small>90%: {fmtNum(selectedUncertainty.objectives[o.name]!.low, o.unit)}–{fmtNum(selectedUncertainty.objectives[o.name]!.high, o.unit)}</small>
                                  ) : null}
                                </div>
                              ))}
                            </div>
                            {selectedUncertainty && (
                              <p className="lab-note lab-answer-confidence">
                                {objs.map((o) => `${Math.round((selectedUncertainty.probabilityBest[o.name] ?? 0) * 100)}% chance this is the ${o.label}`).join(" · ")} — probability from {res.uncertainty?.sampleCount.toLocaleString("en-US")} bootstrap samples.
                              </p>
                            )}
                            <p className="lab-story">{story}</p>
                            <div className="lab-answer-actions">
                              <button type="button" className="ui-button ui-button--primary ui-button--compact" onClick={() => setExportOpen(true)}>Export this plan</button>
                              {selected !== rec && <button type="button" className="lab-link" onClick={() => setSelKey(null)}>Back to the recommendation</button>}
                              {mode === "guided" && <button type="button" className="lab-link" onClick={() => setMode("explore")}>Explore the alternatives →</button>}
                            </div>
                          </>
                        ) : (
                          <div role="alert">
                            <p className="lab-answer-plan">No plan meets every limit.</p>
                            <p className="lab-story">Closest options: {res.closest.slice(0, 3).map((e) => pack.describe(e.plan)).join("; ")}. Relax a limit, add GPUs or unlock a decision.</p>
                          </div>
                        )}
                      </Card>

                      {res.front.length > 0 && (
                        <Card title="Every plan, and the best trade-offs" aside={mode !== "guided" && objs.length > 2 ? (
                          <span className="lab-axes">
                            <Select label="Horizontal axis" value={axes[domain][0]} onChange={(v) => setAxes((A) => ({ ...A, [domain]: [v, A[domain][1]] }))} options={objs.map((o) => ({ value: o.name, label: o.label }))} />
                            <span className="lab-muted">vs</span>
                            <Select label="Vertical axis" value={axes[domain][1]} onChange={(v) => setAxes((A) => ({ ...A, [domain]: [A[domain][0], v] }))} options={objs.map((o) => ({ value: o.name, label: o.label }))} />
                          </span>
                        ) : undefined}>
                          <TradeoffChart res={res} uncertainty={res.uncertainty} objectives={objs} xAxis={axes[domain][0]} yAxis={axes[domain][1]} selected={selected} recommended={rec}
                            ghost={ghost} animate={prefs.animate} compact={mode === "guided"} describe={(e) => pack.describe(e.plan)}
                            onSelect={(e) => setSelKey(planKey(e))} />
                          <p className="lab-note">Click a black point to inspect that plan. Lower and further left is better on both axes; no black point beats another on everything.</p>
                        </Card>
                      )}

                      {prefs.anatomy && selected && <Card title="Anatomy of this plan"><Anatomy pack={innerPack(pack)} e={selected} /></Card>}
                    </>
                  )}
                </div>

                {mode !== "guided" && res && pack && selected && (
                  <aside className="lab-side">
                    {prefs.whatif && whatifs.length > 0 && (
                      <Card title="What if you changed one thing?">
                        <ul className="lab-whatifs">
                          {whatifs.map((c, i) => {
                            const alt = computed!.pl!.evaluate({ ...selected.plan, [c.variable]: c.value });
                            return (
                              <li key={i}>
                                <button type="button" onMouseEnter={() => setGhost(alt)} onMouseLeave={() => setGhost(null)} onFocus={() => setGhost(alt)} onBlur={() => setGhost(null)}
                                  onClick={() => { setSelKey(planKey(alt)); setGhost(null); }}>
                                  <span className="lab-wi-change">{label(c.variable)} → <b>{String(c.value)}</b></span>
                                  <span className="lab-wi-deltas">
                                    {objs.map((o) => {
                                      const d = c.deltas_pct[o.name]!;
                                      const better = o.maximize ? d > 0.5 : d < -0.5, worse = o.maximize ? d < -0.5 : d > 0.5;
                                      return <span key={o.name} className={better ? "good" : worse ? "bad" : ""}>{o.label} {d >= 0 ? "+" : "−"}{Math.abs(d).toFixed(0)}%</span>;
                                    })}
                                  </span>
                                </button>
                              </li>
                            );
                          })}
                        </ul>
                        <p className="lab-note">Hover to preview on the chart; click to switch to that plan. Every number is a real re-evaluation.</p>
                      </Card>
                    )}
                    <Card title="Which decisions matter most">
                      <ul className="lab-sens">
                        {Object.entries(sens).filter(([, v]) => v > 0.5).slice(0, 6).map(([k, v]) => {
                          const max = Math.max(...Object.values(sens), 1);
                          return <li key={k}><span>{label(k)}</span><span className="lab-sens-bar"><i style={{ width: `${(100 * v) / max}%` }} /></span><span>up to {v.toFixed(0)}%</span></li>;
                        })}
                      </ul>
                    </Card>
                    {prefs.equations && <Card title="The equations behind it"><Equations domain={domain} /></Card>}
                    {mode === "rigor" && res.uncertainty && (
                      <Card title="What the 90% interval means">
                        <p>Intervals come from {res.uncertainty.sampleCount.toLocaleString("en-US")} stratified bootstrap refits of the 22 published A100 runs. Each draw keeps the fitted parameters together and adds a log-time residual draw.</p>
                        <p className="lab-note">Leave-one-out coverage was {UNCERTAINTY_VALIDATION.covered ?? Math.round((UNCERTAINTY_VALIDATION.actual ?? 0) * (UNCERTAINTY_VALIDATION.n ?? 0))}/{UNCERTAINTY_VALIDATION.n ?? "n/a"} ({UNCERTAINTY_VALIDATION.actual === null ? "n/a" : `${(UNCERTAINTY_VALIDATION.actual * 100).toFixed(1)}%`}) against a nominal 90% target. With 22 rows, this is validation evidence, not a future guarantee.</p>
                        <p className="lab-note">Source: {res.uncertainty.source}. Serving and co-design remain uncalibrated, so their intervals are not fabricated.</p>
                      </Card>
                    )}
                    {prefs.provenance && (
                      <Card title="Where the numbers come from">
                        <table className="lab-prov">
                          <tbody>
                            {Object.entries((CLUSTERS[String(S.cluster ?? S.serve_cluster)] ?? CLUSTERS.h100_nvlink_ib)!.provenance).map(([k, v]) => (
                              <tr key={k}><td>{k}</td><td className={String(v).startsWith("spec") || String(v).startsWith("calibrated") ? "ok" : "warn"}>{String(v)}</td></tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="lab-note">spec = vendor datasheet; calibrated = fitted to published measurements; placeholder = representative value, replace with yours</p>
                        <details><summary>Limits of this plan (0 = exactly at the limit)</summary>
                          <table className="lab-prov"><tbody>
                            {Object.entries(selected.metrics.constraints).map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{v <= -1 ? "satisfied" : v.toFixed(3)}</td></tr>)}
                          </tbody></table>
                        </details>
                        {selected.metrics.notes.map((n) => <p key={n} className="lab-note">{n}</p>)}
                      </Card>
                    )}
                  </aside>
                )}
              </div>

              {prefs.table && mode !== "guided" && res && res.front.length > 0 && pack && (
                <Card title={`The ${res.front.length} best trade-offs`} className="lab-table-card">
                  <div className="lab-table-scroll">
                    <table className="lab-table">
                      <thead><tr>{objs.map((o) => <th key={o.name}>{o.label}</th>)}<th>Plan</th></tr></thead>
                      <tbody>
                        {res.front.map((e, i) => (
                          <tr key={i} className={e === selected ? "is-on" : ""} onClick={() => setSelKey(planKey(e))} tabIndex={0}
                            onKeyDown={(k) => { if (k.key === "Enter") setSelKey(planKey(e)); }}>
                            {objs.map((o) => {
                              const u = uncertaintyForPlan(res.uncertainty, e.plan)?.objectives[o.name];
                              return <td key={o.name}>{fmtNum(e.metrics.objectives[o.name]!, o.unit)}{u && <small className="lab-table-ci">90% {fmtNum(u.low, o.unit)}–{fmtNum(u.high, o.unit)}</small>}</td>;
                            })}
                            <td>{e === rec && <b>Recommended: </b>}{pack.describe(e.plan)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}
            </div>
          )}
        </main>
      </div>
      {exportOpen && exportFiles.length > 0 && domain && (
        <ExportDialog files={exportFiles} root={`nomo-plan_${domain.replace("llm_", "").replace("arch_", "")}_${new Date().toISOString().slice(0, 10)}`}
          onClose={() => setExportOpen(false)} />
      )}
    </div>
  );
}

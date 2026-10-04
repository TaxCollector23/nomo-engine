// Export files for a planner result. Every file can be downloaded on its own or bundled as one zip.
import { CALIBRATED, CALIBRATION_RESULTS } from "./calibration";
import { MODELS, params } from "./hardware";
import { CodesignPack, InferencePack, SCALING_LAWS, TrainingPack, type Pack, type Plan } from "./packs";
import { innerPack, type DomainId, type Settings } from "./registry";
import type { Evaluated, Result } from "./search";
import { summary } from "./explain";
import type { Planner } from "./search";
import { uncertaintyForPlan } from "./uncertainty";

export interface ExportFile {
  id: string;
  folder: string;          // top-level folder inside the zip
  name: string;
  mime: string;
  description: string;     // plain-English "what is this"
  content: string;
}

export const FOLDERS: Record<string, string> = {
  "1-summary": "Start here: the recommendation in plain English and as structured data",
  "2-data": "Every plan Nomo evaluated, the best trade-offs, and the exact settings used",
  "3-figures": "Charts ready for slides or papers",
  "4-launch-configs": "Starting configurations for the real tools (check against your versions)",
  "5-paper-materials": "LaTeX table, methods text and references for academic write-ups",
};

const ENGINE_VERSION = "nomo-planner 0.1 (browser engine, verified against the Python reference: 14,194 checks)";

function csvEscape(v: unknown): string {
  const s = v === null || v === undefined ? "" : typeof v === "number" ? (Number.isFinite(v) ? String(v) : "") : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function rowsToCsv(pack: Pack, evs: Evaluated[], uncertainty: Result["uncertainty"]): string {
  const vars = pack.variables().map((v) => v.name);
  const objectives = pack.objectives();
  const objs = objectives.map((o) => o.name);
  const cons = evs.length ? Object.keys(evs[0]!.metrics.constraints) : [];
  const intervalHead = objectives.flatMap((o) => [`${o.name}_median`, `${o.name}_interval90_low`, `${o.name}_interval90_high`, `${o.name}_probability_best`]);
  const head = [...vars, ...objs, ...intervalHead, "uncertainty_method", "posterior_sample_count", "feasible", ...cons.map((c) => `constraint_${c}`), "description"];
  const lines = [head.join(",")];
  for (const e of evs) {
    const feas = Object.values(e.metrics.constraints).every((x) => x <= 0);
    const u = uncertaintyForPlan(uncertainty, e.plan);
    const uncertaintyValues = objectives.flatMap((o) => {
      const q = u?.objectives[o.name];
      return [q?.median, q?.low, q?.high, u?.probabilityBest[o.name]];
    });
    lines.push([...vars.map((v) => e.plan[v]), ...objs.map((o) => e.metrics.objectives[o]), ...uncertaintyValues,
      uncertainty ? uncertainty.method : "unavailable", uncertainty?.sampleCount ?? "", feas,
      ...cons.map((c) => e.metrics.constraints[c]), pack.describe(e.plan)].map(csvEscape).join(","));
  }
  return lines.join("\n") + "\n";
}

function fmt(v: number, unit: string): string {
  if (!Number.isFinite(v)) return "n/a";
  if (unit === "USD") return v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(1)}k` : `$${v.toFixed(2)}`;
  if (unit.startsWith("USD /")) return `$${v.toFixed(3)} per 1M tokens`;
  if (unit === "days") return `${v.toFixed(v < 10 ? 1 : 0)} days`;
  if (unit === "ms") return `${v.toFixed(2)} ms`;
  if (unit === "nats/token") return `${v.toFixed(4)} nats/token`;
  if (unit === "points") return `${v.toFixed(2)} points`;
  return `${v.toPrecision(4)} ${unit}`;
}

/** Trade-off chart as a standalone SVG (log-log, first two objectives). */
export function tradeoffSvg(pack: Pack, res: Result, selected: Evaluated | null): string {
  const W = 720, H = 440, pl = 90, pb = 60, pr = 24, pt = 30;
  const [ox, oy] = pack.objectives();
  const pts = res.all.filter((e) => Object.values(e.metrics.constraints).every((x) => x <= 0));
  const xs = pts.map((e) => e.metrics.objectives[ox!.name]!), ys = pts.map((e) => e.metrics.objectives[oy!.name]!);
  const lg = (v: number) => Math.log10(Math.max(v, 1e-12));
  const x0 = Math.min(...xs.map(lg)), x1 = Math.max(...xs.map(lg)), y0 = Math.min(...ys.map(lg)), y1 = Math.max(...ys.map(lg));
  const X = (v: number) => pl + ((lg(v) - x0) / (x1 - x0 || 1)) * (W - pl - pr);
  const Y = (v: number) => H - pb - ((lg(v) - y0) / (y1 - y0 || 1)) * (H - pb - pt);
  const dots = pts.map((e) => `<circle cx="${X(e.metrics.objectives[ox!.name]!).toFixed(1)}" cy="${Y(e.metrics.objectives[oy!.name]!).toFixed(1)}" r="2" fill="#bdb4a8"/>`).join("");
  const front = res.front.map((e) => {
    const x = X(e.metrics.objectives[ox!.name]!), y = Y(e.metrics.objectives[oy!.name]!);
    const u = uncertaintyForPlan(res.uncertainty, e.plan);
    const xi = u?.objectives[ox!.name], yi = u?.objectives[oy!.name];
    return `${xi ? `<line x1="${X(xi.low).toFixed(1)}" x2="${X(xi.high).toFixed(1)}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#8f4638" opacity=".35"/>` : ""}${yi ? `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${Y(yi.low).toFixed(1)}" y2="${Y(yi.high).toFixed(1)}" stroke="#8f4638" opacity=".35"/>` : ""}<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" fill="#11100e"/>`;
  }).join("");
  const sel = selected ? `<circle cx="${X(selected.metrics.objectives[ox!.name]!).toFixed(1)}" cy="${Y(selected.metrics.objectives[oy!.name]!).toFixed(1)}" r="9" fill="none" stroke="#11100e" stroke-width="1.6"/>` : "";
  const ticks = [0, 0.5, 1].map((f) => {
    const vx = 10 ** (x0 + f * (x1 - x0)), vy = 10 ** (y0 + f * (y1 - y0));
    return `<text x="${X(vx)}" y="${H - pb + 18}" text-anchor="middle">${fmt(vx, ox!.unit)}</text><text x="${pl - 8}" y="${Y(vy) + 4}" text-anchor="end">${fmt(vy, oy!.unit)}</text>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#5b554d">
<rect width="${W}" height="${H}" fill="#ffffff"/><rect x="${pl}" y="${pt}" width="${W - pl - pr}" height="${H - pb - pt}" fill="none" stroke="#d4cec4"/>
${dots}${front}${sel}${ticks}
<text x="${(W + pl) / 2}" y="${H - 18}" text-anchor="middle" fill="#11100e" font-size="12">${ox!.label} (log scale)</text>
<text x="18" y="${H / 2}" text-anchor="middle" fill="#11100e" font-size="12" transform="rotate(-90 18 ${H / 2})">${oy!.label} (log scale)</text>
<text x="${pl}" y="18" fill="#11100e" font-size="12">${pts.length.toLocaleString("en-US")} feasible plans (grey), ${res.front.length} best trade-offs (black), chosen plan (circled)</text>
</svg>`;
}

function launchConfigs(domain: DomainId, pack: Pack, plan: Plan): ExportFile[] {
  const out: ExportFile[] = [];
  const warn = "# Generated by Nomo Planner as a STARTING POINT. Flag names follow current Megatron-LM / DeepSpeed / vLLM\n# documentation; verify against the versions you run.\n";
  if (domain === "llm_training" && pack instanceof TrainingPack) {
    const M = pack.p.model, t = Number(plan.tp), p = Number(plan.pp), z = Number(plan.zero), mb = Number(plan.micro_batch);
    const d = Math.floor(Number(plan.devices) / (t * p));
    const seqs = Math.floor(pack.p.globalBatchTokens / pack.p.seqLen);
    const recompute = plan.recompute === "none" ? [] : plan.recompute === "selective"
      ? ["--recompute-granularity selective"] : ["--recompute-granularity full", "--recompute-method uniform", "--recompute-num-layers 1"];
    const args = [
      `--num-layers ${M.layers}`, `--hidden-size ${M.hidden}`, `--num-attention-heads ${M.heads}`,
      ...(M.kvHeads !== M.heads ? ["--group-query-attention", `--num-query-groups ${M.kvHeads}`] : []),
      `--ffn-hidden-size ${M.ffn}`, ...(M.gatedMlp ? ["--swiglu"] : []), `--seq-length ${pack.p.seqLen}`,
      `--tensor-model-parallel-size ${t}`, `--pipeline-model-parallel-size ${p}`, ...(t > 1 ? ["--sequence-parallel"] : []),
      `--micro-batch-size ${mb}`, `--global-batch-size ${seqs}`, ...recompute,
      ...(z >= 1 ? ["--use-distributed-optimizer"] : []), "--bf16", ...(plan.matmul_dtype === "fp8" ? ["--fp8-format hybrid"] : []),
    ];
    out.push({ id: "megatron", folder: "4-launch-configs", name: "megatron-lm-args.sh", mime: "text/x-sh",
      description: "Megatron-LM arguments for this layout (ZeRO-1 maps to --use-distributed-optimizer)",
      content: `#!/usr/bin/env bash\n${warn}${z >= 2 ? "# NOTE: this plan uses ZeRO stage " + z + ", which Megatron-LM's distributed optimizer does not provide;\n# use the DeepSpeed configuration in this folder instead.\n" : ""}# Layout: ${pack.describe(plan)} (data parallel = ${d})\nNOMO_ARGS=(\n  ${args.join("\n  ")}\n)\n# torchrun --nproc_per_node 8 --nnodes ${Math.ceil(Number(plan.devices) / 8)} pretrain_gpt.py "\${NOMO_ARGS[@]}" <data and tokenizer args>\n` });
    const ds = {
      train_batch_size: seqs, train_micro_batch_size_per_gpu: mb, gradient_accumulation_steps: Math.max(1, Math.floor(seqs / (d * mb))),
      zero_optimization: { stage: z, overlap_comm: true, contiguous_gradients: true },
      bf16: { enabled: true }, activation_checkpointing: { partition_activations: plan.recompute !== "none", contiguous_memory_optimization: false },
      _nomo: { note: "Starting point generated by Nomo Planner; tensor/pipeline sizes are set in your launcher.", tensor_parallel: t, pipeline_parallel: p, data_parallel: d },
    };
    out.push({ id: "deepspeed", folder: "4-launch-configs", name: "deepspeed-config.json", mime: "application/json",
      description: "DeepSpeed configuration (batch sizes, ZeRO stage, precision)", content: JSON.stringify(ds, null, 2) + "\n" });
  }
  if (domain === "llm_inference" && pack instanceof InferencePack) {
    const t = Number(plan.tp), B = Number(plan.batch);
    const q = plan.weights === "fp8" ? ["--quantization fp8"] : plan.weights === "int4" ? ["--quantization awq   # int4: serve an AWQ (or GPTQ) int4 checkpoint"] : [];
    const kv = plan.kv === "fp8" ? ["--kv-cache-dtype fp8"] : plan.kv === "int4" ? ["# int4 KV cache: not a standard vLLM option; nearest supported is --kv-cache-dtype fp8"] : [];
    out.push({ id: "vllm", folder: "4-launch-configs", name: "vllm-serve.sh", mime: "text/x-sh",
      description: "vLLM serve command for this configuration",
      content: `#!/usr/bin/env bash\n${warn}# Plan: ${pack.describe(plan)}\nvllm serve <your-model> \\\n  --tensor-parallel-size ${t} \\\n  --max-num-seqs ${B} \\\n  --max-model-len ${pack.p.promptTokens + pack.p.outputTokens}${[...q, ...kv].map((a) => ` \\\n  ${a}`).join("")}\n` });
  }
  if (domain === "arch_codesign" && pack instanceof CodesignPack) {
    const M = pack.model(plan);
    const cfg = {
      architectures: ["LlamaForCausalLM"], model_type: "llama", num_hidden_layers: M.layers, hidden_size: M.hidden,
      intermediate_size: M.ffn, num_attention_heads: M.heads, num_key_value_heads: M.kvHeads, vocab_size: M.vocab,
      max_position_embeddings: pack.p.seqLen, hidden_act: "silu", torch_dtype: "bfloat16",
      _nomo: { parameters: params(M), training_tokens: Number(plan.tokens_per_param) * params(M), note: "Architecture chosen by Nomo Planner; weights are not included." },
    };
    out.push({ id: "hfconfig", folder: "4-launch-configs", name: "model-config.json", mime: "application/json",
      description: "Hugging Face-style architecture config for the recommended model", content: JSON.stringify(cfg, null, 2) + "\n" });
  }
  return out;
}

const REFERENCES = `@inproceedings{narayanan2021megatron, title={Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM}, author={Narayanan, Deepak and Shoeybi, Mohammad and Casper, Jared and others}, booktitle={SC21}, year={2021}, note={arXiv:2104.04473}}
@article{korthikanti2022reducing, title={Reducing Activation Recomputation in Large Transformer Models}, author={Korthikanti, Vijay and Casper, Jared and Lym, Sangkug and others}, year={2022}, note={arXiv:2205.05198}}
@inproceedings{rajbhandari2020zero, title={ZeRO: Memory Optimizations Toward Training Trillion Parameter Models}, author={Rajbhandari, Samyam and Rasley, Jeff and Ruwase, Olatunji and He, Yuxiong}, booktitle={SC20}, year={2020}, note={arXiv:1910.02054}}
@article{chowdhery2022palm, title={PaLM: Scaling Language Modeling with Pathways}, author={Chowdhery, Aakanksha and others}, year={2022}, note={arXiv:2204.02311}}
@article{hoffmann2022chinchilla, title={Training Compute-Optimal Large Language Models}, author={Hoffmann, Jordan and others}, year={2022}, note={arXiv:2203.15556}}
@article{besiroglu2024chinchilla, title={Chinchilla Scaling: A replication attempt}, author={Besiroglu, Tamay and Erdil, Ege and Barnett, Matthew and You, Josh}, year={2024}, note={arXiv:2404.10102}}
@article{sardana2024beyond, title={Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws}, author={Sardana, Nikhil and Portes, Jacob and Doubov, Sasha and Frankle, Jonathan}, year={2024}, note={arXiv:2401.00448}}
@article{kaplan2020scaling, title={Scaling Laws for Neural Language Models}, author={Kaplan, Jared and others}, year={2020}, note={arXiv:2001.08361}}
@article{ainslie2023gqa, title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints}, author={Ainslie, Joshua and others}, year={2023}, note={arXiv:2305.13245}}
@article{williams2009roofline, title={Roofline: An Insightful Visual Performance Model for Multicore Architectures}, author={Williams, Samuel and Waterman, Andrew and Patterson, David}, journal={Communications of the ACM}, volume={52}, number={4}, year={2009}}
@article{thakur2005collectives, title={Optimization of Collective Communication Operations in MPICH}, author={Thakur, Rajeev and Rabenseifner, Rolf and Gropp, William}, journal={International Journal of High Performance Computing Applications}, volume={19}, number={1}, year={2005}}
@misc{nomoplanner, title={Nomo Planner: a constrained multi-objective planner for how AI workloads run on hardware}, author={{Nomo AI}}, year={2026}, note={Software. https://github.com/TaxCollector23/nomo-ai}}
`;

function methodsText(domain: DomainId, pack: Pack): string {
  const cal = pack.calibration;
  if (domain === "llm_training") {
    return `Training plans were generated with Nomo Planner (${ENGINE_VERSION}). For each candidate layout (GPU count; tensor, pipeline and data parallel sizes; ZeRO stage; activation recomputation; micro-batch size; matmul precision) the planner computes per-GPU memory from exact parameter counts, mixed-precision Adam state (16 bytes/parameter, sharded per ZeRO stage; Rajbhandari et al., 2020) and activation memory (Korthikanti et al., 2022), and step time from training FLOPs (6N + 12Lsh per token; Chowdhery et al., 2022), the 1F1B pipeline bubble, and ring all-reduce / all-gather communication (Thakur et al., 2005). All ${pack.variables().reduce((n, v) => n * v.choices.length, 1).toLocaleString("en-US")} combinations were evaluated exhaustively; the reported set is the exact Pareto front of the model. ${cal ? `Efficiency parameters for this hardware were fitted to ${cal.observations} published measurements (${cal.source}); leave-one-out error ${cal.looMapePct}%, error on held-out runs of the fitted strategy ${CALIBRATION_RESULTS.heldOutPtdMape}%. Uncertainty uses a stratified bootstrap; the nominal 90% interval covered 18/22 leave-one-out runs (81.8%), so intervals are evidence with a documented shortfall, not a guarantee.` : "Efficiency parameters for this hardware are uncalibrated placeholders; absolute times and costs are indicative only."}`;
  }
  if (domain === "llm_inference") {
    return `Serving configurations were generated with Nomo Planner (${ENGINE_VERSION}). Decode time per token is modelled with the roofline model (Williams et al., 2009) on bytes moved (weights plus average KV cache) and FLOPs, plus tensor-parallel all-reduce time; prefill is modelled likewise. Memory includes weights and the full KV cache for the batch. Quality loss from reduced precision is an explicit assumption table, not a measurement. Throughput is an uncalibrated upper bound.`;
  }
  const cp = pack as CodesignPack;
  const law = SCALING_LAWS[cp.p.scalingLaw]!;
  return `Architectures were chosen with Nomo Planner (${ENGINE_VERSION}). Predicted loss uses the parametric scaling law L(N, D) = E + A/N^α + B/D^β with ${law.label} coefficients (E = ${law.E}, A = ${law.A}, B = ${law.B}, α = ${law.alpha}, β = ${law.beta}). Lifetime cost is training cost (6ND + 12LshD FLOPs at an assumed ${(cp.p.trainMfu * 100).toFixed(0)}% utilisation) plus serving cost from the roofline serving model at the best feasible parallelism and batch size. The quality effect of grouped- and multi-query attention (Ainslie et al., 2023) is an assumption. Following Sardana et al. (2024), plans above ~100 tokens per parameter extrapolate the scaling law.`;
}

function latexTable(pack: Pack, res: Result): string {
  const objs = pack.objectives();
  const rows = res.front.slice(0, 15).map((e) =>
    `${objs.map((o) => fmt(e.metrics.objectives[o.name]!, o.unit).replace(/\$/g, "\\$").replace(/%/g, "\\%")).join(" & ")} & ${pack.describe(e.plan).replace(/_/g, "\\_").replace(/%/g, "\\%")} \\\\`);
  return `% Generated by Nomo Planner. Best trade-offs (exact Pareto front of the model), first ${rows.length} of ${res.front.length}.
\\begin{table}[t]
\\centering\\small
\\begin{tabular}{${"r".repeat(objs.length)}p{7cm}}
\\toprule
${objs.map((o) => o.label).join(" & ")} & Plan \\\\
\\midrule
${rows.join("\n")}
\\bottomrule
\\end{tabular}
\\caption{Best trade-offs found by exhaustive search over ${res.evaluated.toLocaleString("en-US")} plans.}
\\end{table}
`;
}

export function buildExport(domain: DomainId, settings: Settings, locks: Record<string, unknown>, pl: Planner, res: Result,
  selected: Evaluated): ExportFile[] {
  const pack = pl.pack;
  const date = new Date().toISOString().slice(0, 10);
  const objs = pack.objectives();
  const story = summary(pl, selected.plan);
  const files: ExportFile[] = [];
  const selectedUncertainty = uncertaintyForPlan(res.uncertainty, selected.plan);
  const numbers = objs.map((o) => {
    const q = selectedUncertainty?.objectives[o.name];
    const probability = selectedUncertainty?.probabilityBest[o.name];
    return `- **${o.label}:** ${fmt(selected.metrics.objectives[o.name]!, o.unit)}${q ? ` (90% interval ${fmt(q.low, o.unit)}–${fmt(q.high, o.unit)})` : " (90% interval unavailable)"}${probability === undefined ? "" : `; ${(probability * 100).toFixed(0)}% probability of being best`}`;
  }).join("\n");
  files.push({ id: "summary", folder: "1-summary", name: "plan-summary.md", mime: "text/markdown",
    description: "The recommendation and why, in plain English",
    content: `# Nomo Planner: ${pack.description}\n\nGenerated ${date} by ${ENGINE_VERSION}.\n\n## Chosen plan\n\n${pack.describe(selected.plan)}\n\n${numbers}\n\n## Why\n\n${story}\n\n## How it was found\n\nEvery one of ${res.evaluated.toLocaleString("en-US")} possible plans was evaluated; ${res.front.length} are best trade-offs (no other plan is better on every goal at once).\n\n## Reliability\n\n${pack.calibration ? `Hardware efficiency calibrated on ${pack.calibration.observations} published runs (${pack.calibration.source}); held-out error ${CALIBRATION_RESULTS.heldOutPtdMape}% for the calibrated strategy, leave-one-out ${pack.calibration.looMapePct}%.` : "Hardware efficiency for this setup is uncalibrated: treat absolute numbers as indicative, comparisons between plans as the main result."}\n${selected.metrics.notes.map((n) => `\n- ${n}`).join("")}\n` });
  files.push({ id: "recommendation", folder: "1-summary", name: "recommendation.json", mime: "application/json",
    description: "The chosen plan with every metric, constraint and breakdown",
    content: JSON.stringify({ engine: ENGINE_VERSION, date, domain, plan: selected.plan, description: pack.describe(selected.plan),
      objectives: selected.metrics.objectives, constraints: selected.metrics.constraints, breakdown: selected.metrics.breakdown,
      notes: selected.metrics.notes, calibration: pack.calibration, uncertainty: selectedUncertainty,
      uncertainty_validation: res.uncertainty?.validation ?? null }, (_k, v) => (typeof v === "number" && !Number.isFinite(v) ? null : v), 2) + "\n" });
  files.push({ id: "front", folder: "2-data", name: "best-tradeoffs.csv", mime: "text/csv",
    description: `The ${res.front.length} best trade-off plans`, content: rowsToCsv(pack, res.front, res.uncertainty) });
  files.push({ id: "all", folder: "2-data", name: "all-evaluated-plans.csv", mime: "text/csv",
    description: `All ${res.evaluated.toLocaleString("en-US")} plans with every metric (feasible and not)`, content: rowsToCsv(pack, res.all, res.uncertainty) });
  files.push({ id: "settings", folder: "2-data", name: "settings.json", mime: "application/json",
    description: "Exact inputs, locks and parameters, to reproduce this result",
    content: JSON.stringify({ engine: ENGINE_VERSION, domain, settings, locks,
      calibrated_parameters: domain === "llm_training" ? CALIBRATED[String(settings.cluster)]?.params ?? null : null }, null, 2) + "\n" });
  files.push({ id: "chart", folder: "3-figures", name: "tradeoff-chart.svg", mime: "image/svg+xml",
    description: "Trade-off chart (every feasible plan, best trade-offs, chosen plan)", content: tradeoffSvg(pack, res, selected) });
  files.push(...launchConfigs(domain, innerPack(pack), selected.plan));
  files.push({ id: "table", folder: "5-paper-materials", name: "best-tradeoffs-table.tex", mime: "application/x-tex",
    description: "LaTeX table of the best trade-offs (booktabs)", content: latexTable(pack, res) });
  files.push({ id: "methods", folder: "5-paper-materials", name: "methods.md", mime: "text/markdown",
    description: "Methods paragraph describing exactly how the result was computed", content: methodsText(domain, innerPack(pack)) + "\n" });
  files.push({ id: "refs", folder: "5-paper-materials", name: "references.bib", mime: "application/x-bibtex",
    description: "BibTeX for every method the planner relies on", content: REFERENCES });
  return files;
}

export function readme(files: ExportFile[], root: string): string {
  const byFolder = Object.keys(FOLDERS).filter((f) => files.some((x) => x.folder === f));
  return `NOMO PLANNER EXPORT\n===================\n\n${root}/\n  README.txt              this file\n${byFolder.map((f) => `  ${f}/\n${files.filter((x) => x.folder === f).map((x) => `      ${x.name.padEnd(28)}${x.description}`).join("\n")}`).join("\n")}\n\nFOLDERS\n${byFolder.map((f) => `  ${f.padEnd(20)}${FOLDERS[f]}`).join("\n")}\n\nEngine: ${ENGINE_VERSION}\n`;
}

export async function zipFiles(files: ExportFile[], root: string): Promise<Blob> {
  const { default: JSZip } = await import("jszip");
  const zip = new JSZip();
  const dir = zip.folder(root)!;
  dir.file("README.txt", readme(files, root));
  for (const f of files) dir.folder(f.folder)!.file(f.name, f.content);
  return zip.generateAsync({ type: "blob", compression: "DEFLATE" });
}

export function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export { MODELS };

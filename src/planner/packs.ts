// Domain packs. Port of nomo_planner/domains/*.py; see those files for formula citations.
import {
  CLUSTERS, allGather, allReduce, flops, linkForGroup, p2p, params, paramsPerLayer, headDim, rooflineTime,
  type Cluster, type Link, type Transformer,
} from "./hardware";

export type Choice = string | number;
export type Plan = Record<string, Choice>;
export interface Variable { name: string; choices: Choice[]; label: string; help?: string }
export interface Objective { name: string; unit: string; label: string; maximize?: boolean }
export interface Metrics {
  objectives: Record<string, number>;
  constraints: Record<string, number>; // <= 0 satisfied
  breakdown: Record<string, unknown>;
  notes: string[];
}
export const violation = (m: Metrics) => Object.values(m.constraints).reduce((a, v) => a + Math.max(0, v), 0);
export const feasible = (m: Metrics) => violation(m) <= 0;

export interface Pack {
  name: string;
  description: string;
  calibration: { observations: number; looMapePct: number; source: string } | null;
  variables(): Variable[];
  objectives(): Objective[];
  evaluate(plan: Plan): Metrics;
  describe(plan: Plan): string;
}

// ================================================================== training
export interface TrainingProblem {
  model: Transformer; cluster: Cluster; seqLen: number; globalBatchTokens: number; totalTokens: number;
  deviceCounts: number[]; usableMemory: number;
}
export const TRAINING_DEFAULTS: Record<string, number> = {
  e_matmul: 0.55, dp_overlap: 0.7, overhead_gb: 3.0, bubble_scale: 1.0, zero3_overlap: 0.7, gemm_h_half: 0.0, zero3_latency_us: -1.0,
};

export class TrainingPack implements Pack {
  name = "llm_training";
  description = "Plan the parallel layout of an LLM training run for minimum time and cost.";
  calibration: Pack["calibration"] = null;
  params: Record<string, number>;
  constructor(public p: TrainingProblem, params?: Record<string, number>) {
    this.params = { ...TRAINING_DEFAULTS, ...(params ?? {}) };
  }
  variables(): Variable[] {
    const dtypes = ["bf16", "fp8"].filter((d) => d in this.p.cluster.device.peakFlops);
    return [
      { name: "devices", choices: [...this.p.deviceCounts], label: "number of GPUs" },
      { name: "tp", choices: [1, 2, 4, 8], label: "tensor parallel size", help: "splits each layer's matrices across GPUs" },
      { name: "pp", choices: [1, 2, 4, 8, 16], label: "pipeline stages", help: "splits the layers into consecutive stages" },
      { name: "zero", choices: [0, 1, 2, 3], label: "ZeRO stage", help: "shards optimizer / gradients / weights across data parallel" },
      { name: "recompute", choices: ["none", "selective", "full"], label: "activation recomputation" },
      { name: "micro_batch", choices: [1, 2, 4, 8], label: "micro-batch size (sequences)" },
      { name: "matmul_dtype", choices: dtypes, label: "matmul precision" },
    ];
  }
  objectives(): Objective[] {
    return [{ name: "days", unit: "days", label: "training time" }, { name: "cost_usd", unit: "USD", label: "training cost" }];
  }
  describe(plan: Plan): string {
    const d = Math.floor(Number(plan.devices) / Math.max(1, Number(plan.tp) * Number(plan.pp)));
    const rc = { none: "no", selective: "selective", full: "full" }[String(plan.recompute) as "none"];
    return `${plan.devices} GPUs as TP${plan.tp} x PP${plan.pp} x DP${d}, ZeRO-${plan.zero}, ${rc} recompute, micro-batch ${plan.micro_batch}, ${plan.matmul_dtype}`;
  }
  evaluate(plan: Plan): Metrics {
    const P = this.p, M = P.model, prm = this.params;
    const G = Number(plan.devices), t = Number(plan.tp), p = Number(plan.pp), z = Number(plan.zero);
    const b = Number(plan.micro_batch), rc = String(plan.recompute), dt = String(plan.matmul_dtype);
    const nodes = Math.max(1, Math.ceil(G / P.cluster.devicesPerNode));
    const cl = P.cluster; void nodes;
    const dev = cl.device;
    const s = P.seqLen, h = M.hidden, a = M.heads, L = M.layers;
    const notes: string[] = [];
    const problems: string[] = [];
    if (G % (t * p)) problems.push("GPU count not divisible by TP x PP");
    const d = Math.max(1, Math.floor(G / (t * p)));
    if (a % t) problems.push("attention heads not divisible by TP");
    if (M.kvHeads % t && t % M.kvHeads) problems.push("KV heads incompatible with TP");
    if (L % p) problems.push("layers not divisible by PP");
    const seqs = Math.floor(P.globalBatchTokens / s);
    if (seqs % (d * b)) problems.push("global batch not divisible by DP x micro-batch");
    const m = Math.max(1, Math.floor(seqs / (d * b)));

    const perLayer = paramsPerLayer(M);
    const layersPerStage = Math.floor(L / Math.max(p, 1));
    const stageParams = layersPerStage * perLayer + M.vocab * h * (p > 1 || M.tiedEmbeddings ? 1 : 2);
    const nLocal = stageParams / t;
    const w = (2 * nLocal) / (z >= 3 ? d : 1);
    const g = (2 * nLocal) / (z >= 2 ? d : 1);
    const o = (12 * nLocal) / (z >= 1 ? d : 1);
    let actLayer: number;
    if (rc === "none") actLayer = (s * b * h * (34 + (5 * a * s) / h)) / t;
    else if (rc === "selective") actLayer = (s * b * h * 34) / t;
    else actLayer = (2 * s * b * h) / t;
    const act = actLayer * layersPerStage * Math.min(p, m);
    const logits = p === 1 ? (4 * s * b * M.vocab) / t : 0;
    const mem = w + g + o + act + logits + prm.overhead_gb * 1e9;
    const cap = dev.memBytes * P.usableMemory;

    const N = params(M);
    const attn = 12 * L * s * h;
    const extra = rc === "none" ? 0 : rc === "selective" ? 4 * L * s * h : 2 * N + 4 * L * s * h;
    const hwTok = 6 * N + attn + extra, modelTok = 6 * N + attn;
    const tokens = P.globalBatchTokens;
    const peak = flops(dev, dt);
    const wLocal = h / t;
    const eEff = (prm.e_matmul * wLocal) / (wLocal + prm.gemm_h_half);
    const compute = (hwTok * tokens) / (G * peak * eEff);
    const bubble = 1 + (prm.bubble_scale * (p - 1)) / m;
    const msg = b * s * h * 2.0;
    const nAr = 4 + (rc === "full" ? 2 : 0);
    const tTp = t > 1 ? layersPerStage * m * nAr * allReduce(msg, t, linkForGroup(cl, t)) : 0;
    const tPp = p > 1 ? 2 * m * p2p(msg / t, linkForGroup(cl, p, t)) : 0;
    const dpLink = linkForGroup(cl, d, t * p);
    const dpBytes = 2 * nLocal;
    let tDp = allReduce(dpBytes, d, dpLink) * (1 - prm.dp_overlap);
    if (z >= 3) {
      const Ls = Math.max(1, layersPerStage);
      const z3: Link = prm.zero3_latency_us >= 0 ? { bw: dpLink.bw, latencyS: prm.zero3_latency_us * 1e-6 } : dpLink;
      tDp += 2 * m * Ls * allGather(dpBytes / Ls, d, z3) * (1 - prm.zero3_overlap);
    }
    const step = compute * bubble + tTp + tPp + tDp;
    const seconds = step * (P.totalTokens / tokens);
    const cost = (seconds / 3600) * G * cl.costPerDeviceHour;
    if (dt === "fp8") notes.push("fp8 matmuls: speed modelled; training-quality effects are not modelled");
    return {
      objectives: { days: seconds / 86400, cost_usd: cost },
      constraints: { valid_layout: problems.length ? problems.length : -1.0, memory: mem / cap - 1.0 },
      breakdown: {
        params: N, dp: d, micro_batches: m, step_s: step, tokens_per_s: tokens / step,
        mfu: (modelTok * tokens) / (step * G * peak), energy_mwh: (G * dev.powerW * seconds) / 3.6e9,
        memory_gb: mem / 1e9, memory_cap_gb: cap / 1e9,
        memory_parts_gb: { weights: w / 1e9, grads: g / 1e9, optimizer: o / 1e9, activations: act / 1e9, logits: logits / 1e9, overhead: prm.overhead_gb },
        time_parts_s: { compute, pipeline_bubble: compute * (bubble - 1), tensor_parallel_comm: tTp, pipeline_comm: tPp, data_parallel_comm_exposed: tDp },
        layout_problems: problems,
      },
      notes,
    };
  }
}

// ================================================================== serving
export const BYTES: Record<string, number> = { bf16: 2, fp8: 1, int4: 0.5 };
export const DEFAULT_QUALITY_LOSS = { weights: { bf16: 0, fp8: 0.2, int4: 1.0 }, kv: { bf16: 0, fp8: 0.1, int4: 0.8 } } as Record<string, Record<string, number>>;

export interface ServingProblem {
  model: Transformer; cluster: Cluster; promptTokens: number; outputTokens: number; maxMsPerToken: number | null;
  maxQualityLoss: number; usableMemory: number; batchSizes: number[]; qualityLoss: Record<string, Record<string, number>>;
}

export class InferencePack implements Pack {
  name = "llm_inference";
  description = "Choose precision, parallelism and batch size to serve an LLM at minimum cost per token.";
  calibration: Pack["calibration"] = null;
  overheadGb = 2.0;
  constructor(public p: ServingProblem) {}
  variables(): Variable[] {
    return [
      { name: "weights", choices: ["bf16", "fp8", "int4"], label: "weight precision" },
      { name: "kv", choices: ["bf16", "fp8", "int4"], label: "KV-cache precision" },
      { name: "tp", choices: [1, 2, 4, 8], label: "GPUs per model copy (tensor parallel)" },
      { name: "batch", choices: [...this.p.batchSizes], label: "concurrent sequences" },
    ];
  }
  objectives(): Objective[] {
    return [
      { name: "usd_per_mtok", unit: "USD / 1M output tokens", label: "cost per million tokens" },
      { name: "ms_per_token", unit: "ms", label: "time per generated token" },
      { name: "quality_loss", unit: "points", label: "assumed quality loss" },
    ];
  }
  describe(plan: Plan): string {
    return `${plan.weights} weights, ${plan.kv} KV cache, ${plan.tp} GPU(s) per copy, batch ${plan.batch}`;
  }
  evaluate(plan: Plan): Metrics {
    const P = this.p, M = P.model, cl = P.cluster, dev = cl.device;
    const t = Number(plan.tp), B = Number(plan.batch), wq = String(plan.weights), kq = String(plan.kv);
    const problems: string[] = [];
    if (t > cl.devicesPerNode) problems.push("tensor parallel larger than a node");
    if (M.heads % t || (M.kvHeads % t && t % M.kvHeads)) problems.push("heads not divisible by tensor parallel");
    if (wq === "fp8" && !("fp8" in dev.peakFlops)) problems.push(`${dev.name} has no fp8 support`);
    const cdt = wq === "fp8" && "fp8" in dev.peakFlops ? "fp8" : "bf16";
    const N = params(M), L = M.layers, h = M.hidden, sIn = P.promptTokens, sOut = P.outputTokens;
    const wBytes = N * BYTES[wq]!;
    const kvTok = 2 * L * M.kvHeads * headDim(M) * BYTES[kq]!;
    const kvFull = B * (sIn + sOut) * kvTok;
    const mem = (wBytes + kvFull) / t + this.overheadGb * 1e9;
    const cap = dev.memBytes * P.usableMemory;
    const prefillFlops = ((2 * N * sIn + 2 * L * sIn * sIn * h) * B) / t;
    const prefill = rooflineTime(prefillFlops, wBytes / t, dev, cdt);
    const kvAvg = B * (sIn + sOut / 2) * kvTok;
    const decFlops = ((2 * N + 4 * L * (sIn + sOut / 2) * h) * B) / t;
    const link = linkForGroup(cl, t);
    let decode = rooflineTime(decFlops, (wBytes + kvAvg) / t, dev, cdt);
    decode += t > 1 ? 2 * L * allReduce(B * h * 2.0, t, link) : 0;
    const cycle = prefill + sOut * decode;
    const tokS = (B * sOut) / cycle;
    const usd = ((cl.costPerDeviceHour * t) / 3600 / tokS) * 1e6;
    const q = P.qualityLoss.weights![wq]! + P.qualityLoss.kv![kq]!;
    const cons: Record<string, number> = {
      valid: problems.length ? problems.length : -1.0, memory: mem / cap - 1.0,
      quality: (q - P.maxQualityLoss) / Math.max(P.maxQualityLoss, 1e-9),
    };
    if (P.maxMsPerToken !== null) cons.latency = (decode * 1e3) / P.maxMsPerToken - 1.0;
    const memT = (wBytes + kvAvg) / (dev.memBw * dev.efficiency.memory);
    const cmpT = decFlops / (flops(dev, cdt) * dev.efficiency.matmul);
    return {
      objectives: { usd_per_mtok: usd, ms_per_token: decode * 1e3, quality_loss: q },
      constraints: cons,
      breakdown: {
        tokens_per_s: tokS, tokens_per_s_per_gpu: tokS / t, memory_gb: mem / 1e9, memory_cap_gb: cap / 1e9,
        prefill_s: prefill, decode_step_s: decode, kv_gb: kvFull / 1e9 / t, weights_gb: wBytes / 1e9 / t,
        decode_bound: memT > cmpT ? "memory" : "compute", layout_problems: problems,
        memory_parts_gb: { weights: wBytes / 1e9 / t, "KV cache": kvFull / 1e9 / t, overhead: this.overheadGb },
      },
      notes: ["quality_loss values are assumptions until replaced by your own evaluations"],
    };
  }
}

// ================================================================== co-design
export const SCALING_LAWS: Record<string, { E: number; A: number; B: number; alpha: number; beta: number; label: string; cite: string }> = {
  besiroglu2024: { E: 1.8172, A: 482.01, B: 2085.43, alpha: 0.3478, beta: 0.3658, label: "Besiroglu et al. 2024 (replication)", cite: "arXiv:2404.10102" },
  hoffmann2022: { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28, label: "Hoffmann et al. 2022 (Chinchilla)", cite: "arXiv:2203.15556" },
};
export const DEFAULT_ATTENTION_PENALTY: Record<string, number> = { MHA: 0, "GQA-8": 0.002, MQA: 0.01 };

export interface CodesignProblem {
  trainCluster: Cluster; serveCluster: Cluster; tokensServed: number; promptTokens: number; outputTokens: number;
  maxMsPerToken: number | null; budgetUsd: number | null; maxTrainTokens: number | null; seqLen: number; vocab: number;
  scalingLaw: string; trainMfu: number; attentionLossPenalty: Record<string, number>;
  layers: number[]; hidden: number[]; tokensPerParam: number[];
}

export class CodesignPack implements Pack {
  name = "arch_codesign";
  description = "Choose the model to build: size, shape, attention type and training length, for quality vs lifetime cost.";
  calibration: Pack["calibration"] = null;
  private serveCache = new Map<string, { usd: number; ms: number; tp: number; batch: number } | null>();
  constructor(public p: CodesignProblem) {
    if (!(p.scalingLaw in SCALING_LAWS)) throw new Error(`unknown scaling law '${p.scalingLaw}'`);
  }
  variables(): Variable[] {
    return [
      { name: "layers", choices: [...this.p.layers], label: "layers (depth)" },
      { name: "hidden", choices: [...this.p.hidden], label: "hidden size (width)" },
      { name: "attention", choices: ["MHA", "GQA-8", "MQA"], label: "attention type", help: "fewer key/value heads = smaller KV cache = cheaper serving" },
      { name: "tokens_per_param", choices: [...this.p.tokensPerParam], label: "training tokens per parameter" },
    ];
  }
  objectives(): Objective[] {
    return [{ name: "loss", unit: "nats/token", label: "predicted loss" }, { name: "total_cost_usd", unit: "USD", label: "lifetime cost" }];
  }
  model(plan: Plan): Transformer {
    const h = Number(plan.hidden), L = Number(plan.layers), heads = Math.floor(h / 128);
    const kv = plan.attention === "MHA" ? heads : plan.attention === "GQA-8" ? Math.min(8, heads) : 1;
    const ffn = Math.round(((8 / 3) * h) / 256) * 256;
    return { name: `${L}x${h} ${plan.attention}`, layers: L, hidden: h, heads, kvHeads: kv, ffn, vocab: this.p.vocab, gatedMlp: true, tiedEmbeddings: false };
  }
  describe(plan: Plan): string {
    const N = params(this.model(plan));
    return `${(N / 1e9).toFixed(1)}B params (${plan.layers} layers x ${plan.hidden} wide, ${plan.attention}), trained on ${plan.tokens_per_param} tokens/param (${((Number(plan.tokens_per_param) * N) / 1e12).toFixed(2)}T tokens)`;
  }
  private serving(M: Transformer) {
    const key = `${M.layers}/${M.hidden}/${M.kvHeads}`;
    if (this.serveCache.has(key)) return this.serveCache.get(key)!;
    const P = this.p;
    const pack = new InferencePack({ model: M, cluster: P.serveCluster, promptTokens: P.promptTokens, outputTokens: P.outputTokens,
      maxMsPerToken: P.maxMsPerToken, maxQualityLoss: 1e9, usableMemory: 0.9, batchSizes: [1, 8, 32, 64, 128, 256], qualityLoss: DEFAULT_QUALITY_LOSS });
    let best: { usd: number; ms: number; tp: number; batch: number } | null = null;
    for (const tp of [1, 2, 4, 8]) for (const b of [1, 8, 32, 64, 128, 256]) {
      const m = pack.evaluate({ weights: "bf16", kv: "bf16", tp, batch: b });
      if (feasible(m) && (best === null || m.objectives.usd_per_mtok! < best.usd)) best = { usd: m.objectives.usd_per_mtok!, ms: m.objectives.ms_per_token!, tp, batch: b };
    }
    this.serveCache.set(key, best);
    return best;
  }
  evaluate(plan: Plan): Metrics {
    const P = this.p, M = this.model(plan), N = params(M), tpp = Number(plan.tokens_per_param), D = tpp * N;
    const law = SCALING_LAWS[P.scalingLaw]!;
    let loss = law.E + law.A / N ** law.alpha + law.B / D ** law.beta;
    loss += P.attentionLossPenalty[String(plan.attention)] ?? 0;
    const dev = P.trainCluster.device;
    const trainFlops = 6 * N * D + 12 * M.layers * P.seqLen * M.hidden * D;
    const gpuHours = trainFlops / (flops(dev, "bf16") * P.trainMfu) / 3600;
    const trainCost = gpuHours * P.trainCluster.costPerDeviceHour;
    const serve = this.serving(M);
    const serveCost = serve ? (serve.usd * P.tokensServed) / 1e6 : Infinity;
    const total = trainCost + serveCost;
    const ratio = M.hidden / M.layers;
    const cons: Record<string, number> = {
      shape: ratio >= 32 && ratio <= 256 ? -1.0 : Math.max(32 - ratio, ratio - 256) / 32,
      servable: serve ? -1.0 : 1.0,
    };
    if (P.budgetUsd) cons.budget = total / P.budgetUsd - 1.0;
    if (P.maxTrainTokens) cons.data = D / P.maxTrainTokens - 1.0;
    const notes = ["attention-type quality penalties and train_mfu are assumptions until measured"];
    if (tpp > 100) notes.push("more than ~100 tokens/parameter: the scaling law is extrapolated here (Sardana et al. 2024)");
    return {
      objectives: { loss, total_cost_usd: serve ? total : 1e30 },
      constraints: cons,
      breakdown: {
        params: N, train_tokens: D, train_cost_usd: trainCost, serve_cost_usd: serve ? serveCost : null, train_gpu_hours: gpuHours,
        serving_plan: serve, kv_heads: M.kvHeads, serve_share: serve && total > 0 ? serveCost / total : null,
        cost_parts_musd: { training: trainCost / 1e6, serving: serve ? serveCost / 1e6 : 0 },
      },
      notes,
    };
  }
}

export { CLUSTERS };

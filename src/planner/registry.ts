// Build packs from plain settings (mirrors nomo_planner/registry.py and the Python dataclass defaults).
import { CALIBRATED } from "./calibration";
import { CLUSTERS, MODELS } from "./hardware";
import {
  CodesignPack, DEFAULT_ATTENTION_PENALTY, DEFAULT_QUALITY_LOSS, InferencePack, TrainingPack, type Pack,
} from "./packs";

export type DomainId = "llm_training" | "llm_inference" | "arch_codesign";
export type Settings = Record<string, unknown>;

export const DEFAULT_SETTINGS: Record<DomainId, Settings> = {
  llm_training: { model: "llama3-8b", cluster: "h100_nvlink_ib", seq_len: 4096, global_batch_tokens: 4194304, total_tokens: 1e12,
    device_counts: [64, 128, 256, 512], usable_memory: 0.92 },
  llm_inference: { model: "llama3-8b", cluster: "h100_nvlink_ib", prompt_tokens: 2048, output_tokens: 256, max_ms_per_token: 50,
    max_quality_loss: 1.0, usable_memory: 0.9, batch_sizes: [1, 8, 32, 64, 128, 256] },
  arch_codesign: { train_cluster: "h100_nvlink_ib", serve_cluster: "h100_nvlink_ib", tokens_served: 1e12, prompt_tokens: 1024,
    output_tokens: 256, max_ms_per_token: 50, budget_usd: null, max_train_tokens: null, seq_len: 4096, vocab: 128256,
    scaling_law: "besiroglu2024", train_mfu: 0.4, layers: [16, 24, 32, 40, 48, 64, 80, 96, 120],
    hidden: [2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384], tokens_per_param: [10, 20, 40, 80, 160] },
};

const num = (v: unknown, d: number) => (v === undefined || v === null || v === "" ? d : Number(v));
const opt = (v: unknown) => (v === undefined || v === null || v === "" ? null : Number(v));

export function build(domain: DomainId, s0: Settings): Pack {
  const s = { ...DEFAULT_SETTINGS[domain], ...s0 };
  if (domain === "llm_training") {
    const model = MODELS[String(s.model)];
    const cluster = CLUSTERS[String(s.cluster)];
    if (!model || !cluster) throw new Error("unknown model or hardware");
    const cal = CALIBRATED[cluster.key];
    const pack = new TrainingPack({ model, cluster, seqLen: num(s.seq_len, 4096), globalBatchTokens: num(s.global_batch_tokens, 4194304),
      totalTokens: num(s.total_tokens, 1e12), deviceCounts: (s.device_counts as number[]).map(Number), usableMemory: num(s.usable_memory, 0.92) },
      cal ? cal.params : undefined);
    pack.calibration = cal ? { observations: cal.observations, looMapePct: cal.looMapePct, source: cal.source } : null;
    return pack;
  }
  if (domain === "llm_inference") {
    const model = MODELS[String(s.model)];
    const cluster = CLUSTERS[String(s.cluster)];
    if (!model || !cluster) throw new Error("unknown model or hardware");
    return new InferencePack({ model, cluster, promptTokens: num(s.prompt_tokens, 2048), outputTokens: num(s.output_tokens, 256),
      maxMsPerToken: opt(s.max_ms_per_token), maxQualityLoss: num(s.max_quality_loss, 1), usableMemory: num(s.usable_memory, 0.9),
      batchSizes: (s.batch_sizes as number[]).map(Number), qualityLoss: (s.quality_loss as typeof DEFAULT_QUALITY_LOSS) ?? DEFAULT_QUALITY_LOSS });
  }
  const tc = CLUSTERS[String(s.train_cluster)], sc = CLUSTERS[String(s.serve_cluster)];
  if (!tc || !sc) throw new Error("unknown hardware");
  return new CodesignPack({ trainCluster: tc, serveCluster: sc, tokensServed: num(s.tokens_served, 1e12), promptTokens: num(s.prompt_tokens, 1024),
    outputTokens: num(s.output_tokens, 256), maxMsPerToken: opt(s.max_ms_per_token), budgetUsd: opt(s.budget_usd), maxTrainTokens: opt(s.max_train_tokens),
    seqLen: num(s.seq_len, 4096), vocab: num(s.vocab, 128256), scalingLaw: String(s.scaling_law), trainMfu: num(s.train_mfu, 0.4),
    attentionLossPenalty: (s.attention_loss_penalty as Record<string, number>) ?? DEFAULT_ATTENTION_PENALTY,
    layers: (s.layers as number[]).map(Number), hidden: (s.hidden as number[]).map(Number), tokensPerParam: (s.tokens_per_param as number[]).map(Number) });
}

/** Python-style problem dict (golden files) -> UI settings. */
export function fromPythonProblem(domain: DomainId, p: Record<string, unknown>): Settings {
  const s: Settings = { ...p };
  if (domain === "llm_training" && p.device_counts) s.device_counts = p.device_counts;
  return s;
}

/** Fix some decisions (the generic form of layer locks): every search, explanation and what-if respects them. */
export function withLocks(pack: Pack, locks: Record<string, string | number>): Pack {
  const keys = Object.keys(locks);
  if (!keys.length) return pack;
  return {
    name: pack.name, description: pack.description, calibration: pack.calibration,
    variables: () => pack.variables().map((v) => (v.name in locks && v.choices.includes(locks[v.name]!) ? { ...v, choices: [locks[v.name]!] } : v)),
    objectives: () => pack.objectives(),
    evaluate: (p) => pack.evaluate(p),
    describe: (p) => pack.describe(p),
    // keep access to the concrete pack (anatomy, exports)
    ...({ inner: pack } as object),
  } as Pack;
}

export function innerPack(pack: Pack): Pack {
  return ((pack as unknown as { inner?: Pack }).inner) ?? pack;
}

/**
 * Browser port of nomo_planner/serving.py. It consumes the same LayerGraph
 * used by training and searches per-node weight and attention KV precision.
 * Throughput, quality, and bandwidth values remain explicit assumptions.
 */

import type { LayerGraph } from "./layers";

export type ServingPrecision = "bf16" | "fp8";
export type ServingLocks = Record<string, Partial<{ weightPrecision: ServingPrecision; kvPrecision: ServingPrecision; precision: ServingPrecision; kv: ServingPrecision }>>;

export interface ServingPlan {
  weightPrecision: ServingPrecision[];
  kvPrecision: ServingPrecision[];
}

export interface ServingHardware {
  devices: number;
  memoryBytes: number;
  peakFlops: number;
  memoryBandwidthBytesS: number;
  costPerDeviceHour: number;
  usableMemory: number;
}

export interface ServingProblem {
  graph: LayerGraph;
  hardware?: Partial<ServingHardware>;
  maxQualityPenaltyPct?: number;
  maxCandidates?: number;
  seed?: number;
}

export interface ServingMetrics {
  objectives: {
    latencyS: number;
    costUsdPerRequest: number;
    memoryBytesPerDevice: number;
    throughputRequestsS: number;
    weightBytes: number;
    kvBytes: number;
  };
  constraints: Record<string, number>;
  memoryByDevice: number[];
  qualityPenaltyPct: number;
  notes: string[];
}

export interface ServingSearchResult {
  graph: LayerGraph;
  best: ServingPlan | null;
  bestMetrics: ServingMetrics | null;
  baseline: { plan: ServingPlan; metrics: ServingMetrics };
  globalBest: { plan: ServingPlan; metrics: ServingMetrics };
  perLayerGainPct: number;
  evaluated: number;
  exhaustive: boolean;
  seed: number;
  assumptions: string[];
}

const BYTES: Record<ServingPrecision, number> = { bf16: 2, fp8: 1 };
const SPEEDUP: Record<ServingPrecision, number> = { bf16: 1, fp8: 1.35 };

function hardware(problem: ServingProblem): ServingHardware {
  return {
    devices: 1, memoryBytes: 80e9, peakFlops: 989e12, memoryBandwidthBytesS: 2e12,
    costPerDeviceHour: 3.5, usableMemory: 0.9, ...(problem.hardware ?? {}),
  };
}

function planKey(plan: ServingPlan): string {
  return JSON.stringify([plan.weightPrecision, plan.kvPrecision]);
}

export function repairServingPlan(graph: LayerGraph, input: ServingPlan, locks: ServingLocks = {}): ServingPlan {
  if (input.weightPrecision.length !== graph.nodes.length || input.kvPrecision.length !== graph.nodes.length) {
    throw new Error("serving plan arrays must match graph node count");
  }
  const weightPrecision = input.weightPrecision.map((value) => value === "fp8" ? "fp8" : "bf16" as ServingPrecision);
  const kvPrecision = input.kvPrecision.map((value) => value === "fp8" ? "fp8" : "bf16" as ServingPrecision);
  graph.nodes.forEach((node, index) => {
    if (node.kind !== "attention") kvPrecision[index] = "bf16";
    const lock = locks[node.id] ?? {};
    const weight = lock.weightPrecision ?? lock.precision;
    const kv = lock.kvPrecision ?? lock.kv;
    if (weight) weightPrecision[index] = weight;
    if (kv && node.kind === "attention") kvPrecision[index] = kv;
  });
  return { weightPrecision, kvPrecision };
}

export function evaluateServingPlan(problem: ServingProblem, rawPlan: ServingPlan, locks: ServingLocks = {}): ServingMetrics {
  const graph = problem.graph;
  const plan = repairServingPlan(graph, rawPlan, locks);
  const hw = hardware(problem);
  const devices = Math.max(1, Math.floor(hw.devices));
  const capacity = hw.memoryBytes * hw.usableMemory;
  const weightBytes = graph.nodes.reduce((sum, node, index) => sum + node.parameterCount * BYTES[plan.weightPrecision[index]], 0);
  const kvBytes = graph.nodes.reduce((sum, node, index) => sum + node.kvCacheBytes * BYTES[plan.kvPrecision[index]], 0);
  const memory = weightBytes / devices + kvBytes + hw.memoryBytes * 0.02;
  const compute = graph.nodes.reduce((sum, node, index) => sum + node.forwardFlops / (hw.peakFlops * SPEEDUP[plan.weightPrecision[index]] * devices), 0);
  const transfer = (weightBytes / devices + kvBytes) / hw.memoryBandwidthBytesS;
  const latency = compute + transfer;
  const cost = latency / 3600 * devices * hw.costPerDeviceHour;
  const fp8Params = graph.nodes.reduce((sum, node, index) => sum + (plan.weightPrecision[index] === "fp8" ? node.parameterCount : 0), 0);
  const weightPenalty = fp8Params / Math.max(1, graph.parameterCount) * 0.35;
  const attention = graph.nodes.map((node, index) => node.kind === "attention" ? index : -1).filter((index) => index >= 0);
  const kvFp8 = attention.filter((index) => plan.kvPrecision[index] === "fp8").length;
  const qualityPenalty = weightPenalty + kvFp8 / Math.max(1, attention.length) * 0.2;
  const maxQuality = problem.maxQualityPenaltyPct ?? 1;
  return {
    objectives: {
      latencyS: latency, costUsdPerRequest: cost, memoryBytesPerDevice: memory,
      throughputRequestsS: latency > 0 ? 1 / latency : 0, weightBytes, kvBytes,
    },
    constraints: { memory: memory / capacity - 1, quality: qualityPenalty / Math.max(maxQuality, 1e-12) - 1 },
    memoryByDevice: Array.from({ length: devices }, () => memory), qualityPenaltyPct: qualityPenalty,
    notes: ["FP8 throughput and quality penalty are assumptions until customer evaluation", "weights are tensor-parallel sharded; KV cache is conservatively replicated", "latency uses a roofline-style compute plus memory-transfer estimate"],
  };
}

function feasible(metrics: ServingMetrics): boolean {
  return Object.values(metrics.constraints).every((value) => value <= 0);
}

function dedup(plans: ServingPlan[]): ServingPlan[] {
  const seen = new Set<string>();
  return plans.filter((plan) => {
    const key = planKey(plan);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function candidates(problem: ServingProblem, locks: ServingLocks): { plans: ServingPlan[]; exhaustive: boolean } {
  const { graph } = problem;
  const attention = graph.nodes.map((node, index) => node.kind === "attention" ? index : -1).filter((index) => index >= 0);
  const plans: ServingPlan[] = [];
  const exhaustive = graph.nodes.length <= 6;
  if (exhaustive) {
    // Tiny graphs use a complete binary search, matching the Python reference.
    const total = 1 << graph.nodes.length;
    const kvTotal = 1 << attention.length;
    for (let weightMask = 0; weightMask < total; weightMask += 1) {
      for (let kvMask = 0; kvMask < kvTotal; kvMask += 1) {
        const weight = graph.nodes.map((_node, index) => weightMask & (1 << index) ? "fp8" : "bf16" as ServingPrecision);
        const kv = graph.nodes.map(() => "bf16" as ServingPrecision);
        attention.forEach((index, position) => { kv[index] = kvMask & (1 << position) ? "fp8" : "bf16"; });
        plans.push(repairServingPlan(graph, { weightPrecision: weight, kvPrecision: kv }, locks));
        if (plans.length >= (problem.maxCandidates ?? 20000)) return { plans: dedup(plans), exhaustive: false };
      }
    }
  } else {
    for (const weightValue of ["bf16", "fp8"] as ServingPrecision[]) {
      for (const kvValue of ["bf16", "fp8"] as ServingPrecision[]) {
        plans.push(repairServingPlan(graph, {
          weightPrecision: graph.nodes.map(() => weightValue),
          kvPrecision: graph.nodes.map((node) => node.kind === "attention" ? kvValue : "bf16"),
        }, locks));
      }
    }
    for (const base of [...plans]) {
      for (let index = 0; index < graph.nodes.length; index += 1) {
        const weight = [...base.weightPrecision];
        weight[index] = weight[index] === "bf16" ? "fp8" : "bf16";
        plans.push(repairServingPlan(graph, { weightPrecision: weight, kvPrecision: base.kvPrecision }, locks));
        if (graph.nodes[index].kind === "attention") {
          const kv = [...base.kvPrecision];
          kv[index] = kv[index] === "bf16" ? "fp8" : "bf16";
          plans.push(repairServingPlan(graph, { weightPrecision: base.weightPrecision, kvPrecision: kv }, locks));
        }
        if (plans.length >= (problem.maxCandidates ?? 20000)) return { plans: dedup(plans), exhaustive: false };
      }
    }
  }
  return { plans: dedup(plans), exhaustive };
}

function globalPlan(problem: ServingProblem, weightValue: ServingPrecision, kvValue: ServingPrecision, locks: ServingLocks): { plan: ServingPlan; metrics: ServingMetrics } {
  const { graph } = problem;
  const plan = repairServingPlan(graph, {
    weightPrecision: graph.nodes.map(() => weightValue),
    kvPrecision: graph.nodes.map((node) => node.kind === "attention" ? kvValue : "bf16"),
  }, locks);
  return { plan, metrics: evaluateServingPlan(problem, plan, locks) };
}

export function searchServing(problem: ServingProblem, locks: ServingLocks = {}): ServingSearchResult {
  const seed = problem.seed ?? 20261003;
  const assumptions = ["Serving recommendations use the uploaded shared model graph", "FP8 quality penalty is a bounded assumption, not a customer measurement", "Provide calibrated GPU bandwidth/throughput before production capacity commitments"];
  const baseline = globalPlan(problem, "bf16", "bf16", locks);
  const globals = (["bf16", "fp8"] as ServingPrecision[]).flatMap((weight) => (["bf16", "fp8"] as ServingPrecision[]).map((kv) => globalPlan(problem, weight, kv, locks)));
  const feasibleGlobals = globals.filter(({ metrics }) => feasible(metrics));
  const globalBest = [...(feasibleGlobals.length ? feasibleGlobals : globals)].sort((left, right) => left.metrics.objectives.latencyS - right.metrics.objectives.latencyS || left.metrics.objectives.costUsdPerRequest - right.metrics.objectives.costUsdPerRequest)[0]!;
  const { plans, exhaustive } = candidates(problem, locks);
  const feasiblePlans = plans.map((plan) => ({ plan, metrics: evaluateServingPlan(problem, plan, locks) })).filter(({ metrics }) => feasible(metrics));
  const best = [...feasiblePlans].sort((left, right) => left.metrics.objectives.latencyS - right.metrics.objectives.latencyS || left.metrics.objectives.costUsdPerRequest - right.metrics.objectives.costUsdPerRequest)[0] ?? null;
  const perLayerGainPct = best ? Math.max(0, (globalBest.metrics.objectives.latencyS / best.metrics.objectives.latencyS - 1) * 100) : 0;
  return { graph: problem.graph, best: best?.plan ?? null, bestMetrics: best?.metrics ?? null, baseline, globalBest, perLayerGainPct, evaluated: plans.length, exhaustive, seed, assumptions };
}

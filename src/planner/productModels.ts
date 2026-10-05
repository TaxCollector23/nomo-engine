/** Browser port of nomo_planner/product_models.py. All values are labelled
 * estimates or user inputs in the Product Studio; no market data is hidden. */

export interface ChipSpec { name: string; memoryBandwidthGbps: number; sramGb: number; bf16Tops: number; fp8Tops: number; interconnectGbps: number; neuromorphicCores: number; neuromorphicMemoryGb: number; costUsd: number; powerW: number; hourlyCostUsd: number }
export interface Workload { name: string; denseFlops: number; memoryBytes: number; spikingOps?: number; requiresNeuromorphic?: boolean }
export interface FeasibleChipEvaluation { feasible: true; seconds: number; energyJ: number; costUsd: number; bottleneck: string; mapping: { densePrecision: string; bottleneck: string; neuromorphicCores: number }; notes: string[] }
type ChipEvaluation = FeasibleChipEvaluation | { feasible: false; reason: string };

export function evaluateChip(spec: ChipSpec, workload: Workload, precision: "bf16" | "fp8" = "bf16"): ChipEvaluation {
  const tops = precision === "fp8" ? spec.fp8Tops : spec.bf16Tops;
  if (Math.min(spec.memoryBandwidthGbps, spec.sramGb, tops, spec.powerW, spec.hourlyCostUsd) <= 0) throw new Error("chip specs must be positive");
  if (workload.denseFlops < 0 || workload.memoryBytes < 0 || (workload.spikingOps ?? 0) < 0) throw new Error("workload terms cannot be negative");
  if (workload.requiresNeuromorphic && (spec.neuromorphicCores <= 0 || spec.neuromorphicMemoryGb <= 0)) return { feasible: false, reason: "workload requires neuromorphic cores and memory" };
  const computeS = workload.denseFlops / (tops * 1e12), memoryS = workload.memoryBytes / (spec.memoryBandwidthGbps * 1e9), spikeS = (workload.spikingOps ?? 0) / Math.max(1, spec.neuromorphicCores * 1e9);
  return { feasible: true, seconds: Math.max(computeS, memoryS, spikeS), energyJ: Math.max(computeS, memoryS, spikeS) * spec.powerW, costUsd: Math.max(computeS, memoryS, spikeS) / 3600 * spec.hourlyCostUsd, bottleneck: computeS >= memoryS && computeS >= spikeS ? "compute" : memoryS >= spikeS ? "memory" : "neuromorphic", mapping: { densePrecision: precision, bottleneck: computeS >= memoryS && computeS >= spikeS ? "compute" : memoryS >= spikeS ? "memory" : "neuromorphic", neuromorphicCores: workload.spikingOps ? spec.neuromorphicCores : 0 }, notes: ["chip bandwidth, throughput, power, and price are supplied estimates or customer inputs"] };
}

export function paretoChips(specs: ChipSpec[], workloads: Workload[], precision: "bf16" | "fp8" = "bf16") {
  const rows = specs.flatMap((spec) => {
    const evaluations = workloads.map((workload) => evaluateChip(spec, workload, precision));
    if (evaluations.some((evaluation) => !evaluation.feasible)) return [];
    const feasible = evaluations as FeasibleChipEvaluation[];
    return [{ chip: spec.name, spec, workloads: feasible, totalSeconds: feasible.reduce((sum, evaluation) => sum + evaluation.seconds, 0), totalCostUsd: feasible.reduce((sum, evaluation) => sum + evaluation.costUsd, 0), totalEnergyJ: feasible.reduce((sum, evaluation) => sum + evaluation.energyJ, 0) }];
  });
  return rows.filter((row) => !rows.some((other) => other !== row && other.totalSeconds <= row.totalSeconds && other.totalCostUsd <= row.totalCostUsd && (other.totalSeconds < row.totalSeconds || other.totalCostUsd < row.totalCostUsd))).sort((a, b) => a.totalCostUsd - b.totalCostUsd || a.totalSeconds - b.totalSeconds);
}

export interface ReliabilityProblem { failureRatePerHour: number; checkpointWriteHours: number; restartHours: number; gpuCount: number; gpuCostPerHour: number }
export function evaluateReliability(problem: ReliabilityProblem, checkpointIntervalHours: number) {
  if (Math.min(problem.failureRatePerHour, problem.checkpointWriteHours, problem.restartHours, checkpointIntervalHours) <= 0) throw new Error("reliability inputs must be positive");
  const checkpointOverhead = problem.checkpointWriteHours / checkpointIntervalHours, lostWork = problem.failureRatePerHour * checkpointIntervalHours / 2, restartOverhead = problem.failureRatePerHour * problem.restartHours;
  const goodput = 1 / (1 + checkpointOverhead + lostWork + restartOverhead);
  return { checkpointIntervalHours, goodput, costUsdPerUsefulHour: problem.gpuCount * problem.gpuCostPerHour / goodput };
}
export function optimizeReliability(problem: ReliabilityProblem, intervals = [0.25, 0.5, 1, 2, 4, 8, 12, 24]) {
  const rows = intervals.map((interval) => evaluateReliability(problem, interval));
  const best = [...rows].sort((a, b) => b.goodput - a.goodput)[0]!;
  return { ...best, youngIntervalHours: Math.sqrt(2 * problem.checkpointWriteHours / problem.failureRatePerHour) };
}

export interface TrafficHour { hour: number; arrivalRps: number; serviceRpsPerReplica: number; p99TargetMs: number }
export function sizeFleet(traffic: TrafficHour[], targetUtilization = 0.7, replicaHourlyCost = 1) {
  if (!(targetUtilization > 0 && targetUtilization < 1)) throw new Error("target utilization must be between 0 and 1");
  return traffic.map((row) => {
    const replicas = Math.max(1, Math.ceil(row.arrivalRps / (row.serviceRpsPerReplica * targetUtilization)));
    const spare = replicas * row.serviceRpsPerReplica - row.arrivalRps;
    const p99 = spare <= 0 ? Infinity : 1000 * Math.log(100) / spare;
    return { hour: row.hour, replicas, approxP99Ms: p99, hourlyCostUsd: replicas * replicaHourlyCost, withinTarget: p99 <= row.p99TargetMs, arrivalRps: row.arrivalRps };
  });
}

export interface FineTuneProblem { parameterCount: number; trainingTokens: number; memoryBytes: number; peakTflops: number; gpuMemoryBytes: number; gpuCostPerHour: number; qualityLoss: Record<string, number> }
export function fineTuneOptions(problem: FineTuneProblem) {
  const modes: Record<string, [number, number]> = { full: [16, 1], lora: [0.08, 0.35], qlora: [0.04, 0.25] };
  return Object.entries(modes).map(([mode, [memoryFactor, computeFactor]]) => { const memoryBytes = problem.memoryBytes * memoryFactor; const timeHours = 6 * problem.parameterCount * problem.trainingTokens / (problem.peakTflops * 1e12 * computeFactor) / 3600; return { mode, memoryBytes, timeHours, costUsd: timeHours * problem.gpuCostPerHour, qualityLoss: problem.qualityLoss[mode] ?? NaN, feasible: memoryBytes <= problem.gpuMemoryBytes, notes: "quality is a customer-supplied assumption" }; });
}

export interface TcoOption { name: string; purchaseUsd: number; leaseUsdPerHour: number; utilization: number; years: number; hoursPerYear: number; energyUsdPerHour: number; maintenancePct: number; priceLowUsd?: number; priceHighUsd?: number }
export function evaluateTco(option: TcoOption) {
  if (Math.min(option.purchaseUsd, option.leaseUsdPerHour, option.utilization, option.years, option.hoursPerYear) <= 0 || option.utilization > 1) throw new Error("TCO inputs must be positive and utilization must be in (0, 1]");
  const usefulHours = option.years * option.hoursPerYear * option.utilization;
  const maintenance = option.purchaseUsd * option.maintenancePct / usefulHours;
  const buy = option.purchaseUsd / usefulHours + maintenance + option.energyUsdPerHour / option.utilization;
  const rent = option.leaseUsdPerHour / option.utilization + option.energyUsdPerHour / option.utilization;
  const low = (option.priceLowUsd ?? option.purchaseUsd) / usefulHours + maintenance + option.energyUsdPerHour / option.utilization;
  const high = (option.priceHighUsd ?? option.purchaseUsd) / usefulHours + maintenance + option.energyUsdPerHour / option.utilization;
  return { name: option.name, buyUsdPerUsefulHour: buy, rentUsdPerUsefulHour: rent, rangeLow: Math.min(low, high), rangeHigh: Math.max(low, high), recommended: buy <= rent ? "buy" : "rent", notes: "prices, utilization, depreciation, energy, and maintenance are user-entered inputs" };
}

export interface RLProblem { answerLengthP50: number; answerLengthP95: number; rolloutTokensPerSPerGpu: number; trainTokensPerSPerGpu: number; rewardTokensPerSPerGpu: number; gpuCostPerHour: number; trainingTokensPerUpdate: number; targetSamplesPerHour: number }
export function evaluateRlPlan(problem: RLProblem, options: { rolloutGpus: number; trainGpus: number; rewardGpus: number; colocated: boolean; asynchronous: boolean; batchSize: number; kvPrecision: "bf16" | "fp8" }) {
  if (problem.answerLengthP50 <= 0 || problem.answerLengthP95 < problem.answerLengthP50) throw new Error("answer-length p50/p95 are required customer inputs");
  const factor = options.kvPrecision === "fp8" ? 1.15 : 1;
  const rolloutS = problem.answerLengthP95 / (problem.rolloutTokensPerSPerGpu * options.rolloutGpus * factor), rewardS = problem.answerLengthP95 / (problem.rewardTokensPerSPerGpu * options.rewardGpus), trainS = problem.trainingTokensPerUpdate / (problem.trainTokensPerSPerGpu * options.trainGpus * options.batchSize), syncS = options.asynchronous ? 0 : 0.05 * Math.max(rolloutS, trainS);
  const stepSeconds = rolloutS + trainS + rewardS + syncS, activeGpus = options.colocated ? Math.max(options.rolloutGpus, options.trainGpus, options.rewardGpus) : options.rolloutGpus + options.trainGpus + options.rewardGpus;
  return { samplesPerHour: 3600 * options.batchSize / stepSeconds, costUsdPerUpdate: stepSeconds / 3600 * activeGpus * problem.gpuCostPerHour, stepSeconds, schedule: options.asynchronous ? "async" : "sync", colocation: options.colocated ? "colocated" : "separate", feasible: 3600 * options.batchSize / stepSeconds >= problem.targetSamplesPerHour };
}

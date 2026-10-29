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
/** Fleet sizing with a real queueing tail: exact M/M/1 sojourn p99 P(T>t)=exp(-(mu-lambda)t).
 * The historical 1000*ln(100)/spare understated p99 by roughly the replica count. With a
 * service-time CV > 1 the Pollaczek-Khinchine (Kingman) mean-wait excess is added; CV = 1
 * reproduces M/M/1 exactly. */
export function sizeFleet(traffic: TrafficHour[], targetUtilization = 0.7, replicaHourlyCost = 1, serviceCv = 1) {
  if (!(targetUtilization > 0 && targetUtilization < 1)) throw new Error("target utilization must be between 0 and 1");
  if (!(serviceCv > 0)) throw new Error("service coefficient of variation must be positive");
  return traffic.map((row) => {
    const replicas = Math.max(1, Math.ceil(row.arrivalRps / (row.serviceRpsPerReplica * targetUtilization)));
    const perReplicaRps = row.arrivalRps / replicas;
    const mu = row.serviceRpsPerReplica;
    const rho = perReplicaRps / mu;
    let p99: number, queueModel: string;
    if (rho < 1) {
      const mm1P99S = Math.log(100) / (mu - perReplicaRps);
      const meanWaitMg1 = (rho * (1 + serviceCv * serviceCv)) / (2 * mu * (1 - rho));
      const meanWaitMm1 = rho / (mu - perReplicaRps);
      p99 = 1000 * (mm1P99S + Math.max(0, meanWaitMg1 - meanWaitMm1));
      queueModel = serviceCv === 1 ? "M/M/1 sojourn tail" : "M/G/1 tail (Pollaczek-Khinchine excess)";
    } else {
      p99 = Infinity; queueModel = "saturated (rho >= 1)";
    }
    return { hour: row.hour, replicas, approxP99Ms: p99, hourlyCostUsd: replicas * replicaHourlyCost, withinTarget: p99 <= row.p99TargetMs, arrivalRps: row.arrivalRps, perReplicaUtilization: rho, queueModel };
  });
}

export interface FineTuneProblem { parameterCount: number; trainingTokens: number; memoryBytes: number; peakTflops: number; gpuMemoryBytes: number; gpuCostPerHour: number; qualityLoss: Record<string, number>; adapterParamFraction?: number }
/** Fine-tuning from first principles (replaces the invented 0.08/0.35 and 0.04/0.25 factors).
 * Memory: full = 16 B/param training state (2 weights + 2 grads + 4 fp32 master + 8 Adam) plus
 * activations (recovered from the user-supplied full footprint); LoRA = frozen bf16 base
 * (2 B/param) + adapters (adapterParamFraction of params, 16 B/param state) + activations
 * (Hu et al. 2021, arXiv:2106.09685); QLoRA = NF4 base ~0.55 B/param incl. quantization
 * constants (Dettmers et al. 2023, arXiv:2305.14314). Compute: full ~6ND FLOPs/token;
 * LoRA skips the frozen-weight gradient term ~4ND; QLoRA adds an assumed ~10%
 * de-quantization overhead (4ND/0.9). Quality stays a customer-supplied assumption. */
export function fineTuneOptions(problem: FineTuneProblem) {
  const adapterFraction = problem.adapterParamFraction ?? 0.005;
  const modes: Record<string, { memoryBytes: number; flopsPerToken: number; note: string }> = {
    full: { memoryBytes: problem.memoryBytes, flopsPerToken: 6, note: "16 B/param training state (2 weights + 2 grads + 4 master + 8 Adam) plus activations" },
    lora: { memoryBytes: 2 * problem.parameterCount + 16 * problem.parameterCount * adapterFraction + Math.max(0, problem.memoryBytes - 16 * problem.parameterCount), flopsPerToken: 4, note: `frozen bf16 base (2 B/param) + adapters (${adapterFraction} of params, 16 B/param state) + activations; ~4ND FLOPs/token vs 6ND full (Hu et al. 2021, arXiv:2106.09685)` },
    qlora: { memoryBytes: 0.55 * problem.parameterCount + 16 * problem.parameterCount * adapterFraction + Math.max(0, problem.memoryBytes - 16 * problem.parameterCount), flopsPerToken: 4 / 0.9, note: "NF4 base ~0.55 B/param incl. quantization constants (Dettmers et al. 2023, arXiv:2305.14314) + adapter states + activations; ~10% de-quantization overhead assumed" },
  };
  return Object.entries(modes).map(([mode, model]) => { const timeHours = model.flopsPerToken * problem.parameterCount * problem.trainingTokens / (problem.peakTflops * 1e12) / 3600; return { mode, memoryBytes: model.memoryBytes, timeHours, costUsd: timeHours * problem.gpuCostPerHour, qualityLoss: problem.qualityLoss[mode] ?? NaN, feasible: model.memoryBytes <= problem.gpuMemoryBytes, notes: `quality is a customer-supplied assumption; ${model.note}` }; });
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

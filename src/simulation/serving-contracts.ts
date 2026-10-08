import type { DistributionSpec } from "./distribution";
import type { TimelineEvent } from "./timeline";

/** Serving wire contracts mirrored from serving_sim.py. */
export const SIMULATED_LABEL = "simulated" as const;

export interface TraceRequest {
  requestId: string;
  arrivalS: number;
  promptTokens?: number | null;
  answerTokens?: number | null;
  prefixId?: string | null;
  prefixTokens?: number | null;
  metadata: Readonly<Record<string, unknown>>;
}

export interface ServingDistributionSummary {
  name: string;
  values: readonly number[];
  p50: number | null;
  p90: number | null;
  p99: number | null;
  label: typeof SIMULATED_LABEL;
}

export interface ServingRequestResult {
  requestId: string;
  arrivalS: number;
  promptTokens: number;
  answerTokens: number;
  status: string;
  firstTokenS: number | null;
  completionS: number | null;
  ttftS: number | null;
  e2eS: number | null;
  interTokenLatencyS: readonly number[];
  averageItlS: number | null;
  maxItlS: number | null;
  outputTokens: number;
  prefillReplica: string | null;
  decodeReplica: string | null;
  prefixCacheHit: boolean;
  prefixTokens: number;
  preemptions: number;
  kvPeakBlocks: number;
  error: string | null;
  timeline: readonly TimelineEvent[];
  label: typeof SIMULATED_LABEL;
}

export interface ServingSimulationResult {
  seed: number;
  simulated: true;
  assumptions: Readonly<Record<string, unknown>>;
  requests: readonly ServingRequestResult[];
  timeline: readonly TimelineEvent[];
  timelines: Readonly<Record<string, readonly TimelineEvent[]>>;
  distributions: Readonly<Record<string, ServingDistributionSummary>>;
  metrics: Readonly<Record<string, number | null>>;
  metricLabels: Readonly<Record<string, string>>;
  replicaMetrics: Readonly<Record<string, Readonly<Record<string, number | null>>>>;
  notes: readonly string[];
}

export interface ServingAssumptionsContract {
  seed?: number;
  requestCount?: number;
  arrivalRatePerS?: number;
  promptDistribution?: DistributionSpec;
  answerDistribution?: DistributionSpec;
  replicas?: number;
  tensorParallel?: number;
  prefillTokensPerS?: number;
  decodeTokensPerS?: number;
  kvBlockTokens?: number;
  kvCapacityTokens?: number;
  disaggregate?: boolean;
  metadata?: Readonly<Record<string, unknown>>;
}

/**
 * Request-level serving scheduling remains a server-only reference feature in
 * this browser port.  The Worker accepts serialized serving results and trace
 * contracts; it does not claim live capacity or benchmark validation.
 */
export const SERVING_SERVER_BOUNDARY = "server-only serving discrete-event reference" as const;

import type { Provenance } from "./provenance";

export const SIMULATION_CONTRACT_VERSION = 1 as const;

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type DType = "fp32" | "fp16" | "bf16" | "fp8" | "int8" | "int4";
export const DTYPE_BYTES: Readonly<Record<DType, number>> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
};

export type Shape = readonly number[];

export interface TensorSpec {
  id: string;
  shape: Shape;
  dtype: DType;
  bytes: number;
  producerId?: string;
  consumerIds: readonly string[];
  lifetime?: {
    firstOperatorIndex: number;
    lastOperatorIndex: number;
  };
  persistent?: boolean;
  metadata?: Readonly<Record<string, JsonValue>>;
}

export interface MetricValue<T = number> {
  value: T;
  provenance: Provenance;
}

export type ExecutionContext = "browser" | "worker" | "server";

export interface ServerOnlyCapability {
  id: string;
  execution: "server";
  browserAvailable: false;
  reason: string;
}

export interface SimulationCapabilities {
  execution: ExecutionContext;
  deterministic: true;
  serverOnly: readonly ServerOnlyCapability[];
}

export interface SimulationRequest {
  contractVersion: typeof SIMULATION_CONTRACT_VERSION;
  requestId: string;
  graph: import("./operator-graph").GraphIR;
  topology: import("./topology").Topology;
  seed?: number | string;
  options?: import("./engine").SimulationOptions;
}

export interface SimulationMetricSet {
  makespanSeconds: number;
  totalFlops: number;
  totalBytes: number;
  peakMemoryBytes: number;
  computeUtilization: number;
}

export interface SimulationResponse {
  contractVersion: typeof SIMULATION_CONTRACT_VERSION;
  requestId: string;
  status: "completed" | "failed" | "cancelled";
  result?: import("./engine").SimulationResult;
  error?: string;
}

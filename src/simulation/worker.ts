import { SIMULATION_CONTRACT_VERSION, type SimulationRequest, type SimulationResponse } from "./contracts";
import { runSimulation, type SimulationInput, type SimulationOptions, type SimulationResult } from "./engine";
import { BROWSER_SIMULATION_CAPABILITIES } from "./server-boundary";

export interface SimulationWorkerRunRequest {
  type: "run";
  requestId: string;
  input: SimulationInput;
  seed?: number | string;
  options?: SimulationOptions;
  contractVersion?: typeof SIMULATION_CONTRACT_VERSION;
}

export interface SimulationWorkerCancelRequest {
  type: "cancel";
  requestId: string;
}

export interface SimulationWorkerPingRequest {
  type: "ping";
  requestId?: string;
}

export type SimulationWorkerRequest = SimulationWorkerRunRequest | SimulationWorkerCancelRequest | SimulationWorkerPingRequest;

export interface SimulationWorkerResult {
  type: "result";
  requestId: string;
  result: SimulationResult;
}

export interface SimulationWorkerError {
  type: "error";
  requestId?: string;
  error: string;
}

export interface SimulationWorkerCancelled {
  type: "cancelled";
  requestId: string;
}

export interface SimulationWorkerPong {
  type: "pong";
  requestId?: string;
  contractVersion: typeof SIMULATION_CONTRACT_VERSION;
  capabilities: typeof BROWSER_SIMULATION_CAPABILITIES;
}

export type SimulationWorkerResponse = SimulationWorkerResult | SimulationWorkerError | SimulationWorkerCancelled | SimulationWorkerPong;

export interface WorkerMessagePortLike {
  postMessage(message: SimulationWorkerResponse): void;
}

export interface WorkerHandlerState {
  cancelledRequestIds: Set<string>;
}

export function createWorkerHandler(
  port: WorkerMessagePortLike,
  state: WorkerHandlerState = { cancelledRequestIds: new Set<string>() },
): (request: SimulationWorkerRequest) => void {
  return (request) => {
    if (request.type === "ping") {
      port.postMessage({ type: "pong", requestId: request.requestId, contractVersion: SIMULATION_CONTRACT_VERSION, capabilities: BROWSER_SIMULATION_CAPABILITIES });
      return;
    }
    if (request.type === "cancel") {
      state.cancelledRequestIds.add(request.requestId);
      port.postMessage({ type: "cancelled", requestId: request.requestId });
      return;
    }
    const requestId = request.requestId;
    if (state.cancelledRequestIds.has(requestId)) {
      port.postMessage({ type: "cancelled", requestId });
      return;
    }
    try {
      if (request.contractVersion !== undefined && request.contractVersion !== SIMULATION_CONTRACT_VERSION) {
        throw new Error(`unsupported simulation contract version: ${request.contractVersion}`);
      }
      const result = runSimulation(request.input, { ...request.options, seed: request.seed ?? request.options?.seed });
      if (state.cancelledRequestIds.has(requestId)) {
        port.postMessage({ type: "cancelled", requestId });
        return;
      }
      port.postMessage({ type: "result", requestId, result });
    } catch (error) {
      port.postMessage({ type: "error", requestId, error: error instanceof Error ? error.message : String(error) });
    }
  };
}

export function handleWorkerRequest(
  request: SimulationWorkerRequest,
  postMessage: (response: SimulationWorkerResponse) => void,
  state?: WorkerHandlerState,
): void {
  createWorkerHandler({ postMessage }, state)(request);
}

export function isSimulationWorkerRequest(value: unknown): value is SimulationWorkerRequest {
  if (typeof value !== "object" || value === null) return false;
  const request = value as Record<string, unknown>;
  return request.type === "run" || request.type === "cancel" || request.type === "ping";
}

export function installSimulationWorker(): void {
  const runtime = globalThis as typeof globalThis & {
    addEventListener?: (type: "message", listener: (event: MessageEvent<unknown>) => void) => void;
    postMessage?: (message: SimulationWorkerResponse) => void;
  };
  if (typeof runtime.addEventListener !== "function" || typeof runtime.postMessage !== "function") return;
  const handler = createWorkerHandler({ postMessage: runtime.postMessage.bind(runtime) });
  runtime.addEventListener("message", (event) => {
    if (isSimulationWorkerRequest(event.data)) handler(event.data);
  });
}

/** The browser entry is safe to import in tests; installation only happens in a Worker-like global. */
if (typeof document === "undefined") installSimulationWorker();

export function toSimulationRequest(request: SimulationWorkerRunRequest): SimulationRequest {
  return {
    contractVersion: SIMULATION_CONTRACT_VERSION,
    requestId: request.requestId,
    graph: request.input.graph,
    topology: request.input.topology,
    seed: request.seed ?? request.options?.seed,
    options: request.options,
  };
}

export function responseForResult(requestId: string, result: SimulationResult): SimulationResponse {
  return { contractVersion: SIMULATION_CONTRACT_VERSION, requestId, status: "completed", result };
}

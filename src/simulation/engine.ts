import { SIMULATION_CONTRACT_VERSION, type MetricValue, type SimulationMetricSet } from "./contracts";
import { DeterministicRng, Distribution, sampleNumeric, type NumericDistribution } from "./distribution";
import { calculateGraphAccounting, topologicalOrder, type OperatorGraph, type OperatorNode } from "./operator-graph";
import { BROWSER_SIMULATION_CAPABILITIES } from "./server-boundary";
import { createTimeline, createTimelineEvent, type Timeline, type TimelineEvent } from "./timeline";
import { findTopologyPath, type Topology } from "./topology";
import { deriveProvenance, makeProvenance, type Provenance } from "./provenance";

export type DeviceAssignment = "single" | "round-robin";

export interface SimulationInput {
  graph: OperatorGraph;
  topology: Topology;
}

export interface SimulationOptions {
  seed?: number | string;
  deviceAssignment?: DeviceAssignment;
  /** Multipliers are sampled once per operator and remain fixed for the run. */
  durationDistributions?: Readonly<Record<string, NumericDistribution | Distribution>>;
}

export interface SimulationMetrics extends SimulationMetricSet {
  eventCount: number;
  communicationSeconds: number;
}

export interface SimulationResult {
  contractVersion: typeof SIMULATION_CONTRACT_VERSION;
  graphId: string;
  topologyId: string;
  seed: number | string;
  timeline: Timeline;
  metrics: SimulationMetrics;
  draws: Readonly<Record<string, number>>;
  metricProvenance: Readonly<Record<keyof SimulationMetrics, Provenance>>;
  capabilities: typeof BROWSER_SIMULATION_CAPABILITIES;
  provenance: Provenance;
}

interface ScheduleState {
  deviceAvailable: Map<string, number>;
  deviceLastEvent: Map<string, string | undefined>;
  memoryByDevice: Map<string, number>;
  eventByOperator: Map<string, TimelineEvent>;
  eventById: Map<string, TimelineEvent>;
  linkLastEvent: Map<string, string | undefined>;
}

function deviceIdForOperator(operator: OperatorNode, index: number, deviceIds: readonly string[], assignment: DeviceAssignment): string {
  const attributes = operator.attributes as Record<string, unknown>;
  const explicit = attributes.deviceId ?? attributes.device_id;
  if (typeof explicit === "string" && deviceIds.includes(explicit)) return explicit;
  return assignment === "single" ? deviceIds[0] : deviceIds[index % deviceIds.length];
}

function distributionForOperator(operator: OperatorNode, options: SimulationOptions): NumericDistribution | Distribution | undefined {
  const distributions = options.durationDistributions ?? {};
  return distributions[operator.id] ?? distributions[operator.phase] ?? distributions.default;
}

function durationForOperator(operator: OperatorNode, device: { computeFlops: number; memoryBandwidthBytesPerSecond: number }, multiplier: number): number {
  const computeSeconds = device.computeFlops > 0 ? operator.cost.flops / device.computeFlops : 0;
  const memorySeconds = device.memoryBandwidthBytesPerSecond > 0 ? (operator.cost.bytesRead + operator.cost.bytesWritten) / device.memoryBandwidthBytesPerSecond : 0;
  return Math.max(computeSeconds, memorySeconds) * multiplier;
}

function addDependency(dependencies: string[], id: string | undefined): void {
  if (id !== undefined && !dependencies.includes(id)) dependencies.push(id);
}

function routeKey(source: string, target: string, linkIds: readonly string[]): string {
  return `${source}->${target}:${linkIds.join(",")}`;
}

function initialMemory(graph: OperatorGraph, deviceCount: number): number {
  return graph.accounting.parameterBytes / Math.max(1, deviceCount);
}

function makeMetricProvenance(graph: OperatorGraph, topology: Topology): Readonly<Record<keyof SimulationMetrics, Provenance>> {
  const provenance = deriveProvenance([graph.provenance, topology.provenance], "deterministic browser simulation result");
  return {
    makespanSeconds: provenance,
    totalFlops: deriveProvenance([graph.provenance], "operator graph accounting"),
    totalBytes: deriveProvenance([graph.provenance, topology.provenance], "operator and communication accounting"),
    peakMemoryBytes: deriveProvenance([graph.provenance], "tensor lifetime accounting"),
    computeUtilization: provenance,
    eventCount: provenance,
    communicationSeconds: deriveProvenance([topology.provenance], "topology link accounting"),
  };
}

function emptyState(topology: Topology, graph: OperatorGraph): ScheduleState {
  const baseMemory = initialMemory(graph, topology.devices.length);
  return {
    deviceAvailable: new Map(topology.devices.map((device) => [device.id, 0])),
    deviceLastEvent: new Map(topology.devices.map((device) => [device.id, undefined])),
    memoryByDevice: new Map(topology.devices.map((device) => [device.id, baseMemory])),
    eventByOperator: new Map(),
    eventById: new Map(),
    linkLastEvent: new Map(),
  };
}

function scheduleGraph(input: SimulationInput, options: SimulationOptions, rng: DeterministicRng): { events: TimelineEvent[]; draws: Record<string, number> } {
  const { graph, topology } = input;
  const orderedOperators = topologicalOrder(graph);
  const deviceIds = topology.devices.map((device) => device.id);
  if (!deviceIds.length) throw new Error("simulation topology needs at least one device");
  const state = emptyState(topology, graph);
  const events: TimelineEvent[] = [];
  const draws: Record<string, number> = {};
  const tensorById = new Map(graph.tensors.map((tensor) => [tensor.id, tensor]));
  const assignment = options.deviceAssignment ?? "round-robin";

  orderedOperators.forEach((operator, operatorIndex) => {
    const deviceId = deviceIdForOperator(operator, operatorIndex, deviceIds, assignment);
    const device = topology.devices.find((candidate) => candidate.id === deviceId);
    if (!device) throw new Error(`operator ${operator.id} assigned to unknown device ${deviceId}`);
    const dependencies: string[] = [];
    let startTime = state.deviceAvailable.get(deviceId) ?? 0;
    addDependency(dependencies, state.deviceLastEvent.get(deviceId));

    for (const edge of graph.edges.filter((candidate) => candidate.target === operator.id)) {
      const sourceEvent = state.eventByOperator.get(edge.source);
      if (!sourceEvent) throw new Error(`operator ${operator.id} is scheduled before dependency ${edge.source}`);
      const sourceDeviceId = sourceEvent.deviceId;
      addDependency(dependencies, sourceEvent.id);
      startTime = Math.max(startTime, sourceEvent.endTime);
      if (sourceDeviceId !== deviceId) {
        const path = findTopologyPath(topology, sourceDeviceId, deviceId, edge.bytes);
        const key = routeKey(sourceDeviceId, deviceId, path.linkIds);
        const priorCommunication = state.linkLastEvent.get(key);
        const communicationDependencies: string[] = [];
        addDependency(communicationDependencies, sourceEvent.id);
        addDependency(communicationDependencies, priorCommunication);
        const communicationStart = Math.max(sourceEvent.endTime, priorCommunication ? state.eventById.get(priorCommunication)?.endTime ?? 0 : 0, state.deviceAvailable.get(sourceDeviceId) ?? 0);
        const communication = createTimelineEvent({
          id: `communication.${edge.id}`,
          type: "communication",
          phase: "communication",
          deviceId: sourceDeviceId,
          streamId: "communication",
          startTime: communicationStart,
          durationSeconds: path.durationSeconds,
          dependencyIds: communicationDependencies,
          bytes: edge.bytes,
          metadata: { targetDeviceId: deviceId, path: path.linkIds, edgeId: edge.id },
          provenance: deriveProvenance([edge.provenance, topology.provenance], "cross-device tensor transfer"),
        });
        events.push(communication);
        state.eventById.set(communication.id, communication);
        state.linkLastEvent.set(key, communication.id);
        state.deviceAvailable.set(sourceDeviceId, communication.endTime);
        state.deviceLastEvent.set(sourceDeviceId, communication.id);
        addDependency(dependencies, communication.id);
        startTime = Math.max(startTime, communication.endTime);
      }
    }

    const distribution = distributionForOperator(operator, options);
    const multiplier = distribution === undefined ? 1 : distribution instanceof Distribution ? distribution.sample(rng.fork(operator.id)) : sampleNumeric(distribution, rng.fork(operator.id));
    if (!Number.isFinite(multiplier) || multiplier <= 0) throw new Error(`duration multiplier for ${operator.id} must be positive`);
    if (distribution !== undefined) draws[operator.id] = multiplier;
    const durationSeconds = durationForOperator(operator, device, multiplier);
    const beforeMemory = state.memoryByDevice.get(deviceId) ?? initialMemory(graph, topology.devices.length);
    const outputBytes = operator.outputs.reduce((sum, tensorId) => {
      const tensor = tensorById.get(tensorId);
      return tensor?.persistent ? sum : sum + (tensor?.bytes ?? 0);
    }, 0);
    const releaseBytes = operator.inputs.reduce((sum, tensorId) => {
      const tensor = tensorById.get(tensorId);
      const last = tensor?.lifetime?.lastOperatorIndex;
      return last !== undefined && last <= operatorIndex && !tensor?.persistent ? sum + (tensor?.bytes ?? 0) : sum;
    }, 0);
    const afterMemory = Math.max(initialMemory(graph, topology.devices.length), beforeMemory + outputBytes + operator.cost.workspaceBytes - releaseBytes);
    const event = createTimelineEvent({
      id: `compute.${operator.id}`,
      type: "compute",
      phase: operator.phase,
      operatorId: operator.id,
      deviceId,
      streamId: operator.phase,
      startTime,
      durationSeconds,
      dependencyIds: dependencies,
      bytes: operator.cost.bytesRead + operator.cost.bytesWritten,
      flops: operator.cost.flops,
      memoryBeforeBytes: beforeMemory,
      memoryAfterBytes: afterMemory,
      metadata: { kind: operator.kind, tensorOutputs: operator.outputs },
      provenance: operator.provenance,
    });
    events.push(event);
    state.eventByOperator.set(operator.id, event);
    state.eventById.set(event.id, event);
    state.deviceAvailable.set(deviceId, event.endTime);
    state.deviceLastEvent.set(deviceId, event.id);
    state.memoryByDevice.set(deviceId, afterMemory);
  });
  return { events, draws };
}

export function runSimulation(input: SimulationInput, options: SimulationOptions = {}): SimulationResult {
  const seed = options.seed ?? 20261005;
  const rng = new DeterministicRng(seed);
  const scheduled = scheduleGraph(input, options, rng);
  const timeline = createTimeline(scheduled.events, deriveProvenance([input.graph.provenance, input.topology.provenance], "deterministic browser timeline"), input.topology.devices.length);
  const summary = {
    makespanSeconds: timeline.makespanSeconds,
    eventCount: timeline.events.length,
    computeSeconds: timeline.events.filter((event) => event.type === "compute").reduce((sum, event) => sum + event.durationSeconds, 0),
    communicationSeconds: timeline.events.filter((event) => event.type === "communication").reduce((sum, event) => sum + event.durationSeconds, 0),
    computeUtilization: timeline.makespanSeconds > 0
      ? Math.min(1, timeline.events.filter((event) => event.type === "compute").reduce((sum, event) => sum + event.durationSeconds, 0) / (timeline.makespanSeconds * Math.max(1, input.topology.devices.length)))
      : 0,
  };
  const accounting = calculateGraphAccounting(input.graph.operators, input.graph.tensors);
  const totalBytes = timeline.events.reduce((sum, event) => sum + event.bytes, 0);
  const metrics: SimulationMetrics = {
    makespanSeconds: summary.makespanSeconds,
    totalFlops: accounting.totalFlops,
    totalBytes,
    peakMemoryBytes: timeline.peakMemoryBytes,
    computeUtilization: summary.computeUtilization,
    eventCount: summary.eventCount,
    communicationSeconds: summary.communicationSeconds,
  };
  return {
    contractVersion: SIMULATION_CONTRACT_VERSION,
    graphId: input.graph.id,
    topologyId: input.topology.id,
    seed,
    timeline,
    metrics,
    draws: scheduled.draws,
    metricProvenance: makeMetricProvenance(input.graph, input.topology),
    capabilities: BROWSER_SIMULATION_CAPABILITIES,
    provenance: makeProvenance("derived", "browser simulation engine", { note: "Deterministic analytic schedule; no live hardware measurement." }),
  };
}

export function simulate(input: SimulationInput, options: SimulationOptions = {}): SimulationResult {
  return runSimulation(input, options);
}

export interface SimulationSampleSummary {
  samples: readonly SimulationResult[];
  makespan: MetricValue<{ median: number; low: number; high: number }>;
}

/**
 * Small deterministic sample batches are browser-safe.  Large searches and
 * Python bootstrap fitting are intentionally represented as server-only
 * artifacts (see server-boundary.ts), not implemented in this Worker.
 */
export function runSimulationSamples(input: SimulationInput, count: number, options: SimulationOptions = {}): SimulationSampleSummary {
  if (!Number.isInteger(count) || count <= 0) throw new Error("sample count must be a positive integer");
  const samples = Array.from({ length: count }, (_, index) => runSimulation(input, { ...options, seed: `${options.seed ?? 20261005}:${index}` }));
  const ordered = samples.map((sample) => sample.metrics.makespanSeconds).sort((left, right) => left - right);
  const interpolate = (probability: number): number => {
    const position = (ordered.length - 1) * probability;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    return lower === upper ? ordered[lower] : ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower);
  };
  return {
    samples,
    makespan: {
      value: { median: interpolate(0.5), low: interpolate(0.05), high: interpolate(0.95) },
      provenance: makeProvenance("derived", "browser deterministic sample batch", { note: "Empirical interval over explicit Worker draws; not a calibrated posterior." }),
    },
  };
}

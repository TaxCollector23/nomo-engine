import { SIMULATION_CONTRACT_VERSION } from "./contracts";
import { makeProvenance, normalizeProvenance, type Provenance } from "./provenance";

export type DeviceKind = "gpu" | "cpu" | "tpu" | "accelerator";
export type LinkKind = "nvlink" | "nvswitch" | "pcie" | "infiniband" | "ethernet" | "custom";
export type CollectiveKind = "all-reduce" | "all-gather" | "all-to-all" | "broadcast" | "send";
export type CollectiveAlgorithm = "single" | "ring" | "tree" | "hierarchical";

export interface TopologyDevice {
  id: string;
  kind: DeviceKind;
  nodeId: string;
  index: number;
  computeFlops: number;
  memoryBytes: number;
  memoryBandwidthBytesPerSecond: number;
  provenance: Provenance;
  metadata: Readonly<Record<string, unknown>>;
}

export interface TopologyHost {
  id: string;
  deviceIds: readonly string[];
  nicIds: readonly string[];
  provenance: Provenance;
}

export interface TopologySwitch {
  id: string;
  nodeId?: string;
  portCount: number;
  provenance: Provenance;
}

export interface TopologyLink {
  id: string;
  source: string;
  target: string;
  kind: LinkKind;
  bandwidthBytesPerSecond: number;
  latencySeconds: number;
  bidirectional: boolean;
  provenance: Provenance;
  metadata: Readonly<Record<string, unknown>>;
}

export interface Topology {
  contractVersion: typeof SIMULATION_CONTRACT_VERSION;
  id: string;
  name: string;
  devices: readonly TopologyDevice[];
  hosts: readonly TopologyHost[];
  switches: readonly TopologySwitch[];
  links: readonly TopologyLink[];
  provenance: Provenance;
  assumptions: readonly string[];
}

export interface TopologyPath {
  source: string;
  target: string;
  linkIds: readonly string[];
  nodes: readonly string[];
  bandwidthBytesPerSecond: number;
  latencySeconds: number;
  transferSeconds: number;
  durationSeconds: number;
}

export interface CollectiveRequest {
  kind: CollectiveKind;
  participants: readonly string[];
  bytes: number;
  root?: string;
}

export interface CollectivePlan {
  kind: CollectiveKind;
  algorithm: CollectiveAlgorithm;
  participants: readonly string[];
  bytes: number;
  steps: number;
  bandwidthBytesPerSecond: number;
  latencySeconds: number;
  durationSeconds: number;
  paths: readonly TopologyPath[];
  provenance: Provenance;
}

interface RecordLike {
  [key: string]: unknown;
}

function asRecord(value: unknown): RecordLike {
  return typeof value === "object" && value !== null ? value as RecordLike : {};
}

function first(record: RecordLike, ...keys: string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key];
  }
  return undefined;
}

function finiteNonNegative(value: unknown, label: string, fallback: number): number {
  const number = Number(value ?? fallback);
  if (!Number.isFinite(number) || number < 0) throw new Error(`${label} must be finite and non-negative`);
  return number;
}

function positive(value: unknown, label: string, fallback: number): number {
  const number = Number(value ?? fallback);
  if (!Number.isFinite(number) || number <= 0) throw new Error(`${label} must be finite and positive`);
  return number;
}

function deviceKind(value: unknown): DeviceKind {
  const candidate = String(value ?? "gpu");
  if (candidate === "gpu" || candidate === "cpu" || candidate === "tpu" || candidate === "accelerator") return candidate;
  throw new Error(`unsupported topology device kind: ${candidate}`);
}

function linkKind(value: unknown): LinkKind {
  const candidate = String(value ?? "custom");
  if (["nvlink", "nvswitch", "pcie", "infiniband", "ethernet", "custom"].includes(candidate)) return candidate as LinkKind;
  throw new Error(`unsupported topology link kind: ${candidate}`);
}

export interface BuildTopologyOptions {
  id?: string;
  name?: string;
  deviceCount?: number;
  defaultDeviceKind?: DeviceKind;
  source?: string;
}

export function createDefaultTopology(deviceCount = 1, options: BuildTopologyOptions = {}): Topology {
  if (!Number.isInteger(deviceCount) || deviceCount <= 0) throw new Error("device count must be a positive integer");
  const deviceType = options.defaultDeviceKind ?? "gpu";
  const devices: TopologyDevice[] = Array.from({ length: deviceCount }, (_, index) => ({
    id: `device.${index}`,
    kind: deviceType,
    nodeId: "host.0",
    index,
    computeFlops: 1e12,
    memoryBytes: 80e9,
    memoryBandwidthBytesPerSecond: 1e12,
    provenance: makeProvenance("placeholder", options.source ?? "default topology", { note: "Replace with measured device characteristics." }),
    metadata: {},
  }));
  const links: TopologyLink[] = [];
  for (let index = 0; index + 1 < devices.length; index += 1) {
    links.push({
      id: `link.nvlink.${index}`,
      source: devices[index].id,
      target: devices[index + 1].id,
      kind: "nvlink",
      bandwidthBytesPerSecond: 300e9,
      latencySeconds: 3e-6,
      bidirectional: true,
      provenance: makeProvenance("spec", "default topology", { note: "Illustrative NVLink-like defaults; not a hardware measurement." }),
      metadata: {},
    });
  }
  return finalizeTopology({
    id: options.id ?? `default-${deviceCount}`,
    name: options.name ?? "Default browser topology",
    devices,
    hosts: [{ id: "host.0", deviceIds: devices.map((device) => device.id), nicIds: [], provenance: makeProvenance("placeholder", options.source ?? "default topology") }],
    switches: [],
    links,
    source: options.source ?? "default topology",
    assumptions: ["Default device and link characteristics are placeholders until hardware measurements are supplied."],
  });
}

function finalizeTopology(input: {
  id: string;
  name: string;
  devices: readonly TopologyDevice[];
  hosts: readonly TopologyHost[];
  switches: readonly TopologySwitch[];
  links: readonly TopologyLink[];
  source: string;
  assumptions: readonly string[];
}): Topology {
  const topology: Topology = {
    contractVersion: SIMULATION_CONTRACT_VERSION,
    id: input.id,
    name: input.name,
    devices: [...input.devices],
    hosts: [...input.hosts],
    switches: [...input.switches],
    links: [...input.links],
    provenance: makeProvenance("spec", input.source),
    assumptions: [...new Set(input.assumptions.map(String))],
  };
  validateTopology(topology);
  return topology;
}

export function buildTopology(raw: unknown = {}, options: BuildTopologyOptions = {}): Topology {
  const record = asRecord(raw);
  if (!Object.keys(record).length) return createDefaultTopology(options.deviceCount ?? 1, options);
  const rawDevices = Array.isArray(record.devices) ? record.devices : [];
  const count = Number(record.deviceCount ?? record.device_count ?? options.deviceCount ?? 1);
  const devices: TopologyDevice[] = rawDevices.length ? rawDevices.map((value, index) => {
    const device = asRecord(value);
    const id = String(first(device, "id", "deviceId", "device_id") ?? `device.${index}`);
    const nodeId = String(first(device, "nodeId", "node_id", "hostId", "host_id") ?? "host.0");
    return {
      id,
      kind: deviceKind(device.kind),
      nodeId,
      index: Number.isInteger(Number(device.index)) ? Number(device.index) : index,
      computeFlops: positive(first(device, "computeFlops", "compute_flops", "peakFlops", "peak_flops"), "device computeFlops", 1e12),
      memoryBytes: positive(first(device, "memoryBytes", "memory_bytes"), "device memoryBytes", 80e9),
      memoryBandwidthBytesPerSecond: positive(first(device, "memoryBandwidthBytesPerSecond", "memory_bandwidth_bytes_per_second", "memoryBandwidth", "memory_bandwidth"), "device memory bandwidth", 1e12),
      provenance: normalizeProvenance(device.provenance, makeProvenance("placeholder", String(record.source ?? options.source ?? "topology contract"))),
      metadata: asRecord(first(device, "metadata")),
    };
  }) : Array.from({ length: count }, (_, index) => ({
    id: `device.${index}`,
    kind: options.defaultDeviceKind ?? "gpu",
    nodeId: "host.0",
    index,
    computeFlops: 1e12,
    memoryBytes: 80e9,
    memoryBandwidthBytesPerSecond: 1e12,
    provenance: makeProvenance("placeholder", String(record.source ?? options.source ?? "topology contract")),
    metadata: {},
  }));
  if (!devices.length) throw new Error("topology must contain at least one device");
  const rawHosts = Array.isArray(record.hosts) ? record.hosts : Array.isArray(record.nodes) ? record.nodes : [];
  const hosts: TopologyHost[] = rawHosts.map((value, index) => {
    const host = asRecord(value);
    return {
      id: String(first(host, "id", "hostId", "host_id", "nodeId", "node_id") ?? `host.${index}`),
      deviceIds: Array.isArray(first(host, "deviceIds", "device_ids")) ? (first(host, "deviceIds", "device_ids") as unknown[]).map(String) : devices.filter((device) => device.nodeId === String(first(host, "id", "hostId", "host_id", "nodeId", "node_id") ?? `host.${index}`)).map((device) => device.id),
      nicIds: Array.isArray(first(host, "nicIds", "nic_ids")) ? (first(host, "nicIds", "nic_ids") as unknown[]).map(String) : [],
      provenance: normalizeProvenance(host.provenance, makeProvenance("placeholder", String(record.source ?? options.source ?? "topology contract"))),
    };
  });
  const completeHosts = hosts.length ? hosts : [...new Set(devices.map((device) => device.nodeId))].sort().map((id) => ({
    id,
    deviceIds: devices.filter((device) => device.nodeId === id).map((device) => device.id),
    nicIds: [],
    provenance: makeProvenance("placeholder", String(record.source ?? options.source ?? "topology contract")),
  }));
  const switches: TopologySwitch[] = (Array.isArray(record.switches) ? record.switches : []).map((value, index) => {
    const switchRecord = asRecord(value);
    return {
      id: String(first(switchRecord, "id") ?? `switch.${index}`),
      nodeId: first(switchRecord, "nodeId", "node_id") === undefined ? undefined : String(first(switchRecord, "nodeId", "node_id")),
      portCount: Math.max(1, Math.floor(positive(first(switchRecord, "portCount", "port_count"), "switch port count", 1))),
      provenance: normalizeProvenance(switchRecord.provenance, makeProvenance("placeholder", String(record.source ?? options.source ?? "topology contract"))),
    };
  });
  const links: TopologyLink[] = (Array.isArray(record.links) ? record.links : []).map((value, index) => {
    const link = asRecord(value);
    const source = String(first(link, "source", "from") ?? "");
    const target = String(first(link, "target", "to") ?? "");
    if (!source || !target) throw new Error(`topology link ${index} needs source and target`);
    return {
      id: String(first(link, "id") ?? `link.${index}`),
      source,
      target,
      kind: linkKind(link.kind),
      bandwidthBytesPerSecond: positive(first(link, "bandwidthBytesPerSecond", "bandwidth_bytes_per_second", "bandwidth", "bytesPerSecond", "bytes_per_second"), "link bandwidth", 1e9),
      latencySeconds: finiteNonNegative(first(link, "latencySeconds", "latency_seconds", "latency"), "link latency", 1e-6),
      bidirectional: link.bidirectional !== false,
      provenance: normalizeProvenance(link.provenance, makeProvenance("placeholder", String(record.source ?? options.source ?? "topology contract"))),
      metadata: asRecord(first(link, "metadata")),
    };
  });
  return finalizeTopology({
    id: String(first(record, "id") ?? options.id ?? "topology"),
    name: String(first(record, "name") ?? options.name ?? "Topology contract"),
    devices,
    hosts: completeHosts,
    switches,
    links,
    source: String(record.source ?? options.source ?? "topology contract"),
    assumptions: Array.isArray(record.assumptions) ? record.assumptions.map(String) : [],
  });
}

export function validateTopology(topology: Topology): void {
  if (topology.contractVersion !== SIMULATION_CONTRACT_VERSION) throw new Error("unsupported topology contract version");
  const ids = new Set<string>();
  for (const device of topology.devices) {
    if (ids.has(device.id)) throw new Error(`duplicate topology device id: ${device.id}`);
    ids.add(device.id);
    if (!Number.isFinite(device.computeFlops) || device.computeFlops <= 0) throw new Error(`device ${device.id} has invalid computeFlops`);
  }
  const endpoints = new Set([...topology.devices.map((device) => device.id), ...topology.switches.map((switchNode) => switchNode.id)]);
  const linkIds = new Set<string>();
  for (const link of topology.links) {
    if (linkIds.has(link.id)) throw new Error(`duplicate topology link id: ${link.id}`);
    linkIds.add(link.id);
    if (!endpoints.has(link.source) || !endpoints.has(link.target)) throw new Error(`link ${link.id} references an unknown endpoint`);
    if (!Number.isFinite(link.bandwidthBytesPerSecond) || link.bandwidthBytesPerSecond <= 0 || !Number.isFinite(link.latencySeconds) || link.latencySeconds < 0) {
      throw new Error(`link ${link.id} has invalid performance values`);
    }
  }
}

interface Adjacency {
  node: string;
  link: TopologyLink;
}

function adjacency(topology: Topology): Map<string, Adjacency[]> {
  const result = new Map<string, Adjacency[]>();
  for (const device of topology.devices) result.set(device.id, []);
  for (const switchNode of topology.switches) result.set(switchNode.id, []);
  for (const link of topology.links) {
    result.get(link.source)?.push({ node: link.target, link });
    if (link.bidirectional) result.get(link.target)?.push({ node: link.source, link });
  }
  for (const values of result.values()) values.sort((left, right) => left.link.id.localeCompare(right.link.id) || left.node.localeCompare(right.node));
  return result;
}

export function findTopologyPath(topology: Topology, source: string, target: string, bytes = 0): TopologyPath {
  validateTopology(topology);
  if (!Number.isFinite(bytes) || bytes < 0) throw new Error("path bytes must be finite and non-negative");
  if (source === target) return { source, target, linkIds: [], nodes: [source], bandwidthBytesPerSecond: Number.POSITIVE_INFINITY, latencySeconds: 0, transferSeconds: 0, durationSeconds: 0 };
  const graph = adjacency(topology);
  if (!graph.has(source) || !graph.has(target)) throw new Error(`unknown topology endpoint ${source === target ? source : `${source} or ${target}`}`);
  const distance = new Map<string, number>([[source, 0]]);
  const previous = new Map<string, { node: string; link: TopologyLink }>();
  const unvisited = new Set(graph.keys());
  while (unvisited.size) {
    const current = [...unvisited].sort((left, right) => (distance.get(left) ?? Infinity) - (distance.get(right) ?? Infinity) || left.localeCompare(right))[0];
    if (current === undefined || !Number.isFinite(distance.get(current) ?? Infinity)) break;
    unvisited.delete(current);
    if (current === target) break;
    for (const next of graph.get(current) ?? []) {
      if (!unvisited.has(next.node)) continue;
      const weight = next.link.latencySeconds + bytes / next.link.bandwidthBytesPerSecond;
      const candidate = (distance.get(current) ?? Infinity) + weight;
      const currentDistance = distance.get(next.node) ?? Infinity;
      const prior = previous.get(next.node);
      if (candidate < currentDistance - 1e-18 || (Math.abs(candidate - currentDistance) <= 1e-18 && next.link.id.localeCompare(prior?.link.id ?? "~") < 0)) {
        distance.set(next.node, candidate);
        previous.set(next.node, { node: current, link: next.link });
      }
    }
  }
  if (!previous.has(target)) throw new Error(`no topology path from ${source} to ${target}`);
  const nodes: string[] = [target];
  const links: TopologyLink[] = [];
  let cursor = target;
  while (cursor !== source) {
    const step = previous.get(cursor);
    if (!step) throw new Error(`failed to reconstruct topology path from ${source} to ${target}`);
    links.push(step.link);
    cursor = step.node;
    nodes.push(cursor);
  }
  links.reverse();
  nodes.reverse();
  const bandwidth = Math.min(...links.map((link) => link.bandwidthBytesPerSecond));
  const latency = links.reduce((sum, link) => sum + link.latencySeconds, 0);
  const transfer = bytes / bandwidth;
  return { source, target, linkIds: links.map((link) => link.id), nodes, bandwidthBytesPerSecond: bandwidth, latencySeconds: latency, transferSeconds: transfer, durationSeconds: latency + transfer };
}

function hostForDevice(topology: Topology, deviceId: string): string {
  return topology.devices.find((device) => device.id === deviceId)?.nodeId ?? "";
}

function pathCharacteristics(paths: readonly TopologyPath[]): { bandwidth: number; latency: number } {
  if (!paths.length) return { bandwidth: Number.POSITIVE_INFINITY, latency: 0 };
  return {
    bandwidth: Math.min(...paths.map((path) => path.bandwidthBytesPerSecond)),
    latency: Math.max(...paths.map((path) => path.latencySeconds)),
  };
}

export function planCollective(topology: Topology, request: CollectiveRequest): CollectivePlan {
  if (!Number.isFinite(request.bytes) || request.bytes < 0) throw new Error("collective bytes must be finite and non-negative");
  const participants = [...new Set(request.participants)];
  if (!participants.length) throw new Error("collective needs at least one participant");
  participants.forEach((participant) => {
    if (!topology.devices.some((device) => device.id === participant)) throw new Error(`collective participant is not a device: ${participant}`);
  });
  if (participants.length === 1) {
    return { kind: request.kind, algorithm: "single", participants, bytes: request.bytes, steps: 0, bandwidthBytesPerSecond: Number.POSITIVE_INFINITY, latencySeconds: 0, durationSeconds: 0, paths: [], provenance: makeProvenance("derived", "single-device collective") };
  }
  const hosts = new Set(participants.map((participant) => hostForDevice(topology, participant)));
  const algorithm: CollectiveAlgorithm = participants.length <= 2 ? "tree" : hosts.size > 1 ? "hierarchical" : "ring";
  const paths: TopologyPath[] = [];
  if (request.kind === "send" || request.kind === "broadcast") {
    const root = request.root ?? participants[0];
    for (const participant of participants) if (participant !== root) paths.push(findTopologyPath(topology, root, participant, request.bytes));
  } else {
    for (let index = 0; index < participants.length; index += 1) {
      const next = (index + 1) % participants.length;
      paths.push(findTopologyPath(topology, participants[index], participants[next], request.bytes));
    }
  }
  const characteristics = pathCharacteristics(paths);
  const count = participants.length;
  let steps: number;
  let transferFactor: number;
  if (request.kind === "all-reduce") {
    steps = 2 * (count - 1);
    transferFactor = 2 * (count - 1) / count;
  } else if (request.kind === "all-gather" || request.kind === "all-to-all") {
    steps = count - 1;
    transferFactor = (count - 1) / count;
  } else if (request.kind === "broadcast") {
    steps = algorithm === "tree" ? Math.ceil(Math.log2(count)) : count - 1;
    transferFactor = steps;
  } else {
    steps = 1;
    transferFactor = 1;
  }
  const durationSeconds = steps * characteristics.latency + (characteristics.bandwidth === Number.POSITIVE_INFINITY ? 0 : transferFactor * request.bytes / characteristics.bandwidth);
  return {
    kind: request.kind,
    algorithm,
    participants,
    bytes: request.bytes,
    steps,
    bandwidthBytesPerSecond: characteristics.bandwidth,
    latencySeconds: characteristics.latency,
    durationSeconds,
    paths,
    provenance: makeProvenance("derived", `${algorithm} ${request.kind} topology model`, { note: "Analytic collective estimate; replace link defaults with measurements." }),
  };
}

export function topologyToJSON(topology: Topology): Record<string, unknown> {
  validateTopology(topology);
  return {
    contract_version: topology.contractVersion,
    id: topology.id,
    name: topology.name,
    devices: topology.devices.map((device) => ({
      id: device.id,
      kind: device.kind,
      node_id: device.nodeId,
      index: device.index,
      compute_flops: device.computeFlops,
      memory_bytes: device.memoryBytes,
      memory_bandwidth_bytes_per_second: device.memoryBandwidthBytesPerSecond,
      provenance: device.provenance,
      metadata: device.metadata,
    })),
    hosts: topology.hosts.map((host) => ({ id: host.id, device_ids: [...host.deviceIds], nic_ids: [...host.nicIds], provenance: host.provenance })),
    switches: topology.switches.map((switchNode) => ({ id: switchNode.id, node_id: switchNode.nodeId, port_count: switchNode.portCount, provenance: switchNode.provenance })),
    links: topology.links.map((link) => ({
      id: link.id,
      source: link.source,
      target: link.target,
      kind: link.kind,
      bandwidth_bytes_per_second: link.bandwidthBytesPerSecond,
      latency_seconds: link.latencySeconds,
      bidirectional: link.bidirectional,
      provenance: link.provenance,
      metadata: link.metadata,
    })),
    provenance: topology.provenance,
    assumptions: [...topology.assumptions],
  };
}

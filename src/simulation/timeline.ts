import { SIMULATION_CONTRACT_VERSION } from "./contracts";
import { makeProvenance, normalizeProvenance, type Provenance } from "./provenance";

export type TimelineEventType = "compute" | "communication" | "memory" | "checkpoint" | "io" | (string & {});
export type TimelinePhase = "forward" | "backward" | "update" | "communication" | "system" | (string & {});

/**
 * Shared event object.  The camelCase fields are the Worker schedule
 * contract; the ``eventId/startS/durationS/resource`` aliases mirror the
 * Python product TimelineEvent so exported fixtures can round-trip cleanly.
 */
export class TimelineEvent {
  public readonly id: string;
  public readonly type: TimelineEventType;
  public readonly phase: TimelinePhase;
  public readonly operatorId?: string;
  public readonly deviceId: string;
  public readonly streamId: string;
  public readonly startTime: number;
  public readonly endTime: number;
  public readonly durationSeconds: number;
  public readonly dependencyIds: readonly string[];
  public readonly bytes: number;
  public readonly flops: number;
  public readonly memoryBeforeBytes: number;
  public readonly memoryAfterBytes: number;
  public readonly metadata: Readonly<Record<string, unknown>>;
  public readonly provenance: Provenance;
  public readonly eventId: string;
  public readonly startS: number;
  public readonly durationS: number;
  public readonly resource: string;
  public readonly kind: string;
  public readonly timeS: number;
  public readonly requestId: string;
  public readonly event: string;
  public readonly replicaId: string | null;
  public readonly details: Readonly<Record<string, unknown>>;
  public readonly label: string;
  public readonly sequence: number;
  private readonly shape: "worker" | "product" | "serving";

  public constructor(input: TimelineEventInput);
  public constructor(eventId: string, phase: string, startS: number, durationS: number, resource?: string, kind?: string, metadata?: Readonly<Record<string, unknown>>);
  public constructor(timeS: number, requestId: string, event: string, phase: string, replicaId?: string | null, details?: Readonly<Record<string, unknown>>, label?: string, sequence?: number);
  public constructor(
    inputOrTime: TimelineEventInput | string | number,
    phaseOrRequest?: string,
    startOrEvent?: number | string,
    durationOrPhase?: number | string,
    resourceOrReplica?: string | null,
    kindOrDetails?: string | Readonly<Record<string, unknown>>,
    metadataOrLabel?: Readonly<Record<string, unknown>> | string,
    sequence = 0,
  ) {
    if (typeof inputOrTime === "object") {
      const input = inputOrTime;
      const startTime = finite(input.startTime, "event start time");
      const explicitDuration = input.durationSeconds === undefined ? undefined : finite(input.durationSeconds, "event duration");
      const explicitEnd = input.endTime === undefined ? undefined : finite(input.endTime, "event end time");
      const durationSeconds = explicitDuration ?? (explicitEnd === undefined ? 0 : explicitEnd - startTime);
      const endTime = explicitEnd ?? startTime + durationSeconds;
      if (endTime < startTime - 1e-15) throw new Error(`event ${input.id} ends before it starts`);
      if (Math.abs(endTime - startTime - durationSeconds) > Math.max(1e-12, Math.abs(endTime) * 1e-12)) throw new Error(`event ${input.id} has inconsistent duration`);
      this.id = input.id;
      this.type = input.type;
      this.phase = input.phase;
      this.operatorId = input.operatorId;
      this.deviceId = input.deviceId;
      this.streamId = input.streamId ?? "default";
      this.startTime = startTime;
      this.endTime = endTime;
      this.durationSeconds = durationSeconds;
      this.dependencyIds = [...new Set(input.dependencyIds ?? [])];
      this.bytes = finite(input.bytes ?? 0, "event bytes");
      this.flops = finite(input.flops ?? 0, "event flops");
      this.memoryBeforeBytes = finite(input.memoryBeforeBytes ?? 0, "event memory before");
      this.memoryAfterBytes = finite(input.memoryAfterBytes ?? input.memoryBeforeBytes ?? 0, "event memory after");
      this.metadata = input.metadata ?? {};
      this.provenance = input.provenance ?? makeProvenance("derived", "timeline scheduler");
      this.eventId = this.id;
      this.startS = this.startTime;
      this.durationS = this.durationSeconds;
      this.resource = this.deviceId;
      this.kind = this.type;
      this.timeS = this.startTime;
      this.requestId = "";
      this.event = this.type;
      this.replicaId = null;
      this.details = this.metadata;
      this.label = "simulated";
      this.sequence = 0;
      this.shape = "worker";
      return;
    }
    if (typeof inputOrTime === "number") {
      const timeS = finite(inputOrTime, "timeline time");
      const requestId = phaseOrRequest ?? "";
      const event = String(startOrEvent ?? "event");
      const phase = String(durationOrPhase ?? "system");
      const replicaId = resourceOrReplica ?? null;
      const details = typeof kindOrDetails === "object" && kindOrDetails !== null ? kindOrDetails : {};
      const label = typeof metadataOrLabel === "string" ? metadataOrLabel : "simulated";
      this.id = `${requestId}:${event}:${sequence}`;
      this.type = event;
      this.phase = phase;
      this.operatorId = undefined;
      this.deviceId = replicaId ?? "";
      this.streamId = replicaId ?? "default";
      this.startTime = timeS;
      this.endTime = timeS;
      this.durationSeconds = 0;
      this.dependencyIds = [];
      this.bytes = 0;
      this.flops = 0;
      this.memoryBeforeBytes = 0;
      this.memoryAfterBytes = 0;
      this.metadata = details;
      this.provenance = makeProvenance("derived", label);
      this.eventId = this.id;
      this.startS = timeS;
      this.durationS = 0;
      this.resource = replicaId ?? "";
      this.kind = event;
      this.timeS = timeS;
      this.requestId = requestId;
      this.event = event;
      this.replicaId = replicaId;
      this.details = details;
      this.label = label;
      this.sequence = sequence;
      this.shape = "serving";
      return;
    }
    const eventId = inputOrTime;
    const phase = phaseOrRequest ?? "unknown";
    const startS = finite(startOrEvent as number, "timeline start");
    const durationS = finite(durationOrPhase as number, "timeline duration");
    const resource = resourceOrReplica ?? "";
    const kind = typeof kindOrDetails === "string" ? kindOrDetails : "compute";
    const metadata = typeof metadataOrLabel === "object" && metadataOrLabel !== null ? metadataOrLabel : {};
    this.id = eventId;
    this.type = kind;
    this.phase = phase;
    this.operatorId = undefined;
    this.deviceId = resource;
    this.streamId = resource || "default";
    this.startTime = startS;
    this.endTime = startS + durationS;
    this.durationSeconds = durationS;
    this.dependencyIds = [];
    this.bytes = 0;
    this.flops = 0;
    this.memoryBeforeBytes = 0;
    this.memoryAfterBytes = 0;
    this.metadata = metadata;
    this.provenance = makeProvenance("derived", "timeline event");
    this.eventId = eventId;
    this.startS = startS;
    this.durationS = durationS;
    this.resource = resource;
    this.kind = kind;
    this.timeS = startS;
    this.requestId = "";
    this.event = eventId;
    this.replicaId = null;
    this.details = metadata;
    this.label = "simulated";
    this.sequence = 0;
    this.shape = "product";
  }

  public get endS(): number {
    return this.endTime;
  }

  public asDict(): Record<string, unknown> {
    if (this.shape === "serving") return { time_s: this.timeS, request_id: this.requestId, event: this.event, phase: this.phase, replica_id: this.replicaId, details: this.details, label: this.label, sequence: this.sequence };
    if (this.shape === "product") return { event_id: this.eventId, phase: this.phase, start_s: this.startS, duration_s: this.durationS, end_s: this.endS, resource: this.resource, kind: this.kind, metadata: this.metadata };
    return { id: this.id, type: this.type, phase: this.phase, operator_id: this.operatorId, device_id: this.deviceId, stream_id: this.streamId, start_time: this.startTime, end_time: this.endTime, duration_seconds: this.durationSeconds, dependency_ids: [...this.dependencyIds], bytes: this.bytes, flops: this.flops, memory_before_bytes: this.memoryBeforeBytes, memory_after_bytes: this.memoryAfterBytes, metadata: this.metadata, provenance: this.provenance.asDict() };
  }

  public as_dict(): Record<string, unknown> { return this.asDict(); }
}

export class Timeline {
  public readonly contractVersion: typeof SIMULATION_CONTRACT_VERSION;
  public readonly events: readonly TimelineEvent[];
  public readonly makespanSeconds: number;
  public readonly peakMemoryBytes: number;
  public readonly provenance: Provenance;

  public constructor(events: readonly TimelineEvent[] = [], provenance: Provenance = makeProvenance("derived", "timeline"), deviceCount?: number) {
    this.contractVersion = SIMULATION_CONTRACT_VERSION;
    this.events = [...events].sort(eventOrder);
    this.makespanSeconds = this.events.reduce((maximum, event) => Math.max(maximum, event.endTime), 0);
    this.peakMemoryBytes = this.events.reduce((maximum, event) => Math.max(maximum, event.memoryBeforeBytes, event.memoryAfterBytes), 0);
    this.provenance = provenance;
    void deviceCount;
  }

  public get durationS(): number {
    return this.makespanSeconds;
  }

  public get duration_s(): number { return this.durationS; }

  public append(...events: TimelineEvent[]): Timeline {
    return new Timeline([...this.events, ...events], this.provenance);
  }

  public resourceUtilization(resource: string, horizonS?: number): number {
    const horizon = horizonS ?? this.durationS;
    if (!Number.isFinite(horizon) || horizon < 0) throw new Error("timeline horizon must be finite and non-negative");
    const active = this.events.filter((event) => event.resource === resource).reduce((sum, event) => sum + event.durationSeconds, 0);
    return horizon ? active / horizon : 0;
  }

  public asDict(): Record<string, unknown> {
    return { duration_s: this.durationS, events: this.events.map((event) => event.asDict()) };
  }

  public as_dict(): Record<string, unknown> { return this.asDict(); }

  public toCSV(): string {
    return ["event_id,phase,start_s,duration_s,end_s,resource,kind", ...this.events.map((event) => [event.eventId, event.phase, event.startS, event.durationS, event.endS, event.resource, event.kind].map((value) => JSON.stringify(value)).join(","))].join("\n");
  }
}

export interface TimelineSummary {
  makespanSeconds: number;
  eventCount: number;
  computeSeconds: number;
  communicationSeconds: number;
  busySecondsByDevice: Readonly<Record<string, number>>;
  peakMemoryBytes: number;
  computeUtilization: number;
}

export interface TimelineEventInput {
  id: string;
  type: TimelineEventType;
  phase: TimelinePhase;
  operatorId?: string;
  deviceId: string;
  streamId?: string;
  startTime: number;
  endTime?: number;
  durationSeconds?: number;
  dependencyIds?: readonly string[];
  bytes?: number;
  flops?: number;
  memoryBeforeBytes?: number;
  memoryAfterBytes?: number;
  metadata?: Readonly<Record<string, unknown>>;
  provenance?: Provenance;
}

function finite(value: number, label: string, minimum = 0): number {
  if (!Number.isFinite(value) || value < minimum) throw new Error(`${label} must be finite and >= ${minimum}`);
  return value;
}

function eventOrder(left: TimelineEvent, right: TimelineEvent): number {
  return left.startTime - right.startTime
    || left.endTime - right.endTime
    || left.deviceId.localeCompare(right.deviceId)
    || left.streamId.localeCompare(right.streamId)
    || left.id.localeCompare(right.id);
}

export function createTimelineEvent(input: TimelineEventInput): TimelineEvent {
  return new TimelineEvent(input);
}

export function sortTimelineEvents(events: readonly TimelineEvent[]): TimelineEvent[] {
  return [...events].sort(eventOrder);
}

export function validateTimeline(events: readonly TimelineEvent[]): void {
  const ids = new Set<string>();
  const byId = new Map<string, TimelineEvent>();
  for (const event of events) {
    if (ids.has(event.id)) throw new Error(`duplicate timeline event id: ${event.id}`);
    ids.add(event.id);
    byId.set(event.id, event);
    createTimelineEvent(event);
  }
  for (const event of events) {
    for (const dependencyId of event.dependencyIds) {
      const dependency = byId.get(dependencyId);
      if (!dependency) throw new Error(`event ${event.id} depends on unknown event ${dependencyId}`);
      if (dependency.endTime > event.startTime + 1e-12) throw new Error(`event ${event.id} starts before dependency ${dependencyId} ends`);
    }
  }
}

export function summarizeTimeline(events: readonly TimelineEvent[], deviceCount?: number): TimelineSummary {
  validateTimeline(events);
  const ordered = sortTimelineEvents(events);
  const makespanSeconds = ordered.reduce((maximum, event) => Math.max(maximum, event.endTime), 0);
  const busySecondsByDevice: Record<string, number> = {};
  let computeSeconds = 0;
  let communicationSeconds = 0;
  let peakMemoryBytes = 0;
  for (const event of ordered) {
    busySecondsByDevice[event.deviceId] = (busySecondsByDevice[event.deviceId] ?? 0) + event.durationSeconds;
    if (event.type === "compute") computeSeconds += event.durationSeconds;
    if (event.type === "communication") communicationSeconds += event.durationSeconds;
    peakMemoryBytes = Math.max(peakMemoryBytes, event.memoryBeforeBytes, event.memoryAfterBytes);
  }
  const divisor = Math.max(1, deviceCount ?? Object.keys(busySecondsByDevice).length);
  const computeUtilization = makespanSeconds > 0 ? Math.min(1, computeSeconds / (makespanSeconds * divisor)) : 0;
  return { makespanSeconds, eventCount: events.length, computeSeconds, communicationSeconds, busySecondsByDevice, peakMemoryBytes, computeUtilization };
}

export function createTimeline(events: readonly TimelineEvent[], provenance: Provenance = makeProvenance("derived", "timeline scheduler"), deviceCount?: number): Timeline {
  const ordered = sortTimelineEvents(events);
  validateTimeline(ordered);
  return new Timeline(ordered, provenance, deviceCount);
}

export interface MemoryProfilePoint {
  timeSeconds: number;
  deviceId: string;
  memoryBytes: number;
  eventId: string;
}

export function memoryProfile(events: readonly TimelineEvent[]): MemoryProfilePoint[] {
  return sortTimelineEvents(events)
    .flatMap((event) => [
      { timeSeconds: event.startTime, deviceId: event.deviceId, memoryBytes: event.memoryBeforeBytes, eventId: event.id },
      { timeSeconds: event.endTime, deviceId: event.deviceId, memoryBytes: event.memoryAfterBytes, eventId: event.id },
    ])
    .sort((left, right) => left.timeSeconds - right.timeSeconds || left.deviceId.localeCompare(right.deviceId) || left.eventId.localeCompare(right.eventId));
}

export function timelineToJSON(timeline: Timeline): Record<string, unknown> {
  return {
    contract_version: timeline.contractVersion,
    events: timeline.events.map((event) => ({
      id: event.id,
      type: event.type,
      phase: event.phase,
      operator_id: event.operatorId,
      device_id: event.deviceId,
      stream_id: event.streamId,
      start_time: event.startTime,
      end_time: event.endTime,
      duration_seconds: event.durationSeconds,
      dependency_ids: [...event.dependencyIds],
      bytes: event.bytes,
      flops: event.flops,
      memory_before_bytes: event.memoryBeforeBytes,
      memory_after_bytes: event.memoryAfterBytes,
      metadata: event.metadata,
      provenance: event.provenance,
    })),
    makespan_seconds: timeline.makespanSeconds,
    peak_memory_bytes: timeline.peakMemoryBytes,
    provenance: timeline.provenance,
  };
}

export function timelineFromJSON(raw: unknown): Timeline {
  if (typeof raw !== "object" || raw === null) throw new Error("timeline contract must be an object");
  const record = raw as Record<string, unknown>;
  if (!Array.isArray(record.events)) throw new Error("timeline contract must contain events");
  const events = record.events.map((value, index) => {
    if (typeof value !== "object" || value === null) throw new Error(`timeline event ${index} must be an object`);
    const event = value as Record<string, unknown>;
    const dependencyValue = event.dependencyIds ?? event.dependency_ids;
    return createTimelineEvent({
      id: String(event.id ?? `event.${index}`),
      type: String(event.type ?? "compute") as TimelineEventType,
      phase: String(event.phase ?? "system") as TimelinePhase,
      operatorId: event.operatorId === undefined && event.operator_id === undefined ? undefined : String(event.operatorId ?? event.operator_id),
      deviceId: String(event.deviceId ?? event.device_id ?? "device.0"),
      streamId: String(event.streamId ?? event.stream_id ?? "default"),
      startTime: Number(event.startTime ?? event.start_time ?? 0),
      endTime: Number(event.endTime ?? event.end_time ?? 0),
      durationSeconds: Number(event.durationSeconds ?? event.duration_seconds ?? 0),
      dependencyIds: Array.isArray(dependencyValue) ? dependencyValue.map(String) : [],
      bytes: Number(event.bytes ?? 0),
      flops: Number(event.flops ?? 0),
      memoryBeforeBytes: Number(event.memoryBeforeBytes ?? event.memory_before_bytes ?? 0),
      memoryAfterBytes: Number(event.memoryAfterBytes ?? event.memory_after_bytes ?? 0),
      metadata: typeof event.metadata === "object" && event.metadata !== null ? event.metadata as Record<string, unknown> : {},
      provenance: normalizeProvenance(event.provenance),
    });
  });
  return createTimeline(events, normalizeProvenance(record.provenance));
}

export function buildTimeline(events: readonly (TimelineEvent | Record<string, unknown>)[]): Timeline {
  const normalized = events.map((value, index) => {
    if (value instanceof TimelineEvent) return value;
    const event = value;
    if (event.event_id !== undefined || event.start_s !== undefined) {
      return new TimelineEvent(
        String(event.event_id ?? event.id ?? `event-${index}`),
        String(event.phase ?? "unknown"),
        Number(event.start_s ?? event.start ?? 0),
        Number(event.duration_s ?? event.duration ?? 0),
        String(event.resource ?? ""),
        String(event.kind ?? "compute"),
        typeof event.metadata === "object" && event.metadata !== null ? event.metadata as Record<string, unknown> : {},
      );
    }
    return createTimelineEvent({
      id: String(event.id ?? `event.${index}`),
      type: String(event.type ?? "compute") as TimelineEventType,
      phase: String(event.phase ?? "system"),
      operatorId: event.operatorId === undefined ? undefined : String(event.operatorId),
      deviceId: String(event.deviceId ?? event.device_id ?? "device.0"),
      streamId: String(event.streamId ?? event.stream_id ?? "default"),
      startTime: Number(event.startTime ?? event.start_time ?? 0),
      endTime: event.endTime === undefined && event.end_time === undefined ? undefined : Number(event.endTime ?? event.end_time),
      durationSeconds: event.durationSeconds === undefined && event.duration_seconds === undefined ? undefined : Number(event.durationSeconds ?? event.duration_seconds),
      dependencyIds: Array.isArray(event.dependencyIds ?? event.dependency_ids) ? (Array.isArray(event.dependencyIds) ? event.dependencyIds : event.dependency_ids as unknown[]).map(String) : [],
      bytes: Number(event.bytes ?? 0),
      flops: Number(event.flops ?? 0),
      memoryBeforeBytes: Number(event.memoryBeforeBytes ?? event.memory_before_bytes ?? 0),
      memoryAfterBytes: Number(event.memoryAfterBytes ?? event.memory_after_bytes ?? 0),
      metadata: typeof event.metadata === "object" && event.metadata !== null ? event.metadata as Record<string, unknown> : {},
      provenance: normalizeProvenance(event.provenance),
    });
  });
  return new Timeline(normalized);
}

export function createServingTimelineEvent(timeS: number, requestId: string, event: string, phase: string, replicaId?: string | null, details?: Readonly<Record<string, unknown>>, label = "simulated", sequence = 0): TimelineEvent {
  return new TimelineEvent(timeS, requestId, event, phase, replicaId, details, label, sequence);
}

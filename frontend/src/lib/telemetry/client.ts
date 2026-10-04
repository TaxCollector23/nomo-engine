import { clientId } from "../api";
import { Envelope, isEnvelope, wsUrl } from "./protocol";

export type ConnectionState = "idle" | "connecting" | "open" | "reconnecting" | "closed" | "not_found";

export interface TelemetryClientOptions {
  runId: string;
  onEnvelope: (env: Envelope) => void;
  onState?: (s: ConnectionState, detail?: string) => void;
  /** first sequence number already held by the caller (e.g. restored from cache) */
  since?: number;
  heartbeatMs?: number;
  deadmanMs?: number;
  backoffBaseMs?: number;
  backoffMaxMs?: number;
}

/**
 * Resumable, gap-safe telemetry stream (SPEC §9, §10).
 *
 * Guarantees delivered to `onEnvelope`:
 *   - envelopes arrive in strictly increasing `seq` order, without duplicates;
 *   - a missing seq is never silently skipped: on a gap the socket is recycled with
 *     ?since=<last delivered seq>, and the server answers with a replay or a snapshot;
 *   - a `snapshot` resets the caller's view and sets `last` to its seq.
 * Reconnects use exponential backoff with full jitter; a dead-man timer recycles sockets
 * that stop delivering frames (half-open TCP) even when no close event is raised.
 */
export class TelemetryClient {
  private ws: WebSocket | null = null;
  private last: number;
  private attempt = 0;
  private stopped = false;
  private terminal = false;
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private deadman: ReturnType<typeof setTimeout> | null = null;
  private retry: ReturnType<typeof setTimeout> | null = null;
  private readonly o: Required<Omit<TelemetryClientOptions, "onState">> & Pick<TelemetryClientOptions, "onState">;

  constructor(opts: TelemetryClientOptions) {
    this.o = {
      since: 0, heartbeatMs: 15_000, deadmanMs: 45_000, backoffBaseMs: 250, backoffMaxMs: 10_000, ...opts,
    };
    this.last = this.o.since;
  }

  get lastSeq(): number {
    return this.last;
  }

  start(): void {
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    this.clearTimers();
    if (this.retry) clearTimeout(this.retry);
    this.ws?.close(1000, "client stop");
    this.ws = null;
    this.o.onState?.("closed");
  }

  private open(): void {
    if (this.stopped) return;
    this.o.onState?.(this.attempt === 0 ? "connecting" : "reconnecting");
    const ws = new WebSocket(wsUrl(this.o.runId, this.last, clientId()));
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      this.o.onState?.("open");
      this.armDeadman();
      this.heartbeat = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "ping" }));
      }, this.o.heartbeatMs);
    };

    ws.onmessage = (ev: MessageEvent<string>) => {
      this.armDeadman();
      let msg: unknown;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (typeof msg === "object" && msg !== null && (msg as { type?: string }).type === "pong") return;
      if (!isEnvelope(msg)) return;
      if (msg.type === "snapshot") {
        this.last = msg.seq;
        this.o.onEnvelope(msg);
        return;
      }
      if (msg.seq <= this.last) return;                       // duplicate from replay overlap
      if (msg.seq !== this.last + 1) {                        // gap: recycle and resume
        this.recycle();
        return;
      }
      this.last = msg.seq;
      if (msg.type === "run.completed" || msg.type === "run.failed") this.terminal = true;
      this.o.onEnvelope(msg);
    };

    ws.onclose = (ev: CloseEvent) => {
      this.clearTimers();
      if (this.ws === ws) this.ws = null;
      if (ev.code === 4404) {
        this.stopped = true;
        this.o.onState?.("not_found");
        return;
      }
      if (this.stopped || this.terminal) {
        this.o.onState?.("closed");
        return;
      }
      this.scheduleReconnect();
    };

    ws.onerror = () => {
      /* onclose follows and handles recovery */
    };
  }

  private recycle(): void {
    if (this.terminal) return;                                // finished run: nothing more will arrive
    const ws = this.ws;
    this.ws = null;
    this.clearTimers();
    if (ws) {
      ws.onclose = null;
      ws.close(4000, "gap");
    }
    this.scheduleReconnect(0);
  }

  private scheduleReconnect(forceDelay?: number): void {
    const cap = Math.min(this.o.backoffMaxMs, this.o.backoffBaseMs * 2 ** this.attempt);
    const delay = forceDelay ?? Math.random() * cap;          // full jitter
    this.attempt += 1;
    this.o.onState?.("reconnecting", `retry in ${Math.round(delay)} ms`);
    this.retry = setTimeout(() => this.open(), delay);
  }

  private armDeadman(): void {
    if (this.deadman) clearTimeout(this.deadman);
    this.deadman = setTimeout(() => this.recycle(), this.o.deadmanMs);
  }

  private clearTimers(): void {
    if (this.heartbeat) clearInterval(this.heartbeat);
    if (this.deadman) clearTimeout(this.deadman);
    this.heartbeat = null;
    this.deadman = null;
  }
}

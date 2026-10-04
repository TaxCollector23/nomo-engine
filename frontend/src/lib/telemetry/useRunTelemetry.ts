"use client";

import { useEffect } from "react";

import { TelemetryClient } from "./client";
import type { Envelope } from "./protocol";
import { useRunStore } from "./store";

/** Connects the store to a run's telemetry stream for the lifetime of the calling component. */
export function useRunTelemetry(runId: string): void {
  useEffect(() => {
    const { reset, ingest, setConnection } = useRunStore.getState();
    reset(runId);
    let queue: Envelope[] = [];
    let raf: number | null = null;
    const flush = () => {
      raf = null;
      const batch = queue;
      queue = [];
      ingest(batch);
    };
    const client = new TelemetryClient({
      runId,
      onEnvelope: (env) => {
        queue.push(env);
        if (raf === null) raf = requestAnimationFrame(flush);
      },
      onState: setConnection,
    });
    client.start();
    return () => {
      client.stop();
      if (raf !== null) cancelAnimationFrame(raf);
    };
  }, [runId]);
}

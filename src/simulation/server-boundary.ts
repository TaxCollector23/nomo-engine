import type { ServerOnlyCapability } from "./contracts";

/**
 * These integrations intentionally do not ship in the browser bundle.  The
 * Worker can consume their serialized artifacts, but fitting, filesystem
 * access, telemetry ingestion, and database-backed calibration remain
 * server-only responsibilities.
 */
export const SERVER_ONLY_CAPABILITIES: readonly ServerOnlyCapability[] = [
  {
    id: "python-reference-fit",
    execution: "server",
    browserAvailable: false,
    reason: "Python calibration and bootstrap fitting require the server reference runtime.",
  },
  {
    id: "filesystem-and-database-inputs",
    execution: "server",
    browserAvailable: false,
    reason: "The browser accepts JSON contracts; filesystem and database reads stay behind an API.",
  },
  {
    id: "live-telemetry-ingestion",
    execution: "server",
    browserAvailable: false,
    reason: "Prometheus, trace, and customer telemetry connectors are server integrations.",
  },
  {
    id: "large-monte-carlo-search",
    execution: "server",
    browserAvailable: false,
    reason: "Large searches may exceed the Worker responsiveness budget; the browser consumes exported draws.",
  },
] as const;

export const BROWSER_SIMULATION_CAPABILITIES = {
  execution: "worker" as const,
  deterministic: true as const,
  serverOnly: SERVER_ONLY_CAPABILITIES,
};

export function isServerOnlyCapability(value: unknown): value is ServerOnlyCapability {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return candidate.execution === "server" && candidate.browserAvailable === false && typeof candidate.id === "string";
}

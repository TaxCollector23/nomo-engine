/**
 * Evidence metadata shared by simulation values.
 *
 * The browser simulator may consume a server-produced artifact, but it must
 * never imply that a placeholder or modelled value was measured.  Keep this
 * type small and JSON-safe so it can cross the Worker boundary unchanged.
 */

export type ProvenanceKind =
  | "spec"
  | "placeholder"
  | "calibrated"
  | "user"
  | "measured"
  | "assumption"
  | "derived"
  | "synthetic"
  | "server-only";

export interface ProvenanceInit {
  kind?: ProvenanceKind;
  source?: string;
  /** Optional source URL, file, run id, or artifact identifier. */
  uri?: string;
  artifactId?: string;
  sourceUrl?: string | null;
  /** Field or measurement name within the source. */
  field?: string;
  /** Human-readable boundary note shown in reports. */
  note?: string;
  method?: string | null;
  measured?: boolean;
  calibrated?: boolean;
  notes?: string;
  /** Confidence is descriptive metadata, not a probability unless stated by the source. */
  confidence?: number;
  observedAt?: string | null;
  /** True when the value was produced by a server-only integration. */
  serverOnly?: boolean;
}

/**
 * Superset of the Python ``product_simulations.Provenance`` record and the
 * browser evidence label.  ``artifactId``/``sourceUrl`` preserve calibration
 * provenance; ``kind``/``serverOnly`` preserve execution-boundary metadata.
 */
export class Provenance {
  public readonly kind: ProvenanceKind;
  public readonly source: string;
  public readonly uri?: string;
  public readonly artifactId: string;
  public readonly sourceUrl: string | null;
  public readonly field?: string;
  public readonly note?: string;
  public readonly method: string | null;
  public readonly measured: boolean;
  public readonly calibrated: boolean;
  public readonly notes: string;
  public readonly confidence?: number;
  public readonly observedAt: string | null;
  public readonly serverOnly: boolean;

  public constructor(init: ProvenanceInit = {}) {
    this.kind = init.kind ?? (init.calibrated ? "calibrated" : init.measured ? "measured" : "assumption");
    this.source = init.source ?? "";
    this.uri = init.uri;
    this.artifactId = init.artifactId ?? "";
    this.sourceUrl = init.sourceUrl ?? null;
    this.field = init.field;
    this.note = init.note;
    this.method = init.method ?? null;
    this.measured = init.measured ?? false;
    this.calibrated = init.calibrated ?? false;
    this.notes = init.notes ?? init.note ?? "";
    this.confidence = init.confidence;
    this.observedAt = init.observedAt ?? null;
    this.serverOnly = init.serverOnly ?? this.kind === "server-only";
  }

  public get evidenceGrade(): "calibrated" | "measured" | "sourced-assumption" | "unattributed" {
    if (this.calibrated || this.kind === "calibrated") return "calibrated";
    if (this.measured || this.kind === "measured") return "measured";
    if (this.source || this.sourceUrl || this.uri) return "sourced-assumption";
    return "unattributed";
  }

  public get artifact_id(): string { return this.artifactId; }
  public get source_url(): string | null { return this.sourceUrl; }
  public get observed_at(): string | null { return this.observedAt; }
  public get evidence_grade(): "calibrated" | "measured" | "sourced-assumption" | "unattributed" { return this.evidenceGrade; }

  public asDict(): Record<string, unknown> {
    return {
      kind: this.kind,
      artifact_id: this.artifactId,
      source: this.source,
      source_url: this.sourceUrl,
      uri: this.uri,
      field: this.field,
      observed_at: this.observedAt,
      method: this.method,
      measured: this.measured,
      calibrated: this.calibrated,
      notes: this.notes,
      confidence: this.confidence,
      server_only: this.serverOnly,
    };
  }

  public as_dict(): Record<string, unknown> { return this.asDict(); }
}

export interface Provenanced<T> {
  value: T;
  provenance: Provenance;
}

/** Measured-evidence provenance used by the Python calibration ingestion port. */
export class SourceProvenance {
  public constructor(
    public readonly sourceId: string,
    public readonly citation: string,
    public readonly url: string,
    public readonly doi: string | null = null,
    public readonly accessedAt: string | null = null,
    public readonly notes = "",
  ) {
    if (!sourceId.trim()) throw new Error("sourceId is required");
    if (!citation.trim()) throw new Error("source citation is required");
    if (!/^https?:\/\//i.test(url)) throw new Error("source URL must be HTTP(S)");
  }

  public asDict(): Record<string, unknown> {
    return {
      source_id: this.sourceId,
      citation: this.citation,
      url: this.url,
      ...(this.doi ? { doi: this.doi } : {}),
      ...(this.accessedAt ? { accessed_at: this.accessedAt } : {}),
      ...(this.notes ? { notes: this.notes } : {}),
    };
  }

  public as_dict(): Record<string, unknown> { return this.asDict(); }
}

export const PROVENANCE_KINDS: readonly ProvenanceKind[] = [
  "spec",
  "placeholder",
  "calibrated",
  "user",
  "measured",
  "assumption",
  "derived",
  "synthetic",
  "server-only",
] as const;

export function makeProvenance(
  kind: ProvenanceKind,
  source: string,
  details: ProvenanceInit = {},
): Provenance {
  if (!source.trim()) throw new Error("provenance source must not be empty");
  if (details.confidence !== undefined && (!Number.isFinite(details.confidence) || details.confidence < 0 || details.confidence > 1)) {
    throw new Error("provenance confidence must be between 0 and 1");
  }
  return new Provenance({ ...details, kind, source, serverOnly: details.serverOnly ?? kind === "server-only" });
}

/**
 * Attach a deterministic derived label to a value calculated from other
 * evidence.  Source names are sorted so equivalent input order has identical
 * JSON output and hashes.
 */
export function deriveProvenance(inputs: readonly Provenance[], note: string): Provenance {
  const sources = [...new Set(inputs.map((input) => `${input.kind}:${input.source}`))].sort();
  const serverOnly = inputs.some((input) => input.serverOnly === true || input.kind === "server-only");
  return makeProvenance("derived", sources.join("+") || "simulation", {
    note,
    serverOnly,
  });
}

export function normalizeProvenance(value: unknown, fallback: Provenance = makeProvenance("assumption", "simulation")): Provenance {
  if (typeof value !== "object" || value === null) return fallback;
  const raw = value as Record<string, unknown>;
  const kind = String(raw.kind ?? raw.type ?? fallback.kind) as ProvenanceKind;
  const safeKind = PROVENANCE_KINDS.includes(kind) ? kind : fallback.kind;
  const source = typeof raw.source === "string" && raw.source.trim() ? raw.source : fallback.source;
  const confidence = raw.confidence === undefined ? undefined : Number(raw.confidence);
  return makeProvenance(safeKind, source, {
    uri: typeof raw.uri === "string" ? raw.uri : undefined,
    artifactId: typeof raw.artifactId === "string" ? raw.artifactId : typeof raw.artifact_id === "string" ? raw.artifact_id : undefined,
    sourceUrl: typeof raw.sourceUrl === "string" ? raw.sourceUrl : typeof raw.source_url === "string" ? raw.source_url : undefined,
    field: typeof raw.field === "string" ? raw.field : undefined,
    note: typeof raw.note === "string" ? raw.note : undefined,
    method: typeof raw.method === "string" ? raw.method : undefined,
    measured: raw.measured === true,
    calibrated: raw.calibrated === true,
    notes: typeof raw.notes === "string" ? raw.notes : undefined,
    confidence: confidence !== undefined && Number.isFinite(confidence) ? confidence : undefined,
    observedAt: typeof raw.observedAt === "string" ? raw.observedAt : typeof raw.observed_at === "string" ? raw.observed_at : undefined,
    serverOnly: raw.serverOnly === true || safeKind === "server-only",
  });
}

export function provenanceLabel(provenance: Provenance): string {
  return provenance.serverOnly ? `${provenance.kind} · server-only` : provenance.kind;
}

"use client";

import { useEffect, useState } from "react";

import { api } from "./api";
import type { Catalog, DesignDetail, LayerRow, RunIn, RunStatus } from "./telemetry/protocol";

export interface RunMeta { config: RunIn | null; catalog: Catalog | null; layers: LayerRow[]; modelName: string | null; warnings: string[] }

/** Settings the run was started with, plus the model's layer table and the chip's supported precisions. */
export function useRunMeta(runId: string): RunMeta {
  const [meta, setMeta] = useState<RunMeta>({ config: null, catalog: null, layers: [], modelName: null, warnings: [] });
  useEffect(() => {
    let off = false;
    (async () => {
      try {
        const [r, c] = await Promise.all([api(`/runs/${runId}`), api("/catalog")]);
        if (!r.ok) return;
        const run = (await r.json()) as { config: RunIn; warnings?: string[] };
        const catalog = (await c.json()) as Catalog;
        const m = await api(`/models/${encodeURIComponent(run.config.model)}`);
        const model = m.ok ? ((await m.json()) as { name: string; layer_table: LayerRow[] }) : null;
        if (!off) setMeta({ config: run.config, catalog, layers: model?.layer_table ?? [], modelName: model?.name ?? null,
          warnings: run.warnings ?? [] });
      } catch {
        /* the dashboard still works from the live stream alone */
      }
    })();
    return () => { off = true; };
  }, [runId]);
  return meta;
}

const cache = new Map<string, DesignDetail>();

/** Full details (layers, summary, export options) for one design; only once the search has finished. */
export function useDesign(runId: string, key: string | null, status: RunStatus): { detail: DesignDetail | null; error: string | null } {
  const [detail, setDetail] = useState<DesignDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const done = status === "completed" || status === "stopped";
  useEffect(() => {
    setError(null);
    if (!key || !done) { setDetail(null); return; }
    const ck = `${runId}/${key}`;
    if (cache.has(ck)) { setDetail(cache.get(ck)!); return; }
    let off = false;
    setDetail(null);
    (async () => {
      const r = await api(`/runs/${runId}/designs/${encodeURIComponent(key)}`);
      if (off) return;
      if (!r.ok) { setError(`Could not load this design (${r.status}).`); return; }
      const d = (await r.json()) as DesignDetail;
      cache.set(ck, d);
      setDetail(d);
    })().catch(() => !off && setError("Could not reach the server."));
    return () => { off = true; };
  }, [runId, key, done]);
  return { detail, error };
}

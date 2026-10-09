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
interface DesignState { requestKey: string; detail: DesignDetail | null; error: string | null }

/** Full details (layers, summary, export options) for one design; only once the search has finished. */
export function useDesign(runId: string, key: string | null, status: RunStatus): { detail: DesignDetail | null; error: string | null } {
  const done = status === "completed" || status === "stopped";
  const requestKey = `${runId}/${key ?? ""}/${done ? "done" : "pending"}`;
  const cacheKey = key ? `${runId}/${key}` : null;
  const cached = cacheKey ? cache.get(cacheKey) ?? null : null;
  const [loaded, setLoaded] = useState<DesignState>({ requestKey: "", detail: null, error: null });
  const current = loaded.requestKey === requestKey ? loaded : { requestKey, detail: cached, error: null };
  useEffect(() => {
    if (!key || !done || cached) return;
    let off = false;
    (async () => {
      const r = await api(`/runs/${runId}/designs/${encodeURIComponent(key)}`);
      if (off) return;
      if (!r.ok) { setLoaded({ requestKey, detail: null, error: `Could not load this design (${r.status}).` }); return; }
      const d = (await r.json()) as DesignDetail;
      cache.set(cacheKey!, d);
      setLoaded({ requestKey, detail: d, error: null });
    })().catch(() => !off && setLoaded({ requestKey, detail: null, error: "Could not reach the server." }));
    return () => { off = true; };
  }, [runId, key, done, cached, cacheKey, requestKey]);
  return { detail: current.detail, error: current.error };
}

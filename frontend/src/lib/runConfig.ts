"use client";

import { api } from "./api";
import type { Preset, RunIn, SearchIn } from "./telemetry/protocol";

export const DEFAULT_SEARCH: SearchIn = {
  allow_continuous: true, allow_spiking: true, allow_symbolic: true, codings: ["rate", "ttfs"],
  crossing_penalty: 0, crossing_min_saving_pct: 0, p_crossover: 0.9, p_mutation: 1, archive_capacity: 0,
  patience: 12, asf_weights: null,
};

export function defaultConfig(model = "perception_cnn", hardware = "akd1500"): RunIn {
  return { model, hardware, budgets: { accuracy_drop_max: 3 }, pop_size: 64, generations: 60, seed: 0,
    search: { ...DEFAULT_SEARCH }, pins: {}, lock_symbolic: false, hardware_overrides: null, preset: "balanced_edge" };
}

export function applyPreset(cfg: RunIn, name: string, p: Preset): RunIn {
  return {
    ...cfg, preset: name,
    budgets: { ...cfg.budgets, ...p.settings.budgets },
    search: { ...DEFAULT_SEARCH, ...cfg.search, ...p.settings.search } as SearchIn,
    lock_symbolic: Boolean(p.settings.lock_symbolic),
  };
}

/** Merge a partial config (e.g. from a Copilot action) into a full one. */
export function mergeConfig(cfg: RunIn, patch: Partial<RunIn>): RunIn {
  return {
    ...cfg, ...patch,
    budgets: { ...cfg.budgets, ...(patch.budgets ?? {}) },
    search: { ...DEFAULT_SEARCH, ...cfg.search, ...(patch.search ?? {}) } as SearchIn,
    pins: { ...(cfg.pins ?? {}), ...(patch.pins ?? {}) },
  };
}

export async function readError(r: Response): Promise<string> {
  const detail = await r.json().then((j: { detail?: unknown }) => j.detail).catch(() => r.statusText);
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(" › ")}: ${d.msg}`).join("; ");
  }
  return JSON.stringify(detail);
}

/** Start a search. Returns the run id or throws an Error with a readable message. */
export async function startRun(cfg: RunIn): Promise<string> {
  const pins = Object.fromEntries(Object.entries(cfg.pins ?? {}).filter(([, p]) => p.domain || p.w_bits || p.a_bits));
  const r = await api("/runs", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...cfg, pins }) });
  if (!r.ok) throw new Error(await readError(r));
  return ((await r.json()) as { run_id: string }).run_id;
}

export const PRESET_ORDER = ["battery_saver", "ultra_low_latency", "balanced_edge", "strict_safety"] as const;

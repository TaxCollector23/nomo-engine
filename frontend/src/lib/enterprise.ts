"use client";

import { api } from "./api";
import { readError } from "./runConfig";

export interface CoDesignRequest {
  deployment_keys?: string[];
  pe_rows: number[];
  pe_cols: number[];
  sram_bytes: number[];
  memory_bandwidth_bytes_s: number[];
  precision_bits: number[];
  max_pe_count?: number;
  max_sram_bytes?: number;
  max_latency_s?: number;
  max_energy_j?: number;
}

export interface CoDesignEvaluation {
  key: string;
  deployment_key: string;
  architecture: {
    pe_rows: number;
    pe_cols: number;
    pe_count: number;
    sram_bytes: number;
    memory_bandwidth_bytes_s: number;
    precision_bits: number;
  };
  feasible: boolean;
  rank: number;
  metrics: {
    estimated_energy_j: number;
    estimated_latency_s: number;
    accuracy_loss_pp: number;
    area_proxy: number;
    required_sram_bytes: number;
    array_efficiency: number;
  };
  selection_reasons: string[];
  evidence_sources: Record<string, string>;
}

export interface CoDesignResponse {
  capability: string;
  evaluations: CoDesignEvaluation[];
  front: CoDesignEvaluation[];
  recommended: CoDesignEvaluation | null;
}

export interface EmulationResponse {
  design_key: string;
  model: string;
  hardware_profile: string;
  backend: {
    id: string;
    capability: "simulated" | "measured" | "proxy";
    physical_measurement: boolean;
    description: string;
  };
  summary: {
    vector_count: number;
    cycles: number;
    memory_accesses: number;
    cache_hit_ratio: number;
    spike_count: number;
    measurement: { capability: string; physical_measurement: boolean; note: string };
  };
  layers: Array<{ name: string; cycles: number; memory_accesses: number; cache_hit_ratio: number | null }>;
  warnings: string[];
}

async function jsonRequest<T>(path: string, body: unknown): Promise<T> {
  const response = await api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as T;
}

export function runCoDesign(runId: string, request: CoDesignRequest): Promise<CoDesignResponse> {
  return jsonRequest<CoDesignResponse>(`/runs/${runId}/co-design`, request);
}

export function runEmulation(runId: string, key?: string): Promise<EmulationResponse> {
  return jsonRequest<EmulationResponse>(`/runs/${runId}/emulation`, {
    key: key ?? null,
    vectors: 1,
    config: { limits: { max_vectors: 1 }, trace: { granularity: "stage" } },
  });
}

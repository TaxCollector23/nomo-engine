import { simulateServing } from "../simulation/serving-sim";

/** Shared, side-effect-free clinic engine used by both the main thread and the
 * clinic web worker: one job definition + a sweep of offered loads + the
 * at-load run. Kept pure so the worker is an execution detail, not a
 * different model. */

export interface SimJob {
  rate: number; bursty: boolean; promptMean: number; answerMean: number; cv: number;
  prefillRate: number; decodeRate: number; maxBatch: number; chunk: number; kvTokens: number;
  sloTtftS: number; sloItlS: number; replicas: number; count: number; seed: number;
}

export interface SweepPoint { offered: number; goodput: number; passRatio: number }

export interface ClinicResult { sweep: SweepPoint[]; atLoad: { metrics: Record<string, number | null> } | null }

function lognormal(mean: number, cv: number): { kind: string; mean: number; stddev: number } {
  const sigma = Math.sqrt(Math.log(1 + cv * cv));
  return { kind: "lognormal", mean: Math.log(mean) - (sigma * sigma) / 2, stddev: sigma };
}

export function buildResult(job: SimJob, rates: number[], load: number): ClinicResult {
  const run = (rate: number) =>
    simulateServing({
      arrival_process: job.bursty ? "bursty" : "poisson",
      arrival_rate_per_s: rate,
      burst_rate_per_s: Math.max(job.rate * 4, 2),
      burst_duration_s: 1.0,
      quiet_duration_s: 1.0,
      request_count: job.count,
      prompt_distribution: job.cv <= 0 ? { kind: "fixed", value: Math.max(1, Math.round(job.promptMean)) } : lognormal(job.promptMean, job.cv),
      answer_distribution: job.cv <= 0 ? { kind: "fixed", value: Math.max(1, Math.round(job.answerMean)) } : lognormal(job.answerMean, job.cv),
      prefill_tokens_per_s: job.prefillRate,
      decode_tokens_per_s: job.decodeRate,
      prefill_batch_gain: 0,
      decode_batch_gain: 1,
      prefill_batch_overhead_s: 0,
      decode_batch_overhead_s: 0,
      tensor_parallel: 1,
      tp_efficiency: 1,
      replicas: job.replicas,
      max_batch_size: job.maxBatch,
      chunked_prefill_tokens: job.chunk,
      kv_capacity_tokens: job.kvTokens,
      kv_block_tokens: 16,
      prefix_cache_enabled: true,
      preemption_enabled: true,
      slo_ttft_s: job.sloTtftS,
      slo_itl_s: job.sloItlS,
      slo_e2e_s: null,
      seed: job.seed,
    });
  const sweep = rates.map((offered) => {
    const m = run(offered).metrics;
    return { offered, goodput: Number(m.slo_goodput_requests_per_s), passRatio: Number(m.slo_goodput_ratio) };
  });
  return { sweep, atLoad: { metrics: run(load).metrics } };
}

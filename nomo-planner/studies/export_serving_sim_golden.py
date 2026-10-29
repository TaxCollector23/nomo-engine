"""Export serving-sim parity goldens for the browser port (src/simulation/serving-sim.ts).

Each case runs the non-disaggregated reference simulator on a fixed seed and
records the full metrics dict plus per-request scalars for the headline
distributions. The browser verify step (scripts/verify-engine.mjs) replays the
same configs through the TS port and demands identical numbers.

Usage (from nomo-planner/): PYTHONPATH=. python3 studies/export_serving_sim_golden.py ../scripts/serving-sim-golden.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nomo_planner.serving_sim import (
    ArrivalSpec,
    DistributionSpec,
    ServingAssumptions,
    simulate_serving,
)

METRIC_KEYS = [
    "requests",
    "completed_requests",
    "failed_requests",
    "wall_time_s",
    "throughput_requests_per_s",
    "throughput_output_tokens_per_s",
    "output_tokens",
    "ttft_p50_s", "ttft_p90_s", "ttft_p99_s",
    "itl_p50_s", "itl_p90_s", "itl_p99_s",
    "e2e_p50_s", "e2e_p90_s", "e2e_p99_s",
    "slo_goodput_requests", "slo_goodput_ratio", "slo_goodput_requests_per_s",
    "kv_utilization_avg", "kv_utilization_peak", "kv_utilization_p50",
    "prefix_cache_hits", "prefix_cache_hit_rate", "prefix_cache_evictions",
    "preemptions",
    "tensor_parallel", "replicas",
]


def case_poisson_fixed() -> ServingAssumptions:
    return ServingAssumptions(
        arrival_process=ArrivalSpec(kind="poisson", rate_per_s=4, request_count=120),
        prompt_distribution=DistributionSpec("fixed", value=256),
        answer_distribution=DistributionSpec("fixed", value=64),
        prefill_tokens_per_s=9000,
        decode_tokens_per_s=140,
        max_batch_size=32,
        slo_ttft_s=0.5,
        slo_itl_s=0.1,
    )


def case_lognormal_bursty() -> ServingAssumptions:
    return ServingAssumptions(
        arrival_process=ArrivalSpec(kind="bursty", burst_rate_per_s=14, burst_duration_s=2.0,
                                    quiet_duration_s=1.5, request_count=180),
        prompt_distribution=DistributionSpec("lognormal", mean=5.2, stddev=0.55),
        answer_distribution=DistributionSpec("lognormal", mean=3.6, stddev=0.7),
        prefill_tokens_per_s=8000,
        decode_tokens_per_s=120,
        max_batch_size=48,
        chunked_prefill_tokens=1024,
        kv_capacity_tokens=131072,
        slo_ttft_s=0.8,
        slo_itl_s=0.15,
    )


def case_saturation_preemption() -> ServingAssumptions:
    # Small KV so the run hits capacity, evicts and preempts (recompute path).
    return ServingAssumptions(
        arrival_process=ArrivalSpec(kind="poisson", rate_per_s=9, request_count=140),
        prompt_distribution=DistributionSpec("fixed", value=1024),
        answer_distribution=DistributionSpec("fixed", value=192),
        prefill_tokens_per_s=12000,
        decode_tokens_per_s=150,
        max_batch_size=16,
        kv_capacity_tokens=98304,
        kv_block_tokens=16,
        preemption_enabled=True,
        chunked_prefill_tokens=512,
    )


def case_prefix_cache_rows() -> list[dict]:
    """Replay rows sharing a long system prefix so the LRU cache is exercised."""
    import math
    rows = []
    for index in range(24):
        arrival = round(0.18 * index + 0.03 * ((index * 7) % 5), 4)
        shared = index % 3 == 0  # every third request shares the system prefix
        prompt = 1152 if shared else 640
        rows.append({
            "request_id": f"pfx-{index}",
            "arrival_s": arrival,
            "prompt_tokens": prompt,
            "answer_tokens": 48 + (index % 4) * 16,
            "prefix_id": "system-prompt" if shared else f"user-{index % 5}",
            "prefix_tokens": 768 if shared else 128,
        })
    return rows


def case_prefix_cache() -> ServingAssumptions:
    return ServingAssumptions(
        prefill_tokens_per_s=8000,
        decode_tokens_per_s=110,
        max_batch_size=12,
        prefix_cache_enabled=True,
        prefix_cache_capacity_blocks=256,
    )


def case_tp_replicas() -> ServingAssumptions:
    return ServingAssumptions(
        arrival_process=ArrivalSpec(kind="poisson", rate_per_s=12, request_count=200),
        prompt_distribution=DistributionSpec("fixed", value=512),
        answer_distribution=DistributionSpec("fixed", value=128),
        prefill_tokens_per_s=8000,
        decode_tokens_per_s=120,
        tensor_parallel=2,
        tp_efficiency=0.9,
        replicas=2,
        max_batch_size=64,
    )


def case_replay_rows() -> list[dict]:
    return [
        {"request_id": "r-a", "arrival_s": 0.0, "prompt_tokens": 320, "answer_tokens": 96},
        {"request_id": "r-b", "arrival_s": 0.11, "prompt_tokens": 320, "answer_tokens": 64},
        {"request_id": "r-c", "arrival_s": 0.25, "prompt_tokens": 512, "answer_tokens": 128},
        {"request_id": "r-d", "arrival_s": 0.4, "prompt_tokens": 256, "answer_tokens": 32},
        {"request_id": "r-e", "arrival_s": 0.9, "prompt_tokens": 1280, "answer_tokens": 256},
        {"request_id": "r-f", "arrival_s": 1.2, "prompt_tokens": 640, "answer_tokens": 160},
    ]


def case_continuous_batching_off() -> ServingAssumptions:
    return ServingAssumptions(
        arrival_process=ArrivalSpec(kind="poisson", rate_per_s=3, request_count=60),
        prompt_distribution=DistributionSpec("fixed", value=256),
        answer_distribution=DistributionSpec("fixed", value=64),
        prefill_tokens_per_s=8000,
        decode_tokens_per_s=120,
        max_batch_size=8,
        continuous_batching=False,
        max_decode_steps_before_prefill=3,
    )


def export_case(name: str, assumptions: ServingAssumptions, seed: int, trace_rows=None) -> dict:
    result = simulate_serving(trace_rows, assumptions, seed=seed)
    metrics = {key: result.metrics.get(key) for key in METRIC_KEYS}
    run_seed = seed  # the run-level seed override; exported assumptions must carry it
    requests = [
        {
            "request_id": request.request_id,
            "ttft_s": request.ttft_s,
            "e2e_s": request.e2e_s,
            "output_tokens": request.output_tokens,
            "preemptions": request.preemptions,
            "max_itl_s": request.max_itl_s,
        }
        for request in result.requests
    ]
    return {
        "name": name,
        "seed": seed,
        "assumptions": json.loads(json.dumps({**_assumption_kwargs(assumptions), "seed": run_seed})),
        "trace_rows": trace_rows,
        "metrics": metrics,
        "requests": requests,
        "distributions": {
            name_: {"p50": summary.p50, "p90": summary.p90, "p99": summary.p99, "values": list(summary.values)[:400]}
            for name_, summary in result.distributions.items()
        },
    }


def _assumption_kwargs(assumptions: ServingAssumptions) -> dict:
    """Flat kwargs the browser port accepts (canonical field names)."""
    process = assumptions.arrival_process
    kwargs: dict = {"seed": assumptions.seed}
    if isinstance(process, ArrivalSpec):
        kwargs["arrival_process"] = str(process.kind)
        kwargs["arrival_rate_per_s"] = process.rate_per_s
        kwargs["burst_rate_per_s"] = process.burst_rate_per_s
        kwargs["burst_duration_s"] = process.burst_duration_s
        kwargs["quiet_duration_s"] = process.quiet_duration_s
        kwargs["duration_s"] = process.duration_s
        kwargs["request_count"] = process.request_count
    else:
        kwargs["arrival_process"] = str(process)
        kwargs["arrival_rate_per_s"] = assumptions.arrival_rate_per_s
        kwargs["burst_rate_per_s"] = assumptions.burst_rate_per_s
        kwargs["burst_duration_s"] = assumptions.burst_duration_s
        kwargs["quiet_duration_s"] = assumptions.quiet_duration_s
        kwargs["duration_s"] = assumptions.duration_s
        kwargs["request_count"] = assumptions.request_count
    for source, target in (
        (assumptions.prompt_distribution, "prompt_distribution"),
        (assumptions.answer_distribution, "answer_distribution"),
    ):
        if isinstance(source, DistributionSpec):
            kwargs[target] = {"kind": source.kind, "value": source.value, "mean": source.mean, "stddev": source.stddev,
                              "low": source.low, "high": source.high, "shape": source.shape, "scale": source.scale,
                              "values": list(source.values), "weights": list(source.weights)}
        else:
            kwargs[target] = source
    passthrough = {
        "prefill_tokens_per_s", "decode_tokens_per_s", "draft_tokens_per_s",
        "prefill_batch_overhead_s", "decode_batch_overhead_s",
        "prefill_batch_gain", "decode_batch_gain", "speculative_verification_factor",
        "max_batch_size", "chunked_prefill_tokens", "max_decode_steps_before_prefill",
        "continuous_batching", "kv_block_tokens", "kv_capacity_blocks", "kv_capacity_tokens",
        "eviction_policy", "preemption_enabled", "prefix_cache_enabled",
        "prefix_cache_capacity_blocks", "prefix_cache_min_tokens",
        "speculative_enabled", "speculative_draft_tokens",
        "tensor_parallel", "tp_efficiency", "replicas",
        "slo_ttft_s", "slo_itl_s", "slo_e2e_s",
    }
    for key in passthrough:
        value = getattr(assumptions, key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../scripts/serving-sim-golden.json")
    cases = [
        export_case("poisson_fixed", case_poisson_fixed(), seed=20261027),
        export_case("lognormal_bursty", case_lognormal_bursty(), seed=424242),
        export_case("saturation_preemption", case_saturation_preemption(), seed=777),
        export_case("prefix_cache", case_prefix_cache(), seed=20261027, trace_rows=case_prefix_cache_rows()),
        export_case("tp_replicas", case_tp_replicas(), seed=999),
        export_case("replay_rows", ServingAssumptions(
            prefill_tokens_per_s=8000, decode_tokens_per_s=120, max_batch_size=4, slo_ttft_s=0.4,
        ), seed=31337, trace_rows=case_replay_rows()),
        export_case("continuous_batching_off", case_continuous_batching_off(), seed=555),
    ]
    payload = {"contract": "serving-sim-golden-v1", "cases": cases}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {len(cases)} serving-sim golden cases to {out}")


if __name__ == "__main__":
    main()

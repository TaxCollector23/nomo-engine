from nomo_planner.serving_sim import (
    ArrivalSpec,
    DistributionSpec,
    ServingAssumptions,
    simulate_serving,
)


def test_seeded_generated_arrivals_are_reproducible_and_labelled():
    config = ServingAssumptions(
        arrival_process=ArrivalSpec(kind="poisson", rate_per_s=5, duration_s=3),
        prompt_distribution=DistributionSpec("fixed", value=8),
        answer_distribution=DistributionSpec("fixed", value=3),
        prefill_tokens_per_s=1000,
        decode_tokens_per_s=1000,
    )
    first = simulate_serving(None, config, seed=31)
    second = simulate_serving(None, config, seed=31)
    assert first == second
    assert first.simulated and first.metrics["requests"] > 0
    assert all(result.label == "simulated" for result in first.requests)
    assert first.timeline


def test_bursty_arrivals_and_requested_percentiles_are_exposed():
    result = simulate_serving(
        None,
        ServingAssumptions(
            arrival_process=ArrivalSpec(
                kind="bursty",
                burst_rate_per_s=40,
                burst_duration_s=.05,
                quiet_duration_s=.10,
                request_count=8,
            ),
            prompt_distribution=DistributionSpec("fixed", value=4),
            answer_distribution=DistributionSpec("fixed", value=3),
            prefill_tokens_per_s=1000,
            decode_tokens_per_s=1000,
        ),
        seed=13,
    )
    assert len(result.requests) == 8
    assert "bursty" in result.assumptions["input_mode"]
    for prefix in ("ttft", "inter_token_latency"):
        assert result.metrics[f"{prefix}_p50_s"] is not None
        assert result.metrics[f"{prefix}_p90_s"] is not None
        assert result.metrics[f"{prefix}_p99_s"] is not None


def test_replay_trace_reports_latency_distributions_and_slo_goodput():
    trace = [
        {"request_id": "a", "arrival_time_s": 0, "input_tokens": 8, "output_tokens": 3},
        {"request_id": "b", "arrival_time_s": .01, "prompt_tokens": 4, "answer_tokens": 2},
    ]
    result = simulate_serving(
        trace,
        ServingAssumptions(prefill_tokens_per_s=100, decode_tokens_per_s=20,
                           kv_capacity_tokens=64, kv_block_tokens=4,
                           slo_ttft_s=2, slo_itl_s=2),
        seed=7,
    )
    assert [item.request_id for item in result.requests] == ["a", "b"]
    assert all(item.status == "completed" for item in result.requests)
    assert result.metrics["ttft_p50_s"] is not None
    assert result.metrics["inter_token_latency_p99_s"] is not None
    assert result.metrics["slo_goodput_requests"] == 2
    assert result.metrics["throughput_output_tokens_per_s"] > 0
    assert all(event.label == "simulated" for event in result.timeline)


def test_chunked_prefill_prefix_cache_and_disaggregation_are_visible():
    trace = [
        {"id": "one", "arrival_time_s": 0, "prompt_tokens": 12, "answer_tokens": 2,
         "prefix_key": "shared", "prefix_tokens": 4},
        {"id": "two", "arrival_time_s": .2, "prompt_tokens": 12, "answer_tokens": 2,
         "prefix_key": "shared", "prefix_tokens": 4},
    ]
    result = simulate_serving(
        trace,
        ServingAssumptions(prefill_tokens_per_s=100, decode_tokens_per_s=40,
                           kv_capacity_tokens=128, kv_block_tokens=4, max_batch_size=2,
                           chunked_prefill_tokens=4, prefix_cache_enabled=True,
                           prefix_cache_min_tokens=4, disaggregate=True,
                           prefill_replicas=1, decode_replicas=1, transfer_latency_s=.05),
    )
    events = [event.event for event in result.timeline]
    assert events.count("prefill_chunk") >= 4
    assert "prefix_cache_hit" in events
    assert "kv_transfer" in events
    assert result.metrics["prefix_cache_hits"] >= 1
    assert all(item.status == "completed" for item in result.requests)


def test_speculative_decoding_and_tensor_parallel_replicas_are_reported():
    trace = [{"id": f"r{i}", "arrival_time_s": 0, "prompt_tokens": 2, "answer_tokens": 5}
             for i in range(4)]
    config = ServingAssumptions(prefill_tokens_per_s=100, decode_tokens_per_s=100,
                                kv_capacity_tokens=64, kv_block_tokens=2,
                                replicas=2, tensor_parallel=4,
                                speculative_enabled=True, speculative_draft_tokens=3,
                                speculative_acceptance=DistributionSpec("fixed", value=1),
                                slo_ttft_s=None, slo_itl_s=None, slo_e2e_s=None)
    result = simulate_serving(trace, config, seed=4)
    assert {item.decode_replica for item in result.requests} == {"replica-0", "replica-1"}
    assert result.metrics["tensor_parallel"] == 4
    assert result.metrics["replicas"] == 2
    assert result.metrics["speculative_attempted_tokens"] > 0
    assert all(item.status == "completed" for item in result.requests)


def test_paged_kv_capacity_and_eviction_are_explicit():
    result = simulate_serving(
        [
            {"id": "active", "arrival_time_s": 0, "prompt_tokens": 4, "answer_tokens": 12,
             "prefix_key": "old", "prefix_tokens": 4},
            {"id": "next", "arrival_time_s": .01, "prompt_tokens": 4, "answer_tokens": 2,
             "prefix_key": "new", "prefix_tokens": 4},
        ],
        ServingAssumptions(prefill_tokens_per_s=100, decode_tokens_per_s=10,
                           kv_capacity_tokens=12, kv_block_tokens=4, max_batch_size=1,
                           prefix_cache_enabled=True, prefix_cache_min_tokens=4),
    )
    assert result.metrics["kv_utilization_peak"] <= 1
    assert result.metrics["prefix_cache_evictions"] >= 0
    assert result.timeline


def test_paged_kv_preemption_requeues_and_recomputes_context():
    trace = [
        {"id": str(index), "arrival_time_s": 0, "prompt_tokens": 8, "answer_tokens": 12}
        for index in range(3)
    ]
    result = simulate_serving(
        trace,
        ServingAssumptions(
            prefill_tokens_per_s=1000,
            decode_tokens_per_s=100,
            kv_capacity_tokens=32,
            kv_block_tokens=4,
            prefix_cache_enabled=False,
            max_batch_size=1,
            chunked_prefill_tokens=8,
            max_decode_steps_before_prefill=1,
        ),
    )
    assert result.metrics["preemptions"] > 0
    assert all(item.status == "completed" and item.output_tokens == 12 for item in result.requests)
    assert any(event.event == "preempt" for event in result.timeline)


def test_empirical_and_normal_distributions_are_seeded():
    config = ServingAssumptions(
        arrival_process=ArrivalSpec(kind="poisson", rate_per_s=100, duration_s=.05),
        prompt_distribution=DistributionSpec("empirical", values=(3, 7)),
        answer_distribution=DistributionSpec("normal", mean=4, stddev=0),
        prefill_tokens_per_s=1e6, decode_tokens_per_s=1e6,
    )
    result = simulate_serving(None, config, seed=2)
    assert result.requests
    assert {item.prompt_tokens for item in result.requests} <= {3, 7}
    assert all(item.answer_tokens == 4 for item in result.requests)

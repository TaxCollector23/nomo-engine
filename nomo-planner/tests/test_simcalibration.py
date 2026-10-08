import json

import pytest

from nomo_planner.simcalibration import (
    EvidenceRecord,
    bootstrap_parameter_samples,
    evaluate_held_out,
    fit_parameters,
    load_evidence,
    parameter_intervals,
    parse_evidence_csv,
    parse_evidence_json,
    predictive_interval,
    rank_correlation,
    split_held_out,
)


def _source_payload() -> dict[str, object]:
    return {
        "sources": [
            {
                "id": "lab-2026-01",
                "citation": "Test-only deterministic fixture; not a benchmark measurement",
                "url": "https://example.test/evidence/lab-2026-01",
            }
        ]
    }


def _calibration_payload() -> dict[str, object]:
    payload = _source_payload()
    rows: list[dict[str, object]] = []
    for strategy, values in {
        "PTD-P": (1.8, 2.6, 3.4, 4.2, 5.0),
        "zero3": (2.2, 3.0, 3.8, 4.6, 5.4),
    }.items():
        for index, value in enumerate(values, start=1):
            rows.append(
                {
                    "row_id": f"{strategy}-{index}",
                    "kind": "training",
                    "strategy": strategy,
                    "metric": "step_time",
                    "value": value,
                    "unit": "s",
                    "features": {"batch": index},
                    "source_id": "lab-2026-01",
                    "split": "train" if index <= 3 else "test",
                }
            )
    payload["observations"] = rows
    return payload


def test_json_ingestion_keeps_evidence_kinds_and_source_provenance():
    payload = _source_payload()
    payload.update(
        {
            "operator_microbenchmarks": [
                {
                    "row_id": "gemm-1",
                    "strategy": "fused-gemm",
                    "metrics": {"latency_us": 7.5, "bandwidth_gbps": 920},
                    "unit": "ignored-by-metric",
                    "m": 4096,
                    "n": 4096,
                    "k": 4096,
                    "source_id": "lab-2026-01",
                }
            ],
            "topology_links": [
                {
                    "row_id": "nvlink-1",
                    "strategy": "link-measurement",
                    "metric": "latency_us",
                    "value": 2.0,
                    "unit": "us",
                    "src": "gpu0",
                    "dst": "gpu1",
                    "source_id": "lab-2026-01",
                }
            ],
            "training_observations": [
                {
                    "row_id": "train-1",
                    "strategy": "PTD-P",
                    "metric": "step_time",
                    "value": 1.2,
                    "unit": "s",
                    "batch": 8,
                    "source_id": "lab-2026-01",
                }
            ],
            "serving_observations": [
                {
                    "row_id": "serve-1",
                    "strategy": "paged-kv",
                    "metric": "ttft",
                    "value": 0.03,
                    "unit": "s",
                    "prompt_tokens": 256,
                    "source_id": "lab-2026-01",
                }
            ],
        }
    )
    bundle = parse_evidence_json(json.dumps(payload))

    assert {record.kind for record in bundle.records} == {"operator", "topology", "training", "serving"}
    assert len(bundle.records) == 5
    assert {record.provenance.source_id for record in bundle.records} == {"lab-2026-01"}
    assert bundle.records[0].provenance.url.startswith("https://")
    assert any(record.metric == "latency_us" for record in bundle.records)
    assert any(record.metric == "bandwidth_gbps" for record in bundle.records)


def test_source_report_validates_dates_and_keeps_undated_sources_explicit():
    payload = _source_payload()
    payload["sources"][0]["accessed_at"] = "2026-10-05"
    payload["observations"] = [{
        "row_id": "train-1", "kind": "training", "strategy": "PTD-P",
        "metric": "step_time", "value": 1.0, "unit": "s", "source_id": "lab-2026-01",
    }]
    bundle = parse_evidence_json(json.dumps(payload))
    report = bundle.source_report(as_of="2026-10-06")
    assert report["lab-2026-01"]["row_count"] == 1
    assert report["lab-2026-01"]["freshness"]["status"] == "fresh"
    with pytest.raises(ValueError, match="ISO date"):
        parse_evidence_json(json.dumps({
            "sources": [{"id": "bad", "citation": "bad", "url": "https://example.test/bad", "accessed_at": "yesterday"}],
            "observations": [{"row_id": "bad-1", "kind": "training", "strategy": "x", "metric": "time", "value": 1, "unit": "s", "source_id": "bad"}],
        }))


def test_evidence_bundle_rejects_orphaned_or_duplicate_source_registry_entries():
    from nomo_planner.simcalibration import EvidenceBundle, EvidenceRecord, SourceProvenance

    source = SourceProvenance("declared", "A source", "https://example.test/source")
    orphan = SourceProvenance("orphan", "Another source", "https://example.test/orphan")
    record = EvidenceRecord("row-1", "training", "baseline", "step_time", 1.0, "s", {}, orphan)
    with pytest.raises(ValueError, match="undeclared sources"):
        EvidenceBundle((record,), (source,))
    with pytest.raises(ValueError, match="source_id values must be unique"):
        EvidenceBundle((), (source, source))


def test_csv_ingestion_supports_existing_training_observation_shape():
    csv_text = (
        "row_id,source,url,model,seq_len,global_batch_tokens,cluster,devices,tp,pp,zero,recompute,micro_batch,"
        "precision,metric,value,notes\n"
        "r1,Published training table,https://example.test/paper,gpt,2048,4096,a100,8,2,1,0,full,1,bf16,"
        "throughput,100,interleaved schedule\n"
    )
    bundle = parse_evidence_csv(csv_text)

    record = bundle.records[0]
    assert record.kind == "training"
    assert record.strategy == "PTD-P"
    assert record.unit == "unitless"
    assert record.features["devices"] == 8
    assert record.provenance.citation == "Published training table"


def test_ingestion_rejects_untraceable_or_non_measured_rows():
    missing_source = {
        "row_id": "bad-1",
        "kind": "serving",
        "strategy": "baseline",
        "metric": "latency_s",
        "value": 1.0,
        "unit": "s",
    }
    with pytest.raises(ValueError, match="source"):
        parse_evidence_json(json.dumps({"observations": [missing_source]}))

    synthetic = dict(missing_source)
    synthetic.update(
        {
            "source_id": "source-1",
            "source_citation": "A source",
            "source_url": "https://example.test/source-1",
            "value_type": "proxy",
        }
    )
    with pytest.raises(ValueError, match="synthetic|proxy|predicted"):
        parse_evidence_json(json.dumps({"observations": [synthetic]}))


def test_fit_bootstrap_and_intervals_are_deterministic_and_traceable():
    bundle = parse_evidence_json(json.dumps(_calibration_payload()))
    fit = fit_parameters(bundle.records, strategy="PTD-P")
    same_fit = fit_parameters(list(reversed(bundle.records)), strategy="PTD-P")
    assert fit.coefficients == same_fit.coefficients
    assert set(fit.source_row_ids) == {"PTD-P-1", "PTD-P-2", "PTD-P-3", "PTD-P-4", "PTD-P-5"}

    samples_a = bootstrap_parameter_samples(fit, replicates=8, seed=17)
    samples_b = bootstrap_parameter_samples(fit, replicates=8, seed=17)
    assert [sample.coefficients for sample in samples_a] == [sample.coefficients for sample in samples_b]
    assert all(set(sample.source_row_ids).issubset(set(fit.source_row_ids)) for sample in samples_a)

    intervals = parameter_intervals(fit, replicates=8, seed=17)
    assert set(intervals) == set(fit.terms)
    interval = predictive_interval(fit, {"batch": 5}, replicates=8, seed=17)
    assert interval["low"] <= interval["median"] <= interval["high"]


def test_held_out_metrics_are_reported_by_strategy_without_leakage():
    bundle = parse_evidence_json(json.dumps(_calibration_payload()))
    report = evaluate_held_out(
        bundle,
        bootstrap_replicates=16,
        split_seed=19,
        interval_seed=23,
    )
    assert set(report.by_strategy) == {"PTD-P", "zero3"}
    for metrics in report.by_strategy.values():
        assert metrics.train_observations == 3
        assert metrics.held_out_observations == 2
        assert metrics.mae is not None and metrics.mae >= 0
        assert metrics.mape_pct is not None and metrics.mape_pct >= 0
        assert metrics.ptd_p_pct is not None and metrics.ptd_p_pct >= 0
        assert metrics.coverage is not None and 0 <= metrics.coverage <= 1
    assert {prediction.row_id for prediction in report.predictions} == {
        "PTD-P-4",
        "PTD-P-5",
        "zero3-4",
        "zero3-5",
    }
    assert report.as_dict()["by_strategy"]["PTD-P"]["PTD-P"] >= 0


def test_split_and_rank_correlation_are_stable_and_honest_for_small_sets():
    bundle = parse_evidence_json(json.dumps(_calibration_payload()))
    split_a = split_held_out(bundle, test_fraction=0.4, seed=101)
    split_b = split_held_out(bundle, test_fraction=0.4, seed=101)
    assert [row.row_id for row in split_a.train] == [row.row_id for row in split_b.train]
    assert [row.row_id for row in split_a.held_out] == [row.row_id for row in split_b.held_out]
    assert rank_correlation([1.0], [1.0]) is None
    assert rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0


def test_published_training_csv_is_loadable_without_adding_measurements():
    bundle = load_evidence("data/training_observations.csv")
    assert len(bundle.records) == 22
    assert all(record.kind == "training" for record in bundle.records)
    assert all(record.provenance.url for record in bundle.records)
    assert {record.strategy for record in bundle.records} >= {"PTD-P", "zero3"}

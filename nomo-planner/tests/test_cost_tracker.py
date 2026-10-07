from nomo_planner.cost_tracker import (
    PUBLIC_COST_ROWS,
    compare_provider_to_physical,
    cost_coverage,
    physical_cost_per_mtok,
    source_freshness,
    validate_rows,
)


def test_every_public_cost_row_has_source_and_valid_range():
    validate_rows()
    assert len(PUBLIC_COST_ROWS) >= 5
    assert all(row.source_url.startswith("https://") for row in PUBLIC_COST_ROWS)


def test_physical_cost_requires_measured_user_inputs():
    assert physical_cost_per_mtok(gpu_hourly_cost_usd=2, measured_tokens_per_second=100) == 2 / 3600 / 100 * 1e6
    provider = next(row for row in PUBLIC_COST_ROWS if row.kind == "provider")
    result = compare_provider_to_physical(provider, gpu_hourly_cost_usd=2, measured_tokens_per_second=100)
    assert result["physical_output_usd_per_mtok"] > 0


def test_cost_coverage_does_not_count_closed_provider_rows_as_open_model_evidence():
    coverage = cost_coverage()
    assert coverage["status"] == "partial"
    assert coverage["open_model_training_rows"] == 2
    assert coverage["open_model_serving_rows"] == 0
    assert "open-model serving/token prices" in coverage["missing"]


def test_cost_source_freshness_is_metadata_only_and_undated_rows_stay_undated():
    provider = next(row for row in PUBLIC_COST_ROWS if row.item == "gpt-5.3-codex")
    assert source_freshness(provider, as_of="2026-10-06").status == "fresh"
    model_card = next(row for row in PUBLIC_COST_ROWS if row.item == "Llama 3 8B pretraining")
    assert source_freshness(model_card, as_of="2026-10-06").status == "undated"
    assert source_freshness(provider, as_of="2027-01-03").status == "stale"
    expiring = next(row for row in PUBLIC_COST_ROWS if row.item == "Gemini 3.8 Flash")
    assert source_freshness(expiring, as_of="2027-01-01").status == "expired"

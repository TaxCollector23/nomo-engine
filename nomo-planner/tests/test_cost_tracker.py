from nomo_planner.cost_tracker import PUBLIC_COST_ROWS, compare_provider_to_physical, physical_cost_per_mtok, validate_rows


def test_every_public_cost_row_has_source_and_valid_range():
    validate_rows()
    assert len(PUBLIC_COST_ROWS) >= 5
    assert all(row.source_url.startswith("https://") for row in PUBLIC_COST_ROWS)


def test_physical_cost_requires_measured_user_inputs():
    assert physical_cost_per_mtok(gpu_hourly_cost_usd=2, measured_tokens_per_second=100) == 2 / 3600 / 100 * 1e6
    provider = next(row for row in PUBLIC_COST_ROWS if row.kind == "provider")
    result = compare_provider_to_physical(provider, gpu_hourly_cost_usd=2, measured_tokens_per_second=100)
    assert result["physical_output_usd_per_mtok"] > 0

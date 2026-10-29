from datetime import date

import pytest

from nomo_planner.cost_tracker import (
    PUBLIC_COST_ROWS,
    CostRow,
    compare_provider_to_physical,
    cost_coverage,
    parse_price_csv,
    physical_cost_per_mtok,
    source_freshness,
    validate_rows,
)


def test_every_public_cost_row_has_source_and_valid_range():
    validate_rows()
    assert len(PUBLIC_COST_ROWS) >= 5
    assert all(row.source_url.startswith("https://") for row in PUBLIC_COST_ROWS)


def test_provider_rows_carry_real_cited_list_prices():
    """Provider rows must be real, verifiable published list prices with honest as-of dates."""
    by_item = {row.item: row for row in PUBLIC_COST_ROWS if row.kind == "provider"}
    expected = {
        "GPT-4o": (2.50, 10.00, "2024-08-02", "https://openai.com/api/pricing/"),
        "GPT-4o mini": (0.15, 0.60, "2024-07-18", "https://openai.com/api/pricing/"),
        "Claude 3.5 Sonnet": (3.00, 15.00, "2024-06-20", "https://www.anthropic.com/pricing"),
        "Claude 3 Haiku": (0.25, 1.25, "2024-03-13", "https://www.anthropic.com/pricing"),
        "Gemini 1.5 Flash (<=128k context)": (0.075, 0.30, "2024-05-15", "https://ai.google.dev/pricing"),
        "Gemini 1.5 Pro (<=128k context)": (1.25, 5.00, "2024-05-15", "https://ai.google.dev/pricing"),
    }
    assert set(by_item) == set(expected)
    for item, (input_usd, output_usd, checked_on, url) in expected.items():
        row = by_item[item]
        assert (row.input_usd_per_mtok, row.output_usd_per_mtok) == (input_usd, output_usd)
        assert row.checked_on == checked_on
        assert row.source_url == url
        assert str(checked_on) in row.as_of
        assert "list price" in row.as_of
        assert "not verified" not in row.notes  # cited rows are hand-checked against the source page


def test_provider_rows_are_honestly_stale_not_faked_fresh():
    """2024 list prices must age out of the freshness window instead of pretending to be current."""
    gpt4o = next(row for row in PUBLIC_COST_ROWS if row.item == "GPT-4o")
    assert source_freshness(gpt4o, as_of="2024-09-15").status == "fresh"
    assert source_freshness(gpt4o, as_of="2026-10-06").status == "stale"
    haiku = next(row for row in PUBLIC_COST_ROWS if row.item == "Claude 3 Haiku")
    assert source_freshness(haiku, as_of="2026-10-06").status == "stale"


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
    provider = next(row for row in PUBLIC_COST_ROWS if row.item == "GPT-4o")
    assert source_freshness(provider, as_of="2024-08-15").status == "fresh"
    model_card = next(row for row in PUBLIC_COST_ROWS if row.item == "Llama 3 8B pretraining")
    assert source_freshness(model_card, as_of="2026-10-06").status == "undated"
    assert source_freshness(provider, as_of="2027-01-03").status == "stale"
    expiring = CostRow(
        "Test", "expiring row", "provider", 1.0, 2.0, None, None, "USD / 1M tokens",
        "test source", "https://example.com/pricing", "test row", "synthetic row for expiry logic",
        checked_on="2023-12-01", expires_on="2024-01-01",
    )
    assert source_freshness(expiring, as_of="2024-02-01").status == "expired"


VALID_PRICE_CSV = (
    "provider,model,input_usd_per_mtok,output_usd_per_mtok,as_of,source_url\n"
    'OpenAI,gpt-4o,2.5,10,2024-08-02,https://example.com/openai-pricing\n'
    '"Anthropic, Ltd",claude-3-5-sonnet,3,15,2024-06-20,https://example.com/anthropic-pricing\n'
)


def test_parse_price_csv_accepts_strict_valid_rows():
    rows = parse_price_csv(VALID_PRICE_CSV)
    assert len(rows) == 2
    first = rows[0]
    assert first.provider_or_lab == "OpenAI"
    assert first.item == "gpt-4o"
    assert first.kind == "provider"
    assert first.input_usd_per_mtok == 2.5
    assert first.output_usd_per_mtok == 10.0
    assert first.checked_on == "2024-08-02"
    assert first.source_url == "https://example.com/openai-pricing"
    assert first.open_model is False
    # user-supplied rows must be clearly labelled and pass the shared row validation
    assert "user-supplied" in first.notes
    assert "user-supplied" in first.source_title
    validate_rows(rows)
    # quoted provider names keep their commas
    assert rows[1].provider_or_lab == "Anthropic, Ltd"
    # freshness works on user-supplied rows too: same-day check is fresh, a year later is stale
    assert source_freshness(rows[0], as_of="2024-08-02").status == "fresh"
    assert source_freshness(rows[0], as_of="2025-08-02").status == "stale"


def test_parse_price_csv_rejects_bad_input_with_clear_errors():
    header = "provider,model,input_usd_per_mtok,output_usd_per_mtok,as_of,source_url"
    good = "OpenAI,gpt-4o,2.5,10,2024-08-02,https://example.com/pricing"

    with pytest.raises(ValueError, match="missing columns"):
        parse_price_csv("provider,model,input_usd_per_mtok\nOpenAI,gpt-4o,2.5")
    with pytest.raises(ValueError, match="unsupported columns"):
        parse_price_csv(f"{header},speculative\n{good},no")
    with pytest.raises(ValueError, match="duplicate columns"):
        parse_price_csv(f"{header},provider\n{good},OpenAI")
    with pytest.raises(ValueError, match="row 2.*provider and model"):
        parse_price_csv(f"{header}\n,gpt-4o,2.5,10,2024-08-02,https://example.com/pricing")
    with pytest.raises(ValueError, match="row 3.*input_usd_per_mtok"):
        parse_price_csv(f"{header}\n{good}\nOpenAI,gpt-4o-mini,free,0.6,2024-07-18,https://example.com/pricing")
    with pytest.raises(ValueError, match="row 2.*output_usd_per_mtok"):
        parse_price_csv(f"{header}\nOpenAI,gpt-4o,2.5,-1,2024-08-02,https://example.com/pricing")
    with pytest.raises(ValueError, match="row 2.*as_of"):
        parse_price_csv(f"{header}\nOpenAI,gpt-4o,2.5,10,summer 2024,https://example.com/pricing")
    with pytest.raises(ValueError, match="row 2.*source_url"):
        parse_price_csv(f"{header}\nOpenAI,gpt-4o,2.5,10,2024-08-02,http://example.com/pricing")
    with pytest.raises(ValueError, match="duplicate provider/model pair"):
        parse_price_csv(f"{header}\n{good}\nOpenAI,gpt-4o,3,20,2025-01-01,https://example.com/pricing")
    with pytest.raises(ValueError, match="empty"):
        parse_price_csv("")
    with pytest.raises(ValueError, match="no data rows"):
        parse_price_csv(f"{header}\n")


def test_parse_price_csv_dates_never_claim_fresher_than_supplied():
    rows = parse_price_csv(VALID_PRICE_CSV)
    assert rows[0].checked_on == "2024-08-02"
    assert rows[0].expires_on is None  # no invented expiry dates
    parsed = date.fromisoformat(rows[0].checked_on)
    assert parsed.isoformat() == "2024-08-02"

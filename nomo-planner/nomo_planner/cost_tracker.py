"""Cited public cost rows and transparent physical-serving comparison helpers.

Rows are hand-entered from the cited publisher/model-card pages and include an
as-of note.  This module intentionally does not fetch or scrape prices at
runtime.  Physical serving cost is only computed when the user supplies GPU
hourly cost and measured throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .source_freshness import SourceFreshness, assess_source_freshness, parse_iso_date


@dataclass(frozen=True)
class CostRow:
    provider_or_lab: str
    item: str
    kind: str
    input_usd_per_mtok: float | None
    output_usd_per_mtok: float | None
    published_low_usd: float | None
    published_high_usd: float | None
    unit: str
    source_title: str
    source_url: str
    as_of: str
    notes: str
    open_model: bool = False
    checked_on: str | None = None
    expires_on: str | None = None


PUBLIC_COST_ROWS: tuple[CostRow, ...] = (
    CostRow("Meta", "Llama 3 8B pretraining", "training", None, None, 1_300_000, 1_300_000, "H100 GPU hours", "Llama 3 model card", "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", "published model card", "Cumulative pretraining compute, not a dollar invoice.", open_model=True),
    CostRow("Meta", "Llama 3 70B pretraining", "training", None, None, 6_400_000, 6_400_000, "H100 GPU hours", "Llama 3 model card", "https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md", "published model card", "Cumulative pretraining compute, not a dollar invoice.", open_model=True),
    CostRow("OpenAI", "gpt-5.3-codex", "provider", 3.50, 28.00, None, None, "USD / 1M tokens", "OpenAI API pricing", "https://developers.openai.com/api/docs/pricing", "pricing page accessed 2026-10-04", "Standard fast-mode row recorded from the official pricing page; verify before procurement.", checked_on="2026-10-04"),
    CostRow("Anthropic", "Claude Sonnet 4", "provider", 3.00, 15.00, None, None, "USD / 1M tokens", "Anthropic model pricing", "https://docs.anthropic.com/en/docs/about-claude/pricing", "pricing page accessed 2026-10-04", "Standard global row recorded from the official pricing documentation; verify before procurement.", checked_on="2026-10-04"),
    CostRow("Google", "Gemini 3.8 Flash", "provider", 0.75, 3.75, None, None, "USD / 1M tokens", "Gemini latest-model pricing", "https://ai.google.dev/gemini-api/docs/latest-model", "introductory pricing through 2026-12-31", "Official introductory price; the page states a later standard price.", expires_on="2026-12-31"),
)


def validate_rows(rows: tuple[CostRow, ...] = PUBLIC_COST_ROWS) -> None:
    for row in rows:
        if row.kind not in {"training", "provider"}:
            raise ValueError(f"cost row {row.item} has unsupported kind {row.kind!r}")
        if not row.source_title or not row.source_url.startswith("https://") or not row.as_of:
            raise ValueError(f"cost row {row.item} is missing a source or as-of note")
        for field_name, value in (("checked_on", row.checked_on), ("expires_on", row.expires_on)):
            if value:
                parse_iso_date(value, field=field_name)
        for field_name, value in (("input_usd_per_mtok", row.input_usd_per_mtok), ("output_usd_per_mtok", row.output_usd_per_mtok), ("published_low_usd", row.published_low_usd), ("published_high_usd", row.published_high_usd)):
            if value is not None and value < 0:
                raise ValueError(f"cost row {row.item} has a negative {field_name}")
        if row.kind == "provider" and (row.input_usd_per_mtok is None or row.output_usd_per_mtok is None):
            raise ValueError(f"provider row {row.item} is missing token prices")
        if row.published_low_usd is not None and row.published_high_usd is not None and row.published_low_usd > row.published_high_usd:
            raise ValueError(f"cost row {row.item} has an inverted range")


def source_freshness(row: CostRow, *, as_of: date | str, max_age_days: int = 90) -> SourceFreshness:
    """Report freshness from checked-in dates without fetching ``source_url``."""

    return assess_source_freshness(
        as_of=as_of,
        checked_on=row.checked_on,
        expires_on=row.expires_on,
        max_age_days=max_age_days,
    )


def cost_coverage(rows: tuple[CostRow, ...] = PUBLIC_COST_ROWS) -> dict[str, object]:
    """Describe what the checked-in cost rows do and do not cover.

    ``provider`` rows are intentionally not counted as open-model rows.  The
    current cited data has two open-model training-compute rows and no cited
    open-model serving/token-price rows, so the result stays partial.
    """

    open_training = sum(row.open_model and row.kind == "training" for row in rows)
    open_serving = sum(row.open_model and row.kind == "provider" for row in rows)
    provider_rows = sum(row.kind == "provider" for row in rows)
    missing: list[str] = []
    if open_training == 0:
        missing.append("open-model training compute")
    if open_serving == 0:
        missing.append("open-model serving/token prices")
    return {
        "status": "complete" if not missing else "partial",
        "open_model_training_rows": open_training,
        "open_model_serving_rows": open_serving,
        "provider_rows": provider_rows,
        "missing": tuple(missing),
    }


def physical_cost_per_mtok(*, gpu_hourly_cost_usd: float, measured_tokens_per_second: float) -> float:
    if gpu_hourly_cost_usd <= 0 or measured_tokens_per_second <= 0:
        raise ValueError("GPU hourly cost and measured throughput must be positive")
    return gpu_hourly_cost_usd / 3600 / measured_tokens_per_second * 1_000_000


def compare_provider_to_physical(row: CostRow, *, gpu_hourly_cost_usd: float, measured_tokens_per_second: float) -> dict[str, float | str]:
    if row.kind != "provider" or row.output_usd_per_mtok is None:
        raise ValueError("physical comparison requires a provider output-token row")
    physical = physical_cost_per_mtok(gpu_hourly_cost_usd=gpu_hourly_cost_usd, measured_tokens_per_second=measured_tokens_per_second)
    return {"item": row.item, "provider_output_usd_per_mtok": row.output_usd_per_mtok, "physical_output_usd_per_mtok": physical, "ratio": physical / row.output_usd_per_mtok}

"""Cited public cost rows and transparent physical-serving comparison helpers.

Rows are hand-entered from the cited publisher/model-card pages and include an
as-of note.  Provider rows carry the historic *published list price* and the
date it was published; the freshness engine will honestly report those rows as
stale once they age past the freshness window.  This module intentionally does
not fetch or scrape prices at runtime, and it can also load user-supplied
provider prices from a strict CSV (``parse_price_csv``).  Physical serving cost
is only computed when the user supplies GPU hourly cost and measured
throughput.
"""

from __future__ import annotations

import csv
import io
import math
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
    CostRow("OpenAI", "GPT-4o", "provider", 2.50, 10.00, None, None, "USD / 1M tokens", "OpenAI API pricing", "https://openai.com/api/pricing/", "published list price as of 2024-08-02", "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will report this row as stale. Verify current pricing before procurement.", checked_on="2024-08-02"),
    CostRow("OpenAI", "GPT-4o mini", "provider", 0.15, 0.60, None, None, "USD / 1M tokens", "OpenAI API pricing", "https://openai.com/api/pricing/", "published list price as of 2024-07-18", "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will report this row as stale. Verify current pricing before procurement.", checked_on="2024-07-18"),
    CostRow("Anthropic", "Claude 3.5 Sonnet", "provider", 3.00, 15.00, None, None, "USD / 1M tokens", "Anthropic pricing", "https://www.anthropic.com/pricing", "published list price as of 2024-06-20", "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will report this row as stale. Verify current pricing before procurement.", checked_on="2024-06-20"),
    CostRow("Anthropic", "Claude 3 Haiku", "provider", 0.25, 1.25, None, None, "USD / 1M tokens", "Anthropic pricing", "https://www.anthropic.com/pricing", "published list price as of 2024-03-13", "Historic published list price captured from the official pricing page. Provider prices have changed since; the freshness engine will report this row as stale. Verify current pricing before procurement.", checked_on="2024-03-13"),
    CostRow("Google", "Gemini 1.5 Flash (<=128k context)", "provider", 0.075, 0.30, None, None, "USD / 1M tokens", "Google AI pricing", "https://ai.google.dev/pricing", "published list price as of 2024-05-15", "Historic published list price for the <=128k-context tier captured from the official pricing page. Provider prices have changed since; the freshness engine will report this row as stale. Verify current pricing before procurement.", checked_on="2024-05-15"),
    CostRow("Google", "Gemini 1.5 Pro (<=128k context)", "provider", 1.25, 5.00, None, None, "USD / 1M tokens", "Google AI pricing", "https://ai.google.dev/pricing", "published list price as of 2024-05-15", "Historic published list price for the <=128k-context tier captured from the official pricing page. Provider prices have changed since; the freshness engine will report this row as stale. Verify current pricing before procurement.", checked_on="2024-05-15"),
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


PRICE_CSV_COLUMNS: tuple[str, ...] = (
    "provider",
    "model",
    "input_usd_per_mtok",
    "output_usd_per_mtok",
    "as_of",
    "source_url",
)


def parse_price_csv(text: str) -> tuple[CostRow, ...]:
    """Parse user-supplied provider prices from a strict CSV.

    Required columns (exactly these, no extras): ``provider``,
    ``model``, ``input_usd_per_mtok``, ``output_usd_per_mtok``, ``as_of``,
    ``source_url``.  Every field is validated and errors name the offending
    row.  The produced rows are marked as user-supplied so they are never
    confused with cited rows; they are not verified by Nomo.
    """

    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("provider price CSV is empty")
    headers = [cell.strip().lower() for cell in rows[0]]
    missing = [name for name in PRICE_CSV_COLUMNS if name not in headers]
    if missing:
        raise ValueError(f"provider price CSV is missing columns: {', '.join(missing)}")
    if len(set(headers)) != len(headers):
        raise ValueError("provider price CSV has duplicate columns")
    extra = [name for name in headers if name and name not in PRICE_CSV_COLUMNS]
    if extra:
        raise ValueError(f"provider price CSV has unsupported columns: {', '.join(extra)}; expected only {', '.join(PRICE_CSV_COLUMNS)}")

    def field(values: list[str], name: str) -> str:
        index = headers.index(name)
        return values[index].strip() if index < len(values) else ""

    out: list[CostRow] = []
    seen: set[tuple[str, str]] = set()
    for row_number, values in enumerate(rows[1:], start=2):
        provider = field(values, "provider")
        model = field(values, "model")
        if not provider or not model:
            raise ValueError(f"provider price CSV row {row_number}: provider and model must be non-empty")

        def positive_number(raw: str, column: str) -> float:
            try:
                parsed = float(raw)
            except ValueError as exc:
                raise ValueError(f"provider price CSV row {row_number}: {column} must be a non-negative number") from exc
            if not math.isfinite(parsed) or parsed < 0:
                raise ValueError(f"provider price CSV row {row_number}: {column} must be a non-negative number")
            return parsed

        input_usd = positive_number(field(values, "input_usd_per_mtok"), "input_usd_per_mtok")
        output_usd = positive_number(field(values, "output_usd_per_mtok"), "output_usd_per_mtok")

        as_of = field(values, "as_of")
        try:
            parse_iso_date(as_of, field="as_of")
        except ValueError as exc:
            raise ValueError(f"provider price CSV row {row_number}: as_of must be an ISO date like 2024-08-02") from exc

        source_url = field(values, "source_url")
        if not source_url.startswith("https://"):
            raise ValueError(f"provider price CSV row {row_number}: source_url must start with https://")

        key = (provider.lower(), model.lower())
        if key in seen:
            raise ValueError(f"provider price CSV row {row_number}: duplicate provider/model pair {provider}/{model}")
        seen.add(key)
        out.append(CostRow(
            provider, model, "provider", input_usd, output_usd, None, None, "USD / 1M tokens",
            f"{provider} pricing (user-supplied)", source_url,
            f"user-supplied list price as of {as_of}",
            "user-supplied prices; not verified by Nomo; re-check the source before procurement",
            open_model=False, checked_on=as_of, expires_on=None,
        ))
    if not out:
        raise ValueError("provider price CSV has no data rows")
    return tuple(out)

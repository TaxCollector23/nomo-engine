"""Deterministic freshness checks for checked-in source metadata.

This module never fetches a URL and never decides that a source is valid from
its HTTP response.  It only validates ISO dates already supplied by a
checked-in artifact and reports whether that metadata is current relative to a
caller-provided reference date.  Undated sources remain explicitly undated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


FreshnessStatus = Literal["fresh", "stale", "undated", "expired"]


def parse_iso_date(value: str, *, field: str = "date") -> date:
    """Parse a date-only ISO value and reject ambiguous human prose."""

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD): {value!r}") from exc
    return parsed


@dataclass(frozen=True)
class SourceFreshness:
    """A metadata-only freshness result, not a claim that a URL was fetched."""

    status: FreshnessStatus
    checked_on: str | None
    expires_on: str | None
    age_days: int | None
    expires_in_days: int | None
    max_age_days: int
    reason: str

    @property
    def is_actionable(self) -> bool:
        """Whether a caller should ask for a source re-check before use."""

        return self.status in {"stale", "expired", "undated"}

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checked_on": self.checked_on,
            "expires_on": self.expires_on,
            "age_days": self.age_days,
            "expires_in_days": self.expires_in_days,
            "max_age_days": self.max_age_days,
            "actionable": self.is_actionable,
            "reason": self.reason,
        }


def assess_source_freshness(
    *,
    as_of: date | str,
    checked_on: str | None,
    expires_on: str | None = None,
    max_age_days: int = 180,
    expiry_warning_days: int = 30,
) -> SourceFreshness:
    """Assess only dates present in source metadata.

    ``checked_on`` is the date a price or source page was recorded.  A source
    without that field is ``undated`` even when it has an expiry date.  The
    expiry date is useful context, but it is not converted into a freshness
    claim.  A caller must pass ``as_of`` so tests and reports remain
    deterministic.
    """

    reference = parse_iso_date(as_of, field="as_of") if isinstance(as_of, str) else as_of
    if not isinstance(reference, date):
        raise ValueError("as_of must be an ISO date or datetime.date")
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative")
    if expiry_warning_days < 0:
        raise ValueError("expiry_warning_days must be non-negative")

    checked = parse_iso_date(checked_on, field="checked_on") if checked_on else None
    expiry = parse_iso_date(expires_on, field="expires_on") if expires_on else None
    if checked is not None and checked > reference:
        raise ValueError(f"checked_on {checked.isoformat()} is after as_of {reference.isoformat()}")

    age_days = (reference - checked).days if checked is not None else None
    expires_in_days = (expiry - reference).days if expiry is not None else None
    if expiry is not None and expires_in_days is not None and expires_in_days < 0:
        status: FreshnessStatus = "expired"
        reason = f"source metadata expired on {expiry.isoformat()}"
    elif checked is None:
        status = "undated"
        reason = "source metadata has no checked-on date; re-check before procurement or calibration"
    elif age_days is not None and age_days > max_age_days:
        status = "stale"
        reason = f"checked {age_days} days ago; freshness window is {max_age_days} days"
    else:
        status = "fresh"
        reason = f"checked {age_days} days ago within the {max_age_days}-day freshness window"
        if expires_in_days is not None and expires_in_days <= expiry_warning_days:
            reason += f"; expires in {expires_in_days} days"

    return SourceFreshness(
        status=status,
        checked_on=checked.isoformat() if checked is not None else None,
        expires_on=expiry.isoformat() if expiry is not None else None,
        age_days=age_days,
        expires_in_days=expires_in_days,
        max_age_days=max_age_days,
        reason=reason,
    )


__all__ = ["FreshnessStatus", "SourceFreshness", "assess_source_freshness", "parse_iso_date"]

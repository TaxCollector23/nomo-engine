"""Capability and evidence boundaries for enterprise design profiles."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, Optional

from .profile import EVIDENCE_LEVELS, EnterpriseDesignProfile, ProfileValidationError, load_profile

_REPORT_SCHEMA = "nomo.enterprise/capability-evidence"
_REPORT_VERSION = 1
_REPORT_LEVELS = ("missing",) + EVIDENCE_LEVELS
_EVIDENCE_RANK = {name: index for index, name in enumerate(EVIDENCE_LEVELS)}


class EvidenceValidationError(ValueError):
    """Raised when supplied evidence or capability metadata is malformed."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("invalid capability/evidence data\n" + "\n".join(f"- {error}" for error in errors))


def _level(value: Any, path: str, errors: list[str], *, default: str = "proxy") -> str:
    if value is None:
        return default
    if not isinstance(value, str) or value not in _REPORT_LEVELS:
        errors.append(f"{path}: must be one of {', '.join(_REPORT_LEVELS)}")
        return default
    return value


def _claim(value: Any, path: str, errors: list[str], *, default: str = "proxy") -> Dict[str, Any]:
    if isinstance(value, str):
        return {"evidence_level": _level(value, f"{path}.evidence_level", errors, default=default)}
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected an evidence level or mapping")
        return {"evidence_level": default}
    item = copy.deepcopy(dict(value))
    evidence_level = item.get("evidence_level", item.get("level", item.get("status")))
    item["evidence_level"] = _level(evidence_level, f"{path}.evidence_level", errors, default=default)
    item.pop("level", None)
    if "status" in item and item["status"] in _REPORT_LEVELS:
        item.pop("status")
    return item


def _normalise_capabilities(value: Optional[Mapping[str, Any]], errors: list[str]) -> Dict[str, Dict[str, Any]]:
    defaults: Dict[str, Dict[str, Any]] = {
        "profile_validation": {
            "available": True,
            "evidence_level": "measured",
            "source": "nomo.enterprise.profile schema validation",
        },
        "deterministic_config_merge": {
            "available": True,
            "evidence_level": "measured",
            "source": "nomo.enterprise.profile merge contract",
        },
        "canonical_serialization": {
            "available": True,
            "evidence_level": "measured",
            "source": "nomo.enterprise.profile canonical JSON",
        },
    }
    if value is None:
        return defaults
    if not isinstance(value, Mapping):
        errors.append("capabilities: expected a mapping")
        return defaults
    result = dict(defaults)
    for name in sorted(value):
        if not isinstance(name, str) or not name:
            errors.append("capabilities: names must be non-empty strings")
            continue
        result[name] = _claim(value[name], f"capabilities.{name}", errors)
        available = result[name].get("available", True)
        if not isinstance(available, bool):
            errors.append(f"capabilities.{name}.available: expected a boolean")
            result[name]["available"] = False
    return {key: result[key] for key in sorted(result)}


def _normalise_observed(value: Optional[Mapping[str, Any]], errors: list[str]) -> Dict[str, Dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        errors.append("evidence: expected a mapping")
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for name in sorted(value):
        if not isinstance(name, str) or not name:
            errors.append("evidence: claim names must be non-empty strings")
            continue
        result[name] = _claim(value[name], f"evidence.{name}", errors, default="missing")
    return result


def _hardware_claims(profile: EnterpriseDesignProfile) -> Dict[str, Dict[str, Any]]:
    claims: Dict[str, Dict[str, Any]] = {}
    assumptions = profile.hardware.get("assumptions", {})
    for name in sorted(assumptions):
        item = assumptions[name]
        if isinstance(item, Mapping):
            claim = copy.deepcopy(dict(item))
            level = claim.pop("evidence", "proxy")
            value = claim.pop("value", None)
        else:
            level = "proxy"
            value = item
            claim = {}
        claims[f"hardware_assumption.{name}"] = {
            "required": False,
            "minimum": "proxy",
            "evidence_level": level,
            "source": "profile.hardware.assumptions",
            "value": value,
            **claim,
        }
    return claims


def capability_evidence_report(
    profile: EnterpriseDesignProfile | Mapping[str, Any] | str,
    *,
    evidence: Optional[Mapping[str, Any]] = None,
    capabilities: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a machine-readable claim boundary for a validated profile.

    ``evidence`` is intentionally caller-supplied.  A profile describes what
    must be proven; it cannot turn an analytic estimate into a measurement.
    Each required claim reports ``missing``, ``proxy``, ``simulated``, or
    ``measured`` and whether it meets the profile's minimum level.
    """
    validated = load_profile(profile)
    errors: list[str] = []
    observed = _normalise_observed(evidence, errors)
    capability_rows = _normalise_capabilities(capabilities, errors)
    if errors:
        raise EvidenceValidationError(errors)

    claims: Dict[str, Dict[str, Any]] = {}
    required_names: list[str] = []
    for requirement in validated.required_evidence:
        name = requirement["name"]
        minimum = requirement["minimum"]
        supplied = observed.get(name)
        if supplied is None:
            row = {
                "required": bool(requirement.get("required", True)),
                "minimum": minimum,
                "evidence_level": "missing",
                "status": "missing",
                "satisfies_requirement": not bool(requirement.get("required", True)),
            }
        else:
            level = supplied["evidence_level"]
            row = {
                **copy.deepcopy(supplied),
                "required": bool(requirement.get("required", True)),
                "minimum": minimum,
                "evidence_level": level,
                "status": "accepted" if level != "missing" and _EVIDENCE_RANK.get(level, -1) >= _EVIDENCE_RANK[minimum] else "insufficient",
                "satisfies_requirement": (not bool(requirement.get("required", True)))
                or (level != "missing" and _EVIDENCE_RANK.get(level, -1) >= _EVIDENCE_RANK[minimum]),
            }
        claims[name] = row
        if bool(requirement.get("required", True)):
            required_names.append(name)

    for name, row in observed.items():
        if name not in claims:
            claims[name] = {
                **copy.deepcopy(row),
                "required": False,
                "minimum": "proxy",
                "status": "informational",
                "satisfies_requirement": True,
            }
    for name, row in _hardware_claims(validated).items():
        claims.setdefault(name, row)

    required_rows = [claims[name] for name in required_names]
    satisfied = sum(1 for row in required_rows if row["satisfies_requirement"])
    levels = {level: 0 for level in _REPORT_LEVELS}
    for row in claims.values():
        level = row.get("evidence_level", "missing")
        levels[level] = levels.get(level, 0) + 1
    missing = [name for name in required_names if not claims[name]["satisfies_requirement"]]
    report = {
        "schema": _REPORT_SCHEMA,
        "version": _REPORT_VERSION,
        "profile": {
            "id": validated.id,
            "version": validated.version,
            "fingerprint": validated.fingerprint,
            "organization_id": validated.organization["id"],
            "project_id": validated.project["id"],
        },
        "capabilities": capability_rows,
        "claims": {name: claims[name] for name in sorted(claims)},
        "summary": {
            "required_claims": len(required_names),
            "satisfied_required_claims": satisfied,
            "missing_or_insufficient": missing,
            "by_evidence_level": levels,
            "release_ready": not missing,
        },
        "policy": {
            "claim_policy": validated.safety["claim_policy"],
            "fail_closed": validated.safety["fail_closed"],
            "required_evidence": [copy.deepcopy(item) for item in validated.required_evidence],
        },
    }
    return report


def build_capability_evidence_report(
    profile: EnterpriseDesignProfile | Mapping[str, Any] | str,
    *,
    evidence: Optional[Mapping[str, Any]] = None,
    capabilities: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Descriptive alias for :func:`capability_evidence_report`."""
    return capability_evidence_report(profile, evidence=evidence, capabilities=capabilities)


__all__ = [
    "EvidenceValidationError",
    "build_capability_evidence_report",
    "capability_evidence_report",
]

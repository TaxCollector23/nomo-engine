"""Versioned enterprise design profiles.

An enterprise profile is a checked-in deployment contract rather than another
set of scalar knobs.  It names the team's objectives, constraints, hardware
assumptions, safety policy, precision envelope, evidence needed for claims,
and the organization/project that owns the decision.

The profile's optional ``config`` mapping is the only part that changes an
existing run configuration.  :func:`merge_with_base_config` applies that
mapping recursively to a base ``nomo.yaml`` and records the canonical profile
under ``enterprise.profile``.  Mapping values merge recursively; lists and
scalars are replaced by the profile value.  The base input is never mutated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import load_config

PROFILE_SCHEMA = "nomo.enterprise/design-profile"
PROFILE_VERSION = 1
EVIDENCE_LEVELS = ("proxy", "simulated", "measured")
_EVIDENCE_RANK = {name: index for index, name in enumerate(EVIDENCE_LEVELS)}
_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "id",
    "profile_id",
    "name",
    "description",
    "objectives",
    "constraints",
    "hardware",
    "safety",
    "precision",
    "required_evidence",
    "organization",
    "project",
    "config",
}


class ProfileValidationError(ValueError):
    """A user-facing validation failure with every discovered path error."""

    def __init__(self, errors: str | Sequence[str], *, source: str = "profile") -> None:
        if isinstance(errors, str):
            values = (errors,)
        else:
            values = tuple(str(error) for error in errors)
        self.errors = values
        self.source = source
        detail = "\n".join(f"- {error}" for error in values)
        super().__init__(f"{source}: invalid enterprise design profile\n{detail}")


def _copy_value(value: Any, path: str, errors: list[str]) -> Any:
    """Copy JSON-compatible values while reporting unsupported values clearly."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{path}: number must be finite")
            return None
        return value
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                errors.append(f"{path}: mapping keys must be non-empty strings")
                continue
            result[key] = _copy_value(item, f"{path}.{key}", errors)
        return result
    if isinstance(value, (list, tuple)):
        return [_copy_value(item, f"{path}[{index}]", errors) for index, item in enumerate(value)]
    errors.append(f"{path}: value must be JSON-compatible, got {type(value).__name__}")
    return None


def _mapping(value: Any, path: str, errors: list[str], *, required: bool = True) -> Dict[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected a mapping")
        return {}
    copied = _copy_value(value, path, errors)
    return copied if isinstance(copied, dict) else {}


def _list(value: Any, path: str, errors: list[str], *, required: bool = True) -> list[Any]:
    if value is None and not required:
        return []
    if not isinstance(value, (list, tuple)):
        errors.append(f"{path}: expected a list")
        return []
    copied = _copy_value(value, path, errors)
    return copied if isinstance(copied, list) else []


def _string(value: Any, path: str, errors: list[str], *, required: bool = True, default: str = "") -> str:
    if value is None and not required:
        return default
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected a non-empty string")
        return default
    return value.strip()


def _number(value: Any, path: str, errors: list[str], *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.append(f"{path}: expected a finite number")
        return 0.0
    number = float(value)
    if minimum is not None and number < minimum:
        errors.append(f"{path}: must be >= {minimum:g}")
    return number


def _integer(value: Any, path: str, errors: list[str], *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{path}: expected an integer")
        return 0
    if minimum is not None and value < minimum:
        errors.append(f"{path}: must be >= {minimum}")
    return value


def _alias(mapping: Mapping[str, Any], primary: str, alias: str, path: str, errors: list[str]) -> Any:
    if primary in mapping and alias in mapping:
        errors.append(f"{path}: use only one of '{primary}' or '{alias}'")
    return mapping.get(primary, mapping.get(alias))


def _evidence_level(value: Any, path: str, errors: list[str], *, default: str = "proxy") -> str:
    level = default if value is None else value
    if not isinstance(level, str) or level not in EVIDENCE_LEVELS:
        errors.append(f"{path}: must be one of {', '.join(EVIDENCE_LEVELS)}")
        return default
    return level


def _normalise_bits(value: Any, path: str, errors: list[str]) -> list[int]:
    bits = _list(value, path, errors)
    result: list[int] = []
    for index, item in enumerate(bits):
        result.append(_integer(item, f"{path}[{index}]", errors, minimum=1))
    if not result:
        errors.append(f"{path}: must contain at least one precision")
    return sorted(set(result))


def _normalise_objectives(value: Any, errors: list[str]) -> list[Dict[str, Any]]:
    rows = _list(value, "objectives", errors)
    result: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"objectives[{index}]"
        row = _mapping(raw, path, errors)
        name = _string(row.get("name"), f"{path}.name", errors)
        if name in seen:
            errors.append(f"{path}.name: duplicate objective '{name}'")
        seen.add(name)
        direction = _string(row.get("direction"), f"{path}.direction", errors).lower()
        direction = {"min": "minimize", "max": "maximize"}.get(direction, direction)
        if direction not in {"minimize", "maximize"}:
            errors.append(f"{path}.direction: must be 'minimize' or 'maximize'")
        target = _number(row.get("target"), f"{path}.target", errors)
        unit = _string(row.get("unit"), f"{path}.unit", errors)
        priority = _integer(row.get("priority"), f"{path}.priority", errors, minimum=1)
        row.update({"name": name, "direction": direction, "target": target, "unit": unit, "priority": priority})
        if "evidence" in row:
            row["evidence"] = _evidence_level(row["evidence"], f"{path}.evidence", errors)
        result.append(row)
    if not result:
        errors.append("objectives: must contain at least one named objective")
    return result


def _normalise_hardware(value: Any, errors: list[str]) -> Dict[str, Any]:
    hardware = _mapping(value, "hardware", errors)
    target = _string(hardware.get("target", hardware.get("id")), "hardware.target", errors)
    assumptions = _mapping(hardware.get("assumptions"), "hardware.assumptions", errors)
    if not assumptions:
        errors.append("hardware.assumptions: must contain at least one assumption")
    normalised: Dict[str, Any] = {}
    for key, raw in assumptions.items():
        path = f"hardware.assumptions.{key}"
        if isinstance(raw, Mapping):
            item = _mapping(raw, path, errors)
            if "value" not in item:
                errors.append(f"{path}.value: is required")
                item["value"] = None
            if "evidence" in item:
                item["evidence"] = _evidence_level(item["evidence"], f"{path}.evidence", errors)
            else:
                item["evidence"] = "proxy"
            normalised[key] = item
        else:
            copied = _copy_value(raw, path, errors)
            normalised[key] = {"value": copied, "evidence": "proxy"}
    provenance = _evidence_level(hardware.get("provenance"), "hardware.provenance", errors)
    hardware.update({"target": target, "assumptions": normalised, "provenance": provenance})
    return hardware


def _normalise_safety(value: Any, errors: list[str]) -> Dict[str, Any]:
    safety = _mapping(value, "safety", errors)
    policies = _list(safety.get("policies"), "safety.policies", errors)
    policy_names: list[str] = []
    for index, policy in enumerate(policies):
        policy_names.append(_string(policy, f"safety.policies[{index}]", errors))
    if not policy_names:
        errors.append("safety.policies: must contain at least one policy")
    guards = _list(safety.get("required_guards", []), "safety.required_guards", errors, required=False)
    guard_names = [_string(guard, f"safety.required_guards[{index}]", errors) for index, guard in enumerate(guards)]
    fail_closed = safety.get("fail_closed", True)
    if not isinstance(fail_closed, bool):
        errors.append("safety.fail_closed: expected a boolean")
        fail_closed = True
    claim_policy = safety.get("claim_policy", "allow_proxy_with_label")
    if claim_policy not in {"measured_only", "measured_or_simulated", "allow_proxy_with_label"}:
        errors.append("safety.claim_policy: must be measured_only, measured_or_simulated, or allow_proxy_with_label")
        claim_policy = "allow_proxy_with_label"
    safety.update({"policies": policy_names, "required_guards": guard_names,
                   "fail_closed": fail_closed, "claim_policy": claim_policy})
    return safety


def _normalise_precision(value: Any, errors: list[str]) -> Dict[str, Any]:
    precision = _mapping(value, "precision", errors)
    weight_bits = _alias(precision, "allowed_weight_bits", "weights_bits", "precision", errors)
    activation_bits = _alias(precision, "allowed_activation_bits", "activations_bits", "precision", errors)
    accumulator_bits = _alias(precision, "allowed_accumulator_bits", "accumulator_bits", "precision", errors)
    normalised = dict(precision)
    normalised["allowed_weight_bits"] = _normalise_bits(weight_bits, "precision.allowed_weight_bits", errors)
    normalised["allowed_activation_bits"] = _normalise_bits(activation_bits, "precision.allowed_activation_bits", errors)
    normalised["allowed_accumulator_bits"] = _normalise_bits(
        accumulator_bits, "precision.allowed_accumulator_bits", errors)
    domains = _list(precision.get("allowed_domains"), "precision.allowed_domains", errors)
    domain_names = [_string(domain, f"precision.allowed_domains[{index}]", errors)
                    for index, domain in enumerate(domains)]
    if not domain_names:
        errors.append("precision.allowed_domains: must contain at least one domain")
    normalised["allowed_domains"] = domain_names
    integer_runtime = precision.get("require_integer_runtime", False)
    if not isinstance(integer_runtime, bool):
        errors.append("precision.require_integer_runtime: expected a boolean")
        integer_runtime = False
    normalised["require_integer_runtime"] = integer_runtime
    return normalised


def _normalise_evidence(value: Any, errors: list[str]) -> list[Dict[str, Any]]:
    rows = _list(value, "required_evidence", errors)
    result: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        path = f"required_evidence[{index}]"
        row = _mapping(raw, path, errors)
        name = _string(row.get("name"), f"{path}.name", errors)
        if name in seen:
            errors.append(f"{path}.name: duplicate evidence requirement '{name}'")
        seen.add(name)
        minimum = row.get("minimum", row.get("minimum_level"))
        if "minimum" in row and "minimum_level" in row:
            errors.append(f"{path}: use only one of 'minimum' or 'minimum_level'")
        minimum = _evidence_level(minimum, f"{path}.minimum", errors)
        required = row.get("required", True)
        if not isinstance(required, bool):
            errors.append(f"{path}.required: expected a boolean")
            required = True
        row.update({"name": name, "minimum": minimum, "required": required})
        result.append(row)
    if not result:
        errors.append("required_evidence: must contain at least one evidence requirement")
    return result


def _normalise_owner(value: Any, path: str, errors: list[str]) -> Dict[str, Any]:
    owner = _mapping(value, path, errors)
    owner_id = _string(owner.get("id"), f"{path}.id", errors)
    name = _string(owner.get("name"), f"{path}.name", errors)
    owner.update({"id": owner_id, "name": name})
    return owner


def _normalise_profile(raw: Mapping[str, Any], *, source: str) -> Dict[str, Any]:
    errors: list[str] = []
    data = _mapping(raw, source, errors)
    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    for key in unknown:
        errors.append(f"{source}.{key}: unknown top-level field")
    schema = _string(data.get("schema"), f"{source}.schema", errors)
    if schema != PROFILE_SCHEMA:
        errors.append(f"{source}.schema: expected '{PROFILE_SCHEMA}'")
    version = _integer(data.get("version"), f"{source}.version", errors, minimum=1)
    if version != PROFILE_VERSION:
        errors.append(f"{source}.version: unsupported version {version}; supported version is {PROFILE_VERSION}")
    profile_id = data.get("id", data.get("profile_id"))
    if "id" in data and "profile_id" in data:
        errors.append(f"{source}: use only one of 'id' or 'profile_id'")
    profile_id = _string(profile_id, f"{source}.id", errors)
    if profile_id and not _PROFILE_ID.fullmatch(profile_id):
        errors.append(f"{source}.id: use lowercase letters, numbers, '.', '_' or '-'")
    name = _string(data.get("name"), f"{source}.name", errors)
    description = _string(data.get("description", ""), f"{source}.description", errors, required=False)
    objectives = _normalise_objectives(data.get("objectives"), errors)
    constraints = _mapping(data.get("constraints"), f"{source}.constraints", errors)
    if not constraints:
        errors.append(f"{source}.constraints: must contain at least one constraint")
    hardware = _normalise_hardware(data.get("hardware"), errors)
    safety = _normalise_safety(data.get("safety"), errors)
    precision = _normalise_precision(data.get("precision"), errors)
    required_evidence = _normalise_evidence(data.get("required_evidence"), errors)
    organization = _normalise_owner(data.get("organization"), f"{source}.organization", errors)
    project = _normalise_owner(data.get("project"), f"{source}.project", errors)
    config = _mapping(data.get("config", {}), f"{source}.config", errors, required=False)
    if "enterprise" in config:
        errors.append(f"{source}.config.enterprise: reserved; profile metadata is written by the merge helper")
    if errors:
        raise ProfileValidationError(errors, source=source)
    return {
        "schema": PROFILE_SCHEMA,
        "version": PROFILE_VERSION,
        "id": profile_id,
        "name": name,
        "description": description,
        "objectives": objectives,
        "constraints": constraints,
        "hardware": hardware,
        "safety": safety,
        "precision": precision,
        "required_evidence": required_evidence,
        "organization": organization,
        "project": project,
        "config": config,
    }


@dataclass(frozen=True)
class EnterpriseDesignProfile:
    """Validated immutable shell around a versioned enterprise profile."""

    schema: str
    version: int
    profile_id: str
    name: str
    description: str
    objectives: tuple[Dict[str, Any], ...]
    constraints: Dict[str, Any]
    hardware: Dict[str, Any]
    safety: Dict[str, Any]
    precision: Dict[str, Any]
    required_evidence: tuple[Dict[str, Any], ...]
    organization: Dict[str, Any]
    project: Dict[str, Any]
    config: Dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, source: str = "profile") -> "EnterpriseDesignProfile":
        data = _normalise_profile(raw, source=source)
        return cls(
            schema=data["schema"],
            version=data["version"],
            profile_id=data["id"],
            name=data["name"],
            description=data["description"],
            objectives=tuple(data["objectives"]),
            constraints=data["constraints"],
            hardware=data["hardware"],
            safety=data["safety"],
            precision=data["precision"],
            required_evidence=tuple(data["required_evidence"]),
            organization=data["organization"],
            project=data["project"],
            config=data["config"],
        )

    @property
    def id(self) -> str:
        """The stable profile identifier used in manifests and reports."""
        return self.profile_id

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached canonical mapping suitable for JSON/YAML output."""
        return {
            "schema": self.schema,
            "version": self.version,
            "id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "objectives": copy.deepcopy(list(self.objectives)),
            "constraints": copy.deepcopy(self.constraints),
            "hardware": copy.deepcopy(self.hardware),
            "safety": copy.deepcopy(self.safety),
            "precision": copy.deepcopy(self.precision),
            "required_evidence": copy.deepcopy(list(self.required_evidence)),
            "organization": copy.deepcopy(self.organization),
            "project": copy.deepcopy(self.project),
            "config": copy.deepcopy(self.config),
        }

    def canonical_json(self) -> str:
        return serialize_profile(self)

    @property
    def fingerprint(self) -> str:
        return profile_fingerprint(self)


def _parse_text(text: str, *, suffix: str, source: str) -> Any:
    if suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}") from exc
    try:
        import yaml  # type: ignore
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "enterprise profiles need PyYAML (pip install pyyaml) or JSON-compatible YAML"
            ) from exc
    try:
        return yaml.safe_load(text)
    except Exception as exc:  # PyYAML exposes parser/scanner exceptions without a stable common type.
        raise ValueError(f"{source}: invalid YAML: {exc}") from exc


def load_profile(source: EnterpriseDesignProfile | Mapping[str, Any] | str | Path) -> EnterpriseDesignProfile:
    """Load and validate a profile mapping or a YAML/JSON file."""
    if isinstance(source, EnterpriseDesignProfile):
        return source
    if isinstance(source, Mapping):
        return EnterpriseDesignProfile.from_mapping(source)
    if isinstance(source, Path):
        path = source
        text = path.read_text(encoding="utf-8")
        return load_profile(_parse_text(text, suffix=path.suffix, source=str(path)))
    if isinstance(source, str):
        candidate = Path(source)
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if is_file:
            return load_profile(candidate)
        looks_like_document = "\n" in source or source.lstrip().startswith(("{", "["))
        if looks_like_document:
            return load_profile(_parse_text(source, suffix=".yaml", source="profile text"))
        raise FileNotFoundError(f"enterprise profile not found: {source}")
    raise TypeError("enterprise profile must be a mapping, YAML/JSON path, or profile text")


def validate_profile(source: EnterpriseDesignProfile | Mapping[str, Any] | str | Path) -> EnterpriseDesignProfile:
    """Explicit alias for callers that want validation without discussing loading."""
    return load_profile(source)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key in sorted(overlay):
        value = overlay[key]
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _base_config(source: Mapping[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    if isinstance(source, (str, Path)):
        loaded = load_config(source)
        return copy.deepcopy(loaded)
    raise TypeError("base config must be a mapping or a nomo.yaml path")


def merge_with_base_config(
    base_config: Mapping[str, Any] | str | Path,
    profile: EnterpriseDesignProfile | Mapping[str, Any] | str | Path,
) -> Dict[str, Any]:
    """Apply profile config overrides and attach the canonical enterprise profile.

    Precedence is ``base nomo.yaml < profile.config``.  Dictionaries merge
    recursively, while lists and scalar values are replaced as complete
    values.  The profile itself always replaces ``enterprise.profile`` so the
    resulting config cannot silently describe a different design contract.
    """
    validated = load_profile(profile)
    merged = _deep_merge(_base_config(base_config), validated.config)
    existing_enterprise = merged.get("enterprise", {})
    if existing_enterprise is None:
        existing_enterprise = {}
    if not isinstance(existing_enterprise, Mapping):
        raise ValueError("base config section 'enterprise' must be a mapping when present")
    merged["enterprise"] = _deep_merge(existing_enterprise, {"profile": validated.to_dict()})
    return merged


def merge_profile(
    base_config: Mapping[str, Any] | str | Path,
    profile: EnterpriseDesignProfile | Mapping[str, Any] | str | Path,
) -> Dict[str, Any]:
    """Short alias for :func:`merge_with_base_config`."""
    return merge_with_base_config(base_config, profile)


def merge_config(
    base_config: Mapping[str, Any] | str | Path,
    profile: EnterpriseDesignProfile | Mapping[str, Any] | str | Path,
) -> Dict[str, Any]:
    """Compatibility-friendly alias for profile-aware config merging."""
    return merge_with_base_config(base_config, profile)


def load_and_merge_config(
    base_config: Mapping[str, Any] | str | Path,
    profile: EnterpriseDesignProfile | Mapping[str, Any] | str | Path,
) -> Dict[str, Any]:
    """Load both sources when paths are supplied and return the merged config."""
    return merge_with_base_config(base_config, profile)


def _json_ready(value: Any) -> Any:
    if isinstance(value, EnterpriseDesignProfile):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a profile/config/report deterministically for hashes and manifests."""
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def serialize_profile(profile: EnterpriseDesignProfile | Mapping[str, Any] | str | Path) -> str:
    """Return canonical JSON for a validated profile."""
    return canonical_json(load_profile(profile))


def profile_fingerprint(profile: EnterpriseDesignProfile | Mapping[str, Any] | str | Path) -> str:
    """Return a stable SHA-256 fingerprint of the canonical profile JSON."""
    return hashlib.sha256(serialize_profile(profile).encode("utf-8")).hexdigest()

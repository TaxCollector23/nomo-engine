"""Enterprise design profiles and evidence boundaries for Nomo.

This package is deliberately bounded: it validates a versioned team profile,
merges its explicit run configuration with an existing ``nomo.yaml``, and
produces a machine-readable evidence boundary.  It does not change the search,
export, server, or hardware-in-the-loop execution paths.
"""

from .evidence import (
    EvidenceValidationError,
    build_capability_evidence_report,
    capability_evidence_report,
)
from .profile import (
    EVIDENCE_LEVELS,
    PROFILE_SCHEMA,
    PROFILE_VERSION,
    EnterpriseDesignProfile,
    ProfileValidationError,
    canonical_json,
    load_and_merge_config,
    load_profile,
    merge_config,
    merge_profile,
    merge_with_base_config,
    profile_fingerprint,
    serialize_profile,
    validate_profile,
)

__all__ = [
    "EVIDENCE_LEVELS",
    "PROFILE_SCHEMA",
    "PROFILE_VERSION",
    "EnterpriseDesignProfile",
    "EvidenceValidationError",
    "ProfileValidationError",
    "build_capability_evidence_report",
    "capability_evidence_report",
    "canonical_json",
    "load_and_merge_config",
    "load_profile",
    "merge_config",
    "merge_profile",
    "merge_with_base_config",
    "profile_fingerprint",
    "serialize_profile",
    "validate_profile",
]

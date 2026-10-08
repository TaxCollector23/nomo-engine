"""Stable JSON schema identifiers for the bounded Nomo emulation boundary.

The emulator deliberately has its own wire format.  It is not a claim that a
JSON artifact is a vendor executable or a physical measurement.  The
``capability`` field on every evidence-bearing result is one of
``simulated``, ``measured``, or ``proxy``.
"""
from __future__ import annotations

from typing import Any, Dict


CAPABILITY_LABELS = ("simulated", "measured", "proxy")
ARTIFACT_SCHEMA = "nomo.emulation.artifact/1"
CONFIG_SCHEMA = "nomo.emulation.config/1"
RESULT_SCHEMA = "nomo.emulation.result/1"


_CAPABILITY = {
    "type": "string",
    "enum": list(CAPABILITY_LABELS),
}


_METRIC = {
    "type": "object",
    "required": ["value", "capability"],
    "properties": {
        "value": {},
        "capability": _CAPABILITY,
        "available": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "additionalProperties": False,
}


ARTIFACT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": ARTIFACT_SCHEMA,
    "title": "Nomo bounded emulation artifact",
    "type": "object",
    "required": ["schema", "graph"],
    "properties": {
        "schema": {"const": ARTIFACT_SCHEMA},
        "graph": {"type": "object"},
        "inputs": {"type": "array"},
        "aux": {"type": "array"},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}


CONFIG_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": CONFIG_SCHEMA,
    "title": "Nomo bounded emulation configuration",
    "type": "object",
    "required": ["schema"],
    "properties": {
        "schema": {"const": CONFIG_SCHEMA},
        "hardware": {"type": "object"},
        "limits": {"type": "object"},
        "trace": {"type": "object"},
        "inputs": {"type": "array"},
        "aux": {"type": "array"},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}


RESULT_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": RESULT_SCHEMA,
    "title": "Nomo bounded emulation result",
    "type": "object",
    "required": ["schema", "backend", "graph", "configuration", "summary", "layers", "trace"],
    "properties": {
        "schema": {"const": RESULT_SCHEMA},
        "backend": {
            "type": "object",
            "required": ["id", "capability", "physical_measurement"],
            "properties": {
                "id": {"type": "string"},
                "version": {"type": "string"},
                "capability": _CAPABILITY,
                "physical_measurement": {"const": False},
                "description": {"type": "string"},
                "integration_boundary": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "graph": {"type": "object"},
        "configuration": {"type": "object"},
        "summary": {"type": "object"},
        "layers": {"type": "array"},
        "vectors": {"type": "array"},
        "trace": {"type": "array"},
        "warnings": {"type": "array"},
    },
    "additionalProperties": False,
}


SCHEMAS = {
    "artifact": ARTIFACT_JSON_SCHEMA,
    "config": CONFIG_JSON_SCHEMA,
    "result": RESULT_JSON_SCHEMA,
}


def get_schema(kind: str) -> Dict[str, Any]:
    """Return a copy of the public JSON schema for ``kind``.

    A copy prevents callers from mutating the process-wide schema document and
    accidentally changing later CLI output.
    """
    try:
        import copy

        return copy.deepcopy(SCHEMAS[kind])
    except KeyError as exc:
        raise ValueError(f"unknown emulation schema {kind!r}; expected artifact, config, or result") from exc

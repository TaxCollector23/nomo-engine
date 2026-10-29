"""Declarative ``nomo.yaml`` loading for headless CI runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    try:
        import yaml  # type: ignore
        doc = yaml.safe_load(text)
    except ImportError:
        # JSON is a valid YAML subset and gives a useful fallback in minimal
        # embedded environments where PyYAML is intentionally not installed.
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("nomo.yaml needs PyYAML (pip install pyyaml) or JSON-compatible YAML") from exc
    if not isinstance(doc, dict):
        raise ValueError("nomo.yaml must contain a mapping at the top level")
    return doc


def section(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = doc.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"nomo.yaml section '{name}' must be a mapping")
    return dict(value)


def search_settings(doc: Dict[str, Any]) -> Dict[str, Any]:
    search = section(doc, "search")
    # The file uses human-facing names; CLI/API use pop_size/generations.
    if "population" in search and "pop_size" not in search:
        search["pop_size"] = search.pop("population")
    if "rounds" in search and "generations" not in search:
        search["generations"] = search.pop("rounds")
    return search

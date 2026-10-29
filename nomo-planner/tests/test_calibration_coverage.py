"""The checked-in calibration coverage document must match regeneration.

The coverage document exists so nobody mistakes a one-paper, one-cluster,
one-precision corpus for a general-purpose calibrated model.  These tests
pin the document to the code (regenerate and compare) and pin the honesty
properties that made the document worth writing.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studies.export_calibration_coverage import coverage_facts, render_markdown  # noqa: E402

DOC = ROOT / "docs" / "CALIBRATION_COVERAGE.md"


def test_document_matches_regeneration():
    assert DOC.exists(), "docs/CALIBRATION_COVERAGE.md is missing; run studies/export_calibration_coverage.py"
    assert DOC.read_text(encoding="utf-8") == render_markdown(coverage_facts())


def test_corpus_facts_are_honest():
    f = coverage_facts()
    assert f["rows"] == 22
    # One publication split across two tables; a new publication must surface in the doc.
    assert {s.rsplit("Table", 1)[0].strip() for s in f["source_counts"]} == {"Narayanan et al. 2021"}
    assert f["url"] == "https://arxiv.org/abs/2104.04473"
    assert f["clusters_calibrated"] == ["a100_nvlink_ib"]
    assert set(f["clusters_all"]) == {"h100_nvlink_ib", "a100_nvlink_ib", "l4_pcie", "mi300x_if_roce", "mi325x_if_roce", "mi355x_if_roce"}
    assert f["precisions"] == ["bf16"]
    assert f["seq_lens"] == [2048]
    assert f["model_count"] == 11
    assert f["param_min"] < 2e9 and f["param_max"] > 1e12


def test_document_discloses_limits():
    text = DOC.read_text(encoding="utf-8")
    for needle in ("h100_nvlink_ib", "l4_pcie", "10.9", "first-principles", "Out-of-distribution",
                   "bf16 only", "UNCERTAINTY_COVERAGE.md"):
        assert needle in text, f"coverage document must disclose {needle!r}"

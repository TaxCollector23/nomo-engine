"""M3 audit diffs and self-contained report renderers."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from typing import Any, Mapping

from .auditor import AuditRun


def compare_audits(current: AuditRun, candidate: AuditRun) -> dict[str, Any]:
    """Return a lossless field-by-field diff for current vs candidate config."""

    left, right = current.as_dict(), candidate.as_dict()
    changes = [
        {"field": key, "current": left.get(key), "candidate": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    ]
    return {
        "source_format": current.source_format,
        "candidate_format": candidate.source_format,
        "changes": changes,
        "current": left,
        "candidate": right,
        "warnings": sorted(set(current.warnings) | set(candidate.warnings)),
        "unknown_options": sorted(set(current.unrecognized_options) | set(candidate.unrecognized_options)),
        "same_format_candidate": current.source_format == candidate.source_format,
    }


def render_audit_html(report: Mapping[str, Any]) -> str:
    """Render an escaped HTML audit report that can be shared as one file."""

    title = html.escape(str(report.get("title", "Nomo configuration audit")))
    payload = html.escape(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{title}</title><style>body{{font:16px/1.55 system-ui,sans-serif;max-width:960px;margin:3rem auto;padding:0 1rem;color:#17212b}}"
        "pre{white-space:pre-wrap;background:#f4f6f8;padding:1rem;border-radius:8px;overflow:auto}}"
        f"</style></head><body><main><h1>{title}</h1><pre>{payload}</pre></main></body></html>"
    )


def render_audit_pdf(report: Mapping[str, Any]) -> bytes:
    """Create a minimal, valid PDF with the JSON audit report as text.

    This deliberately avoids a PDF dependency.  Long reports are wrapped into
    multiple text lines; the HTML version remains the richer shareable view.
    """

    raw = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    lines: list[str] = []
    for line in raw.splitlines():
        while len(line) > 100:
            lines.append(line[:100])
            line = line[100:]
        lines.append(line)
    lines = lines[:70]
    escape_pdf = lambda text: text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = ["BT", "/F1 8 Tf", "42 760 Td", "11 TL"]
    for line in lines:
        content.append(f"({escape_pdf(line)}) Tj")
        content.append("0 -11 Td")
    content.append("ET")
    stream = "\n".join(content).encode("ascii", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("ascii"))
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets[1:]))
    output.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(output)


__all__ = ["compare_audits", "render_audit_html", "render_audit_pdf"]

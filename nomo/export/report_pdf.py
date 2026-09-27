"""Executive audit brief (PDF) for one design (SPEC §7.6).

Plain-language first page for decision makers, technical tables after it. Every number is taken
from the same evaluation the dashboard shows; placeholder vs user-supplied hardware numbers and
proxy vs measured accuracy are stated explicitly, never implied.
"""
from __future__ import annotations

import io
import math
import time
from typing import Any, Dict, List, Optional

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

def _register_fonts() -> tuple:
    """Embed Liberation Sans (SIL OFL, metric-compatible with Helvetica) so every viewer renders the
    same glyphs, including the micro sign. Falls back to the built-in Helvetica if the files are missing."""
    import os
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    d = os.path.join(os.path.dirname(__file__), "fonts")
    try:
        pdfmetrics.registerFont(TTFont("NomoSans", os.path.join(d, "LiberationSans-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("NomoSans-Bold", os.path.join(d, "LiberationSans-Bold.ttf")))
        return "NomoSans", "NomoSans-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()
INK = colors.HexColor("#111111")
GREY = colors.HexColor("#6b6b6b")
LIGHT = colors.HexColor("#d9d9d9")
FAINT = colors.HexColor("#f2f2f2")


def _si(v: float, unit: str) -> str:
    for s, p in ((1, ""), (1e-3, "m"), (1e-6, "\u00b5"), (1e-9, "n"), (1e-12, "p")):  # micro sign: in Helvetica
        if abs(v) >= s:
            return f"{v / s:.3g} {p}{unit}"
    return f"{v:.2e} {unit}"


def _pct(new: float, old: float) -> str:
    if old == 0:
        return "n/a"
    d = 100.0 * (new - old) / old
    return f"{d:+.1f}%"


def _styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("b", parent=ss["BodyText"], fontName=FONT, fontSize=9.5, leading=13, textColor=INK)
    return {
        "title": ParagraphStyle("t", parent=base, fontName=FONT_BOLD, fontSize=20, leading=24),
        "sub": ParagraphStyle("s", parent=base, fontSize=10, textColor=GREY),
        "h": ParagraphStyle("h", parent=base, fontName=FONT_BOLD, fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4),
        "body": base,
        "small": ParagraphStyle("sm", parent=base, fontSize=8, leading=10.5, textColor=GREY),
        "cell": ParagraphStyle("c", parent=base, fontSize=8, leading=10),
    }


def _table(rows: List[List[Any]], widths: List[float], header: bool = True) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), FONT, 8), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, LIGHT),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("FONT", (0, 0), (-1, 0), FONT_BOLD, 8), ("BACKGROUND", (0, 0), (-1, 0), FAINT)]
    t.setStyle(TableStyle(st))
    return t


def _pareto_chart(points: List[Dict[str, Any]], selected: str, baseline: Optional[Dict[str, Any]], width: float = 170 * mm,
                  height: float = 80 * mm) -> Drawing:
    """Energy vs latency (log-log), shade = accuracy. Front members outlined; chosen design starred."""
    d = Drawing(width, height)
    padl, padb, padr, padt = 16 * mm, 12 * mm, 6 * mm, 9 * mm
    pw, ph = width - padl - padr, height - padb - padt
    pts = [p for p in points if p["f"][0] > 0 and p["f"][1] > 0]
    if baseline:
        pts = pts + [baseline]
    if not pts:
        return d
    lx = [math.log10(p["f"][0]) for p in pts]
    ly = [math.log10(p["f"][1]) for p in pts]
    x0, x1 = min(lx), max(lx)
    y0, y1 = min(ly), max(ly)
    x1, y1 = (x1 if x1 > x0 else x0 + 1), (y1 if y1 > y0 else y0 + 1)
    accs = [p["f"][2] for p in pts]
    a0, a1 = min(accs), max(accs) if max(accs) > min(accs) else min(accs) + 1

    def X(v):
        return padl + (math.log10(v) - x0) / (x1 - x0) * pw

    def Y(v):
        return padb + (math.log10(v) - y0) / (y1 - y0) * ph

    d.add(Rect(padl, padb, pw, ph, strokeColor=LIGHT, fillColor=None, strokeWidth=0.5))
    for k in range(5):
        tx = x0 + (x1 - x0) * k / 4
        ty = y0 + (y1 - y0) * k / 4
        d.add(String(padl + pw * k / 4, padb - 9, _si(10 ** tx, "J"), fontName=FONT, fontSize=6.5, fillColor=GREY,
                     textAnchor="middle"))
        d.add(String(padl - 2, padb + ph * k / 4 - 2, _si(10 ** ty, "s"), fontName=FONT, fontSize=6.5, fillColor=GREY,
                     textAnchor="end"))
    d.add(String(padl + pw / 2, 1, "energy per inference (log)", fontName=FONT, fontSize=7.5, fillColor=INK, textAnchor="middle"))
    d.add(String(padl, padb + ph + 3, "latency (log)", fontName=FONT, fontSize=7.5, fillColor=INK))
    for p in sorted(points, key=lambda p: p.get("front", False)):
        shade = 0.85 - 0.75 * (p["f"][2] - a0) / (a1 - a0)
        c = colors.Color(shade, shade, shade)
        r = 2.6 if p.get("front") else 1.6
        d.add(Circle(X(p["f"][0]), Y(p["f"][1]), r, fillColor=c,
                     strokeColor=INK if p.get("front") else c, strokeWidth=0.6 if p.get("front") else 0))
    sel = next((p for p in points if p["key"] == selected), None)
    if sel:
        cx, cy = X(sel["f"][0]), Y(sel["f"][1])
        d.add(Circle(cx, cy, 5.5, fillColor=None, strokeColor=INK, strokeWidth=1.2))
        d.add(String(cx + 7, cy + 3, "chosen design", fontName=FONT, fontSize=7, fillColor=INK))
    if baseline:
        bx, by = X(baseline["f"][0]), Y(baseline["f"][1])
        d.add(Rect(bx - 3, by - 3, 6, 6, fillColor=colors.white, strokeColor=INK, strokeWidth=0.8))
        d.add(String(bx + 6, by - 9, "all-continuous baseline", fontName=FONT, fontSize=7, fillColor=GREY))
    d.add(String(width - padr, height - 8, "darker = more accurate · outlined = best trade-offs",
                 fontName=FONT, fontSize=6.5, fillColor=GREY, textAnchor="end"))
    return d


def _bar(frac: float, width: float = 45 * mm, height: float = 3.2 * mm) -> Drawing:
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=FAINT, strokeColor=LIGHT, strokeWidth=0.4))
    d.add(Rect(0, 0, width * max(0.0, min(1.0, frac)), height, fillColor=INK, strokeColor=None))
    return d


def build_pdf(ctx: Dict[str, Any]) -> bytes:
    """ctx keys: model, hw, eval (selected Evaluation), baseline (Evaluation-like dict), front (list of wire
    items), cloud (list of wire items), explanation (list of str), assumptions (list of str),
    weights_source, policy (dict), generated_by."""
    S = _styles()
    ev, model, hw = ctx["eval"], ctx["model"], ctx["hw"]
    cost = ev.cost
    base = ctx["baseline"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"Nomo design brief: {model.name}", author="Nomo")
    story: List[Any] = []

    # ---------------- page 1: the decision
    story += [Paragraph("Nomo design brief", S["title"]),
              Paragraph(f"{model.name} on {hw.name} · generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", S["sub"]),
              Spacer(1, 6 * mm), Paragraph("Summary", S["h"])]
    for line in ctx.get("explanation", []):
        story.append(Paragraph(line, S["body"]))
    story.append(Spacer(1, 3 * mm))
    rows = [["", "all-continuous baseline", "chosen design", "change"],
            ["Energy per inference", _si(base["energy_j"], "J"), _si(cost.energy_j, "J"), _pct(cost.energy_j, base["energy_j"])],
            ["Latency", _si(base["latency_s"], "s"), _si(cost.latency_s, "s"), _pct(cost.latency_s, base["latency_s"])],
            ["Accuracy" + (" (estimated)" if ev.accuracy_source == "proxy" else " (measured)"),
             f"{base['accuracy']:.2f} %", f"{ev.accuracy:.2f} %", f"{ev.accuracy - base['accuracy']:+.2f} pts"],
            ["Meets all budgets", "", "yes" if ev.feasible else f"no (violation {ev.cv:.2f})", ""]]
    story.append(_table(rows, [45 * mm, 42 * mm, 42 * mm, 30 * mm]))
    story += [Spacer(1, 4 * mm), Paragraph("Trade-off map", S["h"]),
              _pareto_chart(ctx["cloud"], ev.key, {"key": "_base", "f": [base["energy_j"], base["latency_s"], base["accuracy"]]}),
              Paragraph("Each dot is a design the search evaluated. Outlined dots are the best trade-offs (no other design is "
                        "better on energy, latency and accuracy at once). The circle marks the chosen design.", S["small"])]

    # ---------------- page 2: the design
    story += [PageBreak(), Paragraph("Layer-by-layer design", S["h"])]
    dom_name = {"ANN": "continuous", "SNN": "spiking", "SYM": "physics formula"}
    stage_by_name = {s.name: s for s in cost.stages if s.kind == "layer"}
    rows = [["#", "layer", "type", "runs as", "precision", "spike coding", "energy share", "memory", "cores"]]
    for i, (spec, gene) in enumerate(zip(model.layers, ev.genome.layers)):
        st = stage_by_name.get(spec.name)
        dom = ["ANN", "SNN", "SYM"][int(gene.domain)]
        prec = "Q16.16" if dom == "SYM" else (f"w{gene.w_bits} / a{gene.a_bits}" if dom == "ANN" else f"w{gene.w_bits} / v{gene.a_bits}")
        coding = f"{'TTFS' if int(gene.coding) == 2 else 'rate'}, T={gene.timesteps}" + (" · learns on-chip" if gene.plastic else "") if dom == "SNN" else "-"
        rows.append([str(i), spec.name, spec.op, dom_name[dom], prec, coding,
                     f"{100 * st.energy_j / max(cost.energy_j, 1e-30):.1f} %" if st else "-",
                     f"{st.mem_bytes / 1024:.1f} KB" if st else "-", str(st.cores) if st and st.cores else "-"])
    story.append(_table(rows, [7 * mm, 26 * mm, 14 * mm, 22 * mm, 20 * mm, 30 * mm, 18 * mm, 17 * mm, 11 * mm]))
    if model.guard_sites:
        g = model.guard_sites[0]
        story.append(Paragraph(f"Safety guard: <b>{g.constraint_id}</b> is enforced on the output of layer "
                               f"{model.layers[g.after_layer].name} in every candidate (mandatory).", S["small"]))

    story += [Paragraph("Chip utilisation", S["h"])]
    rows = [["resource", "used", "capacity", ""]]
    if not hw.snn_dense:
        rows.append(["neuromorphic cores", str(cost.cores_used), str(hw.n_cores), _bar(cost.cores_used / max(1, hw.n_cores))])
    for unit, used in cost.unit_memory.items():
        capv = cost.unit_capacity.get(unit, 0)
        rows.append([f"memory on {unit}", f"{used / 1024:.1f} KB", f"{capv / 1024:.0f} KB", _bar(used / capv if capv else 0)])
    rows.append(["streaming frame period", _si(cost.frame_period_s, "s"), "", ""])
    story.append(_table(rows, [48 * mm, 32 * mm, 32 * mm, 50 * mm]))

    story += [Paragraph("Where the energy goes", S["h"])]
    parts = [(f"{k} compute", v) for k, v in cost.energy_by_domain.items()]
    parts += [("domain crossings", cost.crossing_energy_j), ("always-on (static) power", cost.static_energy_j)]
    rows = [["component", "energy", "share", ""]] + [
        [k, _si(v, "J"), f"{100 * v / max(cost.energy_j, 1e-30):.1f} %", _bar(v / max(cost.energy_j, 1e-30))] for k, v in parts]
    story.append(_table(rows, [58 * mm, 30 * mm, 22 * mm, 52 * mm]))
    if cost.crossings:
        rows = [["crossing", "values", "timesteps", "payload", "energy", "time"]] + [
            [f"{c.src} → {c.dst}", str(c.values), str(c.timesteps or "-"), f"{c.payload_bytes:.0f} B ({c.payload_kind})",
             _si(c.conversion_energy_j + c.transfer_energy_j, "J"), _si(c.conversion_s + c.transfer_s, "s")]
            for c in cost.crossings]
        story += [Paragraph("Domain crossings", S["h"]), _table(rows, [28 * mm, 18 * mm, 18 * mm, 40 * mm, 26 * mm, 26 * mm])]

    # ---------------- page 3: alternatives + assumptions
    story += [PageBreak(), Paragraph("Other best trade-offs found", S["h"])]
    rows = [["energy", "latency", "accuracy", "design (layer codes)"]]
    for it in sorted(ctx["front"], key=lambda it: it["f"][0])[:14]:
        mark = "  (chosen)" if it["key"] == ev.key else ""
        rows.append([_si(it["f"][0], "J"), _si(it["f"][1], "s"), f"{it['f'][2]:.2f} %",
                     Paragraph(it["key"].split("#")[0] + mark, S["cell"])])
    story.append(_table(rows, [24 * mm, 24 * mm, 20 * mm, 102 * mm]))
    story.append(Paragraph("Codes: A = continuous, S = spiking (R rate / T time-to-first-spike, then timesteps; p = learns "
                           "on-chip), Y = physics formula. Numbers after the letter are weight.activation bits.", S["small"]))
    story += [Paragraph("Assumptions and data provenance", S["h"])]
    user_fields = sorted(k for k, v in hw.provenance.items() if v == "user")
    prov = ("All chip coefficients are placeholders, not measurements." if not user_fields else
            f"User-supplied chip parameters: {', '.join(user_fields)}. All other coefficients are placeholders.")
    notes = [prov,
             "Accuracy is " + ("estimated from calibrated per-layer sensitivities, not measured on a test set."
                               if ev.accuracy_source == "proxy" else "measured by the oracle evaluation."),
             f"Weights: {ctx.get('weights_source', 'unknown')}."] + list(ctx.get("assumptions", []))
    for n in notes:
        story.append(Paragraph("• " + n, S["body"]))
    if ctx.get("policy"):
        p = ctx["policy"]
        story.append(Paragraph(f"Search settings: domains allowed {', '.join(p.get('allow', []))}; spike codings "
                               f"{', '.join(p.get('codings', []))}; locked layers {', '.join(p.get('pins', {}).keys()) or 'none'}.",
                               S["body"]))
    story += [Spacer(1, 4 * mm), Paragraph(f"Design key: {ev.key}", S["small"]),
              Paragraph(ctx.get("generated_by", "Generated by Nomo"), S["small"])]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(FONT, 7)
        canvas.setFillColor(GREY)
        canvas.drawString(18 * mm, 9 * mm, f"Nomo · {model.name} · {hw.name}")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()

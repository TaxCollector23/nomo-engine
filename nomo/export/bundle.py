"""One-click export bundle (SPEC §7.7).

`capabilities(ctx, ev)` states, per format, whether this design can be exported and why not in plain
language. `build_bundle(ctx, ev, formats)` returns a zip:

    nomo_<model>_<id>/
      README.md                 what every file is and how to use it (plain English)
      design.json               the design, metrics, settings and provenance (machine-readable)
      report.pdf                executive audit brief
      pytorch/                  deploy_model.py + deploy_weights.npz  (runs with or without PyTorch)
      onnx/                     one .onnx per continuous section (+ safety guard when inside it)
      coreml/                   one zipped .mlpackage per continuous section
      nir/                      float NIR (strict + extended); integer NIR when the C compiler supports the design
      c11/                      header-only nomo_model.h + example main (dense designs; bit-exact integer code)
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .. import __version__
from ..hardware.profiles import SiliconProfile
from ..ir import ModelGraph
from ..search.evaluator import Evaluation, NeurosymbolicEvaluator
from ..search.genome import Coding, Domain

FORMATS = ("design", "pdf", "pytorch", "onnx", "coreml", "nir", "c11")
FORMAT_LABELS = {
    "design": "Design summary (JSON + README)",
    "pdf": "Executive PDF brief",
    "pytorch": "PyTorch script (deploy_model.py)",
    "onnx": "ONNX (TensorRT / ONNX Runtime)",
    "coreml": "Core ML package (Apple devices)",
    "nir": "NIR graph (neuromorphic interchange)",
    "c11": "Bare-metal C11 header (microcontrollers)",
}


@dataclass
class RunContext:
    model: ModelGraph
    weights: Dict[str, Tuple[np.ndarray, np.ndarray]]
    weights_source: str
    hw: SiliconProfile
    evaluator: NeurosymbolicEvaluator
    settings: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    front_fn: Callable[[], List[Evaluation]] = lambda: []
    recommend_fn: Callable[[], Optional[Evaluation]] = lambda: None
    _plans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _calib: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None

    def calib(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._calib is None:
            from .plan import calibration_inputs
            n = 8 if any(l.op == "conv2d" for l in self.model.layers) else 64
            self._calib = calibration_inputs(self.model, n, seed=0)
        return self._calib

    def plan(self, ev: Evaluation) -> Dict[str, Any]:
        if ev.key not in self._plans:
            from .plan import build_plan
            X, A = self.calib()
            self._plans[ev.key] = build_plan(self.model, self.weights, ev.genome, X, A)
            if len(self._plans) > 16:
                self._plans.pop(next(iter(self._plans)))
        return self._plans[ev.key]


# ---------------------------------------------------------------------------
# capability checks
# ---------------------------------------------------------------------------

def _c11_reason(ctx: RunContext, ev: Evaluation) -> Optional[str]:
    if any(l.op != "dense" for l in ctx.model.layers):
        return "the C code generator handles fully-connected layers only; this model has convolutions"
    for spec, g in zip(ctx.model.layers, ev.genome.layers):
        if g.domain == Domain.ANN and (g.a_bits != 8 or g.w_bits > 8):
            return f"layer '{spec.name}' uses {g.w_bits}/{g.a_bits}-bit precision; the C generator needs 8-bit activations"
        if g.domain == Domain.SNN and g.coding == Coding.TTFS:
            return f"layer '{spec.name}' uses time-to-first-spike coding, which the C generator does not support yet"
        if g.domain == Domain.SNN and (g.w_bits > 8 or g.a_bits > 24):
            return f"layer '{spec.name}' precision exceeds the C generator's limits"
    return None


def capabilities(ctx: RunContext, ev: Evaluation) -> Dict[str, Dict[str, Any]]:
    from .onnx_coreml import continuous_sections
    from .nir_float import has_max_pool
    plan = ctx.plan(ev)
    secs = continuous_sections(plan)
    has_snn = any(l["domain"] == "SNN" for l in plan["layers"])
    out: Dict[str, Dict[str, Any]] = {k: {"label": FORMAT_LABELS[k], "available": True, "note": ""} for k in FORMATS}
    if ctx.weights_source.startswith("synthetic"):
        for k in ("pytorch", "onnx", "coreml", "nir", "c11"):
            out[k]["note"] = "uses untrained demo weights (upload your own model for real weights)"
    if not secs:
        for k in ("onnx", "coreml"):
            out[k].update(available=False, reason="every layer in this design is spiking; ONNX and Core ML only describe "
                                                  "continuous networks (use NIR or the PyTorch script)")
    elif has_snn:
        for k in ("onnx", "coreml"):
            out[k]["note"] = (out[k]["note"] + "; " if out[k]["note"] else "") + \
                f"exports the {len(secs)} continuous section(s); spiking parts are in the NIR / PyTorch exports"
    import os
    if os.environ.get("NOMO_DISABLE_COREML") == "1":
        out["coreml"].update(available=False, reason="Core ML export is switched off on this server to save memory")
    else:
        try:
            import coremltools  # noqa: F401
        except Exception:
            out["coreml"].update(available=False, reason="Core ML tools are not installed on this server")
    if has_max_pool(plan):
        out["nir"]["note"] = "max pooling has no standard NIR primitive: only the extended Nomo NIR file is produced"
    reason = _c11_reason(ctx, ev)
    if reason:
        out["c11"].update(available=False, reason=reason)
    return out


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def design_json(ctx: RunContext, ev: Evaluation) -> Dict[str, Any]:
    from ..copilot import baseline
    stages = {s.name: s for s in ev.cost.stages if s.kind == "layer"}
    layers = []
    for spec, g in zip(ctx.model.layers, ev.genome.layers):
        st = stages.get(spec.name)
        layers.append({"name": spec.name, "op": spec.op, "domain": g.domain.name, "w_bits": g.w_bits, "a_bits": g.a_bits,
                       "coding": g.coding.name if g.domain == Domain.SNN else None,
                       "timesteps": g.timesteps or None, "plastic": g.plastic,
                       "energy_j": st.energy_j if st else None, "memory_bytes": st.mem_bytes if st else None,
                       "cores": st.cores if st else None, "params": spec.params, "macs": spec.macs})
    return {
        "format": "nomo.design/1", "nomo_version": __version__, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": ctx.model.name, "hardware": {"id": ctx.hw.id, "name": ctx.hw.name, "provenance": ctx.hw.provenance},
        "design_key": ev.key, "genome": ev.genome.to_wire(),
        "metrics": {"energy_j": ev.cost.energy_j, "latency_s": ev.cost.latency_s, "accuracy_pct": ev.accuracy,
                    "accuracy_source": ev.accuracy_source, "feasible": ev.feasible, "constraint_violation": ev.cv,
                    "domain_crossings": len(ev.cost.crossings), "cores_used": ev.cost.cores_used,
                    "frame_period_s": ev.cost.frame_period_s, "memory_bytes": ev.cost.memory_bytes},
        "baseline_all_continuous": baseline(ctx), "layers": layers,
        "guards": [{"after_layer": ctx.model.layers[s.after_layer].name, "constraint": s.constraint_id} for s in ctx.model.guard_sites],
        "settings": ctx.settings, "weights_source": ctx.weights_source, "assumptions": ctx.assumptions,
    }


def _readme(ctx: RunContext, ev: Evaluation, written: Dict[str, List[str]], caps: Dict[str, Dict[str, Any]],
            summary: str, skipped: Dict[str, str]) -> str:
    L = [f"# Nomo export: {ctx.model.name} on {ctx.hw.name}", "", summary, "",
         f"Design: `{ev.key}`", "", "## What's in this folder", ""]
    howto = {
        "design": "`design.json`: every setting and number for this design, machine-readable.",
        "pdf": "`report.pdf`: a short brief for decision makers (open it first).",
        "pytorch": "`pytorch/deploy_model.py`: run `python deploy_model.py --check` to confirm your computer reproduces "
                   "the results, then `--input x.npy --output y.npy`. Works without PyTorch (numpy fallback); with "
                   "PyTorch, `build_torch_model()` gives an nn.Module.",
        "onnx": "`onnx/*.onnx`: standard ONNX files for ONNX Runtime or TensorRT. Guarded models take a second input `aux`.",
        "coreml": "`coreml/*.mlpackage.zip`: unzip and drag into Xcode (needs macOS/iOS to run).",
        "nir": "`nir/`: neuromorphic interchange files. `*_float.nir` loads in any NIR tool; `*.nomo.nir` keeps Nomo's "
               "extra node types; `int/` (when present) is the bit-exact integer version.",
        "c11": "`c11/nomo_model.h`: one header, no dependencies. In one .c file: `#define NOMO_MODEL_IMPLEMENTATION` "
               "then `#include \"nomo_model.h\"`, and call `nomo_infer(...)`. See `c11/example_main.c`.",
    }
    for k, files in written.items():
        note = caps.get(k, {}).get("note")
        L.append(f"- {howto[k]}" + (f" Note: {note}." if note else ""))
    if skipped:
        L += ["", "## Not included", ""] + [f"- {FORMAT_LABELS[k]}: {r}." for k, r in skipped.items()]
    L += ["", "## Please read", "",
          "- Chip energy and speed numbers are estimates from placeholder coefficients unless you entered your own.",
          "- Accuracy is " + ("estimated, not measured on your data." if ev.accuracy_source == "proxy" else "measured."),
          f"- Weights: {ctx.weights_source}."] + [f"- {a}" for a in ctx.assumptions]
    return "\n".join(L) + "\n"


def build_bundle(ctx: RunContext, ev: Evaluation, formats: List[str]) -> Tuple[bytes, Dict[str, Any]]:
    from ..copilot import summarize
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown export format(s): {unknown}")
    caps = capabilities(ctx, ev)
    plan = ctx.plan(ev)
    root = f"nomo_{ctx.model.name}_{hashlib.sha1(ev.key.encode()).hexdigest()[:8]}"
    written: Dict[str, List[str]] = {}
    skipped: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)

    def put(fmt: str, rel: str, data) -> None:
        zf.writestr(f"{root}/{rel}", data)
        written.setdefault(fmt, []).append(rel)

    summary = summarize(ctx, ev).text
    for fmt in formats:
        if fmt == "design":
            continue                                   # written last, with the README
        if not caps[fmt]["available"]:
            skipped[fmt] = caps[fmt].get("reason", "not available")
            continue
        try:
            if fmt == "pdf":
                from ..copilot import baseline
                from .report_pdf import build_pdf
                front = [e.to_wire() for e in ctx.front_fn()]
                fk = {f["key"] for f in front}
                cloud = [dict(e.to_wire(), front=e.key in fk) for e in list(ctx.evaluator.cache.values())[:3000] if e.feasible]
                put(fmt, "report.pdf", build_pdf({
                    "model": ctx.model, "hw": ctx.hw, "eval": ev, "baseline": baseline(ctx), "front": front, "cloud": cloud,
                    "explanation": [summary], "assumptions": ctx.assumptions, "weights_source": ctx.weights_source,
                    "policy": ctx.settings.get("policy"), "generated_by": f"Generated by Nomo {__version__}"}))
            elif fmt == "pytorch":
                from .plan import run_plan
                from .torch_script import generate
                X, A = ctx.calib()
                rx, ra = X[:4], (A[:4] if A is not None else None)
                src, npz = generate(plan, rx, ra, run_plan(plan, rx, ra), f"Nomo design for {ctx.model.name} on {ctx.hw.name}",
                                    f"Weights: {ctx.weights_source}.")
                put(fmt, "pytorch/deploy_model.py", src)
                put(fmt, "pytorch/deploy_weights.npz", npz)
            elif fmt in ("onnx", "coreml"):
                from .onnx_coreml import continuous_sections, to_coreml_package, to_onnx
                for s, e in continuous_sections(plan):
                    nm = f"{ctx.model.name}_layers_{s}-{e}"
                    if fmt == "onnx":
                        put(fmt, f"onnx/{nm}.onnx", to_onnx(plan, s, e, nm).SerializeToString())
                    else:
                        put(fmt, f"coreml/{nm}.mlpackage.zip", to_coreml_package(plan, s, e, nm))
            elif fmt == "nir":
                import tempfile
                from pathlib import Path

                import nir as nirlib
                from .nir_export import export_all, to_nir_strict, write_extended
                from .nir_float import has_max_pool, plan_to_nomo_graph
                g = plan_to_nomo_graph(plan)
                with tempfile.TemporaryDirectory() as d:
                    if not has_max_pool(plan):
                        nirlib.write(f"{d}/f.nir", to_nir_strict(g))
                        put(fmt, f"nir/{ctx.model.name}_float.nir", Path(f"{d}/f.nir").read_bytes())
                    write_extended(f"{d}/e.nir", g)
                    put(fmt, f"nir/{ctx.model.name}_float.nomo.nir", Path(f"{d}/e.nir").read_bytes())
                    if _c11_reason(ctx, ev) is None:
                        qg = _qgraph(ctx, ev)
                        paths = export_all(qg, f"{d}/int", genome_wire=ev.genome.to_wire())
                        for p in paths.values():
                            put(fmt, f"nir/int/{Path(p).name}", Path(p).read_bytes())
            elif fmt == "c11":
                from .c11 import emit_c11, emit_c11_single_header
                qg = _qgraph(ctx, ev)
                put(fmt, "c11/nomo_model.h", emit_c11_single_header(qg))
                put(fmt, "c11/nomo_model.manifest.json", json.dumps(emit_c11(qg).manifest, indent=2))
                put(fmt, "c11/example_main.c", _c_example(qg))
        except Exception as exc:  # one format failing must not lose the others
            errors[fmt] = f"{type(exc).__name__}: {exc}"
            skipped[fmt] = f"export failed ({type(exc).__name__}: {exc})"
    put("design", "design.json", json.dumps(design_json(ctx, ev), indent=2, default=_jd))
    put("design", "README.md", _readme(ctx, ev, written, caps, summary, skipped))
    zf.close()
    return buf.getvalue(), {"root": root, "files": written, "skipped": skipped, "errors": errors}


def _qgraph(ctx: RunContext, ev: Evaluation):
    from ..runtime.quantize import compile_qgraph
    X, A = ctx.calib()
    return compile_qgraph(ctx.model, ctx.weights, ev.genome, X, A)


def _c_example(qg) -> str:
    return f"""/* Example: compile with  cc -std=c11 -O2 example_main.c -o demo  */
#define NOMO_MODEL_IMPLEMENTATION
#include "nomo_model.h"
#include <stdio.h>

int main(void) {{
  int8_t in[NOMO_IN_SIZE] = {{0}};        /* quantised input: value / {qg.in_scale:.6g}, rounded, clamped to [-128, 127] */
  int32_t aux[NOMO_AUX_SIZE] = {{0}};     /* guard inputs in Q16.16 (value * 65536), e.g. body rates, air density ratio */
  int32_t out[NOMO_OUT_SIZE];            /* outputs in Q16.16 */
{"  aux[NOMO_AUX_SIZE - 1] = 65536;       /* density ratio 1.0 */" if qg.n_aux else ""}
  nomo_infer(in, aux, out);
  for (int i = 0; i < NOMO_OUT_SIZE; ++i) printf("out[%d] = %f\\n", i, out[i] / 65536.0);
  return 0;
}}
"""


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, frozenset):
        return sorted(str(x) for x in o)
    return str(o)

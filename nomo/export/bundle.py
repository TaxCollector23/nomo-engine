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
import tarfile
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

FORMATS = ("design", "pdf", "pytorch", "onnx", "coreml", "nir", "c11", "enterprise")
FORMAT_LABELS = {
    "design": "Design summary (JSON + README)",
    "pdf": "Executive PDF brief",
    "pytorch": "PyTorch script (deploy_model.py)",
    "onnx": "ONNX (TensorRT / ONNX Runtime)",
    "coreml": "Core ML package (Apple devices)",
    "nir": "NIR graph (neuromorphic interchange)",
    "c11": "Bare-metal C11 header (microcontrollers)",
    "enterprise": "Structured enterprise release package",
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
    calibration_data: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
    calibration_report: Optional[Dict[str, Any]] = None
    hitl_measurement: Optional[Dict[str, Any]] = None

    def calib(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._calib is None:
            if self.calibration_data is not None:
                self._calib = self.calibration_data
            else:
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

    def ptq_report_for(self, ev: Evaluation) -> Optional[Dict[str, Any]]:
        """Return selected activation ranges when uploaded calibration data exists."""
        if self.calibration_data is None:
            return self.calibration_report
        if self.calibration_report is not None and self.calibration_report.get("edge_amax"):
            return self.calibration_report
        from ..runtime.ptq import calibrate
        X, A = self.calib()
        try:
            report = calibrate(self.model, self.weights, ev.genome, X, A, method="both")
        except Exception as exc:
            # The upload itself is still valuable for report provenance even
            # when a backend-specific operator has no integer lowering yet.
            report = dict(self.calibration_report or {})
            report.update({"format": "nomo.ptq/1", "status": "unavailable",
                           "reason": f"{type(exc).__name__}: {exc}"})
        self.calibration_report = report
        return report


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
    operator_layers = [l for l in plan["layers"] if l.get("op") not in ("dense", "conv2d")]
    if operator_layers:
        reason = "operator-family blocks retain spatial/token geometry but need a specialized ONNX/Core ML lowering"
        for k in ("onnx", "coreml"):
            out[k].update(available=False, reason=reason)
        out["c11"].update(available=False, reason="the C11 lowering is limited to dense integer kernels")
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
            # coremltools can import on Linux while its native ML Program
            # storage extension is absent.  Probe the actual writer used by
            # `.mlpackage` generation so capability reporting stays truthful.
            from coremltools.libmilstorage import BlobWriter  # noqa: F401
        except Exception:
            out["coreml"].update(available=False,
                                  reason="Core ML export needs coremltools' native ML storage extension (use a supported macOS environment)")
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
    from ..architecture import classify
    from ..modes import mode_constraints
    stages = {s.name: s for s in ev.cost.stages if s.kind == "layer"}
    layers = []
    for spec, g in zip(ctx.model.layers, ev.genome.layers):
        st = stages.get(spec.name)
        layers.append({"name": spec.name, "op": spec.op, "domain": g.domain.name, "w_bits": g.w_bits, "a_bits": g.a_bits,
                       "coding": g.coding.name if g.domain == Domain.SNN else None,
                       "timesteps": g.timesteps or None, "plastic": g.plastic,
                       "energy_j": st.energy_j if st else None, "memory_bytes": st.mem_bytes if st else None,
                       "cores": st.cores if st else None, "params": spec.params, "macs": spec.macs})
    ctx.ptq_report_for(ev)
    try:
        from ..validation import build_sensitivity_report
        evidence = build_sensitivity_report(ctx, ev)
    except Exception as exc:  # evidence must never prevent the core design manifest
        evidence = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "format": "nomo.design/1", "nomo_version": __version__, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": ctx.model.name, "hardware": {"id": ctx.hw.id, "name": ctx.hw.name, "provenance": ctx.hw.provenance},
        "architecture": classify(ctx.model).to_dict(), "operational_mode": mode_constraints(ctx.settings.get("mode")),
        "design_key": ev.key, "genome": ev.genome.to_wire(),
        "metrics": {"energy_j": ev.cost.energy_j, "latency_s": ev.cost.latency_s, "accuracy_pct": ev.accuracy,
                    "accuracy_source": ev.accuracy_source, "feasible": ev.feasible, "constraint_violation": ev.cv,
                    "domain_crossings": len(ev.cost.crossings), "cores_used": ev.cost.cores_used,
                    "frame_period_s": ev.cost.frame_period_s, "memory_bytes": ev.cost.memory_bytes},
        "baseline_all_continuous": baseline(ctx), "layers": layers,
        "guards": [{"after_layer": ctx.model.layers[s.after_layer].name, "constraint": s.constraint_id} for s in ctx.model.guard_sites],
        "settings": ctx.settings, "weights_source": ctx.weights_source,
        "weights_provenance": {"synthetic_or_demo": ctx.weights_source.lower().startswith("synthetic"),
                                "source": ctx.weights_source},
        "calibration": ctx.calibration_report or {"source": "representative generated tensors", "uploaded": False},
        "evidence": evidence, "assumptions": ctx.assumptions,
    }


def _readme(ctx: RunContext, ev: Evaluation, written: Dict[str, List[str]], caps: Dict[str, Dict[str, Any]],
            summary: str, skipped: Dict[str, str]) -> str:
    L = [f"# Nomo export: {ctx.model.name} on {ctx.hw.name}", "", summary, "",
         f"Design: `{ev.key}`", "", "## What's in this folder", ""]
    howto = {
        "package": "`model/`, `runtime/`, `silicon_eda/`, `software_sdk/`, and `validation/` are the canonical structured release tree; legacy format folders are retained for compatibility.",
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
    L += ["", "## Evidence boundaries", "",
          "- `validation/sensitivity_report.json` separates zero-shot sensitivity proxies from measured calibration-runtime fidelity.",
          "- A visible DEMO WEIGHTS disclaimer is included when weights were generated or bundled by Nomo rather than uploaded.",
          "- TTFS spike sparsity and `ops_saved` are computed from the selected calibration tensors; they are not silicon power measurements.",
          "", "## Please read", "",
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
    pdf_blob: Optional[bytes] = None
    torch_src: Optional[str] = None
    torch_npz: Optional[bytes] = None
    onnx_blob: Optional[bytes] = None
    nir_blob: Optional[bytes] = None
    c11_header: Optional[str] = None
    c11_source: Optional[str] = None
    c11_manifest: Optional[Dict[str, Any]] = None
    qg = None
    integer_src: Optional[str] = None
    integer_npz: Optional[bytes] = None
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)

    def put(fmt: str, rel: str, data) -> None:
        zf.writestr(f"{root}/{rel}", data)
        written.setdefault(fmt, []).append(rel)

    summary = summarize(ctx, ev).text
    for fmt in formats:
        if fmt in ("design", "enterprise"):
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
                from ..validation import build_sensitivity_report
                evidence = build_sensitivity_report(ctx, ev)
                pdf_blob = build_pdf({
                    "model": ctx.model, "hw": ctx.hw, "eval": ev, "baseline": baseline(ctx), "front": front, "cloud": cloud,
                    "explanation": [summary], "assumptions": ctx.assumptions, "weights_source": ctx.weights_source,
                    "policy": ctx.settings.get("policy"), "generated_by": f"Generated by Nomo {__version__}",
                    "evidence": evidence})
                put(fmt, "report.pdf", pdf_blob)
            elif fmt == "pytorch":
                from .plan import run_plan
                from .torch_script import generate
                X, A = ctx.calib()
                rx, ra = X[:4], (A[:4] if A is not None else None)
                torch_src, torch_npz = generate(plan, rx, ra, run_plan(plan, rx, ra), f"Nomo design for {ctx.model.name} on {ctx.hw.name}",
                                                f"Weights: {ctx.weights_source}.")
                put(fmt, "pytorch/deploy_model.py", torch_src)
                put(fmt, "pytorch/deploy_weights.npz", torch_npz)
            elif fmt in ("onnx", "coreml"):
                from .onnx_coreml import continuous_sections, to_coreml_package, to_onnx
                for s, e in continuous_sections(plan):
                    nm = f"{ctx.model.name}_layers_{s}-{e}"
                    if fmt == "onnx":
                        blob = to_onnx(plan, s, e, nm).SerializeToString()
                        onnx_blob = onnx_blob or blob
                        put(fmt, f"onnx/{nm}.onnx", blob)
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
                        nir_blob = Path(f"{d}/f.nir").read_bytes()
                        put(fmt, f"nir/{ctx.model.name}_float.nir", nir_blob)
                    write_extended(f"{d}/e.nir", g)
                    put(fmt, f"nir/{ctx.model.name}_float.nomo.nir", Path(f"{d}/e.nir").read_bytes())
                    if _c11_reason(ctx, ev) is None:
                        qg = qg or _qgraph(ctx, ev)
                        paths = export_all(qg, f"{d}/int", genome_wire=ev.genome.to_wire())
                        for p in paths.values():
                            put(fmt, f"nir/int/{Path(p).name}", Path(p).read_bytes())
            elif fmt == "c11":
                from .c11 import emit_c11, emit_c11_single_header
                qg = qg or _qgraph(ctx, ev)
                c11_header = emit_c11_single_header(qg)
                emission = emit_c11(qg)
                c11_source = emission.source
                c11_manifest = emission.manifest
                put(fmt, "c11/nomo_model.h", c11_header)
                put(fmt, "c11/nomo_model.c", c11_source)
                put(fmt, "c11/nomo_model.manifest.json", json.dumps(c11_manifest, indent=2))
                put(fmt, "c11/example_main.c", _c_example(qg))
        except Exception as exc:  # one format failing must not lose the others
            errors[fmt] = f"{type(exc).__name__}: {exc}"
            skipped[fmt] = f"export failed ({type(exc).__name__}: {exc})"
    # The canonical runtime prefers the fully integer QGraph.  If this design
    # has an operator without an integer lowering, the float reference driver
    # remains available with an explicit note in the manifest.
    try:
        qg = qg or _qgraph(ctx, ev)
        from .integer_script import generate as generate_integer
        from .plan import run_plan
        from ..runtime.qgraph import run as run_qgraph
        from ..runtime.quantize import quantize_aux, quantize_input
        X, A = ctx.calib()
        Xi = np.stack([quantize_input(qg, x) for x in X[:4]])
        Ai = np.array([quantize_aux(a) for a in A[:4]], dtype=np.int32) if A is not None else None
        Yi = np.stack([run_qgraph(qg, x, None if Ai is None else Ai[i]) for i, x in enumerate(Xi)])
        integer_src, integer_npz = generate_integer(qg, Xi.astype(np.int8), Ai, Yi)
    except Exception as exc:
        errors.setdefault("integer_runtime", f"{type(exc).__name__}: {exc}")

    manifest = design_json(ctx, ev)
    design_blob = json.dumps(manifest, indent=2, default=_jd)
    put("design", "design.json", design_blob)

    # Canonical structured package.  The direct legacy paths above remain so
    # existing consumers do not break while the README points new consumers at
    # this release tree.
    put("package", "manifest.json", json.dumps({"format": "nomo.export_bundle/1", "design": manifest,
                                                  "skipped": skipped, "errors": errors}, indent=2, default=_jd))
    put("package", "model/weights.npz", _weights_npz(ctx.weights))
    put("package", "model/design.json", design_blob)
    if nir_blob is not None:
        put("package", "model/graph.nir", nir_blob)
    elif "nir" in skipped:
        put("package", "model/graph.nir.unavailable.txt", skipped["nir"])
    if onnx_blob is not None:
        put("package", "model/model_quantized.onnx", onnx_blob)
    elif "onnx" in skipped:
        put("package", "model/model_quantized.onnx.unavailable.txt", skipped["onnx"])
    if torch_src is not None and torch_npz is not None:
        put("package", "runtime/deploy_model_float.py", torch_src)
        put("package", "runtime/deploy_weights_float.npz", torch_npz)
    if integer_src is not None and integer_npz is not None:
        put("package", "runtime/deploy_model.py", integer_src)
        put("package", "runtime/deploy_integer_weights.npz", integer_npz)
        put("package", "software_sdk/deploy_model.py", integer_src)
        put("package", "software_sdk/deploy_integer_weights.npz", integer_npz)
    elif torch_src is not None and torch_npz is not None:
        put("package", "runtime/deploy_model.py", torch_src)
        put("package", "runtime/deploy_weights.npz", torch_npz)
        put("package", "software_sdk/deploy_model.py", torch_src)
    if c11_header is not None and c11_source is not None:
        put("package", "runtime/c11_microkernel/include/kernel.h", c11_header.replace("nomo_model.h", "kernel.h"))
        put("package", "runtime/c11_microkernel/src/kernel.c", c11_source.replace('"nomo_model.h"', '"kernel.h"'))
        put("package", "software_sdk/c11_microkernel/include/kernel.h", c11_header.replace("nomo_model.h", "kernel.h"))
        put("package", "software_sdk/c11_microkernel/src/kernel.c", c11_source.replace('"nomo_model.h"', '"kernel.h"'))
        put("package", "runtime/c11_microkernel/manifest.json", json.dumps(c11_manifest or {}, indent=2))
    put("package", "runtime/CMakeLists.txt", _cmake_manifest())
    from .rtl import emit_chisel, emit_custom_core, emit_eda_scripts, emit_pe_array, proxy_ppa
    first_dense = next((s for s in (qg.stages if qg is not None else []) if hasattr(s, "W")), None)
    rtl_in = int(first_dense.W.shape[1]) if first_dense is not None else int(ctx.model.input_size())
    rtl_out = int(first_dense.W.shape[0]) if first_dense is not None else int(ctx.model.layers[0].out_neurons)
    rtl_w = first_dense.W.ravel().tolist() if first_dense is not None else None
    rtl_b = first_dense.b.ravel().tolist() if first_dense is not None else None
    put("package", "silicon_eda/rtl/pe_array.sv", emit_pe_array(rtl_in, rtl_out, rtl_w, rtl_b))
    put("package", "silicon_eda/rtl/custom_npu_core.sv", emit_custom_core(rtl_in, rtl_out))
    put("package", "silicon_eda/chisel/CustomNpuCore.scala", emit_chisel(rtl_in, rtl_out))
    for name, source in emit_eda_scripts().items():
        put("package", f"silicon_eda/eda_scripts/{name}", source)
    put("package", "silicon_eda/ppa_report.json", json.dumps(proxy_ppa(ctx.model, ev), indent=2))
    try:
        from ..validation import build_sensitivity_report
        evidence = build_sensitivity_report(ctx, ev)
    except Exception as exc:
        evidence = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    put("package", "validation/sensitivity_report.json", json.dumps(evidence, indent=2, default=_jd))
    put("package", "validation/closed_loop_telemetry.json", json.dumps(
        ctx.hitl_measurement or {"status": "not_measured", "note": "attach a HITL measurement to promote proxy PPA claims"}, indent=2, default=_jd))
    if pdf_blob is not None:
        put("package", "validation/Nomo_Research_Brief.pdf", pdf_blob)
    else:
        put("package", "validation/Nomo_Research_Brief.pdf.unavailable.txt",
            "PDF not requested; select the pdf export format.\n")
    put("package", "software_sdk/SystemC_tb/README.md", _systemc_readme())
    put("package", "README.md", _readme(ctx, ev, written, caps, summary, skipped))
    # Keep the historical top-level report filename for callers that opened it
    # directly; the canonical copy is benchmarks/ and validation/.
    if pdf_blob is not None:
        put("package", "benchmarks/Nomo_Research_Brief.pdf", pdf_blob)
    put("package", "benchmarks/sensitivity_report.json", json.dumps(evidence, indent=2, default=_jd))
    put("package", "benchmarks/ppa_report.json", json.dumps(proxy_ppa(ctx.model, ev), indent=2))
    zf.close()
    return buf.getvalue(), {"root": root, "files": written, "skipped": skipped, "errors": errors,
                            "format": "zip", "structured_root": root, "evidence": evidence}


def _qgraph(ctx: RunContext, ev: Evaluation):
    from ..runtime.quantize import CompileOptions, compile_qgraph
    X, A = ctx.calib()
    report = ctx.ptq_report_for(ev)
    amax = (report or {}).get("edge_amax") if report else None
    return compile_qgraph(ctx.model, ctx.weights, ev.genome, X, A, CompileOptions(activation_amax=amax))


def _weights_npz(weights: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bytes:
    arrays: Dict[str, np.ndarray] = {}
    for name, (W, b) in weights.items():
        safe = "".join(c if c.isalnum() else "_" for c in name)
        arrays[f"{safe}_W"] = np.asarray(W)
        arrays[f"{safe}_b"] = np.asarray(b)
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


def _cmake_manifest() -> str:
    return """cmake_minimum_required(VERSION 3.16)
project(nomo_microkernel C)
set(CMAKE_C_STANDARD 11)
add_library(nomo_kernel STATIC src/kernel.c)
target_include_directories(nomo_kernel PUBLIC include)
target_compile_options(nomo_kernel PRIVATE -Wall -Wextra -Werror -pedantic)
"""


def _systemc_readme() -> str:
    return """# SystemC testbench boundary

The release package contains the integer C11 kernel and generated RTL boundary.
Integrate the target's clock/reset and bus adapter here for cycle-accurate
SystemC/Verilator/Gem5 co-simulation.  Nomo does not claim a cycle count until a
backend or hardware-in-the-loop measurement is attached to the package.

Expected telemetry fields are `cycle_count`, `cache_hit_ratio`, `bus_contention`,
`latency_ms`, and `energy_uj`.
"""


def build_archive(ctx: RunContext, ev: Evaluation, formats: List[str], archive: str = "zip") -> Tuple[bytes, Dict[str, Any]]:
    """Build a zip or gzip-compressed tar archive from the same structured tree."""
    data, manifest = build_bundle(ctx, ev, formats)
    kind = archive.lower().replace(".", "")
    if kind in ("zip", ""):
        return data, manifest
    if kind not in ("targz", "tar_gz", "tgz"):
        raise ValueError("archive must be 'zip' or 'tar.gz'")
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar, zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            item = tarfile.TarInfo(info.filename)
            item.size = info.file_size
            item.mtime = 0
            item.mode = 0o644
            tar.addfile(item, io.BytesIO(zf.read(info.filename)))
    manifest = dict(manifest, format="tar.gz")
    return out.getvalue(), manifest


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

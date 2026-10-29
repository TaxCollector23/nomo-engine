"""nomo CLI.

    nomo search  --model attitude_policy --hardware akd1500 [--acc-drop 4] [--energy J] [--latency S]
                 [--pop 64 --gens 60 --seed 0] [--out run.json]
    nomo compile --model attitude_policy --hardware loihi2 --genome run.json|recommended --out build/
                 [--leak-shift 0] [--test-vectors 64]
    nomo serve   [--host 0.0.0.0 --port 8765]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .hardware.profiles import get_profile
from .models.zoo import MODELS, attitude_calibration, synthetic_weights
from .search.evaluator import Budgets, NeurosymbolicEvaluator
from .search.genome import Coding, Domain, Genome, GuardGene, GuardImpl, LayerGene, repair
from .search.nsga2 import NSGA2Config, TriDomainNSGA2Optimizer


def genome_from_wire(w: dict) -> Genome:
    layers = tuple(LayerGene(Domain(d), wb, ab, Coding(c), T, bool(p)) for d, wb, ab, c, T, p in w["layers"])
    guards = tuple(GuardGene(s, GuardImpl(i)) for s, i in w["guards"])
    return Genome(layers, guards)


def cmd_search(a: argparse.Namespace) -> int:
    model, hw = MODELS[a.model](), get_profile(a.hardware)
    b = Budgets(e_max_j=a.energy or math.inf, l_max_s=a.latency or math.inf,
                acc_min=model.base_accuracy - a.acc_drop)
    ev = NeurosymbolicEvaluator(model, hw, b)

    def log(kind: str, data: dict) -> None:
        if kind == "gen.completed":
            print(f"gen {data['gen']:4d}  hv={data['hv']:.4f}  front={len(data['front']):3d}  "
                  f"unique={data['unique']:5d}  feasible={data['feasible_fraction']:.2f}", file=sys.stderr)

    res = TriDomainNSGA2Optimizer(model, hw, ev, NSGA2Config(a.pop, a.gens, seed=a.seed), telemetry=log).run()
    out = {"model": a.model, "hardware": a.hardware, "front": [e.to_wire() for e in res.front],
           "recommended": res.recommended.to_wire() if res.recommended else None, "hv": res.hv_history}
    Path(a.out).write_text(json.dumps(out, indent=1))
    r = res.recommended
    if r:
        print(f"recommended {r.key}\n  E={r.F[0]:.3e} J  L={r.F[1]:.3e} s  Acc={r.accuracy:.2f}%  cv={r.cv:.3f}")
    print(f"front: {len(res.front)} designs, {res.unique} unique evaluations, {res.wall_s:.1f}s -> {a.out}")
    return 0


def cmd_compile(a: argparse.Namespace) -> int:
    from .export.c11 import emit_c11, emit_test_harness
    from .export.nir_export import export_all
    from .runtime.quantize import CompileOptions, compile_qgraph, quantize_aux, quantize_input

    model, hw = MODELS[a.model](), get_profile(a.hardware)
    data = json.loads(Path(a.genome).read_text())
    wire = data["recommended"]["genome"] if "recommended" in data else data["genome"]
    g = repair(genome_from_wire(wire), model, hw)
    if a.model != "attitude_policy":
        print("compile: only dense-chain models with bundled weights/calibration are wired into the CLI", file=sys.stderr)
        return 2
    W = synthetic_weights(model)
    X, AUX = attitude_calibration(256 + a.test_vectors)
    qg = compile_qgraph(model, W, g, X[:256], AUX[:256], CompileOptions(leak_shift=a.leak_shift))
    out = Path(a.out)
    paths = export_all(qg, out, genome_wire=g.to_wire())
    em = emit_c11(qg)
    em.write(out)
    if a.test_vectors:
        Xi = np.stack([quantize_input(qg, x) for x in X[256:]])
        Ai = np.array([quantize_aux(x) for x in AUX[256:]])
        (out / "test_main.c").write_text(emit_test_harness(qg, Xi, Ai))
        cc = ["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-O2", "-o", str(out / "nomo_test"),
              str(out / "nomo_model.c"), str(out / "test_main.c")]
        if subprocess.run(cc).returncode == 0:
            r = subprocess.run([str(out / "nomo_test")], capture_output=True, text=True)
            print(r.stdout.strip())
    print(f"genome {g.key}")
    for k, p in paths.items():
        print(f"  {k:9s} {p}")
    print(f"  C11       {out / 'nomo_model.c'}  (RAM {em.manifest['static_ram_bytes']} B, ROM {em.manifest['const_rom_bytes']} B)")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("nomo.telemetry.server:app", host=a.host, port=a.port, log_level="info",
                access_log=False, proxy_headers=True, forwarded_allow_ips="*")   # Nomo logs access itself
    return 0


def cmd_pipeline(a: argparse.Namespace) -> int:
    """Run search + calibration + structured release export from nomo.yaml."""
    from .config import load_config, search_settings, section
    from .export.bundle import FORMATS, RunContext, build_archive
    from .modes import get_mode

    doc = load_config(a.config)
    model_id = str(doc.get("model", "attitude_policy"))
    hardware_id = str(doc.get("hardware", "akd1500"))
    if model_id in MODELS:
        model, weights, weights_source = MODELS[model_id](), synthetic_weights(MODELS[model_id]()), "synthetic (untrained demo weights)"
    else:
        from .ingest import ingest
        p = Path(model_id)
        if not p.exists():
            raise SystemExit(f"pipeline model '{model_id}' is not a built-in model or readable file")
        model, weights, rep = ingest(p.name, p.read_bytes(), float(doc.get("base_accuracy", 90.0)))
        weights_source = "uploaded file" if rep.weights_source == "file" else f"{rep.weights_source} weights from {p.name}"
    hw = get_profile(hardware_id)
    bdoc = section(doc, "budgets")
    budgets = Budgets(e_max_j=float(bdoc.get("energy_j", math.inf)),
                      l_max_s=float(bdoc.get("latency_s", math.inf)),
                      acc_min=float(bdoc.get("accuracy_min", model.base_accuracy - float(bdoc.get("accuracy_drop_max", 4.0)))),
                      period_max_s=float(bdoc.get("period_s", math.inf)),
                      min_plastic_params=int(bdoc.get("min_plastic_params", 0)))
    sdoc = search_settings(doc)
    ev = NeurosymbolicEvaluator(model, hw, budgets)
    opt = TriDomainNSGA2Optimizer(model, hw, ev, NSGA2Config(
        pop_size=int(sdoc.get("pop_size", 32)), generations=int(sdoc.get("generations", 8)),
        seed=int(sdoc.get("seed", 0)), asf_weights=tuple(sdoc.get("asf_weights", (1, 1, 1)))), telemetry=None)
    result = opt.run()
    if result.recommended is None:
        raise SystemExit("pipeline search produced no recommended design")
    calibration_data = None
    calibration_report = None
    if doc.get("calibration"):
        from .runtime.ptq import parse_calibration_bytes
        cp = Path(str(doc["calibration"]))
        cal = parse_calibration_bytes(cp.name, cp.read_bytes(), model.input_shape,
                                      max([model.constraints[s.constraint_id].n_aux for s in model.guard_sites] or [0]))
        calibration_data, calibration_report = (cal.inputs, cal.aux), cal.to_dict()
    settings = {"config_file": str(a.config), **doc, "mode": doc.get("mode", get_mode(doc.get("mode")).id if doc.get("mode") else None)}
    ctx = RunContext(model, weights, weights_source, hw, ev, settings,
                     ["pipeline executed by nomo.yaml"], opt.archive_front, opt.recommend,
                     calibration_data=calibration_data, calibration_report=calibration_report)
    formats = list(section(doc, "export").get("formats", FORMATS))
    archive = str(section(doc, "export").get("archive", "tar.gz"))
    out = Path(a.out)
    data, manifest = build_archive(ctx, result.recommended, formats, archive)
    out.write_bytes(data)
    print(f"recommended {result.recommended.key}")
    print(f"{archive} release package: {out} ({len(data)} bytes)")
    if manifest["skipped"]:
        print("skipped: " + "; ".join(f"{k}: {v}" for k, v in manifest["skipped"].items()), file=sys.stderr)
    return 0


def cmd_hitl(a: argparse.Namespace) -> int:
    from .hardware.hitl import HITLClient, HITLTarget
    measurement = HITLClient(HITLTarget(a.target, a.endpoint, a.token, a.timeout)).benchmark(
        a.artifact, metadata={"device": a.device}, repeats=a.repeats)
    print(json.dumps(measurement.to_dict(), indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nomo")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--model", default="attitude_policy", choices=sorted(MODELS))
    s.add_argument("--hardware", default="akd1500")
    s.add_argument("--acc-drop", type=float, default=4.0)
    s.add_argument("--energy", type=float)
    s.add_argument("--latency", type=float)
    s.add_argument("--pop", type=int, default=64)
    s.add_argument("--gens", type=int, default=60)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="nomo_run.json")
    s.set_defaults(fn=cmd_search)
    c = sub.add_parser("compile")
    c.add_argument("--model", default="attitude_policy", choices=sorted(MODELS))
    c.add_argument("--hardware", default="akd1500")
    c.add_argument("--genome", required=True, help="run JSON from `nomo search` or {\"genome\": wire}")
    c.add_argument("--out", default="build")
    c.add_argument("--leak-shift", type=int, default=0)
    c.add_argument("--test-vectors", type=int, default=64)
    c.set_defaults(fn=cmd_compile)
    v = sub.add_parser("serve")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    v.set_defaults(fn=cmd_serve)
    pz = sub.add_parser("pipeline", help="run a complete search and structured release from nomo.yaml")
    pz.add_argument("--config", required=True, help="declarative nomo.yaml")
    pz.add_argument("--out", default="nomo_enterprise_release.tar.gz")
    pz.set_defaults(fn=cmd_pipeline)
    h = sub.add_parser("hitl", help="send an exported artifact to a trusted benchmark agent")
    h.add_argument("benchmark", nargs="?", default="benchmark")
    h.add_argument("--endpoint", required=True, help="benchmark agent base URL")
    h.add_argument("--target", default="remote-edge")
    h.add_argument("--artifact", required=True)
    h.add_argument("--device", default="unknown")
    h.add_argument("--token")
    h.add_argument("--timeout", type=float, default=30.0)
    h.add_argument("--repeats", type=int, default=20)
    h.set_defaults(fn=cmd_hitl)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

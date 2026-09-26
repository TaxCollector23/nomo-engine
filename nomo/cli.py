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
    uvicorn.run("nomo.telemetry.server:app", host=a.host, port=a.port, log_level="info")
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
    v.add_argument("--port", type=int, default=8765)
    v.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

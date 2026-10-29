"""Where does spiking actually pay off? A win-region study over the real evaluator.

This study sweeps timestep counts and synaptic weight widths for the checked-in
model zoo on every silicon profile, evaluates all-continuous, all-spiking, and
hybrid genomes through the same NeurosymbolicEvaluator the search uses, and
reports the region where the spiking mapping wins on energy at (proxy)
iso-accuracy. It is evidence about the *cost model*, not about silicon: every
number is labelled with its provenance, and the profiles' energy coefficients
are modeled priors until a measured LUT (hardware/lut.py) replaces them.

Usage (from server/):  python3 studies/spiking_win_region.py [--out studies/spiking_win_region.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nomo.hardware.profiles import PROFILES  # noqa: E402
from nomo.models.zoo import MODELS  # noqa: E402
from nomo.search.evaluator import Budgets, NeurosymbolicEvaluator  # noqa: E402
from nomo.search.genome import Coding, Domain, Genome, LayerGene, default_gene  # noqa: E402


def uniform_genome(model_layers: int, hw, domain: Domain, w_bits: int, a_bits: int,
                   timesteps: int = 0, coding: Coding = Coding.NONE) -> Genome:
    gene = LayerGene(domain=domain, w_bits=w_bits, a_bits=a_bits,
                     coding=coding if domain == Domain.SNN else Coding.NONE,
                     timesteps=timesteps if domain == Domain.SNN else 0, plastic=False)
    return Genome(layers=tuple(gene for _ in range(model_layers)))


def hybrid_genome(model_layers: int, hw, w_bits: int, timesteps: int,
                  ann_prefix: int) -> Genome:
    """First `ann_prefix` layers continuous, the rest spiking (rate coded)."""
    ann = LayerGene(domain=Domain.ANN, w_bits=w_bits, a_bits=w_bits, coding=Coding.NONE, timesteps=0, plastic=False)
    snn = LayerGene(domain=Domain.SNN, w_bits=min(w_bits, max(hw.snn_w_bits)), a_bits=1,
                    coding=Coding.RATE, timesteps=timesteps, plastic=False)
    layers = tuple(ann if i < ann_prefix else snn for i in range(model_layers))
    return Genome(layers=layers)


def best_ann(model, hw) -> dict:
    """Cheapest-energy all-continuous genome across the profile's ANN bit widths."""
    best = None
    for wb in hw.ann_bits:
        ev = evaluator_for(model, hw).evaluate(uniform_genome(len(model.layers), hw, Domain.ANN, wb, wb))
        if best is None or ev.cost.energy_j < best["energy_j"]:
            best = {"energy_j": ev.cost.energy_j, "latency_s": ev.cost.latency_s, "accuracy": ev.accuracy,
                    "key": ev.key, "w_bits": wb}
    return best


def evaluator_for(model, hw) -> NeurosymbolicEvaluator:
    return NeurosymbolicEvaluator(model, hw, Budgets())


def study_profile(model_name: str, profile_id: str, acc_slack_points: float = 2.0,
                  latency_slack: float = 2.0) -> dict:
    model = MODELS[model_name]()
    hw = PROFILES[profile_id]
    ann = best_ann(model, hw)
    rows = []
    for T in hw.timesteps:
        for wb in hw.snn_w_bits:
            ev = evaluator_for(model, hw).evaluate(
                uniform_genome(len(model.layers), hw, Domain.SNN, wb, 1, timesteps=T, coding=Coding.RATE))
            rows.append({
                "genome": "all-spiking", "timesteps": int(T), "w_bits": int(wb),
                "energy_j": float(ev.cost.energy_j), "latency_s": float(ev.cost.latency_s),
                "accuracy_proxy": float(ev.accuracy),
                "accuracy_source": str(ev.accuracy_source), "cores": int(ev.cost.cores_used),
                "beats_ann_energy": bool(ev.cost.energy_j < ann["energy_j"]),
                "iso_accuracy": bool(ev.accuracy >= ann["accuracy"] - acc_slack_points),
                "within_latency_slack": bool(ev.cost.latency_s <= ann["latency_s"] * latency_slack),
            })
    # a hybrid sweep: first half continuous, rest spiking (crossing costs included)
    n = len(model.layers)
    for T in hw.timesteps:
        for prefix in (1, max(1, n // 2)):
            ev = evaluator_for(model, hw).evaluate(hybrid_genome(n, hw, 8, T, prefix))
            rows.append({
                "genome": f"hybrid-ann{prefix}", "timesteps": int(T), "w_bits": int(min(8, max(hw.snn_w_bits))),
                "energy_j": float(ev.cost.energy_j), "latency_s": float(ev.cost.latency_s),
                "accuracy_proxy": float(ev.accuracy),
                "accuracy_source": str(ev.accuracy_source), "cores": int(ev.cost.cores_used),
                "beats_ann_energy": bool(ev.cost.energy_j < ann["energy_j"]),
                "iso_accuracy": bool(ev.accuracy >= ann["accuracy"] - acc_slack_points),
                "within_latency_slack": bool(ev.cost.latency_s <= ann["latency_s"] * latency_slack),
            })
    wins = [r for r in rows if r["beats_ann_energy"] and r["iso_accuracy"] and r["within_latency_slack"]]

    # Where does the win region begin? Sweep the accuracy budget instead of hiding it:
    # for each allowed proxy-accuracy loss (points below the ANN reference), report whether
    # an energy+latency win exists and how large the energy saving gets.
    accuracy_slacks = [0.0, 1.0, 2.0, 5.0, 10.0, 14.0, 20.0]
    win_regions = {}
    for slack in accuracy_slacks:
        w = [r for r in rows if r["beats_ann_energy"] and r["within_latency_slack"]
             and r["accuracy_proxy"] >= ann["accuracy"] - slack]
        win_regions[str(slack)] = {
            "nonempty": bool(w),
            "best_energy_saving_pct": max((100.0 * (1.0 - r["energy_j"] / ann["energy_j"]) for r in w), default=None),
            "min_accuracy_loss_points": max((ann["accuracy"] - r["accuracy_proxy"] for r in w), default=None),
        }
    return {
        "model": model_name,
        "profile": profile_id,
        "profile_name": hw.name,
        "ann_reference": ann,
        "criteria": {"iso_accuracy_slack_points": acc_slack_points, "latency_slack": latency_slack,
                     "note": "accuracy values are the engine's proxy model, not measured task accuracy"},
        "rows": rows,
        "win_region_nonempty": bool(wins),
        "win_regions_by_accuracy_budget": win_regions,
        "best_spiking_energy_j": min((r["energy_j"] for r in rows), default=None),
        "winning_rows": wins,
        "provenance": "analytic cost model with modeled energy coefficients; not hardware validation data",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "spiking_win_region.json"))
    args = ap.parse_args()

    report = {}
    for profile_id in sorted(PROFILES):
        for model_name in sorted(MODELS):
            key = f"{model_name}@{profile_id}"
            report[key] = study_profile(model_name, profile_id)
            r = report[key]
            budgets = r["win_regions_by_accuracy_budget"]
            first_win = next((s for s, v in budgets.items() if v["nonempty"]), None)
            print(f"{key:28s} ann {r['ann_reference']['energy_j'] * 1e3:9.3f} mJ | "
                  f"best snn {r['best_spiking_energy_j'] * 1e3:9.3f} mJ | "
                  f"first win at accuracy budget: {first_win + ' points' if first_win is not None else 'none within 20'}")

    out = os.path.abspath(args.out)
    with open(out, "w") as f:
        json.dump({
            "schema": "nomo-spiking-win-region/1",
            "disclaimer": ("Analytic study over the nomo cost model. Energy coefficients are modeled "
                           "priors pending a measured calibration LUT; accuracy is the engine's proxy. "
                           "This is decision-support evidence, not silicon validation."),
            "studies": report,
        }, f, indent=1)
        f.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

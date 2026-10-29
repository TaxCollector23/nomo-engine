"""Regenerate scripts/golden.json from the Python reference (nomo_planner/domains).

The original generator was lost from the snapshot; this restores it. The
golden problems (domain + settings overrides + ASF weights) are the same seven
cases the browser engine has always been verified against. Everything else —
evaluations, Pareto fronts, recommendations, counterfactuals, calibration
points — is recomputed here from the Python packs so `npm run verify` remains
a true cross-language parity check.

Usage (from nomo-planner/):  python3 studies/export_golden.py --out ../scripts/golden.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nomo_planner.domains import build, from_python_problem  # noqa: E402
from nomo_planner.domains.calibration import calibration_points  # noqa: E402
from nomo_planner.domains.explain import counterfactuals, sensitivity  # noqa: E402
from nomo_planner.domains.packs import feasible  # noqa: E402
from nomo_planner.domains.search import Planner, enumerate_plans, recommend, space_size  # noqa: E402

# The seven verification cases (domain, settings overrides, ASF weights) —
# identical to the cases the original goldens were generated with.
CASES = [
    {"domain": "llm_training", "problem": {"model": "llama3-8b", "cluster": "h100_nvlink_ib"}, "weights": [3.0, 1.0]},
    {"domain": "llm_training", "problem": {"model": "llama2-70b", "cluster": "a100_nvlink_ib", "device_counts": [256, 512, 1024, 2048]}, "weights": [3.0, 1.0]},
    {"domain": "llm_training", "problem": {"model": "gpt3-175b", "cluster": "h100_nvlink_ib", "seq_len": 2048, "total_tokens": 300000000000.0}, "weights": [3.0, 1.0]},
    {"domain": "llm_inference", "problem": {"model": "llama3-8b", "cluster": "h100_nvlink_ib"}, "weights": [3.0, 1.0, 1.0]},
    {"domain": "llm_inference", "problem": {"model": "llama2-70b", "cluster": "l4_pcie", "max_ms_per_token": None}, "weights": [3.0, 1.0, 1.0]},
    {"domain": "arch_codesign", "problem": {"tokens_served": 1000000000000.0, "budget_usd": 5000000.0}, "weights": [3.0, 1.0]},
    {"domain": "arch_codesign", "problem": {"tokens_served": 0, "serve_cluster": "l4_pcie", "scaling_law": "hoffmann2022"}, "weights": [3.0, 1.0]},
    # AMD Instinct parity cases: the TS engine must match Python on the AMD fleet
    # exactly as it does on NVIDIA (same enumerators, same roofline/collective math).
    {"domain": "llm_training", "problem": {"model": "llama3-8b", "cluster": "mi300x_if_roce"}, "weights": [3.0, 1.0]},
    {"domain": "llm_training", "problem": {"model": "llama2-70b", "cluster": "mi355x_if_roce", "device_counts": [256, 512, 1024, 2048]}, "weights": [3.0, 1.0]},
    {"domain": "llm_inference", "problem": {"model": "llama3-8b", "cluster": "mi325x_if_roce"}, "weights": [3.0, 1.0, 1.0]},
]

EVAL_TARGET = 150  # per-case evaluation samples, as in the original goldens

# Scalar breakdown fields compared by verify-engine.mjs (nested objects such as
# memory_parts_gb are intentionally excluded; the checker closes scalars only).
BREAKDOWN_KEYS = {
    "llm_training": ["params", "dp", "micro_batches", "step_s", "tokens_per_s", "mfu", "energy_mwh", "memory_gb", "memory_cap_gb"],
    "llm_inference": ["tokens_per_s", "tokens_per_s_per_gpu", "memory_gb", "memory_cap_gb", "prefill_s", "decode_step_s", "kv_gb", "weights_gb", "tbt_static_ms", "ttft_static_ms"],
    "arch_codesign": ["params", "train_tokens", "train_cost_usd", "serve_cost_usd", "train_gpu_hours", "kv_heads", "serve_share"],
}


def plan_key(plan: dict):
    return tuple(sorted((k, str(v)) for k, v in plan.items()))


def export_case(case: dict) -> dict:
    domain = case["domain"]
    problem = from_python_problem(domain, case["problem"])
    pack = build(domain, problem)
    pl = Planner(pack)
    result = pl.run()  # no weights: mirrors verify-engine's pl.run()
    front = result["front"]
    # verify-engine.mjs calls pl.run() with no weights (ASF weight 1) for
    # "recommended", and re-recommends with the case weights for
    # "weighted_recommended" — mirror that split exactly.
    recommended = result["recommended"]
    weighted = None
    if front:
        weighted = recommend(front, case.get("weights"))
    keys = BREAKDOWN_KEYS[domain]

    def evaluated(plan):
        m = pack.evaluate(plan)
        return {
            "plan": {k: v for k, v in plan.items()},
            "objectives": dict(m["objectives"]),
            "constraints": dict(m["constraints"]),
            "breakdown": {k: m["breakdown"][k] for k in keys if k in m["breakdown"]},
        }

    evals = []
    seen = set()

    def add(plan):
        k = plan_key(plan)
        if plan is None or k in seen or len(evals) >= EVAL_TARGET:
            return
        seen.add(k)
        evals.append(evaluated(plan))

    for e in front:
        add(e["plan"])
    add(recommended["plan"] if recommended else None)
    add(weighted["plan"] if weighted else None)
    stride = max(1, math.floor(space_size(pack) / max(1, EVAL_TARGET - len(evals))))
    for i, plan in enumerate(enumerate_plans(pack)):
        if len(evals) >= EVAL_TARGET:
            break
        if i % stride == 0:
            add(plan)

    cfs = counterfactuals(pl, recommended["plan"]) if recommended else []
    return {
        "domain": domain,
        "problem": case["problem"],
        "space_size": space_size(pack),
        "evals": evals,
        "front": [e["plan"] for e in front],
        "recommended": recommended["plan"] if recommended else None,
        "weighted_recommended": weighted["plan"] if weighted else None,
        "counterfactuals": [
            {"variable": cf["variable"], "value": cf["value"], "deltas_pct": cf["deltas_pct"], "feasible": cf["feasible"], "broken": cf["broken"]}
            for cf in cfs
        ],
        "sensitivity": sensitivity(pl, recommended["plan"]) if recommended else {},
        "weights": case.get("weights"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "golden.json"))
    args = ap.parse_args()

    golden = {"cases": [export_case(c) for c in CASES]}
    cal = []
    for pt in calibration_points():
        cal.append({
            "plan": pt["plan"], "measured_step_s": pt["measured"], "predicted_step_s": pt["predicted"],
            "predicted_uncalibrated": pt["uncalibrated"], "source": pt["source"],
        })
    golden["calibration"] = cal

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(golden, f, indent=1)
        f.write("\n")
    n_fronts = sum(len(c["front"]) for c in golden["cases"])
    print(f"wrote {out}: {len(golden['cases'])} cases, {n_fronts} front plans, {len(cal)} calibration points")


if __name__ == "__main__":
    main()

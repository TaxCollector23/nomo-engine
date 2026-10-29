"""Explanations computed by re-evaluating plans.

Restored Python reference for src/planner/explain.ts (header: "Port of
nomo_planner/explain.py"). Counterfactual deltas are genuine re-evaluations
through the same pack model — never cached deltas.
"""
from __future__ import annotations

from typing import List


def _pct(n: float, o: float) -> float:
    return (100 * (n - o)) / abs(o) if o else 0


def counterfactuals(pl, plan: dict) -> List[dict]:
    base = pl.evaluate(plan)
    objs = pl.pack.objectives()
    out: List[dict] = []
    for v in pl.pack.variables():
        for c in v["choices"]:
            if c == plan[v["name"]]:
                continue
            alt = pl.evaluate({**plan, v["name"]: c})
            out.append({
                "variable": v["name"],
                "value": c,
                "deltas_pct": {o["name"]: _pct(alt["metrics"]["objectives"][o["name"]], base["metrics"]["objectives"][o["name"]]) for o in objs},
                "feasible": all(x <= 0 for x in alt["metrics"]["constraints"].values()) if _violation(alt["metrics"]) <= 0 else False,
                "broken": sorted(k for k, x in alt["metrics"]["constraints"].items() if x > 0),
            })
    return out


def _violation(metrics: dict) -> float:
    return sum(max(0.0, v) for v in metrics["constraints"].values())


def sensitivity(pl, plan: dict) -> dict:
    res = {v["name"]: 0 for v in pl.pack.variables()}
    for cf in counterfactuals(pl, plan):
        if cf["feasible"]:
            res[cf["variable"]] = max(res[cf["variable"]], max(abs(d) for d in cf["deltas_pct"].values()))
    return dict(sorted(res.items(), key=lambda kv: -kv[1]))

"""Exhaustive search, Pareto front, ASF recommendation.

Restored Python reference for src/planner/search.ts (header: "Port of
nomo_planner/search.py"). Enumeration order, non-domination, the Wierzbicki
achievement-scalarizing recommendation, and the front tie-breaking are all
replicated exactly so goldens generated here verify the browser engine.
"""
from __future__ import annotations

import itertools
import json
import math
from functools import cmp_to_key
from typing import Dict, List, Optional

from .packs import feasible, violation


def objective_vector(pack, m: dict) -> List[float]:
    return [(-m["objectives"][o["name"]] if o.get("maximize") else m["objectives"][o["name"]]) for o in pack.objectives()]


def enumerate_plans(pack):
    vars_ = pack.variables()
    if any(len(v["choices"]) == 0 for v in vars_):
        return
    for combo in itertools.product(*[v["choices"] for v in vars_]):
        yield {v["name"]: combo[i] for i, v in enumerate(vars_)}


def space_size(pack) -> int:
    n = 1
    for v in pack.variables():
        n *= len(v["choices"])
    return n


def nondominated(F: List[List[float]]) -> List[bool]:
    n = len(F)
    keep = [True] * n
    for i in range(n):
        fi = F[i]
        for j in range(n):
            if j == i:
                continue
            fj = F[j]
            le = True
            lt = False
            for k in range(len(fi)):
                if fj[k] > fi[k]:
                    le = False
                    break
                if fj[k] < fi[k]:
                    lt = True
            if le and lt:
                keep[i] = False
                break
    return keep


def _lex_cmp(a: List[float], b: List[float]) -> int:
    for k in range(len(a)):
        if a[k] != b[k]:
            return -1 if a[k] < b[k] else 1
    return 0


def to_precision15(x: float) -> str:
    """Deterministic 15-significant-digit key (mirrors JS toPrecision(15) semantics)."""
    if x == 0:
        return "0.00000000000000"
    return format(x, ".15g")


def recommend(front: List[dict], weights: Optional[List[float]] = None) -> Optional[dict]:
    """front: list of evaluated entries with .F vectors (dicts with key 'F')."""
    if not front:
        return None
    G = [[(math.log10(max(f, 1e-300)) if f > 0 else f) for f in e["F"]] for e in front]
    m = len(G[0])
    lo = [min(g[k] for g in G) for k in range(m)]
    hi = [max(g[k] for g in G) for k in range(m)]
    w = weights if weights is not None else [1.0] * m
    best_i = 0
    best_score = math.inf
    for i, g in enumerate(G):
        N = [((g[k] - lo[k]) / ((hi[k] - lo[k]) if hi[k] > lo[k] else 1)) * w[k] for k in range(m)]
        score = max(N) + 1e-3 * sum(N)
        if score < best_score:
            best_score = score
            best_i = i
    return front[best_i]


class Planner:
    def __init__(self, pack):
        self.pack = pack
        self.cache: Dict[str, dict] = {}

    @staticmethod
    def key(plan: dict) -> str:
        return json.dumps([[k, plan[k]] for k in sorted(plan)], separators=(",", ":"))

    def evaluate(self, plan: dict) -> dict:
        k = self.key(plan)
        e = self.cache.get(k)
        if e is None:
            m = self.pack.evaluate(plan)
            e = {"plan": dict(plan), "metrics": {k2: m[k2] for k2 in ("objectives", "constraints", "notes")},
                 "F": objective_vector(self.pack, m)}
            self.cache[k] = e
        return e

    def run(self, weights: Optional[List[float]] = None) -> dict:
        for plan in enumerate_plans(self.pack):
            self.evaluate(plan)
        return self.result(weights)

    def result(self, weights: Optional[List[float]] = None) -> dict:
        all_vals = list(self.cache.values())
        feas = [e for e in all_vals if feasible(e["metrics"])]
        front: List[dict] = []
        if feas:
            keep = nondominated([e["F"] for e in feas])
            nd = [e for e, k in zip(feas, keep) if k]

            def slack(e):
                return max(list(e["metrics"]["constraints"].values()) + [-1.0])

            best: Dict[str, dict] = {}
            for e in nd:
                k = ",".join(to_precision15(f) for f in e["F"])
                cur = best.get(k)
                if cur is None or slack(e) < slack(cur):
                    best[k] = e
            front = sorted(best.values(), key=cmp_to_key(lambda a, b: _lex_cmp(a["F"], b["F"])))
        closest = [] if feas else sorted(all_vals, key=lambda e: violation(e["metrics"]))[:5]
        return {
            "front": front, "closest": closest, "all": all_vals, "evaluated": len(all_vals),
            "exhaustive": True, "recommended": recommend(front, weights),
        }

"""NeurosymbolicEvaluator: genome -> (objectives, constraint violation, diagnostics).

Objectives (all minimised, SPEC §1.3):
    f1 = E(x)            joules per inference (dynamic + static + crossing + routing)
    f2 = L(x)            seconds per inference, end to end
    f3 = 100 - Acc(x)    accuracy loss in percentage points

Constraints g_j(x) <= 0, normalised so that 1.0 means "100 % over budget":
    g_E = E/E_max - 1           g_L = L/L_max - 1          g_A = (A_min - Acc)/10
    g_P = P/P_max - 1           g_M = max_u mem_u/cap_u - 1 g_C = cores/core_budget - 1
    g_Pl = (Pl_min - Pl)/Pl_min (on-device learning capacity, plastic parameters)
    CV(x) = sum_j max(0, g_j)

Accuracy is multi-fidelity: a calibrated proxy (cheap, every candidate) and an
optional oracle (true evaluation: convert + fine-tune + test, expensive) invoked
on a budgeted subset via `promote`. Oracle results replace the proxy in the cache
and feed a per-run linear recalibration of the proxy (a, b in Acc = a + b*proxy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..hardware.cost_model import CostReport, HardwareCostModel
from ..hardware.profiles import SiliconProfile
from ..ir import ModelGraph
from .genome import Coding, Domain, Genome


@dataclass(frozen=True)
class Budgets:
    e_max_j: float = math.inf
    l_max_s: float = math.inf
    acc_min: float = 0.0
    period_max_s: float = math.inf
    min_plastic_params: int = 0


@dataclass
class Evaluation:
    genome: Genome
    key: str
    F: np.ndarray                      # [E_j, L_s, 100 - Acc]
    cv: float
    g: Dict[str, float]
    accuracy: float
    accuracy_source: str               # proxy | oracle
    accuracy_terms: Dict[str, float]
    cost: CostReport
    metrics: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowd: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.cv == 0.0

    def to_wire(self) -> dict:
        return {"key": self.key, "f": [float(self.F[0]), float(self.F[1]), float(self.accuracy)],
                "cv": float(self.cv), "feasible": self.feasible, "rank": int(self.rank),
                "acc_src": self.accuracy_source, "genome": self.genome.to_wire(),
                "crossings": len(self.cost.crossings), "cores": self.cost.cores_used}


# ---------------------------------------------------------------------------
# Accuracy proxy
# ---------------------------------------------------------------------------

def psi(b: int, b_min: int, b_ref: int = 8) -> float:
    """Normalised quantisation-noise factor: 1 at b_min, 0 at >= b_ref (noise power ~ 4^-b)."""
    if b >= b_ref:
        return 0.0
    return (4.0 ** (b_ref - b) - 1.0) / (4.0 ** (b_ref - b_min) - 1.0)


class ProxyAccuracyModel:
    """Acc(x) = A0 - D - rho D^2,   D = sum_l delta_l(g_l) + sum_c delta_c   (SPEC §5)."""

    CROSS = {("ANN", "SNN"): 0.03, ("SNN", "ANN"): 0.05, ("SNN", "SYM"): 0.05,
             ("SYM", "SNN"): 0.03, ("ANN", "SYM"): 0.005, ("SYM", "ANN"): 0.005}
    alpha = 1.0                         # rate-coding error exponent, delta ~ (T_ref / T)^alpha
    beta_ttfs = 0.5                     # TTFS timing-resolution exponent

    def __init__(self, model: ModelGraph) -> None:
        self.model = model
        self.a, self.b = 0.0, 1.0      # recalibration Acc_true ~ a + b * Acc_proxy

    def terms(self, g: Genome) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, (gene, spec) in enumerate(zip(g.layers, self.model.layers)):
            s = spec.sensitivity
            if gene.domain == Domain.ANN:
                d = s.q_w * psi(gene.w_bits, s.b_min) + s.q_a * psi(gene.a_bits, s.b_min)
            elif gene.domain == Domain.SNN:
                d = s.q_w * psi(gene.w_bits, s.b_min)
                if gene.coding == Coding.TTFS:      # spike-time quantisation error, weaker T dependence
                    d += s.c_ttfs * (s.t_ref / gene.timesteps) ** self.beta_ttfs
                else:
                    d += s.c_rate * (s.t_ref / gene.timesteps) ** self.alpha
            else:
                d = s.sym
            out[spec.name] = d
        for c in g.crossings(self.model):
            name = ("ANN", "SNN", "SYM")
            base = self.CROSS.get((name[c.src], name[c.dst]), 0.0)
            if c.src == Domain.SNN and c.timesteps:
                base *= 8.0 / c.timesteps               # count quantisation shrinks with T
            out[f"x{c.edge}:{name[c.src]}>{name[c.dst]}"] = base
        return out

    def raw(self, g: Genome) -> tuple[float, Dict[str, float]]:
        t = self.terms(g)
        D = sum(t.values())
        return self.model.base_accuracy - D - self.model.interaction * D * D, t

    def __call__(self, g: Genome) -> tuple[float, Dict[str, float]]:
        acc, t = self.raw(g)
        return float(min(100.0, max(0.0, self.a + self.b * acc))), t      # accuracy is a percentage

    def recalibrate(self, raw: Sequence[float], true: Sequence[float]) -> None:
        """Least-squares fit of true = a + b * raw_proxy. With < 3 distinct points only the
        offset is fitted (b = 1), which is the minimum-variance choice for tiny samples."""
        x, y = np.asarray(raw, float), np.asarray(true, float)
        if len(x) == 0:
            return
        if len(x) < 3 or np.ptp(x) < 1e-6:
            self.a, self.b = float(np.mean(y - x)), 1.0
            return
        A = np.stack([np.ones_like(x), x], 1)
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        self.a, self.b = float(a), float(b)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

AccuracyOracle = Callable[[ModelGraph, Genome], float]


class NeurosymbolicEvaluator:
    def __init__(self, model: ModelGraph, hw: SiliconProfile, budgets: Budgets,
                 cost_model: Optional[HardwareCostModel] = None,
                 proxy: Optional[ProxyAccuracyModel] = None,
                 oracle: Optional[AccuracyOracle] = None) -> None:
        self.model, self.hw, self.budgets = model, hw, budgets
        self.cost_model = cost_model or HardwareCostModel(hw, model)
        self.proxy = proxy or ProxyAccuracyModel(model)
        self.oracle = oracle
        self.cache: Dict[str, Evaluation] = {}
        self.oracle_results: Dict[str, float] = {}
        self._proxy_at_oracle: Dict[str, float] = {}
        self.n_calls = 0

    # -------------------------------------------------------------- core
    def evaluate(self, g: Genome) -> Evaluation:
        self.n_calls += 1
        hit = self.cache.get(g.key)
        if hit is not None:
            return hit
        cost = self.cost_model.evaluate(g)
        acc_proxy, terms = self.proxy(g)
        if g.key in self.oracle_results:
            acc, src = self.oracle_results[g.key], "oracle"
        else:
            acc, src = acc_proxy, "proxy"
        ev = self._assemble(g, cost, acc, src, terms)
        self.cache[g.key] = ev
        return ev

    def _assemble(self, g: Genome, cost: CostReport, acc: float, src: str, terms: Dict[str, float]) -> Evaluation:
        b, hw = self.budgets, self.hw
        plastic = sum(self.model.layers[i].params for i, l in enumerate(g.layers) if l.plastic)
        mem_ratio = max((cost.unit_memory[u] / cost.unit_capacity[u]) for u in cost.unit_memory
                        if cost.unit_capacity.get(u)) if cost.unit_memory else 0.0
        gv = {
            "energy": cost.energy_j / b.e_max_j - 1.0 if math.isfinite(b.e_max_j) else -1.0,
            "latency": cost.latency_s / b.l_max_s - 1.0 if math.isfinite(b.l_max_s) else -1.0,
            "accuracy": (b.acc_min - acc) / 10.0,
            "period": cost.frame_period_s / b.period_max_s - 1.0 if math.isfinite(b.period_max_s) else -1.0,
            "memory": mem_ratio - 1.0,
            "cores": (cost.cores_used / hw.n_cores - 1.0) if not hw.snn_dense else -1.0,
            "plasticity": ((b.min_plastic_params - plastic) / b.min_plastic_params) if b.min_plastic_params else -1.0,
        }
        cv = float(sum(max(0.0, v) for v in gv.values()))
        F = np.array([cost.energy_j, cost.latency_s, 100.0 - acc], dtype=np.float64)
        metrics = {
            "plastic_params": float(plastic),
            "crossings": float(len(cost.crossings)),
            "crossing_energy_share": cost.crossing_energy_j / max(cost.energy_j, 1e-30),
            "measured_energy_fraction": cost.measured_energy_fraction,
            "frame_period_s": cost.frame_period_s,
            "memory_bytes": cost.memory_bytes,
        }
        return Evaluation(g, g.key, F, cv, gv, acc, src, terms, cost, metrics)

    def rebudget(self, budgets: Budgets) -> None:
        """Re-score every cached candidate against new budgets without re-running cost models."""
        self.budgets = budgets
        for k, ev in list(self.cache.items()):
            self.cache[k] = self._assemble(ev.genome, ev.cost, ev.accuracy, ev.accuracy_source, ev.accuracy_terms)

    # -------------------------------------------------------------- multi-fidelity
    def promote(self, candidates: Sequence[Evaluation], k: int) -> List[Evaluation]:
        """Run the oracle on up to k not-yet-measured candidates (feasible first, then lowest CV)."""
        if self.oracle is None or k <= 0:
            return []
        todo = [e for e in sorted(candidates, key=lambda e: (e.cv, e.rank)) if e.key not in self.oracle_results][:k]
        out = []
        for ev in todo:
            true = float(self.oracle(self.model, ev.genome))
            self.oracle_results[ev.key] = true
            self._proxy_at_oracle[ev.key] = self.proxy.raw(ev.genome)[0]
            new = self._assemble(ev.genome, ev.cost, true, "oracle", ev.accuracy_terms)
            self.cache[ev.key] = new
            out.append(new)
        keys = list(self.oracle_results)
        self.proxy.recalibrate([self._proxy_at_oracle[k_] for k_ in keys], [self.oracle_results[k_] for k_ in keys])
        # refresh proxy-scored cache entries under the recalibrated proxy
        for key, ev in list(self.cache.items()):
            if ev.accuracy_source == "proxy":
                acc, terms = self.proxy(ev.genome)
                self.cache[key] = self._assemble(ev.genome, ev.cost, acc, "proxy", terms)
        return out

"""TriDomainNSGA2Optimizer (SPEC §1-2).

Generational loop, t = 0..G-1:
    Q_t   = vary(P_t)                        (tournament on (rank, crowding), adaptive operators)
    R_t   = P_t U Q_t                        (deduplicated by genome key)
    F_1.. = constrained non-dominated sort of R_t
    P_t+1 = fill fronts in order, truncate the last by crowding distance in normalised space
Telemetry is emitted through a plain callback so the engine has no I/O dependencies.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from ..hardware.profiles import SiliconProfile
from ..ir import ModelGraph
from .evaluator import Evaluation, NeurosymbolicEvaluator
from .genome import Coding, Domain, Genome, GuardGene, LayerGene, default_gene, repair, uniform_genome
from .operators import CROSSOVERS, MUTATIONS, AdaptivePursuit
from .pareto import asf_select, crowding_distance, fast_non_dominated_sort, hv3d, nondominated_mask, normalise

Telemetry = Callable[[str, dict], None]


@dataclass
class NSGA2Config:
    pop_size: int = 64
    generations: int = 60
    p_crossover: float = 0.9
    seed: int = 0
    max_retries: int = 8                 # attempts to produce an unseen genome per offspring
    hv_window: int = 12                  # stop if HV improves < hv_tol over this many generations
    hv_tol: float = 1e-4
    oracle_per_generation: int = 0       # true accuracy evaluations per generation (multi-fidelity)
    log_axes: tuple = (0, 1)
    p_mutation: float = 1.0              # probability of mutating a crossover child (clones always mutate)
    archive_capacity: int = 0            # 0 = unbounded; else keep the most spread-out front members
    asf_weights: Optional[tuple] = None  # recommendation weights (energy, latency, accuracy); None = equal


@dataclass
class SearchResult:
    front: List[Evaluation]              # feasible non-dominated set over the whole archive
    recommended: Optional[Evaluation]
    population: List[Evaluation]
    hv_history: List[float]
    generations_run: int
    evaluations: int
    unique: int
    wall_s: float
    operator_probs: Dict[str, float] = field(default_factory=dict)


class TriDomainNSGA2Optimizer:
    def __init__(self, model: ModelGraph, hw: SiliconProfile, evaluator: NeurosymbolicEvaluator,
                 config: Optional[NSGA2Config] = None, telemetry: Optional[Telemetry] = None) -> None:
        self.model, self.hw, self.ev = model, hw, evaluator
        self.cfg = config or NSGA2Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.emit: Telemetry = telemetry or (lambda kind, data: None)
        self.x_ops = AdaptivePursuit(list(CROSSOVERS))
        self.m_ops = AdaptivePursuit(list(MUTATIONS))
        self.hv_history: List[float] = []
        self._box = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ seeding
    def seed_population(self) -> List[Genome]:
        m, hw, n = self.model, self.hw, self.model.n
        seeds: List[Genome] = [uniform_genome(m, hw, Domain.ANN)]
        for T in sorted(set(hw.timesteps))[1::2]:
            seeds.append(uniform_genome(m, hw, Domain.SNN, T=T))
        seeds.append(uniform_genome(m, hw, Domain.SNN, coding=Coding.TTFS))
        guards = tuple(GuardGene(s.after_layer) for s in m.guard_sites)
        # prefix splits: continuous perception front-end, spiking back-end
        for k in range(1, n):
            L = tuple(default_gene(Domain.ANN if i < k else Domain.SNN, hw) for i in range(n))
            seeds.append(repair(Genome(L, guards), m, hw))
        # symbolic substitution wherever available, on top of the all-ANN and a mid split
        if any(l.symbolic_substitute for l in m.layers):
            for base in (seeds[0], seeds[min(len(seeds) - 1, 2 + n // 2)]):
                L = tuple(default_gene(Domain.SYM, hw) if spec.symbolic_substitute else g
                          for g, spec in zip(base.layers, m.layers))
                seeds.append(repair(Genome(L, guards), m, hw))
        uniq: Dict[str, Genome] = {}
        for s in seeds:
            uniq.setdefault(s.key, s)
        attempts = 0
        while len(uniq) < self.cfg.pop_size and attempts < 50 * self.cfg.pop_size:
            g = self.random_genome()
            uniq.setdefault(g.key, g)
            attempts += 1
        pop = list(uniq.values())[: self.cfg.pop_size]
        # user locks/toggles can leave fewer distinct designs than the population size:
        # pad with repeats (evaluations are cached, so repeats cost nothing) instead of looping forever
        k = 0
        while len(pop) < self.cfg.pop_size:
            pop.append(pop[k % len(uniq)])
            k += 1
        return pop

    def random_genome(self) -> Genome:
        m, hw = self.model, self.hw
        L = []
        from .policy import admissible_domains
        for i, spec in enumerate(m.layers):
            doms = admissible_domains(i, m)
            d = doms[int(self.rng.integers(len(doms)))]
            base = default_gene(d, hw, T=int(self.rng.choice(hw.timesteps)),
                                coding=Coding.TTFS if self.rng.random() < 0.25 else Coding.RATE)
            if d == Domain.ANN:
                b = int(self.rng.choice(hw.ann_bits))
                base = LayerGene(d, b, int(self.rng.choice(hw.ann_bits)))
            elif d == Domain.SNN:
                base = LayerGene(d, int(self.rng.choice(hw.snn_w_bits)), base.a_bits, base.coding, base.timesteps,
                                 bool(self.rng.random() < 0.15))
            L.append(base)
        guards = tuple(GuardGene(s.after_layer) for s in m.guard_sites)
        return repair(Genome(tuple(L), guards), m, hw)

    # ------------------------------------------------------------------ helpers
    def _rank(self, pop: List[Evaluation]) -> List[np.ndarray]:
        F = np.stack([e.F for e in pop])
        cv = np.array([e.cv for e in pop])
        fronts = fast_non_dominated_sort(F, cv)
        lo, hi = F.min(axis=0), F.max(axis=0)
        Fn = normalise(F, lo, hi, self.cfg.log_axes)
        for r, idx in enumerate(fronts):
            cd = crowding_distance(Fn[idx])
            for j, i in enumerate(idx):
                pop[i].rank = r
                pop[i].crowd = float(cd[j])
        return fronts

    def _tournament(self, pop: List[Evaluation]) -> Evaluation:
        a, b = pop[int(self.rng.integers(len(pop)))], pop[int(self.rng.integers(len(pop)))]
        if a.rank != b.rank:
            return a if a.rank < b.rank else b
        return a if a.crowd >= b.crowd else b

    def _select(self, pool: List[Evaluation]) -> List[Evaluation]:
        uniq: Dict[str, Evaluation] = {}
        for e in pool:
            uniq.setdefault(e.key, e)
        pool = list(uniq.values())
        fronts = self._rank(pool)
        nxt: List[Evaluation] = []
        for idx in fronts:
            members = [pool[i] for i in idx]
            if len(nxt) + len(members) <= self.cfg.pop_size:
                nxt.extend(members)
            else:
                members.sort(key=lambda e: -e.crowd)
                nxt.extend(members[: self.cfg.pop_size - len(nxt)])
                break
        return nxt

    def archive_front(self) -> List[Evaluation]:
        feas = [e for e in self.ev.cache.values() if e.feasible]
        if not feas:
            return []
        F = np.stack([e.F for e in feas])
        mask = nondominated_mask(F)
        front = [e for e, k in zip(feas, mask) if k]
        cap = self.cfg.archive_capacity
        if cap and len(front) > cap:                    # keep the most spread-out members
            Ff = np.stack([e.F for e in front])
            cd = crowding_distance(normalise(Ff, Ff.min(0), Ff.max(0), self.cfg.log_axes))
            front = [front[i] for i in np.argsort(-cd, kind="stable")[:cap]]
        return sorted(front, key=lambda e: e.F[1])

    def _hv(self, front: List[Evaluation]) -> float:
        if not front or self._box is None:
            return 0.0
        lo, hi = self._box
        Fn = np.clip(normalise(np.stack([e.F for e in front]), lo, hi, self.cfg.log_axes), 0.0, None)
        return hv3d(Fn, np.array([1.1, 1.1, 1.1]))

    def recommend(self, weights: Optional[np.ndarray] = None) -> Optional[Evaluation]:
        if weights is None and self.cfg.asf_weights is not None:
            weights = np.asarray(self.cfg.asf_weights, float)
        front = self.archive_front()
        if front:
            F = np.stack([e.F for e in front])
            return front[asf_select(normalise(F, F.min(0), F.max(0), self.cfg.log_axes), weights)]
        if not self.ev.cache:
            return None
        return min(self.ev.cache.values(), key=lambda e: e.cv)

    # ------------------------------------------------------------------ variation
    def _offspring(self, pop: List[Evaluation], seen: set) -> tuple:
        cfg = self.cfg
        p1 = self._tournament(pop)
        xo = None
        if self.rng.random() < cfg.p_crossover:
            xo = self.x_ops.sample(self.rng)
            child = CROSSOVERS[xo](p1.genome, self._tournament(pop).genome, self.model, self.hw, self.rng)
        else:
            child = p1.genome
        mo = self.m_ops.sample(self.rng)
        if xo is None or self.rng.random() < cfg.p_mutation:
            child = MUTATIONS[mo](child, self.model, self.hw, self.rng)
        else:
            mo = "none"
        tries = 0
        while (child.key in seen or child.key in self.ev.cache) and tries < cfg.max_retries:
            mo = self.m_ops.sample(self.rng)
            child = MUTATIONS[mo](child, self.model, self.hw, self.rng)
            tries += 1
        changed = [i for i, (a, b) in enumerate(zip(p1.genome.layers, child.layers)) if a != b]
        return child, xo, mo, p1, changed

    # ------------------------------------------------------------------ main loop
    def run(self) -> SearchResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        pop = [self.ev.evaluate(g) for g in self.seed_population()]
        F0 = np.stack([e.F for e in pop])
        lo, hi = F0.min(axis=0), F0.max(axis=0)
        lo = lo * np.array([0.5, 0.5, 1.0]) - np.array([0.0, 0.0, 0.5])
        hi = hi * np.array([1.5, 1.5, 1.0]) + np.array([0.0, 0.0, 0.5])
        self._box = (lo, hi)
        pop = self._select(pop)
        self.emit("run.started", {"model": self.model.name, "hardware": self.hw.name,
                                  "layers": [l.name for l in self.model.layers],
                                  "guard_sites": [s.after_layer for s in self.model.guard_sites],
                                  "pop_size": cfg.pop_size, "generations": cfg.generations,
                                  "objectives": ["energy_j", "latency_s", "accuracy_pct"]})
        self.emit("eval.batch", {"gen": 0, "items": [e.to_wire() for e in pop]})
        gen = 0
        for gen in range(1, cfg.generations + 1):
            if self._stop:
                break
            seen: set = set()
            offspring: List[Evaluation] = []
            origin: Dict[str, tuple] = {}
            new_items = []
            for _ in range(cfg.pop_size):
                child, xo, mo, parent, changed = self._offspring(pop, seen)
                seen.add(child.key)
                fresh = child.key not in self.ev.cache
                ev = self.ev.evaluate(child)
                offspring.append(ev)
                origin[ev.key] = (xo, mo)
                if fresh:
                    item = ev.to_wire()
                    item["parent"] = parent.key
                    item["changed"] = changed
                    item["ops"] = [xo, mo]
                    new_items.append(item)
            pop = self._select(pop + offspring)

            survivors = {e.key for e in pop}
            xo_off, xo_sur, mo_off, mo_sur = {}, {}, {}, {}
            for key, (xo, mo) in origin.items():
                if xo:
                    xo_off[xo] = xo_off.get(xo, 0) + 1
                    xo_sur[xo] = xo_sur.get(xo, 0) + (key in survivors)
                if mo in MUTATIONS:
                    mo_off[mo] = mo_off.get(mo, 0) + 1
                    mo_sur[mo] = mo_sur.get(mo, 0) + (key in survivors)
            self.x_ops.update(xo_off, xo_sur)
            self.m_ops.update(mo_off, mo_sur)

            front = self.archive_front()
            if cfg.oracle_per_generation:
                self.ev.promote(front or pop, cfg.oracle_per_generation)
                pop = self._select([self.ev.cache[e.key] for e in pop])
                front = self.archive_front()
            hv = self._hv(front)
            self.hv_history.append(hv)
            rec = self.recommend()
            self.emit("eval.batch", {"gen": gen, "items": new_items})
            self.emit("gen.completed", {
                "gen": gen, "hv": hv, "evaluations": self.ev.n_calls, "unique": len(self.ev.cache),
                "feasible_fraction": float(np.mean([e.feasible for e in self.ev.cache.values()])),
                "front": [e.key for e in front], "population": [e.key for e in pop],
                "recommended": rec.to_wire() if rec else None,
                "operators": {**self.x_ops.snapshot(), **self.m_ops.snapshot()},
            })
            w = cfg.hv_window
            if len(self.hv_history) > w and self.hv_history[-1] - self.hv_history[-1 - w] < cfg.hv_tol and hv > 0:
                break

        front = self.archive_front()
        rec = self.recommend()
        result = SearchResult(front, rec, pop, self.hv_history, gen, self.ev.n_calls, len(self.ev.cache),
                              time.perf_counter() - t0, {**self.x_ops.snapshot(), **self.m_ops.snapshot()})
        self.emit("run.completed", {"recommended": rec.to_wire() if rec else None,
                                    "front": [e.to_wire() for e in front], "generations": gen,
                                    "evaluations": result.evaluations, "unique": result.unique,
                                    "hv": self.hv_history[-1] if self.hv_history else 0.0,
                                    "wall_s": result.wall_s})
        return result

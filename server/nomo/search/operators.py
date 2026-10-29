"""Variation operators over the tri-domain genome (SPEC §2.3).

Every operator returns repair(child), so the offspring is always in V(model, hw).
Operator choice is adaptive (adaptive pursuit, Thierens 2005): each operator keeps a
quality estimate q_i updated with its offspring's survival into the next population,
and its selection probability chases p_max for the current best operator.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..hardware.profiles import SiliconProfile
from ..ir import ModelGraph
from .genome import (Coding, Domain, Genome, GuardGene, GuardImpl, LayerGene,
                     default_gene, repair)

Rng = np.random.Generator


def boundaries(g: Genome) -> List[int]:
    """Indices i where layer i starts a new segment (i > 0)."""
    return [s.start for s in g.segments()[1:]]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def segment_aligned_crossover(p1: Genome, p2: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    """Two-point crossover whose cut points are drawn from the union of both parents'
    domain boundaries (plus one uniform point for exploration). Cutting at boundaries
    transplants whole segments, so the child inherits intact domain partitions and the
    precision/clock settings co-adapted inside them.
    """
    n = len(p1)
    cuts = sorted(set(boundaries(p1)) | set(boundaries(p2)) | {int(rng.integers(1, n))} if n > 1 else {0})
    if len(cuts) == 1:
        a, b = cuts[0], n
    else:
        a, b = sorted(rng.choice(cuts, size=2, replace=False).tolist())
    layers = p1.layers[:a] + p2.layers[a:b] + p1.layers[b:]
    guards = tuple(h1 if rng.random() < 0.5 else h2 for h1, h2 in zip(p1.guards, p2.guards))
    return repair(Genome(layers, guards), model, hw)


def precision_uniform_crossover(p1: Genome, p2: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    """Keeps p1's partition; where both parents agree on the domain, each precision field
    is inherited uniformly. Mixes quantisation policies without disturbing the partition."""
    out = []
    for a, b in zip(p1.layers, p2.layers):
        if a.domain == b.domain and rng.random() < 0.5:
            out.append(replace(a, w_bits=b.w_bits if rng.random() < 0.5 else a.w_bits,
                               a_bits=b.a_bits if rng.random() < 0.5 else a.a_bits))
        else:
            out.append(a)
    return repair(Genome(tuple(out), p1.guards), model, hw)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _admissible(i: int, model: ModelGraph) -> List[Domain]:
    from .policy import admissible_domains
    return admissible_domains(i, model)


def domain_flip(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    i = int(rng.integers(len(g)))
    opts = [d for d in _admissible(i, model) if d != g.layers[i].domain]
    if not opts:
        return precision_step(g, model, hw, rng)
    d = opts[int(rng.integers(len(opts)))]
    # join a neighbouring segment's clock if one exists, so the flip does not fragment the segment
    nb = [g.layers[j] for j in (i - 1, i + 1) if 0 <= j < len(g) and g.layers[j].domain == d]
    new = nb[0] if nb else default_gene(d, hw)
    L = list(g.layers)
    L[i] = new
    return repair(Genome(tuple(L), g.guards), model, hw)


def boundary_shift(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    """Move a partition boundary one layer left or right by copying the neighbouring gene
    across it. Works for every ordered domain pair (ANN|SNN, SNN|SYM, SYM|ANN, ...)."""
    bs = boundaries(g)
    if not bs:
        return domain_flip(g, model, hw, rng)
    b = bs[int(rng.integers(len(bs)))]
    L = list(g.layers)
    if rng.random() < 0.5:
        L[b] = L[b - 1]            # boundary moves right
    else:
        L[b - 1] = L[b]            # boundary moves left
    return repair(Genome(tuple(L), g.guards), model, hw)


def _ladder(gene: LayerGene, hw: SiliconProfile, field: str) -> Tuple[int, ...]:
    if gene.domain == Domain.ANN:
        return hw.ann_bits
    if gene.domain == Domain.SNN:
        return hw.snn_w_bits if field == "w_bits" else hw.snn_mem_bits
    return (32,)


def precision_step(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    i = int(rng.integers(len(g)))
    gene = g.layers[i]
    field = "w_bits" if rng.random() < 0.6 else "a_bits"
    ladder = sorted(_ladder(gene, hw, field))
    k = ladder.index(getattr(gene, field)) if getattr(gene, field) in ladder else 0
    k = int(np.clip(k + (1 if rng.random() < 0.5 else -1), 0, len(ladder) - 1))
    L = list(g.layers)
    L[i] = replace(gene, **{field: ladder[k]})
    return repair(Genome(tuple(L), g.guards), model, hw)


def precision_cascade(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    """Homogenise weight precision across one segment to the rung of a randomly chosen member.
    Uniform precision inside a segment removes intra-segment requantisation and gives
    power-of-two-aligned scales at the segment's crossings (SPEC §6.2)."""
    segs = g.segments()
    s = segs[int(rng.integers(len(segs)))]
    if s.domain == Domain.SYM:
        return precision_step(g, model, hw, rng)
    src = g.layers[int(rng.integers(s.start, s.end + 1))]
    L = list(g.layers)
    for i in range(s.start, s.end + 1):
        L[i] = replace(L[i], w_bits=src.w_bits)
    return repair(Genome(tuple(L), g.guards), model, hw)


def _snn_segment(g: Genome, rng: Rng):
    segs = [s for s in g.segments() if s.domain == Domain.SNN]
    return segs[int(rng.integers(len(segs)))] if segs else None


def timestep_mutate(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    s = _snn_segment(g, rng)
    if s is None:
        return domain_flip(g, model, hw, rng)
    ladder = sorted(hw.timesteps)
    T = g.layers[s.start].timesteps
    k = int(np.clip(ladder.index(T) + (1 if rng.random() < 0.5 else -1), 0, len(ladder) - 1))
    L = list(g.layers)
    for i in range(s.start, s.end + 1):
        L[i] = replace(L[i], timesteps=ladder[k])
    return repair(Genome(tuple(L), g.guards), model, hw)


def coding_swap(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    s = _snn_segment(g, rng)
    if s is None:
        return domain_flip(g, model, hw, rng)
    c = Coding.TTFS if g.layers[s.start].coding == Coding.RATE else Coding.RATE
    L = list(g.layers)
    for i in range(s.start, s.end + 1):
        L[i] = replace(L[i], coding=c)
    return repair(Genome(tuple(L), g.guards), model, hw)


def plasticity_toggle(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    idx = [i for i, l in enumerate(g.layers) if l.domain == Domain.SNN]
    if not idx or hw.plasticity == "none":
        return timestep_mutate(g, model, hw, rng)
    i = idx[int(rng.integers(len(idx)))]
    L = list(g.layers)
    L[i] = replace(L[i], plastic=not L[i].plastic)
    return repair(Genome(tuple(L), g.guards), model, hw)


def guard_impl_swap(g: Genome, model: ModelGraph, hw: SiliconProfile, rng: Rng) -> Genome:
    if not g.guards:
        return precision_step(g, model, hw, rng)
    j = int(rng.integers(len(g.guards)))
    G = list(g.guards)
    G[j] = GuardGene(G[j].site, GuardImpl.FUSED if G[j].impl == GuardImpl.HOST else GuardImpl.HOST)
    return repair(Genome(g.layers, tuple(G)), model, hw)


MUTATIONS: Dict[str, Callable] = {
    "domain_flip": domain_flip,
    "boundary_shift": boundary_shift,
    "precision_step": precision_step,
    "precision_cascade": precision_cascade,
    "timestep": timestep_mutate,
    "coding_swap": coding_swap,
    "plasticity": plasticity_toggle,
    "guard_impl": guard_impl_swap,
}
CROSSOVERS: Dict[str, Callable] = {
    "segment_aligned": segment_aligned_crossover,
    "precision_uniform": precision_uniform_crossover,
}


# ---------------------------------------------------------------------------
# Adaptive pursuit
# ---------------------------------------------------------------------------

class AdaptivePursuit:
    """P_i <- P_i + beta (P_max - P_i) for i* = argmax q, else P_i + beta (P_min - P_i);
    q_i <- (1 - alpha) q_i + alpha r_i, r_i = survival rate of operator i's offspring."""

    def __init__(self, names: Sequence[str], p_min: float = 0.04, alpha: float = 0.3, beta: float = 0.3) -> None:
        self.names = list(names)
        k = len(self.names)
        self.p_min, self.p_max = p_min, 1.0 - (k - 1) * p_min
        self.alpha, self.beta = alpha, beta
        self.p = np.full(k, 1.0 / k)
        self.q = np.full(k, 0.5)

    def sample(self, rng: Rng) -> str:
        return self.names[int(rng.choice(len(self.names), p=self.p / self.p.sum()))]

    def update(self, offered: Dict[str, int], survived: Dict[str, int]) -> None:
        for i, n in enumerate(self.names):
            if offered.get(n, 0):
                self.q[i] = (1 - self.alpha) * self.q[i] + self.alpha * survived.get(n, 0) / offered[n]
        best = int(np.argmax(self.q))
        for i in range(len(self.names)):
            target = self.p_max if i == best else self.p_min
            self.p[i] += self.beta * (target - self.p[i])

    def snapshot(self) -> Dict[str, float]:
        return {n: float(p) for n, p in zip(self.names, self.p / self.p.sum())}

"""Tri-domain genome.

x = (g_1..g_n ; h_1..h_m)
  g_i = (d_i, bw_i, ba_i, c_i, T_i, p_i)   per-layer gene
      d  in {ANN, SNN, SYM}  execution domain
      bw weight bits, ba activation bits (ANN) / membrane bits (SNN) / datapath bits (SYM)
      c  in {NONE, RATE, TTFS}, T timesteps  (SNN only; shared across an SNN segment)
      p  on-device plasticity flag (SNN only)
  h_j = (site_j, impl_j)                   per-guard gene, impl in {HOST, FUSED}

Validity set V(model, hw) is defined by the invariants enforced in `repair`
(SPEC §2.2). Every operator in operators.py maps V -> V by construction or via repair.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import List, NamedTuple, Optional, Sequence, Tuple

from ..hardware.profiles import SiliconProfile
from ..ir import ModelGraph


class Domain(IntEnum):
    ANN = 0
    SNN = 1
    SYM = 2


class Coding(IntEnum):
    NONE = 0
    RATE = 1
    TTFS = 2


class GuardImpl(IntEnum):
    HOST = 0       # runs on the symbolic engine (crossing into SYM and back)
    FUSED = 1      # fused epilogue on the ANN engine (no crossing); ANN producers only


SYM_BITS = 32      # symbolic datapath is Q16.16 fixed point
TTFS_MIN_T = 4


@dataclass(frozen=True, slots=True)
class LayerGene:
    domain: Domain
    w_bits: int
    a_bits: int
    coding: Coding = Coding.NONE
    timesteps: int = 0
    plastic: bool = False

    def code(self) -> str:
        d = "ASY"[self.domain]
        if self.domain == Domain.SNN:
            c = "RT"[self.coding - 1]
            return f"{d}{self.w_bits}.{self.a_bits}{c}{self.timesteps}{'p' if self.plastic else ''}"
        return f"{d}{self.w_bits}.{self.a_bits}"


@dataclass(frozen=True, slots=True)
class GuardGene:
    site: int
    impl: GuardImpl = GuardImpl.HOST


class Segment(NamedTuple):
    domain: Domain
    start: int
    end: int          # inclusive


class Stage(NamedTuple):
    """One entry of the execution stream: a layer or a guard."""
    kind: str         # "layer" | "guard"
    index: int        # layer index, or guard site (after_layer)
    domain: Domain


class Crossing(NamedTuple):
    src: Domain
    dst: Domain
    edge: int         # index into the stage stream: crossing between stage edge and edge+1 (-1: input)
    values: int       # number of values carried on the crossing edge
    timesteps: int    # timesteps on the spiking side of the crossing (0 if none)


@dataclass(frozen=True)
class Genome:
    layers: Tuple[LayerGene, ...]
    guards: Tuple[GuardGene, ...] = ()

    @property
    def key(self) -> str:
        g = "|".join(f"{h.site}{'HF'[h.impl]}" for h in self.guards)
        return "-".join(l.code() for l in self.layers) + ("#" + g if g else "")

    def __len__(self) -> int:
        return len(self.layers)

    def segments(self) -> List[Segment]:
        out: List[Segment] = []
        i = 0
        while i < len(self.layers):
            d = self.layers[i].domain
            j = i
            while j + 1 < len(self.layers) and self.layers[j + 1].domain == d:
                j += 1
            out.append(Segment(d, i, j))
            i = j + 1
        return out

    def guard(self, site: int) -> Optional[GuardGene]:
        for h in self.guards:
            if h.site == site:
                return h
        return None

    def stages(self) -> List[Stage]:
        """Execution stream. Fused guards inherit the producer's (ANN) domain."""
        out: List[Stage] = []
        for i, g in enumerate(self.layers):
            out.append(Stage("layer", i, g.domain))
            h = self.guard(i)
            if h is not None:
                out.append(Stage("guard", i, g.domain if h.impl == GuardImpl.FUSED else Domain.SYM))
        return out

    def crossings(self, model: ModelGraph) -> List[Crossing]:
        """Domain crossings along the stream, including input encoding and output decoding."""
        st = self.stages()
        out: List[Crossing] = []

        def T_of(stage: Stage) -> int:
            return self.layers[stage.index].timesteps if stage.kind == "layer" else 0

        if st[0].domain == Domain.SNN:          # raw input must be encoded to spikes
            out.append(Crossing(Domain.ANN, Domain.SNN, -1, model.input_size(), T_of(st[0])))
        for k in range(len(st) - 1):
            a, b = st[k], st[k + 1]
            if a.domain != b.domain:
                vals = model.layers[a.index].out_neurons
                T = T_of(a) if a.domain == Domain.SNN else (T_of(b) if b.domain == Domain.SNN else 0)
                out.append(Crossing(a.domain, b.domain, k, vals, T))
        if st[-1].domain == Domain.SNN:         # network output must be decoded
            out.append(Crossing(Domain.SNN, Domain.ANN, len(st) - 1,
                                model.layers[st[-1].index].out_neurons, T_of(st[-1])))
        return out

    # compact form streamed to the dashboard
    def to_wire(self) -> dict:
        return {
            "layers": [[int(l.domain), l.w_bits, l.a_bits, int(l.coding), l.timesteps, int(l.plastic)]
                       for l in self.layers],
            "guards": [[h.site, int(h.impl)] for h in self.guards],
        }


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def snap(v: int, ladder: Sequence[int]) -> int:
    return min(ladder, key=lambda b: (abs(b - v), -b))


def default_gene(domain: Domain, hw: SiliconProfile, T: int = 8, coding: Coding = Coding.RATE) -> LayerGene:
    if domain == Domain.ANN:
        b = snap(8, hw.ann_bits)
        return LayerGene(Domain.ANN, b, b)
    if domain == Domain.SNN:
        return LayerGene(Domain.SNN, snap(4, hw.snn_w_bits), snap(16, hw.snn_mem_bits),
                         coding, snap(T, hw.timesteps))
    return LayerGene(Domain.SYM, SYM_BITS, SYM_BITS)


def uniform_genome(model: ModelGraph, hw: SiliconProfile, domain: Domain, **kw) -> Genome:
    g = Genome(tuple(default_gene(domain, hw, **kw) for _ in model.layers),
               tuple(GuardGene(s.after_layer) for s in model.guard_sites))
    return repair(g, model, hw)


# ---------------------------------------------------------------------------
# Repair: projects any genome onto the validity set V(model, hw)
# ---------------------------------------------------------------------------

def repair(g: Genome, model: ModelGraph, hw: SiliconProfile) -> Genome:
    """Enforce invariants I1..I7 (SPEC §2.2). Idempotent: repair(repair(x)) == repair(x)."""
    n = model.n
    L = list(g.layers)

    # I1 domain admissibility
    for i, (gene, spec) in enumerate(zip(L, model.layers)):
        d = gene.domain
        if d == Domain.SYM and spec.symbolic_substitute is None:
            d = Domain.ANN
        if d == Domain.SNN and not spec.spiking_admissible:
            d = Domain.ANN
        if d != gene.domain:
            L[i] = default_gene(d, hw, T=gene.timesteps or 8,
                                coding=gene.coding if gene.coding != Coding.NONE else Coding.RATE)

    # I2 precision ladders and per-domain field hygiene
    for i, gene in enumerate(L):
        if gene.domain == Domain.ANN:
            L[i] = LayerGene(Domain.ANN, snap(gene.w_bits, hw.ann_bits), snap(gene.a_bits, hw.ann_bits))
        elif gene.domain == Domain.SYM:
            L[i] = LayerGene(Domain.SYM, SYM_BITS, SYM_BITS)
        else:
            coding = gene.coding if gene.coding != Coding.NONE else Coding.RATE
            L[i] = LayerGene(Domain.SNN, snap(gene.w_bits, hw.snn_w_bits), snap(gene.a_bits, hw.snn_mem_bits),
                             coding, snap(gene.timesteps or 8, hw.timesteps), gene.plastic)

    # I3 an SNN segment shares one clock: homogenise (coding, T) by majority vote, ties -> larger T
    tmp = Genome(tuple(L))
    for seg in tmp.segments():
        if seg.domain != Domain.SNN:
            continue
        votes: dict = {}
        for i in range(seg.start, seg.end + 1):
            k = (L[i].coding, L[i].timesteps)
            votes[k] = votes.get(k, 0) + 1
        coding, T = max(votes, key=lambda k: (votes[k], k[1]))
        if coding == Coding.TTFS and T < TTFS_MIN_T:    # TTFS needs >= 2 bits of spike-time resolution
            T = min([t for t in hw.timesteps if t >= TTFS_MIN_T] or [max(hw.timesteps)])
        for i in range(seg.start, seg.end + 1):
            L[i] = replace(L[i], coding=coding, timesteps=T)

    # I4 crossing precision alignment
    #   ANN(ba=b) -> SNN : encoder accumulator lives in SNN membrane bits, need membrane >= b + 2 headroom
    #   SNN(T)    -> ANN(ba=b) : decoded spike count must fit, T <= 2^b - 1
    for i in range(1, n):
        a, b = L[i - 1], L[i]
        if a.domain == Domain.ANN and b.domain == Domain.SNN and b.a_bits < a.a_bits + 2:
            wanted = [m for m in hw.snn_mem_bits if m >= a.a_bits + 2]
            if wanted:
                L[i] = replace(b, a_bits=min(wanted))
            else:
                cap = max(hw.snn_mem_bits) - 2
                L[i - 1] = replace(a, a_bits=max([x for x in hw.ann_bits if x <= cap] or [min(hw.ann_bits)]))
    tmp = Genome(tuple(L))
    for seg in tmp.segments():
        if seg.domain != Domain.SNN or seg.end + 1 >= n:
            continue
        nxt = L[seg.end + 1]
        if nxt.domain == Domain.ANN:
            t_max = (1 << nxt.a_bits) - 1
            if L[seg.start].timesteps > t_max:
                T = max([t for t in hw.timesteps if t <= t_max] or [min(hw.timesteps)])
                for i in range(seg.start, seg.end + 1):
                    L[i] = replace(L[i], timesteps=T)

    # I5 plasticity capability
    for i, gene in enumerate(L):
        if gene.domain != Domain.SNN or not gene.plastic:
            if gene.plastic:
                L[i] = replace(gene, plastic=False)
            continue
        allowed = hw.plasticity == "any" or (hw.plasticity == "final_layer" and i == n - 1)
        if not allowed:
            L[i] = replace(gene, plastic=False)

    # I6 guards: exactly one gene per declared site; FUSED only on ANN producers with capable hardware
    guards = []
    for site in model.guard_sites:
        h = g.guard(site.after_layer) or GuardGene(site.after_layer)
        impl = h.impl
        if impl == GuardImpl.FUSED and not (hw.fused_guard and L[site.after_layer].domain == Domain.ANN):
            impl = GuardImpl.HOST
        guards.append(GuardGene(site.after_layer, impl))

    return Genome(tuple(L), tuple(guards))


def is_valid(g: Genome, model: ModelGraph, hw: SiliconProfile) -> bool:
    return repair(g, model, hw) == g

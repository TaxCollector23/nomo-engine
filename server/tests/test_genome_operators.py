import numpy as np
import pytest

from nomo.hardware.profiles import PROFILES
from nomo.models.zoo import attitude_policy, perception_cnn
from nomo.search.genome import Coding, Domain, GuardImpl, TTFS_MIN_T, is_valid, repair
from nomo.search.nsga2 import TriDomainNSGA2Optimizer
from nomo.search.evaluator import Budgets, NeurosymbolicEvaluator
from nomo.search.operators import CROSSOVERS, MUTATIONS

CASES = [(m, h) for m in (attitude_policy, perception_cnn) for h in PROFILES]


def _opt(mk, hid, seed=0):
    m, hw = mk(), PROFILES[hid]
    opt = TriDomainNSGA2Optimizer(m, hw, NeurosymbolicEvaluator(m, hw, Budgets()))
    opt.rng = np.random.default_rng(seed)
    return m, hw, opt


def _check_invariants(g, m, hw):
    for i, (gene, spec) in enumerate(zip(g.layers, m.layers)):
        if gene.domain == Domain.SYM:
            assert spec.symbolic_substitute
        if gene.domain == Domain.SNN:
            assert spec.spiking_admissible
            assert gene.w_bits in hw.snn_w_bits and gene.a_bits in hw.snn_mem_bits
            assert gene.timesteps in hw.timesteps and gene.coding != Coding.NONE
            if gene.coding == Coding.TTFS:
                assert gene.timesteps >= TTFS_MIN_T or max(hw.timesteps) < TTFS_MIN_T
            if gene.plastic:
                assert hw.plasticity == "any" or (hw.plasticity == "final_layer" and i == m.n - 1)
        else:
            assert not gene.plastic and gene.timesteps == 0 and gene.coding == Coding.NONE
        if gene.domain == Domain.ANN:
            assert gene.w_bits in hw.ann_bits and gene.a_bits in hw.ann_bits
    for seg in g.segments():                                      # I3: one clock per SNN segment
        if seg.domain == Domain.SNN:
            assert len({(g.layers[i].coding, g.layers[i].timesteps) for i in range(seg.start, seg.end + 1)}) == 1
    for i in range(1, m.n):                                       # I4: crossing alignment
        a, b = g.layers[i - 1], g.layers[i]
        if a.domain == Domain.ANN and b.domain == Domain.SNN:
            assert b.a_bits >= a.a_bits + 2 or max(hw.snn_mem_bits) < a.a_bits + 2
        if a.domain == Domain.SNN and b.domain == Domain.ANN:
            assert a.timesteps <= (1 << b.a_bits) - 1
    assert [h.site for h in g.guards] == [s.after_layer for s in m.guard_sites]
    for h in g.guards:
        if h.impl == GuardImpl.FUSED:
            assert hw.fused_guard and g.layers[h.site].domain == Domain.ANN


@pytest.mark.parametrize("mk,hid", CASES)
def test_repair_is_idempotent_projection(mk, hid):
    m, hw, opt = _opt(mk, hid)
    for _ in range(300):
        g = opt.random_genome()
        assert repair(g, m, hw) == g
        assert is_valid(g, m, hw)
        _check_invariants(g, m, hw)


@pytest.mark.parametrize("mk,hid", CASES)
def test_operators_preserve_validity(mk, hid):
    m, hw, opt = _opt(mk, hid, seed=7)
    pop = [opt.random_genome() for _ in range(40)]
    rng = np.random.default_rng(1)
    for _ in range(400):
        a, b = pop[rng.integers(len(pop))], pop[rng.integers(len(pop))]
        for name, x in CROSSOVERS.items():
            c = x(a, b, m, hw, rng)
            assert is_valid(c, m, hw), name
            _check_invariants(c, m, hw)
        for name, mu in MUTATIONS.items():
            c = mu(a, m, hw, rng)
            assert is_valid(c, m, hw), name
            _check_invariants(c, m, hw)


def test_segment_crossover_cuts_on_parent_boundaries():
    m, hw, opt = _opt(perception_cnn, "akd1500", seed=3)
    rng = np.random.default_rng(0)
    produced_new_partition = False
    for _ in range(200):
        a, b = opt.random_genome(), opt.random_genome()
        c = CROSSOVERS["segment_aligned"](a, b, m, hw, rng)
        doms = [l.domain for l in c.layers]
        if doms not in ([l.domain for l in a.layers], [l.domain for l in b.layers]):
            produced_new_partition = True
    assert produced_new_partition


def test_crossings_include_io_conversion():
    m, hw, opt = _opt(attitude_policy, "loihi2")
    from nomo.search.genome import uniform_genome
    g = uniform_genome(m, hw, Domain.SNN)
    kinds = [(c.src, c.dst) for c in g.crossings(m)]
    assert kinds[0] == (Domain.ANN, Domain.SNN)          # input encoding (layer 0 spiking)

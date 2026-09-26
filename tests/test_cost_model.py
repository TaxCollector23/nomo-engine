import math

import numpy as np
import pytest

from nomo.hardware.cost_model import HardwareCostModel
from nomo.hardware.lut import CostLUT
from nomo.hardware.profiles import AKD1500, LOIHI2, Link
from nomo.hardware.routing import Commodity, CrossingRouter
from nomo.hardware.surrogate import BayesianResidualSurrogate
from nomo.models.zoo import attitude_policy, perception_cnn
from nomo.search.genome import Domain, uniform_genome


def _async_dataflow(tau):
    """Barrier-free schedule: layer l starts step t when l-1 finished t and l finished t-1."""
    D, T = len(tau), len(tau[0])
    fin = np.zeros((D, T))
    for t in range(T):
        for l in range(D):
            start = max(fin[l - 1, t] if l else 0.0, fin[l, t - 1] if t else 0.0)
            fin[l, t] = start + tau[l][t]
    return fin[-1, -1]


def test_pipeline_latency_hand_example():
    # D=2, T=2: clocks k=0:{l0 t0}, k=1:{l0 t1, l1 t0}, k=2:{l1 t1}
    tau = [np.array([1.0, 2.0]), np.array([3.0, 0.5])]
    assert HardwareCostModel.pipeline_latency(tau) == 1.0 + max(2.0, 3.0) + 0.5


def test_barrier_pipeline_bounds_async_dataflow():
    rng = np.random.default_rng(0)
    for _ in range(200):
        D, T = int(rng.integers(1, 6)), int(rng.integers(1, 20))
        tau = [rng.random(T) + 0.1 for _ in range(D)]
        L = HardwareCostModel.pipeline_latency(tau)
        assert _async_dataflow(tau) <= L + 1e-9                  # barrier sync is an upper bound
        assert L <= sum(t.sum() for t in tau) + 1e-12             # never worse than sequential
        uni = [np.full(T, 0.7) for _ in range(D)]
        assert np.isclose(HardwareCostModel.pipeline_latency(uni), _async_dataflow(uni))
        assert np.isclose(HardwareCostModel.pipeline_latency(uni), 0.7 * (T + D - 1))


def test_routing_single_commodity_is_min_energy_path_when_time_is_free():
    links = [Link("a", "b", 1e9, 5e-12, 0), Link("b", "c", 1e9, 5e-12, 0), Link("a", "c", 1e9, 20e-12, 0)]
    r = CrossingRouter(links, lam_j_per_s=0.0).route([Commodity("a", "c", 1000.0)])
    assert np.isclose(r.energy_j, 1000 * 10e-12)


def test_routing_splits_over_parallel_paths_under_time_price():
    links = [Link("s", "t", 1e9, 1e-12, 0), Link("s", "t2", 3e9, 1e-12, 0), Link("t2", "t", 3e9, 0.0, 0)]
    d = 4e6
    r = CrossingRouter(links, lam_j_per_s=1e3).route([Commodity("s", "t", d)])
    assert np.isclose(r.tau_s, d / 4e9, rtol=1e-6)            # bandwidth-proportional split minimises tau
    assert max(r.arc_utilisation.values()) <= 1.0 + 1e-9


def test_routing_colocated_is_free():
    r = CrossingRouter(LOIHI2.links, 1.0).route([Commodity("nc_mesh", "nc_mesh", 1e6)])
    assert r.energy_j == 0 and r.tau_s == 0


def test_lut_interpolates_exactly_and_extrapolates_power_law():
    lut = CostLUT()
    key = ("akd1500", "dense", "ANN", 8, 8, 0)
    for s in (1e3, 1e4, 1e5):
        lut.add(key, s, 2e-12 * s ** 0.9, 1e-9 * s)
    assert np.isclose(lut.query(key, 1e4).energy_j, 2e-12 * 1e4 ** 0.9)
    hit = lut.query(key, 1e7)
    assert hit.source == "lut-extrapolated" and np.isclose(hit.energy_j, 2e-12 * 1e7 ** 0.9, rtol=1e-6)
    assert lut.query(("x",) * 3 + (0, 0, 0), 10) is None


def test_lut_overrides_analytic_cost():
    m = attitude_policy()
    g = uniform_genome(m, AKD1500, Domain.ANN)
    base = HardwareCostModel(AKD1500, m).evaluate(g)
    lut = CostLUT()
    for spec in m.layers:
        lut.add(("akd1500", "dense", "ANN", 8, 8, 0), spec.macs, 1e-6, 1e-4)
    rep = HardwareCostModel(AKD1500, m, lut=lut).evaluate(g)
    assert rep.measured_energy_fraction > 0.1 and rep.energy_j != base.energy_j


def test_surrogate_recovers_linear_log_residual():
    rng = np.random.default_rng(0)
    s = BayesianResidualSurrogate(dim=4, prior_var=10.0)
    w = np.array([0.2, -0.3, 0.1, 0.05])
    for _ in range(200):
        phi = np.r_[1.0, rng.normal(size=3)]
        pred = 1e-3
        s.observe(phi, pred * np.exp(phi @ w + rng.normal(0, 0.01)), pred)
    assert np.allclose(s.post.w, w, atol=0.01)
    far = np.r_[1.0, 50.0, 50.0, 50.0]
    assert s.predict(far)[1] > s.predict(np.r_[1.0, 0, 0, 0])[1]


@pytest.mark.parametrize("hw", [LOIHI2, AKD1500])
def test_snn_energy_scales_with_timesteps_and_crossings_are_charged(hw):
    m = perception_cnn()
    cm = HardwareCostModel(hw, m)
    lo = cm.evaluate(uniform_genome(m, hw, Domain.SNN, T=4))
    hi = cm.evaluate(uniform_genome(m, hw, Domain.SNN, T=16))
    assert hi.energy_j > lo.energy_j and hi.latency_s > lo.latency_s
    assert lo.crossing_energy_j > 0 and len(lo.crossings) == 2      # input encode + output decode

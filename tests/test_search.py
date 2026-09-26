import numpy as np

from nomo.hardware.profiles import AKD1500
from nomo.models.zoo import perception_cnn
from nomo.search.evaluator import Budgets, NeurosymbolicEvaluator
from nomo.search.nsga2 import NSGA2Config, TriDomainNSGA2Optimizer
from nomo.search.pareto import nondominated_mask


def _run(oracle=None, gens=20, seed=0):
    m = perception_cnn()
    ev = NeurosymbolicEvaluator(m, AKD1500, Budgets(acc_min=m.base_accuracy - 4), oracle=oracle)
    events = []
    opt = TriDomainNSGA2Optimizer(m, AKD1500, ev, NSGA2Config(pop_size=40, generations=gens, seed=seed,
                                  oracle_per_generation=2 if oracle else 0),
                                  telemetry=lambda k, d: events.append((k, d)))
    return m, ev, opt, opt.run(), events


def test_front_is_feasible_nondominated_and_hybrid():
    m, ev, opt, res, events = _run()
    assert res.front, "expected a feasible front"
    F = np.stack([e.F for e in res.front])
    assert nondominated_mask(F).all() and all(e.feasible for e in res.front)
    assert any(any(l.domain == 1 for l in e.genome.layers) for e in res.front)     # SNN present on front
    assert res.recommended in res.front


def test_archive_hypervolume_is_monotone():
    _, _, _, res, _ = _run(gens=25)
    hv = np.array(res.hv_history)
    assert np.all(np.diff(hv) >= -1e-12)


def test_telemetry_stream_is_well_formed():
    _, _, _, res, events = _run(gens=5)
    kinds = [k for k, _ in events]
    assert kinds[0] == "run.started" and kinds[-1] == "run.completed"
    gens = [d["gen"] for k, d in events if k == "gen.completed"]
    assert gens == list(range(1, len(gens) + 1))
    seen = set()
    for k, d in events:
        if k == "eval.batch":
            for it in d["items"]:
                assert it["key"] not in seen                       # each candidate streamed once
                seen.add(it["key"])


def test_rebudget_rescores_without_recomputing_costs():
    m, ev, _, res, _ = _run(gens=5)
    n = len(ev.cache)
    before = sum(e.feasible for e in ev.cache.values())
    ev.rebudget(Budgets(acc_min=m.base_accuracy - 0.5))
    assert len(ev.cache) == n and sum(e.feasible for e in ev.cache.values()) <= before


def test_oracle_promotion_recalibrates_proxy():
    truth = lambda model, g: 90.0 + 0.5 * (sum(l.w_bits for l in g.layers) / len(g.layers))
    m, ev, _, res, _ = _run(oracle=truth, gens=6)
    assert ev.oracle_results
    measured = [e for e in ev.cache.values() if e.accuracy_source == "oracle"]
    assert measured and all(np.isclose(e.accuracy, truth(m, e.genome)) for e in measured)
    assert (ev.proxy.a, ev.proxy.b) != (0.0, 1.0)

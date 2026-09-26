import numpy as np

from nomo.search.pareto import (asf_select, constrained_dominance_matrix, crowding_distance,
                                fast_non_dominated_sort, hv2d, hv3d, nondominated_mask)


def test_nds_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        F = rng.random((60, 3)).round(2)
        cv = np.where(rng.random(60) < 0.3, rng.random(60), 0.0)
        fronts = fast_non_dominated_sort(F, cv)
        D = constrained_dominance_matrix(F, cv)
        rank = np.empty(60, int)
        for r, idx in enumerate(fronts):
            rank[idx] = r
        assert sorted(np.concatenate(fronts).tolist()) == list(range(60))
        for i in range(60):
            for j in range(60):
                if D[i, j]:
                    assert rank[i] < rank[j]
        for r, idx in enumerate(fronts):
            assert not D[np.ix_(idx, idx)].any()


def test_feasible_always_beats_infeasible():
    F = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    D = constrained_dominance_matrix(F, np.array([0.0, 0.1]))
    assert D[0, 1] and not D[1, 0]


def test_hv2d_exact():
    P = np.array([[0.2, 0.6], [0.5, 0.3]])
    assert np.isclose(hv2d(P, np.array([1.0, 1.0])), 0.8 * 0.4 + 0.5 * 0.3)


def test_hv3d_against_monte_carlo():
    rng = np.random.default_rng(1)
    P = rng.random((25, 3))
    P = P[nondominated_mask(P)]
    ref = np.array([1.1, 1.1, 1.1])
    S = rng.random((400_000, 3)) * ref
    dominated = np.zeros(len(S), bool)
    for p in P:
        dominated |= np.all(S >= p, axis=1)
    mc = dominated.mean() * np.prod(ref)
    assert abs(hv3d(P, ref) - mc) < 4e-3


def test_hv3d_single_box_and_dominated_points():
    ref = np.array([1.0, 1.0, 1.0])
    assert np.isclose(hv3d(np.array([[0.5, 0.5, 0.5]]), ref), 0.125)
    assert np.isclose(hv3d(np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]), ref), 0.125)


def test_crowding_extremes_infinite_and_asf_balanced():
    F = np.array([[0, 1.0], [0.5, 0.5], [1.0, 0]])
    d = crowding_distance(F)
    assert np.isinf(d[0]) and np.isinf(d[2]) and np.isfinite(d[1])
    assert asf_select(F) == 1

"""Pareto machinery (SPEC §1.4, §1.6).

Constrained domination (Deb 2000):  x <_c y  iff
    CV(x) = 0 and CV(y) > 0, or
    CV(x) > 0 and CV(y) > 0 and CV(x) < CV(y), or
    CV(x) = CV(y) = 0 and F(x) <= F(y) componentwise with at least one strict inequality.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def constrained_dominance_matrix(F: np.ndarray, cv: np.ndarray) -> np.ndarray:
    """D[i, j] = True iff i constrained-dominates j.  O(N^2 M) vectorised."""
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    pareto = le & lt
    fi, fj = (cv == 0)[:, None], (cv == 0)[None, :]
    both_feas = fi & fj
    feas_vs_inf = fi & ~fj
    both_inf = ~fi & ~fj & (cv[:, None] < cv[None, :])
    return (both_feas & pareto) | feas_vs_inf | both_inf


def fast_non_dominated_sort(F: np.ndarray, cv: np.ndarray) -> List[np.ndarray]:
    D = constrained_dominance_matrix(F, cv)
    n_dom = D.sum(axis=0)                     # how many dominate j
    fronts: List[np.ndarray] = []
    remaining = np.ones(len(F), dtype=bool)
    while remaining.any():
        cur = np.where(remaining & (n_dom == 0))[0]
        if len(cur) == 0:                     # cannot happen for a strict partial order; guard anyway
            cur = np.where(remaining)[0]
        fronts.append(cur)
        remaining[cur] = False
        n_dom = n_dom - D[cur].sum(axis=0)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        span = F[order[-1], j] - F[order[0], j]
        d[order[0]] = d[order[-1]] = np.inf
        if span <= 0:
            continue
        d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominated = (le & lt).any(axis=0)
    return ~dominated


def normalise(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray, log_axes: Sequence[int] = (0, 1)) -> np.ndarray:
    """Map objectives into [0,1] per axis; energy and latency span decades so use log10 there."""
    X = F.astype(float).copy()
    lo, hi = ideal.astype(float).copy(), nadir.astype(float).copy()
    for j in log_axes:
        X[:, j] = np.log10(np.maximum(X[:, j], 1e-30))
        lo[j], hi[j] = np.log10(max(lo[j], 1e-30)), np.log10(max(hi[j], 1e-30))
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (X - lo) / span


def hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume (minimisation) of points dominating ref."""
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0], kind="stable")]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def hv3d(P: np.ndarray, ref: np.ndarray) -> float:
    """Exact 3-D hypervolume by slicing along f3 (HSO): sum over slabs [z_k, z_{k+1}) of
    slab thickness x HV2D of all points with f3 <= z_k.  O(n^2 log n)."""
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 2], kind="stable")]
    zs = np.append(P[:, 2], ref[2])
    hv = 0.0
    for k in range(len(P)):
        dz = zs[k + 1] - zs[k]
        if dz > 0:
            hv += dz * hv2d(P[: k + 1, :2], ref[:2])
    return hv


def asf_select(Fn: np.ndarray, weights: Optional[np.ndarray] = None, rho: float = 1e-4) -> int:
    """Augmented achievement scalarising function on normalised objectives (utopia at 0):
         s(x) = max_j w_j f_j(x) + rho * sum_j w_j f_j(x)
    Returns argmin. Equal weights pick the knee-like balanced solution."""
    w = np.ones(Fn.shape[1]) if weights is None else np.asarray(weights, float)
    s = np.max(Fn * w, axis=1) + rho * np.sum(Fn * w, axis=1)
    return int(np.argmin(s))

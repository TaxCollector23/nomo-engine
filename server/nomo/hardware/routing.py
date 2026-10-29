"""Crossing routing as a multi-commodity min-cost flow LP.

In steady-state streaming inference every domain crossing of a frame is in flight
concurrently with the crossings of neighbouring frames, so crossings compete for
the same physical links. Each crossing k is a commodity (s_k -> t_k, d_k bytes/frame).

Decision variables: f_{k,a} >= 0 bytes of commodity k on directed arc a, and tau >= 0
(the bottleneck serialisation time per frame).

    minimise    sum_k sum_a e_a f_{k,a}  +  lambda * tau
    subject to  B f_k = d_k (1_{s_k} - 1_{t_k})             for all k   (flow conservation)
                sum_k f_{k,a} / bw_a  <=  tau                 for all a   (min-max congestion)

B is the node-arc incidence matrix. lambda [J/s] prices time in energy units; the
cost model sets lambda = P_static + user weight, so spreading traffic over a
slower path is chosen exactly when the static energy it saves exceeds the
dynamic energy it costs. Solved with HiGHS via scipy.optimize.linprog; results
are memoised on the (rounded) demand signature because many genomes share
identical crossing sets.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .profiles import Link


@dataclass(frozen=True)
class Commodity:
    src: str
    dst: str
    bytes_: float


@dataclass
class RoutingResult:
    energy_j: float
    tau_s: float                          # bottleneck serialisation time per frame
    per_commodity_latency_s: List[float]
    arc_load_bytes: Dict[Tuple[str, str], float]
    arc_utilisation: Dict[Tuple[str, str], float]   # load/bw divided by tau (1.0 = bottleneck)
    status: str


class CrossingRouter:
    def __init__(self, links: Sequence[Link], lam_j_per_s: float) -> None:
        arcs: List[Tuple[str, str, Link]] = []
        for l in links:
            arcs.append((l.src, l.dst, l))
            if l.bidirectional:
                arcs.append((l.dst, l.src, l))
        self.arcs = arcs
        self.nodes = sorted({a[0] for a in arcs} | {a[1] for a in arcs})
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.lam = lam_j_per_s
        self._solve = lru_cache(maxsize=4096)(self._solve_uncached)

    def route(self, commodities: Sequence[Commodity]) -> RoutingResult:
        live = tuple(sorted((c.src, c.dst, float(np.float32(c.bytes_)))
                            for c in commodities if c.src != c.dst and c.bytes_ > 0))
        if not live:
            return RoutingResult(0.0, 0.0, [0.0] * len(commodities), {}, {}, "trivial")
        res = self._solve(live)
        # map back to caller order (co-located commodities cost nothing)
        order = {}
        for i, c in enumerate(live):
            order.setdefault((c[0], c[1], c[2]), []).append(i)
        lat = []
        for c in commodities:
            if c.src == c.dst or c.bytes_ <= 0:
                lat.append(0.0)
            else:
                lst = order[(c.src, c.dst, float(np.float32(c.bytes_)))]
                lat.append(res.per_commodity_latency_s[lst[0]])
        return RoutingResult(res.energy_j, res.tau_s, lat, res.arc_load_bytes, res.arc_utilisation, res.status)

    def _solve_uncached(self, live: Tuple[Tuple[str, str, float], ...]) -> RoutingResult:
        K, A, V = len(live), len(self.arcs), len(self.nodes)
        for s, t, _ in live:
            if s not in self.idx or t not in self.idx:
                raise ValueError(f"crossing endpoint {s}->{t} not present in the interconnect graph")
        # Scaling: raw coefficients (1/bw ~ 1e-9 s/B, e ~ 1e-12 J/B) fall below HiGHS' small-matrix
        # threshold and would be silently dropped. Solve in normalised units
        #   f' = f / D,   tau' = tau * bw_max / D,   objective / (D * e_ref)
        # so every coefficient is O(1); results are mapped back exactly.
        D = max(d for _, _, d in live)
        bw_max = max(l.bw_bytes_per_s for _, _, l in self.arcs)
        e_ref = max(max(l.e_per_byte for _, _, l in self.arcs), self.lam / bw_max, 1e-30)
        nvar = K * A + 1                                    # last variable is tau'
        c = np.zeros(nvar)
        for k in range(K):
            for a, (_, _, l) in enumerate(self.arcs):
                c[k * A + a] = l.e_per_byte / e_ref
        c[-1] = self.lam / bw_max / e_ref

        Aeq = lil_matrix((K * V, nvar))
        beq = np.zeros(K * V)
        for k, (s, t, d) in enumerate(live):
            for a, (u, v, _) in enumerate(self.arcs):
                Aeq[k * V + self.idx[u], k * A + a] = 1.0
                Aeq[k * V + self.idx[v], k * A + a] = -1.0
            beq[k * V + self.idx[s]] = d / D
            beq[k * V + self.idx[t]] = -d / D

        Aub = lil_matrix((A, nvar))
        for a, (_, _, l) in enumerate(self.arcs):
            for k in range(K):
                Aub[a, k * A + a] = bw_max / l.bw_bytes_per_s
            Aub[a, nvar - 1] = -1.0
        bub = np.zeros(A)

        sol = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                      bounds=[(0, None)] * nvar, method="highs")
        if sol.status != 0:
            raise RuntimeError(f"routing LP failed: {sol.message}")
        x = sol.x
        f = x[:-1].reshape(K, A) * D
        f[f < 1e-9 * max(1.0, f.max())] = 0.0
        load = f.sum(axis=0)
        tau = float(max((load[a] / l.bw_bytes_per_s for a, (_, _, l) in enumerate(self.arcs)), default=0.0))
        energy = float(sum(load[a] * l.e_per_byte for a, (_, _, l) in enumerate(self.arcs)))

        lat = []
        for k, (_, _, d) in enumerate(live):
            frac = f[k] / d
            hop = float(sum(frac[a] * l.lat_s for a, (_, _, l) in enumerate(self.arcs)))
            used = [a for a in range(A) if f[k, a] > 0]
            ser = max((load[a] / self.arcs[a][2].bw_bytes_per_s for a in used), default=0.0)
            lat.append(hop + ser)

        arc_load = {(u, v): float(load[a]) for a, (u, v, _) in enumerate(self.arcs) if load[a] > 0}
        util = {(u, v): float(load[a] / l.bw_bytes_per_s / tau) if tau > 0 else 0.0
                for a, (u, v, l) in enumerate(self.arcs) if load[a] > 0}
        return RoutingResult(energy, tau, lat, arc_load, util, "optimal")

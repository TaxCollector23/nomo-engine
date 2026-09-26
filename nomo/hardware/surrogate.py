"""Surrogate residual predictor.

The analytic+LUT model is composable per layer but misses whole-graph effects
(core contention, cache/SRAM spills, host scheduling jitter). We learn the
*log residual* between measured and predicted whole-network metrics:

    r(x) = log y_measured(x) - log y_model(x) = phi(x)^T w + eps,   eps ~ N(0, sigma^2)

with a conjugate Normal-Inverse-Gamma prior  w | s2 ~ N(0, s2 * V0),  s2 ~ IG(a0, b0).
Posterior (closed form, Bishop §3.3 / Murphy §11.7):

    Vn = (V0^-1 + Phi^T Phi)^-1
    wn = Vn Phi^T r
    an = a0 + N/2
    bn = b0 + 1/2 (r^T r - wn^T Vn^-1 wn)

Predictive is Student-t with 2 an dof, mean phi^T wn and scale^2 (bn/an)(1 + phi^T Vn phi).
The corrected metric is  y = y_model * exp(phi^T wn)  (median of the log-normal).
`acquire` ranks candidates for on-silicon measurement by predictive variance
(pure exploration) or by an expected-improvement-style score restricted to the
current Pareto front, so measurement budget is spent where it moves decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..search.genome import Domain, Genome


def features(g: Genome, report) -> np.ndarray:
    """phi(x): scale-free descriptors of a mapped candidate. Length is fixed (D=12)."""
    n = len(g.layers)
    dom = np.array([l.domain for l in g.layers])
    frac = [float(np.mean(dom == d)) for d in (Domain.ANN, Domain.SNN, Domain.SYM)]
    T = [l.timesteps for l in g.layers if l.domain == Domain.SNN]
    wb = np.mean([l.w_bits for l in g.layers])
    cross = len(report.crossings)
    cores = report.cores_used / max(1, report.core_budget)
    util = report.max_link_utilisation
    e_share_snn = report.energy_by_domain.get("SNN", 0.0) / max(report.energy_j, 1e-30)
    e_share_x = report.crossing_energy_j / max(report.energy_j, 1e-30)
    return np.array([
        1.0,
        frac[1], frac[2],
        np.log2(np.mean(T)) if T else 0.0,
        wb / 8.0,
        cross / max(1, n),
        cores,
        util,
        e_share_snn,
        e_share_x,
        float(any(l.plastic for l in g.layers)),
        np.log10(max(report.latency_s, 1e-9)) + 6.0,
    ])


@dataclass
class _Post:
    w: np.ndarray
    V: np.ndarray
    a: float
    b: float


class BayesianResidualSurrogate:
    def __init__(self, dim: int = 12, prior_var: float = 0.25, a0: float = 2.0, b0: float = 0.02) -> None:
        self.dim = dim
        self.V0inv = np.eye(dim) / prior_var
        self.a0, self.b0 = a0, b0
        self.Phi: List[np.ndarray] = []
        self.R: List[float] = []
        self.post = _Post(np.zeros(dim), np.eye(dim) * prior_var, a0, b0)

    @property
    def n_obs(self) -> int:
        return len(self.R)

    def observe(self, phi: np.ndarray, measured: float, predicted: float) -> None:
        if measured <= 0 or predicted <= 0:
            raise ValueError("metrics must be positive to take log residuals")
        self.Phi.append(np.asarray(phi, float))
        self.R.append(float(np.log(measured) - np.log(predicted)))
        self._update()

    def _update(self) -> None:
        P = np.stack(self.Phi)
        r = np.asarray(self.R)
        Vinv = self.V0inv + P.T @ P
        V = np.linalg.inv(Vinv)
        w = V @ (P.T @ r)
        a = self.a0 + len(r) / 2.0
        b = self.b0 + 0.5 * float(r @ r - w @ Vinv @ w)
        self.post = _Post(w, V, a, max(b, 1e-12))

    def predict(self, phi: np.ndarray) -> Tuple[float, float]:
        """Return (mean log-residual, predictive variance)."""
        p = self.post
        mean = float(phi @ p.w)
        scale2 = (p.b / p.a) * (1.0 + float(phi @ p.V @ phi))
        dof = 2.0 * p.a
        var = scale2 * dof / (dof - 2.0) if dof > 2 else float("inf")
        return mean, var

    def correct(self, phi: np.ndarray, predicted: float) -> float:
        if self.n_obs == 0:
            return predicted
        mean, _ = self.predict(phi)
        return float(predicted * np.exp(mean))

    def acquire(self, phis: Sequence[np.ndarray], k: int) -> List[int]:
        """Indices of the k candidates with highest predictive variance (measure these next)."""
        scores = [self.predict(p)[1] for p in phis]
        return list(np.argsort(scores)[::-1][:k])

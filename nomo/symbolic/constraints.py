"""Symbolic (deterministic) nodes.

Two families:

* SymbolicConstraint: a projection y <- Pi_C(y ; aux) onto a physically admissible
  set C, applied to a network output. Projections here are exact Euclidean
  projections because every set is a box (possibly state-dependent), so
  Pi_C = componentwise clamp to [lo(aux), hi(aux)].

* LinearODESubstitute: replaces a learned layer by a known discrete-time LTI map
  obtained by zero-order-hold discretisation of  xdot = A x + B u:
      Ad = e^{A dt},  Bd = int_0^dt e^{A s} ds B      (via the block-matrix exponential)
      y  = [Ad Bd] [x; u]

Each node has two semantics that MUST agree:
  forward_float(...)   reference in float64 (used for verification and accuracy calibration)
  forward_q16(...)     bit-exact Q16.16 integer semantics, identical to the emitted C11 code

Fixed-point conventions (shared with runtime/qformat.py and export/c11.py):
  q16(x)       = round_half_up(x * 2^16) as int32
  q16mul(a, b) = asr(a*b + 2^15, 16)   computed in int64
  q16div by constants is done by multiplying by a precomputed reciprocal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from ..runtime.qformat import Q16_ONE, check64, q16, q16mul, sat32

# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxBound:
    """y_i in [lo_i, hi_i] for channels `channels`."""
    channels: Tuple[int, ...]
    lo: Tuple[float, ...]
    hi: Tuple[float, ...]
    kind: str = "box"


@dataclass(frozen=True)
class ThrustLimit:
    """0 <= T <= T_max * (rho / rho0) ; density ratio is a runtime aux input (aerodynamic derating)."""
    channel: int
    t_max_n: float
    aux_density_ratio: int          # index into aux vector holding rho/rho0
    kind: str = "thrust"


@dataclass(frozen=True)
class RotationalRateBound:
    """Bound commanded body torques so the one-step-ahead body rates stay admissible.

    Rigid body (principal axes, I = diag(Ix, Iy, Iz)):  I wdot = tau - w x (I w)
    Explicit Euler over dt:  w+ = w + dt I^{-1} (tau - g),   g = w x (I w)
      g_x = (Iz - Iy) w_y w_z,  g_y = (Ix - Iz) w_z w_x,  g_z = (Iy - Ix) w_x w_y
    Requiring |w+_i| <= w_max gives, per axis (separable, so the clamp is the exact projection):
      tau_i in [ I_i(-w_max - w_i)/dt + g_i ,  I_i(w_max - w_i)/dt + g_i ]  cap  [-tau_max, tau_max]
    Actuator saturation is a hard physical limit; the rate interval is a safety target. If the
    two intervals are disjoint (the body is already spinning too fast for the actuators to restore
    admissibility in one step), the command saturates at the actuator limit on the side of the rate
    interval, i.e. maximum recovery authority:  tau = tau_max if lo_rate > tau_max else -tau_max.
    """
    channels: Tuple[int, int, int]
    inertia: Tuple[float, float, float]
    dt: float
    w_max: float
    tau_max: float
    aux_omega: Tuple[int, int, int]        # indices into aux vector holding body rates
    kind: str = "rotational_rate"


ConstraintTerm = object  # BoxBound | ThrustLimit | RotationalRateBound


@dataclass(frozen=True)
class SymbolicConstraint:
    id: str
    terms: Tuple[ConstraintTerm, ...]
    n_out: int
    n_aux: int = 0
    description: str = ""

    # ---- cost-model hooks
    def flops(self) -> float:
        f = 0.0
        for t in self.terms:
            if isinstance(t, BoxBound):
                f += 2 * len(t.channels)
            elif isinstance(t, ThrustLimit):
                f += 4
            elif isinstance(t, RotationalRateBound):
                f += 3 * 12 + 6          # gyroscopic term, bounds, clamps
        return f

    def memory_bytes(self) -> float:
        return 64.0 * len(self.terms)

    # ---- float reference
    def forward_float(self, y: np.ndarray, aux: Optional[np.ndarray] = None) -> np.ndarray:
        y = np.array(y, dtype=np.float64, copy=True)
        for t in self.terms:
            if isinstance(t, BoxBound):
                for c, lo, hi in zip(t.channels, t.lo, t.hi):
                    y[c] = min(max(y[c], lo), hi)
            elif isinstance(t, ThrustLimit):
                cap = t.t_max_n * float(aux[t.aux_density_ratio])
                y[t.channel] = min(max(y[t.channel], 0.0), cap)
            elif isinstance(t, RotationalRateBound):
                w = np.array([aux[i] for i in t.aux_omega], dtype=np.float64)
                I = np.array(t.inertia)
                g = np.array([(I[2] - I[1]) * w[1] * w[2], (I[0] - I[2]) * w[2] * w[0], (I[1] - I[0]) * w[0] * w[1]])
                for ax, c in enumerate(t.channels):
                    r_lo = I[ax] * (-t.w_max - w[ax]) / t.dt + g[ax]
                    r_hi = I[ax] * (t.w_max - w[ax]) / t.dt + g[ax]
                    if r_lo > t.tau_max:                # disjoint: saturate toward the rate interval
                        y[c] = t.tau_max
                    elif r_hi < -t.tau_max:
                        y[c] = -t.tau_max
                    else:
                        y[c] = min(max(y[c], max(r_lo, -t.tau_max)), min(r_hi, t.tau_max))
        return y

    # ---- bit-exact Q16.16 semantics (mirrors export/c11.py)
    def forward_q16(self, y: Sequence[int], aux: Optional[Sequence[int]] = None) -> List[int]:
        y = [int(v) for v in y]
        for t in self.terms:
            if isinstance(t, BoxBound):
                for c, lo, hi in zip(t.channels, t.lo, t.hi):
                    y[c] = min(max(y[c], q16(lo)), q16(hi))
            elif isinstance(t, ThrustLimit):
                cap = q16mul(q16(t.t_max_n), int(aux[t.aux_density_ratio]))
                y[t.channel] = min(max(y[t.channel], 0), cap)
            elif isinstance(t, RotationalRateBound):
                p = rot_params_q16(t)
                w = [int(aux[i]) for i in t.aux_omega]
                I = p["I"]
                g = [q16mul(q16mul(I[2] - I[1], w[1]), w[2]),
                     q16mul(q16mul(I[0] - I[2], w[2]), w[0]),
                     q16mul(q16mul(I[1] - I[0], w[0]), w[1])]
                for ax, c in enumerate(t.channels):
                    r_lo = sat32(q16mul(q16mul(I[ax], -p["w_max"] - w[ax]), p["inv_dt"]) + g[ax])
                    r_hi = sat32(q16mul(q16mul(I[ax], p["w_max"] - w[ax]), p["inv_dt"]) + g[ax])
                    if r_lo > p["tau_max"]:
                        y[c] = p["tau_max"]
                    elif r_hi < -p["tau_max"]:
                        y[c] = -p["tau_max"]
                    else:
                        y[c] = min(max(y[c], max(r_lo, -p["tau_max"])), min(r_hi, p["tau_max"]))
        return y

    def to_meta(self) -> Dict[str, object]:
        return {"id": self.id, "n_out": self.n_out, "n_aux": self.n_aux, "description": self.description,
                "terms": [dict(t.__dict__) for t in self.terms]}


def rot_params_q16(t: RotationalRateBound) -> Dict[str, object]:
    return {"I": [q16(v) for v in t.inertia], "inv_dt": q16(1.0 / t.dt),
            "w_max": q16(t.w_max), "tau_max": q16(t.tau_max)}


# ---------------------------------------------------------------------------
# ODE substitute
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearODESubstitute:
    id: str
    A: np.ndarray = field(repr=False)
    B: np.ndarray = field(repr=False)
    dt: float = 0.01
    description: str = ""

    @property
    def n_x(self) -> int:
        return self.A.shape[0]

    @property
    def n_u(self) -> int:
        return self.B.shape[1]

    def phi(self) -> np.ndarray:
        """[Ad Bd] via exp([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]]  (Van Loan)."""
        nx, nu = self.n_x, self.n_u
        M = np.zeros((nx + nu, nx + nu))
        M[:nx, :nx] = self.A
        M[:nx, nx:] = self.B
        E = expm(M * self.dt)
        return np.hstack([E[:nx, :nx], E[:nx, nx:]])

    def flops(self) -> float:
        return 2.0 * self.n_x * (self.n_x + self.n_u)

    def memory_bytes(self) -> float:
        return 4.0 * self.n_x * (self.n_x + self.n_u)

    def forward_float(self, xu: np.ndarray) -> np.ndarray:
        return self.phi() @ np.asarray(xu, dtype=np.float64)

    def phi_q16(self) -> np.ndarray:
        return np.vectorize(q16)(self.phi()).astype(np.int64)

    def forward_q16(self, xu_q16: Sequence[int]) -> List[int]:
        P = self.phi_q16()
        out = []
        for r in range(P.shape[0]):
            acc = 0
            for c in range(P.shape[1]):
                acc = check64(acc + int(P[r, c]) * int(xu_q16[c]))   # Q32.32 accumulate in int64
            out.append(sat32(check64(acc + (1 << 15)) >> 16))
        return out

    def to_meta(self) -> Dict[str, object]:
        return {"id": self.id, "dt": self.dt, "n_x": self.n_x, "n_u": self.n_u, "description": self.description}


Q16_ONE_ = Q16_ONE  # re-export for callers

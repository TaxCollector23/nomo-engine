"""Reference workloads.

attitude_policy  A small-UAV attitude controller: sensor fusion MLP -> learned state
                 predictor (substitutable by a rigid-body LTI model) -> torque/thrust head,
                 with a mandatory physical-limits guard on the actuator commands.
                 Fully compilable end to end (dense chain), used by the C11 tests.

perception_cnn   An event-camera perception backbone for search-only benchmarking
                 (conv layers are costed; the C11 backend lowers dense chains only).

Sensitivities are representative values; production runs populate them from the
one-at-a-time calibration sweep (SPEC §5.2).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..ir import GuardSite, LayerSensitivity, ModelGraph, conv2d, dense
from ..symbolic.constraints import (LinearODESubstitute, RotationalRateBound, SymbolicConstraint,
                                    ThrustLimit)

# Airframe constants for the reference quadrotor (principal inertias in kg m^2).
INERTIA = (0.0023, 0.0023, 0.0040)
DT = 0.004                      # 250 Hz control loop


def rigid_body_substitute() -> LinearODESubstitute:
    """Linearised hover dynamics for x = [p, q, r, phi, theta, psi] (body rates, attitude errors),
    u = [tau_x, tau_y, tau_z, c1, c2, c3] where c are feed-forward rate corrections.
      pdot = tau_x/Ix, ..., phidot = p + c1, ...
    """
    A = np.zeros((6, 6))
    A[3, 0] = A[4, 1] = A[5, 2] = 1.0
    B = np.zeros((6, 6))
    B[0, 0], B[1, 1], B[2, 2] = 1 / INERTIA[0], 1 / INERTIA[1], 1 / INERTIA[2]
    B[3, 3] = B[4, 4] = B[5, 5] = 1.0
    return LinearODESubstitute("rigid_body_lti", A, B, DT, "ZOH-discretised linearised rigid-body attitude dynamics")


def actuator_guard() -> SymbolicConstraint:
    return SymbolicConstraint(
        id="actuator_limits",
        terms=(
            RotationalRateBound(channels=(0, 1, 2), inertia=INERTIA, dt=DT, w_max=8.0, tau_max=0.35,
                                aux_omega=(0, 1, 2)),
            ThrustLimit(channel=3, t_max_n=18.0, aux_density_ratio=3),
        ),
        n_out=4, n_aux=4,
        description="Body-rate-admissible torque projection (Euler rigid body) and density-derated thrust cap",
    )


def attitude_policy() -> ModelGraph:
    S = LayerSensitivity
    layers = [
        dense("fuse0", 24, 128, sensitivity=S(q_w=0.6, q_a=0.5, c_rate=2.2, c_ttfs=2.6), base_rate=0.18),
        dense("fuse1", 128, 128, sensitivity=S(q_w=0.3, q_a=0.3, c_rate=0.9, c_ttfs=1.1), base_rate=0.12),
        dense("fuse2", 128, 96, sensitivity=S(q_w=0.25, q_a=0.25, c_rate=0.7, c_ttfs=0.9), base_rate=0.10),
        dense("fuse3", 96, 64, sensitivity=S(q_w=0.2, q_a=0.2, c_rate=0.6, c_ttfs=0.8), base_rate=0.09),
        dense("state_head", 64, 12, activation="linear", sensitivity=S(q_w=0.4, q_a=0.4, c_rate=1.4, c_ttfs=1.6), base_rate=0.10),
        dense("dynamics", 12, 6, activation="linear", symbolic_substitute="rigid_body_lti",
              sensitivity=S(q_w=0.5, q_a=0.5, c_rate=1.8, c_ttfs=2.0, sym=0.15), base_rate=0.10),
        dense("torque_head", 6, 4, activation="linear", sensitivity=S(q_w=0.7, q_a=0.6, c_rate=2.5, c_ttfs=2.8), base_rate=0.12),
    ]
    return ModelGraph(
        name="attitude_policy", input_shape=(24,), layers=layers, base_accuracy=97.0,
        guard_sites=[GuardSite(after_layer=6, constraint_id="actuator_limits", mandatory=True)],
        constraints={"actuator_limits": actuator_guard()},
        substitutes={"rigid_body_lti": rigid_body_substitute()},
        input_rate=0.2,
    )


def perception_cnn() -> ModelGraph:
    S = LayerSensitivity
    layers = [
        conv2d("conv1", 2, 32, 3, 128, 128, 2, sensitivity=S(q_w=0.8, q_a=0.6, c_rate=1.6, c_ttfs=2.0), base_rate=0.08),
        conv2d("conv2", 32, 64, 3, 64, 64, 2, sensitivity=S(q_w=0.4, q_a=0.4, c_rate=0.8, c_ttfs=1.0), base_rate=0.07),
        conv2d("conv3", 64, 128, 3, 32, 32, 2, sensitivity=S(q_w=0.3, q_a=0.3, c_rate=0.6, c_ttfs=0.8), base_rate=0.06),
        conv2d("conv4", 128, 128, 3, 16, 16, 1, sensitivity=S(q_w=0.25, q_a=0.25, c_rate=0.5, c_ttfs=0.7), base_rate=0.06),
        conv2d("conv5", 128, 256, 3, 16, 16, 2, sensitivity=S(q_w=0.25, q_a=0.25, c_rate=0.5, c_ttfs=0.7), base_rate=0.05),
        dense("fc1", 256 * 8 * 8, 256, sensitivity=S(q_w=0.3, q_a=0.3, c_rate=0.7, c_ttfs=0.9), base_rate=0.08),
        dense("fc2", 256, 11, activation="linear", sensitivity=S(q_w=0.8, q_a=0.7, c_rate=1.8, c_ttfs=2.2), base_rate=0.12),
    ]
    return ModelGraph(name="perception_cnn", input_shape=(2, 128, 128), layers=layers, base_accuracy=95.6,
                      input_rate=0.05)


MODELS = {"attitude_policy": attitude_policy, "perception_cnn": perception_cnn}


def synthetic_weights(model: ModelGraph, seed: int = 0):
    """He-initialised stand-in weights for tests and demos. The learned 'dynamics' layer is
    initialised to its analytic substitute plus noise, mimicking a trained approximation."""
    rng = np.random.default_rng(seed)
    out = {}
    for spec in model.layers:
        if spec.op != "dense":
            continue
        n_out, n_in = spec.weight_shape
        W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_out, n_in))
        b = rng.normal(0.0, 0.05, size=n_out)
        if spec.symbolic_substitute and spec.symbolic_substitute in model.substitutes:
            W = model.substitutes[spec.symbolic_substitute].phi() + rng.normal(0, 0.01, size=(n_out, n_in))
            b = np.zeros(n_out)
        out[spec.name] = (W, b)
    return out


def attitude_calibration(n: int = 256, seed: int = 1):
    """Sensor frames and aux state (body rates rad/s, density ratio) for calibration/tests."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, 24))
    omega = rng.normal(0.0, 3.0, size=(n, 3))
    rho = rng.uniform(0.7, 1.05, size=(n, 1))
    return X, np.hstack([omega, rho])

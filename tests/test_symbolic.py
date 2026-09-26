import numpy as np

from nomo.models.zoo import INERTIA, actuator_guard, rigid_body_substitute
from nomo.runtime.qformat import q16, quantize_multiplier, requant
from nomo.symbolic.constraints import LinearODESubstitute


def _next_rates(tau, w, dt=0.004):
    I = np.array(INERTIA)
    g = np.cross(w, I * w)
    return w + dt * (tau - g) / I


def test_rotational_projection_keeps_rates_admissible_when_feasible():
    con = actuator_guard()
    rot = con.terms[0]
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(2000):
        w = rng.normal(0, 3, 3)
        y = rng.normal(0, 1.0, 4)
        out = con.forward_float(y, np.r_[w, 0.9])
        assert np.all(np.abs(out[:3]) <= rot.tau_max + 1e-12)
        if np.all(np.abs(w) < rot.w_max - 0.5):
            assert np.all(np.abs(_next_rates(out[:3], w)) <= rot.w_max + 1e-9)
            checked += 1
        assert 0.0 <= out[3] <= 18.0 * 0.9 + 1e-12
    assert checked > 1000


def test_disjoint_case_saturates_toward_recovery():
    con = actuator_guard()
    rot = con.terms[0]
    w = np.array([0.0, -40.0, 0.0])            # far beyond w_max: needs more torque than actuators have
    out = con.forward_float(np.array([0.0, -0.3, 0.0, 1.0]), np.r_[w, 1.0])
    assert out[1] == rot.tau_max                # full positive torque to decelerate the negative spin


def test_projection_is_identity_inside_and_idempotent():
    con = actuator_guard()
    aux = np.array([0.1, -0.2, 0.05, 1.0])
    y = np.array([0.01, -0.02, 0.0, 5.0])
    assert np.allclose(con.forward_float(y, aux), y)
    z = con.forward_float(np.array([5.0, -5.0, 5.0, 99.0]), aux)
    assert np.allclose(con.forward_float(z, aux), z)


def test_q16_guard_tracks_float_reference():
    con = actuator_guard()
    rng = np.random.default_rng(2)
    for _ in range(500):
        w = rng.normal(0, 3, 3)
        aux = np.r_[w, rng.uniform(0.7, 1.05)]
        y = rng.normal(0, 0.5, 4) * np.array([1, 1, 1, 30])
        ref = con.forward_float(y, aux)
        q = np.array(con.forward_q16([q16(v) for v in y], [q16(v) for v in aux])) / 65536.0
        assert np.allclose(q, ref, atol=3e-3)


def test_zoh_discretisation_matches_closed_form():
    sub = rigid_body_substitute()
    P = sub.phi()
    dt = sub.dt
    # A nilpotent (A^2 = 0): Ad = I + A dt, Bd = (I dt + A dt^2/2) B
    Ad = np.eye(6) + sub.A * dt
    Bd = (np.eye(6) * dt + sub.A * dt ** 2 / 2) @ sub.B
    assert np.allclose(P, np.hstack([Ad, Bd]))
    lin = LinearODESubstitute("x", np.zeros((2, 2)), np.eye(2), 0.5)
    assert np.allclose(lin.phi(), np.hstack([np.eye(2), 0.5 * np.eye(2)]))


def test_quantize_multiplier_precision():
    rng = np.random.default_rng(0)
    for m in 10 ** rng.uniform(-6, 3, 500):
        m0, sh = quantize_multiplier(m)
        assert 2 ** 30 <= m0 < 2 ** 31
        assert abs(m0 * 2.0 ** -(31 + sh) - m) / m < 1e-9
        acc = int(rng.integers(-2 ** 20, 2 ** 20))
        assert abs(requant(acc, m0, sh) - acc * m) <= 0.5 + abs(acc * m) * 1e-9

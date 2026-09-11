"""Measure-independent box rules: sup error on the box and node counts."""
import numpy as np
import pytest

from minimax.uniform_rule import (
    box_samples,
    build_uniform_rule,
    rule_roundoff_amplification,
    rule_sup_error,
)

ETA = 0.02


def _dense_boundary(box, times):
    """An independent cloud: the four edges, uniform at 12 points per half wave
    of the largest |t| (and of 1/d at the peak), plus geometric points toward
    small |Re d|.  No code shared with the builder's clouds."""
    re_lo, re_hi, im_lo, im_hi = box
    h = min(np.pi / (12.0 * np.max(np.abs(times))), im_lo / 12.0)
    x = np.linspace(re_lo, re_hi, int((re_hi - re_lo) / h) + 2)
    if re_lo > 0.0:
        x = np.union1d(x, np.geomspace(re_lo, re_hi, 20000))
    if re_hi < 0.0:
        x = np.union1d(x, -np.geomspace(-re_hi, -re_lo, 20000))
    y = np.union1d(np.linspace(im_lo, im_hi, int((im_hi - im_lo) / h) + 2),
                   np.geomspace(im_lo, max(im_hi, im_lo * 1.0001), 2000))
    return np.concatenate([x + 1j * im_lo, x + 1j * im_hi, re_lo + 1j * y, re_hi + 1j * y])


def _check(rule, box, eps):
    """The certificate is honest in both directions on an independent dense
    boundary: the sampled sup (a lower bound of the true sup) meets eps, and
    the rule's refined sup_error is not below it."""
    d = _dense_boundary(box, rule.times)
    rho = np.abs(d) if rule.relative else None
    sup, kappa = rule_sup_error(rule.times, rule.weights, d, rho)
    assert sup <= eps, (sup, eps)
    assert rule.sup_error >= sup * (1.0 - 1.0e-3), (rule.sup_error, sup)
    assert kappa <= 1.0e4
    assert np.all(np.isfinite(rule.times)) and np.all(rule.times != 0.0)
    return sup


def test_sign_definite_box_is_laplace_cheap():
    # 1/d on Re d in [2 eta, 400 eta]: Braess-Hackbusch regime.  In RELATIVE
    # error (the currency of a sign-definite box) with Im d up to 30 eta this
    # is 17 nodes; Hackbusch's real-interval tables give ~12 for R = 200.
    box = (2.0 * ETA, 400.0 * ETA, ETA, 30.0 * ETA)
    rule = build_uniform_rule(box, 1.0e-4, time_budget=30.0)
    _check(rule, box, 1.0e-4)
    assert rule.relative
    assert rule.theta_deg < -40.0                        # rotated toward imaginary time
    assert rule.node_count <= 20


def test_negative_sign_definite_box_rotates_the_other_way():
    box = (-100.0 * ETA, -4.0 * ETA, ETA, 20.0 * ETA)
    rule = build_uniform_rule(box, 1.0e-4, time_budget=30.0)
    _check(rule, box, 1.0e-4)
    assert rule.relative
    assert rule.theta_deg >= 40.0                        # the scan's grid includes 40 exactly
    assert rule.node_count <= 16


def test_crossing_box_count_follows_bandwidth():
    # Symmetric crossing box of real width B = 40 eta: real-time ray,
    # count near the Gauss estimate 0.5*(B/eta)*ln(10/eps)/pi ~= 73 for 1e-4.
    box = (-20.0 * ETA, 20.0 * ETA, ETA, 10.0 * ETA)
    rule = build_uniform_rule(box, 1.0e-4, time_budget=60.0)
    _check(rule, box, 1.0e-4)
    assert abs(rule.theta_deg) < 1.0
    assert rule.node_count <= 100                        # interpolatory would be ~150


def test_invalid_box_refuses():
    with pytest.raises(ValueError):
        build_uniform_rule((0.0, 1.0, 0.0, 1.0), 1.0e-4)
    with pytest.raises(ValueError):
        build_uniform_rule((1.0, 0.0, 0.1, 1.0), 1.0e-4)


def test_roundoff_amplification_uses_the_error_currency():
    d = np.asarray([0.0 + 0.1j, 10.0 + 0.1j])
    times = np.asarray([0.2 + 0.0j, 0.7 + 0.0j])
    weights = np.asarray([0.6 - 0.1j, 0.3 + 0.05j])
    terms = np.exp(1j * d[:, None] * times[None, :]) * weights[None, :]
    mass = np.sum(np.abs(terms), axis=1)

    peak = rule_roundoff_amplification(times, weights, d, 0.1)
    relative = rule_roundoff_amplification(times, weights, d, np.abs(d))
    assert peak == pytest.approx(np.max(0.1 * mass))
    assert relative == pytest.approx(np.max(np.abs(d) * mass))
    assert relative > 50.0 * peak


def test_random_boxes_never_refuse_and_hold_on_a_finer_cloud():
    """Property test: every finite box yields an accepted rule, and the sup
    bound holds on a cloud finer than the one the rule was fitted on.
    Crossing, sign-definite (R up to 1e4) and nearly sign-definite boxes."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        kind = rng.choice(["crossing", "sd+", "sd-", "near+", "near-"])
        im_hi = ETA * 10 ** rng.uniform(0, 2)
        if kind == "crossing":
            lo, hi = -ETA * rng.uniform(5, 40), ETA * rng.uniform(5, 30)
        elif kind == "sd+":
            lo = ETA * 10 ** rng.uniform(-0.5, 1.5); hi = lo * 10 ** rng.uniform(0.5, 4)
        elif kind == "sd-":
            hi = -ETA * 10 ** rng.uniform(-0.5, 1.5); lo = hi * 10 ** rng.uniform(0.5, 4)
        elif kind == "near+":
            lo = -ETA * rng.uniform(0.1, 3); hi = ETA * rng.uniform(10, 200)
        else:
            hi = ETA * rng.uniform(0.1, 3); lo = -ETA * rng.uniform(10, 200)
        eps = float(rng.choice([1e-3, 1e-4]))
        box = (lo, hi, ETA, im_hi)
        rule = build_uniform_rule(box, eps, time_budget=5.0)
        assert rule.relative == kind.startswith("sd")
        _check(rule, box, eps)


def test_thin_boxes_certify_on_the_dense_boundary():
    """Real-pole windows have Im d in [eta, 1.01 eta].  On real time the rule's
    error oscillates at its node horizon at every Re d, so a crossing rule must
    hold far from Re d = 0 too (a geometric far field certified 247x-eps rules);
    a thin tail must hold at its near corner (1.8x eps on main, 2026-09-11)."""
    for box in ((-24.0 * ETA, 38.0 * ETA, ETA, 1.01 * ETA),     # Na B06-like crossing
                (1.05 * ETA, 960.0 * ETA, ETA, 1.01 * ETA)):    # Na tail-like
        _check(build_uniform_rule(box, 1.0e-4, time_budget=20.0), box, 1.0e-4)


def test_far_sign_definite_box_samples_stay_inside_the_box():
    """The geometric far field used to start at 30 im_lo even when the box
    starts beyond it, so the angle scan (and, before the boundary cloud, the
    fit and the certificate) saw points outside the box."""
    lo, hi = 77.0 * ETA, 243.0 * ETA
    d = box_samples(lo, hi, ETA, 1.01 * ETA)
    assert d.real.min() >= lo and d.real.max() <= hi


def test_jax_backend_on_cpu_matches_numpy_on_a_small_crossing_box(monkeypatch):
    """The jax reducer (forced, on the CPU device) reaches an accepted rule
    within a few nodes of the numpy one: same algorithm, different
    floating-point route (CholeskyQR2 weights, Cholesky per damping)."""
    pytest.importorskip("jax")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    box = (-8.0 * ETA, 8.0 * ETA, ETA, 5.0 * ETA)          # rank ~35: both finish in seconds
    ref = build_uniform_rule(box, 1.0e-4, time_budget=30.0, backend="numpy")
    rule = build_uniform_rule(box, 1.0e-4, time_budget=30.0, backend="jax")
    _check(rule, box, 1.0e-4)
    assert abs(rule.node_count - ref.node_count) <= 3, (rule.node_count, ref.node_count)


def test_step_budget_is_deterministic_and_ignores_the_clock():
    """``reduction_steps`` makes the accepted rule a function of the inputs:
    two builds agree bit for bit, and a wall budget passed alongside is
    ignored (the same rule again), unlike the wall-clock mode whose node
    count depends on how far the reduction got before the deadline."""
    box = (2.0 * ETA, 400.0 * ETA, ETA, 30.0 * ETA)
    first = build_uniform_rule(box, 1.0e-4, reduction_steps=3)
    second = build_uniform_rule(box, 1.0e-4, reduction_steps=3)
    third = build_uniform_rule(box, 1.0e-4, reduction_steps=3, time_budget=1e-3)
    _check(first, box, 1.0e-4)
    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.times, third.times)
    np.testing.assert_array_equal(first.weights, third.weights)
    # fewer passes cannot reach fewer nodes than more passes
    more = build_uniform_rule(box, 1.0e-4, reduction_steps=12)
    assert more.node_count <= first.node_count

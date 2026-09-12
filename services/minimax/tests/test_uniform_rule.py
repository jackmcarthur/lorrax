"""Measure-independent box rules: sup error on the box and node counts."""
import numpy as np
import pytest

from minimax.fixed_n_start import predict_nodes, start_param
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
    small |Re d|.  No code shared with the builder's clouds.

    12 points per half wave is what makes this an audit -- the builder fits at
    2 and accepts at 6 -- and it is the linspace that carries it.  The
    geometric limbs only have to resolve the log measure toward the near
    corner; at 20000/2000 they were most of this suite's wall for no extra
    discrimination (the reported sup is unchanged to four digits at 1500)."""
    re_lo, re_hi, im_lo, im_hi = box
    h = min(np.pi / (12.0 * np.max(np.abs(times))), im_lo / 12.0)
    x = np.linspace(re_lo, re_hi, int((re_hi - re_lo) / h) + 2)
    if re_lo > 0.0:
        x = np.union1d(x, np.geomspace(re_lo, re_hi, 1500))
    if re_hi < 0.0:
        x = np.union1d(x, -np.geomspace(-re_hi, -re_lo, 1500))
    y = np.union1d(np.linspace(im_lo, im_hi, int((im_hi - im_lo) / h) + 2),
                   np.geomspace(im_lo, max(im_hi, im_lo * 1.0001), 400))
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


def test_wide_crossing_box_does_not_ship_the_rank_on_a_short_budget():
    """The reduction removes nodes one at a time, so before the fixed-N path a
    wide box spent its whole budget and shipped close to the interpolatory
    rank.  Placing the predicted count instead makes a short budget give a
    reduced rule; the certificate is unchanged (runs/DEV/327, lane struct)."""
    box = (-60.0 * ETA, 60.0 * ETA, ETA, 10.0 * ETA)
    rule = build_uniform_rule(box, 1.0e-4, time_budget=60.0)
    _check(rule, box, 1.0e-4)
    assert rule.node_count <= 0.75 * rule.rank, (rule.node_count, rule.rank)


def test_predicted_placement_stays_on_the_ray_and_inside_the_caps():
    """start_param's contract: exactly n ordered nodes, Re s strictly inside
    (0, horizon) -- s = 0 is a zero time node the executor refuses -- and Im s
    within the off-ray caps it was given."""
    box = (-40.0 * ETA, 20.0 * ETA, ETA, 5.0 * ETA)
    horizon, cap = 3.0, 0.05
    s = start_param(box, 1.0e-4, 0.0, horizon, -cap, cap, 24)
    assert s.size == 24
    assert np.all(s.real > 0.0) and np.all(s.real < horizon)
    assert np.all(np.abs(s.imag) <= cap)
    assert np.all(np.diff(s.real) > 0.0)
    assert predict_nodes(box, 1.0e-4) >= 2


def test_invalid_box_refuses():
    with pytest.raises(ValueError):
        build_uniform_rule((0.0, 1.0, 0.0, 1.0), 1.0e-4)
    with pytest.raises(ValueError):
        build_uniform_rule((1.0, 0.0, 0.1, 1.0), 1.0e-4)


def test_physical_negative_tail_holds_independent_certificate():
    """Run309 tail: the former check returned 8.0851e-5 at eps=7.5e-5."""
    box = (-11.604310294360761, -0.289400912133283,
           0.018374661087827496, 0.018374661087827496)
    eps = 7.5e-5
    rule = build_uniform_rule(box, eps, time_budget=30.0, backend="numpy")
    cloud = box_samples(*box, per_unit=10.0, n_im=48)
    error, amplification = rule_sup_error(
        rule.times, rule.weights, cloud, np.abs(cloud))
    assert error <= eps
    assert amplification <= 1.0e4


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


@pytest.mark.slow
def test_random_boxes_never_refuse_and_hold_on_a_finer_cloud():
    """Property test: every finite box yields an accepted rule, and the sup
    bound holds on a cloud finer than the one the rule was fitted on.
    Crossing, sign-definite (R up to 1e4) and nearly sign-definite boxes."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        kind = rng.choice(["crossing", "sd+", "sd-", "near+", "near-"])
        im_hi = ETA * 10 ** rng.uniform(0, 1.4)
        if kind == "crossing":
            lo, hi = -ETA * rng.uniform(4, 16), ETA * rng.uniform(4, 12)
        elif kind == "sd+":
            lo = ETA * 10 ** rng.uniform(-0.5, 1.0); hi = lo * 10 ** rng.uniform(0.5, 2.5)
        elif kind == "sd-":
            hi = -ETA * 10 ** rng.uniform(-0.5, 1.0); lo = hi * 10 ** rng.uniform(0.5, 2.5)
        elif kind == "near+":
            lo = -ETA * rng.uniform(0.1, 3); hi = ETA * rng.uniform(10, 45)
        else:
            hi = ETA * rng.uniform(0.1, 3); lo = -ETA * rng.uniform(10, 45)
        eps = float(rng.choice([1e-3, 1e-4]))
        box = (lo, hi, ETA, im_hi)
        rule = build_uniform_rule(box, eps, time_budget=2.0)
        assert rule.relative == kind.startswith("sd")
        _check(rule, box, eps)


def test_thin_boxes_certify_on_the_dense_boundary():
    """Real-pole windows have Im d in [eta, 1.01 eta].  On real time the rule's
    error oscillates at its node horizon at every Re d, so a crossing rule must
    hold far from Re d = 0 too (a geometric far field certified 247x-eps rules);
    a thin tail must hold at its near corner (1.8x eps on main, 2026-09-11)."""
    for box in ((-14.0 * ETA, 22.0 * ETA, ETA, 1.01 * ETA),     # Na B06-like crossing
                (1.05 * ETA, 240.0 * ETA, ETA, 1.01 * ETA)):    # Na tail-like
        _check(build_uniform_rule(box, 1.0e-4, time_budget=10.0), box, 1.0e-4)


def test_far_sign_definite_box_samples_stay_inside_the_box():
    """The geometric far field used to start at 30 im_lo even when the box
    starts beyond it, so the angle scan (and, before the boundary cloud, the
    fit and the certificate) saw points outside the box."""
    lo, hi = 77.0 * ETA, 243.0 * ETA
    d = box_samples(lo, hi, ETA, 1.01 * ETA)
    assert d.real.min() >= lo and d.real.max() <= hi


@pytest.mark.slow
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
    box = (2.0 * ETA, 60.0 * ETA, ETA, 10.0 * ETA)
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


@pytest.mark.parametrize("seconds,steps", [(0., 10), (float("nan"), 0),
                                         (1., -1), (1., 1.5), (1., True)])
def test_invalid_budget_refuses_before_build(seconds, steps):
    from minimax import uniform_rule_budget
    with pytest.raises(ValueError):
        uniform_rule_budget(seconds, steps)


def test_backend_receipt_uses_the_fitter_policy_resolver(monkeypatch):
    from minimax import uniform_rule_backend_policy
    monkeypatch.setenv("LORRAX_UNIFORM_RULE_BACKEND", " AUTO ")
    assert uniform_rule_backend_policy() == "auto"
    assert uniform_rule_backend_policy("numpy") == "numpy"
    monkeypatch.setenv("LORRAX_UNIFORM_RULE_BACKEND", "invalid")
    with pytest.raises(ValueError, match="must be numpy, jax or auto"):
        uniform_rule_backend_policy()

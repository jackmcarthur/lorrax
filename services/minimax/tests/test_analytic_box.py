"""Derived Sigma box rules: certificates held on an independent boundary."""
import numpy as np
import pytest

from minimax.analytic_box import analytic_box_rule, crossing_nodes, sector_degree
from minimax.uniform_rule import rule_roundoff_amplification, rule_sup_error

ETA = 0.02


def _dense_boundary(box, times):
    """An independent cloud: the four edges, uniform at 12 points per half wave
    of the largest |t| (and of 1/d at the peak), plus geometric points toward
    small |Re d|.  No code shared with the builder's clouds (the builder fits
    at 2 points per half wave and certifies at 6 with a golden refinement)."""
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
    """The certificate is honest in both directions on the dense boundary: the
    sampled sup (a lower bound of the true sup) meets eps, the rule's refined
    sup_error is not below it, and the term mass stays under the executor's
    noise cap 5e-6/6e-8 in the box's currency."""
    d = _dense_boundary(box, rule.times)
    rho = np.abs(d) if rule.relative else np.full(d.size, box[2])
    sup, _kappa = rule_sup_error(rule.times, rule.weights, d, rho)
    assert sup <= eps, (sup, eps)
    assert rule.sup_error >= sup * (1.0 - 1.0e-3), (rule.sup_error, sup)
    assert rule_roundoff_amplification(rule.times, rule.weights, d, rho) <= 83.4
    assert np.all(np.isfinite(rule.times)) and np.all(np.isfinite(rule.weights))
    return sup


def test_sign_definite_box_is_laplace_cheap():
    # 1/d on Re d in [2 eta, 400 eta] with Im d up to 30 eta, relative error:
    # the sector rule rotates toward imaginary time.
    box = (2.0 * ETA, 400.0 * ETA, ETA, 30.0 * ETA)
    rule = analytic_box_rule(box, 1.0e-4)
    _check(rule, box, 1.0e-4)
    assert rule.relative and rule.theta_deg < -30.0
    assert rule.node_count <= 24


def test_negative_sign_definite_box_rotates_the_other_way():
    box = (-100.0 * ETA, -4.0 * ETA, ETA, 20.0 * ETA)
    rule = analytic_box_rule(box, 1.0e-4)
    _check(rule, box, 1.0e-4)
    assert rule.relative and rule.theta_deg > 30.0
    assert rule.node_count <= 20


def test_sign_definite_box_touching_zero_certifies():
    """The Fe 4^3 SC padded cond:pole_tail window, [-2611, -0.11] eta at 3e-5:
    the csc/strip rule ended at 1203 eps after six rungs and the planner
    refused the deck (QUADWIRE, 2026-09-27).  The local law and geometric
    edge sampling make it an ordinary sector rule."""
    box = (-2611.26 * ETA, -0.1108 * ETA, ETA, ETA)
    rule = analytic_box_rule(box, 3.0e-5)
    _check(rule, box, 3.0e-5)
    assert rule.node_count <= 30


def test_symmetric_crossing_box_bends_too():
    """M = m: the straight line left the far image to the growth-capped set
    and ended at 1.49 eps on this tall box; the bent contour certifies."""
    box = (-20.0 * ETA, 20.0 * ETA, ETA, 10.0 * ETA)
    s, info = crossing_nodes(box, 1.0e-4)
    assert info["tau_c"] > 0.0
    rule = analytic_box_rule(box, 1.0e-4)
    _check(rule, box, 1.0e-4)
    assert abs(rule.theta_deg) < 1.0 and rule.node_count <= 80


@pytest.mark.parametrize("box_eta,eps,most", [
    ((-41.0, 83.3), 3.0e-5, 150),       # Fe val window: fitted 151, csc 200
    ((-83.1, 35.7), 3.0e-5, 140),       # Fe cond window: fitted 135, csc 200
])
def test_asymmetric_crossing_box_bends_and_beats_csc(box_eta, eps, most):
    box = (box_eta[0] * ETA, box_eta[1] * ETA, ETA, 1.01 * ETA)
    rule = analytic_box_rule(box, eps)
    _check(rule, box, eps)
    s, info = crossing_nodes(box, eps)
    assert info["tau_c"] > 0.0                      # the bent contour
    assert rule.node_count <= most


def test_narrow_crossing_box_takes_the_sector_rule():
    """A narrow side inside the peak (m = 0.26 eta): the sector law is below
    the bent contour's count, so the sector rule is built (fitted: 9)."""
    box = (-9.12 * ETA, 0.26 * ETA, ETA, 1.01 * ETA)
    assert sector_degree(box, 3.0e-5)[4] < crossing_nodes(box, 3.0e-5)[0].size
    rule = analytic_box_rule(box, 3.0e-5)
    _check(rule, box, 3.0e-5)
    assert not rule.relative and abs(rule.theta_deg) > 1.0
    assert rule.node_count <= 12


def test_the_other_family_is_the_fallback():
    """[-3.72, 65] eta at 5e-4: the bent contour is the cheaper law but its
    ladder ends uncertified (m < 4 eta); the sector rule certifies."""
    box = (-3.72 * ETA, 65.0 * ETA, ETA, 1.01 * ETA)
    rule = analytic_box_rule(box, 5.0e-4)
    _check(rule, box, 5.0e-4)
    assert abs(rule.theta_deg) > 1.0


def test_tall_narrow_crossing_box_is_returned_uncertified():
    """KNOWN LIMIT (KNOWN_LORRAX_ISSUES, QAUDIT claim 2882): a crossing box with
    a narrow side of 4-8 eta and a height of 10 eta or more certifies in
    neither family; main's fitted builder certified such boxes with 21-52
    nodes.  The builder returns its last rung uncertified, and the planner
    refuses the window by name (tests/test_sigma_box_plan.py).  No deck box
    is of this shape today; damped poles in a small excursion window would be."""
    box = (-60.0 * ETA, 6.0 * ETA, ETA, 20.0 * ETA)
    rule = analytic_box_rule(box, 1.0e-4)
    assert rule.sup_error > 1.0e-4
    assert np.all(np.isfinite(rule.times)) and np.all(np.isfinite(rule.weights))


def test_thin_boxes_certify_on_the_dense_boundary():
    for box in ((-14.0 * ETA, 22.0 * ETA, ETA, 1.01 * ETA),     # Na B06-like crossing
                (1.05 * ETA, 240.0 * ETA, ETA, 1.01 * ETA)):    # Na tail-like
        _check(analytic_box_rule(box, 1.0e-4), box, 1.0e-4)


def test_invalid_box_and_dead_arguments_refuse():
    with pytest.raises(ValueError):
        analytic_box_rule((0.0, 1.0, 0.0, 1.0), 1.0e-4)
    with pytest.raises(ValueError):
        analytic_box_rule((1.0, 0.0, 0.1, 1.0), 1.0e-4)
    box = (-8.0 * ETA, 8.0 * ETA, ETA, 4.0 * ETA)
    for dead in ("attempts", "kappa_cap", "time_budget"):
        with pytest.raises(TypeError):
            analytic_box_rule(box, 1.0e-4, **{dead: 10})


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


def test_the_rule_does_not_depend_on_threads():
    """No clock, no search, and the BLAS pin: two builds agree bit for bit
    whatever the caller's thread setting."""
    from minimax import uniform_rule
    controls = uniform_rule._openblas_controls()
    saved = [get() for get, _put in controls]
    for box in ((2.0 * ETA, 60.0 * ETA, ETA, 10.0 * ETA),
                (-14.0 * ETA, 30.0 * ETA, ETA, 4.0 * ETA)):
        rules = []
        try:
            for threads in (2, 4):
                for _get, put in controls:
                    put(threads)
                rules.append(analytic_box_rule(box, 1.0e-4))
        finally:
            for (_get, put), count in zip(controls, saved):
                put(count)
        np.testing.assert_array_equal(rules[0].times, rules[1].times)
        np.testing.assert_array_equal(rules[0].weights, rules[1].weights)


@pytest.mark.slow
def test_random_boxes_certify_on_a_finer_cloud():
    """Crossing, sign-definite (R up to 1e4) and nearly sign-definite boxes
    all return a certified rule that holds on the dense boundary."""
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
        rule = analytic_box_rule(box, eps)
        assert rule.relative == kind.startswith("sd")
        _check(rule, box, eps)

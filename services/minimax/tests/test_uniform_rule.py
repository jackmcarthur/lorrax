"""Measure-independent box rules: sup error on the box and node counts."""
import numpy as np
import pytest

from minimax.uniform_rule import box_samples, build_uniform_rule, rule_sup_error

ETA = 0.02


def _check(rule, box, eps):
    d = box_samples(*box, per_unit=7.0, n_im=8)          # finer than the fit cloud
    sup, kappa = rule_sup_error(rule.times, rule.weights, d)
    assert sup <= 3.0 * eps, (sup, eps)                  # fit cloud vs check cloud
    assert kappa <= 1.0e4
    assert np.all(np.isfinite(rule.times)) and np.all(rule.times != 0.0)
    return sup


def test_sign_definite_box_is_laplace_cheap():
    # 1/d on Re d in [2 eta, 400 eta]: Braess-Hackbusch regime, ~10 nodes.
    box = (2.0 * ETA, 400.0 * ETA, ETA, 30.0 * ETA)
    rule = build_uniform_rule(box, 1.0e-4, time_budget=30.0)
    _check(rule, box, 1.0e-4)
    assert rule.theta_deg < -40.0                        # rotated toward imaginary time
    assert rule.node_count <= 16


def test_negative_sign_definite_box_rotates_the_other_way():
    box = (-100.0 * ETA, -4.0 * ETA, ETA, 20.0 * ETA)
    rule = build_uniform_rule(box, 1.0e-4, time_budget=30.0)
    _check(rule, box, 1.0e-4)
    assert rule.theta_deg > 40.0
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
        d = box_samples(*box, per_unit=10.0, n_im=48)
        sup, kappa = rule_sup_error(rule.times, rule.weights, d)
        assert sup <= 1.5 * eps, (kind, box, sup, eps)
        assert kappa <= 1.0e4 and np.all(rule.times != 0.0)

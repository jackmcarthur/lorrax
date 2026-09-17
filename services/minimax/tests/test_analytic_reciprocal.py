"""Cross-check the three reciprocal representations against their true kernels."""
import numpy as np

import minimax


def test_positive_laplace_shift_and_legacy_form_agree():
    rule = minimax.positive_reciprocal(50, 1e-4)
    x = np.geomspace(1, 50, 3001)
    shifted = rule.evaluate(x)
    legacy = np.exp(-x[:, None] * rule.times) @ (rule.strengths * np.exp(rule.times))
    assert np.all(rule.times > 0) and np.all(rule.strengths > 0)
    assert np.max(np.abs(shifted - 1 / x)) < 1e-4
    assert np.max(np.abs(shifted - legacy)) < 1e-13


def test_odd_sine_rule_meets_its_bound_and_symmetry():
    rule = minimax.odd_reciprocal(120, 1e-4)
    x = np.linspace(1, 120, 6001)
    error = np.max(np.abs(rule.evaluate(x) - 1 / x))
    assert error <= rule.bound <= 1e-4
    assert np.max(np.abs(rule.evaluate(-x) + rule.evaluate(x))) < 1e-12


def test_damped_line_rule_meets_continuum_bound_on_both_sides():
    for height in (0.5, 2.0):
        rule = minimax.damped_line_reciprocal(50 / height, height * 1e-4).rescaled(height)
        x = np.linspace(-50, 50, 4001)
        error = np.max(np.abs(rule.evaluate(x) - 1 / (x + 1j * height)))
        assert error <= rule.bound <= 1e-4
        assert np.all(rule.times > 0)


def test_damped_line_normalized_rule_rescales_exactly():
    normalized = minimax.damped_line_reciprocal(50, 1e-4)
    x = np.linspace(-50, 50, 301)
    for height in (0.5, 2.0):
        physical = normalized.rescaled(height)
        assert physical.node_count == normalized.node_count
        assert np.array_equal(physical.times, normalized.times / height)
        assert np.array_equal(physical.weights, normalized.weights / height)
        assert np.allclose(physical.evaluate(height * x),
                           normalized.evaluate(x) / height, rtol=1e-13, atol=1e-13)

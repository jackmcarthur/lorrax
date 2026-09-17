"""Reciprocal quadrature constructors, with one normalized API.

All tolerances are *absolute* on dimensionless domains.  Constructors have
different guarantees: the sine and line rules have exact-arithmetic continuum
bounds; the positive Laplace rule has a high-precision numerical extremum
audit, not an interval-arithmetic certificate. Sigma PPM uses the damped-line
constructor for real-pole crossing windows.
"""
from __future__ import annotations


def positive_reciprocal(R: float, tolerance: float, *, max_nodes: int = 64,
                        corrections: int = 2, digits: int | None = None):
    """Smallest audited positive exponential rule for ``1/x``, ``1<=x<=R``.

    Returns a ``TargetedLaplaceRule`` with shifted strengths for stable runtime
    evaluation.  ``rule.times, rule.strengths*exp(rule.times)`` is the legacy
    unshifted ``exp(-x*t)`` convention.
    """
    from minimax.analytic_laplace import make_rule
    if not (0 < tolerance < 1) or max_nodes < 1:
        raise ValueError("Need 0<tolerance<1 and max_nodes>=1")
    for n in range(1, max_nodes + 1):
        rule = make_rule(R, n, corrections=corrections, digits=digits)
        if rule.history[-1]["maximum_error"] <= tolerance:
            return rule
    raise ValueError(f"No positive reciprocal rule met {tolerance:g} in {max_nodes} nodes")


def odd_reciprocal(A: float, tolerance: float, *, eta: float = 0.8,
                   precision: int = 0):
    """Analytically bounded sine rule on ``[-A,-1] union [1,A]``."""
    from minimax.analytic_sine import make_rule
    return make_rule(A, tolerance, eta=eta, precision=precision)


def damped_line_reciprocal(bandwidth_over_broadening: float, tolerance: float):
    """Rule for ``1/(u+i)`` on ``|u|<=bandwidth_over_broadening``.

    The error request is absolute in the normalized variable ``u=x/height``.
    For physical ``1/(x+i*height)`` with absolute error ``eps``, pass
    ``(span/height, height*eps)`` and use ``rule.rescaled(height)``.
    """
    from minimax.analytic_line import make_rule
    return make_rule(bandwidth_over_broadening, 1.0, tolerance)


def analytic_line_box_rule(box, eps):
    """Executor-form rule for a crossing denominator on one fixed-height line.

    The normalized analytic rule includes ``exp(-height*t)`` in its weights;
    Sigma's executor puts that damping in ``exp(i*t*d)`` instead.  Undoing it
    here yields the same quadrature without counting it twice.  The rule is
    deliberately limited to a flat imaginary edge and a real interval that
    crosses zero; rectangles and relative-error tails have other contracts.
    """
    from time import perf_counter
    import numpy as np
    from minimax.uniform_rule import UniformRule, rule_sup_error

    re_lo, re_hi, im_lo, im_hi = map(float, box)
    if not (np.isfinite([re_lo, re_hi, im_lo, im_hi]).all()
            and re_lo <= 0 <= re_hi and 0 < im_lo == im_hi
            and np.isfinite(eps) and 0 < eps < 1):
        raise ValueError("analytic line box needs a finite crossing interval, "
                         "one positive height, and 0 < eps < 1")
    started = perf_counter()
    height = im_lo
    span = max(abs(re_lo), abs(re_hi))
    # Leave room for float64 evaluation of the executor convention.
    physical = damped_line_reciprocal(span / height, 0.8 * eps).rescaled(height)
    times = np.asarray(physical.times, np.float64)
    weights = np.asarray(physical.weights * np.exp(height * times), np.complex128)
    sample = np.linspace(re_lo, re_hi, 513) + 1j * height
    sampled_error, kappa = rule_sup_error(times, weights, sample,
                                          np.full(sample.size, height))
    sup_error = max(height * physical.bound, sampled_error)
    if not np.isfinite(sup_error) or sup_error > eps:
        raise RuntimeError("analytic line rule missed the requested executor error")
    return UniformRule(times, weights, tuple(map(float, box)), float(eps),
                       False, 0.0, 0, float(sup_error), float(kappa),
                       perf_counter() - started)

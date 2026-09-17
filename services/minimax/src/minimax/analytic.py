"""Experimental reciprocal quadrature constructors, with one normalized API.

All tolerances are *absolute* on dimensionless domains.  Constructors have
different guarantees: the sine and line rules have exact-arithmetic continuum
bounds; the positive Laplace rule has a high-precision numerical extremum
audit, not an interval-arithmetic certificate.  No production dispatch uses
these constructors until the owner chooses a replacement route.
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

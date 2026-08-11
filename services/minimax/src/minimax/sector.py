"""Certified sinc quadrature for a reciprocal over a complex sector.

For denominators in the fourth quadrant, rotate the Laplace contour by
``theta = pi/4`` and write

    1/d = exp(i theta) int_0^inf exp[-d exp(i theta) s] ds.

After ``s = exp(y)`` the trapezoidal rule converges exponentially and its
rank grows with the logarithm of the radial range.  The returned error is an
analytic bound: the infinite-trapezoid error comes from the standard strip
bound, and the two finite tails are bounded geometrically.  No fitted table,
optimizer, or sampled error estimate enters the rule.
"""

from __future__ import annotations

import numpy as np


def reciprocal_sector_rule(radial_min, radial_max, *, rel_tol=1.0e-8,
                           max_nodes=4096):
    """Approximate ``1/d`` uniformly on a fourth-quadrant radial sector.

    Parameters
    ----------
    radial_min, radial_max
        Positive bounds on ``|d|``.  The rule covers every
        ``arg(d) in [-pi/2, 0]`` between them.
    rel_tol
        Uniform relative-error target.
    max_nodes
        Refusal ceiling on the sinc rule.

    Returns
    -------
    dict
        ``s`` and ``weights`` satisfy
        ``1/d ~= sum(weights * exp(-d*contour*s))``.  ``error_bound`` is
        the sum of the strip and two truncation bounds; ``kappa0`` is the
        absolute-weight amplification at the least-damped sector edge.
    """
    r_min = float(radial_min)
    r_max = float(radial_max)
    tol = float(rel_tol)
    if not (0.0 < r_min <= r_max) or not np.isfinite(r_max):
        raise ValueError(
            "reciprocal_sector_rule: radial bounds must be finite and "
            f"ordered, 0 < radial_min <= radial_max; got [{radial_min!r}, "
            f"{radial_max!r}].")
    if not (0.0 < tol < 1.0):
        raise ValueError(
            "reciprocal_sector_rule: rel_tol must lie in (0, 1); got "
            f"{rel_tol!r}.")
    if int(max_nodes) < 1:
        raise ValueError(
            f"reciprocal_sector_rule: max_nodes={max_nodes!r} is not positive.")

    theta = np.pi / 4.0
    contour = np.exp(1j * theta)
    ratio = r_max / r_min
    decay = np.cos(theta)

    # If |Im y| < strip, both displaced contours still decay.  At the
    # worst sector corner their real exponent is |d| sin(margin), giving
    # the relative infinite-trapezoid bound
    #
    #   [2/sin(margin)] / [exp(2*pi*strip/h) - 1].
    #
    # Search only the proof parameter; every candidate below has the same
    # bound, and the smallest integer node count wins deterministically.
    best = None
    for margin in np.linspace(0.02, 0.45, 173):
        strip = theta - float(margin)
        strip_constant = 2.0 / np.sin(margin)
        step = (2.0 * np.pi * strip
                / np.log1p(3.0 * strip_constant / tol))

        # Omitted negative-k terms: |exp(-z exp(kh))| <= 1.
        left_scale = tol * (1.0 - np.exp(-step)) / (3.0 * step * ratio)
        k_min = int(np.floor(1.0 + np.log(left_scale) / step))

        # Omitted positive-k terms are below the integral tail once the
        # transformed integrand is decreasing.  The least decay anywhere
        # in the rotated sector is radial_min*cos(theta).
        s_tail = np.log(3.0 * ratio / (tol * decay)) / decay
        k_max = int(np.ceil(np.log(s_tail) / step))
        n_nodes = k_max - k_min + 1
        candidate = (n_nodes, -step, k_min, k_max, float(margin),
                     strip_constant)
        if best is None or candidate < best:
            best = candidate

    n_nodes, neg_step, k_min, k_max, margin, strip_constant = best
    step = -neg_step
    strip = theta - margin
    k = np.arange(k_min, k_max + 1, dtype=np.float64)
    scaled_s = np.exp(step * k)
    s = scaled_s / r_min
    weights = contour * step * scaled_s / r_min

    strip_bound = strip_constant / np.expm1(2.0 * np.pi * strip / step)
    left_bound = (step * np.exp(step * (k_min - 1.0))
                  / (1.0 - np.exp(-step)) * ratio)
    right_bound = (ratio / decay
                   * np.exp(-decay * np.exp(step * k_max)))
    error_bound = float(strip_bound + left_bound + right_bound)
    kappa0 = float(np.sum(
        np.abs(weights) * np.exp(-r_min * decay * s)) * r_min)

    if n_nodes > int(max_nodes):
        raise ValueError(
            "reciprocal_sector_rule: the certified sinc rule for radial "
            f"ratio {ratio:.6g} at rel_tol={tol:.1e} needs {n_nodes} nodes, "
            f"above max_nodes={int(max_nodes)}.  Split the radial interval "
            "or raise the ceiling explicitly; width panes do not change this "
            "sector bound.")
    if error_bound > tol * (1.0 + 1.0e-12):  # protects the bound arithmetic
        raise ArithmeticError(
            "reciprocal_sector_rule: constructed bound "
            f"{error_bound:.6e} exceeds rel_tol={tol:.6e}.")

    return {
        "s": s.astype(np.float64),
        "weights": weights.astype(np.complex128),
        "contour": complex(contour),
        "theta": float(theta),
        "step": float(step),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "n_nodes": int(n_nodes),
        "radial_min": r_min,
        "radial_max": r_max,
        "radial_ratio": float(ratio),
        "error_bound": error_bound,
        "strip_bound": float(strip_bound),
        "left_bound": float(left_bound),
        "right_bound": float(right_bound),
        "kappa0": kappa0,
        "provenance": (
            "analytic sinc sector rule; theta=pi/4; "
            f"h={step:.8g}; k=[{k_min},{k_max}]; "
            f"bound={error_bound:.3e}"),
    }


__all__ = ["reciprocal_sector_rule"]

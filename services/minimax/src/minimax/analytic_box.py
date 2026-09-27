"""Closed-form exponential-sum rules for ``1/d`` on a Sigma denominator box.

Same contract as :func:`minimax.uniform_rule.build_uniform_rule`: the box
``(re_lo, re_hi, im_lo, im_hi)`` with ``im_lo > 0``, ``eps`` in the box's
currency (peak-relative ``im_lo |Q - 1/d|`` on a crossing box, relative
``|d| |Q - 1/d|`` on a sign-definite one), and a :class:`UniformRule` in
the executor convention ``1/d ~= sum_k w_k exp(i t_k d)``.  No node is
optimized: every count, node and weight below is a formula of the box and
``eps``.  The certificate is the builder's own boundary check
(:class:`minimax.uniform_rule._BoundaryCloud`); it evaluates the rule and
decides, and on a miss the count moves one rung up a fixed ladder
(``n -> ceil(1.1 n)``), never along a searched parameter.

**Crossing box: csc core plus the pole-corrected sine correction.**  In
units of ``eta = im_lo`` (``u = d/eta``) the midpoint rule of
``1/u = -i int_0^inf exp(itu) dt`` at spacing ``2h`` sums exactly:

    -2ih sum_{j>=0} exp(i(2j+1)hu) = h csc(hu),      Im u > 0.

Its only error is the alias lattice ``h csc(hu) - 1/u``, poles at
``u = +-pi/h, +-2pi/h, ...``.  With ``X = max(|re_lo|, re_hi)/eta`` and
``B = pi/h = X + delta``, ``delta = sqrt(3X)``, the box clears the nearest
alias by ``delta``.  The alias function is removed on the box by the
correction of ``minimax.analytic_sine`` (Stieltjes form
``1/v - csc v = sin v G(sin^2(v/2))``, ``G`` interpolated at the roots of
``V_d(y) - q V_{d-1}(y)``, ``q = tan^2(alpha/4)``, ``alpha = hX``; bound
``h q^(m+3/2)/(1-q)``), which is a sine series in ``hu`` of degree
``m + 1``: harmonics ``+-k h`` (the negative ones are the only negative
times; ``|t| <= (m+1)h/eta``).  Counts, with the error split
``0.7 eps`` core / ``0.1 eps`` correction:

    n = ceil([ln(2h/((1 - e^{-2h}) 0.7 eps)) - h] / 2h)        (core tail)
    m = ceil(ln(h/(0.1 eps (1 - q)))/ln(1/q) - 3/2)            (correction)

so ``n ~= B ln(1/eps)/(2 pi)``: the node density is set by ``max|x|``, not
by the full width.  Sources: reports/archive/quadrature_final (the
ChatGPT conditioning note, csc core 2480-2648; pole_corrected_sine_rule.py,
which is ``analytic_sine`` here).  Open: an asymmetric box pays ``X`` on
both sides; the fitted rules damp the wide side with off-axis times, and
the analytic analogue (a one-sided / bent-contour correction) is not
derived.

**Sign-definite box: elliptic time-Ritz, linear weights.**  The box is
rotated by its mid-angle ``phi`` into the right half plane; ``lo`` is the
rotated box's smallest real part, ``hi`` its largest modulus and ``gamma``
its largest angle off the axis.  Times are :func:`minimax.laplace_ritz.
place_times` on ``[lo, hi]`` (elliptic rates, Lyapunov, one symmetric
eigensolve), rotated back.  Weights are the linear least-squares solution
of ``d Q(d) = 1`` on prescribed Chebyshev points of the box edges, the same
projection the chi remote cells use (no closed-form weights exist for
these times).  Degree:

    n = ceil(ln(16 hi/lo) ln(1/eps) / (pi^2 (1 - 2 gamma/pi)))

``ln(16R) ln(1/eps)/pi^2`` is the elliptic (Zolotarev) count of the real
segment; ``1/(1 - 2 gamma/pi)`` is the strip-width argument: in
``w = ln p`` the box is a rectangle of half-height ``gamma`` in the strip
``|Im w| < pi/2`` where the Blaschke factors live, and removing ``gamma``
from the strip width rescales the segment length by that factor.  It is a
heuristic, not a bound; the ladder catches a miss.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .uniform_rule import UniformRule, _BoundaryCloud, _pinned_blas_threads

__all__ = ["analytic_box_rule", "crossing_counts", "sign_definite_degree"]

#: Error split of the crossing rule: core truncation and alias correction.
_CORE_SHARE, _CORRECTION_SHARE = 0.7, 0.1
#: Count ladder on a certificate miss, and its length.
_LADDER_GROWTH, _LADDER_RUNGS = 1.10, 6


def crossing_counts(box, eps, rung=0):
    """Closed-form ``(h, n, m, params)`` of the crossing rule, in units of ``im_lo``."""
    re_lo, re_hi, im_lo, im_hi = (float(v) for v in box)
    X = max(-re_lo, re_hi) / im_lo
    delta = max(2.0, math.sqrt(3.0 * max(X, 1.0)), 2.0 * im_hi / im_lo)
    B = X + delta
    h = math.pi / B
    alpha = h * X
    ell = -2.0 * math.log(math.tan(alpha / 4.0))
    q = math.exp(-ell)
    core_eps, corr_eps = _CORE_SHARE * eps, _CORRECTION_SHARE * eps
    n = max(1, math.ceil((math.log(2.0 * h / (-math.expm1(-2.0 * h)) / core_eps) - h)
                         / (2.0 * h)))
    m = max(0, math.ceil(math.log(h / (corr_eps * (-math.expm1(-ell)))) / ell - 1.5))
    growth = _LADDER_GROWTH ** rung
    n, m = math.ceil(n * growth), math.ceil(m * growth)
    return h, n, m, dict(X=X, B=B, alpha=alpha, q=q, V=math.sin(alpha / 2.0) ** 2)


def _crossing(box, eps, rung):
    from .analytic_sine import _correction_weights
    h, n, m, p = crossing_counts(box, eps, rung)
    corr, _v = _correction_weights(dict(m=m, h=h, V=p["V"], q=p["q"], alpha=p["alpha"]))
    coeff = {2 * j + 1: -2j * h for j in range(n)}
    for k, w in enumerate(corr, 1):          # sum_k w_k sin(k h u)
        coeff[k] = coeff.get(k, 0.0) + w / 2j
        coeff[-k] = coeff.get(-k, 0.0) - w / 2j
    k = np.array(sorted(coeff))
    scale = float(box[2])
    return (k * h / scale).astype(np.complex128), \
        np.array([coeff[v] for v in k], np.complex128) / scale, 0.0, n


def _corners(box):
    a, b, c, d = (float(v) for v in box)
    return np.array([a + 1j * c, b + 1j * c, b + 1j * d, a + 1j * d])


def sign_definite_degree(box, eps, rung=0):
    """``(degree, lo, hi, gamma, rotation)`` of the sign-definite rule."""
    corners = _corners(box)
    angle = np.angle(corners)
    phi = 0.5 * (angle.min() + angle.max())
    rot = np.exp(-1j * phi)
    z = corners * rot
    lo, hi = float(z.real.min()), float(np.abs(z).max())
    gamma = float(np.abs(np.angle(z)).max())
    n = math.ceil(math.log(16.0 * hi / lo) * math.log(1.0 / eps)
                  / (math.pi ** 2 * (1.0 - 2.0 * gamma / math.pi)))
    n = max(2, n)
    for _ in range(rung):
        n = max(n + 1, math.ceil(n * _LADDER_GROWTH))
    return n, lo, hi, gamma, rot


def _sign_definite(box, eps, rung):
    from scipy import linalg

    from .laplace_ritz import place_times
    n, lo, hi, _gamma, rot = sign_definite_degree(box, eps, rung)
    times = 1j * rot * place_times(lo, hi, 0.0, n)
    corners = _corners(box)
    count = max(80, 8 * n)
    u = 0.5 * (1.0 - np.cos(np.pi * np.arange(count) / (count - 1)))
    d = np.concatenate([p + (q - p) * u for p, q in zip(corners, np.roll(corners, -1))
                        if p != q])
    matrix = d[:, None] * np.exp(1j * d[:, None] * times[None, :])
    norm = np.linalg.norm(matrix, axis=0)
    w = linalg.lstsq(matrix / norm, np.ones(d.size), cond=1e-14,
                     lapack_driver="gelsd")[0] / norm
    theta = -float(np.angle(1j * rot))
    return times, w.astype(np.complex128), theta, n


def analytic_box_rule(box, eps, *, kappa_cap=1.0e4, relative=None, **_ignored):
    """Closed-form rule for ``1/d`` on ``box``; see the module docstring.

    Returns the first rung of the fixed count ladder whose boundary
    certificate passes (sup <= eps and cancellation ratio <= kappa_cap), or
    the last rung uncertified (the planner then refuses the window).
    Keyword arguments of the fitted builder that do not apply (``attempts``)
    are accepted and ignored: nothing here is split or searched.
    """
    re_lo, re_hi, im_lo, im_hi = map(float, box)
    if not (np.isfinite([re_lo, re_hi, im_lo, im_hi]).all() and re_lo <= re_hi
            and 0.0 < im_lo <= im_hi):
        raise ValueError(f"invalid support box {box!r}")
    if relative is None:
        relative = re_lo > 0.0 or re_hi < 0.0
    build = _sign_definite if relative else _crossing
    started = time.perf_counter()
    with _pinned_blas_threads():
        for rung in range(_LADDER_RUNGS):
            times, weights, theta, degree = build(box, eps, rung)
            horizon = max(math.log(10.0 / eps) / im_lo, float(np.abs(times).max()))
            check = _BoundaryCloud((re_lo, re_hi, im_lo, im_hi), 0.0, horizon, eps,
                                   p=6.0, p_target=8.0)
            sup, kappa = check.sup(times, weights, relative)
            if sup <= eps and kappa <= kappa_cap:
                break
    return UniformRule(
        times=times, weights=weights, box=(re_lo, re_hi, im_lo, im_hi),
        eps=float(eps), relative=bool(relative), theta_deg=float(np.rad2deg(theta)),
        rank=int(degree), sup_error=float(sup), kappa_max=float(kappa),
        seconds=time.perf_counter() - started)

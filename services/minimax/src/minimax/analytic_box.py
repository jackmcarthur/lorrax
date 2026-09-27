"""Derived exponential-sum rules for ``1/d`` on a Sigma denominator box.

The box ``(re_lo, re_hi, im_lo, im_hi)`` has ``im_lo > 0``; ``eps`` is in the
box's currency (peak-relative ``im_lo |Q - 1/d|`` on a crossing box,
relative ``|d| |Q - 1/d|`` on a sign-definite one); the result is a
:class:`UniformRule` in the executor convention
``1/d ~= sum_k w_k exp(i t_k d)``.  No node is optimized: every node is a
formula of the box and ``eps``, and the weights are one linear least-squares
solve.  The builder's boundary certificate decides; a miss moves one rung up
a fixed ladder, never along a searched parameter.  The derivation (the
cot + two-Laplace identity, the bent contour, the count laws, the local
extremal-length law) is ``docs/theory/minimax-quadrature.md`` section 7.

**Crossing box: the bent contour** (:func:`crossing_nodes`).  Units
``eta = im_lo``; the box is oriented so its wide side ``M`` is at ``x < 0``
and its narrow side ``m`` at ``x > 0``.  With ``L = ln(1/eps)``,
``Lam = ln(4/eps)`` and ``c = 4`` the nodes are ``s(sigma) = sigma -
i tau(sigma)``, ``tau = min(c/m, (Lam - sigma)/m)``, at density
``gamma B(sigma)/2 pi`` (``B`` the largest live ``|x|``); a vertical leg
``0 -> -i c/m`` carries the wide side's Laplace part; and two Gauss sets on
the imaginary time axes carry the trapezoid's two images at ``+-B0`` (the
endpoint error of the trapezoid rule, exactly two Laplace integrals).  The
growth-side image is capped at ``|Im s| (b - a) <= 3``.  On a tall box
(damped poles, height ``H``) the bend is lowered to
``c = min(4, max(2, 8 m/(H - 1)))`` (:func:`corner_exponent`), the leg gets
uniform nodes for its top-edge phase, and a miss steps ``c, c/2, c/4``.

**Sector rule** (:func:`sector_degree`).  Every box with ``im_lo > 0`` lies
in an open sector of the upper half plane.  Times are the elliptic
time-Ritz times (:func:`minimax.laplace_ritz.place_times`) of the rotated
box's real extent, rotated to the sector's axis ``phi``; the degree is the
local extremal-length law

    n = (L/pi^2) [int (pi/2)/g(l) dl + ln 4 (pi/2)(1/g_near + 1/g_far)],

``g(l)`` the half-gap between the box's arc at radius ``e^l`` and the
sector edges ``phi +- pi/2``; ``phi`` minimizes the law on a fixed grid.  A
sign-definite box takes this rule; a crossing box takes the bent contour,
a narrow side under ``_NARROW_MIN`` = 4 eta built as 4 eta (on a crossing
box the sector rule's rotated times cancel at ``|w| ~ 1e10``, and the
executor's Sigma came out non-finite on core fixture A).

Acceptance of every rung: the refined boundary sup is at most ``eps`` and
the term mass ``rho sum_k |w_k exp(i t_k d)|`` (the executor's noise
amplification, in the certificate's own currency) is at most ``mass_cap``.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .uniform_rule import UniformRule, _BoundaryCloud, _cexp, _pinned_blas_threads

__all__ = ["analytic_box_rule", "corner_exponent", "crossing_nodes", "sector_degree"]

#: Corner amplification of the bent line at the narrow edge, ``e^c``: the
#: largest the executor noise gate admits with room (term mass <= 83 > e^4).
_BEND = 4.0
#: The line ends where every live member is below ``eps/4``: ``Lam = ln(4/eps)``.
_FLOOR = 4.0
#: Growth-side image cap ``|Im s| (b - a)`` (the same cap the fitted rules used).
_GROWTH_CAP = 3.0
#: Ridge of the weight solve, in units of ``eps``: a term-mass penalty.
_RIDGE = 0.05
#: The one calibrated constant: the line margin's floor 1 + 0.01 max(2, L - cM/m).
_MARGIN_SLOPE = 0.01
#: Ladders: the crossing rule raises gamma by 1.1 and adds one node per
#: Laplace set per rung; the sector rule raises n -> max(n + 1, ceil(1.1 n)).
_CROSSING_RUNGS, _SECTOR_RUNGS, _LADDER_GROWTH = 6, 6, 1.10
#: The narrowest side the bent contour is built for, in units of eta.
_NARROW_MIN = 4.0
#: The leg's largest phase at the top edge, tau_c (H - 1), in radians, and the
#: floor of the bend it sets (tall crossing boxes).
_LEG_PHASE, _BEND_MIN = 8.0, 2.0
#: Fit cloud density along Re d, in points per half wave of the largest |t|.
_FIT_POINTS = 2.0


# ------------------------------------------------------------ crossing: nodes
def corner_exponent(box):
    """The bend ``c`` of a crossing box: ``_BEND`` = 4 on a thin box, lowered on
    a tall one so the leg's phase ``tau_c (H - 1) = c (H - 1)/m`` at the top
    edge stays within ``_LEG_PHASE`` radians, floored at ``_BEND_MIN``."""
    re_lo, re_hi, im_lo, im_hi = (float(v) for v in box)
    m = max(min(-re_lo, re_hi) / im_lo, _NARROW_MIN)
    rise = im_hi / im_lo - 1.0
    return min(_BEND, max(_BEND_MIN, _LEG_PHASE * m / max(rise, 1e-12)))


def crossing_nodes(box, eps, rung=0, bend=None):
    """Derived bent-contour nodes ``s`` in units of ``1/im_lo``, and a receipt.

    See the module docstring; ``rung`` raises the line margin by ``1.1**rung``
    and adds ``rung`` nodes to the leg and to each image set; ``bend`` is the
    corner exponent c (default :func:`corner_exponent`).
    """
    c = corner_exponent(box) if bend is None else float(bend)
    re_lo, re_hi, im_lo, im_hi = (float(v) for v in box)
    a, b = re_lo / im_lo, re_hi / im_lo
    flip = b > -a
    M, m = (b, -a) if flip else (-a, b)
    # A narrow side inside the peak is built as _NARROW_MIN: 1/d varies on
    # the scale eta there, so the line needs no finer bend than c/_NARROW_MIN
    # (weights and certificate stay on the true box).
    m = max(m, _NARROW_MIN)
    M = max(M, m)
    L, Lam = math.log(1.0 / eps), math.log(_FLOOR / eps)
    # Always bent, a symmetric box too: the straight line (tau = 0) leaves the
    # far image to the growth-capped set, and on [-20, 20] x [1, 10] eta it
    # ended at 1.49 eps where the bent contour certifies with 73 nodes.
    tau_c = c / m
    floor = 1.0 + _MARGIN_SLOPE * max(2.0, L - c * M / m)
    gamma = min(max(math.sqrt(1.0 + 8.0 * (L + c) / (math.pi * M * Lam)), floor), 1.2)
    gamma *= _LADDER_GROWTH ** rung
    sigma = np.linspace(0.0, Lam, 20001)
    tau = np.minimum(tau_c, np.maximum(0.0, (Lam - sigma) / m))
    width = np.maximum(m * (1.0 + tau * m / L), np.minimum(M, (Lam - sigma) / tau_c))
    nu = np.concatenate([[0.0], np.cumsum(0.5 * (width[1:] + width[:-1]) * np.diff(sigma))])
    nu *= gamma / (2.0 * math.pi)
    n_line = int(math.ceil(nu[-1])) + 1
    at = np.interp(np.arange(n_line, dtype=float), nu, sigma)
    parts = [at - 1j * np.interp(at, sigma, tau)]
    # the leg, graded toward 0; its top node -i tau_c is the line's first node
    kv = int(math.ceil(math.log(max(M * tau_c, 2.0)) * L / math.pi ** 2)) + rung
    parts.append(-1j * tau_c * np.geomspace(0.05 / (M * tau_c), 1.0, kv + 1)[:-1])
    # on a tall box the leg's members e^{v d} turn their phase v*y through
    # tau_c (H - 1) radians: one uniform node per 2 pi / gamma of it
    k_osc = int(gamma * tau_c * (im_hi / im_lo - 1.0) / (2.0 * math.pi))
    if k_osc:
        parts.append(-1j * tau_c * (np.arange(1, k_osc + 1) - 0.5) / k_osc)
    # the trapezoid's images at +-B0: two Gauss sets on the imaginary axes
    B0 = gamma * max(width[0], min(M, Lam / tau_c))
    live = min(M, Lam / tau_c)
    amp = tau_c * m
    K = int(math.ceil(math.log(16.0 * (B0 + live) / max(B0 - m, 1e-9)) * (L + amp)
                      / math.pi ** 2)) + rung
    u = 0.5 * (np.polynomial.legendre.leggauss(K)[0] + 1.0)
    parts.append(-1j * u * ((L + amp) / max(1.0 - m / B0, 0.05)) / B0)
    residual = L - tau_c * M
    top = min(residual / max(1.0 - M / B0, 0.05) if residual > 0.0 else 0.0,
              _GROWTH_CAP * B0 / (M + m))
    if top > 0.0:
        parts.append(1j * u * top / B0)
    s = np.concatenate(parts)
    if flip:
        s = np.conj(s)
    return s, dict(M=M, m=m, c=c, gamma=gamma, tau_c=tau_c, B0=B0, n_line=n_line, flip=flip)


# ------------------------------------------------------------ sector: degree
def _arc_hull(box, r):
    """``(lo, hi)`` angles of the box's points at radius ``r`` (arrays): the
    ends of the circle's arcs inside the box lie on its edges."""
    a, b, c, d = box
    r2 = r[:, None] ** 2
    tol = 1.0e-9 * r[:, None] * (1.0 + r[:, None])
    pts = []
    for y in (c, d):                        # bottom and top edges
        x = np.sqrt(np.maximum(r2 - y * y, 0.0))
        for xx in (x, -x):
            ok = (r2 >= y * y - tol) & (xx >= a - tol) & (xx <= b + tol)
            pts.append((xx, np.full_like(xx, y), ok))
    for x in (a, b):                        # left and right edges
        y = np.sqrt(np.maximum(r2 - x * x, 0.0))
        ok = (r2 >= x * x - tol) & (y >= c - tol) & (y <= d + tol)
        pts.append((np.full_like(y, x), y, ok))
    ang = np.concatenate([np.where(ok, np.arctan2(y, x), np.nan) for x, y, ok in pts], axis=1)
    return np.nanmin(ang, axis=1), np.nanmax(ang, axis=1)


def _corners(box):
    a, b, c, d = (float(v) for v in box)
    return np.array([a + 1j * c, b + 1j * c, b + 1j * d, a + 1j * d])


def sector_degree(box, eps, rung=0):
    """``(degree, lo, hi, phi, law)`` of the sector rule on ``box``."""
    a, b, c, d = (float(v) for v in box)
    r_lo = math.hypot(min(max(0.0, a), b), c)          # nearest box point to 0
    r_hi = float(np.abs(_corners(box)).max())
    edges = np.linspace(math.log(r_lo), math.log(r_hi), 401)
    mid = 0.5 * (edges[1:] + edges[:-1])
    lo, hi = _arc_hull((a, b, c, d), np.exp(np.concatenate([[edges[0]], mid, [edges[-1]]])))
    ends, lo, hi = (lo[[0, -1]], hi[[0, -1]]), lo[1:-1], hi[1:-1]
    law = math.inf
    phi = 0.5 * (float(lo.min()) + float(hi.max()))
    for p in np.linspace(float(lo.min()), float(hi.max()), 200):
        gap = 0.5 * math.pi - np.maximum(hi - p, p - lo)
        end = 0.5 * math.pi - np.maximum(ends[1] - p, p - ends[0])
        if gap.min() <= 0.02 or end.min() <= 0.02:
            continue
        value = (float(np.sum(np.diff(edges) * 0.5 * math.pi / gap))
                 + math.log(4.0) * 0.5 * math.pi * float(np.sum(1.0 / end)))
        if value < law:
            law, phi = value, float(p)
    law *= math.log(1.0 / eps) / math.pi ** 2
    n = max(2, math.ceil(law)) if math.isfinite(law) else 2
    for _ in range(rung):
        n = max(n + 1, math.ceil(n * _LADDER_GROWTH))
    z = _corners(box) * np.exp(-1j * phi)
    return n, float(z.real.min()), float(np.abs(z).max()), phi, law


# ------------------------------------------------------------ weights
def _fit_cloud(box, times):
    """Least-squares rows: the four edges, the real ones at ``_FIT_POINTS``
    per half wave of the largest ``|t|``.  A thin box keeps its top edge: a
    term's modulus changes by ``exp(-Re t (im_hi - im_lo))`` across it, 0.35
    for a sector rule's far times on a 1.01-eta box."""
    a, b, c, d = (float(v) for v in box)
    h = math.pi / (_FIT_POINTS * float(np.abs(times).max()))
    x = np.linspace(a, b, int(math.ceil((b - a) / h)) + 1)
    rows = [x + 1j * c]
    if d > c:
        y = np.geomspace(c, d, 40)[1:-1]
        rows += [x + 1j * d, a + 1j * y, b + 1j * y]
    return np.concatenate(rows)


def _weights(box, times, eps, relative):
    """The weights: one linear least-squares solve on the box boundary.

    Crossing (peak-relative) boxes solve ``im_lo (Q - 1/d) = 0`` on the fit
    cloud with columns scaled to unit maximum (in log space, so an imaginary
    time cannot overflow) and a ridge ``0.05 eps``, which bounds each term's
    largest contribution in the error currency.  Sign-definite (relative)
    boxes solve ``d Q(d) = 1`` on the real edges at geometric spacing in
    ``|Re d|`` (every decade of ``|d|`` counts once, the relative currency's
    measure) and on Chebyshev points of the sides.
    """
    from scipy import linalg
    if relative:
        a, b, c, top = (float(v) for v in box)
        count = max(80, 8 * times.size)
        sign, near, far = (1.0, a, b) if a > 0.0 else (-1.0, -b, -a)
        x = sign * np.geomspace(near, far, count)
        d = [x + 1j * c] + ([x + 1j * top] if top > c else [])
        if top > c:
            u = 0.5 * (1.0 - np.cos(np.pi * np.arange(count) / (count - 1)))
            d += [a + 1j * (c + (top - c) * u), b + 1j * (c + (top - c) * u)]
        d = np.concatenate(d)
        matrix = d[:, None] * _cexp(1j * d[:, None] * times[None, :])
        norm = np.linalg.norm(matrix, axis=0)
        return linalg.lstsq(matrix / norm, np.ones(d.size), cond=1e-14,
                            lapack_driver="gelsd")[0] / norm
    d = _fit_cloud(box, times)
    rho = float(box[2])
    log_max = np.max(-(d[:, None] * times[None, :]).imag, axis=0)
    A = rho * _cexp(1j * d[:, None] * times[None, :] - log_max[None, :])
    # the ridge row prices a term in the error currency (rho times its
    # unit-maximum column), so the penalty is scale-free in eta
    A = np.vstack([A, _RIDGE * eps * rho * math.sqrt(d.size) * np.eye(times.size)])
    f = np.concatenate([rho / d, np.zeros(times.size)])
    q, r = linalg.qr(A, mode="economic")
    return linalg.solve_triangular(r, q.conj().T @ f) * np.exp(-log_max)


# ------------------------------------------------------------ the builder
def _crossing_rule(box, eps, rung, bend):
    s, _ = crossing_nodes(box, eps, rung, bend)
    times = s / float(box[2])
    return times, _weights(box, times, eps, False), 0.0, int(s.size)


def _sector_rule(box, eps, rung, relative):
    from .laplace_ritz import place_times
    n, lo, hi, phi, _law = sector_degree(box, eps, rung)
    rot = np.exp(-1j * phi)
    times = 1j * rot * place_times(lo, hi, 0.0, n)
    return times, _weights(box, times, eps, relative), -float(np.angle(1j * rot)), n


def analytic_box_rule(box, eps, *, mass_cap=83.0, relative=None):
    """Derived rule for ``1/d`` on ``box``; see the module docstring.

    Returns the first rung whose boundary certificate passes (sup <= eps and
    term mass <= ``mass_cap`` in the box's currency), or the last rung built,
    uncertified, which the planner refuses.
    """
    re_lo, re_hi, im_lo, im_hi = map(float, box)
    if not (np.isfinite([re_lo, re_hi, im_lo, im_hi]).all() and re_lo <= re_hi
            and 0.0 < im_lo <= im_hi):
        raise ValueError(f"invalid support box {box!r}")
    box = (re_lo, re_hi, im_lo, im_hi)
    if relative is None:
        relative = re_lo > 0.0 or re_hi < 0.0
    if relative:
        families = [(lambda r: _sector_rule(box, eps, r, True), _SECTOR_RUNGS)]
    else:
        # A crossing box takes the bent contour only. The sector rule
        # certifies narrow crossing boxes, but its rotated times cancel at
        # |w| ~ 1e10 there and the executor's Sigma came out non-finite
        # (core fixture A, [-2.71, 47.63] eta, 2026-09-27).
        # the bend's own fixed ladder: c, c/2, c/4, each with the margin ladder
        c = corner_exponent(box)
        families = [(lambda r, b=b: _crossing_rule(box, eps, r, b), _CROSSING_RUNGS)
                    for b in (c, c / 2.0, c / 4.0)]
    started = time.perf_counter()
    with _pinned_blas_threads():
        for build, rungs in families:
            for rung in range(rungs):
                times, weights, theta, degree = build(rung)
                horizon = max(math.log(10.0 / eps) / im_lo, float(np.abs(times).max()))
                # sampled along the rule's own ray: a sector rule's far times
                # oscillate at |Re t| well above ln(10/eps)/eta
                check = _BoundaryCloud(box, theta, horizon, eps, p=6.0, p_target=8.0)
                sup, kappa, mass = check.sup(times, weights, relative)
                if sup <= eps and mass <= mass_cap:
                    break
            else:
                continue
            break
    return UniformRule(
        times=times, weights=weights, box=box, eps=float(eps), relative=bool(relative),
        theta_deg=float(np.rad2deg(theta)), rank=int(degree), sup_error=float(sup),
        kappa_max=float(kappa), seconds=time.perf_counter() - started)

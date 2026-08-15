#!/usr/bin/env python3
"""Generate and certify the ``damped_line`` quadrature family.

WHAT THIS FAMILY IS.  The fourth cell of ``DESIGN_minimax.md`` R4's 2x2
sampling table -- ``z = omega + i*varpi`` with both parts nonzero, the
strip -- and the only cell with no shipped rule.  Its kernel is the one
the MPA theory plan fixes in section B::

    K_z(Delta) = -2 * integral_0^inf dt e^{i z t} sin(Delta t)
               = -2 * Delta / (Delta**2 - z**2)          (Im z > 0)

A RULE here is a set of time nodes ``t_l`` SHARED across every sample --
that is what line-batching means, the ``G_c(tau) G_v(tau)`` product is
built once per node -- plus per-sample complex weights::

    K_z(Delta) ~= sum_l w_l(z) sin(Delta t_l)

WHAT IS DIFFERENT FROM ``complex_laplace``, and it is three things.

1. **There is no envelope axis.**  In units of the near line's height
   the family is exactly scale free, ``K_z(Delta) = (1/varpi) *
   Khat(Delta/varpi, omega/varpi)``, so an entry is keyed by the
   dimensionless bandwidth ``A = 2*Delta_max/varpi_1`` and the error
   tier and nothing else.  The ``beta`` axis that forced
   ``complex_laplace`` into its own selector has no analogue.
2. **The domain is a BANDWIDTH, not an interval.**  ``Delta`` runs over
   ``[0, Delta_max]`` with no ``Delta_min`` and no gap assumption, which
   is what makes this the occupations-ready route: the gapless case
   that is an ``R -> infinity`` catastrophe on the interval families is
   not a special case here at all.
3. **One entry serves BOTH sampling lines and every n_p.**  The nodes
   are solved against the whole rectangle ``(z-line) x (Delta-band)`` on
   both the near line and the far line at once, so the shipped set is
   certified for every ``omega`` on either line rather than for the
   ``2*n_p`` points of one partition.  Growing ``n_p`` selects rows from
   a weight table that is already certified; it never needs a new solve
   and never interpolates.

WHY THE CAMPAIGN IS A TABULATION AND NOT A FORMULATION.
``GATE0_DAMPED_LINE_LP.md`` measured the sparse route at 1.6-1.9x the
composite and not the 3x the campaign had set as its bar, because the
node count on a linear transition band with oscillatory atoms is a
time-bandwidth product with no logarithm in it.  What this generator
ships is therefore not a cleverer rule.  It is a SCOPE: one global node
set per span and tier, certified once, serving both lines and every
``n_p``, which is worth 1.9-2.1x against building one composite rule per
line and costs nothing extra to use.

THE TWO ROUTES, and the rule that chooses between them.

``positive_composite``
    ``gw.mpa.evaluator.damped_line_rule``'s graded Gauss-Legendre
    panels, ported here so this tool stays free of jax and of ``src``
    (``tests/test_damped_line_tables.py`` asserts the port is
    bit-identical to the shipped function, so the duplication is a
    checked fact and not a hope).  Weights are ``w_l(z) = -2 h_l
    e^{i z t_l}`` in closed form at ANY ``z`` on either line, so a
    composite entry ignores the ``n_p`` and ``alpha`` axes entirely.
    ``kappa0 = 1.0000`` by positivity, independent of node count.

``btv_minimax``
    Bounded-total-variation minimax on the rectangle: a log-tau
    dictionary with a Nyquist spine, orthogonal matching pursuit for
    selection, a cardinality prune under the amplification cap, and
    per-sample weights.  The cap is imposed by a Tikhonov ridge
    bisected until the MEASURED ``sum_l |w_l|`` obeys it -- an L2 ball
    inscribed in the L1 ball, so it is conservative.

THE CAP IS LOAD-BEARING HERE IN A WAY IT WAS NOT FOR THE 1D FAMILY.
Gate zero ran this same selection uncapped and got rules at ``kappa0``
of 406, 620 and 1267 that meet their sup-norm tolerance and would
amplify any error in the per-node chi0 build by three orders -- the
census's cancelling-pair pathology, reproduced.  Every entry here
carries a ``kappa0`` re-measured from its own shipped weights, and the
harness carries one of those specimens as its red twin.

SHIPPING RULE AT A CELL.  Attempt the sparse route under ``kappa0 <=
2``.  Ship it if and only if it (i) certifies on the held-out
rectangle, (ii) measures ``kappa0 <= 2`` from its own weights, and
(iii) uses strictly fewer nodes than THE TWO COMPOSITE RULES IT
REPLACES.  Otherwise ship the composite.  Clause (iii) is what makes a
stalled prune harmless: the fallback is a certified rule of the same
family at the same cell.

Nothing here is imported by ``src/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
# The bundle rides with the minimax SERVICE, not with `common`.  The
# extraction moved it so the door reads its own package data; a generator
# still writing to the old root would produce tables nothing can find.
DEFAULT_OUTPUT_ROOT = (REPO_ROOT / "services" / "minimax" / "src"
                       / "minimax" / "minimax_assets")
FAMILY = "damped_line"
CATALOG_NAME = "catalog_damped_line.json"

# The ladder: C§7's deck stand-ins, build note VI's enlarged silicon deck
# (A = 99.1, served exactly by A = 100), and P§E's fit_line_global rungs.
SPANS = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0)
TIERS = (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)

#: Tiers that ship composite BY CONSTRUCTION, with no sparse attempt.
#: The sparse instrument's wall is already at 1e-10 -- the exact minimax
#: LP returns ``model_status = Unknown`` on rows carrying activities of
#: order 1e10 -- and the composite is closed form, self certifying and
#: costs seconds, so declaring 1e-12 composite-only is a cost decision of
#: no consequence that closes a documented gap.
COMPOSITE_ONLY_TIERS = (1.0e-12,)

#: The published pair, varpi_2/varpi_1 = 1 Ha / 0.1 Ha.  EXACT-match axis:
#: the far line's rows are what the node set was certified against.
LINE_RATIO = 10.0

#: The largest n_p whose partition rows are shipped.  P§B schedules Cu at
#: 12 with a scan to 15; 16 covers every schedule with a rung to spare.
N_P_MAX = 16
ALPHAS = (1, 2)

KAPPA_MAX = 2.0

# The moment identity's ceiling as a fraction of the trivial bound; the
# same gate the complex_laplace generator uses, for the same reason -- it
# refuses a near-constant offset, which no sup-norm test can see.
MOMENT_FRACTION = 0.25

_GL_CACHE: dict[int, tuple] = {}


def _leggauss(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


# ---------------------------------------------------------------------------
# The target, the domain, and the sample rows
# ---------------------------------------------------------------------------

KERNEL_FACTOR = -2.0


def damped_kernel(z, delta):
    """``K_z(Delta) = -2 Delta / (Delta**2 - z**2)``.

    Verbatim ``gw.mpa.evaluator.damped_kernel``.  Refuses a pair sitting
    on the kernel's own pole, which no sample with ``varpi > 0`` can.
    """

    zc = np.asarray(z, dtype=np.complex128)
    d = np.asarray(delta, dtype=np.float64)
    denom = d ** 2 - zc ** 2
    if np.any(denom == 0.0):
        raise ValueError(
            "GATE off_kernel_pole: a sample z coincides with +/- a "
            "transition energy Delta. FALSE case: no sample equals +/- "
            "any transition energy, which every varpi > 0 satisfies.")
    return KERNEL_FACTOR * d / denom


def delta_grid(a_dim, n):
    """The solver's Delta grid: uniform on ``[0, Delta_max]``, ENDPOINTS
    INCLUDED.

    The endpoints are not a detail.  A fit constrained only at interior
    points is unconstrained at the band edge, and the band edge is where
    a sine sum's residual is largest: gate zero measured an early pass
    gridded on cell midpoints certifying 1.9x over budget for that reason
    and for no other.
    """

    return np.linspace(0.0, 0.5 * float(a_dim), int(n))


def heldout_delta(a_dim, n):
    """A Delta grid disjoint from ``delta_grid`` at any n, plus the ends.

    Half-cell offsets of a partition into ``n`` cells cannot coincide
    with the endpoints of a partition into ``m``, so "held out" means
    held out rather than merely denser.
    """

    d_max = 0.5 * float(a_dim)
    m = int(n)
    return np.unique(np.concatenate([
        (np.arange(m) + 0.5) * d_max / m, [0.0, d_max]]))


def partition_fractions(n_p, ladder_floor_octaves=3):
    """The published nested partition; port of ``gw.mpa.sampling``.

    Exact rational arithmetic, so the nesting is bit-exact after the
    float conversion and ``partition_fractions(n)`` is a strict subset of
    ``partition_fractions(n+1)`` for every n.  That is what lets a
    shipped weight table at ``n_p_max`` serve every smaller ``n_p`` by
    selecting rows.
    """

    n = int(n_p)
    if n < 1:
        raise ValueError(
            f"GATE n_p_positive: n_p={n_p!r} is not a positive integer. "
            "FALSE case: int(n_p) >= 1.")
    if n == 1:
        return (Fraction(0),)
    pts = [Fraction(0), Fraction(1)]
    floor_width = Fraction(1, 2 ** int(ladder_floor_octaves))
    while len(pts) < n and (pts[1] - pts[0]) > floor_width:
        pts.insert(1, (pts[0] + pts[1]) / 2)
    while len(pts) < n:
        widths = [pts[i + 1] - pts[i] for i in range(1, len(pts) - 1)]
        widest = max(widths)
        k = widths.index(widest) + 1
        pts.insert(k + 1, (pts[k] + pts[k + 1]) / 2)
    return tuple(pts)


def sample_rows(a_dim, n_p_max=N_P_MAX, alphas=ALPHAS,
                line_ratio=LINE_RATIO):
    """The shipped weight rows: ``(line, fraction, alpha_set, z)``.

    The union over ``alpha`` of the nested partition fractions to
    ``n_p_max``, crossed with the two lines.  A row is identified by its
    EXACT fraction of ``omega_m`` -- the selector matches on that and
    never interpolates.

    THE ONE OMITTED SAMPLE, and why.  The near line's ``omega = 0`` point
    is ``z = 0``, which is the STATIC cell of R4's table and not the
    strip: ``K_0(Delta) = -2/Delta`` diverges at the bottom of the
    bandwidth domain, where this family's rows start.  It is served by
    ``exponential_sum`` and ``gw.mpa.sample_plan`` routes it there, so
    shipping a damped-line row for it would be shipping a rule for a
    function this family does not carry.  The far line's ``omega = 0``
    point is ``i*varpi_2``, which is finite everywhere on the band, and
    it IS shipped.
    """

    omega_max = 0.5 * float(a_dim)
    fracs = partition_fractions(int(n_p_max))
    seen: dict[Fraction, list[int]] = {}
    for a in alphas:
        for f in fracs:
            key = f ** int(a)
            seen.setdefault(key, []).append(int(a))
    rows = []
    for frac in sorted(seen):
        omega = float(frac) * omega_max
        for line, varpi in (("near", 1.0), ("far", float(line_ratio))):
            if line == "near" and frac == 0:
                continue
            rows.append({
                "line": line,
                "varpi": varpi,
                "fraction_num": int(frac.numerator),
                "fraction_den": int(frac.denominator),
                "alpha_sets": sorted(seen[frac]),
                "z": complex(omega, varpi),
            })
    return rows


def solve_points(a_dim, n_omega=25, line_ratio=LINE_RATIO):
    """The rectangle the LP sees: a continuum sweep of BOTH lines.

    Solving the rectangle rather than the partition is what buys
    ``n_p``-proofness, and gate zero priced it at between -1% and +14% of
    the partition's node count.  The alternative is one certificate per
    point of every ``n_p`` scan on every deck.
    """

    omega_max = 0.5 * float(a_dim)
    om = np.linspace(0.0, omega_max, int(n_omega) + 1)
    near = om[om > 0.0] + 1j
    far = om + 1j * float(line_ratio)
    return np.concatenate([near, far])


# ---------------------------------------------------------------------------
# Route 1 -- the positive composite rule
# ---------------------------------------------------------------------------

def _gauss_legendre_panel_bound(order, half_bandwidth, width, envelope):
    """Bernstein-ellipse bound for one panel; verbatim from the
    evaluator."""

    n = float(order)
    a = float(half_bandwidth)
    if n <= a:
        return np.inf
    rho = (n + np.sqrt(n * n - a * a)) / max(a, np.finfo(float).tiny)
    if rho <= 1.0 + 1.0e-14:
        return np.inf
    log_bound = a * (rho - 1.0 / rho) - 2.0 * n * np.log(rho)
    return ((64.0 / 15.0) * (0.5 * float(width)) * float(envelope)
            * np.exp(log_bound) / (1.0 - rho ** (-2.0 * n)))


def damped_line_rule(varpi, freq_max, *, rel_tol=1.0e-8,
                     wavelengths_per_panel=16.0, max_order=256):
    """Port of ``gw.mpa.evaluator.damped_line_rule``, character for
    character.

    Ported rather than imported so this tool stays free of jax and of
    ``src`` -- the sibling generator has the same property and the same
    reason.  The port is not trusted: the harness asserts it reproduces
    the shipped function's nodes and weights bit for bit at every cell
    this catalog ships.
    """

    v = float(varpi)
    f = float(freq_max)
    tol = float(rel_tol)
    wl = float(wavelengths_per_panel)
    if not (v > 0.0) or not np.isfinite(v):
        raise ValueError(
            f"GATE damped_line_positive_varpi: varpi={varpi!r} is not a "
            "finite positive line height. FALSE case: varpi > 0.")
    if not (f > 0.0) or not np.isfinite(f):
        raise ValueError(
            f"GATE damped_line_positive_bandwidth: freq_max={freq_max!r}"
            " is not finite positive. FALSE case: max|omega| + Delta_max"
            " > 0.")
    t_max = np.log(2.0 / tol) / v
    panel_target = wl * 2.0 * np.pi / f
    n_panels = max(1, int(np.ceil(t_max / panel_target)))
    width = t_max / n_panels
    half_bandwidth = (f + v) * width / 4.0
    eps_panel = 0.5 * (tol / v) / n_panels

    nodes, weights, orders = [], [], []
    for k in range(n_panels):
        left = k * width
        envelope = np.exp(-v * left)
        order = int(np.ceil(half_bandwidth)) + 1
        while order <= max_order and _gauss_legendre_panel_bound(
                order, half_bandwidth, width, envelope) > eps_panel:
            order += 1
        if order > max_order:
            raise ValueError(
                f"GATE damped_line_panel_order: panel {k} of {n_panels} "
                f"needs order above max_order={max_order}.")
        x, w = _leggauss(order)
        nodes.append(left + 0.5 * width * (x + 1.0))
        weights.append(0.5 * width * w)
        orders.append(order)

    t = np.concatenate(nodes).astype(np.float64)
    h = np.concatenate(weights).astype(np.float64)
    if not np.all(h > 0.0):
        raise AssertionError(              # pragma: no cover
            "Gauss-Legendre weights must be positive; positivity is this "
            "route's entire stability argument.")
    return {
        "t": t, "h": h, "n_nodes": int(t.size), "n_panels": int(n_panels),
        "orders": tuple(orders), "t_max": float(t_max), "varpi": v,
        "freq_max": f, "a_dim": float(f / v), "rel_tol": tol,
        "kappa0": float(v * np.sum(h * np.exp(-v * t))),
    }


def composite_weights(t, h, z):
    """``w_l(z) = -2 h_l e^{i z t_l}`` -- closed form at any z."""

    return KERNEL_FACTOR * np.asarray(h) * np.exp(
        1j * complex(z) * np.asarray(t))


def composite_entry_rule(a_dim, rel_tol):
    """The composite rule this family ships: the NEAR line's.

    One node set has to serve both lines, and the near line's is the one
    that does.  Its truncation is sized at ``varpi_1`` so on the far line
    the tail is ``e^{-10 t_max}`` rather than ``e^{-t_max}`` -- ten times
    the decades it needs -- while its panels are sized on
    ``max|omega| + Delta_max``, which is the same angular bandwidth on
    both lines.  So the far line rides it with room, which check 2 of the
    harness measures rather than assumes.
    """

    return damped_line_rule(1.0, float(a_dim), rel_tol=float(rel_tol))


def composite_two_line_cost(a_dim, rel_tol, line_ratio=LINE_RATIO):
    """What the per-line composite costs: ONE RULE PER LINE.

    This is the baseline clause (iii) compares against, and it is the
    honest one -- it is what the pipeline spends today
    (``evaluate_samples(batching='per-line')`` calls
    ``damped_line_rule`` once per line and runs two sweeps).
    """

    near = damped_line_rule(1.0, float(a_dim), rel_tol=float(rel_tol))
    far = damped_line_rule(float(line_ratio), float(a_dim),
                           rel_tol=float(rel_tol))
    return int(near["n_nodes"] + far["n_nodes"]), near, far


# ---------------------------------------------------------------------------
# Route 2 -- bounded-total-variation minimax on the rectangle
# ---------------------------------------------------------------------------

def sine_matrix(delta, t):
    """``Phi[m, l] = sin(Delta_m t_l)`` -- the dictionary, evaluated."""

    return np.sin(np.outer(np.asarray(delta, dtype=np.float64),
                           np.asarray(t, dtype=np.float64)))


def dictionary(a_dim, rel_tol, *, n_log=320, tail_factor=1.4,
               floor_factor=1.0e-3):
    """Log-tau dictionary, a Nyquist spine, and the composite's own nodes.

    Three families unioned, and each is here to remove an excuse.  The
    LOG grid is the spec's dictionary and is what a genuinely sparse
    solution wants -- scales, not samples.  The NYQUIST spine (uniform,
    spacing ``pi/Delta_max``) is the finest placement any rule could USE,
    because two nodes closer than that are the same function of ``Delta``
    on the band.  And the COMPOSITE's nodes are in it, so a node count
    above the baseline cannot be blamed on the atoms on offer: the
    selector was free to rediscover the shipped rule.
    """

    d_max = 0.5 * float(a_dim)
    t_max = np.log(2.0 / float(rel_tol))
    lo = float(floor_factor) / d_max
    hi = float(tail_factor) * t_max
    log_part = np.exp(np.linspace(np.log(lo), np.log(hi), int(n_log)))
    step = np.pi / d_max
    spine = np.arange(step, hi + 0.5 * step, step)
    comp = composite_entry_rule(a_dim, rel_tol)["t"]
    t = np.unique(np.concatenate([log_part, spine, comp]))
    return t[t > 0.0]


def _ls(phi, g):
    """Least squares by real QR, with a rank-deficient fallback.

    ``numpy.linalg.lstsq`` uses an SVD driver and promotes the real
    dictionary to complex against a complex right-hand side; measured on
    this problem that is two to three orders slower than a reduced QR,
    which is the difference between a campaign that runs in an evening
    and one that does not.
    """

    from scipy.linalg import solve_triangular

    q, r = np.linalg.qr(np.asarray(phi, dtype=np.float64))
    d = np.abs(np.diag(r))
    if d.size and d.min() > 1.0e-13 * d.max():
        return solve_triangular(r, q.T @ np.asarray(g))
    return np.linalg.lstsq(np.asarray(phi, dtype=np.float64),
                           np.asarray(g), rcond=None)[0]


class _Ridge:
    """One SVD per support; every ridge taken on it is then O(n^2).

    THE AMPLIFICATION CAP, IMPOSED WITHOUT A LINEAR PROGRAM.  The cap is
    ``kappa0 = varpi * sum_l |w_l| / 2 <= kappa_max``, an L1 ball.  A
    least-squares route cannot carry an L1 row, so it carries the L2
    surrogate and bisects the ridge until the MEASURED L1 mass obeys the
    cap.  The surrogate is an L2 ball inscribed in the L1 ball, so it is
    conservative: a node count certified this way is one the exact
    minimax LP certifies too.  What the catalog records is always the
    measured ``sum_abs_w`` and never the surrogate.
    """

    def __init__(self, phi, g):
        self.u, self.s, self.vt = np.linalg.svd(
            np.asarray(phi, dtype=np.float64), full_matrices=False)
        # Project the right-hand sides once; the bisection then never
        # touches the tall matrix again.
        self.c = self.u.T @ np.asarray(g).T          # (n, n_rhs)
        self.live = self.s > self.s[0] * 1.0e-13

    def solve(self, mu, cols=None):
        f = np.where(self.live, self.s / (self.s ** 2 + float(mu)), 0.0)
        c = self.c if cols is None else self.c[:, cols]
        return self.vt.T @ (f[:, None] * c)          # (N, n_rhs)

    def mass(self, mu, cols=None):
        return np.sum(np.abs(self.solve(mu, cols)), axis=0)

    def for_mass(self, cap, col, *, steps=40):
        """Smallest ``mu`` whose fit at column ``col`` obeys ``cap``."""

        cols = [int(col)]
        if float(self.mass(0.0, cols)[0]) <= float(cap):
            return 0.0
        hi = float(self.s[0] ** 2)
        for _ in range(200):
            if float(self.mass(hi, cols)[0]) <= float(cap):
                break
            hi *= 4.0
        lo = 0.0
        for _ in range(int(steps)):
            mid = 0.5 * (lo + hi)
            if float(self.mass(mid, cols)[0]) <= float(cap):
                hi = mid
            else:
                lo = mid
        return hi


def capped_fit(phi, g_all, mass):
    """Ridge-capped least squares, per sample.  Returns weights and the
    per-sample sup error on the fitting grid."""

    g = np.asarray(g_all)
    ridge = _Ridge(phi, g)
    cols = []
    for k in range(g.shape[0]):
        cols.append(ridge.solve(ridge.for_mass(mass[k], k), [k])[:, 0])
    w = np.stack(cols, axis=0)                        # (n_z, N)
    r = np.abs(w @ phi.T - g)
    return w, np.max(r, axis=1)


def lawson_polish(phi, g_all, mass, *, iters=25):
    """Chebyshev refit on a fixed support, under the cap, in pure numpy.

    The exact minimax LP is the formulation the spec names, but it
    inherits HiGHS's ABSOLUTE feasibility tolerance and this catalog
    reaches four decades below it -- at 1e-10 the scaled program returns
    ``model_status = Unknown``.  Lawson's iteratively reweighted least
    squares has no tolerance of its own: every quantity it forms is order
    one and the residual is evaluated directly.  Gate zero measured the
    two agreeing to one node where both are usable (67 against 68 at
    ``A = 20``, 1e-6), which is why this is the polish that ships.
    """

    g = np.asarray(g_all)
    m = phi.shape[0]
    out_w, out_e = [], []
    for k in range(g.shape[0]):
        lam = np.full(m, 1.0 / m)
        best_w, best_e = None, np.inf
        mu = _Ridge(phi, g[k][None, :]).for_mass(mass[k], 0)
        for _ in range(int(iters)):
            s = np.sqrt(lam)[:, None]
            aug = np.vstack([phi * s, np.sqrt(max(mu, 0.0)) * np.eye(
                phi.shape[1])]) if mu > 0 else phi * s
            rhs = (np.concatenate([g[k] * s[:, 0],
                                   np.zeros(phi.shape[1], dtype=complex)])
                   if mu > 0 else g[k] * s[:, 0])
            w = _ls(aug, rhs)
            r = np.abs(phi @ w - g[k])
            e = float(r.max())
            if e < best_e and float(np.sum(np.abs(w))) <= float(mass[k]):
                best_w, best_e = w, e
            tot = (lam * r).sum()
            if tot <= 0.0:
                break
            lam = lam * r / tot
        if best_w is None:
            best_w = _Ridge(phi, g[k][None, :]).solve(mu, [0])[:, 0]
            best_e = float(np.max(np.abs(phi @ best_w - g[k])))
        out_w.append(best_w)
        out_e.append(best_e)
    return np.stack(out_w, axis=0), np.asarray(out_e)


def omp_select(phi, g_all, scale, *, max_nodes, target=0.5):
    """Simultaneous orthogonal matching pursuit, incrementally.

    Node SELECTION only; the weights that ship come from the capped fit.
    Rows are scaled by their own budget so one number drives the whole
    rectangle.

    WHY THIS IS WRITTEN WITH UPDATES RATHER THAN RE-SOLVES.  The naive
    form re-solves a growing least-squares problem and re-scores the
    whole dictionary at every pick, which is ``O(m K^3)`` and
    ``O(n_z m N K)`` respectively -- at ``A = 200`` that is a quarter of
    a teraflop before the prune has started.  Carrying an orthonormal
    basis instead makes the residual update rank one, and the score and
    the orthogonal-complement norms then update in ``O(mN)`` per pick.
    Same picks, three orders cheaper.
    """

    g = np.asarray(g_all) / np.asarray(scale)[:, None]      # (n_z, m)
    phi = np.asarray(phi, dtype=np.float64)                 # (m, N)
    norm2 = np.einsum("ml,ml->l", phi, phi)                 # ||phi_l||^2
    corr = g @ phi                                          # (n_z, N)
    resid = g.copy()                                        # (n_z, m)
    picks, qs = [], []
    live = norm2 > 0.0
    for _ in range(int(max_nodes)):
        denom = np.where(live & (norm2 > 1.0e-12), norm2, np.inf)
        score = np.sum(np.abs(corr) ** 2, axis=0) / denom
        score = np.where(live, score, -1.0)
        j = int(np.argmax(score))
        if score[j] <= 0.0:
            break
        q = phi[:, j].copy()
        for qq in qs:
            q -= qq * (qq @ q)
        nrm = float(np.linalg.norm(q))
        if nrm <= 1.0e-10 * max(float(np.linalg.norm(phi[:, j])), 1.0):
            live[j] = False
            continue
        q /= nrm
        qs.append(q)
        picks.append(j)
        live[j] = False
        pq = phi.T @ q                       # (N,)  overlap of each atom
        gq = g @ q                           # (n_z,)
        # The residual, the score and the orthogonal-complement norms all
        # update rank-one against the new basis vector; nothing here
        # touches the tall matrix more than once per pick.
        corr -= np.outer(gq, pq)
        resid -= np.outer(gq, q)
        norm2 -= pq ** 2
        if float(np.max(np.abs(resid))) <= float(target):
            break
    return np.asarray(picks, dtype=int)


def _usefulness_order(phi, w):
    """Rank nodes by the LEAVE-ONE-OUT residual increase they protect.

    THE PRUNE-ORDERING FIX, and why the obvious ordering is wrong.  Gate
    zero cut the prune's bisection on a ``|w|``-ordered prefix and that
    stalls the moment composite nodes enter the support: Gauss-Legendre
    weights span orders of magnitude, so "smallest ``|w|``" is not "least
    useful", and three (span, tier) cells came back at or above the
    baseline with a held-out error three decades INSIDE budget -- the
    signature of a prune that stopped early rather than of a rule that
    needed those nodes.

    The right measure is the increase in squared residual from deleting
    a column and refitting, which for least squares is exactly

        delta_l = |w_l|^2 / [(Phi^T Phi)^{-1}]_{ll}

    -- the column's weight normalised by its own conditioning, summed
    over samples.  One pseudo-inverse of the Gram matrix gives all of
    them, so the fix costs ``O(N^3)`` once per cut and not ``N`` refits.
    """

    gram = np.asarray(phi, dtype=np.float64).T @ np.asarray(
        phi, dtype=np.float64)
    ginv = np.linalg.pinv(gram, rcond=1.0e-13)
    d = np.maximum(np.abs(np.diag(ginv)), 1.0e-300)
    score = np.sum(np.abs(np.asarray(w)) ** 2, axis=0) / d
    return np.argsort(-score)                     # most useful first


#: Wall-clock budget for ONE cell's sparse attempt, in seconds.  Not a
#: tuning knob but a cost ceiling with a correctness argument behind it:
#: the composite is a certified rule of the same family at every cell, so
#: abandoning a search costs node count and never correctness, and a
#: search that has not converged in this long is one whose answer clause
#: (iii) was going to reject.  The ledger records the abandonment by name
#: so a cell that shipped composite for cost reasons is distinguishable
#: from one that shipped composite because the sparse rule lost.
CELL_WALL_BUDGET_S = 420.0


def prune(t_sup, delta, g_all, budget, mass, *, tries=3, rounds=2,
          max_drops=30, deadline=None):
    """Cardinality prune: bisect on a usefulness-ordered prefix, then
    greedy, to a fixed point.

    Run to a fixed point and not once, because the ordering is computed
    on the CURRENT support and improves as the support shrinks.
    """

    def _out_of_time():
        return deadline is not None and time.perf_counter() > deadline

    t_s = np.asarray(t_sup, dtype=np.float64)
    for _ in range(int(rounds)):
        if _out_of_time():
            break
        before = t_s.size
        phi = sine_matrix(delta, t_s)
        w, _ = capped_fit(phi, g_all, mass)
        order = _usefulness_order(phi, w)
        lo, hi, best = 1, t_s.size, t_s
        while lo < hi:
            if _out_of_time():
                break
            mid = (lo + hi) // 2
            cand = np.sort(t_s[order[:mid]])
            _, err = capped_fit(sine_matrix(delta, cand), g_all, mass)
            if np.all(err <= budget):
                hi, best = mid, cand
            else:
                lo = mid + 1
        t_s = best
        dropping, drops = True, 0
        while (dropping and t_s.size > 2 and drops < int(max_drops)
               and not _out_of_time()):
            dropping = False
            phi = sine_matrix(delta, t_s)
            w, _ = capped_fit(phi, g_all, mass)
            idx = _usefulness_order(phi, w)[::-1]
            for i in idx[:int(tries)]:
                cand = np.delete(t_s, i)
                _, err = capped_fit(sine_matrix(delta, cand), g_all, mass)
                if np.all(err <= budget):
                    t_s, dropping, drops = cand, True, drops + 1
                    break
        if t_s.size >= before:
            break
    return np.sort(t_s)


def solve_sparse(a_dim, rel_tol, *, kappa_max=KAPPA_MAX, n_delta=None,
                 n_omega=33, margin=0.7, max_nodes=None,
                 deadline=None):
    """The sparse route at one cell.  Returns nodes or ``None``."""

    a = float(a_dim)
    n_delta = int(np.clip(16 * a, 600, 1200)) if n_delta is None \
        else int(n_delta)
    z = solve_points(a, n_omega=n_omega)
    delta = delta_grid(a, n_delta)
    g_all = damped_kernel(z[:, None], delta[None, :])
    tol_abs = float(rel_tol) / z.imag
    mass = 2.0 * float(kappa_max) / z.imag
    t_dict = dictionary(a, rel_tol)
    # THE SEARCH CEILING IS THE SHIPPING RULE, NOT A TUNING KNOB.  Clause
    # (iii) says a sparse entry ships only if it uses strictly fewer
    # nodes than the two composite rules it replaces, so a support at or
    # above that count cannot ship whatever its error is, and every
    # solve spent above it is provably wasted.  Bounding selection there
    # also bounds the prune, whose per-call SVD is cubic in the support:
    # unbounded, the A = 200 cells selected two thousand eight hundred
    # atoms and then began pruning them, which is hours of arithmetic
    # spent reaching an answer the shipping rule would reject anyway.
    ceiling = int(max_nodes) if max_nodes is not None else (
        composite_two_line_cost(a, rel_tol)[0] - 1)
    cap = int(min(t_dict.size, max(ceiling, 2)))
    picks = omp_select(sine_matrix(delta, t_dict), g_all, tol_abs,
                       max_nodes=cap, target=0.5 * margin)
    if picks.size == 0 or picks.size >= cap:
        # Selection ran to the ceiling without reaching its target, so
        # the sparse route has already lost clause (iii).  Saying so
        # here is what keeps the tail of the ladder affordable.
        return None
    return prune(t_dict[picks], delta, g_all, margin * tol_abs, mass,
                 deadline=deadline)


# ---------------------------------------------------------------------------
# Certification -- the eight checks
# ---------------------------------------------------------------------------

def payload_digest(t, w):
    """SHA-256 over the canonical little-endian payload bytes.

    The stamp is over the object that SHIPS.  It is the table's bytes
    that must agree across platforms, not the solve that produced them --
    the census measured one host producing two different solves four
    months apart under one cache key, which is exactly what a byte stamp
    names and a re-solve does not.
    """

    h = hashlib.sha256()
    h.update(np.asarray(t, dtype="<f8").tobytes())
    h.update(np.asarray(w, dtype="<c16").tobytes()
             if np.iscomplexobj(w) else
             np.asarray(w, dtype="<f8").tobytes())
    return h.hexdigest()


def moment_residual(t, w, z, a_dim):
    """``|int_0^Dmax (fit - target) dDelta|``, both sides closed form.

    An identity the solver never optimised, integrating the residual
    against a measure no grid point knows about, so a rule that merely
    interpolates the sample points cannot pass it by construction.
    ``int_0^D sin(Delta t) dDelta = (1 - cos(D t))/t`` against the
    kernel's own logarithm.
    """

    d = 0.5 * float(a_dim)
    t = np.asarray(t, dtype=np.float64)
    zc = complex(z)
    fit = np.sum(np.asarray(w) * (1.0 - np.cos(d * t)) / t)
    exact = KERNEL_FACTOR * 0.5 * np.log(
        (d ** 2 - zc ** 2) / (-zc ** 2))
    return float(abs(fit - exact))


def certify(t, w_rows, rows, a_dim, rel_tol, *, kappa_max=KAPPA_MAX,
            n_eval=None, rule="btv_minimax", h=None):
    """The per-entry certificate.  Every field is a measured number."""

    a = float(a_dim)
    n_eval = int(np.clip(64 * a, 2400, 4800)) if n_eval is None \
        else int(n_eval)
    d_eval = heldout_delta(a, n_eval)
    z = np.asarray([r["z"] for r in rows], dtype=np.complex128)
    varpi = z.imag
    tol_abs = float(rel_tol) / varpi
    phi = sine_matrix(d_eval, t)
    g = damped_kernel(z[:, None], d_eval[None, :])
    resid = np.asarray(w_rows) @ phi.T - g
    per_row = np.max(np.abs(resid), axis=1)
    sum_abs = np.sum(np.abs(np.asarray(w_rows)), axis=1)
    kappa = 0.5 * sum_abs * varpi

    near = np.asarray([r["line"] == "near" for r in rows])
    far = ~near
    cert = {
        "node_count": int(np.size(t)),
        "n_rows": int(len(rows)),
        "max_error": float(np.max(per_row)),
        "max_error_over_budget": float(np.max(per_row / tol_abs)),
        "near_line_max_over_budget": float(
            np.max(per_row[near] / tol_abs[near])) if near.any() else 0.0,
        "far_line_max_over_budget": float(
            np.max(per_row[far] / tol_abs[far])) if far.any() else 0.0,
        "kappa0": float(np.max(kappa)),
        "kappa0_min": float(np.min(kappa)),
        "kappa0_bound": float(kappa_max),
        "sum_abs_w": float(np.max(sum_abs)),
        "error_bound": float(rel_tol),
        "held_out_points": int(d_eval.size),
    }
    worst = int(np.argmax(per_row))
    cert["moment_residual"] = moment_residual(
        t, np.asarray(w_rows)[worst], z[worst], a)
    cert["moment_ceiling"] = MOMENT_FRACTION * float(rel_tol) * (
        0.5 * a) / float(varpi[worst])

    # Check 5 -- rescaling.  The family is EXACTLY scale free in varpi,
    # so this is an identity and the check is correspondingly sharp.
    ratios = []
    for scale in (1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4):
        tau = np.asarray(t) / scale
        aw = np.asarray(w_rows) / scale
        zz = z * scale
        dd = d_eval * scale
        rr = aw @ sine_matrix(dd, tau).T - damped_kernel(
            zz[:, None], dd[None, :])
        ratios.append(float(np.max(np.abs(rr), axis=1).max()
                            / (cert["max_error"] / scale)))
    cert["rescale_max_error_ratio"] = float(max(ratios))

    failures = []
    if cert["max_error_over_budget"] > 1.0:
        failures.append("held_out_error")
    if cert["far_line_max_over_budget"] > 1.0:
        failures.append("far_line")
    if cert["kappa0"] > float(kappa_max):
        failures.append("kappa0")
    if cert["moment_residual"] > cert["moment_ceiling"]:
        failures.append("moment")
    if cert["rescale_max_error_ratio"] > 1.05:
        failures.append("rescale")
    if not np.all(np.asarray(t) > 0.0):
        failures.append("positive_nodes")
    if rule == "positive_composite" and (h is None or not np.all(
            np.asarray(h) > 0.0)):
        failures.append("positive_weights")
    cert["failures"] = failures
    cert["certified"] = not failures
    cert["payload_sha256"] = payload_digest(
        t, np.asarray(h) if rule == "positive_composite" else w_rows)
    return cert


# ---------------------------------------------------------------------------
# The sweep driver
# ---------------------------------------------------------------------------

def _token(value, decimals=6):
    return f"{float(value):.{decimals}f}".replace(
        "-", "m").replace("+", "p").replace(".", "p")


def _error_token(value):
    mantissa, exponent = f"{float(value):.1e}".split("e")
    mantissa = mantissa.replace("-", "m").replace(".", "p")
    return f"{mantissa}e{exponent.replace('+', 'p').replace('-', 'm')}"


def entry_filename(a_dim, rel_tol):
    return (f"{FAMILY}_A_{_token(a_dim, 1)}"
            f"_eps_{_error_token(rel_tol)}.npz")


def build_entry(a_dim, rel_tol, *, kappa_max=KAPPA_MAX, verbose=True):
    """One cell: try the sparse route, then fall back.  Records both."""

    a = float(a_dim)
    tol = float(rel_tol)
    rows = sample_rows(a)
    z = np.asarray([r["z"] for r in rows], dtype=np.complex128)
    mass = 2.0 * float(kappa_max) / z.imag
    comp_total, comp_near, comp_far = composite_two_line_cost(a, tol)
    attempts = []
    chosen = None

    if tol not in COMPOSITE_ONLY_TIERS:
        # THE DISCRETISATION RESERVE, AND WHY IT IS A LOOP.  The prune
        # controls the residual on the omega sweep it solves against and
        # on its own Delta grid; the certificate is taken on the SHIPPED
        # partition fractions -- omegas the solve sweep never visited,
        # which is the whole point of solving the rectangle -- and on a
        # Delta grid four times finer.  Solving to a fraction of the
        # budget and certifying at the budget is the honest ordering, and
        # when the reserve turns out too thin the reserve is what
        # shrinks, never the certificate.
        for margin in (0.7, 0.4):
            t0 = time.perf_counter()
            try:
                t_s = solve_sparse(
                    a, tol, kappa_max=kappa_max, margin=margin,
                    deadline=t0 + CELL_WALL_BUDGET_S)
            except Exception as exc:                # pragma: no cover
                attempts.append({
                    "rule": "btv_minimax", "margin": margin,
                    "outcome": f"error:{type(exc).__name__}",
                    "wall_s": round(time.perf_counter() - t0, 2)})
                continue
            if t_s is None:
                attempts.append({"rule": "btv_minimax", "margin": margin,
                                 "outcome": "infeasible",
                                 "wall_s": round(
                                     time.perf_counter() - t0, 2)})
                continue
            n_delta = int(np.clip(16 * a, 600, 1200))
            delta = delta_grid(a, n_delta)
            g_rows = damped_kernel(z[:, None], delta[None, :])
            phi = sine_matrix(delta, t_s)
            if t_s.size <= 200:
                w_rows, _ = lawson_polish(phi, g_rows, mass)
            else:
                w_rows, _ = capped_fit(phi, g_rows, mass)
            cert = certify(t_s, w_rows, rows, a, tol,
                           kappa_max=kappa_max)
            wall = time.perf_counter() - t0
            beats = cert["node_count"] < comp_total
            outcome = ("pass" if cert["certified"] and beats else
                       ("fail:" + ",".join(cert["failures"])
                        if not cert["certified"] else
                        f"rejected:not_fewer_than_composite"
                        f"({cert['node_count']}>={comp_total})"))
            attempts.append({
                "rule": "btv_minimax", "kappa_cap": float(kappa_max),
                "margin": margin, "outcome": outcome,
                "node_count": cert["node_count"],
                "kappa0": cert["kappa0"], "wall_s": round(wall, 2)})
            if cert["certified"] and beats:
                chosen = ("btv_minimax", t_s, w_rows, None, cert, wall)
                break
            if not beats:
                break        # a thinner reserve only adds nodes
    else:
        attempts.append({"rule": "btv_minimax",
                         "outcome": "not_attempted:composite_only_tier"})

    if chosen is None:
        t0 = time.perf_counter()
        rule = composite_entry_rule(a, tol)
        w_rows = np.stack([composite_weights(rule["t"], rule["h"], zz)
                           for zz in z], axis=0)
        cert = certify(rule["t"], w_rows, rows, a, tol,
                       kappa_max=kappa_max, rule="positive_composite",
                       h=rule["h"])
        wall = time.perf_counter() - t0
        attempts.append({
            "rule": "positive_composite", "outcome":
            "pass" if cert["certified"] else
            "fail:" + ",".join(cert["failures"]),
            "node_count": cert["node_count"], "kappa0": cert["kappa0"],
            "wall_s": round(wall, 2)})
        chosen = ("positive_composite", rule["t"], w_rows, rule["h"],
                  cert, wall)

    rule_name, t, w_rows, h, cert, wall = chosen
    entry = {
        "family": FAMILY,
        "rule": rule_name,
        "range_param": "A",
        "A": a,
        "error_bound": tol,
        "error_metric": "linf_abs_complex_modulus_scaled",
        "line_height_ratio": LINE_RATIO,
        "node_count": cert["node_count"],
        "composite_node_count": comp_total,
        "composite_near_nodes": int(comp_near["n_nodes"]),
        "composite_far_nodes": int(comp_far["n_nodes"]),
        "compression_vs_composite": comp_total / cert["node_count"],
        "max_error": cert["max_error"],
        "max_error_over_budget": cert["max_error_over_budget"],
        "near_line_max_over_budget": cert["near_line_max_over_budget"],
        "far_line_max_over_budget": cert["far_line_max_over_budget"],
        "kappa0": cert["kappa0"],
        "kappa0_min": cert["kappa0_min"],
        "kappa0_bound": float(kappa_max),
        "kappa0_tier": "normal" if cert["kappa0"] <= 2.0 else
                       ("versioned_exception" if cert["kappa0"] <= 4.0
                        else "rejected"),
        "sum_abs_w": cert["sum_abs_w"],
        "moment_residual": cert["moment_residual"],
        "moment_ceiling": cert["moment_ceiling"],
        "rescale_max_error_ratio": cert["rescale_max_error_ratio"],
        "certified": cert["certified"],
        "payload_sha256": cert["payload_sha256"],
        "held_out_points": cert["held_out_points"],
        "file": f"{FAMILY}/{entry_filename(a, tol)}",
        "n_row": cert["n_rows"],
        "wall_s": round(wall, 2),
    }
    if rule_name == "btv_minimax":
        entry["n_p_max"] = int(N_P_MAX)
        entry["alpha_sets"] = list(ALPHAS)
        entry["sample_axis"] = "exact_fraction_match"
    else:
        entry["sample_axis"] = "closed_form_any_omega"
    if verbose:
        print(f"  A={a:5.0f} eps={tol:.0e}  {rule_name:18s} "
              f"nodes={cert['node_count']:5d} vs composite {comp_total:5d}"
              f"  ({comp_total / cert['node_count']:.2f}x)  "
              f"kappa0={cert['kappa0']:.3f}  "
              f"err/budget={cert['max_error_over_budget']:.3f}  "
              f"cert={cert['certified']}  {wall:.0f}s", flush=True)
    return entry, rows, t, w_rows, h, attempts


def write_payload(path, t, w_rows, h, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "t": np.asarray(t, dtype="<f8"),
        "w": np.asarray(w_rows, dtype="<c16"),
        "row_varpi": np.asarray([r["varpi"] for r in rows], dtype="<f8"),
        "row_fraction_num": np.asarray(
            [r["fraction_num"] for r in rows], dtype="<i8"),
        "row_fraction_den": np.asarray(
            [r["fraction_den"] for r in rows], dtype="<i8"),
    }
    if h is not None:
        arrays["h"] = np.asarray(h, dtype="<f8")
    with path.open("wb") as fh:
        np.savez(fh, **arrays)


def read_payload(path):
    with path.open("rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            out = {k: np.asarray(data[k]) for k in data.files}
    return out


def catalog_document(entries, ledger, wall):
    tool = Path(__file__).resolve()
    return {
        "schema_version": 2,
        "family": FAMILY,
        "target": {
            "definition": "K_z(Delta) = -2*Delta/(Delta**2 - z**2) for "
                          "Delta in [0, Delta_max], z = omega + i*varpi",
            "version": "damped_line/v1",
            "domain": "bandwidth",
            "domain_note": "There is no Delta_min and no gap "
                           "assumption: the tau route lives on the "
                           "bandwidth domain, which is why the gapless "
                           "case that is an R -> infinity catastrophe on "
                           "the interval families is not a special case "
                           "here. gw.mpa.sample_plan.FAMILIES["
                           "'damped_line']['domain'] already says so.",
            "scale_freedom": "K_z(Delta) = (1/varpi) * "
                             "Khat(Delta/varpi, omega/varpi) exactly, so "
                             "an entry is dimensionless and has NO "
                             "envelope axis. The runtime forms tau = "
                             "t/varpi_1 and w = w_hat/varpi_1.",
            "error_metric": "linf_abs_complex_modulus_scaled",
        },
        "selection_rule": {
            "A": "smallest_tabulated_ge_requested. Rounding UP is "
                 "conservative: a rule at larger A certifies a strictly "
                 "larger rectangle and the request's target is its "
                 "restriction.",
            "error_bound": "largest_tabulated_le_requested",
            "line_height_ratio": "EXACT MATCH REQUIRED. The far line's "
                                 "rows are what the node set was "
                                 "certified against; a different ratio "
                                 "is a different rectangle.",
            "n_p": f"bounded by n_p_max ({N_P_MAX}); weights selected by "
                   "EXACT partition-fraction match, never interpolated. "
                   "The partition nests strictly, so every smaller n_p "
                   "is a row subset.",
            "alpha": "EXACT match for rule='btv_minimax'. IGNORED for "
                     "rule='positive_composite': its nodes are "
                     "omega-independent and its weights are an exact "
                     "unit-modulus phase, so one composite entry serves "
                     "every partition and every alpha. This is the same "
                     "asymmetry complex_laplace already carries on beta.",
        },
        "shipping_rule": {
            "source": "WP9_CAMPAIGN_REPLAN.md section 2",
            "kappa0_definition": "varpi * sum_l |w_l(z)| / 2, worst case "
                                 "over shipped rows",
            "kappa0_reference": "the continuum kernel's own L1 mass "
                                "2/varpi, the same normalisation "
                                "gw.mpa.evaluator.damped_line_rule uses",
            "normal": 2.0,
            "versioned_exception": 4.0,
            "rejected_above": 4.0,
            "clause_iii": "A sparse entry ships only if it uses strictly "
                          "fewer nodes than the two composite rules it "
                          "replaces. Otherwise the composite ships, "
                          "which is why composite_node_count is recorded "
                          "on every entry.",
        },
        "certification": {
            "held_out_grid": "half-cell-offset Delta grid, both band "
                             "edges included, disjoint from the solver's "
                             "and four times finer; the omega axis is "
                             "the shipped sample rows on BOTH lines",
            "checks": [
                "held_out_error",
                "far_line",
                "kappa0",
                "moment",
                "rescale",
                "positive_nodes",
                "n_p_proof",
                "composite_alpha_waiver",
            ],
            "byte_identity": "payload_sha256 is SHA-256 over the "
                             "little-endian float64 t bytes followed by "
                             "the weights (complex128 w for sparse "
                             "entries, float64 h for composite ones). "
                             "The cross-platform claim is about shipped "
                             "bytes and never about a re-solve.",
        },
        "provenance": {
            "tool": f"tools/{tool.name}",
            "tool_sha256": hashlib.sha256(
                tool.read_bytes()).hexdigest(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "gate": "GATE0_DAMPED_LINE_LP.md",
            "plan": "WP9_CAMPAIGN_REPLAN.md",
            "seed_policy": "no seed: the composite is fixed "
                           "Gauss-Legendre panels and the sparse route "
                           "is deterministic (OMP selection, a ridge "
                           "bisection and a Lawson refit, none of which "
                           "draw).",
        },
        "sweep": {
            "solver_wall_seconds": round(float(wall), 2),
            "entries": len(entries),
            "certified": int(sum(1 for e in entries if e["certified"])),
            "by_rule": {
                "btv_minimax": int(sum(
                    1 for e in entries if e["rule"] == "btv_minimax")),
                "positive_composite": int(sum(
                    1 for e in entries
                    if e["rule"] == "positive_composite")),
            },
            "ledger": ledger,
        },
        "tables": entries,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-root", type=Path,
                    default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--spans", type=float, nargs="*", default=SPANS)
    ap.add_argument("--tiers", type=float, nargs="*", default=TIERS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--merge", action="store_true",
        help="keep the entries already in the catalog and compute only "
             "the cells it is missing. A sweep interrupted at the "
             "expensive top of the ladder is resumed rather than "
             "restarted, which is the other half of writing the catalog "
             "incrementally.")
    args = ap.parse_args(argv)

    cells = [(a, t) for a in args.spans for t in args.tiers]
    t0 = time.perf_counter()
    print(f"damped_line: {len(cells)} cells on {args.workers} workers",
          flush=True)

    # WRITTEN INCREMENTALLY, ON PURPOSE.  The first version of this
    # driver collected every cell and wrote the catalog once at the end,
    # which makes a sweep all-or-nothing: an hour of certified cells is
    # lost to one interruption, and on a shared box the expensive cells
    # at the top of the ladder are exactly where an interruption is
    # likely.  Writing after every result costs a few milliseconds per
    # cell and makes a partial sweep a usable artifact that names its
    # own gaps -- the catalog's `sweep.entries` says how many cells it
    # carries, and re-running fills the rest in.
    entries, ledger = [], []
    out = args.output_root / CATALOG_NAME
    if args.merge and out.is_file():
        prior = json.loads(out.read_text(encoding="utf-8"))
        entries = list(prior.get("tables", []))
        ledger = list(prior.get("sweep", {}).get("ledger", []))
        have = {(float(e["A"]), float(e["error_bound"]))
                for e in entries}
        cells = [c for c in cells
                 if (float(c[0]), float(c[1])) not in have]
        print(f"  merging: {len(entries)} entries kept, "
              f"{len(cells)} cells to compute", flush=True)

    def _record(cell, result):
        (a, tol) = cell
        entry, rows, t, w_rows, h, attempts = result
        write_payload(args.output_root / FAMILY / entry_filename(a, tol),
                      t, w_rows, h, rows)
        entries.append(entry)
        ledger.append({"A": a, "error_bound": tol, "attempts": attempts})
        entries.sort(key=lambda e: (e["A"], -e["error_bound"]))
        ledger.sort(key=lambda l: (l["A"], -l["error_bound"]))
        doc = catalog_document(entries, ledger,
                               time.perf_counter() - t0)
        out.write_text(json.dumps(doc, indent=2, sort_keys=False)
                       + "\n", encoding="utf-8")

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            handles = [(c, pool.apply_async(build_entry, c))
                       for c in cells]
            for cell, handle in handles:
                _record(cell, handle.get())
    else:
        for cell in cells:
            _record(cell, build_entry(*cell))

    wall = time.perf_counter() - t0
    doc = catalog_document(entries, ledger, wall)
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} -- {doc['sweep']['certified']}/"
          f"{doc['sweep']['entries']} certified, "
          f"{doc['sweep']['by_rule']} in {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

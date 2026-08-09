#!/usr/bin/env python3
"""Generate and certify the ``complex_laplace`` quadrature family.

WHAT THIS FAMILY IS, AND WHY IT IS ONE FAMILY AND NOT TWO.  The design
brief's R4 and the theory plan's section E both carry two names for
what turns out to be a single object.  ``noncrossing_imag`` is the
chi0 probe's rule -- it fits ``x/(x**2 + omega_hat**2)`` on
``[1, R]`` -- and ``complex_laplace`` is the Sigma-stage Laplace
complement's rule, which fits ``1/(u - i beta)`` on the same interval.
The first is the real part of the second::

    1/(x - i beta) = (x + i beta) / (x**2 + beta**2)

so ``Re`` is ``x/(x**2 + beta**2)``, which is ``common.minimax
._imag_target(x, omega_hat)`` verbatim at ``beta = omega_hat``.  A
table fitted against the COMPLEX target with real nodes and complex
weights therefore serves both consumers from one payload: the
Sigma-side consumer takes ``alpha``, the chi0-probe consumer takes
``alpha.real``, and because ``|Re E| <= |E|`` the complex-modulus
certificate dominates the real one.  That is the whole of R4's
"one campaign" claim, discharged constructively rather than argued.

THE INTEGRAL REPRESENTATION IS THE STABILITY ARGUMENT.  For every
``x > 0``::

    1/(x - i beta) = int_0^inf e^{-x t} e^{i beta t} dt

-- a Laplace transform whose density has unit modulus everywhere.  So
a composite quadrature of that integral with POSITIVE Gauss-Legendre
weights ``h_l`` produces rule weights ``w_l = h_l e^{i beta t_l}`` with
``|w_l| = h_l`` exactly, and the amplification functional

    kappa0 = max_{x in [1,R]}  x * sum_l |w_l| e^{-t_l x}

is bounded by ``x * int_0^T e^{-x t} dt <= 1`` analytically.  This is
the same certificate ``gw.mpa.evaluator.damped_line_rule`` measures at
``1.0000`` for the strip, in the same normalisation: the reference is
the pure-damping envelope integral ``1/x``, not the target's own value.
Measured against the shipped catalog it is a self-consistent metric --
every certified ``noncrossing`` table has strictly positive weights and
measures ``kappa0`` between ``1.0000`` and ``1.0215``.

WHY THE TARGET CANNOT BE A POSITIVE EXPONENTIAL SUM.  ``x/(x**2 +
beta**2)`` rises from zero at ``x = 0`` to a peak at ``x = beta``, so
it is not completely monotone, so by Bernstein's theorem it is NOT the
Laplace transform of a positive measure.  The positivity that makes the
static ``1/x`` family self-certifying is therefore unavailable here in
the weights; it survives only one level up, in the composite rule's
``h_l``, with the sign carried by an ANALYTICALLY KNOWN unit-modulus
density.  That distinction is the reason this file has two routes and
not one.

THE TWO ROUTES.

``positive_composite``
    Graded Gauss-Legendre panels on ``[0, T]``: geometric near zero so
    that ``e^{-R t}`` is resolved, capped at one wavelength of ``beta``
    out to the truncation length.  ``kappa0 = 1.0000`` and
    ``sum|w| = T`` hold by construction, so certification is a
    closed-form identity.  Costs about ``12 * beta`` nodes at the
    ``1e-6`` tier.  Its beta dependence is an exact phase on fixed
    nodes, so ONE such table serves every ``beta`` below the one it was
    built for -- the only route on which the omega-hat axis rounds.

``btv_minimax``
    Bounded-total-variation minimax, solved as a pair of linear
    programs over a fixed log-spaced node dictionary.  Reweighted L1
    drives cardinality down; the amplification cap enters as ``len(x)``
    linear rows, one per grid point, so ``kappa0 <= kappa_max`` is
    imposed rather than hoped for; the support is then polished by an
    exact minimax LP.  The whole solve is convex and deterministic --
    there is no local optimiser and no seed, which is the direct answer
    to the census's finding that the runtime solver "lands in different
    basins" and moves ``sum|w|`` by orders for a ten-percent change in
    ``R``.  Costs about 1.5x the uncertified runtime solve's node count.

Both routes are certified by the same battery (``certify``), and the
catalog records which route each entry carries.

Nothing here is imported by ``src/``.  This is an offline generator; the
tables it writes are staged into their own schema-v2 catalog, and wiring
that catalog into the runtime door is the landing commit's work, not
this one's -- deliberately, because the runtime selection rule has no
beta axis and would otherwise serve a ``beta' != beta`` table, which is
not conservative but simply wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src" / "common" / "minimax_assets"
FAMILY = "complex_laplace"

#: THE TWO CLAUSES SHIP AS TWO CATALOGS, ONE FAMILY.  ``beta`` is one
#: dimensionless number formed from two unrelated numerators -- a fitted
#: pole WIDTH over a window edge on the Sigma side, a sampling line
#: HEIGHT over a band gap on the chi0 and fit side -- and Gate 0 sec 2
#: measured them two orders apart.  An entry swept for one certifies
#: nothing about the other, and because the two ranges OVERLAP near 0.6
#: a single catalog would let a width request match a height entry
#: numerically.  Separate files, separate target versions, and the
#: selector's ``CATALOG_CLAUSE`` stamps each; the stamp is per CATALOG
#: because a sweep is per clause.
CLAUSES = {
    "height": {
        "subdir": "complex_laplace",
        "catalog": "catalog_complex_laplace.json",
        "version": "complex_laplace/v1",
        "numerator": "varpi / x_min -- a sampling line height over a "
                     "band gap (the chi0 probe and the MPA fit stage)",
        "beta_max": 64.0,
        "derivation": (
            "the height clause's edge is an OBSERVATION: the protocol "
            "fixes varpi_1 = 0.2 Ry and varpi_2 = 1 Ha = 2 Ry, the "
            "in-tree decks' gaps run 0.0497 to 0.3427 Ry, and the "
            "resulting omega-hat runs 0.58 to 40.214 "
            "(MINIMAX_REQUEST_CENSUS.md sec 7.2). 64 is the next rung "
            "of a geometric ladder above the largest measured value."),
    },
    "width": {
        "subdir": "complex_laplace_width",
        "catalog": "catalog_complex_laplace_width.json",
        "version": "complex_laplace_width/v1",
        "numerator": "Gamma_p / x_min -- a fitted pole width over a "
                     "window edge (the Sigma-stage Laplace complement)",
        "beta_max": 1.0,
        "derivation": (
            "the width clause's edge is DERIVED and not observed, which "
            "is why it is a rung worth building against. On a "
            "sign-definite branch gw.mpa.sigma_routing.route_pole sets "
            "x_min = min(E_A) + a_p with a_p = Re Omega_p and "
            "min(E_A) >= 0, so x_min >= Re Omega_p. The fit's fourth "
            "guard (gw.mpa.pade_fit, DEFAULT_GUARDS['width_ratio_max'] "
            "= 1.0) caps |Im Omega_p| <= width_ratio_max * Re Omega_p. "
            "Therefore beta = Gamma_p / x_min = |Im Omega_p| / x_min "
            "<= width_ratio_max * Re Omega_p / Re Omega_p = "
            "width_ratio_max = 1 for EVERY pole of EVERY field this "
            "fitter can produce. The rung and the guard are the same "
            "number, so the campaign is bounded by construction rather "
            "than by a histogram that happens to fit. On the crossing "
            "branches the same quantity is bounded more tightly still, "
            "by 1/edge_factor, because those windows floor x_min at "
            "edge_factor*Gamma_p."),
    },
}
CLAUSE = "height"
CATALOG_NAME = CLAUSES[CLAUSE]["catalog"]
SUBDIR = CLAUSES[CLAUSE]["subdir"]


def set_clause(name):
    """Point the module's family constants at one clause's artifacts.

    A module-level switch rather than a threaded argument because every
    path here -- filenames, the catalog name, the ``file`` field on each
    row, the recovery scan -- needs the same answer, and a sweep is
    always for exactly one clause.
    """

    global CLAUSE, CATALOG_NAME, SUBDIR
    if name not in CLAUSES:
        raise ValueError(
            f"GATE known_clause: clause={name!r} is not one of "
            f"{sorted(CLAUSES)}. FALSE case: the envelope has exactly "
            "two clauses and a sweep declares which one it is for, "
            "because beta alone cannot say -- the two overlap near 0.6.")
    CLAUSE = name
    CATALOG_NAME = CLAUSES[name]["catalog"]
    SUBDIR = CLAUSES[name]["subdir"]


# The census's measured request table (MINIMAX_REQUEST_CENSUS.md sec 3.1):
# three R buckets off the existing half-decade grid cover all seven deck
# spans, and six distinct omega-hat values are demanded.
CENSUS_R_BUCKETS = (21.544346900318832, 46.41588833612777, 100.0)
CENSUS_OMEGA_HAT = (5.836, 14.122, 16.006, 16.372, 39.284, 40.214)
DEFAULT_ERROR_BOUNDS = (1.0e-6,)

#: The census's x_min column (sec 3.1), verbatim, keyed by deck.  It is
#: the highest precision that document carries, and it is what the beta
#: selector's deck sweep scores against.
CENSUS_X_MIN_RY = {
    "hbn": 0.342664,
    "cohsex": 0.141617,
    "gnppm": 0.124953,
    "bispinor": 0.122161,
    "si_fast": 0.050916,
    "si_test": 0.050916,
    "si_bse": 0.049734,
}

#: The GN probe frequency every in-tree deck sets, and -- by C sec 7.1 --
#: also the MPA protocol's far sampling line varpi_2 = 1 Ha.  One number
#: serving two stages is why the fit stage's imaginary request set is the
#: same beta grid as production's rather than a second one.
PROBE_OMEGA_P_RY = 2.0

#: WHY THIS TABLE EXISTS AND THE ROUNDED ONE DOES NOT SUFFICE.  Gate 0
#: sec 9 warned that the beta grid is a specification in the deck's own
#: omega-hat and that three decimals is a display precision.  The beta
#: selector made that executable and it bit: scored at
#: ``2 Ry / x_min``, hbn misses its nearest certified band by 32 bands,
#: cohsex by 4.2 and si_fast/si_test by 2.9 -- only gnppm, bispinor and
#: si_bse land inside.  These are the same seven decks at the precision
#: the arithmetic actually produces.
DECK_OMEGA_HAT = {
    name: PROBE_OMEGA_P_RY / x_min
    for name, x_min in CENSUS_X_MIN_RY.items()
}

#: The six distinct values, sorted.  ``si_fast`` and ``si_test`` share an
#: x_min and therefore a beta; they differ only in R.
FULL_PRECISION_OMEGA_HAT = tuple(sorted(set(DECK_OMEGA_HAT.values())))

#: The fit stage's tier.  Build note V measured the evaluate->fit
#: recovery floor here, and the campaign measured that a table at this
#: tier moves it from 5.3e-08 to 2.2e-09 -- below the fit's own
#: synthesis baseline, which is how a floor announces itself.  Production
#: has never asked for a tier other than 1e-6 (C sec 2.6); this one is
#: the fit stage's and nothing else's.
FIT_STAGE_ERROR_BOUND = 1.0e-12

# Shipping rule, theory plan section E: kappa0 <= 2 ships normally,
# 2 < kappa0 <= 4 ships as a versioned exception, kappa0 > 4 is rejected
# regardless of training error.
KAPPA_NORMAL = 2.0
KAPPA_EXCEPTION = 4.0

# The moment identity's ceiling, as a fraction of the trivial bound
# ``error_bound * (R - 1)``.  A residual that equioscillates -- which is
# what both routes produce, one by minimax and one by a decaying
# truncation tail -- carries little DC component, so its integral sits
# well inside that bound: the certified entries measure roughly an
# order of magnitude below a quarter of it.  A quarter is therefore a
# wide gate that still refuses the one thing this check exists for, a
# near-constant offset, which no sup-norm test under the bound can see.
MOMENT_FRACTION = 0.25

_GL_ORDERS = (4, 6, 8, 10, 12, 16, 20, 24, 32)
_GL = {n: np.polynomial.legendre.leggauss(n) for n in _GL_ORDERS}


# ---------------------------------------------------------------------------
# The target and the two measurement grids
# ---------------------------------------------------------------------------

def target(x, beta):
    """``1/(x - i beta)``, the family's definition."""

    return 1.0 / (np.asarray(x, dtype=np.float64) - 1j * float(beta))


def solve_grid(R, n):
    """The log grid the solver sees.  Endpoints included."""

    return np.exp(np.linspace(0.0, np.log(float(R)), int(n)))


def held_out_grid(R, n):
    """A grid DISJOINT from ``solve_grid`` at any n, plus the endpoints.

    Half-cell-offset log points can never coincide with the solver's
    nodes, so "held out" means held out and not merely "denser".  The
    two endpoints are added back because the amplification functional
    and the error both peak at the interval ends and a certificate that
    skipped them would be certifying the wrong thing.
    """

    m = int(n)
    s = np.log(float(R)) * (np.arange(m) + 0.5) / m
    return np.unique(np.concatenate([np.exp(s), [1.0, float(R)]]))


def measure(t, w, R, beta, n_eval=20001):
    """The certificate's numbers, all on the held-out grid."""

    x = held_out_grid(R, n_eval)
    ex = np.exp(-np.outer(x, np.asarray(t, dtype=np.float64)))
    fit = ex @ np.asarray(w, dtype=np.complex128)
    g = target(x, beta)
    absamp = ex @ np.abs(np.asarray(w, dtype=np.complex128))
    return {
        "held_out_max_error": float(np.max(np.abs(g - fit))),
        "real_part_max_error": float(np.max(np.abs(g.real - fit.real))),
        "kappa0": float(np.max(x * absamp)),
        "sum_abs_w": float(np.sum(np.abs(w))),
        "n_eval": int(x.size),
    }


# ---------------------------------------------------------------------------
# Route 1 -- the positive composite rule
# ---------------------------------------------------------------------------

def _panel_exact(a, b, x, beta):
    nu = x - 1j * float(beta)
    return (np.exp(-nu * a) - np.exp(-nu * b)) / nu


def _panel_error(a, b, order, x, beta):
    xg, wg = _GL[order]
    t = 0.5 * (b - a) * (xg + 1.0) + a
    h = 0.5 * (b - a) * wg
    approx = (np.exp(-np.outer(x, t))
              @ (h.astype(np.complex128) * np.exp(1j * float(beta) * t)))
    return float(np.max(np.abs(approx - _panel_exact(a, b, x, beta))))


def composite_rule(R, beta, target_error, beta_min=None,
                   max_panels=4000):
    """Graded positive-composite Gauss-Legendre for the family.

    The panel edges grade geometrically from ``1/R`` -- the scale on
    which ``e^{-R t}`` decays, and therefore the scale the top of the
    interval demands -- and are capped at one wavelength of ``beta``
    thereafter.  Per-panel order is raised until that panel's own
    quadrature error, measured against the closed form and maximised
    over the interval, fits its share of half the budget; the other
    half pays for the truncation at ``T``.  A panel that cannot reach
    its share at the largest tabulated order is bisected instead.

    THE ENVELOPE, AND THE ONE PLACE IT IS NOT FREE.  ``beta`` enters
    the weights only as the unit-modulus phase ``e^{i beta t_l}``, so
    the same nodes serve every beta -- which is the whole reason this
    route can round the beta axis where minimax cannot.  But the two
    error terms move in opposite directions in beta.  The panels get
    HARDER as beta rises (faster oscillation), and the truncation gets
    EASIER (the tail ``e^{-xT}/|x - i beta|`` carries a ``1/|beta|``),
    so a rule sized at ``beta`` alone has a tail that is too short for
    every smaller beta -- by ``sqrt(1 + beta^2)``, which at beta = 40
    is a factor of forty and shows up immediately at beta = 0.  Pass
    ``beta_min`` to declare the bottom of the range the entry serves:
    the truncation is then sized there and the panels at ``beta``, and
    the entry certifies across the whole ``[beta_min, beta]`` band.
    ``beta_min = None`` means the entry serves its own beta only.

    Returns ``(t, w, info)`` with ``w = h * exp(i beta t)`` and ``h``
    strictly positive -- ``info['sum_h']`` is both ``sum|w|`` and the
    analytic bound on it.
    """

    R = float(R)
    beta = float(beta)
    tol = float(target_error)
    b_lo = float(beta) if beta_min is None else float(beta_min)
    if not 0.0 <= b_lo <= beta:
        raise ValueError(
            f"GATE composite_beta_band: beta_min={beta_min!r} is not in "
            f"[0, beta={beta!r}]. FALSE case: 0 <= beta_min <= beta -- "
            "the entry serves a band that ends at the beta it was "
            "phased for, because the panels are sized there.")
    x_probe = solve_grid(R, 200)

    # Truncation: |int_T^inf e^{-x t} e^{i beta t} dt| = e^{-xT}/|x-i b|,
    # worst at x = 1 AND at the smallest beta served; it gets half the
    # budget.
    t_max = max(np.log(2.0 / (0.5 * tol * np.hypot(1.0, b_lo))), 1.0)
    cap = 2.0 * np.pi / max(beta, 1.0)

    edges = [0.0, min(1.0 / R, cap, t_max)]
    while edges[-1] < t_max:
        edges.append(min(edges[-1] + min(edges[-1], cap), t_max))
    queue = list(zip(edges[:-1], edges[1:]))
    budget = 0.5 * tol / max(len(queue), 1)

    nodes, weights, orders = [], [], []
    while queue:
        if len(orders) > max_panels:
            raise RuntimeError(
                f"GATE composite_panel_budget: more than {max_panels} "
                f"panels at R={R}, beta={beta}, tol={tol:.1e}. FALSE "
                "case: the graded panel set closes -- loosen the "
                "tolerance or raise max_panels.")
        a, b = queue.pop(0)
        chosen = None
        for order in _GL_ORDERS:
            if _panel_error(a, b, order, x_probe, beta) <= budget:
                chosen = order
                break
        if chosen is None:
            mid = 0.5 * (a + b)
            queue = [(a, mid), (mid, b)] + queue
            continue
        xg, wg = _GL[chosen]
        nodes.append(0.5 * (b - a) * (xg + 1.0) + a)
        weights.append(0.5 * (b - a) * wg)
        orders.append(chosen)

    t = np.concatenate(nodes)
    h = np.concatenate(weights)
    order = np.argsort(t)
    t, h = t[order], h[order]
    if not np.all(h > 0.0):
        raise AssertionError(                        # pragma: no cover
            "Gauss-Legendre weights must be positive; positivity is "
            "this route's entire stability argument.")
    w = h.astype(np.complex128) * np.exp(1j * beta * t)
    info = {
        "rule": "positive_composite",
        "t_max": float(t_max),
        "n_panels": int(len(orders)),
        "panel_orders": (int(min(orders)), int(max(orders))),
        "sum_h": float(np.sum(h)),
        "beta_min": float(b_lo),
        "kappa0_bound": 1.0,
        "beta_axis": ("exact_phase_on_fixed_nodes" if b_lo < beta
                      else "exact_entry_only"),
    }
    return t, w, info


# ---------------------------------------------------------------------------
# Route 2 -- bounded-total-variation minimax, as two linear programs
# ---------------------------------------------------------------------------

def _lp_blocks(dictionary, x, kappa_max):
    """Residual and amplification blocks shared by both programs.

    The variable vector is ``[wr+, wr-, wi+, wi-]``, each of length
    ``len(dictionary)`` and each nonnegative, so ``|w|`` is linear:
    that is what lets the amplification cap be a row rather than a
    penalty.
    """

    phi = np.exp(-np.outer(x, np.asarray(dictionary, dtype=np.float64)))
    m, n = phi.shape
    pair = csr_matrix(np.hstack([phi, -phi]))
    zero = csr_matrix((m, 2 * n))
    a_re = hstack([pair, zero])
    a_im = hstack([zero, pair])
    a_kap = None
    if kappa_max is not None:
        a_kap = csr_matrix(np.hstack([phi * x[:, None]] * 4))
    return a_re, a_im, a_kap


def _lp_l1(dictionary, x, g, tol, node_weights, kappa_max):
    """min sum_l c_l |w_l|  s.t.  |resid| <= tol, kappa0 <= kappa_max.

    WHY THIS PROGRAM IS WRITTEN IN PHYSICAL UNITS AND STAYS THERE, AND
    WHY THAT PUTS A FLOOR UNDER THE TIER IT CAN REACH.  HiGHS judges
    primal feasibility on an absolute scale of about 1e-7.  At the 1e-6
    tier the residual budget is 5.5e-7, five times that -- tight, and it
    works.  At the fit stage's 1e-12 tier the budget is 5.5e-13, which
    the solver cannot distinguish from zero, so every weight vector
    would look feasible and the answer would mean nothing.

    Dividing the residual rows by ``tol`` fixes that in principle and
    was tried: it makes those rows carry coefficients of order 1e6 while
    the amplification rows stay of order ``R``, and the resulting 1e6:1
    row imbalance made the program report INFEASIBLE at (R = 46.4,
    beta = 16.006) -- a request the unscaled formulation solves in ten
    nodes.  Trading a tier this route cannot reach for the tier
    production actually asks for is a bad trade, so the scaling was
    reverted and the measurement kept.

    The consequence is a real boundary and not a preference: the
    ``btv_minimax`` route serves 1e-6, and the fit stage's 1e-12 tier is
    the ``positive_composite`` route's by construction.  That happens to
    be the right split anyway -- the fit stage evaluates 2n_p samples
    once, so it does not pay for nodes, and what it gains instead is the
    beta envelope that lets an off-grid request be served at all.
    """

    n = len(dictionary)
    a_re, a_im, a_kap = _lp_blocks(dictionary, x, kappa_max)
    box = float(tol) / np.sqrt(2.0)
    rows = [a_re, -a_re, a_im, -a_im]
    rhs = [g.real + box, -g.real + box, g.imag + box, -g.imag + box]
    if a_kap is not None:
        rows.append(a_kap)
        rhs.append(np.full(len(x), float(kappa_max)))
    res = linprog(np.tile(np.asarray(node_weights, dtype=np.float64), 4),
                  A_ub=vstack(rows, format="csr"),
                  b_ub=np.concatenate(rhs),
                  bounds=[(0.0, None)] * (4 * n), method="highs")
    if not res.success:
        return None
    v = res.x
    return (v[:n] - v[n:2 * n]) + 1j * (v[2 * n:3 * n] - v[3 * n:4 * n])


def _grid_error(nodes, w, x, g):
    """Max complex-modulus residual on the solver's own grid."""

    fit = np.exp(-np.outer(x, np.asarray(nodes, dtype=np.float64))) @ \
        np.asarray(w, dtype=np.complex128)
    return float(np.max(np.abs(g - fit)))


def _lp_minimax(support, x, g, kappa_max, scale=1.0):
    """min max|resid| over the support, still under the kappa0 cap.

    ``scale`` is accepted and ignored: the residual rows stay in
    physical units for the reason ``_lp_l1`` records.  The argument is
    kept so the call sites read the same as the program they mirror.
    """

    n = len(support)
    a_re, a_im, a_kap = _lp_blocks(support, x, kappa_max)
    m = len(x)
    col = csr_matrix(-np.ones((m, 1)))
    rows = [hstack([a_re, col]), hstack([-a_re, col]),
            hstack([a_im, col]), hstack([-a_im, col])]
    rhs = [g.real, -g.real, g.imag, -g.imag]
    if a_kap is not None:
        rows.append(hstack([a_kap, csr_matrix((m, 1))]))
        rhs.append(np.full(m, float(kappa_max)))
    c = np.zeros(4 * n + 1)
    c[-1] = 1.0
    res = linprog(c, A_ub=vstack(rows, format="csr"),
                  b_ub=np.concatenate(rhs),
                  bounds=[(0.0, None)] * (4 * n + 1), method="highs")
    if not res.success:
        return None, np.inf
    v = res.x
    w = (v[:n] - v[n:2 * n]) + 1j * (v[2 * n:3 * n] - v[3 * n:4 * n])
    return w, float(v[-1])


def btv_rule(R, beta, target_error, kappa_max=KAPPA_NORMAL,
             n_dict=500, n_grid=600, rounds=5, margin=0.55,
             prune_tries=3):
    """Bounded-total-variation minimax over a fixed node dictionary.

    ``margin`` is the fraction of the tolerance the LP is asked to hit
    on its own grid.  The rest is the discretisation reserve: the LP
    controls the residual only where it samples, and the certificate is
    taken on a grid the LP never saw, so a rule solved exactly to
    ``tol`` on the coarse grid routinely certifies just above it.
    Solving to ``margin * tol`` and certifying at ``tol`` is the honest
    ordering.

    Returns ``(t, w, info)`` or ``None`` if the program is infeasible at
    the requested cap -- which is a real answer and not a failure: it
    means no rule on this dictionary reaches this tolerance without
    exceeding the amplification the shipping rule allows.
    """

    R = float(R)
    tol = float(target_error) * float(margin)
    x = solve_grid(R, n_grid)
    g = target(x, beta)

    # THE RETRY LADDER, AND WHY A CERTIFIED GENERATOR NEEDS ONE.  HiGHS
    # reported this program INFEASIBLE at (R = 100, beta ~ 39.28) with
    # n_dict = 500 and feasible -- same rule, same support size, same
    # sum|w| -- at 400, 512 and 600.  It is a numerical artefact of one
    # dictionary size and not a boundary of the problem, and the first
    # sweep of this campaign believed it: that entry shipped as a
    # 566-node positive composite when an 8-node minimax rule existed.
    # A refusal that depends on a discretisation the caller never chose
    # is not a refusal, so infeasibility is only believed after the
    # ladder, and the ledger records which rung answered.
    sizes = [int(n_dict)] + [int(n_dict) + d for d in (-100, 12, 100)]
    best, used, retries = None, int(n_dict), 0
    for attempt, size in enumerate(sizes):
        dictionary = np.exp(np.linspace(np.log(0.01 / R), np.log(60.0),
                                        size))
        best = _btv_rounds(dictionary, x, g, tol, kappa_max, rounds)
        if best is not None:
            used, retries = size, attempt
            break
    if best is None:
        return None
    dictionary = np.exp(np.linspace(np.log(0.01 / R), np.log(60.0), used))

    t_s, w_s = best
    return _btv_finish(t_s, w_s, x, g, tol, kappa_max, prune_tries,
                       dictionary, used, retries, n_grid, rounds)


def _btv_rounds(dictionary, x, g, tol, kappa_max, rounds):
    """Reweighted-L1 rounds; the smallest admissible support, or None."""

    node_weights = np.ones(len(dictionary))
    best = None
    for _ in range(int(rounds)):
        w = _lp_l1(dictionary, x, g, tol, node_weights, kappa_max)
        if w is None:
            break
        mag = np.abs(w)
        peak = float(mag.max()) if mag.size else 0.0
        if peak <= 0.0:
            break
        support = dictionary[mag > 1.0e-6 * peak]
        w_s, err = _lp_minimax(support, x, g, kappa_max, tol)
        if w_s is None or err > tol:
            # THE POLISH IS AN OPTIMISATION, NOT THE FEASIBILITY PROOF.
            # The L1 program already carries |resid| <= tol and the
            # amplification cap as constraints, so its own solution is
            # admissible by construction; only the thresholding can lose
            # that, and when it does the honest move is to keep the
            # weights the feasible program returned rather than to
            # report the whole request infeasible.  Without this the
            # sweep refuses points its own first round had already
            # solved -- which is what (R = 100, beta ~ 39.28) was.
            keep = mag > 1.0e-6 * peak
            support, w_s = dictionary[keep], w[keep]
            err = _grid_error(support, w_s, x, g)
        if err <= tol and (best is None or len(support) < len(best[0])):
            best = (support, w_s)
        node_weights = 1.0 / (mag + max(1.0e-3 * peak, 1.0e-300))
    return best


def _btv_finish(t_s, w_s, x, g, tol, kappa_max, prune_tries, dictionary,
                used, retries, n_grid, rounds):
    """Greedy cardinality prune, then the record of how it was solved."""

    # Drop the least-weighted node, keep the drop only if the exact
    # minimax LP still fits under the tolerance.
    dropping = True
    while dropping and len(t_s) > 2:
        dropping = False
        for i in np.argsort(np.abs(w_s))[:int(prune_tries)]:
            cand = np.delete(t_s, i)
            w_c, err = _lp_minimax(cand, x, g, kappa_max, tol)
            if w_c is not None and err <= tol:
                t_s, w_s, dropping = cand, w_c, True
                break

    info = {
        "rule": "btv_minimax",
        "kappa0_bound": float(kappa_max),
        "dictionary_size": int(used),
        "dictionary_retries": int(retries),
        "dictionary_range": (float(dictionary[0]), float(dictionary[-1])),
        "solve_grid": int(n_grid),
        "solve_tolerance": float(tol),
        "reweight_rounds": int(rounds),
        "beta_axis": "exact_entry_only",
    }
    return np.asarray(t_s, dtype=np.float64), \
        np.asarray(w_s, dtype=np.complex128), info


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------

def payload_digest(t, w):
    """SHA-256 over the canonical little-endian payload bytes.

    The stamp idiom, applied to the object that actually ships.  It is
    the table's bytes that must agree across platforms, not the solve
    that produced them -- the census measured the same host producing
    two different solves four months apart under one cache key, which
    is exactly the failure a byte stamp names and a re-solve does not.
    """

    h = hashlib.sha256()
    h.update(np.asarray(t, dtype="<f8").tobytes())
    h.update(np.asarray(w, dtype="<c16").tobytes())
    return h.hexdigest()


def moment_residual(t, w, R, beta):
    """``|int_1^R (fit - target) dx|``, both sides in closed form.

    An identity the solver never optimised: it integrates the residual
    against a measure no grid point of the fit knows about, so a rule
    that merely interpolates the sample points cannot pass it by
    construction.
    """

    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.complex128)
    R = float(R)
    fit = np.sum(w * (np.exp(-t) - np.exp(-t * R)) / t)
    exact = np.log((R - 1j * float(beta)) / (1.0 - 1j * float(beta)))
    return float(abs(fit - exact))


def rescale_check(t, w, R, beta, n_draw=32, seed=20260808):
    """Randomized physical rescalings, as theory-plan section E asks.

    The runtime rescales a table from ``[1, R]`` to a physical
    ``[x_min, x_max]`` by ``tau = t/x_min``, ``alpha = w/x_min`` at
    ``omega_p = beta * x_min``.  The scaled absolute error is then
    ``error/x_min``.  This draws ``x_min`` over eleven decades and
    asserts the achieved physical error never exceeds that prediction
    by more than a whisker of rounding -- which is the claim the
    catalog's physical_error_note makes and which nothing has ever
    checked.
    """

    rng = np.random.default_rng(int(seed))
    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.complex128)
    base = measure(t, w, R, beta)["held_out_max_error"]
    worst = 0.0
    for x_min in 10.0 ** rng.uniform(-5.0, 6.0, int(n_draw)):
        tau, alpha = t / x_min, w / x_min
        omega_p = float(beta) * x_min
        x = held_out_grid(float(R), 4001) * x_min
        fit = np.exp(-np.outer(x, tau)) @ alpha
        exact = 1.0 / (x - 1j * omega_p)
        got = float(np.max(np.abs(exact - fit)))
        worst = max(worst, got / (base / x_min))
    return float(worst)


def beta_tolerance(t, w, R, beta, error_bound, n_bisect=24):
    """The widest ``|delta beta|`` this entry still certifies across.

    THE THIRD OPTION C SEC 3.1 DOES NOT LIST.  The census offers
    deck-pinned exact entries, a tabulated grid plus an envelope, or
    quantising ``omega_p`` to the grid -- and rejects rounding because
    the target is a different function at every beta.  All true, and
    all of it skips the quantitative question: HOW different?

        d/dbeta [ 1/(u - i beta) ] = i / (u - i beta)**2

    whose modulus is largest at ``u = 1``, where it is
    ``1/(1 + beta**2)``.  So the target moves by about
    ``|delta beta| / (1 + beta**2)`` -- which at beta = 16 is a
    suppression of 257, and means a table can absorb a beta mismatch
    two and a half orders larger than its own error budget.  That
    matters concretely: the census quotes its six omega-hat values to
    three decimals while the decks produce 16.006056518, and without a
    tolerance band the shipped entry would refuse the very request it
    was generated for.

    This measures the band rather than trusting the derivative: bisect
    on ``delta`` until the entry stops meeting its error bound at
    ``beta + delta``, on both signs, and return the smaller half-width.
    """

    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.complex128)
    x = held_out_grid(float(R), 4001)
    fit = np.exp(-np.outer(x, t)) @ w

    def ok(b):
        g = 1.0 / (x - 1j * float(b))
        return float(np.max(np.abs(g - fit))) <= float(error_bound)

    if not ok(beta):
        return 0.0
    half = np.inf
    for sign in (1.0, -1.0):
        lo, hi = 0.0, max(float(beta), 1.0)
        if ok(beta + sign * hi):
            half = min(half, hi)
            continue
        for _ in range(int(n_bisect)):
            mid = 0.5 * (lo + hi)
            if ok(beta + sign * mid):
                lo = mid
            else:
                hi = mid
        half = min(half, lo)
    return float(half)


def certify(t, w, R, beta, error_bound, kappa_max=KAPPA_EXCEPTION):
    """The per-entry certificate.  Every field is a measured number.

    A dict with a ``passes`` boolean and a ``failures`` list; the
    generator ships an entry only when ``passes`` is true, and the test
    suite re-derives every one of these from the shipped bytes.
    """

    stats = measure(t, w, R, beta)
    cert = dict(stats)
    cert["moment_residual"] = moment_residual(t, w, R, beta)
    cert["moment_ceiling"] = MOMENT_FRACTION * float(error_bound) * (
        float(R) - 1.0)
    cert["rescale_max_error_ratio"] = rescale_check(t, w, R, beta)
    cert["beta_tolerance"] = beta_tolerance(t, w, R, beta, error_bound)
    cert["payload_sha256"] = payload_digest(t, w)
    cert["node_count"] = int(len(t))
    cert["error_bound"] = float(error_bound)

    failures = []
    if cert["held_out_max_error"] > float(error_bound):
        failures.append("held_out_error")
    if cert["real_part_max_error"] > float(error_bound):
        failures.append("real_part_error")
    if cert["kappa0"] > float(kappa_max):
        failures.append("kappa0")
    if cert["moment_residual"] > cert["moment_ceiling"]:
        failures.append("moment")
    if cert["rescale_max_error_ratio"] > 1.05:
        failures.append("rescale")
    if not np.all(np.asarray(t) > 0.0):
        failures.append("positive_nodes")
    cert["failures"] = failures
    cert["passes"] = not failures
    cert["kappa0_tier"] = ("normal" if cert["kappa0"] <= KAPPA_NORMAL
                           else "versioned_exception"
                           if cert["kappa0"] <= KAPPA_EXCEPTION
                           else "rejected")
    return cert


# ---------------------------------------------------------------------------
# The sweep driver
# ---------------------------------------------------------------------------

def _token(value, decimals=6):
    text = f"{float(value):.{decimals}f}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _error_token(value):
    mantissa, exponent = f"{float(value):.1e}".split("e")
    mantissa = mantissa.replace("-", "m").replace(".", "p")
    return f"{mantissa}e{exponent.replace('+', 'p').replace('-', 'm')}"


def entry_filename(R, beta, error_bound):
    """The staged table's name, and the twelve decimals it must carry.

    Beta gets TWELVE decimals, not four, and the reason is the campaign
    this file exists for.  At four decimals a deck's own omega-hat and
    the census's displayed rounding of it collide -- 16.006 and
    16.00601826286684 are one filename -- so the two entries overwrite
    each other and the catalog ends up describing bytes that were
    solved somewhere else.  The name is the only place a staged table
    records which request it answers, which makes it the recovery path
    for a sweep that died before its catalog was written; a name that
    cannot tell two requests apart is not a record.
    """

    return (f"{SUBDIR}_R_{_token(R)}_b_{_token(beta, 12)}"
            f"_eps_{_error_token(error_bound)}.npz")


def write_table(path, t, w, cert):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        np.savez(
            fh,
            tau=np.asarray(t, dtype="<f8"),
            alpha=np.asarray(w, dtype="<c16"),
            max_error=np.asarray(cert["held_out_max_error"],
                                 dtype="<f8"),
            kappa0=np.asarray(cert["kappa0"], dtype="<f8"),
        )


def read_table(path):
    with path.open("rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            return (np.asarray(data["tau"], dtype=np.float64),
                    np.asarray(data["alpha"], dtype=np.complex128),
                    float(data["max_error"][()]),
                    float(data["kappa0"][()]))


def build_entry(R, beta, error_bound, *, prefer="btv", verbose=True):
    """Generate one entry: try the cheap certified route, then fall back.

    The order is an economics decision and it is stated here rather
    than buried.  ``btv_minimax`` is tried first because it costs an
    order of magnitude fewer nodes and every node is one chi0 build in
    the consumer; ``positive_composite`` ships whenever the minimax
    route is infeasible under the amplification cap or fails any
    certificate check.  The entry records which happened.
    """

    attempts = []
    chosen = None
    if prefer == "btv":
        for cap in (KAPPA_NORMAL, KAPPA_EXCEPTION):
            t0 = time.perf_counter()
            got = btv_rule(R, beta, error_bound, kappa_max=cap)
            wall = time.perf_counter() - t0
            if got is None:
                attempts.append({"rule": "btv_minimax", "kappa_cap": cap,
                                 "outcome": "infeasible",
                                 "wall_s": round(wall, 2)})
                continue
            t, w, info = got
            cert = certify(t, w, R, beta, error_bound, kappa_max=cap)
            attempts.append({"rule": "btv_minimax", "kappa_cap": cap,
                             "outcome": "pass" if cert["passes"]
                             else "fail:" + ",".join(cert["failures"]),
                             "node_count": int(len(t)),
                             "kappa0": cert["kappa0"],
                             "wall_s": round(wall, 2)})
            if cert["passes"]:
                chosen = (t, w, info, cert, wall)
                break

    if chosen is None:
        t0 = time.perf_counter()
        # The fallback ships as an ENVELOPE entry: sized to serve every
        # beta down to zero, so a route that had to give up node economy
        # at least buys back the one thing minimax cannot offer.
        t, w, info = composite_rule(R, beta, error_bound, beta_min=0.0)
        wall = time.perf_counter() - t0
        cert = certify(t, w, R, beta, error_bound, kappa_max=KAPPA_EXCEPTION)
        attempts.append({"rule": "positive_composite",
                         "kappa_cap": KAPPA_EXCEPTION,
                         "outcome": "pass" if cert["passes"]
                         else "fail:" + ",".join(cert["failures"]),
                         "node_count": int(len(t)),
                         "kappa0": cert["kappa0"],
                         "wall_s": round(wall, 2)})
        chosen = (t, w, info, cert, wall)

    t, w, info, cert, wall = chosen
    if verbose:
        print(f"  R={R:10.4f} beta={beta:8.3f} eps={error_bound:.0e} "
              f"-> {info['rule']:19s} N={len(t):4d} "
              f"err={cert['held_out_max_error']:.3e} "
              f"kappa0={cert['kappa0']:.4f} "
              f"[{cert['kappa0_tier']}] {wall:.1f}s", flush=True)
    return t, w, info, cert, attempts, wall


def provenance():
    tool = Path(__file__).resolve()
    return {
        "tool": str(tool.relative_to(REPO_ROOT)),
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "lp_solver": "scipy.optimize.linprog(method='highs')",
        "seed_policy": ("no seed: both routes are deterministic convex "
                        "programs or fixed Gauss-Legendre panels. The "
                        "only randomness is the rescaling check's draw, "
                        "which is seeded at 20260808 and is a test, not "
                        "a solve."),
    }


def sweep(output_root, r_values, beta_values, error_bounds, prefer="btv",
          verbose=True, checkpoint=None):
    """Solve, certify and stage the grid, one cell at a time.

    ``checkpoint(entries, ledger, wall)`` is called after EVERY entry,
    and the catalog is what it writes.  THE REASON IS A MEASURED ONE
    FROM THE NEIGHBOURING CAMPAIGN: an all-or-nothing driver on the
    damped_line ladder lost an hour of already-certified cells to a
    single interruption near the top, because the tables were on disk
    but nothing had recorded that they were certified.  Solving is the
    expensive half and certifying is the irreversible half; neither
    should be hostage to the cell after it.
    """

    out_dir = Path(output_root) / SUBDIR
    entries, ledger = [], []
    total_wall = 0.0
    for error_bound in error_bounds:
        for R in r_values:
            for beta in beta_values:
                t, w, info, cert, attempts, wall = build_entry(
                    R, beta, error_bound, prefer=prefer, verbose=verbose)
                total_wall += wall
                name = entry_filename(R, beta, error_bound)
                write_table(out_dir / name, t, w, cert)
                entries.append(_entry_record(
                    R, beta, error_bound, info, cert, name))
                ledger.append({"range_max": float(R), "beta": float(beta),
                               "error_bound": float(error_bound),
                               "attempts": attempts})
                if checkpoint is not None:
                    checkpoint(entries, ledger, total_wall)
    return entries, ledger, total_wall


def catalog_document(entries, ledger, total_wall):
    return {
        "schema_version": 2,
        "family": FAMILY,
        "clause": {
            "name": CLAUSE,
            "numerator": CLAUSES[CLAUSE]["numerator"],
            "beta_max": CLAUSES[CLAUSE]["beta_max"],
            "derivation": CLAUSES[CLAUSE]["derivation"],
            "note": (
                "beta is ONE dimensionless number formed from two "
                "unrelated numerators, and the two ranges overlap near "
                "0.6, so a request must say which clause it is asking "
                "under. common.minimax_beta_selector.CATALOG_CLAUSE "
                "stamps this catalog's target.version with the clause "
                "above and refuses a request from the other one by "
                "name."),
        },
        "target": {
            "definition": "1/(u - i*beta) on u in [1, R]",
            "version": CLAUSES[CLAUSE]["version"],
            "real_part_alias": (
                "Re 1/(u - i*beta) = u/(u^2 + beta^2) is "
                "common.minimax._imag_target(u, omega_hat) at "
                "omega_hat = beta, so alpha.real serves the "
                "noncrossing_imag consumer from the same payload."),
            "error_metric": "linf_abs_complex_modulus_scaled",
            "physical_error_note": (
                "runtime rescales from [1,R] to [x_min, x_max] by "
                "tau/x_min, alpha/x_min at omega_p = beta*x_min, so "
                "absolute error scales as max_error/x_min. The "
                "rescale_max_error_ratio field is that claim, measured "
                "over eleven decades of x_min."),
        },
        "selection_rule": {
            "range": "smallest_tabulated_ge_requested",
            "error_bound": "largest_tabulated_le_requested",
            "max_nodes": "table_node_count_must_not_exceed_request",
            "beta": (
                "|requested - entry beta| <= entry beta_tolerance, a "
                "MEASURED two-sided band and not a rounding rule. The "
                "target is a different function at every beta, so no "
                "direction is conservative; but d/dbeta of the target "
                "has modulus 1/(1 + beta^2) at the hard end of the "
                "interval, so an entry absorbs a mismatch of about "
                "error_bound*(1 + beta^2) -- 2.6e-4 at beta = 16 and "
                "the 1e-6 tier. Entries whose beta_axis is "
                "'exact_phase_on_fixed_nodes' additionally serve the "
                "whole band [beta_min, beta] exactly, because there "
                "beta enters only as a unit-modulus phase on nodes "
                "chosen without reference to it."),
        },
        "shipping_rule": {
            "source": "MPA_THEORY_PLAN.md section E",
            "kappa0_definition":
                "max over u in [1,R] of u * sum_l |w_l| e^{-t_l u}",
            "kappa0_reference": (
                "the pure-damping envelope integral 1/u, which is the "
                "same normalisation gw.mpa.evaluator.damped_line_rule "
                "uses for the strip; every certified shipped "
                "noncrossing table measures between 1.0000 and 1.0215 "
                "under it."),
            "normal": KAPPA_NORMAL,
            "versioned_exception": KAPPA_EXCEPTION,
            "rejected_above": KAPPA_EXCEPTION,
        },
        "certification": {
            "held_out_grid": (
                "half-cell-offset log grid of 20001 points plus both "
                "endpoints, disjoint from every grid the solver saw"),
            "checks": ["held_out_error", "real_part_error", "kappa0",
                       "moment", "rescale", "positive_nodes"],
            "byte_identity": (
                "payload_sha256 is SHA-256 over the little-endian "
                "float64 tau bytes followed by the complex128 alpha "
                "bytes. Cross-platform agreement is asserted on the "
                "TABLE, not on a re-solve."),
        },
        "provenance": provenance(),
        "sweep": {
            "solver_wall_seconds": round(float(total_wall), 2),
            "entries": len(entries),
            "certified": sum(1 for e in entries if e["certified"]),
            "by_rule": {
                rule: sum(1 for e in entries if e["rule"] == rule)
                for rule in sorted({e["rule"] for e in entries})},
            "ledger": ledger,
        },
        "tables": entries,
    }


def _entry_record(R, beta, error_bound, info, cert, name):
    """One catalog row.  The single place a row's shape is written.

    Both the sweep and the disk-adoption path build rows here, so a
    recovered catalog and a freshly swept one cannot differ in shape --
    which is the only way "adopted from disk" can be a provenance note
    rather than a second format.
    """

    return {
        "family": FAMILY,
        "rule": info["rule"],
        "range_param": "R",
        "range_max": float(R),
        "beta_param": "omega_hat",
        "beta": float(beta),
        "beta_min": float(info.get("beta_min", beta)),
        "error_bound": float(error_bound),
        "error_metric": "linf_abs_complex_modulus_scaled",
        "node_count": int(cert["node_count"]),
        "max_error": float(cert["held_out_max_error"]),
        "real_part_max_error": float(cert["real_part_max_error"]),
        "kappa0": float(cert["kappa0"]),
        "kappa0_bound": float(info["kappa0_bound"]),
        "kappa0_tier": cert["kappa0_tier"],
        "sum_abs_w": float(cert["sum_abs_w"]),
        "moment_residual": float(cert["moment_residual"]),
        "moment_ceiling": float(cert["moment_ceiling"]),
        "rescale_max_error_ratio":
            float(cert["rescale_max_error_ratio"]),
        "beta_tolerance": float(cert["beta_tolerance"]),
        "certified": bool(cert["passes"]),
        "payload_sha256": cert["payload_sha256"],
        "beta_axis": info["beta_axis"],
        "file": f"{SUBDIR}/{name}",
        "generation": {k: v for k, v in info.items()
                       if k not in ("rule", "kappa0_bound", "beta_axis")},
    }


def _parse_entry_filename(name):
    """``(R, beta, error_bound)`` back out of a staged table's name.

    The filename is the only place a staged ``.npz`` records which
    request it answers, so an interrupted sweep's tables are readable
    without the catalog that never got written.  Tokens are the
    generator's own: ``p`` for the decimal point, ``m``/``p`` for the
    exponent sign.
    """

    stem = Path(name).stem
    if not stem.startswith(f"{SUBDIR}_R_"):
        return None
    try:
        rest = stem[len(f"{SUBDIR}_R_"):]
        r_tok, rest = rest.split("_b_", 1)
        b_tok, e_tok = rest.split("_eps_", 1)
        mant, expo = e_tok.split("e", 1)
        return (float(r_tok.replace("p", ".", 1).replace("p", "")),
                float(b_tok.replace("p", ".", 1).replace("p", "")),
                float(mant.replace("p", ".").replace("m", "-")
                      + "e" + expo.replace("m", "-").replace("p", "+")))
    except (ValueError, IndexError):
        return None


def adopt_staged(output_root, verbose=True):
    """Rebuild catalog rows from tables already on disk.  No re-solving.

    THE RECOVERY PATH THE NEIGHBOURING CAMPAIGN PAID FOR.  Solving is
    the expensive half; a sweep that dies before writing its catalog has
    the bytes but no record that they were certified, and re-solving to
    recover a record is the wrong trade.  This reads each staged table,
    certifies it, and reconstructs its row -- including which ROUTE made
    it, which is not stored in the payload and is not guessed either:
    the composite rule for that cell is rebuilt (it costs milliseconds)
    and the payload digests are compared.  An exact digest match is a
    proof of provenance; anything else is the minimax route.
    """

    root = Path(output_root)
    rows = []
    for path in sorted((root / SUBDIR).glob("*.npz")):
        parsed = _parse_entry_filename(path.name)
        if parsed is None:
            continue
        R, beta, bound = parsed
        t, w, _err, _kap = read_table(path)
        cert = certify(t, w, R, beta, bound, kappa_max=KAPPA_EXCEPTION)
        comp_t, comp_w, comp_info = composite_rule(R, beta, bound,
                                                   beta_min=0.0)
        is_comp = (payload_digest(comp_t, comp_w)
                   == cert["payload_sha256"])
        info = comp_info if is_comp else {
            "rule": "btv_minimax", "kappa0_bound": KAPPA_NORMAL,
            "beta_axis": "exact_entry_only", "beta_min": beta,
            "adopted_from_disk": True}
        info = dict(info)
        info["adopted_from_disk"] = True
        rows.append(_entry_record(R, beta, bound, info, cert, path.name))
        if verbose:
            print(f"  adopted {path.name}  {info['rule']:19s} "
                  f"N={cert['node_count']:4d} "
                  f"-> {'PASS' if cert['passes'] else cert['failures']}",
                  flush=True)
    return rows


def _entry_key(entry):
    """The identity of a catalog row: (R, beta, tier), at solve precision.

    Rounded at 12 digits because that is the precision the generator's
    own cache keys and the runtime's ``round(..., 12)`` already use; two
    rows agreeing to twelve digits are the same request and the newer
    one replaces the older rather than sitting beside it.
    """

    return (round(float(entry["range_max"]), 12),
            round(float(entry["beta"]), 12),
            float(entry["error_bound"]))


def merge_entries(existing, fresh):
    """Append ``fresh`` to ``existing``, replacing rows of equal identity.

    Sorted by (tier, R, beta) so the file's order is a property of its
    contents and a diff between two sweeps is readable.
    """

    merged = {_entry_key(e): e for e in existing}
    replaced = sum(1 for e in fresh if _entry_key(e) in merged)
    for entry in fresh:
        merged[_entry_key(entry)] = entry
    ordered = sorted(merged.values(),
                     key=lambda e: (-float(e["error_bound"]),
                                    float(e["range_max"]),
                                    float(e["beta"])))
    return ordered, replaced


def recertify(output_root, verbose=True):
    """Re-derive every catalog field from the STAGED BYTES, not the solve.

    WP2's "independently validated before staging" step, and the only
    honest way to change a certificate threshold after a sweep: the
    tables are read back off disk and scored again, so what gets
    stamped is a property of the artefact rather than a memory of the
    process that made it.
    """

    root = Path(output_root)
    doc = json.loads((root / CATALOG_NAME).read_text(encoding="utf-8"))
    for entry in doc["tables"]:
        t, w, _stamped_err, _stamped_kap = read_table(root / entry["file"])
        cap = (KAPPA_NORMAL if entry["rule"] == "btv_minimax"
               else KAPPA_EXCEPTION)
        cert = certify(t, w, entry["range_max"], entry["beta"],
                       entry["error_bound"], kappa_max=cap)
        entry.update({
            "node_count": int(cert["node_count"]),
            "max_error": float(cert["held_out_max_error"]),
            "real_part_max_error": float(cert["real_part_max_error"]),
            "kappa0": float(cert["kappa0"]),
            "kappa0_tier": cert["kappa0_tier"],
            "sum_abs_w": float(cert["sum_abs_w"]),
            "moment_residual": float(cert["moment_residual"]),
            "moment_ceiling": float(cert["moment_ceiling"]),
            "rescale_max_error_ratio":
                float(cert["rescale_max_error_ratio"]),
            "beta_tolerance": float(cert["beta_tolerance"]),
            "certified": bool(cert["passes"]),
            "payload_sha256": cert["payload_sha256"],
        })
        if verbose:
            print(f"  R={entry['range_max']:10.4f} "
                  f"beta={entry['beta']:8.3f} {entry['rule']:19s} "
                  f"N={entry['node_count']:4d} "
                  f"moment={cert['moment_residual']:.3e} / "
                  f"{cert['moment_ceiling']:.3e} "
                  f"-> {'PASS' if cert['passes'] else cert['failures']}",
                  flush=True)
    doc["sweep"]["certified"] = sum(1 for e in doc["tables"]
                                    if e["certified"])
    doc["provenance"] = provenance()
    # The clause block travels with the artifact, so a catalog written
    # before the block existed picks it up here rather than staying the
    # one catalog whose envelope has to be looked up elsewhere.
    fresh = catalog_document([], [], 0.0)
    doc["clause"] = fresh["clause"]
    doc["target"] = fresh["target"]
    doc["shipping_rule"] = fresh["shipping_rule"]
    (root / CATALOG_NAME).write_text(
        json.dumps(doc, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--r-values", type=float, nargs="+",
                        default=list(CENSUS_R_BUCKETS))
    parser.add_argument("--beta-values", type=float, nargs="+",
                        default=None)
    parser.add_argument("--clause", choices=tuple(CLAUSES),
                        default="height",
                        help="which clause of the envelope this sweep is "
                             "for. Sets the catalog file, the target "
                             "version the selector stamps, and the "
                             "directory the tables are staged in.")
    parser.add_argument("--beta-set", choices=("census", "fullprec"),
                        default="census",
                        help="'census' is the displayed three-decimal "
                             "grid; 'fullprec' is 2 Ry / x_min for each "
                             "deck, which is what a request actually "
                             "carries. Ignored when --beta-values is "
                             "given, so a shell can never lose digits "
                             "the generator needs.")
    parser.add_argument("--error-bounds", type=float, nargs="+",
                        default=list(DEFAULT_ERROR_BOUNDS))
    parser.add_argument("--prefer", choices=("btv", "composite"),
                        default="btv")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--append", action="store_true",
                        help="merge into the existing catalog instead of "
                             "replacing it; rows of equal (R, beta, tier) "
                             "are replaced, everything else is kept")
    parser.add_argument("--adopt", action="store_true",
                        help="rebuild catalog rows from the tables "
                             "already staged on disk, without re-solving "
                             "anything; the recovery path for a sweep "
                             "that died before writing its catalog")
    parser.add_argument("--recertify", action="store_true",
                        help="re-score the staged tables and rewrite the "
                             "catalog without re-solving anything")
    args = parser.parse_args(argv)
    set_clause(args.clause)
    if args.beta_values is None:
        args.beta_values = list(
            FULL_PRECISION_OMEGA_HAT if args.beta_set == "fullprec"
            else CENSUS_OMEGA_HAT)

    if args.adopt:
        rows = adopt_staged(args.output_root, verbose=not args.quiet)
        path = Path(args.output_root) / CATALOG_NAME
        ledger = []
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            rows, _replaced = merge_entries(prior.get("tables", []), rows)
            ledger = list(prior.get("sweep", {}).get("ledger", []))
        doc = catalog_document(rows, ledger, 0.0)
        path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n",
                        encoding="utf-8")
        print(f"\nadopted {len(rows)} staged tables "
              f"({doc['sweep']['certified']} certified)")
        return 0 if doc["sweep"]["certified"] == len(rows) else 1

    if args.recertify:
        doc = recertify(args.output_root, verbose=not args.quiet)
        n = len(doc["tables"])
        print(f"\nrecertified {n} entries, "
              f"{doc['sweep']['certified']} passing")
        return 0 if doc["sweep"]["certified"] == n else 1

    print(f"complex_laplace sweep:{len(args.r_values)} R x "
          f"{len(args.beta_values)} beta x {len(args.error_bounds)} tier "
          f"= {len(args.r_values) * len(args.beta_values) * len(args.error_bounds)}"
          " entries", flush=True)
    path = Path(args.output_root) / CATALOG_NAME
    prior_tables, prior_ledger = [], []
    if args.append and path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_tables = prior.get("tables", [])
        prior_ledger = list(prior.get("sweep", {}).get("ledger", []))
    state = {"fresh": 0, "replaced": 0, "total": len(prior_tables)}

    def _checkpoint(entries, ledger, wall):
        merged, replaced = merge_entries(prior_tables, entries)
        doc = catalog_document(merged, prior_ledger + ledger, wall)
        tmp = path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
        state.update(fresh=len(entries), replaced=replaced,
                     total=len(merged),
                     certified=doc["sweep"]["certified"],
                     by_rule=doc["sweep"]["by_rule"])

    entries, ledger, wall = sweep(
        args.output_root, args.r_values, args.beta_values,
        args.error_bounds, prefer=args.prefer, verbose=not args.quiet,
        checkpoint=_checkpoint)
    if not entries:
        print("no entries generated")
        return 1
    doc = {"sweep": {"certified": state.get("certified", 0),
                     "by_rule": state.get("by_rule", {})}}
    print(f"\nwrote {state['fresh']} fresh entries "
          f"({state['replaced']} replacing rows of equal identity); "
          f"catalog now holds {state['total']} "
          f"({state.get('certified', 0)} certified) "
          f"after {wall:.1f} s of solver time")
    for rule, count in doc["sweep"]["by_rule"].items():
        print(f"  {rule:20s} {count}")
    return 0 if state.get("certified", 0) == state["total"] else 1


if __name__ == "__main__":
    sys.exit(main())

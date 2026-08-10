"""The BINNED-WIDTH clause of ``complex_laplace``: one rule per width bin.

WHAT THIS CAMPAIGN IS FOR, IN ONE MEASURED SENTENCE.  The MPA self-energy
pass costs what it costs because a pane is a certified rule and a tau-node
run of its own, and the shipped width clause demands 218 panes on a typical
non-crossing branch and 1312 on pole 7 of the ``si_mpa_0808`` fit -- against
GN-PPM's 24.1 s, a measured 5445 s per pole pass, node-weighted mean pane
occupancy 1.68 %.

READ THOSE PANE COUNTS AS AN UPPER BOUND, NOT AS PRODUCTION DEMAND.
``fix/mpa-qaxis-2026-08-09`` @ ``1e79e8fc`` (the ``screening_content``
declaration) found ``si_mpa_0808`` and every other production fit store
fitted to the FULL screened ``W`` rather than to ``W_c = W - v``, and
``|v|`` is 104-119 % of ``|W|``: that field's four-decade ``Gamma`` spread
and its 26 keV ``Re Omega`` are the bare-Coulomb tail, not physics.  On the
corrected ``W_c`` refit every pole narrows by 80-90x.

NOTHING BELOW DEPENDS ON THAT, and the separation is worth stating exactly
because it is what makes this campaign survive the defect: every number
this tool certifies is a supremum of an error functional over a
``(u, beta)`` rectangle, and NO POLE FIELD ENTERS IT.  The clause, its
derivation, the 30 entries and their band sups are properties of the rule
and the target alone.  What the defect DID put in question -- whether the
shipped ``beta <= 1`` clause already suffices on the narrowed field -- has
since been measured on ``mpa_fit_np8_wc.h5``, and it does: 43 leaves on
pole 7 and 40 on pole 0 against that clause's own 512 ceiling.  So this
campaign ships a ROBUSTNESS ASSET rather than a necessity for Si.  It still
buys ~6x in tau nodes on the real field, and it is the headroom for
wider-Gamma physics (metals, and any field whose widths outrun what
bisection can separate); nothing on the Si path is blocked without it.

The pane count IS the cost, and the pane count is set by
one inequality:

    beta = Gamma_hi / x_min <= beta_max = 1        (the shipped WIDTH clause)

``gw.mpa.sigma_pass._clause_safe_width_split`` bisects a Laplace bucket in
width until every leaf satisfies it, and a continuum of pole widths against
a shallow Laplace edge partitions into O(100) leaves.  Nothing about that is
wrong; it is the per-POLE envelope applied to a SLAB.

THE BINNED CLAUSE, AND WHY ITS EDGE IS ``r`` RATHER THAN 1
----------------------------------------------------------
Bin the widths geometrically at ratio ``r`` first, and serve each bin with
ONE rule.  Then the clause edge is derived rather than chosen, by the same
two facts the width clause's own edge is derived from:

* ``gw.mpa.pade_fit``'s fourth guard caps ``|Im Omega_p| <= Re Omega_p``
  (``width_ratio_max = 1``), so every pole in the bin has ``Gamma_p <= a_p``;
* on a sign-definite branch ``x_min = min(E_A) + a_lo`` with ``a_lo`` the
  bin's own smallest ``Re Omega`` and ``min(E_A) >= 0``.

Let the bin hold widths in ``[Gamma_lo, Gamma_hi]`` with
``Gamma_hi <= r * Gamma_lo``.  The pole attaining ``a_lo`` has some width
``Gamma >= Gamma_lo``, and the guard gives ``a_lo >= Gamma >= Gamma_lo``.
Hence

    beta = Gamma_hi / x_min <= Gamma_hi / (min(E_A) + Gamma_lo)
                            <= Gamma_hi / Gamma_lo <= r.

So a width-binned pane's clause closes at ``r`` EXACTLY, inclusive, and the
bin ratio and the clause edge are the same number -- which is the property
that makes this a clause and not a widening.  It is the owner row registered
at ``sigma_pass.MAX_WIDTH_SPLIT_LEAVES`` ("whether the slab width clause
should ... become slab-aware (beta <= r for a width-binned pane)"), computed.

WHAT AN ENTRY CERTIFIES, AND HOW THE SUP OVER THE BAND IS OBTAINED
-------------------------------------------------------------------
An entry is ``(R, r, eps)`` and it certifies the WORST CASE over the whole
band, not a point:

    sup { | sum_l h_l e^{-z t_l} - 1/z |  :  Re z in [1, R],
                                            -Im z in [0, r] }  <=  eps

with ``h_l > 0`` the composite rule's beta-independent magnitudes.  The
served object at a request's own beta is ``alpha_l = h_l e^{i beta t_l}``,
so the fit is ``sum_l h_l e^{-z t_l}`` at ``z = u - i beta`` exactly -- one
ANALYTIC function of one complex variable, and that is what makes the sup
obtainable rather than merely sampled.

**The method, stated honestly, in three steps.**

1. *Maximum modulus.*  ``Phi(z) = sum_l h_l e^{-z t_l} - 1/z`` is analytic
   on the closed rectangle (the only pole of ``1/z`` is at the origin and
   ``Re z >= 1``), so ``|Phi|`` is subharmonic and attains its maximum on
   the BOUNDARY.  ``Re Phi`` is harmonic and does the same.  A
   two-dimensional sup over the band therefore collapses, exactly and with
   no approximation, to four one-dimensional sups over the rectangle's
   edges.  This is the whole reason the band costs no more to certify than
   the line the shipped width entries already certify.

2. *Per-cell Taylor closure on each edge.*  Each edge is sampled on a grid
   disjoint from anything the builder saw; between samples the sup is closed
   rather than assumed.  On a cell of length ``d``, every interior point is
   within ``d/2`` of one of the two endpoints ``p``, so with Lagrange
   remainder at order ``m = 3``

       |Phi(x)| <= sum_{k<=m} |Phi^(k)(p)| (d/2)^k / k!
                   + C_{m+1} (d/2)^{m+1} / (m+1)!

   where each ``Phi^(k)(p)`` is a CLOSED FORM on the shipped bytes
   (``Phi^(n)(z) = (-1)^n [ sum_l h_l t_l^n e^{-z t_l} - n! z^{-(n+1)} ]``)
   and ``C_{m+1}`` is a rigorous cell-local ceiling,
   ``sum_l h_l t_l^{m+1} e^{-u_left t_l} + (m+1)!/u_left^{m+2}`` -- both
   terms decreasing in ``u``, so evaluating at the cell's left edge bounds
   the cell.  The maximum of the cell bounds is the reported sup.

   The ORDER is where the affordability lives, and it was measured: at
   order 1 the remainder on the shipped 20001-point grid is 4e-10, four
   hundred times the tightest tier, and buying that back by sampling costs
   ~1e5 times more points.  Order 3 puts it at ~1e-16 on the SAME grid.

3. *The remainder is reported, not hidden.*  ``band_closure_slack`` is how
   much of the certified sup is the between-samples remainder rather than a
   measured value.  It runs 1e-16 to 3e-10 across this sweep -- always
   under a percent of the tier, which is a shipping condition and not a
   hope -- and it is the number that says the theorem is carrying the
   certificate rather than the sampling.

The naive alternative -- sample the band densely in two dimensions and
Lipschitz-refine -- was costed and rejected: the crude bound
``|dPhi/dbeta| <= sum_l h_l t_l e^{-t_l} + 1/|z|^2`` is about 2 at ``u = 1``,
so closing a 1e-6 tier to within its own slack would need ~5e6 beta samples
per u sample.  The maximum-modulus reduction is not an optimisation here; it
is what makes the certificate exist.

THE BOUNDARY CONVENTION, DECIDED HERE AND RECORDED FOR EVERYONE
----------------------------------------------------------------
Bin membership is **half-open, closed at the TOP: ``(lo, hi]``**, and the
clause bound is closed at the top too (``beta <= r`` inclusive).  The
deciding criterion is certification-domain consistency, and the rule that
produces it is one sentence: A PANE'S CERTIFIED PARAMETER IS ITS SUPREMUM,
SO A PANE MUST CONTAIN ITS SUPREMUM.  Every rule this path builds is built
at the pane's largest width (``g_hi = max(g_v)`` in
``sigma_pass._mpa_groups_for_bucket``) and at the crossing pane's largest
``Re Omega`` (``T``, via ``a_v <= T``), so a pole sitting exactly on a
boundary belongs to the interval whose certificate was built AT its own
value -- covered inclusively, by exactly one rule, and never by neither.
``sigma_pass._geometric_width_bins_sorted`` is moved from ``side='left'`` to
``side='right'`` to say the same thing on the width axis that Sigma's B-side
predicate has always said on the ``Re Omega`` axis.

Run::

    python tools/generate_binned_width_assets.py --sweep
    python tools/generate_binned_width_assets.py --recertify

Everything shared with the height and width clauses -- the composite rule,
the held-out grid, the moment and rescale checks, the payload digest -- is
IMPORTED from ``generate_imag_minimax_assets`` rather than copied.  That
tool is not edited by this campaign, deliberately: its ``tool_sha256`` is
stamped into the two catalogs it already shipped, and a byte of drift there
would invalidate provenance that has nothing to do with this clause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_imag_minimax_assets import (        # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    FAMILY,
    KAPPA_EXCEPTION,
    KAPPA_NORMAL,
    MOMENT_FRACTION,
    REPO_ROOT,
    _error_token,
    _token,
    composite_rule,
    held_out_grid,
    moment_residual,
    payload_digest,
    rescale_check,
)

#: The clause's own identity.  Third clause of one family: same target
#: ``1/(u - i beta)``, same payload keys, a beta whose NUMERATOR is a
#: width-binned pane's largest width over its Laplace edge.
CLAUSE = "binned_width"
CATALOG_NAME = "catalog_complex_laplace_binned_width.json"
SUBDIR = "complex_laplace_binned_width"
VERSION = "complex_laplace_binned_width/v1"

#: The bin ratios the owner asked for.  Two octaves is the top: at ``r = 4``
#: a four-decade width spread is ~7 bins instead of ~1300 panes, and the
#: composite rule's dimensionless bandwidth ``A = beta + R - 1`` moves from
#: <=100 to <=103, which is why the collapse is nearly free in nodes.
BIN_RATIOS = (2.0, 4.0)

#: The R ladder, verbatim the one the shipped ``complex_laplace_width``
#: sweep certified, because it is the ladder the Sigma-side Laplace windows
#: actually ask on: ``sigma_pass.DEFAULT_LAPLACE_RATIO_MAX = 100`` bounds a
#: bucket's ratio and the crossing stripe/slab windows widen it by the
#: output half-window, so 215.44 is the first rung above every measured
#: request and 10 the first below.
R_LADDER = (10.0, 21.544346900318832, 46.41588833612777, 100.0,
            215.44346900318823)

#: The tiers.  ``plan_branch_groups(rel_tol=...)`` defaults to 1e-8 and that
#: is what production asks; 1e-6 and 1e-12 are the rungs either side, kept
#: because the shipped width clause carries them and a clause ladder with a
#: hole in it is a refusal waiting for a deck that moves one knob.
ERROR_BOUNDS = (1.0e-6, 1.0e-8, 1.0e-12)

#: Samples per edge of the certification rectangle.  20001 is the SAME
#: grid the shipped width clause certifies its single line on, kept
#: deliberately: with the order-3 closure below the remainder lands near
#: 1e-16 there, so the band costs the same sampling as the line and the
#: comparison between the two clauses' error columns is like for like.
#: The u edges are log spaced (the error lives at the bottom of the
#: interval), the beta edges linear (the band is linear in beta).
N_U_EDGE = 20001
N_BETA_EDGE = 20001


# ---------------------------------------------------------------------------
# The band certificate
# ---------------------------------------------------------------------------

def _phi_derivatives(t, h, z, n_max=2):
    """``Phi``, ``Phi'``, ``Phi''`` at every ``z``, from the shipped bytes.

    ``Phi(z) = sum_l h_l e^{-z t_l} - 1/z`` and, since
    ``d^n/dz^n (1/z) = (-1)^n n! z^{-(n+1)}``, the two halves carry the
    SAME sign factor::

        Phi^(n)(z) = (-1)^n [ sum_l h_l t_l^n e^{-z t_l} - n! z^{-(n+1)} ]

    Both halves are closed forms; nothing here is a finite difference,
    which is the point -- a sampled derivative would put the closure's
    rigour back on the sampling it is there to remove.
    """

    t = np.asarray(t, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    z = np.asarray(z, dtype=np.complex128)
    out = []
    # Chunked over z: the outer product is (n_z, n_nodes) complex128 and
    # n_z reaches 2e5 on the beta edges.
    chunk = max(1, int(4_000_000 // max(t.size, 1)))
    for n in range(int(n_max) + 1):
        acc = np.empty(z.shape, dtype=np.complex128)
        for s in range(0, z.size, chunk):
            zz = z[s:s + chunk]
            e = np.exp(-np.outer(zz, t))
            acc[s:s + chunk] = e @ (h * t ** n)
        sign = 1.0 if n % 2 == 0 else -1.0
        out.append(sign * (acc - float(math.factorial(n))
                           * z ** (-(n + 1))))
    return out


#: Order of the Taylor closure between samples.  Three, and the number is
#: measured rather than chosen: at order 1 the remainder
#: ``(d/2)^2 C2/2`` is 4e-10 on the shipped u grid -- four hundred times
#: the 1e-12 tier -- and closing it by sampling instead would need ~1e5
#: times more points.  Each order costs one more closed-form derivative
#: evaluation and buys a factor ``(d/2)/(k+1)``, so order 3 puts the
#: remainder at ~1e-16 on the SAME grid the shipped width clause already
#: uses.  That is the whole reason this certificate is affordable.
CLOSURE_ORDER = 3


def _derivative_ceiling(t, h, u_left, order):
    """A rigorous cell-local ceiling on ``|Phi^(order)|``, falling in ``u``.

    ``|Phi^(n)(z)| <= sum_l h_l t_l^n e^{-u t_l} + n!/|z|^{n+1}`` and
    ``|z| >= u`` on the rectangle, so both terms are bounded by their
    value at the cell's LEFT edge (both are decreasing in ``u``).
    Evaluating there bounds the whole cell -- which is what lets the u
    edges be log spaced without the top of the interval paying the
    bottom's step size.
    """

    t = np.asarray(t, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    u = np.asarray(u_left, dtype=np.float64)
    damp = np.exp(-np.outer(u, t)) @ (h * t ** int(order))
    return damp + float(math.factorial(int(order))) / u ** (int(order) + 1)


def _edge_bound(t, h, z_samples, u_left_for_cells, order=CLOSURE_ORDER):
    """The sup of ``|Phi|`` and ``|Re Phi|`` over ONE closed edge.

    ``z_samples`` are the edge's ordered sample points; consecutive pairs
    are the cells.  On a cell of length ``d``, every interior point is
    within ``d/2`` of one of the two endpoints ``p``, and Taylor with
    Lagrange remainder gives

        |Phi(x)| <= sum_{k<=order} |Phi^(k)(p)| (d/2)^k / k!
                    + C_{order+1} (d/2)^{order+1} / (order+1)!

    with every ``Phi^(k)(p)`` a CLOSED FORM on the shipped bytes and
    ``C_{order+1}`` the cell-local ceiling above.  Taking the max of the
    bracket over both endpoints is a valid bound whichever one is nearer.
    ``|Re Phi| <= |Phi|`` term by term, so the same expansion closes the
    real-part sup.

    Returns ``(sup_modulus, sup_real, max_at_samples_modulus,
    max_at_samples_real, closure_slack)`` -- the last being how much of
    the sup is remainder rather than measurement, which is the number
    that says whether the theorem or the sampling carries the
    certificate.
    """

    z = np.asarray(z_samples, dtype=np.complex128)
    derivs = _phi_derivatives(t, h, z, n_max=int(order))
    a_mod = np.abs(derivs[0])
    a_re = np.abs(derivs[0].real)
    half = 0.5 * np.abs(np.diff(z))
    ceil_next = _derivative_ceiling(
        t, h, np.asarray(u_left_for_cells, dtype=np.float64), int(order) + 1)
    tail = (half ** (int(order) + 1) * ceil_next
            / float(math.factorial(int(order) + 1)))

    def _cells(base):
        left = base[:-1].copy()
        right = base[1:].copy()
        for k in range(1, int(order) + 1):
            term = half ** k / float(math.factorial(k))
            left = left + term * np.abs(derivs[k][:-1])
            right = right + term * np.abs(derivs[k][1:])
        return np.maximum(left, right) + tail

    sup_mod = float(np.max(_cells(a_mod)))
    sup_re = float(np.max(_cells(a_re)))
    meas_mod = float(np.max(a_mod))
    meas_re = float(np.max(a_re))
    return sup_mod, sup_re, meas_mod, meas_re, sup_mod - meas_mod


def band_certificate(t, h, R, beta_lo, beta_hi, *,
                     n_u=N_U_EDGE, n_beta=N_BETA_EDGE):
    """The sup of the error over the whole ``(u, beta)`` band.

    Maximum modulus reduces the rectangle to its boundary; each of the
    four edges is closed cell by cell.  The four edges are reported
    individually because WHICH one carries the sup is a physical fact
    worth having: on every entry of this sweep it is the ``beta = beta_lo``
    edge at ``u = 1``, i.e. the TRUNCATION corner, which is exactly where
    ``composite_rule(beta_min=...)`` sizes the tail -- the certificate
    agreeing with the construction rather than with the campaign's hopes.
    """

    R = float(R)
    b_lo, b_hi = float(beta_lo), float(beta_hi)
    u_edge = np.unique(np.concatenate(
        [held_out_grid(R, int(n_u)), [1.0, R]]))
    b_edge = np.unique(np.concatenate(
        [np.linspace(b_lo, b_hi, int(n_beta)),
         b_lo + (b_hi - b_lo) * (np.arange(int(n_beta)) + 0.5) / int(n_beta)]))
    b_edge = b_edge[(b_edge >= b_lo) & (b_edge <= b_hi)]

    edges = {
        "beta_lo": (u_edge - 1j * b_lo, u_edge[:-1]),
        "beta_hi": (u_edge - 1j * b_hi, u_edge[:-1]),
        "u_lo": (1.0 - 1j * b_edge, np.full(b_edge.size - 1, 1.0)),
        "u_hi": (R - 1j * b_edge, np.full(b_edge.size - 1, R)),
    }
    per_edge = {}
    sup_mod = sup_re = meas_mod = meas_re = 0.0
    slack = 0.0
    for name, (z, u_left) in edges.items():
        sm, sr, mm, mr, sl = _edge_bound(t, h, z, u_left)
        per_edge[name] = {"sup_modulus": sm, "sup_real": sr,
                          "measured_modulus": mm, "measured_real": mr,
                          "n_samples": int(z.size)}
        sup_mod, sup_re = max(sup_mod, sm), max(sup_re, sr)
        meas_mod, meas_re = max(meas_mod, mm), max(meas_re, mr)
        slack = max(slack, sl)
    carrier = max(per_edge, key=lambda k: per_edge[k]["sup_modulus"])
    return {
        "band_beta_min": b_lo,
        "band_beta_max": b_hi,
        "band_sup_modulus": sup_mod,
        "band_sup_real": sup_re,
        "band_measured_modulus": meas_mod,
        "band_measured_real": meas_re,
        "band_closure_slack": slack,
        "band_sup_carrier_edge": carrier,
        "band_edges": per_edge,
        "band_method": (
            "maximum modulus on the analytic rectangle Re z in [1, R], "
            "-Im z in [beta_min, beta_max] (Phi(z) = sum h_l e^{-z t_l} "
            "- 1/z has no pole there), reducing the 2-D sup EXACTLY to the "
            "four edges; each edge closed cell by cell with a two-term "
            "Taylor expansion about the nearer sample and a cell-local "
            "third-derivative ceiling. band_closure_slack is the part of "
            "the reported sup that is remainder rather than measurement."),
    }


# ---------------------------------------------------------------------------
# One entry
# ---------------------------------------------------------------------------

def entry_filename(R, bin_ratio, error_bound):
    """``R``, the bin ratio and the tier, in the campaign's own tokens."""

    return (f"complex_laplace_binned_width_R_{_token(R)}"
            f"_r_{_token(bin_ratio, 1)}"
            f"_b_{_token(bin_ratio, 12)}"
            f"_eps_{_error_token(error_bound)}.npz")


def write_table(path, t, w, cert):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        tau=np.asarray(t, dtype=np.float64),
        alpha=np.asarray(w, dtype=np.complex128),
        max_error=np.float64(cert["band_sup_modulus"]),
        kappa0=np.float64(cert["kappa0"]),
    )


def read_table(path):
    with open(path, "rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            return (np.asarray(data["tau"], dtype=np.float64),
                    np.asarray(data["alpha"], dtype=np.complex128))


def certify_binned(t, w, R, bin_ratio, error_bound, *,
                   n_u=N_U_EDGE, n_beta=N_BETA_EDGE):
    """The per-entry certificate, over the BAND rather than at a point.

    Every check the width clause runs, plus the band sup that is this
    clause's whole reason for existing.  ``kappa0`` is measured over the
    band too: positivity makes it beta-independent in exact arithmetic
    (``|alpha_l| = h_l`` whatever the phase), and measuring it anyway is
    how a payload whose magnitudes are not what the rule claims is
    caught rather than assumed away.
    """

    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(w, dtype=np.complex128)
    h = np.abs(w)
    r = float(bin_ratio)

    cert = band_certificate(t, h, R, 0.0, r, n_u=n_u, n_beta=n_beta)

    x = held_out_grid(float(R), int(n_u))
    absamp = np.exp(-np.outer(x, t)) @ h
    cert["kappa0"] = float(np.max(x * absamp))
    cert["sum_abs_w"] = float(np.sum(h))
    cert["node_count"] = int(t.size)
    cert["error_bound"] = float(error_bound)
    cert["bin_ratio"] = r
    # The moment identity and the physical rescaling, both at the band's
    # hard end -- the beta the panels were sized at.
    cert["moment_residual"] = moment_residual(t, w, R, r)
    cert["moment_ceiling"] = MOMENT_FRACTION * float(error_bound) * (
        float(R) - 1.0)
    cert["rescale_max_error_ratio"] = rescale_check(t, w, R, r)
    cert["payload_sha256"] = payload_digest(t, w)

    failures = []
    if cert["band_sup_modulus"] > float(error_bound):
        failures.append("band_sup_error")
    if cert["band_sup_real"] > float(error_bound):
        failures.append("band_sup_real_error")
    if cert["band_closure_slack"] > 0.01 * float(error_bound):
        # Not an error claim -- a claim about WHO made the claim.  A
        # remainder worth a percent of the tier means the sampling is
        # carrying the certificate and the grid must be finer.
        failures.append("band_closure_slack")
    if cert["kappa0"] > KAPPA_EXCEPTION:
        failures.append("kappa0")
    if cert["moment_residual"] > cert["moment_ceiling"]:
        failures.append("moment")
    if cert["rescale_max_error_ratio"] > 1.05:
        failures.append("rescale")
    if not np.all(t > 0.0):
        failures.append("positive_nodes")
    cert["failures"] = failures
    cert["passes"] = not failures
    cert["kappa0_tier"] = ("normal" if cert["kappa0"] <= KAPPA_NORMAL
                           else "versioned_exception"
                           if cert["kappa0"] <= KAPPA_EXCEPTION
                           else "rejected")
    return cert


def build_entry(R, bin_ratio, error_bound, *, verbose=True, **kw):
    """One ``(R, r, eps)`` cell: build at the band's top, certify the band.

    There is no ``btv_minimax`` attempt here and that is a decision, not
    an omission.  A minimax rule's nodes are solved AT one beta and carry
    no structure in it (``beta_axis = exact_entry_only``), so it cannot
    certify a band at all; the composite route's nodes are chosen without
    reference to beta and beta enters only as a unit-modulus phase, which
    is the one property this clause is built on.  Asking the LP for a
    band rule would be asking the wrong route.
    """

    t0 = time.perf_counter()
    t, w, info = composite_rule(R, float(bin_ratio), error_bound,
                                beta_min=0.0)
    wall = time.perf_counter() - t0
    cert = certify_binned(t, w, R, bin_ratio, error_bound, **kw)
    if verbose:
        print(f"  R={R:10.4f} r={bin_ratio:4.1f} eps={error_bound:.0e} "
              f"-> N={t.size:4d} "
              f"band_sup={cert['band_sup_modulus']:.3e} "
              f"(slack {cert['band_closure_slack']:.1e}, on "
              f"{cert['band_sup_carrier_edge']}) "
              f"kappa0={cert['kappa0']:.6f} "
              f"{'PASS' if cert['passes'] else 'FAIL:' + ','.join(cert['failures'])}"
              f" {wall:.2f}s", flush=True)
    return t, w, info, cert, wall


def entry_record(R, bin_ratio, error_bound, info, cert, name):
    """One catalog row.  Every number on it was measured on these bytes."""

    return {
        "family": FAMILY,
        "rule": "positive_composite",
        "range_param": "R",
        "range_max": float(R),
        "beta_param": "omega_hat",
        "beta": float(bin_ratio),
        "beta_min": 0.0,
        "bin_ratio": float(bin_ratio),
        "bin_convention": "upper_closed",
        "error_bound": float(error_bound),
        "error_metric": "linf_abs_complex_modulus_scaled",
        "node_count": int(cert["node_count"]),
        # ``max_error`` is the BAND sup, not a point measurement, which is
        # the only difference between this row and a width-clause one that
        # a reader has to hold on to.
        "max_error": float(cert["band_sup_modulus"]),
        "real_part_max_error": float(cert["band_sup_real"]),
        "band_beta_min": float(cert["band_beta_min"]),
        "band_beta_max": float(cert["band_beta_max"]),
        "band_measured_modulus": float(cert["band_measured_modulus"]),
        "band_closure_slack": float(cert["band_closure_slack"]),
        "band_sup_carrier_edge": str(cert["band_sup_carrier_edge"]),
        "band_edge_samples": {k: int(v["n_samples"])
                              for k, v in cert["band_edges"].items()},
        "kappa0": float(cert["kappa0"]),
        "kappa0_bound": 1.0,
        "kappa0_tier": str(cert["kappa0_tier"]),
        "sum_abs_w": float(cert["sum_abs_w"]),
        "moment_residual": float(cert["moment_residual"]),
        "moment_ceiling": float(cert["moment_ceiling"]),
        "rescale_max_error_ratio": float(cert["rescale_max_error_ratio"]),
        # The band IS the tolerance on this clause: an entry certified on
        # [0, r] serves every request inside it exactly, so the two-sided
        # near-miss band a point entry needs has nothing to do here.  It is
        # carried at zero rather than omitted because ``beta_covers`` reads
        # it on every entry and a missing field is CatalogCorrupt.
        "beta_tolerance": 0.0,
        "certified": bool(cert["passes"]),
        "payload_sha256": str(cert["payload_sha256"]),
        "beta_axis": "exact_phase_on_fixed_nodes",
        "file": f"{SUBDIR}/{name}",
        "generation": {
            "t_max": float(info["t_max"]),
            "n_panels": int(info["n_panels"]),
            "panel_orders": list(info["panel_orders"]),
            "sum_h": float(info["sum_h"]),
            "beta_min": float(info["beta_min"]),
        },
    }


# ---------------------------------------------------------------------------
# The sweep and the catalog
# ---------------------------------------------------------------------------

def provenance():
    tool = Path(__file__).resolve()
    return {
        "tool": str(tool.relative_to(REPO_ROOT)),
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "shares_with": "tools/generate_imag_minimax_assets.py",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "lp_solver": "not used: this clause has no minimax route",
        "seed_policy": ("no seed: the composite is fixed Gauss-Legendre "
                        "panels and the band certificate is a closed-form "
                        "evaluation on declared grids. The only randomness "
                        "is the rescaling check's draw, seeded at 20260808 "
                        "in the shared tool, which is a test, not a solve."),
    }


def catalog_document(entries, ledger, total_wall):
    return {
        "schema_version": 2,
        "family": FAMILY,
        "clause": {
            "name": CLAUSE,
            "numerator": (
                "Gamma_hi / x_min over a width-BINNED pane -- the largest "
                "width in a pane whose widths span at most the bin ratio r, "
                "over that pane's Laplace edge (the Sigma-stage slab-aware "
                "complement of the per-pole width clause)"),
            "beta_max": float(max(BIN_RATIOS)),
            "derivation": (
                "the binned clause's edge is the BIN RATIO, and it is "
                "derived from the same two facts the per-pole width "
                "clause's edge is. gw.mpa.pade_fit's fourth guard caps "
                "|Im Omega_p| <= width_ratio_max * Re Omega_p at "
                "width_ratio_max = 1, so every pole in a bin has "
                "Gamma_p <= a_p; and on a sign-definite branch "
                "gw.mpa.sigma_routing.route_pole sets x_min = min(E_A) + "
                "a_lo with min(E_A) >= 0. The pole attaining a_lo has some "
                "width Gamma >= Gamma_lo, so a_lo >= Gamma_lo, so "
                "beta = Gamma_hi/x_min <= Gamma_hi/(min(E_A) + Gamma_lo) "
                "<= Gamma_hi/Gamma_lo <= r. The bin ratio and the clause "
                "edge are the same number, inclusive at r, which is what "
                "makes this a clause rather than a widening of the last "
                "one. At r = 1 it degenerates to the shipped width clause "
                "exactly."),
            "bin_convention": (
                "half-open, CLOSED AT THE TOP: (Gamma_lo, Gamma_hi]. A "
                "pane's certified parameter is its supremum -- every rule "
                "on this path is built at max(Gamma) over the pane and the "
                "crossing pane at its largest Re Omega (a <= T) -- so a "
                "pane must contain its supremum, and a pole sitting exactly "
                "on a bin edge is certified by the rule built AT its own "
                "width: by exactly one rule, inclusive of the endpoint, "
                "never by neither. The clause bound is closed at the top "
                "for the same reason (beta <= r, inclusive). "
                "gw.mpa.sigma_pass._geometric_width_bins_sorted uses "
                "searchsorted(side='right') to say this on the width axis."),
            "note": (
                "beta is ONE dimensionless number and this family now "
                "forms it from three unrelated numerators, whose ranges "
                "overlap. minimax.beta_selector.CATALOG_CLAUSE stamps this "
                "catalog's target.version with the clause above and "
                "refuses a request from either other one BY NAME."),
        },
        "target": {
            "definition": "1/(u - i*beta) on u in [1, R], beta in [0, r]",
            "version": VERSION,
            "real_part_alias": (
                "Re 1/(u - i*beta) = u/(u^2 + beta^2), so alpha.real serves "
                "the noncrossing_imag consumer from the same payload."),
            "error_metric": "linf_abs_complex_modulus_scaled",
            "band_note": (
                "every error number on an entry of THIS catalog is a "
                "supremum over the whole (u, beta) band and not a point "
                "measurement at one beta. That is the difference between "
                "this clause and complex_laplace_width/v1, and it is the "
                "difference the pane collapse is bought with."),
            "physical_error_note": (
                "runtime rescales from [1,R] to [x_min, x_max] by "
                "tau/x_min, alpha/x_min at Gamma = beta*x_min, so absolute "
                "error scales as max_error/x_min. rescale_max_error_ratio "
                "is that claim, measured over eleven decades of x_min."),
        },
        "selection_rule": {
            "range": "smallest_tabulated_ge_requested",
            "error_bound": "largest_tabulated_le_requested",
            "max_nodes": "table_node_count_must_not_exceed_request",
            "bin_ratio": (
                "smallest_tabulated_ge_requested. Rounding UP is "
                "conservative: an entry certified over beta in [0, r'] with "
                "r' >= r certifies a superset of the requested band, and "
                "the request's target is its restriction."),
            "beta": (
                "the entry certifies the whole band [0, r] as a SUPREMUM, "
                "so a request anywhere inside it is served exactly -- the "
                "beta_axis is exact_phase_on_fixed_nodes and the phase is "
                "re-evaluated at the request. beta_tolerance is 0 on this "
                "clause because a band certificate has no need of a "
                "near-miss band."),
        },
        "shipping_rule": {
            "source": ("the owner row registered at "
                       "gw.mpa.sigma_pass.MAX_WIDTH_SPLIT_LEAVES"),
            "kappa0_definition": "max over u in [1,R] of u * sum_l |w_l| e^{-t_l u}",
            "kappa0_reference": (
                "the pure-damping envelope integral 1/u; positivity makes "
                "this <= 1 by construction and beta-independent, because "
                "|alpha_l| = h_l whatever the phase."),
            "normal": KAPPA_NORMAL,
            "versioned_exception": KAPPA_EXCEPTION,
            "rejected_above": KAPPA_EXCEPTION,
        },
        "certification": {
            "band_sup_method": (
                "maximum modulus on the analytic rectangle, reducing the "
                "2-D sup EXACTLY to the four edges, then a per-cell Taylor "
                "closure with a cell-local third-derivative ceiling. See "
                "the tool's module docstring for the full statement and "
                "for the costing of the sampled alternative that was "
                "rejected."),
            "held_out_grid": (
                "u edges: half-cell-offset log grid of 20001 points plus "
                "both endpoints, disjoint from every grid the builder saw. "
                "beta edges: 20001 linear points plus their half-cell "
                "offsets. With the order-3 closure the between-samples "
                "remainder lands near 1e-16, which band_closure_slack "
                "reports per entry."),
            "checks": ["band_sup_error", "band_sup_real_error",
                       "band_closure_slack", "kappa0", "moment", "rescale",
                       "positive_nodes"],
            "byte_identity": (
                "payload_sha256 is SHA-256 over the little-endian float64 "
                "tau bytes followed by the complex128 alpha bytes."),
        },
        "provenance": provenance(),
        "sweep": {
            "solver_wall_seconds": round(float(total_wall), 2),
            "entries": len(entries),
            "certified": sum(1 for e in entries if e["certified"]),
            "by_rule": {"positive_composite": len(entries)},
            "ledger": ledger,
        },
        "tables": entries,
    }


def sweep(output_root, *, r_values=R_LADDER, bin_ratios=BIN_RATIOS,
          error_bounds=ERROR_BOUNDS, verbose=True, **kw):
    out_dir = Path(output_root) / SUBDIR
    entries, ledger = [], []
    total = 0.0
    for eps in error_bounds:
        for R in r_values:
            for r in bin_ratios:
                t, w, info, cert, wall = build_entry(
                    R, r, eps, verbose=verbose, **kw)
                total += wall
                name = entry_filename(R, r, eps)
                write_table(out_dir / name, t, w, cert)
                entries.append(entry_record(R, r, eps, info, cert, name))
                ledger.append({
                    "range_max": float(R), "bin_ratio": float(r),
                    "error_bound": float(eps),
                    "attempts": [{
                        "rule": "positive_composite",
                        "outcome": ("pass" if cert["passes"]
                                    else "fail:" + ",".join(cert["failures"])),
                        "node_count": int(cert["node_count"]),
                        "kappa0": cert["kappa0"],
                        "band_sup": cert["band_sup_modulus"],
                        "wall_s": round(wall, 2)}]})
    return entries, ledger, total


def recertify(output_root, *, verbose=True, **kw):
    """Re-derive every certificate from the SHIPPED BYTES and diff it.

    The campaign's own falsification check: a catalog whose numbers can
    only be reproduced by re-running the solver is a catalog that has not
    been checked, which is the census's most uncomfortable finding wearing
    a different hat.
    """

    root = Path(output_root)
    doc = json.loads((root / CATALOG_NAME).read_text(encoding="utf-8"))
    bad = []
    for entry in doc["tables"]:
        t, w = read_table(root / entry["file"])
        cert = certify_binned(t, w, entry["range_max"], entry["bin_ratio"],
                              entry["error_bound"], **kw)
        for key, field in (("band_sup_modulus", "max_error"),
                           ("band_sup_real", "real_part_max_error"),
                           ("kappa0", "kappa0"),
                           ("payload_sha256", "payload_sha256")):
            got, want = cert[key], entry[field]
            same = (got == want if isinstance(want, str)
                    else abs(float(got) - float(want)) <= 1e-15 * max(
                        1.0, abs(float(want))))
            if not same:
                bad.append(f"{entry['file']}: {field} {want!r} -> {got!r}")
        if verbose:
            print(f"  {entry['file']}: band_sup="
                  f"{cert['band_sup_modulus']:.6e} vs claimed "
                  f"{entry['max_error']:.6e} "
                  f"{'OK' if cert['passes'] else 'FAIL'}", flush=True)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--recertify", action="store_true")
    ap.add_argument("--n-u", type=int, default=N_U_EDGE)
    ap.add_argument("--n-beta", type=int, default=N_BETA_EDGE)
    args = ap.parse_args(argv)
    root = Path(args.output_root)

    if args.recertify:
        bad = recertify(root, n_u=args.n_u, n_beta=args.n_beta)
        if bad:
            print("RECERTIFY DISAGREES:")
            for line in bad:
                print("  " + line)
            return 1
        print("recertify: every shipped entry reproduces its own catalog row")
        return 0

    if not args.sweep:
        ap.error("nothing to do: pass --sweep or --recertify")

    t0 = time.perf_counter()
    entries, ledger, wall = sweep(root, n_u=args.n_u, n_beta=args.n_beta)
    doc = catalog_document(entries, ledger, wall)
    (root / CATALOG_NAME).write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"\n{len(entries)} entries, "
          f"{sum(1 for e in entries if e['certified'])} certified, "
          f"solver {wall:.2f} s, total {time.perf_counter() - t0:.1f} s")
    print(f"catalog -> {root / CATALOG_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

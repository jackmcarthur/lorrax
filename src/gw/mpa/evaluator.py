"""The complex-frequency W evaluator: the MPA fallback correctness path.

STAGING LOCATION; the minimax-service design decides the final home.
Landed ahead of that review because the method is settled by
``~/MPA_THEORY_PLAN.md`` section B ("Evaluation") and because the fit
kernel that consumes it is already here.

THE KERNEL, AND WHY IT IS ONE KERNEL
------------------------------------
Theory-plan section B fixes the exact kernel of the damped-tau sweep::

    K_z(Delta) = -2 * integral_0^inf dt e^{-varpi t} e^{i omega t}
                                        sin(Delta t)

Write ``z = omega + i*varpi`` and the damping and the phase are one
factor, ``e^{-varpi t} e^{i omega t} = e^{i z t}``, which is the form
this module uses everywhere::

    K_z(Delta) = -2 * integral_0^inf dt e^{i z t} sin(Delta t)

CLOSED FORM.  For ``Im z > 0`` the integral converges absolutely and
is elementary.  Using ``sin(Delta t) = (e^{i Delta t} - e^{-i Delta
t}) / 2i`` and ``integral_0^inf e^{i s t} dt = i/s`` for ``Im s > 0``::

    integral_0^inf e^{i z t} sin(Delta t) dt
        = (1/2i) [ i/(z+Delta) - i/(z-Delta) ]
        = (1/2) [ 1/(z+Delta) - 1/(z-Delta) ]
        = Delta / (Delta**2 - z**2)

so::

    K_z(Delta) = -2 * Delta / (Delta**2 - z**2)
               = 1/(z - Delta) - 1/(z + Delta)

which is the resonant-plus-antiresonant spectral kernel of chi0 at
complex ``z``, i.e. exactly the object the whole pipeline has always
been evaluating -- at ``z = 0``, at ``z = i*varpi``, at ``z = omega``.
``gw.mpa.sample_plan`` records that identity as the 2x2 table; this
module is the machine for the one cell nothing else evaluates.

BOTH PATHS ARE KEPT, ON PURPOSE
-------------------------------
``damped_kernel`` is the closed form and ``damped_line_rule`` +
``evaluate_damped_points`` are the quadrature.  They are not
alternatives:

* the closed form is the UNIT-TEST ORACLE.  It is exact to rounding
  and it costs one divide, so every claim about the quadrature is
  checked against it rather than against a converged version of
  itself;
* the quadrature is what GENERALISES.  In production the per-node
  quantity is not ``sum_j g_j sin(Delta_j t_l)`` computed from a
  materialised spectrum -- it is a chi0 build at time node ``t_l``, and
  no closed form exists for it.  The quadrature path is the one whose
  internals become that build (see ``evaluate_samples``' ``sweep_fn``
  seam); the closed form can only ever be a test.

THE RULE: POSITIVE COMPOSITE GAUSS-LEGENDRE
-------------------------------------------
Theory-plan section B, "The rule formulation campaign", is explicit
that the certified bounded-amplification global rule is OFFLINE
campaign work, and that "positive composite quadrature is the
stability oracle and fallback -- a pilot measured it at roughly 4-5*A
nodes (about 400/500/800 at A = 80/120/200), stable but not cheap".
This module builds that fallback and nothing beyond it: it is
admissible today, without the certificate, because positivity is what
bounds the amplification.

WHY POSITIVITY IS THE WHOLE POINT.  The projection weights are
``w_l(z) = -2 h_l e^{i z t_l}`` with ``h_l > 0``, so

    sum_l |w_l(z)| = 2 sum_l h_l e^{-varpi t_l}
                   <= 2 integral_0^inf e^{-varpi t} dt = 2/varpi

for every ``omega`` on the line, with the bound INDEPENDENT of the node
count.  ``2/varpi`` is the exact ``L1`` mass of the continuous kernel
weight, so the amplification ratio ``kappa0`` reported by
``damped_line_rule`` is at most 1 and approaches it from below.  A
signed rule buys nodes with cancellation and gives that up; this is
why the campaign's target is *bounded-total-variation* minimax and why
the oracle is *positive* composite quadrature.

PANELS, NOT A GLOBAL RULE -- the documented choice.  The integrand
``e^{i z t} sin(Delta t)`` is an analytic oscillation under an
exponential envelope: the oscillation does not slow down as ``t``
grows but the envelope kills it.  So the rule is built as

* a TRUNCATION at ``t_max`` set by the envelope, and
* uniform PANELS across ``[0, t_max]`` whose width is a fixed number
  of wavelengths of the fastest component, each carrying its own
  Gauss-Legendre rule whose ORDER IS GRADED by the local envelope --
  early panels are resolved to the full tolerance, late panels only to
  the tolerance their amplitude can still violate.

A single global Gauss-Legendre rule was rejected because its order
would have to resolve the fastest oscillation over the whole
``[0, t_max]`` at the accuracy the FIRST panel needs, which is the
grading this rule spends its cheapness on; a double-exponential rule
was rejected because it is a rule for endpoint singularities, which
this integrand does not have.  Both rejections are node-count
arguments, and the node-count table in ``tests/test_mpa_evaluator.py``
is what backs them.

THE NODE COUNT, MEASURED (``tests/test_mpa_evaluator.py``, the
convergence table).  With ``A = freq_max / varpi`` the dimensionless
bandwidth and ``rel_tol`` the requested error relative to the kernel's
own ``1/varpi`` scale::

    rel_tol     A=20        A=80        A=200
    1e-6        110 (5.5A)  414 (5.2A)  1017 (5.1A)
    1e-8        150 (7.5A)  569 (7.1A)  1401 (7.0A)
    1e-10       193 (9.7A)  724 (9.1A)  1797 (9.0A)
    1e-12       238 (11.9A) 895 (11.2A) 2213 (11.1A)

and the measured worst error comes out at 0.51 to 0.69 of the
requested budget ``rel_tol/varpi`` at every one of the twelve entries
-- the rule delivers what it asks for, slightly better, uniformly in
``A``.  Two things to read off it.  The ``1e-6`` row IS the plan's
pilot -- 414 nodes at A=80 against its "about 400" -- so the "4-5*A"
figure is the rule at a 1e-6 tolerance, not the rule as such.  And
``N/A`` grows like ``ln(1/rel_tol)/4``, which is the Gauss-Legendre
resolution floor ``t_max * freq_max / 4`` with ``t_max`` set by the
envelope: the tolerance buys time range, and the time range buys
oscillations that have to be resolved.  ``DEFAULT_REL_TOL`` is 1e-8,
the middle of the table -- two decades tighter than the pilot's tier
for 1.4x its nodes.

WHAT THIS MODULE IS NOT.  It is not line-batched in the physical
sense.  ``evaluate_samples`` will share ONE rule and ONE sweep across a
line when asked (``batching='per-line'``), which is the arithmetic of
theory-plan section B's "common damping and common nodes per line,
scalar per-point projections"; but the sweep it shares is a spectrum
sum, not a chi0 build. The default is ``'per-point'``, which is the
plan's named correctness fallback -- "independent per-point sweeps are
the correctness fallback" -- and the two agree to the rule's tolerance,
which is a test.
"""

import numpy as np

import jax
import jax.numpy as jnp

from gw.mpa import sample_plan
from gw.minimax_config import MinimaxConfig
from gw.minimax_screening import (build_real_quadrature,
                                  solve_laplace_minimax_imag_interval,
                                  solve_laplace_minimax_interval)

#: Requested error of the damped rule, RELATIVE to the kernel's own
#: ``1/varpi`` scale.  See the node-count table in the module
#: docstring for why this row and not another.
DEFAULT_REL_TOL = 1.0e-8

#: Panel width, in wavelengths of the fastest component of the
#: integrand.  Wider panels amortise the per-panel accuracy overhead
#: over more oscillations (the Gauss-Legendre order per panel is
#: ``~pi/2`` per wavelength PLUS a tolerance-driven constant, so the
#: constant is what wide panels dilute) at the price of a higher
#: per-panel order.  16 was measured as the knee, at rel_tol 1e-8 and
#: A = 80: 4 costs 10.50*A, 8 costs 8.39*A, 16 costs 7.11*A, 32 costs
#: 6.29*A with the Gauss-Legendre order doubled to 57-68, and 64 costs
#: 5.80*A at order ~110.  Below 16 the constant dominates; above it,
#: the return is 10% a doubling.
DEFAULT_WAVELENGTHS_PER_PANEL = 16.0

#: Target error handed to the SHIPPED families on the existing-kernel
#: routes.  1e-10 and not tighter because the ``exponential_sum_imag``
#: family has zero shipped tables (DESIGN_minimax R4 / R1) and solves
#: at runtime: 1e-10 returns in milliseconds, 1e-12 measured 96 s.
DEFAULT_KERNEL_TARGET_ERROR = 1.0e-10


def _require_x64():
    """Refuse to run in single precision.  Mirrors ``pade_fit``.

    Every number here is a complex128 resolvent evaluated near its own
    pole; in float32 the strip samples lose the imaginary part that
    makes them well posed.
    """

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "GATE x64_enabled: jax x64 is off, so the evaluator would "
            "run in single precision. FALSE case: "
            "jax.config.update('jax_enable_x64', True) before import, "
            "or JAX_ENABLE_X64=1 in the environment.")


# ---------------------------------------------------------------------------
# The closed form -- the oracle
# ---------------------------------------------------------------------------

def damped_kernel(z, delta):
    """``K_z(Delta) = -2 Delta / (Delta**2 - z**2)``.  Host-side numpy.

    The exact value of theory-plan section B's damped-tau integral; see
    the module docstring for the derivation.  Broadcasts ``z`` against
    ``delta``: pass ``z`` with a trailing axis of ones to get the full
    ``(n_z, n_delta)`` table.

    Refuses a ``(z, Delta)`` pair that sits on the kernel's own pole,
    because the number there is infinite and a caller who meant to
    sample there has a geometry bug rather than a numerical one.
    """

    zc = np.asarray(z, dtype=np.complex128)
    d = np.asarray(delta, dtype=np.float64)
    denom = d ** 2 - zc ** 2
    if np.any(denom == 0.0):
        raise ValueError(
            "GATE off_kernel_pole: a sample z coincides with +/- a "
            "transition energy Delta, where K_z(Delta) is infinite. "
            "FALSE case: no sample point equals +/- any transition "
            "energy -- which every sample with varpi > 0 satisfies "
            "automatically.")
    return sample_plan.KERNEL_FACTOR * d / denom


# ---------------------------------------------------------------------------
# The composite rule
# ---------------------------------------------------------------------------

def _gauss_legendre_panel_bound(order, half_bandwidth, width, envelope):
    """Bernstein-ellipse error bound for one Gauss-Legendre panel.

    ``half_bandwidth`` is ``a = (freq_max + varpi) * width / 4``, the
    panel's oscillation content in the Bernstein parameter: mapping the
    panel to ``[-1, 1]`` turns ``e^{i F t}`` into ``e^{i a' x}`` with
    ``a' = F*width/2``, and on the Bernstein ellipse ``E_rho`` that
    factor is bounded by ``exp(a'*(rho - 1/rho)/2)``, i.e. by
    ``exp(a*(rho - 1/rho))`` with ``a = a'/2``.  The envelope's own
    analytic growth off the real axis contributes the ``+ varpi``.

    With Trefethen's ``|E| <= (64/15) M_rho / (rho^{2n} - 1)`` for the
    ``[-1,1]`` rule, and the panel's Jacobian ``width/2`` and local
    envelope amplitude carried through::

        bound(rho) = (64/15) (width/2) envelope
                     exp(a (rho - 1/rho)) / (1 - rho^{-2n})
                     / rho^{2n}

    which is minimised at ``rho* = (n + sqrt(n^2 - a^2)) / a``, real
    only for ``n >= a``.  That threshold is the resolution floor --
    ``pi/2`` nodes per wavelength -- and it is why no tolerance can buy
    a rule below ``t_max * freq_max / 4`` nodes.

    Returns ``inf`` below the floor.  The bound is an UPPER bound and a
    loose one, which is the right side to be wrong on for a stability
    oracle; the measured error of the assembled rule is set by the
    truncation, not by this term.
    """

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


def damped_line_rule(
    varpi,
    freq_max,
    *,
    rel_tol=DEFAULT_REL_TOL,
    wavelengths_per_panel=DEFAULT_WAVELENGTHS_PER_PANEL,
    max_order=256,
):
    """Nodes and weights of the positive composite rule for one line.

    Parameters
    ----------
    varpi
        The line height, ``Im z``, in the caller's energy unit.  The
        rule's time nodes come out in the reciprocal unit.  Must be
        strictly positive: ``varpi = 0`` is the ``sine_sum`` cell of
        the sampling table, not this one, and the integral does not
        converge absolutely there.
    freq_max
        The largest ANGULAR FREQUENCY the integrand will carry, which
        is ``max|omega| + max Delta`` over the points and the spectrum
        the rule will serve -- NOT ``max Delta`` alone.  The product
        ``e^{i omega t} sin(Delta t)`` beats at ``omega +/- Delta``, so
        a rule sized on the transitions alone under-resolves every
        sample with a real part by up to a factor two.  On the MPA
        protocol ``omega_m`` IS the top transition, so ``freq_max`` is
        ``2*Delta_max`` on the near line's last point: the dimensionless
        bandwidth of the plan's "A ~ 80 for Si-like spans" is this
        quantity over ``varpi``, and ``rule['a_dim']`` reports it.
    rel_tol
        Requested error RELATIVE to ``1/varpi``, the kernel's own peak
        scale (``|K_z|`` reaches ``~1/varpi`` where a transition sits
        under a sample).  Half the budget goes to the truncation and
        half to the panels.
    wavelengths_per_panel
        Panel width in wavelengths of ``freq_max``.
    max_order
        Refusal ceiling on the per-panel Gauss-Legendre order.

    Returns
    -------
    dict
        ``t`` and ``h``, the ``(n_nodes,)`` float64 nodes and strictly
        positive weights; ``t_max``; ``n_panels``; ``orders``, the
        per-panel order (this is the grading, and it is worth printing
        -- it is what makes the rule cheaper than a global one);
        ``a_dim``; ``kappa0``, the measured amplification ratio
        ``varpi * sum_l h_l e^{-varpi t_l}``, which is ``<= 1`` by
        positivity and is the rule's stability certificate; and the
        parameters it was built from.
    """

    v = float(varpi)
    f = float(freq_max)
    tol = float(rel_tol)
    wl = float(wavelengths_per_panel)
    if not (v > 0.0) or not np.isfinite(v):
        raise ValueError(
            f"GATE damped_line_positive_varpi: varpi={varpi!r} is not "
            "a finite positive line height. FALSE case: varpi > 0 -- "
            "varpi = 0 is the sine_sum cell of the sampling table "
            "(gw.mpa.sample_plan.FAMILIES), where the damped-tau "
            "integral does not converge absolutely and this rule does "
            "not apply.")
    if not (f > 0.0) or not np.isfinite(f):
        raise ValueError(
            f"GATE damped_line_positive_bandwidth: freq_max={freq_max!r}"
            " is not a finite positive angular frequency. FALSE case: "
            "freq_max = max|omega| + max Delta > 0.")
    if not (0.0 < tol < 1.0):
        raise ValueError(
            f"GATE damped_line_tolerance: rel_tol={rel_tol!r} is not in "
            "(0, 1). FALSE case: 0 < rel_tol < 1, an error relative to "
            "the 1/varpi scale of the kernel.")
    if not (wl > 0.0) or not np.isfinite(wl):
        raise ValueError(
            f"GATE damped_line_panel_width: wavelengths_per_panel="
            f"{wavelengths_per_panel!r} is not a finite positive count "
            "of wavelengths. FALSE case: wavelengths_per_panel > 0.")

    # Truncation: |tail| <= 2 exp(-varpi t_max)/varpi, and it gets half
    # of the absolute budget rel_tol/varpi.
    t_max = np.log(2.0 / tol) / v

    # Panels: uniform, a whole number of them, width close to the
    # requested wavelength count.
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
                f"GATE damped_line_panel_order: panel {k} of "
                f"{n_panels} needs a Gauss-Legendre order above "
                f"max_order={max_order} to reach {eps_panel:.3e} at "
                f"half-bandwidth {half_bandwidth:.2f}. FALSE case: the "
                "per-panel order fits under max_order -- reduce "
                "wavelengths_per_panel (which lowers the per-panel "
                "bandwidth) or loosen rel_tol.")
        x, w = np.polynomial.legendre.leggauss(order)
        nodes.append(left + 0.5 * width * (x + 1.0))
        weights.append(0.5 * width * w)
        orders.append(order)

    t = np.concatenate(nodes).astype(np.float64)
    h = np.concatenate(weights).astype(np.float64)
    if not np.all(h > 0.0):
        raise AssertionError(          # pragma: no cover
            "Gauss-Legendre weights must be positive; positivity is "
            "the rule's stability argument.")
    return {
        "t": t,
        "h": h,
        "n_nodes": int(t.size),
        "n_panels": int(n_panels),
        "orders": tuple(orders),
        "t_max": float(t_max),
        "panel_width": float(width),
        "varpi": v,
        "freq_max": f,
        "a_dim": float(f / v),
        "rel_tol": tol,
        "kappa0": float(v * np.sum(h * np.exp(-v * t))),
    }


def damped_rectangle_rule(
    gamma_min,
    gamma_max,
    freq_max,
    *,
    rel_tol=DEFAULT_REL_TOL,
    wavelengths_per_panel=DEFAULT_WAVELENGTHS_PER_PANEL,
    max_order=256,
):
    r"""Positive real-time rule for a lower-half-plane rectangle.

    The rule approximates the causal Laplace identity uniformly for
    ``|Re d| <= freq_max`` and
    ``gamma_min <= -Im d <= gamma_max``.  Truncation is controlled by the
    weakest damping, while each Bernstein-ellipse panel bound uses the
    largest imaginary excursion.  This is the width-aware counterpart of
    :func:`damped_line_rule`; using a one-line rule at ``gamma_min`` for
    wider poles has no such panel-error guarantee.
    """
    g0, g1 = map(float, (gamma_min, gamma_max))
    f = float(freq_max)
    tol = float(rel_tol)
    if not (0.0 < g0 <= g1 < np.inf):
        raise ValueError(
            "gamma bounds must satisfy 0 < gamma_min <= gamma_max")
    if not (0.0 < f < np.inf) or not (0.0 < tol < 1.0):
        raise ValueError("freq_max must be positive and rel_tol in (0,1)")

    t_max = np.log(2.0 / tol) / g0
    panel_target = float(wavelengths_per_panel) * 2.0 * np.pi / f
    n_panels = max(1, int(np.ceil(t_max / panel_target)))
    width = t_max / n_panels
    half_bandwidth = (f + g1) * width / 4.0
    eps_panel = 0.5 * (tol / g0) / n_panels
    nodes, weights, orders = [], [], []
    for panel in range(n_panels):
        left = panel * width
        envelope = np.exp(-g0 * left)
        order = int(np.ceil(half_bandwidth)) + 1
        while order <= int(max_order) and _gauss_legendre_panel_bound(
                order, half_bandwidth, width, envelope) > eps_panel:
            order += 1
        if order > int(max_order):
            raise ValueError(
                f"damped rectangle panel {panel} needs order above "
                f"max_order={max_order}")
        x, w = np.polynomial.legendre.leggauss(order)
        nodes.append(left + 0.5 * width * (x + 1.0))
        weights.append(0.5 * width * w)
        orders.append(order)
    t = np.concatenate(nodes).astype(np.float64)
    h = np.concatenate(weights).astype(np.float64)
    return {
        "t": t,
        "h": h,
        "n_nodes": int(t.size),
        "n_panels": int(n_panels),
        "orders": tuple(orders),
        "t_max": float(t_max),
        "panel_width": float(width),
        "gamma_min": g0,
        "gamma_max": g1,
        "freq_max": f,
        "rel_tol": tol,
        "kappa0": float(g0 * np.sum(h * np.exp(-g0 * t))),
    }


def _damped_rectangle_boundary(g0, g1, f, n_x, n_gamma, *, midpoint):
    if midpoint:
        x = -f + (np.arange(int(n_x)) + 0.5) * 2.0 * f / int(n_x)
        fraction = (np.arange(int(n_gamma)) + 0.5) / int(n_gamma)
    else:
        x = np.linspace(-f, f, int(n_x))
        fraction = np.linspace(0.0, 1.0, int(n_gamma))
    gamma = (np.asarray([g0]) if g1 == g0 else
             np.exp(np.log(g0) + fraction * np.log(g1 / g0)))
    return np.unique(np.r_[g0 - 1j * x, g1 - 1j * x,
                           gamma + 1j * f, gamma - 1j * f])


def _damped_rectangle_score(t, h, boundary):
    worst = 0.0
    for left in range(0, boundary.size, 2048):
        z = boundary[left:left + 2048]
        residual = 1.0 - (
            z[:, None] * np.exp(-z[:, None] * t[None, :])) @ h
        worst = max(worst, float(np.max(np.abs(residual))))
    return worst


def damped_rectangle_gauss_rule(
    gamma_min,
    gamma_max,
    freq_max,
    *,
    rel_tol=DEFAULT_REL_TOL,
    max_nodes=500,
):
    r"""Positive global Gauss rule for a damping rectangle.

    This has the same causal integral and positivity certificate as
    :func:`damped_rectangle_rule`, but selects the order of one global
    Gauss--Legendre rule by the relative residual

    ``1 - z * sum(h * exp(-z*t))``, ``z = gamma - 1j*x``.

    The residual is analytic inside the rectangle, so its continuum maximum
    lies on the boundary.  The returned error remains sampled evidence: the
    four edges are checked on a dense half-cell-shifted grid disjoint from the
    fit grid.  ``continuum_error_bound`` adds the conservative derivative
    cover bound; it is reported, not silently substituted for the requested
    sampled metric.
    """
    g0, g1 = map(float, (gamma_min, gamma_max))
    f = float(freq_max)
    tol = float(rel_tol)
    cap = int(max_nodes)
    if not (0.0 < g0 <= g1 < np.inf):
        raise ValueError(
            "gamma bounds must satisfy 0 < gamma_min <= gamma_max")
    if not (0.0 <= f < np.inf) or not (0.0 < tol < 1.0):
        raise ValueError("freq_max must be nonnegative and rel_tol in (0,1)")
    if cap < 2:
        raise ValueError("max_nodes must be at least two")

    fit_boundary = _damped_rectangle_boundary(
        g0, g1, f, 2049, 513, midpoint=False)
    check_boundary = _damped_rectangle_boundary(
        g0, g1, f, 8192, 2048, midpoint=True)

    legendre = {}

    def _rule(order, t_max):
        if int(order) not in legendre:
            legendre[int(order)] = np.polynomial.legendre.leggauss(int(order))
        x, h = legendre[int(order)]
        return 0.5 * t_max * (x + 1.0), 0.5 * t_max * h

    def _score(order, t_max, boundary):
        return _damped_rectangle_score(*_rule(order, t_max), boundary)

    # Only f is oscillatory.  gamma_max creates a short-time boundary layer,
    # so using it in the time-bandwidth seed can over-rank a rule by orders of
    # magnitude.  Gauss error is not monotone in the order; scan the final
    # bracket rather than binary-searching an assumption the family lacks.
    # The usual t_max=log(2/tol)/g0 reserves half of the error for the
    # omitted tail.  That split is convenient, not optimal.  Search a small
    # deterministic family of tail allocations and score the complete
    # resolvent residual, which includes truncation and quadrature together.
    # Positivity and the literal physical eta are unchanged.
    best = None
    starts = []
    for tail_fraction in (0.50, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95):
        t_max = np.log(1.0 / (tail_fraction * tol)) / g0
        start = min(cap, max(2, int(np.floor(f * t_max / 4.0))))
        starts.append(start)
        stop = cap if best is None else min(cap, best[0])
        for candidate in range(start, stop + 1):
            if _score(candidate, t_max, fit_boundary) > tol:
                continue
            heldout = _score(candidate, t_max, check_boundary)
            if heldout <= tol:
                row = (candidate, heldout, t_max, tail_fraction)
                if best is None or row[:2] < best[:2]:
                    best = row
                break
    if best is None:
        raise RuntimeError(
            f"global damped rectangle rule did not reach {tol:g} from "
            f"oscillatory seeds {min(starts)}-{max(starts)} through "
            f"max_nodes={cap}")
    order, sampled_error, t_max, tail_fraction = best
    t, h = _rule(order, t_max)
    z_max = float(np.hypot(g1, f))
    derivative_bound = float(np.sum(
        h * np.exp(-g0 * t) * (1.0 + z_max * t)))
    x_cover = f / 8192.0
    if g1 == g0:
        gamma_cover = 0.0
    else:
        gamma_mid = np.exp(
            np.log(g0)
            + (np.arange(2048) + 0.5) * np.log(g1 / g0) / 2048.0)
        gamma_cover = max(
            float(gamma_mid[0] - g0), float(g1 - gamma_mid[-1]),
            0.5 * float(np.max(np.diff(gamma_mid))))
    continuum_bound = sampled_error + derivative_bound * max(
        x_cover, gamma_cover)
    return {
        "t": t.astype(np.float64),
        "h": h.astype(np.float64),
        "n_nodes": int(order),
        "n_panels": 1,
        "orders": (int(order),),
        "t_max": float(t_max),
        "tail_budget_fraction": float(tail_fraction),
        "gamma_min": g0,
        "gamma_max": g1,
        "freq_max": f,
        "rel_tol": tol,
        "sampled_max_error": float(sampled_error),
        "continuum_error_bound": float(continuum_bound),
        "kappa0": float(g0 * np.sum(h * np.exp(-g0 * t))),
        "rule_type": "global_gauss",
    }


def damped_rectangle_positive_rule(
    gamma_min,
    gamma_max,
    freq_max,
    *,
    rel_tol=DEFAULT_REL_TOL,
    max_nodes=500,
):
    """Choose the smaller passing positive rectangle rule.

    The global rule is normally smaller.  The panelled rule is an independent
    conservative fallback while the global family is certified numerically.
    Both candidates are scored on the same disjoint dense boundary grid.
    """
    g0, g1 = map(float, (gamma_min, gamma_max))
    f, tol = float(freq_max), float(rel_tol)
    candidates = []
    try:
        candidates.append(damped_rectangle_gauss_rule(
            g0, g1, f, rel_tol=tol, max_nodes=max_nodes))
    except RuntimeError:
        pass

    if f == 0.0:
        if not candidates:
            raise RuntimeError(
                f"global zero-frequency rectangle rule did not reach "
                f"{tol:g} through max_nodes={int(max_nodes)}")
        return min(candidates, key=lambda rule: (
            rule["n_nodes"], rule["sampled_max_error"]))

    try:
        panelled = damped_rectangle_rule(g0, g1, f, rel_tol=tol)
    except (RuntimeError, ValueError):
        panelled = None
    boundary = _damped_rectangle_boundary(
        g0, g1, f, 8192, 2048, midpoint=True)
    worst = np.inf
    if panelled is not None and panelled["n_nodes"] <= int(max_nodes):
        worst = _damped_rectangle_score(
            panelled["t"], panelled["h"], boundary)
    if panelled is not None and worst <= tol:
        panelled = dict(panelled)
        panelled.update(sampled_max_error=worst,
                        continuum_error_bound=np.nan,
                        tail_budget_fraction=0.5,
                        rule_type="panelled_gauss")
        candidates.append(panelled)
    if not candidates:
        raise RuntimeError(
            f"no positive damped rectangle rule reached {tol:g} through "
            f"max_nodes={int(max_nodes)}")
    return min(candidates, key=lambda rule: (rule["n_nodes"],
                                             rule["sampled_max_error"]))


def damped_line_projection(rule, z):
    """The per-point scalar weights ``w_l(z) = -2 h_l e^{i z t_l}``.

    This is the ONLY thing that depends on the sample point: the nodes,
    the weights and the sweep over them are shared by every point on
    the line.  Theory-plan section B's "common damping and common nodes
    per line, scalar per-point projections" is this function returning
    a vector whose only ``z`` dependence is a phase times the line's
    common ``e^{-varpi t}``.
    """

    zc = complex(z)
    if abs(zc.imag - rule["varpi"]) > 1.0e-12 * max(rule["varpi"], 1.0):
        raise ValueError(
            f"GATE projection_on_line: z={z!r} has varpi={zc.imag!r} "
            f"but the rule was built for varpi={rule['varpi']!r}. "
            "FALSE case: the point lies on the line whose rule it is "
            "projected against -- the damping is COMMON to the line "
            "and only the phase is per point.")
    t = jnp.asarray(rule["t"], dtype=jnp.float64)
    h = jnp.asarray(rule["h"], dtype=jnp.float64)
    phase = jnp.exp(1j * jnp.asarray(zc, dtype=jnp.complex128) * t)
    return sample_plan.KERNEL_FACTOR * h.astype(jnp.complex128) * phase


def sine_sweep_from_spectrum(t, delta, weight):
    """``S_l = sum_j g_j sin(Delta_j t_l)``.  The stand-in sweep.

    THIS FUNCTION IS THE SEAM.  In production the per-node quantity is
    a chi0 build at time node ``t_l`` -- ``G(t) G(-t)`` contracted into
    the ISDF basis -- and its spectral content is exactly this sum with
    the transition weights of the band pair.  Here the spectrum is
    materialised because the point is to have an oracle; the
    production evaluator passes its own callable of the same shape
    through ``evaluate_samples(..., sine_sweep_fn=...)`` and nothing
    above this line changes.

    ``weight`` may carry trailing axes -- ``(n_delta,)`` for a scalar
    channel, ``(n_delta, n_chan)`` for a tile of ISDF elements -- and
    the result is ``(n_nodes,) + weight.shape[1:]``.
    """

    tt = jnp.asarray(t, dtype=jnp.float64)
    d = jnp.asarray(delta, dtype=jnp.float64)
    g = jnp.asarray(weight, dtype=jnp.float64)
    if d.ndim != 1 or g.shape[0] != d.shape[0]:
        raise ValueError(
            f"GATE spectrum_shapes: delta has shape {d.shape} and "
            f"weight has shape {g.shape}. FALSE case: delta is "
            "(n_delta,) and weight is (n_delta,) + channel_shape.")
    basis = jnp.sin(tt[:, None] * d[None, :])
    return jnp.tensordot(basis, g, axes=(1, 0))


def exp_sweep_from_spectrum(tau, delta, weight):
    """``E_l = sum_j g_j exp(-tau_l Delta_j)``.  The other seam.

    The existing-kernel routes are exponential sums in ``Delta``, so
    their per-node quantity is a Laplace-domain sweep and not a sine
    one -- see ``evaluate_samples`` for what that difference costs the
    unification.  Same shape contract as
    ``sine_sweep_from_spectrum``.
    """

    tt = jnp.asarray(tau, dtype=jnp.float64)
    d = jnp.asarray(delta, dtype=jnp.float64)
    g = jnp.asarray(weight, dtype=jnp.float64)
    if d.ndim != 1 or g.shape[0] != d.shape[0]:
        raise ValueError(
            f"GATE spectrum_shapes: delta has shape {d.shape} and "
            f"weight has shape {g.shape}. FALSE case: delta is "
            "(n_delta,) and weight is (n_delta,) + channel_shape.")
    basis = jnp.exp(-tt[:, None] * d[None, :])
    return jnp.tensordot(basis, g, axes=(1, 0))


def evaluate_damped_points(rule, z_values, sweep):
    """Project one line's sweep onto its points.  ``(n_z,) + chan``.

    The whole line rides one ``sweep``; each point costs one complex
    scalar reduction over the nodes.
    """

    s = jnp.asarray(sweep)
    out = [jnp.tensordot(damped_line_projection(rule, z), s, axes=(0, 0))
           for z in z_values]
    return jnp.stack(out, axis=0)


# ---------------------------------------------------------------------------
# The existing-kernel routes -- the three cells that already ship
# ---------------------------------------------------------------------------

def existing_kernel_rule(point, *, delta_min, delta_max,
                         target_error=DEFAULT_KERNEL_TARGET_ERROR,
                         max_nodes=64, use_shipped_tables=True):
    """Build the shipped quadrature serving one non-strip point.

    THE CONTRACT ADAPTATION, STATED ONCE.  The shipped families are
    exponential sums IN ``Delta`` -- ``sum_l alpha_l e^{-tau_l
    Delta}`` -- and they are solved on a bounded POSITIVE INTERVAL
    ``[delta_min, delta_max]``, whose ratio ``R`` is the parameter the
    catalog is indexed by.  The damped rule is a sine sum IN ``t`` and
    needs no ``delta_min`` at all, only ``varpi`` and a bandwidth.  So
    the sampling object cannot carry one "domain" field: it carries
    ``domain = 'interval'`` for the three shipped cells and
    ``domain = 'bandwidth'`` for the strip
    (``sample_plan.FAMILIES``), and this function is where the
    interval requirement is actually imposed.  A caller with a gapless
    spectrum (metals, intraband transitions down to zero) has an
    ``R -> inf`` problem on the shipped routes and no problem at all on
    the strip -- which is a reason the protocol puts the metal origin
    sample at ``i*1e-5 Ha`` rather than at 0.

    THE SECOND ADAPTATION, on the ``sine_sum`` cell.  Its shipped
    route (``build_real_quadrature``) decomposes
    ``Delta/(Delta**2-omega**2)`` into two ``1/y`` minimaxes and
    returns SIGNED ``tau``: positive on the ``(omega+Delta)`` branch,
    negative on the ``(omega-Delta)`` branch.  So a "time node" of that
    family is not a time at all, and ``exp_sweep_fn`` must tolerate
    ``exp(+|tau| Delta)``.  It does -- ``|tau| ~ 1/omega`` there, so
    the growing branch is harmless -- but the sampling object cannot
    promise that node vectors are nonnegative, and this is why.

    The returned dict is deliberately the same shape as
    ``damped_line_rule``'s: ``t``/``h`` are the nodes and weights to
    sweep and project, ``n_nodes`` is what the cost report counts. The
    ``KERNEL_FACTOR`` of ``-2`` is NOT folded into ``h`` here; it is
    applied once, by ``evaluate_samples``, for both routes.
    """

    fam = point["family"]
    if fam == "damped_line":
        raise ValueError(
            f"GATE existing_kernel_cell: point {point['role']!r} at "
            f"z={point['z']!r} is a strip sample, which has no shipped "
            "family. FALSE case: the point's family is one of "
            "exponential_sum, exponential_sum_imag, sine_sum -- the "
            "strip goes through damped_line_rule.")
    lo = float(delta_min)
    hi = float(delta_max)
    if not (0.0 < lo < hi) or not np.isfinite(hi):
        raise ValueError(
            f"GATE existing_kernel_interval: (delta_min, delta_max)="
            f"({delta_min!r}, {delta_max!r}) is not a bounded positive "
            "transition interval. FALSE case: 0 < delta_min < "
            "delta_max < inf -- the shipped families are solved on "
            "[delta_min, delta_max] and are indexed by its ratio.")
    if fam == "exponential_sum":
        quad = solve_laplace_minimax_interval(
            lo, hi, target_error=float(target_error),
            max_nodes=int(max_nodes),
            use_shipped_tables=bool(use_shipped_tables))
    elif fam == "exponential_sum_imag":
        # ``varpi`` is a sampling line height over the transition floor,
        # so this request carries the height clause of the beta envelope
        # (GATE0_IMAG_ENVELOPE.md sec 2).  At the fit stage's 1e-12 tier
        # nothing is tabulated yet, so the selector refuses by name and
        # this falls through to the runtime solve exactly as before.
        quad = solve_laplace_minimax_imag_interval(
            lo, hi, point["varpi"], target_error=float(target_error),
            max_nodes=int(max_nodes),
            use_shipped_tables=bool(use_shipped_tables))
    else:
        # sine_sum.  build_real_quadrature shifts an existing 1/y
        # minimax on [delta_min, delta_max] by +/- omega, so the
        # static solve is its input, not a duplicate of it.
        if point["omega"] <= hi:
            raise ValueError(
                f"GATE sine_sum_above_band: point {point['role']!r} "
                f"at z={point['z']!r} has omega={point['omega']!r} at "
                f"or below delta_max={hi!r}. The shipped real-axis "
                "route needs omega above every transition so both 1/y "
                "branches stay positive. FALSE case: omega > "
                "delta_max -- which sample_plan.refuse_unsupported "
                "checks before any physics runs.")
        base = solve_laplace_minimax_interval(
            lo, hi, target_error=float(target_error),
            max_nodes=int(max_nodes),
            use_shipped_tables=bool(use_shipped_tables))
        quad = build_real_quadrature(
            base, point["omega"],
            MinimaxConfig(target_error=float(target_error),
                          max_nodes=int(max_nodes)))
    return {
        "t": np.asarray(quad.tau, dtype=np.float64),
        "h": np.asarray(quad.alpha, dtype=np.float64),
        "n_nodes": int(quad.node_count),
        "family": fam,
        "max_error": float(quad.max_error),
        "delta_min": lo,
        "delta_max": hi,
        "target_error": float(target_error),
    }


# ---------------------------------------------------------------------------
# The per-point evaluation contract
# ---------------------------------------------------------------------------

def evaluate_samples(
    plan,
    delta,
    weight,
    *,
    batching="per-point",
    rel_tol=DEFAULT_REL_TOL,
    wavelengths_per_panel=DEFAULT_WAVELENGTHS_PER_PANEL,
    kernel_target_error=DEFAULT_KERNEL_TARGET_ERROR,
    sine_sweep_fn=sine_sweep_from_spectrum,
    exp_sweep_fn=exp_sweep_from_spectrum,
):
    """Evaluate every point of ``plan`` against a transition spectrum.

    THE SIGNATURE IS THE CONTRACT.  A caller hands over the plan and
    the spectrum and gets back one value per plan point, in plan order,
    plus a cost record.  Theory-plan section B's line-batched
    implementation is a change of ``sine_sweep_fn`` and of
    ``batching``, both of which are arguments; nothing a caller writes
    moves when the sweep stops being a spectrum sum and starts being a
    chi0 build.  That is what "the per-point fallback consumes the
    identical object with a different evaluator" has to mean in code.

    Parameters
    ----------
    plan
        A ``gw.mpa.sample_plan`` plan.  Its ``route`` column decides
        which machine serves each point; nothing here re-derives it.
    delta, weight
        The transition spectrum: ``(n_delta,)`` positive energies and
        ``(n_delta,) + channel_shape`` weights.  ``channel_shape`` is
        free, so a tile of ISDF elements evaluates in one call.
    batching
        ``'per-point'`` (default) gives every strip point its own rule,
        sized to its own ``|omega| + Delta_max`` -- the theory plan's
        named correctness fallback, "independent per-point sweeps".
        ``'per-line'`` builds one rule per line, sized to the line's
        widest point, and rides one sweep across it -- the arithmetic
        of the plan's line-batched design.  The two agree to the
        rule's tolerance; the cost record is where they differ.
    rel_tol, wavelengths_per_panel
        Forwarded to ``damped_line_rule``.
    kernel_target_error
        Forwarded to the shipped families.
    sine_sweep_fn, exp_sweep_fn
        The two seams.  ``sine_sweep_fn(t, delta, weight)`` returns
        ``(n_nodes,) + channel_shape``; ``exp_sweep_fn`` likewise for
        the Laplace nodes.

    Returns
    -------
    ``(values, cost)``
        ``values`` is ``(n_points,) + channel_shape`` complex128 in
        plan order -- feed it straight to ``pade_fit.fit_mpa_poles``
        against ``sample_plan.plan_z(plan)``.  ``cost`` is the record
        ``format_evaluator_cost_report`` prints.
    """

    _require_x64()
    if batching not in ("per-point", "per-line"):
        raise ValueError(
            f"GATE batching_known: batching={batching!r} is not a "
            "known evaluation mode. FALSE case: batching is "
            "'per-point' (the theory plan's correctness fallback) or "
            "'per-line' (its line-batched arithmetic).")
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 1 or d.size == 0 or not np.all(d > 0.0):
        raise ValueError(
            f"GATE spectrum_positive: delta has shape {d.shape} and "
            f"min {d.min() if d.size else float('nan')!r}. FALSE case: "
            "delta is a nonempty 1-D array of strictly positive "
            "transition energies.")
    delta_min = float(d.min())
    delta_max = float(d.max())
    sample_plan.refuse_unsupported(plan, delta_max=delta_max)

    points = sample_plan.plan_points(plan)
    routes = sample_plan.plan_routes(plan)
    g = jnp.asarray(weight)
    channel_shape = tuple(np.asarray(weight).shape[1:])
    values = [None] * len(points)

    # --- the three shipped cells: one exponential sweep per point.
    kernel_nodes = 0
    kernel_max_error = 0.0
    for p in routes["existing"]:
        rule = existing_kernel_rule(
            p, delta_min=delta_min, delta_max=delta_max,
            target_error=kernel_target_error)
        swept = exp_sweep_fn(rule["t"], d, g)
        values[p["index"]] = (
            sample_plan.KERNEL_FACTOR
            * jnp.tensordot(jnp.asarray(rule["h"],
                                        dtype=jnp.complex128),
                            jnp.asarray(swept, dtype=jnp.complex128),
                            axes=(0, 0)))
        kernel_nodes += rule["n_nodes"]
        kernel_max_error = max(kernel_max_error, rule["max_error"])

    # --- the strip: the damped-tau rule, shared per line or not.
    damped_nodes = 0
    damped_sweeps = 0
    line_rows = []
    for varpi, line_pts in routes["lines"]:
        if batching == "per-line":
            groups = [line_pts]
        else:
            groups = [(p,) for p in line_pts]
        line_nodes = 0
        worst_kappa0 = 0.0
        order_lo, order_hi = None, None
        for group in groups:
            freq_max = max(abs(p["omega"]) for p in group) + delta_max
            rule = damped_line_rule(
                varpi, freq_max, rel_tol=rel_tol,
                wavelengths_per_panel=wavelengths_per_panel)
            swept = sine_sweep_fn(rule["t"], d, g)
            got = evaluate_damped_points(
                rule, [p["z"] for p in group], swept)
            for k, p in enumerate(group):
                values[p["index"]] = got[k]
            line_nodes += rule["n_nodes"]
            damped_sweeps += 1
            worst_kappa0 = max(worst_kappa0, rule["kappa0"])
            lo_k, hi_k = min(rule["orders"]), max(rule["orders"])
            order_lo = lo_k if order_lo is None else min(order_lo, lo_k)
            order_hi = hi_k if order_hi is None else max(order_hi, hi_k)
        damped_nodes += line_nodes
        line_rows.append({
            "varpi": float(varpi),
            "n_points": len(line_pts),
            "n_sweeps": len(groups),
            "nodes": int(line_nodes),
            "a_dim": float(
                (max(abs(p["omega"]) for p in line_pts) + delta_max)
                / varpi),
            "kappa0": float(worst_kappa0),
            "orders": (int(order_lo), int(order_hi)),
        })

    out = jnp.stack([jnp.asarray(v, dtype=jnp.complex128)
                     for v in values], axis=0)

    # One GN sweep = the static minimax chi0 sweep over the same
    # transition interval at the same tolerance.  It is the unit the
    # theory plan asks the tau-node dispatches to be quoted in.
    gn_quad = solve_laplace_minimax_interval(
        delta_min, delta_max, target_error=kernel_target_error,
        max_nodes=64)
    gn_nodes = int(gn_quad.node_count)
    total_nodes = kernel_nodes + damped_nodes
    cost = {
        "label": plan["label"],
        "census": sample_plan.describe_plan(plan),
        "n_points": len(points),
        "logical_outputs": len(points),
        "channel_shape": channel_shape,
        "batching": batching,
        "delta_interval": (delta_min, delta_max),
        "n_delta": int(d.size),
        "kernel_nodes": int(kernel_nodes),
        "kernel_points": len(routes["existing"]),
        "kernel_max_error": float(kernel_max_error),
        "damped_nodes": int(damped_nodes),
        "damped_points": sum(len(pts) for _, pts in routes["lines"]),
        "damped_sweeps": int(damped_sweeps),
        "lines": tuple(line_rows),
        "node_dispatches": int(total_nodes),
        "gn_sweep_nodes": gn_nodes,
        "gn_sweeps": float(total_nodes) / float(gn_nodes),
        "rq_transforms": len(points),
        "dyson_solves": len(points),
        "rel_tol": float(rel_tol),
        "kernel_target_error": float(kernel_target_error),
    }
    return out, cost


def format_evaluator_cost_report(cost):
    """The evaluation stage's cost, as theory-plan section B demands.

    *"Cost reports must state 2n_p logical outputs, actual tau-node
    dispatches normalized to a GN sweep, and per-output R->q transforms
    and Dyson solves; a line-batched calculation is not 'one build'
    merely because it is one physical sweep."*

    So the node count is printed BESIDE the sweep count, and the
    per-line rows carry both -- ``per-line`` batching turns seven
    sweeps into one without changing the seven R->q transforms and
    seven Dyson solves each of those samples still costs, and the
    report has to make that impossible to misread.
    """

    c = cost
    lines = [
        "",
        "MPA evaluation stage - cost report",
        "-" * 64,
        f"  {c['census']}",
        f"  spectrum        {c['n_delta']} transitions on "
        f"[{c['delta_interval'][0]:.4g}, {c['delta_interval'][1]:.4g}]"
        f", channels {c['channel_shape'] or '()'}",
        f"  batching        {c['batching']}",
        f"  logical outputs {c['logical_outputs']} "
        f"(= 2*n_p samples of W_c)",
        f"  shipped cells   {c['kernel_points']} points, "
        f"{c['kernel_nodes']} nodes, worst family error "
        f"{c['kernel_max_error']:.3e}",
        f"  strip cell      {c['damped_points']} points, "
        f"{c['damped_nodes']} nodes over {c['damped_sweeps']} sweeps "
        f"(rel_tol {c['rel_tol']:.0e})",
    ]
    for row in c["lines"]:
        per_sweep = row["nodes"] / max(row["n_sweeps"], 1)
        lines.append(
            f"    line varpi={row['varpi']:<8.4g} A={row['a_dim']:7.1f} "
            f"{row['n_points']} points / {row['n_sweeps']} sweeps / "
            f"{row['nodes']} nodes "
            f"({per_sweep / max(row['a_dim'], 1e-30):.2f}*A per sweep, "
            f"GL orders {row['orders'][0]}-{row['orders'][1]}, kappa0 "
            f"{row['kappa0']:.4f})")
    lines += [
        f"  node dispatches {c['node_dispatches']} tau nodes "
        f"= {c['gn_sweeps']:.2f} GN sweeps "
        f"(one GN sweep = {c['gn_sweep_nodes']} static minimax nodes "
        f"on the same interval)",
        f"  per output      {c['rq_transforms']} R->q transforms, "
        f"{c['dyson_solves']} Dyson solves "
        "(one each per sample; line batching does not reduce these)",
        "-" * 64,
        "",
    ]
    return "\n".join(lines)

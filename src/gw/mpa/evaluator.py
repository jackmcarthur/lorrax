"""Complex-frequency chi kernels and positive causal quadrature rules.

The current mathematical contract and the distinction between the chi line
rule and the MPA Sigma crossing rule are owned by
``docs/theory/THEORY_mpa_implementation.md``.

For ``z = omega + i*varpi`` in the upper half-plane, the chi transition
kernel is

``K_z(Delta) = -2 integral_0^inf exp(i*z*t) sin(Delta*t) dt``

and its exact scalar oracle is

``K_z(Delta) = -2*Delta/(Delta**2-z**2)``.

Production cannot use the divide because one time node stands for the
separable occupied- and empty-band Green-function contractions.  The closed
form is retained only to test that the quadrature evaluates the right kernel.

``damped_line_rule`` serves a horizontal chi sampling line.  It truncates the
positive real-time integral, partitions it into wavelength-sized panels, and
grades a positive Gauss-Legendre order by the local exponential envelope.
Every sample on the line shares the nodes and positive weights; only its
scalar phase differs.  Positivity bounds the amplification independently of
the number of panels.

``damped_rectangle_positive_rule`` serves the MPA Sigma crossing core.  It
searches positive global Gauss-Legendre rules over the complete damping
rectangle, validates on a disjoint boundary grid, records a derivative-based
continuum cover, and compares against an independently panelled fallback.
Large ``gamma_max`` is treated as a short-time boundary layer, not an
oscillation frequency.  The two rules share a causal integral and positivity
argument, but they solve different domains and are not interchangeable.
"""

import numpy as np

from gw.mpa import sample_plan

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


# ---------------------------------------------------------------------------
# The closed form -- the oracle
# ---------------------------------------------------------------------------

def damped_kernel(z, delta):
    """``K_z(Delta) = -2 Delta / (Delta**2 - z**2)``.  Host-side numpy.

    The exact value of the authoritative MPA chapter's damped-tau integral; see
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

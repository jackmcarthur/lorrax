"""Where one MPA pole's Sigma contribution is integrated, and by which rule.

This is the routing half of the MPA self-energy: given one pole
``Omega_p = a_p - i*Gamma_p`` and the branch it is being summed on, it says
which window serves it, which certified quadrature that window asks for, and
what happens when the catalog cannot answer.  It computes no self-energy; the
kernel that does lives beside it and consumes what this module returns.

THE ONE SUBSTITUTION THAT DEFINES THE MODULE
--------------------------------------------
The two-point plasmon-pole self-energy regularizes its crossing denominator by
hand: ``ppm_windows`` carries a broadening ``xi`` and floors it so the HGL sine
rule stays conditioned.  A multipole pole carries its own width, so on this
path

    1 / (omega~ - E_A - a_p + i*Gamma_p)

is the EXACT object, ``Gamma_p`` is the only imaginary part in it, and adding
``i*delta`` or a fitted ``xi`` on top would count the same broadening twice.
There is no regularization left to choose (MPA_THEORY_PLAN section C).  Every
window edge that was ``edge_factor * xi`` is ``edge_factor * Gamma_p`` here,
which is the "padding proportional to Gamma" the plan's window predicate asks
for, and it is load-bearing in a way the audit below measures.

THE BRANCH PLACEMENT, WRITTEN OUT ONCE
--------------------------------------
``pade_fit`` returns poles in the closed fourth quadrant: ``Re Omega > 0``,
``Im Omega <= 0``, with ``|Im Omega| <= width_ratio_max * Re Omega``.  Given
that, and the tau-kernel's fixed expression

    sum_l alpha_l * exp(i*s*omega*t_l) * exp(-i(E_A - E_ref_A) t_l)
                  * exp(-i(Omega_p - E_ref_B) t_l)

the two placements are forced, and neither is a choice:

``CROSSING`` (denominator ``omega~ - S``, ``S = E_A + Omega_p``; the empty
    A-space on the ``+omega`` half and the occupied A-space on the ``-omega``
    half, exactly as the existing predicate encodes)
        REAL tau nodes, ``s = +1``, ``alpha_l = -i * h_l`` with ``h_l > 0``.
        The exponent is ``i*u*t - Gamma_p*t`` with ``u = omega~ - Re S``, so
        the sum is the exact complex Lorentzian resolvent

            1/(u + i*Gamma_p) = -i * int_0^inf dt e^{i u t} e^{-Gamma_p t},

        dispersive and absorptive parts together.  A sine-only kernel computes
        a different analytic object and is refused on this path.

``SIGN_DEFINITE`` (denominator ``omega~ + S``; the other two branches)
        IMAGINARY tau nodes ``t = -i*tau``, ``s = -1``, ``alpha_l = h_l``.
        The exponent is ``-w*tau + i*Gamma_p*tau`` with ``w = omega~ + Re S``,
        so the sum is ``1/(w - i*Gamma_p)``: in imaginary time the width is a
        PHASE, not extra decay, which is why the shipped real ``1/y`` tables do
        not serve this cell.

Both halves decay as ``e^{-Gamma|t|}`` and the crossing role swaps between the
``+omega`` and ``-omega`` halves; there is no global conjugation identity to
exploit.  Conjugating ``Omega_p`` or swapping the two placements each produces
a finite, plausible number, and each is refused by a red twin in
``tests/test_mpa_sigma_routing.py`` rather than by inspection.

WHAT THE POLE-FIELD AUDIT MEASURED, AND WHAT IT CHANGED
-------------------------------------------------------
Scored on the finalized first-light store (si_mpa_0808, ``Omega_p`` at
``(8, 8, 1128, 1128)``, 81 432 576 live poles, widths spanning 4e-5 eV to
4 keV), with the deck's own ``omega_max = 5 eV`` and ``edge_factor = 1.5``:

* The two CROSSING branches put 27.0 % of poles in the ``a_stripe`` Laplace
  window and 73.0 % in the ``b_slab``, and **not one pole of the 81 million
  asks for ``beta = Gamma_p/x_min`` above 2/3**.  That is structural rather
  than lucky: those windows floor ``x_min`` at ``z_edge =
  edge_factor*Gamma_p``, so ``beta <= 1/edge_factor`` identically.  The
  shipped width clause's ``beta_max = 2/3`` and the deck's
  ``edge_factor = 1.5`` are the same number,
  and :func:`refuse_edge_factor_below_envelope` makes that coincidence a
  checked precondition instead of a silent one -- 85-90 % of the ``|B|`` mass
  sits in the top bucket ``(1/3, 2/3]``, hard against the edge.

* The two SIGN-DEFINITE branches have no such floor:
  ``x_min = min(E_A) + a_p``, and ``min(E_A)`` is half the gap (0.3464 eV
  on this deck).  **11.52 % of
  poles, carrying 62.25 % of the ``|B|`` mass, ask for beta in (2/3, 1]** --
  outside the declared width clause.  The plan's claim that the largest widths
  sit deepest, keeping beta small, HOLDS on the crossing branches
  (corr(log Gamma, log x_min) = 0.27-0.77 per pole) and FAILS here
  (-0.10 to 0.59, and below 0.16 for the four shallowest poles, which are the
  ones carrying the plasmon).

* The ceiling of the whole field is ``beta = 1`` EXACTLY, and it is not a
  property of this deck: ``x_min >= Re Omega`` on the sign-definite branch and
  ``pade_fit``'s fourth guard caps ``|Im Omega| <= width_ratio_max * Re Omega``
  at ``width_ratio_max = 1.0``.  So the width clause any MPA pole field can ask
  for closes at ``beta_max = width_ratio_max``, which is the fitter's own
  number and not a fitted one.

* Against the catalog AS SHIPPED, the refusal count is not 11.52 % but ALL of
  them: ``beta_selector.CATALOG_CLAUSE`` stamps ``complex_laplace/v1`` as the
  HEIGHT clause end to end, so the width clause has zero tables and every MPA
  Laplace window refuses on an empty family.  The adjudication is therefore
  neither "generate a few more entries" nor "the routing needs a different x":
  the x convention is the only dimensionally admissible one (``x_min`` is what
  the catalog rescales by), and what is missing is the width-clause campaign
  itself, at ``beta_max = 1``.

* The crossing core's own composite bandwidth ``A = f_max/Gamma_p`` is benign
  for seven of the eight poles (median 1.5-3.9, p99 <= 10) and has a thin
  narrow-width tail on pole 0 (p99 = 111, max 2.7e6).  That tail is the corner
  :func:`crossing_rule` refuses in rather than building a rule with ten
  million nodes.

The histogram behind those numbers is
``/pscratch/sd/j/jackm/mpa_sigma_audit_0809/pole_envelope_audit.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CROSSING",
    "SIGN_DEFINITE",
    "WindowPlan",
    "RoutingRefusal",
    "beta_for_window",
    "crossing_rule",
    "denominator_can_cross",
    "exact_resolvent",
    "refuse_edge_factor_below_envelope",
    "route_pole",
    "sign_definite_rule",
]


#: The two placements.  Spelled as constants because "crossing" appears in
#: three vocabularies in this tree (the HGL quadrature family, the PPM window
#: name, this placement) and only one of them is this one.
CROSSING = "crossing"
SIGN_DEFINITE = "sign_definite"

#: The width clause's certified edge, mirrored from
#: ``minimax.beta_selector.BETA_CLAUSES['width'].beta_max`` so this module can
#: state its precondition without importing the service at import time.  The
#: mirror is proved equal in ``tests/test_mpa_sigma_routing.py``; if the
#: service moves its edge and this constant does not, that test fails.
#:
#: IT WAS 2/3 AND IS NOW 1.0, and the change is a landing, not a widening.
#: When this module was written the width clause was DECLARED at 2/3 and had
#: zero shipped tables -- ``CATALOG_CLAUSE`` stamped the whole
#: ``complex_laplace`` bundle as the *height* clause -- so every Laplace window
#: on this path refused on an empty family whatever its beta.  The width sweep
#: now ships as its own catalog (``catalog_complex_laplace_width.json``,
#: ``complex_laplace_width/v1``): 15 ``positive_composite`` entries at
#: ``beta = 1.0`` exactly, ``kappa0 = 1.0`` exactly, over
#: ``R in {10, 21.54, 46.42, 100, 215.44}`` at the 1e-6, 1e-8 and 1e-12 tiers.
#:
#: THE EDGE IS DERIVED, WHICH IS WHY IT IS EXACTLY 1.  The clause's own record
#: cites ``sigma_routing.route_pole`` and ``pade_fit``'s fourth guard: on a
#: sign-definite branch ``x_min >= Re Omega_p``, and that guard caps
#: ``|Im Omega| <= width_ratio_max * Re Omega`` at ``width_ratio_max = 1``, so
#: ``beta = Gamma_p/x_min <= 1`` for any pole field this fitter can produce.
#: The catalog's ``measured_range`` is ``(3.6e-06, 1.0)`` -- the audit's own
#: field.  One entry covers the whole band rather than one per bucket because
#: these entries' ``beta_axis`` is ``exact_phase_on_fixed_nodes``: on the
#: composite route ``beta`` enters only as a unit-modulus phase on
#: beta-independent nodes, so the ``beta = 1.0`` table serves every
#: ``beta <= 1`` exactly.
SHIPPED_WIDTH_BETA_MAX = 1.0

#: ``pade_fit.DEFAULT_GUARDS["width_ratio_max"]`` -- the fourth guard's cap
#: on ``|Im Omega| / Re Omega``, and therefore the exact closure of the width
#: clause any pole field this fitter produces can ask for.  Same mirror
#: discipline as above.
FIT_WIDTH_RATIO_MAX = 1.0

#: Node ceiling for a composite crossing rule before it refuses.  The audit's
#: p99 on the worst pole is 111 in ``A``, or about 780 nodes at 7 nodes per
#: unit bandwidth; the tail beyond that is the vanishing-width corner, where a
#: rule sized to resolve the oscillation is not the right answer at any node
#: count.
DEFAULT_MAX_CROSSING_NODES = 4096


class RoutingRefusal(Exception):
    """No certified rule was served, and the message says what would serve one.

    Carried as an exception rather than a returned sentinel because every
    caller of this module is building a self-energy: there is no partial
    answer to fall back to, and a routing that returns "nothing" is a Sigma
    that silently drops a pole.
    """

    def __init__(self, message, *, code, beta=None, a_dim=None,
                 generator=None):
        super().__init__(message)
        self.code = str(code)
        self.beta = None if beta is None else float(beta)
        self.a_dim = None if a_dim is None else float(a_dim)
        self.generator = generator


@dataclass(frozen=True)
class WindowPlan:
    """One window's rule, and the provenance of the rule.

    ``t`` and ``alpha`` are what a ``_SigmaWindow`` carries: the tau nodes in
    the placement's own time axis and the weights that go with them.  The
    remaining fields are the evidence a Sigma pass has to record per the
    design's "each pass records the (pole index, E_ref_B, quadrature
    provenance) triple it used".
    """

    name: str
    placement: str
    t: np.ndarray
    alpha: np.ndarray
    omega_sign: int
    prefactor: float
    project: str
    e_ref_a: float
    e_ref_b: float
    kappa0: float
    max_error: float
    beta: float | None
    a_dim: float | None
    provenance: str
    extras: dict = field(default_factory=dict)

    @property
    def n_tau(self) -> int:
        return int(self.t.shape[0])


def denominator_can_cross(space: str, neg_omega_half: bool) -> bool:
    """True when ``S = E_A + Omega_p`` can coincide with an evaluation point.

    The same predicate ``ppm_windows._build_windows_for_branch`` uses, restated
    here so the MPA path reads its placement off the branch identity rather
    than off a stored sign.  The crossing role belongs to the empty A-space on
    the ``+omega`` half and to the occupied A-space on the ``-omega`` half.
    """
    if space not in ("cond", "val"):
        raise ValueError(
            f"denominator_can_cross: space={space!r} is not 'cond' or 'val'.")
    return (space == "cond") != bool(neg_omega_half)


def exact_resolvent(u, gamma, placement):
    """The closed form each placement's quadrature must reproduce.

    ``CROSSING``      -> ``1/(u + i*gamma)``  with ``u`` the signed distance
                         ``omega~ - Re S``, which changes sign across the pane.
    ``SIGN_DEFINITE`` -> ``1/(u - i*gamma)``  with ``u = omega~ + Re S > 0``.

    This is the oracle the red twins are scored against, and it is deliberately
    the only place either sign is written down as an expression.
    """
    u = np.asarray(u, dtype=np.float64)
    g = float(gamma)
    if placement == CROSSING:
        return 1.0 / (u + 1j * g)
    if placement == SIGN_DEFINITE:
        return 1.0 / (u - 1j * g)
    raise ValueError(
        f"exact_resolvent: placement={placement!r} is neither {CROSSING!r} "
        f"nor {SIGN_DEFINITE!r}.")


def refuse_edge_factor_below_envelope(edge_factor,
                                      beta_max=SHIPPED_WIDTH_BETA_MAX):
    """The crossing branches' beta bound is ``1/edge_factor``.  Check it.

    On a crossing branch every Laplace window floors ``x_min`` at
    ``z_edge = edge_factor * Gamma_p``, so ``beta = Gamma_p/x_min`` cannot
    exceed ``1/edge_factor`` whatever the pole field does -- which is why the
    audit found zero of 81 million poles outside the width clause there.  That
    is a bound the DECK controls, and lowering ``sigma_window_edge_factor``
    below ``1/beta_max`` moves the whole field outside the envelope at once.
    The failure would be a catalog refusal on every window rather than a wrong
    number, but it would arrive with no explanation attached, so it gets one
    here.
    """
    ef = float(edge_factor)
    if not (ef > 0.0) or not np.isfinite(ef):
        raise RoutingRefusal(
            f"sigma_window_edge_factor={edge_factor!r} is not a finite "
            f"positive number; the MPA crossing windows pad by "
            f"edge_factor*Gamma_p and cannot be built without it.",
            code="edge_factor_invalid")
    if ef * float(beta_max) < 1.0:
        raise RoutingRefusal(
            f"sigma_window_edge_factor={ef} puts the MPA crossing branches' "
            f"width clause at beta <= 1/{ef} = {1.0 / ef:.4f}, outside the "
            f"certified width envelope beta_max={float(beta_max):.4f}.  On a "
            f"crossing branch x_min is floored at edge_factor*Gamma_p, so "
            f"this bound holds for every pole of every field at once and the "
            f"whole run would refuse at the catalog rather than at one pole.  "
            f"Raise sigma_window_edge_factor to at least "
            f"{1.0 / float(beta_max):.4f}, or extend the width clause.",
            code="edge_factor_below_envelope",
            beta=1.0 / ef)
    return float(1.0 / ef)


def beta_for_window(gamma_ry, x_min_ry):
    """``beta = Gamma_p / x_min`` -- the width clause the window asks for.

    ``x_min`` is the LOWER EDGE of the window's Laplace interval in Ry, which
    is the quantity the catalog rescales by (its tables live on ``[1, R]``).
    The audit considered and rejected every other candidate denominator: the
    pole position ``a_p`` is not what the rule is scaled by, and the interval
    width ``x_max - x_min`` is the range axis ``R`` and already has its own.
    """
    x = float(x_min_ry)
    if not (x > 0.0) or not np.isfinite(x):
        raise RoutingRefusal(
            f"beta_for_window: x_min={x_min_ry!r} is not a finite positive "
            f"Laplace edge, so no width clause can be formed for this window.",
            code="x_min_invalid")
    return float(abs(float(gamma_ry)) / x)


def _projected_node_count(varpi, freq_max, rel_tol):
    """What the composite rule WILL cost, computed before it is built.

    The same three lines ``evaluator.damped_line_rule`` opens with -- the
    truncation ``t_max = ln(2/tol)/varpi``, the panel count at a fixed number
    of wavelengths of ``freq_max``, and the per-panel order floored by the
    Bernstein resolution bound -- evaluated as arithmetic instead of as a loop.

    IT EXISTS BECAUSE THE REFUSAL MUST COME FIRST.  Checking the node count on
    the built rule is a refusal that arrives after the work: at the audit's
    narrow-width tail (``A`` up to 2.7e6) the builder would run a hundred
    thousand ``leggauss`` calls to produce a rule whose only use is to be
    rejected.  The projection is an UNDER-estimate of the true count (it takes
    the resolution floor, not the tolerance-driven order), so a rule this
    function admits may still be refused on its real size, and one it refuses
    could never have been small enough.
    """
    from gw.mpa.evaluator import DEFAULT_WAVELENGTHS_PER_PANEL
    v, f, tol = float(varpi), float(freq_max), float(rel_tol)
    if not (v > 0.0 and f > 0.0 and 0.0 < tol < 1.0):
        return None                      # the builder's own gates say why
    t_max = np.log(2.0 / tol) / v
    panel_target = DEFAULT_WAVELENGTHS_PER_PANEL * 2.0 * np.pi / f
    n_panels = max(1, int(np.ceil(t_max / panel_target)))
    width = t_max / n_panels
    order_floor = int(np.ceil((f + v) * width / 4.0)) + 1
    return n_panels * order_floor


def _composite(varpi, freq_max, rel_tol, max_nodes, what):
    """The positive composite rule, with its refusals renamed for this path."""
    from gw.mpa.evaluator import damped_line_rule
    projected = _projected_node_count(varpi, freq_max, rel_tol)
    if projected is not None and projected > int(max_nodes):
        raise RoutingRefusal(
            f"{what}: the certified rule for dimensionless bandwidth "
            f"A = f_max/varpi = {float(freq_max) / float(varpi):.4g} needs at "
            f"least {projected} nodes, above the cap "
            f"max_nodes={int(max_nodes)}.  This is the vanishing-width corner "
            f"-- the audit's pole-0 tail reaches A = 2.7e6 -- where resolving "
            f"the oscillation is not the right answer at any node count.  "
            f"Either widen the served pane, raise the cap deliberately, or "
            f"route this pole through a certified minimax entry: "
            f"tools/generate_imag_minimax_assets.py --r <R> --beta <beta> "
            f"--error-tier <eps>",
            code="bandwidth_above_cap",
            a_dim=float(freq_max) / max(float(varpi), 1e-300),
            generator=("tools/generate_imag_minimax_assets.py "
                       "--r <R> --beta <beta> --error-tier <eps>"))
    try:
        rule = damped_line_rule(varpi, freq_max, rel_tol=rel_tol)
    except ValueError as exc:
        raise RoutingRefusal(
            f"{what}: the positive composite rule could not be built "
            f"(varpi={varpi:.6g} Ry, freq_max={freq_max:.6g} Ry, "
            f"rel_tol={rel_tol:.1e}).  Underlying gate: {exc}",
            code="composite_ungated",
            a_dim=float(freq_max) / max(float(varpi), 1e-300)) from exc
    if rule["n_nodes"] > int(max_nodes):
        raise RoutingRefusal(
            f"{what}: the certified rule for dimensionless bandwidth "
            f"A = f_max/varpi = {rule['a_dim']:.4g} needs "
            f"{rule['n_nodes']} nodes, above the cap "
            f"max_nodes={int(max_nodes)}.  This is the vanishing-width "
            f"corner -- the audit's pole-0 tail "
            f"reaches A = 2.7e6 -- where resolving the oscillation is not the "
            f"right answer at any node count.  Either widen the served pane, "
            f"raise the cap deliberately, or route this pole through a "
            f"certified minimax entry: tools/generate_imag_minimax_assets.py "
            f"--r <R> --beta <beta> --error-tier <eps>",
            code="bandwidth_above_cap",
            a_dim=float(rule["a_dim"]),
            generator=("tools/generate_imag_minimax_assets.py "
                       "--r <R> --beta <beta> --error-tier <eps>"))
    # Positivity IS the stability certificate (THEORY_mpa_implementation 4.3):
    # kappa0 = varpi * sum h_l e^{-varpi t_l} <= 1 by construction, and a
    # violation means the weights are not what the rule claims.
    if not (rule["kappa0"] <= 1.0 + 1e-9):
        raise RoutingRefusal(          # pragma: no cover - guards the guard
            f"{what}: the composite rule reports kappa0={rule['kappa0']!r} > 1"
            f", which positivity forbids.  The amplification certificate is "
            f"the only certificate this route has, so it is refused rather "
            f"than used.",
            code="kappa0_above_one")
    return rule


def crossing_rule(gamma_ry, f_max_ry, *, rel_tol=1e-8,
                  max_nodes=DEFAULT_MAX_CROSSING_NODES):
    """Nodes and weights for ``1/(u + i*Gamma_p)`` on the real tau axis.

    ``f_max_ry`` is the largest angular frequency the crossing integrand
    carries over the window it serves -- ``omega_max + (T - min E_A)``, a BEAT
    frequency and not a transition energy (section 4.4).  ``gamma_ry`` is the
    pole's fitted width and is the ONLY damping in the integrand: there is no
    ``xi`` and no ``i*delta`` on this path.

    Returns ``(t, alpha, rule)`` with ``t`` real (cast complex by the caller)
    and ``alpha = -i*h`` -- the ``-i`` being the whole of the difference
    between ``int_0^inf e^{iut}e^{-Gamma t} dt`` and the resolvent it equals.
    """
    g = float(gamma_ry)
    if not (g > 0.0) or not np.isfinite(g):
        raise RoutingRefusal(
            f"crossing_rule: Gamma_p={gamma_ry!r} is not a finite positive "
            f"width.  On the MPA path the fitted width IS the regularization, "
            f"so a zero-width pole on a crossing branch has a genuinely "
            f"singular denominator and there is nothing left to choose: it "
            f"needs the two-point path's xi, a narrower pane, or a fit whose "
            f"guards did not return a real pole.",
            code="zero_width_crossing")
    rule = _composite(g, float(f_max_ry), float(rel_tol), int(max_nodes),
                      "crossing_rule")
    t = np.asarray(rule["t"], dtype=np.float64)
    alpha = (-1j) * np.asarray(rule["h"], dtype=np.float64)
    return t, alpha.astype(np.complex128), rule


def sign_definite_rule(x_min_ry, x_max_ry, gamma_ry, *, rel_tol=1e-8,
                       max_nodes=DEFAULT_MAX_CROSSING_NODES):
    """Nodes and weights for ``1/(w - i*Gamma_p)`` on the imaginary tau axis.

    The width is a PHASE here, not extra decay, so the shipped real ``1/y``
    tables are the wrong function on this cell by two to three orders (section
    C) and are not consulted.  The decay that makes the integral converge is
    ``x_min``, the lower edge of the served interval; the oscillation the rule
    must resolve is ``Gamma_p`` plus the interval's own spread.

    Returns ``(t, alpha, rule)`` with ``t = -i*tau`` and ``alpha = h`` real
    positive.
    """
    x_min = float(x_min_ry)
    x_max = float(x_max_ry)
    if not (0.0 < x_min <= x_max) or not np.isfinite(x_max):
        raise RoutingRefusal(
            f"sign_definite_rule: the served interval [{x_min_ry!r}, "
            f"{x_max_ry!r}] is not a positive ordered interval, so the "
            f"sign-definite denominator is not sign-definite on it.",
            code="interval_invalid")
    g = abs(float(gamma_ry))
    freq = g + (x_max - x_min)
    if freq <= 0.0:
        # A zero-width pole on a degenerate interval is the static limit; the
        # composite rule needs a positive bandwidth, and 1 part in 1e6 of
        # x_min is a bandwidth that costs one panel.
        freq = 1e-6 * x_min
    rule = _composite(x_min, freq, float(rel_tol), int(max_nodes),
                      "sign_definite_rule")
    tau = np.asarray(rule["t"], dtype=np.float64)
    t = (-1j) * tau.astype(np.complex128)
    alpha = np.asarray(rule["h"], dtype=np.float64).astype(np.complex128)
    return t, alpha, rule


def route_pole(
    *,
    a_ry,
    gamma_ry,
    space,
    neg_omega_half,
    omega_max_ry,
    e_a_min_ry,
    e_a_max_ry,
    edge_factor,
    rel_tol=1e-8,
    max_nodes=DEFAULT_MAX_CROSSING_NODES,
):
    """The window plan for ONE pole on ONE branch.

    Mirrors ``ppm_windows._build_windows_for_branch``'s decision -- three
    windows when the denominator can cross, one when it cannot -- with the
    two MPA substitutions: ``z_edge = edge_factor * Gamma_p`` instead of
    ``edge_factor * xi``, and the exact complex resolvent instead of the HGL
    regularized target.  ``a_ry`` is ``Re Omega_p`` and ``gamma_ry`` is
    ``|Im Omega_p|``; the caller has already taken the pole slice apart, so
    this function never sees a complex number and cannot conjugate one.

    Returns a list of :class:`WindowPlan`.
    """
    a = float(a_ry)
    g = abs(float(gamma_ry))
    w_max = float(omega_max_ry)
    e_min = float(e_a_min_ry)
    e_max = float(e_a_max_ry)
    if not (a > 0.0):
        raise RoutingRefusal(
            f"route_pole: Re Omega_p={a_ry!r} is not positive.  pade_fit "
            f"returns poles in the closed fourth quadrant and a pole outside "
            f"it never reached the store through a guard that was running.",
            code="pole_outside_quadrant")
    if e_max < e_min:
        raise ValueError(
            f"route_pole: E_A range [{e_min}, {e_max}] is inverted.")

    crossing = denominator_can_cross(space, bool(neg_omega_half))
    neg = -1.0 if neg_omega_half else 1.0
    plans: list[WindowPlan] = []

    if not crossing:
        # One sign-definite Laplace window over the whole A-space.
        x_min = max(e_min + a, 1e-12)
        x_max = max(e_max + a + w_max, x_min * (1.0 + 1e-9))
        t, alpha, rule = sign_definite_rule(
            x_min, x_max, g, rel_tol=rel_tol, max_nodes=max_nodes)
        plans.append(WindowPlan(
            name="single", placement=SIGN_DEFINITE, t=t, alpha=alpha,
            omega_sign=-1, prefactor=float(-1.0 * neg), project="full",
            e_ref_a=e_min, e_ref_b=a,
            kappa0=float(rule["kappa0"]), max_error=float(rule["rel_tol"]),
            beta=beta_for_window(g, x_min), a_dim=float(rule["a_dim"]),
            provenance=("positive composite Gauss-Legendre, "
                        f"{rule['n_panels']} panels, kappa0="
                        f"{rule['kappa0']:.6f} (positivity certificate)"),
            extras={"x_min_ry": x_min, "x_max_ry": x_max}))
        return plans

    refuse_edge_factor_below_envelope(edge_factor)
    z_edge = float(edge_factor) * g
    T = w_max + z_edge

    if a <= T:
        # core: the crossing window proper, A-space inside the pane.
        f_max = w_max + max(T - e_min, 0.0)
        t, alpha, rule = crossing_rule(
            g, f_max, rel_tol=rel_tol, max_nodes=max_nodes)
        plans.append(WindowPlan(
            name="core", placement=CROSSING, t=t.astype(np.complex128),
            alpha=alpha, omega_sign=+1, prefactor=float(1.0 * neg),
            # THE MERGED ONE-COMPLEX-CHAIN PROJECTION, not the HGL split.  The
            # crossing consumer here forms the full complex product coeff*X and
            # never weights Re and Im by two independent real vectors, which is
            # the property that licensed the merge; the sine-only projection
            # would keep the absorptive part and drop the dispersive one.
            project="full",
            e_ref_a=e_min, e_ref_b=a,
            kappa0=float(rule["kappa0"]), max_error=float(rule["rel_tol"]),
            beta=None, a_dim=float(rule["a_dim"]),
            provenance=("positive composite Gauss-Legendre on the exact "
                        f"complex resolvent, {rule['n_panels']} panels, "
                        f"A={rule['a_dim']:.4g}, kappa0="
                        f"{rule['kappa0']:.6f} (positivity certificate)"),
            extras={"f_max_ry": f_max, "T_ry": T}))
        # a_stripe: A-space above the pane, same pole -- sign-definite again.
        if e_max > T:
            x_min = max(max(T + a - (T - z_edge), z_edge), 1e-12)
            x_max = max(e_max + a, x_min * (1.0 + 1e-9))
            t, alpha, rule = sign_definite_rule(
                x_min, x_max, g, rel_tol=rel_tol, max_nodes=max_nodes)
            plans.append(WindowPlan(
                name="a_stripe", placement=SIGN_DEFINITE, t=t, alpha=alpha,
                omega_sign=+1, prefactor=float(1.0 * neg), project="full",
                e_ref_a=T, e_ref_b=a,
                kappa0=float(rule["kappa0"]),
                max_error=float(rule["rel_tol"]),
                beta=beta_for_window(g, x_min), a_dim=float(rule["a_dim"]),
                provenance=("positive composite Gauss-Legendre, "
                            f"{rule['n_panels']} panels, kappa0="
                            f"{rule['kappa0']:.6f} (positivity certificate)"),
                extras={"x_min_ry": x_min, "x_max_ry": x_max}))
        return plans

    # b_slab: the pole sits above the pane, so nothing crosses; whole A-space.
    x_min = max(max(e_min + a - (T - z_edge), z_edge), 1e-12)
    x_max = max(e_max + a, x_min * (1.0 + 1e-9))
    t, alpha, rule = sign_definite_rule(
        x_min, x_max, g, rel_tol=rel_tol, max_nodes=max_nodes)
    plans.append(WindowPlan(
        name="b_slab", placement=SIGN_DEFINITE, t=t, alpha=alpha,
        omega_sign=+1, prefactor=float(1.0 * neg), project="full",
        e_ref_a=e_min, e_ref_b=a,
        kappa0=float(rule["kappa0"]), max_error=float(rule["rel_tol"]),
        beta=beta_for_window(g, x_min), a_dim=float(rule["a_dim"]),
        provenance=("positive composite Gauss-Legendre, "
                    f"{rule['n_panels']} panels, kappa0="
                    f"{rule['kappa0']:.6f} (positivity certificate)"),
        extras={"x_min_ry": x_min, "x_max_ry": x_max}))
    return plans

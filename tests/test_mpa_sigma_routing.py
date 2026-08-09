"""The MPA Sigma routing, and the two conventions it must not get wrong.

THE RED TWINS ARE THE FIRST GATE, and they come before any real number is
produced, because this codebase has a four-instance history of conjugation
errors that produced finite, plausible, wrong answers.  Two deliberate
mistakes are planted here -- a conjugated ``Omega_p`` and a swapped branch
placement -- and each must fail with a DISCRIMINATING signature, not merely
fail: a test that only asserts "the number moved" passes for a typo too.
"""

import numpy as np
import pytest

from gw.mpa import sigma_routing as R


# ---------------------------------------------------------------------------
#  The quadrature sums, written once, in the form the tau kernel evaluates
# ---------------------------------------------------------------------------

def _crossing_sum(t, alpha, u, gamma, *, conjugate=False):
    """What the tau kernel computes for a crossing window at offset ``u``.

    ``t`` real, ``alpha = -i h``.  The operand supplies ``e^{-i Omega t}``,
    whose imaginary part IS the damping -- so conjugating ``Omega`` flips
    ``e^{-Gamma t}`` to ``e^{+Gamma t}`` and nothing else in the expression
    changes.  That is exactly the planted twin.
    """
    om = np.asarray(gamma, dtype=np.float64)
    damp = np.exp(+om * np.real(t)) if conjugate else np.exp(-om * np.real(t))
    phase = np.exp(1j * np.asarray(u)[..., None] * np.real(t)[None, :])
    return np.sum(alpha[None, :] * phase * damp[None, :], axis=-1)


def _sign_definite_sum(t, alpha, w, gamma, *, swap_branch=False):
    """What the tau kernel computes for a sign-definite window at ``w``.

    ``t = -i tau``, ``alpha = h``.  ``swap_branch=True`` plants the crossing
    placement here -- real tau nodes and the ``-i`` weight -- which is the
    mistake an occupation predicate reading the wrong way makes.
    """
    tau = np.real(1j * np.asarray(t))
    g = float(gamma)
    w = np.asarray(w, dtype=np.float64)[..., None]
    if swap_branch:
        return np.sum(((-1j) * np.real(alpha))[None, :]
                      * np.exp(1j * w * tau[None, :])
                      * np.exp(-g * tau)[None, :], axis=-1)
    return np.sum(np.real(alpha)[None, :] * np.exp(-w * tau[None, :])
                  * np.exp(1j * g * tau)[None, :], axis=-1)


# ---------------------------------------------------------------------------
#  GATE 1 -- the crossing kernel IS the exact complex Lorentzian resolvent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma_ev,f_max_ev", [(0.6, 12.0), (1.5, 20.0),
                                               (2.8, 30.0), (8.0, 40.0)])
def test_crossing_rule_reproduces_the_exact_resolvent(gamma_ev, f_max_ev):
    """No i*delta, no xi: the fitted width alone closes the integral.

    The widths are the first-light field's own quartiles (0.6-2.8 eV, with the
    8 eV arm reaching pole 0's tail), so this is not a synthetic corner.
    """
    ryd = 13.605693122994
    g, f = gamma_ev / ryd, f_max_ev / ryd
    t, alpha, rule = R.crossing_rule(g, f, rel_tol=1e-9)
    u = np.linspace(-f, f, 401)
    got = _crossing_sum(t, alpha, u, g)
    exact = R.exact_resolvent(u, g, R.CROSSING)
    err = np.max(np.abs(got - exact)) * g          # relative to the 1/g scale
    assert err < 1e-6, f"max relative error {err:.3e} at A={rule['a_dim']:.3g}"
    # Positivity is the certificate; the weights must actually be positive.
    assert np.all(np.imag(alpha) < 0.0)
    assert np.allclose(np.real(alpha), 0.0)
    assert rule["kappa0"] <= 1.0 + 1e-12


def test_sign_definite_rule_reproduces_its_own_resolvent():
    """In imaginary time the width is a phase, and the sign of it is fixed."""
    ryd = 13.605693122994
    x_min, x_max, g = 1.0 / ryd, 40.0 / ryd, 1.5 / ryd
    t, alpha, rule = R.sign_definite_rule(x_min, x_max, g, rel_tol=1e-9)
    w = np.linspace(x_min, x_max, 401)
    got = _sign_definite_sum(t, alpha, w, g)
    exact = R.exact_resolvent(w, g, R.SIGN_DEFINITE)
    err = np.max(np.abs(got - exact)) * x_min
    assert err < 1e-5, f"max relative error {err:.3e}"
    assert np.all(np.real(alpha) > 0.0)
    assert rule["kappa0"] <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
#  GATE 2 -- RED TWIN A: a conjugated Omega_p
# ---------------------------------------------------------------------------

def test_red_twin_conjugated_omega_fails_by_divergence_not_by_a_little():
    """``Omega_p -> conj(Omega_p)`` turns the damping into growth.

    The discriminating signature is not "the answer moved": it is that the
    error grows like ``e^{Gamma t_max}`` with the truncation, so the twin's
    residual is many orders above the correct one AND scales with the
    tolerance the correct rule was built to.  A tighter rel_tol makes the
    CORRECT rule better and the twin WORSE, which no ordinary bug does.
    """
    ryd = 13.605693122994
    g, f = 1.5 / ryd, 20.0 / ryd
    u = np.linspace(-f, f, 201)
    exact = R.exact_resolvent(u, g, R.CROSSING)

    errs = {}
    for tol in (1e-6, 1e-9, 1e-12):
        t, alpha, _ = R.crossing_rule(g, f, rel_tol=tol, max_nodes=1 << 16)
        good = np.max(np.abs(_crossing_sum(t, alpha, u, g) - exact)) * g
        bad = np.max(np.abs(
            _crossing_sum(t, alpha, u, g, conjugate=True) - exact)) * g
        errs[tol] = (good, bad)
        assert bad > 1e3 * max(good, 1e-15), (
            f"conjugated twin at rel_tol={tol:.0e} is only {bad:.3e} against "
            f"{good:.3e} -- it must fail loudly, not quietly")

    # The signature: tightening the tolerance improves the truth and degrades
    # the twin.  This is what separates "conjugated" from "slightly wrong".
    assert errs[1e-12][0] < errs[1e-6][0]
    assert errs[1e-12][1] > errs[1e-6][1]


def test_red_twin_conjugated_omega_loses_the_definite_absorptive_sign():
    """Second, independent signature: the absorptive part stops being one sign.

    ``Im[1/(u + i*Gamma)] < 0`` at every ``u`` -- one sign over the whole pane
    is what time ordering buys, and it is the property a spectral function is
    read off.  The conjugated twin's sum is dominated by a tail that GROWS with
    tau, so its imaginary part oscillates and changes sign across most of the
    pane.  Counting sign changes discriminates the twin from a merely
    inaccurate rule, which keeps the sign and loses only digits; a magnitude
    test alone would not, because an over-tight rule is also large.
    """
    ryd = 13.605693122994
    g, f = 1.5 / ryd, 20.0 / ryd
    t, alpha, _ = R.crossing_rule(g, f, rel_tol=1e-9)
    u = np.linspace(-f, f, 201)
    good = _crossing_sum(t, alpha, u, g)
    bad = _crossing_sum(t, alpha, u, g, conjugate=True)

    def sign_changes(z):
        s = np.sign(np.imag(z))
        return int(np.count_nonzero(np.diff(s)))

    assert np.all(np.imag(good) < 0.0)
    assert sign_changes(good) == 0
    assert sign_changes(bad) > 0.5 * (u.size - 1), (
        f"conjugated twin changed absorptive sign only "
        f"{sign_changes(bad)} times in {u.size - 1} steps")


# ---------------------------------------------------------------------------
#  GATE 3 -- RED TWIN B: a swapped branch placement
# ---------------------------------------------------------------------------

def test_red_twin_swapped_branch_computes_the_other_resolvent():
    """Occupations select the branch; swapping them selects the wrong pole.

    The twin is a legal-looking sum -- finite, smooth, the right magnitude --
    and it is the resolvent of the OTHER placement.  The signature is that its
    residual against its own placement's oracle is large while its residual
    against the OTHER oracle is small, which is the statement "this is the
    wrong analytic object" and not "this is inaccurate".
    """
    ryd = 13.605693122994
    x_min, x_max, g = 1.0 / ryd, 40.0 / ryd, 1.5 / ryd
    t, alpha, _ = R.sign_definite_rule(x_min, x_max, g, rel_tol=1e-9)
    w = np.linspace(x_min, x_max, 201)
    truth = R.exact_resolvent(w, g, R.SIGN_DEFINITE)

    good = _sign_definite_sum(t, alpha, w, g)
    bad = _sign_definite_sum(t, alpha, w, g, swap_branch=True)
    assert np.max(np.abs(good - truth)) * x_min < 1e-5
    assert np.max(np.abs(bad - truth)) * x_min > 1e-2

    # And the absorptive sign is the discriminator, on the same pane.
    assert np.all(np.imag(truth) > 0.0)          # 1/(w - i*Gamma)
    assert np.mean(np.imag(bad) < 0.0) > 0.9     # the crossing placement's


def test_the_two_placements_are_not_each_others_conjugate():
    """There is no global conjugation identity to exploit, and here it is.

    ``Sigma_c(-omega) = -[Sigma_c(omega)]*`` is quoted often and is false; the
    reflection holds per pole term, which is exactly what moves the crossing
    role between the cond and val A-spaces.  If the two placements WERE
    conjugates of one another the routing could build one and reflect it, so
    the falseness is pinned rather than remembered.
    """
    u = np.linspace(0.05, 3.0, 64)
    g = 0.11
    a = R.exact_resolvent(u, g, R.CROSSING)
    b = R.exact_resolvent(u, g, R.SIGN_DEFINITE)
    assert np.allclose(b, np.conj(a))            # at EQUAL argument, yes
    # but the arguments are never equal: one is omega~ - Re S (signed, crosses
    # zero) and the other omega~ + Re S (strictly positive), so no reflection
    # of the branch as a whole exists.
    assert R.denominator_can_cross("cond", False)
    assert not R.denominator_can_cross("val", False)
    assert not R.denominator_can_cross("cond", True)
    assert R.denominator_can_cross("val", True)


# ---------------------------------------------------------------------------
#  GATE 4 -- the envelope audit's verdict, as preconditions
# ---------------------------------------------------------------------------

def test_the_mirrored_constants_match_the_service_and_the_fitter():
    """The two numbers this module mirrors are proved, not copied and hoped."""
    from minimax.beta_selector import BETA_CLAUSES, WIDTH
    assert BETA_CLAUSES[WIDTH].beta_max == R.SHIPPED_WIDTH_BETA_MAX
    from gw.mpa.pade_fit import DEFAULT_GUARDS
    assert DEFAULT_GUARDS["width_ratio_max"] == R.FIT_WIDTH_RATIO_MAX


def test_crossing_branch_beta_is_capped_by_the_edge_factor():
    """The audit found 0 of 81 432 576 poles outside the width clause here.

    It is structural: ``x_min`` is floored at ``edge_factor*Gamma_p``, so
    ``beta <= 1/edge_factor``.  Swept over four decades of width and the whole
    depth range of the first-light field, the bound must never be touched.
    """
    ryd = 13.605693122994
    e_min, e_max = 0.3464 / ryd, 56.36 / ryd     # the deck's own cond spectrum
    w_max = 5.0 / ryd
    edge = 1.5
    worst = 0.0
    n_served = n_refused = 0
    for gamma_ev in (0.001, 0.01, 0.1, 0.6, 1.5, 2.8, 8.0, 30.0):
        for a_ev in (0.5, 2.0, 6.0, 12.0, 30.0, 120.0):
            try:
                plans = R.route_pole(
                    a_ry=a_ev / ryd, gamma_ry=gamma_ev / ryd, space="cond",
                    neg_omega_half=False, omega_max_ry=w_max,
                    e_a_min_ry=e_min, e_a_max_ry=e_max, edge_factor=edge,
                    rel_tol=1e-6)
            except R.RoutingRefusal as exc:
                # The narrow-width corner refuses BY NAME.  That is the other
                # half of the audit's verdict and is not a hole in this bound.
                assert exc.code == "bandwidth_above_cap"
                n_refused += 1
                continue
            n_served += 1
            for pl in plans:
                if pl.beta is not None:
                    worst = max(worst, pl.beta)
    assert n_served and n_refused          # both arms of the sweep are live
    assert worst <= 1.0 / edge + 1e-12, worst
    assert worst <= R.SHIPPED_WIDTH_BETA_MAX + 1e-12


def test_sign_definite_branch_reaches_the_width_clause_ceiling():
    """Where the audit's 62.25 % of |B| mass past 2/3 comes from, and that the
    clause that now serves it reaches exactly as far as the fitter can ask.

    On a sign-definite branch ``x_min = min(E_A) + a_p`` with no edge floor,
    and ``min(E_A)`` is half the gap.  A pole at the fitter's own width guard
    (``Gamma = Re Omega``) therefore asks for beta just under 1.  When this
    cell was written that was OUTSIDE the shipped clause and the assertion
    read ``2/3 < beta``; the width catalog has since landed at
    ``beta_max = 1.0``, so what the cell pins now is the pair of facts that
    survived the landing: the branch really does reach the ceiling, and the
    ceiling really is the fitter's guard, so the request is served rather than
    refused.
    """
    ryd = 13.605693122994
    e_min, e_max = 0.3464 / ryd, 12.32 / ryd     # the deck's own val spectrum
    a_ev = 20.0
    plans = R.route_pole(
        a_ry=a_ev / ryd, gamma_ry=a_ev / ryd,    # Gamma = Re Omega, the guard
        space="val", neg_omega_half=False, omega_max_ry=5.0 / ryd,
        e_a_min_ry=e_min, e_a_max_ry=e_max, edge_factor=1.5,
        rel_tol=1e-6, max_nodes=1 << 16)
    assert len(plans) == 1 and plans[0].name == "single"
    beta = plans[0].beta
    # Past the OLD 2/3 declaration -- the finding that motivated the campaign.
    assert beta > 2.0 / 3.0
    # And inside the clause as it now ships, which is the fitter's own guard.
    assert beta <= R.SHIPPED_WIDTH_BETA_MAX
    assert R.SHIPPED_WIDTH_BETA_MAX == R.FIT_WIDTH_RATIO_MAX
    assert beta > 0.9


def test_edge_factor_below_the_envelope_refuses_by_name():
    """Lowering the deck's edge factor moves the WHOLE field out at once.

    The boundary moved with the clause: the crossing branches are bounded by
    ``beta <= 1/edge_factor``, so the precondition is ``edge_factor >= 1/
    beta_max``.  At the declared 2/3 that meant 1.5 -- exactly the deck
    default, which is why the coincidence was worth checking.  At the landed
    1.0 it means 1.0, so the deck now has headroom it did not have, and the
    refusal bites only below that.
    """
    with pytest.raises(R.RoutingRefusal) as exc:
        R.refuse_edge_factor_below_envelope(0.5)
    assert exc.value.code == "edge_factor_below_envelope"
    assert "sigma_window_edge_factor" in str(exc.value)
    # 1.0 is the new boundary and is admissible; 1.5 is the deck default and
    # now sits inside with room to spare.
    assert R.refuse_edge_factor_below_envelope(1.0) == pytest.approx(1.0)
    assert R.refuse_edge_factor_below_envelope(1.5) == pytest.approx(2.0 / 3.0)
    # The old edge, checked against the old clause, would still have refused
    # at edge_factor = 1.0 -- pinned so the loosening is deliberate and dated.
    with pytest.raises(R.RoutingRefusal):
        R.refuse_edge_factor_below_envelope(1.0, beta_max=2.0 / 3.0)


def test_vanishing_width_on_a_crossing_branch_refuses_by_name():
    """The audit's pole-0 tail (A up to 2.7e6) refuses, never builds."""
    with pytest.raises(R.RoutingRefusal) as exc:
        R.crossing_rule(0.0, 1.0)
    assert exc.value.code == "zero_width_crossing"
    assert "the fitted width IS the regularization" in str(exc.value)

    with pytest.raises(R.RoutingRefusal) as exc:
        R.crossing_rule(1e-7, 1.0, rel_tol=1e-9, max_nodes=64)
    assert exc.value.code == "bandwidth_above_cap"
    assert exc.value.generator and "generate_imag_minimax_assets" in \
        exc.value.generator


def test_a_pole_outside_the_fourth_quadrant_refuses():
    """The guards run at fit time; the routing does not trust that they did."""
    with pytest.raises(R.RoutingRefusal) as exc:
        R.route_pole(a_ry=-0.1, gamma_ry=0.01, space="cond",
                     neg_omega_half=False, omega_max_ry=0.37,
                     e_a_min_ry=0.03, e_a_max_ry=4.0, edge_factor=1.5)
    assert exc.value.code == "pole_outside_quadrant"


# ---------------------------------------------------------------------------
#  GATE 5 -- the routing's shape, per branch
# ---------------------------------------------------------------------------

def test_crossing_branch_shallow_pole_gets_core_plus_stripe():
    ryd = 13.605693122994
    plans = R.route_pole(
        a_ry=3.0 / ryd, gamma_ry=1.0 / ryd, space="cond",
        neg_omega_half=False, omega_max_ry=5.0 / ryd,
        e_a_min_ry=0.35 / ryd, e_a_max_ry=56.0 / ryd, edge_factor=1.5)
    assert [p.name for p in plans] == ["core", "a_stripe"]
    core = plans[0]
    assert core.placement == R.CROSSING and core.omega_sign == +1
    # The merged one-complex-chain projection, never the sine-only one.
    assert core.project == "full"
    assert core.beta is None and core.a_dim is not None


def test_crossing_branch_deep_pole_gets_the_slab_only():
    ryd = 13.605693122994
    plans = R.route_pole(
        a_ry=40.0 / ryd, gamma_ry=8.0 / ryd, space="cond",
        neg_omega_half=False, omega_max_ry=5.0 / ryd,
        e_a_min_ry=0.35 / ryd, e_a_max_ry=56.0 / ryd, edge_factor=1.5)
    assert [p.name for p in plans] == ["b_slab"]
    assert plans[0].placement == R.SIGN_DEFINITE


def test_the_minus_omega_half_carries_its_own_minus_one():
    ryd = 13.605693122994
    kw = dict(a_ry=3.0 / ryd, gamma_ry=1.0 / ryd, omega_max_ry=5.0 / ryd,
              e_a_min_ry=0.35 / ryd, e_a_max_ry=56.0 / ryd, edge_factor=1.5)
    pos = R.route_pole(space="cond", neg_omega_half=False, **kw)
    neg = R.route_pole(space="val", neg_omega_half=True, **kw)
    assert [p.name for p in pos] == [p.name for p in neg]
    for p, n in zip(pos, neg):
        assert p.prefactor == -n.prefactor


# ---------------------------------------------------------------------------
#  GATE 6 -- the door is open: the audited field is SERVED, not refused
# ---------------------------------------------------------------------------

#: The measured percentile box of the first-light pole field (si_mpa_0808,
#: 81 432 576 live poles), in eV.  Percentiles [0, 1, 25, 50, 75, 99, 100] of
#: ``|Im Omega|`` and ``Re Omega``, straight out of pole_envelope_audit.json.
#: Encoded here rather than read from /pscratch so the cell runs anywhere; the
#: numbers are the audit's, and if the field is ever re-fitted they should be
#: re-measured rather than adjusted to keep this green.
AUDIT_GAMMA_EV = (3.959e-05, 0.06004, 1.418, 3.565, 16.27, 488.0, 4015.0)
AUDIT_A_EV = (0.2647, 2.601, 10.53, 23.09, 38.64, 1112.0, 6451.0)
AUDIT_E_COND_EV = (0.3464, 56.3596)
AUDIT_H_VAL_EV = (0.3464, 12.3247)


def _route_the_audited_box(max_nodes=R.DEFAULT_MAX_CROSSING_NODES):
    """Route the field's percentile box on all four branches.

    ``Gamma > Re Omega`` combinations are skipped, not because they would be
    awkward but because ``pade_fit``'s fourth guard means they cannot exist.
    """
    ryd = 13.605693122994
    branches = (
        ("cond", False, AUDIT_E_COND_EV), ("val", False, AUDIT_H_VAL_EV),
        ("cond", True, AUDIT_E_COND_EV), ("val", True, AUDIT_H_VAL_EV))
    served, refused, betas = 0, [], []
    for g in AUDIT_GAMMA_EV:
        for a in AUDIT_A_EV:
            if g > a:
                continue
            for space, neg, (e_lo, e_hi) in branches:
                try:
                    plans = R.route_pole(
                        a_ry=a / ryd, gamma_ry=g / ryd, space=space,
                        neg_omega_half=neg, omega_max_ry=5.0 / ryd,
                        e_a_min_ry=e_lo / ryd, e_a_max_ry=e_hi / ryd,
                        edge_factor=1.5, rel_tol=1e-8, max_nodes=max_nodes)
                except R.RoutingRefusal as exc:
                    refused.append((g, a, space, neg, exc))
                    continue
                served += 1
                betas += [p.beta for p in plans if p.beta is not None]
    return served, refused, betas


def test_no_pole_of_the_audited_field_refuses_on_the_width_clause():
    """THE DOOR IS OPEN, and this is the cell that says so by measurement.

    Before the width catalog landed, every Laplace window on this path refused
    on an empty family whatever its beta.  With ``complex_laplace_width/v1``
    shipped at ``beta_max = 1.0``, the whole measured field must route without
    a single clause refusal -- so a width-clause refusal on si_mpa_0808's field
    is now a REGRESSION, not a gap, and that is what this asserts.
    """
    served, refused, betas = _route_the_audited_box()
    assert served > 0 and betas
    clause_refusals = [r for r in refused
                       if r[4].code not in ("bandwidth_above_cap",)]
    assert not clause_refusals, (
        "a pole of the audited field was refused for a reason other than the "
        "crossing bandwidth cap: "
        + "; ".join(f"Gamma={g:g} eV a={a:g} eV {sp},neg={n}: {e.code}"
                    for g, a, sp, n, e in clause_refusals))
    worst = max(betas)
    assert worst <= R.SHIPPED_WIDTH_BETA_MAX, (
        f"worst width clause asked for is beta={worst:.6f}, above the shipped "
        f"edge {R.SHIPPED_WIDTH_BETA_MAX}")
    # Measured, and recorded so a drift is visible rather than merely absent:
    # 0.694 is the worst the box asks for, against a clause edge of 1.0.
    assert 0.60 < worst < 0.75


def test_the_only_refusing_corner_is_the_fields_narrowest_width():
    """And it is a cost statement, not a catalog gap.

    The one corner that refuses is the field's ZEROTH percentile width,
    Gamma = 3.96e-5 eV, on the crossing core, at a dimensionless bandwidth
    A = 2.4e5.  It is not served at any node count -- raising the cap to
    131 072 does not help -- because resolving a quarter of a million
    oscillations is not the right answer at any price.  So the refusal is
    correct, it names the generator, and it is confined: the audit's own
    percentiles put A at or below 111 for 99 % of the worst pole and at or
    below 10 for the other seven.
    """
    _, refused, _ = _route_the_audited_box()
    assert refused, "the narrow-width corner should still refuse"
    for g, _a, _sp, _neg, exc in refused:
        assert exc.code == "bandwidth_above_cap"
        assert exc.generator
        assert "generate_imag_minimax_assets" in exc.generator
        assert g == min(AUDIT_GAMMA_EV), (
            f"a width other than the field's narrowest ({g} eV) refused; the "
            f"corner is supposed to be the vanishing-width one alone")
        assert exc.a_dim > 1e4

    # Raising the cap by five octaves does not move it, which is what makes it
    # a physics statement rather than a budget one.
    _, refused_big, _ = _route_the_audited_box(max_nodes=1 << 17)
    assert len(refused_big) == len(refused)

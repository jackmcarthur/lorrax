"""The MPA pass loop: its signs, its narrow-pole routing, and its lemma.

WHAT THIS FILE SCORES, AND WHY IT IS NOT THE ROUTING FILE'S JOB.
``tests/test_mpa_sigma_routing.py`` scores each quadrature against the
analytic object it is supposed to reproduce -- the crossing sum against
``1/(u + i*Gamma)``, the sign-definite sum against ``1/(w - i*Gamma)`` --
and plants a conjugated pole and a swapped placement against each.  Every
one of those cells passes with the crossing core's PREFACTOR set either
way, because a prefactor is not part of the object being scored: GATE 1
compares ``sum alpha e^{iut}`` to ``exact_resolvent`` with no prefactor in
the expression at all.

That gap was not hypothetical.  The core window shipped with
``prefactor = +1.0 * neg`` while the two-point path's crossing core, the
two-point path's three Laplace windows, and this module's own ``single``,
``a_stripe`` and ``b_slab`` all carry the OPPOSITE convention: their raw
quadrature sums reproduce ``+1/(omega - S)`` and their prefactors turn it
into ``-1/(omega - S)``.  The measurement that settles it is in
:func:`test_the_legacy_crossing_core_and_the_mpa_one_carry_the_same_sign`
below -- the HGL sine sum reproduces ``G_hgl(u) -> +1/u`` and carries
``-1``, so a complex core reproducing ``+1/(u + i*Gamma)`` owes ``-1`` too.
The symptom of the bug was a finite, smooth, correctly-shaped Sigma with
the crossing contribution entering backwards, which is exactly the class of
defect this campaign has hit four times.

So the gates here are RELATIVE ones.  Every cell compares a composite of
window objects against the analytic pole sum they are supposed to add up
to, which is the only comparison that can see a sign that is wrong only
between windows.
"""

import numpy as np
import pytest

from gw.mpa import sigma_pass as SP
from gw.mpa import sigma_routing as R
from gw.ppm_accumulators import _project_tau_onto_omega_np


RYD = 13.605693122994


# ---------------------------------------------------------------------------
#  The scalar analogue of the device tau loop.
#
#  Every HOST object below is the production one: the window objects come
#  from ``plan_branch_groups``, the per-tau weight is
#  ``alpha * exp(-i*E_ref_sum*t)`` exactly as ``minimax_tau_integrate_sigma``
#  forms it, and the omega projection is ``_project_tau_onto_omega_np``, the
#  single omega projector in the tree.  Only the device contraction --
#  G(tau)*W(tau), the FFT round trip and the psi projection -- is replaced
#  by its one-element analogue, which is legal because that contraction is
#  linear in W(tau) and the whole question here is what multiplies it.
# ---------------------------------------------------------------------------

def _sigma_tau_scalar(t, *, E_A, mask_A, omega_operand, B, mask_B,
                      e_ref_a, e_ref_b):
    """``sigma(tau)`` for the one-band, one-mode analogue of the kernel."""
    g = np.sum(np.where(mask_A, np.exp(-1j * (E_A - e_ref_a) * t), 0.0 + 0.0j))
    w = np.sum(np.where(mask_B,
                        B * np.exp(-1j * (omega_operand - e_ref_b) * t),
                        0.0 + 0.0j))
    return g * w


def _integrate_group(group, omega_vec, *, E_A, mask_A, B):
    """One window group's contribution at every omega on the branch."""
    out = np.zeros((omega_vec.size,), dtype=np.complex128)
    for win in group.windows:
        t_host = np.asarray(win.nodes.t, dtype=np.complex128)
        alpha = np.asarray(win.nodes.alpha, dtype=np.complex128)
        alpha_eff = alpha * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_host)
        for i in range(t_host.size):
            sig = _sigma_tau_scalar(
                t_host[i], E_A=E_A, mask_A=mask_A & win.mask_A,
                omega_operand=group.omega_operand, B=B, mask_B=group.dense_mask_B(),
                e_ref_a=win.E_ref_A, e_ref_b=win.E_ref_B)
            if win.project_code == 0:
                s_re, s_im = np.asarray([[[sig]]]), None
            else:
                s_re = np.asarray([[[np.real(sig)]]])
                s_im = np.asarray([[[np.imag(sig)]]])
            out += _project_tau_onto_omega_np(
                s_re, s_im, omega_vec, complex(t_host[i]),
                complex(alpha_eff[i]), float(win.omega_sign),
                float(win.prefactor), int(win.project_code))[:, 0, 0, 0]
    return out


def _sigma_through_the_loop(*, omega_ry, E_cond, E_val, omega_p, b_p, xi_ry,
                            edge_factor=1.5, order=None, rel_tol=1e-10,
                            binned_width_clause=None):
    """Run the real planner over the real branches, one pole per pass.

    ``omega_p`` / ``b_p`` are ``(n_p, n_modes)``; the pass loop walks the
    first axis, which is the store's pole axis.
    """
    omega_ry = np.asarray(omega_ry, dtype=np.float64)
    idx_pos = np.where(omega_ry >= 0.0)[0]
    idx_neg = np.where(omega_ry < 0.0)[0]
    total = np.zeros(omega_ry.shape, dtype=np.complex128)
    walk = range(omega_p.shape[0]) if order is None else order
    for p in walk:
        a = np.real(omega_p[p]).astype(np.float64)
        g = np.abs(np.imag(omega_p[p])).astype(np.float64)
        live = np.ones(a.shape, dtype=bool)
        for space, neg, E_A, idx in (
                ("cond", False, E_cond, idx_pos),
                ("val", False, E_val, idx_pos),
                ("cond", True, E_cond, idx_neg),
                ("val", True, E_val, idx_neg)):
            if idx.size == 0:
                continue
            omega_abs = np.abs(omega_ry[idx])
            groups, _ = plan(
                a=a, g=g, live=live, E_A=E_A, omega_abs=omega_abs,
                space=space, neg=neg, xi_ry=xi_ry, edge_factor=edge_factor,
                rel_tol=rel_tol,
                binned_width_clause=binned_width_clause)
            mask_A = np.ones(E_A.shape, dtype=bool)
            for grp in groups:
                total[idx] += _integrate_group(
                    grp, omega_abs, E_A=E_A, mask_A=mask_A, B=b_p[p])
    return total


def plan(*, a, g, live, E_A, omega_abs, space, neg, xi_ry, edge_factor,
         rel_tol, binned_width_clause=None):
    """``plan_branch_groups`` with the test's fixed quadrature knobs."""
    return SP.plan_branch_groups(
        a_ry=a, gamma_ry=g, live_mask=live, E_A_host=E_A,
        base_mask_A_host=np.ones(E_A.shape, dtype=bool),
        omega_nonneg_ry=omega_abs, space=space, neg_omega_half=neg,
        xi_ry=xi_ry, edge_factor=edge_factor, rel_tol=rel_tol,
        binned_width_clause=binned_width_clause,
        target_error=1e-8, laplace_max_nodes=64, crossing_eps_q=1e-10,
        crossing_max_nodes=400, use_shipped_minimax_tables=True,
        log_tag="", print_fn=lambda *a, **k: None)


def _analytic_sigma(omega_ry, E_cond, E_val, omega_p, b_p):
    """The fused all-pole evaluation -- the closed form the loop must equal.

    The tree's convention, established by every window in it and measured
    in the cell below: ``Sigma_c`` is MINUS the naive pole sum.
    """
    # Written out plainly rather than broadcast-golfed: the sum is over
    # (A state, pole, mode) and there is nothing to be gained by hiding it.
    out = np.zeros(np.shape(omega_ry), dtype=np.complex128)
    for iw, om in enumerate(np.asarray(omega_ry, dtype=np.float64)):
        acc = 0.0 + 0.0j
        for p in range(omega_p.shape[0]):
            for m in range(omega_p.shape[1]):
                acc -= np.sum(b_p[p, m] / (om - E_cond - omega_p[p, m]))
                acc -= np.sum(b_p[p, m] / (om + E_val + omega_p[p, m]))
        out[iw] = acc
    return out


# ---------------------------------------------------------------------------
#  GATE 1 -- the sign, measured between the two crossing cores
# ---------------------------------------------------------------------------

def test_the_legacy_crossing_core_and_the_mpa_one_carry_the_same_sign():
    """The measurement that fixes the core prefactor, kept as a cell.

    The two-point crossing core fits ``G_hgl``, whose sine sum tends to
    ``+1/u``; it carries ``prefactor = -1`` on the ``+omega`` half.  The MPA
    crossing core's composite sum tends to ``+1/(u + i*Gamma)`` -- the same
    sign of the same object -- so it owes the same ``-1``.  Both halves are
    measured here rather than asserted, because the whole point is that
    neither is obvious from reading either module.
    """
    from gw.minimax_screening import solve_phase_minimax_bandwidth

    q = solve_phase_minimax_bandwidth(
        24.0, target_error=1e-6, max_nodes=200, eps_q=1e-10,
        target_kind="hgl", use_shipped_tables=True)
    tau = np.asarray(q.tau, dtype=np.float64)
    al = np.asarray(q.alpha, dtype=np.float64)
    for u in (5.0, 10.0, 20.0):
        got = float(np.sum(al * np.sin(u * tau)))
        assert got > 0.0
        assert abs(got * u - 1.0) < 1e-2, (
            f"the HGL sine sum at u={u} is {got:.6f}, not +1/u -- the "
            f"legacy crossing core's sign premise moved")

    g, f = 1.0 / RYD, 12.0 / RYD
    t, alpha, _ = R.crossing_rule(g, f, rel_tol=1e-10)
    u = np.linspace(0.2 * f, f, 32)
    got = np.sum(alpha[None, :] * np.exp(1j * u[:, None] * np.real(t)[None, :])
                 * np.exp(-g * np.real(t))[None, :], axis=-1)
    assert np.allclose(got, 1.0 / (u + 1j * g), rtol=1e-6)
    assert np.all(np.real(got) > 0.0)

    # And therefore: the two cores' prefactors must be equal.
    plans = R.route_pole(
        a_ry=3.0 / RYD, gamma_ry=1.0 / RYD, space="cond",
        neg_omega_half=False, omega_max_ry=5.0 / RYD,
        e_a_min_ry=0.35 / RYD, e_a_max_ry=56.0 / RYD, edge_factor=1.5)
    core = [p for p in plans if p.name == "core"][0]
    assert core.prefactor == -1.0, (
        "the MPA crossing core's prefactor is not the two-point core's "
        "-1; the crossing contribution would enter Sigma backwards")


@pytest.mark.parametrize("space,neg", [("cond", False), ("val", False),
                                       ("cond", True), ("val", True)])
def test_every_window_of_a_pass_sums_to_the_analytic_pole_term(space, neg):
    """All four branches, window composite against the closed form.

    This is the cell the prefactor bug would have failed and the routing
    file's cells could not: it scores the SUM of the windows a branch
    returns against ``-B/(omega -/+ (E_A + Omega))``, so a window whose own
    quadrature is perfect but whose sign is wrong relative to its
    neighbours shows up immediately.
    """
    E_A = np.array([1.0, 6.0, 20.0]) / RYD
    a = np.array([3.0, 25.0]) / RYD
    g = np.array([1.2, 4.0]) / RYD
    B = np.array([0.7 - 0.2j, -0.3 + 0.11j])
    xi = 0.5 / RYD
    omega_abs = np.linspace(0.0, 5.0 / RYD, 6)

    groups, stats = plan(a=a, g=g, live=np.ones(2, dtype=bool), E_A=E_A,
                         omega_abs=omega_abs, space=space, neg=neg,
                         xi_ry=xi, edge_factor=1.5, rel_tol=1e-10)
    assert stats["n_wide"] == 2 and stats["n_narrow"] == 0
    got = np.zeros(omega_abs.shape, dtype=np.complex128)
    mask_A = np.ones(E_A.shape, dtype=bool)
    for grp in groups:
        got += _integrate_group(grp, omega_abs, E_A=E_A, mask_A=mask_A, B=B)

    # THE FORM IS THE A-SPACE'S, NOT THE CROSSING PREDICATE'S.  The empty
    # A-space always contributes ``1/(omega_rel - (E_c + Omega))`` and the
    # occupied one always ``1/(omega_rel + (h_v + Omega))``; which of them
    # can vanish -- which one CROSSES -- is decided by the sign of
    # ``omega_rel``, which is why the crossing role moves between the two
    # halves.  Writing the target off the crossing predicate instead is the
    # mistake that makes the -omega half look broken when it is not.
    Om = a - 1j * g
    omega_rel = (-1.0 if neg else 1.0) * omega_abs
    want = np.zeros_like(got)
    for m in range(2):
        for e in E_A:
            denom = ((omega_rel - (e + Om[m])) if space == "cond"
                     else (omega_rel + (e + Om[m])))
            want += -B[m] / denom
    scale = np.max(np.abs(want))
    err = np.max(np.abs(got - want)) / scale
    assert err < 2e-6, f"branch {space}/neg={neg} composite is off by {err:.3e}"


# ---------------------------------------------------------------------------
#  GATE 2 -- the narrow-pole routing decision, and its red twin
# ---------------------------------------------------------------------------

def test_a_pole_either_side_of_xi_takes_the_branch_it_should():
    """THE RED TWIN THE ROUTING DECISION OWES.

    Two poles identical in every respect but their width, engineered to sit
    one ulp-scale step either side of ``xi``.  The narrow one must be
    summed by the two-point crossing machinery (a window carrying the HGL
    ``project='imag'`` projection, built at ``xi``) and the wide one by the
    complex composite (``project='full'`` throughout).  A threshold that
    silently rounded, or a split that sent both the same way, fails here.
    """
    E_A = np.array([1.0, 6.0]) / RYD
    xi = 0.5 / RYD
    omega_abs = np.linspace(0.0, 5.0 / RYD, 4)
    a = np.array([3.0 / RYD, 3.0 / RYD])

    below = np.array([xi * (1.0 - 1e-9), xi * (1.0 - 1e-9)])
    above = np.array([xi * (1.0 + 1e-9), xi * (1.0 + 1e-9)])

    g_lo, s_lo = plan(a=a, g=below, live=np.ones(2, dtype=bool), E_A=E_A,
                      omega_abs=omega_abs, space="cond", neg=False,
                      xi_ry=xi, edge_factor=1.5, rel_tol=1e-8)
    g_hi, s_hi = plan(a=a, g=above, live=np.ones(2, dtype=bool), E_A=E_A,
                      omega_abs=omega_abs, space="cond", neg=False,
                      xi_ry=xi, edge_factor=1.5, rel_tol=1e-8)

    assert s_lo["n_narrow"] == 2 and s_lo["n_wide"] == 0
    assert s_hi["n_narrow"] == 0 and s_hi["n_wide"] == 2
    assert [grp.name for grp in g_lo] == ["legacy"]
    assert all(grp.name != "legacy" for grp in g_hi)
    # The discriminating signature, not merely "the group name changed":
    # the legacy route is the one that carries an HGL crossing window, and
    # the complex route is the one that never does.
    assert any(w.crossing_kind == "hgl" for grp in g_lo for w in grp.windows)
    assert all(w.crossing_kind is None for grp in g_hi for w in grp.windows)
    assert all(w.project == "full" for grp in g_hi for w in grp.windows)


def test_the_narrow_corner_that_refused_is_served_by_the_split():
    """The audit's 2.4e5-bandwidth corner, routed instead of refused.

    ``Gamma = 3.959e-5 eV`` on a crossing branch is the field's zeroth
    percentile and the one corner ``route_pole`` refuses at any node count.
    Through the split it is served -- by the machinery whose smearing is
    four orders wider than the width it is being asked to resolve, which is
    the whole argument -- and the refusal is gone.
    """
    E_A = np.array([0.3464, 20.0]) / RYD
    a = np.array([0.2647 / RYD])
    g_narrow = np.array([3.959e-5 / RYD])
    omega_abs = np.linspace(0.0, 5.0 / RYD, 4)
    xi = SP.narrow_pole_threshold_ry(0.25 / RYD, 5.0 / RYD, 1.5)
    assert xi > 20.0 * float(g_narrow[0]), "the smearing must dominate"

    with pytest.raises(R.RoutingRefusal) as exc:
        R.route_pole(a_ry=float(a[0]), gamma_ry=float(g_narrow[0]),
                     space="cond", neg_omega_half=False,
                     omega_max_ry=5.0 / RYD, e_a_min_ry=float(E_A[0]),
                     e_a_max_ry=float(E_A[-1]), edge_factor=1.5)
    assert exc.value.code == "bandwidth_above_cap"
    assert exc.value.a_dim > 1e4

    groups, stats = plan(a=a, g=g_narrow, live=np.ones(1, dtype=bool),
                         E_A=E_A, omega_abs=omega_abs, space="cond",
                         neg=False, xi_ry=xi, edge_factor=1.5, rel_tol=1e-8)
    assert stats["n_narrow"] == 1 and groups
    assert [grp.name for grp in groups] == ["legacy"]


def test_the_threshold_is_the_smearing_the_legacy_path_actually_uses():
    """Not the deck key: the deck key AFTER the conditioning floor.

    The floor engages on every default run, taking the requested 0.25 eV to
    0.476 eV on the production grid.  Splitting on the requested value
    would put every pole between the two on the complex route while the
    route it is compared against had already smeared them.
    """
    requested = 0.25 / RYD
    xi = SP.narrow_pole_threshold_ry(requested, 5.0 / RYD, 1.5)
    assert xi > requested
    assert 0.47 < xi * RYD < 0.48, f"xi = {xi * RYD:.4f} eV"
    # And with no crossing pane to condition, the deck's own value stands.
    assert SP.narrow_pole_threshold_ry(requested, 0.0, 1.5) == requested


# ---------------------------------------------------------------------------
#  GATE 3 -- the fourteen-pass lemma, run through the production loop
# ---------------------------------------------------------------------------

def _field(n_p=4, n_modes=3, seed=11):
    """A synthetic pole field INSIDE the fitter's own fourth guard.

    ``pade_fit``'s fourth guard caps ``|Im Omega| <= width_ratio_max *
    Re Omega`` at ``width_ratio_max = 1``, and the whole width-clause
    envelope is derived from it.  A test field that draws ``Gamma``
    independently of ``a`` produces poles the fitter cannot emit, and the
    slab clause then refuses -- correctly, and for a reason that has
    nothing to do with the loop.  (It did, on the first run of this cell.)
    """
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(2.0, 30.0, size=(n_p, n_modes))) / RYD
    g = a * rng.uniform(0.05, 0.9, size=(n_p, n_modes))
    B = (rng.normal(size=(n_p, n_modes))
         + 1j * rng.normal(size=(n_p, n_modes))) * 0.3
    return a - 1j * g, B


def test_the_pass_loop_reproduces_the_fused_all_pole_evaluation():
    """GATE (b): pass-by-pass against the closed form, no deck involved.

    Every host object on the path is the production one -- the planner, the
    window objects, the per-tau weights, the single omega projector -- and
    the reference is the analytic all-pole sum, evaluated fused.  The
    accuracy the loop is allowed is its quadrature's own, and nothing more.
    """
    omega_p, b_p = _field()
    E_cond = np.array([1.0, 4.5, 12.0]) / RYD
    E_val = np.array([0.6, 3.2]) / RYD
    omega_ry = np.linspace(-4.0, 4.0, 9) / RYD
    xi = 0.5 / RYD

    got = _sigma_through_the_loop(
        omega_ry=omega_ry, E_cond=E_cond, E_val=E_val, omega_p=omega_p,
        b_p=b_p, xi_ry=xi)
    want = _analytic_sigma(omega_ry, E_cond, E_val, omega_p, b_p)
    err = np.max(np.abs(got - want)) / np.max(np.abs(want))
    assert err < 5e-7, f"pass loop vs fused all-pole: {err:.3e}"


def test_the_re_association_is_the_only_difference_the_pass_order_makes():
    """The lemma's one caveat, exhibited rather than remembered.

    Walking the poles in the reverse order computes the same sum with a
    different association.  It must agree to floating-point association and
    NOT be bit-identical -- if it were bit-identical the pinned order would
    be decoration, and if it moved by more than a few ulps the lemma would
    be false.
    """
    omega_p, b_p = _field(seed=23)
    E_cond = np.array([1.0, 9.0]) / RYD
    E_val = np.array([0.8]) / RYD
    omega_ry = np.linspace(-3.0, 3.0, 5) / RYD
    xi = 0.5 / RYD

    fwd = _sigma_through_the_loop(
        omega_ry=omega_ry, E_cond=E_cond, E_val=E_val, omega_p=omega_p,
        b_p=b_p, xi_ry=xi, order=range(omega_p.shape[0]))
    rev = _sigma_through_the_loop(
        omega_ry=omega_ry, E_cond=E_cond, E_val=E_val, omega_p=omega_p,
        b_p=b_p, xi_ry=xi, order=list(range(omega_p.shape[0]))[::-1])
    rel = np.max(np.abs(fwd - rev)) / np.max(np.abs(fwd))
    assert rel < 1e-13, f"re-association moved the answer by {rel:.3e}"


# ---------------------------------------------------------------------------
#  GATE 4 -- the slab, and the two refusals that keep it honest
# ---------------------------------------------------------------------------

def test_the_laplace_buckets_bound_the_interval_ratio():
    """A four-decade pole field costs a handful of buckets, not one per octave."""
    a = np.array([0.2647, 2.6, 23.09, 1112.0, 6451.0]) / RYD
    buckets = SP._laplace_buckets(
        a, e_lo=0.3464 / RYD,
        e_hi=56.36 / RYD, omega_max=5.0 / RYD, r_max=100.0)
    assert 1 <= len(buckets) <= 5, buckets
    covered = np.zeros(a.shape, dtype=bool)
    for lo, hi in buckets:
        covered |= (a >= lo) & (a <= hi)
        # Either the ratio edge bound holds, or the bucket took the
        # geometric step it takes when the crossing pane has already
        # floored x_min and the ratio is not what the rule pays.
        x_min = max(0.3464 / RYD + lo - 5.0 / RYD, 1e-12)
        x_max = 56.36 / RYD + hi + 5.0 / RYD
        assert (x_max / x_min <= 100.0 * (1.0 + 1e-9)
                or hi <= lo * 100.0 * (1.0 + 1e-9))
    assert covered.all(), "a pole fell outside every bucket"


def test_the_q_axis_verdict_names_the_three_cases():
    """Full BZ, declared wedge of THIS zone, and neither.

    The refusal this replaces (``refuse_wedge_pole_slab``) turned away
    every store with ``n_q != n_k_tot``.  What is left to refuse is
    narrower and is about IDENTITY rather than about the unfold: a store
    whose zone is not this run's zone cannot be unfolded into it, and
    reading its rows as if it were would sum a different crystal's
    screening.
    """
    full = {"n_q": 64, "n_q_full": 64, "q_storage": "full"}
    wedge = {"n_q": 8, "n_q_full": 64, "q_storage": "ibz"}
    assert SP.resolve_pole_q_axis(full, 64) is False
    assert SP.resolve_pole_q_axis(wedge, 64) is True

    # A wedge of a DIFFERENT zone: 8 of 32 offered to a 64-point run.
    with pytest.raises(ValueError) as exc:
        SP.resolve_pole_q_axis({"n_q": 8, "n_q_full": 32,
                                "q_storage": "ibz"}, 64)
    assert "neither that zone nor a wedge of it" in str(exc.value)

    # An undeclared short axis: no tables, so no way to know what it is.
    with pytest.raises(ValueError) as exc:
        SP.resolve_pole_q_axis({"n_q": 8, "n_q_full": 8,
                                "q_storage": "full"}, 64)
    assert "stamp_fit_unfold_tables" in str(exc.value)


def test_the_pass_report_names_the_legacy_routed_count():
    """The announcement the routing decision owes the reader."""
    recs = [SP.PassRecord(pole_index=0, n_legacy_modes=3, n_mpa_modes=97,
                          n_tau_nodes=210)]
    text = SP.format_pass_report(recs, xi_ev=0.476)
    assert "legacy-routed modes: 3 of 100 (3.00 %)" in text
    assert "xi = 0.4760 eV" in text


# ---------------------------------------------------------------------------
#  The process axis _to_host_np prepends, and the pass loop that died on it
# ---------------------------------------------------------------------------

def test_the_gathered_mask_keeps_the_shape_it_was_gathered_from():
    """A single-process gather must not change an array's rank.

    THE MEASURED FAILURE.  ``ppm_windows._to_host_np(x, tiled=False)`` goes
    through ``process_allgather``, which prepends a length-``process_count``
    axis even when that count is one, so a ``(64, 1128, 1128)`` mask comes
    back ``(1, 64, 1128, 1128)``.  The first consumer in the pass loop is
    ``np.min(a_host, where=live, initial=np.inf)`` with ``a_host`` rank 3,
    and numpy refuses a rank-4 ``where`` with "input operand has more
    dimensions than allowed by the axis remapping".  The two-point path
    never met this because its only consumer of a gathered array ends in
    ``.reshape(-1)[0]``, to which a leading 1 is invisible.
    """
    src = np.zeros((3, 4, 5), dtype=bool)

    def gather_with_process_axis(a, *, dtype, tiled):
        assert tiled is False
        return np.asarray(a, dtype=dtype)[None, ...]

    out = SP._host_at_source_shape(src, bool, gather_with_process_axis)
    assert out.shape == (3, 4, 5)
    assert out.dtype == np.dtype(bool)
    # and the reduction that died now runs
    a = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    live = np.ones_like(src, dtype=bool)
    live_g = SP._host_at_source_shape(live, bool, gather_with_process_axis)
    assert float(np.min(a, where=live_g, initial=np.inf)) == 0.0


def test_a_gather_that_is_not_a_reshape_is_refused():
    """The FALSE case: this strips a process axis, not any leading axis.

    A gather that came back with a different ELEMENT COUNT is not the
    process axis at all -- it is a different object, and reshaping it would
    either raise somewhere unhelpful or, worse, succeed against a stale
    buffer.  So the mismatch is named here.
    """
    src = np.zeros((3, 4, 5), dtype=bool)

    def gather_wrong_count(a, *, dtype, tiled):
        return np.zeros((2,) + np.shape(a), dtype=dtype)

    with pytest.raises(ValueError) as exc:
        SP._host_at_source_shape(src, bool, gather_wrong_count)
    assert "not a reshape of it" in str(exc.value)


def test_a_gather_that_already_has_the_right_shape_is_untouched():
    """Multi-process arrays come back at global shape; identity, not a bug."""
    src = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    def gather_global(a, *, dtype, tiled):
        return np.asarray(a, dtype=dtype)

    out = SP._host_at_source_shape(src, np.float64, gather_global)
    assert out.shape == src.shape
    assert np.array_equal(out, src)

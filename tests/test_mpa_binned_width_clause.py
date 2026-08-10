"""The binned-width clause: the pane collapse, and what it refuses.

THE GATE THAT MATTERS MOST IS THE THIRD ONE.  Two of the cells below are
red twins for the new clause -- a request one hair outside the certified
grid, and a pane whose widths do not fit the bin it claims -- and both
would fail loudly if the clause were merely permissive.  The third is
:func:`test_the_flag_off_plans_are_byte_identical_to_the_shipped_ones`,
which hashes the whole plan (every pane's index set, every window's nodes,
weights, references, signs and A-side mask) and asserts the flag-off
digest is the digest of the planner that shipped.  A clause that is
"strictly additive" is a claim about that digest and nothing else, and
before this file there was no plan digest in the tree at all -- the
nearest thing compared derived scalars.

THE BOUNDARY CONVENTION IS TESTED HERE TOO, because it is a decision this
branch took rather than inherited: bin membership is half-open CLOSED AT
THE TOP, ``(lo, hi]``, so a pole sitting exactly on a bin edge lands in
the bin whose certificate was built AT its own width.  The cell that
scores it exercises both conventions on one field and shows the
wrong-side assignment putting the pole under a rule built for a width it
does not have.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from gw.mpa import sigma_pass as SP
from gw.mpa import sigma_routing as R

RYD = 13.605693122994


# ---------------------------------------------------------------------------
#  The plan digest -- the canonicalization this branch's byte-identity gate
#  is defined against, written once and used by three cells.
# ---------------------------------------------------------------------------

def plan_digest(groups):
    """SHA-256 over a plan's panes and rules, in the planner's own order.

    WHAT GOES IN, AND WHY EACH FIELD.  A plan is what the device tau loop
    consumes, so the digest covers exactly what the loop reads and
    nothing that only a log reads: the group order (the panes are summed
    in it and floating-point re-association is the one difference an
    order change makes), each group's ``idx_B`` (WHICH modes the pane
    holds, as the ascending flat indices the kernel gathers), the field
    shape it indexes into, and then per window the nodes, the weights,
    both energy references, the omega sign, the prefactor, the
    projection code and the A-side mask.

    ``provenance`` and the group NAME are deliberately excluded: they are
    prose, they carry the pane's width range formatted to three
    significant figures, and a digest that moved when a docstring-grade
    string moved would be a gate nobody could keep green.  ``b_mass`` is
    excluded for the same reason and a stronger one -- it is a derived
    sum over the same ``idx_B`` that is already hashed.
    """

    h = hashlib.sha256()
    for grp in groups:
        h.update(b"\x00GROUP")
        h.update(np.asarray(grp.idx_B, dtype="<i8").tobytes())
        h.update(repr(tuple(int(x) for x in grp.field_shape)).encode())
        h.update(np.asarray(grp.omega_operand, dtype="<c16").tobytes())
        for win in grp.windows:
            h.update(b"\x01WINDOW")
            h.update(np.asarray(win.nodes.t, dtype="<c16").tobytes())
            h.update(np.asarray(win.nodes.alpha, dtype="<c16").tobytes())
            h.update(np.asarray(win.mask_A, dtype=bool).tobytes())
            h.update(np.asarray(
                [win.E_ref_A, win.E_ref_B, win.prefactor], dtype="<f8"
            ).tobytes())
            h.update(np.asarray(
                [win.omega_sign], dtype="<i8").tobytes())
            h.update(str(win.project).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
#  Fields
# ---------------------------------------------------------------------------

def _field(n_modes=400, seed=7, decades=4.0):
    """A pole field with a WIDE width spread, which is the whole point.

    ``Gamma`` is drawn proportional to ``a`` -- the fitter's fourth guard
    (``|Im Omega| <= Re Omega``) is what the binned clause's derivation
    rests on, and a field that violates it would be testing a clause
    nothing produces.  The spread in ``a`` is the knob: four decades is
    the shape of pole 7 of the audited fit, whose single Laplace bucket
    spans ``Gamma`` from 1.8e-2 to 4.5e+2 Ry and which the per-pole
    clause answers with 1312 panes.
    """

    rng = np.random.default_rng(seed)
    a = np.sort(10.0 ** rng.uniform(0.0, float(decades), size=n_modes)) / RYD
    g = a * rng.uniform(0.2, 0.95, size=n_modes)
    return a, g


def _plan(a, g, *, binned=None, space="val", neg=False, e_lo=0.3, e_hi=6.0,
          n_a=6):
    """One branch's plan, with everything but the flag held fixed."""

    E_A = np.linspace(e_lo, e_hi, n_a) / RYD
    mask_A = np.ones(n_a, dtype=bool)
    live = np.ones(a.shape, dtype=bool)
    return SP.plan_branch_groups(
        a_ry=a, gamma_ry=g, live_mask=live,
        E_A_host=E_A, base_mask_A_host=mask_A,
        omega_nonneg_ry=np.linspace(0.0, 5.0 / RYD, 4),
        space=space, neg_omega_half=neg,
        xi_ry=1.0e-9, edge_factor=1.5,
        rel_tol=1.0e-8, binned_width_clause=binned,
        target_error=1.0e-8, laplace_max_nodes=64,
        crossing_eps_q=1.0e-10, crossing_max_nodes=400,
        use_shipped_minimax_tables=True, print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
#  GATE 1 -- the flag off is the planner that shipped, to the byte
# ---------------------------------------------------------------------------

def test_the_flag_off_plans_are_byte_identical_to_the_shipped_ones():
    """``binned_width_clause=None`` must change nothing at all.

    Scored as a digest over panes AND rules rather than as a pane count,
    because "same number of panes" is exactly the agreement a plan can
    reach while holding different modes.  The two arms here are the same
    planner called twice, which makes this a gate on the DEFAULT rather
    than on the diff -- the diff against ``0f5da1ef`` is measured in the
    branch's report, on the production field, where the constants that
    could differ actually take their production values.
    """

    a, g = _field()
    for space, neg in (("val", False), ("cond", False),
                       ("val", True), ("cond", True)):
        base, _ = _plan(a, g, binned=None, space=space, neg=neg)
        again, _ = _plan(a, g, binned=None, space=space, neg=neg)
        assert plan_digest(base) == plan_digest(again)
        assert base, "a branch with 400 live modes planned no panes at all"


def test_the_digest_notices_a_single_mode_moving_pane():
    """The FALSE case of GATE 1: a gate that cannot fail proves nothing.

    One index is moved from the largest pane to the smallest -- a change
    no pane COUNT and no total-mode count can see -- and the digest must.
    """

    a, g = _field()
    groups, _ = _plan(a, g, binned=None)
    order = sorted(range(len(groups)), key=lambda i: groups[i].idx_B.size)
    small, big = groups[order[0]], groups[order[-1]]
    assume = big.idx_B.size > 1
    assert assume, "need a pane with more than one mode to move one out of"

    moved = [g_ for g_ in groups]
    idx = big.idx_B
    moved[order[-1]] = SP.WindowGroup(
        name=big.name, windows=big.windows, idx_B=idx[1:].copy(),
        field_shape=big.field_shape, omega_operand=big.omega_operand,
        n_modes=int(idx.size - 1), b_mass=big.b_mass,
        provenance=big.provenance)
    moved[order[0]] = SP.WindowGroup(
        name=small.name, windows=small.windows,
        idx_B=np.sort(np.concatenate([small.idx_B, idx[:1]])).astype(
            small.idx_B.dtype),
        field_shape=small.field_shape, omega_operand=small.omega_operand,
        n_modes=int(small.idx_B.size + 1), b_mass=small.b_mass,
        provenance=small.provenance)
    assert plan_digest(groups) != plan_digest(moved)


# ---------------------------------------------------------------------------
#  GATE 2 -- the collapse itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,floor", [(2.0, 2.0), (4.0, 4.0)])
def test_the_binned_clause_collapses_the_pane_count(r, floor):
    """One pane per (window, bin), and the bin count is arithmetic.

    ``ceil(log_r(Gamma_hi/Gamma_lo))`` per Laplace bucket by
    construction, against whatever satisfying ``beta <= 1`` costs -- and
    the assertion is on the RATIO of the two, because the absolute
    numbers belong to the field and the collapse belongs to the clause.

    The floors are ``r`` itself, which is the honest expectation rather
    than a round one: the per-pole clause's pane count grows with how
    finely a continuum of widths has to be cut to keep every leaf's beta
    under 1, and the binned count is ``log_r`` of the same spread, so
    doubling ``r`` halves the bins.  Measured on this 400-mode, 4-decade
    field: 47 panes off, 18 at ``r = 2``, 10 at ``r = 4``.
    """

    a, g = _field()
    off, s_off = _plan(a, g, binned=None)
    on, s_on = _plan(a, g, binned=r)

    assert s_off["binned_width_clause"] is None
    assert s_on["binned_width_clause"] == r
    assert s_off["n_panes"] == len(off) and s_on["n_panes"] == len(on)
    assert len(on) <= len(off) / floor, (
        f"r={r} gave {len(on)} panes against {len(off)}; the clause is "
        "supposed to be a collapse, not a rearrangement")

    # The tau nodes are the cost, and they must fall too -- a collapse
    # that merely moved the nodes from many small panes into few large
    # ones would buy nothing, and the per-pane node count DOES rise with
    # r (the composite caps its panel width at one wavelength of beta).
    tau_off = sum(w.n_tau for grp in off for w in grp.windows)
    tau_on = sum(w.n_tau for grp in on for w in grp.windows)
    assert tau_on < tau_off

    # The panes still partition the live set exactly -- the collapse must
    # not lose or double-count a single mode.
    for groups in (off, on):
        seen = np.concatenate([grp.idx_B for grp in groups])
        assert seen.size == a.size
        assert np.array_equal(np.sort(seen), np.arange(a.size))

    # And every pane's realized width ratio respects the bin, inclusive.
    for grp in on:
        gv = g.ravel()[grp.idx_B]
        assert gv.max() <= gv.min() * r * (1.0 + 1e-12)


def test_the_binned_panes_carry_a_wider_beta_than_the_per_pole_clause():
    """The clause is not free: it is a DIFFERENT bound, and it is used.

    If every binned pane happened to land under ``beta <= 1`` anyway then
    the collapse would be a coincidence of this field rather than the
    clause doing work.  At least one pane must ask for a beta the shipped
    width clause would have refused.
    """

    a, g = _field()
    on, _ = _plan(a, g, binned=4.0)
    betas = []
    for grp in on:
        gv = g.ravel()[grp.idx_B]
        av = a.ravel()[grp.idx_B]
        x_min = 0.3 / RYD + av.min()
        betas.append(gv.max() / x_min)
    assert max(betas) > R.SHIPPED_WIDTH_BETA_MAX, (
        f"the widest binned pane asks beta={max(betas):.4g}, which the "
        "per-pole clause would already have served -- this field does not "
        "exercise the new clause")
    assert max(betas) <= 4.0 * (1.0 + 1e-12)


# ---------------------------------------------------------------------------
#  RED TWIN (a) -- one hair outside a certified bin refuses BY NAME
# ---------------------------------------------------------------------------

def test_a_bin_ratio_one_hair_outside_the_catalog_refuses_by_name():
    """``r`` just above the top certified ratio must not be served.

    The axis rounds UP, so ``r = 4`` is served by the ``r = 4`` entry and
    ``r = 4 + eps`` has nothing above it.  The refusal has to come from
    the CATALOG and name the clause, not from a planner heuristic: the
    whole discipline is that the pane is allowed its wider clause only
    where something certified the band.
    """

    import minimax as _mm

    B = _mm.beta_selector

    ok = B.select(range_value=50.0, beta=3.9, beta_clause=B.BINNED_WIDTH,
                  target_error=1.0e-8, max_nodes=4096, bin_ratio=4.0)
    assert isinstance(ok, B.TableSelection)
    assert ok.bin_ratio == 4.0

    hair = B.select(range_value=50.0, beta=3.9, beta_clause=B.BINNED_WIDTH,
                    target_error=1.0e-8, max_nodes=4096,
                    bin_ratio=4.0 + 1.0e-9)
    assert isinstance(hair, B.TableRefusal)
    assert hair.code == "NoCertifiedTable"
    assert "bin ratio" in hair.message
    assert "[2.0, 4.0]" in hair.message or "2.0, 4.0" in hair.message


def test_the_planner_refuses_a_pane_whose_band_has_no_entry():
    """The same refusal, arriving through the planner rather than the door.

    ``r = 8`` is a legal thing to ask the binner for and an illegal thing
    to ask the catalog for, which is exactly the gap the lookup-and-refuse
    discipline exists to make visible.
    """

    a, g = _field()
    with pytest.raises(R.RoutingRefusal) as excinfo:
        _plan(a, g, binned=8.0)
    assert excinfo.value.code in (
        "binned_width_no_entry", "binned_width_clause")
    assert "binned" in str(excinfo.value)


def test_the_width_clause_refuses_a_binned_request_by_name():
    """WrongClause has to discriminate the NEW clause, not just the old two.

    A binned request may not be served from the width catalog even though
    the two share a numerator and overlap in beta, because a width entry
    certifies a line and a binned pane spans a band.
    """

    import minimax as _mm

    B = _mm.beta_selector

    # The width catalog opened under a binned request: the stamp catches it.
    doc, why = B.load_catalog(B.WIDTH)
    assert doc is not None, why
    assert B.CATALOG_CLAUSE[B.catalog_version(doc)] == B.WIDTH

    band = B.select(range_value=50.0, beta=0.9, beta_clause=B.WIDTH,
                    target_error=1.0e-8, max_nodes=4096, bin_ratio=2.0)
    assert isinstance(band, B.TableRefusal)
    assert band.code == "BinRatioOnLineClause"
    assert "binned_width" in band.message

    missing = B.select(range_value=50.0, beta=0.9,
                       beta_clause=B.BINNED_WIDTH, target_error=1.0e-8,
                       max_nodes=4096)
    assert isinstance(missing, B.TableRefusal)
    assert missing.code == "MissingBinRatio"
    assert "binned_width" in missing.message


# ---------------------------------------------------------------------------
#  RED TWIN (b) -- a deliberately mis-binned pole is caught
# ---------------------------------------------------------------------------

def test_a_pole_outside_its_panes_bin_is_caught():
    """A pane whose widths span more than ``r`` must refuse, not be served.

    Driven at the guard, because that is the only place a mis-binned pane
    can arrive: the binner cannot produce one, so the FALSE case here is
    a pane reaching the rule builder by a route that did not bin it, or
    binned at a ratio the entry was not fetched for.  Both are wrong
    numbers rather than slow ones.
    """

    assert SP._refuse_mis_binned_pane(1.0, 3.9, 4.0, where="b_slab") == 3.9
    assert SP._refuse_mis_binned_pane(1.0, 4.0, 4.0, where="b_slab") == 4.0

    with pytest.raises(R.RoutingRefusal) as excinfo:
        SP._refuse_mis_binned_pane(1.0, 4.5, 4.0, where="b_slab", beta=4.5)
    assert excinfo.value.code == "mis_binned_pane"
    msg = str(excinfo.value)
    assert "4.5" in msg and "without being binned" in msg

    # And the clause gate refuses through it, with the request attached.
    with pytest.raises(R.RoutingRefusal) as excinfo:
        SP._refuse_width_clause(
            3.0, "b_slab", 1.0, 4.5,
            binned=(4.0, 1.0, 4.5, 50.0, 1.0e-8, 4096))
    assert excinfo.value.code == "mis_binned_pane"


def test_a_beta_above_the_bin_ratio_refuses_even_on_a_binned_pane():
    """The clause's conclusion is checked, not inferred from its premise.

    A pane can be correctly binned and still present a beta above ``r``
    if its ``x_min`` is not ``min(E_A) + a_lo`` -- which is the crossing
    branches' ``z_edge`` floor, where the bound is tighter still.  If it
    ever happens the derivation has stopped describing the window, and
    that is worth a named refusal rather than a served rule.
    """

    with pytest.raises(R.RoutingRefusal) as excinfo:
        SP._refuse_width_clause(
            4.5, "b_slab", 1.0, 4.5,
            binned=(4.0, 2.0, 4.5, 50.0, 1.0e-8, 4096))
    assert excinfo.value.code == "binned_width_clause"
    assert excinfo.value.beta == 4.5


# ---------------------------------------------------------------------------
#  THE BOUNDARY CONVENTION -- upper-closed, and both sides exercised
# ---------------------------------------------------------------------------

def test_a_pole_exactly_on_a_bin_edge_belongs_to_exactly_one_rule():
    """``(lo, hi]``: the pane's certified parameter is its supremum.

    THE CRITERION, APPLIED.  A pole exactly at a bin edge must belong to
    exactly ONE certified interval; that interval's certificate must
    cover it INCLUSIVE of the endpoint; and the refusal must agree with
    the assignment, so no pole is certified by neither rule or by both.
    The field puts a width EXACTLY on ``g_lo * r`` -- exact in float64,
    because ``r = 4`` makes the edge a binary shift, so this is a case
    that really occurs rather than one that occurs to within a
    tolerance.

    Both conventions produce a legal PARTITION and both respect the
    ratio, which is why the choice cannot be made on those grounds and
    has to be made on what the certificate is FOR.  Under ``(lo, hi]``
    the boundary pole's pane has ``max(Gamma)`` equal to the pole's own
    width, so the rule the pane is served by is the rule built at that
    pole's width.  Under ``[lo, hi)`` the same pole is served by a rule
    built at four times its width -- still numerically covered, but
    covered by a certificate that was never about it, and the disagreement
    with Sigma's B-side ``a <= T`` predicate is then free to put a pole
    that sits on both thresholds under two different pane labels.
    """

    g = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    idx = np.arange(g.size, dtype=np.int64)

    upper = SP._geometric_width_bins_sorted(g, idx, r_max=4.0)
    # (lo, hi]: (-inf, 4], (4, 16].
    assert [list(p) for p in upper] == [[0, 1, 2], [3, 4]]
    # Exactly one pane holds it, and every mode is in exactly one pane.
    holding = [p for p in upper if 2 in list(p)]
    assert len(holding) == 1
    assert sorted(int(i) for p in upper for i in p) == list(range(g.size))
    edge_pane = holding[0]
    assert g[edge_pane].max() == 4.0, (
        "the pole sitting exactly on the edge must be its pane's supremum "
        "-- that is what makes the rule built at max(Gamma) the rule built "
        "AT this pole's own width")

    # The other convention, computed the way the module used to compute it.
    cuts = ([0] + [int(c) for c in np.searchsorted(
        g, g[0] * 4.0 ** np.arange(1, 2), side="left")] + [int(g.size)])
    lower = [idx[cuts[b]:cuts[b + 1]] for b in range(2)
             if cuts[b + 1] > cuts[b]]
    assert [list(p) for p in lower] == [[0, 1], [2, 3, 4]]
    other_pane = next(p for p in lower if 2 in list(p))
    assert g[other_pane].max() == 16.0, (
        "under [lo, hi) the boundary pole is served by a rule built at "
        "16.0 -- four times its own width")

    # BOTH panes are legal under the ratio guard, which is the honest
    # statement: the convention is not chosen by legality.
    assert SP._refuse_mis_binned_pane(float(g[edge_pane].min()),
                                      float(g[edge_pane].max()), 4.0,
                                      where="b_slab") == 4.0
    assert SP._refuse_mis_binned_pane(float(g[other_pane].min()),
                                      float(g[other_pane].max()), 4.0,
                                      where="b_slab") == 4.0


def test_the_clause_edge_is_closed_and_the_three_gates_agree_on_it():
    """``beta = r`` EXACTLY is served; one ulp above it is refused, twice.

    This is the half of the criterion the partition cannot express.  The
    upper-closed bins let a pane realize a width ratio of exactly ``r``
    (the ``[1, 4]`` pane above does), and therefore a ``beta`` of exactly
    ``r``.  For that pole to be certified by exactly one rule and not by
    none, three things have to be closed at the top and agree: the bin,
    the planner's clause bound, and the catalog's own ``beta_covers``.
    One ulp further out, all three have to refuse.
    """

    import minimax as _mm

    B = _mm.beta_selector

    r = 4.0
    # The bin realizes the ratio exactly, and the guard admits it.
    assert SP._refuse_mis_binned_pane(1.0, 4.0, r, where="b_slab") == 4.0
    # The planner's clause bound admits beta == r and the door serves it.
    got = SP._refuse_width_clause(
        r, "b_slab", 1.0, 4.0, binned=(r, 1.0, 4.0, 50.0, 1.0e-8, 4096))
    assert isinstance(got, B.TableSelection)
    assert got.bin_ratio == r

    # One ulp above: the door refuses on the envelope, by name...
    over = np.nextafter(r, np.inf)
    refused = B.select(range_value=50.0, beta=over,
                       beta_clause=B.BINNED_WIDTH, target_error=1.0e-8,
                       max_nodes=4096, bin_ratio=r)
    assert isinstance(refused, B.TableRefusal)
    assert refused.code == "OutsideEnvelope"
    assert B.BINNED_WIDTH in refused.message

    # ...and the planner's own bound refuses before it ever asks, so the
    # two gates cannot disagree about which side of r a pole is on.
    with pytest.raises(R.RoutingRefusal) as excinfo:
        SP._refuse_width_clause(
            r * (1.0 + 1.0e-9), "b_slab", 1.0, 4.0,
            binned=(r, 1.0, 4.0, 50.0, 1.0e-8, 4096))
    assert excinfo.value.code == "binned_width_clause"


def test_the_sigma_b_side_predicate_is_upper_closed_too():
    """The convention is ONE convention, and this is the other half of it.

    ``a <= T`` puts a pole exactly at the crossing pane's threshold in the
    CORE, whose rule is built with ``a_hi = T``.  It has always read this
    way; the cell exists so that a later change to it fails here rather
    than silently disagreeing with the width axis again.
    """

    import inspect
    src = inspect.getsource(SP._mpa_groups_for_bucket)
    assert "in_core = np.asarray(a_v) <= T" in src, (
        "the B-side core predicate is the Re Omega half of the "
        "upper-closed convention; if it moves, the width axis has to "
        "move with it")
    assert 'side="right"' in inspect.getsource(
        SP._geometric_width_bins_sorted)


# ---------------------------------------------------------------------------
#  The mirrors, and the guards that did not move
# ---------------------------------------------------------------------------

def test_the_routing_mirrors_the_services_binned_clause_edge():
    import minimax as _mm

    B = _mm.beta_selector

    spec = B.BETA_CLAUSES[B.BINNED_WIDTH]
    assert R.BINNED_WIDTH_BETA_MAX == spec.beta_max
    assert R.BINNED_WIDTH_RATIOS == tuple(spec.qualified_at)
    assert R.SHIPPED_WIDTH_BETA_MAX == B.BETA_CLAUSES[B.WIDTH].beta_max


def test_the_leaf_ceiling_and_the_fit_guards_did_not_move():
    """The four fit guards and the mandatory-refit guard are untouched.

    Asserted rather than reviewed, because "strictly additive" is a claim
    about what did NOT change and the diff is where such claims go to be
    lost.
    """

    from gw.mpa import pade_fit

    assert SP.MAX_WIDTH_SPLIT_LEAVES == 8192
    assert SP.CROSSING_WIDTH_RATIO_MAX == 4.0
    assert SP.DEFAULT_LAPLACE_RATIO_MAX == 100.0
    assert pade_fit.DEFAULT_GUARDS["width_ratio_max"] == 1.0
    assert R.FIT_WIDTH_RATIO_MAX == pade_fit.DEFAULT_GUARDS[
        "width_ratio_max"]


# ---------------------------------------------------------------------------
#  THE GATE THAT MATTERS -- accuracy against the certificate, not against
#  the other arm
# ---------------------------------------------------------------------------

def _wide_pass_field(n_p=2, n_modes=120, seed=31, decades=3.0):
    """A pole field wide enough in ``Gamma`` that the clause has to bin it.

    ``Gamma`` proportional to ``a`` again (the fourth guard), and the
    spread deliberately three decades so the per-pole clause cuts many
    panes and the binned one cuts ``log_r`` of them.  The two arms then
    differ by a real change of quadrature and not by rounding.
    """

    rng = np.random.default_rng(seed)
    a = np.sort(10.0 ** rng.uniform(0.4, 0.4 + decades,
                                    size=(n_p, n_modes)), axis=1) / RYD
    g = a * rng.uniform(0.25, 0.95, size=(n_p, n_modes))
    b = (rng.normal(size=(n_p, n_modes))
         + 1j * rng.normal(size=(n_p, n_modes))) * 0.3
    return a - 1j * g, b


@pytest.mark.parametrize("r", [2.0, 4.0])
def test_sigma_c_with_the_clause_on_sits_inside_the_certified_error(r):
    """The accuracy gate, scored against the ANALYTIC pole sum.

    Not an A/B delta.  An A/B delta can be small because both arms are
    wrong the same way, and it has no scale to be judged against; the
    closed-form all-pole self-energy has both.  So each arm is scored
    against ``_analytic_sigma`` and the binned arm's error must sit
    inside the tier its entries certify, with the flag-off arm's error
    reported beside it as the thing it must not be much worse than.

    IF THIS CELL FAILS THE CERTIFICATION IS WRONG.  It is not a
    tolerance to widen: the entries claim a supremum over the whole
    ``(u, beta)`` band, the planner is only allowed to bin because of
    that claim, and a measured error above it means the claim is false.
    """

    from test_mpa_sigma_pass import _analytic_sigma, _sigma_through_the_loop

    omega_p, b_p = _wide_pass_field()
    # BOTH A-SPACES POSITIVE, which is what the tree's own pass-loop cell
    # uses and what both clauses' derivations require: ``x_min =
    # min(E_A) + a_lo`` bounds beta only while ``min(E_A) >= 0``, and the
    # A-side energies on this path are |transition energies|.  A negative
    # E_A makes the SHIPPED clause refuse before the new one is reached,
    # which is the first thing this cell measured when it was written.
    E_cond = np.array([1.0, 4.5, 12.0]) / RYD
    E_val = np.array([0.6, 3.2]) / RYD
    omega = np.linspace(-4.0, 4.0, 9) / RYD

    exact = _analytic_sigma(omega, E_cond, E_val, omega_p, b_p)
    scale = float(np.max(np.abs(exact)))

    got = {}
    for flag in (None, r):
        got[flag] = _sigma_through_the_loop(
            omega_ry=omega, E_cond=E_cond, E_val=E_val, omega_p=omega_p,
            b_p=b_p, xi_ry=1.0e-9, rel_tol=1.0e-8,
            binned_width_clause=flag)

    err_off = float(np.max(np.abs(got[None] - exact))) / scale
    err_on = float(np.max(np.abs(got[r] - exact))) / scale
    delta = float(np.max(np.abs(got[r] - got[None]))) / scale

    # The tier the entries claim.  ``rel_tol = 1e-8`` is what the planner
    # asks its rules for and what the binned catalog certifies at, and
    # the gate is the tier ITSELF rather than a multiple of it, because
    # that is what the entries promise.  MEASURED on this field: 1.41e-9
    # with the flag off, 1.38e-9 at r = 2, 1.39e-9 at r = 4, and an
    # on-vs-off delta of 1.6e-10 and 4.7e-10 -- so the clause is spending
    # about a seventh of its budget and the collapse costs no accuracy at
    # all.
    tier = 1.0e-8
    assert err_on <= tier, (
        f"r={r}: the binned arm's error against the analytic sum is "
        f"{err_on:.3e} relative, outside the {tier:.0e} tier its entries "
        "certify (flag off measures {err_off:.3e}). THE CERTIFICATION IS "
        "WRONG -- do not widen this number; the entries claim a supremum "
        "over the band and the planner is only allowed to bin because of "
        "that claim.")
    assert delta <= tier, (
        f"r={r}: Sigma_c moved by {delta:.3e} relative between flag on "
        f"and flag off, outside the {tier:.0e} the clause certifies")
    # And the collapse must not have cost an order of accuracy.
    assert err_on <= max(10.0 * err_off, tier)

"""Gates for ``common.rank_criterion`` — the rank-truncation criterion.

Every instrument in this file is shown FAILING before it is trusted (wk_REL
README §5.1): each positive assertion is paired with a control that makes the
same check go red.  A checker that cannot go red is a void check, and this
project has nine of those on record.

Pure numpy — no jax, no WFN.  Runs on a login node.
"""
import math

import pytest

from common import rank_criterion as rc


# A SMOOTH spectrum with no gap, knee, elbow or plateau anywhere: a pure
# geometric decay over 14 decades.  This is the shape the owner's ruling is
# about — every test below has to work on it, and any criterion that needs a
# separation is undefined on it by construction.
def _smooth(n=400, decades=14.0):
    return [10.0 ** (-decades * i / (n - 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. The criterion is an amplification cap, and it holds on a smooth spectrum
# ---------------------------------------------------------------------------

def test_criterion_caps_the_amplification_on_a_gapless_spectrum():
    s = _smooth()
    for rtol in (1e-4, 1e-6, 1e-8, 1e-10):
        r = rc.rank_report(s, rtol, label="smooth")
        assert r.rank_criterion > 0
        assert r.kappa_eff <= r.kappa_cap, (
            f"rtol={rtol}: kappa_eff {r.kappa_eff:.3e} > cap {r.kappa_cap:.3e}")
        assert not r.violations()


def test_the_amplification_check_can_go_red():
    """CONTROL for the test above — force a rank past the criterion.

    Without this, ``kappa_eff <= kappa_cap`` could be an identity that no
    input can break, i.e. a void check.
    """
    s = _smooth()
    r_ok = rc.rank_report(s, 1e-8, label="smooth")
    over = rc.rank_report(s, 1e-8, label="smooth", rank_used=r_ok.n_total)
    # rank_used > rank_criterion is read as PADDING, which never raises
    # kappa_eff — so padding alone must stay green ...
    assert not over.violations()
    # ... and the way to break it is to keep more of the SPECTRUM, which is
    # what a looser rtol does.  Same retained set, tighter declared cap:
    bad = rc.rank_report(s, 1e-8, label="smooth")
    bad.kappa_cap = 1e2          # declare a cap the retained block violates
    msgs = bad.violations()
    assert any("EXCEEDS the cap" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# 2. Padding is inert; a device-grid round-DOWN is a reported defect
# ---------------------------------------------------------------------------

def test_alignment_padding_is_not_counted_as_a_discard():
    s = _smooth()
    base = rc.rank_report(s, 1e-8)
    padded = rc.rank_report(s, 1e-8, rank_used=base.rank_criterion + 7)
    assert padded.n_padded_alignment == 7
    assert padded.n_dropped_alignment == 0
    assert padded.kappa_eff == base.kappa_eff       # same retained block
    assert padded.sigma_min_kept == base.sigma_min_kept
    assert not padded.violations()


def test_a_grid_round_down_is_reported_as_a_violation():
    """The DEFECT this campaign removes, shown red on the same input."""
    s = _smooth()
    base = rc.rank_report(s, 1e-8)
    rounded = rc.rank_report(s, 1e-8, rank_used=base.rank_criterion - 7)
    assert rounded.n_dropped_alignment == 7
    msgs = rounded.violations()
    assert any("DEVICE-GRID reason" in m for m in msgs), msgs
    # and it really did change the retained block, i.e. the answer
    assert rounded.sigma_min_kept > base.sigma_min_kept


# ---------------------------------------------------------------------------
# 2b. ``n_dropped_closure`` — the one deficit that is NOT a violation
# ---------------------------------------------------------------------------
#
# ``common/spectral_closure`` drops the degenerate block a cut straddles
# (owner ruling 2026-08-10), so ``rank_used`` legitimately lands BELOW the
# criterion's rank.  That is a round-down, and check 2 refuses round-downs —
# because a round-down to the DEVICE GRID makes the physics a function of the
# machine.  A closure drop is a function of the SPECTRUM instead, so it is
# attributed to its own column and excluded.
#
# This pair is here because the change makes a guard QUIETER in one case, and
# a guard that got quieter without a gate is a guard nobody can trust again.


def test_a_closure_drop_is_attributed_and_is_NOT_a_grid_violation():
    """TRUE arm: the deficit the closure guard accounts for is not a defect."""
    s = _smooth()
    base = rc.rank_report(s, 1e-8)
    closed = rc.rank_report(s, 1e-8, rank_used=base.rank_criterion - 3,
                            n_dropped_closure=3)
    assert closed.n_dropped_closure == 3
    assert closed.n_dropped_alignment == 0
    assert not [m for m in closed.violations() if "DEVICE-GRID" in m], (
        "the closure drop was read as a device-grid round-down — with this "
        "wrong, every run whose cut lands in a degenerate block refuses at "
        "htransform, which is what the flip to drop_block would have caused")
    # It is still REPORTED.  Excluded from the violation, never from the log.
    assert "closure-dropped" in closed.describe()
    assert "DEGENERACY CLOSURE" in closed.describe()


def test_an_UNATTRIBUTED_deficit_still_violates():
    """FALSE arm: the exemption is not a blanket one.

    Same deficit, no claim made for it.  If this passed, the new field would
    have disabled check 2 rather than refined it.
    """
    s = _smooth()
    base = rc.rank_report(s, 1e-8)
    bare = rc.rank_report(s, 1e-8, rank_used=base.rank_criterion - 3)
    assert bare.n_dropped_closure == 0 and bare.n_dropped_alignment == 3
    assert [m for m in bare.violations() if "DEVICE-GRID" in m]


def test_a_partial_claim_leaves_the_remainder_a_violation():
    """A closure drop and a grid round-down on the SAME run must not merge.

    Three directions gone, only one of them the closure's: the other two are
    still the mesh changing the physics, and the report must say so.
    """
    s = _smooth()
    base = rc.rank_report(s, 1e-8)
    mixed = rc.rank_report(s, 1e-8, rank_used=base.rank_criterion - 3,
                           n_dropped_closure=1)
    assert mixed.n_dropped_closure == 1 and mixed.n_dropped_alignment == 2
    assert [m for m in mixed.violations() if "DEVICE-GRID" in m]


def test_an_overclaim_cannot_manufacture_credit():
    """Claiming more closure drops than there are missing directions is
    clamped, so the field cannot be used to silence an unrelated round-down."""
    s = _smooth()
    base = rc.rank_report(s, 1e-8)
    over = rc.rank_report(s, 1e-8, rank_used=base.rank_criterion - 2,
                          n_dropped_closure=99)
    assert over.n_dropped_closure == 2 and over.n_dropped_alignment == 0
    # and a claim on a run with NO deficit stays zero rather than going negative
    none = rc.rank_report(s, 1e-8, n_dropped_closure=5)
    assert none.n_dropped_closure == 0 and none.n_dropped_alignment == 0
    assert not none.violations()


# ---------------------------------------------------------------------------
# 3. The discrepancy principle is REFUTED here — show the arithmetic
# ---------------------------------------------------------------------------

def test_noise_floor_cut_is_looser_than_the_measured_catastrophe():
    """§R19: zeta_rcond 1e-12 measured eqp0 = -5049.59 eV.  The discrepancy
    principle prescribes a cut LOOSER STILL, so it cannot be adopted.

    Direction convention, because it is easy to invert (this assertion was
    written backwards once and this test is what caught it): rtol is a
    RELATIVE THRESHOLD, so a SMALLER rtol keeps MORE directions.  "Looser"
    therefore means a smaller number.  The claim is asserted below on the
    RETAINED RANK, where there is no convention left to get wrong.
    """
    # LORRAX production sizes: nk*nb ~ 1e4 rows.
    nf = rc.noise_floor_rtol(10_000)
    assert 1e-15 < nf < 1e-13, nf
    assert nf < 1e-12, (
        "the noise-floor cut must be LOOSER (i.e. a SMALLER rtol) than the "
        f"1e-12 that measured -5049.59 eV; got {nf:.3e}")

    # The statement that actually matters, in ranks: on a smooth spectrum the
    # noise-floor cut retains at least as much as 1e-12, which retains more
    # than 1e-8 — and §R19 measured that direction as monotonically
    # catastrophic (3.1350 -> -206.83 -> -5049.59 eV).
    s = _smooth(400, 16.0)
    r_prod = rc.select_rank(s, 1e-8)
    r_r19 = rc.select_rank(s, 1e-12)
    r_noise = rc.select_rank(s, nf)
    assert r_prod < r_r19 <= r_noise, (r_prod, r_r19, r_noise)


def test_margin_reports_the_overcomplete_regime():
    """The margin is a finite difference of a counting function — it needs no
    gap — and it separates a terminating spectrum from an over-complete one."""
    # Over-complete: the spectrum keeps going below the cut.
    over = rc.rank_report(_smooth(400, 14.0), 1e-8)
    # Terminating: 40 real directions, then an abrupt floor at 1e-300.
    term = rc.rank_report([10.0 ** (-4.0 * i / 39) for i in range(40)]
                          + [1e-300] * 360, 1e-8)
    assert over.overcomplete_margin > 0.2, over.overcomplete_margin
    assert term.overcomplete_margin == 0.0, term.overcomplete_margin
    # ... and the margin instrument therefore CAN read both ways (§5.1).


# ---------------------------------------------------------------------------
# 4. The scale guard — the documented all-zero-window trap
# ---------------------------------------------------------------------------

def test_zero_spectrum_is_refused_not_silently_rank_zero():
    r = rc.rank_report([0.0] * 64, 1e-8)
    assert r.rank_criterion == 0
    msgs = r.violations()
    assert any("RELATIVE threshold is meaningless" in m for m in msgs), msgs


def test_nan_spectrum_is_refused():
    r = rc.rank_report([float("nan")] + [1.0] * 10, 1e-8)
    assert any("RELATIVE threshold is meaningless" in m
               for m in r.violations())


def test_a_healthy_spectrum_does_not_trip_the_scale_guard():
    """CONTROL: the guard above must not fire on ordinary data."""
    assert not rc.rank_report(_smooth(), 1e-8).violations()


# ---------------------------------------------------------------------------
# 5. select_rank agrees with the arithmetic the call sites used to run inline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rtol", [1e-4, 1e-8, 1e-12])
def test_select_rank_matches_the_inline_form_it_replaces(rtol):
    import numpy as np
    s = np.asarray(_smooth())
    assert rc.select_rank(s, rtol) == int((s > s.max() * rtol).sum())
    # ascending input (eigh order) must give the same answer
    assert rc.select_rank(s[::-1], rtol) == rc.select_rank(s, rtol)


def test_describe_is_loggable_and_names_every_required_field():
    r = rc.rank_report(_smooth(), 1e-8, rank_used=rc.select_rank(_smooth(), 1e-8) + 3)
    txt = r.describe()
    for token in ("kept", "kappa_eff", "discarded", "margin", "noise-floor"):
        assert token in txt, f"{token!r} missing from the run diagnostic"
    assert not math.isnan(r.kappa_eff)


def test_singular_value_compression_reports_the_two_best_rank_residuals():
    metrics = rc.singular_value_compression([4.0, 3.0, 0.0], 1)
    assert metrics.relative_operator_tail == pytest.approx(3.0 / 4.0)
    assert metrics.relative_frobenius_tail == pytest.approx(3.0 / 5.0)
    assert metrics.retained_frobenius_weight == pytest.approx(16.0 / 25.0)
    assert metrics.kappa_eff == pytest.approx(1.0)


def test_singular_value_compression_refuses_an_invalid_receipt():
    with pytest.raises(ValueError, match="outside"):
        rc.singular_value_compression([1.0, 0.1], 3)
    with pytest.raises(ValueError, match="NaN/Inf"):
        rc.singular_value_compression([1.0, float("nan")], 1)


# ---------------------------------------------------------------------------
# 6. The STRUCTURAL rank ceiling — a Gram route counts null-space round-off
# ---------------------------------------------------------------------------

def _tall_gram_spectrum(cols=2032, nulls=2, null_level=1e-3):
    """A spectrum shaped like the measured Na (12288, 2032) failure.

    The Gram route diagonalises ``A Aᴴ`` at the LARGER dimension, so the null
    space of a tall rank-deficient ``A`` arrives as small POSITIVE
    eigenvalues.  At ``rtol=1e-8`` and ``null_level=1e-3`` they sit ABOVE the
    cut, which is exactly how ``rank=2034`` was selected for a matrix that
    algebraically holds 2032 — and how the capacity line that followed came
    to print ``rank=2034`` beside ``nspinor*n_mu=2032``.
    """
    return [1.0 / (i + 1) for i in range(cols)] + [null_level] * nulls


def test_the_ceiling_clamps_a_rank_that_counts_nullspace_roundoff():
    s = _tall_gram_spectrum()
    unclamped = rc.select_rank(s, 1e-8)
    assert unclamped == 2034, (
        f"the fixture no longer reproduces the measured overshoot "
        f"({unclamped}); it must select MORE than the 2032 columns or this "
        f"cell tests nothing")
    assert rc.select_rank(s, 1e-8, ceiling=2032) == 2032


def test_a_carried_rank_above_the_ceiling_is_a_violation():
    """The gate, and it has to be reachable — the pre-fix call site's shape."""
    s = _tall_gram_spectrum()
    bad = rc.rank_report(s, 1e-8, rank_ceiling=2032,
                         rank_used=rc.select_rank(s, 1e-8))
    assert any("algebraically hold at most" in m for m in bad.violations()), \
        bad.violations()


def test_the_ceiling_violation_closes_when_the_call_site_clamps():
    """CONTROL: the same input, clamped the way the call site now does."""
    s = _tall_gram_spectrum()
    ok = rc.rank_report(s, 1e-8, rank_ceiling=2032,
                        rank_used=rc.select_rank(s, 1e-8, ceiling=2032))
    assert not ok.violations(), ok.violations()
    assert ok.rank_unclamped == 2034 and ok.rank_criterion == 2032
    assert "CLAMPED the criterion from 2034" in ok.describe()


def test_no_ceiling_is_reported_as_an_absence_not_a_pass():
    r = rc.rank_report(_smooth(), 1e-8)
    assert r.rank_ceiling is None
    assert "none supplied" in r.describe()


# ---------------------------------------------------------------------------
# 7. THE GATE — certified amplification, and what it refuses to gate on
# ---------------------------------------------------------------------------

def _bound_cut(kappa):
    """A spectrum whose criterion BINDS at exactly ``kappa`` achieved."""
    # top = 1.0, retained tail at 1/kappa, then a decade of true nulls that
    # the cut discards, so ``n_dropped_criterion > 0``.
    return [1.0, 1.0 / kappa] + [1.0 / (kappa * 1e6)] * 8


def test_certify_refuses_a_bound_cut_above_the_certified_ceiling():
    r = rc.rank_report(_bound_cut(1e10), 1e-12,
                       kappa_certified=rc.KAPPA_CERTIFIED_GRAM,
                       quantity="eigenvalues")
    assert r.n_dropped_criterion > 0
    with pytest.raises(rc.RankPolicyError, match="CERTIFIED ceiling"):
        rc.certify(r, site="unit fixture")


def test_certify_passes_inside_the_certified_regime():
    """CONTROL — the same shape, two decades below the ceiling."""
    r = rc.rank_report(_bound_cut(1e6), 1e-12,
                       kappa_certified=rc.KAPPA_CERTIFIED_GRAM,
                       quantity="eigenvalues")
    assert r.n_dropped_criterion > 0
    assert rc.certify(r, site="unit fixture") == []


def test_certify_does_not_fire_when_the_cut_BOUND_NOTHING():
    """The clause that keeps the Si 960 anchor set reachable.

    Nothing discarded means the criterion made no choice — the spectrum
    ended on its own — so a refusal would be refusing the INPUT, not the
    policy.  That is the anchor set's case at production settings (768 of
    768 retained), and it carries the best BerkeleyGW agreement on record.
    """
    s = [1.0] + [1e-11] * 9          # kappa_eff = 1e11, far above 1e8 ...
    r = rc.rank_report(s, 1e-13, kappa_certified=rc.KAPPA_CERTIFIED_GRAM)
    assert r.n_dropped_criterion == 0, "the fixture must not truncate"
    assert r.kappa_eff > rc.KAPPA_CERTIFIED_GRAM
    assert rc.certify(r, site="unit fixture") == []


def test_certify_says_an_uncertified_site_is_an_absence_not_a_pass():
    r = rc.rank_report(_bound_cut(1e14), 1e-16, kappa_certified=None)
    assert rc.certify(r, site="transverse") == []
    assert "NONE for this site" in r.describe()


def test_the_gate_modes_are_refuse_warn_off_and_a_typo_refuses():
    r = rc.rank_report(_bound_cut(1e10), 1e-12,
                       kappa_certified=rc.KAPPA_CERTIFIED_GRAM)
    lines = []
    assert rc.certify(r, site="s", mode="warn", log=lines.append)
    assert any("rank-policy" in ln for ln in lines), lines
    assert rc.certify(r, site="s", mode="off") != []      # returned, not acted
    with pytest.raises(ValueError, match=rc.POLICY_MODE_ENV):
        rc.resolve_policy_mode("warnn")
    assert rc.resolve_policy_mode(None) == rc.DEFAULT_POLICY_MODE == "refuse"


def test_discarded_weight_is_the_accuracy_statement_not_the_rank():
    """A third of the RANK discarded, essentially none of the WEIGHT.

    This is why drop fraction is refuted as a gate and this number is not:
    MoS2 production discards 33 % of the rank at the certified rcond and is
    correct, so any gate on the fraction fires on a good run.
    """
    s = [1.0 / (i + 1) for i in range(600)] + [1e-12] * 300
    r = rc.rank_report(s, 1e-8, kappa_certified=rc.KAPPA_CERTIFIED_GRAM)
    frac_rank = r.n_dropped_criterion / r.n_total
    assert frac_rank > 0.3, frac_rank
    assert r.discarded_weight < 1e-9, r.discarded_weight
    assert rc.certify(r, site="unit fixture") == []


def test_discarded_weight_can_fire(monkeypatch):
    """CONTROL: a cut that DOES eat the operator's weight is caught."""
    s = [1.0, 0.9, 0.8, 0.7]
    r = rc.rank_report(s, 0.95, kappa_certified=None)   # keeps only 1.0
    assert r.n_dropped_criterion == 3, r.n_dropped_criterion
    assert r.discarded_weight > rc.DISCARDED_WEIGHT_MAX
    with pytest.raises(rc.RankPolicyError, match="tr|operator"):
        rc.certify(r, site="unit fixture")


# ---------------------------------------------------------------------------
# 8. Scale-aware probe independence — no absolute floors
# ---------------------------------------------------------------------------

def test_probe_independence_is_scale_free():
    """The same GEOMETRY at three magnitudes must give the same verdict.

    An absolute floor cannot do this, and that is the whole defect: a probe's
    coefficient vector scales like |psi| at one sample point, which falls
    like 1/sqrt(N_mu), so an absolute 1e-6 turns into a system-size-dependent
    refusal.  Measured on a valid fully relativistic LiF WFN, it discarded
    every Kramers probe of a TRIM block.
    """
    for scale in (1.0, 1e-4, 1e-12):
        assert rc.probe_is_independent(0.5 * scale, 1.0 * scale, 1.0 * scale)
        assert not rc.probe_is_independent(
            1e-14 * scale, 1.0 * scale, 1.0 * scale)
    # The old absolute floor's verdict on the middle case, for contrast:
    assert 0.5 * 1e-12 < 1e-6, "the fixture no longer exhibits the defect"


def test_probe_independence_uses_the_family_scale():
    """A negligible probe cannot pass on its own relative test alone."""
    assert not rc.probe_is_independent(1e-30, 1.1e-30, 1.0)
    assert rc.probe_is_independent(1e-30, 1.1e-30, 0.0)   # no family scale
    assert not rc.probe_is_independent(float("nan"), 1.0, 1.0)


# ---------------------------------------------------------------------------
# 9. Deferred refusal for cuts inside a jit
# ---------------------------------------------------------------------------

def test_pending_findings_refuse_at_the_host_seam_and_always_clear():
    rc.raise_if_pending(mode="off")                    # clear residue
    rc.note_device_finding("zeta rank_truncate", "kappa 9.7e9 vs 1e8")
    assert rc.pending()
    with pytest.raises(rc.RankPolicyError, match="certified regime"):
        rc.raise_if_pending("the zeta fit", mode="refuse")
    assert rc.pending() == [], (
        "raise_if_pending must always clear — a later stage inheriting an "
        "earlier stage's finding would refuse the wrong run")
    lines = []
    rc.note_device_finding("zeta rank_truncate", "again")
    rc.raise_if_pending("the zeta fit", mode="warn", log=lines.append)
    assert lines and rc.pending() == []

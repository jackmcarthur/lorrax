"""Layer L-a: the k-star contract, on HAND-TYPED tables.  The 183.61 eV class.

This file carries the three CRITICAL gaps the survey ranked first (§7.2):
``trs_reference`` discrimination (G2), ``_star_conj_flags``' XOR (G3), and
the fact that ``star_select`` / ``star_broadcast`` / ``star_spread`` /
``KStarMap`` had no collected test anywhere in the tree (G4).  Everything
here is numpy and index arrays: the star helpers take ``(irr_idx_k,
sym_idx_k, n_sym_spatial)`` plus one operand, so the whole tier runs in
milliseconds with no HDF5, no mesh and no deck.

WHY THE TABLES ARE TYPED OUT AND NOT DERIVED — the survey §5.2b trap, which
:func:`test_a_derived_grid_cannot_carry_these_tests` MEASURES rather than
asserts by assertion.  Building ``(irr_idx_k, sym_idx_k)`` by calling
``find_irreducible_bz_points`` on a 3x3x1 grid with ``[I, -I]`` DOES give
time-reversed rows — four of nine — so the antiunitary branch fires and
everything looks exercised.  But the ``irr_kgrid_int=None`` branch picks
LEX-MIN orbit representatives, and a lex-min representative is always
spatial, so NO star begins on a time-reversal row, the two conjugation
predicates agree on all nine rows, and the ``ibz_slab`` round trip is
0.000e+00 whether the rule is right or wrong.  A generated grid is a
tautology wearing a measurement's clothes.

WHERE THE LITERALS COME FROM.  They are the SHIPPING op-selection policy's
own output on two real decks, re-derived from ``SymMaps(wfn)`` itself at
e9340d1 (WAVE1_BRIEF ruling 5) and committed as
``data/star_tables_e9340d1.json``.  The typed copies and the JSON check
each other in both directions: the JSON catches a typo here, and the
literals catch a silent regeneration there.  ``gnppm_debug`` is the
STRONGEST in-tree discriminator — 5 stars of sizes [1,2,2,2,2], 4 of them
beginning on a TRS row, predicates disagreeing on 8 of 9 rows.  (27cc885's
commit message says the opposite — "on gnppm_debug and bispinor_debug every
star is a singleton, so the XOR is identically zero".  That MECHANISM claim
is refuted by the production class; the commit's CONCLUSION is unaffected.
``test_the_27cc885_singleton_claim_is_refuted_by_the_tables`` is where that
erratum lives in executable form.)

ANTI-TAUTOLOGY, EVERY TABLE-DRIVEN CELL.  :func:`_disagreement` counts the
rows where ``star_row`` and ``ibz_slab`` pick different conjugations, and
every cell below asserts the count BEFORE it uses the table — 8 on gnppm, 6
on cohsex.  A precondition failure, not a skip: if the op-selection policy
ever moves (survey §8.1, the 15.9 eV fork) these tests stop discriminating,
and a skip would let that happen quietly.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from symmetry_maps import (KStarMap, find_irreducible_bz_points,
                           star_broadcast, star_select, star_spread)
from symmetry_maps.maps import _star_conj_flags, _star_row_order

_TESTS = os.path.dirname(os.path.abspath(__file__))
_TABLES = os.path.join(_TESTS, "data", "star_tables_e9340d1.json")

# ---------------------------------------------------------------------------
# The hand-typed tables (survey §5.0b; re-derived from SymMaps at e9340d1).
# ---------------------------------------------------------------------------

#: gnppm_debug / bispinor_debug — identical star structure.  MoS2 3x3,
#: symmorphic, ntran=2, nrk=nk_tot=9, 4 time-reversed rows.
GNPPM_IRR = np.array([0, 2, 2, 6, 8, 7, 6, 7, 8], dtype=np.int32)
GNPPM_SYM = np.array([0, 2, 0, 2, 2, 2, 0, 0, 0], dtype=np.int32)
GNPPM_NSS = 2

#: cohsex_debug — MoS2, ntran=12, nrk=4, nk_tot=9, 3 time-reversed rows.
#: Star label 2's first full-BZ row carries ``sym_idx 12 = ntran``, a PURE
#: time reversal: this is the deck 27cc885 measured its 183.61 eV on.
COHSEX_IRR = np.array([0, 2, 2, 2, 3, 2, 2, 2, 3], dtype=np.int32)
COHSEX_SYM = np.array([0, 12, 0, 3, 0, 14, 15, 2, 1], dtype=np.int32)
COHSEX_NSS = 12

#: (name, irr, sym, n_sym_spatial, expected disagreement count)
DECKS = (
    ("gnppm_debug", GNPPM_IRR, GNPPM_SYM, GNPPM_NSS, 8),
    ("bispinor_debug", GNPPM_IRR, GNPPM_SYM, GNPPM_NSS, 8),
    ("cohsex_debug", COHSEX_IRR, COHSEX_SYM, COHSEX_NSS, 6),
)
_CASES = [pytest.param(*d, id=d[0]) for d in DECKS]


def _disagreement(irr, sidx, nss) -> int:
    """Rows where ``star_row`` and ``ibz_slab`` choose differently.

    ``star_row``'s predicate is ``trs(member) XOR trs(star's kept row)``;
    ``ibz_slab``'s is ``trs(member)`` alone.  They coincide exactly while
    every star's first full-BZ row is spatial — a property of the
    op-selection policy, not of the physics.
    """
    _, xor = _star_conj_flags(irr, sidx, nss)
    own = np.asarray(sidx) >= int(nss)
    return int(np.sum(xor != own))


def _require_discriminating(irr, sidx, nss, expect: int) -> None:
    """THE ANTI-TAUTOLOGY PRECONDITION.  Fails, never skips.

    A cell that merely runs on these tables and passes proves nothing if
    the two predicates agree everywhere: it would pass identically with the
    conjugation rule inverted.  Asserting the disagreement COUNT first is
    what makes every assertion after it evidence.
    """
    got = _disagreement(irr, sidx, nss)
    assert got == expect and got > 0, (
        f"PRECONDITION: star_row and ibz_slab must disagree on {expect} of "
        f"{len(irr)} rows for this table to discriminate the two predicates; "
        f"measured {got}.  Either the table was edited or the op-selection "
        f"policy in find_symmetry_ops_simple moved (survey §8.1, the 15.9 eV "
        f"fork).  This is a FAILURE and not a skip on purpose — a suite that "
        f"skipped here would go green while measuring nothing.")


def _hermitian_star_consistent(irr, sidx, nss, nb=4, seed=0):
    """A star-consistent Hermitian ``A_full``, built by an INDEPENDENT loop.

    One Hermitian matrix per STAR label, written to every member of that
    star, conjugated where the member's TRS-ness differs from the star's
    kept row.  The loop is a plain python ``for`` over rows using
    ``_star_conj_flags`` only for the PREDICATE — it never calls
    ``star_broadcast``, so ``star_broadcast(star_select(A), trs_reference="star_row") == A`` is a
    claim about the helpers and not a restatement of how ``A`` was made.
    """
    rng = np.random.default_rng(seed)
    base = {}
    for label in sorted({int(v) for v in irr}):
        m = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        base[label] = 0.5 * (m + m.conj().T)
    _, xor = _star_conj_flags(irr, sidx, nss)
    out = np.zeros((len(irr), nb, nb), dtype=np.complex128)
    for k in range(len(irr)):
        b = base[int(irr[k])]
        out[k] = np.conj(b) if xor[k] else b
    return out


# ---------------------------------------------------------------------------
# The tables themselves
# ---------------------------------------------------------------------------

def test_the_typed_tables_match_the_committed_derivation():
    """Typo guard one way, regeneration guard the other.

    The literals above are what a reader of this file reasons about; the
    JSON is what ``SymMaps(wfn)`` actually produced at e9340d1.  Checking
    them against each other means a slip in either is loud: mistype a digit
    here and this fails, regenerate the JSON from a moved policy and this
    fails too.  Neither file is derived from the other.
    """
    with open(_TABLES) as fh:
        ref = json.load(fh)
    for name, irr, sidx, nss, _ in DECKS:
        row = ref[name]
        np.testing.assert_array_equal(irr, row["irr_idx_k"], f"{name}: irr")
        np.testing.assert_array_equal(sidx, row["sym_idx_k"], f"{name}: sym")
        assert nss == row["n_sym_spatial"], name


def test_the_27cc885_singleton_claim_is_refuted_by_the_tables():
    """The erratum, in executable form (design brief, owner-visible row).

    27cc885's message states: "On gnppm_debug and bispinor_debug every star
    is a singleton, so the XOR is identically zero — no conjugation at all."
    The production tables say otherwise, and this is where that is written
    down so nobody re-derives the wrong scope from the commit log.  The
    commit's CONCLUSION (pass ``trs_reference`` explicitly) is unaffected.
    """
    rows, labels = _star_row_order(GNPPM_IRR)
    sizes = np.bincount(GNPPM_IRR)[np.asarray(labels)]
    assert sizes.tolist() == [1, 2, 2, 2, 2], "not all-singleton"
    trs = GNPPM_SYM >= GNPPM_NSS
    assert int(trs[rows].sum()) == 4, "four stars begin on a TRS row"
    _, xor = _star_conj_flags(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    assert int(xor.sum()) > 0, "the XOR is NOT identically zero"
    assert _disagreement(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS) == 8


def test_a_derived_grid_cannot_carry_these_tests():
    """THE §5.2b TRAP, measured — so nobody 'simplifies' the literals away.

    A 3x3x1 grid reduced with ``[I, -I]`` looks like it exercises the
    antiunitary branch: four of nine rows come back time-reversed.  It does
    not exercise the PREDICATE, because the derive-IBZ branch picks lex-min
    orbit representatives and a lex-min representative is always spatial —
    so zero stars begin on a TRS row and the two predicates agree on every
    row.  Both facts are asserted here.  If someone ever makes the derived
    grid discriminate, this cell fails and tells them the literals above
    have a cheaper replacement; until then it is the reason they stay.
    """
    grid = np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(1),
                                indexing="ij"), -1).reshape(-1, 3)
    syms = np.stack([np.eye(3, dtype=int), -np.eye(3, dtype=int)])
    irr, sidx, _ = find_irreducible_bz_points(grid, syms)
    trs = sidx >= 1
    assert int(trs.sum()) == 4, (
        f"the derived grid should still HAVE time-reversed rows "
        f"(got {int(trs.sum())}), or it would not even reach the branch")
    refs, _ = _star_conj_flags(irr, sidx, 1)
    stars_with_trs_first = len({int(r) for r in refs if trs[r]})
    assert stars_with_trs_first == 0, (
        f"the derived grid grew a TRS-first star ({stars_with_trs_first}); "
        f"§5.2b's trap no longer holds and this file's hand-typed tables "
        f"may have a derivable replacement — check before deleting them")
    assert _disagreement(irr, sidx, 1) == 0, (
        "the derived grid's two predicates must AGREE everywhere; that is "
        "exactly why it cannot carry I5/I6")


@pytest.mark.parametrize("name,irr,sidx,nss,expect", _CASES)
def test_the_predicates_disagree_on_a_real_deck(name, irr, sidx, nss, expect):
    """I5.  The disagreement is real, and it is this many rows."""
    _require_discriminating(irr, sidx, nss, expect)


def test_the_disagreement_count_can_fail():
    """RED TWIN for the precondition itself.

    Feed it a table whose stars all begin on a spatial row — the case the
    derived grid produces — and the precondition must refuse rather than
    wave it through.  Without this cell, ``_require_discriminating`` could
    be a no-op and every anti-tautology assertion in this file would be
    decorative.
    """
    irr = np.array([0, 0, 1, 1], dtype=np.int32)
    sidx = np.array([0, 2, 0, 2], dtype=np.int32)   # both stars start spatial
    assert _disagreement(irr, sidx, 2) == 0
    with pytest.raises(AssertionError, match="PRECONDITION"):
        _require_discriminating(irr, sidx, 2, 8)


# ---------------------------------------------------------------------------
# I1 / I2 / I3 — the round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,irr,sidx,nss,expect", _CASES)
def test_select_broadcast_round_trip_is_exact(name, irr, sidx, nss, expect):
    """I1 + I3.  ``star_broadcast(star_select(A), trs_reference="star_row") == A``, BIT-exact.

    Measured before 3e002f2's fix: 1.796e-01 relative on gnppm_debug and
    1.087e-14 after (the fix's own numbers).  Here the operand is built to
    the correct rule by an independent loop, so exactness is the whole
    claim — ``np.array_equal``, not a tolerance: a gather and a conjugation
    introduce no arithmetic.

    I3 is the same assertion read differently: ``star_select``'s kept rows
    and ``star_broadcast``'s ``A_irr`` addressing are the same array
    (``_star_row_order``).  If they were not, the round trip would return
    another star's matrix at every affected k and this would fail.
    """
    _require_discriminating(irr, sidx, nss, expect)
    A = _hermitian_star_consistent(irr, sidx, nss)
    sel, labels = star_select(A, irr)
    back = star_broadcast(sel, irr, sidx, nss, irr_labels=labels, trs_reference="star_row")
    assert np.array_equal(np.asarray(back), A), (
        f"{name}: round trip is not exact; "
        f"max|Δ| = {np.abs(np.asarray(back) - A).max():.3e}")
    assert star_spread(A, irr, sidx, nss) == 0.0


def test_the_round_trip_survives_non_monotone_first_occurrence():
    """I2.  gnppm's first-occurrence labels are [0, 2, 6, 8, 7].

    NON-MONOTONE — positions 3 and 4 are out of sorted order — which is the
    case ``np.unique`` ordering gets wrong and ``_star_row_order`` gets
    right.  ``star_broadcast``'s docstring cites exactly this label set
    (:2044, attributing it to "mos2_4x4"; the re-derivation shows it is
    gnppm's own table).  The assertion is the PRECONDITION here: if the
    labels ever come back sorted, this cell is no longer testing I2 and
    says so instead of passing.
    """
    rows, labels = _star_row_order(GNPPM_IRR)
    assert labels.tolist() == [0, 2, 6, 8, 7]
    assert labels.tolist() != sorted(labels.tolist()), (
        "PRECONDITION: gnppm's first-occurrence labels must be non-monotone "
        "for this to be I2's case")
    A = _hermitian_star_consistent(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    sel, got_labels = star_select(A, GNPPM_IRR)
    np.testing.assert_array_equal(got_labels, labels)
    back = star_broadcast(sel, GNPPM_IRR, GNPPM_SYM, GNPPM_NSS,
                          irr_labels=labels, trs_reference="star_row")
    assert np.array_equal(np.asarray(back), A)


def test_sorted_label_order_breaks_the_round_trip():
    """RED TWIN for I2/I3 — the ``np.unique`` ordering, shown failing.

    Address ``A_irr`` by SORTED label while the rows were kept in
    first-occurrence order and positions 3 and 4 swap: labels [0,2,6,8,7]
    against [0,2,6,7,8].  The broadcast then returns a DIFFERENT STAR's
    matrix at every affected k, silently, with the right shape and the
    right hermiticity.
    """
    A = _hermitian_star_consistent(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    sel, labels = star_select(A, GNPPM_IRR)
    order = np.argsort(labels)
    wrong = star_broadcast(sel[order], GNPPM_IRR, GNPPM_SYM, GNPPM_NSS,
                           irr_labels=labels, trs_reference="star_row")
    err = np.abs(np.asarray(wrong) - A).max() / np.abs(A).max()
    assert err > 0.1, (
        f"sorted-order addressing must be visibly wrong; got {err:.3e}")


# ---------------------------------------------------------------------------
# I5 / I6 — the wrong predicate, and where its error lives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,irr,sidx,nss,expect", _CASES)
def test_the_wrong_flavour_error_is_entirely_off_diagonal(
        name, irr, sidx, nss, expect):
    """I6.  THE 183.61 eV SIGNATURE, and why nothing caught it for a month.

    ``A_irr`` here came from ``star_select`` — its rows are FULL-BZ rows
    carrying a ``sym_idx`` of their own — so ``trs_reference="star_row"`` is
    the correct predicate and ``"ibz_slab"`` is the wrong one.  Applying the
    wrong one conjugates whole stars, and a conjugation of a HERMITIAN
    matrix leaves its diagonal EXACTLY intact: the real diagonal is real,
    ``conj`` is the identity on it.

    That is the entire reason the bug survived: 27cc885 measured 183.61 eV
    against an independently computed V_H with "the DIAGONAL left exactly
    intact", so the electron count, hermiticity, the spectrum, the eqp.dat
    V_H column and every diagonal observable were unchanged.  A test that
    only checks diagonals is VACUOUS here — asserted below as ``== 0.0``,
    exactly, not "small".
    """
    _require_discriminating(irr, sidx, nss, expect)
    A = _hermitian_star_consistent(irr, sidx, nss)
    sel, labels = star_select(A, irr)
    wrong = np.asarray(star_broadcast(sel, irr, sidx, nss, irr_labels=labels,
                                      trs_reference="ibz_slab"))
    delta = wrong - A
    diag = np.array([np.abs(np.diag(delta[k])).max() for k in range(len(irr))])
    assert diag.max() == 0.0, (
        f"{name}: the wrong predicate moved a DIAGONAL by {diag.max():.3e}; "
        f"the 183.61 eV class is defined by leaving it exactly alone")
    rel = float(np.abs(delta).max() / np.abs(A).max())
    assert rel > 0.1, (
        f"{name}: the wrong predicate must be visibly wrong OFF the "
        f"diagonal; got {rel:.3e}")


def test_the_two_flavours_agree_when_no_star_begins_on_a_trs_row():
    """I5's other half, and I9's scope statement.

    On a table where every star's first row is spatial — Si's shape, and
    the derived grid's — the two predicates are the SAME function.  A test
    that ran only there would prove nothing about the conjugation, which is
    why the negative control is stated rather than assumed.
    """
    irr = np.array([0, 0, 1, 1], dtype=np.int32)
    sidx = np.array([0, 2, 0, 2], dtype=np.int32)
    A = _hermitian_star_consistent(irr, sidx, 2)
    sel, labels = star_select(A, irr)
    a = np.asarray(star_broadcast(sel, irr, sidx, 2, irr_labels=labels,
                                  trs_reference="star_row"))
    b = np.asarray(star_broadcast(sel, irr, sidx, 2, irr_labels=labels,
                                  trs_reference="ibz_slab"))
    assert np.array_equal(a, b)
    assert _disagreement(irr, sidx, 2) == 0


def test_an_unknown_trs_reference_refuses():
    """I7.  ``"auto"`` RAISES, naming both legal values.

    A default-on-unknown would be the worst of the three behaviours: the
    caller who wrote ``"auto"`` believed something was being decided for
    them, and the two branches disagree on real decks.
    """
    A = _hermitian_star_consistent(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    sel, labels = star_select(A, GNPPM_IRR)
    with pytest.raises(ValueError) as ei:
        star_broadcast(sel, GNPPM_IRR, GNPPM_SYM, GNPPM_NSS,
                       irr_labels=labels, trs_reference="auto")
    msg = str(ei.value)
    assert "star_row" in msg and "ibz_slab" in msg and "auto" in msg


# ---------------------------------------------------------------------------
# star_spread, and the red twin that proves it can see anything at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,irr,sidx,nss,expect", _CASES)
def test_star_spread_sees_an_unconjugated_operand(name, irr, sidx, nss,
                                                  expect):
    """RED TWIN.  Drop the conjugation and ``star_spread`` goes to O(1).

    The survey's own §5.2b probe measured ``star_spread(un-conjugated A) =
    4.802e+00`` on gnppm's table — an ABSOLUTE number on that probe's own
    random operand, so it is quoted as provenance and the assertion is made
    on the RELATIVE spread, which is a property of the rule rather than of
    the draw.  (Measured here on this file's seed-0 operand: 2.401e+00
    absolute, 0.98 relative on gnppm.)

    Both directions matter: a correct operand must give exactly 0.0 (the
    cell above) and a wrong one must give O(1), or ``star_spread`` is not
    the gauge-mismatch detector ``sc_iteration.py:762`` relies on it being.
    """
    _require_discriminating(irr, sidx, nss, expect)
    A = _hermitian_star_consistent(irr, sidx, nss)
    sel, labels = star_select(A, irr)
    pos = {int(v): i for i, v in enumerate(labels)}
    flat = np.stack([np.asarray(sel)[pos[int(v)]] for v in irr])
    rel = star_spread(flat, irr, sidx, nss) / np.abs(A).max()
    assert rel > 0.5, (
        f"{name}: an un-conjugated operand must show a LARGE star spread; "
        f"got {rel:.3e} relative")


def test_star_spread_sees_a_permuted_member():
    """S11 check 2, ported: the negative control from the uncollected gate.

    ``tests/multi_device/star_invariance_gate.py`` matches no pytest
    pattern, needs a ``STAR_DECK`` env var and returns 1 when it is unset,
    so its four checks have never run in CI.  This is its check 2 — permute
    the bands at ONE star member and the spread must move — with the deck
    replaced by a hand-typed table, which is what makes it collectable.
    """
    A = _hermitian_star_consistent(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    assert star_spread(A, GNPPM_IRR, GNPPM_SYM, GNPPM_NSS) == 0.0
    bad = A.copy()
    victim = int(np.where(GNPPM_IRR == GNPPM_IRR[1])[0][-1])
    bad[victim] = bad[victim][::-1, :]
    rel = star_spread(bad, GNPPM_IRR, GNPPM_SYM,
                      GNPPM_NSS) / np.abs(A).max()
    assert rel > 1e-3, f"permuting bands at one member must show: {rel:.3e}"


def test_singleton_stars_contribute_nothing_to_the_spread():
    """``_spread_tables`` compares members, and a singleton has none.

    Worth pinning because it is the shape that makes a spread test pass
    vacuously: a deck whose stars are all singletons reports 0.000 for any
    operand whatever.  The gate file's own docstring says so; here it is
    measured, so a suite that only ever ran on such a deck can be told
    apart from one that ran on a discriminating one.
    """
    irr = np.arange(6, dtype=np.int32)
    sidx = np.zeros(6, dtype=np.int32)
    rng = np.random.default_rng(1)
    A = (rng.standard_normal((6, 3, 3))
         + 1j * rng.standard_normal((6, 3, 3)))
    assert star_spread(A, irr, sidx, 1) == 0.0
    km = KStarMap(irr, sidx, 1)
    assert km.is_identity and km.spread_rel(A) == 0.0


# ---------------------------------------------------------------------------
# KStarMap — the REAL class (G4), against the free functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,irr,sidx,nss,expect", _CASES)
def test_the_class_and_the_free_functions_are_bit_equal(name, irr, sidx, nss,
                                                        expect):
    """G4.  Every ``KStarMap`` method against its free-function twin.

    The class caches the row tables in ``__init__`` and the free functions
    rebuild them per call; a divergence between the two would show up as
    the SC loop and the diagnostics disagreeing about which rows are
    conjugated, which is invisible until it costs 183 eV.  Bit-equal, not
    close: both sides are the same gather.
    """
    _require_discriminating(irr, sidx, nss, expect)
    A = _hermitian_star_consistent(irr, sidx, nss)
    km = KStarMap(irr, sidx, nss)
    sel, labels = star_select(A, irr)
    np.testing.assert_array_equal(km.select(A), sel)
    np.testing.assert_array_equal(km.labels, labels)
    np.testing.assert_array_equal(
        km.broadcast(sel), star_broadcast(sel, irr, sidx, nss,
                                          irr_labels=labels, trs_reference="star_row"))
    assert km.spread(A) == star_spread(A, irr, sidx, nss)
    assert km.spread_rel(A) == km.spread(A) / np.abs(A).max()
    assert km.nk_full == len(irr)
    assert km.nk_irr == len(labels)
    assert km.reduction == len(irr) / len(labels)
    assert km.is_identity is (len(labels) == len(irr))


def test_the_identity_map_is_a_no_op_in_both_directions():
    """``KStarMap.identity`` — the symmetry-off path, byte-identical.

    A driver written against this class reads the same whether or not
    symmetry is in use; that only holds if the identity map's select and
    broadcast are genuinely no-ops rather than approximately so.
    """
    rng = np.random.default_rng(7)
    A = (rng.standard_normal((5, 3, 3)) + 1j * rng.standard_normal((5, 3, 3)))
    km = KStarMap.identity(5)
    assert km.is_identity and km.reduction == 1.0 and km.nk_irr == 5
    np.testing.assert_array_equal(km.select(A), A)
    np.testing.assert_array_equal(km.broadcast(A), A)
    assert km.spread(A) == 0.0 and km.spread_rel(A) == 0.0


def test_the_class_refuses_mismatched_index_shapes():
    """RED TWIN.  ``irr_idx`` and ``sym_idx`` are the pair that must agree.

    Bundling the three arrays is the class's whole reason for existing —
    "no call site can supply two of the three".  A constructor that
    accepted a length mismatch would give that up silently.
    """
    with pytest.raises(ValueError, match="n_k_full"):
        KStarMap(np.zeros(4, dtype=np.int32), np.zeros(3, dtype=np.int32), 1)


def test_from_sym_reads_the_two_tables_it_names():
    """``KStarMap.from_sym`` on a SymMaps-shaped object.

    A duck-typed stand-in rather than a deck: this cell is about which two
    attributes the classmethod reads, and the deck-table suite is where the
    real ``SymMaps`` gets bit-compared.  (``test_symmetry_maps_deck_tables``
    runs ``from_sym`` against a production ``SymMaps`` too.)
    """
    class _Sym:
        irr_idx_k = GNPPM_IRR
        sym_idx_k = GNPPM_SYM

    km = KStarMap.from_sym(_Sym(), GNPPM_NSS)
    np.testing.assert_array_equal(km.irr_idx, GNPPM_IRR)
    np.testing.assert_array_equal(km.sym_idx, GNPPM_SYM)
    assert km.n_sym_spatial == GNPPM_NSS


def test_the_scissor_weight_orderings_hold_on_the_real_class():
    """G4.  ``tests/test_scissor_weights.py``'s two ordering assertions,
    re-run against ``KStarMap`` instead of its ``_FakeKStar`` stand-in.

    That file duck-types the class deliberately (it needs no jax) and says
    so; the consequence is that its claim — ``k_star_weights`` comes back in
    ``select``'s row order, including when first occurrence is non-monotone
    — has never been checked against the code the SC loop actually calls.
    The multiplicities are computed here from ``select`` on a column of
    ones, which is exactly what ``gw.scissor.k_star_weights`` does.
    """
    irr = np.array([0, 1, 2, 1, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5])
    km = KStarMap(irr, np.zeros(irr.size, dtype=np.int32), 1)
    counts = np.bincount(irr)[np.asarray(km.labels)]
    assert counts.tolist() == [1, 2, 1, 1, 2, 2, 2, 2, 2, 1]
    assert int(counts.sum()) == irr.size

    nonmono = np.array([5, 5, 0, 2, 0, 2, 2])
    km2 = KStarMap(nonmono, np.zeros(nonmono.size, dtype=np.int32), 1)
    assert km2.labels.tolist() == [5, 0, 2], (
        "PRECONDITION: this table's first-occurrence order must be "
        "non-monotone, or it is not the case test_scissor_weights pins")
    assert np.bincount(nonmono)[np.asarray(km2.labels)].tolist() == [2, 2, 3]


# ---------------------------------------------------------------------------
# S3 — the contract sc_iteration._check_kstar_spread depends on
# ---------------------------------------------------------------------------

def test_spread_rel_returns_a_float_and_propagates_nan():
    """S3.  ``sc_iteration.py:780`` is written ``not (spread <= tol)``.

    Deliberately, so that NaN REFUSES rather than passing: ``nan > tol`` is
    False and would let a poisoned operand through.  That only works if
    ``spread_rel`` returns a real float and does not sanitize NaN, on host
    AND device operands — the driver hands it a sharded ``jax.Array``.  The
    device half at a real mesh extent lives in the emulated-mesh tier; here
    it is the type and the NaN.
    """
    import jax
    import jax.numpy as jnp

    km = KStarMap(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    A = _hermitian_star_consistent(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    host = km.spread_rel(A)
    assert isinstance(host, float) and not isinstance(host, np.ndarray)
    dev = km.spread_rel(jax.device_put(jnp.asarray(A)))
    assert isinstance(dev, float) and dev == host

    poisoned = A.copy()
    poisoned[1, 0, 0] = np.nan
    bad = km.spread_rel(poisoned)
    assert np.isnan(bad)
    assert not (bad <= 1e-6), (
        "a NaN spread must REFUSE; `not (x <= tol)` is the spelling that "
        "makes it, and `x > tol` is the one that does not")


def test_the_nan_check_can_fail():
    """RED TWIN for the spelling above.

    ``nan > tol`` is False — the form that lets a poisoned operand through.
    Asserting that here is what makes the cell above evidence about the
    spelling rather than about NaN.
    """
    nan = float("nan")
    assert not (nan > 1e-6)
    assert not (nan <= 1e-6)


def test_the_helpers_keep_a_device_operand_on_the_device():
    """S11 check 4 at 1x1: same numbers, still a ``jax.Array``.

    Bit-identical, not close — a gather is a gather.  The SHARDED half (a
    real 2x2, output spec preserved) is the emulated-mesh tier's; this cell
    is the part that runs on any machine with jax at all, including the
    one-device dev box.
    """
    import jax
    import jax.numpy as jnp

    A = _hermitian_star_consistent(GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)
    dev = jnp.asarray(A)
    sel_h, labels = star_select(A, GNPPM_IRR)
    sel_d, labels_d = star_select(dev, GNPPM_IRR)
    assert isinstance(sel_d, jax.Array)
    np.testing.assert_array_equal(labels_d, labels)
    assert float(np.abs(np.asarray(sel_d) - sel_h).max()) == 0.0
    back_h = star_broadcast(sel_h, GNPPM_IRR, GNPPM_SYM, GNPPM_NSS,
                            irr_labels=labels, trs_reference="star_row")
    back_d = star_broadcast(sel_d, GNPPM_IRR, GNPPM_SYM, GNPPM_NSS,
                            irr_labels=labels, trs_reference="star_row")
    assert isinstance(back_d, jax.Array)
    assert float(np.abs(np.asarray(back_d) - back_h).max()) == 0.0
    assert star_spread(dev, GNPPM_IRR, GNPPM_SYM, GNPPM_NSS) == star_spread(
        A, GNPPM_IRR, GNPPM_SYM, GNPPM_NSS)

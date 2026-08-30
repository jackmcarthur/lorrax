"""Layer L-a+: the four in-tree decks, through an 11-attribute header stub.

THE STANDING ACCEPTANCE GATE.  ``SymMaps(stub)``'s ``(irr_idx_k,
sym_idx_k)`` is bit-compared against ``data/star_tables_e9340d1.json`` on
all four decks.  That JSON was produced by the PRODUCTION class at e9340d1
(WAVE1_BRIEF ruling 5, the re-derivation mandate), so this is the extraction's
acceptance criterion turned into a test that keeps running: the op-selection
policy is register-don't-touch (survey §8.1), and a bit-equality gate on
every deck is the tripwire that says if it ever moves.  A failure here is
never a tolerance question.

WHY A STUB RATHER THAN ``WfnLoader``.  See ``_deck_stub.py``: ``SymMaps``
reads exactly eleven attributes, ten of them straight out of ``mf_header``.
Reading them with h5py makes this tier a SERVICE test — it needs no lorrax
on ``sys.path`` and touches kilobytes of a 14-52 MiB file.  The PARITY ARM
below then re-derives the same tables through the production
``SymMaps(WfnLoader(...))`` whenever lorrax IS importable, so the stub can
never quietly drift into being a different question.

THE I5 PRECONDITION TRIPWIRE.  ``cohsex_debug``'s TRS rows are GRATUITOUS:
its IBZ is spatially reduced (the real-fixture gate in
``tests/test_density_symmetry_check.py`` checks the 2c reference), and the
time-reversal rows come from the shipping
op-selection policy's missing ``break``, not from the mesh.  3e002f2 says so
in as many words: the policy "produces gratuitous TRS rows on decks where
every k is rotation-reachable".  So the deck's whole discriminating power
for I5/I6 is a property of a register-don't-touch item.  The precondition
below therefore FAILS rather than skips: if the policy is ever changed, the
15.9 eV fork moved and this is where the tree finds out.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from _deck_stub import (DECK_WFN, DECKS, SYMMAPS_ATTRS, deck_available,
                        deck_path, read_deck)
from symmetry_maps import (
    KStarMap,
    SymMaps,
    build_spatial_operator_tables,
    star_broadcast,
    star_select,
)
from symmetry_maps.maps import _star_conj_flags, _star_row_order

_TESTS = os.path.dirname(os.path.abspath(__file__))
_TABLES = os.path.join(_TESTS, "data", "star_tables_e9340d1.json")


def _reference():
    with open(_TABLES) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def tables():
    return _reference()


def _deck(name):
    """A read stub, or an honest skip naming what is absent.

    NOT ``pytest.importorskip`` on the deck: a missing 44 MiB WFN in a
    checkout without the fixture blobs is a different fact from a missing
    h5py, and the skip-honesty allowlist distinguishes them.
    """
    pytest.importorskip("h5py", reason="h5py is not importable")
    if not deck_available(name):
        pytest.skip(f"no {name}/{DECK_WFN[name]} in this checkout "
                    f"(fixture blobs absent)")
    return read_deck(name)


# ---------------------------------------------------------------------------
# The stub itself
# ---------------------------------------------------------------------------

def test_the_stub_supplies_exactly_what_symmaps_reads():
    """The eleven attributes, checked against the MODULE, not the docstring.

    ``maps.py`` is the source of truth for what ``SymMaps`` consumes; this
    parses it for ``wfn.<name>`` and asserts the stub covers every one.  A
    new attribute added to ``SymMaps.__init__`` therefore fails HERE, with
    the name, rather than as an ``AttributeError`` inside a table build.
    """
    import re

    import symmetry_maps.maps as maps
    with open(maps.__file__) as fh:
        used = set(re.findall(r"\bwfn\.([A-Za-z_][A-Za-z_0-9]*)",
                              fh.read()))
    # ``bvec`` / ``fft_grid`` / ``atom_positions`` appear on the loader's own
    # stub but not in this module; the set below is what THIS module reads.
    assert used == set(SYMMAPS_ATTRS), (
        f"maps.py reads {sorted(used)}; the stub declares "
        f"{sorted(SYMMAPS_ATTRS)}.  Difference: "
        f"{sorted(used ^ set(SYMMAPS_ATTRS))}")


@pytest.mark.parametrize("deck", DECKS)
def test_the_stub_carries_every_declared_attribute(deck):
    """And the read actually populated them, with the right shapes."""
    d = _deck(deck)
    for name in SYMMAPS_ATTRS:
        assert hasattr(d, name), f"{deck}: stub has no {name}"
    assert d.kpoints.shape == (d.nkpts, 3)
    assert np.asarray(d.kgrid).shape == (3,)
    assert d.sym_matrices.shape[0] >= d.ntran
    assert d.translations.shape[0] >= d.ntran
    assert d.atom_crys.shape == (len(d.atom_types), 3)
    assert d.trs_holds is True, (
        "trs_holds is PINNED True, not defaulted; see _deck_stub's "
        "module docstring")


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", DECKS)
def test_the_deck_tables_are_bit_identical_to_the_committed_derivation(
        deck, tables):
    """THE ACCEPTANCE GATE.  Four decks, two integer tables, bit-for-bit.

    ``np.array_equal`` on int arrays: there is no tolerance to widen here,
    and the failure message quotes both tables so the diff is readable
    without a debugger.  The derived scalars are checked alongside because
    a table can be right while the star STRUCTURE is not (a permuted
    ``sym_idx`` with the same multiset would pass an eyeball).
    """
    d = _deck(deck)
    ref = tables[deck]
    sym = SymMaps(d)
    irr = np.asarray(sym.irr_idx_k)
    sidx = np.asarray(sym.sym_idx_k)
    assert np.array_equal(irr, np.asarray(ref["irr_idx_k"])), (
        f"{deck}: irr_idx_k moved\n  got {irr.tolist()}\n  ref "
        f"{ref['irr_idx_k']}")
    assert np.array_equal(sidx, np.asarray(ref["sym_idx_k"])), (
        f"{deck}: sym_idx_k moved\n  got {sidx.tolist()}\n  ref "
        f"{ref['sym_idx_k']}")
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    assert nss == ref["n_sym_spatial"]
    assert int(sym.nk_tot) == ref["nk_tot"]
    assert int(sym.nk_red) == ref["nrk"]
    assert bool(sym.trs_allowed) is ref["trs_allowed"]
    rows, labels = _star_row_order(irr)
    assert labels.tolist() == ref["star_first_occurrence_labels"]
    assert np.bincount(irr)[np.asarray(labels)].tolist() == ref["star_sizes"]
    assert int((sidx >= nss).sum()) == ref["n_trs_rows"]
    assert int((sidx[rows] >= nss).sum()) == ref["n_stars_trs_first_row"]
    _, xor = _star_conj_flags(irr, sidx, nss)
    assert int(np.sum(xor != (sidx >= nss))) == \
        ref["disagree_star_row_vs_ibz_slab"]
    assert (len(labels) == len(irr)) is ref["all_singleton"]


def test_the_bit_equality_gate_can_fail():
    """RED TWIN.  Collapse ONE symmetry op and ``sym_idx_k`` must move.

    The corruption is on a COPY of the header (``DeckHeader.replacing``);
    the fixture on disk is read-only and is not touched.  Without this cell
    the gate above could be comparing an array to itself through some
    accident of the fixture wiring and nobody would know.

    cohsex_debug, and NOT the other three, for a reason worth writing down
    — see :func:`test_gnppm_selects_only_the_identity_and_its_time_reversal`.
    """
    d = _deck("cohsex_debug")
    ref = _reference()["cohsex_debug"]
    sym = SymMaps(d)
    assert np.array_equal(np.asarray(sym.sym_idx_k),
                          np.asarray(ref["sym_idx_k"]))
    assert 1 in {int(v) for v in np.asarray(sym.sym_idx_k)}, (
        "PRECONDITION: op 1 must actually be SELECTED on this deck, or "
        "collapsing it proves nothing")
    mt = np.array(d.sym_matrices)
    mt[1] = np.eye(3, dtype=mt.dtype)          # collapse op 1 to the identity
    broken = SymMaps(d.replacing(sym_matrices=mt))
    assert not np.array_equal(np.asarray(broken.sym_idx_k),
                              np.asarray(ref["sym_idx_k"])), (
        "collapsing a SELECTED symmetry op left sym_idx_k identical; the "
        "gate above is not reading what it thinks it is")


@pytest.mark.parametrize("deck", ["gnppm_debug", "bispinor_debug"])
def test_gnppm_selects_only_the_identity_and_its_time_reversal(deck):
    """MEASURED, and the reason the red twin above is not parametrized.

    On gnppm_debug and bispinor_debug the shipping policy selects exactly
    two rows of ``sym_mats_k``: 0 (the identity) and ``ntran`` (a PURE time
    reversal).  The deck's spatial op 1 (σ_h) is never chosen — every k it
    could reach is reached by time reversal first, because the outer
    ``ikbar`` loop has no ``break`` and the highest matching irreducible k
    wins.  So collapsing op 1 to the identity on these decks changes
    NOTHING, and a red twin written that way would pass by accident on
    cohsex and fail as a false alarm here.

    That is also why these two decks are the strongest ``star_row`` vs
    ``ibz_slab`` discriminators in the tree: four of five stars begin on a
    time-reversal row precisely because time reversal is doing all the
    folding.
    """
    d = _deck(deck)
    sym = SymMaps(d)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    used = {int(v) for v in np.asarray(sym.sym_idx_k)}
    assert used == {0, nss}, (
        f"{deck} now selects ops {sorted(used)}; measured at e9340d1 it "
        f"selects only {{0, ntran={nss}}}, and the op-selection policy is "
        f"register-don't-touch (survey §8.1)")
    mt = np.array(d.sym_matrices)
    mt[1] = np.eye(3, dtype=mt.dtype)
    same = SymMaps(d.replacing(sym_matrices=mt))
    np.testing.assert_array_equal(np.asarray(same.sym_idx_k),
                                  np.asarray(sym.sym_idx_k))


# ---------------------------------------------------------------------------
# The retired parent map (design decision 4)
# ---------------------------------------------------------------------------

class _MinimalDeck:
    """A deck whose stored symmetries cannot reach its own full grid.

    ``nrk = 1`` at Γ with a 2x2x1 grid and two ops: three of the four
    full-BZ points have no preimage.  Synthetic, because no in-tree deck is
    broken this way — and the diagnostic that fires here is the one the
    retired parent-map loop used to own.
    """

    def __init__(self):
        self.kpoints = np.zeros((1, 3))
        self.kgrid = np.asarray([2, 2, 1], dtype=np.int32)
        self.shift = np.zeros(3)
        self.nkpts = 1
        self.ntran = 2
        self.sym_matrices = np.stack([np.eye(3, dtype=np.int32),
                                      np.diag([1, -1, 1]).astype(np.int32)])
        self.translations = np.zeros((2, 3))
        self.avec = np.eye(3)
        self.atom_types = np.array([1])
        self.atom_crys = np.zeros((1, 3))
        self.trs_holds = True


@pytest.mark.parametrize("deck", DECKS)
def test_symmaps_no_longer_publishes_the_lowest_sym_parent_map(deck):
    """Decision 4.  ``kpoint_map`` is GONE, and the used half stayed.

    It was a python triple loop computing, by its own tie-break rule, the
    same k_full -> k_irr relationship ``find_symmetry_ops_simple`` computes
    by the SHIPPING rule (survey §8.1, register-don't-touch).  Two
    independent derivations of one fact, one of them published and read by
    nothing live — 3e002f2 flagged that they agreed on all four fixtures,
    which is the shape of a second source of truth waiting to drift.

    What ``create_kpoint_symmetry_map`` keeps is the uniform grid, and that
    is asserted to be exactly ``unfolded_kpts`` so "kept its used half" is a
    measurement rather than a claim.
    """
    d = _deck(deck)
    sym = SymMaps(d)
    assert not hasattr(sym, "kpoint_map"), (
        "SymMaps still publishes kpoint_map; decision 4 dropped it and the "
        "only readers in the tree are under misc/archived_tests/")
    grid = np.asarray(sym.create_kpoint_symmetry_map(d))
    assert grid.ndim == 2 and grid.shape[1] == 3
    np.testing.assert_array_equal(grid, np.asarray(sym.unfolded_kpts))


def test_incomplete_symmetry_metadata_refuses_instead_of_fabricating_gamma():
    """An uncovered full-k row must never inherit stored Γ/identity.

    The retired loop carried the class's only "WFN symmetry data
    incomplete" diagnostic.  Its successor in
    ``find_symmetry_ops_simple``, whose ``matched`` array already computes
    the same predicate (``sym_mats_k`` is a group, so "some S maps k_full
    into the IBZ" and "some S maps some IBZ k onto k_full" have the same
    truth value), used to warn and leave the index arrays at their zero
    initialization.  That silently substituted Γ/identity wavefunctions.

    Constructed synthetically: Γ-only IBZ against a 2x2x1 grid, so three of
    four full-BZ points are unreachable.  The refusal names the count and
    first offending k.
    """
    with pytest.raises(ValueError, match=r"incomplete: 3 of 4"):
        SymMaps(_MinimalDeck())


def test_operator_tables_do_not_require_a_complete_k_map():
    ops = build_spatial_operator_tables(_MinimalDeck())
    assert ops.sym_matrices.shape == (2, 3, 3)
    assert ops.sym_mats_k.shape == (4, 3, 3)
    assert ops.U_spinor.shape == (2, 2, 2)


class _IdentityOnlyTrReducedDeck:
    def __init__(self, *, trs_holds=True):
        self.kpoints = np.asarray([
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ])
        self.kgrid = np.asarray([4, 1, 1], dtype=np.int32)
        self.shift = np.zeros(3)
        self.nkpts = 3
        self.ntran = 1
        self.sym_matrices = np.eye(3, dtype=np.int32)[None, :, :]
        self.translations = np.zeros((1, 3))
        self.avec = np.eye(3)
        self.atom_types = np.array([1])
        self.atom_crys = np.zeros((1, 3))
        self.trs_holds = bool(trs_holds)


def test_identity_only_tr_reduced_mesh_uses_the_synthesized_minus_identity():
    sym = SymMaps(_IdentityOnlyTrReducedDeck())
    assert sym.nk_tot == 4
    assert int(sym.irr_idx_k[3]) == 1
    assert int(sym.sym_idx_k[3]) == 1  # synthesized time-reversal row


def test_identity_only_tr_reduced_mesh_refuses_when_reference_breaks_trs():
    with pytest.warns(RuntimeWarning, match="TIME-REVERSAL SYMMETRY IS BROKEN"):
        with pytest.raises(ValueError, match=r"1 of 4 full-BZ k-points"):
            SymMaps(_IdentityOnlyTrReducedDeck(trs_holds=False))


def test_identity_full_grid_rejects_duplicate_coordinates():
    deck = _IdentityOnlyTrReducedDeck()
    deck.kpoints = np.concatenate([deck.kpoints, deck.kpoints[:1]], axis=0)
    deck.nkpts = 4
    with pytest.raises(ValueError, match="coordinates do not cover"):
        SymMaps(deck)


def test_the_in_tree_decks_do_not_trip_that_warning():
    """The other half — otherwise the twin above proves the warning fires
    on everything, which would be a different defect wearing a pass."""
    import warnings as _w
    for deck in DECKS:
        d = _deck(deck)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            SymMaps(d)
        bad = [str(x.message) for x in caught
               if "symmetry data incomplete" in str(x.message)]
        assert not bad, f"{deck}: {bad}"


# ---------------------------------------------------------------------------
# The I5 tripwire — cohsex_debug's TRS rows are a POLICY artefact
# ---------------------------------------------------------------------------

def test_cohsex_still_has_a_star_that_begins_on_a_time_reversal_row():
    """THE §8.1 TRIPWIRE.  Fails, never skips.

    Two conditions, and I5/I6 need BOTH:

    * at least one ``sym_idx >= ntran`` — the deck reaches the antiunitary
      branch at all;
    * at least one STAR whose first full-BZ row is one of those — which is
      what makes ``star_row`` and ``ibz_slab`` different functions.  The
      second does not follow from the first: a deck can have plenty of
      time-reversed rows and still begin every star on a spatial one (the
      derived 3x3x1 grid does exactly that, and so does mos2_5x5).

    MEASURED at e9340d1: star label 2's first row carries ``sym_idx 12 =
    ntran``, a PURE time reversal, and the predicates disagree on 6 of 9
    rows.  If this cell fails, the op-selection policy at
    ``find_symmetry_ops_simple:1343-1351`` moved — that is the 15.9 eV eqp
    V_H fork (survey §8.1, register-don't-touch), and the correct response
    is to find out who moved it, not to relax this.
    """
    d = _deck("cohsex_debug")
    sym = SymMaps(d)
    sidx = np.asarray(sym.sym_idx_k)
    irr = np.asarray(sym.irr_idx_k)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    assert nss == int(d.ntran) == 12
    trs = sidx >= nss
    assert int(trs.sum()) >= 1, (
        f"cohsex_debug no longer produces ANY sym_idx >= ntran={nss}; the "
        f"op-selection policy moved (survey §8.1, the 15.9 eV fork) and "
        f"this deck no longer reaches the antiunitary branch")
    rows, labels = _star_row_order(irr)
    first_trs = [int(labels[j]) for j, r in enumerate(rows) if trs[r]]
    assert first_trs, (
        f"cohsex_debug has {int(trs.sum())} time-reversed rows but NO star "
        f"begins on one, so star_row and ibz_slab are now the same "
        f"function and I5/I6 have no in-tree deck.  The op-selection "
        f"policy moved (survey §8.1); see the design brief's "
        f"register-don't-touch item 13 before changing anything here")
    assert 2 in first_trs, (
        f"star label 2 is the one 27cc885 measured its 183.61 eV on; the "
        f"TRS-first stars are now {first_trs}")
    assert int(sidx[rows[labels.tolist().index(2)]]) == nss, (
        "star label 2's first row must carry sym_idx == ntran exactly (a "
        "PURE time reversal), which is what 27cc885 reports")


@pytest.mark.parametrize("deck", DECKS)
def test_the_two_predicates_are_the_same_function_only_on_si(deck, tables):
    """I9, the scope statement, as a table over all four decks.

    27cc885 is active on cohsex_debug, gnppm_debug and bispinor_debug and a
    NO-OP on Si.  A test that ran only on Si would prove nothing about the
    conjugation and would look identical to one that proved everything —
    which is skip-honesty's twin, and why the count is asserted per deck
    against the committed table rather than "> 0 somewhere".
    """
    d = _deck(deck)
    sym = SymMaps(d)
    irr = np.asarray(sym.irr_idx_k)
    sidx = np.asarray(sym.sym_idx_k)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    _, xor = _star_conj_flags(irr, sidx, nss)
    n = int(np.sum(xor != (sidx >= nss)))
    assert n == tables[deck]["disagree_star_row_vs_ibz_slab"]
    if deck == "si_cohsex_debug":
        assert n == 0, "Si is the negative control and must stay one"
    else:
        assert n > 0


# ---------------------------------------------------------------------------
# The parity arm — the production loader, when lorrax is importable
# ---------------------------------------------------------------------------

def _wfn_loader_or_skip():
    try:
        from file_io import WfnLoader
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"no lorrax file_io on this path ({type(exc).__name__}); "
                    f"the stub arm above is the standalone claim")
    return WfnLoader


@pytest.mark.parametrize("deck", DECKS)
def test_the_stub_and_the_production_loader_build_the_same_tables(deck,
                                                                  tables):
    """THE PARITY ARM.  ``SymMaps(stub)`` == ``SymMaps(WfnLoader(...))``.

    Without it the stub is an unfalsifiable convenience: it could drift
    into supplying subtly different arrays (a τ that is not divided by 2π,
    an ``atom_crys`` built from the wrong transpose) and every cell above
    would still be self-consistently green.  With it, the two paths have to
    keep agreeing, and the JSON keeps agreeing with both.

    Skip-honesty makes this arm MUST-RUN whenever lorrax is present: it is
    the cell that connects the service's own tier to the tree it serves.
    """
    Loader = _wfn_loader_or_skip()
    d = _deck(deck)
    loader = Loader(deck_path(deck), backend="eager")
    try:
        prod = SymMaps(loader)
    finally:
        close = getattr(loader, "close", None)
        if callable(close):
            close()
    stub = SymMaps(d)
    np.testing.assert_array_equal(np.asarray(prod.irr_idx_k),
                                  np.asarray(stub.irr_idx_k))
    np.testing.assert_array_equal(np.asarray(prod.sym_idx_k),
                                  np.asarray(stub.sym_idx_k))
    np.testing.assert_array_equal(np.asarray(prod.irr_idx_k),
                                  np.asarray(tables[deck]["irr_idx_k"]))
    np.testing.assert_allclose(np.asarray(prod.R_cart),
                               np.asarray(stub.R_cart), atol=1e-13)
    np.testing.assert_allclose(np.asarray(prod.translations),
                               np.asarray(stub.translations), atol=1e-15)


# ---------------------------------------------------------------------------
# validate_atomic_symmetries — S4, the convention oracle
# ---------------------------------------------------------------------------

def test_the_atomic_symmetry_check_passes_on_every_in_tree_deck():
    """S4.  96/96 Si atom mappings, and the same for the other three.

    Four docstrings in this tree cite this routine as the convention
    oracle: "96/96 Si atom mappings pass with this convention; the wrong
    direction ``mtrx·r + τ`` gives 48/96 and breaks Si Fd-3m closure".  The
    48-of-96 half is the next cell.  Si is the interesting deck here — it
    is the only genuinely non-symmorphic one in the tree (``max|tnp| =
    3.142`` ⇒ τ_frac = 1/2), so a τ convention error shows up on it and
    nowhere else.
    """
    for deck in DECKS:
        d = _deck(deck)
        sym = SymMaps(d)
        assert sym.validate_atomic_symmetries(d) == [], deck
    si = _deck("si_cohsex_debug")
    assert int(si.ntran) == 48 and len(si.atom_types) == 2, (
        "PRECONDITION: 48 ops x 2 atoms is where the 96 comes from")
    assert float(np.abs(si.translations[:si.ntran]).max()) > 3.0, (
        "PRECONDITION: Si must still be the non-symmorphic deck (max|tnp| "
        "~ 3.142); with tnp = 0 this cell would not test the τ convention")


def test_the_wrong_rotation_direction_fails_half_the_mappings():
    """S4's constructible-FALSE — the 48/96 the docstrings cite.

    ``validate_atomic_symmetries`` applies ``inv(mtrx)·r + τ``.  Handing it
    a sym table whose ops have been INVERTED makes it effectively apply
    ``mtrx·r + τ`` — the wrong direction — and Fd-3m closure breaks.  The
    corruption is on a COPY of the header; the fixture is read-only.

    The routine reports the FIRST failing atom per op and moves on, so the
    count is ops-that-failed, not atom-mappings.  Asserted as "about half
    the ops", which is the same statement the docstrings make in atom
    units: 48 of 96 mappings is 24 of 48 ops.
    """
    d = _deck("si_cohsex_debug")
    sym = SymMaps(d)
    assert sym.validate_atomic_symmetries(d) == []
    inverted = np.rint(np.linalg.inv(
        np.asarray(d.sym_matrices[:d.ntran], dtype=float))).astype(
        d.sym_matrices.dtype)
    failures = sym.validate_atomic_symmetries(d.replacing(
        sym_matrices=inverted))
    n_ops = int(d.ntran)
    assert len(failures) >= n_ops // 4, (
        f"the wrong rotation direction must break Fd-3m closure; only "
        f"{len(failures)} of {n_ops} ops failed")
    assert len(failures) < n_ops, (
        f"ALL {n_ops} ops failing would mean the corruption broke something "
        f"other than the direction — the identity must still map")
    assert "sym " in failures[0] and "atom " in failures[0], (
        f"a failure must name the op and the atom: {failures[0]!r}")


def test_a_single_corrupted_op_is_caught_and_the_others_are_not():
    """One bad op, one failure — localisation, not a blanket refusal.

    A validator that failed everything on one bad op would be useless for
    finding which one.
    """
    d = _deck("si_cohsex_debug")
    sym = SymMaps(d)
    mt = np.array(d.sym_matrices)
    mt[1] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=mt.dtype)
    failures = sym.validate_atomic_symmetries(d.replacing(sym_matrices=mt))
    assert len(failures) == 1, f"expected exactly one bad op: {failures}"
    assert failures[0].startswith("sym 1:"), failures[0]


# ---------------------------------------------------------------------------
# The star helpers against a REAL SymMaps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", DECKS)
def test_the_round_trip_is_exact_on_the_production_tables(deck):
    """I1 on the real thing, not on a hand-typed copy of it.

    The star-contract tier proves the algebra; this proves the algebra is
    being handed the tables ``SymMaps`` actually builds — including Si's
    64-row, 8-star, 24-member-max case, which no hand-typed table in this
    suite reproduces.
    """
    d = _deck(deck)
    sym = SymMaps(d)
    irr = np.asarray(sym.irr_idx_k)
    sidx = np.asarray(sym.sym_idx_k)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    _, xor = _star_conj_flags(irr, sidx, nss)
    rng = np.random.default_rng(11)
    base = {}
    for label in sorted({int(v) for v in irr}):
        m = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
        base[label] = 0.5 * (m + m.conj().T)
    A = np.stack([np.conj(base[int(irr[k])]) if xor[k] else base[int(irr[k])]
                  for k in range(len(irr))])
    km = KStarMap.from_sym(sym, nss)
    sel, labels = star_select(A, irr)
    assert np.array_equal(np.asarray(km.select(A)), np.asarray(sel))
    back = star_broadcast(sel, irr, sidx, nss, irr_labels=labels, trs_reference="star_row")
    assert np.array_equal(np.asarray(back), A)
    assert km.spread(A) == 0.0

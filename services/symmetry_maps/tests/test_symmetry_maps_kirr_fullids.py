"""``SymMaps.kirr_fullids`` — the wedge row map, and the twin that says it
is not an enumeration.

WHAT THE MAP IS FOR.  Every array inside the GW driver lives on the
unfolded full BZ; BerkeleyGW's ``eqp0.dat`` / ``eqp1.dat`` and the
``WFN_qp.h5`` writer list only the irreducible wedge, in the order the WFN
file stores it.  ``kirr_fullids[i]`` is the bridge: the row of
``unfolded_kpts`` that IS the file's irreducible k-point ``i``.  Nothing
downstream re-derives it, and nothing downstream checks it — ``gw_output``
gathers every eqp column with it and ``dump_qp_wfn_artifacts`` reads the
rotation ``U`` at those rows — so if it names the wrong row the outputs are
mislabelled while every array behind them is correct, which is a defect
that survives norms, degeneracy checks and the electron count.

WHY THE FILE EXISTS (2026-08-08).  Until ``fix/kirr-fullids-2026-08-08``
the map was derived from the STAR LABELS: "the first full-BZ row carrying
irreducible label ``i``", read out of ``irr_idx_k``, with an identity
fallback ``kirr_fullids[i] = i`` for any label no row carried.  Both halves
are unsound, because ``irr_idx_k`` is not required to use every label —
``find_symmetry_ops_simple``'s op-selection policy has no ``break``, so a
full-BZ row reachable from more than one stored IBZ point is labelled with
the HIGHEST of them and the lower ones are orphaned.  MEASURED at bc37b4d3:
``gnppm_debug`` and ``bispinor_debug`` got ``[0,1,1,3,4,5,3,5,4]`` where the
answer is ``[0..8]``, ``cohsex_debug`` got ``[0,1,1,4]`` for ``[0,1,2,4]``,
and only ``si_cohsex_debug`` — 48 operations, eight disjoint stars, no
orphaned label — was right, by luck rather than by construction.

THE CELLS BELOW ARE ORDERED FROM CONTRACT TO ANCHOR TO REFUSAL.  The first
is the contract itself and it discriminates on three of the four decks.
The second is the Si 4x4x4 anchor, pinned against BerkeleyGW's own mean
field so the claim "these are the right eight k" is settled by an external
code and not by the class under test.  The third is the red twin the
service form asks for: a permuted IBZ list, where an answer of
``[0..N-1]`` is exactly the wrong one.  The fourth is the refusal that
replaced the silent fallback.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from _deck_stub import DECKS, deck_available, deck_path, read_deck
from symmetry_maps import SymMaps

#: Ry → eV, the value BerkeleyGW itself prints its ``Emf`` column with.
_RY2EV = 13.60569300965081

#: The Si 4x4x4 wedge, in ``wfn.kpoints`` order.  This is the "true
#: first-occurrence list" of BGW_CD_COMPARISON_DESIGN §7.7.1 — the eight
#: rows of the 64-point uniform grid that carry the file's own IBZ points.
_SI_WEDGE = [0, 1, 2, 5, 6, 7, 10, 27]


def _deck(name):
    pytest.importorskip("h5py", reason="h5py is not importable")
    if not deck_available(name):
        pytest.skip(f"no {name} WFN in this checkout (fixture blobs absent)")
    return read_deck(name)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deck", DECKS)
def test_the_wedge_row_is_the_stored_k_itself(deck):
    """``unfolded_kpts[kirr_fullids[i]] == wfn.kpoints[i]``, on every deck.

    This is the whole contract, written the way its consumers state it, and
    it is checked modulo a reciprocal lattice vector because the uniform
    grid is wrapped into ``[0, 1)`` and a stored k need not be.

    The tolerance is 1e-6 — ``find_symmetry_ops_simple``'s, so the two
    tables that address the same grid are built with the same number.  It
    is not tight to machine precision on purpose: the 3x3x1 decks store
    their k with eight decimal digits, so the exact grid value 1/3 arrives
    as 0.33333333 and the residual is 3.3e-10.  On ``si_cohsex_debug``,
    whose k are exact binary fractions, the same quantity is 1.1e-16.
    """
    h = _deck(deck)
    sym = SymMaps(h)
    kf = np.asarray(sym.kirr_fullids)
    assert kf.shape == (int(h.nkpts),)

    delta = np.asarray(sym.unfolded_kpts)[kf] - np.asarray(h.kpoints, float)
    delta -= np.rint(delta)
    worst = np.max(np.abs(delta), axis=1)
    bad = np.where(worst > 1e-6)[0]
    assert bad.size == 0, (
        f"{deck}: kirr_fullids names a row that is not the stored k at IBZ "
        f"{bad.tolist()} — wanted {np.asarray(h.kpoints)[bad].tolist()}, "
        f"row {kf[bad].tolist()} holds "
        f"{np.asarray(sym.unfolded_kpts)[kf[bad]].tolist()}")


@pytest.mark.parametrize("deck", DECKS)
def test_no_two_irreducible_points_share_a_wedge_row(deck):
    """Injectivity, stated separately because it is the visible symptom.

    Two IBZ points mapped to one full-BZ row is what an ``eqp0.dat`` reader
    sees: a duplicated block of numbers under two different k labels, and
    some other k of the wedge missing entirely.  It follows from the cell
    above — distinct k cannot both equal one row — but it is the property a
    person reading the output would name, so it gets its own failure
    message.
    """
    h = _deck(deck)
    kf = np.asarray(SymMaps(h).kirr_fullids)
    rows, counts = np.unique(kf, return_counts=True)
    dup = rows[counts > 1]
    assert dup.size == 0, (
        f"{deck}: full-BZ rows {dup.tolist()} are each claimed by more than "
        f"one irreducible k (kirr_fullids = {kf.tolist()}); the wedge output "
        f"would print that row twice and drop {int(h.nkpts) - rows.size} k")


# ---------------------------------------------------------------------------
# The Si 4x4x4 anchor, adjudicated by BerkeleyGW
# ---------------------------------------------------------------------------

def test_the_si_4x4x4_wedge_is_the_true_first_occurrence_list():
    """The pinned value, and the identity operation that goes with it.

    ``[0, 1, 2, 5, 6, 7, 10, 27]`` is the list BGW_CD_COMPARISON_DESIGN
    §7.7.1 names as the star's true first occurrences on this deck.  The
    second assertion is the property ``dump_qp_wfn_artifacts`` needs and
    cannot state for itself: each of those rows is reached from its IBZ
    parent by the IDENTITY operation, so the ψ and U taken there are the
    STORED ones and not a rotated or time-reversed image.  It holds on this
    deck because its eight stars are disjoint; it is asserted here rather
    than assumed, and it is deliberately NOT asserted on the 3x3x1 decks,
    where the op-selection policy assigns a rotation (and on
    ``cohsex_debug`` a time-reversal row) to some wedge rows even though
    the k itself is exactly right.  That difference is a property of the
    register-don't-touch policy, not of this map.
    """
    h = _deck("si_cohsex_debug")
    sym = SymMaps(h)
    assert np.asarray(sym.kirr_fullids).tolist() == _SI_WEDGE
    assert np.asarray(sym.sym_idx_k)[np.asarray(sym.kirr_fullids)].tolist() \
        == [0] * 8


def test_the_si_wedge_rows_carry_berkeleygws_own_mean_field():
    """The external anchor: BGW's ``Emf`` at all eight k, all sixteen bands.

    The point of the cell is that it is not self-referential.  The full-BZ
    mean-field table is built the way the driver builds it — every full-BZ
    row inherits its star parent's eigenvalues through ``irr_idx_k`` — and
    is then GATHERED AT ``kirr_fullids``, exactly as ``write_results``
    gathers its eqp columns.  What comes out is compared against
    ``bgw_sigma_hp_noavg.dat``, which is BerkeleyGW output transcribed
    verbatim and never touched by LORRAX.

    This is the cell that would have caught §7.7.1's symptom directly: a
    wedge row carrying a different member of the same star reports that
    star's other k's energies, so the four-fold multiplets at X arrive as
    two two-fold ones and the residual is eV-scale rather than µeV-scale.
    """
    import h5py

    h = _deck("si_cohsex_debug")
    sym = SymMaps(h)

    with h5py.File(deck_path("si_cohsex_debug"), "r") as f:
        el_ry = np.asarray(f["mf_header/kpoints/el"][()])[0]   # (nk, nb), Ry

    # The driver's unfold: a full-BZ row's energies are its star parent's.
    enk_full_ev = el_ry[np.asarray(sym.irr_idx_k)] * _RY2EV
    emitted = enk_full_ev[np.asarray(sym.kirr_fullids)][:, :16]

    ref = deck_path("si_cohsex_debug", "bgw_sigma_hp_noavg.dat")
    assert os.path.isfile(ref), f"missing BerkeleyGW anchor {ref}"
    rows = np.loadtxt(ref, comments="#")
    bgw_emf = np.full((8, 16), np.nan)
    for r in rows:
        bgw_emf[int(r[0]) - 1, int(r[4]) - 1] = r[5]
    assert not np.isnan(bgw_emf).any(), "the BGW anchor is not 8 k x 16 bands"

    worst = np.max(np.abs(emitted - bgw_emf))
    assert worst < 1e-5, (
        "the emitted wedge does not carry BerkeleyGW's mean field: worst "
        f"|Emf_LORRAX - Emf_BGW| = {worst:.6e} eV at row/band "
        f"{np.unravel_index(int(np.argmax(np.abs(emitted - bgw_emf))), emitted.shape)}"
        f"; kirr_fullids = {np.asarray(sym.kirr_fullids).tolist()}")


# ---------------------------------------------------------------------------
# The red twin, and the refusal
# ---------------------------------------------------------------------------

class _Stub:
    """The eleven attributes ``SymMaps`` reads, on a 4x1x1 chain.

    Deliberately synthetic and fixture-free: the red twin has to run in a
    checkout with no WFN blobs, because the property it defends — that this
    map is not an enumeration of the IBZ list — is a property of the code
    and not of any deck.  ``{I, -I}`` is the smallest symmetry group that
    makes the general branch run at all (``ntran <= 1`` takes the trivial
    path, where the full grid IS the stored list) and it folds the 4-point
    chain to three stars, so a permuted list has orphaned labels — the
    ingredient the old construction needed to go wrong.
    """

    kgrid = np.array([4, 1, 1], dtype=np.int32)
    shift = np.zeros(3)
    ntran = 2
    sym_matrices = np.stack([np.eye(3, dtype=np.int32),
                             -np.eye(3, dtype=np.int32)])
    translations = np.zeros((2, 3))
    avec = np.eye(3)
    atom_types = np.array([1])
    atom_crys = np.zeros((1, 3))
    trs_holds = True

    def __init__(self, kpoints):
        self.kpoints = np.asarray(kpoints, dtype=float)
        self.nkpts = int(self.kpoints.shape[0])


_CHAIN = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0],
                   [0.50, 0.0, 0.0], [0.75, 0.0, 0.0]])


@pytest.mark.parametrize("perm,expected", [
    ([0, 1, 2, 3], [0, 1, 2, 3]),
    ([3, 2, 1, 0], [3, 2, 1, 0]),
    ([2, 3, 0, 1], [2, 3, 0, 1]),
])
def test_a_permuted_irreducible_list_is_followed_not_enumerated(perm, expected):
    """THE RED TWIN.  The map tracks the stored order; it does not invent one.

    Only the first row of the table is the identity permutation, and it is
    there as the control.  The other two hand ``SymMaps`` the same four k in
    a different order, and the answer has to move with them — a map that
    returned ``[0, 1, 2, 3]`` for a REVERSED list would be handing every
    wedge output the mirror image of the k it labelled.

    MEASURED at bc37b4d3, the old star-label construction on these three
    inputs: ``[0,1,2,1]``, ``[0,2,1,0]``, ``[2,1,0,1]``.  All three are
    wrong, all three duplicate a row, and none of them raised.
    """
    sym = SymMaps(_Stub(_CHAIN[perm]))
    got = np.asarray(sym.kirr_fullids).tolist()
    assert got == expected
    if perm != [0, 1, 2, 3]:
        assert got != list(range(4)), (
            "kirr_fullids came out as a plain enumeration for a permuted "
            "irreducible list — the exact failure mode this twin exists for")


def test_an_irreducible_point_off_the_uniform_grid_is_refused():
    """The fallback is gone; a k that is not on the grid raises by name.

    What used to happen instead was ``kirr_fullids[i] = i``: an unrelated
    row, silently, for a file whose k-list and whose kgrid/shift disagree.
    There is no defensible row to return in that case — every wedge-shaped
    output would be indexed with a guess — so the class refuses and the
    message says which k and how far off it is.
    """
    off_grid = _CHAIN.copy()
    off_grid[2, 0] = 0.4   # not a multiple of 1/4
    with pytest.raises(RuntimeError, match="not on the uniform"):
        SymMaps(_Stub(off_grid))

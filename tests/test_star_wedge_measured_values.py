"""The star-wedge measurements of 2026-08-17, pinned against the committed decks.

WHY THIS FILE EXISTS.  The kin-ion k-basis investigation of 2026-08-17
produced a set of numbers that live nowhere else: the 170 per-artifact
unfolding cells that used to surround them were deleted on purpose
(``chore/cull-per-artifact-unfold-tests-2026-08-17``), because they tested
plumbing.  The numbers are not plumbing.  This file keeps them, and nothing
else — one cell per measured fact, each naming the log it was measured in.

THE ANCHOR IS THE COMMITTED FIXTURE, ALWAYS.  Every cell below reads
``tests/regression/<deck>/`` and nothing that a regeneration produced.  That
is deliberate: the run directories these numbers came out of are on
``$SCRATCH`` and are purge-eligible, so a cell that needed one would be a
cell that stops meaning anything the week the purge runs.  What is in git is
what is asserted.

WHAT IS *NOT* HERE.  The centroid orbit-closure gate is
``services/symmetry_maps/tests/test_symmetry_maps_closure.py`` and stays the
single sym test; the six-row centroid resolution table is
``test_symmetry_maps_qgrid_resolution.py``.  Neither is repeated here.

PROVENANCE.  Logs ``L1``-``L19`` under
``reports/kin_ion_star_wedge_2026-08-17/artifacts/`` in the sandbox repo,
mirrored durably to
``/global/cfs/cdirs/m4598/jackm/kin_ion_star_wedge_2026-08-17/``.  Each
section names its own log.  Where a number is quoted below it is quoted to
the digits the log printed it in.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from symmetry_maps import SymMaps                               # noqa: E402
from symmetry_maps.maps import _star_row_order                  # noqa: E402

h5py = pytest.importorskip("h5py")

from file_io.kin_ion import (                                   # noqa: E402
    IRR_IDX_DATASET, K_STORAGE_ATTR, K_STORAGE_IBZ,
    K_STORAGE_VERSION, K_STORAGE_VERSION_ATTR, N_SYM_SPATIAL_ATTR,
    SYM_IDX_DATASET, read_full_bz_dataset, read_star_map,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REG = os.path.join(_REPO, "tests", "regression")


# ===========================================================================
# § THE SIX DECKS, AS MEASURED
# ===========================================================================
# ``L15_wedge_tables.log`` (all six, one line each) and ``L9_star_probe.log``
# (the TRS row counts, printed there as ``TIME-REVERSED ROWS``).  Read the
# table as: the WFN carries ``nk_red`` k, the symmetry service resolves them
# into ``n_orbits`` stars over ``nk_tot`` full-BZ k, and ``n_trs`` of those
# full-BZ rows are reached only through time reversal.
#
# THE POINT OF THE TABLE is the first three decks, where ``nk_red`` and
# ``n_orbits`` DIFFER.  There the WFN's own k-set is not the star wedge, and
# a sweep over ``wfn.load(k="ibz")`` computes rows that the reader then
# overwrites with its own unfold — the defect
# ``kin_ion: sweep and store the STAR WEDGE`` fixed.  On the last three
# (``nk_red == n_orbits``) the wedge is the k-set, the k-spec is returned
# unchanged, and those decks take the literal pre-fix path byte for byte.
#
#            deck                  wfn            nk_red n_orbits nk_tot n_trs
_DECKS = (
    ("gnppm_debug",      "WFN.h5",       9,  5,  9, 4),
    ("bispinor_debug",   "WFN.h5",       9,  5,  9, 4),
    ("cohsex_debug",     "WFNsmall.h5",  4,  3,  9, 3),
    ("si_cohsex_debug",  "WFN.h5",       8,  8, 64, 0),
    ("si_bse_debug",     "WFN.h5",       8,  8, 64, 0),
    ("hbn_cohsex_debug", "WFN.h5",      18, 18, 18, 0),
)
_DECK_IDS = [d[0] for d in _DECKS]

#: The three where the two wedges have different lengths — the decks that
#: make every cell in this file mean something.  ``cohsex_debug`` and
#: ``bispinor_debug`` are also the two that exercise the ANTIUNITARY branch
#: with a nontrivial star structure (3 and 4 time-reversed rows).
_SPLIT_DECKS = tuple(d for d in _DECKS if d[2] != d[3])
#: The three cut at a true IBZ, which must keep the old path exactly.
_IDENTITY_DECKS = tuple(d for d in _DECKS if d[2] == d[3])


class _Header:
    """The eleven attributes ``SymMaps`` reads, straight out of ``mf_header``.

    A header stub rather than ``WfnLoader`` so a deck costs milliseconds on a
    68 MiB file and depends on nothing but h5py.  The service suite's
    ``test_symmetry_maps_deck_tables.py`` carries the parity arm proving the
    stub and the production loader build the same tables; this file leans on
    that rather than repeating it.
    """

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            g = f["mf_header"]
            avec = g["crystal/avec"][:]
            apos = g["crystal/apos"][:]
            self.kpoints = g["kpoints/rk"][:]
            self.kgrid = g["kpoints/kgrid"][:]
            self.shift = g["kpoints/shift"][:]
            self.nkpts = int(g["kpoints/nrk"][()])
            self.ntran = int(g["symmetry/ntran"][()])
            self.sym_matrices = g["symmetry/mtrx"][:]
            self.translations = g["symmetry/tnp"][:]
            self.avec = avec
            self.atom_types = g["crystal/atyp"][:]
            self.atom_crys = np.einsum("ij,kj->ki",
                                       np.linalg.inv(avec).T, apos)
            self.trs_holds = True


def _sym_or_skip(deck, wfn):
    """``SymMaps`` for a committed deck, or skip if the blob is absent."""
    path = os.path.join(_REG, deck, wfn)
    if not os.path.isfile(path):
        pytest.skip(f"no {deck}/{wfn} in this checkout (fixture blobs absent)")
    return SymMaps(_Header(path))


def _tables(sym):
    """``(irr, sidx, nss, par, trs)`` — the five arrays every cell here uses.

    ``par[ik]`` is the full-BZ row that is ``ik``'s STAR PARENT, and
    ``trs[ik]`` says whether reaching ``ik`` from it took time reversal.
    Both come straight off ``SymMaps``: ``par = kirr_fullids[irr_idx_k]``
    composes "which star is ``ik`` in" with "which full-BZ row is that star's
    IBZ point", and ``sym_idx_k >= n_sym_spatial`` is the same conjugation
    predicate ``unfold_psi`` uses when it BUILDS psi(Sk).
    """
    irr = np.asarray(sym.irr_idx_k, dtype=np.int32)
    sidx = np.asarray(sym.sym_idx_k, dtype=np.int32)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    par = np.asarray(sym.kirr_fullids, dtype=np.int64)[irr]
    return irr, sidx, nss, par, sidx >= nss


def _star_residual(A, par, trs):
    """``(diag, offdiag, trs_offdiag)`` of ``A`` against its own star relation.

    THE DIRECT PARENT TEST, and deliberately not the wedge API:
    ``ref[ik] = conj^TRS(A[par[ik]])``, elementwise ``|ref - A|``, split by
    whether the element is on the band diagonal.  Written with numpy and
    ``SymMaps`` tables only, so it cannot pass by agreeing with the routine
    it is checking.  ``trs_offdiag`` is the same maximum restricted to the
    time-reversed rows, which is where an antiunitary bug lives and where a
    unitary-only one cannot reach.
    """
    ref = A[par]
    ref = np.where(trs[(...,) + (None,) * (A.ndim - 1)], np.conj(ref), ref)
    d = np.abs(ref - A)
    eye = np.eye(A.shape[-2], A.shape[-1], dtype=bool)
    trs_off = float(d[trs][..., ~eye].max()) if trs.any() else 0.0
    return float(d[..., eye].max()), float(d[..., ~eye].max()), trs_off


def _write_wedge_file(path, arrays, *, irr_wedge, sidx, nss):
    """A ``kin_ion.h5`` written the way the FIXED generator writes one.

    Wedge slab, renumbered table, the stamp trio.  h5py only — no FFI, no
    device mesh — so this is the cheap half of the format contract.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset(IRR_IDX_DATASET, data=np.asarray(irr_wedge, np.int32))
        f.create_dataset(SYM_IDX_DATASET, data=np.asarray(sidx, np.int32))
        for name, arr in arrays.items():
            ds = f.create_dataset(name, data=arr, dtype=np.complex128)
            ds.attrs[K_STORAGE_ATTR] = K_STORAGE_IBZ
            ds.attrs[K_STORAGE_VERSION_ATTR] = int(K_STORAGE_VERSION)
            ds.attrs[N_SYM_SPATIAL_ATTR] = int(nss)
    return str(path)


# ===========================================================================
# § 1  THE SWEEP COUNTS AND THE WEDGE-TABLE IDENTITY
# ===========================================================================

@pytest.mark.parametrize("deck,wfn,nk_red,n_orbits,nk_tot,n_trs", _DECKS,
                         ids=_DECK_IDS)
def test_the_star_wedge_sweep_counts_are_what_was_measured(
        deck, wfn, nk_red, n_orbits, nk_tot, n_trs):
    """MEASURED ``L15_wedge_tables.log``, six lines, ``VERDICT PASS``::

        gnppm_debug       nk_red= 9 n_orbits= 5 nk_tot= 9 swept= 5  9->5 k
        bispinor_debug    nk_red= 9 n_orbits= 5 nk_tot= 9 swept= 5  9->5 k
        cohsex_debug      nk_red= 4 n_orbits= 3 nk_tot= 9 swept= 3  4->3 k
        si_cohsex_debug   nk_red= 8 n_orbits= 8 nk_tot=64 swept= 8  8->8 k
        si_bse_debug      nk_red= 8 n_orbits= 8 nk_tot=64 swept= 8  8->8 k
        hbn_cohsex_debug  nk_red=18 n_orbits=18 nk_tot=18 swept=18 18->18 k

    The saving is the point on the first three and the ABSENCE of a saving is
    the point on the last three: ``8->8`` and ``18->18`` are what "this deck
    takes the literal old path" looks like from here, and if one of them ever
    became ``18->17`` a deck that is supposed to be untouched would have
    started moving.

    The TRS counts are ``L9_star_probe.log``'s ``TIME-REVERSED ROWS`` field
    (``4/9``, ``4/9``, ``3/9``, ``0/64``, ``0/64``, ``0/18``).
    """
    sym = _sym_or_skip(deck, wfn)
    irr, sidx, nss, _par, trs = _tables(sym)

    assert int(sym.nk_red) == nk_red, f"{deck}: nk_red moved"
    assert int(sym.nk_tot) == nk_tot, f"{deck}: nk_tot moved"
    assert len(set(int(v) for v in irr)) == n_orbits, (
        f"{deck}: the service resolves a different number of stars than the "
        f"{n_orbits} measured on 2026-08-17")
    assert int(trs.sum()) == n_trs, (
        f"{deck}: {int(trs.sum())} time-reversed rows against {n_trs} "
        f"measured — the antiunitary branch is reached a different number "
        f"of times than when these numbers were taken")


@pytest.mark.parametrize("deck,wfn,nk_red,n_orbits,nk_tot,n_trs", _DECKS,
                         ids=_DECK_IDS)
def test_renumber_composed_with_rows_returns_the_parent_map_exactly(
        deck, wfn, nk_red, n_orbits, nk_tot, n_trs):
    """THE identity, ``rows[irr_idx_wedge] == SymMaps.irr_idx_k``, all six decks.

    MEASURED ``L15_wedge_tables.log``: ``compose=True`` on every deck, beside
    ``count=True order=True dense=True star_tables_agrees=True``.

    WHY IT IS THE ONE THAT MATTERS.  ``count`` and ``dense`` only say the
    table is well formed.  ``compose`` says it addresses the RIGHT star: the
    renumbered table composed with the rows it renumbers has to give the
    parent map back exactly, or the file's unfold reaches a different star
    than the sweep computed.  That failure is invisible in every diagonal
    observable — the diagonal of a wrong star member is still a plausible
    number — which is why it is asserted here rather than left to a physics
    gate to notice.

    ``compact_star_tables`` is the production renumbering (one rule in the
    tree; ``sigma_mnk.h5`` has used it since its own wedge storage landed)
    and ``gw.kin_ion_io.star_wedge_rows`` is a four-line wrapper over it.
    Checked here against the SERVICE's own ``_star_row_order`` so the cell
    runs without the FFI gate that importing ``gw.kin_ion_io`` fires; the
    wrapper's agreement is the ``star_tables_agrees`` arm of the same log.
    """
    from file_io.sigma_output import compact_star_tables

    sym = _sym_or_skip(deck, wfn)
    irr, _sidx, _nss, _par, _trs = _tables(sym)
    rows_to_keep, irr_idx_wedge = compact_star_tables(irr)
    rows = irr[rows_to_keep]

    assert rows.size == n_orbits, (
        f"{deck}: swept {rows.size} k for {n_orbits} orbits")
    assert np.array_equal(rows, _star_row_order(irr)[1]), (
        f"{deck}: the swept rows must be star_select's own first-occurrence "
        f"order — sorting them returns another star's matrix wherever the "
        f"two differ (on gnppm_debug the labels are [0, 2, 6, 8, 7], NOT the "
        f"sorted [0, 2, 6, 7, 8])")
    assert np.array_equal(np.unique(irr_idx_wedge), np.arange(n_orbits)), (
        f"{deck}: the filed table must index the STORED rows densely, which "
        f"is what read_star_map's star count checks")
    assert np.array_equal(rows[irr_idx_wedge], irr), (
        f"{deck}: renumber-composed-with-rows is not the parent map")


@pytest.mark.parametrize("deck,wfn,nk_red,n_orbits,nk_tot,n_trs",
                         _SPLIT_DECKS,
                         ids=[d[0] for d in _SPLIT_DECKS])
def test_these_three_decks_really_do_separate_the_two_wedges(
        deck, wfn, nk_red, n_orbits, nk_tot, n_trs):
    """PRECONDITION for everything above, stated so a fixture change shows.

    If a regenerated WFN ever made ``nk_red == n_orbits`` on these three, the
    cells above would still pass while testing nothing at all: the file wedge
    and the star wedge would coincide and the renumbering would be the
    identity.  ``gnppm_debug`` 9 k over 5 stars, ``bispinor_debug`` 9 over 5,
    ``cohsex_debug`` 4 over 3 (``L15_wedge_tables.log``).
    """
    sym = _sym_or_skip(deck, wfn)
    irr, _sidx, _nss, _par, _trs = _tables(sym)
    assert len(set(int(v) for v in irr)) < int(sym.nk_red), (
        f"{deck}: nk_red {int(sym.nk_red)} == n_orbits, so this deck no "
        f"longer separates the file wedge from the star wedge")


@pytest.mark.parametrize("deck,wfn,nk_red,n_orbits,nk_tot,n_trs",
                         _IDENTITY_DECKS,
                         ids=[d[0] for d in _IDENTITY_DECKS])
def test_the_true_ibz_decks_keep_the_literal_old_path(
        deck, wfn, nk_red, n_orbits, nk_tot, n_trs):
    """The other half of the sweep claim: ``si_*`` and ``hbn_*`` do not move.

    On a WFN cut at a true IBZ the swept rows are ``arange(nk_red)`` and the
    renumbering is the identity, so the k-spec is returned as ``"ibz"``
    unchanged and the deck keeps the same loader cache key, the same read and
    the same scan.  MEASURED ``L15_wedge_tables.log``: ``8->8``, ``8->8``,
    ``18->18``.  This cell is what makes "byte-for-byte the old path" an
    assertion instead of a claim.

    MIND THE TWO INDEX SPACES, which is the trap this cell was written wrong
    in first.  ``compact_star_tables`` returns rows of the FULL BZ — on
    ``si_cohsex_debug`` they are ``[0, 1, 2, 5, 6, 7, 10, 27]``, the same
    eight ``test_symmetry_maps_kirr_fullids.py`` pins as ``_SI_WEDGE``.
    ``star_wedge_rows`` then maps them into the WFN's OWN k axis by indexing
    the label table, ``irr[rows_to_keep]``, and it is THAT which is
    ``arange(nk_red)`` on a true-IBZ deck.  Asserting on the raw full-BZ rows
    instead reads as "this deck moved" on a deck that did not move.
    """
    from file_io.sigma_output import compact_star_tables

    sym = _sym_or_skip(deck, wfn)
    irr, _sidx, _nss, _par, _trs = _tables(sym)
    rows_to_keep, irr_idx_wedge = compact_star_tables(irr)
    wfn_rows = irr[rows_to_keep]

    assert np.array_equal(wfn_rows, np.arange(int(sym.nk_red))), (
        f"{deck}: the swept rows are no longer the WFN's own k axis, so this "
        f"deck has started taking the wedge path it is supposed to skip")
    assert np.array_equal(irr_idx_wedge, irr), (
        f"{deck}: the renumbering is no longer the identity")


# ===========================================================================
# § 2  A WEDGE-STORED FILE IS EXACTLY STAR-COVARIANT WHEN READ BACK
# ===========================================================================

@pytest.mark.parametrize("deck,wfn,nk_red,n_orbits,nk_tot,n_trs",
                         _SPLIT_DECKS,
                         ids=[d[0] for d in _SPLIT_DECKS])
def test_a_wedge_stored_slab_reads_back_exactly_star_covariant(
        deck, wfn, nk_red, n_orbits, nk_tot, n_trs, tmp_path):
    """MEASURED ``0.000000e+00``, diagonal AND off-diagonal, on FRESH files.

    ``L2_accept.log`` (gnppm_debug) and ``L17_parent_two.log`` (cohsex_debug,
    bispinor_debug), on ``kin_ion`` AND ``v_hartree``, through the direct
    parent test with no wedge API::

        gnppm_debug     UNFOLDED kin_ion    diag=0.000000e+00 offdiag=0.000000e+00
                        UNFOLDED v_hartree  diag=0.000000e+00 offdiag=0.000000e+00
        cohsex_debug    UNFOLDED kin_ion    diag=0.000000e+00 offdiag=0.000000e+00
                        UNFOLDED v_hartree  diag=0.000000e+00 offdiag=0.000000e+00
        bispinor_debug  UNFOLDED kin_ion    diag=0.000000e+00 offdiag=0.000000e+00
                        UNFOLDED v_hartree  diag=0.000000e+00 offdiag=0.000000e+00

    against the same probe's BASELINE arm on the pre-fix generator
    (``L2_accept.log``, gnppm_debug): ``STORED slab kin_ion offdiag=3.315331e+01``,
    ``v_hartree offdiag=3.360989e+01`` Ry.  Thirty-three Rydberg is what the
    zero is being measured against.

    WHY THE ZERO MEANS SOMETHING.  It is exact, not "small": the reader's
    gather and the parent gather are the same gather, so any disagreement at
    all is a wrong star and the tolerance is 0.  And the star relation on
    these three decks is ANTIUNITARY on 3 (cohsex), 4 (bispinor) and 4
    (gnppm) of nine full-BZ rows — the ``[TRS rows offdiag=0.000000e+00]``
    field in every line above — so a conjugation dropped anywhere in the
    round trip lands off-diagonal and cannot hide.  A deck with no
    time-reversed rows would give the same zero for a much weaker reason,
    which is why ``si_*`` and ``hbn_*`` are not parametrized here.

    The fresh files themselves are regeneration products on purge-eligible
    scratch, so what is pinned is the INVARIANT that makes them zero: a slab
    stored on the star wedge with the renumbered table beside it comes back
    off ``read_full_bz_dataset`` exactly star-covariant, on synthetic data
    that cannot be accidentally symmetric.
    """
    from file_io.sigma_output import compact_star_tables

    sym = _sym_or_skip(deck, wfn)
    irr, sidx, nss, par, trs = _tables(sym)
    _rows, irr_idx_wedge = compact_star_tables(irr)

    assert int(trs.sum()) == n_trs > 0, (
        f"PRECONDITION: {deck} must exercise the antiunitary branch, or the "
        f"zero below is a much weaker statement than the docstring claims")

    # Synthetic, non-Hermitian, complex, distinct per row: a slab that is
    # star-covariant only because the reader made it so.
    rng = np.random.default_rng(0)
    nb = 6
    slab = (rng.standard_normal((n_orbits, nb, nb))
            + 1j * rng.standard_normal((n_orbits, nb, nb)))
    path = _write_wedge_file(tmp_path / "kin_ion.h5",
                             {"kin_ion": slab, "v_hartree": slab * 1.7 - 0.3},
                             irr_wedge=irr_idx_wedge, sidx=sidx, nss=nss)

    for name in ("kin_ion", "v_hartree"):
        full = read_full_bz_dataset(path, name)
        assert full.shape == (nk_tot, nb, nb), (
            f"{deck}/{name}: the reader returned {full.shape}, not the full BZ")
        diag, off, trs_off = _star_residual(full, par, trs)
        assert (diag, off, trs_off) == (0.0, 0.0, 0.0), (
            f"{deck}/{name}: a wedge-stored slab read back is NOT exactly "
            f"star-covariant — diag={diag:.6e} offdiag={off:.6e} "
            f"[TRS rows offdiag={trs_off:.6e}].  Measured 0.000000e+00 on "
            f"all three on 2026-08-17 (L2_accept.log, L17_parent_two.log)")


def test_read_star_map_counts_distinct_stars_not_the_max_label(tmp_path):
    """RED TWIN for the refusal that did not fire.

    ``gnppm_debug``'s own table: nine full-BZ k over five stars, labelled
    ``[0, 2, 2, 6, 8, 7, 6, 7, 8]``.  ``max + 1`` is 9 and the pre-fix slab
    had 9 rows, so the old spelling passed exactly the file it exists to
    catch — it tested the writers' arithmetic instead of their output.
    ``np.unique(irr).size`` is 5 and refuses.

    Runs without the FFI: the writer here is h5py and the reader under test
    is the pure-host one.
    """
    irr = np.array([0, 2, 2, 6, 8, 7, 6, 7, 8], dtype=np.int32)
    assert int(irr.max()) + 1 == irr.size, (
        "PRECONDITION: this is the one table max+1 gets wrong")
    path = _write_wedge_file(
        tmp_path / "kin_ion.h5",
        {"kin_ion": np.zeros((irr.size, 2, 2), dtype=np.complex128)},
        irr_wedge=irr, sidx=np.zeros(irr.size, dtype=np.int32), nss=1)
    with pytest.raises(ValueError, match="do not describe the same"):
        read_star_map(path, "kin_ion")


# ===========================================================================
# § 3  THE COMMITTED kin_ion.h5 FIXTURES ARE STALE, AND BY HOW MUCH
# ===========================================================================
# THESE ARE KNOWN-STALE VALUES, NOT ASPIRATIONS.  Four of the six committed
# ``kin_ion.h5`` predate wedge storage: they were computed independently at
# every full-BZ k, carry no ``k_storage`` attr, and are therefore read
# VERBATIM — which is correct, and is the reason the default matters.  Their
# rows do not satisfy the star relation, and the deviations below are the
# measurement of that, not a bug being tolerated.
#
# They are asserted rather than xfailed because their VALUE is the artifact.
# An xfail would record only "still wrong"; the number records how wrong, and
# a change in it is a finding either way — a fixture regenerated on the fixed
# generator should drop to 0, and any other movement means the committed
# blob or the symmetry tables under it changed without anyone saying so.
#
# Reinterpreting one of these files as compressible would move physics by up
# to 7.8 Ry.  The only thing standing between the reader and that is the
# missing-attr default, which is why ``stamp is absent`` is asserted here too.
#
#   deck                diag           offdiag        log
_FIXTURE_STALENESS = (
    ("gnppm_debug",      0.0,          3.557152e-14, "L9_star_probe.log"),
    ("bispinor_debug",   1.726205e-01, 6.681457e-02, "L17_parent_two.log"),
    ("cohsex_debug",     1.286205e-01, 7.781940e+00, "L17_parent_two.log"),
    ("si_cohsex_debug",  2.000875e-03, 1.756297e-03, "L10_si_resid.log"),
    ("si_bse_debug",     0.0,          0.0,          "L9_star_probe.log"),
    ("hbn_cohsex_debug", 0.0,          0.0,          "L9_star_probe.log"),
)


@pytest.mark.parametrize("deck,diag_ref,off_ref,log", _FIXTURE_STALENESS,
                         ids=[r[0] for r in _FIXTURE_STALENESS])
def test_the_committed_kin_ion_fixtures_carry_their_measured_deviation(
        deck, diag_ref, off_ref, log):
    """The four stale fixtures and the two clean ones, measured 2026-08-17.

    ``max|conj^TRS(A[parent]) - A|`` on the committed full-BZ blob, split
    diagonal / off-diagonal::

        deck              diag           offdiag        what it is
        gnppm_debug       0.000000e+00   3.557152e-14   round-off; effectively covariant
        bispinor_debug    1.726205e-01   6.681457e-02   PRE-STAMPING, independent full-BZ
        cohsex_debug      1.286205e-01   7.781940e+00   PRE-STAMPING, worst in the tree
        si_cohsex_debug   2.000875e-03   1.756297e-03   PRE-STAMPING, no TRS rows at all
        si_bse_debug      0.000000e+00   0.000000e+00   generator-written, wedge-exact
        hbn_cohsex_debug  0.000000e+00   0.000000e+00   generator-written, wedge-exact

    ``si_cohsex_debug`` is the informative one: 64 k, ZERO time-reversed
    rows, so its 2.000875e-03 Ry cannot be a conjugation error of any kind.
    It is the committed file's own independent computation disagreeing with
    its own symmetry — the negative control that says the other three
    numbers are not all one bug.  ``L10_si_resid.log`` decomposes it further:
    degenerate off-diagonal 1.522610e-03, nondegenerate off-diagonal
    1.756297e-03, and the eight rows that ARE their own parent are 0 by
    construction.

    ``si_bse_debug`` and ``hbn_cohsex_debug`` are the two this generator
    actually wrote, and they are exactly zero.  They are the positive control
    in the same cell, and they are the reason the four nonzero rows read as a
    property of the OLD fixtures rather than of the measurement.

    SUPERSEDES the withdrawn 31.05 / 12.44 / 8.04 triple, which came from
    ``unfold_file_wedge_to_full_bz`` dropping ``irr_labels`` and therefore
    addressing ``A_irr`` by position in ``np.unique`` instead of by label.
    Fixed in ``symmetry_maps: unfold_file_wedge_to_full_bz must pass
    irr_labels``; ``L8_two_unfolds.log`` shows the helper and the production
    reader agreeing to ``max|helper - production| = 0.000000e+00`` on all six
    decks after it.

    DO NOT REGENERATE THESE FIXTURES to make the numbers go to zero.  If that
    is ever done deliberately, this table is where the before/after lives.
    """
    wfn = dict((d[0], d[1]) for d in _DECKS)[deck]
    kin = os.path.join(_REG, deck, "kin_ion.h5")
    if not os.path.isfile(kin):
        pytest.skip(f"no {deck}/kin_ion.h5 in this checkout "
                    f"(fixture blobs absent)")
    sym = _sym_or_skip(deck, wfn)
    _irr, _sidx, _nss, par, trs = _tables(sym)

    with h5py.File(kin, "r") as f:
        assert K_STORAGE_ATTR not in f["kin_ion"].attrs, (
            f"{deck}/kin_ion.h5 has GROWN a {K_STORAGE_ATTR!r} stamp.  It is "
            f"now read as a wedge, which is a different array than every "
            f"number in this table was measured on — regenerate the table, "
            f"do not relax the assertion")
        A = np.asarray(f["kin_ion"][()])

    assert A.shape[0] == int(sym.nk_tot), (
        f"{deck}: the committed blob has {A.shape[0]} k rows against "
        f"nk_tot={int(sym.nk_tot)} — it is not a full-BZ file any more")

    diag, off, _trs_off = _star_residual(A, par, trs)
    # Four significant figures: these are file contents and float64 gathers,
    # so they are reproducible far tighter than this, but the point of the
    # cell is to catch a MOVE, not to police the last bit.
    assert diag == pytest.approx(diag_ref, rel=1e-4, abs=1e-12), (
        f"{deck}: committed kin_ion.h5 star deviation (diagonal) is "
        f"{diag:.6e}, was {diag_ref:.6e} in {log} on 2026-08-17")
    assert off == pytest.approx(off_ref, rel=1e-4, abs=1e-12), (
        f"{deck}: committed kin_ion.h5 star deviation (off-diagonal) is "
        f"{off:.6e}, was {off_ref:.6e} in {log} on 2026-08-17")

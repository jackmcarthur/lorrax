"""``qp_wfn_rotations.h5`` on the file wedge — the option, and its refusals.

WHAT THIS FILE IS ABOUT.  ``U_mnk`` is a stack of eigenvectors, so "the
star relation holds" is a statement about how a particular run produced
them, not a property of the physics.  The writer therefore does not TRUST
a caller that asks for the wedge: it runs the reader's own round trip on
the arrays in hand and keeps the wedge only if the reconstruction is
exact.  These cells pin that, in both directions, and pin the thing the
whole format rests on — that a file with no stamp is read as full-BZ.
"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from file_io.kin_ion import (                                    # noqa: E402
    IRR_IDX_DATASET, K_STORAGE_ATTR, K_STORAGE_FULL, K_STORAGE_IBZ,
    K_STORAGE_VERSION, K_STORAGE_VERSION_ATTR, N_SYM_SPATIAL_ATTR,
    SYM_IDX_DATASET,
)
from file_io.qp_wfn import (                                     # noqa: E402
    QP_ROT_FULL_BZ_DATASETS, QP_ROT_K_DATASETS, QP_ROTATIONS_K_STORAGE,
    qp_rotations_k_storage, read_qp_rotations_full_bz,
    write_qp_rotations_h5,
)


# ---------------------------------------------------------------------------
# A tiny star map, spelled out rather than derived
# ---------------------------------------------------------------------------
# 6 full-BZ k over 3 stars, on a FILE WEDGE of 3 rows.  Row 2's star has a
# TIME-REVERSED member, which is the half of the rule that a deck with no
# antiunitary row cannot test at all — and two of the six committed decks
# have none, so a fixture without one here would leave the conjugation
# untested everywhere.
_NK_FULL, _NK_RED, _NSS = 6, 3, 4
_IRR = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
_SYM = np.array([0, 1, 0, 2, 0, _NSS + 1], dtype=np.int32)   # row 5 is TRS
_KIRR_TO_KFULL = np.array([0, 2, 4], dtype=np.int32)
_TABLES = (_IRR, _SYM, _NSS)


def _star_consistent_payload(nb=3, seed=0):
    """Full-BZ arrays that ARE the unfold of their own wedge rows.

    Built by unfolding, so the cell tests the writer rather than the
    fixture: anything the writer keeps must come back through the reader.
    """
    rng = np.random.default_rng(seed)
    u_red = (rng.normal(size=(_NK_RED, nb, nb))
             + 1j * rng.normal(size=(_NK_RED, nb, nb)))
    e_red = rng.normal(size=(_NK_RED, nb))
    k_red = rng.normal(size=(_NK_RED, 3))
    from file_io.kin_ion import broadcast_ibz_to_full_bz as _bc
    # kpoints are NOT star-consistent and must not be: k is the one quantity
    # here that the symmetry operation changes.  Distinct rows on purpose, so
    # a cell that reduced them would fail.
    return (np.asarray(_bc(u_red, *_TABLES)),
            np.asarray(_bc(e_red, *_TABLES)),
            rng.normal(size=(_NK_FULL, 3)))


def _write(tmp_path, name, U, E, kpts, **kw):
    path = str(tmp_path / name)
    stored = write_qp_rotations_h5(
        path, U_mnk=U, E_qp_nk=E, band_start=0, band_stop=U.shape[-1],
        kpoints_crys=kpts, nkx=2, nky=1, nkz=1,
        kirr_to_kfull=_KIRR_TO_KFULL, **kw)
    return path, stored


# ---------------------------------------------------------------------------
# 1.  The round trip, per artifact — the acceptance the migration owes
# ---------------------------------------------------------------------------

def test_wedge_stored_file_unfolds_back_to_the_full_bz_arrays(tmp_path):
    """write wedge -> read -> unfold == the full-BZ array the old path wrote.

    BIT-IDENTICAL, not close: the unfold is a gather with a conditional
    ``conj``, and neither of those is allowed to move a float.  A tolerance
    here would hide a real reconstruction error behind an allowance nobody
    could justify from the arithmetic.
    """
    U, E, kpts = _star_consistent_payload()
    ref, stored = _write(tmp_path, "wedge.h5", U, E, kpts,
                         k_storage="ibz", star_tables=_TABLES)
    assert stored == K_STORAGE_IBZ

    with h5py.File(ref, "r") as f:
        assert f["U_mnk"].shape[0] == _NK_RED, "arrays were not reduced"
        # ...and the coordinate/index tables did NOT move
        assert f["kpoints_crys"].shape[0] == _NK_FULL
        assert np.array_equal(f["kpoints_crys"][()], kpts)
        assert np.array_equal(f["kirr_to_kfull"][()], _KIRR_TO_KFULL)

    got = read_qp_rotations_full_bz(ref)
    assert np.array_equal(got["U_mnk"], U)
    assert np.array_equal(got["E_qp_nk_hartree"], E)
    assert np.array_equal(got["E_qp_nk_rydberg"], E * 2.0)


def test_the_wedge_and_full_arms_read_back_identically(tmp_path):
    """The A/B a consumer actually cares about: same numbers, fewer bytes.

    "Fewer bytes" is asserted on the ARRAYS, not on the file, and the
    distinction is not pedantry — MEASURED here, the wedge file is BIGGER
    (11,880 against 10,240) at this fixture's 3x3x3 scale, because two
    index tables plus four sets of attrs cost more HDF5 metadata than a
    6-row-to-3-row reduction of a 3x3 complex block saves.  The saving is
    real and proportional at deck scale (Si production: ``U_mnk``
    {64,60,60} complex128 = 3.52 MB against {8,60,60} = 0.44 MB), and an
    assertion on the file size would be asserting the metadata overhead of
    whatever fixture happened to be here.
    """
    U, E, kpts = _star_consistent_payload()
    p_full, s_full = _write(tmp_path, "full.h5", U, E, kpts)
    p_ibz, s_ibz = _write(tmp_path, "ibz.h5", U, E, kpts,
                          k_storage="auto", star_tables=_TABLES)
    assert (s_full, s_ibz) == (K_STORAGE_FULL, K_STORAGE_IBZ)
    a = read_qp_rotations_full_bz(p_full)
    b = read_qp_rotations_full_bz(p_ibz)
    for name in QP_ROT_K_DATASETS:
        assert np.array_equal(a[name], b[name]), name

    def _array_bytes(path):
        with h5py.File(path, "r") as f:
            return sum(int(f[n].id.get_storage_size())
                       for n in QP_ROT_K_DATASETS)

    full_b, ibz_b = _array_bytes(p_full), _array_bytes(p_ibz)
    assert ibz_b * _NK_FULL == full_b * _NK_RED, (
        f"the k-indexed arrays should shrink by exactly "
        f"{_NK_FULL}/{_NK_RED}: {full_b} -> {ibz_b} bytes")


# ---------------------------------------------------------------------------
# 2.  The default that the four non-star fixtures exist for
# ---------------------------------------------------------------------------

def test_an_unstamped_file_is_read_as_full_bz_and_never_reinterpreted(tmp_path):
    """No ``k_storage`` attr means FULL, even when the rows would fit a wedge.

    This is the rule the committed pre-format fixtures depend on.  MEASURED
    2026-08-16 on the tree's own ``kin_ion.h5`` fixtures, whose rows do NOT
    satisfy the star relation: ``gnppm_debug`` 31.05 Ry, ``bispinor_debug``
    12.44 Ry, ``cohsex_debug`` 8.04 Ry, ``si_cohsex_debug`` 2.0e-3 Ry.
    Reading any of those as compressible would move physics by tens of Ry.

    The payload here is deliberately NOT star-consistent, so a reader that
    guessed from shape rather than from the attr would return different
    numbers and this cell would fail.
    """
    rng = np.random.default_rng(7)
    U = (rng.normal(size=(_NK_FULL, 3, 3))
         + 1j * rng.normal(size=(_NK_FULL, 3, 3)))
    E = rng.normal(size=(_NK_FULL, 3))
    kpts = rng.normal(size=(_NK_FULL, 3))
    path, stored = _write(tmp_path, "legacy.h5", U, E, kpts)
    assert stored == K_STORAGE_FULL

    with h5py.File(path, "r") as f:
        for name in QP_ROT_K_DATASETS + QP_ROT_FULL_BZ_DATASETS:
            assert K_STORAGE_ATTR not in f[name].attrs, (
                f"{name} carries a {K_STORAGE_ATTR} attr on the full arm; "
                f"the full-BZ file must be byte-for-byte what it always was")
        assert IRR_IDX_DATASET not in f and SYM_IDX_DATASET not in f

    assert qp_rotations_k_storage(path) == K_STORAGE_FULL
    got = read_qp_rotations_full_bz(path)
    assert np.array_equal(got["U_mnk"], U)
    assert np.array_equal(got["E_qp_nk_hartree"], E)


# ---------------------------------------------------------------------------
# 3.  The proof, and what happens when it fails
# ---------------------------------------------------------------------------

def test_auto_falls_back_to_full_when_the_rows_are_not_gathers(tmp_path):
    """An independent ``eigh`` per full-BZ k is NOT compressible, and ``auto``
    is the arm that notices instead of the arm that assumes.

    This is the one-shot driver's case (``gw_jax``'s ``vmap(eigh)`` over the
    full-BZ k axis).  An eigenvector is defined up to a phase and, inside a
    degenerate multiplet, up to a unitary mixing, so its off-wedge rows are
    a different gauge and dropping them loses information no gather rebuilds.
    """
    rng = np.random.default_rng(3)
    U = (rng.normal(size=(_NK_FULL, 3, 3))
         + 1j * rng.normal(size=(_NK_FULL, 3, 3)))
    E = rng.normal(size=(_NK_FULL, 3))
    kpts = rng.normal(size=(_NK_FULL, 3))
    said = []
    path, stored = _write(tmp_path, "oneshot.h5", U, E, kpts,
                          k_storage="auto", star_tables=_TABLES,
                          print_fn=said.append)
    assert stored == K_STORAGE_FULL
    assert any("FULL BZ" in s for s in said), said
    assert any("U_mnk max|" in s for s in said), (
        "the fallback must name the array that failed and by how much")
    with h5py.File(path, "r") as f:
        assert f["U_mnk"].shape[0] == _NK_FULL
    assert np.array_equal(read_qp_rotations_full_bz(path)["U_mnk"], U)


def test_ibz_refuses_rather_than_falling_back(tmp_path):
    """``ibz`` is for a run that wants to be told, not quietly accommodated."""
    rng = np.random.default_rng(3)
    U = (rng.normal(size=(_NK_FULL, 3, 3))
         + 1j * rng.normal(size=(_NK_FULL, 3, 3)))
    E = rng.normal(size=(_NK_FULL, 3))
    kpts = rng.normal(size=(_NK_FULL, 3))
    with pytest.raises(ValueError, match="not the unfold of the wedge rows"):
        _write(tmp_path, "refuse.h5", U, E, kpts,
               k_storage="ibz", star_tables=_TABLES)


def test_a_wedge_row_that_is_never_a_parent_is_refused_not_written(tmp_path):
    """The writer must not produce a file its own reader would refuse.

    ``kin_ion.read_star_map`` refuses when the stored k extent does not equal
    ``irr_idx_k.max() + 1``, because it cannot tell that from a truncated
    slab.  The round trip alone does NOT catch it: the round trip only reads
    rows the tables point at, so a stored k that is never an orbit parent
    reconstructs perfectly and still leaves an unreadable file.  This is the
    register's ``cohsex_debug`` shape — file wedge 4, star wedge 3, where
    stored row 1 is the time-reverse of row 2 and never a parent.
    """
    # FOUR stored k over THREE orbits, so wedge row 3 is never a parent —
    # and the rows are chosen so the ROUND TRIP still succeeds, which is what
    # makes this cell test the parent condition ALONE rather than two
    # blockers at once.
    from file_io.kin_ion import broadcast_ibz_to_full_bz as _bc
    tables = (_IRR, _SYM, _NSS)
    rng = np.random.default_rng(11)
    nb = 3
    u3 = (rng.normal(size=(_NK_RED, nb, nb))
          + 1j * rng.normal(size=(_NK_RED, nb, nb)))
    U = np.asarray(_bc(u3, *tables))
    E = np.asarray(_bc(rng.normal(size=(_NK_RED, nb)), *tables))
    kpts = rng.normal(size=(_NK_FULL, 3))
    rows = np.array([0, 2, 4, 5], dtype=np.int32)       # 4 rows, 3 parents

    def _w(k_storage, **kw):
        return write_qp_rotations_h5(
            str(tmp_path / f"np_{k_storage}.h5"), U_mnk=U, E_qp_nk=E,
            band_start=0, band_stop=nb, kpoints_crys=kpts,
            nkx=2, nky=1, nkz=1, kirr_to_kfull=rows,
            k_storage=k_storage, star_tables=tables, **kw)

    with pytest.raises(ValueError, match="never an orbit parent"):
        _w("ibz")
    said = []
    assert _w("auto", print_fn=said.append) == K_STORAGE_FULL
    assert any("never an orbit parent" in s for s in said), said


def test_the_wedge_arm_refuses_without_the_tables_it_would_file(tmp_path):
    """A tensor whose reconstruction tables live elsewhere silently decays."""
    U, E, kpts = _star_consistent_payload()
    with pytest.raises(ValueError, match="star_tables"):
        _write(tmp_path, "notables.h5", U, E, kpts, k_storage="auto")


def test_an_unrecognised_k_storage_is_not_read_as_the_default(tmp_path):
    U, E, kpts = _star_consistent_payload()
    with pytest.raises(ValueError, match="none of"):
        _write(tmp_path, "bogus.h5", U, E, kpts, k_storage="wedge")
    assert "auto" in QP_ROTATIONS_K_STORAGE


# ---------------------------------------------------------------------------
# 4.  The stamp contract is kin_ion's, not a second copy of it
# ---------------------------------------------------------------------------

def test_the_stamp_is_the_kin_ion_contract_and_the_tables_travel_with_it(tmp_path):
    U, E, kpts = _star_consistent_payload()
    path, _ = _write(tmp_path, "stamped.h5", U, E, kpts,
                     k_storage="ibz", star_tables=_TABLES)
    with h5py.File(path, "r") as f:
        assert np.array_equal(f[IRR_IDX_DATASET][()], _IRR)
        assert np.array_equal(f[SYM_IDX_DATASET][()], _SYM)
        for name in QP_ROT_K_DATASETS:
            a = f[name].attrs
            assert a[K_STORAGE_ATTR] == K_STORAGE_IBZ
            assert int(a[K_STORAGE_VERSION_ATTR]) == K_STORAGE_VERSION
            assert int(a[N_SYM_SPATIAL_ATTR]) == _NSS


def test_the_coordinate_and_index_tables_do_not_move(tmp_path):
    """``kpoints_crys`` / ``kirr_to_kfull`` stay full-BZ, and that is the design.

    The unfold is a GATHER — every member of a star gets its parent's row —
    which is right for an operator that commutes with the symmetry and wrong
    for the k-VECTORS, because k is the one quantity in this file that the
    operation changes.  MEASURED on a real ``si_cohsex_debug`` run:
    reducing ``kpoints_crys`` and gathering it back gives
    ``max|Δ| = 7.500000e-01``, and 3/4 is not a reciprocal-lattice vector, so
    no modulo-G reading rescues it.

    They cost 1,536 and 32 bytes there against 3.5 MB of ``U_mnk``, so
    nothing is lost by leaving them alone — and leaving them alone is what
    lets ``rotate_wfn_to_qp`` and ``eqp_bgw`` unfold the arrays and then
    index by full-BZ k exactly as before.
    """
    U, E, kpts = _star_consistent_payload()
    path, stored = _write(tmp_path, "idx.h5", U, E, kpts,
                          k_storage="ibz", star_tables=_TABLES)
    assert stored == K_STORAGE_IBZ
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["kirr_to_kfull"][()], _KIRR_TO_KFULL)
        assert np.array_equal(f["kpoints_crys"][()], kpts)
        for name in QP_ROT_FULL_BZ_DATASETS:
            assert K_STORAGE_ATTR not in f[name].attrs, (
                f"{name} must NOT be stamped — it did not move")
    # the consumers' composition: unfold, then index by full-BZ k
    got = read_qp_rotations_full_bz(path)
    assert np.array_equal(got["U_mnk"][_KIRR_TO_KFULL], U[_KIRR_TO_KFULL])

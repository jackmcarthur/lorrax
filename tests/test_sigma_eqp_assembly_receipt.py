"""Raw Sigma operators versus a completed, conditioned EQP assembly."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from file_io.kin_ion import (
    IRR_IDX_DATASET,
    K_STORAGE_ATTR,
    K_STORAGE_IBZ,
    K_STORAGE_VERSION,
    K_STORAGE_VERSION_ATTR,
    N_SYM_SPATIAL_ATTR,
    SYM_IDX_DATASET,
)
from file_io.sigma_output import (
    EQP_ASSEMBLY_C_OMEGA_CANDIDATE_DATASET,
    EQP_ASSEMBLY_EXPECTED_DATASET,
    EQP_ASSEMBLY_SCHEMA_VERSION,
    SIGMA_OPERATOR_STATE_ATTR,
    SIGMA_OPERATOR_STATE_RAW,
    SIGMA_OPERATOR_STATE_VERSION,
    SIGMA_OPERATOR_STATE_VERSION_ATTR,
    append_eqp_assembly_receipt_h5,
    read_eqp_assembly_receipt,
    read_sigma_eqp_diagonal_window,
)
from gw.eqp_bgw import assemble_eqp, resolve_hartree_diag_ev


OMEGA = np.array([-1.0, 0.0, 1.0])


def _sigma_file(path):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("omega_ev", data=OMEGA)
        h5.create_dataset(
            "sigma_c_kij_ev", data=np.zeros((OMEGA.size, 1, 2, 2)))
    return path


def _assembly(h, x, c=None, *, curve=None, source="stored"):
    h = np.asarray(h, dtype=np.float64)
    c = np.zeros_like(h, dtype=np.complex128) if c is None else np.asarray(c)
    curve = (np.broadcast_to(c, (OMEGA.size,) + h.shape).copy()
             if curve is None else np.asarray(curve, dtype=np.complex128))
    return assemble_eqp(
        kpoints_irr_frac=np.zeros((h.shape[0], 3)),
        band_offset=0,
        e_dft_ev=np.zeros_like(h),
        kin_ion_diag_ev=np.zeros_like(h),
        hartree_diag_ev=h,
        sigma_x_diag_ev=np.asarray(x, dtype=np.float64),
        sigma_c_omega_diag_ev=curve,
        omega_rel_ev=OMEGA,
        e_dft_rel_ev=np.zeros_like(h),
        hartree_source=source,
        kin_ion_has_hartree=(source == "folded"),
        hartree_already_resolved=True,
        mean_field_gate=False,
        print_fn=lambda *_: None,
    )


def _append(path, h, x, c=None, *, rows=None, source="stored",
            policy="bgw_average"):
    """Append with the FILE-wedge rows the call site resolved.

    ``rows`` defaults to ``arange(nk)`` because the synthetic files here
    store the full BZ, where the file wedge IS every row.  The cells that
    exercise a real IBZ-stored cube pass their own.
    """
    assembly = _assembly(h, x, c, source=source)
    nk = np.asarray(h).shape[0]
    append_eqp_assembly_receipt_h5(
        path,
        assembly=assembly,
        file_wedge_full_bz_rows=(
            np.arange(nk) if rows is None else np.asarray(rows)),
        degeneracy_policy=policy,
        degeneracy_tol_ry=1.0e-6,
    )
    return read_eqp_assembly_receipt(path)


def _stamp_raw(h5):
    for name in ("sigma_c_kij_ev",):
        h5[name].attrs[SIGMA_OPERATOR_STATE_ATTR] = SIGMA_OPERATOR_STATE_RAW
        h5[name].attrs[SIGMA_OPERATOR_STATE_VERSION_ATTR] = (
            SIGMA_OPERATOR_STATE_VERSION)


def test_legacy_raw_scalar_and_two_resolved_receipts_are_distinct(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    assert read_eqp_assembly_receipt(path) is None

    exact_scalar = np.array([[10.0, 12.0]])
    legacy_raw = np.array([[9.0, 13.0]])
    got, rule = resolve_hartree_diag_ev(
        hartree_diag_ev=legacy_raw,
        hartree_source="stored",
        exact_hartree_diag_ev=exact_scalar,
        print_fn=lambda *_: None,
    )
    assert rule == "substituted"
    assert np.array_equal(got, exact_scalar)

    resolved_scalar = np.array([[11.0, 11.0]])
    receipt = _append(path, resolved_scalar, [[-3.0, -3.0]])
    got, rule = resolve_hartree_diag_ev(
        hartree_diag_ev=receipt["hartree_diag_ev"],
        hartree_source="stored",
        hartree_already_resolved=True,
        print_fn=lambda *_: None,
    )
    assert rule == "as-given"
    assert np.array_equal(got, resolved_scalar)

    # The completed assembly preserves a separately resolved direct residual
    # without a magnitude test or sector guess.
    resolved_plus_direct = resolved_scalar + np.array([[0.25, -0.25]])
    receipt = _append(path, resolved_plus_direct, [[-3.0, -3.0]])
    assert np.array_equal(receipt["hartree_diag_ev"], resolved_plus_direct)
    assert receipt["hartree_exchange_basis"] == "dft_band"
    assert receipt["correlation_basis"] == "dft_band"


def test_legacy_diagonal_hyperslabs_equal_small_full_cube_reference(tmp_path):
    path = tmp_path / "legacy_sigma_mnk.h5"
    omega = np.array([-1.0, 0.0, 1.0])
    x = np.arange(2 * 4 * 4).reshape(2, 4, 4).astype(np.complex128)
    h = (100.0 + x).astype(np.complex128)
    c = np.arange(3 * 2 * 4 * 4).reshape(3, 2, 4, 4).astype(np.complex128)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("omega_ev", data=omega)
        h5.create_dataset("sigma_sx_kij_ev", data=x)
        h5.create_dataset("hartree_kij_ev", data=h)
        h5.create_dataset("sigma_c_kij_ev", data=c)

    got = read_sigma_eqp_diagonal_window(
        path, full_bz_rows=np.array([1, 0]), band_start=1, band_stop=3)
    rows = np.array([1, 0])
    expect_x = np.diagonal(x[rows, 1:3, 1:3], axis1=1, axis2=2)
    expect_h = np.diagonal(h[rows, 1:3, 1:3], axis1=1, axis2=2)
    expect_c = np.diagonal(
        c[:, rows, 1:3, 1:3], axis1=2, axis2=3)
    np.testing.assert_array_equal(got["sigma_x_diag_ev"], expect_x)
    np.testing.assert_array_equal(got["hartree_diag_ev"], expect_h)
    np.testing.assert_array_equal(got["sigma_c_omega_diag_ev"], expect_c)
    np.testing.assert_array_equal(got["omega_rel_ev"], omega)


def test_receipt_replays_the_same_conditioned_curve_and_assembly(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    h_raw = np.array([[10.0, 12.0]])
    x_raw = np.array([[-4.0, -2.0]])
    h_live = np.array([[11.0, 11.0]])
    x_live = np.array([[-3.0, -3.0]])
    c_raw = np.array([[1.0, 3.0]])
    raw_curve = c_raw[None, :, :] + OMEGA[:, None, None] * [[0.2, 0.4]]
    conditioned_curve = np.full((3, 1, 2), 2.0) + (
        OMEGA[:, None, None] * 0.3)
    live = _assembly(h_live, x_live, curve=conditioned_curve)
    append_eqp_assembly_receipt_h5(
        path,
        assembly=live,
        file_wedge_full_bz_rows=np.arange(1),
        degeneracy_policy="bgw_average",
        degeneracy_tol_ry=1.0e-6,
    )
    receipt = read_eqp_assembly_receipt(path)

    common = dict(
        kpoints_irr_frac=np.zeros((1, 3)),
        band_offset=0,
        e_dft_ev=np.zeros((1, 2)),
        kin_ion_diag_ev=np.zeros((1, 2)),
        omega_rel_ev=OMEGA,
        e_dft_rel_ev=np.zeros((1, 2)),
        mean_field_gate=False,
        print_fn=lambda *_: None,
    )
    legacy = assemble_eqp(
        **common,
        hartree_diag_ev=h_raw,
        sigma_x_diag_ev=x_raw,
        sigma_c_omega_diag_ev=raw_curve,
        hartree_source="stored",
        exact_hartree_diag_ev=h_raw,
    )
    rebuilt = assemble_eqp(
        **common,
        hartree_diag_ev=receipt["hartree_diag_ev"],
        sigma_x_diag_ev=receipt["sigma_x_diag_ev"],
        sigma_c_omega_diag_ev=receipt["sigma_c_omega_diag_ev"],
        hartree_source="stored",
        hartree_already_resolved=True,
    )
    assert np.array_equal(legacy.eqp0_ev, [[7.0, 13.0]])
    assert np.array_equal(rebuilt.eqp0_ev, [[10.0, 10.0]])
    assert np.array_equal(rebuilt.eqp0_ev, live.eqp0_ev)
    assert np.array_equal(rebuilt.eqp1_ev, live.eqp1_ev)
    assert not np.array_equal(legacy.eqp1_ev, rebuilt.eqp1_ev)
    assert np.array_equal(
        rebuilt.sigma_c_at_dft_diag_ev,
        receipt["sigma_c_at_dft_diag_ev"])


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ("expected_missing", "receipt.*missing"),
        ("raw_without_expectation", "new raw-operator artifact.*missing"),
        ("bad_raw_stamp", "new EQP schema.*raw operator state"),
        ("curve_candidate", "interrupted candidate write"),
        ("held_schema_1", "expectation .*1.*expected 3"),
        ("held_schema_2", "expectation .*2.*expected 3"),
    ],
)
def test_new_schema_states_fail_closed(tmp_path, state, match):
    """Every partial, unknown, or interrupted v3 state refuses as one gate."""
    path = _sigma_file(tmp_path / f"{state}.h5")
    if state in {"expected_missing", "raw_without_expectation"}:
        with h5py.File(path, "a") as h5:
            _stamp_raw(h5)
            if state == "expected_missing":
                h5.create_dataset(
                    EQP_ASSEMBLY_EXPECTED_DATASET,
                    data=np.asarray(EQP_ASSEMBLY_SCHEMA_VERSION, np.int32))
    else:
        _append(path, [[11.0, 11.0]], [[-3.0, -3.0]])
        with h5py.File(path, "a") as h5:
            if state == "bad_raw_stamp":
                del h5["sigma_c_kij_ev"].attrs[SIGMA_OPERATOR_STATE_ATTR]
            elif state == "curve_candidate":
                h5.create_dataset(
                    EQP_ASSEMBLY_C_OMEGA_CANDIDATE_DATASET,
                    data=np.zeros((3, 1, 2)))
            else:
                h5[EQP_ASSEMBLY_EXPECTED_DATASET][()] = int(state[-1])
    with pytest.raises(ValueError, match=match):
        read_eqp_assembly_receipt(path)


def test_appender_refuses_new_raw_cube_missing_creation_marker(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    with h5py.File(path, "a") as h5:
        _stamp_raw(h5)
    with pytest.raises(ValueError, match="raw-operator stamps.*marker"):
        _append(path, [[11.0, 11.0]], [[-3.0, -3.0]])


def test_receipt_checks_rows_and_policy_before_writing(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    assembly = _assembly([[11.0, 11.0]], [[-3.0, -3.0]])
    wrong_nk = _assembly(
        [[11.0, 11.0], [12.0, 12.0]],
        [[-3.0, -3.0], [-4.0, -4.0]],
    )
    # One row index per assembly k row, or nothing is stamped.
    with pytest.raises(ValueError, match="want one full-BZ row index"):
        append_eqp_assembly_receipt_h5(
            path,
            assembly=wrong_nk,
            file_wedge_full_bz_rows=np.arange(1),
            degeneracy_policy="bgw_average",
            degeneracy_tol_ry=1.0e-6,
        )
    # Counted right, but naming a k this 1-row full-BZ cube does not have.
    with pytest.raises(ValueError, match="out of range for the 1-k mesh"):
        append_eqp_assembly_receipt_h5(
            path,
            assembly=wrong_nk,
            file_wedge_full_bz_rows=np.arange(2),
            degeneracy_policy="bgw_average",
            degeneracy_tol_ry=1.0e-6,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        append_eqp_assembly_receipt_h5(
            path,
            assembly=wrong_nk,
            file_wedge_full_bz_rows=np.array([0, 0]),
            degeneracy_policy="bgw_average",
            degeneracy_tol_ry=1.0e-6,
        )
    with pytest.raises(ValueError, match="unknown EQP degeneracy policy"):
        append_eqp_assembly_receipt_h5(
            path,
            assembly=assembly,
            file_wedge_full_bz_rows=np.arange(1),
            degeneracy_policy="guess",
            degeneracy_tol_ry=1.0e-6,
        )


def test_receipt_rows_are_the_call_site_s_own_file_wedge(tmp_path):
    """The stamped rows are the ones the caller passed.

    Two stars over a 5-k mesh, and a file wedge of rows [3, 2] rather than
    the star-map answer [0, 2]: row 3 is the other member of star 4.  A
    writer deriving the rows would mislabel which k the payload belongs to.
    """
    path = tmp_path / "ibz_sigma_mnk.h5"
    irr = np.array([4, 4, 9, 4, 9], dtype=np.int32)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("omega_ev", data=OMEGA)
        ds = h5.create_dataset(
            "sigma_c_kij_ev", data=np.zeros((OMEGA.size, 2, 2, 2)))
        ds.attrs[K_STORAGE_ATTR] = K_STORAGE_IBZ
        ds.attrs[K_STORAGE_VERSION_ATTR] = K_STORAGE_VERSION
        ds.attrs[N_SYM_SPATIAL_ATTR] = 1
        h5.create_dataset(IRR_IDX_DATASET, data=irr)
        h5.create_dataset(SYM_IDX_DATASET, data=np.zeros_like(irr))

    receipt = _append(
        path,
        [[11.0, 11.0], [12.0, 12.0]],
        [[-3.0, -3.0], [-4.0, -4.0]],
        rows=np.array([3, 2]),
    )
    np.testing.assert_array_equal(
        receipt["file_wedge_full_bz_rows"], np.array([3, 2]))


# The real deck's k topology, read off tests/regression/gnppm_debug (MoS2
# 3x3x1): nrk is 9 on a [3,3,1] grid, so the WFN stores the whole zone and
# the file wedge is all nine rows, while the run's sigma_mnk.h5 keeps five
# cube rows under these star tables.  Every other fixture in this file has
# nk_red == n_orbits and so cannot see a 9-vs-5 deck.
GNPPM_IRR_IDX_K = np.array([0, 1, 1, 2, 3, 4, 2, 4, 3], dtype=np.int32)
GNPPM_SYM_IDX_K = np.array([0, 2, 0, 2, 2, 2, 0, 0, 0], dtype=np.int32)
GNPPM_N_SYM_SPATIAL = 2
GNPPM_N_STARS = 5
GNPPM_NK_RED = 9


def _gnppm_topology_cube(path, nb=2):
    """A tiny cube carrying the REAL gnppm_debug star tables."""
    with h5py.File(path, "w") as h5:
        h5.create_dataset("omega_ev", data=OMEGA)
        ds = h5.create_dataset(
            "sigma_c_kij_ev",
            data=np.zeros((OMEGA.size, GNPPM_N_STARS, nb, nb)))
        ds.attrs[K_STORAGE_ATTR] = K_STORAGE_IBZ
        ds.attrs[K_STORAGE_VERSION_ATTR] = K_STORAGE_VERSION
        ds.attrs[N_SYM_SPATIAL_ATTR] = GNPPM_N_SYM_SPATIAL
        h5.create_dataset(IRR_IDX_DATASET, data=GNPPM_IRR_IDX_K)
        h5.create_dataset(SYM_IDX_DATASET, data=GNPPM_SYM_IDX_K)
    return path


def test_receipt_accepts_the_real_gnppm_file_wedge_over_a_star_wedge_cube(
        tmp_path):
    """9 assembly rows against a 5-row cube is correct on this deck.

    The case that was red: until 2026-08-27 the writer derived the rows
    from the cube's star map and required equal counts, so this deck
    refused its own assembly ("completed assembly has 9 k rows, raw
    artifact resolves to 5 canonical rows"; a CrI3 bispinor run hit the
    same message on 2026-08-26).  Nothing was corrupt — the assembly is on
    the file wedge and the cube on the star wedge.
    """
    path = _gnppm_topology_cube(tmp_path / "sigma_mnk.h5")
    h = np.arange(GNPPM_NK_RED * 2, dtype=np.float64).reshape(GNPPM_NK_RED, 2)
    receipt = _append(path, h, -h, rows=np.arange(GNPPM_NK_RED))
    assert receipt["hartree_diag_ev"].shape == (GNPPM_NK_RED, 2)
    np.testing.assert_array_equal(
        receipt["file_wedge_full_bz_rows"], np.arange(GNPPM_NK_RED))
    # The payload really is the nine independent rows, not five unfolded.
    np.testing.assert_array_equal(receipt["hartree_diag_ev"], h)
    with h5py.File(path, "r") as h5:
        assert h5["sigma_c_kij_ev"].shape[1] == GNPPM_N_STARS


def test_receipt_refuses_a_file_wedge_that_misses_one_of_the_cube_s_stars(
        tmp_path):
    """Five rows, the count the old gate demanded, covering three stars.

    Rows [0,1,2,3,6] reach stars {0,1,2} and leave stars 3 and 4 with no
    assembly row.  A count check cannot see this.
    """
    path = _gnppm_topology_cube(tmp_path / "sigma_mnk.h5")
    bad_rows = np.array([0, 1, 2, 3, 6])
    assert np.unique(GNPPM_IRR_IDX_K[bad_rows]).size < GNPPM_N_STARS
    h = np.zeros((bad_rows.size, 2))
    with pytest.raises(ValueError, match="does not cover the raw cube"):
        _append(path, h, h, rows=bad_rows)


def test_receipt_lands_on_the_real_gnppm_run(gnppm_session):
    """The real input: the sigma_mnk.h5 a driver run wrote.

    Everything else in this file builds its own HDF5.  This reads what
    ``write_results`` produced on the tracked deck — the path that used to
    die inside the appender before eqp0.dat existed.  The file wedge being
    longer than the star wedge is the property under test, so it is
    asserted: on a deck where they coincide this cell proves nothing.

    The receipt is read through h5py, not through
    ``read_eqp_assembly_receipt``, because that reader still refuses this
    deck — sigma_eval_rel_ev is on the star wedge.  Both halves are pinned
    here, so closing that open ruling turns this cell red.
    """
    import os

    from file_io.sigma_output import (
        EQP_ASSEMBLY_DATASET,
        EQP_ASSEMBLY_FILE_ROWS_ATTR,
        read_eqp_assembly_receipt,
    )

    path = os.path.join(str(gnppm_session.run_dir), "sigma_mnk.h5")
    assert os.path.isfile(path), f"the gnppm session wrote no {path}"
    with h5py.File(path, "r") as h5:
        nk_stored = int(h5["sigma_c_kij_ev"].shape[1])
        nk_full = int(h5[IRR_IDX_DATASET].shape[0])
        assert EQP_ASSEMBLY_DATASET in h5, "the run left no EQP receipt"
        ds = h5[EQP_ASSEMBLY_DATASET]
        nk_receipt = int(ds.shape[1])
        rows = np.asarray(ds.attrs[EQP_ASSEMBLY_FILE_ROWS_ATTR])
    assert rows.shape == (nk_receipt,)
    assert nk_stored < nk_receipt <= nk_full, (
        f"gnppm_debug should have a file wedge longer than its star wedge; "
        f"got {nk_receipt} receipt rows against {nk_stored} stored and "
        f"{nk_full} full-BZ k.  If this deck ever becomes nk_red == n_orbits, "
        f"this cell no longer discriminates and must be re-aimed.")
    # Every stored star is reached by one of the receipt's rows.
    with h5py.File(path, "r") as h5:
        compact = np.asarray(h5[IRR_IDX_DATASET][()])
    assert np.unique(compact[rows]).size == nk_stored
    # Open ruling: the evaluation stamp is on the other wedge.
    with pytest.raises(ValueError, match="STAR wedge and the receipt"):
        read_eqp_assembly_receipt(path)

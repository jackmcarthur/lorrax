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


def _append(path, h, x, c=None, *, source="stored", policy="bgw_average"):
    assembly = _assembly(h, x, c, source=source)
    append_eqp_assembly_receipt_h5(
        path,
        assembly=assembly,
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


def test_receipt_derives_rows_and_validates_policy_before_writing(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    assembly = _assembly([[11.0, 11.0]], [[-3.0, -3.0]])
    wrong_nk = _assembly(
        [[11.0, 11.0], [12.0, 12.0]],
        [[-3.0, -3.0], [-4.0, -4.0]],
    )
    with pytest.raises(ValueError, match="canonical star map"):
        append_eqp_assembly_receipt_h5(
            path,
            assembly=wrong_nk,
            degeneracy_policy="bgw_average",
            degeneracy_tol_ry=1.0e-6,
        )
    with pytest.raises(ValueError, match="unknown EQP degeneracy policy"):
        append_eqp_assembly_receipt_h5(
            path,
            assembly=assembly,
            degeneracy_policy="guess",
            degeneracy_tol_ry=1.0e-6,
        )


def test_receipt_rows_are_raw_star_map_first_occurrences(tmp_path):
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
    )
    np.testing.assert_array_equal(
        receipt["file_wedge_full_bz_rows"], np.array([0, 2]))

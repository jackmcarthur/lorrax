"""Focused gates for raw Sigma operators versus live EQP diagonals.

The full H/X cubes are operator artifacts and stay raw.  The receipt is the
small, authoritative H/X/C(E_DFT) post-conditioning input to the sole EQP
assembler.  These cells discriminate the three states that the run56 defect
collapsed: legacy raw scalar, resolved scalar, and resolved scalar plus an
independent direct residual.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from file_io.sigma_output import (
    EQP_ASSEMBLY_EXPECTED_ATTR,
    EQP_ASSEMBLY_SCHEMA_VERSION,
    SIGMA_OPERATOR_STATE_ATTR,
    SIGMA_OPERATOR_STATE_RAW,
    SIGMA_OPERATOR_STATE_VERSION,
    SIGMA_OPERATOR_STATE_VERSION_ATTR,
    append_eqp_assembly_receipt_h5,
    read_eqp_assembly_receipt,
)
from gw.eqp_bgw import assemble_eqp, resolve_hartree_diag_ev


def _sigma_file(path):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("sigma_c_kij_ev", data=np.zeros((1, 1, 2, 2)))
    return path


def _append(path, h, x, c=None, *, source="stored", c_basis="dft_band"):
    h = np.asarray(h, dtype=np.float64)
    append_eqp_assembly_receipt_h5(
        path,
        hartree_diag_ev=h,
        sigma_x_diag_ev=np.asarray(x, dtype=np.float64),
        sigma_c_at_dft_diag_ev=(
            np.zeros_like(h, dtype=np.complex128) if c is None else c),
        file_wedge_full_bz_rows=np.arange(h.shape[0], dtype=np.int64),
        band_start=0,
        band_stop=2,
        degeneracy_policy="bgw_average",
        degeneracy_tol_ry=1.0e-6,
        correlation_basis=c_basis,
        hartree_source=source,
        kin_ion_has_hartree=(source == "folded"),
    )
    return read_eqp_assembly_receipt(path)


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

    # The receipt is the actual assembler operand, so a separately resolved
    # direct residual is preserved without a magnitude test or sector guess.
    resolved_scalar_plus_direct = resolved_scalar + np.array([[0.25, -0.25]])
    receipt = _append(
        path,
        resolved_scalar_plus_direct,
        [[-3.0, -3.0]],
        c_basis="qp_band",
    )
    got, rule = resolve_hartree_diag_ev(
        hartree_diag_ev=receipt["hartree_diag_ev"],
        hartree_source="stored",
        hartree_already_resolved=True,
        print_fn=lambda *_: None,
    )
    assert rule == "as-given"
    assert np.array_equal(got, resolved_scalar_plus_direct)
    assert receipt["hartree_exchange_basis"] == "dft_band"
    assert receipt["correlation_basis"] == "qp_band"


def test_receipt_reproduces_post_degeneracy_eqp_without_reprojecting(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    h_raw = np.array([[10.0, 12.0]])
    x_raw = np.array([[-4.0, -2.0]])
    h_live = np.array([[11.0, 11.0]])
    x_live = np.array([[-3.0, -3.0]])
    c_raw = np.array([[1.0, 3.0]])
    c_live = np.array([[2.0, 2.0]])
    receipt = _append(path, h_live, x_live, c_live)

    common = dict(
        kpoints_irr_frac=np.zeros((1, 3)),
        band_offset=0,
        e_dft_ev=np.zeros((1, 2)),
        kin_ion_diag_ev=np.zeros((1, 2)),
        sigma_c_omega_diag_ev=np.broadcast_to(c_raw, (3, 1, 2)).copy(),
        omega_rel_ev=np.array([-1.0, 0.0, 1.0]),
        e_dft_rel_ev=np.zeros((1, 2)),
        mean_field_gate=False,
        print_fn=lambda *_: None,
    )
    legacy = assemble_eqp(
        **common,
        hartree_diag_ev=h_raw,
        sigma_x_diag_ev=x_raw,
        hartree_source="stored",
        exact_hartree_diag_ev=h_raw,
    )
    rebuilt = assemble_eqp(
        **common,
        hartree_diag_ev=receipt["hartree_diag_ev"],
        sigma_x_diag_ev=receipt["sigma_x_diag_ev"],
        sigma_c_at_dft_diag_ev=receipt["sigma_c_at_dft_diag_ev"],
        hartree_source="stored",
        hartree_already_resolved=True,
    )
    assert np.array_equal(legacy.eqp0_ev, [[7.0, 13.0]])
    assert np.array_equal(rebuilt.eqp0_ev, [[10.0, 10.0]])
    assert np.array_equal(rebuilt.eqp1_ev, rebuilt.eqp0_ev)


def test_partial_new_receipt_refuses_instead_of_falling_back_to_legacy(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    with h5py.File(path, "a") as h5:
        h5.attrs[EQP_ASSEMBLY_EXPECTED_ATTR] = EQP_ASSEMBLY_SCHEMA_VERSION
    with pytest.raises(ValueError, match="receipt.*missing"):
        read_eqp_assembly_receipt(path)


def test_unreleased_v1_receipt_cannot_collide_with_v2_semantics(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    _append(path, [[11.0, 11.0]], [[-3.0, -3.0]])
    with h5py.File(path, "a") as h5:
        h5.attrs[EQP_ASSEMBLY_EXPECTED_ATTR] = 1
    with pytest.raises(ValueError, match="expectation .*1.*expected 2"):
        read_eqp_assembly_receipt(path)


def test_new_raw_cube_without_closed_receipt_refuses_as_interrupted(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    with h5py.File(path, "a") as h5:
        ds = h5["sigma_c_kij_ev"]
        ds.attrs[SIGMA_OPERATOR_STATE_ATTR] = SIGMA_OPERATOR_STATE_RAW
        ds.attrs[SIGMA_OPERATOR_STATE_VERSION_ATTR] = SIGMA_OPERATOR_STATE_VERSION
    with pytest.raises(ValueError, match="new raw-operator artifact.*missing"):
        read_eqp_assembly_receipt(path)


def test_appender_refuses_new_raw_cube_missing_creation_marker(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    with h5py.File(path, "a") as h5:
        ds = h5["sigma_c_kij_ev"]
        ds.attrs[SIGMA_OPERATOR_STATE_ATTR] = SIGMA_OPERATOR_STATE_RAW
        ds.attrs[SIGMA_OPERATOR_STATE_VERSION_ATTR] = SIGMA_OPERATOR_STATE_VERSION
    with pytest.raises(ValueError, match="raw-operator stamps.*creation marker"):
        _append(path, [[11.0, 11.0]], [[-3.0, -3.0]])


def test_receipt_validates_band_window_and_policy_before_writing(tmp_path):
    path = _sigma_file(tmp_path / "sigma_mnk.h5")
    with pytest.raises(ValueError, match="band window"):
        append_eqp_assembly_receipt_h5(
            path,
            hartree_diag_ev=np.zeros((1, 2)),
            sigma_x_diag_ev=np.zeros((1, 2)),
            sigma_c_at_dft_diag_ev=np.zeros((1, 2)),
            file_wedge_full_bz_rows=np.array([0]),
            band_start=0,
            band_stop=3,
            degeneracy_policy="bgw_average",
            degeneracy_tol_ry=1.0e-6,
            correlation_basis="dft_band",
            hartree_source="stored",
            kin_ion_has_hartree=False,
        )
    with pytest.raises(ValueError, match="unknown EQP degeneracy policy"):
        append_eqp_assembly_receipt_h5(
            path,
            hartree_diag_ev=np.zeros((1, 2)),
            sigma_x_diag_ev=np.zeros((1, 2)),
            sigma_c_at_dft_diag_ev=np.zeros((1, 2)),
            file_wedge_full_bz_rows=np.array([0]),
            band_start=0,
            band_stop=2,
            degeneracy_policy="guess",
            degeneracy_tol_ry=1.0e-6,
            correlation_basis="dft_band",
            hartree_source="stored",
            kin_ion_has_hartree=False,
        )

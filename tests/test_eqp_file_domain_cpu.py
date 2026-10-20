"""File EQP rows sample, but never reconstruct, a full independent Sigma."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


def _symmetry(rows):
    nk = 25
    return SimpleNamespace(
        nk_tot=nk, nk_red=len(rows), kirr_fullids=np.asarray(rows, np.int32),
        unfolded_kpts=np.indices((5, 5, 1)).reshape(3, -1).T / [5, 5, 1],
        irr_idx_k=np.arange(nk, dtype=np.int32), sym_idx_k=np.zeros(nk, np.int32),
        sym_mats_k=np.asarray([np.eye(3, dtype=int)] * 2))


def _source(tmp_path, *, stamp="identity", omega_grid=False):
    from common.units import RYD_TO_EV
    from file_io import sigma_output as so
    from file_io.kin_ion import IRR_IDX_DATASET, SYM_IDX_DATASET
    from gw.gw_output import GWResults

    nk, nb = 25, 2
    energies = np.broadcast_to([-.5, .3], (nk, nb)).copy()
    energies += np.arange(nk)[:, None] * .001
    zeros = np.zeros((nk, nb, nb), complex)
    kinetic = np.eye(nb)[None] * energies[:, None]
    diagonal = (.02 + np.arange(nk)[:, None] * .0002) * np.ones((nk, nb))
    diagonal = diagonal + .003j
    pair = np.broadcast_to(np.eye(nb)[None] * diagonal[:, None], (2, nk, nb, nb)).copy()
    covered = np.ones((nk, nb)); covered[::3] = 0
    operand = np.stack((diagonal * RYD_TO_EV, diagonal * RYD_TO_EV,
                        np.full((nk, nb), .1), covered))
    path = tmp_path / "sigma.h5"
    if stamp == "identity":
        star = (np.arange(nk), np.zeros(nk, int), 1)
    elif stamp == "identity_actions":
        star = (np.arange(nk), np.ones(nk, int), 1)
    elif stamp == "reducing":
        star = (np.arange(nk) % 5, np.zeros(nk, int), 1)
    else:
        star = None
    curve = np.broadcast_to(diagonal[None] * RYD_TO_EV, (3, nk, nb)).copy()
    # Small format fixture, not an emulation of the native parallel writer.
    # Its k storage still comes from the real format owner's extraction.
    payload, attrs, compact, syms = so.extract_and_stamp_k_irr(dict(
        sigma_c_kij_ev=pair * RYD_TO_EV, sigma_total_kij_ev=pair * RYD_TO_EV,
        sigma_eval_rel_ev=energies * RYD_TO_EV), star, nk_full=nk,
        on_shell_pairs=None if not omega_grid else ())
    with h5py.File(path, "w") as f:
        f[so.EQP_ASSEMBLY_EXPECTED_DATASET] = so.EQP_ASSEMBLY_SCHEMA_VERSION
        if compact is not None:
            f[IRR_IDX_DATASET] = compact
            f[SYM_IDX_DATASET] = syms
        for name, value in payload.items():
            ds = f.create_dataset(name, data=value)
            ds.attrs.update(attrs(name) or {})
            if name != so.SIGMA_EVAL_DATASET:
                ds.attrs[so.SIGMA_OPERATOR_STATE_ATTR] = so.SIGMA_OPERATOR_STATE_RAW
                ds.attrs[so.SIGMA_OPERATOR_STATE_VERSION_ATTR] = so.SIGMA_OPERATOR_STATE_VERSION
        axis = so.OMEGA_DATASET if omega_grid else so.ON_SHELL_ENERGIES_DATASET.format(table="pair")
        f[axis] = np.array([-20., 0., 20.]) if omega_grid else energies[None] * RYD_TO_EV
        f[axis].attrs[so.OMEGA_REFERENCE_ATTR] = 0.
        f[axis].attrs[so.OMEGA_REFERENCE_PROVENANCE_ATTR] = "test-current"
        ev = f[so.SIGMA_EVAL_DATASET]
        ev.attrs[so.SIGMA_EVAL_PROVENANCE_ATTR] = so.SIGMA_EVAL_AT_E_DFT
        ev.attrs[so.OMEGA_COVERAGE_N_ATTR] = np.count_nonzero(covered == 0)
        ev.attrs[so.OMEGA_COVERAGE_FRAC_ATTR] = np.mean(covered == 0)
        ev.attrs[so.OMEGA_COVERAGE_POLICY_ATTR] = "static"
    results = GWResults(sig_sx=zeros, sig_coh=zeros, sig_h=zeros, sig_x=zeros,
        E_qp_ry=energies + .01, U_qp=None, E_dft_ry=energies,
        kin_ion_ry=kinetic, band_start=0, band_stop=nb, use_ppm=True,
        sigma_c_diag_at_dft_ry=diagonal, sigma_omega_h5_path=str(path),
        sigma_c_on_shell_diag=None if omega_grid else operand,
        sigma_c_omega_diag_ev=curve if omega_grid else None,
        omega_rel_ev=np.array([-20., 0., 20.]) if omega_grid else None,
        efermi_ev=0., e_eval_ev=energies * RYD_TO_EV)
    return path, results, operand


def _write(results, tmp_path, computational, file_sym):
    from gw.gw_config import QPSolver
    from gw.gw_output import write_results
    write_results(results, str(tmp_path / "sigma.dat"), str(tmp_path / "eqp0.dat"),
        str(tmp_path / "eqp1.dat"), str(tmp_path), (5, 5, 1), computational,
        SimpleNamespace(symmetry=lambda: file_sym), print_fn=lambda *a, **kw: None,
        degeneracy_policy="disabled", degeneracy_tol_ry=0., write_qp_rotations=False,
        qp_solver=QPSolver.ONE_SHOT_DFT)


@pytest.mark.parametrize("stamp", ["identity", "full"])
@pytest.mark.parametrize("same_domain", [False, True])
def test_writer_receipt_and_reader_use_source_file_rows_not_computational_rows(
        tmp_path, stamp, same_domain):
    from file_io.restart_bundle import read_eqp_assembly_receipt
    from file_io.sigma_output import EQP_ASSEMBLY_FILE_ROWS_ATTR, EQP_ASSEMBLY_DATASET

    path, results, operand = _source(tmp_path, stamp=stamp)
    rows = np.array([0, 1, 2, 5, 6, 7, 10, 11, 12])
    file_sym = _symmetry(rows)
    computational = file_sym if same_domain else _symmetry(np.arange(25))
    with h5py.File(path, "r") as f:
        raw = {key: f[key][()] for key in ("sigma_c_kij_ev", "sigma_total_kij_ev",
                                          "sigma_eval_rel_ev")}
    _write(results, tmp_path, computational, file_sym)
    receipt = read_eqp_assembly_receipt(path)
    np.testing.assert_array_equal(receipt["file_wedge_full_bz_rows"], rows)
    np.testing.assert_array_equal(receipt["eval_energies_rel_ev"], results.e_eval_ev[rows])
    np.testing.assert_array_equal(receipt["sigma_c_on_shell"], operand[:, rows])
    assert receipt["eval_energies_provenance"] == "at_e_dft"
    assert receipt["eval_coverage"]["n_uncovered"] == np.count_nonzero(operand[-1, rows] == 0)
    assert receipt["eval_coverage"]["scope"] == "file_wedge"
    assert receipt["eval_source_coverage"]["n_uncovered"] == np.count_nonzero(operand[-1] == 0)
    with h5py.File(path, "r") as f:
        for name, value in raw.items():
            np.testing.assert_array_equal(f[name][()], value)
        np.testing.assert_array_equal(f[EQP_ASSEMBLY_DATASET].attrs[EQP_ASSEMBLY_FILE_ROWS_ATTR], rows)
        assert f["sigma_c_kij_ev"].shape == (2, 25, 2, 2)
    assert results.E_qp_ry.shape == (25, 2)
    # Nine text k blocks, never a symmetry reconstruction of the 25 inputs.
    from file_io.restart_bundle import read_bgw_eqp
    points, _, energies, _ = read_bgw_eqp(str(tmp_path / "eqp0.dat"))
    assert energies.shape == (9, 2)
    np.testing.assert_allclose(points, computational.unfolded_kpts[rows], atol=1e-9)


def test_omega_grid_subset_does_not_relabel_full_coverage_counts(tmp_path):
    from file_io.restart_bundle import read_eqp_assembly_receipt
    path, results, _ = _source(tmp_path, omega_grid=True)
    _write(results, tmp_path, _symmetry(np.arange(25)), _symmetry([0, 5, 10]))
    receipt = read_eqp_assembly_receipt(path)
    assert receipt["eval_coverage"] is None
    assert receipt["eval_source_coverage"]["scope"] == "raw_full_bz"


def test_legacy_reducing_map_receipt_is_unchanged_and_never_substitutes_stars(tmp_path):
    from file_io.restart_bundle import read_eqp_assembly_receipt
    path, results, _ = _source(tmp_path, stamp="reducing")
    rows = np.arange(5)
    sym = _symmetry(rows)
    sym.irr_idx_k = np.arange(25) % 5
    _write(results, tmp_path, sym, sym)
    receipt = read_eqp_assembly_receipt(path)
    np.testing.assert_array_equal(receipt["eval_energies_rel_ev"], results.e_eval_ev[rows])
    assert "eval_source_coverage" not in receipt
    assert "scope" not in receipt["eval_coverage"]


def test_identity_row_labels_with_nontrivial_actions_do_not_bypass_coverage(tmp_path):
    path, results, _ = _source(tmp_path, stamp="identity_actions")
    with pytest.raises(ValueError, match="does not cover the raw cube"):
        _write(results, tmp_path, _symmetry(np.arange(25)), _symmetry([0, 1, 2]))


def test_reducing_map_reader_refuses_unequal_eval_axis_instead_of_unfolding(tmp_path):
    from file_io.restart_bundle import read_eqp_assembly_receipt
    path, results, _ = _source(tmp_path, stamp="reducing")
    _write(results, tmp_path, _symmetry(np.arange(25)), _symmetry(np.arange(9)))
    with pytest.raises(ValueError, match="STAR wedge"):
        read_eqp_assembly_receipt(path)


@pytest.mark.parametrize("rows,match", [([0, 1], "does not cover the raw cube"),
                                        ([0, 1, 2, 3, 25], "out of range"),
                                        (list(range(25)), "got shape")])
def test_reducing_map_coverage_and_invalid_row_witnesses_still_refuse(tmp_path, rows, match):
    from gw.eqp_bgw import assemble_eqp
    from file_io.sigma_output import append_eqp_assembly_receipt_h5
    path, _, _ = _source(tmp_path, stamp="reducing")
    n = 5 if len(rows) != 2 else 2
    assembly = assemble_eqp(kpoints_irr_frac=np.zeros((n, 3)), band_offset=0,
        e_dft_ev=np.zeros((n, 2)), kin_ion_diag_ev=np.zeros((n, 2)),
        hartree_diag_ev=np.zeros((n, 2)), sigma_x_diag_ev=np.zeros((n, 2)),
        sigma_c_on_shell=np.stack([np.zeros((n, 2))] * 3 + [np.ones((n, 2))]),
        e_dft_rel_ev=np.zeros((n, 2)), print_fn=lambda *a, **kw: None)
    with pytest.raises(ValueError, match=match):
        append_eqp_assembly_receipt_h5(path, assembly=assembly,
            file_wedge_full_bz_rows=rows, degeneracy_policy="disabled", degeneracy_tol_ry=0.)

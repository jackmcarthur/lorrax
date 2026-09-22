"""The canonical SC Hamiltonian must survive an optional full-WFN write."""
import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from gw import sc_iteration
from file_io import qp_wfn, restart_bundle


@pytest.mark.parametrize("write_full_wfn", [False, True])
def test_small_qp_artifact_preserves_accepted_hamiltonian(
        tmp_path, monkeypatch, write_full_wfn):
    # The off-diagonal pair belongs to the retained block; the third band
    # has a scissor energy.  An unpartitioned post-Sigma H is a different
    # physical artifact even if its frontier gap looks close.
    h = np.array([[[-0.8, 0.2j, 0.0],
                   [-0.2j, 0.4, 0.0],
                   [0.0, 0.0, 1.9]]], dtype=np.complex128)
    state = SimpleNamespace(H_qp_dft=h, occupation_state=None, outputs=None)
    points = np.zeros((1, 3))
    sym = SimpleNamespace(
        unfolded_kpts=points, kirr_fullids=np.array([0]),
        nk_red=1,
        irr_idx_k=np.array([0]), sym_idx_k=np.array([0]),
        sym_mats_k=np.stack([np.eye(3), -np.eye(3)]))
    wfn = SimpleNamespace(
        energies=np.array([[[-0.7, 0.3, 1.5]]]), kpoints=points,
        nelec=1, nspinor=1, nbands=3, nkpts=1, path=None,
        occupation_state_capacity=2.0, symmetry=lambda: sym)

    def diagonalize(hamiltonian, n_occ, mesh):
        assert hamiltonian is h
        energy, rotation = np.linalg.eigh(hamiltonian)
        return energy, rotation, float((energy[0, 0] + energy[0, 1]) / 2)

    monkeypatch.setattr(sc_iteration, "_diagonalize_and_get_efermi", diagonalize)
    full_wfn_calls = []

    def full_wfn_writer(path, **kwargs):
        full_wfn_calls.append(path)
        assert write_full_wfn, "full orbital write must stay optional"

    monkeypatch.setattr(qp_wfn, "write_qp_wfn_h5", full_wfn_writer)
    monkeypatch.setattr(qp_wfn, "validate_qp_wfn_h5", lambda *_a, **_k: None)
    wfn_path, qp_path, _, reported_energies = sc_iteration.dump_qp_wfn_artifacts(
        state, n_occ=1, mesh_xy=None, wfn=wfn, sym=sym,
        band_slices=SimpleNamespace(b0=0, b3=3), kgrid=(1, 1, 1),
        output_dir=str(tmp_path), write_wfn_h5=write_full_wfn,
        print_fn=lambda *_a: None)

    artifact = restart_bundle.read_qp_rotations_artifact(qp_path)
    qp_wfn.authenticate_qp_rotations_source_wfn(
        artifact, wfn, artifact_path=qp_path)
    u, e = artifact["U_mnk"], artifact["E_qp_nk_rydberg"]
    reconstructed = (u * e[:, None, :]) @ u.conj().swapaxes(-1, -2)
    np.testing.assert_allclose(reconstructed, h, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(reported_energies, e)
    assert len(full_wfn_calls) == int(write_full_wfn)
    assert (wfn_path is not None) == write_full_wfn

    if not write_full_wfn:
        inputs = SimpleNamespace(
            input_dir=str(tmp_path), meta=SimpleNamespace(nelec=1, kgrid=(1,1,1), b_id_4_user=3),
            mesh_xy=None, kstar=None, wfn=wfn, sym=sym,
            band_slices=SimpleNamespace(b0=0, b3=3),
            config=SimpleNamespace(qp_rotations_k_storage="auto", occupation_clamp_tol=1e-3),
            print_fn=lambda *_a: None)
        sc_iteration._write_sc_seed(inputs, state)
        seed = restart_bundle.read_qp_rotations_artifact(
            str(tmp_path / "sc_seed" / "qp_wfn_rotations.h5"))
        np.testing.assert_array_equal(seed["E_qp_nk_rydberg"], e)
        np.testing.assert_array_equal(seed["U_mnk"], u)
        assert not (tmp_path / "sc_seed" / "WFN_qp.h5").exists()


def test_sc_driver_publishes_small_artifact_outside_full_wfn_guard():
    """A conditional caller recreates the measured wrong-H publication."""
    tree = ast.parse(inspect.getsource(sc_iteration.run_sc_driver))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "dump_qp_wfn_artifacts"]
    assert len(calls) == 1
    call = calls[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and call in list(ast.walk(node)):
            assert "write_wfn_h5" not in ast.unparse(node.test)
    assert "write_wfn_h5" in {keyword.arg for keyword in call.keywords}


def test_metal_qp_wfn_occupations_follow_the_final_full_ladder(
        tmp_path, monkeypatch):
    """Direct and compact publication must carry one full E/f state."""
    from gw.efermi import OccupationState, assert_fixed_n
    from gw.scissor import ScissorFit
    import symmetry_maps

    active_energies = np.array([
        [-1.0, -0.9, -0.8],
        [-1.0, 0.1, 0.2],
    ], dtype=np.float64)
    dft_energies = np.array([
        [-1.0, -0.9, -0.8, 1.0],
        [-1.0, 0.1, 0.2, 0.3],
    ], dtype=np.float64)
    rotations = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    h = np.array(
        [np.diag(row) for row in active_energies], dtype=np.complex128)
    carried = OccupationState(
        f_kn=np.array([[1.0, 1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0, 0.0]]),
        mu_ry=0.0, smearing_family="fd", smearing_width_ry=0.02,
        n_electrons=4.0)
    tail_fit = ScissorFit(
        alpha_v=-9.0, beta_v_ev=123.0,
        alpha_c=1.0, beta_c_ev=0.75,
        n_fit_v=2, n_fit_c=2, rmse_v_ev=0.0, rmse_c_ev=0.0,
        w_fit_v=2.0, w_fit_c=2.0)
    state = SimpleNamespace(
        H_qp_dft=h, occupation_state=carried,
        outputs=SimpleNamespace(tail_scissor_fit=tail_fit))
    points = np.zeros((2, 3))
    sym = SimpleNamespace(
        unfolded_kpts=points, kirr_fullids=np.array([0, 1]),
        irr_idx_k=np.array([0, 1]), sym_idx_k=np.array([0, 0]),
        sym_mats_k=np.stack([np.eye(3), -np.eye(3)]))
    wfn = SimpleNamespace(
        energies=np.array([dft_energies]), kpoints=points,
        kweights=np.array([0.5, 0.5]),
        nelec=2, num_electrons=4.0, nspinor=1, nbands=4, nkpts=2,
        path=None, occupation_state_capacity=2.0, symmetry=lambda: sym)

    monkeypatch.setattr(
        sc_iteration, "_diagonalize_and_get_efermi",
        lambda *_args: (active_energies, rotations, 0.0))
    monkeypatch.setattr(
        symmetry_maps, "unfold_file_wedge_to_full_bz",
        lambda _sym, table: np.asarray(table))
    monkeypatch.setattr(
        symmetry_maps, "reduce_full_bz_to_file_wedge",
        lambda _sym, table: np.asarray(table))
    calls = []
    monkeypatch.setattr(
        qp_wfn, "write_qp_wfn_h5",
        lambda _path, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        qp_wfn, "validate_qp_wfn_h5", lambda *_a, **_k: None)

    _, qp_path, mu_ry, _ = sc_iteration.dump_qp_wfn_artifacts(
        state, n_occ=2, mesh_xy=None, wfn=wfn, sym=sym,
        band_slices=SimpleNamespace(b0=0, b3=3),
        logical_band_stop=4, kgrid=(2, 1, 1), output_dir=str(tmp_path),
        write_wfn_h5=True, print_fn=lambda *_a: None)

    occupations = calls[0]["occupations_kn"]
    final_state = calls[0]["occupation_state"]
    assert occupations[0, 2] > 0.99  # three occupied states at the first k
    assert occupations[1, 1] < 0.01  # one occupied state at the second k
    assert not np.array_equal(
        occupations, np.array([[1, 1, 0, 0], [1, 1, 0, 0]]))
    assert mu_ry == final_state.mu_ry
    assert final_state.occ_hash != carried.occ_hash
    assert_fixed_n(final_state, np.array([0.5, 0.5]), state_capacity=2.0)

    artifact = restart_bundle.read_qp_rotations_artifact(qp_path)
    direct_final_energies = np.asarray(calls[0]["enk_full_base_ry"]).copy()
    direct_final_energies[:, :3] = calls[0]["enk_active_qp_ry"]
    assert not np.array_equal(direct_final_energies[:, 3:],
                              dft_energies[:, 3:])
    np.testing.assert_array_equal(
        artifact["E_full_nk_rydberg"], direct_final_energies)
    np.testing.assert_array_equal(
        artifact["occupations_kn"], np.asarray(final_state.f_kn))
    assert artifact["occupation_provenance"] == {
        "occ_hash": final_state.occ_hash,
        "mu_ry": final_state.mu_ry,
        "smearing_family": final_state.smearing_family,
        "smearing_width_ry": final_state.smearing_width_ry,
        "n_electrons": final_state.n_electrons,
    }

    from postprocess.rotate_wfn_to_qp import _stored_final_state_args
    rebuilt = _stored_final_state_args(artifact, np.array([1, 0]))
    companion_final_energies = rebuilt["enk_full_base_ry"].copy()
    companion_final_energies[:, :3] = artifact[
        "E_qp_nk_rydberg"][[1, 0]]
    np.testing.assert_array_equal(
        companion_final_energies, direct_final_energies[[1, 0]])
    np.testing.assert_array_equal(
        rebuilt["occupations_kn"], np.asarray(final_state.f_kn)[[1, 0]])
    assert rebuilt["occupation_state"].occ_hash == final_state.occ_hash

    h5py = pytest.importorskip("h5py")
    from file_io.qp_wfn import QP_WFN_OCC_HASH_ATTR
    with h5py.File(qp_path, "a") as h5:
        del h5.attrs[QP_WFN_OCC_HASH_ATTR]
    with pytest.raises(ValueError, match="incomplete final occupation"):
        restart_bundle.read_qp_rotations_artifact(qp_path)


def test_driver_gap_uses_the_accepted_sc_spectrum(tmp_path):
    """Execute the actual report call with deliberately different H spectra."""
    from pathlib import Path
    from common.units import RYD_TO_EV
    from gw.production_report import GWProductionReport

    source = Path(sc_iteration.__file__).with_name("gw_jax.py").read_text()
    tree = ast.parse(source)
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "qp_gap")
    output = []
    report = GWProductionReport(
        str(tmp_path / "report"), runtime=SimpleNamespace(process_index=0),
        debug=False, stdout=output.append)
    namespace = {
        "report": report, "band_slices": SimpleNamespace(b0=0, b2=1),
        "enk_dft": np.array([[0.0, 0.5]]) / RYD_TO_EV,
        "E_full": np.array([[0.1, 2.5]]) / RYD_TO_EV,
        "sc_qp_energies_ry": np.array([[0.2, 1.4]]) / RYD_TO_EV,
    }
    eval(compile(ast.Expression(call), "actual_driver_gap_call", "eval"), namespace)
    assert any("Full-matrix effective-H gap: 1.20000 eV" in line for line in output)
    assert not any("2.40000 eV" in line for line in output)
    output.clear()
    namespace["sc_qp_energies_ry"] = None
    eval(compile(ast.Expression(call), "actual_driver_gap_call", "eval"), namespace)
    assert any("Full-matrix effective-H gap: 2.40000 eV" in line for line in output)

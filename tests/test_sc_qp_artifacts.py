"""The canonical SC Hamiltonian must survive an optional full-WFN write."""
import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from gw import sc_iteration
from file_io import qp_wfn
# main moved the QP-rotation READERS into the shared restart bundle; the
# writer and the WFN authentication stayed in qp_wfn.
from file_io import restart_bundle as qp_reader


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
        irr_idx_k=np.array([0]), sym_idx_k=np.array([0]),
        sym_mats_k=np.stack([np.eye(3), -np.eye(3)]))
    # ``dump_qp_wfn_artifacts`` reduces the full BZ to the FILE wedge through
    # ``wfn.symmetry()``, so the fake WFN has to answer that the same way the
    # loader's does -- with this deck's own tables.
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
    wfn_path, qp_path, _, reported_energies = sc_iteration.dump_qp_wfn_artifacts(
        state, n_occ=1, mesh_xy=None, wfn=wfn, sym=sym,
        band_slices=SimpleNamespace(b0=0, b3=3), kgrid=(1, 1, 1),
        output_dir=str(tmp_path), write_wfn_h5=write_full_wfn,
        print_fn=lambda *_a: None)

    artifact = qp_reader.read_qp_rotations_artifact(qp_path)
    qp_wfn.authenticate_qp_rotations_source_wfn(
        artifact, wfn, artifact_path=qp_path)
    u, e = artifact["U_mnk"], artifact["E_qp_nk_rydberg"]
    reconstructed = (u * e[:, None, :]) @ u.conj().swapaxes(-1, -2)
    np.testing.assert_allclose(reconstructed, h, atol=2e-15, rtol=0)
    np.testing.assert_array_equal(reported_energies, e)
    assert len(full_wfn_calls) == int(write_full_wfn)
    assert (wfn_path is not None) == write_full_wfn


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

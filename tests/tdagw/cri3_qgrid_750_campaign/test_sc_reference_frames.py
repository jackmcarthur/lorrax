"""Metadata-only lifecycle tests; NOT MPI/GPU/SC physics verification."""
import gc
from dataclasses import make_dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "sc_reference_frames", Path(__file__).with_name("run_sc_reference_frames.py"))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
Verdict = make_dataclass("Verdict", [("converged", bool)])


def fixture_owner(*, converge=True, max_one=False, fail=False, wrong_rotation=False):
    saved, faces = {}, []
    energy = np.array([[1., 2., 8., 99.], [3., 4., 9., 99.]])
    U = np.broadcast_to(np.eye(2, dtype=complex), (2, 2, 2)).copy()
    inputs = NS(wfns_dft=object(), wfns_transverse=None, kstar=None)
    H = U * energy[:, None, :2]
    owner = NS()

    def rotate(source, unitary, **kwargs):
        face = np.empty((2, 3, 4), complex)
        faces.append(weakref.ref(face))
        return NS(enk=energy + (1. if wrong_rotation else 0.),
                  occ=np.array([[1., 0., 0., 0.]] * 2),
                  slices=NS(b0=0, b4_logical=3, sigma=slice(0, 2)), face=face)

    def mapping(state, supplied):
        bundle = owner.rotate_wavefunctions(inputs.wfns_dft, U,
                                             active_slice=slice(0, 2))
        # Fake donation proves capture has already completed, not an end hook.
        assert any(name.endswith(".npz") for name in saved)
        bundle.enk[:] = -123.
        if fail:
            raise RuntimeError("fake wall/map failure")
        return NS(H_qp_dft=state.H_qp_dft + 20., outputs=NS(
            tail_scissor_fit=None, screening=NS(shared_pole={"path": "model"},
                                               iteration_head=None)))

    def solve(state, supplied, **kwargs):
        output = owner.gw_iteration_map(state, supplied)
        return NS(H_qp_dft=output.H_qp_dft if max_one else state.H_qp_dft,
                  iteration=1, convergence_verdict=Verdict(converge)), [.1]

    owner.rotate_wavefunctions, owner.gw_iteration_map = rotate, mapping
    owner.run_self_consistency = solve

    def publish(name, **kw):
        assert name not in saved
        saved[name] = kw

    status, restore = helper.install_capture(
        owner, gather=np.asarray, publish=publish,
        describe=lambda *_: ({"scope": "FAKE_METADATA_ONLY"}, {}), expected_bands=3)
    return owner, inputs, NS(H_qp_dft=H, iteration=0), saved, status, restore, faces


@pytest.mark.parametrize("converge,max_one,expected", [
    (True, False, "CONVERGED_ACCEPTED_INPUT"),
    (False, False, "NOT_CONVERGED"),
    (True, True, "ONE_MAP_DIAGNOSTIC")])
def test_native_input_not_output_join_and_no_face_retention(converge, max_one, expected):
    owner, inputs, state, saved, status, restore, faces = fixture_owner(
        converge=converge, max_one=max_one)
    try:
        final, history = owner.run_self_consistency(
            state, inputs, max_iter=1 if max_one else 30)
        assert final.iteration == 1 and history == [.1]
        record = saved["native_solver_return.json"]["record"]
        assert record["status"] == expected
        assert record["reference_ready"] is False  # outer CLI must also finish
        assert record["accepted_input_join"] is (not max_one)
        frame = saved[status["latest"]["frame"] + ".npz"]["arrays"]
        np.testing.assert_array_equal(frame["energy_ry"], [[1, 2, 8], [3, 4, 9]])
        assert set(frame) == {"energy_padded_ry", "occupation_padded", "U_active",
                              "input_H_loop_ry", "energy_ry", "occupations"}
        gc.collect()
        assert all(ref() is None for ref in faces)
        assert not any(hasattr(value, "enk") for value in status.values())
    finally:
        restore()


def test_failed_map_retains_input_but_no_solver_or_completion_claim():
    owner, inputs, state, saved, status, restore, _ = fixture_owner(fail=True)
    try:
        with pytest.raises(RuntimeError, match="fake wall"):
            owner.run_self_consistency(state, inputs, max_iter=30)
        assert any(name.endswith(".npz") for name in saved)
        assert "native_solver_return.json" not in saved
        assert not any(name.endswith("_map_complete.json") for name in saved)
        assert status["solver"] is None and status["maps"] == 0
    finally:
        restore()


def test_rotation_mismatch_refuses_no_accepted_reference():
    owner, inputs, state, saved, status, restore, _ = fixture_owner(wrong_rotation=True)
    try:
        with pytest.raises(ValueError, match="current input H"):
            owner.run_self_consistency(state, inputs, max_iter=30)
        assert not saved and status["solver"] is None
    finally:
        restore()


def test_atomic_output_is_exclusive_and_numpy_only(tmp_path):
    path = tmp_path / "frame.npz"
    helper.exclusive_write(path, arrays={"E": np.arange(4.)})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        helper.exclusive_write(path, arrays={"E": np.zeros(4)})
    assert path.read_bytes() == before
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["E"], np.arange(4.))


def test_unknown_or_different_final_input_cannot_be_converged():
    H = np.eye(2)
    latest = dict(input_h_sha256=helper.array_digest(H + 1), frame="old")
    result = helper.final_classification(H, latest, NS(converged=True), 30)
    assert result["status"] == "NOT_CONVERGED"
    assert result["accepted_input_join"] is False


def test_real_native_metadata_contract_without_wavefunction_faces(tmp_path):
    """Real config/basis/SCInputs/fingerprint types; no native numerical stage."""
    import jax
    from jax.sharding import Mesh
    from common.centroid_basis import PackedCentroidBasis
    from common.parallel_transport import bind_wfn_fingerprint, fingerprint_from_binding
    from gw.gw_config import LorraxConfig
    from gw.sc_iteration import SCInputs
    from symmetry_maps import KStarMap
    source = Path(__file__).resolve().parents[3]
    deck = tmp_path/'metadata_only.in'
    deck.write_text('[cohsex]\nsys_dim=2\nnumber_bands=750\nnumber_bands_protected=160\n'
        'qgrid=8 8 1\nhead_correction=full\nsc_head_update=dft_velocity\n'
        'qp_solver=self_consistent\ncompute_mode=mpa\nsigma_w_model=shared_pole\n'
        'restart=false\nwrite_restart_tensors=false\nwrite_wfn_h5=false\nqp_rotations_k_storage=full\n')
    config = LorraxConfig.from_input_file(str(deck), resolve_hardware=False)
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1), ("x", "y"))
    indices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
    basis = PackedCentroidBasis.build(indices, None, (2, 2, 1), mesh, identity=True)
    wfn = NS(energies=np.zeros((2, 3)), kpoints=np.zeros((2, 3)),
             nelec=2, nspinor=2, nbands=3)
    binding = bind_wfn_fingerprint(wfn)
    kstar = KStarMap(np.array([0, 0, 1, 1]), np.zeros(4, dtype=int), 1)
    inputs = SCInputs(
        wfns_dft=None, V_q=None, kin_ion_dft=None, head_channel=None,
        quad=None, e_ref=0., static_head_terms=None, head_resolver=None,
        screening_model_fn=None, config=config,
        meta=NS(mu_basis=basis, kgrid=(8, 8, 1), qgrid=(8, 8, 1)),
        mesh_xy=mesh, sym=NS(unfolded_kpts=np.zeros((4, 3))), wfn=wfn,
        centroid_indices=indices, band_slices=None, input_dir=".", partition=None,
        e_dft_active_kn_ry=None, valence_mask_active_kn=None,
        material_class="insulator", kstar=kstar, wfn_fingerprint_binding=binding,
        charge_zeta_identity={"scheme": "contract-test", "digest": "abc"})
    record, arrays = helper.describe_native_frame(
        inputs, None, source=source, deck=deck, revision="1134957",
        code_hashes={"metadata_test": "not_material_evidence"})
    assert record["head_correction"] == "full"
    assert record["sc_head_update"] == "dft_velocity"
    assert record["wfn_fingerprint"] == fingerprint_from_binding(binding, wfn)
    assert record["centroid_logical"] == 3
    np.testing.assert_array_equal(arrays["input_loop_full_rows"], [0, 2])
    np.testing.assert_array_equal(arrays["centroid_canonical_indices"], indices)
    import json
    json.dumps(record, allow_nan=False)
    assert record["large_face_retention"] is False

"""Seed-only charge-SC Hamiltonian transport into a new SC run."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from file_io.restart_bundle import validate_qp_rotations_frame
from gw import sc_iteration as sc


def test_qp_seed_reconstruction_partition_enters_real_rcrop_seam(
    monkeypatch, tmp_path,
):
    rng = np.random.default_rng(20260920)
    nk, nb = 2, 4
    rotations = []
    for _ in range(nk):
        raw = np.eye(nb) + 1.0e-2 * (
            rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb)))
        rotations.append(np.linalg.qr(raw)[0])
    U = np.asarray(rotations)
    E = np.broadcast_to(
        np.asarray([-1.0, -0.2, 0.4, 1.0]), (nk, nb)).copy()
    kpoints = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    artifact = {
        "U_mnk": U,
        "E_qp_nk_rydberg": E,
        "band_range": np.asarray([3, 3 + nb]),
        "kgrid": np.asarray([2, 1, 1]),
        "kpoints_crys": kpoints,
    }

    returned_U, returned_E, band_range = validate_qp_rotations_frame(
        artifact, kgrid=(2, 1, 1), kpoints_crys=kpoints,
        artifact_path="qp_wfn_rotations.h5")
    H = np.einsum(
        "kmn,kn,kln->kml", returned_U, returned_E,
        np.conj(returned_U), optimize=True)
    direct = np.stack([
        U[k] @ np.diag(E[k]) @ U[k].conj().T for k in range(nk)])

    np.testing.assert_allclose(H, direct, rtol=0.0, atol=2.0e-15)
    np.testing.assert_array_equal(returned_E, E)
    assert band_range == (3, 3 + nb)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(H), E, rtol=0.0, atol=2.0e-15)

    broken = dict(artifact, U_mnk=U.copy())
    broken["U_mnk"][0, 0, 0] += 1.0e-5
    with pytest.raises(ValueError, match="not unitary"):
        validate_qp_rotations_frame(
            broken, kgrid=(2, 1, 1), kpoints_crys=kpoints,
            artifact_path="qp_wfn_rotations.h5")

    # The initializer's partition is classified from this seed eigensystem,
    # then handed through run_self_consistency into the real rCROP driver.
    # The map and accelerator arithmetic are cheap doubles; rCROP's startup
    # metric construction is real, which pins the pre-map partition seam.
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    cfg = SimpleNamespace(
        sc=SimpleNamespace(
            exact_degeneracy_tol_ev=1.0e-4, buffer_nbands=0,
            dump_dir=None),
        sigma=SimpleNamespace(omega_min_ev=-5.0, omega_max_ev=5.0),
        screening=SimpleNamespace(occ_broadening_ev=0.0),
    )
    full_reference = np.zeros((nk, 3 + nb), dtype=np.float64)
    full_reference[:, :3] = np.asarray([-4.0, -3.0, -2.0])
    full_reference[:, 3:] = E
    inputs = SimpleNamespace(
        sym=SimpleNamespace(unfolded_kpts=kpoints),
        wfn=SimpleNamespace(
            energies=full_reference[None, ...], efermi=10.0,
            kgrid=(2, 1, 1)),
        wfns_dft=SimpleNamespace(enk=jnp.asarray(E)),
        e_dft_active_kn_ry=E,
        config=cfg,
        band_slices=SimpleNamespace(
            b0=3, b3=3 + nb, sigma=slice(0, nb),
            sigma_range=(3, 3 + nb)),
        meta=SimpleNamespace(nelec=2),
        material_class="insulator",
        parallel_transport=None,
        kstar=SimpleNamespace(is_identity=True),
        initial_state_role="external_qp_seed",
        partition=SimpleNamespace(
            protected_mask=np.zeros(nb, dtype=bool),
            in_range_mask=np.zeros(nb, dtype=bool)),
        mesh_xy=mesh,
        input_dir=str(tmp_path),
        wfn_fingerprint_binding=None,
        print_fn=lambda *_args: None,
        record_fn=None,
    )
    monkeypatch.setattr(
        "symmetry_maps.unfold_file_wedge_to_full_bz",
        lambda _sym, values: np.asarray(values))
    monkeypatch.setattr(
        "file_io.restart_bundle.read_qp_rotations_artifact",
        lambda _path: artifact)
    monkeypatch.setattr(
        "file_io.qp_wfn.authenticate_qp_rotations_source_wfn",
        lambda *_args, **_kwargs: "mock-source-fingerprint")
    state = sc.make_initial_state_from_qp_rotations(
        inputs, "qp_wfn_rotations.h5")
    np.testing.assert_allclose(
        np.asarray(state.H_qp_dft), direct, rtol=0.0, atol=2.0e-15)
    partition = state.partition
    expected_protected = np.broadcast_to(
        np.asarray([False, True, True, False]), (nk, nb))
    np.testing.assert_array_equal(
        np.asarray(partition.protected_mask), expected_protected)

    # The seeded map must carry the same freeze decision through its output
    # scissor step, which may otherwise promote new frontier identities.
    classified, _, _, _, frozen = sc._classify_sc_partition(
        E, U, None, previous_partition=partition, iteration=0,
        inputs=inputs, current_mu_ry=0.1)
    assert frozen
    np.testing.assert_array_equal(
        classified.protected_mask, partition.protected_mask)

    payload = SimpleNamespace(scissor_fit=None, tail_scissor_fit=None)
    seen = []

    def fake_map(state, _inputs):
        seen.append(state.partition)
        return sc.SCState(
            H_qp_dft=state.H_qp_dft + 1.0e-3,
            iteration=state.iteration + 1,
            partition=state.partition,
            outputs=payload)

    verdict = SimpleNamespace(
        converged=False, summary=lambda: "synthetic nonconverged map")
    monkeypatch.setattr(sc, "gw_iteration_map", fake_map)
    monkeypatch.setattr(
        sc, "_sc_identity_for_call",
        lambda _i, state, _ein, _eout, _hist, cutoff_ev: (verdict, state))
    monkeypatch.setattr(
        sc, "_sc_map_gain_for_call",
        lambda _i, _s, _e, previous: (None, previous))
    monkeypatch.setattr(sc, "_write_sc_eqp_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(sc, "_clear_sc_eqp_snapshots", lambda *_a, **_k: None)
    monkeypatch.setattr(
        sc, "_clear_sc_rotation_snapshots", lambda *_a, **_k: None)
    monkeypatch.setattr(
        sc, "_kshard_eigh_kernels",
        lambda _mesh: (None, lambda value: jnp.linalg.eigvalsh(value)))

    import mixing.acceleration as acceleration

    def fake_rcrop(residual_fn, x, **_kwargs):
        residual = residual_fn(x)
        return acceleration.AccelerationResult(
            x=x, residual_norms=jnp.asarray([jnp.linalg.norm(residual)]),
            iterations=1, converged=False)

    monkeypatch.setattr(acceleration, "rcrop_nojit", fake_rcrop)
    final, _ = sc.run_self_consistency(
        state, inputs, max_iter=2, accelerator="rcrop", history_depth=1)
    assert seen and seen[0] is partition
    assert final.partition is partition

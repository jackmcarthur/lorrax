"""Seed-only charge-SC Hamiltonian transport into a new SC run."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from file_io.qp_wfn import write_qp_rotations_h5
from file_io.restart_bundle import (
    read_qp_rotations_artifact, validate_qp_rotations_frame)
from gw import sc_iteration as sc
from gw.scissor import ScissorFit


def test_qp_seed_reconstruction_partition_enters_real_anderson_seam(
    monkeypatch, tmp_path,
):
    rng = np.random.default_rng(20260920)
    nk, nk_loop, nb = 64, 13, 4
    rotations = []
    for _ in range(nk):
        raw = np.eye(nb) + 1.0e-2 * (
            rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb)))
        rotations.append(np.linalg.qr(raw)[0])
    U = np.asarray(rotations)
    E = np.broadcast_to(
        np.asarray([-1.0, -0.2, 0.4, 1.0]), (nk, nb)).copy()
    kpoints = np.zeros((nk, 3), dtype=np.float64)
    kpoints[:, 0] = np.arange(nk) / nk
    artifact = {
        "U_mnk": U,
        "E_qp_nk_rydberg": E,
        "band_range": np.asarray([3, 3 + nb]),
        "kgrid": np.asarray([4, 4, 4]),
        "kpoints_crys": kpoints,
    }

    returned_U, returned_E, band_range = validate_qp_rotations_frame(
        artifact, kgrid=(4, 4, 4), kpoints_crys=kpoints,
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
            broken, kgrid=(4, 4, 4), kpoints_crys=kpoints,
            artifact_path="qp_wfn_rotations.h5")

    # The companion's accepted partition and active law round-trip through the
    # format owner, then enter run_self_consistency through the real rCROP seam.
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
    wfn = SimpleNamespace(
        energies=full_reference[None, ...], kpoints=kpoints,
        efermi=10.0, kgrid=(4, 4, 4), nelec=2, nspinor=1,
        nbands=3 + nb, path=None)
    protected = np.broadcast_to(
        np.asarray([False, True, True, False]), (nk, nb)).copy()
    protected[-1] = [True, False, True, False]
    in_range = np.broadcast_to(
        np.asarray([False, True, False, False]), (nk, nb)).copy()
    fit = ScissorFit(
        alpha_v=1.0, beta_v_ev=0.0,
        alpha_c=0.75, beta_c_ev=9.5,
        n_fit_v=39, n_fit_c=0,
        rmse_v_ev=0.0, rmse_c_ev=0.0,
        w_fit_v=192.0, w_fit_c=0.0)
    policy = {
        "protected_mask": protected,
        "in_range_mask": in_range,
        "active_scissor": vars(fit),
    }
    artifact_path = str(tmp_path / "qp_wfn_rotations.h5")
    write_qp_rotations_h5(
        artifact_path, U_mnk=U, E_qp_nk=E * 0.5,
        band_start=3, band_stop=3 + nb, kpoints_crys=kpoints,
        nkx=4, nky=4, nkz=4, kirr_to_kfull=np.arange(nk_loop),
        source_wfn=wfn, sc_seed_policy=policy)
    stored = read_qp_rotations_artifact(artifact_path)["sc_seed_policy"]
    np.testing.assert_array_equal(stored["protected_mask"], protected)
    np.testing.assert_array_equal(stored["in_range_mask"], in_range)
    assert stored["active_scissor"] == vars(fit)

    kstar = SimpleNamespace(
        is_identity=False,
        select=lambda values: np.asarray(values)[:nk_loop],
        broadcast=lambda values: np.broadcast_to(
            np.asarray(values)[0], (nk,) + np.asarray(values).shape[1:]))
    inputs = SimpleNamespace(
        sym=SimpleNamespace(unfolded_kpts=kpoints),
        wfn=wfn,
        wfns_dft=SimpleNamespace(enk=jnp.asarray(E)),
        e_dft_active_kn_ry=E,
        config=cfg,
        band_slices=SimpleNamespace(
            b0=3, b3=3 + nb, sigma=slice(0, nb),
            sigma_range=(3, 3 + nb)),
        meta=SimpleNamespace(nelec=2),
        material_class="insulator",
        parallel_transport=None,
        kstar=kstar,
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
    state = sc.make_initial_state_from_qp_rotations(
        inputs, artifact_path)
    np.testing.assert_allclose(
        np.asarray(state.H_qp_dft), direct[:nk_loop], rtol=0.0, atol=2.0e-15)
    partition = state.partition
    np.testing.assert_array_equal(partition.protected_mask, protected)
    np.testing.assert_array_equal(partition.in_range_mask, in_range)
    assert sc._frozen_scissor_fits(state) == (fit, None)

    # A present policy with no active fit is an established ``None`` law,
    # distinct from a legacy artifact that has no policy.  Preserve the tuple
    # so rCROP does not recapture a newly fitted active law on map 0.
    none_policy_path = str(tmp_path / "qp_wfn_rotations_none_policy.h5")
    write_qp_rotations_h5(
        none_policy_path, U_mnk=U, E_qp_nk=E * 0.5,
        band_start=3, band_stop=3 + nb, kpoints_crys=kpoints,
        nkx=4, nky=4, nkz=4, kirr_to_kfull=np.arange(nk_loop),
        source_wfn=wfn,
        sc_seed_policy={
            "protected_mask": protected,
            "in_range_mask": in_range,
            "active_scissor": None,
        })
    none_policy_state = sc.make_initial_state_from_qp_rotations(
        inputs, none_policy_path)
    assert none_policy_state.frozen_scissor_fits == (None, None)
    assert sc._frozen_scissor_fits(none_policy_state) == (None, None)

    # A legacy U/E-only companion keeps the established seed classification
    # and starts without a frozen law; absence must not manufacture identity.
    with monkeypatch.context() as legacy:
        legacy.setattr(
            "file_io.restart_bundle.read_qp_rotations_artifact",
            lambda _path: artifact)
        legacy.setattr(
            "file_io.qp_wfn.authenticate_qp_rotations_source_wfn",
            lambda *_args, **_kwargs: "mock-source-fingerprint")
        legacy.setattr(
            "symmetry_maps.unfold_file_wedge_to_full_bz",
            lambda _sym, values: np.asarray(values))
        legacy_state = sc.make_initial_state_from_qp_rotations(
            inputs, "legacy_qp_wfn_rotations.h5")
    expected_legacy = np.broadcast_to(
        np.asarray([False, True, True, False]), (nk, nb))
    np.testing.assert_array_equal(
        legacy_state.partition.protected_mask, expected_legacy)
    assert legacy_state.frozen_scissor_fits is None

    payload = SimpleNamespace(scissor_fit=None, tail_scissor_fit=None)
    seen = []

    def fake_map(state, _inputs):
        seen.append(state.partition)
        assert sc._frozen_scissor_fits(state) == (fit, None)
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

    def fake_anderson(residual_fn, x, **_kwargs):
        residual = residual_fn(x)
        return acceleration.AccelerationResult(
            x=x, residual_norms=jnp.asarray([jnp.linalg.norm(residual)]),
            iterations=1, converged=False)

    monkeypatch.setattr(acceleration, "anderson_nojit", fake_anderson)
    final, _ = sc.run_self_consistency(
        state, inputs, max_iter=2, accelerator="anderson", history_depth=1)
    assert seen and seen[0] is partition
    assert final.partition is partition
    assert final.frozen_scissor_fits == (fit, None)

"""Discriminating contracts for the htransform orbital observable."""

import numpy as np


def test_fourier_hermitian_accepts_component_axis():
    import jax.numpy as jnp
    from bandstructure.htransform import fourier_hermitian

    rng = np.random.default_rng(202609051)
    R = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    raw = (rng.normal(size=(len(R), 3, 4, 4))
           + 1j * rng.normal(size=(len(R), 3, 4, 4)))
    q = np.asarray([[0.173, -0.219, 0.0]])
    value = np.asarray(fourier_hermitian(
        jnp.asarray(q), jnp.asarray(R), jnp.asarray(raw))
    )
    phase = np.exp(-2j * np.pi * (q @ R.T))
    contracted = 0.5 * np.tensordot(phase, raw, axes=((1,), (0,)))
    ref = contracted + contracted.swapaxes(-1, -2).conj()
    np.testing.assert_allclose(value, ref, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(value, value.swapaxes(-1, -2).conj())


def test_projected_velocity_roundtrip_recovers_band_operator_on_coarse_grid():
    import jax
    import jax.numpy as jnp
    from bandstructure.htransform import (
        build_R_grid_np,
        build_galerkin_velocity_R,
        fourier_hermitian,
    )
    from jax.sharding import Mesh

    rng = np.random.default_rng(202609052)
    grid = (2, 2, 1)
    nk, nb = 4, 4
    coefficients = []
    for _ in range(nk):
        z = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        coefficients.append(np.linalg.qr(z)[0])
    coefficients = np.asarray(coefficients)
    raw = (rng.normal(size=(3, nk, nb, nb))
           + 1j * rng.normal(size=(3, nk, nb, nb)))
    velocity = 0.5 * (raw + raw.swapaxes(-1, -2).conj())
    devices = np.asarray(jax.devices())
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    mesh = Mesh(devices, ("x", "y"))
    velocity_R = build_galerkin_velocity_R(
        jnp.asarray(coefficients), jnp.asarray(velocity), grid, mesh)
    q = np.stack(np.meshgrid(
        np.arange(2) / 2.0, np.arange(2) / 2.0, [0.0], indexing="ij"),
        axis=-1).reshape(-1, 3)
    basis_velocity = np.asarray(fourier_hermitian(
        jnp.asarray(q), jnp.asarray(build_R_grid_np(grid)), velocity_R))
    vectors = coefficients.transpose(0, 2, 1)
    got = np.einsum(
        "bim,baij,bjn->bamn", np.conj(vectors), basis_velocity, vectors,
        optimize=True)
    np.testing.assert_allclose(
        got, velocity.transpose(1, 0, 2, 3), rtol=3e-13, atol=3e-13)


def test_shared_jax_orbital_contraction_matches_incumbent_numpy():
    import jax.numpy as jnp
    from psp.orbital_response import (
        orbital_cA_cB_jax,
        orbital_pieces_at_k,
    )

    rng = np.random.default_rng(202609053)
    nk, nb, nocc = 3, 7, 3
    energies = np.stack([
        np.arange(nb, dtype=np.float64) * 0.37 + 0.01 * k
        for k in range(nk)])
    raw = (rng.normal(size=(nk, 3, nb, nb))
           + 1j * rng.normal(size=(nk, 3, nb, nb)))
    velocity = 0.5 * (raw + raw.swapaxes(-1, -2).conj())
    got_a, got_b = orbital_cA_cB_jax(
        jnp.asarray(velocity), jnp.asarray(energies),
        nocc=nocc, deps_tol_ry=1.0e-10)
    ref = [orbital_pieces_at_k(
        velocity[k], energies[k], nocc, 1.0e-10) for k in range(nk)]
    ref_a = np.stack([x[0].sum(axis=(1, 2)) for x in ref])
    ref_b = np.stack([x[1].sum(axis=(1, 2)) for x in ref])
    np.testing.assert_allclose(np.asarray(got_a), ref_a, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(np.asarray(got_b), ref_b, rtol=3e-13, atol=3e-13)


def test_dipole_reuse_streams_full_bz_and_applies_typed_projection(
        tmp_path, monkeypatch):
    import h5py
    from psp.orbital_magnetization import run_dipole_orbmag
    from psp.orbital_response import orbital_pieces_at_k

    rng = np.random.default_rng(202609054)
    nk, nb, nocc = 3, 5, 2
    raw = (rng.normal(size=(3, nk, nb, nb))
           + 1j * rng.normal(size=(3, nk, nb, nb)))
    velocity = 0.5 * (raw + raw.swapaxes(-1, -2).conj())
    energies = np.stack([
        np.arange(nb, dtype=np.float64) * 0.23 + 0.01 * k
        for k in range(nk)])
    path = tmp_path / "dipole.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("dipole_cart", data=velocity)
        h5.create_dataset("deltaE", data=np.zeros((nk, nb, nb)))
        h5.attrs.update({
            "prov_nval": 2, "prov_ncond": 3, "prov_nband": nb,
            "prov_nb_written": nb, "prov_bispinor": False,
            "prov_skip_vnl": False, "prov_vnl_mode": "analytic",
            "prov_vnl_velocity_sign": 1.0,
        })

    checked = {}

    def _authenticate(*_args, **kwargs):
        checked.update(kwargs)
        return True

    monkeypatch.setattr(
        "psp.dipole_store.check_dipole_provenance", _authenticate)

    class _Wfn:
        pass

    wfn = _Wfn()
    wfn.nkpts = nk
    wfn.energies = energies[None]

    class _Sym:
        nk_tot = nk
        irr_idx_k = np.arange(nk)
        active_symmetry_rows = np.asarray([0, 1], dtype=np.int32)
        trs_allowed = False
        operation_typing_source = "qe-schema"

        @staticmethod
        def cartesian_action(_rows, *, axial, time_odd):
            assert axial and time_odd
            return np.asarray((np.eye(3), np.diag((-1.0, -1.0, 1.0))))

    cA, cB, *_tail = run_dipole_orbmag(
        wfn, _Sym(), path, nb, nocc, 1.0e-10, (0.0, 0.0, 1.0))
    ref = [orbital_pieces_at_k(
        velocity[:, k], energies[k], nocc, 1.0e-10) for k in range(nk)]
    ref_a = np.mean([x[0].sum(axis=(1, 2)) for x in ref], axis=0)
    ref_b = np.mean([x[1].sum(axis=(1, 2)) for x in ref], axis=0)
    np.testing.assert_allclose(cA, [0.0, 0.0, ref_a[2]])
    np.testing.assert_allclose(cB, [0.0, 0.0, ref_b[2]])
    assert checked["bispinor"] is False
    assert checked["skip_vnl"] is False
    assert checked["vnl_mode"] == "analytic"
    assert checked["vnl_velocity_sign"] == 1.0

"""Discriminating contracts for the htransform orbital observable."""

import numpy as np


def test_fourier_derivative_matches_centered_difference():
    import jax.numpy as jnp
    from bandstructure.htransform import (
        fourier_hermitian_with_crystal_derivative,
    )

    rng = np.random.default_rng(202609051)
    R = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    raw = (rng.normal(size=(len(R), 4, 4))
           + 1j * rng.normal(size=(len(R), 4, 4)))
    q = np.asarray([[0.173, -0.219, 0.0]])
    value, derivative = fourier_hermitian_with_crystal_derivative(
        jnp.asarray(q), jnp.asarray(R), jnp.asarray(raw))
    value = np.asarray(value)
    derivative = np.asarray(derivative)
    step = 1.0e-6
    for axis in range(3):
        dq = np.zeros_like(q)
        dq[:, axis] = step
        plus = np.asarray(fourier_hermitian_with_crystal_derivative(
            jnp.asarray(q + dq), jnp.asarray(R), jnp.asarray(raw))[0])
        minus = np.asarray(fourier_hermitian_with_crystal_derivative(
            jnp.asarray(q - dq), jnp.asarray(R), jnp.asarray(raw))[0])
        np.testing.assert_allclose(
            derivative[:, axis], (plus - minus) / (2.0 * step),
            rtol=2.0e-10, atol=2.0e-10)
    np.testing.assert_allclose(value, value.swapaxes(-1, -2).conj())


def test_divided_difference_recovers_original_velocity():
    import jax.numpy as jnp
    from bandstructure.htransform import (
        velocity_from_transformed_derivative,
    )

    rng = np.random.default_rng(202609052)
    nb = 5
    z = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
    vectors, _ = np.linalg.qr(z)
    energies = np.asarray([[-1.1, -0.4, 0.2, 0.9, 1.8]])
    transformed = energies + 0.17 * energies ** 3
    raw = (rng.normal(size=(1, 3, nb, nb))
           + 1j * rng.normal(size=(1, 3, nb, nb)))
    velocity_eigen = 0.5 * (raw + raw.swapaxes(-1, -2).conj())
    for i in range(nb):
        velocity_eigen[:, :, i, i] = 0.0
    dE = energies[:, :, None] - energies[:, None, :]
    dlam = transformed[:, :, None] - transformed[:, None, :]
    offdiag = ~np.eye(nb, dtype=bool)[None]
    transformed_slope = np.divide(
        dlam, dE, out=np.zeros_like(dlam), where=offdiag)
    df_eigen = velocity_eigen * transformed_slope[:, None]
    df_global = np.einsum(
        "bim,bamn,bjn->baij", vectors[None], df_eigen,
        vectors[None].conj(), optimize=True)

    got, unsafe = velocity_from_transformed_derivative(
        jnp.asarray(df_global), jnp.asarray(vectors[None]),
        jnp.asarray(transformed), jnp.asarray(energies),
        degeneracy_tolerance_ry=1.0e-10)
    assert not np.any(np.asarray(unsafe))
    np.testing.assert_allclose(
        np.asarray(got), velocity_eigen, rtol=2.0e-12, atol=2.0e-12)


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

    monkeypatch.setattr(
        "psp.get_dipole_mtxels.check_dipole_provenance",
        lambda *_a, **_k: True)

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

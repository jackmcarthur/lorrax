"""Small independent oracles for the no-wavefunction QSGW head kernels."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from common.chi_from_dipole import compute_S_omega
from gw.qsgw_head import (
    _structured_delta_kernel,
    assemble_head_manifold,
    head_s_tensor_sharded,
    reduced_covector_to_cartesian,
    rotate_velocity_active_to_qp,
    rotate_velocity_to_qp,
)

jax.config.update("jax_enable_x64", True)


def _mesh():
    devices = np.asarray(jax.devices())
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _haar(rng, n):
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    phase = np.diag(r)
    return q * (phase / np.abs(phase))[None, :]


def test_reduced_covector_conversion_uses_row_basis_convention():
    B = np.asarray(
        [
            [1.8, 0.2, -0.1],
            [0.4, 1.3, 0.3],
            [-0.2, 0.1, 0.9],
        ]
    )
    reduced = np.arange(3 * 2 * 2).reshape(3, 2, 2) / 7.0
    got = np.asarray(reduced_covector_to_cartesian(reduced, B))
    ref = np.einsum("ij,jab->iab", np.linalg.inv(B), reduced)
    wrong_transpose = np.einsum("ij,jab->iab", np.linalg.inv(B).T, reduced)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=2e-15)
    assert np.max(np.abs(got - wrong_transpose)) > 1e-2


def test_distributed_velocity_rotation_matches_explicit_u_dagger_v_u():
    rng = np.random.default_rng(551)
    nk, nb = 3, 8
    v = rng.standard_normal((3, nk, nb, nb)) + 1j * rng.standard_normal((3, nk, nb, nb))
    U = np.stack([_haar(rng, nb) for _ in range(nk)])
    got = np.asarray(
        rotate_velocity_to_qp(jnp.asarray(v), jnp.asarray(U), mesh=_mesh())
    )
    ref = np.einsum("kmp,akmn,knq->akpq", np.conj(U), v, U, optimize=True)
    np.testing.assert_allclose(got, ref, rtol=3e-15, atol=3e-15)
    # Negative control: U v U-dagger shares norms but is not this basis map.
    wrong = np.einsum("kpm,akmn,kqn->akpq", U, v, np.conj(U), optimize=True)
    assert np.max(np.abs(got - wrong)) > 1e-2


def test_sharded_s_tensor_matches_legacy_dft_formula():
    rng = np.random.default_rng(813)
    nk, nb, nb_pad, nocc = 4, 7, 8, 3
    energies_logical = np.sort(rng.uniform(-0.8, 1.4, (nk, nb)), axis=1)
    occ_logical = np.zeros((nk, nb))
    occ_logical[:, :nocc] = 1.0
    velocity_logical = rng.standard_normal((3, nk, nb, nb)) + 1j * rng.standard_normal(
        (3, nk, nb, nb)
    )
    energies = np.pad(
        energies_logical, ((0, 0), (0, nb_pad - nb)), constant_values=1.0e6
    )
    occ = np.pad(occ_logical, ((0, 0), (0, nb_pad - nb)))
    velocity = np.pad(
        velocity_logical, ((0, 0), (0, 0), (0, nb_pad - nb), (0, nb_pad - nb))
    )
    omegas = np.asarray([0.0 + 0.0j, 0.0 + 0.61j])
    volume = 83.5
    got = np.asarray(
        head_s_tensor_sharded(
            jnp.asarray(velocity),
            jnp.asarray(energies),
            jnp.asarray(occ),
            jnp.asarray(omegas),
            mesh=_mesh(),
            nocc=nocc,
            nb_logical=nb,
            cell_volume=volume,
            nk_tot=nk,
            nspin=1,
            nspinor=1,
        )
    )
    delta_e = energies_logical[:, :, None] - energies_logical[:, None, :]
    ref = np.asarray(
        compute_S_omega(
            jnp.asarray(velocity_logical),
            jnp.asarray(delta_e),
            jnp.asarray(occ_logical),
            volume,
            nk,
            1,
            1,
            jnp.asarray(omegas),
        )
    )
    np.testing.assert_allclose(got, ref, rtol=4e-15, atol=4e-15)


def test_head_manifold_embedding_preserves_inactive_identity_and_cross_blocks():
    rng = np.random.default_rng(19)
    nk, na, nf = 2, 3, 6
    d = rng.standard_normal((nk, na, na)) + 1j * rng.standard_normal((nk, na, na))
    U = np.stack([_haar(rng, na) for _ in range(nk)])
    d_full, U_full = assemble_head_manifold(
        jnp.asarray(d), jnp.asarray(U), nb_storage=nf, mesh=_mesh()
    )
    d_full, U_full = np.asarray(d_full), np.asarray(U_full)
    np.testing.assert_array_equal(d_full[:, :na, :na], d)
    np.testing.assert_array_equal(d_full[:, na:, :], 0.0)
    np.testing.assert_array_equal(d_full[:, :, na:], 0.0)
    np.testing.assert_array_equal(U_full[:, :na, :na], U)
    np.testing.assert_array_equal(
        U_full[:, na:, na:], np.broadcast_to(np.eye(nf - na), (nk, nf - na, nf - na))
    )
    np.testing.assert_array_equal(U_full[:, :na, na:], 0.0)
    np.testing.assert_array_equal(U_full[:, na:, :na], 0.0)


def test_s_tensor_ignores_mesh_padding_bands():
    rng = np.random.default_rng(71)
    nk, nb, nb_pad, nocc = 2, 5, 8, 2
    e = np.sort(rng.normal(size=(nk, nb_pad)), axis=1)
    f = np.zeros_like(e)
    f[:, :nocc] = 1.0
    v = rng.normal(size=(3, nk, nb_pad, nb_pad)) + 1j * rng.normal(
        size=(3, nk, nb_pad, nb_pad)
    )
    # Deliberately enormous pad entries: a logical-manifold mask must make
    # them exactly inert rather than merely small.
    v[:, :, nb:, :] *= 1e9
    v[:, :, :, nb:] *= 1e9
    kw = dict(
        mesh=_mesh(),
        nocc=nocc,
        nb_logical=nb,
        cell_volume=31.0,
        nk_tot=nk,
        nspin=1,
        nspinor=1,
    )
    got = np.asarray(head_s_tensor_sharded(v, e, f, [0.2j], **kw))
    delta_e = e[:, :nb, None] - e[:, None, :nb]
    ref = np.asarray(
        compute_S_omega(
            jnp.asarray(v[:, :, :nb, :nb]),
            jnp.asarray(delta_e),
            jnp.asarray(f[:, :nb]),
            31.0,
            nk,
            1,
            1,
            jnp.asarray([0.2j]),
        )
    )
    # The padded contraction contains explicit zero terms and XLA may group
    # its reduction differently; inert means roundoff parity, not byte parity.
    np.testing.assert_allclose(got, ref, rtol=3e-15, atol=3e-15)


def test_structured_covariant_kernel_matches_dense_commutator():
    rng = np.random.default_rng(410)
    nk, nb, na = 3, 8, 4
    raw_a = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    A = raw_a + np.swapaxes(raw_a.conj(), -1, -2)
    raw_active = rng.normal(size=(nk, na, na)) + 1j * rng.normal(size=(nk, na, na))
    active = raw_active + np.swapaxes(raw_active.conj(), -1, -2)
    delta = np.zeros((nk, nb, nb), dtype=np.complex128)
    delta[:, :na, :na] = active
    delta[:, np.arange(na, nb), np.arange(na, nb)] = rng.normal(size=(nk, nb - na))
    partial = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))

    got = np.asarray(
        _structured_delta_kernel(_mesh(), na)(
            jnp.asarray(delta), jnp.asarray(A), jnp.asarray(partial)
        )
    )
    ref = partial - 1j * (
        np.einsum("akim,kmj->akij", A, delta, optimize=True)
        - np.einsum("kim,akmj->akij", delta, A, optimize=True)
    )
    np.testing.assert_allclose(got, ref, rtol=3e-14, atol=3e-14)


def test_active_rotation_matches_dense_block_diagonal_rotation():
    rng = np.random.default_rng(411)
    nk, nb, na = 2, 8, 4
    velocity = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    U = np.stack([_haar(rng, na) for _ in range(nk)])

    got = np.asarray(
        rotate_velocity_active_to_qp(
            jnp.asarray(velocity), jnp.asarray(U), mesh=_mesh()
        )
    )
    U_full = np.broadcast_to(np.eye(nb, dtype=np.complex128), (nk, nb, nb)).copy()
    U_full[:, :na, :na] = U
    ref = np.einsum(
        "kmi,akmn,knj->akij", U_full.conj(), velocity, U_full, optimize=True
    )
    np.testing.assert_allclose(got, ref, rtol=3e-14, atol=3e-14)


def test_s_tensor_uses_k_dependent_occupations_not_band_cut():
    rng = np.random.default_rng(412)
    nk, nb, logical = 2, 8, 5
    energies = np.broadcast_to(np.arange(nb, dtype=np.float64), (nk, nb)).copy()
    energies[1] += 0.17
    occupations = np.zeros((nk, nb), dtype=np.float64)
    occupations[0, [0, 2]] = 1.0
    occupations[1, [1, 3]] = 1.0
    velocity = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    omega = 0.31j
    volume = 29.0

    got = np.asarray(
        head_s_tensor_sharded(
            jnp.asarray(velocity),
            jnp.asarray(energies),
            jnp.asarray(occupations),
            [omega],
            mesh=_mesh(),
            nocc=2,
            nb_logical=logical,
            cell_volume=volume,
            nk_tot=nk,
            nspin=1,
            nspinor=1,
        )
    )[0]
    ref = np.zeros((3, 3), dtype=np.complex128)
    prefactor = 4.0 / (volume * nk)
    for k in range(nk):
        for c in range(logical):
            for v in range(logical):
                de = energies[k, c] - energies[k, v]
                fd = occupations[k, v] - occupations[k, c]
                if de <= 0.0:
                    continue
                weight = prefactor * fd / (de * (omega * omega - de * de))
                ref += weight * np.outer(
                    velocity[:, k, c, v].conj(), velocity[:, k, c, v]
                )
    np.testing.assert_allclose(got, ref, rtol=5e-14, atol=5e-14)

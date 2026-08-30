"""Ordinary VNL uses atom-local couplings without a total_R-square carrier."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from psp import vnl_ops
from psp.dft_operators import (
    HamiltonianK,
    apply_H_k_from_G,
    build_matrix_k,
)
from solvers.sternheimer_solve import SternheimerOp


def _complex(rng, shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)


def _fixture(nspinor: int, seed: int = 13):
    rng = np.random.default_rng(seed + nspinor)
    channels = []
    blocks = []
    offset = 0
    for ich, (width, natoms) in enumerate(((2, 2), (3, 1))):
        raw = _complex(rng, (2, 2, width, width))
        E = raw + np.conj(raw.transpose(1, 0, 3, 2))
        channels.append(vnl_ops.ChannelMeta(
            l=ich, nbeta=width, msize=1, R=width,
            tau=np.zeros((natoms, 3)), E=E,
            beta_table_start=0, natoms=natoms))
        for _ in range(natoms):
            blocks.append((offset, offset + width, ich))
            offset += width
    couplings = vnl_ops._build_projector_couplings(
        channels, tuple(blocks), total_R=offset, nspinor=nspinor)

    dense = np.zeros(
        (nspinor, nspinor, offset, offset), dtype=np.complex128)
    for start, stop, ich in blocks:
        dense[:, :, start:stop, start:stop] = (
            channels[ich].E[:nspinor, :nspinor])
    return rng, couplings, dense


@pytest.mark.parametrize("nspinor", [1, 2])
def test_apply_matrix_diagonal_and_velocity_match_dense_oracle(nspinor):
    rng, couplings, dense = _fixture(nspinor)
    nband, nrow, nG = 4, dense.shape[-1], 11
    psi = jnp.asarray(_complex(rng, (nband, nspinor, nG)))
    Z = jnp.asarray(_complex(rng, (nrow, nG)))
    dZ = jnp.asarray(_complex(rng, (3, nrow, nG)))

    P = np.einsum("RG,nsG->Rsn", np.conj(Z), psi, optimize=True)
    D = np.einsum("stRQ,Qtn->Rsn", dense, P, optimize=True)
    expected_apply = np.einsum("RG,Rsn->nsG", Z, D, optimize=True)
    expected_matrix = np.einsum(
        "Rsm,Rsn->mn", np.conj(P), D, optimize=True)
    expected_diag = np.real(np.einsum(
        "RG,ssRQ,QG->G", np.conj(Z), dense, Z, optimize=True))

    dP = np.einsum("aRG,nsG->aRsn", np.conj(dZ), psi, optimize=True)
    dD = np.einsum("stRQ,aQtn->aRsn", dense, dP, optimize=True)
    expected_velocity_ket = (
        np.einsum("aRG,Rsn->ansG", dZ, D, optimize=True)
        + np.einsum("RG,aRsn->ansG", Z, dD, optimize=True))
    expected_velocity_matrix = np.einsum(
        "msG,ansG->amn", np.conj(psi), expected_velocity_ket,
        optimize=True)

    np.testing.assert_allclose(
        vnl_ops.apply_vnl(psi, Z, couplings), expected_apply,
        rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(
        vnl_ops.vnl_matrix(psi, Z, couplings), expected_matrix,
        rtol=5e-13, atol=5e-13)
    np.testing.assert_allclose(
        vnl_ops.vnl_diagonal(Z, couplings), expected_diag,
        rtol=5e-13, atol=5e-13)
    velocity_ket = vnl_ops.apply_vnl_velocity_to_ket(
        psi, Z, dZ, couplings)
    np.testing.assert_allclose(
        velocity_ket, expected_velocity_ket, rtol=8e-13, atol=8e-13)
    np.testing.assert_allclose(
        vnl_ops.vnl_velocity_matrix(psi, Z, dZ, couplings),
        expected_velocity_matrix, rtol=8e-13, atol=8e-13)

    # The finite-q/general endpoint contracts the velocity-applied ket with a
    # different bra.  Pin its conjugation separately from the q=0 shortcut.
    other_bra = jnp.asarray(_complex(rng, (3, nspinor, nG)))
    got_finite_q = jnp.einsum(
        "msG,ansG->amn", jnp.conj(other_bra), velocity_ket,
        optimize=True)
    want_finite_q = np.einsum(
        "msG,ansG->amn", np.conj(other_bra), expected_velocity_ket,
        optimize=True)
    np.testing.assert_allclose(
        got_finite_q, want_finite_q, rtol=8e-13, atol=8e-13)


@pytest.mark.parametrize("nspinor", [1, 2])
def test_empty_projector_set_is_exact_zero(nspinor):
    couplings = vnl_ops.VNLProjectorCouplings(
        E_rows=jnp.zeros((nspinor, nspinor, 0, 1), jnp.complex128),
        partner_rows=jnp.zeros((0, 1), jnp.int32))
    psi = jnp.ones((2, nspinor, 5), jnp.complex128)
    Z = jnp.zeros((0, 5), jnp.complex128)
    dZ = jnp.zeros((3, 0, 5), jnp.complex128)
    np.testing.assert_array_equal(vnl_ops.apply_vnl(psi, Z, couplings), 0.0)
    np.testing.assert_array_equal(vnl_ops.vnl_matrix(psi, Z, couplings), 0.0)
    np.testing.assert_array_equal(vnl_ops.vnl_diagonal(Z, couplings), 0.0)
    np.testing.assert_array_equal(
        vnl_ops.apply_vnl_velocity_to_ket(psi, Z, dZ, couplings), 0.0)


@pytest.mark.parametrize("nspinor", [1, 2])
def test_hamiltonian_consumers_delegate_to_the_compact_vnl_owner(nspinor):
    rng, couplings, _ = _fixture(nspinor)
    nband, nG = 3, 8
    psi = jnp.asarray(_complex(rng, (nband, nspinor, nG)))
    Z = jnp.asarray(_complex(rng, (7, nG)))
    grid = (2, 2, 2)
    coordinates = np.asarray(list(np.ndindex(grid)), dtype=np.int32)
    Gx, Gy, Gz = (jnp.asarray(coordinates[:, axis]) for axis in range(3))
    zero_T = jnp.zeros((nG,), jnp.float64)
    zero_V = jnp.zeros(grid, jnp.float64)
    mask = jnp.ones((nG,), jnp.bool_)

    expected_ket = vnl_ops.apply_vnl(psi, Z, couplings)
    actual_ket = apply_H_k_from_G(
        psi, zero_T, zero_V, Gx, Gy, Gz, Z, couplings, mask)
    np.testing.assert_allclose(actual_ket, expected_ket, rtol=5e-13, atol=5e-13)

    psi_box = jnp.zeros((nband, nspinor, *grid), jnp.complex128)
    psi_box = psi_box.at[:, :, Gx, Gy, Gz].set(psi)
    actual_matrix = build_matrix_k(
        psi_box, zero_T, zero_V, Gx, Gy, Gz, Z, couplings, mask)
    expected_matrix = vnl_ops.vnl_matrix(psi, Z, couplings)
    np.testing.assert_allclose(
        actual_matrix, expected_matrix, rtol=5e-13, atol=5e-13)


def test_storage_is_linear_in_atoms_and_lowering_has_no_total_r_square_D():
    nspinor, width, natoms = 2, 3, 1000
    E = np.zeros((2, 2, width, width), dtype=np.complex128)
    channel = vnl_ops.ChannelMeta(
        l=1, nbeta=1, msize=width, R=width,
        tau=np.zeros((natoms, 3)), E=E,
        beta_table_start=0, natoms=natoms)
    blocks = tuple(
        (a * width, (a + 1) * width, 0) for a in range(natoms))
    total_R = natoms * width
    couplings = vnl_ops._build_projector_couplings(
        [channel], blocks, total_R=total_R, nspinor=nspinor)

    assert couplings.E_rows.shape == (2, 2, total_R, width)
    assert couplings.partner_rows.shape == (total_R, width)
    dense_bytes = 16 * nspinor ** 2 * total_R ** 2
    compact_bytes = total_R * width * (16 * nspinor ** 2 + 4)
    actual_bytes = (
        couplings.E_rows.size * couplings.E_rows.dtype.itemsize
        + couplings.partner_rows.size * couplings.partner_rows.dtype.itemsize)
    assert actual_bytes == compact_bytes
    assert compact_bytes < dense_bytes // 100
    assert "E_super" not in {field.name for field in dataclasses.fields(
        vnl_ops.VNLSetup)}
    assert "E_super" not in {field.name for field in dataclasses.fields(
        vnl_ops.VNLKData)}
    assert "vnl_couplings" in {
        field.name for field in dataclasses.fields(HamiltonianK)}
    assert "vnl_E" not in {
        field.name for field in dataclasses.fields(HamiltonianK)}
    assert "vnl_couplings" in SternheimerOp.__slots__
    assert "vnl_E" not in SternheimerOp.__slots__

    # Optimized lowering is the executable structure: it may contain the
    # bounded (spin,spin,R,width) carrier, never (spin,spin,R,R).
    small_R, nG, nband = 7, 9, 2
    _, small, _ = _fixture(nspinor)
    ir = vnl_ops.apply_vnl.lower(
        jax.ShapeDtypeStruct((nband, nspinor, nG), jnp.complex128),
        jax.ShapeDtypeStruct((small_R, nG), jnp.complex128),
        jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), small),
    ).as_text()
    assert f"tensor<2x2x{small_R}x{small_R}xcomplex<f64>>" not in ir
    assert f"tensor<2x2x{small_R}x3xcomplex<f64>>" in ir

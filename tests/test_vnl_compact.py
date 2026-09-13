"""Compact SOC blocks retain dense operator physics and skip padded atoms."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from psp.vnl_ops import (compact_vnl_coupling, apply_vnl,
                         projector_coupling_diagonal)
from psp.dft_operators import apply_H_k_from_G, apply_H_k_batched
from psp.ionic_gspace import species_structure_factors


def _fixture(nspin):
    rng = np.random.default_rng(891)
    channels, blocks = [], []
    total = 0
    for width, atoms in ((2, 3), (3, 2)):
        mat = rng.normal(size=(nspin * width, nspin * width)) + 1j * rng.normal(size=(nspin * width, nspin * width))
        mat = mat + mat.conj().T
        E = mat.reshape(nspin, width, nspin, width).transpose(0, 2, 1, 3)
        channels.append(SimpleNamespace(R=width, E=E))
        for _ in range(atoms):
            blocks.append((total, total + width, len(channels)-1))
            total += width
    setup = SimpleNamespace(channels=channels, coupled_row_blocks=tuple(blocks),
                            nspinor=nspin, total_R=total)
    dense = np.zeros((nspin, nspin, total, total), dtype=complex)
    for start, stop, ich in blocks:
        dense[:, :, start:stop, start:stop] = channels[ich].E
    Z = rng.normal(size=(total, 8)) + 1j*rng.normal(size=(total, 8))
    psi = rng.normal(size=(5, nspin, 8)) + 1j*rng.normal(size=(5, nspin, 8))
    return compact_vnl_coupling(setup), jnp.asarray(dense), jnp.asarray(Z), jnp.asarray(psi)


@pytest.mark.parametrize('nspin', [1, 2])
def test_compact_soc_action_diagonal_and_hamiltonian(nspin):
    compact, dense, Z, psi = _fixture(nspin)
    np.testing.assert_allclose(apply_vnl(psi,Z,compact), apply_vnl(psi,Z,dense), rtol=2e-13,atol=2e-12)
    diagonal = jax.jit(projector_coupling_diagonal)
    np.testing.assert_allclose(diagonal(Z,compact), diagonal(Z,dense), rtol=2e-13,atol=2e-12)
    indices = jnp.indices((2,2,2)).reshape(3,-1)
    args = (psi,jnp.arange(8.),jnp.arange(8.).reshape(2,2,2),*indices,Z)
    np.testing.assert_allclose(apply_H_k_from_G(*args,compact,jnp.ones(8,dtype=bool)),
                               apply_H_k_from_G(*args,dense,jnp.ones(8,dtype=bool)),rtol=2e-13,atol=2e-12)
    np.testing.assert_allclose(
        apply_H_k_batched(*args, compact, jnp.ones(8,dtype=bool), vector_batch=3),
        apply_H_k_from_G(*args, dense, jnp.ones(8,dtype=bool)),
        rtol=2e-13, atol=2e-12)
    assert sum(E.size for E in compact.groups) < dense.size


def test_structure_factor_inactive_nan_atoms_and_reverse_mode():
    tau = jnp.asarray([[[.1,.2,.3],[.4,.2,.1],[np.nan]*3],
                       [[.2,.4,.3],[np.nan]*3,[np.nan]*3]])
    counts = jnp.asarray([2,1])
    G = jnp.asarray([[0.,0.,0.],[1.,2.,-1.],[2.,-1.,1.]])
    result = species_structure_factors(tau,counts,G,3)
    reference = jnp.stack([jnp.exp(-2j*jnp.pi*(G @ tau[0,:2].T)).sum(axis=1),
                           jnp.exp(-2j*jnp.pi*(G @ tau[1,:1].T)).sum(axis=1)])
    np.testing.assert_allclose(result,reference,rtol=2e-14,atol=2e-14)
    grad = jax.grad(lambda positions: jnp.real(species_structure_factors(positions,counts,G,3)).sum())(tau)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad[0,2],0)
    np.testing.assert_array_equal(grad[1,1:],0)

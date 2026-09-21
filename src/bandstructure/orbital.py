"""Interpolate a supplied band operator in htransform's Galerkin basis.

This does not differentiate the fitted Hamiltonian: its full-Bloch basis
would require a connection term. The caller supplies and labels the operator,
including any covariant quasiparticle correction, before interpolation.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.fft_helpers import make_flat_k_ifftn
from gw.qsgw_head import rotate_velocity_to_qp
from runtime.padding import pad_axis


def interpolate_band_operator(operator_cart, source_coefficients,
                              path_coefficients, kpath, kgrid, mesh):
    """Return (q_carrier,component,band,band) matrices in the path basis.

    ``operator_cart`` is (component,k,n,n), ``source_coefficients`` is
    (k,n,rank), and ``path_coefficients`` is (q_carrier,rank,band), exactly
    as returned by htransform. Matrix elements use bra n, ket m. Persistent
    Galerkin operators have both rank axes distributed; one P-point q batch
    is reshaped onto the composite mesh before the local band rotation.
    """
    from bandstructure.htransform import build_R_grid_np
    nk, nb, rank = source_coefficients.shape
    if (operator_cart.ndim != 4 or operator_cart.shape[1] != nk
            or operator_cart.shape[-1] != operator_cart.shape[-2]
            or operator_cart.shape[-1] < nb
            or path_coefficients.shape[1] != rank
            or int(np.prod(kgrid)) != nk):
        raise ValueError('Band operator, source basis and path coefficients disagree')
    nq = len(kpath)
    if nq > path_coefficients.shape[0]:
        raise ValueError('Path coefficients do not cover the requested k path')
    coefficients = pad_axis(
        source_coefficients, operator_cart.shape[-1], axis=1).array
    # The shared two-sided distributed contraction is U^H O U. Here
    # U=conj(C), giving C^T O C*, since C[k,n,a]=<B_a|psi_kn>.
    operator_basis = rotate_velocity_to_qp(
        operator_cart, jnp.conj(coefficients), mesh=mesh)
    face = NamedSharding(mesh, P(None, None, 'x', 'y'))
    inverse = make_flat_k_ifftn(
        mesh, tuple(kgrid), P(None, None, None, None, 'x', 'y'), norm='backward')
    operator_R = jax.jit(
        lambda value: inverse(jnp.moveaxis(value, 0, 1)),
        out_shardings=face)(operator_basis)
    del operator_basis
    R = jnp.asarray(build_R_grid_np(kgrid))
    q_matrix = NamedSharding(mesh, P(('x', 'y'), None, None, None))
    q_coeff = NamedSharding(mesh, P(('x', 'y'), None, None))
    step = int(mesh.size)

    @partial(jax.jit, out_shardings=q_matrix)
    def rotate(q, c, lattice_operator):
        phase = jnp.exp(-2j * jnp.pi * (q @ R.T))
        value = jnp.einsum('qk,kamn->qamn', phase, lattice_operator)
        value = jax.lax.with_sharding_constraint(value, face)
        value = jax.lax.with_sharding_constraint(value, q_matrix)
        value = 0.5 * (value + value.swapaxes(-1, -2).conj())
        c = jax.lax.with_sharding_constraint(c, q_coeff)
        return jnp.einsum('qmi,qamn,qnj->qaij', c.conj(), value, c,
                          optimize=True)

    out = []
    for start in range(0, nq, step):
        stop = min(start + step, nq)
        q = jnp.asarray(kpath[start:stop])
        c = path_coefficients[start:stop]
        q = pad_axis(q, step, axis=0).array
        c = pad_axis(c, step, axis=0).array
        out.append(rotate(q, c, operator_R))
    return jnp.concatenate(out, axis=0)

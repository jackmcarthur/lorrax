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

from common import timing
from common.fft_helpers import make_flat_k_ifftn
from common.staged_reshard import concatenate_sharded_axis, face_to_batch_reshard
from gw.qsgw_head import rotate_velocity_to_qp
from runtime.padding import pad_axis, padded_axis


def interpolate_band_operator(operator_cart, source_coefficients,
                              path_coefficients, kpath, kgrid, mesh):
    """Return (q_carrier,component,band,band) matrices in the path basis.

    ``operator_cart`` is (component,k,n,n), ``source_coefficients`` is
    (k,n,rank), and ``path_coefficients`` is (q_carrier,rank,band), exactly
    as returned by htransform. Matrix elements use bra n, ket m. Persistent
    Galerkin operators have both rank axes distributed; one P-point q batch
    is reshaped onto the composite mesh before the local band rotation.
    """
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
    # q slices are local in a band-sharded carrier; slicing the global
    # q-sharded result directly can all-gather the entire dense-grid table.
    n_return = int(path_coefficients.shape[-1])
    carrier = padded_axis(n_return, mesh, name="orbital path bands",
                          spec=P(None, None, ('x', 'y')), axis=2).carrier
    path_coefficients = jax.jit(
        lambda c: pad_axis(c, carrier, axis=2).array,
        out_shardings=NamedSharding(mesh, P(None, None, ('x', 'y'))))(
            path_coefficients)
    parts = []
    for a in range(operator_cart.shape[0]):
        with timing.fenced_section(f"orbital component {a}", announce=True):
            part = _interpolate_component(
                operator_cart[a:a+1], source_coefficients,
                path_coefficients, kpath, kgrid, mesh, n_return)
            parts.append(jax.block_until_ready(part))
    return jnp.concatenate(parts, axis=1)


def _interpolate_component(operator_cart, source_coefficients,
                           path_coefficients, kpath, kgrid, mesh, n_return):
    from bandstructure.fh_interp import build_R_grid_np
    nk, nb, rank = source_coefficients.shape
    nq = len(kpath)
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
    exchange = face_to_batch_reshard(mesh)

    @partial(jax.jit, out_shardings=q_matrix)
    def rotate(q, c, lattice_operator):
        phase = jnp.exp(-2j * jnp.pi * (q @ R.T))
        value = jnp.einsum('qk,kamn->qamn', phase, lattice_operator)
        value = jax.lax.with_sharding_constraint(value, face)
        # Use the existing two all-to-all transfer; a generic face -> q
        # constraint can make XLA replicate this rank-squared buffer.
        value = exchange(value.reshape((-1, rank, rank))).reshape(value.shape)
        value = 0.5 * (value + value.swapaxes(-1, -2).conj())
        c = jax.lax.with_sharding_constraint(c, q_coeff)[:, :, :n_return]
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
    return concatenate_sharded_axis(out, 0, mesh, q_matrix.spec)

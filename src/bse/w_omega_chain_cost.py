"""Comparison-only reduced solves for full-port omega-chain cost measurements.

The production evaluator is unchanged. This module avoids its dense (m*p)^2
host matrix when measuring a chain whose probe width is the full port count.
"""
from __future__ import annotations

import numpy as np


def orthonormalize_seed(seed, sharding):
    """TSQR of the excitation-space seed without forming its normal equations.

    Parameters
    ----------
    seed : jax.Array
        Complex seed block, shape (p, c, v, k), with the supplied NamedSharding.
        Every local excitation tile must contain at least p rows.
    sharding : jax.sharding.NamedSharding
        Existing pair-basis layout. Only the small p-by-p local R factors are
        gathered across its named mesh axes; the excitation-space Q stays tiled.

    Returns
    -------
    q : jax.Array
        Orthonormal block with the same shape and sharding as seed.
    r : numpy.ndarray
        Replicated host p-by-p factor satisfying seed = q @ r, in column form.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    axes = tuple(sharding.mesh.axis_names)
    count = sharding.mesh.size

    def local(block):
        p = block.shape[0]
        matrix = block.reshape(p, -1).T
        if matrix.shape[0] < p:
            raise ValueError('TSQR cost probe requires at least p local pair rows')
        q_local, r_local = jnp.linalg.qr(matrix, mode='reduced')
        small = jax.lax.all_gather(r_local, axes, axis=0, tiled=False).reshape(count*p, p)
        q_small, r = jnp.linalg.qr(small, mode='reduced')
        start = jax.lax.axis_index(axes) * p
        section = jax.lax.dynamic_slice_in_dim(q_small, start, p, axis=0)
        q = (q_local @ section).T.reshape(block.shape)
        return q, r

    mapped = shard_map(local, mesh=sharding.mesh, in_specs=sharding.spec,
                       out_specs=(sharding.spec, P()), check_vma=False)
    q, r = jax.jit(mapped)(seed)
    return q, np.asarray(jax.device_get(r))


def solve_chain_resolvent(alpha, beta, r0, z, *, m_use=None):
    """Solve (z² I - T) C = [R0; 0] by block elimination.

    T is the Hermitian block-tridiagonal Lanczos operator used by
    ``w_omega_chain.eval_w_omega_chain``. Complex z² off the real axis makes
    every principal shifted block nonsingular; no inter-block pivot is needed.

    Parameters
    ----------
    alpha : numpy.ndarray
        Replicated host diagonal blocks, shape (m, p, p), in Ry².
    beta : numpy.ndarray
        Replicated host subdiagonal blocks, shape (m, p, p), in Ry².
        The final residual block is unused.
    r0 : numpy.ndarray
        Replicated host seed factor, shape (p, p).
    z : complex
        Frequency in Ry. Its square must have nonzero imaginary part.
    m_use : int, optional
        Number of chain blocks consumed, at most m.

    Returns
    -------
    numpy.ndarray
        Coefficients of shape (m_use, p, p), replicated on host. Storage is
        O(m_use*p²), including elimination factors, rather than O((m_use*p)²).
    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    r0 = np.asarray(r0)
    m = len(alpha) if m_use is None else int(m_use)
    p = r0.shape[0]
    if not 1 <= m <= len(alpha):
        raise ValueError('m_use must lie within the stored chain')
    if alpha.shape[1:] != (p, p) or beta.shape != alpha.shape or r0.shape != (p, p):
        raise ValueError('inconsistent chain block shapes')
    z2 = complex(z) ** 2
    if z2.imag == 0:
        raise ValueError('cost probe requires z squared off the real spectrum')
    eye = np.eye(p, dtype=np.complex128)
    upper_solutions = np.empty((max(0, m-1), p, p), dtype=np.complex128)
    values = np.empty((m, p, p), dtype=np.complex128)
    diagonal = z2 * eye - alpha[0]
    rhs = r0
    for j in range(m):
        if j + 1 < m:
            upper = -beta[j].conj().T
            solved = np.linalg.solve(diagonal, np.concatenate((upper, rhs), axis=1))
            upper_solutions[j] = solved[:, :p]
            values[j] = solved[:, p:]
            lower = -beta[j]
            diagonal = z2 * eye - alpha[j+1] - lower @ upper_solutions[j]
            rhs = -lower @ values[j]
        else:
            values[j] = np.linalg.solve(diagonal, rhs)
    for j in range(m-2, -1, -1):
        values[j] -= upper_solutions[j] @ values[j+1]
    return values

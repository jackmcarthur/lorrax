"""Experimental fixed-dictionary Ritz comparisons, outside production dispatch.

The dictionary contains X_j=(s_j-T)^-1 B q_j and optional infinity states
B q_j. G=X.H X, H=X.H T X and O=B.H X use the physical W convention of
shared_pole_constructor. Only the retained subspace changes in this module.
"""
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from common.shard_map import shard_map
from gw.shared_pole_constructor import _hermitian, reduce_shared_pole_pencil


def fixed_gram_ritz(pencil, active_columns, requested, *, eigh, matmul, gates):
    """Retain exactly the largest requested normalized-Gram eigendirections.

    Parameters
    ----------
    pencil : tuple of arrays
        G/H [b,R,R] and O [b,n,R], complex128, in the caller's local or
        face layout. Units and normalization follow the production owner.
    active_columns : array
        Replicated bool [b,R], excluding only inert carrier padding.
    requested : int
        Fixed model order chosen before inspecting Ritz poles.
    eigh, matmul : callable
        Resolved production dense operations, preserving the caller's layout.
    gates : mapping
        Production gate table. Its Gram cut is replaced by the spectral gap
        enclosing the requested rank; the positivity guard stays 1e-12.

    Returns
    -------
    model, diagnostics, coefficients : tuple
        Production output arrays and predicates. ``exact_order`` and
        ``requested_gram_positive`` must pass before admitting any export.

    This is Rayleigh--Ritz on the top-K equilibrated dictionary Gram space,
    not an iterative interpolant. K determines the cut, never vice versa.
    """
    g, _, _ = pencil
    if not 0 < requested <= g.shape[-1]:
        raise ValueError('requested order must fit the dictionary')
    diagonal = jnp.real(jnp.diagonal(g, axis1=-2, axis2=-1))
    scale = jnp.where(active_columns,
                      1 / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1)), 0)
    normalized = scale[:, :, None] * g * scale[:, None, :]
    spectrum, vectors = eigh(_hermitian(normalized))
    high = spectrum[:, -requested] / spectrum[:, -1]
    low = (spectrum[:, -requested-1] / spectrum[:, -1]
           if requested < g.shape[-1] else jnp.zeros_like(high))
    # The production predicate broadcasts one threshold over [batch,R].
    threshold = ((low + high) * .5)[:, None]
    fixed_gates = deepcopy(gates)
    fixed_gates['normalized_gram_keep']['threshold'] = threshold
    calls = 0

    def reuse_gram(matrix):
        nonlocal calls
        calls += 1
        return (spectrum, vectors) if calls == 1 else eigh(matrix)

    model, diagnostics, coefficients = reduce_shared_pole_pencil(
        pencil, active_columns, eigh=reuse_gram, matmul=matmul,
        gates=fixed_gates)
    diagnostics.update(
        exact_order=diagnostics['retained_rank'] == requested,
        requested_gram_positive=jnp.isfinite(high) & (high >= 1e-12)
        & (low < high),
        requested_gram_min_relative=high,
        exact_order_cut=threshold[:, 0])
    return model, diagnostics, coefficients


def tiled_dictionary_projection_pivots(gram, *, k_max, mesh):
    """Run ARK's dictionary projection recurrence with face-local storage.

    Parameters
    ----------
    gram : complex array, shape (R, R)
        Unscaled physical dictionary Gram, sharded P('x', 'y'). R must
        be divisible by both mesh axes, with inert dictionary padding zero.
    k_max : int
        Requested pivot count. Padding the workspace never changes it.
    mesh : jax.sharding.Mesh
        Named x/y mesh owning the input face and the factor workspace.

    Returns
    -------
    indices, scores, valid : arrays, shape (k_max,)
        Replicated pivot indices, squared projection defects, and positive
        finite pivot predicates. No threshold chooses the requested order.

    This is the recurrence in ARK's ``dictionary_projection_pivots``:
    L[:,m] = (G[:,j] - L @ conj(L[j,:])) / sqrt(residual[j]), followed
    by residual -= abs(L[:,m])**2. Only the distributed extraction differs:
    masked local slices and psum move one row/column, never the whole L.
    Physical column normalization and lowest-index tie breaking are retained.
    """
    n = gram.shape[0]
    if gram.shape != (n, n) or not 0 < k_max <= n:
        raise ValueError('requested pivot count must fit the square dictionary')
    nx, ny = int(mesh.shape['x']), int(mesh.shape['y'])
    if n % nx or n % ny:
        raise ValueError('dictionary carrier must be divisible by named axes')
    k_carrier = ((k_max + ny - 1) // ny) * ny

    @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'),
             out_specs=(P(), P(), P()), check_vma=False)
    def select(local_gram):
        rows = jnp.arange(n // nx) + jax.lax.axis_index('x') * (n // nx)
        columns = jnp.arange(n // ny) + jax.lax.axis_index('y') * (n // ny)
        factor_columns = (jnp.arange(k_carrier // ny)
                          + jax.lax.axis_index('y') * (k_carrier // ny))
        diagonal = jax.lax.psum(jnp.sum(jnp.where(
            rows[:, None] == columns[None, :], local_gram, 0), axis=1), 'y')
        initial = (jnp.zeros((n // nx, k_carrier // ny), local_gram.dtype),
                   jnp.real(diagonal), jnp.zeros(n // nx, bool))

        def step(carry, m):
            factors, residual, selected = carry
            eligible = jnp.where(selected, -jnp.inf, residual)
            score = jax.lax.pmax(jnp.max(eligible), 'x')
            pivot = jax.lax.pmin(jnp.min(jnp.where(
                (~selected) & (eligible == score), rows, n)), 'x')
            valid = jnp.isfinite(score) & (score > 0) & (pivot < n)
            pivot_row = jax.lax.psum(jnp.sum(jnp.where(
                rows[:, None] == pivot, factors, 0), axis=0), 'x')
            gram_column = jax.lax.psum(jnp.sum(jnp.where(
                columns[None, :] == pivot, local_gram, 0), axis=1), 'y')
            projected = jax.lax.psum(jnp.sum(
                factors * pivot_row.conj()[None, :], axis=1), 'y')
            column = (gram_column - projected) / jnp.sqrt(
                jnp.where(valid, score, 1))
            factors = jnp.where(factor_columns[None, :] == m,
                                column[:, None], factors)
            residual = residual - jnp.abs(column)**2
            selected = selected | (rows == pivot)
            return (factors, residual, selected), (pivot, score, valid)

        return jax.lax.scan(step, initial,
                            jnp.arange(k_max, dtype=jnp.int32))[1]

    return select(gram)

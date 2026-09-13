"""Experimental fixed-dictionary Ritz comparisons, outside production dispatch.

The dictionary contains X_j=(s_j-T)^-1 B q_j and optional infinity states
B q_j. G=X.H X, H=X.H T X and O=B.H X use the physical W convention of
shared_pole_constructor. Only the retained subspace changes in this module.
"""
from copy import deepcopy

import jax.numpy as jnp

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

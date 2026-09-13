"""Shared scale-relative rank discovery for iterative eigensolvers."""
import jax.numpy as jnp

_RANK_DROP_RTOL = 1e-10
_TINY = 1e-300

def _rank_whitener(S):
    """One rank cutoff and coefficient construction for Davidson correction blocks."""
    e, U = jnp.linalg.eigh(S)                 # ascending, e real
    e = e[::-1]
    U = U[:, ::-1]                            # descending
    thresh = _RANK_DROP_RTOL * jnp.maximum(e[0], 0.0)
    keep = e > thresh
    rank = jnp.sum(keep.astype(jnp.int32))
    inv_sqrt = jnp.where(keep, 1.0 / jnp.sqrt(jnp.maximum(e, _TINY)), 0.0)
    M = U * inv_sqrt[None, :].astype(U.dtype)     # M[i, m] = U[i,m]·e_m^-1/2
    return M, rank


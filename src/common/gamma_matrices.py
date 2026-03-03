import jax.numpy as jnp

"""Dirac gamma matrices represented as JAX arrays.

The matrices are provided in the standard Dirac representation
and stored as ``jax.numpy`` arrays.
"""

# Pauli matrices
sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

# Standard Dirac gamma matrices (4x4)
# JM: actually I replace gamma0-3 with gamma0*gamma0-3, so that I can use psidag = conj(psi) rather than psibar = conj(psi) gamma0

gamma0 = jnp.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=jnp.complex128)

gamma1 = jnp.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]], dtype=jnp.complex128)

gamma2 = jnp.array([[0, 0, 0, -1j],
                   [0, 0, 1j, 0],
                   [0, -1j, 0, 0],
                   [1j, 0, 0, 0]], dtype=jnp.complex128)

gamma3 = jnp.array([[0, 0, 1, 0],
                   [0, 0, 0, -1],
                   [1, 0, 0, 0],
                   [0, -1, 0, 0]], dtype=jnp.complex128)

# gamma^5 = i gamma^0 gamma^1 gamma^2 gamma^3
gamma5 = jnp.array([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0]], dtype=jnp.complex128)

def _to_sparse(mat):
    """Return row indices, column indices, and values of nonzero entries."""
    r, c = jnp.nonzero(mat)
    return r, c, mat[r, c]


gammas = [gamma0, gamma1, gamma2, gamma3]
gammas_sparse = [_to_sparse(g) for g in gammas]

__all__ = [
    "sigma_x", "sigma_y", "sigma_z",
    "gamma0", "gamma1", "gamma2", "gamma3", "gamma5",
]

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
    # All gamma matrices in this module have exactly four non-zero entries.
    # Keep the size static so this module is safe to import while JAX is
    # tracing a caller.
    r, c = jnp.nonzero(mat, size=4)
    return r, c, mat[r, c]


gammas = [gamma0, gamma1, gamma2, gamma3]
gammas_sparse = [_to_sparse(g) for g in gammas]

# γ̃^μ are signed/permuted identity matrices.  These arrays encode
# left multiplication by rows: (γψ)_a = phase[a] * ψ[perm[a]].
gamma_left_perms = jnp.array([
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [3, 2, 1, 0],
    [2, 3, 0, 1],
], dtype=jnp.int32)
gamma_left_phases = jnp.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [-1j, 1j, -1j, 1j],
    [1, -1, 1, -1],
], dtype=jnp.complex128)

# Right multiplication by columns: (Gγ)_d = phase[d] * G[..., perm[d]].
gamma_right_perms = jnp.array([
    [0, 1, 2, 3],
    [3, 2, 1, 0],
    [3, 2, 1, 0],
    [2, 3, 0, 1],
], dtype=jnp.int32)
gamma_right_phases = jnp.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1j, -1j, 1j, -1j],
    [1, -1, 1, -1],
], dtype=jnp.complex128)

__all__ = [
    "sigma_x", "sigma_y", "sigma_z",
    "gamma0", "gamma1", "gamma2", "gamma3", "gamma5",
    "gammas", "gammas_sparse",
    "gamma_left_perms", "gamma_left_phases",
    "gamma_right_perms", "gamma_right_phases",
]

from .gpu_utils import xp
#import cupy as xp
#import numpy as np

"""Dirac gamma matrices for use with CPU or GPU arrays.

The matrices are provided in the standard Dirac representation
and stored as ``xp`` arrays so code can remain agnostic to
NumPy or CuPy backends.
"""

# Pauli matrices
sigma_x = xp.array([[0, 1], [1, 0]], dtype=xp.complex128)
sigma_y = xp.array([[0, -1j], [1j, 0]], dtype=xp.complex128)
sigma_z = xp.array([[1, 0], [0, -1]], dtype=xp.complex128)

# Standard Dirac gamma matrices (4x4)
# JM: actually I replace gamma0-3 with gamma0*gamma0-3, so that I can use psidag = conj(psi) rather than psibar = conj(psi) gamma0

gamma0 = xp.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=xp.complex128)

gamma1 = xp.array([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0]], dtype=xp.complex128)

gamma2 = xp.array([[0, 0, 0, -1j],
                   [0, 0, 1j, 0],
                   [0, -1j, 0, 0],
                   [1j, 0, 0, 0]], dtype=xp.complex128)

gamma3 = xp.array([[0, 0, 1, 0],
                   [0, 0, 0, -1],
                   [1, 0, 0, 0],
                   [0, -1, 0, 0]], dtype=xp.complex128)

# gamma^5 = i gamma^0 gamma^1 gamma^2 gamma^3
gamma5 = xp.array([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0],
                   [0, 1, 0, 0]], dtype=xp.complex128)

def _to_sparse(mat):
    """Return row indices, column indices, and values of nonzero entries."""
    r, c = xp.nonzero(mat)
    return r, c, mat[r, c]


gammas = [gamma0, gamma1, gamma2, gamma3]
gammas_sparse = [_to_sparse(g) for g in gammas]

__all__ = [
    "sigma_x", "sigma_y", "sigma_z",
    "gamma0", "gamma1", "gamma2", "gamma3", "gamma5",
]

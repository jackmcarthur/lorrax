"""Comparison-only reduced solves for full-port omega-chain cost measurements.

The production evaluator is unchanged. This module avoids its dense (m*p)^2
host matrix when measuring a chain whose probe width is the full port count.
"""
from __future__ import annotations

import numpy as np


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

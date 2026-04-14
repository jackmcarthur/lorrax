"""psp/dft_precond.py — DFT-specific preconditioner and initial guess for Davidson.

These callables plug into solvers.davidson without that module knowing
anything about DFT, plane waves, or k-points.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Preconditioner: diagonal (QE's g_psi convention)
# ═══════════════════════════════════════════════════════════════════════

def make_dft_preconditioner(h_diag: jax.Array):
    """Return a precond_fn(R, eigenvalues) → P for Davidson.

    Uses the QE diagonal preconditioner:
        P_nG = R_nG / max(|h_diag_G − ε_n|, 0.01)
    then normalises each correction vector.
    """
    @jax.jit
    def precond_fn(R, eigenvalues):
        _EPS = 1e-2
        denom = h_diag[None, None, :] - eigenvalues[:, None, None]
        denom = jnp.where(jnp.abs(denom) < _EPS, jnp.sign(denom) * _EPS, denom)
        denom = jnp.where(denom == 0.0, _EPS, denom)
        P = R / denom
        norms = jnp.sqrt(jnp.sum(jnp.abs(P) ** 2, axis=(1, 2)))
        return P / jnp.maximum(norms, 1e-30)[:, None, None]
    return precond_fn


# ═══════════════════════════════════════════════════════════════════════
#  Initial guess: lowest-|k+G|² plane waves
# ═══════════════════════════════════════════════════════════════════════

def make_pw_init(T_diag: jax.Array, n_channels: int, verbose: bool = True):
    """Return an init_fn(apply_H, n_eig) → (X0, HX0) for Davidson.

    Selects the 2×n_eig lowest-|k+G|² plane waves, projects H into that
    subspace, and rotates to the lowest n_eig eigenvectors.

    Parameters
    ----------
    T_diag : (dim,) — kinetic energy diagonal |k+G|² (padding entries ≥ 1e10)
    n_channels : spinor dimension (1 or 2)
    """
    T_np = np.asarray(T_diag)
    nG = len(T_np)
    order = np.argsort(T_np)

    def init_fn(apply_H, n_eig):
        n_pw = min(2 * n_eig, nG)
        n_basis = min(n_pw * n_channels, nG * n_channels)

        rows = np.repeat(np.arange(n_channels), n_pw)[:n_basis]
        cols = np.tile(order[:n_pw], n_channels)[:n_basis]
        V_pw = np.zeros((n_basis, n_channels, nG), dtype=np.complex128)
        V_pw[np.arange(n_basis), rows, cols] = 1.0
        V_pw = jnp.asarray(V_pw)

        HV_pw = apply_H(V_pw)

        Hc = jnp.einsum('msG,nsG->mn', jnp.conj(V_pw), HV_pw, optimize=True)
        Hc = 0.5 * (Hc + Hc.conj().T)
        eigvals, C = jnp.linalg.eigh(Hc)
        C_low = C[:, :n_eig]

        X0 = jnp.einsum('ij,isG->jsG', C_low, V_pw, optimize=True)
        HX0 = jnp.einsum('ij,isG->jsG', C_low, HV_pw, optimize=True)

        if verbose:
            eigs = np.asarray(eigvals[:n_eig])
            print(f"  Initial guess: {n_basis} PWs → {n_eig} vectors, "
                  f"eig [{eigs[0]:.6f}, {eigs[-1]:.6f}] Ry")
        return X0, HX0

    return init_fn

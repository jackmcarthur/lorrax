"""
solvers/davidson.py — Block Davidson iterative eigensolver.

Finds the lowest n_eig eigenvalues of a Hermitian operator H given only
callables for the matvec, preconditioner, and initial guess.
No DFT or physics knowledge — works for any Hermitian eigenproblem
with state vectors of shape (batch, n_channels, dim).

Algorithm (after QE cegterg.f90):
  1. init_fn builds the starting subspace
  2. Each iteration: Gram projection → generalized eigh via Cholesky →
     Ritz vectors → residuals → precond_fn → expand
  3. Fixed block-size expansion (avoids JIT recompilation)
  4. Restart to Ritz vectors when subspace exceeds m_max

Usage
-----
    from solvers.davidson import davidson
    eigenvalues, eigenvectors = davidson(
        apply_H, n_eig=12, precond_fn=precond, init_fn=init,
    )
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  JIT'd subspace projection kernel
# ═══════════════════════════════════════════════════════════════════════

def _generalized_eigh(A, B):
    """Solve A v = λ B v via Cholesky reduction.  JIT-compatible."""
    m = B.shape[0]
    B_reg = B + 1e-12 * jnp.eye(m, dtype=B.dtype)
    L = jnp.linalg.cholesky(B_reg)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.conj().T
    C = 0.5 * (C + C.conj().T)
    eigenvalues, V = jnp.linalg.eigh(C)
    eigenvectors = jnp.linalg.solve(L.conj().T, V)
    return eigenvalues, eigenvectors


@functools.partial(jax.jit, static_argnames=('n_eig',))
def _ritz_and_residuals(V, HV, n_eig):
    """Project → solve → Ritz vectors → residuals.

    Returns (eigenvalues, X, HX, R, res_norms).
    """
    Hc = jnp.einsum('msG,nsG->mn', jnp.conj(V), HV, optimize=True)
    Sc = jnp.einsum('msG,nsG->mn', jnp.conj(V), V, optimize=True)
    Hc = 0.5 * (Hc + Hc.conj().T)
    Sc = 0.5 * (Sc + Sc.conj().T)

    eig_all, C_all = _generalized_eigh(Hc, Sc)
    Lambda = eig_all[:n_eig]
    C_N = C_all[:, :n_eig]

    X = jnp.einsum('mn,msG->nsG', C_N, V, optimize=True)
    HX = jnp.einsum('mn,msG->nsG', C_N, HV, optimize=True)

    R = HX - X * Lambda[:, None, None]
    res_norms = jnp.sqrt(jnp.sum(jnp.abs(R) ** 2, axis=(1, 2)))

    return Lambda, X, HX, R, res_norms


def _default_precond(R, eigenvalues):
    """Identity preconditioner (no-op)."""
    norms = jnp.sqrt(jnp.sum(jnp.abs(R) ** 2, axis=(1, 2)))
    return R / jnp.maximum(norms, 1e-30)[:, None, None]


# ═══════════════════════════════════════════════════════════════════════
#  Warmup
# ═══════════════════════════════════════════════════════════════════════

def warmup_davidson_jit(n_eig: int, dim: int, n_channels: int,
                        m_max: int | None = None):
    """Pre-compile _ritz_and_residuals at all subspace sizes."""
    if m_max is None:
        m_max = 4 * n_eig
    for m in range(n_eig, m_max + n_eig, n_eig):
        V = jnp.zeros((min(m, m_max), n_channels, dim), dtype=jnp.complex128)
        HV = jnp.zeros_like(V)
        _ritz_and_residuals(V, HV, n_eig)


# ═══════════════════════════════════════════════════════════════════════
#  Davidson solver
# ═══════════════════════════════════════════════════════════════════════

def davidson(
    apply_H,
    *,
    n_eig: int,
    precond_fn=None,
    init_fn=None,
    X0: jax.Array | None = None,
    m_max: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Block Davidson iterative eigensolver.

    Parameters
    ----------
    apply_H : (m, n_channels, dim) → (m, n_channels, dim)
        Hermitian matvec.
    n_eig : number of lowest eigenvalues to converge.
    precond_fn : (R, eigenvalues) → P
        Preconditioner: maps residuals (n_eig, n_channels, dim) and
        eigenvalues (n_eig,) to normalised corrections P.
        Default: identity (normalised residuals).
    init_fn : (apply_H, n_eig) → (X0, HX0)
        Initial subspace builder. Returns (n_eig, n_channels, dim) arrays.
        Default: must provide X0 instead.
    X0 : (n_eig, n_channels, dim) — explicit initial vectors (alternative to init_fn).
    m_max : max subspace dimension before restart (default 4 × n_eig).

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n_channels, dim)
    """
    if precond_fn is None:
        precond_fn = _default_precond
    if m_max is None:
        m_max = 4 * n_eig

    # ── initial subspace ──
    if X0 is not None:
        V = jnp.asarray(X0[:n_eig], dtype=jnp.complex128)
        HV = apply_H(V)
    elif init_fn is not None:
        V, HV = init_fn(apply_H, n_eig)
    else:
        raise ValueError("Provide either init_fn or X0")

    if verbose:
        print(f"Davidson: n_eig={n_eig}, dim={V.shape[-1]}, m_max={m_max}")

    eigenvalues = None

    for it in range(1, max_iter + 1):
        # ── GPU: project + solve + Ritz + residual (one JIT) ──
        Lambda, X, HX, R, res = _ritz_and_residuals(V, HV, n_eig)

        # ── precondition (caller-provided, possibly JIT'd) ──
        P = precond_fn(R, Lambda)

        # ── convergence check (CPU) ──
        res_np = np.asarray(res)
        rel_tol = tol * np.maximum(1.0, np.abs(np.asarray(Lambda)))
        conv = res_np < rel_tol
        n_conv = 0
        for i in range(n_eig):
            if conv[i]:
                n_conv = i + 1
            else:
                break

        eigenvalues = np.asarray(Lambda)
        if verbose and (it <= 5 or it % 5 == 0 or n_conv == n_eig):
            print(f"  iter {it:3d}: m={V.shape[0]:3d}  "
                  f"eig[0]={float(Lambda[0]):12.6f}  "
                  f"eig[{n_eig-1}]={float(Lambda[n_eig-1]):12.6f}  "
                  f"res=[{res_np.min():.1e},{res_np.max():.1e}]  "
                  f"conv={n_conv}/{n_eig}")

        if n_conv == n_eig:
            if verbose:
                print(f"  Converged all {n_eig} in {it} iterations.")
            return eigenvalues, np.asarray(X)

        # ── expand subspace ──
        HP = apply_H(P)
        V = jnp.concatenate([V, P], axis=0)
        HV = jnp.concatenate([HV, HP], axis=0)

        # ── restart if too large ──
        if V.shape[0] > m_max:
            if verbose:
                print(f"  iter {it}: restart (m={V.shape[0]} > m_max={m_max})")
            V, HV = X, HX

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations. "
              f"Best: {n_conv}/{n_eig}")
    return eigenvalues, np.asarray(X)

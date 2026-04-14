"""
solvers/davidson.py — Block Davidson iterative eigensolver.

Finds the lowest n_eig eigenvalues of a Hermitian operator H given only
the matvec callable H @ x.  No DFT, no k-points — works for any problem
whose state vectors have shape (batch, n_channels, dim).

Algorithm (after QE cegterg.f90):
  1. Initial guess from diagonal → subspace H projection → rotate
  2. Each iteration: Gram projection → generalized eigh via Cholesky →
     Ritz vectors → residuals → diagonal preconditioner → expand
  3. Fixed block-size expansion (avoids JIT recompilation)
  4. Restart to Ritz vectors when subspace exceeds m_max (default 4×n_eig)

Usage
-----
    from solvers.davidson import davidson, warmup_davidson_jit
    warmup_davidson_jit(dim, n_channels, n_eig)
    eigenvalues, eigenvectors = davidson(
        apply_H, h_diag=diag, dim=dim, n_channels=2, n_eig=12,
    )
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  JIT'd kernels
# ═══════════════════════════════════════════════════════════════════════

def _generalized_eigh(A, B):
    """Solve A v = λ B v via Cholesky reduction.  JIT-compatible.

    Regularises B with a small shift to handle near-singular overlap
    from non-orthogonal subspace expansion.
    """
    # Regularise: ensure B is safely positive definite
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
def _subspace_step(V, HV, h_diag, n_eig):
    """Project → solve → Ritz vectors → residuals → precondition.

    Single fused JIT: everything between apply_H calls.

    Returns (eigenvalues, X, HX, res_norms, P) where:
      eigenvalues : (n_eig,)
      X, HX       : (n_eig, n_channels, nG) Ritz vectors and H-images
      res_norms   : (n_eig,) residual norms
      P           : (n_eig, n_channels, nG) normalised preconditioned corrections
    """
    # ── project ──
    Hc = jnp.einsum('msG,nsG->mn', jnp.conj(V), HV, optimize=True)
    Sc = jnp.einsum('msG,nsG->mn', jnp.conj(V), V, optimize=True)
    Hc = 0.5 * (Hc + Hc.conj().T)
    Sc = 0.5 * (Sc + Sc.conj().T)

    # ── solve generalized eigenproblem ──
    eig_all, C_all = _generalized_eigh(Hc, Sc)
    Lambda = eig_all[:n_eig]
    C_N = C_all[:, :n_eig]

    # ── Ritz vectors ──
    X = jnp.einsum('mn,msG->nsG', C_N, V, optimize=True)
    HX = jnp.einsum('mn,msG->nsG', C_N, HV, optimize=True)

    # ── residuals ──
    R = HX - X * Lambda[:, None, None]
    res_norms = jnp.sqrt(jnp.sum(jnp.abs(R) ** 2, axis=(1, 2)))

    # ── preconditioner: g(G) = 1 / (h_diag(G) - ε), clamped ──
    _EPS = 1e-2
    denom = h_diag[None, None, :] - Lambda[:, None, None]
    denom = jnp.where(jnp.abs(denom) < _EPS, jnp.sign(denom) * _EPS, denom)
    denom = jnp.where(denom == 0.0, _EPS, denom)
    P = R / denom

    # ── normalise ──
    norms_P = jnp.sqrt(jnp.sum(jnp.abs(P) ** 2, axis=(1, 2)))
    P = P / jnp.maximum(norms_P, 1e-30)[:, None, None]

    return Lambda, X, HX, res_norms, P


# ═══════════════════════════════════════════════════════════════════════
#  Initial guess
# ═══════════════════════════════════════════════════════════════════════

def _build_initial_subspace(apply_H, T_diag, nG, n_channels, n_eig, verbose=True):
    """Lowest-|k+G|² plane waves → H projection → rotate → n_eig vectors."""
    n_pw = min(2 * n_eig, nG)
    n_basis = min(n_pw * n_channels, nG * n_channels)

    # Sort G-vectors by |k+G|², skipping padding (T_diag ≥ 1e10)
    T_np = np.asarray(T_diag)
    order = np.argsort(T_np)

    # Build basis vectorized: one-hot in (spinor, G) space
    # rows[i] = spinor index, cols[i] = G index for the i-th basis vector
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

    V = jnp.einsum('ij,isG->jsG', C_low, V_pw, optimize=True)
    HV = jnp.einsum('ij,isG->jsG', C_low, HV_pw, optimize=True)

    if verbose:
        eigs = np.asarray(eigvals[:n_eig])
        print(f"  Initial guess: {V_pw.shape[0]} basis → {n_eig} vectors, "
              f"eig [{eigs[0]:.6f}, {eigs[-1]:.6f}]")
    return V, HV


# ═══════════════════════════════════════════════════════════════════════
#  Davidson solver
# ═══════════════════════════════════════════════════════════════════════

def warmup_davidson_jit(nG: int, n_channels: int, n_eig: int, m_max: int | None = None):
    """Pre-compile _subspace_step at all subspace sizes.

    Call once before the k-point loop to front-load JIT compilation.
    Costs ~2s total, eliminates all recompilation during Davidson.
    """
    if m_max is None:
        m_max = 4 * n_eig
    h = jnp.zeros(nG, dtype=jnp.float64)
    for m in range(n_eig, m_max + n_eig, n_eig):
        V = jnp.zeros((min(m, m_max), n_channels, nG), dtype=jnp.complex128)
        HV = jnp.zeros_like(V)
        _subspace_step(V, HV, h, n_eig)


def davidson(
    apply_H,
    *,
    h_diag: jax.Array,
    dim: int,
    n_channels: int,
    n_eig: int,
    diag_for_init: jax.Array | None = None,
    m_max: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    X0: jax.Array | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Block Davidson iterative eigensolver.

    Finds the lowest n_eig eigenvalues of a Hermitian operator H, given
    only the matvec callable apply_H.  State vectors are (batch, n_channels, dim).

    Parameters
    ----------
    apply_H : (m, n_channels, dim) → (m, n_channels, dim)
        Hermitian matvec.
    h_diag : (dim,)
        Operator diagonal for the preconditioner.
    dim, n_channels : basis dimensions.
    n_eig : number of lowest eigenvalues to converge.
    diag_for_init : (dim,) diagonal for initial guess ordering (e.g. kinetic energy).
    m_max : max subspace dimension before restart (default 4 × n_eig).

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n_channels, dim)
    """
    nG = dim
    if m_max is None:
        m_max = 4 * n_eig

    # ── initial subspace ──
    if X0 is not None:
        V = jnp.asarray(X0[:n_eig], dtype=jnp.complex128)
        HV = apply_H(V)
    else:
        if diag_for_init is None:
            raise ValueError("diag_for_init required for default initial guess")
        V, HV = _build_initial_subspace(apply_H, diag_for_init, nG, n_channels, n_eig,
                                         verbose=verbose)

    if verbose:
        print(f"Davidson: n_eig={n_eig}, dim={nG}, m_max={m_max}")

    eigenvalues = None

    for it in range(1, max_iter + 1):
        m = V.shape[0]

        # ── GPU: project + solve + Ritz + residual + precondition (one JIT) ──
        Lambda, X, HX, res, P = _subspace_step(V, HV, h_diag, n_eig)

        # ── convergence check (CPU, cheap) ──
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
            print(f"  iter {it:3d}: m={m:3d}  "
                  f"eig[0]={float(Lambda[0]):12.6f}  "
                  f"eig[{n_eig-1}]={float(Lambda[n_eig-1]):12.6f}  "
                  f"res=[{res_np.min():.1e},{res_np.max():.1e}]  "
                  f"conv={n_conv}/{n_eig}")

        if n_conv == n_eig:
            if verbose:
                print(f"  Converged all {n_eig} bands in {it} iterations.")
            return eigenvalues, np.asarray(X)

        # ── apply_H to all n_eig corrections (fixed batch size) ──
        HP = apply_H(P)

        # ── expand subspace with all n_eig vectors ──
        # Converged bands add zero corrections (from the preconditioner:
        # their residuals are ~0), which don't affect the Gram matrices.
        # Keeping the block size fixed ensures m grows in steps of n_eig,
        # matching the warmup shapes exactly.
        V = jnp.concatenate([V, P], axis=0)
        HV = jnp.concatenate([HV, HP], axis=0)

        # ── restart if subspace too large ──
        if V.shape[0] > m_max:
            if verbose:
                print(f"  iter {it}: restart (m={V.shape[0]} > m_max={m_max})")
            V, HV = X, HX

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations. "
              f"Best: {n_conv}/{n_eig}")
    return eigenvalues, np.asarray(X)

"""
psp/davidson.py — Block Davidson eigensolver for plane-wave DFT.

Finds the lowest n_tgt eigenvalues of H at a single k-point.

The algorithm follows QE's cegterg:
  - Subspace expansion with preconditioned residuals
  - Generalized eigenproblem Hc v = ε Sc v (non-orthogonal basis)
  - Thick restart to Ritz vectors when subspace exceeds m_max
  - Diagonal preconditioner: g(G) = 1 / (h_diag(G) - ε)

Performance: all linear-algebra between apply_H calls is fused into
two JIT'd kernels (_project_subspace and _expand_subspace) to minimize
Python dispatch overhead.  apply_H is the only external JIT call per
iteration.

Usage
-----
    from psp.davidson import davidson_k

    eigenvalues, eigenvectors = davidson_k(
        apply_H=lambda psi: ...,
        h_diag=H_k.h_diag,
        nG=H_k.nG,
        nspinor=2,
        n_tgt=n_bands,
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


@functools.partial(jax.jit, static_argnames=('n_tgt',))
def _subspace_step(V, HV, h_diag, n_tgt):
    """Project → solve → Ritz vectors → residuals → precondition.

    Single fused JIT: everything between apply_H calls.

    Returns (eigenvalues, X, HX, res_norms, P) where:
      eigenvalues : (n_tgt,)
      X, HX       : (n_tgt, nspinor, nG) Ritz vectors and H-images
      res_norms   : (n_tgt,) residual norms
      P           : (n_tgt, nspinor, nG) normalised preconditioned corrections
    """
    # ── project ──
    Hc = jnp.einsum('msG,nsG->mn', jnp.conj(V), HV, optimize=True)
    Sc = jnp.einsum('msG,nsG->mn', jnp.conj(V), V, optimize=True)
    Hc = 0.5 * (Hc + Hc.conj().T)
    Sc = 0.5 * (Sc + Sc.conj().T)

    # ── solve generalized eigenproblem ──
    eig_all, C_all = _generalized_eigh(Hc, Sc)
    Lambda = eig_all[:n_tgt]
    C_N = C_all[:, :n_tgt]

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

def _build_initial_subspace(apply_H, T_diag, nG, nspinor, n_tgt, verbose=True):
    """Lowest-|k+G|² plane waves → H projection → rotate → n_tgt vectors."""
    n_pw = min(2 * n_tgt, nG)
    n_basis = min(n_pw * nspinor, nG * nspinor)

    # Sort G-vectors by |k+G|², skipping padding (T_diag ≥ 1e10)
    T_np = np.asarray(T_diag)
    order = np.argsort(T_np)

    # Build basis vectorized: one-hot in (spinor, G) space
    # rows[i] = spinor index, cols[i] = G index for the i-th basis vector
    rows = np.repeat(np.arange(nspinor), n_pw)[:n_basis]
    cols = np.tile(order[:n_pw], nspinor)[:n_basis]
    V_pw = np.zeros((n_basis, nspinor, nG), dtype=np.complex128)
    V_pw[np.arange(n_basis), rows, cols] = 1.0
    V_pw = jnp.asarray(V_pw)

    HV_pw = apply_H(V_pw)

    Hc = jnp.einsum('msG,nsG->mn', jnp.conj(V_pw), HV_pw, optimize=True)
    Hc = 0.5 * (Hc + Hc.conj().T)
    eigvals, C = jnp.linalg.eigh(Hc)
    C_low = C[:, :n_tgt]

    V = jnp.einsum('ij,isG->jsG', C_low, V_pw, optimize=True)
    HV = jnp.einsum('ij,isG->jsG', C_low, HV_pw, optimize=True)

    if verbose:
        eigs = np.asarray(eigvals[:n_tgt])
        print(f"  Initial guess: {V_pw.shape[0]} PWs → {n_tgt} vectors, "
              f"eig [{eigs[0]:.4f}, {eigs[-1]:.4f}] Ry")
    return V, HV


# ═══════════════════════════════════════════════════════════════════════
#  Davidson solver
# ═══════════════════════════════════════════════════════════════════════

def davidson_k(
    apply_H,
    *,
    h_diag: jax.Array,
    nG: int,
    nspinor: int,
    n_tgt: int,
    T_diag: jax.Array | None = None,
    m_max: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    X0: jax.Array | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Block Davidson eigensolver for one k-point.

    Parameters
    ----------
    apply_H : (m, nspinor, nG) → (m, nspinor, nG)
        H|ψ⟩ matvec in sparse-G representation.
    h_diag : (nG,)
        Hamiltonian diagonal for preconditioner.
    nG, nspinor : basis dimensions.
    n_tgt : number of lowest eigenvalues to converge.
    T_diag : (nG,) |k+G|², needed only for initial guess.
    m_max : max subspace dimension before restart (default 4 × n_tgt).

    Returns
    -------
    eigenvalues : (n_tgt,) in Ry.
    eigenvectors : (n_tgt, nspinor, nG) sparse-G.
    """
    if m_max is None:
        m_max = 4 * n_tgt

    # ── initial subspace ──
    if X0 is not None:
        V = jnp.asarray(X0[:n_tgt], dtype=jnp.complex128)
        HV = apply_H(V)
    else:
        if T_diag is None:
            raise ValueError("T_diag required for default initial guess")
        V, HV = _build_initial_subspace(apply_H, T_diag, nG, nspinor, n_tgt,
                                         verbose=verbose)

    if verbose:
        print(f"Davidson: n_tgt={n_tgt}, nG={nG}, m_max={m_max}")

    eigenvalues = None

    for it in range(1, max_iter + 1):
        m = V.shape[0]

        # ── GPU: project + solve + Ritz + residual + precondition (one JIT) ──
        Lambda, X, HX, res, P = _subspace_step(V, HV, h_diag, n_tgt)

        # ── convergence check (CPU, cheap) ──
        res_np = np.asarray(res)
        rel_tol = tol * np.maximum(1.0, np.abs(np.asarray(Lambda)))
        conv = res_np < rel_tol
        n_conv = 0
        for i in range(n_tgt):
            if conv[i]:
                n_conv = i + 1
            else:
                break

        eigenvalues = np.asarray(Lambda)
        if verbose and (it <= 5 or it % 5 == 0 or n_conv == n_tgt):
            print(f"  iter {it:3d}: m={m:3d}  "
                  f"eig[0]={float(Lambda[0]):12.6f}  "
                  f"eig[{n_tgt-1}]={float(Lambda[n_tgt-1]):12.6f}  "
                  f"res=[{res_np.min():.1e},{res_np.max():.1e}]  "
                  f"conv={n_conv}/{n_tgt}")

        if n_conv == n_tgt:
            if verbose:
                print(f"  Converged all {n_tgt} bands in {it} iterations.")
            return eigenvalues, np.asarray(X)

        # ── select unconverged roots ──
        active = jnp.array([i for i in range(n_tgt) if not conv[i]])
        P_act = P[active]

        # ── GPU: apply_H to corrections ──
        HP = apply_H(P_act)

        # ── expand subspace ──
        V = jnp.concatenate([V, P_act], axis=0)
        HV = jnp.concatenate([HV, HP], axis=0)

        # ── restart if subspace too large ──
        if V.shape[0] > m_max:
            if verbose:
                print(f"  iter {it}: restart (m={V.shape[0]} > m_max={m_max})")
            V, HV = X, HX

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations. "
              f"Best: {n_conv}/{n_tgt}")
    return eigenvalues, np.asarray(X)

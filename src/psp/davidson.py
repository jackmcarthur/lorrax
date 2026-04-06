"""
psp/davidson.py — Thick-restart block Davidson eigensolver.

Finds the lowest N_tgt eigenvalues and eigenvectors of the PW DFT
Hamiltonian H_k at a single k-point using only a black-box matvec.

The algorithm follows the design in DAVIDSON.md:
  - Block expansion with Teter-Payne-Allan preconditioner
  - Two-pass DGKS subspace orthogonalisation
  - Thick restart to Ritz vectors when subspace grows too large
  - Hard locking of lowest contiguous converged prefix

Data layout: all PW-space arrays are in sparse-G representation
(nb, nspinor, nG).  Dense projected-subspace arrays (Theta, C, etc.)
are replicated.  No all-gather of G-distributed wavefunctions is
ever required — only all-reduce of small dense matrices.

Usage
-----
    from psp.davidson import davidson_k

    eigenvalues, eigenvectors = davidson_k(
        apply_H=lambda psi: dft_ops.apply(psi, kops),
        T_diag=kops.T_diag,
        nG=kops.nG,
        nspinor=2,
        n_tgt=n_occ,        # number of bands to converge
    )
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Teter-Payne-Allan preconditioner
# ---------------------------------------------------------------------------

def _tpa_precondition(
    R: jax.Array,
    X: jax.Array,
    T_diag: jax.Array,
    delta_T: float = 1e-4,
) -> jax.Array:
    """Apply TPA preconditioner to residual block.

    R     : (b, nspinor, nG) residuals for active roots
    X     : (b, nspinor, nG) corresponding Ritz vectors
    T_diag: (nG,) kinetic diagonal |k+G|^2

    Returns P : (b, nspinor, nG) preconditioned directions
    """
    # Mean kinetic energy per band: T_bar[b] = sum_{s,G} T_G |x[b,s,G]|^2
    T_bar = jnp.sum(
        T_diag[None, None, :] * jnp.abs(X) ** 2, axis=(1, 2)
    )  # (b,)
    T_bar = jnp.maximum(T_bar, delta_T)

    # xi[b,G] = T_G / T_bar[b]
    xi = T_diag[None, None, :] / T_bar[:, None, None]

    # TPA rational: f(x) = (27+18x+12x^2+8x^3) / (27+18x+12x^2+8x^3+16x^4)
    num = 27.0 + xi * (18.0 + xi * (12.0 + xi * 8.0))
    den = num + 16.0 * xi ** 4
    f = num / den

    return f * R


# ---------------------------------------------------------------------------
# Distributed Gram matrix (all-reduce placeholder)
# ---------------------------------------------------------------------------

def _gram(A: jax.Array, B: jax.Array) -> jax.Array:
    """Gram matrix A^dag B.  (m, nspinor, nG) x (n, nspinor, nG) -> (m, n).

    On a single device this is just an einsum.  For multi-GPU G-sharding,
    replace with local einsum + jax.lax.psum over the G-axis.
    """
    return jnp.einsum('msG,nsG->mn', jnp.conj(A), B, optimize=True)


def _gram_diag(A: jax.Array) -> jax.Array:
    """Diagonal of A^dag A: squared norms per vector.  Returns (m,)."""
    return jnp.sum(jnp.abs(A) ** 2, axis=(1, 2))


# ---------------------------------------------------------------------------
# Orthonormalise a block via eigendecomposition of overlap
# ---------------------------------------------------------------------------

def _orthonormalise(V: jax.Array, tau_dep: float = 1e-12):
    """Orthonormalise V, dropping directions with overlap eigenvalue < tau_dep.

    Returns (V_ortho, rank).
    """
    S = _gram(V, V)
    S = 0.5 * (S + S.conj().T)  # symmetrise
    eigvals, U = jnp.linalg.eigh(S)
    mask = eigvals > tau_dep
    rank = int(jnp.sum(mask))
    # Keep only well-conditioned directions
    inv_sqrt = jnp.where(mask, 1.0 / jnp.sqrt(jnp.maximum(eigvals, tau_dep)), 0.0)
    # V_ortho = V @ U @ diag(inv_sqrt)
    V_ortho = jnp.einsum('vsG,ij,j->isG', V, U, inv_sqrt, optimize=True)
    return V_ortho[:rank], rank


# ---------------------------------------------------------------------------
# Core Davidson iteration
# ---------------------------------------------------------------------------

def davidson_k(
    apply_H,
    T_diag: jax.Array,
    nG: int,
    nspinor: int,
    n_tgt: int,
    *,
    block_size: int = 16,
    m_max: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    tau_dep: float = 1e-12,
    delta_T: float = 1e-4,
    n_ortho: int = 2,
    X0: jax.Array | None = None,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Block Davidson eigensolver for one k-point.

    Parameters
    ----------
    apply_H : callable
        Black-box H|psi>.  Takes (m, nspinor, nG) → same shape.
    T_diag : (nG,) kinetic diagonal for TPA preconditioner.
    nG : number of G-vectors.
    nspinor : number of spinor components.
    n_tgt : number of lowest eigenvalues to converge.
    block_size : expansion block size per iteration.
    m_max : max subspace size before restart (default n_tgt + 3*block_size).
    max_iter : maximum outer iterations.
    tol : convergence tolerance on residual norms.
    tau_dep : linear-dependence threshold.
    delta_T : floor for mean kinetic energy in preconditioner.
    n_ortho : number of orthogonalisation passes (1 or 2).
    X0 : (n_tgt, nspinor, nG) initial guess, or None for random.
    verbose : print convergence info.

    Returns
    -------
    eigenvalues : (n_tgt,) in Ry, ascending.
    eigenvectors : (n_tgt, nspinor, nG) sparse-G Ritz vectors.
    """
    if m_max is None:
        m_max = n_tgt + 3 * block_size

    b = min(block_size, n_tgt)

    # -- Step 0: initial guess -------------------------------------------
    if X0 is not None:
        V = jnp.asarray(X0[:n_tgt], dtype=jnp.complex128)
    else:
        # Random initial guess in sparse-G
        key = jax.random.PRNGKey(42)
        V = jax.random.normal(key, (n_tgt, nspinor, nG), dtype=jnp.float64)
        V = V.astype(jnp.complex128)

    V, rank = _orthonormalise(V, tau_dep)
    if verbose:
        print(f"Davidson: n_tgt={n_tgt}, nG={nG}, nspinor={nspinor}, "
              f"block={b}, m_max={m_max}")
        print(f"  Initial guess rank: {rank}")

    # -- Step 1: initial H application -----------------------------------
    W = apply_H(V)  # W = H V

    n_locked = 0
    eigenvalues = None

    for it in range(1, max_iter + 1):
        m = V.shape[0]

        # -- 2a: projected Hamiltonian -----------------------------------
        Theta = _gram(V, W)
        Theta = 0.5 * (Theta + Theta.conj().T)

        # -- 2b: Rayleigh-Ritz ------------------------------------------
        eigvals, C = jnp.linalg.eigh(Theta)
        # Take lowest n_tgt
        Lambda = eigvals[:n_tgt]
        C_N = C[:, :n_tgt]

        # -- 2c: Ritz vectors and H-images ------------------------------
        # C_N: (m, n_tgt), V: (m, nspinor, nG) → X: (n_tgt, nspinor, nG)
        X = jnp.einsum('mn,msG->nsG', C_N, V, optimize=True)
        HX = jnp.einsum('mn,msG->nsG', C_N, W, optimize=True)

        # -- 2d: residuals ----------------------------------------------
        R = HX - X * Lambda[:, None, None]
        res_norms = jnp.sqrt(_gram_diag(R))

        # -- 2e: convergence check --------------------------------------
        rel_tol = tol * jnp.maximum(1.0, jnp.abs(Lambda))
        converged = res_norms < rel_tol
        # Lowest contiguous converged prefix
        n_conv = 0
        conv_np = np.asarray(converged)
        for i in range(n_tgt):
            if conv_np[i]:
                n_conv = i + 1
            else:
                break

        eigenvalues = np.asarray(Lambda)
        if verbose and (it <= 5 or it % 5 == 0 or n_conv == n_tgt):
            max_res = float(jnp.max(res_norms))
            min_res = float(jnp.min(res_norms))
            print(f"  iter {it:3d}: m={m:3d}  "
                  f"eig[0]={float(Lambda[0]):12.6f}  "
                  f"eig[{n_tgt-1}]={float(Lambda[n_tgt-1]):12.6f}  "
                  f"res=[{min_res:.1e},{max_res:.1e}]  "
                  f"conv={n_conv}/{n_tgt}")

        if n_conv == n_tgt:
            if verbose:
                print(f"  Converged all {n_tgt} bands in {it} iterations.")
            return eigenvalues, np.asarray(X)

        # -- 2f: active block -------------------------------------------
        # Choose lowest unconverged roots
        active = []
        for i in range(n_tgt):
            if not conv_np[i]:
                active.append(i)
            if len(active) >= b:
                break
        active = jnp.array(active)
        b_act = len(active)

        R_act = R[active]
        X_act = X[active]

        # -- 2g: precondition -------------------------------------------
        P = _tpa_precondition(R_act, X_act, T_diag, delta_T)

        # -- 2h: orthogonalise against subspace -------------------------
        for _pass in range(n_ortho):
            B = _gram(V, P)        # (m_sub, b_act)
            # P -= V @ B : subtract projection of P onto V
            P = P - jnp.einsum('mn,msG->nsG', B, V, optimize=True)

        # -- 2i: orthonormalise candidate block -------------------------
        P, bp = _orthonormalise(P, tau_dep)

        if bp == 0:
            # Linearly dependent — restart
            if verbose:
                print(f"  iter {it}: restart (candidate block dependent)")
            V = X
            W = HX
            continue

        # -- 2j: apply H to new block -----------------------------------
        HP = apply_H(P)

        # -- 2k: augment subspace ---------------------------------------
        V = jnp.concatenate([V, P], axis=0)
        W = jnp.concatenate([W, HP], axis=0)

        # -- 2l: restart if needed --------------------------------------
        if V.shape[0] > m_max:
            if verbose:
                print(f"  iter {it}: restart (m={V.shape[0]} > m_max={m_max})")
            V = X
            W = HX

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations. "
              f"Best: {n_conv}/{n_tgt}")
    return eigenvalues, np.asarray(X)

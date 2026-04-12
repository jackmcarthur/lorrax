"""
psp/davidson.py — Block Davidson eigensolver for plane-wave DFT.

Finds the lowest n_tgt eigenvalues of H_k using iterative subspace
expansion with a diagonal preconditioner (QE's g_psi convention).

The default initial guess uses the lowest-|k+G|² plane waves, rotated
by a subspace diagonalization of H in that basis.  This mirrors QE's
init_wfc → rotate_wfc pipeline and works without atomic wavefunctions.

Usage
-----
    from psp.davidson import davidson_k

    eigenvalues, eigenvectors = davidson_k(
        apply_H=lambda psi: ...,   # H|ψ⟩ matvec
        T_diag=H_k.T_diag,
        nG=H_k.nG,
        nspinor=2,
        n_tgt=n_bands,
    )
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Initial guess: low-G plane waves → subspace rotation
# ═══════════════════════════════════════════════════════════════════════

def _build_initial_guess(
    apply_H,
    T_diag: jax.Array,
    nG: int,
    nspinor: int,
    n_tgt: int,
    n_basis: int | None = None,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Build initial subspace from lowest-|k+G|² plane waves.

    1. Select n_basis plane waves with smallest T_diag (= |k+G|²).
    2. Build H in that basis via apply_H.
    3. Diagonalize → take lowest n_tgt eigenvectors.

    Returns (V, W) where V is (n_tgt, nspinor, nG) orthonormal
    and W = H V.
    """
    if n_basis is None:
        # Use 2× the target, capped by nG (for each spinor component)
        n_basis = min(2 * n_tgt, nG * nspinor)

    # Select G-indices with smallest |k+G|²
    order = np.argsort(np.asarray(T_diag))
    n_pw = min(n_basis // nspinor, nG)  # plane waves per spinor

    # Build sparse-G basis: one plane wave per basis vector
    # For nspinor=2: first n_pw vectors have spinor-up, next n_pw spinor-down
    basis = []
    for s in range(nspinor):
        for ig in range(n_pw):
            v = jnp.zeros((nspinor, nG), dtype=jnp.complex128)
            v = v.at[s, order[ig]].set(1.0)
            basis.append(v)
            if len(basis) >= n_basis:
                break
        if len(basis) >= n_basis:
            break

    V_pw = jnp.stack(basis, axis=0)  # (n_basis, nspinor, nG)
    n_actual = V_pw.shape[0]

    if verbose:
        print(f"  Initial guess: {n_actual} plane waves, "
              f"T range [{float(T_diag[order[0]]):.3f}, "
              f"{float(T_diag[order[min(n_pw-1, nG-1)]]):.3f}] Ry")

    # Apply H to the plane-wave basis
    W_pw = apply_H(V_pw)

    # Project: H_mn = <V_pw|W_pw>
    H_small = jnp.einsum('msG,nsG->mn', jnp.conj(V_pw), W_pw, optimize=True)
    H_small = 0.5 * (H_small + H_small.conj().T)

    # Diagonalize — keep ALL vectors as the initial subspace.
    # Davidson needs room to expand; starting with exactly n_tgt
    # leaves no orthogonal directions for the residuals.
    eigvals, C = jnp.linalg.eigh(H_small)

    # Rotate back to sparse-G (full rotated basis)
    V = jnp.einsum('ij,isG->jsG', C, V_pw, optimize=True)
    W = jnp.einsum('ij,isG->jsG', C, W_pw, optimize=True)

    if verbose:
        eigs = np.asarray(eigvals[:n_tgt])
        print(f"  Subspace rotation: eig range [{eigs[0]:.4f}, {eigs[-1]:.4f}] Ry"
              f" (kept {n_actual} basis vectors)")

    return V, W


# ═══════════════════════════════════════════════════════════════════════
#  Preconditioner: QE's g_psi (diagonal)
# ═══════════════════════════════════════════════════════════════════════

def _precondition(
    R: jax.Array,
    h_diag: jax.Array,
    eigenvalues: jax.Array,
) -> jax.Array:
    """QE-style diagonal preconditioner: ψ'(G) = ψ(G) / (h_diag(G) − ε).

    h_diag should include the local potential average:
        h_diag(G) = |k+G|² + V_scf(G=0)
    so the denominator is never near zero for the lowest bands.

    R          : (b, nspinor, nG) residuals
    h_diag     : (nG,) Hamiltonian diagonal approximation
    eigenvalues: (b,) current eigenvalue estimates
    """
    eps = 1e-2
    denom = h_diag[None, None, :] - eigenvalues[:, None, None]
    denom = jnp.where(jnp.abs(denom) < eps, jnp.sign(denom) * eps, denom)
    denom = jnp.where(denom == 0.0, eps, denom)
    return R / denom


# ═══════════════════════════════════════════════════════════════════════
#  Gram matrix and orthonormalization
# ═══════════════════════════════════════════════════════════════════════

def _gram(A: jax.Array, B: jax.Array) -> jax.Array:
    """⟨A|B⟩ Gram matrix. (m, nspinor, nG) × (n, nspinor, nG) → (m, n)."""
    return jnp.einsum('msG,nsG->mn', jnp.conj(A), B, optimize=True)


def _gram_diag(A: jax.Array) -> jax.Array:
    """Squared norms per vector. Returns (m,)."""
    return jnp.sum(jnp.abs(A) ** 2, axis=(1, 2))


def _orthonormalise(V: jax.Array, tau_dep: float = 1e-12):
    """Orthonormalise via overlap eigendecomposition. Returns (V_ortho, rank)."""
    S = _gram(V, V)
    S = 0.5 * (S + S.conj().T)
    eigvals, U = jnp.linalg.eigh(S)
    mask = eigvals > tau_dep
    rank = int(jnp.sum(mask))
    inv_sqrt = jnp.where(mask, 1.0 / jnp.sqrt(jnp.maximum(eigvals, tau_dep)), 0.0)
    V_ortho = jnp.einsum('vsG,ij,j->isG', V, U, inv_sqrt, optimize=True)
    return V_ortho[:rank], rank


# ═══════════════════════════════════════════════════════════════════════
#  Davidson iteration
# ═══════════════════════════════════════════════════════════════════════

def davidson_k(
    apply_H,
    T_diag: jax.Array,
    nG: int,
    nspinor: int,
    n_tgt: int,
    *,
    h_diag: jax.Array | None = None,
    block_size: int = 16,
    m_max: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    tau_dep: float = 1e-12,
    n_ortho: int = 2,
    X0: jax.Array | None = None,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Block Davidson eigensolver for one k-point.

    Parameters
    ----------
    apply_H : callable
        H|ψ⟩ matvec.  (m, nspinor, nG) → (m, nspinor, nG).
    T_diag : (nG,) |k+G|² diagonal.
    nG, nspinor : basis dimensions.
    n_tgt : number of lowest eigenvalues to converge.
    h_diag : (nG,) Hamiltonian diagonal for preconditioner.
        Should be |k+G|² + V_loc(G=0) + V_NL_diag(G) (QE convention).
        If None, falls back to T_diag (kinetic only — less stable).
    X0 : optional (n, nspinor, nG) initial guess.
        If None, builds from lowest-G plane waves + subspace rotation.

    Returns
    -------
    eigenvalues : (n_tgt,) Ry, ascending.
    eigenvectors : (n_tgt, nspinor, nG) sparse-G.
    """
    if m_max is None:
        m_max = max(n_tgt + 3 * min(block_size, n_tgt),
                    6 * n_tgt)  # room for initial basis + expansion
    b = min(block_size, n_tgt)

    if h_diag is None:
        h_diag = T_diag

    # ── initial guess ──
    if X0 is not None:
        V = jnp.asarray(X0[:n_tgt], dtype=jnp.complex128)
        V, rank = _orthonormalise(V, tau_dep)
        W = apply_H(V)
    else:
        # Use enough plane waves that the subspace diag gives ~mRy accuracy.
        # For typical PW DFT, ~30× n_tgt is sufficient; cap at all PWs.
        n_basis = min(max(30 * n_tgt, 200), nG * nspinor)
        V, W = _build_initial_guess(apply_H, T_diag, nG, nspinor, n_tgt,
                                     n_basis=n_basis, verbose=verbose)

    if verbose:
        print(f"Davidson: n_tgt={n_tgt}, nG={nG}, nspinor={nspinor}, "
              f"block={b}, m_max={m_max}")

    eigenvalues = None

    for it in range(1, max_iter + 1):
        m = V.shape[0]

        # ── projected Hamiltonian ──
        Theta = _gram(V, W)
        Theta = 0.5 * (Theta + Theta.conj().T)

        # ── Rayleigh-Ritz ──
        eigvals, C = jnp.linalg.eigh(Theta)
        Lambda = eigvals[:n_tgt]
        C_N = C[:, :n_tgt]

        # ── Ritz vectors and H-images ──
        X = jnp.einsum('mn,msG->nsG', C_N, V, optimize=True)
        HX = jnp.einsum('mn,msG->nsG', C_N, W, optimize=True)

        # ── residuals ──
        R = HX - X * Lambda[:, None, None]
        res_norms = jnp.sqrt(_gram_diag(R))

        # ── convergence ──
        rel_tol = tol * jnp.maximum(1.0, jnp.abs(Lambda))
        converged = res_norms < rel_tol
        conv_np = np.asarray(converged)
        n_conv = 0
        for i in range(n_tgt):
            if conv_np[i]:
                n_conv = i + 1
            else:
                break

        eigenvalues = np.asarray(Lambda)
        if verbose and (it <= 5 or it % 5 == 0 or n_conv == n_tgt):
            print(f"  iter {it:3d}: m={m:3d}  "
                  f"eig[0]={float(Lambda[0]):12.6f}  "
                  f"eig[{n_tgt-1}]={float(Lambda[n_tgt-1]):12.6f}  "
                  f"res=[{float(jnp.min(res_norms)):.1e},"
                  f"{float(jnp.max(res_norms)):.1e}]  "
                  f"conv={n_conv}/{n_tgt}")

        if n_conv == n_tgt:
            if verbose:
                print(f"  Converged all {n_tgt} bands in {it} iterations.")
            return eigenvalues, np.asarray(X)

        # ── active block (unconverged roots with non-trivial residual) ──
        # Only expand roots whose residual is large enough to produce a
        # useful correction.  Nearly-converged roots add only noise after
        # orthogonalisation, which contaminates the subspace with zero-
        # eigenvalue vectors.
        expansion_floor = max(tol * 100, 1e-4)
        active = []
        for i in range(n_tgt):
            if not conv_np[i] and float(res_norms[i]) > expansion_floor:
                active.append(i)
            if len(active) >= b:
                break
        if len(active) == 0:
            # All residuals below floor — just tighten via Rayleigh-Ritz
            # by expanding with the worst residual only
            worst = int(np.argmax(np.asarray(res_norms[:n_tgt])))
            active = [worst]
        b_act = len(active)
        active = jnp.array(active)
        R_act = R[active]

        # ── precondition: QE diagonal g_psi ──
        P = _precondition(R_act, h_diag, Lambda[active])

        # ── orthogonalise against subspace (QE: two-pass Gram-Schmidt) ──
        for _pass in range(n_ortho):
            overlap = _gram(V, P)
            P = P - jnp.einsum('mn,msG->nsG', overlap, V, optimize=True)

        # ── normalize each vector individually (QE convention) ──
        # Drop vectors with norm below threshold instead of
        # eigendecomposition (which amplifies noise in degenerate directions)
        norms = jnp.sqrt(_gram_diag(P))
        keep = []
        for n in range(b_act):
            if float(norms[n]) > tau_dep:
                keep.append(n)
        if len(keep) == 0:
            if verbose:
                print(f"  iter {it}: restart (candidate block dependent)")
            V, W = X, HX
            continue
        keep_idx = jnp.array(keep)
        P = P[keep_idx] / norms[keep_idx, None, None]
        bp = len(keep)

        # ── expand subspace ──
        HP = apply_H(P)
        V = jnp.concatenate([V, P], axis=0)
        W = jnp.concatenate([W, HP], axis=0)

        # ── restart if subspace too large ──
        if V.shape[0] > m_max:
            if verbose:
                print(f"  iter {it}: restart (m={V.shape[0]} > m_max={m_max})")
            V, W = X, HX

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations. "
              f"Best: {n_conv}/{n_tgt}")
    return eigenvalues, np.asarray(X)

"""
psp/davidson.py — Block Davidson eigensolver for plane-wave DFT.

Finds the lowest n_tgt eigenvalues of H at a single k-point.

The algorithm follows QE's cegterg:
  - Subspace expansion with preconditioned residuals
  - Generalized eigenproblem Hc v = ε Sc v (non-orthogonal basis)
  - Thick restart to Ritz vectors when subspace exceeds m_max
  - Diagonal preconditioner: g(G) = 1 / (h_diag(G) - ε)

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

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh as scipy_eigh


# ═══════════════════════════════════════════════════════════════════════
#  Linear algebra helpers
# ═══════════════���═══════════════════════════════════════════════════════

def _gram(A: jax.Array, B: jax.Array) -> jax.Array:
    """⟨A|B⟩ Gram matrix.  (m, nspinor, nG) × (n, ...) → (m, n)."""
    return jnp.einsum('msG,nsG->mn', jnp.conj(A), B, optimize=True)


def _norms(A: jax.Array) -> jax.Array:
    """Per-vector norms.  (m, nspinor, nG) → (m,)."""
    return jnp.sqrt(jnp.sum(jnp.abs(A) ** 2, axis=(1, 2)))


# ═══════════════════════════════════════════════════════════════════════
#  Initial guess
# ═══���════════════════════════════════════════════════���══════════════════

def _build_initial_subspace(
    apply_H,
    T_diag: jax.Array,
    nG: int,
    nspinor: int,
    n_tgt: int,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Lowest-|k+G|² plane waves → H projection → rotate → n_tgt vectors.

    Returns (V, HV) where V is (n_tgt, nspinor, nG).
    """
    # 4× oversampling: enough for mRy-level initial accuracy
    n_pw = min(2 * n_tgt, nG)
    n_basis = min(n_pw * nspinor, nG * nspinor)
    order = np.argsort(np.asarray(T_diag))

    # One plane wave per basis vector, cycling spinor channels
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

    V_pw = jnp.stack(basis, axis=0)
    HV_pw = apply_H(V_pw)

    # Diagonalize H in the PW basis, take lowest n_tgt
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


# ═════════════════════════════════════════���═════════════════════════════
#  Preconditioner
# ════════════════════════��══════════════════════════════════════════════

def _precondition(
    R: jax.Array,
    h_diag: jax.Array,
    eigenvalues: jax.Array,
) -> jax.Array:
    """Diagonal preconditioner: R(G) / (h_diag(G) − ε).

    h_diag = |k+G|² + V_loc(G=0) + V_NL_diag(G)  (QE's convention).
    Clamps small denominators to ±0.01 Ry.
    """
    _EPS = 1e-2
    denom = h_diag[None, None, :] - eigenvalues[:, None, None]
    denom = jnp.where(jnp.abs(denom) < _EPS, jnp.sign(denom) * _EPS, denom)
    denom = jnp.where(denom == 0.0, _EPS, denom)
    return R / denom


# ══════════════════════════���════════════════════════════════════════════
#  Davidson solver
# ═════════════════════��══════════════════════════════════���══════════════

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
        Build with ``dft_operators.build_h_diag``.
    nG, nspinor : basis dimensions.
    n_tgt : number of lowest eigenvalues to converge.
    T_diag : (nG,) |k+G|², needed only for initial guess (ignored if X0 given).
    m_max : max subspace dimension before restart (default 4 × n_tgt).
    X0 : (n, nspinor, nG) initial guess, or None for PW subspace rotation.

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
            raise ValueError("T_diag required for default initial guess (or provide X0)")
        V, HV = _build_initial_subspace(apply_H, T_diag, nG, nspinor, n_tgt,
                                         verbose=verbose)

    if verbose:
        print(f"Davidson: n_tgt={n_tgt}, nG={nG}, m_max={m_max}")

    eigenvalues = None

    for it in range(1, max_iter + 1):
        m = V.shape[0]

        # ── projected generalized eigenproblem: Hc v = ε Sc v ─���
        Hc = _gram(V, HV)
        Sc = _gram(V, V)
        Hc = 0.5 * (Hc + Hc.conj().T)
        Sc = 0.5 * (Sc + Sc.conj().T)

        eig_all, C_all = scipy_eigh(np.asarray(Hc), np.asarray(Sc))
        eig_all = jnp.asarray(eig_all)
        C_all = jnp.asarray(C_all)

        Lambda = eig_all[:n_tgt]
        C_N = C_all[:, :n_tgt]

        # ── Ritz vectors ──
        X = jnp.einsum('mn,msG->nsG', C_N, V, optimize=True)
        HX = jnp.einsum('mn,msG->nsG', C_N, HV, optimize=True)

        # ── residuals: R = H|X⟩ − ε|X⟩ ──
        R = HX - X * Lambda[:, None, None]
        res = _norms(R)

        # ── convergence check ──
        rel_tol = tol * jnp.maximum(1.0, jnp.abs(Lambda))
        conv = np.asarray(res < rel_tol)
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
                  f"res=[{float(jnp.min(res)):.1e},{float(jnp.max(res)):.1e}]  "
                  f"conv={n_conv}/{n_tgt}")

        if n_conv == n_tgt:
            if verbose:
                print(f"  Converged all {n_tgt} bands in {it} iterations.")
            return eigenvalues, np.asarray(X)

        # ── select unconverged roots for expansion ──
        active = [i for i in range(n_tgt) if not conv[i]]
        R_act = R[jnp.array(active)]

        # ── precondition + normalize ──
        P = _precondition(R_act, h_diag, Lambda[jnp.array(active)])
        norms_P = _norms(P)
        keep = [i for i in range(len(active)) if float(norms_P[i]) > 1e-12]
        if len(keep) == 0:
            V, HV = X, HX
            continue
        P = P[jnp.array(keep)] / norms_P[jnp.array(keep), None, None]

        # ── expand subspace ──
        HP = apply_H(P)
        V = jnp.concatenate([V, P], axis=0)
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

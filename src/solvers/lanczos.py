"""
solvers/lanczos.py — Lanczos iterative eigensolvers.

Finds the lowest n_eig eigenvalues of a Hermitian operator H given only
a callable matvec.  No physics knowledge — works for any Hermitian
eigenproblem.

Three variants:
  - simple_lanczos_eig:  Python-loop, full reorthogonalization
  - lanczos_eig_jit:     lax.fori_loop, partial reorthogonalization (JIT-able)
  - block_lanczos_eig:   Block Lanczos for shaped state vectors

Usage
-----
    from solvers.lanczos import lanczos_eig_jit
    eigenvalues, eigenvectors = lanczos_eig_jit(matvec, n=1000, n_eig=10)
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def block_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    shape: tuple[int, ...],
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    tol: float = 1e-8,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Block Lanczos for lowest eigenvalues of a Hermitian operator.

    Parameters
    ----------
    matvec : (block_size, *shape) -> (block_size, *shape)
        Hermitian matvec operating on a block of vectors.
    shape : tuple
        Shape of a single state vector.
    n_eig : int
        Number of lowest eigenvalues to compute.
    block_size : int
        Number of vectors per Lanczos block.
    max_iter : int
        Maximum number of block iterations.
    tol : float
        Convergence tolerance on beta norm.
    seed : int
        Random seed for initial vectors.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, *shape)
    """
    n_flat = int(np.prod(shape))
    key = jax.random.PRNGKey(seed)

    k1, k2 = jax.random.split(key)
    Q0 = jax.random.normal(k1, (block_size, *shape), dtype=jnp.float64)
    Q0 = Q0 + 1j * jax.random.normal(k2, (block_size, *shape), dtype=jnp.float64)

    Q0_flat = Q0.reshape(block_size, n_flat)
    Q0_flat, _ = jnp.linalg.qr(Q0_flat.T)
    Q0_flat = Q0_flat.T

    Q_blocks = [Q0_flat]
    alpha_blocks = []
    beta_blocks = []

    Q_current = Q0_flat.reshape(block_size, *shape)

    for j in range(max_iter):
        Z = matvec(Q_current)
        Z_flat = Z.reshape(block_size, n_flat)
        Q_current_flat = Q_current.reshape(block_size, n_flat)

        alpha_j = Q_current_flat.conj() @ Z_flat.T
        alpha_blocks.append(alpha_j)

        Z_flat = Z_flat - alpha_j.T @ Q_current_flat
        if j > 0:
            Q_prev_flat = Q_blocks[-2]
            Z_flat = Z_flat - beta_blocks[-1].T @ Q_prev_flat

        for Q_old in Q_blocks:
            proj = Z_flat @ Q_old.conj().T
            Z_flat = Z_flat - proj @ Q_old

        Z_flat_T, R = jnp.linalg.qr(Z_flat.T)
        # P1: beta_j = R (NOT R.T).  With Z_col = Z_flat.T = Q R, the block
        # recurrence residual is Z = Q_{j+1} R, so the sub-diagonal block of T is
        # exactly R and the super-diagonal is R^H.  The old ``R.T`` transposed
        # both off-diagonal blocks (they came out conj(R)/R.T instead of R/R^H),
        # which the final (T+T^H)/2 masked for eigenVALUES but corrupted the
        # T->Q_all mapping used for eigenVECTORS (solver_program P1).
        beta_j = R
        beta_blocks.append(beta_j)

        beta_norm = jnp.linalg.norm(beta_j)
        if beta_norm < tol * block_size:
            print(f"Block Lanczos converged at iteration {j+1}")
            break

        Q_next_flat = Z_flat_T.T
        Q_blocks.append(Q_next_flat)
        Q_current = Q_next_flat.reshape(block_size, *shape)

    n_blocks = len(alpha_blocks)
    T_size = n_blocks * block_size
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)

    for i, alpha in enumerate(alpha_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        T = T.at[start:end, start:end].set(alpha)

        if i < len(beta_blocks) - 1:
            beta = beta_blocks[i]
            T = T.at[end:end + block_size, start:end].set(beta)
            T = T.at[start:end, end:end + block_size].set(beta.conj().T)

    T = (T + T.conj().T) / 2

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T.real)[:n_eig]
    eigenvalues = evals_T[idx].real

    Q_all = jnp.concatenate(Q_blocks[:n_blocks], axis=0)
    eigenvectors_flat = vecs_T[:, idx].T @ Q_all
    eigenvectors = eigenvectors_flat.reshape(n_eig, *shape)

    norms = jnp.linalg.norm(eigenvectors.reshape(n_eig, -1), axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms.reshape(n_eig, *([1] * len(shape)))

    return eigenvalues, eigenvectors


def simple_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Simple Lanczos with full reorthogonalization (Python loop).

    Parameters
    ----------
    matvec : (n,) -> (n,)
        Hermitian matvec on flat vectors.
    n : int
        Vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    max_iter : int
        Maximum Lanczos iterations.
    seed : int
        Random seed for initial vector.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n)
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    q = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q = q + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)

    Q = jnp.zeros((n, max_iter + 1), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)

    for j in range(max_iter):
        z = matvec(q)
        alpha = alpha.at[j].set(jnp.vdot(q, z).real)

        if j > 0:
            z = z - beta[j - 1] * Q[:, j - 1]
        z = z - alpha[j] * q

        for i in range(j + 1):
            proj = jnp.vdot(Q[:, i], z)
            z = z - proj * Q[:, i]

        beta = beta.at[j].set(jnp.linalg.norm(z))
        if beta[j] < 1e-12:
            max_iter = j + 1
            break

        q = z / beta[j]
        Q = Q.at[:, j + 1].set(q)

    T = jnp.diag(alpha[:max_iter])
    if max_iter > 1:
        off = beta[:max_iter - 1]
        T = T + jnp.diag(off, 1) + jnp.diag(off, -1)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]

    eigenvalues = evals_T[idx]
    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T

    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms

    return eigenvalues, eigenvectors


def lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
    n_reorth: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled Lanczos using lax.fori_loop.

    Parameters
    ----------
    matvec : (n,) -> (n,)
        Hermitian matvec on flat vectors.
    n : int
        Vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    max_iter : int
        Maximum Lanczos iterations (fixed for JIT).
    seed : int
        Random seed for initial vector.
    n_reorth : int
        Window size for partial reorthogonalization.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n)
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    q0 = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q0 = q0 + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q0 = q0 / jnp.linalg.norm(q0)

    # +1 column so the last iteration does not overwrite Q[:, max_iter-1] (P1).
    Q = jnp.zeros((n, max_iter + 1), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q0)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)

    def lanczos_step(j, carry):
        Q, alpha, beta, q_prev = carry
        z = matvec(q_prev)

        alpha_j = jnp.vdot(q_prev, z).real
        alpha = alpha.at[j].set(alpha_j)

        z = z - alpha_j * q_prev
        q_prev_prev = Q[:, jnp.maximum(j - 1, 0)]
        beta_prev = jnp.where(j > 0, beta[j - 1], 0.0)
        z = z - beta_prev * q_prev_prev

        def reorth_body(i, z_acc):
            valid = i < j
            q_i = Q[:, i]
            proj = jnp.where(valid, jnp.vdot(q_i, z_acc), 0.0 + 0j)
            return z_acc - proj * q_i

        start_idx = jnp.maximum(0, j - n_reorth)
        z = lax.fori_loop(start_idx, j + 1, reorth_body, z)

        beta_j = jnp.linalg.norm(z)
        beta = beta.at[j].set(beta_j)

        q_next = z / jnp.maximum(beta_j, 1e-15)
        Q = Q.at[:, j + 1].set(q_next)

        return (Q, alpha, beta, q_next)

    init_carry = (Q, alpha, beta, q0)
    Q, alpha, beta, _ = lax.fori_loop(0, max_iter, lanczos_step, init_carry)

    T = jnp.diag(alpha)
    off_diag = beta[:-1]
    T = T + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]

    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)

    return eigenvalues, eigenvectors


def _build_block_tridiag(alpha_all, beta_all, max_iter: int, bs: int):
    """Build the block-tridiagonal T from per-iter (bs,bs) blocks.

    Done inside the jit by ``lax.fori_loop`` so the trace-time HLO stays
    O(1) instead of unrolling ``max_iter`` slot updates. Used by both
    the fixed-iter and convergence-driven block Lanczos paths.
    """
    T_size = bs * max_iter
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)

    def body(j, T):
        s = j * bs
        T = lax.dynamic_update_slice(T, alpha_all[j], (s, s))
        # Off-diagonal beta only when j+1 < max_iter (zero alpha/beta past
        # the end keeps the slot a no-op even when j is at the boundary).
        T = lax.dynamic_update_slice(T, beta_all[j], (s + bs, s))
        T = lax.dynamic_update_slice(
            T, jnp.conj(beta_all[j]).T, (s, s + bs))
        return T

    T = lax.fori_loop(0, max_iter - 1, body, T)
    # Final diagonal block (no off-diagonal past the end).
    s_last = (max_iter - 1) * bs
    T = lax.dynamic_update_slice(T, alpha_all[max_iter - 1], (s_last, s_last))
    return (T + jnp.conj(T).T) * 0.5


def block_lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    seed: int = 42,
    n_reorth: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled block Lanczos using ``lax.fori_loop``.

    Same algorithm as :func:`block_lanczos_eig`, but all state lives in
    pre-allocated arrays so the body fits in ``lax.fori_loop`` and the
    caller's outer jit can fuse this with the matvec.  The matvec
    operates on a *block* of trial vectors

        matvec : (block_size, n) -> (block_size, n)

    so the BSE-style ring matvec processes ``block_size`` vectors per
    call.  That makes the per-call GEMMs ``block_size`` times larger
    (better arithmetic intensity / GPU occupancy) and reduces the host
    dispatch count by ``block_size`` for the same total Krylov
    dimension.

    The total Krylov dimension is ``block_size * max_iter``; pick
    ``max_iter`` so this is comparable to a single-vector Lanczos's
    ``max_iter``.

    Parameters
    ----------
    matvec : (block_size, n) -> (block_size, n)
        Hermitian matvec on a block of flat vectors.
    n : int
        Single-vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    block_size : int
        Vectors per Lanczos block.
    max_iter : int
        Block iterations (fixed for JIT). Total Krylov size = block_size·max_iter.
    seed : int
        Random seed for initial block.
    n_reorth : int
        Window size (in *blocks*) for partial reorthogonalisation.
    """
    bs = int(block_size)
    T_size = bs * int(max_iter)

    # Initial orthonormal block via QR of random complex Gaussian.
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0, _ = jnp.linalg.qr(Q0)                          # (n, bs)

    # Ring buffer of all Q-blocks: (max_iter + 1, n, bs) — the +1 slot holds the
    # final Q_next so the last iteration does NOT overwrite Q_{max_iter-1} (the
    # slot-overwrite bug, solver_program P1: it corrupted the last Krylov block
    # in the eigenvector reconstruction).  alpha/beta: (max_iter, bs, bs).
    Q_all = jnp.zeros((int(max_iter) + 1, n, bs), dtype=jnp.complex128)
    Q_all = Q_all.at[0].set(Q0)
    alpha_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)

    def body(j, carry):
        Q_all, alpha_all, beta_all = carry
        Q_j = Q_all[j]                                 # (n, bs)
        # Block matvec over (bs, n) → (bs, n); transpose to (n, bs).
        Z = matvec(Q_j.T).T                            # (n, bs)

        alpha_j = jnp.conj(Q_j).T @ Z                  # (bs, bs)
        alpha_all = alpha_all.at[j].set(alpha_j)
        Z = Z - Q_j @ alpha_j

        # Subtract Q_{j-1} · β_{j-1}^H (skip on j=0).
        Q_jm1 = Q_all[jnp.maximum(j - 1, 0)]
        beta_prev = beta_all[jnp.maximum(j - 1, 0)]
        Z = jnp.where(j > 0, Z - Q_jm1 @ jnp.conj(beta_prev).T, Z)

        # Partial reorth over the last n_reorth blocks.
        def reorth_body(i, Z_acc):
            valid = i < j
            Q_i = Q_all[i]
            proj = jnp.where(valid, jnp.conj(Q_i).T @ Z_acc, jnp.zeros((bs, bs), dtype=Z_acc.dtype))
            return Z_acc - Q_i @ proj
        start = jnp.maximum(0, j - n_reorth)
        Z = lax.fori_loop(start, j + 1, reorth_body, Z)

        # QR(Z) → next block + β_j.  Write to slot j+1 (always valid with the
        # +1 buffer, 1..max_iter) — no clobber of the current Q_j.
        Q_next, beta_j = jnp.linalg.qr(Z)              # (n, bs), (bs, bs)
        beta_all = beta_all.at[j].set(beta_j)
        Q_all = Q_all.at[j + 1].set(Q_next)
        return (Q_all, alpha_all, beta_all)

    Q_all, alpha_all, beta_all = lax.fori_loop(
        0, int(max_iter), body, (Q_all, alpha_all, beta_all))

    # Block-tridiagonal T built inside-jit (no Python loop unroll).
    T = _build_block_tridiag(alpha_all, beta_all, int(max_iter), bs)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]

    # Q_all is (max_iter + 1, n, bs); the T basis is the first max_iter blocks.
    Q_full = jnp.transpose(Q_all[:int(max_iter)], (1, 0, 2)).reshape(n, T_size)
    eigenvectors = (Q_full @ vecs_T[:, idx]).T          # (n_eig, n)
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    return eigenvalues, eigenvectors


def block_lanczos_eig_jit_converged(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    check_every: int = 4,
    min_iter: int | None = None,
    seed: int = 42,
    n_reorth: int = 2,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convergence-driven block Lanczos via ``lax.while_loop``.

    Same algorithm as :func:`block_lanczos_eig_jit`, but the iteration
    count is decided by Ritz-eigenvalue stability rather than fixed:

      every ``check_every`` block-iters, build the partial T (with
      future-block α/β set to zero), eigh it, and compare the lowest
      ``n_eig`` Ritz values against the previous check.  Exit when

          max_i |λ_i - λ_i_prev| < rtol·max(|λ_i|, atol).

    The pre-allocated buffers fix the upper bound at ``max_iter`` block
    iterations (so ``max_iter * block_size`` total Krylov dimension);
    the ``while_loop`` carry includes the running iteration count and
    the previous Ritz values for comparison.

    Returns (eigenvalues, eigenvectors, n_iter_done) — the third value
    is the actual block iteration count where the loop exited (≤
    ``max_iter``).
    """
    bs = int(block_size)
    M = int(max_iter)
    T_size = bs * M
    if min_iter is None:
        min_iter = max(2 * check_every, max(1, n_eig // bs + 1))
    min_iter = int(min_iter)

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0, _ = jnp.linalg.qr(Q0)

    # +1 Krylov slot so the final block does not overwrite Q_{M-1} (P1).
    Q_all = jnp.zeros((M + 1, n, bs), dtype=jnp.complex128).at[0].set(Q0)
    alpha_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    last_evals = jnp.full((n_eig,), jnp.inf, dtype=jnp.float64)
    converged = jnp.bool_(False)

    def step(j, Q_all, alpha_all, beta_all):
        Q_j = Q_all[j]
        Z = matvec(Q_j.T).T
        alpha_j = jnp.conj(Q_j).T @ Z
        alpha_all = alpha_all.at[j].set(alpha_j)
        Z = Z - Q_j @ alpha_j
        Q_jm1 = Q_all[jnp.maximum(j - 1, 0)]
        beta_prev = beta_all[jnp.maximum(j - 1, 0)]
        Z = jnp.where(j > 0, Z - Q_jm1 @ jnp.conj(beta_prev).T, Z)

        def reorth_body(i, Z_acc):
            valid = i < j
            Q_i = Q_all[i]
            proj = jnp.where(
                valid, jnp.conj(Q_i).T @ Z_acc,
                jnp.zeros((bs, bs), dtype=Z_acc.dtype))
            return Z_acc - Q_i @ proj
        start = jnp.maximum(0, j - n_reorth)
        Z = lax.fori_loop(start, j + 1, reorth_body, Z)

        Q_next, beta_j = jnp.linalg.qr(Z)
        beta_all = beta_all.at[j].set(beta_j)
        Q_all = Q_all.at[j + 1].set(Q_next)
        return Q_all, alpha_all, beta_all

    def cond(state):
        j, _, _, _, _, conv = state
        return jnp.logical_and(j < M, jnp.logical_not(conv))

    def body(state):
        j, Q_all, alpha_all, beta_all, last_evals, _ = state
        Q_all, alpha_all, beta_all = step(j, Q_all, alpha_all, beta_all)

        # Convergence check — only every ``check_every`` iters and after
        # ``min_iter`` warmup. ``jax.lax.cond`` keeps both branches
        # constant-cost (no Python-level branching).
        do_check = jnp.logical_and(
            (j + 1) >= min_iter,
            ((j + 1) % check_every) == 0,
        )

        def _check_branch(args):
            alpha_all, beta_all, last_evals, j_done = args
            T = _build_block_tridiag(alpha_all, beta_all, M, bs)
            # Mask inactive (zero) part of T by adding a large constant
            # to its diagonal — pushes inactive eigvals out of the
            # spectrum so jnp.sort()[:n_eig] picks only real Ritz vals.
            LARGE = jnp.asarray(1.0e6, dtype=T.real.dtype)
            pos = jnp.arange(M * bs)
            active = (j_done + 1) * bs                        # completed iters × bs
            mask = (pos >= active).astype(T.real.dtype) * LARGE
            T = T + jnp.diag(mask).astype(T.dtype)
            ev = jnp.linalg.eigvalsh(T)
            ev = jnp.sort(ev)[:n_eig]
            scale = jnp.maximum(jnp.abs(ev), atol)
            delta = jnp.max(jnp.abs(ev - last_evals) / scale)
            new_conv = delta < rtol
            return ev, new_conv

        def _skip_branch(args):
            _, _, last_evals, _ = args
            return last_evals, jnp.bool_(False)

        new_evals, new_conv = lax.cond(
            do_check, _check_branch, _skip_branch,
            (alpha_all, beta_all, last_evals, j),
        )
        return (j + 1, Q_all, alpha_all, beta_all, new_evals, new_conv)

    init = (jnp.int32(0), Q_all, alpha_all, beta_all, last_evals, converged)
    j_final, Q_all, alpha_all, beta_all, _, _ = lax.while_loop(cond, body, init)

    # Final eigh — mask inactive blocks the same way as the convergence
    # check, otherwise the zero eigvals from unfilled iters dominate
    # ``sort()[:n_eig]``.
    T = _build_block_tridiag(alpha_all, beta_all, M, bs)
    LARGE_F = jnp.asarray(1.0e6, dtype=T.real.dtype)
    pos_F = jnp.arange(T_size)
    active_F = j_final * bs
    mask_F = (pos_F >= active_F).astype(T.real.dtype) * LARGE_F
    T = T + jnp.diag(mask_F).astype(T.dtype)
    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]
    Q_full = jnp.transpose(Q_all[:M], (1, 0, 2)).reshape(n, T_size)
    eigenvectors = (Q_full @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    return eigenvalues, eigenvectors, j_final

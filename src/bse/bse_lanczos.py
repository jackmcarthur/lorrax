"""Lanczos solvers for BSE."""
from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from .bse_serial import apply_bse_hamiltonian_single_device


def block_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    shape: Tuple[int, ...],
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    tol: float = 1e-8,
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """Block Lanczos for lowest eigenvalues."""
    n_flat = np.prod(shape)
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
        beta_j = R.T
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
) -> Tuple[jax.Array, jax.Array]:
    """Simple Lanczos (Python loop)."""
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
) -> Tuple[jax.Array, jax.Array]:
    """JIT Lanczos using lax.fori_loop."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    q0 = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q0 = q0 + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q0 = q0 / jnp.linalg.norm(q0)

    Q = jnp.zeros((n, max_iter), dtype=jnp.complex128)
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
        Q = Q.at[:, jnp.minimum(j + 1, max_iter - 1)].set(q_next)

        return (Q, alpha, beta, q_next)

    init_carry = (Q, alpha, beta, q0)
    Q, alpha, beta, _ = lax.fori_loop(0, max_iter, lanczos_step, init_carry)

    T = jnp.diag(alpha)
    off_diag = beta[:-1]
    T = T + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]

    eigenvectors = (Q @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)

    return eigenvalues, eigenvectors


def solve_bse(
    psi_c: jax.Array,
    psi_v: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
    n_eig: int = 20,
    max_iter: int = 100,
    use_block: bool = False,
    block_size: int = 4,
    use_jit_lanczos: bool = True,
    n_reorth: int = 10,
    include_W: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Solve BSE for lowest exciton eigenvalues."""
    nk, nc, _, _ = psi_c.shape
    nv = psi_v.shape[1]
    shape = (nc, nv, nk)
    n_flat = nc * nv * nk

    @partial(jax.jit, static_argnames=("nkx", "nky", "nkz", "include_W"))
    def _matvec_impl(v, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz, include_W):
        X = v.reshape(1, nc, nv, nk)
        HX = apply_bse_hamiltonian_single_device(
            X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz, include_W
        )
        return HX.reshape(-1)

    matvec_flat = partial(
        _matvec_impl,
        psi_c=psi_c,
        psi_v=psi_v,
        eps_c=eps_c,
        eps_v=eps_v,
        W_q=W_q,
        V_q0=V_q0,
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        include_W=include_W,
    )

    matvec_block = partial(
        lambda X: apply_bse_hamiltonian_single_device(
            X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz, include_W
        )
    )

    if use_block:
        eigenvalues, eigenvectors = block_lanczos_eig(
            matvec_block, shape, n_eig=n_eig, block_size=block_size, max_iter=max_iter
        )
    elif use_jit_lanczos:
        eigenvalues, eigenvectors = lanczos_eig_jit(
            matvec_flat, n_flat, n_eig=n_eig, max_iter=max_iter, n_reorth=n_reorth
        )
        eigenvectors = eigenvectors.reshape(n_eig, *shape)
    else:
        eigenvalues, eigenvectors = simple_lanczos_eig(
            matvec_flat, n_flat, n_eig=n_eig, max_iter=max_iter
        )
        eigenvectors = eigenvectors.reshape(n_eig, *shape)

    return eigenvalues, eigenvectors

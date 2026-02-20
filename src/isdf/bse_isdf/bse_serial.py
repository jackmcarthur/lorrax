"""Single-device BSE kernels and utilities."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .bse_preconditioner import energy_diff_cv_k


def symmetrize_W_q(W_q: jax.Array, nkx: int, nky: int, nkz: int) -> jax.Array:
    """Symmetrize W(q) to enforce W(q) = W(-q)†."""
    qx = jnp.arange(nkx)
    qy = jnp.arange(nky)
    qz = jnp.arange(nkz)

    minus_qx = (nkx - qx) % nkx
    minus_qy = (nky - qy) % nky
    minus_qz = (nkz - qz) % nkz

    W_minus_q = W_q[:, :, minus_qx[:, None, None], minus_qy[None, :, None], minus_qz[None, None, :]]
    W_minus_q_dag = jnp.conj(W_minus_q).transpose(1, 0, 2, 3, 4)
    return (W_q + W_minus_q_dag) / 2


def compute_pair_amplitude(psi_c: jax.Array, psi_v: jax.Array) -> jax.Array:
    """M(k,c,v,μ) = Σ_s conj(ψ_c[k,c,s,μ]) * ψ_v[k,v,s,μ]."""
    return jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c), psi_v)


def apply_D(X: jax.Array, eps_c: jax.Array, eps_v: jax.Array) -> jax.Array:
    """Apply diagonal term (ε_c - ε_v) * X."""
    delta_E = energy_diff_cv_k(eps_c, eps_v)[None, :, :, :]
    return delta_E * X


def apply_bse_hamiltonian_single_device(
    X: jax.Array,
    psi_c: jax.Array,
    psi_v: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    """Single-device BSE Hamiltonian for verification."""
    nk = nkx * nky * nkz
    nspinor = psi_c.shape[2]
    n_rmu = psi_c.shape[3]

    delta_E = energy_diff_cv_k(eps_c, eps_v)[None, :, :, :]
    D_term = delta_E * X

    M = compute_pair_amplitude(psi_c, psi_v)
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=jnp.float64))

    S_V = jnp.einsum("kcvN,bcvk->bNk", M, X) / sqrt_nk
    U_V = jnp.einsum("MN,bNk->bMk", V_q0, S_V)
    V_term = jnp.einsum("kcvM,bMk->bcvk", jnp.conj(M), U_V) / sqrt_nk

    R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v), X)
    T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c, R)
    T_k = T.reshape(X.shape[0], n_rmu, n_rmu, nspinor, nspinor, nkx, nky, nkz)
    T_R = jnp.fft.ifftn(T_k, axes=(5, 6, 7), norm="ortho")
    W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
    U_R = W_R[None, :, :, None, None, :, :, :] * T_R
    U_q = jnp.fft.fftn(U_R, axes=(5, 6, 7), norm="ortho")
    U = U_q.reshape(X.shape[0], n_rmu, n_rmu, nspinor, nspinor, nk)
    A = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c), U)
    W_term = jnp.einsum("kvsN,bcNsk->bcvk", psi_v, A) / sqrt_nk

    return D_term + V_term - W_term


@partial(jax.jit, static_argnums=(6, 7, 8))
def apply_bse_hamiltonian_single_device_jit(
    X: jax.Array,
    psi_c: jax.Array,
    psi_v: jax.Array,
    eps_c: jax.Array,
    eps_v: jax.Array,
    W_q: jax.Array,
    V_q0: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    return apply_bse_hamiltonian_single_device(
        X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz
    )

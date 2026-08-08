"""Single-device BSE kernels and utilities."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from common.fft_helpers import local_fftn3, local_ifftn3
from .bse_preconditioner import energy_diff_cv_k


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
    include_W: bool = True,
) -> jax.Array:
    """Single-device BSE Hamiltonian for verification."""
    nk = nkx * nky * nkz
    nspinor = psi_c.shape[2]
    n_rmu = psi_c.shape[3]

    delta_E = energy_diff_cv_k(eps_c, eps_v)[None, :, :, :]
    D_term = delta_E * X

    M = compute_pair_amplitude(psi_c, psi_v)
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

    # Exchange (q=0) is DENSE in (k,k') — <cvk|K^x|c'v'k'> couples every k to
    # every k' through the single V_q0 tile (VERDICT.md).  Encode k-SUMS into a
    # k-free transition-Hartree density S[b,nu] = (1/sqrt_Nk) sum_{k'c'v'} M X,
    # one V_q0 solve, then the decode broadcasts back at every k.  The two
    # 1/sqrt_Nk compose to the 1/Nk the dense formula requires.
    #
    # Conjugation: the transition density <0|rho|Psi> = sum A_cvk psi_c psi_v*
    # puts the CONJUGATED vertex on the forward (encode) leg and the bare vertex
    # on the back-contract, i.e. K^x = M V M^dagger with M = compute_pair_amplitude.
    # The reverse assignment builds conj(M) V M^T = conj(K^x), which no symmetry
    # operator can commute with alongside the (correct) W term.
    S_V = jnp.einsum("kcvN,bcvk->bN", jnp.conj(M), X) / sqrt_nk
    U_V = jnp.einsum("MN,bN->bM", V_q0, S_V)
    V_term = jnp.einsum("kcvM,bM->bcvk", M, U_V) / sqrt_nk

    if not include_W:
        return D_term + V_term

    R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v), X)
    T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c, R)
    # FFTs through the fft_helpers local kernels ("ONE source for the
    # local FFT") — single-device reference, values byte-identical.
    T_k = T.reshape(X.shape[0], n_rmu, n_rmu, nspinor, nspinor, nkx, nky, nkz)
    T_R = local_ifftn3(T_k, axes=(5, 6, 7), norm="ortho")
    W_R = local_ifftn3(W_q, axes=(2, 3, 4), norm="ortho")
    U_R = W_R[None, :, :, None, None, :, :, :] * T_R
    U_q = local_fftn3(U_R, axes=(5, 6, 7), norm="ortho")
    U = U_q.reshape(X.shape[0], n_rmu, n_rmu, nspinor, nspinor, nk)
    A = jnp.einsum("kctM,bMNtsk->bcNsk", jnp.conj(psi_c), U)
    W_term = jnp.einsum("kvsN,bcNsk->bcvk", psi_v, A) / sqrt_nk

    return D_term + V_term - W_term


@partial(jax.jit, static_argnums=(7, 8, 9, 10))
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
    include_W: bool = True,
) -> jax.Array:
    return apply_bse_hamiltonian_single_device(
        X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz, include_W
    )

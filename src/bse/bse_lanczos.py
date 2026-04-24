"""Lanczos solvers for BSE.

The generic Lanczos algorithms live in solvers.lanczos.  This module
re-exports them and provides the BSE-specific solve_bse wrapper that
builds the matvec from BSE physics arrays.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from solvers.lanczos import (
    block_lanczos_eig,
    simple_lanczos_eig,
    lanczos_eig_jit,
)
from .bse_serial import apply_bse_hamiltonian_single_device, compute_pair_amplitude


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

    # Precompute matvec-invariants once so the Lanczos hot loop doesn't
    # redo an ifftn on W_q (200× redundancy on 200-iter Lanczos) or an
    # einsum pair-amplitude on psi_c,psi_v.  Both depend only on fixed
    # GW/wfn inputs, not on the Lanczos vector v.  NB: precomputing
    # conj(psi_c,psi_v) actually slowed things down (jit closure cost
    # beat einsum-fused conj savings) so we leave those inlined.
    if include_W:
        W_R = jax.jit(lambda W: jnp.fft.ifftn(W, axes=(2, 3, 4), norm="ortho"))(W_q)
    else:
        W_R = jnp.zeros_like(W_q)
    M = jax.jit(compute_pair_amplitude)(psi_c, psi_v)

    @partial(jax.jit, static_argnames=("nkx", "nky", "nkz", "include_W"), donate_argnums=(0,))
    def _matvec_impl(v, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, W_R, M,
                     nkx, nky, nkz, include_W):
        X = v.reshape(1, nc, nv, nk)
        HX = apply_bse_hamiltonian_single_device(
            X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz, include_W,
            W_R=W_R, M=M,
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
        W_R=W_R,
        M=M,
        nkx=nkx,
        nky=nky,
        nkz=nkz,
        include_W=include_W,
    )

    matvec_block = partial(
        lambda X: apply_bse_hamiltonian_single_device(
            X, psi_c, psi_v, eps_c, eps_v, W_q, V_q0, nkx, nky, nkz, include_W,
            W_R=W_R, M=M,
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

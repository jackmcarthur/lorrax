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

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_jittable_local_ifftn_3d

from solvers.lanczos import (
    block_lanczos_eig,
    block_lanczos_eig_jit,
    block_lanczos_eig_jit_converged,
    simple_lanczos_eig,
    lanczos_eig_jit,
)
from .bse_serial import apply_bse_hamiltonian_single_device
from .bse_ring_comm import build_bse_ring_matvec, make_bse_shardings


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


def solve_bse_sharded(
    data: dict,
    mesh_xy: Mesh,
    *,
    n_eig: int = 20,
    max_iter: int = 200,
    n_reorth: int = 10,
    include_W: bool = True,
    block_size: int = 1,
    rtol: float = 0.0,
    atol: float = 1e-8,
    check_every: int = 4,
) -> Tuple[jax.Array, jax.Array]:
    """Sharded BSE Lanczos using the (μ,ν) ring matvec.

    Drop-in faster replacement for ``solve_bse`` when the mesh has more
    than one device. Reuses ``bse_ring_comm.build_bse_ring_matvec`` —
    same kernel as FEAST, so the per-iteration matvec
    (a) parallelises over (px·py) GPUs,
    (b) precomputes ``W_R = ifft(W_q)`` once outside the Lanczos loop
        instead of per-iter (the dominant cost in the single-device
        path — 200 × 3D-FFTs of an (n_μ, n_μ, nkx,nky,nkz) tensor).

    Vector layout:
        Trial vector ``X`` is shape ``(1, n_cond_pad, n_val_pad, nk)``
        sharded ``P(None, "x", "y", None)``. The Lanczos solver
        operates on a flat ``(n_flat,)`` representation; a thin
        bridge reshapes + applies ``with_sharding_constraint`` once
        per matvec, so collectives inside the matvec remain on the
        ``(x, y)`` mesh.

    The data dict must come from ``bse_io.load_bse_data_from_restart_sharded``
    and contains psi_{c,v}_{X,Y}, eps_c, eps_v, V_q0 (P("x","y")), W_q
    (P("x","y",None,None,None)), plus any q=0 head injection already
    applied at load time.
    """
    sh = make_bse_shardings(mesh_xy)
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    nkx = int(data["nkx"])
    nky = int(data["nky"])
    nkz = int(data["nkz"])
    nk = nkx * nky * nkz
    n_flat = nc_pad * nv_pad * nk
    bs = int(block_size)
    shape = (bs, nc_pad, nv_pad, nk)

    matvec_ring = build_bse_ring_matvec(
        mesh_xy, nkx, nky, nkz, include_W=include_W,
    )

    # W_R = ifft_q(W_q) computed ONCE inside the outer jit. Use the
    # gw_jax custom-partitioned IFFT helper — plain ``jnp.fft.ifftn`` on
    # a sharded tensor inserts a 337-MiB all-gather around the FFT under
    # current JAX even when the FFT axes are unsharded; the helper hides
    # the FFT in an opaque primitive so XLA only sees a per-device local
    # FFT (axes (2,3,4) of W_q are replicated; (μ,ν) stay on x,y).
    if include_W:
        _W_local_ifftn = make_jittable_local_ifftn_3d(
            mesh_xy, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm='ortho')
    else:
        _W_local_ifftn = None

    rep_eig = NamedSharding(mesh_xy, P())  # eigenvalues / eigenvectors come back replicated.

    # End-to-end jit with explicit in/out shardings + donate the bulky
    # buffers we won't need post-Lanczos.
    @partial(
        jax.jit,
        in_shardings=(
            sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
            sh.eps, sh.eps, sh.W, sh.V,
        ),
        out_shardings=(rep_eig, rep_eig, rep_eig),
        donate_argnums=(6,),  # W_q — only used to build W_R
    )
    def _full_run(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_q, V_q0):
        if include_W:
            W_R = _W_local_ifftn(W_q)
        else:
            W_R = W_q
        if bs == 1:
            # Single-vector matvec — accept (n_flat,) and reshape to (1, c, v, k).
            def matvec(v_flat):
                X = v_flat.reshape(shape)
                X = jax.lax.with_sharding_constraint(X, sh.X)
                HX = matvec_ring(
                    X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                    eps_c, eps_v, W_R, V_q0,
                )
                return HX.reshape(-1)
            evs, evecs = lanczos_eig_jit(
                matvec, n_flat, n_eig=n_eig, max_iter=max_iter, n_reorth=n_reorth,
            )
            return evs, evecs, jnp.int32(max_iter)
        else:
            # Block matvec — accept (block_size, n_flat) and reshape to
            # (block_size, c, v, k). Each call processes ``block_size``
            # vectors at once → ``block_size``-larger GEMMs (better GPU
            # occupancy) and ``block_size``-fewer host dispatches.
            def matvec_block(V_block):
                X = V_block.reshape(shape)
                X = jax.lax.with_sharding_constraint(X, sh.X)
                HX = matvec_ring(
                    X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                    eps_c, eps_v, W_R, V_q0,
                )
                return HX.reshape(bs, -1)
            if rtol > 0.0:
                # Convergence-driven: ``lax.while_loop`` exits when the
                # n_eig lowest Ritz values stabilise within ``rtol``.
                return block_lanczos_eig_jit_converged(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=bs, max_iter=max_iter,
                    rtol=rtol, atol=atol, check_every=check_every,
                    n_reorth=n_reorth,
                )
            else:
                evs, evecs = block_lanczos_eig_jit(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=bs, max_iter=max_iter, n_reorth=n_reorth,
                )
                return evs, evecs, jnp.int32(max_iter)

    eigenvalues, eigenvectors, n_iter_done = _full_run(
        data["psi_c_X"], data["psi_c_Y"],
        data["psi_v_X"], data["psi_v_Y"],
        data["eps_c"], data["eps_v"],
        data["W_q"], data["V_q0"],
    )
    eigenvectors = eigenvectors.reshape(n_eig, 1, nc_pad, nv_pad, nk)
    return eigenvalues, eigenvectors, n_iter_done

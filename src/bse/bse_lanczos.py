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

from common.fft_helpers import make_sharded_ifftn_3d

from solvers.lanczos import (
    block_lanczos_eig,
    block_lanczos_eig_jit,
    block_lanczos_eig_jit_converged,
    simple_lanczos_eig,
    lanczos_eig_jit,
)
from .bse_serial import apply_bse_hamiltonian_single_device
from .bse_ring_comm import make_bse_shardings


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
    solver_kind: str = "lanczos",
    davidson_n_random_init: int = 5,
    davidson_eps_shift_Ry: float = 1e-3,
    tda: bool = True,
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

    ``tda`` (default True) selects the Tamm-Dancoff resonant-only Hamiltonian
    (the block/single-vector Lanczos + Davidson paths below).  ``tda=False``
    dispatches to the structure-preserving full-BSE eigensolver
    ``bse_nontda.solve_bse_nontda_sharded`` — the ONE non-TDA seam, no parallel
    solver stack — which returns paired (X, Y) eigenvectors (X^H X - Y^H Y = +1).
    """
    if not tda:
        from .bse_nontda import solve_bse_nontda_sharded
        return solve_bse_nontda_sharded(
            data, mesh_xy, n_eig=n_eig, include_W=include_W)

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

    # Trial-stack matvec (scan-inside-shard_map): applies H to the whole
    # (block_size / Davidson subspace) stack with ONE T-tensor alive regardless
    # of the stack width — strictly better peak memory than the legacy
    # ring/gather/simple matvecs (whose T scaled with the stack). Same
    # (X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0) -> HX
    # signature, so it is a drop-in for both the bs==1 and block paths below.
    # The legacy ``matvec_kind`` selector is retired here; see the retirement
    # note in bse_stack_matvec (ring/gather/simple + selector deletion pending).
    from .bse_stack_matvec import build_bse_stack_matvec
    matvec_ring = build_bse_stack_matvec(
        mesh_xy, nkx, nky, nkz, kernel="bse" if include_W else "rpa",
    )

    # W_R = ifft_q(W_q) computed ONCE inside the outer jit. Use the
    # gw_jax custom-partitioned IFFT helper — plain ``jnp.fft.ifftn`` on
    # a sharded tensor inserts a 337-MiB all-gather around the FFT under
    # current JAX even when the FFT axes are unsharded; the helper hides
    # the FFT in an opaque primitive so XLA only sees a per-device local
    # FFT (axes (2,3,4) of W_q are replicated; (μ,ν) stay on x,y).
    if include_W:
        # 3D cuFFT in one shot rather than 3 sequential 1D custom_partitioning
        # calls (the older ``make_jittable_local_ifftn_3d``).  Same correctness
        # constraint (FFT axes replicated); ~5–10× fewer transposes in the
        # generated HLO.
        _W_local_ifftn = make_sharded_ifftn_3d(
            mesh_xy, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm='ortho')
    else:
        _W_local_ifftn = None

    # ── Davidson path: doesn't fit inside a single jit wrap (Python-side
    # iteration + on-the-fly Ritz solve), so build matvec + W_R outside and
    # delegate to the shape-agnostic ``solvers.davidson.davidson``.  Returns
    # the same `(eigenvalues, eigenvectors, n_iter_done)` tuple as the
    # Lanczos path so callers don't branch.
    if solver_kind == "davidson":
        from solvers.davidson import davidson, warmup_davidson_jit
        from .bse_davidson_helpers import bse_diagonal_precond, init_bse_subspace

        psi_c_X = data["psi_c_X"]; psi_c_Y = data["psi_c_Y"]
        psi_v_X = data["psi_v_X"]; psi_v_Y = data["psi_v_Y"]
        eps_c   = data["eps_c"];   eps_v   = data["eps_v"]
        V_q0    = data["V_q0"];    W_q     = data["W_q"]
        M_X     = data["M_X"];     M_Y     = data["M_Y"]  # hoisted V-term pair-amps (P3)
        W_R = _W_local_ifftn(W_q) if include_W else W_q

        def apply_H(V):    # V: (m, nc_pad, nv_pad, nk) sharded P(None,"x","y",None)
            V = jax.lax.with_sharding_constraint(V, sh.X)
            return matvec_ring(V, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                               eps_c, eps_v, W_R, V_q0, M_X, M_Y)

        bse_sharding = NamedSharding(mesh_xy, P(None, "x", "y", None))
        precond_fn = bse_diagonal_precond(
            eps_c, eps_v, sharding=NamedSharding(mesh_xy, P("x", "y", None)),
            epsilon_shift=davidson_eps_shift_Ry)
        X0 = init_bse_subspace(
            eps_c, eps_v, n_eig=n_eig, n_random=davidson_n_random_init,
            mesh=mesh_xy, sharding=bse_sharding)

        # Pre-compile _ritz_and_residuals at every subspace size m ∈ {n_eig,
        # 2·n_eig, …, m_max} so the Davidson loop does not pay 4 separate
        # XLA compiles as the subspace grows between restarts. Compiles
        # ~2 s otherwise; with warmup this is a one-time up-front cost.
        m_max_warm = 4 * n_eig
        warmup_davidson_jit(
            n_eig=n_eig,
            trailing_shape=tuple(X0.shape[1:]),
            m_max=m_max_warm,
            dtype=X0.dtype,
            sharding=bse_sharding,
        )

        eigenvalues, eigenvectors = davidson(
            apply_H, n_eig=n_eig, precond_fn=precond_fn, X0=X0,
            max_iter=max_iter, tol=atol if atol > 0 else 1e-8,
            verbose=True,
        )
        # Match Lanczos return shape: (n_eig, bs=1, nc_pad, nv_pad, nk).
        eigenvectors = eigenvectors.reshape(n_eig, 1, nc_pad, nv_pad, nk)
        return eigenvalues, eigenvectors, jnp.int32(max_iter)

    rep_eig = NamedSharding(mesh_xy, P())  # eigenvalues / eigenvectors come back replicated.

    # End-to-end jit with explicit in/out shardings + donate the bulky
    # buffers we won't need post-Lanczos.
    @partial(
        jax.jit,
        in_shardings=(
            sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
            sh.eps, sh.eps, sh.W, sh.V, sh.psi_x, sh.psi_y,
        ),
        out_shardings=(rep_eig, rep_eig, rep_eig),
        # NB: W_q (arg 6) is NOT donated — W_R = ifft(W_q) is a fresh buffer with no
        # aliasable same-shape output, so the donation was always declined (cosmetic,
        # no copy) and only emitted a "donated buffers not usable" warning (audit P5).
    )
    def _full_run(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_q, V_q0, M_X, M_Y):
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
                    eps_c, eps_v, W_R, V_q0, M_X, M_Y,
                )
                return HX.reshape(-1)
            if rtol > 0.0:
                # Convergence-driven path: route through the block-Lanczos
                # while_loop with bs=1 (mathematically the same as a
                # single-vector Lanczos with early exit).
                def matvec_block(V_block):
                    return matvec(V_block.reshape(-1)).reshape(1, -1)
                return block_lanczos_eig_jit_converged(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=1, max_iter=max_iter,
                    rtol=rtol, atol=atol, check_every=check_every,
                    n_reorth=n_reorth,
                )
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
                    eps_c, eps_v, W_R, V_q0, M_X, M_Y,
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
        data["M_X"], data["M_Y"],
    )
    eigenvectors = eigenvectors.reshape(n_eig, 1, nc_pad, nv_pad, nk)
    return eigenvalues, eigenvectors, n_iter_done

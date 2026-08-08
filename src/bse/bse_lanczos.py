"""Lanczos solvers for BSE.

The generic Lanczos algorithms live in solvers.lanczos.  This module
re-exports them and provides the BSE-specific solve_bse wrapper that
builds the matvec from BSE physics arrays.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_sharded_ifftn_3d

from solvers.lanczos import (
    FULL_REORTH,
    reorth_kind,
    alpha_herm_sink,
    block_lanczos_eig,
    block_lanczos_eig_jit,
    block_lanczos_eig_jit_converged,
    report_alpha_herm,
    simple_lanczos_eig,
    split_alpha_sink,
    lanczos_eig_jit,
)
from .bse_serial import apply_bse_hamiltonian_single_device
from .bse_ring_comm import make_bse_shardings
import common.timing as timing


REORTH_ENV = "LORRAX_LANCZOS_REORTH"


def reorth_route() -> str:
    """Read the Lanczos reorthogonalisation dial and validate it.

    THE ENVIRONMENT IS READ HERE, not in ``solvers.lanczos``.  ``solvers`` is
    L2 -- physics-agnostic mathematics that
    ``tests/test_layering.py::test_no_l2_module_reads_the_environment``
    requires to be a function of its arguments -- so the solver takes a route
    TOKEN and this module owns the variable, exactly as
    ``bse_stack_matvec.matvec_opts`` owns ``LORRAX_BSE_MATVEC_OPT`` one layer
    above the kernels it steers.

    Unset/empty selects the default (batched ``cgs2``: ``2*max_iter``
    collectives).  ``mgs`` is the legacy per-vector sweep --
    ``max_iter(max_iter+1)/2`` collectives, 20 100 of them on the Si record
    deck at 200 iterations -- kept reachable for bisects and for reproducing
    pre-2026-08-08 runs.  An unknown token REFUSES: a perf dial that can be
    misspelled into a silent no-op makes every A/B built on it void, and now
    that ``cgs2`` is the default a typo must not hand back the slow route
    either.  ``reorth_kind`` raises; the message names this variable.
    """
    return reorth_kind(os.environ.get(REORTH_ENV, ""))


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
    n_reorth: int = FULL_REORTH,
    include_W: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Solve BSE for lowest exciton eigenvalues."""
    _reorth = reorth_route()
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
            matvec_flat, n_flat, n_eig=n_eig, max_iter=max_iter,
            n_reorth=n_reorth, reorth=_reorth,
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
    n_reorth: int = FULL_REORTH,
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

    # Resolve the reorth dial ONCE here (L1 reads the env; the L2 solver
    # takes the token) -- same shape as ``krylov_rep`` below.
    _reorth = reorth_route()
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
    with timing.section("bse.solve.matvec_build"):
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

    # ── W_R = ifft_q(W_q) at a REAL top-level dispatch boundary, W_q DONATED ──
    # This used to run INSIDE ``_full_run`` (and inside the Davidson block).
    # There it can never free W_q: W_q is a jit PARAMETER, so its buffer is
    # owned by the caller for the whole call and XLA has no same-shape OUTPUT
    # to alias it to — the in-code note at the old ``_full_run`` decorator
    # records exactly that ("donation was always declined").  The result was
    # BOTH W_q and W_R resident for the entire Lanczos: 2 x (μ_pad/p_x) x
    # (ν_pad/p_y) x nk x 16 bytes per rank (2 x 404 MB at μ=10015 / P=64;
    # ∝ μ²/P, so 2 x 4.1 GB at μ=32k).  Hoisting the transform into its own
    # donated jit lets XLA alias W_R onto W_q's buffer and lets the caller
    # drop the Python reference, so only ONE copy survives into the solve.
    # Value-identical: same helper, same axes, same norm, same operand.
    if include_W:
        with timing.section("bse.solve.W_ifft") as _sec_wifft:
            _W_ifft_donated = jax.jit(_W_local_ifftn, donate_argnums=(0,))
            W_R = _W_ifft_donated(data["W_q"])
            data["W_q"] = None          # release the caller-side reference
            _sec_wifft.watch(W_R)
    else:
        W_R = data["W_q"]

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
        V_q0    = data["V_q0"]
        M_X     = data["M_X"];     M_Y     = data["M_Y"]  # hoisted V-term pair-amps (P3)
        # W_R already built above (donated top-level ifft).

        def apply_H(V):    # V: (m, nc_pad, nv_pad, nk) sharded P(None,"x","y",None)
            V = jax.lax.with_sharding_constraint(V, sh.X)
            return matvec_ring(V, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                               eps_c, eps_v, W_R, V_q0, M_X, M_Y)

        bse_sharding = NamedSharding(mesh_xy, P(None, "x", "y", None))
        precond_fn = bse_diagonal_precond(
            eps_c, eps_v, sharding=NamedSharding(mesh_xy, P("x", "y", None)),
            epsilon_shift=davidson_eps_shift_Ry)
        X0 = init_bse_subspace(
            eps_c, eps_v, n_eig=n_eig,
            n_cond=int(data["n_cond"]), n_val=int(data["n_val"]),
            n_random=davidson_n_random_init,
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
    # ``krep``: the Krylov basis sharding, or None to leave it to GSPMD (the
    # shipped behaviour).  Resolved ONCE here, not per matvec call.
    from .bse_stack_matvec import matvec_opts as _mv_opts
    krylov_rep = rep_eig if "krep" in _mv_opts() else None

    # The static half of the α-Hermiticity reports the Krylov solve collects
    # (solver name + α form).  Filled at TRACE time by ``_full_run`` below and
    # read back on the host after the call; the numbers themselves come back
    # as ``_full_run``'s fourth output.  See ``solvers.lanczos``'s header for
    # why the report may not be a ``jax.debug.callback`` here: a host callback
    # makes this module — the single program holding the whole Krylov loop —
    # unpersistable in JAX's compilation cache, so the 2.1 s XLA compile was
    # paid on every warm run.
    _alpha_labels: list = []

    # End-to-end jit with explicit in/out shardings + donate the bulky
    # buffers we won't need post-Lanczos.
    @partial(
        jax.jit,
        in_shardings=(
            sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
            sh.eps, sh.eps, sh.W, sh.V, sh.psi_x, sh.psi_y,
        ),
        # Fourth entry covers the α-Hermiticity payload: a pytree prefix, so
        # ``rep_eig`` applies to each of its (replicated, scalar) leaves.
        out_shardings=(rep_eig, rep_eig, rep_eig, rep_eig),
        # NB: arg 6 is now W_R (already ifft'd, DONATED at its own top-level
        # boundary above) rather than W_q.  Donating it HERE is still declined
        # — there is no aliasable same-shape output of a Lanczos solve — which
        # is the original audit-P5 observation; the fix was to move the
        # transform out, not to donate the eigensolve's inputs.
    )
    def _full_run(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X, M_Y):
        with alpha_herm_sink() as _sink:
            evs, evecs, n_it = _krylov(
                psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                W_R, V_q0, M_X, M_Y,
            )
        labels, payload = split_alpha_sink(_sink)
        _alpha_labels[:] = labels
        return evs, evecs, n_it, payload

    def _krylov(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X, M_Y):
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
                    n_reorth=n_reorth, reorth=_reorth,
                )
            evs, evecs = lanczos_eig_jit(
                matvec, n_flat, n_eig=n_eig, max_iter=max_iter,
                n_reorth=n_reorth, reorth=_reorth,
            )
            return evs, evecs, jnp.int32(max_iter)
        else:
            # Block matvec — accept (block_size, n_flat) and reshape to
            # (block_size, c, v, k). Each call processes ``block_size``
            # vectors at once → ``block_size``-larger GEMMs (better GPU
            # occupancy) and ``block_size``-fewer host dispatches.
            def matvec_block(V_block):
                # ``krep`` (LORRAX_BSE_MATVEC_OPT): pin the FLAT Krylov axis to
                # replicated.  Unconstrained it inherits ``sh.X``'s tiling
                # through the reshape, which puts every reorthogonalisation dot
                # product, the QR and the Ritz eigh on a sharded axis and turns
                # each into a collective.  Costs one replicated
                # (max_iter+1, n_flat, bs) basis per rank — right for a small
                # pair space, wrong for a large one; see the dial docs.
                if krylov_rep is not None:
                    V_block = jax.lax.with_sharding_constraint(V_block,
                                                               krylov_rep)
                X = V_block.reshape(shape)
                X = jax.lax.with_sharding_constraint(X, sh.X)
                HX = matvec_ring(
                    X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                    eps_c, eps_v, W_R, V_q0, M_X, M_Y,
                )
                HX = HX.reshape(bs, -1)
                if krylov_rep is not None:
                    HX = jax.lax.with_sharding_constraint(HX, krylov_rep)
                return HX
            if rtol > 0.0:
                # Convergence-driven: ``lax.while_loop`` exits when the
                # n_eig lowest Ritz values stabilise within ``rtol``.
                return block_lanczos_eig_jit_converged(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=bs, max_iter=max_iter,
                    rtol=rtol, atol=atol, check_every=check_every,
                    n_reorth=n_reorth, reorth=_reorth,
                )
            else:
                evs, evecs = block_lanczos_eig_jit(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=bs, max_iter=max_iter, n_reorth=n_reorth,
                    reorth=_reorth,
                )
                return evs, evecs, jnp.int32(max_iter)

    # ONE program: trace + XLA compile + the entire Krylov loop's execution.
    # Split compile from execution by lowering/compiling first, so the two
    # costs the campaign A/Bs are separately visible instead of fused into
    # ``bse.eigensolve``.  Value-identical: the same jit, the same operands.
    _args = (
        data["psi_c_X"], data["psi_c_Y"],
        data["psi_v_X"], data["psi_v_Y"],
        data["eps_c"], data["eps_v"],
        W_R, data["V_q0"],
        data["M_X"], data["M_Y"],
    )
    with timing.section("bse.solve.krylov_compile"):
        _full_run_c = _full_run.lower(*_args).compile()
    with timing.section("bse.solve.krylov_run") as _sec_kr:
        eigenvalues, eigenvectors, n_iter_done, _alpha = _full_run_c(*_args)
        _sec_kr.watch(eigenvalues)
        _sec_kr.watch(eigenvectors)
    # The α-Hermiticity gate, run on the host on the scalars the jit just
    # returned.  Identical check to the in-jit callback it replaces; it is
    # OUT here so that ``jit__full_run`` carries no host callback and JAX will
    # persist it (a cached module is ~2.1 s of XLA compile per warm run).
    report_alpha_herm(_alpha_labels, _alpha)
    eigenvectors = eigenvectors.reshape(n_eig, 1, nc_pad, nv_pad, nk)
    return eigenvalues, eigenvectors, n_iter_done

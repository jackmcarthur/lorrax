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
import numpy as np

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import replicate_to_mesh
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

#: Third element of ``solve_bse_sharded``'s return tuple when the route ran a
#: FIXED number of iterations and therefore never measured a convergence point.
#:
#: Three of the four routes here (``lanczos_eig_jit``, ``block_lanczos_eig_jit``
#: and ``davidson``) run their full iteration budget unconditionally.  They used
#: to return ``jnp.int32(max_iter)``, which is the BUDGET wearing a
#: MEASUREMENT's name: a caller reading ``n_iter_done`` cannot tell "converged
#: after 200" from "was told to do 200".  Only
#: ``block_lanczos_eig_jit_converged`` (``rtol > 0``) returns a real count.
#:
#: The honest form is a sentinel that cannot be mistaken for a count.  Callers
#: that want the budget ask for the budget, through ``iters_reported`` below --
#: which is the only place the two meanings are allowed to meet.
N_ITER_NOT_MEASURED = -1


def iters_reported(n_iter_done, budget: int) -> int:
    """Map ``n_iter_done`` onto something safe to print or compare.

    Returns the measured count when the route measured one, and the iteration
    ``budget`` when it did not.  Centralised so "how many iterations did it
    run" has ONE answer and the sentinel can never leak into arithmetic as
    ``-1`` (which is what a bare ``int(n_iter_done)`` would hand a caller
    computing e.g. ``n_done * block_size``).
    """
    n = int(n_iter_done)
    return int(budget) if n < 0 else n


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
    davidson_m_max: int | None = None,
    davidson_precond: str = "bare",
    davidson_olsen: bool = False,
    trlan_m_max: int | None = None,
    trlan_n_keep: int | None = None,
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
    # takes the token) -- resolved in ONE place, like every other dial.
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

    # ``auto`` is resolved to one of the two real routes HERE, before anything
    # else reads the flag, so every later branch sees only "bare" or "exact".
    if davidson_precond == "auto":
        from .bse_davidson_helpers import resolve_precond_route
        _resolved = resolve_precond_route("auto", n_flat)
        print(f"Davidson preconditioner: auto -> {_resolved} "
              f"(bse_dim={n_flat})", flush=True)
        davidson_precond = _resolved

    # The q=0 slice of W_q, kept for the exact-diagonal preconditioner.  It
    # must be taken HERE, before the donated ifft below consumes W_q: deriving
    # it from W_R instead would make the answer depend on that ifft's norm
    # convention, and it is a (mu, nu) slice of a (mu, nu, 4, 4, 4) tensor, so
    # keeping it costs 1/64 of one W_q.
    # STASHED ON THE PAYLOAD, not held in a local.  Two reasons, both
    # measured on this branch (PRECOND_BUILD_FREE.md):
    #  * the donated ifft below sets ``data["W_q"] = None``, so a SECOND
    #    eigensolve over the same payload used to raise here — ``None`` is not
    #    subscriptable.  The exact route was single-shot per payload;
    #  * the exact-diagonal build memoises on operand IDENTITY, so a fresh
    #    slice object every call would miss every time.  One stashed slice
    #    makes the second and later builds free.
    # It costs 1/64 of one W_q and is dropped with the payload.
    _need_W_q0 = (include_W and solver_kind == "davidson"
                  and davidson_precond == "exact")
    if _need_W_q0 and data.get("_W_q0_for_precond") is None:
        if data.get("W_q") is None:
            raise ValueError(
                "--davidson-precond exact needs the q=0 slice of W_q, but "
                "this payload's W_q has already been consumed by the donated "
                "ifft and no slice was stashed.  Reload the payload.")
        data["_W_q0_for_precond"] = data["W_q"][:, :, 0, 0, 0]
    _W_q0 = data.get("_W_q0_for_precond") if _need_W_q0 else None

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
    # Lanczos path so callers don't branch — see the RETURN CONVENTION note at
    # the bottom of this branch for what "the same tuple" has to mean.
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
        # ── preconditioner route ──────────────────────────────────────
        # "bare"  : 1/(dE - lam + eps)          -- what LORRAX has always done
        # "exact" : 1/(diag(H) - lam + eps)     -- what BerkeleyGW hands PRIMME
        # The exact diagonal is assembled ONCE here (two einsums, ~0.3 GFLOP,
        # under a third of one matvec).  The per-iteration application is
        # elementwise on the (c, v, k) shards either way: same shape, same
        # sharding, ZERO collectives, no new shard_map, no retrace.
        _diag_H = None
        if davidson_precond == "exact":
            if _W_q0 is None:
                raise ValueError(
                    "--davidson-precond exact needs the W term; this run has "
                    "include_W=False, so diag(H) is just dE and the bare "
                    "route is already exact.")
            from .bse_davidson_helpers import build_bse_exact_diagonal
            from .bse_davidson_helpers import exact_diagonal_memo_stats
            _memo_before = exact_diagonal_memo_stats()["hits"]
            with timing.section("bse.solve.exact_diag"):
                _diag_H = build_bse_exact_diagonal(
                    eps_c, eps_v, psi_c_X, psi_v_Y, _W_q0, M_X, M_Y, V_q0,
                    nk, sharding=NamedSharding(mesh_xy, P("x", "y", None)))
                _diag_H.block_until_ready()
            _memoised = exact_diagonal_memo_stats()["hits"] > _memo_before
            print(f"Davidson preconditioner: EXACT diag(H) "
                  f"(dE + (V_x - W_d)/nk), "
                  f"{'from memo (same payload)' if _memoised else 'assembled'}",
                  flush=True)
        else:
            print("Davidson preconditioner: bare dE", flush=True)
        if davidson_olsen:
            print("Davidson preconditioner: Olsen correction ON", flush=True)

        precond_fn = bse_diagonal_precond(
            eps_c, eps_v, sharding=NamedSharding(mesh_xy, P("x", "y", None)),
            epsilon_shift=davidson_eps_shift_Ry,
            diag_H=_diag_H, olsen=davidson_olsen)
        X0 = init_bse_subspace(
            eps_c, eps_v, n_eig=n_eig,
            n_cond=int(data["n_cond"]), n_val=int(data["n_val"]),
            n_random=davidson_n_random_init,
            mesh=mesh_xy, sharding=bse_sharding)


        # ── perf-campaign probe hooks (env-gated, no effect when unset) ──
        # LORRAX_DAV_MVSCAN=w1,w2,...  time apply_H at each block width, so
        #   the matvec-count currency can be converted to seconds honestly:
        #   a width-20 block matvec is NOT 20 independent width-1 matvecs.
        # LORRAX_DAV_TRACE=<path.npz>  dump the per-iteration history.
        import os as _os
        _mvscan = _os.environ.get("LORRAX_DAV_MVSCAN", "")
        if _mvscan:
            import time as _t
            print("[mvscan] block-width scaling of the BSE matvec", flush=True)
            print("[mvscan] width  wall_s     s/vector   rel_to_w1", flush=True)
            _per1 = None
            for _w in [int(x) for x in _mvscan.split(",") if x.strip()]:
                _Vw = jnp.repeat(X0[:1], _w, axis=0) if _w > n_eig else X0[:_w]
                _Vw = jax.lax.with_sharding_constraint(_Vw, sh.X)
                _ = apply_H(_Vw); _.block_until_ready()          # warm
                _reps = 5
                _t0 = _t.perf_counter()
                for _ in range(_reps):
                    _o = apply_H(_Vw)
                _o.block_until_ready()
                _dt = (_t.perf_counter() - _t0) / _reps
                if _per1 is None:
                    _per1 = _dt / _w
                print(f"[mvscan] {_w:5d}  {_dt:9.5f}  {_dt/_w:9.6f}  "
                      f"{(_dt/_w)/_per1:8.3f}", flush=True)
                del _Vw, _o

        # Pre-compile _ritz_and_residuals at every subspace size m ∈ {n_eig,
        # 2·n_eig, …, m_max} so the Davidson loop does not pay 4 separate
        # XLA compiles as the subspace grows between restarts. Compiles
        # ~2 s otherwise; with warmup this is a one-time up-front cost.
        from solvers.davidson import DEFAULT_M_MAX_FACTOR
        m_max_warm = (int(davidson_m_max) if davidson_m_max
                      else DEFAULT_M_MAX_FACTOR * n_eig)
        warmup_davidson_jit(
            n_eig=n_eig,
            trailing_shape=tuple(X0.shape[1:]),
            m_max=m_max_warm,
            dtype=X0.dtype,
            sharding=bse_sharding,
        )

        print(f"Davidson: m_max={m_max_warm} ({m_max_warm // n_eig}x n_eig), "
              f"storing 2*m_max = {2 * m_max_warm} trial vectors", flush=True)
        eigenvalues, eigenvectors = davidson(
            apply_H, n_eig=n_eig, precond_fn=precond_fn, X0=X0,
            m_max=m_max_warm,
            max_iter=max_iter, tol=atol if atol > 0 else 1e-8,
            verbose=True,
        )

        # ── perf-campaign probe: trace + history dump (env-gated) ──
        from solvers.davidson import (
            TRACE_COUNTS as _DAV_TRACES,
            MATVEC_APPLICATIONS as _DAV_MV,
            LAST_RUN as _DAV_HIST,
            instrumentation_summary as _dav_summary,
        )
        print(_dav_summary(), flush=True)
        _trace_path = _os.environ.get("LORRAX_DAV_TRACE", "")
        if _trace_path and jax.process_index() == 0:
            import numpy as _np
            _np.savez(
                _trace_path,
                iter=_np.asarray(_DAV_HIST.get('iter', [])),
                mv=_np.asarray(_DAV_HIST.get('mv', [])),
                m=_np.asarray(_DAV_HIST.get('m', [])),
                eig=_np.asarray(_DAV_HIST.get('eig', [])),
                res=_np.asarray(_DAV_HIST.get('res', [])),
                t=_np.asarray(_DAV_HIST.get('t', [])),
                n_distinct_programs=_np.asarray(len(_DAV_TRACES)),
                total_matvec_applications=_np.asarray(_DAV_MV[0]),
            )
            print(f"[dav-instr] history -> {_trace_path}", flush=True)

        # ── RETURN CONVENTION ────────────────────────────────────────────────
        # Match the Lanczos return CONVENTION, not merely its shape.  The
        # Lanczos routes below pin ``out_shardings=(rep_eig, …)`` with
        # ``rep_eig = NamedSharding(mesh_xy, P())``, i.e. eigenvalues and
        # eigenvectors come back REPLICATED.  Davidson's ``X`` inherits the
        # SOLVE sharding ``P(None,"x","y",None)`` instead (it is
        # ``jnp.einsum('mn,m...->n...', x, V)`` over a sharded ``V``), and the
        # reshape below does not change that.  Two solvers reachable from one
        # ``--solver`` flag were therefore returning different layouts under one
        # documented tuple, and every consumer that believed the docstring was
        # holding a live grenade: ``--solver davidson --write-eigs`` died on
        # EVERY rank at P>1 in ``bse_io.write_eigenvectors_stream``'s
        # ``device_get`` ("Fetching value for `jax.Array` that spans
        # non-addressable (non process local) devices").
        #
        # ``bse_io`` now fetches through ``gather_to_host`` and so survives any
        # layout — but that is the WRITER being defensive, and it does not make
        # the tuple honest for the next consumer.  This is the other half: the
        # convention is declared here and enforced here, so "the same tuple as
        # the Lanczos path" is true of the sharding as well as the shape.
        # Cost: the same replicated eigenvector set the Lanczos path has always
        # returned (n_eig × nc_pad × nv_pad × nk complex128 per rank).
        rep_eig_dav = NamedSharding(mesh_xy, P())
        eigenvectors = jax.jit(lambda a: a, out_shardings=rep_eig_dav)(
            eigenvectors.reshape(n_eig, 1, nc_pad, nv_pad, nk))
        # ``davidson`` returns eigenvalues as a HOST numpy array (it needs them
        # on the host for its own convergence test), where Lanczos returns a
        # replicated device array.  ``replicate_to_mesh`` declares the identical
        # per-rank copy as the replica with NO collective; its precondition —
        # bit-identical on every rank — holds because every rank ran the same
        # deterministic reduction to get it.
        eigenvalues = replicate_to_mesh(np.asarray(eigenvalues), mesh_xy)
        # ``N_ITER_NOT_MEASURED``, not ``max_iter``: the branch this merge
        # carries was written when this route still returned its budget, and
        # registered that as its deferred item 4.  ``main`` has since landed
        # the sentinel (``bse_lanczos.py:49``, read at ``bse_jax.py:212``), so
        # the deferral is satisfied here and the honest value is kept.
        return eigenvalues, eigenvectors, jnp.int32(N_ITER_NOT_MEASURED)


    # ── Thick-restart Lanczos: fixed-shape, memory-capped Krylov ──────────
    # Same Krylov convergence as the unrestarted path, from a basis of
    # ``m_max + 1`` slots instead of ``max_iter + 1``.  On a 1024-dim deck
    # that trade is a loss (the unrestarted solve already fits); it exists
    # for production ``bse_dim``, where a 400-vector basis does not.
    #
    # Everything is preallocated and every trip count is a compile-time
    # constant, so the iteration traces at exactly one shape.  It opens NO
    # shard_map of its own: the subspace algebra is ellipsis einsums whose
    # batch axis is replicated and whose trailing axes carry ``sh.X``, which
    # GSPMD lowers to the two CGS2 collectives per step; the only shard_map
    # in the solve is the one ``matvec_ring`` already owns.
    if solver_kind == "trlan":
        from solvers.thick_restart_lanczos import thick_restart_lanczos_eig

        psi_c_X = data["psi_c_X"]; psi_c_Y = data["psi_c_Y"]
        psi_v_X = data["psi_v_X"]; psi_v_Y = data["psi_v_Y"]
        eps_c   = data["eps_c"];   eps_v   = data["eps_v"]
        V_q0    = data["V_q0"]
        M_X     = data["M_X"];     M_Y     = data["M_Y"]

        m_max = int(trlan_m_max) if trlan_m_max else max(3 * n_eig, 60)
        n_keep = int(trlan_n_keep) if trlan_n_keep else n_eig + 10
        # ``--max-lanczos-iter`` is read as a budget of matvec APPLICATIONS,
        # so the A/B against the unrestarted path is like-for-like: both
        # spend the same number of H applications.
        per_cycle = m_max - n_keep
        n_restarts = max(0, -(-(int(max_iter) - m_max) // per_cycle))
        apps = m_max + n_restarts * per_cycle
        bse_sharding = NamedSharding(mesh_xy, P(None, "x", "y", None))
        rep_tr = NamedSharding(mesh_xy, P())

        print(f"Thick-restart Lanczos: n_eig={n_eig} m_max={m_max} "
              f"n_keep={n_keep} restarts={n_restarts} -> {apps} matvec "
              f"applications, basis {m_max + 1} slots "
              f"({(m_max + 1) / (int(max_iter) + 1):.2f}x the unrestarted "
              f"basis at the same budget)", flush=True)

        # ONE jit, tensors as ARGUMENTS.  They span every process, so closing
        # over them inside the traced loop body would carry them as constants
        # -- the shape of mistake bse_feast's cached runner makes; inside a
        # fori_loop it does not raise, it just builds a program that does not
        # finish compiling.  Measured: 13 min without this, and counting.
        @partial(
            jax.jit,
            in_shardings=(
                sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                sh.eps, sh.eps, sh.W, sh.V, sh.psi_x, sh.psi_y,
            ),
            # Replicated out, matching the Lanczos path's contract, so
            # --write-eigs is addressable at P>1 for this solver.
            out_shardings=(rep_tr, rep_tr, rep_tr),
        )
        def _trlan_run(pcx, pcy, pvx, pvy, ec, ev, WR, Vq0, MX, MY):
            def apply_H(V):
                V = jax.lax.with_sharding_constraint(V, sh.X)
                return matvec_ring(V, pcx, pcy, pvx, pvy, ec, ev, WR, Vq0,
                                   MX, MY)
            return thick_restart_lanczos_eig(
                apply_H, (nc_pad, nv_pad, nk),
                n_eig=n_eig, m_max=m_max, n_keep=n_keep,
                n_restarts=n_restarts, sharding=bse_sharding,
            )

        eigenvalues, eigenvectors, alpha_im = _trlan_run(
            psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0,
            M_X, M_Y)
        print(f"Thick-restart Lanczos: max |Im <q,Hq>| = "
              f"{float(alpha_im):.3e} Ry", flush=True)
        eigenvectors = eigenvectors.reshape(n_eig, 1, nc_pad, nv_pad, nk)
        # An HONEST third slot: the matvec applications actually spent, not
        # an echo of the budget (CONVERGENCE_CENSUS recommendation 2).
        return eigenvalues, eigenvectors, jnp.int32(apps)

    rep_eig = NamedSharding(mesh_xy, P())  # eigenvalues / eigenvectors come back replicated.
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
            return evs, evecs, jnp.int32(N_ITER_NOT_MEASURED)
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
                HX = HX.reshape(bs, -1)
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
                return evs, evecs, jnp.int32(N_ITER_NOT_MEASURED)

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

"""Lanczos solvers for BSE.

The generic Lanczos algorithms live in solvers.lanczos.  This module
re-exports them and provides ``solve_bse_sharded``, the one BSE solve (any
mesh, including 1x1), built on the trial-stack matvec.
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
from common.fft_helpers import get_donated_kfft_kminor

from solvers.lanczos import (
    FULL_REORTH,
    alpha_herm_sink,
    block_lanczos_eig_jit,
    block_lanczos_eig_jit_converged,
    report_alpha_herm,
    simple_lanczos_eig,
    split_alpha_sink,
)
from common.log import get_logger

log = get_logger(__name__)

#: Third element of ``solve_bse_sharded``'s return tuple when the route ran a
#: FIXED number of iterations and therefore never measured a convergence point.
#:
#: The fixed-iteration routes here (``block_lanczos_eig_jit`` at any block
#: size, ``davidson``) run their full iteration budget unconditionally.  They used
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


from common.units import RYD_TO_EV

#: The Davidson residual tolerance of the default lowest-N solve, meV.
DAVIDSON_TOL_MEV = 0.01
RY_TO_MEV = RYD_TO_EV * 1.0e3

from .bse_ring_comm import make_bse_shardings
from .bse_head import head_shifted_eps_c
import common.timing as timing


#: Host threads per emulated CPU device the planned Davidson needs: the CPU
#: subspace provider's host callbacks share the CPU client's pool with the
#: devices' collective rendezvous.  Measured on 4 host devices (2x2): 4 threads
#: hang in an all-to-all until XLA's rendezvous timeout (rc 134); 16 run.
CPU_DAVIDSON_THREADS_PER_DEVICE = 4


def _refuse_cpu_davidson_threads(mesh_xy) -> None:
    """``GATE cpu_davidson_threads``: refuse up front where the CPU provider hangs."""
    local = [d for d in mesh_xy.devices.flat if d.process_index == jax.process_index()]
    if local[0].platform != "cpu" or len(local) < 2:
        return
    threads, need = len(os.sched_getaffinity(0)), CPU_DAVIDSON_THREADS_PER_DEVICE * len(local)
    if threads < need:
        raise RuntimeError(
            f"GATE cpu_davidson_threads: {len(local)} emulated CPU devices in one process "
            f"on {threads} host threads; the planned Davidson needs >= {need} "
            f"({CPU_DAVIDSON_THREADS_PER_DEVICE} per device), or its host callbacks starve "
            "the devices' collective rendezvous and XLA aborts on a timeout.  Fix: give "
            f"the process {need} cores, or run one device per process")


def _price_davidson_basis(capacity, X0, mesh_xy) -> None:
    """Book and admit the Davidson basis and its images: 2 x capacity complex128
    vectors of the transition dimension, sharded over the mesh."""
    from lxkit import memory
    from common.gpu_utils import device_budget_bytes
    n = int(np.prod(X0.shape[1:]))
    basis = memory.priced(0.0, "bse Davidson basis + images (2 capacity x N x 16 B / P)",
                          spread=2.0 * capacity * n * 16, ranks=int(mesh_xy.devices.size))
    memory.book(basis)
    memory.admit("bse.davidson_basis", basis, device_budget_bytes())


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
    davidson_tol_Ry: float = DAVIDSON_TOL_MEV / RY_TO_MEV,
    tda: bool = True,
    keep_W_R: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """The BSE eigensolve (Lanczos / block / Davidson / TRLan) on any mesh.

    The single-device ``solve_bse`` it used to sit beside was deleted with
    ``bse_serial`` (C10, 2026-09-24); a one-device run is this on a 1x1 mesh.
    The per-iteration matvec is the trial-stack matvec, which
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

    ``davidson_tol_Ry`` is the Davidson stopping rule: every root's residual
    norm ``||H x - θ x||`` below it (times ``max(1, |θ|)``).  For a Hermitian H
    a residual r puts an exact eigenvalue within ``||r||`` of θ, so the default,
    0.01 meV, bounds each root's error by 0.01 meV without assuming a gap; the
    usual error is ``||r||^2 / gap``, far smaller.

    ``keep_W_R`` leaves the transformed screened kernel on the payload as
    ``data["W_R"]`` (one (μ, ν, k) tile per rank, the solve's own) for a second
    consumer in the same process (``bse_jax --haydock-iter``); a payload that
    already carries ``W_R`` is solved on it without a second transform.
    """
    if os.environ.get("LORRAX_LANCZOS_REORTH", "").strip():
        raise ValueError("LORRAX_LANCZOS_REORTH is retired: CGS2 is the only reorthogonalisation route; unset it")
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
    with timing.section("bse.solve.matvec_build"):
        matvec_ring = build_bse_stack_matvec(
            mesh_xy, nkx, nky, nkz, kernel="bse" if include_W else "rpa",
        )

    # ``auto`` is resolved to one of the two real routes HERE, before anything
    # else reads the flag, so every later branch sees only "bare" or "exact".
    if davidson_precond == "auto":
        from .bse_davidson_helpers import resolve_precond_route
        _resolved = resolve_precond_route("auto", n_flat)
        log.info(f"Davidson preconditioner: auto -> {_resolved} "
                 f"(bse_dim={n_flat})")
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

    # ── W_R = ifft_q(W_q) at a REAL top-level dispatch boundary, W_q DONATED ──
    # This used to run INSIDE ``_full_run`` (and inside the Davidson block).
    # There it can never free W_q: W_q is a jit PARAMETER, so its buffer is
    # owned by the caller for the whole call and XLA has no same-shape OUTPUT
    # to alias it to ("donation was always declined").  The result was BOTH
    # W_q and W_R resident for the entire Lanczos: 2 x (μ_pad/p_x) x
    # (ν_pad/p_y) x nk x 16 bytes per rank (2 x 404 MB at μ=10015 / P=64;
    # ∝ μ²/P, so 2 x 4.1 GB at μ=32k).  Hoisting the transform into its own
    # donated jit lets XLA alias W_R onto W_q's buffer and lets the caller
    # drop the Python reference, so only ONE copy survives into the solve.
    # Value-identical: same helper, same axes, same norm, same operand.
    if include_W and data.get("W_R") is not None:
        W_R = data["W_R"]               # a previous solve's transform (keep_W_R)
    elif include_W:
        with timing.section("bse.solve.W_ifft") as _sec_wifft:
            # The sharded helper, not ``jnp.fft.ifftn``: a plain ifftn on a
            # sharded tensor inserts a 337-MiB all-gather around the FFT under
            # current JAX even when the transformed axes are unsharded.  The
            # helper hides the transform in an opaque per-device primitive, so
            # XLA only ever sees a local FFT (axes (2,3,4) of W_q are
            # replicated; μ, ν stay on x, y), 3-D in one shot rather than three
            # sequential rank-1 custom_partitioning calls.
            #
            # The transform and its donating jit come from the MEMOISED accessor,
            # not from a fresh ``make_sharded_ifftn_3d`` + ``jax.jit`` pair built
            # here.  Built here, the wrapper object was new on every call, so jax's
            # dispatch cache started empty and a byte-identical program was
            # re-traced, re-lowered and re-probed against the persistent compile
            # cache every time this function ran — ~20-23 ms against ~1.0 ms of
            # execution on the record deck at P=4, and flat across calls, which is
            # the construction signature (FIX_construction_defects.md §1; the same
            # defect PRECOND_BUILD_FREE.md §3.1 fixed for the exact-diagonal build
            # and named at §7.2).  Donation is unchanged: the accessor sets
            # ``donate_argnums=(0,)`` and the alias is gated in
            # tests/test_bse_w_ifft_hoist.py.
            _W_ifft_donated = get_donated_kfft_kminor(
                mesh_xy, tuple(int(n) for n in data["W_q"].shape[-3:]), sh.W.spec,
                kind="ifftn", norm='ortho')
            W_R = _W_ifft_donated(data["W_q"])
            data["W_q"] = None          # release the caller-side reference
            _sec_wifft.watch(W_R)
        if keep_W_R:
            data["W_R"] = W_R
    else:
        W_R = data["W_q"]

    # ── Davidson path: pass resident operator/preconditioner arrays as
    # explicit operands of the fixed-capacity solve. Returns
    # the same `(eigenvalues, eigenvectors, n_iter_done)` tuple as the
    # Lanczos path so callers don't branch — see the RETURN CONVENTION note at
    # the bottom of this branch for what "the same tuple" has to mean.
    if solver_kind == "davidson":
        from solvers.davidson import davidson
        from .bse_davidson_helpers import bse_diagonal_precond, init_bse_subspace

        psi_c_X = data["psi_c_X"]; psi_c_Y = data["psi_c_Y"]
        psi_v_X = data["psi_v_X"]; psi_v_Y = data["psi_v_Y"]
        eps_c   = head_shifted_eps_c(data["eps_c"], data)
        eps_v   = data["eps_v"]
        V_q0    = data["V_q0"]
        M       = data["M"]  # hoisted V-term pair amplitude (P3)
        # W_R already built above (donated top-level ifft).

        operator_data = (psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                         eps_c, eps_v, W_R, V_q0, M)

        def apply_H(solver_data, V):
            V = jax.lax.with_sharding_constraint(V, sh.X)
            return matvec_ring(V, *solver_data[0])

        bse_sharding = NamedSharding(mesh_xy, P(None, "x", "y", None))
        _refuse_cpu_davidson_threads(mesh_xy)
        # ── preconditioner route ──────────────────────────────────────
        # "bare"  : 1/(dE - lam + eps)          -- what LORRAX has always done
        # "exact" : 1/(diag(H) - lam + eps)     -- what BerkeleyGW hands PRIMME
        # The exact diagonal is assembled ONCE here (two einsums, ~0.3 GFLOP,
        # under a third of one matvec).  The per-iteration application is
        # elementwise on the (c, v, k) shards either way: same shape, same
        # sharding, ZERO collectives, no new shard_map, no retrace.
        _diag_H = None
        if davidson_precond == "exact":
            # include_W=False (RPA): _W_q0 is None and the build emits no
            # screened-direct term, so the diagonal is dE + w V_x / nk.
            from .bse_davidson_helpers import build_bse_exact_diagonal
            from .bse_davidson_helpers import exact_diagonal_memo_stats
            _memo_before = exact_diagonal_memo_stats()["hits"]
            with timing.section("bse.solve.exact_diag"):
                _diag_H = build_bse_exact_diagonal(
                    eps_c, eps_v, psi_c_X, psi_v_Y, _W_q0, M, V_q0,
                    nk, sharding=NamedSharding(mesh_xy, P("x", "y", None)))
                _diag_H.block_until_ready()
            _memoised = exact_diagonal_memo_stats()["hits"] > _memo_before
            log.info(f"Davidson preconditioner: EXACT diag(H) "
                     f"(dE + (V_x{' - W_d' if include_W else ''})/nk), "
                     f"{'from memo (same payload)' if _memoised else 'assembled'}")
        else:
            log.info("Davidson preconditioner: bare dE")
        if davidson_olsen:
            log.info("Davidson preconditioner: Olsen correction ON")

        precond_fn = bse_diagonal_precond(
            eps_c, eps_v, sharding=NamedSharding(mesh_xy, P("x", "y", None)),
            epsilon_shift=davidson_eps_shift_Ry,
            diag_H=_diag_H, olsen=davidson_olsen)
        solver_data = (operator_data, precond_fn.data)
        precond_apply = precond_fn.apply

        def apply_precond(payload, R, values, X):
            return precond_apply(payload[1], R, values, X)

        X0 = init_bse_subspace(
            eps_c, eps_v, n_eig=n_eig,
            n_cond=int(data["n_cond"]), n_val=int(data["n_val"]),
            n_random=davidson_n_random_init,
            mesh=mesh_xy, sharding=bse_sharding)


        # ── perf-campaign probe hooks (env-gated, no effect when unset) ──
        # LORRAX_DAV_MVSCAN=w1,w2,...  time apply_H at each block width, so
        #   the matvec-count currency can be converted to seconds honestly:
        #   a width-20 block matvec is NOT 20 independent width-1 matvecs.
        # LORRAX_DAV_TRACE=<path.npz>  dump the final solver snapshot.
        import os as _os
        _mvscan = _os.environ.get("LORRAX_DAV_MVSCAN", "")
        if _mvscan:
            import time as _t
            log.info("[mvscan] block-width scaling of the BSE matvec")
            log.info("[mvscan] width  wall_s     s/vector   rel_to_w1")
            _per1 = None
            for _w in [int(x) for x in _mvscan.split(",") if x.strip()]:
                _Vw = jnp.repeat(X0[:1], _w, axis=0) if _w > n_eig else X0[:_w]
                _Vw = jax.lax.with_sharding_constraint(_Vw, sh.X)
                _ = apply_H(solver_data, _Vw); _.block_until_ready()          # warm
                _reps = 5
                _t0 = _t.perf_counter()
                for _ in range(_reps):
                    _o = apply_H(solver_data, _Vw)
                _o.block_until_ready()
                _dt = (_t.perf_counter() - _t0) / _reps
                if _per1 is None:
                    _per1 = _dt / _w
                log.info(f"[mvscan] {_w:5d}  {_dt:9.5f}  {_dt/_w:9.6f}  "
                         f"{(_dt/_w)/_per1:8.3f}")
                del _Vw, _o

        # One fixed-capacity solve replaces the former shape warmup ladder.
        from solvers.davidson import DEFAULT_M_MAX_FACTOR
        m_max_warm = (int(davidson_m_max) if davidson_m_max
                      else DEFAULT_M_MAX_FACTOR * n_eig)
        log.info(f"Davidson: m_max={m_max_warm} ({m_max_warm // n_eig}x n_eig), "
                 f"storing 2*m_max = {2 * m_max_warm} trial vectors")
        _price_davidson_basis(max(m_max_warm, 2 * n_eig), X0, mesh_xy)
        eigenvalues, eigenvectors = davidson(
            apply_H, n_eig=n_eig, precond_fn=apply_precond, X0=X0,
            data=solver_data,
            m_max=m_max_warm,
            max_iter=max_iter, tol=float(davidson_tol_Ry),
            verbose=True,
            # Soft locking: a root under the tolerance adds no correction, so H
            # is applied only to the open roots' directions.  Si 8^3 lowest 40
            # at 0.01 meV (dense CPU count): 265 H applications, 800 without.
            lock_converged=True,
        )

        # ── perf-campaign probe: trace + history dump (env-gated) ──
        from solvers.davidson import (
            TRACE_COUNTS as _DAV_TRACES,
            MATVEC_APPLICATIONS as _DAV_MV,
            LAST_RUN as _DAV_HIST,
            instrumentation_summary as _dav_summary,
        )
        log.debug(_dav_summary())
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
            log.debug(f"[dav-instr] final snapshot -> {_trace_path}")

        # Keep the physical (c, v) axes distributed across all processors.
        # The writer gathers explicitly when serializing eigenvectors.
        root_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y", None))
        eigenvectors = jax.jit(
            lambda a: a.reshape(n_eig, 1, nc_pad, nv_pad, nk),
            out_shardings=root_sharding,
        )(eigenvectors)
        # ``davidson`` returns eigenvalues as a HOST numpy array (it needs them
        # on the host for its own convergence test), where Lanczos returns a
        # replicated device array.  ``replicate_to_mesh`` declares the identical
        # per-rank copy as the replica with NO collective; its precondition —
        # bit-identical on every rank — holds because every rank ran the same
        # deterministic reduction to get it.
        eigenvalues = replicate_to_mesh(np.asarray(eigenvalues), mesh_xy)
        # The iterations the while_loop actually ran: Davidson stops on its
        # residual test, so this is a measured count, not the budget.
        return eigenvalues, eigenvectors, jnp.int32(int(_DAV_HIST["iter"][0]))


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
        eps_c   = head_shifted_eps_c(data["eps_c"], data)
        eps_v   = data["eps_v"]
        V_q0    = data["V_q0"]
        M       = data["M"]

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
        from distrib_la import plan_subspace
        trlan_plan = plan_subspace(
            capacity=m_max + 1, n_eig=n_keep, vector_sharding=bse_sharding)

        log.info(f"Thick-restart Lanczos: n_eig={n_eig} m_max={m_max} "
                 f"n_keep={n_keep} restarts={n_restarts} -> {apps} matvec "
                 f"applications, basis {m_max + 1} slots "
                 f"({(m_max + 1) / (int(max_iter) + 1):.2f}x the unrestarted "
                 f"basis at the same budget)")

        # ONE jit, tensors as ARGUMENTS.  They span every process, so closing
        # over them inside the traced loop body would carry them as constants
        # -- the shape of mistake bse_feast's cached runner makes; inside a
        # fori_loop it does not raise, it just builds a program that does not
        # finish compiling.  Measured: 13 min without this, and counting.
        @partial(
            jax.jit,
            in_shardings=(
                sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                sh.eps, sh.eps, sh.W, sh.V, sh.M,
            ),
            # Keep Ritz vectors distributed; the streaming writer owns host I/O.
            out_shardings=(rep_tr, bse_sharding, rep_tr),
        )
        def _trlan_run(pcx, pcy, pvx, pvy, ec, ev, WR, Vq0, M):
            def apply_H(V):
                V = jax.lax.with_sharding_constraint(V, sh.X)
                return matvec_ring(V, pcx, pcy, pvx, pvy, ec, ev, WR, Vq0, M)
            return thick_restart_lanczos_eig(
                apply_H, (nc_pad, nv_pad, nk),
                n_eig=n_eig, m_max=m_max, n_keep=n_keep,
                n_restarts=n_restarts, sharding=bse_sharding,
                subspace_plan=trlan_plan,
            )

        eigenvalues, eigenvectors, alpha_im = _trlan_run(
            psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0,
            M)
        log.info(f"Thick-restart Lanczos: max |Im <q,Hq>| = "
                 f"{float(alpha_im):.3e} Ry")
        eigenvectors = eigenvectors.reshape(n_eig, 1, nc_pad, nv_pad, nk)
        # An HONEST third slot: the matvec applications actually spent, not
        # an echo of the budget (CONVERGENCE_CENSUS recommendation 2).
        return eigenvalues, eigenvectors, jnp.int32(apps)

    from distrib_la import plan_subspace
    from .bse_window import pad_zone_mask_np
    lanczos_vector_shape = (nc_pad, nv_pad, nk)
    # The Krylov space lives on the physical transitions, as in
    # exciton_bands: the start block has no support on the mesh pad, which
    # the operator keeps at exactly zero, so the recurrence and its
    # α-Hermiticity scale are the same numbers at every P.  The depth
    # counts physical transitions.
    lanczos_support = pad_zone_mask_np(
        data["n_cond"], data["n_val"], nc_pad, nv_pad, nk)[0]
    n_phys = int(data["n_cond"]) * int(data["n_val"]) * nk
    lanczos_depth = max(1, min(int(max_iter), n_phys // bs))
    lanczos_plan = plan_subspace(
        capacity=(lanczos_depth + 1) * bs, n_eig=n_eig,
        max_block_size=max(bs, n_eig),
        vector_sharding=NamedSharding(mesh_xy, P(None, "x", "y", None)))
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
            sh.eps, sh.eps, sh.W, sh.V, sh.M,
        ),
        # Fourth entry covers the α-Hermiticity payload: a pytree prefix, so
        # ``rep_eig`` applies to each of its (replicated, scalar) leaves.
        out_shardings=(rep_eig, sh.X, rep_eig, rep_eig),
        # NB: arg 6 is now W_R (already ifft'd, DONATED at its own top-level
        # boundary above) rather than W_q.  Donating it HERE is still declined
        # — there is no aliasable same-shape output of a Lanczos solve — which
        # is the original audit-P5 observation; the fix was to move the
        # transform out, not to donate the eigensolve's inputs.
    )
    def _full_run(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M):
        with alpha_herm_sink() as _sink:
            evs, evecs, n_it = _krylov(
                psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
                W_R, V_q0, M,
            )
        labels, payload = split_alpha_sink(_sink)
        _alpha_labels[:] = labels
        return evs, evecs, n_it, payload

    def _krylov(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M):
        if bs == 1:
            # Single-vector matvec — accept (n_flat,) and reshape to (1, c, v, k).
            def matvec(v_flat):
                X = v_flat.reshape(shape)
                X = jax.lax.with_sharding_constraint(X, sh.X)
                HX = matvec_ring(
                    X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                    eps_c, eps_v, W_R, V_q0, M,
                )
                return HX[0]
            # The block kernel at block_size=1 is the single-vector Lanczos.
            def matvec_block(V_block):
                return matvec(V_block[0])[None]
            if rtol > 0.0:
                # Convergence-driven path: the block while_loop, early exit.
                return block_lanczos_eig_jit_converged(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=1, max_iter=max_iter,
                    rtol=rtol, atol=atol, check_every=check_every,
                    n_reorth=n_reorth,
                    subspace_plan=lanczos_plan, vector_shape=lanczos_vector_shape,
                    structured_vectors=True, support=lanczos_support,
                )
            evs, evecs = block_lanczos_eig_jit(
                matvec_block, n_flat, n_eig=n_eig,
                block_size=1, max_iter=max_iter, n_reorth=n_reorth,
                subspace_plan=lanczos_plan, vector_shape=lanczos_vector_shape,
                structured_vectors=True, support=lanczos_support,
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
                    eps_c, eps_v, W_R, V_q0, M,
                )
                return HX
            if rtol > 0.0:
                # Convergence-driven: ``lax.while_loop`` exits when the
                # n_eig lowest Ritz values stabilise within ``rtol``.
                return block_lanczos_eig_jit_converged(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=bs, max_iter=max_iter,
                    rtol=rtol, atol=atol, check_every=check_every,
                    n_reorth=n_reorth,
                    subspace_plan=lanczos_plan, vector_shape=lanczos_vector_shape,
                    structured_vectors=True, support=lanczos_support,
                )
            else:
                evs, evecs = block_lanczos_eig_jit(
                    matvec_block, n_flat, n_eig=n_eig,
                    block_size=bs, max_iter=max_iter, n_reorth=n_reorth,
                    subspace_plan=lanczos_plan, vector_shape=lanczos_vector_shape,
                    structured_vectors=True, support=lanczos_support,
                )
                return evs, evecs, jnp.int32(N_ITER_NOT_MEASURED)

    # ONE program: trace + XLA compile + the entire Krylov loop's execution.
    # Split compile from execution by lowering/compiling first, so the two
    # costs the campaign A/Bs are separately visible instead of fused into
    # ``bse.eigensolve``.  Value-identical: the same jit, the same operands.
    _args = (
        data["psi_c_X"], data["psi_c_Y"],
        data["psi_v_X"], data["psi_v_Y"],
        head_shifted_eps_c(data["eps_c"], data), data["eps_v"],
        W_R, data["V_q0"],
        data["M"],
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

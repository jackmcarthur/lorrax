"""The Tamm-Dancoff BSE eigensolve: the Davidson on the trial-stack matvec.

(The module keeps the name of the Lanczos solvers it used to hold; they were
retired 2026-11-27.)  ``solve_bse_sharded`` is the one BSE solve, on any mesh
(1x1 included): the Davidson of ``solvers.davidson`` with the exact diagonal
preconditioner on the trial-stack matvec, or, with ``tda=False``, a hand-off to
``bse_nontda`` (the dense structure-preserving solve or the full-BSE Davidson).
It also holds the Davidson's stop line, basis pricing and thread gate that the
BSE drivers share.
"""
from __future__ import annotations

import os
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import replicate_to_mesh

from common.log import get_logger

log = get_logger(__name__)


from common.units import RYD_TO_EV

#: The Davidson residual tolerance of the default lowest-N solve, meV.
DAVIDSON_TOL_MEV = 0.01
RY_TO_MEV = RYD_TO_EV * 1.0e3
#: Margin roots of the BSE Davidson: it solves n_eig + this many, stops on the
#: lowest n_eig and reports those, so a multiplet split at the n_eig boundary
#: is held in the block instead of being dropped (audit S1).
DAVIDSON_MARGIN_ROOTS = 8


def davidson_stop_line(*, n_eig, tol_mev, budget, cap, refused, fix, where=""):
    """The last Davidson solve (``solvers.davidson.LAST_RUN``) as one log line over its
    lowest ``n_eig`` roots.  Any stop but CONVERGED refuses (``GATE davidson_converged``);
    a driver calls this before it writes anything.  ``cap`` names the iteration cap,
    ``refused`` what was not written, ``fix`` the remedy."""
    from solvers.davidson import LAST_RUN as run
    from solvers.davidson_fixed import CONVERGED, STALLED
    res_mev = (np.asarray(run["res"][0]) * RY_TO_MEV)[:int(n_eig)]
    n_conv = int(np.sum(res_mev < tol_mev))
    status = int(run["status"])
    word = ("converged" if status == CONVERGED else
            "STALLED" if status == STALLED else "NOT converged")
    it, worst = int(run["iter"][0]), float(np.max(res_mev))
    line = (f"Davidson {word}{where}: {n_conv}/{n_eig} roots with residual < "
            f"{tol_mev:g} meV in {it}/{int(budget)} iterations, "
            f"{int(run['mv'][0])} matvecs, {int(run['restarts'])} restarts; "
            f"max residual {worst:.3e} meV")
    log.info(line)
    if status != CONVERGED:
        raise RuntimeError(
            f"GATE davidson_converged: Davidson stopped {word}{where} with {n_conv}/{n_eig} "
            f"roots below the {tol_mev:g} meV residual after {it}/{int(budget)} iterations "
            f"({cap}); max residual {worst:.3e} meV.  {refused}  Fix: {fix}")
    return line

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
    include_W: bool = True,
    davidson_m_max: int | None = None,
    davidson_precond: str = "exact",
    davidson_olsen: bool = False,
    davidson_n_random_init: int = 5,
    davidson_eps_shift_Ry: float = 1e-3,
    davidson_tol_Ry: float = DAVIDSON_TOL_MEV / RY_TO_MEV,
    tda: bool = True,
    keep_W_R: bool = False,
    davidson_margin: int = DAVIDSON_MARGIN_ROOTS,
    davidson_margin_tol_Ry: float | None = None,
    davidson_plan_key=None,
    full_solver: str = "dense",
    full_on_indefinite: str = "refuse",
) -> Tuple[jax.Array, jax.Array]:
    """The BSE eigensolve (the Tamm-Dancoff Davidson, or the full BSE) on any mesh.

    The single-device ``solve_bse`` it used to sit beside was deleted with
    ``bse_serial`` (C10, 2026-09-24); a one-device run is this on a 1x1 mesh.
    The per-iteration matvec is the trial-stack matvec, which
    (a) parallelises over (px·py) GPUs,
    (b) precomputes ``W_R = ifft(W_q)`` once outside the iteration
        instead of per-iter (the dominant cost in the single-device
        path — 200 × 3D-FFTs of an (n_μ, n_μ, nkx,nky,nkz) tensor).

    Vector layout:
        Trial vectors ``X`` have shape ``(m, n_cond_pad, n_val_pad, nk)``
        sharded ``P(None, "x", "y", None)``; ``apply_H`` applies
        ``with_sharding_constraint`` once per matvec, so collectives inside
        the matvec remain on the ``(x, y)`` mesh.

    The data dict must come from ``bse_io.load_bse_data_from_restart_sharded``
    and contains psi_{c,v}_{X,Y}, eps_c, eps_v, V_q0 (P("x","y")), W_q
    (P("x","y",None,None,None)), plus any q=0 head injection already
    applied at load time.

    ``tda`` (default True) selects the Tamm-Dancoff resonant-only Hamiltonian
    (the Davidson below).  ``tda=False``
    dispatches to the structure-preserving full-BSE eigensolver
    ``bse_nontda.solve_bse_nontda_sharded`` — the ONE non-TDA seam, no parallel
    solver stack — which returns paired (X, Y) eigenvectors (X^H X - Y^H Y = +1).
    ``full_solver`` picks its route.  ``'dense'`` (default) reads ``n_eig`` and
    ``include_W`` only: it is the dense structure-preserving solve, so every
    Davidson argument below is unread there (``bse_jax``
    refuses such deck keys under ``full_bse``).  ``'davidson'`` is the
    structure-preserving Davidson (``solvers.bse_sp_davidson``): it reads
    ``max_iter`` (the iteration cap), ``davidson_tol_Ry``, ``davidson_m_max``
    (basis capacity in real-form vectors, default ``20 n_eig``),
    ``davidson_margin``, ``davidson_margin_tol_Ry``, ``davidson_precond``,
    ``davidson_eps_shift_Ry``, ``davidson_n_random_init``,
    ``davidson_olsen``, ``davidson_plan_key`` and ``full_on_indefinite``
    (``'refuse'`` or ``'continue'``: what to do when the stability matrix is
    not positive definite, ``bse_nontda.solve_bse_nontda_sharded``).  Both
    full routes leave ``data["W_R"]`` on the payload, so ``keep_W_R`` needs
    nothing more.  A payload may carry ``head_tensor`` and, for a finite pair
    momentum, ``partner`` (``bse_nontda.operand_sets``).

    ``davidson_tol_Ry`` is the Davidson stopping rule: every root's residual
    norm ``||H x - θ x||`` below it (times ``max(1, |θ|)``).  For a Hermitian H
    a residual r puts an exact eigenvalue within ``||r||`` of θ, so the default,
    0.01 meV, bounds each root's error by 0.01 meV without assuming a gap; the
    usual error is ``||r||^2 / gap``, far smaller.

    ``keep_W_R`` leaves the transformed screened kernel on the payload as
    ``data["W_R"]`` (one (μ, ν, k) tile per rank, the solve's own) for a second
    consumer in the same process (``bse_jax haydock = true``); a payload that
    already carries ``W_R`` is solved on it without a second transform.

    Davidson (the default) solves ``n_eig + davidson_margin`` roots and stops
    when the lowest ``n_eig`` pass the residual test; only those are returned.
    ``davidson_margin_tol_Ry`` (default: ``davidson_tol_Ry``) is the residual at
    which a margin root locks (it stays in the basis, uncorrected).
    ``davidson_plan_key`` (any hashable) reuses the planned solver across calls
    on same-shape payloads.  A payload may carry ``head_tensor = (D_head,
    M_head)``, the stack matvec's head-term operands (Davidson only).
    """
    if not tda:
        from .bse_nontda import solve_bse_nontda_sharded
        if full_solver == "dense":
            if full_on_indefinite != "refuse":
                raise ValueError(
                    f"GATE nontda_route_option: full_on_indefinite={full_on_indefinite!r} cannot "
                    "be honoured by the dense full-BSE solve (got: full_solver='dense'; want: "
                    "full_solver='davidson').  Why: the dense solve refuses a stability matrix "
                    "that is not positive definite and has no continued solve.  Fix: "
                    "full_solver='davidson' (bse_jax: full_bse_solver = davidson).")
            return solve_bse_nontda_sharded(
                data, mesh_xy, n_eig=n_eig, include_W=include_W)
        if full_solver != "davidson":
            raise ValueError(
                f"GATE nontda_route_option: the full-BSE route cannot honour "
                f"full_solver={full_solver!r} (want: full_solver 'dense' or 'davidson').  "
                f"Why: these are the two full-BSE eigensolvers; any other name would run "
                f"neither.  Fix: full_solver='davidson' or 'dense'.")
        return solve_bse_nontda_sharded(
            data, mesh_xy, n_eig=n_eig, include_W=include_W, solver="davidson",
            davidson_tol_mev=float(davidson_tol_Ry) * RY_TO_MEV,
            davidson_max_iter=int(max_iter), davidson_m_max=davidson_m_max,
            davidson_margin=int(davidson_margin),
            davidson_margin_tol_mev=(None if davidson_margin_tol_Ry is None else
                                     float(davidson_margin_tol_Ry) * RY_TO_MEV),
            davidson_precond=davidson_precond,
            davidson_eps_shift_Ry=float(davidson_eps_shift_Ry),
            davidson_olsen=bool(davidson_olsen),
            davidson_n_random_init=int(davidson_n_random_init),
            on_indefinite=full_on_indefinite, davidson_plan_key=davidson_plan_key)

    sh = make_bse_shardings(mesh_xy)
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    nkx = int(data["nkx"])
    nky = int(data["nky"])
    nkz = int(data["nkz"])
    nk = nkx * nky * nkz
    n_flat = nc_pad * nv_pad * nk

    # Trial-stack matvec (scan-inside-shard_map): applies H to the whole
    # Davidson subspace stack with ONE T-tensor alive regardless of the stack
    # width — strictly better peak memory than the legacy ring/gather/simple
    # matvecs (whose T scaled with the stack).  The legacy ``matvec_kind``
    # selector is retired here; see the retirement note in bse_stack_matvec
    # (ring/gather/simple + selector deletion pending).
    from .bse_stack_matvec import build_bse_stack_matvec
    # ``head_tensor`` (exciton_bands' mini-BZ head): ``(D_head, M_head)`` operands
    # of the stack matvec's head term.
    head_tensor = data.get("head_tensor")
    # ``occupation_weights`` (n_c, n_v, nk): sqrt(f_v - f_c) of the transitions,
    # the applier's last operand; A = dE + w K w with zero-weight pairs masked.
    occupation_weights = data.get("occupation_weights")
    with timing.section("bse.solve.matvec_build"):
        matvec_ring = build_bse_stack_matvec(
            mesh_xy, nkx, nky, nkz, kernel="bse" if include_W else "rpa",
            exchange_channel=data.get("exchange_channel", "singlet"),
            **({"head_tensor": True} if head_tensor is not None else {}),
            **({"occupations": True} if occupation_weights is not None else {}),
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
    _need_W_q0 = include_W and davidson_precond == "exact"
    if _need_W_q0:
        from .bse_davidson_helpers import w_q0_for_precond
        _W_q0 = w_q0_for_precond(data)
    else:
        _W_q0 = None

    # ── W_R = ifft_q(W_q), once, at a top-level dispatch boundary, W_q DONATED ──
    # ``bse_davidson_helpers.payload_w_r`` (the one transform of every BSE
    # eigensolve and of the Haydock recursion).  It cannot run inside the
    # solve's own jit: there W_q is a jit PARAMETER, owned by the caller for the
    # whole call, so both tiles stayed resident, 2 x (μ_pad/p_x) x (ν_pad/p_y)
    # x nk x 16 bytes per rank (2 x 404 MB at μ=10015 / P=64; ∝ μ²/P).  The
    # donated transform leaves ONE.  A payload that already carries ``W_R`` (a
    # previous solve's, ``keep_W_R``) is solved on it.
    from .bse_davidson_helpers import payload_w_r
    if include_W and data.get("W_R") is None:
        with timing.section("bse.solve.W_ifft") as _sec_wifft:
            W_R = payload_w_r(data, mesh_xy, sh)
            _sec_wifft.watch(W_R)
        if not keep_W_R:
            data.pop("W_R")             # the solve's own tile; the payload keeps it only when asked
    else:
        W_R = payload_w_r(data, mesh_xy, sh, include_W=include_W)

    # ── Davidson path: pass resident operator/preconditioner arrays as
    # explicit operands of the fixed-capacity solve.  Returns
    # `(eigenvalues, eigenvectors, n_iter_done)`, the iteration count measured.
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
                     eps_c, eps_v, W_R, V_q0, M, *(head_tensor or ()),
                     *(() if occupation_weights is None else (occupation_weights,)))

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
                nk, sharding=NamedSharding(mesh_xy, P("x", "y", None)),
                exchange_channel=data.get("exchange_channel", "singlet"))
            _diag_H.block_until_ready()
        _memoised = exact_diagonal_memo_stats()["hits"] > _memo_before
        log.info(f"Davidson preconditioner: EXACT diag(H) "
                 f"(dE + (V_x{' - W_d' if include_W else ''})/nk), "
                 f"{'from memo (same payload)' if _memoised else 'assembled'}")
    else:
        log.info("Davidson preconditioner: bare dE")
    if davidson_olsen:
        log.info("Davidson preconditioner: Olsen correction ON")
    if occupation_weights is not None:
        # The diagonal of A = dE + w K w; masked pairs parked far above the roots.
        from .bse_davidson_helpers import occupation_diagonal
        _diag_H = occupation_diagonal(_diag_H, eps_c, eps_v, occupation_weights,
                                      sharding=NamedSharding(mesh_xy, P("x", "y", None)))

    precond_fn = bse_diagonal_precond(
        eps_c, eps_v, sharding=NamedSharding(mesh_xy, P("x", "y", None)),
        epsilon_shift=davidson_eps_shift_Ry,
        diag_H=_diag_H, olsen=davidson_olsen)
    solver_data = (operator_data, precond_fn.data)
    precond_apply = precond_fn.apply

    def apply_precond(payload, R, values, X):
        return precond_apply(payload[1], R, values, X)

    # Margin roots (DAVIDSON_MARGIN_ROOTS): solve n_solve, stop on and report
    # the lowest n_eig.  Fewer when the physical space cannot hold a
    # 2 x n_solve basis.
    n_phys = int(data["n_cond"]) * int(data["n_val"]) * nk
    n_solve = n_eig + max(0, min(int(davidson_margin), n_phys // 2 - n_eig))
    if occupation_weights is not None:
        from .bse_davidson_helpers import occupation_start_block
        X0 = occupation_start_block(
            eps_c, eps_v, occupation_weights, n_rows=n_solve, n_cond=int(data["n_cond"]),
            n_val=int(data["n_val"]), mesh=mesh_xy, sharding=bse_sharding)
    else:
        X0 = init_bse_subspace(
            eps_c, eps_v, n_eig=n_solve,
            n_cond=int(data["n_cond"]), n_val=int(data["n_val"]),
            n_random=davidson_n_random_init,
            mesh=mesh_xy, sharding=bse_sharding)


    # One fixed-capacity solve replaces the former shape warmup ladder.
    from solvers.davidson import DEFAULT_M_MAX_FACTOR
    m_max_warm = max(int(davidson_m_max) if davidson_m_max
                     else DEFAULT_M_MAX_FACTOR * n_eig, 2 * n_solve)
    m_max_warm = min(m_max_warm, int(np.prod(X0.shape[1:])))
    log.info(f"Davidson: {n_eig} roots + {n_solve - n_eig} margin"
             + ("" if davidson_margin_tol_Ry is None else
                f" (locked at {float(davidson_margin_tol_Ry) * RY_TO_MEV:g} meV)") + ", "
             f"m_max={m_max_warm}, storing 2*m_max = {2 * m_max_warm} trial vectors")
    _price_davidson_basis(m_max_warm, X0, mesh_xy)
    eigenvalues, eigenvectors = davidson(
        apply_H, n_eig=n_solve, n_converge=n_eig, precond_fn=apply_precond, X0=X0,
        # A caller that solves many same-shape payloads (exciton_bands, one per
        # Q) names the run; the key adds everything the two programs close over.
        plan_key=(None if davidson_plan_key is None else (
            davidson_plan_key, mesh_xy, bool(include_W), head_tensor is not None,
            occupation_weights is not None,
            data.get("exchange_channel", "singlet"),
            davidson_precond, bool(davidson_olsen), float(davidson_eps_shift_Ry),
            tuple((tuple(a.shape), str(a.dtype)) for a in operator_data))),
        data=solver_data,
        m_max=m_max_warm,
        max_iter=max_iter, tol=float(davidson_tol_Ry),
        verbose=True,
        # Soft locking: a root under the tolerance adds no correction, so H
        # is applied only to the open roots' directions.  Si 8^3 lowest 40
        # at 0.01 meV (dense CPU count): 265 H applications, 800 without.
        lock_converged=True,
        margin_lock=(1.0 if davidson_margin_tol_Ry is None else
                     max(1.0, float(davidson_margin_tol_Ry) / float(davidson_tol_Ry))),
    )

    from solvers.davidson import (
        LAST_RUN as _DAV_HIST, instrumentation_summary as _dav_summary)
    log.debug(_dav_summary())

    # Keep the physical (c, v) axes distributed across all processors.
    # The writer gathers explicitly when serializing eigenvectors.
    root_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y", None))
    eigenvectors = jax.jit(
        lambda a: a[:n_eig].reshape(n_eig, 1, nc_pad, nv_pad, nk),
        out_shardings=root_sharding,
    )(eigenvectors)
    # ``davidson`` returns eigenvalues as a HOST numpy array (it needs them
    # on the host for its own convergence test).  ``replicate_to_mesh`` declares the identical
    # per-rank copy as the replica with NO collective; its precondition —
    # bit-identical on every rank — holds because every rank ran the same
    # deterministic reduction to get it.
    eigenvalues = replicate_to_mesh(np.asarray(eigenvalues)[:n_eig], mesh_xy)
    # The iterations the while_loop actually ran: Davidson stops on its
    # residual test, so this is a measured count, not the budget.
    return eigenvalues, eigenvectors, jnp.int32(int(_DAV_HIST["iter"][0]))

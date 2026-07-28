"""Static χ₀ and W computation using ISDF + minimax quadrature.

All inter-function arrays use flat k/q indices: chi(nq, μ, μ), V(nq, μ, μ), W(nq, μ, μ).
The 3D k-grid only appears inside FFT helpers.

W Dyson solve — exactly TWO plans (input key ``w_dyson_solver``):

``local`` (default)
    q-parallel shard_map: q's scattered ``P(('x','y'),None,None)``, one
    dense pivoted LU per q on the owning rank, W constrained back out to
    ``P(None,'x','y')`` through a staged relayout.  Fast at moderate P;
    every rank must hold whole (μ, μ) tiles for its q's.
``distributed``
    2-D-sharded backsolve: A_q = 1 − V_q·χ_q formed by stacked block
    GEMMs with every operand at ``P(None,'x','y')``, factored and solved
    through the ``ffi.linalg`` plan facade (ScaLAPACK ``pzgetrf`` /
    ``pzgetrs`` on CPU meshes, cuSOLVERMp on CUDA).  No rank ever
    materialises a full (μ, μ) tile — the memory ceiling that matters at
    thousands of low-memory processes.  W lands natively in
    ``P(None,'x','y')`` (no relayout).
"""
import os
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.experimental import compilation_cache as jax_compilation_cache
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import numpy as np

from common import Meta, jax_profile
from common.jax_compile_cache import ensure_jax_compile_cache
from runtime.padding import round_up, solve_at_logical
from .minimax_screening import MinimaxNodes


# ============================================================================
# Cache and sharding registry
# ============================================================================

_chi_minimax_kernel_cache: dict = {}
_w_solve_cache: dict = {}


# ============================================================================
# χ₀ kernel — minimax quadrature
# ============================================================================

def _get_chi_minimax_kernel(mesh_xy: Mesh, kgrid: tuple[int, int, int]):
    """Build chi0 kernel with device-local FFTs.  Returns flat-q χ₀(nq, μ, μ)."""
    from common.fft_helpers import make_flat_k_fftn

    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    cache_key = (id(mesh_xy), kgrid)
    if cache_key in _chi_minimax_kernel_cache:
        return _chi_minimax_kernel_cache[cache_key]

    # Flat-k FFT helpers — callers see only (nk, *trail) arrays.
    #
    # Historical form had Gv via ifftn (sign +ikR) and Gc via fftn (sign -ikR),
    # with einsum 'Rambn, Rbnam -> Rmn' swapping the μ_m/μ_n positions across
    # the two operands.  That forced Gc (or Gv) to reshard its μ sharding to
    # make the contracted index consistent, and landed chi_R in
    # P(None, 'y', 'x') — which then had to reshard AGAIN at the hand-off to
    # the W-solve (which consumes chi in P(None, 'x', 'y')).
    #
    # We exploit G's per-k Hermitian property ``G_k(μ,ν) = G_k(ν,μ)*``.  After
    # FT, ``G_R(μ,ν) = G_{-R}(ν,μ)*``, so running Gv's k→R with the SAME sign
    # as Gc's (both fftn, not one fft + one ifft) gives a Gv_R that equals
    # ``conj(original_Gv_R)`` with (μ_m, μ_n) swapped to the Gc-natural order.
    # The chi0 einsum then collapses to an element-wise product + spin sum:
    #
    #    chi_R(m,n) = Σ_{a,b} Gc_R(a,m,b,n) · conj(Gv_R(a,m,b,n))
    #
    # identical index order on both operands, no reshard.  Verified to
    # machine precision against the original formulation.
    #
    # Both Gs now share their natural 5-D sharding P(_, _, 'x', _, 'y')
    # (μ_first on x from psi_xn, μ_second on y from psi_yr).  chi_R inherits
    # P(_, 'x', 'y') naturally — aligned with V for W-solve, so the post-chi0
    # reshard into the fused W-solve drops out too.
    from .wavefunction_bundle import (
        G_FFT7D_SPEC as _G_spec,
        G_FLATK_SPEC as _G_out_flatk,
        CHI_Q_SPEC as _chi_spec,
        CHI_R_SPEC as _chi_R_spec,
        PSI_XN_SPEC as _psi_xn_spec,
        PSI_YR_SPEC as _psi_yr_spec,
    )
    _Gv_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _Gc_fftn        = make_flat_k_fftn(mesh_xy, kgrid, _G_spec,   norm='ortho')
    _chi_fftn_local = make_flat_k_fftn(mesh_xy, kgrid, _chi_spec, norm='ortho')

    from .greens_function_kernel import build_G_tau
    # Scalars / 1-D arrays replicated across all devices.
    _rep0 = P()             # scalar
    _rep1 = P(None)         # (nb,) band-indexed

    _G_k_shard = NamedSharding(mesh_xy, _G_out_flatk)
    _chi_R_shard = NamedSharding(mesh_xy, _chi_R_spec)

    @partial(jax.jit,
             in_shardings=(NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0),
                            NamedSharding(mesh_xy, _rep0)),
             out_shardings=(_G_k_shard, _G_k_shard))
    def _build_Gv_Gc(psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
                    enk_v, enk_c, tau_scalar, vmax, cmin):
        # phases_v = exp(-τ (vmax - e_v)) = exp(-(-τ)(e_v - vmax))  → t=-τ, e_ref=vmax
        # phases_c = exp(-τ (e_c - cmin))                            → t=+τ, e_ref=cmin
        Gv_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_v_xn, psi_v_yr, enk_v, -tau_scalar, e_ref=vmax),
            _G_k_shard)
        Gc_k = jax.lax.with_sharding_constraint(
            build_G_tau(psi_c_xn, psi_c_yr, enk_c,  tau_scalar, e_ref=cmin),
            _G_k_shard)
        # Hermitian-swap conj (see FFT-convention block comment above) —
        # belongs at the call site, NOT inside build_G_tau.
        return jnp.conj(Gv_k), jnp.conj(Gc_k)

    # MinimaxNodes pytree (t, alpha) — both replicated across devices.
    _nodes_shard = MinimaxNodes(
        t=NamedSharding(mesh_xy, _rep1),
        alpha=NamedSharding(mesh_xy, _rep1),
    )

    @partial(jax.jit,
             in_shardings=(_nodes_shard,
                            NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_yr_spec),
                            NamedSharding(mesh_xy, _psi_xn_spec),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep1),
                            NamedSharding(mesh_xy, _rep0),       # vmax
                            NamedSharding(mesh_xy, _rep0)),      # cmin
             out_shardings=_chi_R_shard,
             static_argnums=())
    def minimax_tau_integrate_chi(
        nodes, psi_v_xn, psi_v_yr, psi_c_yr, psi_c_xn,
        enk_v, enk_c, vmax, cmin,
    ):
        """Full τ sweep accumulating χ_R, then one R→q FFT.

        Sibling of ``ppm_sigma.minimax_tau_integrate_sigma`` — takes a
        ``MinimaxNodes`` pytree in the same slot.  For chi0 the nodes
        arrive with purely-real τ (``time_axis='real'``) and complex α
        whose Im part is zero; ``alpha`` already includes the chi0
        prefactor ``-2·α_quad·exp(-τ·E_gap)`` so the scan body only
        scales the per-τ contraction.

        For each τ node: build Gv, Gc via build_G_tau; FFT both to R;
        element-wise contract (Σ_{a,b} Gc_R · conj(Gv_R)) into chi_R;
        accumulate weighted by α; final back-FFT to q.  All collectives
        and dispatch happen inside one compiled graph — no Python loop.
        """
        n_rmu = psi_v_xn.shape[2]
        chi_R_zero = jax.lax.with_sharding_constraint(
            jnp.zeros((nk, n_rmu, n_rmu), dtype=jnp.complex128), _chi_R_shard)

        def _body(chi_R_acc, xs):
            t_scalar, alpha_scalar = xs
            # ``t`` arrives complex (pytree dtype); chi0's Laplace quad
            # places it with Im=0.  Cast to float64 so _build_Gv_Gc's
            # float64 tau signature — and build_G_tau's downstream exp —
            # stay on the exact numerical path that produced the locked
            # MoS2 3×3 regression hash.
            tau_real = jnp.real(t_scalar).astype(jnp.float64)
            Gv_k, Gc_k = _build_Gv_Gc(psi_v_xn, psi_v_yr,
                                      psi_c_yr, psi_c_xn,
                                      enk_v, enk_c, tau_real, vmax, cmin)
            Gv_R = _Gv_fftn(Gv_k)
            Gc_R = _Gc_fftn(Gc_k)
            # chi_R(m, n) = Σ_{a,b} Gc_R(a,m,b,n) · conj(Gv_R(a,m,b,n))
            chi_tau = jax.lax.with_sharding_constraint(
                jnp.einsum('Rambn,Rambn->Rmn',
                           Gc_R, jnp.conj(Gv_R), optimize=True),
                _chi_R_shard)
            # α is complex; its Im part is zero for the chi0 Laplace
            # window.  Multiplying complex·complex is identical to
            # float·complex at the hardware level when Im(α)=0.
            return chi_R_acc + alpha_scalar * chi_tau, None

        final_R, _ = jax.lax.scan(
            _body, chi_R_zero, (nodes.t, nodes.alpha))
        return _chi_fftn_local(final_R)

    # Minimax quadrature always delivers ≥1 node — the compiled scan
    # handles any n≥1 without a short-circuit wrapper.
    _chi_minimax_kernel_cache[cache_key] = minimax_tau_integrate_chi
    return minimax_tau_integrate_chi




# ============================================================================
# W solve — plan 1 of 2: LOCAL (q-parallel per-q dense LU)
# ============================================================================

def _get_w_solve_fn_local(mesh_xy: Mesh, nq: int, n_rmu: int,
                          n_rmu_logical: int | None = None):
    """W = (I - V χ)⁻¹ V via q-parallel shard_map.  All arrays flat-q: (nq, μ, μ).

    The LOCAL plan: q's are scattered over all devices
    (``P(('x','y'),None,None)``) and each rank runs one dense pivoted LU
    (``lu_factor``/``lu_solve``) per owned q.  LU is the right inner
    solve: A is SQUARE and generically well conditioned (it is I minus a
    term whose spectral radius is < 1 wherever the RPA screening is
    physical — an eigenvalue of Vχ₀ reaching 1 is a plasmon instability,
    not a numerical one).  One factorisation, one triangular pair of
    solves.

    ``n_rmu_logical``: when smaller than ``n_rmu`` (μ-padded inputs),
    the per-q pivoted LU is μ-SLICED to the logical extent and the W
    pad rows/cols are zero-filled after (their exact value: V pad rows
    are zero).  Load-bearing for device-count invariance — LU at the
    padded extent regroups partial sums per pad extent, and the
    resulting 1e-8-rel W wobble is amplified to eV on near-pole GN-PPM
    bands (reports/device_invariance_2026-07-08/ROOT_CAUSE.md, charge
    manifestation).  At zero pad the slice/fill are no-ops.
    """
    from jax.experimental.shard_map import shard_map

    n_log = int(n_rmu_logical) if n_rmu_logical is not None else int(n_rmu)
    if n_log > int(n_rmu):
        raise ValueError(
            f"_get_w_solve_fn_local: n_rmu_logical={n_log} exceeds extent {n_rmu}")
    mu_pad = int(n_rmu) - n_log

    cache_key = ("local", id(mesh_xy), nq, n_rmu, n_log)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    q_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    # ── W COMES OUT 2-D SHARDED: W_q(μ_X, ν_Y) ────────────────────────
    # This used to be ``rep_3d = P(None, None, None)`` — a full
    # all-gather of the whole (nq, μ, μ) stack onto every rank, and the
    # last replicated O(nq·μ²) object in the production path (scorecard
    # J.2 #3: nq·μ²·16 per rank, ×2 for the static+probe pair, break-μ
    # ≈ 4.4 k, re-paid every SC iteration).  Nothing wanted it: every
    # consumer either is layout-agnostic (sigma_dispatch, sc_iteration,
    # gw_jax) or immediately re-imposes exactly this layout —
    # ``symmetry_maps.unfold_v_q`` (P(None,'x','y') in and out),
    # ``cohsex_sigma._convolve``'s 5-D V_FFT5D_SPEC = P(None,None,None,
    # 'x','y'), ``ppm_sigma.fit_ppm``'s q_shard, and
    # ``head_wing_schur`` which literally undid the replication by hand.
    # The ONLY q-index anywhere is ``screening.py``'s ``W[0]``
    # hermiticity gate, which is a two-reduction check on one tile.
    # The distributed plan has always returned P(None,'x','y'), so the
    # whole downstream chain is already proven on this layout.
    #
    # Collective-wise the final constraint changes from an ALL-GATHER
    # (every rank ends holding nq·μ²·16 B) to an ALL-TO-ALL (nq·μ²·16/P
    # per rank).  The values are untouched — the shard_map above
    # computes the same numbers either way and this is pure data
    # movement — so the change is bit-exact by construction.
    nat_3d = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    # Intermediate sharding for the reshard from P(None,'x','y') → q_shard.
    # Routing through P('x',None,'y') (x parks on nq, y stays on μ₂) lets
    # SPMD plan it as two single-axis all_to_alls instead of the
    # "Involuntary full rematerialization" it falls into when asked to
    # un-shard both x and y on μ simultaneously via a fully replicated
    # intermediate.  Measured at Si 4×4×4 60Ry (nq=64, μ=1200, 2×2 mesh):
    #   via a fully replicated P(None,None,None) intermediate:
    #                          peak 2.95 GB/dev (temp 2.21 GB) — Involuntary Remat
    #   via P('x',None,'y'): peak 1.11 GB/dev (temp 0.37 GB)  -- 62% reduction
    reshard_mid = NamedSharding(mesh_xy, P('x', None, 'y'))
    q_spec = P(('x', 'y'), None, None)

    # ``chi_flat`` is donated (position 1): the caller releases χ₀ right
    # after this call (module contract, same as the distributed plan —
    # the ``del chi0_q_solve`` inside ``screening.py``'s ``W.exec``
    # timing block).  ``V_flat`` is NOT donated — V is reused
    # by COHSEX Σ_SX, Σ_COH, Σ_X and the PPM fit's Wc = W - V step.
    @partial(jax.jit, donate_argnums=(1,))
    def _solve_w(V_flat: jax.Array, chi_flat: jax.Array, pref: jax.Array) -> jax.Array:
        """V_flat, chi_flat: (nq, μ, μ).  Returns W: (nq, μ, μ)."""
        nq_local = V_flat.shape[0]
        n = V_flat.shape[1]
        chi_scaled = pref * chi_flat

        # Pad to device count then reshard to q-parallel
        total_devices = mesh_xy.devices.size
        nq_padded = round_up(nq_local, total_devices)
        pad = nq_padded - nq_local
        V_padded = jnp.pad(V_flat, ((0, pad), (0, 0), (0, 0))) if pad > 0 else V_flat
        chi_padded = jnp.pad(chi_scaled, ((0, pad), (0, 0), (0, 0))) if pad > 0 else chi_scaled
        V_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(V_padded, reshard_mid), q_shard)
        chi_q = jax.lax.with_sharding_constraint(
            jax.lax.with_sharding_constraint(chi_padded, reshard_mid), q_shard)

        def _local_solve(V_local, chi_local):
            nq_dev = V_local.shape[0]

            def _dyson_log(V_log, chi_log):
                # Solve at the LOGICAL μ extent (see _get_w_solve_fn_local
                # docstring; slice/zero-refill via solve_at_logical).
                # V/χ pad rows are exact zeros, so the sliced system IS
                # the logical Dyson system; the W pad block is exactly
                # zero (A_pad = I, RHS_pad = 0).
                A = jnp.eye(n_log, dtype=V_log.dtype) - V_log @ chi_log
                lu, piv = jsp_linalg.lu_factor(A)
                return jsp_linalg.lu_solve((lu, piv), V_log)

            def solve_one(iq, W_acc):
                W_row = solve_at_logical(
                    _dyson_log, n_log, (V_local[iq], chi_local[iq]),
                    pad_axes=(-2, -1))
                return jax.lax.dynamic_update_slice(
                    W_acc, W_row[None, :, :], (iq, 0, 0))
            return jax.lax.fori_loop(0, nq_dev, solve_one, jnp.zeros_like(V_local))

        W_flat = shard_map(
            _local_solve, mesh=mesh_xy,
            in_specs=(q_spec, q_spec), out_specs=q_spec,
        )(V_q, chi_q)

        if pad > 0:
            W_flat = W_flat[:nq_local]
        # Land W on P(None,'x','y') through the SAME single-axis staging the
        # input reshard uses, in reverse:
        #     q-parallel [px·py,1,1] -> P('x',None,'y') [px,1,py]
        #                            -> P(None,'x','y') [1,px,py]
        # Asking SPMD for the composite in ONE step makes it
        # replicate-then-partition: MEASURED on a real 2×2 CUDA mesh and on
        # CPU, `[SPMD] Involuntary full rematerialization ... cannot go from
        # {devices=[4,1,1]} to {devices=[1,2,2]} ... op_name=
        # "jit(_solve_w)/shard_map"`, 1 per rank, where the base (replicated)
        # output produced none.  That transient is the whole nq·μ² object
        # this change exists to stop materialising, so the staging is
        # load-bearing, not cosmetic.  Each stage moves ONE mesh axis, which
        # is a single all_to_all — the same reasoning (and the same
        # `reshard_mid`) as the 62 %-peak-reduction note above.
        W_flat = jax.lax.with_sharding_constraint(W_flat, reshard_mid)
        return jax.lax.with_sharding_constraint(W_flat, nat_3d)

    _w_solve_cache[cache_key] = _solve_w
    return _solve_w


# ============================================================================
# W solve — plan 2 of 2: DISTRIBUTED (2-D-sharded stacked-GEMM backsolve)
# ============================================================================

def _get_w_solve_fn_distributed(mesh_xy: Mesh, nq: int, n_rmu: int,
                                n_rmu_logical: int):
    """W = solve(A, V), A = (1 − pref·V·χ₀), everything 2-D sharded.

    The DISTRIBUTED plan — the scale-out route for thousands of
    low-memory processes, in the same architectural family as the
    ζ-fit's distributed rank-truncate tier
    (:func:`isdf.core._factor_c_q_distributed_rank_truncate`):

    1. **A build** — per q-block, ``A = I − V·(pref·χ)`` as a 2-D block
       GEMM inside ``shard_map``: rank (x, y) all-gathers V's row block
       along 'y' (full k for its i rows, μ·μ/Px per rank) and χ's column
       block along 'x' (full k for its j columns, μ·μ/Py per rank),
       multiplies locally, and subtracts from its identity tile.  The
       gathers are STRUCTURAL — inside shard_map the partitioner cannot
       hoist them into a full-stack gather (the per_q-tier lesson,
       quality pattern #4).  The q loop is chunked HOST-side so one
       collective instruction never exceeds ``LORRAX_COLLECTIVE_CHUNK_MB``
       (the AF transport bound; separate XLA executions cannot be
       re-combined by a compiler pass).
    2. **Factor + backsolve** — ONE resolved
       :class:`ffi.linalg.plan.LinalgPlan` for ``solve_lu`` with
       ``backend='distributed'`` (ScaLAPACK ``pzgetrf``/``pzgetrs`` on a
       CPU mesh, cuSOLVERMp on CUDA — ``resolve._DISTRIBUTED_DEFAULT``),
       consuming the block-cyclic tiles where they already live.

    **No rank ever materialises a full (μ, μ) tile**: inputs, A, the LU
    factors and W all stay ``P(None,'x','y')`` (per-rank blocks of
    μ/Px × μ/Py; the largest per-rank transient is the μ·μ/min(Px,Py)
    gathered GEMM operand).  W lands natively in ``P(None,'x','y')`` —
    no relayout, unlike the local plan.

    Padding contract, and why it is exact: V and χ pad rows/cols are
    exact zeros (the bilinear-in-zero-padded-ψ contract), so at the
    PADDED extent ``A = [[A_log, 0], [0, I]]`` and ``RHS = [[V_log], [0]]``
    hold EXACTLY — the identity-embedded block-diagonal system whose
    solution is ``[[W_log], [0]]``; partial pivoting cannot mix the
    blocks (every pad column is a unit vector, every pad row is zero in
    the logical columns).  W's pad rows/cols are masked to exact zeros
    after the solve (same contract as ``solve_at_logical``'s
    zero-refill on the local plan).  Unlike the local plan the LOGICAL
    block is formed/factored at the padded extent, so W here carries the
    ≤1e-8-rel pad-extent regrouping wobble — which is subsumed by the
    block-cyclic factorisation's own non-bit-identity; this plan's
    numerical contract is the Dyson residual (``LORRAX_W_RESIDUAL_CHECK``),
    not bit-identity with the local plan.

    Geometry/capability failures (host lib absent, non-square or 1-D
    mesh, n not divisible, process coverage) RAISE at resolve time with
    the resolver's own message — an explicitly requested distributed
    solve never silently downgrades to the local plan (quality pattern
    #6/#8).
    """
    n_ext = int(n_rmu)
    n_log = int(n_rmu_logical)
    if n_log > n_ext:
        raise ValueError(
            f"_get_w_solve_fn_distributed: n_rmu_logical={n_log} exceeds "
            f"extent {n_ext}")

    cache_key = ("distributed", id(mesh_xy), nq, n_ext, n_log)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    from jax.experimental.shard_map import shard_map
    from ffi.linalg.plan import plan as linalg_plan
    # House chunking pattern — single source (scorecard AF): one emitted
    # collective's payload is bounded by LORRAX_COLLECTIVE_CHUNK_MB.
    # TODO(release): promote _chunk_q/_chunk_log to a public home (e.g.
    # common/collectives.chunk_q/chunk_log) and import them publicly from
    # both isdf/core and here — gw physics code should not reach into
    # another package's underscore-private namespace, and _chunk_log's
    # module-global dedup set is cross-package shared mutable state with
    # no public contract (audit fix/zq 2026-07-28, _idx 29; needs an
    # isdf/core edit, outside this fix's file set).
    from isdf.core import _chunk_q, _chunk_log

    # Every guard fires HERE (vocabulary, platform, capability, process
    # coverage, mesh geometry — including the 1-D-mesh cusolvermp refusal
    # — and divisibility) — resolve.py's ladder, with its own messages.
    # ``distributed`` maps to the platform default (ScaLAPACK on cpu,
    # cuSOLVERMp on CUDA) in ONE place.  No compensating ``p.is_native``
    # re-check is needed: an explicit request that cannot be honored
    # raises at resolve time (the former silent 1-D-mesh degenerate-to-
    # native was removed; audit fix/zq 2026-07-28), so a returned plan is
    # always the distributed backend it names.
    p = linalg_plan("solve_lu", mesh_xy, backend="distributed", n=n_ext)

    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    if jax.process_index() == 0:
        print(f"  [W solve] w_dyson_solver=distributed -> {p.describe()}",
              flush=True)

    # The two collectives ``_a_local`` emits, per q (2-D block GEMM):
    #   all_gather('y')  V   (μ/Px, μ/Py) -> (μ/Px, μ)  = μ²/Px · 16 B
    #   all_gather('x')  χ   (μ/Px, μ/Py) -> (μ, μ/Py)  = μ²/Py · 16 B
    # The BIGGER of the two sets the q-block (see ``_chunk_q``).
    per_q_coll = max(n_ext * (n_ext // px), n_ext * (n_ext // py)) * 16

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(P(None, 'x', 'y'), P(None, 'x', 'y')),
             out_specs=P(None, 'x', 'y'), check_rep=False)
    def _a_local(V_loc, chi_loc):
        # A[q,i,j] = δ_ij − Σ_k V[q,i,k]·χs[q,k,j] on my (i on 'x',
        # j on 'y') tile.  Classic 2-D block GEMM pairing — same shape
        # of communication as ``isdf.core._distributed_pinv_apply``.
        V_row = jax.lax.all_gather(V_loc, 'y', axis=2, tiled=True)
        chi_col = jax.lax.all_gather(chi_loc, 'x', axis=1, tiled=True)
        prod = jnp.einsum('qik,qkj->qij', V_row, chi_col)
        i0 = jax.lax.axis_index('x') * (n_ext // px)
        j0 = jax.lax.axis_index('y') * (n_ext // py)
        eye_tile = jnp.equal(
            i0 + jnp.arange(n_ext // px)[:, None],
            j0 + jnp.arange(n_ext // py)[None, :]).astype(V_loc.dtype)
        return eye_tile[None, :, :] - prod

    @partial(jax.jit, donate_argnums=(2,), out_shardings=nat)
    def _a_chunk(V_blk, chi_blk, A_acc, q0):
        return jax.lax.dynamic_update_slice(
            A_acc, _a_local(V_blk, chi_blk), (q0, 0, 0))

    # χ is donated here (module contract: the caller releases χ₀ after
    # solve_w — see screening.py's ``del chi0_q_solve``).
    _scale = jax.jit(lambda c, pref: pref * c,
                     donate_argnums=(0,), out_shardings=nat)
    _zeros_like = jax.jit(jnp.zeros_like, out_shardings=nat)
    # RHS must be a FRESH buffer, never an alias of the caller's V —
    # the FFI backsolve DONATES both operands (docs/dev/linalg_ffi.md
    # "Sharp edges") and V is still needed by Σ_SX/Σ_COH/Σ_X and the
    # PPM fit's Wc = W − V.
    _copy = jax.jit(jnp.copy, out_shardings=nat)

    if n_log < n_ext:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=P(None, 'x', 'y'), out_specs=P(None, 'x', 'y'),
                 check_rep=False)
        def _mask_pads_local(W_loc):
            # W pad rows/cols → exact zeros (they are already exact by
            # the block-diagonal argument above; the mask makes the
            # contract structural, mirroring the ζ tier's pad-row mask).
            i0 = jax.lax.axis_index('x') * (n_ext // px)
            j0 = jax.lax.axis_index('y') * (n_ext // py)
            ri = (i0 + jnp.arange(n_ext // px)) < n_log
            cj = (j0 + jnp.arange(n_ext // py)) < n_log
            return jnp.where(ri[None, :, None] & cj[None, None, :], W_loc, 0)
        _mask_pads = jax.jit(_mask_pads_local, donate_argnums=(0,))
    else:
        _mask_pads = None

    def _solve_w_dist(V_flat: jax.Array, chi_flat: jax.Array,
                      pref: jax.Array) -> jax.Array:
        """V_flat, chi_flat: (nq, μ, μ) at P(None,'x','y').  Returns W
        (nq, μ, μ) at P(None,'x','y').  chi_flat's buffer is consumed."""
        nq_local = int(V_flat.shape[0])
        qb = _chunk_q(nq_local, per_q_coll)
        _chunk_log('W Dyson A-build (GEMM)', nq_local, qb, per_q_coll)
        chi_scaled = _scale(chi_flat, pref)
        A = _zeros_like(V_flat)
        # Host-level q-block loop: ONE XLA execution per block, so the
        # emitted all_gather payloads are bounded by construction and
        # cannot be re-combined by a compiler pass (AF note in
        # isdf/core).  At most two compiled shapes (full + remainder).
        for q0 in range(0, nq_local, qb):
            q1 = min(q0 + qb, nq_local)
            A = _a_chunk(V_flat[q0:q1], chi_scaled[q0:q1], A, q0)
        B = _copy(V_flat)
        # ONE plan call for the whole stack: one descriptor, one
        # workspace; A and B are donated into the FFI.
        W = p.batched(A, B)
        if _mask_pads is not None:
            W = _mask_pads(W)
        # House falsy vocabulary — same parse (and same rationale comment)
        # as common/collectives.py's LORRAX_CHECK_REPLICA fix (workstream
        # AT): the narrow "0"/""/"false" tuple this replaced meant
        # LORRAX_W_RESIDUAL_CHECK=off/no/False silently ENABLED the
        # diagnostic — which must be OFF when taking collective-table
        # probes (docs/dev/env_vars.md).  (audit fix/zq 2026-07-28)
        if os.environ.get("LORRAX_W_RESIDUAL_CHECK", "0").strip().lower() \
                not in ("", "0", "false", "no", "off"):
            _w_residual_report(V_flat, chi_scaled, W, n_ext)
        return W

    _w_solve_cache[cache_key] = _solve_w_dist
    return _solve_w_dist


def _w_residual_report(V_flat, chi_scaled, W, n_ext, n_check: int = 4):
    """Direct Dyson residual ‖(1−Vχ)W − V‖/‖V‖ on the first few q.

    THE strict numerical contract of the distributed plan (a
    block-cyclic LU is not bit-comparable to the local per-q LU; the
    residual is what certifies the solve — quality pattern #6, "test
    what executes").  Diagnostic-only, opt-in via
    ``LORRAX_W_RESIDUAL_CHECK=1``; never on in the traced production
    path, so the collective-table gate is taken with it OFF.
    """
    ns = min(int(V_flat.shape[0]), int(n_check))

    @jax.jit
    def _res(V_s, chi_s, W_s):
        A_s = jnp.eye(n_ext, dtype=V_s.dtype)[None, :, :] - V_s @ chi_s
        num = jnp.linalg.norm((A_s @ W_s - V_s).reshape(ns, -1), axis=1)
        den = jnp.linalg.norm(V_s.reshape(ns, -1), axis=1)
        return num / den

    r = np.asarray(jax.device_get(_res(V_flat[:ns], chi_scaled[:ns], W[:ns])))
    if jax.process_index() == 0:
        vals = "  ".join(f"q{iq}={v:.3e}" for iq, v in enumerate(r))
        print(f"  [W solve] Dyson residual |(1-Vchi)W - V|/|V| ({ns} q): "
              f"{vals}  max={r.max():.3e}", flush=True)


def _w_solve_pref_scalar(meta) -> float:
    """The 2/(√N_k · n_spin · n_spinor) prefactor in front of χ₀ in the
    Dyson solve.  Same value for both plans; pulled out so the dispatch
    helper below isn't the only place it's computed."""
    nq = int(meta.nk_tot)
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    return 2.0 / (float(max(1, nq)) ** 0.5 * float(nspin) * float(nspinor))


def _resolve_w_solve_fn(meta, mesh_xy, *, n_rmu, dyson_solver=None):
    """Return ``(solve_fn, pref)`` for the requested W plan.

    Single source of truth for the two-plan dispatch.  Both ``solve_w``
    and ``precompile_solve_w`` go through this helper — the dispatch
    logic exists in one place.

    ``dyson_solver`` (input key ``w_dyson_solver``) selects the plan:

    ``local`` (default; ``auto`` is an alias)
        per-q pivoted LU inside the q-parallel shard_map —
        :func:`_get_w_solve_fn_local`.
    ``distributed``
        the 2-D-sharded stacked-GEMM backsolve through the linalg plan
        facade — :func:`_get_w_solve_fn_distributed`.  Refuses loudly at
        resolve time when the mesh/build cannot run it; never silently
        downgrades.

    W comes out ``P(None,'x','y')`` on BOTH — that is the module's
    output contract, not a per-plan detail.
    """
    from .gw_config import normalize_w_dyson_solver
    dyson = normalize_w_dyson_solver(dyson_solver)
    nq = int(meta.nk_tot)
    pref_scalar = _w_solve_pref_scalar(meta)

    # ``meta.n_rmu`` is a HARD read: a soft getattr fallback here
    # silently restored the padded-extent LU for any meta-like object
    # missing the field (opt-out-by-omission, PADDING_AUDIT item 3).
    # Synthetic-meta callers must carry n_rmu (= the logical extent).
    n_log = int(meta.n_rmu)

    if dyson == "distributed":
        solve_fn = _get_w_solve_fn_distributed(mesh_xy, nq, n_rmu, n_log)
    else:
        solve_fn = _get_w_solve_fn_local(
            mesh_xy, nq, n_rmu, n_rmu_logical=n_log)
    return solve_fn, jnp.asarray(pref_scalar, dtype=jnp.complex128)


def solve_w(V_q, chi0_q, meta, mesh_xy, *, dyson_solver=None):
    """W(q) = (I − V χ₀)⁻¹ V  via a Dyson solve.  **W comes out sharded.**

    All arrays flat-q: V(nq, μ, μ), χ₀(nq, μ, μ) → W(nq, μ, μ).

    **Output contract:** ``W`` is ``P(None, 'x', 'y')`` — 2-D sharded
    W_q(μ_X, ν_Y) — on both plans, and stays that way into its
    consumers (Σ_SX/Σ_COH's 5-D FFT spec, the PPM fit, the IBZ unfold,
    the restart writer).

    ``dyson_solver`` (input key ``w_dyson_solver``) picks one of the
    TWO plans — see :func:`_resolve_w_solve_fn`:

    - ``local`` (default): q-parallel reshard + per-q dense LU via
      shard_map.  Legal on any mesh; each rank holds whole (μ, μ)
      tiles for its q's.
    - ``distributed``: 2-D-sharded stacked-GEMM backsolve through the
      ffi.linalg plan facade (ScaLAPACK on CPU, cuSOLVERMp on CUDA).
      No rank ever materialises a full (μ, μ) tile — the P→∞ memory
      ceiling.  Slower than ``local`` at moderate P; that is priced and
      accepted (the point is the per-rank memory ceiling, not speed).

    ``chi0_q``'s buffer is CONSUMED (donated) on both plans — the
    caller must drop its reference after this call.
    """
    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy, n_rmu=chi0_q.shape[1], dyson_solver=dyson_solver)
    with jax_profile.annotation("W_solve"):
        return solve_fn(V_q, chi0_q, pref)


def compute_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=0.0):
    """Compute χ₀(q) from a wavefunction bundle and minimax quadrature.

    Returns flat-q array (nq, μ, μ).

    ``quad.tau`` and ``quad.alpha`` approximate either 1/x (static) or
    x/(x²+ωp²) (imaginary-frequency) on [x_min, x_max] where x = E_c - E_v.
    The physical χ₀ is::

        χ₀ = -2 Σ_ℓ α_ℓ Σ_{v,c} |M_vc|² exp(-τ_ℓ (E_c - E_v))

    A uniform energy shift via ``energy_reference`` is applied to both
    valence and conduction energies before building the minimax factors.
    Because only differences enter, this is algebraically invariant; the
    knob lets callers align the global zero (e.g. midgap, VBM, CBM).
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))

    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    eref = 0.0 if energy_reference is None else float(energy_reference)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax

    tau = np.asarray(quad.tau, dtype=np.float64)
    # Fold the chi0 prefactor (-2 · exp(-τ·E_gap)) into α so the τ-scan
    # body can apply a single weighted add per node.  ``MinimaxNodes``
    # carries both in complex128; τ has Im=0 for the Laplace quad.
    alpha_chi = -2.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid)
    return kernel(
        nodes,
        wfns.xn(s.val), wfns.yr(s.val),
        wfns.yr(s.cond), wfns.xn(s.cond),
        enk_v - jnp.asarray(eref, dtype=enk_v.dtype),
        enk_c - jnp.asarray(eref, dtype=enk_c.dtype),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    )


def precompile_chi0(wfns, quad, meta, mesh_xy, *, energy_reference=None):
    """AOT lower+compile of the χ₀ minimax kernel at the real input
    shapes/shardings — warms the JAX in-process cache so the first
    ``compute_chi0`` call is execution-only.  Call inside a dedicated
    ``timing.section('chi0_W.chi.compile')`` block to separate compile
    from exec in the end-of-run timing report.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    eref = 0.0 if energy_reference is None else float(energy_reference)
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64) - eref
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64) - eref
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    E_gap = cmin - vmax
    tau = np.asarray(quad.tau, dtype=np.float64)
    if len(tau) == 0:
        return  # compute_chi0 falls through to a static-zeros path — nothing to compile
    alpha_chi = -2.0 * np.asarray(quad.alpha, dtype=np.float64) * np.exp(-tau * E_gap)
    nodes = MinimaxNodes(
        t=jnp.asarray(tau, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_chi, dtype=jnp.complex128),
    )

    kernel = _get_chi_minimax_kernel(mesh_xy, kgrid)
    kernel.lower(
        nodes,
        wfns.xn(s.val), wfns.yr(s.val),
        wfns.yr(s.cond), wfns.xn(s.cond),
        enk_v - jnp.asarray(eref, dtype=enk_v.dtype),
        enk_c - jnp.asarray(eref, dtype=enk_c.dtype),
        jnp.asarray(vmax, dtype=jnp.float64),
        jnp.asarray(cmin, dtype=jnp.float64),
    ).compile()


def precompile_solve_w(V_q, chi0_q, meta, mesh_xy, *, dyson_solver=None):
    """AOT lower+compile of the W-solve jit.  See ``precompile_chi0``.

    Goes through the same ``_resolve_w_solve_fn`` dispatch as
    :func:`solve_w` so both paths agree on which jit to compile.
    """
    ensure_jax_compile_cache()
    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy, n_rmu=chi0_q.shape[1], dyson_solver=dyson_solver)
    # The DISTRIBUTED plan is a plain function around chunked jits + one
    # FFI call, not a single jit, so there is nothing to lower here —
    # the first real call builds the BLACS descriptor and compiles its
    # own modules (scorecard L §5, amortised from call 2; the ζ tier
    # behaves the same way).
    if not hasattr(solve_fn, "lower"):
        return
    solve_fn.lower(V_q, chi0_q, pref).compile()

"""Static χ₀ and W computation using ISDF + minimax quadrature.

All inter-function arrays use flat k/q indices: chi(nq, μ, μ), V(nq, μ, μ), W(nq, μ, μ).
The 3D k-grid only appears inside FFT helpers.
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
# W solve with two-stage resharding (following load_wfns pattern)
# ============================================================================

def _get_w_solve_fn(mesh_xy: Mesh, nq: int, n_rmu: int,
                    n_rmu_logical: int | None = None):
    """W = (I - V χ)⁻¹ V via q-parallel shard_map.  All arrays flat-q: (nq, μ, μ).

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
            f"_get_w_solve_fn: n_rmu_logical={n_log} exceeds extent {n_rmu}")
    mu_pad = int(n_rmu) - n_log

    cache_key = (id(mesh_xy), nq, n_rmu, n_log)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    q_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    rep_3d = NamedSharding(mesh_xy, P(None, None, None))
    # Intermediate sharding for the reshard from P(None,'x','y') → q_shard.
    # Routing through P('x',None,'y') (x parks on nq, y stays on μ₂) lets
    # SPMD plan it as two single-axis all_to_alls instead of the
    # "Involuntary full rematerialization" it falls into when asked to
    # un-shard both x and y on μ simultaneously via a fully replicated
    # intermediate.  Measured at Si 4×4×4 60Ry (nq=64, μ=1200, 2×2 mesh):
    #   via rep_3d: peak 2.95 GB/dev (temp 2.21 GB) — SPMD Involuntary Remat
    #   via P('x',None,'y'): peak 1.11 GB/dev (temp 0.37 GB)  -- 62% reduction
    reshard_mid = NamedSharding(mesh_xy, P('x', None, 'y'))
    q_spec = P(('x', 'y'), None, None)

    # ``chi_flat`` is donated (position 1): the caller releases χ₀ right
    # after this call (see ``gw_jax.main`` — the ``del chi0_q`` inside the
    # ``W.exec`` timing block).  ``V_flat`` is NOT donated — V is reused
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
                # Solve at the LOGICAL μ extent (see _get_w_solve_fn
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
        return jax.lax.with_sharding_constraint(W_flat, rep_3d)

    _w_solve_cache[cache_key] = _solve_w
    return _solve_w


def _get_w_solve_fn_low_mem(mesh_xy: Mesh, nq: int, n_rmu: int, dtype):
    """Low-mem W-solve: fused cuBLASMp + cuSOLVERMp FFI.

    One distributed FFI call runs the entire symmetric Cholesky Dyson
    solve inside the device:
        v = X X†                 (cusolverMp potrf)
        H = I − X† (pref·χ) X    (2 cublasMp gemms + identity kernel)
        L_H = chol(H)            (cusolverMp potrf)
        W = X H⁻¹ X†             (2 cublasMp trsms + 1 cublasMp gemm)

    No JAX-level intermediates, no opportunity for XLA to reshard or
    rematerialize — all matmuls are distributed-native in-device.
    """
    from ffi.cublasmp import batched_fused_w_solve
    from jax.sharding import NamedSharding

    cache_key = ("low_mem_fused", id(mesh_xy), nq, n_rmu, dtype)
    if cache_key in _w_solve_cache:
        return _w_solve_cache[cache_key]

    nat = NamedSharding(mesh_xy, P(None, "x", "y"))

    def _solve_w_low(V_q, chi0_q, pref):
        # The fused FFI reads pref as a compile-time complex scalar attr,
        # so pref must be a Python scalar here (not a jnp array).
        # chi0_q now comes out of compute_chi0 in P(None, 'x', 'y')
        # (same as V_q) — no reshard needed at the hand-off.
        V_q    = jax.lax.with_sharding_constraint(V_q, nat)
        chi0_q = jax.lax.with_sharding_constraint(chi0_q, nat)
        return batched_fused_w_solve(V_q, chi0_q, pref, mesh=mesh_xy)

    _w_solve_cache[cache_key] = _solve_w_low
    return _solve_w_low


def _w_solve_pref_scalar(meta) -> float:
    """The 2/(√N_k · n_spin · n_spinor) prefactor in front of χ₀ in the
    Dyson solve.  Same value for both backends; pulled out so the
    dispatch helper below isn't the only place it's computed."""
    nq = int(meta.nk_tot)
    nspin = max(1, int(getattr(meta, 'nspin', 1)))
    nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
    return 2.0 / (float(max(1, nq)) ** 0.5 * float(nspin) * float(nspinor))


def _normalize_screening_solver(solver_or_mode):
    """Accept either a :class:`ScreeningSolver` or the legacy string
    (``"high_mem"``/``"low_mem"``/``"auto"``) and return the enum.

    Kept narrow so the only place legacy strings cross over is here.
    """
    from .gw_config import ScreeningSolver, _LEGACY_ISDF_MEMORY_MODE
    if isinstance(solver_or_mode, ScreeningSolver):
        return solver_or_mode
    s = (solver_or_mode or "auto").strip().lower()
    if s in _LEGACY_ISDF_MEMORY_MODE:
        return _LEGACY_ISDF_MEMORY_MODE[s]
    raise ValueError(
        f"solver={solver_or_mode!r} invalid; pass a ScreeningSolver enum "
        f"or one of {sorted(_LEGACY_ISDF_MEMORY_MODE)}."
    )


def _resolve_w_solve_fn(meta, mesh_xy, *, solver, dtype, n_rmu):
    """Return ``(solve_fn, pref)`` for the requested screening solver.

    Single source of truth for the JAX-native vs cuBLASMp-FFI fork.
    Both ``solve_w`` and ``precompile_solve_w`` go through this helper —
    the dispatch logic exists in one place.
    """
    from .gw_config import ScreeningSolver
    solver = _normalize_screening_solver(solver)
    nq = int(meta.nk_tot)
    pref_scalar = _w_solve_pref_scalar(meta)

    if solver is ScreeningSolver.CUBLASMP_FFI:
        # Fused FFI consumes pref as a Python complex scalar (compile-time attr).
        # NOTE: the distributed FFI solve runs at the PADDED extent (its
        # block-cyclic layout needs mesh divisibility) — it retains the
        # ≤1e-8-rel pad-extent sensitivity the JAX_NATIVE path removes.
        solve_fn = _get_w_solve_fn_low_mem(mesh_xy, nq, n_rmu, dtype)
        return solve_fn, complex(pref_scalar)

    # JAX_NATIVE (q-parallel reshard + per-rank LU via shard_map).
    # Per-q LU runs at the LOGICAL μ extent (meta.n_rmu); W pad
    # rows/cols are zero-filled — see _get_w_solve_fn.  ``meta.n_rmu``
    # is a HARD read: a soft getattr fallback here silently restored
    # the padded-extent LU for any meta-like object missing the field
    # (opt-out-by-omission, PADDING_AUDIT item 3).  Synthetic-meta
    # callers must carry n_rmu (= the logical array extent).
    solve_fn = _get_w_solve_fn(
        mesh_xy, nq, n_rmu, n_rmu_logical=int(meta.n_rmu))
    return solve_fn, jnp.asarray(pref_scalar, dtype=jnp.complex128)


def solve_w(V_q, chi0_q, meta, mesh_xy, *, solver=None, memory_mode=None):
    """W(q) = (I − V χ₀)⁻¹ V  via q-parallel Dyson solve.

    All arrays flat-q: V(nq, μ, μ), χ₀(nq, μ, μ) → W(nq, μ, μ).

    Pass either ``solver`` (a :class:`ScreeningSolver` enum) or the
    legacy ``memory_mode`` string ("high_mem" / "low_mem" / "auto").
    Legacy callers that still hand in the string keep working —
    ``_normalize_screening_solver`` does the coercion in one place.

    Solver semantics:

    - :attr:`ScreeningSolver.JAX_NATIVE` (default): q-parallel reshard
      + per-rank LU via shard_map.  Legal on any mesh, one all-gather +
      one all-scatter of (μ, μ) blocks.
    - :attr:`ScreeningSolver.CUBLASMP_FFI`: fused symmetric Cholesky
      W = X H⁻¹ X†; no reshard to q-parallel, but matmuls can reshard
      internally (JAX-planned).  Requires χ such that I − X†χX is PD.
    """
    chosen = solver if solver is not None else memory_mode
    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy,
        solver=chosen, dtype=V_q.dtype, n_rmu=chi0_q.shape[1],
    )
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
    alpha_chi = -2.0 * np.asarray(quad.alpha, dtype=np.complex128) * np.exp(-tau * E_gap)
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
    alpha_chi = -2.0 * np.asarray(quad.alpha, dtype=np.complex128) * np.exp(-tau * E_gap)
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


def precompile_solve_w(V_q, chi0_q, meta, mesh_xy, *, solver=None, memory_mode=None):
    """AOT lower+compile of the W-solve jit.  See ``precompile_chi0``.

    Goes through the same ``_resolve_w_solve_fn`` dispatch as
    :func:`solve_w` so both paths agree on which jit to compile.  The
    cuBLASMp FFI path uses a slightly different ``.lower(V_q, chi0_q)``
    signature (pref is folded into the compile-time attribute set, not
    a runtime arg) so the precompile is a thin per-solver branch here.
    """
    from .gw_config import ScreeningSolver
    ensure_jax_compile_cache()
    chosen = solver if solver is not None else memory_mode
    solver_enum = _normalize_screening_solver(chosen)
    nq = int(meta.nk_tot)
    n_rmu = chi0_q.shape[1]

    if solver_enum is ScreeningSolver.CUBLASMP_FFI:
        # Also primes the cuBLASMp context handle via get_or_init_context.
        from ffi.cublasmp import batched_fused_w_solve_jit
        pref_scalar = _w_solve_pref_scalar(meta)
        jit_fn = batched_fused_w_solve_jit(
            dtype=V_q.dtype, nq=nq, n=n_rmu,
            pref=complex(pref_scalar), mesh=mesh_xy,
        )
        jit_fn.lower(V_q, chi0_q).compile()
        return

    solve_fn, pref = _resolve_w_solve_fn(
        meta, mesh_xy,
        solver=ScreeningSolver.JAX_NATIVE, dtype=V_q.dtype, n_rmu=n_rmu,
    )
    solve_fn.lower(V_q, chi0_q, pref).compile()

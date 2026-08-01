"""Device τ-kernel unit for the Σ_c(ω) GN-PPM integration.

The single-tau integrand kernel plus its cache/AOT machinery:

    σ^τ_nmk(τ) = project[ FFT[ G(τ) · W(τ) / √N_k ] ]
    G(τ)       = diag[ e^{-i(E_A - E_ref_A)·τ} ] · mask_A           (A = val or cond)
    W(τ)       = Σ_μν  B_q · e^{-i(Ω_q - E_ref_B)·τ}  · mask_B      (PPM pole sum)

This is the Σ_PPM file where SPMD / sharding / HLO expertise is required —
the deferred scan / collective-flush notes live here.  The projection tail
itself (the two-stage psum_scatter band reshard, its axis-order/stacking/
de-promotion levers and the gated MKL-GEMM FFI body) is SUBSUMED by the
shared primitive ``common.contract_bands.contract_bands_block_reshard``
(owner directive 2026-07-28) — this module keeps only the Σ-specific
channel algebra and the kernel plumbing around it.

The module-level kernel caches (`_sigma_tau_kernel_cache`,
`_sigma_kij_kernel_cache`) are co-located with the factories that read them.
This is load-bearing: ``precompile_sigma`` (the AOT prewarm called from
``ppm_pipeline``) must hit the *same* cache dicts as the runtime path, or the
first per-τ dispatch pays a full compile inside execution.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common import timing
from common.jax_compile_cache import ensure_jax_compile_cache


_sigma_tau_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}
_sigma_kij_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}


def _stage_timing_enabled() -> bool:
    """``LORRAX_SIGMA_TAU_TIMING=1`` selects the stage-split instrumented τ kernel.

    Diagnostic knob (2026-07-28; evidence: AQ 4962c/P=64 HLO module_0912 —
    'sigma.exec 272.040' is a single opaque row, 176 τ dispatches at a uniform
    ~1.51 s that no existing timing row decomposes).  When ON, the per-τ body
    is dispatched as its cached stage jits (W-phase build / G build / flat-k
    IFFTs / G·W multiply + forward FFT / ψ-projection + reduce-scatter), each
    wrapped in a blocking ``timing.section`` sub-row, so ONE run splits the
    per-τ wall into those stages.  When OFF (default) the production fused
    ``_tau_kernel`` jit is returned unchanged — the flag is read once at
    kernel-factory time and is part of the kernel cache key, so the disabled
    path pays zero per-τ overhead.

    Read at USE time, truthy-parsed like common.timing's trace flags.  This is
    an observability knob, not policy: the staged variant evaluates the exact
    same jnp op sequence (same primitives, same order, no algebraic rewrites),
    only in separate XLA modules with per-stage blocking — numerics identical;
    walltime is NOT comparable to the fused path (cross-stage fusion and the
    async-D2H overlap of ppm_accumulators are deliberately serialized).
    Scale-neutral: overhead is O(1) host work per τ stage, independent of
    n_atoms / N_μ / nk / P / backend.
    """
    return os.environ.get("LORRAX_SIGMA_TAU_TIMING", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _fft_ffi_fused_enabled() -> bool:
    """``LORRAX_FFT_FFI_FUSED=1`` routes the τ kernel's IFFT·(G·W)·FFT step
    through ONE fused MKL FFT (DFTI API) host-FFI entry point
    (``common.fft_helpers.make_flat_k_gw_conv``) so the R-space G tile never
    materializes.  O(N log N) FFTs via MKL's DFTI descriptor API reading the
    dot-layout tile directly (NOT a DFT-as-matmul) — see the backend block
    in fft_helpers.  Independent of ``LORRAX_FFT_FFI`` (which switches the
    decomposed helpers); default OFF, production path untouched.  Read at
    kernel-factory time and part of the kernel cache keys.  Announce/refuse
    semantics live in the FFT service factory (raises if the platform's .so
    lacks the handler).

    THE FLAG IS NOT READ HERE (2026-07-30).  It used to be — a consumer
    parsing ``in ("1","true","yes","on")`` with no grammar check and no
    announcement, so ``=yes`` worked, ``=Y`` silently did nothing, and
    neither said anything.  That violated the FFT service's own stated rule
    ("these helpers stay THE single FFT entry point — the backend switch
    happens here and nowhere else", ``fft_helpers.py:306-307``).  The gate
    now lives with the handler it gates (``ffi.mklfft.FUSED_GATE``, on the
    shared ``ffi.gate.Gate``), with the same strict grammar as the
    other two dials; every spelling that worked before still works."""
    from ffi.mklfft import fused_fft_ffi_enabled
    return fused_fft_ffi_enabled()


def _make_project_ri_reduce_scatter(
    mesh_xy: Mesh, *, merged_x: bool = False,
) -> Callable[..., jax.Array]:
    """Build the shard_map'd ψ* σ ψ projector that reduce-scatters the output.

    Drop-in replacement for ``wavefunction_bundle.project_ri`` at the tail of
    ``_sigma_kij_kernel``.  Preserves the math exactly:

        Σ_mn(k) = Σ_{s, μ} Σ_{s', μ'}  ψ*_m(k, s, μ) · σ(k, s, μ, s', μ')
                                        · ψ_n(k, s', μ')

    Since 2026-07-28 (owner directive) this factory is a thin Σ-specific
    wrapper over the SHARED primitive
    ``common.contract_bands.contract_bands_block_reshard`` — THE single
    source of truth for the two-stage psum_scatter band projection+reshard
    and its movement levers (axis-order swap: large partial over the
    node-local 'y' groups; AK.9 channel stacking: one collective per mesh
    axis; L-GEMM f64-split de-promotion of real-operand right GEMMs; the
    impl=mpi world-collective-first warm-up; the divisibility refusal; and
    the gated LORRAX_BANDS_GEMM_FFI MKL batched-GEMM body, whose flag is
    part of this module's kernel cache keys).  Lever evidence and the
    scaling doctrine (never materialize an (m, n, k)- or (μ, μ)-sized
    object on one rank) live in that module's docstrings;
    wk_REL/RESHARD_OVERHEAD_MEMO.md and lgemm_notes.md carry the
    measurements.  The collectives, payload dtypes/shapes and replica
    groups are byte-identical to the pre-primitive bodies (colltable-gated,
    jobs 7878942/7878977 tables).

    Input sharding (global → per-rank):
        ψ_xr  P(None, None, None, 'x')       (nk, m, s, μ_X)
        σ     P(None, None, 'x', None, 'y')  (nk, s, μ_X, s', μ_Y)
        ψ_yn  P(None, None, 'y', None)       (nk, s', μ_Y, n)
    Output sharding:
        merged_x=False: (S_R, S_I) tuple, each (nk, m_X, n_Y) at
        P(None, 'x', 'y') — a tuple rather than a (2, nk, m, n) stack so
        the caller never pays a gather+broadcast pjit pair unpacking a
        sharded array (blocks on is_fully_addressable multi-process).
        merged_x=True: ONE complex X = ψ†σψ, (nk, m_X, n_Y) sharded, each
        mesh axis carrying ONE psum_scatter at HALF the stacked payload.

    CHANNEL PLANS — the Σ-OWNED algebra this wrapper adds to the primitive
    (owner ruling 2026-07-28; merge made the DEFAULT Laplace path by owner
    order the same day; derivation:
    wk_REL/DERIVATION_channel_hermiticity.md, manual §7.5):

    * ``merged_x=False`` → primitive ``channels="split_reim"``: σ^τ is
      split elementwise into σ_R = Re σ^τ, σ_I = Im σ^τ BEFORE projection
      and each real channel rides its own de-promoted GEMM chain,
      S_R = ψ†σ_Rψ, S_I = ψ†σ_Iψ.  REQUIRED for crossing (HGL core)
      windows: their host consumer (``_project_tau_onto_omega_np``,
      project_code=1) weights S_R and S_I with two INDEPENDENT real
      ω-vectors (Re(c)·S_I + Im(c)·S_R), and no per-slice symmetry can
      reconstruct the pair — the crossing phases enter σ^τ symmetrically
      under (μ,ν)→(ν,μ), so σ^τ is neither Hermitian nor
      complex-symmetric per slice; the only surviving relation
      σ^τ† = σ^{−τ} pairs different abscissae on a one-sided grid and
      buys nothing.  Nor is the pair recoverable from X = ψ†σψ: X
      carries 2n² real dof against the 4n² needed, and the
      Hermitian/anti-Hermitian split of X fails structurally (for
      Laplace windows both S_R and i·S_I are Hermitian → it returns
      (X, 0)).
    * ``merged_x=True`` → primitive ``channels="none"``: the single
      complex chain X = ψ†σψ = S_R + i·S_I — the path EVERY Laplace
      (project="full") window dispatches, licensed by BILINEARITY alone
      (their consumer forms only c·(S_R+i·S_I) = c·X); no symmetry
      assumption enters and crossing windows must NOT use it.  Half the
      GEMM work, half the collective payload, half the D2H bytes.
      NOT f64-split — genuinely complex × complex at minimal flops; the
      split was tried and REFUTED (job 7878942: Eigen dgemm ~172 GF/s is
      per-flop below its zgemm's 295; patch preserved at
      wk_REL/lgemm_full_2026-07-28.patch).  The BLAS-rate lever for BOTH
      plans is the primitive's LORRAX_BANDS_GEMM_FFI dial.
      Both plans coexist in every Σ run; no env flag selects between
      them.

    Same byte volume as the original pair of psums, but the output is
    sharded (m_X, n_Y) so every downstream coeff·σ multiply stays local —
    which is the whole point.  A downstream Σ_c(ω, k, m, n) accumulator
    that keeps this layout end-to-end holds a per-rank buffer of
    (n_b/p_x)·(n_b/p_y)·n_ω·n_k — ~100× smaller than a replicated Σ_μν,
    which is the scaling argument for shipping this layout end-to-end.

    Deferred follow-up if that on-GPU sharded accumulator is ever wired:
        (a) m-chunking at add-τ so σ^τ arrives one m-strip at a time rather
            than a full (m_full, n_Y/p) shard (default chunk = 1 tile = m/p);
            needed when (m, n, k, ω) per-rank stops fitting.
        (b) τ batching via lax.scan over a stacked τ axis — previously tried
            and reverted (regressed sigma_ppm ~80% at MoS2 3×3: multiple n_τ
            compiles, no amortization, lost async-dispatch overlap).  Re-add
            only when per-τ Python dispatch cost exceeds those costs.  (The
            primitive's ``extra`` stack axis is the natural carrier if it
            returns.)
        (c) collective-flush SlabIO variant (stage many τ on GPU, one
            parallel-HDF5 write at window close).

    Requires m % p_x == 0 and n % p_y == 0 — enforced by the primitive's
    actionable refusal (pad m and n INDEPENDENTLY via
    ``ppm_sigma.pad_sigma_window``; the exactly-zero pad block is dropped
    by ``ppm_sigma.strip_sigma_window`` before Σ leaves the branch).
    """
    from common.contract_bands import contract_bands_block_reshard

    return contract_bands_block_reshard(
        mesh_xy,
        channels=("none" if merged_x else "split_reim"),
        divisibility_hint=(
            "Both existing sigma call sites pad via pad_sigma_window; "
            "meta.py rounds b_id_4 to world_size but NOT the sigma band "
            "window (b3-b0), so a new unpadded caller is a real hazard, "
            "not a tautology."),
    )


def _get_sigma_kij_kernel(
    *,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    merged_x: bool = False,
) -> Callable[..., jax.Array]:
    """Return a jit-compatible sigma-kij kernel with device-local FFTs.

    The tail project (ψ* σ ψ → Σ_mn) uses the reduce-scatter variant
    (_make_project_ri_reduce_scatter) so the emitted σ^τ is sharded
    (m_X, n_Y) without any downstream reshuffle.  ``merged_x`` selects the
    projection plan (owner ruling 2026-07-28, default semantics since the
    same-day owner order): False = two-channel (S_R, S_I) output — the
    crossing-window path; True = the Laplace plan emitting the single
    complex X = ψ†σψ (channel-plan doc: _make_project_ri_reduce_scatter).
    Both are built in every Σ run.
    """

    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    from common.fft_helpers import (
        make_flat_k_fftn, make_flat_k_gw_conv, make_flat_k_ifftn)
    # The stage-timing flag is part of the cache key: the two variants are
    # different callables (fused jit vs staged dispatcher) and must not
    # shadow each other across a flag flip inside one process (tests).
    # Likewise every factory-time FFI dial (FFT, fused FFT, contract_bands
    # GEMM — one owner: ffi.ffi_dial_key()): a flip mid-process must rebuild
    # the kernel.  ``merged_x`` keys the projection plan: BOTH kernels live
    # in the cache simultaneously in every Σ run (Laplace windows dispatch
    # the merged one, crossing windows the two-channel one).
    from ffi import ffi_dial_key
    pipeline_key = (id(mesh_xy), kgrid, _stage_timing_enabled(),
                    ffi_dial_key(), bool(merged_x))
    if pipeline_key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[pipeline_key]

    from .wavefunction_bundle import G_FFT7D_SPEC as _G_spec, V_FFT5D_SPEC as _V_spec

    ensure_jax_compile_cache()
    inv_sqrt_nk = -1.0 / np.sqrt(float(nk_tot))
    use_fused_ffi = _fft_ffi_fused_enabled()
    if use_fused_ffi:
        # ONE fused MKL FFT (DFTI API) host-FFI call per rank per τ:
        # sigma_k = fftn(ifftn(G_k)·ifftn(W_q)[:,None,:,None,:]·inv_sqrt_nk)
        # with the R-space G tile chunked away inside the handler.  The
        # decomposed helpers below are deliberately NOT built on this route
        # (their announce/probe belongs to LORRAX_FFT_FFI).
        _gw_conv = make_flat_k_gw_conv(
            mesh_xy, kgrid, _G_spec, _V_spec,
            norm='ortho', mult=inv_sqrt_nk)
    else:
        _G_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, _G_spec, norm='ortho')
        _G_fftn  = make_flat_k_fftn( mesh_xy, kgrid, _G_spec, norm='ortho')
        _V_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, _V_spec, norm='ortho')

    from .greens_function_kernel import build_G_tau

    _project_ri_rs = _make_project_ri_reduce_scatter(mesh_xy, merged_x=merged_x)

    @partial(jax.jit, donate_argnums=(8,))
    def _sigma_kij_kernel(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_ref_A, t_node, W_q,
    ):
        """Σ_kij = project_rs[ FFT[ G(R) · W(R) / √Nk ] ].  All flat-k.

        Output: the (S_R, S_I) tuple (two-channel plan) or the single
        complex X = ψ†σψ (merged Laplace plan, ``merged_x=True``).

        W_q is (nq, μ, μ) flat-q — same layout as all other flat-k arrays.
        ``W_q`` is **donated**: it's built fresh each τ by ``_build_W_t_q``
        and only consumed here, so XLA can reuse its buffer for the
        ``V_R = _V_ifftn(W_q)`` output instead of allocating a separate
        intermediate.  DONATION-AUDIT NOTE (2026-07-28): in PRODUCTION this
        jit is traced INTO the outer ``_tau_kernel`` jit, where inner-jit
        donation is inert (donation acts only at top-level dispatch — same
        house fact as the ζ r-chunk jits, SPEEDUP_SCORECARD.md audit row
        (d)); there W_t_q is a module-internal temp and buffer reuse is
        XLA's liveness analysis + the FFI in-place aliases.  The annotation
        is kept for any future top-level dispatch of this kernel.

        G(t) = build_G_tau(psi, E_A, 1j·t_node, e_ref=E_ref_A, mask=mask_A),
        i.e. the unified ISDF-basis G builder with pure-imaginary t
        (real-time evolution).  Output (Σ_ri) emerges (m_X, n_Y)-sharded
        from the final shard_map.
        """
        G_k = build_G_tau(
            psi_coh_xn, psi_coh_yr, E_A, 1j * t_node,
            e_ref=E_ref_A, mask=mask_A,
        )
        if use_fused_ffi:
            sigma_k = _gw_conv(G_k, W_q)
        else:
            G_R = _G_ifftn(G_k)
            V_R = _V_ifftn(W_q)[:, None, :, None, :]  # (nk,1,μ,1,μ) broadcast to G shape
            sigma_k = _G_fftn(G_R * V_R * inv_sqrt_nk)
        return _project_ri_rs(psi_proj_xr, sigma_k, psi_proj_yn)

    if not _stage_timing_enabled():
        _sigma_kij_kernel_cache[pipeline_key] = _sigma_kij_kernel
        return _sigma_kij_kernel

    # ------------------------------------------------------------------
    # Stage-split instrumented variant (LORRAX_SIGMA_TAU_TIMING=1 only).
    #
    # Same op sequence as ``_sigma_kij_kernel`` above, dispatched as five
    # cached stage jits so blocking ``timing.section`` sub-rows attribute
    # the per-τ wall (evidence: AQ 4962c/P=64 HLO module_0912, 2026-07-28
    # — the 1.51 s/τ was indivisible).  Notes:
    #   * ``sigma.tau.project_rs`` covers the ψ-projection dots AND their
    #     psum_scatters together — they live inside one shard_map (the
    #     two-channel body, or the merged single-chain body when
    #     ``merged_x``; the crossing family keeps two channels per the
    #     owner ruling 2026-07-28); a per-op split of dot vs collective
    #     wait comes from a profiler trace (jax_profile.trace_section
    #     wraps the first window per branch in ppm_sigma), not from
    #     restructuring the kernel.
    #   * DONATION AUDIT (2026-07-28, owner directive with the FFT-FFI
    #     prototype): each stage jit now donates every operand that is DEAD
    #     after its call — G_k → G_ifft, W_q → V_ifft, (G_R, V_R) →
    #     mult_fft, sigma_k → project.  Verified against this dispatcher:
    #     none of those is referenced again after its consuming stage (the
    #     ``sec.watch`` block-until-ready runs in the PRODUCING stage's
    #     section, before the donation), and the ψ/E/mask operands are
    #     loop-invariant across τ so they are never donated.  This closes
    #     the staged path's previously-documented "W_q NOT donated here"
    #     gap and drops the big dead tiles (399 MB G_k / G_R, 100 MB W_q /
    #     V_R at nb=128/P=64) from the staged peak.  Donation is data
    #     movement only — stage rows measure the same ops.
    #   * Stage boundaries force materialization of G_k / G_R / V_R that
    #     the fused module may keep in fft-native layout, so staged wall
    #     ≠ fused wall; the rows are for RATIO attribution.
    #   * LORRAX_FFT_FFI_FUSED=1: the three FFT-adjacent rows collapse into
    #     ONE row ``sigma.tau.GW_conv_ffi`` (the fused MKL FFT (DFTI API)
    #     handler does IFFT·multiply·FFT in one call) — A/B against the
    #     sum G_ifft + V_ifft + GW_mult_fft of the reference table.
    # Scale-neutral: per-τ host overhead is O(#stages), independent of
    # n_atoms / N_μ / nk / P and identical on CPU and GPU backends.
    # ------------------------------------------------------------------
    _G_build_j = jax.jit(
        lambda xn, yr, E_A, mask_A, E_ref_A, t_node: build_G_tau(
            xn, yr, E_A, 1j * t_node, e_ref=E_ref_A, mask=mask_A))
    if use_fused_ffi:
        _conv_j = jax.jit(_gw_conv, donate_argnums=(0, 1))
    else:
        _G_ifft_j = jax.jit(_G_ifftn, donate_argnums=(0,))
        _V_ifft_j = jax.jit(lambda W_q: _V_ifftn(W_q)[:, None, :, None, :],
                            donate_argnums=(0,))
        _mult_fft_j = jax.jit(lambda G_R, V_R: _G_fftn(G_R * V_R * inv_sqrt_nk),
                              donate_argnums=(0, 1))
    _project_j = jax.jit(_project_ri_rs, donate_argnums=(1,))

    def _sigma_kij_kernel_staged(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_ref_A, t_node, W_q,
    ):
        with timing.section("sigma.tau.G_build") as sec:
            G_k = _G_build_j(psi_coh_xn, psi_coh_yr, E_A, mask_A,
                             E_ref_A, t_node)
            sec.watch(G_k)
        if use_fused_ffi:
            with timing.section("sigma.tau.GW_conv_ffi") as sec:
                sigma_k = _conv_j(G_k, W_q)
                sec.watch(sigma_k)
        else:
            with timing.section("sigma.tau.G_ifft") as sec:
                G_R = _G_ifft_j(G_k)
                sec.watch(G_R)
            with timing.section("sigma.tau.V_ifft") as sec:
                V_R = _V_ifft_j(W_q)
                sec.watch(V_R)
            with timing.section("sigma.tau.GW_mult_fft") as sec:
                sigma_k = _mult_fft_j(G_R, V_R)
                sec.watch(sigma_k)
        with timing.section("sigma.tau.project_rs") as sec:
            out = _project_j(psi_proj_xr, sigma_k, psi_proj_yn)
            sec.watch(out)
        return out

    _sigma_kij_kernel_cache[pipeline_key] = _sigma_kij_kernel_staged
    return _sigma_kij_kernel_staged


def _get_sigma_tau_kernel(
    *,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    merged_x: bool = False,
) -> Callable[..., jax.Array]:
    """Return a cached tau-node sigma builder with jittable local FFTs.

    ``merged_x=False`` (default): the two-channel kernel returning
    (S_R, S_I) — the only kernel crossing windows may use.
    ``merged_x=True``: the merged Laplace-plan kernel returning the single
    complex X = ψ†σψ (channel-plan doc:
    ``_make_project_ri_reduce_scatter``); dispatched by
    ``ppm_sigma`` for EVERY project="full" window (the default and only
    Laplace path — owner order 2026-07-28).
    """

    kgrid = tuple(int(x) for x in kgrid)
    # ffi.ffi_dial_key(): the ONE owner of the factory-time FFI dial tuple —
    # must match _get_sigma_kij_kernel's pipeline_key components exactly.
    from ffi import ffi_dial_key
    cache_key = (id(mesh_xy), kgrid, _stage_timing_enabled(),
                 ffi_dial_key(), bool(merged_x))
    if cache_key in _sigma_tau_kernel_cache:
        return _sigma_tau_kernel_cache[cache_key]

    ensure_jax_compile_cache()
    q_mu_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    sigma_kij_kernel = _get_sigma_kij_kernel(mesh_xy=mesh_xy, kgrid=kgrid,
                                             merged_x=merged_x)

    @jax.jit
    def _build_W_t_q(B_q, Omega_q, mask_B, E_ref_B, t_node):
        """W(τ) = Σ_q B_q · exp(-i·(Ω_q - E_ref_B)·τ) · mask_B.

        (A-side G now built inside sigma_kij_kernel via build_G_tau, so
        the tau-operand helper only shapes the PPM-pole-sum B-side.)
        """
        phase_B = jnp.exp(-1j * (Omega_q - E_ref_B) * t_node)
        W_t_q = jnp.where(mask_B, B_q * phase_B,
                          jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        return jax.lax.with_sharding_constraint(W_t_q, q_mu_shard)

    @jax.jit
    def _tau_kernel(
        psi_coh_xn, psi_coh_yr,
        psi_proj_xr, psi_proj_yn,
        E_A, mask_A, B_q, Omega_q, mask_B,
        E_ref_A, E_ref_B, t_node,
    ):
        W_t_q = _build_W_t_q(B_q, Omega_q, mask_B, E_ref_B, t_node)
        return sigma_kij_kernel(
            psi_coh_xn, psi_coh_yr,
            psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_t_q,
        )

    if _stage_timing_enabled():
        # Stage-split diagnostic dispatcher (see _stage_timing_enabled):
        # _build_W_t_q is already its own jit, so calling it from Python
        # gives the 'sigma.tau.w_phase' row for free; sigma_kij_kernel is
        # the staged variant from _get_sigma_kij_kernel (same cache-key
        # flag) and emits the remaining stage rows.  Numerics: identical
        # op sequence to the fused _tau_kernel above.
        def _tau_kernel_staged(
            psi_coh_xn, psi_coh_yr,
            psi_proj_xr, psi_proj_yn,
            E_A, mask_A, B_q, Omega_q, mask_B,
            E_ref_A, E_ref_B, t_node,
        ):
            with timing.section("sigma.tau.w_phase") as sec:
                W_t_q = _build_W_t_q(B_q, Omega_q, mask_B, E_ref_B, t_node)
                sec.watch(W_t_q)
            return sigma_kij_kernel(
                psi_coh_xn, psi_coh_yr,
                psi_proj_xr, psi_proj_yn,
                E_A, mask_A, E_ref_A, t_node, W_t_q,
            )

        _sigma_tau_kernel_cache[cache_key] = _tau_kernel_staged
        return _tau_kernel_staged

    _sigma_tau_kernel_cache[cache_key] = _tau_kernel
    return _tau_kernel


def precompile_sigma(wfns, ppm, meta, mesh_xy: Mesh) -> None:
    """AOT lower + compile the per-τ sigma kernel.

    Parallel to :func:`w_isdf.precompile_chi0` / ``precompile_solve_w``:
    lower the cached ``_tau_kernel`` at the real input shapes/shardings
    and eagerly ``.compile()`` it so the first per-τ dispatch inside
    ``compute_sigma_c_ppm_omega_grid`` is execution-only.  Call inside
    a dedicated ``timing.section('sigma.compile')`` block to split
    compile from exec in the end-of-run timing report.

    The kernel is shape-invariant across the four ω-sign × cond/val
    branches (ψ / E_A / mask_A / B_q / Ω_q / mask_B / scalars all have
    fixed shape+dtype+sharding; only values change per window) — so
    one AOT compile covers every branch.  Every Σ run carries two kernels
    — the merged Laplace X kernel and the two-channel crossing kernel —
    and both are compiled here.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    # BOTH kernels dispatch at runtime (Laplace windows the merged X kernel,
    # crossing windows the two-channel one) — AOT both so neither pays a
    # first-dispatch compile inside sigma.exec.
    tau_kernels = [
        _get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid),
        _get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True),
    ]

    s = wfns.slices
    psi_coh_xn  = wfns.xn(s.full)
    psi_coh_yr  = wfns.yr(s.full)
    psi_proj_xr = wfns.xr(s.sigma)
    psi_proj_yn = wfns.yn(s.sigma)
    # Mesh-pad the QP band window EXACTLY as ``ppm_sigma._run_sigma_branch``
    # does at runtime.  This is load-bearing twice over: the reduce-scatter
    # projector asserts m % p_x == 0 / n % p_y == 0 (so an unpadded AOT
    # lowering fires the guard here, which is where 7874338 died), and the AOT
    # signature must match the runtime one shape-for-shape or pjit silently
    # re-traces and the precompile buys nothing.
    from .ppm_sigma import pad_sigma_window
    psi_proj_xr, psi_proj_yn, _nb_real = pad_sigma_window(
        psi_proj_xr, psi_proj_yn, mesh_xy)

    # Representative non-ψ inputs — values don't matter for AOT, only
    # the full `(shape, dtype, sharding, committed-ness)` tuple must
    # match the runtime signature or pjit re-traces.  Specifically:
    #   * E_A at runtime comes from ``_prepare_sigma_state`` (jit output)
    #     — committed to the mesh as ``NamedSharding(P(None, None))``.
    #     Must device_put the dummy to match, otherwise pjit sees
    #     ``UnspecifiedValue`` vs ``P(None, None)`` and re-compiles.
    #   * mask_A, scalars: at runtime go through ``jnp.asarray(numpy_val)``
    #     which stays uncommitted — leave as plain jnp to match.
    #   * mask_B inherits Ω_q's sharding, same as ``_materialize_window_mask_B``.
    #
    # Placement uses ``device_put_process_local``, NOT a bare
    # ``jax.device_put`` (AA.1 / AO.1): ``jnp.zeros(...)`` is an
    # UNCOMMITTED ``jax.Array``, and ``_device_put_sharding_impl`` takes
    # the ``multihost_utils.assert_equal`` (= ``process_allgather``)
    # branch for an uncommitted operand onto a sharding that is not fully
    # addressable.  ``rep_2d`` is multi-process, so the bare call was a
    # real P-linear collective inside the AOT precompile.  The helper
    # produces the SAME committed ``NamedSharding(P(None, None))`` array,
    # which is the only property this dummy has to have.  The replication
    # precondition holds trivially (all-zeros).
    nb_full = int(psi_coh_xn.shape[-1])
    rep_2d  = NamedSharding(mesh_xy, P(None, None))
    from common.collectives import device_put_process_local
    E_A     = device_put_process_local(
        np.zeros((int(meta.nk_tot), nb_full), dtype=np.float64), rep_2d)
    mask_A  = jnp.ones((int(meta.nk_tot), nb_full), dtype=bool)
    mask_B  = jnp.ones_like(ppm.Omega_q, dtype=bool)
    E_ref_A = jnp.asarray(0.0, dtype=jnp.float64)
    E_ref_B = jnp.asarray(0.0, dtype=jnp.float64)
    t_node  = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

    for tau_kernel in tau_kernels:
        if hasattr(tau_kernel, "lower"):
            tau_kernel.lower(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, ppm.B_q, ppm.Omega_q, mask_B,
                E_ref_A, E_ref_B, t_node,
            ).compile()
        else:
            # Stage-split diagnostic dispatcher (LORRAX_SIGMA_TAU_TIMING=1): a
            # plain Python callable over five stage jits, so there is no single
            # ``.lower()``.  Prewarm by EXECUTING it once at the real
            # shapes/shardings — signature match with the runtime path is then
            # guaranteed by construction (the precompile-signature drift this
            # AOT helper exists to prevent), at the cost of ~one τ-node of
            # execution inside sigma.compile.  All ranks reach this call
            # synchronously (module contract), so the psum_scatters inside are
            # collective-safe.  Output is discarded.
            out = tau_kernel(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, ppm.B_q, ppm.Omega_q, mask_B,
                E_ref_A, E_ref_B, t_node,
            )
            jax.block_until_ready(out)

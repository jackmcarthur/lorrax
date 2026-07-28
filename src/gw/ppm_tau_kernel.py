"""Device τ-kernel unit for the Σ_c(ω) GN-PPM integration.

The single-tau integrand kernel plus its cache/AOT machinery:

    σ^τ_nmk(τ) = project[ FFT[ G(τ) · W(τ) / √N_k ] ]
    G(τ)       = diag[ e^{-i(E_A - E_ref_A)·τ} ] · mask_A           (A = val or cond)
    W(τ)       = Σ_μν  B_q · e^{-i(Ω_q - E_ref_B)·τ}  · mask_B      (PPM pole sum)

This is the only Σ_PPM file where SPMD / sharding / HLO expertise is required —
the reduce-scatter layout doc and the deferred scan / collective-flush notes all
live here.

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
    semantics live in the fft_helpers factory (raises if the host .so lacks
    the handler)."""
    return os.environ.get("LORRAX_FFT_FFI_FUSED", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _project_ri_local(psi_xr_local, sigma_k_local, psi_yn_local):
    """Rank-local body of the ψ* σ ψ reduce-scatter projection.

    ONE source for the projection tail, called from inside the standalone
    projector's shard_map (``_make_project_ri_reduce_scatter``).  Kept at
    module level so any future kernel variant shares the byte-identical
    body (the 2026-07-28 monolithic-fusion experiment did exactly that; see
    wk_REL/sigma_perf_results.md — refuted at nb=128, patch preserved).
    Local shapes:

        psi_xr_local  (nk, m, s, μ_X_loc)
        sigma_k_local (nk, s, μ_X_loc, s', μ_Y_loc)
        psi_yn_local  (nk, s', μ_Y_loc, n)
        → (re, im) each (nk, m/p_x, n/p_y)

    Re/im CHANNELS stay separate end-to-end (OWNER_HOLD 2026-07-28: the
    channel decomposition is mathematically substantive — each channel's
    einsum/cast sequence below must stay per-channel).  Two approved
    movement-only levers are applied on top:

    * AK.9 stacking (scorecard, 2026-07-28): both channels' psum_scatter
      payloads ride ONE collective per mesh axis, stacked on a fresh
      leading axis — 4 → 2 messages/τ at identical bytes.  Bit-exact by
      construction: reduce-scatter sums ELEMENTWISE over the same replica
      groups in the same rank order ("a rank-wise elementwise sum cannot
      care that two independent arrays were concatenated" — AK.9);
      stack/index are pure data movement.  AK.9's precondition (transport
      question settled) is met on this run family — the certified AS.7
      mpi/mlx cell.
    * Axis-order swap (owner-approved 2026-07-28): μ_Y contracted FIRST so
      the LARGE stacked partial reduce-scatters over 'y' (consecutive-rank
      groups: node-local pairs at 2 ranks/node) and only the small final
      block rides the stride-p_x all-inter-node 'x' groups (evidence: HLO
      module_0912 replica_groups {0,8,...,56} for 'x' vs {0..7} for 'y';
      payloads 2×40.9 MB vs 2×0.52 MB pre-swap).  Per-channel contraction
      order becomes ψ*·(σ·ψ) — same math/flops, value-level identical,
      gated by the 1e-12 parity suite (not claimed bit-exact).

    Scale-neutral: both levers are flat in n_atoms / N_μ / nk / nb / P and
    backend; no new N_μ²-sized object is created.
    """
    def _right(sigma_real_or_imag):
        # 'ksxty' × 'ktyn' -> 'ksxn'  (contracts s', local μ_Y)
        return jnp.einsum(
            'ksxty,ktyn->ksxn',
            sigma_real_or_imag, psi_yn_local, optimize=True)

    right_re = _right(jnp.real(sigma_k_local))
    right_im = _right(jnp.imag(sigma_k_local))
    # ONE psum_scatter(y) for both channels; n is axis 4 of the stack:
    # (2, nk, s, μ_X_loc, n) → (2, nk, s, μ_X_loc, n/p_y)
    right_rs = jax.lax.psum_scatter(
        jnp.stack([right_re, right_im], axis=0), 'y',
        scatter_dimension=4, tiled=True)

    def _left(right_rs_ch):
        # 'kmsx' × 'ksxn' -> 'kmn'  (contracts s, local μ_X)
        return jnp.einsum(
            'kmsx,ksxn->kmn',
            jnp.conj(psi_xr_local), right_rs_ch, optimize=True)

    result_re = _left(right_rs[0])
    result_im = _left(right_rs[1])
    # ONE psum_scatter(x) for both channels; m is axis 2 of the stack:
    # (2, nk, m, n/p_y) → (2, nk, m/p_x, n/p_y)
    out = jax.lax.psum_scatter(
        jnp.stack([result_re, result_im], axis=0), 'x',
        scatter_dimension=2, tiled=True)
    return (out[0].astype(jnp.complex128),
            out[1].astype(jnp.complex128))


def _make_project_ri_reduce_scatter(mesh_xy: Mesh) -> Callable[..., jax.Array]:
    """Build a shard_map'd ψ* σ ψ that reduce-scatters the output.

    Drop-in replacement for ``wavefunction_bundle.project_ri`` at the tail of
    ``_sigma_kij_kernel``.  Preserves the math exactly:

        Σ_mn(k) = Σ_{s, μ} Σ_{s', μ'}  ψ*_m(k, s, μ) · σ(k, s, μ, s', μ')
                                        · ψ_n(k, s', μ')

    Input sharding (global → per-rank):
        ψ_xr  P(None, None, None, 'x')       (nk, m, s, μ_X)
        σ     P(None, None, 'x', None, 'y')  (nk, s, μ_X, s', μ_Y)
        ψ_yn  P(None, None, 'y', None)       (nk, s', μ_Y, n)

    Output sharding:
        (sigma_re, sigma_im) each at  P(None, 'x', 'y')   (nk, m_X, n_Y)

    Returns the re/im parts as a tuple rather than a single (2, nk, m, n)
    stack — avoids the tuple-unpack at the caller (which would trigger a
    gather+broadcast pjit pair for a sharded array and blocks on
    is_fully_addressable in multi-process mode).

    Comms inside:  the two implicit psums of the original einsum
        psum(y)       over the μ_Y contraction axis   (FIRST — large payload)
        psum(x)       over the μ_X contraction axis   (second — small payload)
    become:
        psum_scatter(y, scatter_dim=n)   — reduces μ_Y AND scatters n on y
        psum_scatter(x, scatter_dim=m)   — reduces μ_X AND scatters m on x
    with two packaging/locality levers on top of the original design:
      * AK.9 stacking (scorecard, 2026-07-28): BOTH re/im channels ride each
        collective in one stacked payload — 2 collectives per τ instead of 4
        at identical bytes, bit-exact (elementwise rank-sum is indifferent
        to concatenation).
      * axis-order swap (owner-approved 2026-07-28, movement-only): the
        μ_Y-side contraction runs FIRST so the LARGE partial (the m-full
        (nk, s, μ_X_loc, n) block) reduce-scatters over the 'y' axis, whose
        consecutive-rank replica groups have node-local pairs at 2
        ranks/node, while the stride-p_x 'x' groups (zero SHM locality on
        the AQ layout — HLO module_0912 groups {0,8,16,...}) now carry only
        the small (nk, m, n/p_y) block.  Per-channel contraction order
        becomes ψ*·(σ·ψ) instead of (ψ*·σ)·ψ — same math and flops,
        value-level (not bit-level) identical; gated by the 1e-12 output
        parity suite.  Scale-neutral: locality preference and message
        count are flat in n_atoms / N_μ / nk / nb / P and backend.
    The channel MATH stays per-channel — see the OWNER_HOLD note in
    ``_project_ri_local``.

    Same NCCL byte volume as the original pair of psums (on-ring LL128), but
    the output is sharded (m_X, n_Y) so every downstream coeff·σ multiply
    stays local — which is the whole point.  A downstream Σ_c(ω, k, m, n)
    accumulator that keeps this layout end-to-end holds a per-rank buffer of
    (n_b/p_x)·(n_b/p_y)·n_ω·n_k — ~100× smaller than a replicated Σ_μν, which
    is the scaling argument for shipping this layout end-to-end.

    Deferred follow-up if that on-GPU sharded accumulator is ever wired:
        (a) m-chunking at add-τ so σ^τ arrives one m-strip at a time rather
            than a full (m_full, n_Y/p) shard (default chunk = 1 tile = m/p);
            needed when (m, n, k, ω) per-rank stops fitting.
        (b) τ batching via lax.scan over a stacked τ axis — previously tried
            and reverted (regressed sigma_ppm ~80% at MoS2 3×3: multiple n_τ
            compiles, no amortization, lost async-dispatch overlap).  Re-add
            only when per-τ Python dispatch cost exceeds those costs.
        (c) collective-flush SlabIO variant (stage many τ on GPU, one
            parallel-HDF5 write at window close).

    Requires m % p_x == 0 and n % p_y == 0.  Callers satisfy this via
    ``ppm_sigma.pad_sigma_window`` (independent per-axis zero-pads:
    m → multiple of p_x, n → multiple of p_y; the exactly-zero pad block
    is dropped by ``ppm_sigma.strip_sigma_window`` before Σ leaves the
    branch).
    """
    from jax.experimental.shard_map import shard_map

    in_specs = (
        P(None, None, None, 'x'),          # psi_xr  : (nk, m, s, μ_X)
        P(None, None, 'x', None, 'y'),     # sigma_k : (nk, s, μ_X, s', μ_Y)
        P(None, None, 'y', None),          # psi_yn  : (nk, s', μ_Y, n)
    )
    # 2-tuple output: re part, im part.  Each (nk, m_X, n_Y) sharded.
    out_specs = (P(None, 'x', 'y'), P(None, 'x', 'y'))

    _sm = shard_map(_project_ri_local, mesh=mesh_xy,
                    in_specs=in_specs, out_specs=out_specs,
                    check_rep=False)

    # Guard the divisibility this kernel requires (see docstring): the two
    # psum_scatters split m over p_x and n over p_y, so an indivisible sigma
    # band window would otherwise crash cryptically deep inside psum_scatter
    # (or, with a future non-tiled variant, misalign silently).  Convert that
    # into a clear, actionable failure that names the fix.  No behaviour change
    # for valid (divisible) inputs — identity passthrough.  meta.py rounds
    # b_id_4 to world_size but NOT the sigma band window (b3-b0), so this is a
    # real, reachable precondition, not a tautology.
    p_x, p_y = mesh_xy.shape['x'], mesh_xy.shape['y']

    def _project_ri_reduce_scatter(psi_xr, sigma_k, psi_yn):
        m, n = psi_xr.shape[1], psi_yn.shape[3]
        # Message names the ACTUAL fix (pad_sigma_window's independent
        # per-axis pads), not the old p_x*p_y product rule that
        # pad_sigma_window's docstring denounces as up-to-3.16x waste
        # (audit fix/zq 2026-07-28).
        assert m % p_x == 0 and n % p_y == 0, (
            f"sigma reduce-scatter needs the band window divisible by the "
            f"mesh: m={m} must be a multiple of p_x={p_x} and n={n} of "
            f"p_y={p_y}.  Pad m and n INDEPENDENTLY (m -> p_x, n -> p_y) "
            f"via gw.ppm_sigma.pad_sigma_window at the caller — both "
            f"existing call sites do; do NOT round to a multiple of "
            f"p_x*p_y (meta.py rounds b_id_4 but NOT the sigma window "
            f"b3-b0, so a new unpadded caller is a real hazard).")
        return _sm(psi_xr, sigma_k, psi_yn)

    return _project_ri_reduce_scatter


def _get_sigma_kij_kernel(
    *,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
) -> Callable[..., jax.Array]:
    """Return a jit-compatible sigma-kij kernel with device-local FFTs.

    The tail project (ψ* σ ψ → Σ_mn) uses the reduce-scatter variant
    (_make_project_ri_reduce_scatter) so the emitted σ^τ is sharded
    (m_X, n_Y) without any downstream reshuffle.
    """

    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    from common.fft_helpers import (
        fft_ffi_enabled, make_flat_k_fftn, make_flat_k_gw_conv,
        make_flat_k_ifftn)
    # The stage-timing flag is part of the cache key: the two variants are
    # different callables (fused jit vs staged dispatcher) and must not
    # shadow each other across a flag flip inside one process (tests).
    # Likewise the two FFT-FFI flags (read at factory time by fft_helpers /
    # _fft_ffi_fused_enabled): a flip mid-process must rebuild the kernel.
    pipeline_key = (id(mesh_xy), kgrid, _stage_timing_enabled(),
                    fft_ffi_enabled(), _fft_ffi_fused_enabled())
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

    _project_ri_rs = _make_project_ri_reduce_scatter(mesh_xy)

    @partial(jax.jit, donate_argnums=(8,))
    def _sigma_kij_kernel(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_ref_A, t_node, W_q,
    ):
        """Σ_kij = project_rs[ FFT[ G(R) · W(R) / √Nk ] ].  All flat-k.

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
    #     two psum_scatters together — they live inside one shard_map and
    #     the re/im channel decomposition there is OWNER-HELD; a per-op
    #     split of dot vs collective wait comes from a profiler trace
    #     (jax_profile.trace_section wraps the first window per branch in
    #     ppm_sigma), not from restructuring the kernel.
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
) -> Callable[..., jax.Array]:
    """Return a cached tau-node sigma builder with jittable local FFTs."""

    kgrid = tuple(int(x) for x in kgrid)
    from common.fft_helpers import fft_ffi_enabled
    cache_key = (id(mesh_xy), kgrid, _stage_timing_enabled(),
                 fft_ffi_enabled(), _fft_ffi_fused_enabled())
    if cache_key in _sigma_tau_kernel_cache:
        return _sigma_tau_kernel_cache[cache_key]

    ensure_jax_compile_cache()
    q_mu_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    sigma_kij_kernel = _get_sigma_kij_kernel(mesh_xy=mesh_xy, kgrid=kgrid)

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
    one AOT compile covers every branch.
    """
    ensure_jax_compile_cache()
    kgrid = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    tau_kernel = _get_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=kgrid)

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
    nb_full = int(psi_coh_xn.shape[-1])
    rep_2d  = NamedSharding(mesh_xy, P(None, None))
    E_A     = jax.device_put(
        jnp.zeros((int(meta.nk_tot), nb_full), dtype=jnp.float64), rep_2d)
    mask_A  = jnp.ones((int(meta.nk_tot), nb_full), dtype=bool)
    mask_B  = jnp.ones_like(ppm.Omega_q, dtype=bool)
    E_ref_A = jnp.asarray(0.0, dtype=jnp.float64)
    E_ref_B = jnp.asarray(0.0, dtype=jnp.float64)
    t_node  = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

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

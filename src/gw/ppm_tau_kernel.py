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

The module-level kernel caches are co-located with the factories that read
them.  In particular, :func:`get_sigma_spatial_kernel` is the one reusable
``G_k x W_q -> Sigma`` owner; ansatz adapters cache only their G/W synthesis.
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
_sigma_spatial_kernel_cache: dict[
    tuple[object, ...], Callable[..., jax.Array]
] = {}
_sigma_shared_tau_kernel_cache: dict[
    tuple[object, ...], Callable[..., jax.Array]
] = {}


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
    through ONE fused FFTW3-ABI host-FFI entry point
    (``common.fft_helpers.make_flat_k_gw_conv``) so the R-space G tile never
    materializes.  O(N log N) FFTs via the FFTW3 advanced-layout plans reading the
    dot-layout tile directly (NOT a DFT-as-matmul) — see the backend block
    in fft_helpers.  Independent of ``LORRAX_FFT_FFI``; default ON since
    the FFI-required ruling (decisions.md 2026-08-01) — ``=0`` opts out to
    the decomposed three-transform chain, which is itself FFI-served.  Read
    at kernel-factory time and part of the kernel cache keys.
    Announce/refuse semantics live in the FFT service factory (raises if
    the platform's .so lacks the handler).

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
      S_R = ψ†σ_Rψ, S_I = ψ†σ_Iψ.  What crossing (HGL core) windows
      dispatch — but NO LONGER because they must.  This bullet used to
      argue that the crossing consumer weights S_R and S_I with two
      INDEPENDENT real ω-vectors (Re(c)·S_I + Im(c)·S_R) and that the
      pair is unrecoverable from X (2n² real dof against 4n²).  That
      independent-weight consumer WAS THE DEFECT: it is the elementwise
      Im[c·σ], which completes the crossing window's one-sided τ grid
      only where σ^τ is complex-symmetric, i.e. only at k ≡ −k
      (KNOWN_FAILURES, fixed 2026-08-09).  The correct completion,
      (Z − Z†)/2i with Z = Σ_τ c·X, reads ONLY X — so the dof argument
      no longer applies to this window either, and the split survives
      here as a channel-plan/perf choice.  Collapsing crossing onto the
      merged kernel is a real option and an owner-scoped change (it moves
      the collective payloads and the D2H volume); until it is taken,
      ``_project_tau_onto_omega_np``'s cross-pairing guards keep the
      carrier contract honest.
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


def get_sigma_spatial_kernel(
    *,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    merged_x: bool = False,
) -> Callable[..., jax.Array]:
    """Return the ansatz-neutral ``G_k x W_q -> Sigma_kij`` kernel.

    This is the extension seam for alternate propagators and screened
    interactions.  GN-PPM and MPA construct their own ``G_k`` and ``W_q``;
    this owner performs only the flat-k FFT convolution and band projection.
    A two-point vertex-corrected W can therefore enter without forking the
    spatial kernel.  A genuine three-point vertex remains a different
    contraction and should not be hidden behind this interface.
    """
    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    from common.fft_helpers import (
        make_flat_k_fftn, make_flat_k_gw_conv, make_flat_k_ifftn)
    from ffi import ffi_dial_key
    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key(),
           bool(merged_x))
    if key in _sigma_spatial_kernel_cache:
        return _sigma_spatial_kernel_cache[key]
    from .wavefunction_bundle import (G_FFT7D_SPEC as _G_spec,
                                      V_FFT5D_SPEC as _V_spec)
    ensure_jax_compile_cache()
    inv_sqrt_nk = -1.0 / np.sqrt(float(nk_tot))
    use_fused_ffi = _fft_ffi_fused_enabled()
    if use_fused_ffi:
        # ONE fused FFTW3-ABI (host) / cuFFT (CUDA) FFI call per rank per τ:
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

    project = _make_project_ri_reduce_scatter(mesh_xy, merged_x=merged_x)

    @partial(jax.jit, donate_argnums=(2, 3))
    def spatial(psi_proj_xr, psi_proj_yn, G_k, W_q):
        if use_fused_ffi:
            sigma_k = _gw_conv(G_k, W_q)
        else:
            G_R = _G_ifftn(G_k)
            V_R = _V_ifftn(W_q)[:, None, :, None, :]
            sigma_k = _G_fftn(G_R * V_R * inv_sqrt_nk)
        return project(psi_proj_xr, sigma_k, psi_proj_yn)
    if not _stage_timing_enabled():
        _sigma_spatial_kernel_cache[key] = spatial
        return spatial
    if use_fused_ffi:
        _conv_j = jax.jit(_gw_conv, donate_argnums=(0, 1))
    else:
        _G_ifft_j = jax.jit(_G_ifftn, donate_argnums=(0,))
        _V_ifft_j = jax.jit(lambda W_q: _V_ifftn(W_q)[:, None, :, None, :],
                            donate_argnums=(0,))
        _mult_fft_j = jax.jit(lambda G_R, V_R: _G_fftn(G_R * V_R * inv_sqrt_nk),
                              donate_argnums=(0, 1))
    _project_j = jax.jit(project, donate_argnums=(1,))

    def spatial_staged(
        psi_proj_xr, psi_proj_yn, G_k, W_q,
    ):
        """Diagnostic split of the same spatial operation sequence."""
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

    _sigma_spatial_kernel_cache[key] = spatial_staged
    return spatial_staged


def _get_sigma_kij_kernel(
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int], merged_x: bool = False,
) -> Callable[..., jax.Array]:
    """GN/MPA adapter that builds G and calls the shared spatial kernel."""
    from ffi import ffi_dial_key
    key = (id(mesh_xy), tuple(map(int, kgrid)), _stage_timing_enabled(),
           ffi_dial_key(), bool(merged_x))
    if key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[key]
    from .greens_function_kernel import build_G_tau
    spatial = get_sigma_spatial_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=merged_x)

    @partial(jax.jit, donate_argnums=(8,))
    def kernel(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_ref_A, t_node, W_q,
    ):
        G_k = build_G_tau(
            psi_coh_xn, psi_coh_yr, E_A, 1j * t_node,
            e_ref=E_ref_A, mask=mask_A)
        return spatial(psi_proj_xr, psi_proj_yn, G_k, W_q)
    if not _stage_timing_enabled():
        _sigma_kij_kernel_cache[key] = kernel
        return kernel
    build_g = jax.jit(
        lambda xn, yr, E, mask, ref, t: build_G_tau(
            xn, yr, E, 1j * t, e_ref=ref, mask=mask))

    def staged(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_ref_A, t_node, W_q,
    ):
        with timing.section("sigma.tau.G_build") as sec:
            G_k = build_g(psi_coh_xn, psi_coh_yr, E_A, mask_A,
                          E_ref_A, t_node)
            sec.watch(G_k)
        return spatial(psi_proj_xr, psi_proj_yn, G_k, W_q)

    _sigma_kij_kernel_cache[key] = staged
    return staged


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
    def _build_W_t_q(B_q, Omega_q, mask_B, Om_lo, Om_hi, E_ref_B, t_node):
        """W(τ) = Σ_q B_q · exp(-i·(Ω_q - E_ref_B)·τ), windowed to (Om_lo, Om_hi].

        (A-side G now built inside sigma_kij_kernel via build_G_tau, so
        the tau-operand helper only shapes the PPM-pole-sum B-side.)

        THE WINDOW IS NOT MATERIALISED.  ``Om_lo``/``Om_hi`` are two f64
        SCALARS; the per-pole predicate is recomputed from ``Omega_q``
        right here and ANDed into the select that already existed, so XLA
        fuses it into that select's predicate and it never becomes a
        buffer.  What they replace was a full ``(nq, μ_pad, μ_pad)`` bool
        tile — ``base_mask_B & (Ω ≤ T)`` — built eagerly per window in
        ``ppm_sigma`` and pinned as a kernel operand for every τ node of
        the scan (~148 MiB/rank at μ_pad = 24,960 / P = 64, live through
        the most memory-intensive stage of Σ).  See
        ``ppm_windows.window_mask_B_bounds`` for the ``(lo, hi]``
        convention, which since 2026-08-10 is also the one
        ``windowed_exp_iEt`` uses on the A side — both sides of Σ now
        assign a boundary pole downward, into the pane whose supremum
        it is.

        ``mask_B`` stays an array: it is the fit-VALIDITY selector
        (``|Ω| > 1e-14`` ∧ the GN-PPM ``good`` flag), which is a property
        of each fitted pole and not an interval in Ω, so there are no two
        numbers to ship.  It is also ONE tile for the whole Σ stage
        rather than one per window.
        """
        phase_B = jnp.exp(-1j * (Omega_q - E_ref_B) * t_node)
        in_window = (Omega_q > Om_lo) & (Omega_q <= Om_hi)
        W_t_q = jnp.where(mask_B & in_window, B_q * phase_B,
                          jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        return jax.lax.with_sharding_constraint(W_t_q, q_mu_shard)

    @jax.jit
    def _tau_kernel(
        psi_coh_xn, psi_coh_yr,
        psi_proj_xr, psi_proj_yn,
        E_A, mask_A, B_q, Omega_q, mask_B, Om_lo, Om_hi,
        E_ref_A, E_ref_B, t_node,
    ):
        W_t_q = _build_W_t_q(B_q, Omega_q, mask_B, Om_lo, Om_hi,
                             E_ref_B, t_node)
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
            E_A, mask_A, B_q, Omega_q, mask_B, Om_lo, Om_hi,
            E_ref_A, E_ref_B, t_node,
        ):
            with timing.section("sigma.tau.w_phase") as sec:
                W_t_q = _build_W_t_q(B_q, Omega_q, mask_B, Om_lo, Om_hi,
                                     E_ref_B, t_node)
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


def build_shared_w_tau(B_poles, Omega_poles, pole_indices, bounds,
                       phase_real, E_ref_B, t_node):
    """Build one W(tau) tile from selected multipole fields.

    ``bounds`` rows are ``(a_gt, a_le, gamma_ge, gamma_gt, gamma_lt,
    gamma_le)``.  Each row selects one pole field; ``phase_real`` chooses
    the accepted near-axis functional ``Re(Omega)`` for that row, otherwise
    the fitted complex pole is used.  The pole axis is never materialized in
    W: the loop carries one ``(q, mu, nu)`` tile.
    """
    def _add(index, W_t):
        pole = jax.lax.dynamic_index_in_dim(
            pole_indices, index, axis=0, keepdims=False)
        omega = jax.lax.dynamic_index_in_dim(
            Omega_poles, pole, axis=0, keepdims=False)
        residue = jax.lax.dynamic_index_in_dim(
            B_poles, pole, axis=0, keepdims=False)
        b = jax.lax.dynamic_index_in_dim(
            bounds, index, axis=0, keepdims=False)
        use_real = jax.lax.dynamic_index_in_dim(
            phase_real, index, axis=0, keepdims=False)
        a = jnp.real(omega)
        gamma = -jnp.imag(omega)
        selected = ((a > b[0]) & (a <= b[1])
                    & (gamma >= b[2]) & (gamma > b[3])
                    & (gamma < b[4]) & (gamma <= b[5]))
        phase = jnp.where(use_real, a + 0.0j, omega)
        return W_t + jnp.where(
            selected,
            residue * jnp.exp(-1j * (phase - E_ref_B) * t_node),
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))

    return jax.lax.fori_loop(
        0, pole_indices.shape[0], _add, jnp.zeros_like(B_poles[0]))


def get_shared_sigma_tau_kernel(
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int],
) -> Callable[..., jax.Array]:
    """Return the GN tau kernel with a selected multipole W(tau) builder.

    All spatial work is the established ``build_G_tau -> fused GW FFT ->
    contract_bands`` path.  This wrapper changes only the scalar frequency
    synthesis: several poles and compatible windows are summed into one W
    tile before that unchanged convolution.  It always uses the single
    complex projection carrier; HGL's missing sine arm is completed once by
    :class:`gw.ppm_accumulators.DeviceOmegaAccumulator` after the tau sum.
    """
    kgrid = tuple(int(x) for x in kgrid)
    from ffi import ffi_dial_key

    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key())
    if key in _sigma_shared_tau_kernel_cache:
        return _sigma_shared_tau_kernel_cache[key]

    ensure_jax_compile_cache()
    q_mu_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    sigma_kij = _get_sigma_kij_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True)

    @jax.jit
    def _build(B_poles, Omega_poles, pole_indices, bounds,
               phase_real, E_ref_B, t_node):
        W_t = build_shared_w_tau(
            B_poles, Omega_poles, pole_indices, bounds,
            phase_real, E_ref_B, t_node)
        return jax.lax.with_sharding_constraint(W_t, q_mu_sharding)

    if not _stage_timing_enabled():
        @jax.jit
        def _tau(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, B_poles, Omega_poles, pole_indices, bounds,
            phase_real, E_ref_A, E_ref_B, t_node,
        ):
            W_t = _build(B_poles, Omega_poles, pole_indices, bounds,
                         phase_real, E_ref_B, t_node)
            return sigma_kij(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, E_ref_A, t_node, W_t)

        _sigma_shared_tau_kernel_cache[key] = _tau
        return _tau

    def _tau_staged(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, B_poles, Omega_poles, pole_indices, bounds,
        phase_real, E_ref_A, E_ref_B, t_node,
    ):
        with timing.section("sigma.tau.w_phase") as sec:
            W_t = _build(B_poles, Omega_poles, pole_indices, bounds,
                         phase_real, E_ref_B, t_node)
            sec.watch(W_t)
        return sigma_kij(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_t)

    _sigma_shared_tau_kernel_cache[key] = _tau_staged
    return _tau_staged


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
    #   * mask_B is ``base_mask_B`` at runtime and inherits Ω_q's sharding;
    #     the all-true dummy here matches that.  (The per-window Ω split it
    #     used to carry is now the Om_lo/Om_hi scalars below — see
    #     ``ppm_windows.window_mask_B_bounds``.)
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
    # The Ω window is two SCALARS now, not a (nq, μ, μ) tile — see
    # ``ppm_windows.window_mask_B_bounds``.  ±inf is the "all" window and
    # matches the runtime dtype/uncommitted-ness of the ``jnp.asarray``
    # the branch loop does, so the AOT signature still matches exactly.
    Om_lo   = jnp.asarray(-np.inf, dtype=jnp.float64)
    Om_hi   = jnp.asarray(np.inf, dtype=jnp.float64)
    E_ref_A = jnp.asarray(0.0, dtype=jnp.float64)
    E_ref_B = jnp.asarray(0.0, dtype=jnp.float64)
    t_node  = jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128)

    for tau_kernel in tau_kernels:
        if hasattr(tau_kernel, "lower"):
            tau_kernel.lower(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, ppm.B_q, ppm.Omega_q, mask_B, Om_lo, Om_hi,
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
                E_A, mask_A, ppm.B_q, ppm.Omega_q, mask_B, Om_lo, Om_hi,
                E_ref_A, E_ref_B, t_node,
            )
            jax.block_until_ready(out)

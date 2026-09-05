"""Shared device tau kernel for dynamic Sigma.

The single-tau integrand kernel plus its cache/AOT machinery:

    σ^τ_nmk(τ) = project[ FFT[ G(τ) · W(τ) / √N_k ] ]
    G(τ)       = diag[ e^{-i(E_A - E_ref_A)·τ} ] · mask_A           (A = val or cond)
    W(τ)       = Σ_pμν B_pq · e^{-i(Ω_pq - E_ref_B)·τ} · selector_p

This is the Σ_PPM file where SPMD / sharding / HLO expertise is required —
the deferred scan / collective-flush notes live here.  The projection tail
itself (the two-stage psum_scatter band reshard, its axis-order/stacking/
de-promotion levers and the gated MKL-GEMM FFI body) is SUBSUMED by the
shared primitive ``common.contract_bands.contract_bands_block_reshard``
(owner directive 2026-07-28) — this module keeps only the Σ-specific
channel algebra and the kernel plumbing around it.

The module-level kernel caches are co-located with the factories that read
them.  :func:`get_sigma_spatial_kernel` is the reusable
``G_k x W_q -> Sigma`` owner, and :func:`get_shared_sigma_tau_kernel` is the
only dynamic-pole synthesis wrapper used by ``gw.mpa.sigma``.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common import timing
from common.jax_compile_cache import ensure_jax_compile_cache


_sigma_kij_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}
#: Cache of :class:`SpatialKernel` PAIRS (prep_w, conv_project) — not of a
#: single callable, since 2026-08-15: the W-only half is hoistable and the
#: band-bracket loop hoists it.
_sigma_spatial_kernel_cache: dict[tuple[object, ...], "SpatialKernel"] = {}
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
    layout: str = "legacy", face_shape=None, face_band_extent=None,
) -> Callable[..., jax.Array]:
    """Build the shard_map'd ψ* σ ψ projector that reduce-scatters the output.

    ``layout='face'`` (2026-08-22, the dynamic PPM/MPA Σ_c(τ) face port):
    routes through ``common.contract_bands.contract_bands_block_reshard(
    layout='face', channels=...)`` instead of the shard_map body below —
    the SAME channel-plan dispatch (``merged_x`` -> ``channels='none'``
    single complex chain, else ``channels='split_reim'`` two-channel
    plan), just the face mechanism (two planned GEMMs) rather than the
    shard_map/psum_scatter chain.  Requires ``face_shape``.  See
    :func:`common.contract_bands._face_project_kernel`'s own docstring
    for the algebra.

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
      here only as an unselected low-level channel-plan option.  The sole
      shared dynamic executor selects the merged carrier below.
    * ``merged_x=True`` → primitive ``channels="none"``: the single
      complex chain X = ψ†σψ = S_R + i·S_I — the path EVERY Laplace
      dynamic window dispatches, licensed by BILINEARITY alone
      (their consumer forms only c·(S_R+i·S_I) = c·X); no symmetry
      assumption enters.  Half the
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
        layout=layout,
        face_shape=face_shape,
        face_band_extent=face_band_extent,
    )


class SpatialKernel(NamedTuple):
    """The ``G_k x W_q -> Sigma_kij`` owner, split at its ONE τ-local seam.

    ``prep_w(W_q) -> W_prep``
        Everything in the chain that depends on W and NOT on G.  On the
        decomposed chain that is ``ifftn(W)`` — the R-space screened
        interaction — and hoisting it is the one real saving available to a
        caller that contracts SEVERAL G(τ) against the same W(τ) (the band
        brackets).  On the fused ``gw_conv`` chain it is the IDENTITY,
        because that entry point's ABI takes W in k-space and performs its
        transform inside the pinned handler; see the note in
        :func:`get_sigma_spatial_kernel`.
    ``conv_project(psi_xr, psi_yn, G_k, W_prep) -> Sigma``
        The G-dependent remainder: the G transform, the R-space multiply,
        the forward transform and the ψ projection.  Paid ONCE PER G(τ).

    Composed back to back this is exactly the single callable this factory
    used to return; the split exists so the caller can place the loop
    boundary between the two halves instead of around both.
    """
    prep_w: Callable[..., jax.Array]
    conv_project: Callable[..., jax.Array]


def get_sigma_spatial_kernel(
    *,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    merged_x: bool = False,
    layout: str = "legacy",
    face_shape=None,
    face_band_extent=None,
) -> SpatialKernel:
    """Return the ansatz-neutral ``G_k x W_q -> Sigma_kij`` kernel.

    ``layout``/``face_shape`` (2026-08-22): forwarded ONLY to the
    projection tail (:func:`_make_project_ri_reduce_scatter`).  The FFT
    convolution above it (``ifftn``/``fftn``/the fused ``gw_conv`` FFI
    call) is layout-AGNOSTIC and unchanged either way: ``build_G``/
    ``build_G_tau`` already return the SAME ``(nk, s, μ_X, s, μ_Y)``
    G-tile shape/sharding under both layouts (``greens_function_kernel``'s
    own module docstring), so the spatial convolution never sees a psi
    operand at all — it is exactly the "convolution stays put" precedent
    ``cohsex_sigma._make_cohsex_kernels`` already set for its own
    ``_convolve`` closure, reused here rather than re-derived.

    This is the extension seam for alternate propagators and screened
    interactions.  GN-PPM and MPA construct their own ``G_k`` and ``W_q``;
    this owner performs only the flat-k FFT convolution and band projection.
    A two-point vertex-corrected W can therefore enter without forking the
    spatial kernel.  A genuine three-point vertex remains a different
    contraction and should not be hidden behind this interface.

    Returned as a :class:`SpatialKernel` pair rather than one callable so a
    caller with several G(τ) per W(τ) can hoist the W half.  WHAT THAT BUYS,
    AND WHERE IT DOES NOT:

      * decomposed chain (``LORRAX_FFT_FFI_FUSED=0``) — ``ifftn(W)`` is one
        of the three transforms, so hoisting it across ``n`` G's takes the
        FFT count from ``3n`` to ``2n + 1``;
      * fused chain (``=1``, the certified production default) — all three
        transforms are inside ONE ``gw_conv`` FFI call whose signature is
        ``(G_k, W_k) -> sigma_k``.  ``ifftn(W)`` therefore happens inside the
        handler and is recomputed per G.  Hoisting it needs a second handler
        entry that accepts W already in R-space, i.e. a C++/ABI change to
        ``liblorrax_ffi.so``; that is deliberately NOT done here.  The cost is
        ``n`` W-transforms instead of one — a third of the FFT chain.
    """
    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    from common.fft_helpers import (
        make_flat_k_fftn, make_flat_k_gw_conv, make_flat_k_ifftn)
    from ffi import ffi_dial_key
    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key(),
           bool(merged_x), layout, face_shape, face_band_extent)
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

    project = _make_project_ri_reduce_scatter(
        mesh_xy, merged_x=merged_x, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent)

    @jax.jit
    def prep_w(W_q):
        """The W-only half of the chain — see :class:`SpatialKernel`."""
        if use_fused_ffi:
            return W_q
        return _V_ifftn(W_q)[:, None, :, None, :]

    @partial(jax.jit, donate_argnums=(2,))
    def conv_project(psi_proj_xr, psi_proj_yn, G_k, W_prep):
        if use_fused_ffi:
            sigma_k = _gw_conv(G_k, W_prep)
        else:
            sigma_k = _G_fftn(_G_ifftn(G_k) * W_prep * inv_sqrt_nk)
        return project(psi_proj_xr, sigma_k, psi_proj_yn)
    if not _stage_timing_enabled():
        pair = SpatialKernel(prep_w=prep_w, conv_project=conv_project)
        _sigma_spatial_kernel_cache[key] = pair
        return pair
    if use_fused_ffi:
        _conv_j = jax.jit(_gw_conv, donate_argnums=(0,))
    else:
        _G_ifft_j = jax.jit(_G_ifftn, donate_argnums=(0,))
        _V_ifft_j = jax.jit(lambda W_q: _V_ifftn(W_q)[:, None, :, None, :],
                            donate_argnums=(0,))
        _mult_fft_j = jax.jit(lambda G_R, V_R: _G_fftn(G_R * V_R * inv_sqrt_nk),
                              donate_argnums=(0,))
    _project_j = jax.jit(project, donate_argnums=(1,))

    def prep_w_staged(W_q):
        """``sigma.tau.w_prep`` — the ONCE-PER-τ half, timed on its own row.

        On the fused chain this is the identity and the row reads ~0: that
        is the measurement, not an instrumentation gap.  It is what says
        whether ``ifftn(W)`` was genuinely hoisted or is being paid inside
        ``sigma.tau.GW_conv_ffi`` once per bracket.
        """
        if use_fused_ffi:
            return W_q
        with timing.section("sigma.tau.w_prep") as sec:
            V_R = _V_ifft_j(W_q)
            sec.watch(V_R)
        return V_R

    def conv_project_staged(psi_proj_xr, psi_proj_yn, G_k, W_prep):
        """Diagnostic split of the same spatial operation sequence."""
        if use_fused_ffi:
            with timing.section("sigma.tau.GW_conv_ffi") as sec:
                sigma_k = _conv_j(G_k, W_prep)
                sec.watch(sigma_k)
        else:
            with timing.section("sigma.tau.G_ifft") as sec:
                G_R = _G_ifft_j(G_k)
                sec.watch(G_R)
            with timing.section("sigma.tau.GW_mult_fft") as sec:
                sigma_k = _mult_fft_j(G_R, W_prep)
                sec.watch(sigma_k)
        with timing.section("sigma.tau.project_rs") as sec:
            out = _project_j(psi_proj_xr, sigma_k, psi_proj_yn)
            sec.watch(out)
        return out

    pair = SpatialKernel(prep_w=prep_w_staged, conv_project=conv_project_staged)
    _sigma_spatial_kernel_cache[key] = pair
    return pair


#: The band-bracket plan a caller gets when it asks for none: ONE bracket
#: over every band, and NO leading bracket axis on the output.  This is the
#: MPA / shared-multipole shape and it is what ``brackets=None`` means.
_NO_BRACKETS = None


def _stack_channels(outs, mesh_xy: Mesh):
    """Stack a per-bracket list of kernel outputs on a new LEADING axis.

    The τ kernel's output is either one complex array (merged Laplace plan)
    or the ``(S_R, S_I)`` pair (crossing plan), so the stack has to be
    channel-wise; a single ``jnp.stack`` on the tuple would build a
    (2, n_brk, ...) object and silently swap the two axes' meaning.

    The bracket axis is pinned REPLICATED and the (nk, m_X, n_Y) sharding
    the reduce-scatter projector produced is restated explicitly rather than
    left to XLA's propagation through the concatenate: the accumulator reads
    ``addressable_shards``/``.sharding`` off this array and places every
    host tile by that index, so a silently drifted layout would misplace
    tiles rather than fail.
    """
    spec = P(None, None, 'x', 'y')

    def _one(chan):
        return jax.lax.with_sharding_constraint(
            jnp.stack(chan, axis=0), NamedSharding(mesh_xy, spec))

    if isinstance(outs[0], tuple):
        return tuple(_one([o[c] for o in outs])
                     for c in range(len(outs[0])))
    return _one(outs)


def _get_sigma_kij_kernel(
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int], merged_x: bool = False,
    brackets: tuple[tuple[int, int], ...] | None = _NO_BRACKETS,
    layout: str = "legacy", face_shape=None, face_band_extent=None,
    pack_brackets: bool = True,
    energy_windows: bool = False,
) -> Callable[..., jax.Array]:
    """GN/MPA adapter that builds G and calls the shared spatial kernel.

    ``brackets``
        ``None`` (the MPA/shared-multipole shape): one G(τ) over the whole
        band axis, output shape exactly as it has always been.
        A tuple of half-open ``(lo, hi)`` band ranges (the GN-PPM shape,
        length 1 by default): one G(τ) PER BRACKET, each contracted against
        the SAME ``prep_w(W_q)``, stacked on a new LEADING axis.

    THE BRACKETS ARE STATIC SLICES, NOT MASKS — under ``layout='legacy'``.
    A boolean mask would leave every bracket's ``build_G_tau`` einsum
    contracting over all ``nb`` bands with two thirds of them multiplied by
    zero — the band-side GEMM would cost 3× with nothing to show for it.
    Slicing makes the partition real: each band's orbital outer product is
    formed exactly once across the three brackets, so the G-build work is
    the same total as one full-band run.  The bounds are Python ints baked
    into the trace, so this is still ONE compiled program and ONE dispatch
    per τ.

    WHAT DOES NOT PARTITION, and why the feature is not free: ``G_k`` lives
    in the ISDF centroid basis, whose extent is ``N_μ`` — independent of how
    many bands went into it.  The FFT chain and the ψ projection that follow
    therefore cost the same per bracket as they do for the full band range,
    and three brackets pay them three times.  That 3× on the dominant term
    is the accepted price of the feature; see :class:`SpatialKernel` for the
    one part of it (``ifftn(W)``) that can be hoisted, and where it cannot.

    ``layout='face'`` (2026-08-22): the two-face carrier's ``psi_mun``/
    ``psi_nmu`` span the FULL [b0,b4) loaded extent and cannot legally be
    SLICED to an arbitrary band sub-range — a legal face matrix must stay
    mesh-divisible on both axes, and an arbitrary bracket edge is not
    (report obstacle #3).  Two sibling bracket loops exist for this layout:

    * :func:`_bracketed_face` (the ORIGINAL, 2026-08-22 bring-up) never
      slices ψ — every bracket pays the FULL nb_full G-build and the
      partition is expressed as a band-range MASK intersected into
      ``mask_A`` instead ("weight, don't window", the same fix
      ``build_G_tau``'s own ``mask``/``phases`` seam and
      ``Wavefunctions.band_mask`` already use for this exact problem).
      This is a NAMED extra cost over legacy's disjoint-slice partition
      (bracket count × the full G-build instead of one full-band G-build
      split into disjoint pieces).  KEPT as the fallback for windows where
      packing cannot apply, and as the parity oracle every packed-vs-mask
      test in this tree diffs against.
    * :func:`_bracketed_face_packed` (2026-08-23, ``pack_brackets=True``,
      the default) uses ``wavefunction_bundle.pack_band_window`` to
      materialize each bracket's OWN compact ψ pair ONCE (a one-time
      distributed repack, done by the CALLER before this kernel's first
      per-τ dispatch — see that function's docstring), so each bracket's
      G-build GEMM runs at its OWN padded width instead of the shared
      full-``nb_full`` one.  Engaged only when there is more than one
      bracket (``len(brackets) > 1``): the production
      ``head=full``/``gn_ppm`` single-bracket deck already pays the
      SAME one full-width G-build either way (``pack_band_window``'s own
      trivial-window fast path returns the resident carrier unchanged),
      so routing it through the packed calling convention would add a
      new argument shape to the most-exercised, already-gated path for
      zero benefit — deliberately scoped OUT of the packed convention
      rather than unified for its own sake.  When packing applies,
      ``psi_coh_xn``/``psi_coh_yr`` (the two slots this factory's callers
      hand the "coh"/G-build ψ operands through) are TUPLES of length
      ``len(brackets)`` instead of bare arrays — see
      :func:`_bracketed_face_packed`'s own docstring for the contract.

    Also builds this layout's own ``distrib_la.gemm_plan``(s) for the
    G-build ONCE, here, at kernel-factory time — mirroring
    ``cohsex_sigma._make_cohsex_kernels_face``'s identical G-plan
    construction, never resolved inside the per-τ ``lax.scan``: ONE plan
    at the full ``nb_full`` width (used by the trivial-bracket / MASK /
    ``brackets is None`` paths), plus — when packing applies — one
    ADDITIONAL plan per bracket, each sized at that bracket's own padded
    width.  Plans are cheap to build eagerly relative to a Σ run (one
    warmup call each, at factory time, amortized over every τ node the
    returned kernel is later called for); building ``n_brackets`` of them
    is the intended trade for shrinking every one of possibly hundreds of
    per-τ G-build GEMMs from width ``nb_full`` to the bracket's own width.
    """
    if layout not in ("legacy", "face"):
        raise ValueError(
            f"_get_sigma_kij_kernel: layout must be 'legacy' or 'face', "
            f"got {layout!r}")
    if layout == "face" and face_shape is None:
        raise ValueError(
            "_get_sigma_kij_kernel(layout='face') requires "
            "face_shape=(nk, nb_full, n_rmu, nspinor)")
    from ffi import ffi_dial_key
    key = (id(mesh_xy), tuple(map(int, kgrid)), _stage_timing_enabled(),
           ffi_dial_key(), bool(merged_x), brackets, layout, face_shape,
           face_band_extent, bool(pack_brackets), bool(energy_windows))
    if key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[key]
    from .greens_function_kernel import build_G_tau
    spatial = get_sigma_spatial_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=merged_x,
        layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent)

    g_plan = None
    g_plans = None
    use_packed = False
    if layout == "face":
        from distrib_la import gemm_plan
        nk_f, nb_full_f, n_rmu_f, ns_f = (int(v) for v in face_shape)
        mu_s_f = n_rmu_f * ns_f
        g_plan = gemm_plan(mesh_xy, m=mu_s_f, k=nb_full_f, n=mu_s_f,
                           nq=nk_f, dtype=jnp.complex128)
        use_packed = bool(pack_brackets) and brackets is not None \
            and len(brackets) > 1
        if use_packed:
            from runtime.padding import padded_axis
            px = int(mesh_xy.shape["x"])
            py = int(mesh_xy.shape["y"])
            if px != py:
                raise ValueError(
                    f"_get_sigma_kij_kernel(layout='face', pack_brackets="
                    f"True): mesh is {px}x{py}, not square -- see "
                    f"wavefunction_bundle.pack_band_window's identical "
                    f"refusal.")
            g_plans = []
            for lo, hi in brackets:
                hi_ = nb_full_f if hi is None else hi
                width = hi_ - lo
                w_pad = padded_axis(
                    width, mesh_xy, name="PPM bracket band carrier",
                    specs=((P(None, None, None, "x"), 3),
                           (P(None, "y", None, None), 1))).carrier
                if lo == 0 and hi_ == nb_full_f:
                    g_plans.append(g_plan)   # identical shape -- reuse
                else:
                    g_plans.append(gemm_plan(
                        mesh_xy, m=mu_s_f, k=w_pad, n=mu_s_f, nq=nk_f,
                        dtype=jnp.complex128))

    def _g_from_selector(xn, yr, E, sel, E_min, E_max, ref, t):
        # The A-side selector operand is dtype-dispatched (static at trace
        # time): bool = the incumbent band-identity mask, bit-exact; float =
        # the metallic mask×weight product (f or 1−f folded by the executor),
        # applied through build_G_tau's band_weight seam — one G builder,
        # weights never clipped (finite-occupation-screening.md).
        #
        # THE BAND BRACKETS AND THE OCCUPATION WEIGHT ARE SEPARABLE, and this
        # is where that separability is cashed in.  The bracket loop below
        # slices EVERY per-band operand -- ``sel`` included -- before calling
        # this builder, and ``sel[:, lo:hi]`` is the same index operation
        # whether ``sel`` is a bool band-identity mask or a float occupation
        # weight.  The partition is an index operation on the band axis; the
        # occupation is an elementwise factor along it.  Neither needs to know
        # about the other, so this dispatch stays dtype-only and the bracket
        # loop stays occupation-agnostic.
        if sel.dtype == jnp.bool_:
            if energy_windows:
                return build_G_tau(
                    xn, yr, E, 1j * t, e_ref=ref, mask=sel,
                    E_min=E_min, E_max=E_max, layout=layout, gemm=g_plan)
            return build_G_tau(xn, yr, E, 1j * t, e_ref=ref, mask=sel,
                               layout=layout, gemm=g_plan)
        if energy_windows:
            return build_G_tau(
                xn, yr, E, 1j * t, e_ref=ref, band_weight=sel,
                E_min=E_min, E_max=E_max, layout=layout, gemm=g_plan)
        return build_G_tau(xn, yr, E, 1j * t, e_ref=ref, band_weight=sel,
                           layout=layout, gemm=g_plan)

    def _bracketed_face(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                        E_A, mask_A, E_min, E_max, E_ref_A, t_node,
                        W_prep, build_g, conv):
        """Face sibling of :func:`_bracketed` — see this factory's own
        docstring's ``layout='face'`` section.  ψ is never sliced; each
        bracket narrows ``mask_A`` by a band-range predicate instead.
        ``mask_A``'s LAST axis is always the band axis (both the ``(nk,
        nb)`` and ``(1, nk, nb)`` shapes the legacy sibling's own comment
        documents), so broadcasting a ``(nb_full,)`` predicate against it
        is correct at either rank with no reshape."""
        nb_full = int(mask_A.shape[-1])
        idx = jnp.arange(nb_full)
        outs = []
        prev = None
        for lo, hi in brackets:
            if prev is not None:
                (psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                 E_A, mask_A, W_prep, prev) = jax.lax.optimization_barrier(
                    (psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                     E_A, mask_A, W_prep, prev))
            hi_ = nb_full if hi is None else hi
            in_range = (idx >= lo) & (idx < hi_)
            mask_bracket = (mask_A & in_range if mask_A.dtype == jnp.bool_
                           else mask_A * in_range.astype(mask_A.dtype))
            G_k = build_g(psi_coh_xn, psi_coh_yr, E_A, mask_bracket,
                         E_min, E_max, E_ref_A, t_node)
            prev = conv(psi_proj_xr, psi_proj_yn, G_k, W_prep)
            outs.append(prev)
        return _stack_channels(outs, mesh_xy)

    def _g_from_selector_packed(
        xn, yr, E, sel, E_min, E_max, ref, t, gemm,
    ):
        """Same dtype dispatch as :func:`_g_from_selector`, but ``gemm`` is
        an explicit argument rather than the factory's shared ``g_plan``
        closure — each bracket calls this with ITS OWN packed-width plan
        (:func:`_bracketed_face_packed`)."""
        if sel.dtype == jnp.bool_:
            if energy_windows:
                return build_G_tau(
                    xn, yr, E, 1j * t, e_ref=ref, mask=sel,
                    E_min=E_min, E_max=E_max, layout="face", gemm=gemm)
            return build_G_tau(xn, yr, E, 1j * t, e_ref=ref, mask=sel,
                               layout="face", gemm=gemm)
        if energy_windows:
            return build_G_tau(
                xn, yr, E, 1j * t, e_ref=ref, band_weight=sel,
                E_min=E_min, E_max=E_max, layout="face", gemm=gemm)
        return build_G_tau(xn, yr, E, 1j * t, e_ref=ref, band_weight=sel,
                           layout="face", gemm=gemm)

    def _bracketed_face_packed(psi_coh_xn, psi_coh_yr, psi_proj_xr,
                               psi_proj_yn, E_A, mask_A, E_min, E_max,
                               E_ref_A, t_node, W_prep, g_plans, conv):
        """Packed sibling of :func:`_bracketed_face` — see this factory's
        own docstring's ``layout='face'`` section for when this is used
        instead of the mask loop.

        ``psi_coh_xn``/``psi_coh_yr`` are TUPLES of length
        ``len(brackets)``: the packed per-bracket ``(psi_mun_w, psi_nmu_w)``
        pairs a caller built ONCE via
        ``wavefunction_bundle.pack_band_window`` — one call per bracket, at
        Sigma-procedure start (``ppm_sigma.compute_sigma_c_ppm_omega_grid``)
        — and reused across every τ node / ω window this kernel is later
        called for.  This function itself does NO packing and issues NO
        collective; it only indexes into the tuples and slices the small
        REPLICATED ``E_A``/``mask_A`` operands to match (a local per-rank
        op, exactly like the mask sibling's own ``[:, lo:hi]``/``[...,
        lo:hi]`` slices — the difference is that HERE ψ itself has already
        been narrowed too, so the G-build GEMM below runs at the bracket's
        own padded width via ``g_plans[i]`` instead of the shared
        full-``nb_full`` plan every masked call pays).

        Zero-pad columns beyond each bracket's true width are exact-zero ψ
        (``pack_band_window``'s own contract), so the ``E_A``/``mask_A``
        pad value below is arithmetically inert — matched to whatever
        ``jnp.pad``'s zero fill gives (``False``/``0.0``) rather than
        computed, for the same reason the ψ pad needs no special value.

        Same ONE-G-at-a-time discipline as :func:`_bracketed_face`
        (optimization barrier threading ``prev``), same output stacking.
        """
        nb_full = int(mask_A.shape[-1])
        outs = []
        prev = None
        for i, (lo, hi) in enumerate(brackets):
            if prev is not None:
                (psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                 E_A, mask_A, W_prep, prev) = jax.lax.optimization_barrier(
                    (psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                     E_A, mask_A, W_prep, prev))
            hi_ = nb_full if hi is None else hi
            width = hi_ - lo
            w_pad = int(psi_coh_xn[i].shape[-1])
            E_w = jax.lax.slice_in_dim(E_A, lo, hi_, axis=1)
            sel_w = jax.lax.slice_in_dim(mask_A, lo, hi_, axis=mask_A.ndim - 1)
            if w_pad != width:
                E_w = jnp.pad(E_w, ((0, 0), (0, w_pad - width)))
                sel_w = jnp.pad(
                    sel_w,
                    [(0, 0)] * (sel_w.ndim - 1) + [(0, w_pad - width)])
            G_k = _g_from_selector_packed(
                psi_coh_xn[i], psi_coh_yr[i], E_w, sel_w, E_min, E_max,
                E_ref_A, t_node, g_plans[i])
            prev = conv(psi_proj_xr, psi_proj_yn, G_k, W_prep)
            outs.append(prev)
        return _stack_channels(outs, mesh_xy)

    def _bracketed(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                   E_A, mask_A, E_min, E_max, E_ref_A, t_node,
                   W_prep, build_g, conv):
        """The bracket loop.  ONE G(τ) live at a time, by construction."""
        outs = []
        prev = None
        for lo, hi in brackets:
            if prev is not None:
                # ONE G(τ) AT A TIME — the owner's memory constraint, made
                # structural rather than hoped for.  The brackets are
                # mutually independent, so XLA is free to schedule all three
                # G builds ahead of the first contraction that consumes one;
                # that would hold three centroid-basis G tiles live where the
                # constraint allows one.  Threading the previous bracket's
                # RESULT through an optimization barrier alongside this
                # bracket's operands makes bracket i's output an ancestor of
                # bracket i+1's G build, so the build cannot be hoisted above
                # the consumption and ordinary liveness frees each G before
                # the next is allocated.  A barrier moves no data.
                (psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                 E_A, mask_A, W_prep, prev) = jax.lax.optimization_barrier(
                    (psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                     E_A, mask_A, W_prep, prev))
            # ``mask_A[..., lo:hi]``, NOT ``mask_A[:, lo:hi]``.  The band axis
            # is the LAST one on this array and only sometimes the second:
            # the occupancy mask arrives as ``(nk, nb)`` on a 2x2 processor
            # mesh, which squeezes the leading nspin axis, and as
            # ``(1, nk, nb)`` on a 1x1 mesh, which does not.  ``[:, lo:hi]``
            # is the band axis in the first case and cuts *nk* in the second,
            # so a 1x1-mesh run died in ``build_G_tau``'s shape normalisation
            # with ``cannot reshape (1, 9, 52) into (9, 42)`` (MoS2 gnppm,
            # 3 brackets) and ``(1, 20, 20) -> (64, 20)`` (Si, ONE bracket —
            # so this was never specific to the extrapolation; the trivial
            # single-bracket plan hits it too, which made it a defect in the
            # DEFAULT path).  Ellipsis indexing is correct at either rank.
            # ``psi_coh_yr`` and ``E_A`` keep ``[:, lo:hi]`` because their
            # band axis really is axis 1 at every rank they are built with.
            # ``greens_function_kernel.build_G_tau``'s reshape-to-``enk``
            # workaround still runs and is now redundant rather than
            # load-bearing: it could not save this path anyway, because it
            # runs AFTER the slice that corrupted the array.
            G_k = build_g(psi_coh_xn[..., lo:hi], psi_coh_yr[:, lo:hi],
                          E_A[:, lo:hi], mask_A[..., lo:hi], E_min, E_max,
                          E_ref_A, t_node)
            prev = conv(psi_proj_xr, psi_proj_yn, G_k, W_prep)
            outs.append(prev)
        return _stack_channels(outs, mesh_xy)

    if not _stage_timing_enabled():
        _build_g = _g_from_selector

        def _kernel_impl(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_q,
        ):
            # ONE W preparation per τ, ABOVE the bracket loop.  Explicit, not
            # left to CSE: on the decomposed chain this is ``ifftn(W)``, the
            # only transform in the chain that does not depend on G.
            W_prep = spatial.prep_w(W_q)
            if brackets is None:
                G_k = _build_g(psi_coh_xn, psi_coh_yr, E_A, mask_A,
                               E_min, E_max, E_ref_A, t_node)
                return spatial.conv_project(
                    psi_proj_xr, psi_proj_yn, G_k, W_prep)
            if use_packed:
                return _bracketed_face_packed(
                    psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                    E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_prep,
                    g_plans, spatial.conv_project)
            bracket_loop = _bracketed_face if layout == "face" else _bracketed
            return bracket_loop(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_prep,
                _build_g, spatial.conv_project)

        if energy_windows:
            kernel = partial(jax.jit, donate_argnums=(10,))(_kernel_impl)
        else:
            @partial(jax.jit, donate_argnums=(8,))
            def kernel(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, E_ref_A, t_node, W_q,
            ):
                return _kernel_impl(
                    psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                    E_A, mask_A, None, None, E_ref_A, t_node, W_q)

        _sigma_kij_kernel_cache[key] = kernel
        return kernel

    if layout == "face":
        raise NotImplementedError(
            "_get_sigma_kij_kernel(layout='face'): LORRAX_SIGMA_TAU_TIMING "
            "stage-split diagnostic is not ported for the face carrier — "
            "an opt-in profiling knob, not the production path; set "
            "LORRAX_SIGMA_TAU_TIMING=0 (the default) under low_mem_bands.")

    build_g = jax.jit(_g_from_selector)

    def _build_g_timed(xn, yr, E, mask, E_min, E_max, ref, t):
        with timing.section("sigma.tau.G_build") as sec:
            G_k = build_g(xn, yr, E, mask, E_min, E_max, ref, t)
            sec.watch(G_k)
        return G_k

    def _staged_impl(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_q,
    ):
        W_prep = spatial.prep_w(W_q)
        if brackets is None:
            G_k = _build_g_timed(psi_coh_xn, psi_coh_yr, E_A, mask_A,
                                 E_min, E_max, E_ref_A, t_node)
            return spatial.conv_project(
                psi_proj_xr, psi_proj_yn, G_k, W_prep)
        # Python-level loop over already-jitted stages: the barrier is not
        # available (and not needed) here — each stage jit completes before
        # the next is dispatched, so the one-G-at-a-time property holds a
        # fortiori.  Walltime is NOT comparable to the fused path.
        outs = []
        for lo, hi in brackets:
            # ``[..., lo:hi]`` for the same reason as the fused path above —
            # this is the SECOND copy of that slice and it carried the same
            # defect.  The registered issue named only the fused one
            # (``ppm_tau_kernel.py:465``), so a fix applied there alone would
            # have left every ``LORRAX_SIGMA_STAGE_TIMING`` run on a 1x1 mesh
            # still broken, and broken in the mode an operator reaches for
            # precisely when they are already debugging.
            G_k = _build_g_timed(
                psi_coh_xn[..., lo:hi], psi_coh_yr[:, lo:hi],
                E_A[:, lo:hi], mask_A[..., lo:hi], E_min, E_max,
                E_ref_A, t_node)
            outs.append(spatial.conv_project(
                psi_proj_xr, psi_proj_yn, G_k, W_prep))
        return _stack_channels(outs, mesh_xy)

    if energy_windows:
        staged = _staged_impl
    else:
        def staged(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_q,
        ):
            return _staged_impl(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, None, None, E_ref_A, t_node, W_q)

    _sigma_kij_kernel_cache[key] = staged
    return staged




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
    brackets: tuple[tuple[int, int], ...] | None = _NO_BRACKETS,
    layout: str = "legacy", face_shape=None, face_band_extent=None,
    pack_brackets: bool = True,
) -> Callable[..., jax.Array]:
    """Return the GN tau kernel with a selected multipole W(tau) builder.

    All spatial work is the established ``build_G_tau -> fused GW FFT ->
    contract_bands`` path.  This wrapper changes only the scalar frequency
    synthesis: several poles and compatible windows are summed into one W
    tile before that unchanged convolution.  It always uses the single
    complex projection carrier; HGL's missing sine arm is completed once by
    :class:`gw.ppm_accumulators.DeviceOmegaAccumulator` after the tau sum.

    This is the entry point ``gw.mpa.sigma`` calls for every dynamic pole
    model.  ``brackets=None`` retains the ordinary MPA shape; a tuple asks the
    same spatial kernel for a leading disjoint band-bracket axis.  ``layout``,
    ``face_shape`` and ``pack_brackets`` forward to
    :func:`_get_sigma_kij_kernel` unchanged.
    """
    kgrid = tuple(int(x) for x in kgrid)
    if brackets is not None:
        brackets = tuple((int(lo), None if hi is None else int(hi))
                         for lo, hi in brackets)
    from ffi import ffi_dial_key

    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key(),
           brackets, layout, face_shape, face_band_extent,
           bool(pack_brackets))
    if key in _sigma_shared_tau_kernel_cache:
        return _sigma_shared_tau_kernel_cache[key]

    ensure_jax_compile_cache()
    q_mu_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))

    sigma_kij = _get_sigma_kij_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True,
        brackets=brackets, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent,
        pack_brackets=pack_brackets)

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

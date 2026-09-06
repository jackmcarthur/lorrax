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

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common import timing
from common.jax_compile_cache import ensure_jax_compile_cache
from runtime.env_flags import env_bool


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
    return env_bool("LORRAX_SIGMA_TAU_TIMING", False)


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
    mesh_xy: Mesh, *, merged_x: bool = True,
    layout: str = "face", face_shape=None, face_band_extent=None,
    k_unfold_plan=None,
) -> Callable[..., jax.Array]:
    """Project raw-parent rows; the completed frequency sum owns the band unfold."""
    from common.contract_bands import contract_bands_block_reshard

    if k_unfold_plan is None or face_shape is None:
        raise ValueError("Sigma projection requires canonical face shapes and a typed parent unfold plan.")
    if layout != "face" or not merged_x:
        raise ValueError(
            "_make_project_ri_reduce_scatter(k_unfold_plan=...) requires the "
            "face layout and the merged single-complex projection chain.")
    inner = contract_bands_block_reshard(
        mesh_xy, channels="none", layout="face",
        face_shape=(k_unfold_plan.n_parent, *face_shape[1:]),
        face_band_extent=face_band_extent)
    k_rows = np.asarray(k_unfold_plan.parent_full_rows, dtype=np.int32)

    def project(psi_xr, sigma_k, psi_yn):
        sigma_parent = jnp.take(sigma_k, jnp.asarray(k_rows), axis=0)
        return inner(psi_xr, sigma_parent, psi_yn)

    return project


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
    merged_x: bool = True,
    layout: str = "face",
    face_shape=None,
    face_band_extent=None,
    k_unfold_plan=None,
) -> SpatialKernel:
    """Convolve a Green tile with one prepared W tile and project on typed raw parents."""
    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    from common.fft_helpers import (
        make_flat_k_fftn, make_flat_k_gw_conv, make_flat_k_ifftn)
    from ffi import ffi_dial_key
    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key(),
           bool(merged_x), layout, face_shape, face_band_extent,
           k_unfold_plan)
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
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan)

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
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int], merged_x: bool = True,
    brackets: tuple[tuple[int, int], ...] | None = _NO_BRACKETS,
    layout: str = "face", face_shape=None, face_band_extent=None,
    energy_windows: bool = False,
    k_unfold_plan=None,
) -> Callable[..., jax.Array]:
    """Build Green functions with band-range masks and contract each bracket against one prepared W."""
    if layout != "face" or face_shape is None or k_unfold_plan is None:
        raise ValueError("Sigma tau requires canonical face shapes and a typed parent unfold plan.")
    from ffi import ffi_dial_key
    key = (id(mesh_xy), tuple(map(int, kgrid)), _stage_timing_enabled(),
           ffi_dial_key(), bool(merged_x), brackets, layout, face_shape,
           face_band_extent, bool(energy_windows),
           k_unfold_plan)
    if key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[key]
    from .greens_function_kernel import build_G_tau
    # G, W and projection faces share the run's packed centroid order,
    # as in the static kernels. Pole batches convert only at the store seam.
    spatial = get_sigma_spatial_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=merged_x,
        layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan)

    from distrib_la import gemm_plan
    _, nb_full_f, n_rmu_f, ns_f = (int(v) for v in face_shape)
    mu_s_f = n_rmu_f * ns_f
    g_plan = gemm_plan(mesh_xy, m=mu_s_f, k=nb_full_f, n=mu_s_f,
                       nq=k_unfold_plan.n_full, dtype=jnp.complex128)

    from common.shard_map import shard_map
    specs = (P(None, None, "x", "y"), P(None, "x", None, "y"))

    def unfold(xn, yr):
        return (k_unfold_plan.unfold_face(
            xn, vertex=0, spin_axis=1, mu_axis=2, mesh_axis="x"),
                k_unfold_plan.unfold_face(
            yr, vertex=0, spin_axis=2, mu_axis=3, mesh_axis="y"))
    unfold = shard_map(unfold, mesh=mesh_xy, in_specs=specs,
                       out_specs=specs, check_vma=False)

    def _g_from_selector(xn, yr, E, sel, E_min, E_max, ref, t):
        """Apply boolean identity masks or signed occupation weights without clipping."""
        xn, yr = unfold(xn, yr)
        rows = jnp.asarray(k_unfold_plan.irr_idx)
        sel = jnp.take(jnp.reshape(sel, E.shape), rows, axis=0)
        E = jnp.take(E, rows, axis=0)
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
        """Mask each bracket on the last band axis while retaining one Green tile at a time."""
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
            return _bracketed_face(
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

    if brackets is not None:
        raise NotImplementedError(
            "_get_sigma_kij_kernel(layout='face'): LORRAX_SIGMA_TAU_TIMING "
            "stage-split diagnostic is not ported for bracketed face "
            "carriers — an opt-in profiling knob, not the production path; "
            "set LORRAX_SIGMA_TAU_TIMING=0 (the default) for that case.")

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
        G_k = _build_g_timed(psi_coh_xn, psi_coh_yr, E_A, mask_A,
                             E_min, E_max, E_ref_A, t_node)
        return spatial.conv_project(psi_proj_xr, psi_proj_yn, G_k, W_prep)

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
    layout: str = "face", face_shape=None, face_band_extent=None,
    k_unfold_plan=None,
) -> Callable[..., jax.Array]:
    """Build selected multipole W(tau) tiles for the shared complex Sigma contraction."""
    kgrid = tuple(int(x) for x in kgrid)
    if brackets is not None:
        brackets = tuple((int(lo), None if hi is None else int(hi))
                         for lo, hi in brackets)
    from ffi import ffi_dial_key

    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key(),
           brackets, layout, face_shape, face_band_extent,
           k_unfold_plan)
    if key in _sigma_shared_tau_kernel_cache:
        return _sigma_shared_tau_kernel_cache[key]

    ensure_jax_compile_cache()
    q_mu_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))

    sigma_kij = _get_sigma_kij_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True,
        brackets=brackets, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent,
        k_unfold_plan=k_unfold_plan)

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

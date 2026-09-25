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


# THE names of the staged tau diagnostic's bands, owned here because this
# module is what opens the sections.  ``gw.mpa.sigma`` aggregates exactly
# this tuple; a name that is not spelled the same in both places is a row
# that silently disappears from the profile, so both ends read these.
TAU_PHASE_W_PHASE = "sigma.tau.w_phase"
TAU_PHASE_W_PREP = "sigma.tau.w_prep"
TAU_PHASE_G_BUILD = "sigma.tau.G_build"
TAU_PHASE_GW_CONV_FFI = "sigma.tau.GW_conv_ffi"
TAU_PHASE_PROJECT_RS = "sigma.tau.project_rs"

#: In dispatch order within one tau node.
TAU_KERNEL_PROFILE_PHASES = (
    TAU_PHASE_W_PHASE,
    TAU_PHASE_W_PREP,
    TAU_PHASE_G_BUILD,
    TAU_PHASE_GW_CONV_FFI,
    TAU_PHASE_PROJECT_RS,
)


def _stage_timing_enabled() -> bool:
    """``LORRAX_SIGMA_TAU_TIMING=1`` selects the stage-split instrumented τ kernel.

    Diagnostic knob (2026-07-28; evidence: AQ 4962c/P=64 HLO module_0912 —
    'sigma.exec 272.040' is a single opaque row, 176 τ dispatches at a uniform
    ~1.51 s that no existing timing row decomposes).  When ON, the per-τ body
    is dispatched as its cached stage jits (W-phase build / W prep / G build /
    the fused G·W k-convolution / ψ-projection + reduce-scatter), each
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


def _make_project_ri_reduce_scatter(
    mesh_xy: Mesh, *, merged_x: bool = True,
    layout: str = "face", face_shape=None, face_band_extent=None,
    k_unfold_plan=None,
) -> Callable[..., jax.Array]:
    """Project raw-parent rows; the completed frequency sum owns the band unfold."""
    from common.contract_bands import contract_bands_block_reshard

    if k_unfold_plan is None or face_shape is None:
        raise ValueError("Sigma projection requires canonical face shapes and a typed parent unfold plan.")
    if layout not in ("face", "axis") or not merged_x:
        raise ValueError(
            "_make_project_ri_reduce_scatter(k_unfold_plan=...) requires the "
            "face layout and the merged single-complex projection chain.")
    inner = contract_bands_block_reshard(
        mesh_xy, channels="none", layout=layout,
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
        Everything in the chain that depends on W and NOT on G: the
        k-convolution router's ``prep`` (``ifftn(W)`` into R space on CUDA,
        the identity on the cpu handler, which transforms W itself).  Hoisting
        it is the saving available to a caller that contracts SEVERAL G(τ)
        against the same W(τ) (the band brackets).
    ``conv_project(psi_xr, psi_yn, G_parents, W_prep) -> Sigma``
        The G-dependent remainder: the router's fused unfold convolution
        (``make_kconv_klead_unfold``: the typed unfold of the raw-parent
        Green, its spin action and the spin-major reorder on the load, then
        the G transform, R-space multiply and forward transform, one pass)
        and the ψ projection.  Paid ONCE PER G(τ).  ``G_parents`` is the
        :class:`gw.greens_function_kernel.ParentGreen` pair; Σ_k leaves
        spin-major, the face projector's contract.
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
    """Convolve a Green tile with one prepared W tile and project on typed raw parents.

        Σ_k = -1/√N_k · fftn( ifftn(G_k) · ifftn(W_q)[:, None, :, None, :] )

    through the k-convolution router: ``common.fft_helpers.make_kconv_klead``
    prepares W, and ``make_kconv_klead_unfold`` convolves the raw-parent Green
    with the typed unfold on its load (nvidia-mathdx on CUDA; the service's
    table composition and the FFTW gw_conv handler on cpu).
    """
    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    from common.fft_helpers import make_kconv_klead, make_kconv_klead_unfold
    from ffi import ffi_dial_key
    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key(),
           bool(merged_x), layout, face_shape, face_band_extent,
           k_unfold_plan)
    if key in _sigma_spatial_kernel_cache:
        return _sigma_spatial_kernel_cache[key]
    from .wavefunction_bundle import (SIGMA_CONV_G7D_SPEC as _G_spec,
                                      V_FFT5D_SPEC as _V_spec)
    ensure_jax_compile_cache()
    kconv = make_kconv_klead(mesh_xy, kgrid, _G_spec, _V_spec,
                             norm='ortho', mult=-1.0 / np.sqrt(float(nk_tot)))
    if k_unfold_plan is None:
        raise ValueError("Sigma spatial kernel requires the typed parent unfold plan.")
    unfold_conv = make_kconv_klead_unfold(mesh_xy, kgrid, k_unfold_plan.unfold_load_tables(),
                                          norm='ortho', mult=-1.0 / np.sqrt(float(nk_tot)))
    project = _make_project_ri_reduce_scatter(
        mesh_xy, merged_x=merged_x, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan)

    @jax.jit
    def prep_w(W_q):
        """The W-only half of the chain — see :class:`SpatialKernel`."""
        return kconv.prep(W_q)

    @partial(jax.jit, donate_argnums=(2,))
    def conv_project(psi_proj_xr, psi_proj_yn, G_parents, W_prep):
        # Raw-parent Green in, spin-major full-k Σ_k out: the typed unfold,
        # spin action and reorder are the convolution's load.
        sigma_k = unfold_conv(G_parents.G, G_parents.transpose, W_prep)
        return project(psi_proj_xr, sigma_k, psi_proj_yn)
    if not _stage_timing_enabled():
        pair = SpatialKernel(prep_w=prep_w, conv_project=conv_project)
        _sigma_spatial_kernel_cache[key] = pair
        return pair
    _conv_j = jax.jit(lambda G_p, W_prep: unfold_conv(G_p.G, G_p.transpose, W_prep),
                      donate_argnums=(0,))
    _project_j = jax.jit(project, donate_argnums=(1,))

    def prep_w_staged(W_q):
        """``sigma.tau.w_prep`` — the ONCE-PER-τ half, timed on its own row."""
        with timing.section(TAU_PHASE_W_PREP) as sec:
            W_prep = prep_w(W_q)
            sec.watch(W_prep)
        return W_prep

    def conv_project_staged(psi_proj_xr, psi_proj_yn, G_k, W_prep):
        """Diagnostic split of the same spatial operation sequence."""
        with timing.section(TAU_PHASE_GW_CONV_FFI) as sec:
            sigma_k = _conv_j(G_k, W_prep)
            sec.watch(sigma_k)
        with timing.section(TAU_PHASE_PROJECT_RS) as sec:
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


def _sigma_spin_pair_stream(*, mesh_xy, kgrid, layout, face_shape,
                            face_band_extent, k_unfold_plan):
    """Σ_c(τ) on raw parents, one spin block of the Green at a time (A4).

        Σ_nm(k̄) = Σ_ab Σ_μν ψ*_{m,a}(μ) [G_ab(τ) ⋆_k W(τ)](k̄; μ, ν) ψ_{n,b}(ν)

    ``G_ab`` is built at full k from the parents unfolded by their typed ψ
    action (the full-k Green's own (a, b) block), convolved with the same
    prepared ``W`` by the k-convolution router, restricted to the parent
    rows and projected on the parents' ``a`` and ``b`` spinor rows.  No
    ``ns²`` Green or Σ_k exists; the live set is a few ``16·N_k·μ²/P``
    blocks.  Arguments match the τ kernel's
    ``(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn, E_A, sel, E_min,
    E_max, E_ref_A, t_node, W_prep)``.
    """
    from common.fft_helpers import make_kconv_klead
    from distrib_la import gemm_plan
    from .greens_function_kernel import (
        _phase_band_interval, _weighted_tau_phases, build_G, pair_stream_layout,
        spin_pair_rows, to_pair_stream_layout, unfold_parent_faces)
    from .wavefunction_bundle import (SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC,
                                      sigma_conv_operand)

    nk_tot = int(np.prod(kgrid))
    _, nb, mu, ns = (int(v) for v in face_shape)
    n_full = int(k_unfold_plan.n_full)
    kconv = make_kconv_klead(mesh_xy, kgrid, SIGMA_CONV_G7D_SPEC, V_FFT5D_SPEC,
                             norm='ortho', mult=-1.0 / np.sqrt(float(nk_tot)))
    pair_layout = pair_stream_layout(n_full=n_full, ns=ns, mu=mu, nb=nb,
                                     layout=layout, mesh=mesh_xy)
    pair_plan = gemm_plan(mesh_xy, m=mu, k=nb, n=mu, nq=n_full,
                          dtype=jnp.complex128, layout=pair_layout,
                          enable_active_range=True)
    project = _make_project_ri_reduce_scatter(
        mesh_xy, merged_x=True, layout=layout, face_shape=(n_full, nb, mu, 1),
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan)
    irr = np.asarray(k_unfold_plan.irr_idx, dtype=np.int32)

    def stream(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
               E_A, sel, E_min, E_max, E_ref_A, t_node, W_prep):
        weights = dict(mask=sel) if sel.dtype == jnp.bool_ else dict(band_weight=sel)
        phases = _weighted_tau_phases(E_A, 1j * t_node, e_ref=E_ref_A,
                                      E_min=E_min, E_max=E_max, **weights)
        phases = jnp.take(phases, jnp.asarray(irr), axis=0)
        lo, hi = _phase_band_interval(phases)
        psi_mun, psi_nmu = to_pair_stream_layout(*unfold_parent_faces(
            k_unfold_plan, psi_coh_xn, psi_coh_yr, layout=layout),
            layout=pair_layout, mesh=mesh_xy)

        def block(index):
            left, right = spin_pair_rows(psi_mun, psi_nmu, index, ns)
            G_ab = build_G(left, right, phases=phases, layout=pair_layout,
                           gemm=pair_plan, band_range=(jnp.minimum(lo, hi), hi))
            S_ab = kconv.apply(sigma_conv_operand(G_ab), W_prep)
            proj_left = jax.lax.dynamic_slice_in_dim(psi_proj_xr, index // ns, 1, axis=2)
            proj_right = jax.lax.dynamic_slice_in_dim(psi_proj_yn, index % ns, 1, axis=1)
            return project(proj_left, S_ab, proj_right)

        first = block(0)
        total, _ = jax.lax.scan(lambda acc, index: (acc + block(index), None),
                                first, jnp.arange(1, ns * ns), unroll=1)
        return total

    return stream


def _get_sigma_kij_kernel(
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int], merged_x: bool = True,
    brackets: tuple[tuple[int, int], ...] | None = _NO_BRACKETS,
    layout: str = "face", face_shape=None, face_band_extent=None,
    energy_windows: bool = False,
    k_unfold_plan=None,
) -> Callable[..., jax.Array]:
    """Build Green functions with band-range masks and contract each bracket against one prepared W."""
    if layout not in ("face", "axis") or face_shape is None or k_unfold_plan is None:
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
    _, nb, mu, ns = face_shape
    g_plan = gemm_plan(mesh_xy, m=mu * ns, k=nb, n=mu * ns,
                       nq=k_unfold_plan.n_parent, dtype=jnp.complex128, layout=layout,
                       enable_active_range=True)
    # A4: for ns > 1 the spin pairs stream, so no ns² Σ_k or convolution
    # transient exists.
    pairs = (None if int(ns) == 1 else _sigma_spin_pair_stream(
        mesh_xy=mesh_xy, kgrid=kgrid, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan))

    def _g_from_selector(xn, yr, E, sel, E_min, E_max, ref, t, band_range=None):
        """Apply boolean identity masks or signed occupation weights without clipping."""
        options = dict(e_ref=ref, layout=layout, gemm=g_plan,
                       k_unfold_plan=k_unfold_plan, band_range=band_range,
                       trim_zero_bands=True, unfold=False)
        options["mask" if sel.dtype == jnp.bool_ else "band_weight"] = sel
        if energy_windows:
            options.update(E_min=E_min, E_max=E_max)
        return build_G_tau(xn, yr, E, 1j * t, **options)

    def _bracketed_face(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                        E_A, mask_A, E_min, E_max, E_ref_A, t_node,
                        W_prep, build_g, conv):
        """Mask each bracket on the last band axis while retaining one Green tile at a time."""
        nb_full = int(mask_A.shape[-1])
        idx = jnp.arange(nb_full)
        endpoints = jnp.asarray(
            [(lo, nb_full if hi is None else hi) for lo, hi in brackets],
            dtype=jnp.int32)

        def one(_, bounds):
            lo, hi = bounds
            in_range = (idx >= lo) & (idx < hi)
            mask_bracket = (mask_A & in_range if mask_A.dtype == jnp.bool_
                           else mask_A * in_range.astype(mask_A.dtype))
            if pairs is not None:
                return None, pairs(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                                   E_A, mask_bracket, E_min, E_max, E_ref_A,
                                   t_node, W_prep)
            G_k = build_g(psi_coh_xn, psi_coh_yr, E_A, mask_bracket,
                         E_min, E_max, E_ref_A, t_node,
                         band_range=(lo, hi))
            projected = conv(psi_proj_xr, psi_proj_yn, G_k, W_prep)
            return None, projected

        # Only the small band-projected outputs acquire a bracket axis.
        # G and its FFT/convolution temporaries stay inside the loop body.
        _, outs = jax.lax.scan(one, None, endpoints, unroll=1)
        sharding = NamedSharding(mesh_xy, P(None, None, 'x', 'y'))
        return jax.tree.map(
            lambda value: jax.lax.with_sharding_constraint(value, sharding),
            outs)

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
            if brackets is None and pairs is not None:
                return pairs(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                             E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_prep)
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

    if brackets is not None or pairs is not None:
        raise NotImplementedError(
            "_get_sigma_kij_kernel(layout='face'): LORRAX_SIGMA_TAU_TIMING "
            "stage-split diagnostic is not ported for bracketed face "
            "carriers or spin-pair streams (ns > 1) — an opt-in profiling "
            "knob, not the production path; set LORRAX_SIGMA_TAU_TIMING=0 "
            "(the default) for that case.")

    build_g = jax.jit(_g_from_selector)

    def _build_g_timed(xn, yr, E, mask, E_min, E_max, ref, t):
        with timing.section(TAU_PHASE_G_BUILD) as sec:
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
                       phase_real, E_ref_B, t_node, active_count=None):
    """Build one W(tau) tile from selected multipole fields.

    ``bounds`` rows are ``(a_gt, a_le, gamma_ge, gamma_gt, gamma_lt,
    gamma_le)``.  Each row selects one pole field; ``phase_real`` chooses
    the accepted near-axis functional ``Re(Omega)`` for that row, otherwise
    the fitted complex pole is used.  The pole axis is never materialized in
    W: the loop carries one ``(q, mu, nu)`` tile. ``active_count`` is a
    replicated dynamic scalar naming the occupied selector prefix; omitted
    counts evaluate every selector row.
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
        0, (pole_indices.shape[0] if active_count is None else active_count),
        _add, jnp.zeros_like(B_poles[0]))


def get_shared_sigma_tau_kernel(
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int],
    brackets: tuple[tuple[int, int], ...] | None = _NO_BRACKETS,
    layout: str = "face", face_shape=None, face_band_extent=None,
    k_unfold_plan=None, _sigma_kij=None,
    w_synthesis=None,
    cache: bool = True,
) -> Callable[..., jax.Array]:
    """Build selected multipole W(tau) tiles for the shared complex Sigma contraction.

    ``w_synthesis`` optionally replaces the resident-pole builder with the
    resolved shared-pole model's W builder. It takes the same eight operands
    as the resident builder (``..., t_node, active_count``), returns the
    complete full-q ``P(None,'x','y')`` tile, and runs outside jit so a
    bounded reader can supply q/K panels between compiled calls; kernels
    built with it are never cached, because the builder owns resources.

    ``cache=False`` builds the kernel without reading or writing the
    process-wide incumbent cache: a compile-only measurement of the
    incumbent route on a run that never dispatches it must not leave its
    control executable behind for a later caller.
    """
    kgrid = tuple(int(x) for x in kgrid)
    if brackets is not None:
        brackets = tuple((int(lo), None if hi is None else int(hi))
                         for lo, hi in brackets)
    from ffi import ffi_dial_key

    key = (id(mesh_xy), kgrid, _stage_timing_enabled(), ffi_dial_key(),
           brackets, layout, face_shape, face_band_extent,
           k_unfold_plan)
    if (w_synthesis is None and _sigma_kij is None and cache
            and key in _sigma_shared_tau_kernel_cache):
        return _sigma_shared_tau_kernel_cache[key]

    ensure_jax_compile_cache()
    q_mu_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))

    sigma_kij = _sigma_kij if _sigma_kij is not None else _get_sigma_kij_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True,
        brackets=brackets, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent,
        k_unfold_plan=k_unfold_plan)

    def finish(kernel):
        # Never publish a kernel whose builder owns resident faces and an open
        # reader, nor one built around a caller's spatial kernel.
        if _sigma_kij is None and w_synthesis is None and cache:
            _sigma_shared_tau_kernel_cache[key] = kernel
        return kernel

    @jax.jit
    def _build(B_poles, Omega_poles, pole_indices, bounds,
               phase_real, E_ref_B, t_node, active_count=None):
        W_t = build_shared_w_tau(
            B_poles, Omega_poles, pole_indices, bounds,
            phase_real, E_ref_B, t_node, active_count)
        return jax.lax.with_sharding_constraint(W_t, q_mu_sharding)

    if w_synthesis is not None:
        _build = w_synthesis

    if not _stage_timing_enabled() and w_synthesis is None:
        @jax.jit
        def _tau(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, B_poles, Omega_poles, pole_indices, bounds,
            phase_real, E_ref_A, E_ref_B, t_node, active_count=None,
        ):
            W_t = _build(B_poles, Omega_poles, pole_indices, bounds,
                         phase_real, E_ref_B, t_node, active_count)
            return sigma_kij(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, E_ref_A, t_node, W_t)

        return finish(_tau)

    profile_stages = _stage_timing_enabled()

    def _tau_staged(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, B_poles, Omega_poles, pole_indices, bounds,
        phase_real, E_ref_A, E_ref_B, t_node, active_count=None,
    ):
        if profile_stages:
            with timing.section(TAU_PHASE_W_PHASE) as sec:
                W_t = _build(B_poles, Omega_poles, pole_indices, bounds,
                             phase_real, E_ref_B, t_node, active_count)
                sec.watch(W_t)
        else:
            # A resident model has a Python storage closure, but its device
            # kernels still dispatch asynchronously. Only the explicit
            # stage profiler needs a host wait between W and G*W.
            W_t = _build(B_poles, Omega_poles, pole_indices, bounds,
                         phase_real, E_ref_B, t_node, active_count)
        return sigma_kij(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_t)

    # A model builder may own resident faces and an open reader for this SC
    # map. Never retain that resource closure in the process-wide jit cache.
    return finish(_tau_staged)

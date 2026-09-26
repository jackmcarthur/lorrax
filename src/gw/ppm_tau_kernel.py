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
``G_k x W_q -> Sigma`` owner, :func:`get_shared_sigma_tau_kernel` is the
resident pole route's τ body, and a shared-pole model's W synthesis rides the
same ``sigma_kij`` through ``gw.mpa.sigma.SynthesisTau``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common.jax_compile_cache import ensure_jax_compile_cache


_sigma_kij_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}
#: Cache of :class:`SpatialKernel` PAIRS (prep_w, conv_project) — not of a
#: single callable, since 2026-08-15: the W-only half is hoistable and the
#: band-bracket loop hoists it.
_sigma_spatial_kernel_cache: dict[tuple[object, ...], "SpatialKernel"] = {}


_sigma_shared_tau_kernel_cache: dict[
    tuple[object, ...], Callable[..., jax.Array]
] = {}


def _make_project_ri_reduce_scatter(
    mesh_xy: Mesh, *, merged_x: bool = True,
    layout: str = "face", face_shape=None, face_band_extent=None,
    k_unfold_plan=None,
) -> Callable[..., jax.Array]:
    """Project Σ on the raw-parent rows (``(n_parent, ns, μ, ns, ν)`` in); the completed frequency sum owns the band unfold."""
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
    return inner


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
    partner_tiles: int = 1,
) -> SpatialKernel:
    """Convolve a Green tile with one prepared W tile and project on typed raw parents.

        Σ_k = -1/√N_k · fftn( ifftn(G_k) · ifftn(W_q)[:, None, :, None, :] )

    through the k-convolution router: ``common.fft_helpers.make_kconv_klead``
    prepares W, and ``make_kconv_klead_unfold`` convolves the raw-parent Green
    with the typed unfold on its load (nvidia-mathdx on CUDA; the service's
    table composition and the FFTW gw_conv handler on cpu).

    The output Σ_k is stored and projected in ``d x d`` spin blocks when the whole
    spin group's output would not fit beside the parent Green
    (``greens_function_kernel.sigma_spin_block``; ``partner_tiles`` counts the
    partner Green a caller holds: 1 for complex weights, 0 when it is read as
    conj(G)): Σ is linear in the block, so the passes sum to the same Σ.
    """
    kgrid = tuple(int(x) for x in kgrid)
    nk_tot = kgrid[0] * kgrid[1] * kgrid[2]
    from common.fft_helpers import make_kconv_klead, make_kconv_klead_unfold
    from ffi import ffi_dial_key
    key = (id(mesh_xy), kgrid, ffi_dial_key(),
           bool(merged_x), layout, face_shape, face_band_extent,
           k_unfold_plan, int(partner_tiles))
    if key in _sigma_spatial_kernel_cache:
        return _sigma_spatial_kernel_cache[key]
    from .wavefunction_bundle import (SIGMA_CONV_G7D_SPEC as _G_spec,
                                      V_FFT5D_SPEC as _V_spec)
    ensure_jax_compile_cache()
    kconv = make_kconv_klead(mesh_xy, kgrid, _G_spec, _V_spec,
                             norm='ortho', mult=-1.0 / np.sqrt(float(nk_tot)))
    if k_unfold_plan is None:
        raise ValueError("Sigma spatial kernel requires the typed parent unfold plan.")
    from .greens_function_kernel import sigma_spin_block
    ns = int(face_shape[3])
    d = sigma_spin_block(n_parent=k_unfold_plan.n_parent, n_rmu=int(face_shape[2]), ns=ns,
                         mesh=mesh_xy, partner_tiles=partner_tiles)
    unfold_conv = make_kconv_klead_unfold(mesh_xy, kgrid, k_unfold_plan.unfold_load_tables(),
                                          store_rows=k_unfold_plan.parent_full_rows,
                                          norm='ortho', mult=-1.0 / np.sqrt(float(nk_tot)),
                                          spin_block=d)
    project = _make_project_ri_reduce_scatter(
        mesh_xy, merged_x=merged_x, layout=layout,
        face_shape=face_shape if d == ns else (*face_shape[:3], d),
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan)
    blocks = [(a0, b0) for a0 in range(0, ns, d) for b0 in range(0, ns, d)]

    def convolve_project(psi_proj_xr, psi_proj_yn, G_parents, W_prep):
        """Σ on the parent rows, one stored output spin block at a time (one pass at d = ns)."""
        total = None
        for a0, b0 in blocks:
            sigma_parent = unfold_conv(G_parents.G, G_parents.transpose, W_prep,
                                       conj_partner=G_parents.conj_partner, a0=a0, b0=b0)
            left = (psi_proj_xr if d == ns
                    else jax.lax.slice_in_dim(psi_proj_xr, a0, a0 + d, axis=2))
            right = (psi_proj_yn if d == ns
                     else jax.lax.slice_in_dim(psi_proj_yn, b0, b0 + d, axis=1))
            part = project(left, sigma_parent, right)
            total = part if total is None else total + part
        return total

    @jax.jit
    def prep_w(W_q):
        """The W-only half of the chain — see :class:`SpatialKernel`."""
        return kconv.prep(W_q)

    @partial(jax.jit, donate_argnums=(2,))
    def conv_project(psi_proj_xr, psi_proj_yn, G_parents, W_prep):
        # Raw-parent Green in, spin-major Σ_k out on the parent rows only: the
        # typed unfold, spin action and reorder are the convolution's load,
        # and the other full-k rows are never stored.
        return convolve_project(psi_proj_xr, psi_proj_yn, G_parents, W_prep)
    pair = SpatialKernel(prep_w=prep_w, conv_project=conv_project)
    _sigma_spatial_kernel_cache[key] = pair
    return pair


#: The band-bracket plan a caller gets when it asks for none: ONE bracket
#: over every band, and NO leading bracket axis on the output.  This is the
#: MPA / shared-multipole shape and it is what ``brackets=None`` means.
_NO_BRACKETS = None


def _get_sigma_kij_kernel(
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int], merged_x: bool = True,
    brackets: tuple[tuple[int, int], ...] | None = _NO_BRACKETS,
    layout: str = "face", face_shape=None, face_band_extent=None,
    energy_windows: bool = False,
    k_unfold_plan=None,
    q_wedge=None,
) -> Callable[..., jax.Array]:
    """Build Green functions with band-range masks and contract each bracket against one prepared W.

    ``q_wedge`` (a ``symmetry_maps.QirrOperator`` of tables): W(τ) arrives on
    the q wedge and is prepared by mathdx mode 9 on the pair-transpose rule,
    its partner tile ``W_pt`` (the tile built from the conjugated residues)
    read on antiunitary rows and the device load tables ``load`` passed as
    arguments; the kernel then takes ``(..., W_q, W_pt, load)``."""
    if layout not in ("face", "axis") or face_shape is None or k_unfold_plan is None:
        raise ValueError("Sigma tau requires canonical face shapes and a typed parent unfold plan.")
    from ffi import ffi_dial_key
    key = (id(mesh_xy), tuple(map(int, kgrid)),
           ffi_dial_key(), bool(merged_x), brackets, layout, face_shape,
           face_band_extent, bool(energy_windows),
           k_unfold_plan, None if q_wedge is None else q_wedge.wedge_key())
    if key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[key]
    from .greens_function_kernel import build_G_tau
    # G, W and projection faces share the run's packed centroid order,
    # as in the static kernels. Pole batches convert only at the store seam.
    spatial = get_sigma_spatial_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=merged_x,
        layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan)

    if q_wedge is None:
        def prep_w(W_q, W_pt=None, load=None):
            return spatial.prep_w(W_q)
    else:
        from common.fft_helpers import make_kfft_klead_unfold
        _pair = dataclasses.replace(q_wedge, values=None, load=None, trs_rule="pair_transpose")
        _door = make_kfft_klead_unfold(mesh_xy, kgrid, _pair.load_tables(mesh_xy), norm="ortho")

        def prep_w(W_q, W_pt=None, load=None):
            return _door(W_q, W_pt, load)

    from distrib_la import gemm_plan
    _, nb, mu, ns = face_shape
    g_plan = gemm_plan(mesh_xy, m=mu * ns, k=nb, n=mu * ns,
                       nq=k_unfold_plan.n_parent, dtype=jnp.complex128, layout=layout,
                       enable_active_range=True)

    def _g_from_selector(xn, yr, E, sel, E_min, E_max, ref, t, band_range=None,
                         real_phases=None):
        """Apply boolean identity masks or signed occupation weights without clipping."""
        # The exponent is -i·t·(E - ref): real when Re t == 0 (``real_phases``).
        options = dict(e_ref=ref, layout=layout, gemm=g_plan,
                       k_unfold_plan=k_unfold_plan, band_range=band_range,
                       trim_zero_bands=True, unfold=False, real_phases=real_phases)
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

    def _kernel_impl(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_q, W_pt=None, load=None,
        real_phases=None,
    ):
        # ONE W preparation per τ, ABOVE the bracket loop.  Explicit, not
        # left to CSE: on the decomposed chain this is ``ifftn(W)``, the
        # only transform in the chain that does not depend on G.
        W_prep = prep_w(W_q, W_pt, load)
        build_g = partial(_g_from_selector, real_phases=real_phases)
        if brackets is None:
            G_k = build_g(psi_coh_xn, psi_coh_yr, E_A, mask_A,
                          E_min, E_max, E_ref_A, t_node)
            return spatial.conv_project(
                psi_proj_xr, psi_proj_yn, G_k, W_prep)
        return _bracketed_face(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_prep,
            build_g, spatial.conv_project)

    # ``real_phases`` (static): the window's host-side statement that its
    # node times are imaginary-axis (see ``build_G_tau``); omitted, the G
    # partner is chosen by a runtime predicate.
    if energy_windows:
        kernel = partial(jax.jit, donate_argnums=(10,),
                         static_argnames=("real_phases",))(_kernel_impl)
    else:
        @partial(jax.jit, donate_argnums=(8,), static_argnames=("real_phases",))
        def kernel(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_q, W_pt=None, load=None,
            real_phases=None,
        ):
            return _kernel_impl(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, None, None, E_ref_A, t_node, W_q, W_pt, load,
                real_phases=real_phases)

    _sigma_kij_kernel_cache[key] = kernel
    return kernel


def _wedge_residues(B_poles):
    """``(B, load)``: residues on the q wedge arrive as a ``QirrOperator``
    carrying the device load tables of the mode-9 prep (jit arguments, never
    program constants); full-zone residues are a plain array and ``None``."""
    from symmetry_maps import QirrOperator
    if isinstance(B_poles, QirrOperator):
        return B_poles.values, B_poles.load
    return B_poles, None


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
    q_wedge=None,
) -> Callable[..., jax.Array]:
    """Build selected multipole W(tau) tiles for the shared complex Sigma contraction.

    The resident pole route's τ body: W(τ) from the resident residues
    (:func:`build_shared_w_tau`) contracted by the cached ``sigma_kij``. A
    shared-pole model supplies its own W synthesis through
    ``gw.mpa.sigma.SynthesisTau`` instead, over the same ``sigma_kij``.
    """
    kgrid = tuple(int(x) for x in kgrid)
    if brackets is not None:
        brackets = tuple((int(lo), None if hi is None else int(hi))
                         for lo, hi in brackets)
    from ffi import ffi_dial_key

    key = (id(mesh_xy), kgrid, ffi_dial_key(),
           brackets, layout, face_shape, face_band_extent,
           k_unfold_plan, None if q_wedge is None else q_wedge.wedge_key())
    if _sigma_kij is None and key in _sigma_shared_tau_kernel_cache:
        return _sigma_shared_tau_kernel_cache[key]

    ensure_jax_compile_cache()
    q_mu_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))

    sigma_kij = _sigma_kij if _sigma_kij is not None else _get_sigma_kij_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=True,
        brackets=brackets, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent,
        k_unfold_plan=k_unfold_plan, q_wedge=q_wedge)
    # On the q wedge W(τ) is read by the pair-transpose rule: an antiunitary
    # row takes the tile built from the conjugated residues (the fitted
    # fields are Hermitian; the time factor is not conjugated).
    partner_needed = False
    if q_wedge is not None:
        _pair = dataclasses.replace(q_wedge, values=None, load=None, trs_rule="pair_transpose")
        partner_needed = bool(np.any(np.asarray(_pair.load_tables(mesh_xy).trs)))

    @jax.jit
    def _build(B_poles, Omega_poles, pole_indices, bounds,
               phase_real, E_ref_B, t_node, active_count=None):
        W_t = build_shared_w_tau(
            B_poles, Omega_poles, pole_indices, bounds,
            phase_real, E_ref_B, t_node, active_count)
        return jax.lax.with_sharding_constraint(W_t, q_mu_sharding)

    @jax.jit
    def _tau(
        psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
        E_A, mask_A, B_poles, Omega_poles, pole_indices, bounds,
        phase_real, E_ref_A, E_ref_B, t_node, active_count=None,
    ):
        B_poles, load = _wedge_residues(B_poles)
        W_t = _build(B_poles, Omega_poles, pole_indices, bounds,
                     phase_real, E_ref_B, t_node, active_count)
        W_pt = (_build(jnp.conj(B_poles), Omega_poles, pole_indices, bounds,
                       phase_real, E_ref_B, t_node, active_count)
                if partner_needed else None)
        return sigma_kij(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_t, W_pt, load)

    # Never publish a kernel built around a caller's spatial kernel.
    if _sigma_kij is None:
        _sigma_shared_tau_kernel_cache[key] = _tau
    return _tau

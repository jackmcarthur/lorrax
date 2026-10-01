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
    k_unfold_plan=None, row_block=None,
) -> Callable[..., jax.Array]:
    """Project Σ on the raw-parent rows (``(n_parent, ns, x-block, ns, ν)`` blocks in); the completed frequency sum owns the band unfold."""
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
        face_band_extent=face_band_extent, row_block=row_block)
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
    ``price``
        ``sigma_spin_block``'s analytic plan (``d``, the live bytes, the new
        bytes of one pass and the room), which the window executable's
        compiled figure is checked against.
    """
    prep_w: Callable[..., jax.Array]
    conv_project: Callable[..., jax.Array]
    price: dict | None = None


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

    The output Σ_k is stored and projected in (ns/d)² x blocks (local left
    centroids on the face projector's slab pieces, every spin) when the whole
    output would not fit beside the parent Green
    (``greens_function_kernel.sigma_spin_block`` sizes ``d``; ``partner_tiles``
    counts the partner Green a caller holds: 1 for complex weights, 0 when it
    is read as conj(G)).  Σ is linear in the block, so the passes sum to the
    same Σ (round-off: the μ sum is split), and each pass reads only its own
    pairs' Green and W.
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
    from .greens_function_kernel import sigma_row_blocks, sigma_spin_block
    ns = int(face_shape[3])
    price = {}
    d = sigma_spin_block(n_parent=k_unfold_plan.n_parent, n_rmu=int(face_shape[2]), ns=ns,
                         n_full=nk_tot, n_band=int(face_shape[1]), mesh=mesh_xy,
                         partner_tiles=partner_tiles, plan=price)
    unfold_conv = make_kconv_klead_unfold(mesh_xy, kgrid, k_unfold_plan.unfold_load_tables(),
                                          store_rows=k_unfold_plan.parent_full_rows,
                                          norm='ortho', mult=-1.0 / np.sqrt(float(nk_tot)))
    blocks = sigma_row_blocks(n_rmu=int(face_shape[2]), ns=ns, d=d, mesh=mesh_xy)
    project = _make_project_ri_reduce_scatter(
        mesh_xy, merged_x=merged_x, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan,
        row_block=None if d == ns else blocks[0][1] * blocks[0][3])

    def convolve_project(psi_proj_xr, psi_proj_yn, G_parents, W_prep):
        """Σ on the parent rows, one stored output x block at a time (one pass at d = ns).

        ψ is oriented once and every block adds into one rank-local band
        partial, so the band-block reduce-scatter runs once per call.  The
        orientation waits on the first block (an optimization barrier), so
        its 1/P faces never sit beside the whole parent Green in the
        convolution's peak; each later block's convolution waits on the
        previous block's projection, so one Σ block is live at a time (the
        price ``sigma_spin_block`` admits)."""
        faces = acc = None
        G, Gt, W = G_parents.G, G_parents.transpose, W_prep
        for rows in blocks:
            if acc is not None:
                acc, G, Gt, W = jax.lax.optimization_barrier((acc, G, Gt, W))
            sigma_parent = unfold_conv(G, Gt, W, conj_partner=G_parents.conj_partner, rows=rows)
            if faces is None:
                sigma_parent, left, right = jax.lax.optimization_barrier(
                    (sigma_parent, psi_proj_xr, psi_proj_yn))
                faces = project.prepare(left, right)
            acc = project.accumulate(faces, sigma_parent, rows=rows, acc=acc)
        return project.finish(acc)

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
    pair = SpatialKernel(prep_w=prep_w, conv_project=conv_project, price=price or None)
    _sigma_spatial_kernel_cache[key] = pair
    return pair


#: The band-bracket plan a caller gets when it asks for none: ONE bracket
#: over every band, and NO leading bracket axis on the output.  This is the
#: MPA / shared-multipole shape and it is what ``brackets=None`` means.
_NO_BRACKETS = None


def sigma_subtile_operands(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn, *, mesh_xy):
    """The Σ τ operands as the sub-tile kernel reads them, placed once per Σ call.

    The Green faces become band-complete rows (``gw.subtile_stream.band_complete``)
    and the projection faces the axis projector's operands
    (``subtile_stream.projection_complete``): one exchange here, none in any τ
    node.  :func:`_sigma_subtile_kernel` states the same placements, so face
    operands are also accepted (they are then placed inside the window).
    """
    from .subtile_stream import band_complete, projection_complete

    @jax.jit
    def place(xn, yr, xr, yn):
        return (*band_complete(xn, yr, mesh_xy), *projection_complete(xr, yn, mesh_xy))
    return place(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn)


def _sigma_subtile_kernel(*, mesh_xy, kgrid, brackets, face_shape, face_band_extent,
                          energy_windows, k_unfold_plan, q_wedge):
    """Σ_k(τ) for W(τ) on the q wedge, row pass by row pass (``gw.subtile_stream``).

        Σ_mn(k) = Σ_{μ ∈ passes} Σ_ν ψ*_m(μ) [G ⋆ W](k)_{μν} ψ_n(ν)

    is linear in the μ rows, so each rank's ``(μ_X, ν_Y)`` tile runs in row
    passes of whole centroid orbits (cuts admissible for both the Green's and
    W's unfold tables, :func:`subtile_stream.orbit_cuts`).  Per pass:

    - W's pass rows on its q parents (a local slice of ``W_q``/``W_pt``)
      enter mathdx mode 9 with the device load cut to the pass
      (``subtile_stream.pass_load``): ``W_prep`` exists for the pass's rows only;
    - the parent Green on the pass's ψ rows: one local GEMM over the active
      bands of the band-complete ψ (``layout='axis'``), no exchange;
    - mode 7 with the Green's unfold tables cut to the pass, and the axis
      band projection of the pass's rows into a rank-local partial;

    and one band-block reduce-scatter per bracket ends the node.  Pass sizes
    come from :data:`runtime.tiles.TILE_BYTES` and the shapes
    (:func:`subtile_stream.plan_rows`).  Brackets run inside each pass, so a
    pass's ``W_prep`` serves every bracket.  Returns ``(kernel, price)``.
    """
    from common.contract_bands import contract_bands_block_reshard
    from common.fft_helpers import make_kconv_klead_unfold, make_kfft_klead_unfold
    from distrib_la import gemm_plan
    from runtime.tiles import TILE_BYTES
    from .greens_function_kernel import build_G_tau, has_antiunitary_rows
    from .subtile_stream import (band_complete, fold_passes, orbit_cuts, pass_load, pass_rows,
                                 pass_tables, plan_rows, projection_complete)

    kgrid = tuple(int(v) for v in kgrid)
    nk = int(np.prod(kgrid))
    _, nb, n_rmu, ns = (int(v) for v in face_shape)
    nb_sig = nb if face_band_extent is None else int(face_band_extent)
    n_parent = int(k_unfold_plan.n_parent)
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    local_rows, nu = n_rmu // px, n_rmu // py
    g_tables = k_unfold_plan.unfold_load_tables()
    pair = dataclasses.replace(q_wedge, values=None, load=None, trs_rule="pair_transpose")
    w_tables = pair.load_tables(mesh_xy)
    if int(w_tables.lsrc.shape[1]) != n_rmu or int(g_tables.lsrc.shape[1]) != n_rmu * ns:
        raise ValueError(
            f"Sigma tau: W's q-wedge tables carry {w_tables.lsrc.shape[1]} left endpoints and "
            f"the Green's {g_tables.lsrc.shape[1]}; the faces carry {n_rmu} centroids x {ns}")
    partner = int(has_antiunitary_rows(k_unfold_plan))
    n_w = int(w_tables.n_parent)
    # One local row's live set: the parent Green and its partner, mode 7's
    # output, W_prep, the pass's slices of W and its partner, the pass's ψ
    # rows (Green and projection) and the projector's ν-contracted rows.
    row_bytes = 16 * ((2 + partner) * n_parent * ns * ns * nu + nk * nu + 2 * n_w * nu
                      + n_parent * ns * (nb + 2 * nb_sig))
    passes = plan_rows(local_rows, row_bytes, lambda: sorted(
        set(orbit_cuts(g_tables.lsrc, px, ns)) & set(orbit_cuts(w_tables.lsrc, px, 1))))
    door9 = make_kfft_klead_unfold(mesh_xy, kgrid, w_tables, norm="ortho")
    mult = -1.0 / np.sqrt(float(nk))
    stages = []
    for x0, xr in passes:
        whole = (x0, xr) == (0, local_rows)
        tables = g_tables if whole else pass_tables(g_tables, x0, xr, px, ns)
        # Not warmed: the plan runs inside the window executable.
        gemm = gemm_plan(mesh_xy, m=px * xr * ns, k=nb, n=n_rmu * ns, nq=n_parent,
                         dtype=jnp.complex128, layout="axis", enable_active_range=True,
                         warmup=False)
        conv = make_kconv_klead_unfold(mesh_xy, kgrid, tables,
                                       store_rows=k_unfold_plan.parent_full_rows,
                                       norm="ortho", mult=mult)
        project = contract_bands_block_reshard(
            mesh_xy, channels="none", layout="axis", face_shape=(n_parent, nb, px * xr, ns),
            right_face_shape=(n_parent, nb, n_rmu, ns), face_band_extent=face_band_extent)
        stages.append((whole, gemm, conv, project))
    finish = stages[0][3].finish
    psi_bytes = 16 * n_parent * ns * (nb + nb_sig) * (local_rows + nu)
    price = dict(d=ns, ns=ns, passes=len(passes), tile=float(TILE_BYTES),
                 new=float(max(xr for _, xr in passes) * row_bytes + psi_bytes))
    from common.gpu_utils import record_stage_price
    record_stage_price(f"Sigma tau, {len(passes)} row pass(es)", price["new"],
                       section="sigma.tau_sweep")
    if jax.process_index() == 0:
        print(f"Sigma tau stream: {len(passes)} row pass(es) of {max(xr for _, xr in passes)} "
              f"local rows ({local_rows} per rank)", flush=True)
    selectors = (None,) if brackets is None else tuple(
        (int(lo), None if hi is None else int(hi)) for lo, hi in brackets)
    partial_spec = NamedSharding(mesh_xy, P(None, ("x", "y")))
    w_spec = P(None, "x", "y")

    def green(psi_p, cols, E, sel, E_min, E_max, ref, t, gemm, band_range):
        """The parent Green on the pass's rows: identity masks or signed weights, no clipping."""
        options = dict(e_ref=ref, layout="axis", gemm=gemm, k_unfold_plan=k_unfold_plan,
                       band_range=band_range, trim_zero_bands=True, unfold=False)
        options["mask" if sel.dtype == jnp.bool_ else "band_weight"] = sel
        if energy_windows:
            options.update(E_min=E_min, E_max=E_max)
        return build_G_tau(psi_p, cols, E, 1j * t, **options)

    def _kernel_impl(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                     E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_q, W_pt=None, load=None):
        if load is None:
            raise ValueError("Sigma tau: W on the q wedge needs its device load tables")
        rows_all, cols_all = band_complete(psi_coh_xn, psi_coh_yr, mesh_xy)
        left_all, right_all = projection_complete(psi_proj_xr, psi_proj_yn, mesh_xy)
        n_mask = int(mask_A.shape[-1])
        idx = jnp.arange(n_mask)
        bracket_masks, bracket_ranges, live = [], [], []
        for bounds in selectors:
            if bounds is None:
                bracket_masks.append(mask_A)
                bracket_ranges.append(None)
                live.append(None)
                continue
            lo, hi = bounds[0], n_mask if bounds[1] is None else bounds[1]
            in_range = (idx >= lo) & (idx < hi)
            masked = (mask_A & in_range if mask_A.dtype == jnp.bool_
                      else mask_A * in_range.astype(mask_A.dtype))
            bracket_masks.append(masked)
            bracket_ranges.append((lo, hi))
            # A bracket with no live band builds an identically zero Green:
            # it is skipped and adds nothing (replicated predicate).
            live.append(jnp.any(masked != 0))
        zero = jax.lax.with_sharding_constraint(
            jnp.zeros((1, px * py * n_parent, nb_sig, nb_sig), jnp.complex128), partial_spec)

        def step(p, x0, xr, accs, operands):
            whole, gemm, conv, project = stages[p]
            W, Wt, rows, cols, left, right = operands
            if not whole:
                W, Wt = (None if a is None else pass_rows(a, mesh_xy, x0, xr, axis=1, spec=w_spec)
                         for a in (W, Wt))
                rows = pass_rows(rows, mesh_xy, x0, xr, axis=2)
                left = pass_rows(left, mesh_xy, x0, xr, axis=3)
            w_prep = door9(W, Wt, load if whole else pass_load(load, mesh_xy, x0, xr))
            faces = (jnp.conj(left), right)
            out = []
            for b, (sel, band_range) in enumerate(zip(bracket_masks, bracket_ranges)):
                if b:
                    # One bracket's Green at a time.
                    prev, w_prep, rows = jax.lax.optimization_barrier((tuple(out), w_prep, rows))
                    out = list(prev)

                def add(acc, sel=sel, band_range=band_range, w_prep=w_prep, rows=rows):
                    G = green(rows, cols, E_A, sel, E_min, E_max, E_ref_A, t_node, gemm,
                              band_range)
                    sigma = conv(G.G, G.transpose, w_prep, conj_partner=G.conj_partner)
                    return project.accumulate(faces, sigma, acc=acc)
                out.append(add(accs[b]) if live[b] is None
                           else jax.lax.cond(live[b], add, lambda a: a, accs[b]))
            return tuple(out)

        accs = fold_passes(passes, step, (zero,) * len(selectors),
                           (W_q, W_pt, rows_all, cols_all, left_all, right_all))
        if brackets is None:
            return finish(accs[0])
        return jax.lax.with_sharding_constraint(
            jnp.stack([finish(acc) for acc in accs]),
            NamedSharding(mesh_xy, P(None, None, "x", "y")))

    if energy_windows:
        kernel = partial(jax.jit, donate_argnums=(10,))(_kernel_impl)
    else:
        @partial(jax.jit, donate_argnums=(8,))
        def kernel(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                   E_A, mask_A, E_ref_A, t_node, W_q, W_pt=None, load=None):
            return _kernel_impl(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                                E_A, mask_A, None, None, E_ref_A, t_node, W_q, W_pt, load)
    return kernel, price


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
    the q wedge, its partner tile ``W_pt`` (the tile built from the conjugated
    residues) read on antiunitary rows and the device load tables ``load``
    passed as arguments; the kernel then takes ``(..., W_q, W_pt, load)`` and
    runs row pass by row pass (:func:`_sigma_subtile_kernel`).  Without it
    (full-zone residues) W is prepared whole by the k-convolution router."""
    if layout not in ("face", "axis") or face_shape is None or k_unfold_plan is None:
        raise ValueError("Sigma tau requires canonical face shapes and a typed parent unfold plan.")
    from ffi import ffi_dial_key
    key = (id(mesh_xy), tuple(map(int, kgrid)),
           ffi_dial_key(), bool(merged_x), brackets, layout, face_shape,
           face_band_extent, bool(energy_windows),
           k_unfold_plan, None if q_wedge is None else q_wedge.wedge_key())
    if key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[key]
    if q_wedge is not None:
        kernel, price = _sigma_subtile_kernel(
            mesh_xy=mesh_xy, kgrid=kgrid, brackets=brackets, face_shape=face_shape,
            face_band_extent=face_band_extent, energy_windows=energy_windows,
            k_unfold_plan=k_unfold_plan, q_wedge=q_wedge)
        _sigma_kij_kernel_cache[key] = kernel
        _SIGMA_PASS_PRICE[id(kernel)] = price
        return kernel
    from .greens_function_kernel import build_G_tau
    # G, W and projection faces share the run's packed centroid order,
    # as in the static kernels. Pole batches convert only at the store seam.
    spatial = get_sigma_spatial_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, merged_x=merged_x,
        layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent, k_unfold_plan=k_unfold_plan)

    def prep_w(W_q, W_pt=None, load=None):
        return spatial.prep_w(W_q)

    from distrib_la import gemm_plan
    _, nb, mu, ns = face_shape
    # Not warmed: the plan runs inside the window executable, and a warm-up would
    # hold full-size dummy C and D tiles at plan time.
    g_plan = gemm_plan(mesh_xy, m=mu * ns, k=nb, n=mu * ns,
                       nq=k_unfold_plan.n_parent, dtype=jnp.complex128, layout=layout,
                       enable_active_range=True, warmup=False)

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
        """Mask each bracket on the last band axis while retaining one Green tile at a time.

        A bracket whose selector has no nonzero band builds an identically zero
        Green (every band weight is an exact zero), so its convolution and
        projection are exact zeros too: it is skipped and contributes zeros.
        The predicate is the replicated selector's, so every rank takes the
        same branch."""
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

            def live(_):
                G_k = build_g(psi_coh_xn, psi_coh_yr, E_A, mask_bracket,
                             E_min, E_max, E_ref_A, t_node,
                             band_range=(lo, hi))
                return conv(psi_proj_xr, psi_proj_yn, G_k, W_prep)

            shape = jax.eval_shape(live, None)
            projected = jax.lax.cond(
                jnp.any(mask_bracket != 0), live,
                lambda _: jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), shape),
                None)
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
    ):
        # ONE W preparation per τ, ABOVE the bracket loop.  Explicit, not
        # left to CSE: on the decomposed chain this is ``ifftn(W)``, the
        # only transform in the chain that does not depend on G.
        W_prep = prep_w(W_q, W_pt, load)
        if brackets is None:
            G_k = _g_from_selector(psi_coh_xn, psi_coh_yr, E_A, mask_A,
                                   E_min, E_max, E_ref_A, t_node)
            return spatial.conv_project(
                psi_proj_xr, psi_proj_yn, G_k, W_prep)
        return _bracketed_face(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_prep,
            _g_from_selector, spatial.conv_project)

    if energy_windows:
        kernel = partial(jax.jit, donate_argnums=(10,))(_kernel_impl)
    else:
        @partial(jax.jit, donate_argnums=(8,))
        def kernel(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_q, W_pt=None, load=None,
        ):
            return _kernel_impl(
                psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                E_A, mask_A, None, None, E_ref_A, t_node, W_q, W_pt, load)

    _sigma_kij_kernel_cache[key] = kernel
    _SIGMA_PASS_PRICE[id(kernel)] = spatial.price
    return kernel


#: ``sigma_spin_block``'s plan per cached Σ kernel (cached for the process,
#: so its id is stable), for the window executable's compiled check.
_SIGMA_PASS_PRICE: dict[int, dict | None] = {}


def sigma_pass_price(kernel):
    """The analytic Σ pass plan behind ``kernel`` (a Σ_kij or spatial kernel), or None."""
    return _SIGMA_PASS_PRICE.get(id(kernel), getattr(kernel, "price", None))


def _wedge_residues(B_poles):
    """``(B, load)``: residues on the q wedge arrive in a carrier with ``values`` and
    the device ``load`` tables their transport reads (jit arguments, never program
    constants): the ISDF ``QirrOperator`` (the mode-9 prep's tables) or the plane-wave
    ``gw.plane_wave_pipeline.SphereResidues`` (the pair convolution's tables).
    Full-zone residues are a plain array and ``None``."""
    if hasattr(B_poles, "values") and hasattr(B_poles, "load"):
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

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
them. :func:`_sigma_subtile_kernel` owns the bounded scalar spatial
contraction; :func:`get_shared_sigma_tau_kernel` is the resident pole
route's τ body, and a tangential model's W synthesis rides the same
``sigma_kij`` through ``gw.mpa.sigma.SynthesisTau``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common.jax_compile_cache import ensure_jax_compile_cache
from common.log import get_logger

log = get_logger(__name__)



_sigma_kij_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}


_sigma_shared_tau_kernel_cache: dict[Callable, Callable] = {}

#: The Σ Green kconv call's placed load tables per (mesh, parent plan); bounded.
#: Plans are run objects, stable across SC maps, so a run places them once.
_SIGMA_KCONV_TABLES: dict = {}


def sigma_kconv_tables(mesh_xy: Mesh, k_unfold_plan):
    """The mode-7 Green kconv call's load tables on the devices, placed once per mesh and plan.

    ``symmetry_maps.device_load_tables`` of ``k_unfold_plan.unfold_load_tables()``.
    The q-wedge Σ kernel (:func:`_sigma_subtile_kernel`) reads them as its
    ``g_load`` argument and cuts them to each row pass on the device
    (``subtile_stream.window_load``), so no Σ program holds table constants:
    baked, every row pass's kconv call carried its own copy of the row/phase tables
    as HLO literals (21 passes at the Fe 20^3 P64-local tile: compile
    24 -> 68.6 s).
    """
    key = (mesh_xy, k_unfold_plan)
    hit = _SIGMA_KCONV_TABLES.get(key)
    if hit is None:
        from symmetry_maps import device_load_tables
        placed = device_load_tables(k_unfold_plan.unfold_load_tables(), mesh_xy)
        while len(_SIGMA_KCONV_TABLES) >= 2:
            _SIGMA_KCONV_TABLES.pop(next(iter(_SIGMA_KCONV_TABLES)))
        # The plan rides along so its id in the key cannot be reused.
        hit = _SIGMA_KCONV_TABLES[key] = (k_unfold_plan, placed)
    return hit[1]


#: The band-bracket plan a caller gets when it asks for none: ONE bracket
#: over every band, and NO leading bracket axis on the output.  This is the
#: MPA / shared-multipole shape and it is what ``brackets=None`` means.
_NO_BRACKETS = None


def bracket_selectors(selector, selectors):
    """Each band bracket's Green selector, band range and liveness: ``(masks, ranges, live)``.

    ``selectors``: ``(None,)`` (one bracket over every band: the selector
    itself, no range, no predicate) or ``(lo, hi)`` rows, ``hi`` None for the
    last band.  ``selector`` is a window's band mask or signed weight with the
    bands on its last axis; each bracket keeps it on ``[lo, hi)``.  A bracket
    with no live band builds an identically zero Green, so ``live`` (a
    replicated predicate) lets the caller skip it.  One owner for the scalar
    Σ τ kernel and the four-current sector node (``gw.mpa.sector_sigma``).
    """
    n_mask = int(selector.shape[-1])
    idx = jnp.arange(n_mask)
    masks, ranges, live = [], [], []
    for bounds in selectors:
        if bounds is None:
            masks.append(selector)
            ranges.append(None)
            live.append(None)
            continue
        lo, hi = bounds[0], n_mask if bounds[1] is None else bounds[1]
        in_range = (idx >= lo) & (idx < hi)
        masked = (selector & in_range if selector.dtype == jnp.bool_
                  else selector * in_range.astype(selector.dtype))
        masks.append(masked)
        ranges.append((lo, hi))
        live.append(jnp.any(masked != 0))
    return masks, ranges, live



def _sigma_subtile_kernel(*, mesh_xy, kgrid, brackets, face_shape, face_band_extent,
                          energy_windows, k_unfold_plan, q_wedge, static=False,
                          sum_layout="axis"):
    """Σ_k(τ) for W(τ) on the q wedge, row pass by row pass (``gw.subtile_stream``).

        Σ_mn(k) = Σ_{μ ∈ passes} Σ_ν ψ*_m(μ) [G ⋆ W](k)_{μν} ψ_n(ν)

    is linear in the μ rows, so each rank's ``(μ_X, ν_Y)`` tile runs in row
    passes of whole centroid orbits (cuts admissible for both the Green's and
    W's unfold tables, :func:`subtile_stream.orbit_cuts`), every pass one
    window of ``R`` rows (:func:`subtile_stream.plan_windows`) and the passes
    one ``lax.scan`` (:func:`subtile_stream.scan_passes`), so the program does
    not grow with the pass count.  Per pass:

    - W's window rows on its q parents (a traced slice of ``W_q``/``W_pt``)
      enter mathdx mode 9 with the device load cut to the window
      (``subtile_stream.window_load``): ``W_prep`` exists for the window only;
    - the parent Green on the window's ψ rows: one local GEMM over the active
      bands of the band-complete ψ (``layout='axis'``), no exchange;
    - mode 7 with the Green's device load ``g_load`` (:func:`sigma_kconv_tables`)
      cut to the window, its sources -1 outside the pass's live rows, and the
      axis band projection of the window's rows (zero outside the live rows)
      into a rank-local partial;

    and one band-block reduce-scatter per bracket ends the node.  Both
    k-convolutions read their tables as operands, so the program holds no
    table constants.  ``R`` comes from :data:`runtime.tiles.TILE_BYTES`, the
    shapes and the orbit block.  Brackets run inside each pass, so a pass's
    ``W_prep`` serves every bracket.  Returns ``(kernel, price)``; the kernel
    takes ``(..., W_q, W_pt, load, g_load)``.

    ``static``: the same node at τ = 0 for a Hermitian static interaction (V,
    W(0), W − V; ``gw.cohsex_sigma``'s Σ_x, SX and COH).  Its antiunitary rows
    read the interaction by the wedge's conj rule (no partner tile, ``W_pt``
    None) and the Green's real weights give its partner as conj(G) on the
    mode-7 load (no partner GEMM).
    """
    from common.contract_bands import contract_bands_block_reshard, face_row_blocks
    from common.fft_helpers import make_kconv_klead_unfold, make_kfft_klead_unfold, x_block_rows
    from distrib_la import gemm_plan
    from runtime.tiles import TILE_BYTES
    from .greens_function_kernel import build_G_tau, has_antiunitary_rows, sigma_kconv_scratch
    from .subtile_stream import (GREEN_ROWS_SPEC, orbit_cuts, plan_windows, projection_complete,
                                 scan_passes, window_green_rows, window_load, window_rows,
                                 window_tables, _window_table)

    kgrid = tuple(int(v) for v in kgrid)
    nk = int(np.prod(kgrid))
    _, nb, n_rmu, ns = (int(v) for v in face_shape)
    nb_sig = nb if face_band_extent is None else int(face_band_extent)
    n_parent = int(k_unfold_plan.n_parent)
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    local_rows, nu = n_rmu // px, n_rmu // py
    g_tables = k_unfold_plan.unfold_load_tables()
    pair = dataclasses.replace(q_wedge, values=None, load=None,
                               trs_rule="conj" if static else "pair_transpose")
    w_tables = pair.load_tables(mesh_xy)
    if int(w_tables.lsrc.shape[1]) != n_rmu or int(g_tables.lsrc.shape[1]) != n_rmu * ns:
        raise ValueError(
            f"Sigma tau: W's q-wedge tables carry {w_tables.lsrc.shape[1]} left endpoints and "
            f"the Green's {g_tables.lsrc.shape[1]}; the faces carry {n_rmu} centroids x {ns}")
    partner = int(has_antiunitary_rows(k_unfold_plan) and not static)
    n_w = int(w_tables.n_parent)
    # One local row's live set: the parent Green and its partner, mode 7's
    # output, W_prep, the pass's slices of W and its partner, the pass's ψ
    # rows (Green and projection) and the projector's ν-contracted rows.
    row_bytes = 16 * ((2 + partner) * n_parent * ns * ns * nu + nk * nu
                      + (1 + int(not static)) * n_w * nu + n_parent * ns * (nb + 2 * nb_sig))
    R, windows = plan_windows(local_rows, row_bytes, lambda: sorted(
        set(orbit_cuts(g_tables.lsrc, px, ns)) & set(orbit_cuts(w_tables.lsrc, px, 1))))
    whole = len(windows) == 1
    kconv9 = make_kfft_klead_unfold(mesh_xy, kgrid, w_tables, norm="ortho")
    mult = -1.0 / np.sqrt(float(nk))
    # One window shape for every pass: the host tables give the k-convolution
    # its shapes, and each pass reads g_load's cut (subtile_stream.window_load).
    tables = g_tables if whole else window_tables(g_tables, R, px, ns)
    # Not warmed: the plan runs inside the window executable.
    gemm = gemm_plan(mesh_xy, m=px * R * ns, k=nb, n=n_rmu * ns, nq=n_parent,
                     dtype=jnp.complex128, layout=sum_layout, enable_active_range=True,
                     warmup=False)
    conv = make_kconv_klead_unfold(mesh_xy, kgrid, tables,
                                   store_rows=k_unfold_plan.parent_full_rows,
                                   norm="ortho", mult=mult)
    # An indivisible orbit can exceed the input-row target. Keep its Green
    # once, but cap the stored output using the whole-spin row pieces the
    # shared projector owns. CPU/reference FFTs crop only after forming the
    # full output, so subdivision there would repeat work without this saving.
    output_row_bytes = 16 * n_parent * ns * ns * nu
    output_blocks = (None,)
    if conv.stores_rows_in_kernel and R * output_row_bytes > TILE_BYTES:
        divisors = (d for d in range(ns, 0, -1) if ns % d == 0)
        for d in divisors:
            candidate = face_row_blocks(R, py, (ns // d) ** 2)
            output_blocks = candidate
            if max(b[1] * b[3] for b in candidate) * output_row_bytes <= TILE_BYTES:
                break
        if len(output_blocks) == 1:
            output_blocks = (None,)
    stored_rows = R if output_blocks == (None,) else max(b[1] * b[3] for b in output_blocks)
    project = contract_bands_block_reshard(
        mesh_xy, channels="none", layout="axis", face_shape=(n_parent, nb, px * R, ns),
        right_face_shape=(n_parent, nb, n_rmu, ns), face_band_extent=face_band_extent,
        row_block=None if output_blocks == (None,) else stored_rows)
    finish = project.finish
    # ψ rows and columns, the projection faces, and the Green's right operand (one copy of the
    # columns, formed once per node).
    sum_bytes = nb * (local_rows + 2 * nu) if sum_layout == "axis" else 2 * nb * n_rmu / (px * py)
    psi_bytes = 16 * n_parent * ns * (sum_bytes + nb_sig * (local_rows + nu))
    scratch = sigma_kconv_scratch(kgrid=kgrid, ns=ns, rows=stored_rows, n_right=nu, mesh=mesh_xy)
    price = dict(d=ns, ns=ns, passes=len(windows), tile=float(TILE_BYTES), scratch=int(scratch),
                 output_blocks=len(output_blocks), stored_rows=stored_rows,
                 new=float(R * row_bytes - (R - stored_rows) * output_row_bytes + psi_bytes + scratch))
    from lxkit import memory
    # A static node is priced in its caller's section (Σ exchange, static COHSEX).
    # A dynamic sweep books its selected full-window price at operand placement.
    label = "Sigma static" if static else "Sigma tau"
    if static:
        memory.book(memory.priced(price["new"],
                                  f"{label}, {len(windows)} row pass(es)", increment=True))
    log.debug(f"{label} stream: {len(windows)} row pass(es) of {R} local rows "
          f"({local_rows} per rank)")
    selectors = (None,) if brackets is None else tuple(
        (int(lo), None if hi is None else int(hi)) for lo, hi in brackets)
    partial_spec = NamedSharding(mesh_xy, P(None, ("x", "y")))
    w_spec = P(None, "x", "y")

    def green(psi_p, g_right, E, sel, E_min, E_max, ref, t, gemm, band_range):
        """The parent Green on the pass's rows: identity masks or signed weights, no clipping.

        An antiunitary row reads the conjugate-face partner, built by its own
        local GEMM at every node (``real_weights=False``): no device predicate
        stops the node loop, and the partner is the one the face route's
        single exchange formed.  A static node's weights are real, so its
        partner is conj(G), read on the mode-7 load. ``g_right`` is the
        preformed axis right operand or the borrowed canonical right face.
        A face build owns its conjugation/layout workspace, rather than
        retaining a separate full face beside every convolution. For axis,
        every τ node, row pass and bracket reads the same preformed operand,
        and the partner is ``conj(A·diag(w*)·B)`` from the same two operands.
        """
        options = dict(e_ref=ref, layout=sum_layout, gemm=gemm, k_unfold_plan=k_unfold_plan,
                       band_range=band_range, trim_zero_bands=True, unfold=False,
                       real_weights=bool(static), right=g_right if sum_layout == "axis" else None)
        options["mask" if sel.dtype == jnp.bool_ else "band_weight"] = sel
        if energy_windows:
            options.update(E_min=E_min, E_max=E_max)
        return build_G_tau(psi_p, g_right if sum_layout == "face" else None, E, 1j * t, **options)

    def _kernel_impl(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                     E_A, mask_A, E_min, E_max, E_ref_A, t_node, W_q, W_pt=None, load=None,
                     g_load=None):
        if load is None or g_load is None:
            raise ValueError("Sigma tau: W on the q wedge and the Green kconv call need their device "
                             "load tables (load, g_load = sigma_kconv_tables)")
        # Axis operands arrive preformed; face operands remain the borrowed
        # canonical arrays, including the right face's separate spin axis.
        rows_spec = GREEN_ROWS_SPEC if sum_layout == "axis" else P(None, None, "x", "y")
        right_spec = P(None, None, "y") if sum_layout == "axis" else P(None, "x", None, "y")
        rows_all = jax.lax.with_sharding_constraint(psi_coh_xn, NamedSharding(mesh_xy, rows_spec))
        g_right = jax.lax.with_sharding_constraint(psi_coh_yr, NamedSharding(mesh_xy, right_spec))
        left_all, right_all = projection_complete(psi_proj_xr, psi_proj_yn, mesh_xy)
        bracket_masks, bracket_ranges, live = bracket_selectors(mask_A, selectors)
        zero = jax.lax.with_sharding_constraint(
            jnp.zeros((1, px * py * n_parent, nb_sig, nb_sig), jnp.complex128), partial_spec)

        W, Wt, rows, left = W_q, W_pt, rows_all, left_all
        faces_right = right_all
        output_bounds = (None if output_blocks == (None,) else _window_table(tuple(
            (b[0], b[0] + (b[3] - 1) * b[2] + b[1]) for b in output_blocks)))

        def one_pass(accs, W, Wt, rows, left, w_load, g_pass, rows_live=None):
            # ``rows_live``: a padded window's live rows [lo, hi); the
            # k-convolutions skip the rest and return them zero.
            w_prep = kconv9(W, Wt, w_load, live=rows_live)
            faces = (jnp.conj(left), faces_right)
            out = []
            for b, (sel, band_range) in enumerate(zip(bracket_masks, bracket_ranges)):
                if b:
                    # One bracket's Green at a time.
                    prev, w_prep, rows = jax.lax.optimization_barrier((tuple(out), w_prep, rows))
                    out = list(prev)

                def add(acc, sel=sel, band_range=band_range, w_prep=w_prep, rows=rows):
                    G = green(rows, g_right, E_A, sel, E_min, E_max, E_ref_A, t_node, gemm,
                              band_range)
                    if output_blocks == (None,):
                        sigma = conv(G.G, G.transpose, w_prep, conj_partner=G.conj_partner,
                                     load=g_pass, live=rows_live)
                        return project.accumulate(faces, sigma, acc=acc)

                    def body(total, j):
                        # The runtime live operand binds each convolution to
                        # its scan iteration, preventing loop-invariant lifting
                        # of all stored outputs beside the whole orbit Green.
                        bounds = jax.lax.optimization_barrier(jax.lax.dynamic_index_in_dim(
                            output_bounds, j, 0, keepdims=False))
                        outer = (jnp.asarray((0, R), jnp.int32)
                                 if rows_live is None else rows_live)
                        lo = jnp.maximum(bounds[0], outer[0])
                        live_rows = jnp.stack((lo, jnp.maximum(lo, jnp.minimum(bounds[1], outer[1]))))

                        def block(total, block_rows):
                            # The window's live interval is in source rows;
                            # mode 7 masks positions of the selected output.
                            # x blocks list those source rows monotonically.
                            source_rows = jnp.asarray(x_block_rows(block_rows))
                            stored_live = jnp.stack((jnp.sum(source_rows < live_rows[0]),
                                                     jnp.sum(source_rows < live_rows[1]))).astype(jnp.int32)
                            sigma = conv(G.G, G.transpose, w_prep,
                                conj_partner=G.conj_partner, rows=block_rows,
                                load=g_pass, live=stored_live)
                            return project.accumulate(faces, sigma, rows=block_rows, acc=total)
                        branches = tuple(partial(block, block_rows=b) for b in output_blocks)
                        return jax.lax.switch(j, branches, total), None
                    return jax.lax.scan(body, acc, jnp.arange(len(output_blocks), dtype=jnp.int32),
                                        unroll=1)[0]
                out.append(add(accs[b]) if live[b] is None
                           else jax.lax.cond(live[b], add, lambda a: a, accs[b]))
            return tuple(out)

        accs = (zero,) * len(selectors)
        def green_window(s=None):
            if sum_layout == "axis":
                return window_green_rows(rows, mesh_xy, s, None if s is None else R)
            return rows if s is None else window_rows(rows, mesh_xy, s, R, axis=2, spec=rows_spec)

        if whole:
            accs = one_pass(accs, W, Wt, green_window(), left, load, g_load)
        else:
            def step(s, lo, hi, accs):
                # The pass's window: W's and the ψ rows, the projection rows
                # zeroed outside the live rows, and both tables cut there.
                Wp, Wtp = (None if a is None else window_rows(a, mesh_xy, s, R, axis=1, spec=w_spec)
                           for a in (W, Wt))
                return one_pass(accs, Wp, Wtp, green_window(s),
                                window_rows(left, mesh_xy, s, R, axis=3, live=(lo, hi)),
                                window_load(load, mesh_xy, s, lo, hi, R),
                                window_load(g_load, mesh_xy, s, lo, hi, R, ns),
                                jnp.stack([lo, hi]).astype(jnp.int32))
            accs = scan_passes(windows, step, accs)
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
                   E_A, mask_A, E_ref_A, t_node, W_q, W_pt=None, load=None, g_load=None):
            return _kernel_impl(psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
                                E_A, mask_A, None, None, E_ref_A, t_node, W_q, W_pt, load,
                                g_load)
    return kernel, price


def _get_sigma_kij_kernel(
    *, mesh_xy: Mesh, kgrid: tuple[int, int, int],
    brackets: tuple[tuple[int, int], ...] | None = _NO_BRACKETS,
    layout: str = "face", face_shape=None, face_band_extent=None,
    energy_windows: bool = False,
    k_unfold_plan=None,
    q_wedge=None,
    static: bool = False,
    sum_layout: str = "axis",
) -> Callable[..., jax.Array]:
    """Cache the bounded scalar spatial node with runtime Green/W load tables.

    ``q_wedge`` names the interaction's typed load, including the identity
    table for a full-zone source. The public scalar executor creates that
    identity metadata without changing the source's storage or read. W and
    its complex antiunitary partner remain runtime operands; every row's
    W preparation serves all band brackets.
    """
    if layout not in ("face", "axis") or face_shape is None or k_unfold_plan is None:
        raise ValueError("Sigma tau requires canonical face shapes and a typed parent unfold plan.")
    if q_wedge is None:
        raise ValueError("Scalar Sigma node requires typed q metadata; full-zone sources "
                         "use the identity table in the public Sigma executor.")
    from ffi import ffi_dial_key
    from common.fft_helpers import kconv_backend
    key = (id(mesh_xy), tuple(map(int, kgrid)),
           ffi_dial_key(), kconv_backend(mesh_xy, kgrid), brackets, layout, face_shape,
           face_band_extent, bool(energy_windows),
           k_unfold_plan, q_wedge.wedge_key(), bool(static), sum_layout)
    if key in _sigma_kij_kernel_cache:
        return _sigma_kij_kernel_cache[key]
    kernel, price = _sigma_subtile_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid, brackets=brackets, face_shape=face_shape,
        face_band_extent=face_band_extent, energy_windows=energy_windows,
        k_unfold_plan=k_unfold_plan, q_wedge=q_wedge, static=static, sum_layout=sum_layout)
    kernel.price = price
    _sigma_kij_kernel_cache[key] = kernel
    return kernel


def sigma_pass_price(kernel):
    """The analytic Σ pass plan behind ``kernel`` (a Σ_kij or spatial kernel), or None."""
    return getattr(kernel, "price", None)


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
        # The counter-indexed rows are read once, behind a barrier: remat may
        # otherwise clone one after the counter's in-place increment (R82).
        pole, b, use_real = jax.lax.optimization_barrier(tuple(
            jax.lax.dynamic_index_in_dim(a, index, axis=0, keepdims=False)
            for a in (pole_indices, bounds, phase_real)))
        omega = jax.lax.dynamic_index_in_dim(
            Omega_poles, pole, axis=0, keepdims=False)
        residue = jax.lax.dynamic_index_in_dim(
            B_poles, pole, axis=0, keepdims=False)
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
    sum_layout: str = "axis",
) -> Callable[..., jax.Array]:
    """Build selected multipole W(tau) tiles for the shared complex Sigma contraction.

    The resident pole route's τ body: W(τ) from the resident residues
    (:func:`build_shared_w_tau`) contracted by the cached ``sigma_kij``. A
    tangential model supplies its own W synthesis through
    ``gw.mpa.sigma.SynthesisTau`` instead, over the same ``sigma_kij``.
    Scalar ISDF calls supply typed ``q_wedge`` metadata; a custom spatial
    ``_sigma_kij`` (the plane-wave owner) retains its own load contract.
    """
    kgrid = tuple(int(x) for x in kgrid)
    if brackets is not None:
        brackets = tuple((int(lo), None if hi is None else int(hi))
                         for lo, hi in brackets)
    ensure_jax_compile_cache()
    sigma_kij = _sigma_kij if _sigma_kij is not None else _get_sigma_kij_kernel(
        mesh_xy=mesh_xy, kgrid=kgrid,
        brackets=brackets, layout=layout, face_shape=face_shape,
        face_band_extent=face_band_extent,
        k_unfold_plan=k_unfold_plan, q_wedge=q_wedge, sum_layout=sum_layout)
    if _sigma_kij is None and sigma_kij in _sigma_shared_tau_kernel_cache:
        return _sigma_shared_tau_kernel_cache[sigma_kij]
    q_mu_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
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
        phase_real, E_ref_A, E_ref_B, g_load, t_node, active_count=None,
    ):
        # ``g_load``: the Green kconv call's placed tables on the q wedge
        # (:func:`sigma_kconv_tables`), ``None`` otherwise.
        B_poles, load = _wedge_residues(B_poles)
        W_t = _build(B_poles, Omega_poles, pole_indices, bounds,
                     phase_real, E_ref_B, t_node, active_count)
        W_pt = (_build(jnp.conj(B_poles), Omega_poles, pole_indices, bounds,
                       phase_real, E_ref_B, t_node, active_count)
                if partner_needed else None)
        return sigma_kij(
            psi_coh_xn, psi_coh_yr, psi_proj_xr, psi_proj_yn,
            E_A, mask_A, E_ref_A, t_node, W_t, W_pt, load,
            *(() if g_load is None else (g_load,)))

    _tau.price = sigma_pass_price(sigma_kij)
    # The resolved spatial program owns all geometry/provider identity.
    # Never publish a kernel built around a caller's spatial kernel.
    if _sigma_kij is None:
        _sigma_shared_tau_kernel_cache[sigma_kij] = _tau
    return _tau

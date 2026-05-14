import gc
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

from . import Meta
from . import timing
from .fft_helpers import (
    make_sharded_ifftn_3d,
)


def load_kpoint_fftbox(wfn, sym, meta, k_idx, nb):
    """Load a single k-point's wavefunction into the FFT box on GPU.

    Returns jax array of shape (nb, nspinor, nx, ny, nz), ~0.55 GiB for 12x12.

    Migrated to :class:`file_io.wfn_loader.WfnLoader` + ``to_box``.  ``sym``
    is unused (the loader's full-BZ unfold is internal to ``load(k=[k_idx])``);
    kept in the signature for caller-API back-compat.
    """
    del sym
    from common.wfn_transforms import to_box

    loader = wfn  # reuse top-level WfnLoader; do NOT re-open (would re-slurp coeffs)
    psi = loader.load(bands=(0, int(nb)), k=[int(k_idx)],
                      sharding=None)               # (1, nb, nspinor_wfn, ngkmax)
    if int(meta.nspinor) > int(loader.nspinor):
        ns_pad = int(meta.nspinor) - int(loader.nspinor)
        psi = jnp.pad(psi, ((0, 0), (0, 0), (0, ns_pad), (0, 0)))
    # Single-rank legacy helper: build the 1×1 trivial mesh inline so
    # ``to_box`` runs through the same mesh-required code path as the
    # multi-rank callers.
    _mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                  axis_names=('x', 'y'))
    psi_box = to_box(psi, loader.box_index(k=[int(k_idx)]),
                      tuple(int(s) for s in meta.fft_grid),
                      mesh=_mesh)
    return psi_box[0]                                # strip the singleton k-axis


def get_enk_bandrange(wfn, sym, bandrange, sigma_bandrange, nspinor=2):
    """Return band energies and per-band weights for a given band window.

    Args:
        wfn: WFNReader providing energies and Fermi level
        sym: SymMaps with mappings between irreducible and full k sets
        bandrange: tuple[int,int] inclusive-exclusive (start, end) bands to extract
        sigma_bandrange: tuple[int,int] band window used to compute weighting
        nspinor: Number of spinor components (2 for Pauli, 4 for bispinor)

    Returns:
        enk: jax.Array of shape (nk_full, nb)
        weights: jax.Array of shape (nk_full, nb * nspinor) with simple val/cond weights

    ────────────────────────────────────────────────────────────────────────
    NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL.
    ────────────────────────────────────────────────────────────────────────
    Everything in this function operates on tiny host-side arrays
    (nk × nb ~ a few thousand doubles).  Using ``jnp`` would force each
    reduction/where/repeat to be dispatched as its own pjit at trace time
    — ~16 standalone pjit compilations per run, for zero runtime benefit
    (the arithmetic is ms-scale on host).  Rewriting to ``jnp`` reverses
    a deliberate compile-cache trim (commit 31b5961, 2026-04-18).

    Only cast to ``jax.Array`` at return so the caller gets the pytree
    type it expects.  Do NOT "fix" this back to ``jnp``.
    ────────────────────────────────────────────────────────────────────────
    """
    # Energies are stored on irreducible k; expand to full k using mapping.
    band_lo = int(bandrange[0])
    band_hi = int(bandrange[1])
    nb = band_hi - band_lo
    irk_to_k = np.asarray(sym.irr_idx_k)
    # Handle file-short case (band_hi > nbnd in WFN.h5): read what's
    # available, sentinel-fill the rest so f_n=step(E_F-e)=0 for padded
    # bands.  Using a finite "max(real e) + 1 Ry" instead of ∞ keeps
    # PPM resolvent arithmetic 1/(ω - e + iη) safe under fp warnings.
    nb_in_file = int(wfn.energies.shape[2])
    band_hi_eff = min(band_hi, nb_in_file)
    en_irk = np.asarray(
        wfn.energies[0, :, band_lo:band_hi_eff], dtype=np.float64)
    if band_hi_eff < band_hi:
        sentinel = float(np.asarray(wfn.energies[0, :, :]).max()) + 1.0
        pad = np.full((en_irk.shape[0], band_hi - band_hi_eff), sentinel,
                      dtype=np.float64)
        en_irk = np.concatenate([en_irk, pad], axis=1)
    enk = en_irk[irk_to_k, :]                                   # (nk_full, nb)

    # Weighting heuristic: 1/sqrt(Ec - E) for conduction, 1/sqrt(E - Ev) for
    # valence, capped and normalized; sigma band window set to exactly 1.
    sigma_lo = int(sigma_bandrange[0])
    sigma_hi = int(sigma_bandrange[1])
    enk_sigma_lo = max(sigma_lo - band_lo, 0)
    enk_sigma_hi = min(sigma_hi - band_lo, nb)
    energies_full = np.asarray(wfn.energies[0, :, :], dtype=np.float64)[irk_to_k, :]
    energies_sigma = energies_full[:, sigma_lo:sigma_hi]
    E_min = float(energies_sigma.min())
    E_max = float(energies_sigma.max())
    efermi = float(wfn.efermi)

    val_weights = 1.0 / np.sqrt(np.maximum(E_max - enk, 1e-12))
    cond_weights = 1.0 / np.sqrt(np.maximum(enk - E_min, 1e-12))
    weights = np.where(enk <= efermi, val_weights, cond_weights)
    wmax = weights.max()
    if wmax > 0:
        weights = weights / wmax
    weights[:, enk_sigma_lo:enk_sigma_hi] = 1.0
    weights = np.repeat(weights, repeats=nspinor, axis=1)

    return jnp.asarray(enk), jnp.asarray(weights)


from .bispinor_init import get_small_psi_component  # noqa: F401 — re-export for callers


def read_Gvecs_to_devices(
    wfn, sym, bandrange, meta: Meta, bispinor: bool, mesh_xy: Mesh,
    k_range: tuple[int, int] | None = None,
):
    """G-space wfns on a 2-D mesh, band-sharded, scattered to FFT box.

    Returns ``(global_psi_Gtot, nb_logical)`` where ``global_psi_Gtot``
    has shape ``(nk, nb_padded, nspinor, nx, ny, nz)`` sharded
    ``P(None, ('x','y'), None, None, None, None)`` — same contract as
    the legacy implementation (kept available as ``read_Gvecs_to_devices_legacy``
    for one release if a caller needs to bisect).

    The migrated body is thin: :class:`file_io.wfn_loader.WfnLoader`
    + :func:`common.wfn_transforms.to_box`.  Symmetry unfold, τ-phase,
    TR conjugation, spinor rotation, band-axis padding/sharding, and
    the bispinor lift all happen inside ``WfnLoader.load``.  ``sym``
    is unused (the loader builds its own SymMaps lazily); kept in
    the signature so existing callers don't have to change.

    Memory note: this function still materialises the FFT-box
    representation for caller back-compat.  The g_flat path
    (:meth:`WfnLoader.load` directly) is ~6-11% the size of the FFT
    box — when the PsiGStore → PsiGCache rewrite lands (next P4c
    sub-step), the GW driver hot loop will consume g_flat and call
    ``to_rchunk`` per r-chunk instead of holding the FFT box.
    """
    del sym
    from common.wfn_transforms import to_box

    b_lo, b_hi = int(bandrange[0]), int(bandrange[1])
    nb_logical = b_hi - b_lo
    if k_range is None:
        k = "full_bz"
    else:
        k = list(range(int(k_range[0]), int(k_range[1])))

    sharding = P(None, ("x", "y"), None, None)

    loader = wfn  # reuse top-level WfnLoader
    psi_G_flat = loader.load(
        bands=(b_lo, b_hi), k=k, sharding=sharding,
        bispinor=bool(bispinor),
    )
    ns_after_lift = 4 if bispinor else int(loader.nspinor)
    if int(meta.nspinor) > ns_after_lift:
        ns_pad = int(meta.nspinor) - ns_after_lift
        psi_G_flat = jnp.pad(
            psi_G_flat, ((0, 0), (0, 0), (0, ns_pad), (0, 0)))
    psi_box = to_box(psi_G_flat, loader.box_index(k=k),
                      tuple(int(s) for s in meta.fft_grid),
                      mesh=mesh_xy)
    return psi_box, nb_logical



# ============================================================================
# R-CHUNK EXTRACTION: Contiguous r-space chunking via flattened r-index
# ============================================================================
# R-chunking advantage: r in [r_start, r_end) is contiguous in r-space and can
# be written to HDF5 in a single sequential operation. This allows arbitrary
# chunk sizes by slicing along the flattened xyz index.
# ============================================================================

# Cache for rchunk extraction function
_rchunk_slice_cache = {}


def get_sharded_wfns_rchunk_slice(
    global_psi_Gtot: jax.Array,
    meta: Meta,
    r_start,
    r_chunk_size: int,
    kvecs_frac: np.ndarray,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
) -> jax.Array:
    """
    FFT wavefunctions and extract r-chunk via flattened r-index slicing.

    R-chunking gives CONTIGUOUS r-indices: slicing r in [r_start, r_start+r_chunk_size)
    produces a contiguous block in the flattened xyz order and can be written
    to HDF5 in a single sequential operation.

    Args:
        global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
        meta: Meta object
        r_start: starting R-index (Python int for driver calls, or jax
            scalar tracer when this function is called inside an outer jit)
        r_chunk_size: static Python int — width of the slice.  Must be
            a concrete int so the FFT output shape is known at trace.
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        mesh_xy: Device mesh
        band_range: (b_start, b_end)

    Returns:
        psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')
    """
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    fft_grid = meta.fft_grid
    nx, ny, nz = fft_grid
    r_chunk_size = int(r_chunk_size)
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rtot = nx * ny * nz
    
    # Cache key - use hash of kvecs since it's constant for a given system
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = ('rchunk_slice', id(mesh_xy), nk_tot, nspinor, r_chunk_size, nx, ny, nz, kvecs_hash)
    
    if cache_key not in _rchunk_slice_cache:
        out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))

        local_ifftn = make_sharded_ifftn_3d(
            mesh_xy,
            P(None, ('x', 'y'), None, None, None, None),
            P(None, ('x', 'y'), None, None, None, None)
        )

        # No intermediate replicated shard — use two-step all-gather + all-to-all
        # to avoid materializing the full array on every device.
        kvecs_cached = jnp.asarray(kvecs_frac)
        n_rtot_cached = n_rtot

        # r_chunk_size is static (from cache key), r_start is dynamic
        r_chunk_size_static = r_chunk_size

        band_shard = P(None, ('x', 'y'), None, None)

        @partial(jax.jit, static_argnames=('nb_static',))
        def _fft_and_rslice(psi_G, r_start_dyn, nb_static):
            """FFT + phase + r-slice. Returns band-sharded r-chunk.
            Resharding happens in a SEPARATE call to prevent XLA from
            rematerializing the FFT to satisfy the output layout."""
            from common.wfn_transforms import apply_bloch_phase
            psi_r = local_ifftn(psi_G)
            psi_r = apply_bloch_phase(
                psi_r, kvecs_cached, (nx, ny, nz))
            psi_r = psi_r * jnp.sqrt(n_rtot_cached)

            # Keep padded band count (divisible by mesh size)
            nb_padded = psi_r.shape[1]
            psi_flat = psi_r.reshape(nk_tot, nb_padded, nspinor, n_rtot_cached)

            # Local r-slice via shard_map
            def _local_rslice(psi_local, r_start_arr):
                return jax.lax.dynamic_slice_in_dim(
                    psi_local, r_start_arr[0], r_chunk_size_static, axis=3)
            psi_rchunk = shard_map(
                _local_rslice, mesh=mesh_xy,
                in_specs=(band_shard, P()), out_specs=band_shard,
            )(psi_flat, jnp.array([r_start_dyn]))
            return psi_rchunk

        # Reshard {-,XY,-,-} → {-,X,-,Y} → {-,-,-,Y} (y-first).
        #
        # Stage through P(None,'x',None,'y') — do the all_to_all on 'y'
        # FIRST (split rchunk, concat bands) while the tile is still
        # small (bytes stay constant), then the final all_gather on 'x'
        # inflates the tile by p_x.  X-first inflates BEFORE the
        # all_to_all and pays 2× NCCL staging on the inflated tile;
        # y-first pays 1× (small all_to_all on input-sized tile, then
        # one all_gather).
        #
        # ``out_shardings=final`` on the jit is load-bearing: without it
        # XLA treats the second with_sharding_constraint as a
        # suggestion and silently drops the final all_gather-x,
        # returning stage sharding P(None,'x',None,'y').  With
        # out_shardings set the final sharding is a hard contract so
        # the all_gather is guaranteed to run.
        #
        # Measured at MoS2 3×3 nosym (nk=9, nb=80, Br=46080, 2×2 mesh):
        #   x-first hints:  1.86 GB/dev
        #   y-first + out_shardings:  1.33 GB/dev  (~28 % reduction)
        #
        # Using a separate jit from the FFT prevents XLA from
        # rematerializing the FFT during the reshard.
        _final_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        _stage_X_rchunk_Y = NamedSharding(mesh_xy, P(None, 'x', None, 'y'))

        @partial(jax.jit, out_shardings=_final_Y)
        def _reshard_rchunk(psi_rchunk):
            """Reshard r-chunk y-first: {-,XY,-,-} → {-,X,-,Y} → {-,-,-,Y}."""
            return jax.lax.with_sharding_constraint(psi_rchunk, _stage_X_rchunk_Y)

        def _extract_rchunk_slice(psi_G, r_start_dyn, nb_static):
            psi_rchunk = _fft_and_rslice(psi_G, r_start_dyn, nb_static)
            psi_rchunk = _reshard_rchunk(psi_rchunk)
            # Trim to actual band count outside the resharding JIT
            psi_rchunk = psi_rchunk[:, :nb_static, :, :]
            return psi_rchunk

        _rchunk_slice_cache[cache_key] = _extract_rchunk_slice

    return _rchunk_slice_cache[cache_key](global_psi_Gtot, r_start, nb)


def iter_psi_rchunk_bandwise(
    wfn, sym, meta, mesh_xy, band_range, r_start, r_end, bispinor,
    band_chunk_size: int = 16,
    k_chunk_size: int = 0,
    band_chunk_ranges: list[tuple[int, int]] | None = None,
):
    """Generator: yield ``(bc_range, psi_bc_Y)`` one band chunk at a time.

    Each yielded ``psi_bc_Y`` has shape
    ``(nk, bc_range[1]-bc_range[0], ns, r_end-r_start)`` sharded
    ``P(None, None, None, 'y')`` — the FFT-to-r-chunk slab of just
    the current band chunk.  The caller is responsible for
    accumulating contributions (e.g. ``P += einsum(ψ_L_bc, ψ_R_bc)``)
    so only one band chunk's r-chunk shard is live at any moment,
    decoupling the pair-density peak from the total band count.

    ``band_chunk_ranges`` lets the caller dictate chunk boundaries —
    pass a list to respect left/right pair-density endpoints so every
    yielded chunk lies fully inside one (or both) of those ranges and
    no out-of-range einsums ever dispatch.  When None, contiguous
    chunks of ``band_chunk_size`` are built from ``band_range``.

    Migrated to :class:`file_io.wfn_loader.WfnLoader` + ``to_rchunk``.
    ``sym`` is unused (loader builds its own SymMaps).  The retired
    ``cached_gspace`` / ``kvecs_frac`` / ``use_phdf5`` parameters have
    been dropped — PsiGStore is the in-process g_flat cache;
    WfnLoader's ``backend='auto'`` picks phdf5/eager.
    """
    del sym
    from common.wfn_transforms import to_rchunk

    b_start, b_end = band_range
    nk_tot = int(meta.nk_tot)
    nk_batch = nk_tot if k_chunk_size <= 0 else min(k_chunk_size, nk_tot)
    n_rchunk = int(r_end - r_start)

    if band_chunk_ranges is None:
        nb_total = b_end - b_start
        num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
        band_chunk_ranges = [
            (b_start + i * band_chunk_size,
             min(b_start + (i + 1) * band_chunk_size, b_end))
            for i in range(num_band_chunks)
        ]

    # JIT'd zero-allocator used by the k-chunked path (memoised by
    # shape).  Keeps the top-level ``jnp.zeros`` from being
    # rematerialised replicated on every device.
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    _zeros_Y_cache: dict = {}
    def _zeros_Y(shape):
        fn = _zeros_Y_cache.get(shape)
        if fn is None:
            fn = jax.jit(
                lambda: jnp.zeros(shape, dtype=jnp.complex128),
                out_shardings=out_Y)
            _zeros_Y_cache[shape] = fn
        return fn()

    sharding_load = P(None, ('x', 'y'), None, None)

    loader = wfn  # reuse top-level WfnLoader
    g_index_full = loader.box_index(k="full_bz")
    sym_loader = loader._ensure_sym()
    kgrid_arr = np.asarray(meta.kgrid, dtype=np.float64)
    kvecs_frac_full = (
        np.asarray(sym_loader.kvecs_asints, dtype=np.float64)
        / kgrid_arr[None, :])

    for bc_range in band_chunk_ranges:
        if nk_batch >= nk_tot:
            psi_G_flat = loader.load(
                bands=bc_range, k="full_bz",
                sharding=sharding_load, bispinor=bispinor)
            psi_bc_Y = to_rchunk(
                psi_G_flat, g_index_full, meta.fft_grid,
                int(r_start), n_rchunk, mesh=mesh_xy, norm="ortho",
                kvecs_frac=jnp.asarray(kvecs_frac_full))
            psi_bc_Y = jax.lax.with_sharding_constraint(psi_bc_Y, out_Y)
            del psi_G_flat
            yield bc_range, psi_bc_Y
        else:
            nb_chunk = bc_range[1] - bc_range[0]
            nspinor = meta.nspinor
            psi_bc_Y_full = _zeros_Y(
                (nk_tot, nb_chunk, nspinor, n_rchunk))
            for k0 in range(0, nk_tot, nk_batch):
                k1 = min(k0 + nk_batch, nk_tot)
                k_ids = list(range(k0, k1))
                psi_G_flat = loader.load(
                    bands=bc_range, k=k_ids,
                    sharding=sharding_load, bispinor=bispinor)
                psi_k_chunk = to_rchunk(
                    psi_G_flat,
                    g_index_full[k0:k1],
                    meta.fft_grid, int(r_start), n_rchunk,
                    mesh=mesh_xy, norm="ortho",
                    kvecs_frac=jnp.asarray(kvecs_frac_full[k0:k1]))
                psi_k_chunk = jax.lax.with_sharding_constraint(
                    psi_k_chunk, out_Y)
                psi_bc_Y_full = psi_bc_Y_full.at[
                    k0:k1, :, :, :].set(psi_k_chunk)
                del psi_G_flat, psi_k_chunk
            yield bc_range, psi_bc_Y_full


# ============================================================================
# Unified band-chunked FFT backend for centroid and z-chunk extraction
# ============================================================================

# Cache for centroid extraction function
_centroid_extract_cache = {}


def get_sharded_wfns_centroids(
    global_psi_Gtot: jax.Array,
    meta: Meta,
    centroid_indices: jax.Array,
    kvecs_frac: np.ndarray,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    """
    FFT wavefunctions and extract centroids for a single band chunk.
    
    This is the centroid-extraction counterpart to get_sharded_wfns_rchunk_slice.
    Both use the same caching and staging patterns for memory efficiency.
    
    Args:
        global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
        meta: Meta object
        centroid_indices: (n_rmu, 3) centroid grid coordinates
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
    
    Returns:
        psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
        psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
    """
    from runtime.padding import round_up_to_mesh_product

    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    fft_grid = meta.fft_grid
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rmu = len(centroid_indices)
    # Pad μ to ``world_size`` (= ∏ p_a over the device mesh).  The output
    # sharding ``out_Y = P(None, None, None, 'y')`` is single-axis 'y' on
    # the μ dim and the input/output of this jit is a top-level boundary,
    # so n_rmu must divide p_y.  Padding to world_size satisfies any
    # single-axis or product-axis spec on the μ dim with one rule.  Pad
    # rows are zero (jnp.pad after the gather), so downstream
    # pair-density / Σ contractions see no contribution from the pad.
    # Mirrors the band-axis pattern (``b_id_4`` padded vs ``b_id_4_user``
    # logical) at common/meta.py:99-100 and load_wfns.py:952-959.
    n_rmu_padded = round_up_to_mesh_product(n_rmu, mesh_xy)

    # Cache key.  The compiled gather closes over the centroid positions, so
    # different centroid files with the same count must not share a closure.
    centroid_indices_np = np.asarray(jax.device_get(centroid_indices), dtype=np.int64)
    centroids_hash = hash((centroid_indices_np.shape, centroid_indices_np.tobytes()))
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = (
        'centroid_extract', id(mesh_xy), nk_tot, nspinor, n_rmu, n_rmu_padded,
        nx, ny, nz, kvecs_hash, centroids_hash,
    )
    
    if cache_key not in _centroid_extract_cache:
        out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        # out_X: transposed shape is (nk, n_rmu, nb, ns) for pair density einsum
        out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
        null_4 = NamedSharding(mesh_xy, P(None, None, None, None))
        
        local_ifftn = make_sharded_ifftn_3d(
            mesh_xy,
            P(None, ('x', 'y'), None, None, None, None),
            P(None, ('x', 'y'), None, None, None, None)
        )
        
        # Pre-compute phase grids and kvecs in closure
        fx_cached = jnp.arange(nx, dtype=jnp.float64)[None, :, None, None] / nx
        fy_cached = jnp.arange(ny, dtype=jnp.float64)[None, None, :, None] / ny
        fz_cached = jnp.arange(nz, dtype=jnp.float64)[None, None, None, :] / nz
        kvecs_cached = jnp.asarray(kvecs_frac)
        n_rtot_cached = n_rtot
        
        # Pre-compute centroid linear indices (int64 for XLA compatibility)
        centroids = jnp.asarray(centroid_indices_np, dtype=jnp.int64)
        centroid_lin = (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(jnp.int64)
        
        # The band axis is padded to be divisible by p_x*p_y for the FFT.
        # Keep the padded count through the gather and reshard — trimming
        # to the actual band count happens OUTSIDE the JIT. If we trim
        # inside, the non-divisible band count causes XLA to rematerialize
        # the FFT to satisfy the output sharding.
        stage_Y_4d = NamedSharding(mesh_xy, P(None, 'y', None, None))

        @jax.jit
        def _fft_gather_reshard(psi_G):
            """FFT → phase → gather centroids → reshard. Keeps padded bands."""
            psi_r = local_ifftn(psi_G)
            phase_spatial = jnp.exp(
                2j * jnp.pi * (
                    kvecs_cached[:, 0:1, None, None] * fx_cached
                    + kvecs_cached[:, 1:2, None, None] * fy_cached
                    + kvecs_cached[:, 2:3, None, None] * fz_cached
                )
            )
            psi_r = psi_r * phase_spatial[:, None, None, :, :, :]
            psi_r = psi_r * jnp.sqrt(n_rtot_cached)
            # Do NOT trim bands here — keep padded count for clean sharding.
            nb_padded = psi_r.shape[1]

            # Flatten spatial dims and gather centroids
            psi_rtot = psi_r.reshape(nk_tot, nb_padded, nspinor, -1)
            psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)
            # psi_rmu: (nk, nb_padded, ns, n_rmu) sharded {-, XY, -, -}

            # μ-axis pad: the gather outputs at logical ``n_rmu``; the
            # output sharding ``out_Y = P(None, None, None, 'y')`` is a
            # top-level boundary that requires divisibility, so we
            # zero-pad axis 3 up to ``n_rmu_padded`` here.  Pad rows are
            # zero, ensuring downstream bilinear consumers (pair density,
            # CCT, Σ_X) see no contribution from the pad.
            if n_rmu_padded > n_rmu:
                psi_rmu = jnp.pad(
                    psi_rmu,
                    ((0, 0), (0, 0), (0, 0), (0, n_rmu_padded - n_rmu)),
                )

            # Two-step reshard on the PADDED array (divisible by both p_x and p_y):
            # Step 1: {-,XY,-,-} → {-,Y,-,-} (all-gather along X)
            psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, stage_Y_4d)
            # Step 2: {-,Y,-,-} → {-,-,-,Y} (all-to-all along Y)
            psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, out_Y)

            # Conjugate-transpose for pair density
            psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))
            psi_rmuT = jax.lax.with_sharding_constraint(psi_rmuT, out_X)

            return psi_rmu, psi_rmuT

        def _extract_centroids(psi_G, nb_actual):
            psi_rmu, psi_rmuT = _fft_gather_reshard(psi_G)
            # Trim to actual band count OUTSIDE the JIT
            psi_rmu = psi_rmu[:, :nb_actual, :, :]
            psi_rmuT = psi_rmuT[:, :, :nb_actual, :]
            return psi_rmu, psi_rmuT

        _centroid_extract_cache[cache_key] = _extract_centroids

    return _centroid_extract_cache[cache_key](global_psi_Gtot, nb)


def load_centroids_band_chunked(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    bispinor: bool,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
    band_chunk_size: int = 64,
    k_chunk_size: int | None = None,
    *,
    use_phdf5: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """
    Load centroid-sampled wavefunctions using band AND k-point chunking.

    Memory-safe version that loops over band chunks (and optionally k-point
    chunks) to avoid OOM when loading all bands/k-points at once for FFT.

    The FFT box array psi_Gtot_local has shape (nk, nb, nspinor, *fft_grid)
    and scales as O(nk * nb * n_rtot). For large k-grids (e.g. 10x10x10 =
    1000 k-points), this exceeds GPU memory. K-chunking processes a subset
    of k-points at a time, accumulating only the centroid-space outputs
    (which are O(nk * nb * n_rmu) — much smaller since n_rmu << n_rtot).

    Args:
        wfn: WFNReader
        sym: SymMaps
        meta: Meta object
        centroid_indices: (n_rmu, 3) centroid grid coordinates
        bispinor: Whether to use bispinor
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
        band_chunk_size: Bands to FFT at once (memory control)
        k_chunk_size: K-points to FFT at once (None = all at once).
            When set, processes k-points in batches to control the size
            of the FFT box array (the dominant memory bottleneck).
        use_phdf5: If True, pull G-space wavefunctions through
            :class:`common.phdf5_wfn_reader.PhdfWfnReader` (parallel HDF5
            FFI + on-device symmetry unfold).  Default False keeps the
            legacy ``WFNReader`` + ``read_Gvecs_to_devices`` path so
            existing callers are unaffected.  The phdf5 path opens the
            file once for the whole call and streams each (band-chunk,
            k-chunk) rectangle directly onto device without slurping
            ``wfns/coeffs`` into host RAM — suitable for WFN files that
            don't fit in host memory.  Handles both the all-k path and
            the k-chunked memory-budget path; for a k-chunk, the
            reader fetches only the irreducible-BZ k's backing that
            chunk (via the existing SymMaps-based dedup) and unfolds
            on device.

    Returns:
        psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
        psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)

    n_rmu divisibility: the kernels here shard the n_rmu axis by a
    single mesh axis (``'x'`` alone in psi_rmuT_X, ``'y'`` alone in
    psi_rmu_Y) — so n_rmu only needs to divide one axis size, not the
    product.  ``mesh.x = mesh.y = 4`` and 668 / 4 = 167 ✓; no padding
    needed at this layer.  The undivisibility shows up only at the
    V_q read where the trailing axis is sharded by the *product*
    ``('x', 'y')`` = 16; SlabIO's auto-pad on the on-disk dataset
    closes that gap (see ``file_io.slab_io.create_dataset``).
    """
    del use_phdf5  # WfnLoader's backend='auto' picks phdf5 when it's safe
    del sym        # WfnLoader builds its own SymMaps lazily
    from common.wfn_transforms import gflat_to_rmu
    from runtime.padding import round_up_to_mesh_product

    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rmu = int(centroid_indices.shape[0])
    centroid_idx_np = np.asarray(centroid_indices, dtype=np.int32)
    n_rtot = int(meta.fft_grid[0]) * int(meta.fft_grid[1]) * int(meta.fft_grid[2])

    # Defect 3 (zeta_rchunk_memory_model_2026-05-13/defect_catalog.md):
    # the legacy bc-loop here, paired with the unsharded FFT box inside
    # ``to_rmu``, materialised an unsharded ``c128[nk, band_chunk, ns,
    # nx, ny, nz]`` transient on every rank — Peak A in
    # ``gw/gflat_memory_model.py``.  Single slot, but the §0
    # zero-replicated-intermediates principle still bites.  ``gflat_to_rmu``
    # fuses the bc/k iteration into one shard_map + lax.scan whose
    # per-iter FFT box is sharded along the band axis on ``('x','y')``
    # and aliased across scan iters; the legacy ``band_chunk_size`` /
    # ``k_chunk_size`` knobs collapse into the single ``chunk_size``
    # below (rows of the flat (nk · nb_local) axis per scan iter).

    # Per-iter FFT box bound for ``gflat_to_rmu``: each scan iter holds
    # one ``c128[cs, ns, nx, ny, nz]`` box per rank.  ``peak_copies``
    # is the same conservative XLA scratch multiplier used historically
    # by the old k_chunk_size autodetect (4 on single-rank, 9 on
    # multi-rank — covers the IFFT scratch + IFFT output).
    n_devices = jax.device_count()
    peak_copies = 4 if n_devices == 1 else 9
    gpu_mem_bytes = 36e9
    if hasattr(meta, 'memory_per_device_gb') and meta.memory_per_device_gb > 0:
        gpu_mem_bytes = meta.memory_per_device_gb * 1e9

    # Translate legacy hints (band_chunk_size, k_chunk_size) into the
    # new flat-row count.  Both default to a non-None value; an explicit
    # k_chunk_size from the caller bounds rows × nb_padded, otherwise we
    # pick cs purely from the per-rank HBM budget.  In either case the
    # budget bound applies last so cs can never exceed it.
    cs_budget = max(1, int(gpu_mem_bytes
                           // (nspinor * n_rtot * 16 * peak_copies)))
    if k_chunk_size is not None and k_chunk_size > 0:
        # Honor an explicit cap: at most ``k_chunk_size`` k-points
        # worth of work per iter, sized against the per-rank band
        # block.  Same per-iter footprint as the legacy nested loops.
        n_bands_per_rank = max(
            1, (nb_total + n_devices - 1) // n_devices)
        cs_hint = int(k_chunk_size) * min(
            int(band_chunk_size), n_bands_per_rank)
        cs = max(1, min(cs_hint, cs_budget))
    else:
        cs = cs_budget

    # Output shardings + accumulators.  Same final layout as before:
    # psi_rmu_Y has the centroid axis on 'y'; psi_rmuT_X has it on 'x'.
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
    stage_Y_4d = NamedSharding(mesh_xy, P(None, 'y', None, None))

    n_rmu_padded = round_up_to_mesh_product(n_rmu, mesh_xy)
    sharding_load = P(None, ('x', 'y'), None, None)

    loader = wfn  # reuse top-level WfnLoader
    g_index_full = loader.box_index(k="full_bz")
    sym_loader = loader._ensure_sym()
    kgrid_arr = np.asarray(meta.kgrid, dtype=np.float64)
    kvecs_frac_full = (
        np.asarray(sym_loader.kvecs_asints, dtype=np.float64)
        / kgrid_arr[None, :])

    # Pull all (nk_tot, nb_padded, ns, ngkmax) ψ(G-flat) onto device in
    # one collective load.  The G-flat tensor is small relative to the
    # FFT box (n_rmu << n_rtot, ngkmax << n_rtot too once the G-sphere
    # cutoff is applied) so a single-shot load fits comfortably even at
    # CrI3 6×6 80 Ry scale (~50 GB total / mesh.size).  The band-pad
    # padding happens inside ``loader.load`` so ``nb_padded`` is the
    # mesh-aligned extent expected by ``gflat_to_rmu``.
    with timing.section("load_centroids.loader_load"):
        psi_G_flat = loader.load(
            bands=band_range, k="full_bz",
            sharding=sharding_load, bispinor=bispinor)
        jax.block_until_ready(psi_G_flat)
    nb_padded = int(psi_G_flat.shape[1])

    # One shard_map + scan: full (nk, nb_padded) extent, centroid
    # samples emitted band-sharded.  FFT box exists once at scan-body
    # scope and is per-rank-local (mesh.size× smaller than the legacy
    # unsharded transient).
    with timing.section("load_centroids.gflat_to_rmu"):
        psi_rmu_band = gflat_to_rmu(
            psi_G_flat, g_index_full, centroid_idx_np,
            mesh=mesh_xy, fft_grid=meta.fft_grid,
            kvecs_frac=jnp.asarray(kvecs_frac_full),
            norm="ortho", chunk_size=cs)
        jax.block_until_ready(psi_rmu_band)
    del psi_G_flat

    # Single global reshard {None, XY, None, None} → {None, None, None, Y}
    # plus a conjugate-transpose into the rmuT_X layout.  Two-step
    # reshard via ``stage_Y_4d`` for the same SPMD reason as the
    # legacy per-chunk path: a single all-to-all on the band axis
    # before the second all-to-all onto the n_rmu axis.
    @jax.jit
    def _reshard_all(psi_rmu_band):
        if n_rmu_padded > n_rmu:
            psi_rmu_band = jnp.pad(
                psi_rmu_band,
                ((0, 0), (0, 0), (0, 0), (0, n_rmu_padded - n_rmu)),
            )
        psi_rmu = jax.lax.with_sharding_constraint(psi_rmu_band, stage_Y_4d)
        psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, out_Y)
        psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))
        psi_rmuT = jax.lax.with_sharding_constraint(psi_rmuT, out_X)
        return psi_rmu, psi_rmuT

    with timing.section("load_centroids.reshard"):
        psi_rmu_all, psi_rmuT_all = _reshard_all(psi_rmu_band)
        jax.block_until_ready(psi_rmuT_all)
    del psi_rmu_band

    # Slice off the band pad rows added by ``loader.load``.  When
    # ``nb_padded == nb_total`` this is a no-op slice that XLA folds away.
    if nb_padded > nb_total:
        psi_rmu_all = psi_rmu_all[:, :nb_total, :, :]
        psi_rmuT_all = psi_rmuT_all[:, :, :nb_total, :]
        psi_rmu_all = jax.lax.with_sharding_constraint(psi_rmu_all, out_Y)
        psi_rmuT_all = jax.lax.with_sharding_constraint(psi_rmuT_all, out_X)
    gc.collect()

    # Zero user-band-pad rows (unchanged contract).
    nb_user_in_range = max(0, meta.b_id_4_user - b_start)
    if nb_user_in_range < nb_total:
        zero_y = jnp.zeros_like(psi_rmu_all[:, nb_user_in_range:nb_total, :, :])
        zero_x = jnp.zeros_like(psi_rmuT_all[:, :, nb_user_in_range:nb_total, :])
        psi_rmu_all = psi_rmu_all.at[
            :, nb_user_in_range:nb_total, :, :].set(zero_y)
        psi_rmuT_all = psi_rmuT_all.at[
            :, :, nb_user_in_range:nb_total, :].set(zero_x)

    return psi_rmu_all, psi_rmuT_all

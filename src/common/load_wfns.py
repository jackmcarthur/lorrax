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
    irk_to_k = np.asarray(sym.irk_to_k_map)
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
    from common.wfn_transforms import to_rmu
    from runtime.padding import round_up_to_mesh_product

    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rmu = int(centroid_indices.shape[0])
    centroid_idx_np = np.asarray(centroid_indices, dtype=np.int32)

    # k_chunk_size autodetect — bound the FFT-box transient peak inside
    # to_rmu (the only place a full r-box exists in the new pipeline).
    #
    # NB: we cost per-k against ``nb_padded`` (NOT ``bands_per_shard``).
    # XLA empirically materialises the (n_k, nb_padded, ns, nx, ny, nz)
    # FFT box UNSHARDED on every rank at CrI3 6×6×1 80 Ry scale — the
    # band-axis ``with_sharding_constraint`` and the
    # ``make_jittable_local_ifftn_3d`` helper both fail to keep the box
    # sharded once XLA's FFT planner pulls the gather + IFFT into a
    # single fused HLO module.  Sizing per_k against the unsharded
    # ``nb_padded`` extent ensures the chunker picks ``k_chunk_size``
    # small enough that even the worst-case unsharded box fits.  Costs
    # ~16× more headroom on a 2-D 4×4 mesh than the sharded estimate
    # but avoids the 40 GB OOM seen on CrI3 6×6 (matched what we see
    # in HLO rematerialization warnings: peak ≈ 4·box_unsharded).
    if k_chunk_size is None:
        n_rtot = meta.fft_grid[0] * meta.fft_grid[1] * meta.fft_grid[2]
        n_devices = jax.device_count()
        nb_chunk = min(band_chunk_size, nb_total)
        bands_per_shard = (nb_chunk + n_devices - 1) // n_devices
        nb_padded = bands_per_shard * n_devices

        peak_copies = 4 if n_devices == 1 else 9
        gpu_mem_bytes = 36e9
        if hasattr(meta, 'memory_per_device_gb') and meta.memory_per_device_gb > 0:
            gpu_mem_bytes = meta.memory_per_device_gb * 1e9

        per_k_bytes = nb_padded * nspinor * n_rtot * 16 * peak_copies
        k_chunk_size = max(1, min(int(gpu_mem_bytes / per_k_bytes), nk_tot))

    k_chunk_size = min(k_chunk_size, nk_tot)
    num_k_chunks = (nk_tot + k_chunk_size - 1) // k_chunk_size
    needs_k_chunking = num_k_chunks > 1
    if needs_k_chunking:
        print(f"  K-point chunking: {num_k_chunks} chunks of {k_chunk_size} "
              f"(total {nk_tot} k-points)")

    # Output shardings + accumulators.  Same final layout as before:
    # psi_rmu_Y has the centroid axis on 'y'; psi_rmuT_X has it on 'x'.
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
    stage_Y_4d = NamedSharding(mesh_xy, P(None, 'y', None, None))

    n_rmu_padded = round_up_to_mesh_product(n_rmu, mesh_xy)
    psi_rmu_all = jnp.zeros(
        (nk_tot, nb_total, nspinor, n_rmu_padded), dtype=jnp.complex128)
    psi_rmuT_all = jnp.zeros(
        (nk_tot, n_rmu_padded, nb_total, nspinor), dtype=jnp.complex128)
    psi_rmu_all = jax.lax.with_sharding_constraint(psi_rmu_all, out_Y)
    psi_rmuT_all = jax.lax.with_sharding_constraint(psi_rmuT_all, out_X)

    # Reshard helper: ``to_rmu`` returns ``(nk, nb_padded, ns, n_rmu)``
    # with the band axis sharded on ('x', 'y').  Downstream consumers
    # want the n_rmu axis sharded (P(None, None, None, 'y') for psi_rmu_Y
    # and P(None, 'x', None, None) for the conjugate-transposed
    # psi_rmuT_X).  Cached per (nb_chunk_padded, nk_chunk) shape so
    # repeated band-chunk × k-chunk calls share the compiled jit.
    _reshard_cache: dict = {}

    def _make_reshard_fn(nb_chunk_padded: int, nk_chunk: int):
        key = (int(nb_chunk_padded), int(nk_chunk))
        fn = _reshard_cache.get(key)
        if fn is not None:
            return fn

        @jax.jit
        def _reshard(psi_rmu_band):
            # Input: (nk_chunk, nb_chunk_padded, ns, n_rmu) band-sharded.
            if n_rmu_padded > n_rmu:
                psi_rmu_band = jnp.pad(
                    psi_rmu_band,
                    ((0, 0), (0, 0), (0, 0), (0, n_rmu_padded - n_rmu)),
                )
            # Two-step reshard {-,XY,-,-} → {-,Y,-,-} → {-,-,-,Y}.
            psi_rmu = jax.lax.with_sharding_constraint(psi_rmu_band, stage_Y_4d)
            psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, out_Y)
            # Conjugate-transpose for pair density.
            psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))
            psi_rmuT = jax.lax.with_sharding_constraint(psi_rmuT, out_X)
            return psi_rmu, psi_rmuT

        _reshard_cache[key] = _reshard
        return _reshard

    num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
    sharding_load = P(None, ('x', 'y'), None, None)

    loader = wfn  # reuse top-level WfnLoader
    g_index_full = loader.box_index(k="full_bz")
    sym_loader = loader._ensure_sym()
    kgrid_arr = np.asarray(meta.kgrid, dtype=np.float64)
    kvecs_frac_full = (
        np.asarray(sym_loader.kvecs_asints, dtype=np.float64)
        / kgrid_arr[None, :])

    # NOTE: an AsyncWfnReader (file_io/wfn_loader.py) is available and
    # was tried here to pipeline ``loader.load(bc+1)`` against bc[i]'s
    # ``to_rmu``.  At MoS2 3×3 scale, the GPU H2D/compute overlap
    # measured by xprof stayed exactly 0.000 even with depth-2
    # prefetch and forced 3 band chunks — XLA's stream scheduler
    # doesn't pipeline our H2D against compute here.  Keep the
    # synchronous path; revisit the async pattern when scale grows
    # (CrI3) or when the wider zeta/V_q async-reader story lands.
    for bc_idx in range(num_band_chunks):
        bc_start = b_start + bc_idx * band_chunk_size
        bc_end = min(bc_start + band_chunk_size, b_end)
        bc_range = (bc_start, bc_end)
        nb_chunk = bc_end - bc_start
        local_bc_start = bc_idx * band_chunk_size
        local_bc_end = local_bc_start + nb_chunk

        if not needs_k_chunking:
            with timing.section("load_centroids.loader_load"):
                psi_G_flat = loader.load(
                    bands=bc_range, k="full_bz",
                    sharding=sharding_load, bispinor=bispinor)
                jax.block_until_ready(psi_G_flat)
            with timing.section("load_centroids.to_rmu"):
                psi_rmu_band = to_rmu(
                    psi_G_flat, g_index_full, meta.fft_grid,
                    centroid_idx_np, norm="ortho",
                    kvecs_frac=jnp.asarray(kvecs_frac_full),
                    mesh=mesh_xy)
                jax.block_until_ready(psi_rmu_band)
            nb_padded_chunk = int(psi_rmu_band.shape[1])
            reshard_fn = _make_reshard_fn(nb_padded_chunk, nk_tot)
            with timing.section("load_centroids.reshard"):
                psi_rmu_chunk, psi_rmuT_chunk = reshard_fn(psi_rmu_band)
                jax.block_until_ready(psi_rmuT_chunk)
            psi_rmu_chunk = psi_rmu_chunk[:, :nb_chunk, :, :]
            psi_rmuT_chunk = psi_rmuT_chunk[:, :, :nb_chunk, :]
            psi_rmu_all = psi_rmu_all.at[
                :, local_bc_start:local_bc_end, :, :].set(psi_rmu_chunk)
            psi_rmuT_all = psi_rmuT_all.at[
                :, :, local_bc_start:local_bc_end, :].set(psi_rmuT_chunk)
            del psi_G_flat, psi_rmu_band, psi_rmu_chunk, psi_rmuT_chunk
        else:
            for kc_idx in range(num_k_chunks):
                kc_start = kc_idx * k_chunk_size
                kc_end = min(kc_start + k_chunk_size, nk_tot)
                nk_chunk = kc_end - kc_start
                k_ids = list(range(kc_start, kc_end))

                psi_G_flat = loader.load(
                    bands=bc_range, k=k_ids,
                    sharding=sharding_load, bispinor=bispinor)
                kvecs_chunk = kvecs_frac_full[kc_start:kc_end]
                g_index_chunk = g_index_full[kc_start:kc_end]
                psi_rmu_band = to_rmu(
                    psi_G_flat, g_index_chunk, meta.fft_grid,
                    centroid_idx_np, norm="ortho",
                    kvecs_frac=jnp.asarray(kvecs_chunk),
                    mesh=mesh_xy)
                nb_padded_chunk = int(psi_rmu_band.shape[1])
                reshard_fn = _make_reshard_fn(nb_padded_chunk, nk_chunk)
                psi_rmu_kchunk, psi_rmuT_kchunk = reshard_fn(psi_rmu_band)
                psi_rmu_kchunk = psi_rmu_kchunk[:, :nb_chunk, :, :]
                psi_rmuT_kchunk = psi_rmuT_kchunk[:, :, :nb_chunk, :]
                psi_rmu_all = psi_rmu_all.at[
                    kc_start:kc_end, local_bc_start:local_bc_end, :, :
                ].set(psi_rmu_kchunk)
                psi_rmuT_all = psi_rmuT_all.at[
                    kc_start:kc_end, :, local_bc_start:local_bc_end, :
                ].set(psi_rmuT_kchunk)
                del psi_G_flat, psi_rmu_band, psi_rmu_kchunk, psi_rmuT_kchunk
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

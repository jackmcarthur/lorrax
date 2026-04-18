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
    make_jittable_local_ifftn_3d,
)


def _make_kchunk_meta(meta: Meta, kc_start: int, kc_end: int):
    """Create a shallow copy of Meta with nk_tot adjusted for a k-chunk.

    The returned object shares all attributes with the original except
    nk_tot, which is set to the chunk size. This lets read_Gvecs_to_devices
    and get_sharded_wfns_centroids process a subset of k-points using the
    same code paths.
    """
    from copy import copy
    meta_chunk = copy(meta)
    meta_chunk.nk_tot = kc_end - kc_start
    return meta_chunk


def load_kpoint_fftbox(wfn, sym, meta, k_idx, nb):
    """Load a single k-point's wavefunction into the FFT box on GPU.

    Returns jax array of shape (nb, nspinor, nx, ny, nz), ~0.55 GiB for 12x12.
    """
    nx, ny, nz = meta.fft_grid
    band_indices = np.arange(nb)
    gvecs_k = np.asarray(sym.get_gvecs_kfull(wfn, k_idx))
    cnk = sym.get_cnk_fullzone_batch(wfn, band_indices, k_idx)
    psi = np.zeros((nb, meta.nspinor, nx, ny, nz), dtype=np.complex128)
    psi[:, :meta.nspinor_wfnfile, gvecs_k[:, 0], gvecs_k[:, 1], gvecs_k[:, 2]] = cnk
    return jnp.asarray(psi)


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
    en_irk = np.asarray(wfn.energies[0, :, band_lo:band_hi], dtype=np.float64)
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
    """
    Non-jitted: load cnk(G) for k-points and (padded) band shards into a global
    sharded G-space FFT box over a 2D mesh ['x','y'] along the band axis.

    Args:
        k_range: Optional (kc_start, kc_end) to load only a subset of k-points.
            When set, meta.nk_tot should match the chunk size (kc_end - kc_start).
            SymMaps operations use the original full-BZ indices offset by kc_start.

    Returns the global sharded array global_psi_Gtot and nb_actual.
    """
    nb = bandrange[1] - bandrange[0]

    # 2D device mesh already provided (mesh_xy); derive grid dims
    devices_2d = mesh_xy.devices
    grid_x, grid_y = devices_2d.shape
    total_devices = grid_x * grid_y

    # Bands per shard and total padded bands
    bands_per_shard = (nb + total_devices - 1) // total_devices
    total_bands_padded = bands_per_shard * total_devices

    # Map local devices to their (x, y) coordinates in the mesh
    local_devices = list(jax.local_devices())
    local_coords = [tuple(np.argwhere(np.asarray(devices_2d) == d)[0]) for d in local_devices]
    local_flat_ids = [cx * grid_y + cy for (cx, cy) in local_coords]
    order = np.argsort(local_flat_ids)
    local_coords = [local_coords[i] for i in order]
    n_local_shards = len(local_coords)

    # Local buffer for k-points in range and this process's band shards (G-space)
    kc_start_alloc = k_range[0] if k_range is not None else 0
    kc_end_alloc = k_range[1] if k_range is not None else sym.nk_tot
    nk_alloc = kc_end_alloc - kc_start_alloc
    psi_Gtot_local = np.zeros(
        (nk_alloc, n_local_shards * bands_per_shard, meta.nspinor, *meta.fft_grid),
        dtype=np.complex128,
    )

    def place_band_into_local(j: int) -> tuple[int, int] | None:
        global_shard = j // bands_per_shard
        shard_x, shard_y = divmod(global_shard, grid_y)
        try:
            local_slot = local_coords.index((shard_x, shard_y))
        except ValueError:
            return None
        offset = j % bands_per_shard
        return local_slot, offset

    # Pre-compute which bands this process owns (avoids checking every band per k-point)
    with timing.section("load_wfns.precompute_owned"):
        owned_band_indices = []  # global band indices
        local_band_indices = []  # where they go in local buffer
        for j in range(nb):
            placement = place_band_into_local(j)
            if placement is not None:
                local_slot, offset = placement
                owned_band_indices.append(bandrange[0] + j)
                local_band_indices.append(local_slot * bands_per_shard + offset)
        owned_band_indices = np.array(owned_band_indices, dtype=np.int64)
        local_band_indices = np.array(local_band_indices, dtype=np.int64)
        n_owned = len(owned_band_indices)

    # Determine k-point range
    kc_start = k_range[0] if k_range is not None else 0
    kc_end = k_range[1] if k_range is not None else sym.nk_tot
    nk_chunk = kc_end - kc_start

    # Load G-coefficients for k-points in range
    # Phase 1: HDF5 reads (inherently serial) - collect all data
    with timing.section("load_wfns.k_loop"):
        # Pre-compute ngk and max for padding (use full-BZ indices for SymMaps)
        ngk_all = np.array([int(wfn.ngk[sym.irk_to_k_map[kc_start + k]]) for k in range(nk_chunk)])
        max_ngk = int(ngk_all.max())

        # Pre-allocate padded arrays for batched scatter
        n_local_bands = n_local_shards * bands_per_shard
        psi_Gspace_all = np.zeros((nk_chunk, n_local_bands, meta.nspinor, max_ngk), dtype=np.complex128)
        gvecs_all = np.zeros((nk_chunk, max_ngk, 3), dtype=np.int32)

        for k_local in range(nk_chunk):
            k_idx = kc_start + k_local  # full-BZ index for SymMaps
            gvecs_k_rot = np.asarray(sym.get_gvecs_kfull(wfn, k_idx))
            ngk = ngk_all[k_local]
            
            # Store G-vectors (padded)
            gvecs_all[k_local, :ngk, :] = gvecs_k_rot

            # Batch read and rotate all owned bands at once
            if n_owned > 0:
                cnk_batch = sym.get_cnk_fullzone_batch(wfn, owned_band_indices, k_idx)
                psi_Gspace_all[k_local, local_band_indices, 0:meta.nspinor_wfnfile, :ngk] = cnk_batch

            # Expand to 4 components if requested
            if bispinor:
                psi_Gspace_local = psi_Gspace_all[k_local, :, :, :ngk]
                psi_Gspace_all[k_local, :, 2:4, :ngk] = np.asarray(get_small_psi_component(
                    jnp.asarray(gvecs_k_rot),
                    jnp.asarray(sym.unfolded_kpts[k_idx], dtype=jnp.float64),
                    jnp.asarray(wfn.bvec, dtype=jnp.float64),
                    jnp.asarray(psi_Gspace_local),
                ))
    
    # Phase 2: Scatter to FFT box (NumPy advanced indexing)
    # Note: JAX scatter is slower due to immutability overhead in loop
    with timing.section("load_wfns.scatter"):
        for k_local in range(nk_chunk):
            ngk = ngk_all[k_local]
            gvecs_k = gvecs_all[k_local, :ngk]
            psi_k = psi_Gspace_all[k_local, :, :, :ngk]
            # Scatter: psi_Gtot_local[k, :, :, gx, gy, gz] = psi_k.T
            psi_Gtot_local[k_local, :, :, gvecs_k[:, 0], gvecs_k[:, 1], gvecs_k[:, 2]] = np.transpose(psi_k, (2, 0, 1))

    # Promote local buffer to a global sharded JAX array over bands across both [x,y]
    with timing.section("load_wfns.make_global_array"):
        global_shape = (nk_chunk, total_bands_padded, meta.nspinor, *meta.fft_grid)
        band_sharding = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None, None, None))
        # Use device_put for faster host-to-device transfer (9x faster than jnp.asarray)
        psi_local_jax = jax.device_put(psi_Gtot_local)
        global_psi_Gtot = jax.make_array_from_process_local_data(
            band_sharding, psi_local_jax, global_shape
        )

    return global_psi_Gtot, nb





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
    r_start: int,
    r_end: int,
    kvecs_frac: np.ndarray,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
) -> jax.Array:
    """
    FFT wavefunctions and extract r-chunk via flattened r-index slicing.
    
    R-chunking gives CONTIGUOUS r-indices: slicing r in [r_start, r_end)
    produces a contiguous block in the flattened xyz order and can be written
    to HDF5 in a single sequential operation.
    
    Args:
        global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
        meta: Meta object
        r_start, r_end: R-index range [r_start, r_end)
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
    r_chunk_size = r_end - r_start
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rtot = nx * ny * nz
    
    # Cache key - use hash of kvecs since it's constant for a given system
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = ('rchunk_slice', id(mesh_xy), nk_tot, nspinor, r_chunk_size, nx, ny, nz, kvecs_hash)
    
    if cache_key not in _rchunk_slice_cache:
        out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        
        local_ifftn = make_jittable_local_ifftn_3d(
            mesh_xy, 
            P(None, ('x', 'y'), None, None, None, None),
            P(None, ('x', 'y'), None, None, None, None)
        )
        
        # No intermediate replicated shard — use two-step all-gather + all-to-all
        # to avoid materializing the full array on every device.
        
        # Pre-compute phase grids and kvecs ONCE in closure
        fx_cached = jnp.arange(nx, dtype=jnp.float64)[None, :, None, None] / nx
        fy_cached = jnp.arange(ny, dtype=jnp.float64)[None, None, :, None] / ny
        fz_cached = jnp.arange(nz, dtype=jnp.float64)[None, None, None, :] / nz
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

        # Reshard {-,XY,-,-} → {-,-,-,Y}: requires all-gather of bands.
        # The intermediate (nb_pad/p_y bands × full r_chunk) is the binding
        # memory constraint. The chunk solver must account for this when
        # sizing r_chunk. Using a separate JIT prevents XLA from
        # rematerializing the FFT during the reshard.
        _final_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        _stage_Y = NamedSharding(mesh_xy, P(None, 'y', None, None))

        @jax.jit
        def _reshard_rchunk(psi_rchunk):
            """Reshard r-chunk: {-,XY,-,-} → {-,Y,-,-} → {-,-,-,Y}."""
            psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, _stage_Y)
            psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, _final_Y)
            return psi_rchunk

        def _extract_rchunk_slice(psi_G, r_start_dyn, nb_static):
            psi_rchunk = _fft_and_rslice(psi_G, r_start_dyn, nb_static)
            psi_rchunk = _reshard_rchunk(psi_rchunk)
            # Trim to actual band count outside the resharding JIT
            psi_rchunk = psi_rchunk[:, :nb_static, :, :]
            return psi_rchunk

        _rchunk_slice_cache[cache_key] = _extract_rchunk_slice

    return _rchunk_slice_cache[cache_key](global_psi_Gtot, r_start, nb)


def get_psi_rchunk(
    wfn, sym, meta, mesh_xy, band_range, r_start, r_end, bispinor,
    band_chunk_size: int = 16,
    k_chunk_size: int = 0,
    cached_gspace: list[tuple[jax.Array, tuple[int, int]]] | None = None,
    kvecs_frac: np.ndarray | None = None,
) -> jax.Array:
    """
    Extract an r-chunk of wavefunctions in real space.

    If cached_gspace is provided (from load_gspace_for_bands), reuses the
    pre-loaded G-space data — avoids redundant HDF5 reads across r-chunk
    iterations. Otherwise, loads G-space fresh from WFN.h5 each call.

    Args:
        wfn, sym: WFNReader and SymMaps (unused if cached_gspace is provided)
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
        r_start, r_end: R-index range (contiguous in flattened xyz)
        bispinor: Whether to use bispinor (unused if cached_gspace is provided)
        band_chunk_size: Bands to FFT at once
        k_chunk_size: K-points per batch (0 = all at once)
        cached_gspace: Pre-loaded G-space from load_gspace_for_bands()
        kvecs_frac: (nk, 3) k-vectors in fractional coords. If None,
            computed from sym.kvecs_asints / meta.kgrid.

    Returns:
        psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')
    """
    if kvecs_frac is None:
        kgrid = np.array(meta.kgrid)
        kvecs_frac = sym.kvecs_asints / kgrid[None, :]
    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    n_rchunk = r_end - r_start

    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    psi_rchunk_all = jnp.zeros((nk_tot, nb_total, nspinor, n_rchunk), dtype=jnp.complex128)
    psi_rchunk_all = jax.lax.with_sharding_constraint(psi_rchunk_all, out_Y)

    nk_batch = nk_tot if k_chunk_size <= 0 else min(k_chunk_size, nk_tot)

    if cached_gspace is not None:
        # Fast path: reuse pre-loaded G-space (no HDF5 reads)
        for bc_idx, (global_psi_Gtot, bc_range) in enumerate(cached_gspace):
            nb_chunk = bc_range[1] - bc_range[0]
            psi_rchunk_chunk = get_sharded_wfns_rchunk_slice(
                global_psi_Gtot, meta, r_start, r_end, kvecs_frac, mesh_xy, bc_range)
            local_bc_start = bc_idx * band_chunk_size
            psi_rchunk_all = psi_rchunk_all.at[:, local_bc_start:local_bc_start + nb_chunk, :, :].set(psi_rchunk_chunk)
            del psi_rchunk_chunk
    else:
        # Slow path: load G-space from HDF5 for each band chunk
        num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
        for bc_idx in range(num_band_chunks):
            bc_start = b_start + bc_idx * band_chunk_size
            bc_end = min(bc_start + band_chunk_size, b_end)
            bc_range = (bc_start, bc_end)
            global_psi_Gtot, _ = read_Gvecs_to_devices(wfn, sym, bc_range, meta, bispinor, mesh_xy)

            if nk_batch >= nk_tot:
                psi_rchunk_chunk = get_sharded_wfns_rchunk_slice(
                    global_psi_Gtot, meta, r_start, r_end, kvecs_frac, mesh_xy, bc_range)
                local_bc_start = bc_idx * band_chunk_size
                psi_rchunk_all = psi_rchunk_all.at[:, local_bc_start:local_bc_start + (bc_end - bc_start), :, :].set(psi_rchunk_chunk)
                del psi_rchunk_chunk
            else:
                local_bc_start = bc_idx * band_chunk_size
                nb_chunk = bc_end - bc_start
                for k0 in range(0, nk_tot, nk_batch):
                    k1 = min(k0 + nk_batch, nk_tot)
                    sub_meta = _make_kchunk_meta(meta, k0, k1)
                    psi_k_chunk = get_sharded_wfns_rchunk_slice(
                        global_psi_Gtot[k0:k1], sub_meta, r_start, r_end,
                        kvecs_frac[k0:k1], mesh_xy, bc_range)
                    psi_rchunk_all = psi_rchunk_all.at[k0:k1, local_bc_start:local_bc_start + nb_chunk, :, :].set(psi_k_chunk)
                    del psi_k_chunk
            del global_psi_Gtot

    return psi_rchunk_all


# Backward compatibility alias
def get_psi_rchunk_from_cached(
    cached_gspace, meta, mesh_xy, band_range, r_start, r_end, kvecs_frac,
    band_chunk_size: int = 16,
) -> jax.Array:
    """Deprecated: use get_psi_rchunk(cached_gspace=..., kvecs_frac=...) instead."""
    return get_psi_rchunk(
        wfn=None, sym=None, meta=meta, mesh_xy=mesh_xy,
        band_range=band_range, r_start=r_start, r_end=r_end,
        bispinor=False, band_chunk_size=band_chunk_size,
        cached_gspace=cached_gspace, kvecs_frac=kvecs_frac)


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
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    fft_grid = meta.fft_grid
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rmu = len(centroid_indices)
    
    # Cache key
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = ('centroid_extract', id(mesh_xy), nk_tot, nspinor, n_rmu, nx, ny, nz, kvecs_hash)
    
    if cache_key not in _centroid_extract_cache:
        out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        # out_X: transposed shape is (nk, n_rmu, nb, ns) for pair density einsum
        out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
        null_4 = NamedSharding(mesh_xy, P(None, None, None, None))
        
        local_ifftn = make_jittable_local_ifftn_3d(
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
        centroids = jnp.asarray(centroid_indices, dtype=jnp.int64)
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

    Returns:
        psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
        psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
    """
    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    n_rmu = len(centroid_indices)

    # Auto-detect k_chunk_size from memory budget if not specified.
    #
    # Single-GPU: FFT peak is 4× the shard (input + output + 2 staging).
    # Multi-GPU: collective buffers for the {-,XY,-,-} → {-,-,-,Y} reshard
    # add ~5× more, giving ~9× total. Measured on Si 4×4×4, 4 A100 GPUs
    # (see runs/Si/04_si_4x4x4_memory_assay/assay_report.md).
    if k_chunk_size is None:
        n_rtot = meta.fft_grid[0] * meta.fft_grid[1] * meta.fft_grid[2]
        n_devices = jax.device_count()
        nb_chunk = min(band_chunk_size, nb_total)
        bands_per_shard = (nb_chunk + n_devices - 1) // n_devices

        peak_copies = 4 if n_devices == 1 else 9
        gpu_mem_bytes = 36e9
        if hasattr(meta, 'memory_per_device_gb') and meta.memory_per_device_gb > 0:
            gpu_mem_bytes = meta.memory_per_device_gb * 1e9

        per_k_bytes = bands_per_shard * nspinor * n_rtot * 16 * peak_copies
        k_chunk_size = max(1, min(int(gpu_mem_bytes / per_k_bytes), nk_tot))

    k_chunk_size = min(k_chunk_size, nk_tot)
    num_k_chunks = (nk_tot + k_chunk_size - 1) // k_chunk_size
    needs_k_chunking = num_k_chunks > 1

    if needs_k_chunking:
        print(f"  K-point chunking: {num_k_chunks} chunks of {k_chunk_size} "
              f"(total {nk_tot} k-points)")

    # Get k-vectors
    kgrid = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid[None, :]

    # Output shardings
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))

    # Allocate output arrays for all bands and k-points
    psi_rmu_all = jnp.zeros((nk_tot, nb_total, nspinor, n_rmu), dtype=jnp.complex128)
    psi_rmuT_all = jnp.zeros((nk_tot, n_rmu, nb_total, nspinor), dtype=jnp.complex128)
    psi_rmu_all = jax.lax.with_sharding_constraint(psi_rmu_all, out_Y)
    psi_rmuT_all = jax.lax.with_sharding_constraint(psi_rmuT_all, out_X)

    # Process bands in chunks
    num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size

    for bc_idx in range(num_band_chunks):
        bc_start = b_start + bc_idx * band_chunk_size
        bc_end = min(bc_start + band_chunk_size, b_end)
        bc_range = (bc_start, bc_end)
        nb_chunk = bc_end - bc_start
        local_bc_start = bc_idx * band_chunk_size
        local_bc_end = local_bc_start + nb_chunk

        if not needs_k_chunking:
            # Original path: load all k-points at once
            global_psi_Gtot, _ = read_Gvecs_to_devices(
                wfn, sym, bc_range, meta, bispinor, mesh_xy)
            psi_rmu_chunk, psi_rmuT_chunk = get_sharded_wfns_centroids(
                global_psi_Gtot, meta, centroid_indices, kvecs_frac, mesh_xy, bc_range)
            psi_rmu_all = psi_rmu_all.at[:, local_bc_start:local_bc_end, :, :].set(psi_rmu_chunk)
            psi_rmuT_all = psi_rmuT_all.at[:, :, local_bc_start:local_bc_end, :].set(psi_rmuT_chunk)
            del global_psi_Gtot, psi_rmu_chunk, psi_rmuT_chunk
        else:
            # K-chunked path: process k-points in batches
            for kc_idx in range(num_k_chunks):
                kc_start = kc_idx * k_chunk_size
                kc_end = min(kc_start + k_chunk_size, nk_tot)
                nk_chunk = kc_end - kc_start

                # Create a temporary Meta-like object with the k-chunk's nk_tot
                # so that read_Gvecs_to_devices and get_sharded_wfns_centroids
                # operate on the chunk only.
                meta_kchunk = _make_kchunk_meta(meta, kc_start, kc_end)
                kvecs_frac_chunk = kvecs_frac[kc_start:kc_end]

                # Load G-space for this k-chunk × band-chunk
                global_psi_Gtot, _ = read_Gvecs_to_devices(
                    wfn, sym, bc_range, meta_kchunk, bispinor, mesh_xy,
                    k_range=(kc_start, kc_end))

                # FFT and extract centroids
                psi_rmu_kchunk, psi_rmuT_kchunk = get_sharded_wfns_centroids(
                    global_psi_Gtot, meta_kchunk, centroid_indices,
                    kvecs_frac_chunk, mesh_xy, bc_range)

                # Accumulate into full output
                psi_rmu_all = psi_rmu_all.at[kc_start:kc_end, local_bc_start:local_bc_end, :, :].set(psi_rmu_kchunk)
                psi_rmuT_all = psi_rmuT_all.at[kc_start:kc_end, :, local_bc_start:local_bc_end, :].set(psi_rmuT_kchunk)

                del global_psi_Gtot, psi_rmu_kchunk, psi_rmuT_kchunk
                gc.collect()
    
    return psi_rmu_all, psi_rmuT_all

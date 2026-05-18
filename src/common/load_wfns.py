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
    # Round-6 canonical accessor: pass the loader-cached device-resident
    # sphere index directly to gflat_to_rmu so the captured-in-closure
    # buffer IS the canonical one shared with psi_G_store's
    # _g_index_dev.  Pre-Round-6 we passed the host numpy from
    # loader.box_index(...) and let gflat_to_rmu's content-hash cache
    # build its own device buffer — distinct sharding (no NamedSharding
    # on the wfn_transforms path) meant the cached dedup didn't unify
    # with box_index_dev's REPLICATED NamedSharding buffer.  Result: 3
    # device buffers for the same (nk, nx, ny, nz) data (agent_l Round-5
    # §2 measured live count = 3 at pre_rchunk_loop).
    g_index_full = loader.box_index_dev(k="full_bz", mesh=mesh_xy)
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
    #
    # Past-mnband zero-pad (the contract promised by Meta docstring at
    # ``common/meta.py:100-117``): ``b_id_4 = _round_up(b_id_4_user,
    # world_size)`` may exceed the file's ``mnband`` whenever
    # ``world_size`` rounds the user's band count past the file extent
    # (CrI3 6×6 30Ry SOC: mnband=86, world_size=16 ⇒ b_id_4=96).
    # ``WfnLoader.load`` validates ``b_hi <= self.nbands`` at
    # ``file_io/wfn_loader.py:678``; without the cap below every rank
    # raises at module init.  Cap the loader call at ``loader.nbands``
    # and zero-pad the band axis up to ``nb_total = b_end - b_start``,
    # preserving the (None, ('x','y'), None, None) sharding.  The pad
    # rows are physically zero — same contract the user-band-pad zero
    # block below applies to the (b_id_4_user, b_id_4) slots.
    file_nbands = int(loader.nbands)
    b_end_in_file = min(b_end, file_nbands)
    with timing.section("load_centroids.loader_load"):
        psi_G_flat = loader.load(
            bands=(b_start, b_end_in_file), k="full_bz",
            sharding=sharding_load, bispinor=bispinor)
        jax.block_until_ready(psi_G_flat)
    nb_loaded = int(psi_G_flat.shape[1])
    if nb_loaded < nb_total:
        pad_bands = nb_total - nb_loaded
        psi_G_flat = jnp.concatenate(
            [psi_G_flat,
             jnp.zeros(
                 (psi_G_flat.shape[0], pad_bands,
                  psi_G_flat.shape[2], psi_G_flat.shape[3]),
                 dtype=psi_G_flat.dtype)],
            axis=1)
        psi_G_flat = jax.lax.with_sharding_constraint(
            psi_G_flat, NamedSharding(mesh_xy, sharding_load))
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

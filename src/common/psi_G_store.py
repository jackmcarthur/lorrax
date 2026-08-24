"""Host-resident ψ(G-flat) staging for the one-time ψ(r) cache build.

The one-time cache builder consumes ψ(G) one band chunk at a time, and the
ISDF fit later slices ψ(r) by band chunk and r chunk.  Holding ψ in full FFT-box
representation on host costs ``nb · ns · nx · ny · nz`` complex128 per
rank — for CrI3-class systems that's tens of GB.  Holding ψ in G-flat
representation instead costs ``nb · ns · ngkmax`` per rank, which is
~6-11% of the box for typical GW grids.

This rewrite (P4c) replaces the legacy g_box host-cache with a G-flat
staging pipeline:

* :class:`PsiGStore` stores per-rank tiles of shape
  ``(nk, nb_local, ns, ngkmax)`` instead of ``(nk, nb_local, ns, nx,
  ny, nz)``.
* :meth:`PsiGStore._slice_local_tile_bc` returns one bc's per-rank
  band slab via ``io_callback``, padded to ``(nk, _bpd_max, ns,
  ngkmax)`` so the enclosing ``lax.scan`` body sees a static return
  shape.  :func:`isdf.core.build_psi_r_cache_sm` iterates band chunks via
  ``lax.scan`` inside its ``shard_map`` body, pulling one bc per iteration
  via the slicer.  The resulting ψ(r) cache is band-flat-sharded over the
  full mesh.

The store populates once, feeds the one-time ψ(r) cache build, and is
released before the r-chunk loop.  Its host footprint is
``nk · nb_total · ns · ngkmax · 16 / P`` bytes per process during that
build only.

The reader adapters (legacy h5py vs phdf5) collapse to a single
:class:`wfn_loader.WfnLoader` whose ``backend='auto'`` picks the
right path.
"""
from __future__ import annotations

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from common.shard_map import shard_map
from common.wfn_layout import band_sphere_spec
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from runtime.padding import spec_divisor


def _zero_user_band_pad_in_shard(
    shard_data: np.ndarray,
    *,
    bc_range: tuple[int, int],
    shard_band_slice: slice,
    user_band_stop: int,
) -> np.ndarray:
    """Zero locally-owned padded bands inside one ψ(G-flat) shard.

    ``Meta.b_id_4`` may be larger than the user's requested ``nband`` so
    the band axis divides the device mesh.  The loader can still return
    real DFT coefficients for those padded slots when the WFN file has
    enough bands.  The centroid loader zeros them after extraction; this
    helper applies the same contract to the host ψ(G-flat) cache used
    by the r-chunk ζ fit.
    """
    b0, _ = (int(bc_range[0]), int(bc_range[1]))
    s0 = 0 if shard_band_slice.start is None else int(shard_band_slice.start)
    s1 = shard_data.shape[1] if shard_band_slice.stop is None else int(shard_band_slice.stop)
    step = 1 if shard_band_slice.step is None else int(shard_band_slice.step)
    if step != 1:
        raise ValueError(
            f"ψ(G-flat) shard band slice must be contiguous; got {shard_band_slice!r}")

    local_global_bands = b0 + np.arange(s0, s1, dtype=np.int64)
    pad_mask = local_global_bands >= int(user_band_stop)
    if not np.any(pad_mask):
        return shard_data

    out = np.array(shard_data, copy=True)
    out[:, pad_mask, :, :] = 0.0
    return out


def _mesh_device_coords(mesh: Mesh) -> dict:
    """Map ``id(device) → (x_idx, y_idx)`` for every device in the mesh."""
    coords = {}
    devs = np.asarray(mesh.devices)
    for idx, dev in np.ndenumerate(devs):
        coords[id(dev)] = tuple(int(i) for i in idx)
    return coords


def assert_band_chunks_divisible(band_chunk_ranges, world_size: int) -> None:
    """Refuse a band chunk whose width the per-rank tile would floor-divide.

    THE FLOOR DIVISION IS THE DEFECT SITE, so this is the refusal site.
    :class:`PsiGStore` splits each chunk's bands across all ``P`` ranks as
    ``(b_hi - b_lo) // P``; a width that is not a multiple of ``P`` loses
    ``width % P`` bands right there, in the store's own band accounting,
    and every consumer downstream then works on a store that is short
    those bands with the right shape, the right dtype and no other
    symptom.

    MEASURED (JID 57187694,
    ``reports/zeta_residue_2026-08-17/evidence/baseline_p4.log``): on the
    80 Ry scalar-Si deck the P=4 baseline's logical 50-band window split
    16+16+16+2; the last chunk gives ``bpd = 2 // 4 = 0`` and its two
    bands contributed nothing to ``z_q`` — at rc=0.

    ``gw.isdf_fitting`` fixed the PRODUCTION path on 2026-08-17 by padding
    the transport range up to a ``P`` multiple (``_bfe_transport``), and
    ``isdf.core.z_q_from_psi_sm`` carries an equivalent consumer-side
    check.  Neither makes this one redundant: a guard living in ONE
    consumer is a guard the next consumer does not have, and the padding
    lives in ONE producer.  This function is on the object that performs
    the division, which is the only place every caller must pass through.

    ``ValueError`` and not ``assert``: the fix is a user input key
    (``band_chunk_size``), and an assert vanishes under ``python -O``,
    re-arming exactly the silent band-dropping it guards.
    """
    p = int(world_size)
    if p <= 0:
        raise ValueError(f"world_size must be positive, got {world_size!r}")
    bad = [(i, int(b_lo), int(b_hi))
           for i, (b_lo, b_hi) in enumerate(band_chunk_ranges)
           if (int(b_hi) - int(b_lo)) % p]
    if not bad:
        return
    detail = "; ".join(
        f"chunk {i} = [{lo}, {hi}) is {hi - lo} bands, which floor-divides "
        f"to {(hi - lo) // p} per rank and drops {(hi - lo) % p}"
        for i, lo, hi in bad)
    raise ValueError(
        f"PsiGStore: band-chunk width is not divisible by the world size "
        f"{p}, so the per-rank band tile would silently drop the "
        f"remainder.  {detail}.  Set band_chunk_size to a multiple of {p}, "
        f"or pad the transport range up to one the way gw.isdf_fitting "
        f"does (`_bfe_transport`) and zero the non-physical tail bands.")


class PsiGStore:
    """Host-resident ψ(G-flat) staging for ``build_psi_r_cache_sm``.

    Per locally-addressable mesh cell ``(x, y)`` owns one contiguous
    host tile of shape ``(nk, nb_local, ns, ngkmax)``.  The band axis
    inside each tile is ordered by band-chunk (bc) — block 0 holds
    bc 0's local bands, block 1 holds bc 1's local bands, and so on.
    For CrI3-scale, the new shape is ~14× smaller than the legacy
    g_box ``(nk, nb_local, ns, nx, ny, nz)`` shape.

    :meth:`_slice_local_tile_bc` is the per-iter host-tile slicer used
    by the ``io_callback`` inside the cache builder's ``lax.scan`` body.
    It returns one bc's per-rank slab padded to
    ``(nk, _bpd_max, ns, ngkmax)`` so the scan body sees a static
    output shape every iteration.
    """

    def __init__(
        self,
        *,
        loader,
        mesh_xy: Mesh,
        band_chunk_ranges: tuple[tuple[int, int], ...],
        meta,
        bispinor: bool = False,
    ):
        self.loader = loader
        self.mesh = mesh_xy
        self.band_chunk_ranges = tuple(tuple(bc) for bc in band_chunk_ranges)
        self.meta = meta
        self.bispinor = bool(bispinor)

        nk = int(meta.nk_tot)
        ns = int(meta.nspinor)
        ngkmax = int(loader.ngkmax)
        p = spec_divisor(mesh_xy, band_sphere_spec(), axis=1)

        # Per-bc local band count: bands_per_device for ONE bc.  Used to
        # compute the per-rank tile's band-axis offsets (bc-stacked
        # ordering); the per-rank tile's full band axis is contiguous
        # across all bcs and lives at ``self._per_rank_shape[1]``.
        assert_band_chunks_divisible(self.band_chunk_ranges, p)
        bpd_per_bc = [(b_hi - b_lo) // p for (b_lo, b_hi) in self.band_chunk_ranges]
        self._bpd_per_bc = tuple(bpd_per_bc)
        # Padded uniform per-bc local band count — used by
        # ``_slice_local_tile_bc`` so an ``io_callback`` inside a
        # ``lax.scan`` body sees a static return shape regardless of
        # which bc the traced index resolves to.  Round 6 Phase 2
        # restoration of the field originally added in commit
        # ``cdd0fba`` (deleted in ``5cadd4b`` when the flat-axis path
        # took over).  ``io_callback`` REQUIRES static ``out_sds`` at
        # trace time; ``_bpd_max`` is the closure-static value the
        # caller closes into ``ShapeDtypeStruct``.
        self._bpd_max = max(bpd_per_bc) if bpd_per_bc else 0
        offsets = [0]
        for bpd in bpd_per_bc:
            offsets.append(offsets[-1] + bpd)
        self._bc_band_offsets = tuple(offsets)
        self._nb_local = offsets[-1]
        self._per_rank_shape = (nk, self._nb_local, ns, ngkmax)

        self._dtype = jnp.complex128
        self._coords = _mesh_device_coords(mesh_xy)
        # host_tiles[(x, y)] = one contiguous numpy array of shape
        # _per_rank_shape, populated once below.
        self._host_tiles: dict = {}

        # Cache the box index (g_index) and Bloch-phase ingredients on
        # device once — they're shared across every cache-builder callback.
        self._g_index_dev: jax.Array | None = None
        self._kvecs_frac_dev: jax.Array | None = None

        self._populate_from_loader()
        if jax.process_index() == 0:
            tile_gb = self._per_rank_shape_bytes() / 1e9
            print(f"  ψ(G-flat) host cache: {tile_gb:.2f} GB/process resident")

    # ---------------------------------------------------------------------
    # Population — pulls from the WfnLoader, scatters into per-(x,y) tiles.
    # ---------------------------------------------------------------------
    def _populate_from_loader(self) -> None:
        """One ``loader.load(bands=bc)`` per band-chunk, then split the
        returned sharded jax.Array into per-(x, y) tiles on host.

        ``loader.load`` is a collective on the FFI backend or a
        broadcast-then-device-put on the eager backend; either way each
        rank's local shard of the (band-sharded) output is what we
        actually need to copy into the host tile.
        """
        # Allocate tiles on first population.
        for (x, y) in self._coords.values():
            if (x, y) not in self._host_tiles:
                self._host_tiles[(x, y)] = np.empty(
                    self._per_rank_shape, dtype=np.complex128)

        # NOTE: this read is deliberately SYNCHRONOUS.  A prefetching
        # async wfn reader (a worker thread issuing ``loader.load(bc+1)``
        # against bc[i]'s shard_to_host copy) was implemented, measured,
        # and deleted 2026-07-25: at MoS2 3×3 scale xprof shows
        # H2D/compute overlap_frac = 0.000 even with depth-2 prefetch —
        # XLA's stream scheduler does not pipeline our H2D against
        # compute.  If the async-reader story comes back at larger scale
        # (CrI3), rebuild it on ``common.async_io.AsyncDispatcher``, which
        # is still here and drives the SlabIO write side.
        from common import timing
        from common.wfn_transforms import load_psi_gflat_padded
        sharding_spec = band_sphere_spec()
        for bc_idx, bc_range in enumerate(self.band_chunk_ranges):
            bc_start, bc_end = int(bc_range[0]), int(bc_range[1])
            b_lo = self._bc_band_offsets[bc_idx]
            b_hi = self._bc_band_offsets[bc_idx + 1]
            # Past-mnband zero-pad — the cap-at-file-nbands + zero-pad +
            # reshard dance is single-sourced in
            # :func:`common.wfn_transforms.load_psi_gflat_padded` (same
            # contract as ``load_centroids_band_chunked``, commit
            # 2129fad).  ``None`` return = the entire bc range starts
            # at/past ``loader.nbands``: only band-pad rows the
            # ``_zero_user_band_pad_in_shard`` post-step would zero out
            # anyway, AND the per-rank host-tile slice for these rows is
            # empty (``bpd_per_bc[bc] == 0`` since ``nb_total < p`` for
            # the all-pad tail bcs), so skip the load entirely and
            # zero-fill the tile span directly.
            with timing.section("psi_G_store.populate.loader_load"):
                psi_G_bc = load_psi_gflat_padded(
                    self.loader, (bc_start, bc_end), mesh_xy=self.mesh,
                    bispinor=self.bispinor, k="full_bz",
                    sharding=sharding_spec)
                if psi_G_bc is not None:
                    jax.block_until_ready(psi_G_bc)
            if psi_G_bc is None:
                if b_hi - b_lo > 0:
                    for (x, y) in self._coords.values():
                        self._host_tiles[(x, y)][:, b_lo:b_hi, :, :] = 0
                continue
            with timing.section("psi_G_store.populate.shard_to_host"):
                for shard in psi_G_bc.addressable_shards:
                    x, y = self._coords[id(shard.device)]
                    tile = self._host_tiles[(x, y)]
                    shard_band_slice = shard.index[1]
                    data = _zero_user_band_pad_in_shard(
                        np.asarray(shard.data),
                        bc_range=bc_range,
                        shard_band_slice=shard_band_slice,
                        user_band_stop=int(getattr(self.meta, "b_id_4_user", self.meta.b_id_4)),
                    )
                    tile[:, b_lo:b_hi, :, :] = data
            del psi_G_bc  # release device memory before next bc

        # Stage box_index + kvecs once on device; reused across every
        # fetch.  These don't depend on the band range or r-chunk.
        if self._g_index_dev is None:
            # ``WfnLoader.box_index_dev`` deduplicates the device-resident
            # ``(nk, nx, ny, nz) int32`` g_index across every
            # ``psi_G_store`` instance that shares the same loader+mesh.
            # Without this dedupe, every ``fit_zeta_to_h5`` channel
            # (charge + 3 transverse on bispinor) device_put'd a fresh
            # REPLICATED buffer (0.16 GB/rank each), accumulating to
            # ~1.3 GB/rank wasted by V_q time (agent_h §3 Finding 3).
            self._g_index_dev = self.loader.box_index_dev(
                k="full_bz", mesh=self.mesh)
            kgrid = np.asarray(self.meta.kgrid, dtype=np.float64)
            sym = self.loader.symmetry()
            kvecs_frac = np.asarray(
                sym.kvecs_asints, dtype=np.float64) / kgrid[None, :]
            # Process-local placement — see
            # ``common.collectives.device_put_process_local``: on a
            # multi-process mesh ``jax.device_put(numpy, sharding)``
            # fires JAX's hidden ``assert_equal`` all-gather.
            from common.collectives import device_put_process_local
            self._kvecs_frac_dev = device_put_process_local(
                kvecs_frac,
                NamedSharding(self.mesh, P(None, None)))

    def _clear_tiles(self) -> None:
        self._host_tiles.clear()

    # ---------------------------------------------------------------------
    # Per-rank host-tile slice for one bc, padded to a static shape.
    # ---------------------------------------------------------------------
    # Round 6 Phase 2 restoration of the helper originally added in
    # commit ``cdd0fba`` and removed in ``5cadd4b`` when the (now-buggy)
    # flat-axis ``psi_G_device_full`` path took over.  The production consumer
    # is the io_callback inside ``build_psi_r_cache_sm``'s ``lax.scan`` body;
    # ``z_q_from_psi_sm`` retains a compatibility-only direct consumer.
    #
    # Static-shape contract: ``io_callback`` requires its ``out_sds`` to
    # be static at trace time, AND ``lax.scan`` requires the body output
    # shape to be uniform across iters.  ``_bpd_max = max(bpd_per_bc)``
    # is closure-static at ``__init__``; short-final-bc bands are
    # zero-padded to the same shape every iter.  The downstream L/R
    # band-mask zeros out pad rows so they contribute mathematically zero
    # to the pair-density einsum.
    #
    # NOTE the ``np.zeros`` (NOT ``np.empty``) on the pad-row buffer:
    # the math-neutrality of pad rows depends on them being EXACTLY
    # zero.  ``np.empty`` would leave garbage that the L/R mask might
    # zero-out at the einsum but could still pollute IFFT precision.

    def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx) -> np.ndarray:
        """Per-rank host-tile slice for one bc, padded to ``(nk, _bpd_max, ns, ngkmax)``.

        Parameters
        ----------
        x_idx, y_idx
            ``jax.lax.axis_index('x') / ('y')`` int32 scalars (resolved
            to Python ints inside the io_callback host fn).
        bc_idx
            Traced int32 scalar in ``[0, len(band_chunk_ranges))``.

        Returns
        -------
        np.ndarray
            Shape ``(nk, _bpd_max, ns, ngkmax)`` c128.  The first
            ``self._bpd_per_bc[bc]`` band rows hold the real bc data;
            the remaining ``_bpd_max - bpd_per_bc[bc]`` rows are zero
            (math-neutral when consumed under a band mask).

        Lifetime contract: host tiles must remain valid for the full
        duration of the enclosing kernel jit because ``io_callback`` fires
        asynchronously inside ``lax.scan``.  ``isdf_fitting.py`` blocks on
        the completed ψ(r) cache before closing this store.
        """
        x, y, bc = int(x_idx), int(y_idx), int(bc_idx)
        if not 0 <= bc < len(self.band_chunk_ranges):
            raise ValueError(
                f"_slice_local_tile_bc: bc_idx={bc} not in "
                f"[0, {len(self.band_chunk_ranges)})")
        tile = self._host_tiles[(x, y)]
        b_lo = self._bc_band_offsets[bc]
        b_hi = self._bc_band_offsets[bc + 1]
        nk, _, ns, ngkmax = tile.shape
        out = np.zeros((nk, self._bpd_max, ns, ngkmax), dtype=tile.dtype)
        out[:, : b_hi - b_lo, :, :] = tile[:, b_lo:b_hi, :, :]
        return out

    @property
    def g_index(self) -> jax.Array:
        """Replicated ``(nk_tot, nx, ny, nz)`` int32 box-index tensor.

        Staged on device by ``_populate_from_loader``.  Used by
        ``gflat_to_rchunk``.
        """
        if self._g_index_dev is None:
            raise RuntimeError(
                "g_index: store population did not stage the box index")
        return self._g_index_dev

    @property
    def kvecs_frac(self) -> jax.Array:
        """Replicated ``(nk_tot, 3)`` float64 fractional k-vectors."""
        if self._kvecs_frac_dev is None:
            raise RuntimeError(
                "kvecs_frac: store population did not stage k vectors")
        return self._kvecs_frac_dev

    def close(self) -> None:
        """Release all host tiles; the shared loader remains caller-owned."""
        self._clear_tiles()

    def _per_rank_shape_bytes(self) -> int:
        return int(np.prod(self._per_rank_shape)) * 16  # complex128


def build_psi_G_store(
    *,
    wfn,
    mesh_xy: Mesh,
    meta,
    band_chunk_ranges,
    bispinor: bool = False,
) -> PsiGStore:
    """Construct the one ψ(G-flat) host store.

    Single backend choice: :class:`wfn_loader.WfnLoader`.
    ``backend='auto'`` picks the FFI phdf5 path when multi-rank GPU +
    mesh + .so present; falls back to eager h5py otherwise.  CPU and
    single-process tests get the eager path automatically.
    """
    loader = wfn  # reuse top-level WfnLoader; opening a second one would
                  # re-slurp wfns/coeffs into host RAM.
    return PsiGStore(
        loader=loader, mesh_xy=mesh_xy,
        band_chunk_ranges=band_chunk_ranges, meta=meta,
        bispinor=bispinor)

"""Host-resident ψ(G-flat) store with on-demand FFT-to-r-chunk fetches.

The ISDF fit kernel consumes ψ(r) one band-chunk × one r-chunk at a
time inside a Python-unrolled bc-loop.  Holding ψ in full FFT-box
representation on host costs ``nb · ns · nx · ny · nz`` complex128 per
rank — for CrI3-class systems that's tens of GB.  Holding ψ in G-flat
representation instead costs ``nb · ns · ngkmax`` per rank, which is
~6-11% of the box for typical GW grids.

This rewrite (P4c) replaces the legacy g_box host-cache with a
**g_flat host-cache + on-demand to_rchunk** pipeline:

* :class:`PsiGStore` stores per-rank tiles of shape
  ``(nk, nb_local, ns, ngkmax)`` instead of ``(nk, nb_local, ns, nx,
  ny, nz)``.
* :meth:`PsiGStore.psi_G_device_full` (lazy property) pulls the full
  per-rank tile to device once per ``begin_rchunk`` window via a
  shard_map+io_callback.  Downstream consumers
  (:func:`common.wfn_transforms.gflat_to_rchunk`,
  :func:`common.wfn_transforms.gflat_to_rmu`) handle the
  ψ(G) → ψ(rchunk) / ψ(r_μ) transforms in a single shard_map+scan
  over the per-rank ``(nk · nb_local)`` flat axis — the FFT box is
  never materialised as a persistent buffer.

Lifecycle modes are unchanged:

* :class:`HostPsiGStore`   – populate once at construction, keep
                             resident for the full run.  Host
                             footprint = ``nk · nb_total · ns ·
                             ngkmax · 16 / P`` bytes per process.
* :class:`RereadPsiGStore` – ``begin_rchunk`` repopulates; ``end_rchunk``
                             frees.  Zero persistent residency between
                             r-chunks.

The reader adapters (legacy h5py vs phdf5) collapse to a single
:class:`file_io.wfn_loader.WfnLoader` whose ``backend='auto'`` picks the
right path.
"""
from __future__ import annotations

from functools import partial
from typing import Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# Sharding spec for the ψ(G-flat) tile that the production kernel
# consumes after io_callback.  (n_k, nb, ns, ngkmax) with the band axis
# flat-sharded over (x, y).
_PSI_G_FLAT_SPEC = P(None, ('x', 'y'), None, None)


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


class PsiGStore:
    """Host-resident ψ(G-flat) store with on-device FFT-to-r-chunk fetches.

    Per locally-addressable mesh cell ``(x, y)`` owns one contiguous
    host tile of shape ``(nk, nb_local, ns, ngkmax)``.  The band axis
    inside each tile is ordered by band-chunk (bc) — block 0 holds
    bc 0's local bands, block 1 holds bc 1's local bands, and so on.
    For CrI3-scale, the new shape is ~14× smaller than the legacy
    g_box ``(nk, nb_local, ns, nx, ny, nz)`` shape.

    :attr:`psi_G_device_full` is the lazy device-side full ψ(G) tile,
    pulled once per ``begin_rchunk`` window via shard_map+io_callback.
    Downstream consumers (:func:`common.wfn_transforms.gflat_to_rchunk`,
    :func:`common.wfn_transforms.gflat_to_rmu`) take that tensor + the
    cached ``g_index`` / ``kvecs_frac`` and produce ψ(rchunk) / ψ(r_μ)
    inside a single shard_map+scan body.  The FFT box never lives as a
    persistent buffer.
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
        p = int(mesh_xy.shape['x']) * int(mesh_xy.shape['y'])

        # Per-bc local band count: bands_per_device for ONE bc.  Used to
        # compute the per-rank tile's band-axis offsets (bc-stacked
        # ordering); the per-rank tile's full band axis is contiguous
        # across all bcs and lives at ``self._per_rank_shape[1]``.
        bpd_per_bc = [(b_hi - b_lo) // p for (b_lo, b_hi) in self.band_chunk_ranges]
        offsets = [0]
        for bpd in bpd_per_bc:
            offsets.append(offsets[-1] + bpd)
        self._bc_band_offsets = tuple(offsets)
        self._nb_local = offsets[-1]
        self._per_rank_shape = (nk, self._nb_local, ns, ngkmax)

        self._dtype = jnp.complex128
        self._coords = _mesh_device_coords(mesh_xy)
        # host_tiles[(x, y)] = one contiguous numpy array of shape
        # _per_rank_shape, or absent before begin_rchunk fills it.
        self._host_tiles: dict = {}

        # Cache the box index (g_index) and Bloch-phase ingredients on
        # device once — they're shared across every fetch_psi_rchunk
        # call regardless of which band-chunk or r-chunk is asked for.
        self._g_index_dev: jax.Array | None = None
        self._kvecs_frac_dev: jax.Array | None = None
        # Lazy device-side full ψ(G) tensor (Path D integration: pulled
        # once per begin_rchunk window, consumed by ``gflat_to_rchunk``).
        # Invalidated whenever host tiles change (``_clear_tiles``).
        self._psi_G_device_full: jax.Array | None = None

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

        # NOTE: an AsyncWfnReader (file_io/wfn_loader.py) is available
        # and was tried here to pipeline ``loader.load(bc+1)`` against
        # bc[i]'s shard_to_host copy.  At MoS2 3×3 scale, xprof shows
        # H2D/compute overlap_frac = 0.000 even with depth-2 prefetch
        # — XLA's stream scheduler doesn't pipeline our H2D against
        # compute here.  Keep the synchronous path until either scale
        # grows (CrI3) or the broader async-reader story comes back.
        from common import timing
        sharding_spec = P(None, ('x', 'y'), None, None)
        for bc_idx, bc_range in enumerate(self.band_chunk_ranges):
            with timing.section("psi_G_store.populate.loader_load"):
                psi_G_bc = self.loader.load(
                    bands=bc_range, k="full_bz",
                    sharding=sharding_spec,
                    bispinor=self.bispinor,
                )
                jax.block_until_ready(psi_G_bc)
            b_lo = self._bc_band_offsets[bc_idx]
            b_hi = self._bc_band_offsets[bc_idx + 1]
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
            rep = NamedSharding(self.mesh, P(*([None] * 4)))
            # Pass numpy directly (no ``jnp.asarray`` wrap): the wrap
            # makes a single-device jax.Array first, then device_put
            # broadcasts to replicated via an all-reduce.  Numpy input
            # to device_put(replicated) takes the per-process-local
            # placement path — no comm.
            self._g_index_dev = jax.device_put(
                self.loader.box_index(k="full_bz"), rep)
            kgrid = np.asarray(self.meta.kgrid, dtype=np.float64)
            sym = self.loader._ensure_sym()
            kvecs_frac = np.asarray(
                sym.kvecs_asints, dtype=np.float64) / kgrid[None, :]
            self._kvecs_frac_dev = jax.device_put(
                kvecs_frac,
                NamedSharding(self.mesh, P(None, None)))

    def _clear_tiles(self) -> None:
        self._host_tiles.clear()
        # Invalidate the lazy device-side full ψ(G) — its host source is gone.
        self._psi_G_device_full = None

    # ---------------------------------------------------------------------
    # Lazy full ψ(G) device tensor — Path D consumer entry point.
    # ---------------------------------------------------------------------
    @property
    def psi_G_device_full(self) -> jax.Array:
        """ψ(G-flat) for ALL bcs, on device, band-axis sharded.

        Global shape ``(nk_tot, nb_total, ns, ngkmax)`` with sharding
        ``P(None, ('x','y'), None, None)``.  ``nb_total = sum(bc_size
        for bc in band_chunk_ranges)``; on each rank the local band
        axis spans the rank's share of the *canonical* band sharding
        (i.e., ``[r·nb_total/P, (r+1)·nb_total/P)`` of the global
        index), NOT the bc-stacked order of the host tile.

        Implementation: pulls each band-chunk's per-rank tile via
        io_callback (one shard_map per bc), then assembles them with
        ``jnp.concatenate(axis=1)``.  The concatenate is what
        re-aligns the per-rank layout from bc-stacked (host order) to
        the canonical band sharding the consumer expects — without
        this step, ``psi_Y_full[:, _l_lo:_l_hi]`` slicing in the
        kernel would pull the wrong bands per rank (rank r's bc-stacked
        local axis mixes bands from many global indices, which is not
        what ``P(None, ('x','y'), …)`` slicing expects).  This mirrors
        the legacy bc-loop + ``jnp.concatenate`` semantics exactly.

        Lazy + cached.  First access triggers the io_callback +
        concatenate pipeline.  Subsequent accesses return the same
        ``jax.Array``.  Invalidated by ``_clear_tiles`` (and therefore
        by ``RereadPsiGStore.end_rchunk``).

        Consumed by ``_make_fit_one_rchunk_kernel._kernel`` via
        ``gflat_to_rchunk(psi_G_device_full, g_index, …)`` — replaces
        the legacy per-bc ``fetch_psi_rchunk`` + concatenate.
        """
        if self._psi_G_device_full is not None:
            return self._psi_G_device_full
        if not self._host_tiles:
            raise RuntimeError(
                "psi_G_device_full: host tiles empty; call begin_rchunk first")

        nk, _, ns, ngkmax = self._per_rank_shape
        tiles = self._host_tiles
        dtype = self._dtype
        offsets = self._bc_band_offsets

        def _make_pull(b_lo: int, b_hi: int):
            per_rank_bc = (nk, b_hi - b_lo, ns, ngkmax)
            out_sds = jax.ShapeDtypeStruct(per_rank_bc, dtype)
            # Bind b_lo, b_hi via default args (avoid Python late-binding).
            def _slice_bc(x_idx, y_idx, _lo=b_lo, _hi=b_hi):
                return tiles[(int(x_idx), int(y_idx))][:, _lo:_hi, :, :]

            @partial(
                shard_map,
                mesh=self.mesh,
                in_specs=(),
                out_specs=_PSI_G_FLAT_SPEC,
                check_rep=False,
            )
            def _pull_bc():
                x_idx = jax.lax.axis_index('x')
                y_idx = jax.lax.axis_index('y')
                return io_callback(
                    _slice_bc, out_sds, x_idx, y_idx, ordered=True)
            return _pull_bc()

        parts = [_make_pull(offsets[i], offsets[i + 1])
                 for i in range(len(offsets) - 1)]
        # ``jnp.concatenate`` of band-sharded arrays inserts the
        # reshuffle that puts the result in canonical band sharding —
        # this is the load-bearing step (see docstring).
        self._psi_G_device_full = jnp.concatenate(parts, axis=1)
        return self._psi_G_device_full

    @property
    def g_index(self) -> jax.Array:
        """Replicated ``(nk_tot, nx, ny, nz)`` int32 box-index tensor.

        Staged on device by ``_populate_from_loader``; raises if accessed
        before any ``begin_rchunk`` call.  Used by ``gflat_to_rchunk``.
        """
        if self._g_index_dev is None:
            raise RuntimeError(
                "g_index: not staged; call begin_rchunk (or use a populated "
                "store) first")
        return self._g_index_dev

    @property
    def kvecs_frac(self) -> jax.Array:
        """Replicated ``(nk_tot, 3)`` float64 fractional k-vectors."""
        if self._kvecs_frac_dev is None:
            raise RuntimeError(
                "kvecs_frac: not staged; call begin_rchunk (or use a populated "
                "store) first")
        return self._kvecs_frac_dev

    # ---------------------------------------------------------------------
    # Lifecycle — subclasses override
    # ---------------------------------------------------------------------
    def begin_rchunk(self, r_start: int, r_end: int) -> None:
        """Called by the Python driver before the fit_one_rchunk jit
        runs.  Default: no-op (tiles populated once at construction).
        ``RereadPsiGStore`` overrides to refresh the tiles."""

    def end_rchunk(self) -> None:
        """Called by the driver AFTER ``block_until_ready`` on the jit
        output.  Default: no-op.  ``RereadPsiGStore`` overrides to free
        host tiles before the next r-chunk."""

    def close(self) -> None:
        """Release all host tiles and drop the loader reference."""
        self._clear_tiles()

class HostPsiGStore(PsiGStore):
    """ψ(G-flat) loaded once, kept resident on host for the full run.

    Per-rank footprint: ``nk · nb_total · ns · ngkmax · 16 / P`` bytes.
    For CrI3 80 Ry 6x6 with ngkmax≈70k, 1000 bands, 4 spinor (bispinor),
    16-GPU mesh: ~28 GB / process — fits comfortably on Perlmutter
    HBM80 hosts.  Same system in g_box form would be ~400 GB / process
    (won't fit).  ~14× smaller than the legacy host-resident layout.
    """

    def __init__(self, *, loader, mesh_xy, band_chunk_ranges, meta,
                  bispinor: bool = False):
        super().__init__(
            loader=loader, mesh_xy=mesh_xy,
            band_chunk_ranges=band_chunk_ranges, meta=meta,
            bispinor=bispinor)
        self._populate_from_loader()
        if jax.process_index() == 0:
            tile_gb = self._per_rank_shape_bytes() / 1e9
            print(f"  ψ(G-flat) host cache: {tile_gb:.2f} GB/process resident")

    def _per_rank_shape_bytes(self) -> int:
        return int(np.prod(self._per_rank_shape)) * 16  # complex128


class RereadPsiGStore(PsiGStore):
    """ψ(G-flat) re-read from the loader at every r-chunk; freed between."""

    def begin_rchunk(self, r_start: int, r_end: int) -> None:
        self._populate_from_loader()

    def end_rchunk(self) -> None:
        self._clear_tiles()


def build_psi_G_store(
    *,
    wfn,
    sym,
    mesh_xy: Mesh,
    meta,
    band_chunk_ranges,
    bispinor: bool = False,
    mode: Literal["host_cache", "file_reread"] = "host_cache",
) -> PsiGStore:
    """Construct the ψ(G-flat) store matching ``mode``.

    Single backend choice: :class:`file_io.wfn_loader.WfnLoader`.
    ``backend='auto'`` picks the FFI phdf5 path when multi-rank GPU +
    mesh + .so present; falls back to eager h5py otherwise.  CPU and
    single-process tests get the eager path automatically.

    ``sym`` is kept in the signature for caller-API back-compat but is
    ignored — the loader builds its own ``SymMaps`` lazily from the WFN's
    ``mf_header``.
    """
    del sym
    loader = wfn  # reuse top-level WfnLoader; opening a second one would
                  # re-slurp wfns/coeffs into host RAM.
    if mode == "host_cache":
        return HostPsiGStore(
            loader=loader, mesh_xy=mesh_xy,
            band_chunk_ranges=band_chunk_ranges, meta=meta,
            bispinor=bispinor)
    if mode == "file_reread":
        return RereadPsiGStore(
            loader=loader, mesh_xy=mesh_xy,
            band_chunk_ranges=band_chunk_ranges, meta=meta,
            bispinor=bispinor)
    raise ValueError(
        f"ψ(G-flat) store mode must be 'host_cache' or 'file_reread', got {mode!r}")

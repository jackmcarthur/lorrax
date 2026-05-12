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
* :meth:`PsiGStore.fetch_psi_rchunk` pulls a g_flat slice via
  ``io_callback`` and immediately calls
  :func:`common.wfn_transforms.to_rchunk` on device — the FFT box is
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

    :meth:`fetch_psi_rchunk` pulls a g_flat slice via ``io_callback``
    then immediately calls :func:`common.wfn_transforms.to_rchunk` on
    device.  The FFT box never lives as a persistent buffer — it's a
    transient inside the jit.  Output sharding is band-axis on
    ``('x','y')``; consumers downstream see the same (n_k, nb, ns,
    n_rchunk) layout the legacy ``get_sharded_wfns_rchunk_slice``
    produced.
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

        # Per-bc local band count: bands_per_device for ONE bc.
        bpd_per_bc = [(b_hi - b_lo) // p for (b_lo, b_hi) in self.band_chunk_ranges]
        self._bpd_per_bc = tuple(bpd_per_bc)

        # Within each rank's tile the band axis is stacked in bc-order.
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

        # TODO(perf, scale-only): real async-IO via a dedicated reader
        # thread + ``queue.Queue`` (mirror of
        # ``_slab_io_ffi._dispatch_loop`` at
        # file_io/_slab_io_ffi.py:339-410).  The naive Python-level
        # reorder (issue bc[i+1]'s ``loader.load`` before ``shard_to_host``
        # of bc[i]) does NOT give overlap — the phdf5 FFI's host-side read
        # blocks before returning, confirmed in xprof: H2D overlap_frac
        # stayed 0.000.  A daemon-thread variant (HDF5 releases the GIL
        # during I/O) is the path forward at CrI3 / large-bc scale.
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
            self._g_index_dev = jax.device_put(
                jnp.asarray(self.loader.box_index(k="full_bz")), rep)
            kgrid = np.asarray(self.meta.kgrid, dtype=np.float64)
            sym = self.loader._ensure_sym()
            kvecs_frac = np.asarray(
                sym.kvecs_asints, dtype=np.float64) / kgrid[None, :]
            self._kvecs_frac_dev = jax.device_put(
                jnp.asarray(kvecs_frac),
                NamedSharding(self.mesh, P(None, None)))

    def _clear_tiles(self) -> None:
        self._host_tiles.clear()

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

    # ---------------------------------------------------------------------
    # In-jit fetch
    # ---------------------------------------------------------------------
    def _bc_index(self, band_range: tuple[int, int]) -> int:
        """Map a global ``band_range`` to its band-chunk index."""
        key = (int(band_range[0]), int(band_range[1]))
        for i, bc in enumerate(self.band_chunk_ranges):
            if bc == key:
                return i
        raise ValueError(
            f"band_range {band_range} not in band_chunk_ranges; "
            f"fetch_psi_rchunk only supports the pre-declared bc ranges")

    def fetch_psi_rchunk(
        self,
        band_range: tuple[int, int],
        r_start_dyn,
        r_chunk_size: int,
        *,
        k_range: tuple[int, int] | None = None,
    ) -> jax.Array:
        """ψ(r-chunk) for the given band-chunk × r-chunk window.

        Pipeline (each step happens inside the same jit fragment so
        XLA can fuse them):

          1. ``io_callback`` pulls each rank's host tile slice
             ``(nk_slice, bc_size_local, ns, ngkmax)`` directly onto
             device, sharded ``P(None, ('x','y'), None, None)``.
          2. :func:`common.wfn_transforms.to_rchunk` scatters to the
             FFT box, IFFTs, applies the per-k Bloch phase
             ``exp(+2πi k·r)`` separably, and slices the flat-r slab
             ``[r_start, r_start + r_chunk_size)``.  Uses
             ``norm='ortho'`` to match the legacy
             :func:`common.load_wfns.get_sharded_wfns_rchunk_slice`
             scale convention (``1/√N``).

        Output shape ``(nk_slice, bc_size, ns, r_chunk_size)`` c128,
        sharded ``P(None, ('x','y'), None, None)``.

        Parameters
        ----------
        band_range
            One of the pre-declared ``band_chunk_ranges``.
        r_start_dyn
            ``int`` or jax scalar tracer; flat-r start index.  Tracer
            is fine — ``to_rchunk`` only requires ``r_chunk_size`` to
            be static.
        r_chunk_size
            Static int — width of the r slab.
        k_range
            Optional ``(k_lo, k_hi)`` for k-chunking; defaults to
            the full k axis.
        """
        from common.wfn_transforms import to_rchunk

        bc_idx = self._bc_index(band_range)
        b_lo = self._bc_band_offsets[bc_idx]
        b_hi = self._bc_band_offsets[bc_idx + 1]
        nk_total = int(self.meta.nk_tot)
        k_lo, k_hi = (0, nk_total) if k_range is None else (
            int(k_range[0]), int(k_range[1]))

        nk_slice = k_hi - k_lo
        bpd = b_hi - b_lo
        ns = self._per_rank_shape[2]
        ngkmax = self._per_rank_shape[3]
        per_rank_out_shape = (nk_slice, bpd, ns, ngkmax)
        dtype = self._dtype
        tiles = self._host_tiles

        def _slice_local_tile(x_idx, y_idx):
            tile = tiles[(int(x_idx), int(y_idx))]
            return tile[k_lo:k_hi, b_lo:b_hi, :, :]

        out_sds = jax.ShapeDtypeStruct(per_rank_out_shape, dtype)

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(),
            out_specs=_PSI_G_FLAT_SPEC,
            check_rep=False,
        )
        def _pull():
            x_idx = jax.lax.axis_index('x')
            y_idx = jax.lax.axis_index('y')
            return io_callback(
                _slice_local_tile, out_sds, x_idx, y_idx, ordered=True)

        psi_G_flat = _pull()

        # Slice the kvecs to match the k_range, then dispatch to_rchunk.
        kvecs_frac = self._kvecs_frac_dev[k_lo:k_hi]
        return to_rchunk(
            psi_G_flat,
            self._g_index_dev[k_lo:k_hi] if k_range is not None
                else self._g_index_dev,
            tuple(int(s) for s in self.meta.fft_grid),
            r_start_dyn, int(r_chunk_size),
            kvecs_frac=kvecs_frac,
            norm="ortho",
        )


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

"""Host-resident ψ(G) store with io_callback-based on-device slicing.

The ISDF fit kernel consumes ψ(G) one band-chunk (and potentially one
k-chunk) at a time inside a Python-unrolled bc-loop.  Keeping the full
ψ(G) tensor on device as a tuple of jit arguments holds every band-chunk
live for the entire jit execution — at Si 10³ scale that's 8 × 422 MiB
of persistent device memory for data most of the kernel isn't even
looking at.

This module stores the full ``(nk, nb_total, ns, nx, ny, nz)`` ψ(G)
tensor on the HOST in per-rank n_XY band-sharded layout — each process's
host memory holds one ``(nk, nb_local, ns, nx, ny, nz)`` contiguous tile
per locally-addressable (x, y) mesh cell, where ``nb_local = nb_total / P``
and ``P = p_x · p_y``.  Inside the jit body, :meth:`PsiGStore.fetch_psi_G`
pulls any ``(band_range, k_range)`` slice onto device by local tile
slicing — no cross-rank communication, no persistent device residency.

Two lifecycle modes:

* :class:`HostPsiGStore`   – one phdf5 read at startup fills the host
                             tiles; reused across every r-chunk of the
                             fit.  Host footprint stays resident for
                             the full run.
* :class:`RereadPsiGStore` – ``begin_rchunk`` repopulates the host
                             tiles via a fresh phdf5 read; ``end_rchunk``
                             frees them.  Zero persistent host residency
                             between r-chunks; trades disk I/O for RAM.

Only the two options above exist — the old device-resident-tuple cache
and phdf5-per-bc paths are gone.  See ``common.isdf_fitting`` for the
sole caller.
"""
from __future__ import annotations

from functools import partial
import os
from typing import Literal

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


# Sharding spec for the ψ(G) tensor that the production kernel expects.
# (nk, nb, ns, nx, ny, nz) with the band axis flat-sharded over (x, y).
_PSI_G_SPEC = P(None, ('x', 'y'), None, None, None, None)


def _mesh_device_coords(mesh: Mesh) -> dict:
    """Map ``id(device) → (x_idx, y_idx)`` for every device in the mesh.

    Used at host-tile build time: each addressable shard of a sharded
    jax.Array carries its own ``shard.device``; we need its (x, y) cell
    so the tile is indexed the same way the in-jit fetch will query it.
    """
    coords = {}
    devs = np.asarray(mesh.devices)
    for idx, dev in np.ndenumerate(devs):
        coords[id(dev)] = tuple(int(i) for i in idx)
    return coords


class PsiGStore:
    """Host-resident ψ(G) store with on-device slice fetches.

    Each locally-addressable mesh cell ``(x, y)`` owns one contiguous
    host tile of shape ``(nk, nb_local, ns, nx, ny, nz)`` where
    ``nb_local = nb_total / P``.  The band axis inside each tile is
    ordered by band-chunk (bc) — block 0 holds bc 0's local bands,
    block 1 holds bc 1's local bands, and so on — so a contiguous
    bc_range maps to a contiguous band slice inside the tile.

    :meth:`fetch_psi_G` produces a sharded jax.Array for any
    ``(band_range, k_range)`` window by slicing each rank's tile
    locally and gathering via ``shard_map`` + ``io_callback``.  No
    cross-rank communication; each rank contributes only its own
    ``(x, y)`` slab.

    Subclasses supply the ``begin_rchunk`` / ``end_rchunk`` lifecycle
    hooks that decide when the tiles are populated/cleared.
    """

    def __init__(
        self,
        *,
        reader,
        mesh_xy: Mesh,
        band_chunk_ranges: tuple[tuple[int, int], ...],
        meta,
    ):
        self.reader = reader
        self.mesh = mesh_xy
        self.band_chunk_ranges = tuple(tuple(bc) for bc in band_chunk_ranges)
        self.meta = meta

        nk = int(meta.nk_tot)
        ns = int(meta.nspinor)
        nx, ny, nz = (int(d) for d in meta.fft_grid)
        p = int(mesh_xy.shape['x']) * int(mesh_xy.shape['y'])

        # Per-bc local band count: bands_per_device for ONE bc.  Each rank
        # gets ``bc_size / P`` bands per bc after sharded read.  Enforced
        # by ``PhdfWfnReader.coeffs_gspace`` (raises if bc_size % P != 0).
        bpd_per_bc = [(b_hi - b_lo) // p for (b_lo, b_hi) in self.band_chunk_ranges]
        self._bpd_per_bc = tuple(bpd_per_bc)

        # Within each rank's tile the band axis is stacked in bc-order:
        # offsets[i] .. offsets[i+1] holds this rank's slab for bc i.
        offsets = [0]
        for bpd in bpd_per_bc:
            offsets.append(offsets[-1] + bpd)
        self._bc_band_offsets = tuple(offsets)
        self._nb_local = offsets[-1]
        self._per_rank_shape = (nk, self._nb_local, ns, nx, ny, nz)

        self._dtype = jnp.complex128
        self._coords = _mesh_device_coords(mesh_xy)
        # host_tiles[(x, y)] = one contiguous numpy array of shape
        # _per_rank_shape, or None before begin_rchunk fills it.
        self._host_tiles: dict = {}

    # ---------------------------------------------------------------------
    # Population (subclasses decide when this is called)
    # ---------------------------------------------------------------------
    def _populate_from_reader(self) -> None:
        """Stack per-bc phdf5 reads into each rank's contiguous host tile.

        One read per band-chunk is collective across all ranks (phdf5
        semantics); we peel off each rank's addressable shard, copy to
        host, and write it into the ``(x, y)`` tile at the bc's offset
        slot.  After the final read every rank's tile is fully populated.
        """
        # Allocate tiles on first population.  Re-use buffers across
        # begin_rchunk calls to avoid repeated alloc/free churn.
        for (x, y) in self._coords.values():
            if (x, y) not in self._host_tiles:
                self._host_tiles[(x, y)] = np.empty(
                    self._per_rank_shape, dtype=np.complex128)

        for bc_idx, bc_range in enumerate(self.band_chunk_ranges):
            psi_G_bc = self.reader.coeffs_gspace(bc_range)
            b_lo = self._bc_band_offsets[bc_idx]
            b_hi = self._bc_band_offsets[bc_idx + 1]
            for shard in psi_G_bc.addressable_shards:
                x, y = self._coords[id(shard.device)]
                tile = self._host_tiles[(x, y)]
                tile[:, b_lo:b_hi, :, :, :, :] = np.asarray(shard.data)
            del psi_G_bc  # release device memory before next bc

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
        """Release all host tiles and drop the reader reference."""
        self._clear_tiles()

    # ---------------------------------------------------------------------
    # In-jit fetch
    # ---------------------------------------------------------------------
    def _bc_index(self, band_range: tuple[int, int]) -> int:
        """Map a global ``band_range`` to its band-chunk index.

        The kernel's bc-loop iterates ``self.band_chunk_ranges`` in
        order; every fetch asks for one of those exact ranges.  We
        look it up rather than accepting arbitrary ranges so the tile
        offset table is definitely correct.  If future callers want
        arbitrary band windows they can extend this lookup.
        """
        key = (int(band_range[0]), int(band_range[1]))
        for i, bc in enumerate(self.band_chunk_ranges):
            if bc == key:
                return i
        raise ValueError(
            f"band_range {band_range} not in band_chunk_ranges; "
            f"fetch_psi_G only supports the pre-declared bc ranges")

    def fetch_psi_G(
        self,
        band_range: tuple[int, int],
        k_range: tuple[int, int] | None = None,
    ) -> jax.Array:
        """Return a sharded jax.Array holding ψ(G) for this slice.

        Layout:
            global shape = ``(nk_slice, bc_size, ns, nx, ny, nz)``
            sharding     = ``P(None, ('x','y'), None, None, None, None)``
            where ``nk_slice = nk`` (full k, default) or ``k_range[1] - k_range[0]``.

        Implementation: each locally-addressable shard runs a
        ``shard_map`` body that ``io_callback``s into the host tile and
        returns the pre-sliced numpy view.  No cross-rank traffic; each
        rank reads only its own ``(x, y)`` tile.

        ``band_range`` must be one of the pre-declared
        ``band_chunk_ranges`` — the kernel's bc-loop only ever asks for
        those.  ``k_range`` is optional and defaults to the full k axis;
        callers can pass a narrower range to support future k-chunking
        without changing this interface.
        """
        bc_idx = self._bc_index(band_range)
        b_lo = self._bc_band_offsets[bc_idx]
        b_hi = self._bc_band_offsets[bc_idx + 1]
        nk_total = int(self.meta.nk_tot)
        k_lo, k_hi = (0, nk_total) if k_range is None else (
            int(k_range[0]), int(k_range[1]))

        nk_slice = k_hi - k_lo
        bpd = b_hi - b_lo
        ns, nx, ny, nz = self._per_rank_shape[2:]
        per_rank_out_shape = (nk_slice, bpd, ns, nx, ny, nz)
        dtype = self._dtype
        tiles = self._host_tiles  # closure capture

        def _slice_local_tile(x_idx, y_idx):
            tile = tiles[(int(x_idx), int(y_idx))]
            return tile[k_lo:k_hi, b_lo:b_hi, :, :, :, :]

        out_sds = jax.ShapeDtypeStruct(per_rank_out_shape, dtype)

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(),
            out_specs=_PSI_G_SPEC,
            check_rep=False,
        )
        def _pull():
            x_idx = jax.lax.axis_index('x')
            y_idx = jax.lax.axis_index('y')
            return io_callback(
                _slice_local_tile, out_sds, x_idx, y_idx, ordered=True)

        return _pull()


class HostPsiGStore(PsiGStore):
    """ψ(G) loaded once, kept resident on host for the full run.

    Best when host RAM can hold ``nb_total · nk · ns · nr · 16 / P``
    bytes per process comfortably.  Per-rank footprint: for Si 10³ this
    is 3.32 GB per process; for MoS2 3×3 it is 0.26 GB.
    """

    def __init__(self, *, reader, mesh_xy, band_chunk_ranges, meta):
        super().__init__(
            reader=reader, mesh_xy=mesh_xy,
            band_chunk_ranges=band_chunk_ranges, meta=meta)
        self._populate_from_reader()
        if jax.process_index() == 0:
            tile_gb = self._per_rank_shape_bytes() / 1e9
            print(f"  ψ(G) host cache: {tile_gb:.2f} GB/process resident")

    def _per_rank_shape_bytes(self) -> int:
        return int(np.prod(self._per_rank_shape)) * 16  # complex128


class RereadPsiGStore(PsiGStore):
    """ψ(G) re-read from phdf5 at every r-chunk; freed between.

    Use when host RAM can't hold the full tile.  Every r-chunk pays one
    full phdf5 collective read; host residency is zero between r-chunks.
    """

    def begin_rchunk(self, r_start: int, r_end: int) -> None:
        self._populate_from_reader()

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
    """Construct the ψ(G) store matching ``mode``.

    The normal multi-GPU path drives reads through
    :class:`common.phdf5_wfn_reader.PhdfWfnReader`.  CPU runs cannot use
    that CUDA-only FFI reader, and small developer environments may not
    have its native dependencies loaded, so ``host_cache`` falls back to
    the canonical h5py ``WFNReader`` path when PHDF5 is unavailable.
    """
    reader = _build_gspace_reader(
        wfn=wfn, sym=sym, meta=meta, mesh_xy=mesh_xy, bispinor=bispinor)
    if mode == "host_cache":
        return HostPsiGStore(
            reader=reader, mesh_xy=mesh_xy,
            band_chunk_ranges=band_chunk_ranges, meta=meta)
    if mode == "file_reread":
        return RereadPsiGStore(
            reader=reader, mesh_xy=mesh_xy,
            band_chunk_ranges=band_chunk_ranges, meta=meta)
    raise ValueError(
        f"ψ(G) store mode must be 'host_cache' or 'file_reread', got {mode!r}")


class _LegacyGspaceReader:
    """Adapter exposing ``coeffs_gspace`` via ``read_Gvecs_to_devices``."""

    def __init__(self, *, wfn, sym, meta, mesh_xy: Mesh, bispinor: bool):
        self.wfn = wfn
        self.sym = sym
        self.meta = meta
        self.mesh_xy = mesh_xy
        self.bispinor = bool(bispinor)

    def coeffs_gspace(
        self,
        band_range: tuple[int, int],
        *,
        k_ids=None,
    ) -> jax.Array:
        if k_ids is not None:
            k_ids_np = np.asarray(k_ids, dtype=np.int64)
            if not np.array_equal(k_ids_np, np.arange(int(self.meta.nk_tot))):
                raise ValueError(
                    "legacy ψ(G) fallback only supports full k-point order")
        from common.load_wfns import read_Gvecs_to_devices
        psi_G, _ = read_Gvecs_to_devices(
            self.wfn, self.sym, band_range, self.meta, self.bispinor,
            self.mesh_xy)
        return psi_G

    def close(self) -> None:
        pass


class _Phdf5GspaceReader:
    """Adapter that carries the driver's bispinor flag into PHDF5 reads."""

    def __init__(self, reader, *, bispinor: bool):
        self.reader = reader
        self.bispinor = bool(bispinor)

    def coeffs_gspace(
        self,
        band_range: tuple[int, int],
        *,
        k_ids=None,
    ) -> jax.Array:
        return self.reader.coeffs_gspace(
            band_range, k_ids=k_ids, bispinor=self.bispinor)

    def close(self) -> None:
        # The underlying PHDF5 reader is cached by common.load_wfns and
        # may be shared by other users in this process.
        pass


def _build_gspace_reader(*, wfn, sym, meta, mesh_xy: Mesh, bispinor: bool):
    """Choose the PHDF5 reader when usable, otherwise the h5py fallback."""
    force = str(os.environ.get("LORRAX_GSPACE_READER", "auto")).strip().lower()
    if force not in {"auto", "phdf5", "legacy"}:
        raise ValueError(
            "LORRAX_GSPACE_READER must be 'auto', 'phdf5', or 'legacy'")
    if force == "legacy" or (force == "auto" and jax.default_backend() == "cpu"):
        return _LegacyGspaceReader(
            wfn=wfn, sym=sym, meta=meta, mesh_xy=mesh_xy, bispinor=bispinor)

    wfn_path = getattr(wfn, "_filename", None)
    if wfn_path is None:
        if force == "phdf5":
            raise ValueError(
                "ψ(G) PHDF5 reader requires wfn._filename, but it is unset")
        return _LegacyGspaceReader(
            wfn=wfn, sym=sym, meta=meta, mesh_xy=mesh_xy, bispinor=bispinor)

    try:
        from common.load_wfns import _get_phdf5_reader
        return _Phdf5GspaceReader(
            _get_phdf5_reader(wfn_path, mesh_xy), bispinor=bispinor)
    except (FileNotFoundError, OSError):
        if force == "phdf5":
            raise
        return _LegacyGspaceReader(
            wfn=wfn, sym=sym, meta=meta, mesh_xy=mesh_xy, bispinor=bispinor)

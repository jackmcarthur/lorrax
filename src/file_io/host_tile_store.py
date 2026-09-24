"""HostTileStore — tile-major, pinned host RAM for a grid of sharded tiles.

One resource: a grid of tiles of ONE global shape, each held as a
``jax.Array`` in ``pinned_host`` memory with the same sharding as the
device tile it came from.  A rank's share of a tile is therefore its own
page-locked buffer, and moving a tile is one contiguous DMA per rank —
no ``np.ascontiguousarray``, no strided host copy, no first-touch page
faults on a huge ``np.zeros``.

    store = HostTileStore(mesh=mesh, grid=(n_Gt, n_batch),
                          tile_shape=(Q, b, G_tile), spec=P(None, XY, None),
                          dtype=np.complex128)
    store.put_tile_async((t, beta), z_tile)     # D2H; returns at once
    store.wait()                                # every put has landed
    z = store.get_tile_async((t, beta))         # H2D; z is the future
    store.to_disk(path); HostTileStore.from_disk(path, mesh=mesh, spec=spec)

TRANSFERS ARE ``jax.jit`` IDENTITIES WITH ``out_shardings``, one per memory
kind.  That is the placement route the B4 gate allows outside the I/O
layer (TASTE 3), it moves only the shard each device already owns (same
sharding in and out, so no collective), and it is asynchronous: dispatch
returns in ~0.1 ms while a 2 GiB copy lands in ~110 ms (measured, evidence
``runs/runtime/io_layer_20260923/p4_xfer2``).  So the write-behind queue is
JAX's dispatch queue and the prefetch future is the returned array.

WHAT ASYNC DOES AND DOES NOT BUY (measured, same evidence).  Host-side it
is real: the caller keeps dispatching.  Device-side, a transfer issued as
its own program runs on the compute stream and does NOT overlap a
separate compute program (1 GiB H2D + an independent GEMM chain: 0.318 s
vs 0.324 s serial).  Passing :meth:`host_tile` straight into the consuming
``jit`` lets XLA schedule the copy inside that program, which overlapped
partially (0.288 s vs 0.324 s).  Plan for PCIe time, not for hiding it.

PADDING is the caller's, through :mod:`runtime.padding`: every tile has
the same carrier shape and every sharded axis of it must already be
mesh-divisible under ``spec``.  An uneven logical extent is padded to its
carrier with ``padded_axis``/``pad_to_axis`` before the put and stripped
with ``strip_axis`` after the get; the constructor authenticates the
carrier and refuses otherwise.

RATES, P4 A100, per rank (``p4_tilesize``, ``p4_putprobe``, ``p4_tilestore``):
D2H and H2D 24-26 GB/s from 32 MiB to 4 GiB tiles, which is the PCIe floor.
One exception: the FIRST fill of the pinned pool runs at ~5.5-5.8 GB/s,
because XLA's pinned-host allocator page-locks new memory as it grows.  The
pool is kept per process, so later stores reuse it at full rate.  Today's
pageable ZStore host tier measures 5.3 GB/s writing (first touch of its
``np.zeros``) and 13.4-13.8 GB/s reading.  This store measures 5.8 GB/s
writing into a cold pool and 24-25 GB/s reading.

PINNED MEMORY IS PAGE-LOCKED NODE RAM.  It comes from XLA's pinned-host
allocator, per process, and four ranks share a node's DRAM.  Replacing a
tile briefly holds the old and the new buffer.  The store reports
:attr:`nbytes_per_rank`; placement (host vs disk) is the memory planner's
decision, not this module's.

Disk spill goes through :mod:`file_io.slab_io` and nothing else: dataset
``tiles`` of shape ``(*grid, *tile_shape)``, one hyperslab per tile.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from runtime.padding import authenticate_padded_axis, spec_divisor

__all__ = ["HostTileStore"]

_HOST = "pinned_host"
_DEVICE = "device"


@lru_cache(maxsize=None)
def _mover(mesh, spec, memory_kind: str):
    """``jit`` identity landing a tile in ``memory_kind``, sharded ``spec``."""
    # ponytail: no thread pool and no custom future; JAX's async dispatch is
    # the write-behind queue and the returned jax.Array is the prefetch handle.
    out = NamedSharding(mesh, spec, memory_kind=memory_kind)
    return jax.jit(lambda a: a, out_shardings=out)


@lru_cache(maxsize=None)
def _movers(mesh, spec, memory_kind: str, n: int):
    """One ``jit`` identity moving ``n`` tiles at once (one dispatch)."""
    out = NamedSharding(mesh, spec, memory_kind=memory_kind)
    return jax.jit(lambda ts: ts, out_shardings=tuple(out for _ in range(n)))


@lru_cache(maxsize=None)
def _lead_axes(mesh, spec, g: int):
    """Device-side reshape between a tile and its ``(1,)*g + tile`` hyperslab.

    Returns ``(add, drop)``; both keep every rank's shard in place.
    """
    add = jax.jit(lambda a: a.reshape((1,) * g + a.shape),
                  out_shardings=NamedSharding(mesh, P(*((None,) * g), *spec)))
    drop = jax.jit(lambda a: a.reshape(a.shape[g:]),
                   out_shardings=NamedSharding(mesh, spec))
    return add, drop


class HostTileStore:
    """A grid of equal sharded tiles held in pinned host memory.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        The run's device mesh; every tile is sharded over it.
    grid : tuple[int, ...]
        Tile-grid extents, e.g. ``(n_G_tiles, n_mu_batches)``.  A tile index
        is a tuple in this grid (a bare int for a 1-D grid).
    tile_shape : tuple[int, ...]
        GLOBAL shape of one tile (the padded carrier).  Every axis that
        ``spec`` shards must be divisible by its mesh divisor.
    spec : PartitionSpec
        Sharding of one tile, identical on device and on host.
    dtype
        Element type of every tile.

    COLLECTIVE only in the sense every SPMD placement is: all ranks put and
    get the same indices in the same order.  No transfer crosses a process.
    """

    def __init__(self, *, mesh, grid, tile_shape, spec, dtype):
        self.mesh = mesh
        self.grid = tuple(int(g) for g in (grid if np.ndim(grid) else (grid,)))
        self.tile_shape = tuple(int(s) for s in tile_shape)
        self.spec = P(*spec)
        self.dtype = np.dtype(dtype)
        if len(tuple(self.spec)) > len(self.tile_shape):
            raise ValueError(
                f"HostTileStore: spec {self.spec} has more entries than "
                f"tile_shape {self.tile_shape}")
        # ponytail: one carrier shape for every tile; ragged tiles are padded
        # by the caller through runtime.padding rather than stored ragged.
        for ax, n in enumerate(self.tile_shape):
            authenticate_padded_axis(
                n, n, spec_divisor(mesh, self.spec, ax),
                name=f"HostTileStore tile axis {ax}")
        self._dev_sharding = NamedSharding(mesh, self.spec)
        self._tiles: dict[tuple[int, ...], jax.Array] = {}
        self._pending: list[jax.Array] = []

    # -- indexing ---------------------------------------------------------
    def _key(self, idx) -> tuple[int, ...]:
        key = tuple(int(i) for i in (idx if np.ndim(idx) else (idx,)))
        if len(key) != len(self.grid) or not all(
                0 <= i < g for i, g in zip(key, self.grid)):
            raise IndexError(
                f"HostTileStore: tile index {idx!r} is outside grid "
                f"{self.grid}")
        return key

    @property
    def tile_bytes_per_rank(self) -> int:
        """Bytes of one tile held by one rank (its local shard)."""
        n = int(np.prod(self.tile_shape)) * self.dtype.itemsize
        div = 1
        for ax in range(len(self.tile_shape)):
            div *= spec_divisor(self.mesh, self.spec, ax)
        return n // div

    @property
    def nbytes_per_rank(self) -> int:
        """Pinned host bytes this rank holds now (tiles written so far)."""
        return len(self._tiles) * self.tile_bytes_per_rank

    # -- transfers --------------------------------------------------------
    def put_tile_async(self, idx, tile: jax.Array) -> None:
        """Copy device ``tile`` into slot ``idx``; returns before it lands.

        ``tile`` must have exactly ``tile_shape``, ``dtype`` and the store's
        sharding: a different sharding would make the copy a reshard (a
        collective), so it is refused rather than performed.  The caller may
        drop ``tile`` at once; JAX keeps its buffer alive until the copy is
        done.  :meth:`wait` blocks until every put has landed.
        """
        key = self._key(idx)
        if (tuple(tile.shape) != self.tile_shape
                or np.dtype(tile.dtype) != self.dtype):
            raise ValueError(
                f"HostTileStore.put_tile_async({key}): got "
                f"{tuple(tile.shape)} {np.dtype(tile.dtype)}, want "
                f"{self.tile_shape} {self.dtype}")
        if not tile.sharding.is_equivalent_to(self._dev_sharding, tile.ndim):
            raise ValueError(
                f"HostTileStore.put_tile_async({key}): tile sharding "
                f"{tile.sharding} is not the store's {self._dev_sharding}; "
                f"reshard at the producer, where the cost is visible.")
        h = _mover(self.mesh, self.spec, _HOST)(tile)
        self._tiles[key] = h
        self._pending.append(h)

    def get_tile_async(self, idx) -> jax.Array:
        """Slot ``idx`` on device, under ``spec``.  Returns before it lands.

        The returned ``jax.Array`` is the future: use it, or
        ``block_until_ready`` it.  Issue the next tile's get before
        consuming this one to keep the host side busy.
        """
        return _mover(self.mesh, self.spec, _DEVICE)(self.host_tile(idx))

    def get_tiles_async(self, idxs) -> tuple:
        """Slots ``idxs`` on device in ONE dispatch (a G tile's every batch):
        the per-call latency of :meth:`get_tile_async` paid once."""
        tiles = tuple(self.host_tile(i) for i in idxs)
        return _movers(self.mesh, self.spec, _DEVICE, len(tiles))(tiles)

    def host_tile(self, idx) -> jax.Array:
        """The pinned-host ``jax.Array`` in slot ``idx`` (no copy).

        Pass it straight into a consuming ``jit`` to let XLA schedule the
        H2D inside that program (see the module docstring).
        """
        key = self._key(idx)
        try:
            return self._tiles[key]
        except KeyError:
            raise KeyError(
                f"HostTileStore: tile {key} was never put "
                f"({len(self._tiles)} of {int(np.prod(self.grid))} "
                f"written)") from None

    def wait(self) -> None:
        """Block until every :meth:`put_tile_async` so far has landed."""
        pending, self._pending = self._pending, []
        jax.block_until_ready(pending)

    def close(self) -> None:
        """Release every pinned tile."""
        self.wait()
        self._tiles.clear()

    # -- disk spill (slab_io only) ---------------------------------------
    def to_disk(self, path) -> None:
        """Write every tile to ``path`` as dataset ``tiles (*grid, *tile_shape)``.

        COLLECTIVE (a SlabIO file).  Refuses an incomplete store.
        """
        from file_io.slab_io import SlabIO
        n_all = int(np.prod(self.grid))
        if len(self._tiles) != n_all:
            raise ValueError(
                f"HostTileStore.to_disk: {len(self._tiles)} of {n_all} tiles "
                f"written; refusing to spill a partial store.")
        g = len(self.grid)
        # ponytail: each tile goes host->device->(SlabIO's own D2H)->disk; the
        # extra PCIe hop is ~20 GB/s against a disk path of a few GB/s.
        add, _ = _lead_axes(self.mesh, self.spec, g)
        with SlabIO(path, mode="w", mesh=self.mesh) as io:
            io.create_dataset("tiles", shape=self.grid + self.tile_shape,
                              dtype=self.dtype)
            io.write_attr("tile_grid", np.asarray(self.grid, np.int64))
            io.write_attr("tile_shape", np.asarray(self.tile_shape, np.int64))
            for key in np.ndindex(*self.grid):
                io.write_slab("tiles", add(self.get_tile_async(key)),
                              offset=key + (0,) * len(self.tile_shape))

    @classmethod
    def from_disk(cls, path, *, mesh, spec) -> "HostTileStore":
        """Rebuild a store written by :meth:`to_disk`, sharded ``spec``."""
        from file_io.slab_io import SlabIO
        with SlabIO(path, mode="r", mesh=mesh) as io:
            grid = tuple(int(v) for v in io.read_small("tile_grid"))
            tile_shape = tuple(int(v) for v in io.read_small("tile_shape"))
            g = len(grid)
            lead = (None,) * g
            store = None
            _, drop = _lead_axes(mesh, P(*spec), g)
            to_host = _mover(mesh, P(*spec), _HOST)
            for key in np.ndindex(*grid):
                raw = io.read_slab(
                    "tiles", shape=(1,) * g + tile_shape,
                    offset=key + (0,) * len(tile_shape),
                    mesh=mesh, partition_spec=P(*lead, *spec))
                if store is None:
                    store = cls(mesh=mesh, grid=grid, tile_shape=tile_shape,
                                spec=spec, dtype=raw.dtype)
                store._tiles[key] = to_host(drop(raw))
                store._pending.append(store._tiles[key])
        store.wait()
        return store

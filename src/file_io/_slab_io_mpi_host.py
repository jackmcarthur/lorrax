"""MPI-IO SlabIO backend — host-side parallel HDF5 via mpi4py + h5py.

Spiritual equivalent of :mod:`file_io._slab_io_ffi` for the JAX CPU
backend.  Each rank writes its own hyperslab via collective MPI-IO
through h5py(parallel) + mpi4py — no allgather, no rank-0 bottleneck.

Architecture deliberately parallels :class:`_FfiBackend` *except* for
the async writer thread.  We do writes synchronously on the main
thread because:

1. There's no D2H to overlap (the FFI's worker thread exists to
   overlap CUDA memcpy with H5Dwrite; on CPU XLA the "device" memory
   IS host memory, no copy needed).
2. Cray MPICH defaults to ``MPI_THREAD_SINGLE``; doing MPI-IO calls
   from a worker thread then ``H5Fclose`` on the main thread deadlocks
   because MPI's internal state is rank-thread-specific.  Forcing
   ``MPI_THREAD_MULTIPLE`` via ``mpi4py.rc.thread_level = "multiple"``
   would let us mirror the FFI's threaded design, but the perf win
   isn't worth the added complexity at this scale.

What we DO match from :class:`_FfiBackend`:

* Same Lustre prestripe on rank 0 before any rank opens the file.
* Same per-rank hyperslab arithmetic derived from JAX's
  ``Array.addressable_shards[i].index`` (a tuple of ``slice`` per dim
  giving the global slab of each local shard).
* Same ``valid_shape`` clipping for arrays padded for even sharding.
* Same deferred ``write_attr`` mechanism for tiny replicated metadata.

The only thing this backend doesn't do that :class:`_FfiBackend` does
is the cudaMemcpy D2H — on JAX's CPU backend the "device" memory IS
host memory, so the local shard is already in numpy-addressable RAM
and we hand the pointer straight to h5py.

Selected by :class:`gw.gw_config.SlabIOBackend.PHDF5_HOST`.  The
``gw_config`` auto-router maps ``use_ffi_io=true`` on the CPU backend
to this module; on the GPU backend the FFI path is preferred.
"""
from __future__ import annotations

import os
from typing import Sequence

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ._slab_io_ffi import (
    _barrier,
    _lustre_prestripe,
    _normalize_slab_request,
    _normalize_valid_shape,
    _rank0,
)


def _local_shard_and_global_offset(
    A: jax.Array,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return ``(local_numpy, global_offset)`` for the process-local shard.

    LORRAX runs one JAX device per process under multi-process (mesh on
    ``mesh_xy``), so each process has exactly one addressable shard.

    The shard's ``.index`` is a tuple of ``slice`` objects giving the
    GLOBAL start/stop along each axis.  Slabs are always contiguous
    along each axis (no broadcast tiling) so ``.start`` is the offset
    within A.shape.  Replicated axes give ``slice(0, A.shape[ax])`` —
    every process holds the full axis and writes the same overlapping
    rows; under independent MPI-IO that's a redundant write but
    semantically correct (every rank writes identical bytes).
    """
    shards = A.addressable_shards
    if len(shards) != 1:
        # Multi-device-per-process (e.g. GPU with N visible devices
        # under a single process).  Not the LORRAX CPU mesh-xy regime
        # but worth a clear error rather than silent wrong data.
        raise RuntimeError(
            f"_slab_io_mpi_host expects 1 addressable shard per process; "
            f"got {len(shards)} for A.shape={tuple(A.shape)}.  Did you "
            f"set --xla_force_host_platform_device_count > 1 on a "
            f"multi-process run?")
    shard = shards[0]
    local = np.asarray(shard.data)
    # Replicated axes have ``slice(None, None)`` (no explicit bounds);
    # treat ``start=None`` as 0 (the full-axis slab starts at 0).
    offset = tuple(int(s.start) if s.start is not None else 0
                   for s in shard.index)
    return local, offset


def _clip_shard_to_valid(
    local: np.ndarray,
    shard_offset: tuple[int, ...],
    slab_offset: tuple[int, ...],
    valid_shape: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Clip a local shard to the valid (non-padded) region of the slab.

    Returns ``(clipped_local, dataset_offset)`` or ``None`` if this
    shard is entirely outside the valid region (its data should not be
    written at all).

    Arithmetic:

      For axis ``d`` the shard covers global indices
      ``[shard_offset[d], shard_offset[d] + local.shape[d])`` inside
      the slab; the slab covers dataset indices
      ``[slab_offset[d], slab_offset[d] + valid_shape[d])``; the
      intersection is what we actually write.  Below the valid region
      means ``shard_end <= 0`` (impossible because shard_offset >= 0);
      above means ``shard_start >= valid_shape[d]``, in which case the
      shard is in the padded tail and we skip the write.

    All axes must have non-empty intersection or we return ``None``.
    """
    ndim = local.ndim
    ds_offsets: list[int] = []
    local_slices: list[slice] = []
    for d in range(ndim):
        shard_start_in_slab = shard_offset[d]
        shard_end_in_slab = shard_offset[d] + local.shape[d]
        valid_end_in_slab = valid_shape[d]
        # Intersection within the slab coordinate system:
        write_start_in_slab = shard_start_in_slab
        write_end_in_slab = min(shard_end_in_slab, valid_end_in_slab)
        if write_end_in_slab <= write_start_in_slab:
            return None  # shard fully in padded tail
        # Map slab-local to dataset-global:
        ds_offsets.append(int(slab_offset[d] + write_start_in_slab))
        # Slice into the local shard's first-axis-N elements:
        keep = write_end_in_slab - write_start_in_slab
        local_slices.append(slice(0, keep))
    return local[tuple(local_slices)], tuple(ds_offsets)


# ---------------------------------------------------------------------------
class _MpiHostBackend:
    """Collective MPI-IO SlabIO via parallel h5py + mpi4py.

    The interface mirrors :class:`_slab_io_ffi._FfiBackend` so they
    plug in interchangeably from :class:`SlabIO`.  See module docstring
    for the architectural rationale.
    """

    def __init__(self, path: str, mesh: Mesh, mode: str = "w") -> None:
        # Importing mpi4py inits MPI on first use.  We lazy-import here
        # so file_io.slab_io stays importable when MPI isn't built into
        # the venv (single-process tests, non-MPI smoke runs).
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD

        self.path = path
        self.mesh = mesh
        self.mode = mode

        # Pre-stripe the file on rank 0 so Lustre's per-stripe layout
        # actually matches the MPI-IO hints we'll use below.  Only on
        # 'w'; 'a'/'r' inherit the existing inode's stripe layout.
        # Barrier after so all ranks see the inode before H5Fopen.
        if mode == "w" and _rank0():
            stripe_count = int(os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "16"))
            stripe_size = os.environ.get("LORRAX_PHDF5_STRIPE_SIZE_FS", "4M")
            _lustre_prestripe(path, stripe_count=stripe_count,
                              stripe_size=stripe_size)
        if mode == "w":
            _barrier("slab_io_mpi_host_prestripe")

        h5_mode = {"w": "w", "a": "a", "r": "r"}[mode]
        self._fh = h5py.File(path, h5_mode, driver="mpio", comm=self._comm)

        # write_attr accumulates small (rank-0-only) writes that we
        # defer to close() so they don't interleave with collective
        # MPI-IO on the same file handle.  Same hazard as the FFI
        # path's _deferred_attrs.
        self._deferred_attrs: list[tuple[str, object]] = []

    # ------------------------------------------------------------------
    def create_dataset(
        self,
        name: str,
        *,
        shape: Sequence[int],
        dtype,
        chunks: Sequence[int] | None = None,
        attrs: dict | None = None,
    ) -> None:
        # H5Dcreate is collective; all ranks must reach it before any
        # rank writes.  Synchronous writes elsewhere in this module
        # guarantee that ordering naturally.
        if name in self._fh:
            # idempotent on 'a' mode: respect the existing dataset
            return
        shape_t = tuple(int(s) for s in shape)
        h5_dtype = jnp.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
        # Collective H5Dcreate via h5py.  All ranks participate;
        # parallel HDF5 broadcasts the dataset metadata from rank 0.
        ds = self._fh.create_dataset(
            name, shape=shape_t, dtype=h5_dtype,
            chunks=tuple(chunks) if chunks else None,
        )
        if attrs:
            # h5py attribute write is small + replicated; all ranks
            # write identical bytes, parallel HDF5 dedup's.
            for k, v in attrs.items():
                ds.attrs[k] = v

    def write_attr(self, name: str, value) -> None:
        # Tiny rank-0-only writes — defer to close() to avoid mixing
        # h5py serial writes with active MPI-IO on the same file.
        self._deferred_attrs.append((name, value))

    # ------------------------------------------------------------------
    def write_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
        global_shape: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        dtype=None,
        chunks: Sequence[int] | None = None,
        k_chunk_size: int | None = None,  # noqa: ARG002 — allgather-only knob
    ) -> None:
        """Write A as a per-rank hyperslab of dataset ``name``.

        ``A`` is a sharded ``jax.Array``; each rank pulls its local
        shard via ``A.addressable_shards[0]``, clips it to the
        ``valid_shape`` extent (drops the padded tail), and writes the
        clipped block at ``(slab_offset + shard_offset_within_slab)``
        in the dataset.

        Independent MPI-IO writes — no rank-rank collective synchro
        at write time.  Equivalent to the FFI backend's default
        (``LORRAX_PHDF5_INDEPENDENT=1`` rationale, see
        ``ffi/phdf5/cpp/ctx.h``).
        """
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)

        # Resolve offset / shapes via the same helpers the FFI uses.
        off, slab_shape, gshape = _normalize_slab_request(
            op="write_slab", name=name, offset=offset,
            slab_shape=tuple(A.shape), global_shape=global_shape,
            check_bounds=False)
        vshape = _normalize_valid_shape(
            op="write_slab", name=name, valid_shape=valid_shape,
            slab_shape=slab_shape, offset=off, global_shape=gshape)

        # Ensure dataset exists (collective if it needs creating).
        if name not in self._fh:
            self.create_dataset(name, shape=gshape, dtype=A.dtype,
                                chunks=chunks)

        local, shard_offset = _local_shard_and_global_offset(A)
        clipped = _clip_shard_to_valid(local, shard_offset, off, vshape)
        if clipped is None:
            # This rank's shard is entirely in the padded tail.  Under
            # independent MPI-IO no participation required from this
            # rank for this dataset; collective ops elsewhere still
            # need all-rank entry, which the caller ensures.
            return
        arr, ds_offset = clipped
        dset = self._fh[name]
        slc = tuple(slice(o, o + s) for o, s in zip(ds_offset, arr.shape))
        dset[slc] = arr

    # ------------------------------------------------------------------
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        mesh: Mesh | None = None,
        partition_spec: P | None = None,
        as_numpy: bool = False,
    ) -> jax.Array:
        """Read a hyperslab into a sharded JAX array.

        Each rank reads its own piece from the dataset (independent
        MPI-IO via h5py) and assembles the result via JAX's
        ``make_array_from_single_device_arrays`` (no allgather across
        processes — each process delivers its local block sized to
        match ``partition_spec``).
        """
        mesh = mesh or self.mesh
        if shape is None or dtype is None:
            ds = self._fh[name]
            ds_shape = tuple(int(s) for s in ds.shape)
            ds_dtype = np.dtype(ds.dtype)
            if shape is None:
                shape = ds_shape
            if dtype is None:
                dtype = ds_dtype

        off, read_shape, _ = _normalize_slab_request(
            op="read_slab", name=name, offset=offset,
            slab_shape=tuple(shape), global_shape=None,
            check_bounds=False)
        vshape = _normalize_valid_shape(
            op="read_slab", name=name, valid_shape=valid_shape,
            slab_shape=read_shape, offset=off, global_shape=None)

        if partition_spec is None:
            partition_spec = P(*([None] * len(read_shape)))

        # Build a target sharding so the resulting JAX array advertises
        # the right partition spec; we'll plumb each rank's local
        # block into it via make_array_from_single_device_arrays.
        sharding = NamedSharding(mesh, partition_spec)

        # Probe shard layout by constructing a same-shape zero array,
        # pulling addressable_shards[0].index for THIS rank, then
        # reading exactly that hyperslab from the file.  This is the
        # cheap way to ask JAX "what slab does this rank own?" without
        # re-implementing the partition spec arithmetic.
        proto = jax.device_put(
            jnp.zeros(tuple(read_shape), dtype=jnp.dtype(dtype)),
            sharding,
        )
        shard = proto.addressable_shards[0]
        local_shape = tuple(shard.data.shape)
        # Same None-handling as _local_shard_and_global_offset: replicated
        # axes have slice(None, None).
        shard_offset = tuple(int(s.start) if s.start is not None else 0
                             for s in shard.index)

        # Read the local hyperslab.  Clip to valid_shape: anything in the
        # padded tail is zero-filled by JAX's array assembly.
        clipped = _clip_shard_to_valid(
            np.zeros(local_shape, dtype=np.dtype(dtype)),  # shape carrier
            shard_offset, off, vshape,
        )
        host_local = np.zeros(local_shape, dtype=np.dtype(dtype))
        if clipped is not None:
            _, ds_offset = clipped
            keep_shape = tuple(min(ls, ve - so)
                               for ls, ve, so in zip(local_shape, vshape,
                                                     shard_offset))
            slc = tuple(slice(o, o + s)
                        for o, s in zip(ds_offset, keep_shape))
            # Read into the prefix of host_local; the padded tail stays
            # zeroed out.  h5py.dataset[slc] returns a freshly-allocated
            # ndarray; we copy into host_local's prefix.
            read_block = self._fh[name][slc]
            sel = tuple(slice(0, s) for s in keep_shape)
            host_local[sel] = read_block

        # Assemble per-process locals back into a globally-sharded array.
        device = jax.local_devices()[0]
        local_arr = jax.device_put(host_local, device)
        result = jax.make_array_from_single_device_arrays(
            tuple(read_shape), sharding, [local_arr],
        )
        if as_numpy:
            return np.asarray(host_local)
        return result

    # ------------------------------------------------------------------
    def accumulate_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
    ) -> None:
        """``dset[off:off+A.shape] += A`` via collective RMW.

        Mirrors the FFI backend: each rank reads its hyperslab, adds
        its local A shard, writes back.  Drains before/after to keep
        the read + write rounds non-interleaved.
        """
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)
        off = tuple(offset) if offset is not None else tuple([0] * A.ndim)

        existing = self.read_slab(
            name, shape=tuple(A.shape), dtype=A.dtype,
            offset=off, mesh=self.mesh,
            partition_spec=A.sharding.spec if isinstance(
                A.sharding, NamedSharding) else None,
        )
        self.write_slab(name, existing + A, offset=off, global_shape=None)

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Flush deferred small writes via h5py.  All ranks call
        # create_dataset (it's collective under parallel HDF5); we
        # broadcast the rank-0 value to all ranks so every rank writes
        # the same bytes.
        if self._deferred_attrs:
            for ds_name, value in self._deferred_attrs:
                arr = np.asarray(value) if _rank0() else None
                arr = self._comm.bcast(arr, root=0)
                # Default: top-level dataset.  Could extend to
                # "dset:attr" form if a caller ever needs per-dataset
                # attrs deferred; today no caller does.
                if ds_name not in self._fh:
                    self._fh.create_dataset(ds_name, data=arr)
        # All ranks close — collective MPI-IO H5Fclose.
        self._fh.close()

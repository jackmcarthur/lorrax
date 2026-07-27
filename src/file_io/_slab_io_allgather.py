"""Default SlabIO backend: process_allgather + rank-0 h5py.

Canonical CPU SlabIO path.  ``gw_config.LorraxConfig.from_input_file``
auto-routes ``use_ffi_io=true`` on the JAX CPU backend through this
module (since the phdf5 FFI is CUDA-only at the C++ level — see
``src/ffi/phdf5/cpp/write_ffi.cc``).  No CUDA dependency; works in any
container with h5py + jax installed.

Pattern: every rank gathers the global array via
``jax.experimental.multihost_utils.process_allgather(A, tiled=True)``,
then rank-0 writes the hyperslab via serial h5py.  Slower than the FFI's
collective MPI-IO at production scale (rank-0 disk bandwidth bottleneck,
plus full-array gather memory) but byte-identical output.  The FFI
backend lives in ``_slab_io_ffi.py`` and is selected on the GPU backend
when ``use_ffi_io=true``.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from ._slab_io_ffi import (
    _barrier,
    _normalize_slab_request,
    _normalize_valid_shape,
    _rank0,
)


def _to_host(A: Any) -> np.ndarray:
    """Fully-replicated host ndarray for A, regardless of sharding.

    Dispatch on stable ``jax.Array`` metadata rather than on the gather's
    return shape:

    * Plain numpy: return as-is.
    * Single process (``process_count() == 1``): plain device_get.
    * Multi-process JAX Array, ``A.is_fully_replicated``: every device
      already holds the full array (e.g. a ``SingleDeviceSharding`` lift
      of a numpy on every host).  Skip the gather entirely and return
      ``A.addressable_data(0)``.  This both saves the identity-jit
      reshard cost and avoids the Path-(D) ``(world * N0, *rest)``
      stacking shape that ``process_allgather(tiled=True)`` returns for
      fully-addressable inputs.
    * Multi-process JAX Array, not fully replicated: this is always
      non-fully-addressable under LORRAX's mesh-xy sharding (see
      ``jax.Array`` docstring: "fully replicated is not equal to fully
      addressable; a fully replicated array can span multiple hosts").
      ``process_allgather(tiled=True)`` then takes Path (B) of
      ``_handle_array_process_allgather`` — it identity-jits to ``P()``
      and returns shape exactly ``A.shape``.  No post-process needed.

    See ``reports/.../PROCESS_ALLGATHER_DESIGN_REVIEW_2026-05-20.md`` for
    the full design rationale and the empirical sharding inventory.
    """
    if isinstance(A, np.ndarray):
        return A
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(A))
    if getattr(A, 'is_fully_replicated', False):
        return np.asarray(A.addressable_data(0))
    gathered = multihost_utils.process_allgather(A, tiled=True)
    return np.asarray(jax.device_get(gathered))


class _AllgatherBackend:
    """Rank-0-only h5py file handle with collective process barriers.

    Non-rank-0 processes hold `_file = None` and no-op on every method;
    they only participate in barriers so the collective illusion is
    preserved from the caller's POV.
    """

    def __init__(self, path: str, mode: str = "w") -> None:
        self.path = os.path.abspath(path)
        self.mode = mode
        self._file: h5py.File | None = None
        # Dedicated per-rank 'r' handle opened on first read_slab call
        # and cached for the duration of the SlabIO lifetime.  Reopening
        # per read was an 80 % time hit in the V_q compute loop where
        # many small hyperslabs come through.
        self._read_file: h5py.File | None = None

        d = os.path.dirname(self.path)
        if d and _rank0():
            os.makedirs(d, exist_ok=True)
        _barrier(f"slab_io_open/{self.path}")
        if _rank0():
            self._file = h5py.File(self.path, mode)

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
        if _rank0() and self._file is not None:
            if name in self._file:
                del self._file[name]
            ds = self._file.create_dataset(
                name, shape=tuple(shape), dtype=dtype,
                chunks=tuple(chunks) if chunks else None,
            )
            if attrs:
                for k, v in attrs.items():
                    ds.attrs[k] = v
        _barrier(f"slab_io_create_dataset/{name}")

    # ------------------------------------------------------------------
    def write_attr(self, name: str, value) -> None:
        """Write a small rank-0-only dataset (no allgather).

        For scalars / small metadata like ``omega_ev``: either host
        numpy / python scalars / replicated JAX arrays — we skip the
        allgather and store directly.
        """
        if _rank0() and self._file is not None:
            if name in self._file:
                del self._file[name]
            if isinstance(value, np.ndarray):
                host = value
            elif isinstance(value, (int, float, complex, list, tuple)):
                host = np.asarray(value)
            else:
                host = np.asarray(jax.device_get(value))
            self._file.create_dataset(name, data=host)
        _barrier(f"slab_io_write_attr/{name}")

    # ------------------------------------------------------------------
    # Same padding contract as PHDF5: gather the physical padded slab,
    # but rank 0 writes only the ``valid_shape`` prefix.
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
        k_chunk_size: int | None = None,
    ) -> None:
        """Write A as a hyperslab starting at `offset` inside dataset `name`.

        If the dataset does not yet exist, it's created with
        ``shape = global_shape`` (defaults to A.shape) and the caller's
        ``chunks`` / ``dtype``.
        """
        off, local_shape, gshape = _normalize_slab_request(
            op="write_slab", name=name, offset=offset,
            slab_shape=A.shape, global_shape=global_shape,
            check_bounds=False)
        vshape = _normalize_valid_shape(
            op="write_slab", name=name, valid_shape=valid_shape,
            slab_shape=local_shape, offset=off, global_shape=gshape)

        # Gather once; rank 0 then owns the full slab.
        host = _to_host(A)
        if dtype is None:
            dtype = host.dtype

        if _rank0() and self._file is not None:
            if name not in self._file:
                self._file.create_dataset(
                    name, shape=gshape, dtype=dtype,
                    chunks=tuple(chunks) if chunks else None,
                )
            dset = self._file[name]
            slicer = tuple(slice(o, o + s) for o, s in zip(off, vshape))
            src_slicer = tuple(slice(0, s) for s in vshape)
            if k_chunk_size and len(vshape) >= 2 and vshape[1] > k_chunk_size:
                # Chunk along axis=1 (matches the legacy sigma k_chunk
                # pattern, which streams the write to keep rank-0 memory
                # sane for very large omega-dim).
                k = k_chunk_size
                for k0 in range(0, vshape[1], k):
                    k1 = min(k0 + k, vshape[1])
                    sub = tuple(
                        slice(off[i], off[i] + vshape[i]) if i != 1
                        else slice(off[1] + k0, off[1] + k1)
                        for i in range(len(off))
                    )
                    idx_src = tuple(
                        slice(0, vshape[i]) if i != 1 else slice(k0, k1)
                        for i in range(len(vshape))
                    )
                    dset[sub] = np.asarray(host[idx_src], dtype=dtype)
            else:
                dset[slicer] = np.asarray(host[src_slicer], dtype=dtype)
        _barrier(f"slab_io_write/{name}")

    # ------------------------------------------------------------------
    # Read the logical ``valid_shape`` prefix and embed it into a
    # zero-filled physical ``shape`` for padded sharded consumers.
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        mesh=None,
        as_numpy: bool = False,
        partition_spec=None,
    ) -> jax.Array:
        """Read a hyperslab into a JAX array.

        Every process reads the hyperslab independently via its OWN
        h5py handle (a second file descriptor, opened in 'r' mode for
        non-rank-0 processes).  This matches the historical LORRAX
        pattern: with HDF5_USE_FILE_LOCKING=FALSE on Lustre / GPFS,
        N independent readers are cheaper than rank-0-reads +
        ``broadcast_one_to_all`` (which has per-call coordinator
        overhead that dominates for the many small reads in the V_q
        loop).

        ``partition_spec`` (optional): when set, the result is placed on
        the mesh with this PartitionSpec instead of being replicated.
        Lets the unified V_q tile driver call ``read_slab(...,
        partition_spec=P(None,None,('x','y')))`` regardless of backend —
        the FFI backend reads sharded directly, this backend reads the
        full slab on every rank then ``device_put``s with the requested
        sharding.  Each rank loads the same data and JAX scatters at
        ``device_put`` time.
        """
        # Per-rank cached 'r' handle; one h5py.File per SlabIO lifetime.
        if self._read_file is None:
            self._read_file = h5py.File(self.path, 'r')
        dset = self._read_file[name]
        full_shape = tuple(dset.shape) if shape is None else tuple(shape)
        off, out_shape, _ = _normalize_slab_request(
            op="read_slab", name=name, offset=offset,
            slab_shape=full_shape, global_shape=None, check_bounds=False)
        vshape = _normalize_valid_shape(
            op="read_slab", name=name, valid_shape=valid_shape,
            slab_shape=out_shape, offset=off, global_shape=None)
        # ---- SHARDED fast path ------------------------------------------
        # When the caller asks for a sharded result, read ONLY this
        # process's shard.  The whole-slab path below allocates the FULL
        # global tensor as host numpy TWICE on EVERY rank (``read_host`` +
        # the zero-padded ``host``) — for the V_q batched ζ read
        # (``gw/v_q_g_flat.py::_make_read_all_ibz``, one call for all
        # n_q) that is
        #     n_q · μ_pad · ngkmax · 16  ×2
        # = 11.8 GB/rank at MoS2 12×12 full-BZ (n_q=144, μ_pad=320,
        # ngkmax=8603), i.e. 23.6 GB/node at 2 ranks/node, against a
        # ~79 MB/rank sharded read.  That is the allocation that grew the
        # node from 22 to 53 GB in one 30 s sample and aborted all 80 ranks
        # entering V_q in job 7874242 (the ``raw_buffer.h:149 Check failed:
        # buffer_.IsConcrete()`` LOG(FATAL), rc=134).  Each process already
        # owns its own 'r' handle, so it can read its own hyperslab
        # directly — same bytes, same values, 1/P the host memory.
        #
        # The result is byte-identical to the whole-slab path: every shard
        # takes exactly the elements the target sharding assigns it, and
        # positions past ``vshape`` stay zero (the μ-pad contract).
        if (not as_numpy) and mesh is not None and partition_spec is not None:
            from jax.sharding import NamedSharding
            sharding = NamedSharding(mesh, partition_spec)
            try:
                idx_map = sharding.addressable_devices_indices_map(
                    tuple(out_shape))
            except Exception:
                idx_map = None
            if idx_map:
                ndim = len(out_shape)
                arrays = []
                for dev, ix in idx_map.items():
                    los, his = [], []
                    for ax in range(ndim):
                        sl = ix[ax] if ix is not None else slice(None)
                        lo = 0 if sl.start is None else int(sl.start)
                        hi = int(out_shape[ax]) if sl.stop is None \
                            else int(sl.stop)
                        los.append(lo)
                        his.append(hi)
                    local_shape = tuple(h - l for l, h in zip(los, his))
                    # Intersect this shard with the VALID (on-disk) region;
                    # everything outside stays zero.
                    r_lo = [min(l, int(v)) for l, v in zip(los, vshape)]
                    r_hi = [min(h, int(v)) for h, v in zip(his, vshape)]
                    local = np.zeros(local_shape,
                                     dtype=dtype or dset.dtype)
                    if all(b > a for a, b in zip(r_lo, r_hi)):
                        disk = tuple(slice(off[ax] + r_lo[ax],
                                           off[ax] + r_hi[ax])
                                     for ax in range(ndim))
                        dst = tuple(slice(r_lo[ax] - los[ax],
                                          r_hi[ax] - los[ax])
                                    for ax in range(ndim))
                        local[dst] = dset[disk]
                    arrays.append(jax.device_put(local, dev))
                out = jax.make_array_from_single_device_arrays(
                    tuple(out_shape), sharding, arrays)
                # Complete the transfer while the source numpy is still
                # referenced: XLA:CPU can adopt a large host buffer
                # zero-copy, and letting it fall out of scope with the
                # definition event still pending is the other way to reach
                # the IsConcrete() CHECK.
                jax.block_until_ready(out)
                return out

        slicer = tuple(slice(o, o + s) for o, s in zip(off, vshape))
        read_host = np.asarray(dset[slicer], dtype=dtype) if dtype \
                    else np.asarray(dset[slicer])
        host = np.zeros(out_shape, dtype=read_host.dtype)
        host[tuple(slice(0, s) for s in vshape)] = read_host
        # Return-numpy fast path: skip the H2D+D2H round-trip that the
        # default jax.Array return forces — crucial for V_q which reads
        # many small hyperslabs straight into host numpy stacks.
        if as_numpy:
            return host

        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            from common.collectives import device_put_process_local
            spec = partition_spec if partition_spec is not None else P()
            # Process-local placement: every rank read the same slab from
            # the same file, so plain ``device_put``'s hidden
            # ``assert_equal`` all-gather (P × slab bytes, scorecard AA.1)
            # would verify a tautology.  LORRAX_CHECK_REPLICA=1 re-arms it.
            out = device_put_process_local(host, NamedSharding(mesh, spec))
        else:
            out = jnp.asarray(host)
        # See the note above — keep ``host`` alive until the transfer lands.
        jax.block_until_ready(out)
        return out

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._read_file is not None:
            self._read_file.close()
            self._read_file = None
        if self._file is not None:
            self._file.close()
            self._file = None
        _barrier(f"slab_io_close/{self.path}")

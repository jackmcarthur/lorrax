"""Default SlabIO backend: process_allgather + rank-0 h5py.

This is the fallback / default implementation — byte-identical to the
hand-rolled pattern that used to live in every call site
(``jax.experimental.multihost_utils.process_allgather(A, tiled=False)``
-> rank-0 ``h5py.File``).  Consolidated here so the slab_io public
surface has one place to look for bugs, tuning, or convention changes.

Does NOT import ``ffi.phdf5``; works in any container with h5py +
jax installed.  The FFI backend lives in ``_slab_io_ffi.py``.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils


def _rank0() -> bool:
    return jax.process_index() == 0


def _barrier(tag: str) -> None:
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def _to_host(A: Any) -> np.ndarray:
    """Fully-replicated host ndarray for A, regardless of sharding.

    For numpy / already-replicated inputs this is a cheap cast.  For
    sharded JAX arrays it's ``process_allgather(tiled=False)`` followed
    by ``device_get`` and, if the gather added a leading process-axis
    (``ndim == A.ndim + 1``), collapsing by taking the rank-0 copy —
    all processes see identical data so any index works.
    """
    if isinstance(A, np.ndarray):
        return A
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(A))
    gathered = multihost_utils.process_allgather(A, tiled=False)
    host = np.asarray(jax.device_get(gathered))
    try:
        expected = tuple(int(d) for d in A.shape)
    except Exception:
        expected = None
    if expected is not None and host.shape != expected:
        # tiled=False commonly returns (world, *A.shape); flatten.
        if host.ndim == len(expected) + 1 and host.shape[1:] == expected:
            host = host[0]
        else:
            raise RuntimeError(
                f"_to_host: unexpected gather shape {host.shape} for "
                f"A.shape={expected}")
    return host


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
    def write_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
        global_shape: Sequence[int] | None = None,
        dtype=None,
        chunks: Sequence[int] | None = None,
        k_chunk_size: int | None = None,
    ) -> None:
        """Write A as a hyperslab starting at `offset` inside dataset `name`.

        If the dataset does not yet exist, it's created with
        ``shape = global_shape`` (defaults to A.shape) and the caller's
        ``chunks`` / ``dtype``.
        """
        local_shape = tuple(A.shape)
        off = tuple(offset) if offset is not None else tuple([0] * len(local_shape))
        gshape = tuple(global_shape) if global_shape is not None else local_shape
        if len(off) != len(gshape) or len(off) != len(local_shape):
            raise ValueError(
                f"slab rank mismatch: offset {off}, "
                f"global_shape {gshape}, local {local_shape}")

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
            slicer = tuple(slice(o, o + s) for o, s in zip(off, local_shape))
            if k_chunk_size and len(local_shape) >= 2 and local_shape[1] > k_chunk_size:
                # Chunk along axis=1 (matches the legacy sigma k_chunk
                # pattern, which streams the write to keep rank-0 memory
                # sane for very large omega-dim).
                k = k_chunk_size
                for k0 in range(0, local_shape[1], k):
                    k1 = min(k0 + k, local_shape[1])
                    sub = tuple(
                        slice(off[i], off[i] + local_shape[i]) if i != 1
                        else slice(off[1] + k0, off[1] + k1)
                        for i in range(len(off))
                    )
                    idx_src = tuple(
                        slice(None) if i != 1 else slice(k0, k1)
                        for i in range(len(local_shape))
                    )
                    dset[sub] = np.asarray(host[idx_src], dtype=dtype)
            else:
                dset[slicer] = np.asarray(host, dtype=dtype)
        _barrier(f"slab_io_write/{name}")

    # ------------------------------------------------------------------
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
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
        off = tuple(offset) if offset is not None else tuple([0] * len(full_shape))
        slicer = tuple(slice(o, o + s) for o, s in zip(off, full_shape))
        host = np.asarray(dset[slicer], dtype=dtype) if dtype \
               else np.asarray(dset[slicer])
        # Return-numpy fast path: skip the H2D+D2H round-trip that the
        # default jax.Array return forces — crucial for V_q which reads
        # many small hyperslabs straight into host numpy stacks.
        if as_numpy:
            return host

        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            spec = partition_spec if partition_spec is not None else P()
            return jax.device_put(host, NamedSharding(mesh, spec))
        return jnp.asarray(host)

    # ------------------------------------------------------------------
    def accumulate_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
    ) -> None:
        """``dset[offset : offset+count] += A`` — read-modify-write.

        Used by the ppm_sigma Σ_c(ω) stream mode.  Default backend:
        gather A, then rank-0 reads the existing slab, adds, writes back.
        """
        host = _to_host(A)
        local_shape = tuple(host.shape)
        off = tuple(offset) if offset is not None else tuple([0] * len(local_shape))

        if _rank0() and self._file is not None:
            dset = self._file[name]
            slicer = tuple(slice(o, o + s) for o, s in zip(off, local_shape))
            buf = np.asarray(dset[slicer], dtype=host.dtype)
            buf = buf + host
            dset[slicer] = buf
        _barrier(f"slab_io_accumulate/{name}")

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._read_file is not None:
            self._read_file.close()
            self._read_file = None
        if self._file is not None:
            self._file.close()
            self._file = None
        _barrier(f"slab_io_close/{self.path}")

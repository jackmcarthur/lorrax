"""SlabIO — unified sharded-slab HDF5 I/O with pluggable backend.

One helper for all large-sharded-array writes/reads: replaces the
ad-hoc ``process_allgather`` → rank-0 ``h5py`` patterns sprinkled
across the codebase.

Two backends, selected by ``use_ffi_io: bool``:

- ``False`` (default) — :mod:`file_io._slab_io_allgather`.  Gather to
  rank 0 via ``jax.experimental.multihost_utils.process_allgather``,
  write with plain ``h5py``.  Byte-identical to today's hand-rolled
  pattern.  Works anywhere ``h5py`` + ``jax`` work.
- ``True`` — :mod:`file_io._slab_io_ffi`.  Collective MPI-IO via
  ``ffi.phdf5``; each rank writes its own hyperslab directly.  Lazy
  import; only loads the FFI when this flag is set.

Three primitives:

- ``SlabIO`` — context manager for files that see multiple writes
  (the isdf_fitting zeta loop, ppm_sigma stream).  Opens once,
  creates/ensures datasets, writes/reads/accumulates, closes.
- ``write_slab`` / ``read_slab`` / ``accumulate_slab`` — single-shot
  free functions, open+op+close.  Equivalent to a one-line ``with
  SlabIO(...) as io: io.X(...)``.

All methods accept an ``offset`` N-tuple giving where the local slab
lands in the dataset.  ``global_shape`` defaults to ``A.shape`` for
whole-dataset writes; pass it explicitly when writing a sub-slab of
a larger dataset (e.g. one ω-batch of Σ_c(ω,k,i,j)).
"""
from __future__ import annotations

from typing import Any, Sequence

import jax

from ._slab_io_allgather import _AllgatherBackend

__all__ = [
    "SlabIO",
    "write_slab",
    "read_slab",
    "accumulate_slab",
]


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------
class SlabIO:
    """Unified sharded-slab HDF5 file handle.

    Usage::

        with SlabIO(path, mode="w", mesh=mesh, use_ffi_io=False) as io:
            io.create_dataset("A", shape=(N, M), dtype=jnp.complex128)
            io.write_slab("A", A_sharded)          # whole dataset
            io.write_slab("A", chunk, offset=(i, 0))  # sub-slab
            B = io.read_slab("A", shape=(N, M), dtype=jnp.complex128)

    Parameters
    ----------
    path : str
        HDF5 file path on a shared filesystem.
    mode : {"w", "a", "r"}
        HDF5 open mode.  Must be ``"w"`` to create, ``"a"`` to
        append, ``"r"`` to read.
    mesh : jax.sharding.Mesh, optional
        Required by the FFI backend; the allgather backend ignores
        it (rank-0 always owns the file).
    use_ffi_io : bool
        ``True`` routes through :mod:`ffi.phdf5` (collective MPI-IO);
        ``False`` uses ``process_allgather`` + rank-0 ``h5py``.
    """

    def __init__(
        self,
        path: str,
        *,
        mode: str = "w",
        mesh=None,
        use_ffi_io: bool = False,
    ) -> None:
        self.path = path
        self.mode = mode
        self.mesh = mesh
        self.use_ffi_io = use_ffi_io
        if use_ffi_io:
            if mesh is None:
                raise ValueError("use_ffi_io=True requires mesh")
            from ._slab_io_ffi import _FfiBackend
            self._backend = _FfiBackend(path, mesh=mesh, mode=mode)
        else:
            self._backend = _AllgatherBackend(path, mode=mode)

    # ------------------------------------------------------------------
    def __enter__(self) -> "SlabIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._backend.close()

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
        """Pre-create a dataset with the given shape + dtype + chunks."""
        self._backend.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, attrs=attrs)

    def write_attr(self, name: str, value) -> None:
        """Write a small rank-0-only dataset (e.g. omega_ev).

        Skips the allgather; good for scalars / small metadata arrays
        that are replicated or already on host.
        """
        self._backend.write_attr(name, value)

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
        """Write A as a hyperslab of dataset ``name``.

        ``A`` is an N-D ``jax.Array`` (possibly sharded) or numpy
        array.  Its sharding is inferred from ``A.sharding`` on the
        FFI path; ignored on the allgather path (which gathers the
        whole thing to rank 0).

        ``offset`` (default all zeros) + ``A.shape`` define the
        hyperslab.  ``global_shape`` only matters at dataset creation
        time; if the dataset already exists it's validated against
        the file.

        ``k_chunk_size`` is an allgather-backend-only knob that
        streams the rank-0 write along axis 1 to keep memory bounded
        for large-omega writes (matches the legacy sigma_output
        k_chunk pattern).  Ignored by the FFI backend.
        """
        self._backend.write_slab(
            name, A,
            offset=offset, global_shape=global_shape,
            dtype=dtype, chunks=chunks, k_chunk_size=k_chunk_size,
        )

    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
        mesh=None,
        partition_spec=None,
    ) -> jax.Array:
        """Read a hyperslab into a JAX array.

        On the allgather path, returns a replicated host-backed JAX
        array.  On the FFI path, returns a sharded array with
        ``partition_spec`` on ``mesh`` (default: replicated).
        """
        if self.use_ffi_io:
            return self._backend.read_slab(
                name, shape=shape, dtype=dtype, offset=offset,
                mesh=mesh or self.mesh, partition_spec=partition_spec)
        return self._backend.read_slab(
            name, shape=shape, dtype=dtype, offset=offset, mesh=mesh)

    def accumulate_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
    ) -> None:
        """``dset[offset : offset+A.shape] += A`` (read-modify-write).

        Used by the Σ_c(ω) stream-mode accumulator in ppm_sigma.  The
        default backend does the gather-then-rank-0-RMW that
        ppm_sigma already does today; the FFI backend does a collective
        read → add → collective write at the same hyperslab, which
        lifts today's single-process restriction on stream mode.
        """
        self._backend.accumulate_slab(name, A, offset=offset)


# ---------------------------------------------------------------------------
# Single-shot free functions
# ---------------------------------------------------------------------------
def write_slab(
    path: str,
    ds_name: str,
    A,
    *,
    offset: Sequence[int] | None = None,
    global_shape: Sequence[int] | None = None,
    mesh=None,
    mode: str = "a",
    chunks: Sequence[int] | None = None,
    attrs: dict | None = None,
    use_ffi_io: bool = False,
) -> None:
    """Open + write + close for a one-off dataset write."""
    with SlabIO(path, mode=mode, mesh=mesh, use_ffi_io=use_ffi_io) as io:
        gshape = global_shape if global_shape is not None else tuple(A.shape)
        # Pre-create with user-provided chunks/attrs on first write.
        io.create_dataset(
            ds_name, shape=gshape, dtype=A.dtype,
            chunks=chunks, attrs=attrs)
        io.write_slab(ds_name, A, offset=offset, global_shape=gshape)


def read_slab(
    path: str,
    ds_name: str,
    *,
    shape: Sequence[int] | None = None,
    dtype=None,
    offset: Sequence[int] | None = None,
    mesh=None,
    partition_spec=None,
    use_ffi_io: bool = False,
) -> jax.Array:
    """Open + read + close for a one-off dataset read."""
    with SlabIO(path, mode="r", mesh=mesh, use_ffi_io=use_ffi_io) as io:
        return io.read_slab(
            ds_name, shape=shape, dtype=dtype, offset=offset,
            mesh=mesh, partition_spec=partition_spec)


def accumulate_slab(
    path: str,
    ds_name: str,
    A,
    *,
    offset: Sequence[int] | None = None,
    mesh=None,
    use_ffi_io: bool = False,
) -> None:
    """Open + read-modify-write + close for a one-off accumulate."""
    with SlabIO(path, mode="a", mesh=mesh, use_ffi_io=use_ffi_io) as io:
        io.accumulate_slab(ds_name, A, offset=offset)

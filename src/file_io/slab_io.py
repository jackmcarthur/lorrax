"""SlabIO — unified sharded-slab HDF5 I/O with pluggable backend.

One helper for all large-sharded-array writes/reads: replaces the
ad-hoc ``process_allgather`` → rank-0 ``h5py`` patterns sprinkled
across the codebase.

Two backends, selected by ``backend: SlabIOBackend``:

- :attr:`SlabIOBackend.H5PY_ALLGATHER` (default) —
  :mod:`file_io._slab_io_allgather`.  Gather to rank 0 via
  ``jax.experimental.multihost_utils.process_allgather``, write with
  plain ``h5py``.  Byte-identical to today's hand-rolled pattern.
  Works anywhere ``h5py`` + ``jax`` work.
- :attr:`SlabIOBackend.PHDF5_FFI` — :mod:`file_io._slab_io_ffi`.
  Collective MPI-IO via ``ffi.phdf5``; each rank writes its own
  hyperslab directly.  Lazy import; only loads the FFI when selected.

The legacy ``use_ffi_io: bool`` kwarg is still accepted on every entry
point and silently coerced via :func:`_normalize_slab_backend` — pass
the enum in new code.

Three primitives:

- ``SlabIO`` — context manager for files that see multiple writes
  (the isdf_fitting zeta loop, ppm_sigma stream).  Opens once,
  creates/ensures datasets, writes/reads/accumulates, closes.
- ``write_slab`` / ``read_slab`` / ``accumulate_slab`` — single-shot
  free functions, open+op+close.  Equivalent to a one-line ``with
  SlabIO(...) as io: io.X(...)``.

All methods accept an ``offset`` N-tuple giving where the local slab
lands in the dataset.  ``valid_shape`` defaults to ``A.shape`` for
ordinary writes; pass a smaller prefix when ``A`` is padded for even
sharding but the padded tail should not be written. ``global_shape``
defaults to ``A.shape`` for whole-dataset writes; pass it explicitly
when writing a sub-slab of a larger dataset (e.g. one ω-batch of
Σ_c(ω,k,i,j)).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import jax

from ._slab_io_allgather import _AllgatherBackend

__all__ = [
    "SlabIO",
    "write_slab",
    "read_slab",
    "accumulate_slab",
]


def _normalize_slab_backend(backend, use_ffi_io):
    """Resolve ``(backend, use_ffi_io)`` to a :class:`SlabIOBackend`.

    Accepts:
    - ``backend=SlabIOBackend.PHDF5_FFI | H5PY_ALLGATHER`` (preferred)
    - ``use_ffi_io=True | False`` (legacy boolean — coerced)
    - both ``None``: defaults to allgather

    The legacy boolean is the only place strings/bools cross over into
    the enum world for SlabIO; everywhere else stays in enum land.
    """
    from gw.gw_config import SlabIOBackend  # avoid circular import at module load
    if backend is not None:
        if isinstance(backend, SlabIOBackend):
            return backend
        raise TypeError(
            f"backend={backend!r} must be SlabIOBackend, "
            f"not {type(backend).__name__}"
        )
    if use_ffi_io is None:
        return SlabIOBackend.H5PY_ALLGATHER
    return (SlabIOBackend.PHDF5_FFI if bool(use_ffi_io)
            else SlabIOBackend.H5PY_ALLGATHER)


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------
class SlabIO:
    """Unified sharded-slab HDF5 file handle.

    Usage::

        from gw.gw_config import SlabIOBackend
        with SlabIO(path, mode="w", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
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
    backend : SlabIOBackend, optional
        Selects the underlying I/O path.  Defaults to allgather when
        omitted.  The legacy ``use_ffi_io: bool`` kwarg is also
        accepted for back-compat.
    """

    def __init__(
        self,
        path,
        *,
        mode: str = "w",
        mesh=None,
        backend=None,
        use_ffi_io: bool | None = None,
    ) -> None:
        from gw.gw_config import SlabIOBackend
        # Accept Path | str | bytes uniformly.  The FFI backend's C
        # bindings ``.encode()`` the string, which raises
        # ``AttributeError: 'PosixPath' object has no attribute 'encode'``
        # — historically callers had to remember to ``str()`` paths
        # themselves; now SlabIO does it once.
        self.path = str(path) if not isinstance(path, str) else path
        self.mode = mode
        self.mesh = mesh
        self.backend = _normalize_slab_backend(backend, use_ffi_io)
        # Boolean shortcut still used internally by ``read_slab`` for
        # branch dispatch — kept as a derived attribute, not the source
        # of truth.
        self.use_ffi_io = (self.backend is SlabIOBackend.PHDF5_FFI)
        if self.use_ffi_io:
            if mesh is None:
                raise ValueError(
                    "backend=SlabIOBackend.PHDF5_FFI requires mesh")
            from ._slab_io_ffi import _FfiBackend
            self._backend = _FfiBackend(self.path, mesh=mesh, mode=mode)
        else:
            self._backend = _AllgatherBackend(self.path, mode=mode)

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
        """Pre-create a dataset with the given shape + dtype + chunks.

        Padding policy: SlabIO writes ``A`` at exactly the dataset's
        shape; pass ``valid_shape`` on ``write_slab`` for chunk writes
        or to clip a padded ``A`` down to a smaller logical extent.
        Driver-side padding/unpadding is the caller's responsibility
        (see ``runtime.padding`` in the agent/padding-refactor branch);
        files always store the logical shape so they can be re-read
        on a different mesh size.
        """
        self._backend.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, attrs=attrs)

    def write_attr(self, name: str, value) -> None:
        """Write a small rank-0-only dataset (e.g. omega_ev).

        Skips the allgather; good for scalars / small metadata arrays
        that are replicated or already on host.
        """
        self._backend.write_attr(name, value)

    # Padding contract for sharded producers: make ``A.shape`` divisible
    # by the mesh, but pass ``valid_shape`` for the logical prefix that
    # should reach disk.  The padded tail of ``A`` is ignored.
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
        """Write A as a hyperslab of dataset ``name``.

        ``A`` is an N-D ``jax.Array`` (possibly sharded) or numpy
        array.  Its sharding is inferred from ``A.sharding`` on the
        FFI path; ignored on the allgather path (which gathers the
        whole thing to rank 0).

        ``offset`` (default all zeros) + ``valid_shape`` define the
        logical hyperslab.  ``valid_shape`` defaults to ``A.shape``;
        pass a smaller prefix when the physical array is padded for
        sharding but the padded tail should not be written.

        ``k_chunk_size`` is an allgather-backend-only knob that
        streams the rank-0 write along axis 1 to keep memory bounded
        for large-omega writes (matches the legacy sigma_output
        k_chunk pattern).  Ignored by the FFI backend.
        """
        self._backend.write_slab(
            name, A,
            offset=offset, global_shape=global_shape,
            valid_shape=valid_shape,
            dtype=dtype, chunks=chunks, k_chunk_size=k_chunk_size,
        )

    # Padding contract for sharded consumers: request a mesh-divisible
    # physical ``shape`` and pass smaller ``valid_shape`` for the file
    # prefix to populate.  The returned padded tail is zero-filled.
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        mesh=None,
        partition_spec=None,
        as_numpy: bool = False,
    ) -> jax.Array:
        """Read a hyperslab.

        On the allgather path, returns a replicated host-backed JAX
        array (or a plain ``np.ndarray`` if ``as_numpy=True`` — useful
        for readers that feed host-side numpy stacks straight into a
        GPU compute kernel later, skipping a pointless H2D+D2H round
        trip).  On the FFI path, returns a sharded array with
        ``partition_spec`` on ``mesh`` (``as_numpy`` still forces a
        host ndarray via ``device_get``).

        ``shape`` is the physical output shape and may include padding.
        ``valid_shape`` is the logical file extent to read into the
        prefix of that output; padded tail elements are zero-filled.
        Defaults to ``shape``.
        """
        if self.use_ffi_io:
            arr = self._backend.read_slab(
                name, shape=shape, dtype=dtype, offset=offset,
                valid_shape=valid_shape,
                mesh=mesh or self.mesh, partition_spec=partition_spec)
        else:
            arr = self._backend.read_slab(
                name, shape=shape, dtype=dtype, offset=offset,
                valid_shape=valid_shape,
                mesh=mesh, as_numpy=as_numpy,
                partition_spec=partition_spec)
        if as_numpy and not isinstance(arr, np.ndarray):
            arr = np.asarray(jax.device_get(arr))
        return arr

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
    valid_shape: Sequence[int] | None = None,
    mesh=None,
    mode: str = "a",
    chunks: Sequence[int] | None = None,
    attrs: dict | None = None,
    backend=None,
    use_ffi_io: bool | None = None,
) -> None:
    """Open + write + close for a one-off dataset write."""
    with SlabIO(path, mode=mode, mesh=mesh,
                backend=backend, use_ffi_io=use_ffi_io) as io:
        gshape = global_shape if global_shape is not None else tuple(A.shape)
        # Pre-create with user-provided chunks/attrs on first write.
        io.create_dataset(
            ds_name, shape=gshape, dtype=A.dtype,
            chunks=chunks, attrs=attrs)
        io.write_slab(
            ds_name, A, offset=offset, global_shape=gshape,
            valid_shape=valid_shape)


def read_slab(
    path: str,
    ds_name: str,
    *,
    shape: Sequence[int] | None = None,
    dtype=None,
    offset: Sequence[int] | None = None,
    valid_shape: Sequence[int] | None = None,
    mesh=None,
    partition_spec=None,
    backend=None,
    use_ffi_io: bool | None = None,
) -> jax.Array:
    """Open + read + close for a one-off dataset read."""
    with SlabIO(path, mode="r", mesh=mesh,
                backend=backend, use_ffi_io=use_ffi_io) as io:
        return io.read_slab(
            ds_name, shape=shape, dtype=dtype, offset=offset,
            valid_shape=valid_shape,
            mesh=mesh, partition_spec=partition_spec)


def accumulate_slab(
    path: str,
    ds_name: str,
    A,
    *,
    offset: Sequence[int] | None = None,
    mesh=None,
    backend=None,
    use_ffi_io: bool | None = None,
) -> None:
    """Open + read-modify-write + close for a one-off accumulate."""
    with SlabIO(path, mode="a", mesh=mesh,
                backend=backend, use_ffi_io=use_ffi_io) as io:
        io.accumulate_slab(ds_name, A, offset=offset)

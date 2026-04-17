"""FFI SlabIO backend — collective MPI-IO via ``ffi.phdf5``.

Opt-in path (``use_ffi_io=True``).  Imported lazily by
:mod:`file_io.slab_io` so the default allgather path works without
``liblorrax_ffi.so`` being built.

Every operation derives per-rank hyperslab offsets from the sharding
spec of the JAX array being written (or a caller-provided one for
reads) plus a global-origin ``offset`` argument.  The C++ handler
un-ravels the rank id through ``mesh_shape`` and advances along every
sharded dim.  See ``ffi/phdf5/cpp/write_ffi.cc`` for the C++ side.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Lazy imports happen inside the class methods; module-level imports
# of ffi.phdf5 would break users who don't build the FFI .so.


def _sharding_to_axis_for_dim(
    sharding: NamedSharding, ndim: int,
) -> tuple[int, ...]:
    """For a NamedSharding over a Mesh, return a per-dim mesh-axis index
    (or -1 for replicated).  Length always equals ``ndim``.

    JAX canonicalises ``PartitionSpec(None, None)`` -> ``PartitionSpec()``
    for fully replicated arrays, and in general pads / truncates trailing
    Nones — so we iterate by the array's ndim and treat missing entries
    as replicated.

    Only supports single-axis-per-dim layouts; raises if a dim is
    sharded over multiple mesh axes.
    """
    axis_names = list(sharding.mesh.axis_names)
    spec = list(sharding.spec)
    out: list[int] = []
    for i in range(ndim):
        s = spec[i] if i < len(spec) else None
        if s is None:
            out.append(-1)
        elif isinstance(s, str):
            if s not in axis_names:
                raise ValueError(
                    f"sharding spec dim {i}: axis '{s}' not in mesh "
                    f"axis_names {axis_names}")
            out.append(axis_names.index(s))
        elif isinstance(s, (list, tuple)):
            if len(s) != 1:
                raise NotImplementedError(
                    f"multi-axis dim {i}={s!r}: not supported by the FFI "
                    "backend yet; use use_ffi_io=False")
            out.append(axis_names.index(s[0]))
        else:
            raise ValueError(f"unrecognised spec element at dim {i}: {s!r}")
    return tuple(out)


def _replicated_sharding(mesh: Mesh, ndim: int) -> NamedSharding:
    """All-None PartitionSpec on `mesh` for an ndim-D array."""
    return NamedSharding(mesh, P(*([None] * ndim)))


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class _FfiBackend:
    """Collective MPI-IO SlabIO backend."""

    def __init__(self, path: str, mesh: Mesh, mode: str = "w") -> None:
        # Lazy import — keeps file_io importable without the FFI built.
        from ffi.phdf5 import open_file as _open_file, close_file as _close_file
        from ffi.common import ffi_loader as _loader

        self._open_file = _open_file
        self._close_file = _close_file
        self._loader = _loader

        self.path = path
        self.mesh = mesh
        self.mode = mode
        self.fh: int = self._open_file(path, mesh=mesh, mode=mode)
        self._ds_ids: dict[str, int] = {}
        # write_attr needs plain h5py (the FFI doesn't expose a
        # collective attr-write path), so we defer attr writes to
        # close() — concurrent h5py + MPI-IO on the same file would
        # corrupt HDF5 metadata.
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
        ds_id = self._loader.phdf5_ensure_dataset(
            self.fh, name, tuple(int(s) for s in shape),
            str(jnp.dtype(dtype).name),
        )
        self._ds_ids[name] = ds_id
        # chunks + attrs are runtime-set on the underlying H5 dataset.
        # The FFI backend doesn't yet expose a collective "set chunks"
        # after H5Dcreate (it would need a new ctypes entry and some
        # care around MPI-IO dataset transfer property lists).  The
        # caller's `chunks=` argument is a hint for the writer; when the
        # dataset is created by the FFI path the H5 library picks
        # contiguous layout + the FAPL-level alignment set in ctx
        # init.  For v1 this matches the OpenMPI-stack perf ceiling;
        # user can pre-create with h5py + chunks if needed.
        if chunks is not None or attrs is not None:
            import warnings
            warnings.warn(
                "FFI backend: chunks/attrs on create_dataset currently no-op; "
                "pre-create with h5py if you need explicit chunking or attrs.")

    # ------------------------------------------------------------------
    def write_attr(self, name: str, value) -> None:
        # Deferred to close() to avoid interleaving rank-0 h5py with
        # active MPI-IO on the same file.  Small arrays only; this is
        # not meant for large data.
        self._deferred_attrs.append((name, value))

    # ------------------------------------------------------------------
    def _ds_id(self, name: str, readonly: bool = False) -> int:
        if name in self._ds_ids:
            return self._ds_ids[name]
        if readonly:
            ds_id = self._loader.phdf5_open_dataset_ro(self.fh, name)
        else:
            raise RuntimeError(
                f"dataset '{name}' not registered — call create_dataset first")
        self._ds_ids[name] = ds_id
        return ds_id

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
        from ffi.phdf5.write import ffi_write_call

        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)
        # Ensure placement: if not sharded on our mesh, put as replicated.
        if not isinstance(A.sharding, NamedSharding) or A.sharding.mesh is not self.mesh:
            A = jax.device_put(A, _replicated_sharding(self.mesh, A.ndim))

        axis_for_dim = _sharding_to_axis_for_dim(A.sharding, A.ndim)
        off = tuple(offset) if offset is not None else tuple([0] * A.ndim)
        gshape = tuple(global_shape) if global_shape is not None else tuple(A.shape)

        # Ensure the dataset exists; create_dataset may or may not have
        # been called — this handles the single-shot `write_slab` case
        # by ensuring with the caller-provided global_shape (default =
        # A.shape when writing a whole dataset).
        if name not in self._ds_ids:
            ds_id = self._loader.phdf5_ensure_dataset(
                self.fh, name, tuple(int(s) for s in gshape),
                str(jnp.dtype(A.dtype).name),
            )
            self._ds_ids[name] = ds_id

        ds_id = self._ds_ids[name]
        ctx_handle = self.fh
        mesh_shape = tuple(self.mesh.shape[ax] for ax in self.mesh.axis_names)
        in_specs = A.sharding.spec  # PartitionSpec

        def _per_rank(A_local):
            return ffi_write_call(
                A_local,
                ctx_handle=int(ctx_handle),
                ds_id=int(ds_id),
                offset_base=off,
                mesh_shape=mesh_shape,
                axis_for_dim=axis_for_dim,
            )

        shard_map(
            _per_rank, mesh=self.mesh,
            in_specs=in_specs, out_specs=P(),
            check_rep=False,
        )(A).block_until_ready()

    # ------------------------------------------------------------------
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int],
        dtype,
        offset: Sequence[int] | None = None,
        mesh: Mesh | None = None,
        partition_spec: P | None = None,
    ) -> jax.Array:
        from ffi.phdf5.read import ffi_read_call

        mesh = mesh or self.mesh
        off = tuple(offset) if offset is not None else tuple([0] * len(shape))

        # Default: fully replicated.  Caller can provide partition_spec
        # to shard the read.
        if partition_spec is None:
            partition_spec = P(*([None] * len(shape)))
        sharding = NamedSharding(mesh, partition_spec)
        axis_for_dim = _sharding_to_axis_for_dim(sharding, len(shape))
        mesh_shape = tuple(mesh.shape[ax] for ax in mesh.axis_names)

        # Per-rank output shape: global shape / mesh_shape[axis] per sharded dim.
        local_shape = list(shape)
        for d, ax in enumerate(axis_for_dim):
            if ax >= 0:
                local_shape[d] = int(local_shape[d]) // int(mesh_shape[ax])
        out_struct = jax.ShapeDtypeStruct(tuple(local_shape), jnp.dtype(dtype))

        ds_id = self._ds_id(name, readonly=True)
        ctx_handle = self.fh

        def _per_rank():
            return ffi_read_call(
                out_struct,
                ctx_handle=int(ctx_handle),
                ds_id=int(ds_id),
                offset_base=off,
                mesh_shape=mesh_shape,
                axis_for_dim=axis_for_dim,
            )

        result = shard_map(
            _per_rank, mesh=mesh,
            in_specs=(), out_specs=partition_spec,
            check_rep=False,
        )()
        result.block_until_ready()
        return result

    # ------------------------------------------------------------------
    def accumulate_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
    ) -> None:
        """dset[off:off+A.shape] += A — collective read-modify-write."""
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)
        if not isinstance(A.sharding, NamedSharding) or A.sharding.mesh is not self.mesh:
            A = jax.device_put(A, _replicated_sharding(self.mesh, A.ndim))

        off = tuple(offset) if offset is not None else tuple([0] * A.ndim)
        existing = self.read_slab(
            name, shape=tuple(A.shape), dtype=A.dtype,
            offset=off, mesh=self.mesh, partition_spec=A.sharding.spec,
        )
        A_new = existing + A
        self.write_slab(name, A_new, offset=off, global_shape=None)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.fh:
            self._close_file(self.fh)
            self.fh = 0
        # Now that MPI-IO has released the file, rank 0 can safely
        # reopen with h5py to tack on any deferred small-metadata
        # datasets (omega_ev and friends).
        if self._deferred_attrs:
            import h5py
            import numpy as np
            if jax.process_index() == 0:
                with h5py.File(self.path, "a") as h5:
                    for name, value in self._deferred_attrs:
                        if name in h5:
                            del h5[name]
                        host = value
                        if not isinstance(host, np.ndarray):
                            host = np.asarray(jax.device_get(host))
                        h5.create_dataset(name, data=host)
            try:
                multihost_utils.sync_global_devices("slab_io_ffi_close_attrs")
            except Exception:
                pass
            self._deferred_attrs = []

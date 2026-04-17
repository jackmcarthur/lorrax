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


def _sharding_to_axis_for_dim(sharding: NamedSharding) -> tuple[int, ...]:
    """For a NamedSharding over a Mesh, return a per-dim mesh-axis index
    (or -1 for replicated).  Only supports single-axis-per-dim layouts;
    raises if a dim is sharded over multiple mesh axes.

    The mesh axis index is the position of the axis name in
    ``mesh.axis_names`` (row-major flatten of the rank id uses this
    ordering).
    """
    axis_names = list(sharding.mesh.axis_names)
    spec = sharding.spec
    out = []
    for i, s in enumerate(spec):
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
            self.fh, name,
            # phdf5_ensure_dataset currently takes n_rows + n_cols; for
            # N-D we degrade to whatever it already does (creates with
            # the dtype we pass; shape comes from first write).  When
            # we extend to N-D dataset-create, this grows to take full
            # shape+chunks.  For now the FFI create-on-first-write
            # semantics cover 2-D; we rely on the allgather backend for
            # > 2-D create-with-chunks (see slab_io.py dispatch).
            int(shape[0]) if len(shape) >= 1 else 1,
            int(shape[1]) if len(shape) >= 2 else 1,
            str(jnp.dtype(dtype).name),
        )
        self._ds_ids[name] = ds_id
        # TODO(phdf5): chunks + attrs via a new ctypes entry; meanwhile
        # these get set by the allgather backend on create, or ignored.
        if chunks is not None or attrs is not None:
            import warnings
            warnings.warn(
                "FFI backend: chunks/attrs on create_dataset currently no-op; "
                "use allgather backend to set them, or ensure the dataset "
                "already exists with the desired chunking before writing.")

    # ------------------------------------------------------------------
    def write_attr(self, name: str, value) -> None:
        # FFI backend has no native small-metadata path; fall back to
        # rank-0 h5py tacked onto the same file.  Safe because
        # lrx_phdf5_open left the MPI-IO FAPL in charge, but we still
        # need to avoid collisions with MPI-IO's own locks.  The cheap
        # fix: open with h5py in append mode on rank 0 only, write,
        # close — mirrors the allgather backend's write_attr.
        import h5py
        if jax.process_index() == 0:
            with h5py.File(self.path, "a") as h5:
                if name in h5:
                    del h5[name]
                import numpy as np
                h5.create_dataset(
                    name, data=np.asarray(jax.device_get(value))
                    if not isinstance(value, np.ndarray) else value,
                )
        try:
            multihost_utils.sync_global_devices(f"slab_io_ffi_attr/{name}")
        except Exception:
            pass

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

        axis_for_dim = _sharding_to_axis_for_dim(A.sharding)
        off = tuple(offset) if offset is not None else tuple([0] * A.ndim)
        gshape = tuple(global_shape) if global_shape is not None else tuple(A.shape)

        # Ensure the dataset exists with the right shape.  The FFI's
        # phdf5_ensure_dataset currently only takes (n_rows, n_cols) —
        # see TODO in create_dataset.  For N-D datasets we require the
        # dataset to have been created already (either by a prior call
        # or by the caller with h5py).  This is a known v1 limitation.
        if name not in self._ds_ids:
            # Will fail on N-D create; we expect create_dataset to have
            # been called (or this is 2-D).  Retry logic kept simple.
            ds_id = self._loader.phdf5_ensure_dataset(
                self.fh, name,
                int(gshape[0]) if len(gshape) >= 1 else 1,
                int(gshape[1]) if len(gshape) >= 2 else 1,
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
        axis_for_dim = _sharding_to_axis_for_dim(sharding)
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

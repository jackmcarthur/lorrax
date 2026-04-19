"""Sharded-slab FFI reader — thin shard_map wrapper around PhdfReadFfi.

Preferred public entry point is :mod:`file_io.slab_io`, which
dispatches between allgather-broadcast-from-rank-0 (default) and this
FFI backend.

The underlying C++ handler is N-D and derives per-rank hyperslab
offsets from ``ctx->rank`` + mesh_shape / axis_for_dim attrs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common import ffi_loader
from ..common.ffi_loader import get_lib
from .context import validate_mesh_2d

__all__ = [
    "read_sharded_slab", "ffi_read_call",
    "ffi_read_kchunk_call", "read_kchunk_sharded",
]

_FFI_TARGET = "lorrax_phdf5_read"
_FFI_TARGET_KCHUNK = "lorrax_phdf5_read_kchunk"


def ffi_read_call(
    out_struct: jax.ShapeDtypeStruct,
    offset_base: jax.Array,
    *,
    ctx_handle: int,
    ds_id: int,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
) -> jax.Array:
    """Low-level FFI call.  ``offset_base`` is a jax.Array of shape
    (ndim,) dtype int64 — passed as a traced Buffer input (not an FFI
    Attr), mirroring ffi_write_call so shard_map closures compile
    ONCE per dataset-ndim-dtype-sharding tuple and re-dispatch across
    chunks with different offsets.  See ffi.phdf5.write for the
    ``axis_count_per_dim`` + ``axis_flat`` encoding.
    """
    return jax.ffi.ffi_call(_FFI_TARGET, out_struct)(
        offset_base,
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
    )


def read_sharded_slab(
    fh: int,
    ds_name: str,
    *,
    global_shape: tuple[int, int],
    dtype,
    mesh: Mesh,
) -> jax.Array:
    """Read a 2-D dataset into a P('x','y')-sharded JAX array."""
    p, q = validate_mesh_2d(mesh)
    n_rows, n_cols = int(global_shape[0]), int(global_shape[1])
    if n_rows % p or n_cols % q:
        raise ValueError(
            f"global shape ({n_rows},{n_cols}) must divide mesh ({p},{q})")
    get_lib()

    ds_id = ffi_loader.phdf5_open_dataset_ro(fh, ds_name)
    local_shape = (n_rows // p, n_cols // q)
    out_struct = jax.ShapeDtypeStruct(local_shape, jnp.dtype(dtype))

    offset_array = jnp.zeros((2,), dtype=jnp.int64)

    def _per_rank(offset_local):
        return ffi_read_call(
            out_struct,
            offset_local,
            ctx_handle=int(fh),
            ds_id=int(ds_id),
            mesh_shape=(p, q),
            axis_count_per_dim=(1, 1),
            axis_flat=(0, 1),
        )

    return shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(),), out_specs=P("x", "y"),
        check_rep=False,
    )(offset_array)


# ---------------------------------------------------------------------------
#  k-chunk reader — one handler invocation returns n_kchunk independently-
#  located hyperslab windows of the same dataset, packed into a prepended
#  n_kchunk axis.  See cpp/read_ffi.cc for the design note on why this
#  does n_kchunk sequential H5Dread calls (all inside one XLA op) rather
#  than one compound-hyperslab H5S_SELECT_OR read: the SELECT_OR path
#  breaks for variable-ngk layouts because the hyperslab union
#  deduplicates overlap, leaving the memspace/filespace element counts
#  mismatched.
# ---------------------------------------------------------------------------
def ffi_read_kchunk_call(
    out_struct: jax.ShapeDtypeStruct,
    offset_base: jax.Array,
    *,
    ctx_handle: int,
    ds_id: int,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    n_kchunk: int,
) -> jax.Array:
    """Low-level kchunk FFI call.

    ``offset_base`` is shape ``(n_kchunk, N_file)`` int64, where each row
    gives the user-level hyperslab origin for one k-chunk element on the
    N_file-dimensional dataset.  The C++ handler adds the per-rank shard
    offset (derived from axis_count_per_dim / axis_flat) and issues one
    collective H5Dread per k-chunk element into a single pinned buffer
    inside one XLA op.

    ``out_struct`` is ``(n_kchunk, *per_rank_file_shape)`` — the leading
    n_kchunk axis is always replicated across ranks; only the trailing
    file dims are sharded per the usual mesh_shape / axis_count / axis_flat
    encoding.  ``n_kchunk`` is passed as a compile-time Attr and must
    match ``out_struct.shape[0]``.
    """
    return jax.ffi.ffi_call(_FFI_TARGET_KCHUNK, out_struct)(
        offset_base,
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
        n_kchunk=int(n_kchunk),
    )


def read_kchunk_sharded(
    fh: int,
    ds_name: str,
    *,
    n_kchunk: int,
    file_global_shape: Sequence[int],
    per_rank_file_shape: Sequence[int],
    dtype,
    mesh: Mesh,
    file_partition_spec: P,
) -> callable:
    """Build a jitted callable that reads ``n_kchunk`` hyperslab windows
    of an N-D dataset into a ``(n_kchunk, *file_global_shape_sharded)``
    array with the leading axis replicated.

    Parameters
    ----------
    fh : int
        Context handle from ``open_file``.
    ds_name : str
        HDF5 dataset path (e.g. ``"wfns/coeffs"``).
    n_kchunk : int
        Number of k-chunk elements.  Compile-time constant.
    file_global_shape : tuple of int, length N_file
        Full dataset shape (just for divisibility check + sanity).
    per_rank_file_shape : tuple of int, length N_file
        Shape of one k-chunk slab on one rank.  For a sharded band axis
        this is ``(bands_per_shard, ns, ngkmax, 2)``.
    dtype : numpy dtype
        Dataset element type (``np.float64`` for ``wfns/coeffs``).
    mesh : Mesh
        2-D mesh ``('x','y')``.
    file_partition_spec : PartitionSpec
        How the ``file_global_shape`` dims are sharded on ``mesh``.
        Length must equal ``N_file``.  A leading ``None`` for the
        prepended n_kchunk axis is added automatically.

    Returns
    -------
    A callable ``f(offset_base)`` where ``offset_base`` is a jax.Array of
    shape ``(n_kchunk, N_file)`` int64.  Returns a jax.Array of global
    shape ``(n_kchunk, *file_global_shape_sharded)``, sharded according
    to ``file_partition_spec``.
    """
    # Reuse the sharding encoder living in _slab_io_ffi to keep a single
    # source of truth.
    from file_io._slab_io_ffi import _sharding_to_axis_info as sharding_to_axis_info

    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    n_file = len(per_rank_file_shape)
    if len(file_partition_spec) > n_file:
        raise ValueError(
            f"file_partition_spec has {len(file_partition_spec)} entries "
            f"but per_rank_file_shape has only {n_file} dims")
    if len(file_global_shape) != n_file:
        raise ValueError(
            f"file_global_shape ndim {len(file_global_shape)} "
            f"!= per_rank_file_shape ndim {n_file}")

    # Encode the FILE dims' sharding via the shared helper (expects a
    # NamedSharding, so construct one over the file dims only).
    from jax.sharding import NamedSharding
    file_sharding = NamedSharding(mesh, file_partition_spec)
    axis_count_per_dim, axis_flat = sharding_to_axis_info(file_sharding, n_file)

    get_lib()
    ds_id = ffi_loader.phdf5_open_dataset_ro(fh, ds_name)

    # Per-rank output shape: prepended n_kchunk + per_rank_file_shape.
    out_local_shape = (n_kchunk,) + tuple(int(s) for s in per_rank_file_shape)
    out_struct = jax.ShapeDtypeStruct(out_local_shape, jnp.dtype(dtype))

    # Global sharding: n_kchunk replicated, file dims sharded as given.
    full_partition_spec = P(None, *tuple(file_partition_spec))

    def _per_rank(offset_base_local):
        return ffi_read_kchunk_call(
            out_struct,
            offset_base_local,
            ctx_handle=int(fh),
            ds_id=int(ds_id),
            mesh_shape=(p, q),
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            n_kchunk=int(n_kchunk),
        )

    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(),), out_specs=full_partition_spec,
        check_rep=False,
    )
    return jax.jit(sm_bare)

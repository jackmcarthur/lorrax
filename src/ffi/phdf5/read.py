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

__all__ = ["read_sharded_slab", "ffi_read_call"]

_FFI_TARGET = "lorrax_phdf5_read"


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

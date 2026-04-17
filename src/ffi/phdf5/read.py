"""Sharded-slab FFI reader — thin shard_map wrapper around PhdfReadFfi.

Preferred public entry point is :mod:`file_io.slab_io`, which
dispatches between allgather-broadcast-from-rank-0 (default) and this
FFI backend.

The underlying C++ handler is N-D and derives per-rank hyperslab
offsets from ``ctx->rank`` + mesh_shape / axis_for_dim attrs.
"""
from __future__ import annotations

from typing import Sequence

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
    *,
    ctx_handle: int,
    ds_id: int,
    offset_base: Sequence[int],
    mesh_shape: Sequence[int],
    axis_for_dim: Sequence[int],
) -> jax.Array:
    """Low-level FFI call.  Returns the rank-local shard (N-D).

    Attrs match the N-D C++ contract; caller computes them.  Use
    inside a ``shard_map`` body with ``in_specs=()``.
    """
    return jax.ffi.ffi_call(_FFI_TARGET, out_struct)(
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
        offset_base=tuple(int(x) for x in offset_base),
        mesh_shape=tuple(int(x) for x in mesh_shape),
        axis_for_dim=tuple(int(x) for x in axis_for_dim),
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

    def _per_rank():
        return ffi_read_call(
            out_struct,
            ctx_handle=int(fh),
            ds_id=int(ds_id),
            offset_base=(0, 0),
            mesh_shape=(p, q),
            axis_for_dim=(0, 1),
        )

    return shard_map(
        _per_rank, mesh=mesh,
        in_specs=(), out_specs=P("x", "y"),
        check_rep=False,
    )()

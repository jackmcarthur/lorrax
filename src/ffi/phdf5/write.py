"""Sharded-slab FFI writer — thin shard_map wrapper around PhdfWriteFfi.

Preferred public entry point for gw_jax / isdf / etc. is
:mod:`file_io.slab_io`, which dispatches on ``use_ffi_io`` between the
allgather-and-rank-0-h5py backend (default) and this FFI backend.
Call this module directly only for 2-D block-partitioned writes where
you know you want the FFI path.

The underlying C++ handler is N-D and derives per-rank hyperslab
offsets from ``ctx->rank`` + the mesh_shape / axis_for_dim attrs.
"""
from __future__ import annotations

from functools import partial
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common import ffi_loader
from ..common.ffi_loader import get_lib
from .context import validate_mesh_2d

__all__ = ["write_sharded_slab", "ffi_write_call"]

_FFI_TARGET = "lorrax_phdf5_write"


def ffi_write_call(
    A_local: jax.Array,
    offset_base: jax.Array,
    *,
    ctx_handle: int,
    ds_id: int,
    mesh_shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
) -> jax.Array:
    """Low-level FFI call for one rank's local shard.  Returns token.

    ``offset_base`` is a jax.Array of shape (ndim,) dtype int64 — passed
    as a traced Buffer input (not an FFI Attr) so that shard_map closures
    compile ONCE per dataset-ndim-dtype-sharding tuple and re-dispatch
    across chunks with different offsets.  Without this, each chunk
    triggers a fresh ~400 ms XLA compile for the FFI body (measured at
    MoS2 3x3 scale, see reports/zeta_offset_runtime_2026-04-19/).

    The other mesh/axis attrs ARE compile-time attrs — they don't change
    across chunks of a given dataset, so no recompile happens.

    Use inside a ``shard_map`` body.
    """
    token_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    return jax.ffi.ffi_call(_FFI_TARGET, token_spec, has_side_effect=True)(
        A_local,
        offset_base,
        ctx_handle=int(ctx_handle),
        ds_id=int(ds_id),
        mesh_shape=np.asarray(mesh_shape, dtype=np.int64),
        axis_count_per_dim=np.asarray(axis_count_per_dim, dtype=np.int64),
        axis_flat=np.asarray(axis_flat, dtype=np.int64),
    )


def write_sharded_slab(
    fh: int,
    ds_name: str,
    A: jax.Array,
    *,
    mesh: Mesh,
    global_shape: tuple[int, int] | None = None,
) -> jax.Array:
    """Write a 2-D P('x','y')-sharded JAX array to an open HDF5 dataset.

    Thin wrapper on top of the N-D FFI: sets
    ``mesh_shape=(p,q)`` and ``axis_for_dim=(0,1)`` so each rank writes
    its (rank/q, rank%q) hyperslab.  Shipped as the simplest 2-D entry
    point; for N-D writes use :mod:`file_io.slab_io`.
    """
    if A.ndim != 2:
        raise NotImplementedError(
            "write_sharded_slab is 2-D only; use file_io.slab_io for N-D")
    p, q = validate_mesh_2d(mesh)
    n_rows, n_cols = (global_shape if global_shape is not None else A.shape)
    if n_rows % p or n_cols % q:
        raise ValueError(
            f"global shape ({n_rows},{n_cols}) must divide mesh ({p},{q})")
    get_lib()

    ds_id = ffi_loader.phdf5_ensure_dataset(
        fh, ds_name, (int(n_rows), int(n_cols)),
        str(jnp.dtype(A.dtype).name))

    offset_array = jnp.zeros((2,), dtype=jnp.int64)

    def _per_rank(A_local, offset_local):
        return ffi_write_call(
            A_local, offset_local,
            ctx_handle=int(fh),
            ds_id=int(ds_id),
            mesh_shape=(p, q),
            axis_count_per_dim=(1, 1),
            axis_flat=(0, 1),
        )

    return shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P("x", "y"), P()), out_specs=P(),
        check_rep=False,
    )(A, offset_array)

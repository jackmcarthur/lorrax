"""Write-sharded-slab FFI wrapper.

Inside ``shard_map(in_specs=P('x','y'), out_specs=P())``, each process
hands its local shard to ``lorrax_phdf5_write``.  The FFI handler D2H-
stages to a pinned buffer and does a blocking ``H5Dwrite`` collective
over MPI-IO.  The host thread blocks for the duration of the write;
the CUDA device stream keeps running previously-queued ops.

Returns an opaque ``(1,) int32`` token — pass it through subsequent
JAX ops (or just ignore it) to preserve ordering when the write's
completion needs to be a barrier for later work.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common import ffi_loader
from ..common.ffi_loader import get_lib
from .context import validate_mesh_2d

__all__ = ["write_sharded_slab"]

_FFI_TARGET = "lorrax_phdf5_write"


def write_sharded_slab(
    fh: int,
    ds_name: str,
    A: jax.Array,
    *,
    mesh: Mesh,
    global_shape: tuple[int, int] | None = None,
) -> jax.Array:
    """Write a 2-D sharded JAX array to a hyperslab of an open HDF5 dataset.

    Parameters
    ----------
    fh
        Handle returned by :func:`open_file`.
    ds_name
        HDF5 path (e.g. ``"A"`` or ``"/group/A"``).  The dataset is
        created (or opened if it already exists) collectively by all
        ranks before the FFI call; subsequent writes to the same name
        hit the cached handle.
    A
        ``jax.Array`` sharded ``P('x','y')`` over ``mesh``.  Each
        process writes its local shard to its corresponding hyperslab.
    mesh
        2-D mesh labelled ``('x','y')``.
    global_shape
        Optional global ``(n_rows, n_cols)``.  Default: ``A.shape``.

    Returns
    -------
    token
        ``(1,) int32`` array usable as an ordering marker in downstream
        JAX code.
    """
    if A.ndim != 2:
        raise NotImplementedError("write_sharded_slab supports 2-D for v1")
    p, q = validate_mesh_2d(mesh)
    n_rows, n_cols = (global_shape if global_shape is not None else A.shape)
    if n_rows % p or n_cols % q:
        raise ValueError(
            f"global shape ({n_rows},{n_cols}) must divide mesh ({p},{q})")
    get_lib()

    # Collective ensure: H5Dcreate (or H5Dopen) happens in all ranks
    # together on the host before we enter the shard_map.  Returns a
    # hid_t we pass through the FFI as an int64 attribute.
    ds_id = ffi_loader.phdf5_ensure_dataset(
        fh, ds_name,
        int(n_rows), int(n_cols),
        str(jnp.dtype(A.dtype).name))

    token_spec = jax.ShapeDtypeStruct((1,), jnp.int32)

    # Each rank's (row_start, col_start) is derived in C++ from ctx->rank
    # + p/q with row-major convention — we can't pass shard_map axis_index
    # as an FFI attr (attrs must be hashable / compile-time).
    def _per_rank(A_local):
        return jax.ffi.ffi_call(_FFI_TARGET, token_spec)(
            A_local,
            ctx_handle=int(fh),
            ds_id=int(ds_id),
            n_rows=int(n_rows),
            n_cols=int(n_cols),
        )

    return shard_map(
        _per_rank,
        mesh=mesh,
        in_specs=P("x", "y"),
        out_specs=P(),
        check_rep=False,
    )(A)

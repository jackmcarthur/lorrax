"""Read-sharded-slab FFI wrapper.

Inside ``shard_map(in_specs=(), out_specs=P('x','y'))``, each process
reads its local hyperslab of the dataset via ``H5Dread`` into a pinned
host buffer, then H2D-memcpies it into the XLA-owned device output
buffer.  The host thread blocks for the duration of the read; the
CUDA device stream keeps running previously-queued ops.

Usage::

    A = read_sharded_slab(fh, "eigenvectors",
                           global_shape=(n, n),
                           dtype=jnp.complex128,
                           mesh=mesh)

Only one read is physically in flight at a time (shared pinned host
buffer), but JAX's async dispatch overlaps the read with device
compute queued before it — same "async-enough" story as the writer.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common import ffi_loader
from ..common.ffi_loader import get_lib
from .context import validate_mesh_2d

__all__ = ["read_sharded_slab"]

_FFI_TARGET = "lorrax_phdf5_read"


def read_sharded_slab(
    fh: int,
    ds_name: str,
    *,
    global_shape: tuple[int, int],
    dtype,
    mesh: Mesh,
) -> jax.Array:
    """Read a 2-D HDF5 dataset into a ``P('x','y')``-sharded JAX array.

    Parameters
    ----------
    fh
        Handle returned by :func:`open_file` (file must be opened
        ``mode='r'`` or ``'a'``).
    ds_name
        HDF5 path of the existing dataset to read.
    global_shape
        Tuple ``(n_rows, n_cols)`` — must equal the dataset's on-disk
        shape and be evenly divisible by the mesh.
    dtype
        Element dtype (``jnp.dtype`` or anything ``jnp.dtype(...)``
        accepts).  Must match the dataset's native type.
    mesh
        2-D mesh labelled ``('x','y')``.

    Returns
    -------
    jax.Array
        Globally shaped ``(n_rows, n_cols)``, sharded ``P('x','y')``
        across ``mesh``.
    """
    p, q = validate_mesh_2d(mesh)
    n_rows, n_cols = int(global_shape[0]), int(global_shape[1])
    if n_rows % p or n_cols % q:
        raise ValueError(
            f"global shape ({n_rows},{n_cols}) must divide mesh ({p},{q})")
    get_lib()

    # Collective H5Dopen on host before entering shard_map.  Returns a
    # hid_t that the FFI reuses via an int64 attr.
    ds_id = ffi_loader.phdf5_open_dataset_ro(fh, ds_name)

    local_shape = (n_rows // p, n_cols // q)
    dtype = jnp.dtype(dtype)
    out_spec = jax.ShapeDtypeStruct(local_shape, dtype)

    def _per_rank():
        return jax.ffi.ffi_call(_FFI_TARGET, out_spec)(
            ctx_handle=int(fh),
            ds_id=int(ds_id),
            n_rows=n_rows,
            n_cols=n_cols,
        )

    return shard_map(
        _per_rank,
        mesh=mesh,
        in_specs=(),
        out_specs=P("x", "y"),
        check_rep=False,
    )()

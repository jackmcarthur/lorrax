"""Per-process singleton context for cuSOLVERMp.

A cuSOLVERMp session bundles (NCCL comm, CUDA stream, handle, grid,
workspace).  We lazily build one per (mesh shape, layout) key and cache
it for the life of the process.  Torn down by ``atexit``.

Cross-process bootstrap
-----------------------
cuSOLVERMp needs its own NCCL communicator.  We build it via:

    rank 0 :  ncclGetUniqueId     -> 128 raw bytes
    all    :  broadcast_one_to_all (via jax.experimental.multihost_utils)
    each   :  ncclCommInitRank

This re-uses the JAX distributed KV store that ``jax.distributed.initialize()``
already put in place.
"""

from __future__ import annotations

import atexit
import os
import threading
from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import numpy as np
from jax.sharding import Mesh

from ..common import ffi_loader
from ..common.broadcast import broadcast_bytes

__all__ = [
    "get_or_init_context",
    "MeshKey",
]

# (p, q, layout) is the unique identity of a grid.  Reusing the same context
# across calls with matching (p, q) is fine; cusolverMp grids are reusable.
MeshKey = Tuple[int, int, bool]  # (p, q, col_major)

_LOCK = threading.Lock()
_CACHE: Dict[MeshKey, int] = {}    # MeshKey -> opaque ctx int64


def _mesh_key(mesh: Mesh, col_major: bool) -> MeshKey:
    """Extract (p, q, layout) from a 2-axis Mesh labelled ('x', 'y')."""
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"Mesh must have axes ('x', 'y'); got {mesh.axis_names}")
    p = mesh.shape["x"]
    q = mesh.shape["y"]
    return (int(p), int(q), bool(col_major))


def _broadcast_unique_id(size_bytes: int,
                         *, key: str = "lorrax_ffi/cusolvermp/nccl_unique_id/v0"
                         ) -> np.ndarray:
    """Broadcast an ncclUniqueId from rank 0 to all JAX processes."""
    ffi_loader.get_lib()
    buf = np.zeros(size_bytes, dtype=np.uint8)
    if int(jax.process_index()) == 0:
        ffi_loader.fill_nccl_unique_id(int(buf.ctypes.data))
    return broadcast_bytes(buf, key=key)


def _make_ctx(mesh: Mesh, col_major: bool) -> int:
    ffi_loader.get_lib()

    rank  = int(jax.process_index())
    world = int(jax.process_count())

    p, q, _ = _mesh_key(mesh, col_major)
    if p * q != world:
        raise ValueError(
            f"mesh.shape['x']*mesh.shape['y'] = {p*q} does not match "
            f"jax.process_count() = {world}.  cuSOLVERMp needs one rank "
            f"per grid cell in this scaffolding.")

    uid_nbytes = ffi_loader.nccl_unique_id_bytes()
    uid = _broadcast_unique_id(uid_nbytes)

    ctx = ffi_loader.create_cusolvermp_context(
        rank=rank, world_size=world,
        nccl_unique_id_addr=int(uid.ctypes.data),
        nccl_unique_id_nbytes=uid_nbytes,
        p=p, q=q,
        grid_layout_col_major=col_major,
    )
    return int(ctx)


def get_or_init_context(mesh: Mesh, *, col_major: bool = True) -> int:
    """Return the opaque int handle for the cusolverMp context of this mesh.

    Thread-safe.  On first call for a given (p,q,layout), bootstraps a new
    NCCL communicator + cusolverMp handle + grid via a collective across
    all JAX processes.  Subsequent calls return the cached handle.
    """
    key = _mesh_key(mesh, col_major)
    with _LOCK:
        h = _CACHE.get(key)
        if h is not None:
            return h
        h = _make_ctx(mesh, col_major)
        _CACHE[key] = h
        return h


# Best-effort teardown.  On abnormal exits (signals, segfaults) this may not
# run, but that's acceptable — the OS reclaims GPU memory when the process
# dies.  The destroy functions are idempotent/guarded against null.
def _atexit_teardown() -> None:
    try:
        ffi_loader.get_lib()
    except Exception:
        return
    for key, h in list(_CACHE.items()):
        try:
            ffi_loader.destroy_cusolvermp_context(int(h))
        except Exception:
            pass
    _CACHE.clear()


atexit.register(_atexit_teardown)

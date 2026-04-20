"""Per-process singleton context for the SLATE FFI.

A SLATE context is cheap to build — just a dup'd MPI_Comm plus identity
(rank, world, p, q).  We still cache one per mesh-shape to avoid
re-dup'ing the communicator on every call.

Bootstrap: SLATE wants ``MPI_Init_thread(MPI_THREAD_MULTIPLE)`` and an
MPI communicator.  phdf5's ``phdf5_init_mpi()`` already initialises MPI
the same way, so if phdf5 is used in the same process we piggyback; if
not, ``lrx_slate_init_mpi`` (exported from context.cc) inits it directly.
"""
from __future__ import annotations

import atexit
import threading
from typing import Dict, Tuple

import jax
from jax.sharding import Mesh

from ..common import ffi_loader

__all__ = ["get_or_init_context"]

MeshKey = Tuple[int, int]  # (p, q)

_LOCK = threading.Lock()
_CACHE: Dict[MeshKey, int] = {}


def _mesh_key(mesh: Mesh) -> MeshKey:
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"Mesh must have axes ('x', 'y'); got {mesh.axis_names}")
    return (int(mesh.shape["x"]), int(mesh.shape["y"]))


def _make_ctx(mesh: Mesh) -> int:
    ffi_loader.get_lib()
    rank  = int(jax.process_index())
    world = int(jax.process_count())
    p, q  = _mesh_key(mesh)
    if p * q != world:
        raise ValueError(
            f"mesh.shape['x']*mesh.shape['y'] = {p*q} does not match "
            f"jax.process_count() = {world}.  SLATE's heev wants one "
            f"process per grid cell.")
    # SLATE's heev also requires a square process grid.
    if p != q:
        raise ValueError(
            f"SLATE heev requires a square process grid (p==q); "
            f"got p={p}, q={q}.")
    return int(ffi_loader.create_slate_context(
        rank=rank, world_size=world, p=p, q=q))


def get_or_init_context(mesh: Mesh) -> int:
    """Return the opaque int handle for the SLATE context of this mesh.

    Thread-safe.  First call collectively dups MPI_COMM_WORLD.
    """
    key = _mesh_key(mesh)
    with _LOCK:
        h = _CACHE.get(key)
        if h is not None:
            return h
        h = _make_ctx(mesh)
        _CACHE[key] = h
        return h


def _atexit_teardown() -> None:
    try:
        ffi_loader.get_lib()
    except Exception:
        return
    for _, h in list(_CACHE.items()):
        try:
            ffi_loader.destroy_slate_context(int(h))
        except Exception:
            pass
    _CACHE.clear()


atexit.register(_atexit_teardown)

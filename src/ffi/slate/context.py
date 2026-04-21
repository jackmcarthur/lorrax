"""Per-process singleton context for the SLATE FFI.

A SLATE context is cheap to build — just a dup'd (and rank-remapped)
MPI_Comm plus identity (rank, world, p, q).  We cache one per mesh-shape
to avoid re-dup'ing the communicator on every call.

Bootstrap: SLATE wants ``MPI_Init_thread(MPI_THREAD_MULTIPLE)`` and an
MPI communicator.  phdf5's ``phdf5_init_mpi()`` already initialises MPI
the same way, so if phdf5 is used in the same process we piggyback; if
not, ``lrx_slate_init_mpi`` (exported from context.cc) inits it directly.

The C++ side rank-remaps MPI_COMM_WORLD so SLATE's hardcoded
``GridOrder::Col`` matches JAX's ``P('x','y')`` shard-to-rank mapping.
See ``cpp/context.cc`` for the formula and ``ffi.slate.cholesky`` for
the local-transpose pairing on the Python side.

Restrictions
------------
* Mesh must have axis names ``('x', 'y')``.  If you have a 3-D mesh
  (e.g. for batched ops with a ``'batch'`` axis), build a 2-D
  sub-mesh out of its (x, y) axes for the SLATE call — the FFI
  doesn't support sub-mesh selection yet.
* For 4 GPUs total, ``p * q == jax.process_count()``.  Sub-mesh
  partial-world support is a planned extension; see ``README.md``.
* ``heev`` further requires ``p == q``; ``potrf`` and ``trsm`` don't.
"""
from __future__ import annotations

import atexit
import threading
from typing import Dict, Tuple

import jax
from jax.sharding import Mesh

from ..common import ffi_loader

__all__ = ["get_or_init_context", "validate_mesh"]

MeshKey = Tuple[int, int]  # (p, q)

_LOCK = threading.Lock()
_CACHE: Dict[MeshKey, int] = {}


def validate_mesh(mesh: Mesh, *, require_square: bool = False) -> Tuple[int, int]:
    """Validate that ``mesh`` is usable by SLATE FFI ops; return ``(p, q)``.

    Checks: required axis names ``('x', 'y')``; ``p * q ==
    jax.process_count()``; optionally ``p == q`` (for ``heev``).
    """
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"slate FFI: mesh must have axes ('x', 'y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = int(jax.process_count())
    if p * q != world:
        raise ValueError(
            f"slate FFI: mesh {p}x{q} (={p*q}) != jax.process_count() = {world}. "
            f"Sub-mesh / partial-world calls aren't supported yet — see "
            f"src/ffi/slate/README.md.")
    if require_square and p != q:
        raise ValueError(
            f"slate FFI: this op requires a square mesh (p == q); got {p}x{q}.")
    return p, q


def _make_ctx(mesh: Mesh) -> int:
    ffi_loader.get_lib()
    rank  = int(jax.process_index())
    world = int(jax.process_count())
    p, q  = validate_mesh(mesh)
    return int(ffi_loader.create_slate_context(
        rank=rank, world_size=world, p=p, q=q))


def get_or_init_context(mesh: Mesh) -> int:
    """Return the opaque int handle for the SLATE context of this mesh.

    Thread-safe.  First call collectively dups MPI_COMM_WORLD.
    """
    key = validate_mesh(mesh)
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

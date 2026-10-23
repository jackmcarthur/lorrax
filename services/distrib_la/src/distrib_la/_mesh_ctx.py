"""The per-process MPI mesh context shared by the ScaLAPACK (host) and ELPA (ROCM) backends.

A context is a rank-remapped split of ``MPI_COMM_WORLD`` plus its identity (rank,
world, p, q), built by ``cpp/mesh/context.cc``.  The remap makes a column-major
process grid on that comm match JAX's ``P('x','y')`` shard-to-rank map.  One context
is cached per platform and mesh shape; the host and device legs resolve keys in
separate native registries, so a context created by one library is never sent to
the other.

Bootstrap: the handlers need ``MPI_Init_thread(MPI_THREAD_MULTIPLE)``.  phdf5's
``phdf5_init_mpi()`` initialises MPI the same way; if phdf5 is not used in the
process, ``lrx_mesh_init_mpi`` inits it directly.  The mesh must have axes
``('x', 'y')`` and ``p * q == jax.process_count()``.
"""
from __future__ import annotations

import atexit
import threading
from typing import Dict, Tuple

import jax
from jax.sharding import Mesh

from distrib_la import loader
from distrib_la.resolve import mesh_platform

__all__ = ["ensure_registered", "get_or_init_context", "context_key", "validate_mesh"]

ContextKey = Tuple[str, int, int]  # (platform, p, q)

_LOCK = threading.Lock()
_CACHE: Dict[ContextKey, Tuple[int, str]] = {}
#: (platform, p, q) -> the context's FFI ``ctx_key``.
_KEYS: Dict[ContextKey, int] = {}


def ensure_registered(mesh: Mesh, target: str | None = None) -> None:
    """Load + register the FFI library for the MESH's device platform (idempotent).

    ``jax.ffi.ffi_call`` resolves the target registered for the lowering platform, so
    the library that matters is the one matching the mesh's devices, not necessarily
    the default backend.  ``target`` is accepted for the callers' symmetry.
    """
    del target
    loader.get_lib(mesh_platform(mesh))


def validate_mesh(mesh: Mesh, *, require_square: bool = False) -> Tuple[int, int]:
    """Validate that ``mesh`` is usable by a mesh-context op; return ``(p, q)``.

    Checks: axis names ``('x', 'y')``; ``p * q == jax.process_count()``; optionally
    ``p == q``.
    """
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"distrib_la mesh context: mesh must have axes ('x', 'y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = int(jax.process_count())
    if p * q != world:
        raise ValueError(
            f"distrib_la mesh context: mesh {p}x{q} (={p*q}) != jax.process_count() = "
            f"{world}. Sub-mesh / partial-world calls aren't supported.")
    if require_square and p != q:
        raise ValueError(
            f"distrib_la mesh context: this op requires a square mesh (p == q); got {p}x{q}.")
    return p, q


def _ctx_config(shape, platform: str) -> str:
    """The canonical configuration string of a mesh context."""
    p, q = shape
    return f"lorrax-ctx/v1|mesh|world|{p}x{q}|{platform}"


def get_or_init_context(mesh: Mesh) -> int:
    """Return the opaque int handle for this mesh's context.  Thread-safe; the first
    call collectively splits ``MPI_COMM_WORLD``."""
    shape = validate_mesh(mesh)
    plat = mesh_platform(mesh)
    key = (plat,) + shape
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            return entry[0]
        ensure_registered(mesh)
        h = int(loader.create_mesh_context(
            rank=int(jax.process_index()), world_size=int(jax.process_count()),
            p=shape[0], q=shape[1], platform=plat))
        _KEYS[key] = loader.bind_context(plat, _ctx_config(shape, plat), h)
        _CACHE[key] = (h, plat)
        return h


def context_key(mesh: Mesh) -> int:
    """The FFI ``ctx_key`` attribute of this mesh's context (see
    ``distrib_la.loader.context_key``): a pure function of the configuration."""
    get_or_init_context(mesh)
    return _KEYS[(mesh_platform(mesh),) + validate_mesh(mesh)]


def _atexit_teardown() -> None:
    for k, (h, plat) in list(_CACHE.items()):
        try:
            loader.unbind_context(plat, _KEYS.pop(k, 0))
            loader.destroy_mesh_context(int(h), platform=plat)
        except Exception:                                  # noqa: BLE001
            pass
    _CACHE.clear()


atexit.register(_atexit_teardown)

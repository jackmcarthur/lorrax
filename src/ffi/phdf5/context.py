"""Python-side lifecycle for the parallel-HDF5 FFI.

``open_file`` is collective: the underlying ``lrx_phdf5_open`` calls
``MPI_Init_thread(MPI_THREAD_MULTIPLE)`` if needed, duplicates
``MPI_COMM_WORLD``, and invokes ``H5Fcreate``/``H5Fopen`` with
parallel-IO property lists cached on the context.  The returned
``int64`` handle is the address of a C++ ``PhdfCtx`` struct.

File handles are cached per-process so repeated ``open_file(path)``
calls return the same handle until ``close_file`` is called.
"""
from __future__ import annotations

import atexit
import threading
from typing import Dict, Optional, Tuple

import jax
from jax.sharding import Mesh

from ..common import ffi_loader

__all__ = ["open_file", "close_file", "platform_for_handle",
           "validate_mesh_2d"]

_LOCK = threading.Lock()
# path -> (int64 ctx handle, owning platform "CUDA"|"cpu").  The platform
# is recorded at open and used for EVERY lifecycle call on the handle: a
# PhdfCtx* allocated by one platform's .so (its heap, its HDF5/MPI state)
# must never be handed to the other library — the dual-platform
# (JAX_PLATFORMS=cuda,cpu) hazard the mesh routing exists for.  Routing
# only the open by mesh platform while close/ensure_dataset followed the
# JAX default backend was exactly that bug (audit fix/zq 2026-07-28).
_FILE_CTXS: Dict[str, Tuple[int, str]] = {}


def validate_mesh_2d(mesh: Mesh) -> tuple[int, int]:
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    return int(mesh.shape["x"]), int(mesh.shape["y"])


def _platform_for_mesh(mesh: Mesh) -> str:
    """Which FFI library owns the collective context for ``mesh`` — "CUDA"
    for a GPU mesh, "cpu" for a CPU mesh.  The phdf5 read handlers are
    registered under both platform strings (same jax.ffi target names), and
    the collective open/dataset lifecycle must go through the matching .so.
    """
    try:
        plat = mesh.devices.flat[0].platform
    except Exception:
        plat = ""
    return "CUDA" if plat in ("gpu", "cuda") else "cpu"


_MODE_FLAGS = {"w": 0, "a": 1, "r": 2}


def open_file(path: str, *, mesh: Mesh, mode: str = "w") -> int:
    """Collective open/create of a parallel-HDF5 file.

    Parameters
    ----------
    path
        Absolute filesystem path for the HDF5 file.  Must be reachable
        by every JAX process (shared FS — Lustre/GPFS/NFS).
    mesh
        2-D ``Mesh`` labelled ``('x','y')``.  Shape must equal
        ``jax.process_count()``.
    mode
        ``'w'`` truncate+create, ``'a'`` append-or-create, ``'r'``
        read-only.

    Returns
    -------
    int
        Opaque handle (int64 address of the C++ ``PhdfCtx``).  Pass to
        ``write_sharded_slab`` and ``close_file``.
    """
    if mode not in _MODE_FLAGS:
        raise ValueError(f"mode must be w/a/r; got {mode!r}")
    platform = _platform_for_mesh(mesh)
    ffi_loader.get_lib(platform)
    p, q = validate_mesh_2d(mesh)
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}×{q}={p*q} != jax.process_count()={jax.process_count()}")

    with _LOCK:
        if path in _FILE_CTXS:
            return _FILE_CTXS[path][0]
        ctx = ffi_loader.phdf5_open(
            path, p, q,
            int(jax.process_index()), int(jax.process_count()),
            _MODE_FLAGS[mode], platform=platform,
        )
        _FILE_CTXS[path] = (ctx, platform)
        return ctx


def platform_for_handle(ctx_handle: int) -> Optional[str]:
    """The platform ("CUDA"/"cpu") whose library allocated ``ctx_handle``,
    or None for an unknown handle.  Every lifecycle call on a handle
    (``phdf5_close`` / ``phdf5_ensure_dataset`` / ``phdf5_open_dataset_ro``)
    must go through the owning platform's .so — this is the lookup the
    write-side lifecycle sites use to route theirs."""
    with _LOCK:
        for _ctx, _plat in _FILE_CTXS.values():
            if _ctx == int(ctx_handle):
                return _plat
    return None


def close_file(path_or_handle) -> None:
    """Collective close.  Accepts either a path (the original open_file
    argument) or the int handle returned from open_file.

    Routed to the platform library that OPENED the handle (recorded in
    ``_FILE_CTXS`` at open) — not the JAX default backend, which in a
    dual-platform process could be the other library and would then free
    a foreign PhdfCtx* against foreign HDF5/MPI state.  An unknown handle
    (not opened through this module) falls back to the default-backend
    library, the pre-existing best guess."""
    global _FILE_CTXS
    with _LOCK:
        platform = None
        if isinstance(path_or_handle, str):
            entry = _FILE_CTXS.pop(path_or_handle, None)
            ctx = entry[0] if entry is not None else None
            platform = entry[1] if entry is not None else None
        else:
            ctx = int(path_or_handle)
            # Drop any path entries pointing at this ctx, keeping its
            # recorded platform.
            for k in [k for k, v in _FILE_CTXS.items() if v[0] == ctx]:
                platform = _FILE_CTXS[k][1]
                _FILE_CTXS.pop(k, None)
        if ctx is not None and ctx != 0:
            # platform=None (unknown handle) follows the JAX default
            # backend inside ffi_loader — hardcoding "CUDA" would fail on
            # a CPU node where only the host lib exists.
            ffi_loader.phdf5_close(int(ctx), platform=platform)


def _atexit_close_all() -> None:
    """Close any files still open at process exit — catches forgotten
    close_file calls.  Runs on every process.  Each handle closes through
    its own recorded platform library."""
    with _LOCK:
        for path, (ctx, platform) in list(_FILE_CTXS.items()):
            try:
                ffi_loader.phdf5_close(int(ctx), platform=platform)
            except Exception:
                pass
        _FILE_CTXS.clear()


atexit.register(_atexit_close_all)

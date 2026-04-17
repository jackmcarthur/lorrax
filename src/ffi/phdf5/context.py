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
from typing import Dict

import jax
from jax.sharding import Mesh

from ..common import ffi_loader

__all__ = ["open_file", "close_file", "validate_mesh_2d"]

_LOCK = threading.Lock()
_FILE_CTXS: Dict[str, int] = {}              # path -> int64 ctx handle


def validate_mesh_2d(mesh: Mesh) -> tuple[int, int]:
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    return int(mesh.shape["x"]), int(mesh.shape["y"])


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
    ffi_loader.get_lib()
    p, q = validate_mesh_2d(mesh)
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}×{q}={p*q} != jax.process_count()={jax.process_count()}")

    with _LOCK:
        if path in _FILE_CTXS:
            return _FILE_CTXS[path]
        ctx = ffi_loader.phdf5_open(
            path, p, q,
            int(jax.process_index()), int(jax.process_count()),
            _MODE_FLAGS[mode],
        )
        _FILE_CTXS[path] = ctx
        return ctx


def close_file(path_or_handle) -> None:
    """Collective close.  Accepts either a path (the original open_file
    argument) or the int handle returned from open_file."""
    global _FILE_CTXS
    ffi_loader.get_lib()
    with _LOCK:
        if isinstance(path_or_handle, str):
            ctx = _FILE_CTXS.pop(path_or_handle, None)
        else:
            ctx = int(path_or_handle)
            # Drop any path entries pointing at this ctx.
            for k in [k for k, v in _FILE_CTXS.items() if v == ctx]:
                _FILE_CTXS.pop(k, None)
        if ctx is not None and ctx != 0:
            ffi_loader.phdf5_close(int(ctx))


def _atexit_close_all() -> None:
    """Close any files still open at process exit — catches forgotten
    close_file calls.  Runs on every process."""
    try:
        ffi_loader.get_lib()
    except Exception:
        return
    with _LOCK:
        for path, ctx in list(_FILE_CTXS.items()):
            try:
                ffi_loader.phdf5_close(int(ctx))
            except Exception:
                pass
        _FILE_CTXS.clear()


atexit.register(_atexit_close_all)

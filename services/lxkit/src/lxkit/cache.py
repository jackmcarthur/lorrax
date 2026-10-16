"""The one per-user root for LORRAX's persistent caches.

``$SCRATCH/.cache/lorrax/<name>``, or ``~/.cache/lorrax/<name>`` where the
site defines no ``SCRATCH``: on scratch, never beside an input, one tree per
user.  The XLA compile cache (``common.jax_compile_cache``), the mathdx cubin
cache (``ffi.fft.cubin_cache_dir``), the WFN time-reversal stamps
(``symmetry_maps.density_symmetry_check``) and the ``tests/hsuite`` run tree
(``tests.hsuite.rank_session``) each take one subdirectory.  Each
owner keeps its own key rule, format and pruning; this module owns where the
tree is and how a file in it is published (:func:`atomic_write_bytes`): whole
or not at all, under a temporary name unique per host, process and call, so
ranks, jobs and nodes sharing one tree on Lustre never read a partial file and
never write into each other's temporaries.  Stdlib only.
"""

from __future__ import annotations

import os
import secrets
import socket
from pathlib import Path

__all__ = ["atomic_write_bytes", "user_cache_dir"]


def user_cache_dir(name: str) -> Path:
    """``<root>/.cache/lorrax/<name>`` with root ``$SCRATCH`` or ``~``."""
    if not name or os.sep in name or name in {".", ".."}:
        raise ValueError(f"cache name must be one path component: {name!r}")
    root = os.environ.get("SCRATCH") or os.path.expanduser("~")
    return Path(root) / ".cache" / "lorrax" / name


def atomic_write_bytes(path, data: bytes) -> Path:
    """Publish ``data`` at ``path`` whole or not at all; the path.

    The bytes go to a temporary beside the target, named for this host,
    process and call and created with ``O_EXCL`` (a pid alone repeats across
    nodes on a shared tree), are fsynced, and are renamed over the target.  A
    reader therefore sees the previous file or the new one, never a partial
    one; concurrent writers of one key each publish a whole file and the last
    rename wins.  On failure the temporary is removed, the target is untouched
    and the error propagates.  The directory fsync is best effort.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}."
                         f"{secrets.token_hex(6)}.tmp")
    created = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created = True
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if created:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return path

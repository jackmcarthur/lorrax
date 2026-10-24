"""The one per-user root for LORRAX's persistent caches.

``$SCRATCH/.cache/lorrax/<name>``, or ``${XDG_CACHE_HOME:-~/.cache}/lorrax/<name>``
where the site defines no writable ``SCRATCH``: on scratch, never beside an input, one tree per
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

__all__ = ["atomic_write_bytes", "user_cache_dir", "user_cache_root"]


def user_cache_root() -> Path:
    """The per-user tree every LORRAX cache lives in: ``LORRAX_CACHE_ROOT`` when
    set, else ``$SCRATCH/.cache/lorrax`` when ``SCRATCH`` names a writable
    directory, else ``${XDG_CACHE_HOME:-~/.cache}/lorrax``.

    The default follows the site convention: HPC centres export ``SCRATCH``
    for the fast, large, per-user file system, and a home directory is small and
    often slow on compute nodes.  An exported ``SCRATCH`` that is missing or
    read-only falls back to the user cache directory instead of failing the
    first write.  ``LORRAX_CACHE_ROOT`` is the fix the install self-check
    (``scripts/perf_check.py``) names when that default is slow per file (a
    cluster whose only cache candidate is an NFS or DVS home).
    ``config/perlmutter/gpu_env.sh`` applies the same rule to
    ``CUDA_CACHE_PATH``.
    """
    explicit = os.environ.get("LORRAX_CACHE_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    scratch = os.environ.get("SCRATCH", "")
    if scratch and os.path.isdir(scratch) and os.access(scratch, os.W_OK | os.X_OK):
        return Path(scratch) / ".cache" / "lorrax"
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "lorrax"


def user_cache_dir(name: str) -> Path:
    """``<user_cache_root()>/<name>``: one subdirectory per cache owner."""
    if not name or os.sep in name or name in {".", ".."}:
        raise ValueError(f"cache name must be one path component: {name!r}")
    return user_cache_root() / name


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

"""The one per-user root for LORRAX's persistent caches.

``$SCRATCH/.cache/lorrax/<name>``, or ``~/.cache/lorrax/<name>`` where the
site defines no ``SCRATCH``: on scratch, never beside an input, one tree per
user.  The XLA compile cache (``common.jax_compile_cache``), the mathdx cubin
cache (``ffi.fft.cubin_cache_dir``) and the WFN time-reversal stamps
(``symmetry_maps.density_symmetry_check``) each take one subdirectory.  Each
owner keeps its own key rule, format and pruning; this module owns only where
the tree is.  Stdlib only.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["user_cache_dir"]


def user_cache_dir(name: str) -> Path:
    """``<root>/.cache/lorrax/<name>`` with root ``$SCRATCH`` or ``~``."""
    if not name or os.sep in name or name in {".", ".."}:
        raise ValueError(f"cache name must be one path component: {name!r}")
    root = os.environ.get("SCRATCH") or os.path.expanduser("~")
    return Path(root) / ".cache" / "lorrax" / name

"""Re-export shim — implementation moved to ``ffi/io.py`` (wave 2).

``read_kchunk_sharded`` was on this list until 2026-08-07; it had zero
callers and lost the measurement to ``read_kchunk_union_sharded`` at every
deck (``ffi/io.py``'s reader docstring has the numbers), so it was deleted
rather than re-exported.  A shim that keeps a name its implementation no
longer has is how a deleted spelling acquires a second life.
"""
from ..io import (  # noqa: F401
    ffi_read_call, ffi_read_kchunk_call, ffi_read_kchunk_union_call,
    read_sharded_slab, read_kchunk_union_sharded,
)

__all__ = [
    "ffi_read_call",
    "ffi_read_kchunk_call",
    "ffi_read_kchunk_union_call",
    "read_sharded_slab",
    "read_kchunk_union_sharded",
]

"""Re-export shim — implementation moved to ``ffi/io.py`` (wave 2)."""
from ..io import (  # noqa: F401
    ffi_read_call, ffi_read_kchunk_call, ffi_read_kchunk_union_call,
    read_sharded_slab, read_kchunk_sharded, read_kchunk_union_sharded,
)

__all__ = [
    "ffi_read_call",
    "ffi_read_kchunk_call",
    "ffi_read_kchunk_union_call",
    "read_sharded_slab",
    "read_kchunk_sharded",
    "read_kchunk_union_sharded",
]

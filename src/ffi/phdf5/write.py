"""Re-export shim — implementation moved to ``ffi/io.py`` (wave 2)."""
from ..io import write_sharded_slab, ffi_write_call  # noqa: F401

__all__ = ["write_sharded_slab", "ffi_write_call"]

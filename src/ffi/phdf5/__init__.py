"""Re-export shim — the implementation moved to ``ffi/io.py`` (wave 2,
docs/architecture/ffi_layout.md §3/§6, 2026-08-01).  Consumers
(file_io/_slab_io_ffi.py, file_io/wfn_loader.py) still import
``ffi.phdf5``; deleting this package is the gate that the consumer
migration to ``ffi.io`` is complete (`grep -rn "ffi\\.phdf5"` empty).
Nothing here but re-exports."""
from ..io import (  # noqa: F401
    open_file, close_file, write_sharded_slab, read_sharded_slab,
)

__all__ = [
    "open_file", "close_file",
    "write_sharded_slab", "read_sharded_slab",
]

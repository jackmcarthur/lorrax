"""Re-export shim — implementation moved to ``ffi/linalg/_slate.py`` (wave 2)."""
from ..linalg._slate import (  # noqa: F401
    ensure_registered, get_or_init_context,
    validate_mesh, validate_tile_layout,
)

__all__ = [
    "ensure_registered",
    "get_or_init_context",
    "validate_mesh",
    "validate_tile_layout",
]

"""Re-export shim — implementation moved to ``ffi/io.py`` (wave 2)."""
from ..io import (  # noqa: F401
    open_file, close_file, platform_for_handle, validate_mesh_2d,
    _platform_for_mesh,
)

__all__ = ["open_file", "close_file", "platform_for_handle",
           "validate_mesh_2d"]

"""Re-export shim — implementation moved to ``ffi/linalg/_slate.py`` (wave 2)."""
from ..linalg._slate import distributed_eigh  # noqa: F401

__all__ = ["distributed_eigh"]

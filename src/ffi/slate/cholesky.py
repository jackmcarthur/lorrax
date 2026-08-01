"""Re-export shim — implementation moved to ``ffi/linalg/_slate.py`` (wave 2)."""
from ..linalg._slate import SlateLowerL, distributed_cholesky  # noqa: F401

__all__ = ["SlateLowerL", "distributed_cholesky"]

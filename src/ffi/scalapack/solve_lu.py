"""Re-export shim — implementation moved to ``ffi/linalg/_scalapack.py`` (wave 2)."""
from ..linalg._scalapack import batched_distributed_solve_lu  # noqa: F401

__all__ = ["batched_distributed_solve_lu"]

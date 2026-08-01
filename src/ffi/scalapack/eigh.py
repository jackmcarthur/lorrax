"""Re-export shim — implementation moved to ``ffi/linalg/_scalapack.py`` (wave 2)."""
from ..linalg._scalapack import (  # noqa: F401
    batched_distributed_eigh, distributed_eigh, validate_eigh_mesh,
)

__all__ = ["batched_distributed_eigh", "distributed_eigh", "validate_eigh_mesh"]

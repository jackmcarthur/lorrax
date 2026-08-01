"""Re-export shim — implementation moved to ``ffi/linalg/_scalapack.py`` (wave 2).

The LU family is three ops since the transverse factor hoist (2026-08):
the fused solve plus the split getrf/getrs pair (factor once per channel,
back-solve per r-chunk).
"""
from ..linalg._scalapack import (  # noqa: F401
    batched_distributed_getrf,
    batched_distributed_getrs,
    batched_distributed_solve_lu,
)

__all__ = ["batched_distributed_solve_lu",
           "batched_distributed_getrf",
           "batched_distributed_getrs"]

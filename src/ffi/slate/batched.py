"""Re-export shim — implementation moved to ``ffi/linalg/_slate.py`` (wave 2).

``_mesh_key`` stays importable from here: ``ffi.scalapack`` used to reach it
via this path and ``common/collectives.py`` documents it as the canonical
mesh cache key.
"""
from ..linalg._slate import (  # noqa: F401
    SlateBatchedLowerL,
    _mesh_key,
    batched_distributed_cholesky,
    batched_distributed_trsm,
)

__all__ = [
    "SlateBatchedLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_trsm",
]

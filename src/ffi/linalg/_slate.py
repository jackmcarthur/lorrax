"""Re-export shim — implementation moved to ``distrib_la._slate``.

Deletion is the replumb-complete gate.  Reachers: the ``ffi.slate``
re-export package (five modules), and ``isdf/core.py:46``
(``from ffi.linalg._slate import _mesh_key`` — the one private name a
lorrax module still reaches for; it keys the ζ-fit's own kernel cache).
"""
from distrib_la._slate import SlateBatchedLowerL  # noqa: F401
from distrib_la._slate import SlateLowerL  # noqa: F401
from distrib_la._slate import _mesh_key  # noqa: F401
from distrib_la._slate import batched_distributed_cholesky  # noqa: F401
from distrib_la._slate import batched_distributed_trsm  # noqa: F401
from distrib_la._slate import distributed_cholesky  # noqa: F401
from distrib_la._slate import distributed_eigh  # noqa: F401
from distrib_la._slate import distributed_trsm  # noqa: F401
from distrib_la._slate import ensure_registered  # noqa: F401
from distrib_la._slate import get_or_init_context  # noqa: F401
from distrib_la._slate import get_or_init_subrow_context  # noqa: F401
from distrib_la._slate import validate_mesh  # noqa: F401
from distrib_la._slate import validate_tile_layout  # noqa: F401

__all__ = [
    "SlateBatchedLowerL", "SlateLowerL",
    "batched_distributed_cholesky", "batched_distributed_trsm",
    "distributed_cholesky", "distributed_eigh", "distributed_trsm",
    "ensure_registered", "get_or_init_context", "get_or_init_subrow_context",
    "validate_mesh", "validate_tile_layout",
]

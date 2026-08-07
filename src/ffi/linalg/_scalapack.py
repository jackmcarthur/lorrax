"""Re-export shim — implementation moved to ``distrib_la._scalapack``.

Deletion is the replumb-complete gate.  Reachers: the ``ffi.scalapack``
re-export package (three modules).
"""
from distrib_la._scalapack import batched_distributed_eigh  # noqa: F401
from distrib_la._scalapack import batched_distributed_getrf  # noqa: F401
from distrib_la._scalapack import batched_distributed_getrs  # noqa: F401
from distrib_la._scalapack import batched_distributed_solve_lu  # noqa: F401
from distrib_la._scalapack import distributed_eigh  # noqa: F401
from distrib_la._scalapack import validate_eigh_mesh  # noqa: F401

__all__ = [
    "batched_distributed_eigh", "batched_distributed_getrf",
    "batched_distributed_getrs", "batched_distributed_solve_lu",
    "distributed_eigh", "validate_eigh_mesh",
]

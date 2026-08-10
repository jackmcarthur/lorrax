"""Re-export shim — implementation moved to ``distrib_la._cusolvermp``.

Deletion is the replumb-complete gate.  The surviving reachers are the five
bench drivers under ``tests/bench/`` — the count ``ffi.cusolvermp.batched: 5``
in ``tests/test_layering.py`` is exactly those.  ``tests/test_ffi_linalg_contract.py``,
also named here once, migrated to
``services/distrib_la/tests/test_distrib_la_contract.py`` and reaches
``distrib_la`` directly.
"""
from distrib_la._cusolvermp import CusolverMpBatchedLowerL  # noqa: F401
from distrib_la._cusolvermp import batched_distributed_cholesky  # noqa: F401
from distrib_la._cusolvermp import batched_distributed_potrs  # noqa: F401
from distrib_la._cusolvermp import batched_distributed_solve_lu  # noqa: F401
from distrib_la._cusolvermp import cholesky_handle_to_natural_L  # noqa: F401

__all__ = [
    "CusolverMpBatchedLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_potrs",
    "batched_distributed_solve_lu",
    "cholesky_handle_to_natural_L",
]

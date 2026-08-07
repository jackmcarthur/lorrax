"""Re-export shim — implementation moved to ``distrib_la._cusolvermp``.

Deletion is the replumb-complete gate.  Reachers:
``tests/bench/cusolvermp_eigh_test.py`` and
``tests/test_ffi_linalg_contract.py:380``.
"""
from distrib_la._cusolvermp import distributed_eigh  # noqa: F401

__all__ = ["distributed_eigh"]

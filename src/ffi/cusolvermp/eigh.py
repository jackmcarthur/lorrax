"""Re-export shim — implementation moved to ``distrib_la._cusolvermp``.

Deletion is the replumb-complete gate.  No caller remains:
``services/distrib_la/bench/cusolvermp_eigh_test.py`` imports
``distrib_la._cusolvermp``.  The second
reacher this note used to name, ``tests/test_ffi_linalg_contract.py:380``,
migrated to ``services/distrib_la/tests/test_distrib_la_contract.py`` and
reaches ``distrib_la`` directly.
"""
from distrib_la._cusolvermp import distributed_eigh  # noqa: F401

__all__ = ["distributed_eigh"]

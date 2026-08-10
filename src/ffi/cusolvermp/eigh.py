"""Re-export shim — implementation moved to ``distrib_la._cusolvermp``.

Deletion is the replumb-complete gate.  The one surviving reacher is
``tests/bench/cusolvermp_eigh_test.py`` — the count ``ffi.cusolvermp.eigh: 1``
in ``tests/test_layering.py`` is that bench and nothing else.  The second
reacher this note used to name, ``tests/test_ffi_linalg_contract.py:380``,
migrated to ``services/distrib_la/tests/test_distrib_la_contract.py`` and
reaches ``distrib_la`` directly.
"""
from distrib_la._cusolvermp import distributed_eigh  # noqa: F401

__all__ = ["distributed_eigh"]

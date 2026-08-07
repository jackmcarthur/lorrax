"""Re-export shim — implementation moved to ``distrib_la.dispatch``.

Deletion is the replumb-complete gate.  Reachers: ``gw/qsgw_density.py:659``
and ``tests/multi_device/batched_eigh_dispatch_gate.py:57``.
"""
from distrib_la.dispatch import EIGH_BACKENDS  # noqa: F401
from distrib_la.dispatch import dispatch_batched_eigh  # noqa: F401

__all__ = ["dispatch_batched_eigh", "EIGH_BACKENDS"]

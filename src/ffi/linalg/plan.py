"""Re-export shim — implementation moved to ``distrib_la.plan``.

Deletion is the replumb-complete gate.  Reachers: ``gw/w_isdf.py:491``
(``from ffi.linalg.plan import plan``), ``tests/multi_device/
batched_eigh_dispatch_gate.py:152`` (``sys.modules['ffi.linalg.plan']``).
"""
from distrib_la.plan import Plan  # noqa: F401
from distrib_la.plan import ensure_sharding  # noqa: F401
from distrib_la.plan import plan  # noqa: F401

#: Old name for :class:`distrib_la.Plan` — docstring references only.
LinalgPlan = Plan

__all__ = ["LinalgPlan", "Plan", "ensure_sharding", "plan"]

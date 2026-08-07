"""Re-export shim — implementation moved to ``distrib_la.resolve``.

Deletion is the replumb-complete gate.  Reachers: ``gw/gw_config.py:443``
and ``tests/test_bse_setup_qchunk.py:335`` (``BACKEND_CHOICES``),
``runtime/__init__.py:1577`` (``list_backends``).
"""
from distrib_la.resolve import BACKEND_CHOICES  # noqa: F401
from distrib_la.resolve import CHOLESKY_BACKENDS  # noqa: F401
from distrib_la.resolve import EIGH_BACKENDS  # noqa: F401
from distrib_la.resolve import LU_BACKENDS  # noqa: F401
from distrib_la.resolve import NATIVE  # noqa: F401
from distrib_la.resolve import NATIVE2D  # noqa: F401
from distrib_la.resolve import OPS  # noqa: F401
from distrib_la.resolve import backend_module  # noqa: F401
from distrib_la.resolve import list_backends  # noqa: F401
from distrib_la.resolve import mesh_is_cpu  # noqa: F401
from distrib_la.resolve import mesh_platform  # noqa: F401
from distrib_la.resolve import resolve_backend  # noqa: F401

__all__ = [
    "BACKEND_CHOICES", "CHOLESKY_BACKENDS", "EIGH_BACKENDS", "LU_BACKENDS",
    "NATIVE", "NATIVE2D", "OPS", "backend_module", "list_backends",
    "mesh_is_cpu", "mesh_platform", "resolve_backend",
]

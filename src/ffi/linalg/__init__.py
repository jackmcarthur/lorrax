"""Re-export shim — the distributed-linalg facade moved to ``distrib_la``.

``services/distrib_la/`` is the standalone service now (charter wave 0);
this package only re-exports its public names so lorrax stays green while
the call sites migrate.  New code imports ``distrib_la`` directly.

DELETING THIS PACKAGE IS THE REPLUMB-COMPLETE GATE.  Its last import
disappearing from ``src/`` is what "the replumb landed" means, and that is
checkable: ``tests/test_layering.py``'s service-door rule counts the edges.

The submodules (``plan``, ``resolve``, ``dispatch``, ``_slate``,
``_scalapack``) are re-exported through the same package-relative imports
the original used, so ``sys.modules['ffi.linalg.plan']`` and
``from ffi.linalg._slate import _mesh_key`` keep resolving exactly as they
did.  Each names its reachers, so the replumb knows what it is removing.
"""
from __future__ import annotations

from ffi import _services

_services.ensure_on_path()

from .dispatch import dispatch_batched_eigh                    # noqa: E402
from .plan import LinalgPlan, Plan, ensure_sharding, plan      # noqa: E402
from .resolve import (                                         # noqa: E402
    BACKEND_CHOICES,
    CHOLESKY_BACKENDS,
    EIGH_BACKENDS,
    LU_BACKENDS,
    NATIVE,
    OPS,
    backend_module,
    list_backends,
    mesh_is_cpu,
    mesh_platform,
    resolve_backend,
)

__all__ = [
    "BACKEND_CHOICES",
    "CHOLESKY_BACKENDS",
    "EIGH_BACKENDS",
    "LU_BACKENDS",
    "LinalgPlan",
    "NATIVE",
    "OPS",
    "Plan",
    "backend_module",
    "dispatch_batched_eigh",
    "ensure_sharding",
    "list_backends",
    "mesh_is_cpu",
    "mesh_platform",
    "plan",
    "resolve_backend",
]

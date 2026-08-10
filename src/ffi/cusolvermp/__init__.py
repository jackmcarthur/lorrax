"""Re-export shim — the cuSOLVERMp backend moved to ``distrib_la``.

``distrib_la._cusolvermp`` merged the former ``context.py``/``eigh.py``/
``batched.py`` into one backend module, the shape ``_slate`` and
``_scalapack`` already had.  ``distrib_la.backend_module('cusolvermp')``
hands out that module; this package keeps the old import path working for
the bench drivers under ``tests/bench/``.  It used to also serve
``tests/test_ffi_linalg_contract.py``; that file migrated to
``services/distrib_la/tests/test_distrib_la_contract.py`` (marker
``distrib_la``) and no longer reaches this path.

Deletion is the replumb-complete gate.  **cuSOLVERMp itself is NOT a
deletion candidate** — ``ffi_layout.md`` §5 once listed it and the phase
charter overturned that ruling; the premises are measurably false (zero
``src/`` vendor imports, SLATE has no LU handler at all).
"""
from __future__ import annotations

from ffi import _services

_services.ensure_on_path()

from .batched import CusolverMpBatchedLowerL  # noqa: E402,F401
from .batched import batched_distributed_cholesky  # noqa: E402,F401
from .batched import batched_distributed_potrs  # noqa: E402,F401
from .batched import batched_distributed_solve_lu  # noqa: E402,F401
from .batched import cholesky_handle_to_natural_L  # noqa: E402,F401
from .eigh import distributed_eigh  # noqa: E402,F401

__all__ = [
    "CusolverMpBatchedLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_potrs",
    "batched_distributed_solve_lu",
    "cholesky_handle_to_natural_L",
    "distributed_eigh",
]

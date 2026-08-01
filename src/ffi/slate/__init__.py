"""Re-export shim — the SLATE backend implementation moved to
``ffi/linalg/_slate.py`` (wave 2, docs/architecture/ffi_layout.md §3/§6,
2026-08-01).  ``linalg.resolve.backend_module("slate")`` hands out this
package; consumers and the bench tests import these names from here.
Deleting this package is the gate that the consumer migration to
``ffi.linalg._slate`` is complete.

SLATE is a tile-based, MPI + GPU dense linear-algebra library from ICL,
driven from JAX FFI handlers, one process per GPU on a 2-D ``('x','y')``
mesh; see ``src/ffi/slate/README.md`` and the module docstrings in
``ffi/linalg/_slate.py`` for layout/MPI/GPU-comm notes.
"""
from __future__ import annotations

from ..linalg._slate import (  # noqa: F401
    SlateBatchedLowerL,
    SlateLowerL,
    batched_distributed_cholesky,
    batched_distributed_trsm,
    distributed_cholesky,
    distributed_eigh,
    distributed_trsm,
)

__all__ = [
    "SlateBatchedLowerL",
    "SlateLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_trsm",
    "distributed_cholesky",
    "distributed_eigh",
    "distributed_trsm",
]

"""SLATE FFI — distributed dense linear algebra on GPUs from JAX.

SLATE is a tile-based, MPI + GPU library from ICL.  This package calls
SLATE from inside JAX FFI handlers, one process per GPU on a 2-D
``('x', 'y')`` mesh.  See ``src/ffi/slate/README.md`` for the design
notes (layout, MPI, GPU-aware comms) and ``src/ffi/slate/{cholesky,
trsm, eigh}.py`` for per-op shape contracts.

Public API::

    from ffi.slate import distributed_cholesky, distributed_trsm
    L = distributed_cholesky(A, mesh=mesh)        # SlateLowerL handle
    X = distributed_trsm(L, B, mesh=mesh, op='N') # forward solve

    from ffi.slate import distributed_eigh
    W, Q = distributed_eigh(A, mesh=mesh)         # eigenvectors buggy; W ok

cholesky + trsm work on any p×q mesh (p == q for eigh).
"""
from __future__ import annotations

from .cholesky import SlateLowerL, distributed_cholesky  # noqa: F401
from .eigh import distributed_eigh  # noqa: F401
from .trsm import distributed_trsm  # noqa: F401

__all__ = [
    "SlateLowerL",
    "distributed_cholesky",
    "distributed_eigh",
    "distributed_trsm",
]

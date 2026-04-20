"""SLATE FFI — distributed Hermitian eigendecomposition via ``slate::heev``.

Usage::

    from ffi.slate import distributed_eigh
    W, Q = distributed_eigh(A, mesh=mesh)

SLATE is a tile-based, MPI + GPU linear algebra library from ICL.  This
wrapper calls ``slate::heev`` from inside a JAX FFI, one process per GPU
on a square ``('x','y')`` mesh.  See ``eigh.py`` for the shape contract.
"""
from __future__ import annotations

from .eigh import distributed_eigh  # noqa: F401

__all__ = ["distributed_eigh"]

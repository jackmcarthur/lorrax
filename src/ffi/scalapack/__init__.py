"""Re-export shim — the ScaLAPACK backend implementation moved to
``ffi/linalg/_scalapack.py`` (wave 2, docs/architecture/ffi_layout.md
§3/§6, 2026-08-01).  ``linalg.resolve.backend_module("scalapack")`` hands
out this package.  Deleting this package is the gate that the consumer
migration is complete.

**ScaLAPACK is a published API, not a product.**  The ops call the
eleven-symbol ScaLAPACK + C-BLACS Fortran ABI hand-declared in
``src/ffi/cpp/scalapack/blacs_grid.h``; which implementation is linked
(Intel MKL on Frontera, Cray LibSci, netlib, AOCL, SLATE's overlay) is
decided entirely in ``src/ffi/cpp/CMakeLists.txt`` and nothing in this
package can observe it — do not write a vendor name into this family's
code or docstrings as if it were a fact about the build.

The ops (all need a SQUARE or 1-D ``('x','y')`` mesh):

* :func:`batched_distributed_solve_lu` — fused ``pXgetrf`` + ``pXgetrs``,
  the ``distributed_lu`` axis (legacy callers).
* :func:`batched_distributed_getrf` / :func:`batched_distributed_getrs` —
  the SPLIT pair for the hoisted transverse ζ factor stage (factor once
  per channel, back-solve per r-chunk; ipiv threaded verbatim).
* :func:`batched_distributed_eigh` / :func:`distributed_eigh` —
  ``pzheevd`` / ``pdsyevd``, the permanent CPU backend for distributed
  eigh (SLATE's host ``heev`` SIGSEGVs; bug L-2).
"""
from ..linalg._scalapack import (  # noqa: F401
    batched_distributed_eigh,
    batched_distributed_getrf,
    batched_distributed_getrs,
    batched_distributed_solve_lu,
    distributed_eigh,
)

__all__ = [
    "batched_distributed_eigh",
    "distributed_eigh",
    "batched_distributed_solve_lu",
    "batched_distributed_getrf",
    "batched_distributed_getrs",
]

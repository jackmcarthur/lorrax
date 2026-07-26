"""ScaLAPACK FFI — host-platform distributed linear algebra.

The portable CPU-backend twin of ``ffi.cusolvermp``.  Handlers live in
``liblorrax_ffi_host.so`` (see ``src/ffi/common/cpp/host/``); ScaLAPACK +
BLACS come from the vendor BLAS the host library already links (MKL on
Frontera, Cray LibSci on Perlmutter), so neither op adds a dependency.

Two ops:

* :func:`batched_distributed_solve_lu` — ``pXgetrf`` + ``pXgetrs``, the
  ``distributed_lu`` axis (transverse ζ channels).
* :func:`batched_distributed_eigh` / :func:`distributed_eigh` —
  ``pzheevd`` / ``pdsyevd``, the **permanent CPU backend for distributed
  eigh** (SLATE's host ``heev`` SIGSEGVs; bug L-2).

Both need a SQUARE or 1-D ``('x','y')`` mesh: the ScaLAPACK descriptors
want square blocks (``MB == NB``), which the one-tile-per-rank layout only
provides there.
"""
from .eigh import batched_distributed_eigh, distributed_eigh
from .solve_lu import batched_distributed_solve_lu

__all__ = [
    "batched_distributed_eigh",
    "distributed_eigh",
    "batched_distributed_solve_lu",
]

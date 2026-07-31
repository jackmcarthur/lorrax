"""ScaLAPACK FFI — host-platform distributed linear algebra.

The portable CPU-backend twin of ``ffi.cusolvermp``.  Handlers live in
``liblorrax_ffi_host.so`` (see ``src/ffi/common/cpp/host/``).

**ScaLAPACK is a published API, not a product.**  These two ops call the
eight-symbol ScaLAPACK + C-BLACS Fortran ABI hand-declared in
``src/ffi/scalapack/cpp/blacs_grid.h``, and that ABI is implemented by
Intel MKL (Frontera), Cray LibSci (Perlmutter), netlib, AMD AOCL, and
SLATE's own ``scalapack_api`` layer.  Which one is linked is decided
entirely in ``src/ffi/common/cpp/host/CMakeLists.txt``
(``LORRAX_SCALAPACK_LIBRARIES`` overrides the MKL probe with any link
line at all) and **nothing in this package can observe it** — the
capability check keys only on LORRAX's own handler symbols.  So do not
write a vendor name into this family's code or docstrings as if it were
a fact about the build.  Whichever it is, the host library already links
it for the other host handlers, so neither op adds a dependency.

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

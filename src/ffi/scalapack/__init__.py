"""ScaLAPACK FFI — host-platform distributed linear algebra (Cray LibSci).

The portable CPU-backend twin of ``ffi.cusolvermp`` for the
``distributed_lu`` axis.  Handlers live in ``liblorrax_ffi_host.so``
(see ``src/ffi/common/cpp/host/``); no extra dependency beyond the
libsci the host library already links.
"""
from .solve_lu import batched_distributed_solve_lu

__all__ = ["batched_distributed_solve_lu"]

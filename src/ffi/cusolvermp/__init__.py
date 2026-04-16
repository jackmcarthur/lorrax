"""cuSOLVERMp FFI subpackage.

Public API:

    from ffi.cusolvermp import distributed_eigh

    evals, Q = distributed_eigh(A, mesh=mesh)
"""
from .eigh import distributed_eigh

__all__ = ["distributed_eigh"]

"""cuSOLVERMg (multi-GPU, single-process) FFI subpackage.

Public API:

    from ffi.cusolvermg import eigh_mg

    evals, Q = eigh_mg(A)   # A is a (n, n) jax.Array on device 0
"""
from .eigh import eigh_mg

__all__ = ["eigh_mg"]

"""cuBLASMp FFI subpackage.

Public API::

    from ffi.cublasmp import batched_distributed_gemm

    C = batched_distributed_gemm(A, B, C, mesh=mesh,
                                  alpha=1.0, beta=0.0,
                                  transa='N', transb='N')
"""
from .batched import (
    batched_distributed_gemm,
    batched_fused_w_solve,
    batched_fused_w_solve_jit,
)

__all__ = [
    "batched_distributed_gemm",
    "batched_fused_w_solve",
    "batched_fused_w_solve_jit",
]

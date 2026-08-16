"""cuBLASMp FFI compatibility subpackage.

Public API::

    from ffi.cublasmp import batched_distributed_gemm

    C = batched_distributed_gemm(A, B, C, mesh=mesh,
                                  alpha=1.0, beta=0.0,
                                  transa='N', transb='N')

The GEMM implementation now belongs to the standalone ``distrib_la``
service.  This historical spelling is a forwarding shim; new code imports
``distrib_la.matmul`` from the service door.
"""
from distrib_la import matmul as _matmul

from .batched import batched_fused_w_solve, batched_fused_w_solve_jit


def batched_distributed_gemm(A, B, C, *, mesh, alpha=1.0, beta=0.0,
                             transa="N", transb="N"):
    """Compatibility forwarding call to explicit cuBLASMp matmul."""
    return _matmul(
        A, B, C, mesh=mesh, alpha=alpha, beta=beta,
        transa=transa, transb=transb, backend="cublasmp")

__all__ = [
    "batched_distributed_gemm",
    "batched_fused_w_solve",
    "batched_fused_w_solve_jit",
]

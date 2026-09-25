"""The fused cuBLASMp W solve, ``W = X (I - X^H pref·chi X)^-1 X^H`` per q, ``X X^H = V``.

No production caller yet: the W Dyson solve is ``distrib_la``'s
``solve_lu``.  Kept as the candidate distributed large-P path (owner,
2026-09-25).  The batched cuBLASMp GEMM is ``distrib_la.matmul``.
"""
from .batched import (
    batched_fused_w_solve,
    batched_fused_w_solve_jit,
)

__all__ = [
    "batched_fused_w_solve",
    "batched_fused_w_solve_jit",
]

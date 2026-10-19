"""The distributed Cholesky of a face stack, one library per leg: C[q] = L L^H, then X = C^-1 B.

``factor`` takes C (nq, n, n) at P(None,'x','y') through ``plan('cholesky')`` and keeps it
(cuSOLVERMp factors a copy); ``solve`` takes B (nq, n, m)
in the same layout and returns X there. CUDA: cuSOLVERMp batched potrf/potrs. ROCm: SLATE
potrf and two trsm per q. The host: the plan's native route and two triangular solves. A factor that is
not positive definite raises from the library's info (cuSOLVERMp, SLATE) or comes back
non-finite (native); the caller names the refusal (isdf.cplus).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Cholesky(NamedTuple):
    backend: str
    handle: Any       # cuSOLVERMp batched handle, SLATE handles per q, or the dense L stack
    mesh: Any


def factor(C, mesh) -> Cholesky:
    """``plan('cholesky')`` on the stack; SLATE's row returns a handle per tile (``one_handle``)."""
    from distrib_la.plan import plan
    from distrib_la.resolve import mesh_platform
    # 'auto' has no ROCm row (it resolves to the replicated native factor there): SLATE, by name.
    p = plan("cholesky", mesh, n=int(C.shape[-1]), backend="slate" if mesh_platform(mesh) == "ROCM" else "auto")
    if p.backend == "slate":
        return Cholesky("slate", [p(C[q]) for q in range(C.shape[0])], mesh)
    return Cholesky(p.backend, p.batched(C + 0 if p.backend == "cusolvermp" else C), mesh)


def solve(F: Cholesky, B):
    if F.backend == "cusolvermp":
        from distrib_la._cusolvermp import batched_distributed_potrs
        return batched_distributed_potrs(F.handle, B, mesh=F.mesh)
    if F.backend == "slate":
        from distrib_la._slate import distributed_trsm
        return jnp.stack([distributed_trsm(L, distributed_trsm(L, B[q], mesh=F.mesh, op="N"), mesh=F.mesh, op="C")
                          for q, L in enumerate(F.handle)])
    tri = jax.lax.linalg.triangular_solve
    y = tri(F.handle, B, left_side=True, lower=True)
    return tri(F.handle, y, left_side=True, lower=True, conjugate_a=True, transpose_a=True)

"""Per-q cuSOLVERMp potrf + potrs on the full (Px, Py) process mesh.

Use case: a stack of ``Nq`` independent Hermitian PD matrices ``A_q`` of
shape ``(N, N)``, each 2D-sharded across the full device mesh; ``A_q``
is too big to fit on one GPU even alone, so the distribution buys
memory scalability.  Right-hand sides ``B_q`` have shape ``(N, Mrhs)``
(``Mrhs`` can be much larger than ``N``) and share the same column
sharding.

Sharding contract (matches ``distributed_eigh``'s P('x','y') layout,
extended to the batch dim):

    A : (Nq, N, N)       P(None, 'x', 'y')   # per-rank local (Nq, N/Px, N/Py)
    B : (Nq, N, Mrhs)    P(None, 'x', 'y')   # per-rank local (Nq, N/Px, Mrhs/Py)
    L : (Nq, N, N)       P(None, 'x', 'y')   # same as A
    X : (Nq, N, Mrhs)    P(None, 'x', 'y')   # same as B

No reshards in or out — the caller feeds and consumes the natural
``P(None, 'x', 'y')`` layout used throughout the ISDF / zeta pipeline.
The FFI handler loops over q and issues one ``cusolverMpPotrf`` /
``cusolverMpPotrs`` call per matrix on the world-wide (Px, Py) grid.

Why not batch across 'x' (per-X-row sub-comm) for batch parallelism?
The earlier sub-comm variant is faster per matrix (2 ranks each,
simpler NCCL) but forces the caller to reshard to ``P('x', None, 'y')``
and complicates the downstream layout.  For the ISDF workflow the net
wall-clock is similar (measured), so the simpler full-mesh layout
wins on readability + zero reshard cost.

Restrictions:
  * Mesh is 2-D with axes ('x', 'y').
  * Dtypes F64 / C128.
  * The world-wide cuSOLVERMp context must be created with
    ``col_major=False`` so tile-(i, j) → rank ``i*Py + j`` matches
    JAX's row-major mesh reshape.  ``get_or_init_context(mesh,
    col_major=False)`` handles this and caches the ctx.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_context

__all__ = [
    "CusolverMpBatchedLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_potrs",
]

_POTRF_TARGET = "lorrax_cusolvermp_batched_potrf"
_POTRS_TARGET = "lorrax_cusolvermp_batched_potrs"

# jit(shard_map(...)) cache per signature.  See
# src/ffi/phdf5/ARCHITECTURE.md §2.4: eager shard_map re-traces per call;
# wrapping in jax.jit once amortises.
_JIT_CACHE: dict = {}

def _mesh_key(mesh: Mesh):
    return (tuple(mesh.axis_names), tuple(int(s) for s in mesh.shape.values()))


@dataclass(frozen=True)
class CusolverMpBatchedLowerL:
    """Opaque handle wrapping the batched Cholesky factor.

    Attributes
    ----------
    raw : jax.Array
        Shape ``(Nq, N, N)``, sharded ``P(None, 'y', 'x')``.  Bytes are
        cuSOLVERMp's col-major L tiles per slice (from the inner-dim
        transpose that translates JAX row-major to cuSOLVERMp col-major).
        Pass this back to ``batched_distributed_potrs``; don't read the
        inner (N, N) directly — use ``to_jax_lower()`` if you need a
        conventional row-major lower-triangular L.
    mesh : Mesh
    n : int
    mb : int           # col block of A's descriptor (N/Px in the default)
    nb : int           # row block of A's descriptor (N/Py in the default)
    nbatch : int       # Nq
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    mb: int
    nb: int
    nbatch: int


def _validate_mesh(mesh: Mesh):
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    Px = int(mesh.shape["x"])
    Py = int(mesh.shape["y"])
    if Px * Py != jax.process_count():
        raise ValueError(
            f"mesh {Px}x{Py} != jax.process_count() = {jax.process_count()}")
    return Px, Py


def batched_distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
) -> CusolverMpBatchedLowerL:
    """Factor each ``A[q]`` = ``L[q] L[q]^H`` via ``cusolverMpPotrf``.

    Input sharding: ``P(None, 'x', 'y')`` — exactly the layout produced
    by the ISDF pipeline's ``C_q_flat`` (no reshard needed).
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"batched_distributed_cholesky: expected (Nq, N, N); "
            f"got {A.shape}")
    Px, Py = _validate_mesh(mesh)
    nq = int(A.shape[0])
    n  = int(A.shape[1])
    if n % Px != 0:
        raise ValueError(f"N={n} must be divisible by Px={Px}.")
    if n % Py != 0:
        raise ValueError(f"N={n} must be divisible by Py={Py}.")

    get_lib()
    # Row-major grid → tile (i, j) on rank i*Py + j = JAX rank ordering.
    ctx_handle = get_or_init_context(mesh, col_major=False)

    mb = n // Px          # A's row block (per-rank local row count)
    nb = n // Py          # A's col block (per-rank local col count)

    key = ("potrf", _mesh_key(mesh), A.dtype, nq, n, mb, nb, int(ctx_handle))
    jit_potrf = _JIT_CACHE.get(key)
    if jit_potrf is None:
        # Inner-dim transpose per slice: (Nq, N/Px, N/Py) row-major →
        # (Nq, N/Py, N/Px) row-major ≡ (N/Px, N/Py) col-major per slice.
        L_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        attrs = dict(nq=nq, n=n, mb=mb, nb=nb, ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=P(None, "x", "y"),
                 out_specs=P(None, "y", "x"),
                 check_rep=False)
        def _potrf(local_A):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            return jax.ffi.ffi_call(_POTRF_TARGET, L_local_T)(local_A_T, **attrs)

        jit_potrf = jax.jit(_potrf)
        _JIT_CACHE[key] = jit_potrf

    L_raw_T = jit_potrf(A)
    return CusolverMpBatchedLowerL(
        raw=L_raw_T, mesh=mesh, n=n, mb=mb, nb=nb, nbatch=nq)


def batched_distributed_potrs(
    L: CusolverMpBatchedLowerL,
    B: jax.Array,
    *,
    mesh: Mesh,
) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` using the factor from ``batched_distributed_cholesky``.

    Input sharding: ``B`` in ``P(None, 'x', 'y')``.  Output has the same
    sharding.  No reshard — the caller's Z_col layout flows straight
    through.
    """
    Px, Py = _validate_mesh(mesh)
    if mesh.axis_names != L.mesh.axis_names:
        raise ValueError("potrs mesh axis names don't match the handle's mesh.")
    if B.ndim != 3:
        raise ValueError(
            f"batched_distributed_potrs: expected 3D B; got {B.shape}")
    nq = L.nbatch
    n  = L.n
    if int(B.shape[0]) != nq:
        raise ValueError(f"B.shape[0]={B.shape[0]} != Nq={nq}")
    if int(B.shape[1]) != n:
        raise ValueError(f"B.shape[1]={B.shape[1]} != N={n}")
    mrhs = int(B.shape[2])
    if mrhs % Py != 0:
        raise ValueError(f"Mrhs={mrhs} must be divisible by Py={Py}.")
    if L.raw.dtype != B.dtype:
        raise ValueError(f"L.dtype {L.raw.dtype} != B.dtype {B.dtype}")

    get_lib()
    ctx_handle = get_or_init_context(mesh, col_major=False)

    # descA : mb=L.mb (rows block = N/Px), nb=L.nb (cols block = N/Py)
    # descB : mb_B = L.mb   (row block must equal A's row block — each rank's
    #                         local row count lld = N/Px)
    #         nb_B = Mrhs/Py (one col-tile per rank, matches JAX's
    #                         contiguous P(None, 'x', 'y') slab)
    mb_a = L.mb
    nb_a = L.nb
    mb_b = L.mb
    nb_b = mrhs // Py

    key = ("potrs", _mesh_key(mesh), B.dtype,
           nq, n, mrhs, mb_a, nb_a, mb_b, nb_b, int(ctx_handle))
    jit_potrs = _JIT_CACHE.get(key)
    if jit_potrs is None:
        X_local_T = jax.ShapeDtypeStruct(
            (nq, mrhs // Py, n // Px), B.dtype)
        attrs = dict(
            nq=nq, n=n, mrhs=mrhs,
            mb_a=mb_a, nb_a=nb_a, mb_b=mb_b, nb_b=nb_b,
            ctx_handle=int(ctx_handle),
        )

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "y", "x"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_rep=False)
        def _potrs(local_L, local_B):
            # B row-major (Nq, N/Px, Mrhs/Py) → inner transpose →
            # (Nq, Mrhs/Py, N/Px) row-major = (N/Px, Mrhs/Py) col-major per slice.
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            X_T = jax.ffi.ffi_call(_POTRS_TARGET, X_local_T)(
                local_L, local_B_T, **attrs)
            # Untranspose inside the shard_map → output is the natural
            # row-major (Nq, N/Px, Mrhs/Py) matching P(None, 'x', 'y').
            return jnp.transpose(X_T, (0, 2, 1))

        jit_potrs = jax.jit(_potrs)
        _JIT_CACHE[key] = jit_potrs

    return jit_potrs(L.raw, B)

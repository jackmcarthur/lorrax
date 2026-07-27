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
    "batched_distributed_solve_lu",
    "cholesky_handle_to_natural_L",
]

_POTRF_TARGET = "lorrax_cusolvermp_batched_potrf"
_POTRS_TARGET = "lorrax_cusolvermp_batched_potrs"
_SOLVE_LU_TARGET = "lorrax_cusolvermp_batched_solve_lu"

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
    if Px != Py:
        # cusolverMpPotrf requires square ScaLAPACK blocks (mb == nb);
        # our one-tile-per-rank descriptors give mb = N/Px, nb = N/Py, so
        # a non-square mesh always violates it — bufferSize fails with
        # CUSOLVER_STATUS_INVALID_VALUE (=3) at every size (verified 4x1
        # and 1x4, 2026-07-10).  The isdf 'auto' resolver already routes
        # 1-D meshes to the in-tree sharded_cholesky; this guard turns
        # the residual explicit-override path into a clear error.
        raise ValueError(
            f"batched_distributed_cholesky: mesh {Px}x{Py} is not square "
            f"— cusolverMpPotrf needs mb == nb (square block-cyclic).  "
            f"Use a square mesh or solver_kind='sharded_cholesky'.")

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
            # Alias A → L: factor in place, skip the D2D memcpy in the
            # handler.  Saves ~1 GB transient VRAM per rank at Si scale.
            return jax.ffi.ffi_call(
                _POTRF_TARGET, L_local_T,
                input_output_aliases={0: 0},
            )(local_A_T, **attrs)

        jit_potrf = jax.jit(_potrf, donate_argnums=(0,))
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

    ***cuSOLVERMp 0.6.0 bug on 2D grids***: this FFI returns wrong
    answers (quiet, no error) when ``NRHS ≤ N`` on a 2D process grid
    (Px>1 AND Py>1).  ``NRHS ≥ N + Py`` works correctly to machine
    precision.  Callers whose natural ``NRHS`` is small relative to N
    must pad with zero columns and slice the result.  The zeta-fit
    chunked path escapes the bug because its ``NRHS = n_rchunk`` is
    typically ≫ N.  (Not an issue on 0.7+, which the W distributed
    plan's ``solve_lu`` route requires anyway.)
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
            # Alias B → X: in-place solve, skip the D2D memcpy in the
            # handler.  Saves ~1.7 GB transient VRAM per rank at Si scale.
            X_T = jax.ffi.ffi_call(
                _POTRS_TARGET, X_local_T,
                input_output_aliases={1: 0},
            )(local_L, local_B_T, **attrs)
            # Untranspose inside the shard_map → output is the natural
            # row-major (Nq, N/Px, Mrhs/Py) matching P(None, 'x', 'y').
            return jnp.transpose(X_T, (0, 2, 1))

        jit_potrs = jax.jit(_potrs, donate_argnums=(1,))
        _JIT_CACHE[key] = jit_potrs

    return jit_potrs(L.raw, B)


def cholesky_handle_to_natural_L(L_handle: CusolverMpBatchedLowerL) -> jax.Array:
    """Materialize the opaque Cholesky handle into a regular ``jax.Array``.

    The handle wraps ``L_raw_T`` sharded ``P(None, 'y', 'x')`` whose bytes
    are cuSOLVERMp's col-major ``L`` tiles per slice.  Reinterpreted as
    row-major these bytes are exactly ``L^T``.  Transposing the inner two
    dims (a materialized XLA transpose) gives back ``L`` in row-major
    natural layout with sharding ``P(None, 'x', 'y')``.  We then apply a
    per-rank ``tril`` mask to zero the garbage upper-triangle that
    cuSOLVERMp leaves untouched (it's whatever was in A's upper when A
    was donated).

    This is needed by consumers that want a conventional ``L`` jax.Array
    (e.g. the symmetric W-solve path: ``W = X H⁻¹ X†`` where we form
    ``X† χ X`` via matmul).

    Returns
    -------
    L : jax.Array
        Shape ``(Nq, N, N)``, sharded ``P(None, 'x', 'y')``.
        ``L[q]`` is the lower-triangular Cholesky factor of the original
        A[q], row-major, with the upper triangle zeroed.
    """
    mesh = L_handle.mesh
    Px = int(mesh.shape["x"])
    Py = int(mesh.shape["y"])
    nq = L_handle.nbatch
    n  = L_handle.n
    mb = L_handle.mb   # N/Px
    nb = L_handle.nb   # N/Py

    @partial(shard_map, mesh=mesh,
             in_specs=P(None, "y", "x"),
             out_specs=P(None, "x", "y"),
             check_rep=False)
    def _to_natural(L_raw_T):
        # L_raw_T local bytes = L^T (row-major view of col-major L bytes).
        # Swap inner dims to recover L in row-major (materialized copy).
        L = jnp.transpose(L_raw_T, (0, 2, 1))
        # Per-rank triangular mask: each rank owns the tile with global
        # row range [x_idx*mb, (x_idx+1)*mb) and col range
        # [y_idx*nb, (y_idx+1)*nb).  Entry is kept iff global_row >= global_col.
        x_idx = jax.lax.axis_index("x")
        y_idx = jax.lax.axis_index("y")
        rows = jnp.arange(mb) + x_idx * mb
        cols = jnp.arange(nb) + y_idx * nb
        tril = (rows[:, None] >= cols[None, :]).astype(L.dtype)
        return L * tril[None, :, :]

    return _to_natural(L_handle.raw)


def batched_distributed_solve_lu(
    A: jax.Array,
    B: jax.Array,
    *,
    mesh: Mesh,
) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` via per-q cuSOLVERMp ``getrf`` + ``getrs``.

    Validated on cuSOLVERMp 0.7.2 across 2×2 / 1×4 / 4×1 meshes for
    indefinite Hermitian and general A; residuals 10⁻¹³–10⁻¹⁵ for C128
    at N up to 512.  cuSOLVERMp 0.6.0 returned garbage on Px>1 AND Py>1
    (info=0 but wrong X); 0.7+ fixes it.

    Used by the ζ-fit transverse channels (indefinite CCT^μ) and — via
    the ``ffi.linalg`` plan facade on CUDA meshes — by the W Dyson
    ``w_dyson_solver = distributed`` plan for the general
    (non-Hermitian) ``I - V χ`` system.  Pivot vectors are allocated
    per call inside the FFI and never surfaced to Python.

    Input sharding: both ``A`` and ``B`` in ``P(None, 'x', 'y')``.
    Output has the same shape and sharding as ``B``.  ``A`` is donated —
    XLA may reuse its buffer for the LU factors (which we then discard).

    Restrictions:
      * A: (Nq, N, N), B: (Nq, N, NRHS) — N % Px == 0, N % Py == 0,
        NRHS % Py == 0.  Square N on both sides because cuSOLVERMp's
        descB must have the same row-block as descA (mb_b = N/Px).
      * Dtypes F64 / C128.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"batched_distributed_solve_lu: expected A of shape (Nq, N, N); "
            f"got {A.shape}")
    if B.ndim != 3 or int(B.shape[0]) != int(A.shape[0]) or int(B.shape[1]) != int(A.shape[1]):
        raise ValueError(
            f"batched_distributed_solve_lu: B must be (Nq, N, NRHS) with "
            f"Nq, N matching A; got A={A.shape}, B={B.shape}")
    if A.dtype != B.dtype:
        raise ValueError(f"A.dtype {A.dtype} != B.dtype {B.dtype}")
    Px, Py = _validate_mesh(mesh)
    nq   = int(A.shape[0])
    n    = int(A.shape[1])
    nrhs = int(B.shape[2])
    if n % Px != 0:
        raise ValueError(f"N={n} must be divisible by Px={Px}.")
    if n % Py != 0:
        raise ValueError(f"N={n} must be divisible by Py={Py}.")
    if nrhs % Py != 0:
        raise ValueError(f"NRHS={nrhs} must be divisible by Py={Py}.")

    get_lib()
    ctx_handle = get_or_init_context(mesh, col_major=False)

    mb_a = n // Px
    nb_a = n // Py
    mb_b = n // Px
    nb_b = nrhs // Py

    key = ("solve_lu", _mesh_key(mesh), A.dtype,
           nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, int(ctx_handle))
    jit_solve = _JIT_CACHE.get(key)
    if jit_solve is None:
        A_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // Py, n // Px), B.dtype)
        attrs = dict(
            nq=nq, n=n, nrhs=nrhs,
            mb_a=mb_a, nb_a=nb_a, mb_b=mb_b, nb_b=nb_b,
            ctx_handle=int(ctx_handle),
        )

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_rep=False)
        def _solve(local_A, local_B):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            # Single output X_T.  A is donated in-place (getrf
            # scribbles LU factors into A's buffer), B → X aliasing
            # makes getrs in-place on B.
            X_T = jax.ffi.ffi_call(
                _SOLVE_LU_TARGET, X_local_T,
                input_output_aliases={1: 0},
            )(local_A_T, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_solve = jax.jit(_solve, donate_argnums=(0, 1))
        _JIT_CACHE[key] = jit_solve

    return jit_solve(A, B)

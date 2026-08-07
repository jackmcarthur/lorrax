"""cuSOLVERMp FFI backend — CUDA distributed dense linear algebra.

Ported from LORRAX ``src/ffi/cusolvermp/{context,eigh,batched}.py`` at
96a6399 and merged into ONE backend module, which is the shape
:mod:`._slate` and :mod:`._scalapack` already had (wave-2,
``ffi_layout.md`` §3/§6).  Nothing is renamed; the FFI target strings and
the C++ handler symbols are FROZEN (invariant 1).

**NOT a deletion candidate.**  ``ffi_layout.md`` §5 once listed cuSOLVERMp
for removal; the phase charter overturned that ruling and the premises are
measurably false — zero ``src/`` vendor imports, SLATE has no LU handler at
all, and the preconditions the old ruling named were never met.

Section docstrings from the merged modules follow inline.
"""
from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from distrib_la import loader
from distrib_la._collectives import broadcast_bytes
from distrib_la._shard_map import shard_map

__all__ = [
    "CusolverMpBatchedLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_potrs",
    "batched_distributed_solve_lu",
    "cholesky_handle_to_natural_L",
    "distributed_eigh",
    "get_or_init_context",
]

# =========================================================================
#  Context lifecycle  (merged from cusolvermp/context.py)
# =========================================================================
"""Per-process singleton context for cuSOLVERMp.

A cuSOLVERMp session bundles (NCCL comm, CUDA stream, handle, grid,
workspace).  We lazily build one per (mesh shape, layout) key and cache
it for the life of the process.  Torn down by ``atexit``.

Cross-process bootstrap
-----------------------
cuSOLVERMp needs its own NCCL communicator.  We build it via:

    rank 0 :  ncclGetUniqueId     -> 128 raw bytes
    all    :  broadcast_bytes     -> the jax distributed KV store
    each   :  ncclCommInitRank

which re-uses the KV store ``jax.distributed.initialize()`` already put in
place.  ``broadcast_bytes`` and not ``multihost_utils.broadcast_one_to_all``
because the latter promotes ``uint8`` to ``uint64`` under x64 and scrambles
the payload (see :mod:`distrib_la._collectives`).
"""

# (p, q, layout) is the unique identity of a grid.  Reusing the same
# context across calls with matching (p, q) is fine; cusolverMp grids are
# reusable.
MeshKey = Tuple[int, int, bool]

_CTX_LOCK = threading.Lock()
_CTX_CACHE: Dict[MeshKey, int] = {}


def _ctx_key(mesh: Mesh, col_major: bool) -> MeshKey:
    """Extract (p, q, layout) from a 2-axis Mesh labelled ('x', 'y')."""
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"Mesh must have axes ('x', 'y'); got {mesh.axis_names}")
    return (int(mesh.shape["x"]), int(mesh.shape["y"]), bool(col_major))


def _make_ctx(mesh: Mesh, col_major: bool) -> int:
    loader.get_lib("CUDA")
    rank = int(jax.process_index())
    world = int(jax.process_count())
    p, q, _ = _ctx_key(mesh, col_major)
    if p * q != world:
        raise ValueError(
            f"mesh.shape['x']*mesh.shape['y'] = {p*q} does not match "
            f"jax.process_count() = {world}.  cuSOLVERMp needs one rank "
            f"per grid cell in this scaffolding.")

    uid_nbytes = loader.nccl_unique_id_bytes()
    buf = np.zeros(uid_nbytes, dtype=np.uint8)
    if rank == 0:
        loader.fill_nccl_unique_id(int(buf.ctypes.data))
    # The key namespace must be unique per (p, q, layout) so multiple
    # cuSOLVERMp contexts can coexist in one process.
    uid = broadcast_bytes(buf, key=(
        f"lorrax_ffi/cusolvermp/nccl_unique_id/v0"
        f"/{p}x{q}/{'col' if col_major else 'row'}"))

    return int(loader.create_cusolvermp_context(
        rank=rank, world_size=world,
        nccl_unique_id_addr=int(uid.ctypes.data),
        nccl_unique_id_nbytes=uid_nbytes,
        p=p, q=q, grid_layout_col_major=col_major))


def get_or_init_context(mesh: Mesh, *, col_major: bool = True) -> int:
    """The opaque int handle for the cuSOLVERMp context of this mesh.

    Thread-safe.  On first call for a given (p, q, layout), bootstraps a
    new NCCL communicator + cusolverMp handle + grid via a collective
    across all JAX processes.
    """
    key = _ctx_key(mesh, col_major)
    with _CTX_LOCK:
        h = _CTX_CACHE.get(key)
        if h is not None:
            return h
        h = _make_ctx(mesh, col_major)
        _CTX_CACHE[key] = h
        return h


def _atexit_teardown() -> None:
    # Best-effort.  On abnormal exits (signals, segfaults) this may not
    # run, but the OS reclaims GPU memory when the process dies and the
    # destroy calls are idempotent/null-guarded.
    try:
        loader.get_lib("CUDA")
    except Exception:                                          # noqa: BLE001
        return
    for h in list(_CTX_CACHE.values()):
        try:
            loader.destroy_cusolvermp_context(int(h))
        except Exception:                                      # noqa: BLE001
            pass
    _CTX_CACHE.clear()


atexit.register(_atexit_teardown)


# =========================================================================
#  eigh  (merged from cusolvermp/eigh.py)
# =========================================================================
"""``distributed_eigh`` — JAX FFI wrapper around ``cusolverMpSyevd``.

Layout note
-----------
cuSOLVERMp interprets the input buffer as a column-major ScaLAPACK tile.
JAX stores arrays row-major.  For a Hermitian matrix A = A^H, passing the
row-major local shard to cuSOLVERMp is equivalent to handing it A^T; since
the eigenvalues of A^T equal those of A (and of conj(A) for Hermitian),
the returned eigenvalues are correct.

The eigenVECTORS inherit the same untransposed layout: the returned ``Q``
raw buffer satisfies ``Q.conj().T[:, i] = i-th eigenvector`` (for F64,
``Q.T[:, i]``) — verified on 1x1 and 2x2 meshes, pinned by
``tests/test_ffi_linalg_contract.py``.  Callers that need conventional
column eigenvectors must conj-transpose; :func:`distrib_la.plan._eigh_columns`
is the ONE place that knows it.  (The SLATE eigh returns TRUE column
eigenvectors — the two wrappers share shapes but NOT the Q convention.)

Mesh restriction: ``cusolverMpSyevd`` requires square ScaLAPACK blocks
(mb == nb).  On a non-square mesh the natural one-tile-per-rank blocks
are rectangular and syevd DEADLOCKS inside a collective instead of
returning an error status (observed 4x1/1x4, 2026-07-10) — so this
wrapper rejects p != q up front.
"""

_EIGH_TARGET = "lorrax_cusolvermp_eigh"
_POTRF_TARGET = "lorrax_cusolvermp_batched_potrf"
_POTRS_TARGET = "lorrax_cusolvermp_batched_potrs"
_SOLVE_LU_TARGET = "lorrax_cusolvermp_batched_solve_lu"

# jit(shard_map(...)) cache per signature: eager shard_map re-traces per
# call; wrapping in jax.jit once amortises it.
_JIT_CACHE: dict = {}


def _mesh_key(mesh: Mesh):
    return (tuple(mesh.axis_names), tuple(int(s) for s in mesh.shape.values()))


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


def distributed_eigh(
    A: jax.Array,
    *,
    mesh: Mesh,
    compute_evecs: bool = True,
    block_size: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition via cuSOLVERMp.

    Parameters
    ----------
    A
        Square ``(n, n)`` array, dtype ``float64`` (symmetric) or
        ``complex128`` (Hermitian), sharded ``P('x','y')`` on ``mesh``.
    mesh
        2-D ``Mesh`` labelled ``('x','y')``.  ``mesh.shape['x'] *
        mesh.shape['y']`` must equal ``jax.process_count()``, and ``n``
        must be divisible by both axis sizes.
    compute_evecs
        If False, only eigenvalues are meaningful (Q is still returned but
        its contents should be ignored).
    block_size
        Override the 2-D block-cyclic tile size.  Default ``n/p`` (one
        tile per rank — matches JAX's block sharding, eigenvectors come
        out in-place).  Smaller blocks speed up the solve at larger n
        (block=256 gives 2.4× at n=16k) but return eigenvectors in a
        block-cyclic permutation of the input basis.

    Returns
    -------
    W
        ``(n,)`` ``float64`` eigenvalues, ascending, replicated.
    Q
        ``(n, n)`` raw eigenvector buffer, same dtype as A, sharded
        ``P('x','y')``.  ``Q.conj().T`` holds the eigenvectors as
        columns (see the section Layout note).
    """
    # --- validate ---------------------------------------------------------
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_eigh: expected a square matrix, "
                         f"got {A.shape}")
    p, q = _validate_mesh(mesh)
    if p != q:
        raise ValueError(
            f"distributed_eigh: mesh {p}x{q} is not square — "
            f"cusolverMpSyevd requires mb == nb and DEADLOCKS (no error "
            f"status) on the rectangular one-tile-per-rank blocks a "
            f"non-square mesh produces.  Use a square mesh.")
    n = int(A.shape[0])
    if n % p != 0 or n % q != 0:
        raise ValueError(
            f"n={n} must be divisible by mesh shape ({p},{q}).")

    # --- one-time setup per process (lazy, cached) ------------------------
    loader.get_lib("CUDA")                  # load the .so, register targets
    ctx_handle = get_or_init_context(mesh)  # NCCL + cal_comm + cusolverMp

    mb = n // p if block_size is None else block_size
    nb = n // q if block_size is None else block_size

    # Local output shapes + partition specs for shard_map.
    W_local = jax.ShapeDtypeStruct((n,), jnp.float64)           # replicated
    Q_local = jax.ShapeDtypeStruct((n // p, n // q), A.dtype)
    # Attributes forwarded to the C++ handler.
    attrs = dict(n=n, mb=mb, nb=nb,
                 ctx_handle=int(ctx_handle),
                 compute_evecs=bool(compute_evecs))

    @partial(shard_map,
             mesh=mesh,
             in_specs=P("x", "y"),
             out_specs=(P(), P("x", "y")),
             check_vma=False)
    def _call(local_A):
        return jax.ffi.ffi_call(
            _EIGH_TARGET, (W_local, Q_local))(local_A, **attrs)

    W, Q = _call(A)
    return W, Q


# =========================================================================
#  batched potrf / potrs / solve_lu  (merged from cusolvermp/batched.py)
# =========================================================================
"""Per-q cuSOLVERMp potrf + potrs on the full (Px, Py) process mesh.

Use case: a stack of ``Nq`` independent Hermitian PD matrices ``A_q`` of
shape ``(N, N)``, each 2D-sharded across the full device mesh; ``A_q``
is too big to fit on one GPU even alone, so the distribution buys
memory scalability.  Right-hand sides ``B_q`` have shape ``(N, Mrhs)``
(``Mrhs`` can be much larger than ``N``) and share the same column
sharding.

Sharding contract (``distributed_eigh``'s P('x','y') layout, extended to
the batch dim)::

    A : (Nq, N, N)       P(None, 'x', 'y')   # per-rank (Nq, N/Px, N/Py)
    B : (Nq, N, Mrhs)    P(None, 'x', 'y')   # per-rank (Nq, N/Px, Mrhs/Py)
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

Restrictions: mesh is 2-D ('x','y'); dtypes F64 / C128; the world-wide
context is created with ``col_major=False`` so tile-(i, j) → rank
``i*Py + j`` matches JAX's row-major mesh reshape.
"""


@dataclass(frozen=True)
class CusolverMpBatchedLowerL:
    """Opaque handle wrapping the batched Cholesky factor.

    Attributes
    ----------
    raw : jax.Array
        Shape ``(Nq, N, N)``, sharded ``P(None, 'y', 'x')``.  Bytes are
        cuSOLVERMp's col-major L tiles per slice (from the inner-dim
        transpose that translates JAX row-major to cuSOLVERMp col-major).
        Pass this back to :func:`batched_distributed_potrs`; don't read the
        inner (N, N) directly — use :func:`cholesky_handle_to_natural_L` if
        you need a conventional row-major lower-triangular L.
    mesh : Mesh
    n : int
    mb : int           # row block of A's descriptor (N/Px in the default)
    nb : int           # col block of A's descriptor (N/Py in the default)
    nbatch : int       # Nq
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    mb: int
    nb: int
    nbatch: int


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
    n = int(A.shape[1])
    if n % Px != 0:
        raise ValueError(f"N={n} must be divisible by Px={Px}.")
    if n % Py != 0:
        raise ValueError(f"N={n} must be divisible by Py={Py}.")
    if Px != Py:
        # cusolverMpPotrf requires square ScaLAPACK blocks (mb == nb);
        # our one-tile-per-rank descriptors give mb = N/Px, nb = N/Py, so
        # a non-square mesh always violates it — bufferSize fails with
        # CUSOLVER_STATUS_INVALID_VALUE (=3) at every size (verified 4x1
        # and 1x4, 2026-07-10).  The facade's guard 5 refuses 1-D meshes
        # before this; the guard here turns the residual direct-call path
        # into a clear error.
        raise ValueError(
            f"batched_distributed_cholesky: mesh {Px}x{Py} is not square "
            f"— cusolverMpPotrf needs mb == nb (square block-cyclic).  "
            f"Use a square mesh or a native cholesky backend.")

    loader.get_lib("CUDA")
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
                 check_vma=False)
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
    """Solve ``A[q] X[q] = B[q]`` using :func:`batched_distributed_cholesky`'s
    factor.

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
    n = L.n
    if int(B.shape[0]) != nq:
        raise ValueError(f"B.shape[0]={B.shape[0]} != Nq={nq}")
    if int(B.shape[1]) != n:
        raise ValueError(f"B.shape[1]={B.shape[1]} != N={n}")
    mrhs = int(B.shape[2])
    if mrhs % Py != 0:
        raise ValueError(f"Mrhs={mrhs} must be divisible by Py={Py}.")
    if L.raw.dtype != B.dtype:
        raise ValueError(f"L.dtype {L.raw.dtype} != B.dtype {B.dtype}")

    loader.get_lib("CUDA")
    ctx_handle = get_or_init_context(mesh, col_major=False)

    # descA : mb=L.mb (rows block = N/Px), nb=L.nb (cols block = N/Py)
    # descB : mb_B = L.mb   (row block must equal A's row block — each rank's
    #                         local row count lld = N/Px)
    #         nb_B = Mrhs/Py (one col-tile per rank, matches JAX's
    #                         contiguous P(None, 'x', 'y') slab)
    mb_a, nb_a = L.mb, L.nb
    mb_b, nb_b = L.mb, mrhs // Py

    key = ("potrs", _mesh_key(mesh), B.dtype,
           nq, n, mrhs, mb_a, nb_a, mb_b, nb_b, int(ctx_handle))
    jit_potrs = _JIT_CACHE.get(key)
    if jit_potrs is None:
        X_local_T = jax.ShapeDtypeStruct((nq, mrhs // Py, n // Px), B.dtype)
        attrs = dict(nq=nq, n=n, mrhs=mrhs,
                     mb_a=mb_a, nb_a=nb_a, mb_b=mb_b, nb_b=nb_b,
                     ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "y", "x"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_vma=False)
        def _potrs(local_L, local_B):
            # B row-major (Nq, N/Px, Mrhs/Py) → inner transpose →
            # (Nq, Mrhs/Py, N/Px) row-major = (N/Px, Mrhs/Py) col-major
            # per slice.
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

    Needed by consumers that want a conventional ``L`` jax.Array (e.g.
    the symmetric W-solve path: ``W = X H⁻¹ X†`` formed via matmul).

    Returns
    -------
    L : jax.Array
        Shape ``(Nq, N, N)``, sharded ``P(None, 'x', 'y')``; ``L[q]`` is
        the lower-triangular Cholesky factor of the original ``A[q]``,
        row-major, with the upper triangle zeroed.
    """
    mesh = L_handle.mesh
    mb = L_handle.mb   # N/Px
    nb = L_handle.nb   # N/Py

    @partial(shard_map, mesh=mesh,
             in_specs=P(None, "y", "x"),
             out_specs=P(None, "x", "y"),
             check_vma=False)
    def _to_natural(L_raw_T):
        # L_raw_T local bytes = L^T (row-major view of col-major L bytes).
        # Swap inner dims to recover L in row-major (materialized copy).
        L = jnp.transpose(L_raw_T, (0, 2, 1))
        # Per-rank triangular mask: each rank owns the tile with global
        # row range [x_idx*mb, (x_idx+1)*mb) and col range
        # [y_idx*nb, (y_idx+1)*nb).  Entry kept iff global_row >= global_col.
        rows = jnp.arange(mb) + jax.lax.axis_index("x") * mb
        cols = jnp.arange(nb) + jax.lax.axis_index("y") * nb
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
    (info=0 but wrong X); 0.7+ fixes it — which is why the LD_LIBRARY_PATH
    order that selects between the two stages is load-bearing.

    Used by the ζ-fit transverse channels (indefinite CCT^μ) and — via the
    plan facade on CUDA meshes — by the W Dyson ``w_dyson_solver =
    distributed`` plan for the general (non-Hermitian) ``I - V χ`` system.
    Pivot vectors are allocated per call inside the FFI and never surfaced
    to Python; there is no split factor/solve pair for this route, so
    :func:`distrib_la.factor` refuses it and names this function.

    Input sharding: both ``A`` and ``B`` in ``P(None, 'x', 'y')``.  Output
    has the same shape and sharding as ``B``.  ``A`` is donated — XLA may
    reuse its buffer for the LU factors (which we then discard).

    Restrictions: A (Nq, N, N), B (Nq, N, NRHS) with N % Px == 0,
    N % Py == 0, NRHS % Py == 0 — square N on both sides because
    cuSOLVERMp's descB must have the same row-block as descA (mb_b = N/Px).
    Dtypes F64 / C128.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"batched_distributed_solve_lu: expected A of shape (Nq, N, N); "
            f"got {A.shape}")
    if (B.ndim != 3 or int(B.shape[0]) != int(A.shape[0])
            or int(B.shape[1]) != int(A.shape[1])):
        raise ValueError(
            f"batched_distributed_solve_lu: B must be (Nq, N, NRHS) with "
            f"Nq, N matching A; got A={A.shape}, B={B.shape}")
    if A.dtype != B.dtype:
        raise ValueError(f"A.dtype {A.dtype} != B.dtype {B.dtype}")
    Px, Py = _validate_mesh(mesh)
    nq = int(A.shape[0])
    n = int(A.shape[1])
    nrhs = int(B.shape[2])
    if n % Px != 0:
        raise ValueError(f"N={n} must be divisible by Px={Px}.")
    if n % Py != 0:
        raise ValueError(f"N={n} must be divisible by Py={Py}.")
    if nrhs % Py != 0:
        raise ValueError(f"NRHS={nrhs} must be divisible by Py={Py}.")

    loader.get_lib("CUDA")
    ctx_handle = get_or_init_context(mesh, col_major=False)

    mb_a, nb_a = n // Px, n // Py
    mb_b, nb_b = n // Px, nrhs // Py

    key = ("solve_lu", _mesh_key(mesh), A.dtype,
           nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, int(ctx_handle))
    jit_solve = _JIT_CACHE.get(key)
    if jit_solve is None:
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // Py, n // Px), B.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs,
                     mb_a=mb_a, nb_a=nb_a, mb_b=mb_b, nb_b=nb_b,
                     ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_vma=False)
        def _solve(local_A, local_B):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            # Single output X_T.  A is donated in-place (getrf scribbles
            # LU factors into A's buffer); B → X aliasing makes getrs
            # in-place on B.
            X_T = jax.ffi.ffi_call(
                _SOLVE_LU_TARGET, X_local_T,
                input_output_aliases={1: 0},
            )(local_A_T, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_solve = jax.jit(_solve, donate_argnums=(0, 1))
        _JIT_CACHE[key] = jit_solve

    return jit_solve(A, B)

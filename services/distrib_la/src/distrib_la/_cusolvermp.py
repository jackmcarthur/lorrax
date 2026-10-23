"""cuSOLVERMp FFI backend — CUDA distributed dense linear algebra.

Ported from LORRAX ``src/ffi/cusolvermp/{context,eigh,batched}.py`` at
96a6399 and merged into ONE backend module, which is the shape
:mod:`._scalapack` already had (wave-2,
``ffi_layout.md`` §3/§6).  Nothing is renamed; the FFI target strings and
the C++ handler symbols are FROZEN (invariant 1).

**NOT a deletion candidate.**  ``ffi_layout.md`` §5 once listed cuSOLVERMp
for removal; the phase charter overturned that ruling and the premises are
measurably false — zero ``src/`` vendor imports, and the preconditions the old ruling named were never met.

Section docstrings from the merged modules follow inline.
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import threading
from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la import loader
from distrib_la._collectives import broadcast_bytes
from distrib_la._cyclic import cyclic_to_block, largest_divisor, pad_with_sentinels
from distrib_la._shape import ipiv_local_len
from jax import shard_map
from distrib_la.resolve import mesh_key

#: ``distrib_la`` is independently installable: it logs on the ``lorrax.`` namespace directly.
_LOG = logging.getLogger("lorrax.distrib_la")

__all__ = [
    "CusolverMpBatchedLowerL",
    "CusolverMpBatchedLU",
    "batched_distributed_cholesky",
    "batched_distributed_getrf",
    "batched_distributed_getrs",
    "batched_distributed_potrs",
    "batched_distributed_solve_lu",
    "cholesky_handle_to_natural_L",
    "distributed_eigh",
    "context_key",
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
#: (p, q, layout) -> the context's FFI ``ctx_key`` (loader.context_key).
_CTX_KEYS: Dict[MeshKey, int] = {}


def _ctx_config(key: MeshKey) -> str:
    """The canonical configuration string of a cuSOLVERMp/cuBLASMp context."""
    p, q, col_major = key
    return f"lorrax-ctx/v1|cusolvermp|{p}x{q}|{'col' if col_major else 'row'}"


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
        _CTX_KEYS[key] = loader.bind_context("CUDA", _ctx_config(key), h)
        _CTX_CACHE[key] = h
        return h


def context_key(mesh: Mesh, *, col_major: bool = True) -> int:
    """The FFI ``ctx_key`` attribute of this mesh's cuSOLVERMp context.

    A pure function of (p, q, layout), identical on every rank and in every
    run, so a module that carries it stays cacheable; the native handler
    resolves it to the live context this call ensured and bound.
    """
    get_or_init_context(mesh, col_major=col_major)
    return _CTX_KEYS[_ctx_key(mesh, col_major)]


def _atexit_teardown() -> None:
    # Best-effort.  On abnormal exits (signals, segfaults) this may not
    # run, but the OS reclaims GPU memory when the process dies and the
    # destroy calls are idempotent/null-guarded.  No context, nothing to destroy: loading the
    # CUDA library at exit on a host without one (a login node) crashes the process.
    if not _CTX_CACHE:
        return
    try:
        loader.get_lib("CUDA")
    except Exception:                                          # noqa: BLE001
        return
    for k, h in list(_CTX_CACHE.items()):
        try:
            loader.unbind_context("CUDA", _CTX_KEYS.pop(k, 0))
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
``Q.T[:, i]``) — verified on 1x1 and 2x2 meshes, pinned by this service's
own ``tests/test_distrib_la_contract.py`` (the pin migrated here with the
backend; it was ``tests/test_ffi_linalg_contract.py`` in LORRAX).  Callers
that need conventional
column eigenvectors must conj-transpose; :func:`distrib_la.plan._eigh_columns`
is the ONE place that knows it.  (The ELPA and ScaLAPACK eighs return TRUE
column eigenvectors — the wrappers share shapes but NOT the Q convention.)

Mesh restriction: ``cusolverMpSyevd`` requires square ScaLAPACK blocks
(mb == nb).  On a non-square mesh the natural one-tile-per-rank blocks
are rectangular and syevd DEADLOCKS inside a collective instead of
returning an error status (observed 4x1/1x4, 2026-07-10) — so this
wrapper rejects p != q up front.
"""

_EIGH_TARGET = "lorrax_cusolvermp_eigh"
_POTRF_TARGET = "lorrax_cusolvermp_batched_potrf"
_POTRS_TARGET = "lorrax_cusolvermp_batched_potrs"
_SOLVE_LU_TARGET = "lorrax_cusolvermp_batched_solve_lu_factors"
_GETRF_TARGET = "lorrax_cusolvermp_batched_getrf"
_GETRS_TARGET = "lorrax_cusolvermp_batched_getrs"

# jit(shard_map(...)) cache per signature: eager shard_map re-traces per
# call; wrapping in jax.jit once amortises it.
_JIT_CACHE: dict = {}


#: The package's ONE mesh cache key (:func:`distrib_la.mesh_key`).
#:
#: This module used to carry a WEAKER one -- axis names and extents only,
#: no platform and no device ids -- so two 1x1 meshes over different
#: devices in one process (the CPU/GPU pair a mixed test session builds)
#: hashed equal and the second took the first's compiled kernel.  A stale
#: HIT, which is the one failure mode a cache key must not have.  Using the
#: package key can only cost an extra compile, never a wrong one.
_mesh_key = mesh_key


# Block-cyclic eigh by relabeling (the helpers live in distrib_la._cyclic,
# shared with the ELPA eigh).  A P('x','y') operand is one (n/p, n/p)
# tile per rank.  Handing cuSOLVERMp the SAME local buffers with a smaller
# square block mb (mb | n/p) makes it read them as Pi A Pi^T, Pi sending rank
# r's local block c to global block c*p + r.  Rows and columns get the same
# Pi (square grid), so the eigenvalues are A's and the eigenvector components
# read back in A's order with no data movement.  The eigenvalue index is laid
# out the same way, so _cyclic_rows_to_block puts Q's rows back in ascending
# order (one all_to_all over 'x', n^2/P per rank).  The tridiagonalization
# then keeps every rank busy as the trailing matrix shrinks.  Measured (c128,
# median s, one tile per rank -> mb 250, the un-permute included):
#   P16  n 16000: 17.50 -> 10.59   8000: 4.46 -> 3.56   4000: 1.43 -> 1.38
#   P64  n 16000: 20.05 -> 15.65   8000: 7.13 -> 6.47   4000: 2.91 -> 2.80
# mb 125 matches 250 at P16; 500 is slower at P64; n/p <= 256 is unchanged.
# A tile whose largest divisor up to 256 is small (n/p prime or a small
# multiple of one: 3954 = 2*3*659 gives block 6) is padded instead, to the
# smallest tile with a divisor in [128, 256] (solve_layout): blocks 1-8 ran
# 5-34x slower than one tile per rank (P4 bench, KNOWN_LORRAX_ISSUES).
# LU and Cholesky keep one tile per rank: there block-cyclic getrs/potrs are
# 2-3x slower and getrf gains nothing (same bench).
_BLOCK_MAX = 256
_BLOCK_MIN = 128

#: LU above this local edge runs block-cyclic BY RELABELING too (square meshes):
#: the same local buffers read with a square block <= _BLOCK_MAX, so cuSOLVERMp
#: factors Pi A Pi^T and solves for Pi B, and X comes back in A's row order with
#: no data movement (Pi' = Pi).  At one tile per rank the getrf/getrs workspaces
#: grow with the tile: A100-80GB P4 f64 nrhs=n peaked at 53.0 GB/rank at n=30720
#: and 83.5 GB at 40960, and complex128 nrhs=n ran out of device memory from
#: n=30720 (dlascale pmA/pmB).  At or below the bound one tile per rank stays (the
#: measured faster layout there, comment above), so no existing call changes.
_LU_RELABEL_LOCAL_MIN = 8192


def rhs_block(n: int, nrhs: int, Px: int, Py: int, mb: int) -> int:
    """The RHS column block ``nb_b``: one block per process column at one tile per
    rank; a divisor of the local RHS width <= :data:`_BLOCK_MAX` when the LU is
    relabeled, so the handler can solve the RHS in chunks whose getrs workspace
    fits (cpp/cusolvermp/batched_solve_lu_ffi.cc, getrs_chunking).  Columns are
    independent and read and written through the same map: no movement."""
    local = nrhs // Py
    return local if mb == n // Px else largest_divisor(local, _BLOCK_MAX)


def _announce_lu(kind, mesh, nq, n, nrhs, mb_a, nb_a, mb_b=None, nb_b=None):
    """One rank-0 line per distributed LU program built: its descriptors, so a run's log shows
    the layout it solved at (run30k 2026-10-19: a stalled Dyson LU printed none)."""
    if int(jax.process_index()) != 0:
        return
    Px, Py = (int(v) for v in _validate_mesh(mesh))
    rhs = ("" if nb_b is None else
           f"; B {n}x{nrhs} mb_b {mb_b} nb_b {nb_b}, {(nrhs // Py) // max(1, nb_b)} RHS block(s) per rank "
           "(getrs chunks them within the room posted for the solve, else 2 GiB)")
    relabel = "relabeled block-cyclic" if (mb_a, nb_a) != (n // Px, n // Py) else "one tile per rank"
    _LOG.debug(f"  [distrib_la.cusolvermp] {kind} {nq} x {n}^2 on the {Px}x{Py} mesh: A mb {mb_a} nb {nb_a} "
               f"({relabel}){rhs}.")


def post_getrs_room(n: int, nrhs: int, nbytes) -> bool:
    """Post ``nbytes`` per rank (``None`` withdraws it) as the room the getrs chunks of a fused
    ``n x nrhs`` solve may take beside everything else its program holds: the handler solves in
    the widest chunks whose workspace fits the ranks' smallest figure, 2 GiB without one
    (cpp/cusolvermp/batched_solve_lu_ffi.cc, getrs_chunk).  Run-time state read by the next solve
    of that shape, never an attribute, so the compiled program does not carry the room.  False
    where the loaded library predates the entry (it keeps 2 GiB)."""
    fn = getattr(loader.get_lib("CUDA"), "lrx_lu_getrs_room", None)
    if fn is None:
        return False
    fn.argtypes, fn.restype = (ctypes.c_int64,) * 3, ctypes.c_int
    return fn(int(n), int(nrhs), -1 if nbytes is None else max(0, int(nbytes))) == 0


def lu_blocks(n: int, Px: int, Py: int) -> tuple[int, int]:
    """``(mb, nb)`` of the LU descriptors: one tile per rank, or a square relabeled
    block on a square mesh from 4x4 up, or where the local edge exceeds
    :data:`_LU_RELABEL_LOCAL_MIN` (the 2x2 bound, measured on one node). At one tile
    per rank on 4x4 (n 30000, run30k, 2026-10-19) getrf/getrs ran in 4 serialized
    7500-wide panel steps across nodes and the RHS could not be chunked: SMs idle."""
    one = (n // Px, n // Py)
    if Px != Py or Px == 1 or (Px < 4 and n // Px <= _LU_RELABEL_LOCAL_MIN):
        return one
    block = _largest_block(n // Px)
    return (block, block) if block >= _BLOCK_MIN else one


def _largest_block(local: int) -> int:
    """The largest divisor of a tile edge ``local`` that is at most :data:`_BLOCK_MAX`."""
    return largest_divisor(local, _BLOCK_MAX)


def _block_size(n: int, p: int) -> int:
    """The eigh block edge for an ``n x n`` operand on a ``p x p`` grid, unpadded.

    The largest divisor of ``n/p`` that is at most :data:`_BLOCK_MAX`, so
    every rank keeps its ``(n/p, n/p)`` buffer; ``n/p`` itself when it is
    that small. :func:`solve_layout` decides whether the solve pads instead.
    """
    return _largest_block(n // p)


def solve_layout(n: int, p: int) -> tuple[int, int]:
    """``(N, mb)``: the side an ``n x n`` eigh solves at on a ``p x p`` grid, and its block.

    ``(n, _block_size(n, p))`` when that block is at least :data:`_BLOCK_MIN`
    or ``n/p`` is at most :data:`_BLOCK_MAX`. Otherwise each rank's tile is
    padded to the smallest edge with a divisor in [128, 256] (at most 127
    rows per rank; 3954 -> 3956, block 172), and the padded rows carry
    sentinel eigenvalues below the spectrum (:func:`distributed_eigh`).
    """
    local = n // p
    block = _largest_block(local)
    if local <= _BLOCK_MAX or block >= _BLOCK_MIN:
        return n, block
    padded = next(m for m in range(local + 1, local + _BLOCK_MIN + 1)
                  if _largest_block(m) >= _BLOCK_MIN)
    return p * padded, _largest_block(padded)


def drop_sentinels(W, Q, n: int):
    """The ``n`` live eigenpairs of a :func:`pad_with_sentinels` solve, raw layout.

    The sentinels are the ``side - n`` lowest eigenvalues; ``Q``'s rows are
    the eigenvalue index (cuSOLVERMp's raw buffer), its columns the operand's
    rows, of which the first ``n`` are ``a``'s.
    """
    side = int(W.shape[-1])
    return W[side - n:], Q[side - n:, :n]


def retry_block(n: int, p: int) -> int | None:
    """A smaller block for the same solve side, or None: the retry layout.

    The largest divisor of the solve tile (:func:`solve_layout`) at most half
    the default block. The checked eigh solves there again when the default
    layout's result fails its check; the solve side, its operand and its
    output are the first attempt's, and the vendor workspace of a smaller
    block is not larger (n 7912, P4, complex: 284 MB at block 86, 317 MB at
    172), so the retry needs no more memory than the attempt it replaces. One
    tile per rank is never a retry: its workspace is 2.0 GB there and its wall
    exceeds the default's at large n/p. Below block 8 there is none (blocks
    1-4 ran 5-34x slower than one tile).
    """
    side, block = solve_layout(n, p)
    local = side // p
    return next((d for d in range(block // 2, 7, -1) if local % d == 0), None)


def _cyclic_rows_to_block(Q, *, n: int, p: int, mb: int):
    """Inside a shard_map: put rows held block-cyclically over 'x' in order
    (:func:`distrib_la._cyclic.cyclic_to_block` on ``'x'``)."""
    return cyclic_to_block(Q, n=n, p=p, mb=mb, axis="x")


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
    block: int | None = None,
    side: int | None = None,
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
        Must be True.  ``False`` (jobz='N') is REFUSED — see the guard
        below and ``resolve.resolve_backend`` guard 2c (bug L-3).
    block
        The square block edge, a divisor of the solve tile; default
        :func:`solve_layout`'s block. :func:`retry_block` is the smaller
        block the checked eigh retries at.
    side
        The solve side ``N >= n``, a multiple of the mesh; default
        :func:`solve_layout`'s (``side=n`` with ``block=_block_size(n, p)``
        is the unpadded layout, for benches).

    The solve runs block-cyclic on the operand's own tiles (the relabeling
    note above :data:`_BLOCK_MAX`). Where the tile has no divisor in
    [128, 256] it runs on a padded side ``N`` (:func:`solve_layout`): the
    padded rows and columns are zero except their diagonal, which holds
    distinct sentinels ``-2 b (1 + j/N)`` below the Gershgorin bound ``b`` of
    ``A``, so they are decoupled, come out as the ``N - n`` lowest
    eigenpairs, and are dropped. The result is ``A``'s ``n`` eigenpairs.

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
    # compute_evecs=False FIRST, before anything expensive.  The resolver's
    # guard 2c refuses this too, but it CANNOT be the only place: ``Plan``
    # forwards **kwargs straight to this wrapper (plan.py:318), so
    # ``plan("eigh", mesh, backend="cusolvermp")(A, compute_evecs=False)``
    # reaches here having passed a resolve that never saw the flag.  Same
    # lesson as bug L-1 in the other direction: a rule enforced in only one
    # of the two places is a rule with a hole in it.
    if not compute_evecs:
        raise RuntimeError(
            "cuSOLVERMp distributed_eigh: compute_evecs=False is REJECTED "
            "(bug L-3).  cuSOLVERMp 0.7.2 returns cusolverMpSyevd "
            "status=7 (CUSOLVER_STATUS_INTERNAL_ERROR) for jobz='N' at "
            "every n measured — 64/256/1024/4096 on a real 4-process 2x2 "
            "CUDA mesh, jobid 56447670, _reports/perf_gpu2x2_decomp.json. "
            "bufferSize succeeds and eigh_ffi.cc only forwards the flag, "
            "so the defect is the library's.  Pass compute_evecs=True "
            "(the default) and ignore Q, or use jnp.linalg.eigvalsh.")
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
    ctx_key = context_key(mesh)  # NCCL + cal_comm + cusolverMp

    if side is None:
        side, default = solve_layout(n, p)
    else:
        side, default = int(side), _block_size(int(side), p)
    mb = nb = default if block is None else int(block)
    if side < n or side % p or (side // p) % mb:
        raise ValueError(f"distributed_eigh: block {mb} does not divide the solve tile {side // p}")

    # jit(shard_map(...)) with a per-signature cache — the shape EVERY other
    # FFI wrapper in this package has.  This one was the exception: it
    # called shard_map at eager top level with no jax.jit around it, and the
    # exception was recorded rather than fixed (the note lived in
    # dispatch.py:230).  An eager shard_map RE-TRACES on every call, so a
    # caller that loops q paid one trace per matrix and got no cached
    # executable; measured on the same construction with a CPU stand-in,
    # 8 calls traced the body 8 times eagerly and ONCE under jit.
    #
    # It also matters for correctness of the batched dispatcher's serial
    # fallback: that path is the one taken on CUDA, and it calls this
    # wrapper Nq times.
    key = ("eigh", _mesh_key(mesh), A.dtype, n, side, mb, nb,
           bool(compute_evecs), int(ctx_key))
    jit_eigh = _JIT_CACHE.get(key)
    if jit_eigh is None:
        # Local output shapes + partition specs for shard_map.
        W_local = jax.ShapeDtypeStruct((side,), jnp.float64)    # replicated
        Q_local = jax.ShapeDtypeStruct((side // p, side // q), A.dtype)
        # Attributes forwarded to the C++ handler.
        attrs = dict(n=side, mb=mb, nb=nb,
                     ctx_key=int(ctx_key),
                     compute_evecs=bool(compute_evecs))

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=P("x", "y"),
                 out_specs=(P(), P("x", "y")),
                 check_vma=False)
        def _call(local_A):
            # The third result is the operand the solver destroys, aliased to
            # the input (ABI 10): the handler solves in that buffer instead of
            # a private copy in its scratch. Where the input is an
            # intermediate (the checked chain's deflated copy) XLA hands its
            # buffer over; where it is a caller's live array (a parameter
            # this wrapper does not donate) XLA copies it first, so the
            # caller's A is never written (EIGHMEM).
            W, Q, _destroyed = jax.ffi.ffi_call(
                _EIGH_TARGET, (W_local, Q_local, Q_local),
                input_output_aliases={0: 2})(local_A, **attrs)
            # Q's rows (axis 0, over 'x') are the eigenvalue index, which
            # cuSOLVERMp lays out block-cyclically like A's; put them back
            # in W's ascending order.
            if mb != side // p:
                Q = _cyclic_rows_to_block(Q, n=side, p=p, mb=mb)
            return W, Q

        face = NamedSharding(mesh, P("x", "y"))

        def _padded(a):
            whole = jax.lax.with_sharding_constraint(pad_with_sentinels(a, side), face)
            W, Q = _call(whole)
            W, Q = drop_sentinels(W, Q, n)
            return W, jax.lax.with_sharding_constraint(Q, face)

        # NO donate_argnums / input_output_aliases, deliberately: this
        # wrapper never donated its operand and distrib_la.plan.DONATES
        # declares eigh as donating nothing.  Adding donation here would
        # silently invalidate A for the caller — and the serial batched
        # path slices A[q] out of a stack the next iteration still needs.
        jit_eigh = jax.jit(_call if side == n else _padded)
        _JIT_CACHE[key] = jit_eigh

    return jit_eigh(A)


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


@dataclass(frozen=True)
class CusolverMpBatchedLU:
    """Opaque cuSOLVERMp LU factors and their per-rank pivot rows.

    ``raw`` contains the col-major block-cyclic LU bytes at
    ``P(None, 'y', 'x')``.  ``ipiv`` is sharded over the flattened
    ``('x','y')`` mesh so every process retains only the pivot row written
    by cuSOLVERMp.  Neither member leaves :class:`distrib_la.FactorToken`.
    """
    raw: jax.Array
    ipiv: jax.Array
    mesh: Mesh
    n: int
    mb: int
    nb: int
    nbatch: int


def batched_distributed_getrf(
    A: jax.Array,
    *,
    mesh: Mesh,
) -> CusolverMpBatchedLU:
    """Factor ``A[q]`` once via cuSOLVERMp ``getrf``.

    ``A`` is ``P(None,'x','y')`` and donated.  The returned handle keeps
    the factors and pivots in their native per-rank layouts; callers use
    :func:`batched_distributed_getrs` rather than inspecting either.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"batched_distributed_getrf: expected (Nq, N, N); got {A.shape}")
    Px, Py = _validate_mesh(mesh)
    nq, n = int(A.shape[0]), int(A.shape[1])
    if n % Px != 0 or n % Py != 0:
        raise ValueError(
            f"N={n} must be divisible by Px={Px} and Py={Py}.")

    loader.get_lib("CUDA")
    ctx_key = context_key(mesh, col_major=False)
    mb, nb = lu_blocks(n, Px, Py)
    ipiv_len = ipiv_local_len(n, Px, mb)
    key = ("getrf", _mesh_key(mesh), A.dtype, nq, n, mb, nb,
           int(ctx_key))
    jit_getrf = _JIT_CACHE.get(key)
    if jit_getrf is None:
        _announce_lu("getrf", mesh, nq, n, None, mb, nb)
        LU_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        ipiv_local = jax.ShapeDtypeStruct((nq, ipiv_len), jnp.int64)
        attrs = dict(nq=nq, n=n, mb=mb, nb=nb,
                     ipiv_len=ipiv_len, ctx_key=int(ctx_key))

        @partial(shard_map, mesh=mesh,
                 in_specs=P(None, "x", "y"),
                 out_specs=(P(None, "y", "x"), P(None, ("x", "y"))),
                 check_vma=False)
        def _getrf(local_A):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            return jax.ffi.ffi_call(
                _GETRF_TARGET, (LU_local_T, ipiv_local),
                input_output_aliases={0: 0},
            )(local_A_T, **attrs)

        jit_getrf = jax.jit(_getrf, donate_argnums=(0,))
        _JIT_CACHE[key] = jit_getrf

    LU_raw_T, ipiv = jit_getrf(A)
    return CusolverMpBatchedLU(
        raw=LU_raw_T, ipiv=ipiv, mesh=mesh, n=n, mb=mb, nb=nb,
        nbatch=nq)


def batched_distributed_getrs(
    LU: CusolverMpBatchedLU,
    B: jax.Array,
    *,
    mesh: Mesh,
) -> jax.Array:
    """Back-solve ``A[q] X[q] = B[q]`` from a split CUDA LU handle."""
    Px, Py = _validate_mesh(mesh)
    if mesh.axis_names != LU.mesh.axis_names:
        raise ValueError("getrs mesh axis names don't match the LU handle.")
    if (B.ndim != 3 or int(B.shape[0]) != LU.nbatch
            or int(B.shape[1]) != LU.n):
        raise ValueError(
            f"batched_distributed_getrs: B must be ({LU.nbatch}, {LU.n}, "
            f"NRHS); got {B.shape}")
    if LU.raw.dtype != B.dtype:
        raise ValueError(f"LU.dtype {LU.raw.dtype} != B.dtype {B.dtype}")
    nrhs = int(B.shape[2])
    if nrhs % Py != 0:
        raise ValueError(f"NRHS={nrhs} must be divisible by Py={Py}.")

    loader.get_lib("CUDA")
    ctx_key = context_key(mesh, col_major=False)
    mb_b, nb_b = LU.mb, rhs_block(LU.n, nrhs, Px, Py, LU.mb)
    ipiv_len = ipiv_local_len(LU.n, Px, LU.mb)
    key = ("getrs", _mesh_key(mesh), B.dtype, LU.nbatch, LU.n, nrhs,
           LU.mb, LU.nb, mb_b, nb_b, int(ctx_key))
    jit_getrs = _JIT_CACHE.get(key)
    if jit_getrs is None:
        _announce_lu("getrs", mesh, LU.nbatch, LU.n, nrhs, LU.mb, LU.nb, mb_b, nb_b)
        X_local_T = jax.ShapeDtypeStruct(
            (LU.nbatch, nrhs // Py, LU.n // Px), B.dtype)
        attrs = dict(nq=LU.nbatch, n=LU.n, nrhs=nrhs,
                     mb_a=LU.mb, nb_a=LU.nb, mb_b=mb_b, nb_b=nb_b,
                     ipiv_len=ipiv_len, ctx_key=int(ctx_key))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "y", "x"), P(None, ("x", "y")),
                           P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_vma=False)
        def _getrs(local_LU, local_ipiv, local_B):
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            X_T = jax.ffi.ffi_call(
                _GETRS_TARGET, X_local_T,
                input_output_aliases={2: 0},
            )(local_LU, local_ipiv, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_getrs = jax.jit(_getrs, donate_argnums=(2,))
        _JIT_CACHE[key] = jit_getrs

    return jit_getrs(LU.raw, LU.ipiv, B)


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
    ctx_key = context_key(mesh, col_major=False)

    mb = n // Px          # A's row block (per-rank local row count)
    nb = n // Py          # A's col block (per-rank local col count)

    key = ("potrf", _mesh_key(mesh), A.dtype, nq, n, mb, nb, int(ctx_key))
    jit_potrf = _JIT_CACHE.get(key)
    if jit_potrf is None:
        # Inner-dim transpose per slice: (Nq, N/Px, N/Py) row-major →
        # (Nq, N/Py, N/Px) row-major ≡ (N/Px, N/Py) col-major per slice.
        L_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        attrs = dict(nq=nq, n=n, mb=mb, nb=nb, ctx_key=int(ctx_key))

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
    ctx_key = context_key(mesh, col_major=False)

    # descA : mb=L.mb (rows block = N/Px), nb=L.nb (cols block = N/Py)
    # descB : mb_B = L.mb   (row block must equal A's row block — each rank's
    #                         local row count lld = N/Px)
    #         nb_B = Mrhs/Py (one col-tile per rank, matches JAX's
    #                         contiguous P(None, 'x', 'y') slab)
    mb_a, nb_a = L.mb, L.nb
    mb_b, nb_b = L.mb, mrhs // Py

    key = ("potrs", _mesh_key(mesh), B.dtype,
           nq, n, mrhs, mb_a, nb_a, mb_b, nb_b, int(ctx_key))
    jit_potrs = _JIT_CACHE.get(key)
    if jit_potrs is None:
        X_local_T = jax.ShapeDtypeStruct((nq, mrhs // Py, n // Px), B.dtype)
        attrs = dict(nq=nq, n=n, mrhs=mrhs,
                     mb_a=mb_a, nb_a=nb_a, mb_b=mb_b, nb_b=nb_b,
                     ctx_key=int(ctx_key))

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
    plan facade on CUDA meshes — by the W Dyson's distributed plan
    (``linalg = distributed``) for the general (non-Hermitian) ``I - V χ`` system.
    This fused entry remains for one-shot callers.  Repeated-RHS consumers
    use :func:`batched_distributed_getrf` plus
    :func:`batched_distributed_getrs` through the opaque top-level
    :func:`distrib_la.factor` / :func:`distrib_la.solve` token surface.

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
    ctx_key = context_key(mesh, col_major=False)

    mb_a, nb_a = lu_blocks(n, Px, Py)
    mb_b, nb_b = mb_a, rhs_block(n, nrhs, Px, Py, mb_a)

    key = ("solve_lu", _mesh_key(mesh), A.dtype,
           nq, n, nrhs, mb_a, nb_a, mb_b, nb_b, int(ctx_key))
    jit_solve = _JIT_CACHE.get(key)
    if jit_solve is None:
        _announce_lu("solve_lu", mesh, nq, n, nrhs, mb_a, nb_a, mb_b, nb_b)
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // Py, n // Px), B.dtype)
        LU_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs,
                     mb_a=mb_a, nb_a=nb_a, mb_b=mb_b, nb_b=nb_b,
                     ctx_key=int(ctx_key))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_vma=False)
        def _solve(local_A, local_B):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            # getrf and getrs work in place: the factors overwrite A and X
            # overwrites B, both declared, so XLA copies an operand only
            # while something still reads it.
            X_T, _ = jax.ffi.ffi_call(
                _SOLVE_LU_TARGET, (X_local_T, LU_local_T),
                input_output_aliases={0: 1, 1: 0},
            )(local_A_T, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_solve = jax.jit(_solve, donate_argnums=(0, 1))
        _JIT_CACHE[key] = jit_solve

    return jit_solve(A, B)

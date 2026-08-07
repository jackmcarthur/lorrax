"""SLATE FFI backend — distributed dense linear algebra (CUDA + host).

Ported verbatim from LORRAX ``src/ffi/linalg/_slate.py`` at 96a6399 (itself
the wave-2 merge of the former ``ffi.slate`` package: ``context.py``,
``cholesky.py``, ``trsm.py``, ``eigh.py``, ``batched.py``).  Nothing is
renamed; the only edits are the import lines — ``shard_map`` now comes from
:mod:`distrib_la._shard_map` and the loader from :mod:`distrib_la.loader`.

Reached only through the facade (:mod:`distrib_la.resolve` /
:mod:`distrib_la.plan`), which owns the guard table — including bug L-2:
SLATE's host ``heev`` SIGSEGVs, so the CPU distributed eigh is ScaLAPACK.

Section docstrings from the merged modules follow inline.
"""
from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from functools import partial
from typing import Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from distrib_la import loader
from distrib_la._shard_map import shard_map
from distrib_la.resolve import mesh_key

__all__ = [
    # context (was slate/context.py)
    "ensure_registered", "get_or_init_context",
    "validate_mesh", "validate_tile_layout",
    # single-matrix ops
    "SlateLowerL", "distributed_cholesky", "distributed_trsm",
    "distributed_eigh",
    # batched ops (was slate/batched.py)
    "SlateBatchedLowerL", "batched_distributed_cholesky",
    "batched_distributed_trsm",
]

# =========================================================================
#  Context lifecycle  (merged from slate/context.py; original docstring below)
# =========================================================================
"""Per-process singleton context for the SLATE FFI.

A SLATE context is cheap to build — just a dup'd (and rank-remapped)
MPI_Comm plus identity (rank, world, p, q).  We cache one per mesh-shape
to avoid re-dup'ing the communicator on every call.

Bootstrap: SLATE wants ``MPI_Init_thread(MPI_THREAD_MULTIPLE)`` and an
MPI communicator.  phdf5's ``phdf5_init_mpi()`` already initialises MPI
the same way, so if phdf5 is used in the same process we piggyback; if
not, ``lrx_slate_init_mpi`` (exported from context.cc) inits it directly.

The C++ side rank-remaps MPI_COMM_WORLD so SLATE's hardcoded
``GridOrder::Col`` matches JAX's ``P('x','y')`` shard-to-rank mapping.
See ``cpp/context.cc`` for the formula and ``distributed_cholesky`` below
for the local-transpose pairing on the Python side.

Restrictions
------------
* Mesh must have axis names ``('x', 'y')``.  If you have a 3-D mesh
  (e.g. for batched ops with a ``'batch'`` axis), build a 2-D
  sub-mesh out of its (x, y) axes for the SLATE call — the FFI
  doesn't support sub-mesh selection yet.
* For 4 GPUs total, ``p * q == jax.process_count()``.  Sub-mesh
  partial-world support is a planned extension.
* ``heev`` further requires ``p == q``; ``potrf`` and ``trsm`` don't.
"""

MeshKey = Tuple[int, int]  # (p, q)

_LOCK = threading.Lock()
# (p, q) -> (handle, ffi platform that created it).  A SlateCtx is pure MPI
# and platform-agnostic, so a GPU mesh and a CPU mesh of the same shape share
# one ctx; the platform is remembered only so teardown goes through a library
# that is definitely loaded.
_CACHE: Dict[MeshKey, Tuple[int, str]] = {}
_SUBROW_CACHE: Dict[MeshKey, Tuple[int, str]] = {}


def _mesh_platform(mesh: Mesh) -> str:
    plat = mesh.devices.flat[0].platform
    return "CUDA" if plat in ("gpu", "cuda") else "cpu"


def ensure_registered(mesh: Mesh) -> None:
    """Load + register the FFI library for the MESH's device platform.

    ``jax.ffi.ffi_call`` resolves the target registered for the lowering
    platform, so the library that matters is the one matching the mesh's
    devices — not necessarily the default backend (e.g. slate ops on a
    CPU-device mesh inside a GPU-backend process).  Idempotent.
    """
    loader.get_lib(_mesh_platform(mesh))


def validate_mesh(mesh: Mesh, *, require_square: bool = False) -> Tuple[int, int]:
    """Validate that ``mesh`` is usable by SLATE FFI ops; return ``(p, q)``.

    Checks: required axis names ``('x', 'y')``; ``p * q ==
    jax.process_count()``; optionally ``p == q`` (for ``heev``).
    """
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"slate FFI: mesh must have axes ('x', 'y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = int(jax.process_count())
    if p * q != world:
        raise ValueError(
            f"slate FFI: mesh {p}x{q} (={p*q}) != jax.process_count() = {world}. "
            f"Sub-mesh / partial-world calls aren't supported yet.")
    if require_square and p != q:
        raise ValueError(
            f"slate FFI: this op requires a square mesh (p == q); got {p}x{q}.")
    return p, q


def validate_tile_layout(n: int, nb: int, p: int, q: int, *, what: str,
                         allow_row_grid: bool = False) -> None:
    """Reject (n, nb, p, q) combos where JAX block shards ≠ SLATE tiles.

    ``fromDevices`` interprets each rank's buffer as 2-D block-cyclic
    tiles of size ``nb``; JAX hands the FFI a CONTIGUOUS (n/p, n/q)
    block.  The two coincide only when every multi-proc mesh axis holds
    exactly one tile per rank (``nb == n/axis``); a single-proc axis is
    layout-free (all tiles local, contiguous col-major).  Violations
    don't fail loudly — SLATE silently assembles a permuted global
    matrix (or throws an uncatchable ``blas::Error`` from an OpenMP task,
    killing every rank) — so they must be rejected here.

    ``p == 1 < q`` grids additionally trip a size-dependent SLATE
    assertion (``internal_batch.hh:290: group.ld[m] == Mij.stride()``,
    SIGABRT on every rank): the local buffer stride is ``lld = n`` while
    tiles are ``nb = n/q``, and SLATE's device-region batching requires
    uniform strides within a group.  ``p >= q`` meshes have
    ``lld == nb`` and are safe.  Reproduced 2026-07-10 on 1x4 potrf
    (n=64, c128) with the eval build @ ded15290.  ``allow_row_grid``
    opts the batched wrappers out of this check: their per-slice (1, Py)
    sub-grid is the same stride class but is production-validated at
    GWJAX scale on 2x2 — the assert there is size-dependent (the
    nbatch=8/n=128 1x4 repro) and remains an accepted, documented risk.
    """
    if p > 1 and q > 1 and p != q:
        raise ValueError(
            f"{what}: mesh {p}x{q} unsupported — with both axes > 1 the "
            f"square SLATE tile size cannot give one tile per rank on "
            f"both axes unless p == q.  Use a square or Nx1 mesh.")
    if p == 1 and q > 1 and not allow_row_grid:
        raise ValueError(
            f"{what}: mesh {p}x{q} unsupported — on a 1xq grid the local "
            f"stride (lld = n) differs from the tile size (nb = n/q) and "
            f"SLATE's device-region batching aborts every rank with the "
            f"internal_batch.hh:290 stride assertion.  Use the "
            f"transposed (qx1) mesh instead.")
    if p > 1 and nb != n // p:
        raise ValueError(
            f"{what}: block_size={nb} != n/p={n // p} on a {p}x{q} mesh — "
            f"JAX's block sharding only matches SLATE's block-cyclic "
            f"layout at one tile per rank.  Omit block_size.")
    if q > 1 and nb != n // q:
        raise ValueError(
            f"{what}: block_size={nb} != n/q={n // q} on a {p}x{q} mesh — "
            f"JAX's block sharding only matches SLATE's block-cyclic "
            f"layout at one tile per rank.  Omit block_size.")


def _make_ctx(mesh: Mesh) -> int:
    ensure_registered(mesh)
    rank = int(jax.process_index())
    world = int(jax.process_count())
    p, q = validate_mesh(mesh)
    return int(loader.create_slate_context(
        rank=rank, world_size=world, p=p, q=q,
        platform=_mesh_platform(mesh)))


def get_or_init_context(mesh: Mesh) -> int:
    """Return the opaque int handle for the SLATE context of this mesh.

    Thread-safe.  First call collectively dups MPI_COMM_WORLD.
    """
    key = validate_mesh(mesh)
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            return entry[0]
        h = _make_ctx(mesh)
        _CACHE[key] = (h, _mesh_platform(mesh))
        return h


def _make_subrow_ctx(mesh: Mesh) -> int:
    ensure_registered(mesh)
    rank = int(jax.process_index())
    world = int(jax.process_count())
    Px, Py = validate_mesh(mesh)
    return int(loader.create_slate_subrow_context(
        rank=rank, world_size=world, Px=Px, Py=Py,
        platform=_mesh_platform(mesh)))


def _get_or_init_subrow_context(mesh: Mesh) -> int:
    """The per-X-row Y-axis sub-comm SLATE context for this mesh.

    PRIVATE.  It was exported, and re-exported twice more through the
    ffi.slate shims, with no caller outside this file at 96a6399 -- a
    per-process singleton over an MPI sub-communicator is not something a
    consumer has any business holding.  Its two real callers are the two
    batched wrappers below.

    ``MPI_COMM_WORLD`` split by x-coordinate: one comm of size ``Py`` per
    X-row.  Thread-safe; the first call collectively splits.  Used by the
    batched ops, where each X-row handles an independent slice of a
    ``(Nbatch, N, N)`` input distributed along ``'x'`` (batch) and ``'y'``
    (inner matrix).
    """
    key = validate_mesh(mesh)
    with _LOCK:
        entry = _SUBROW_CACHE.get(key)
        if entry is not None:
            return entry[0]
        h = _make_subrow_ctx(mesh)
        _SUBROW_CACHE[key] = (h, _mesh_platform(mesh))
        return h


def _atexit_teardown() -> None:
    for cache in (_CACHE, _SUBROW_CACHE):
        for _, (h, plat) in list(cache.items()):
            try:
                loader.destroy_slate_context(int(h), platform=plat)
            except Exception:                                  # noqa: BLE001
                pass
        cache.clear()


atexit.register(_atexit_teardown)


# =========================================================================
#  cholesky  (merged from slate/cholesky.py)
# =========================================================================
"""``distributed_cholesky`` — JAX FFI wrapper around ``slate::potrf``.

Layout
------
JAX is row-major; SLATE tiles are column-major.  Each rank locally
transposes its own shard via ``shard_map`` + ``jnp.transpose`` (no
inter-rank comm) so the bytes SLATE reads are correctly col-major.
Combined with the C++-side MPI rank remap (``cpp/context.cc``), SLATE
sees the user's matrix correctly assembled on **any p × q mesh**.

Returns a :class:`SlateLowerL` opaque handle.  Two ways to consume it:

  * Pass the handle to :func:`distributed_trsm` — trsm knows the handle's
    layout and feeds it straight back to SLATE.
  * Call :meth:`SlateLowerL.to_jax_lower` for a conventional row-major
    lower-triangular L.

Only ``Uplo::Lower`` (SLATE's potrf), dtypes F64 / C128.
"""

_POTRF_FFI_TARGET = "lorrax_slate_potrf"


@dataclass(frozen=True)
class SlateLowerL:
    """Opaque handle to a SLATE-format Cholesky lower factor.

    Attributes
    ----------
    raw : jax.Array
        Shape ``(n, n)``, sharded ``P('y', 'x')`` on ``mesh``.  Bytes
        are SLATE's col-major L tiles; JAX reading this directly would
        give ``L.T`` (or ``L.conj().T`` for complex).
    mesh : Mesh
        2-D mesh the factor was computed on.
    n : int
        Matrix side length.
    nb : int
        Tile block size SLATE used.
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    nb: int

    def to_jax_lower(self) -> jax.Array:
        """Return L in the conventional JAX row-major lower-triangular form.

        Local-transposes the per-rank buffer (P('y','x') → P('x','y'),
        no inter-rank comm) and strips the (zeroed) strict-upper.
        """
        @partial(shard_map, mesh=self.mesh,
                 in_specs=P("y", "x"), out_specs=P("x", "y"),
                 check_vma=False)
        def _local_T(local):
            return jnp.transpose(local, (1, 0))
        return jnp.tril(_local_T(self.raw))


def distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
    block_size: Optional[int] = None,
) -> SlateLowerL:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_cholesky: expected square A; got {A.shape}")
    p, q = validate_mesh(mesh)
    n = int(A.shape[0])
    if n % p != 0 or n % q != 0:
        raise ValueError(
            f"distributed_cholesky: n={n} must be divisible by both mesh "
            f"axes ({p},{q}).")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    # Default tile size divides by the larger grid axis so each rank
    # still holds >=1 tile along both directions.
    nb = n // max(p, q) if block_size is None else int(block_size)
    validate_tile_layout(n, nb, p, q, what="distributed_cholesky")

    # Local transpose via shard_map: each rank flips its own (n/p, n/q)
    # row-major shard to (n/q, n/p) row-major.  Bytes are the same set
    # the rank already had — no inter-rank communication.  After the
    # transpose, those bytes equal the original block in col-major
    # layout, which SLATE reads correctly.  Combined with the comm-remap
    # on the C++ side (context.cc), SLATE assembles the global A
    # correctly for any p x q mesh.
    L_local_T = jax.ShapeDtypeStruct((n // q, n // p), A.dtype)
    attrs = dict(n=n, nb=nb, ctx_handle=int(ctx_handle))

    @partial(shard_map, mesh=mesh,
             in_specs=P("x", "y"), out_specs=P("y", "x"),
             check_vma=False)
    def _potrf(local_A):
        local_A_T = jnp.transpose(local_A, (1, 0))
        return jax.ffi.ffi_call(_POTRF_FFI_TARGET, L_local_T)(local_A_T, **attrs)

    L_raw_T = _potrf(A)
    return SlateLowerL(raw=L_raw_T, mesh=mesh, n=n, nb=nb)


# =========================================================================
#  trsm  (merged from slate/trsm.py)
# =========================================================================
"""``distributed_trsm`` — JAX FFI wrapper around ``slate::trsm``.

Solves one of:
    op(A) @ X = alpha * B   (side='L')
    X @ op(A) = alpha * B   (side='R')

Layout: same scheme as ``distributed_cholesky`` (per-rank local
transpose + C++ side rank remap).

For the :class:`SlateLowerL` handle path the underlying buffer is
already in SLATE-tile col-major (``P('y','x')``-sharded), so A skips
the local transpose; ``side``/``uplo`` are pinned by the handle and
``op`` selects forward (``'N'``: ``L X = B``) vs adjoint (``'C'``:
``L^H X = B``).
"""

_TRSM_FFI_TARGET = "lorrax_slate_trsm"

_SIDE = {"L": 0, "R": 1}
_UPLO = {"L": 0, "U": 1}
_OP = {"N": 0, "T": 1, "C": 2}
_DIAG = {"N": 0, "U": 1}


def distributed_trsm(
    A,
    B: jax.Array,
    *,
    mesh: Mesh,
    side: Literal["L", "R"] = "L",
    uplo: Literal["L", "U"] = "L",
    op: Literal["N", "T", "C"] = "N",
    diag: Literal["N", "U"] = "N",
    alpha: complex | float = 1.0,
    block_size: int | None = None,
) -> jax.Array:
    """Distributed triangular solve.

    A : SlateLowerL or square 2-D jax.Array sharded ``P('x','y')``.
    B : 2-D jax.Array sharded ``P('x','y')``.

    For the SlateLowerL handle path, side/uplo are pinned to the
    standard cholesky-factor convention: op='N' is forward solve
    (L @ X = B); op='C' is adjoint (L^H @ X = B).
    """
    p, q = validate_mesh(mesh)

    if isinstance(A, SlateLowerL):
        # Handle.raw is already in SLATE-tile col-major (P('y','x') sharded).
        if op not in ("N", "C"):
            raise ValueError(f"distributed_trsm(SlateLowerL, ...): "
                             f"op must be 'N' or 'C'; got {op!r}")
        side, uplo = "L", "L"
        n = A.n
        if block_size is None:
            block_size = A.nb
        if mesh.axis_names != A.mesh.axis_names:
            raise ValueError(
                "trsm mesh axis names don't match the handle's mesh.")
        A_is_handle = True
        A_arg = A.raw
    else:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"distributed_trsm: expected square A; got {A.shape}")
        n = int(A.shape[0])
        A_is_handle = False
        A_arg = A

    if B.ndim != 2:
        raise ValueError(f"distributed_trsm: expected 2D B; got {B.shape}")
    if A_arg.dtype != B.dtype:
        raise ValueError(
            f"distributed_trsm: A.dtype {A_arg.dtype} != B.dtype {B.dtype}")

    if side == "L":
        if B.shape[0] != n:
            raise ValueError(
                f"side='L' requires B.shape[0]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[1])
    else:
        if B.shape[1] != n:
            raise ValueError(
                f"side='R' requires B.shape[1]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[0])
    # B is sharded P('x','y'): axis 0 over p, axis 1 over q — for BOTH
    # sides (the old per-side (n % p, m % q) check validated the wrong
    # axes for side='R').
    if int(B.shape[0]) % p != 0 or int(B.shape[1]) % q != 0:
        raise ValueError(
            f"B.shape={tuple(B.shape)} must be divisible by the mesh axes "
            f"({p},{q}) along (rows, cols)")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)
    nb = n // max(p, q) if block_size is None else int(block_size)
    validate_tile_layout(n, nb, p, q, what="distributed_trsm")

    # Local transpose of B: each rank flips its (B0/p, B1/q) row-major
    # shard to (B1/q, B0/p) row-major (= original block in col-major
    # layout).  Local op only; no inter-rank comm.
    bshape_local_T = (B.shape[1] // q, B.shape[0] // p)
    X_local_T = jax.ShapeDtypeStruct(bshape_local_T, B.dtype)

    alpha_c = complex(alpha)
    attrs = dict(
        n=n, m=m, nb=nb,
        side=_SIDE[side], uplo=_UPLO[uplo], op=_OP[op], diag=_DIAG[diag],
        alpha_re=float(alpha_c.real),
        alpha_im=float(alpha_c.imag),
        ctx_handle=int(ctx_handle),
    )

    # When A is a handle, A_arg is already P('y','x') sharded with
    # col-major bytes; just feed it through.  Otherwise, local-transpose.
    if A_is_handle:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P("y", "x"), P("x", "y")),
                 out_specs=P("y", "x"), check_vma=False)
        def _trsm(local_A_handle, local_B):
            local_B_T = jnp.transpose(local_B, (1, 0))
            return jax.ffi.ffi_call(_TRSM_FFI_TARGET, X_local_T)(
                local_A_handle, local_B_T, **attrs)
        X_T = _trsm(A_arg, B)
    else:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P("x", "y"), P("x", "y")),
                 out_specs=P("y", "x"), check_vma=False)
        def _trsm(local_A, local_B):
            local_A_T = jnp.transpose(local_A, (1, 0))
            local_B_T = jnp.transpose(local_B, (1, 0))
            return jax.ffi.ffi_call(_TRSM_FFI_TARGET, X_local_T)(
                local_A_T, local_B_T, **attrs)
        X_T = _trsm(A_arg, B)

    # Local-transpose X back to user's P('x','y') row-major.
    @partial(shard_map, mesh=mesh,
             in_specs=P("y", "x"), out_specs=P("x", "y"),
             check_vma=False)
    def _untranspose(local_X_T):
        return jnp.transpose(local_X_T, (1, 0))
    return _untranspose(X_T)


# =========================================================================
#  eigh  (merged from slate/eigh.py)
# =========================================================================
"""``distributed_eigh`` — JAX FFI wrapper around ``slate::heev``.

Shape contract mirrors the cuSOLVERMp wrapper so call sites can swap
backends — with one upgrade: the returned ``Q`` here is the TRUE
eigenvector matrix (columns are eigenvectors of A, ``A @ Q == Q @
diag(W)``), not the conj-transposed raw buffer cuSOLVERMp returns.

Layout: same scheme as ``distributed_cholesky`` — per-rank local
transpose on the input (row-major JAX shard → col-major bytes for
SLATE) paired with the C++ comm rank-remap, and a local untranspose on
the returned eigenvector tiles.
"""

_EIGH_FFI_TARGET = "lorrax_slate_eigh"


def distributed_eigh(
    A: jax.Array,
    *,
    mesh: Mesh,
    compute_evecs: bool = True,
    block_size: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition via SLATE's ``heev``.

    Parameters
    ----------
    A
        Square ``(n, n)`` array, dtype ``float64`` (symmetric) or
        ``complex128`` (Hermitian), sharded ``P('x','y')`` on ``mesh``.
    mesh
        2-D ``Mesh`` labelled ``('x','y')``.  Must be *square*
        (``mesh.shape['x'] == mesh.shape['y']``) — SLATE's heev rejects
        rectangular grids.
    compute_evecs
        If False, only eigenvalues are meaningful; ``Q`` is still returned
        but its contents are unspecified.
    block_size
        Override SLATE's tile size ``nb``.  Default ``n/p``, which is the
        ONLY value for which JAX's block sharding coincides with SLATE's
        block-cyclic tile grid on a multi-rank mesh — overriding it there
        makes SLATE assemble a permuted global matrix (eigenvalues survive
        the symmetric permutation; eigenvectors do not).  Overriding is
        therefore rejected on multi-rank meshes; on a 1×1 mesh any ``nb``
        is layout-safe.

    Returns
    -------
    W
        ``(n,)`` real eigenvalues, ascending, replicated.  Dtype is the
        real-part of A's dtype (``float64`` for both F64 and C128).
    Q
        ``(n, n)`` eigenvectors of A (as columns), same dtype as A,
        sharded ``P('x','y')``.  ``A @ Q == Q @ diag(W)`` to machine
        precision.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_eigh: expected a square matrix, "
                         f"got {A.shape}")
    p, q = validate_mesh(mesh, require_square=True)
    n = int(A.shape[0])
    if n % p != 0:
        raise ValueError(
            f"distributed_eigh: n={n} must be divisible by mesh axis size {p}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    nb = n // p if block_size is None else int(block_size)
    validate_tile_layout(n, nb, p, q, what="distributed_eigh")

    local_rows = n // p
    local_cols = n // q
    # Real-part dtype for eigenvalues (float64 for both F64 and C128).
    w_dtype = jnp.float64 if A.dtype in (jnp.complex128, jnp.float64) \
        else jnp.float32
    W_local = jax.ShapeDtypeStruct((n,), w_dtype)               # replicated
    # Q comes back as SLATE col-major tiles = row-major bytes of the
    # per-rank block of Q^T; assembled at P('y','x') then locally
    # untransposed (square mesh: local shape is (n/q, n/p)).
    Q_local_T = jax.ShapeDtypeStruct((local_cols, local_rows), A.dtype)

    attrs = dict(
        n=n, nb=nb,
        ctx_handle=int(ctx_handle),
        compute_evecs=bool(compute_evecs),
    )

    @partial(shard_map,
             mesh=mesh,
             in_specs=P("x", "y"),
             out_specs=(P(), P("y", "x")),
             check_vma=False)
    def _call(local_A):
        # Local transpose: row-major shard bytes → col-major block, the
        # layout SLATE reads (pure local op, pairs with the C++ comm
        # rank-remap).  Same convention as distributed_cholesky.
        local_A_T = jnp.transpose(local_A, (1, 0))
        return jax.ffi.ffi_call(
            _EIGH_FFI_TARGET, (W_local, Q_local_T))(local_A_T, **attrs)

    W, Q_T = _call(A)

    @partial(shard_map, mesh=mesh,
             in_specs=P("y", "x"), out_specs=P("x", "y"),
             check_vma=False)
    def _untranspose(local_Q_T):
        return jnp.transpose(local_Q_T, (1, 0))

    return W, _untranspose(Q_T)


# =========================================================================
#  batched ops  (merged from slate/batched.py)
# =========================================================================
"""``batched_distributed_{cholesky,trsm}`` — 3-D batched SLATE wrappers.

Use case: a batch of ``Nbatch`` independent ``N × N`` Hermitian PD
matrices, where each one is too big for a single GPU so must be 2-D
sharded, and multiple matrices should run in parallel across the mesh.

Sharding contract
-----------------
Inputs / outputs have shape ``(Nbatch, N, N)`` (cholesky) or
``(Nbatch, N, Nrhs)`` (trsm RHS), sharded ``P('x', None, 'y')`` on a
2-D ``('x', 'y')`` mesh of shape ``(Px, Py)``.  The batch is split along
``'x'`` and the per-matrix distribution happens along ``'y'``.  Each
X-row of the mesh gets its own MPI sub-communicator of size ``Py``, and
all ranks in a row cooperatively process the ``Nbatch/Px`` matrices
local to that row.

SLATE itself has no native batched potrf/trsm, so the C++ handler is
literally a ``for`` loop over the per-rank batch dimension — no
communication is shared across batch iterations.  This beats
``lax.scan``-ing the single-matrix primitive because the sub-comm and
device context are set up once and reused.

Restrictions
------------
* Mesh must be 2-D with axes ``('x', 'y')``.
* ``Nbatch`` must be divisible by ``Px``; ``N`` (and ``Nrhs``) divisible
  by ``Py``.
* Dtypes: F64 / C128.
"""

_POTRF_TARGET = "lorrax_slate_batched_potrf"
_TRSM_TARGET = "lorrax_slate_batched_trsm"

# _SIDE/_UPLO/_OP/_DIAG: shared with the single-matrix trsm section above
# (batched.py carried an identical copy; merged).

# jit(shard_map(...)) cache per signature.  shard_map in eager mode
# re-traces per call; wrapping in jax.jit once and reusing eliminates
# that cost when the user loops the same-shape op (e.g. batched_trsm
# across many RHS chunks).
_JIT_CACHE: dict = {}


#: The package's ONE mesh cache key (:func:`distrib_la.mesh_key`).  This
#: module's private spelling is kept because ``_scalapack`` and eight call
#: sites in this file import it by that name; it is an alias, not a copy.
_mesh_key = mesh_key


@dataclass(frozen=True)
class SlateBatchedLowerL:
    """Batched-cholesky output handle.

    Attributes
    ----------
    raw : jax.Array
        Shape ``(Nbatch, N, N)``, sharded ``P('x', 'y', None)``.  Bytes
        hold SLATE's col-major L tiles per slice; JAX reading the inner
        ``(N, N)`` directly would give ``L.T`` (or ``L.conj().T``).
    mesh : Mesh
    n : int
        Per-slice side length.
    nb : int
    nbatch : int
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    nb: int
    nbatch: int


def batched_distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
    block_size: Optional[int] = None,
) -> SlateBatchedLowerL:
    """Factor each ``A[q]`` = ``L[q] L[q]^H`` via ``slate::potrf``.

    Parameters
    ----------
    A : jax.Array
        Shape ``(Nbatch, N, N)``, dtype F64 or C128, sharded
        ``P('x', None, 'y')``.
    mesh : Mesh
    block_size : int, optional
        Per-matrix SLATE tile size.  Default ``N // Py``.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"batched_distributed_cholesky: expected (Nbatch, N, N); "
            f"got {A.shape}")
    Px, Py = validate_mesh(mesh)
    nbatch = int(A.shape[0])
    n = int(A.shape[1])
    if nbatch % Px != 0:
        raise ValueError(
            f"batched_distributed_cholesky: Nbatch={nbatch} must be "
            f"divisible by mesh 'x' axis size {Px}.")
    if n % Py != 0:
        raise ValueError(
            f"batched_distributed_cholesky: N={n} must be divisible by "
            f"mesh 'y' axis size {Py}.")

    ensure_registered(mesh)
    ctx_handle = _get_or_init_subrow_context(mesh)

    nb_batch_local = nbatch // Px
    nb = n // Py if block_size is None else int(block_size)
    # Per-slice SLATE grid is (1, Py) on the sub-row comm.
    validate_tile_layout(n, nb, 1, Py, what="batched_distributed_cholesky",
                         allow_row_grid=True)

    key = ("potrf", _mesh_key(mesh), A.dtype,
           nbatch, n, nb, int(ctx_handle))
    jit_potrf = _JIT_CACHE.get(key)
    if jit_potrf is None:
        # Local transpose of the inner two dims: (Nb_local, N, N/Py) row-major
        # → (Nb_local, N/Py, N) row-major, which is (N, N/Py) col-major per
        # slice — the layout SLATE's fromDevices expects with p=1, q=Py.
        L_local_T = jax.ShapeDtypeStruct(
            (nb_batch_local, n // Py, n), A.dtype)
        attrs = dict(
            nbatch_local=nb_batch_local, n=n, nb=nb,
            ctx_handle=int(ctx_handle),
        )

        @partial(shard_map, mesh=mesh,
                 in_specs=P("x", None, "y"),
                 out_specs=P("x", "y", None),
                 check_vma=False)
        def _potrf(local_A):
            # local_A shape: (Nb_local, N, N/Py)
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            return jax.ffi.ffi_call(_POTRF_TARGET, L_local_T)(local_A_T, **attrs)

        jit_potrf = jax.jit(_potrf)
        _JIT_CACHE[key] = jit_potrf

    L_raw_T = jit_potrf(A)
    return SlateBatchedLowerL(
        raw=L_raw_T, mesh=mesh, n=n, nb=nb, nbatch=nbatch)


def batched_distributed_trsm(
    A,
    B: jax.Array,
    *,
    mesh: Mesh,
    side: Literal["L", "R"] = "L",
    uplo: Literal["L", "U"] = "L",
    op: Literal["N", "T", "C"] = "N",
    diag: Literal["N", "U"] = "N",
    alpha: complex | float = 1.0,
    block_size: Optional[int] = None,
) -> jax.Array:
    """Batched triangular solve: one ``slate::trsm`` per batch slice.

    ``A`` is a :class:`SlateBatchedLowerL` handle (preferred) or a plain
    batched ``(Nbatch, N, N)`` array sharded ``P('x', None, 'y')``.  For
    the handle path, ``side/uplo`` are pinned to ``'L'/'L'`` and
    ``op='N'`` / ``'C'`` selects forward / adjoint solve.

    ``B`` has shape ``(Nbatch, N, Nrhs)`` sharded ``P('x', None, 'y')``.
    Output has the same shape / sharding as ``B``.
    """
    Px, Py = validate_mesh(mesh)

    if isinstance(A, SlateBatchedLowerL):
        if op not in ("N", "C"):
            raise ValueError(
                f"batched_distributed_trsm(SlateBatchedLowerL, ...): "
                f"op must be 'N' or 'C'; got {op!r}")
        side, uplo = "L", "L"
        nbatch = A.nbatch
        n = A.n
        if block_size is None:
            block_size = A.nb
        if mesh.axis_names != A.mesh.axis_names:
            raise ValueError(
                "batched trsm mesh axis names don't match the handle's mesh.")
        A_is_handle = True
        A_arg = A.raw
    else:
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError(
                f"batched_distributed_trsm: expected (Nbatch, N, N) A; "
                f"got {A.shape}")
        nbatch = int(A.shape[0])
        n = int(A.shape[1])
        A_is_handle = False
        A_arg = A

    if B.ndim != 3:
        raise ValueError(
            f"batched_distributed_trsm: expected 3D B; got {B.shape}")
    if int(B.shape[0]) != nbatch:
        raise ValueError(
            f"batched_distributed_trsm: B.shape[0]={B.shape[0]} != "
            f"Nbatch={nbatch}")
    if A_arg.dtype != B.dtype:
        raise ValueError(
            f"batched_distributed_trsm: A.dtype {A_arg.dtype} != "
            f"B.dtype {B.dtype}")

    if side == "L":
        if int(B.shape[1]) != n:
            raise ValueError(
                f"side='L' requires B.shape[1]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[2])
    else:
        if int(B.shape[2]) != n:
            raise ValueError(
                f"side='R' requires B.shape[2]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[1])

    if nbatch % Px != 0:
        raise ValueError(
            f"Nbatch={nbatch} must be divisible by mesh 'x' axis {Px}.")
    if n % Py != 0 or m % Py != 0:
        raise ValueError(
            f"N={n}, M={m} must be divisible by mesh 'y' axis {Py}.")

    ensure_registered(mesh)
    ctx_handle = _get_or_init_subrow_context(mesh)
    nbatch_local = nbatch // Px
    nb = n // Py if block_size is None else int(block_size)
    validate_tile_layout(n, nb, 1, Py, what="batched_distributed_trsm",
                         allow_row_grid=True)

    alpha_c = complex(alpha)
    key = ("trsm", _mesh_key(mesh), B.dtype, A_is_handle,
           nbatch, n, m, nb,
           _SIDE[side], _UPLO[uplo], _OP[op], _DIAG[diag],
           float(alpha_c.real), float(alpha_c.imag),
           int(ctx_handle))
    jit_trsm = _JIT_CACHE.get(key)
    if jit_trsm is None:
        # Expected per-slice: (B.shape[1], B.shape[2]/Py) row-major → transpose
        # inner two dims → (B.shape[2]/Py, B.shape[1]) row-major ≡ col-major.
        X_local_T_shape = (nbatch_local, B.shape[2] // Py, B.shape[1])
        X_local_T = jax.ShapeDtypeStruct(X_local_T_shape, B.dtype)

        attrs = dict(
            nbatch_local=nbatch_local,
            n=n, m=m, nb=nb,
            side=_SIDE[side], uplo=_UPLO[uplo],
            op=_OP[op], diag=_DIAG[diag],
            alpha_re=float(alpha_c.real),
            alpha_im=float(alpha_c.imag),
            ctx_handle=int(ctx_handle),
        )

        if A_is_handle:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P("x", "y", None), P("x", None, "y")),
                     out_specs=P("x", None, "y"),
                     check_vma=False)
            def _trsm(local_A_handle, local_B):
                local_B_T = jnp.transpose(local_B, (0, 2, 1))
                X_T = jax.ffi.ffi_call(_TRSM_TARGET, X_local_T)(
                    local_A_handle, local_B_T, **attrs)
                # Untranspose inside the shard_map → skip a second pass.
                return jnp.transpose(X_T, (0, 2, 1))
        else:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P("x", None, "y"), P("x", None, "y")),
                     out_specs=P("x", None, "y"),
                     check_vma=False)
            def _trsm(local_A, local_B):
                local_A_T = jnp.transpose(local_A, (0, 2, 1))
                local_B_T = jnp.transpose(local_B, (0, 2, 1))
                X_T = jax.ffi.ffi_call(_TRSM_TARGET, X_local_T)(
                    local_A_T, local_B_T, **attrs)
                return jnp.transpose(X_T, (0, 2, 1))

        jit_trsm = jax.jit(_trsm)
        _JIT_CACHE[key] = jit_trsm

    return jit_trsm(A_arg, B)

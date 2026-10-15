"""SLATE FFI backend — distributed dense linear algebra (CUDA, ROCM + host).

Ported verbatim from LORRAX ``src/ffi/linalg/_slate.py`` at 96a6399 (itself
the wave-2 merge of the former ``ffi.slate`` package: ``context.py``,
``cholesky.py``, ``trsm.py``, ``eigh.py``, ``batched.py``).  Nothing is
renamed; the only edits are the import lines — ``shard_map`` now comes from
``jax`` and the loader from :mod:`distrib_la.loader`.

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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la import loader
from distrib_la._cyclic import cyclic_to_block, largest_divisor, pad_with_sentinels
from jax import shard_map
from distrib_la.resolve import mesh_key, mesh_platform

__all__ = [
    # context (was slate/context.py)
    "ensure_registered", "get_or_init_context", "context_key",
    "validate_mesh", "validate_tile_layout",
    # single-matrix ops
    "SlateLowerL", "distributed_cholesky",
    "distributed_eigh", "eigh_layout",
    # the distributed LU family (ROCM leg): fused + split getrf/getrs
    "batched_distributed_solve_lu",
    "batched_distributed_getrf",
    "batched_distributed_getrs",
]

# =========================================================================
#  Context lifecycle  (merged from slate/context.py; original docstring below)
# =========================================================================
"""Per-process singleton context for the SLATE FFI.

A SLATE context is cheap to build — just a dup'd (and rank-remapped)
MPI_Comm plus identity (rank, world, p, q).  We cache one per platform and
mesh shape to avoid re-dup'ing the communicator on every call.  The host
and CUDA handlers resolve keys in separate native registries.

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
* ``heev`` further requires ``p == q``; ``potrf`` doesn't.
"""

MeshKey = Tuple[int, int]  # (p, q)
ContextKey = Tuple[str, int, int]  # (platform, p, q)

_LOCK = threading.Lock()
# The two platform libraries own separate ctx_key registries and lifetimes.
# A context created by one library must never be sent to the other.
_CACHE: Dict[ContextKey, Tuple[int, str]] = {}
_SUBROW_CACHE: Dict[ContextKey, Tuple[int, str]] = {}


def _mesh_platform(mesh: Mesh) -> str:
    """The mesh's leg key (``lxkit.LEGS``: cpu / CUDA / ROCM), from the device
    vendor -- never from ``device.platform``, which reads ``gpu`` on ROCm too."""
    return mesh_platform(mesh)


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
    shape = validate_mesh(mesh)
    plat = _mesh_platform(mesh)
    key = (plat,) + shape
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            return entry[0]
        h = _make_ctx(mesh)
        _KEYS[("world",) + key] = loader.bind_context(
            plat, _ctx_config("world", shape, plat), h)
        _CACHE[key] = (h, plat)
        return h


#: ("world" | "subrow", platform, p, q) -> the context's FFI ``ctx_key``.
_KEYS: dict = {}


def _ctx_config(kind: str, key, platform: str) -> str:
    """The canonical configuration string of a SLATE context (ScaLAPACK shares it)."""
    p, q = key
    return f"lorrax-ctx/v1|slate|{kind}|{p}x{q}|{platform}"


def context_key(mesh: Mesh) -> int:
    """The FFI ``ctx_key`` attribute of this mesh's SLATE context (see
    ``distrib_la.loader.context_key``): a pure function of the configuration."""
    get_or_init_context(mesh)
    return _KEYS[("world", _mesh_platform(mesh)) + validate_mesh(mesh)]


def _subrow_context_key(mesh: Mesh) -> int:
    _get_or_init_subrow_context(mesh)
    return _KEYS[("subrow", _mesh_platform(mesh)) + validate_mesh(mesh)]


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
    shape = validate_mesh(mesh)
    plat = _mesh_platform(mesh)
    key = (plat,) + shape
    with _LOCK:
        entry = _SUBROW_CACHE.get(key)
        if entry is not None:
            return entry[0]
        h = _make_subrow_ctx(mesh)
        _KEYS[("subrow",) + key] = loader.bind_context(
            plat, _ctx_config("subrow", shape, plat), h)
        _SUBROW_CACHE[key] = (h, plat)
        return h


def _atexit_teardown() -> None:
    for kind, cache in (("world", _CACHE), ("subrow", _SUBROW_CACHE)):
        for k, (h, plat) in list(cache.items()):
            try:
                loader.unbind_context(plat, _KEYS.pop((kind,) + k, 0))
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

Returns a :class:`SlateLowerL` opaque handle; :meth:`SlateLowerL.to_jax_lower`
gives a conventional row-major lower-triangular L.

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
    ctx_key = context_key(mesh)

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
    attrs = dict(n=n, nb=nb, ctx_key=int(ctx_key))

    @partial(shard_map, mesh=mesh,
             in_specs=P("x", "y"), out_specs=P("y", "x"),
             check_vma=False)
    def _potrf(local_A):
        local_A_T = jnp.transpose(local_A, (1, 0))
        return jax.ffi.ffi_call(_POTRF_FFI_TARGET, L_local_T)(local_A_T, **attrs)

    L_raw_T = _potrf(A)
    return SlateLowerL(raw=L_raw_T, mesh=mesh, n=n, nb=nb)


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

Block size: block-cyclic by relabeling (:mod:`distrib_la._cyclic`).  SLATE
reads the operand's own ``(n/p, n/p)`` buffers with a small tile ``nb``
(``nb | n/p``), i.e. ``Pi A Pi^T``; the eigenvector rows come back in ``A``'s
order and only the eigenvalue index (Q's columns, over ``'y'``) is permuted
back, by one ``all_to_all``.  The tile matters because heev's second stage
(``hb2st``, band -> tridiagonal) runs on rank 0's HOST at band width ``nb``
(cost ~ n^2 nb) after a gather of the band to rank 0: at one tile per rank
(``nb = n/p``) that stage dominated (f64 n 8192, 2x2: 146 s vs 2.8 s for a
single-GCD native eigh, SLATE_HIP_REPORT).  :func:`eigh_layout` picks the
tile; :data:`_EIGH_BLOCK_MIN`/:data:`_EIGH_BLOCK_MAX` were measured on MI250X
(SLATE_PERF_REPORT).

Several tiles per rank means many concurrent SLATE tasks calling host
BLAS/LAPACK: the SLATE must link a host BLAS that is safe for concurrent
callers (on Frontier the sequential LOCKING OpenBLAS of
``config/frontier/build_openblas_seq.sh``).  Against the sequential LibSci
heev at nb < n/p returns wrong eigenpairs with no error.
"""

_EIGH_FFI_TARGET = "lorrax_slate_eigh"

#: The heev tile range.  Measured (P4 2x2 MI250X, steady s, SLATE_PERF_REPORT):
#:   f64  n 8192: nb 128 10.9, 256 11.4, n/p 146     c128 n 8192: 128 17.4, 256 22.5
#:   f64  n 4096: nb 128  3.7, 256  3.2              c128 n 4096: 128  5.1, 256  5.3
#: nb 64 put unmtr_hb2st at 6 s (n 8192 f64; 1.7x the nb-128 wall).
_EIGH_BLOCK_MAX = 128
_EIGH_BLOCK_MIN = 64

_JIT_CACHE: dict = {}


def eigh_layout(n: int, p: int) -> Tuple[int, int]:
    """``(N, nb)``: the side an ``n x n`` heev solves at on a ``p x p`` grid, and its tile.

    ``(n, nb)`` with ``nb`` the largest divisor of ``n/p`` at most
    :data:`_EIGH_BLOCK_MAX`, when that is at least :data:`_EIGH_BLOCK_MIN` or
    ``n/p`` is at most :data:`_EIGH_BLOCK_MAX` (then ``nb = n/p``, one tile per
    rank).  Otherwise each rank's tile is padded to the smallest edge with a
    divisor in the range (at most ``_EIGH_BLOCK_MIN - 1`` rows per rank); the
    padded rows carry sentinel eigenvalues below the spectrum
    (:func:`distrib_la._cyclic.pad_with_sentinels`), dropped on return.
    """
    local = n // p
    block = largest_divisor(local, _EIGH_BLOCK_MAX)
    if local <= _EIGH_BLOCK_MAX or block >= _EIGH_BLOCK_MIN:
        return n, block
    padded = next(m for m in range(local + 1, local + _EIGH_BLOCK_MIN + 1)
                  if largest_divisor(m, _EIGH_BLOCK_MAX) >= _EIGH_BLOCK_MIN)
    return p * padded, largest_divisor(padded, _EIGH_BLOCK_MAX)


def distributed_eigh(
    A: jax.Array,
    *,
    mesh: Mesh,
    compute_evecs: bool = True,
    block_size: int | None = None,
    side: int | None = None,
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
        SLATE's tile ``nb``, a divisor of the solve tile ``N/p``; default
        :func:`eigh_layout`'s.  Any divisor is layout-safe (relabeling, see
        the section note); ``N/p`` is the former one-tile-per-rank layout.
    side
        The solve side ``N >= n``, a multiple of ``p``; default
        :func:`eigh_layout`'s (``side=n`` is the unpadded layout, for benches).

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
    ctx_key = context_key(mesh)

    if side is None:
        side, default = eigh_layout(n, p)
    else:
        side = int(side)
        default = largest_divisor(side // p, _EIGH_BLOCK_MAX)
    nb = default if block_size is None else int(block_size)
    if side < n or side % p or nb < 1 or (side // p) % nb:
        raise ValueError(
            f"distributed_eigh: tile {nb} does not divide the solve tile "
            f"{side // p} (side {side}, {p}x{q} mesh).")

    key = ("eigh", mesh_key(mesh), A.dtype, n, side, nb,
           bool(compute_evecs), int(ctx_key))
    jit_eigh = _JIT_CACHE.get(key)
    if jit_eigh is None:
        local = side // p
        # Real-part dtype for eigenvalues (float64 for both F64 and C128).
        w_dtype = jnp.float64 if A.dtype in (jnp.complex128, jnp.float64) \
            else jnp.float32
        W_local = jax.ShapeDtypeStruct((side,), w_dtype)            # replicated
        # Q comes back as SLATE col-major tiles = row-major bytes of the
        # per-rank block of Q^T; assembled at P('y','x') then locally
        # untransposed (square mesh: local shape is (N/q, N/p)).
        Q_local_T = jax.ShapeDtypeStruct((local, local), A.dtype)
        attrs = dict(n=side, nb=nb, ctx_key=int(ctx_key),
                     compute_evecs=bool(compute_evecs))

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=P("x", "y"),
                 out_specs=(P(), P("x", "y")),
                 check_vma=False)
        def _call(local_A):
            # Local transpose: row-major shard bytes → col-major block, the
            # layout SLATE reads (pure local op, pairs with the C++ comm
            # rank-remap).  Same convention as distributed_cholesky.
            local_A_T = jnp.transpose(local_A, (1, 0))
            W, Q_T = jax.ffi.ffi_call(
                _EIGH_FFI_TARGET, (W_local, Q_local_T))(local_A_T, **attrs)
            # Q_T's rows are Q's columns: the eigenvalue index, over 'y',
            # laid out block-cyclically at nb < N/p.  Put them in W's
            # ascending order, then untranspose (Q at P('x','y')).
            if compute_evecs and nb != local:
                Q_T = cyclic_to_block(Q_T, n=side, p=q, mb=nb, axis="y")
            return W, jnp.transpose(Q_T, (1, 0))

        face = NamedSharding(mesh, P("x", "y"))

        def _padded(a):
            whole = jax.lax.with_sharding_constraint(pad_with_sentinels(a, side), face)
            W, Q = _call(whole)
            # The sentinels are the side - n lowest eigenpairs; Q holds
            # columns, of which the first n rows are a's.
            W, Q = W[side - n:], Q[:n, side - n:]
            return W, jax.lax.with_sharding_constraint(Q, face)

        # No donation: plan.DONATES declares eigh as donating nothing.
        jit_eigh = jax.jit(_call if side == n else _padded)
        _JIT_CACHE[key] = jit_eigh

    return jit_eigh(A)


# =========================================================================
#  solve_lu  (slate::getrf PartialPiv + slate::getrs, Target::Devices)
# =========================================================================
"""``batched_distributed_solve_lu`` and the split ``getrf`` / ``getrs`` pair.

The SLATE twin of the ScaLAPACK and cuSOLVERMp LU families, with the same
call contract, so :mod:`distrib_la.plan` drives the three the same way::

    A : (Nq, N, N)     P(None, 'x', 'y')   (donated -- factored in place)
    B : (Nq, N, NRHS)  P(None, 'x', 'y')   (donated -- solved in place)
    X : same shape/sharding as B

Each matrix of the stack is spread over the WHOLE mesh (the world SLATE
context, one tile per rank: ``nb = N / max(p, q)``), and the handler walks
the stack.  Geometry is SLATE's single-matrix class (:func:`validate_tile_layout`):
square or ``p x 1`` meshes; ``1 x q`` trips SLATE's stride assert.

The split pair returns the pivots as an opaque int64 buffer, ``(Nq, P*2N)``
at ``P(None, ('x','y'))``: every rank's own copy of SLATE's replicated
``Pivots`` serialised as (tile index, element offset) pairs.  Feed ``LU`` and
``piv`` back to :func:`batched_distributed_getrs` verbatim, on the same mesh.
The fused entry runs the same getrf + getrs core, so getrf-once +
getrs-per-chunk reproduces it to the bit.

Built on the ROCM leg only (``distrib_la.resolve``: ``('solve_lu','slate')``
is a ROCM row; CUDA's distributed LU is cuSOLVERMp, the host's ScaLAPACK).
"""

_SOLVE_LU_TARGET = "lorrax_slate_batched_solve_lu_factors"
_GETRF_TARGET = "lorrax_slate_batched_getrf"
_GETRS_TARGET = "lorrax_slate_batched_getrs"

#: jit(shard_map) cache per signature (eager shard_map re-traces per call).
_LU_JIT_CACHE: dict = {}


def _lu_geometry(A, mesh: Mesh, *, what: str):
    """The LU family's guard ladder: shape, registered mesh, SLATE tile layout.
    Returns ``(p, q, nq, n, nb)``."""
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"slate {what}: expected A of shape (Nq, N, N); got {A.shape}")
    p, q = validate_mesh(mesh)
    nq, n = int(A.shape[0]), int(A.shape[1])
    if n % p or n % q:
        raise ValueError(
            f"slate {what}: N={n} must be divisible by both mesh axes ({p}, {q}).")
    nb = n // max(p, q)
    validate_tile_layout(n, nb, p, q, what=f"slate {what}")
    return p, q, nq, n, nb


def _lu_rhs(B, nq: int, n: int, q: int, dtype, *, what: str) -> int:
    if B.ndim != 3 or int(B.shape[0]) != nq or int(B.shape[1]) != n:
        raise ValueError(
            f"slate {what}: B must be (Nq, N, NRHS) = ({nq}, {n}, *); got {B.shape}")
    if B.dtype != dtype:
        raise ValueError(f"slate {what}: B.dtype {B.dtype} != {dtype}")
    nrhs = int(B.shape[2])
    if nrhs % q:
        raise ValueError(f"slate {what}: NRHS={nrhs} must be divisible by q={q}.")
    return nrhs


def batched_distributed_solve_lu(A: jax.Array, B: jax.Array, *, mesh: Mesh) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` by per-q ``slate::getrf`` + ``slate::getrs``.

    See the section docstring for the contract.  ``A`` and ``B`` are donated
    (the factors overwrite A, X overwrites B); the pivots stay internal.
    """
    p, q, nq, n, nb = _lu_geometry(A, mesh, what="solve_lu")
    nrhs = _lu_rhs(B, nq, n, q, A.dtype, what="solve_lu")
    ensure_registered(mesh)
    ctx_key = int(context_key(mesh))
    nb_b = nrhs // q
    key = ("slate_lu", mesh_key(mesh), A.dtype, nq, n, nrhs, nb, ctx_key)
    fn = _LU_JIT_CACHE.get(key)
    if fn is None:
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // q, n // p), A.dtype)
        LU_local_T = jax.ShapeDtypeStruct((nq, n // q, n // p), A.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs, nb=nb, nb_b=nb_b, ctx_key=ctx_key)

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"), check_vma=False)
        def _solve(local_A, local_B):
            # Row-major shard -> col-major local block (the wrappers' convention).
            X_T, _ = jax.ffi.ffi_call(
                _SOLVE_LU_TARGET, (X_local_T, LU_local_T),
                input_output_aliases={0: 1, 1: 0},
            )(jnp.transpose(local_A, (0, 2, 1)), jnp.transpose(local_B, (0, 2, 1)), **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        fn = jax.jit(_solve, donate_argnums=(0, 1))
        _LU_JIT_CACHE[key] = fn
    return fn(A, B)


def batched_distributed_getrf(A: jax.Array, *, mesh: Mesh) -> tuple[jax.Array, jax.Array]:
    """Factor ``A[q]`` once by ``slate::getrf`` (PartialPiv).

    Returns ``(LU, piv)``: ``LU`` at ``P(None,'x','y')`` (SLATE's factors in
    this layout on this mesh) and ``piv`` int64 ``(Nq, P*2N)`` at
    ``P(None, ('x','y'))``.  Both are opaque: feed them to
    :func:`batched_distributed_getrs` verbatim.  ``A`` is donated.
    """
    p, q, nq, n, nb = _lu_geometry(A, mesh, what="getrf")
    ensure_registered(mesh)
    ctx_key = int(context_key(mesh))
    piv_len = 2 * n
    key = ("slate_getrf", mesh_key(mesh), A.dtype, nq, n, nb, ctx_key)
    fn = _LU_JIT_CACHE.get(key)
    if fn is None:
        LU_local_T = jax.ShapeDtypeStruct((nq, n // q, n // p), A.dtype)
        piv_local = jax.ShapeDtypeStruct((nq, piv_len), jnp.int64)
        attrs = dict(nq=nq, n=n, nb=nb, piv_len=piv_len, ctx_key=ctx_key)

        @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),),
                 out_specs=(P(None, "x", "y"), P(None, ("x", "y"))), check_vma=False)
        def _getrf(local_A):
            LU_T, piv = jax.ffi.ffi_call(
                _GETRF_TARGET, (LU_local_T, piv_local),
                input_output_aliases={0: 0},
            )(jnp.transpose(local_A, (0, 2, 1)), **attrs)
            return jnp.transpose(LU_T, (0, 2, 1)), piv

        fn = jax.jit(_getrf, donate_argnums=(0,))
        _LU_JIT_CACHE[key] = fn
    return fn(A)


def batched_distributed_getrs(LU: jax.Array, piv: jax.Array, B: jax.Array, *,
                              mesh: Mesh) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` from :func:`batched_distributed_getrf`'s
    ``(LU, piv)`` by ``slate::getrs``.  ``B`` is donated; X has B's
    shape and sharding."""
    p, q, nq, n, nb = _lu_geometry(LU, mesh, what="getrs")
    nrhs = _lu_rhs(B, nq, n, q, LU.dtype, what="getrs")
    piv_len = 2 * n
    want = (nq, p * q * piv_len)
    if tuple(piv.shape) != want or piv.dtype != jnp.int64:
        raise ValueError(
            f"slate getrs: piv must be int64 {want} as getrf returned it; "
            f"got {tuple(piv.shape)} {piv.dtype}")
    ensure_registered(mesh)
    ctx_key = int(context_key(mesh))
    nb_b = nrhs // q
    key = ("slate_getrs", mesh_key(mesh), LU.dtype, nq, n, nrhs, nb, ctx_key)
    fn = _LU_JIT_CACHE.get(key)
    if fn is None:
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // q, n // p), B.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs, nb=nb, nb_b=nb_b, piv_len=piv_len,
                     ctx_key=ctx_key)

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, ("x", "y")), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"), check_vma=False)
        def _getrs(local_LU, local_piv, local_B):
            X_T = jax.ffi.ffi_call(
                _GETRS_TARGET, X_local_T, input_output_aliases={2: 0},
            )(jnp.transpose(local_LU, (0, 2, 1)), local_piv,
              jnp.transpose(local_B, (0, 2, 1)), **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        fn = jax.jit(_getrs, donate_argnums=(2,))
        _LU_JIT_CACHE[key] = fn
    return fn(LU, piv, B)

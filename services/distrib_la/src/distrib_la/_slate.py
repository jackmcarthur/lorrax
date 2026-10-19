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
from jax.sharding import Mesh, PartitionSpec as P

from distrib_la import loader
from lxkit import native_provider as _native
from distrib_la._cyclic import cyclic_to_block, largest_divisor
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
    "batched_distributed_getrs", "lu_layout",
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


def ensure_registered(mesh: Mesh, target: str | None = None) -> None:
    """Load + register the FFI library for the MESH's device platform.

    ``jax.ffi.ffi_call`` resolves the target registered for the lowering
    platform, so the library that matters is the one matching the mesh's
    devices — not necessarily the default backend (e.g. slate ops on a
    CPU-device mesh inside a GPU-backend process).  Idempotent.

    With ``target``, refuse by name when that handler would run against a
    SLATE its leg was not built against: the host leg's SLATE handlers in a
    device process, where the device leg's SLATE answers those SONAMEs
    (``lxkit.native_provider.refuse_if_borrowed``).
    """
    platform = _mesh_platform(mesh)
    loader.get_lib(platform)
    if target is not None:
        _native.refuse_if_borrowed(platform, target, unusable_cls=loader.LibraryUnusable)


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


# -------------------------------------------------------------------------
#  Device memory: SLATE's tile pools inside XLA's arena, per rank
# -------------------------------------------------------------------------
# SLATE's Target::Devices drivers keep their device tiles in a per-matrix pool
# that, left alone, allocates with a raw device_malloc OUTSIDE the XLA arena.  On
# ROCM the BFC arena is preallocated (runtime.pool_fraction, 60.18 of 68.70 GB per
# MI250X GCD), so SLATE got ~7 GB: eigh aborted every rank above n ~ 20k c128 at P4
# (blas::Error "out of memory, in function device_malloc", thrown from an OpenMP
# task) and potrf hung at 51200 c128.  The device handlers seed each matrix's pool
# with arena bytes at the measured lower envelope of the pool's high-water, from
# THIS rank's own tiles, so no rank's device peak rises above unseeded SLATE's.
# Where the bytes come from depends on whether the ranks' seeds are equal:
#   * eigh: a u8 workspace OUTPUT sized here (eigh_workspace_bytes).  On every
#     square grid the per-rank A / Z / Z1d counts are equal, and a workspace output
#     is live only during the call, so XLA's buffer assignment overlaps it with the
#     program's temporaries before and after the call.  Scratch-allocator bytes sit
#     ON TOP of the whole temp allocation: measured +0.15 .. +1.06 GB per rank at
#     n = 8192 ... 20480 against unseeded lufix, where the workspace output is
#     -0.03 .. -1.17 GB (DISTRIB_LA_SCALE_REPORT addendum, round 2).
#   * potrf, LU: xla::ffi::ScratchAllocator inside the handler (pool_seed.h
#     ScratchSeed), each rank its own bytes; a rank whose seed is 0 allocates
#     nothing, and the Python calls are main's (plus potrf's ``seed_min_bytes``).
#     potrf's reservation is one block per local LOWER tile -- 0 on a rank that owns
#     none -- so a uniform buffer would raise that rank's peak by the others' seed.
#     LU's measured arena bytes are the same either way.
# The rules live in the handlers (eigh_ffi.cc a_seed_tiles, batched_solve_lu_ffi.cc
# seed_want, potrf_ffi.cc); eigh_workspace_bytes below is eigh's rule, max over ranks.
#: The outside-pool allowance the memory model grants distributed linalg (the value of
#: ``runtime.DISTRIBUTED_LINALG_OUTSIDE_BYTES``; distrib_la cannot import ``runtime`` --
#: layering -- so tests/test_slate_pool_rules.py pins the two equal).  A rank's potrf
#: reservation is seeded only past it: within it unseeded SLATE fits outside the arena.
_SLATE_OUTSIDE_ALLOWANCE_BYTES = 3e9


def _device_leg(mesh: Mesh) -> bool:
    """The device handlers (ROCM, CUDA) take eigh's workspace and potrf's
    ``seed_min_bytes``; the host leg's do not."""
    return _mesh_platform(mesh) in ("ROCM", "CUDA")


_POOL_ALIGN = 256   # pool_seed.h PoolSeed's carve alignment


def _rank_tiles(mt: int, nt: int, p: int, q: int, x: int, y: int, *,
                lower: bool = False) -> int:
    """Tiles of grid rank ``(x, y)`` in a 2-D block-cyclic ``p x q`` layout (lower
    triangle only for a Hermitian matrix)."""
    rows = [i for i in range(mt) if i % p == x]
    cols = [j for j in range(nt) if j % q == y]
    if not lower:
        return len(rows) * len(cols)
    return sum(1 for i in rows for j in cols if i >= j)


def eigh_workspace_bytes(side: int, nb: int, p: int, q: int, itemsize: int,
                         compute_evecs: bool = True) -> int:
    """The seeded heev driver's workspace: the largest per-rank seed over the grid.

    Rank ``(x, y)`` seeds A with ``max(lower, full - mt/2)`` of its own tiles (he2hb's
    measured high-water, min over ranks), Z with its local tiles and Z1d with its
    columns of the ``1 x P`` column-cyclic grid -- the same rules as eigh_ffi.cc,
    carved at :data:`_POOL_ALIGN`.  Equal on every rank of a square grid.
    """
    mt = nt = side // nb
    tile = nb * nb * itemsize
    ranks = p * q
    best = 0
    for x in range(p):
        for y in range(q):
            r = x * q + y
            pools = [max(_rank_tiles(mt, nt, p, q, x, y, lower=True),
                         _rank_tiles(mt, nt, p, q, x, y) - mt // 2)]
            if compute_evecs:
                z1d_cols = (nt - 1 - r) // ranks + 1 if r < nt else 0
                pools += [_rank_tiles(mt, nt, p, q, x, y), mt * z1d_cols]
            total = 0
            for blocks in pools:
                total = -(-total // _POOL_ALIGN) * _POOL_ALIGN + blocks * tile
            best = max(best, total)
    return max(best, _POOL_ALIGN)


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

    ensure_registered(mesh, _POTRF_FFI_TARGET)
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
    if _device_leg(mesh):
        # The device handler seeds a rank's SLATE pool (from XLA's scratch, that rank
        # only) only when its reservation exceeds this allowance (cpp/slate/potrf_ffi.cc).
        attrs["seed_min_bytes"] = int(_SLATE_OUTSIDE_ALLOWANCE_BYTES)

    @partial(shard_map, mesh=mesh,
             in_specs=P("x", "y"), out_specs=P("y", "x"),
             check_vma=False)
    def _potrf(local_A):
        local_A_T = jnp.transpose(local_A, (1, 0))
        return jax.ffi.ffi_call(_POTRF_FFI_TARGET, L_local_T)(local_A_T, **attrs)

    L_raw_T = _potrf(A)
    return SlateLowerL(raw=L_raw_T, mesh=mesh, n=n, nb=nb)


_TRSM_FFI_TARGET = "lorrax_slate_trsm"


def distributed_trsm(L: "SlateLowerL", B: jax.Array, *, mesh: Mesh, op: str = "N") -> jax.Array:
    """Solve ``L X = B`` (``op='N'``) or ``L^H X = B`` (``op='C'``) for a :class:`SlateLowerL`
    factor; ``B`` and ``X`` are 2-D at ``P('x','y')`` (restored from 8deacaeae^ for the
    Cholesky solve, the handle path only)."""
    if op not in ("N", "C"):
        raise ValueError(f"distributed_trsm: op must be 'N' or 'C'; got {op!r}")
    p, q = validate_mesh(mesh)
    n, m = L.n, int(B.shape[1])
    if B.ndim != 2 or B.shape[0] != n or B.shape[0] % p or m % q or L.raw.dtype != B.dtype:
        raise ValueError(f"distributed_trsm: B {B.shape} {B.dtype} against L n={n} {L.raw.dtype} on {p}x{q}")
    ensure_registered(mesh, _TRSM_FFI_TARGET)
    attrs = dict(n=n, m=m, nb=L.nb, side=0, uplo=0, op={"N": 0, "C": 2}[op], diag=0,
                 alpha_re=1.0, alpha_im=0.0, ctx_key=int(context_key(mesh)))
    X_local_T = jax.ShapeDtypeStruct((m // q, n // p), B.dtype)

    @partial(shard_map, mesh=mesh, in_specs=(P("y", "x"), P("x", "y")), out_specs=P("x", "y"),
             check_vma=False)
    def _trsm(local_L, local_B):
        X_T = jax.ffi.ffi_call(_TRSM_FFI_TARGET, X_local_T)(local_L, jnp.transpose(local_B, (1, 0)), **attrs)
        return jnp.transpose(X_T, (1, 0))
    return _trsm(L.raw, B)


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
    divisor in the range (at most ``_EIGH_BLOCK_MIN - 1`` rows per rank): the
    plan zero-pads to that side and deflates the pads.
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
) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition via SLATE's ``heev``.

    Parameters
    ----------
    A
        Square ``(n, n)`` array, dtype ``float64`` (symmetric) or
        ``complex128`` (Hermitian), sharded ``P('x','y')`` on ``mesh``;
        ``n`` must be :func:`eigh_layout`'s side (the plan zero-pads to it).
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

    ensure_registered(mesh, _EIGH_FFI_TARGET)
    ctx_key = context_key(mesh)

    side, default = eigh_layout(n, p)
    if side != n:
        raise ValueError(f"distributed_eigh: n={n} is not a solve side on a {p}x{q} mesh; "
                         f"pad it to {side} first (distrib_la.plan pads with zero rows)")
    nb = default if block_size is None else int(block_size)
    if nb < 1 or (side // p) % nb:
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
        results = (W_local, Q_local_T)
        if _device_leg(mesh):
            results = results + (jax.ShapeDtypeStruct((eigh_workspace_bytes(
                side, nb, p, q, A.dtype.itemsize, bool(compute_evecs)),), jnp.uint8),)

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
                _EIGH_FFI_TARGET, results)(local_A_T, **attrs)[:2]
            # Q_T's rows are Q's columns: the eigenvalue index, over 'y',
            # laid out block-cyclically at nb < N/p.  Put them in W's
            # ascending order, then untranspose (Q at P('x','y')).
            if compute_evecs and nb != local:
                Q_T = cyclic_to_block(Q_T, n=side, p=q, mb=nb, axis="y")
            return W, jnp.transpose(Q_T, (1, 0))

        # No donation: plan.DONATES declares eigh as donating nothing.
        jit_eigh = jax.jit(_call)
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
context), and the handler walks the stack.  Block-cyclic BY RELABELING
(:mod:`distrib_la._cyclic`), as the eigh: the handler reads each rank's own
``(N/p, N/q)`` buffer with a tile ``nb`` dividing both local edges
(:func:`lu_layout`), so SLATE factors ``Pi A Pi'^T`` and solves for ``Pi B``.
On a square mesh ``Pi' = Pi`` and X comes back in A's row order with no data
movement; on a ``p x 1`` mesh X's rows come back block-cyclic over ``'x'`` and
one ``all_to_all`` (:func:`distrib_la._cyclic.cyclic_to_block`) orders them.
Geometry: square or ``p x 1`` meshes; ``1 x q`` trips SLATE's stride assert.
Several tiles per rank means concurrent SLATE tasks in host BLAS: the SLATE must
link the locking sequential OpenBLAS (see the eigh section; verifier GATE 3d).

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

#: The getrf tile range (P4 2x2 MI250X, pool log ``distla_lunb``, DISTLA report).
_LU_BLOCK_MAX = 256
_LU_BLOCK_MIN = 64


def lu_layout(n: int, p: int, q: int) -> int:
    """The getrf/getrs tile ``nb`` for an ``n x n`` matrix on a ``p x q`` mesh.

    The largest divisor of both local edges (``gcd(n/p, n/q)``) at most
    :data:`_LU_BLOCK_MAX`, when it is at least :data:`_LU_BLOCK_MIN` or the edge is
    at most :data:`_LU_BLOCK_MAX`; otherwise one tile per rank (``n/max(p,q)``,
    the former layout: correct, with the host-panel cost it had).
    """
    import math
    local = math.gcd(n // p, n // q)
    block = largest_divisor(local, _LU_BLOCK_MAX)
    if local <= _LU_BLOCK_MAX or block >= _LU_BLOCK_MIN:
        return block
    return n // max(p, q)


def _lu_geometry(A, mesh: Mesh, *, what: str, block_size: int | None = None):
    """The LU family's guard ladder: shape, registered mesh, mesh class, tile.
    Returns ``(p, q, nq, n, nb)``.  The mesh class mirrors
    :func:`validate_tile_layout`'s (square or ``p x 1``); the tile rule is the
    relabeling one (``nb`` divides ``n/p`` and ``n/q``), not one tile per rank."""
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"slate {what}: expected A of shape (Nq, N, N); got {A.shape}")
    p, q = validate_mesh(mesh)
    nq, n = int(A.shape[0]), int(A.shape[1])
    if n % p or n % q:
        raise ValueError(
            f"slate {what}: N={n} must be divisible by both mesh axes ({p}, {q}).")
    if p > 1 and q > 1 and p != q:
        raise ValueError(f"slate {what}: mesh {p}x{q} unsupported (square or Nx1 only).")
    if p == 1 and q > 1:
        raise ValueError(
            f"slate {what}: 1x{q} meshes hit a SLATE stride assert "
            f"(internal_batch.hh:290); use the {q}x1 mesh.")
    nb = lu_layout(n, p, q) if block_size is None else int(block_size)
    if nb < 1 or (n // p) % nb or (n // q) % nb:
        raise ValueError(
            f"slate {what}: tile {nb} must divide both local edges ({n // p}, {n // q}).")
    return p, q, nq, n, nb


def _rows_in_order(X, *, n: int, p: int, q: int, nb: int):
    """Inside a shard_map: X's local rows ``(nq, n/p, c)`` in A's order.

    SLATE writes X's row ``i`` at its relabeled slot: on a square mesh that is
    row ``i`` itself (``Pi' = Pi``), on a ``p x 1`` mesh row ``i`` sits
    block-cyclically over ``'x'`` -- one :func:`cyclic_to_block`.
    """
    if p == 1 or q > 1 or nb == n // p:
        return X
    nq, rows, cols = X.shape
    flat = jnp.transpose(X, (1, 0, 2)).reshape(rows, nq * cols)
    flat = cyclic_to_block(flat, n=n, p=p, mb=nb, axis="x")
    return jnp.transpose(flat.reshape(rows, nq, cols), (1, 0, 2))


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


def batched_distributed_solve_lu(A: jax.Array, B: jax.Array, *, mesh: Mesh,
                                 block_size: int | None = None) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` by per-q ``slate::getrf`` + ``slate::getrs``.

    See the section docstring for the contract.  ``A`` and ``B`` are donated
    (the factors overwrite A, X overwrites B); the pivots stay internal.
    ``block_size`` overrides :func:`lu_layout` (benches; any divisor of both
    local edges is layout-safe).
    """
    p, q, nq, n, nb = _lu_geometry(A, mesh, what="solve_lu", block_size=block_size)
    nrhs = _lu_rhs(B, nq, n, q, A.dtype, what="solve_lu")
    ensure_registered(mesh, _SOLVE_LU_TARGET)
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
            return _rows_in_order(jnp.transpose(X_T, (0, 2, 1)), n=n, p=p, q=q, nb=nb)

        fn = jax.jit(_solve, donate_argnums=(0, 1))
        _LU_JIT_CACHE[key] = fn
    return fn(A, B)


def batched_distributed_getrf(A: jax.Array, *, mesh: Mesh,
                              block_size: int | None = None) -> tuple[jax.Array, jax.Array]:
    """Factor ``A[q]`` once by ``slate::getrf`` (PartialPiv).

    Returns ``(LU, piv)``: ``LU`` at ``P(None,'x','y')`` (SLATE's factors in
    this layout on this mesh) and ``piv`` int64 ``(Nq, P*2N)`` at
    ``P(None, ('x','y'))``.  Both are opaque: feed them to
    :func:`batched_distributed_getrs` verbatim (same ``block_size``).  ``A`` is donated.
    """
    p, q, nq, n, nb = _lu_geometry(A, mesh, what="getrf", block_size=block_size)
    ensure_registered(mesh, _GETRF_TARGET)
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
                              mesh: Mesh, block_size: int | None = None) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` from :func:`batched_distributed_getrf`'s
    ``(LU, piv)`` by ``slate::getrs``.  ``B`` is donated; X has B's
    shape and sharding."""
    p, q, nq, n, nb = _lu_geometry(LU, mesh, what="getrs", block_size=block_size)
    nrhs = _lu_rhs(B, nq, n, q, LU.dtype, what="getrs")
    piv_len = 2 * n
    want = (nq, p * q * piv_len)
    if tuple(piv.shape) != want or piv.dtype != jnp.int64:
        raise ValueError(
            f"slate getrs: piv must be int64 {want} as getrf returned it; "
            f"got {tuple(piv.shape)} {piv.dtype}")
    ensure_registered(mesh, _GETRS_TARGET)
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
            return _rows_in_order(jnp.transpose(X_T, (0, 2, 1)), n=n, p=p, q=q, nb=nb)

        fn = jax.jit(_getrs, donate_argnums=(2,))
        _LU_JIT_CACHE[key] = fn
    return fn(LU, piv, B)

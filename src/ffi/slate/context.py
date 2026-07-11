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
See ``cpp/context.cc`` for the formula and ``ffi.slate.cholesky`` for
the local-transpose pairing on the Python side.

Restrictions
------------
* Mesh must have axis names ``('x', 'y')``.  If you have a 3-D mesh
  (e.g. for batched ops with a ``'batch'`` axis), build a 2-D
  sub-mesh out of its (x, y) axes for the SLATE call — the FFI
  doesn't support sub-mesh selection yet.
* For 4 GPUs total, ``p * q == jax.process_count()``.  Sub-mesh
  partial-world support is a planned extension; see ``README.md``.
* ``heev`` further requires ``p == q``; ``potrf`` and ``trsm`` don't.
"""
from __future__ import annotations

import atexit
import threading
from typing import Dict, Tuple

import jax
from jax.sharding import Mesh

from ..common import ffi_loader

__all__ = [
    "ensure_registered",
    "get_or_init_context",
    "get_or_init_subrow_context",
    "validate_mesh",
    "validate_tile_layout",
]

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
    ffi_loader.get_lib(_mesh_platform(mesh))


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
            f"Sub-mesh / partial-world calls aren't supported yet — see "
            f"src/ffi/slate/README.md.")
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
    README's nbatch=8/n=128 1x4 repro) and remains an accepted risk
    documented in README.md.
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
    rank  = int(jax.process_index())
    world = int(jax.process_count())
    p, q  = validate_mesh(mesh)
    return int(ffi_loader.create_slate_context(
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
    rank  = int(jax.process_index())
    world = int(jax.process_count())
    Px, Py = validate_mesh(mesh)
    return int(ffi_loader.create_slate_subrow_context(
        rank=rank, world_size=world, Px=Px, Py=Py,
        platform=_mesh_platform(mesh)))


def get_or_init_subrow_context(mesh: Mesh) -> int:
    """Return the opaque int handle for a per-X-row Y-axis sub-comm SLATE ctx.

    The sub-comm is ``MPI_COMM_WORLD`` split by x-coordinate: one comm of
    size ``Py`` per X-row.  Thread-safe.  First call collectively splits
    MPI_COMM_WORLD.  Intended for batched ops where each X-row handles an
    independent slice of a ``(Nbatch, N, N)`` input distributed along
    ``'x'`` (batch) and ``'y'`` (inner matrix).
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
    for _, (h, plat) in list(_CACHE.items()):
        try:
            ffi_loader.destroy_slate_context(int(h), platform=plat)
        except Exception:
            pass
    _CACHE.clear()
    for _, (h, plat) in list(_SUBROW_CACHE.items()):
        try:
            ffi_loader.destroy_slate_context(int(h), platform=plat)
        except Exception:
            pass
    _SUBROW_CACHE.clear()


atexit.register(_atexit_teardown)

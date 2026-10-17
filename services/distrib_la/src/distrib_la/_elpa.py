"""``distributed_eigh`` on ELPA -- the ROCM leg's distributed Hermitian eigensolver.

ELPA 2026.02.001 on the AMD GPU (``config/frontier/build_elpa_hip.sh``, bound by
``cpp/elpa/eigh_ffi.cc``).  The ONLY solver is ELPA's ONE-stage path: GPU
tridiagonalization, GPU divide-and-conquer, GPU back-transformation.  Measured on MI250X
(ELPA_ROCM_REPORT.md), P=4 (2x2), cold wall, sampled residual/orthogonality <= 1.9e-14:

    n       complex128: 1-stage  2-stage  SLATE-HIP     float64: 1-stage  2-stage  SLATE-HIP
    20480               23.1     71.5     121.8                  15.3     34.5     84.4
    30720               56.5     186.6    aborts                 33.7     88.2     213.6

ELPA's two-stage path is NOT reachable: it is slower on this hardware at every size measured
and returns wrong eigenvectors (sampled residual and orthogonality ~0.1) on the 4x4 grid
at n >= 47104 (ELPA 2026.02.001; ELPA_TWO_STAGE_UPSTREAM_REPORT.md).  The handler pins
ELPA_SOLVER_1STAGE and refuses ``ELPA_FORCE_*`` environment overrides.

The operand contract is SLATE's (:func:`distrib_la._slate.distributed_eigh`): each rank's
``(N/p, N/p)`` block, locally transposed to column-major bytes, read as ELPA's block-cyclic
local array at tile ``nb | N/p`` on the square ``p x p`` grid, so ELPA solves
``Pi A Pi^T``; the eigenvector rows come back in ``A``'s order and only Q's columns are
permuted back (:mod:`distrib_la._cyclic`).  The handler shares the SLATE family's per-mesh
MPI context, as ScaLAPACK does.

Tile: ELPA's GPU kernels need ``nb`` a power of two (any other value silently moves work
to the host).  1-stage: nb 64 and 128 measured equal (20480, both dtypes: 23.1 vs 23.0 s
complex128, 15.3 vs 15.5 s float64), so 64, which divides more local edges.  A local edge with
no such divisor is padded to the next multiple of the tile, with sentinel eigenvalues below
the spectrum (:func:`pad_with_sentinels`).

Device memory, all of it XLA's (inside the arena) during the solve:

* ELPA overwrites its input: the transposed operand (an XLA temporary, never the caller's
  ``A``) is aliased to a scratch output (``input_output_aliases``), so there is no copy;
* ELPA's own device buffers come from the workspace's pool through the patched library's
  ``lorrax_elpa_set_device_allocator``; the pool is the lower envelope of ELPA's measured
  live high-water (:func:`pool_bytes`); a larger need falls back to a device malloc outside
  the arena, counted (``LORRAX_ELPA_POOL_REPORT=1``);
* every rocBLAS handle ELPA created gets a user-owned workspace from the workspace's tail
  (``rocblas_set_workspace``; :func:`rocblas_bytes`), detached after the call, so rocBLAS
  and rocSOLVER (``stedc``) hold no device memory between calls.  A too-small rocBLAS
  workspace makes the call fail without computing: the patched ELPA counts every failed
  rocBLAS/rocSOLVER status and the handler refuses the solve (no silent result).
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la import _slate
from distrib_la._cyclic import cyclic_to_block, pad_with_sentinels
from distrib_la.resolve import mesh_key

__all__ = ["distributed_eigh", "eigh_layout", "workspace_bytes", "pool_bytes", "rocblas_bytes"]

_EIGH_FFI_TARGET = "lorrax_elpa_eigh"
_ALIGN = 256

#: Preferred tile per dtype (measured, module docstring) and the smallest tile used before
#: padding instead.
_TILE = {"complex128": 64, "float64": 64}
_TILE_MIN = 64

#: ELPA's live device high-water through the allocator hook, per rank:
#: ``POOL_PER_LOCAL2[dtype] * (N/p)^2 + POOL_PER_LOCAL * (N/p)`` bytes, the lower envelope
#: of the measured points, MI250X, P = 4 and 16 (ELPA_ROCM_REPORT.md, table "pool";
#: tests/test_distrib_la_elpa_gate.py carries them): 1.5 complex128 / 2 float64 local
#: matrices.
POOL_PER_LOCAL2 = {"complex128": 24.0, "float64": 16.0}
POOL_PER_LOCAL = 2048.0

#: rocBLAS workspace per ELPA rocBLAS handle: rocBLAS's managed high-water per handle,
#: MI250X, ROCm 7.14.1, ELPA 2026.02.001, 2x2, L = side/p = 10240 and 25600 (job 5700309,
#: ELPA_ROCM_REPORT.md addendum), fitted through both points and scaled by
#: :data:`_RB_MARGIN`; never below rocBLAS's 32 MiB default:
#:   solver handle 0 (rocsolver_dstedc in ELPA's solve_tridi, both dtypes): 0.454 / 2.657 GB
#:     -> 3.8707 L^2 + 4700 L (ELPA's sub-problems are L/2 at p = 2, smaller at p >= 4);
#:   BLAS handle 0 (back-transformation): c128 0.203 / 0.454, f64 0.118 / 0.244 GB
#:     -> 1024 * itemsize * L + 34 MB;
#:   every other handle: never above the 32 MiB default.
#: (Job 5700309 counted 15 rocBLAS handles in the process: ELPA's 14 plus the bench driver's.)
_RB_STEDC = (3.8707, 4700.0)
_RB_BACK = (1024.0, 34.0e6)
_RB_OTHER = 32 << 20
_RB_MARGIN = 1.25

_JIT_CACHE: dict = {}


def _pow2_floor(x: int) -> int:
    return 1 << (max(1, int(x)).bit_length() - 1)


def eigh_layout(n: int, p: int, dtype) -> Tuple[int, int]:
    """``(N, nb)``: the side ELPA solves an ``n x n`` eigh at on a ``p x p`` grid, and its tile.

    ``nb`` is the preferred power-of-two tile when it divides ``n/p``, else the largest
    power of two in ``[_TILE_MIN, preferred]`` that does; otherwise the local edge is padded
    to the next multiple of the preferred tile (fewer than ``nb`` rows per rank).  A local
    edge below the tile uses the largest power of two not above it (the one-device route
    serves such sizes by default).
    """
    pref = _TILE[jnp.dtype(dtype).name]
    local = n // p
    if local < pref:
        nb = _pow2_floor(local)
        return (n, nb) if local % nb == 0 else (p * (-(-local // nb) * nb), nb)
    nb = pref
    while nb >= _TILE_MIN:
        if local % nb == 0:
            return n, nb
        nb //= 2
    return p * (-(-local // pref) * pref), pref


def _align(b: float) -> int:
    return -(-int(b) // _ALIGN) * _ALIGN


def elpa_threads() -> int:
    """ELPA's ``omp_get_max_threads()`` in this process: OMP_NUM_THREADS's first value, else
    the CPU affinity count (ELPA creates that many BLAS and solver handles per object)."""
    import os
    v = (os.environ.get("OMP_NUM_THREADS") or "").split(",")[0].strip()
    if v.isdigit() and int(v) > 0:
        return int(v)
    return max(1, len(os.sched_getaffinity(0)))


def rocblas_regions(side: int, p: int, dtype) -> Tuple[int, int, int]:
    """``(back, stedc, other)``: rocBLAS workspace bytes for BLAS handle 0, solver handle 0
    and each other handle (module constants)."""
    local = side // p
    item = jnp.dtype(dtype).itemsize
    a, b = _RB_STEDC
    stedc = max(_RB_OTHER, _RB_MARGIN * (a * local * local + b * local))
    c, d = _RB_BACK
    back = max(_RB_OTHER, _RB_MARGIN * (c * item * local + d))
    return _align(back), _align(stedc), _align(_RB_OTHER)


def rocblas_bytes(side: int, p: int, dtype, threads: int | None = None) -> int:
    """The rocBLAS tail of the workspace: one region per ELPA rocBLAS handle, 2 T for
    T :func:`elpa_threads` (T BLAS + T solver handles per ELPA object; measured: 14 at T = 7,
    gate job 5700378)."""
    back, stedc, other = rocblas_regions(side, p, dtype)
    t = elpa_threads() if threads is None else int(threads)
    return back + stedc + (2 * t - 2) * other


def pool_bytes(side: int, p: int, dtype) -> int:
    """ELPA's device pool for a side-``side`` solve on a ``p x p`` grid (module constants)."""
    local = side // p
    b = POOL_PER_LOCAL2[jnp.dtype(dtype).name] * local * local + POOL_PER_LOCAL * local
    return -(-int(b) // _ALIGN) * _ALIGN


def workspace_bytes(side: int, p: int, dtype, threads: int | None = None) -> int:
    """The handler's workspace: ELPA's pool, then the rocBLAS regions."""
    return pool_bytes(side, p, dtype) + rocblas_bytes(side, p, dtype, threads)


def distributed_eigh(A: jax.Array, *, mesh: Mesh, compute_evecs: bool = True,
                     block_size: int | None = None,
                     side: int | None = None) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition on ELPA (ROCM leg).

    Same contract as :func:`distrib_la._slate.distributed_eigh`: ``A`` square, float64 or
    complex128, sharded ``P('x','y')`` on a square ``('x','y')`` mesh; returns ``W``
    ``(n,)`` ascending, replicated, and ``Q`` ``(n, n)`` with the eigenvectors as columns,
    ``P('x','y')``.  ``block_size`` must be a power of two dividing ``side/p``; ``side``
    defaults to :func:`eigh_layout`'s.  ELPA's one-stage solver only (module docstring).
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"elpa distributed_eigh: expected a square matrix, got {A.shape}")
    if A.dtype not in (jnp.complex128, jnp.float64):
        raise ValueError(f"elpa distributed_eigh: dtype {A.dtype} (supported: complex128, float64)")
    p, q = _slate.validate_mesh(mesh, require_square=True)
    n = int(A.shape[0])
    if n % p:
        raise ValueError(f"elpa distributed_eigh: n={n} must be divisible by mesh axis size {p}.")
    if side is None:
        side, nb = eigh_layout(n, p, A.dtype)
    else:
        side = int(side)
        nb = eigh_layout(side, p, A.dtype)[1]
    if block_size is not None:
        nb = int(block_size)
    local = side // p
    if side < n or side % p or nb < 1 or nb & (nb - 1) or local % nb:
        raise ValueError(f"elpa distributed_eigh: tile {nb} must be a power of two dividing "
                         f"the solve tile {local} (side {side}, {p}x{q} mesh).")
    _slate.ensure_registered(mesh, _EIGH_FFI_TARGET)
    ctx_key = _slate.context_key(mesh)
    key = ("eigh", mesh_key(mesh), A.dtype, n, side, nb, bool(compute_evecs), int(ctx_key),
           elpa_threads())
    fn = _JIT_CACHE.get(key)
    if fn is None:
        threads = elpa_threads()
        back, stedc, other = rocblas_regions(side, p, A.dtype)
        results = (jax.ShapeDtypeStruct((side,), jnp.float64),
                   jax.ShapeDtypeStruct((local, local), A.dtype),
                   jax.ShapeDtypeStruct((workspace_bytes(side, p, A.dtype, threads),), jnp.uint8),
                   jax.ShapeDtypeStruct((local, local), A.dtype))     # ELPA's destroyed input
        attrs = dict(n=side, nb=nb, ctx_key=int(ctx_key), compute_evecs=bool(compute_evecs),
                     pool_bytes=pool_bytes(side, p, A.dtype), rb_back_bytes=back,
                     rb_stedc_bytes=stedc, rb_other_bytes=other)

        @partial(shard_map, mesh=mesh, in_specs=P("x", "y"), out_specs=(P(), P("x", "y")),
                 check_vma=False)
        def _call(local_A):
            # Col-major bytes of the rank's block (ELPA's local array); see the module note.
            # Operand 0 (the transposed block, an XLA temporary) IS result 3: ELPA works in place.
            W, Q_T = jax.ffi.ffi_call(_EIGH_FFI_TARGET, results, input_output_aliases={0: 3})(
                jnp.transpose(local_A, (1, 0)), **attrs)[:2]
            # Q_T's rows are Q's columns (the eigenvalue index, over 'y'), block-cyclic at
            # nb < N/p: into W's ascending order, then untranspose (Q at P('x','y')).
            if compute_evecs and nb != local:
                Q_T = cyclic_to_block(Q_T, n=side, p=q, mb=nb, axis="y")
            return W, jnp.transpose(Q_T, (1, 0))

        face = NamedSharding(mesh, P("x", "y"))

        def _padded(a):
            whole = jax.lax.with_sharding_constraint(pad_with_sentinels(a, side), face)
            W, Q = _call(whole)
            W, Q = W[side - n:], Q[:n, side - n:]
            return W, jax.lax.with_sharding_constraint(Q, face)

        # No donation: plan.DONATES declares eigh as donating nothing.
        fn = jax.jit(_call if side == n else _padded)
        _JIT_CACHE[key] = fn
    return fn(A)

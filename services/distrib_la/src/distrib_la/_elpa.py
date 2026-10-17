"""``distributed_eigh`` on ELPA -- the ROCM leg's distributed Hermitian eigensolver.

ELPA 2026.02.001 on the AMD GPU (``config/frontier/build_elpa_hip.sh``, bound by
``cpp/elpa/eigh_ffi.cc``).  The DEFAULT solver is ELPA's ONE-stage path: GPU
tridiagonalization, GPU divide-and-conquer, GPU back-transformation.  Measured on MI250X
(ELPA_ROCM_REPORT.md), P=4 (2x2), cold wall, sampled residual/orthogonality <= 2.4e-14:

    n       complex128: 1-stage  2-stage  SLATE-HIP     float64: 1-stage  2-stage  SLATE-HIP
    20480               23.0     71.5     121.8                  15.3     34.5     84.4
    30720               (table)  186.6    aborts                 33.7     88.2     213.6

The two-stage path (``solver=2``; host band -> tridiagonal) is slower on this hardware at
every size measured and returns non-orthonormal eigenvectors (orthogonality ~0.1, residual
exact) on the 4x4 grid at n >= 47104 (n = 46080 is exact; 1-stage is exact at 47104), so it
refuses on grids of p >= 4 above :data:`_TWO_STAGE_MAX_SIDE_P4`; it stays selectable for
the measurements that put it there.

The operand contract is SLATE's (:func:`distrib_la._slate.distributed_eigh`): each rank's
``(N/p, N/p)`` block, locally transposed to column-major bytes, read as ELPA's block-cyclic
local array at tile ``nb | N/p`` on the square ``p x p`` grid, so ELPA solves
``Pi A Pi^T``; the eigenvector rows come back in ``A``'s order and only Q's columns are
permuted back (:mod:`distrib_la._cyclic`).  The handler shares the SLATE family's per-mesh
MPI context, as ScaLAPACK does.

Tile: ELPA's GPU kernels need ``nb`` a power of two (any other value silently moves work
to the host).  1-stage: nb 64 and 128 measured equal (20480, both dtypes: 23.1 vs 23.0 s
complex128, 15.3 vs 15.5 s float64), so 64, which divides more local edges.  2-stage:
complex128 64, float64 128 (32 was 1.45x, 128 1.16x the complex128 wall).  A local edge with
no such divisor is padded to the next multiple of the tile, with sentinel eigenvalues below
the spectrum (:func:`pad_with_sentinels`).

Device memory: the workspace output (an XLA buffer) holds the A copy (ELPA destroys its
input; A is never written) and ELPA's own device buffers, which the patched library takes
from it through ``lorrax_elpa_set_device_allocator``.  The pool is the lower envelope of
ELPA's measured live high-water (:func:`pool_bytes`); a larger need falls back to a device
malloc outside the arena, counted (``LORRAX_ELPA_POOL_REPORT=1``).
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

__all__ = ["distributed_eigh", "eigh_layout", "workspace_bytes", "pool_bytes", "SOLVER"]

_EIGH_FFI_TARGET = "lorrax_elpa_eigh"
_ALIGN = 256

#: The default ELPA solver (1: one-stage, 2: two-stage); see the module docstring.
SOLVER = 1
#: The largest side the two-stage solver is measured exact at on a p >= 4 grid (4x4).
_TWO_STAGE_MAX_SIDE_P4 = 46080

#: Preferred tile per (solver, dtype) (measured, module docstring) and the smallest tile
#: used before padding instead.
_TILE = {(1, "complex128"): 64, (1, "float64"): 64,
         (2, "complex128"): 64, (2, "float64"): 128}
_TILE_MIN = 64

#: ELPA's live device high-water through the allocator hook, per rank:
#: ``POOL_PER_LOCAL2[(solver, dtype)] * (N/p)^2 + POOL_PER_LOCAL * (N/p)`` bytes, the lower
#: envelope of the measured points, MI250X, P = 4 and 16 (ELPA_ROCM_REPORT.md, table
#: "pool"; tests/test_distrib_la_elpa_gate.py carries them).  2-stage: three real local
#: matrices for both dtypes; 1-stage: 1.5 complex128 / 2 float64 local matrices.
POOL_PER_LOCAL2 = {(1, "complex128"): 24.0, (1, "float64"): 16.0,
                   (2, "complex128"): 24.0, (2, "float64"): 24.0}
POOL_PER_LOCAL = 2048.0

_JIT_CACHE: dict = {}


def _pow2_floor(x: int) -> int:
    return 1 << (max(1, int(x)).bit_length() - 1)


def eigh_layout(n: int, p: int, dtype, solver: int = SOLVER) -> Tuple[int, int]:
    """``(N, nb)``: the side ELPA solves an ``n x n`` eigh at on a ``p x p`` grid, and its tile.

    ``nb`` is the preferred power-of-two tile when it divides ``n/p``, else the largest
    power of two in ``[_TILE_MIN, preferred]`` that does; otherwise the local edge is padded
    to the next multiple of the preferred tile (fewer than ``nb`` rows per rank).  A local
    edge below the tile uses the largest power of two not above it (the one-device route
    serves such sizes by default).
    """
    pref = _TILE[(int(solver), jnp.dtype(dtype).name)]
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


def _a_copy_bytes(local: int, itemsize: int) -> int:
    """Bytes of the A copy at the head of the workspace (cpp/elpa/eigh_ffi.cc a_copy_bytes)."""
    return -(-local * local * itemsize // _ALIGN) * _ALIGN


def pool_bytes(side: int, p: int, dtype, solver: int = SOLVER) -> int:
    """ELPA's device pool for a side-``side`` solve on a ``p x p`` grid (module constants)."""
    local = side // p
    b = POOL_PER_LOCAL2[(int(solver), jnp.dtype(dtype).name)] * local * local + POOL_PER_LOCAL * local
    return -(-int(b) // _ALIGN) * _ALIGN


def workspace_bytes(side: int, p: int, dtype, solver: int = SOLVER) -> int:
    """The handler's workspace: the A copy plus ELPA's pool."""
    return _a_copy_bytes(side // p, jnp.dtype(dtype).itemsize) + pool_bytes(side, p, dtype, solver)


def distributed_eigh(A: jax.Array, *, mesh: Mesh, compute_evecs: bool = True,
                     block_size: int | None = None, side: int | None = None,
                     solver: int = SOLVER) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition on ELPA (ROCM leg).

    Same contract as :func:`distrib_la._slate.distributed_eigh`: ``A`` square, float64 or
    complex128, sharded ``P('x','y')`` on a square ``('x','y')`` mesh; returns ``W``
    ``(n,)`` ascending, replicated, and ``Q`` ``(n, n)`` with the eigenvectors as columns,
    ``P('x','y')``.  ``block_size`` must be a power of two dividing ``side/p``; ``side``
    defaults to :func:`eigh_layout`'s; ``solver`` is ELPA's 1- or 2-stage path.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"elpa distributed_eigh: expected a square matrix, got {A.shape}")
    if A.dtype not in (jnp.complex128, jnp.float64):
        raise ValueError(f"elpa distributed_eigh: dtype {A.dtype} (supported: complex128, float64)")
    solver = int(solver)
    if solver not in (1, 2):
        raise ValueError(f"elpa distributed_eigh: solver {solver} (1: one-stage, 2: two-stage)")
    p, q = _slate.validate_mesh(mesh, require_square=True)
    n = int(A.shape[0])
    if n % p:
        raise ValueError(f"elpa distributed_eigh: n={n} must be divisible by mesh axis size {p}.")
    if side is None:
        side, nb = eigh_layout(n, p, A.dtype, solver)
    else:
        side = int(side)
        nb = eigh_layout(side, p, A.dtype, solver)[1]
    if block_size is not None:
        nb = int(block_size)
    local = side // p
    if side < n or side % p or nb < 1 or nb & (nb - 1) or local % nb:
        raise ValueError(f"elpa distributed_eigh: tile {nb} must be a power of two dividing "
                         f"the solve tile {local} (side {side}, {p}x{q} mesh).")
    if solver == 2 and p >= 4 and side > _TWO_STAGE_MAX_SIDE_P4:
        raise ValueError(
            f"elpa distributed_eigh: the two-stage solver returns non-orthonormal eigenvectors "
            f"on the {p}x{q} grid at side {side} > {_TWO_STAGE_MAX_SIDE_P4} (ELPA 2026.02.001, "
            f"measured: ELPA_ROCM_REPORT.md); use solver=1 (the default).")
    _slate.ensure_registered(mesh, _EIGH_FFI_TARGET)
    ctx_key = _slate.context_key(mesh)
    key = ("eigh", mesh_key(mesh), A.dtype, n, side, nb, solver, bool(compute_evecs), int(ctx_key))
    fn = _JIT_CACHE.get(key)
    if fn is None:
        results = (jax.ShapeDtypeStruct((side,), jnp.float64),
                   jax.ShapeDtypeStruct((local, local), A.dtype),
                   jax.ShapeDtypeStruct((workspace_bytes(side, p, A.dtype, solver),), jnp.uint8))
        attrs = dict(n=side, nb=nb, solver=solver, ctx_key=int(ctx_key),
                     compute_evecs=bool(compute_evecs))

        @partial(shard_map, mesh=mesh, in_specs=P("x", "y"), out_specs=(P(), P("x", "y")),
                 check_vma=False)
        def _call(local_A):
            # Col-major bytes of the rank's block (ELPA's local array); see the module note.
            W, Q_T = jax.ffi.ffi_call(_EIGH_FFI_TARGET, results)(
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

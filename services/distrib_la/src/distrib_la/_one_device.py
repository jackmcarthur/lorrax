"""``one_device`` — the size route of ``distributed`` (resolve guard 1b).

Where one matrix fits one device, a distributed library's cost is its fixed
per-call charge (§ "Distributed is a capacity route" in
``docs/services/distrib_la/backends.md``).  On the ROCM leg that charge was
measured at 0.22-0.29 s per eigh at n = 32-128 complex128 against
0.0016-0.0045 s native (pool log distla_bound1).  This
backend serves that range with the leg's native JAX kernel and the FFI
backends' operand contract, so a ``distributed`` call site runs it unchanged:

    eigh      A (n, n) P('x','y') or (B, n, n) P(None,'x','y')
              -> W replicated, Z (eigenvector columns) in A's layout
    solve_lu  A (B, n, n), B (B, n, NRHS) at P(None,'x','y') -> X in B's layout

The movement is route (c)'s (:mod:`distrib_la._batch_reshard`): one
all_to_all puts whole matrices on ranks, the local kernel solves them
(``native_eigh``: the leg's eigh route with zero-row deflation;
``local_lu.solve_local``: ``jnp.linalg.solve``, or the leg's span-blocked LU), the inverse all_to_all returns the face layout.  A
single matrix is a stack of one: rank 0 solves it, the other ranks hold the
synthetic padding, and the inverse exchange scatters rank 0's result, so every
rank's tile comes from ONE solve.

Memory.  A rank holds at most ``round_n^2`` matrix elements' worth of whole
matrices per round (``round_n = min(max_n, ONE_DEVICE_ROUND_N row)``,
:func:`distrib_la.resolve.one_device_round_n`; ``max_n`` the selection row, :data:`distrib_la.resolve.ONE_DEVICE_MAX_N`,
measured at complex128): a stack whose ``ceil(B/P)`` matrices per rank exceed
that runs in equal rounds (:func:`rounds_for`).  A shape rule, the same on every
rank, so no size is exchanged.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._batch_reshard import (batch_reshard_call, reshard_program,
                                       reshard_rounds_call, validate_batch_reshard_operands)
from distrib_la.resolve import (ONE_DEVICE, mesh_key, mesh_platform, one_device_max_n,
                                one_device_round_n)

__all__ = ["distributed_eigh", "batched_distributed_eigh",
           "batched_distributed_solve_lu", "rounds_for", "held_per_rank",
           "one_device_bytes", "ONE_DEVICE_LU_TILES"]

#: Whole n^2 tiles a rank holds while ``one_device`` solves one LU system with
#: n right-hand sides: A, B and X whole plus the factor's working pair (the local
#: plan's price, ``gw.w_isdf._dyson_route``).
ONE_DEVICE_LU_TILES = 5


def one_device_bytes(op: str, n: int, itemsize: int = 16) -> int:
    """Per-rank device bytes of the smallest ``one_device`` round: ONE whole matrix.

    eigh: :data:`distrib_la.plan.BATCH_EIGH_TILES` n^2 elements (the stack price of
    route (c), :func:`distrib_la.eigh_stack_bytes`); solve_lu:
    :data:`ONE_DEVICE_LU_TILES`.  complex128 by default, the dtype the leg rows
    were measured at.  A shape rule: every rank computes the same figure."""
    if op == "eigh":
        from distrib_la.plan import BATCH_EIGH_TILES
        tiles = BATCH_EIGH_TILES
    elif op == "solve_lu":
        tiles = ONE_DEVICE_LU_TILES
    else:
        raise ValueError(f"one_device_bytes: no one_device route for {op!r}")
    return int(tiles) * int(n) * int(n) * int(itemsize)

_JIT_CACHE: dict = {}


def _bound(op: str, mesh: Mesh, n: int, max_n: int | None) -> int:
    """The leg's bound (or the caller's override, for gates on a CPU mesh)."""
    bound = one_device_max_n(op, mesh_platform(mesh)) if max_n is None else int(max_n)
    if bound is None:
        raise ValueError(
            f"GATE one_device: {op} '{ONE_DEVICE}' has no size row on platform "
            f"{mesh_platform(mesh)!r} (distrib_la.resolve.ONE_DEVICE_MAX_N).")
    if n > bound:
        raise ValueError(
            f"GATE one_device: {op} '{ONE_DEVICE}' at n={n} exceeds the bound {bound} "
            f"(distrib_la.resolve.ONE_DEVICE_MAX_N); 'distributed' resolves the "
            f"library there.")
    return bound


def held_per_rank(nb: int, n: int, ranks: int, bound: int, width: int | None = None) -> int:
    """Whole matrices a rank holds per round: ``ceil(B/P)``, capped so that their
    ``n * max(n, width)`` elements stay within ``bound^2`` (at least one)."""
    width = n if width is None else max(n, int(width))
    return max(1, min(-(-nb // ranks), (bound * bound) // (n * width)))


def rounds_for(nb: int, n: int, ranks: int, bound: int, width: int | None = None) -> int:
    """Rounds of a ``(B, n, n)`` stack: ``ceil(ceil(B/P) / held_per_rank)``."""
    return -(-(-(-nb // ranks)) // held_per_rank(nb, n, ranks, bound, width))


def _ranks(mesh: Mesh) -> int:
    return int(mesh.shape["x"]) * int(mesh.shape["y"])


def batched_distributed_eigh(A, *, mesh: Mesh, compute_evecs: bool = True,
                             block_size=None, side=None, max_n: int | None = None,
                             rounds: int | None = None):
    """Eigh of a face stack ``(B, n, n)`` at ``P(None,'x','y')``, each matrix on one device.

    ``compute_evecs``, ``block_size`` and ``side`` are the library wrappers'
    keywords; they configure a distributed descriptor this route does not have
    and are accepted for the shared call contract (the vectors are always
    returned, as on every eigh route).  ``max_n`` overrides the leg row (gates).
    """
    del compute_evecs, block_size, side
    nb, n = int(A.shape[0]), int(A.shape[-1])
    bound = _bound("eigh", mesh, n, max_n)
    if rounds is None:
        rounds = rounds_for(nb, n, _ranks(mesh), one_device_round_n("eigh", mesh_platform(mesh), bound))
    return reshard_rounds_call("eigh", mesh, A, rounds=int(rounds))


def distributed_eigh(A, *, mesh: Mesh, compute_evecs: bool = True,
                     block_size=None, side=None, max_n: int | None = None):
    """Eigh of ONE matrix ``(n, n)`` at ``P('x','y')``: a stack of one (rank 0 solves)."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"one_device eigh: expected a square matrix, got {A.shape}")
    n = int(A.shape[-1])
    _bound("eigh", mesh, n, max_n)
    key = ("eigh1", mesh_key(mesh), n, str(A.dtype))
    fn = _JIT_CACHE.get(key)
    if fn is None:
        validate_batch_reshard_operands("eigh", mesh, (jax.ShapeDtypeStruct((1, n, n), A.dtype),))
        program = reshard_program("eigh", mesh, (((1, n, n), str(A.dtype)),))
        face = NamedSharding(mesh, P("x", "y"))

        def one(a):
            w, z = program(a[None])
            return w[0], jax.lax.with_sharding_constraint(z[0], face)
        fn = _JIT_CACHE[key] = jax.jit(one)
    return fn(A)


def batched_distributed_solve_lu(A, B, *, mesh: Mesh, max_n: int | None = None,
                                 rounds: int | None = None):
    """Solve ``A[q] X[q] = B[q]`` with each matrix on one device (``jnp.linalg.solve``).

    ``A`` (B, n, n) and ``B`` (B, n, NRHS) at ``P(None,'x','y')``; X in B's
    layout.  Nothing is donated by this route (``Plan.DONATES`` stays the
    conservative caller contract: a caller never relies on survival).
    """
    validate_batch_reshard_operands("solve_lu", mesh, (A, B))
    nb, n, nrhs = int(A.shape[0]), int(A.shape[-1]), int(B.shape[-1])
    bound = _bound("solve_lu", mesh, n, max_n)
    if rounds is None:
        rounds = rounds_for(nb, n, _ranks(mesh),
                            one_device_round_n("solve_lu", mesh_platform(mesh), bound), width=nrhs)
    rounds = max(1, min(int(rounds), nb))
    if rounds == 1:
        return batch_reshard_call("solve_lu", mesh, (A, B))
    m = -(-nb // rounds)
    key = ("lu_rounds", mesh_key(mesh), nb, n, nrhs, str(A.dtype), rounds)
    fn = _JIT_CACHE.get(key)
    if fn is None:
        program = reshard_program("solve_lu", mesh, (((m, n, n), str(A.dtype)),
                                                     ((m, n, nrhs), str(B.dtype))))
        face = NamedSharding(mesh, P(None, "x", "y"))
        # The last slice starts at B - m; the matrices it shares with the
        # previous slice are solved again and written with the same values.
        starts = np.asarray([min(r * m, nb - m) for r in range(rounds)], np.int32)

        @partial(jax.jit, out_shardings=face)
        def fn(a, b):
            def body(x_all, start):
                cut = partial(jax.lax.dynamic_slice_in_dim, start_index=start, slice_size=m, axis=0)
                x = program(jax.lax.with_sharding_constraint(cut(a), face),
                            jax.lax.with_sharding_constraint(cut(b), face))
                return jax.lax.with_sharding_constraint(
                    jax.lax.dynamic_update_slice_in_dim(x_all, x, start, axis=0), face), None
            x, _ = jax.lax.scan(body, jax.lax.with_sharding_constraint(jnp.zeros_like(b), face), starts,
                                unroll=1)
            return x
        _JIT_CACHE[key] = fn
    return fn(A, B)

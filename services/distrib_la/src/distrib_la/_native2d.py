"""``native2d`` — 2-D block-distributed Cholesky, in pure JAX.

Absorbed from LORRAX ``src/common/cholesky_2d.py`` at 96a6399.  It is a
BACKEND, not a mode of ``native``: a different algorithm with a different
memory profile, chosen by name.

What it is for
--------------
The replicated route reshards ``C_q`` from ``P(None,'x','y')`` to
``P(('x','y'),None,None)`` and pays an involuntary full rematerialization
for it.  This one works directly on the 2-D sharded matrix::

    n = 10k, P = 128     reshard route   1.6 GB/device  (may OOM)
                         2-D blocked     5   MB/device

O(J log P) communication rounds via ``psum``, √P less bandwidth than a 1-D
distribution, and ``lax.map`` over the q axis so the whole stack is ONE XLA
dispatch (~18x over a Python loop).

The tile layout is INTERNAL
---------------------------
The kernel works on 5-axis tiles ``(nq, J, J, b, b)`` at
``P(None,'x','y',None,None)``, which is not the facade's
``P(None,'x','y')``.  That mismatch used to be the caller's problem:
``isdf/core.py`` held ``dense_to_tiles`` at :3233 and ``tiles_to_dense`` at
:3241 with two ``with_sharding_constraint`` calls between them, so the
5-axis layout leaked into a physics module.  Here it lives behind
:func:`cholesky` and nothing outside this file names ``J`` or ``b``.

Marking, and the defect that made it a shared helper
----------------------------------------------------
This is the ONE kernel in distrib_la written with VMA checking ON (every
FFI wrapper passes ``check_vma=False``, which exempts its carries), so it
is the one that has to mark its loop carry.  It used to carry a private
``try: lax.pcast / except AttributeError: identity`` guard commented "jax
<= 0.8 has no VMA tracking" — and that comment was FALSE.  Tracking starts
at jax 0.7.0 and 0.7.x has no ``lax.pcast`` at all, so the except-branch
installed a NO-OP on exactly the generation that enforces the marking,
including the container the file's own docstring named.
:func:`lxkit.jax_compat.mark_varying` owns the decision now, with the
measured version table and a refusal instead of a silent identity.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from lxkit.jax_compat import mark_varying

from distrib_la._shard_map import shard_map

__all__ = ["cholesky", "block_size_for", "dense_to_tiles", "tiles_to_dense"]


def block_size_for(n: int, Pr: int, Pc: int) -> tuple[int, int]:
    """``(block_size, J)`` for an ``n x n`` matrix on a ``Pr x Pc`` mesh.

    Ported from ``common.fft_helpers.compute_block_size_for_2d_cholesky``,
    which is where it sat purely because that module already had ``math``
    imported; nothing about it is an FFT fact.

    The constraints are fundamental to any 2-D blocked algorithm::

        n % block_size == 0     the matrix divides into whole tiles
        J % Pr == 0             tile ROWS distribute evenly on 'x'
        J % Pc == 0             tile COLS distribute evenly on 'y'

    with ``J = n / block_size``.  The simplest solution is
    ``J = lcm(Pr, Pc)``; when ``n`` does not divide that, multiples of it
    are tried, then any valid factorization.  A raise here is the honest
    answer for an ``n`` this mesh cannot tile — and it fires at resolve
    time (:func:`distrib_la.resolve.resolve_backend` calls it when ``n`` is
    given), not halfway through a factor stage.
    """
    target_J = math.lcm(Pr, Pc)
    if n % target_J == 0:
        return n // target_J, target_J
    for mult in range(2, 20):
        J = target_J * mult
        if n % J == 0:
            return n // J, J
    for b in range(n, 0, -1):
        if n % b == 0:
            J = n // b
            if J % Pr == 0 and J % Pc == 0:
                return b, J
    raise ValueError(
        f"native2d: no valid block size for n={n} on a {Pr}x{Pc} mesh.  n "
        f"must be divisible by lcm({Pr},{Pc})={target_J}, or by a multiple "
        f"of it.  Pad n, or use a backend with no tile decomposition.")


def dense_to_tiles(A: jax.Array, b: int) -> jax.Array:
    """``(..., n, n)`` dense → ``(..., J, J, b, b)`` lower tiles, J = n/b.

    Tiles strictly above the diagonal (i < j) are zeroed: this kernel
    reads the lower triangle only, and leaving the upper live would make a
    non-Hermitian input silently produce a plausible answer.
    """
    *batch, n, _ = A.shape
    if n % b:
        raise ValueError(f"native2d: n={n} must be divisible by b={b}")
    J = n // b
    # (..., n, n) -> (..., J, b, J, b) -> (..., J, J, b, b)
    tiles = jnp.moveaxis(A.reshape(*batch, J, b, J, b), -2, -3)
    i, j = jnp.meshgrid(jnp.arange(J), jnp.arange(J), indexing="ij")
    mask = (i >= j).reshape((1,) * len(batch) + (J, J, 1, 1))
    return jnp.where(mask, tiles, 0.0)


def tiles_to_dense(tiles: jax.Array, b: int) -> jax.Array:
    """``(..., J, J, b, b)`` → ``(..., n, n)`` dense, n = J*b.  The exact
    inverse of :func:`dense_to_tiles` on the lower triangle."""
    *batch, J, _, _, _ = tiles.shape
    return jnp.moveaxis(tiles, -3, -2).reshape(*batch, J * b, J * b)


def _chol_2d_single(mesh: Mesh, J: int, b: int):
    """The core right-looking blocked Cholesky, one matrix.

    Takes ``(J, J, b, b)`` tiles sharded ``P('x','y',None,None)`` and
    returns the factor in the same format and sharding::

        for k in range(J):
            1. POTRF: the diagonal owner factors L[k,k] = chol(A[k,k])
            2. broadcast L[k,k] via psum          (O(log P) rounds)
            3. TRSM: column-k owners set L[i>k,k] = A[i,k] @ L[k,k]^-H
            4. broadcast the panel L[:,k] via psum
            5. SYRK: everyone updates A[i,j] -= L[i,k] @ L[j,k]^H, i,j>k
    """
    Pr = int(mesh.shape["x"])
    Pc = int(mesh.shape["y"])
    if J % Pr or J % Pc:
        raise ValueError(
            f"native2d: J={J} tiles must divide by both mesh axes "
            f"({Pr},{Pc}); block_size_for() picks a J that does.")
    J_row, J_col = J // Pr, J // Pc

    @partial(shard_map, mesh=mesh,
             in_specs=P("x", "y", None, None),
             out_specs=P("x", "y", None, None))
    def _local(A_local: jax.Array) -> jax.Array:
        px = lax.axis_index("x")
        py = lax.axis_index("y")
        dtype = A_local.dtype
        my_row_start = px * J_row          # JAX uses BLOCK distribution
        my_col_start = py * J_col

        def right_solve_LH(block, L):
            """Solve X @ L^H = block for X."""
            return solve_triangular(L.conj(), block.T, lower=True, trans="N").T

        def body_k(A_local, k):
            diag_px = k // J_row           # who owns diagonal block (k, k)
            diag_py = k // J_col
            is_diag_owner = jnp.logical_and(px == diag_px, py == diag_py)
            is_col_k_owner = (py == diag_py)
            k_loc_row = k % J_row          # local indices, valid on owner
            k_loc_col = k % J_col

            # 1. POTRF on the diagonal owner.
            Lkk_local = jnp.linalg.cholesky(A_local[k_loc_row, k_loc_col])
            A_local = jnp.where(
                is_diag_owner,
                A_local.at[k_loc_row, k_loc_col].set(Lkk_local),
                A_local)

            # 2. Broadcast Lkk via psum (tree-based, O(log P)).
            Lkk = lax.psum(
                jnp.where(is_diag_owner, Lkk_local, jnp.zeros((b, b), dtype)),
                ("x", "y"))

            # 3. TRSM: column-k owners update their rows i > k.
            def do_trsm(i_loc, A):
                should = jnp.logical_and(is_col_k_owner,
                                         my_row_start + i_loc > k)
                Lik = right_solve_LH(A[i_loc, k_loc_col], Lkk)
                return jnp.where(should, A.at[i_loc, k_loc_col].set(Lik), A)

            A_local = lax.fori_loop(0, J_row, do_trsm, A_local)

            # 4. Broadcast the panel L[:, k] via psum.  From jax 0.7.0
            # shard_map tracks varying manual axes: this carry must be
            # marked varying on ('x','y') because fill_panel writes
            # per-device-conditionally while jnp.zeros starts with an EMPTY
            # varying set, and unmarked the carry-type check rejects the
            # loop with "the varying manual axes do not match".  Mark the
            # axes the BODY introduces, not every mesh axis: over-marking a
            # carry that leaves through a replicated out_specs is itself an
            # error (measured; see lxkit.jax_compat).
            panel_init = mark_varying(jnp.zeros((J, b, b), dtype), ("x", "y"))

            def fill_panel(i_loc, panel):
                i_glob = my_row_start + i_loc
                should = jnp.logical_and(is_col_k_owner, i_glob >= k)
                return jnp.where(
                    should,
                    panel.at[i_glob].set(A_local[i_loc, k_loc_col]),
                    panel)

            panel = lax.psum(
                lax.fori_loop(0, J_row, fill_panel, panel_init), ("x", "y"))

            # 5. SYRK: update blocks (i, j) with i > k, j > k, i >= j.
            def do_syrk_col(j_loc, A):
                j_glob = my_col_start + j_loc
                Ljk = panel[j_glob]

                def do_syrk_row(i_loc, A):
                    i_glob = my_row_start + i_loc
                    should = jnp.logical_and(
                        jnp.logical_and(i_glob > k, j_glob > k),
                        i_glob >= j_glob)
                    update = panel[i_glob] @ Ljk.conj().T
                    return jnp.where(
                        should,
                        A.at[i_loc, j_loc].set(A[i_loc, j_loc] - update),
                        A)

                return lax.fori_loop(0, J_row, do_syrk_row, A)

            A_local = lax.fori_loop(0, J_col, do_syrk_col, A_local)
            return A_local, None

        A_local, _ = lax.scan(body_k, A_local, jnp.arange(J), unroll=1)
        return A_local

    return _local


_KERNEL_CACHE: dict = {}


def _batched_kernel(mesh: Mesh, J: int, b: int):
    """``jit(lax.map(single))`` over the q axis — ONE XLA dispatch for the
    whole stack (~18x over a Python loop), memoized per (mesh, J, b)."""
    key = (tuple(mesh.axis_names),
           tuple(int(s) for s in mesh.shape.values()),
           tuple(d.id for d in mesh.devices.flat), J, b)
    fn = _KERNEL_CACHE.get(key)
    if fn is None:
        single = _chol_2d_single(mesh, J, b)
        fn = jax.jit(lambda stack: lax.map(single, stack))
        _KERNEL_CACHE[key] = fn
    return fn


def cholesky(A: jax.Array, *, mesh: Mesh,
             block_size: int | None = None) -> jax.Array:
    """Cholesky-factor every ``A[q]``; DENSE in, DENSE out.

    ``A`` is ``(nq, n, n)`` at ``P(None,'x','y')`` and the returned
    lower-triangular factor is in the same shape and sharding.  The 5-axis
    tile layout the kernel runs on exists only between these two lines.

    ``block_size`` defaults to :func:`block_size_for`; passing one is how a
    caller that has already computed a decomposition avoids recomputing it,
    not an invitation to pick a different tiling.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"native2d cholesky: expected (nq, n, n); got {A.shape}")
    n = int(A.shape[1])
    Pr, Pc = int(mesh.shape["x"]), int(mesh.shape["y"])
    if block_size is None:
        block_size, J = block_size_for(n, Pr, Pc)
    else:
        block_size = int(block_size)
        if n % block_size:
            raise ValueError(
                f"native2d cholesky: n={n} is not divisible by "
                f"block_size={block_size}")
        J = n // block_size

    tiles = jax.lax.with_sharding_constraint(
        dense_to_tiles(A, block_size),
        NamedSharding(mesh, P(None, "x", "y", None, None)))
    L_tiles = _batched_kernel(mesh, J, block_size)(tiles)
    return jax.lax.with_sharding_constraint(
        tiles_to_dense(L_tiles, block_size),
        NamedSharding(mesh, P(None, "x", "y")))

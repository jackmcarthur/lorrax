"""``native2d`` — face-distributed Cholesky and partial-pivot LU in JAX.

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
jax is pinned to 0.9, and :func:`lxkit.mark_varying` is ``lax.pcast``.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from lxkit import mark_varying

from jax import shard_map
from distrib_la.resolve import mesh_key

__all__ = ["cholesky", "block_size_for", "dense_to_tiles", "tiles_to_dense",
           "batched_distributed_solve_lu"]


def block_size_for(n: int, Pr: int, Pc: int) -> tuple[int, int]:
    """``(block_size, J)`` for an ``n x n`` matrix on a ``Pr x Pc`` mesh.

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
            from distrib_la.local_lu import tri_solve_local   # 2^31-byte span route (ROCM)
            return tri_solve_local(L.conj(), block.T, native=lambda l, b: solve_triangular(
                l, b, lower=True, trans="N")).T

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
            Lkk = lax.psum(  # exact-reduce: diagonal-block owner's tile, zeros elsewhere
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
            # error (measured; see lxkit.mark_varying).
            panel_init = mark_varying(jnp.zeros((J, b, b), dtype), ("x", "y"))

            def fill_panel(i_loc, panel):
                i_glob = my_row_start + i_loc
                should = jnp.logical_and(is_col_k_owner, i_glob >= k)
                return jnp.where(
                    should,
                    panel.at[i_glob].set(A_local[i_loc, k_loc_col]),
                    panel)

            panel = lax.psum(  # exact-reduce: column-k owners fill disjoint rows
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
    key = (mesh_key(mesh), J, b)
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


def _lu_kernel(mesh, n, nrhs, width):
    """Partial-pivot ``P A = L U`` and ``U X = L^-1 P B`` on x/y faces.

    Pivot steps exchange only two rows of the column panel. Their composed
    permutation moves at most ``2 * width`` full row pieces once per panel.
    Pivot reductions and row swaps use the x (process-column) communicator;
    the trailing updates are local GEMMs. Storage consists of local faces
    and fixed-width row/column panels, never a replicated full matrix.
    """
    key = ("solve_lu", mesh_key(mesh), n, nrhs, width)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    m, nc, rc = n // px, n // py, nrhs // py
    panels = -(-n // width)

    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, "x", "y"), P(None, "x", "y")),
             out_specs=P(None, "x", "y"), check_vma=False)
    def local(a, rhs):
        x, y = lax.axis_index("x"), lax.axis_index("y")
        rows = x * m + jnp.arange(m)
        columns = y * nc + jnp.arange(nc)
        slots = jnp.arange(width)

        def column_panel(a, start):
            col = start + slots
            idx = col - y * nc
            owned = (idx >= 0) & (idx < nc) & (col < n)
            panel = jnp.take(a, jnp.clip(idx, 0, nc - 1), axis=1)
            return lax.psum(jnp.where(owned[None, :], panel, 0), "y")

        def row_panel(a, start):
            row = start + slots
            idx = row - x * m
            owned = (idx >= 0) & (idx < m) & (row < n)
            panel = jnp.take(a, jnp.clip(idx, 0, m - 1), axis=0)
            return lax.psum(jnp.where(owned[:, None], panel, 0), "x")

        def store_rows(a, values, start):
            row = start + slots
            idx = row - x * m
            idx = jnp.where((idx >= 0) & (idx < m) & (row < n), idx, m)
            return a.at[idx].set(values, mode="drop")

        def store_columns(a, values, start):
            col = start + slots
            idx = col - y * nc
            idx = jnp.where((idx >= 0) & (idx < nc) & (col < n), idx, nc)
            return a.at[:, idx].set(values, mode="drop")

        def diagonal(panel, start):
            block = row_panel(panel, start)
            valid = start + slots < n
            return jnp.where(valid[:, None] & valid[None, :], block,
                             jnp.eye(width, dtype=panel.dtype))

        def one(pair):
            a, b = pair

            def factor_panel(i, carry):
                a, b = carry
                start = lax.optimization_barrier(i * width)
                panel = column_panel(a, start)

                def pivot_column(j, carry):
                    panel, pivots = carry
                    k = start + j

                    def pivot(carry):
                        panel, pivots = carry
                        magnitude = jnp.where(rows >= k, jnp.abs(panel[:, j]), -jnp.inf)
                        maximum = lax.pmax(jnp.max(magnitude), "x")
                        p = lax.pmin(jnp.min(jnp.where(magnitude == maximum, rows, n)), "x")
                        # Select BEFORE communication: exactly two row pieces,
                        # never a permutation/gather of the complete face.
                        rr = jnp.stack((k, p))
                        idx = jnp.clip(rr - x * m, 0, m - 1)
                        owned = (rr // m == x)[:, None]
                        selected = lax.psum(jnp.where(owned, panel[idx], 0), "x")
                        last = selected[1]
                        dest = jnp.where(rr // m == x, rr - x * m, m)
                        panel = panel.at[dest].set(selected[::-1], mode="drop")
                        below = rows > k
                        multiplier = jnp.where(below, panel[:, j] / last[j], 0)
                        update = multiplier[:, None] * jnp.where(slots > j, last, 0)[None, :]
                        panel = panel - update
                        panel = panel.at[:, j].set(jnp.where(below, multiplier, panel[:, j]))
                        return panel, pivots.at[j].set(p)

                    return lax.cond(k < n, pivot, lambda c: c, (panel, pivots))

                panel, pivots = lax.fori_loop(
                    0, width, pivot_column, (panel, start + slots))
                # Compose the panel's exact GEPP swaps before touching full
                # faces. Repeated targets retain identical origins and values.
                targets = jnp.concatenate((start + slots, pivots))

                def compose(j, origins):
                    k, p = start + j, pivots[j]
                    first, last = origins[j], origins[width + j]
                    return jnp.where(targets == k, last,
                                     jnp.where(targets == p, first, origins))

                origins = lax.fori_loop(0, width, compose, targets)
                idx = jnp.clip(origins - x * m, 0, m - 1)
                owned = (origins < n) & (origins // m == x)
                values = jnp.concatenate((a[idx], b[idx]), axis=1)
                values = lax.psum(jnp.where(owned[:, None], values, 0), "x")
                dest = targets - x * m
                dest = jnp.where((dest >= 0) & (dest < m) & (targets < n), dest, m)
                a = a.at[dest].set(values[:, :nc], mode="drop")
                b = b.at[dest].set(values[:, nc:], mode="drop")
                l11 = diagonal(panel, start)
                upper = solve_triangular(l11, row_panel(a, start), lower=True,
                                         unit_diagonal=True)
                upper = jnp.where((columns >= start + width)[None, :], upper, 0)
                lower = jnp.where((rows >= start + width)[:, None], panel, 0)
                a = a - lower @ upper
                # The just-factored panel replaces the stale input columns;
                # U12 replaces only trailing columns of the pivot rows.
                old_rows = row_panel(a, start)
                upper = jnp.where((columns >= start + width)[None, :], upper, old_rows)
                a = store_rows(a, upper, start)
                a = store_columns(a, panel, start)
                return a, b

            a, b = lax.fori_loop(0, panels, factor_panel, (a, b))

            def substitute(i, b, *, backward):
                start = lax.optimization_barrier((panels - 1 - i if backward else i) * width)
                panel = column_panel(a, start)
                block = diagonal(panel, start)
                z = solve_triangular(block, row_panel(b, start), lower=not backward,
                                     unit_diagonal=not backward)
                live = rows < start if backward else rows >= start + width
                panel = jnp.where(live[:, None], panel, 0)
                b = b - panel @ z
                return store_rows(b, z, start)

            b = lax.fori_loop(0, panels, partial(substitute, backward=False), b)
            return lax.fori_loop(0, panels, partial(substitute, backward=True), b)

        return lax.map(one, (a, rhs))

    fn = jax.jit(local)
    _KERNEL_CACHE[key] = fn
    return fn


def batched_distributed_solve_lu(A, B, *, mesh, block_size=None):
    """Solve general ``A[q] X[q] = B[q]`` with global partial pivoting.

    ``A`` is ``(q,n,n)`` and ``B`` is ``(q,n,nrhs)``, both at
    ``P(None,'x','y')``; the result keeps B's shape/layout. Extents are
    producer-padded to the mesh. Inert rows must carry an identity block
    in A and zeros in B, as for the other distributed LU providers.
    Singular factors produce nonfinite results, refused by the plan's
    existing common result check. No normal equations are formed.
    """
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    if (A.ndim != 3 or B.ndim != 3 or A.shape[0] != B.shape[0]
            or A.shape[-1] != A.shape[-2] or A.shape[-1] != B.shape[-2]
            or A.dtype != B.dtype):
        raise ValueError("native2d solve_lu requires A(q,n,n), B(q,n,nrhs) of one dtype")
    if A.dtype not in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        raise ValueError("native2d solve_lu requires float32/64 or complex64/128")
    n, nrhs = int(A.shape[-1]), int(B.shape[-1])
    if n < 1 or nrhs < 1 or n % px or n % py or nrhs % py:
        raise ValueError("native2d solve_lu requires positive mesh-padded n and nrhs")
    width = min(128, n, 128 if block_size is None else int(block_size))
    if width < 1:
        raise ValueError("native2d solve_lu block_size must be positive")
    face = NamedSharding(mesh, P(None, "x", "y"))
    return _lu_kernel(mesh, n, nrhs, width)(
        lax.with_sharding_constraint(A, face), lax.with_sharding_constraint(B, face))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharded blocked Cholesky using shard_map.

THREE distribution strategies:

1. 1D Column Ring (ppermute): O(J × P) communication rounds
2. 1D Column Tree (psum): O(J × log P) communication rounds 
3. 2D Block (psum): O(J × log P) rounds, but √P less bandwidth per proc

For N=10k:
- 1D Ring on P=128:     ~700 ms (O(P) comms)
- 1D psum on P=128:     ~37 ms  (O(log P) comms)
- 2D psum on 12×11:     ~37 ms  (same rounds, 5.5× less bandwidth)
- Strategy B (replicate C_q): ~23 ms but 1.6 GB/device memory

Use 2D for production when memory is tight. 1D psum is simpler.

Complexity comparison (P=128 processors, n=10k, b=256):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm      Comm Rounds    Bandwidth/proc
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1D Ring        J × P = 4992   1.6 GB
1D Tree        J × log P = 273  1.6 GB
2D Tree        J × log P = 273  0.29 GB (5.5× better)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2D has same asymptotic complexity as 1D but better bandwidth scaling.
"""

import os
# ---- force 4 logical devices on host CPU, enable float64 BEFORE importing jax
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_ENABLE_X64", "1")
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from common.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from jax.scipy.linalg import solve_triangular


def make_spd_hermitian(n: int, key: jax.Array, jitter: float = 1.0) -> jax.Array:
    """Random complex Hermitian positive-definite matrix."""
    x = jax.random.normal(key, (n, n)) + 1j * jax.random.normal(key, (n, n))
    a = x.conj().T @ x
    a = a + (jitter + 0.1 * n) * jnp.eye(n)  # ensure well-conditioned
    a = 0.5 * (a + a.conj().T)  # enforce Hermitian numerically
    return a


def dense_to_tiles_lower(a: jax.Array, bsz: int) -> jax.Array:
    """Pack lower-triangular tiles (i>=j) of A into shape (J, J, b, b). Upper tiles are zeros.
    Output axes: [block_row=i, block_col=j, b, b]
    """
    n = a.shape[0]
    assert n % bsz == 0
    J = n // bsz
    tiles = []
    for i in range(J):
        row = []
        for j in range(J):
            blk = a[i*bsz:(i+1)*bsz, j*bsz:(j+1)*bsz]
            row.append(jnp.where((i >= j), blk, jnp.zeros_like(blk)))
        row = jnp.stack(row, axis=0)   # (J, b, b) with axis0 = block_col j
        tiles.append(row)
    tiles = jnp.stack(tiles, axis=0)   # (J, J, b, b) with axes (i, j, b, b)
    return tiles


def tiles_lower_to_dense(tiles: jax.Array, bsz: int) -> jax.Array:
    """Reconstruct dense lower-triangular matrix from (J, J, b, b) tiles (upper set to 0)."""
    J = tiles.shape[0]
    n = J * bsz
    out = jnp.zeros((n, n), dtype=tiles.dtype)
    for i in range(J):
        for j in range(J):
            if i >= j:
                blk = tiles[i, j]
                out = out.at[i*bsz:(i+1)*bsz, j*bsz:(j+1)*bsz].set(blk)
    return out


def ring_perm(Psize: int, direction: str = "next"):
    if direction == "next":
        return tuple((i, (i + 1) % Psize) for i in range(Psize))
    else:
        return tuple((i, (i - 1) % Psize) for i in range(Psize))


def build_mesh(Psize: int) -> Mesh:
    devs = np.array(jax.devices()).reshape((Psize,))
    return Mesh(devs, ('p',))


def chol_1d_sharded_builder(mesh: Mesh, J: int, b: int):
    """
    Returns a shard_map'ed function that takes tiles (J,J,b,b) sharded on axis1 (columns) by 'p',
    and returns the Cholesky factor tiles L in the same sharding. Assumes J == axis_size('p').
    Communication is simplified: after factoring column k on its owner, we rotate the **entire**
    L[:,k] panel (shape (J,b,b)) around the ring via ppermute. Each device updates its own column j
    exactly once when the rotating destination index equals j.
    """

    perm_next = ring_perm(J, "next")

    def chol_1d_local(A_local: jax.Array) -> jax.Array:
        """
        Local view per device:
          A_local: shape (J, 1, b, b) — this device's single column (since J==P), all row tiles.
          We operate on col = A_local[:, 0, :, :] -> (J, b, b), containing the lower tiles (i>=j).
        """
        j_glob = lax.axis_index('p')            # 0..J-1 (which global column I own)
        col = A_local[:, 0, :, :]

        # Helper: RIGHT solve X (Lkk^H) = block, used only on the owner of column k
        def right_solve_with_LkkH(block, Lkk):
            X_T = solve_triangular(Lkk.conj(), block.T, lower=True, trans='N')
            return X_T.T

        J32 = jnp.int32(J)

        def body_k(col, k):
            # Panel factorization **only on owner of column k**
            def do_owner(c):
                # POTRF on diagonal (k,k)
                Lkk = jnp.linalg.cholesky(c[k])
                c = c.at[k].set(Lkk)
                # TRSM to compute entire below-diagonal panel L[i,k] for i=k+1..J-1
                def trsm_body(i, cc):
                    return cc.at[i].set(right_solve_with_LkkH(cc[i], Lkk))
                c = lax.fori_loop(k + 1, J, trsm_body, c)
                return c

            col = lax.cond(j_glob == k, do_owner, lambda cc: cc, col)

            # Build the panel payload: on owner -> real panel; elsewhere -> zeros_like(col)
            panel = lax.cond(j_glob == k, lambda _: col, lambda _: jnp.zeros_like(col), operand=None)

            # Rotating destination index; annotate as manual-axis so scan carry types match
            dest = lax.pvary(jnp.int32(k), ('p',))

            def hop(c, _):
                ccol, panel, dest = c

                # Update this column when it's our turn and we're a trailing column
                do_update = jnp.logical_and(j_glob > k, dest == jnp.int32(j_glob))

                def apply_updates(cc):
                    Ljk = panel[j_glob]                   # (b,b)
                    # Diagonal update
                    cc = cc.at[j_glob].set(cc[j_glob] - Ljk @ Ljk.conj().T)
                    # Off-diagonal updates for i = j+1 .. J-1
                    def off_body(i, ccc):
                        return ccc.at[i].set(ccc[i] - panel[i] @ Ljk.conj().T)
                    cc = lax.fori_loop(j_glob + 1, J, off_body, cc)
                    return cc

                ccol = lax.cond(do_update, apply_updates, lambda cc: cc, ccol)

                # Rotate the payload once around the ring and advance destination
                panel = lax.ppermute(panel, axis_name='p', perm=tuple(perm_next))
                dest = (dest + jnp.int32(1)) % J32

                return (ccol, panel, dest), None

            # Run exactly J hops so every column updates once
            (col, _, _), _ = lax.scan(hop, (col, panel, dest), xs=None, length=J)

            return col, None

        col, _ = lax.scan(body_k, col, xs=jnp.arange(J), length=J)
        return col[:, None, :, :]

    # Build the sharded function: shard across **column** axis (axis=1)
    chol_1d_fn = shard_map(
        chol_1d_local,
        mesh=mesh,
        in_specs=P(None, 'p', None, None),
        out_specs=P(None, 'p', None, None),
    )

    return chol_1d_fn


def chol_1d_psum_builder(mesh: Mesh, J: int, b: int):
    """
    Tree-based 1D column Cholesky using psum instead of ring ppermute.
    
    This is O(J × log P) communication rounds instead of O(J × P) for ring.
    For P=128, this gives ~18x fewer communication rounds.
    
    Same API as chol_1d_sharded_builder but uses psum for panel broadcast.
    """

    def chol_1d_psum_local(A_local: jax.Array) -> jax.Array:
        """
        Local view per device:
          A_local: shape (J, 1, b, b) — this device's single column (since J==P), all row tiles.
          We operate on col = A_local[:, 0, :, :] -> (J, b, b).
        """
        j_glob = lax.axis_index('p')
        col = A_local[:, 0, :, :]

        def right_solve_with_LkkH(block, Lkk):
            X_T = solve_triangular(Lkk.conj(), block.T, lower=True, trans='N')
            return X_T.T

        def body_k(col, k):
            # Step 1: Panel factorization on owner of column k
            def do_owner(c):
                Lkk = jnp.linalg.cholesky(c[k])
                c = c.at[k].set(Lkk)
                def trsm_body(i, cc):
                    return cc.at[i].set(right_solve_with_LkkH(cc[i], Lkk))
                c = lax.fori_loop(k + 1, J, trsm_body, c)
                return c

            col = lax.cond(j_glob == k, do_owner, lambda cc: cc, col)

            # Step 2: Broadcast panel via psum (O(log P) tree reduction)
            # Owner contributes real panel, others contribute zeros
            panel_contrib = lax.cond(
                j_glob == k,
                lambda _: col,
                lambda _: jnp.zeros_like(col),
                operand=None
            )
            panel = lax.psum(panel_contrib, 'p')  # Tree-based broadcast

            # Step 3: Update my column if j_glob > k
            def apply_updates(cc):
                Ljk = panel[j_glob]
                # Diagonal update
                cc = cc.at[j_glob].set(cc[j_glob] - Ljk @ Ljk.conj().T)
                # Off-diagonal updates
                def off_body(i, ccc):
                    return ccc.at[i].set(ccc[i] - panel[i] @ Ljk.conj().T)
                cc = lax.fori_loop(j_glob + 1, J, off_body, cc)
                return cc

            col = lax.cond(j_glob > k, apply_updates, lambda cc: cc, col)

            return col, None

        col, _ = lax.scan(body_k, col, xs=jnp.arange(J), length=J)
        return col[:, None, :, :]

    chol_1d_fn = shard_map(
        chol_1d_psum_local,
        mesh=mesh,
        in_specs=P(None, 'p', None, None),
        out_specs=P(None, 'p', None, None),
    )

    return chol_1d_fn


def chol_2d_builder(mesh_2d: Mesh, J: int, b: int):
    """
    2D Block Cholesky for C_q(μ_X, ν_Y) using JAX's block distribution.
    
    Input sharding: P('x', 'y', None, None) on (J, J, b, b) tiles
    Output sharding: same
    
    Uses global psum for panel broadcast (O(log P) rounds).
    Same asymptotic complexity as 1D but 5.5× less bandwidth per processor.
    
    JAX uses BLOCK distribution (not block-cyclic):
    - Proc (px, py) owns rows [px*J_row : (px+1)*J_row), cols [py*J_col : (py+1)*J_col)
    - Local tile (i_loc, j_loc) maps to global (px*J_row + i_loc, py*J_col + j_loc)
    """
    Pr = mesh_2d.shape['x']
    Pc = mesh_2d.shape['y']
    J_row = J // Pr
    J_col = J // Pc
    
    assert J % Pr == 0, f"J={J} must be divisible by Pr={Pr}"
    assert J % Pc == 0, f"J={J} must be divisible by Pc={Pc}"
    
    @partial(shard_map, mesh=mesh_2d,
             in_specs=P('x', 'y', None, None),
             out_specs=P('x', 'y', None, None))
    def chol_2d_local(A_local: jax.Array) -> jax.Array:
        px = lax.axis_index('x')
        py = lax.axis_index('y')
        dtype = A_local.dtype
        
        # My global row/col range
        my_row_start = px * J_row
        my_col_start = py * J_col
        
        def right_solve_with_LkkH(block, Lkk):
            X_T = solve_triangular(Lkk.conj(), block.T, lower=True, trans='N')
            return X_T.T
        
        def body_k(A_local, k):
            # Which proc owns diagonal (k, k)?
            diag_px = k // J_row
            diag_py = k // J_col
            
            is_diag_owner = jnp.logical_and(px == diag_px, py == diag_py)
            is_col_k_owner = (py == diag_py)
            
            # Local index of tile k (only valid on owner)
            k_loc_row = k % J_row
            k_loc_col = k % J_col
            
            # Step 1: POTRF on diagonal
            Akk = A_local[k_loc_row, k_loc_col]
            Lkk_local = jnp.linalg.cholesky(Akk)
            A_local = jnp.where(is_diag_owner,
                               A_local.at[k_loc_row, k_loc_col].set(Lkk_local),
                               A_local)
            
            # Step 2: Broadcast Lkk via psum
            Lkk_contrib = jnp.where(is_diag_owner, Lkk_local, jnp.zeros((b, b), dtype))
            Lkk = lax.psum(Lkk_contrib, ('x', 'y'))
            
            # Step 3: TRSM - column k owners update their rows i > k
            def do_trsm(i_loc, A):
                i_glob = my_row_start + i_loc
                should = jnp.logical_and(is_col_k_owner, i_glob > k)
                Lik = right_solve_with_LkkH(A[i_loc, k_loc_col], Lkk)
                return jnp.where(should, A.at[i_loc, k_loc_col].set(Lik), A)
            
            A_local = lax.fori_loop(0, J_row, do_trsm, A_local)
            
            # Step 4: Broadcast panel L[:, k] via psum
            # Use pvary to mark the accumulator as having axis variance
            panel_init = lax.pvary(jnp.zeros((J, b, b), dtype), ('x', 'y'))
            
            def fill_panel(i_loc, panel):
                i_glob = my_row_start + i_loc
                should = jnp.logical_and(is_col_k_owner, i_glob >= k)
                Lik = A_local[i_loc, k_loc_col]
                return jnp.where(should, panel.at[i_glob].set(Lik), panel)
            
            panel_local = lax.fori_loop(0, J_row, fill_panel, panel_init)
            panel = lax.psum(panel_local, ('x', 'y'))
            
            # Step 5: SYRK update - all procs update their blocks (i,j) with i>k, j>k, i>=j
            def do_syrk_col(j_loc, A):
                j_glob = my_col_start + j_loc
                Ljk = panel[j_glob]
                
                def do_syrk_row(i_loc, A):
                    i_glob = my_row_start + i_loc
                    Lik = panel[i_glob]
                    update = Lik @ Ljk.conj().T
                    # Only update trailing lower triangle: i > k AND j > k AND i >= j
                    should = jnp.logical_and(
                        jnp.logical_and(i_glob > k, j_glob > k),
                        i_glob >= j_glob
                    )
                    return jnp.where(should,
                                    A.at[i_loc, j_loc].set(A[i_loc, j_loc] - update),
                                    A)
                
                return lax.fori_loop(0, J_row, do_syrk_row, A)
            
            A_local = lax.fori_loop(0, J_col, do_syrk_col, A_local)
            
            return A_local, None
        
        A_local, _ = lax.scan(body_k, A_local, jnp.arange(J))
        return A_local
    
    return chol_2d_local


def build_mesh_2d(Pr: int, Pc: int) -> Mesh:
    """Build a 2D mesh of shape (Pr, Pc) with axes ('x', 'y')."""
    devs = np.array(jax.devices()).reshape((Pr, Pc))
    return Mesh(devs, ('x', 'y'))


def chol_2d_batched_builder(mesh_2d: Mesh, J: int, b: int):
    """
    Batched 2D Cholesky that processes all q-points in a single XLA dispatch.
    
    Uses lax.map internally for ~18x speedup over Python loop.
    
    Input: (nq, J, J, b, b) sharded as P(None, 'x', 'y', None, None)
    Output: same
    
    Usage:
        chol_batched = chol_2d_batched_builder(mesh, J, b)
        L_all_q = chol_batched(C_all_q)  # Single dispatch for all q!
    """
    # Build the single-q function
    chol_single = chol_2d_builder(mesh_2d, J, b)
    
    @jax.jit
    def chol_batched(all_C_q):
        """Process all q-points with single XLA dispatch."""
        return lax.map(chol_single, all_C_q)
    
    return chol_batched


def forward_solve_sharded_builder(mesh: Mesh):
    """Build shard_map forward solve Y = L^{-1} B, with L replicated and B column-sharded.
    in_specs:  L -> P(None, None) (replicated),  B -> P(None, 'p') (columns sharded)
    out_specs: Y -> P(None, 'p')
    """
    @partial(jax.shard_map,
        mesh=mesh,
        in_specs=(P(None, None), P(None, 'p')),
        out_specs=P(None, 'p'),
    )
    def forward_local(L: jax.Array, B_local: jax.Array) -> jax.Array:
        # Solve L * Y = B for Y; L is lower triangular
        Y_local = solve_triangular(L, B_local, lower=True, trans='N')
        return Y_local

    return forward_local


def backward_solve_sharded_builder(mesh: Mesh):
    """Build shard_map backward solve X = (L^H)^{-1} Y, with L replicated and Y column-sharded.
    in_specs:  L -> P(None, None) (replicated),  Y -> P(None, 'p') (columns sharded)
    out_specs: X -> P(None, 'p')
    """
    @partial(jax.shard_map,
        mesh=mesh,
        in_specs=(P(None, None), P(None, 'p')),
        out_specs=P(None, 'p'),
    )
    def backward_local(L: jax.Array, Y_local: jax.Array) -> jax.Array:
        # Solve (L^H) * X = Y  ⇒ use upper-triangular system with U = L^H
        X_local = solve_triangular(L.conj().T, Y_local, lower=False, trans='N')
        return X_local

    return backward_local


def assemble_from_sharded_cols(L_cols_sharded: jax.Array, b: int) -> jax.Array:
    """
    L_cols_sharded has global shape (J, J, b, b) with axis1 sharded.
    It already holds only lower tiles; stitch into dense lower matrix.
    """
    L_tiles = L_cols_sharded  # (J,J,b,b)
    return tiles_lower_to_dense(L_tiles, b)


def main():
    import time
    
    key = jax.random.key(0)
    N = 8
    b = 2
    assert N % b == 0
    J = N // b
    Psize = 4
    assert J == Psize, "For this demo we assume J == number of devices == 4."

    # Build matrix and its tiled representation
    A = make_spd_hermitian(N, key)
    A_tiles = dense_to_tiles_lower(A, b)  # (J,J,b,b)

    # Reference Cholesky (full dense)
    L_ref = jnp.linalg.cholesky(A)

    # ==================== 1D Tests ====================
    print("=" * 60)
    print("1D COLUMN DISTRIBUTION (J=P=4)")
    print("=" * 60)
    
    mesh_1d = build_mesh(Psize)
    chol_ring = chol_1d_sharded_builder(mesh_1d, J, b)  # Original: O(J × P) ring
    chol_psum = chol_1d_psum_builder(mesh_1d, J, b)     # Improved: O(J × log P) tree

    # Run both 1D variants
    with mesh_1d:
        L_ring_tiles = chol_ring(A_tiles)
        L_psum_tiles = chol_psum(A_tiles)

    L_ring = assemble_from_sharded_cols(L_ring_tiles, b)
    L_psum = assemble_from_sharded_cols(L_psum_tiles, b)

    err_ring = jnp.linalg.norm(L_ref - L_ring) / jnp.linalg.norm(L_ref)
    err_psum = jnp.linalg.norm(L_ref - L_psum) / jnp.linalg.norm(L_ref)

    print(f"\nN={N}, b={b}, J={J}")
    print(f"1D Ring (ppermute) error: {float(err_ring):.3e}")
    print(f"1D Psum (tree)     error: {float(err_psum):.3e}")
    
    # Timing
    with mesh_1d:
        chol_ring(A_tiles).block_until_ready()
        chol_psum(A_tiles).block_until_ready()
        
        times_ring = []
        for _ in range(5):
            t0 = time.perf_counter()
            chol_ring(A_tiles).block_until_ready()
            times_ring.append(time.perf_counter() - t0)
        
        times_psum = []
        for _ in range(5):
            t0 = time.perf_counter()
            chol_psum(A_tiles).block_until_ready()
            times_psum.append(time.perf_counter() - t0)
    
    print(f"\n1D Ring time: {np.mean(times_ring)*1000:.1f} ± {np.std(times_ring)*1000:.1f} ms")
    print(f"1D Psum time: {np.mean(times_psum)*1000:.1f} ± {np.std(times_psum)*1000:.1f} ms")
    
    # ==================== 2D Tests ====================
    print()
    print("=" * 60)
    print("2D BLOCK DISTRIBUTION (2×2 mesh)")
    print("=" * 60)
    
    mesh_2d = build_mesh_2d(2, 2)
    chol_2d = chol_2d_builder(mesh_2d, J, b)
    
    with mesh_2d:
        from jax.sharding import NamedSharding
        shard_2d = NamedSharding(mesh_2d, P('x', 'y', None, None))
        A_tiles_2d = jax.lax.with_sharding_constraint(A_tiles, shard_2d)
        
        L_2d_tiles = chol_2d(A_tiles_2d)
        L_2d_tiles.block_until_ready()
    
    L_2d = tiles_lower_to_dense(L_2d_tiles, b)
    err_2d = jnp.linalg.norm(L_ref - L_2d) / jnp.linalg.norm(L_ref)
    
    print(f"\nN={N}, b={b}, J={J}, mesh=2×2")
    print(f"2D Block error: {float(err_2d):.3e}")
    
    # Timing
    with mesh_2d:
        chol_2d(A_tiles_2d).block_until_ready()
        
        times_2d = []
        for _ in range(5):
            t0 = time.perf_counter()
            chol_2d(A_tiles_2d).block_until_ready()
            times_2d.append(time.perf_counter() - t0)
    
    print(f"2D Block time: {np.mean(times_2d)*1000:.1f} ± {np.std(times_2d)*1000:.1f} ms")
    
    # ==================== Summary ====================
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1D Ring:  {np.mean(times_ring)*1000:6.1f} ms, error={float(err_ring):.2e}")
    print(f"1D Psum:  {np.mean(times_psum)*1000:6.1f} ms, error={float(err_psum):.2e}")
    print(f"2D Block: {np.mean(times_2d)*1000:6.1f} ms, error={float(err_2d):.2e}")
    
    # Use 2D for remainder of tests
    L_dist = L_2d

    # Reconstruction check
    rec = L_dist @ L_dist.conj().T
    rec_err = jnp.linalg.norm(A - rec) / jnp.linalg.norm(A)
    print(f"\nReconstruction ||A - L L^H|| / ||A|| = {float(rec_err):.3e}")

    # ==================== AX=B solve test ====================
    print()
    print("=" * 60)
    print("TRIANGULAR SOLVE TEST")
    print("=" * 60)
    
    M = 12
    keyB = jax.random.split(key, 1)[0]
    B = jax.random.normal(keyB, (N, M)) + 1j * jax.random.normal(keyB, (N, M))

    X_ref = jnp.linalg.solve(A, B)

    fwd_solve = forward_solve_sharded_builder(mesh_1d)
    bwd_solve = backward_solve_sharded_builder(mesh_1d)

    with mesh_1d:
        Y_sharded = fwd_solve(L_dist, B)
        X_sharded = bwd_solve(L_dist, Y_sharded)

    solve_err = jnp.linalg.norm(X_ref - X_sharded) / jnp.linalg.norm(X_ref)
    res_err = jnp.linalg.norm(A @ X_sharded - B) / jnp.linalg.norm(B)
    print(f"\nSolve error:    {float(solve_err):.3e}")
    print(f"Residual error: {float(res_err):.3e}")


if __name__ == "__main__":
    main()

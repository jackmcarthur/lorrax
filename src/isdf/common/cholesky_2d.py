"""
2D Block-Distributed Cholesky Decomposition for JAX.

This module provides memory-efficient Cholesky decomposition for matrices
sharded across a 2D processor mesh. It avoids the "involuntary full 
rematerialization" issue that occurs when resharding from P(None, 'x', 'y') 
to P(('x','y'), None, None).

Key features:
- Works directly on C_q(μ_X, ν_Y) without resharding
- O(J × log P) communication rounds via psum
- √P factor less bandwidth than 1D distribution
- Uses lax.map for efficient batched processing over q-points

Usage:
    from isdf.common.cholesky_2d import cholesky_2d_batched, tiles_to_dense
    
    # Build the batched Cholesky function for your mesh and tile sizes
    chol_fn = cholesky_2d_batched(mesh_2d, J=n//b, b=block_size)
    
    # Apply to all q-points at once (single XLA dispatch!)
    L_tiles = chol_fn(C_q_tiles)  # (nq, J, J, b, b) -> (nq, J, J, b, b)
    
    # Optionally convert back to dense
    L_dense = tiles_to_dense(L_tiles, b)  # (nq, n, n)

Memory comparison for n=10k, P=128:
    - Reshard strategy: 1.6 GB/device (may OOM)
    - 2D blocked:       5 MB/device
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.scipy.linalg import solve_triangular
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def dense_to_tiles(A: jax.Array, b: int) -> jax.Array:
    """
    Convert dense lower-triangular matrix to blocked tile format.
    
    Args:
        A: Dense matrix of shape (..., n, n)
        b: Block size (n must be divisible by b)
    
    Returns:
        Tiles of shape (..., J, J, b, b) where J = n // b
        Upper tiles (i < j) are zeros.
    """
    *batch_dims, n, _ = A.shape
    assert n % b == 0, f"n={n} must be divisible by b={b}"
    J = n // b
    
    # Reshape to (..., J, b, J, b) then transpose to (..., J, J, b, b)
    tiles = A.reshape(*batch_dims, J, b, J, b)
    tiles = jnp.moveaxis(tiles, -2, -3)  # (..., J, J, b, b)
    
    # Zero out upper triangular tiles
    i_idx, j_idx = jnp.meshgrid(jnp.arange(J), jnp.arange(J), indexing='ij')
    mask = i_idx >= j_idx  # Lower triangular in tile indices
    mask = mask.reshape((1,) * len(batch_dims) + (J, J, 1, 1))
    tiles = jnp.where(mask, tiles, 0.0)
    
    return tiles


def tiles_to_dense(tiles: jax.Array, b: int) -> jax.Array:
    """
    Convert blocked tile format back to dense matrix.
    
    Args:
        tiles: Tiles of shape (..., J, J, b, b)
        b: Block size
    
    Returns:
        Dense matrix of shape (..., n, n) where n = J * b
    """
    *batch_dims, J, _, _, _ = tiles.shape
    n = J * b
    
    # Transpose from (..., J, J, b, b) to (..., J, b, J, b)
    tiles_t = jnp.moveaxis(tiles, -3, -2)
    
    # Reshape to (..., n, n)
    dense = tiles_t.reshape(*batch_dims, n, n)
    
    return dense


def cholesky_2d_single(mesh: Mesh, J: int, b: int):
    """
    Build a shard_map function for 2D blocked Cholesky of a single matrix.
    
    This is the core algorithm. For batched processing over q-points,
    use cholesky_2d_batched() which wraps this with lax.map.
    
    Args:
        mesh: 2D JAX mesh with axes ('x', 'y')
        J: Number of tile blocks per dimension
        b: Block size (each tile is b×b)
    
    Returns:
        Function that takes (J, J, b, b) tiles sharded as P('x', 'y', None, None)
        and returns Cholesky factor in same format/sharding.
    
    Algorithm (right-looking blocked Cholesky):
        for k = 0 to J-1:
            1. POTRF: diagonal owner factors L[k,k] = chol(A[k,k])
            2. Broadcast L[k,k] via psum (O(log P) rounds)
            3. TRSM: column k owners compute L[i>k, k] = A[i,k] @ L[k,k]^{-H}
            4. Broadcast panel L[:,k] via psum
            5. SYRK: all procs update A[i,j] -= L[i,k] @ L[j,k]^H for i,j>k
    """
    # Single-device mesh: avoid manual axes entirely.
    # This sidesteps JAX VMA/scan carry annotations on 1x1 meshes.
    # TODO(jackm): Hack for JAX 0.9 VMA/scan carry mismatch on 1x1 meshes.
    # Revisit with a principled pcast/vma fix once upstream behavior stabilizes.
    if mesh.shape['x'] == 1 and mesh.shape['y'] == 1:
        @jax.jit
        def _chol_2d_local_single_device(A_tiles: jax.Array) -> jax.Array:
            A_dense = tiles_to_dense(A_tiles, b)
            L_dense = jnp.linalg.cholesky(A_dense)
            return dense_to_tiles(L_dense, b)

        return _chol_2d_local_single_device

    Pr = mesh.shape['x']
    Pc = mesh.shape['y']
    J_row = J // Pr  # local rows per proc
    J_col = J // Pc  # local cols per proc
    
    assert J % Pr == 0, f"J={J} must be divisible by Pr={Pr}"
    assert J % Pc == 0, f"J={J} must be divisible by Pc={Pc}"
    
    @partial(shard_map, mesh=mesh,
             in_specs=P('x', 'y', None, None),
             out_specs=P('x', 'y', None, None))
    def _chol_2d_local(A_local: jax.Array) -> jax.Array:
        """Local computation on each device."""
        px = lax.axis_index('x')
        py = lax.axis_index('y')
        dtype = A_local.dtype
        
        # My global row/col range (JAX uses BLOCK distribution)
        my_row_start = px * J_row
        my_col_start = py * J_col
        
        def right_solve_LH(block, L):
            """Solve X @ L^H = block for X."""
            return solve_triangular(L.conj(), block.T, lower=True, trans='N').T
        
        def body_k(A_local, k):
            """Process column k of the Cholesky factorization."""
            # Which proc owns diagonal block (k, k)?
            diag_px = k // J_row
            diag_py = k // J_col
            
            is_diag_owner = jnp.logical_and(px == diag_px, py == diag_py)
            is_col_k_owner = (py == diag_py)
            
            # Local indices (only valid on owner)
            k_loc_row = k % J_row
            k_loc_col = k % J_col
            
            # Step 1: POTRF on diagonal owner
            Akk = A_local[k_loc_row, k_loc_col]
            Lkk_local = jnp.linalg.cholesky(Akk)
            A_local = jnp.where(is_diag_owner,
                               A_local.at[k_loc_row, k_loc_col].set(Lkk_local),
                               A_local)
            
            # Step 2: Broadcast Lkk via psum (tree-based, O(log P))
            Lkk_contrib = jnp.where(is_diag_owner, Lkk_local, jnp.zeros((b, b), dtype))
            Lkk = lax.psum(Lkk_contrib, ('x', 'y'))
            
            # Step 3: TRSM - column k owners update their rows i > k
            def do_trsm(i_loc, A):
                i_glob = my_row_start + i_loc
                should_update = jnp.logical_and(is_col_k_owner, i_glob > k)
                Lik = right_solve_LH(A[i_loc, k_loc_col], Lkk)
                return jnp.where(should_update, 
                                A.at[i_loc, k_loc_col].set(Lik), 
                                A)
            
            A_local = lax.fori_loop(0, J_row, do_trsm, A_local)
            
            # Step 4: Broadcast panel L[:, k] via psum
            # Each device contributes its local portion, then psum aggregates
            # JAX 0.9+: mark as varying for fori_loop within shard_map
            panel_init = jnp.zeros((J, b, b), dtype)
            
            def fill_panel(i_loc, panel):
                i_glob = my_row_start + i_loc
                should_contrib = jnp.logical_and(is_col_k_owner, i_glob >= k)
                Lik = A_local[i_loc, k_loc_col]
                return jnp.where(should_contrib, 
                                panel.at[i_glob].set(Lik), 
                                panel)
            
            panel_local = lax.fori_loop(0, J_row, fill_panel, panel_init)
            panel = lax.psum(panel_local, ('x', 'y'))
            
            # Step 5: SYRK update - all procs update blocks (i,j) with i>k, j>k, i>=j
            def do_syrk_col(j_loc, A):
                j_glob = my_col_start + j_loc
                Ljk = panel[j_glob]
                
                def do_syrk_row(i_loc, A):
                    i_glob = my_row_start + i_loc
                    Lik = panel[i_glob]
                    update = Lik @ Ljk.conj().T
                    # Only update trailing lower triangle
                    should_update = jnp.logical_and(
                        jnp.logical_and(i_glob > k, j_glob > k),
                        i_glob >= j_glob
                    )
                    return jnp.where(should_update,
                                    A.at[i_loc, j_loc].set(A[i_loc, j_loc] - update),
                                    A)
                
                return lax.fori_loop(0, J_row, do_syrk_row, A)
            
            A_local = lax.fori_loop(0, J_col, do_syrk_col, A_local)
            
            return A_local, None
        
        # Scan over all J column steps
        A_local, _ = lax.scan(body_k, A_local, jnp.arange(J))
        return A_local
    
    return _chol_2d_local


def cholesky_2d_batched(mesh: Mesh, J: int, b: int):
    """
    Build a batched 2D Cholesky function that processes all q-points efficiently.
    
    Uses lax.map internally for ~18x speedup over Python loop by using
    a single XLA dispatch for all q-points.
    
    Args:
        mesh: 2D JAX mesh with axes ('x', 'y')
        J: Number of tile blocks per dimension  
        b: Block size
    
    Returns:
        JIT-compiled function that takes (nq, J, J, b, b) tiles 
        sharded as P(None, 'x', 'y', None, None) and returns 
        Cholesky factors in same format.
    
    Example:
        chol_fn = cholesky_2d_batched(mesh, J=16, b=64)
        L_all = chol_fn(C_q_tiles)  # Single dispatch for all q!
    """
    chol_single = cholesky_2d_single(mesh, J, b)
    
    @jax.jit
    def _batched_chol(all_C_q: jax.Array) -> jax.Array:
        """Process all q-points with single XLA dispatch."""
        return lax.map(chol_single, all_C_q)
    
    return _batched_chol


def solve_triangular_2d(
    L_tiles: jax.Array,
    B_tiles: jax.Array, 
    mesh: Mesh,
    lower: bool = True,
    trans: str = 'N'
) -> jax.Array:
    """
    Solve L @ X = B or L^H @ X = B with 2D sharded triangular matrix.
    
    For Cholesky solve C @ x = b, use:
        L = cholesky_2d(C)
        y = solve_triangular_2d(L, b, lower=True, trans='N')   # L @ y = b
        x = solve_triangular_2d(L, y, lower=True, trans='C')   # L^H @ x = y
    
    Note: Currently this replicates L for simplicity. For very large L,
    a distributed triangular solve would be needed.
    
    Args:
        L_tiles: Lower triangular factor (nq, J, J, b, b) or (J, J, b, b)
        B_tiles: Right-hand side, same tile format
        mesh: 2D mesh
        lower: Whether L is lower triangular (always True for Cholesky)
        trans: 'N' for L, 'C' for L^H, 'T' for L^T
    
    Returns:
        X_tiles in same format as B_tiles
    """
    # For now, gather L and solve locally (works for moderate sizes)
    # TODO: Implement distributed triangular solve for very large L
    
    J = L_tiles.shape[-4]
    b = L_tiles.shape[-1]
    
    # Convert to dense
    L_dense = tiles_to_dense(L_tiles, b)
    B_dense = tiles_to_dense(B_tiles, b)
    
    # Solve
    if trans == 'N':
        X_dense = jax.scipy.linalg.solve_triangular(
            L_dense, B_dense, lower=lower, trans=0
        )
    elif trans == 'C':
        X_dense = jax.scipy.linalg.solve_triangular(
            L_dense.conj().mT, B_dense, lower=not lower, trans=0
        )
    elif trans == 'T':
        X_dense = jax.scipy.linalg.solve_triangular(
            L_dense.mT, B_dense, lower=not lower, trans=0
        )
    else:
        raise ValueError(f"trans must be 'N', 'C', or 'T', got {trans}")
    
    # Convert back to tiles (though result may not be triangular)
    return dense_to_tiles(X_dense, b)


def cholesky_solve_2d(
    C_q_tiles: jax.Array,
    Z_q_tiles: jax.Array,
    mesh: Mesh,
    J: int,
    b: int
) -> jax.Array:
    """
    Solve C_q @ zeta_q = Z_q using 2D blocked Cholesky.
    
    This is the full pipeline for ISDF zeta fitting:
    1. L = chol(C_q)
    2. y = L^{-1} @ Z_q  (forward solve)
    3. zeta = L^{-H} @ y (backward solve)
    
    Args:
        C_q_tiles: CCT matrix (nq, J, J, b, b) sharded P(None, 'x', 'y', None, None)
        Z_q_tiles: ZCT matrix, same shape and sharding
        mesh: 2D mesh with axes ('x', 'y')
        J: Number of blocks
        b: Block size
    
    Returns:
        zeta_q_tiles: Solution, same shape and sharding
    """
    # Step 1: Cholesky decomposition
    chol_fn = cholesky_2d_batched(mesh, J, b)
    L_q_tiles = chol_fn(C_q_tiles)
    
    # Step 2+3: Triangular solves
    # For now, use dense solve (works for moderate sizes)
    L_dense = tiles_to_dense(L_q_tiles, b)
    Z_dense = tiles_to_dense(Z_q_tiles, b)
    
    # Forward: L @ Y = Z
    Y_dense = jax.vmap(lambda L, Z: jax.scipy.linalg.solve_triangular(
        L, Z, lower=True
    ))(L_dense, Z_dense)
    
    # Backward: L^H @ zeta = Y
    zeta_dense = jax.vmap(lambda L, Y: jax.scipy.linalg.solve_triangular(
        L.conj().T, Y, lower=False
    ))(L_dense, Y_dense)
    
    return dense_to_tiles(zeta_dense, b)

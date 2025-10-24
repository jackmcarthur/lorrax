#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharded 1-D column-wise Cholesky using shard_map + ppermute ring collectives.
- 4 logical host devices
- N=8, block size b=2 (J=4)
- Compares distributed result to dense reference

This version fixes:
- in_specs/out_specs usage (call-style shard_map)
- proper (J,J,b,b) tiling with column axis=1 sharded
- cond branch manual-axis mismatches via zeros_like
- ring carry type mismatches by removing the extra flags and using a rotating
  integer destination "dest" instead of bool markers
- simplified ring: rotate the **entire panel** L[:,k] (shape (J,b,b))
  and update when dest == j_glob
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
from jax.experimental.shard_map import shard_map
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

    # Build mesh and sharded Cholesky
    mesh = build_mesh(Psize)
    chol_sharded = chol_1d_sharded_builder(mesh, J, b)

    # Run distributed Cholesky
    with mesh:
        # The input A_tiles will be sharded on axis1 across the mesh by shard_map
        L_cols_sharded = chol_sharded(A_tiles)  # (J,J,b,b) sharded on axis1

    # Assemble dense L from sharded tiles
    L_dist = assemble_from_sharded_cols(L_cols_sharded, b)

    # Compare
    err_fro = jnp.linalg.norm(L_ref - L_dist)
    rel_err = err_fro / jnp.linalg.norm(L_ref)

    print("A (Hermitian SPD) =\n", np.array(A))
    print("\nReference L (jnp.linalg.cholesky) =\n", np.array(L_ref))
    print("\nDistributed L (shard_map + ppermute ring) =\n", np.array(L_dist))
    print(f"\n||L_ref - L_dist||_F = {float(err_fro):.3e}   (relative {float(rel_err):.3e})")

    # Quick correctness sanity: A ≈ L L^H
    rec = L_dist @ L_dist.conj().T
    rec_err = jnp.linalg.norm(A - rec) / jnp.linalg.norm(A)
    print(f"Relative reconstruction error ||A - L L^H|| / ||A|| = {float(rec_err):.3e}")

    # ======= AX=B solve test: column-sharded B with replicated L =======
    M = 12  # choose M divisible by Psize so axis-1 sharding works cleanly
    keyB = jax.random.split(key, 1)[0]
    B = jax.random.normal(keyB, (N, M)) + 1j * jax.random.normal(keyB, (N, M))

    # Reference solution on dense A
    X_ref = jnp.linalg.solve(A, B)

    # Build shard-mapped forward and backward solves (replicated L, sharded B/Y/X)
    fwd_solve = forward_solve_sharded_builder(mesh)
    bwd_solve = backward_solve_sharded_builder(mesh)

    with mesh:
        # B is automatically sharded along axis-1 by in_specs (None, 'p')
        Y_sharded = fwd_solve(L_dist, B)
        X_sharded = bwd_solve(L_dist, Y_sharded)

    # Compare solutions
    solve_err = jnp.linalg.norm(X_ref - X_sharded) / jnp.linalg.norm(X_ref)
    res_err = jnp.linalg.norm(A @ X_sharded - B) / jnp.linalg.norm(B)
    print(f"\nAX=B solve test:")
    print(f"Solve error ||X_ref - X||/||X_ref|| = {float(solve_err):.3e}")
    print(f"Residual error ||A X - B||/||B||  = {float(res_err):.3e}")


if __name__ == "__main__":
    main()

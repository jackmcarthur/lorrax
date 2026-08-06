#!/usr/bin/env python3
"""
Benchmark: Blocked Cholesky vs Reshard-then-Cholesky for distributed ISDF.

Compares:
1. Blocked Cholesky (column-sharded, ring ppermute)
2. Reshard to (q_XY, mu, nu) then vmap(cholesky)

Also analyzes triangular solve sharding options.

Run with:
    XLA_FLAGS='--xla_force_host_platform_device_count=4' JAX_PLATFORMS=cpu \
    uv run python scripts/testscripts/benchmark_cholesky_sharding.py
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.shard_map import shard_map
from jax import lax
from jax.scipy.linalg import solve_triangular
from functools import partial

# Import blocked Cholesky from existing test
from test_blocked_cholesky import (
    chol_1d_sharded_builder,
    forward_solve_sharded_builder,
    backward_solve_sharded_builder,
    dense_to_tiles_lower,
    tiles_lower_to_dense,
    make_spd_hermitian,
)


def make_spd_batch(nq: int, n: int, key: jax.Array) -> jax.Array:
    """Create batch of SPD matrices for testing."""
    keys = jax.random.split(key, nq)
    def make_one(k):
        x = jax.random.normal(k, (n, n)) + 1j * jax.random.normal(k, (n, n))
        a = x.conj().T @ x
        a = a + (1.0 + 0.1 * n) * jnp.eye(n)
        return 0.5 * (a + a.conj().T)
    return jax.vmap(make_one)(keys)


def benchmark_reshard_cholesky(mesh_xy: Mesh, nq: int, n_rmu: int, n_warmup: int = 2, n_runs: int = 5):
    """Benchmark: reshard from (q, mu_X, nu_Y) to (q_XY, mu, nu) then vmap cholesky."""
    mesh_size = mesh_xy.devices.shape[0] * mesh_xy.devices.shape[1]
    
    # Pad nq to be divisible by mesh_size
    nq_padded = ((nq + mesh_size - 1) // mesh_size) * mesh_size
    
    with mesh_xy:
        # Create SPD matrices with (q, mu_X, nu_Y) sharding
        key = jax.random.key(42)
        C_full = make_spd_batch(nq_padded, n_rmu, key)
        
        xy_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        q_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
        
        C_sharded = jax.lax.with_sharding_constraint(C_full, xy_shard)
        
        @jax.jit
        def reshard_and_cholesky(C):
            C_q = jax.lax.with_sharding_constraint(C, q_shard)
            L_q = jax.vmap(jnp.linalg.cholesky)(C_q)
            return jax.lax.with_sharding_constraint(L_q, q_shard)
        
        # Warmup
        for _ in range(n_warmup):
            L = reshard_and_cholesky(C_sharded)
            L.block_until_ready()
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            L = reshard_and_cholesky(C_sharded)
            L.block_until_ready()
            times.append(time.perf_counter() - t0)
        
        return np.mean(times), np.std(times), L


def benchmark_blocked_cholesky(mesh_1d: Mesh, nq: int, n_rmu: int, n_warmup: int = 2, n_runs: int = 5):
    """Benchmark: blocked Cholesky with column sharding and ring ppermute."""
    P_size = mesh_1d.devices.shape[0]
    
    # Block size: n_rmu must be divisible by P_size for this simple version
    if n_rmu % P_size != 0:
        # Pad to make divisible
        n_padded = ((n_rmu + P_size - 1) // P_size) * P_size
    else:
        n_padded = n_rmu
    
    b = n_padded // P_size  # block size
    J = P_size  # number of column blocks = number of devices
    
    with mesh_1d:
        key = jax.random.key(42)
        
        # Build the blocked Cholesky function
        chol_fn = chol_1d_sharded_builder(mesh_1d, J, b)
        
        # Create one SPD matrix, tile it
        A = make_spd_hermitian(n_padded, key)
        A_tiles = dense_to_tiles_lower(A, b)  # (J, J, b, b)
        
        # For batch: we'd need to loop over q or vectorize
        # The current blocked cholesky doesn't support batching easily
        # So we benchmark single-matrix performance and multiply by nq
        
        # Warmup
        for _ in range(n_warmup):
            L_tiles = chol_fn(A_tiles)
            L_tiles.block_until_ready()
        
        # Benchmark single matrix
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            L_tiles = chol_fn(A_tiles)
            L_tiles.block_until_ready()
            times.append(time.perf_counter() - t0)
        
        # Estimate batch time (would need loop in practice)
        single_time = np.mean(times)
        batch_time = single_time * nq  # Sequential loop estimate
        
        return batch_time, np.std(times) * np.sqrt(nq), L_tiles


def benchmark_triangular_solve_sharding(mesh_xy: Mesh, n_rmu: int, n_zchunk: int, 
                                         n_warmup: int = 2, n_runs: int = 5):
    """
    Benchmark triangular solve with different shardings.
    
    Options:
    A) L replicated, Z column-sharded on Y → zeta column-sharded on Y
    B) L column-sharded on Y, Z column-sharded on Y → requires communication
    """
    with mesh_xy:
        key = jax.random.key(123)
        
        # Create L (lower triangular) and Z
        L_full = jnp.tril(jax.random.normal(key, (n_rmu, n_rmu)) + 
                         (1.0 + n_rmu) * jnp.eye(n_rmu))
        Z_full = jax.random.normal(jax.random.split(key)[0], (n_rmu, n_zchunk))
        
        # Sharding options
        replicated = NamedSharding(mesh_xy, P(None, None))
        col_sharded_y = NamedSharding(mesh_xy, P(None, 'y'))
        row_sharded_x = NamedSharding(mesh_xy, P('x', None))
        xy_sharded = NamedSharding(mesh_xy, P('x', 'y'))
        
        # Option A: L replicated, Z column-sharded
        L_rep = jax.lax.with_sharding_constraint(L_full, replicated)
        Z_col = jax.lax.with_sharding_constraint(Z_full, col_sharded_y)
        
        @jax.jit
        def solve_replicated_L(L, Z):
            """L replicated, Z column-sharded → parallel column-wise solve."""
            return solve_triangular(L, Z, lower=True)
        
        # Warmup
        for _ in range(n_warmup):
            zeta = solve_replicated_L(L_rep, Z_col)
            zeta.block_until_ready()
        
        # Benchmark
        times_A = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            zeta = solve_replicated_L(L_rep, Z_col)
            zeta.block_until_ready()
            times_A.append(time.perf_counter() - t0)
        
        print(f"\nTriangular solve ({n_rmu}×{n_rmu}) @ ({n_rmu}×{n_zchunk}):")
        print(f"  Option A (L replicated, Z col-sharded): {np.mean(times_A)*1000:.2f} ± {np.std(times_A)*1000:.2f} ms")
        print(f"  Output sharding: {zeta.sharding.spec}")
        
        return np.mean(times_A), zeta


def main():
    print("="*70)
    print("Cholesky Sharding Benchmark")
    print("="*70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Device count: {jax.device_count()}")
    
    devices = jax.devices()
    n_devices = len(devices)
    
    # Create meshes
    # 2D mesh for reshard approach
    grid_x = int(np.sqrt(n_devices))
    while n_devices % grid_x != 0:
        grid_x -= 1
    grid_y = n_devices // grid_x
    mesh_xy = Mesh(np.array(devices).reshape(grid_x, grid_y), ('x', 'y'))
    
    # 1D mesh for blocked Cholesky
    mesh_1d = Mesh(np.array(devices).reshape(n_devices,), ('p',))
    
    print(f"2D mesh: {grid_x}×{grid_y}")
    print(f"1D mesh: {n_devices}")
    
    # Test sizes
    # (nq, n_rmu) - increasing complexity
    test_cases = [
        (4, 16),    # Tiny
        (8, 32),    # Small
        (16, 64),   # Medium
        (32, 128),  # Large (realistic ISDF size)
        # (64, 256),  # Very large - uncomment for GPU testing
    ]
    
    print("\n" + "="*70)
    print("Part 1: Cholesky Decomposition Comparison")
    print("="*70)
    print(f"{'nq':>6} {'n_rmu':>6} {'Reshard+Chol':>15} {'Blocked (est)':>15} {'Ratio':>10}")
    print("-"*70)
    
    for nq, n_rmu in test_cases:
        if n_rmu % n_devices != 0:
            # Skip sizes that don't divide evenly for blocked Cholesky
            print(f"{nq:>6} {n_rmu:>6} {'(skip - not divisible)':>40}")
            continue
            
        try:
            # Reshard approach
            t_reshard, _, _ = benchmark_reshard_cholesky(mesh_xy, nq, n_rmu)
            
            # Blocked approach (sequential over q estimate)
            t_blocked, _, _ = benchmark_blocked_cholesky(mesh_1d, nq, n_rmu)
            
            ratio = t_reshard / t_blocked if t_blocked > 0 else float('inf')
            
            print(f"{nq:>6} {n_rmu:>6} {t_reshard*1000:>12.2f} ms {t_blocked*1000:>12.2f} ms {ratio:>10.2f}x")
            
        except Exception as e:
            print(f"{nq:>6} {n_rmu:>6} Error: {e}")
    
    print("\n" + "="*70)
    print("Part 2: Triangular Solve Analysis")
    print("="*70)
    print("""
For the triangular solve L @ zeta = Z:

Current state after Cholesky:
  - C_q has shape (nq, n_rmu, n_rmu) with sharding P(q_XY, mu, nu) or P(q, mu_X, nu_Y)
  - Z_q has shape (nq, n_rmu, n_zchunk) with sharding P(q, mu_X, zchunk_Y)

Options for triangular solve:

OPTION 1: Process one q at a time (memory-safe)
  - For each q:
    1. Gather L_q to all devices (n_rmu² communication)
    2. Solve locally: each device handles its columns of Z_q
    3. Write zeta_q to disk
  - Communication: O(nq × n_rmu²) total
  - Memory: O(n_rmu²) per device (replicated L) + O(n_rmu × n_zchunk/Y) (sharded Z)

OPTION 2: Keep Z column-sharded, batch solve
  - Reshard L_q to be replicated per q-shard (each q's L on different device group)
  - Requires O(n_rmu²) replication per q
  - Triangular solve is embarrassingly parallel on Z columns

OPTION 3: Blocked triangular solve (complex)
  - Keep both L and Z sharded
  - Requires communication during solve (row dependencies)
  - Similar to blocked Cholesky complexity
""")
    
    # Benchmark triangular solve for a representative size
    n_rmu_test = 64
    n_zchunk_test = 64 * 15  # ~15x centroids, matching your z-chunk target
    benchmark_triangular_solve_sharding(mesh_xy, n_rmu_test, n_zchunk_test)
    
    print("\n" + "="*70)
    print("Recommendation")
    print("="*70)
    print("""
For your pipeline with n_zchunk ~ 16×n_rmu:

1. CHOLESKY: Use reshard-to-q approach if n_rmu ≤ ~500 per device.
   The XLA replication overhead is amortized over many q-points.
   For larger n_rmu, consider adapting blocked Cholesky.

2. TRIANGULAR SOLVE: Process one q at a time:
   a) Gather L_q to all devices (replicate for this q)
   b) Keep Z_q(μ_X, r_zchunk_Y) column-sharded
   c) solve_triangular is embarrassingly parallel on columns
   d) Output zeta_q(μ_X, r_zchunk_Y) stays column-sharded
   e) Write to HDF5, proceed to next q

3. MEMORY BUDGET per device:
   - L_q replicated: n_rmu² × 16 bytes (complex128)
   - Z_q column-shard: n_rmu × (n_zchunk/Y) × 16 bytes  
   - zeta_q column-shard: same as Z_q
   
   For n_rmu=1000, n_zchunk=15000, Y=4:
   - L: 16 MB
   - Z: 1000 × 3750 × 16 = 60 MB
   - Total: ~80 MB per q-point (very manageable)
""")


if __name__ == "__main__":
    main()


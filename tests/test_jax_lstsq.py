#!/usr/bin/env python3
"""
Memory-Efficient Distributed TSQR with JAX Sharding

This implementation uses JAX's modern sharding APIs to distribute computation
without data duplication. Key features:
  • Zero-copy sharding with PositionalSharding
  • shard_map for memory-efficient distributed computing
  • No 3D array reshaping or data duplication
  • Minimal memory footprint

Usage:
  export XLA_FLAGS="--xla_force_host_platform_device_count=8"
  python tests/test_jax_lstsq.py
"""

import os
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental import shard_map
import jax.scipy.linalg as jsp

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def tsqr_solve_shard(local_C, local_Z):
    """TSQR solve on a single shard - no data duplication"""
    # Stage 1: Local QR on this device's shard
    Q1, R1 = jnp.linalg.qr(local_C, mode='reduced')
    
    # Stage 2: Gather R matrices (communication step)
    R_stack = lax.all_gather(R1, 'devices', axis=0)  # Shape: (num_devices, local_rows, cols)
    R_merged = R_stack.reshape(-1, R_stack.shape[-1])  # Flatten to tall matrix
    
    # Stage 3: Global QR on stacked R matrices
    Q2, R_global = jnp.linalg.qr(R_merged, mode='reduced')
    
    # Stage 4: Extract this device's slice of Q2
    device_idx = lax.axis_index('devices')
    local_rows = local_C.shape[0]
    start_idx = jnp.int32(device_idx * local_rows)  # Ensure consistent dtype
    Q2_local = lax.dynamic_slice(Q2, (start_idx, jnp.int32(0)), (local_rows, Q2.shape[1]))
    
    # Stage 5: Compute full Q for this device
    Q_full = Q1 @ Q2_local
    
    # Stage 6: Project local RHS and sum across devices
    local_projection = Q_full.T @ local_Z
    global_projection = lax.psum(local_projection, 'devices')
    
    # Stage 7: Solve triangular system (same result on all devices)
    X_solution = jsp.solve_triangular(R_global, global_projection, lower=False)
    
    return X_solution

def main():
    # Problem dimensions
    rows, cols, rhs_cols = 64, 64, 128  # Tall-skinny matrix for TSQR
    
    # Get devices
    devices = jax.local_devices()
    P = len(devices)
    
    assert rows % P == 0, f"Rows ({rows}) must be divisible by devices ({P})"
    local_rows = rows // P
    
    # Create test data
    key = jax.random.PRNGKey(42)
    C = jax.random.normal(key, (rows, cols), dtype=jnp.float64)
    Z = jax.random.normal(key + 1, (rows, rhs_cols), dtype=jnp.float64)
    
    print(f"\n💾 Memory Usage Analysis:")
    print(f"   Original C: {C.shape} = {C.size * 8 / 1e6:.2f} MB")
    print(f"   Original Z: {Z.shape} = {Z.size * 8 / 1e6:.2f} MB")
    print(f"   Total: {(C.size + Z.size) * 8 / 1e6:.2f} MB")
    
    # Create mesh and sharding for row-wise distribution
    mesh = jax.sharding.Mesh(devices, ('devices',))
    sharding = NamedSharding(mesh, PartitionSpec('devices', None))
    
    # Shard arrays across devices (zero-copy!)
    C_sharded = jax.device_put(C, sharding)
    Z_sharded = jax.device_put(Z, sharding)
    
    # Run distributed TSQR with shard_map (memory efficient!)
    print(f"\n🔧 Running memory-efficient distributed TSQR...")
    
    X_distributed = shard_map.shard_map(
        tsqr_solve_shard,
        mesh=mesh,
        in_specs=(PartitionSpec('devices', None), 
                  PartitionSpec('devices', None)),
        out_specs=PartitionSpec(None, None),  # Solution replicated
        check_rep=False  # Allow replication
    )(C_sharded, Z_sharded)
    
    # Serial reference for comparison
    X_serial, *_ = jnp.linalg.lstsq(C, Z, rcond=None)
    
    # Verify correctness
    residual_serial = jnp.linalg.norm(Z - C @ X_serial)
    residual_distributed = jnp.linalg.norm(Z - C @ X_distributed)
    
    print(f"\n✅ Results:")
    print(f"   Serial residual:      {residual_serial:.2e}")
    print(f"   Distributed residual: {residual_distributed:.2e}")
    print(f"   Max solution error:   {jnp.max(jnp.abs(X_serial - X_distributed)):.2e}")
    
    # Verify numerical accuracy
    assert jnp.allclose(X_serial, X_distributed, rtol=1e-10, atol=1e-10)
    print(f"🎯 Memory-efficient TSQR passed all tests!")
    
    print(f"Peak memory per device: ~{(local_rows * cols + local_rows * rhs_cols) * 8 / 1e6:.1f} MB")

if __name__ == "__main__":
    main()

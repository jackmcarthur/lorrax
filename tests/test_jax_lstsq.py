#!/usr/bin/env python3
"""
Distributed TSQR Least‑Squares Solve in JAX (pmap) with Uneven Distribution

Solves C · X = Z via a two‑stage TSQR that works with ANY number of devices,
even when the number of rows doesn't divide evenly. Features:

• Automatic load balancing: distributes rows as evenly as possible
• Zero-padding strategy: pads short arrays to enable pmap
• Masking during computation: ignores padded regions  
• Works with any matrix size and device count

Example: 67 rows on 8 devices → [9,9,9,8,8,8,8,8] rows per device

Usage:
  export XLA_FLAGS="--xla_force_host_platform_device_count=8" 
  export JAX_ENABLE_X64=1
  python test_tsqr_lstsq.py
"""

import os
import pytest
import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy.linalg as jsp

# Ensure these are set before JAX initialization
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=7"
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_platform_name", "cpu")

# 1) Enable 64‑bit
jax.config.update("jax_enable_x64", True)

def tsqr_least_squares(local_C, local_Z, actual_rows):
    """
    JIT-optimized TSQR-based least squares on a local block with uneven sharding.
    Returns solution X after global triangular solve.
    """
    max_rpd, n = local_C.shape
    
    # Create row mask once and apply to both inputs
    mask = (jnp.arange(max_rpd) < actual_rows)[:, None]  # (max_rpd, 1) for broadcasting
    masked_C = local_C * mask
    masked_Z = local_Z * mask
    
    # Stage 1: Local QR on masked data
    Q1, R1 = jnp.linalg.qr(masked_C, mode='reduced')
    
    # Stage 2: Gather R factors and do global QR  
    R_stacked = lax.all_gather(R1, 'i').reshape(-1, n)  # Gather and flatten in one step
    Q2_full, R_global = jnp.linalg.qr(R_stacked, mode='reduced')
    
    # Stage 3: Extract this device's Q2 slice and compute combined Q
    device_start = jnp.int32(lax.axis_index('i') * R1.shape[0])  # Use R1.shape[0] directly
    Q2_slice = lax.dynamic_slice(Q2_full, (device_start, jnp.int32(0)), R1.shape)  # Same shape as R1
    Q_combined = (Q1 * mask) @ Q2_slice  # Apply mask and compute in one step
    
    # Stage 4: Project RHS globally and solve
    z_global = lax.psum(Q_combined.T @ masked_Z, 'i')
    return jsp.solve_triangular(R_global, z_global, lower=False)

def main():
    # Test multiple challenging cases
    test_cases = [
        (64, 64, 128),  # 67 doesn't divide evenly by 8 devices
        (100, 32, 50),  # 100 rows → [13,13,13,13,12,12,12,12] distribution  
        (15, 8, 10),    # Very small problem
        (200, 16, 75),  # Larger problem
    ]
    
    for case_i, (rows, cols, rhs) in enumerate(test_cases):
        print(f"\n🧪 Test Case {case_i+1}: {rows}×{cols} @ X = {rows}×{rhs}")
        test_single_case(rows, cols, rhs)

@pytest.mark.parametrize("rows,cols,rhs", [(15, 8, 10)])
def test_single_case(rows, cols, rhs):
    """Test a single problem size"""
    # Discover devices
    devices = jax.local_devices()
    P = len(devices)
    print(f"   🔧 Solving on {P} devices")

    # Calculate uneven distribution
    base_rpd = rows // P  # Base rows per device
    extra_rows = rows % P  # Some devices get one extra row
    
    # Device i gets: base_rpd + (1 if i < extra_rows else 0) rows
    rows_per_device = [base_rpd + (1 if i < extra_rows else 0) for i in range(P)]
    max_rpd = max(rows_per_device)  # For padding
    
    print(f"   📊 Row distribution: {rows_per_device} (max {max_rpd}/device)")

    # Random data
    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    C = jax.random.normal(key,      (rows, cols),  dtype=jnp.float64)
    Z = jax.random.normal(key+1,    (rows, rhs),   dtype=jnp.float64)

    # Serial reference
    X_ref, *_ = jnp.linalg.lstsq(C, Z, rcond=None)

    # Shard data manually according to rows_per_device
    offsets = [sum(rows_per_device[:i]) for i in range(P)]
    C_sh = jnp.stack([
        jnp.pad(
            C[offset:offset + rows_per_device[i]],
            ((0, max_rpd - rows_per_device[i]), (0, 0))
        )
        for i, offset in enumerate(offsets)
    ])
    Z_sh = jnp.stack([
        jnp.pad(
            Z[offset:offset + rows_per_device[i]],
            ((0, max_rpd - rows_per_device[i]), (0, 0))
        )
        for i, offset in enumerate(offsets)
    ])
    actual_rows_sh = jnp.array(rows_per_device)

    # PMAP the TSQR least‑squares solve with actual row counts
    X_sh = jax.pmap(tsqr_least_squares, axis_name='i')(C_sh, Z_sh, actual_rows_sh)  # (P, cols, rhs)
    X_tsqr = X_sh[0]  # all devices get the same X_local

    # Compare residuals
    res_ref  = jnp.linalg.norm(Z - C @ X_ref)
    res_tsqr = jnp.linalg.norm(Z - C @ X_tsqr)

    print(f"   📈 Serial residual:      {res_ref:.3e}")
    print(f"   📈 Distributed residual: {res_tsqr:.3e}")
    
    try:
        assert jnp.allclose(res_ref, res_tsqr, rtol=1e-3)
        print("   ✅ PASSED - Matches serial solution!")
    except AssertionError:
        print(f"   ❌ FAILED - Residuals differ too much")
        raise

if __name__ == "__main__":
    main()


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
# Ensure these are set before JAX initialization
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=7")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy.linalg as jsp

# 1) Enable 64‑bit
jax.config.update("jax_enable_x64", True)

def tsqr_least_squares(local_C, local_Z, actual_rows):
    """
    Performs TSQR-based least squares on a local block with uneven sharding:
      local_C : (max_rows_per_dev, n) - may be padded with zeros  
      local_Z : (max_rows_per_dev, m) - may be padded with zeros
      actual_rows : scalar - actual number of rows on this device (before padding)
    Returns local piece of X (n×m), after a global triangular solve.
    """
    max_rpd, n = local_C.shape
    _, m = local_Z.shape
    
    # Create mask for actual vs padded rows
    row_indices = jnp.arange(max_rpd)
    mask = row_indices < actual_rows  # (max_rpd,) boolean mask
    
    # Apply mask to get effective data (masked rows become zero)
    masked_C = local_C * mask[:, None]  # (max_rpd, n)
    masked_Z = local_Z * mask[:, None]  # (max_rpd, m)
    
    # Stage 1: QR on masked data (zeros don't affect the result much)
    Q1, R1 = jnp.linalg.qr(masked_C, mode='reduced')   # Q1: (max_rpd, min(max_rpd,n)), R1: (min(max_rpd,n), n)

    # Stage 2: gather R1 across devices → flatten → QR with padding
    R_stack = lax.all_gather(R1, 'i')                  # (P, max_rpd, n)
    all_actual_rows = lax.all_gather(actual_rows, 'i') # (P,) - actual rows per device
    
    # Simple approach: flatten and do QR on padded matrix
    # The zero-padded rows won't affect the R factor significantly
    P, max_rpd_gathered, n = R_stack.shape
    device_idx = lax.axis_index('i')
    R_flat = R_stack.reshape(-1, n)                    # (P*max_rpd, n)
    
    # Do QR on the full (padded) matrix - padding is zeros so won't affect much
    Q2_full, R_global = jnp.linalg.qr(R_flat, mode='reduced')  # Q2: (P*max_rpd, n), R: (n, n)

    # Find this device's slice in Q2_full  
    start_idx = jnp.int32(device_idx * max_rpd_gathered)
    Q2_local = lax.dynamic_slice(Q2_full, (start_idx, jnp.int32(0)), (max_rpd, n))
    
    # Get the actual dimensions after reduced QR
    qr_rank = min(max_rpd, n)  # This is Q1.shape[1] and should be R1.shape[0]
    
    # Q2_local corresponds to R1 factors, so we need the right slice size
    Q2_slice = lax.dynamic_slice(Q2_full, (start_idx, jnp.int32(0)), (qr_rank, n))  # (qr_rank, n)
    
    # Apply masking (Q1 already has correct shape from reduced QR)  
    Q1_masked = Q1 * mask[:, None]                     # (max_rpd, qr_rank)
    
    # Compute full Q for this device: Q1 @ Q2_slice
    # Q1_masked: (max_rpd, qr_rank), Q2_slice: (qr_rank, n) -> (max_rpd, n)
    Q_full = Q1_masked @ Q2_slice                       # (max_rpd, n)

    # Project Z and sum for Q^T Z (with masking)
    z_local = Q_full.T @ masked_Z                       # (n, m)
    z_global = lax.psum(z_local, 'i')                   # (n, m)

    # Solve R_global · X = Q^T Z
    X_local = jsp.solve_triangular(R_global, z_global, lower=False)
    return X_local                                      # (n, m)

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
    
    print(f"\n🎯 ALL TESTS PASSED! 🎯")
    print(f"✨ Successfully demonstrated uneven distribution TSQR:")
    print(f"   • Works with ANY number of rows and devices")
    print(f"   • Automatic load balancing across devices") 
    print(f"   • Zero-padding strategy with masking")
    print(f"   • Excellent numerical accuracy maintained")
    print(f"   • JAX-compatible distributed computing")

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

    # Shard C and Z by rows with padding for uneven distribution
    C_sh = jnp.zeros((P, max_rpd, cols), dtype=jnp.float64)
    Z_sh = jnp.zeros((P, max_rpd, rhs), dtype=jnp.float64)
    actual_rows_sh = jnp.array(rows_per_device)
    
    # Fill each device's shard with actual data
    start_row = 0
    for dev_i in range(P):
        end_row = start_row + rows_per_device[dev_i]
        C_sh = C_sh.at[dev_i, :rows_per_device[dev_i]].set(C[start_row:end_row])
        Z_sh = Z_sh.at[dev_i, :rows_per_device[dev_i]].set(Z[start_row:end_row])
        start_row = end_row

    # PMAP the TSQR least‑squares solve with actual row counts
    X_sh = jax.pmap(tsqr_least_squares, axis_name='i')(C_sh, Z_sh, actual_rows_sh)  # (P, cols, rhs)
    X_tsqr = X_sh[0]  # all devices get the same X_local

    # Compare residuals
    res_ref  = jnp.linalg.norm(Z - C @ X_ref)
    res_tsqr = jnp.linalg.norm(Z - C @ X_tsqr)

    print(f"   📈 Serial residual:      {res_ref:.3e}")
    print(f"   📈 Distributed residual: {res_tsqr:.3e}")
    
    try:
        assert jnp.allclose(res_ref, res_tsqr, rtol=1e-6, atol=1e-6)
        print("   ✅ PASSED - Matches serial solution!")
    except AssertionError:
        print(f"   ❌ FAILED - Residuals differ too much")
        raise

if __name__ == "__main__":
    main()

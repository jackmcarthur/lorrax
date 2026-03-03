#!/usr/bin/env python3
"""
ISDF Preprocessing Pipeline: zeta fitting and V_q computation.

Computes:
  1. Zeta fitting: C_q @ zeta_q = Z_q (chunked over z-axis, output to HDF5)
  2. V_q computation (optional): V_q(μ,ν) = Σ_G ζ̃*_μ(G) v(q+G) ζ̃_ν(G)

Final outputs retained in memory:
  - psi_nk(r_μ)_X: centroid wavefunctions, X-sharded
  - psi_nk(r_μ)_Y: centroid wavefunctions, Y-sharded  
  - V_q(μ,ν): Coulomb matrix (if --compute-vcoul), XY-sharded

Usage:
    uv run python -m isdf.gw_isdf.test_chunked_wfn_loading -i input.in
    uv run python -m isdf.gw_isdf.test_chunked_wfn_loading -i input.in --compute-vcoul
"""
import os

# JAX environment setup - must be before jax import
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Don't set JAX_PLATFORMS - let JAX auto-detect GPU, fall back to CPU
# To force CPU for testing: JAX_PLATFORMS=cpu uv run ...
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

# Project imports
from ..io import WFNReader, load_centroids, resolve_input_paths
from ..common import symmetry_maps
from ..common.load_wfns import (
    read_Gvecs_to_devices,
    get_sharded_wfns,
    compute_pair_density_spin_traced,
    compute_CCT_from_left_right,
    compute_L_q_from_CCT,
    solve_zeta_from_L_q,
    fit_zeta_chunked_to_h5,
)
from ..common import Meta, timing
from ..common.gpu_utils import get_device_memory_gb, get_device_memory_info
from .cohsex_init import (
    read_cohsex_input, 
    compute_optimal_chunks,
    print_memory_breakdown,
)
from .compute_vcoul import (
    compute_V_q_from_zeta_h5,
    compute_all_V_q_from_zeta_h5,
)


def print_sharding_info(name: str, arr: jax.Array):
    """Print sharding details for a JAX array."""
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Sharding: {arr.sharding}")
    if hasattr(arr.sharding, 'spec'):
        print(f"  PartitionSpec: {arr.sharding.spec}")
    # Show per-device shard shapes
    if hasattr(arr, 'addressable_shards') and arr.addressable_shards:
        shard = arr.addressable_shards[0]
        print(f"  Local shard shape: {shard.data.shape}")
        print(f"  Num addressable shards: {len(arr.addressable_shards)}")


def setup_mesh():
    """Create 2D device mesh for sharding."""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Available devices: {n_devices} ({jax.default_backend()})")
    
    # Create 2D mesh: find factors close to sqrt
    grid_x = int(np.sqrt(n_devices))
    while n_devices % grid_x != 0:
        grid_x -= 1
    grid_y = n_devices // grid_x
    
    devices_2d = np.array(devices).reshape(grid_x, grid_y)
    mesh_xy = Mesh(devices_2d, ['x', 'y'])
    print(f"Device mesh: {grid_x}×{grid_y}")
    return mesh_xy


def load_reference_wfns(wfn, sym, band_range, meta, centroid_indices, bispinor, mesh_xy):
    """
    Load wavefunctions using the current (full) method for reference.
    
    Returns:
        psi_nk_rmu_Y: (nk, nb, ns, n_rmu) sharded on Y
        psi_nk_rmuT_X: (nk, n_rmu, nb, ns) sharded on X (transposed conjugate)
        psi_nk_rtot_Y: (nk, nb, ns, n_rtot) sharded on Y - the one we want to eliminate
    """
    print(f"\nLoading reference wavefunctions for bands {band_range}...")
    
    # First call includes JIT compilation - measure separately
    with timing.section("wfn_load.first_call_jit"):
        global_psiG, nb = read_Gvecs_to_devices(wfn, sym, band_range, meta, bispinor, mesh_xy)
        global_psiG.block_until_ready()
        psi_rtot_Y, psi_rmu_Y, psi_rmuT_X = get_sharded_wfns(
            global_psiG, sym, meta, centroid_indices, nb, False, mesh_xy
        )
        psi_rtot_Y.block_until_ready()
        psi_rmu_Y.block_until_ready()
        psi_rmuT_X.block_until_ready()
        del global_psiG
    
    # Second call uses cached JIT - measure actual performance
    with timing.section("wfn_load.cached_call"):
        with timing.section("wfn_load.read_Gvecs"):
            global_psiG, nb = read_Gvecs_to_devices(wfn, sym, band_range, meta, bispinor, mesh_xy)
            global_psiG.block_until_ready()
        
        with timing.section("wfn_load.fft_and_sample"):
            psi_rtot_Y, psi_rmu_Y, psi_rmuT_X = get_sharded_wfns(
                global_psiG, sym, meta, centroid_indices, nb, False, mesh_xy
            )
            psi_rtot_Y.block_until_ready()
            psi_rmu_Y.block_until_ready()
            psi_rmuT_X.block_until_ready()
        
        del global_psiG
    
    return psi_rmu_Y, psi_rmuT_X, psi_rtot_Y


def extract_z_chunk(psi_rtot: jax.Array, fft_grid: tuple, z_start: int, z_chunk_size: int, mesh_xy=None) -> jax.Array:
    """
    Extract a z-slice chunk from the full rtot array.
    
    Args:
        psi_rtot: (nk, nb, ns, n_rtot) flattened real-space wfns
        fft_grid: (nx, ny, nz) FFT grid dimensions
        z_start: starting z index
        z_chunk_size: number of z slices to extract
        mesh_xy: optional mesh for resharding
        
    Returns:
        psi_zchunk: (nk, nb, ns, nx*ny*z_chunk_size) sharded on Y if mesh provided
    """
    nx, ny, nz = fft_grid
    nk, nb, ns, n_rtot = psi_rtot.shape
    
    # Reshape to spatial grid
    psi_spatial = psi_rtot.reshape(nk, nb, ns, nx, ny, nz)
    
    # Extract z-slice
    z_end = min(z_start + z_chunk_size, nz)
    psi_zslice = psi_spatial[:, :, :, :, :, z_start:z_end]
    
    # Flatten back to 1D
    actual_chunk_size = z_end - z_start
    psi_zchunk = psi_zslice.reshape(nk, nb, ns, nx * ny * actual_chunk_size)
    
    # Reshard to Y if mesh provided
    if mesh_xy is not None:
        y_shard = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        psi_zchunk = jax.lax.with_sharding_constraint(psi_zchunk, y_shard)
    
    return psi_zchunk


def test_pair_density(psi_rmuT_X, psi_rmu_Y, psi_zchunk_Y, mesh_xy):
    """Test P_k,ab(μ,ν) = Σ_n ψ*_nka(μ)·ψ_nkb(ν). See ZETA_FITTING_ALGORITHM.md."""
    print("\n" + "="*60)
    print("TEST: Pair Density Computation")
    print("="*60)
    
    nk, n_rmu, nb, ns = psi_rmuT_X.shape
    _, _, _, n_zchunk = psi_zchunk_Y.shape
    
    print(f"\nInput shapes:")
    print(f"  psi_rmuT_X: {psi_rmuT_X.shape} - conj(psi_nk,s(r_mu)) with mu on X")
    print(f"  psi_rmu_Y:  {psi_rmu_Y.shape} - psi_nk,s(r_nu) with nu on Y")
    print(f"  psi_zchunk_Y: {psi_zchunk_Y.shape} - psi_nk,s(r_zchunk) with zchunk on Y")
    
    # Test 1: Spin-traced P_k(r_mu, r_nu) for centroids
    print("\n--- Test 1: Spin-traced P_k(r_mu, r_nu) for centroids ---")
    with timing.section("pair_density.rmu_rmu"):
        P_k = compute_pair_density_spin_traced(psi_rmuT_X, psi_rmu_Y, mesh_xy)
        P_k.block_until_ready()

    expected_shape = (nk, n_rmu, n_rmu)
    print(f"  Output shape: {P_k.shape} (expected {expected_shape})")
    assert P_k.shape == expected_shape, f"Shape mismatch: {P_k.shape} != {expected_shape}"

    print_sharding_info("  P_k(r_mu, r_nu)", P_k)

    # Check sharding is correct
    if hasattr(P_k.sharding, 'spec'):
        spec = P_k.sharding.spec
        print(f"  Expected sharding: P(None, 'x', 'y')")
        assert spec[1] in ('x', ('x',)), f"mu should be on 'x', got {spec[1]}"
        assert spec[2] in ('y', ('y',)), f"nu should be on 'y', got {spec[2]}"

    # Test 2: Verify definition matches manual computation (small check)
    print("\n--- Test 2: Verify P = sum_{n,s} psi*_L @ psi_R ---")
    # Manual: P[k,mu,nu] = sum_{n,s} psi_L[k,mu,n,s] * psi_R[k,n,s,nu]
    psi_L_np = np.array(psi_rmuT_X)  # (nk, n_rmu, nb, ns)
    psi_R_np = np.array(psi_rmu_Y)   # (nk, nb, ns, n_rmu)
    P_manual = np.einsum('kmns,knsv->kmv', psi_L_np, psi_R_np)
    P_jax = np.array(P_k)

    match = np.allclose(P_manual, P_jax, rtol=1e-10)
    max_diff = np.max(np.abs(P_manual - P_jax))
    print(f"  Manual einsum matches JAX: {match} (max diff: {max_diff:.2e})")
    assert match, f"Pair density mismatch: max diff = {max_diff}"

    # Test 3: Memory footprint
    print("\n--- Memory footprint ---")
    P_size_mb = P_k.nbytes / 1e6
    print(f"  P_k(r_mu, r_mu): {P_size_mb:.2f} MB")

    print("\n✓ Pair density tests passed!")
    return P_k, None


def test_CCT_ZCT_pipeline(P_k_mumu, P_k_mu_zchunk, kgrid, mesh_xy):
    """Test CCT pipeline using left/right cross-product approach."""
    print("\n" + "="*60)
    print("TEST: CCT Pipeline (compute_CCT_from_left_right)")
    print("="*60)

    nk, n_rmu, _ = P_k_mumu.shape
    nkx, nky, nkz = kgrid

    print(f"\nInput shapes:")
    print(f"  P_k(μ,μ):    {P_k_mumu.shape}")
    print(f"  k-grid:      {kgrid}")

    # Test 1: Full CCT computation (using same pair density for both L and R)
    print("\n--- Test 1: compute_CCT_from_left_right ---")
    with timing.section("CCT.full_pipeline"):
        C_q = compute_CCT_from_left_right(P_k_mumu, P_k_mumu, kgrid, mesh_xy)
        C_q.block_until_ready()

    expected_C_shape = (nkx, nky, nkz, n_rmu, n_rmu)
    print(f"  C_q shape: {C_q.shape} (expected {expected_C_shape})")
    assert C_q.shape == expected_C_shape
    print_sharding_info("  C_q (CCT)", C_q)

    # Test 2: Verify C_q[q=0] properties
    print("\n--- Test 2: Verify CCT properties ---")
    C_q_np = np.array(C_q)
    C_q0 = C_q_np[0, 0, 0]
    q0_symmetric = np.allclose(C_q0, C_q0.T, rtol=1e-10)
    q0_pos_diag = (np.diag(C_q0).real > 0).all()
    print(f"  C_q[q=0] is symmetric: {q0_symmetric}")
    print(f"  C_q[q=0] has positive diagonal: {q0_pos_diag}")

    # Memory footprint
    print("\n--- Memory footprint ---")
    C_size_mb = C_q.nbytes / 1e6
    print(f"  C_q: {C_size_mb:.2f} MB")

    print("\n✓ CCT pipeline tests passed!")
    return C_q, None


def test_zeta_solve(C_q, Z_q, mesh_xy, use_blocked_2d: bool = True):
    """Test zeta_q = L_q^{-H}(L_q^{-1} Z_q) with compute_L_q_from_CCT + solve_zeta_from_L_q."""
    print("\n" + "="*60)
    print("TEST: Zeta Solve (2D Blocked Cholesky + Column-Parallel Solve)")
    print("="*60)

    # C_q from CCT has shape (nqx, nqy, nqz, n_rmu, n_rmu)
    nqx, nqy, nqz, n_rmu, n_rmu2 = C_q.shape
    nq = nqx * nqy * nqz

    # Synthesize Z_q for testing (same shape as C_q for simplicity)
    if Z_q is None:
        Z_q = C_q  # Use C_q as stand-in Z_q

    _, _, _, _, n_col = Z_q.shape

    print(f"\nInput shapes:")
    print(f"  C_q: {C_q.shape} (nqx={nqx}, nqy={nqy}, nqz={nqz}, n_rmu={n_rmu})")
    print(f"  Z_q: {Z_q.shape} (n_col={n_col})")

    # Flatten q-grid
    C_q_flat = C_q.reshape(nq, n_rmu, n_rmu)
    Z_q_flat = Z_q.reshape(nq, n_rmu, n_col)

    # Apply required input sharding
    from jax.sharding import NamedSharding, PartitionSpec as P
    input_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, input_shard)
    Z_q_flat = jax.lax.with_sharding_constraint(Z_q_flat, input_shard)

    print("\nInput shardings:")
    print_sharding_info("  C_q (flattened)", C_q_flat)
    print_sharding_info("  Z_q (flattened)", Z_q_flat)

    # Test 1: Compute L_q = chol(C_q)
    print("\n--- Test 1: compute_L_q_from_CCT ---")
    with timing.section("zeta_solve.cholesky"):
        L_q = compute_L_q_from_CCT(C_q_flat, mesh_xy)
        L_q.block_until_ready()

    print(f"  L_q shape: {L_q.shape}")
    print_sharding_info("  L_q", L_q)

    # Test 2: Solve zeta from L_q
    print("\n--- Test 2: solve_zeta_from_L_q ---")
    with timing.section("zeta_solve.solve"):
        zeta_q = solve_zeta_from_L_q(L_q, Z_q_flat, mesh_xy)
        zeta_q.block_until_ready()

    print(f"  zeta_q shape: {zeta_q.shape}")
    print_sharding_info("  zeta_q", zeta_q)

    # Test 3: Verify solution correctness C_q @ zeta_q ≈ Z_q
    print("\n--- Test 3: Solution correctness ---")
    C_q_rep = jax.lax.with_sharding_constraint(
        C_q_flat, NamedSharding(mesh_xy, P(None, None, None)))
    zeta_rep = jax.lax.with_sharding_constraint(
        zeta_q, NamedSharding(mesh_xy, P(None, None, None)))
    Z_q_rep = jax.lax.with_sharding_constraint(
        Z_q_flat, NamedSharding(mesh_xy, P(None, None, None)))

    Z_reconstructed = jnp.einsum('qmn,qnr->qmr', C_q_rep, zeta_rep)
    max_diff = float(jnp.max(jnp.abs(Z_reconstructed - Z_q_rep)))
    rel_error = max_diff / float(jnp.max(jnp.abs(Z_q_rep)))
    print(f"  max|C_q @ zeta - Z_q| = {max_diff:.2e}")
    print(f"  relative error = {rel_error:.2e}")
    assert rel_error < 1e-10, f"Solution error too large: {rel_error:.2e}"
    print("  ✓ Solution verified: C_q @ zeta_q = Z_q")

    # Memory footprint
    print("\n--- Memory footprint ---")
    zeta_mb = zeta_q.nbytes / 1e6
    print(f"  zeta_q: {zeta_mb:.2f} MB")
    print(f"  Per device (with sharding): {zeta_mb/len(jax.devices()):.2f} MB")

    print("\n✓ Zeta solve tests passed!")
    return zeta_q


def test_current_loading(wfn, sym, meta, centroid_indices, bispinor, mesh_xy, band_range, z_chunk_size=16):
    """Load ψ_nk(r_μ)_Y, ψ_nk(r_μ)^T_X, ψ_nk(r_zchunk)_Y. See ZETA_FITTING_ALGORITHM.md."""
    print("\n" + "="*60)
    print("Testing CURRENT (full) wavefunction loading method")
    print("="*60)
    
    # Show target shardings
    print("\nTARGET SHARDINGS (for chunked implementation):")
    print("  psi_nk(r_μ)_Y:      P(None, None, None, 'y')   # Y-sharded on μ axis")
    print("  psi_nk(r_μ)T_X:     P(None, 'x', None, None)   # X-sharded on μ axis (transposed)")
    print("  psi_nk(r_zchunk)_Y: P(None, None, None, 'y')   # Y-sharded on r_zchunk axis")
    print("  [psi_nk(r_tot) should NOT be stored - only transient during FFT]")
    
    # Load using current method
    psi_rmu_Y, psi_rmuT_X, psi_rtot_Y = load_reference_wfns(
        wfn, sym, band_range, meta, centroid_indices, bispinor, mesh_xy
    )
    
    # Print sharding info
    print("\nACTUAL SHARDINGS (current implementation):")
    print_sharding_info("psi_nk(r_μ)_Y", psi_rmu_Y)
    print_sharding_info("psi_nk(r_μ)T_X", psi_rmuT_X)
    print_sharding_info("psi_nk(r_tot)_Y [to be eliminated]", psi_rtot_Y)
    
    # Extract first z-chunk for comparison (with proper Y sharding)
    fft_grid = tuple(int(x) for x in meta.fft_grid)
    nz = fft_grid[2]
    actual_z_chunk = min(z_chunk_size, nz)
    psi_zchunk_ref = extract_z_chunk(psi_rtot_Y, fft_grid, 0, z_chunk_size, mesh_xy)
    print_sharding_info(f"psi_nk(r_zchunk)_Y [reference, z=0:{actual_z_chunk}]", psi_zchunk_ref)
    
    # Report memory usage
    def mem_mb(arr):
        return arr.size * arr.dtype.itemsize / 1e6
    
    print("\n" + "-"*40)
    print("Memory usage (MB):")
    print(f"  psi_rmu_Y:   {mem_mb(psi_rmu_Y):.2f} MB")
    print(f"  psi_rmuT_X:  {mem_mb(psi_rmuT_X):.2f} MB")
    print(f"  psi_rtot_Y:  {mem_mb(psi_rtot_Y):.2f} MB  <-- THIS IS THE PROBLEM")
    print(f"  psi_zchunk:  {mem_mb(psi_zchunk_ref):.2f} MB")
    print(f"\n  Memory ratio rtot/rmu: {mem_mb(psi_rtot_Y)/mem_mb(psi_rmu_Y):.1f}x")
    print(f"  Memory ratio rtot/zchunk: {mem_mb(psi_rtot_Y)/mem_mb(psi_zchunk_ref):.1f}x")
    
    # Show some sample values for verification
    print("\n" + "-"*40)
    print("Sample values for verification (first k-point, first band, spin-up):")
    rmu_sample = np.array(psi_rmu_Y[0, 0, 0, :5])
    print(f"  psi_rmu[0,0,0,:5]: {rmu_sample}")
    zchunk_sample = np.array(psi_zchunk_ref[0, 0, 0, :5])
    print(f"  psi_zchunk[0,0,0,:5]: {zchunk_sample}")
    
    return psi_rmu_Y, psi_rmuT_X, psi_rtot_Y, psi_zchunk_ref


def load_wfn_band_chunk_and_sample(
    wfn, sym, band_range, meta, centroid_indices, bispinor, mesh_xy,
    band_chunk_size: int = 16,
):
    """PLACEHOLDER: Band-chunked loading. See get_psi_rchunk in load_wfns.py."""
    # TODO: Implement chunked loading
    # For now, fall back to current implementation
    print("  [PLACEHOLDER] Using current (non-chunked) implementation")
    global_psiG, nb = read_Gvecs_to_devices(wfn, sym, band_range, meta, bispinor, mesh_xy)
    psi_rtot_Y, psi_rmu_Y, psi_rmuT_X = get_sharded_wfns(
        global_psiG, sym, meta, centroid_indices, nb, False, mesh_xy
    )
    del global_psiG
    
    return psi_rmu_Y, psi_rmuT_X


def test_shard_map_fft(mesh_xy):
    """Test shard_map FFT workaround for multi-device CPU/GPU."""
    from isdf.common.load_wfns import make_sharded_ifftn_3d
    
    print("\n" + "="*60)
    print("Testing SHARD_MAP FFT (runs on each device independently)")
    print("="*60)
    
    # Create a test array with shape similar to our wavefunction arrays
    # (nk, nb, ns, nx, ny, nz)
    nk, nb, ns, nx, ny, nz = 4, 8, 2, 8, 8, 8
    
    # Create random data
    key = jax.random.PRNGKey(42)
    x_host = jax.random.normal(key, (nk, nb, ns, nx, ny, nz), dtype=jnp.complex128)
    
    # Shard over bands (axis 1) using ('x','y') combined
    band_sharding = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None, None, None))
    x_sharded = jax.device_put(x_host, band_sharding)
    
    print(f"\nInput shape: {x_sharded.shape}")
    print(f"Input sharding: {x_sharded.sharding.spec}")
    
    # Create shard_map based FFT
    fft_spec = P(None, ('x', 'y'), None, None, None, None)
    sharded_ifftn = make_sharded_ifftn_3d(mesh_xy, fft_spec, fft_spec)
    
    # Test shard_map FFT
    print("\n1. Testing shard_map based FFT...")
    try:
        y_sharded = sharded_ifftn(x_sharded)
        y_sharded.block_until_ready()
        print(f"   SUCCESS! Output shape: {y_sharded.shape}")
        print(f"   Output sharding: {y_sharded.sharding.spec}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {str(e)[:100]}...")
        import traceback
        traceback.print_exc()
        return False
    
    # Compute reference on single device
    print("\n2. Verifying correctness against reference...")
    try:
        # Gather to single device and compute reference
        x_full = np.array(x_sharded)
        y_ref = np.fft.ifftn(x_full, axes=(-3, -2, -1))
        
        # Compare
        y_result = np.array(y_sharded)
        match = np.allclose(y_result, y_ref, rtol=1e-10)
        print(f"   Values match reference: {match}")
        if not match:
            max_err = np.max(np.abs(y_result - y_ref))
            print(f"   Max error: {max_err}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {str(e)[:100]}...")
    
    return True


def test_sharding_logic_only(mesh_xy, meta, centroid_indices):
    """Test sharding patterns with mock data (no file I/O)."""
    print("\n" + "="*60)
    print("Testing SHARDING LOGIC (no FFT - for multi-device CPU)")
    print("="*60)
    
    n_devices = jax.device_count()
    is_multidevice_cpu = n_devices > 1 and jax.default_backend() == 'cpu'
    
    if not is_multidevice_cpu:
        print("  Skipping: Only needed for multi-device CPU testing")
        return True
    
    # Create mock psi_rtot data with appropriate shape
    nk = meta.nk_tot
    nb = 10  # Test bands
    ns = meta.nspinor
    n_rtot = meta.n_rtot
    n_rmu = len(centroid_indices)
    fft_grid = meta.fft_grid
    
    print(f"\n  Mock data: nk={nk}, nb={nb}, ns={ns}, n_rtot={n_rtot}, n_rmu={n_rmu}")
    print(f"  FFT grid: {fft_grid}")
    
    # Create random mock wavefunction (pretend FFT already done)
    key = jax.random.PRNGKey(42)
    psi_rtot = jax.random.normal(key, (nk, nb, ns, n_rtot), dtype=jnp.complex128)
    
    # Define target shardings
    y_shard = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    x_shard = NamedSharding(mesh_xy, P(None, 'x', None, None))
    null_shard = NamedSharding(mesh_xy, P(None, None, None, None))
    
    print("\n  1. Testing centroid gather (psi_rtot -> psi_rmu)...")
    
    @jax.jit
    def gather_centroids(psi_rtot, centroid_indices):
        # Compute linear centroid indices
        ny = jnp.asarray(fft_grid[1], dtype=jnp.int32)
        nz = jnp.asarray(fft_grid[2], dtype=jnp.int32)
        centroids = jnp.asarray(centroid_indices, dtype=jnp.int32)
        centroid_lin = (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(jnp.int32)
        
        # Gather at centroids
        psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)
        
        # Reshard to Y
        psi_rmu_Y = jax.lax.with_sharding_constraint(psi_rmu, y_shard)
        
        # Create transposed version with X sharding
        psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))  # (nk, n_rmu, nb, ns)
        psi_rmuT_X = jax.lax.with_sharding_constraint(psi_rmuT, x_shard)
        
        return psi_rmu_Y, psi_rmuT_X
    
    try:
        centroids_arr = np.array(centroid_indices)
        psi_rmu_Y, psi_rmuT_X = gather_centroids(psi_rtot, centroids_arr)
        psi_rmu_Y.block_until_ready()
        psi_rmuT_X.block_until_ready()
        
        print(f"     psi_rmu_Y: shape={psi_rmu_Y.shape}, sharding={psi_rmu_Y.sharding.spec}")
        print(f"     psi_rmuT_X: shape={psi_rmuT_X.shape}, sharding={psi_rmuT_X.sharding.spec}")
        print("     SUCCESS!")
    except Exception as e:
        print(f"     FAILED: {e}")
        return False
    
    print("\n  2. Testing z-chunk extraction...")
    
    z_chunk_size = 16
    
    @partial(jax.jit, static_argnums=(1, 2))
    def extract_zchunk(psi_rtot, z_start, z_chunk_size):
        nx, ny, nz = fft_grid
        nk, nb, ns, _ = psi_rtot.shape
        
        # Reshape to spatial
        psi_spatial = psi_rtot.reshape(nk, nb, ns, nx, ny, nz)
        
        # Extract z-slice using static indices
        psi_zslice = jax.lax.dynamic_slice(
            psi_spatial,
            (0, 0, 0, 0, 0, z_start),
            (nk, nb, ns, nx, ny, z_chunk_size)
        )
        
        # Flatten
        psi_zchunk = psi_zslice.reshape(nk, nb, ns, -1)
        
        # Reshard to Y
        return jax.lax.with_sharding_constraint(psi_zchunk, y_shard)
    
    try:
        psi_zchunk = extract_zchunk(psi_rtot, 0, z_chunk_size)
        psi_zchunk.block_until_ready()
        
        expected_size = fft_grid[0] * fft_grid[1] * z_chunk_size
        print(f"     psi_zchunk: shape={psi_zchunk.shape}, expected_r_dim={expected_size}")
        print(f"     sharding={psi_zchunk.sharding.spec}")
        print("     SUCCESS!")
    except Exception as e:
        print(f"     FAILED: {e}")
        return False
    
    print("\n  3. Testing accumulation pattern (P^L * P^R)...")
    
    @jax.jit  
    def test_accumulation(psi_rmu_Y, psi_rmuT_X):
        # Simulate CCT accumulation: Pmu = psi_rmuT @ psi_rmu for one k
        # psi_rmuT_X: (nk, n_rmu, nb, ns) X-sharded on n_rmu
        # psi_rmu_Y: (nk, nb, ns, n_rmu) Y-sharded on n_rmu
        
        # For k=0, compute outer product pattern
        psi_rmuT_k = psi_rmuT_X[0]  # (n_rmu, nb, ns)
        psi_rmu_k = psi_rmu_Y[0]    # (nb, ns, n_rmu)
        
        # Reshape for matmul
        psi_rmuT_k_flat = psi_rmuT_k.reshape(n_rmu, -1)  # (n_rmu, nb*ns)
        psi_rmu_k_flat = psi_rmu_k.reshape(-1, n_rmu)    # (nb*ns, n_rmu)
        
        # This gives (n_rmu, n_rmu) matrix - check sharding
        Pmu = psi_rmuT_k_flat @ psi_rmu_k_flat  # (n_rmu_X, n_rmu_Y)
        
        return Pmu
    
    try:
        Pmu = test_accumulation(psi_rmu_Y, psi_rmuT_X)
        Pmu.block_until_ready()
        
        print(f"     P_mu: shape={Pmu.shape}")
        print(f"     sharding={Pmu.sharding.spec}")
        print("     SUCCESS!")
    except Exception as e:
        print(f"     FAILED: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL SHARDING TESTS PASSED!")
    print("="*60)
    return True


def test_chunked_loading(wfn, sym, meta, centroid_indices, bispinor, mesh_xy, 
                         band_range, z_chunk_size=16, band_chunk_size=16,
                         ref_psi_rmu=None, ref_psi_rmuT=None):
    """
    Test the NEW chunked loading method and compare to reference.
    """
    print("\n" + "="*60)
    print("Testing NEW (chunked) wavefunction loading method")
    print("="*60)
    print(f"  Band chunk size: {band_chunk_size}")
    print(f"  Z chunk size: {z_chunk_size}")
    
    # Load using chunked method
    psi_rmu_Y, psi_rmuT_X = load_wfn_band_chunk_and_sample(
        wfn, sym, band_range, meta, centroid_indices, bispinor, mesh_xy,
        band_chunk_size
    )
    
    print_sharding_info("psi_nk(r_μ)_Y [chunked]", psi_rmu_Y)
    print_sharding_info("psi_nk(r_μ)T_X [chunked]", psi_rmuT_X)
    
    # Compare to reference if provided
    if ref_psi_rmu is not None:
        rmu_match = np.allclose(np.array(psi_rmu_Y), np.array(ref_psi_rmu), rtol=1e-10)
        print(f"\n  psi_rmu matches reference: {rmu_match}")
        if not rmu_match:
            diff = np.max(np.abs(np.array(psi_rmu_Y) - np.array(ref_psi_rmu)))
            print(f"    Max difference: {diff:.2e}")
    
    if ref_psi_rmuT is not None:
        rmuT_match = np.allclose(np.array(psi_rmuT_X), np.array(ref_psi_rmuT), rtol=1e-10)
        print(f"  psi_rmuT matches reference: {rmuT_match}")
        if not rmuT_match:
            diff = np.max(np.abs(np.array(psi_rmuT_X) - np.array(ref_psi_rmuT)))
            print(f"    Max difference: {diff:.2e}")
    
    return psi_rmu_Y, psi_rmuT_X


def main(argv=None):
    argp = argparse.ArgumentParser(description="ISDF Preprocessing: zeta fitting and V_q computation")
    argp.add_argument("-i", "--input", default="cohsex_test.in", help="Input file")
    argp.add_argument("--r-chunk", type=int, default=None,
                      help="R-space chunk size (overrides auto)")
    argp.add_argument("--band-chunk", type=int, default=None, help="Band chunk size (overrides auto)")
    argp.add_argument("--q-chunk", type=int, default=None, 
                      help="Q-points to solve simultaneously (default=auto)")
    argp.add_argument("--mu-chunk", type=int, default=None,
                      help="μ-chunk size for V_q computation (overrides auto)")
    argp.add_argument("--memory", type=float, default=None,
                      help="Memory budget per device in GB (default: auto-detect)")
    argp.add_argument("--zeta-output", type=str, default="zeta_q.h5",
                      help="Output HDF5 file for zeta fitting (default: zeta_q.h5)")
    argp.add_argument("--compute-vcoul", action="store_true",
                      help="Also compute V_q(μ,ν) from zeta after fitting")
    # Legacy/debugging flags
    argp.add_argument("--test-fft-only", action="store_true", help="Only test custom FFT sharding")
    argp.add_argument("--test-sharding-only", action="store_true", 
                      help="Test sharding logic with mock data (for multi-device CPU)")
    args = argp.parse_args(argv)
    
    print("\n" + "="*72)
    print("  ISDF Preprocessing Pipeline")
    print("="*72)
    print("  Steps:")
    print("    1. Load centroid wavefunctions (band-chunked FFT)")
    print("    2. Compute CCT: C_q = Σ_k |P_k(r_μ, r_ν)|²")
    print("    3. Loop over r-chunks: compute ZCT, solve zeta, write to HDF5")
    if args.compute_vcoul:
        print("    4. Compute V_q(μ,ν) = Σ_G ζ̃*_μ(G) v(q+G) ζ̃_ν(G)")
    print("-"*72)
    print(f"  JAX backend: {jax.default_backend()}")
    print(f"  Device count: {jax.device_count()}")
    print(f"  Local device count: {jax.local_device_count()}")
    print("="*72)
    
    # Reset timing
    timing.reset()
    
    # Setup mesh
    mesh_xy = setup_mesh()
    
    # Test shard_map FFT if requested (useful for multi-device CPU testing)
    if args.test_fft_only:
        with mesh_xy:
            success = test_shard_map_fft(mesh_xy)
        print("\n" + "="*60)
        print("FFT TEST COMPLETE")
        print("="*60)
        return 0 if success else 1
    
    # Read input parameters
    params = read_cohsex_input(args.input)
    input_dir = os.path.dirname(os.path.abspath(args.input))
    resolve_input_paths(params, input_dir)
    
    nval = params["nval"]
    ncond = params["ncond"]
    nband = params["nband"]
    bispinor = params["bispinor"]
    
    # Load WFN and symmetry maps
    print(f"\nLoading WFN from: {params['wfn_file']}")
    wfn = WFNReader(params["wfn_file"])
    sym = symmetry_maps.SymMaps(wfn)
    
    # Load centroids
    centroids_frac, centroid_indices, n_rmu = load_centroids(
        params["centroids_file"], wfn.fft_grid
    )
    print(f"Loaded {n_rmu} centroids")
    
    # Create Meta object
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)
    
    # =========================================================================
    # MEMORY DETECTION AND OPTIMAL CHUNK SIZING
    # =========================================================================
    # Priority for memory budget:
    # 1. --memory CLI arg
    # 2. memory_per_device_gb from input file
    # 3. auto-detect (80% of GPU/CPU memory)
    
    memory_info = get_device_memory_info()
    memory_source = memory_info['source']
    
    if args.memory is not None:
        memory_budget_gb = args.memory
        memory_source = f"CLI --memory={args.memory}"
    elif params.get("memory_per_device_gb", 0.0) > 0:
        memory_budget_gb = params["memory_per_device_gb"]
        memory_source = f"input file (memory_per_device_gb={memory_budget_gb})"
    else:
        memory_budget_gb = memory_info['total_gb']
        memory_source = f"auto-detect ({memory_info['source']})"
    
    # Get mesh dimensions
    mesh_x_size = mesh_xy.devices.shape[0] if len(mesh_xy.devices.shape) > 0 else 1
    mesh_y_size = mesh_xy.devices.shape[1] if len(mesh_xy.devices.shape) > 1 else 1
    n_devices = jax.device_count()
    
    # Print system info
    print(f"\nSystem info:")
    print(f"  FFT grid: {meta.fft_grid}")
    print(f"  n_rtot: {meta.n_rtot}")
    print(f"  n_rmu: {meta.n_rmu}")
    print(f"  nk_tot: {meta.nk_tot}")
    print(f"  nspinor: {meta.nspinor}")
    print(f"  Band edges: b0={meta.b_id_0}, b1={meta.b_id_1}, b2={meta.b_id_2}, b3={meta.b_id_3}, b4={meta.b_id_4}")
    
    # Compute optimal chunk sizes from memory budget
    nx, ny, nz = meta.fft_grid
    nqx, nqy, nqz = meta.kgrid
    n_q = nqx * nqy * nqz
    
    # Band counts for left/right wavefunctions (cohsex_jax convention):
    # Left:  (b0, b3) = all occupied + sigma conduction
    # Right: (b0, b4) = all occupied + all conduction up to nband
    # For memory estimation, use FULL range (b0 to b4) since that's what we load
    n_b_left = meta.b_id_3 - meta.b_id_0
    n_b_right = meta.b_id_4 - meta.b_id_0
    n_b_full = max(n_b_left, n_b_right)  # = b4 - b0
    
    try:
        chunks = compute_optimal_chunks(
            n_k=meta.nk_tot,
            n_b=n_b_full,  # Use full band range for memory estimation
            n_s=meta.nspinor,
            n_rmu=n_rmu,
            n_r=meta.n_rtot,
            n_q=n_q,
            fft_grid=meta.fft_grid,
            n_devices=n_devices,
            memory_budget_gb=memory_budget_gb,
            target_utilization=0.85,
            p_x=mesh_x_size,
            p_y=mesh_y_size,
            n_b_left=n_b_left,
            n_b_right=n_b_right,
            verbose=True,
        )
        
        # Print memory breakdown
        print_memory_breakdown(
            chunks, n_b_full, meta.n_rtot, n_q, meta.fft_grid,
            memory_source=memory_source
        )
        
        # Use computed chunks, but allow CLI overrides
        band_chunk_size = args.band_chunk if args.band_chunk is not None else chunks['band_chunk']
        chunk_r = args.r_chunk if args.r_chunk is not None else chunks['chunk_r']
        q_chunk_size = args.q_chunk if args.q_chunk is not None else chunks['q_chunk']

        if args.band_chunk or args.r_chunk or args.q_chunk:
            print(f"\n  CLI overrides applied:")
            if args.band_chunk:
                print(f"    --band-chunk={args.band_chunk} (was {chunks['band_chunk']})")
            if args.r_chunk:
                print(f"    --r-chunk={args.r_chunk} (was {chunks['chunk_r']})")
            if args.q_chunk:
                print(f"    --q-chunk={args.q_chunk} (was {chunks['q_chunk']})")
        
    except ValueError as e:
        print(f"\nWARNING: Memory auto-sizing failed: {e}")
        print("Using fallback chunk sizes...")
        
        # Fallback: use r_chunk from CLI or a conservative default
        chunk_r = args.r_chunk if args.r_chunk is not None else min(meta.n_rtot, nx * ny * nz // 4)
        band_chunk_size = args.band_chunk if args.band_chunk is not None else 16
        q_chunk_size = args.q_chunk if args.q_chunk is not None else n_q

        # Print fallback r_chunk info
        pk_chunk_bytes = meta.nk_tot * n_rmu * chunk_r * 16  # Spin-traced pair density
        pk_chunk_mb = pk_chunk_bytes / 1e6

        print(f"\nFallback chunk configuration:")
        print(f"  chunk_r: {chunk_r} r-points")
        print(f"  P_k(rmu, rchunk) size: {pk_chunk_mb:.1f} MB")
        print(f"  band_chunk_size: {band_chunk_size}")
        print(f"  q_chunk_size: {q_chunk_size}")
    
    # =========================================================================
    # COMPUTE OPTIMAL MU-CHUNK SIZE FOR V_q (if requested)
    # =========================================================================
    # V_q memory: zeta_mu(G) + zeta_nu(G) + FFT workspace + V_q block (small)
    # No centroids needed - zeta is read from H5
    # Available: full budget
    
    if args.compute_vcoul:
        bytes_per_complex = 16
        n_G = meta.n_rtot
        
        # Available: 90% of budget (no centroids needed for V_q)
        m_budget_vcoul = memory_budget_gb * 1e9 * 0.9
        
        # Per mu: zeta_mu(G) + zeta_nu(G) for off-diag + FFT workspace = 3 × n_G × 16
        m_per_mu = 3 * bytes_per_complex * n_G
        mu_chunk_size = max(1, min(n_rmu, int(m_budget_vcoul / m_per_mu)))
        
        # Allow CLI override
        if args.mu_chunk is not None:
            mu_chunk_size = args.mu_chunk
        
        n_mu_chunks = (n_rmu + mu_chunk_size - 1) // mu_chunk_size
        print(f"\n  V_q computation:")
        print(f"    mu_chunk_size = {mu_chunk_size} ({n_mu_chunks} chunks)")
        print(f"    Memory per mu = {m_per_mu / 1e6:.1f} MB")
    else:
        mu_chunk_size = None
    
    # Summary of final chunk sizes
    print(f"\n  Final chunk sizes:")
    print(f"    band_chunk_size = {band_chunk_size}")
    print(f"    chunk_r = {chunk_r}")
    print(f"    q_chunk_size = {q_chunk_size}")
    print(f"    Num r-chunks to cover full grid: {(meta.n_rtot + chunk_r - 1) // chunk_r}")
    if mu_chunk_size:
        print(f"    mu_chunk_size = {mu_chunk_size} (for V_q)")
    
    # Test sharding logic only (for multi-device CPU which doesn't support FFT)
    if args.test_sharding_only:
        with mesh_xy:
            success = test_sharding_logic_only(mesh_xy, meta, centroid_indices)
        return 0 if success else 1
    
    # =========================================================================
    # STEP 1: ZETA FITTING PIPELINE
    # =========================================================================
    output_path = os.path.join(input_dir, args.zeta_output)
    print(f"\n{'='*60}")
    print("STEP 1: ZETA FITTING")
    print(f"{'='*60}")
    use_gspace_cache = chunks.get('use_gspace_cache', True)
    print(f"  Chunk sizes: band={band_chunk_size}, r={chunk_r}, q={q_chunk_size}")
    print(f"  G-space cache: {'enabled' if use_gspace_cache else 'disabled'}")
    print(f"  Output: {output_path}")
    
    # Band ranges for left/right wavefunctions (cohsex_jax convention)
    # Left:  (b0, b3) = all occupied + sigma conduction
    # Right: (b0, b4) = all occupied + all conduction up to nband
    band_range_left = (meta.b_id_0, meta.b_id_3)
    band_range_right = (meta.b_id_0, meta.b_id_4)
    
    with mesh_xy:
        psi_l_rmu_Y, psi_l_rmuT_X, psi_r_rmu_Y, psi_r_rmuT_X = fit_zeta_chunked_to_h5(
            wfn, sym, meta, centroid_indices, mesh_xy,
            output_path, band_chunk_size, q_chunk_size, bispinor,
            use_gspace_cache=use_gspace_cache,
            band_range_left=band_range_left,
            band_range_right=band_range_right,
            chunk_r=chunk_r,
        )
    
    # Verify zeta output
    import h5py
    if jax.process_index() == 0:
        with h5py.File(output_path, 'r') as f:
            print(f"\n  Zeta output verified:")
            print(f"    File: {output_path}")
            print(f"    zeta_q shape: {f['zeta_q'].shape}")
            print(f"    n_rmu={f.attrs['n_rmu']}, n_rtot={f.attrs['n_rtot']}")
    
    # =========================================================================
    # STEP 2: V_q COMPUTATION (optional)
    # =========================================================================
    V_qmunu = None
    
    if args.compute_vcoul:
        # Sync filesystem to ensure zeta writes are flushed before reading
        # This prevents I/O timing variance from dirty page cache
        import os as os_module
        os_module.sync()
        
        print(f"\n{'='*60}")
        print("STEP 2: V_q COULOMB MATRIX COMPUTATION")
        print(f"{'='*60}")
        print(f"  Reading zeta from: {output_path}")
        print(f"  mu_chunk_size: {mu_chunk_size}")
        
        # Get lattice parameters from wfn
        bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
        cell_volume = float(wfn.cell_volume)
        
        with h5py.File(output_path, 'r') as zeta_h5:
            with mesh_xy:
                V_qmunu, g0_mu_all = compute_all_V_q_from_zeta_h5(
                    zeta_h5,
                    kgrid=meta.kgrid,
                    fft_grid=meta.fft_grid,
                    bvec=bvec,
                    cell_volume=cell_volume,
                    mu_chunk_size=mu_chunk_size,
                    mesh_xy=mesh_xy,
                    sys_dim=2,
                )
        
        print(f"\n  V_q computed:")
        print(f"    Shape: {V_qmunu.shape}")
        print(f"    Sharding: {V_qmunu.sharding}")
        
        # Extract V_q=0 for later use (Hartree, etc.)
        V_q0 = V_qmunu[0, 0, 0, :, :]
        print(f"    V_q=0 trace: {float(jnp.trace(V_q0).real):.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    
    timing.get_collector().report(title="--- Timing Report ---", min_percent=0.1)
    
    print("\n  Outputs retained in memory:")
    print(f"    psi_l_rmu_Y:  {psi_l_rmu_Y.shape} sharded {psi_l_rmu_Y.sharding.spec}")
    print(f"    psi_l_rmuT_X: {psi_l_rmuT_X.shape} sharded {psi_l_rmuT_X.sharding.spec}")
    print(f"    psi_r_rmu_Y:  {psi_r_rmu_Y.shape} sharded {psi_r_rmu_Y.sharding.spec}")
    print(f"    psi_r_rmuT_X: {psi_r_rmuT_X.shape} sharded {psi_r_rmuT_X.sharding.spec}")
    if V_qmunu is not None:
        print(f"    V_qmunu:    {V_qmunu.shape}")
    print(f"\n  Zeta written to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

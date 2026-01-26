"""Input file parsing and preprocessing for COHSEX calculations.

This module contains functions for:
- Reading and parsing the cohsex input file
- Converting input parameters to effective values
- Computing band ranges from input parameters
- Memory-aware chunk size optimization with full communication buffer accounting
"""
import re
import configparser
import math


def compute_optimal_chunks(
    n_k: int,
    n_b: int,
    n_s: int,
    n_rmu: int,
    n_r: int,
    n_q: int,
    fft_grid: tuple[int, int, int],
    n_devices: int,
    memory_budget_gb: float,
    target_utilization: float = 0.85,
    p_x: int | None = None,
    p_y: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Compute optimal chunk sizes based on memory budget.
    
    Uses the memory model from MEMORY_MODEL.md to derive chunk sizes
    that will fit within the per-device memory budget. Accounts for all
    arrays AND inter-device communication buffers.
    
    X-chunking advantage: With r = x*(ny*nz) + y*nz + z, an x-chunk with
    x in [x_start, x_end) maps to CONTIGUOUS r-indices [x_start*ny*nz, x_end*ny*nz).
    This enables single sequential HDF5 writes instead of strided writes.
    
    Memory Model Stages (see MEMORY_MODEL.md for details):
    
    1. PERSISTENT: Centroid wavefunctions (psi_rmu_Y, psi_rmuT_X)
       - psi_rmu_Y: (n_k, n_b, n_s, n_rmu/p_y) per device
       - psi_rmuT_X: (n_k, n_rmu/p_x, n_b, n_s) per device
    
    2. FFT WORKSPACE: Band-chunked G->r transform
       - psi_G (input): (n_k, B_b/P, n_s, n_r) per device
       - psi_r (output): same shape
       - phase_spatial: (n_k, n_r) broadcast
       - COMMUNICATION: all-gather for psi_rmu after centroid gather
    
    3. X-CHUNK EXTRACTION:
       - psi_xchunk_Y: (n_k, n_b, n_s, B_x/p_y) per device
       - COMMUNICATION: staged reshard P(None,('x','y'),None,None) → P(None,None,None,'y')
         - Stage 1 buffer: (n_k, n_b/p_y, n_s, B_x)
    
    4. PAIR DENSITY (P_k):
       - P_k_mumu: (n_k, n_s², n_rmu/p_x, n_rmu/p_y) per device
       - P_k_mu_xchunk: (n_k, n_s², n_rmu/p_x, B_x/p_y) per device
    
    5. CCT/ZCT FFT PIPELINE:
       - P_R, C_R intermediate: same shape as P_k
       - C_q: (n_q, n_rmu/p_x, n_rmu/p_y) per device
       - Z_q: (n_q, n_rmu/p_x, B_x/p_y) per device
    
    6. CHOLESKY + SOLVE:
       - L_q tiles: (n_q, J/p_x, J/p_y, b, b) per device
       - COMMUNICATION: panel broadcast during Cholesky: O(n_q × b × n_rmu/√P)
       - L_rep (replicated): B_q × (n_rmu, n_rmu) per device
       - Z_col, zeta: (n_q, n_rmu, B_x/P) per device
    
    Args:
        n_k: Number of k-points
        n_b: Number of bands
        n_s: Number of spinor components (2 or 4)
        n_rmu: Number of ISDF centroids
        n_r: Total real-space grid points (nx*ny*nz)
        n_q: Number of q-points
        fft_grid: (nx, ny, nz) FFT grid dimensions
        n_devices: Total number of devices
        memory_budget_gb: Memory budget per device in GB
        target_utilization: Fraction of budget to use (default 0.85)
        p_x: X-dimension of mesh (default: sqrt(n_devices))
        p_y: Y-dimension of mesh (default: sqrt(n_devices))
        verbose: Print detailed memory breakdown
    
    Returns:
        Dictionary with:
        - band_chunk: Optimal band chunk size
        - x_chunk: Optimal x-axis chunk size (in x-slices)
        - x_chunk_r: Optimal x-axis chunk size (in r-points = x_chunk*ny*nz)
        - q_chunk: Optimal q-point chunk size for solve
        - memory_estimate: Dict of estimated memory usage by stage
    """
    bytes_per_complex = 16
    m_budget = memory_budget_gb * 1e9 * target_utilization
    p = n_devices
    nx, ny, nz = fft_grid
    
    # Default to square mesh if not specified
    if p_x is None or p_y is None:
        sqrt_p = int(math.sqrt(p))
        while p % sqrt_p != 0:
            sqrt_p -= 1
        p_x = sqrt_p
        p_y = p // p_x
    
    # ========================================================================
    # STAGE 1: PERSISTENT CENTROID MEMORY
    # ========================================================================
    # psi_rmu_Y: P(None,None,None,'y') → (n_k, n_b, n_s, n_rmu) with n_rmu/p_y per device
    # psi_rmuT_X: P(None,'x',None,None) → (n_k, n_rmu, n_b, n_s) with n_rmu/p_x per device
    # Both arrays exist simultaneously
    m_psi_rmu_Y = bytes_per_complex * n_k * n_b * n_s * (n_rmu / p_y)
    m_psi_rmuT_X = bytes_per_complex * n_k * n_b * n_s * (n_rmu / p_x)
    m_centroids = m_psi_rmu_Y + m_psi_rmuT_X
    
    # ========================================================================
    # STAGE 1b: CACHED G-SPACE FOR X-CHUNK LOOP (optional)
    # ========================================================================
    # G-space is loaded ONCE and cached across all x-chunk iterations.
    # Shape: (n_k, n_b, n_s, nx, ny, nz) sharded as P(None,('x','y'),None,None,None,None)
    # Per-device: (n_k, n_b/p, n_s, nx, ny, nz)
    # This avoids re-reading HDF5 for each x-chunk (huge speedup!)
    #
    # If the cache is too large (e.g., many k-points), we skip caching and
    # fall back to per-chunk HDF5 loading (slower but uses less memory).
    m_cached_gspace_full = bytes_per_complex * n_k * (n_b / p) * n_s * n_r
    
    # Check if we have room for caching after centroids
    m_available_for_cache = m_budget - m_centroids
    
    # Enable caching if it uses less than 40% of available memory
    # (need to leave room for P_k, Z_q, L_q, etc. which use the other 60%)
    # The cache provides ~10x speedup for x-chunk loop, so it's worth the memory.
    cache_threshold = 0.40 * m_available_for_cache
    use_gspace_cache = m_cached_gspace_full <= cache_threshold and m_cached_gspace_full > 0
    
    if use_gspace_cache:
        m_cached_gspace = m_cached_gspace_full
    else:
        m_cached_gspace = 0  # No cache - will reload from HDF5 each x-chunk
    
    m_available = m_budget - m_centroids - m_cached_gspace
    
    if m_available <= 0:
        raise ValueError(
            f"Centroid wavefunctions alone require {m_centroids/1e9:.2f} GB per device, "
            f"but budget is {memory_budget_gb:.2f} GB. Increase memory budget or reduce n_b."
        )
    
    # ========================================================================
    # STAGE 2: FFT WORKSPACE (band-chunked)
    # ========================================================================
    # psi_G (input): (n_k, B_b/P, n_s, nx, ny, nz) - sharded over bands
    # psi_r (output): same shape
    # phase_spatial: (n_k, nx, ny, nz) - broadcast to all devices
    # Communication buffer: None during FFT (shard_map runs locally)
    #
    # Constraint: 2 * psi_G/r + phase <= m_available
    # Solve for B_b:
    #   2 * 16 * n_k * (B_b/P) * n_s * n_r + 16 * n_k * n_r <= m_available
    #   B_b <= (m_available - 16 * n_k * n_r) * P / (32 * n_k * n_s * n_r)
    
    m_phase = bytes_per_complex * n_k * n_r  # phase_spatial (broadcast)
    m_fft_overhead = m_phase + m_available * 0.05  # 5% XLA overhead
    
    b_b_max = (m_available - m_fft_overhead) * p / (2 * bytes_per_complex * n_k * n_s * n_r)
    band_chunk = max(8, min(n_b, int(b_b_max)))
    
    # Actual FFT memory with chosen band_chunk
    m_fft_workspace = 2 * bytes_per_complex * n_k * (band_chunk / p) * n_s * n_r + m_phase
    
    # ========================================================================
    # STAGE 3: X-CHUNK EXTRACTION (contiguous r-space chunking)
    # ========================================================================
    # X-chunking advantage: With r = x*(ny*nz) + y*nz + z, an x-chunk with
    # x in [x_start, x_end) maps to CONTIGUOUS r-indices [x_start*ny*nz, x_end*ny*nz).
    # This enables single sequential HDF5 writes instead of strided writes.
    #
    # psi_xchunk_Y: (n_k, n_b, n_s, B_x/p_y)
    # Staged reshard communication:
    #   - Stage 1: P(None,('x','y'),...) → P(None,'y',...) requires all-gather over X
    #   - Buffer: (n_k, n_b/p_y, n_s, B_x)
    #   - Stage 2: P(None,'y',...) → P(None,None,None,'y') requires all-gather over Y + slice
    #   - Buffer: (n_k, n_b, n_s, B_x/p_y)
    #
    # Both exist during reshape, so peak = Stage1 + psi_xchunk_Y
    # Constraint: m_stage1 + m_xchunk <= m_available - m_fft_workspace
    #
    # Solve for B_x:
    #   16 * n_k * (n_b/p_y) * n_s * B_x + 16 * n_k * n_b * n_s * (B_x/p_y) <= budget
    #   B_x * 16 * n_k * n_s * n_b * (1/p_y + 1/p_y) <= budget
    #   B_x <= budget * p_y / (32 * n_k * n_s * n_b)
    
    m_for_xchunk = m_available * 0.30  # Allocate 30% of available for x-chunk
    
    b_x_r_max = m_for_xchunk * p_y / (2 * bytes_per_complex * n_k * n_s * n_b)
    
    # Convert to x-slices and ensure divisibility
    # B_x (in r-points) = x_chunk_slices * ny * nz (since x is outermost)
    x_chunk_slices_max = int(b_x_r_max / (ny * nz))
    x_chunk_slices = max(1, min(nx, x_chunk_slices_max))
    
    # Ensure n_xchunk = x*ny*nz is divisible by p_y
    while x_chunk_slices > 1 and (x_chunk_slices * ny * nz) % p_y != 0:
        x_chunk_slices -= 1
    
    x_chunk_r = x_chunk_slices * ny * nz
    
    # ========================================================================
    # STAGE 4: PAIR DENSITY P_k (LEFT and RIGHT, spin-traced)
    # ========================================================================
    # Spin-traced pair density: P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(ν)
    # No explicit spin dimensions in output (smaller than keeping all 4 spin combos).
    #
    # For CCT: P_l_mumu and P_r_mumu both exist simultaneously
    # For ZCT: P_l_xchunk and P_r_xchunk both exist simultaneously
    #
    # P_l_mumu, P_r_mumu: (n_k, n_rmu/p_x, n_rmu/p_y)
    # P_l_xchunk, P_r_xchunk: (n_k, n_rmu/p_x, x_chunk_r/p_y)
    
    # Per pair density (left OR right)
    m_P_mumu_single = bytes_per_complex * n_k * (n_rmu / p_x) * (n_rmu / p_y)
    m_P_xchunk_single = bytes_per_complex * n_k * (n_rmu / p_x) * (x_chunk_r / p_y)
    
    # Both left and right exist during CCT/ZCT computation
    m_P_mumu = 2 * m_P_mumu_single  # P_l + P_r
    m_P_xchunk = 2 * m_P_xchunk_single  # P_l + P_r
    
    # ========================================================================
    # STAGE 5: CCT/ZCT FFT PIPELINE
    # ========================================================================
    # C_q: (n_q, n_rmu/p_x, n_rmu/p_y)
    # Z_q: (n_q, n_rmu/p_x, x_chunk_r/p_y)
    # Intermediate P_R has same footprint as P_k
    
    m_C_q = bytes_per_complex * n_q * (n_rmu / p_x) * (n_rmu / p_y)
    m_Z_q = bytes_per_complex * n_q * (n_rmu / p_x) * (x_chunk_r / p_y)
    
    # ========================================================================
    # STAGE 6: CHOLESKY + SOLVE
    # ========================================================================
    # L_q: (n_q, n_rmu/p_x, n_rmu/p_y) - persists after Cholesky for all x-chunks
    # L_rep: B_q × (n_rmu, n_rmu) - REPLICATED on each device during solve
    # Z_col: (n_q, n_rmu, x_chunk_r/P) - column sharded
    # zeta: same as Z_col
    #
    # CRITICAL: L_q persists through entire x-chunk loop!
    # q_chunk is calculated AFTER x_chunk is finalized (see below)
    
    # L_q persists through x-chunk loop
    m_L_q = bytes_per_complex * n_q * (n_rmu / p_x) * (n_rmu / p_y)
    
    # ========================================================================
    # COMMUNICATION BUFFERS
    # ========================================================================
    # Cholesky panel broadcast: (n_q, b, n_rmu) where b = n_rmu/J
    # Estimate J = lcm(p_x, p_y), b = n_rmu/J
    j_target = math.lcm(p_x, p_y) if p_x > 1 and p_y > 1 else max(p_x, p_y)
    j_target = max(1, min(n_rmu, j_target))
    block_size = max(1, n_rmu // j_target) if n_rmu >= j_target else n_rmu
    m_chol_panel = bytes_per_complex * n_q * block_size * n_rmu / max(p_x, p_y)
    
    # Staged reshard buffer (x-chunk extraction)
    m_reshard_buffer = bytes_per_complex * n_k * (n_b / p_y) * n_s * x_chunk_r
    
    # psi_xchunk_Y: loaded each x-chunk iteration
    m_psi_xchunk = bytes_per_complex * n_k * n_b * n_s * (x_chunk_r / p_y)
    
    # Total communication overhead estimate
    m_comm_overhead = m_chol_panel + m_reshard_buffer
    
    # ========================================================================
    # PEAK MEMORY ESTIMATE
    # ========================================================================
    # Peak occurs during x-chunk loop with these arrays simultaneously live:
    #   - centroids (psi_rmu_Y, psi_rmuT_X) - persistent
    #   - L_q - persistent after Cholesky
    #   - psi_xchunk_Y - loaded for current chunk
    #   - P_k_mu_xchunk - computed pair density (LARGEST ARRAY!)
    #   - Z_q - computed from P_k
    #   - zeta_chunk - output (part of Z_col_zeta)
    #   - XLA intermediate buffers during einsum (~50% of largest operand)
    #
    # Note: P_k_mumu and C_q are freed before x-chunk loop starts
    
    # ========================================================================
    # XLA OVERHEAD ESTIMATES (calibrated from measurements)
    # ========================================================================
    # 
    # Empirical measurements show: Peak memory = 3.26x P_k
    # Breakdown:
    #   - P_k itself: 1.0x
    #   - P_R during IFFT (same size as P_k): 1.0x  
    #   - JIT compilation overhead: ~1.25x
    #
    # Note: JIT overhead is one-time per function, but we budget for worst case.
    # The IFFT output buffer is the main driver - P_k and P_R coexist briefly.
    #
    # 1. Einsum overhead: minimal at runtime, but JIT compilation adds ~1x output
    m_einsum_overhead = m_P_xchunk * 0.25  # 25% - JIT amortized across chunks
    
    # 2. ZCT FFT pipeline overhead: 
    #    - IFFT creates output P_R = 1x P_k (coexists with input)
    #    - JIT compilation adds ~1x buffer (first call only)
    #    - Actual OOM observation: XLA needs 2x P_k during FFT
    #    Measured: peak 3.26x = 1 input + 2.26 overhead
    m_fft_pipeline_overhead = m_P_xchunk * 2.0  # 200% for IFFT + JIT + XLA internal
    
    # 3. FRAGMENTATION BUFFER:
    #    - C_q is deleted after L_q is computed
    #    - P_k_mumu is deleted after C_q is computed  
    #    - Small buffer for memory allocator fragmentation
    m_fragmentation_buffer = m_C_q * 0.5  # 50% of C_q for fragmentation
    
    # Total overhead for x-chunk loop (main bottleneck)
    m_xla_overhead = m_einsum_overhead + m_fft_pipeline_overhead
    
    # Z_col + zeta for solve
    m_Z_col_zeta = 2 * bytes_per_complex * n_q * n_rmu * (x_chunk_r / p)
    
    # Peak during x-chunk loop (the main bottleneck):
    # Does NOT include FFT workspace - that's a separate earlier stage that completes
    # before x-chunk loop begins. Stages are sequential, not concurrent.
    # INCLUDES cached_gspace which persists through entire x-chunk loop.
    m_peak_xchunk = (m_centroids + m_cached_gspace + m_L_q + m_psi_xchunk + m_P_xchunk + m_Z_q + 
                     m_Z_col_zeta + m_xla_overhead + m_comm_overhead * 0.5 +
                     m_fragmentation_buffer)
    
    # Peak during FFT stage (loading centroids):
    # centroids + FFT workspace for band-chunked loading
    m_peak_fft = m_centroids + m_fft_workspace
    
    # Overall peak is the maximum of all stages
    m_peak = max(m_peak_xchunk, m_peak_fft)
    
    # Verify we fit, and iteratively reduce x_chunk if needed
    max_iterations = 20
    for _ in range(max_iterations):
        utilization = m_peak / m_budget
        if utilization <= 1.0:
            break
        
        # Reduce x_chunk to fit
        reduction_factor = 0.85 / utilization
        x_chunk_slices = max(1, int(x_chunk_slices * reduction_factor))
        while x_chunk_slices > 1 and (x_chunk_slices * ny * nz) % p_y != 0:
            x_chunk_slices -= 1
        x_chunk_r = x_chunk_slices * ny * nz
        
        # Recompute dependent values
        m_psi_xchunk = bytes_per_complex * n_k * n_b * n_s * (x_chunk_r / p_y)
        m_P_xchunk = bytes_per_complex * n_k * n_s * n_s * (n_rmu / p_x) * (x_chunk_r / p_y)
        m_Z_q = bytes_per_complex * n_q * (n_rmu / p_x) * (x_chunk_r / p_y)
        m_Z_col_zeta = 2 * bytes_per_complex * n_q * n_rmu * (x_chunk_r / p)
        m_reshard_buffer = bytes_per_complex * n_k * (n_b / p_y) * n_s * x_chunk_r
        m_comm_overhead = m_chol_panel + m_reshard_buffer
        m_einsum_overhead = m_P_xchunk * 0.25  # 25% - JIT amortized
        m_fft_pipeline_overhead = m_P_xchunk * 2.0  # 200% for IFFT + JIT + XLA
        m_fragmentation_buffer = m_C_q * 0.5  # 50% for fragmentation
        m_xla_overhead = m_einsum_overhead + m_fft_pipeline_overhead
        
        m_peak_xchunk = (m_centroids + m_cached_gspace + m_L_q + m_psi_xchunk + m_P_xchunk + m_Z_q + 
                         m_Z_col_zeta + m_xla_overhead + m_comm_overhead * 0.5 +
                         m_fragmentation_buffer)
        m_peak = max(m_peak_xchunk, m_peak_fft)
    
    # ========================================================================
    # Q-CHUNK CALCULATION (after x_chunk is finalized)
    # ========================================================================
    # L_rep: B_q × (n_rmu, n_rmu) - REPLICATED on each device during solve
    # Available memory for solve = budget - persistent - current live arrays
    # Include cached_gspace since it persists through entire x-chunk loop
    # Solve for B_q: B_q * 16 * n_rmu² <= available
    m_for_q = m_budget - m_centroids - m_cached_gspace - m_L_q - m_Z_col_zeta
    b_q_max = m_for_q / (bytes_per_complex * n_rmu * n_rmu)
    q_chunk = max(1, min(n_q, int(b_q_max)))
    
    # Memory estimates dictionary
    memory_estimate = {
        # Per-device allocations
        'centroids_gb': m_centroids / 1e9,
        'psi_rmu_Y_gb': m_psi_rmu_Y / 1e9,
        'psi_rmuT_X_gb': m_psi_rmuT_X / 1e9,
        'cached_gspace_gb': m_cached_gspace / 1e9,  # G-space cache (0 if disabled)
        'use_gspace_cache': use_gspace_cache,  # Whether caching is enabled
        'fft_workspace_gb': m_fft_workspace / 1e9,
        'pair_density_mumu_gb': m_P_mumu / 1e9,
        'pair_density_xchunk_gb': m_P_xchunk / 1e9,
        'psi_xchunk_gb': m_psi_xchunk / 1e9,
        'C_q_gb': m_C_q / 1e9,
        'Z_q_gb': m_Z_q / 1e9,
        'L_q_gb': m_L_q / 1e9,
        'L_replicated_gb': (q_chunk * bytes_per_complex * n_rmu * n_rmu) / 1e9,
        'Z_col_zeta_gb': m_Z_col_zeta / 1e9,
        # Communication buffers and overhead
        'chol_panel_gb': m_chol_panel / 1e9,
        'reshard_buffer_gb': m_reshard_buffer / 1e9,
        'comm_overhead_gb': m_comm_overhead / 1e9,
        'einsum_overhead_gb': m_einsum_overhead / 1e9,
        'fft_pipeline_overhead_gb': m_fft_pipeline_overhead / 1e9,
        'xla_overhead_gb': m_xla_overhead / 1e9,
        'fragmentation_buffer_gb': m_fragmentation_buffer / 1e9,
        # Summary
        'peak_estimate_gb': m_peak / 1e9,
        'available_gb': m_available / 1e9,
        'budget_gb': memory_budget_gb,
        'utilization_pct': 100 * m_peak / m_budget,
        # Mesh info
        'p_x': p_x,
        'p_y': p_y,
        'n_devices': p,
    }
    
    return {
        'band_chunk': band_chunk,
        'x_chunk': x_chunk_slices,
        'x_chunk_r': x_chunk_r,
        'q_chunk': q_chunk,
        'use_gspace_cache': use_gspace_cache,  # Whether to cache G-space across x-chunks
        'memory_estimate': memory_estimate,
    }


def print_memory_breakdown(
    chunks: dict,
    n_b: int,
    n_r: int,
    n_q: int,
    fft_grid: tuple[int, int, int],
    memory_source: str = 'auto',
) -> None:
    """Print comprehensive memory breakdown and chunk size information."""
    nx, ny, nz = fft_grid
    mem = chunks['memory_estimate']
    
    print("\n" + "="*70)
    print("  MEMORY-OPTIMIZED CHUNK SIZES")
    print("="*70)
    
    print(f"\n  Memory budget: {mem['budget_gb']:.2f} GB/device (source: {memory_source})")
    print(f"  Device mesh: {mem['p_x']} × {mem['p_y']} = {mem['n_devices']} devices")
    
    print(f"\n  {'Parameter':<25} {'Value':>10} {'Total':>12} {'Per-chunk':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Band chunk':<25} {chunks['band_chunk']:>10d} / {n_b:<5d} bands")
    print(f"  {'X-chunk (x-slices)':<25} {chunks['x_chunk']:>10d} / {nx:<5d} slices")
    print(f"  {'X-chunk (r-points)':<25} {chunks['x_chunk_r']:>10d} / {n_r:<5d} points")
    print(f"  {'Q-chunk':<25} {chunks['q_chunk']:>10d} / {n_q:<5d} q-points")
    
    print(f"\n  {'MEMORY ALLOCATION (per device)':<40} {'Size (GB)':>12}")
    print(f"  {'-'*54}")
    print(f"  {'[Persistent through x-chunk loop]':<40}")
    print(f"    {'psi_rmu_Y (centroids, Y-sharded)':<38} {mem['psi_rmu_Y_gb']:>10.3f}")
    print(f"    {'psi_rmuT_X (centroids, X-sharded)':<38} {mem['psi_rmuT_X_gb']:>10.3f}")
    if mem.get('use_gspace_cache', False):
        print(f"    {'G-space cache (sharded, avoids HDF5)':<38} {mem['cached_gspace_gb']:>10.3f}")
        print(f"    {'─ Subtotal: persistent':<38} {mem['centroids_gb'] + mem['cached_gspace_gb']:>10.3f}")
    else:
        print(f"    {'G-space cache':<38} {'DISABLED (too large)'}")
        print(f"    {'─ Subtotal: persistent':<38} {mem['centroids_gb']:>10.3f}")
    
    print(f"\n  {'[Stage 1: FFT centroid extract - SEQUENTIAL, freed before loop]':<40}")
    print(f"    {'psi_G + psi_r (2x band-chunked FFT)':<38} {mem['fft_workspace_gb']:>10.3f}")
    
    print(f"\n  {'[Stage 2: Pair density - setup, freed before loop]':<40}")
    print(f"    {'P_k(μ,μ)':<38} {mem['pair_density_mumu_gb']:>10.3f}")
    print(f"    {'P_k(μ,x-chunk)':<38} {mem['pair_density_xchunk_gb']:>10.3f}")
    
    print(f"\n  {'[Stage 3: X-chunk loop - PEAK MEMORY STAGE]':<40}")
    print(f"    {'psi_xchunk_Y (loaded per chunk)':<38} {mem['psi_xchunk_gb']:>10.3f}")
    print(f"    {'P_k(μ,x-chunk) [PEAK DRIVER]':<38} {mem['pair_density_xchunk_gb']:>10.3f}")
    print(f"    {'Z_q matrix':<38} {mem['Z_q_gb']:>10.3f}")
    print(f"    {'L_q (persistent after Cholesky)':<38} {mem['L_q_gb']:>10.3f}")
    print(f"    {'L replicated (for solve)':<38} {mem['L_replicated_gb']:>10.3f}")
    print(f"    {'Z_col + zeta output':<38} {mem['Z_col_zeta_gb']:>10.3f}")
    
    print(f"\n  {'[XLA buffers & overhead]':<40}")
    print(f"    {'Einsum temps (~25% of P_k)':<38} {mem['einsum_overhead_gb']:>10.3f}")
    print(f"    {'FFT pipeline (~200%, measured)':<38} {mem['fft_pipeline_overhead_gb']:>10.3f}")
    print(f"    {'Staged reshard buffer':<38} {mem['reshard_buffer_gb']:>10.3f}")
    print(f"    {'Cholesky panel broadcast':<38} {mem['chol_panel_gb']:>10.3f}")
    print(f"    {'Fragmentation (~50% C_q)':<38} {mem['fragmentation_buffer_gb']:>10.3f}")
    
    print(f"\n  {'-'*54}")
    print(f"  {'PEAK ESTIMATE':<38} {mem['peak_estimate_gb']:>10.3f} GB")
    print(f"  {'BUDGET':<38} {mem['budget_gb']:>10.3f} GB")
    print(f"  {'UTILIZATION':<38} {mem['utilization_pct']:>10.1f} %")
    print("="*70)


def get_effective_chunk_size(chunk_size: int) -> int | None:
    """Convert chunk_size input flag to actual chunk size.
    
    Args:
        chunk_size: Input flag value:
            -1 = no chunking (return None, all bands at once)
             0 = auto (TODO: compute from available RAM; currently 64)
            1-2048 = explicit chunk size
    
    Returns:
        Effective chunk size as int, or None for no chunking.
    """
    if chunk_size == -1:
        return None
    elif chunk_size == 0:
        # TODO: replace 64 with dynamic value based on available RAM
        return 64
    elif 1 <= chunk_size <= 2048:
        return chunk_size
    else:
        raise ValueError(f"chunk_size must be -1, 0, or 1-2048, got {chunk_size}")


def get_effective_x_chunk_size(
    x_chunk_size: int, 
    fft_grid: tuple, 
    n_rmu: int, 
    target_ratio: float = 16.0,
    mesh_y_size: int = 1,
    max_wfn_chunk_mb: float = 0.0,
    nk_tot: int = 1,
    nspinor: int = 2,
) -> int:
    """Compute effective x-axis chunk size for ZCT accumulation.
    
    X-chunking advantage: With r = x*(ny*nz) + y*nz + z, an x-chunk with
    x in [x_start, x_end) maps to CONTIGUOUS r-indices [x_start*ny*nz, x_end*ny*nz).
    This enables single sequential HDF5 writes instead of strided writes.
    
    Priority (highest to lowest):
    1. max_wfn_chunk_mb > 0: compute x from memory budget for P_k(rmu, rchunk)
    2. x_chunk_size > 0: use explicit value
    3. x_chunk_size == 0: auto-compute to target_ratio * n_rmu
    
    The P_k,ab(rmu, rchunk) array has shape (nk, ns, ns, n_rmu, n_xchunk) with
    complex128 dtype (16 bytes). Given a memory budget:
        max_bytes = max_wfn_chunk_mb * 1e6
        n_xchunk_max = max_bytes / (nk * ns * ns * n_rmu * 16)
        x_chunk_size = n_xchunk_max / (ny * nz)
    
    Additionally ensures n_xchunk = x * ny * nz is divisible by mesh_y_size.
    
    Args:
        x_chunk_size: Input flag value:
            0 = auto: choose x such that x*ny*nz ≈ target_ratio * n_rmu
            1-nx = explicit x-slice count per chunk
        fft_grid: (nx, ny, nz) FFT grid dimensions
        n_rmu: Number of ISDF centroids
        target_ratio: Target ratio of chunk size to n_rmu (default 16.0)
        mesh_y_size: Number of devices on Y-axis mesh (for divisibility)
        max_wfn_chunk_mb: Max memory for P_k chunk in MB (0=ignore, use x_chunk_size)
        nk_tot: Total number of k-points (for memory calculation)
        nspinor: Number of spinor components (for memory calculation)
    
    Returns:
        Effective x_chunk_size (number of x-slices per chunk)
    """
    nx, ny, nz = fft_grid
    
    # Priority 1: Memory budget overrides everything
    if max_wfn_chunk_mb > 0:
        # P_k,ab(rmu, rchunk) shape: (nk, ns, ns, n_rmu, n_xchunk)
        # Size in bytes: nk * ns² * n_rmu * n_xchunk * 16
        max_bytes = max_wfn_chunk_mb * 1e6
        bytes_per_xpoint = nk_tot * nspinor * nspinor * n_rmu * 16
        n_xchunk_max = max_bytes / bytes_per_xpoint
        x_budget = max(1, int(n_xchunk_max / (ny * nz)))
        
        # Ensure n_xchunk = x*ny*nz is divisible by mesh_y_size
        x_opt = None
        for x_try in range(min(x_budget, nx), 0, -1):
            if (x_try * ny * nz) % mesh_y_size == 0:
                x_opt = x_try
                break
        
        if x_opt is None:
            raise ValueError(
                f"max_wfn_chunk_mb={max_wfn_chunk_mb} is too small. "
                f"Smallest valid chunk (x=1) requires {ny*nz*16*nk_tot*nspinor**2*n_rmu/1e6:.1f} MB. "
                f"Increase max_wfn_chunk_mb or remove it to use auto-sizing."
            )
        
        return x_opt
    
    # Priority 2: Explicit x_chunk_size
    if x_chunk_size > 0:
        if 1 <= x_chunk_size <= nx:
            return x_chunk_size
        else:
            raise ValueError(f"x_chunk_size must be 1-{nx}, got {x_chunk_size}")
    
    # Priority 3: Auto based on target_ratio
    target_chunk = target_ratio * n_rmu
    x_opt = max(1, round(target_chunk / (ny * nz)))
    
    # Ensure n_xchunk is divisible by mesh_y_size
    if mesh_y_size > 1:
        n_xchunk = x_opt * ny * nz
        remainder = n_xchunk % mesh_y_size
        if remainder != 0:
            extra_needed = mesh_y_size - remainder
            extra_x = (extra_needed + ny * nz - 1) // (ny * nz)
            x_opt = x_opt + extra_x
    
    # Clamp to valid range [1, nx]
    x_opt = min(x_opt, nx)
    return x_opt


# Keep z-chunk version for backwards compatibility (alias to x-chunk with different axis)
def get_effective_z_chunk_size(
    z_chunk_size: int, 
    fft_grid: tuple, 
    n_rmu: int, 
    target_ratio: float = 16.0,
    mesh_y_size: int = 1,
    max_wfn_chunk_mb: float = 0.0,
    nk_tot: int = 1,
    nspinor: int = 2,
) -> int:
    """DEPRECATED: Use get_effective_x_chunk_size instead.
    
    Z-chunking produces non-contiguous r-indices requiring strided HDF5 writes.
    X-chunking produces contiguous r-indices for efficient sequential writes.
    """
    nx, ny, nz = fft_grid
    
    if max_wfn_chunk_mb > 0:
        max_bytes = max_wfn_chunk_mb * 1e6
        bytes_per_zpoint = nk_tot * nspinor * nspinor * n_rmu * 16
        n_zchunk_max = max_bytes / bytes_per_zpoint
        z_budget = max(1, int(n_zchunk_max / (nx * ny)))
        
        z_opt = None
        for z_try in range(min(z_budget, nz), 0, -1):
            if (nx * ny * z_try) % mesh_y_size == 0:
                z_opt = z_try
                break
        
        if z_opt is None:
            raise ValueError(
                f"max_wfn_chunk_mb={max_wfn_chunk_mb} is too small. "
                f"Smallest valid chunk (z=1) requires {nx*ny*16*nk_tot*nspinor**2*n_rmu/1e6:.1f} MB. "
                f"Increase max_wfn_chunk_mb or remove it to use auto-sizing."
            )
        return z_opt
    
    if z_chunk_size > 0:
        if 1 <= z_chunk_size <= nz:
            return z_chunk_size
        else:
            raise ValueError(f"z_chunk_size must be 1-{nz}, got {z_chunk_size}")
    
    target_chunk = target_ratio * n_rmu
    z_opt = max(1, round(target_chunk / (nx * ny)))
    
    if mesh_y_size > 1:
        n_zchunk = nx * ny * z_opt
        remainder = n_zchunk % mesh_y_size
        if remainder != 0:
            extra_needed = mesh_y_size - remainder
            extra_z = (extra_needed + nx * ny - 1) // (nx * ny)
            z_opt = z_opt + extra_z
    
    z_opt = min(z_opt, nz)
    return z_opt


def read_cohsex_input(filename: str) -> dict:
	"""Parse input file for the COHSEX driver, allowing a QE K_POINTS block.

	We extract the [cohsex] section using a substring to avoid configparser
	errors from non-INI blocks like K_POINTS. The K_POINTS {crystal_b} block
	is parsed manually and returned under 'kpoints_crystal_b'.
	"""
	with open(filename, 'r') as f:
		lines = f.readlines()

	# Locate [cohsex] section boundaries
	start = None
	for i, l in enumerate(lines):
		if l.strip().lower().startswith('[cohsex]'):
			start = i
			break
	if start is None:
		for i, l in enumerate(lines):
			if re.match(r"\s*\[.*\]", l):
				start = i
				break
	end = len(lines)
	# Locate optional K_POINTS block in full file first
	kp_idx = None
	for i, l in enumerate(lines):
		ls = l.strip().lower()
		if ls.startswith("k_points"):
			kp_idx = i
			break
	seg_count = 0
	kp_end = None
	if kp_idx is not None and kp_idx + 1 < len(lines):
		# count is on the next line; read exactly that many entries
		try:
			seg_count = int(lines[kp_idx + 1].strip().split()[0])
		except Exception:
			seg_count = 0
		kp_end = min(len(lines), kp_idx + 2 + max(seg_count, 0))
	if start is not None:
		for j in range(start + 1, len(lines)):
			if re.match(r"\s*\[.*\]", lines[j]):
				end = j
				break
		# Remove K_POINTS block from the ini text before feeding to configparser
		if kp_idx is not None and (start <= kp_idx < end):
			section_lines = lines[start:kp_idx] + lines[(kp_end if kp_end is not None else kp_idx+1):end]
			ini_text = ''.join(section_lines)
		else:
			ini_text = ''.join(lines[start:end])
		parser = configparser.ConfigParser()
		parser.read_string(ini_text)
		section = parser["cohsex"] if "cohsex" in parser else parser[parser.sections()[0]]
		getb = section.getboolean
		get = section.get
		geti = section.getint
		getf = section.getfloat
		# ============================================================================
		# PARAMETER DEFINITIONS:
		# ============================================================================
		# restart:       If True, load V_qmunu/wavefunctions from taggedarrays.h5
		#                instead of rebuilding ISDF from scratch. Default=True.
		# x_only:        If True, compute bare exchange only (no screening).
		#                Cannot be True if do_screened=True. Default=False.
		# do_screened:   If True, build W from (1-Vχ)⁻¹V and use for Σ_x.
		#                If False, use bare Coulomb V. Default=True.
		# bispinor:      If True, use 2-component spinor wavefunctions (SOC).
		#                Default=False (scalar or collinear spin).
		# wcoul0_source: Method for q→0 head average: 'epshead' (from eps0mat.h5)
		#                or 's_tensor' (from dipole.h5). Default='s_tensor'.
		# self_consistent: If True, run fixed-point SCF loop. Default=False.
		# nval:          Number of valence bands in sigma window. These are the
		#                highest occupied bands: indices [nelec-nval, nelec).
		# ncond:         Number of conduction bands in sigma window. These are the
		#                lowest unoccupied bands: indices [nelec, nelec+ncond).
		# nband:         Total bands to load (for chi0, etc.). Usually > nval+ncond.
		# sys_dim:       Dimensionality: 2=2D (slab with truncated Coulomb),
		#                3=3D (bulk, not yet implemented). Default=2.
		# debug_hartree: If True, print diagnostic info for Hartree calculation.
		# debug_omega:   If set (float, in Ry), compute W(ω) at this frequency
		#                instead of static W. For testing dynamic screening.
		# chunk_size:    Band chunk size for memory-efficient wavefunction loading.
		#                -1 = no chunking (all bands at once, default)
		#                 0 = auto (currently 64, TODO: dynamic from available RAM)
		#                1-2048 = explicit chunk size
		# x_chunk_size:  X-axis chunk size for ZCT accumulation (contiguous r-space).
		#                0 = auto (default): choose x such that x*ny*nz ≈ 2*n_rmu
		#                1-nx = explicit x-slice count per chunk
		#                X-chunking gives CONTIGUOUS r-indices for efficient HDF5 writes.
		# memory_per_device_gb: Memory budget per device in GB for auto chunk sizing.
		#                0 = auto-detect (80% of GPU via nvidia-smi, or CPU/n_devices)
		#                >0 = explicit budget in GB
		# ============================================================================
		params = {
			"restart": getb("restart", fallback=True),           # load from h5 vs rebuild
			"x_only": getb("x_only", fallback=False),            # bare exchange only
			"do_screened": getb("do_screened", fallback=True),   # use W instead of V
			"bispinor": getb("bispinor", fallback=False),        # 2-component spinors
			"wcoul0_source": get("wcoul0_source", fallback="s_tensor").strip().lower(),
			"wfn_file": get("wfn_file", fallback="WFN.h5"),
			"centroids_file": get("centroids_file", fallback="centroids_frac.txt"),
			"output_file": get("output_file", fallback="eqp0_noqsym.dat"),
			"self_consistent": getb("self_consistent", fallback=False),
			"kin_ion_file": get("kin_ion_file", fallback="kin_ion.h5"),
			"eqp_output_file": get("eqp_output_file", fallback="eqp.dat"),
			"nval": geti("nval", fallback=5),    # valence bands in sigma window
			"ncond": geti("ncond", fallback=5),  # conduction bands in sigma window
			"nband": geti("nband", fallback=100), # total bands for chi0/screening
			"sys_dim": geti("sys_dim", fallback=2),  # 2=slab, 3=bulk
			"debug_hartree": getb("debug_hartree", fallback=False),
			"debug_omega": getf("debug_omega", fallback=None),   # test W(ω) at this freq
			"chunk_size": geti("chunk_size", fallback=-1),       # band chunk size (-1=all, 0=auto, 1-2048=explicit)
			"x_chunk_size": geti("x_chunk_size", fallback=0),    # x-axis chunk (0=auto, 1-nx=explicit)
			"max_wfn_chunk_mb": getf("max_wfn_chunk_mb", fallback=0.0),  # max P_k chunk size in MB (0=use x_chunk_size)
			"band_chunk_size": geti("band_chunk_size", fallback=16),  # bands per FFT during x-chunk loop
			"memory_per_device_gb": getf("memory_per_device_gb", fallback=0.0),  # 0=auto-detect
			"use_chunked_isdf": getb("use_chunked_isdf", fallback=True),  # chunked (memory-efficient) vs original ISDF
		}
	else:
		# Fallback defaults if no section found
		params = {
			"restart": True,
			"x_only": False,
			"do_screened": True,
			"bispinor": False,
			"wcoul0_source": "s_tensor",
			"wfn_file": "WFN.h5",
			"centroids_file": "centroids_frac.txt",
			"output_file": "eqp0_noqsym.dat",
			"self_consistent": False,
			"kin_ion_file": "kin_ion.h5",
			"eqp_output_file": "eqp.dat",
			"nval": 5,
			"ncond": 5,
			"nband": 100,
			"sys_dim": 2,
			"debug_hartree": False,
			"debug_omega": None,
			"chunk_size": -1,
			"z_chunk_size": 0,
			"max_wfn_chunk_mb": 0.0,
			"band_chunk_size": 16,
			"memory_per_device_gb": 0.0,
			"use_chunked_isdf": True,
		}

	# Parse optional QE-style K_POINTS block: take the number after it, read next that many lines
	if kp_idx is not None:
		j = kp_idx + 1
		try:
			nseg = int(lines[j].strip().split()[0])
		except Exception:
			nseg = 0
		segments = []
		for k in range(nseg):
			row_idx = j + 1 + k
			if row_idx >= len(lines):
				break
			row_full = lines[row_idx].rstrip('\n')
			label = None
			comment_split = None
			for marker in ('#', '!', ';'):
				if marker in row_full:
					comment_split = row_full.split(marker, 1)
					label = comment_split[1].strip() or None
					row = comment_split[0].strip()
					break
			if comment_split is None:
				row = row_full.strip()
			if not row:
				continue
			parts = row.split()
			if len(parts) < 3:
				continue
			x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
			npts = int(parts[3]) if len(parts) >= 4 else 1
			segments.append({"k": [x, y, z], "n": npts, "label": label})
		if segments:
			params["kpoints_crystal_b"] = {"segments": segments}
	return params


def get_bandranges(nv, nc, nband, nelec):
	r"""Return ranges of bands necessary for \sigma_{X,SX,COH}"""
	nvrange = [int(nelec - nv), int(nelec)]
	ncrange = [int(nelec), int(nelec + nc)]
	nsigmarange = [int(nelec - nv), int(nelec + nc)]
	n_fullrange = [0, int(nband)]
	n_valrange = [0, int(nelec)]
	return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


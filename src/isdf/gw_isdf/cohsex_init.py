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
    n_b_left: int | None = None,
    n_b_right: int | None = None,
) -> dict:
    """Derive chunk sizes that saturate (but do not exceed) the memory budget."""
    if memory_budget_gb <= 0:
        raise ValueError("memory_per_device_gb must be > 0 for automatic chunk sizing.")
    if n_devices <= 0:
        raise ValueError("n_devices must be at least 1.")

    bytes_per_complex = 16.0
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    p = max(1, int(n_devices))
    n_b_full = int(n_b)
    nb_left = int(n_b_left) if n_b_left is not None else n_b_full
    nb_right = int(n_b_right) if n_b_right is not None else n_b_full

    # Default mesh: prefer the most square factorisation.
    if p_x is None or p_y is None:
        sqrt_p = int(math.sqrt(p))
        while sqrt_p > 1 and p % sqrt_p != 0:
            sqrt_p -= 1
        p_x = sqrt_p
        p_y = p // p_x
    if p_x <= 0 or p_y <= 0:
        raise ValueError(f"Invalid mesh dimensions p_x={p_x}, p_y={p_y}.")

    def to_gb(value: float) -> float:
        return value / 1e9

    m_budget = memory_budget_gb * 1e9 * target_utilization
    n_k_f = float(n_k)
    n_s_f = float(n_s)
    n_rmu_f = float(n_rmu)
    n_q_f = float(n_q)
    n_r_f = float(n_r)

    # Stage 0: centroid storage
    m_centroids_full = bytes_per_complex * n_k_f * n_s_f * n_rmu_f * n_b_full * (1 / p_y + 1 / p_x)
    m_centroids_persist = bytes_per_complex * n_k_f * n_s_f * n_rmu_f * (nb_left + nb_right) * (1 / p_y + 1 / p_x)
    m_centroid_copy_peak = m_centroids_full + m_centroids_persist
    if m_centroid_copy_peak > m_budget:
        raise ValueError(
            f"Centroid storage requires {to_gb(m_centroid_copy_peak):.2f} GB/device but only "
            f"{memory_budget_gb:.2f} GB were allocated. Reduce band counts or increase the budget."
        )

    # Stage 1: band-chunked FFT workspace for centroid extraction
    phase_bytes = bytes_per_complex * n_k_f * n_r_f
    headroom_fft = m_budget - m_centroids_full
    if headroom_fft <= phase_bytes:
        raise ValueError("Insufficient memory for even a single-band FFT chunk.")

    fft_denom = 2 * bytes_per_complex * n_k_f * n_s_f * n_r_f
    b_b_max = (headroom_fft - phase_bytes) * p / fft_denom
    if b_b_max < 1:
        raise ValueError(
            "Band chunk computation produced <1 band. Increase the memory budget or decrease system size."
        )
    band_chunk = max(1, min(n_b_full, int(b_b_max)))
    m_fft_workspace = 2 * bytes_per_complex * n_k_f * (band_chunk / p) * n_s_f * n_r_f + phase_bytes
    peak_fft_stage = m_centroids_full + m_fft_workspace

    # Stage 2: pair-density build for C_q (before x-chunk loop)
    m_pair_mumu = bytes_per_complex * n_k_f * (n_rmu_f / p_x) * (n_rmu_f / p_y)
    m_C_q = bytes_per_complex * n_q_f * (n_rmu_f / p_x) * (n_rmu_f / p_y)
    stage_cct = m_centroids_persist + 2 * m_pair_mumu + m_C_q
    if stage_cct > m_budget:
        raise ValueError(
            f"C_q build requires {to_gb(stage_cct):.2f} GB/device which exceeds the budget. Reduce n_rmu or n_q."
        )
    m_L_q = m_C_q

    # Optional G-space cache (sum of all cached band chunks)
    m_cached_gspace_full = bytes_per_complex * n_k_f * (n_b_full / p) * n_s_f * n_r_f

    def compute_chunk_metrics(x_chunk_r: int, base_const: float) -> tuple[float, str, dict]:
        chunk_r = float(x_chunk_r)
        per_y = chunk_r / p_y
        m_xchunk = bytes_per_complex * n_k_f * n_b_full * n_s_f * per_y
        m_pair = bytes_per_complex * n_k_f * (n_rmu_f / p_x) * per_y
        m_z = bytes_per_complex * n_q_f * (n_rmu_f / p_x) * per_y
        m_zcol = bytes_per_complex * n_q_f * n_rmu_f * (chunk_r / p)
        stage_pair = base_const + m_xchunk + 2 * m_pair
        stage_zct = base_const + 2 * m_pair + m_z
        stage_solve = base_const + 2 * m_zcol + bytes_per_complex * (n_rmu_f ** 2)
        peak = max(stage_pair, stage_zct, stage_solve)
        if peak == stage_pair:
            name = 'pair'
        elif peak == stage_zct:
            name = 'zct'
        else:
            name = 'solve'
        info = {
            'psi_xchunk': m_xchunk,
            'pair_xchunk': 2 * m_pair,
            'Z_q': m_z,
            'Z_col': m_zcol,
            'stage_pair': stage_pair,
            'stage_zct': stage_zct,
            'stage_solve': stage_solve,
        }
        return peak, name, info

    def choose_x_chunk(use_cache: bool) -> dict | None:
        m_cache = m_cached_gspace_full if use_cache else 0.0
        base_const = m_centroids_persist + m_L_q + m_cache
        headroom = m_budget - base_const
        if headroom <= bytes_per_complex * (n_rmu_f ** 2):
            return None

        limits = []
        denom_pair = n_k_f * n_b_full * n_s_f + 2 * n_k_f * (n_rmu_f / p_x)
        if denom_pair > 0:
            limits.append(headroom * p_y / (bytes_per_complex * denom_pair))
        denom_zct = (n_rmu_f / p_x) * (2 * n_k_f + n_q_f)
        if denom_zct > 0:
            limits.append(headroom * p_y / (bytes_per_complex * denom_zct))
        solve_numer = headroom - bytes_per_complex * (n_rmu_f ** 2)
        denom_solve = 2 * n_q_f * n_rmu_f / p
        if denom_solve > 0 and solve_numer > 0:
            limits.append(solve_numer * p / (bytes_per_complex * denom_solve))
        if not limits:
            limits.append(float(n_r))
        x_chunk_r_guess = min(float(n_r), max(float(ny * nz), min(limits)))

        x_slices = min(nx, max(1, int(x_chunk_r_guess // (ny * nz))))

        def make_divisible(val: int) -> int:
            out = val
            while out > 1 and (out * ny * nz) % p_y != 0:
                out -= 1
            return out

        x_slices = make_divisible(x_slices)
        if x_slices <= 0:
            return None

        while x_slices > 0:
            x_chunk_r = x_slices * ny * nz
            peak_bytes, stage_name, info = compute_chunk_metrics(x_chunk_r, base_const)
            if peak_bytes <= m_budget:
                return {
                    'x_chunk': x_slices,
                    'x_chunk_r': x_chunk_r,
                    'peak_bytes': peak_bytes,
                    'stage_name': stage_name,
                    'stage_info': info,
                    'cache_bytes': m_cache,
                    'base_const': base_const,
                }
            x_slices = make_divisible(x_slices - 1)
        return None

    chunk_result = choose_x_chunk(use_cache=True) or choose_x_chunk(use_cache=False)
    if chunk_result is None:
        raise ValueError(
            "Unable to find an x_chunk that fits the memory budget (with or without G-space caching)."
        )

    use_gspace_cache = chunk_result['cache_bytes'] > 0
    x_chunk_slices = chunk_result['x_chunk']
    x_chunk_r = chunk_result['x_chunk_r']
    stage_info = chunk_result['stage_info']
    peak_chunk_bytes = chunk_result['peak_bytes']
    base_const = chunk_result['base_const']

    m_Z_col = stage_info['Z_col']
    l_rep_bytes = bytes_per_complex * (n_rmu_f ** 2)
    base_solve = base_const + 2 * m_Z_col
    available_for_q = max(0.0, m_budget - base_solve)
    if available_for_q < l_rep_bytes:
        q_chunk = 1
    else:
        q_chunk = max(1, min(n_q, int(available_for_q // l_rep_bytes)))

    overall_peak = max(peak_chunk_bytes, peak_fft_stage, m_centroid_copy_peak, stage_cct)
    bottleneck = chunk_result['stage_name'] if peak_chunk_bytes >= max(peak_fft_stage, m_centroid_copy_peak, stage_cct) else (
        'fft' if peak_fft_stage >= max(m_centroid_copy_peak, stage_cct) else (
            'centroid_copy' if m_centroid_copy_peak >= stage_cct else 'C_q'
        )
    )

    memory_estimate = {
        'centroids_full_gb': to_gb(m_centroids_full),
        'centroids_gb': to_gb(m_centroids_persist),
        'centroid_copy_peak_gb': to_gb(m_centroid_copy_peak),
        'cached_gspace_gb': to_gb(chunk_result['cache_bytes']),
        'use_gspace_cache': use_gspace_cache,
        'fft_workspace_gb': to_gb(m_fft_workspace),
        'peak_fft_gb': to_gb(peak_fft_stage),
        'stage_cct_gb': to_gb(stage_cct),
        'L_q_gb': to_gb(m_L_q),
        'psi_xchunk_gb': to_gb(stage_info['psi_xchunk']),
        'pair_density_xchunk_gb': to_gb(stage_info['pair_xchunk']),
        'Z_q_gb': to_gb(stage_info['Z_q']),
        'Z_col_gb': to_gb(stage_info['Z_col']),
        'zeta_gb': to_gb(stage_info['Z_col']),
        'L_rep_per_q_gb': to_gb(l_rep_bytes),
        'stage_pair_gb': to_gb(stage_info['stage_pair']),
        'stage_zct_gb': to_gb(stage_info['stage_zct']),
        'stage_solve_gb': to_gb(stage_info['stage_solve']),
        'peak_chunk_gb': to_gb(peak_chunk_bytes),
        'peak_estimate_gb': to_gb(overall_peak),
        'budget_gb': memory_budget_gb,
        'effective_budget_gb': to_gb(m_budget),
        'utilization_pct': 100.0 * overall_peak / m_budget,
        'bottleneck': bottleneck,
        'p_x': p_x,
        'p_y': p_y,
        'n_devices': p,
        'x_chunk_r': x_chunk_r,
        'centroids_bytes': m_centroids_persist,
        'effective_budget_bytes': m_budget,
        'available_vcoul_gb': to_gb(max(0.0, m_budget - m_centroids_persist)),
    }

    return {
        'band_chunk': band_chunk,
        'x_chunk': x_chunk_slices,
        'x_chunk_r': x_chunk_r,
        'q_chunk': q_chunk,
        'use_gspace_cache': use_gspace_cache,
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
    """Print memory breakdown based on two bottleneck stages."""
    nx, ny, nz = fft_grid
    mem = chunks['memory_estimate']
    
    print("\n" + "="*70)
    print("  MEMORY-OPTIMIZED CHUNK SIZES")
    print("="*70)
    
    print(f"\n  Memory budget: {mem['budget_gb']:.2f} GB/device (source: {memory_source})")
    print(f"  Device mesh: {mem['p_x']} × {mem['p_y']} = {mem['n_devices']} devices")
    
    print(f"\n  {'Parameter':<25} {'Value':>10} {'Total':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Band chunk':<25} {chunks['band_chunk']:>10d} / {n_b:<5d} bands")
    print(f"  {'X-chunk (x-slices)':<25} {chunks['x_chunk']:>10d} / {nx:<5d} slices")
    print(f"  {'X-chunk (r-points)':<25} {chunks['x_chunk_r']:>10d} / {n_r:<5d} points")
    print(f"  {'Q-chunk':<25} {chunks['q_chunk']:>10d} / {n_q:<5d} q-points")
    
    print(f"\n  {'SIMULTANEOUS ALLOCATIONS':<40} {'Size (GB)':>12}")
    print(f"  {'-'*54}")
    
    # Persistent arrays
    print(f"  {'[Persistent]':<40}")
    print(f"    {'Centroid union (load stage)':<38} {mem['centroids_full_gb']:>10.3f}")
    print(f"    {'Centroids (4 arrays: l/r × X/Y)':<38} {mem['centroids_gb']:>10.3f}")
    if mem.get('use_gspace_cache', False):
        print(f"    {'G-space cache':<38} {mem['cached_gspace_gb']:>10.3f}")
    else:
        print(f"    {'G-space cache':<38} {'(disabled)'}")
    print(f"    {'L_q (Cholesky factor)':<38} {mem['L_q_gb']:>10.3f}")
    
    # CCT stage
    print(f"\n  {'[Stage: C_q build]':<40}")
    print(f"    {'Pair densities + C_q':<38} {mem['stage_cct_gb']:>10.3f}")
    
    # Pair density stage
    print(f"\n  {'[Stage: Pair density]':<40}")
    print(f"    {'psi_nk(rchunk) all bands':<38} {mem['psi_xchunk_gb']:>10.3f}")
    print(f"    {'P_l + P_r (spin-traced)':<38} {mem['pair_density_xchunk_gb']:>10.3f}")
    print(f"    {'─ STAGE TOTAL':<38} {mem['stage_pair_gb']:>10.3f}")
    
    # ZCT stage
    print(f"\n  {'[Stage: ZCT / FFT pipeline]':<40}")
    print(f"    {'Z_q(rmu, rchunk)':<38} {mem['Z_q_gb']:>10.3f}")
    print(f"    {'─ STAGE TOTAL':<38} {mem['stage_zct_gb']:>10.3f}")
    
    # Solve
    print(f"\n  {'[Stage: Solve (psi_xchunk deleted)]':<40}")
    print(f"    {'Z_col (resharded)':<38} {mem['Z_col_gb']:>10.3f}")
    print(f"    {'zeta_q (output)':<38} {mem['zeta_gb']:>10.3f}")
    print(f"    {'L_rep (replicated per q)':<38} {mem['L_rep_per_q_gb']:>10.3f}")
    print(f"    {'─ STAGE TOTAL':<38} {mem['stage_solve_gb']:>10.3f}")
    
    print(f"\n  {'-'*54}")
    bottleneck = mem.get('bottleneck', 'pair')
    print(f"  {'PEAK ('+bottleneck+')':<38} {mem['peak_estimate_gb']:>10.3f} GB")
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

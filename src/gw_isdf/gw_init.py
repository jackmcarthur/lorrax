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
    target_utilization: float = 0.97,
    p_x: int | None = None,
    p_y: int | None = None,
    verbose: bool = True,
    n_b_left: int | None = None,
    n_b_right: int | None = None,
    pair_density_channels: int = 1,
    r_chunk_override: int | None = None,
    zct_stage_cap_gb: float | None = None,
) -> dict:
    """Derive chunk sizes that saturate (but do not exceed) the memory budget."""
    if memory_budget_gb <= 0:
        raise ValueError("memory_per_device_gb must be > 0 for automatic chunk sizing.")
    if n_devices <= 0:
        raise ValueError("n_devices must be at least 1.")
    if target_utilization <= 0 or target_utilization > 1.0:
        raise ValueError("target_utilization must be in (0, 1].")
    if pair_density_channels < 1:
        raise ValueError("pair_density_channels must be >= 1.")

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
    m_zct_cap = None if zct_stage_cap_gb is None else float(zct_stage_cap_gb) * 1e9
    if m_zct_cap is not None and m_zct_cap <= 0:
        m_zct_cap = None
    n_k_f = float(n_k)
    n_s_f = float(n_s)
    n_rmu_f = float(n_rmu)
    n_q_f = float(n_q)
    n_r_f = float(n_r)
    pair_channels_f = float(pair_density_channels)

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
    m_pair_mumu = pair_channels_f * bytes_per_complex * n_k_f * (n_rmu_f / p_x) * (n_rmu_f / p_y)
    m_C_q = bytes_per_complex * n_q_f * (n_rmu_f / p_x) * (n_rmu_f / p_y)
    stage_cct = m_centroids_persist + 2 * m_pair_mumu + m_C_q
    if stage_cct > m_budget:
        raise ValueError(
            f"C_q build requires {to_gb(stage_cct):.2f} GB/device which exceeds the budget. Reduce n_rmu or n_q."
        )
    m_L_q = m_C_q
    # One fully replicated L matrix (single q) used by triangular solve.
    l_rep_bytes = bytes_per_complex * (n_rmu_f ** 2)

    # Optional G-space cache (sum of all cached band chunks)
    m_cached_gspace_full = bytes_per_complex * n_k_f * (n_b_full / p) * n_s_f * n_r_f

    def compute_chunk_metrics(chunk_r: int, base_const: float) -> tuple[float, str, dict]:
        """Compute peak memory for each stage given an r-chunk size."""
        cr = float(chunk_r)
        per_y = cr / p_y

        # Sharded local sizes (for stages that work correctly with sharding)
        m_psi_chunk = bytes_per_complex * n_k_f * n_b_full * n_s_f * per_y
        m_pair_local = pair_channels_f * bytes_per_complex * n_k_f * (n_rmu_f / p_x) * per_y
        m_z_local = bytes_per_complex * n_q_f * (n_rmu_f / p_x) * per_y
        m_zcol = bytes_per_complex * n_q_f * n_rmu_f * (cr / p)

        # GLOBAL (unsharded) sizes in ZCT/FFT stage where XLA materializes
        # full pair-density operands and FFT temporaries.
        # P_l and P_r have shape (nk, spin_channels, n_rmu, chunk_r);
        # Z_q has (n_q, n_rmu, chunk_r).
        m_pair_global = pair_channels_f * bytes_per_complex * n_k_f * n_rmu_f * cr
        m_z_global = bytes_per_complex * n_q_f * n_rmu_f * cr
        # XProf (jit__compute_ZCT_LR) shows the dominant live set as:
        #   - pair inputs         : 2 * P
        #   - FFT temporaries     : 2 * P
        #   - output (Z_q)        : 1 * Z
        # giving a stage coefficient of 4*P + Z.
        m_zct_pair_inputs = 2.0 * m_pair_global
        m_zct_fft_temps = 2.0 * m_pair_global
        m_zct_output = m_z_global
        m_zct_peak = m_zct_pair_inputs + m_zct_fft_temps + m_zct_output

        # Solve stage components.
        m_solve_z_io = 2.0 * m_zcol
        m_solve_tri_temps = 2.0 * m_zcol
        # XProf shows an additional local L_q-sized layout/transposition temp.
        m_solve_l_temp = m_L_q
        # q_chunk=1 still replicates one full L matrix.
        m_solve_l_rep = l_rep_bytes

        stage_pair = base_const + m_psi_chunk + 2 * m_pair_local
        stage_zct = base_const + m_zct_peak
        stage_solve = base_const + m_solve_z_io + m_solve_tri_temps + m_solve_l_temp + m_solve_l_rep
        peak = max(stage_pair, stage_zct, stage_solve)
        if peak == stage_pair:
            name = 'pair'
        elif peak == stage_zct:
            name = 'zct'
        else:
            name = 'solve'
        info = {
            'psi_chunk': m_psi_chunk,
            'pair_chunk': 2 * m_pair_local,
            'zct_pair_inputs': m_zct_pair_inputs,
            'zct_fft_temps': m_zct_fft_temps,
            'zct_output': m_zct_output,
            'pair_fft_peak': m_zct_peak,
            'pair_global': m_pair_global,
            'z_global': m_z_global,
            'Z_q': m_z_local,
            'Z_col': m_zcol,
            'solve_z_io': m_solve_z_io,
            'solve_tri_temps': m_solve_tri_temps,
            'solve_l_temp': m_solve_l_temp,
            'solve_l_rep': m_solve_l_rep,
            'stage_pair': stage_pair,
            'stage_zct': stage_zct,
            'stage_solve': stage_solve,
            'fft_overhead_factor': (4.0 * pair_channels_f * n_k_f + n_q_f) / max(1.0, n_k_f),
        }
        return peak, name, info

    def choose_r_chunk(use_cache: bool, override: int | None = None) -> dict | None:
        m_cache = m_cached_gspace_full if use_cache else 0.0
        base_const = m_centroids_persist + m_L_q + m_cache
        headroom = m_budget - base_const
        if headroom <= bytes_per_complex * (n_rmu_f ** 2):
            return None
        headroom_zct = None
        if m_zct_cap is not None:
            headroom_zct = m_zct_cap - base_const

        limit_info = {}

        if override is not None and override > 0:
            # Explicit override: use it directly
            r_chunk_r = min(int(override), int(n_r))
        else:
            # Auto-compute from memory limits (same logic as old choose_x_chunk,
            # but round only to p_y divisibility, not ny*nz boundaries)
            limits = []
            # Pair density stage limit (uses sharded local sizes)
            denom_pair = n_k_f * n_b_full * n_s_f + 2 * pair_channels_f * n_k_f * (n_rmu_f / p_x)
            if denom_pair > 0:
                limit_pair = headroom * p_y / (bytes_per_complex * denom_pair)
                limits.append(limit_pair)
                limit_info['limit_pair'] = limit_pair

            # ZCT/FFT stage limit from XProf-observed 4*P + Z live set.
            denom_fft_global = (4 * pair_channels_f * n_k_f + n_q_f) * n_rmu_f
            if denom_fft_global > 0:
                limit_fft = headroom / (bytes_per_complex * denom_fft_global)
                limits.append(limit_fft)
                limit_info['limit_fft_global'] = limit_fft
                if headroom_zct is not None and headroom_zct > 0:
                    limit_fft_soft = headroom_zct / (bytes_per_complex * denom_fft_global)
                    limits.append(limit_fft_soft)
                    limit_info['limit_fft_soft'] = limit_fft_soft

            # Solve stage limit
            solve_numer = headroom - m_L_q - l_rep_bytes
            denom_solve = 4 * n_q_f * n_rmu_f / p
            if denom_solve > 0 and solve_numer > 0:
                limit_solve = solve_numer * p / (bytes_per_complex * denom_solve)
                limits.append(limit_solve)
                limit_info['limit_solve'] = limit_solve

            if not limits:
                limits.append(float(n_r))
                limit_info['limit_default'] = float(n_r)

            r_chunk_r = min(int(n_r), max(1, int(min(limits))))

        # Ensure divisibility along mesh Y for sharding on r
        if p_y > 1 and (r_chunk_r % p_y) != 0:
            r_chunk_r = r_chunk_r - (r_chunk_r % p_y)
        if r_chunk_r <= 0:
            return None

        # Search downward for a chunk that fits the budget
        while r_chunk_r > 0:
            peak_bytes, stage_name, info = compute_chunk_metrics(r_chunk_r, base_const)
            zct_ok = (m_zct_cap is None) or (info['stage_zct'] <= m_zct_cap)
            if peak_bytes <= m_budget and zct_ok:
                x_slices_est = max(1, int(math.ceil(r_chunk_r / float(ny * nz))))
                return {
                    'x_chunk': x_slices_est,
                    'chunk_r': r_chunk_r,
                    'peak_bytes': peak_bytes,
                    'stage_name': stage_name,
                    'stage_info': info,
                    'cache_bytes': m_cache,
                    'base_const': base_const,
                    'limit_info': limit_info,
                }
            # Step down by p_y to maintain divisibility
            r_chunk_r -= max(1, p_y)
            if p_y > 1 and r_chunk_r > 0 and (r_chunk_r % p_y) != 0:
                r_chunk_r = r_chunk_r - (r_chunk_r % p_y)
        return None

    chunk_result = choose_r_chunk(use_cache=True, override=r_chunk_override) or \
                   choose_r_chunk(use_cache=False, override=r_chunk_override)
    if chunk_result is None:
        raise ValueError(
            "Unable to find an r-chunk that fits the memory budget (with or without G-space caching)."
        )

    use_gspace_cache = chunk_result['cache_bytes'] > 0
    x_chunk_slices = chunk_result['x_chunk']
    chunk_r = chunk_result['chunk_r']
    stage_info = chunk_result['stage_info']
    peak_chunk_bytes = chunk_result['peak_bytes']
    base_const = chunk_result['base_const']

    m_Z_col = stage_info['Z_col']
    # Base solve footprint for q_chunk=1.
    base_solve = base_const + 4 * m_Z_col + m_L_q + l_rep_bytes
    available_for_q = max(0.0, m_budget - base_solve)
    if available_for_q <= 0:
        q_chunk = 1
    else:
        # Additional q-points beyond the first add one replicated L each.
        q_chunk = max(1, min(n_q, 1 + int(available_for_q // l_rep_bytes)))

    stage_solve_selected = base_const + 4 * m_Z_col + m_L_q + q_chunk * l_rep_bytes
    peak_chunk_bytes = max(peak_chunk_bytes, stage_solve_selected)

    chunk_stage_peaks = {
        'pair': stage_info['stage_pair'],
        'zct': stage_info['stage_zct'],
        'solve': stage_solve_selected,
    }
    chunk_stage_name = max(chunk_stage_peaks, key=chunk_stage_peaks.get)

    overall_peak = max(peak_chunk_bytes, peak_fft_stage, m_centroid_copy_peak, stage_cct)
    bottleneck = chunk_stage_name if peak_chunk_bytes >= max(peak_fft_stage, m_centroid_copy_peak, stage_cct) else (
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
        'psi_chunk_gb': to_gb(stage_info['psi_chunk']),
        'pair_density_chunk_gb': to_gb(stage_info['pair_chunk']),
        'pair_fft_peak_gb': to_gb(stage_info['pair_fft_peak']),
        'pair_global_gb': to_gb(stage_info['pair_global']),
        'Z_q_global_gb': to_gb(stage_info.get('z_global', 0.0)),
        'fft_overhead_factor': stage_info['fft_overhead_factor'],
        'Z_q_gb': to_gb(stage_info['Z_q']),
        'Z_col_gb': to_gb(stage_info['Z_col']),
        'zeta_gb': to_gb(stage_info['Z_col']),
        'zct_pair_inputs_gb': to_gb(stage_info.get('zct_pair_inputs', 0.0)),
        'zct_fft_temps_gb': to_gb(stage_info.get('zct_fft_temps', 0.0)),
        'zct_output_gb': to_gb(stage_info.get('zct_output', 0.0)),
        'solve_z_io_gb': to_gb(stage_info.get('solve_z_io', 0.0)),
        'solve_tri_temps_gb': to_gb(stage_info.get('solve_tri_temps', 0.0)),
        'solve_l_temp_gb': to_gb(stage_info.get('solve_l_temp', 0.0)),
        'solve_l_rep_gb': to_gb(q_chunk * l_rep_bytes),
        'L_rep_per_q_gb': to_gb(l_rep_bytes),
        'stage_pair_gb': to_gb(stage_info['stage_pair']),
        'stage_zct_gb': to_gb(stage_info['stage_zct']),
        'stage_solve_gb': to_gb(stage_solve_selected),
        'stage_solve_min_gb': to_gb(stage_info['stage_solve']),
        'peak_chunk_gb': to_gb(peak_chunk_bytes),
        'peak_estimate_gb': to_gb(overall_peak),
        'budget_gb': memory_budget_gb,
        'effective_budget_gb': to_gb(m_budget),
        'utilization_pct': 100.0 * overall_peak / m_budget,
        'bottleneck': bottleneck,
        'p_x': p_x,
        'p_y': p_y,
        'n_devices': p,
        'pair_density_channels': pair_density_channels,
        'chunk_r': chunk_r,
        'limit_info': chunk_result.get('limit_info', {}),
        'centroids_bytes': m_centroids_persist,
        'effective_budget_bytes': m_budget,
        'available_vcoul_gb': to_gb(max(0.0, m_budget - m_centroids_persist)),
        'zct_stage_cap_gb': to_gb(m_zct_cap) if m_zct_cap is not None else None,
    }

    return {
        'band_chunk': band_chunk,
        'x_chunk': x_chunk_slices,
        'chunk_r': chunk_r,
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
    """Print memory breakdown and stage bottleneck drivers."""
    nx, ny, nz = fft_grid
    mem = chunks['memory_estimate']
    
    print("\n" + "="*70)
    print("  MEMORY-OPTIMIZED CHUNK SIZES")
    print("="*70)
    
    print(f"\n  Memory budget: {mem['budget_gb']:.2f} GB/device (source: {memory_source})")
    print(f"  Device mesh: {mem['p_x']} × {mem['p_y']} = {mem['n_devices']} devices")
    pair_channels = int(mem.get('pair_density_channels', 1))
    
    print(f"\n  {'Parameter':<25} {'Value':>10} {'Total':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Band chunk':<25} {chunks['band_chunk']:>10d} / {n_b:<5d} bands")
    print(f"  {'R-chunk (r-points)':<25} {chunks['chunk_r']:>10d} / {n_r:<5d} points")
    print(f"  {'R-chunk (x-slices est)':<25} {chunks['x_chunk']:>10d} / {nx:<5d} slices")
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
    print(f"    {'psi_nk(rchunk) all bands':<38} {mem['psi_chunk_gb']:>10.3f}")
    pair_label = 'P_l + P_r (spin-traced)' if pair_channels == 1 else f'P_l + P_r ({pair_channels} spin channels)'
    print(f"    {pair_label:<38} {mem['pair_density_chunk_gb']:>10.3f}")
    print(f"    {'─ STAGE TOTAL':<38} {mem['stage_pair_gb']:>10.3f}")
    
    # ZCT stage
    fft_factor = float(mem.get('fft_overhead_factor', 4.0))
    print(f"\n  {'[Stage: ZCT / FFT pipeline]':<40}")
    print(f"    {'P_l, P_r (local sharded)':<38} {mem['pair_density_chunk_gb']:>10.3f}")
    print(f"    {'Pair inputs (2×global P)':<38} {mem.get('zct_pair_inputs_gb', 0):>10.3f}")
    print(f"    {'FFT temporaries (2×global P)':<38} {mem.get('zct_fft_temps_gb', 0):>10.3f}")
    print(f"    {'Z_q output (global)':<38} {mem.get('zct_output_gb', 0):>10.3f}")
    print(f"    {'Effective coeff (' + str(round(fft_factor, 2)) + ' × P_global)':<38} {mem.get('pair_fft_peak_gb', 0):>10.3f}")
    print(f"    {'─ STAGE TOTAL':<38} {mem['stage_zct_gb']:>10.3f}")
    
    # Solve
    print(f"\n  {'[Stage: Solve (L^-1 Z)]':<40}")
    print(f"    {'Z input + output':<38} {mem.get('solve_z_io_gb', 0):>10.3f}")
    print(f"    {'Triangular-solve temps':<38} {mem.get('solve_tri_temps_gb', 0):>10.3f}")
    print(f"    {'L_q local temp':<38} {mem.get('solve_l_temp_gb', 0):>10.3f}")
    print(f"    {'L replication (' + str(chunks['q_chunk']) + '×)':<38} {mem.get('solve_l_rep_gb', 0):>10.3f}")
    print(f"    {'L_rep (per q)':<38} {mem['L_rep_per_q_gb']:>10.3f}")
    print(f"    {'─ STAGE TOTAL':<38} {mem['stage_solve_gb']:>10.3f}")
    
    print(f"\n  {'-'*54}")
    bottleneck = mem.get('bottleneck', 'pair')
    print(f"  {'PEAK ('+bottleneck+')':<38} {mem['peak_estimate_gb']:>10.3f} GB")
    print(f"  {'BUDGET':<38} {mem['budget_gb']:>10.3f} GB")
    print(f"  {'UTILIZATION':<38} {mem['utilization_pct']:>10.1f} %")
    limit_info = mem.get('limit_info', {})
    if limit_info:
        print(f"\n  {'CHUNK LIMIT ESTIMATES (r-points)':<40} {'':>12}")
        for key in ("limit_pair", "limit_fft_global", "limit_fft_soft", "limit_solve", "limit_default"):
            if key in limit_info:
                print(f"  {key:<38} {limit_info[key]:>10.1f}")
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
		# r_chunk_size:  R-axis chunk size (flattened xyz index, contiguous in r-space).
		#                0 = auto (default): auto-compute from memory budget.
		#                >0 = explicit r-chunk size in r-points.
		# memory_per_device_gb: Memory budget per device in GB for auto chunk sizing.
		#                0 = auto-detect (80% of GPU via nvidia-smi, or CPU/n_devices)
		#                >0 = explicit budget in GB
		# isdf_pair_mode: ISDF pair-density pathway used in CCT/ZCT:
		#                'spin_traced' (default) or
		#                'spin_matrix_frobenius' (keep spin channels, sum_ab after contraction)
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
			"r_chunk_size": geti("r_chunk_size", fallback=0),    # r-axis chunk (0=auto, >0=explicit)
			"band_chunk_size": geti("band_chunk_size", fallback=16),  # bands per FFT during r-chunk loop
			"memory_per_device_gb": getf("memory_per_device_gb", fallback=0.0),  # 0=auto-detect
			"isdf_pair_mode": get("isdf_pair_mode", fallback="spin_traced").strip().lower(),
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
			"r_chunk_size": 0,
			"band_chunk_size": 16,
			"memory_per_device_gb": 0.0,
			"isdf_pair_mode": "spin_traced",
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

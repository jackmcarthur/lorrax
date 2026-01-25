"""Input file parsing and preprocessing for COHSEX calculations.

This module contains functions for:
- Reading and parsing the cohsex input file
- Converting input parameters to effective values
- Computing band ranges from input parameters
- Memory-aware chunk size optimization
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
    n_devices: int,
    memory_budget_gb: float,
    target_utilization: float = 0.85,
) -> dict:
    """
    Compute optimal chunk sizes based on memory budget.
    
    Uses the memory model from MEMORY_MODEL.md to derive chunk sizes
    that will fit within the per-device memory budget.
    
    Args:
        n_k: Number of k-points
        n_b: Number of bands
        n_s: Number of spinor components (2 or 4)
        n_rmu: Number of ISDF centroids
        n_r: Total real-space grid points (nx*ny*nz)
        n_q: Number of q-points
        n_devices: Total number of devices
        memory_budget_gb: Memory budget per device in GB
        target_utilization: Fraction of budget to use (default 0.85)
    
    Returns:
        Dictionary with:
        - band_chunk: Optimal band chunk size
        - z_chunk: Optimal z-axis chunk size
        - q_chunk: Optimal q-point chunk size for solve
        - memory_estimate: Dict of estimated memory usage by stage
    """
    m_budget = memory_budget_gb * 1e9 * target_utilization
    p = n_devices
    sqrt_p = math.sqrt(p)
    
    # Centroid memory (persistent): 2 copies, each (nk, nb, ns, n_rmu/sqrt(P))
    m_mu = 2 * 16 * n_k * n_b * n_s * n_rmu / sqrt_p
    m_available = m_budget - m_mu
    
    if m_available <= 0:
        raise ValueError(
            f"Centroid wavefunctions alone require {m_mu/1e9:.2f} GB per device, "
            f"but budget is {memory_budget_gb:.2f} GB. Increase memory budget or reduce n_b."
        )
    
    # Band chunk size: FFT workspace = 2 * (nk * B_b/P * ns * n_r) for input+output
    # Solve for B_b <= m_available * P / (2 * 16 * n_k * n_s * n_r)
    b_b_max = m_available * p / (2 * 16 * n_k * n_s * n_r)
    band_chunk = max(16, min(n_b, int(b_b_max)))
    
    # Z-chunk size: pair density P_k = (nk * ns² * n_rmu/Px * B_z/Py)
    # Approximate Px ≈ Py ≈ sqrt(P)
    # B_z <= m_available * P / (16 * n_k * n_s² * n_rmu)
    b_z_max = m_available * p / (16 * n_k * n_s * n_s * n_rmu)
    z_chunk = max(n_rmu, min(n_r, int(b_z_max)))  # At least as big as n_rmu
    
    # Q-chunk size: L replication = B_q * 16 * n_rmu² per device
    # Also need Z buffer: 16 * n_q * n_rmu * B_z / P
    m_z_buffer = 16 * n_q * n_rmu * z_chunk / p
    b_q_max = (m_available - m_z_buffer) / (16 * n_rmu * n_rmu)
    q_chunk = max(1, min(n_q, int(b_q_max)))
    
    # Memory estimates by stage
    memory_estimate = {
        'centroids_gb': m_mu / 1e9,
        'fft_workspace_gb': 2 * 16 * n_k * (band_chunk / p) * n_s * n_r / 1e9,
        'pair_density_gb': 16 * n_k * n_s * n_s * (n_rmu / sqrt_p) * (z_chunk / sqrt_p) / 1e9,
        'l_replicated_gb': q_chunk * 16 * n_rmu * n_rmu / 1e9,
        'available_gb': m_available / 1e9,
        'budget_gb': memory_budget_gb,
    }
    
    return {
        'band_chunk': band_chunk,
        'z_chunk': z_chunk,
        'q_chunk': q_chunk,
        'memory_estimate': memory_estimate,
    }


def print_chunk_info(chunks: dict, n_b: int, n_r: int, n_q: int) -> None:
    """Print chunk size information in a formatted table."""
    print("\n=== Memory-Optimized Chunk Sizes ===")
    print(f"  Band chunk:   {chunks['band_chunk']:6d} / {n_b} bands")
    print(f"  Z chunk:      {chunks['z_chunk']:6d} / {n_r} r-points")
    print(f"  Q chunk:      {chunks['q_chunk']:6d} / {n_q} q-points")
    print("\n=== Estimated Memory Usage (per device) ===")
    mem = chunks['memory_estimate']
    print(f"  Centroids:      {mem['centroids_gb']:6.2f} GB (persistent)")
    print(f"  FFT workspace:  {mem['fft_workspace_gb']:6.2f} GB")
    print(f"  Pair density:   {mem['pair_density_gb']:6.2f} GB")
    print(f"  L replicated:   {mem['l_replicated_gb']:6.2f} GB")
    print(f"  --------------------------------")
    print(f"  Available:      {mem['available_gb']:6.2f} GB")
    print(f"  Budget:         {mem['budget_gb']:6.2f} GB")


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
    """Compute effective z-axis chunk size for ZCT accumulation.
    
    Priority (highest to lowest):
    1. max_wfn_chunk_mb > 0: compute z from memory budget for P_k(rmu, rchunk)
    2. z_chunk_size > 0: use explicit value
    3. z_chunk_size == 0: auto-compute to target_ratio * n_rmu
    
    The P_k,ab(rmu, rchunk) array has shape (nk, ns, ns, n_rmu, n_zchunk) with
    complex128 dtype (16 bytes). Given a memory budget:
        max_bytes = max_wfn_chunk_mb * 1e6
        n_zchunk_max = max_bytes / (nk * ns * ns * n_rmu * 16)
        z_chunk_size = n_zchunk_max / (nx * ny)
    
    Additionally ensures n_zchunk = nx * ny * z is divisible by mesh_y_size.
    
    Args:
        z_chunk_size: Input flag value:
            0 = auto: choose z such that nx*ny*z ≈ target_ratio * n_rmu
            1-nz = explicit z-slice count per chunk
        fft_grid: (nx, ny, nz) FFT grid dimensions
        n_rmu: Number of ISDF centroids
        target_ratio: Target ratio of chunk size to n_rmu (default 16.0)
        mesh_y_size: Number of devices on Y-axis mesh (for divisibility)
        max_wfn_chunk_mb: Max memory for P_k chunk in MB (0=ignore, use z_chunk_size)
        nk_tot: Total number of k-points (for memory calculation)
        nspinor: Number of spinor components (for memory calculation)
    
    Returns:
        Effective z_chunk_size (number of z-slices per chunk)
    """
    nx, ny, nz = fft_grid
    
    # Priority 1: Memory budget overrides everything
    if max_wfn_chunk_mb > 0:
        # P_k,ab(rmu, rchunk) shape: (nk, ns, ns, n_rmu, n_zchunk)
        # Size in bytes: nk * ns² * n_rmu * n_zchunk * 16
        max_bytes = max_wfn_chunk_mb * 1e6
        bytes_per_zpoint = nk_tot * nspinor * nspinor * n_rmu * 16
        n_zchunk_max = max_bytes / bytes_per_zpoint
        z_budget = max(1, int(n_zchunk_max / (nx * ny)))
        
        # Ensure n_zchunk = nx*ny*z is divisible by mesh_y_size
        # Find the largest z <= z_budget where n_zchunk is divisible
        z_opt = None
        for z_try in range(min(z_budget, nz), 0, -1):
            if (nx * ny * z_try) % mesh_y_size == 0:
                z_opt = z_try
                break
        
        if z_opt is None:
            # No valid z found - this means we'd need more chunks than nz
            raise ValueError(
                f"max_wfn_chunk_mb={max_wfn_chunk_mb} is too small. "
                f"Smallest valid chunk (z=1) requires {nx*ny*16*nk_tot*nspinor**2*n_rmu/1e6:.1f} MB. "
                f"Increase max_wfn_chunk_mb or remove it to use auto-sizing."
            )
        
        return z_opt
    
    # Priority 2: Explicit z_chunk_size
    if z_chunk_size > 0:
        if 1 <= z_chunk_size <= nz:
            return z_chunk_size
        else:
            raise ValueError(f"z_chunk_size must be 1-{nz}, got {z_chunk_size}")
    
    # Priority 3: Auto based on target_ratio
    target_chunk = target_ratio * n_rmu
    z_opt = max(1, round(target_chunk / (nx * ny)))
    
    # Ensure n_zchunk is divisible by mesh_y_size
    if mesh_y_size > 1:
        n_zchunk = nx * ny * z_opt
        # Round up to next multiple of mesh_y_size if needed
        remainder = n_zchunk % mesh_y_size
        if remainder != 0:
            # Try increasing z to make it divisible
            extra_needed = mesh_y_size - remainder
            extra_z = (extra_needed + nx * ny - 1) // (nx * ny)
            z_opt = z_opt + extra_z
    
    # Clamp to valid range [1, nz]
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
		# z_chunk_size:  Z-axis chunk size for ZCT accumulation.
		#                0 = auto (default): choose z such that nx*ny*z ≈ 2*n_rmu
		#                1-nz = explicit z-slice count per chunk
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
			"z_chunk_size": geti("z_chunk_size", fallback=0),    # z-axis chunk (0=auto, 1-nz=explicit)
			"max_wfn_chunk_mb": getf("max_wfn_chunk_mb", fallback=0.0),  # max P_k chunk size in MB (0=use z_chunk_size)
			"band_chunk_size": geti("band_chunk_size", fallback=16),  # bands per FFT during z-chunk loop
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


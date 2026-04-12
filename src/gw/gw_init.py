"""Input file parsing and preprocessing for COHSEX calculations.

This module contains functions for:
- Reading and parsing the cohsex input file
- Converting input parameters to effective values
- Computing band ranges from input parameters
- Memory-aware chunk size optimization with full communication buffer accounting
- Runtime configuration resolution (memory budget, chunking, ZCT caps)
- ISDF fitting / restart orchestration
"""
import os
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp

import common.timing as timing



def compute_optimal_chunks(
    n_k: int, n_b: int, n_s: int, n_rmu: int, n_r: int, n_q: int,
    fft_grid: tuple[int, int, int], n_devices: int, memory_budget_gb: float,
    target_utilization: float = 0.97,
    p_x: int | None = None, p_y: int | None = None,
    verbose: bool = True,
    n_b_left: int | None = None, n_b_right: int | None = None,
    pair_density_channels: int = 1,
    r_chunk_override: int | None = None,
    zct_stage_cap_gb: float | None = None,
) -> dict:
    """Derive ISDF chunk sizes that saturate (but do not exceed) the memory budget.

    Models 6 pipeline stages (FFT, pair density, ZCT, reshard, solve, gather),
    each calibrated against XLA HLO profiling on CrI3 (16-GPU, 75×75×200 grid).

    Memory multipliers:
      - FFT:     3× (input + output + cuFFT scratch, donated)
      - ZCT:     4× pair data (P_r alive + 3× left IFFT JIT)
      - Reshard: 4× m_zcol (input + output + 2× NCCL temp)
      - Solve:   Z_col(donated) + per-q (Z_slice + L_rep + L_allgather + 2×trsm + Z_T)
      - Gather:  zeta(sharded) + per-q (input + output + NCCL)
    """
    if memory_budget_gb <= 0:
        raise ValueError("memory_per_device_gb must be > 0.")

    B = 16.0  # bytes per complex128
    nx, ny, nz = fft_grid
    p = max(1, int(n_devices))
    nb = int(n_b)
    nb_l = int(n_b_left) if n_b_left is not None else nb
    nb_r = int(n_b_right) if n_b_right is not None else nb
    nk, ns, mu, nq, nr = float(n_k), float(n_s), float(n_rmu), float(n_q), float(n_r)
    pc = float(pair_density_channels)

    if p_x is None or p_y is None:
        sq = int(math.sqrt(p))
        while sq > 1 and p % sq != 0:
            sq -= 1
        p_x, p_y = sq, p // sq

    def _mem(*dims, shard=1):
        """Bytes for a complex128 array with given dimensions, divided by shard."""
        result = B
        for d in dims:
            result *= d
        return result / shard

    def _limit(headroom, *dims, shard=1):
        """Max chunk_r such that _mem(*dims, chunk_r, shard=shard) ≤ headroom."""
        denom = B
        for d in dims:
            denom *= d
        return headroom * shard / denom if denom > 0 else nr

    m_budget = memory_budget_gb * 1e9 * target_utilization
    m_zct_cap = float(zct_stage_cap_gb) * 1e9 if zct_stage_cap_gb and zct_stage_cap_gb > 0 else None

    # ---- Persistent arrays ----
    m_centroids_full = _mem(nk, ns, mu, nb, shard=p_y) + _mem(nk, ns, mu, nb, shard=p_x)
    m_centroids = _mem(nk, ns, mu, nb_l + nb_r, shard=p_y) + _mem(nk, ns, mu, nb_l + nb_r, shard=p_x)
    if m_centroids_full + m_centroids > m_budget:
        raise ValueError(
            f"Centroid storage requires {(m_centroids_full + m_centroids)/1e9:.2f} GB/device "
            f"but only {memory_budget_gb:.2f} GB allocated.")

    # ---- Band-chunked FFT (centroid extraction) ----
    FFT_COPIES = 3
    m_phase = _mem(nk, nr)
    headroom_fft = m_budget - m_centroids_full
    if headroom_fft <= m_phase:
        raise ValueError("Insufficient memory for even a single-band FFT chunk.")
    one_band = FFT_COPIES * _mem(nk, ns, nr)
    bpd_max = max(1, int((headroom_fft - m_phase) / one_band))
    band_chunk = min(nb, bpd_max * p)
    bpd = max(1, -(-band_chunk // p))  # ceil div (matches read_Gvecs_to_devices)
    m_fft_workspace = FFT_COPIES * _mem(nk, bpd, ns, nr) + m_phase
    peak_fft_stage = m_centroids_full + m_fft_workspace

    # ---- C_q build (pair density + Cholesky) ----
    m_pair_mumu = _mem(pc, nk, mu, mu, shard=p_x * p_y)
    m_C_q = _mem(nq, mu, mu, shard=p_x * p_y)
    stage_cct = m_centroids + 2 * m_pair_mumu + m_C_q
    if stage_cct > m_budget:
        raise ValueError(f"C_q build requires {stage_cct/1e9:.2f} GB/device — exceeds budget.")
    m_L_q = m_C_q

    # ---- G-space cache (optional) ----
    m_gspace_cache = _mem(nk, nb, ns, nr, shard=p)

    # ---- Per-k FFT cost (for k_batch sizing) ----
    per_k_fft = FFT_COPIES * _mem(bpd, ns, nr) + _mem(nr)

    # ---- Stage peak calculator ----
    def _stage_peaks(cr, base):
        """Compute per-stage peak memory for a given r-chunk size and base cost."""
        # FFT: psi_rchunk output + 1 k-batch FFT transient
        m_psi = _mem(nk, nb, ns, cr, shard=p_y)
        s_fft = base + m_psi + per_k_fft

        # Pair density: psi_chunk + P_l + P_r
        s_pair = base + m_psi + 2 * _mem(pc, nk, mu, cr, shard=p_x * p_y)

        # ZCT: P_r alive during left IFFT (1×) + left IFFT JIT (3×) = 4× pair data
        s_zct = base + 4.0 * _mem(pc, nk, mu, cr, shard=p_x * p_y)

        # Reshard: input + output + 2× NCCL temp = 4× m_zcol
        m_zcol = _mem(nq, mu, cr, shard=p)
        s_reshard = base + 4.0 * m_zcol

        # Solve (q_chunk=1): Z_col + zeta (2× m_zcol) + per-q arrays
        m_solve_per_q = (
            _mem(mu, cr, shard=p)              # Z_slice
            + _mem(mu, mu, shard=p_x * p_x)    # L_rep (sharded)
            + 3 * _mem(mu, mu)                  # L_allgather + trsm_fwd + trsm_bwd
            + _mem(cr, mu, shard=p)             # Z_transpose
        )
        s_solve = base + 2 * m_zcol + m_solve_per_q

        # Gather (q_gather=1): zeta(sharded) + input + output + NCCL
        m_gather_per_q = _mem(mu, cr, shard=p) + 2 * _mem(mu, cr)
        s_gather = base + m_zcol + m_gather_per_q

        stages = {'fft': s_fft, 'pair': s_pair, 'zct': s_zct,
                  'reshard': s_reshard, 'solve': s_solve, 'gather': s_gather}
        peak = max(stages.values())
        name = max(stages, key=stages.get)

        # k_batch: how many k-points fit in FFT headroom (throughput, not feasibility)
        fft_head = m_budget - base - m_psi
        k_batch = min(nk, max(1.0, float(int(fft_head * 0.5 / per_k_fft)))) if per_k_fft > 0 and fft_head > per_k_fft else 1.0

        return peak, name, {
            'zcol': m_zcol, 'solve_per_q': m_solve_per_q,
            'gather_per_q': m_gather_per_q, 'k_batch': int(k_batch),
            'stages': stages,
        }

    # ---- R-chunk search ----
    def _find_r_chunk(use_cache, override=None):
        m_cache = m_gspace_cache if use_cache else 0.0
        base = m_centroids + m_L_q + m_cache
        headroom = m_budget - base
        if headroom <= _mem(mu, mu):
            return None
        headroom_zct = (m_zct_cap - base) if m_zct_cap is not None else None

        if override and override > 0:
            cr = min(int(override), int(nr))
        else:
            # Analytical limits per stage (max cr that fits)
            limits = {}
            fft_head = headroom - per_k_fft
            if fft_head > 0:
                limits['limit_fft'] = _limit(fft_head, nk, nb, ns, shard=p_y)
            limits['limit_pair'] = _limit(headroom, nk, nb, ns, shard=p_y) if nk * nb * ns > 0 else nr
            # Pair uses psi + 2×P; combined denominator:
            denom_pair = nk * nb * ns + 2 * pc * nk * (mu / p_x)
            limits['limit_pair'] = headroom * p_y / (B * denom_pair) if denom_pair > 0 else nr
            limits['limit_zct'] = _limit(headroom, 4.0 * pc * nk * (mu / p_x), shard=p_y)
            if headroom_zct is not None and headroom_zct > 0:
                limits['limit_zct_soft'] = _limit(headroom_zct, 4.0 * pc * nk * (mu / p_x), shard=p_y)
            limits['limit_reshard'] = _limit(headroom, 4.0 * nq * mu, shard=p)
            # Solve: subtract R-independent cost, then limit on R-dependent terms
            solve_const = _mem((mu / p_x) ** 2) + 3 * _mem(mu, mu)
            solve_head = headroom - solve_const
            if solve_head > 0:
                limits['limit_solve'] = _limit(solve_head, (2.0 * nq + 2.0) * mu, shard=p)
            limits['limit_gather'] = _limit(headroom, mu * (nq / p + 1.0 / p + 2.0))

            cr = min(int(nr), max(1, int(min(limits.values())))) if limits else int(nr)

        # Round down to device-divisible
        pt = p_x * p_y
        if pt > 1 and cr % pt != 0:
            cr -= cr % pt
        if cr <= 0:
            return None

        # Search downward for feasible chunk
        while cr > 0:
            peak, name, info = _stage_peaks(cr, base)
            zct_ok = m_zct_cap is None or info['stages']['zct'] <= m_zct_cap
            if peak <= m_budget and zct_ok:
                return {'chunk_r': cr, 'peak': peak, 'name': name, 'info': info,
                        'cache_bytes': m_cache, 'base': base,
                        'limit_info': limits if not (override and override > 0) else {}}
            cr -= max(1, pt)
            if pt > 1 and cr > 0 and cr % pt != 0:
                cr -= cr % pt
        return None

    # ---- Main search: try with cache, then without, reducing band_chunk if needed ----
    result = _find_r_chunk(True, r_chunk_override) or _find_r_chunk(False, r_chunk_override)
    while result is None and bpd > 1:
        bpd = max(1, bpd // 2)
        band_chunk = min(nb, bpd * p)
        bpd = max(1, -(-band_chunk // p))
        per_k_fft = FFT_COPIES * _mem(bpd, ns, nr) + _mem(nr)
        m_fft_workspace = FFT_COPIES * _mem(nk, bpd, ns, nr) + m_phase
        peak_fft_stage = m_centroids_full + m_fft_workspace
        if verbose:
            print(f"    Reducing band_chunk to {band_chunk} (bands/device={bpd})")
        result = _find_r_chunk(True, r_chunk_override) or _find_r_chunk(False, r_chunk_override)
    if result is None:
        raise ValueError("Unable to find r-chunk that fits the memory budget.")

    chunk_r = result['chunk_r']
    info = result['info']
    base = result['base']

    # ---- Derive q_chunk (solve) and q_gather (HDF5 write) ----
    m_zcol = info['zcol']
    avail_solve = max(0.0, m_budget - base - 2 * m_zcol)
    q_chunk = max(1, min(n_q, int(avail_solve / info['solve_per_q']))) if info['solve_per_q'] > 0 else 1

    avail_gather = max(0.0, m_budget - base - m_zcol)
    q_gather = max(1, min(n_q, int(avail_gather / info['gather_per_q']))) if info['gather_per_q'] > 0 else n_q

    # ---- Overall peak ----
    overall_peak = max(result['peak'],
                       base + 2 * m_zcol + q_chunk * info['solve_per_q'],
                       base + m_zcol + q_gather * info['gather_per_q'],
                       peak_fft_stage, m_centroids_full + m_centroids, stage_cct)
    bottleneck = result['name']

    k_chunk = info['k_batch']
    if k_chunk < int(n_k) and verbose:
        print(f"    K-point chunking: {k_chunk} k-pts per FFT batch (total {int(n_k)})")

    return {
        'band_chunk': band_chunk,
        'chunk_r': chunk_r,
        'q_chunk': q_chunk,
        'q_gather': q_gather,
        'k_chunk': k_chunk,
        'use_gspace_cache': result['cache_bytes'] > 0,
        'memory_estimate': {
            'peak_estimate_gb': overall_peak / 1e9,
            'budget_gb': memory_budget_gb,
            'bottleneck': bottleneck,
            'available_vcoul_gb': max(0.0, m_budget - m_centroids) / 1e9,
            'limit_info': result.get('limit_info', {}),
        },
    }


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


from .gw_config import read_lorrax_input, read_cohsex_input  # noqa: F401 — public API


def get_bandranges(nv, nc, nband, nelec):
	r"""Return ranges of bands necessary for \sigma_{X,SX,COH}"""
	nvrange = [int(nelec - nv), int(nelec)]
	ncrange = [int(nelec), int(nelec + nc)]
	nsigmarange = [int(nelec - nv), int(nelec + nc)]
	n_fullrange = [0, int(nband)]
	n_valrange = [0, int(nelec)]
	return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange



def fit_zeta(wfn, sym, meta, centroid_indices, mesh_xy, cfg, tmp_dir, print_fn=print):
	"""Fit ISDF interpolation vectors ζ and write to HDF5.

	Returns (zeta_h5_path, psi_l_yr, psi_r_yr) where:
	  - zeta_h5_path: path to the zeta HDF5 file
	  - psi_l_yr:  left centroid wfns  (nk, nb_l, ns, n_rmu), Y-sharded
	  - psi_r_yr:  right centroid wfns (nk, nb_r, ns, n_rmu), Y-sharded
	"""
	from common.isdf_fitting import fit_zeta_chunked_to_h5
	import numpy as np

	b0, b1, b2, b3, b4 = meta.band_edges
	band_range_left = (b0, b3)
	band_range_right = (b1, b4)

	isdf_pair_mode = str(cfg.isdf_pair_mode).strip().lower()
	pair_channels = 1 if isdf_pair_mode == "spin_traced" else meta.nspinor ** 2

	chunks = compute_optimal_chunks(
		n_k=meta.nk_tot, n_b=b4 - b0, n_s=meta.nspinor,
		n_rmu=meta.n_rmu, n_r=meta.n_rtot, n_q=meta.nk_tot,
		fft_grid=meta.fft_grid, n_devices=jax.device_count(),
		memory_budget_gb=cfg.memory_per_device_gb,
		target_utilization=cfg.chunk_target_utilization,
		p_x=mesh_xy.devices.shape[0], p_y=mesh_xy.devices.shape[1],
		n_b_left=band_range_left[1] - band_range_left[0],
		n_b_right=band_range_right[1] - band_range_right[0],
		pair_density_channels=pair_channels, verbose=True,
		r_chunk_override=cfg.r_chunk_override if cfg.r_chunk_override > 0 else None,
		zct_stage_cap_gb=cfg.zct_stage_cap_gb,
	)

	mem_est = chunks.get('memory_estimate', {})
	if jax.process_index() == 0 and mem_est:
		peak = mem_est.get('peak_estimate_gb', 0.0)
		budget = mem_est.get('budget_gb', 0.0)
		bottleneck = mem_est.get('bottleneck', 'unknown')
		print_fn(f"    Memory estimate: peak {peak:.2f} GB (budget {budget:.2f} GB), bottleneck={bottleneck}")
		for key, val in mem_est.get('limit_info', {}).items():
			print_fn(f"      {key}: {val:.1f}")

	zeta_h5_path = os.path.join(tmp_dir, "zeta_q.h5")
	print_fn(f"\n  Chunked ISDF fitting:")
	print_fn(f"    Band chunks: {chunks['band_chunk']}")
	print_fn(f"    R chunks:    {chunks['chunk_r']} (contiguous r-space)")
	print_fn(f"    Q chunks:    {chunks['q_chunk']}")
	print_fn(f"    Pair mode:   {isdf_pair_mode}")
	print_fn(f"    Zeta output: {zeta_h5_path}")

	with timing.section("gw_jax.zeta_fit_chunked"):
		psi_l_yr, psi_r_yr, _psi_l_xn, _psi_r_xn = fit_zeta_chunked_to_h5(
			wfn=wfn, sym=sym, meta=meta,
			centroid_indices=centroid_indices, mesh_xy=mesh_xy,
			chunk_r=chunks['chunk_r'], output_file=zeta_h5_path,
			band_chunk_size=chunks['band_chunk'],
			q_chunk_size=chunks['q_chunk'],
			q_gather_size=chunks.get('q_gather', 0),
			bispinor=cfg.bispinor,
			use_gspace_cache=chunks.get('use_gspace_cache', True),
			band_range_left=band_range_left,
			band_range_right=band_range_right,
			isdf_pair_mode=isdf_pair_mode,
			k_chunk_size=chunks.get('k_chunk', 0),
		)

	return zeta_h5_path, psi_l_yr, psi_r_yr, mem_est


def compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est=None, print_fn=print):
	"""Compute bare Coulomb V_qmunu from zeta HDF5 and write G0 back.

	Returns (V_qmunu, G0) where:
	  - V_qmunu: (1, npol, npol, nkx, nky, nkz, n_rmu, n_rmu)
	  - G0: (n_rmu,) ζ_μ(G=0) at q=0
	"""
	from .compute_vcoul import compute_all_V_q_from_zeta_h5
	from common import jax_profile
	import numpy as np
	import h5py

	# Filesystem sync before reading
	if jax.process_index() == 0:
		os.sync()
	jax.experimental.multihost_utils.sync_global_devices("zeta_flush")

	bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)

	# Memory budget for V_q computation
	if mem_est is None:
		mem_est = {}
	budget_gb = float(mem_est.get('available_vcoul_gb', cfg.memory_per_device_gb))
	try:
		from common.gpu_utils import get_device_memory_info
		budget_gb = min(budget_gb, float(get_device_memory_info().get('budget_gb', budget_gb)))
	except Exception:
		pass

	n_G = meta.n_rtot
	m_budget = max(0.1, budget_gb) * 1e9
	m_per_mu = 3 * 16 * n_G  # 2 zeta + 1 FFT workspace, complex128
	mu_chunk = max(1, min(meta.n_rmu, int(m_budget / m_per_mu)))
	q_batch = 1
	if mu_chunk >= meta.n_rmu and meta.nk_tot > 1:
		bytes_per_q = 2.0 * 16 * meta.n_rmu * n_G
		q_batch = max(1, min(4, meta.nk_tot, int(m_budget // max(1.0, bytes_per_q))))
	print_fn(f"    V_q budget:    {budget_gb:.2f} GB")
	print_fn(f"    V_q mu chunks: {mu_chunk}")
	if q_batch > 1:
		print_fn(f"    V_q q batches: {q_batch}")
	bare_cutoff = getattr(cfg, 'bare_coulomb_cutoff', None)
	if bare_cutoff is not None:
		print_fn(f"    V_q bare cutoff: {bare_cutoff:.1f} Ry")

	with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
		with h5py.File(zeta_h5_path, 'r') as zeta_h5:
			with mesh_xy:
				V_q_raw, G0_all = compute_all_V_q_from_zeta_h5(
					zeta_h5, kgrid=meta.kgrid, fft_grid=meta.fft_grid,
					bvec=bvec, cell_volume=meta.cell_volume,
					mu_chunk_size=mu_chunk, mesh_xy=mesh_xy,
					sys_dim=meta.sys_dim,
					q_batch_size=q_batch if mu_chunk >= meta.n_rmu else None,
					bdot=np.asarray(wfn.bdot, dtype=np.float64) if meta.sys_dim == 0 else None,
					mc_average_vcoul_body=cfg.mc_average_vcoul_body,
					bare_coulomb_cutoff=bare_cutoff,
				)

	# Write G0 = ζ_μ(G=0) at q=0 back to zeta file
	G0_gathered = jax.experimental.multihost_utils.process_allgather(G0_all)
	if G0_gathered.ndim == 5 and G0_gathered.shape[0] == 1:
		G0_gathered = G0_gathered[0]
	if jax.process_index() == 0:
		with h5py.File(zeta_h5_path, 'a') as f:
			if 'g0_mu' in f:
				del f['g0_mu']
			f.create_dataset('g0_mu', data=np.asarray(G0_gathered))
	jax.experimental.multihost_utils.sync_global_devices("g0_write")

	# Reshape to (1, npol, npol, nkx, nky, nkz, n_rmu, n_rmu)
	nkx, nky, nkz = meta.kgrid
	V_qmunu = jnp.array(jnp.broadcast_to(
		V_q_raw[None, None, None, :, :, :, :, :],
		(1, meta.npol, meta.npol, nkx, nky, nkz, meta.n_rmu, meta.n_rmu)))

	# G0 at q=0
	G0 = G0_gathered
	while G0.ndim > 1:
		G0 = G0[0]

	print_fn(f"\n  V_q computed:")
	print_fn(f"    Shape: {V_qmunu.shape}")
	print_fn(f"    V_q=0 trace: {jnp.trace(V_q_raw[0, 0, 0]).real:.4f}")

	return V_qmunu, G0


def build_wavefunction_bundle(
	wfn, sym, meta, band_slices, mesh_xy,
	*, psi_l_yr=None, psi_r_yr=None,
	psi_full_yr=None, psi_full_xn=None,
	enk_full=None, print_fn=print,
):
	"""Build Wavefunctions bundle from either ISDF or restart arrays.

	Fresh ISDF: pass psi_l_yr + psi_r_yr (left/right halves assembled internally).
	Restart:    pass psi_full_yr (+ optional psi_full_xn to skip resharding).
	"""
	from .wavefunction_bundle import build_wavefunctions, build_wavefunctions_from_full
	from common.load_wfns import get_enk_bandrange

	b0, b1, b2, b3, b4 = meta.band_edges
	if enk_full is None:
		enk_full, _ = get_enk_bandrange(
			wfn, sym, (b0, b4), (b1, b3), nspinor=meta.nspinor)

	if psi_full_yr is not None:
		wfns = build_wavefunctions_from_full(
			psi_full_yr, psi_xn_full=psi_full_xn,
			enk_full=enk_full, slices=band_slices, mesh_xy=mesh_xy)
	else:
		wfns = build_wavefunctions(
			psi_l_yr, psi_r_yr,
			enk_full=enk_full, slices=band_slices, mesh_xy=mesh_xy)

	print_fn(f"  Wavefunctions built (b0:b4={band_slices.nb_full} bands, "
	         f"4 sharded copies: xn/xr/yr/yn)")
	return wfns


def prepare_isdf_and_wavefunctions(
	*, cfg, wfn, sym, meta, centroid_indices, band_slices,
	mesh_xy, tmp_dir, tensors_filename, print0, **_ignored,
):
	"""Run ISDF fitting (or load restart), build wavefunction bundle.

	Pipeline:
	  1. fit_zeta       → ζ(q) to H5, centroid wavefunctions
	  2. compute_V_q    → bare Coulomb V_qmunu from ζ(q)
	  3. build_wavefunction_bundle → 4-copy sharded Wavefunctions

	Returns SimpleNamespace(V_qmunu, wf_bundle).
	"""
	from file_io import write_restart_state_to_h5, save_restart_state_per_proc

	if not cfg.restart:
		# --- ISDF fitting ---
		with mesh_xy:
			zeta_path, psi_l_yr, psi_r_yr, mem_est = fit_zeta(
				wfn, sym, meta, centroid_indices, mesh_xy,
				cfg, tmp_dir, print_fn=print0)

			V_qmunu, G0 = compute_V_q(
				zeta_path, wfn, meta, mesh_xy, cfg,
				mem_est=mem_est, print_fn=print0)

		# --- Wavefunction bundle ---
		with timing.section("gw_jax.wavefunction_setup"):
			wfns = build_wavefunction_bundle(
				wfn, sym, meta, band_slices, mesh_xy,
				psi_l_yr=psi_l_yr, psi_r_yr=psi_r_yr, print_fn=print0)

		# --- Restart checkpoint ---
		write_restart_state_to_h5(
			tensors_filename, V_qmunu,
			wfns.psi_yr, wfns.enk, None,
			G0_mu_nu=G0, init_W0=True)
		save_restart_state_per_proc(
			os.path.join(tmp_dir, "isdf_tensors"),
			V_qmunu, None, wfns.psi_yr, wfns.enk, meta, mesh_xy)
		V_qmunu.block_until_ready()
		print0("  Chunked ISDF path complete")
	else:
		# --- Restart from H5 ---
		from file_io import load_restart_state_from_h5
		with timing.section("gw_jax.restart_load"):
			V_qmunu, _S, psi_full_xn, psi_full_yr, enk_full, _V0, _G0 = (
				load_restart_state_from_h5(tensors_filename, mesh_xy, band_slices=band_slices))
			print0("  Loaded restart tensors from H5.")
			wfns = build_wavefunction_bundle(
				wfn, sym, meta, band_slices, mesh_xy,
				psi_full_yr=psi_full_yr, psi_full_xn=psi_full_xn,
				enk_full=enk_full, print_fn=print0)

	return SimpleNamespace(V_qmunu=V_qmunu, wf_bundle=wfns)

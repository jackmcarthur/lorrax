"""ISDF fitting orchestration and memory-aware chunk sizing for LORRAX GW.

  compute_optimal_chunks — per-device memory model for the 6-stage ISDF pipeline
  fit_zeta / compute_V_q / build_wavefunction_bundle — pipeline steps
  prepare_isdf_and_wavefunctions — top-level orchestrator called by main()
"""
import os
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import h5py

import common.timing as timing
from common import jax_profile



def compute_optimal_chunks(
    meta, mesh_xy, memory_budget_gb: float,
    target_utilization: float = 0.97,
    verbose: bool = True,
    n_b_left: int | None = None, n_b_right: int | None = None,
    r_chunk_override: int | None = None,
    zct_stage_cap_gb: float | None = None,
) -> dict:
    """Derive ISDF chunk sizes that fit within the per-device memory budget.

    System dimensions come from meta; device grid from mesh_xy.
    The ISDF r-chunk loop has 6 stages, each with cost linear in chunk_r (cr):

        stage_cost(cr) = base + αᵢ·cr + cᵢ

    where base = centroids + L_q + optional G-space cache (persistent arrays),
    αᵢ is the per-cr byte coefficient, and cᵢ is the cr-independent overhead.
    The optimal cr is min over stages of (headroom − cᵢ) / αᵢ — one division
    per stage, no iterative search.

    The 6 stages and their XLA HLO-calibrated multipliers (CrI3, 16 GPUs):

        FFT:     ψ(r_chunk) accumulator + 3× per-k FFT transient (cuFFT)
        Pair:    ψ(r_chunk) + 2× pair density P(nk, μ, cr)
        ZCT:     4× pair data  (P_r alive outside JIT + 3× donated left IFFT)
        Reshard: 4× Z_col      (input + output + 2× NCCL all-to-all temp)
        Solve:   2× Z_col + per-q (2× Z_slice + L_rep + 3× L_full)
        Gather:  Z_col + per-q (input/p + 2× replicated output incl. NCCL)

    Sharding: centroids and pair densities shard on (p_x, p_y).  Z_col and
    solve arrays shard on p = p_x·p_y combined.  The α coefficients encode
    the per-device sizes after sharding.
    """
    if memory_budget_gb <= 0:
        raise ValueError("memory_per_device_gb must be > 0.")

    B = 16.0  # bytes per complex128
    p_x, p_y = mesh_xy.devices.shape
    p = p_x * p_y
    nb = meta.band_edges[4] - meta.band_edges[0]  # b4 - b0
    nb_l = int(n_b_left) if n_b_left is not None else nb
    nb_r = int(n_b_right) if n_b_right is not None else nb
    nk, ns, mu, nq, nr = float(meta.nk_tot), float(meta.nspinor), float(meta.n_rmu), float(meta.nk_tot), float(meta.n_rtot)
    nx, ny, nz = meta.fft_grid

    def _mem(*dims, shard=1):
        """complex128 array bytes: 16 · ∏(dims) / shard."""
        result = B
        for d in dims:
            result *= d
        return result / shard

    m_budget = memory_budget_gb * 1e9 * target_utilization
    m_zct_cap = float(zct_stage_cap_gb) * 1e9 if zct_stage_cap_gb and zct_stage_cap_gb > 0 else None

    # ---- Persistent arrays (always resident during chunk loop) ----
    # Centroids: left/right × X-sharded + Y-sharded copies.
    # "full" = during initial load (all nb bands); "persist" = after slicing to nb_l + nb_r.
    m_centroids_full = _mem(nk, ns, mu, nb, shard=p_y) + _mem(nk, ns, mu, nb, shard=p_x)
    m_centroids = _mem(nk, ns, mu, nb_l + nb_r, shard=p_y) + _mem(nk, ns, mu, nb_l + nb_r, shard=p_x)
    if m_centroids_full + m_centroids > m_budget:
        raise ValueError(
            f"Centroid storage requires {(m_centroids_full + m_centroids)/1e9:.2f} GB/device "
            f"but only {memory_budget_gb:.2f} GB allocated.")

    # ---- Band-chunked FFT (centroid extraction, runs before chunk loop) ----
    # ψ(G) → IFFT → ψ(r): 3× buffer (input + output + cuFFT scratch) + phase array.
    # Bands are sharded across all p devices; bpd = ceil(band_chunk / p).
    FFT_COPIES = 3
    m_phase = _mem(nk, nr)
    headroom_fft = m_budget - m_centroids_full
    if headroom_fft <= m_phase:
        raise ValueError("Insufficient memory for even a single-band FFT chunk.")
    one_band = FFT_COPIES * _mem(nk, ns, nr)
    bpd_max = max(1, int((headroom_fft - m_phase) / one_band))
    band_chunk = min(nb, bpd_max * p)
    bpd = max(1, -(-band_chunk // p))  # ceil div matching read_Gvecs_to_devices
    peak_fft_stage = m_centroids_full + FFT_COPIES * _mem(nk, bpd, ns, nr) + m_phase

    # ---- C_q build (pair density → C_q → L_q, runs before chunk loop) ----
    stage_cct = m_centroids + 2 * _mem(nk, mu, mu, shard=p_x * p_y) + _mem(nq, mu, mu, shard=p_x * p_y)
    if stage_cct > m_budget:
        raise ValueError(f"C_q build requires {stage_cct/1e9:.2f} GB/device — exceeds budget.")
    m_L_q = _mem(nq, mu, mu, shard=p_x * p_y)  # L_q persists; same size as C_q

    # ---- G-space cache: stores ψ(G) for all bands to avoid re-reading H5 per chunk ----
    m_gspace_cache = _mem(nk, nb, ns, nr, shard=p)

    # ======================================================================
    # Per-cr cost coefficients α (bytes per unit chunk_r on one device).
    # Every chunk-loop stage cost is:  base + αᵢ·cr [+ cᵢ]
    # The optimal cr = min over i of (headroom − cᵢ) / αᵢ.
    # ======================================================================
    α_psi     = _mem(nk, nb, ns, shard=p_y)        # ψ(r_chunk) output accumulator
    α_pair    = _mem(nk, mu, shard=p_x * p_y)  # one pair density P(nk, μ, cr)
    α_zcol    = _mem(nq, mu, shard=p)               # Z_col column slice
    α_z_slice = _mem(mu, shard=p)                   # per-q Z_slice or Z_transpose
    α_gather  = _mem(mu, shard=p) + 2 * _mem(mu)    # sharded input + replicated output + NCCL
    c_solve   = _mem(mu, mu, shard=p_x * p_x) + 3 * _mem(mu, mu)  # L_rep(sharded) + 3×L_full(μ²)

    def _max_cr(headroom, fft_cost_per_k):
        """Invert each stage's linear cost to get max feasible cr per stage."""
        limits = {
            'fft':     (headroom - fft_cost_per_k) / α_psi if α_psi > 0 else nr,
            'pair':    headroom / (α_psi + 2 * α_pair) if (α_psi + 2 * α_pair) > 0 else nr,
            'zct':     headroom / (4 * α_pair) if α_pair > 0 else nr,
            'reshard': headroom / (4 * α_zcol) if α_zcol > 0 else nr,
            'solve':   (headroom - c_solve) / (2 * α_zcol + 2 * α_z_slice) if (2 * α_zcol + 2 * α_z_slice) > 0 else nr,
            'gather':  headroom / (α_zcol + α_gather) if (α_zcol + α_gather) > 0 else nr,
        }
        # Optional soft cap on ZCT stage (env var override for tight-memory systems)
        if m_zct_cap is not None and α_pair > 0:
            zct_headroom = m_zct_cap - (m_budget - headroom)
            if zct_headroom > 0:
                limits['zct'] = min(limits['zct'], zct_headroom / (4 * α_pair))
        return limits

    def _eval_stages(cr, base, fft_cost_per_k):
        """Forward-evaluate all stage costs at a given cr (algebraic inverse of _max_cr)."""
        m_zcol = α_zcol * cr
        m_solve_per_q = 2 * α_z_slice * cr + c_solve
        m_gather_per_q = α_gather * cr
        stages = {
            'fft':     base + α_psi * cr + fft_cost_per_k,
            'pair':    base + (α_psi + 2 * α_pair) * cr,
            'zct':     base + 4 * α_pair * cr,
            'reshard': base + 4 * m_zcol,
            'solve':   base + 2 * m_zcol + m_solve_per_q,
            'gather':  base + m_zcol + m_gather_per_q,
        }
        # k_batch: fill remaining FFT headroom for throughput (does not affect cr choice)
        fft_head = m_budget - base - α_psi * cr
        k_batch = 1
        if fft_cost_per_k > 0 and fft_head > fft_cost_per_k:
            k_batch = min(int(nk), max(1, int(fft_head * 0.5 / fft_cost_per_k)))
        return stages, m_zcol, m_solve_per_q, m_gather_per_q, k_batch

    def _find_r_chunk(use_cache, fft_cost_per_k, override=None):
        """Compute optimal cr from direct formula, round down to p-divisible."""
        base = m_centroids + m_L_q + (m_gspace_cache if use_cache else 0.0)
        headroom = m_budget - base
        if headroom <= c_solve:
            return None
        if override and override > 0:
            cr = min(int(override), int(nr))
        else:
            cr = min(int(nr), max(0, int(min(_max_cr(headroom, fft_cost_per_k).values()))))
        pt = p_x * p_y  # cr must be divisible by total device count for solve sharding
        if pt > 1 and cr > 0:
            cr -= cr % pt
        if cr <= 0:
            return None
        stages, m_zcol, m_spq, m_gpq, k_batch = _eval_stages(cr, base, fft_cost_per_k)
        return {
            'chunk_r': cr, 'peak': max(stages.values()),
            'bottleneck': max(stages, key=stages.get), 'stages': stages,
            'base': base, 'zcol': m_zcol, 'solve_per_q': m_spq,
            'gather_per_q': m_gpq, 'k_batch': k_batch,
            'cache_bytes': m_gspace_cache if use_cache else 0.0,
        }

    # ---- Solve for chunk sizes: try cache→no-cache, halving band_chunk if needed ----
    fft_per_k = FFT_COPIES * _mem(bpd, ns, nr) + _mem(nr)
    result = _find_r_chunk(True, fft_per_k, r_chunk_override) or \
             _find_r_chunk(False, fft_per_k, r_chunk_override)
    while result is None and bpd > 1:
        bpd = max(1, bpd // 2)
        band_chunk = min(nb, bpd * p)
        bpd = max(1, -(-band_chunk // p))
        fft_per_k = FFT_COPIES * _mem(bpd, ns, nr) + _mem(nr)
        peak_fft_stage = m_centroids_full + FFT_COPIES * _mem(nk, bpd, ns, nr) + m_phase
        if verbose:
            print(f"    Reducing band_chunk to {band_chunk} (bands/device={bpd})")
        result = _find_r_chunk(True, fft_per_k, r_chunk_override) or \
                 _find_r_chunk(False, fft_per_k, r_chunk_override)
    if result is None:
        raise ValueError("Unable to find r-chunk that fits the memory budget.")

    # ---- q_chunk (solve) and q_gather (H5 write): same linear model, different base ----
    base, m_zcol = result['base'], result['zcol']
    avail_solve = max(0.0, m_budget - base - 2 * m_zcol)
    q_chunk = max(1, min(int(nq), int(avail_solve / result['solve_per_q']))) if result['solve_per_q'] > 0 else 1
    avail_gather = max(0.0, m_budget - base - m_zcol)
    q_gather = max(1, min(int(nq), int(avail_gather / result['gather_per_q']))) if result['gather_per_q'] > 0 else int(nq)

    # ---- Overall peak across all stages (chunk loop + pre-loop) ----
    overall_peak = max(
        result['peak'],
        base + 2 * m_zcol + q_chunk * result['solve_per_q'],
        base + m_zcol + q_gather * result['gather_per_q'],
        peak_fft_stage, m_centroids_full + m_centroids, stage_cct)

    k_chunk = result['k_batch']
    if k_chunk < int(nk) and verbose:
        print(f"    K-point chunking: {k_chunk} k-pts per FFT batch (total {int(nk)})")

    return {
        'band_chunk': band_chunk,
        'chunk_r': result['chunk_r'],
        'q_chunk': q_chunk,
        'q_gather': q_gather,
        'k_chunk': k_chunk,
        'use_gspace_cache': result['cache_bytes'] > 0,
        'memory_estimate': {
            'peak_estimate_gb': overall_peak / 1e9,
            'budget_gb': memory_budget_gb,
            'bottleneck': result['bottleneck'],
            'available_vcoul_gb': max(0.0, m_budget - m_centroids) / 1e9,
            'limit_info': {k: v / 1e9 for k, v in result['stages'].items()},
        },
    }


def get_effective_chunk_size(chunk_size: int) -> int | None:
    """Convert chunk_size flag: -1=None (all bands), 0=auto (64), 1-2048=explicit."""
    if chunk_size == -1:
        return None
    if chunk_size == 0:
        return 64
    if 1 <= chunk_size <= 2048:
        return chunk_size
    raise ValueError(f"chunk_size must be -1, 0, or 1-2048, got {chunk_size}")


# Backward-compatible re-exports
from .gw_config import read_lorrax_input, read_cohsex_input  # noqa: F401


def get_bandranges(nv, nc, nband, nelec):
	r"""Return ranges of bands necessary for \sigma_{X,SX,COH}.

	Legacy helper used by psp/get_DFT_mtxels.py.  GW code uses BandSlices instead.
	"""
	nvrange = [int(nelec - nv), int(nelec)]
	ncrange = [int(nelec), int(nelec + nc)]
	nsigmarange = [int(nelec - nv), int(nelec + nc)]
	n_fullrange = [0, int(nband)]
	n_valrange = [0, int(nelec)]
	return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


def fit_zeta(wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices, tmp_dir,
             psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print):
	"""Fit ISDF interpolation vectors ζ and write to HDF5.

	The caller supplies (a) the full-range centroid wavefunctions
	(``psi_rmu_Y`` / ``psi_rmuT_X``, spanning [b0, b4) as returned by
	``load_centroids_band_chunked``) and (b) the chunk plan from
	:func:`compute_optimal_chunks`.  Returns ``(zeta_h5_path, mem_est)``.
	"""
	from common.isdf_fitting import fit_zeta_chunked_to_h5

	# ISDF left/right band windows (pair density needs asymmetric ranges)
	band_range_left = (band_slices.b0, band_slices.b3)   # all val + sigma cond
	band_range_right = (band_slices.b1, band_slices.b4)   # sigma val + all cond

	mem_est = chunks.get('memory_estimate', {})
	aot_peak_gb = None  # filled in below if the AOT model is available
	if jax.process_index() == 0 and mem_est:
		print_fn(f"    Memory estimate: peak {mem_est['peak_estimate_gb']:.2f} GB "
		         f"(budget {mem_est['budget_gb']:.2f} GB), bottleneck={mem_est['bottleneck']}")
		stages = mem_est.get('limit_info', {})
		if stages:
			print_fn(f"    Per-stage: " + "  ".join(f"{k}={v:.2f}" for k, v in stages.items()) + " GB")

		# AOT-derived driver-level peak — sanity-check against per-stage
		# heuristic.  When ``use_aot_chunk_chooser`` is set, the AOT
		# chooser's pick supersedes ``chunks['chunk_r']`` and
		# ``chunks['band_chunk']``; otherwise this block just LOGS the
		# predicted peak alongside the heuristic output.  Silent when
		# the fit artifact is missing or the prediction throws.
		try:
			from gw.aot_memory_model import (
				predict_kernel_peak, SysDims, MeshSpec, Knobs,
				choose_chunks_analytic, choose_chunks_heuristic,
				describe_chunks,
			)
			p_x, p_y = mesh_xy.devices.shape
			# n_b = UNION range (bytes of psi_G cache / band-chunk FFT).
			# n_b_sum = nb_L + nb_R (L+R pair-density work + centroid bytes).
			# Symmetric production GW (nval == nelec, nband == nelec+ncond)
			# has nb_L == nb_R == nb_full so n_b_sum == 2·n_b; asymmetric
			# windows (pseudobands, sub-valence, extra-cond) are picked
			# up correctly here.
			nb_L = band_range_left[1] - band_range_left[0]
			nb_R = band_range_right[1] - band_range_right[0]
			nb_full = (max(band_range_left[1], band_range_right[1])
			           - min(band_range_left[0], band_range_right[0]))
			aot_sys = SysDims(
				kgrid=tuple(meta.kgrid),
				fft_grid=tuple(meta.fft_grid),
				n_rmu=int(meta.n_rmu),
				n_s=int(meta.nspinor),
				n_b=int(nb_full),
				n_b_sum=int(nb_L + nb_R),
				n_r=int(meta.n_rtot),
			)
			aot_mesh = MeshSpec(p_x=int(p_x), p_y=int(p_y))
			if cfg.use_aot_chunk_chooser:
				# 20/80 heuristic is the default — simpler, no DoE deps.
				# Falls back to the regressed-fit analytic chooser if
				# the user sets LORRAX_CHOOSER_MODE=analytic.
				_mode = os.environ.get("LORRAX_CHOOSER_MODE", "heuristic")
				budget = (cfg.memory_per_device_gb * 1e9
				          * cfg.chunk_target_utilization)
				if _mode == "analytic":
					choice = choose_chunks_analytic(
						aot_sys, aot_mesh,
						budget_bytes=budget,
						kernel_name="fit_one_rchunk", tag="current",
					)
				else:
					choice = choose_chunks_heuristic(
						aot_sys, aot_mesh, budget_bytes=budget,
					)
				print_fn(f"    {describe_chunks(choice)}")
				# Override the heuristic's chunk_r / band_chunk.  Keep
				# q_chunk, q_gather, k_chunk from the old chooser since
				# the AOT model doesn't cover them yet.
				chunks['chunk_r'] = int(choice.chunk_r)
				chunks['band_chunk'] = int(choice.band_chunk)
				aot_peak_gb = choice.peak_bytes / 1e9
			else:
				aot_peak = predict_kernel_peak(
					"fit_one_rchunk", aot_sys,
					Knobs.of(chunk_r=int(chunks['chunk_r']),
					         band_chunk=int(chunks['band_chunk'])),
					aot_mesh, tag="current",
				)
				print_fn(f"    AOT fit_one_rchunk peak (driver-level): "
				         f"{aot_peak / 1e9:.2f} GB")
				aot_peak_gb = aot_peak / 1e9
		except Exception as _aot_exc:
			if cfg.use_aot_chunk_chooser:
				print_fn(f"    AOT chooser FAILED ({_aot_exc!r}); "
				         f"falling back to heuristic.")

	zeta_h5_path = os.path.join(tmp_dir, "zeta_q.h5")
	print_fn(f"\n  Chunked ISDF fitting:")
	print_fn(f"    Band chunks: {chunks['band_chunk']}")
	print_fn(f"    R chunks:    {chunks['chunk_r']} (contiguous r-space)")
	print_fn(f"    Q chunks:    {chunks['q_chunk']}")
	print_fn(f"    Zeta output: {zeta_h5_path}")

	# Band norms for pseudobands normalization (1.0 for deterministic bands)
	_band_norms = getattr(wfn, 'band_norms', None)

	with timing.section("gw_jax.zeta_fit_chunked"), jax_profile.trace_section("zeta_fit"):
		peak_bytes = fit_zeta_chunked_to_h5(
			wfn=wfn, sym=sym, meta=meta,
			centroid_indices=centroid_indices, mesh_xy=mesh_xy,
			chunk_r=chunks['chunk_r'], output_file=zeta_h5_path,
			psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
			band_chunk_size=chunks['band_chunk'],
			q_chunk_size=chunks['q_chunk'],
			q_gather_size=chunks.get('q_gather', 0),
			bispinor=cfg.bispinor,
			use_gspace_cache=chunks.get('use_gspace_cache', True),
			band_range_left=band_range_left,
			band_range_right=band_range_right,
			k_chunk_size=chunks.get('k_chunk', 0),
			band_norms=_band_norms,
			use_ffi_io=cfg.use_ffi_io,
			use_phdf5_gspace=cfg.use_phdf5_gspace,
		)

	budget_gb = mem_est.get('budget_gb', cfg.memory_per_device_gb)
	if peak_bytes > 0:
		peak_gb = peak_bytes / 1e9
		print_fn(f"    GPU high-water mark: {peak_gb:.2f} GB / {budget_gb:.2f} GB budget "
		         f"({100 * peak_gb / budget_gb:.0f}%)")
		# γ calibration: runtime peak vs AOT-predicted peak.  γ > 1 means
		# AOT under-predicts (expected for FFT-heavy kernels because
		# cuFFT scratch is invisible to memory_analysis); γ < 1 means
		# AOT over-predicts (often XLA remat triggered at runtime).
		# Logged for manual tracking — wire a CLI to roll these into
		# per-kernel γ calibration once we have enough data.
		if aot_peak_gb is not None and aot_peak_gb > 0:
			gamma = peak_gb / aot_peak_gb
			print_fn(f"    γ (runtime / AOT-pred) = {gamma:.3f}  "
			         f"(AOT predicted {aot_peak_gb:.2f} GB)")

	return zeta_h5_path, mem_est


def compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est=None, print_fn=print, bgw_v_grid_fn=None):
	"""Compute bare Coulomb V_qmunu from zeta HDF5 and write G0 back.

	Returns (V_qmunu, G0) where V_qmunu has shape (1, npol, npol, nkx, nky, nkz, μ, μ)
	and G0 is (n_rmu,) ζ_μ(G=0) at q=0.
	"""
	from .compute_vcoul import compute_all_V_q_from_zeta_h5

	if jax.process_index() == 0:
		os.sync()
	jax.experimental.multihost_utils.sync_global_devices("zeta_flush")

	bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)

	# V_q memory model: 2 zeta reads + 1 FFT workspace = 3× (μ × n_G × 16 bytes)
	if mem_est is None:
		mem_est = {}
	budget_gb = float(mem_est.get('available_vcoul_gb', cfg.memory_per_device_gb))
	try:
		from common.gpu_utils import get_device_memory_info
		budget_gb = min(budget_gb, float(get_device_memory_info().get('budget_gb', budget_gb)))
	except Exception:
		pass
	m_budget = max(0.1, budget_gb) * 1e9
	m_per_mu = 3 * 16 * meta.n_rtot
	mu_chunk = max(1, min(meta.n_rmu, int(m_budget / m_per_mu)))
	q_batch = 1
	if mu_chunk >= meta.n_rmu and meta.nk_tot > 1:
		q_batch = max(1, min(4, meta.nk_tot, int(m_budget // max(1.0, 2.0 * 16 * meta.n_rmu * meta.n_rtot))))

	# get_device_memory_info reports this rank's free HBM, which can
	# differ across ranks (pool fragmentation, non-symmetric allocations
	# before this point) → mu_chunk + q_batch can diverge across ranks.
	# compute_all_V_q_from_zeta_h5 then branches on (n_chunks == 1) at
	# line ~884; divergent branches across ranks deadlock on the first
	# collective inside one branch (observed 2026-04-18 w/ FFI read).
	# Fix: broadcast rank 0's numbers to every rank so they agree.
	if jax.process_count() > 1:
		_mq = jnp.asarray([int(mu_chunk), int(q_batch)], dtype=jnp.int64)
		_mq = jax.experimental.multihost_utils.broadcast_one_to_all(_mq)
		mu_chunk = int(jax.device_get(_mq)[0])
		q_batch = int(jax.device_get(_mq)[1])

	# Auto-default to 4·ecutwfc: tightest |q+G|² cutoff that drops zero
	# physics (triangle ineq on ψ*ψ bounds pair-density at |G|² ≤ 4·ecutwfc)
	# and activates the V_q sphere gather in compute_vcoul.
	if cfg.bare_coulomb_cutoff is None:
		vcoul_cutoff_ry = 4.0 * float(wfn.ecutwfc)
		print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry (auto: 4·ecutwfc)")
	else:
		vcoul_cutoff_ry = float(cfg.bare_coulomb_cutoff)
		print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry")

	print_fn(f"    V_q budget:    {budget_gb:.2f} GB")
	print_fn(f"    V_q mu chunks: {mu_chunk}")
	if q_batch > 1:
		print_fn(f"    V_q q batches: {q_batch}")

	from file_io.slab_io import SlabIO
	with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
		# Force use_ffi_io=False for the V_q zeta READ: V_q overlaps
		# disk I/O with GPU compute via a background ``ThreadPoolExecutor``,
		# which is incompatible with SlabIO's FFI-read shard_map
		# dispatch — the collective H5Dread can't safely interleave
		# with the main thread's JAX ops on the same rank.  The write
		# path uses the FFI (``cfg.use_ffi_io``) separately; only
		# this specific read is pinned to the allgather/h5py path.
		with SlabIO(zeta_h5_path, mode='r', mesh=mesh_xy,
		            use_ffi_io=False) as zeta_io:
			with mesh_xy:
				V_q_raw, G0_all = compute_all_V_q_from_zeta_h5(
					zeta_io, kgrid=meta.kgrid, fft_grid=meta.fft_grid,
					bvec=bvec, cell_volume=meta.cell_volume,
					mu_chunk_size=mu_chunk, mesh_xy=mesh_xy,
					sys_dim=meta.sys_dim,
					q_batch_size=q_batch if mu_chunk >= meta.n_rmu else None,
					bdot=np.asarray(wfn.bdot, dtype=np.float64) if meta.sys_dim == 0 else None,
					mc_average_vcoul_body=cfg.mc_average_vcoul_body,
					bare_coulomb_cutoff=vcoul_cutoff_ry,
					bgw_v_grid_fn=bgw_v_grid_fn,
					n_rmu=meta.n_rmu, n_rtot=meta.n_rtot,
				)

	# Write G0 = ζ_μ(G=0) at q=0 back to zeta file via SlabIO's deferred
	# attr path (small; rank-0-only after MPI-IO file is closed).
	G0_gathered = jax.experimental.multihost_utils.process_allgather(G0_all)
	if G0_gathered.ndim == 5 and G0_gathered.shape[0] == 1:
		G0_gathered = G0_gathered[0]
	if jax.process_index() == 0:
		with h5py.File(zeta_h5_path, 'a') as f:
			if 'g0_mu' in f:
				del f['g0_mu']
			f.create_dataset('g0_mu', data=np.asarray(G0_gathered))
	jax.experimental.multihost_utils.sync_global_devices("g0_write")

	# Add polarization axes: (nkx, nky, nkz, μ, μ) → (1, npol, npol, nkx, nky, nkz, μ, μ)
	nkx, nky, nkz = meta.kgrid
	V_qmunu = jnp.array(jnp.broadcast_to(
		V_q_raw[None, None, None], (1, meta.npol, meta.npol, nkx, nky, nkz, meta.n_rmu, meta.n_rmu)))

	G0 = G0_gathered
	while G0.ndim > 1:
		G0 = G0[0]

	print_fn(f"\n  V_q computed:")
	print_fn(f"    Shape: {V_qmunu.shape}")
	print_fn(f"    V_q=0 trace: {jnp.trace(V_q_raw[0, 0, 0]).real:.4f}")
	return V_qmunu, G0


def build_wavefunction_bundle(
	wfn, sym, meta, band_slices, mesh_xy,
	*, psi_rmu_Y, psi_rmuT_X, enk_full=None, print_fn=print,
):
	"""Build 4-copy Wavefunctions bundle from the two centroid-sampled
	arrays produced by ``load_centroids_band_chunked``.
	"""
	from .wavefunction_bundle import build_wavefunctions
	from common.load_wfns import get_enk_bandrange

	if enk_full is None:
		enk_full, _ = get_enk_bandrange(
			wfn, sym, band_slices.full_range,
			(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)

	wfns = build_wavefunctions(
		psi_rmu_Y, psi_rmuT_X,
		enk_full=enk_full, slices=band_slices, mesh_xy=mesh_xy)

	print_fn(f"  Wavefunctions built (b0:b4={band_slices.nb_full} bands, "
	         f"4 sharded copies: xn/xr/yr/yn)")
	return wfns


def prepare_isdf_and_wavefunctions(
	*, cfg, wfn, sym, meta, centroid_indices, band_slices,
	mesh_xy, tmp_dir, tensors_filename, print0, bgw_v_grid_fn=None, **_ignored,
):
	"""ISDF pipeline (non-restart path reads top-to-bottom):

	  1. ``compute_optimal_chunks`` → chunk plan (band/r/q chunk sizes).
	  2. ``load_centroids_band_chunked`` → ψ at centroids for [b0, b4).
	  3. ``fit_zeta`` → ζ.h5 (consumes ψ slices for pair density).
	  4. ``compute_V_q`` → V_qmunu, G0 (reads ζ from disk).
	  5. Flush V_q / G0 / enk + W0 placeholder to restart H5 (mode="w").
	  6. ``build_wavefunctions`` → 4-copy Wavefunctions bundle (reuses ψ).
	  7. Append ``psi_full_y`` (= wfns.psi_yr) to restart H5 (mode="a").

	Returns SimpleNamespace(V_qmunu, wf_bundle).
	"""
	from file_io import write_restart_state_to_h5, save_restart_state_per_proc
	from common.load_wfns import load_centroids_band_chunked

	if not cfg.restart:
		from common.load_wfns import get_enk_bandrange

		with mesh_xy:
			# Plan chunks (band/r/q sizes).
			chunks = compute_optimal_chunks(
				meta, mesh_xy,
				memory_budget_gb=cfg.memory_per_device_gb,
				target_utilization=cfg.chunk_target_utilization,
				n_b_left=band_slices.b3 - band_slices.b0,
				n_b_right=band_slices.b4 - band_slices.b1,
				pair_density_channels=(1 if cfg.isdf_pair_mode == "spin_traced"
				                       else meta.nspinor ** 2),
				r_chunk_override=cfg.r_chunk_override if cfg.r_chunk_override > 0 else None,
				zct_stage_cap_gb=cfg.zct_stage_cap_gb,
			)

			# Load centroid ψ once for the full [b0, b4) range; reused by
			# both the zeta fit (sliced into halves internally) and the
			# downstream Wavefunctions bundle.
			with timing.section("gw_jax.load_centroid_wfns"):
				psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
					wfn, sym, meta, centroid_indices, cfg.bispinor, mesh_xy,
					band_range=band_slices.full_range,
					band_chunk_size=chunks['band_chunk'],
				)

			zeta_path, mem_est = fit_zeta(
				wfn, sym, meta, centroid_indices, mesh_xy,
				cfg, band_slices, tmp_dir,
				psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print0)
			V_qmunu, G0 = compute_V_q(
				zeta_path, wfn, meta, mesh_xy, cfg,
				mem_est=mem_est, print_fn=print0,
				bgw_v_grid_fn=bgw_v_grid_fn)

			enk_full, _ = get_enk_bandrange(
				wfn, sym, band_slices.full_range,
				(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)

			# Flush V_q / G0 / enk + W0 placeholder immediately.
			write_restart_state_to_h5(
				tensors_filename,
				V_qmunu=V_qmunu, G0_mu_nu=G0, enk_full=enk_full,
				init_W0=True, mesh=mesh_xy, use_ffi_io=cfg.use_ffi_io,
				mode="w",
			)

			with timing.section("gw_jax.wavefunction_setup"):
				wfns = build_wavefunction_bundle(
					wfn, sym, meta, band_slices, mesh_xy,
					psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
					enk_full=enk_full, print_fn=print0)

			# Append ψ to the now-open restart file.
			write_restart_state_to_h5(
				tensors_filename,
				psi_full_y=wfns.psi_yr, mesh=mesh_xy,
				use_ffi_io=cfg.use_ffi_io, mode="a",
			)
		save_restart_state_per_proc(
			os.path.join(tmp_dir, "isdf_tensors"),
			V_qmunu, None, wfns.psi_yr, wfns.enk, meta, mesh_xy)
		V_qmunu.block_until_ready()
		print0("  Chunked ISDF path complete")
	else:
		from file_io import load_restart_state_from_h5
		with timing.section("gw_jax.restart_load"):
			rs = load_restart_state_from_h5(
				tensors_filename, mesh_xy, band_slices=band_slices)
			V_qmunu = rs.V_qmunu
			print0("  Loaded restart tensors from H5.")
			wfns = build_wavefunction_bundle(
				wfn, sym, meta, band_slices, mesh_xy,
				psi_rmu_Y=rs.psi_rmu_Y, psi_rmuT_X=rs.psi_rmuT_X,
				enk_full=rs.enk_full, print_fn=print0)

	return SimpleNamespace(V_qmunu=V_qmunu, wf_bundle=wfns)

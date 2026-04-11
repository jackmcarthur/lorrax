import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from types import SimpleNamespace
jax.config.update("jax_enable_x64", True)

# Initialize JAX distributed only when running multi-process.
# Guard: when run via `python -m gw.gw_jax`, the module executes as __main__
# first, then gw_init.py re-imports it as gw.gw_jax.  Module-level globals
# are NOT shared between these two namespace copies, so we use an env-var
# sentinel that persists across re-imports within the same process.
_DISTRIBUTED_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"

def _maybe_init_jax_distributed():
	if os.environ.get(_DISTRIBUTED_SENTINEL):
		return
	proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
						 os.environ.get("JAX_NUM_PROCESSES",
						 os.environ.get("SLURM_NTASKS", "1"))))
	if proc_count > 1:
		# Prefer auto-detection (NERSC pattern): JAX reads SLURM env vars directly
		try:
			jax.distributed.initialize()
			os.environ[_DISTRIBUTED_SENTINEL] = "1"
			return
		except Exception:
			pass
		# Fallback: explicit coordinator from SLURM_NODELIST (first node)
		coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
		if coord is None:
			import subprocess
			nodelist = os.environ.get("SLURM_NODELIST")
			if nodelist:
				try:
					result = subprocess.run(
						["scontrol", "show", "hostnames", nodelist],
						capture_output=True, text=True, check=True
					)
					first_host = result.stdout.strip().split("\n")[0]
					coord = f"{first_host}:12355"
				except Exception:
					pass
			if coord is None:
				host = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME") or "localhost"
				coord = f"{host}:12355"
		proc_id = int(os.environ.get("JAX_PROCESS_INDEX", os.environ.get("SLURM_PROCID", "0")))
		jax.distributed.initialize(coordinator_address=coord,
								   num_processes=proc_count,
								   process_id=proc_id)
	os.environ[_DISTRIBUTED_SENTINEL] = "1"

# Global mesh for sharding across bands
_maybe_init_jax_distributed()
try:
	_default_devices = jax.devices()
except RuntimeError as exc:
	if "Unknown backend: 'gpu'" in str(exc):
		os.environ.pop("JAX_PLATFORM_NAME", None)
		os.environ["JAX_PLATFORMS"] = "cpu"
		_default_devices = jax.devices("cpu")
	else:
		raise
from file_io import (
    WFNReader,
    write_sigma_to_file, write_eqp1, write_eqp_g0w0, write_sigma_omega_h5,
    write_chunked_complex_dataset_h5,
    write_sigma_freq_debug_table,
    write_qp_rotations_h5, load_kin_ion_submatrix,
    load_centroids, resolve_input_paths,
)
from common import symmetry_maps
from common.load_wfns import get_enk_bandrange
from common.isdf_fitting import fit_zeta_chunked_to_h5
from .compute_vcoul import compute_all_V_q_from_zeta_h5
from .gw_init import (
	compute_optimal_chunks,
	get_effective_chunk_size,
	read_cohsex_input,
	resolve_runtime_config,
	prepare_isdf_and_wavefunctions,
)
from .get_windows import get_window_info
from .gw_driver_helpers import (
	build_ppm_sigma_runtime_options,
	build_screening_setup,
	maybe_build_ctsp_windows,
)
from .w_isdf import compute_screening
from .minimax_config import (
	minimax_config_from_params,
	sigma_quadrature_config_from_params,
)
from .ppm_sigma import (
	compute_w0_wiwp_and_ppm_from_minimax,
	compute_sigma_c_ppm_laplace,
	compute_sigma_c_ppm_omega_grid,
)
from .qsgw_utils import (
	solve_diagonal_sigma_fixed_point,
	build_qsgw_sigma_xc,
	load_sigma_xc_diag_from_h5,
	build_qsgw_sigma_xc_from_h5,
	plot_qp_energy_comparison,
)
from .head_correction import (
	compute_static_head_terms_from_sample,
	fit_head_gn_from_samples,
	format_head_diagnostics,
	format_head_pair_diagnostics,
	format_head_sample_diagnostics,
	format_static_head_diagnostics,
	resolve_head_sample,
	static_head_terms_to_kij,
)
from .wavefunction_bundle import BandSlices, Wavefunctions
from mixing.acceleration import (
    rcrop_nojit, hermitian_to_upper_flat, upper_flat_to_hermitian
)
from common import Meta
from common import jax_profile
import common.timing as timing
import h5py
import builtins


def make_shardings(mesh_xy: Mesh) -> SimpleNamespace:
	"""Centralize all NamedSharding declarations used in this file."""
	return SimpleNamespace(
		# General 2D shardings
		xy_shard = NamedSharding(mesh_xy, P('x', 'y')),
		replicated_2 = NamedSharding(mesh_xy, P(None, None)),
		y_shard_vec = NamedSharding(mesh_xy, P('y')),
		xy0_2 = NamedSharding(mesh_xy, P(('x','y'), None)),
		x0y1_2 = NamedSharding(mesh_xy, P('x','y')),
		# Multi-dim array shardings
		x6y7_8 = NamedSharding(mesh_xy, P(None, None, None, None, None, None, 'x', 'y')),
		x3y4_5 = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y')),
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y')),
		x3_4 = NamedSharding(mesh_xy, P(None, None, None, 'x')),
		y2_4 = NamedSharding(mesh_xy, P(None, None, 'y', None)),
		x2_4 = NamedSharding(mesh_xy, P(None, None, 'x', None)),
		# Pipeline shardings
		XT_shard = NamedSharding(mesh_xy, P(None, None, 'x', None)),
		Y_shard = NamedSharding(mesh_xy, P(None, None, None, 'y')),
		X_shard = NamedSharding(mesh_xy, P(None, None, None, 'x')),
		YT_shard = NamedSharding(mesh_xy, P(None, None, 'y', None)),
		V_shard = NamedSharding(mesh_xy, P('x', 'y', None, None, None)),
		out_shard = NamedSharding(mesh_xy, P(None, None, None)),
	)

def write_w_copies_debug_h5(
	path,
	*,
	W0_screen_q=None,
	W0_ppm_q=None,
	Wiwp_ppm_q=None,
	print_fn=print,
):
	"""Write q=0 ISDF W copies and emit compact norm/difference diagnostics."""
	def _q000_munu(W_q):
		if W_q is None:
			return None
		# Use process_allgather for multi-process compatibility
		slc = W_q[0, 0, 0, 0, :, 0, :]
		slc = jax.experimental.multihost_utils.process_allgather(slc, tiled=False)
		return np.asarray(slc, dtype=np.complex128)

	W0_screen = _q000_munu(W0_screen_q)
	W0_ppm = _q000_munu(W0_ppm_q)
	Wiwp_ppm = _q000_munu(Wiwp_ppm_q)

	def _fnorm(x):
		return float(np.linalg.norm(x, ord="fro")) if x is not None else None

	norm_screen = _fnorm(W0_screen)
	norm_ppm0 = _fnorm(W0_ppm)
	norm_ppmi = _fnorm(Wiwp_ppm)

	diff_abs = None
	diff_rel = None
	if (W0_screen is not None) and (W0_ppm is not None):
		diff_abs = float(np.max(np.abs(W0_screen - W0_ppm)))
		ref = max(float(np.max(np.abs(W0_screen))), 1.0e-16)
		diff_rel = diff_abs / ref

	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with h5py.File(path, "w") as h5:
		if W0_screen is not None:
			h5.create_dataset("W0_screen_q000_munu", data=W0_screen)
		if W0_ppm is not None:
			h5.create_dataset("W0_ppm_q000_munu", data=W0_ppm)
		if Wiwp_ppm is not None:
			h5.create_dataset("Wiwp_ppm_q000_munu", data=Wiwp_ppm)
		if norm_screen is not None:
			h5.attrs["fro_W0_screen_q000"] = norm_screen
		if norm_ppm0 is not None:
			h5.attrs["fro_W0_ppm_q000"] = norm_ppm0
		if norm_ppmi is not None:
			h5.attrs["fro_Wiwp_ppm_q000"] = norm_ppmi
		if diff_abs is not None:
			h5.attrs["maxabs_W0_screen_minus_ppm_q000"] = diff_abs
		if diff_rel is not None:
			h5.attrs["rel_W0_screen_minus_ppm_q000"] = diff_rel

	print_fn(f"  W-copy debug h5:        {path}")
	if norm_screen is not None:
		print_fn(f"    ||W0(screen,q=0)||_F = {norm_screen:.10e}")
	if norm_ppm0 is not None:
		print_fn(f"    ||W0(ppm,q=0)||_F    = {norm_ppm0:.10e}")
	if norm_ppmi is not None:
		print_fn(f"    ||W(iωp,ppm,q=0)||_F = {norm_ppmi:.10e}")
	if diff_abs is not None:
		print_fn(
			f"    max|W0(screen)-W0(ppm)| = {diff_abs:.6e} "
			f"(rel={diff_rel:.6e})"
		)


def fit_zeta_and_compute_V_q_chunked(
	wfn,
	sym,
	meta: Meta,
	centroid_indices: jax.Array,
	mesh_xy: Mesh,
	output_dir: str,
	bispinor: bool = False,
	memory_budget_gb: float = 6.0,
	sys_dim: int = 2,
	r_chunk_override: int = 0,
	target_utilization: float = 0.97,
	zct_stage_cap_gb: float | None = None,
	isdf_pair_mode: str = "spin_traced",
	mc_average_vcoul_body: bool = True,
	bare_coulomb_cutoff: float | None = None,
):
	"""
	Chunked zeta fitting and V_q computation pipeline.

	This replaces the per-q-point zeta fitting in the main loop with a memory-efficient
	chunked approach that:
	1. Loads wavefunctions for full band range (b0 to b4)
	2. Slices for left (b0→b3) and right (b1→b4) band windows
	3. Fits zeta via z-chunked algorithm and writes to HDF5
	4. Reads zeta back and computes V_qmunu
	
	Args:
		wfn: WFNReader object
		sym: SymMaps object
		meta: Meta object with system info
		centroid_indices: ISDF centroid indices
		mesh_xy: 2D device mesh
		output_dir: Directory for zeta HDF5 file
		bispinor: Whether to use bispinor wavefunctions
		memory_budget_gb: Memory budget per device in GB
		sys_dim: System dimensionality (2 or 3)
		r_chunk_override: If > 0, use explicit r-chunk size (flattened xyz index).
		target_utilization: Fraction of memory_budget_gb to target in chunk sizing.
		zct_stage_cap_gb: Optional soft cap for ZCT stage peak (GB).
		isdf_pair_mode: Pair-density pathway for CCT/ZCT:
			- "spin_traced" (default)
			- "spin_matrix_frobenius" (explicit spin channels, sum_ab after contraction)
	
	Returns:
		Dictionary with:
		- V_qmunu: (1, npol, npol, nkx, nky, nkz, n_rmu, n_rmu) Coulomb matrix
		- v_q0_noG0_munu: (n_rmu, n_rmu) V_q at q=0 with G=0 excluded
		- G0_mu_nu: (n_rmu,) ζ_μ(G=0) saved for restart/debug diagnostics
		- psi_l_rmu_Y: Left centroid wfns, Y-sharded
		- psi_l_rmuT_X: Left conjugated wfns, X-sharded  
		- psi_r_rmu_Y: Right centroid wfns, Y-sharded
		- psi_r_rmuT_X: Right conjugated wfns, X-sharded
		- zeta_h5_path: Path to zeta HDF5 file
	"""
	import os

	isdf_pair_mode = str(isdf_pair_mode).strip().lower()
	if isdf_pair_mode not in ("spin_traced", "spin_matrix_frobenius"):
		raise ValueError(
			f"Unknown isdf_pair_mode={isdf_pair_mode!r}. "
			"Expected 'spin_traced' or 'spin_matrix_frobenius'."
		)
	pair_density_channels = 1 if isdf_pair_mode == "spin_traced" else meta.nspinor * meta.nspinor
	
	n_devices = jax.device_count()
	p_x = mesh_xy.devices.shape[0]
	p_y = mesh_xy.devices.shape[1]
	
	# Band ranges for left and right (gw_jax convention)
	# Left: nvplussigrange = (b0, b3) = all valence + sigma-window conduction
	# Right: ncplussigrange = (b1, b4) = sigma-window valence + all conduction
	b0, b1, b2, b3, b4 = meta.band_edges
	band_range_left = (b0, b3)
	band_range_right = (b1, b4)
	n_b_full = b4 - b0
	
	# Compute optimal chunk sizes
	nqx, nqy, nqz = meta.kgrid
	n_q = nqx * nqy * nqz
	
	chunks = compute_optimal_chunks(
		n_k=meta.nk_tot,
		n_b=n_b_full,
		n_s=meta.nspinor,
		n_rmu=meta.n_rmu,
		n_r=meta.n_rtot,
		n_q=n_q,
		fft_grid=meta.fft_grid,
		n_devices=n_devices,
		memory_budget_gb=memory_budget_gb,
		target_utilization=target_utilization,
		p_x=p_x,
		p_y=p_y,
		n_b_left=band_range_left[1] - band_range_left[0],
		n_b_right=band_range_right[1] - band_range_right[0],
		pair_density_channels=pair_density_channels,
		verbose=True,
		r_chunk_override=r_chunk_override if r_chunk_override > 0 else None,
		zct_stage_cap_gb=zct_stage_cap_gb,
	)
	
	band_chunk_size = chunks['band_chunk']
	chunk_r = chunks['chunk_r']
	q_chunk_size = chunks['q_chunk']
	k_chunk_size = chunks.get('k_chunk', 0)
	use_gspace_cache = chunks.get('use_gspace_cache', True)
	mem_est = chunks.get('memory_estimate', {})

	if jax.process_index() == 0 and mem_est:
		peak_gb = mem_est.get('peak_estimate_gb', 0.0)
		budget_gb = mem_est.get('budget_gb', 0.0)
		bottleneck = mem_est.get('bottleneck', 'unknown')
		print(f"    Memory estimate: peak {peak_gb:.2f} GB (budget {budget_gb:.2f} GB), bottleneck={bottleneck}")
		limit_info = mem_est.get('limit_info', {})
		if limit_info:
			print("    Chunk limit estimates (r-points):")
			for key in ("limit_pair", "limit_zct", "limit_zct_soft", "limit_reshard", "limit_solve", "limit_gather", "limit_default"):
				if key in limit_info:
					print(f"      {key}: {limit_info[key]:.1f}")
	
	# Output path for zeta
	zeta_h5_path = os.path.join(output_dir, "zeta_q.h5")

	print("\n  Chunked ISDF fitting:")
	print(f"    Band chunks: {band_chunk_size}")
	print(f"    R chunks:    {chunk_r} (contiguous r-space)")
	print(f"    Q chunks:    {q_chunk_size}")
	if k_chunk_size > 0 and k_chunk_size < meta.nk_tot:
		print(f"    K chunks:    {k_chunk_size} (k-batched FFT)")
	print(f"    Pair mode:   {isdf_pair_mode}")
	print(f"    G-space cache: {'enabled' if use_gspace_cache else 'disabled'}")
	print(f"    Zeta output: {zeta_h5_path}")
	
	# Step 1: Fit zeta and write to HDF5
	with timing.section("gw_jax.zeta_fit_chunked"):
		psi_l_rmu_Y, psi_l_rmuT_X, psi_r_rmu_Y, psi_r_rmuT_X = fit_zeta_chunked_to_h5(
			wfn=wfn,
			sym=sym,
			meta=meta,
			centroid_indices=centroid_indices,
			mesh_xy=mesh_xy,
			chunk_r=chunk_r,
			output_file=zeta_h5_path,
			band_chunk_size=band_chunk_size,
			q_chunk_size=q_chunk_size,
			q_gather_size=chunks.get('q_gather', 0),
			bispinor=bispinor,
			use_gspace_cache=use_gspace_cache,
			band_range_left=band_range_left,
			band_range_right=band_range_right,
			isdf_pair_mode=isdf_pair_mode,
			k_chunk_size=k_chunk_size,
		)
	
	# Step 2: Compute V_qmunu from zeta
	# Ensure filesystem is flushed before reading
	if jax.process_index() == 0:
		os.sync()
	jax.experimental.multihost_utils.sync_global_devices("zeta_flush")
	
	bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
	cell_volume = float(wfn.cell_volume)
	
	# V_q memory: zeta_mu(G) + zeta_nu(G) + FFT workspace + V_q block
	# Use memory headroom reported by chunk sizing (centroids remain resident).
	n_G = meta.n_rtot
	bytes_per_complex = 16
	model_budget_vcoul_gb = float(mem_est.get('available_vcoul_gb', memory_budget_gb))
	runtime_budget_vcoul_gb = model_budget_vcoul_gb
	try:
		from common.gpu_utils import get_device_memory_info
		mem_info_now = get_device_memory_info()
		runtime_budget_vcoul_gb = min(
			model_budget_vcoul_gb,
			float(mem_info_now.get('budget_gb', model_budget_vcoul_gb)),
		)
	except Exception:
		pass
	m_budget_vcoul = max(0.1, runtime_budget_vcoul_gb) * 1e9
	# Each mu needs: 2 × n_G × 16 (zeta_mu + zeta_nu for off-diag) + 1 × n_G × 16 (FFT workspace)
	m_per_mu = 3 * bytes_per_complex * n_G
	mu_chunk_vcoul = max(1, min(meta.n_rmu, int(m_budget_vcoul / m_per_mu)))
	q_batch_vcoul = 1
	n_q_total = n_q
	if mu_chunk_vcoul >= meta.n_rmu and n_q_total > 1:
		# Single-chunk path may batch q-points; keep this memory-aware.
		bytes_per_q_batch = 2.0 * bytes_per_complex * meta.n_rmu * n_G
		q_batch_by_mem = max(1, int(m_budget_vcoul // max(1.0, bytes_per_q_batch)))
		q_batch_vcoul = max(1, min(4, n_q_total, q_batch_by_mem))
	print(f"    V_q budget:    {runtime_budget_vcoul_gb:.2f} GB")
	print(f"    V_q mu chunks: {mu_chunk_vcoul}")
	if q_batch_vcoul > 1:
		print(f"    V_q q batches: {q_batch_vcoul}")
	if bare_coulomb_cutoff is not None:
		print(f"    V_q bare cutoff: {bare_coulomb_cutoff:.1f} Ry")
	
	with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
		with h5py.File(zeta_h5_path, 'r') as zeta_h5:
			with mesh_xy:
				V_qmunu_raw, g0_mu_all = compute_all_V_q_from_zeta_h5(
					zeta_h5,
					kgrid=meta.kgrid,
					fft_grid=meta.fft_grid,
					bvec=bvec,
					cell_volume=cell_volume,
					mu_chunk_size=mu_chunk_vcoul,
					mesh_xy=mesh_xy,
					sys_dim=sys_dim,
					q_batch_size=q_batch_vcoul if mu_chunk_vcoul >= meta.n_rmu else None,
					bdot=np.asarray(wfn.bdot, dtype=np.float64) if sys_dim == 0 else None,
					mc_average_vcoul_body=mc_average_vcoul_body,
					bare_coulomb_cutoff=bare_coulomb_cutoff,
				)
	
	# Write G0 (ζ_μ(G=0) for each q) to the zeta HDF5 file for restart/reuse
	# g0_mu_all shape: (nqx, nqy, nqz, n_rmu)
	# g0_mu_all may be sharded across processes; gather before numpy conversion
	g0_mu_local = jax.experimental.multihost_utils.process_allgather(g0_mu_all)
	if g0_mu_local.ndim == 5 and g0_mu_local.shape[0] == 1:
		g0_mu_local = g0_mu_local[0]
	if jax.process_index() == 0:
		with h5py.File(zeta_h5_path, 'a') as f:
			g0_np = np.asarray(g0_mu_local)
			if 'g0_mu' in f:
				del f['g0_mu']  # Overwrite if exists
			f.create_dataset('g0_mu', data=g0_np)
			print(f"    G0 written to {zeta_h5_path} (shape: {g0_np.shape})")
	jax.experimental.multihost_utils.sync_global_devices("g0_write")
	
	# TODO: Compute and store S_qmunu = ⟨ζ_μ|ζ_ν⟩ overlap matrix during V_q computation.
	# This would be useful for Hartree potential and other applications.
	# S_q = zeta^H @ zeta, can be computed from zeta chunks as sum over q.
	
	# Reshape V_qmunu to match expected format: (1, npol, npol, nkx, nky, nkz, n_rmu, n_rmu)
	# Our V_qmunu_raw is (nkx, nky, nkz, n_rmu, n_rmu)
	V_qmunu = V_qmunu_raw[None, None, None, :, :, :, :, :]  # Add leading dims
	# Broadcast to (1, npol, npol, ...)
	V_qmunu = jnp.broadcast_to(V_qmunu, (1, meta.npol, meta.npol, nqx, nqy, nqz, meta.n_rmu, meta.n_rmu))
	V_qmunu = jnp.array(V_qmunu)  # Force copy to avoid broadcast issues
	
	# Extract v_q0 (q=0 with G=0 excluded) and G0 (ζ_μ at G=0)
	v_q0_noG0_munu = V_qmunu_raw[0, 0, 0, :, :]  # At q=(0,0,0)
	# G0 = ζ_μ(G=0) at q=0. g0_mu_local may be 3D or 4D depending on
	# how process_allgather handled the leading dimension. Extract q=(0,0,0).
	_g0 = g0_mu_local
	while _g0.ndim > 1:
		_g0 = _g0[0]
	G0_mu_nu = _g0  # (n_rmu,)
	
	print("\n  V_q computed:")
	print(f"    Shape: {V_qmunu.shape}")
	print(f"    V_q=0 trace: {jnp.trace(v_q0_noG0_munu).real:.4f}")
	
	return {
		'V_qmunu': V_qmunu,
		'v_q0_noG0_munu': v_q0_noG0_munu,
		'G0_mu_nu': G0_mu_nu,
		'psi_l_rmu_Y': psi_l_rmu_Y,
		'psi_l_rmuT_X': psi_l_rmuT_X,
		'psi_r_rmu_Y': psi_r_rmu_Y,
		'psi_r_rmuT_X': psi_r_rmuT_X,
		'zeta_h5_path': zeta_h5_path,
	}


# ================= JAX-sharded Sigma pipeline =================

def build_G_munu(psi_xn, psi_yr, Gij):
	"""Build the ISDF Green's function G_μν(k) = Σ_ij ψ*_ik(μ) G_ij(k) ψ_jk(ν).

	Args:
		psi_xn: (nk, s, μ_X, n) — bands fast, μ on X.  Conjugated internally.
		psi_yr: (nk, n, s, μ_Y) — centroids fast, μ on Y.  Conjugated internally.
		Gij:    (nk, n, n) — band-space Green's function / projector.

	Returns:
		G_k: (nk, s, μ_X, s, μ_Y) — ISDF Green's function, μ distributed (X, Y).

	Spins are NOT contracted; the output carries both spinor indices.
	Zero-communication contraction when μ_X and μ_Y live on different mesh axes.
	"""
	return jnp.einsum('ksxi,kij,kjty->ksxty', psi_xn, Gij, jnp.conj(psi_yr), optimize=True)

def build_G_munu_ri(psi_xn, psi_yr):
	"""Resolution-of-identity Green's function G_μν(k) = Σ_n ψ*_nk(μ) ψ_nk(ν).

	Sums over ALL bands with unit weight (no occupation projector).
	Used for the Coulomb-hole self-energy term.

	Args:
		psi_xn: (nk, s, μ_X, n) — bands fast, μ on X.
		psi_yr: (nk, n, s, μ_Y) — centroids fast, μ on Y.

	Returns:
		G_k: (nk, s, μ_X, s, μ_Y).
	"""
	return jnp.einsum('ksxn,knty->ksxty', psi_xn, jnp.conj(psi_yr), optimize=True)

# Backward-compatibility alias (used as callback in ppm_sigma)
get_G_mu_nu_jax = build_G_munu

def get_G_R_jax(G_k, nkx, nky, nkz, sharded_ifftn=None):
	"""(nk, s1,rmu1,s2,rmu2) -> (s1,rmu1,s2,rmu2,nkx,nky,nkz).

	If sharded_ifftn is provided, uses it for the FFT (shard_map, local).
	Otherwise falls back to bare jnp.fft (works on 1 GPU, OOMs on 16).
	"""
	G_k = G_k.transpose(1, 2, 3, 4, 0)
	G_k = G_k.reshape(*G_k.shape[:4], nkx, nky, nkz)
	G_k = jax.lax.with_sharding_constraint(G_k, P(None, 'x', None, 'y', None, None, None))
	if sharded_ifftn is not None:
		return sharded_ifftn(G_k)
	return jnp.fft.ifftn(G_k, axes=(4,5,6), norm='ortho')

def get_sigma_static_mu_nu_jax(G_R, V_mu_nu, nk_tot, bispinor=False,
                               sharded_ifftn=None, sharded_fftn=None):
	"""Compute sigma in (s1,rmu1,s2,rmu2,nkx,nky,nkz) basis via convolution.

	If sharded_ifftn/sharded_fftn provided, uses them for V and sigma FFTs.
	"""
	V_R = V_mu_nu[None, :, None, :, :, :, :]
	V_R = jnp.array(V_R, copy=True)

	if bispinor:
		gamma0_diag = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64)
		gamma0_vertex = gamma0_diag[:, None] * gamma0_diag[None, :]
		gamma0_vertex = gamma0_vertex[:, None, :, None, None, None, None]
		G_R = G_R * gamma0_vertex

	if sharded_ifftn is not None:
		V_R_fft = sharded_ifftn(V_R)
	else:
		V_R_fft = jnp.fft.ifftn(V_R, axes=(4,5,6), norm='ortho')

	sigma_R = G_R * V_R_fft * (-1.0 / jnp.sqrt(nk_tot))
	sigma_R = jax.lax.with_sharding_constraint(sigma_R, P(None, 'x', None, 'y', None, None, None))

	if sharded_fftn is not None:
		return sharded_fftn(sigma_R)
	return jnp.fft.fftn(sigma_R, axes=(4,5,6), norm='ortho')

def project_sigma_to_bands(psi_xr, psi_yn, sigma_k_munu):
	"""Project self-energy from ISDF (s,μ) basis to band basis.

	Σ_mn(k) = Σ_{s,t,μ,ν} ψ*_m(s,μ_X) Σ(s,μ_X,t,ν_Y) ψ_n(t,ν_Y)

	Both spinor indices (s, t) and both centroid indices (μ, ν) are contracted.
	The (s,μ) pairs are contiguous in memory for both psi_xr and psi_yn.

	Args:
		psi_xr: (nk, nb, s, μ_X) — centroids fast, μ on X.  Conjugated internally.
		psi_yn: (nk, s, μ_Y, nb) — bands fast, μ on Y.
		sigma_k_munu: (s, μ, s, μ, nkx, nky, nkz) — ISDF self-energy.

	Returns:
		sigma_kij: (nk, nb, nb) — band-space self-energy matrix.
	"""
	nkx, nky, nkz = sigma_k_munu.shape[-3:]
	nk = nkx * nky * nkz
	sigma_k = sigma_k_munu.transpose(4, 5, 6, 0, 1, 2, 3).reshape(nk, *sigma_k_munu.shape[:4])
	left = jnp.einsum('kmsx,ksxty->kmty', jnp.conj(psi_xr), sigma_k, optimize=True)
	return jnp.einsum('kmty,ktyn->kmn', left, psi_yn, optimize=True)



def get_sigma_static_kij_channels_jax(psi_sigX, psi_sigTY, sigma_k_munu):
	"""Project Re/Im sigma channels from (spinor,rmu) basis to band basis.

	For a complex sigma tensor X in the centroid basis, this returns the two
	band-space channels needed for exact window projection without storing
	Σ_k(μ,ν,ω):

	  channel 0: K[Re X]
	  channel 1: K[Im X]

	where K[.] denotes the band projection map. These two channels are
	sufficient to reconstruct K[Re(cX)] or K[Im(cX)] for any complex scalar c.

	Args:
		psi_sigX: (nk, nb, nspinor, rmu) wavefunctions
		psi_sigTY: (nk, nspinor, rmu, nb) transposed wavefunctions
		sigma_k_munu: (nspinor, rmu1, nspinor, rmu2, nkx, nky, nkz) self-energy

	Returns:
		sigma_kij_ri: (2, nk, nb, nb) with channels [K[Re X], K[Im X]]
	"""
	nkx, nky, nkz = sigma_k_munu.shape[-3:]
	nk = nkx * nky * nkz
	sigma_k = sigma_k_munu.transpose(4, 5, 6, 0, 1, 2, 3).reshape(nk, *sigma_k_munu.shape[:4]) # (nk,s1,rmu1,s2,rmu2)
	sigma_k_ri = jnp.stack(
		(
			jnp.real(sigma_k),
			jnp.imag(sigma_k),
		),
		axis=0,
	)
	left = jnp.einsum('kmsx,cksxty->ckmty', jnp.conj(psi_sigX), sigma_k_ri, optimize=True)
	return jnp.einsum('ckmty,ktyn->ckmn', left, psi_sigTY, optimize=True).astype(jnp.complex128)

# ================= Hartree =================

def build_density_from_Gij(psi_yr, Gij, nk_tot):
	"""Charge density at centroids: ρ_μ = (1/Nk) Σ_{k,ij,s} G_ij ψ*_i(s,μ) ψ_j(s,μ).

	Spinor index is contracted (spin-traced).  Uses psi_yr because μ
	appears twice — the contraction is local on Y.

	Args:
		psi_yr: (nk, n, s, μ_Y) — centroids fast, μ on Y.
		Gij:    (nk, n, n) — Green's function / occupation projector.
		nk_tot: total k-points for BZ averaging.

	Returns:
		rho_mu: (μ,) charge density at centroid positions.
	"""
	psi_ij = jnp.einsum('kisx,kjsx->kijx', jnp.conj(psi_yr), psi_yr, optimize=True)
	rho_mu = jnp.einsum('kij,kijx->x', Gij, psi_ij, optimize=True)
	rho_mu = jnp.real(rho_mu) / jnp.asarray(nk_tot, dtype=jnp.float64)
	return rho_mu

def build_hartree_potential(rho_mu, V0_munu):
	"""Hartree potential at centroids: [Vρ]_μ = Σ_ν V0(μ,ν) ρ_ν."""
	return jnp.einsum('xy,y->x', V0_munu, rho_mu, optimize=True)

def project_potential_to_bands(psi_xr, Vrho_mu):
	"""Project local potential to bands: ⟨m|V|n⟩ = Σ_{s,μ} ψ*_m(s,μ) V(μ) ψ_n(s,μ).

	Uses psi_xr (centroids fast, μ on X) — same copy as Σ projection LHS.

	Args:
		psi_xr: (nk, n, s, μ_X) — centroids fast, μ on X.
		Vrho_mu: (μ,) potential at centroids.

	Returns:
		V_kmn: (nk, nb, nb) potential matrix elements.
	"""
	return jnp.einsum('kmsx,x,knsx->kmn', jnp.conj(psi_xr), Vrho_mu, psi_xr, optimize=True)

def print_scf_diagnostics(Gij_final, U_full, nelec, nb_sigma, print_fn=print):
	"""Print diagnostic checks for SC-COHSEX convergence (Gij, U unitarity)."""
	Gij_trace = jnp.real(jnp.trace(Gij_final[0]))
	Gij_diag = jnp.real(jnp.diagonal(Gij_final[0]))
	print_fn(f"[Diagnostic] Gij_final trace at k=0: {float(Gij_trace):.4f} (should be {nelec})")
	print_fn(f"[Diagnostic] Gij_final diag[:5] at k=0: {np.array(Gij_diag[:5])}")
	print_fn(f"[Diagnostic] Gij_final diag sum at k=0: {float(jnp.sum(Gij_diag)):.4f}")

	UdagU = jnp.einsum('kim,kin->kmn', jnp.conj(U_full[0:1]), U_full[0:1])
	unitarity_err = jnp.max(jnp.abs(UdagU[0] - jnp.eye(nb_sigma)))
	print_fn(f"[Diagnostic] U unitarity error at k=0: {float(unitarity_err):.2e} (should be ~0)")

	U_diag = jnp.abs(jnp.diagonal(U_full[0]))
	print_fn(f"[Diagnostic] |U| diagonal[:5] at k=0: {np.array(U_diag[:5])} (should be ~1 if no mixing)")
	print_fn(f"[Diagnostic] |U| diagonal[25:30] at k=0: {np.array(U_diag[25:30])} (valence-cond boundary)")

	U_col0_abs = jnp.abs(U_full[0, :, 0])
	top_contrib = jnp.argsort(U_col0_abs)[::-1][:5]
	print_fn(f"[Diagnostic] Lowest QP state: top DFT contributors = {np.array(top_contrib)}")
	print_fn(f"[Diagnostic] Lowest QP state: their |U| values = {np.array(U_col0_abs[top_contrib])}")


def main(argv=None):
	global sym
	argp = argparse.ArgumentParser(description="COHSEX self-energy driver")
	argp.add_argument(
		"-i",
		"--input",
		default="cohsex_test.in",
		help="Input file",
	)
	args = argp.parse_args(argv)
	
	# Gate prints to rank 0 during main
	_orig_print = builtins.print
	def _print0(*a, **k):
		if jax.process_index() == 0:
			k.setdefault("flush", True)
			_orig_print(*a, **k)
	builtins.print = _print0
 
	params = read_cohsex_input(args.input)
	
	# ========================================================================
	# INITIALIZATION - Build device mesh early for banner
	# ========================================================================
	current_backend = jax.default_backend()
	n_devices = len(jax.devices())
	n_procs = jax.process_count()
	device_names = jax.devices()[0].device_kind if n_devices > 0 else "unknown"
	
	# Construct device mesh early so we can report it in the banner
	total_devices = jax.process_count() * jax.local_device_count()
	grid_x = int(np.sqrt(total_devices))
	while total_devices % grid_x != 0:
		grid_x -= 1
	grid_y = total_devices // grid_x
	devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
	mesh_xy = Mesh(devices_2d, ['x', 'y'])
	
	_print0("")
	_print0("=" * 72)
	_print0("  COHSEX-JAX: Self-Energy Calculation")
	_print0("=" * 72)
	_print0(f"  Backend: {current_backend.upper():<8}  Devices: {n_devices}  Mesh: {grid_x}×{grid_y}  Processes: {n_procs}")
	_print0(f"  Device type: {device_names}")
	# XLA allocator diagnostics
	_preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "unset")
	_mem_frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "unset")
	_print0(f"  XLA preallocate: {_preallocate}  mem_fraction: {_mem_frac}")
	try:
		_stats = jax.devices()[0].memory_stats()
		if _stats:
			_bl = _stats.get('bytes_limit', 0) / 1e9
			_bu = _stats.get('bytes_in_use', 0) / 1e9
			_print0(f"  XLA pool: limit={_bl:.2f} GB, in_use={_bu:.2f} GB, avail={_bl-_bu:.2f} GB")
	except Exception:
		pass
	_print0("=" * 72)
	_print0("")
	
	# Resolve relative paths against the input file's directory
	input_dir = os.path.dirname(os.path.abspath(args.input))
	resolve_input_paths(params, input_dir)
	nval = params["nval"]
	ncond = params["ncond"]
	nband = params["nband"]
	sys_dim = params["sys_dim"]  # 0=molecule/box, 2=slab
	self_consistent = bool(params.get("self_consistent", False))
	use_ppm_sigma = bool(params.get("use_ppm_sigma", False))

	ryd2ev = 13.6056980659

	import time as _boot_time
	_t_boot = _boot_time.perf_counter()
	_print0(f"  [TIMING-BOOT] input parsed: {_boot_time.perf_counter() - _t_boot:.3f}s")

	global wfn
	_t_wfn = _boot_time.perf_counter()
	wfn = WFNReader(params["wfn_file"])
	_print0(f"  [TIMING-BOOT] WFNReader: {_boot_time.perf_counter() - _t_wfn:.3f}s")
	_t_sym = _boot_time.perf_counter()
	sym = symmetry_maps.SymMaps(wfn)
	_print0(f"  [TIMING-BOOT] SymMaps: {_boot_time.perf_counter() - _t_sym:.3f}s")
	
	# Load centroids
	_t_cent = _boot_time.perf_counter()
	centroids_frac, centroid_indices, _n_rmu = load_centroids(params["centroids_file"], wfn.fft_grid)
	_print0(f"  [TIMING-BOOT] load_centroids: {_boot_time.perf_counter() - _t_cent:.3f}s")
	# Resolve tmp_dir and output path relative to input file directory
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{_n_rmu}.h5")
	_print0(f"  ISDF basis: {_n_rmu} centroids")
	_print0(f"  FFT grid: {wfn.fft_grid[0]}×{wfn.fft_grid[1]}×{wfn.fft_grid[2]}   Cell volume: {wfn.cell_volume:.2f} a.u.³")
	_print0("")

	# windows for polarizability and sigma
	epsq = 0.01

	# Resolve runtime configuration (memory budget, chunking, control flags)
	_t_cfg = _boot_time.perf_counter()
	cfg = resolve_runtime_config(params, rank=jax.process_index())
	_print0(f"  [TIMING-BOOT] runtime_config: {_boot_time.perf_counter() - _t_cfg:.3f}s")
	global bispinor
	bispinor = cfg.bispinor
	do_screened = cfg.do_screened
	do_G0 = cfg.do_G0

	meta = Meta.from_system(wfn, sym, nval, ncond, nband, _n_rmu, bispinor)
	meta.rank = jax.process_index()
	meta.n_proc = jax.process_count()
	meta.sys_dim = sys_dim
	meta.bispinor = bispinor
	meta.chunk_size = get_effective_chunk_size(params["chunk_size"])

	b0, b1, b2, b3, b4 = meta.band_edges
	band_slices = BandSlices.from_band_edges(b0, b1, b2, b3, b4)

	def print0(*a, **k):
		if meta.rank == 0:
			k.setdefault("flush", True)
			print(*a, **k)

	sh = make_shardings(mesh_xy)

	chunk_str = "disabled" if meta.chunk_size is None else str(meta.chunk_size)
	print0(f"  Band chunk size: {chunk_str}")

	# ISDF fitting or restart loading
	timing.reset()
	_t_isdf = _boot_time.perf_counter()
	isdf = prepare_isdf_and_wavefunctions(
		cfg=cfg,
		wfn=wfn,
		sym=sym,
		meta=meta,
		centroid_indices=centroid_indices,
		band_slices=band_slices,
		mesh_xy=mesh_xy,
		sh=sh,
		tmp_dir=tmp_dir,
		tensors_filename=tensors_filename,
		print0=print0,
	)
	print0(f"  [TIMING-BOOT] prepare_isdf_and_wavefunctions: {_boot_time.perf_counter() - _t_isdf:.3f}s")
	V_qmunu = isdf.V_qmunu
	V_qmunu_nohead = V_qmunu
	v_q0_noG0_munu = isdf.v_q0_noG0_munu
	# Retained in restart payloads for diagnostics / external checks, but no longer
	# used in the active q=0 head treatment.
	_ = isdf.G0_mu_nu
	wfns = isdf.wf_bundle
	s = wfns.slices
	# Legacy alias kept until all sigma_views references are removed
	sigma_views = isdf.sigma_views

	# Compute screened Coulomb W = (1 - Vχ)⁻¹ V
	if do_screened:
			with timing.section("gw_jax.chi0_W"):
				with jax_profile.trace_section("chi0_W"):
					minimax_config = minimax_config_from_params(params)
					screening_setup = build_screening_setup(params, minimax_config)
					window_pairs = maybe_build_ctsp_windows(
						screening_setup,
						epsq=epsq,
						wfn=wfn,
						nband=nband,
						window_builder=get_window_info,
					)
					# In the normal minimax + PPM-sigma workflow, screening only needs
					# the static W(q). The dynamic W(iωp) / GN-PPM fit is recomputed in
					# ppm_sigma.py, so skip the redundant imag-frequency continuation here.
					screening_ppm_omega_p = screening_setup.ppm_omega_p
					if use_ppm_sigma and screening_setup.screening_method == "minimax":
						screening_ppm_omega_p = None
					W_q = compute_screening(
						V_qmunu, wfns, window_pairs, meta, mesh_xy,
						omega=screening_setup.omega_eval,
						screening_method=screening_setup.screening_method,
						minimax_config=screening_setup.minimax_config,
						ppm_omega_p=screening_ppm_omega_p,
						ppm_fallback_omega=screening_setup.ppm_fallback_omega,
						tensors_filename=tensors_filename,
						print0=print0,
					)

	# Extract bare (no-head) V_μν. This is kept for Hartree construction where
	# the G=0 contribution must remain excluded.
	V_mu_nu_nohead = jnp.asarray(V_qmunu)[0, 0, 0].transpose(3, 4, 0, 1, 2)
	with mesh_xy:
		V_mu_nu_nohead = jax.lax.with_sharding_constraint(V_mu_nu_nohead, sh.V_shard)
	static_head_terms = None

	# ============================================================================
	# q→0 head handling
	# ============================================================================
	# The active driver no longer injects the q=0 head into ISDF μν tensors via
	# the old rank-1 G0 projector. All physical head handling now happens through:
	#   - exact band-diagonal static terms for COHSEX
	#   - scalar head samples / fits for GN-PPM diagnostics and future dynamic use
	#
	# The saved G0_mu_nu restart vector remains useful for debugging and external
	# analysis, but it is no longer part of the active GWJAX self-energy path.
	# ============================================================================
	if do_G0 and not use_ppm_sigma:
		# For static COHSEX, add the q=0 head exactly in band space rather than
		# approximating it through any μν-space head injection.
		head_static = resolve_head_sample(
			params,
			input_dir,
			wfn,
			sym,
			meta,
			print0,
			omega=0.0 + 0.0j,
		)
		print0(format_head_sample_diagnostics(head_static, include_screened=do_screened))
		band_ids = np.arange(int(b0), int(b3), dtype=np.int32)
		occ_mask_static = band_ids < int(wfn.nelec)
		static_head_terms = compute_static_head_terms_from_sample(
			head_static,
			occ=occ_mask_static,
			cell_volume=float(wfn.cell_volume),
			nk_tot=int(meta.nk_tot),
		)
		print0(format_static_head_diagnostics(static_head_terms))

	# Canonical V tensors for downstream terms.
	# The ISDF body path remains headless; any q=0 head contribution is handled
	# separately in band space through the exact static terms above.
	V_mu_nu_exchange = V_mu_nu_nohead
	V_mu_nu_for_coh = V_mu_nu_nohead

	# Extract W_μν in (n_rmu, n_rmu, nkx, nky, nkz) layout.
	# For do_screened=false, SX should reduce to bare exchange with headed V.
	if do_screened:
		W_mu_nu = W_q[:,:,:,0,:,0,:].transpose(3, 4, 0, 1, 2)
		with mesh_xy:
			W_mu_nu = jax.lax.with_sharding_constraint(W_mu_nu, sh.V_shard)
	else:
		W_mu_nu = V_mu_nu_exchange

	# Static COHSEX Green's function: G_ij = δ_{ij} f_i (projector onto occupied)
	nb_sigma = int(b3 - b0)
	nk = meta.nk_tot
	nelec = int(wfn.nelec)
	Gij_static = jnp.zeros((nk, nb_sigma, nb_sigma), dtype=jnp.complex128)
	occ_diag = jnp.arange(min(nelec, nb_sigma))
	Gij_static = Gij_static.at[:, occ_diag, occ_diag].set(1.0 + 0.0j)
	Gij_shard = NamedSharding(mesh_xy, P(None, None, None))
	Gij_static = jax.device_put(Gij_static, Gij_shard)

	# Device-local FFT helpers for the replicated k-grid axes.
	from common.fft_helpers import (
		make_jittable_local_fftn_3d,
		make_jittable_local_ifftn_3d,
	)
	_G_spec = P(None, 'x', None, 'y', None, None, None)
	_local_ifftn_G = make_jittable_local_ifftn_3d(mesh_xy, _G_spec, _G_spec, norm='ortho', axes=(4, 5, 6))
	_local_fftn_G = make_jittable_local_fftn_3d(mesh_xy, _G_spec, _G_spec, norm='ortho', axes=(4, 5, 6))

	# NamedSharding objects for sigma (matching chi0 pattern — bare P() may
	# not resolve correctly on multi-node without explicit mesh reference)
	_G_7d_shard = NamedSharding(mesh_xy, P(None, 'x', None, 'y', None, None, None))

	# Static system constants closed over by JIT functions below.
	# These are compile-time constants — JAX folds them during tracing.
	_nkx, _nky, _nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
	_nk_tot = int(meta.nk_tot)
	_inv_sqrt_nk = -1.0 / jnp.sqrt(_nk_tot)

	@jax.jit
	def _G_to_7d(G_k):
		"""Reshape (nk, s, μ, s, μ) → (s, μ, s, μ, nkx, nky, nkz) for FFT."""
		G_7d = G_k.transpose(1,2,3,4,0).reshape(
			G_k.shape[1], G_k.shape[2], G_k.shape[3], G_k.shape[4], _nkx, _nky, _nkz)
		return jax.lax.with_sharding_constraint(G_7d, _G_7d_shard)

	@jax.jit
	def _convolve_R(G_R, W_or_V):
		"""Convolve G(R) with interaction in real space: Σ(R) = G(R) * V(R) / √Nk."""
		V_R = _local_ifftn_G(W_or_V[None, :, None, :, :, :, :])
		sigma_R = G_R * V_R * _inv_sqrt_nk
		return jax.lax.with_sharding_constraint(sigma_R, _G_7d_shard)

	@jax.jit
	def _sx_branch_jit(psi_xn_sig, psi_yr_sig, psi_xr_sig, psi_yn_sig, Gij, W):
		G_R = _local_ifftn_G(_G_to_7d(build_G_munu(psi_xn_sig, psi_yr_sig, Gij)))
		sigma_sx_k = _local_fftn_G(_convolve_R(G_R, W))
		return project_sigma_to_bands(psi_xr_sig, psi_yn_sig, sigma_sx_k)

	@jax.jit
	def _coh_branch_jit(psi_xn_full, psi_yr_full, psi_xr_sig, psi_yn_sig, W, V):
		G_RI_R = _local_ifftn_G(_G_to_7d(build_G_munu_ri(psi_xn_full, psi_yr_full)))
		sigma_coh_k = _local_fftn_G(_convolve_R(G_RI_R, W - V))
		return -0.5 * project_sigma_to_bands(psi_xr_sig, psi_yn_sig, sigma_coh_k)

	@jax.jit
	def _hartree_jit(psi_yr, Gij, V0, psi_xr):
		rho = build_density_from_Gij(psi_yr, Gij, _nk_tot)
		return project_potential_to_bands(psi_xr, build_hartree_potential(rho, V0))

	def static_sigma_pipeline(wfns, W, V, V0, Gij):
		"""Compute static COHSEX Σ_SX, Σ_COH, V_H from Wavefunctions bundle."""
		import gc as _gc
		_gc.collect()

		s = wfns.slices
		xn_sig, yr_sig = wfns.xn(s.sigma), wfns.yr(s.sigma)
		xr_sig, yn_sig = wfns.xr(s.sigma), wfns.yn(s.sigma)

		sigma_sx_kij = _sx_branch_jit(xn_sig, yr_sig, xr_sig, yn_sig, Gij, W)
		sigma_coh_kij = _coh_branch_jit(
			wfns.xn(s.full), wfns.yr(s.full), xr_sig, yn_sig, W, V)
		hartree_kmn = _hartree_jit(yr_sig, Gij, V0, xr_sig)

		return sigma_sx_kij, sigma_coh_kij, hartree_kmn

	def _apply_static_head_terms(sigma_sx_kij, sigma_coh_kij):
		if static_head_terms is None:
			return sigma_sx_kij, sigma_coh_kij
		sx_head, coh_head = static_head_terms_to_kij(
			static_head_terms,
			nk_tot=meta.nk_tot,
			do_screened=do_screened,
		)
		if not do_screened:
			coh_head = jnp.zeros_like(coh_head)
		sx_head = jax.device_put(sx_head, Gij_shard)
		coh_head = jax.device_put(coh_head, Gij_shard)
		return sigma_sx_kij + sx_head, sigma_coh_kij + coh_head

	fft_vol_au = float(wfn.cell_volume / np.prod(wfn.fft_grid))

	with mesh_xy:
		with timing.section("gw_jax.pipeline"):
			sigma_sx_kbar_ij_jax, sigma_coh_kbar_ij_jax, hartree_kbar_ij_jax = \
				static_sigma_pipeline(wfns, W_mu_nu, V_mu_nu_for_coh, v_q0_noG0_munu, Gij_static)
			sigma_sx_kbar_ij_jax, sigma_coh_kbar_ij_jax = _apply_static_head_terms(
				sigma_sx_kbar_ij_jax,
				sigma_coh_kbar_ij_jax,
			)
			sigma_sx_kbar_ij_jax.block_until_ready()
			sigma_coh_kbar_ij_jax.block_until_ready()
			hartree_kbar_ij_jax.block_until_ready()

	sigma_omega_h5_path = None
	sigma_munu_stream_path = None
	omega_grid_ev = None
	sigma_c_at_dft_ev = None
	sigma_xc_at_dft_ev = None
	sigma_c_plus_at_dft_ev = None
	sigma_c_minus_at_dft_ev = None
	sigma_c_invalid_static_diag_ev = None
	omega_dft_rel_ev = None
	efermi_dft_ev = None
	vbm_dft_ev = None
	cbm_dft_ev = None
	# Optional: replace static COH with GN-PPM frequency-integrated Sigma^c.
	if use_ppm_sigma:
		if not do_screened:
			raise ValueError("use_ppm_sigma=true requires do_screened=true.")
		if self_consistent:
			raise NotImplementedError("use_ppm_sigma is currently supported only for self_consistent=false.")
		sigma_sx_screened_ref = sigma_sx_kbar_ij_jax
		sigma_coh_static_ref = sigma_coh_kbar_ij_jax
		minimax_config = minimax_config_from_params(params)
		sigma_quadrature = sigma_quadrature_config_from_params(params)
		ppm_options = build_ppm_sigma_runtime_options(params, input_dir=input_dir, ryd2ev=ryd2ev)
		omega_p_ry = ppm_options.omega_p_ry
		ppm_fallback = ppm_options.ppm_fallback
		omega_grid_ev = ppm_options.omega_grid_ev
		omega_grid_ry = ppm_options.omega_grid_ry
		sigma_regularization_ry = ppm_options.sigma_regularization_ry
		sigma_edge_factor = ppm_options.sigma_edge_factor
		sigma_omega_batch_size = ppm_options.sigma_omega_batch_size
		sigma_omega_accumulation = ppm_options.sigma_omega_accumulation
		ppm_sigma_scale = ppm_options.ppm_sigma_scale
		ppm_sigma_flip_neg = ppm_options.ppm_sigma_flip_neg
		ppm_invalid_mode = ppm_options.ppm_invalid_mode
		sigma_debug_split_contrib = ppm_options.sigma_debug_split_contrib
		sigma_freq_debug_output = ppm_options.sigma_freq_debug_output
		fermi_reference = ppm_options.fermi_reference
		sigma_at_dft_extrapolate = ppm_options.sigma_at_dft_extrapolate
		sigma_at_dft_energies = ppm_options.sigma_at_dft_energies
		ppm_sigma_debug_static_norm = ppm_options.ppm_sigma_debug_static_norm
		ppm_static_cohsex_check = ppm_options.ppm_static_cohsex_check
		sigma_debug_quadrature = ppm_options.sigma_debug_quadrature
		sigma_debug_quadrature_samples = ppm_options.sigma_debug_quadrature_samples
		sigma_munu_h5_path = ppm_options.sigma_munu_h5_path
		sigma_kij_h5_path = ppm_options.sigma_kij_h5_path
		write_w_copies_debug = ppm_options.write_w_copies_debug
		w_copies_debug_file = ppm_options.w_copies_debug_file
		sigma_freq_debug_file = ppm_options.sigma_freq_debug_file
		if sigma_freq_debug_output and (not sigma_debug_split_contrib):
			sigma_debug_split_contrib = True
			print0("  NOTE: enabling sigma_debug_split_contrib for sigma_freq_debug_output")
		if sigma_freq_debug_output and sigma_omega_accumulation == "kij_stream":
			sigma_omega_accumulation = "kij"
			print0("  NOTE: forcing sigma_omega_accumulation='kij' for sigma_freq_debug_output")
		print0("")
		print0("-" * 72)
		print0("  GN-PPM + FREQUENCY-INTEGRATED SIGMA")
		print0("-" * 72)
		if abs(ppm_sigma_scale - 1.0) > 1.0e-12:
			print0(f"  NOTE: applying GN-PPM Σ^c scale factor = {ppm_sigma_scale:.6g}")
		if ppm_invalid_mode != "static_limit":
			print0(f"  NOTE: GN invalid-mode policy = {ppm_invalid_mode}")
		if fermi_reference not in ("vbm", "midgap"):
			raise ValueError("fermi_reference must be 'vbm' or 'midgap'.")
		if fermi_reference == "midgap":
			print0("  NOTE: using midgap reference for Σ^c windowing")
		if ppm_sigma_flip_neg:
			print0("  NOTE: debug: flipping sign of ω<E_F Σ^c branch")
		if do_G0:
			try:
				head_0 = resolve_head_sample(
					params, input_dir, wfn, sym, meta, print0, omega=0.0 + 0.0j
				)
				head_i = resolve_head_sample(
					params, input_dir, wfn, sym, meta, print0, omega=1j * float(omega_p_ry)
				)
				epsh0 = head_0.wcoul0 / head_0.vc0 if abs(head_0.vc0) > 1.0e-16 else complex(np.nan, np.nan)
				epshi = head_i.wcoul0 / head_i.vc0 if abs(head_i.vc0) > 1.0e-16 else complex(np.nan, np.nan)
				print0(
					f"  epsinv head (ω=0):      {epsh0.real: .6f}{epsh0.imag:+.6e}i"
					f"  [source={head_0.source}]"
				)
				print0(
					f"  epsinv head (ω=iωp):    {epshi.real: .6f}{epshi.imag:+.6e}i"
					f"  [ωp={omega_p_ry:.6f} Ry, source={head_i.source}]"
				)
			except Exception as exc:
				print0(f"  epsinv head diagnostics unavailable: {exc}")
		with timing.section("gw_jax.ppm_sigma"):
			import time as _ppm_time
			_ppm_t0 = _ppm_time.perf_counter()
			# Keep the q=0 head out of the ISDF body PPM fit. The scalar head is
			# better treated as a separate diagnostic channel than injected into
			# W^c before pole extraction.
			if do_G0:
				head_0 = resolve_head_sample(
					params, input_dir, wfn, sym, meta, print0, omega=0.0 + 0.0j
				)
				head_i = resolve_head_sample(
					params, input_dir, wfn, sym, meta, print0, omega=1j * float(omega_p_ry)
				)
				head_gn = fit_head_gn_from_samples(
					head_0,
					head_i,
					omega_p_ry=omega_p_ry,
				)
				print0(format_head_pair_diagnostics(head_0, head_i))
				print0(format_head_diagnostics(head_gn, float(wfn.cell_volume)))
			else:
				head_gn = None
			ppm = compute_w0_wiwp_and_ppm_from_minimax(
				V_qmunu_nohead,
				wfns,
				meta,
				mesh_xy,
				minimax_config=minimax_config,
				omega_p_ry=omega_p_ry,
				minimax_energy_reference=(
					minimax_config.energy_reference
					if minimax_config.energy_reference is not None else fermi_reference
				),
				fallback_omega=ppm_fallback,
				print0=print0,
			)
			_ppm_t1 = _ppm_time.perf_counter()
			print0(f"  [TIMING] PPM build: {_ppm_t1 - _ppm_t0:.1f}s")
			if meta.rank == 0 and write_w_copies_debug and w_copies_debug_file:
				write_w_copies_debug_h5(
					w_copies_debug_file,
					W0_screen_q=W_q,
					W0_ppm_q=ppm.W0_q,
					Wiwp_ppm_q=ppm.Wiwp_q,
					print_fn=print0,
				)
			_ppm_t2 = _ppm_time.perf_counter()
			print0(f"  [TIMING] W-debug write: {_ppm_t2 - _ppm_t1:.1f}s")

			def _build_sigma_x_bare_kij():
				with mesh_xy:
					sigma_x_bare_kij_local = _sx_branch_jit(
						wfns.xn(s.sigma), wfns.yr(s.sigma),
						wfns.xr(s.sigma), wfns.yn(s.sigma),
						Gij_static, V_mu_nu_exchange,
					)
				sigma_x_bare_kij_local.block_until_ready()
				return sigma_x_bare_kij_local

			sigma_omega = compute_sigma_c_ppm_omega_grid(
				psi_coh_rmuT_X=wfns.xn(s.full),
				psi_coh_rmu_Y=wfns.yr(s.full),
				psi_proj_rmu_X=wfns.xr(s.sigma),
				psi_proj_rmuT_Y=wfns.yn(s.sigma),
				enk_full=wfns.enk[:, s.full],
				occ_full=wfns.occ[:, s.full],
				B_mu_nu=ppm.B_mu_nu,
				Omega_mu_nu=ppm.Omega_mu_nu,
				Wc0_mu_nu=ppm.Wc0_mu_nu,
				valid_mask_mu_nu=ppm.valid_mask_mu_nu,
				omega_values_ry=omega_grid_ry,
				nkx=meta.nkx,
				nky=meta.nky,
				nkz=meta.nkz,
				nk_tot=meta.nk_tot,
				bispinor=bispinor,
				mesh_xy=mesh_xy,
				quadrature_config=sigma_quadrature,
				regularization_width_ry=sigma_regularization_ry,
				edge_factor=sigma_edge_factor,
				omega_batch_size=sigma_omega_batch_size,
				omega_accumulation=sigma_omega_accumulation,
				sigma_munu_h5_path=sigma_munu_h5_path or None,
				sigma_kij_h5_path=sigma_kij_h5_path or None,
				sigma_scale=ppm_sigma_scale,
				sigma_flip_neg=ppm_sigma_flip_neg,
				invalid_mode=ppm_invalid_mode,
				debug_split_contrib=sigma_debug_split_contrib,
				fermi_reference=fermi_reference,
				debug_quadrature=sigma_debug_quadrature,
				debug_quadrature_samples=sigma_debug_quadrature_samples,
				get_G_mu_nu_fn=get_G_mu_nu_jax,
				get_G_R_fn=lambda G_k, nkx, nky, nkz: get_G_R_jax(
					G_k,
					nkx,
					nky,
					nkz,
					sharded_ifftn=_local_ifftn_G,
				),
				get_sigma_mu_nu_fn=lambda G_R, V, nk, bisp=False: get_sigma_static_mu_nu_jax(
					G_R,
					V,
					nk,
					bisp,
					sharded_ifftn=_local_ifftn_G,
					sharded_fftn=_local_fftn_G,
				),
				get_sigma_kij_channels_fn=get_sigma_static_kij_channels_jax,
				ppm_static_cohsex_check=ppm_static_cohsex_check,
				print0=print0,
			)
			_ppm_t3 = _ppm_time.perf_counter()
			print0(f"  [TIMING] compute_sigma_c_ppm_omega_grid: {_ppm_t3 - _ppm_t2:.1f}s")
			sigma_coh_ppm_omega_kij = sigma_omega.sigma_c_kij
			iw0 = 0 if ppm_static_cohsex_check else int(np.argmin(np.abs(omega_grid_ry)))
			if sigma_coh_ppm_omega_kij is None:
				if not sigma_omega.sigma_kij_h5_path:
					raise RuntimeError("PPM Sigma stream requested but no sigma_kij_h5_path provided.")
				with h5py.File(sigma_omega.sigma_kij_h5_path, "r") as h5:
					sigma_coh_ppm_kij = jnp.asarray(h5["sigma_c_kij_ry"][iw0], dtype=jnp.complex128)
				print0("  NOTE: Σ_c(kij,ω) streamed to disk; skipping in-memory QSGW/diag-SC diagnostics.")
			else:
				sigma_coh_ppm_kij = sigma_coh_ppm_omega_kij[iw0]
				sigma_coh_ppm_kij.block_until_ready()

			sigma_x_bare_kij = _build_sigma_x_bare_kij()

			if ppm_static_cohsex_check:
				# With E=0 and omega=0, the PPM pipeline should give Sigma_cor = SX-X + COH,
				# NOT just COH. Sigma^c = Sigma_xc - Sigma_X = (SX + COH) - X = (SX-X) + COH.
				ryd2ev_local = 13.605693122994
				ppm_diag = np.real(np.diagonal(np.asarray(sigma_coh_ppm_kij), axis1=1, axis2=2)) * ryd2ev_local
				sx_diag = np.real(np.diagonal(np.asarray(sigma_sx_kbar_ij_jax), axis1=1, axis2=2)) * ryd2ev_local
				coh_diag = np.real(np.diagonal(np.asarray(sigma_coh_kbar_ij_jax), axis1=1, axis2=2)) * ryd2ev_local
				x_diag = np.real(np.diagonal(np.asarray(sigma_x_bare_kij), axis1=1, axis2=2)) * ryd2ev_local
				ref_cor_diag = (sx_diag - x_diag) + coh_diag
				diff_diag = ppm_diag - ref_cor_diag
				print0("  *** PPM STATIC COHSEX CHECK RESULTS (diagonal, eV) ***")
				print0(f"  max|PPM - Cor|    = {np.max(np.abs(diff_diag)):.6e} eV")
				print0(f"  mean|PPM - Cor|   = {np.mean(np.abs(diff_diag)):.6e} eV")
				nb_show = min(6, ppm_diag.shape[1])
				for ik in range(min(4, ppm_diag.shape[0])):
					n_occ_show = int(
						max(
							0,
							min(int(wfn.nelec) - int(params.get("band_index_min", 0)), ppm_diag.shape[1]),
						)
					)
					i_start = max(0, n_occ_show - nb_show // 2)
					i_end = min(ppm_diag.shape[1], i_start + nb_show)
					for ib in range(i_start, i_end):
						p = ppm_diag[ik, ib]
						r = ref_cor_diag[ik, ib]
						print0(f"    k={ik} n={ib}: PPM={p:10.4f}  Cor={r:10.4f}  diff={p-r:10.4f}")

			if ppm_sigma_debug_static_norm:
				# Static-COH normalization check using W^c(0) from PPM: W^c(0) = -2 B / Omega.
				with mesh_xy:
					Wc0_mu_nu = jnp.where(
						ppm.Omega_mu_nu != 0.0,
						(-2.0 * ppm.B_mu_nu) / ppm.Omega_mu_nu,
						jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
					)
					Wc0_mu_nu = jax.lax.with_sharding_constraint(Wc0_mu_nu, sh.V_shard)
					sigma_coh_ppm_w0_kij = _coh_branch_jit(
						wfns.xn(s.full), wfns.yr(s.full),
						wfns.xr(s.sigma), wfns.yn(s.sigma),
						Wc0_mu_nu + V_mu_nu_for_coh, V_mu_nu_for_coh,
					)
				sigma_coh_ppm_w0_kij.block_until_ready()
				diff_w0_abs = float(jnp.max(jnp.abs(sigma_coh_ppm_w0_kij - sigma_coh_kbar_ij_jax)))
				ref_w0 = max(float(jnp.max(jnp.abs(sigma_coh_kbar_ij_jax))), 1.0e-16)
				diff_dyn_abs = float(jnp.max(jnp.abs(sigma_coh_ppm_kij - sigma_coh_ppm_w0_kij)))
				ref_dyn = max(float(jnp.max(jnp.abs(sigma_coh_ppm_w0_kij))), 1.0e-16)
				print0(
					f"  PPM static-COH check (Wc0): abs={diff_w0_abs:.6e}, rel={diff_w0_abs / ref_w0:.6e}"
				)
				print0(
					f"  PPM dynamic vs Wc0 COH: abs={diff_dyn_abs:.6e}, rel={diff_dyn_abs / ref_dyn:.6e}"
				)
				w0_diag = np.real(np.diagonal(np.array(sigma_coh_ppm_w0_kij), axis1=1, axis2=2))
				dyn_diag = np.real(np.diagonal(np.array(sigma_coh_ppm_kij), axis1=1, axis2=2))
				mask = np.abs(w0_diag) > 1.0e-8
				if np.any(mask):
					ratio = dyn_diag[mask] / w0_diag[mask]
					diff_dyn_diag = np.max(np.abs(dyn_diag - w0_diag))
					ref_dyn_diag = max(float(np.max(np.abs(w0_diag))), 1.0e-16)
					print0(
						f"  PPM dyn/Wc0 COH diag ratio: median={np.median(ratio):.3f}, mean={np.mean(ratio):.3f}"
					)
					print0(
						f"  PPM dynamic vs Wc0 COH (diag): abs={diff_dyn_diag:.6e}, rel={diff_dyn_diag / ref_dyn_diag:.6e}"
					)

			if meta.rank == 0:
				# Evaluate Sigma_c(E_DFT) and Sigma_xc(E_DFT) for BGW comparisons and eqp_g0w0 output.
				energies_full_dft, _ = get_enk_bandrange(
					wfn, sym, (b0, b3), (b0, b3),
					nspinor=meta.nspinor,
				)
				energies_dft_ev = np.asarray(energies_full_dft) * ryd2ev
				# Derive the occupied count directly from the sigma-window band edges
				# instead of relying on an auxiliary occupancy array. This keeps the
				# E_DFT interpolation reference aligned with the local sigma window.
				n_occ_local = int(max(0, min(b2 - b0, energies_dft_ev.shape[1])))
				if n_occ_local > 0:
					vbm_ev = float(np.max(energies_dft_ev[:, :n_occ_local]))
				else:
					vbm_ev = float(np.max(energies_dft_ev))
				if fermi_reference == "midgap" and n_occ_local < energies_dft_ev.shape[1]:
					cbm_ev = float(np.min(energies_dft_ev[:, n_occ_local:]))
					efermi_dft_ev = 0.5 * (vbm_ev + cbm_ev)
				else:
					cbm_ev = None
					efermi_dft_ev = vbm_ev
				vbm_dft_ev = vbm_ev
				cbm_dft_ev = cbm_ev
				omega_dft_rel_ev = energies_dft_ev - efermi_dft_ev
				if fermi_reference == "midgap" and cbm_dft_ev is not None:
					print0(
						f"  Sigma(E_DFT) reference: E_F(midgap)={efermi_dft_ev:.6f} eV "
						f"from VBM={vbm_dft_ev:.6f} eV, CBM={cbm_dft_ev:.6f} eV"
					)
				else:
					print0(
						f"  Sigma(E_DFT) reference: E_F(VBM)={efermi_dft_ev:.6f} eV "
						f"from VBM={vbm_dft_ev:.6f} eV"
					)

				def _interp_complex_on_grid(omega_ev, values_omega, x_ev):
					if (x_ev < float(omega_ev[0]) or x_ev > float(omega_ev[-1])) and (not sigma_at_dft_extrapolate):
						return complex(np.nan, np.nan)
					xr = float(np.clip(x_ev, float(omega_ev[0]), float(omega_ev[-1])))
					v_re = np.interp(xr, omega_ev, np.real(values_omega))
					v_im = np.interp(xr, omega_ev, np.imag(values_omega))
					return complex(v_re, v_im)

				if sigma_coh_ppm_omega_kij is None:
					if not sigma_omega.sigma_kij_h5_path:
						raise RuntimeError("Σ_c(ω) is streamed without sigma_kij_h5_path; cannot evaluate Sigma_c(E_DFT).")
					with h5py.File(sigma_omega.sigma_kij_h5_path, "r") as h5:
						dset_c = h5["sigma_c_kij_ry"]
						n_omega = int(dset_c.shape[0])
						nk = int(dset_c.shape[1])
						nb = int(dset_c.shape[2])
						sigma_c_diag_omega_ev = np.zeros((n_omega, nk, nb), dtype=np.complex128)
						o_chunks = max(1, min(sigma_omega_batch_size, n_omega))
						for ibeg in range(0, n_omega, o_chunks):
							iend = min(ibeg + o_chunks, n_omega)
							block = np.asarray(dset_c[ibeg:iend], dtype=np.complex128) * ryd2ev
							sigma_c_diag_omega_ev[ibeg:iend] = np.diagonal(block, axis1=2, axis2=3)
				else:
					sigma_c_diag_omega_ev = np.diagonal(
						np.asarray(sigma_coh_ppm_omega_kij, dtype=np.complex128), axis1=2, axis2=3
					) * ryd2ev

				nk = sigma_c_diag_omega_ev.shape[1]
				nb = sigma_c_diag_omega_ev.shape[2]
				sigma_c_at_dft_ev = np.zeros((nk, nb), dtype=np.complex128)
				for ik in range(nk):
					for ib in range(nb):
						sigma_c_at_dft_ev[ik, ib] = _interp_complex_on_grid(
							omega_grid_ev, sigma_c_diag_omega_ev[:, ik, ib], omega_dft_rel_ev[ik, ib]
						)
				sigma_c_plus_at_dft_ev = np.full((nk, nb), np.nan + 1j * np.nan, dtype=np.complex128)
				sigma_c_minus_at_dft_ev = np.full((nk, nb), np.nan + 1j * np.nan, dtype=np.complex128)
				if sigma_omega.sigma_c_plus_kij is not None:
					sigma_c_plus_diag_omega_ev = np.diagonal(
						np.asarray(sigma_omega.sigma_c_plus_kij, dtype=np.complex128), axis1=2, axis2=3
					) * ryd2ev
					for ik in range(nk):
						for ib in range(nb):
							sigma_c_plus_at_dft_ev[ik, ib] = _interp_complex_on_grid(
								omega_grid_ev, sigma_c_plus_diag_omega_ev[:, ik, ib], omega_dft_rel_ev[ik, ib]
							)
				if sigma_omega.sigma_c_minus_kij is not None:
					sigma_c_minus_diag_omega_ev = np.diagonal(
						np.asarray(sigma_omega.sigma_c_minus_kij, dtype=np.complex128), axis1=2, axis2=3
					) * ryd2ev
					for ik in range(nk):
						for ib in range(nb):
							sigma_c_minus_at_dft_ev[ik, ib] = _interp_complex_on_grid(
								omega_grid_ev, sigma_c_minus_diag_omega_ev[:, ik, ib], omega_dft_rel_ev[ik, ib]
							)
				if sigma_omega.sigma_c_invalid_static_kij is not None:
					sigma_c_invalid_static_diag_ev = np.diagonal(
						np.asarray(sigma_omega.sigma_c_invalid_static_kij, dtype=np.complex128), axis1=1, axis2=2
					) * ryd2ev
				else:
					sigma_c_invalid_static_diag_ev = np.zeros((nk, nb), dtype=np.complex128)
				sigma_x_diag_ev = np.diagonal(np.asarray(sigma_x_bare_kij), axis1=1, axis2=2) * ryd2ev
				sigma_xc_at_dft_ev = sigma_x_diag_ev + sigma_c_at_dft_ev

			coh_diff_abs = float(jnp.max(jnp.abs(sigma_coh_ppm_kij - sigma_coh_kbar_ij_jax)))
			coh_ref = max(float(jnp.max(jnp.abs(sigma_coh_kbar_ij_jax))), 1.0e-16)
			sx_diff_abs = float(jnp.max(jnp.abs(sigma_x_bare_kij - sigma_sx_screened_ref)))
			sx_ref = max(float(jnp.max(jnp.abs(sigma_sx_screened_ref))), 1.0e-16)
			static_total_ref = sigma_sx_screened_ref + sigma_coh_static_ref
			gw_total_0 = sigma_x_bare_kij + sigma_coh_ppm_kij
			total_diff_abs = float(jnp.max(jnp.abs(gw_total_0 - static_total_ref)))
			total_ref = max(float(jnp.max(jnp.abs(static_total_ref))), 1.0e-16)
			print0(f"  Replacing static COH with PPM-integrated Σ^c(ω={omega_grid_ry[iw0]:.6f} Ry)")
			print0(f"  Σ^c difference vs static COH: abs={coh_diff_abs:.6e}, rel={coh_diff_abs / coh_ref:.6e}")
			print0(f"  Bare Σ^X vs screened Σ^SX: abs={sx_diff_abs:.6e}, rel={sx_diff_abs / sx_ref:.6e}")
			print0(
				f"  [Σ^X + Σ^c(0)] vs [Σ^SX + Σ^COH]: abs={total_diff_abs:.6e} Ry "
				f"({total_diff_abs * ryd2ev:.6e} eV), rel={total_diff_abs / total_ref:.6e}"
			)
			sigma_sx_kbar_ij_jax = sigma_x_bare_kij
			sigma_coh_kbar_ij_jax = sigma_coh_ppm_kij

			sigma_omega_h5_path = params.get("sigma_omega_h5_file", "sigma_mnk.h5")
			if not os.path.isabs(sigma_omega_h5_path):
				sigma_omega_h5_path = os.path.join(input_dir, sigma_omega_h5_path)

			if sigma_coh_ppm_omega_kij is not None:
				if meta.rank == 0:
					write_sigma_omega_h5(
						sigma_omega_h5_path,
						omega_grid_ev,
						None,
						sigma_c_kij_ev=ryd2ev * sigma_coh_ppm_omega_kij,
						sigma_sx_kij_ev=ryd2ev * sigma_sx_kbar_ij_jax,
						hartree_kij_ev=ryd2ev * hartree_kbar_ij_jax,
					)
					if (
						sigma_omega.sigma_c_plus_kij is not None
						or sigma_omega.sigma_c_minus_kij is not None
						or sigma_omega.sigma_c_invalid_static_kij is not None
					):
						if sigma_omega.sigma_c_plus_kij is not None:
							write_chunked_complex_dataset_h5(
								sigma_omega_h5_path,
								"sigma_c_plus_kij_ev",
								ryd2ev * sigma_omega.sigma_c_plus_kij,
								mode="a",
							)
						if sigma_omega.sigma_c_minus_kij is not None:
							write_chunked_complex_dataset_h5(
								sigma_omega_h5_path,
								"sigma_c_minus_kij_ev",
								ryd2ev * sigma_omega.sigma_c_minus_kij,
								mode="a",
							)
						if sigma_omega.sigma_c_invalid_static_kij is not None:
							write_chunked_complex_dataset_h5(
								sigma_omega_h5_path,
								"sigma_c_invalid_static_kij_ev",
								ryd2ev * sigma_omega.sigma_c_invalid_static_kij,
								mode="a",
							)
					# Recompute Sigma_c(E_DFT) from the written sigma_c_kij_ev payload so
					# eqp0_noqsym_w.dat and sigma_mnk.h5 cannot diverge.
					if sigma_c_at_dft_ev is not None:
						with h5py.File(sigma_omega_h5_path, "a") as h5:
							if "sigma_c_kij_ev" not in h5:
								raise RuntimeError(
									f"{sigma_omega_h5_path} missing sigma_c_kij_ev; cannot evaluate Sigma_c(E_DFT)."
								)
							sigma_c_diag_omega_ev_file = np.diagonal(
								np.asarray(h5["sigma_c_kij_ev"], dtype=np.complex128), axis1=2, axis2=3
							)
							nk = sigma_c_diag_omega_ev_file.shape[1]
							nb = sigma_c_diag_omega_ev_file.shape[2]
							sigma_c_at_dft_from_file = np.zeros((nk, nb), dtype=np.complex128)
							for ik in range(nk):
								for ib in range(nb):
									sigma_c_at_dft_from_file[ik, ib] = _interp_complex_on_grid(
										omega_grid_ev,
										sigma_c_diag_omega_ev_file[:, ik, ib],
										omega_dft_rel_ev[ik, ib],
									)
							sigma_x_diag_ev = np.diagonal(np.asarray(sigma_x_bare_kij), axis1=1, axis2=2) * ryd2ev
							sigma_c_at_dft_ev = sigma_c_at_dft_from_file
							sigma_xc_at_dft_ev = sigma_x_diag_ev + sigma_c_at_dft_ev
							if "omega_dft_rel_ev" in h5:
								del h5["omega_dft_rel_ev"]
							if "sigma_c_at_dft_ev" in h5:
								del h5["sigma_c_at_dft_ev"]
							if "sigma_xc_at_dft_ev" in h5:
								del h5["sigma_xc_at_dft_ev"]
							h5.create_dataset("omega_dft_rel_ev", data=np.asarray(omega_dft_rel_ev, dtype=np.float64))
							h5.create_dataset("sigma_c_at_dft_ev", data=np.asarray(sigma_c_at_dft_ev, dtype=np.complex128))
							h5.create_dataset("sigma_xc_at_dft_ev", data=np.asarray(sigma_xc_at_dft_ev, dtype=np.complex128))
							h5.attrs["sigma_at_dft_energies"] = True
							h5.attrs["fermi_reference"] = str(fermi_reference)
							h5.attrs["efermi_dft_ev"] = float(efermi_dft_ev)
							h5.attrs["vbm_dft_ev"] = float(vbm_dft_ev)
							h5.attrs["cbm_dft_ev"] = float(cbm_dft_ev) if cbm_dft_ev is not None else np.nan
						in_grid = (
							(np.asarray(omega_dft_rel_ev) >= float(np.min(omega_grid_ev)))
							& (np.asarray(omega_dft_rel_ev) <= float(np.max(omega_grid_ev)))
						)
						n_in = int(np.count_nonzero(in_grid))
						n_tot = int(in_grid.size)
						print0(
							f"  Sigma(E_DFT) in-grid states: {n_in}/{n_tot} "
							f"within [{float(np.min(omega_grid_ev)):.3f}, {float(np.max(omega_grid_ev)):.3f}] eV"
						)
			elif sigma_omega.sigma_kij_h5_path and meta.rank == 0:
				# Stream Σ_c(kij,ω) from disk to avoid holding full Nω in memory.
				with h5py.File(sigma_omega.sigma_kij_h5_path, "r") as h5_in:
					dset_c = h5_in["sigma_c_kij_ry"]
					n_omega = int(dset_c.shape[0])
					nk = int(dset_c.shape[1])
					nb = int(dset_c.shape[2])
					o_chunks = max(1, min(sigma_omega_batch_size, n_omega))
					chunks = (o_chunks, max(1, min(4, nk)), nb, nb)
					with h5py.File(sigma_omega_h5_path, "w") as h5_out:
						h5_out.create_dataset("omega_ev", data=np.asarray(omega_grid_ev, dtype=np.float64))
						dset_total = h5_out.create_dataset(
							"sigma_total_kij_ev",
							shape=dset_c.shape,
							dtype=np.complex128,
							chunks=chunks,
						)
						dset_c_ev = h5_out.create_dataset(
							"sigma_c_kij_ev",
							shape=dset_c.shape,
							dtype=np.complex128,
							chunks=chunks,
						)
						h5_out.create_dataset("sigma_sx_kij_ev", data=ryd2ev * np.array(sigma_sx_kbar_ij_jax))
						h5_out.create_dataset("hartree_kij_ev", data=ryd2ev * np.array(hartree_kbar_ij_jax))
						for ibeg in range(0, n_omega, o_chunks):
							iend = min(ibeg + o_chunks, n_omega)
							idx = slice(ibeg, iend)
							sigma_c_ev = ryd2ev * np.array(dset_c[idx], dtype=np.complex128)
							dset_c_ev[idx] = sigma_c_ev
							sigma_total_ev = (
								sigma_c_ev
								+ ryd2ev * np.array(sigma_sx_kbar_ij_jax)[None, ...]
								+ ryd2ev * np.array(hartree_kbar_ij_jax)[None, ...]
							)
							dset_total[idx] = sigma_total_ev

			if sigma_omega.sigma_munu_h5_path:
				sigma_munu_stream_path = sigma_omega.sigma_munu_h5_path
				with h5py.File(sigma_omega_h5_path, "a") as h5:
					h5.attrs["sigma_munu_h5_path"] = sigma_omega.sigma_munu_h5_path
				print0(f"  Streamed Σc(μ,ν,ω):      {sigma_omega.sigma_munu_h5_path}")


	# Initial Σ = SX + COH + Hartree (all three combined for self-consistency)
	sigma_total_full = sigma_sx_kbar_ij_jax + sigma_coh_kbar_ij_jax + hartree_kbar_ij_jax
	
	# Save INITIAL sigma for one-shot diagnostic (before self-consistency changes them)
	sigma_sx_initial = sigma_sx_kbar_ij_jax
	sigma_coh_initial = sigma_coh_kbar_ij_jax
	hartree_initial = hartree_kbar_ij_jax

	qp_band_start = int(b0)
	qp_band_stop = int(b3)
	kin_ion_path = params["kin_ion_file"]
	kin_ion_full = load_kin_ion_submatrix(kin_ion_path, qp_band_start, qp_band_stop)
	# NOTE: kin_ion stores H_DFT - V_xc (and typically excludes V_H unless kin_ion_io was run with --do-hartree).
	# This keeps the QP Hamiltonian in the standard GW form:
	#   H_QP = (H_DFT - V_xc) + V_H + Sigma_xc(omega).
	nelec = int(wfn.nelec)

	if use_ppm_sigma and meta.rank == 0 and sigma_omega_h5_path and os.path.exists(sigma_omega_h5_path):
		def _gap_k0(E_kn_ev, iv, ic):
			if iv < 0 or ic < 0:
				return None
			if E_kn_ev.shape[1] <= max(iv, ic):
				return None
			return float(E_kn_ev[0, ic] - E_kn_ev[0, iv])

		# Compare static COHSEX and dynamic-omega=0 spectra on the same Hamiltonian baseline.
		H_static_ref = np.array(kin_ion_full + sigma_sx_screened_ref + sigma_coh_static_ref + hartree_kbar_ij_jax)
		H_static_ref = 0.5 * (H_static_ref + np.conj(np.swapaxes(H_static_ref, -1, -2)))
		E_static_ref_ev = np.linalg.eigvalsh(H_static_ref) * ryd2ev

		H_dyn0 = np.array(kin_ion_full + sigma_sx_kbar_ij_jax + sigma_coh_kbar_ij_jax + hartree_kbar_ij_jax)
		H_dyn0 = 0.5 * (H_dyn0 + np.conj(np.swapaxes(H_dyn0, -1, -2)))
		E_dyn0_ev = np.linalg.eigvalsh(H_dyn0) * ryd2ev

		# Diagonal self-consistency for E = diag(kin_ion + V_H) + Re Sigma_xc(E).
		# Sigma_xc(omega) is computed on an omega grid referenced to E_F, so
		# solve in the same relative-energy scale (E - E_F), then shift back.
		h0_diag_ev_abs = np.real(np.diagonal(np.array(kin_ion_full + hartree_kbar_ij_jax), axis1=1, axis2=2)) * ryd2ev
		sigma_sx_host_ev = ryd2ev * np.array(sigma_sx_kbar_ij_jax)
		sigma_xc_diag_omega_ev = np.real(
			load_sigma_xc_diag_from_h5(
				sigma_omega_h5_path,
				sigma_sx_host_ev,
			)
		)
		occ_idx = max(0, min(int(s.occ.stop) - 1, E_dyn0_ev.shape[1] - 1))
		if fermi_reference == "midgap" and (occ_idx + 1) < E_dyn0_ev.shape[1]:
			vbm_ev = float(np.max(E_dyn0_ev[:, occ_idx]))
			cbm_ev = float(np.min(E_dyn0_ev[:, occ_idx + 1]))
			efermi_ref_ev = 0.5 * (vbm_ev + cbm_ev)
		else:
			efermi_ref_ev = float(np.max(E_dyn0_ev[:, occ_idx]))
		h0_diag_ev_rel = h0_diag_ev_abs - efermi_ref_ev
		E_dyn0_rel_ev = E_dyn0_ev - efermi_ref_ev
		omega_lo = float(np.min(omega_grid_ev))
		omega_hi = float(np.max(omega_grid_ev))
		in_omega_grid = (E_dyn0_rel_ev >= omega_lo) & (E_dyn0_rel_ev <= omega_hi)
		E_diag_sc_rel_ev, conv_mask, n_iter_diag = solve_diagonal_sigma_fixed_point(
			h0_diag_ev_rel,
			sigma_xc_diag_omega_ev,
			np.asarray(omega_grid_ev, dtype=np.float64),
			max_iter=120,
			tol_ev=1.0e-7,
			mixing=0.6,
		)
		E_diag_sc_rel_ev = np.where(in_omega_grid, E_diag_sc_rel_ev, E_dyn0_rel_ev)
		conv_mask = np.asarray(conv_mask, dtype=bool) & np.asarray(in_omega_grid, dtype=bool)
		E_diag_sc_ev = E_diag_sc_rel_ev + efermi_ref_ev
		n_in_grid = int(np.count_nonzero(in_omega_grid))
		conv_frac = float(np.mean(conv_mask[in_omega_grid].astype(np.float64))) if n_in_grid > 0 else 1.0

		iv_edge = int(occ_idx)
		ic_edge = int(min(occ_idx + 1, E_dyn0_ev.shape[1] - 1))
		gap_static = _gap_k0(E_static_ref_ev, iv_edge, ic_edge)
		gap_dyn0 = _gap_k0(E_dyn0_ev, iv_edge, ic_edge)
		gap_diag_sc = _gap_k0(E_diag_sc_ev, iv_edge, ic_edge)
		if gap_static is not None and gap_dyn0 is not None and gap_diag_sc is not None:
			print0(
				f"  Gap@k=0 (E{ic_edge+1}-E{iv_edge+1}): "
				f"static={gap_static:.6f} eV, omega0={gap_dyn0:.6f} eV, diag-SC={gap_diag_sc:.6f} eV"
			)
			print0(f"  Gap shifts: omega0-static={gap_dyn0-gap_static:+.6f} eV, diagSC-static={gap_diag_sc-gap_static:+.6f} eV")
		print0(
			"  Diagonal Sigma(E) fixed point: "
			f"in-grid={n_in_grid}/{in_omega_grid.size}, converged={100.0*conv_frac:.1f}% in-grid states "
			f"in {n_iter_diag} iterations"
		)

		# Construct QSGW static Sigma_xc from dynamic Sigma_xc(omega).
		qsgw_diag = build_qsgw_sigma_xc_from_h5(
			sigma_omega_h5_path,
			sigma_sx_host_ev,
			np.asarray(omega_grid_ev, dtype=np.float64),
			E_diag_sc_rel_ev,
		)
		print0(
			"  QSGW Σ_xc interpolation: "
			f"clipped={int(qsgw_diag['n_interp_clipped'])}/{int(np.prod(E_diag_sc_rel_ev.shape))} "
			f"({100.0*qsgw_diag['frac_interp_clipped']:.2f}%) "
			f"outside [{qsgw_diag['omega_min_ev']:.3f}, {qsgw_diag['omega_max_ev']:.3f}] eV"
		)

		with h5py.File(sigma_omega_h5_path, "a") as h5:
			if "qp_static_cohsex_ev" in h5:
				del h5["qp_static_cohsex_ev"]
			h5.create_dataset("qp_static_cohsex_ev", data=np.asarray(E_static_ref_ev, dtype=np.float64))
			if "qp_omega0_ev" in h5:
				del h5["qp_omega0_ev"]
			h5.create_dataset("qp_omega0_ev", data=np.asarray(E_dyn0_ev, dtype=np.float64))
			if "qp_diag_self_consistent_ev" in h5:
				del h5["qp_diag_self_consistent_ev"]
			h5.create_dataset("qp_diag_self_consistent_ev", data=np.asarray(E_diag_sc_ev, dtype=np.float64))
			if "qsgw_interp_clipped_count" in h5:
				del h5["qsgw_interp_clipped_count"]
			h5.create_dataset("qsgw_interp_clipped_count", data=np.asarray(int(qsgw_diag["n_interp_clipped"]), dtype=np.int64))
			if "qsgw_interp_clipped_fraction" in h5:
				del h5["qsgw_interp_clipped_fraction"]
			h5.create_dataset("qsgw_interp_clipped_fraction", data=np.asarray(float(qsgw_diag["frac_interp_clipped"]), dtype=np.float64))

		plot_path = os.path.join(input_dir, "qp_energy_compare.png")
		try:
			plot_qp_energy_comparison(
				plot_path,
				h0_diag_ev_abs,
				E_static_ref_ev,
				E_dyn0_ev,
				E_diag_sc_ev,
			)
			print0(f"  QP energy plot:         {plot_path}")
		except Exception as exc:
			print0(f"  WARNING: failed to generate QP plot: {exc}")
	
	if self_consistent:
		# Self-consistent COHSEX: iterate until Σ converges
		# Key equations:
		#   H = (H_DFT - V_xc) + V_H + Σ_xc
		#   Diagonalize: H U = U ε
		#   G_ij = U @ diag(f) @ U†  where f = occupation (1 for occupied, 0 for empty)
		#   Compute new Σ_SX, Σ_COH, V_H using G_ij
		#   Σ_new = Σ_SX + Σ_COH + V_H
		# Wavefunctions stay fixed; rotation is encoded in G_ij.
		
		n_upper = nb_sigma * (nb_sigma + 1) // 2  # Upper triangle size per k-point
		
		def sigma_iteration_step(sigma_upper_flat: jax.Array) -> jax.Array:
			"""One iteration of self-consistent COHSEX.
			
			Takes/returns upper triangle of Σ (Hermitian optimization).
			"""
			# Restore full Hermitian matrix from upper triangle
			# Input is (nk * n_upper,), reshape to (nk, n_upper) then convert
			sigma_upper = sigma_upper_flat.reshape(nk, n_upper)
			sigma_full = upper_flat_to_hermitian(sigma_upper, nb_sigma)  # (nk, nb_sigma, nb_sigma)
			
			# Diagonalize H = H_DFT + Σ
			H_full = kin_ion_full + sigma_full
			H_full = 0.5 * (H_full + jnp.conj(jnp.swapaxes(H_full, -1, -2)))
			_, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_full)
			
			# Build G_ij = U @ diag(f) @ U† (projector onto occupied states)
			# f = [1,1,...,1,0,0,...,0] with nelec ones
			f_occ = (jnp.arange(nb_sigma) < nelec).astype(jnp.float64)
			Gij_new = jnp.einsum('kim,m,kjm->kij', U_full, f_occ, jnp.conj(U_full), optimize=True)
			
			# Compute new Σ using original wavefunctions but updated G_ij
			with mesh_xy:
				sigma_sx_new, sigma_coh_new, hartree_new = \
					static_sigma_pipeline(wfns, W_mu_nu, V_mu_nu_for_coh, v_q0_noG0_munu, Gij_new)
			sigma_sx_new, sigma_coh_new = _apply_static_head_terms(sigma_sx_new, sigma_coh_new)
			
			# Combine and extract upper triangle
			sigma_new = sigma_sx_new + sigma_coh_new + hartree_new
			return hermitian_to_upper_flat(sigma_new).flatten()
		
		def residual_fn(sigma_upper_flat: jax.Array) -> jax.Array:
			sigma_next = sigma_iteration_step(sigma_upper_flat)
			return sigma_next - sigma_upper_flat
		
		# Initial guess: upper triangle of Σ
		sigma0_upper = hermitian_to_upper_flat(sigma_total_full).flatten()
		
		# Run rCROP acceleration (Python loop to avoid XLA constant folding)
		result = rcrop_nojit(
			residual_fn,
			sigma0_upper,
			m=3,
			maxit=40,
			tol=1e-5,
			print_fn=print0 if meta.rank == 0 else None,
		)
		
		# Restore final Σ
		sigma_final = result.x.reshape(nk, n_upper)
		sigma_total_full = upper_flat_to_hermitian(sigma_final, nb_sigma)
		
		# Final diagonalization
		H_qp_mnk = kin_ion_full + sigma_total_full
		H_qp_mnk = 0.5 * (H_qp_mnk + jnp.conj(jnp.swapaxes(H_qp_mnk, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_qp_mnk)
		
		# Get final components for output
		f_occ = (jnp.arange(nb_sigma) < nelec).astype(jnp.float64)
		Gij_final = jnp.einsum('kim,m,kjm->kij', U_full, f_occ, jnp.conj(U_full), optimize=True)
		
		print_scf_diagnostics(Gij_final, U_full, nelec, nb_sigma, print0)

		with mesh_xy:
			sigma_sx_kbar_ij_jax, sigma_coh_kbar_ij_jax, hartree_kbar_ij_jax = \
				static_sigma_pipeline(wfns, W_mu_nu, V_mu_nu_for_coh, v_q0_noG0_munu, Gij_final)
		sigma_sx_kbar_ij_jax, sigma_coh_kbar_ij_jax = _apply_static_head_terms(
			sigma_sx_kbar_ij_jax,
			sigma_coh_kbar_ij_jax,
		)
	else:
		# One-shot: diagonalize H = (H_DFT - V_xc) + V_H + Σ_xc
		H_qp_mnk = kin_ion_full + sigma_total_full
		H_qp_mnk = 0.5 * (H_qp_mnk + jnp.conj(jnp.swapaxes(H_qp_mnk, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_qp_mnk)

	energies_full_dft, _ = get_enk_bandrange(
		wfn, sym, (qp_band_start, qp_band_stop), (qp_band_start, qp_band_stop),
		nspinor=meta.nspinor,
	)
	energies_dft_ev = jnp.asarray(energies_full_dft) * ryd2ev
	energies_qp_ev = E_full * ryd2ev

	# Rotate sigma components to QP basis: Σ_QP = U† Σ U
	# U_full: (nk, nb_sigma, nb_sigma) eigenvector matrix from diagonalizing H_QP
	# U[k,i,m] = ⟨i_DFT|m_QP⟩ = component i of eigenvector m
	# σ_QP[m,n] = ⟨m_QP|σ|n_QP⟩ = Σ_{ij} conj(U[i,m]) σ[i,j] U[j,n]
	def rotate_to_qp_basis(sigma_dft, U):
		# sigma_dft: (nk, i, j) in DFT basis
		# U: (nk, i, m) where columns m are eigenvectors
		# sigma_qp = U† @ sigma @ U → output (nk, m, n) in QP basis
		return jnp.einsum('kim,kij,kjn->kmn', jnp.conj(U), sigma_dft, U, optimize=True)
	
	sigma_sx_qp = rotate_to_qp_basis(sigma_sx_kbar_ij_jax, U_full)
	sigma_coh_qp = rotate_to_qp_basis(sigma_coh_kbar_ij_jax, U_full)
	hartree_qp = rotate_to_qp_basis(hartree_kbar_ij_jax, U_full)
	
	# Diagnostic: check trace preservation after rotation
	if self_consistent:
		hartree_qp_diag = jnp.real(jnp.diagonal(hartree_qp[0]))
		hartree_qp_trace = jnp.sum(hartree_qp_diag)
		print(f"[Diagnostic] Hartree QP-basis diag[:5] (Ry): {np.array(hartree_qp_diag[:5])}")
		print(f"[Diagnostic] Hartree QP-basis trace (Ry): {float(hartree_qp_trace):.4f} (should match DFT-basis trace)")
	energies_dft_ev_host = np.array(energies_dft_ev)
	energies_qp_ev_host = np.array(energies_qp_ev)
	sx_col_label = "sigSX"
	corr_col_label = "sigCOH"
	total_col_label = "sigTOT"
	if use_ppm_sigma:
		sx_col_label = "sigX"
		corr_col_label = "sigC_EDFT" if sigma_c_at_dft_ev is not None else "sigCw0"
		total_col_label = "sigXC_EDFT" if sigma_c_at_dft_ev is not None else "sigXC"

	# Keep eqp0_noqsym_w.dat consistent with sigma_mnk.h5 by reloading Sigma_c(E_DFT)
	# from the written HDF5 payload when available.
	if (
		use_ppm_sigma
		and sigma_c_at_dft_ev is not None
		and meta.rank == 0
		and sigma_omega_h5_path
		and os.path.exists(sigma_omega_h5_path)
	):
		try:
			with h5py.File(sigma_omega_h5_path, "r") as h5:
				if "sigma_c_at_dft_ev" in h5:
					sigma_c_at_dft_ev = np.asarray(h5["sigma_c_at_dft_ev"], dtype=np.complex128)
				if "omega_dft_rel_ev" in h5 and efermi_dft_ev is not None:
					omega_dft_rel_file = np.asarray(h5["omega_dft_rel_ev"], dtype=np.float64)
					omega_dft_rel_expected = np.asarray(energies_dft_ev_host, dtype=np.float64) - float(efermi_dft_ev)
					max_omega_ref_err = float(np.max(np.abs(omega_dft_rel_file - omega_dft_rel_expected)))
					if max_omega_ref_err > 1.0e-8:
						raise RuntimeError(
							f"{sigma_omega_h5_path} stores omega_dft_rel_ev inconsistent with the active "
							f"WFN/band window: max|Δ|={max_omega_ref_err:.6e} eV"
						)
		except RuntimeError:
			raise
		except Exception as exc:
			print0(f"  WARNING: failed to reload sigma_c_at_dft_ev from {sigma_omega_h5_path}: {exc}")

	# Write PRE-self-consistency sigma to eqp0_noqsym (initial, one-shot values).
	# Only rank 0 should touch the shared text outputs.
	if meta.rank == 0:
		sigma_sx_initial_host = np.array(sigma_sx_initial)
		if use_ppm_sigma and sigma_c_at_dft_ev is not None:
			# Expand diagonal Sigma_c(E_DFT) to (nk, nb, nb) for output formatting.
			sigma_c_diag_ev = np.array(sigma_c_at_dft_ev)
			nk, nb = sigma_c_diag_ev.shape
			sigma_coh_initial_host = np.zeros((nk, nb, nb), dtype=np.complex128)
			for ik in range(nk):
				sigma_coh_initial_host[ik].flat[:: nb + 1] = sigma_c_diag_ev[ik]
		else:
			sigma_coh_initial_host = np.array(sigma_coh_initial)
		hartree_initial_host = np.array(hartree_initial)
		write_sigma_to_file(
			ryd2ev * sigma_sx_initial_host, params["output_file"],
			sigma_coh_kij_eV=(sigma_coh_initial_host if (use_ppm_sigma and sigma_c_at_dft_ev is not None) else ryd2ev * sigma_coh_initial_host),
			hartree_kij_eV=ryd2ev * hartree_initial_host,
			sx_label=sx_col_label,
			corr_label=corr_col_label,
			total_label=total_col_label,
		)
	if (
		meta.rank == 0
		and use_ppm_sigma
		and sigma_freq_debug_output
		and sigma_c_at_dft_ev is not None
		and omega_dft_rel_ev is not None
	):
		kin_ion_diag_ev = np.diagonal(np.asarray(kin_ion_full), axis1=1, axis2=2) * ryd2ev
		sex_static_diag_ev = np.diagonal(np.asarray(sigma_sx_screened_ref), axis1=1, axis2=2) * ryd2ev
		coh_static_diag_ev = np.diagonal(np.asarray(sigma_coh_static_ref), axis1=1, axis2=2) * ryd2ev
		x_diag_ev = np.diagonal(np.asarray(sigma_x_bare_kij), axis1=1, axis2=2) * ryd2ev
		sigma_c_w0_diag_ev = np.diagonal(np.asarray(sigma_coh_ppm_kij), axis1=1, axis2=2) * ryd2ev
		if sigma_c_plus_at_dft_ev is None:
			sigma_c_plus_at_dft_ev = np.full_like(sigma_c_at_dft_ev, np.nan + 1j * np.nan, dtype=np.complex128)
		if sigma_c_minus_at_dft_ev is None:
			sigma_c_minus_at_dft_ev = np.full_like(sigma_c_at_dft_ev, np.nan + 1j * np.nan, dtype=np.complex128)
		if sigma_c_invalid_static_diag_ev is None:
			sigma_c_invalid_static_diag_ev = np.zeros_like(sigma_c_at_dft_ev, dtype=np.complex128)
		debug_path = write_sigma_freq_debug_table(
			sigma_freq_debug_file,
			energies_dft_ev=np.asarray(energies_dft_ev_host, dtype=np.float64),
			omega_rel_dft_ev=np.asarray(omega_dft_rel_ev, dtype=np.float64),
			kin_ion_diag_ev=kin_ion_diag_ev,
			sigma_sex_static_diag_ev=sex_static_diag_ev,
			sigma_coh_static_diag_ev=coh_static_diag_ev,
			sigma_x_diag_ev=x_diag_ev,
			sigma_c_w0_diag_ev=sigma_c_w0_diag_ev,
			sigma_c_plus_edft_ev=np.asarray(sigma_c_plus_at_dft_ev, dtype=np.complex128),
			sigma_c_minus_edft_ev=np.asarray(sigma_c_minus_at_dft_ev, dtype=np.complex128),
			sigma_c_invalid_static_diag_ev=np.asarray(sigma_c_invalid_static_diag_ev, dtype=np.complex128),
			sigma_c_edft_ev=np.asarray(sigma_c_at_dft_ev, dtype=np.complex128),
		)
		print0(f"  Sigma frequency debug:  {debug_path}")

		# Write POST-self-consistency sigma to eqp0_sc (rotated to QP basis)
		sc_output_file = None
		if self_consistent and meta.rank == 0:
			sigma_sx_final_host = np.array(sigma_sx_qp)
			sigma_coh_final_host = np.array(sigma_coh_qp)
			hartree_final_host = np.array(hartree_qp)
			sc_output_file = params["output_file"].replace("eqp0", "eqp0_sc").replace(".dat", "_sc.dat")
			if sc_output_file == params["output_file"]:
				sc_output_file = params["output_file"].replace(".dat", "_sc.dat")
			write_sigma_to_file(
				ryd2ev * sigma_sx_final_host, sc_output_file,
				sigma_coh_kij_eV=ryd2ev * sigma_coh_final_host,
				hartree_kij_eV=ryd2ev * hartree_final_host,
				sx_label=sx_col_label,
				corr_label=corr_col_label,
				total_label=total_col_label,
			)

	# Write eqp1.dat for self-consistent runs
	eqp1_written = None
	if self_consistent and meta.rank == 0:
		sigma_total_initial = sigma_sx_initial + sigma_coh_initial + hartree_initial
		H_oneshot_diag = np.array(jnp.real(
			jnp.diagonal(kin_ion_full, axis1=1, axis2=2) +
			jnp.diagonal(sigma_total_initial, axis1=1, axis2=2)
		))
		eqp1_written = write_eqp1(
			os.path.join(input_dir, "eqp1.dat"),
			energies_dft_ev_host, energies_qp_ev_host,
			H_oneshot_diag * ryd2ev,
			meta.nkx, meta.nky, meta.nkz, nb_sigma,
		)
	elif (not self_consistent) and meta.rank == 0 and (sigma_xc_at_dft_ev is not None):
		# G0W0 comparison: evaluate diagonal (H0 + Sigma_xc(E_DFT)) next to E_DFT.
		h0_diag_ev_abs = np.real(np.diagonal(np.array(kin_ion_full + hartree_kbar_ij_jax), axis1=1, axis2=2)) * ryd2ev
		g0w0_diag_ev = h0_diag_ev_abs + sigma_xc_at_dft_ev
		eqp_g0w0_path = os.path.join(input_dir, "eqp_g0w0.dat")
		write_eqp_g0w0(eqp_g0w0_path, energies_dft_ev_host, g0w0_diag_ev)
		print0(f"  G0W0 diag (E_DFT):     {eqp_g0w0_path}")

	# Write QP rotation matrices and eigenvalues to h5 file
	if meta.rank == 0:
		qp_rot_path = os.path.join(input_dir, "qp_wfn_rotations.h5")
		# Generate full k-mesh for output
		kpoints_full = np.array(sym.unfolded_kpts, dtype=np.float64)
		# Get reduced k-points and mapping for WFN.h5 lookup
		kpoints_reduced = np.array(wfn.kpoints, dtype=np.float64)
		kirr_to_kfull = np.array(sym.kirr_fullids, dtype=np.int32)
		# Convert eigenvalues from Rydberg to Hartree
		E_qp_hartree = np.array(E_full) / 2.0  # Ry -> Ha
		U_host = np.array(U_full)
		write_qp_rotations_h5(
			qp_rot_path,
			U_mnk=U_host,
			E_qp_nk=E_qp_hartree,
			band_start=qp_band_start,
			band_stop=qp_band_stop,
			kpoints_crys=kpoints_full,
			nkx=meta.nkx,
			nky=meta.nky,
			nkz=meta.nkz,
			kpoints_reduced=kpoints_reduced,
			kirr_to_kfull=kirr_to_kfull,
		)
	else:
		qp_rot_path = None

	# ========================================================================
	# OUTPUT FILES SUMMARY
	# ========================================================================
	print0("")
	print0("-" * 72)
	print0("  OUTPUT FILES")
	print0("-" * 72)
	print0(f"  Sigma matrix elements:  {params['output_file']}")
	if sigma_omega_h5_path:
		print0(f"  Sigma_mnk(ω) h5:        {sigma_omega_h5_path}")
	if sigma_munu_stream_path:
		print0(f"  Sigma_c(μ,ν,ω) h5:      {sigma_munu_stream_path}")
	if self_consistent:
		print0(f"  Sigma (SC rotated):     {sc_output_file}")
	if eqp1_written:
		print0(f"  QP energies (eqp1):     {eqp1_written}")
	if qp_rot_path:
		print0(f"  QP rotations (h5):      {qp_rot_path}")
	print0(f"  Restart arrays:         {tensors_filename}")
	print0("")

	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

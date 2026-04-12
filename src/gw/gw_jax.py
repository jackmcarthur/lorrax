import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
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

_maybe_init_jax_distributed()
try:
	jax.devices()
except RuntimeError as exc:
	if "Unknown backend: 'gpu'" in str(exc):
		os.environ.pop("JAX_PLATFORM_NAME", None)
		os.environ["JAX_PLATFORMS"] = "cpu"
	else:
		raise
from file_io import (
    WFNReader,
    write_sigma_to_file, write_eqp_g0w0, write_sigma_omega_h5,
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
from .gw_driver_helpers import (
	build_ppm_sigma_runtime_options,
	build_screening_setup,
)
from .w_isdf import compute_screening
from .minimax_config import (
	minimax_config_from_params,
	sigma_quadrature_config_from_params,
)
from .ppm_sigma import (
	compute_w0_wiwp_and_ppm_from_minimax,
	compute_sigma_c_ppm_omega_grid,
)
from .qsgw_utils import (
	print_scf_diagnostics,
	solve_diagonal_sigma_fixed_point,
	load_sigma_xc_diag_from_h5,
	build_qsgw_sigma_xc_from_h5,
)
from .head_correction import (
	compute_static_head_terms_from_sample,
	format_head_sample_diagnostics,
	format_static_head_diagnostics,
	resolve_head_sample,
	static_head_terms_to_kij,
)
from .wavefunction_bundle import BandSlices
from mixing.acceleration import (
    rcrop_nojit, hermitian_to_upper_flat, upper_flat_to_hermitian
)
from common import Meta
from common import jax_profile
import common.timing as timing
import h5py
import builtins


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
		- V: (n_rmu, n_rmu) V_q at q=0 with G=0 excluded
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
	cell_volume = meta.cell_volume
	
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
	V = V_qmunu_raw[0, 0, 0, :, :]  # At q=(0,0,0)
	# G0 = ζ_μ(G=0) at q=0. g0_mu_local may be 3D or 4D depending on
	# how process_allgather handled the leading dimension. Extract q=(0,0,0).
	_g0 = g0_mu_local
	while _g0.ndim > 1:
		_g0 = _g0[0]
	G0_mu_nu = _g0  # (n_rmu,)
	
	print("\n  V_q computed:")
	print(f"    Shape: {V_qmunu.shape}")
	print(f"    V_q=0 trace: {jnp.trace(V).real:.4f}")
	
	return {
		'V_qmunu': V_qmunu,
		'G0_mu_nu': G0_mu_nu,
		'psi_l_rmu_Y': psi_l_rmu_Y,
		'psi_l_rmuT_X': psi_l_rmuT_X,
		'psi_r_rmu_Y': psi_r_rmu_Y,
		'psi_r_rmuT_X': psi_r_rmuT_X,
		'zeta_h5_path': zeta_h5_path,
	}


# ================= ISDF sigma =================
#
# Sigma_SX  = -project[ FFT[ G_occ(R) * W(R) / sqrt(Nk) ] ]
# Sigma_COH = +project[ FFT[ G_RI(R) * (W-V)(R) / (2*sqrt(Nk)) ] ]
# V_H       = project[ V0 * rho ]

from .projection_kernel import project as _project

def _hartree(wfns, Gij, V, nk_tot):
	"""V_H(m,n,k) = <m| V(q=0,noG0) * rho |n>.  V is flat-k (nk,μ,μ); uses V[0]."""
	s = wfns.slices
	psi_yr, psi_xr = wfns.yr(s.sigma), wfns.xr(s.sigma)
	rho = jnp.real(jnp.einsum('kisx,kjsx,kij->x', jnp.conj(psi_yr), psi_yr, Gij, optimize=True))
	Vrho = jnp.einsum('xy,y->x', V[0], rho / jnp.asarray(nk_tot, dtype=jnp.float64), optimize=True)
	return jnp.einsum('kmsx,x,knsx->kmn', jnp.conj(psi_xr), Vrho, psi_xr, optimize=True)

from .greens_function_kernel import build_G

def _flatten_V_W(V_qmunu, W_q, meta):
	"""Flatten V_qmunu → V(nk,μ,μ) and W_q → W(nk,μ,μ).  W_q=None → W=V."""
	V = jnp.asarray(V_qmunu)[0, 0, 0].reshape(-1, V_qmunu.shape[-2], V_qmunu.shape[-1])
	return V, (V if W_q is None else W_q)

def _build_Gij(meta, mesh_xy):
	"""Occupation projector G_ij = diag(1,...,1,0,...,0) for sigma bands."""
	nocc = min(meta.nelec, meta.nb_sigma)
	Gij = jnp.zeros((meta.nk_tot, meta.nb_sigma, meta.nb_sigma), dtype=jnp.complex128)
	Gij = Gij.at[:, :nocc, :nocc].set(jnp.eye(nocc, dtype=jnp.complex128))
	return jax.device_put(Gij, NamedSharding(mesh_xy, P(None, None, None)))


def _compute_static_head(params, input_dir, wfn, sym, meta, do_screened, print0):
	"""Resolve q→0 head and compute exact band-diagonal head terms for COHSEX."""
	head = resolve_head_sample(params, input_dir, wfn, sym, meta, print0, omega=0.0+0.0j)
	print0(format_head_sample_diagnostics(head, include_screened=do_screened))
	occ_mask = np.arange(meta.nb_sigma, dtype=np.int32) < meta.nelec
	terms = compute_static_head_terms_from_sample(
		head, occ=occ_mask, cell_volume=meta.cell_volume, nk_tot=meta.nk_tot)
	print0(format_static_head_diagnostics(terms))
	return terms


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
	_, centroid_indices, _n_rmu = load_centroids(params["centroids_file"], wfn.fft_grid)
	_print0(f"  [TIMING-BOOT] load_centroids: {_boot_time.perf_counter() - _t_cent:.3f}s")
	# Resolve tmp_dir and output path relative to input file directory
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{_n_rmu}.h5")
	_print0(f"  ISDF basis: {_n_rmu} centroids")
	_print0(f"  FFT grid: {wfn.fft_grid[0]}×{wfn.fft_grid[1]}×{wfn.fft_grid[2]}   Cell volume: {wfn.cell_volume:.2f} a.u.³")
	_print0("")

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
		tmp_dir=tmp_dir,
		tensors_filename=tensors_filename,
		print0=print0,
	)
	print0(f"  [TIMING-BOOT] prepare_isdf_and_wavefunctions: {_boot_time.perf_counter() - _t_isdf:.3f}s")
	V_qmunu = isdf.V_qmunu
	wfns = isdf.wf_bundle
	s = wfns.slices

	# Compute screened Coulomb W = (1 - Vχ)⁻¹ V
	if do_screened:
			with timing.section("gw_jax.chi0_W"):
				with jax_profile.trace_section("chi0_W"):
					minimax_config = minimax_config_from_params(params)
					screening_setup = build_screening_setup(params, minimax_config)
					screening_ppm_omega_p = screening_setup.ppm_omega_p
					if use_ppm_sigma:
						screening_ppm_omega_p = None
					W_q = compute_screening(
						V_qmunu, wfns, meta, mesh_xy,
						minimax_config=screening_setup.minimax_config,
						ppm_omega_p=screening_ppm_omega_p,
						ppm_fallback_omega=screening_setup.ppm_fallback_omega,
						tensors_filename=tensors_filename,
						print0=print0,
					)

	V, W = _flatten_V_W(V_qmunu, W_q if do_screened else None, meta)
	Gij = _build_Gij(meta, mesh_xy)

	# q→0 head correction (exact band-diagonal terms for static COHSEX)
	static_head_terms = None
	if do_G0 and not use_ppm_sigma:
		static_head_terms = _compute_static_head(
			params, input_dir, wfn, sym, meta, do_screened, print0)

	kgrid = meta.kgrid
	_nk_tot = int(meta.nk_tot)

	# FFT helpers: flat-k ↔ flat-R.  Callers pass flat arrays, never see 3D k-grid.
	from common.fft_helpers import make_jittable_local_fftn_3d, make_jittable_local_ifftn_3d

	def _make_fft_pair(spec):
		"""Build IFFT/FFT pair that accept flat-k-first and return flat-R/k-first."""
		shard = NamedSharding(mesh_xy, spec)
		raw_ifftn = make_jittable_local_ifftn_3d(mesh_xy, spec, spec, norm='ortho', axes=(0, 1, 2))
		raw_fftn = make_jittable_local_fftn_3d(mesh_xy, spec, spec, norm='ortho', axes=(0, 1, 2))
		@jax.jit
		def ifftn(x_k):
			x_3d = jax.lax.with_sharding_constraint(x_k.reshape(*kgrid, *x_k.shape[1:]), shard)
			return raw_ifftn(x_3d).reshape(_nk_tot, *x_k.shape[1:])
		@jax.jit
		def fftn(x_R):
			x_3d = jax.lax.with_sharding_constraint(x_R.reshape(*kgrid, *x_R.shape[1:]), shard)
			return raw_fftn(x_3d).reshape(_nk_tot, *x_R.shape[1:])
		return ifftn, fftn

	_G_ifftn, _G_fftn = _make_fft_pair(P(None, None, None, None, 'x', None, 'y'))
	_V_ifftn, _ = _make_fft_pair(P(None, None, None, 'x', 'y'))

	_inv_sqrt_nk = -1.0 / jnp.sqrt(_nk_tot)

	@jax.jit
	def _convolve(G_k, V, prefactor):
		G_R = _G_ifftn(G_k)
		V_R = _V_ifftn(V)[:, None, :, None, :]
		return prefactor * _G_fftn(G_R * V_R * _inv_sqrt_nk)

	# ---- Static COHSEX: Σ_SX, Σ_COH, V_H ----

	@jax.jit
	def sigma_sx(wfns, Gij, W):
		s = wfns.slices
		G_occ = build_G(wfns.xn(s.sigma), wfns.yr(s.sigma), Gij=Gij)
		return _project(wfns.xr(s.sigma), wfns.yn(s.sigma), _convolve(G_occ, W, 1.0))

	@jax.jit
	def sigma_coh(wfns, W, V):
		s = wfns.slices
		G_ri = build_G(wfns.xn(s.full), wfns.yr(s.full))
		return _project(wfns.xr(s.sigma), wfns.yn(s.sigma), _convolve(G_ri, W - V, -0.5))

	@jax.jit
	def hartree(wfns, Gij, V):
		return _hartree(wfns, Gij, V, _nk_tot)

	def _add_head(sig_sx, sig_coh):
		if static_head_terms is None:
			return sig_sx, sig_coh
		sx_h, coh_h = static_head_terms_to_kij(
			static_head_terms, nk_tot=meta.nk_tot, do_screened=do_screened)
		if not do_screened:
			coh_h = jnp.zeros_like(coh_h)
		rep = NamedSharding(mesh_xy, P(None, None, None))
		return sig_sx + jax.device_put(sx_h, rep), sig_coh + jax.device_put(coh_h, rep)

	# ---- Execute static COHSEX ----
	import gc; gc.collect()
	with mesh_xy:
		with timing.section("gw_jax.sigma"):
			sig_sx  = sigma_sx(wfns, Gij, W)
			sig_coh = sigma_coh(wfns, W, V)
			sig_h   = hartree(wfns, Gij, V)

			sig_sx, sig_coh = _add_head(sig_sx, sig_coh)
			
			sig_sx.block_until_ready()
			sig_coh.block_until_ready()
			sig_h.block_until_ready()

	# Bare exchange Σ_X (used for both COHSEX comparison and PPM Σ^c = Σ_xc - Σ_X)
	with mesh_xy:
		sig_x = sigma_sx(wfns, Gij, V)
	sig_x.block_until_ready()

	# ---- GN-PPM: replace static COH with frequency-integrated Σ^c ----
	sigma_omega_h5_path = None
	sigma_c_at_dft_ev = None
	sigma_xc_at_dft_ev = None
	omega_dft_rel_ev = None
	efermi_dft_ev = None
	if use_ppm_sigma:
		if not do_screened:
			raise ValueError("use_ppm_sigma=true requires do_screened=true.")
		if self_consistent:
			raise NotImplementedError("use_ppm_sigma not supported with self_consistent.")
		
		laplace_quad = minimax_config_from_params(params)
		sigma_window_quad = sigma_quadrature_config_from_params(params)
		
		ppm_options = build_ppm_sigma_runtime_options(params, input_dir=input_dir, ryd2ev=ryd2ev)
		if ppm_options.fermi_reference not in ("vbm", "midgap"):
			raise ValueError("fermi_reference must be 'vbm' or 'midgap'.")
		print0("\n" + "-" * 72 + "\n  GN-PPM + FREQUENCY-INTEGRATED SIGMA\n" + "-" * 72)

		with timing.section("gw_jax.ppm_sigma"):
			# Build PPM poles from minimax W(0) and W(iωp)
			eref = laplace_quad.energy_reference or ppm_options.fermi_reference
			ppm = compute_w0_wiwp_and_ppm_from_minimax(
				V_qmunu, wfns, meta, mesh_xy,
				minimax_config=laplace_quad, omega_p_ry=ppm_options.omega_p_ry,
				minimax_energy_reference=eref, fallback_omega=ppm_options.ppm_fallback,
				print0=print0)

			# Frequency-integrated Σ^c(ω)
			sigma_omega = compute_sigma_c_ppm_omega_grid(
				wfns, ppm, meta, mesh_xy, ppm_options,
				sigma_window_quad=sigma_window_quad,
				print0=print0,
			)
			sigma_c_omega = sigma_omega.sigma_c_kij  # (n_omega, nk, nb, nb) or None if streamed

			# Evaluate Σ_c at DFT energies
			sigma_c_at_dft_ev = None
			sigma_xc_at_dft_ev = None
			omega_dft_rel_ev = None
			efermi_dft_ev = None
			if meta.rank == 0:
				enk_dft, _ = get_enk_bandrange(wfn, sym,
					(meta.band_edges[0], meta.band_edges[3]),
					(meta.band_edges[0], meta.band_edges[3]), nspinor=meta.nspinor)
				enk_dft_ev = np.asarray(enk_dft) * ryd2ev
				n_occ = min(meta.nelec, enk_dft_ev.shape[1])
				vbm_ev = float(np.max(enk_dft_ev[:, :n_occ]))
				cbm_ev = float(np.min(enk_dft_ev[:, n_occ:])) if n_occ < enk_dft_ev.shape[1] else vbm_ev
				efermi_dft_ev = 0.5 * (vbm_ev + cbm_ev)
				omega_dft_rel_ev = enk_dft_ev - efermi_dft_ev
				print0(f"  E_F(midgap) = {efermi_dft_ev:.6f} eV  (VBM={vbm_ev:.6f}, CBM={cbm_ev:.6f})")

				# Interpolate diagonal Σ_c(ω) at each DFT energy
				omega_ev = np.asarray(ppm_options.omega_grid_ev, dtype=np.float64)
				if sigma_c_omega is not None:
					sig_c_diag = np.diagonal(np.asarray(sigma_c_omega), axis1=2, axis2=3) * ryd2ev
				else:
					with h5py.File(sigma_omega.sigma_kij_h5_path, "r") as h5:
						sig_c_diag = np.diagonal(
							np.asarray(h5["sigma_c_kij_ry"], dtype=np.complex128), axis1=2, axis2=3) * ryd2ev
				nk, nb = sig_c_diag.shape[1], sig_c_diag.shape[2]
				sigma_c_at_dft_ev = np.array([
					[complex(np.interp(omega_dft_rel_ev[ik, ib], omega_ev, np.real(sig_c_diag[:, ik, ib])),
					         np.interp(omega_dft_rel_ev[ik, ib], omega_ev, np.imag(sig_c_diag[:, ik, ib])))
					 for ib in range(nb)] for ik in range(nk)], dtype=np.complex128)
				sig_x_diag_ev = np.diagonal(np.asarray(sig_x), axis1=1, axis2=2) * ryd2ev
				sigma_xc_at_dft_ev = sig_x_diag_ev + sigma_c_at_dft_ev

			# Replace static COHSEX with PPM results
			sig_sx = sig_x
			# sig_coh is replaced by a diagonal Σ_c(E_DFT) matrix for output;
			# the full Σ_c(ω) is in sigma_c_omega / sigma_omega_h5

			# Write sigma_mnk.h5
			sigma_omega_h5_path = params.get("sigma_omega_h5_file", "sigma_mnk.h5")
			if not os.path.isabs(sigma_omega_h5_path):
				sigma_omega_h5_path = os.path.join(input_dir, sigma_omega_h5_path)
			if meta.rank == 0:
				if sigma_c_omega is not None:
					write_sigma_omega_h5(
						sigma_omega_h5_path, ppm_options.omega_grid_ev, None,
						sigma_c_kij_ev=ryd2ev * sigma_c_omega,
						sigma_sx_kij_ev=ryd2ev * sig_sx,
						hartree_kij_ev=ryd2ev * sig_h)
				elif sigma_omega.sigma_kij_h5_path:
					with h5py.File(sigma_omega.sigma_kij_h5_path, "r") as h5_in:
						dset_c = h5_in["sigma_c_kij_ry"]
						n_omega, nk, nb = dset_c.shape[0], dset_c.shape[1], dset_c.shape[2]
						batch = max(1, min(ppm_options.sigma_omega_batch_size, n_omega))
						with h5py.File(sigma_omega_h5_path, "w") as h5_out:
							h5_out.create_dataset("omega_ev", data=np.asarray(ppm_options.omega_grid_ev, dtype=np.float64))
							dset_out = h5_out.create_dataset("sigma_c_kij_ev",
								shape=dset_c.shape, dtype=np.complex128, chunks=(batch, max(1, min(4, nk)), nb, nb))
							h5_out.create_dataset("sigma_sx_kij_ev", data=ryd2ev * np.array(sig_sx))
							h5_out.create_dataset("hartree_kij_ev", data=ryd2ev * np.array(sig_h))
							for ibeg in range(0, n_omega, batch):
								dset_out[ibeg:min(ibeg+batch, n_omega)] = \
									ryd2ev * np.array(dset_c[ibeg:min(ibeg+batch, n_omega)], dtype=np.complex128)

	# ---- QP Hamiltonian: H_QP = (H_DFT - V_xc) + V_H + Σ_xc ----
	sigma_total = sig_sx + sig_coh + sig_h
	kin_ion = load_kin_ion_submatrix(params["kin_ion_file"], meta.band_edges[0], meta.band_edges[3])

	# PPM diagonal self-consistency and QSGW (rank 0 only)
	if use_ppm_sigma and meta.rank == 0 and sigma_omega_h5_path and os.path.exists(sigma_omega_h5_path):
		H_qp = np.array(kin_ion + sigma_total)
		H_qp = 0.5 * (H_qp + np.conj(np.swapaxes(H_qp, -1, -2)))
		E_qp_ev = np.linalg.eigvalsh(H_qp) * ryd2ev

		# Diagonal fixed-point: E = diag(H0) + Re Σ_xc(E)
		h0_diag_ev = np.real(np.diagonal(np.array(kin_ion + sig_h), axis1=1, axis2=2)) * ryd2ev
		occ_idx = min(meta.nelec, E_qp_ev.shape[1] - 1)
		vbm = float(np.max(E_qp_ev[:, occ_idx]))
		cbm = float(np.min(E_qp_ev[:, occ_idx + 1])) if occ_idx + 1 < E_qp_ev.shape[1] else vbm
		efermi = 0.5 * (vbm + cbm)
		omega_ev = np.asarray(ppm_options.omega_grid_ev, dtype=np.float64)
		sigma_xc_diag = np.real(load_sigma_xc_diag_from_h5(sigma_omega_h5_path, ryd2ev * np.array(sig_sx)))
		E_sc, conv, n_iter = solve_diagonal_sigma_fixed_point(
			h0_diag_ev - efermi, sigma_xc_diag, omega_ev, max_iter=120, tol_ev=1e-7, mixing=0.6)
		in_grid = (E_qp_ev * ryd2ev - efermi >= omega_ev[0]) & (E_qp_ev * ryd2ev - efermi <= omega_ev[-1])
		E_sc = np.where(in_grid, E_sc, E_qp_ev - efermi) + efermi
		n_in = int(np.count_nonzero(in_grid))
		print0(f"  Diagonal SC: {n_in}/{in_grid.size} states in grid, {n_iter} iterations")

		qsgw = build_qsgw_sigma_xc_from_h5(sigma_omega_h5_path,
			ryd2ev * np.array(sig_sx), omega_ev, E_sc - efermi)
		print0(f"  QSGW: {int(qsgw['n_interp_clipped'])} clipped "
			f"({100*qsgw['frac_interp_clipped']:.1f}%)")

	if self_consistent:
		# SC-COHSEX iteration
		n_upper = meta.nb_sigma * (meta.nb_sigma + 1) // 2
		nk = meta.nk_tot

		def _sc_step(sigma_upper_flat):
			sigma_full = upper_flat_to_hermitian(
				sigma_upper_flat.reshape(nk, n_upper), meta.nb_sigma)
			H = 0.5 * ((kin_ion + sigma_full) + jnp.conj(jnp.swapaxes(kin_ion + sigma_full, -1, -2)))
			_, U = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)
			f = (jnp.arange(meta.nb_sigma) < meta.nelec).astype(jnp.float64)
			Gij_new = jnp.einsum('kim,m,kjm->kij', U, f, jnp.conj(U), optimize=True)
			with mesh_xy:
				sx_new = sigma_sx(wfns, Gij_new, W)
				coh_new = sigma_coh(wfns, W, V)
				h_new = hartree(wfns, Gij_new, V)
			sx_new, coh_new = _add_head(sx_new, coh_new)
			return hermitian_to_upper_flat(sx_new + coh_new + h_new).flatten()

		result = rcrop_nojit(
			lambda x: _sc_step(x) - x,
			hermitian_to_upper_flat(sigma_total).flatten(),
			m=3, maxit=40, tol=1e-5,
			print_fn=print0 if meta.rank == 0 else None)
		sigma_total = upper_flat_to_hermitian(result.x.reshape(nk, n_upper), meta.nb_sigma)

		# Final sigma components from converged Gij
		H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)
		f = (jnp.arange(meta.nb_sigma) < meta.nelec).astype(jnp.float64)
		Gij_final = jnp.einsum('kim,m,kjm->kij', U_full, f, jnp.conj(U_full), optimize=True)
		print_scf_diagnostics(Gij_final, U_full, meta.nelec, meta.nb_sigma, print0)
		with mesh_xy:
			sig_sx  = sigma_sx(wfns, Gij_final, W)
			sig_coh = sigma_coh(wfns, W, V)
			sig_h   = hartree(wfns, Gij_final, V)
		sig_sx, sig_coh = _add_head(sig_sx, sig_coh)
	else:
		H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)

	# ---- DFT and QP energies ----
	b0, b3 = meta.band_edges[0], meta.band_edges[3]
	enk_dft, _ = get_enk_bandrange(wfn, sym, (b0, b3), (b0, b3), nspinor=meta.nspinor)
	E_dft_ev = np.array(enk_dft) * ryd2ev
	E_qp_ev = np.array(E_full) * ryd2ev

	# ---- Output ----
	if meta.rank == 0:
		# eqp0.dat — main QP output
		write_sigma_to_file(
			ryd2ev * np.array(sig_sx), params["output_file"],
			sigma_coh_kij_eV=ryd2ev * np.array(sig_coh),
			hartree_kij_eV=ryd2ev * np.array(sig_h),
			sx_label="sigX" if use_ppm_sigma else "sigSX",
			corr_label="sigC" if use_ppm_sigma else "sigCOH",
			total_label="sigXC" if use_ppm_sigma else "sigTOT",
		)

		# G0W0 diagonal: E_QP = diag(H0 + Σ_xc(E_DFT))
		if not self_consistent and sigma_xc_at_dft_ev is not None:
			h0_diag = np.real(np.diagonal(np.array(kin_ion + sig_h), axis1=1, axis2=2)) * ryd2ev
			write_eqp_g0w0(os.path.join(input_dir, "eqp_g0w0.dat"), E_dft_ev, h0_diag + sigma_xc_at_dft_ev)
			print0(f"  G0W0 diag (E_DFT):     {os.path.join(input_dir, 'eqp_g0w0.dat')}")

		# QP rotation matrices
		write_qp_rotations_h5(
			os.path.join(input_dir, "qp_wfn_rotations.h5"),
			U_mnk=np.array(U_full),
			E_qp_nk=np.array(E_full) / 2.0,
			band_start=b0, band_stop=b3,
			kpoints_crys=np.array(sym.unfolded_kpts, dtype=np.float64),
			nkx=meta.nkx, nky=meta.nky, nkz=meta.nkz,
			kpoints_reduced=np.array(wfn.kpoints, dtype=np.float64),
			kirr_to_kfull=np.array(sym.kirr_fullids, dtype=np.int32),
		)

	print0(f"\n  Output: {params['output_file']}")
	if sigma_omega_h5_path:
		print0(f"  Sigma(ω): {sigma_omega_h5_path}")
	print0(f"  Restart:  {tensors_filename}")
	print0("")
	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

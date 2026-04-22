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
		# Under the Cray MPICH stack, each rank runs on exactly one GPU
		# via CUDA_VISIBLE_DEVICES=$SLURM_LOCALID (set by select_gpu.sh
		# in lxrun).  JAX's no-args distributed.initialize() assumes
		# each process owns *all* local GPUs, then hangs in the
		# jax.devices() topology exchange ("GetKeyValue() timed out
		# with key: cuda:local_topology/cuda/1") because each rank is
		# looking for peers that don't exist on its side.  Detect local
		# GPU count from CUDA_VISIBLE_DEVICES and pass local_device_ids
		# explicitly — same pattern used by slate_*_test.py,
		# cusolvermp_eigh_test.py, and psp.run_nscf.
		cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
		n_local = len([x for x in cv.split(",") if x.strip()]) if cv else 0
		init_kwargs = {"local_device_ids": list(range(n_local))} if n_local else {}
		try:
			jax.distributed.initialize(**init_kwargs)
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
    WFNReader, write_sigma_omega_h5,
    load_kin_ion_submatrix, load_centroids,
)
from common import symmetry_maps
from common.load_wfns import get_enk_bandrange
from .gw_config import LorraxConfig
from .gw_init import (
	get_effective_chunk_size,
	prepare_isdf_and_wavefunctions,
)
from .gw_driver_helpers import build_ppm_sigma_runtime_options
from .w_isdf import (
	build_static_quadrature,
	build_imag_quadrature,
	compute_chi0,
	flatten_V_qmunu,
	solve_w,
)
from .ppm_sigma import (
	fit_gn_ppm,
	compute_ppm_sigma_omega_grid,
)
from .cohsex_sigma import (
	build_Gij,
	compute_cohsex_sigma,
	get_cohsex_kernels,
	_add_static_head,
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



# ================= ISDF sigma =================
#
# Static COHSEX kernels (Σ_SX, Σ_COH, V_H) live in cohsex_sigma.py.


def _build_mesh():
	"""Construct 2D device mesh with most-square factorization."""
	total = jax.process_count() * jax.local_device_count()
	gx = int(np.sqrt(total))
	while gx > 1 and total % gx != 0:
		gx -= 1
	return Mesh(np.array(jax.devices()).reshape(gx, total // gx), ['x', 'y'])


def _compute_static_head(config, input_dir, wfn, sym, meta, do_screened, print0):
	"""Resolve q→0 head and compute exact band-diagonal head terms for COHSEX."""
	# resolve_head_sample expects a dict-like interface for params
	head_params = {
		"wcoul0_source": config.wcoul0_source,
		"wcoul0_eta": config.wcoul0_eta,
		"vhead": config.vhead,
		"whead_0freq": config.whead_0freq,
		"whead_imfreq": config.whead_imfreq,
	}
	head = resolve_head_sample(head_params, input_dir, wfn, sym, meta, print0, omega=0.0+0.0j)
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
	
	# Gate prints to rank 0
	_orig_print = builtins.print
	def print0(*a, **k):
		if jax.process_index() == 0:
			k.setdefault("flush", True)
			_orig_print(*a, **k)
	builtins.print = print0
 
	# ========================================================================
	# CONFIGURATION
	# ========================================================================
	config = LorraxConfig.from_input_file(args.input, print_fn=print0)
	input_dir = config.input_dir
	ryd2ev = 13.6056980659

	# ========================================================================
	# INITIALIZATION
	# ========================================================================
	current_backend = jax.default_backend()
	n_devices = len(jax.devices())
	n_procs = jax.process_count()
	device_names = jax.devices()[0].device_kind if n_devices > 0 else "unknown"

	mesh_xy = _build_mesh()
	grid_x, grid_y = mesh_xy.devices.shape

	# Eagerly init MPI_THREAD_MULTIPLE via the phdf5 FFI so the first
	# collective H5Fcreate (in zeta_fit_chunked) doesn't pay the
	# ~400 ms MPI_Init_thread cost on the critical path.  Overlaps
	# with the JAX compile phases that follow.  No-op if FFI isn't
	# used; cheap to attempt and swallow errors.
	if getattr(config, "use_ffi_io", False):
		try:
			from ffi.common.ffi_loader import phdf5_init_mpi
			phdf5_init_mpi()
		except Exception as _e:
			print0(f"  [phdf5 init_mpi] skipped: {_e}")

	# Enable JAX persistent compile cache for the WHOLE run (not just
	# w_isdf / ppm_sigma where it was previously activated).  On a
	# warm cache this reliably removes ~3 s of XLA compile from the
	# cold-start path at MoS2 3x3 (measured 2026-04-19, 267 entries
	# = 1.9 MiB on disk).  Opt-out by setting ISDF_JAX_CACHE_DIR=""
	# before launch.  Default cache location honours ISDF_JAX_CACHE_DIR
	# → XDG_CACHE_HOME/isdf_jax_compilation → ~/.cache/isdf_jax_compilation.
	try:
		from common.jax_compile_cache import ensure_jax_compile_cache
		ensure_jax_compile_cache()
	except Exception as _e:
		print0(f"  [jax compile cache] skipped: {_e}")

	from .gw_output import print_banner, print_section, print_system_summary, write_results, GWResults
	print_banner(
		backend=current_backend, n_devices=n_devices,
		grid_x=grid_x, grid_y=grid_y, n_procs=n_procs,
		device_kind=device_names, print_fn=print0,
	)

	global wfn
	wfn = WFNReader(config.wfn_file)
	sym = symmetry_maps.SymMaps(wfn)
	_, centroid_indices, _n_rmu = load_centroids(config.centroids_file, wfn.fft_grid)
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{_n_rmu}.h5")
	print_system_summary(
		n_rmu=_n_rmu, fft_grid=wfn.fft_grid,
		cell_volume=wfn.cell_volume, print_fn=print0,
	)

	meta = Meta.from_system(wfn, sym, config.nval, config.ncond, config.nband, _n_rmu, config.bispinor)
	meta.rank = jax.process_index()
	meta.n_proc = jax.process_count()
	meta.sys_dim = config.sys_dim
	meta.bispinor = config.bispinor
	meta.chunk_size = get_effective_chunk_size(config.chunk_size)

	band_slices = BandSlices.from_band_edges(*meta.band_edges)

	# Optional BGW vcoul override: use BGW's MC-averaged v(q+G) for all G
	# (LORRAX's internal mc_average_vcoul_body only averages G=0).  This is
	# purely diagnostic — enables bit-reproducible BGW comparisons.
	bgw_v_grid_fn = None
	if config.use_bgw_vcoul:
		if config.bgw_vcoul_file is None:
			raise ValueError("use_bgw_vcoul=true requires bgw_vcoul_file to be set")
		from file_io import read_bgw_vcoul, fill_v_grid_for_q
		bgw_path = config.bgw_vcoul_file
		if not os.path.isabs(bgw_path):
			bgw_path = os.path.join(input_dir, bgw_path)
		print0(f"  BGW vcoul override: loading {bgw_path}")
		_bgw_table = read_bgw_vcoul(bgw_path)
		print0(f"    {_bgw_table.q_fracs.shape[0]} unique q-points, "
		       f"G counts per q: {[len(g) for g in _bgw_table.G_miller_per_q]}")
		_cell_vol = float(wfn.cell_volume)
		_fft_grid = tuple(int(x) for x in wfn.fft_grid)
		# Reciprocal-space symmetry operators.  BGW's vcoul file only stores
		# unique IBZ q's; mapping LORRAX's full-BZ q to those requires the
		# full crystal sym group.  A nosym WFN stores only identity in
		# mf_header/symmetry/mtrx, so allow pulling the 48 ops from an aux
		# sym-reduced WFN when provided (bgw_vcoul_sym_wfn).
		if config.bgw_vcoul_sym_wfn:
			aux_path = config.bgw_vcoul_sym_wfn
			if not os.path.isabs(aux_path):
				aux_path = os.path.join(input_dir, aux_path)
			with h5py.File(aux_path, "r") as _fsym:
				_sym_real = np.asarray(_fsym["mf_header/symmetry/mtrx"][:], dtype=np.int32)
			print0(f"    crystal sym ops loaded from {aux_path}: {_sym_real.shape[0]}")
			_sym_mats_k = _sym_real.transpose(0, 2, 1).copy()
		else:
			_sym_mats_k = np.asarray(sym.sym_mats_k, dtype=np.int32)
		def bgw_v_grid_fn(q_frac_tuple):
			return fill_v_grid_for_q(
				_bgw_table, q_frac_tuple, _fft_grid, _cell_vol,
				sym_mats_k=_sym_mats_k)

	# ISDF fitting or restart loading
	timing.reset()
	isdf = prepare_isdf_and_wavefunctions(
		cfg=config,
		wfn=wfn,
		sym=sym,
		meta=meta,
		centroid_indices=centroid_indices,
		band_slices=band_slices,
		mesh_xy=mesh_xy,
		tmp_dir=tmp_dir,
		tensors_filename=tensors_filename,
		print0=print0,
		bgw_v_grid_fn=bgw_v_grid_fn,
	)
	V_qmunu = isdf.V_qmunu
	wfns = isdf.wf_bundle

	# --- Screening: χ₀ → W = (1 − Vχ)⁻¹ V ---
	V_q = flatten_V_qmunu(V_qmunu)
	if config.do_screened:
		with timing.section("gw_jax.chi0_W"):
			with jax_profile.trace_section("chi0_W"):
				quad, e_ref = build_static_quadrature(wfns, config.minimax_config, print_fn=print0)
				chi0_q = compute_chi0(wfns, quad, meta, mesh_xy, energy_reference=e_ref)
				W_q = solve_w(V_q, chi0_q, meta, mesh_xy,
				              memory_mode=config.isdf_memory_mode)
				chi0_q.block_until_ready()
				W_q.block_until_ready()
				print0(f"  |χ(0)|_max = {float(jnp.max(jnp.abs(chi0_q))):.6e}  "
				       f"(minimax, {quad.node_count} nodes)")

	if not config.do_screened:
		W_q = V_q  # unscreened: W = V
	Gij = build_Gij(meta, mesh_xy)

	# q→0 head correction (exact band-diagonal terms for static COHSEX)
	static_head_terms = None
	if config.do_G0 and not config.use_ppm_sigma:
		static_head_terms = _compute_static_head(
			config, input_dir, wfn, sym, meta, config.do_screened, print0)

	# ---- Static COHSEX: Σ_SX, Σ_COH, V_H + bare Σ_X ----
	import gc; gc.collect()
	with timing.section("gw_jax.sigma"):
		cohsex = compute_cohsex_sigma(
			wfns, V_q, W_q, meta, mesh_xy,
			Gij=Gij, do_screened=config.do_screened,
			static_head_terms=static_head_terms,
			compute_bare_x=True,
		)
	sig_sx  = cohsex["sig_sx"]
	sig_coh = cohsex["sig_coh"]
	sig_h   = cohsex["sig_h"]
	sig_x   = cohsex["sig_x"]

	# Print bare Σ_X diagonal for ISDF quality assessment
	sig_x_diag = np.real(np.diagonal(np.asarray(sig_x), axis1=1, axis2=2)) * ryd2ev
	print0(f"  Bare Σ_X diagonal (eV), k=0: "
	       + "  ".join(f"{sig_x_diag[0, i]:.4f}" for i in range(min(8, sig_x_diag.shape[1]))))

	# ---- GN-PPM: replace static COH with frequency-integrated Σ^c ----
	sigma_omega_h5_path = None
	sigma_c_at_dft_ev = None
	sigma_xc_at_dft_ev = None
	omega_dft_rel_ev = None
	efermi_dft_ev = None
	if config.use_ppm_sigma:
		if not config.do_screened:
			raise ValueError("use_ppm_sigma=true requires do_screened=true.")
		if config.self_consistent:
			raise NotImplementedError("use_ppm_sigma not supported with self_consistent.")

		ppm_options = build_ppm_sigma_runtime_options(config, input_dir=input_dir, ryd2ev=ryd2ev)
		print_section("GN-PPM + FREQUENCY-INTEGRATED SIGMA", print0)

		with timing.section("gw_jax.ppm_sigma"):
			# χ₀(iωp) → W(iωp) → GN-PPM pole fit
			quad_imag = build_imag_quadrature(
				quad, config.ppm_omega_p, config.minimax_config, print_fn=print0)
			chi0_imag = compute_chi0(wfns, quad_imag, meta, mesh_xy, energy_reference=e_ref)
			Wiwp_q = solve_w(V_q, chi0_imag, meta, mesh_xy,
			                 memory_mode=config.isdf_memory_mode)
			chi0_imag.block_until_ready()
			Wiwp_q.block_until_ready()

			ppm = fit_gn_ppm(
				W_q, Wiwp_q, V_q, config.ppm_omega_p, mesh_xy,
				fallback_omega=config.ppm_fallback_omega,
				n_nodes_static=quad.node_count, print_fn=print0)

			# Frequency-integrated Σ^c(ω)
			# Temporary profiling hooks: pf.region + pre/post snapshots bracket the sigma PPM call
			try:
				import sys as _sys
				_sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
				import pf as _pf
				_pf_art = os.environ.get("PF_ARTIFACTS_DIR", "profile")
				_pf.snapshot_memory(f"{_pf_art}/memprof/sigma_ppm_pre.prof", label="sigma_ppm_pre")
			except Exception as _e:
				_pf = None
				print0(f"  [pf] profiling hooks unavailable: {_e}")
			_cm = _pf.region("sigma_ppm") if _pf is not None else timing.section("sigma_ppm_body")
			with _cm:
				sigma_omega = compute_ppm_sigma_omega_grid(
					wfns, ppm, meta, mesh_xy, ppm_options,
					sigma_window_quad=config.sigma_quadrature_config,
					print_fn=print0,
				)
			if _pf is not None:
				try:
					_pf.snapshot_memory(f"{_pf_art}/memprof/sigma_ppm_post.prof", label="sigma_ppm_post")
				except Exception:
					pass
			sigma_c_omega = sigma_omega.sigma_c_kij  # (n_omega, nk, nb, nb) or None if streamed

			# Evaluate Σ_c at DFT energies
			sigma_c_at_dft_ev = None
			sigma_xc_at_dft_ev = None
			omega_dft_rel_ev = None
			efermi_dft_ev = None
			if meta.rank == 0:
				enk_dft, _ = get_enk_bandrange(wfn, sym,
					band_slices.sigma_range,
					band_slices.sigma_range, nspinor=meta.nspinor)
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
			sigma_omega_h5_path = config.sigma_omega_h5_file
			if not os.path.isabs(sigma_omega_h5_path):
				sigma_omega_h5_path = os.path.join(input_dir, sigma_omega_h5_path)
			# SlabIO handles rank-0 dispatch internally; both backends
			# need all ranks to enter, so no `if meta.rank == 0:` guard.
			if sigma_c_omega is not None:
				write_sigma_omega_h5(
					sigma_omega_h5_path, ppm_options.omega_grid_ev, None,
					sigma_c_kij_ev=ryd2ev * sigma_c_omega,
					sigma_sx_kij_ev=ryd2ev * sig_sx,
					hartree_kij_ev=ryd2ev * sig_h,
					mesh=mesh_xy,
					use_ffi_io=getattr(config, "use_ffi_io", False))
			elif meta.rank == 0 and sigma_omega.sigma_kij_h5_path:
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
	kin_ion = load_kin_ion_submatrix(config.kin_ion_file, band_slices.b0, band_slices.b3)

	# PPM diagonal self-consistency and QSGW (rank 0 only)
	if config.use_ppm_sigma and meta.rank == 0 and sigma_omega_h5_path and os.path.exists(sigma_omega_h5_path):
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
		# In-grid test is against DFT energy (the Sigma(omega) grid is indexed
		# by omega = E - E_F and is only meaningful where E_DFT lies inside it).
		# QP eigenvalues from H_qp diagonalization are unreliable here because
		# pseudobands' non-unit-norm coefficients give eigvalsh(H_qp) garbage
		# values for the compressed high-energy states.
		E_dft_rel_ev = np.asarray(omega_dft_rel_ev, dtype=np.float64)
		E_qp_rel_ev = E_qp_ev - efermi
		in_grid = (E_dft_rel_ev >= omega_ev[0]) & (E_dft_rel_ev <= omega_ev[-1])
		n_in = int(np.count_nonzero(in_grid))

		if config.sigma_at_dft_extrapolate and np.any(in_grid) and np.any(~in_grid):
			# Scissor: ΔE_n = E_QP_n − E_DFT_n fit against E_DFT, separate
			# slopes/intercepts for valence and conduction.  Out-of-grid bands
			# get E_QP = E_DFT + (α·E_DFT + β) using the fitted line.
			from .scissor import fit_scissor
			nb_sc = E_sc.shape[1]
			occ_mask_kn = (np.arange(nb_sc)[None, :] < meta.nelec)
			occ_mask_kn = np.broadcast_to(occ_mask_kn, E_sc.shape).astype(bool)
			delta_e_measured = np.real(E_sc - E_dft_rel_ev)
			fit = fit_scissor(E_dft_rel_ev, delta_e_measured,
							  valence_mask_kn=occ_mask_kn,
							  fit_mask_kn=in_grid)
			print0(f"  Scissor fit: {fit.summary()}")
			extrap_rel = E_dft_rel_ev + fit.predict(E_dft_rel_ev, occ_mask_kn)
			E_sc = np.where(in_grid, E_sc, extrap_rel) + efermi
		else:
			E_sc = np.where(in_grid, E_sc, E_qp_rel_ev) + efermi
		print0(f"  Diagonal SC: {n_in}/{in_grid.size} states in grid, {n_iter} iterations")

		qsgw = build_qsgw_sigma_xc_from_h5(sigma_omega_h5_path,
			ryd2ev * np.array(sig_sx), omega_ev, E_sc - efermi)
		print0(f"  QSGW: {int(qsgw['n_interp_clipped'])} clipped "
			f"({100*qsgw['frac_interp_clipped']:.1f}%)")

	if config.self_consistent:
		# SC-COHSEX iteration — reuse the cached jit'd kernels directly so
		# the fixed-point driver can vary Gij.
		sigma_sx_k, sigma_coh_k, hartree_k = get_cohsex_kernels(meta, mesh_xy)
		def _add_head(a, b):
			return _add_static_head(
				a, b, static_head_terms=static_head_terms,
				meta=meta, mesh_xy=mesh_xy, do_screened=config.do_screened)
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
				sx_new = sigma_sx_k(wfns, Gij_new, W_q)
				coh_new = sigma_coh_k(wfns, W_q, V_q)
				h_new = hartree_k(wfns, Gij_new, V_q)
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
			sig_sx  = sigma_sx_k(wfns, Gij_final, W_q)
			sig_coh = sigma_coh_k(wfns, W_q, V_q)
			sig_h   = hartree_k(wfns, Gij_final, V_q)
		sig_sx, sig_coh = _add_head(sig_sx, sig_coh)
	else:
		H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)

	# ---- DFT and QP energies ----
	enk_dft, _ = get_enk_bandrange(wfn, sym,
		band_slices.sigma_range, band_slices.sigma_range, nspinor=meta.nspinor)

	# ---- Output ----
	results = GWResults(
		sig_sx=np.array(sig_sx),
		sig_coh=np.array(sig_coh),
		sig_h=np.array(sig_h),
		E_qp_ry=np.array(E_full),
		U_qp=np.array(U_full),
		E_dft_ry=np.array(enk_dft),
		kin_ion_ry=np.array(kin_ion),
		band_start=band_slices.b0,
		band_stop=band_slices.b3,
		use_ppm=config.use_ppm_sigma,
		self_consistent=config.self_consistent,
		sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
		sigma_omega_h5_path=sigma_omega_h5_path,
		tensors_filename=tensors_filename,
	)
	if meta.rank == 0:
		write_results(
			results,
			output_file=config.output_file,
			input_dir=input_dir,
			kpoints_crys=np.array(sym.unfolded_kpts, dtype=np.float64),
			kgrid=meta.kgrid,
			kpoints_reduced=np.array(wfn.kpoints, dtype=np.float64),
			kirr_to_kfull=np.array(sym.kirr_fullids, dtype=np.int32),
			print_fn=print0,
		)
	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

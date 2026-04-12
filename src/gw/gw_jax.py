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



# ================= ISDF sigma =================
#
# Sigma_SX  = -project[ FFT[ G_occ(R) * W(R) / sqrt(Nk) ] ]
# Sigma_COH = +project[ FFT[ G_RI(R) * (W-V)(R) / (2*sqrt(Nk)) ] ]
# V_H       = project[ V0 * rho ]

from .projection_kernel import project as _project

def _hartree(wfns, Gij, V_q, nk_tot):
	"""V_H(m,n,k) = <m| V(q=0,noG0) * rho |n>.  V_q is flat-k (nk,μ,μ); uses V_q[0]."""
	s = wfns.slices
	psi_yr, psi_xr = wfns.yr(s.sigma), wfns.xr(s.sigma)
	rho = jnp.real(jnp.einsum('kisx,kjsx,kij->x', jnp.conj(psi_yr), psi_yr, Gij, optimize=True))
	Vrho = jnp.einsum('xy,y->x', V_q[0], rho / jnp.asarray(nk_tot, dtype=jnp.float64), optimize=True)
	return jnp.einsum('kmsx,x,knsx->kmn', jnp.conj(psi_xr), Vrho, psi_xr, optimize=True)

from .greens_function_kernel import build_G


def _build_Gij(meta, mesh_xy):
	"""Occupation projector G_ij = diag(1,...,1,0,...,0) for sigma bands."""
	nocc = min(meta.nelec, meta.nb_sigma)
	Gij = jnp.zeros((meta.nk_tot, meta.nb_sigma, meta.nb_sigma), dtype=jnp.complex128)
	Gij = Gij.at[:, :nocc, :nocc].set(jnp.eye(nocc, dtype=jnp.complex128))
	return jax.device_put(Gij, NamedSharding(mesh_xy, P(None, None, None)))


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
	
	# Gate prints to rank 0 during main
	_orig_print = builtins.print
	def _print0(*a, **k):
		if jax.process_index() == 0:
			k.setdefault("flush", True)
			_orig_print(*a, **k)
	builtins.print = _print0
 
	# ========================================================================
	# CONFIGURATION
	# ========================================================================
	config = LorraxConfig.from_input_file(args.input, print_fn=_print0)
	input_dir = config.input_dir
	ryd2ev = 13.6056980659

	# ========================================================================
	# INITIALIZATION
	# ========================================================================
	current_backend = jax.default_backend()
	n_devices = len(jax.devices())
	n_procs = jax.process_count()
	device_names = jax.devices()[0].device_kind if n_devices > 0 else "unknown"

	total_devices = jax.process_count() * jax.local_device_count()
	grid_x = int(np.sqrt(total_devices))
	while total_devices % grid_x != 0:
		grid_x -= 1
	grid_y = total_devices // grid_x
	devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
	mesh_xy = Mesh(devices_2d, ['x', 'y'])

	from .gw_output import print_banner, print_system_summary, write_results, GWResults
	print_banner(
		backend=current_backend, n_devices=n_devices,
		grid_x=grid_x, grid_y=grid_y, n_procs=n_procs,
		device_kind=device_names, print_fn=_print0,
	)

	import time as _boot_time
	_t_boot = _boot_time.perf_counter()
	_print0(f"  [TIMING-BOOT] input parsed: {_boot_time.perf_counter() - _t_boot:.3f}s")

	global wfn
	_t_wfn = _boot_time.perf_counter()
	wfn = WFNReader(config.wfn_file)
	_print0(f"  [TIMING-BOOT] WFNReader: {_boot_time.perf_counter() - _t_wfn:.3f}s")
	_t_sym = _boot_time.perf_counter()
	sym = symmetry_maps.SymMaps(wfn)
	_print0(f"  [TIMING-BOOT] SymMaps: {_boot_time.perf_counter() - _t_sym:.3f}s")

	_t_cent = _boot_time.perf_counter()
	_, centroid_indices, _n_rmu = load_centroids(config.centroids_file, wfn.fft_grid)
	_print0(f"  [TIMING-BOOT] load_centroids: {_boot_time.perf_counter() - _t_cent:.3f}s")
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{_n_rmu}.h5")
	print_system_summary(
		n_rmu=_n_rmu, fft_grid=wfn.fft_grid,
		cell_volume=wfn.cell_volume, print_fn=_print0,
	)

	meta = Meta.from_system(wfn, sym, config.nval, config.ncond, config.nband, _n_rmu, config.bispinor)
	meta.rank = jax.process_index()
	meta.n_proc = jax.process_count()
	meta.sys_dim = config.sys_dim
	meta.bispinor = config.bispinor
	meta.chunk_size = get_effective_chunk_size(config.chunk_size)

	band_slices = BandSlices.from_band_edges(*meta.band_edges)

	def print0(*a, **k):
		if meta.rank == 0:
			k.setdefault("flush", True)
			print(*a, **k)

	# ISDF fitting or restart loading
	timing.reset()
	_t_isdf = _boot_time.perf_counter()
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
	)
	print0(f"  [TIMING-BOOT] prepare_isdf_and_wavefunctions: {_boot_time.perf_counter() - _t_isdf:.3f}s")
	V_qmunu = isdf.V_qmunu
	wfns = isdf.wf_bundle

	# --- Screening: χ₀ → W = (1 − Vχ)⁻¹ V ---
	V_q = flatten_V_qmunu(V_qmunu)
	if config.do_screened:
		with timing.section("gw_jax.chi0_W"):
			with jax_profile.trace_section("chi0_W"):
				quad, e_ref = build_static_quadrature(wfns, config.minimax_config, print_fn=print0)
				chi0_q = compute_chi0(wfns, quad, meta, mesh_xy, energy_reference=e_ref)
				W_q = solve_w(V_q, chi0_q, meta, mesh_xy)
				chi0_q.block_until_ready()
				W_q.block_until_ready()
				print0(f"  |χ(0)|_max = {float(jnp.max(jnp.abs(chi0_q))):.6e}  "
				       f"(minimax, {quad.node_count} nodes)")

	if not config.do_screened:
		W_q = V_q  # unscreened: W = V
	Gij = _build_Gij(meta, mesh_xy)

	# q→0 head correction (exact band-diagonal terms for static COHSEX)
	static_head_terms = None
	if config.do_G0 and not config.use_ppm_sigma:
		static_head_terms = _compute_static_head(
			config, input_dir, wfn, sym, meta, config.do_screened, print0)

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
	def _convolve(G_k, V_or_W, prefactor):
		G_R = _G_ifftn(G_k)
		V_R = _V_ifftn(V_or_W)[:, None, :, None, :]
		return prefactor * _G_fftn(G_R * V_R * _inv_sqrt_nk)

	# ---- Static COHSEX: Σ_SX, Σ_COH, V_H ----

	@jax.jit
	def sigma_sx(wfns, Gij, W_q):
		s = wfns.slices
		G_occ = build_G(wfns.xn(s.sigma), wfns.yr(s.sigma), Gij=Gij)
		return _project(wfns.xr(s.sigma), wfns.yn(s.sigma), _convolve(G_occ, W_q, 1.0))

	@jax.jit
	def sigma_coh(wfns, W_q, V_q):
		s = wfns.slices
		G_ri = build_G(wfns.xn(s.full), wfns.yr(s.full))
		return _project(wfns.xr(s.sigma), wfns.yn(s.sigma), _convolve(G_ri, W_q - V_q, -0.5))

	@jax.jit
	def hartree(wfns, Gij, V_q):
		return _hartree(wfns, Gij, V_q, _nk_tot)

	def _add_head(sig_sx, sig_coh):
		if static_head_terms is None:
			return sig_sx, sig_coh
		sx_h, coh_h = static_head_terms_to_kij(
			static_head_terms, nk_tot=meta.nk_tot, do_screened=config.do_screened)
		if not config.do_screened:
			coh_h = jnp.zeros_like(coh_h)
		rep = NamedSharding(mesh_xy, P(None, None, None))
		return sig_sx + jax.device_put(sx_h, rep), sig_coh + jax.device_put(coh_h, rep)

	# ---- Execute static COHSEX ----
	import gc; gc.collect()
	with mesh_xy:
		with timing.section("gw_jax.sigma"):
			sig_sx  = sigma_sx(wfns, Gij, W_q)
			sig_coh = sigma_coh(wfns, W_q, V_q)
			sig_h   = hartree(wfns, Gij, V_q)

			sig_sx, sig_coh = _add_head(sig_sx, sig_coh)

			sig_sx.block_until_ready()
			sig_coh.block_until_ready()
			sig_h.block_until_ready()

	# Bare exchange Σ_X (used for both COHSEX comparison and PPM Σ^c = Σ_xc - Σ_X)
	with mesh_xy:
		sig_x = sigma_sx(wfns, Gij, V_q)
	sig_x.block_until_ready()

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
		print0("\n" + "-" * 72 + "\n  GN-PPM + FREQUENCY-INTEGRATED SIGMA\n" + "-" * 72)

		with timing.section("gw_jax.ppm_sigma"):
			# χ₀(iωp) → W(iωp) → GN-PPM pole fit
			quad_imag = build_imag_quadrature(
				quad, config.ppm_omega_p, config.minimax_config, print_fn=print0)
			chi0_imag = compute_chi0(wfns, quad_imag, meta, mesh_xy, energy_reference=e_ref)
			Wiwp_q = solve_w(V_q, chi0_imag, meta, mesh_xy)
			chi0_imag.block_until_ready()
			Wiwp_q.block_until_ready()

			ppm = fit_gn_ppm(
				W_q, Wiwp_q, V_q, config.ppm_omega_p, mesh_xy,
				fallback_omega=config.ppm_fallback_omega,
				n_nodes_static=quad.node_count, print_fn=print0)

			# Frequency-integrated Σ^c(ω)
			sigma_omega = compute_sigma_c_ppm_omega_grid(
				wfns, ppm, meta, mesh_xy, ppm_options,
				sigma_window_quad=config.sigma_quadrature_config,
				print_fn=print0,
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
			sigma_omega_h5_path = config.sigma_omega_h5_file
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
	kin_ion = load_kin_ion_submatrix(config.kin_ion_file, meta.band_edges[0], meta.band_edges[3])

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
		in_grid = (E_qp_ev * ryd2ev - efermi >= omega_ev[0]) & (E_qp_ev * ryd2ev - efermi <= omega_ev[-1])
		E_sc = np.where(in_grid, E_sc, E_qp_ev - efermi) + efermi
		n_in = int(np.count_nonzero(in_grid))
		print0(f"  Diagonal SC: {n_in}/{in_grid.size} states in grid, {n_iter} iterations")

		qsgw = build_qsgw_sigma_xc_from_h5(sigma_omega_h5_path,
			ryd2ev * np.array(sig_sx), omega_ev, E_sc - efermi)
		print0(f"  QSGW: {int(qsgw['n_interp_clipped'])} clipped "
			f"({100*qsgw['frac_interp_clipped']:.1f}%)")

	if config.self_consistent:
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
				sx_new = sigma_sx(wfns, Gij_new, W_q)
				coh_new = sigma_coh(wfns, W_q, V_q)
				h_new = hartree(wfns, Gij_new, V_q)
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
			sig_sx  = sigma_sx(wfns, Gij_final, W_q)
			sig_coh = sigma_coh(wfns, W_q, V_q)
			sig_h   = hartree(wfns, Gij_final, V_q)
		sig_sx, sig_coh = _add_head(sig_sx, sig_coh)
	else:
		H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)

	# ---- DFT and QP energies ----
	b0, b3 = meta.band_edges[0], meta.band_edges[3]
	enk_dft, _ = get_enk_bandrange(wfn, sym, (b0, b3), (b0, b3), nspinor=meta.nspinor)

	# ---- Output ----
	results = GWResults(
		sig_sx=np.array(sig_sx),
		sig_coh=np.array(sig_coh),
		sig_h=np.array(sig_h),
		E_qp_ry=np.array(E_full),
		U_qp=np.array(U_full),
		E_dft_ry=np.array(enk_dft),
		kin_ion_ry=np.array(kin_ion),
		band_start=b0,
		band_stop=b3,
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
			kgrid=(meta.nkx, meta.nky, meta.nkz),
			kpoints_reduced=np.array(wfn.kpoints, dtype=np.float64),
			kirr_to_kfull=np.array(sym.kirr_fullids, dtype=np.int32),
			print_fn=print0,
		)
	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

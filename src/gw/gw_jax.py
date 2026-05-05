from runtime import set_default_env
set_default_env()  # BEFORE `import jax` — JAX reads env at import time.

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update("jax_enable_x64", True)

from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()

# Back-compat shim: a few sandbox scripts import this name.  Prefer
# ``runtime.init_jax_distributed`` in new code.
_maybe_init_jax_distributed = init_jax_distributed
from file_io import (
    WFNReader, write_sigma_omega_h5,
    load_kin_ion_submatrix, load_centroids,
)
from common import symmetry_maps
from common.load_wfns import get_enk_bandrange
from .gw_config import ComputeMode, LorraxConfig
from .gw_init import (
	get_effective_chunk_size,
	prepare_isdf_and_wavefunctions,
)
from .gw_driver_helpers import (
	build_bgw_v_grid_fn,
	setup_runtime,
)
from .w_isdf import (
	build_static_quadrature,
	compute_chi0,
	flatten_V_qmunu,
	precompile_chi0,
	precompile_solve_w,
	solve_w,
)
from .ppm_pipeline import compute_ppm_sigma_pipeline
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
	HeadResolver,
	compute_static_head_terms_from_sample,
	format_head_sample_diagnostics,
	format_static_head_diagnostics,
)
from .wavefunction_bundle import BandSlices
from mixing.acceleration import (
    rcrop_nojit, hermitian_to_upper_flat, upper_flat_to_hermitian
)
from common import Meta, RYD_TO_EV
from common import jax_profile
import common.timing as timing



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


def _compute_static_head(head_resolver, meta, do_screened, print0):
	"""Resolve q→0 head and compute exact band-diagonal head terms for COHSEX."""
	head = head_resolver.at(0.0 + 0.0j)
	print0(format_head_sample_diagnostics(head, include_screened=do_screened))
	occ_mask = np.arange(meta.nb_sigma, dtype=np.int32) < meta.nelec
	terms = compute_static_head_terms_from_sample(
		head, occ=occ_mask, cell_volume=meta.cell_volume, nk_tot=meta.nk_tot)
	print0(format_static_head_diagnostics(terms))
	return terms


def main(argv=None):
	argp = argparse.ArgumentParser(description="COHSEX self-energy driver")
	argp.add_argument(
		"-i",
		"--input",
		default="cohsex_test.in",
		help="Input file",
	)
	args = argp.parse_args(argv)

	# Rank-gated print used as ``print_fn=`` throughout the driver.  We do
	# NOT clobber ``builtins.print`` — that historically affected every
	# imported library, including ones that legitimately want to write
	# from non-zero ranks (logging, error paths).
	def print0(*a, **k):
		if jax.process_index() == 0:
			k.setdefault("flush", True)
			print(*a, **k)

	# ========================================================================
	# CONFIGURATION
	# ========================================================================
	config = LorraxConfig.from_input_file(args.input, print_fn=print0)
	input_dir = config.input_dir

	# ========================================================================
	# INITIALIZATION
	# ========================================================================
	current_backend = jax.default_backend()
	n_devices = len(jax.devices())
	n_procs = jax.process_count()
	device_names = jax.devices()[0].device_kind if n_devices > 0 else "unknown"

	mesh_xy = _build_mesh()
	grid_x, grid_y = mesh_xy.devices.shape

	setup_runtime(config, mesh_xy, print_fn=print0)

	from .gw_output import print_banner, print_section, print_system_summary, write_results, GWResults
	print_banner(
		backend=current_backend, n_devices=n_devices,
		grid_x=grid_x, grid_y=grid_y, n_procs=n_procs,
		device_kind=device_names, print_fn=print0,
	)

	wfn = WFNReader(config.paths.wfn_file)
	sym = symmetry_maps.SymMaps(wfn)
	_, centroid_indices, _n_rmu = load_centroids(config.paths.centroids_file, wfn.fft_grid)
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
	meta.chunk_size = get_effective_chunk_size(config.memory.chunk_size)

	band_slices = BandSlices.from_band_edges(*meta.band_edges)

	# Single resolver for every q→0 head sample we'll need this run; the
	# COHSEX static head, the W0 restart-flush head, and the PPM dynamic
	# head all read from the same plumbing (overrides → epshead → s_tensor)
	# so they share one cache.  See ``head_correction.HeadResolver``.
	head_resolver = HeadResolver(config, input_dir, wfn, sym, meta, print0)

	# Optional BGW vcoul override (purely diagnostic — bit-reproducible BGW
	# comparisons).  Returns None when ``use_bgw_vcoul`` is False.
	bgw_v_grid_fn = build_bgw_v_grid_fn(
		config, wfn=wfn, sym=sym, input_dir=input_dir, print_fn=print0)

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
				# Split compile vs exec for χ₀ and W.  Each section's
				# wall time is read off the end-of-run timing report
				# under ``gw_jax.chi0_W.{chi,W}.{compile,exec}``.  The
				# explicit ``block_until_ready`` inside the exec sections
				# is load-bearing: it (a) pins chi.exec / W.exec wall time
				# to the actual dispatched compute (not just the host
				# dispatch), and (b) drops the last Python reference to
				# χ₀ before the W-solve call so XLA can donate that
				# buffer.  Do NOT use ``_chi_sec.watch(...)`` here — it
				# keeps a bound ``block_until_ready`` method alive on the
				# section object past W-solve, which blocks donation.
				quad, e_ref = build_static_quadrature(wfns, config.minimax_config, print_fn=print0)
				with timing.section("chi.compile"):
					precompile_chi0(wfns, quad, meta, mesh_xy,
					                energy_reference=e_ref)
				with timing.section("chi.exec"):
					chi0_q = compute_chi0(wfns, quad, meta, mesh_xy,
					                      energy_reference=e_ref)
					chi0_q.block_until_ready()
				# isdf_memory_mode is the legacy alias for backend.screening_solver
				with timing.section("W.compile"):
					precompile_solve_w(V_q, chi0_q, meta, mesh_xy,
					                   memory_mode=config.isdf_memory_mode)
				with timing.section("W.exec"):
					W_q = solve_w(V_q, chi0_q, meta, mesh_xy,
					              memory_mode=config.isdf_memory_mode)
					# χ₀ is donated inside solve_w — the reference is
					# now invalid.  Do NOT touch ``chi0_q`` after this.
					del chi0_q
					W_q.block_until_ready()

	if not config.do_screened:
		W_q = V_q  # unscreened: W = V

	# ── Persist W0_qmunu + q=0 head scalars to the restart file ──────────
	# Downstream consumers (BSE, future Σ-builders) reload these and apply
	# the rank-1 head update via ``head_correction.apply_q0_head_rank1``.
	# The ``whead`` axis is length 1 for COHSEX (just static) and length 2
	# for GN-PPM (static + iω_p).  ``vhead``/``whead_*`` cohsex.in overrides
	# flow through automatically because ``HeadResolver`` consults the
	# config's override fields first before falling back to s_tensor/epshead.
	if config.do_screened and os.path.exists(tensors_filename):
		from file_io import write_w0_qmunu_to_h5, write_head_scalars_to_h5
		nkx, nky, nkz = (int(x) for x in meta.kgrid)
		W_q_8d = W_q.reshape(1, 1, 1, nkx, nky, nkz, W_q.shape[-2], W_q.shape[-1])
		write_w0_qmunu_to_h5(tensors_filename, W_q_8d,
		                     mesh=mesh_xy, use_ffi_io=config.use_ffi_io)
		head_static = head_resolver.at(0.0 + 0.0j)
		if config.compute_mode.is_dynamic:
			# GN-PPM: probe at iωp on the imaginary axis.
			# HL-PPM: probe at Ω on the real axis (above all transitions).
			if config.compute_mode is ComputeMode.HL_PPM:
				omega_imp = complex(float(config.ppm.omega_p), 0.0)
				_omega_grid_entry = float(omega_imp.real)
			else:
				omega_imp = 1j * float(config.ppm.omega_p)
				_omega_grid_entry = float(omega_imp.imag)
			head_imag = head_resolver.at(omega_imp)
			whead_arr = np.array(
				[head_static.wcoul0, head_imag.wcoul0], dtype=np.complex128)
			omega_grid = np.array([0.0, _omega_grid_entry], dtype=np.float64)
		else:
			whead_arr = np.array([head_static.wcoul0], dtype=np.complex128)
			omega_grid = np.array([0.0], dtype=np.float64)
		write_head_scalars_to_h5(
			tensors_filename,
			vhead=complex(head_static.vc0),
			whead=whead_arr,
			omega_grid=omega_grid,
		)
		print0(
			f"  Persisted W0_qmunu + q=0 head scalars: "
			f"vhead={head_static.vc0.real:.3f} a.u.,  "
			f"whead[ω=0]={whead_arr[0].real:.3f} a.u."
			+ (f",  whead[iωp]={whead_arr[1].real:.3f} a.u." if len(whead_arr) > 1 else "")
		)

	Gij = build_Gij(meta, mesh_xy)

	# q→0 head correction.  The bare-X head is the same physical quantity in
	# both COHSEX and PPM modes; gating this on ``not use_ppm_sigma`` was
	# the original ``Bare Σ_X missing q→0 head'' bug (skill compare/SKILL.md
	# §4i).  The SX/COH head pieces are also attached to the static
	# sig_sx/sig_coh in compute_cohsex_sigma, but for PPM those static values
	# are overwritten downstream (sig_sx ← sig_x, sig_c ← PPM-evaluated
	# correlation), so only the X-head survives — which is the piece needed.
	static_head_terms = None
	if config.do_G0:
		static_head_terms = _compute_static_head(
			head_resolver, meta, config.do_screened, print0)

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

	# Print bare Σ_X diagonal for ISDF quality assessment.  Apply BGW-style
	# degenerate-set averaging (mirrors Sigma/shiftenergy.f90) unless
	# explicitly disabled via ``no_degen_averaging``.  Without this, the
	# QE basis-dependent splitting within degenerate manifolds shows up
	# as a few-meV spread across symmetry-equivalent bands.
	from .degen_average import average_within_degenerate_sets
	sig_x_diag = np.real(np.diagonal(np.asarray(sig_x), axis1=1, axis2=2)) * RYD_TO_EV
	if not config.no_degen_averaging:
		_enk_sigma_ry, _ = get_enk_bandrange(
			wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
			nspinor=meta.nspinor)
		sig_x_diag = average_within_degenerate_sets(
			sig_x_diag,
			energies_kn_ry=np.asarray(_enk_sigma_ry, dtype=np.float64),
			tol_ry=float(config.degen_avg_tol_ry),
		)
	print0(f"  Bare Σ_X diagonal (eV), k=0: "
	       + "  ".join(f"{sig_x_diag[0, i]:.4f}" for i in range(min(8, sig_x_diag.shape[1]))))

	# ---- Mode-pivoted dispatch ----
	# ``compute_mode`` is the single axis describing the self-energy ansatz
	# (X_ONLY / COHSEX / GN_PPM / HL_PPM); ``self_consistent`` is orthogonal.
	# Static COHSEX matrices were already computed above (the bare-X pass
	# reuses the same kernel), so X_ONLY and COHSEX both fall through with
	# the existing sig_sx / sig_coh / sig_x.  Dynamic modes go through the
	# PPM pipeline.
	#
	# History note (kept here because it explains a specific decision and
	# is not yet captured anywhere else): the analytic q→0 head injected
	# at the end of ``compute_ppm_sigma_pipeline`` was re-added in
	# 2026-04-25 after being removed in 1542342 (Apr-10).  Magnitude is
	# ±W^c(0)/(2·V_cell·N_k) on-shell — ~1.24 eV/band on Si 4×4×4 60b.
	# See reports/mos2_kgrid_gnppm_head_convergence_2026-4-10/.
	mode = config.compute_mode
	ppm_outputs = None
	if mode.is_dynamic:
		ppm_outputs = compute_ppm_sigma_pipeline(
			wfns=wfns,
			V_q=V_q, W_q=W_q, sig_x=sig_x, sig_h=sig_h,
			quad=quad, e_ref=e_ref,
			config=config, meta=meta, mesh_xy=mesh_xy,
			head_resolver=head_resolver,
			band_slices=band_slices, wfn=wfn, sym=sym,
			input_dir=input_dir,
			print_fn=print0,
		)
	sigma_omega_h5_path = ppm_outputs.sigma_omega_h5_path if ppm_outputs else None
	sigma_c_at_dft_ev   = ppm_outputs.sigma_c_at_dft_ev   if ppm_outputs else None
	sigma_xc_at_dft_ev  = ppm_outputs.sigma_xc_at_dft_ev  if ppm_outputs else None
	omega_dft_rel_ev    = ppm_outputs.omega_dft_rel_ev    if ppm_outputs else None
	efermi_dft_ev       = ppm_outputs.efermi_dft_ev       if ppm_outputs else None
	# ω-grid Σ_c diagonal for the BGW eqp1.dat Z-factor (PPM modes only).
	# ppm_outputs.sigma_c_omega is (n_omega, nk_full, nb, nb) Ry, post-head;
	# we hand the diagonal in eV to the writer which then central-diffs it.
	if ppm_outputs is not None and ppm_outputs.sigma_c_omega is not None:
		sigma_c_omega_diag_ev = (
			np.diagonal(np.asarray(ppm_outputs.sigma_c_omega),
			            axis1=2, axis2=3) * RYD_TO_EV
		)
		omega_rel_ev = np.asarray(ppm_outputs.ppm_options.omega_grid_ev)
	else:
		sigma_c_omega_diag_ev = None
		omega_rel_ev = None
	ppm_options         = ppm_outputs.ppm_options         if ppm_outputs else None

	# ---- QP Hamiltonian: H_QP = (H_DFT - V_xc) + V_H + Σ_xc ----
	sigma_total = sig_sx + sig_coh + sig_h
	kin_ion = load_kin_ion_submatrix(config.paths.kin_ion_file, band_slices.b0, band_slices.b3)

	# Dynamic-mode diagonal self-consistency + QSGW (rank 0 only)
	if mode.is_dynamic and meta.rank == 0 and sigma_omega_h5_path and os.path.exists(sigma_omega_h5_path):
		H_qp = np.array(kin_ion + sigma_total)
		H_qp = 0.5 * (H_qp + np.conj(np.swapaxes(H_qp, -1, -2)))
		E_qp_ev = np.linalg.eigvalsh(H_qp) * RYD_TO_EV

		# Diagonal fixed-point: E = diag(H0) + Re Σ_xc(E)
		h0_diag_ev = np.real(np.diagonal(np.array(kin_ion + sig_h), axis1=1, axis2=2)) * RYD_TO_EV
		occ_idx = min(meta.nelec, E_qp_ev.shape[1] - 1)
		vbm = float(np.max(E_qp_ev[:, occ_idx]))
		cbm = float(np.min(E_qp_ev[:, occ_idx + 1])) if occ_idx + 1 < E_qp_ev.shape[1] else vbm
		efermi = 0.5 * (vbm + cbm)
		omega_ev = np.asarray(ppm_options.omega_grid_ev, dtype=np.float64)
		# PPM mode: Σ_xc(ω) = Σ_x_bare + Σ_c(ω); the H5 file holds Σ_c only.
		sigma_xc_diag = np.real(load_sigma_xc_diag_from_h5(sigma_omega_h5_path, RYD_TO_EV * np.array(sig_x)))
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

		if config.ppm.sigma_at_dft_extrapolate and np.any(in_grid) and np.any(~in_grid):
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
			RYD_TO_EV * np.array(sig_x), omega_ev, E_sc - efermi)
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

	# PPM mode: feed the writer the on-shell diag(Σ_c(E_DFT)) (Ry) so the
	# eqp0.dat "sigC" column reports dynamic correlation directly comparable
	# to BGW's (SX-X)+CH at Eo=E_DFT.  Off-diagonals stay zero — the full
	# Σ_c(ω, k, i, j) tensor is in sigma_mnk.h5 for callers that need them.
	sigma_c_diag_at_dft_ry = (
		sigma_c_at_dft_ev / RYD_TO_EV
		if (mode.is_dynamic and meta.rank == 0 and sigma_c_at_dft_ev is not None)
		else None
	)

	# ---- Output ----
	results = GWResults(
		sig_sx=np.array(sig_sx),
		sig_coh=np.array(sig_coh),
		sig_h=np.array(sig_h),
		sig_x=np.array(sig_x),
		E_qp_ry=np.array(E_full),
		U_qp=np.array(U_full),
		E_dft_ry=np.array(enk_dft),
		kin_ion_ry=np.array(kin_ion),
		band_start=band_slices.b0,
		band_stop=band_slices.b3,
		use_ppm=mode.is_dynamic,
		self_consistent=config.self_consistent,
		sigma_c_diag_at_dft_ry=sigma_c_diag_at_dft_ry,
		sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
		sigma_c_omega_diag_ev=sigma_c_omega_diag_ev,
		omega_rel_ev=omega_rel_ev,
		sigma_omega_h5_path=sigma_omega_h5_path,
		tensors_filename=tensors_filename,
	)
	if meta.rank == 0:
		write_results(
			results,
			sigma_diag_file=config.paths.sigma_diag_file,
			eqp0_file=config.paths.eqp0_file,
			eqp1_file=config.paths.eqp1_file,
			input_dir=input_dir,
			kpoints_crys=np.array(sym.unfolded_kpts, dtype=np.float64),
			kgrid=meta.kgrid,
			kpoints_irr_frac=np.array(wfn.kpoints, dtype=np.float64),
			kpoints_reduced=np.array(wfn.kpoints, dtype=np.float64),
			kirr_to_kfull=np.array(sym.kirr_fullids, dtype=np.int32),
			print_fn=print0,
			no_degen_averaging=config.no_degen_averaging,
			degen_avg_tol_ry=config.degen_avg_tol_ry,
		)
	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

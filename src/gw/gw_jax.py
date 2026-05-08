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
	build_qsgw_sigma_xc,
	extract_sigma_diag_replicated,
	print_scf_diagnostics,
	solve_diagonal_sigma_fixed_point,
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
	# V_qmunu is already flat-q ``(nq, μ, μ)``; downstream consumers
	# bind ``V_q`` to that array directly.  ``flatten_V_qmunu`` is kept
	# as a back-compat no-op for restart paths that may still feed in
	# the legacy 8-D layout.
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
				with timing.section("W.compile"):
					precompile_solve_w(V_q, chi0_q, meta, mesh_xy,
					                   solver=config.backend.screening_solver)
				with timing.section("W.exec"):
					W_q = solve_w(V_q, chi0_q, meta, mesh_xy,
					              solver=config.backend.screening_solver)
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
		                     mesh=mesh_xy, backend=config.backend.slab_io)
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
	ppm_options         = ppm_outputs.ppm_options         if ppm_outputs else None

	# ---- QP Hamiltonian: H_QP = (H_DFT - V_xc) + V_H + Σ_xc ----
	#
	# Sharding: kin_ion + all four static-COHSEX components (Σ_SX, Σ_COH,
	# V_H, Σ_X) are loaded **fully replicated** on the device mesh.  The
	# only sharded object surviving past this point is the dynamic-Σ_c
	# ω-tensor produced by ``compute_ppm_sigma_pipeline``, which is
	# collapsed into a replicated Σ_xc^QSGW by ``build_qsgw_sigma_xc``
	# below.  This lets the rest of post-processing (eigh, scissor,
	# eqp output) operate on replicated arrays without resharding seams.
	kin_ion = load_kin_ion_submatrix(
		config.paths.kin_ion_file, band_slices.b0, band_slices.b3,
		mesh=mesh_xy, backend=config.backend.slab_io,
	)

	# ---- Mode-pivoted Σ_xc dispatch.  All branches yield ``sigma_total``
	# replicated on the mesh as Σ_xc + V_H (Ry).
	#
	# History note (kept here because it explains a specific decision and
	# is not yet captured anywhere else): the analytic q→0 head injected
	# at the end of ``compute_ppm_sigma_pipeline`` was re-added in
	# 2026-04-25 after being removed in 1542342 (Apr-10).  Magnitude is
	# ±W^c(0)/(2·V_cell·N_k) on-shell — ~1.24 eV/band on Si 4×4×4 60b.
	# See reports/mos2_kgrid_gnppm_head_convergence_2026-4-10/.
	E_sc_ev = None
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

		sigma_total = sig_sx + sig_coh + sig_h
		result = rcrop_nojit(
			lambda x: _sc_step(x) - x,
			hermitian_to_upper_flat(sigma_total).flatten(),
			m=3, maxit=40, tol=1e-5,
			print_fn=print0 if meta.rank == 0 else None)
		sigma_total = upper_flat_to_hermitian(result.x.reshape(nk, n_upper), meta.nb_sigma)

		# Final sigma components from converged Gij
		H_for_diag = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
		_, U_diag = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_for_diag)
		f = (jnp.arange(meta.nb_sigma) < meta.nelec).astype(jnp.float64)
		Gij_final = jnp.einsum('kim,m,kjm->kij', U_diag, f, jnp.conj(U_diag), optimize=True)
		print_scf_diagnostics(Gij_final, U_diag, meta.nelec, meta.nb_sigma, print0)
		with mesh_xy:
			sig_sx  = sigma_sx_k(wfns, Gij_final, W_q)
			sig_coh = sigma_coh_k(wfns, W_q, V_q)
			sig_h   = hartree_k(wfns, Gij_final, V_q)
		sig_sx, sig_coh = _add_head(sig_sx, sig_coh)
		sigma_total = sig_sx + sig_coh + sig_h
	elif mode.is_dynamic and ppm_outputs is not None and ppm_outputs.sigma_c_omega is not None:
		# G0W0/QSGW: diagonal-Σ(E) fixed point (with optional scissor) →
		# QSGW Σ_xc^QSGW.  Restart-friendly: this whole block consumes only
		# the on-device ``sigma_c_omega`` plus replicated (sig_x, sig_h),
		# so a future outer QSGW iteration loop can pass refreshed inputs
		# in without touching the disk.
		# All quantities below are in **Rydberg** until the scissor's print
		# summary and final eV outputs.  Σ_c(ω) lives natively in Ry on the
		# Ry ω-grid; mixing that with eV-converted h0/Σ_x is a footgun.
		omega_grid_ry = np.asarray(ppm_options.omega_grid_ry, dtype=np.float64)

		# Diagonal Σ_c(ω, k, n) and Σ_x(k, n) replicated on host, in Ry.
		sigma_c_diag_w_kn_ry = np.asarray(extract_sigma_diag_replicated(
			ppm_outputs.sigma_c_omega, mesh_xy))
		sigma_x_diag_kn_ry = np.real(
			np.diagonal(np.asarray(sig_x), axis1=1, axis2=2))
		sigma_xc_diag_w_kn_ry = sigma_c_diag_w_kn_ry + sigma_x_diag_kn_ry[None, :, :]

		# Diagonal Σ(E) fixed point in Ry.
		h0_diag_ry = np.real(
			np.diagonal(np.asarray(kin_ion + sig_h), axis1=1, axis2=2))
		efermi_ry = float(efermi_dft_ev) / RYD_TO_EV
		E_sc_rel_ry, _, n_iter = solve_diagonal_sigma_fixed_point(
			h0_diag_ry - efermi_ry, sigma_xc_diag_w_kn_ry, omega_grid_ry,
			max_iter=120, tol_ev=1.0e-7 / RYD_TO_EV, mixing=0.6,
		)

		# Per-band scissor for out-of-grid bands.  A band is "in-grid" iff
		# E_DFT[k, n] lies in [ω_min, ω_max] for every k; if any single k
		# is outside, the band gets the scissor uniformly across k (the
		# diagonal solver clipped Σ_c at the ω-boundary for the offending
		# k, which would otherwise contaminate the band's k-dispersion).
		# The scissor itself is fitted on in-grid bands only.  Default
		# fallback when the scissor flag is off: E_DFT (the natural
		# zeroth-order QP correction = 0 estimate); the older fallback
		# of using ``eigvalsh(H_qp)`` was unreliable for pseudobands.
		from .scissor import classify_bands_in_grid, fit_scissor
		E_dft_rel_ry = np.asarray(omega_dft_rel_ev, dtype=np.float64) / RYD_TO_EV
		band_in_grid, in_grid_kn_band = classify_bands_in_grid(
			E_dft_rel_ry, float(omega_grid_ry[0]), float(omega_grid_ry[-1]))
		n_bands_in = int(band_in_grid.sum())
		n_bands_total = int(band_in_grid.size)
		print0(
			f"  Diagonal SC: {n_bands_in}/{n_bands_total} bands fully in grid, "
			f"{n_iter} iterations")
		if (
			config.ppm.sigma_at_dft_extrapolate
			and 0 < n_bands_in < n_bands_total
		):
			occ_mask_kn = np.broadcast_to(
				np.arange(E_sc_rel_ry.shape[1])[None, :] < meta.nelec,
				E_sc_rel_ry.shape).astype(bool)
			# Fit in eV so the printed slopes/intercepts are human-readable.
			fit = fit_scissor(
				E_dft_rel_ry * RYD_TO_EV,
				np.real(E_sc_rel_ry - E_dft_rel_ry) * RYD_TO_EV,
				valence_mask_kn=occ_mask_kn,
				fit_mask_kn=in_grid_kn_band,
			)
			print0(f"  Scissor fit: {fit.summary()}")
			extrap_rel_ry = E_dft_rel_ry + fit.predict(
				E_dft_rel_ry * RYD_TO_EV, occ_mask_kn) / RYD_TO_EV
			E_sc_rel_ry = np.where(in_grid_kn_band, E_sc_rel_ry, extrap_rel_ry)
		else:
			E_sc_rel_ry = np.where(in_grid_kn_band, E_sc_rel_ry, E_dft_rel_ry)
		E_sc_rel_ev = E_sc_rel_ry * RYD_TO_EV
		E_sc_ev = E_sc_rel_ev + (efermi_ry * RYD_TO_EV)

		# QSGW Σ_xc^QSGW: sharded ω-tensor + replicated E_sc → replicated Σ_xc.
		# Build kernel takes ω-grid and evaluation energies in **eV**; we
		# convert at the seam (kernel internals convert; result is Ry).
		sig_x_rep = jax.device_put(jnp.asarray(sig_x),
			NamedSharding(mesh_xy, P(None, None, None)))
		sigma_xc_qsgw_kij_ry, qsgw_diag = build_qsgw_sigma_xc(
			ppm_outputs.sigma_c_omega, sig_x_rep,
			omega_grid_ry * RYD_TO_EV, E_sc_rel_ev, mesh_xy,
		)
		print0(f"  QSGW: {int(qsgw_diag['n_clipped'])} clipped "
			f"({100*qsgw_diag['frac_clipped']:.1f}%)")
		sigma_total = sigma_xc_qsgw_kij_ry + sig_h
	else:
		# Static modes (X_ONLY, COHSEX) and dynamic-streamed fallback.
		sigma_total = sig_sx + sig_coh + sig_h

	# ---- BGW-style degenerate-set averaging at the H-build seam ----
	# Mirrors Sigma/shiftenergy.f90; replaces the previous per-component
	# averaging at the writer.  Applied to:
	#   - sigma_total's diagonal           → consistent E_qp from eigh
	#   - sig_sx, sig_coh, sig_h, sig_x    → consistent sigma_diag.dat
	#   - sigma_c_at_dft_ev (1-D)          → consistent eqp.dat ``sigC``
	# Off-diagonals are preserved.  The averaging is cheap (a host loop
	# over k of contiguous degeneracy groups) so the redundancy across
	# components is not a perf concern.
	if not config.no_degen_averaging:
		from .degen_average import (
			apply_to_matrix_diagonals,
			average_within_degenerate_sets,
		)
		_enk_sigma_ry, _ = get_enk_bandrange(
			wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
			nspinor=meta.nspinor)
		_e_kn_ry = np.asarray(_enk_sigma_ry, dtype=np.float64)
		_tol = float(config.degen_avg_tol_ry)
		_dav_rep = NamedSharding(mesh_xy, P(None, None, None))
		def _dav(M):
			return jax.device_put(jnp.asarray(apply_to_matrix_diagonals(
				np.asarray(M), _e_kn_ry, _tol)), _dav_rep)
		sigma_total = _dav(sigma_total)
		sig_sx, sig_coh, sig_h, sig_x = _dav(sig_sx), _dav(sig_coh), _dav(sig_h), _dav(sig_x)
		if sigma_c_at_dft_ev is not None:
			sigma_c_at_dft_ev = average_within_degenerate_sets(
				np.asarray(sigma_c_at_dft_ev, dtype=np.complex128),
				_e_kn_ry, _tol)

	# ---- Single H-build + diagonalization on replicated arrays ----
	H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
	E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)

	# ---- DFT energies for the writer ----
	enk_dft, _ = get_enk_bandrange(wfn, sym,
		band_slices.sigma_range, band_slices.sigma_range, nspinor=meta.nspinor)

	# PPM mode: feed the writer the on-shell diag(Σ_c(E_DFT)) (Ry) so the
	# eqp0.dat "sigC" column reports dynamic correlation directly comparable
	# to BGW's (SX-X)+CH at Eo=E_DFT.  Off-diagonals stay zero — the full
	# Σ_c(ω, k, i, j) tensor is in sigma_mnk.h5 for callers that need them.
	# Σ_c diagonal on the ω-grid: feed the eqp1.dat writer's central-diff
	# Z-factor.  Pulled from the on-device sharded tensor when available.
	if ppm_outputs is not None and ppm_outputs.sigma_c_omega is not None:
		sigma_c_omega_diag_ev = np.asarray(extract_sigma_diag_replicated(
			ppm_outputs.sigma_c_omega, mesh_xy)) * RYD_TO_EV
		omega_rel_ev = np.asarray(ppm_options.omega_grid_ev)
	else:
		sigma_c_omega_diag_ev = None
		omega_rel_ev = None

	sigma_c_diag_at_dft_ry = (
		sigma_c_at_dft_ev / RYD_TO_EV
		if (mode.is_dynamic and sigma_c_at_dft_ev is not None)
		else None
	)

	# ---- Optional Σ-decomposition debug table (rank-0, all in eV) ----
	# Single seam at the H-build output: dumps the diagonal pieces that
	# feed E_QP so a downstream investigator can verify
	# E_QP ≟ kin_ion + V_H + Σ_xc(E_DFT).  Head corrections are exposed
	# as their own columns where applicable: PPM mode adds
	# ``sig_c_head(Edft)``; static modes add ``x_head``, ``sex_head``,
	# ``coh_head`` from the BGW-style q→0 head terms.
	if meta.rank == 0 and config.debug.sigma_freq_debug_output:
		from file_io import write_sigma_freq_debug_table
		_e_dft_ev_full = np.asarray(enk_dft, dtype=np.float64) * RYD_TO_EV
		_kin_diag_ev = np.real(
			np.diagonal(np.asarray(kin_ion), axis1=1, axis2=2)) * RYD_TO_EV
		_v_h_diag_ev = np.real(
			np.diagonal(np.asarray(sig_h), axis1=1, axis2=2)) * RYD_TO_EV
		_sig_x_diag_ev = np.real(
			np.diagonal(np.asarray(sig_x), axis1=1, axis2=2)) * RYD_TO_EV
		_e_qp_ev = np.asarray(E_full, dtype=np.float64) * RYD_TO_EV
		_nk, _nb = _e_dft_ev_full.shape
		# Static-COHSEX q→0 head: band-diagonal ``(nb,)`` shifts applied
		# in-place to Σ_x / Σ_SX / Σ_COH inside ``cohsex_sigma``.  The
		# bare-X piece (``sigma_x_diag``) is added in PPM mode too (since
		# Σ_x is static there as well), so ``x_head`` is emitted whenever
		# the head was computed.  ``sex_head`` / ``coh_head`` are
		# screened-channel pieces that only apply when ``do_screened``.
		def _broadcast_head_diag_to_kij(diag_n_ry: np.ndarray) -> np.ndarray:
			return np.broadcast_to(
				np.real(np.asarray(diag_n_ry)) * RYD_TO_EV, (_nk, _nb)
			).astype(np.float64)

		_cols = [
			("E_dft", _e_dft_ev_full),
			("Edft-Ef", _e_dft_ev_full - float(efermi_dft_ev or 0.0)),
			("kin_ion", _kin_diag_ev),
			("V_H", _v_h_diag_ev),
			("x_bare", _sig_x_diag_ev),
		]
		if static_head_terms is not None:
			_cols.append((
				"x_head",
				_broadcast_head_diag_to_kij(static_head_terms.sigma_x_diag),
			))
		if mode.is_dynamic and sigma_c_at_dft_ev is not None:
			_cols.append(("sig_c(Edft)", sigma_c_at_dft_ev))
			# PPM analytic head interpolated at the same E_DFT − E_F used
			# for ``sig_c(Edft)`` (same ω-grid, same linear-interp recipe
			# → cancellation analyses work column-by-column).
			head_w_kn_ry = (
				ppm_outputs.head_sigma_diag_w_kn_ry
				if ppm_outputs is not None else None)
			if head_w_kn_ry is not None:
				_omega_ry = np.asarray(ppm_options.omega_grid_ry, np.float64)
				_eval_ry = (np.asarray(omega_dft_rel_ev, np.float64)
				            / RYD_TO_EV)
				_n_omega = _omega_ry.size
				_e_clamped = np.clip(_eval_ry, _omega_ry[0], _omega_ry[-1])
				_idx_hi = np.clip(np.searchsorted(
					_omega_ry, _e_clamped, side="left"), 1, _n_omega - 1)
				_idx_lo = _idx_hi - 1
				_w_hi = ((_e_clamped - _omega_ry[_idx_lo])
				         / np.where(_omega_ry[_idx_hi] > _omega_ry[_idx_lo],
				                    _omega_ry[_idx_hi] - _omega_ry[_idx_lo],
				                    1.0))
				_w_lo = 1.0 - _w_hi
				_k_idx = np.arange(_nk)[:, None]
				_n_idx = np.arange(_nb)[None, :]
				_lo = head_w_kn_ry[_idx_lo, _k_idx, _n_idx]
				_hi = head_w_kn_ry[_idx_hi, _k_idx, _n_idx]
				_cols.append((
					"sig_c_head(Edft)",
					(_w_lo * _lo + _w_hi * _hi) * RYD_TO_EV,
				))
		else:
			_cols.append(
				("sex_0", np.real(np.diagonal(
					np.asarray(sig_sx), axis1=1, axis2=2)) * RYD_TO_EV))
			_cols.append(
				("coh_0", np.real(np.diagonal(
					np.asarray(sig_coh), axis1=1, axis2=2)) * RYD_TO_EV))
			if static_head_terms is not None and config.do_screened:
				_cols.append((
					"sex_head",
					_broadcast_head_diag_to_kij(static_head_terms.sigma_sx_diag),
				))
				_cols.append((
					"coh_head",
					_broadcast_head_diag_to_kij(static_head_terms.sigma_coh_diag),
				))
		_cols.append(("E_qp", _e_qp_ev))
		write_sigma_freq_debug_table(
			config.debug.sigma_freq_debug_file, _cols)
		print0(f"  Sigma freq debug: {config.debug.sigma_freq_debug_file}")

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
		# Degen averaging was applied once at the H-build seam upstream;
		# the writer just serializes the already-averaged Σ components.
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
		)
	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

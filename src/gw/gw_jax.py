from runtime import set_default_env
set_default_env()  # BEFORE `import jax` — JAX reads env at import time.

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
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
from common.wfn_transforms import get_enk_bandrange
from .gw_config import ComputeMode, LorraxConfig, QPSolver
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
	flatten_V_qmunu,
)
from .screening import compute_screening, screening_requests_for
from .sigma_dispatch import compute_sigma_xc
from .qsgw_utils import (
	extract_sigma_diag_replicated,
	solve_qp,
)
from .head_correction import (
	HeadResolver,
	compute_static_head_terms_from_sample,
	format_head_sample_diagnostics,
	format_static_head_diagnostics,
)
from .wavefunction_bundle import BandSlices
from common import Meta, RYD_TO_EV
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
	# Resolve + validate the QP-energy axis up front so inconsistent
	# (qp_solver × compute_mode × accumulation) combinations fail before
	# any heavy compute (see ``LorraxConfig.qp_solver``).
	qp_solver = config.qp_solver

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

	from .gw_output import (
		GWResults, persist_w0_and_head, print_banner,
		print_system_summary, write_freq_debug, write_qp_wfn_oneshot,
		write_results,
	)
	print_banner(
		backend=current_backend, n_devices=n_devices,
		grid_x=grid_x, grid_y=grid_y, n_procs=n_procs,
		device_kind=device_names, print_fn=print0,
	)

	wfn = WFNReader(config.paths.wfn_file, mesh=mesh_xy)
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

	# DFT eigenvalues on the Σ band window (Ry) — one fetch, reused by the
	# Σ_X diagnostic, the SC initial state, degeneracy averaging, and the
	# results writer.
	enk_dft, _ = get_enk_bandrange(
		wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
		nspinor=meta.nspinor)

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
	# Bispinor: σ^B reads V^{i,j} tiles from v_q_bispinor.h5 and
	# samples ψ at the transverse-centroid Wfns bundle (None when
	# bispinor=False or centroids_file_current is unset).
	wfns_transverse = getattr(isdf, 'wf_bundle_transverse', None)
	bispinor_v_q_path = (
		os.path.join(tmp_dir, 'v_q_bispinor.h5')
		if wfns_transverse is not None else None
	)

	# --- Screening: χ₀ → W = (1 − Vχ)⁻¹ V at every ω the Σ scheme needs ---
	# ``compute_mode`` is the single axis describing the self-energy ansatz
	# (X_ONLY / COHSEX / GN_PPM / HL_PPM); ``qp_solver`` (how QP energies
	# are extracted from Σ) is orthogonal.  X_ONLY needs no screening.
	# V_qmunu is already flat-q ``(nq, μ, μ)``; ``flatten_V_qmunu`` is a
	# back-compat no-op for restart paths that may feed the legacy layout.
	V_q = flatten_V_qmunu(V_qmunu)
	mode = config.compute_mode
	do_screened = mode is not ComputeMode.X_ONLY
	quad, e_ref = None, None
	if do_screened:
		# The minimax τ-axis, solved on G's actual spectral range — shared
		# by every χ₀ build this run (static + probe W here, SC re-solves).
		quad, e_ref = build_static_quadrature(
			wfns, config.minimax_config, print_fn=print0)
	# SC solves its own W's inside the iteration map; the static W is
	# still solved once here to seed the W0 restart flush.
	requests = screening_requests_for(mode, config)
	if qp_solver is QPSolver.SELF_CONSISTENT:
		requests = [r for r in requests if r.role == "static"]
	W_by_role = compute_screening(
		wfns, V_q, requests, quad=quad, e_ref=e_ref,
		sym=sym, centroid_indices=centroid_indices,
		config=config, meta=meta, mesh_xy=mesh_xy, print_fn=print0)

	# Persist W0_qmunu + q=0 head scalars to the ISDF restart file for
	# downstream consumers (BSE, future Σ-builders); no-op unless screened
	# and the restart file exists.
	persist_w0_and_head(
		W_by_role.get("static", V_q),
		tensors_filename=tensors_filename, head_resolver=head_resolver,
		config=config, meta=meta, mesh_xy=mesh_xy, print_fn=print0)

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
			head_resolver, meta, do_screened, print0)

	# ---- Σ_xc + V_H: ONE dispatch for every mode ----
	# The same ``compute_sigma_xc`` call the SC iteration map makes each
	# step — static COHSEX kernels for X_ONLY/COHSEX, the PPM pipeline
	# (fit → 4-branch τ-integration → analytic q→0 head → at-DFT interp)
	# for the dynamic modes, with the QSGW-symmetrised Σ_xc evaluated at
	# E_DFT (textbook G0W0; ``solve_qp`` re-evaluates for fixed_point).
	# SC-iteration-1 ≡ this call, pinned by test_sc_oneshot_equivalence.
	# SC runs skip it — the iteration map would re-do this work on iter 1.
	#
	# History note (kept here because it explains a specific decision and
	# is not yet captured anywhere else): the analytic q→0 head injected
	# at the end of ``compute_ppm_sigma_pipeline`` was re-added in
	# 2026-04-25 after being removed in 1542342 (Apr-10).  Magnitude is
	# ±W^c(0)/(2·V_cell·N_k) on-shell — ~1.24 eV/band on Si 4×4×4 60b.
	# See reports/mos2_kgrid_gnppm_head_convergence_2026-4-10/.
	import gc; gc.collect()
	sigma_result = None
	if qp_solver is not QPSolver.SELF_CONSISTENT:
		with timing.section("gw_jax.sigma"):
			sigma_result = compute_sigma_xc(
				mode,
				wfns=wfns, V_q=V_q, W_by_role=W_by_role,
				e_qp_ev=np.asarray(enk_dft, dtype=np.float64) * RYD_TO_EV,
				static_head_terms=static_head_terms,
				head_resolver=head_resolver,
				quad=quad, e_ref=e_ref,
				config=config, meta=meta, mesh_xy=mesh_xy,
				sym=sym, wfn=wfn, band_slices=band_slices,
				input_dir=input_dir,
				wfns_transverse=wfns_transverse,
				bispinor_v_q_path=bispinor_v_q_path,
				print_fn=print0,
			)

		# Print bare Σ_X diagonal for ISDF quality assessment.  Apply
		# BGW-style degenerate-set averaging (mirrors Sigma/shiftenergy.f90)
		# unless disabled — without it, the QE basis-dependent splitting
		# within degenerate manifolds shows up as a few-meV spread across
		# symmetry-equivalent bands.
		from .degen_average import average_within_degenerate_sets
		sig_x_diag = np.real(np.diagonal(
			np.asarray(sigma_result.sigma_x_kij_ry),
			axis1=1, axis2=2)) * RYD_TO_EV
		if not config.no_degen_averaging:
			sig_x_diag = average_within_degenerate_sets(
				sig_x_diag,
				energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
				tol_ry=float(config.degen_avg_tol_ry),
			)
		print0(f"  Bare Σ_X diagonal (eV), k=0: "
		       + "  ".join(f"{sig_x_diag[0, i]:.4f}" for i in range(min(8, sig_x_diag.shape[1]))))

	# Post-Σ seam: extract every downstream-consumed field into a bare
	# local, so the writer / freq_debug / QP solve all read uniform names
	# regardless of whether the data came from the one-shot SigmaResult
	# above or (below) from a converged SC SigmaResult.  PPM-only fields
	# are None in static modes.
	if sigma_result is not None:
		sig_h   = sigma_result.v_h_kij_ry
		sig_x   = sigma_result.sigma_x_kij_ry
		sig_sx  = (sigma_result.sigma_sx_kij_ry
		           if sigma_result.sigma_sx_kij_ry is not None
		           else jnp.zeros_like(sig_x))
		sig_coh = (sigma_result.sigma_coh_kij_ry
		           if sigma_result.sigma_coh_kij_ry is not None
		           else jnp.zeros_like(sig_x))
		sigma_omega_h5_path = sigma_result.sigma_omega_h5_path
		sigma_c_at_dft_ev   = sigma_result.sigma_c_at_dft_diag_ev
		omega_dft_rel_ev    = sigma_result.omega_dft_rel_ev
		efermi_dft_ev       = sigma_result.efermi_dft_ev
		sigma_c_omega       = sigma_result.sigma_c_omega_kij_ry
		head_sigma_diag_w_kn_ry = sigma_result.head_sigma_diag_w_kn_ry
		omega_grid_ev = (
			np.asarray(sigma_result.omega_grid_ev, dtype=np.float64)
			if sigma_result.omega_grid_ev is not None else None)
		omega_grid_ry = (
			np.asarray(sigma_result.omega_grid_ry, dtype=np.float64)
			if sigma_result.omega_grid_ry is not None else None)
		# Σ_xc(E_DFT) diagonal (eV) — drives eqp_g0w0.dat (PPM one-shot
		# only).  Same spelling as the PPM pipeline's step 4.
		sigma_xc_at_dft_ev = (
			np.diagonal(np.asarray(sig_x), axis1=1, axis2=2) * RYD_TO_EV
			+ sigma_c_at_dft_ev
			if sigma_c_at_dft_ev is not None else None)

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
	sc_rms_history: list[float] = []
	if qp_solver is QPSolver.SELF_CONSISTENT:
		# SC-GW iteration map — mode-agnostic.  Each step rotates ψ via
		# U_qp from eigh(H_qp_dft), then recomputes χ₀ → W → Σ_xc via
		# the mode-orthogonal compute_sigma_xc dispatch (X_ONLY / COHSEX
		# / GN_PPM / HL_PPM all use the same map).  The carry is just
		# H_qp_dft on the active subspace; convergence is judged on RMS
		# ΔE between consecutive eigvalsh.
		from .sc_iteration import (
			SCInputs, dump_qp_wfn_artifacts, dump_sigma_omega_h5_final,
			make_initial_state_from_dft, run_self_consistency)
		from .band_partition import BandPartition
		from .scissor import classify_bands_in_grid

		_e_dft_active_kn_ry = jnp.asarray(
			np.asarray(enk_dft, dtype=np.float64))
		_nb_active = _e_dft_active_kn_ry.shape[1]
		_val_mask_active = jnp.broadcast_to(
			jnp.arange(_nb_active) < int(meta.nelec),
			_e_dft_active_kn_ry.shape)

		# In-range mask: bands whose E_DFT lies inside [σ_ω_min, σ_ω_max]
		# at *every* k.  Bands outside the ω-grid get the per-iteration
		# scissor (otherwise their Σ_c is clamped at the grid edge → the
		# QSGW H-build feeds garbage diagonals that explode the iteration).
		_efermi_ev = float(wfn.efermi) * RYD_TO_EV
		_omega_min_ev = float(config.ppm.omega_min_ev) + _efermi_ev
		_omega_max_ev = float(config.ppm.omega_max_ev) + _efermi_ev
		_e_dft_ev = np.asarray(enk_dft, dtype=np.float64) * RYD_TO_EV
		_band_in_grid, _ = classify_bands_in_grid(
			_e_dft_ev, _omega_min_ev, _omega_max_ev)
		_in_range = jnp.asarray(_band_in_grid, dtype=bool)
		# Default protected = in-range: these bands carry full off-diag Σ.
		# Out-of-range bands take the scissor, no off-diag mixing.
		_protected = _in_range
		print0(
			f"  SC partition: protected/in-range = {int(_band_in_grid.sum())}"
			f"/{int(_band_in_grid.size)} bands"
		)
		_partition = BandPartition(
			protected_mask=_protected, in_range_mask=_in_range)
		_partition.warn_if_protected_outside_grid(print_fn=print0)

		_sc_inputs = SCInputs(
			wfns_dft=wfns, V_q=V_q, kin_ion_dft=kin_ion,
			quad=quad, e_ref=e_ref,
			static_head_terms=static_head_terms,
			head_resolver=head_resolver,
			config=config, meta=meta, mesh_xy=mesh_xy,
			sym=sym, wfn=wfn, centroid_indices=centroid_indices,
			band_slices=band_slices, input_dir=input_dir,
			partition=_partition,
			e_dft_active_kn_ry=_e_dft_active_kn_ry,
			valence_mask_active_kn=_val_mask_active,
			print_fn=print0,
		)
		_state_init = make_initial_state_from_dft(_sc_inputs)
		# Loop knobs from ``config.sc`` (the LORRAX_SC_* env vars are
		# deprecated overrides, applied at config construction).
		_sc = config.sc
		print0(f"  SC: mode={mode.value}, max_iter={_sc.max_iter}, "
		       f"tol={_sc.tol_ev:.1e} eV, accel={_sc.accelerator}"
		       + (f", depth={_sc.history_depth}" if _sc.accelerator == "rcrop"
		          else f", α={_sc.mixing:.2f}"))
		_state_final, sc_rms_history = run_self_consistency(
			_state_init, _sc_inputs,
			max_iter=_sc.max_iter, tol_ev=_sc.tol_ev,
			accelerator=_sc.accelerator,
			history_depth=_sc.history_depth,
			mixing=_sc.mixing,
		)
		_sigma_result = _state_final.last_sigma_result
		print0(
			f"  SC done: {len(sc_rms_history)} iterations"
			+ (f", final RMS ΔE = {sc_rms_history[-1]:.4e} eV"
				if sc_rms_history else " (one-shot)"))

		# Post-SC dumps: WFN_qp.h5 (drop-in BSE / restart input),
		# qp_wfn_rotations.h5 ((U, E_qp) companion), and the converged
		# sigma_mnk.h5 (intermediate iterations skipped the H5 write,
		# so this is the single end-of-run write).  WFN_qp.h5 uses
		# ``final_qp_eigenstates(state_final.H_qp_dft)`` which is the
		# converged DFT-basis H — so its eigenvalues + U are the *true*
		# QP eigenstates of the SC fixed point.  The basis-mixed
		# rebuild + eigh at the post-Σ seam below (line ~660) gives a
		# DIFFERENT (incorrect) U / E for SC mode because
		# ``_sigma_result.sigma_xc_kij_ry`` lives in QP basis but
		# ``kin_ion`` is DFT basis — fixed below by overriding
		# ``sigma_total`` with ``state_final.H_qp_dft - kin_ion``.
		if config.debug.write_wfn_h5:
			dump_qp_wfn_artifacts(
				_state_final, n_occ=int(meta.nelec), mesh_xy=mesh_xy,
				wfn=wfn, band_slices=band_slices, kgrid=meta.kgrid,
				output_dir=input_dir, print_fn=print0,
			)
		sigma_omega_h5_path = dump_sigma_omega_h5_final(
			_state_final, config=config, meta=meta, mesh_xy=mesh_xy,
			input_dir=input_dir, print_fn=print0,
		)

		# Overwrite the post-PPM-seam bare locals from the converged
		# SigmaResult.  Same names and shapes as the one-shot path, so
		# the downstream writer / freq_debug code is identical for SC
		# and one-shot.  PPM-only fields stay None for static SC modes.
		#
		# CRITICAL: ``_sigma_result.{v_h_kij_ry, sigma_x_kij_ry,
		# sigma_xc_kij_ry}`` live in the **QP basis** (compute_sigma_xc
		# was called with the rotated wfn bundle).  Downstream code
		# (post-Σ H build + eigh, writer, freq_debug) is written for
		# DFT-basis matrices — kin_ion is DFT basis throughout — so we
		# must rotate every QP-basis SigmaResult field back to DFT.
		# The U of record is ``state_final.last_sigma_basis_U`` — the
		# unitary that DEFINED the basis the last compute_sigma_xc ran
		# in.  NOT the converged U from ``final_qp_eigenstates``: that
		# is one iteration ahead and agrees only at the fixed point
		# (using it mis-rotated Σ_x/V_H by tens of eV at max_iter=1 —
		# caught by test_sc_oneshot_equivalence).
		from .sc_iteration import _rotate_to_dft_basis
		_U_jax = jnp.asarray(_state_final.last_sigma_basis_U)
		sig_h = _rotate_to_dft_basis(_sigma_result.v_h_kij_ry, _U_jax)
		sig_x = _rotate_to_dft_basis(_sigma_result.sigma_x_kij_ry, _U_jax)
		_sigma_xc_dft = _rotate_to_dft_basis(
			_sigma_result.sigma_xc_kij_ry, _U_jax)
		sigma_total = _sigma_xc_dft + sig_h
		sig_sx = (_rotate_to_dft_basis(_sigma_result.sigma_sx_kij_ry, _U_jax)
		          if _sigma_result.sigma_sx_kij_ry is not None
		          else jnp.zeros_like(sig_x))
		sig_coh = (_rotate_to_dft_basis(_sigma_result.sigma_coh_kij_ry, _U_jax)
		           if _sigma_result.sigma_coh_kij_ry is not None
		           else jnp.zeros_like(sig_x))
		sigma_c_at_dft_ev = _sigma_result.sigma_c_at_dft_diag_ev
		omega_dft_rel_ev = _sigma_result.omega_dft_rel_ev
		efermi_dft_ev = float(wfn.efermi) * RYD_TO_EV
		sigma_c_omega = _sigma_result.sigma_c_omega_kij_ry
		head_sigma_diag_w_kn_ry = _sigma_result.head_sigma_diag_w_kn_ry
		omega_grid_ev = (
			np.asarray(_sigma_result.omega_grid_ev, dtype=np.float64)
			if _sigma_result.omega_grid_ev is not None else None)
		omega_grid_ry = (
			np.asarray(_sigma_result.omega_grid_ry, dtype=np.float64)
			if _sigma_result.omega_grid_ry is not None else None)
		# (sigma_omega_h5_path was set above by dump_sigma_omega_h5_final.)
		if sigma_c_at_dft_ev is not None:
			sigma_xc_at_dft_ev = sigma_c_at_dft_ev + np.real(np.diagonal(
				np.asarray(sig_x), axis1=1, axis2=2)) * RYD_TO_EV
		else:
			sigma_xc_at_dft_ev = None
	else:
		# ---- update_H[Σ; qp_solver] — one-shot (non-SC) paths ----
		# ``one_shot_dft``: Σ_xc was already QSGW-built at E_DFT inside
		# ``compute_sigma_xc`` (pass-through; also covers the static modes
		# and the streamed-Σ_c stand-in).  ``fixed_point``: diagonal
		# on-shell solve + scissor + QSGW rebuild at the solved energies.
		# eqp0.dat/eqp1.dat are at-DFT in every case (written downstream
		# from ``sigma_c_at_dft_ev`` / the ω-grid diag, not from here).
		sigma_total = solve_qp(
			qp_solver, sigma_result, kin_ion,
			config=config, meta=meta, mesh_xy=mesh_xy, print_fn=print0)

	# ---- BGW-style degenerate-set averaging at the H-build seam ----
	# (mirrors Sigma/shiftenergy.f90; see ``degen_average``).
	if not config.no_degen_averaging:
		from .degen_average import average_sigma_components
		(sigma_total, sig_sx, sig_coh, sig_h, sig_x,
		 sigma_c_at_dft_ev) = average_sigma_components(
			sigma_total, sig_sx, sig_coh, sig_h, sig_x, sigma_c_at_dft_ev,
			energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
			tol_ry=float(config.degen_avg_tol_ry),
			mesh_xy=mesh_xy)

	# ---- Single H-build + diagonalization on replicated arrays ----
	H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
	E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)

	# ---- One-shot WFN_qp.h5 dump (drop-in BSE / restart input).  SC
	# already wrote its own WFN_qp.h5 above via dump_qp_wfn_artifacts
	# (using state_final.H_qp_dft) — same physics, slightly different
	# numerics from the post-Σ-seam eigh path.  Skip the second write
	# in SC to avoid clobbering.
	if config.debug.write_wfn_h5 and qp_solver is not QPSolver.SELF_CONSISTENT:
		write_qp_wfn_oneshot(
			U_full, E_full, wfn=wfn, band_slices=band_slices,
			input_dir=input_dir, print_fn=print0)

	# PPM mode: feed the writer the on-shell diag(Σ_c(E_DFT)) (Ry) so the
	# eqp0.dat "sigC" column reports dynamic correlation directly comparable
	# to BGW's (SX-X)+CH at Eo=E_DFT.  Off-diagonals stay zero — the full
	# Σ_c(ω, k, i, j) tensor is in sigma_mnk.h5 for callers that need them.
	# Σ_c diagonal on the ω-grid: feed the eqp1.dat writer's central-diff
	# Z-factor.  Pulled from the on-device sharded tensor when available.
	if sigma_c_omega is not None:
		sigma_c_omega_diag_ev = np.asarray(extract_sigma_diag_replicated(
			sigma_c_omega, mesh_xy)) * RYD_TO_EV
		omega_rel_ev = omega_grid_ev
	else:
		sigma_c_omega_diag_ev = None
		omega_rel_ev = None

	sigma_c_diag_at_dft_ry = (
		sigma_c_at_dft_ev / RYD_TO_EV
		if (mode.is_dynamic and sigma_c_at_dft_ev is not None)
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
		self_consistent=qp_solver is QPSolver.SELF_CONSISTENT,
		sigma_c_diag_at_dft_ry=sigma_c_diag_at_dft_ry,
		sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
		sigma_c_omega_diag_ev=sigma_c_omega_diag_ev,
		omega_rel_ev=omega_rel_ev,
		efermi_ev=efermi_dft_ev,
		sigma_omega_h5_path=sigma_omega_h5_path,
		tensors_filename=tensors_filename,
	)
	if meta.rank == 0:
		# Optional Σ-decomposition debug table (no-op unless
		# ``debug.sigma_freq_debug_output``; see ``gw_output.write_freq_debug``).
		write_freq_debug(
			results, config=config,
			static_head_terms=static_head_terms,
			omega_dft_rel_ev=omega_dft_rel_ev,
			head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
			omega_grid_ry=omega_grid_ry,
			print_fn=print0,
		)
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

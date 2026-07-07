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
from common.wfn_transforms import get_enk_bandrange
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
)
from .qsgw_utils import (
	build_qsgw_sigma_xc,
	extract_sigma_diag_replicated,
	solve_diagonal_sigma_fixed_point,
)
from .head_correction import (
	HeadResolver,
	compute_static_head_terms_from_sample,
	format_head_sample_diagnostics,
	format_static_head_diagnostics,
)
from .wavefunction_bundle import BandSlices
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

	# --- Screening: χ₀ → W = (1 − Vχ)⁻¹ V ---
	# V_qmunu is already flat-q ``(nq, μ, μ)``; downstream consumers
	# bind ``V_q`` to that array directly.  ``flatten_V_qmunu`` is kept
	# as a back-compat no-op for restart paths that may still feed in
	# the legacy 8-D layout.
	V_q = flatten_V_qmunu(V_qmunu)
	if config.do_screened:
		# IBZ cascade for W = (1 − v χ₀)⁻¹ v: slice V_q, χ₀_q to IBZ
		# rows before ``solve_w`` so the per-q Cholesky/LU factor runs
		# only on ``n_q_ibz`` blocks; W_q comes out at IBZ shape and we
		# unfold back to full BZ via the SAME helper V_q uses (it's the
		# same physics — W is bilinear in centroids and rotates by
		# centroid double-permute + L-phase + TRS conj under sym).
		# Explicit-full-BZ debug bypass matches the V_q gate; IBZ
		# activation otherwise depends only on orbit-closure of the
		# centroid set (checked downstream in ``_resolve_ibz_q_list``).
		_use_ibz_w_requested = not bool(
			int(os.environ.get('LORRAX_FORCE_FULL_BZ', '0')))
		_ibz_tables = None
		if _use_ibz_w_requested and getattr(sym, 'q_irr_full_idx', None) is not None:
			from .v_q_g_flat import _resolve_ibz_q_list
			_ibz_tables = _resolve_ibz_q_list(
				sym=sym, centroid_indices=centroid_indices,
				kgrid=tuple(meta.kgrid),
				fft_grid=tuple(meta.fft_grid),
				verbose=False)
			(_, _q_irr_frac, _full_to_irr_idx, _full_to_irr_sym,
			 _sym_perm, _L_table, _use_ibz_w) = _ibz_tables
		else:
			_use_ibz_w = False

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
				# IBZ slice on V_q and χ₀_q.  Both retain the canonical
				# ``P(None, 'x', 'y')`` sharding; the helper locks it in.
				if _use_ibz_w:
					from common.symmetry_maps import slice_q_full_to_ibz
					_nat = NamedSharding(mesh_xy, P(None, 'x', 'y'))
					with timing.section("W.slice_to_ibz"):
						V_q_solve = slice_q_full_to_ibz(
							V_q, sym.q_irr_full_idx, out_sharding=_nat)
						chi0_q_solve = slice_q_full_to_ibz(
							chi0_q, sym.q_irr_full_idx, out_sharding=_nat)
						del chi0_q
						chi0_q_solve.block_until_ready()
				else:
					V_q_solve = V_q
					chi0_q_solve = chi0_q
				with timing.section("W.compile"):
					precompile_solve_w(V_q_solve, chi0_q_solve, meta, mesh_xy,
					                   solver=config.backend.screening_solver)
				with timing.section("W.exec"):
					W_q_solve = solve_w(V_q_solve, chi0_q_solve, meta, mesh_xy,
					              solver=config.backend.screening_solver)
					# χ₀ is donated inside solve_w — the reference is
					# now invalid.  Do NOT touch ``chi0_q_solve`` after this.
					del chi0_q_solve
					W_q_solve.block_until_ready()
				# IBZ → full-BZ unfold (centroid double-permute + L-phase
				# + TRS conj) — same helper V_q uses.  Σ_COH/SX still
				# iterate over the full BZ in the k-q sums.
				if _use_ibz_w:
					from common.symmetry_maps import unfold_v_q
					with timing.section("W.unfold_to_full_bz"):
						_n_sym_spatial = int(
							np.asarray(_sym_perm).shape[0]) // 2
						W_q = unfold_v_q(
							W_q_solve,
							irr_idx=_full_to_irr_idx,
							sym_idx=_full_to_irr_sym,
							sym_perm=_sym_perm, L_table=_L_table,
							q_irr_frac=_q_irr_frac,
							mesh_xy=mesh_xy,
							n_sym_spatial=_n_sym_spatial)
						del W_q_solve
						W_q.block_until_ready()
				else:
					W_q = W_q_solve

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
		# W_q is already flat-q (nq, μ, μ).  The W0_qmunu placeholder
		# created by ``write_restart_state_to_h5(init_W0=True)`` is
		# rank-3 (sized from V_qmunu), so we write W_q at flat-q.
		# Previously this reshaped to legacy 8-D and tripped
		# ``phdf5 async write: dataset rank mismatch ds=/W0_qmunu
		# file_rank=3 write_rank=8``.  Downstream (BSE) consumers
		# of W0_qmunu were already updated to flat-q in commit a052a1c.
		write_w0_qmunu_to_h5(tensors_filename, W_q,
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
			wfns_transverse=wfns_transverse,
			bispinor_v_q_path=bispinor_v_q_path,
			backend=config.backend.slab_io,
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
	# Skip the standalone PPM pipeline when running SC-dynamic — the
	# QSGW iteration map calls compute_sigma_xc internally each step
	# and would re-do this work on iter 1.
	if mode.is_dynamic and not config.self_consistent:
		# Mode-orthogonal screening: ask the scheme which W's it needs
		# (PPM modes need a probe-frequency W in addition to the static
		# one already in W_q), evaluate them in one pass, and hand the
		# {role → W_q} dict to the Σ build.
		from .screening import compute_screening, screening_requests_for
		_requests = [r for r in screening_requests_for(mode, config)
		             if r.role != "static"]   # static W already solved above
		_W_extra = compute_screening(
			wfns, V_q, _requests,
			quad=quad, e_ref=e_ref,
			config=config, meta=meta, mesh_xy=mesh_xy,
			print_fn=print0,
		)
		ppm_outputs = compute_ppm_sigma_pipeline(
			wfns=wfns,
			V_q=V_q,
			W_static_q=W_q,
			W_probe_q=_W_extra["probe"],
			sig_x=sig_x, sig_h=sig_h,
			quad=quad, e_ref=e_ref,
			config=config, meta=meta, mesh_xy=mesh_xy,
			head_resolver=head_resolver,
			band_slices=band_slices, wfn=wfn, sym=sym,
			input_dir=input_dir,
			print_fn=print0,
		)
	# Post-PPM seam: extract every downstream-consumed field into a bare
	# local, so the writer / freq_debug / SC branch all read uniform names
	# regardless of whether the data came from a one-shot ``ppm_outputs``,
	# from a converged SC ``SigmaResult``, or is None (static modes).
	sigma_omega_h5_path = ppm_outputs.sigma_omega_h5_path if ppm_outputs else None
	sigma_c_at_dft_ev   = ppm_outputs.sigma_c_at_dft_ev   if ppm_outputs else None
	sigma_xc_at_dft_ev  = ppm_outputs.sigma_xc_at_dft_ev  if ppm_outputs else None
	omega_dft_rel_ev    = ppm_outputs.omega_dft_rel_ev    if ppm_outputs else None
	efermi_dft_ev       = ppm_outputs.efermi_dft_ev       if ppm_outputs else None
	sigma_c_omega       = ppm_outputs.sigma_c_omega       if ppm_outputs else None
	head_sigma_diag_w_kn_ry = (
		ppm_outputs.head_sigma_diag_w_kn_ry if ppm_outputs else None)
	omega_grid_ev = (
		np.asarray(config.omega_grid_ev, dtype=np.float64)
		if ppm_outputs else None)
	omega_grid_ry = (
		np.asarray(config.omega_grid_ry, dtype=np.float64)
		if ppm_outputs else None)
	del ppm_outputs                              # all fields now in bare locals

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
	if config.self_consistent:
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

		_enk_active, _ = get_enk_bandrange(
			wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
			nspinor=meta.nspinor)
		_e_dft_active_kn_ry = jnp.asarray(
			np.asarray(_enk_active, dtype=np.float64))
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
		_e_dft_ev = np.asarray(_enk_active, dtype=np.float64) * RYD_TO_EV
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
			sym=sym, wfn=wfn,
			band_slices=band_slices, input_dir=input_dir,
			partition=_partition,
			e_dft_active_kn_ry=_e_dft_active_kn_ry,
			valence_mask_active_kn=_val_mask_active,
			print_fn=print0,
		)
		_state_init = make_initial_state_from_dft(_sc_inputs)
		# TODO: plumb max_iter / tol_ev through config; env vars for now
		# so the first end-to-end QSGW test can vary them without rebuild.
		_max_iter = int(os.environ.get("LORRAX_SC_MAX_ITER", "20"))
		_tol_ev = float(os.environ.get("LORRAX_SC_TOL_EV", "1.0e-4"))
		_accel = os.environ.get("LORRAX_SC_ACCEL", "rcrop")
		_history_depth = int(os.environ.get("LORRAX_SC_DEPTH", "5"))
		_mixing = float(os.environ.get("LORRAX_SC_MIXING", "1.0"))
		print0(f"  SC: mode={mode.value}, max_iter={_max_iter}, "
		       f"tol={_tol_ev:.1e} eV, accel={_accel}"
		       + (f", depth={_history_depth}" if _accel == "rcrop"
		          else f", α={_mixing:.2f}"))
		_state_final, sc_rms_history = run_self_consistency(
			_state_init, _sc_inputs,
			max_iter=_max_iter, tol_ev=_tol_ev,
			accelerator=_accel,
			history_depth=_history_depth,
			mixing=_mixing,
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
		from .sc_iteration import final_qp_eigenstates
		_enk_qp_active_ry, _U_kmn, _efermi_final_ry = final_qp_eigenstates(
			_state_final, n_occ=int(meta.nelec), mesh_xy=mesh_xy,
		)
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
		# must rotate every QP-basis SigmaResult field back to DFT via
		# the converged U from ``final_qp_eigenstates`` before plumbing
		# into the bare locals.  Without this, the line-660 eigh sees
		# ``kin_ion`` (DFT) + ``sigma_xc`` (QP) and produces nonsense
		# eigenvalues for eqp0.dat (off by tens of eV per band on MoS2).
		from .sc_iteration import _rotate_to_dft_basis
		_U_jax = jnp.asarray(_U_kmn)
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
	elif mode.is_dynamic and sigma_c_omega is not None:
		# NB (streamed-mode fallthrough): in KIJ_STREAM accumulation
		# ``sigma_c_omega`` is None (Σ_c lives only in the sigma_kij h5, which
		# _inject_analytic_head has already head-corrected in place), so this
		# on-shell QP fixed-point solve is skipped — streamed runs get the
		# at-DFT diagnostic Σ_c only, not on-shell QP energies.  See WS1 /
		# Bug B in reports/sigma_ppm_tighten_2026-07-04.
		# G0W0/QSGW: diagonal-Σ(E) fixed point (with optional scissor) →
		# QSGW Σ_xc^QSGW.  Restart-friendly: this whole block consumes only
		# the on-device ``sigma_c_omega`` plus replicated (sig_x, sig_h),
		# so a future outer QSGW iteration loop can pass refreshed inputs
		# in without touching the disk.
		# All quantities below are in **Rydberg** until the scissor's print
		# summary and final eV outputs.  Σ_c(ω) lives natively in Ry on the
		# Ry ω-grid; mixing that with eV-converted h0/Σ_x is a footgun.

		# Diagonal Σ_c(ω, k, n) and Σ_x(k, n) replicated on host, in Ry.
		sigma_c_diag_w_kn_ry = np.asarray(extract_sigma_diag_replicated(
			sigma_c_omega, mesh_xy))
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
			# Sort-and-pair semantics (per-k argsort on each of E_DFT and
			# E_QP independently) live inside ``fit_scissor`` and are
			# robust to QSGW reorderings; one-shot G0W0 has no
			# reordering and the sort is a no-op.
			fit = fit_scissor(
				E_dft_rel_ry * RYD_TO_EV,
				E_sc_rel_ry * RYD_TO_EV,
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

		# QSGW Σ_xc^QSGW: sharded ω-tensor + replicated E_sc → replicated Σ_xc.
		# Build kernel takes ω-grid and evaluation energies in **eV**; we
		# convert at the seam (kernel internals convert; result is Ry).
		sig_x_rep = jax.device_put(jnp.asarray(sig_x),
			NamedSharding(mesh_xy, P(None, None, None)))
		sigma_xc_qsgw_kij_ry, qsgw_diag = build_qsgw_sigma_xc(
			sigma_c_omega, sig_x_rep,
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
			return jax.device_put(apply_to_matrix_diagonals(
				np.asarray(M), _e_kn_ry, _tol), _dav_rep)
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

	# ---- One-shot WFN_qp.h5 dump (drop-in BSE / restart input).  SC
	# already wrote its own WFN_qp.h5 above via dump_qp_wfn_artifacts
	# (using state_final.H_qp_dft) — same physics, slightly different
	# numerics from the post-Σ-seam eigh path.  Skip the second write
	# in SC to avoid clobbering.
	# The one-shot dump is an IBZ-only writer: it rotates the DFT bands the
	# WFN carries coefficients for, so it needs Σ (hence U_full/E_full) on the
	# same k-set as the WFN.  When Σ is unfolded to the full BZ (line 275) the
	# eigh runs on nk_full > wfn.nkpts and this path cannot build the artifact
	# — skip with a warning rather than crash the whole run (the writer would
	# raise ValueError on the k-count mismatch).
	if config.debug.write_wfn_h5 and not config.self_consistent:
		if int(U_full.shape[0]) != int(wfn.nkpts):
			print0(
				f"  QP WFN (one-shot): skipped — Σ on {int(U_full.shape[0])} "
				f"k-points but WFN carries {int(wfn.nkpts)} (IBZ); the one-shot "
				f"dump only supports IBZ-Σ. Set debug.write_wfn_h5=false to silence.")
		else:
			from file_io.qp_wfn import write_qp_wfn_h5
			_qp_wfn_path = os.path.join(input_dir, "WFN_qp.h5")
			if jax.process_index() == 0:
				write_qp_wfn_h5(
					_qp_wfn_path, wfn=wfn,
					U_kmn=np.asarray(U_full, dtype=np.complex128),
					enk_active_qp_ry=np.asarray(E_full, dtype=np.float64),
					band_start=band_slices.b0, band_stop=band_slices.b3,
				)
			try:
				from jax.experimental import multihost_utils as _mh
				_mh.sync_global_devices("oneshot_qp_wfn_h5_write")
			except Exception:
				pass
			print0(f"  QP WFN (one-shot): {_qp_wfn_path}")

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

	# ---- Optional Σ-decomposition debug table (rank-0, all in eV) ----
	# Single seam at the H-build output: dumps the diagonal pieces that
	# feed the BGW-format ``eqp0`` (Σ_tot at E_DFT) and ``eqp1`` (the same
	# Σ_tot − E_DFT extrapolated by the Z-factor central-difference slope
	# of dRe[Σ_c]/dω at E=E_DFT).  By construction
	# ``eqp0 ≟ kin_ion + V_H + x_bare + sig_c(Edft).Re`` exactly, and
	# ``eqp1 ≟ E_DFT + Z·(eqp0 − E_DFT)`` with Z=1 in static modes
	# (degenerate eqp1==eqp0).  Head corrections are also exposed as
	# their own columns: ``x_head`` (always when the head is computed)
	# and either ``sig_c_head(Edft)`` (PPM) or ``sex_head/coh_head``
	# (static, screened mode).
	if meta.rank == 0 and config.debug.sigma_freq_debug_output:
		from file_io import write_sigma_freq_debug_table
		from .eqp_bgw import (
			compute_eqp_diag, compute_z_factor_from_omega_grid)
		_e_dft_ev_full = np.asarray(enk_dft, dtype=np.float64) * RYD_TO_EV
		_kin_diag_ev = np.real(
			np.diagonal(np.asarray(kin_ion), axis1=1, axis2=2)) * RYD_TO_EV
		_v_h_diag_ev = np.real(
			np.diagonal(np.asarray(sig_h), axis1=1, axis2=2)) * RYD_TO_EV
		_sig_x_diag_ev = np.real(
			np.diagonal(np.asarray(sig_x), axis1=1, axis2=2)) * RYD_TO_EV
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
			# Compute Σ_c(E_DFT) + Z via the SAME recipe the eqp{0,1}.dat
			# writer uses, so the freq_debug sig_c(Edft) column and the
			# eqp0/eqp1 columns are bit-consistent.  PPM pipeline's own
			# interp produces a numerically slightly different value (~10
			# meV) due to a different vectorisation path.
			_e_dft_rel_ev = np.asarray(omega_dft_rel_ev, dtype=np.float64)
			_sigma_c_at_dft_for_eqp, _z_factor = (
				compute_z_factor_from_omega_grid(
					sigma_c_omega_diag_ev=np.asarray(
						sigma_c_omega_diag_ev, dtype=np.complex128),
					omega_rel_ev=np.asarray(omega_rel_ev, dtype=np.float64),
					e_dft_rel_ev=_e_dft_rel_ev,
				))
			_cols.append(("sig_c(Edft)", _sigma_c_at_dft_for_eqp))
			# PPM analytic head interpolated at the same E_DFT − E_F used
			# for ``sig_c(Edft)`` (same ω-grid, same linear-interp recipe
			# → cancellation analyses work column-by-column).
			if head_sigma_diag_w_kn_ry is not None:
				from .qsgw_utils import interp_along_omega
				_eval_ry = (np.asarray(omega_dft_rel_ev, np.float64)
				            / RYD_TO_EV)
				_cols.append((
					"sig_c_head(Edft)",
					interp_along_omega(
						head_sigma_diag_w_kn_ry,
						omega_grid_ry,
						_eval_ry,
					) * RYD_TO_EV,
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
		# eqp0 / eqp1 — same math as the eqp{0,1}.dat writer.  In PPM
		# mode (Σ_c, Z) were already computed above for the
		# ``sig_c(Edft)`` column; in static modes the combined Σ_SX +
		# Σ_COH plays the role of Σ_c(E_DFT) and Z = 1 so eqp1 == eqp0.
		if mode.is_dynamic and sigma_c_at_dft_ev is not None:
			pass  # reuses _sigma_c_at_dft_for_eqp / _z_factor from above
		else:
			_sigma_c_at_dft_for_eqp = np.real(np.diagonal(
				np.asarray(sig_sx + sig_coh), axis1=1, axis2=2)) * RYD_TO_EV
			_z_factor = None
		_eqp0_ev, _eqp1_ev = compute_eqp_diag(
			kin_ion_diag_ev=_kin_diag_ev,
			hartree_diag_ev=_v_h_diag_ev,
			sigma_x_diag_ev=_sig_x_diag_ev,
			sigma_c_at_dft_diag_ev=np.asarray(
				_sigma_c_at_dft_for_eqp, dtype=np.complex128),
			e_dft_ev=_e_dft_ev_full,
			z_factor=_z_factor,
		)
		_cols.append(("eqp0", _eqp0_ev.astype(np.float64)))
		_cols.append(("eqp1", _eqp1_ev.astype(np.float64)))
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
		efermi_ev=efermi_dft_ev,
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

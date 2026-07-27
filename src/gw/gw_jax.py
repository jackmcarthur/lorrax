"""GWJAX — the LORRAX GW driver.

``main()`` is the physics scaffold.  Each line below is one stage call
in this file, in execution order:

    ζ, V, ψ         = prepare_isdf_and_wavefunctions(...)  # ISDF basis + bare Coulomb   (gw_init)
    quad            = build_static_quadrature(E_nk)        # minimax τ-axis, solved on
                                                           #   G's spectral range        (minimax_screening)
    {W(ω)}          = compute_screening(ψ, V, requests)    # χ₀(G(τ)) → per-q Dyson
                                                           #   solves, one W per (ω,
                                                           #   role) the Σ scheme asks   (screening)
    Σ_xc(ω), V_H    = compute_sigma_xc(ψ, V, {W})          # Σ_x ⊕ Σ_c [PPM fit +
                                                           #   4-branch τ-integration]
                                                           #   ⊕ q→0 head channel        (sigma_dispatch)
    Σ_total         = solve_qp(Σ) | run_sc_driver(...)     # update_H per qp_solver      (qsgw_utils, sc_iteration)
    E_qp, U_qp      = eigh(kin_ion + Σ_total)              # + degenerate-set averaging  (degen_average)
    eqp0/eqp1/σ.dat = write_results(...)                   # writers, debug tables       (gw_output)

Two orthogonal config axes pivot the flow: ``compute_mode`` — the
self-energy ansatz (``x_only`` / ``cohsex`` / ``gn_ppm`` / ``hl_ppm``) —
and ``qp_solver`` — how QP energies are extracted from Σ
(``one_shot_dft`` / ``fixed_point`` / ``self_consistent``).  The
self-consistent path iterates the same ``compute_sigma_xc`` dispatch;
iteration 1 reproduces the one-shot result exactly (gated by
``tests/test_invariance_gates.py::test_sc_iteration1_equals_one_shot``).

Deliberate physics absences (do not "fix" these): G(τ) is never
materialized — it exists only as ψψ*-phases inside the χ₀/Σ kernels;
the q→0 Coulomb head is a scalar channel threaded through every stage,
not a stage; W is evaluated at exactly the two frequencies {0, iω_p} a
one-pole model is determined by.  See ``docs/theory/physics.md``.
"""
from runtime import bootstrap
bootstrap()  # env + jax.distributed + CPU fallback — BEFORE `import jax`.

import argparse
import gc
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)

from file_io import (
    WFNReader,
    load_kin_ion_submatrix, load_centroids,
)
from common import Meta, RYD_TO_EV, symmetry_maps
from common.wfn_transforms import get_enk_bandrange
import common.timing as timing
from .gw_config import ComputeMode, LorraxConfig, QPSolver
from .gw_init import prepare_isdf_and_wavefunctions
from .compute_vcoul import build_bgw_v_grid_fn
from .minimax_screening import build_static_quadrature
from .screening import compute_screening, screening_requests_for
from .sigma_dispatch import compute_sigma_xc
from .qsgw_utils import extract_sigma_diag_replicated, solve_qp
from .degen_average import (
	average_sigma_components,
	average_within_degenerate_sets,
)
from .head_correction import (
	HeadResolver,
	compute_static_head_terms_from_sample,
	format_head_sample_diagnostics,
	format_static_head_diagnostics,
)
from .wavefunction_bundle import BandSlices
from .gw_output import (
	GWResults,
	persist_w0_and_head,
	print_banner,
	print_system_summary,
	write_freq_debug,
	write_qp_wfn_oneshot,
	write_results,
)


def _build_mesh():
	"""Construct 2D device mesh with most-square factorization."""
	total = jax.process_count() * jax.local_device_count()
	gx = int(np.sqrt(total))
	while gx > 1 and total % gx != 0:
		gx -= 1
	return Mesh(np.array(jax.devices()).reshape(gx, total // gx), ['x', 'y'])


def _setup_runtime(config, mesh_xy, *, print_fn=print) -> None:
	"""Pre-init NCCL + phdf5 MPI + JAX persistent compile cache.

	All three are best-effort startup optimizations whose absence is not
	fatal:

	- **NCCL warmup**: pre-allocates communicators for full-mesh and
	  per-axis psums so the first real collective (sigma's all-reduce-
	  start) doesn't eat a timed section.  No-op in single-process.
	- **phdf5 ``MPI_Init_thread``**: when the slab-IO backend is the
	  phdf5 FFI, eagerly enter ``MPI_THREAD_MULTIPLE`` so the first
	  collective ``H5Fcreate`` (in ``zeta_fit_chunked``) doesn't pay
	  the ~400 ms MPI_Init cost on the critical path; failures are
	  logged and swallowed.
	- **JAX persistent compile cache**: enable the XDG-style on-disk
	  cache so a warm run skips XLA compilation entirely — safe at
	  every process count since scorecard AH (measured at P=8 on the
	  fixture: 373 compiles/rank cold, 5 warm).  Opt out via
	  ``ISDF_JAX_CACHE_DIR=""``.
	"""
	from runtime import nccl_warmup
	from .gw_config import SlabIOBackend

	with timing.section("nccl_warmup"):
		nccl_warmup(mesh_xy)

	if config.backend.slab_io is SlabIOBackend.PHDF5_FFI:
		try:
			from ffi.common.ffi_loader import phdf5_init_mpi
			phdf5_init_mpi()
		except Exception as exc:
			print_fn(f"  [phdf5 init_mpi] skipped: {exc}")
	elif config.backend.slab_io is SlabIOBackend.PHDF5_HOST:
		# mpi4py initialises MPI on first import; do it here so the
		# ~400 ms MPI_Init cost is amortised before the first SlabIO
		# open (same rationale as the FFI's phdf5_init_mpi).
		try:
			from mpi4py import MPI  # noqa: F401
		except Exception as exc:
			print_fn(f"  [phdf5_host mpi4py init] skipped: {exc}")

	try:
		from common.jax_compile_cache import ensure_jax_compile_cache
		ensure_jax_compile_cache()
	except Exception as exc:
		print_fn(f"  [jax compile cache] skipped: {exc}")


def _compute_static_head(head_resolver, meta, do_screened, print0):
	"""Resolve the q→0 head sample and its exact band-diagonal Σ terms.

	Used by every mode: the bare-X head piece applies to static and
	dynamic Σ alike (the SX/COH pieces additionally apply when screened).
	"""
	head = head_resolver.at(0.0 + 0.0j)
	print0(format_head_sample_diagnostics(head, include_screened=do_screened))
	occ_mask = np.arange(meta.nb_sigma, dtype=np.int32) < meta.nelec
	terms = compute_static_head_terms_from_sample(
		head, occ=occ_mask, cell_volume=meta.cell_volume, nk_tot=meta.nk_tot)
	print0(format_static_head_diagnostics(terms))
	return terms


def main(argv=None):
	argp = argparse.ArgumentParser(allow_abbrev=False,
		description="LORRAX GW driver — COHSEX / GN-PPM / HL-PPM self-energy, "
		            "one-shot or self-consistent (see gw_config.ComputeMode / QPSolver)")
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

	# ---- Configuration ----
	# The two orthogonal physics axes are resolved + validated up front so
	# inconsistent (qp_solver × compute_mode × accumulation) combinations
	# fail before any heavy compute (see ``LorraxConfig.qp_solver``).
	config = LorraxConfig.from_input_file(args.input, print_fn=print0)
	input_dir = config.input_dir
	qp_solver = config.qp_solver     # how QP energies are extracted from Σ
	mode = config.compute_mode       # the self-energy ansatz
	do_screened = mode is not ComputeMode.X_ONLY

	# ---- Runtime initialization: device mesh, NCCL/MPI/compile-cache ----
	mesh_xy = _build_mesh()
	_setup_runtime(config, mesh_xy, print_fn=print0)
	print_banner(
		backend=jax.default_backend(),
		n_devices=len(jax.devices()),
		grid_x=mesh_xy.devices.shape[0], grid_y=mesh_xy.devices.shape[1],
		n_procs=jax.process_count(),
		device_kind=jax.devices()[0].device_kind if jax.devices() else "unknown",
		print_fn=print0,
	)

	# ---- System inputs: WFN, symmetry tables, ISDF centroids ----
	wfn = WFNReader(config.paths.wfn_file, mesh=mesh_xy)
	sym = symmetry_maps.SymMaps(wfn)
	_, centroid_indices, n_rmu = load_centroids(config.paths.centroids_file, wfn.fft_grid)
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{n_rmu}.h5")
	print_system_summary(
		n_rmu=n_rmu, fft_grid=wfn.fft_grid,
		cell_volume=wfn.cell_volume, print_fn=print0,
	)

	meta = Meta.from_system(wfn, sym, config.nval, config.ncond, config.nband, n_rmu, config.bispinor)
	meta.rank = jax.process_index()
	meta.n_proc = jax.process_count()
	meta.sys_dim = config.sys_dim
	meta.bispinor = config.bispinor
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

	# ---- Screening: χ₀ → W = (1 − Vχ)⁻¹ V at every ω the Σ scheme needs ----
	# X_ONLY requests no screening at all.
	V_q = V_qmunu               # flat-q (nq, μ, μ) — compute and restart alike
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
	# TIMED, and it was not.  This call gathers the whole (nq, μ, μ) W0
	# onto one rank on the ``h5py_allgather`` backend and writes it — the
	# stage AF.4c measured at ~1.7 MB/s aggregate and 2 h 55 m of total
	# silence at c2406 — yet it sat between two timed stages with no
	# section of its own, so it appeared in the run's wall clock and in
	# NO row of the stage table.  Naming it is the precondition for
	# anyone attributing that wall time (the write path itself is
	# workstream AE/AF's; this is the instrument, not the fix).
	with timing.section("gw_jax.persist_w0"):
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
	gc.collect()   # drop ISDF-stage temporaries before the Σ build
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

		# ── Σ stage gate ─────────────────────────────────────────────
		# Σ_x[n,n] = −Σ_{m∈occ} ⟨nm|V|mn⟩ is a negative-definite
		# quadratic form in a positive-semidefinite kernel: every
		# diagonal entry is strictly negative in a correct run, whatever
		# the system.  A positive one is a sign / conjugation /
		# band-index slip, not a convergence issue.  The magnitude
		# bracket is deliberately loose (bare exchange runs −40…−5 eV
		# for the production decks) — it exists to catch a units or
		# basis-normalisation slip, not to police physics.
		from common import sanity
		sanity.check_finite("Σ_x", sigma_result.sigma_x_kij_ry, print_fn=print0)
		sanity.check_finite("V_H", sigma_result.v_h_kij_ry, print_fn=print0)
		sanity.check_sign("Σ_x diagonal (eV)", sig_x_diag,
		                  expect="negative", print_fn=print0)
		sanity.check_in_range("Σ_x diagonal (eV)", sig_x_diag,
		                      -200.0, 0.0, unit="eV", print_fn=print0)

	# ---- QP Hamiltonian: H_QP = (H_DFT - V_xc) + V_H + Σ_xc ----
	#
	# Sharding: kin_ion + all four static-COHSEX components (Σ_SX, Σ_COH,
	# V_H, Σ_X) are loaded **fully replicated** on the device mesh.  The
	# only sharded object surviving past this point is the dynamic-Σ_c
	# ω-tensor produced by ``compute_ppm_sigma_pipeline``, which is
	# collapsed into a replicated Σ_xc^QSGW by ``build_qsgw_sigma_xc``
	# below.  This lets the rest of post-processing (eigh, scissor,
	# eqp output) operate on replicated arrays without resharding seams.
	# Provenance gate BEFORE the read: kin_ion.h5 fixes the Coulomb
	# truncation convention and the band window for the whole mean-field
	# side of H₀, and ``has_hartree`` decides whether the ISDF ``sig_h``
	# is added on top.  A silent disagreement here lands as tens of eV in
	# a ~500 eV cancellation, so it is checked loudly, once.
	from file_io import validate_kin_ion_against_run
	kin_ion_attrs = validate_kin_ion_against_run(
		config.paths.kin_ion_file,
		sys_dim=config.sys_dim,
		nk=meta.nk_tot,
		band_stop=band_slices.b3,
		print_fn=print0,
	)
	# Which V_H source this run will use, resolved once and printed.  Only
	# the LEGACY ``folded`` case means "V_H is inside kin_ion's values";
	# ``stored``/``gspace`` supply it as a separate matrix that the Σ seam
	# substitutes for the ISDF quadrature, and ``isdf`` keeps the latter.
	from file_io.kin_ion import resolve_hartree_source
	hartree_source = resolve_hartree_source(
		config.paths.kin_ion_file, config.hartree_source, print_fn=print0)
	print0(f"  hartree_source: requested={config.hartree_source} "
	       f"→ resolved={hartree_source}")
	kin_ion_has_hartree = (hartree_source == "folded")
	kin_ion = load_kin_ion_submatrix(
		config.paths.kin_ion_file, band_slices.b0, band_slices.b3,
		mesh=mesh_xy, backend=config.backend.slab_io,
	)

	# ---- update_H[Σ; qp_solver] — all branches yield ``sigma_total``
	# (Σ_xc + V_H, Ry, DFT basis, replicated) whose eigh gives E_qp/U_qp.
	if qp_solver is QPSolver.SELF_CONSISTENT:
		# SC-QSGW: iterate ψ-rotation → χ₀ → W → Σ_xc (the same
		# compute_sigma_xc dispatch, mode-agnostic) to the fixed point;
		# the returned SigmaResult is already rotated back to the DFT
		# basis and its sigma_omega_h5_path points at the converged
		# single-write sigma_mnk.h5.  See ``sc_iteration.run_sc_driver``.
		from .sc_iteration import run_sc_driver
		sigma_result, sigma_total, _ = run_sc_driver(
			wfns, V_q, kin_ion,
			quad=quad, e_ref=e_ref,
			static_head_terms=static_head_terms,
			head_resolver=head_resolver,
			config=config, meta=meta, mesh_xy=mesh_xy,
			sym=sym, wfn=wfn, centroid_indices=centroid_indices,
			band_slices=band_slices, input_dir=input_dir,
			enk_dft=enk_dft, print_fn=print0)
	else:
		# One-shot: ``one_shot_dft`` = Σ_xc was already QSGW-built at
		# E_DFT inside compute_sigma_xc (pass-through; also covers static
		# modes and the streamed-Σ_c stand-in); ``fixed_point`` = diagonal
		# on-shell solve + scissor + QSGW rebuild at the solved energies.
		# eqp0.dat/eqp1.dat are at-DFT in every case (written downstream
		# from ``sigma_c_at_dft_ev`` / the ω-grid diag, not from here).
		sigma_total = solve_qp(
			qp_solver, sigma_result, kin_ion,
			config=config, meta=meta, mesh_xy=mesh_xy, print_fn=print0)

	# ---- Post-Σ seam: bare locals from the (DFT-basis) SigmaResult ----
	# One extraction for SC and one-shot alike; PPM-only fields are None
	# in static modes.
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

	# ---- BGW-style degenerate-set averaging at the H-build seam ----
	# (mirrors Sigma/shiftenergy.f90; see ``degen_average``).
	if not config.no_degen_averaging:
		(sigma_total, sig_sx, sig_coh, sig_h, sig_x,
		 sigma_c_at_dft_ev) = average_sigma_components(
			sigma_total, sig_sx, sig_coh, sig_h, sig_x, sigma_c_at_dft_ev,
			energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
			tol_ry=float(config.degen_avg_tol_ry),
			mesh_xy=mesh_xy)

	# ---- Single H-build + diagonalization on replicated arrays ----
	# Gate the two inputs to the QP diagonalization *before* eigh: LAPACK
	# on a NaN-bearing matrix returns without complaining, and the garbage
	# then propagates into eqp0/eqp1/WFN_qp.h5 with rc=0.  ``kin_ion`` also
	# comes off disk (kin_ion.h5), so this doubles as the content check on
	# that interface.
	from common import sanity
	sanity.check_finite("kin_ion (from kin_ion.h5)", kin_ion, print_fn=print0)
	sanity.check_finite("Σ_total (Σ_xc + V_H)", sigma_total, print_fn=print0)

	H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
	E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)
	sanity.check_finite("E_qp (eigh of H_QP)", E_full, print_fn=print0)

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
		kin_ion_has_hartree=kin_ion_has_hartree,
		hartree_source=hartree_source,
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
	if meta.rank == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

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
	eqp0/eqp1[/eqp2]/σ.dat = write_results(...)            # writers, debug tables       (gw_output)

Two orthogonal config axes pivot the flow: ``compute_mode`` — the
self-energy ansatz (``x_only`` / ``cohsex`` / ``gn_ppm`` / ``hl_ppm``,
plus ``mpa``, which is declared on the axis and refused at entry until
its Σ stage lands) — and ``qp_solver`` — how QP energies are extracted
from Σ (``one_shot_dft`` / ``fixed_point`` / ``self_consistent``).  The
self-consistent path iterates the same ``compute_sigma_xc`` dispatch;
iteration 1 reproduces the one-shot result exactly (gated by
``tests/test_invariance_gates.py::test_sc_iteration1_equals_one_shot``).

Deliberate physics absences (do not "fix" these): G(τ) is never
materialized — it exists only as ψψ*-phases inside the χ₀/Σ kernels;
the q→0 Coulomb head is a scalar channel threaded through every stage,
not a stage; W is evaluated at exactly the two frequencies {0, iω_p} a
one-pole model is determined by.  See ``docs/theory/physics.md``.
"""
from runtime import (
	debug_print, debug_print_enabled, initialize_communicator_stack, rank0_print,
)

#: THE startup call.  One line brings up everything below the physics: the
#: JAX env defaults, the fail-fast excepthook, the CPU-collectives
#: announcement, the CPU-only GPU-plugin skip, ``jax.distributed``, the
#: GPU-or-CPU resolution, the device mesh with every MPI/NCCL communicator it
#: needs already created, the persistent compile cache, and the rank-0 block
#: stating everything it resolved.  It MUST stay above this module's own
#: ``import jax``: the env defaults only bind before jax reads them.
#:
#: Idempotent so a library or harness that imports more than one entry point
#: in one process gets the same stack rather than a second mesh.
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import argparse
import gc
import os
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from file_io import (
    load_kin_ion_submatrix, load_centroid_basis,
)
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from common import Meta, RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange
import common.timing as timing
from .gw_config import (
	HeadCorrection, LorraxConfig, QPSolver, ScreeningDiagrams,
	refuse_unimplemented_compute_mode)
from .gw_init import (prepare_isdf_and_wavefunctions,
	                  check_band_sum_degeneracy, resolve_zeta_fit_edge,
	                  zeta_fit_band_ranges)
from .compute_vcoul import build_bgw_v_grid_fn
from .minimax_screening import build_static_quadrature
from .screening import (
	compute_screening_model, driver_persists_w0, screening_requests_for)
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
from .production_report import (
	EQP0_FILE_ROLE,
	EQP1_FILE_ROLE,
	QP_ROTATIONS_FILE_ROLE,
	QP_WFN_FILE_ROLE,
	GWProductionReport,
)
from runtime.production_stream import ProductionStdout
from .gw_output import (
	GWResults,
	print_system_summary,
	write_freq_debug,
	write_qp_wfn_oneshot,
	write_qsgw_qp_ladders,
	write_results,
)
import symmetry_maps                                            # noqa: E402


def _setup_runtime() -> None:
	"""Pre-init MPI for the one parallel-HDF5 transport.

	**phdf5 ``MPI_Init_thread``**: when the slab-IO backend is the phdf5
	FFI, eagerly enter ``MPI_THREAD_MULTIPLE`` so the first collective
	``H5Fcreate`` (in ``zeta_fit_chunked``) doesn't pay the ~400 ms
	MPI_Init cost on the critical path. A bring-up failure refuses the run.

	This is what is LEFT of the old driver-local runtime setup. Everything
	else it used to do — the JAX persistent compile cache in particular —
	moved into ``runtime.initialize_communicator_stack`` at the top of this
	module. Transport selection no longer depends on parsed configuration.
	"""
	# Unconditional since 2026-08-06: there is one transport and it always
	# needs this MPI.  This used to branch on ``config.backend.slab_io`` --
	# PHDF5_FFI got a hard refusal here, PHDF5_HOST got a printed warning,
	# H5PY_ALLGATHER got nothing -- which is why the docstring above says
	# this step could not move with the rest of the runtime setup.  With one
	# transport it no longer depends on the parsed config at all.
	#
	# NOT swallowed.  This call is the phdf5 FFI's MPI_Init_thread.  If it
	# fails, every subsequent collective H5Fcreate/H5Dwrite in the run is
	# either dead or -- worse -- silently singleton, and a singleton-MPI run
	# under independent I/O produces plausible, wrong-looking-at-nothing
	# output.  A swallowed bring-up failure here is precisely how "phdf5
	# works multi-node" and "phdf5 fails multi-node" could BOTH be believed
	# for months, so it refuses, naming the launcher flag that fixes the
	# usual cause.
	from ffi.common.ffi_loader import phdf5_init_mpi
	try:
		phdf5_init_mpi()
	except Exception as exc:
		raise RuntimeError(
			f"the phdf5 FFI's MPI bring-up ({type(exc).__name__}: {exc}) "
			f"FAILED.  Every parallel write in this run goes through that "
			f"MPI.  Usual cause: the launcher's PMI flavour does not match "
			f"the MPI library -- on Perlmutter/Shifter launch with "
			f"`srun --mpi=cray_shasta` (pmi2 and pmix both yield singleton "
			f"MPI, where every rank sees world_size==1 and independent "
			f"writes still produce a plausible file).  There is no "
			f"non-parallel writer to fall back to and that is deliberate; "
			f"see docs/architecture/slab_io.md."
		) from exc


def _compute_static_head(
		head_resolver, meta, do_screened, print0, *, require_screened=True):
	"""Resolve the q→0 head sample and its exact band-diagonal Σ terms.

	Used by every mode: the bare-X head piece applies to static and
	dynamic Σ alike (the SX/COH pieces additionally apply when screened).
	"""
	head = (head_resolver.at(0.0 + 0.0j) if require_screened
	        else head_resolver.direct_at(0.0 + 0.0j))
	print0(format_head_sample_diagnostics(head, include_screened=do_screened))
	# TODO(metal-sigma): this one-shot static Sigma head still uses the
	# ifmax band boundary.  Port it with the rest of metallic Sigma.
	occ_mask = np.arange(meta.nb_sigma, dtype=np.int32) < meta.nelec
	terms = compute_static_head_terms_from_sample(
		head, occ=occ_mask, cell_volume=meta.cell_volume, nk_tot=meta.nk_tot)
	print0(format_static_head_diagnostics(terms))
	return terms


def main(argv=None):
	_description = (
		"LORRAX GW driver — X-only / COHSEX / GN-PPM / HL-PPM / MPA "
		"self-energy, one-shot or self-consistent (see "
		"gw_config.ComputeMode / QPSolver).")
	argp = argparse.ArgumentParser(
		allow_abbrev=False, description=_description)
	argp.add_argument(
		"-i",
		"--input",
		default="cohsex_test.in",
		help="Input file",
	)
	args = argp.parse_args(argv)

	# ---- Stage timing: ONE table, and it sums to the wall -------------------
	# ``timing.reset()`` used to sit just above the ISDF call, which threw
	# away everything the prologue had already recorded.  Resetting HERE,
	# before any stage runs, is what lets the prologue appear at all.
	# ``_t_main`` is the wall this table is closed against
	# (``report(wall=...)``), so the printed rows plus ``(untimed)`` always
	# add up to the run — a reader can tell a complete accounting from a
	# partial one without doing arithmetic.
	#
	# The startup stack now runs ABOVE this reset (it runs above ``main()``),
	# so its own ``collective_warmup`` section is wiped here.  That is fine
	# and deliberate: ``initialize_communicator_stack`` measured every phase
	# itself and handed the numbers back in ``RUNTIME.facts['elapsed']``, and
	# the epilogue re-records them as a DECOMPOSITION of the pre-main span
	# rather than as extra rows.
	_t_main = time.perf_counter()
	# Work done BEFORE main(): the module body's
	# ``initialize_communicator_stack()`` (env, jax.distributed, backend
	# init, mesh + clique warm-up) and every import under it.  Measured 75.0 s
	# to first output on a cold node vs 2.1 s warm — the largest single row in
	# a small run, and previously in no row at all.
	_pre_main = timing.process_elapsed_s()
	timing.reset()

	# Configuration parsing predates the scientific report file.  Its deck
	# echo is forensic detail, so it follows the driver's ONE debug switch;
	# production output begins with GWProductionReport below.
	print0 = debug_print

	# ---- Configuration ----
	# The two orthogonal physics axes are resolved + validated up front so
	# inconsistent (qp_solver × compute_mode × accumulation) combinations
	# fail before any heavy compute (see ``LorraxConfig.qp_solver``).
	config = LorraxConfig.from_input_file(args.input, print_fn=print0)
	input_dir = config.input_dir
	qp_solver = config.qp_solver     # how QP energies are extracted from Σ
	mode = config.compute_mode       # the self-energy ansatz
	report = GWProductionReport(
		config.paths.report_file, runtime=RUNTIME,
		debug=debug_print_enabled(), stdout=rank0_print)
	production_stdout = ProductionStdout(
		debug=debug_print_enabled(), rank=RUNTIME.process_index,
		warning_fn=report.legacy_print)
	production_stdout.install()
	report.stdout = (rank0_print if debug_print_enabled()
	                 else production_stdout.emit)
	print0 = report.legacy_print
	report.begin(input_file=args.input, config=config)
	report.architecture()
	report.method(config=config)
	# A mode may be DECLARED on the axis before its Σ stage exists (today:
	# ``mpa``).  Refusing here — before the WFN read, before ISDF, before
	# any allocation is spent — is the difference between an operator
	# learning in the first second and learning after the ζ fit.  The
	# refusal names the mode; a typo'd mode value never reaches this line
	# because ``config.compute_mode`` already raised on it.
	refuse_unimplemented_compute_mode(mode, context="the LORRAX GW driver")
	# Dynamic SC currently returns two basis sectors by design: finalized H/X
	# operators are rotated back to DFT bands, while the full C(omega) operator
	# remains in QP bands for the QSGW ansatz.  Refuse that unsupported output
	# contract here, before screening or the expensive SC loop.  A second guard
	# remains at the post-Sigma seam as an invariant against future routing drift.
	if qp_solver is QPSolver.SELF_CONSISTENT and mode.is_dynamic:
		raise ValueError(
			"self-consistent dynamic EQP output is not basis-consistent: H and "
			"Sigma_x would be dft_band while the full C(omega) operator remains "
			"qp_band. Rotate the full correlation operator before running this "
			"combination.")
	do_screened = mode.needs_screening
	print0(
		f"  Head policy: head_correction={config.head.correction.value}; "
		f"screening_diagrams={config.screening.diagrams.value}; "
		f"direct diagnostic source={config.head.wcoul0_source}. "
		+ ({
			HeadCorrection.FULL: "macroscopic W, local fields exactly once",
			HeadCorrection.NO_LOCAL_FIELDS: "diagnostic epsilon head",
			HeadCorrection.OFF: "no special Gamma-cell contribution",
		}[config.head.correction]))

	# ---- The runtime is already up ----------------------------------------
	# ``RUNTIME`` was built by ``initialize_communicator_stack()`` at the top
	# of this module, above ``import jax``, because the JAX env defaults only
	# bind before jax reads them.  ``RUNTIME.mesh`` is THE run's square
	# ('x','y') mesh with every communicator it will need ALREADY created —
	# the warm-up is not optional and not the physics' job:
	#   * a mesh this process owns no device on is refused there, naming the
	#     caller, instead of surfacing a bare StopIteration deeper down;
	#   * ``warm_mesh_cliques`` (CPU/MPI) ran as well as ``nccl_warmup``
	#     (GPU/NCCL).  This driver used to call only the latter, so under
	#     ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` its MPI cliques were
	#     created incidentally, by whichever physics kernel happened to fire
	#     the first collective (``common/zeta_projection.py``,
	#     ``common/contract_bands.py`` each warm their own mesh).  That works
	#     only while those early programs stay small enough for XLA's
	#     SEQUENTIAL thunk executor; the parallel executor lands on the
	#     ``MPI_Is_thread_main`` refusal that killed the BSE TDA Lanczos
	#     (32 refusals at P=16, gate 7881216).
	# Do NOT call prepare_mesh() again here: a second Mesh object is a second
	# set of communicators and a second copy of every shape-keyed jit cache.
	mesh_xy = RUNTIME.mesh
	_setup_runtime()

	# HOW MANY HDF5 LIBRARY INSTANCES IS THIS PROCESS CARRYING?  Measured
	# from /proc/self/maps, not asserted (audit A1 fix 3).  Two
	# instances — h5py's bundled libhdf5 and the FFI's cray
	# libhdf5_parallel — alternately touching one file is the standing
	# explanation for the metallic driver's iteration-3 rank-0 segfault,
	# and until now the condition was INFERRED from the deployed
	# artifacts' NEEDED entries rather than observed in the running
	# process.  Healthy inventory is diagnostic-only and quiet by default;
	# an unsafe condition is always printed.  ``file_io.hdf5_owner`` probes
	# again after each SC iteration's store cycle, with the paths that were
	# touched through both.
	from file_io.hdf5_owner import probe as _hdf5_probe
	_hdf5_probe("startup", print_fn=print0)

	# ---- System inputs: WFN, symmetry tables, ISDF centroids ----
	wfn = WfnLoader(config.paths.wfn_file, mesh=mesh_xy)
	sym = symmetry_maps.SymMaps(wfn)
	centroid_basis = load_centroid_basis(
		config.paths.centroids_file, wfn.fft_grid, sym=sym)
	centroid_indices = centroid_basis.centroid_indices
	n_rmu = centroid_basis.n_rmu
	# This receipt is reporting evidence.  ``resolve_qgrid_symmetry_tables``
	# remains the one q-storage policy/table authority and deliberately
	# remeasures through the same symmetry_maps service at its execution seam.
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{n_rmu}.h5")
	print_system_summary(
		n_rmu=n_rmu, fft_grid=wfn.fft_grid,
		cell_volume=wfn.cell_volume, print_fn=print0,
	)

	meta = Meta.from_system(wfn, sym, config.nval, config.ncond, config.nband,
	                        n_rmu, config.bispinor,
	                        nband_chi=config.bands.chi,
	                        nband_sigma=config.bands.sigma)
	meta.rank = RUNTIME.process_index
	meta.n_proc = RUNTIME.process_count
	meta.sys_dim = config.sys_dim
	meta.bispinor = config.bispinor
	band_slices = BandSlices.from_band_edges(
		*meta.band_edges, b4_chi=meta.b_id_4_chi, b4_sigma=meta.b_id_4_sigma)
	# THE ``max`` IS NEVER SILENT.  Which of the two counts sized the ISDF ζ
	# fit is exactly the thing a reader of this log needs and cannot infer:
	# ``nband`` in the deck echo is already the max, so without this line a
	# split deck and an unsplit one at the larger count print the same
	# number.  Printed here, above the ζ fit, because this is where the
	# window it is about is decided.
	# RESOLVED, not requested: ``config.zeta_nband`` is a logical deck count
	# and the edge the fit gets is measured against the PADDED ``b4``.  This
	# banner used to print "sized for 700 bands" on a deck whose resolver and
	# memory planner were both acting on 160 (gw_init.resolve_zeta_fit_edge is
	# the one place that comparison is made).
	zeta_fit_edge = resolve_zeta_fit_edge(
		band_slices, getattr(config, "zeta_nband", None))
	print0(f"  {config.bands.describe(zeta_fit_edge)}")
	if config.bands.split:
		print0(f"    chi0/W sums bands [{band_slices.b0}, "
		       f"{band_slices.b4_chi}); Sigma sums bands [{band_slices.b0}, "
		       f"{band_slices.b4_sigma}); psi is LOADED over "
		       f"[{band_slices.b0}, {band_slices.b4}) "
		       f"(padded from {meta.b_id_4_user} to the world size).")
	check_band_sum_degeneracy(wfn, config, band_slices, log=print0)

	# ---- sigma_omega_layout=sharded: resolve-time geometry/backend gate ----
	# The config-level axis checks (self_consistent) already ran
	# in ``config.qp_solver``; the two conditions below need the mesh and the
	# σ window, known only here.  Refusing NOW costs seconds; refusing at the
	# Σ stage would waste the whole ζ fit (pattern #6: the resolve-time check
	# must test what will execute).  The Σ driver re-checks divisibility at
	# its own seam as the last-line guard.
	# THE SAME precondition the Σ drivers enforce, asked EARLY.  It is one
	# function (``ppm_sigma.assert_sharded_sigma_window_divides_mesh``); this
	# used to be a third hand-written copy of the modulus test, alongside the
	# PPM branch's and — absent entirely — the MPA executor's.
	#
	# ``compute_mode = mpa`` reaches it whatever the deck says about the
	# layout, because the MPA executor emits the sharded cube unconditionally
	# (the dispatch refuses ``replicated`` by name for that reason).
	_wants_sharded_cube = (
		config.sigma.omega_layout == "sharded"
		or getattr(mode, "value", str(mode)) == "mpa")
	if mode.is_dynamic and _wants_sharded_cube:
		from .ppm_sigma import assert_sharded_sigma_window_divides_mesh
		_p_x = int(mesh_xy.devices.shape[0])
		_p_y = int(mesh_xy.devices.shape[1])
		_nbs = int(meta.nb_sigma)
		assert_sharded_sigma_window_divides_mesh(
			_nbs, mesh_xy,
			ansatz=f"compute_mode = {getattr(mode, 'value', mode)}")
		# The second refusal that used to live here -- sharded layout with
		# slab_io=h5py_allgather at P>1, which would have re-introduced the
		# full Σ_c(ω) cube gather inside the sigma_mnk.h5 writer -- is gone
		# with the tier.  It was door 7 of 7 and the only one that read
		# ``jax.process_count()`` raw instead of the launcher-aware count,
		# so it was also the weakest.  Nothing can select that writer now.
		print0(
			f"  sigma_omega_layout = sharded: Σ_c(ω,k,m,n) stays "
			f"(m_X, n_Y)-tiled on the {_p_x}x{_p_y} mesh end-to-end "
			f"(consumers read tiles; no full-cube replication).")

	# DFT eigenvalues on the Σ band window (Ry) — one fetch, reused by the
	# Σ_X diagnostic, the SC initial state, degeneracy averaging, and the
	# results writer.
	enk_dft, _ = get_enk_bandrange(
		wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
		nspinor=meta.nspinor)
	_zeta_ranges = zeta_fit_band_ranges(
		band_slices, zeta_fit_edge, log=lambda *args, **kwargs: None)
	report.environment(config=config, wfn=wfn)
	report.sampling(wfn=wfn, sym=sym, centroids=centroid_basis)
	report.bands(
		config=config, wfn=wfn, band_slices=band_slices,
		zeta_ranges=_zeta_ranges)

	# Single resolver for every q→0 head sample we'll need this run; the
	# COHSEX static head, the W0 restart-flush head, and the PPM dynamic
	# head all read from the same plumbing (overrides → epshead → s_tensor)
	# so they share one cache.  See ``head_correction.HeadResolver``.
	head_resolver = HeadResolver(config, input_dir, wfn, sym, meta, print0)

	# Optional BGW vcoul override (purely diagnostic — bit-reproducible BGW
	# comparisons).  Returns None when ``use_bgw_vcoul`` is False.
	bgw_v_grid_fn = build_bgw_v_grid_fn(
		config, wfn=wfn, sym=sym, input_dir=input_dir, print_fn=print0)

	# Everything from ``main()`` entry to here is the driver PROLOGUE:
	# config parse, the PHDF5 MPI pre-init, WFN + symmetry +
	# centroid reads, the head resolver.  (The mesh, its collective warm-up
	# and the compile cache are NOT here any more — they happen above
	# ``main()`` in ``initialize_communicator_stack`` and are reported as
	# their own rows.)  It is
	# executed exactly once and, on a cold node, it is the largest single row
	# in this table (75.0 s to first output, job 7881949) — so it is named.
	# ``timing.record`` rather than a ``with`` block deliberately: the block
	# above is another workstream's and must not be re-indented for a timer.
	timing.record("gw_jax.startup", time.perf_counter() - _t_main)

	# ISDF fitting or restart loading
	with timing.section("gw_jax.isdf", announce=True,
	                    label="ISDF basis + wavefunctions"):
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
	# LOUD guard (quality pattern #7): the Σ kernels' Σ^B fold-in is a
	# structural no-op when ``wfns_transverse``/``bispinor_v_q_path`` is
	# None — a bispinor run reaching Σ without them would exit rc=0 with
	# Σ^B silently dropped.  Both producer paths (fit + restart) raise
	# with specifics before this point; this is the last-line invariant.
	if config.bispinor and wfns_transverse is None:
		raise RuntimeError(
			"bispinor = true but no transverse-centroid Wfns bundle was "
			"produced (Σ^B would be silently dropped).  Check "
			"centroids_file_current and, on restart, that the restart "
			"file carries psi_full_y_transverse.")
	bispinor_v_q_path = (
		os.path.join(tmp_dir, 'v_q_bispinor.h5')
		if wfns_transverse is not None else None
	)
	if bispinor_v_q_path is not None and not os.path.exists(bispinor_v_q_path):
		raise RuntimeError(
			f"bispinor: {bispinor_v_q_path} is missing.  The V^{{i,j}} "
			f"tile file is written by the non-restart pipeline "
			f"(compute_V_q); on restart it must still be present in "
			f"tmp/.  Rerun with restart = false to regenerate it.")

	# ---- Screening: χ₀ → W = (1 − Vχ)⁻¹ V at every ω the Σ scheme needs ----
	# X_ONLY requests no screening at all.
	V_q = V_qmunu               # flat-q (nq, μ, μ) — compute and restart alike
	quad, e_ref = None, None
	if do_screened:
		# The minimax τ-axis, solved on G's actual spectral range — shared
		# by every χ₀ build this run (static + probe W here, SC re-solves).
		# TIMED because it is the classic mis-attribution on this path: the
		# crossing-minimax solve costs ~95 s cold with no cache and no
		# shipped table (XPROF_TRACE_GUIDE §"Known LORRAX cost centers"),
		# and with no row of its own that 95 s reads as "GW startup".
		with timing.section("gw_jax.minimax_quadrature", announce=True,
		                    label="minimax tau-axis"):
			# The minimax service announces every served/uncertified rule with
			# ``warnings.warn`` in each process.  This request is deterministic
			# and collective, so keep its loud provenance warning once on the
			# output owner instead of repeating it P times.  The filter is local
			# to this call; exceptions and warnings from later rank-local work
			# remain untouched.
			with warnings.catch_warnings():
				if jax.process_index() != 0:
					warnings.simplefilter("ignore", RuntimeWarning)
				quad, e_ref = build_static_quadrature(
					wfns, config.minimax_config, print_fn=print0)

	# One-shot and QSGW now share one response/finalization implementation.
	# Build the irreducible DFT tensor and its wings on the exact chi0 band
	# manifold before W, then retain only the tiny finalized head after W.
	oneshot_head_response = None
	oneshot_head_requests = None
	oneshot_mpa_plan = None
	if (do_screened
			and config.head.correction is HeadCorrection.FULL
			and config.screening.diagrams is ScreeningDiagrams.W_RPA
			# Every self-consistent mode builds its exact frequency plan and
			# response inside the map.  A pre-map response would exist only to
			# seed a restart artifact and could be mistaken for final physics.
			and qp_solver is not QPSolver.SELF_CONSISTENT):
		from .qsgw_head import build_dft_head_response
		if mode.value == "mpa":
			from .mpa import sample_plan
			from .mpa.model import make_mpa_plan
			oneshot_mpa_plan = make_mpa_plan(config, quad)
			oneshot_omegas = np.asarray(
				sample_plan.plan_z(oneshot_mpa_plan), dtype=np.complex128)
		else:
			oneshot_head_requests = screening_requests_for(mode, config)
			oneshot_omegas = np.asarray(
				[complex(r.omega_ry) for r in oneshot_head_requests],
				dtype=np.complex128)
		if oneshot_omegas.size:
			oneshot_head_response = build_dft_head_response(
				wfns, oneshot_omegas, input_dir=input_dir, mesh=mesh_xy,
				wfn=wfn, meta=meta, config=config)
			print0(
				"  head_correction=full: built direct DFT response and "
				"head/body wings on the chi0 transition manifold; finalizing "
				"once against the resident W(Gamma).")
	# SC solves W inside each map and persists only the final accepted map.
	# Do not perform a redundant DFT screening solve here: besides its cost,
	# that seed body used to survive long enough to be paired with a final head.
	if qp_solver is QPSolver.SELF_CONSISTENT:
		W_by_role = {}
	else:
		with timing.section("gw_jax.screening", announce=True,
		                    label="screening (chi0 -> W)"):
			W_by_role = compute_screening_model(
				mode, wfns, V_q, quad=quad, e_ref=e_ref, sym=sym,
				centroid_indices=centroid_indices, config=config, meta=meta,
				mesh_xy=mesh_xy, run_dir=os.path.join(tmp_dir, "mpa"), wfn=wfn,
				label="oneshot", head_resolver=head_resolver,
				head_channel=getattr(isdf, 'head_channel', None),
				mpa_plan=oneshot_mpa_plan,
				iteration_head_response=oneshot_head_response,
				tensors_filename=tensors_filename,
				print_fn=print0)

	if oneshot_head_response is not None:
		if mode.value == "mpa":
			final_head = W_by_role.get("iteration_head")
			if final_head is None:
				raise RuntimeError(
					"head_correction=full: MPA screening returned no finalized "
					"head samples")
		else:
			from .qsgw_head import finalize_iteration_head_samples
			final_head = finalize_iteration_head_samples(
				oneshot_head_response, wfn=wfn, meta=meta, config=config,
				mesh=mesh_xy, requests=oneshot_head_requests,
				W_by_role=W_by_role)
		head_resolver.install_samples(final_head.samples)

	# Persist W0_qmunu + q=0 head scalars to the ISDF restart file for
	# downstream consumers (BSE, future Σ-builders); no-op unless screened
	# and the restart file exists.
	# TIMED, and it was not.  The stage was measured at ~1.7 MB/s
	# aggregate and 2 h 55 m of total silence at c2406 (AF.4c), back when
	# this call gathered the whole (nq, μ, μ) W0 onto one rank on the
	# ``h5py_allgather`` backend to write it.  That backend is gone
	# (233a830d) and the write is SlabIO's per-rank tile path now, so the
	# number is history, not a prediction — yet the call still sat
	# between two timed stages with no
	# section of its own, so it appeared in the run's wall clock and in
	# NO row of the stage table.  Naming it is the precondition for
	# anyone attributing that wall time (the write path itself is
	# workstream AE/AF's; this is the instrument, not the fix).
	# ``sym``/``centroid_indices`` below are for the q-storage resolution
	# ONLY (see the callee): W0 must land on the same q-set V did, and the
	# way to be sure of that is to ask the same resolution point about the
	# same centroid set rather than to infer it from a shape.
	if (driver_persists_w0(mode, config)
			and qp_solver is not QPSolver.SELF_CONSISTENT):
		with timing.section("gw_jax.persist_w0"):
			from .gw_output import persist_w0_and_head
			persist_w0_and_head(
				W_by_role.get("static", V_q),
				tensors_filename=tensors_filename, head_resolver=head_resolver,
				config=config, meta=meta, mesh_xy=mesh_xy,
				sym=sym, centroid_indices=centroid_indices,
				print_fn=print0)

	# q→0 head correction.  The bare-X head is the same physical quantity in
	# both COHSEX and PPM modes; gating this on ``not use_ppm_sigma`` was
	# the original ``Bare Σ_X missing q→0 head'' bug (skill compare/SKILL.md
	# §4i).  The SX/COH head pieces are also attached to the static
	# sig_sx/sig_coh in compute_cohsex_sigma, but for PPM those static values
	# are overwritten downstream (sig_sx ← sig_x, sig_c ← PPM-evaluated
	# correlation), so only the X-head survives — which is the piece needed.
	static_head_terms = None
	if config.do_G0:
		# A screened SC+FULL map always builds/folds its own head.  Supplying
		# and printing a direct DFT seed here would be false provenance even
		# though the map later replaces it.  OFF/NLF and unscreened X_ONLY keep
		# the direct/default route because that is the policy they consume.
		_sc_full = (qp_solver is QPSolver.SELF_CONSISTENT
		            and do_screened
		            and config.head.correction is HeadCorrection.FULL)
		if not _sc_full:
			with timing.section("gw_jax.static_head"):
				static_head_terms = _compute_static_head(
					head_resolver, meta, do_screened, print0,
					# MPA has no {0,probe} persistence grid; its one-shot
					# static contribution here is bare X from the direct sample.
					require_screened=(mode.value != "mpa" and
					                  qp_solver is not QPSolver.SELF_CONSISTENT))

	# ---- Σ_xc + V_H: ONE dispatch for every mode ----
	# The same ``compute_sigma_xc`` call the SC iteration map makes each
	# step — static COHSEX kernels for X_ONLY/COHSEX, the PPM pipeline
	# (fit → 4-branch τ-integration → analytic q→0 head → at-DFT interp)
	# for the dynamic modes, with the QSGW-symmetrised Σ_xc evaluated at
	# E_DFT (a one-shot full-matrix effective Hamiltonian, distinct from the
	# fixed-state diagonal G0W0 output; ``solve_qp`` re-evaluates for
	# fixed_point).
	# SC-iteration-1 ≡ this call, pinned by tests/test_invariance_gates.py
	# ::test_sc_iteration1_equals_one_shot.
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
				quad=quad,
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
	# Called for its gate side effect only: it RAISES on any provenance
	# disagreement and prints the file's V_H storage summary.  The returned
	# attrs dict is deliberately not bound — the V_H routing this run
	# actually uses is re-resolved from the file by resolve_hartree_source
	# below, and it is THAT resolution (hartree_source /
	# kin_ion_has_hartree) which flows into the GWResults output
	# provenance (release audit 2026-07-28: the previous ``kin_ion_attrs``
	# binding was dead).
	# TIMED as one row: the gate, the source resolution and the slab read are
	# a single logical stage ("get H₀ off disk") and the read is a distributed
	# H5 slab load whose cost is a file-system property, not a physics one.
	_t_kin = time.perf_counter()
	validate_kin_ion_against_run(
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
		mesh=mesh_xy,
	)
	timing.record("gw_jax.kin_ion_load", time.perf_counter() - _t_kin)

	# ---- update_H[Σ; qp_solver] — all branches yield ``sigma_total``
	# (Σ_xc + V_H, Ry, DFT basis, replicated) whose eigh gives E_qp/U_qp.
	# ``rotations_written`` is run_sc_driver's own report of whether it
	# wrote qp_wfn_rotations.h5; the writer below reads the fact rather
	# than re-deriving the predicate.
	rotations_written = False
	final_static_head_terms = static_head_terms
	if qp_solver is QPSolver.SELF_CONSISTENT:
		# SC-QSGW: iterate ψ-rotation → χ₀ → W → Σ_xc (the same
		# compute_sigma_xc dispatch, mode-agnostic) to the fixed point;
		# the returned SigmaResult is already rotated back to the DFT
		# basis and its sigma_omega_h5_path points at the converged
		# single-write sigma_mnk.h5.  See ``sc_iteration.run_sc_driver``.
		from .sc_iteration import run_sc_driver
		# Executed once; it is the whole SC loop, so it is the run's biggest
		# row when it fires and must not hide inside ``(untimed)``.
		with timing.section("gw_jax.sc_driver", announce=True,
		                    label="self-consistent QSGW driver"):
			sc_result = run_sc_driver(
				wfns, V_q, kin_ion,
				head_channel=getattr(isdf, 'head_channel', None),
				quad=quad, e_ref=e_ref,
				static_head_terms=static_head_terms,
				head_resolver=head_resolver,
				config=config, meta=meta, mesh_xy=mesh_xy,
				sym=sym, wfn=wfn, centroid_indices=centroid_indices,
				band_slices=band_slices, input_dir=input_dir,
				tensors_filename=tensors_filename,
				enk_dft=enk_dft, print_fn=print0)
		sigma_result = sc_result.sigma_result_dft
		sigma_total = sc_result.sigma_total_dft
		rotations_written = sc_result.rotations_written
		final_static_head_terms = sc_result.static_head_terms_dft
	else:
		# One-shot: ``one_shot_dft`` = Σ_xc was already QSGW-built at
		# E_DFT inside compute_sigma_xc (pass-through; also covers static
		# modes and the streamed-Σ_c stand-in); ``fixed_point`` = diagonal
		# on-shell solve + scissor + QSGW rebuild at the solved energies.
		# eqp0.dat/eqp1.dat are at-DFT in every case (written downstream
		# from ``sigma_c_at_dft_ev`` / the ω-grid diag, not from here).
		with timing.section("gw_jax.solve_qp"):
			sigma_total = solve_qp(
				qp_solver, sigma_result, kin_ion,
				config=config, meta=meta, mesh_xy=mesh_xy, print_fn=print0)

	# Optional additional ladder: iterate ONLY the already-built, full-matrix
	# one-shot Sigma(omega).  This deliberately sits after the sole screening /
	# Sigma stage and calls neither dispatch, so write_eqp2 cannot accidentally
	# turn into a second GW calculation.  The callee rotates the fixed cube into
	# each updated QP basis before evaluating its off-diagonals.
	eqp2_result = None
	if config.eqp2.enabled:
		from .sc_iteration import run_fixed_sigma_evsc
		with timing.section("gw_jax.eqp2_evsc", announce=True,
		                    label="fixed-Sigma eigenvalue self-consistency"):
			eqp2_result = run_fixed_sigma_evsc(
				sigma_result, kin_ion, enk_dft,
				config=config, meta=meta, band_slices=band_slices, wfn=wfn,
				mesh_xy=mesh_xy, print_fn=print0)

	# ---- Post-Σ seam: bare locals from the SigmaResult ----
	# One extraction for SC and one-shot alike; PPM-only fields are None
	# in static modes.
	#
	# TWO BASES ON ONE OBJECT on the SC path, by design: the finalize
	# rotated ``sigma_dispatch.ROTATED_TO_DFT_FIELDS`` (sig_h, sig_x,
	# sig_sx, sig_coh below) back to the DFT basis and left
	# ``SIGMA_BASIS_FIELDS`` (sigma_c_omega, sigma_c_at_dft_ev) in the QP
	# basis, where the QSGW ansatz
	# that consumes Σ_c(ω) is defined.  Their band diagonals meet in
	# ``sigma_xc_at_dft_ev`` below and in eqp{0,1}.dat
	# (``eqp_bgw.compute_eqp_diag``); that sum is basis-consistent only
	# at U = identity.  One-shot: every field is DFT basis and the
	# question does not arise.  The separable analytic head diagonal is cheap
	# enough to rotate without materialising its dense matrix; SC returns that
	# as a separate DFT-basis diagnostic while leaving SigmaResult's
	# basis-of-computation field untouched.
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
	# The energies THIS Σ was evaluated at (E_DFT one-shot, the map's input
	# QP energies under SC).  Not degen-averaged below with the Σ channels:
	# it is a spectrum, not a self-energy component.
	e_eval_ev           = sigma_result.e_eval_ev
	efermi_dft_ev       = sigma_result.efermi_dft_ev
	sigma_c_omega       = sigma_result.sigma_c_omega_kij_ry
	if (qp_solver is QPSolver.SELF_CONSISTENT
			and sigma_c_omega is not None):
		raise ValueError(
			"self-consistent dynamic EQP output is not basis-consistent: H and "
			"Sigma_x have been rotated to dft_band, while the full C(omega) "
			"operator remains qp_band. Refusing to combine their diagonals until "
			"the full C(omega) operator is rotated consistently.")
	head_sigma_diag_w_kn_ry = (
		sc_result.head_sigma_diag_dft_w_kn_ry
		if qp_solver is QPSolver.SELF_CONSISTENT
		else sigma_result.head_sigma_diag_w_kn_ry)
	omega_grid_ev = (
		np.asarray(sigma_result.omega_grid_ev, dtype=np.float64)
		if sigma_result.omega_grid_ev is not None else None)
	omega_grid_ry = (
		np.asarray(sigma_result.omega_grid_ry, dtype=np.float64)
		if sigma_result.omega_grid_ry is not None else None)
	# ---- BGW-style degenerate-set averaging at the H-build seam ----
	# (mirrors Sigma/shiftenergy.f90; see ``degen_average``).
	if not config.no_degen_averaging:
		(sigma_total, sig_sx, sig_coh, sig_h, sig_x,
		 sigma_c_at_dft_ev) = average_sigma_components(
			sigma_total, sig_sx, sig_coh, sig_h, sig_x, sigma_c_at_dft_ev,
			energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
			tol_ry=float(config.degen_avg_tol_ry),
			mesh_xy=mesh_xy)
		# Head-only debug columns must undergo the SAME DFT-degenerate-set
		# averaging as the Sigma components they decompose.  After the QP->DFT
		# diagonal transform an occupied-projector contribution need not be
		# identical in an arbitrary basis inside a degenerate manifold.
		def _average_head_diag(diag):
			arr = np.asarray(diag)
			if arr.ndim == 1:
				arr = np.broadcast_to(arr, np.asarray(enk_dft).shape)
			if arr.ndim in (2, 3):
				return average_within_degenerate_sets(
					arr, energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
					tol_ry=float(config.degen_avg_tol_ry))
			raise ValueError(
				f"head diagnostic has unsupported shape {arr.shape}")

		if final_static_head_terms is not None:
			import dataclasses
			final_static_head_terms = dataclasses.replace(
				final_static_head_terms,
				sigma_x_diag=_average_head_diag(
					final_static_head_terms.sigma_x_diag),
				sigma_sx_diag=_average_head_diag(
					final_static_head_terms.sigma_sx_diag),
				sigma_sx_minus_x_diag=_average_head_diag(
					final_static_head_terms.sigma_sx_minus_x_diag),
				sigma_coh_diag=_average_head_diag(
					final_static_head_terms.sigma_coh_diag),
			)
		if head_sigma_diag_w_kn_ry is not None:
			head_sigma_diag_w_kn_ry = _average_head_diag(
				head_sigma_diag_w_kn_ry)

	# Σ_xc(E_DFT) diagonal (eV) — drives eqp_g0w0.dat (PPM one-shot
	# only).  Form it AFTER the one canonical conditioning seam above: forming
	# this sum before ``average_sigma_components`` made eqp_g0w0 retain raw,
	# unequal degenerate diagonals even while every other live text output used
	# the conditioned X/C pair.  With averaging disabled the seam is a no-op,
	# so this same expression deliberately preserves the raw red twin.
	sigma_xc_at_dft_ev = (
		np.diagonal(np.asarray(sig_x), axis1=1, axis2=2) * RYD_TO_EV
		+ sigma_c_at_dft_ev
		if sigma_c_at_dft_ev is not None else None)

	# ---- Single H-build + diagonalization on replicated arrays ----
	# Gate the two inputs to the QP diagonalization *before* eigh: LAPACK
	# on a NaN-bearing matrix returns without complaining, and the garbage
	# then propagates into eqp0/eqp1/WFN_qp.h5 with rc=0.  ``kin_ion`` also
	# comes off disk (kin_ion.h5), so this doubles as the content check on
	# that interface.
	# REFUSALS, not warnings, and that distinction is the 2026-08-15 bcc-Fe
	# finding: an all-NaN Σ_c produced an all-NaN E_QP column, a NaN E_F and
	# a NaN scissor fit, and the run exited rc=0 in 883 s with only a warning
	# line to show for it (JID 57051742, CLAIMS 204).  The SC path catches
	# that on its SECOND map call, through
	# ``_solve_head_occupations -> OccupationState`` finiteness; a one-shot
	# run has no second map call, so this seam -- which both paths cross -- is
	# where the guard has to be.  ``LORRAX_ALLOW_NONFINITE_RESULT=1`` is the
	# named escape for forensics.
	from common import sanity
	sanity.refuse_nonfinite("kin_ion (from kin_ion.h5)", kin_ion,
	                        print_fn=print0,
	                        detail="kin_ion.h5 is the mean-field side of H0; "
	                               "regenerate it from THIS run's input file.")
	sanity.refuse_nonfinite(
		"Σ_total (Σ_xc + V_H)", sigma_total, print_fn=print0,
		detail="A finite, well-conditioned pole set producing a non-finite "
		       "Σ is a defect in the contraction or in what was handed to "
		       "it, not a convergence problem.")

	# TIMED: nk independent (nb_sigma, nb_sigma) Hermitian eigensolves.  It is
	# one statement and normally seconds, but it is O(nk·nb³) and it is the
	# only dense LAPACK call on the post-Σ path, so it is the row that tells
	# you when the band window (not the physics) became the cost.
	with timing.section("gw_jax.qp_eigh") as _sec_eigh:
		H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)
		_sec_eigh.watch(E_full, U_full)
	sanity.refuse_nonfinite(
		"E_qp (eigh of H_QP)", E_full, print_fn=print0,
		detail="LAPACK returns without complaining on a NaN-bearing matrix, "
		       "so this is the last place a NaN spectrum can be stopped "
		       "before eqp0/eqp1/WFN_qp.h5.")
	_t_out = time.perf_counter()

	# ---- One-shot WFN_qp.h5 dump (drop-in BSE / restart input).  SC
	# already wrote its own WFN_qp.h5 above via dump_qp_wfn_artifacts
	# (using state_final.H_qp_dft) — same physics, slightly different
	# numerics from the post-Σ-seam eigh path.  Skip the second write
	# in SC to avoid clobbering.
	if config.debug.write_wfn_h5 and qp_solver is not QPSolver.SELF_CONSISTENT:
		write_qp_wfn_oneshot(
			U_full, E_full, wfn=wfn, band_slices=band_slices,
			input_dir=input_dir, qp_solver=qp_solver, print_fn=print0)

	# PPM mode: feed the writer the on-shell diag(Σ_c(E_DFT)) (Ry) so the
	# eqp0.dat "sigC" column reports dynamic correlation directly comparable
	# to BGW's (SX-X)+CH at Eo=E_DFT.  Off-diagonals stay zero — the full
	# Σ_c(ω, k, i, j) tensor is in sigma_mnk.h5 for callers that need them.
	# Σ_c diagonal on the ω-grid: feed the eqp1.dat writer's central-diff
	# Z-factor.  Pulled from the on-device sharded tensor when available.
	if sigma_c_omega is not None:
		sigma_c_omega_diag_ev = np.asarray(extract_sigma_diag_replicated(
			sigma_c_omega, mesh_xy)) * RYD_TO_EV
		if not config.no_degen_averaging:
			# Output-only diagonal curve.  The full persisted operator stays raw,
			# while this curve uses the SAME canonical group owner at every omega;
			# assemble_eqp consequently derives C(E_DFT) and Z from one function.
			sigma_c_omega_diag_ev = average_within_degenerate_sets(
				sigma_c_omega_diag_ev,
				energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
				tol_ry=float(config.degen_avg_tol_ry))
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
		e_eval_ev=e_eval_ev,
		efermi_ev=efermi_dft_ev,
		sigma_omega_h5_path=sigma_omega_h5_path,
		tensors_filename=tensors_filename,
		kin_ion_has_hartree=kin_ion_has_hartree,
		hartree_source=hartree_source,
		E_eqp2_ry=(None if eqp2_result is None
		             else np.asarray(eqp2_result.energies_ry)),
		eqp2_iterations=(None if eqp2_result is None
		                 else int(eqp2_result.iterations)),
		eqp2_residual_ev=(None if eqp2_result is None
		                  else float(eqp2_result.residual_ev)),
		eqp2_tol_ev=(None if eqp2_result is None
		             else float(config.eqp2.tol_ev)),
	)
	if meta.rank == 0:
		# Optional Σ-decomposition debug table (no-op unless
		# ``debug.sigma_freq_debug_output``; see ``gw_output.write_freq_debug``).
		write_freq_debug(
			results, config=config,
			static_head_terms=final_static_head_terms,
			omega_dft_rel_ev=omega_dft_rel_ev,
			head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
			omega_grid_ry=omega_grid_ry,
			sym=sym,
			e_eval_ev=e_eval_ev,
			print_fn=print0,
		)
		# The QP-ladder half of sigma_mnk.h5's opt-in plotting appendix
		# (no-op unless ``write_qsgw_datasets``).  HERE and not at the Σ
		# seam because two of the three ladders need ``kin_ion``, which
		# ``compute_sigma_xc`` never sees; the QSGW cube itself was
		# already appended by whichever path wrote the file
		# (``qsgw_utils.write_qsgw_sigma_cube``).  Rank-0 and barrier-free
		# like every other writer in this block: eigenvalues are basis-
		# free, so this one seam is correct for the one-shot and the
		# self-consistent paths alike.
		write_qsgw_qp_ladders(
			results, config=config,
			e_qp_ry=results.E_qp_ry,
			sigma_c_omega_diag_ev=sigma_c_omega_diag_ev,
			omega_grid_ev=omega_grid_ev,
			sigma_c_omega=sigma_c_omega,
			print_fn=print0,
		)
		# Degen averaging was applied once at the H-build seam upstream;
		# the writer just serializes the already-averaged Σ components.
		write_results(
			results,
			sigma_diag_file=config.paths.sigma_diag_file,
			eqp0_file=config.paths.eqp0_file,
			eqp1_file=config.paths.eqp1_file,
			eqp2_file=config.paths.eqp2_file,
			input_dir=input_dir,
			kgrid=meta.kgrid,
			# THE symmetry object, not tables off it.  Every k-basis
			# decision is made where the data is written, through
			# ``symmetry_maps.reduce_full_bz_to_file_wedge``.
			sym=sym,
			write_qp_rotations=not rotations_written,
			qp_rotations_k_storage=config.qp_rotations_k_storage,
			qp_solver=qp_solver,
			print_fn=print0,
		)
		timing.record("gw_jax.output", time.perf_counter() - _t_out)
	if _pre_main is not None:
		# DECOMPOSE the pre-main span; do not add rows to it.  The entry
		# point timed its own phases (it happened before ``timing.reset()``,
		# so its own ``collective_warmup`` section was wiped), and the
		# remainder of ``_pre_main`` is the import storm — 75.0 s cold vs
		# 2.1 s warm, job 7881949.  Recording the phases AND the whole span
		# would double-count and break the table's "rows + (untimed) ==
		# wall" property, which is the only thing that lets a reader tell a
		# complete accounting from a partial one.
		_phases = RUNTIME.facts.get("elapsed", {})
		for _phase, _secs in sorted(_phases.items()):
			if _phase != "total":
				timing.record(f"gw_jax.runtime_stack.{_phase}", _secs)
		timing.record("gw_jax.imports",
		              max(_pre_main - _phases.get("total", 0.0), 0.0))
	_wall = time.perf_counter() - _t_main + (_pre_main or 0.0)
	if meta.rank == 0 and debug_print_enabled():
		# ``wall=`` closes the table: printed rows + ``(untimed)`` == the
		# whole PROCESS when /proc gave us the pre-main span, else main().
		timing.report(print_fn=print0, title="--- Timing ---", wall=_wall)

	report.sigma_coverage(
		config=config, band_slices=band_slices, enk_dft_ry=enk_dft,
		sigma_result=sigma_result)
	report.qp_gap(
		band_slices=band_slices, e_dft_ry=enk_dft, e_qp_ry=E_full)
	if eqp2_result is not None:
		report.eqp2_summary(
			band_slices=band_slices,
			e_eqp2_ry=eqp2_result.energies_ry,
			iterations=eqp2_result.iterations,
			residual_ev=eqp2_result.residual_ev,
			tol_ev=config.eqp2.tol_ev)
	_file_rows = [
		("input deck", "read", args.input),
		("DFT wavefunctions", "read", config.paths.wfn_file),
		("ISDF centroids", "read", config.paths.centroids_file),
		("mean-field Hamiltonian", "read", config.paths.kin_ion_file),
		("long-wave dipoles", "read" if os.path.exists(os.path.join(
			input_dir, "dipole.h5")) else "absent",
		 os.path.join(input_dir, "dipole.h5")),
		("parallel transport", "read" if os.path.exists(
			config.paths.parallel_transport_file) else "absent",
		 config.paths.parallel_transport_file),
		("ISDF restart tensors", "present" if os.path.exists(tensors_filename)
		 else "absent", tensors_filename),
		("self-energy table", "written" if os.path.exists(
			config.paths.sigma_diag_file) else "absent",
		 config.paths.sigma_diag_file),
		(EQP0_FILE_ROLE,
		 "written" if os.path.exists(config.paths.eqp0_file)
		 else "absent", config.paths.eqp0_file),
		(EQP1_FILE_ROLE,
		 "written" if os.path.exists(
			config.paths.eqp1_file) else "absent", config.paths.eqp1_file),
		(QP_ROTATIONS_FILE_ROLE, "written" if os.path.exists(
			os.path.join(input_dir, "qp_wfn_rotations.h5")) else "absent",
		 os.path.join(input_dir, "qp_wfn_rotations.h5")),
		(QP_WFN_FILE_ROLE, "written" if os.path.exists(
			os.path.join(input_dir, "WFN_qp.h5")) else "absent",
		 os.path.join(input_dir, "WFN_qp.h5")),
	]
	if config.eqp2.enabled:
		_file_rows.append((
			"fixed-Sigma evSC energies", "written" if os.path.exists(
				config.paths.eqp2_file) else "absent", config.paths.eqp2_file))
	if config.paths.centroids_file_current:
		_file_rows.insert(3, (
			"current centroids", "read", config.paths.centroids_file_current))
	if sigma_omega_h5_path:
		_file_rows.append((
			"dynamic Sigma spectrum",
			"written" if os.path.exists(sigma_omega_h5_path) else "absent",
			sigma_omega_h5_path))
	_file_rows.append(("calculation report", "written", report.path))
	report.timings(timing.records(), wall=_wall)
	report.warnings()
	report.files(_file_rows)
	report.finish()
	production_stdout.close()

	return 0


if __name__ == "__main__":
	from runtime import run_main_and_finalize
	run_main_and_finalize(main)

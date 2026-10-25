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
    E_qp, U_qp      = eigh(kin_ion + Σ_total)
	eqp0/eqp1/σ.dat = write_results(...)            # writers, debug tables       (gw_output)

Two orthogonal config axes pivot the flow: ``compute_mode`` — the
self-energy ansatz (``x_only`` / ``cohsex`` / ``gn_ppm`` / ``hl_ppm`` /
``mpa``) — and ``qp_solver`` — how QP energies are extracted
from Σ (``one_shot_dft`` / ``self_consistent``).  The
self-consistent path iterates the same ``compute_sigma_xc`` dispatch;
iteration 1 reproduces the one-shot result exactly (gated by
``tests/test_invariance_gates.py::test_sc_iteration1_equals_one_shot``).

Deliberate physics absences (do not "fix" these): G(τ) is never
materialized — it exists only as ψψ*-phases inside the χ₀/Σ kernels;
the q→0 Coulomb head is a scalar channel threaded through every stage,
not a stage; W is evaluated at exactly the two frequencies {0, iω_p} a
one-pole model is determined by.  See ``docs/theory/physics.md``.
"""
import argparse
import contextlib


def build_parser() -> argparse.ArgumentParser:
	"""The CLI.  ABOVE the startup call so ``--help`` can reach it.

	Needs nothing but ``argparse``, which is what makes the seam below
	possible; see :mod:`runtime.cli_seam`.
	"""
	argp = argparse.ArgumentParser(
		allow_abbrev=False,
		description=(
			"LORRAX GW driver — X-only / COHSEX / GN-PPM / HL-PPM / MPA "
			"self-energy, one-shot or self-consistent (see "
			"gw_config.ComputeMode / QPSolver)."))
	argp.add_argument(
		"-i",
		"--input",
		default="cohsex_test.in",
		help="Input file",
	)
	return argp


if __name__ == "__main__":
	# Argv is answered before any runtime exists — runtime/cli_seam.py.
	from runtime.cli_seam import refuse_bad_argv
	refuse_bad_argv(build_parser())

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

import gc
import os
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from file_io import (
    load_kin_ion_submatrix, load_centroid_basis,
)
from wfn_loader import WfnLoader                                    # noqa: E402
from common import Meta, RYD_TO_EV
from common.log import get_logger, message
from common.wfn_transforms import get_enk_bandrange
import common.timing as timing

log = get_logger(__name__)

from .gw_config import (
	ComputeMode, HeadCorrection, LorraxConfig, QPSolver,
	ScreeningDiagrams, coerce_screening_diagrams, incumbent_bispinor_head_record,
	packed_bare_transverse_route,
	packed_photon_replaces_charge_sigma, packed_photon_screens_current,
	refuse_unimplemented_compute_mode, uses_dynamic_packed_photon_route,
	folds_scalar_head, uses_direct_bispinor_tangential_head,
	uses_full_bispinor_tangential, uses_bare_transverse_tangential,
	uses_four_spinor_finite_q_charge, uses_static_photon_response,
	infer_material_class, resolve_mpa_sampling_alpha,
	validate_material_inputs)
from .gw_init import (prepare_isdf_and_wavefunctions,
                      prepare_band_metadata,
	                  check_band_extrapolation_floor,
	                  check_band_sum_degeneracy, resolve_zeta_fit_edge,
	                  zeta_fit_band_ranges)
from .compute_vcoul import build_bgw_v_grid_fn
from .minimax_screening import build_static_quadrature
from .efermi import solve_oneshot_occupations
from . import quadrature_log
from .screening import (
	compute_screening_model, driver_persists_w0, restart_static_w, screening_requests_for)
from .sigma_dispatch import (
	SIGMA_KSET_FULL_BZ, SIGMA_KSET_STAR_WEDGE, compute_sigma_xc,
	sigma_result_on_kset, validate_band_extrapolation)
from .qsgw_utils import solve_qp
from .dynamic_sigma import extract_sigma_diag_logical
from .degen_average import (
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
	head_detail,
	log_stage_row,
	v_q_detail,
	w_roles_detail,
)
from runtime.run_session import RunStreams
from .gw_output import (
	GWResults,
	print_system_summary,
	write_freq_debug,
	write_qp_wfn_oneshot,
	write_qsgw_qp_ladders,
	write_results,
)

def _stage_count(qp_solver) -> int:
	"""How many manual stages the main log numbers: ISDF, Screening,
	Self-energy and Quasiparticle bands for a one-shot run; ISDF, the
	self-consistent driver and Quasiparticle bands for QSGW."""
	return 3 if qp_solver is QPSolver.SELF_CONSISTENT else 4


def _manual_stage(index, total, title, *, name):
	"""One ``[index/total] title`` stage of the main log (``timing.manual_stage``),
	which is also the timing section ``name`` the TIMING table reads."""
	return timing.manual_stage(index, total, title, name=name)


class _GWStages:
	"""The main log's stages after ISDF, each opened by the helper that starts it
	and closed by the next one (or by leaving the ``with``, an exception included).

	A stage is opened only between top-level sections, so the timing stack
	stays nested; ``result`` sets the open stage's ``=>`` text.
	"""

	def __init__(self, qp_solver):
		self.total = _stage_count(qp_solver)
		self.one_shot = qp_solver is not QPSolver.SELF_CONSISTENT
		self._stack = None
		self.current = None

	def open(self, index, title, *, name):
		self.close()
		self._stack = contextlib.ExitStack()
		self.current = self._stack.enter_context(
			_manual_stage(index, self.total, title, name=name))
		return self.current

	def result(self, text):
		if self.current is not None:
			self.current.result(text)

	def close(self, *exc_info):
		stack, self._stack, self.current = self._stack, None, None
		if stack is not None:
			stack.__exit__(*(exc_info or (None, None, None)))

	def __enter__(self):
		return self

	def __exit__(self, *exc_info):
		self.close(*exc_info)
		return False


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
		head_resolver, meta, do_screened, *,
		require_screened=True, occupation_state=None):
	"""Resolve the q→0 head sample and its exact band-diagonal Σ terms.

	Used by every mode: the bare-X head piece applies to static and
	dynamic Σ alike (the SX/COH pieces additionally apply when screened).
	"""
	head = (head_resolver.at(0.0 + 0.0j) if require_screened
	        else head_resolver.direct_at(0.0 + 0.0j))
	log.debug(format_head_sample_diagnostics(head, include_screened=do_screened))
	occ_mask = (np.arange(meta.nb_sigma, dtype=np.int32) < meta.nelec
	            if occupation_state is None else
	            np.asarray(occupation_state.f_kn)[:, :meta.nb_sigma])
	terms = compute_static_head_terms_from_sample(
		head, occ=occ_mask, cell_volume=meta.cell_volume, nk_tot=meta.nk_tot)
	log.debug(format_static_head_diagnostics(terms))
	return terms




def _open_production_report(args):
    """Produce the resolved configuration and its scientific report."""
    _config_provenance = []
    def _config_print(*values, sep=" ", **kwargs):
        text = sep.join(str(value) for value in values)
        log.debug(text)
        if "[config provenance]" in text:
            _config_provenance.append(text)
    print0 = _config_print
    config = LorraxConfig.from_input_file(args.input, print_fn=print0)
    input_dir = config.input_dir
    qp_solver = config.qp_solver     # how QP energies are extracted from Σ
    mode = config.compute_mode       # the self-energy ansatz
    report = GWProductionReport(
        config.paths.report_file, runtime=RUNTIME,
        debug=debug_print_enabled(), stdout=rank0_print)
    # The report path comes from the deck, so the report opens here and its
    # streams beside it: the handler setup every RunSession uses.
    streams = RunStreams(RUNTIME, debug=debug_print_enabled(), report=report).open()
    print0 = report.legacy_print
    timing.set_memory_sink(report.progress)
    report.begin(input_file=args.input)
    refuse_unimplemented_compute_mode(mode, context="the LORRAX GW driver")
    validate_band_extrapolation(config, mode)
    do_screened = mode.needs_screening
    return (config, input_dir, qp_solver, mode, report, streams, print0, _config_provenance, do_screened)


def _report_head_and_photon_policy(config, report):
    """Report the resolved head and photon policies."""
    direct_photon = uses_direct_bispinor_tangential_head(config)
    log.info(
        f"  Head policy: head_correction={config.head.correction.value}; "
        f"screening_diagrams={config.screening.diagrams.value}; "
        f"direct diagnostic source={config.head.wcoul0_source}. "
        + ({
            HeadCorrection.FULL: (
                "bare Coulomb head" if config.compute_mode is ComputeMode.X_ONLY
                else "first-order direct four-current Gamma head; no "
                "local-field fold on the four-current bank" if direct_photon
                else "macroscopic W, local fields exactly once"),
            HeadCorrection.OFF: "no special Gamma-cell contribution",
        }[config.head.correction]))
    if config.bispinor:
        _bispinor_note = {
            "full_static_cohsex": (
                " (packed no-pair 4x4 static response with the Gamma-cell "
                "completion: bare <D> into V, charge S00/wing head into W, "
                "Hall CT/TC from static_gauge_hall_file when present)"
                if config.head.correction is HeadCorrection.FULL else
                " (packed no-pair 4x4 static response; DEBUG: Gamma-cell "
                "head disabled by head_correction=off)"),
        }.get(config.bispinor_gw.value, "")
        log.info(
            f"  Bispinor GW policy: bispinor_gw={config.bispinor_gw.value}"
            f"{_bispinor_note}")
        _bare_taken, _bare_reason = packed_bare_transverse_route(config)
        if uses_full_bispinor_tangential(config):
            report.progress(
                "Photon route   : FULL tangential CC/CT/TC/TT screening "
                "and sector Sigma; four-current Dyson at each bank frequency")
        elif uses_bare_transverse_tangential(config):
            report.progress(
                "Photon route   : four-spinor CC full-frequency tangential "
                "screening plus bare TT exchange; no CT/TT Dyson solve "
                "[theory: docs/theory/bispinor-gw.md#bare-transverse]")
        elif config.bispinor_gw.value == "full_static_cohsex":
            report.progress(
                "Photon route   : packed screened static photon operator "
                "(sixteen response and Sigma blocks; coupled 4x4 Dyson solve; "
                "Gamma-cell completion carries charge, mixed, and transverse heads)")
        else:
            report.progress(
                "Photon route   : "
                + ("bare CC + TT exchange (no chi0, W, or Dyson solve)"
                   if config.compute_mode is ComputeMode.X_ONLY else
                   "packed static photon operator (chi_TT = chi_CT = 0; scalar "
                   "Dyson on CC, W_packed = diag(W_00, D_TT); the Gamma-cell "
                   "completion carries both the charge head and the bare "
                   f"<D_TT>) -- {_bare_reason}"
                   if _bare_taken else
                   "incumbent charge-screened W + Sigma^B "
                   f"(gw.sigma_x_bispinor) -- {_bare_reason}")
                + " [theory: docs/theory/bispinor-gw.md#bare-transverse]")
        if not uses_static_photon_response(config):
            _banner, _head_record = incumbent_bispinor_head_record(config)
            if _banner:
                log.warning(
                    "Gamma-cell head disabled by head_correction=off on the "
                    "incumbent bispinor route",
                    extra=message(
                        "bispinor_head_off",
                        lines=("no scalar <v>/W_h band-diagonal head and no "
                               "transverse q=Gamma head",
                               "a k-grid convergence / debugging setting, NOT a "
                               "production calculation"),
                        fix="head_correction = full"))
            report.progress(f"Photon head    : {_head_record}")


def _load_system_inputs(config, input_dir, mesh_xy, report, print0, _config_provenance):
    """Produce wavefunctions, symmetry, centroid basis and output paths."""
    wfn = WfnLoader(config.paths.wfn_file, mesh=mesh_xy)
    material_class = infer_material_class(wfn.occs)
    config = resolve_mpa_sampling_alpha(          # its SETTINGS row states it
        config, material_class, print_fn=print0)
    validate_material_inputs(config, material_class)
    log.info(f"  Material class: {material_class} (inferred from WFN occupations)")
    sym = wfn.symmetry()
    centroid_basis = load_centroid_basis(
        config.paths.centroids_file, wfn.fft_grid, sym=sym)
    centroid_sets = [centroid_basis]
    if config.bispinor and config.paths.centroids_file_current:
        centroid_sets.append(load_centroid_basis(
            config.paths.centroids_file_current, wfn.fft_grid, sym=sym))
    nonclosed = [basis.path for basis in centroid_sets if not basis.orbit_closed]
    if nonclosed:
        log.warning("WARNING: non-orbit-closed centroid set(s): " + ", ".join(nonclosed)
               + "; using unreduced parents (n_parent = nk) and full q on the same "
               "parent route. Generate orbit-closed centroids with kmeans to restore reduction.")
        sym = sym.trivial_view()
        diagrams = coerce_screening_diagrams(config.screening.diagrams)
        if diagrams is not ScreeningDiagrams.W_RPA:
            raise ValueError(
                f"GATE resolvent_ladder_trivial_view: screening_diagrams = "
                f"{diagrams.value} with a non-orbit-closed centroid set.  The "
                f"resolvent ladder solves on the WFN's reduced q wedge, while the "
                f"centroid set forces the unreduced trivial view (every q); want an "
                f"orbit-closed centroid set; fix: regenerate the centroids with "
                f"kmeans (orbit-closed), or use screening_diagrams = w_rpa; doc: "
                f"docs/architecture/symmetry_register.md §7.")
    # Before any basis, bank or constructor: the w_bse ladder and HL-PPM
    # refuse an ordered store, and full_tangential a store with time
    # reversal, here, on the final symmetry verdict.
    from .screening import refuse_full_tangential_with_trs, refuse_w_bse_without_trs
    from .ppm_pipeline import refuse_hl_ppm_without_trs
    refuse_w_bse_without_trs(config, sym)
    refuse_full_tangential_with_trs(config, sym)
    refuse_hl_ppm_without_trs(config, sym)
    centroid_indices = centroid_basis.centroid_indices
    n_rmu = centroid_basis.n_rmu
    tmp_dir = os.path.join(input_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{n_rmu}.h5")
    print_system_summary(
        n_rmu=n_rmu, fft_grid=wfn.fft_grid,
        cell_volume=wfn.cell_volume,
    )
    report.devices(config=config)
    # The deck's resolved keys, with the deck-resolution notes the parse printed.
    report.settings_block(
        config=config, n_mu=n_rmu, n_q_irr=int(sym.nk_red),
        processes=RUNTIME.process_count, resolved=_config_provenance,
        tensors_file=tensors_filename)
    # The QP request (number_bands_protected) resolves against the WFN here.
    config = config.with_band_request(wfn, print0)
    return (config, wfn, material_class, sym, centroid_basis, centroid_indices, n_rmu, tmp_dir, tensors_filename)




def _report_sampling_and_bands(
        band_slices, centroid_basis, config, material_class, mesh_xy, meta, mode,
        report, sym, wfn, zeta_fit_edge):
    """Produce DFT band energies and report the sampled system."""
    _p_x = int(mesh_xy.devices.shape[0])
    _p_y = int(mesh_xy.devices.shape[1])
    _nbs = int(meta.nb_sigma)
    if mode.is_dynamic:
        from .ppm_sigma import sigma_band_axis
        _sigma_axis = sigma_band_axis(
            _nbs, mesh_xy,
            ansatz=f"compute_mode = {getattr(mode, 'value', mode)}")
        log.debug(
            f"  Sigma omega layout: sharded; Σ_c(ω,k,m,n) stays "
            f"(m_X, n_Y)-tiled on the {_p_x}x{_p_y} mesh at "
            f"logical/carrier={_sigma_axis.logical}/{_sigma_axis.carrier} "
            f"(consumers read tiles; no full-cube replication).")
    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
        nspinor=meta.nspinor)
    _zeta_ranges = zeta_fit_band_ranges(
        band_slices, zeta_fit_edge, log=lambda *args, **kwargs: None)
    report.input_files(config=config, wfn=wfn, sym=sym, centroids=centroid_basis,
                       zeta_ranges=_zeta_ranges, band_slices=band_slices,
                       material_class=material_class)
    report.sampling(wfn=wfn, sym=sym, centroids=centroid_basis,
                    consequence=report.trs_consequence(
                        config=config, sym=sym, material_class=material_class))
    report.bands(
        config=config, wfn=wfn, band_slices=band_slices,
        zeta_ranges=_zeta_ranges)
    return (enk_dft)


def _prepare_isdf_carriers(
        band_slices, bgw_v_grid_fn, centroid_indices, config, material_class, mesh_xy, meta,
        mode, print0, qp_solver, sym, tensors_filename, tmp_dir, wfn, enk_dft, *,
        centroid_basis=None):
    """Produce the fitted ISDF operators and wavefunction views."""
    from .tangential_screening import compile_w_model_ahead
    compile_w_model_ahead(config, meta, sym, mesh_xy)
    with _manual_stage(1, _stage_count(qp_solver), "ISDF basis and Coulomb metric",
                       name="gw_jax.isdf") as _stage:
        _n_sites = int(np.asarray(centroid_indices).shape[0])
        _n_ops = int(np.asarray(sym.Rinv_grid).shape[0])
        _closed = bool(getattr(centroid_basis, "orbit_closed", True))
        log_stage_row("centroids", f"{_n_sites}, " + (
            f"orbit-closed under the {_n_ops} operations" if _closed else
            "not orbit-closed: the run uses the unreduced parents"), stage="isdf")
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
        # V is a QirrOperator on the q wedge (its array in .values) or a plain array.
        _v_shape = np.shape(getattr(isdf.V_qmunu, "values", isdf.V_qmunu))
        _n_q, _n_mu = (int(v) for v in _v_shape[:2])
        log_stage_row("V(q)", v_q_detail(
            n_q=_n_q, n_mu=_n_mu,
            restart_file=tensors_filename if config.restart else None), stage="isdf")
        if not config.restart and bool(getattr(config, "write_restart_tensors", False)):
            # The one file this stage's results go to (written after W).
            log_stage_row("file written", os.path.join(
                "tmp", os.path.basename(tensors_filename)) + " (after W, for bse)",
                stage="isdf")
        _stage.result(f"zeta and V(q) {'read for' if config.restart else 'on'} "
                      f"{_n_mu} centroids, {_n_q} q")
    V_qmunu = isdf.V_qmunu
    wfns = isdf.wf_bundle
    green_parent_carrier = getattr(isdf, 'green_parent_carrier', None)
    wfns_sigma = wfns
    wfns_screening = wfns
    oneshot_occupation_state = (
        solve_oneshot_occupations(
            config, wfn, wfns, material_class, mesh_xy=mesh_xy, print_fn=print0)
        if (qp_solver is not QPSolver.SELF_CONSISTENT
            and (mode.is_pole_route or mode is ComputeMode.X_ONLY)) else None)
    if oneshot_occupation_state is not None:
        log.info(
            f"  one-shot occupations: fixed-N {oneshot_occupation_state.smearing_family} state, "
            f"mu={oneshot_occupation_state.mu_ry * RYD_TO_EV:.8f} eV, "
            f"width={oneshot_occupation_state.smearing_width_ry:.10f} Ry, "
            f"occ_hash={oneshot_occupation_state.occ_hash}")
    if config.sigma.w_model == "tangential" and qp_solver is not QPSolver.SELF_CONSISTENT:
        from .tangential_recipe import (SUPPORT_READ_PAD_EV, bind_tangential_census,
                                         resolve_tangential_recipe)
        from centroid.sampling_metric import full_k_quadrature_weights
        census_inputs = dict(trs_allowed=sym.trs_allowed,
                             state_capacity=wfn.occupation_state_capacity,
                             kweights=full_k_quadrature_weights(wfn, wfn.symmetry()))
        bind_tangential_census(
            wfns, meta, occupation_state=oneshot_occupation_state, **census_inputs)
        from .qp_support import requested_reads_ev
        energy, states, frame_mu_ev = _oneshot_requested(
            config, enk_dft, wfn, oneshot_occupation_state, material_class)
        support_reads_ev = frame_mu_ev + requested_reads_ev(
            energy, states, config.sigma.omega_step_ev, pad_ev=SUPPORT_READ_PAD_EV)
        meta.tangential_recipe = resolve_tangential_recipe(
            config, wfns, meta, mesh_xy=mesh_xy, print_fn=print0,
            support_reads_ev=support_reads_ev)
        from .ppm_pipeline import observing_window_source, publish_window_source
        if observing_window_source():
            # TD-aGW's tangential refit re-resolves this recipe at its own census
            # (tdagw.dynsig.spm_screening); one empty list test otherwise.
            publish_window_source(recipe_inputs=dict(
                census_inputs, support_reads_ev=np.array(support_reads_ev, np.float64)))
    wfns_transverse = getattr(isdf, 'wf_bundle_transverse', None)
    if config.bispinor and wfns_transverse is None:
        raise RuntimeError(
            "bispinor = true but no transverse-centroid Wfns bundle was "
            "produced (Σ^B would be silently dropped).  Check "
            "centroids_file_current and, on restart, that the restart "
            "file carries psi_parent_y_transverse.")
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
    return (V_qmunu, bispinor_v_q_path, green_parent_carrier, isdf, oneshot_occupation_state, wfns, wfns_screening, wfns_sigma, wfns_transverse)


def _prepare_oneshot_response(
        config, do_screened, input_dir, material_class, mesh_xy, meta, mode, print0,
        qp_solver, wfn, wfns, wfns_sigma, *, occupation_state=None, stages):
    """Produce quadrature and direct head-response inputs for one-shot screening.

    On a metal ``occupation_state`` is the one fixed-N state the body and
    Sigma consume; the head takes it too, with its intraband term."""
    quad, e_ref = None, None
    if stages.one_shot:
        # Stage 2 is the ``gw_jax.screening`` section: the minimax rule, the
        # head response, chi0 -> W, the head fold and the W persist.
        stages.open(2, "Screening", name="gw_jax.screening")
    if do_screened:
        with timing.section("gw_jax.minimax_quadrature", announce=True,
                            label="minimax tau-axis"):
            with warnings.catch_warnings():
                if jax.process_index() != 0:
                    warnings.simplefilter("ignore", RuntimeWarning)
                quad, e_ref = build_static_quadrature(
                    wfns, config.minimax_config,
                    occupation_width_ry=(
                        float(config.occ_broadening_ry)
                        if getattr(config, "occ_smearing_family", None)
                        else None),
                    print_fn=print0)
            quadrature_log.record_minimax(
                "static", quad, target=config.minimax_config.target_error)
    oneshot_head_response = None
    oneshot_head_requests = None
    oneshot_mpa_plan = None
    # The four-current sector bank builds its own direct CC/CT/TC/TT head
    # (``gw.photon_direct_head``) and consumes no scalar head response, as on
    # the SC route (``sc_iteration``).
    if (do_screened
            and folds_scalar_head(config)
            and config.screening.diagrams is ScreeningDiagrams.W_RPA
            and not packed_photon_replaces_charge_sigma(config)
            and qp_solver is not QPSolver.SELF_CONSISTENT):
        from .qsgw_head import build_dft_head_response, head_quadrature_omegas
        if mode.is_pole_route:
            from .mpa import sample_plan
            if config.sigma.w_model == "tangential":
                from .tangential_head import tangential_head_plan
                oneshot_mpa_plan = tangential_head_plan(
                    config, meta.tangential_recipe, material_class=material_class)
            else:
                from .mpa.model import make_mpa_plan
                oneshot_mpa_plan = make_mpa_plan(
                    config, quad, material_class=material_class)
            # The fit nodes, then the head's imaginary-axis moment rule.
            oneshot_omegas = np.asarray(
                tuple(sample_plan.plan_z(oneshot_mpa_plan))
                + head_quadrature_omegas(config), dtype=np.complex128)
        else:
            oneshot_head_requests = screening_requests_for(mode, config)
            oneshot_omegas = np.asarray(
                [complex(r.omega_ry) for r in oneshot_head_requests],
                dtype=np.complex128)
        reused_mpa_fit_owns_head = (
            mode.is_pole_route
            and config.mpa.fit_reuse_file is not None)
        if oneshot_omegas.size and not reused_mpa_fit_owns_head:
            oneshot_head_response = build_dft_head_response(
                wfns_sigma, oneshot_omegas,
                input_dir=input_dir, mesh=mesh_xy,
                wfn=wfn, meta=meta, config=config,
                occupation_state=(occupation_state
                                  if material_class == "metal" else None))
            if material_class == "metal":
                from .qsgw_head import metal_head_summary
                log.info("  " + metal_head_summary(
                    oneshot_head_response, occupation_state))
            log.debug(
                "  head_correction=full: built direct DFT response and "
                "head/body wings on the chi0 transition manifold; finalizing "
                "once against the resident W(Gamma).")
        elif oneshot_omegas.size:
            log.info(
                "  MPA fit reuse: deferring to the stored finalized full head; "
                "skipped the redundant direct DFT head/body-wing allocation. "
                "The live fit and sampling plan will still be authenticated.")
    return (quad, e_ref, oneshot_head_response, oneshot_head_requests, oneshot_mpa_plan)


def _report_packed_screening(_dynamic_packed, _screens_current, mode, photon_response, report):
    """Report the resolved packed screening and Gamma completion."""
    log.info(
        "  static photon route: "
        + ("full_static_cohsex (sixteen chi blocks, one packed "
           "Dyson solve)" if _screens_current else
           "bare_transverse as the packed path (chi_TT = chi_CT = 0; "
           "scalar Dyson on CC, W_packed = diag(W_00, D_TT))")
        + ("" if not _dynamic_packed else
           "; DYNAMIC packed route: the CC block carries "
           f"compute_mode = {mode.value} at every omega and the "
           "fifteen current blocks are frozen at omega = 0"))
    log.debug(
        "  static photon response: "
        f"approximation={photon_response.approximation}, "
        f"current_model={photon_response.current_model}, "
        f"current_contact={photon_response.current_contact}, "
        f"packed_extent={photon_response.layout.packed_extent}")
    if _dynamic_packed:
        report.progress(
            "Photon Sigma   : dynamic packed route -- "
            f"W_packed(omega) = diag(W_00(omega), W_TT, W_CT) with "
            f"compute_mode = {mode.value} on the CHARGE block and "
            "the fifteen current blocks STATIC (omega = 0). The "
            "charge q->0 head is the dynamic model's; the TT/CT "
            "q->0 head is the packed Gamma-cell completion's. "
            "Sigma^B is the TT block of the packed operator, not "
            "a separate term.")
    else:
        report.progress(
            "Photon Sigma   : static packed route -- all sixteen "
            "Lorentz blocks contracted once; Sigma_xc = Sigma_SX + "
            "Sigma_COH and the per-state CC / CT+TC / TT split is "
            "written to sigma_diag.dat.")
    _hc = photon_response.head_completion
    report.progress(
        "Photon head    : "
        + ("DEBUG: Gamma-cell head disabled by head_correction=off "
           "(headless packed body; NOT a production calculation)"
           if _hc is None else
           "Gamma-cell completion applied (bare <D> into V, "
           "charge S00/wing head into W); "
           f"hall_source={_hc.hall_source}; "
           f"sigma_H={np.asarray(_hc.sigma_H).tolist()} bohr^-1; "
           f"ward={_hc.ward_residual:.3e}; "
               f"hermiticity={_hc.hermiticity_residual:.3e}; "
               f"dyson_forward_bound={_hc.max_dyson_forward_error_bound:.3e}; "
               f"cubature_orders={_hc.cubature_receipt.orders}"))
    if _hc is not None:
        report.progress(
            "Photon WS cert : "
            f"orders={_hc.cubature_receipt.orders}; "
            f"nodes={_hc.observed_physical_counts}; "
            f"final_error_ratio="
            f"{_hc.mixed_convergence_error_ratios[-1]:.3e}; "
            f"max_dyson_backward_residual="
            f"{_hc.max_backward_residual:.3e}")


def _run_oneshot_screening(
        V_q, bispinor_v_q_path, centroid_indices, config, e_ref, green_parent_carrier,
        head_resolver, isdf, material_class, mesh_xy, meta, mode, oneshot_head_response,
        oneshot_mpa_plan, oneshot_occupation_state, print0, qp_solver, quad, report, sym,
        tensors_filename, tmp_dir, wfn, wfns_screening, wfns_transverse, *, stages):
    """Produce the one-shot screening roles and packed response."""
    photon_response = None
    if qp_solver is QPSolver.SELF_CONSISTENT:
        W_by_role = {}
    else:
        if uses_static_photon_response(config):
            if wfns_transverse is None or bispinor_v_q_path is None:
                raise RuntimeError(
                    "static packed-photon screening requires the transverse "
                    "wavefunction "
                    "bundle and v_q_bispinor.h5; refusing a charge-only W.")
            _screens_current = packed_photon_screens_current(config)
            _dynamic_packed = uses_dynamic_packed_photon_route(config)
            _W_charge = None
            if not _screens_current or _dynamic_packed:
                W_by_role = compute_screening_model(
                    mode, wfns_screening, V_q, quad=quad, e_ref=e_ref, sym=sym,
                    centroid_indices=centroid_indices, config=config,
                    meta=meta, mesh_xy=mesh_xy,
                    run_dir=os.path.join(tmp_dir, "mpa"), wfn=wfn,
                    wfn_fingerprint_binding=isdf.wfn_fingerprint_binding,
                    charge_zeta_identity=isdf.charge_zeta_identity,
                wfns_transverse=wfns_transverse, mu_bases=isdf.mu_bases,
                bispinor_v_q_path=bispinor_v_q_path,
                photon_g0_vectors=isdf.photon_g0_vectors,
                    label="oneshot", head_resolver=head_resolver,
                    head_channel=getattr(isdf, 'head_channel', None),
                    mpa_plan=oneshot_mpa_plan,
                    iteration_head_response=oneshot_head_response,
                    occupation_state=oneshot_occupation_state,
                    material_class=material_class,
                    tensors_filename=tensors_filename,
                    static_only=not _dynamic_packed,
                    print_fn=print0)
                if not _screens_current:
                    _W_charge = W_by_role.get("static")
                    if _W_charge is None:
                        raise RuntimeError(
                            "the packed bare-transverse route needs the incumbent "
                            "static W(omega=0) on the charge block; the screening "
                            "model returned no 'static' role.")
            from .w_isdf import compute_static_photon_response
            photon_response = compute_static_photon_response(
                wfns_screening, wfns_transverse, quad, bispinor_v_q_path,
                meta, mesh_xy,
                screen_current=_screens_current,
                mu_bases=isdf.mu_bases,
                W_charge=_W_charge,
                wfn=wfn, config=config,
                photon_g0_vectors=isdf.photon_g0_vectors,
                wf_binding_charge=isdf.wf_binding_charge,
                wf_binding_transverse=isdf.wf_binding_transverse,
                wfn_fingerprint_binding=isdf.wfn_fingerprint_binding,
                energy_reference=e_ref,
                dyson_solver=config.backend.w_dyson_solver,
                distrib_la_batched_route=(
                    config.backend.distrib_la_batched_route),
                print_fn=print0)
            _W_charge = None
            if not _dynamic_packed:
                W_by_role = {}
            _report_packed_screening(_dynamic_packed, _screens_current, mode, photon_response, report)
        else:
            W_by_role = compute_screening_model(
                mode, wfns_screening, V_q, quad=quad, e_ref=e_ref, sym=sym,
                centroid_indices=centroid_indices, config=config, meta=meta,
                mesh_xy=mesh_xy, run_dir=os.path.join(tmp_dir, "mpa"), wfn=wfn,
                wfn_fingerprint_binding=isdf.wfn_fingerprint_binding,
                charge_zeta_identity=isdf.charge_zeta_identity,
                wfns_transverse=wfns_transverse, mu_bases=isdf.mu_bases,
                bispinor_v_q_path=bispinor_v_q_path,
                photon_g0_vectors=isdf.photon_g0_vectors,
                label="oneshot", head_resolver=head_resolver,
                head_channel=getattr(isdf, 'head_channel', None),
                mpa_plan=oneshot_mpa_plan,
                iteration_head_response=oneshot_head_response,
                occupation_state=oneshot_occupation_state,
                material_class=material_class,
                tensors_filename=tensors_filename,
                print_fn=print0)
        # The rules this stage used, and what it built, under its banner.
        from .production_report import log_chi_rules
        log_chi_rules()
        _log_screening_rows(W_by_role, V_q, config, mode, sym)
        stages.result(_screening_result(W_by_role, V_q, config, mode))
    return (W_by_role, photon_response, green_parent_carrier, wfns_screening)


def _n_q_of(V):
	"""Leading (q) extent of V: a QirrOperator on the wedge, or a plain array."""
	return int(np.shape(getattr(V, "values", V))[0])


def _screening_roles(W_by_role, config, mode):
	"""The screened W roles this run built, ``(role, omega_ry or None)``."""
	requested = [(r.role, r.omega_ry) for r in screening_requests_for(mode, config)
	             if r.role in W_by_role]
	return requested or [(str(role), None) for role, value in W_by_role.items()
	                     if not isinstance(value, (bool, type(None)))
	                     and not str(role).startswith(("iteration_", "mpa_fit"))]


def _log_screening_rows(W_by_role, V_q, config, mode, sym):
	"""The Screening stage's ``W roles`` and ``q->0 head`` rows."""
	roles = _screening_roles(W_by_role, config, mode)
	if not roles:
		return
	log_stage_row("W roles", w_roles_detail(
		roles, n_q=_n_q_of(V_q), trs_allowed=bool(sym.trs_allowed)),
		stage="screening")
	log_stage_row("q->0 head", head_detail(config), stage="screening")


def _screening_result(W_by_role, V_q, config, mode):
	"""The Screening stage's ``=>`` line."""
	roles = _screening_roles(W_by_role, config, mode)
	if not roles:
		return ("no screened W (bare exchange)" if not mode.needs_screening
		        else "screened response built")
	names = ["W(0)" if str(role) == "static" else
	         ("W(i omega_p)" if omega is None or complex(omega).real == 0.0 else "W(omega_p)")
	         if str(role) == "probe" else str(role) for role, omega in roles]
	return f"W on {_n_q_of(V_q)} q: " + " and ".join(names)


def _install_oneshot_head(
        W_by_role, config, head_resolver, mesh_xy, meta, mode, oneshot_head_requests,
        oneshot_head_response, wfn):
    """Install the finalized one-shot head samples in their resolver."""
    if (oneshot_head_response is not None
            and not packed_photon_replaces_charge_sigma(config)):
        if mode.is_pole_route:
            if W_by_role.get("mpa_fit_reused", False):
                final_head = None
                log.info(
                    "  MPA screening reuse: consuming the certified stored "
                    "body/head; no second head fold performed.")
            else:
                final_head = W_by_role.get("iteration_head")
                if final_head is None:
                    raise RuntimeError(
                        "head_correction=full: MPA screening returned no "
                        "finalized head samples")
        else:
            from .qsgw_head import finalize_iteration_head_samples
            final_head = finalize_iteration_head_samples(
                oneshot_head_response, wfn=wfn, meta=meta, config=config,
                mesh=mesh_xy, requests=oneshot_head_requests,
                W_by_role=W_by_role)
        if final_head is not None:
            head_resolver.install_samples(final_head.samples)


def _persist_screening(
        V_q, W_by_role, centroid_indices, config, head_resolver, mesh_xy, meta, mode,
        print0, qp_solver, sym, tensors_filename, photon_response=None,
        material_class=None, head_response=None):
    """Persist the screened static body and head on the canonical q set."""
    try:
        from .gw_output import stamp_w0_route
        stamp_w0_route(tensors_filename, config)        # read by the BSE's GATE bse_mpa_w0
        if (driver_persists_w0(mode, config)
                and qp_solver is not QPSolver.SELF_CONSISTENT):
            W_static = restart_static_w(
                mode, W_by_role, V_q, config=config, meta=meta, mesh_xy=mesh_xy,
                material_class=material_class, head_source=head_resolver, print_fn=print0)
            if W_static is None:
                return
            with timing.section("gw_jax.persist_w0"):
                from .gw_output import persist_w0_and_head
                from .qsgw_head import static_head_wings_host
                persist_w0_and_head(
                    W_static, photon_response=photon_response,
                    head_wings=static_head_wings_host(head_response, mesh_xy),
                    tensors_filename=tensors_filename, head_resolver=head_resolver,
                    config=config, meta=meta, mesh_xy=mesh_xy,
                    sym=sym, centroid_indices=centroid_indices,
                    static_head_only=mode.is_pole_route,
                    print_fn=print0)
    finally:
        if qp_solver is not QPSolver.SELF_CONSISTENT:
            from .tangential_screening import release_coulomb_cache
            release_coulomb_cache()


def _prepare_static_head(config, do_screened, head_resolver, meta, mode,
                         qp_solver, occupation_state):
    """Produce the static head terms required by the selected QP solver."""
    static_head_terms = None
    if (config.do_G0
            and not packed_photon_replaces_charge_sigma(config)):
        _sc_full = (qp_solver is QPSolver.SELF_CONSISTENT
                    and do_screened
                    and folds_scalar_head(config))
        if not _sc_full:
            with timing.section("gw_jax.static_head"):
                static_head_terms = _compute_static_head(
                    head_resolver, meta, do_screened,
                    require_screened=(do_screened and not mode.is_pole_route
                                      and qp_solver is not QPSolver.SELF_CONSISTENT),
                    occupation_state=occupation_state)
    return (static_head_terms)


def _oneshot_requested(config, enk_dft, wfn, occupation_state, material_class):
    """The one-shot's requested states: ``(E - mu [k, band] eV, mask, mu eV)``.

    The Sigma window's identities the W model treats as active, judged in the
    frame the Sigma build measures from (``efermi.sigma_frame_mu_ev``).  One
    source for the sampled Sigma support and W's line sites
    (``qp_support.requested_reads_ev``).
    """
    from .efermi import sigma_frame_mu_ev
    from .qp_support import requested_states
    from .tangential_recipe import active_band_mask
    e_ry = np.asarray(enk_dft, dtype=np.float64)
    metal = material_class == "metal" and occupation_state is not None
    mu_ev = sigma_frame_mu_ev(
        config, wfn, e_ry,
        float(occupation_state.mu_ry) if metal else float(wfn.efermi),
        occupation_state if metal else None)
    energy = e_ry * RYD_TO_EV - mu_ev
    states = requested_states(
        config.sigma, config.sc.frozen_core_bands, energy,
        np.ones(e_ry.shape, dtype=bool), active_band_mask(e_ry, float(wfn.efermi)))
    return energy, states, mu_ev


def _oneshot_sampled_support(config, enk_dft, wfn, meta, occupation_state,
                             material_class):
    """The one-shot's sampled Sigma(omega) support: the first plan of
    ``gw.qp_support`` (the deck request joined with every requested state's
    E_DFT +/- 2 eV), the near grid SC map 0 plans, plus the coarse windows of
    the coarse states the requested set leaves out.

    The requested states are the Sigma window's identities the W model treats
    as active, judged in the frame the Sigma build measures from
    (``efermi.sigma_frame_mu_ev``).  Under ``cover`` an active state outside
    the requested grid reads its own Sigma(E), and a coarse state outside the
    requested set reads its own Sigma(E) on SC map 0's coarse windows
    (``qp_support.oneshot_support_ev``).
    """
    from dataclasses import replace

    from .qp_support import coarse_identities, oneshot_support_ev
    requested = np.asarray(config.omega_grid_ev, dtype=np.float64)
    energy, states, _ = _oneshot_requested(
        config, enk_dft, wfn, occupation_state, material_class)
    coarse = coarse_identities(
        getattr(meta, "coarse_class", None), enk_dft, config.compute_mode)
    grid, windows = oneshot_support_ev(config.sigma, requested, energy, states, coarse)
    if grid.size == requested.size and not windows:
        return config
    log.debug(f"  Sigma sampled support ({config.sigma.out_of_grid}, plan 0): "
             f"[{requested[0]:+.6f}, {requested[-1]:+.6f}] -> "
             f"[{grid[0]:+.6f}, {grid[-1]:+.6f}] eV, {len(windows)} coarse window(s)")
    return replace(config, sc_omega_grid_ev=tuple(float(x) for x in grid),
                   sc_coarse_windows_ev=tuple(windows) or None)


def _run_oneshot_sigma(
        V_q, W_by_role, band_slices, bispinor_v_q_path, config, enk_dft, head_resolver,
        input_dir, isdf, material_class, mesh_xy, meta, mode, oneshot_occupation_state,
        photon_response, print0, qp_solver, quad, static_head_terms, sym, wfn, wfns_sigma,
        wfns_transverse, *, report=None, stages):
    """Produce one-shot Sigma and release its consumed screening bodies."""
    gc.collect()   # drop ISDF-stage temporaries before the Σ build
    sigma_result = None
    if qp_solver is not QPSolver.SELF_CONSISTENT:
        if mode.is_dynamic:
            config = _oneshot_sampled_support(
                config, enk_dft, wfn, meta, oneshot_occupation_state,
                material_class)
        with contextlib.nullcontext(
                stages.open(3, "Self-energy", name="gw_jax.sigma")) as _stage:
            sigma_result = compute_sigma_xc(
                mode,
                consume_screening=True,
                wfns=wfns_sigma, V_q=V_q, W_by_role=W_by_role,
                e_qp_ev=np.asarray(enk_dft, dtype=np.float64) * RYD_TO_EV,
                static_head_terms=static_head_terms,
                head_resolver=head_resolver,
                quad=quad,
                config=config, meta=meta, mesh_xy=mesh_xy,
                sym=sym, wfn=wfn, band_slices=band_slices,
                input_dir=input_dir,
                wfns_transverse=wfns_transverse,
                bispinor_v_q_path=bispinor_v_q_path, mu_bases=isdf.mu_bases,
                photon_response=photon_response,
                occupation_state=oneshot_occupation_state,
                material_class=material_class,
                print_fn=print0,
            )
            if report is not None:
                # The grid, its coverage and the band tail, under this stage.
                report.sigma_coverage(
                    config=config, band_slices=band_slices, enk_dft_ry=enk_dft,
                    sigma_result=sigma_result)
            _stage.result(_sigma_result_text(mode, band_slices, sym))
        sigma_result = sigma_result_on_kset(
            sigma_result, kset=SIGMA_KSET_FULL_BZ, nk=int(meta.nk_tot))
        W_by_role = {}
        photon_response = None
        gc.collect()
        sig_x_diag = np.real(np.diagonal(
            np.asarray(sigma_result.sigma_x_kij_ry),
            axis1=1, axis2=2)) * RYD_TO_EV
        if not config.no_degen_averaging:
            sig_x_diag = average_within_degenerate_sets(
                sig_x_diag,
                energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
                tol_ry=float(config.degen_avg_tol_ry),
            )
        log.info(f"  Bare Σ_X diagonal (eV), k=0: "
               + "  ".join(f"{sig_x_diag[0, i]:.4f}" for i in range(min(8, sig_x_diag.shape[1]))))
        from common import sanity
        sanity.check_finite("Σ_x", sigma_result.sigma_x_kij_ry, print_fn=print0)
        sanity.check_finite("V_H", sigma_result.v_h_kij_ry, print_fn=print0)
        sanity.check_sign("Σ_x diagonal (eV)", sig_x_diag,
                          expect="negative", print_fn=print0)
        sanity.check_in_range("Σ_x diagonal (eV)", sig_x_diag,
                              -200.0, 0.0, unit="eV", print_fn=print0)
    return (sigma_result, W_by_role, photon_response)


def _sigma_result_text(mode, band_slices, sym):
	"""The Self-energy stage's ``=>`` line: what was built, on which states."""
	parts = {ComputeMode.X_ONLY: "Sigma_x", ComputeMode.COHSEX: "Sigma_x, Sigma_SX + Sigma_COH"}
	n_bands = int(band_slices.b3) - int(band_slices.b0)
	return (f"{parts.get(mode, 'Sigma_x, Sigma_c')} on {n_bands} bands x "
	        f"{int(sym.nk_red)} k")


def _load_kinetic_ionic_hamiltonian(band_slices, config, mesh_xy, meta, print0, wfn, *,
                                   sigma_result=None, static_head=None):
    """Produce the authenticated kinetic and ionic band matrix."""
    from file_io import validate_kin_ion_against_run
    from file_io.qe_save_reader import dft_sys_dim
    _t_kin = time.perf_counter()
    validate_kin_ion_against_run(
        config.paths.kin_ion_file,
        expected_bispinor=config.bispinor,
        sys_dim=dft_sys_dim(wfn, fallback=config.sys_dim),
        nk=meta.nk_tot,
        band_stop=band_slices.b3,
        nspinor=int(wfn.nspinor),
        print_fn=print0,
    )
    kin_ion = load_kin_ion_submatrix(
        config.paths.kin_ion_file, band_slices.b0, band_slices.b3,
        mesh=mesh_xy,
    )
    timing.record("gw_jax.kin_ion_load", time.perf_counter() - _t_kin)
    from .ppm_pipeline import publish_window_hf_reference
    publish_window_hf_reference(kin_ion, sigma_result, static_head, meta, band_slices)
    return (kin_ion)


def _solve_qp_stage(
        V_q, band_slices, bispinor_v_q_path, centroid_indices, config, e_ref, enk_dft,
        head_resolver, input_dir, isdf, kin_ion, material_class, mesh_xy, meta, print0,
        qp_solver, quad, report, sigma_result, static_head_terms, sym, tensors_filename,
        wfn, wfns_sigma, wfns_transverse, *, stages):
    """Produce the QP Hamiltonian and optional self-consistency results.

    ``kin_ion`` ``None`` loads it here (the mean-field load opens one-shot
    stage 4, Quasiparticle bands)."""
    if stages.one_shot:
        stages.open(4, "Quasiparticle bands", name="gw_jax.qp_bands")
    if kin_ion is None:
        kin_ion = _load_kinetic_ionic_hamiltonian(
            band_slices, config, mesh_xy, meta, print0, wfn,
            sigma_result=sigma_result, static_head=static_head_terms)
    sc_result = None
    rotations_written = False
    final_static_head_terms = static_head_terms
    if qp_solver is QPSolver.SELF_CONSISTENT:
        from .sc_iteration import run_sc_driver
        with contextlib.nullcontext(stages.open(
                2, "Self-consistent QSGW", name="gw_jax.sc_driver")) as _stage:
            # The chi0 time rules built so far (the minimax tau axis), under
            # the SC banner; the end-of-run block does not repeat them.
            from .production_report import log_chi_rules
            log_chi_rules()
            sc_result = run_sc_driver(
                wfns_sigma, V_q, kin_ion,
                wfns_transverse=wfns_transverse,
                bispinor_v_q_path=bispinor_v_q_path, mu_bases=isdf.mu_bases,
                photon_g0_vectors=isdf.photon_g0_vectors,
                head_channel=getattr(isdf, 'head_channel', None),
                wfn_fingerprint_binding=isdf.wfn_fingerprint_binding,
                charge_zeta_identity=isdf.charge_zeta_identity,
                quad=quad, e_ref=e_ref,
                static_head_terms=static_head_terms,
                head_resolver=head_resolver,
                screening_model_fn=compute_screening_model,
                config=config, meta=meta, mesh_xy=mesh_xy,
                sym=sym, wfn=wfn, centroid_indices=centroid_indices,
                band_slices=band_slices, input_dir=input_dir,
                tensors_filename=tensors_filename,
                enk_dft=enk_dft, material_class=material_class,
                print_fn=print0, record_fn=report.progress)
            _maps = list(getattr(sc_result, "rms_history_ev", ()) or ())
            from .sc_report import last_run, sc_stage_verdict
            _stage.result(sc_stage_verdict(last_run(), n_fallback=len(_maps)))
        stages.open(3, "Quasiparticle bands", name="gw_jax.qp_bands")
        sigma_result = sc_result.sigma_result_dft
        sigma_total = sc_result.sigma_total_dft
        rotations_written = sc_result.rotations_written
        final_static_head_terms = sc_result.static_head_terms_dft
        if sigma_result.kset == SIGMA_KSET_STAR_WEDGE:
            from ffi import _services
            _services.ensure_on_path()
            from symmetry_maps import KStarMap
            _output_kstar = KStarMap.from_sym(sym, int(wfn.ntran))
            kin_ion = _output_kstar.select(kin_ion)
            enk_dft = _output_kstar.select(enk_dft)
    else:
        with timing.section("gw_jax.solve_qp"):
            sigma_total = solve_qp(
                qp_solver, sigma_result, kin_ion,
                config=config, mesh_xy=mesh_xy, print_fn=print0)
    return (enk_dft, final_static_head_terms, kin_ion, rotations_written, sc_result, sigma_result, sigma_total)


def _sigma_output_fields(
        config, enk_dft, final_static_head_terms, mesh_xy, qp_solver, sc_result,
        sigma_result, sigma_total):
    """Produce the existing Sigma fields and degenerate-state output averages."""
    sig_h   = sigma_result.v_h_kij_ry
    sig_h_scalar = sigma_result.v_h_scalar_kij_ry
    h_transverse = sigma_result.h_transverse_kij_ry
    sig_x   = sigma_result.sigma_x_kij_ry
    sig_sx  = (sigma_result.sigma_sx_kij_ry
               if sigma_result.sigma_sx_kij_ry is not None
               else jnp.zeros_like(sig_x))
    sig_coh = (sigma_result.sigma_coh_kij_ry
               if sigma_result.sigma_coh_kij_ry is not None
               else jnp.zeros_like(sig_x))
    photon_head_sigma_diag_tskn_ry = (
        sigma_result.photon_head_sigma_diag_tskn_ry)
    photon_head_sigma_basis = sigma_result.photon_head_sigma_basis
    sigma_lorentz_skij_ry = sigma_result.sigma_lorentz_skij_ry
    sigma_c_odd_at_dft_ev = sigma_result.sigma_c_odd_at_dft_diag_ev
    if photon_head_sigma_diag_tskn_ry is None:
        photon_head_sigma_diag_tskn_ry = np.zeros(
            (3, 3) + tuple(np.asarray(enk_dft).shape), dtype=np.complex128)
    sigma_omega_h5_path = sigma_result.sigma_omega_h5_path
    sigma_c_at_dft_ev   = sigma_result.sigma_c_at_dft_diag_ev
    omega_dft_rel_ev    = sigma_result.omega_dft_rel_ev
    e_eval_ev           = sigma_result.e_eval_ev
    efermi_dft_ev       = sigma_result.efermi_dft_ev
    # The omega-grid debug path (sigma_omega_grid_debug): the cube, its head
    # diagonal (the DFT-basis one under SC) and the grid, for the debug tables.
    grid_debug = None
    if sigma_result.sigma_c_omega_kij_ry is not None:
        grid_debug = dict(
            sigma_c_omega=sigma_result.sigma_c_omega_kij_ry,
            head_sigma_diag_w_kn_ry=(
                sc_result.head_sigma_diag_dft_w_kn_ry
                if qp_solver is QPSolver.SELF_CONSISTENT
                else sigma_result.head_sigma_diag_w_kn_ry),
            omega_grid_ev=(
                np.asarray(sigma_result.omega_grid_ev, dtype=np.float64)
                if sigma_result.omega_grid_ev is not None else None),
            omega_grid_ry=(
                np.asarray(sigma_result.omega_grid_ry, dtype=np.float64)
                if sigma_result.omega_grid_ry is not None else None))
    # The dynamic head at E_DFT (the on-shell ``dft0`` table), for the
    # debug tables; an SC run has no DFT-basis Sigma_c(E_DFT) on shell.
    head_at_dft_ry = None
    head = sigma_result.head_sigma_diag_w_kn_ry
    if (qp_solver is not QPSolver.SELF_CONSISTENT and head is not None
            and sigma_result.sigma_c_on_shell is not None):
        request = sigma_result.sigma_c_on_shell.request
        head_at_dft_ry = head.on_shell(request.tables_ry[request.slot("dft0")])[0]
    # Average extracted reporting arrays only. Keep every full operator
    # intact for diagonalization, including the one-shot Hamiltonian.
    if not config.no_degen_averaging:
        if sigma_c_at_dft_ev is not None:
            sigma_c_at_dft_ev = average_within_degenerate_sets(
                np.asarray(sigma_c_at_dft_ev, dtype=np.complex128),
                energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
                tol_ry=float(config.degen_avg_tol_ry))
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
        if head_at_dft_ry is not None:
            head_at_dft_ry = _average_head_diag(head_at_dft_ry)
        if grid_debug is not None and grid_debug["head_sigma_diag_w_kn_ry"] is not None:
            grid_debug["head_sigma_diag_w_kn_ry"] = _average_head_diag(
                grid_debug["head_sigma_diag_w_kn_ry"])
        photon_head_sigma_diag_tskn_ry = average_within_degenerate_sets(
            np.asarray(photon_head_sigma_diag_tskn_ry),
            energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
            tol_ry=float(config.degen_avg_tol_ry))
    from gw.qsgw_utils import static_sigma_diag_to_host
    sig_x_diag_ry = static_sigma_diag_to_host(sig_x, mesh_xy)
    if not config.no_degen_averaging:
        sig_x_diag_ry = average_within_degenerate_sets(
            sig_x_diag_ry, energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
            tol_ry=float(config.degen_avg_tol_ry))
    sigma_xc_at_dft_ev = (
        sig_x_diag_ry * RYD_TO_EV
        + sigma_c_at_dft_ev
        if sigma_c_at_dft_ev is not None else None)
    if (grid_debug is not None and grid_debug["head_sigma_diag_w_kn_ry"] is not None
            and grid_debug["omega_grid_ev"] is not None and omega_dft_rel_ev is not None
            and config.debug.sigma_freq_debug_output):
        # The grid's charge-head column: its head interpolated at E_DFT.
        from .qsgw_utils import interp_along_omega
        head_at_dft_ry = interp_along_omega(
            np.asarray(grid_debug["head_sigma_diag_w_kn_ry"]),
            np.asarray(grid_debug["omega_grid_ev"]), np.asarray(omega_dft_rel_ev),
            context="charge-head Sigma_c at E_DFT",
            print_fn=lambda *args, **kwargs: None)
    return (e_eval_ev, efermi_dft_ev, final_static_head_terms, grid_debug, h_transverse, head_at_dft_ry, omega_dft_rel_ev, photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, sig_coh, sig_h, sig_h_scalar, sig_sx, sig_x, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev, sigma_lorentz_skij_ry, sigma_omega_h5_path, sigma_total, sigma_xc_at_dft_ev)


def _diagonalize_qp_hamiltonian(
        band_slices, config, input_dir, kin_ion, mesh_xy, print0, qp_solver, sigma_total, wfn,
        sc_result=None, *, stages):
    """Produce the QP eigensystem and initialize output timing.

    SC: the accepted map's spectrum, as the SC driver wrote it to eqp0/eqp1
    (WFN_qp and the rotations are the SC driver's too); no second,
    unpartitioned eigh of kin_ion + Sigma_final, and no U.
    """
    if sc_result is not None:
        stages.result("QP bands of the accepted map")
        return (sc_result.qp_energies_ry, None, time.perf_counter())
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
    with timing.section("gw_jax.qp_eigh") as _sec_eigh:
        # The SC loop's eigensolver entry point: k staged over the mesh, each device
        # solving nk/P matrices, U replicated for the host writers below.
        from gw.sc_iteration import qp_eigh, qp_hamiltonian_sum
        E_full, U_full = qp_eigh(
            qp_hamiltonian_sum(kin_ion, sigma_total, mesh_xy),
            mesh_xy=mesh_xy, config=config, print_fn=print0)
        _sec_eigh.watch(E_full, U_full)
    _nk_qp = int(np.shape(E_full)[0])
    _nk_full = int(np.prod(np.asarray(wfn.kgrid).reshape(-1)[:3]))
    _nk_ibz = int(np.shape(np.asarray(wfn.kpoints))[0])
    _kset = (" (full zone)" if _nk_qp == _nk_full else
             " (IBZ)" if _nk_qp == _nk_ibz else "")
    stages.result(f"H(QP) diagonalized on {_nk_qp} k{_kset}")
    sanity.refuse_nonfinite(
        "E_qp (eigh of H_QP)", E_full, print_fn=print0,
        detail="LAPACK returns without complaining on a NaN-bearing matrix, "
               "so this is the last place a NaN spectrum can be stopped "
               "before eqp0/eqp1/WFN_qp.h5.")
    _t_out = time.perf_counter()
    if config.debug.write_wfn_h5:
        write_qp_wfn_oneshot(
            U_full, E_full, wfn=wfn, band_slices=band_slices,
            input_dir=input_dir, qp_solver=qp_solver, mesh=mesh_xy,
            print_fn=print0)
    return (E_full, U_full, _t_out)


def _sigma_diagnostic_fields(
        config, enk_dft, final_static_head_terms, h_transverse, head_at_dft_ry,
        mesh_xy, mode, photon_head_sigma_diag_tskn_ry,
        sig_coh, sig_h, sig_h_scalar, sig_sx, sigma_c_at_dft_ev,
        sigma_lorentz_skij_ry, sigma_xc_at_dft_ev, grid_debug=None, sigma_result=None,
        sigma_c_odd_at_dft_ev=None):
    """Produce the existing host diagnostic diagonals and head-sector split
    (and, under sigma_omega_grid_debug, the grid's: :func:`_grid_debug_diagonals`)."""
    from gw.qsgw_utils import static_sigma_diag_to_host
    sigma_c_omega_diag_ev, omega_rel_ev, sigma_c_odd_at_dft_ev = _grid_debug_diagonals(
        grid_debug, config, enk_dft, mesh_xy, sigma_result, sigma_c_odd_at_dft_ev)
    sigma_c_diag_at_dft_ry = (
        sigma_c_at_dft_ev / RYD_TO_EV
        if (mode.is_dynamic and sigma_c_at_dft_ev is not None)
        else None
    )
    head_sigma_split_skn_ry = None
    if config.bispinor and config.debug.sigma_freq_debug_output:
        head_sigma_split_skn_ry = np.asarray(
            photon_head_sigma_diag_tskn_ry[1]
            + photon_head_sigma_diag_tskn_ry[2]).copy()
        charge_head = None
        if final_static_head_terms is not None:
            if mode is ComputeMode.X_ONLY or mode.is_dynamic:
                charge_head = np.asarray(final_static_head_terms.sigma_x_diag)
            else:
                charge_head = np.asarray(
                    final_static_head_terms.sigma_sx_diag
                    + final_static_head_terms.sigma_coh_diag)
            if charge_head.ndim == 1:
                charge_head = np.broadcast_to(
                    charge_head, np.asarray(enk_dft).shape)
        if mode.is_dynamic and head_at_dft_ry is not None:
            charge_head = (head_at_dft_ry if charge_head is None
                           else charge_head + head_at_dft_ry)
        if charge_head is not None:
            head_sigma_split_skn_ry[0] += charge_head
    sig_sx_diag_ry = static_sigma_diag_to_host(sig_sx, mesh_xy)
    sig_coh_diag_ry = static_sigma_diag_to_host(sig_coh, mesh_xy)
    sig_h_diag_ry = static_sigma_diag_to_host(sig_h, mesh_xy)
    sig_h_scalar_diag_ry = static_sigma_diag_to_host(sig_h_scalar, mesh_xy)
    h_transverse_diag_ry = (
        None if h_transverse is None
        else static_sigma_diag_to_host(h_transverse, mesh_xy))
    if not config.no_degen_averaging:
        (sig_sx_diag_ry, sig_coh_diag_ry, sig_h_scalar_diag_ry,
         h_transverse_diag_ry) = (
            None if diagonal is None else average_within_degenerate_sets(
                diagonal, energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
                tol_ry=float(config.degen_avg_tol_ry))
            for diagonal in (sig_sx_diag_ry, sig_coh_diag_ry,
                             sig_h_scalar_diag_ry, h_transverse_diag_ry))
        sig_h_diag_ry = (sig_h_scalar_diag_ry if h_transverse_diag_ry is None
                         else sig_h_scalar_diag_ry + h_transverse_diag_ry)
    sigma_lorentz_diag_skn_ry = None
    if sigma_lorentz_skij_ry is not None:
        sigma_lorentz_diag_skn_ry = np.stack([
            static_sigma_diag_to_host(
                sigma_lorentz_skij_ry[sector], mesh_xy)
            for sector in range(3)
        ])
        if not config.no_degen_averaging:
            sigma_lorentz_diag_skn_ry = average_within_degenerate_sets(
                sigma_lorentz_diag_skn_ry,
                energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
                tol_ry=float(config.degen_avg_tol_ry))
        if mode.is_dynamic and sigma_xc_at_dft_ev is not None:
            _output_total_ry = np.asarray(sigma_xc_at_dft_ev) / RYD_TO_EV
            sigma_lorentz_diag_skn_ry[0] = (
                _output_total_ry - sigma_lorentz_diag_skn_ry[1]
                - sigma_lorentz_diag_skn_ry[2])
    return (h_transverse_diag_ry, head_sigma_split_skn_ry, sig_coh_diag_ry, sig_h_diag_ry, sig_h_scalar_diag_ry, sig_sx_diag_ry, sigma_c_diag_at_dft_ry, sigma_lorentz_diag_skn_ry, omega_rel_ev, sigma_c_odd_at_dft_ev, sigma_c_omega_diag_ev)


def _grid_debug_diagonals(grid_debug, config, enk_dft, mesh_xy, sigma_result,
                          sigma_c_odd_at_dft_ev):
    """The omega-grid debug path's host diagonals (``sigma_omega_grid_debug``).

    ``(sigma_c_omega_diag_ev, omega_rel_ev, sigma_c_odd_at_dft_ev)``: the
    cube's (n_omega, nk, nb) diagonal in eV for the eqp assembly's C(omega)
    curve, its grid, and the odd diagonal averaged within degenerate sets as
    the curve is.  ``(None, None, sigma_c_odd_at_dft_ev)`` on shell.
    """
    if grid_debug is None:
        return None, None, sigma_c_odd_at_dft_ev
    # The (n_omega, nk, nb) diagonal is read only by the writers, which
    # run on the reporting rank: the other ranks keep no host copy.
    sigma_c_omega_diag_ev = extract_sigma_diag_logical(
        grid_debug["sigma_c_omega"], mesh_xy,
        band_axis=sigma_result.sigma_band_axis,
        host=jax.process_index() == 0)
    if sigma_c_omega_diag_ev is not None:
        sigma_c_omega_diag_ev = sigma_c_omega_diag_ev * RYD_TO_EV
    if not config.no_degen_averaging:
        if sigma_c_omega_diag_ev is not None:
            sigma_c_omega_diag_ev = average_within_degenerate_sets(
                sigma_c_omega_diag_ev,
                energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
                tol_ry=float(config.degen_avg_tol_ry))
        if sigma_c_odd_at_dft_ev is not None:
            sigma_c_odd_at_dft_ev = average_within_degenerate_sets(
                np.asarray(sigma_c_odd_at_dft_ev),
                energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
                tol_ry=float(config.degen_avg_tol_ry))
    return sigma_c_omega_diag_ev, grid_debug["omega_grid_ev"], sigma_c_odd_at_dft_ev


def _attach_raw_sigma(results, config, enk_dft, sigma_result):
    """Carry the on-shell eqp operand and its raw (no band tail) twin, averaged.

    Same degenerate-set averaging as ``_sigma_output_fields`` applies to the
    extrapolated diagonals, so the raw columns equal an extrapolation-off run's.
    """
    import dataclasses
    if sigma_result.sigma_c_omega_kij_ry is not None:
        # The omega grid (sigma_omega_grid_debug): the raw C(omega) curve and
        # its Sigma_c(E_DFT), conditioned as the main pair.
        raw_omega_ry = sigma_result.sigma_c_omega_diag_unextrap_ry
        raw_at_dft_ev = sigma_result.sigma_c_at_dft_diag_unextrap_ev
        if raw_omega_ry is None or raw_at_dft_ev is None:
            return results
        raw_omega_ev = np.asarray(raw_omega_ry) * RYD_TO_EV
        raw_at_dft_ev = np.asarray(raw_at_dft_ev, dtype=np.complex128)
        if not config.no_degen_averaging:
            energies = np.asarray(enk_dft, dtype=np.float64)
            tol = float(config.degen_avg_tol_ry)
            raw_omega_ev = average_within_degenerate_sets(
                raw_omega_ev, energies_kn_ry=energies, tol_ry=tol)
            raw_at_dft_ev = average_within_degenerate_sets(
                raw_at_dft_ev, energies_kn_ry=energies, tol_ry=tol)
        return dataclasses.replace(
            results, sigma_c_omega_diag_raw_ev=raw_omega_ev,
            sigma_c_diag_at_dft_raw_ry=raw_at_dft_ev / RYD_TO_EV)
    return dataclasses.replace(
        results,
        sigma_c_on_shell_diag=_averaged_on_shell(
            sigma_result.sigma_c_on_shell_diag, config, enk_dft),
        sigma_c_on_shell_diag_raw=_averaged_on_shell(
            sigma_result.sigma_c_on_shell_diag_unextrap, config, enk_dft))


def _averaged_on_shell(operand, config, enk_dft):
    """An on-shell eqp operand averaged within degenerate sets (None passes through).

    Its three complex rows (``eqp_bgw.ON_SHELL_AT_DFT`` ... ``ON_SHELL_DSIGMA``)
    are averaged, so Z is formed from the averaged derivative; the coverage
    row is a per-state fact.
    """
    if operand is None or config.no_degen_averaging:
        return operand
    energies = np.asarray(enk_dft, dtype=np.float64)
    tol = float(config.degen_avg_tol_ry)
    rows = [average_within_degenerate_sets(
        np.asarray(row, dtype=np.complex128), energies_kn_ry=energies, tol_ry=tol)
        for row in operand[:3]]
    return np.stack(rows + [operand[3]])


def _assemble_gw_results(
        E_full, U_full, band_slices, config, e_eval_ev, efermi_dft_ev, enk_dft,
        h_transverse, h_transverse_diag_ry, kin_ion, mode,
        photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, qp_solver, sig_coh,
        sig_coh_diag_ry, sig_h, sig_h_diag_ry, sig_h_scalar, sig_h_scalar_diag_ry, sig_sx,
        sig_sx_diag_ry, sig_x, sig_x_diag_ry, sigma_c_diag_at_dft_ry, sigma_c_odd_at_dft_ev,
        sigma_lorentz_diag_skn_ry, sigma_omega_h5_path, sigma_result,
        sigma_xc_at_dft_ev, tensors_filename, sigma_c_omega_diag_ev=None, omega_rel_ev=None):
    """Produce the existing GW result record from resolved output fields."""
    results = GWResults(
        sig_sx=sig_sx,
        sig_coh=sig_coh,
        sig_h=sig_h,
        sig_h_scalar=sig_h_scalar,
        h_transverse=h_transverse,
        sig_x=sig_x,
        sig_sx_diag_ry=sig_sx_diag_ry,
        sig_coh_diag_ry=sig_coh_diag_ry,
        sig_h_diag_ry=sig_h_diag_ry,
        sig_h_scalar_diag_ry=sig_h_scalar_diag_ry,
        h_transverse_diag_ry=h_transverse_diag_ry,
        sig_x_diag_ry=sig_x_diag_ry,
        E_qp_ry=np.array(E_full),
        U_qp=None if U_full is None else np.array(U_full),
        E_dft_ry=np.array(enk_dft),
        kin_ion_ry=np.array(kin_ion),
        band_start=band_slices.b0,
        band_stop=band_slices.b3,
        use_ppm=mode.is_dynamic,
        self_consistent=qp_solver is QPSolver.SELF_CONSISTENT,
        sigma_kset=sigma_result.kset,
        sigma_c_diag_at_dft_ry=sigma_c_diag_at_dft_ry,
        sigma_c_omega_diag_ev=sigma_c_omega_diag_ev,
        omega_rel_ev=omega_rel_ev,
        sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
        e_eval_ev=e_eval_ev,
        efermi_ev=efermi_dft_ev,
        sigma_omega_h5_path=sigma_omega_h5_path,
        tensors_filename=tensors_filename,
        photon_head_sigma_diag_tskn_ry=np.asarray(
            photon_head_sigma_diag_tskn_ry),
        photon_head_sigma_basis=photon_head_sigma_basis,
        sigma_lorentz_diag_skn_ry=sigma_lorentz_diag_skn_ry,
        sigma_c_odd_diag_at_dft_ry=(
            None if sigma_c_odd_at_dft_ev is None
            else np.asarray(sigma_c_odd_at_dft_ev) / RYD_TO_EV),
    )
    return _attach_raw_sigma(results, config, enk_dft, sigma_result)


def _write_gw_results(
        _t_out, config, e_eval_ev, final_static_head_terms, grid_debug, head_at_dft_ry,
        input_dir, meta, omega_dft_rel_ev, print0, qp_solver,
        results, rotations_written, sym, wfn):
    """Write the QP and diagnostic results on the reporting rank."""
    if meta.rank == 0:
        write_freq_debug(
            results, config=config,
            static_head_terms=final_static_head_terms,
            omega_dft_rel_ev=omega_dft_rel_ev,
            head_at_dft_ry=head_at_dft_ry,
            head_sigma_diag_w_kn_ry=(None if grid_debug is None
                                     else grid_debug["head_sigma_diag_w_kn_ry"]),
            omega_grid_ry=None if grid_debug is None else grid_debug["omega_grid_ry"],
            sym=sym, file_sym=wfn.symmetry(),
            e_eval_ev=e_eval_ev,
            print_fn=print0,
        )
        write_qsgw_qp_ladders(
            results, config=config,
            e_qp_ry=results.E_qp_ry,
            omega_grid_ev=None if grid_debug is None else grid_debug["omega_grid_ev"],
            sigma_c_omega=None if grid_debug is None else grid_debug["sigma_c_omega"],
            print_fn=print0,
        )
        write_results(
            results,
            sigma_diag_file=config.paths.sigma_diag_file,
            eqp0_file=config.paths.eqp0_file,
            eqp1_file=config.paths.eqp1_file,
            input_dir=input_dir,
            kgrid=meta.kgrid,
            sym=sym,
            wfn=wfn,
            degeneracy_policy=(
                "disabled" if config.no_degen_averaging else "bgw_average"),
            degeneracy_tol_ry=float(config.degen_avg_tol_ry),
            write_qp_rotations=not rotations_written,
            qp_rotations_k_storage=config.qp_rotations_k_storage,
            qp_solver=qp_solver,
            print_fn=print0,
        )
        timing.record("gw_jax.output", time.perf_counter() - _t_out)


def _record_process_wall(_pre_main, _t_main):
    """Record the pre-main rows and return the complete process wall time."""
    if _pre_main is not None:
        _phases = RUNTIME.facts.get("elapsed", {})
        for _phase, _secs in sorted(_phases.items()):
            if _phase != "total":
                timing.record(f"gw_jax.runtime_stack.{_phase}", _secs)
        timing.record("gw_jax.imports",
                      max(_pre_main - _phases.get("total", 0.0), 0.0))
    return time.perf_counter() - _t_main + (_pre_main or 0.0)


def _close_timing(_pre_main, _t_main, meta, print0):
    """Produce the complete process wall time and timing decomposition."""
    _wall = _record_process_wall(_pre_main, _t_main)
    # Every process: the stage-memory table's max and min over ranks.
    timing.gather_peaks()
    if meta.rank == 0 and debug_print_enabled():
        timing.report(print_fn=print0, title="--- Timing ---", wall=_wall)
    return (_wall)


def _report_final_observables(
        E_full, band_slices, config, enk_dft, head_sigma_split_skn_ry,
        q0_certificates, report, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev,
        sigma_lorentz_diag_skn_ry, sigma_result, *, sym=None, wfn=None):
    """Report the sector summaries, any quadrature or Sigma coverage no stage
    showed, and RESULTS (the band edges and the gap correction)."""
    if sigma_lorentz_diag_skn_ry is not None:
        _labels = ("CC", "CT+TC", "TT")
        _parts_ev = np.asarray(sigma_lorentz_diag_skn_ry) * RYD_TO_EV
        report.progress(
            "Sigma blocks   : " + "; ".join(
                f"{label} max|diag|={np.max(np.abs(part)):.6e} eV, "
                f"mean|diag|={np.mean(np.abs(part)):.6e} eV"
                for label, part in zip(_labels, _parts_ev)))
    if head_sigma_split_skn_ry is not None:
        _head_parts_ev = np.asarray(head_sigma_split_skn_ry) * RYD_TO_EV
        report.progress(
            "Head Sigma     : Gamma-cell contribution; " + "; ".join(
                f"{label} max|diag|={np.max(np.abs(part)):.6e} eV, "
                f"mean|diag|={np.mean(np.abs(part)):.6e} eV"
                for label, part in zip(("CC", "CT+TC", "TT"),
                                       _head_parts_ev)))
    if sigma_c_odd_at_dft_ev is not None:
        _odd_abs = np.abs(np.asarray(sigma_c_odd_at_dft_ev))
        # An SC run on shell has no DFT-basis Sigma_c(E_DFT): the share is
        # then of |Sigma_x| (named in the line below).
        _xc_diag_ev = (
            np.sum(np.asarray(sigma_lorentz_diag_skn_ry), axis=0) * RYD_TO_EV
            if sigma_lorentz_diag_skn_ry is not None else
            np.asarray(sig_x_diag_ry) * RYD_TO_EV
            + (0.0 if sigma_c_at_dft_ev is None else np.asarray(sigma_c_at_dft_ev)))
        _xc_abs = np.abs(_xc_diag_ev)
        _max_share = (np.max(_odd_abs) / np.max(_xc_abs)
                      if np.max(_xc_abs) > 0.0 else np.max(_odd_abs))
        _mean_share = (np.mean(_odd_abs) / np.mean(_xc_abs)
                       if np.mean(_xc_abs) > 0.0 else np.mean(_odd_abs))
        _odd_prefix = (
            "MPA odd Sigma  "
            if config.compute_mode.is_pole_route
            else "GN odd Sigma   ")
        _odd_line = (
            f"{_odd_prefix}: measured-broken-TR ordered residue; "
            f"max|sigC_odd|={np.max(_odd_abs):.6e} eV; "
            f"mean|sigC_odd|={np.mean(_odd_abs):.6e} eV; "
            f"max-share-of-|Sigma_{'xc' if sigma_c_at_dft_ev is not None or sigma_lorentz_diag_skn_ry is not None else 'x'}|={_max_share:.6e}; "
            f"mean-share-of-|Sigma_{'xc' if sigma_c_at_dft_ev is not None or sigma_lorentz_diag_skn_ry is not None else 'x'}|={_mean_share:.6e}")
        if sigma_result.ppm_probe_hermiticity_residual is not None:
            _odd_line += (
                f"; W(iomega_p)-Hermiticity="
                f"{sigma_result.ppm_probe_hermiticity_residual:.3e}")
        if sigma_result.ppm_odd_even_residue_ratio is not None:
            _odd_line += (
                f"; max|D|/max|B|="
                f"{sigma_result.ppm_odd_even_residue_ratio:.3e}")
        report.progress(_odd_line)
    if q0_certificates:
        _q0_cert = max(q0_certificates, key=lambda item: item.final_error_ratio)
        report.progress(
            "Slab WS cert   : exact q=0 charge-head cubature; "
            f"orders={_q0_cert.orders}; nodes={_q0_cert.physical_counts}; "
            f"polygon_edges={_q0_cert.polygon_edges}; evaluations="
            f"{len(q0_certificates)}; max_final_error_ratio="
            f"{_q0_cert.final_error_ratio:.3e} (<=1 required)")
    _sc = config.qp_solver.value == "self_consistent"
    report.quadrature(self_consistent=_sc)
    # A no-op when the Self-energy stage showed it (one-shot).
    report.sigma_coverage(
        config=config, band_slices=band_slices, enk_dft_ry=enk_dft,
        sigma_result=sigma_result, final=True)
    # SC: the loop's record (read only on an SC run; see gw.sc_report.last_run).
    from .sc_report import last_run
    report.results(
        band_slices=band_slices, e_dft_ry=enk_dft, e_qp_ry=E_full,
        config=config, sym=sym, wfn=wfn, sigma_result=sigma_result,
        sc_run=last_run() if _sc else None)


def _published_artifacts(rows):
    """Refuse to call a run completed while an artifact is unpublished.

    A run that prints the completion line while an artifact is missing or
    still inside a collective write transaction is not a usable run: on
    2026-09-19 the converged Fe runs 10p/10q/10r each printed the line and
    left ``sigma_mnk.h5`` at ``lorrax_io_committed = 0``.  Rows are
    ``(label, path, required)``; a present file carrying the commit receipt
    dataset must read 1.
    """
    import h5py

    ok = True
    log.info("  published artifacts:")
    for label, path, required in rows:
        if path is None:
            log.info(f"    {label}: not requested")
            continue
        path = str(path)
        if not os.path.exists(path):
            log.info(f"    {label}: {'MISSING' if required else 'absent'} ({path})")
            ok = ok and not required
            continue
        state = "present"
        try:
            with h5py.File(path, "r") as h5:
                if "lorrax_io_committed" in h5:
                    committed = int(np.asarray(h5["lorrax_io_committed"])[0])
                    state = f"committed={committed}"
                    if committed != 1:
                        ok = False
        except Exception as exc:              # unreadable is not published
            state = f"unreadable ({type(exc).__name__})"
            ok = False
        log.info(f"    {label}: {state} ({path})")
    return ok


def _report_file_rows(args, config, input_dir, report, sigma_omega_h5_path, tensors_filename,
                      sigma_result=None):
    """Produce the report rows for consumed and generated files."""
    raw_columns = getattr(
        sigma_result, "sigma_c_at_dft_diag_unextrap_ev", None) is not None
    _file_rows = [
        ("input deck", "read", args.input),
        ("DFT wavefunctions", "read", config.paths.wfn_file),
        ("ISDF centroids", "read", config.paths.centroids_file),
        ("mean-field Hamiltonian", "read", config.paths.kin_ion_file),
        ("long-wave dipoles", "read" if os.path.exists(os.path.join(
            input_dir, "dipole.h5")) else "absent",
         os.path.join(input_dir, "dipole.h5")),
        # Its raw (no band tail) columns are named under Dynamic Sigma.
        ("self-energy table" + (" (+ raw columns)" if raw_columns else ""),
         "written" if os.path.exists(
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
    ]
    if os.path.exists(config.paths.parallel_transport_file):
        _file_rows.insert(5, ("parallel transport", "read",
                              config.paths.parallel_transport_file))
    if config.restart or (getattr(config, "write_restart_tensors", True)
                          and os.path.exists(tensors_filename)):
        _file_rows.insert(5, ("ISDF restart tensors",
                              "read" if config.restart else "written", tensors_filename))
    _wfn_qp = os.path.join(input_dir, "WFN_qp.h5")
    # The row is shown when the file was written, or when the deck asked for
    # it (write_wfn_h5 is on by default, and the one-shot writer may skip it).
    _wfn_qp_asked = "write_wfn_h5" in (getattr(config, "raw_input_keys", ()) or ())
    if bool(config.debug.write_wfn_h5) and (os.path.exists(_wfn_qp) or _wfn_qp_asked):
        _file_rows.append((QP_WFN_FILE_ROLE,
                           "written" if os.path.exists(_wfn_qp) else "absent", _wfn_qp))
    if config.paths.centroids_file_current and (
            os.path.realpath(str(config.paths.centroids_file_current))
            != os.path.realpath(str(config.paths.centroids_file))):
        _file_rows.insert(3, (
            "current centroids", "read", config.paths.centroids_file_current))
    if sigma_omega_h5_path:
        _file_rows.append((
            "dynamic Sigma spectrum",
            "written" if os.path.exists(sigma_omega_h5_path) else "absent",
            sigma_omega_h5_path))
    _file_rows.append(("calculation report", "written", report.path))
    return (_file_rows)


def main(argv=None):
	"""Run the GW stages; see docs/dev/gw_config_contracts.md."""
	args = build_parser().parse_args(argv)
	_t_main = time.perf_counter()
	_pre_main = timing.process_elapsed_s()
	timing.reset()
	quadrature_log.reset()
	opened = _open_production_report(args)
	try:
		return _run_gw_stages(args, _t_main, _pre_main, opened)
	except BaseException as exc:
		# A named refusal (GATE ...) ends the run before the completion
		# report; its stage table is the only record of where the wall went.
		_report_refused_run(opened[4], _pre_main, _t_main, exc)
		raise
	finally:
		timing.set_memory_sink()
		opened[5].close()


def _report_refused_run(report, _pre_main, _t_main, exc):
	"""Print the major-stage table and name the refusal, then let it raise."""
	try:
		_wall = _record_process_wall(_pre_main, _t_main)
		if report.rank == 0 and debug_print_enabled():
			timing.report(print_fn=report.legacy_print, title="--- Timing ---",
			              wall=_wall)
		report.timings(timing.records(), wall=_wall)
		report.finish(status="ERROR", wall=_wall, error=exc)
	except Exception:  # noqa: BLE001 - a report failure must not mask the refusal
		pass


def _run_gw_stages(args, _t_main, _pre_main, opened):
	"""The GW stages from the opened report to the completion line."""
	(
	    config, input_dir, qp_solver, mode, report, _streams, print0,
	    _config_provenance, do_screened) = opened
	_report_head_and_photon_policy(config, report)
	mesh_xy = RUNTIME.mesh
	_setup_runtime()
	from file_io.hdf5_owner import probe as _hdf5_probe
	_hdf5_probe("startup", print_fn=print0)
	(
	    config, wfn, material_class, sym, centroid_basis, centroid_indices, n_rmu, tmp_dir,
	    tensors_filename) = _load_system_inputs(
	    config, input_dir, mesh_xy, report, print0, _config_provenance)
	(
	    meta, band_slices, zeta_fit_edge) = prepare_band_metadata(
	    centroid_indices, config, mesh_xy, n_rmu, print0, sym, wfn)
	(
	    enk_dft) = _report_sampling_and_bands(
	    band_slices, centroid_basis, config, material_class, mesh_xy, meta, mode,
	    report, sym, wfn, zeta_fit_edge)
	q0_certificates = []
	head_resolver = HeadResolver(
		config, input_dir, wfn, sym, meta, print0,
		q0_certificate_fn=q0_certificates.append, mesh=mesh_xy)
	bgw_v_grid_fn = build_bgw_v_grid_fn(
		config, wfn=wfn, sym=sym, input_dir=input_dir, print_fn=print0)
	timing.record("gw_jax.startup", time.perf_counter() - _t_main)
	(
	    V_qmunu, bispinor_v_q_path, green_parent_carrier, isdf, oneshot_occupation_state, wfns,
	    wfns_screening, wfns_sigma, wfns_transverse) = _prepare_isdf_carriers(
	    band_slices, bgw_v_grid_fn, centroid_indices, config, material_class, mesh_xy, meta,
	    mode, print0, qp_solver, sym, tensors_filename, tmp_dir, wfn, enk_dft,
	    centroid_basis=centroid_basis)
	V_q = V_qmunu              # flat-q (nq, μ, μ) — compute and restart alike
	# Stages 2-4 of the main log (one-shot: Screening, Self-energy, QP bands;
	# QSGW: its driver, QP bands), each opened by the helper that starts it.
	with _GWStages(qp_solver) as stages:
		(
		    quad, e_ref, oneshot_head_response, oneshot_head_requests, oneshot_mpa_plan) = _prepare_oneshot_response(
		    config, do_screened, input_dir, material_class, mesh_xy, meta, mode, print0, qp_solver,
		    wfn, wfns, wfns_sigma, occupation_state=oneshot_occupation_state, stages=stages)
		(
		    W_by_role, photon_response, green_parent_carrier, wfns_screening) = _run_oneshot_screening(
		    V_q, bispinor_v_q_path, centroid_indices, config, e_ref, green_parent_carrier,
		    head_resolver, isdf, material_class, mesh_xy, meta, mode, oneshot_head_response,
		    oneshot_mpa_plan, oneshot_occupation_state, print0, qp_solver, quad, report, sym,
		    tensors_filename, tmp_dir, wfn, wfns_screening, wfns_transverse, stages=stages)
		_install_oneshot_head(
		    W_by_role, config, head_resolver, mesh_xy, meta, mode, oneshot_head_requests,
		    oneshot_head_response, wfn)
		_persist_screening(
		    V_q, W_by_role, centroid_indices, config, head_resolver, mesh_xy, meta, mode, print0,
		    qp_solver, sym, tensors_filename, photon_response=photon_response,
		    material_class=material_class, head_response=oneshot_head_response)
		(
		    static_head_terms) = _prepare_static_head(
		    config, do_screened, head_resolver, meta, mode, qp_solver,
		    oneshot_occupation_state)
		(
		    sigma_result, W_by_role, photon_response) = _run_oneshot_sigma(
		    V_q, W_by_role, band_slices, bispinor_v_q_path, config, enk_dft, head_resolver,
		    input_dir, isdf, material_class, mesh_xy, meta, mode, oneshot_occupation_state,
		    photon_response, print0, qp_solver, quad, static_head_terms, sym, wfn, wfns_sigma,
		    wfns_transverse, report=report, stages=stages)
		(
		    enk_dft, final_static_head_terms, kin_ion, rotations_written, sc_result,
		    sigma_result, sigma_total) = _solve_qp_stage(
		    V_q, band_slices, bispinor_v_q_path, centroid_indices, config, e_ref, enk_dft,
		    head_resolver, input_dir, isdf, None, material_class, mesh_xy, meta, print0,
		    qp_solver, quad, report, sigma_result, static_head_terms, sym, tensors_filename, wfn,
		    wfns_sigma, wfns_transverse, stages=stages)
		(
		    e_eval_ev, efermi_dft_ev, final_static_head_terms, grid_debug, h_transverse,
		    head_at_dft_ry, omega_dft_rel_ev,
		    photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, sig_coh, sig_h, sig_h_scalar,
		    sig_sx, sig_x, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev,
		    sigma_lorentz_skij_ry, sigma_omega_h5_path, sigma_total, sigma_xc_at_dft_ev) = _sigma_output_fields(
		    config, enk_dft, final_static_head_terms, mesh_xy, qp_solver, sc_result, sigma_result,
		    sigma_total)
		(
		    E_full, U_full, _t_out) = _diagonalize_qp_hamiltonian(
		    band_slices, config, input_dir, kin_ion, mesh_xy, print0, qp_solver, sigma_total, wfn,
		    sc_result, stages=stages)
		(
		    h_transverse_diag_ry, head_sigma_split_skn_ry, sig_coh_diag_ry,
		    sig_h_diag_ry, sig_h_scalar_diag_ry, sig_sx_diag_ry, sigma_c_diag_at_dft_ry,
		    sigma_lorentz_diag_skn_ry, omega_rel_ev, sigma_c_odd_at_dft_ev,
		    sigma_c_omega_diag_ev) = _sigma_diagnostic_fields(
		    config, enk_dft, final_static_head_terms, h_transverse, head_at_dft_ry,
		    mesh_xy, mode, photon_head_sigma_diag_tskn_ry, sig_coh,
		    sig_h, sig_h_scalar, sig_sx, sigma_c_at_dft_ev,
		    sigma_lorentz_skij_ry, sigma_xc_at_dft_ev, grid_debug, sigma_result,
		    sigma_c_odd_at_dft_ev)
		(
		    results) = _assemble_gw_results(
		    E_full, U_full, band_slices, config, e_eval_ev, efermi_dft_ev, enk_dft,
		    h_transverse, h_transverse_diag_ry, kin_ion, mode,
		    photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, qp_solver, sig_coh,
		    sig_coh_diag_ry, sig_h, sig_h_diag_ry, sig_h_scalar, sig_h_scalar_diag_ry, sig_sx,
		    sig_sx_diag_ry, sig_x, sig_x_diag_ry, sigma_c_diag_at_dft_ry, sigma_c_odd_at_dft_ev,
		    sigma_lorentz_diag_skn_ry, sigma_omega_h5_path, sigma_result,
		    sigma_xc_at_dft_ev, tensors_filename, sigma_c_omega_diag_ev, omega_rel_ev)
		_write_gw_results(
		    _t_out, config, e_eval_ev, final_static_head_terms, grid_debug, head_at_dft_ry, input_dir,
		    meta, omega_dft_rel_ev, print0, qp_solver, results,
		    rotations_written, sym, wfn)
	(_wall) = _close_timing(_pre_main, _t_main, meta, print0)
	_report_final_observables(
	    E_full, band_slices, config, enk_dft, head_sigma_split_skn_ry,
	    q0_certificates, report, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev,
	    sigma_lorentz_diag_skn_ry, sigma_result, sym=sym, wfn=wfn)
	(
	    _file_rows) = _report_file_rows(
	    args, config, input_dir, report, sigma_omega_h5_path, tensors_filename,
	    sigma_result)
	report.timings(timing.records(), wall=_wall)
	report.warnings()
	report.files(_file_rows)
	_completion_artifacts = [
	    ("Sigma matrix elements", sigma_omega_h5_path, True),
	    ("restart tensors", tensors_filename, False),
	]
	if sc_result is not None:
	    # The SC writer validates each private sibling before atomic replace.
	    # Requiring both final names in the existing completion manifest makes
	    # the report's terminal status the bundle receipt: a process killed
	    # between the two replacements leaves no completed-run claim.
	    _completion_artifacts.append((
	        "SC QP rotations",
	        os.path.join(input_dir, "qp_wfn_rotations.h5"), True))
	    if bool(config.debug.write_wfn_h5):
	        _completion_artifacts.append((
	            "SC QP WFN", os.path.join(input_dir, "WFN_qp.h5"), True))
	if not _published_artifacts(tuple(_completion_artifacts)):
	    raise RuntimeError(
	        "GATE gw_artifacts_unpublished: got: a required artifact is "
	        "missing or left uncommitted (manifest above); want: every "
	        "listed artifact present and committed; why: a run that prints "
	        "the completion line while a published artifact is missing or "
	        "still inside its write transaction is not a usable run")
	report.finish(wall=_wall)
	return 0


if __name__ == "__main__":
	from runtime import run_main_and_finalize
	run_main_and_finalize(main)

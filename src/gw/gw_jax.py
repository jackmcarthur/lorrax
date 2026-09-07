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
import argparse


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
from common.wfn_transforms import get_enk_bandrange
import common.timing as timing
from .gw_config import (
	ComputeMode, HeadCorrection, LorraxConfig, QPSolver,
	ScreeningDiagrams, incumbent_bispinor_head_record,
	packed_bare_transverse_route,
	packed_photon_replaces_charge_sigma, packed_photon_screens_current,
	refuse_unimplemented_compute_mode, uses_dynamic_packed_photon_route,
	uses_four_spinor_finite_q_charge, uses_static_photon_response,
	infer_material_class, resolve_mpa_sampling_alpha,
	validate_material_inputs)
from .gw_init import (prepare_isdf_and_wavefunctions,
	                  check_band_sum_degeneracy, resolve_zeta_fit_edge,
	                  zeta_fit_band_ranges)
from .compute_vcoul import build_bgw_v_grid_fn
from .minimax_screening import build_static_quadrature
from .screening import (
	compute_screening_model, driver_persists_w0, screening_requests_for)
from .sigma_dispatch import (
	SIGMA_KSET_FULL_BZ, SIGMA_KSET_STAR_WEDGE, compute_sigma_xc,
	sigma_result_on_kset)
from .qsgw_utils import solve_qp
from .dynamic_sigma import extract_sigma_diag_logical
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


def _oneshot_mpa_occupation_state(config, wfn, wfns, material_class,
                                  mesh_xy=None, print_fn=print):
	"""Solve the fixed-N MP1 state consumed by one-shot metallic MPA.

	The self-consistent driver already solves this state at map entry.  A
	non-SC driver must solve it once from the DFT spectrum and thread that same
	record to the MPA fit and Sigma; reconstructing either the chemical
	potential or occupations at a consumer would violate the one-state rule.

	One state per RUN also means one state across RANKS: the head fit is
	stamped with rank 0's ``occ_hash`` and every rank's Sigma body asserts
	against it, so the table is solved locally and then rank 0's copy is
	broadcast (one psum) to all processes.  Measured on Na 8x8x8 at P=16
	(2026-09-01): ranks 2 and 7 solved a table whose bytes differed from
	rank 0's at the same mu to 12 digits and refused with "head fit and
	Sigma body carry different occupation states".
	"""
	if material_class != "metal":
		return None
	from psp.get_DFT_mtxels import spin_degeneracy_factor
	from common.collectives import process_count, process_rank, psum_replicate
	from .efermi import OccupationState

	energies = np.asarray(wfns.enk, dtype=np.float64)
	if energies.ndim != 2 or min(energies.shape) < 1:
		raise ValueError(
			"one-shot metallic MPA occupation energies must be nonempty "
			f"(nk,nb), got {energies.shape}")
	nk = int(energies.shape[0])
	kweights = np.full(nk, 1.0 / float(nk), dtype=np.float64)
	local = OccupationState.solve_mp1(
		energies, kweights, float(wfn.num_electrons),
		float(config.occ_broadening_ry),
		state_capacity=spin_degeneracy_factor(wfn),
		clamp_tol=float(config.occupation_clamp_tol))
	if mesh_xy is None or process_count() <= 1:
		return local
	root = 1.0 if process_rank() == 0 else 0.0
	f_kn = psum_replicate(np.asarray(local.f_kn, dtype=np.float64) * root, mesh_xy)
	mu_ry = float(psum_replicate(np.array([float(local.mu_ry)]) * root, mesh_xy)[0])
	state = OccupationState(
		f_kn=f_kn, mu_ry=mu_ry, smearing_family=local.smearing_family,
		smearing_width_ry=float(local.smearing_width_ry),
		n_electrons=float(local.n_electrons))
	if state.occ_hash != local.occ_hash:
		print_fn(
			f"  one-shot occupations: rank {process_rank()} solved occ_hash="
			f"{local.occ_hash} (mu={float(local.mu_ry):.12g}); using rank 0's "
			f"{state.occ_hash} (mu={mu_ry:.12g}) so head and body agree")
	return state


def _open_production_report(args):
    """Produce the resolved configuration and its scientific report."""
    _config_provenance = []
    def _config_print(*values, sep=" ", **kwargs):
        debug_print(*values, sep=sep, **kwargs)
        text = sep.join(str(value) for value in values)
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
    production_stdout = ProductionStdout(
        debug=debug_print_enabled(), rank=RUNTIME.process_index,
        warning_fn=report.legacy_print)
    production_stdout.install()
    report.stdout = (rank0_print if debug_print_enabled()
                     else production_stdout.emit)
    print0 = report.legacy_print
    report.begin(input_file=args.input, config=config)
    refuse_unimplemented_compute_mode(mode, context="the LORRAX GW driver")
    do_screened = mode.needs_screening
    return (config, input_dir, qp_solver, mode, report, production_stdout, print0, _config_provenance, do_screened)


def _report_head_and_photon_policy(config, print0, report):
    """Report the resolved head and photon policies."""
    print0(
        f"  Head policy: head_correction={config.head.correction.value}; "
        f"screening_diagrams={config.screening.diagrams.value}; "
        f"direct diagnostic source={config.head.wcoul0_source}. "
        + ({
            HeadCorrection.FULL: "macroscopic W, local fields exactly once",
            HeadCorrection.NO_LOCAL_FIELDS: "diagnostic epsilon head",
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
        print0(
            f"  Bispinor GW policy: bispinor_gw={config.bispinor_gw.value}"
            f"{_bispinor_note}")
        _bare_taken, _bare_reason = packed_bare_transverse_route(config)
        if config.bispinor_gw.value == "full_static_cohsex":
            report.progress(
                "Photon route   : packed screened static photon operator "
                "(sixteen response and Sigma blocks; coupled 4x4 Dyson solve; "
                "Gamma-cell completion carries charge, mixed, and transverse heads)")
        else:
            report.progress(
                "Photon route   : "
                + ("packed static photon operator (chi_TT = chi_CT = 0; scalar "
                   "Dyson on CC, W_packed = diag(W_00, D_TT); the Gamma-cell "
                   "completion carries both the charge head and the bare "
                   f"<D_TT>) -- {_bare_reason}"
                   if _bare_taken else
                   "incumbent charge-screened W + Sigma^B "
                   "(gw.sigma_x_bispinor) with the scalar band-diagonal q->0 "
                   f"head -- {_bare_reason}"))
        if not uses_static_photon_response(config):
            _banner, _head_record = incumbent_bispinor_head_record(config)
            if _banner:
                print0(_banner)
            report.progress(f"Photon head    : {_head_record}")


def _load_system_inputs(config, input_dir, mesh_xy, report, print0, _config_provenance):
    """Produce wavefunctions, symmetry, centroid basis and output paths."""
    wfn = WfnLoader(config.paths.wfn_file, mesh=mesh_xy)
    material_class = infer_material_class(wfn.occs)
    config = resolve_mpa_sampling_alpha(
        config, material_class, print_fn=report.progress)
    validate_material_inputs(config, material_class)
    print0(f"  Material class: {material_class} (inferred from WFN occupations)")
    sym = wfn.symmetry()
    centroid_basis = load_centroid_basis(
        config.paths.centroids_file, wfn.fft_grid, sym=sym)
    centroid_sets = [centroid_basis]
    if config.bispinor and config.paths.centroids_file_current:
        centroid_sets.append(load_centroid_basis(
            config.paths.centroids_file_current, wfn.fft_grid, sym=sym))
    nonclosed = [basis.path for basis in centroid_sets if not basis.orbit_closed]
    if nonclosed:
        print0("WARNING: non-orbit-closed centroid set(s): " + ", ".join(nonclosed)
               + "; using unreduced parents (n_parent = nk) and full q on the same "
               "parent route. Generate orbit-closed centroids with kmeans to restore reduction.")
        sym = sym.trivial_view()
    centroid_indices = centroid_basis.centroid_indices
    n_rmu = centroid_basis.n_rmu
    tmp_dir = os.path.join(input_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{n_rmu}.h5")
    print_system_summary(
        n_rmu=n_rmu, fft_grid=wfn.fft_grid,
        cell_volume=wfn.cell_volume, print_fn=print0,
    )
    report.layout_dials(
        config=config, n_mu=n_rmu, n_q_irr=int(sym.nk_red),
        processes=RUNTIME.process_count)
    if _config_provenance:
        report.heading("Configuration provenance")
        for line in _config_provenance:
            report.emit(line.strip())
    report.architecture()
    report.method(config=config)
    return (config, wfn, material_class, sym, centroid_basis, centroid_indices, n_rmu, tmp_dir, tensors_filename)


def _prepare_band_metadata(centroid_indices, config, mesh_xy, n_rmu, print0, sym, wfn):
    """Produce the physical and padded band windows on the packed centroid basis."""
    charge_bispinor = uses_four_spinor_finite_q_charge(
        config.bispinor, config.bispinor_gw)
    _sc_buffer = (int(config.sc.buffer_nbands)
                  if config.qp_solver is QPSolver.SELF_CONSISTENT else 0)
    from common.centroid_basis import PackedCentroidBasis
    mu_basis = PackedCentroidBasis.build(
        centroid_indices, sym, wfn.fft_grid, mesh_xy)
    print0(f"  {mu_basis.describe()}")
    meta = Meta.from_system(wfn, sym,
                            int(config.nval) + _sc_buffer,
                            int(config.ncond) + _sc_buffer, config.nband,
                            n_rmu, charge_bispinor,
                            nband_chi=config.bands.chi,
                            nband_sigma=config.bands.sigma,
                            mesh_xy=mesh_xy, mu_basis=mu_basis)
    if _sc_buffer:
        print0(
            f"  SC buffer: {int(config.nval)}/{int(config.ncond)} named "
            f"valence/conduction window + {_sc_buffer} diagonal state(s) "
            f"at each edge; mode={config.sc.buffer_mode}, "
            f"tail_fit={config.sc.tail_fit}")
    meta.rank = RUNTIME.process_index
    meta.n_proc = RUNTIME.process_count
    meta.sys_dim = config.sys_dim
    meta.bispinor = charge_bispinor
    band_slices = BandSlices.from_band_edges(
        *meta.band_edges, b4_chi=meta.b_id_4_chi,
        b4_sigma=meta.b_id_4_sigma, b4_logical=meta.b_id_4_user)
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
    return (meta, band_slices, zeta_fit_edge)


def _report_sampling_and_bands(
        band_slices, centroid_basis, config, material_class, mesh_xy, meta, mode, print0,
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
        print0(
            f"  Sigma omega layout: sharded; Σ_c(ω,k,m,n) stays "
            f"(m_X, n_Y)-tiled on the {_p_x}x{_p_y} mesh at "
            f"logical/carrier={_sigma_axis.logical}/{_sigma_axis.carrier} "
            f"(consumers read tiles; no full-cube replication).")
    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
        nspinor=meta.nspinor)
    _zeta_ranges = zeta_fit_band_ranges(
        band_slices, zeta_fit_edge, log=lambda *args, **kwargs: None)
    report.environment(config=config, wfn=wfn)
    report.sampling(wfn=wfn, sym=sym, centroids=centroid_basis)
    report.trs_pathways(config=config, sym=sym, material_class=material_class)
    report.bands(
        config=config, wfn=wfn, band_slices=band_slices,
        zeta_ranges=_zeta_ranges)
    return (enk_dft)


def _prepare_isdf_carriers(
        band_slices, bgw_v_grid_fn, centroid_indices, config, material_class, mesh_xy, meta,
        mode, print0, qp_solver, sym, tensors_filename, tmp_dir, wfn):
    """Produce the fitted ISDF operators and wavefunction views."""
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
    green_parent_carrier = getattr(isdf, 'green_parent_carrier', None)
    sigma_parent_carrier = getattr(isdf, 'sigma_parent_carrier', None)
    wfns_sigma = wfns
    wfns_screening = wfns
    oneshot_occupation_state = (
        _oneshot_mpa_occupation_state(
            config, wfn, wfns, material_class, mesh_xy=mesh_xy, print_fn=print0)
        if (qp_solver is not QPSolver.SELF_CONSISTENT
            and mode.value == "mpa") else None)
    if oneshot_occupation_state is not None:
        print0(
            "  one-shot occupations: fixed-N MP1 state, "
            f"mu={oneshot_occupation_state.mu_ry * RYD_TO_EV:.8f} eV, "
            f"width={oneshot_occupation_state.smearing_width_ry:.10f} Ry, "
            f"occ_hash={oneshot_occupation_state.occ_hash}")
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
        qp_solver, wfn, wfns, wfns_sigma):
    """Produce quadrature and direct head-response inputs for one-shot screening."""
    quad, e_ref = None, None
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
    oneshot_head_response = None
    oneshot_head_requests = None
    oneshot_mpa_plan = None
    if (do_screened
            and config.head.correction is HeadCorrection.FULL
            and config.screening.diagrams is ScreeningDiagrams.W_RPA
            and not packed_photon_replaces_charge_sigma(config)
            and qp_solver is not QPSolver.SELF_CONSISTENT):
        from .qsgw_head import build_dft_head_response
        if mode.value == "mpa":
            from .mpa import sample_plan
            from .mpa.model import make_mpa_plan
            oneshot_mpa_plan = make_mpa_plan(
                config, quad, material_class=material_class)
            oneshot_omegas = np.asarray(
                sample_plan.plan_z(oneshot_mpa_plan), dtype=np.complex128)
        else:
            oneshot_head_requests = screening_requests_for(mode, config)
            oneshot_omegas = np.asarray(
                [complex(r.omega_ry) for r in oneshot_head_requests],
                dtype=np.complex128)
        reused_mpa_fit_owns_head = (
            mode is ComputeMode.MPA
            and config.mpa.fit_reuse_file is not None)
        if oneshot_omegas.size and not reused_mpa_fit_owns_head:
            oneshot_head_response = build_dft_head_response(
                wfns_sigma, oneshot_omegas,
                input_dir=input_dir, mesh=mesh_xy,
                wfn=wfn, meta=meta, config=config)
            print0(
                "  head_correction=full: built direct DFT response and "
                "head/body wings on the chi0 transition manifold; finalizing "
                "once against the resident W(Gamma).")
        elif oneshot_omegas.size:
            print0(
                "  MPA fit reuse: deferring to the stored finalized full head; "
                "skipped the redundant direct DFT head/body-wing allocation. "
                "The live fit and sampling plan will still be authenticated.")
    return (quad, e_ref, oneshot_head_response, oneshot_head_requests, oneshot_mpa_plan)


def _report_packed_screening(_dynamic_packed, _screens_current, mode, photon_response, print0, report):
    """Report the resolved packed screening and Gamma completion."""
    print0(
        "  static photon route: "
        + ("full_static_cohsex (sixteen chi blocks, one packed "
           "Dyson solve)" if _screens_current else
           "bare_transverse as the packed path (chi_TT = chi_CT = 0; "
           "scalar Dyson on CC, W_packed = diag(W_00, D_TT))")
        + ("" if not _dynamic_packed else
           "; DYNAMIC packed route: the CC block carries "
           f"compute_mode = {mode.value} at every omega and the "
           "fifteen current blocks are frozen at omega = 0"))
    print0(
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
        tensors_filename, tmp_dir, wfn, wfns_screening, wfns_transverse):
    """Produce the one-shot screening roles and packed response."""
    photon_response = None
    if qp_solver is QPSolver.SELF_CONSISTENT:
        W_by_role = {}
    else:
        with timing.section("gw_jax.screening", announce=True,
                            label="screening (chi0 -> W)"):
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
                _report_packed_screening(_dynamic_packed, _screens_current, mode, photon_response, print0, report)
            else:
                W_by_role = compute_screening_model(
                    mode, wfns_screening, V_q, quad=quad, e_ref=e_ref, sym=sym,
                    centroid_indices=centroid_indices, config=config, meta=meta,
                    mesh_xy=mesh_xy, run_dir=os.path.join(tmp_dir, "mpa"), wfn=wfn,
                    wfn_fingerprint_binding=isdf.wfn_fingerprint_binding,
                    charge_zeta_identity=isdf.charge_zeta_identity,
                    label="oneshot", head_resolver=head_resolver,
                    head_channel=getattr(isdf, 'head_channel', None),
                    mpa_plan=oneshot_mpa_plan,
                    iteration_head_response=oneshot_head_response,
                    occupation_state=oneshot_occupation_state,
                    material_class=material_class,
                    tensors_filename=tensors_filename,
                    print_fn=print0)
        if green_parent_carrier is not None:
            isdf.green_parent_carrier = None
            wfns_screening = None
            green_parent_carrier = None
            gc.collect()
            print0("  Parent-k Green carrier detached from the screening view "
                   "(the Sigma view keeps it).")
    return (W_by_role, photon_response, green_parent_carrier, wfns_screening)


def _install_oneshot_head(
        W_by_role, config, head_resolver, mesh_xy, meta, mode, oneshot_head_requests,
        oneshot_head_response, print0, wfn):
    """Install the finalized one-shot head samples in their resolver."""
    if (oneshot_head_response is not None
            and not packed_photon_replaces_charge_sigma(config)):
        if mode.value == "mpa":
            if W_by_role.get("mpa_fit_reused", False):
                final_head = None
                print0(
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
        print0, qp_solver, sym, tensors_filename):
    """Persist the screened static body and head on the canonical q set."""
    if (not packed_photon_replaces_charge_sigma(config)
            and driver_persists_w0(mode, config)
            and qp_solver is not QPSolver.SELF_CONSISTENT):
        with timing.section("gw_jax.persist_w0"):
            from .gw_output import persist_w0_and_head
            persist_w0_and_head(
                W_by_role.get("static", V_q),
                tensors_filename=tensors_filename, head_resolver=head_resolver,
                config=config, meta=meta, mesh_xy=mesh_xy,
                sym=sym, centroid_indices=centroid_indices,
                print_fn=print0)


def _prepare_static_head(config, do_screened, head_resolver, meta, mode, print0, qp_solver):
    """Produce the static head terms required by the selected QP solver."""
    static_head_terms = None
    if (config.do_G0
            and not packed_photon_replaces_charge_sigma(config)):
        _sc_full = (qp_solver is QPSolver.SELF_CONSISTENT
                    and do_screened
                    and config.head.correction is HeadCorrection.FULL)
        if not _sc_full:
            with timing.section("gw_jax.static_head"):
                static_head_terms = _compute_static_head(
                    head_resolver, meta, do_screened, print0,
                    require_screened=(mode.value != "mpa" and
                                      qp_solver is not QPSolver.SELF_CONSISTENT))
    return (static_head_terms)


def _run_oneshot_sigma(
        V_q, W_by_role, band_slices, bispinor_v_q_path, config, enk_dft, head_resolver,
        input_dir, isdf, material_class, mesh_xy, meta, mode, oneshot_occupation_state,
        photon_response, print0, qp_solver, quad, static_head_terms, sym, wfn, wfns_sigma,
        wfns_transverse):
    """Produce one-shot Sigma and release its consumed screening bodies."""
    gc.collect()   # drop ISDF-stage temporaries before the Σ build
    sigma_result = None
    if qp_solver is not QPSolver.SELF_CONSISTENT:
        with timing.section("gw_jax.sigma"):
            sigma_result = compute_sigma_xc(
                mode,
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
        print0(f"  Bare Σ_X diagonal (eV), k=0: "
               + "  ".join(f"{sig_x_diag[0, i]:.4f}" for i in range(min(8, sig_x_diag.shape[1]))))
        from common import sanity
        sanity.check_finite("Σ_x", sigma_result.sigma_x_kij_ry, print_fn=print0)
        sanity.check_finite("V_H", sigma_result.v_h_kij_ry, print_fn=print0)
        sanity.check_sign("Σ_x diagonal (eV)", sig_x_diag,
                          expect="negative", print_fn=print0)
        sanity.check_in_range("Σ_x diagonal (eV)", sig_x_diag,
                              -200.0, 0.0, unit="eV", print_fn=print0)
    return (sigma_result, W_by_role, photon_response)


def _load_kinetic_ionic_hamiltonian(band_slices, config, mesh_xy, meta, print0, wfn):
    """Produce the authenticated kinetic and ionic band matrix."""
    from file_io import validate_kin_ion_against_run
    _t_kin = time.perf_counter()
    validate_kin_ion_against_run(
        config.paths.kin_ion_file,
        expected_bispinor=config.bispinor,
        expected_bispinor_gw_mode=config.bispinor_gw.value,
        sys_dim=config.sys_dim,
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
    return (kin_ion)


def _solve_qp_stage(
        V_q, band_slices, bispinor_v_q_path, centroid_indices, config, e_ref, enk_dft,
        head_resolver, input_dir, isdf, kin_ion, material_class, mesh_xy, meta, print0,
        qp_solver, quad, report, sigma_result, static_head_terms, sym, tensors_filename,
        wfn, wfns_sigma, wfns_transverse):
    """Produce the QP Hamiltonian and optional self-consistency results."""
    sc_result = None
    rotations_written = False
    final_static_head_terms = static_head_terms
    if qp_solver is QPSolver.SELF_CONSISTENT:
        from .sc_iteration import run_sc_driver
        with timing.section("gw_jax.sc_driver", announce=True,
                            label="self-consistent QSGW driver"):
            sc_result = run_sc_driver(
                wfns_sigma, V_q, kin_ion,
                wfns_transverse=wfns_transverse,
                bispinor_v_q_path=bispinor_v_q_path, mu_bases=isdf.mu_bases,
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
                config=config, meta=meta, mesh_xy=mesh_xy, print_fn=print0)
    eqp2_result = None
    if config.eqp2.enabled:
        from .sc_iteration import run_fixed_sigma_evsc
        with timing.section("gw_jax.eqp2_evsc", announce=True,
                            label="fixed-Sigma eigenvalue self-consistency"):
            eqp2_result = run_fixed_sigma_evsc(
                sigma_result, kin_ion, enk_dft,
                config=config, meta=meta, band_slices=band_slices, wfn=wfn,
                mesh_xy=mesh_xy, print_fn=print0)
    return (enk_dft, eqp2_result, final_static_head_terms, kin_ion, rotations_written, sc_result, sigma_result, sigma_total)


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
    sigma_c_omega       = sigma_result.sigma_c_omega_kij_ry
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
    if not config.no_degen_averaging:
        (sigma_total, sig_sx, sig_coh, sig_h, sig_h_scalar,
         h_transverse, sig_x,
         sigma_c_at_dft_ev) = average_sigma_components(
            sigma_total, sig_sx, sig_coh, sig_h, sig_h_scalar,
            h_transverse, sig_x, sigma_c_at_dft_ev,
            energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
            tol_ry=float(config.degen_avg_tol_ry),
            mesh_xy=mesh_xy)
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
        photon_head_sigma_diag_tskn_ry = average_within_degenerate_sets(
            np.asarray(photon_head_sigma_diag_tskn_ry),
            energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
            tol_ry=float(config.degen_avg_tol_ry))
    from gw.qsgw_utils import static_sigma_diag_to_host
    sig_x_diag_ry = static_sigma_diag_to_host(sig_x, mesh_xy)
    sigma_xc_at_dft_ev = (
        sig_x_diag_ry * RYD_TO_EV
        + sigma_c_at_dft_ev
        if sigma_c_at_dft_ev is not None else None)
    return (e_eval_ev, efermi_dft_ev, final_static_head_terms, h_transverse, head_sigma_diag_w_kn_ry, omega_dft_rel_ev, omega_grid_ev, omega_grid_ry, photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, sig_coh, sig_h, sig_h_scalar, sig_sx, sig_x, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev, sigma_c_omega, sigma_lorentz_skij_ry, sigma_omega_h5_path, sigma_total, sigma_xc_at_dft_ev)


def _diagonalize_qp_hamiltonian(
        band_slices, config, input_dir, kin_ion, print0, qp_solver, sigma_total, wfn):
    """Produce the QP eigensystem and initialize output timing."""
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
        H = 0.5 * ((kin_ion + sigma_total) + jnp.conj(jnp.swapaxes(kin_ion + sigma_total, -1, -2)))
        E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H)
        _sec_eigh.watch(E_full, U_full)
    sanity.refuse_nonfinite(
        "E_qp (eigh of H_QP)", E_full, print_fn=print0,
        detail="LAPACK returns without complaining on a NaN-bearing matrix, "
               "so this is the last place a NaN spectrum can be stopped "
               "before eqp0/eqp1/WFN_qp.h5.")
    _t_out = time.perf_counter()
    if config.debug.write_wfn_h5 and qp_solver is not QPSolver.SELF_CONSISTENT:
        write_qp_wfn_oneshot(
            U_full, E_full, wfn=wfn, band_slices=band_slices,
            input_dir=input_dir, qp_solver=qp_solver, print_fn=print0)
    return (E_full, U_full, _t_out)


def _sigma_diagnostic_fields(
        config, enk_dft, final_static_head_terms, h_transverse, head_sigma_diag_w_kn_ry,
        mesh_xy, mode, omega_dft_rel_ev, omega_grid_ev, photon_head_sigma_diag_tskn_ry,
        sig_coh, sig_h, sig_h_scalar, sig_sx, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev,
        sigma_c_omega, sigma_lorentz_skij_ry, sigma_result, sigma_xc_at_dft_ev):
    """Produce the existing host diagnostic diagonals and head-sector split."""
    from gw.qsgw_utils import static_sigma_diag_to_host
    if sigma_c_omega is not None:
        sigma_c_omega_diag_ev = extract_sigma_diag_logical(
            sigma_c_omega, mesh_xy,
            band_axis=sigma_result.sigma_band_axis) * RYD_TO_EV
        if not config.no_degen_averaging:
            sigma_c_omega_diag_ev = average_within_degenerate_sets(
                sigma_c_omega_diag_ev,
                energies_kn_ry=np.asarray(enk_dft, dtype=np.float64),
                tol_ry=float(config.degen_avg_tol_ry))
            if sigma_c_odd_at_dft_ev is not None:
                sigma_c_odd_at_dft_ev = average_within_degenerate_sets(
                    np.asarray(sigma_c_odd_at_dft_ev),
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
        if mode.is_dynamic and head_sigma_diag_w_kn_ry is not None:
            from .qsgw_utils import (interp_along_omega,
                                     resolve_out_of_range_policy)
            charge_corr = interp_along_omega(
                np.asarray(head_sigma_diag_w_kn_ry),
                np.asarray(omega_grid_ev), np.asarray(omega_dft_rel_ev),
                out_of_range=resolve_out_of_range_policy(),
                context="charge-head Sigma_c at E_DFT",
                print_fn=lambda *args, **kwargs: None)
            charge_head = (charge_corr if charge_head is None
                           else charge_head + charge_corr)
        if charge_head is not None:
            head_sigma_split_skn_ry[0] += charge_head
    sig_sx_diag_ry = static_sigma_diag_to_host(sig_sx, mesh_xy)
    sig_coh_diag_ry = static_sigma_diag_to_host(sig_coh, mesh_xy)
    sig_h_diag_ry = static_sigma_diag_to_host(sig_h, mesh_xy)
    sig_h_scalar_diag_ry = static_sigma_diag_to_host(sig_h_scalar, mesh_xy)
    h_transverse_diag_ry = (
        None if h_transverse is None
        else static_sigma_diag_to_host(h_transverse, mesh_xy))
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
    return (h_transverse_diag_ry, head_sigma_split_skn_ry, omega_rel_ev, sig_coh_diag_ry, sig_h_diag_ry, sig_h_scalar_diag_ry, sig_sx_diag_ry, sigma_c_diag_at_dft_ry, sigma_c_odd_at_dft_ev, sigma_c_omega_diag_ev, sigma_lorentz_diag_skn_ry)


def _assemble_gw_results(
        E_full, U_full, band_slices, config, e_eval_ev, efermi_dft_ev, enk_dft, eqp2_result,
        h_transverse, h_transverse_diag_ry, kin_ion, mode, omega_rel_ev,
        photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, qp_solver, sig_coh,
        sig_coh_diag_ry, sig_h, sig_h_diag_ry, sig_h_scalar, sig_h_scalar_diag_ry, sig_sx,
        sig_sx_diag_ry, sig_x, sig_x_diag_ry, sigma_c_diag_at_dft_ry, sigma_c_odd_at_dft_ev,
        sigma_c_omega_diag_ev, sigma_lorentz_diag_skn_ry, sigma_omega_h5_path, sigma_result,
        sigma_xc_at_dft_ev, tensors_filename):
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
        U_qp=np.array(U_full),
        E_dft_ry=np.array(enk_dft),
        kin_ion_ry=np.array(kin_ion),
        band_start=band_slices.b0,
        band_stop=band_slices.b3,
        use_ppm=mode.is_dynamic,
        self_consistent=qp_solver is QPSolver.SELF_CONSISTENT,
        sigma_kset=sigma_result.kset,
        sigma_c_diag_at_dft_ry=sigma_c_diag_at_dft_ry,
        sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
        sigma_c_omega_diag_ev=sigma_c_omega_diag_ev,
        omega_rel_ev=omega_rel_ev,
        e_eval_ev=e_eval_ev,
        efermi_ev=efermi_dft_ev,
        sigma_omega_h5_path=sigma_omega_h5_path,
        tensors_filename=tensors_filename,
        E_eqp2_ry=(None if eqp2_result is None
                     else np.asarray(eqp2_result.energies_ry)),
        eqp2_iterations=(None if eqp2_result is None
                         else int(eqp2_result.iterations)),
        eqp2_residual_ev=(None if eqp2_result is None
                          else float(eqp2_result.residual_ev)),
        eqp2_tol_ev=(None if eqp2_result is None
                     else float(config.eqp2.tol_ev)),
        photon_head_sigma_diag_tskn_ry=np.asarray(
            photon_head_sigma_diag_tskn_ry),
        photon_head_sigma_basis=photon_head_sigma_basis,
        sigma_lorentz_diag_skn_ry=sigma_lorentz_diag_skn_ry,
        sigma_c_odd_diag_at_dft_ry=(
            None if sigma_c_odd_at_dft_ev is None
            else np.asarray(sigma_c_odd_at_dft_ev) / RYD_TO_EV),
    )
    return (results)


def _write_gw_results(
        _t_out, config, e_eval_ev, final_static_head_terms, head_sigma_diag_w_kn_ry,
        input_dir, meta, omega_dft_rel_ev, omega_grid_ev, omega_grid_ry, print0, qp_solver,
        results, rotations_written, sigma_c_omega, sigma_c_omega_diag_ev, sym, wfn):
    """Write the QP and diagnostic results on the reporting rank."""
    if meta.rank == 0:
        write_freq_debug(
            results, config=config,
            static_head_terms=final_static_head_terms,
            omega_dft_rel_ev=omega_dft_rel_ev,
            head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
            omega_grid_ry=omega_grid_ry,
            sym=sym, file_sym=wfn.symmetry(),
            e_eval_ev=e_eval_ev,
            print_fn=print0,
        )
        write_qsgw_qp_ladders(
            results, config=config,
            e_qp_ry=results.E_qp_ry,
            sigma_c_omega_diag_ev=sigma_c_omega_diag_ev,
            omega_grid_ev=omega_grid_ev,
            sigma_c_omega=sigma_c_omega,
            print_fn=print0,
        )
        write_results(
            results,
            sigma_diag_file=config.paths.sigma_diag_file,
            eqp0_file=config.paths.eqp0_file,
            eqp1_file=config.paths.eqp1_file,
            eqp2_file=config.paths.eqp2_file,
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


def _close_timing(_pre_main, _t_main, meta, print0):
    """Produce the complete process wall time and timing decomposition."""
    if _pre_main is not None:
        _phases = RUNTIME.facts.get("elapsed", {})
        for _phase, _secs in sorted(_phases.items()):
            if _phase != "total":
                timing.record(f"gw_jax.runtime_stack.{_phase}", _secs)
        timing.record("gw_jax.imports",
                      max(_pre_main - _phases.get("total", 0.0), 0.0))
    _wall = time.perf_counter() - _t_main + (_pre_main or 0.0)
    if meta.rank == 0 and debug_print_enabled():
        timing.report(print_fn=print0, title="--- Timing ---", wall=_wall)
    return (_wall)


def _report_final_observables(
        E_full, band_slices, config, enk_dft, eqp2_result, head_sigma_split_skn_ry,
        q0_certificates, report, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev,
        sigma_lorentz_diag_skn_ry, sigma_result):
    """Report the final Sigma coverage, sector summaries and QP gaps."""
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
        _xc_diag_ev = (
            np.sum(np.asarray(sigma_lorentz_diag_skn_ry), axis=0) * RYD_TO_EV
            if sigma_lorentz_diag_skn_ry is not None else
            np.asarray(sig_x_diag_ry) * RYD_TO_EV
            + np.asarray(sigma_c_at_dft_ev))
        _xc_abs = np.abs(_xc_diag_ev)
        _max_share = (np.max(_odd_abs) / np.max(_xc_abs)
                      if np.max(_xc_abs) > 0.0 else np.max(_odd_abs))
        _mean_share = (np.mean(_odd_abs) / np.mean(_xc_abs)
                       if np.mean(_xc_abs) > 0.0 else np.mean(_odd_abs))
        _odd_prefix = (
            "MPA odd Sigma  "
            if getattr(config.compute_mode, "value", config.compute_mode) == "mpa"
            else "GN odd Sigma   ")
        _odd_line = (
            f"{_odd_prefix}: measured-broken-TR ordered residue; "
            f"max|sigC_odd|={np.max(_odd_abs):.6e} eV; "
            f"mean|sigC_odd|={np.mean(_odd_abs):.6e} eV; "
            f"max-share-of-|Sigma_xc|={_max_share:.6e}; "
            f"mean-share-of-|Sigma_xc|={_mean_share:.6e}")
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


def _report_file_rows(args, config, input_dir, report, sigma_omega_h5_path, tensors_filename):
    """Produce the report rows for consumed and generated files."""
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
    return (_file_rows)


def main(argv=None):
	"""Run the GW stages; see docs/architecture/decisions.md."""
	args = build_parser().parse_args(argv)
	_t_main = time.perf_counter()
	_pre_main = timing.process_elapsed_s()
	timing.reset()
	(
	    config, input_dir, qp_solver, mode, report, production_stdout, print0,
	    _config_provenance, do_screened) = _open_production_report(
	    args)
	_report_head_and_photon_policy(config, print0, report)
	mesh_xy = RUNTIME.mesh
	_setup_runtime()
	from file_io.hdf5_owner import probe as _hdf5_probe
	_hdf5_probe("startup", print_fn=print0)
	(
	    config, wfn, material_class, sym, centroid_basis, centroid_indices, n_rmu, tmp_dir,
	    tensors_filename) = _load_system_inputs(
	    config, input_dir, mesh_xy, report, print0, _config_provenance)
	(
	    meta, band_slices, zeta_fit_edge) = _prepare_band_metadata(
	    centroid_indices, config, mesh_xy, n_rmu, print0, sym, wfn)
	(
	    enk_dft) = _report_sampling_and_bands(
	    band_slices, centroid_basis, config, material_class, mesh_xy, meta, mode, print0,
	    report, sym, wfn, zeta_fit_edge)
	q0_certificates = []
	head_resolver = HeadResolver(
		config, input_dir, wfn, sym, meta, print0,
		q0_certificate_fn=q0_certificates.append)
	bgw_v_grid_fn = build_bgw_v_grid_fn(
		config, wfn=wfn, sym=sym, input_dir=input_dir, print_fn=print0)
	timing.record("gw_jax.startup", time.perf_counter() - _t_main)
	(
	    V_qmunu, bispinor_v_q_path, green_parent_carrier, isdf, oneshot_occupation_state, wfns,
	    wfns_screening, wfns_sigma, wfns_transverse) = _prepare_isdf_carriers(
	    band_slices, bgw_v_grid_fn, centroid_indices, config, material_class, mesh_xy, meta,
	    mode, print0, qp_solver, sym, tensors_filename, tmp_dir, wfn)
	V_q = V_qmunu               # flat-q (nq, μ, μ) — compute and restart alike
	(
	    quad, e_ref, oneshot_head_response, oneshot_head_requests, oneshot_mpa_plan) = _prepare_oneshot_response(
	    config, do_screened, input_dir, material_class, mesh_xy, meta, mode, print0, qp_solver,
	    wfn, wfns, wfns_sigma)
	(
	    W_by_role, photon_response, green_parent_carrier, wfns_screening) = _run_oneshot_screening(
	    V_q, bispinor_v_q_path, centroid_indices, config, e_ref, green_parent_carrier,
	    head_resolver, isdf, material_class, mesh_xy, meta, mode, oneshot_head_response,
	    oneshot_mpa_plan, oneshot_occupation_state, print0, qp_solver, quad, report, sym,
	    tensors_filename, tmp_dir, wfn, wfns_screening, wfns_transverse)
	_install_oneshot_head(
	    W_by_role, config, head_resolver, mesh_xy, meta, mode, oneshot_head_requests,
	    oneshot_head_response, print0, wfn)
	_persist_screening(
	    V_q, W_by_role, centroid_indices, config, head_resolver, mesh_xy, meta, mode, print0,
	    qp_solver, sym, tensors_filename)
	(
	    static_head_terms) = _prepare_static_head(
	    config, do_screened, head_resolver, meta, mode, print0, qp_solver)
	(
	    sigma_result, W_by_role, photon_response) = _run_oneshot_sigma(
	    V_q, W_by_role, band_slices, bispinor_v_q_path, config, enk_dft, head_resolver,
	    input_dir, isdf, material_class, mesh_xy, meta, mode, oneshot_occupation_state,
	    photon_response, print0, qp_solver, quad, static_head_terms, sym, wfn, wfns_sigma,
	    wfns_transverse)
	(kin_ion) = _load_kinetic_ionic_hamiltonian(band_slices, config, mesh_xy, meta, print0, wfn)
	(
	    enk_dft, eqp2_result, final_static_head_terms, kin_ion, rotations_written, sc_result,
	    sigma_result, sigma_total) = _solve_qp_stage(
	    V_q, band_slices, bispinor_v_q_path, centroid_indices, config, e_ref, enk_dft,
	    head_resolver, input_dir, isdf, kin_ion, material_class, mesh_xy, meta, print0,
	    qp_solver, quad, report, sigma_result, static_head_terms, sym, tensors_filename, wfn,
	    wfns_sigma, wfns_transverse)
	(
	    e_eval_ev, efermi_dft_ev, final_static_head_terms, h_transverse,
	    head_sigma_diag_w_kn_ry, omega_dft_rel_ev, omega_grid_ev, omega_grid_ry,
	    photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, sig_coh, sig_h, sig_h_scalar,
	    sig_sx, sig_x, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev, sigma_c_omega,
	    sigma_lorentz_skij_ry, sigma_omega_h5_path, sigma_total, sigma_xc_at_dft_ev) = _sigma_output_fields(
	    config, enk_dft, final_static_head_terms, mesh_xy, qp_solver, sc_result, sigma_result,
	    sigma_total)
	(
	    E_full, U_full, _t_out) = _diagonalize_qp_hamiltonian(
	    band_slices, config, input_dir, kin_ion, print0, qp_solver, sigma_total, wfn)
	(
	    h_transverse_diag_ry, head_sigma_split_skn_ry, omega_rel_ev, sig_coh_diag_ry,
	    sig_h_diag_ry, sig_h_scalar_diag_ry, sig_sx_diag_ry, sigma_c_diag_at_dft_ry,
	    sigma_c_odd_at_dft_ev, sigma_c_omega_diag_ev, sigma_lorentz_diag_skn_ry) = _sigma_diagnostic_fields(
	    config, enk_dft, final_static_head_terms, h_transverse, head_sigma_diag_w_kn_ry,
	    mesh_xy, mode, omega_dft_rel_ev, omega_grid_ev, photon_head_sigma_diag_tskn_ry, sig_coh,
	    sig_h, sig_h_scalar, sig_sx, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev, sigma_c_omega,
	    sigma_lorentz_skij_ry, sigma_result, sigma_xc_at_dft_ev)
	(
	    results) = _assemble_gw_results(
	    E_full, U_full, band_slices, config, e_eval_ev, efermi_dft_ev, enk_dft, eqp2_result,
	    h_transverse, h_transverse_diag_ry, kin_ion, mode, omega_rel_ev,
	    photon_head_sigma_basis, photon_head_sigma_diag_tskn_ry, qp_solver, sig_coh,
	    sig_coh_diag_ry, sig_h, sig_h_diag_ry, sig_h_scalar, sig_h_scalar_diag_ry, sig_sx,
	    sig_sx_diag_ry, sig_x, sig_x_diag_ry, sigma_c_diag_at_dft_ry, sigma_c_odd_at_dft_ev,
	    sigma_c_omega_diag_ev, sigma_lorentz_diag_skn_ry, sigma_omega_h5_path, sigma_result,
	    sigma_xc_at_dft_ev, tensors_filename)
	_write_gw_results(
	    _t_out, config, e_eval_ev, final_static_head_terms, head_sigma_diag_w_kn_ry, input_dir,
	    meta, omega_dft_rel_ev, omega_grid_ev, omega_grid_ry, print0, qp_solver, results,
	    rotations_written, sigma_c_omega, sigma_c_omega_diag_ev, sym, wfn)
	(_wall) = _close_timing(_pre_main, _t_main, meta, print0)
	_report_final_observables(
	    E_full, band_slices, config, enk_dft, eqp2_result, head_sigma_split_skn_ry,
	    q0_certificates, report, sig_x_diag_ry, sigma_c_at_dft_ev, sigma_c_odd_at_dft_ev,
	    sigma_lorentz_diag_skn_ry, sigma_result)
	(
	    _file_rows) = _report_file_rows(
	    args, config, input_dir, report, sigma_omega_h5_path, tensors_filename)
	report.timings(timing.records(), wall=_wall)
	report.warnings()
	report.files(_file_rows)
	report.finish()
	production_stdout.close()
	return 0


if __name__ == "__main__":
	from runtime import run_main_and_finalize
	run_main_and_finalize(main)

"""BSE JAX entry points and CLI wrappers."""
from __future__ import annotations

import os

# THE startup call (runtime module docstring) before any jax-collective
# code: env defaults (x64), jax.distributed (the ring matvec uses
# lax.psum/ppermute on the 2D mesh, which is silent-wrong if processes
# don't agree on a shared distributed runtime), CPU fallback, the run's
# clique-warmed square ('x','y') mesh, compile cache, rank-0 report.
# MUST run before this module's own `import jax`.  ``create_mesh_2d()``
# below returns this same startup mesh (plus the BSE-specific
# process_allgather warm-up bse_ring_comm documents).
from runtime import debug_print, initialize_communicator_stack
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import jax

import common.timing as timing
from common.band_degeneracy import DEFAULT_MODE, DEGENERACY_TOL_RY, MODES
from common.log import get_logger, record
from common.preprocessing_output import (ScientificProductionReport,
                                         manual_stage, stage_row)
from common.report_blocks import Setting
from common.scientific_output import policy
from common.units import RYD_TO_EV
from runtime.run_session import RunSession

from .bse_deck import (DeckError, Key, check_dipole_operator, check_fine_grid_keys, deck_params,
                       deck_source, eqp_file, key_of, note_extended_kernel, omega_count,
                       read_settings, refuse_retired_keys, refuse_retired_solver,
                       spin_kernel)
from .bse_preconditioner import bgw_spin_kernel
from .bse_ring_comm import (
    create_mesh_2d,
    create_mesh_xy_from_flags,
    make_bse_shardings,
)
from .bse_stack_matvec import build_bse_stack_matvec_full
from .bse_preconditioner import compute_pair_amplitude
from file_io.restart_bundle import (_find_restart_file)
from .bse_lanczos import simple_lanczos_eig
from .bse_io import write_eigenvectors_stream
from .report_lines import (bse_results_lines, galerkin_rows, locality_rows,
                           window_direct_gap)

log = get_logger(__name__)

__all__ = [
    "build_bse_stack_matvec_full",
    "compute_pair_amplitude",
    "create_mesh_2d",
    "make_bse_shardings",
    "simple_lanczos_eig",
]


#: The report's major-stage table: ``(label, timing sections...)``.
_STAGES = (
    ("restart input", "bse.load"),
    ("BSE eigensolve", "bse.eigensolve"),
    ("exciton dipoles", "bse.exciton_dipoles"),
    ("eigenvector write", "bse.write_eigenvectors"),
    ("Haydock absorption", "bse.haydock"),
)


def _host_replica(array):
    """A replicated or process-local energy table as a host array, read from
    this process's own buffer (no collective); ``None`` for a sharded one."""
    import numpy as np
    try:
        if getattr(array, "is_fully_addressable", True):
            return np.asarray(jax.device_get(array))
        if array.sharding.is_fully_replicated:
            return np.asarray(array.addressable_data(0))
    except Exception:                                    # noqa: BLE001 — report only
        return None
    return None


def _window_gap(data, n_val: int, n_cond: int):
    """``(E_gap Ry, k 1-based)`` of the solved window, for RESULTS only."""
    eps_v, eps_c = _host_replica(data["eps_v"]), _host_replica(data["eps_c"])
    if eps_v is None or eps_c is None:
        return None
    return window_direct_gap(eps_v, eps_c, n_val=n_val, n_cond=n_cond)


def _preview_lanczos(
    input_file: str,
    n_val: int,
    n_cond: int,
    n_eig: int = 5,
    write_eigs: int | None = None,
    dipole_file: str | None = None,
    dipole_position_fix: bool = False,
    helicity_table: bool = False,
    max_lanczos_iter: int | None = None,
    include_W: bool = True,
    eqp_file: str | None = None,
    qp_rotations_file: str | None = None,
    n_occ: int | None = None,
    davidson_m_max: int | None = None,
    davidson_precond: str = "exact",
    davidson_tol_mev: float = 0.01,
    davidson_margin: int = 8,
    davidson_margin_tol_mev: float | None = None,
    davidson_olsen: bool = False,
    davidson_eps_shift: float | None = None,
    tda: bool = True,
    degeneracy_mode: str = DEFAULT_MODE,
    degeneracy_tol_ry: float = DEGENERACY_TOL_RY,
    haydock: dict | None = None,
    spectrum: dict | None = None,
    cell_volume: float | None = None,
    kgrid: tuple | None = None,
    report=None,
    full_bse_solver: str = "dense",
    full_bse_instability: str = "refuse",
) -> dict:
    full_davidson = not tda and full_bse_solver == "davidson"
    if not tda and not full_davidson and kgrid is not None:
        # GATE nontda_dense_size before the restart load and the W transform.
        from math import prod
        from .bse_densify import _resolve_kgrid_fine
        from .bse_nontda import refuse_dense_window
        refuse_dense_window(n_cond, n_val, prod(
            _resolve_kgrid_fine(None, input_file) or kgrid), create_mesh_2d())
    restart_file = _find_restart_file(input_file)
    # ONE solve path on any mesh (1x1 included): the sharded loader and
    # ``solve_bse_sharded`` on the trial-stack matvec.  The single-device
    # ``solve_bse`` / ``bse_serial`` twin was deleted with C10 (2026-09-24).
    _htransform_rank_records = []
    _htransform_quality_records = []

    # Sharded ring matvec — parallelises (μ,ν) and avoids per-iter
    # 3D-FFT of W_q (precomputes W_R once outside the Lanczos loop).
    from .bse_io import load_bse_data_from_restart_sharded
    from .bse_lanczos import solve_bse_sharded
    mesh_xy = create_mesh_2d()
    # n_occ-aware band split: load_bse_data_from_restart_sharded
    # auto-detects valence by ``mean_enk < fermi_energy``; user-given
    # n_occ replaces that detection identically to _load_ring_subset.
    # We pass fermi_energy=0.0 (default) and rely on enk_full's reference.
    n_stages = 2 + (write_eigs is not None) + (haydock is not None)
    with manual_stage(1, n_stages, "BSE window from the restart",
                      name="bse.load") as stage:
        data = load_bse_data_from_restart_sharded(
            restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
            input_file=input_file, n_occ=n_occ,
            degeneracy_mode=degeneracy_mode,
            degeneracy_tol_ry=degeneracy_tol_ry,
            htransform_rank_record_fn=_htransform_rank_records.append,
            htransform_quality_record_fn=(
                _htransform_quality_records.append),
            eqp_file=eqp_file,
            qp_rotations_file=qp_rotations_file,
            velocity_file=dipole_file,
            velocity_position_fix=dipole_position_fix,
            q0_head="diagonal",       # operands via bse_head.head_shifted_eps_c
            need_W=include_W,         # local_fields: no W read, rebuilt or kept
            coupling_legs=not tda,    # TDA: psi_c_Y / psi_v_X are never read
        )
        qp_rotation = data.get("qp_rotation")
        grid_x, grid_y = mesh_xy.devices.shape
        # EQP override on enk_full (BGW eqp1.dat semantics).  A kgrid_fine
        # bundle took the QP energies inside its htransform; re-slicing the
        # coarse restart here would put coarse ε on the fine k axis.
        n_occ_eqp = None
        if eqp_file is not None and not data["densified"]:
            # Re-slice the band window on the eqp-corrected energies. Uses the
            # loader-CLAMPED band counts (data['n_val']/data['n_cond']), not the
            # raw CLI n_val/n_cond, so an over-request can't slice out of bounds.
            from .bse_io import apply_eqp_and_reslice_bands
            data["eps_v"], data["eps_c"], n_occ_eqp = apply_eqp_and_reslice_bands(
                restart_file, eqp_file, input_file,
                int(data["n_val"]), int(data["n_cond"]), n_occ, grid_x,
                grid_y, degeneracy_mode=degeneracy_mode,
                degeneracy_tol_ry=degeneracy_tol_ry)
        # The BSE-grid htransform (kgrid_fine) and QP-rotation receipts: one
        # row each here, the whole receipt in the JSON sidecar.
        from common.scientific_output import (htransform_quality_lines,
                                              spectral_compression_lines)
        for receipt in _htransform_rank_records:
            log.debug("BSE-grid htransform spectral compression", extra=record(
                "bse.htransform.spectral_compression",
                lines=spectral_compression_lines(receipt)))
            for label, value in galerkin_rows(receipt):
                stage_row(label, value)
        for receipt in _htransform_quality_records:
            log.debug("BSE-grid htransform interpolation quality", extra=record(
                "bse.htransform.quality", lines=htransform_quality_lines(receipt)))
            for label, value in locality_rows(receipt):
                stage_row(label, value)
        if qp_rotation is not None:
            from .bse_qp_rotation import report_lines, summary_text
            log.debug("QP-rotated basis", extra=record(
                "bse.qp_rotation", lines=list(report_lines(qp_rotation))))
            stage_row("QP basis", summary_text(qp_rotation))
        nkx = data["nkx"]; nky = data["nky"]; nkz = data["nkz"]
        nk = nkx * nky * nkz
        stage_row("bands", f"{int(data['n_val'])} valence below the occupied "
                           f"boundary, {int(data['n_cond'])} conduction above it")
        stage_row("transitions", f"{int(data['n_val']) * int(data['n_cond']) * nk} "
                                 f"({int(data['n_val'])} x {int(data['n_cond'])} x "
                                 f"{nk} k); {int(data['n_cond_pad']) * int(data['n_val_pad']) * nk} "
                                 "padded")
        stage.result(f"psi, W and V on {int(data['n_rmu'])} centroids, "
                     f"{nkx} x {nky} x {nkz} k")
    nkx = data["nkx"]; nky = data["nky"]; nkz = data["nkz"]
    nk = nkx * nky * nkz
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    # The window this run is ACTUALLY solving, read off the loader — which
    # clamped the request to what the file holds and then, under an
    # explicit ``band_degeneracy = snap``, may have widened it outward past
    # a cut multiplet.  Everything downstream that needs to name
    # bands uses these, never the ``n_val``/``n_cond`` arguments: those are
    # the request, and the request is stale the moment the guard fires.
    n_val_eff = int(data["n_val"])
    n_cond_eff = int(data["n_cond"])
    bse_dim = nc_pad * nv_pad * nk
    log.info(f"BSE problem (sharded {grid_x}x{grid_y}): "
             f"{nc_pad} cond × {nv_pad} val × {nk} k = {bse_dim} dim")
    if max_lanczos_iter is None:
        max_lanczos_iter = max(30, min(200, bse_dim // 2))
    # ``max_lanczos_iter`` is the iteration cap of the Davidson (either one).
    block_max_iter = max(1, max_lanczos_iter)
    if full_davidson:
        log.info(f"Full BSE: structure-preserving Davidson on the {bse_dim}-state window "
                 f"(paired trial vectors over the pair applier A X + B conj(X), no dense "
                 f"assembly): {int(n_eig)} roots + up to {int(davidson_margin)} margin, "
                 f"≤ {block_max_iter} iterations, residual stop {float(davidson_tol_mev):g} meV, "
                 f"{davidson_precond} diagonal preconditioner; an unstable reference state: "
                 f"{full_bse_instability}")
    elif not tda:
        log.info(f"Full BSE: dense structure-preserving solve of the {bse_dim}-state "
                 f"window (A and B assembled, host eigh of K z = omega Sigma z); "
                 f"the Davidson settings are not read")
    else:
        log.info(f"Davidson: ≤ {block_max_iter} iterations")
    # ONE row for the eigensolve — compile + every iteration.  It
    # is the number worth comparing across decks, and the only honest way
    # to say "the BSE solve cost X" (the matvec itself runs many times
    # and must NOT be timed per call).
    with manual_stage(2, n_stages, "BSE eigensolve",
                      name="bse.eigensolve") as _sec:
        if tda:
            stage_row("solver", f"davidson, {int(n_eig)} roots, at most "
                                f"{block_max_iter} iterations")
        elif full_davidson:
            stage_row("solver", f"structure-preserving Davidson for the full BSE, {int(n_eig)} "
                                f"roots + up to {int(davidson_margin)} margin, at most "
                                f"{block_max_iter} iterations")
        else:
            stage_row("solver", f"dense structure-preserving full BSE, "
                                f"{int(n_eig)} of {int(bse_dim)} roots")
        eigenvalues, eigenvectors, _ = solve_bse_sharded(
            data, mesh_xy, n_eig=n_eig, max_iter=block_max_iter,
            include_W=include_W, tda=tda,
            davidson_m_max=davidson_m_max,
            davidson_eps_shift_Ry=(1e-3 if davidson_eps_shift is None
                                   else davidson_eps_shift),
            davidson_precond=davidson_precond,
            davidson_olsen=davidson_olsen,
            davidson_tol_Ry=float(davidson_tol_mev) / (RYD_TO_EV * 1.0e3),
            davidson_margin=int(davidson_margin),
            davidson_margin_tol_Ry=(None if davidson_margin_tol_mev is None else
                                    float(davidson_margin_tol_mev) / (RYD_TO_EV * 1.0e3)),
            keep_W_R=haydock is not None,
            full_solver=full_bse_solver, full_on_indefinite=full_bse_instability,
        )
        _sec.watch(eigenvalues, eigenvectors)
        _sec.result(f"{int(n_eig)} roots of the {int(bse_dim)}-state "
                    + ("TDA" if tda else "full") + " BSE")
    stopped = None
    if tda:
        stopped = _davidson_report(n_eig=n_eig, tol_mev=float(davidson_tol_mev),
                                   budget=block_max_iter, dim=bse_dim)
    stability_rows = ()
    if full_davidson:
        stopped, stability_rows = _full_davidson_report()
    ryd2ev = 13.6056980659
    log.info(f"Lowest {n_eig} eigenvalues (Ry): {eigenvalues}")
    log.info(f"Lowest {n_eig} eigenvalues (eV): {eigenvalues * ryd2ev}")

    eigenvalues_host = jax.device_get(eigenvalues).real
    helicity_lines = ()
    dipoles = None

    d_alpha = de_op = None
    if dipole_file is not None:
        from file_io.restart_bundle import read_metadata
        from .absorption_common import window_dipoles, window_transition_energies
        from .bse_io import resolve_n_occ
        from .bse_window import _parse_wfn_path
        n_occ_dip = (n_occ_eqp if n_occ_eqp is not None else resolve_n_occ(
            read_metadata(restart_file)["energies"], n_occ=n_occ,
            input_file=input_file))
        with timing.section("bse.exciton_dipoles"):
            d_alpha, de_cv = window_dipoles(
                data, dipole_file, n_occ=n_occ_dip,
                wfn_path=_parse_wfn_path(input_file))
        # The noeh spectra sit at the operator's own (QP) transitions, not at
        # the velocity's (DFT) energies that divide the dipoles.
        de_op = window_transition_energies(data)
        if de_op.shape != tuple(de_cv.shape):
            raise ValueError(f"window transitions {de_op.shape} vs dipoles {tuple(de_cv.shape)}")

    n_write = 0
    dipole_attrs = None
    if dipole_file is not None:
        # ⟨0|r̂_α|S⟩ per root (every root, for eigenvalues_b*.dat and
        # the brightest in RESULTS; eigenvectors.h5 keeps the written
        # states'): the window dipoles Haydock seeds
        # from (``absorption_common.window_dipoles``: the dipole.h5 slice
        # at the n_occ the loader and ``eqp_corrections`` settled on, a kgrid_fine
        # bundle's own, or a qp_rotations_file bundle's QP-basis dipoles),
        # contracted against the solve's distributed eigenvectors in state
        # blocks (no gather).
        from .absorption_common import (build_dipole_vector_bse,
                                        exciton_dipoles_distributed)
        with timing.section("bse.exciton_dipoles", announce=True):
            if data["densified"]:
                dipole_attrs = {
                    "route": data["dipoles"]["route"],
                    "source": data["dipoles"]["source"],
                    "velocity": data["dipoles"]["velocity"],
                    "on_grid_rel": data["dipoles"]["on_grid_rel"]}
            dipoles = exciton_dipoles_distributed(
                eigenvectors, build_dipole_vector_bse(
                    d_alpha, n_cond_pad=nc_pad, n_val_pad=nv_pad),
                int(n_eig))
        stage_row("dipoles", f"<0|r|S> for {int(dipoles.shape[0])} states "
                             f"from {os.path.basename(str(dipole_file))}")
        if helicity_table and report is not None:
            # Report-only: the per-state σ± strengths of the dipoles just
            # stored, under the one convention ``inplane_helicity`` owns.
            # Printed in RESULTS.
            from .absorption_common import helicity_table_lines
            helicity_lines = tuple(helicity_table_lines(
                eigenvalues_host[:dipoles.shape[0]] * ryd2ev, dipoles, nk))
    if write_eigs is not None:
        with manual_stage(3, n_stages, "Exciton dipoles and eigenvectors",
                          name="bse.output") as _out:
            n_write = n_eig if write_eigs < 0 else min(write_eigs, n_eig)
            from .bse_window import _parse_wfn_path
            with timing.section("bse.write_eigenvectors", announce=True):
                write_eigenvectors_stream(
                    "eigenvectors.h5",
                    eigenvalues,
                    eigenvectors,
                    n_val_eff,
                    n_cond_eff,
                    nkx,
                    nky,
                    nkz,
                    n_write,
                    use_tda=tda,
                    dipoles=None if dipoles is None else dipoles[:n_write],
                    qp_rotations_file=(None if qp_rotation is None
                                       else qp_rotation["path"]),
                    dipole_attrs=dipole_attrs,
                    k_shift=tuple(data.get("k_shift", (0.0, 0.0, 0.0))),
                    spin_kernel=bgw_spin_kernel(int(data["psi_c_X"].shape[2]), include_W,
                                               channel=data.get("exchange_channel", "singlet")),
                    wfn_path=_parse_wfn_path(input_file),
                    wfn_grid=not data["densified"],
                )
            _out.result(f"eigenvectors.h5: {int(n_write)} states"
                        + (" with <0|r|S>" if dipoles is not None else ""))

    # Every root, in BerkeleyGW's own files (with its dipoles when computed).
    energies_ev = eigenvalues_host[:n_eig] * ryd2ev
    noeh = None
    if d_alpha is not None:
        from .absorption_common import window_band_energies
        ec_ry, ev_ry = window_band_energies(data)
        noeh = (ec_ry * ryd2ev, ev_ry * ryd2ev, d_alpha)
    roots_files = _write_all_roots(
        input_file, energies_ev, dipoles, nk, noeh=noeh,
        polarization=(spectrum or haydock or {}).get("polarization"),
        triplet=bgw_spin_kernel(int(data["psi_c_X"].shape[2]), include_W,
                               channel=data.get("exchange_channel", "singlet")) == 0)
    spectrum_files = []
    if dipoles is not None and spectrum is not None:
        # BerkeleyGW's absorption_eh.dat / absorption_noeh.dat from the solved
        # states and the window transitions (complete below the highest root).
        from .absorption_common import polarization_directions, write_state_spectra
        from .bse_window import _parse_wfn_path
        directions, labels = polarization_directions(
            spectrum["polarization"], _parse_wfn_path(input_file))
        spectrum_files = write_state_spectra(
            energies_Ry=eigenvalues_host[:n_eig], exciton_dipoles=dipoles,
            d_alpha=d_alpha, de_cv=de_op, directions=directions, labels=labels,
            V_cell=float(cell_volume), nk=nk, n_spinor=int(data["psi_c_X"].shape[2]),
            width_eV=spectrum["width_ev"], kind=spectrum["kind"],
            omegas_eV=spectrum["omegas_ev"], no_eps1=spectrum["no_eps1"],
            prefix=spectrum["prefix"])
        log.info(f"absorption spectra from {int(n_eig)} states: {', '.join(spectrum_files)}")
    if haydock is not None:
        # ε₂(ω) by the Haydock recursion on THIS payload: the loaded bundle,
        # the eigensolve's W_R (keep_W_R) and the same matvec, so the second
        # process's startup, load (htransform + densify on kgrid_fine), W
        # transform and compiles are not paid again.
        from .absorption_common import (build_dipole_vector_bse,
                                        polarization_directions, project_dipoles)
        from .absorption_haydock import haydock_coefficients, write_haydock_outputs
        from .bse_window import _parse_wfn_path
        directions, labels = polarization_directions(
            haydock["polarization"], _parse_wfn_path(input_file))
        with manual_stage(n_stages, n_stages, "Haydock absorption",
                          name="bse.haydock") as _hay:
            stage_row("recursion", f"{len(labels)} block(s) "
                                   f"({' '.join(labels)}) x "
                                   f"{int(haydock['n_iter'])} steps on the "
                                   f"eigensolve's payload")
            d_block = project_dipoles(build_dipole_vector_bse(
                d_alpha, n_cond_pad=nc_pad, n_val_pad=nv_pad), directions)
            alphas, betas, norms = haydock_coefficients(
                data, mesh_xy, d_block, int(haydock["n_iter"]))
            write_haydock_outputs(
                out_prefix=haydock["out_prefix"], directions=directions,
                labels=labels, alphas=alphas,
                betas=betas, norms=norms, d_alpha=d_alpha, de_cv=de_op,
                V_cell=float(cell_volume), nk=nk, eta_eV=haydock["eta_ev"],
                n_spin=int(haydock["n_spin"]),
                n_spinor=int(data["psi_c_X"].shape[2]),
                omega_min_eV=haydock["omega_min_ev"],
                omega_max_eV=haydock["omega_max_ev"],
                n_omega=int(haydock["n_omega"]), no_eps1=haydock["no_eps1"],
                n_iter=int(haydock["n_iter"]),
                metadata=({"dipole_route": data["dipoles"]["route"],
                           "dipole_source": data["dipoles"]["source"],
                           "dipole_on_grid_rel": data["dipoles"]["on_grid_rel"]}
                          if data["densified"] else None))
            _hay.result(f"{haydock['out_prefix']}.h5, eta "
                        f"{haydock['eta_ev']:.4f} eV")

    if report is not None:
        # RESULTS: the exciton table, the window's lowest direct transition
        # and the binding of S1 against it, the oscillator strengths when
        # computed.  The harness reads every S row on the rank-0 stdout, so
        # stdout keeps the untrimmed table; the main log lists at most 20.
        gap = _window_gap(data, n_val_eff, n_cond_eff)
        kind = "QP" if (eqp_file is not None or qp_rotation is not None) else "DFT"

        def results(trim):
            return bse_results_lines(
                energies_ev,
                gap=None if gap is None else (gap[0] * ryd2ev, gap[1], kind),
                dipoles=dipoles, n_k=nk, helicity=helicity_lines,
                eigenvectors=(None if write_eigs is None else
                              f"eigenvectors.h5, {int(n_write)} states"
                              + (" with <0|r|S>" if dipoles is not None else "")),
                roots_file=_files_text(roots_files), trim=trim,
                extra_rows=(() if stopped is None else (("eigensolve", stopped),))
                + tuple(stability_rows))
        _emit_results(report, results(True), results(False))

    return {
        "n_val": int(n_val_eff),
        "n_cond": int(n_cond_eff),
        "nk": int(nk),
        "bse_dim": int(bse_dim),
        "restart_file": str(restart_file),
        "eigenvector_file": "eigenvectors.h5" if write_eigs is not None else None,
        "haydock_file": (None if haydock is None
                         else f"{haydock['out_prefix']}.h5"),
        "roots_files": tuple(roots_files),
        "spectrum_files": tuple(spectrum_files),
    }


def _davidson_report(*, n_eig, tol_mev, budget, dim):
    """The Davidson stop as one log line and the RESULTS ``eigensolve`` row;
    any stop but CONVERGED refuses (``GATE davidson_converged``,
    ``bse_lanczos.davidson_stop_line``) before the driver writes anything."""
    from .bse_lanczos import davidson_stop_line
    return davidson_stop_line(
        n_eig=n_eig, tol_mev=tol_mev, budget=budget,
        cap=(f"the cap is solver_max_iter, default max(30, min(200, dim // 2)) = "
             f"{max(30, min(200, int(dim) // 2))} at dim {int(dim)}"),
        refused="No eigenvector, spectrum or Haydock output was written.",
        fix="raise solver_max_iter or davidson_m_max, or loosen davidson_tol_mev")


def _full_davidson_report():
    """The full-BSE Davidson's stop line (the RESULTS ``eigensolve`` row) and
    its ``stability`` rows, from ``bse_nontda.LAST_DAVIDSON``.  The solve has
    already refused any stop but CONVERGED (``GATE nontda_davidson_converged``)
    and, unless ``full_bse_instability = continue``, an indefinite stability
    matrix (``GATE sp_davidson_definite``).  The rows give ``lam_min`` of
    ``K = [[A, B], [B*, A*]]`` in meV and every unstable (complex) or
    non-positive root the solve found; those are not among the roots written."""
    import numpy as np
    from .bse_nontda import LAST_DAVIDSON as run
    ry_mev = RYD_TO_EV * 1.0e3
    lam, n_special = float(run["lam_min_ry"]) * ry_mev, int(run["n_special"])
    rows = [("stability", f"lam_min(K) = {lam:.6g} meV: " + (
        "K = [[A, B], [B*, A*]] is positive definite on the solved space" if n_special == 0
        and run["info"]["indefinite_at"] is None else
        "K = [[A, B], [B*, A*]] is NOT positive definite, the reference state is unstable"))]
    for j, w in enumerate(np.asarray(run["omega_ry"])[:n_special]):
        w = complex(w) * ry_mev
        rows.append((f"unstable root {j + 1}", f"omega = {w.real:+.6f} {w.imag:+.6f} i meV; "
                     "not written to the eigenvalue or eigenvector files"))
    return run["line"], tuple(rows)


def _files_text(paths) -> str:
    """``eigenvalues.dat`` or ``eigenvalues_b1.dat .. _b3.dat``."""
    names = [os.path.basename(str(p)) for p in paths]
    return names[0] if len(names) == 1 else f"{names[0]} .. {names[-1]}"


def _write_all_roots(input_file, energies_ev, dipoles, nk: int, *, noeh=None,
                     polarization=None, triplet=False) -> list[str]:
    """Every root to BerkeleyGW's ``eigenvalues.dat`` (energies, or the dipole
    along a deck ``polarization``) or, with dipoles and no ``polarization``,
    its three-axis ``eigenvalues_b1.dat`` .. ``_b3.dat``
    (``absorption_common.write_bgw_eigenvalues``), rank 0 writing.  The
    header's ``vol`` is ``N_k`` times the deck WFN's cell volume and its
    ``nspin, nspinor`` that WFN's.  ``noeh`` ``(ec_eV, ev_eV, d_alpha)`` also
    writes the window transitions to ``eigenvalues*_noeh.dat``
    (``absorption_common.write_bgw_eigenvalues_noeh``); those paths are not
    returned.  Returns the paths (on every rank)."""
    from .absorption_common import (eigenvalue_file_names, polarization_directions,
                                    write_bgw_eigenvalues, write_bgw_eigenvalues_noeh)
    from .bse_window import _parse_wfn_path
    wfn_path = _parse_wfn_path(input_file)
    directions, labels = polarization_directions(polarization, wfn_path)
    import h5py
    with h5py.File(wfn_path, "r") as f:
        celvol = float(f["mf_header/crystal/celvol"][()])
        n_spin = int(f["mf_header/kpoints/nspin"][()])
        n_spinor = int(f["mf_header/kpoints/nspinor"][()])
    if noeh is not None:
        write_bgw_eigenvalues_noeh(".", *noeh, n_spin=n_spin, n_spinor=n_spinor,
                                   vol_supercell=float(nk) * celvol,
                                   directions=directions, labels=labels)
    if jax.process_index() == 0:
        return write_bgw_eigenvalues(".", energies_ev, dipoles, n_spin=n_spin,
                                     n_spinor=n_spinor,
                                     vol_supercell=float(nk) * celvol,
                                     directions=directions, labels=labels,
                                     triplet=triplet)
    return (["eigenvalues.dat"] if dipoles is None else
            eigenvalue_file_names(labels))


def _emit_results(report, log_lines, stdout_lines) -> None:
    """RESULTS: ``log_lines`` to the main log, ``stdout_lines`` to the live
    stdout (one block when they agree, as they do up to 20 roots)."""
    if list(log_lines) == list(stdout_lines):
        report.emit_lines(log_lines)
        return
    stdout = report.stdout
    report.stdout = lambda _line: None
    try:
        report.emit_lines(log_lines)
    finally:
        report.stdout = stdout
    if report.rank == 0:
        for line in stdout_lines:
            stdout(line)



def build_parser():
    """The BSE command line: the deck, the device mesh and the output names.

    Every physics setting is a deck key (``BSE_KEYS``, BerkeleyGW's
    absorption.x names); an old physics flag refuses before this parser runs
    (``runtime.renames.CLI_FLAGS``)."""
    import argparse

    parser = argparse.ArgumentParser(
        allow_abbrev=False, description="BSE eigensolve; settings are deck keys "
        "(BerkeleyGW absorption.x names) in the deck passed with -i")
    parser.add_argument("-i", "--input", help="the GW deck, which also carries the BSE keys")
    parser.add_argument("--px", type=int, default=None,
                        help="mesh rows; default = the run's square startup "
                             "mesh")
    parser.add_argument("--py", type=int, default=None,
                        help="mesh columns; must equal --px, and px*py must "
                             "be the job's device count")
    parser.add_argument("--report-file", default=None,
                        help="human-readable run report (default: bse.out)")
    parser.add_argument("--out-prefix", default=None,
                        help="prefix of the absorption files (default: absorption)")
    return parser


def _settings_parser():
    """The BSE settings and their defaults, one attribute per deck key of
    ``BSE_KEYS`` (never parsed from a command line).  ``band_degeneracy``
    takes its choices from ``common.band_degeneracy``."""
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--band-degeneracy", choices=MODES,
                        default=DEFAULT_MODE,
                        help="what to do when --n-val/--n-cond cut a "
                             "degenerate multiplet: 'strict' (the default "
                             "since 2026-08-10) refuses and names the counts "
                             "that would work; 'snap' widens the window "
                             "OUTWARD to the multiplet boundary and says so; "
                             "'off' proceeds on the cut multiplet.  A widened "
                             "window is a different calculation, so widening "
                             "is opt-in.  Same guard, same defaults as "
                             "bse.exciton_bands.")
    parser.add_argument("--degeneracy-tol-ry", type=float,
                        default=DEGENERACY_TOL_RY, dest="degeneracy_tol_ry",
                        help="'same multiplet' tolerance in Ry (default 1 meV)")
    parser.add_argument("--n-eig", type=int, default=5)
    parser.add_argument("--tda", action="store_true",
                        help="Use Tamm-Dancoff approximation (TDA). Default is full non-TDA.")
    parser.add_argument("--rpa", action="store_true",
                        help="Use RPA kernel (D+V only), skip W0 term entirely.")
    parser.add_argument("--bse", action="store_true",
                        help="Use full BSE kernel (D+V-W). Overrides default RPA.")
    parser.add_argument("--kpm-window-count", type=int, default=4,
                        help="Number of DOS-weighted windows the KPM route reports.")
    parser.add_argument(
        "--write-eigs",
        nargs="?",
        const=-1,
        type=int,
        help="Write eigenvectors.h5 (optional N, default: n-eig).",
    )
    parser.add_argument(
        "--dipole", default=None, metavar="DIPOLE_H5",
        help="contract each computed state against this dipole.h5 "
             "(psp.get_dipole_mtxels, the file absorption_haydock reads): "
             "the TDA's X, or both halves of the full BSE's (X, Y), "
             "sum_t X_t d_t* - Y_t d_t.  <0|r|S> goes to the eigenvalue "
             "files, the absorption spectra and, with --write-eigs, "
             "exciton_data/dipoles in eigenvectors.h5.  Under kgrid_fine it "
             "is the deck (coarse) WFN's dipole.h5: the densifier carries it "
             "through the htransform into the fine states")
    parser.add_argument(
        "--dipole-position-fix", action="store_true",
        help="kgrid_fine with --eqp: the dipoles are the exact <c|r|v> of the "
             "fine QP states, from the DFT f(H) eigenbasis of the window "
             "(bse_densify.densified_dipoles qp_states), instead of "
             "v/dE_RQ.  Opt in when the logged QP/DFT window mixing is large "
             "(strongly band-dependent QP shifts, CrI3-type)")
    parser.add_argument(
        "--helicity-table", action="store_true",
        help="with --dipole: add each written state's in-plane circular "
             "strengths to the report, I+- = |D_x -+ i D_y|^2/(2 N_k) per unit "
             "cell for D = <0|r|S>, and eta = (I+ - I-)/(I+ + I-); sigma+ is the "
             "field E0 (cos wt, sin wt, 0), x -> y, counter-clockwise seen "
             "from +z (absorption_common.inplane_helicity)")
    parser.add_argument(
        "--max-lanczos-iter",
        type=int,
        default=None,
        help="Iteration budget (default: auto-scale with problem size, "
             "max(30, min(200, bse_dim // 2))).  The CAP on Davidson "
             "iterations; the solve stops at its residual test "
             "(--davidson-tol-mev).",
    )
    parser.add_argument(
        "--solver",
        choices=("davidson",),
        default="davidson",
        help="Eigensolver for the lowest --n-eig roots: ``davidson``, block "
             "Davidson with the exact diagonal preconditioner, stopped when "
             "every root's residual is below --davidson-tol-mev.  The only "
             "Tamm-Dancoff eigensolver since 2026-11-27.",
    )
    parser.add_argument(
        "--davidson-precond", choices=("bare", "exact", "auto"),
        default="exact",
        help="Davidson preconditioner diagonal. ``exact`` (default) divides by "
             "the assembled diag(H) = dE + (V_x - W_d)/nk (dE + V_x/nk under "
             "--rpa), which is what BerkeleyGW hands PRIMME; it is assembled "
             "once per payload and the per-iteration cost is identical to the "
             "bare route. ``bare`` divides by the transition energy "
             "dE = E_c - E_v. ``auto`` chooses on the transition-space "
             "dimension at bse_davidson_helpers.EXACT_PRECOND_AUTO_MIN_DIM.",
    )
    parser.add_argument(
        "--davidson-tol-mev", type=float, default=0.01,
        help="Davidson stopping rule, meV: every root's residual norm "
             "||H x - E x|| below it (relative above |E| = 1 Ry).  A residual "
             "r puts an exact eigenvalue within ||r|| of E, so this bounds "
             "each root's eigenvalue error; the error is typically "
             "||r||^2/gap, far below it.  Default 0.01 meV.",
    )
    parser.add_argument(
        "--davidson-margin", type=int, default=8,
        help="Davidson: extra roots solved above --n-eig (at most half the "
             "transition space) and not reported; the stop reads the lowest "
             "--n-eig.  Default 8.",
    )
    parser.add_argument(
        "--davidson-margin-tol-mev", type=float, default=None,
        help="Davidson: residual (meV) at which a margin root locks: it stays "
             "in the basis and gets no further correction.  Default: "
             "--davidson-tol-mev.",
    )
    parser.add_argument(
        "--davidson-olsen", action="store_true",
        help="Davidson: Olsen-correct the preconditioned direction (project "
             "out its component along the current Ritz vector). Two batched "
             "inner products per iteration. This is what makes a SMALL "
             "--davidson-eps-shift safe.",
    )
    parser.add_argument(
        "--davidson-eps-shift", type=float, default=None,
        help="Davidson: preconditioner regularisation in Ry. Default 1e-3 "
             "(= 13.6 meV).",
    )
    parser.add_argument(
        "--davidson-m-max", type=int, default=None,
        help="Davidson: max subspace dimension before restart. Default "
             "10*n_eig (measured knee). Memory is 2*m_max trial vectors, "
             "because Davidson stores V and HV.",
    )
    parser.add_argument(
        "--full-bse-solver", choices=("dense", "davidson"), default="dense",
        help="Eigensolver of the full BSE (full_bse = true).  ``dense`` (the "
             "default): A and B assembled, host eigensolve, N <= 4096.  "
             "``davidson``: the structure-preserving block Davidson over the pair "
             "applier A X + B conj(X) (solvers.bse_sp_davidson), no assembly and no "
             "size limit, stopped when every root's residual is below "
             "--davidson-tol-mev.  It reads solver_max_iter, davidson_tol_mev, "
             "davidson_margin, davidson_margin_tol_mev, davidson_m_max (basis "
             "capacity in real-form vectors, default 20 n_eig; storage twice that), "
             "davidson_precond and davidson_eps_shift.",
    )
    parser.add_argument(
        "--full-bse-instability", choices=("refuse", "continue"), default="refuse",
        help="full_bse_solver = davidson: what to do when K = [[A, B], [B*, A*]] is "
             "found not positive definite (an unstable reference state: an imaginary "
             "or non-positive root).  ``refuse`` (default) stops with the measured "
             "lam_min.  ``continue`` solves on, reports the unstable root(s) in the "
             "log and RESULTS, and writes the positive roots above them.",
    )
    parser.add_argument(
        "--haydock-iter", type=int, default=0,
        help="with --tda --dipole: after the eigensolve, run N steps of the "
             "Haydock recursion (bse.absorption_haydock's) on the same loaded "
             "payload and kernel, in this process, and write "
             "--haydock-out-prefix.h5 plus one .dat per polarization.  0 "
             "(default) = off.  Same physics and outputs as the separate "
             "driver, without its startup, load and W transform.")
    parser.add_argument("--haydock-pol", default="xyz",
                        help="with --haydock-iter: Cartesian polarizations to "
                             "recurse, a subset of 'xyz' (default all); one "
                             "matvec per polarization per step")
    parser.add_argument("--haydock-eta-ev", type=float, default=0.1,
                        help="with --haydock-iter: Lorentzian broadening (eV)")
    parser.add_argument("--haydock-omega-min-ev", type=float, default=0.0)
    parser.add_argument("--haydock-omega-max-ev", type=float, default=15.0)
    parser.add_argument("--haydock-n-omega", type=int, default=1500)
    parser.add_argument("--haydock-n-spin", type=int, default=1,
                        help="with --haydock-iter: spin multiplicity of the "
                             "eps2 prefactor (2 only for collinear "
                             "spin-polarised); n_spinor is the payload's")
    parser.add_argument("--haydock-no-eps1", action="store_true",
                        help="with --haydock-iter: skip the Kramers-Kronig eps1")
    parser.add_argument("--haydock-out-prefix", default="absorption_haydock",
                        help="with --haydock-iter: output prefix (h5 and the "
                             "absorption_haydock_b<1-3>_eh.dat files beside it)")
    parser.add_argument("--kpm-dos", action="store_true", help="Run KPM Chebyshev DOS and exit.")
    parser.add_argument("--kpm-n-moments", type=int, default=100, help="Chebyshev moments M for KPM.")
    parser.add_argument("--kpm-n-random", type=int, default=4, help="Stochastic trace vectors R for KPM.")
    parser.add_argument("--kpm-n-lanczos", type=int, default=100, help="Lanczos steps for KPM spectral bounds.")
    parser.add_argument("--kpm-emin-ev", type=float, default=None, help="Override KPM E_min (eV).")
    parser.add_argument("--kpm-emax-ev", type=float, default=None, help="Override KPM E_max (eV).")
    parser.add_argument("--kpm-plot-file", type=str, default="bse_dos_kpm.png", help="KPM DOS plot output file.")
    parser.add_argument("--eqp", type=str, default=None, help="QP energy file (eqp1.dat format).")
    parser.add_argument(
        "--qp-rotations", default=None, metavar="QP_ROTATIONS_H5",
        help="the GW run's qp_wfn_rotations.h5: solve in the eigenbasis of "
             "its H_QP = U diag(E_QP) U^H (rotated psi at the centroids, "
             "E_QP, rotated dipoles; bse.bse_qp_rotation).  The window "
             "counts QP states.  Mutually exclusive with eqp_corrections.")
    parser.add_argument("--n-occ", type=int, default=None, help="Number of occupied bands.")
    parser.add_argument("--degeneracy-check-override", action="store_true")
    # TDA (BerkeleyGW's default; full_bse opts out) and the screened kernel
    # (D + w V - W; local_fields = true drops W) are the defaults.
    parser.set_defaults(tda=True, bse=True)
    parser.add_argument("--haydock", action="store_true")
    parser.add_argument("--number-iterations", type=int, default=None)
    parser.add_argument("--delta-frequency", type=float, default=None)
    parser.add_argument("--polarization", default=None)
    parser.add_argument("--lorentzian-broadening", action="store_true")
    parser.add_argument("--gaussian-broadening", action="store_true")
    return parser


def _write_eigenvectors(n: int):
    """BerkeleyGW's ``write_eigenvectors``: 0 none, < 0 every computed state, > 0 that many."""
    return None if n == 0 else -1 if n < 0 else n


#: Deck key -> settings attribute.  BerkeleyGW absorption.x names where the
#: meaning matches (``number_val_bands_fine``, ``number_eigenvalues``,
#: ``energy_resolution`` ...); LORRAX's own solver keys keep LORRAX names.
BSE_KEYS = {
    "number_val_bands_fine": Key("n_val", "int"),
    "number_cond_bands_fine": Key("n_cond", "int"),
    "highest_occupied_band": Key("n_occ", "int"),
    "band_degeneracy": Key("band_degeneracy", choices=tuple(MODES)),
    "degeneracy_check_override": Key("degeneracy_check_override", "bool"),
    "degeneracy_tol_ry": Key("degeneracy_tol_ry", "float"),
    "qp_rotations_file": Key("qp_rotations", "path"),
    "dipole_file": Key("dipole", "path"),
    "dipole_position_fix": Key("dipole_position_fix", "bool"),
    "full_bse": Key("tda", "bool", convert=lambda full: not full),
    "full_bse_solver": Key("full_bse_solver", choices=("dense", "davidson")),
    "full_bse_instability": Key("full_bse_instability", choices=("refuse", "continue")),
    "local_fields": Key("rpa", "bool"),
    "number_eigenvalues": Key("n_eig", "int"),
    "write_eigenvectors": Key("write_eigs", "int", convert=_write_eigenvectors),
    "helicity_table": Key("helicity_table", "bool"),
    "solver": Key("solver", choices=("davidson",)),
    "solver_max_iter": Key("max_lanczos_iter", "int"),
    "davidson_precond": Key("davidson_precond", choices=("bare", "exact", "auto")),
    "davidson_tol_mev": Key("davidson_tol_mev", "float"),
    "davidson_margin": Key("davidson_margin", "int"),
    "davidson_margin_tol_mev": Key("davidson_margin_tol_mev", "float"),
    "davidson_olsen": Key("davidson_olsen", "bool"),
    "davidson_eps_shift": Key("davidson_eps_shift", "float"),
    "davidson_m_max": Key("davidson_m_max", "int"),
    "haydock": Key("haydock", "bool"),
    "number_iterations": Key("number_iterations", "int"),
    "energy_resolution": Key("haydock_eta_ev", "float"),
    "omega_min_ev": Key("haydock_omega_min_ev", "float"),
    "omega_max_ev": Key("haydock_omega_max_ev", "float"),
    "delta_frequency": Key("delta_frequency", "float"),
    "no_eps1": Key("haydock_no_eps1", "bool"),
    "lorentzian_broadening": Key("lorentzian_broadening", "bool"),
    "gaussian_broadening": Key("gaussian_broadening", "bool"),
    "polarization": Key("polarization", "floats"),
    "kpm_dos": Key("kpm_dos", "bool"),
    "kpm_window_count": Key("kpm_window_count", "int"),
    "kpm_n_moments": Key("kpm_n_moments", "int"),
    "kpm_n_random": Key("kpm_n_random", "int"),
    "kpm_n_lanczos": Key("kpm_n_lanczos", "int"),
    "kpm_emin_ev": Key("kpm_emin_ev", "float"),
    "kpm_emax_ev": Key("kpm_emax_ev", "float"),
    "kpm_plot_file": Key("kpm_plot_file", "str"),
}

#: Deck keys the other BSE drivers also read (bse.absorption_haydock,
#: bse.exciton_bands, bse.bsemat_cli).  One deck serves them all, so a route of
#: this driver that does not read one of these leaves it alone.
_SHARED_KEYS = frozenset({
    "number_val_bands_fine", "number_cond_bands_fine", "highest_occupied_band",
    "band_degeneracy", "degeneracy_check_override", "degeneracy_tol_ry",
    "dipole_file", "dipole_position_fix", "number_eigenvalues", "solver",
    "solver_max_iter", "davidson_tol_mev", "number_iterations",
    "energy_resolution", "omega_min_ev", "omega_max_ev", "delta_frequency", "no_eps1",
    "polarization", "lorentzian_broadening", "gaussian_broadening",
})

#: The settings each route reads, beyond the ones every route reads.  A key
#: the deck sets on a route that does not read it refuses by name: a route
#: that drops the QP energies, the root count, the occupations or the solver
#: settings silently would come back on DFT energies.
_COMMON_DESTS = {"input", "n_val", "n_cond", "px", "py", "rpa", "bse", "tda",
                 "kpm_dos"}
_ROUTE_DESTS = {
    "kpm": {
        "kpm_n_moments", "kpm_n_random", "kpm_n_lanczos", "kpm_window_count",
        "kpm_plot_file", "kpm_emin_ev", "kpm_emax_ev"},
    "lanczos": {
        "n_eig", "write_eigs", "report_file", "max_lanczos_iter", "solver",
        "davidson_precond", "davidson_olsen", "davidson_eps_shift",
        "davidson_m_max", "davidson_tol_mev", "davidson_margin",
        "davidson_margin_tol_mev",
        "full_bse_solver", "full_bse_instability",
        "eqp", "n_occ", "band_degeneracy", "degeneracy_tol_ry", "dipole",
        "dipole_position_fix", "helicity_table", "qp_rotations", "haydock",
        "number_iterations", "delta_frequency", "polarization",
        "lorentzian_broadening", "gaussian_broadening", *(
            "haydock_" + k for k in ("iter", "pol", "eta_ev", "omega_min_ev",
                                     "omega_max_ev", "n_omega", "n_spin",
                                     "no_eps1", "out_prefix"))},
}

#: The route names the refusals print ("lanczos" is the eigensolve).
_ROUTE_NAMES = {"lanczos": "eigensolve", "kpm": "kpm_dos"}

#: The Davidson settings ``full_bse_solver = davidson`` reads (with the shared
#: ``davidson_tol_mev`` and ``solver_max_iter``).
_FULL_DAVIDSON_DESTS = frozenset({
    "davidson_precond", "davidson_olsen", "davidson_eps_shift", "davidson_m_max",
    "davidson_tol_mev", "davidson_margin", "davidson_margin_tol_mev"})


def _deck_settings(input_file, parser):
    """``(args, named)``: the settings the deck's BSE keys give, checked.

    ``named`` is the set of deck keys the deck names.  A deck key this driver
    cannot use refuses through ``parser.error``.
    """
    schema = _settings_parser()
    try:
        params, named = deck_params(input_file)
        refuse_retired_keys(params, named)
        refuse_retired_solver(params, named)
        args, named = read_settings(
            input_file, schema.parse_args([]), BSE_KEYS,
            required=("number_val_bands_fine", "number_cond_bands_fine"),
            params=params, named=named)
        from gw.gw_config import require_deck_keys
        require_deck_keys(named, "band_request", "number_bands", "centroids_file",
                          error=DeckError)
        args.eqp = eqp_file(params, named, input_file)
        if args.degeneracy_check_override:
            if "band_degeneracy" in named and args.band_degeneracy != "off":
                raise DeckError("`degeneracy_check_override` is band_degeneracy = off; "
                                "the deck also sets band_degeneracy = "
                                f"{args.band_degeneracy}")
            args.band_degeneracy = "off"
        args.haydock_iter = (int(args.number_iterations)
                             if args.haydock and args.number_iterations else
                             _HAYDOCK_DEFAULT_ITER if args.haydock else 0)
        if args.delta_frequency:
            args.haydock_n_omega = omega_count(args.haydock_omega_min_ev,
                                               args.haydock_omega_max_ev,
                                               float(args.delta_frequency))
        check_dipole_operator(params, named, args.dipole)
        check_fine_grid_keys(params, named)
        from .bse_window import _parse_wfn_path
        kernel = spin_kernel(params, named, _parse_wfn_path(input_file))
        args.rpa, args.spin_channel = not kernel["include_w"], kernel["channel"]
        args.kernel_label = kernel["label"]
        args.kernel_name = "spinor" if kernel["nspinor"] == 2 else kernel["channel"]
        note_extended_kernel(params, named, log)
    except DeckError as exc:
        parser.error(str(exc))
    return schema, args, named


#: Haydock steps when ``haydock = true`` names no ``number_iterations``
#: (bse.absorption_haydock's default).
_HAYDOCK_DEFAULT_ITER = 200


def parse_args(argv=None):
    """``(route, args)``: the command line, then the deck's BSE keys, checked."""
    import sys
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    cli = parser.parse_args(raw)
    if cli.input is None:
        parser.error("Default run requires -i/--input.")
    schema, args, named = _deck_settings(cli.input, parser)
    args.input, args.px, args.py = cli.input, cli.px, cli.py
    args.report_file = cli.report_file
    if cli.out_prefix is not None:
        args.haydock_out_prefix = cli.out_prefix
    args.deck_named = frozenset(named)
    by_key = key_of(BSE_KEYS)
    name = lambda dest: "eqp_corrections" if dest == "eqp" else by_key(dest)  # noqa: E731
    schema_dests = set(vars(schema.parse_args([])))
    route = "kpm" if args.kpm_dos else "lanczos"
    ignored = sorted(
        f"`{name(dest)}`" for dest, value in vars(args).items()
        if dest not in _COMMON_DESTS | _ROUTE_DESTS[route]
        and name(dest) not in _SHARED_KEYS
        and dest in schema_dests
        and value != schema.get_default(dest))
    if ignored:
        parser.error(
            f"{', '.join(ignored)} {'is' if len(ignored) == 1 else 'are'} not "
            f"read by the {_ROUTE_NAMES[route]} route and would be ignored; drop "
            f"{'it' if len(ignored) == 1 else 'them'} or choose the route "
            f"that reads {'it' if len(ignored) == 1 else 'them'} (the default, "
            f"the eigensolve with QP energies and eigenvectors; kpm_dos = true for "
            f"the density of states)")
    if args.eqp is not None and args.qp_rotations is not None:
        parser.error("eqp_corrections and qp_rotations_file are two descriptions "
                     "of the QP state: the first relabels DFT bands with diagonal "
                     "QP energies, the second rotates them into the eigenbasis of "
                     "H_QP; choose one")
    if route == "lanczos" and args.tda:
        full_only = sorted(
            f"`{dest}`" for dest in ("full_bse_solver", "full_bse_instability")
            if getattr(args, dest) != schema.get_default(dest))
        if full_only:
            parser.error(
                f"{', '.join(full_only)} {'is' if len(full_only) == 1 else 'are'} read "
                f"with full_bse = true only (the Tamm-Dancoff eigensolve is selected by "
                f"`solver`) and would be ignored; set full_bse = true or drop "
                f"{'it' if len(full_only) == 1 else 'them'}")
    if route == "lanczos" and not args.tda:
        # full_bse = true takes bse_lanczos.solve_bse_sharded -> bse_nontda.  Its
        # dense solve (full_bse_solver = dense, the default) reads n_eig and the
        # kernel only; full_bse_solver = davidson reads _FULL_DAVIDSON_DESTS too.
        # The same unread-key rule as above.
        full_davidson = args.full_bse_solver == "davidson"
        read = _FULL_DAVIDSON_DESTS if full_davidson else frozenset()
        unread = sorted(
            f"`{name(dest)}`"
            for dest in _FULL_DAVIDSON_DESTS
            if dest not in read and name(dest) not in _SHARED_KEYS
            and getattr(args, dest) != schema.get_default(dest))
        if unread:
            parser.error(
                f"{', '.join(unread)} {'is' if len(unread) == 1 else 'are'} not "
                f"read by full_bse = true (the dense structure-preserving solve "
                f"returns the lowest n_eig roots with no Davidson "
                f"iteration) and would be ignored; drop "
                f"{'it' if len(unread) == 1 else 'them'}, set full_bse_solver = davidson "
                f"for the full-BSE Davidson, or set full_bse = false "
                f"for the eigensolve that reads {'it' if len(unread) == 1 else 'them'}")
        # The keys other BSE drivers share are left alone by the rule above.  The
        # dense solve reads neither the iteration cap nor ``davidson_tol_mev``.
        shared_unread = []
        if not full_davidson:
            for dest in ("max_lanczos_iter", "davidson_tol_mev"):
                if getattr(args, dest) != schema.get_default(dest):
                    shared_unread.append((
                        name(dest), "the dense solve has no iteration; set "
                                    "full_bse_solver = davidson to read it, or drop it"))
        if shared_unread:
            parser.error(
                ", ".join(f"`{key}`" for key, _ in shared_unread)
                + f" {'is' if len(shared_unread) == 1 else 'are'} not read by full_bse = true"
                + ("" if full_davidson else " with full_bse_solver = dense")
                + " and would be ignored: "
                + "; ".join(f"`{key}`: {why}" for key, why in shared_unread))
        if not full_davidson and args.full_bse_instability != "refuse":
            parser.error(
                "`full_bse_instability` is read by full_bse_solver = davidson only: the "
                "dense solve refuses a K = [[A, B], [B*, A*]] that is not positive "
                "definite and has no continued solve; set full_bse_solver = davidson "
                "or drop it")
    if args.davidson_margin < 0:
        parser.error("davidson_margin must be >= 0")
    if (args.davidson_margin_tol_mev is not None
            and args.davidson_margin_tol_mev < args.davidson_tol_mev):
        parser.error("davidson_margin_tol_mev is a looser lock for the margin "
                     "roots: it must be >= davidson_tol_mev")
    if args.haydock_iter < 0:
        parser.error("number_iterations must be > 0")
    if args.haydock_iter > 0:
        if not args.tda or args.dipole is None:
            parser.error("haydock = true runs the TDA Haydock recursion seeded "
                         "by the window dipoles: it needs the TDA (no full_bse) "
                         "and dipole_file")
        if args.rpa:
            parser.error("haydock = true runs absorption_haydock's screened "
                         "recursion (D + w V - W): drop local_fields")
        if args.qp_rotations is not None:
            parser.error("haydock = true on a qp_rotations_file bundle is not "
                         "supported: run bse.absorption_haydock's inputs")
    if args.lorentzian_broadening and args.gaussian_broadening:
        parser.error("lorentzian_broadening and gaussian_broadening name two "
                     "profiles; keep one (the default is gaussian, as BerkeleyGW's "
                     "diagonalization)")
    args.spectrum_prefix = "absorption" if cli.out_prefix is None else cli.out_prefix
    if args.helicity_table and args.dipole is None:
        parser.error("helicity_table reports the per-state dipoles: it needs "
                     "dipole_file")
    return route, args


def settings_rows(args, *, source) -> tuple:
    """The BSE SETTINGS rows from the deck's BSE keys; ``source(*keys)`` says
    whether the deck names a key (``bse_deck.deck_source``)."""
    use_tda = args.tda
    include_w = not (args.rpa or not args.bse)
    basis_file = args.qp_rotations or args.eqp
    full_davidson = not use_tda and args.full_bse_solver == "davidson"
    rows = (
        Setting("full_bse", "off" if use_tda else "on", source("full_bse"),
                "Tamm-Dancoff Hermitian BSE" if use_tda else
                "full resonant-antiresonant BSE"),
        Setting("kernel", getattr(args, "kernel_name", "singlet") if include_w else "local_fields",
                source("spin_singlet", "spin_triplet", "local_fields", "spinor"),
                getattr(args, "kernel_label", "")),
        (Setting("full_bse_solver", "davidson", source("full_bse_solver"),
                 f"structure-preserving Davidson (paired trial vectors over the pair applier "
                 f"A X + B conj(X)): block of {int(args.n_eig)} roots + up to "
                 f"{int(args.davidson_margin)} margin, {args.davidson_precond} diagonal "
                 f"preconditioner paired as [D; -D]; no dense assembly")
         if full_davidson else
         Setting("solver", "dense" if not use_tda else
                 str(args.solver),
                 "derived" if not use_tda else source("solver"),
                 ("dense structure-preserving solve of the assembled A and B (host eigh); "
                  "no iteration, so no Davidson setting is read"
                  if not use_tda else
                  f"block of {int(args.n_eig)} roots + up to {int(args.davidson_margin)} "
                  f"margin, {args.davidson_precond} "
                  f"diagonal preconditioner; trial-stack matvec"))),
        (None if not use_tda and not full_davidson else   # the dense full solve has no stopping rule
         Setting("davidson_tol_mev", f"{float(args.davidson_tol_mev):g}",
                 source("davidson_tol_mev"),
                 "stop when every root's residual is below this (meV)")),
        (Setting("full_bse_instability", str(args.full_bse_instability),
                 source("full_bse_instability"),
                 "K = [[A, B], [B*, A*]] not positive definite: the run refuses with the "
                 "measured lam_min" if args.full_bse_instability == "refuse" else
                 "K = [[A, B], [B*, A*]] not positive definite: the solve continues, reports "
                 "the unstable roots and writes the positive ones above them")
         if full_davidson else None),
        Setting("qp_rotations_file" if args.qp_rotations else "eqp_corrections",
                os.path.basename(str(basis_file)) if basis_file else "none",
                source("qp_rotations_file", "eqp_corrections", "eqp_co_corrections"),
                "QP-rotated bands, eigenstates of the GW H_QP" if args.qp_rotations
                else "DFT bands, diagonal QP energies" if args.eqp
                else "DFT bands and energies"),
        Setting("band_degeneracy", str(args.band_degeneracy),
                source("band_degeneracy", "degeneracy_check_override"),
                f"tolerance {float(args.degeneracy_tol_ry) * RYD_TO_EV * 1.0e3:.3f} meV"),
    ) + ((Setting("number_iterations", str(int(args.haydock_iter)),
                  source("haydock", "number_iterations"),
                  f"Haydock eps2 ({'b1 b2 b3' if args.polarization is None else 'e'}) "
                  f"on the eigensolve's payload, "
                  f"eta {float(args.haydock_eta_ev):g} eV"),)
         if getattr(args, "haydock_iter", 0) else ())
    return tuple(row for row in rows if row is not None)


def _haydock_settings(args) -> dict | None:
    """The Haydock deck keys as ``_preview_lanczos``'s ``haydock`` bundle (None: off)."""
    if args.haydock_iter <= 0:
        return None
    return {"n_iter": int(args.haydock_iter), "polarization": args.polarization,
            "eta_ev": float(args.haydock_eta_ev),
            "omega_min_ev": float(args.haydock_omega_min_ev),
            "omega_max_ev": float(args.haydock_omega_max_ev),
            "n_omega": int(args.haydock_n_omega), "n_spin": int(args.haydock_n_spin),
            "no_eps1": bool(args.haydock_no_eps1),
            "out_prefix": str(args.haydock_out_prefix)}


def _spectrum_settings(args) -> dict | None:
    """The absorption_eh/noeh settings (None without dipole_file); TDA and full BSE alike."""
    if args.dipole is None:
        return None
    import numpy as np
    return {"polarization": args.polarization,
            "width_ev": float(args.haydock_eta_ev),
            "kind": "lorentzian" if args.lorentzian_broadening else "gaussian",
            "omegas_ev": np.linspace(float(args.haydock_omega_min_ev),
                                     float(args.haydock_omega_max_ev),
                                     int(args.haydock_n_omega)),
            "no_eps1": bool(args.haydock_no_eps1),
            "prefix": args.spectrum_prefix}


def _file_rows(result, wfn_path, args):
    """The OUTPUT FILES rows: what the run read and wrote."""
    rows = [("GW/BSE restart", "read", result["restart_file"]),
            ("wavefunctions", "read", wfn_path)]
    rows += [(label, "written", path) for label, path in (
        ("exciton eigenvectors", result["eigenvector_file"]),
        ("Haydock absorption", result["haydock_file"])) if path]
    rows += [("exciton energies, all roots", "written", path)
             for path in result["roots_files"]]
    rows += [("absorption spectrum", "written", path)
             for path in result.get("spectrum_files", ())]
    rows += [(label, "read", path) for label, path in (
        ("QP corrections", args.eqp), ("QP rotations", args.qp_rotations)) if path]
    rows.append(("input deck", "read", args.input))
    return rows


def _report_environment(report, args):
    """Open the wavefunction and report its operator and sampling context."""
    from .bse_window import _parse_wfn_path
    from wfn_loader import WfnLoader
    wfn_path = _parse_wfn_path(args.input)
    wfn = WfnLoader(wfn_path, mesh=RUNTIME.mesh)
    sym = wfn.symmetry()
    report.environment(wfn=wfn, lines=(
        "Transition data: distributed band and centroid blocks on X x Y",
        "Eigensolve path: " + (
            policy(args.solver, ("davidson",)) if args.tda else
            "structure-preserving Davidson, full BSE" if args.full_bse_solver == "davidson"
            else "dense structure-preserving solve, full BSE"),
        "Eigenvectors   : " + (
            "written to a separate numerical artifact"
            + (" with per-state dipoles <0|r|S>" if args.dipole else "")
            + (" and their in-plane helicity table" if args.helicity_table
               else "")
            if args.write_eigs is not None else "not requested"),
    ))
    report.sampling(wfn=wfn, sym=sym)
    return wfn_path, wfn


def main(argv=None) -> int:
    route, args = parse_args(argv)

    # Omitted --px/--py = the run's canonical square mesh, not 1x1.  A shape
    # that is not the job's refuses here, above every route, and the KPM
    # route is handed the resolved shape.
    mesh_xy = create_mesh_xy_from_flags(args.px, args.py)
    args.px, args.py = tuple(int(n) for n in mesh_xy.devices.shape)
    use_tda = args.tda
    use_rpa = args.rpa or not args.bse

    if route == "kpm":
        from . import bse_kpm
        bse_kpm.run(bse_kpm.settings(
            args.input, n_val=args.n_val, n_cond=args.n_cond, px=args.px,
            py=args.py, n_moments=args.kpm_n_moments,
            n_random=args.kpm_n_random, n_lanczos=args.kpm_n_lanczos,
            n_windows=args.kpm_window_count, plot_file=args.kpm_plot_file,
            rpa=use_rpa, tda=use_tda, emin_ev=args.kpm_emin_ev,
            emax_ev=args.kpm_emax_ev))
        return 0

    # Non-TDA (full BSE) now flows through the same preview via the
    # ``solve_bse_sharded(tda=False)`` dispatch -> ``bse_nontda`` (structure-
    # preserving definite-pencil / product solve).  TDA stays the default.
    report_path = os.path.abspath(args.report_file or "bse.out")
    with RunSession(RUNTIME, "bse", ScientificProductionReport, report_path,
                    stages=_STAGES, driver="bse",
                    purpose="Bethe-Salpeter eigensolve") as run:
        report = run.report
        report.begin(input_file=args.input)
        report.devices(mesh_role="BSE transition axes c x v")
        report.settings(settings_rows(args, source=deck_source(args.deck_named)))

        wfn_path, wfn = _report_environment(report, args)

        result = _preview_lanczos(
            args.input,
            args.n_val,
            args.n_cond,
            n_eig=args.n_eig,
            write_eigs=args.write_eigs,
            dipole_file=args.dipole,
            dipole_position_fix=args.dipole_position_fix,
            helicity_table=args.helicity_table,
            max_lanczos_iter=args.max_lanczos_iter,
            include_W=not (args.rpa or not args.bse),
            eqp_file=args.eqp,
            qp_rotations_file=args.qp_rotations,
            n_occ=args.n_occ,
            davidson_m_max=args.davidson_m_max,
            davidson_precond=args.davidson_precond,
            davidson_olsen=args.davidson_olsen,
            davidson_eps_shift=args.davidson_eps_shift,
            davidson_tol_mev=args.davidson_tol_mev,
            davidson_margin=args.davidson_margin,
            davidson_margin_tol_mev=args.davidson_margin_tol_mev,
            degeneracy_mode=args.band_degeneracy,
            degeneracy_tol_ry=args.degeneracy_tol_ry,
            tda=use_tda,
            haydock=_haydock_settings(args),
            spectrum=_spectrum_settings(args),
            cell_volume=float(wfn.cell_volume),
            kgrid=tuple(int(n) for n in wfn.kgrid),
            report=report,
            full_bse_solver=args.full_bse_solver,
            full_bse_instability=args.full_bse_instability,
        )

        wfn.close()
        run.complete(files=_file_rows(result, wfn_path, args))
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)

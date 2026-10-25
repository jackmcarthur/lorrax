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
                       read_settings, spin_kernel)
from .bse_preconditioner import bgw_spin_kernel
from .bse_ring_comm import (
    create_mesh_2d,
    create_mesh_xy_from_flags,
    make_bse_shardings,
)
from .bse_stack_matvec import build_bse_stack_matvec_full
from .bse_preconditioner import compute_pair_amplitude
from file_io.restart_bundle import (_find_restart_file)
from .bse_lanczos import (
    simple_lanczos_eig,
    iters_reported,
)
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
    block_size: int = 1,
    rtol: float = 0.0,
    check_every: int = 4,
    n_reorth: int = -1,
    solver_kind: str = "davidson",
    davidson_m_max: int | None = None,
    davidson_precond: str = "exact",
    davidson_tol_mev: float = 0.01,
    davidson_margin: int = 8,
    davidson_margin_tol_mev: float | None = None,
    davidson_olsen: bool = False,
    davidson_eps_shift: float | None = None,
    trlan_m_max: int | None = None,
    trlan_n_keep: int | None = None,
    tda: bool = True,
    degeneracy_mode: str = DEFAULT_MODE,
    degeneracy_tol_ry: float = DEGENERACY_TOL_RY,
    haydock: dict | None = None,
    spectrum: dict | None = None,
    cell_volume: float | None = None,
    kgrid: tuple | None = None,
    report=None,
) -> dict:
    if not tda and kgrid is not None:
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
    # ``max_lanczos_iter`` is the *total* Krylov dimension upper
    # bound; block path divides by block_size.
    block_max_iter = max(1, max_lanczos_iter // max(1, block_size))
    mode = (f"convergence-driven (rtol={rtol:.1e}, every {check_every} iters)"
            if rtol > 0 else "fixed")
    if block_size > 1:
        log.info(f"Block Lanczos [{mode}]: ≤ {block_max_iter} block iter × "
                 f"block_size={block_size} = ≤ {block_max_iter * block_size} Krylov dim")
    else:
        log.info(f"Lanczos [{mode}]: ≤ {block_max_iter} iterations")
    # Resolve full-reorth sentinel (-1) to the actual Krylov depth.
    n_reorth_eff = block_max_iter if n_reorth < 0 else n_reorth
    # ONE row for the eigensolve — compile + every Krylov iteration.  It
    # is the number worth comparing across decks, and the only honest way
    # to say "the BSE solve cost X" (the matvec itself runs
    # block_max_iter x block_size times and must NOT be timed per call).
    with manual_stage(2, n_stages, "BSE eigensolve",
                      name="bse.eigensolve") as _sec:
        stage_row("solver", f"{solver_kind}, {int(n_eig)} roots, block size "
                            f"{int(block_size)}, at most {block_max_iter} iterations"
                            + (f", stops at relative change {rtol:.1e}" if rtol > 0
                               else ""))
        eigenvalues, eigenvectors, n_iter_done = solve_bse_sharded(
            data, mesh_xy, n_eig=n_eig, max_iter=block_max_iter,
            include_W=include_W, block_size=block_size,
            rtol=rtol, check_every=check_every, n_reorth=n_reorth_eff,
            solver_kind=solver_kind, tda=tda,
            davidson_m_max=davidson_m_max,
            davidson_eps_shift_Ry=(davidson_eps_shift
                if davidson_eps_shift else 1e-3),
            davidson_precond=davidson_precond,
            davidson_olsen=davidson_olsen,
            davidson_tol_Ry=float(davidson_tol_mev) / (RYD_TO_EV * 1.0e3),
            davidson_margin=int(davidson_margin),
            davidson_margin_tol_Ry=(None if davidson_margin_tol_mev is None else
                                    float(davidson_margin_tol_mev) / (RYD_TO_EV * 1.0e3)),
            trlan_m_max=trlan_m_max, trlan_n_keep=trlan_n_keep,
            keep_W_R=haydock is not None,
        )
        _sec.watch(eigenvalues, eigenvectors)
        _sec.result(f"{int(n_eig)} roots of the {int(bse_dim)}-state "
                    + ("TDA" if tda else "full") + " BSE")
    # Fixed-iteration routes return N_ITER_NOT_MEASURED; only the
    # convergence-driven route (rtol > 0) reports a real count.  Going
    # through the helper is what keeps the sentinel out of arithmetic.
    n_done = iters_reported(n_iter_done, block_max_iter)
    stopped = None
    if rtol > 0 and solver_kind == "lanczos":
        tag = "Block Lanczos" if block_size > 1 else "Lanczos"
        log.info(f"{tag} exited at iter {n_done}/{block_max_iter} "
                 f"(Krylov dim = {n_done * block_size})")
        stopped = (f"stopped after {n_done} of {block_max_iter} iterations "
                   f"(Krylov dimension {n_done * block_size})")
    if solver_kind == "davidson" and tda:
        stopped = _davidson_report(n_eig=n_eig, tol_mev=float(davidson_tol_mev),
                                   budget=block_max_iter, dim=bse_dim)
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
                extra_rows=(() if stopped is None else (("eigensolve", stopped),)))
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
    parser.add_argument("--feast-n-lanczos", type=int, default=10, help="Lanczos steps for FEAST bounds.")
    parser.add_argument("--feast-buffer", type=float, default=0.05, help="Emax buffer fraction for FEAST windows.")
    parser.add_argument("--feast-n-quad1", type=int, default=4, help="Quadrature points for FEAST iteration 1.")
    parser.add_argument("--feast-n-quad2", type=int, default=8, help="Quadrature points for FEAST iteration 2+.")
    parser.add_argument("--feast-quadrature", type=str, default="ellipse",
                        choices=["zolotarev", "ellipse"],
                        help="Quadrature type for FEAST filter (default: ellipse).")
    parser.add_argument(
        "--feast-units-ev-per-ry",
        type=float,
        default=13.6056980659,
        help="Energy conversion Ry -> eV for FEAST report.",
    )
    parser.add_argument("--feast-ritz-count", type=int, default=4, help="Ritz values per window.")
    parser.add_argument("--gmres-max-iter", type=int, default=10, help="GMRES iterations per shift.")
    parser.add_argument("--gmres-tol", type=float, default=1e-2, help="GMRES relative tolerance.")
    parser.add_argument("--gmres-seed", type=int, default=0, help="GMRES random seed.")
    parser.add_argument("--gmres-fp32", action="store_true",
                        help="Use FP32 data/GMRES for shifted solves.")
    parser.add_argument("--tda", action="store_true",
                        help="Use Tamm-Dancoff approximation (TDA). Default is full non-TDA.")
    parser.add_argument("--rpa", action="store_true",
                        help="Use RPA kernel (D+V only), skip W0 term entirely.")
    parser.add_argument("--bse", action="store_true",
                        help="Use full BSE kernel (D+V-W). Overrides default RPA.")
    parser.add_argument("--kpm-window-count", type=int, default=4,
                        help="Number of KPM-derived windows for FEAST.")
    parser.add_argument("--feast", action="store_true",
                        help="FEAST window solver (the default route until 2026-10-21; "
                             "now explicit).  Refuses unless every in-window Ritz pair "
                             "is below --feast-tol-mev (GATE feast_converged)")
    parser.add_argument("--feast-tol-mev", type=float, default=0.01,
                        help="--feast: every in-window Ritz pair's residual "
                             "||H x - E x|| below this (meV), the Davidson default")
    parser.add_argument("--feast-iter", type=int, default=2,
                        help="--feast: FEAST subspace iterations (bse_feast's default)")
    parser.add_argument(
        "--feast-window1",
        nargs=2,
        metavar=("A", "B"),
        help="Override FEAST window 1 bounds in eV (use 'auto' for B).",
    )
    parser.add_argument(
        "--feast-window2",
        nargs=2,
        metavar=("A", "B"),
        help="Override FEAST window 2 bounds in eV (use 'auto' for B).",
    )
    parser.add_argument(
        "--write-eigs",
        nargs="?",
        const=-1,
        type=int,
        help="Write eigenvectors.h5 (optional N, default: n-eig).",
    )
    parser.add_argument(
        "--dipole", default=None, metavar="DIPOLE_H5",
        help="with --write-eigs (TDA): contract each written state against "
             "this dipole.h5 (psp.get_dipole_mtxels, the file "
             "absorption_haydock reads) and store <0|r|S> as "
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
             "max(30, min(200, bse_dim // 2))).  --solver lanczos: the Krylov "
             "dimension it runs (block iters x --block-size).  --solver "
             "davidson: the CAP on block iterations; the solve stops at its "
             "residual test (--davidson-tol-mev).  --solver trlan: the matvec "
             "budget.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Block-Lanczos block size (default 1 = single-vector Lanczos). "
             "Larger blocks → larger per-call GEMMs and fewer host dispatches.",
    )
    parser.add_argument(
        "--lanczos-rtol",
        type=float,
        default=0.0,
        help="Convergence threshold on Ritz-eigenvalue change between checks. "
             "0 (default) = fixed --max-lanczos-iter iterations, no early exit. "
             ">0 enables ``lax.while_loop`` Lanczos that exits when the lowest "
             "n-eig Ritz values stabilise within rtol (block-Lanczos only).",
    )
    parser.add_argument(
        "--lanczos-check-every",
        type=int,
        default=4,
        help="Convergence check cadence in block iterations (default 4).",
    )
    parser.add_argument(
        "--n-reorth",
        type=int,
        default=-1,
        help="Lanczos partial-reorthogonalisation window. -1 (default) "
             "= full reorth (= max_lanczos_iter); essential for highly "
             "degenerate spectra (e.g. spinor BSE) so Ritz vectors stay "
             "orthogonal across the full Krylov basis. Smaller windows "
             "(e.g. 10) give ghost eigenvalues that destroy per-state "
             "oscillator strengths, and since 2026-08-08 they are no "
             "longer even faster: the batched reorth route "
             "(solvers.lanczos, cgs2) costs 2*max_iter collectives "
             "whatever the window. There is no speed reason to narrow "
             "this.",
    )
    parser.add_argument(
        "--solver",
        choices=("lanczos", "davidson", "trlan"),
        default="davidson",
        help="Eigensolver for the lowest --n-eig roots.  ``davidson`` (the "
             "default since 2026-10-20): block Davidson with the exact "
             "diagonal preconditioner, stopped when every root's residual is "
             "below --davidson-tol-mev.  ``lanczos``: single-vector or block "
             "Lanczos at a fixed Krylov dimension (--max-lanczos-iter), or to "
             "--lanczos-rtol.  ``trlan``: thick-restart Lanczos, the Krylov "
             "convergence of ``lanczos`` from a basis capped at "
             "--trlan-m-max slots, for a fixed matvec budget.  Flags of a "
             "solver that is not selected refuse.",
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
        "--trlan-m-max", type=int, default=None,
        help="Thick-restart Lanczos: basis slots (the memory cap). "
             "Default max(3*n_eig, 60). The useful range is 3-5x n_eig.",
    )
    parser.add_argument(
        "--trlan-n-keep", type=int, default=None,
        help="Thick-restart Lanczos: Ritz vectors retained across a restart. "
             "Default n_eig+10. Must satisfy n_eig <= n_keep < m_max.",
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


def _solver_name(name: str) -> str:
    """The deck's solver names; ``block_lanczos`` is the code's ``lanczos``."""
    return "lanczos" if name == "block_lanczos" else name


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
    "local_fields": Key("rpa", "bool"),
    "number_eigenvalues": Key("n_eig", "int"),
    "write_eigenvectors": Key("write_eigs", "int", convert=_write_eigenvectors),
    "helicity_table": Key("helicity_table", "bool"),
    "solver": Key("solver", choices=("davidson", "block_lanczos", "trlan"),
                  convert=_solver_name),
    "solver_max_iter": Key("max_lanczos_iter", "int"),
    "block_size": Key("block_size", "int"),
    "lanczos_rtol": Key("lanczos_rtol", "float"),
    "lanczos_check_every": Key("lanczos_check_every", "int"),
    "lanczos_n_reorth": Key("n_reorth", "int"),
    "davidson_precond": Key("davidson_precond", choices=("bare", "exact", "auto")),
    "davidson_tol_mev": Key("davidson_tol_mev", "float"),
    "davidson_margin": Key("davidson_margin", "int"),
    "davidson_margin_tol_mev": Key("davidson_margin_tol_mev", "float"),
    "davidson_olsen": Key("davidson_olsen", "bool"),
    "davidson_eps_shift": Key("davidson_eps_shift", "float"),
    "davidson_m_max": Key("davidson_m_max", "int"),
    "trlan_m_max": Key("trlan_m_max", "int"),
    "trlan_n_keep": Key("trlan_n_keep", "int"),
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
    "feast": Key("feast", "bool"),
    "feast_n_lanczos": Key("feast_n_lanczos", "int"),
    "feast_buffer": Key("feast_buffer", "float"),
    "feast_n_quad1": Key("feast_n_quad1", "int"),
    "feast_n_quad2": Key("feast_n_quad2", "int"),
    "feast_quadrature": Key("feast_quadrature", choices=("zolotarev", "ellipse")),
    "feast_ritz_count": Key("feast_ritz_count", "int"),
    "feast_tol_mev": Key("feast_tol_mev", "float"),
    "feast_iter": Key("feast_iter", "int"),
    "feast_window1": Key("feast_window1", "str", convert=str.split),
    "feast_window2": Key("feast_window2", "str", convert=str.split),
    "gmres_max_iter": Key("gmres_max_iter", "int"),
    "gmres_tol": Key("gmres_tol", "float"),
    "gmres_seed": Key("gmres_seed", "int"),
    "gmres_fp32": Key("gmres_fp32", "bool"),
}

#: Deck keys the other BSE drivers also read (bse.absorption_haydock,
#: bse.exciton_bands, bse.bsemat_cli).  One deck serves them all, so a route of
#: this driver that does not read one of these leaves it alone.
_SHARED_KEYS = frozenset({
    "number_val_bands_fine", "number_cond_bands_fine", "highest_occupied_band",
    "band_degeneracy", "degeneracy_check_override", "degeneracy_tol_ry",
    "dipole_file", "dipole_position_fix", "number_eigenvalues", "solver",
    "solver_max_iter", "block_size", "davidson_tol_mev", "number_iterations",
    "energy_resolution", "omega_min_ev", "omega_max_ev", "delta_frequency", "no_eps1",
    "polarization", "lorentzian_broadening", "gaussian_broadening",
})

#: The settings each route reads, beyond the ones every route reads.  A key
#: the deck sets on a route that does not read it refuses by name: the FEAST
#: route used to drop the QP energies, the root count, the occupations and
#: the solver settings silently, so a QP run came back on DFT energies.
_COMMON_DESTS = {"input", "n_val", "n_cond", "px", "py", "rpa", "bse", "tda",
                 "feast", "kpm_dos"}
_ROUTE_DESTS = {
    "feast": {
        "feast_n_lanczos", "feast_buffer", "feast_n_quad1", "feast_n_quad2",
        "feast_quadrature", "feast_units_ev_per_ry", "feast_ritz_count",
        "gmres_max_iter", "gmres_tol", "gmres_seed", "gmres_fp32",
        "kpm_window_count", "kpm_n_moments", "kpm_n_random", "kpm_n_lanczos",
        "feast_window1", "feast_window2", "feast_tol_mev", "feast_iter"},
    "kpm": {
        "kpm_n_moments", "kpm_n_random", "kpm_n_lanczos", "kpm_window_count",
        "kpm_plot_file", "kpm_emin_ev", "kpm_emax_ev"},
    "lanczos": {
        "n_eig", "write_eigs", "report_file", "max_lanczos_iter", "block_size",
        "lanczos_rtol", "lanczos_check_every", "n_reorth", "solver",
        "davidson_precond", "davidson_olsen", "davidson_eps_shift",
        "davidson_m_max", "davidson_tol_mev", "davidson_margin",
        "davidson_margin_tol_mev", "trlan_m_max", "trlan_n_keep",
        "eqp", "n_occ", "band_degeneracy", "degeneracy_tol_ry", "dipole",
        "dipole_position_fix", "helicity_table", "qp_rotations", "haydock",
        "number_iterations", "delta_frequency", "polarization",
        "lorentzian_broadening", "gaussian_broadening", *(
            "haydock_" + k for k in ("iter", "pol", "eta_ev", "omega_min_ev",
                                     "omega_max_ev", "n_omega", "n_spin",
                                     "no_eps1", "out_prefix"))},
}

#: The route names the refusals print ("lanczos" is the eigensolve).
_ROUTE_NAMES = {"lanczos": "eigensolve", "feast": "feast", "kpm": "kpm_dos"}

#: On the eigensolve route, the settings only one ``solver`` reads.  One set
#: away from its default under another solver refuses: the default solver
#: changed from block Lanczos to Davidson on 2026-10-20, and a deck that tuned
#: the Lanczos recurrence must not run Davidson and drop the tuning in silence.
_SOLVER_DESTS = {
    "lanczos": {"block_size", "lanczos_rtol", "lanczos_check_every", "n_reorth"},
    "davidson": {"davidson_precond", "davidson_olsen", "davidson_eps_shift",
                 "davidson_m_max", "davidson_tol_mev", "davidson_margin",
                 "davidson_margin_tol_mev"},
    "trlan": {"trlan_m_max", "trlan_n_keep"},
}


def _deck_settings(input_file, parser):
    """``(args, named)``: the settings the deck's BSE keys give, checked.

    ``named`` is the set of deck keys the deck names.  A deck key this driver
    cannot use refuses through ``parser.error``.
    """
    schema = _settings_parser()
    try:
        params, named = deck_params(input_file)
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
    if args.kpm_dos and args.feast:
        parser.error("deck keys `kpm_dos` and `feast` each select a route (the "
                     "eigensolve is the default); give one")
    route = "kpm" if args.kpm_dos else "feast" if args.feast else "lanczos"
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
            f"the eigensolve with QP energies and eigenvectors; feast = true for "
            f"FEAST windows; kpm_dos = true for the density of states)")
    if args.eqp is not None and args.qp_rotations is not None:
        parser.error("eqp_corrections and qp_rotations_file are two descriptions "
                     "of the QP state: the first relabels DFT bands with diagonal "
                     "QP energies, the second rotates them into the eigenbasis of "
                     "H_QP; choose one")
    if route == "lanczos":
        wrong = sorted(
            f"`{name(dest)}`"
            for solver, dests in _SOLVER_DESTS.items() if solver != args.solver
            for dest in dests
            if name(dest) not in _SHARED_KEYS
            and getattr(args, dest) != schema.get_default(dest))
        if wrong:
            solver = "block_lanczos" if args.solver == "lanczos" else args.solver
            parser.error(
                f"{', '.join(wrong)} {'is' if len(wrong) == 1 else 'are'} not "
                f"read by solver = {solver} (the default since 2026-10-20 "
                f"is davidson); name the solver that reads "
                f"{'it' if len(wrong) == 1 else 'them'} or drop "
                f"{'it' if len(wrong) == 1 else 'them'}")
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
    if args.dipole is not None and not args.tda:
        parser.error("dipole_file gives the per-state dipoles, the absorption "
                     "spectra and the Haydock seed of the TDA: drop full_bse (the "
                     "full-BSE oscillator strength needs both halves of the "
                     "paired vector)")
    if args.lorentzian_broadening and args.gaussian_broadening:
        parser.error("lorentzian_broadening and gaussian_broadening name two "
                     "profiles; keep one (the default is gaussian, as BerkeleyGW's "
                     "diagonalization)")
    args.spectrum_prefix = "absorption" if cli.out_prefix is None else cli.out_prefix
    if args.helicity_table and args.dipole is None:
        parser.error("helicity_table reports the per-state dipoles: it needs "
                     "dipole_file (with write_eigenvectors and the TDA)")
    return route, args


def settings_rows(args, *, source) -> tuple:
    """The BSE SETTINGS rows from the deck's BSE keys; ``source(*keys)`` says
    whether the deck names a key (``bse_deck.deck_source``)."""
    use_tda = args.tda
    include_w = not (args.rpa or not args.bse)
    basis_file = args.qp_rotations or args.eqp
    return (
        Setting("full_bse", "off" if use_tda else "on", source("full_bse"),
                "Tamm-Dancoff Hermitian BSE" if use_tda else
                "full resonant-antiresonant BSE"),
        Setting("kernel", getattr(args, "kernel_name", "singlet") if include_w else "local_fields",
                source("spin_singlet", "spin_triplet", "local_fields", "spinor"),
                getattr(args, "kernel_label", "")),
        Setting("solver", "block_lanczos" if args.solver == "lanczos" else str(args.solver),
                source("solver"),
                (f"block of {int(args.n_eig)} roots + up to {int(args.davidson_margin)} "
                 f"margin, {args.davidson_precond} "
                 f"diagonal preconditioner; trial-stack matvec"
                 if args.solver == "davidson" else
                 f"block size {int(args.block_size)}, {int(args.n_eig)} roots; "
                 "trial-stack matvec")),
        (Setting("davidson_tol_mev", f"{float(args.davidson_tol_mev):g}",
                 source("davidson_tol_mev"),
                 "stop when every root's residual is below this (meV)")
         if args.solver == "davidson" else
         Setting("lanczos_rtol", f"{float(args.lanczos_rtol):g}", source("lanczos_rtol"),
                 f"stop at this relative change, checked every "
                 f"{int(args.lanczos_check_every)} iterations"
                 if args.lanczos_rtol > 0.0 else "fixed Krylov dimension: no convergence test")),
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
    """The absorption_eh/noeh settings (None without dipole_file or under full_bse)."""
    if args.dipole is None or not args.tda:
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
        "Eigensolve path: " + policy(
            args.solver, ("lanczos", "davidson", "trlan")),
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
    # that is not the job's refuses here, above every route, and the FEAST
    # and KPM routes are handed the resolved shape.
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

    if route == "feast":
        if use_tda:
            # GATE feast_tda_complex: the TDA contour sums 2 Re over the upper
            # half-plane from real starts into real Ritz vectors, exact only
            # for a real symmetric A; LORRAX's A is complex Hermitian.
            raise SystemExit(
                "lorrax: GATE feast_tda_complex: feast = true with the TDA cannot "
                "converge on a complex A (real Ritz vectors); use the default "
                "Davidson eigensolve (drop feast = true), or full_bse = true.")
        log.warning("NOTE: feast = true is deprecated and experimental; the "
                    "default Davidson eigensolve is the supported route.",
                    extra={"lorrax": {"id": "feast_deprecated", "kind": "NOTE"}})
        from . import bse_feast
        bse_feast.run(bse_feast.settings(
            args.input, n_val=args.n_val, n_cond=args.n_cond, px=args.px,
            py=args.py, n_lanczos=args.feast_n_lanczos,
            buffer=args.feast_buffer, n_quad1=args.feast_n_quad1,
            n_quad2=args.feast_n_quad2, quadrature=args.feast_quadrature,
            units_ev_per_ry=args.feast_units_ev_per_ry, feast_ritz=True,
            feast_ritz_count=args.feast_ritz_count,
            gmres_max_iter=args.gmres_max_iter, gmres_tol=args.gmres_tol,
            gmres_seed=args.gmres_seed, gmres_fp32=args.gmres_fp32,
            # KPM-weighted windows unless the deck names its own: bse_feast
            # reads feast_window1/2 only when windows_kpm is off.
            rpa=use_rpa, tda=use_tda,
            windows_kpm=args.feast_window1 is None and args.feast_window2 is None,
            windows_kpm_count=args.kpm_window_count,
            kpm_n_moments=args.kpm_n_moments,
            kpm_n_random=args.kpm_n_random, kpm_seed=args.gmres_seed,
            kpm_n_energy_pts=2000, kpm_n_lanczos=args.kpm_n_lanczos,
            window1=args.feast_window1, window2=args.feast_window2,
            feast_iter=args.feast_iter, feast_tol_mev=args.feast_tol_mev))
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
            block_size=args.block_size,
            rtol=args.lanczos_rtol,
            check_every=args.lanczos_check_every,
            n_reorth=args.n_reorth,
            solver_kind=args.solver,
            davidson_m_max=args.davidson_m_max,
            davidson_precond=args.davidson_precond,
            davidson_olsen=args.davidson_olsen,
            davidson_eps_shift=args.davidson_eps_shift,
            davidson_tol_mev=args.davidson_tol_mev,
            davidson_margin=args.davidson_margin,
            davidson_margin_tol_mev=args.davidson_margin_tol_mev,
            trlan_m_max=args.trlan_m_max,
            trlan_n_keep=args.trlan_n_keep,
            degeneracy_mode=args.band_degeneracy,
            degeneracy_tol_ry=args.degeneracy_tol_ry,
            tda=use_tda,
            haydock=_haydock_settings(args),
            spectrum=_spectrum_settings(args),
            cell_volume=float(wfn.cell_volume),
            kgrid=tuple(int(n) for n in wfn.kgrid),
            report=report,
        )

        wfn.close()
        run.complete(files=_file_rows(result, wfn_path, args))
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)

"""Rank-zero, human-readable GW calculation report.

This is the scientific output of ``gwjax``.  It intentionally does not echo
library inventories, per-rank messages, HDF5 implementation details or the
driver's historical diagnostic prose.  Those remain available behind the one
driver-wide ``LORRAX_DEBUG_PRINT`` switch and in the launcher's own log.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from common.scientific_output import (
    FLOAT_DIGITS,
    abs_path,
    architecture_lines,
    band_range,
    centroid_orbit_line,
    file_table_lines,
    numerical_environment_lines,
    symmetry_sampling_lines,
)
from common.units import RYD_TO_EV
from .gw_config import qp_solver_semantics


_WARNING_WORDS = (
    "WARNING", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
    "NOT TRUSTWORTHY", "DEPRECATED",
)

# Stable role labels for the driver's closing file table.  These deliberately
# name the energy definition rather than inferring a method from ``eqpN``.
EQP0_FILE_ROLE = "fixed-DFT-state diagonal zeroth-order energies"
EQP1_FILE_ROLE = "fixed-DFT-state diagonal Z-linearized energies"
QP_ROTATIONS_FILE_ROLE = "full-matrix effective-H rotations"
QP_WFN_FILE_ROLE = "matched full-matrix QP wavefunctions"



def _human_bytes(n_bytes: float) -> str:
    """GiB above one GiB, MiB below, three significant figures either way."""
    gib = float(n_bytes) / 2**30
    if gib >= 1.0:
        return f"{gib:.3g} GiB"
    return f"{float(n_bytes) / 2**20:.3g} MiB"

def layout_dial_record_lines(
        *, config, n_mu: int, n_q_irr: int, processes: int) -> tuple[str, ...]:
    """Build the startup record for the two user-facing memory/layout dials."""
    n_mu = int(n_mu)
    q_per_task = (int(n_q_irr) + int(processes) - 1) // int(processes)
    matrix_gib = (
        q_per_task * n_mu * n_mu * np.dtype(np.complex128).itemsize
        / 1024 ** 3)
    layout = str(config.backend.linalg).strip().lower()
    lines = [
        f"[config provenance] linalg = {layout} "
        f"({config.backend.linalg_provenance})",
    ]
    if layout == "local":
        lines.append(
            "linalg = local: fastest, but each task holds ceil(N_q_irr/P) "
            "complete N_mu x N_mu dense matrices "
            f"(here ceil({int(n_q_irr)}/{int(processes)}) x {n_mu}^2 x "
            f"16 B = {_human_bytes(matrix_gib * 2**30)} per task, complex128) plus their "
            "factor workspace; on large systems where that is a large "
            "fraction of memory per task, set linalg = distributed")
    else:
        lines.append(
            "linalg = distributed: dense N_mu x N_mu matrices are 2D "
            f"distributed across the process mesh (N_mu = {n_mu})")

    low_mem = bool(config.memory.low_mem_bands)
    provenance = getattr(
        config.memory, "low_mem_bands_provenance",
        "deck" if "low_mem_bands" in config.raw_input_keys else "default")
    lines.append(
        f"[config provenance] low_mem_bands = {str(low_mem).lower()} "
        f"({provenance})")
    if low_mem:
        lines.append(
            "low_mem_bands = true: the two-face wavefunction carrier (band "
            f"chunks of {int(config.memory.band_chunk_size)}); required for the "
            "raw-parent (k_irr) route, which contracts G and the ISDF pair "
            "densities on the WFN's own k rows.  On an unreduced k grid the "
            "four-copy carrier (false) is faster when it fits.")
    return tuple(lines)


class GWProductionReport:
    """One clean GW report, owned and written only by process zero."""

    def __init__(self, path: str, *, runtime, debug: bool, stdout) -> None:
        self.path = abs_path(path)
        self.runtime = runtime
        self.debug = bool(debug)
        self.stdout = stdout
        self.rank = int(getattr(runtime, "process_index", 0))
        self._warnings: list[str] = []
        self._warnings_emitted = False
        self._stream = None
        if self.rank == 0:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.path, "w", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def emit(self, line: str = "") -> None:
        """Write one rank-zero line to both the report and live stdout."""
        if self.rank != 0:
            return
        text = str(line)
        self.stdout(text)
        if self._stream is not None:
            self._stream.write(text + "\n")

    def legacy_print(self, *args, sep: str = " ", **kwargs) -> None:
        """Sink historical component chatter unless driver debug is enabled.

        High-signal warnings are retained for one consolidated block at the
        end.  Exceptions still use stderr through the shared fail-fast path.
        """
        text = sep.join(str(v) for v in args)
        # Fixed-SC quadrature identity is a physics invariant, not backend
        # chatter: retain its compact receipt so every map's exact node set
        # and zero-rebuild claim remain auditable after live stdout is gone.
        if (text.startswith("  SC fixed quadrature: ")
                or text.startswith("    SC fixed window: ")):
            self.progress(text)
            return
        # The long-loop cadence is part of the scientific run record, not
        # component chatter.  LoopProgress owns these three stable shapes.
        if (text.startswith("Started ") or text.startswith("Finished ")
                or (text.startswith("[ ") and " | " in text
                    and " / " in text)):
            self.progress(text)
            return
        if self.debug:
            self.stdout(text, **kwargs)
            return
        upper = text.upper()
        if any(word in upper for word in _WARNING_WORDS):
            self._retain_warning(text)

    def _retain_warning(self, text: str) -> None:
        """Retain one normalized warning for the final consolidated block."""
        cleaned = " ".join(str(text).split())
        if cleaned and cleaned not in self._warnings:
            self._warnings.append(cleaned)

    def heading(self, title: str) -> None:
        self.emit()
        self.emit(title.upper())
        self.emit("-" * len(title))

    def progress(self, line: str) -> None:
        """Write one deliberately selected rank-zero progress line."""
        self.emit(line)

    def layout_dials(
            self, *, config, n_mu: int, n_q_irr: int, processes: int) -> None:
        """Record resolved layout/memory dials before allocating dense data."""
        for line in layout_dial_record_lines(
                config=config, n_mu=n_mu, n_q_irr=n_q_irr,
                processes=processes):
            self.progress(line)

    def begin(self, *, input_file: str, config) -> None:
        self.emit("=" * 78)
        self.emit("LORRAX GW CALCULATION")
        self.emit("=" * 78)
        self.emit(f"Input          : {abs_path(input_file)}")
        self.emit(f"Method         : {config.compute_mode.value} / "
                  f"{config.qp_solver.value}")

    def architecture(self) -> None:
        self.heading("Processor architecture")
        for line in architecture_lines(
                self.runtime, mesh_role="centroid axes X x Y"):
            self.emit(line)

    def method(self, *, config) -> None:
        """Dense statement of the physical approximations that actually run."""
        mode = config.compute_mode.value
        solver = config.qp_solver.value
        mode_text = {
            "x_only": "bare exchange",
            "cohsex": "static screened exchange + Coulomb hole",
            "gn_ppm": "Godby-Needs plasmon-pole GW",
            "hl_ppm": "Hybertsen-Louie plasmon-pole GW",
            "mpa": "multipole-approximation GW",
        }.get(mode, mode)
        solver_text = qp_solver_semantics(solver).description
        screening = getattr(config, "screening", None)
        diagrams = getattr(getattr(screening, "diagrams", None), "value", "-")
        diagram_text = {
            "w_rpa": "RPA Dyson series",
            "w_bse": "electron-hole ladder series",
        }.get(diagrams, diagrams)
        head = getattr(config, "head", None)
        head_mode = getattr(getattr(head, "correction", None), "value", "-")
        head_source = getattr(head, "wcoul0_source", "-")
        dim = int(getattr(config, "sys_dim", 3))
        geometry = {3: "3D bulk", 2: "2D slab", 1: "1D wire",
                    0: "0D isolated"}.get(dim, f"{dim}D")

        self.heading("Method and physical pathways")
        self.emit(f"Self-energy    : {mode_text}; {solver_text}")
        self.emit(
            f"QP consistency  : {solver} | other options: fixed_point "
            "(diagonal on-shell), self_consistent (rebuild G/W/Sigma)")
        eqp2 = getattr(config, "eqp2", None)
        if bool(getattr(eqp2, "enabled", False)):
            self.emit(
                "EQP2 treatment  : fixed-Sigma eigenvalue self-consistency; "
                "semicore E-E_F + conduction scissor outside the protected "
                "window; final post-rotation map required; "
                f"max|dE| cutoff={float(eqp2.tol_ev) * 1e3:.3f} meV "
                "(non-scissored); "
                f"accelerator={eqp2.accelerator}"
                + (f"(depth={int(eqp2.history_depth)})"
                   if str(eqp2.accelerator) == "rcrop" else "")
                + f"; max_iter={int(eqp2.max_iter)}; screening unchanged")
        else:
            self.emit(
                "EQP2 treatment  : off (set write_eqp2=true for fixed-Sigma "
                "eigenvalue self-consistency)")
        self.emit(f"Screening      : {diagram_text}; "
                  f"{getattr(screening, 'method', '-')} imaginary-axis quadrature")
        self.emit(f"Long wavelength: head={head_mode}; source={head_source}")
        self.emit(f"Coulomb system : {geometry}; Hartree=live G-space")
        self.emit("Spin channels  : " + (
            "charge + transverse current (bispinor)"
            if bool(getattr(config, "bispinor", False)) else "charge (scalar)"))
        self.emit("Degenerate sets: " + (
            "left in the input gauge" if bool(getattr(
                config, "no_degen_averaging", False))
            else "averaged at "
            f"{float(getattr(config, 'degen_avg_tol_ry', 0.0)) * RYD_TO_EV:.5e} eV"))
        self.emit("ISDF state     : " + (
            "restart requested" if bool(getattr(config, "restart", False))
            else "fresh fit requested"))

    def environment(self, *, config, wfn) -> None:
        f = self.runtime.facts
        backend = str(f.get("backend", "unknown")).lower()
        platform = "CUDA" if backend in ("gpu", "cuda") else backend
        self.heading("Numerical environment")
        for line in numerical_environment_lines(self.runtime):
            self.emit(line)
        self.emit(f"Wavefunctions  : {getattr(wfn, 'backend', 'unknown')} reader")
        self.emit(f"Dense linalg   : {config.backend.linalg} layout")

        # Report only controls with a caller in this calculation.  In
        # particular, the k-leading candidate has no production Sigma caller,
        # and the k-minor member belongs only to ladder-W screening.
        relevant = {"LORRAX_FFT_FFI", "LORRAX_FFT_FFI_FUSED",
                    "LORRAX_BANDS_GEMM_FFI", "LORRAX_CONV_KPAIR_FFI"}
        if getattr(config.screening.diagrams, "value", "") == "w_bse":
            relevant.add("LORRAX_CONV_KMINOR_FFI")
        descriptions = {
            "LORRAX_FFT_FFI": "flat-k FFTs for ISDF, chi0 and Sigma",
            "LORRAX_FFT_FFI_FUSED": "fused tau-domain convolution",
            "LORRAX_BANDS_GEMM_FFI": "right-GEMM contraction forming G(tau)",
            "LORRAX_CONV_KPAIR_FFI": "pair-density convolution forming V(q)",
            "LORRAX_CONV_KMINOR_FFI": "fused convolution in ladder-W",
        }
        cuda_engines = {
            "LORRAX_FFT_FFI": "cuFFT flat-k FFI",
            "LORRAX_FFT_FFI_FUSED": "cuFFT fused-convolution FFI",
            "LORRAX_CONV_KPAIR_FFI": "cuFFT pair-convolution FFI",
            "LORRAX_CONV_KMINOR_FFI": "cuFFT k-minor convolution FFI",
        }
        host_engines = {
            # The host FFT target resolves its linked FFTW3-ABI engine inside
            # the library; RuntimeFacts intentionally does not guess a vendor.
            "LORRAX_FFT_FFI": "host flat-k FFT FFI",
            "LORRAX_FFT_FFI_FUSED": "host fused-convolution FFI",
        }
        for dial in f.get("ffi_dials", ()):
            name = str(dial.get("env", ""))
            if name not in relevant:
                continue
            supported_here = platform in tuple(dial.get("platforms", ()))
            if not supported_here:
                state = "NATIVE"
                implementation = str(dial.get("off_label") or "JAX/XLA lowering")
            elif dial.get("enabled"):
                state = "ON"
                engine_names = (cuda_engines if platform == "CUDA"
                                else host_engines)
                implementation = engine_names.get(
                    name, str(dial.get("target") or "platform FFI kernel"))
            else:
                state = "OFF"
                implementation = str(dial.get("off_label") or "default lowering")
            self.emit(f"{name:<26} = {state:<6} : "
                      f"{descriptions[name]} ({implementation})")

    def sampling(self, *, wfn, sym, centroids=None) -> None:
        self.heading("Crystal symmetry and Brillouin-zone sampling")
        for line in symmetry_sampling_lines(wfn, sym, digits=FLOAT_DIGITS):
            self.emit(line)
        if centroids is not None:
            self.emit(centroid_orbit_line(centroids))

    def trs_pathways(self, *, config, sym, material_class="insulator") -> None:
        """Record the automatic TRS-dependent branch taken by this run."""
        allowed = bool(sym.trs_allowed)
        self.emit(
            "TRS pathways   : automatic from SymMaps.trs_allowed="
            f"{str(allowed).lower()}; no input override")
        mode = config.compute_mode.value
        if mode == "mpa":
            from .mpa.model import chi0_orientation_route

            self.emit(
                "MPA chi0 route : "
                + chi0_orientation_route(
                    material_class, trs_allowed=allowed))
        elif mode == "gn_ppm":
            reuse = str(getattr(config.ppm, "probe_chi_reuse", "off"))
            route = (
                "ordered orientations with the TR-odd channel"
                if not allowed else "incumbent symmetric completion")
            self.emit(f"GN probe route  : {route}; ppm_probe_chi_reuse={reuse}")
        elif mode == "hl_ppm":
            self.emit(
                "HL probe route  : incumbent real-axis completion"
                + ("; measured-broken-TR limitation retained"
                   if not allowed else ""))

    def bands(self, *, config, wfn, band_slices, zeta_ranges) -> None:
        b = band_slices
        n_e = float(getattr(wfn, "num_electrons", np.nan))
        self.heading("Band spaces and energy coverage")
        self.emit(f"Electrons      : {n_e:.5f}; occupied-band boundary = {b.b2}")
        self.emit(f"Occupied bands : {band_range(b.b0, b.b2)}")
        self.emit(f"QP valence     : {band_range(b.b1, b.b2)}")
        self.emit(f"QP conduction  : {band_range(b.b2, b.b3)}")
        self.emit(f"QP matrix      : {band_range(b.b0, b.b3)}")
        # ``b4_logical`` is the physical loaded top (tagged_arrays.py); a
        # band-slices object without it (older producers, test stubs) means
        # no mesh padding, so the padded top is the logical top.
        b4_logical = int(getattr(b, "b4_logical", b.b4))
        logical_chi = min(int(b.b4_chi), b4_logical)
        logical_sigma = min(int(b.b4_sigma), b4_logical)
        self.emit(f"chi0/W sum     : {band_range(b.b0, logical_chi)}")
        self.emit(f"Sigma sum      : {band_range(b.b0, logical_sigma)}")
        self.emit(f"Loaded ISDF psi: {band_range(b.b0, b4_logical)}")
        if int(b.b4) != b4_logical:
            self.emit(
                f"Band carrier    : {band_range(b.b0, b.b4)} "
                f"({int(b.b4) - b4_logical} zero-pad rows)")
        self.emit(f"zeta fit       : left {band_range(*zeta_ranges[0])}; "
                  f"right {band_range(*zeta_ranges[1])}")

    def sigma_coverage(self, *, config, band_slices, enk_dft_ry,
                       sigma_result) -> None:
        """Report the dynamic grid against Sigma's resolved energy origin.

        The result owns this origin: for a metal it is the fixed-N chemical
        potential of the accepted map, while for an insulator it is the
        configured VBM or midgap.  Reconstructing it from ``wfn.efermi``
        would silently misreport every non-midgap calculation.
        """
        if not config.compute_mode.is_dynamic:
            return
        b = band_slices
        omega_grid = np.asarray(sigma_result.omega_grid_ev, dtype=np.float64)
        grid_lo, grid_hi = float(omega_grid[0]), float(omega_grid[-1])
        ef_ev = float(sigma_result.efermi_dft_ev)
        provenance = (getattr(sigma_result, "omega_reference_provenance", None)
                      or config.sigma.fermi_reference)
        energies = np.asarray(enk_dft_ry, dtype=np.float64) * RYD_TO_EV
        rel = energies - ef_ev
        lv0, lv1 = b.b1 - b.b0, b.b2 - b.b0
        lc0, lc1 = b.b2 - b.b0, b.b3 - b.b0

        def span(i0, i1):
            if i1 <= i0:
                return None
            values = rel[:, i0:i1]
            return float(np.min(values)), float(np.max(values))

        val_span, cond_span = span(lv0, lv1), span(lc0, lc1)
        target = [v for pair in (val_span, cond_span) if pair is not None
                  for v in pair]

        self.heading("Dynamic Sigma energy coverage")
        self.emit(f"Energy origin   : E_F = {ef_ev:+.5f} eV ({provenance})")
        grid_note = (f"; step={float(config.sigma.omega_step_ev):.5f} eV; "
                     f"{int(omega_grid.size)} points")
        self.emit(f"Sigma omega    : [{grid_lo:+.5f}, {grid_hi:+.5f}] eV "
                  f"relative to E_F{grid_note}")
        if val_span is not None:
            self.emit(f"DFT QP valence : [{val_span[0]:+.5f}, "
                      f"{val_span[1]:+.5f}] eV relative to E_F")
        if cond_span is not None:
            self.emit(f"DFT QP conduct.: [{cond_span[0]:+.5f}, "
                      f"{cond_span[1]:+.5f}] eV relative to E_F")
        if target:
            target_lo, target_hi = min(target), max(target)
            lower_margin = target_lo - grid_lo
            upper_margin = grid_hi - target_hi
            if lower_margin >= 0.0 and upper_margin >= 0.0:
                self.emit("Coverage status : COMPLETE")
                self.emit(f"Grid margins    : {lower_margin:.5f} eV below; "
                          f"{upper_margin:.5f} eV above protected DFT states")
            else:
                self.emit("Coverage status : INCOMPLETE")
                shortfalls = []
                if lower_margin < 0.0:
                    shortfalls.append(f"{-lower_margin:.5f} eV below")
                if upper_margin < 0.0:
                    shortfalls.append(f"{-upper_margin:.5f} eV above")
                self.emit("Grid shortfall  : " + "; ".join(shortfalls)
                          + " protected DFT states")
                coverage = getattr(sigma_result, "omega_coverage", None)
                policy_name = str(getattr(coverage, "policy", "unknown"))
                affected = ""
                if coverage is not None:
                    mask = np.asarray(getattr(coverage, "mask_kn", ()))
                    default_uncovered = (int(np.count_nonzero(~mask.astype(
                        bool))) if mask.size else 0)
                    n_uncovered = int(getattr(
                        coverage, "n_uncovered", default_uncovered))
                    if mask.size:
                        affected = (f"; Sigma(E_DFT) has {n_uncovered}/"
                                    f"{mask.size} out-of-grid cells")
                self._retain_warning(
                    "WARNING: dynamic Sigma grid is incomplete for protected "
                    "DFT states (" + "; ".join(shortfalls) + f"){affected}; "
                    f"out-of-range policy={policy_name}. Widen "
                    "sigma_omega_min_ev / sigma_omega_max_ev or add a "
                    "sigma_omega_patches_ev window; use "
                    "LORRAX_OMEGA_OUT_OF_RANGE=refuse when endpoint values "
                    "must never enter an output.")

        state = "ON" if config.sigma.band_extrapolation else "OFF"
        estimator = (getattr(
            sigma_result, "band_extrapolation_estimator", None)
            or config.sigma.band_extrapolation_estimator)
        scheme = (getattr(sigma_result, "band_extrapolation_scheme", None)
                  or config.sigma.band_extrapolation_bracket_scheme)
        self.emit(f"Band tail      : {state} | estimator={estimator} | "
                  f"brackets={scheme}")
        counts = getattr(sigma_result, "band_extrapolation_counts", None)
        if counts:
            self.emit("Tail calculations: " + " / ".join(
                f"N{i}={int(value)}" for i, value in enumerate(counts, start=1))
                + " cumulative bands")

    def files(self, rows: Iterable[tuple[str, str, str]]) -> None:
        self.heading("Output files and inputs")
        for line in file_table_lines(rows, omit_paths=(self.path,)):
            self.emit(line)

    def qp_gap(self, *, band_slices, e_dft_ry, e_qp_ry) -> None:
        """Summarize the fundamental gap without duplicating EQP tables."""
        b = band_slices
        dft = np.asarray(e_dft_ry, dtype=np.float64) * RYD_TO_EV
        qp = np.asarray(e_qp_ry, dtype=np.float64) * RYD_TO_EV
        iv = int(b.b2 - b.b0 - 1)
        ic = int(b.b2 - b.b0)
        self.heading("Fundamental gap")
        if (dft.ndim != 2 or qp.ndim != 2 or dft.shape != qp.shape
                or iv < 0 or ic >= dft.shape[1]):
            self.emit("Gap            : unavailable (no complete protected "
                      "valence/conduction pair in the QP result)")
            return
        dft_gap = float(np.min(dft[:, ic]) - np.max(dft[:, iv]))
        qp_gap = float(np.min(qp[:, ic]) - np.max(qp[:, iv]))
        if not np.isfinite(dft_gap) or not np.isfinite(qp_gap):
            self.emit("Gap            : unavailable (non-finite band edge)")
            return
        state = "insulating" if qp_gap > 0.0 else "metallic/overlapping"
        self.emit(f"DFT gap        : {dft_gap:.5f} eV")
        self.emit(f"Full-matrix effective-H gap: {qp_gap:.5f} eV ({state})")
        self.emit(f"Gap correction : {qp_gap - dft_gap:+.5f} eV relative to DFT")
        self.emit("State energies  : written to the EQP files listed below")

    def eqp2_summary(self, *, band_slices, e_eqp2_ry,
                     iterations: int, residual_ev: float,
                     tol_ev: float) -> None:
        """Report the optional fixed-Sigma ladder and its actual criterion."""
        if e_eqp2_ry is None:
            return
        e = np.asarray(e_eqp2_ry, dtype=np.float64) * RYD_TO_EV
        iv = int(band_slices.b2 - band_slices.b0 - 1)
        ic = int(band_slices.b2 - band_slices.b0)
        self.heading("Fixed-Sigma eigenvalue consistency")
        self.emit(f"Convergence     : {int(iterations)} map evaluations; "
                  f"max|dE|={float(residual_ev) * 1e3:.6f} meV "
                  f"<= {float(tol_ev) * 1e3:.6f} meV over non-scissored "
                  "states; final post-rotation map verified")
        self.emit("Updated internally: QP eigenvalues/eigenvectors and the "
                  "basis representation of stored Sigma(omega)")
        self.emit("EQP2 eigenvectors: internal to this consistency loop; "
                  "not serialized")
        self.emit("QP WFN/rotations: remain the ordinary one-shot "
                  "full-matrix artifacts; eqp2.dat contains the final "
                  "EQP2 eigenvalue spectrum only")
        self.emit("Held fixed      : screening, W, and all self-energy diagrams")
        self.emit("Outside grid    : semicore preserves E-E_F; high conduction "
                  "uses the in-grid affine scissor")
        if e.ndim == 2 and iv >= 0 and ic < e.shape[1]:
            gap = float(np.min(e[:, ic]) - np.max(e[:, iv]))
            state = "insulating" if gap > 0.0 else "metallic/overlapping"
            self.emit(f"EQP2 gap       : {gap:.5f} eV ({state})")

    def timings(self, records, *, wall: float) -> None:
        """Print accumulated, non-overlapping major scientific stages.

        ``gw_jax.isdf`` and ``gw_jax.screening`` are inclusive parent timers.
        Their named child rows are therefore subtracted from a residual setup
        row instead of being added beside the parent.  This keeps the report's
        invariant explicit: every displayed stage contributes to the wall
        exactly once, while retaining the zeta/V and chi0/W distinctions that
        operators use to diagnose scaling.
        """
        rows = list(records)

        def total(predicate):
            return sum(float(row["inclusive"]) for row in rows
                       if predicate(row))

        def top_level(*names):
            wanted = set(names)
            return total(lambda r: r["name"] in wanted
                         and tuple(r.get("path", (r["name"],)))
                         == (r["name"],))

        def outer_prefixed(prefix, *, within=None):
            def selected(row):
                path = tuple(row.get("path", (row["name"],)))
                if not row["name"].startswith(prefix):
                    return False
                if within is not None and (not path or path[0] != within):
                    return False
                return not any(str(parent).startswith(prefix)
                               for parent in path[:-1])
            return total(selected)

        isdf_total = top_level("gw_jax.isdf")
        zeta = outer_prefixed(
            "gw_jax.zeta_fit_chunked", within="gw_jax.isdf")
        v_q = total(lambda r: r["name"] == "gw_jax.V_q_compute"
                    and tuple(r.get("path", ()))[:1] == ("gw_jax.isdf",))
        restart_load = total(
            lambda r: r["name"] == "gw_jax.restart_load"
            and tuple(r.get("path", ()))[:1] == ("gw_jax.isdf",))
        isdf_support = max(isdf_total - zeta - v_q - restart_load, 0.0)

        screening_total = top_level("gw_jax.screening")
        chi0 = outer_prefixed("chi.", within="gw_jax.screening")
        w_screen = outer_prefixed("W.", within="gw_jax.screening")
        screening_support = max(screening_total - chi0 - w_screen, 0.0)

        # The dynamic-Sigma executor opens ``sigma.rule_plan`` (box-rule
        # fitting, cached by box and tolerance) and ``sigma.tau_sweep`` (the
        # tau contraction) under gw_jax.sigma or, in a self-consistent run,
        # under gw_jax.sc_driver; nothing else of Sigma is separately named.
        sigma_total = top_level("gw_jax.sigma", "gw_jax.sc_driver")
        sigma_plan = outer_prefixed("sigma.rule_plan")
        sigma_sweep = outer_prefixed("sigma.tau_sweep")
        sigma_other = max(sigma_total - sigma_plan - sigma_sweep, 0.0)

        stages = [
            ("runtime bring-up", total(lambda r: r["name"].startswith(
                "gw_jax.runtime_stack."))),
            ("pre-main + imports", top_level("gw_jax.imports")),
            ("input + run setup", top_level("gw_jax.startup")),
            ("zeta", zeta),
            ("V(q)", v_q),
            ("restart load", restart_load),
            ("ISDF setup + I/O", isdf_support),
            ("minimax quadrature", top_level("gw_jax.minimax_quadrature")),
            ("chi0", chi0),
            ("W", w_screen),
            ("screening support", screening_support),
            ("W persist + q0 head", top_level(
                "gw_jax.persist_w0", "gw_jax.static_head")),
            ("Sigma rule plan", sigma_plan),
            ("Sigma tau sweep", sigma_sweep),
            ("Sigma other", sigma_other),
            ("mean-field load", top_level("gw_jax.kin_ion_load")),
            ("QP solve + diagonalize", top_level(
                "gw_jax.solve_qp", "gw_jax.qp_eigh")),
            ("fixed-Sigma evSC", top_level("gw_jax.eqp2_evsc")),
            ("result writes", top_level("gw_jax.output")),
        ]
        # A stage absent in a mode should not occupy a zero-valued report row.
        # Use the table's two-decimal display threshold so a floating-point
        # subtraction residual cannot survive the filter as ``0.00 s``.
        stages = [(name, seconds) for name, seconds in stages
                  if seconds >= 0.005]
        accounted = sum(seconds for _name, seconds in stages)
        stages.append(("other driver work", max(float(wall) - accounted, 0.0)))
        self.heading("Major-stage timing")
        self.emit("  stage                    wall (s)     fraction")
        for name, seconds in stages:
            self.emit(f"  {name:<22} {seconds:10.2f}  "
                      f"{100.0 * seconds / wall if wall else 0.0:9.2f}%")
        self.emit(f"  {'total run':<22} {wall:10.2f}  {100.0:9.2f}%")

    def warnings(self) -> None:
        if self._warnings_emitted:
            return
        self._warnings_emitted = True
        warnings = self._display_warnings()
        if warnings:
            self.heading("Warnings")
            for warning in warnings:
                self.emit(f"  {warning}")

    def _display_warnings(self) -> list[str]:
        """Collapse repeated catalog receipts into one numerical warning."""
        minimax = [warning for warning in self._warnings
                   if "minimax: served " in warning and
                   "UNCERTIFIED" in warning]
        others = [warning for warning in self._warnings
                  if warning not in minimax]
        if not minimax:
            return others

        def numbers(pattern: str) -> list[float]:
            values = []
            for warning in minimax:
                match = re.search(pattern, warning)
                if match is not None:
                    values.append(float(match.group(1)))
            return values

        targets = numbers(r"\btarget\s+([0-9.eE+-]+)")
        errors = numbers(r"\bmax_err\s+([0-9.eE+-]+)")
        nodes = [int(value) for value in
                 numbers(r"->\s+([0-9]+)\s+nodes")]
        detail = f"{len(minimax)} catalog entries used"
        if nodes:
            node_text = (str(nodes[0]) if min(nodes) == max(nodes) else
                         f"{min(nodes)}-{max(nodes)}")
            detail += f" ({node_text} nodes)"
        if targets:
            target_text = (f"{targets[0]:.5e}" if min(targets) == max(targets)
                           else f"{min(targets):.5e}-{max(targets):.5e}")
            detail += f"; requested tolerance {target_text}"
        if errors:
            detail += f"; worst reported error {max(errors):.5e}"
        return [
            "Minimax quadrature: " + detail + ". Catalog generator/backend "
            "provenance is UNCERTIFIED.",
            *others,
        ]

    def finish(self, *, status: str = "completed") -> None:
        self.warnings()
        self.emit()
        self.emit(f"LORRAX GW calculation {status}.")
        self.emit(f"Report written to {self.path}")
        self.close()


__all__ = ["GWProductionReport"]

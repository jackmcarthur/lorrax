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
    file_table_lines,
    numerical_environment_lines,
    policy,
    symmetry_sampling_lines,
)
from common.units import RYD_TO_EV


_WARNING_WORDS = (
    "WARNING", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
    "NOT TRUSTWORTHY", "DEPRECATED",
)


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
            cleaned = " ".join(text.split())
            if cleaned and cleaned not in self._warnings:
                self._warnings.append(cleaned)

    def heading(self, title: str) -> None:
        self.emit()
        self.emit(title.upper())
        self.emit("-" * len(title))

    def progress(self, line: str) -> None:
        """Write one deliberately selected rank-zero progress line."""
        self.emit(line)

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
        solver_text = {
            "one_shot_dft": "one-shot QP Hamiltonian at DFT energies",
            "fixed_point": "diagonal fixed-point QP solve",
            "self_consistent": "self-consistent QSGW",
        }.get(solver, solver)
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
        self.emit(f"Screening      : {diagram_text}; "
                  f"{getattr(screening, 'method', '-')} imaginary-axis quadrature")
        self.emit(f"Long wavelength: head={head_mode}; source={head_source}")
        self.emit(f"Coulomb system : {geometry}; Hartree source=" + policy(
            getattr(config, "hartree_source", "-"),
            ("auto", "stored", "isdf", "gspace")))
        self.emit("Spin channels  : " + (
            "charge + transverse current (bispinor)"
            if bool(getattr(config, "bispinor", False)) else "charge (scalar)"))
        self.emit("Degenerate sets: " + (
            "left in the input gauge" if bool(getattr(
                config, "no_degen_averaging", False))
            else f"averaged at {float(getattr(config, 'degen_avg_tol_ry', 0.0)):.5e} Ry"))
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
        self.emit(f"ISDF solve     : {config.backend.charge_zeta_solve} | "
                  "back-solve policy=" + policy(
                      config.backend.distributed_zeta_solve,
                      ("auto", "replicated", "per_q", "distributed")))
        self.emit(f"W Dyson solve  : {config.backend.w_dyson_solver} | "
                  "LU policy=" + policy(
                      config.backend.distributed_lu,
                      ("auto", "off", "cusolvermp", "scalapack")))
        self.emit("QP eigensolve  : " + policy(
            config.backend.eigh_backend,
            ("auto", "off", "distributed", "cusolvermp", "slate",
             "scalapack")))

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

    def sampling(self, *, wfn, sym) -> None:
        self.heading("Crystal symmetry and Brillouin-zone sampling")
        for line in symmetry_sampling_lines(wfn, sym, digits=FLOAT_DIGITS):
            self.emit(line)

    def bands(self, *, config, wfn, band_slices, zeta_ranges) -> None:
        b = band_slices
        n_e = float(getattr(wfn, "num_electrons", np.nan))
        self.heading("Band spaces and energy coverage")
        self.emit(f"Electrons      : {n_e:.5f}; occupied-band boundary = {b.b2}")
        self.emit(f"Occupied bands : {band_range(b.b0, b.b2)}")
        self.emit(f"QP valence     : {band_range(b.b1, b.b2)}")
        self.emit(f"QP conduction  : {band_range(b.b2, b.b3)}")
        self.emit(f"QP matrix      : {band_range(b.b0, b.b3)}")
        self.emit(f"chi0/W sum     : {band_range(b.b0, b.b4_chi)}")
        self.emit(f"Sigma sum      : {band_range(b.b0, b.b4_sigma)}")
        self.emit(f"Loaded ISDF psi: {band_range(b.b0, b.b4)}")
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
        grid_lo = float(config.sigma.omega_min_ev)
        grid_hi = float(config.sigma.omega_max_ev)
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
        omega_grid = np.asarray(
            getattr(sigma_result, "omega_grid_ev", ()), dtype=np.float64)
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
        for line in file_table_lines(rows):
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
        self.emit(f"QP gap         : {qp_gap:.5f} eV ({state})")
        self.emit(f"Gap correction : {qp_gap - dft_gap:+.5f} eV relative to DFT")
        self.emit("State energies  : written to the EQP files listed below")

    def timings(self, records, *, wall: float) -> None:
        """Print accumulated, non-overlapping major scientific stages."""
        def total(predicate):
            return sum(float(row["inclusive"]) for row in records
                       if predicate(row))

        stages = [
            ("zeta", total(lambda r: r["name"].startswith(
                "gw_jax.zeta_fit_chunked"))),
            ("V(q)", total(lambda r: r["name"] == "gw_jax.V_q_compute")),
            ("chi0", total(lambda r: r["name"].startswith("chi.") and
                           not any(p.startswith("chi.") for p in r["path"][:-1]))),
            ("W", total(lambda r: r["name"].startswith("W.") and
                        not any(p.startswith("W.") for p in r["path"][:-1]))),
            ("Sigma", total(lambda r: r["name"] in
                            ("gw_jax.sigma", "gw_jax.sc_driver"))),
        ]
        self.heading("Major-stage timing")
        self.emit("  stage             wall (s)     fraction")
        for name, seconds in stages:
            self.emit(f"  {name:<12} {seconds:14.5f}  "
                      f"{100.0 * seconds / wall if wall else 0.0:9.5f}%")
        self.emit(f"  {'total run':<12} {wall:14.5f}  {100.0:9.5f}%")

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

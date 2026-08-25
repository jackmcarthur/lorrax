"""Rank-zero, human-readable GW calculation report.

This is the scientific output of ``gwjax``.  It intentionally does not echo
library inventories, per-rank messages, HDF5 implementation details or the
driver's historical diagnostic prose.  Those remain available behind the one
driver-wide ``LORRAX_DEBUG_PRINT`` switch and in the launcher's own log.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from common.units import RYD_TO_EV


_WARNING_WORDS = (
    "WARNING", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
    "NOT TRUSTWORTHY", "UNCERTIFIED", "DEPRECATED",
)


def _fmt_range(lo: int, hi: int) -> str:
    """A human interval with its exact code-index convention."""
    if hi <= lo:
        return "none"
    return f"{lo + 1}-{hi}  (indices [{lo},{hi}))"


def _matrix_text(matrix) -> str:
    rows = np.asarray(matrix, dtype=np.int64).reshape(3, 3)
    return "[" + "; ".join(" ".join(f"{int(v):2d}" for v in row)
                            for row in rows) + "]"


def _abs(path: str | os.PathLike | None) -> str:
    return "-" if not path else os.path.abspath(os.fspath(path))


class GWProductionReport:
    """One clean GW report, owned and written only by process zero."""

    def __init__(self, path: str, *, runtime, debug: bool, stdout) -> None:
        self.path = _abs(path)
        self.runtime = runtime
        self.debug = bool(debug)
        self.stdout = stdout
        self.rank = int(getattr(runtime, "process_index", 0))
        self._warnings: list[str] = []
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

    def begin(self, *, input_file: str, config) -> None:
        self.emit("=" * 78)
        self.emit("LORRAX GW CALCULATION")
        self.emit("=" * 78)
        self.emit(f"Input          : {_abs(input_file)}")
        self.emit(f"Method         : {config.compute_mode.value} / "
                  f"{config.qp_solver.value}")

    def architecture(self) -> None:
        f = self.runtime.facts
        p = int(f.get("process_count", 1))
        ndev = int(f.get("n_devices", 0))
        nlocal = int(f.get("n_local_devices", 0))
        mesh = tuple(int(v) for v in f.get("mesh_shape", (1, 1)))
        threads = f.get("threads") or {}
        affinity = threads.get("affinity")
        thread_fields = []
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "LORRAX_MKLBLAS_THREADS"):
            value = threads.get(key)
            if value not in (None, ""):
                thread_fields.append(f"{key}={value}")

        self.heading("Processor architecture")
        self.emit(f"MPI ranks      : {p}")
        self.emit(f"Accelerators   : {ndev} {str(f.get('backend', 'unknown')).upper()} "
                  f"devices ({nlocal} per rank), {f.get('device_kind', 'unknown')}")
        self.emit(f"Processor mesh : {mesh[0]} x {mesh[1]}  "
                  f"(centroid axes X x Y)")
        self.emit(f"CPU allocation : {affinity if affinity is not None else '?'} "
                  f"cores per rank")
        self.emit("Host threads   : " + (", ".join(thread_fields)
                                         if thread_fields else "runtime defaults"))

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
        self.emit(f"Coulomb system : {geometry}; Hartree source="
                  f"{getattr(config, 'hartree_source', '-')}")
        self.emit("Spin channels  : " + (
            "charge + transverse current (bispinor)"
            if bool(getattr(config, "bispinor", False)) else "charge (scalar)"))
        self.emit("Degenerate sets: " + (
            "left in the input gauge" if bool(getattr(
                config, "no_degen_averaging", False))
            else f"averaged at {float(getattr(config, 'degen_avg_tol_ry', 0.0)):g} Ry"))
        self.emit("ISDF state     : " + (
            "restart requested" if bool(getattr(config, "restart", False))
            else "fresh fit requested"))

    def environment(self, *, config, wfn) -> None:
        f = self.runtime.facts
        backend = str(f.get("backend", "unknown")).lower()
        platform = "CUDA" if backend in ("gpu", "cuda") else backend
        precision = "FP64 / complex128" if f.get("x64") else "FP32 / complex64"
        collectives = "NCCL" if backend in ("gpu", "cuda", "rocm") else str(
            (f.get("collectives") or {}).get("impl", "local")).upper()

        self.heading("Numerical environment")
        self.emit(f"JAX/JAXLIB     : {f.get('jax_version', 'unknown')} / "
                  f"{f.get('jaxlib_version', 'unknown')} | "
                  f"{precision} | {collectives} collectives")
        self.emit(f"Wavefunctions  : {getattr(wfn, 'backend', 'unknown')} reader")
        self.emit(f"ISDF solve     : {config.backend.charge_zeta_solve} | "
                  f"back-solve policy={config.backend.distributed_zeta_solve}")
        self.emit(f"W Dyson solve  : {config.backend.w_dyson_solver} | "
                  f"LU policy={config.backend.distributed_lu}")
        self.emit(f"QP eigensolve  : {config.backend.eigh_backend}")

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
                implementation = str(dial.get("target") or "platform FFI kernel")
            else:
                state = "OFF"
                implementation = str(dial.get("off_label") or "default lowering")
            self.emit(f"{name:<26} = {state:<6} : "
                      f"{descriptions[name]} ({implementation})")

    def sampling(self, *, wfn, sym) -> None:
        n_sym = int(np.asarray(sym.Rinv_grid).shape[0])
        tau = np.asarray(sym.translations, dtype=np.float64)[:n_sym] / (2.0 * np.pi)
        weights = np.asarray(wfn.kweights, dtype=np.float64)
        weights = weights / float(np.sum(weights))

        self.heading("Crystal symmetry and Brillouin-zone sampling")
        self.emit("Real-space action: r' = R^-1 r + tau")
        self.emit(f"Spatial operations: {n_sym}; time reversal: "
                  f"{'used' if bool(sym.trs_allowed) else 'not used'}")
        for i, (rotation, shift) in enumerate(zip(sym.Rinv_grid, tau), start=1):
            self.emit(f"  S{i:02d}  R^-1={_matrix_text(rotation)}  "
                      f"tau=({shift[0]: .8f} {shift[1]: .8f} {shift[2]: .8f})")

        kgrid = tuple(int(v) for v in np.asarray(wfn.kgrid).reshape(-1)[:3])
        shift = tuple(float(v) for v in np.asarray(wfn.shift).reshape(-1)[:3])
        self.emit(f"Full BZ grid   : {int(sym.nk_tot)} k points | mesh "
                  f"{kgrid[0]} x {kgrid[1]} x {kgrid[2]} | "
                  f"shift ({shift[0]:g}, {shift[1]:g}, {shift[2]:g})")
        self.emit(f"Stored IBZ     : {int(sym.nk_red)} k points")
        self.emit("  ik        kx           ky           kz          weight")
        for ik, (point, weight) in enumerate(zip(np.asarray(wfn.kpoints), weights),
                                                start=1):
            self.emit(f"  {ik:3d}  {point[0]: .9f}  {point[1]: .9f}  "
                      f"{point[2]: .9f}  {weight: .10f}")

    def bands(self, *, config, wfn, band_slices, zeta_ranges) -> None:
        b = band_slices
        n_e = float(getattr(wfn, "num_electrons", np.nan))
        self.heading("Band spaces and energy coverage")
        self.emit(f"Electrons      : {n_e:.8g}; occupied-band boundary = {b.b2}")
        self.emit(f"Occupied bands : {_fmt_range(b.b0, b.b2)}")
        self.emit(f"QP valence     : {_fmt_range(b.b1, b.b2)}")
        self.emit(f"QP conduction  : {_fmt_range(b.b2, b.b3)}")
        self.emit(f"QP matrix      : {_fmt_range(b.b0, b.b3)}")
        self.emit(f"chi0/W sum     : {_fmt_range(b.b0, b.b4_chi)}")
        self.emit(f"Sigma sum      : {_fmt_range(b.b0, b.b4_sigma)}")
        self.emit(f"Loaded ISDF psi: {_fmt_range(b.b0, b.b4)}")
        self.emit(f"zeta fit       : left {_fmt_range(*zeta_ranges[0])}; "
                  f"right {_fmt_range(*zeta_ranges[1])}")

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
        self.emit(f"Energy origin   : E_F = {ef_ev:+.6f} eV ({provenance})")
        omega_grid = np.asarray(
            getattr(sigma_result, "omega_grid_ev", ()), dtype=np.float64)
        grid_note = (f"; step={float(config.sigma.omega_step_ev):g} eV; "
                     f"{int(omega_grid.size)} points")
        self.emit(f"Sigma omega    : [{grid_lo:+.3f}, {grid_hi:+.3f}] eV "
                  f"relative to E_F{grid_note}")
        self.emit(f"Absolute window: [{ef_ev + grid_lo:+.3f}, "
                  f"{ef_ev + grid_hi:+.3f}] eV")
        if val_span is not None:
            self.emit(f"DFT QP valence : [{val_span[0]:+.3f}, "
                      f"{val_span[1]:+.3f}] eV relative to E_F")
        if cond_span is not None:
            self.emit(f"DFT QP conduct.: [{cond_span[0]:+.3f}, "
                      f"{cond_span[1]:+.3f}] eV relative to E_F")
        if target:
            target_lo, target_hi = min(target), max(target)
            self.emit(f"Omega margins  : {target_lo - grid_lo:+.3f} eV below; "
                      f"{grid_hi - target_hi:+.3f} eV above protected DFT states")

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
        self.heading("Files")
        self.emit("  role                     state       path")
        for role, state, path in rows:
            self.emit(f"  {role:<24} {state:<11} {_abs(path)}")

    def qp_energies(self, *, wfn, sym, band_slices, e_dft_ry, e_qp_ry) -> None:
        b = band_slices
        rows = np.asarray(sym.kirr_fullids, dtype=np.int64)
        dft = np.asarray(e_dft_ry, dtype=np.float64)[rows] * RYD_TO_EV
        qp = np.asarray(e_qp_ry, dtype=np.float64)[rows] * RYD_TO_EV
        i0, i1 = b.b1 - b.b0, b.b3 - b.b0
        self.heading("Quasiparticle energies")
        self.emit("  ik  band      E_DFT (eV)       E_QP (eV)     Delta (eV)")
        for ik in range(dft.shape[0]):
            for local in range(i0, i1):
                self.emit(f"  {ik + 1:3d} {b.b0 + local + 1:5d}  "
                          f"{dft[ik, local]:14.6f}  {qp[ik, local]:14.6f}  "
                          f"{qp[ik, local] - dft[ik, local]:12.6f}")

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
            self.emit(f"  {name:<12} {seconds:14.3f}  "
                      f"{100.0 * seconds / wall if wall else 0.0:9.2f}%")
        self.emit(f"  {'total run':<12} {wall:14.3f}  {100.0:9.2f}%")

    def finish(self, *, status: str = "completed") -> None:
        if self._warnings:
            self.heading("Warnings")
            for warning in self._warnings:
                self.emit(f"  {warning}")
        self.emit()
        self.emit(f"LORRAX GW calculation {status}.")
        self.emit(f"Report written to {self.path}")
        self.close()


__all__ = ["GWProductionReport"]

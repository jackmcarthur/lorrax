"""Rank-zero scientific report for Hamiltonian interpolation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from common.scientific_output import (
    FLOAT_DIGITS,
    abs_path,
    architecture_lines,
    band_range,
    centroid_orbit_line,
    clean_rounded,
    file_table_lines,
    htransform_quality_lines,
    numerical_environment_lines,
    policy,
    spectral_compression_lines,
    symmetry_sampling_lines,
)
from common.units import RYD_TO_EV


_WARNING_WORDS = (
    "WARNING", "[WARN]", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
    "NOT TRUSTWORTHY", "DEPRECATED",
)


class HTransformProductionReport:
    """One human-readable htransform stream, written only by process zero."""

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
        if self.rank != 0:
            return
        text = str(line)
        self.stdout(text)
        if self._stream is not None:
            self._stream.write(text + "\n")

    def progress(self, line: str) -> None:
        self.emit(line)

    def legacy_print(self, *args, sep: str = " ", **kwargs) -> None:
        text = sep.join(str(v) for v in args)
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

    def begin(self, *, input_file: str, output_file: str,
              energy_source: str) -> None:
        self.emit("=" * 78)
        self.emit("LORRAX HAMILTONIAN INTERPOLATION")
        self.emit("=" * 78)
        self.emit(f"Input          : {abs_path(input_file)}")
        self.emit(f"Band output    : {abs_path(output_file)}")
        self.emit(f"Energy source  : {energy_source}")

    def architecture(self) -> None:
        self.heading("Processor architecture")
        for line in architecture_lines(self.runtime):
            self.emit(line)

    def environment(self, *, params, wfn, gram_plan, fine_plan,
                    fine_enabled: bool) -> None:
        self.heading("Numerical environment")
        for line in numerical_environment_lines(self.runtime):
            self.emit(line)
        self.emit(f"Wavefunctions  : {getattr(wfn, 'backend', 'unknown')} reader")
        requested_eigh = resolve_text(params.get("eigh_backend", "auto"))
        eigh_text = policy(
            requested_eigh,
            ("auto", "off", "distributed", "cusolvermp", "slate",
             "scalapack"))
        if requested_eigh.strip().lower() != str(gram_plan.requested).lower():
            eigh_text += f" -> {gram_plan.requested}"
        if str(gram_plan.requested).lower() != str(gram_plan.backend).lower():
            eigh_text += f" -> {gram_plan.backend}"
        geometry = ("replicated native JAX" if gram_plan.is_native else
                    f"one matrix tile over {int(gram_plan.mesh.shape['x'])} x "
                    f"{int(gram_plan.mesh.shape['y'])} ranks")
        self.emit(f"Gram eigensolve: {eigh_text}; {geometry}; n={int(gram_plan.n)}")
        if fine_enabled:
            fine_text = policy(
                fine_plan.requested,
                ("auto", "off", "distributed", "cusolvermp", "slate",
                 "scalapack"))
            if str(fine_plan.requested) != str(fine_plan.backend):
                fine_text += f" -> {fine_plan.backend}"
            self.emit("Fine-k eigensolve: " + fine_text
                      + "; matrix extent follows retained Galerkin rank")
            batch_text = policy(
                fine_plan.requested_batched_route,
                ("auto", "batch_reshard"))
            if str(fine_plan.requested_batched_route) != str(fine_plan.batched_route):
                batch_text += f" -> {fine_plan.batched_route}"
            self.emit("Batched LA     : " + batch_text)
        else:
            self.emit("Fine-k eigensolve: not used (get_centroids_fi = false)")
            self.emit("Batched LA     : not used by this calculation")
        self.emit("LA alternatives : native JAX; cuSOLVERMp on GPU; SLATE or "
                  "ScaLAPACK on CPU. Batched LA is a schedule, not a backend")
        self.emit("Debug stream   : " + (
            "ON via LORRAX_DEBUG_PRINT" if self.debug else
            "OFF (set LORRAX_DEBUG_PRINT=1 for kernel diagnostics)"))

    def sampling(self, *, wfn, sym, centroids=None) -> None:
        self.heading("Crystal symmetry and Brillouin-zone sampling")
        for line in symmetry_sampling_lines(wfn, sym, digits=FLOAT_DIGITS):
            self.emit(line)
        if centroids is not None:
            self.emit(centroid_orbit_line(centroids))

    def interpolation_space(self, *, params, wfn, meta, result,
                            enk_sigma_ry, ctilde, centroid_file: str,
                            energy_source: str, centroids=None) -> None:
        start = int(result["band_start"])
        keep = int(result["nb_keep"])
        fit = int(result["nb_fit"])
        energies = np.asarray(enk_sigma_ry, dtype=np.float64)
        finite = energies[np.isfinite(energies)]
        n_electrons = float(getattr(
            wfn, "num_electrons", getattr(wfn, "nelec", np.nan)))
        self.heading("Interpolation and band spaces")
        self.emit(f"Electrons      : {n_electrons:.5f}; "
                  f"occupied-band boundary = {int(getattr(wfn, 'nelec', 0))}")
        self.emit(f"Returned bands : {band_range(start, start + keep)}")
        self.emit(f"Fitted bands   : {band_range(start, start + fit)} "
                  f"({int(result['n_guard_bands'])} upper guard bands)")
        self.emit(f"Source states  : {energy_source}; "
                  f"{int(energies.size)} k-resolved band energies")
        if finite.size:
            finite_ev = finite * RYD_TO_EV
            self.emit(f"Source E range : [{float(np.min(finite_ev)):.5f}, "
                      f"{float(np.max(finite_ev)):.5f}] eV; span "
                      f"{float(np.ptp(finite_ev)):.5f} eV")
        if fit > keep and energies.ndim == 2 and energies.shape[0] >= fit:
            guard_block_ev = energies[keep:fit] * RYD_TO_EV
            self.emit(
                f"Guard bands    : {band_range(start + keep, start + fit)}; "
                f"E range [{float(np.min(guard_block_ev)):.5f}, "
                f"{float(np.max(guard_block_ev)):.5f}] eV; span "
                f"{float(np.ptp(guard_block_ev)):.5f} eV")
            requested_top = energies[keep - 1]
            fitted_top = energies[fit - 1]
            guard_ev = (
                float(np.min(fitted_top)) - float(np.max(requested_top))) \
                * RYD_TO_EV
            vertical_ev = float(np.min(fitted_top - requested_top)) * RYD_TO_EV
            relation = "separated" if guard_ev >= 0.0 else "ranges overlap"
            self.emit(
                f"Guard buffer   : {vertical_ev:.5f} eV minimum same-k "
                f"separation, E(band {start + fit}) - E(band "
                f"{start + keep})")
            self.emit(
                f"Global guard   : {guard_ev:+.5f} eV = min E(band "
                f"{start + fit}) - max E(band {start + keep}); {relation}")
        self.emit(f"Centroid file  : {abs_path(centroid_file)}")
        if centroids is None:
            self.emit(f"Centroid sites : {int(meta.n_rmu)} requested/loaded")
        elif int(centroids.source_n_rmu) == int(centroids.n_rmu):
            self.emit(
                f"Centroid sites : {int(centroids.n_rmu)} requested from "
                "the full table")
        else:
            self.emit(
                f"Centroid sites : requested subset {int(centroids.n_rmu)} "
                f"of {int(centroids.source_n_rmu)} source rows")
        self.emit(f"Galerkin basis : rank {int(ctilde.shape[2])} shared across "
                  f"{int(result['nk_total'])} coarse-grid k points")
        multiplier = float(params.get("htransform_rank_multiplier", 0.0))
        self.emit("Rank policy    : " + (
            "full numerical span" if multiplier == 0.0 else
            f"reduced target {multiplier:.5f} x bands"))
        transform = result.get("f_transform")
        if transform is not None:
            scale_band = start + int(transform["scale_band_local"]) + 1
            shoulder_band = start + int(transform["shoulder_band_local"]) + 1
            self.emit(
                f"f-transform    : a={float(transform['a_ry']) * RYD_TO_EV:.5f} eV "
                f"(4 x bandwidth of band {scale_band}); "
                f"n={float(transform['n']):.2f}")
            self.emit(
                f"Zero shoulder  : max E(band {shoulder_band}) = "
                f"{float(transform['shift_ry']) * RYD_TO_EV:.5f} eV; "
                "the full top-band "
                "dispersion lies at or below the cutoff")

    def spectral_compression(self, receipt) -> None:
        self.heading("Spectral compression and validation")
        for line in spectral_compression_lines(receipt):
            self.emit(line)

    def htransform_quality(self, receipt) -> None:
        self.heading("Real-space locality and interpolation risk")
        for line in htransform_quality_lines(receipt):
            self.emit(line)

    def path_summary(self, *, result) -> None:
        kpath, _, node_indices, node_labels, _ = result["kpath_data"]
        self.heading("Interpolation path")
        if kpath is None:
            self.emit("Path           : none; no K_POINTS {crystal_b} path "
                      "was supplied")
            return
        points = clean_rounded(kpath, digits=FLOAT_DIGITS)
        self.emit(f"Path sampling  : {int(points.shape[0])} points; "
                  f"{len(node_indices)} labelled nodes")
        for number, (idx, label) in enumerate(
                zip(node_indices, node_labels), start=1):
            point = points[int(idx)]
            self.emit(f"  N{number:02d}  {label or '-':>3}  "
                      f"k=({point[0]: .5f} {point[1]: .5f} "
                      f"{point[2]: .5f})  path index {int(idx) + 1}")
        for number in range(len(node_indices) - 1):
            lo = int(node_indices[number])
            hi = int(node_indices[number + 1])
            intervals = hi - lo
            left = node_labels[number] or f"N{number + 1}"
            right = node_labels[number + 1] or f"N{number + 2}"
            self.emit(
                f"  L{number + 1:02d}  {left} -> {right}: {intervals} "
                f"intervals; {intervals + 1} endpoint-inclusive points")
        path_range = result.get("path_range")
        if path_range is not None:
            self.emit(
                f"Energy range   : "
                f"[{float(path_range[0]) * RYD_TO_EV:+.5f}, "
                f"{float(path_range[1]) * RYD_TO_EV:+.5f}] eV relative to VBM")

    def timings(self, records, *, wall: float) -> None:
        def total(name: str) -> float:
            return sum(float(row["inclusive"]) for row in records
                       if row["name"] == name)

        stages = [
            ("runtime bring-up", sum(
                float(row["inclusive"]) for row in records
                if row["name"].startswith("htransform.runtime_stack."))),
            ("pre-main + imports", total("htransform.imports")),
            ("input + Galerkin", total("initialize_wfns")),
            ("fH construction", total("ht.build_fH_R")),
            ("path eigensolve", total("ht.kpath_loop")),
            ("path recovery", total("ht.post_kpath")
             + total("ht.kpath_host_tail")),
            ("fine-k wavefunctions", total("wfns_fi")),
        ]
        accounted = sum(seconds for _, seconds in stages)
        stages.append(("other driver work", max(float(wall) - accounted, 0.0)))
        self.heading("Major-stage timing")
        self.emit("  stage                    wall (s)     fraction")
        for name, seconds in stages:
            self.emit(f"  {name:<22} {seconds:14.2f}  "
                      f"{100.0 * seconds / wall if wall else 0.0:9.2f}%")
        self.emit(f"  {'total run':<22} {float(wall):14.2f}  "
                  f"{100.0:9.2f}%")

    def warnings(self) -> None:
        if self._warnings_emitted:
            return
        self._warnings_emitted = True
        if self._warnings:
            self.heading("Warnings")
            for warning in self._warnings:
                self.emit(f"  {warning}")

    def files(self, rows: Iterable[tuple[str, str, str]]) -> None:
        self.heading("Output files and inputs")
        for line in file_table_lines(rows, omit_paths=(self.path,)):
            self.emit(line)

    def finish(self, *, status: str = "completed") -> None:
        self.warnings()
        self.emit()
        self.emit(f"LORRAX Hamiltonian interpolation {status}.")
        self.emit(f"Report written to {self.path}")
        self.close()


def resolve_text(value) -> str:
    """Enum-or-string display without importing a driver resolver."""
    return str(getattr(value, "value", value))


__all__ = ["HTransformProductionReport"]

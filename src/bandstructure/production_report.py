"""Rank-zero scientific report for Hamiltonian interpolation."""

from __future__ import annotations

import numpy as np

from common.production_report import ProductionReport
from common.report_blocks import kv_lines
from common.scientific_output import (
    FLOAT_DIGITS,
    abs_path,
    band_range,
    clean_rounded,
    policy,
)
from common.units import RYD_TO_EV


class HTransformProductionReport(ProductionReport):
    """One human-readable htransform stream, written only by process zero."""

    driver = "htransform"
    purpose = "Hamiltonian interpolation"

    def begin(self, *, input_file: str, output_file: str,
              energy_source: str) -> None:
        """The shared header plus the energy source (the band file is an output row)."""
        super().begin(input_file=input_file)
        self.emit_lines(kv_lines([("Energy source", energy_source)], indent=0))

    def environment(self, *, params, wfn, fine_plan,
                    fine_enabled: bool) -> None:
        self.heading("Numerical environment")
        self.emit(f"Wavefunctions  : {getattr(wfn, 'backend', 'unknown')} reader")
        self.emit(
            "Galerkin basis : nodal, orbit-closed nodes (the ISDF centroids up "
            "to 1.25 x 22 x fitted bands, else 22 x fitted bands by k-means); "
            "quadrature metric + Loewdin; zero-weight nodes dropped")
        if fine_enabled:
            fine_text = policy(
                fine_plan.requested,
                ("auto", "off", "distributed", "cusolvermp", "slate",
                 "scalapack"))
            if str(fine_plan.requested) != str(fine_plan.backend):
                fine_text += f" -> {fine_plan.backend}"
            from bandstructure.fh_interp import TOP_BLOCK_PAD
            self.emit(f"Fine-k eigensolve: top block (fitted bands + "
                      f"{TOP_BLOCK_PAD}) by subspace iteration; operand "
                      f"layout " + fine_text)
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
        self.emit(f"Galerkin basis : nodal rank {int(ctilde.shape[2])} shared "
                  f"across {int(result['nk_total'])} coarse-grid k points")
        self.emit("Rank policy    : rank = spin components x nodes; nodes "
                  "= the ISDF centroids, or k-means above 1.25 x 22 x fitted "
                  "bands; zero-weight nodes dropped")
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
        super().timings(stages, wall=wall)


def resolve_text(value) -> str:
    """Enum-or-string display without importing a driver resolver."""
    return str(getattr(value, "value", value))


__all__ = ["HTransformProductionReport"]

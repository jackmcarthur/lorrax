"""Rank-zero scientific report for Hamiltonian interpolation.

The htransform main log is the shared header, DEVICES, SYMMETRY and K POINTS,
three ``[i/3]`` stages (wavefunctions and Galerkin basis, the interpolation
along the k path, the outputs) with a few ``key   value`` rows each, then
RESULTS (:func:`htransform_results_lines`, a pure function of host values)
and the shared footer.  The numerical environment and the full htransform
receipts go to the JSON sidecar.
"""

from __future__ import annotations

import os

import numpy as np

from common.log import get_logger, record
from common.preprocessing_output import (galerkin_rows, locality_rows,
                                         results_lines, stage_row)
from common.production_report import ProductionReport
from common.report_blocks import kv_lines
from common.scientific_output import (
    FLOAT_DIGITS,
    band_range,
    clean_rounded,
    htransform_quality_lines,
    policy,
    spectral_compression_lines,
)
from common.units import RYD_TO_EV

log = get_logger(__name__)

#: Bands the RESULTS band table lists in full; a wider window lists the six
#: on each side of the VBM.
BAND_ROWS_MAX = 24
BAND_ROWS_EDGE = 6


# ── pure blocks ──────────────────────────────────────────────────────────────
def centroid_orbit_text(verdict) -> str:
    """The ``centroid orbit`` row from a centroid closure verdict."""
    n_ops, n_bad = int(verdict.n_sym), int(verdict.n_violating)
    sites = int(verdict.n_centroids)
    if bool(verdict.closed):
        return f"closed: {n_ops - n_bad}/{n_ops} operations map the {sites} sites onto themselves"
    return (f"NOT closed: {n_bad}/{n_ops} operations move the {sites}-site set; "
            f"worst residual {float(verdict.worst_residual):.1e} at "
            f"S{int(verdict.worst_op) + 1:02d}")


def path_text(kpath, node_indices, node_labels) -> str:
    """The k path as a stage row value: the count, then one line per node."""
    if kpath is None:
        return "none; no K_POINTS {crystal_b} path was supplied"
    points = clean_rounded(kpath, digits=FLOAT_DIGITS)
    lines = [f"{int(points.shape[0])} points, {len(node_indices)} labelled nodes"]
    for number, (idx, label) in enumerate(zip(node_indices, node_labels), start=1):
        point = points[int(idx)]
        lines.append(f"N{number:02d}  {label or '-':>3}  k = ({point[0]: .5f} "
                     f"{point[1]: .5f} {point[2]: .5f})  path index {int(idx) + 1}")
    return "\n".join(lines)


def guard_rows(energies_ry, *, start: int, keep: int, fit: int) -> list[tuple[str, str]]:
    """The guard-band rows: the guard block's range and its separation from
    the top returned band, on the coarse-grid energies ``(nb, nk)`` in Ry."""
    energies = np.asarray(energies_ry, dtype=np.float64)
    if not (fit > keep and energies.ndim == 2 and energies.shape[0] >= fit):
        return []
    guard_ev = energies[keep:fit] * RYD_TO_EV
    requested_top, fitted_top = energies[keep - 1], energies[fit - 1]
    overlap_ev = (float(np.min(fitted_top)) - float(np.max(requested_top))) * RYD_TO_EV
    vertical_ev = float(np.min(fitted_top - requested_top)) * RYD_TO_EV
    return [("guard bands",
             f"{band_range(start + keep, start + fit)}: {float(np.min(guard_ev)):.5f} .. "
             f"{float(np.max(guard_ev)):.5f} eV on the coarse grid; at least "
             f"{vertical_ev:.5f} eV above band {start + keep} at the same k"
             + ("" if overlap_ev >= 0.0 else
                f"; the ranges overlap by {-overlap_ev:.5f} eV"))]


def _zero_text(zero: str) -> str:
    """``the VBM`` or ``E_F (DFT)``: the zero of the path energies."""
    return "the VBM" if zero == "VBM" else str(zero)


def band_table_lines(energies_ev, *, band_start: int, n_val: int,
                     max_rows: int = BAND_ROWS_MAX,
                     edge: int = BAND_ROWS_EDGE, zero: str = "VBM") -> list[str]:
    """Each returned band's range on the path, eV relative to ``zero``: the
    VBM (its band and the CBM's tagged), or a metal's ``"E_F (DFT)"``."""
    energies = np.atleast_2d(np.asarray(energies_ev, dtype=np.float64))
    nb = energies.shape[1]
    rows = list(range(nb))
    if nb > max_rows:
        vbm = int(n_val) - 1
        rows = [b for b in rows if vbm - edge < b <= vbm + edge]
    lines = [f"  {'band':>6}{'min (eV)':>12}{'max (eV)':>12}{'width (eV)':>12}"
             f"   (on the path, relative to {_zero_text(zero)})"]
    for b in rows:
        lo, hi = float(np.min(energies[:, b])), float(np.max(energies[:, b]))
        tag = ("" if zero != "VBM" else "  VBM" if b == int(n_val) - 1
               else "  CBM" if b == int(n_val) else "")
        lines.append(f"  {band_start + b + 1:>6}{lo:>+12.5f}{hi:>+12.5f}{hi - lo:>12.5f}{tag}")
    if len(rows) < nb:
        lines.append(f"  [{edge} bands on each side of the VBM; all {nb} are in the "
                     "band file]")
    return lines


def htransform_results_lines(*, energies_ev, band_start: int, n_val: int,
                             n_keep: int, n_fit: int, on_grid=None,
                             guard: list[tuple[str, str]] = (),
                             energy_source: str, written: str,
                             zero: str = "VBM") -> list[str]:
    """The htransform RESULTS block.

    ``energies_ev`` are the returned bands on the path ``(nk_path, n_keep)``,
    relative to ``zero`` (the VBM, or a metal's ``"E_F (DFT)"``); ``on_grid`` is ``(n_points, max |dE| eV, RMS eV)``
    over the path points that lie on the coarse grid (``None``: the path has
    none, so the interpolation has no on-grid check).
    """
    rows = []
    if on_grid is not None and int(on_grid[0]) > 0:
        rows.append(("on-grid error", f"max |dE| {float(on_grid[1]) * 1e3:.4f} meV, RMS "
                                      f"{float(on_grid[2]) * 1e3:.4f} meV over "
                                      f"{int(on_grid[0])} path point"
                                      f"{'' if int(on_grid[0]) == 1 else 's'} on the "
                                      "coarse grid"))
    else:
        rows.append(("on-grid error", "no path point lies on the coarse grid"))
    rows.append(("bands", f"{band_range(band_start, band_start + n_keep)} returned, "
                          f"{band_range(band_start, band_start + n_fit)} fitted "
                          f"({int(n_fit) - int(n_keep)} guard)"))
    rows += list(guard)
    energies = np.asarray(energies_ev, dtype=np.float64)
    if energies.size:
        rows.append(("energy range", f"{float(energies.min()):+.5f} .. "
                                     f"{float(energies.max()):+.5f} eV relative to "
                                     f"{_zero_text(zero)}"))
    rows.append(("energy source", energy_source))
    rows.append(("written", written))
    table = (band_table_lines(energies, band_start=band_start, n_val=n_val,
                              zero=zero)
             if energies.size else [])
    return results_lines(rows, table=table)


def record_receipts(key: str, *, rank=None, quality=None) -> None:
    """The full Galerkin and locality receipts, for the JSON sidecar."""
    if rank is not None:
        log.debug("htransform spectral compression", extra=record(
            f"{key}.spectral_compression", lines=spectral_compression_lines(rank)))
    if quality is not None:
        log.debug("htransform interpolation quality", extra=record(
            f"{key}.quality", lines=htransform_quality_lines(quality)))


# ── the report ───────────────────────────────────────────────────────────────
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
        """Reader, Galerkin policy and fine-k plan: JSON sidecar only."""
        lines = [
            "Galerkin basis: nodal, orbit-closed nodes (the ISDF centroids up "
            "to 1.25 x 22 x fitted bands, else 22 x fitted bands by k-means); "
            "quadrature metric + Loewdin; zero-weight nodes dropped"]
        if fine_enabled:
            fine_text = policy(
                fine_plan.requested,
                ("auto", "off", "distributed", "cusolvermp", "slate",
                 "scalapack"))
            if str(fine_plan.requested) != str(fine_plan.backend):
                fine_text += f" -> {fine_plan.backend}"
            from bandstructure.fh_interp import TOP_BLOCK_PAD
            lines.append(f"Fine-k eigensolve: top block (fitted bands + "
                         f"{TOP_BLOCK_PAD}) by subspace iteration; operand "
                         f"layout " + fine_text)
            batch_text = policy(
                fine_plan.requested_batched_route,
                ("auto", "batch_reshard"))
            if str(fine_plan.requested_batched_route) != str(fine_plan.batched_route):
                batch_text += f" -> {fine_plan.batched_route}"
            lines.append("Batched LA: " + batch_text)
        else:
            lines.append("Fine-k eigensolve: not used (get_centroids_fi = false)")
        log.debug("numerical environment", extra=record(
            "environment",
            wavefunction_reader=str(getattr(wfn, "backend", "unknown")),
            lines=lines))

    def setup_rows(self, *, centroid_file: str, centroids, n_rmu: int,
                   rank_record) -> None:
        """Stage 1's rows: the centroid table, its orbit closure, the basis."""
        if centroids is None:
            sites = f"{int(n_rmu)} requested/loaded"
        elif int(centroids.source_n_rmu) == int(centroids.n_rmu):
            sites = f"{int(centroids.n_rmu)}, the full table"
        else:
            sites = (f"subset {int(centroids.n_rmu)} of "
                     f"{int(centroids.source_n_rmu)} rows")
        stage_row("centroids", f"{sites} of {os.path.basename(str(centroid_file))}")
        if centroids is not None and getattr(centroids, "closure", None) is not None:
            stage_row("centroid orbit", centroid_orbit_text(centroids.closure))
        record_receipts("htransform", rank=rank_record)
        for label, value in galerkin_rows(rank_record):
            stage_row(label, value)

    def transform_rows(self, *, result, quality_record) -> None:
        """Stage 2's rows: the f-transform, the locality receipt, the path."""
        start = int(result["band_start"])
        transform = result.get("f_transform")
        if transform is not None:
            scale_band = start + int(transform["scale_band_local"]) + 1
            shoulder_band = start + int(transform["shoulder_band_local"]) + 1
            stage_row("f-transform",
                      f"a = {float(transform['a_ry']) * RYD_TO_EV:.5f} eV (4 x the "
                      f"bandwidth of band {scale_band}), n = {float(transform['n']):.2f}; "
                      f"zero shoulder at max E(band {shoulder_band}) = "
                      f"{float(transform['shift_ry']) * RYD_TO_EV:.5f} eV")
        record_receipts("htransform", quality=quality_record)
        for label, value in locality_rows(quality_record):
            stage_row(label, value)
        kpath, _, node_indices, node_labels, _ = result["kpath_data"]
        stage_row("k path", path_text(kpath, node_indices, node_labels))

    def results(self, *, result, enk_sigma_ry, n_val: int, energy_source: str,
                written: str) -> None:
        """RESULTS from the interpolation's own host outputs."""
        start, keep, fit = (int(result["band_start"]), int(result["nb_keep"]),
                            int(result["nb_fit"]))
        energies = result.get("energies_sorted")
        energies_ev = (np.zeros((0, keep)) if energies is None
                       else np.asarray(energies, dtype=np.float64) * RYD_TO_EV)
        n_on = int(np.size(result.get("coincident_path_indices", ())))
        on_grid = None
        if n_on and result.get("coincident_max_abs_ry") is not None:
            on_grid = (n_on, float(result["coincident_max_abs_ry"]) * RYD_TO_EV,
                       float(result["coincident_rms_ry"]) * RYD_TO_EV)
        self.emit_lines(htransform_results_lines(
            energies_ev=energies_ev, band_start=start, n_val=int(n_val),
            n_keep=keep, n_fit=fit, on_grid=on_grid,
            guard=guard_rows(enk_sigma_ry, start=start, keep=keep, fit=fit),
            energy_source=energy_source, written=written,
            zero=str(result.get("energy_zero", "VBM"))))

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


__all__ = [
    "HTransformProductionReport",
    "band_table_lines",
    "centroid_orbit_text",
    "guard_rows",
    "htransform_results_lines",
    "path_text",
]

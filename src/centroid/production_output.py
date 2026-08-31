"""Pure formatting helpers for the production centroid coordinate file."""

from __future__ import annotations

import os

from common.scientific_output import (
    architecture_lines,
    numerical_environment_lines,
)


def prune_band_ranges(args, n_val: int, n_cond: int):
    """Resolved pair-density windows, shared by pruning and provenance."""
    top = int(n_val) + int(n_cond)
    fit_window = getattr(args, "fit_window", None)
    if fit_window is not None:
        if args.prune_window != "v_x_vc":
            raise ValueError(
                "--fit-window cannot be combined with a non-default "
                "--prune-window")
        try:
            fields = str(fit_window).split(",")
            if len(fields) != 2:
                raise ValueError
            left = tuple(int(v) for v in fields[0].split(":"))
            right = tuple(int(v) for v in fields[1].split(":"))
            if len(left) != 2 or len(right) != 2:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--fit-window must be L0:L1,R0:R1 using integer, 0-based "
                "half-open bounds") from exc
        if any(lo < 0 or hi <= lo or hi > top
               for lo, hi in (left, right)):
            raise ValueError(
                f"--fit-window ranges must be nonempty and lie in [0,{top}); "
                f"got left={left}, right={right}")
        return left, right, "explicit feature pair"
    if args.prune_window == "v_x_c":
        return (0, int(n_val)), (int(n_val), top), "valence x conduction"
    if args.prune_window == "vc_x_vc":
        return (0, top), (0, top), "full protected square"
    return (0, int(n_val)), (0, top), "valence x (valence + conduction)"


def format_centroid_header(*, density_fit: str, source_wfn: str,
                           weight_label: str, num_electrons: float,
                           occupied_boundary: int, fft_grid, kgrid, shift,
                           seed: int, rho_power: float, requested: int,
                           candidates: int, written: int, pruning: str,
                           prune_rank: int | None, prune_left, prune_right,
                           prune_label: str, orbit_aware: bool, n_sym: int,
                           density_mode: str) -> str:
    """Centroid-table provenance, kept pure so the file contract is gated."""
    rank_note = ("" if prune_rank is None else
                 f"; achieved numerical rank={int(prune_rank)}")
    channel = ("gamma^0 (charge) ISDF" if density_mode == "scalar" else
               "gamma^{1,2,3} (current) ISDF")
    return (
        "LORRAX ISDF centroid coordinates (fractional crystal units)\n"
        f"density fit: {density_fit}\n"
        f"source wavefunctions: {os.path.abspath(source_wfn)}\n"
        f"source density: {weight_label}\n"
        f"electrons: {float(num_electrons):.8g}; "
        f"occupied-band boundary: {int(occupied_boundary)}\n"
        f"FFT grid: {tuple(fft_grid)}; k grid: {tuple(kgrid)}; "
        f"shift: {tuple(shift)}\n"
        f"selection: weighted k-means; seed={int(seed)}; "
        f"rho_power={float(rho_power):g}; requested={int(requested)}; "
        f"candidates={int(candidates)}; written={int(written)}\n"
        f"pruning: {pruning}{rank_note}; pair-density windows "
        f"left={tuple(prune_left)}, right={tuple(prune_right)} "
        f"({prune_label})\n"
        f"symmetry closure: {'orbit-aware' if orbit_aware else 'literal points'}; "
        f"spatial operations={int(n_sym)}\n"
        f"coordinates: x y z snapped to FFT grid; {int(written)} unique points\n"
        f"intended channels: {channel}"
    )


def format_kmeans_report(*, header: str, source_wfn: str,
                         centroid_file: str, report_file: str,
                         wfn_backend: str, elapsed_s: float, runtime,
                         warnings=()) -> str:
    """Compact scientific output for one completed centroid selection."""
    lines = [
        "=" * 78,
        "LORRAX ISDF CENTROID SELECTION",
        "=" * 78,
        "",
        "PROCESSOR ARCHITECTURE",
        "----------------------",
        *architecture_lines(runtime, mesh_role="centroid selection"),
        "",
        "NUMERICAL ENVIRONMENT",
        "---------------------",
        *numerical_environment_lines(runtime),
        f"Wavefunctions  : {wfn_backend} reader",
        "Selection      : density-weighted k-means on the real-space FFT grid",
        "",
        "CENTROID PROVENANCE",
        "-------------------",
        *header.splitlines(),
        "",
        "FILES",
        "-----",
        f"Source WFN     : {os.path.abspath(source_wfn)}",
        f"Centroids      : {os.path.abspath(centroid_file)}",
        f"Report         : {os.path.abspath(report_file)}",
        "",
        f"Selection wall : {float(elapsed_s):.3f} s",
    ]
    retained = [" ".join(str(item).split()) for item in warnings if str(item).strip()]
    if retained:
        lines.extend(["", "WARNINGS", "--------"])
        lines.extend(f"  {item}" for item in retained)
    lines.append("LORRAX ISDF centroid selection completed.")
    return "\n".join(lines) + "\n"


__all__ = [
    "format_centroid_header",
    "format_kmeans_report",
    "prune_band_ranges",
]

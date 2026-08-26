"""Pure presentation helpers shared by LORRAX's scientific drivers.

This module formats facts owned elsewhere.  It does not inspect input files,
resolve runtime policies, construct symmetry maps, or derive a second set of
band edges.  Drivers hand it their canonical ``RuntimeFacts``, ``WfnLoader``
and ``SymMaps`` objects and receive stable, human-readable lines.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np


FLOAT_DIGITS = 5
TAU_DIGITS = 3


def clean_rounded(values, *, digits: int = FLOAT_DIGITS) -> np.ndarray:
    """Return display values with round-to-zero results made positive zero."""
    array = np.asarray(values, dtype=np.float64).copy()
    array[np.abs(array) < 0.5 * 10.0 ** (-digits)] = 0.0
    return array


def abs_path(path) -> str:
    """An absolute display path, with ``-`` for an absent optional path."""
    return "-" if not path else os.path.abspath(os.fspath(path))


def band_range(lo: int, hi: int) -> str:
    """Format a zero-based half-open code interval as human band numbers."""
    lo, hi = int(lo), int(hi)
    return "none" if hi <= lo else f"{lo + 1}-{hi}"


def policy(value, choices: Iterable[str]) -> str:
    """Show the alternatives whenever the active policy spelling is auto."""
    text = str(value)
    if text.strip().lower() != "auto":
        return text
    alternatives = [str(item) for item in choices
                    if str(item).strip().lower() != "auto"]
    return (f"auto (other choices: {', '.join(alternatives)})"
            if alternatives else "auto")


def centroid_orbit_line(loaded_centroids) -> str:
    """Compact presentation of a canonical centroid-closure verdict."""
    verdict = loaded_centroids.closure
    n_sym = int(verdict.n_sym)
    n_bad = int(verdict.n_violating)
    n_sites = int(verdict.n_centroids)
    tol = float(verdict.tol)
    if bool(verdict.closed):
        return (
            f"Centroid orbit: CLOSED : {n_sym - n_bad}/{n_sym} spatial "
            f"operations preserve {n_sites} sites (tol={tol:.5e})")
    return (
        f"Centroid orbit: NOT CLOSED : {n_bad}/{n_sym} spatial operations "
        f"fail to preserve the {n_sites}-site set; worst residual "
        f"{float(verdict.worst_residual):.5e} at S{int(verdict.worst_op) + 1:02d} "
        f"(tol={tol:.5e})")


def architecture_lines(runtime, *, mesh_role: str | None = None) -> list[str]:
    """Compact rank/device/thread geometry from the canonical runtime facts."""
    facts = runtime.facts
    p = int(facts.get("process_count", 1))
    ndev = int(facts.get("n_devices", 0))
    nlocal = int(facts.get("n_local_devices", 0))
    mesh = tuple(int(v) for v in facts.get("mesh_shape", (1, 1)))
    threads = facts.get("threads") or {}
    affinity = threads.get("affinity")
    thread_fields = []
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "LORRAX_MKLBLAS_THREADS"):
        value = threads.get(key)
        if value not in (None, ""):
            thread_fields.append(f"{key}={value}")
    role = f"  ({mesh_role})" if mesh_role else ""
    local_note = f", {nlocal} per rank" if nlocal else ""
    return [
        f"MPI ranks      : {p}",
        f"Accelerators   : {ndev} {str(facts.get('backend', 'unknown')).upper()} "
        f"devices ({facts.get('device_kind', 'unknown')}{local_note})",
        f"Processor mesh : {mesh[0]} x {mesh[1]}{role}",
        f"CPU allocation : {affinity if affinity is not None else '?'} "
        "cores per rank",
        "Host threads   : " + (", ".join(thread_fields)
                                 if thread_fields else "runtime defaults"),
    ]


def numerical_environment_lines(runtime) -> list[str]:
    facts = runtime.facts
    precision = ("FP64 / complex128" if facts.get("x64")
                 else "FP32 / complex64")
    backend = str(facts.get("backend", "unknown")).lower()
    collectives = ("NCCL" if backend in ("gpu", "cuda", "rocm") else
                   str((facts.get("collectives") or {}).get(
                       "impl", "local")).upper())
    return [
        f"JAX/JAXLIB     : {facts.get('jax_version', 'unknown')} / "
        f"{facts.get('jaxlib_version', 'unknown')} | {precision} | "
        f"{collectives} collectives",
    ]


def _matrix_text(matrix) -> str:
    rows = np.asarray(matrix, dtype=np.int64).reshape(3, 3)
    return "[" + "; ".join(" ".join(f"{int(v):2d}" for v in row)
                            for row in rows) + "]"


def _fractional_translation(translation) -> np.ndarray:
    return np.asarray(translation, dtype=np.float64) / (2.0 * np.pi)


def symmetry_sampling_lines(wfn, sym, *, digits: int = FLOAT_DIGITS,
                            enumerate_ibz: bool = True) -> list[str]:
    """Render the canonical spatial operations and BZ/IBZ sampling.

    The density/TRS statements below only quote ``WfnLoader.density_symmetry``
    and ``SymMaps.trs_allowed``.  No symmetry is rechecked here.
    """
    rotations = np.asarray(sym.Rinv_grid)
    n_sym = int(rotations.shape[0])
    translations = np.asarray(sym.translations, dtype=np.float64)[:n_sym]
    tau = translations / (2.0 * np.pi)
    fractional_tau = np.any(
        np.abs(tau - np.rint(tau)) > 10.0 ** (-(digits + 2)), axis=1)
    n_fractional = int(np.count_nonzero(fractional_tau))
    lines = [
        "Real-space action: r' = R^-1 r + tau",
        f"Spatial group   : {n_sym} operations; {n_fractional} with fractional "
        "translations",
    ]

    receipt = getattr(wfn, "density_symmetry", None)
    if receipt is None:
        lines.append(
            "Density check   : unavailable; symmetry metadata accepted "
            "without a retained receipt")
        lines.append(
            f"Time reversal  : {'enabled' if bool(sym.trs_allowed) else 'disabled'} "
            "for BZ unfolding")
    else:
        m_rel = ("n/a" if receipt.m_rel is None else
                 f"{float(receipt.m_rel):.{digits}e}")
        tested = int(np.count_nonzero(np.isfinite(receipt.spatial_residual)))
        finite = np.asarray(receipt.spatial_residual, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        max_res = ("untested" if finite.size == 0 else
                   f"{float(np.max(finite)):.{digits}e}")
        lines.append(
            f"Density check   : {tested}/{len(receipt.spatial_residual)} "
            f"spatial operations tested; max residual {max_res}")
        mesh_implies = ("unknown" if receipt.trs_implied_by_mesh is None else
                        "yes" if receipt.trs_implied_by_mesh else "no")
        lines.append(
            f"Time reversal  : {'HOLDS' if receipt.trs_holds else 'BROKEN'} "
            f"({receipt.trs_basis}; |m|/|rho|={m_rel}; "
            f"coverage={100.0 * float(receipt.trs_coverage):.{digits}f}%; "
            f"mesh requires it={mesh_implies})")
        lines.append(
            f"TRS unfolding  : {'enabled' if bool(sym.trs_allowed) else 'disabled'} "
            "from the measured density verdict")

    display_tau = clean_rounded(tau, digits=TAU_DIGITS)
    for i, (rotation, shift) in enumerate(
            zip(rotations, display_tau), start=1):
        lines.append(
            f"  S{i:02d}  R^-1={_matrix_text(rotation)}  "
            f"tau=({shift[0]: .{TAU_DIGITS}f} "
            f"{shift[1]: .{TAU_DIGITS}f} "
            f"{shift[2]: .{TAU_DIGITS}f})")

    kgrid = tuple(int(v) for v in np.asarray(wfn.kgrid).reshape(-1)[:3])
    shift = tuple(float(v) for v in clean_rounded(
        np.asarray(wfn.shift).reshape(-1)[:3], digits=digits))
    lines.append(
        f"Full BZ grid   : {int(sym.nk_tot)} k points | mesh "
        f"{kgrid[0]} x {kgrid[1]} x {kgrid[2]} | shift "
        f"({shift[0]:.{digits}f}, {shift[1]:.{digits}f}, "
        f"{shift[2]:.{digits}f})")
    lines.append(f"Stored IBZ     : {int(sym.nk_red)} k points")
    if enumerate_ibz:
        weights = np.asarray(wfn.kweights, dtype=np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum != 0.0:
            weights = weights / weight_sum
        lines.append("  ik        kx        ky        kz     weight")
        points = clean_rounded(wfn.kpoints, digits=digits)
        weights = clean_rounded(weights, digits=digits)
        for ik, (point, weight) in enumerate(zip(points, weights), start=1):
            lines.append(
                f"  {ik:3d}  {point[0]: .{digits}f}  {point[1]: .{digits}f}  "
                f"{point[2]: .{digits}f}  {weight: .{digits}f}")
    return lines


def file_table_lines(rows: Iterable[tuple[str, str, str]], *,
                     omit_paths: Iterable[str] = ()) -> list[str]:
    omitted = {abs_path(path) for path in omit_paths}
    lines = ["  role                     state       path"]
    lines.extend(
        f"  {role:<24} {state:<11} {abs_path(path)}"
        for role, state, path in rows if abs_path(path) not in omitted)
    return lines


def pseudopotential_file_rows(
        pseudos, *, fallback: str = "") -> list[tuple[str, str, str]]:
    """Return one clean input-file row per loaded pseudopotential."""
    rows = []
    for element, pseudo in sorted(pseudos.items()):
        path = getattr(pseudo, "_source_path", "") or fallback
        rows.append((f"pseudopotential {element}", "read", path))
    if not rows and fallback:
        rows.append(("pseudopotentials", "read", fallback))
    return rows


__all__ = [
    "FLOAT_DIGITS",
    "TAU_DIGITS",
    "abs_path",
    "architecture_lines",
    "band_range",
    "centroid_orbit_line",
    "clean_rounded",
    "file_table_lines",
    "numerical_environment_lines",
    "policy",
    "pseudopotential_file_rows",
    "symmetry_sampling_lines",
]

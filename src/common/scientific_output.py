"""Pure presentation helpers shared by LORRAX's scientific drivers.

This module formats facts owned elsewhere.  It does not inspect input files,
resolve runtime policies, construct symmetry maps, or derive a second set of
band edges.  The symmetry, time-reversal and k-point blocks are
:mod:`common.report_blocks`'.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np


FLOAT_DIGITS = 5


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


def spectral_compression_lines(receipt, *,
                               label: str = "Galerkin basis") -> list[str]:
    """Render the canonical htransform spectral-compression receipt.

    The receipt is produced at the one truncation site in
    ``streaming_galerkin_solve``.  This helper only presents that decision;
    it neither selects a rank nor repeats the degeneracy test.  Consequently
    htransform, BSE and exciton-band drivers cannot drift onto different
    explanations of the same retained subspace.
    """
    if receipt.get("method") == "kmeans_nodes":
        retained = int(receipt["retained_rank"])
        carried = int(receipt["carried_rank"])
        n_pad = int(receipt["null_padding"])
        n_nodes, n_cent = int(receipt["nodes"]), int(receipt["centroids"])
        return [
            f"{label:<15}: {int(receipt['stacked_states'])} stacked states "
            f"sampled at {n_nodes} nodes",
            f"Nodes          : {n_nodes} orbit-closed nodes, "
            f"{int(receipt['orbits'])} orbit(s): "
            + ("the ISDF centroids" if n_nodes == n_cent else
               f"k-means or zero-weight drop; the {n_cent} ISDF "
               "centroids are back-map targets")
            + f"; no rank search; fit {float(receipt['fit_seconds']):.2f} s",
            f"Metric         : orbit-NNLS quadrature weights, eps=max_k"
            f"||Phi^H W Phi - 1||_2={float(receipt['metric_eps']):.3e}, "
            f"{int(receipt['zero_weight_nodes'])} zero-weight node(s); "
            "Loewdin-orthonormal coefficients",
            "Error meaning     : the coefficients are exactly orthonormal; "
            "the fine-k band-energy error needs an independent reference",
            f"Distributed carry: rank {carried} = {retained} physical "
            f"(spin x node) + {n_pad} exact-null mesh pad; no rank was "
            "dropped for device alignment",
        ]

    report = receipt["numerical_report"]
    metrics = receipt["compression"]
    numerical_closure = receipt.get("numerical_closure")
    model_closure = receipt.get("model_closure")
    retained = int(receipt["retained_rank"])
    numerical = int(receipt["numerical_rank"])
    carried = int(receipt["carried_rank"])
    n_pad = int(receipt["null_padding"])
    multiplier = float(receipt["rank_multiplier"])

    lines = [
        f"{label:<15}: {int(receipt['stacked_states'])} stacked states x "
        f"{int(receipt['site_spin_columns'])} site-spin samples; structural "
        f"ceiling {int(receipt['structural_ceiling'])}",
        f"Numerical cut   : rtol={float(report.rtol):.5e} "
        f"(amplification cap {float(report.kappa_cap):.5e}) -> rank "
        f"{numerical}",
    ]
    if multiplier == 0.0:
        lines.append(f"Model order     : full numerical span; retained rank {retained}")
    else:
        lines.append(
            f"Model order     : requested {multiplier:.5f} x fitted bands; "
            f"retained rank {retained} of numerical rank {numerical}")

    def _closure_line(name: str, info) -> str | None:
        if info is None:
            return None
        direction = str(info["direction"])
        if bool(info["fired"]):
            moved = int(info["n_keep"]) - int(info["n_keep_closed"])
            action = "dropped" if direction == "drop_block" else "kept"
            return (
                f"{name:<15}: whole-block {direction}; cut "
                f"{int(info['n_keep'])} -> {int(info['n_keep_closed'])}; "
                f"{action} {abs(moved)} direction(s) from a "
                f"{len(info['members'])}-value degenerate block "
                f"(relative span {float(info['span_rel']):.5e})")
        gap = float(info["gap_rel"])
        gap_text = "spectrum edge" if not np.isfinite(gap) else f"gap {gap:.5e}"
        return (
            f"{name:<15}: whole-block {direction}; unchanged at rank "
            f"{int(info['n_keep_closed'])} ({gap_text})")

    for closure_name, closure in (
            ("Numerical block", numerical_closure),
            ("Model block", model_closure)):
        line = _closure_line(closure_name, closure)
        if line is not None:
            lines.append(line)

    kappa = ("n/a" if metrics.kappa_eff is None
             else f"{float(metrics.kappa_eff):.5e}")
    sigma_min = ("n/a" if metrics.sigma_min_kept is None
                 else f"{float(metrics.sigma_min_kept):.5e}")
    lines.extend((
        f"Conditioning     : sigma_max={float(metrics.sigma_max):.5e}; "
        f"sigma_min(kept)={sigma_min}; kappa_eff={kappa}",
        f"Sampled-psi tail: best-rank relative residual "
        f"{float(metrics.relative_operator_tail):.5e} (operator norm), "
        f"{float(metrics.relative_frobenius_tail):.5e} (Frobenius norm); "
        f"retained weight {100.0 * float(metrics.retained_frobenius_weight):.2f}%",
        "Error meaning     : residuals above quantify compression of the "
        "sampled wavefunction matrix; they are not a band-energy error bound",
        f"Distributed carry: rank {carried} = {retained} physical + {n_pad} "
        "exact-null mesh pad; no rank was dropped for device alignment",
    ))
    return lines


def htransform_quality_lines(receipt) -> list[str]:
    """Render the shared htransform representation/locality receipt.

    The row-isometry number screens recovery on the fitted coarse grid.  The
    real-space number measures how much of ``f(H)_R`` reaches the outer shell
    of the available Born--von Karman supercell.  Neither is silently promoted
    into an off-grid band-energy error: that requires an independent fine-k
    reference, as the report states explicitly.
    """
    isometry = float(receipt["row_isometry_max"])
    cap = receipt.get("row_isometry_cap")
    shell = 100.0 * float(receipt["outer_shell_l2_fraction"])
    shell_max = 100.0 * float(receipt["outer_shell_max_over_r0"])
    wall = float(receipt["locality_wall_seconds"])
    isometry_line = f"Row isometry   : max |C C^H - I|={isometry:.5e}"
    if cap is None:
        isometry_line += " (diagnostic only; no per-k gauge repair)"
    else:
        isometry_line += f" (acceptance cap {float(cap):.5e})"
    return [
        isometry_line,
        f"f(H)_R locality: outer-shell ||f(H)_R||_F / total = {shell:.2f}%; "
        f"largest shell ||f(H)_R||_F / R=0 = {shell_max:.2f}%",
        "Error meaning  : the locality receipt diagnoses finite-supercell "
        "aliasing risk; it is not an off-grid energy-error bound",
        "Fine-k error   : unavailable without an independent fine-k "
        "wavefunction reference",
        f"Receipt cost   : {wall:.2f} s; one sharded reduction, no matrix "
        "gather",
    ]


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
    "abs_path",
    "band_range",
    "clean_rounded",
    "htransform_quality_lines",
    "policy",
    "pseudopotential_file_rows",
    "spectral_compression_lines",
]

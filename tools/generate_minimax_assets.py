#!/usr/bin/env python3
"""Generate shipped minimax quadrature tables and a machine-readable catalog.

The generated catalog is consumed by the ``minimax`` service door.  Tables are
fit on scaled intervals so runtime lookup can safely round a requested range
upward and reuse the nearest stricter shipped table.

REWRITTEN AGAINST THE DOOR (2026-08-08), not shimmed.  This tool used to be
the only consumer of ``gw.minimax_screening``'s private ``lru_cache``d solve
wrappers, and it is about to become the MPA generator campaign's sweep
driver — so a coupling to another package's internals was the one thing it
could not be allowed to inherit across the extraction.  It now calls
``minimax.solve_uncertified``, which is the door's own name for "run the
offline solver in-process and say so".  That announcement is correct here
and not noise: a table is uncertified until this tool certifies it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np
import scipy
from scipy.optimize import brentq, minimize_scalar


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_SERVICES = REPO_ROOT / "services"
if _SERVICES.is_dir():
    for _svc in sorted(_SERVICES.iterdir()):
        _svc_src = _svc / "src"
        if _svc_src.is_dir() and str(_svc_src) not in sys.path:
            sys.path.append(str(_svc_src))

from minimax import (  # noqa: E402
    noncrossing_kappa0,
    payload_sha256,
    solve_uncertified,
)


DEFAULT_ERROR_BOUNDS = (1.0e-6, 2.0e-7)
DEFAULT_NONCROSSING_R_VALUES = tuple(
    float(v) for v in np.logspace(1.0, 5.0, num=(5 - 1) * 3 + 1)
)
DEFAULT_OUTPUT_ROOT = (REPO_ROOT / "services" / "minimax" / "src" / "minimax"
                       / "minimax_assets")
_NONCROSSING_CERT_GRID_SIZE = 20_001
_NONCROSSING_KAPPA0_BOUND = 2.0


def _format_float_token(value: float, *, decimals: int = 6) -> str:
    text = f"{float(value):.{decimals}f}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _format_error_token(value: float) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e")
    mantissa = mantissa.replace("-", "m").replace(".", "p")
    exponent = exponent.replace("+", "p").replace("-", "m")
    return f"{mantissa}e{exponent}"


def _ensure_clean_dir(path: Path, *, clobber: bool) -> None:
    if path.exists() and clobber:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_table(
    path: Path,
    *,
    tau: np.ndarray,
    alpha: np.ndarray,
    max_error: float,
    certificate: dict[str, object] | None = None,
) -> None:
    fields = {
        "tau": np.asarray(tau, dtype=np.float64),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "max_error": np.asarray(float(max_error), dtype=np.float64),
    }
    if certificate is not None:
        fields.update({
            "kappa0": np.asarray(certificate["kappa0"], dtype=np.float64),
            "sum_abs_alpha": np.asarray(
                certificate["sum_abs_alpha"], dtype=np.float64),
            "payload_sha256": np.asarray(certificate["payload_sha256"]),
        })
    with path.open("wb") as fh:
        np.savez_compressed(fh, **fields)


def _read_table(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with path.open("rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            return (
                np.asarray(data["tau"], dtype=np.float64),
                np.asarray(data["alpha"], dtype=np.float64),
                float(data["max_error"][()]),
            )


def _noncrossing_held_out_grid(
    range_value: float,
    n_eval: int = _NONCROSSING_CERT_GRID_SIZE,
) -> np.ndarray:
    """Half-cell log grid plus endpoints, disjoint from solver grids."""
    count = int(n_eval)
    if count < 2:
        raise ValueError("noncrossing certificate needs at least two cells")
    log_r = math.log(float(range_value))
    interior = np.exp(log_r * (np.arange(count) + 0.5) / count)
    return np.unique(np.concatenate(([1.0], interior, [float(range_value)])))


def certify_noncrossing_inverse(
    tau: np.ndarray,
    alpha: np.ndarray,
    range_value: float,
    error_bound: float,
) -> dict[str, object]:
    """Certify one shipped ``1/x`` rule from its final numerical payload.

    The nonlinear solver's own training residual is not evidence here.  This
    uses a deterministic half-cell held-out log grid, brackets every sign
    change of the analytic residual derivative, and refines each held-out
    local absolute-error extremum.  The latter protects the achieved maximum
    against cancellation noise in the derivative evaluation itself.
    """
    tau = np.asarray(tau, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    R = float(range_value)
    target = float(error_bound)
    failures: list[str] = []
    if (tau.ndim != 1 or alpha.shape != tau.shape or tau.size == 0
            or not np.all(np.isfinite(tau))
            or not np.all(np.isfinite(alpha))):
        raise ValueError("noncrossing certificate requires finite 1-D arrays")
    if not (np.isfinite(R) and R > 1.0 and np.isfinite(target)
            and target > 0.0):
        raise ValueError("noncrossing certificate requires R>1 and eps>0")
    if not np.all(tau > 0.0):
        failures.append("positive_nodes")
    if not np.all(alpha > 0.0):
        failures.append("positive_weights")

    def residual_scalar(x: float) -> float:
        return float(1.0 / x - np.dot(np.exp(-tau * x), alpha))

    def derivative_scalar(x: float) -> float:
        return float(-1.0 / (x * x)
                     + np.dot(alpha * tau, np.exp(-tau * x)))

    grid = _noncrossing_held_out_grid(R)
    exp_grid = np.exp(-grid[:, None] * tau[None, :])
    residual_grid = 1.0 / grid - exp_grid @ alpha
    derivative_grid = (-1.0 / grid**2
                       + exp_grid @ (alpha * tau))

    derivative_roots: list[float] = []
    for lo, hi, d_lo, d_hi in zip(
            grid[:-1], grid[1:], derivative_grid[:-1], derivative_grid[1:]):
        if d_lo == 0.0:
            derivative_roots.append(float(lo))
        elif d_lo * d_hi < 0.0:
            derivative_roots.append(float(brentq(
                derivative_scalar, float(lo), float(hi),
                xtol=np.nextafter(0.0, 1.0),
                rtol=4.0 * np.finfo(np.float64).eps,
            )))

    abs_grid = np.abs(residual_grid)
    local_indices = np.flatnonzero(
        (abs_grid[1:-1] >= abs_grid[:-2])
        & (abs_grid[1:-1] >= abs_grid[2:])) + 1
    local_refined: list[float] = []
    for index in local_indices:
        result = minimize_scalar(
            lambda x: -abs(residual_scalar(float(x))),
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-15, "maxiter": 1000},
        )
        if not result.success:
            failures.append("extremum_refinement")
        else:
            local_refined.append(float(result.x))

    refined_points = np.unique(np.concatenate((
        grid,
        np.asarray(derivative_roots, dtype=np.float64),
        np.asarray(local_refined, dtype=np.float64),
    )))
    refined_error = max(abs(residual_scalar(float(x)))
                        for x in refined_points)
    held_out_error = float(np.max(abs_grid))
    kappa0 = noncrossing_kappa0(
        tau, alpha, R, n_eval=_NONCROSSING_CERT_GRID_SIZE)
    sum_abs_alpha = float(np.sum(np.abs(alpha)))

    rescale_ratio = 0.0
    for scale in (1.0e-5, 1.0e-2, 1.0, 1.0e3, 1.0e6):
        x_phys = refined_points * scale
        fit_phys = (np.exp(-x_phys[:, None]
                           * (tau / scale)[None, :])
                    @ (alpha / scale))
        exact_phys = 1.0 / x_phys
        got = float(np.max(np.abs(exact_phys - fit_phys)))
        rescale_ratio = max(rescale_ratio, got / (refined_error / scale))

    if refined_error > target:
        failures.append("refined_error")
    if kappa0 > _NONCROSSING_KAPPA0_BOUND:
        failures.append("kappa0")
    if rescale_ratio > 1.0 + 1.0e-7:
        failures.append("rescale")
    return {
        "certified": not failures,
        "failures": failures,
        "max_error": float(refined_error),
        "held_out_max_error": held_out_error,
        "derivative_root_count": len(derivative_roots),
        "local_refinement_count": len(local_refined),
        "n_eval": int(grid.size),
        "kappa0": float(kappa0),
        "kappa0_bound": _NONCROSSING_KAPPA0_BOUND,
        "sum_abs_alpha": sum_abs_alpha,
        "rescale_max_error_ratio": float(rescale_ratio),
        "payload_sha256": payload_sha256(tau, alpha),
    }


def _generator_provenance() -> dict[str, str]:
    """Exact tool/source/backend identity for a newly certified row."""
    tool = Path(__file__).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    backend = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "certifier": "held-out-log+derivative-roots+bounded-extrema/v1",
    }
    backend_sha256 = hashlib.sha256(
        json.dumps(backend, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "tool": str(tool.relative_to(REPO_ROOT)),
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "generator_commit": commit,
        **backend,
        "backend_sha256": backend_sha256,
    }


def _iter_error_bounds(values: Iterable[float]) -> list[float]:
    cleaned = sorted({float(v) for v in values if float(v) > 0.0}, reverse=True)
    if not cleaned:
        raise ValueError("Need at least one positive error bound.")
    return cleaned


def _build_readme_text(*, noncrossing_r_values: list[float],
                       error_bounds: list[float]) -> str:
    return f"""# Shipped Minimax Quadratures

This directory contains precomputed minimax quadrature tables for runtime reuse by
`gw.minimax_screening`.

## Selection rule

At runtime, the lookup code chooses:

1. the smallest tabulated range that is greater than or equal to the requested range
2. the loosest tabulated error bound that is still less than or equal to the requested target
3. a table whose node count does not exceed the caller's `max_nodes`

If no shipped table matches, runtime falls back to the exact solver.

## Error conventions

### Noncrossing

Tables are generated on the scaled interval `[1, R]` with the absolute L-infinity error
convention:

`max_y | 1/y - approx(y) | <= error_bound`, for `y in [1, R]`.

When used for a physical interval `[x_min, x_max]`, runtime rescales the nodes and weights.
The physical absolute error then scales as `error_bound / x_min`.

This is not a relative-at-endpoint criterion.

## Sweep values in this bundle

- Error bounds: {", ".join(f"{v:.1e}" for v in error_bounds)}
- Noncrossing `R` values: {", ".join(f"{v:.6g}" for v in noncrossing_r_values)}

The machine-readable descriptor is `catalog.json`.
"""


def _build_catalog(
    *,
    tables: list[dict[str, object]],
    noncrossing_r_values: list[float],
    error_bounds: list[float],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "selection_rule": {
            "range": "smallest_tabulated_ge_requested",
            "error_bound": "largest_tabulated_le_requested",
            "max_nodes": "table_node_count_must_not_exceed_request",
        },
        "shipping_rule": {
            "kappa0_definition": (
                "max over u in [1,R] of u * sum_l |alpha_l| exp(-tau_l u)"),
            "normal": _NONCROSSING_KAPPA0_BOUND,
            "rejected_above": _NONCROSSING_KAPPA0_BOUND,
        },
        "conventions": {
            "noncrossing": {
                "problem": "1/x on [1, R]",
                "error_metric": "linf_abs_scaled_1_over_x",
                "physical_error_note": "runtime rescales tables from [1,R] to [x_min,x_max], so absolute error scales as error_bound / x_min",
            },
        },
        "sweeps": {
            "error_bounds": [float(v) for v in error_bounds],
            "noncrossing_R_values": [float(v) for v in noncrossing_r_values],
        },
        "tables": tables,
    }


def _load_existing_tables(output_root: Path) -> dict[str, dict[str, object]]:
    catalog_path = output_root / "catalog.json"
    if not catalog_path.exists():
        return {}
    try:
        with catalog_path.open("r", encoding="utf-8") as fh:
            catalog = json.load(fh)
    except Exception:
        return {}

    tables = catalog.get("tables", [])
    if not isinstance(tables, list):
        return {}

    result: dict[str, dict[str, object]] = {}
    for entry in tables:
        if not isinstance(entry, dict):
            continue
        rel_path = entry.get("file")
        if not isinstance(rel_path, str) or not rel_path:
            continue
        if not (output_root / rel_path).exists():
            continue
        result[rel_path] = dict(entry)
    return result


def _merged_sweep_values(
    output_root: Path,
    *,
    error_bounds: list[float],
    noncrossing_r_values: list[float],
    clobber: bool,
) -> tuple[list[float], list[float]]:
    """Keep the catalog census cumulative under ``--no-clobber``.

    The solve loops still receive exactly the requested cells.  These values
    are descriptive catalog/README metadata only; replacing them with the
    latest one-cell request made an append falsely claim that all older
    shipped rows had disappeared.
    """

    if clobber:
        return error_bounds, noncrossing_r_values
    try:
        prior = json.loads(
            (output_root / "catalog.json").read_text(encoding="utf-8"))
        sweeps = prior.get("sweeps", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return error_bounds, noncrossing_r_values

    def _union(prior_values, requested_values, *, reverse=False):
        return sorted({float(value) for value in
                       list(prior_values or []) + list(requested_values)},
                      reverse=reverse)

    catalog_errors = _union(
        sweeps.get("error_bounds"), error_bounds, reverse=True)
    catalog_noncrossing = _union(
        sweeps.get("noncrossing_R_values"),
        noncrossing_r_values,
    )
    return catalog_errors, catalog_noncrossing


def _sorted_table_entries(table_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    def _sort_key(entry: dict[str, object]) -> tuple[object, ...]:
        return (
            str(entry.get("family", "")),
            float(entry.get("range_max", 0.0)),
            float(entry.get("error_bound", 0.0)),
            str(entry.get("target_kind", "")),
            str(entry.get("file", "")),
        )

    return sorted(table_map.values(), key=_sort_key)


def _flush_outputs(
    output_root: Path,
    *,
    tables: list[dict[str, object]],
    noncrossing_r_values: list[float],
    error_bounds: list[float],
) -> dict[str, object]:
    catalog = _build_catalog(
        tables=tables,
        noncrossing_r_values=noncrossing_r_values,
        error_bounds=error_bounds,
    )
    with (output_root / "catalog.json").open("w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, sort_keys=False)
        fh.write("\n")
    with (output_root / "README.md").open("w", encoding="utf-8") as fh:
        fh.write(
            _build_readme_text(
                noncrossing_r_values=noncrossing_r_values,
                error_bounds=error_bounds,
            )
        )
    return catalog


def generate_assets(
    *,
    output_root: Path,
    error_bounds: list[float],
    noncrossing_r_values: list[float],
    clobber: bool,
) -> dict[str, object]:
    noncrossing_dir = output_root / "noncrossing"
    _ensure_clean_dir(noncrossing_dir, clobber=clobber)

    table_map = {} if clobber else {
        path: entry
        for path, entry in _load_existing_tables(output_root).items()
        if entry.get("family") == "noncrossing"
    }
    (catalog_error_bounds,
     catalog_noncrossing_r_values) = _merged_sweep_values(
         output_root,
         error_bounds=error_bounds,
         noncrossing_r_values=noncrossing_r_values,
         clobber=clobber,
     )
    total_tables = len(error_bounds) * len(noncrossing_r_values)
    table_index = 0

    for err in error_bounds:
        for R in noncrossing_r_values:
            table_index += 1
            fname = (
                f"noncrossing_R_{_format_float_token(R)}"
                f"_eps_{_format_error_token(err)}.npz"
            )
            rel_path = Path("noncrossing") / fname
            table_path = noncrossing_dir / fname
            if table_path.exists():
                tau, alpha, max_error = _read_table(table_path)
                action = "reuse"
            else:
                _q = solve_uncertified(
                    family="noncrossing",
                    target="inverse",
                    range_value=float(R),
                    error_bound=float(err),
                    n_max=64,
                )
                tau, alpha, max_error = _q.nodes, _q.weights, _q.max_error
                action = "solve"
            certificate = certify_noncrossing_inverse(
                tau, alpha, float(R), float(err))
            if not certificate["certified"]:
                raise RuntimeError(
                    f"noncrossing R={R:g} eps={err:.1e} failed "
                    f"certification: {certificate['failures']}")
            max_error = float(certificate["max_error"])
            _write_table(
                table_path, tau=tau, alpha=alpha, max_error=max_error,
                certificate=certificate)
            table_map[rel_path.as_posix()] = {
                "family": "noncrossing",
                "error_bound": float(err),
                "error_metric": "linf_abs_scaled_1_over_x",
                "range_param": "R",
                "range_max": float(R),
                "node_count": int(len(tau)),
                "max_error": float(max_error),
                "held_out_max_error": float(
                    certificate["held_out_max_error"]),
                "kappa0": float(certificate["kappa0"]),
                "kappa0_bound": float(certificate["kappa0_bound"]),
                "sum_abs_alpha": float(certificate["sum_abs_alpha"]),
                "rescale_max_error_ratio": float(
                    certificate["rescale_max_error_ratio"]),
                "payload_sha256": str(certificate["payload_sha256"]),
                "certified": True,
                "certification": {
                    "method": (
                        "held-out log grid plus endpoints, analytic "
                        "derivative roots, and bounded local extrema"),
                    "held_out_grid_size": int(certificate["n_eval"]),
                    "derivative_root_count": int(
                        certificate["derivative_root_count"]),
                    "local_refinement_count": int(
                        certificate["local_refinement_count"]),
                    "checks": [
                        "refined_error", "positive_nodes",
                        "positive_weights", "kappa0", "rescale",
                    ],
                },
                "provenance": _generator_provenance(),
                "file": rel_path.as_posix(),
            }
            print(
                f"[{table_index:02d}/{total_tables:02d}] {action:5s} "
                f"noncrossing R={R:.6g} err<={err:.1e} "
                f"nodes={len(tau)} max|Δ|={float(max_error):.3e}"
            )
            _flush_outputs(
                output_root,
                tables=_sorted_table_entries(table_map),
                noncrossing_r_values=catalog_noncrossing_r_values,
                error_bounds=catalog_error_bounds,
            )

    catalog = _flush_outputs(
        output_root,
        tables=_sorted_table_entries(table_map),
        noncrossing_r_values=catalog_noncrossing_r_values,
        error_bounds=catalog_error_bounds,
    )
    return catalog


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where catalog.json, README.md, and .npz tables will be written.",
    )
    ap.add_argument(
        "--error-bound",
        type=float,
        action="append",
        default=None,
        help="Absolute error bound to tabulate. Repeat for multiple values.",
    )
    ap.add_argument(
        "--noncrossing-r-min",
        type=float,
        default=1.0e1,
        help="Minimum noncrossing R value.",
    )
    ap.add_argument(
        "--noncrossing-r-max",
        type=float,
        default=1.0e5,
        help="Maximum noncrossing R value.",
    )
    ap.add_argument(
        "--noncrossing-intervals-per-decade",
        type=int,
        default=3,
        help="Number of log-intervals per decade for the noncrossing R sweep.",
    )
    ap.add_argument(
        "--no-clobber",
        action="store_true",
        help="Keep existing output files and overwrite entries in place instead of recreating the directory tree.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    error_bounds = _iter_error_bounds(args.error_bound or DEFAULT_ERROR_BOUNDS)
    decades = math.log10(args.noncrossing_r_max) - math.log10(args.noncrossing_r_min)
    n_intervals = int(round(decades * int(args.noncrossing_intervals_per_decade)))
    noncrossing_r_values = [
        float(v)
        for v in np.logspace(
            math.log10(args.noncrossing_r_min),
            math.log10(args.noncrossing_r_max),
            num=n_intervals + 1,
        )
    ]
    catalog = generate_assets(
        output_root=args.output_root,
        error_bounds=error_bounds,
        noncrossing_r_values=noncrossing_r_values,
        clobber=not args.no_clobber,
    )
    print(f"Wrote minimax assets to {args.output_root}")
    print(f"  tables: {len(catalog['tables'])}")
    print("  family: noncrossing")
    print(f"  noncrossing R values: {len(noncrossing_r_values)}")
    print(f"  error bounds: {', '.join(f'{v:.1e}' for v in error_bounds)}")


if __name__ == "__main__":
    main()

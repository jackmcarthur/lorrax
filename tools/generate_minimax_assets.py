#!/usr/bin/env python3
"""Generate shipped minimax quadrature tables and a machine-readable catalog.

The generated catalog is consumed by ``gw.minimax_screening``. Tables are fit on
scaled intervals so runtime lookup can safely round a requested range upward and
reuse the nearest stricter shipped table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gw import minimax_screening as ms  # noqa: E402


DEFAULT_ERROR_BOUNDS = (1.0e-6, 2.0e-7)
DEFAULT_CROSSING_A_VALUES = tuple(float(v) for v in range(20, 201, 20))
DEFAULT_NONCROSSING_R_VALUES = tuple(
    float(v) for v in np.logspace(1.0, 5.0, num=(5 - 1) * 3 + 1)
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "src" / "common" / "minimax_assets"
DEFAULT_FAMILIES = ("crossing", "noncrossing")


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


def _write_table(path: Path, *, tau: np.ndarray, alpha: np.ndarray, max_error: float) -> None:
    with path.open("wb") as fh:
        np.savez_compressed(
            fh,
            tau=np.asarray(tau, dtype=np.float64),
            alpha=np.asarray(alpha, dtype=np.float64),
            max_error=np.asarray(float(max_error), dtype=np.float64),
        )


def _read_table(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with path.open("rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            return (
                np.asarray(data["tau"], dtype=np.float64),
                np.asarray(data["alpha"], dtype=np.float64),
                float(data["max_error"][()]),
            )


def _iter_error_bounds(values: Iterable[float]) -> list[float]:
    cleaned = sorted({float(v) for v in values if float(v) > 0.0}, reverse=True)
    if not cleaned:
        raise ValueError("Need at least one positive error bound.")
    return cleaned


def _iter_families(values: Iterable[str]) -> list[str]:
    cleaned = []
    for value in values:
        family = str(value).strip().lower()
        if family not in {"crossing", "noncrossing"}:
            raise ValueError(f"Unsupported family {value!r}; expected 'crossing' or 'noncrossing'.")
        if family not in cleaned:
            cleaned.append(family)
    if not cleaned:
        raise ValueError("Need at least one quadrature family.")
    return cleaned


def _build_readme_text(*, crossing_a_values: list[float], noncrossing_r_values: list[float], error_bounds: list[float], crossing_eps_q: float, crossing_target_kind: str) -> str:
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

### Crossing

Crossing tables are generated for the absolute L-infinity error on the target function
itself:

`max_u | G(u) - approx(u) | <= error_bound`, for `u in [0, A_dim]`.

## Sweep values in this bundle

- Error bounds: {", ".join(f"{v:.1e}" for v in error_bounds)}
- Crossing target kind: `{crossing_target_kind}`
- Crossing `eps_q`: {crossing_eps_q:.3e}
- Crossing `A_dim` values: {", ".join(f"{v:.6g}" for v in crossing_a_values)}
- Noncrossing `R` values: {", ".join(f"{v:.6g}" for v in noncrossing_r_values)}

The machine-readable descriptor is `catalog.json`.
"""


def _build_catalog(
    *,
    tables: list[dict[str, object]],
    crossing_a_values: list[float],
    noncrossing_r_values: list[float],
    error_bounds: list[float],
    crossing_eps_q: float,
    crossing_target_kind: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "selection_rule": {
            "range": "smallest_tabulated_ge_requested",
            "error_bound": "largest_tabulated_le_requested",
            "max_nodes": "table_node_count_must_not_exceed_request",
        },
        "conventions": {
            "noncrossing": {
                "problem": "1/x on [1, R]",
                "error_metric": "linf_abs_scaled_1_over_x",
                "physical_error_note": "runtime rescales tables from [1,R] to [x_min,x_max], so absolute error scales as error_bound / x_min",
            },
            "crossing": {
                "problem": "G(u) on [0, A_dim]",
                "error_metric": "linf_abs_target",
                "eps_q": float(crossing_eps_q),
                "target_kind": str(crossing_target_kind),
            },
        },
        "sweeps": {
            "error_bounds": [float(v) for v in error_bounds],
            "crossing_A_dim_values": [float(v) for v in crossing_a_values],
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
    crossing_a_values: list[float],
    noncrossing_r_values: list[float],
    error_bounds: list[float],
    crossing_eps_q: float,
    crossing_target_kind: str,
) -> dict[str, object]:
    catalog = _build_catalog(
        tables=tables,
        crossing_a_values=crossing_a_values,
        noncrossing_r_values=noncrossing_r_values,
        error_bounds=error_bounds,
        crossing_eps_q=crossing_eps_q,
        crossing_target_kind=crossing_target_kind,
    )
    with (output_root / "catalog.json").open("w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, sort_keys=False)
        fh.write("\n")
    with (output_root / "README.md").open("w", encoding="utf-8") as fh:
        fh.write(
            _build_readme_text(
                crossing_a_values=crossing_a_values,
                noncrossing_r_values=noncrossing_r_values,
                error_bounds=error_bounds,
                crossing_eps_q=crossing_eps_q,
                crossing_target_kind=crossing_target_kind,
            )
        )
    return catalog


def generate_assets(
    *,
    output_root: Path,
    error_bounds: list[float],
    crossing_a_values: list[float],
    noncrossing_r_values: list[float],
    crossing_eps_q: float,
    crossing_target_kind: str,
    families: list[str],
    clobber: bool,
) -> dict[str, object]:
    crossing_dir = output_root / "crossing"
    noncrossing_dir = output_root / "noncrossing"
    if "crossing" in families:
        _ensure_clean_dir(crossing_dir, clobber=clobber)
    if "noncrossing" in families:
        _ensure_clean_dir(noncrossing_dir, clobber=clobber)

    table_map = {} if clobber else _load_existing_tables(output_root)
    total_tables = 0
    if "crossing" in families:
        total_tables += len(error_bounds) * len(crossing_a_values)
    if "noncrossing" in families:
        total_tables += len(error_bounds) * len(noncrossing_r_values)
    table_index = 0

    if "crossing" in families:
        for err in error_bounds:
            for A_dim in crossing_a_values:
                table_index += 1
                fname = (
                    f"crossing_{crossing_target_kind}"
                    f"_A_{_format_float_token(A_dim)}"
                    f"_eps_{_format_error_token(err)}"
                    f"_epsq_{_format_error_token(crossing_eps_q)}.npz"
                )
                rel_path = Path("crossing") / fname
                table_path = crossing_dir / fname
                if table_path.exists():
                    tau, alpha, max_error = _read_table(table_path)
                    action = "reuse"
                else:
                    tau, alpha, max_error = ms._solve_crossing_scaled_cached(
                        round(float(A_dim), 12),
                        round(float(err), 14),
                        500,
                        round(float(crossing_eps_q), 12),
                        str(crossing_target_kind),
                    )
                    _write_table(table_path, tau=tau, alpha=alpha, max_error=max_error)
                    action = "solve"
                table_map[rel_path.as_posix()] = {
                    "family": "crossing",
                    "target_kind": str(crossing_target_kind),
                    "error_bound": float(err),
                    "error_metric": "linf_abs_target",
                    "range_param": "A_dim",
                    "range_max": float(A_dim),
                    "eps_q": float(crossing_eps_q),
                    "node_count": int(len(tau)),
                    "max_error": float(max_error),
                    "file": rel_path.as_posix(),
                }
                print(
                    f"[{table_index:02d}/{total_tables:02d}] {action:5s} crossing "
                    f"A={A_dim:.6g} err<={err:.1e} nodes={len(tau)} max|Δ|={float(max_error):.3e}"
                )
                _flush_outputs(
                    output_root,
                    tables=_sorted_table_entries(table_map),
                    crossing_a_values=crossing_a_values,
                    noncrossing_r_values=noncrossing_r_values,
                    error_bounds=error_bounds,
                    crossing_eps_q=crossing_eps_q,
                    crossing_target_kind=crossing_target_kind,
                )

    if "noncrossing" in families:
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
                    tau, alpha, max_error = ms._solve_noncrossing_scaled_cached(
                        round(float(math.log(R)), 12),
                        round(float(err), 14),
                        64,
                    )
                    _write_table(table_path, tau=tau, alpha=alpha, max_error=max_error)
                    action = "solve"
                table_map[rel_path.as_posix()] = {
                    "family": "noncrossing",
                    "error_bound": float(err),
                    "error_metric": "linf_abs_scaled_1_over_x",
                    "range_param": "R",
                    "range_max": float(R),
                    "node_count": int(len(tau)),
                    "max_error": float(max_error),
                    "file": rel_path.as_posix(),
                }
                print(
                    f"[{table_index:02d}/{total_tables:02d}] {action:5s} noncrossing "
                    f"R={R:.6g} err<={err:.1e} nodes={len(tau)} max|Δ|={float(max_error):.3e}"
                )
                _flush_outputs(
                    output_root,
                    tables=_sorted_table_entries(table_map),
                    crossing_a_values=crossing_a_values,
                    noncrossing_r_values=noncrossing_r_values,
                    error_bounds=error_bounds,
                    crossing_eps_q=crossing_eps_q,
                    crossing_target_kind=crossing_target_kind,
                )

    catalog = _flush_outputs(
        output_root,
        tables=_sorted_table_entries(table_map),
        crossing_a_values=crossing_a_values,
        noncrossing_r_values=noncrossing_r_values,
        error_bounds=error_bounds,
        crossing_eps_q=crossing_eps_q,
        crossing_target_kind=crossing_target_kind,
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
        "--family",
        type=str,
        action="append",
        default=None,
        help="Quadrature family to generate: crossing or noncrossing. Repeat to request both.",
    )
    ap.add_argument(
        "--crossing-a-max",
        type=float,
        default=200.0,
        help="Largest crossing A_dim value to tabulate.",
    )
    ap.add_argument(
        "--crossing-a-step",
        type=float,
        default=20.0,
        help="Step size for crossing A_dim sweep.",
    )
    ap.add_argument(
        "--crossing-eps-q",
        type=float,
        default=1.0e-3,
        help="eps_q parameter for crossing HGL/Fermi support truncation.",
    )
    ap.add_argument(
        "--crossing-target-kind",
        type=str,
        default="hgl",
        choices=("hgl", "fermi"),
        help="Crossing target family to generate.",
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
    families = _iter_families(args.family or DEFAULT_FAMILIES)
    crossing_count = int(round(args.crossing_a_max / args.crossing_a_step))
    crossing_a_values = [float(v) for v in np.arange(1, crossing_count + 1) * float(args.crossing_a_step)]
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
        crossing_a_values=crossing_a_values,
        noncrossing_r_values=noncrossing_r_values,
        crossing_eps_q=float(args.crossing_eps_q),
        crossing_target_kind=str(args.crossing_target_kind),
        families=families,
        clobber=not args.no_clobber,
    )
    print(f"Wrote minimax assets to {args.output_root}")
    print(f"  tables: {len(catalog['tables'])}")
    print(f"  families: {', '.join(families)}")
    print(f"  crossing A values: {len(crossing_a_values)}")
    print(f"  noncrossing R values: {len(noncrossing_r_values)}")
    print(f"  error bounds: {', '.join(f'{v:.1e}' for v in error_bounds)}")


if __name__ == "__main__":
    main()

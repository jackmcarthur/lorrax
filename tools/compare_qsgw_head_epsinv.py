#!/usr/bin/env python3
"""Compare a Lorrax QSGW-head diagnostic with a BerkeleyGW epsilon head.

The Lorrax input is a whitespace-separated text file with three required
columns::

    omega_ev Re_epsinv00 Im_epsinv00

Two optional columns, ``Re_chi00 Im_chi00``, are preserved in the report.
Metadata may be supplied in comments such as ``# broadening_ev: 0.10`` and
``# source: ...``.  If broadening is absent, pass
``--lorrax-broadening-ev`` explicitly.

The BerkeleyGW input can be ``EpsInvDyn`` text or ``eps0mat.h5``.  Only the
real-frequency branch is compared.  ``EpsInvDyn`` does not encode the
imaginary part of the real-axis frequency, so ``--bgw-broadening-ev`` is
required for that format.  Frequencies are compared in eV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np


RYD_TO_EV = 13.6056980659
_Q_HEADER = re.compile(
    r"^#\s*q=\s*"
    r"([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+"
    r"nmtx\s*=",
)


def _as_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _unit_factor_to_ev(unit: str) -> float:
    normalized = unit.lower()
    if normalized == "ev":
        return 1.0
    if normalized == "ry":
        return RYD_TO_EV
    if normalized == "ha":
        return 2.0 * RYD_TO_EV
    raise ValueError(f"unsupported frequency unit: {unit}")


def _validate_increasing(name: str, values: np.ndarray) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _select_q(qpoints: np.ndarray, requested_index: int | None) -> int:
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(qpoints):
            raise ValueError(
                f"q index {requested_index} is outside [0, {len(qpoints) - 1}]",
            )
        return requested_index
    return int(np.argmin(np.linalg.norm(qpoints, axis=1)))


def _parse_comment_metadata(line: str) -> tuple[str, str] | None:
    content = line.lstrip()[1:].strip()
    if ":" in content:
        key, value = content.split(":", 1)
    elif "=" in content:
        key, value = content.split("=", 1)
    else:
        return None
    key = key.strip().lower().replace(" ", "_")
    return key, value.strip()


def read_lorrax_head(
    path: str | Path,
    broadening_override_ev: float | None = None,
) -> dict:
    """Read the self-contained Lorrax head diagnostic schema."""
    path = Path(path)
    metadata: dict[str, str] = {}
    rows: list[list[float]] = []
    column_count: int | None = None

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            item = _parse_comment_metadata(stripped)
            if item is not None:
                metadata[item[0]] = item[1]
            continue
        tokens = stripped.split()
        if column_count is None:
            column_count = len(tokens)
        if len(tokens) != column_count:
            raise ValueError(
                f"{path}:{line_number}: expected {column_count} columns, "
                f"found {len(tokens)}",
            )
        if column_count not in (3, 5):
            raise ValueError(
                f"{path}:{line_number}: expected 3 or 5 numeric columns, "
                f"found {column_count}",
            )
        try:
            rows.append([_as_float(token) for token in tokens])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric data row") from exc

    if not rows:
        raise ValueError(f"{path}: no numeric data rows")
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: data contains non-finite values")
    _validate_increasing("Lorrax omega_ev", data[:, 0])

    metadata_broadening = metadata.get("broadening_ev")
    parsed_broadening = (
        _as_float(metadata_broadening) if metadata_broadening is not None else None
    )
    if parsed_broadening is not None and parsed_broadening < 0.0:
        raise ValueError("Lorrax broadening_ev must be nonnegative")
    if broadening_override_ev is not None and broadening_override_ev < 0.0:
        raise ValueError("--lorrax-broadening-ev must be nonnegative")
    if parsed_broadening is None and broadening_override_ev is None:
        raise ValueError(
            "Lorrax input has no broadening_ev metadata; pass "
            "--lorrax-broadening-ev",
        )
    if parsed_broadening is not None and broadening_override_ev is not None:
        if not np.isclose(parsed_broadening, broadening_override_ev, atol=1.0e-12):
            raise ValueError(
                "Lorrax broadening metadata and command-line override disagree: "
                f"{parsed_broadening} versus {broadening_override_ev} eV",
            )
    broadening_ev = (
        parsed_broadening
        if parsed_broadening is not None
        else float(broadening_override_ev)
    )

    result = {
        "path": str(path),
        "format": "lorrax_text",
        "source": metadata.get("source", str(path)),
        "metadata": metadata,
        "omega_ev": data[:, 0],
        "epsinv00": data[:, 1] + 1j * data[:, 2],
        "broadening_ev": broadening_ev,
        "frequency_unit_input": "eV",
    }
    if data.shape[1] == 5:
        result["chi00"] = data[:, 3] + 1j * data[:, 4]
    return result


def read_epsinvdyn(
    path: str | Path,
    broadening_ev: float | None,
    q_index: int | None = None,
    frequency_unit: str = "eV",
) -> dict:
    """Read real-axis epsilon-head values from BerkeleyGW ``EpsInvDyn``."""
    path = Path(path)
    if broadening_ev is None:
        raise ValueError(
            "EpsInvDyn does not record real-axis broadening; pass "
            "--bgw-broadening-ev",
        )
    if broadening_ev < 0.0:
        raise ValueError("--bgw-broadening-ev must be nonnegative")

    blocks: list[dict] = []
    current: dict | None = None
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        match = _Q_HEADER.match(line.strip())
        if match:
            current = {
                "qpoint": np.asarray([_as_float(value) for value in match.groups()]),
                "rows": [],
            }
            blocks.append(current)
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current is None:
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            continue
        try:
            row = [_as_float(token) for token in tokens[:3]]
        except ValueError:
            continue
        current["rows"].append((line_number, row))

    if not blocks:
        raise ValueError(f"{path}: no '# q= ... nmtx=' blocks found")
    qpoints = np.asarray([block["qpoint"] for block in blocks])
    selected = _select_q(qpoints, q_index)
    rows = blocks[selected]["rows"]
    if not rows:
        raise ValueError(f"{path}: selected q block {selected} has no data rows")
    data = np.asarray([row for _, row in rows], dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: selected q block contains non-finite values")

    # Full-frequency EpsInvDyn appends the imaginary-axis branch after the
    # strictly increasing real-axis grid.  Its first text column is zero on
    # that branch, so the contour frequencies cannot be reconstructed here.
    real_count = 1
    while real_count < len(data):
        if data[real_count, 0] <= data[real_count - 1, 0]:
            break
        real_count += 1
    real_data = data[:real_count]
    factor = _unit_factor_to_ev(frequency_unit)
    omega_ev = real_data[:, 0] * factor
    _validate_increasing("BerkeleyGW real-axis omega_ev", omega_ev)
    return {
        "path": str(path),
        "format": "EpsInvDyn",
        "source": str(path),
        "omega_ev": omega_ev,
        "epsinv00": real_data[:, 1] + 1j * real_data[:, 2],
        "broadening_ev": float(broadening_ev),
        "qpoint": qpoints[selected],
        "q_index": selected,
        "frequency_unit_input": frequency_unit,
        "ignored_nonreal_rows": len(data) - real_count,
    }


def _normalize_n_by_width(
    values: np.ndarray,
    count: int,
    width: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(values)
    if values.shape == (count, width):
        return values
    if values.shape == (width, count):
        return values.T
    raise ValueError(
        f"{name} has shape {values.shape}; expected {(count, width)} or "
        f"{(width, count)}",
    )


def read_eps0_h5(
    path: str | Path,
    q_index: int | None = None,
    frequency_unit: str = "eV",
) -> dict:
    """Read the real-axis epsilon-inverse head from BerkeleyGW ``eps0mat.h5``."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("reading eps0mat.h5 requires h5py") from exc

    path = Path(path)
    with h5py.File(path, "r") as handle:
        matrix_type = int(np.asarray(handle["eps_header/params/matrix_type"])[()])
        if matrix_type != 0:
            raise ValueError(
                f"{path}: matrix_type={matrix_type}, expected 0 (epsilon inverse)",
            )
        nq = int(np.asarray(handle["eps_header/qpoints/nq"])[()])
        nfreq = int(np.asarray(handle["eps_header/freqs/nfreq"])[()])
        nfreq_imag = int(np.asarray(handle["eps_header/freqs/nfreq_imag"])[()])
        if nfreq_imag < 0 or nfreq_imag >= nfreq:
            raise ValueError(
                f"{path}: invalid nfreq_imag={nfreq_imag} for nfreq={nfreq}",
            )
        qpoints = _normalize_n_by_width(
            np.asarray(handle["eps_header/qpoints/qpts"]), nq, 3, "qpts",
        )
        freqs = _normalize_n_by_width(
            np.asarray(handle["eps_header/freqs/freqs"]), nfreq, 2, "freqs",
        )
        selected = _select_q(qpoints, q_index)
        matrix = handle["mats/matrix"]
        if matrix.ndim != 6:
            raise ValueError(
                f"{path}: mats/matrix has rank {matrix.ndim}; expected rank 6",
            )
        if matrix.shape[0] != nq or matrix.shape[2] != nfreq:
            raise ValueError(
                f"{path}: matrix shape {matrix.shape} disagrees with nq={nq}, "
                f"nfreq={nfreq}",
            )
        real_count = nfreq - nfreq_imag
        head_parts = np.asarray(matrix[selected, 0, :real_count, 0, 0, :])

    if head_parts.ndim != 2 or head_parts.shape[1] not in (1, 2):
        raise ValueError(
            f"{path}: matrix head has shape {head_parts.shape}; expected (nfreq, 1|2)",
        )
    epsinv00 = head_parts[:, 0].astype(float).astype(complex)
    if head_parts.shape[1] == 2:
        epsinv00 += 1j * head_parts[:, 1]
    factor = _unit_factor_to_ev(frequency_unit)
    omega_complex_ev = (freqs[:real_count, 0] + 1j * freqs[:real_count, 1]) * factor
    omega_ev = omega_complex_ev.real
    _validate_increasing("BerkeleyGW real-axis omega_ev", omega_ev)
    broadenings = omega_complex_ev.imag
    if not np.allclose(broadenings, broadenings[0], rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"{path}: real-axis broadening is not constant: "
            f"min={broadenings.min()} max={broadenings.max()} eV",
        )
    if broadenings[0] < -1.0e-12:
        raise ValueError(f"{path}: real-axis broadening is negative")
    return {
        "path": str(path),
        "format": "eps0mat.h5",
        "source": str(path),
        "omega_ev": omega_ev,
        "epsinv00": epsinv00,
        "broadening_ev": float(broadenings[0]),
        "qpoint": np.asarray(qpoints[selected], dtype=float),
        "q_index": selected,
        "frequency_unit_input": frequency_unit,
        "ignored_nonreal_rows": nfreq_imag,
    }


def align_and_compare(
    lorrax: dict,
    bgw: dict,
    frequency_tolerance_ev: float = 1.0e-8,
    broadening_tolerance_ev: float = 1.0e-8,
) -> dict:
    """Align BerkeleyGW values to Lorrax frequencies and compute differences."""
    if frequency_tolerance_ev < 0.0 or broadening_tolerance_ev < 0.0:
        raise ValueError("alignment tolerances must be nonnegative")
    lorrax_eta = float(lorrax["broadening_ev"])
    bgw_eta = float(bgw["broadening_ev"])
    if not np.isclose(
        lorrax_eta,
        bgw_eta,
        rtol=0.0,
        atol=broadening_tolerance_ev,
    ):
        raise ValueError(
            "real-axis broadenings do not match: "
            f"Lorrax={lorrax_eta:.12g} eV, BerkeleyGW={bgw_eta:.12g} eV "
            f"(tolerance {broadening_tolerance_ev:.3g} eV)",
        )

    lorrax_omega = np.asarray(lorrax["omega_ev"], dtype=float)
    bgw_omega = np.asarray(bgw["omega_ev"], dtype=float)
    _validate_increasing("Lorrax omega_ev", lorrax_omega)
    _validate_increasing("BerkeleyGW omega_ev", bgw_omega)
    direct = lorrax_omega.shape == bgw_omega.shape and np.allclose(
        lorrax_omega,
        bgw_omega,
        rtol=0.0,
        atol=frequency_tolerance_ev,
    )
    bgw_values = np.asarray(bgw["epsinv00"], dtype=complex)
    if direct:
        aligned_bgw = bgw_values.copy()
        alignment = "direct"
        interpolated_count = 0
    else:
        if (
            lorrax_omega[0] < bgw_omega[0] - frequency_tolerance_ev
            or lorrax_omega[-1] > bgw_omega[-1] + frequency_tolerance_ev
        ):
            raise ValueError(
                "Lorrax frequency grid extends outside the BerkeleyGW grid; "
                "extrapolation is disabled",
            )
        # Clamp endpoints that differ only within the declared tolerance.
        targets = lorrax_omega.copy()
        targets[np.abs(targets - bgw_omega[0]) <= frequency_tolerance_ev] = bgw_omega[0]
        targets[np.abs(targets - bgw_omega[-1]) <= frequency_tolerance_ev] = bgw_omega[-1]
        aligned_bgw = np.interp(targets, bgw_omega, bgw_values.real)
        aligned_bgw = aligned_bgw + 1j * np.interp(
            targets, bgw_omega, bgw_values.imag,
        )
        alignment = "linear_interpolation"
        exact = np.any(
            np.isclose(
                lorrax_omega[:, None],
                bgw_omega[None, :],
                rtol=0.0,
                atol=frequency_tolerance_ev,
            ),
            axis=1,
        )
        interpolated_count = int(np.count_nonzero(~exact))

    lorrax_values = np.asarray(lorrax["epsinv00"], dtype=complex)
    difference = lorrax_values - aligned_bgw
    abs_difference = np.abs(difference)
    result = {
        "omega_ev": lorrax_omega,
        "lorrax_epsinv00": lorrax_values,
        "bgw_epsinv00": aligned_bgw,
        "difference": difference,
        "abs_difference": abs_difference,
        "alignment": alignment,
        "interpolated_count": interpolated_count,
        "frequency_tolerance_ev": frequency_tolerance_ev,
        "broadening_tolerance_ev": broadening_tolerance_ev,
        "broadening_ev": lorrax_eta,
        "mae_abs": float(np.mean(abs_difference)),
        "rmse_complex": float(np.sqrt(np.mean(abs_difference**2))),
        "max_abs": float(np.max(abs_difference)),
        "max_abs_omega_ev": float(lorrax_omega[np.argmax(abs_difference)]),
    }
    if "chi00" in lorrax:
        result["lorrax_chi00"] = np.asarray(lorrax["chi00"], dtype=complex)
    return result


def write_report(path: str | Path, lorrax: dict, bgw: dict, comparison: dict) -> None:
    """Write a self-describing TSV comparison report."""
    path = Path(path)
    qpoint = bgw.get("qpoint")
    qtext = "unknown" if qpoint is None else " ".join(f"{x:.12g}" for x in qpoint)
    headers = [
        "# comparison: Lorrax minus BerkeleyGW epsinv_00(q0, omega)",
        f"# lorrax_source: {lorrax['source']}",
        f"# bgw_source: {bgw['source']}",
        f"# bgw_format: {bgw['format']}",
        f"# bgw_q_index: {bgw.get('q_index', 'unknown')}",
        f"# bgw_qpoint_crystal: {qtext}",
        "# comparison_frequency_unit: eV",
        f"# bgw_frequency_input_unit: {bgw['frequency_unit_input']}",
        f"# real_axis_broadening_ev: {comparison['broadening_ev']:.12g}",
        f"# alignment: {comparison['alignment']}",
        f"# interpolated_point_count: {comparison['interpolated_count']}",
        f"# ignored_bgw_nonreal_rows: {bgw.get('ignored_nonreal_rows', 0)}",
        f"# mae_abs: {comparison['mae_abs']:.12e}",
        f"# rmse_complex: {comparison['rmse_complex']:.12e}",
        f"# max_abs: {comparison['max_abs']:.12e}",
        f"# max_abs_omega_ev: {comparison['max_abs_omega_ev']:.12g}",
    ]
    columns = [
        "omega_ev",
        "lorrax_Re_epsinv00",
        "lorrax_Im_epsinv00",
        "bgw_Re_epsinv00",
        "bgw_Im_epsinv00",
        "diff_Re",
        "diff_Im",
        "abs_diff",
    ]
    arrays = [
        comparison["omega_ev"],
        comparison["lorrax_epsinv00"].real,
        comparison["lorrax_epsinv00"].imag,
        comparison["bgw_epsinv00"].real,
        comparison["bgw_epsinv00"].imag,
        comparison["difference"].real,
        comparison["difference"].imag,
        comparison["abs_difference"],
    ]
    if "lorrax_chi00" in comparison:
        columns.extend(["lorrax_Re_chi00", "lorrax_Im_chi00"])
        arrays.extend(
            [comparison["lorrax_chi00"].real, comparison["lorrax_chi00"].imag],
        )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(headers))
        handle.write("\n# columns: " + " ".join(columns) + "\n")
        np.savetxt(handle, np.column_stack(arrays), fmt="%.12e")


def plot_report(path: str | Path, comparison: dict) -> None:
    """Plot real, imaginary, and absolute-difference head diagnostics."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("--out-png requires matplotlib") from exc

    omega = comparison["omega_ev"]
    lorrax_values = comparison["lorrax_epsinv00"]
    bgw_values = comparison["bgw_epsinv00"]
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(7.0, 8.0))
    axes[0].plot(omega, lorrax_values.real, label="Lorrax")
    axes[0].plot(omega, bgw_values.real, "--", label="BerkeleyGW")
    axes[0].set_ylabel(r"Re $\epsilon^{-1}_{00}$")
    axes[0].legend()
    axes[1].plot(omega, lorrax_values.imag)
    axes[1].plot(omega, bgw_values.imag, "--")
    axes[1].set_ylabel(r"Im $\epsilon^{-1}_{00}$")
    axes[2].semilogy(omega, np.maximum(comparison["abs_difference"], 1.0e-18))
    axes[2].set_ylabel(r"$|\Delta\epsilon^{-1}_{00}|$")
    axes[2].set_xlabel(r"$\omega$ (eV)")
    figure.suptitle(
        f"alignment={comparison['alignment']}, "
        f"eta={comparison['broadening_ev']:.6g} eV",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _infer_bgw_format(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() in (".h5", ".hdf5") or "eps0mat" in name:
        return "eps0"
    return "epsinvdyn"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lorrax-head", required=True, type=Path)
    parser.add_argument("--bgw-head", required=True, type=Path)
    parser.add_argument(
        "--bgw-format",
        choices=("auto", "epsinvdyn", "eps0"),
        default="auto",
    )
    parser.add_argument("--bgw-q-index", type=int)
    parser.add_argument(
        "--bgw-frequency-unit",
        choices=("eV", "Ry", "Ha"),
        default="eV",
        help="unit stored in the BerkeleyGW input (default: eV)",
    )
    parser.add_argument("--lorrax-broadening-ev", type=float)
    parser.add_argument("--bgw-broadening-ev", type=float)
    parser.add_argument("--frequency-tol-ev", type=float, default=1.0e-8)
    parser.add_argument("--broadening-tol-ev", type=float, default=1.0e-8)
    parser.add_argument("--out-dat", required=True, type=Path)
    parser.add_argument("--out-png", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        lorrax = read_lorrax_head(
            args.lorrax_head,
            broadening_override_ev=args.lorrax_broadening_ev,
        )
        bgw_format = (
            _infer_bgw_format(args.bgw_head)
            if args.bgw_format == "auto"
            else args.bgw_format
        )
        if bgw_format == "epsinvdyn":
            bgw = read_epsinvdyn(
                args.bgw_head,
                broadening_ev=args.bgw_broadening_ev,
                q_index=args.bgw_q_index,
                frequency_unit=args.bgw_frequency_unit,
            )
        else:
            bgw = read_eps0_h5(
                args.bgw_head,
                q_index=args.bgw_q_index,
                frequency_unit=args.bgw_frequency_unit,
            )
            if args.bgw_broadening_ev is not None and not np.isclose(
                bgw["broadening_ev"],
                args.bgw_broadening_ev,
                rtol=0.0,
                atol=args.broadening_tol_ev,
            ):
                raise ValueError(
                    "eps0mat.h5 broadening and --bgw-broadening-ev disagree: "
                    f"{bgw['broadening_ev']} versus {args.bgw_broadening_ev} eV",
                )
        comparison = align_and_compare(
            lorrax,
            bgw,
            frequency_tolerance_ev=args.frequency_tol_ev,
            broadening_tolerance_ev=args.broadening_tol_ev,
        )
        write_report(args.out_dat, lorrax, bgw, comparison)
        if args.out_png is not None:
            plot_report(args.out_png, comparison)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    qpoint = " ".join(f"{x:.8g}" for x in bgw["qpoint"])
    print(f"BerkeleyGW q[{bgw['q_index']}] = ({qpoint})")
    print(
        f"frequency unit: eV; broadening: {comparison['broadening_ev']:.8g} eV",
    )
    print(
        f"alignment: {comparison['alignment']} "
        f"({comparison['interpolated_count']} interpolated points)",
    )
    print(
        f"N={len(comparison['omega_ev'])} "
        f"MAE(|delta|)={comparison['mae_abs']:.6e} "
        f"RMSE={comparison['rmse_complex']:.6e} "
        f"max|delta|={comparison['max_abs']:.6e} "
        f"at {comparison['max_abs_omega_ev']:.8g} eV",
    )
    print(f"wrote {args.out_dat}")
    if args.out_png is not None:
        print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

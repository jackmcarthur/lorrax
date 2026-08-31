#!/usr/bin/env python3
"""Generate and certify causal reciprocal tables for MPA crossings.

The dimensionless target is

``1 / (x + i*g)``, ``|x| <= A`` and ``1 <= g <= 100``.

Each table stores positive real times and complex weights in

``Q(x+i*g) = sum_r alpha_r exp(i*t_r*(x+i*g))``.

The advertised error is relative to the ``1/g`` envelope:

``max g * |Q(x+i*g) - 1/(x+i*g)| <= error_bound``.

Generation is offline.  The positive Gauss rule supplies ``alpha=-i*h``.
Certification is independent of the solver's training grid.  It bounds the
requested metric by the maximum-modulus residual
``|1-z*sum(h*exp(-z*t))|``, ``z=g-i*x``, and finds that maximum on all four
rectangle edges using dense held-out grids, analytic derivative roots, and
bounded local-extremum refinement.  A dense two-dimensional grid then
re-scores the requested family metric directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
from scipy.optimize import brentq, minimize_scalar


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
MINIMAX_SRC = REPO_ROOT / "services" / "minimax" / "src"
for source in (SRC_ROOT, MINIMAX_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from gw.mpa.evaluator import damped_rectangle_gauss_rule  # noqa: E402
from minimax import payload_sha256  # noqa: E402


ASSET_ROOT = (MINIMAX_SRC / "minimax" / "minimax_assets")
CATALOG_PATH = ASSET_ROOT / "catalog.json"
A_VALUES = (10.0, 20.0, 40.0)
ERROR_BOUNDS = (1.0e-4, 1.0e-5)
G_MAX = 100.0
MEASURED_NA_G_MAX = 90.1414665872282
MAX_NODES = 420
KAPPA_HARD_CAP = 10.0
KAPPA_PREFERRED_CAP = 2.0
EDGE_GRID_SIZE = 32_769
G_EDGE_GRID_SIZE = 16_385
REFERENCE_X_SIZE = 4_097
REFERENCE_G_SIZE = 257


def _float_token(value: float) -> str:
    return f"{float(value):.6f}".replace("-", "m").replace(".", "p")


def _error_token(value: float) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e")
    return (mantissa.replace("-", "m").replace(".", "p")
            + "e" + exponent.replace("+", "p").replace("-", "m"))


def _residual(tau: np.ndarray, h: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return ``1-z*sum(h*exp(-z*t))`` without a large temporary."""
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.empty(z.size, dtype=np.complex128)
    for start in range(0, z.size, 2048):
        part = z[start:start + 2048]
        summed = np.exp(-part[:, None] * tau[None, :]) @ h
        out[start:start + part.size] = 1.0 - part * summed
    return out


def _residual_and_derivative(
    tau: np.ndarray,
    h: np.ndarray,
    z: complex,
) -> tuple[complex, complex]:
    """Residual and its analytic derivative with respect to ``z``."""
    phase = np.exp(-complex(z) * tau)
    summed = np.dot(phase, h)
    moment = np.dot(phase, h * tau)
    residual = 1.0 - complex(z) * summed
    derivative = -summed + complex(z) * moment
    return complex(residual), complex(derivative)


def _edge_certificate(
    tau: np.ndarray,
    h: np.ndarray,
    parameter: np.ndarray,
    z_of_parameter,
    dz_of_parameter,
) -> dict[str, object]:
    """Find every resolved maximum of one analytic rectangle edge."""
    parameter = np.asarray(parameter, dtype=np.float64)
    z = np.asarray(z_of_parameter(parameter), dtype=np.complex128)
    residual = _residual(tau, h, z)

    def value(s: float) -> float:
        r, _dr = _residual_and_derivative(tau, h, z_of_parameter(float(s)))
        return abs(r)

    def slope(s: float) -> float:
        r, dr = _residual_and_derivative(tau, h, z_of_parameter(float(s)))
        return float(2.0 * np.real(
            np.conj(r) * dr * dz_of_parameter(float(s))))

    slopes = np.asarray([slope(float(s)) for s in parameter])
    roots: list[float] = []
    for lo, hi, left, right in zip(
            parameter[:-1], parameter[1:], slopes[:-1], slopes[1:]):
        if left == 0.0:
            roots.append(float(lo))
        elif left * right < 0.0:
            roots.append(float(brentq(
                slope, float(lo), float(hi),
                xtol=np.nextafter(0.0, 1.0),
                rtol=4.0 * np.finfo(np.float64).eps,
            )))

    magnitude = np.abs(residual)
    local = np.flatnonzero(
        (magnitude[1:-1] >= magnitude[:-2])
        & (magnitude[1:-1] >= magnitude[2:])) + 1
    refined: list[float] = []
    for index in local:
        result = minimize_scalar(
            lambda s: -value(float(s)),
            bounds=(float(parameter[index - 1]),
                    float(parameter[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-15, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError("causal crossing extremum refinement failed")
        refined.append(float(result.x))

    candidates = np.unique(np.concatenate((
        parameter[[0, -1]],
        np.asarray(roots, dtype=np.float64),
        np.asarray(refined, dtype=np.float64),
    )))
    maximum = max(value(float(s)) for s in candidates)
    return {
        "max_error": float(maximum),
        "held_out_max_error": float(np.max(magnitude)),
        "derivative_root_count": len(roots),
        "local_refinement_count": len(refined),
    }


def _dense_family_error(
    tau: np.ndarray,
    alpha: np.ndarray,
    A: float,
) -> float:
    """Dense held-out score of the exact advertised ``g*abs(error)``."""
    x = np.linspace(-float(A), float(A), REFERENCE_X_SIZE)
    g = np.geomspace(1.0, G_MAX, REFERENCE_G_SIZE)
    worst = 0.0
    for gamma in g:
        d = x + 1.0j * gamma
        fitted = np.empty(x.size, dtype=np.complex128)
        for start in range(0, x.size, 2048):
            part = d[start:start + 2048]
            fitted[start:start + part.size] = (
                np.exp(1.0j * part[:, None] * tau[None, :]) @ alpha)
        error = gamma * np.abs(fitted - 1.0 / d)
        worst = max(worst, float(np.max(error)))
    return worst


def _kappa0(tau: np.ndarray, alpha: np.ndarray) -> float:
    """Maximum ``g*sum|alpha|*exp(-g*t)`` on the shipped width range."""
    g = np.geomspace(1.0, G_MAX, 20_001)
    envelope = g * (np.exp(-g[:, None] * tau[None, :])
                    @ np.abs(alpha))
    index = int(np.argmax(envelope))
    left = g[max(0, index - 1)]
    right = g[min(g.size - 1, index + 1)]
    if left == right:
        return float(envelope[index])
    result = minimize_scalar(
        lambda value: -float(value * np.dot(
            np.exp(-value * tau), np.abs(alpha))),
        bounds=(float(left), float(right)), method="bounded",
        options={"xatol": 1.0e-15, "maxiter": 1000},
    )
    return max(float(envelope[index]), float(-result.fun))


def certify(
    tau: np.ndarray,
    alpha: np.ndarray,
    A: float,
    error_bound: float,
) -> dict[str, object]:
    """Certify one final numerical payload against its complete family."""
    tau = np.asarray(tau, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.complex128)
    h = np.real(1.0j * alpha)
    if (tau.ndim != 1 or alpha.shape != tau.shape or not tau.size
            or not np.all(np.isfinite(tau))
            or not np.all(np.isfinite(alpha))):
        raise ValueError("causal crossing payload must be finite 1-D arrays")
    if np.any(tau <= 0.0):
        raise ValueError("causal crossing times must be positive")
    if not np.array_equal(alpha, -1.0j * h) or np.any(h <= 0.0):
        raise ValueError("causal crossing alpha must be negative imaginary")

    x_grid = np.linspace(-float(A), float(A), EDGE_GRID_SIZE)
    log_g_grid = np.linspace(0.0, np.log(G_MAX), G_EDGE_GRID_SIZE)
    edges = [
        _edge_certificate(
            tau, h, x_grid,
            lambda x, g=1.0: g - 1.0j * np.asarray(x),
            lambda _x: -1.0j),
        _edge_certificate(
            tau, h, x_grid,
            lambda x, g=G_MAX: g - 1.0j * np.asarray(x),
            lambda _x: -1.0j),
        _edge_certificate(
            tau, h, log_g_grid,
            lambda log_g, x=-float(A): np.exp(log_g) - 1.0j * x,
            lambda log_g: np.exp(log_g)),
        _edge_certificate(
            tau, h, log_g_grid,
            lambda log_g, x=float(A): np.exp(log_g) - 1.0j * x,
            lambda log_g: np.exp(log_g)),
    ]
    certified_bound = max(float(edge["max_error"]) for edge in edges)
    measured_error = _dense_family_error(tau, alpha, float(A))
    kappa0 = _kappa0(tau, alpha)
    failures = []
    if certified_bound > float(error_bound):
        failures.append("certified_error")
    if measured_error > certified_bound * (1.0 + 2.0e-12):
        failures.append("dense_reference_exceeds_certificate")
    if kappa0 > KAPPA_HARD_CAP:
        failures.append("kappa0")
    return {
        "certified": not failures,
        "failures": failures,
        "max_error": certified_bound,
        "measured_family_error": measured_error,
        "held_out_max_error": max(
            float(edge["held_out_max_error"]) for edge in edges),
        "derivative_root_count": sum(
            int(edge["derivative_root_count"]) for edge in edges),
        "local_refinement_count": sum(
            int(edge["local_refinement_count"]) for edge in edges),
        "kappa0": kappa0,
        "sum_abs_alpha": float(np.sum(np.abs(alpha))),
        "payload_sha256": payload_sha256(tau, alpha),
    }


def _provenance() -> dict[str, str]:
    tool = Path(__file__).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    backend = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
    }
    backend_sha = hashlib.sha256(
        json.dumps(backend, sort_keys=True).encode()).hexdigest()
    return {
        "tool": "tools/generate_causal_crossing_assets.py",
        "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "generator_commit": commit,
        **backend,
        "certifier": "causal-rectangle-boundary-extrema+dense-family/v1",
        "backend_sha256": backend_sha,
    }


def _table_path(A: float, error_bound: float) -> Path:
    name = (f"crossing_causal_A_{_float_token(A)}_G_"
            f"{_float_token(G_MAX)}_eps_{_error_token(error_bound)}.npz")
    return ASSET_ROOT / "crossing_causal" / name


def _write_table(
    path: Path,
    tau: np.ndarray,
    alpha: np.ndarray,
    certificate: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            tau=np.asarray(tau, dtype=np.float64),
            alpha=np.asarray(alpha, dtype=np.complex128),
            max_error=np.asarray(certificate["max_error"], np.float64),
            measured_family_error=np.asarray(
                certificate["measured_family_error"], np.float64),
            kappa0=np.asarray(certificate["kappa0"], np.float64),
            payload_sha256=np.asarray(certificate["payload_sha256"]),
        )


def _entry(
    path: Path,
    A: float,
    error_bound: float,
    tau: np.ndarray,
    certificate: dict[str, object],
    provenance: dict[str, str],
) -> dict[str, object]:
    return {
        "family": "crossing_causal",
        "target_kind": "causal_reciprocal",
        "error_bound": float(error_bound),
        "error_metric": "linf_g_times_abs_reciprocal_error",
        "range_param": "A_dim",
        "range_max": float(A),
        "width_ratio_max": G_MAX,
        "measured_na_width_ratio_max": MEASURED_NA_G_MAX,
        "node_count": int(tau.size),
        "max_error": float(certificate["max_error"]),
        "measured_family_error": float(
            certificate["measured_family_error"]),
        "held_out_max_error": float(certificate["held_out_max_error"]),
        "kappa0": float(certificate["kappa0"]),
        "kappa0_preferred_bound": KAPPA_PREFERRED_CAP,
        "kappa0_hard_bound": KAPPA_HARD_CAP,
        "sum_abs_alpha": float(certificate["sum_abs_alpha"]),
        "payload_sha256": str(certificate["payload_sha256"]),
        "certified": True,
        "certification": {
            "method": ("maximum-modulus boundary certificate with analytic "
                       "derivative roots, bounded local extrema, and dense "
                       "two-dimensional family re-score"),
            "edge_grid_size": EDGE_GRID_SIZE,
            "gamma_edge_grid_size": G_EDGE_GRID_SIZE,
            "reference_x_size": REFERENCE_X_SIZE,
            "reference_gamma_size": REFERENCE_G_SIZE,
            "derivative_root_count": int(
                certificate["derivative_root_count"]),
            "local_refinement_count": int(
                certificate["local_refinement_count"]),
            "checks": [
                "positive_real_times",
                "negative_imaginary_weights",
                "certified_error",
                "dense_family_reference",
                "kappa0",
            ],
        },
        "provenance": provenance,
        "file": str(path.relative_to(ASSET_ROOT)),
    }


def _update_catalog(entries: list[dict[str, object]]) -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog["conventions"]["crossing_causal"] = {
        "problem": "1/(x+i*g) on x in [-A_dim,A_dim], g in [1,100]",
        "representation": "Q(z)=sum alpha_r exp(i*tau_r*z), tau_r>0",
        "error_metric": "max g*abs(Q(x+i*g)-1/(x+i*g))",
        "physical_error_note": ("runtime scales z by gamma_min, so absolute "
                                "error is bounded by error_bound/gamma_min"),
        "width_ratio_max": G_MAX,
    }
    catalog["sweeps"]["crossing_causal_A_dim_values"] = list(A_VALUES)
    catalog["sweeps"]["crossing_causal_error_bounds"] = list(ERROR_BOUNDS)
    old = [row for row in catalog["tables"]
           if row.get("family") != "crossing_causal"]
    insert = next((i for i, row in enumerate(old)
                   if row.get("family") == "noncrossing"), len(old))
    catalog["tables"] = old[:insert] + entries + old[insert:]
    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")


def generate(*, write_assets: bool) -> list[dict[str, object]]:
    """Generate, certify, optionally write, and print the six tables."""
    provenance = _provenance()
    entries = []
    for A in A_VALUES:
        for error_bound in ERROR_BOUNDS:
            rule = damped_rectangle_gauss_rule(
                1.0, G_MAX, A, rel_tol=error_bound / 4.0,
                max_nodes=MAX_NODES)
            tau = np.asarray(rule["t"], dtype=np.float64)
            alpha = -1.0j * np.asarray(rule["h"], dtype=np.float64)
            certificate = certify(tau, alpha, A, error_bound)
            if not certificate["certified"]:
                raise RuntimeError(
                    f"A={A:g} eps={error_bound:g} failed "
                    f"{certificate['failures']}")
            path = _table_path(A, error_bound)
            entries.append(_entry(
                path, A, error_bound, tau, certificate, provenance))
            if write_assets:
                _write_table(path, tau, alpha, certificate)
            print(
                f"A={A:g} G={G_MAX:g} eps={error_bound:.0e} "
                f"nodes={tau.size} measured="
                f"{certificate['measured_family_error']:.12e} "
                f"certified={certificate['max_error']:.12e} "
                f"kappa0={certificate['kappa0']:.12e}")
    if write_assets:
        _update_catalog(entries)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-assets", action="store_true",
        help="Write the six payloads and merge their rows into catalog.json.")
    args = parser.parse_args()
    generate(write_assets=bool(args.write_assets))


if __name__ == "__main__":
    main()

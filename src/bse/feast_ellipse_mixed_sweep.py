"""Ellipse FEAST sweep with mixed quadrature across two iterations.

Iteration 1 uses n_quad in {4,6,8}, iteration 2 always uses n_quad=8.
Sweeps gamma and GMRES tolerance; reports accuracy vs exact eigenvalues.
"""
from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import build_bse_ring_matvec, make_bse_shardings
from .bse_feast import (
    WindowSpec,
    RY_TO_EV_DEFAULT,
    feast_ellipse_quadrature,
    estimate_spectral_bounds_sharded,
    build_preconditioner_diagonal_sharded,
    _get_feast_runner,
    _rayleigh_ritz,
    _build_ritz_vectors,
)
from .feast_sweep import EXACT_EIGENVALUES_EV

jax.config.update("jax_enable_x64", True)


GAMMA_FIXED = 0.2


@dataclass(frozen=True)
class EllipseSweepConfig:
    n_quad1: int
    n_quad2: int
    gmres_tol: float


@dataclass
class EllipseSweepResult:
    config: EllipseSweepConfig
    ritz_evals_eV: list[float]
    max_error_meV: float
    mean_error_meV: float
    n_matched: int
    gmres_avg: list[float]
    gmres_max: list[int]
    gmres_min: list[int]
    total_matvecs: list[int]
    wall_time_s: float


def _match_eigenvalues(
    ritz_eV: np.ndarray,
    exact_eV: np.ndarray,
    window_a: float,
    window_b: float,
) -> tuple[float, float, int]:
    exact_in = exact_eV[(exact_eV >= window_a) & (exact_eV <= window_b)]
    if len(ritz_eV) == 0 or len(exact_in) == 0:
        return 0.0, 0.0, 0
    n_match = min(len(ritz_eV), len(exact_in))
    errors = np.abs(ritz_eV[:n_match] - exact_in[:n_match]) * 1000.0
    return float(errors.max()), float(errors.mean()), int(n_match)


def _create_mesh_xy(px: int, py: int) -> Mesh:
    devices = jax.devices()
    n_devices = len(devices)
    if px * py > n_devices:
        raise ValueError(f"Requested px*py={px*py} devices, but only {n_devices} available")
    mesh_devices = np.array(devices[: px * py]).reshape(px, py)
    return Mesh(mesh_devices, axis_names=("x", "y"))


def run_sweep(
    data: dict,
    mesh_xy: Mesh,
    window: WindowSpec,
    n_ritz: int,
    gmres_max_iter: int,
    configs: list[EllipseSweepConfig],
    exact_eV: np.ndarray,
    seed: int,
    ry_to_ev: float,
) -> list[EllipseSweepResult]:
    results: list[EllipseSweepResult] = []
    sh = make_bse_shardings(mesh_xy)
    matvec = build_bse_ring_matvec(mesh_xy, data["nkx"], data["nky"], data["nkz"])

    if "W_R" not in data:
        data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")
    diag_h = build_preconditioner_diagonal_sharded(data, mesh_xy)

    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])

    for idx, cfg in enumerate(configs, start=1):
        print("\n" + "=" * 72)
    print(
        f"Ellipse sweep {idx}/{len(configs)}: "
        f"n_quad1={cfg.n_quad1}, n_quad2={cfg.n_quad2}, "
        f"gamma={GAMMA_FIXED:.2f}, gmres_tol={cfg.gmres_tol:.1e}"
    )
        print("=" * 72)

        runner1 = _get_feast_runner(
            matvec,
            data,
            cfg.n_quad1,
            n_ritz,
            gmres_max_iter,
            cfg.gmres_tol,
            ry_to_ev,
        )
        runner2 = _get_feast_runner(
            matvec,
            data,
            cfg.n_quad2,
            n_ritz,
            gmres_max_iter,
            cfg.gmres_tol,
            ry_to_ev,
        )

        z1, w1 = feast_ellipse_quadrature(window, cfg.n_quad1, GAMMA_FIXED)
        z2, w2 = feast_ellipse_quadrature(window, cfg.n_quad2, GAMMA_FIXED)
        z1_jnp = jnp.asarray(z1)
        w1_jnp = jnp.asarray(w1)
        z2_jnp = jnp.asarray(z2)
        w2_jnp = jnp.asarray(w2)

        key = jax.random.PRNGKey(seed + idx)
        X_list = []
        for _ in range(n_ritz):
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
            x = x / jnp.linalg.norm(x)
            x = jax.lax.with_sharding_constraint(x, sh.X)
            X_list.append(x.astype(jnp.complex128))
        X_batch = jnp.stack(X_list, axis=0)

        gmres_avg: list[float] = []
        gmres_max: list[int] = []
        gmres_min: list[int] = []
        total_matvecs: list[int] = []

        t0 = time.time()
        # Iteration 1 (n_quad1)
        print("  FEAST iteration 1/2 (n_quad1)")
        filtered_batch, iters = runner1(X_batch, z1_jnp, w1_jnp, diag_h)
        iters_host = np.asarray(jax.device_get(iters))
        avg_i = float(iters_host.mean())
        max_i = int(iters_host.max())
        min_i = int(iters_host.min())
        total_i = int(iters_host.sum() + n_ritz * cfg.n_quad1)
        gmres_avg.append(avg_i)
        gmres_max.append(max_i)
        gmres_min.append(min_i)
        total_matvecs.append(total_i)
        print(f"    GMRES iters: avg={avg_i:.2f}, min={min_i}, max={max_i}, total_matvecs={total_i}")

        filtered = [filtered_batch[i] for i in range(n_ritz)]
        ritz_result = _rayleigh_ritz(
            matvec, filtered, data, s_cutoff=0.01
        )
        ritz_vecs = _build_ritz_vectors(filtered, ritz_result.ritz_coeffs, sh.X)

        X_next = [v.astype(jnp.complex128) for v in ritz_vecs]
        while len(X_next) < n_ritz:
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
            x = x / jnp.linalg.norm(x)
            x = jax.lax.with_sharding_constraint(x, sh.X)
            X_next.append(x.astype(jnp.complex128))
        X_batch = jnp.stack(X_next, axis=0)

        # Iteration 2 (n_quad2, verbose)
        print("  FEAST iteration 2/2 (n_quad2)")
        filtered_batch, iters = runner2(X_batch, z2_jnp, w2_jnp, diag_h)
        iters_host = np.asarray(jax.device_get(iters))
        avg_i = float(iters_host.mean())
        max_i = int(iters_host.max())
        min_i = int(iters_host.min())
        total_i = int(iters_host.sum() + n_ritz * cfg.n_quad2)
        gmres_avg.append(avg_i)
        gmres_max.append(max_i)
        gmres_min.append(min_i)
        total_matvecs.append(total_i)
        print(f"    GMRES iters: avg={avg_i:.2f}, min={min_i}, max={max_i}, total_matvecs={total_i}")

        filtered = [filtered_batch[i] for i in range(n_ritz)]
        ritz_result = _rayleigh_ritz(
            matvec, filtered, data, s_cutoff=0.01
        )
        ev_str = ", ".join(f"{v * ry_to_ev:.6f}" for v in ritz_result.ritz_evals)
        print(f"    Ritz evals (eV): [{ev_str}]")

        wall = time.time() - t0

        ritz_eV = np.array([v * ry_to_ev for v in ritz_result.ritz_evals], dtype=np.float64)
        max_err, mean_err, n_match = _match_eigenvalues(
            ritz_eV, exact_eV, window.a_eV, window.b_eV
        )

        results.append(EllipseSweepResult(
            config=cfg,
            ritz_evals_eV=[float(v) for v in ritz_eV],
            max_error_meV=max_err,
            mean_error_meV=mean_err,
            n_matched=n_match,
            gmres_avg=gmres_avg,
            gmres_max=gmres_max,
            gmres_min=gmres_min,
            total_matvecs=total_matvecs,
            wall_time_s=wall,
        ))

        print(f"  Accuracy vs exact (matched={n_match}): max={max_err:.2f} meV, mean={mean_err:.2f} meV")
        print(f"  Wall time: {wall:.2f} s")

    return results


def print_summary(results: list[EllipseSweepResult]) -> None:
    print("\n" + "=" * 72)
    print(f"ELLIPSE MIXED-QUAD SWEEP SUMMARY (window [0, 2] eV, gamma={GAMMA_FIXED:.2f})")
    print("=" * 72)
    header = (
        f"{'nq1->nq2':>9} {'tol':>7} "
        f"{'gmres_avg1':>10} {'gmres_avg2':>10} {'matvecs':>8} "
        f"{'max_err(meV)':>13} {'time(s)':>8}"
    )
    print(header)
    for r in results:
        matvec_total = int(sum(r.total_matvecs))
        print(
            f"{r.config.n_quad1:2d}->{r.config.n_quad2:<2d} "
            f"{r.config.gmres_tol:7.0e} {r.gmres_avg[0]:10.2f} {r.gmres_avg[1]:10.2f} "
            f"{matvec_total:8d} {r.max_error_meV:13.2f} {r.wall_time_s:8.2f}"
        )
    print("=" * 72)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ellipse FEAST mixed-quad sweep (window [0,2] eV)")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--n-ritz", type=int, default=8)
    parser.add_argument("--gmres-max-iter", type=int, default=20)
    parser.add_argument("--gmres-tol", type=float, nargs="+", default=[1e-2])
    parser.add_argument("--n-quad1", type=int, nargs="+", default=[4])
    parser.add_argument("--n-quad2", type=int, default=8)
    parser.add_argument("--buffer", type=float, default=0.05, help="E_max buffer factor")
    parser.add_argument("--n-lanczos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--units-ev-per-ry", type=float, default=RY_TO_EV_DEFAULT)
    args = parser.parse_args(argv)

    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)

    print("Loading BSE data...", flush=True)
    data = load_bse_data_from_restart_sharded(
        restart_file,
        n_val=args.n_val,
        n_cond=args.n_cond,
        mesh_xy=mesh_xy,
        input_file=args.input,
    )

    print("Estimating spectral bounds...", flush=True)
    bounds = estimate_spectral_bounds_sharded(
        data,
        mesh_xy,
        n_lanczos=args.n_lanczos,
    )
    e_min_ry = bounds["e_min_ry"]
    e_max_ry = bounds["e_max_ry_raw"] * (1.0 + args.buffer)
    e_min_eV = e_min_ry * args.units_ev_per_ry
    e_max_eV = e_max_ry * args.units_ev_per_ry
    print(f"E_min = {e_min_eV:.3f} eV, E_max = {e_max_eV:.3f} eV", flush=True)

    window = WindowSpec("W1", 0.0, 2.0, "ellipse sweep")
    exact_eV = EXACT_EIGENVALUES_EV
    exact_in = exact_eV[(exact_eV >= window.a_eV) & (exact_eV <= window.b_eV)]
    print(f"Using hardcoded exact eigenvalues in window: {np.array2string(exact_in, precision=6)}")

    configs = [
        EllipseSweepConfig(nq1, args.n_quad2, tol)
        for nq1, tol in itertools.product(args.n_quad1, args.gmres_tol)
    ]

    results = run_sweep(
        data=data,
        mesh_xy=mesh_xy,
        window=window,
        n_ritz=args.n_ritz,
        gmres_max_iter=args.gmres_max_iter,
        configs=configs,
        exact_eV=exact_eV,
        seed=args.seed,
        ry_to_ev=args.units_ev_per_ry,
    )

    print_summary(results)


if __name__ == "__main__":
    main()

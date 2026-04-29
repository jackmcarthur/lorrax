"""Zolotarev FEAST sweep focused on the first window [0, 2] eV.

Runs a small parameter sweep over:
  - n_quad: 4, 6, 8
  - FEAST iterations: 1, 2
  - GMRES tolerances: 1e-2, 1e-3
  - Zolotarev transition width scale (rho_scale)

Reports Ritz values in the window and accuracy vs exact eigenvalues (if provided).
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
    estimate_spectral_bounds_sharded,
    build_preconditioner_diagonal_sharded,
    _get_feast_runner,
    _rayleigh_ritz,
    _build_ritz_vectors,
    _zolotarev_step_poles_weights,
)
from .feast_sweep import EXACT_EIGENVALUES_EV

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ZoloSweepConfig:
    n_quad: int
    rho_scale: float
    feast_iter: int
    gmres_tol: float


@dataclass
class ZoloSweepResult:
    config: ZoloSweepConfig
    ritz_evals_eV: list[float]
    max_error_meV: float
    mean_error_meV: float
    n_matched: int
    gmres_avg: list[float]
    gmres_max: list[int]
    gmres_min: list[int]
    total_matvecs: list[int]
    wall_time_s: float


def _zolotarev_quadrature_custom(
    window: WindowSpec,
    n_quad: int,
    lambda_min_eV: float,
    lambda_max_eV: float,
    rho_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Zolotarev quadrature with adjustable transition width scale."""
    a = window.a_eV
    b = window.b_eV
    rho = rho_scale * 0.5 * (b - a)
    n_left = n_quad // 2
    n_right = n_quad - n_left

    z_L, w_L = _zolotarev_step_poles_weights(a, n_left, lambda_min_eV, lambda_max_eV, rho)
    z_R, w_R = _zolotarev_step_poles_weights(b, n_right, lambda_min_eV, lambda_max_eV, rho)
    z_nodes = np.concatenate([z_L, z_R])
    w_weights = np.concatenate([w_L, -w_R])
    return z_nodes, w_weights


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
    bounds_eV: tuple[float, float],
    configs: list[ZoloSweepConfig],
    exact_eV: np.ndarray,
    seed: int,
    ry_to_ev: float,
) -> list[ZoloSweepResult]:
    results: list[ZoloSweepResult] = []
    sh = make_bse_shardings(mesh_xy)
    matvec = build_bse_ring_matvec(mesh_xy, data["nkx"], data["nky"], data["nkz"])

    if "W_R" not in data:
        data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")
    diag_h = build_preconditioner_diagonal_sharded(data, mesh_xy)

    lambda_min_eV, lambda_max_eV = bounds_eV
    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])

    for idx, cfg in enumerate(configs, start=1):
        print("\n" + "=" * 72)
        print(
            f"Zolotarev sweep {idx}/{len(configs)}: "
            f"n_quad={cfg.n_quad}, rho_scale={cfg.rho_scale:.2f}, "
            f"feast_iter={cfg.feast_iter}, gmres_tol={cfg.gmres_tol:.1e}"
        )
        print("=" * 72)

        runner = _get_feast_runner(
            matvec,
            data,
            cfg.n_quad,
            n_ritz,
            gmres_max_iter,
            cfg.gmres_tol,
            ry_to_ev,
        )

        z_nodes, w_weights = _zolotarev_quadrature_custom(
            window,
            cfg.n_quad,
            lambda_min_eV,
            lambda_max_eV,
            cfg.rho_scale,
        )
        z_jnp = jnp.asarray(z_nodes)
        w_jnp = jnp.asarray(w_weights)

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

        ritz_result = None
        t0 = time.time()
        for it in range(cfg.feast_iter):
            is_last = (it == cfg.feast_iter - 1)
            print(f"  FEAST iteration {it + 1}/{cfg.feast_iter}")

            filtered_batch, iters = runner(X_batch, z_jnp, w_jnp, diag_h)
            iters_host = np.asarray(jax.device_get(iters))

            avg_i = float(iters_host.mean())
            max_i = int(iters_host.max())
            min_i = int(iters_host.min())
            total_i = int(iters_host.sum() + n_ritz * cfg.n_quad)
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

            if not is_last:
                ritz_vecs = _build_ritz_vectors(filtered, ritz_result.ritz_coeffs, sh.X)
                X_next = [v.astype(jnp.complex128) for v in ritz_vecs]
                while len(X_next) < n_ritz:
                    key, subkey = jax.random.split(key)
                    x = jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
                    x = x / jnp.linalg.norm(x)
                    x = jax.lax.with_sharding_constraint(x, sh.X)
                    X_next.append(x.astype(jnp.complex128))
                X_batch = jnp.stack(X_next, axis=0)

        wall = time.time() - t0
        assert ritz_result is not None

        ritz_eV = np.array([v * ry_to_ev for v in ritz_result.ritz_evals], dtype=np.float64)
        max_err, mean_err, n_match = _match_eigenvalues(
            ritz_eV, exact_eV, window.a_eV, window.b_eV
        )

        results.append(ZoloSweepResult(
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


def print_summary(results: list[ZoloSweepResult]) -> None:
    print("\n" + "=" * 72)
    print("ZOLOTAREV SWEEP SUMMARY (window [0, 2] eV)")
    print("=" * 72)
    header = (
        f"{'nq':>3} {'rho':>4} {'fi':>2} {'tol':>7} "
        f"{'gmres_avg':>10} {'gmres_max':>10} {'max_err(meV)':>13} {'time(s)':>8}"
    )
    print(header)
    for r in results:
        avg_last = r.gmres_avg[-1]
        max_last = r.gmres_max[-1]
        print(
            f"{r.config.n_quad:3d} {r.config.rho_scale:4.2f} {r.config.feast_iter:2d} "
            f"{r.config.gmres_tol:7.0e} {avg_last:10.2f} {max_last:10d} "
            f"{r.max_error_meV:13.2f} {r.wall_time_s:8.2f}"
        )
    print("=" * 72)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Zolotarev FEAST sweep (window [0,2] eV)")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--n-ritz", type=int, default=8)
    parser.add_argument("--gmres-max-iter", type=int, default=20)
    parser.add_argument("--gmres-tol", type=float, nargs="+", default=[1e-2, 1e-3])
    parser.add_argument("--n-quad", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--feast-iter", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--rho-scale", type=float, nargs="+", default=[0.5, 1.0, 1.5])
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

    window = WindowSpec("W1", 0.0, 2.0, "zolotarev sweep")
    exact_eV = EXACT_EIGENVALUES_EV
    exact_in = exact_eV[(exact_eV >= window.a_eV) & (exact_eV <= window.b_eV)]
    print(f"Using hardcoded exact eigenvalues in window: {np.array2string(exact_in, precision=6)}")

    configs = [
        ZoloSweepConfig(nq, rho, fi, tol)
        for nq, rho, fi, tol in itertools.product(
            args.n_quad, args.rho_scale, args.feast_iter, args.gmres_tol
        )
    ]

    results = run_sweep(
        data=data,
        mesh_xy=mesh_xy,
        window=window,
        n_ritz=args.n_ritz,
        gmres_max_iter=args.gmres_max_iter,
        bounds_eV=(e_min_eV, e_max_eV),
        configs=configs,
        exact_eV=exact_eV,
        seed=args.seed,
        ry_to_ev=args.units_ev_per_ry,
    )

    print_summary(results)


if __name__ == "__main__":
    main()

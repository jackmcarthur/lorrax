"""FEAST parameter sweep for convergence analysis.

Loads BSE data once, computes exact eigenvalues via full diagonalization,
then sweeps FEAST parameters (n_quad, gamma, n_ritz, window, feast_iter)
and reports accuracy vs cost.

Usage:
    JAX_PLATFORMS=cpu uv run python -m bse.feast_sweep \
        -i projects/test_isdf/cohsex_test.in --n-val 4 --n-cond 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass, asdict

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_serial import apply_bse_hamiltonian_single_device
from .bse_feast import (
    WindowSpec,
    RY_TO_EV_DEFAULT,
    run_feast_ritz,
    _create_mesh_xy,
    estimate_spectral_bounds_sharded,
)

jax.config.update("jax_enable_x64", True)


@dataclass
class SweepResult:
    """Result of a single FEAST sweep point."""
    window_a: float
    window_b: float
    n_quad: int
    gamma: float
    n_ritz: int
    gmres_tol: float
    feast_iter: int
    ritz_evals_eV: list[float]
    n_physical: int
    n_total: int
    s_evals: list[float]
    wall_time_s: float
    # Derived (filled in post-processing):
    max_error_meV: float = 0.0
    mean_error_meV: float = 0.0
    n_matched: int = 0
    total_matvecs_est: int = 0


def build_full_bse_matrix(data: dict, ry_to_ev: float = RY_TO_EV_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Build the full dense BSE Hamiltonian and diagonalize it.

    Returns (eigenvalues_eV, eigenvectors) sorted ascending.
    N = n_cond * n_val * nk is typically small for test systems.
    """
    n_cond = int(data["n_cond"])
    n_val = int(data["n_val"])
    nkx = int(data["nkx"])
    nky = int(data["nky"])
    nkz = int(data["nkz"])
    nk = nkx * nky * nkz
    N = n_cond * n_val * nk

    # We need un-padded data for single-device matvec.
    psi_c = np.asarray(jax.device_get(data["psi_c_X"]))[:, :n_cond, :, :]
    psi_v = np.asarray(jax.device_get(data["psi_v_X"]))[:, :n_val, :, :]
    eps_c = np.asarray(jax.device_get(data["eps_c"]))[:, :n_cond]
    eps_v = np.asarray(jax.device_get(data["eps_v"]))[:, :n_val]
    W_q = np.asarray(jax.device_get(data["W_q"]))
    V_q0 = np.asarray(jax.device_get(data["V_q0"]))

    psi_c_j = jnp.asarray(psi_c)
    psi_v_j = jnp.asarray(psi_v)
    eps_c_j = jnp.asarray(eps_c)
    eps_v_j = jnp.asarray(eps_v)
    W_q_j = jnp.asarray(W_q)
    V_q0_j = jnp.asarray(V_q0)

    print(f"Building full BSE matrix: N = {n_cond} × {n_val} × {nk} = {N}")

    # Build matrix column by column using the matvec.
    H = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        # Construct unit vector i in (1, n_cond, n_val, nk) layout.
        e_i = np.zeros((1, n_cond, n_val, nk), dtype=np.float64)
        c_idx = (i // (n_val * nk))
        v_idx = (i % (n_val * nk)) // nk
        k_idx = i % nk
        e_i[0, c_idx, v_idx, k_idx] = 1.0

        he_i = apply_bse_hamiltonian_single_device(
            jnp.asarray(e_i),
            psi_c_j, psi_v_j, eps_c_j, eps_v_j,
            W_q_j, V_q0_j, nkx, nky, nkz,
        )
        H[:, i] = np.asarray(jax.device_get(he_i)).reshape(N)

    # Symmetrize (BSE-TDA is Hermitian).
    H = 0.5 * (H + H.T)

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues_eV = eigenvalues * ry_to_ev
    print(f"Exact eigenvalues (eV), first 10: {np.array2string(eigenvalues_eV[:10], precision=6)}")
    return eigenvalues_eV, eigenvectors


def match_eigenvalues(
    ritz_eV: np.ndarray,
    exact_eV: np.ndarray,
    window_a: float,
    window_b: float,
) -> tuple[float, float, int]:
    """Match Ritz eigenvalues to exact ones in the window.

    Returns (max_error_meV, mean_error_meV, n_matched).
    """
    # Exact eigenvalues in or near the window (with 5% margin).
    margin = 0.05 * (window_b - window_a)
    mask = (exact_eV >= window_a - margin) & (exact_eV <= window_b + margin)
    exact_in = exact_eV[mask]

    if len(ritz_eV) == 0 or len(exact_in) == 0:
        return 0.0, 0.0, 0

    # Match each Ritz value to closest exact.
    n_match = min(len(ritz_eV), len(exact_in))
    errors = []
    for i in range(n_match):
        err = abs(ritz_eV[i] - exact_in[i]) * 1000  # meV
        errors.append(err)

    return max(errors), float(np.mean(errors)), n_match


def run_sweep(
    data: dict,
    mesh_xy: Mesh,
    exact_eV: np.ndarray,
    ry_to_ev: float,
    configs: list[dict],
) -> list[SweepResult]:
    """Run FEAST for each configuration and collect results."""
    results = []

    for i, cfg in enumerate(configs):
        window = WindowSpec(
            f"sweep_{i}",
            cfg["window_a"],
            cfg["window_b"],
            f"sweep point {i+1}/{len(configs)}",
        )
        import sys
        print(f"\n{'='*60}")
        print(f"Sweep {i+1}/{len(configs)}: window=[{cfg['window_a']:.2f}, {cfg['window_b']:.2f}], "
              f"n_quad={cfg['n_quad']}, gamma={cfg['gamma']}, n_ritz={cfg['n_ritz']}, "
              f"feast_iter={cfg['feast_iter']}")
        print(f"{'='*60}")
        sys.stdout.flush()

        t0 = time.time()
        try:
            rr = run_feast_ritz(
                data,
                mesh_xy,
                [window],
                cfg["n_quad"],
                cfg["gamma"],
                cfg["n_ritz"],
                cfg.get("gmres_max_iter", 10),
                cfg.get("gmres_tol", 1e-2),
                seed=42,
                ry_to_ev=ry_to_ev,
                s_cutoff=0.01,
                feast_iter=cfg["feast_iter"],
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        wall = time.time() - t0

        rr_data = rr[window.name]
        ritz_eV = np.array([v * ry_to_ev for v in rr_data.ritz_evals])

        max_err, mean_err, n_matched = match_eigenvalues(
            ritz_eV, exact_eV, cfg["window_a"], cfg["window_b"],
        )

        result = SweepResult(
            window_a=cfg["window_a"],
            window_b=cfg["window_b"],
            n_quad=cfg["n_quad"],
            gamma=cfg["gamma"],
            n_ritz=cfg["n_ritz"],
            gmres_tol=cfg.get("gmres_tol", 1e-2),
            feast_iter=cfg["feast_iter"],
            ritz_evals_eV=[float(v) for v in ritz_eV],
            n_physical=rr_data.n_physical,
            n_total=rr_data.n_total,
            s_evals=[float(s) for s in rr_data.s_evals],
            wall_time_s=wall,
            max_error_meV=max_err,
            mean_error_meV=mean_err,
            n_matched=n_matched,
        )
        results.append(result)

        print(f"  Ritz (eV): {[f'{v:.4f}' for v in ritz_eV]}")
        print(f"  Max error: {max_err:.1f} meV, mean: {mean_err:.1f} meV, matched: {n_matched}")
        print(f"  Wall time: {wall:.2f} s")
        sys.stdout.flush()

    return results


def generate_configs() -> list[dict]:
    """Generate the sweep configurations from the accuracy notes."""
    configs = []

    # Primary sweep: contour quadrature × subspace × window × feast_iter
    n_quads = [4, 8, 16, 32]
    gammas = [0.2, 0.4, 0.8]
    n_ritz_list = [4, 8, 12]
    windows = [
        (0.0, 2.0),   # eigenvalues well inside
        (0.0, 1.91),  # eigenvalue near boundary
    ]
    feast_iters = [1, 2, 3]

    # Fix GMRES at tol=1e-2 (shown to be irrelevant in Obs 1).
    for (wa, wb), nq, gam, nr, fi in itertools.product(
        windows, n_quads, gammas, n_ritz_list, feast_iters,
    ):
        configs.append({
            "window_a": wa,
            "window_b": wb,
            "n_quad": nq,
            "gamma": gam,
            "n_ritz": nr,
            "gmres_tol": 1e-2,
            "gmres_max_iter": 10,
            "feast_iter": fi,
        })

    return configs


def generate_focused_configs(exact_eV: np.ndarray | None = None) -> list[dict]:
    """Smaller sweep focused on the most informative parameter combos.

    If exact_eV is provided, automatically picks windows with small numbers
    of eigenvalues. Otherwise uses default windows from the accuracy notes.
    """
    configs = []

    # Choose windows based on the actual spectrum.
    if exact_eV is not None:
        # Find a sparse region: pick a window with ~4-8 eigenvalues.
        # Strategy: look for the lowest few positive eigenvalues.
        pos_eV = exact_eV[exact_eV > 0]
        if len(pos_eV) >= 4:
            # Window 1: around first 4-6 positive eigenvalues.
            w1_a = 0.0
            w1_b = float(pos_eV[5]) + 0.05 if len(pos_eV) > 5 else float(pos_eV[-1]) + 0.05
            # Window 2: tight around first 2-3 (boundary study).
            w2_b = float(pos_eV[2]) + 0.01
        else:
            w1_a, w1_b = 0.0, 2.0
            w2_b = 1.91
        # Window 3: larger window with more states for n_ritz study.
        wider_eV = exact_eV[(exact_eV > 0) & (exact_eV < 2.0)]
        w3_b = min(2.0, float(wider_eV[-1]) + 0.05) if len(wider_eV) > 0 else 2.0
    else:
        w1_a, w1_b = 0.0, 2.0
        w2_b = 1.91
        w3_b = 2.0

    windows_main = [(w1_a, w1_b)]
    windows_boundary = [(w1_a, w2_b)]
    windows_wide = [(w1_a, w3_b)]

    # Sweep n_quad with fixed gamma, n_ritz (filter quality study).
    for nq in [4, 8, 16, 32]:
        for gam in [0.2, 0.4, 0.8]:
            for nr in [4, 8]:
                for fi in [1, 2]:
                    for wa, wb in windows_main:
                        configs.append({
                            "window_a": wa, "window_b": wb,
                            "n_quad": nq, "gamma": gam, "n_ritz": nr,
                            "gmres_tol": 1e-2, "gmres_max_iter": 10,
                            "feast_iter": fi,
                        })

    # Boundary window study.
    for nq in [4, 8, 16, 32]:
        for gam in [0.4]:
            for nr in [8]:
                for fi in [1, 2]:
                    for wa, wb in windows_boundary:
                        configs.append({
                            "window_a": wa, "window_b": wb,
                            "n_quad": nq, "gamma": gam, "n_ritz": nr,
                            "gmres_tol": 1e-2, "gmres_max_iter": 10,
                            "feast_iter": fi,
                        })

    # n_ritz study at fixed n_quad=16, gamma=0.4.
    for nr in [4, 8, 12]:
        for fi in [1, 2, 3]:
            for wa, wb in windows_wide:
                configs.append({
                    "window_a": wa, "window_b": wb,
                    "n_quad": 16, "gamma": 0.4, "n_ritz": nr,
                    "gmres_tol": 1e-2, "gmres_max_iter": 10,
                    "feast_iter": fi,
                })

    # GMRES tol verification (small sample).
    for tol in [1e-2, 1e-4]:
        for wa, wb in windows_main:
            configs.append({
                "window_a": wa, "window_b": wb,
                "n_quad": 8, "gamma": 0.4, "n_ritz": 8,
                "gmres_tol": tol, "gmres_max_iter": 10,
                "feast_iter": 1,
            })

    return configs


def print_report(results: list[SweepResult], exact_eV: np.ndarray) -> str:
    """Generate a comprehensive convergence report."""
    lines = []
    lines.append("=" * 80)
    lines.append("FEAST PARAMETER SWEEP CONVERGENCE REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Exact eigenvalues summary.
    lines.append("EXACT EIGENVALUES (eV), first 20 positive:")
    pos_eV = exact_eV[exact_eV > 0]
    for i, e in enumerate(pos_eV[:20]):
        lines.append(f"  {i+1:3d}: {e:12.6f}")
    lines.append("")

    # Identify windows used in sweep.
    window_keys = sorted(set((r.window_a, r.window_b) for r in results))
    for wa, wb in window_keys:
        n_in = int(np.sum((exact_eV >= wa) & (exact_eV <= wb)))
        lines.append(f"  Window [{wa:.2f}, {wb:.2f}] eV: {n_in} exact eigenvalues inside")
    lines.append("")

    # Identify the "main" and "boundary" windows.
    main_wb = max(r.window_b for r in results)
    boundary_wbs = sorted(set(r.window_b for r in results))

    # 1. Filter quality: n_quad convergence (main window, n_ritz=8, iter=1).
    lines.append("-" * 80)
    lines.append(f"1. FILTER QUALITY: n_quad convergence")
    lines.append(f"   (main window, n_ritz=8, feast_iter=1)")
    lines.append("-" * 80)
    lines.append(f"{'window_b':>8} {'n_quad':>6} {'gamma':>6} {'n_phys':>6} {'max_err(meV)':>12} {'mean_err(meV)':>13} {'time(s)':>8}")
    for r in sorted(results, key=lambda x: (x.window_b, x.gamma, x.n_quad)):
        if r.window_b == main_wb and r.n_ritz == 8 and r.feast_iter == 1:
            lines.append(f"{r.window_b:8.2f} {r.n_quad:6d} {r.gamma:6.1f} {r.n_physical:6d} {r.max_error_meV:12.1f} {r.mean_error_meV:13.1f} {r.wall_time_s:8.1f}")
    lines.append("")

    # 2. Gamma tradeoff.
    lines.append("-" * 80)
    lines.append("2. GAMMA TRADEOFF: ellipse aspect ratio")
    lines.append(f"   (main window, n_ritz=8, feast_iter=1)")
    lines.append("-" * 80)
    lines.append(f"{'gamma':>6} {'n_quad':>6} {'n_phys':>6} {'max_err(meV)':>12} {'mean_err(meV)':>13} {'time(s)':>8}")
    for r in sorted(results, key=lambda x: (x.gamma, x.n_quad)):
        if r.window_b == main_wb and r.n_ritz == 8 and r.feast_iter == 1:
            lines.append(f"{r.gamma:6.1f} {r.n_quad:6d} {r.n_physical:6d} {r.max_error_meV:12.1f} {r.mean_error_meV:13.1f} {r.wall_time_s:8.1f}")
    lines.append("")

    # 3. FEAST iteration convergence.
    lines.append("-" * 80)
    lines.append("3. FEAST ITERATION CONVERGENCE")
    lines.append(f"   (main window, n_quad=16, gamma=0.4)")
    lines.append("-" * 80)
    lines.append(f"{'n_ritz':>6} {'iter':>4} {'n_phys':>6} {'max_err(meV)':>12} {'mean_err(meV)':>13} {'time(s)':>8}")
    for r in sorted(results, key=lambda x: (x.n_ritz, x.feast_iter)):
        if r.window_b == main_wb and r.n_quad == 16 and abs(r.gamma - 0.4) < 0.01:
            lines.append(f"{r.n_ritz:6d} {r.feast_iter:4d} {r.n_physical:6d} {r.max_error_meV:12.1f} {r.mean_error_meV:13.1f} {r.wall_time_s:8.1f}")
    lines.append("")

    # 4. n_ritz study.
    lines.append("-" * 80)
    lines.append("4. SUBSPACE SIZE: n_ritz sufficiency")
    lines.append(f"   (wider window, n_quad=16, gamma=0.4, feast_iter=2)")
    lines.append("-" * 80)
    lines.append(f"{'n_ritz':>6} {'n_phys':>6} {'max_err(meV)':>12} {'mean_err(meV)':>13} {'Ritz evals (eV)':>40}")
    for r in sorted(results, key=lambda x: x.n_ritz):
        if r.n_quad == 16 and abs(r.gamma - 0.4) < 0.01 and r.feast_iter == 2:
            ev_str = ", ".join(f"{v:.4f}" for v in r.ritz_evals_eV[:8])
            if len(r.ritz_evals_eV) > 8:
                ev_str += ", ..."
            lines.append(f"{r.n_ritz:6d} {r.n_physical:6d} {r.max_error_meV:12.1f} {r.mean_error_meV:13.1f}   [{ev_str}]")
    lines.append("")

    # 5. Window boundary effects.
    lines.append("-" * 80)
    lines.append("5. WINDOW BOUNDARY EFFECTS")
    lines.append("   (n_ritz=8, gamma=0.4)")
    lines.append("-" * 80)
    lines.append(f"{'window':>16} {'n_quad':>6} {'iter':>4} {'n_phys':>6} {'max_err(meV)':>12} {'Ritz evals (eV)':>50}")
    for r in sorted(results, key=lambda x: (x.window_b, x.n_quad, x.feast_iter)):
        if r.n_ritz == 8 and abs(r.gamma - 0.4) < 0.01:
            ev_str = ", ".join(f"{v:.4f}" for v in r.ritz_evals_eV[:6])
            if len(r.ritz_evals_eV) > 6:
                ev_str += ", ..."
            wstr = f"[{r.window_a:.1f}, {r.window_b:.2f}]"
            lines.append(f"{wstr:>16} {r.n_quad:6d} {r.feast_iter:4d} {r.n_physical:6d} {r.max_error_meV:12.1f}   [{ev_str}]")
    lines.append("")

    # 6. GMRES tol verification.
    lines.append("-" * 80)
    lines.append("6. GMRES TOLERANCE CHECK")
    lines.append(f"   (main window, n_quad=8, gamma=0.4, n_ritz=8, feast_iter=1)")
    lines.append("-" * 80)
    lines.append(f"{'gmres_tol':>10} {'max_err(meV)':>12} {'Ritz evals (eV)':>50}")
    for r in sorted(results, key=lambda x: x.gmres_tol):
        if r.window_b == main_wb and r.n_quad == 8 and abs(r.gamma - 0.4) < 0.01 and r.n_ritz == 8 and r.feast_iter == 1:
            ev_str = ", ".join(f"{v:.6f}" for v in r.ritz_evals_eV)
            lines.append(f"{r.gmres_tol:10.0e} {r.max_error_meV:12.1f}   [{ev_str}]")
    lines.append("")

    # 7. Full results table sorted by accuracy.
    lines.append("-" * 80)
    lines.append("7. ALL RESULTS SORTED BY ACCURACY")
    lines.append("-" * 80)
    lines.append(f"{'window':>16} {'nq':>3} {'gam':>4} {'nr':>3} {'fi':>2} {'n_phys':>6} {'max_err':>8} {'mean_err':>8} {'time':>6}")
    for r in sorted(results, key=lambda x: x.max_error_meV):
        wstr = f"[{r.window_a:.1f},{r.window_b:.2f}]"
        lines.append(f"{wstr:>16} {r.n_quad:3d} {r.gamma:4.1f} {r.n_ritz:3d} {r.feast_iter:2d} {r.n_physical:6d} {r.max_error_meV:8.1f} {r.mean_error_meV:8.1f} {r.wall_time_s:6.1f}")
    lines.append("")

    # Summary recommendations.
    lines.append("=" * 80)
    lines.append("SUMMARY & RECOMMENDATIONS")
    lines.append("=" * 80)

    # Find cheapest config with <1 meV error.
    sub_mev = [r for r in results if 0 < r.max_error_meV < 1.0 and r.n_matched > 0]
    if sub_mev:
        cheapest = min(sub_mev, key=lambda r: r.wall_time_s)
        lines.append(f"\nCheapest config with <1 meV error:")
        lines.append(f"  window=[{cheapest.window_a}, {cheapest.window_b}], n_quad={cheapest.n_quad}, "
                     f"gamma={cheapest.gamma}, n_ritz={cheapest.n_ritz}, feast_iter={cheapest.feast_iter}")
        lines.append(f"  max_error={cheapest.max_error_meV:.2f} meV, time={cheapest.wall_time_s:.1f} s")
    else:
        lines.append("\nNo config achieved <1 meV accuracy.")

    sub_10 = [r for r in results if 0 < r.max_error_meV < 10.0 and r.n_matched > 0]
    if sub_10:
        cheapest = min(sub_10, key=lambda r: r.wall_time_s)
        lines.append(f"\nCheapest config with <10 meV error:")
        lines.append(f"  window=[{cheapest.window_a}, {cheapest.window_b}], n_quad={cheapest.n_quad}, "
                     f"gamma={cheapest.gamma}, n_ritz={cheapest.n_ritz}, feast_iter={cheapest.feast_iter}")
        lines.append(f"  max_error={cheapest.max_error_meV:.2f} meV, time={cheapest.wall_time_s:.1f} s")

    sub_50 = [r for r in results if 0 < r.max_error_meV < 50.0 and r.n_matched > 0]
    if sub_50:
        cheapest = min(sub_50, key=lambda r: r.wall_time_s)
        lines.append(f"\nCheapest config with <50 meV error:")
        lines.append(f"  window=[{cheapest.window_a}, {cheapest.window_b}], n_quad={cheapest.n_quad}, "
                     f"gamma={cheapest.gamma}, n_ritz={cheapest.n_ritz}, feast_iter={cheapest.feast_iter}")
        lines.append(f"  max_error={cheapest.max_error_meV:.2f} meV, time={cheapest.wall_time_s:.1f} s")

    lines.append("")
    report = "\n".join(lines)
    return report


def generate_minimal_configs() -> list[dict]:
    """Minimal sweep matching the accuracy notes exactly.

    Uses windows [0, 2.0] and [0, 1.91] as specified.
    Focuses on the most informative parameter combos.
    """
    configs = []

    # 1. n_quad convergence at fixed gamma=0.4, n_ritz=8 (the key sweep).
    for nq in [4, 8, 16, 32]:
        for fi in [1, 2, 3]:
            configs.append({
                "window_a": 0.0, "window_b": 2.0,
                "n_quad": nq, "gamma": 0.4, "n_ritz": 8,
                "gmres_tol": 1e-2, "gmres_max_iter": 10,
                "feast_iter": fi,
            })

    # 2. Gamma sweep at fixed n_quad=8, n_ritz=8.
    for gam in [0.2, 0.4, 0.8]:
        for nq in [4, 8, 16]:
            configs.append({
                "window_a": 0.0, "window_b": 2.0,
                "n_quad": nq, "gamma": gam, "n_ritz": 8,
                "gmres_tol": 1e-2, "gmres_max_iter": 10,
                "feast_iter": 1,
            })

    # 3. n_ritz study at fixed n_quad=8, gamma=0.4.
    for nr in [4, 8, 12]:
        for fi in [1, 2]:
            configs.append({
                "window_a": 0.0, "window_b": 2.0,
                "n_quad": 8, "gamma": 0.4, "n_ritz": nr,
                "gmres_tol": 1e-2, "gmres_max_iter": 10,
                "feast_iter": fi,
            })

    # 4. Window boundary: [0, 1.91] (eigenvalue at boundary).
    for nq in [4, 8, 16, 32]:
        for fi in [1, 2]:
            configs.append({
                "window_a": 0.0, "window_b": 1.91,
                "n_quad": nq, "gamma": 0.4, "n_ritz": 8,
                "gmres_tol": 1e-2, "gmres_max_iter": 10,
                "feast_iter": fi,
            })

    # 5. GMRES tol check.
    for tol in [1e-2, 1e-4]:
        configs.append({
            "window_a": 0.0, "window_b": 2.0,
            "n_quad": 8, "gamma": 0.4, "n_ritz": 8,
            "gmres_tol": tol, "gmres_max_iter": 10,
            "feast_iter": 1,
        })

    return configs


# Known exact eigenvalues (eV) from full diag with isdf_tensors_600 data,
# n_val=4, n_cond=4, 3×3×1 k-grid.
EXACT_EIGENVALUES_EV = np.array([
    1.853621, 1.853655, 1.917182, 1.917187, 1.963235, 1.963259,
    2.067847, 2.067879, 2.296806, 2.296820, 2.310239, 2.310433,
])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FEAST parameter sweep")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--full", action="store_true", help="Run full sweep (216 configs)")
    parser.add_argument("--focused", action="store_true", help="Run focused sweep (~66 configs)")
    parser.add_argument("--output", default="feast_sweep_results.json", help="JSON output file")
    parser.add_argument("--units-ev-per-ry", type=float, default=RY_TO_EV_DEFAULT)
    parser.add_argument("--full-diag", action="store_true",
                        help="Compute exact eigenvalues via full diag (slow for large n_rmu)")
    args = parser.parse_args(argv)

    mesh_xy = _create_mesh_xy(1, 1)
    restart_file = _find_restart_file(args.input)

    print("Loading BSE data...", flush=True)
    data = load_bse_data_from_restart_sharded(
        restart_file,
        n_val=args.n_val,
        n_cond=args.n_cond,
        mesh_xy=mesh_xy,
        input_file=args.input,
    )

    if args.full_diag:
        print("\nComputing exact eigenvalues (full diagonalization)...", flush=True)
        t0 = time.time()
        exact_eV, _ = build_full_bse_matrix(data, ry_to_ev=args.units_ev_per_ry)
        print(f"Full diag time: {time.time() - t0:.1f} s\n", flush=True)
    else:
        exact_eV = EXACT_EIGENVALUES_EV
        print(f"Using hardcoded exact eigenvalues (first 8): "
              f"{np.array2string(exact_eV[:8], precision=4)}", flush=True)

    # Generate sweep configs.
    if args.full:
        configs = generate_configs()
    elif args.focused:
        configs = generate_focused_configs(exact_eV)
    else:
        configs = generate_minimal_configs()

    # Deduplicate.
    seen = set()
    unique_configs = []
    for cfg in configs:
        key = (cfg["window_a"], cfg["window_b"], cfg["n_quad"], cfg["gamma"],
               cfg["n_ritz"], cfg["gmres_tol"], cfg["feast_iter"])
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)
    configs = unique_configs

    print(f"Running {len(configs)} FEAST configurations...")

    # Pre-compute W_R once (run_feast_ritz does it internally, but we need it in data).
    # Not needed: run_feast_ritz handles this.

    results = run_sweep(data, mesh_xy, exact_eV, args.units_ev_per_ry, configs)

    # Print report.
    report = print_report(results, exact_eV)
    print(report)

    # Save raw results to JSON.
    json_data = {
        "exact_eigenvalues_eV": exact_eV.tolist(),
        "results": [asdict(r) for r in results],
    }
    with open(args.output, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()

"""KPM (Kernel Polynomial Method) for BSE density of states.

Computes Chebyshev moments of the BSE Hamiltonian using stochastic
trace estimation, applies Jackson damping, and reconstructs the DOS.

The generic Chebyshev algorithms live in solvers.chebyshev.  This module
builds BSE-specific matvecs and random vector generators, then delegates
to the solver.

Usage:
    python -m bse.bse_kpm -i cohsex.inp --n-val 4 --n-cond 4 --n-moments 200
"""
from __future__ import annotations

import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from solvers.chebyshev import (
    jackson_coefficients,
    chebyshev_moments,
    reconstruct_dos,
    partition_windows,
)
from .bse_ring_comm import build_bse_ring_matvec, build_bse_ring_matvec_full, make_bse_shardings
from .bse_feast import estimate_spectral_bounds_sharded, _create_mesh_xy, _build_gmres_data_fp32
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
import common.timing as timing

jax.config.update("jax_enable_x64", True)

RY_TO_EV = 13.6056980659


def make_bse_h_tilde(matvec, data, e_center, half_width):
    """Return a @jax.jit'd rescaled BSE matvec: H_tilde x = (Hx - center*x) / half_width.

    Binds BSE arrays from *data* into the closure so the returned callable
    has signature (x,) -> H_tilde x with no dict access at call time.
    """
    psi_c_X = data["psi_c_X"]
    psi_c_Y = data["psi_c_Y"]
    psi_v_X = data["psi_v_X"]
    psi_v_Y = data["psi_v_Y"]
    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
    W_R = data["W_R"]
    V_q0 = data["V_q0"]
    M_X = data["M_X"]  # hoisted V-term pair-amps (audit P3)
    M_Y = data["M_Y"]

    dtype_real = eps_c.dtype
    e_center_jnp = jnp.asarray(e_center, dtype=dtype_real)
    inv_hw = jnp.asarray(1.0 / half_width, dtype=dtype_real)

    @jax.jit
    def apply_h_tilde(x):
        hx = matvec(x, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X, M_Y)
        return (hx - e_center_jnp * x) * inv_hw

    return apply_h_tilde


def make_bse_random_vector(data, use_tda):
    """Return a callable (key,) -> x_random that generates masked Rademacher vectors.

    Resolves TDA/non-TDA branching at factory time so the returned function
    has no Python-level conditionals (clean for tracing).
    """
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_cond = int(data["n_cond"])
    n_val = int(data["n_val"])
    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])

    dtype_real = data["eps_c"].dtype
    dtype_cplx = jnp.complex64 if dtype_real == jnp.float32 else jnp.complex128

    mask = jnp.zeros((1, n_cond_pad, n_val_pad, nk), dtype=dtype_real)
    mask = mask.at[:, :n_cond, :n_val, :].set(1.0)

    shape_1 = (1, n_cond_pad, n_val_pad, nk)

    if use_tda:
        def fn(key):
            x = 2.0 * jax.random.bernoulli(key, shape=shape_1).astype(dtype_real) - 1.0
            return (x * mask).astype(dtype_cplx)
    else:
        mask_full = jnp.stack([mask, mask], axis=0)

        def fn(key):
            k_x, k_y = jax.random.split(key)
            x0 = 2.0 * jax.random.bernoulli(k_x, shape=shape_1).astype(dtype_real) - 1.0
            x1 = 2.0 * jax.random.bernoulli(k_y, shape=shape_1).astype(dtype_real) - 1.0
            x = jnp.stack([x0, x1], axis=0)
            return (x * mask_full).astype(dtype_cplx)

    return fn


def run_kpm_dos(
    data: dict,
    mesh_xy: Mesh,
    n_moments: int = 200,
    n_random: int = 5,
    seed: int = 0,
    n_lanczos: int = 100,
    buffer: float = 0.05,
    ry_to_ev: float = RY_TO_EV,
    n_energy_pts: int = 2000,
    plot_file: str = "bse_dos_kpm.png",
    emin_ev: float | None = None,
    emax_ev: float | None = None,
    n_windows: int = 10,
    emit_outputs: bool = True,
    include_W: bool = True,
    use_tda: bool = True,
) -> dict:
    """Run KPM DOS calculation: bounds, moments, reconstruction, plot."""
    if use_tda:
        matvec = build_bse_ring_matvec(
            mesh_xy,
            data["nkx"],
            data["nky"],
            data["nkz"],
            include_W=include_W,
        )
    else:
        matvec = build_bse_ring_matvec_full(
            mesh_xy,
            data["nkx"],
            data["nky"],
            data["nkz"],
            include_W=include_W,
        )

    data_fp32 = _build_gmres_data_fp32(data)
    data_fp32.update(
        {
            "nkx": data["nkx"],
            "nky": data["nky"],
            "nkz": data["nkz"],
            "n_val": data["n_val"],
            "n_cond": data["n_cond"],
            "n_val_pad": data["n_val_pad"],
            "n_cond_pad": data["n_cond_pad"],
        }
    )
    if include_W:
        data_fp32["W_R"] = jnp.fft.ifftn(data_fp32["W_q"], axes=(2, 3, 4), norm="ortho")
    else:
        data_fp32["W_R"] = data_fp32["W_q"]

    # --- Spectral bounds from Lanczos ---
    print(f"Estimating spectral bounds (Lanczos min={n_lanczos})...")
    bounds = estimate_spectral_bounds_sharded(
        data,
        mesh_xy,
        n_lanczos=n_lanczos,
        n_lanczos_max=max(n_lanczos, 50),
        include_W=include_W,
        use_tda=use_tda,
    )
    e_min_ry = bounds["e_min_ry"]
    e_max_ry = bounds["e_max_ry_raw"]

    print(f"  E_min (Lanczos)  = {e_min_ry:.6f} Ry = {e_min_ry * ry_to_ev:.3f} eV")
    print(f"  E_max (Lanczos)  = {e_max_ry:.6f} Ry = {e_max_ry * ry_to_ev:.3f} eV")

    # Apply manual overrides (in eV -> convert to Ry).
    if emin_ev is not None:
        e_min_ry = emin_ev / ry_to_ev
        print(f"  E_min (override) = {e_min_ry:.6f} Ry = {emin_ev:.3f} eV")
    if emax_ev is not None:
        e_max_ry = emax_ev / ry_to_ev
        print(f"  E_max (override) = {e_max_ry:.6f} Ry = {emax_ev:.3f} eV")

    bandwidth = e_max_ry - e_min_ry
    if use_tda:
        e_min_buf = e_min_ry - buffer * bandwidth
        e_max_buf = e_max_ry + buffer * bandwidth
        # BSE-TDA eigenvalues are positive; don't go below zero.
        e_min_buf = max(0.0, e_min_buf)
        e_center = 0.5 * (e_max_buf + e_min_buf)
        half_width = 0.5 * (e_max_buf - e_min_buf)
    else:
        max_abs = max(abs(e_min_ry), abs(e_max_ry))
        e_max_buf = max_abs * (1.0 + buffer)
        e_min_buf = -e_max_buf
        e_center = 0.0
        half_width = e_max_buf

    print(f"  E_min (buffered) = {e_min_buf:.6f} Ry = {e_min_buf * ry_to_ev:.3f} eV")
    print(f"  E_max (buffered) = {e_max_buf:.6f} Ry = {e_max_buf * ry_to_ev:.3f} eV")
    print(f"  Center = {e_center:.6f} Ry = {e_center * ry_to_ev:.3f} eV")
    print(f"  Half-width = {half_width:.6f} Ry = {half_width * ry_to_ev:.3f} eV")

    # --- Build BSE-specific callables ---
    apply_h_tilde = make_bse_h_tilde(matvec, data_fp32, e_center, half_width)
    random_vector_fn = make_bse_random_vector(data_fp32, use_tda)

    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_cond = int(data["n_cond"])
    n_val = int(data["n_val"])
    bse_dim = n_cond * n_val * nk
    if not use_tda:
        bse_dim *= 2

    # --- Chebyshev moments (generic solver) ---
    print(f"\nComputing {n_moments} Chebyshev moments with {n_random} random vectors...")
    print(f"  Total matvecs: {n_random * n_moments}")
    with timing.section("kpm.moments"):
        mu = chebyshev_moments(
            apply_h_tilde,
            bse_dim,
            n_moments,
            n_random,
            seed=seed,
            dtype_real=data_fp32["eps_c"].dtype,
            make_random_vector=random_vector_fn,
        )

    print(f"\n  mu_0 = {mu[0]:.6f}  (should be ~1.0)")
    print(f"  First 10 raw moments: {np.array2string(mu[:10], precision=6)}")

    # --- Jackson damping ---
    sigma = jackson_coefficients(n_moments)
    mu_damped = mu * sigma

    # --- Reconstruct DOS ---
    e_center_eV = e_center * ry_to_ev
    half_width_eV = half_width * ry_to_ev
    if use_tda:
        E_min_plot = max(0, e_min_buf * ry_to_ev - 0.5)
        E_max_plot = e_max_buf * ry_to_ev + 0.5
    else:
        E_min_plot = e_min_buf * ry_to_ev - 0.5
        E_max_plot = e_max_buf * ry_to_ev + 0.5
    E_grid = np.linspace(E_min_plot, E_max_plot, n_energy_pts)

    rho = reconstruct_dos(mu_damped, E_grid, e_center_eV, half_width_eV)
    rho = np.maximum(rho, 0.0)  # clip ringing artefacts near edges

    if use_tda:
        omega_min_eV = E_grid[1] if E_grid[0] <= 0.0 else E_grid[0]
        omega_min_eV = max(float(omega_min_eV), float(e_min_ry * ry_to_ev))
    else:
        omega_min_eV = 0.01

    windows_ry = partition_windows(
        E_grid, rho, n_windows=n_windows,
        energy_min=omega_min_eV,
    ) / ry_to_ev
    window_edges_eV = windows_ry.ravel() * ry_to_ev

    if emit_outputs:
        # --- Plot ---
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.plot(E_grid, rho, "b-", linewidth=0.8)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("DOS (states/eV)")
        ax.set_title(f"BSE Density of States  (KPM, M={n_moments}, R={n_random})")
        ax.set_xlim(E_min_plot, E_max_plot)
        ax.set_ylim(bottom=0)
        ax.axhline(y=0, color="k", linewidth=0.3)
        for edge in window_edges_eV:
            ax.axvline(edge, color="tab:orange", linewidth=0.7, alpha=0.6)
        ax.grid(True, alpha=0.3)

        # Moment decay subplot.
        ax2 = axes[1]
        ax2.semilogy(np.arange(n_moments + 1), np.abs(mu), "k-", linewidth=0.5, label="raw |mu_p|")
        ax2.semilogy(np.arange(n_moments + 1), np.abs(mu_damped), "r-", linewidth=0.5, label="Jackson |mu_p|")
        ax2.set_xlabel("Chebyshev order p")
        ax2.set_ylabel("|mu_p|")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_file, dpi=150)
        print(f"\nDOS plot saved to {plot_file}")
        plt.close(fig)

        # --- Save data ---
        npz_file = plot_file.rsplit(".", 1)[0] + ".npz"
        np.savez(
            npz_file,
            mu_raw=mu,
            mu_damped=mu_damped,
            jackson_sigma=sigma,
            E_grid_eV=E_grid,
            rho=rho,
            e_center_ry=e_center,
            half_width_ry=half_width,
            e_min_ry=e_min_ry,
            e_max_ry=e_max_ry,
            n_moments=n_moments,
            n_random=n_random,
        )
        print(f"Moments + DOS data saved to {npz_file}")

    return {
        "mu_raw": mu,
        "mu_damped": mu_damped,
        "E_grid_eV": E_grid,
        "rho": rho,
        "e_center_ry": e_center,
        "half_width_ry": half_width,
        "windows_ry": windows_ry,
        "window_edges_eV": window_edges_eV,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="KPM density of states for BSE")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--n-moments", type=int, default=200,
                        help="Number of Chebyshev moments M (cost: R*M matvecs)")
    parser.add_argument("--n-random", type=int, default=4,
                        help="Stochastic trace vectors R (5-10 is usually enough)")
    parser.add_argument("--n-lanczos", type=int, default=100,
                        help="Lanczos steps for spectral bounds (default 100; "
                             "needs enough iterations to converge E_max)")
    parser.add_argument("--buffer", type=float, default=0.05,
                        help="Fractional buffer on spectral bounds (default 5%%)")
    parser.add_argument("--emin-ev", type=float, default=None,
                        help="Override E_min in eV (skip Lanczos lower bound)")
    parser.add_argument("--emax-ev", type=float, default=None,
                        help="Override E_max in eV (skip Lanczos upper bound)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-energy-pts", type=int, default=2000)
    parser.add_argument("--n-windows", type=int, default=10,
                        help="Number of DOS-weighted windows to compute.")
    parser.add_argument("--plot-file", type=str, default="bse_dos_kpm.png")
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV)
    parser.add_argument("--rpa", action="store_true",
                        help="Use RPA kernel (D+V only), skip W0 term entirely.")
    parser.add_argument("--tda", action="store_true",
                        help="Use Tamm-Dancoff approximation (TDA). Default is full non-TDA.")
    parser.add_argument("--nohead", action="store_true",
                        help="Use headless V/W0 arrays if present (V_qmunu_nohead, W0_qmunu_nohead).")
    args = parser.parse_args(argv)

    timing.reset()

    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)

    with timing.section("kpm.load"):
        data = load_bse_data_from_restart_sharded(
            restart_file,
            n_val=args.n_val,
            n_cond=args.n_cond,
            mesh_xy=mesh_xy,
            use_nohead=args.nohead,
            input_file=args.input,
        )

    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_c = int(data["n_cond"])
    n_v = int(data["n_val"])
    bse_dim = n_c * n_v * nk
    if not args.tda:
        bse_dim *= 2
    print(f"BSE dimension: {n_c} cond x {n_v} val x {nk} k = {bse_dim}")
    print(f"KPM parameters: M={args.n_moments} moments, R={args.n_random} random vectors")
    print(f"Total matvecs: {args.n_random * args.n_moments}")

    with timing.section("kpm.total"):
        run_kpm_dos(
            data,
            mesh_xy,
            n_moments=args.n_moments,
            n_random=args.n_random,
            seed=args.seed,
            n_lanczos=args.n_lanczos,
            buffer=args.buffer,
            ry_to_ev=args.ry_to_ev,
            n_energy_pts=args.n_energy_pts,
            plot_file=args.plot_file,
            emin_ev=args.emin_ev,
            emax_ev=args.emax_ev,
            n_windows=args.n_windows,
            include_W=not args.rpa,
            use_tda=args.tda,
        )

    timing.report(print_fn=print, title="--- KPM Timing ---")


if __name__ == "__main__":
    main()

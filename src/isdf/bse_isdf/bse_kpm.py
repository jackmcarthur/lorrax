"""KPM (Kernel Polynomial Method) for BSE density of states.

Computes Chebyshev moments of the BSE Hamiltonian using stochastic
trace estimation, applies Jackson damping, and reconstructs the DOS.

Usage:
    python -m isdf.bse_isdf.bse_kpm -i cohsex.inp --n-val 4 --n-cond 4 --n-moments 200
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .bse_ring_comm import build_bse_ring_matvec, make_bse_shardings
from .bse_feast import estimate_spectral_bounds_sharded, _create_mesh_xy
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
import isdf.common.timing as timing

jax.config.update("jax_enable_x64", True)

RY_TO_EV = 13.6056980659


def jackson_coefficients(M: int) -> np.ndarray:
    """Jackson damping coefficients sigma_p^{(J)} for p = 0, ..., M."""
    p = np.arange(M + 1)
    sigma = (
        (M - p + 1) * np.cos(np.pi * p / (M + 1))
        + np.sin(np.pi * p / (M + 1)) / np.tan(np.pi / (M + 1))
    ) / (M + 1)
    return sigma


def chebyshev_moments(
    matvec,
    data: dict,
    e_center: float,
    half_width: float,
    n_moments: int,
    n_random: int,
    seed: int = 0,
) -> np.ndarray:
    """Compute Chebyshev moments mu_0, ..., mu_M via stochastic trace.

    Parameters
    ----------
    matvec : callable
        Sharded BSE ring matvec.
    data : dict
        BSE data dict with W_R already computed.
    e_center, half_width : float
        Rescaling parameters (Rydbergs): H_tilde = (H - e_center) / half_width.
    n_moments : int
        Number of Chebyshev moments M (computes mu_0 through mu_M).
    n_random : int
        Number of stochastic random vectors R.
    seed : int
        Random seed.

    Returns
    -------
    mu : ndarray of shape (n_moments + 1,)
        Raw (undamped) Chebyshev moments.
    """
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_cond = int(data["n_cond"])
    n_val = int(data["n_val"])
    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])
    bse_dim = n_cond * n_val * nk

    psi_c_X = data["psi_c_X"]
    psi_c_Y = data["psi_c_Y"]
    psi_v_X = data["psi_v_X"]
    psi_v_Y = data["psi_v_Y"]
    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
    W_R = data["W_R"]
    V_q0 = data["V_q0"]

    e_center_jnp = jnp.float64(e_center)
    inv_hw = jnp.float64(1.0 / half_width)

    @jax.jit
    def apply_h_tilde(x):
        hx = matvec(
            x, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
            eps_c, eps_v, W_R, V_q0,
        )
        return (hx - e_center_jnp * x) * inv_hw

    key = jax.random.PRNGKey(seed)
    mu = np.zeros(n_moments + 1)

    # Mask: only put random entries in physical (non-padded) bands.
    # Padded bands have psi=0 and eps=0, so H maps them to
    # -e_center/hw * x which is OUTSIDE [-1,1] for BSE spectra.
    # Masking keeps the recurrence stable.
    mask = jnp.zeros((1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
    mask = mask.at[:, :n_cond, :n_val, :].set(1.0)

    for r in range(n_random):
        print(f"  Random vector {r + 1}/{n_random}...")
        key, subkey = jax.random.split(key)

        # Rademacher +/-1 random vector (real).
        x_rand = (
            2.0 * jax.random.bernoulli(
                subkey, shape=(1, n_cond_pad, n_val_pad, nk),
            ).astype(jnp.float64) - 1.0
        )
        x_rand = (x_rand * mask).astype(jnp.complex128)

        t_prev = x_rand                     # t_0 = X
        t_curr = apply_h_tilde(x_rand)      # t_1 = H_tilde X

        mu[0] += float(jax.device_get(jnp.vdot(x_rand, t_prev).real))
        mu[1] += float(jax.device_get(jnp.vdot(x_rand, t_curr).real))

        for p in range(2, n_moments + 1):
            t_new = 2.0 * apply_h_tilde(t_curr) - t_prev
            mu[p] += float(jax.device_get(jnp.vdot(x_rand, t_new).real))
            t_prev = t_curr
            t_curr = t_new

            if p % 50 == 0:
                print(f"    moment {p}/{n_moments}")

    mu /= n_random * bse_dim
    return mu


def reconstruct_dos(
    mu: np.ndarray,
    E_grid_eV: np.ndarray,
    e_center_eV: float,
    half_width_eV: float,
) -> np.ndarray:
    """Reconstruct DOS on a physical energy grid.

    Uses the correct normalization:
        rho(E) = [mu_0 + 2 sum_{p=1}^M mu_p T_p(e)] / (pi * hw * sqrt(1 - e^2))
    where e = (E - e_center) / hw.  Integrates to 1 over the spectrum.

    Parameters
    ----------
    mu : ndarray
        (Jackson-damped) Chebyshev moments, length M+1.
    E_grid_eV : ndarray
        Energy grid in eV.
    e_center_eV, half_width_eV : float
        Rescaling center and half-width in eV.

    Returns
    -------
    rho : ndarray
        DOS in units of states/eV.
    """
    E_tilde = (E_grid_eV - e_center_eV) / half_width_eV
    E_tilde = np.clip(E_tilde, -1 + 1e-10, 1 - 1e-10)

    M = len(mu) - 1

    # Chebyshev sum via recurrence on scalars.
    T_prev = np.ones_like(E_tilde)   # T_0(e) = 1
    T_curr = E_tilde.copy()          # T_1(e) = e

    cheb_sum = mu[0] * T_prev + 2.0 * mu[1] * T_curr

    for p in range(2, M + 1):
        T_new = 2.0 * E_tilde * T_curr - T_prev
        cheb_sum += 2.0 * mu[p] * T_new
        T_prev = T_curr
        T_curr = T_new

    rho = cheb_sum / (np.pi * half_width_eV * np.sqrt(1 - E_tilde**2))
    return rho


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
) -> dict:
    """Run KPM DOS calculation: bounds, moments, reconstruction, plot."""
    matvec = build_bse_ring_matvec(mesh_xy, data["nkx"], data["nky"], data["nkz"])

    # W_q -> W_R for ring matvec.
    data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")

    # --- Spectral bounds from Lanczos ---
    print(f"Estimating spectral bounds ({n_lanczos} Lanczos steps)...")
    bounds = estimate_spectral_bounds_sharded(data, mesh_xy, n_lanczos=n_lanczos)
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
    e_min_buf = e_min_ry - buffer * bandwidth
    e_max_buf = e_max_ry + buffer * bandwidth
    # BSE-TDA eigenvalues are positive; don't go below zero.
    e_min_buf = max(0.0, e_min_buf)

    e_center = 0.5 * (e_max_buf + e_min_buf)
    half_width = 0.5 * (e_max_buf - e_min_buf)

    print(f"  E_min (buffered) = {e_min_buf:.6f} Ry = {e_min_buf * ry_to_ev:.3f} eV")
    print(f"  E_max (buffered) = {e_max_buf:.6f} Ry = {e_max_buf * ry_to_ev:.3f} eV")
    print(f"  Center = {e_center:.6f} Ry = {e_center * ry_to_ev:.3f} eV")
    print(f"  Half-width = {half_width:.6f} Ry = {half_width * ry_to_ev:.3f} eV")

    # --- Chebyshev moments ---
    print(f"\nComputing {n_moments} Chebyshev moments with {n_random} random vectors...")
    print(f"  Total matvecs: {n_random * n_moments}")
    with timing.section("kpm.moments"):
        mu = chebyshev_moments(
            matvec, data, e_center, half_width,
            n_moments, n_random, seed,
        )

    print(f"\n  mu_0 = {mu[0]:.6f}  (should be ~1.0)")
    print(f"  First 10 raw moments: {np.array2string(mu[:10], precision=6)}")

    # --- Jackson damping ---
    sigma = jackson_coefficients(n_moments)
    mu_damped = mu * sigma

    # --- Reconstruct DOS ---
    e_center_eV = e_center * ry_to_ev
    half_width_eV = half_width * ry_to_ev
    E_min_plot = max(0, e_min_buf * ry_to_ev - 0.5)
    E_max_plot = e_max_buf * ry_to_ev + 0.5
    E_grid = np.linspace(E_min_plot, E_max_plot, n_energy_pts)

    rho = reconstruct_dos(mu_damped, E_grid, e_center_eV, half_width_eV)
    rho = np.maximum(rho, 0.0)  # clip ringing artefacts near edges

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
    parser.add_argument("--n-random", type=int, default=5,
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
    parser.add_argument("--plot-file", type=str, default="bse_dos_kpm.png")
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV)
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
        )

    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_c = int(data["n_cond"])
    n_v = int(data["n_val"])
    bse_dim = n_c * n_v * nk
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
        )

    timing.report(print_fn=print, title="--- KPM Timing ---")


if __name__ == "__main__":
    main()

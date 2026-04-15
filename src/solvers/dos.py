"""
solvers/dos.py — Matrix-free density of states via KPM.

End-to-end DOS from a Hermitian matvec: estimate spectrum bounds,
compute Chebyshev moments, reconstruct the smoothed DOS on an energy grid.

No physics knowledge — works for any Hermitian operator.

Usage
-----
    from solvers.dos import compute_dos, estimate_spectrum

    E_min, E_max = estimate_spectrum(apply_H, dim)
    dos_result = compute_dos(apply_H, dim, n_moments=2000, n_random=20)

    # dos_result is a DOSResult with:
    #   .E_grid    — energy grid
    #   .rho       — DOS (states per unit energy)
    #   .E_min, .E_max — spectrum bounds
    #   .center, .half_width — rescaling parameters
    #   .mu_raw    — raw Chebyshev moments (before damping)
    #   .mu_damped — Jackson-damped moments
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from solvers.lanczos import simple_lanczos_eig
from solvers.chebyshev import (
    jackson_coefficients,
    chebyshev_moments,
    reconstruct_dos,
)


@dataclass
class DOSResult:
    """Output of compute_dos — everything needed for downstream windowing."""
    E_grid: np.ndarray       # (n_grid,) energy grid
    rho: np.ndarray          # (n_grid,) DOS in states per unit energy
    E_min: float             # estimated spectrum lower bound
    E_max: float             # estimated spectrum upper bound
    center: float            # (E_max + E_min) / 2
    half_width: float        # (E_max - E_min) / 2
    mu_raw: np.ndarray       # (n_moments+1,) undamped Chebyshev moments
    mu_damped: np.ndarray    # (n_moments+1,) Jackson-damped moments
    n_moments: int
    n_random: int


def estimate_spectrum(
    apply_H: Callable,
    dim: int,
    n_lanczos: int = 30,
    pad_fraction: float = 0.02,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate (E_min, E_max) of a Hermitian operator via Lanczos.

    Runs a short Lanczos iteration to find extremal eigenvalues,
    then pads by pad_fraction to ensure strict spectral containment.

    Parameters
    ----------
    apply_H : (dim,) → (dim,)
        Hermitian matvec (flat vectors, not batched).
    dim : int
        Vector dimension.
    n_lanczos : int
        Number of Lanczos iterations (30 is usually plenty).
    pad_fraction : float
        Fractional padding beyond extremal eigenvalues.

    Returns
    -------
    E_min, E_max : float
        Padded spectrum bounds.
    """
    # Get both lowest and highest eigenvalues
    evals_low, _ = simple_lanczos_eig(apply_H, dim, n_eig=1,
                                       max_iter=n_lanczos, seed=seed)
    # For E_max: solve for -H and negate
    def neg_H(x):
        return -apply_H(x)
    evals_high_neg, _ = simple_lanczos_eig(neg_H, dim, n_eig=1,
                                            max_iter=n_lanczos, seed=seed + 1)

    E_lo = float(np.min(np.asarray(evals_low)))
    E_hi = -float(np.min(np.asarray(evals_high_neg)))

    span = E_hi - E_lo
    return E_lo - pad_fraction * span, E_hi + pad_fraction * span


def compute_dos(
    apply_H: Callable,
    dim: int,
    *,
    n_moments: int = 2000,
    n_random: int = 20,
    n_grid: int = 10000,
    E_min: float | None = None,
    E_max: float | None = None,
    n_lanczos: int = 30,
    seed: int = 0,
    verbose: bool = True,
) -> DOSResult:
    """Compute the density of states via the Kernel Polynomial Method.

    End-to-end: estimates spectrum bounds (if not given), runs the
    Chebyshev recurrence with stochastic trace estimation, applies
    Jackson damping, and reconstructs the DOS on a dense energy grid.

    Parameters
    ----------
    apply_H : (dim,) → (dim,)
        Hermitian matvec (flat vectors, not batched).
    dim : int
        Vector dimension.
    n_moments : int
        Number of Chebyshev moments. Higher = sharper features.
        2000 resolves ~1 mRy features for a 1 Ry bandwidth.
    n_random : int
        Number of stochastic trace vectors. 20-40 is typical.
    n_grid : int
        Energy grid density for the reconstructed DOS.
    E_min, E_max : float, optional
        Spectrum bounds. Estimated via Lanczos if not provided.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    DOSResult with E_grid, rho, spectrum bounds, and moments.
    """
    # ── Spectrum bounds ──
    if E_min is None or E_max is None:
        if verbose:
            print("  Estimating spectrum bounds via Lanczos...")
        E_min_est, E_max_est = estimate_spectrum(apply_H, dim,
                                                  n_lanczos=n_lanczos, seed=seed)
        E_min = E_min if E_min is not None else E_min_est
        E_max = E_max if E_max is not None else E_max_est

    center = (E_max + E_min) / 2.0
    half_width = (E_max - E_min) / 2.0
    if verbose:
        print(f"  Spectrum: [{E_min:.4f}, {E_max:.4f}] "
              f"(center={center:.4f}, half_width={half_width:.4f})")

    # ── Rescaled matvec ──
    def apply_H_tilde(x):
        return (apply_H(x) - center * x) / half_width

    # ── Chebyshev moments ──
    if verbose:
        print(f"  Computing {n_moments} Chebyshev moments "
              f"with {n_random} random vectors...")
    mu_raw = chebyshev_moments(apply_H_tilde, dim, n_moments, n_random,
                                seed=seed, verbose=verbose)

    # ── Jackson damping ──
    sigma = jackson_coefficients(n_moments)
    mu_damped = mu_raw * sigma

    # ── DOS reconstruction ──
    E_grid = np.linspace(E_min, E_max, n_grid)
    rho = reconstruct_dos(mu_damped, E_grid, center, half_width)

    if verbose:
        total_states = float(np.trapz(rho, E_grid))
        print(f"  DOS integral: {total_states:.1f} states "
              f"(dim={dim})")

    return DOSResult(
        E_grid=E_grid, rho=rho,
        E_min=E_min, E_max=E_max,
        center=center, half_width=half_width,
        mu_raw=mu_raw, mu_damped=mu_damped,
        n_moments=n_moments, n_random=n_random,
    )


def dos_weighted_windows(
    E_grid: np.ndarray,
    rho: np.ndarray,
    E_cross: float,
    E_max: float,
    *,
    galerkin_order: int = 1,
    tau: float | None = None,
    n_windows_target: int | None = None,
) -> np.ndarray:
    """DOS-weighted window partition for pseudobands.

    Places window boundaries so each window contributes roughly equal
    error to χ⁰, using the actual KPM density of states rather than
    the free-electron √E assumption.

    The per-window error bound (from Altman SI Eq. S36, generalized
    to Galerkin block order k) scales as:

        n_j^eff · σ_j^{2k} / ε̄_j^{2k+1} = τ  (constant across windows)

    where n_j^eff = ∫ ρ(E) dE over window j, σ_j ≈ Δ_j/√12 is the
    energy spread, and ε̄_j is the window center (from E_F).

    Parameters
    ----------
    E_grid : (n_grid,) energy grid (absolute energies, not relative to E_F)
    rho : (n_grid,) DOS on that grid (states per unit energy)
    E_cross : float — lower boundary (absolute energy, = E_F + ε_cross)
    E_max : float — upper boundary
    galerkin_order : int — Ritz block order k (1 = single-pole, higher = moment-matched)
    tau : float, optional — error tolerance per window. If None, set via n_windows_target.
    n_windows_target : int, optional — target number of windows (sets tau by bisection).
        One of tau or n_windows_target must be provided.

    Returns
    -------
    boundaries : (N_S+1,) array of window boundary energies.
    """
    # Sort and clip to [E_cross, E_max]
    order = np.argsort(E_grid)
    E_sorted = E_grid[order]
    rho_sorted = np.maximum(rho[order], 0.0)

    mask = (E_sorted >= E_cross) & (E_sorted <= E_max)
    E_s = E_sorted[mask]
    rho_s = rho_sorted[mask]
    if len(E_s) < 2:
        return np.array([E_cross, E_max])

    # Cumulative state count N(E) = ∫_{E_cross}^{E} ρ dE'
    dE = np.diff(E_s)
    rho_avg = 0.5 * (rho_s[1:] + rho_s[:-1])
    cdf = np.concatenate(([0.0], np.cumsum(rho_avg * dE)))

    k = galerkin_order

    def _place_windows(tau_val):
        """Walk up the CDF placing boundaries at equal-error intervals."""
        bounds = [E_cross]
        while bounds[-1] < E_max - 1e-12:
            e_lo = bounds[-1]
            # Binary search for e_hi such that the error metric = tau_val
            lo_idx = np.searchsorted(E_s, e_lo)
            best_hi = E_max
            for trial in np.linspace(e_lo + (E_max - e_lo) * 0.001, E_max, 500):
                hi_idx = np.searchsorted(E_s, trial)
                if hi_idx <= lo_idx:
                    continue
                n_eff = np.interp(trial, E_s, cdf) - np.interp(e_lo, E_s, cdf)
                delta = trial - e_lo
                eps_bar = 0.5 * (trial + e_lo) - E_cross + E_cross  # center from E_F=0 proxy
                # Use center of window as ε̄ (distance from Fermi level)
                # For conduction: ε̄ ≈ (e_lo + trial) / 2 measured from E_F
                # We use E_cross as proxy for E_F offset since E_cross = E_F + ε_cross
                sigma_approx = delta / np.sqrt(12.0)
                metric = n_eff * sigma_approx ** (2 * k) / max(eps_bar, 1e-30) ** (2 * k + 1)
                if metric >= tau_val:
                    best_hi = trial
                    break
            bounds.append(min(best_hi, E_max))
            if best_hi >= E_max:
                break
        bounds[-1] = E_max
        return np.array(bounds)

    if tau is not None:
        return _place_windows(tau)

    if n_windows_target is None:
        raise ValueError("Provide either tau or n_windows_target")

    # Binary search for tau that gives the target window count
    tau_lo, tau_hi = 1e-30, 1e10
    for _ in range(50):
        tau_mid = np.sqrt(tau_lo * tau_hi)  # geometric mean for log-scale search
        bounds = _place_windows(tau_mid)
        n_w = len(bounds) - 1
        if n_w > n_windows_target:
            tau_lo = tau_mid  # too many windows → increase tau (less strict)
        else:
            tau_hi = tau_mid
        if abs(n_w - n_windows_target) <= 1:
            break
    return _place_windows(tau_mid)


def geometric_windows(
    E_cross: float,
    E_max: float,
    F: float = 0.10,
) -> np.ndarray:
    """Geometric window partition: ε_j = (1+F)^j · ε_cross.

    Simple heuristic from Altman et al. — assumes free-electron DOS.
    Prefer dos_weighted_windows when the actual KPM DOS is available.
    """
    boundaries = [E_cross]
    while boundaries[-1] < E_max:
        boundaries.append(boundaries[-1] * (1.0 + F))
    boundaries[-1] = E_max
    return np.array(boundaries)

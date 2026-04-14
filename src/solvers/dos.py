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


def geometric_windows(
    E_cross: float,
    E_max: float,
    F: float = 0.10,
) -> np.ndarray:
    """Geometric window partition: ε_j = (1+F)^j · ε_cross.

    Returns (N_S+1,) array of boundary energies (measured from E_F=0),
    starting at E_cross and ending at E_max.
    """
    boundaries = [E_cross]
    while boundaries[-1] < E_max:
        boundaries.append(boundaries[-1] * (1.0 + F))
    boundaries[-1] = E_max
    return np.array(boundaries)

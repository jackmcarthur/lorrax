"""
solvers/pseudobands.py — Matrix-free Ritz pseudobands via Chebyshev-Jackson filtering.

Replaces explicit diagonalization (BGW Parabands) with a matrix-free procedure:
  1. KPM DOS → spectrum bounds + window partition  (solvers.dos)
  2. CJ boundary filters → telescoping recurrence  (one pass of M_max matvecs)
  3. Per-window Galerkin-Ritz → pseudoband vectors  (k×k dense eigh per window)

Output: deterministic eigenstates (from Davidson) + weighted Ritz pseudobands,
drop-in compatible with BGW/LORRAX GW codes. Pseudobands carry non-unit norm
encoding their spectral weight.

Physics-agnostic — works for H_DFT, H_BSE, or any Hermitian operator.

Usage
-----
    from solvers.pseudobands import ritz_pseudobands

    result = ritz_pseudobands(
        apply_H, dim=ngkmax,
        Phi_det=davidson_evecs, E_det=davidson_evals,
        E_fermi=0.0,
    )
    # result.Phi_out: (n_det + N_S*k, dim) — all bands
    # result.E_out:   (n_det + N_S*k,) — eigenvalues/pseudo-energies
    # result.weights: (n_det + N_S*k,) — 1.0 for det, sqrt(n_eff/k) for pseudo
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from solvers.chebyshev import jackson_coefficients
from solvers.dos import compute_dos, dos_weighted_windows, geometric_windows, DOSResult


# ═══════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PseudobandsResult:
    """Output of ritz_pseudobands."""
    Phi_out: np.ndarray     # (n_total, dim) — deterministic + pseudoband vectors
    E_out: np.ndarray       # (n_total,) — eigenvalues / pseudo-energies
    weights: np.ndarray     # (n_total,) — 1.0 for det, sqrt(n_eff/k) for pseudo
    n_det: int              # number of deterministic bands
    n_pseudo: int           # number of pseudobands
    n_windows: int          # number of spectral windows
    dos: DOSResult          # the KPM DOS used for partitioning


# ═══════════════════════════════════════════════════════════════════════
#  §4: Boundary filter coefficients
# ═══════════════════════════════════════════════════════════════════════

def _boundary_coefficients(tilde_eps: np.ndarray, M_max: int) -> np.ndarray:
    """Chebyshev-Jackson coefficients for cumulative step filters.

    For each boundary b in tilde_eps, computes the damped coefficients
    c_n^(b) for the smoothed step function C_b(E) ≈ 1_{[-1, b]}(E).

    Parameters
    ----------
    tilde_eps : (N_S+1,) rescaled boundary energies in [-1, 1]
    M_max : Chebyshev order

    Returns
    -------
    coeffs : (N_S+1, M_max) damped Chebyshev coefficients
    """
    n_bounds = len(tilde_eps)
    g = jackson_coefficients(M_max - 1)  # (M_max,) Jackson dampers

    coeffs = np.zeros((n_bounds, M_max))
    ns = np.arange(M_max)

    for j in range(n_bounds):
        b = tilde_eps[j]
        gamma = np.zeros(M_max)
        gamma[0] = 1.0 - np.arccos(np.clip(b, -1, 1)) / np.pi
        gamma[1:] = -2.0 / (np.pi * ns[1:]) * np.sin(ns[1:] * np.arccos(np.clip(b, -1, 1)))
        coeffs[j] = gamma * g

    return coeffs


# ═══════════════════════════════════════════════════════════════════════
#  §5: Telescoping Chebyshev recurrence (JIT'd)
# ═══════════════════════════════════════════════════════════════════════

def _telescoping_filter(
    apply_H: Callable,
    Omega: jax.Array,
    coeffs: jax.Array,
    center: float,
    half_width: float,
    M_max: int,
) -> jax.Array:
    """Run M_max block matvecs, accumulating N_S+1 boundary filters simultaneously.

    Parameters
    ----------
    apply_H : (k, dim) → (k, dim) — Hermitian block matvec
    Omega : (k, dim) — random starting block
    coeffs : (N_S+1, M_max) — per-boundary damped Chebyshev coefficients
    center, half_width : spectral rescaling parameters
    M_max : Chebyshev order

    Returns
    -------
    Y : (N_S, k, dim) — per-window filtered blocks (telescoped)
    """
    n_bounds = coeffs.shape[0]
    k, dim = Omega.shape

    def apply_H_tilde(x):
        return (apply_H(x) - center * x) / half_width

    # Initialize recurrence
    T_prev = Omega                          # T_0
    T_curr = apply_H_tilde(Omega)           # T_1

    # Initialize accumulators: A[j] = c[j,0]*T_0 + c[j,1]*T_1
    A = coeffs[:, 0, None, None] * T_prev[None, :, :] \
      + coeffs[:, 1, None, None] * T_curr[None, :, :]  # (N_S+1, k, dim)

    # Chebyshev recurrence: T_n = 2*H_tilde*T_{n-1} - T_{n-2}
    def body(n, carry):
        A, T_prev, T_curr = carry
        T_new = 2.0 * apply_H_tilde(T_curr) - T_prev
        A = A + coeffs[:, n, None, None] * T_new[None, :, :]
        return A, T_curr, T_new

    A, _, _ = jax.lax.fori_loop(2, M_max, body, (A, T_prev, T_curr))

    # Telescope: Y_j = A_j - A_{j-1}
    Y = A[1:] - A[:-1]  # (N_S, k, dim)
    return Y


# ═══════════════════════════════════════════════════════════════════════
#  §6: Per-window Galerkin-Ritz extraction
# ═══════════════════════════════════════════════════════════════════════

def _galerkin_ritz(
    apply_H: Callable,
    Y_j: jax.Array,
    Phi_det: jax.Array | None,
    Q_prev: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Extract Ritz vectors from a filtered block.

    Parameters
    ----------
    apply_H : (k, dim) → (k, dim)
    Y_j : (k, dim) — filtered block for this window
    Phi_det : (n_det, dim) or None — deterministic eigenvectors to deflate
    Q_prev : (k, dim) or None — previous window's Q for cross-window orthogonalization

    Returns
    -------
    Xi : (k, dim) — Ritz pseudoband vectors (orthonormal)
    theta : (k,) — Ritz eigenvalues
    Q : (k, dim) — orthonormal basis (for next window's deflation)
    """
    # Deflate against deterministic manifold
    if Phi_det is not None and Phi_det.shape[0] > 0:
        overlap = jnp.einsum('nd,kd->nk', jnp.conj(Phi_det), Y_j)
        Y_j = Y_j - jnp.einsum('nk,nd->kd', overlap, Phi_det)

    # Cross-window orthogonalization (§7)
    if Q_prev is not None:
        overlap = jnp.einsum('pd,kd->pk', jnp.conj(Q_prev), Y_j)
        Y_j = Y_j - jnp.einsum('pk,pd->kd', overlap, Q_prev)

    # Economy QR
    Q, R = jnp.linalg.qr(Y_j.T)  # Q: (dim, k), R: (k, k)
    Q = Q.T  # (k, dim) — orthonormal rows

    # Galerkin matrix: H_proj = Q† H Q
    HQ = apply_H(Q)  # (k, dim)
    H_proj = jnp.einsum('kd,ld->kl', jnp.conj(Q), HQ)
    H_proj = 0.5 * (H_proj + H_proj.conj().T)  # enforce Hermitian

    # Diagonalize k×k matrix
    theta, S = jnp.linalg.eigh(H_proj)

    # Ritz vectors
    Xi = jnp.einsum('kl,kd->ld', S.T, Q)  # (k, dim)

    return Xi, theta, Q


# ═══════════════════════════════════════════════════════════════════════
#  §6 continued: window weight from KPM DOS
# ═══════════════════════════════════════════════════════════════════════

def _window_weights(
    dos: DOSResult,
    boundaries: np.ndarray,
    coeffs: np.ndarray,
    M_max: int,
) -> np.ndarray:
    """Compute effective state count n_j^eff per window from the KPM DOS.

    Uses the Jackson-smoothed bandpass envelope w_j(E)² weighted by ρ(E).

    Returns (N_S,) array of n_j^eff.
    """
    E_grid = dos.E_grid
    E_tilde = (E_grid - dos.center) / dos.half_width
    E_tilde = np.clip(E_tilde, -1 + 1e-10, 1 - 1e-10)
    n_grid = len(E_grid)
    N_S = len(boundaries) - 1

    # Precompute Chebyshev polynomials on the energy grid
    T = np.zeros((M_max, n_grid))
    T[0] = 1.0
    if M_max > 1:
        T[1] = E_tilde
    for n in range(2, M_max):
        T[n] = 2.0 * E_tilde * T[n-1] - T[n-2]

    # Compute bandpass envelope w_j(E) = C_j(E) - C_{j-1}(E)
    n_eff = np.zeros(N_S)
    for j in range(N_S):
        w_coeffs = coeffs[j+1] - coeffs[j]  # telescoped coefficients
        w_j = np.einsum('n,ng->g', w_coeffs, T)  # (n_grid,)
        n_eff[j] = float(np.trapz(dos.rho * w_j**2, E_grid))

    return np.maximum(n_eff, 0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Top-level: ritz_pseudobands
# ═══════════════════════════════════════════════════════════════════════

def ritz_pseudobands(
    apply_H: Callable,
    dim: int,
    *,
    Phi_det: np.ndarray | None = None,
    E_det: np.ndarray | None = None,
    E_fermi: float = 0.0,
    F: float = 0.10,
    k: int = 6,
    M_max: int = 1500,
    C_m: float = 1.0,
    n_kpm_moments: int = 4000,
    n_kpm_random: int = 40,
    n_windows_target: int | None = None,
    dos_result: DOSResult | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> PseudobandsResult:
    """Matrix-free Ritz pseudobands via Chebyshev-Jackson filtering.

    Parameters
    ----------
    apply_H : (batch, dim) → (batch, dim) — Hermitian block matvec
    dim : vector dimension
    Phi_det : (n_det, dim) — Davidson eigenvectors (protected region)
    E_det : (n_det,) — their eigenvalues
    E_fermi : Fermi energy (default 0)
    F : window ratio (for geometric fallback or crossover energy)
    k : Galerkin block size per window
    M_max : Chebyshev order cap
    C_m : filter sharpness constant
    n_kpm_moments : KPM moment count for DOS
    n_kpm_random : KPM trace probes
    n_windows_target : target number of windows (sets tau by bisection)
    dos_result : pre-computed DOSResult (skips KPM if provided)
    seed : random seed
    verbose : print progress

    Returns
    -------
    PseudobandsResult with Phi_out, E_out, weights, metadata.
    """
    if Phi_det is None:
        Phi_det = np.zeros((0, dim), dtype=np.complex128)
        E_det = np.array([], dtype=np.float64)
    Phi_det = np.asarray(Phi_det)
    E_det = np.asarray(E_det)
    n_det = Phi_det.shape[0]

    # Flat matvec for KPM (unbatched)
    def apply_H_flat(x):
        return apply_H(x[None, :])[0]

    # ── §2: KPM DOS ──
    if dos_result is None:
        if verbose:
            print("Pseudobands: computing KPM DOS...")
        dos = compute_dos(apply_H_flat, dim, n_moments=n_kpm_moments,
                          n_random=n_kpm_random, seed=seed, verbose=verbose)
    else:
        dos = dos_result

    B = 2.0 * dos.half_width
    E_max = dos.E_max

    # ── §3: Window partition ──
    eps_cross = C_m * np.pi * B / (F * M_max)
    if verbose:
        print(f"  Crossover energy: ε_cross = {eps_cross:.4f} "
              f"(E_F + ε_cross = {E_fermi + eps_cross:.4f})")

    # Check Davidson coverage
    if n_det > 0 and np.max(E_det) - E_fermi < eps_cross:
        print(f"  WARNING: Davidson region ends at {np.max(E_det):.4f} "
              f"but ε_cross = {E_fermi + eps_cross:.4f}. "
              f"Extend Davidson to cover the crossover energy.")

    E_cross_abs = E_fermi + eps_cross
    E_max_abs = E_max

    if n_windows_target is not None:
        boundaries = dos_weighted_windows(
            dos.E_grid, dos.rho, E_cross_abs, E_max_abs,
            galerkin_order=1, n_windows_target=n_windows_target)
    else:
        boundaries = geometric_windows(eps_cross, E_max - E_fermi, F)
        boundaries = boundaries + E_fermi  # convert to absolute

    N_S = len(boundaries) - 1
    if verbose:
        print(f"  {N_S} spectral windows from {boundaries[0]:.4f} to {boundaries[-1]:.4f}")

    # Rescaled boundaries
    tilde_eps = (boundaries - dos.center) / dos.half_width

    # ── §4: Filter coefficients ──
    coeffs = _boundary_coefficients(tilde_eps, M_max)

    # ── §5: Telescoping recurrence ──
    if verbose:
        print(f"  Running {M_max} Chebyshev iterations (block size {k})...")
    key = jax.random.PRNGKey(seed + 42)
    Omega = jax.random.normal(key, (k, dim), dtype=jnp.float64)
    Omega = Omega + 0j  # complex

    coeffs_j = jnp.asarray(coeffs, dtype=jnp.float64)
    Y_all = _telescoping_filter(apply_H, Omega, coeffs_j,
                                 dos.center, dos.half_width, M_max)
    Y_all = np.asarray(Y_all)  # (N_S, k, dim)

    if verbose:
        print(f"  Filtered blocks: {Y_all.shape}")

    # ── §6-7: Per-window Galerkin-Ritz + cross-window orthogonalization ──
    Phi_det_j = jnp.asarray(Phi_det, dtype=jnp.complex128) if n_det > 0 else None
    n_eff = _window_weights(dos, boundaries, coeffs, M_max)

    Xi_list = []
    theta_list = []
    weight_list = []
    Q_prev = None

    for j in range(N_S):
        Y_j = jnp.asarray(Y_all[j], dtype=jnp.complex128)
        Xi_j, theta_j, Q_j = _galerkin_ritz(apply_H, Y_j, Phi_det_j, Q_prev)

        Xi_np = np.asarray(Xi_j)
        theta_np = np.asarray(theta_j)
        w_j = np.sqrt(max(n_eff[j], 0.0) / k)

        Xi_list.append(Xi_np * w_j)  # weight absorbed into wavefunction
        theta_list.append(theta_np)
        weight_list.append(np.full(k, w_j))

        Q_prev = Q_j

        if verbose and (j < 3 or j == N_S - 1 or (j + 1) % 10 == 0):
            print(f"    window {j+1}/{N_S}: n_eff={n_eff[j]:.1f}, "
                  f"θ=[{theta_np[0]:.4f}, {theta_np[-1]:.4f}]")

    # ── Output assembly ──
    if Xi_list:
        Phi_pseudo = np.concatenate(Xi_list, axis=0)
        E_pseudo = np.concatenate(theta_list)
        w_pseudo = np.concatenate(weight_list)
    else:
        Phi_pseudo = np.zeros((0, dim), dtype=np.complex128)
        E_pseudo = np.array([], dtype=np.float64)
        w_pseudo = np.array([], dtype=np.float64)

    Phi_out = np.concatenate([Phi_det, Phi_pseudo], axis=0)
    E_out = np.concatenate([E_det, E_pseudo])
    weights = np.concatenate([np.ones(n_det), w_pseudo])

    if verbose:
        total_eff = float(np.sum(n_eff))
        print(f"  Total: {n_det} det + {N_S * k} pseudo = {Phi_out.shape[0]} bands "
              f"(Σ n_eff = {total_eff:.1f})")

    return PseudobandsResult(
        Phi_out=Phi_out, E_out=E_out, weights=weights,
        n_det=n_det, n_pseudo=N_S * k, n_windows=N_S,
        dos=dos,
    )

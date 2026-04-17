"""
solvers/pseudobands_v2.py — Galerkin-Ritz pseudobands with Gauss-quadrature energies.

Per-window: Ritz gives directions, Gauss quadrature of the windowed DOS
gives spectral pole locations and weights. Shifted CJ boundaries enforce
a quadratic partition of unity (Σ w_j² ≈ 1).

Three window types:
  1. Protected: Davidson eigenstates with weight 1.0
  2. Davidson windows: exact eigenstates above protected region, Ritz + Gauss
  3. CJ windows: Chebyshev-filtered Galerkin-Ritz + Gauss

Usage
-----
    from solvers.pseudobands_v2 import ritz_pseudobands_v2
    result = ritz_pseudobands_v2(apply_H, dim, Phi_dav=evecs, E_dav=evals)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from solvers.chebyshev import jackson_coefficients
from solvers.dos import compute_dos, dos_weighted_windows, compute_window_partition, DOSResult


# ═══════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PseudobandsResult:
    Phi_out: np.ndarray       # (n_prot + N_S*k, dim)
    E_out: np.ndarray         # (n_prot + N_S*k,)
    weights: np.ndarray       # (n_prot + N_S*k,)
    n_prot: int
    n_pseudo: int
    n_windows: int
    n_dav_windows: int
    n_cj_windows: int
    dos: DOSResult
    windows: np.ndarray       # (N_S+1,) boundaries


# ═══════════════════════════════════════════════════════════════════════
#  Shifted CJ boundary coefficients (§5)
# ═══════════════════════════════════════════════════════════════════════

def _shifted_boundary_coefficients(
    boundaries: np.ndarray,
    center: float,
    half_width: float,
    M_max: int,
    cj_windows: np.ndarray,  # indices of CJ windows
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Compute CJ coefficients with shifted internal boundaries.

    Each internal boundary gets two cumulatives at ε ± δ where δ = π/(2M).
    Outer boundaries are unshifted.

    Returns
    -------
    coeffs : (n_cumulatives, M_max) — damped Chebyshev coefficients
    window_pairs : list of (idx_upper, idx_lower) per CJ window
    """
    delta = np.pi / (2 * M_max)  # shift in rescaled units
    g = jackson_coefficients(M_max - 1)
    ns = np.arange(M_max)

    # Build list of rescaled positions for all needed cumulatives
    positions = []
    window_pairs = []

    for j in cj_windows:
        e_lo = (boundaries[j] - center) / half_width
        e_hi = (boundaries[j + 1] - center) / half_width

        # Lower: shift down (unless it's the very first boundary)
        if j == 0:
            b_lo = np.clip(e_lo, -1.0, 1.0)
        else:
            b_lo = np.clip(e_lo - delta, -1.0, 1.0)

        # Upper: shift up (unless it's the very last boundary)
        if j + 1 == len(boundaries) - 1:
            b_hi = np.clip(e_hi, -1.0, 1.0)
        else:
            b_hi = np.clip(e_hi + delta, -1.0, 1.0)

        idx_lo = len(positions)
        positions.append(b_lo)
        idx_hi = len(positions)
        positions.append(b_hi)
        window_pairs.append((idx_hi, idx_lo))

    # Compute coefficients for all positions
    n_cum = len(positions)
    coeffs = np.zeros((n_cum, M_max))

    for i, b in enumerate(positions):
        gamma = np.zeros(M_max)
        gamma[0] = 1.0 - np.arccos(np.clip(b, -1, 1)) / np.pi
        gamma[1:] = -2.0 / (np.pi * ns[1:]) * np.sin(
            ns[1:] * np.arccos(np.clip(b, -1, 1)))
        coeffs[i] = gamma * g

    return coeffs, window_pairs


# ═══════════════════════════════════════════════════════════════════════
#  Telescoping CJ filter (§6)
# ═══════════════════════════════════════════════════════════════════════

def _telescoping_filter(
    apply_H: Callable,
    Omega: jax.Array,
    coeffs: jax.Array,
    center: float,
    half_width: float,
    M_max: int,
) -> jax.Array:
    """M_max block matvecs accumulating all boundary cumulatives.

    Returns A: (n_cumulatives, k, dim) — accumulated filter outputs.
    """
    def apply_H_tilde(x):
        return (apply_H(x) - center * x) / half_width

    T_prev = Omega
    T_curr = apply_H_tilde(Omega)

    A = (coeffs[:, 0, None, None] * T_prev[None, :, :]
       + coeffs[:, 1, None, None] * T_curr[None, :, :])

    def body(n, carry):
        A, T_prev, T_curr = carry
        T_new = 2.0 * apply_H_tilde(T_curr) - T_prev
        A = A + coeffs[:, n, None, None] * T_new[None, :, :]
        return A, T_curr, T_new

    A, _, _ = jax.lax.fori_loop(2, M_max, body, (A, T_prev, T_curr))
    return A


# ═══════════════════════════════════════════════════════════════════════
#  Gauss quadrature from power moments (§8)
# ═══════════════════════════════════════════════════════════════════════

def _gauss_from_moments(moments: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k-point Gauss quadrature from 2k power moments via modified Chebyshev / Stieltjes.

    Builds the k×k Jacobi (tridiagonal) matrix from the moment sequence
    m_0, m_1, ..., m_{2k-1} using the Golub-Welsch algorithm.

    Returns (nodes, weights) each shape (k,).
    """
    if moments[0] < 1e-30:
        return np.zeros(k), np.zeros(k)

    # Build Hankel matrix and solve for recurrence coefficients
    # Using the classic approach: construct the tridiagonal Jacobi matrix
    # from the moments via the Lanczos algorithm on the moment matrix.
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            H[i, j] = moments[i + j]

    # Cholesky of Hankel for stability
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition for semi-definite case
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.maximum(eigvals, 0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    # Build the Jacobi matrix from the sigma vectors
    # Using the Stieltjes procedure (stable for k ≤ 6)
    alpha = np.zeros(k)
    beta = np.zeros(k)

    # Initialize: pi_{-1} = 0, pi_0 = 1
    # alpha_j = <E pi_j, pi_j> / <pi_j, pi_j>
    # beta_j = <pi_j, pi_j> / <pi_{j-1}, pi_{j-1}>
    # where <f, g> = ∫ f(E) g(E) dμ(E) = Σ_n f(E_n) g(E_n) w_n

    # For power moments, use the discretized Stieltjes:
    # We work directly with the Hankel matrix
    sigma_prev = np.zeros(2 * k)
    sigma_curr = moments[:2 * k].copy()

    alpha[0] = moments[1] / moments[0]
    beta[0] = moments[0]

    for j in range(1, k):
        sigma_next = np.zeros(2 * k)
        for l in range(j, 2 * k - j):
            sigma_next[l] = (sigma_curr[l + 1] - alpha[j - 1] * sigma_curr[l]
                             - beta[j - 1] * sigma_prev[l])

        alpha[j] = (sigma_next[j + 1] / sigma_next[j]
                     - sigma_curr[j] / sigma_curr[j - 1]) if abs(sigma_next[j]) > 1e-30 else 0.0
        beta[j] = sigma_next[j] / sigma_curr[j - 1] if abs(sigma_curr[j - 1]) > 1e-30 else 0.0

        sigma_prev = sigma_curr
        sigma_curr = sigma_next

    # Build tridiagonal Jacobi matrix
    J = np.diag(alpha)
    for j in range(k - 1):
        off = np.sqrt(max(beta[j + 1], 0.0))
        J[j, j + 1] = off
        J[j + 1, j] = off

    # Eigendecomposition gives nodes and weights
    nodes, vecs = np.linalg.eigh(J)
    weights = moments[0] * vecs[0, :] ** 2

    # Sort ascending
    order = np.argsort(nodes)
    return nodes[order], weights[order]


def _compute_window_moments(
    E_grid: np.ndarray,
    rho: np.ndarray,
    w_j: np.ndarray,
    n_moments: int,
    E_shift: float = 0.0,
    E_scale: float = 1.0,
) -> np.ndarray:
    """Compute power moments m_n = ∫ x^n w_j(E) ρ(E) dE where x = (E - shift)/scale.

    Clips to the window support (|w_j| > 1e-6) to avoid pollution from
    high-order monomials far outside the window.
    """
    # Restrict to window support
    support = np.abs(w_j) > 1e-6
    if not np.any(support):
        return np.zeros(n_moments)
    E_s = E_grid[support]
    w_s = w_j[support]
    rho_s = np.maximum(rho[support], 0.0)
    x = (E_s - E_shift) / E_scale
    integrand_base = w_s * rho_s
    moments = np.zeros(n_moments)
    for n in range(n_moments):
        moments[n] = float(np.trapezoid(integrand_base * x**n, E_s))
    return moments


def _compute_window_moments_discrete(
    E_eig: np.ndarray,
    n_moments: int,
    E_shift: float = 0.0,
    E_scale: float = 1.0,
) -> np.ndarray:
    """Power moments from discrete eigenvalues in shifted/scaled coords."""
    x = (E_eig - E_shift) / E_scale
    moments = np.zeros(n_moments)
    for n in range(n_moments):
        moments[n] = float(np.sum(x**n))
    return moments


# ═══════════════════════════════════════════════════════════════════════
#  CJ window indicator on energy grid (for Gauss moments)
# ═══════════════════════════════════════════════════════════════════════

def _cj_window_on_grid(
    E_grid: np.ndarray,
    b_lo: float,
    b_hi: float,
    center: float,
    half_width: float,
    M_max: int,
) -> np.ndarray:
    """Evaluate CJ window indicator w_j(E) = C(b_hi) - C(b_lo) on a grid."""
    g = jackson_coefficients(M_max - 1)
    ns = np.arange(M_max)
    e = np.clip((E_grid - center) / half_width, -0.9999, 0.9999)
    arccos_e = np.arccos(e)
    T = np.cos(ns[:, None] * arccos_e[None, :])  # (M, n_grid)

    result = np.zeros(len(E_grid))
    for b_abs in [b_hi, b_lo]:
        # Rescale boundary to [-1, 1]
        b = np.clip((b_abs - center) / half_width, -1.0, 1.0)
        gamma = np.zeros(M_max)
        gamma[0] = 1.0 - np.arccos(b) / np.pi
        gamma[1:] = -2.0 / (np.pi * ns[1:]) * np.sin(ns[1:] * np.arccos(b))
        step = (gamma * g) @ T
        if b_abs == b_hi:
            result += step
        else:
            result -= step
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Per-window Galerkin-Ritz (§7)
# ═══════════════════════════════════════════════════════════════════════

def _galerkin_ritz_cj(
    apply_H: Callable,
    Y_j: jax.Array,
    Phi_dav: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """CJ window: deflate, QR, Galerkin eigh → Ritz vectors + eigenvalues."""
    # Deflate against all Davidson states
    if Phi_dav is not None and Phi_dav.shape[0] > 0:
        overlap = jnp.einsum('nd,kd->nk', jnp.conj(Phi_dav), Y_j)
        Y_j = Y_j - jnp.einsum('nk,nd->kd', overlap, Phi_dav)

    # Economy QR
    Q, R = jnp.linalg.qr(Y_j.T)
    Q = Q.T  # (k, dim)

    # Galerkin: H_proj = Q† H Q
    HQ = apply_H(Q)
    H_proj = jnp.einsum('kd,ld->kl', jnp.conj(Q), HQ)
    H_proj = 0.5 * (H_proj + H_proj.conj().T)

    theta, S = jnp.linalg.eigh(H_proj)
    Xi = jnp.einsum('kl,kd->ld', S.T, Q)  # Ritz vectors (k, dim)
    return Xi, theta


def _galerkin_ritz_dav(
    Phi_window: np.ndarray,
    E_window: np.ndarray,
    k: int,
    rng_key: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Davidson window: Ritz from exact eigenstates. No matvec needed.

    If n_states ≤ k: use eigenstates directly (pad with zeros).
    If n_states > k: random-phase project to rank k, Galerkin on stored evals.
    """
    n = Phi_window.shape[0]
    dim = Phi_window.shape[1]

    if n <= k:
        # Use eigenstates directly; pad to k
        Xi = np.zeros((k, dim), dtype=Phi_window.dtype)
        theta = np.full(k, np.mean(E_window) if n > 0 else 0.0)
        Xi[:n] = Phi_window
        theta[:n] = E_window
        return Xi, theta

    # Random-phase project to rank k
    phases = jax.random.uniform(rng_key, (n, k), minval=0.0, maxval=2.0 * np.pi)
    R = jnp.exp(1j * phases)  # (n, k)
    Phi_j = jnp.asarray(Phi_window, dtype=jnp.complex128)
    Z = jnp.einsum('nd,na->ad', Phi_j, R)  # (k, dim)

    # QR
    Q, _ = jnp.linalg.qr(Z.T)
    Q = Q.T  # (k, dim)

    # Galerkin from stored eigenvalues: H_proj = (Φ†Q)† diag(E) (Φ†Q)
    overlap = jnp.einsum('nd,kd->nk', jnp.conj(Phi_j), Q)  # (n, k)
    E_j = jnp.asarray(E_window)
    H_proj = jnp.einsum('nk,n,nl->kl', jnp.conj(overlap), E_j, overlap)
    H_proj = 0.5 * (H_proj + H_proj.conj().T)

    theta, S = jnp.linalg.eigh(H_proj)
    Xi = jnp.einsum('kl,kd->ld', S.T, Q)

    return np.asarray(Xi), np.asarray(theta)


# ═══════════════════════════════════════════════════════════════════════
#  Window placement with n_min floor (§3)
# ═══════════════════════════════════════════════════════════════════════

def _place_windows_v2(
    E_grid: np.ndarray,
    rho: np.ndarray,
    E_start: float,
    E_max: float,
    n_windows_target: int,
    n_min: int,
    dim: int,
) -> np.ndarray:
    """Place window boundaries with equal-error rule + n_min state floor.

    Each window must contain at least n_min states (from DOS integral × dim).
    """
    mask = (E_grid >= E_start) & (E_grid <= E_max)
    E_s = E_grid[mask]
    rho_s = np.maximum(rho[mask], 0.0)
    if len(E_s) < 2:
        return np.array([E_start, E_max])

    # Cumulative state count
    cdf = np.concatenate(([0.0], np.cumsum(
        0.5 * (rho_s[1:] + rho_s[:-1]) * np.diff(E_s)))) * dim

    total_states = cdf[-1]

    def _place(tau_val):
        bounds = [E_start]
        while bounds[-1] < E_max - 1e-12:
            e_lo = bounds[-1]
            n_lo = np.interp(e_lo, E_s, cdf)

            # Find e_hi: enforce both error metric ≤ tau AND n_eff ≥ n_min
            best_hi = E_max
            for trial in np.linspace(e_lo + (E_max - e_lo) * 0.001, E_max, 500):
                n_trial = np.interp(trial, E_s, cdf)
                n_eff = n_trial - n_lo
                delta = trial - e_lo
                eps_bar = 0.5 * (trial + e_lo)
                sigma = delta / np.sqrt(12.0)
                metric = n_eff * sigma**2 / max(eps_bar, 1e-30)**3

                if metric >= tau_val and n_eff >= n_min:
                    best_hi = trial
                    break
            bounds.append(min(best_hi, E_max))
            if best_hi >= E_max:
                break
        bounds[-1] = E_max
        return np.array(bounds)

    # Binary search for tau to hit target window count
    tau_lo, tau_hi = 1e-30, 1e10
    for _ in range(50):
        tau_mid = np.sqrt(tau_lo * tau_hi)
        bounds = _place(tau_mid)
        n_w = len(bounds) - 1
        if n_w > n_windows_target:
            tau_lo = tau_mid
        else:
            tau_hi = tau_mid
        if abs(n_w - n_windows_target) <= 1:
            break
    return _place(tau_mid)


# ═══════════════════════════════════════════════════════════════════════
#  Top-level entry point
# ═══════════════════════════════════════════════════════════════════════

def ritz_pseudobands_v2(
    apply_H: Callable,
    dim: int,
    *,
    Phi_dav: np.ndarray | None = None,
    E_dav: np.ndarray | None = None,
    n_prot: int | None = None,
    E_fermi: float = 0.0,
    F: float = 0.10,
    k: int = 6,
    M_max: int = 1500,
    n_kpm_moments: int = 500,
    n_kpm_random: int = 10,
    n_windows_target: int = 40,
    dos_result: DOSResult | None = None,
    n_prot_override: int | None = None,
    boundaries_override: np.ndarray | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> PseudobandsResult:
    """Galerkin-Ritz pseudobands with Gauss-quadrature energies (v2).

    Parameters
    ----------
    apply_H : (dim,) → (dim,) Hermitian matvec
    dim : vector dimension (nspinor * ngkmax)
    Phi_dav : (n_dav, dim) Davidson eigenvectors
    E_dav : (n_dav,) Davidson eigenvalues
    n_prot : number of protected bands (default: auto from E_cross)
    E_fermi : Fermi energy
    F : window ratio
    k : pseudobands per window (≤ 6 recommended)
    M_max : Chebyshev order
    n_windows_target : target number of windows (≤ 100)
    dos_result : pre-computed DOSResult
    n_prot_override : fixed protected count (for k>0 consistency)
    seed : random seed
    verbose : print progress
    """
    if Phi_dav is None:
        Phi_dav = np.zeros((0, dim), dtype=np.complex128)
        E_dav = np.array([], dtype=np.float64)
    Phi_dav = np.asarray(Phi_dav)
    E_dav = np.asarray(E_dav)
    n_dav = Phi_dav.shape[0]

    apply_H_block = jax.vmap(apply_H)

    # ── §2: KPM DOS ──
    if dos_result is None:
        if verbose:
            print("  Computing KPM DOS...")
        dos = compute_dos(apply_H, dim, n_moments=n_kpm_moments,
                          n_random=n_kpm_random, seed=seed, verbose=verbose)
    else:
        dos = dos_result

    B = 2.0 * dos.half_width
    cj_resolution = np.pi * B / M_max

    # ── §3: Window placement ──
    # Protected region: either specified or from crossover energy
    if n_prot is not None:
        n_protected = min(n_prot, n_dav)
    else:
        eps_cross = cj_resolution / F
        E_cross_abs = E_fermi + eps_cross
        n_protected = int(np.sum(E_dav < E_cross_abs))

    if n_prot_override is not None:
        n_protected = n_prot_override

    # Ensure at least 1 protected band exists
    n_protected = max(n_protected, min(1, n_dav))

    E_start = float(E_dav[n_protected - 1]) if n_protected > 0 else E_fermi
    Phi_protected = Phi_dav[:n_protected]
    E_protected = E_dav[:n_protected]

    # Available Davidson bands for stochastic/Ritz windows
    Phi_avail = Phi_dav[n_protected:]
    E_avail = E_dav[n_protected:]
    E_dav_max = float(E_dav[-1]) if n_dav > 0 else E_start

    if verbose:
        print(f"  CJ resolution: {cj_resolution:.4f} Ry")
        print(f"  Protected: {n_protected} bands (E < {E_start:.4f} Ry)")
        print(f"  Available: {len(E_avail)} bands up to {E_dav_max:.4f} Ry")

    # Place windows with n_min floor (or use override from k=0)
    if boundaries_override is not None:
        boundaries = boundaries_override
    else:
        boundaries = _place_windows_v2(
            dos.E_grid, dos.rho, E_start, dos.E_max,
            n_windows_target=n_windows_target, n_min=k, dim=dim)
    N_S = len(boundaries) - 1

    if verbose:
        widths = np.diff(boundaries)
        print(f"  {N_S} windows [{boundaries[0]:.2f}, {boundaries[-1]:.2f}] Ry")
        print(f"  Widths: min={widths.min():.4f}, max={widths.max():.4f}")

    # ── §4: Classify windows ──
    is_dav = np.zeros(N_S, dtype=bool)
    dav_indices = []  # per-window: indices into Phi_avail
    for j in range(N_S):
        in_win = (E_avail >= boundaries[j]) & (E_avail < boundaries[j + 1])
        idx = np.where(in_win)[0]
        dav_indices.append(idx)
        if boundaries[j + 1] <= E_dav_max + 1e-10:
            is_dav[j] = True

    cj_windows = np.where(~is_dav)[0]
    dav_windows = np.where(is_dav)[0]
    n_dav_win = len(dav_windows)
    n_cj_win = len(cj_windows)

    if verbose:
        print(f"  Classification: {n_dav_win} Davidson, {n_cj_win} CJ")

    # ── §5-6: CJ filter ──
    Y_cj = {}  # window index → filtered block (k, dim)
    cj_boundary_data = {}  # window index → (b_lo_rescaled, b_hi_rescaled)

    if n_cj_win > 0:
        delta_rescaled = np.pi / (2 * M_max)
        coeffs, win_pairs = _shifted_boundary_coefficients(
            boundaries, dos.center, dos.half_width, M_max, cj_windows)

        if verbose:
            print(f"  Running {M_max} Chebyshev iterations (block size {k})...")

        rng = jax.random.PRNGKey(seed + 42)
        # Random PW phases per §6
        phases = jax.random.uniform(rng, (k, dim), minval=0.0, maxval=2.0 * np.pi)
        Omega = jnp.exp(1j * phases).astype(jnp.complex128)

        coeffs_j = jnp.asarray(coeffs, dtype=jnp.float64)
        A_all = _telescoping_filter(apply_H_block, Omega, coeffs_j,
                                     dos.center, dos.half_width, M_max)
        A_all = np.asarray(A_all)

        # Extract per-window filtered blocks Y_j = A[hi] - A[lo]
        for i, j in enumerate(cj_windows):
            idx_hi, idx_lo = win_pairs[i]
            Y_cj[j] = A_all[idx_hi] - A_all[idx_lo]  # (k, dim)

            # Store rescaled boundary positions for window indicator
            e_lo = (boundaries[j] - dos.center) / dos.half_width
            e_hi = (boundaries[j + 1] - dos.center) / dos.half_width
            b_lo = e_lo - delta_rescaled if j > 0 else e_lo
            b_hi = e_hi + delta_rescaled if j + 1 < len(boundaries) - 1 else e_hi
            cj_boundary_data[j] = (float(b_lo), float(b_hi))

        if verbose:
            print(f"  Filter done: {len(Y_cj)} CJ blocks")

    # ── §7-8: Per-window Ritz + Gauss ──
    Phi_dav_j = jnp.asarray(Phi_dav, dtype=jnp.complex128) if n_dav > 0 else None
    rng = jax.random.PRNGKey(seed + 200)

    Xi_list = []
    E_list = []
    w_list = []

    for j in range(N_S):
        e_lo, e_hi = boundaries[j], boundaries[j + 1]

        if is_dav[j]:
            # ── Davidson window ──
            idx = dav_indices[j]
            n_in = len(idx)
            rng, subkey = jax.random.split(rng)

            Xi_j, theta_ritz = _galerkin_ritz_dav(
                Phi_avail[idx], E_avail[idx], k, subkey)

            # For Davidson windows, we HAVE the exact eigenvalues — use them
            # directly rather than going through moment → Gauss reconstruction.
            # If n_in ≤ k: each eigenstate gets its own node with weight 1.
            # If n_in > k: Gauss quadrature compresses n_in poles into k nodes.
            if n_in <= k:
                nodes = np.full(k, 0.5 * (e_lo + e_hi))
                gauss_w = np.zeros(k)
                nodes[:n_in] = E_avail[idx]
                gauss_w[:n_in] = 1.0  # each eigenstate = 1 state
            elif n_in > 0:
                e_mid = 0.5 * (e_lo + e_hi)
                e_span = max(e_hi - e_lo, 1e-6)
                moments = _compute_window_moments_discrete(
                    E_avail[idx], 2 * k, E_shift=e_mid, E_scale=e_span)
                nodes, gauss_w = _gauss_from_moments(moments, k)
                nodes = nodes * e_span + e_mid  # unshift
                bad = np.isnan(nodes) | np.isnan(gauss_w)
                if np.any(bad):
                    nodes[bad] = e_mid
                    gauss_w[bad] = 0.0
            else:
                nodes = np.full(k, 0.5 * (e_lo + e_hi))
                gauss_w = np.zeros(k)

            mode = "dav"
        else:
            # ── CJ window ──
            Y_j_arr = jnp.asarray(Y_cj[j], dtype=jnp.complex128)
            Xi_j_jax, theta_ritz = _galerkin_ritz_cj(
                apply_H_block, Y_j_arr, Phi_dav_j)
            Xi_j = np.asarray(Xi_j_jax)
            theta_ritz = np.asarray(theta_ritz)

            # CJ window indicator for Gauss moments
            b_lo_r, b_hi_r = cj_boundary_data[j]
            w_E = _cj_window_on_grid(
                dos.E_grid, b_lo_r * dos.half_width + dos.center,
                b_hi_r * dos.half_width + dos.center,
                dos.center, dos.half_width, M_max)

            # n_eff from DOS integral over the CJ window indicator
            n_eff_j = float(np.trapezoid(w_E * np.maximum(dos.rho, 0), dos.E_grid)) * dim

            # Try Gauss quadrature from shifted moments
            e_mid = 0.5 * (e_lo + e_hi)
            e_span = max(e_hi - e_lo, 1e-6)
            moments = _compute_window_moments(
                dos.E_grid, dos.rho, w_E, 2 * k,
                E_shift=e_mid, E_scale=e_span)
            moments *= dim
            nodes, gauss_w = _gauss_from_moments(moments, k)
            nodes = nodes * e_span + e_mid

            # Use Ritz eigenvalues for energy placement, DOS for total weight.
            # Gauss quadrature is available but Ritz energies are more robust
            # for the GW pole structure (they come from the actual Hamiltonian).
            nodes = np.sort(np.real(theta_ritz))
            gauss_w = np.full(k, n_eff_j / k)
            mode = "CJ"

        # Sort Ritz and Gauss ascending, pair by order
        ritz_order = np.argsort(np.real(theta_ritz) if np.iscomplexobj(theta_ritz)
                                else theta_ritz)
        gauss_order = np.argsort(nodes)

        Xi_sorted = Xi_j[ritz_order]
        nodes_sorted = nodes[gauss_order]
        weights_sorted = np.sqrt(np.maximum(gauss_w[gauss_order], 0.0))

        # Universal rule: drop windows carrying less than half a state.
        # Zero coefficients + energies at window midpoint keeps downstream
        # cmin/vmax computations well-behaved (no phantom eigenvalues).
        n_eff_total = float(np.sum(gauss_w))
        if n_eff_total < 0.5:
            Xi_sorted = np.zeros_like(Xi_sorted)
            nodes_sorted = np.full(k, 0.5 * (e_lo + e_hi), dtype=np.float64)
            weights_sorted = np.zeros(k, dtype=np.float64)
            if verbose and (j < 5 or j == N_S - 1 or (j + 1) % 10 == 0):
                mode = "empty"

        # Absorb weight into wavefunction
        Xi_list.append(Xi_sorted * weights_sorted[:, None])
        E_list.append(nodes_sorted)
        w_list.append(weights_sorted)

        if verbose and (j < 5 or j == N_S - 1 or (j + 1) % 10 == 0):
            n_eff = float(np.sum(gauss_w))
            print(f"    w{j+1}/{N_S} [{mode:3s}]: [{e_lo:.2f},{e_hi:.2f}] "
                  f"n_eff={n_eff:.1f}, "
                  f"E=[{nodes_sorted[0]:.3f},{nodes_sorted[-1]:.3f}]")

    # ── Output assembly ──
    if Xi_list:
        Phi_pseudo = np.concatenate(Xi_list, axis=0)
        E_pseudo = np.concatenate(E_list)
        w_pseudo = np.concatenate(w_list)
    else:
        Phi_pseudo = np.zeros((0, dim), dtype=np.complex128)
        E_pseudo = np.array([], dtype=np.float64)
        w_pseudo = np.array([], dtype=np.float64)

    Phi_out = np.concatenate([Phi_protected, Phi_pseudo], axis=0)
    E_out = np.concatenate([E_protected, E_pseudo])
    weights = np.concatenate([np.ones(n_protected), w_pseudo])

    if verbose:
        n_pseudo_total = N_S * k
        total_eff = float(np.sum(w_pseudo**2))
        print(f"  Total: {n_protected} prot + {n_pseudo_total} pseudo "
              f"({n_dav_win * k} dav + {n_cj_win * k} CJ) "
              f"= {Phi_out.shape[0]} bands (Σ w² = {total_eff:.1f})")

    return PseudobandsResult(
        Phi_out=Phi_out, E_out=E_out, weights=weights,
        n_prot=n_protected, n_pseudo=N_S * k, n_windows=N_S,
        n_dav_windows=n_dav_win, n_cj_windows=n_cj_win,
        dos=dos, windows=boundaries,
    )

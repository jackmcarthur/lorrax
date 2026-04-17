"""
solvers/pseudobands.py — Hybrid pseudobands: stochastic + CJ-Ritz.

Three regions:
  1. Protected eigenstates (from Davidson): weight 1.0, passed through as-is.
  2. Stochastic pseudobands: random-phase linear combinations of exact
     eigenstates within each window. Used for windows where the CJ filter
     can't resolve (near the conduction edge / det band max).
  3. CJ-Ritz pseudobands: Chebyshev-Jackson filtered Ritz vectors for
     high-energy windows where the filter is reliable.

Physics-agnostic — works for H_DFT, H_BSE, or any Hermitian operator.

Usage
-----
    from solvers.pseudobands import ritz_pseudobands

    result = ritz_pseudobands(
        apply_H, dim=ngkmax,
        Phi_det=davidson_evecs, E_det=davidson_evals,
        E_fermi=0.0,
    )
    # result.Phi_out: (n_protected + N_S*k, dim) — protected + pseudobands
    # result.E_out:   matching eigenvalues/pseudo-energies
    # result.weights: 1.0 for protected, sqrt(n_eff/k) for pseudo
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from solvers.chebyshev import jackson_coefficients
from solvers.dos import (
    compute_dos, dos_weighted_windows, geometric_windows,
    compute_window_partition, DOSResult, WindowPartition,
)


# ═══════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PseudobandsResult:
    """Output of ritz_pseudobands."""
    Phi_out: np.ndarray     # (n_total, dim) — protected + pseudoband vectors
    E_out: np.ndarray       # (n_total,) — eigenvalues / pseudo-energies
    weights: np.ndarray     # (n_total,) — 1.0 for protected, sqrt(n/k) for pseudo
    n_det: int              # number of protected (deterministic) bands
    n_pseudo: int           # number of pseudobands
    n_windows: int          # number of spectral windows
    n_stochastic: int       # number of stochastic windows (from exact eigenstates)
    n_cj: int               # number of CJ-Ritz windows
    dos: DOSResult          # the KPM DOS used for partitioning
    windows: WindowPartition | None = None


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
#  Stochastic pseudobands from exact eigenstates
# ═══════════════════════════════════════════════════════════════════════

def _stochastic_pseudobands(
    Phi_window: np.ndarray,
    E_window: np.ndarray,
    k: int,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct k pseudobands from n exact eigenstates via random phases.

    Each pseudoband is a random linear combination:
        ξ_α = (1/√n) Σ_{i=1}^{n} exp(i·θ_{α,i}) · ψ_i

    Cross-window overlap cancels in expectation because eigenstates from
    different windows are orthogonal.

    Parameters
    ----------
    Phi_window : (n, dim) — exact eigenstates in this window
    E_window : (n,) — their eigenvalues
    k : number of pseudobands to construct
    key : JAX random key

    Returns
    -------
    Xi : (k, dim) — pseudoband vectors (unit norm)
    E_pseudo : (k,) — pseudo-energies (mean eigenvalue per pseudoband)
    """
    n, dim = Phi_window.shape
    # Random phases: each pseudoband gets n independent uniform phases
    phases = jax.random.uniform(key, (k, n), minval=0.0, maxval=2.0 * np.pi)
    coeffs = jnp.exp(1j * phases) / jnp.sqrt(n)  # (k, n)

    # Linear combination
    Xi = jnp.dot(coeffs, jnp.asarray(Phi_window, dtype=jnp.complex128))  # (k, dim)
    Xi = np.asarray(Xi)

    # Pseudo-energies: <ξ|H|ξ> = (1/n) Σ_i E_i (all cross-terms cancel)
    E_pseudo = np.full(k, np.mean(E_window))

    return Xi, E_pseudo


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
    n_kpm_moments: int = 500,
    n_kpm_random: int = 10,
    n_windows_target: int | None = None,
    dos_result: DOSResult | None = None,
    n_protected: int | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> PseudobandsResult:
    """Hybrid pseudobands: stochastic (low-energy) + CJ-Ritz (high-energy).

    Parameters
    ----------
    apply_H : (dim,) → (dim,) — Hermitian matvec (vmapped internally)
    dim : vector dimension
    Phi_det : (n_det, dim) — Davidson eigenvectors
    E_det : (n_det,) — their eigenvalues
    E_fermi : Fermi energy (default 0)
    F : window ratio (crossover energy formula)
    k : block size per window (pseudobands per window)
    M_max : Chebyshev order cap
    C_m : filter sharpness constant
    n_kpm_moments : KPM moment count for DOS
    n_kpm_random : KPM trace probes
    n_windows_target : target number of windows (sets tau by bisection)
    dos_result : pre-computed DOSResult (skips KPM if provided)
    n_protected : fixed count of protected det bands (for WFN consistency
        across k-points). If None, determined from eigenvalues at this k.
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

    apply_H_block = jax.vmap(apply_H)

    # ── §2: KPM DOS ──
    if dos_result is None:
        if verbose:
            print("Pseudobands: computing KPM DOS...")
        dos = compute_dos(apply_H, dim, n_moments=n_kpm_moments,
                          n_random=n_kpm_random, seed=seed, verbose=verbose)
    else:
        dos = dos_result

    B = 2.0 * dos.half_width
    E_max = dos.E_max

    # ── §3: Window partition ──
    cj_resolution = np.pi * B / M_max
    eps_cross = C_m * cj_resolution / F
    if verbose:
        print(f"  CJ resolution: π·B/M = {cj_resolution:.4f} Ry")
        print(f"  Crossover energy: ε_cross = {eps_cross:.4f} "
              f"(E_F + ε_cross = {E_fermi + eps_cross:.4f})")

    E_cross_abs = E_fermi + eps_cross
    E_max_abs = E_max

    if n_windows_target is not None:
        boundaries = dos_weighted_windows(
            dos.E_grid, dos.rho, E_cross_abs, E_max_abs,
            galerkin_order=1, n_windows_target=n_windows_target)
    else:
        boundaries = geometric_windows(eps_cross, E_max - E_fermi, F)
        boundaries = boundaries + E_fermi

    N_S = len(boundaries) - 1

    # ── Classify det bands: protected vs available for stochastic ──
    # Protected: det bands below the first window boundary (kept as-is)
    # Available: det bands within the window region (used for stochastic PB)
    E_window_start = boundaries[0]
    protected_mask = E_det < E_window_start
    n_prot_auto = int(np.sum(protected_mask))

    if n_protected is None:
        n_protected = n_prot_auto
    else:
        # Use the fixed count. Sort by energy and take the first n_protected
        # as protected, rest as available — ensures consistent band count
        # across k-points even when eigenvalue distributions differ.
        sort_idx = np.argsort(E_det)
        protected_mask = np.zeros(n_det, dtype=bool)
        protected_mask[sort_idx[:n_protected]] = True

    Phi_protected = Phi_det[protected_mask]
    E_protected = E_det[protected_mask]

    available_mask = ~protected_mask
    Phi_available = Phi_det[available_mask]
    E_available = E_det[available_mask]

    if verbose:
        print(f"  Det bands: {n_protected} protected (E < {E_window_start:.4f}), "
              f"{int(np.sum(available_mask))} available for stochastic")
        print(f"  {N_S} spectral windows from {boundaries[0]:.4f} to {boundaries[-1]:.4f}")
        widths = np.diff(boundaries)
        print(f"  Window widths: min={widths.min():.4f}, max={widths.max():.4f}, "
              f"first 3: [{', '.join(f'{w:.4f}' for w in widths[:3])}]")

    # ── Classify windows: stochastic vs CJ ──
    # Stochastic: window has ≥1 det eigenstate → random-phase construction.
    # CJ-Ritz: no det eigenstates → use Chebyshev filter.
    # Works even when n_det_in_window < k: the k pseudobands are random
    # combinations of n_states eigenstates, with weight sqrt(n_states/k).
    window_is_stochastic = np.zeros(N_S, dtype=bool)
    window_det_indices = []
    for j in range(N_S):
        in_win = (E_available >= boundaries[j]) & (E_available < boundaries[j + 1])
        indices = np.where(in_win)[0]
        window_det_indices.append(indices)
        if len(indices) >= 1:
            window_is_stochastic[j] = True

    n_stochastic = int(np.sum(window_is_stochastic))
    n_cj = N_S - n_stochastic

    if verbose:
        print(f"  Window classification: {n_stochastic} stochastic, {n_cj} CJ-Ritz")

    # ── Stochastic windows: build from exact eigenstates ──
    # ── CJ windows: need the telescoping filter ──

    # Only run the CJ filter if there are CJ windows
    cj_indices = np.where(~window_is_stochastic)[0]
    Y_cj = None
    if n_cj > 0:
        # Build CJ filter for the CJ windows only.
        # The telescoping filter computes ALL N_S windows at once (they share
        # the Chebyshev recurrence), so we compute them all and pick the CJ ones.
        tilde_eps = (boundaries - dos.center) / dos.half_width
        coeffs = _boundary_coefficients(tilde_eps, M_max)

        if verbose:
            print(f"  Running {M_max} Chebyshev iterations (block size {k})...")
        key = jax.random.PRNGKey(seed + 42)
        Omega = jax.random.normal(key, (k, dim), dtype=jnp.float64)
        Omega = Omega + 0j

        coeffs_j = jnp.asarray(coeffs, dtype=jnp.float64)
        Y_all = _telescoping_filter(apply_H_block, Omega, coeffs_j,
                                     dos.center, dos.half_width, M_max)
        Y_all = np.asarray(Y_all)  # (N_S, k, dim)

        if verbose:
            print(f"  Filtered blocks: {Y_all.shape}")

    # ── Window weights from DOS ──
    win_part = compute_window_partition(dos, boundaries)
    n_eff = win_part.n_eff * dim

    # ── Per-window construction ──
    # ALL det bands are used for CJ deflation (both protected and available)
    Phi_all_det_j = jnp.asarray(Phi_det, dtype=jnp.complex128) if n_det > 0 else None

    Xi_list = []
    theta_list = []
    weight_list = []
    Q_prev = None
    rng = jax.random.PRNGKey(seed + 100)

    for j in range(N_S):
        e_lo, e_hi = boundaries[j], boundaries[j + 1]
        mid = 0.5 * (e_lo + e_hi)

        if window_is_stochastic[j]:
            # ── Stochastic: random-phase combination of exact eigenstates ──
            idx = window_det_indices[j]
            rng, subkey = jax.random.split(rng)
            Xi_np, theta_np = _stochastic_pseudobands(
                Phi_available[idx], E_available[idx], k, subkey)
            n_states = len(idx)
            w_j = np.sqrt(n_states / k)
            mode = "stoch"
        else:
            # ── CJ-Ritz ──
            Y_j = jnp.asarray(Y_all[j], dtype=jnp.complex128)
            Xi_j, theta_j, Q_j = _galerkin_ritz(
                apply_H_block, Y_j, Phi_all_det_j, Q_prev)
            Xi_np = np.asarray(Xi_j)
            theta_np = np.asarray(theta_j)
            outside = (theta_np < e_lo - cj_resolution) | (theta_np > e_hi + cj_resolution)
            if np.any(outside):
                w_j = 0.0
                mode = "CJ-0"
            else:
                w_j = np.sqrt(max(n_eff[j], 0.0) / k)
                mode = "CJ"
            Q_prev = Q_j

        # Universal rule: drop windows carrying less than half a state.
        # Covers CJ-Ritz leak (CJ-0) and any stochastic/CJ window whose
        # DOS integral is below threshold.  Zero coefficients + energy
        # at window midpoint keeps the WFN physically consistent and
        # prevents the stored θ from corrupting cmin downstream.
        if n_eff[j] < 0.5 or w_j == 0.0:
            Xi_np = np.zeros_like(Xi_np)
            theta_np = np.full(k, mid, dtype=np.float64)
            w_j = 0.0
            if mode != "CJ-0":
                mode = "empty"

        Xi_list.append(Xi_np * w_j)
        theta_list.append(theta_np)
        weight_list.append(np.full(k, w_j))

        if verbose and (j < 5 or j == N_S - 1 or (j + 1) % 10 == 0):
            print(f"    window {j+1}/{N_S} [{mode:5s}]: [{e_lo:.2f},{e_hi:.2f}] "
                  f"n_eff={n_eff[j]:.1f}, w={w_j:.3f}, "
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

    # Protected det bands + all pseudobands (stochastic + CJ)
    Phi_out = np.concatenate([Phi_protected, Phi_pseudo], axis=0)
    E_out = np.concatenate([E_protected, E_pseudo])
    weights = np.concatenate([np.ones(n_protected), w_pseudo])

    if verbose:
        total_eff = float(np.sum(n_eff))
        n_pseudo_total = N_S * k
        print(f"  Total: {n_protected} protected + {n_pseudo_total} pseudo "
              f"({n_stochastic * k} stochastic + {n_cj * k} CJ) "
              f"= {Phi_out.shape[0]} bands (Σ n_eff = {total_eff:.1f})")

    return PseudobandsResult(
        Phi_out=Phi_out, E_out=E_out, weights=weights,
        n_det=n_protected, n_pseudo=N_S * k, n_windows=N_S,
        n_stochastic=n_stochastic, n_cj=n_cj,
        dos=dos, windows=win_part,
    )

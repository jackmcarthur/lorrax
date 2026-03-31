"""
Non-crossing minimax for imaginary-frequency denominators.

Fits x/(x^2 + omega_p^2) on [1, R] with decaying exponentials:
  x/(x^2 + omega_p^2) ≈ sum_l w_l exp(-t_l x)

This is the kernel needed for chi^0(i*omega_p) in CTSP:
  1/(E - i*omega_p) + 1/(E + i*omega_p) = 2E/(E^2 + omega_p^2)

The function is smooth, positive, and decays as 1/x for x >> omega_p,
so the same exponential-sum basis as the static case works.

Usage:
  t, w, err = solve_noncrossing_imag(N, R, omega_hat)

where omega_hat = omega_p / E_gap is the dimensionless imaginary frequency.
Physical: sum w_l exp(-t_l E / E_gap) ≈ E/(E^2+omega_p^2) / E_gap on [E_gap, E_bw].

Integrates with minimax.py by reusing the same VarPro + Lawson machinery.
"""

import warnings
import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def _target(x, omega_hat):
    """x/(x^2 + omega_hat^2) on the evaluation grid."""
    return x / (x**2 + omega_hat**2)


# ---------------------------------------------------------------------------
# VarPro internals (same structure as minimax._nc_*)
# ---------------------------------------------------------------------------

def _varpro_residual(s, x_grid, g, W_sqrt):
    """VarPro residual: (I - UU^T)(W * g)."""
    t = np.exp(s)
    Phi = np.exp(-np.outer(x_grid, t)) * W_sqrt[:, None]
    g_w = g * W_sqrt
    U, _, _ = np.linalg.svd(Phi, full_matrices=False)
    return g_w - U @ (U.T @ g_w)


def _solve_once(s_init, x_grid, g, s_lo, s_hi, weights=None):
    """One VarPro solve (scipy TRF)."""
    M = len(x_grid)
    W_sqrt = np.sqrt(weights) if weights is not None else np.ones(M)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = least_squares(
            _varpro_residual, s_init,
            args=(x_grid, g, W_sqrt),
            method='trf',
            bounds=(np.full_like(s_init, s_lo), np.full_like(s_init, s_hi)),
            ftol=1e-14, xtol=1e-14, gtol=1e-14,
            max_nfev=200 * len(s_init),
        )
    s = res.x

    # Recover linear weights
    t = np.exp(s)
    Phi = np.exp(-np.outer(x_grid, t))
    if weights is not None:
        Phi_w = Phi * W_sqrt[:, None]
        g_w = g * W_sqrt
    else:
        Phi_w = Phi
        g_w = g
    w = np.linalg.lstsq(Phi_w, g_w, rcond=None)[0]
    return s, w


def _solve_at_R(N, Ri, omega_hat, s_init, lawson_iter=4):
    """Solve at a single R with Lawson IRLS."""
    M = max(200, 15 * N)
    x_grid = np.exp(np.linspace(0, np.log(Ri), M))
    g = _target(x_grid, omega_hat)

    s_lo = -np.log(2.0 * Ri) - 1.0
    s_hi = np.log(max(5.0 * N, 10.0)) + 1.0
    s_init = np.clip(s_init, s_lo, s_hi)

    # L2 fit
    s, w = _solve_once(s_init, x_grid, g, s_lo, s_hi)

    # Lawson IRLS
    for k in range(lawson_iter):
        Phi = np.exp(-np.outer(x_grid, np.exp(s)))
        e = g - Phi @ w
        ae = np.abs(e)
        delta = max(1e-2 * np.max(ae), 1e-30)
        irls_w = 1.0 / np.maximum(ae, delta)
        irls_w /= np.sum(irls_w)
        s, w = _solve_once(s, x_grid, g, s_lo, s_hi, weights=irls_w)

    return s, w


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_noncrossing_imag(N, R, omega_hat, lawson_iter=4):
    """Compute N-point minimax quadrature for x/(x^2+omega_hat^2) on [1, R].

    Uses continuation in R from R=2, same as the static solver.

    Parameters
    ----------
    N : int
        Number of quadrature points.
    R : float
        Dynamic range x_max/x_min (>= 2).
    omega_hat : float
        Dimensionless imaginary frequency omega_p / x_min.
        omega_hat = 0 recovers the static 1/x case (up to normalization).
    lawson_iter : int
        Lawson IRLS passes per R step.

    Returns
    -------
    t : ndarray (N,)
        Exponents (sorted ascending).
    w : ndarray (N,)
        Weights.
    err : float
        L-infinity error on [1, R].
    """
    M_eval = max(1000, 40 * N)
    x_eval = np.exp(np.linspace(0, np.log(R), M_eval))
    g_eval = _target(x_eval, omega_hat)

    s_lo = -np.log(2.0 * R) - 1.0
    s_hi = np.log(max(5.0 * N, 10.0)) + 1.0

    def _try_init(s0):
        # Continuation: sweep R from modest start
        R_start = max(2.0, min(np.exp(0.7 * N), R))
        R_sched = []
        r_ = R_start
        while r_ < R:
            R_sched.append(r_)
            r_ = min(r_ * 4.0, R)
        R_sched.append(R)

        s_ = s0.copy()
        for Ri in R_sched:
            s_, w_ = _solve_at_R(N, Ri, omega_hat, s_, lawson_iter=lawson_iter)

        t_ = np.exp(s_)
        approx = np.exp(-np.outer(x_eval, t_)) @ w_
        err_ = np.max(np.abs(g_eval - approx))
        return s_, w_, err_

    # Hackbusch init (works well even for modified target)
    s_hack = np.log(np.pi**2 * (np.arange(1, N + 1) - 0.5)
                    / (2.0 * np.log(4.0 * R)))
    best_s, best_w, best_err = _try_init(s_hack)

    # Log-uniform init (direct, no continuation)
    s_unif = np.linspace(s_lo + 0.5, s_hi - 0.5, N)
    s2, w2 = _solve_at_R(N, R, omega_hat, s_unif, lawson_iter=lawson_iter)
    t2 = np.exp(s2)
    err2 = np.max(np.abs(g_eval - np.exp(-np.outer(x_eval, t2)) @ w2))
    if err2 < best_err:
        best_s, best_w, best_err = s2, w2, err2

    t = np.exp(best_s)
    order = np.argsort(t)
    return t[order], best_w[order], best_err


def noncrossing_imag_grids(R, omega_hat, eps, N_start=2, N_max=60):
    """Find minimum N achieving error < eps.

    Returns t, w, N, err.
    """
    for N in range(N_start, N_max + 1):
        t, w, err = solve_noncrossing_imag(N, R, omega_hat)
        if err < eps:
            return t, w, N, err
    return t, w, N_max, err


def evaluate_noncrossing_imag(x, t, w):
    """Evaluate sum_l w_l exp(-t_l x)."""
    x = np.asarray(x)
    return np.exp(-np.outer(x, t)) @ w


def rescale_noncrossing_imag(t, w, E_gap):
    """Rescale from [1, R] to physical units [E_gap, E_bw].

    Physical: sum w_phys exp(-t_phys E) ≈ E/(E^2+omega_p^2)
    where omega_p = omega_hat * E_gap.

    t_phys = t / E_gap,  w_phys = w / E_gap.
    """
    return t / E_gap, w / E_gap


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*65)
    print("Non-crossing minimax for x/(x^2 + omega_hat^2) on [1, R]")
    print("="*65)

    R = 52.0  # from the example: E_bw/E_gap = 6.39/0.123

    # Verify static limit: omega_hat=0 should recover 1/x behavior
    print(f"\nR = {R:.1f}")
    print(f"\n--- Static limit (omega_hat → 0, target ≈ 1/x) ---")
    for N in [5, 7, 9, 11]:
        t, w, err = solve_noncrossing_imag(N, R, omega_hat=0.01)
        print(f"  N={N:2d}: err={err:.2e}")

    # The physical case from the other Claude's example
    print(f"\n--- omega_hat = 16.3 (omega_p=2.0 Ry, E_gap=0.123 Ry) ---")
    for N in [5, 7, 9, 11, 13, 15]:
        t, w, err = solve_noncrossing_imag(N, R, omega_hat=16.3)
        print(f"  N={N:2d}: err={err:.2e}")

    # Verify accuracy at a few x values
    print(f"\n--- Pointwise check at N=11, omega_hat=16.3 ---")
    t, w, err = solve_noncrossing_imag(11, R, omega_hat=16.3)
    for x in [1.0, 2.0, 5.0, 10.0, 20.0, 52.0]:
        exact = x / (x**2 + 16.3**2)
        approx = float(np.exp(-x * t) @ w)
        print(f"  x={x:5.1f}: exact={exact:.6e}, approx={approx:.6e}, "
              f"err={abs(exact-approx):.2e}")

    # Sweep omega_hat
    print(f"\n--- Node count at eps=1e-3 for various omega_hat ---")
    print(f"{'omega_hat':>10s} {'N':>4s} {'err':>10s}")
    for omega_hat in [0.01, 0.1, 1.0, 5.0, 10.0, 16.3, 50.0, 100.0]:
        t, w, N, err = noncrossing_imag_grids(R, omega_hat, eps=1e-3)
        print(f"{omega_hat:10.2f} {N:4d} {err:10.2e}")
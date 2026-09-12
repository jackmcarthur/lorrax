"""
Minimax quadrature for GW self-energy frequency integration.

Three solver families for energy denominators:

Non-crossing (x > 0):
    1/x ≈ Σ w_l exp(-t_l x) on [1, R]
    O(ln R) nodes.  Error: ε ≈ 0.31·exp[-N(3.55/ln R + 0.68)]

Crossing (x changes sign, regularized):
    G(u) ≈ Σ w_l sin(τ_l u) on [0, A]
    O(A) nodes.  Error: ε ≈ exp(-0.93 - 14.25·N/A)

Imaginary-axis:
    x/(x²+ω²) ≈ Σ w_l exp(-t_l x) on [1, R]
    Same VarPro+Lawson as non-crossing with modified target.

References:
  Hackbusch, Comput. Vis. Sci. 21, 1 (2019)
  Helmich-Paris & Visscher, J. Comput. Phys. 321, 927 (2016)
  Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)
  Golub & Pereyra, SIAM J. Numer. Anal. 10, 413 (1973)
"""

import warnings
import numpy as np
from scipy.special import wofz
from scipy.optimize import least_squares, linprog, brentq


# ================================================================
# Target functions for crossing regularizations
# ================================================================

def G_hgl(u):
    """HGL target: Im[sqrt(pi/2) * exp(-(u+i)^2/2) * (1 + i*erfi((u+i)/sqrt2))].

    Rewritten via Faddeeva w(z) for numerical stability:
      sqrt(pi/2) * (2*exp(-z^2/2) - w(-z/sqrt2)),  z = u + i.
    """
    u = np.asarray(u, dtype=float)
    z = u + 1j
    val = np.sqrt(np.pi / 2.0) * (2.0 * np.exp(-z**2 / 2.0) - wofz(-z / np.sqrt(2.0)))
    result = np.imag(val)
    result = np.where(np.abs(u) < 1e-30, 0.0, result)
    return result


def G_fermi(u):
    """Fermi target: 1/u - pi/(2 sinh(pi u/2))."""
    u = np.asarray(u, dtype=float)
    safe = np.where(np.abs(u) < 1e-14, 1.0, u)
    val = 1.0 / safe - np.pi / (2.0 * np.sinh(np.pi * safe / 2.0))
    val = np.where(np.abs(u) < 1e-14, 0.0, val)
    return val


def tau_max_hgl(eps_q):
    """Effective support for HGL weight: sqrt(2 ln(1/eps_q))."""
    return np.sqrt(2.0 * np.log(1.0 / eps_q))


def tau_max_fermi(eps_q):
    """Effective support for Fermi weight: 0.5 ln(1/eps_q)."""
    return 0.5 * np.log(1.0 / eps_q)


# ================================================================
# NON-CROSSING: 1/x on [1, R] with exponential sums
# ================================================================

# --- Empirical error scaling (R^2 = 0.995 on eps in [1e-5, 1e-2]) ---
_NC_LN_C = np.log(0.3112)
_NC_A = 3.5456
_NC_B = 0.6845






# --- VarPro + Lawson: the warm start for the Remez solver below ---











# --- Remez exchange solver (enhanced, standalone use) ---





















# ================================================================
# IMAGINARY-AXIS: x/(x²+ω²) on [1, R] with exponential sums
# ================================================================

def _imag_target(x, omega_hat):
    """x/(x^2 + omega_hat^2) on the evaluation grid."""
    return x / (x**2 + omega_hat**2)


def _imag_varpro_residual(s, x_grid, g, W_sqrt):
    """VarPro residual: (I - UU^T)(W * g)."""
    t = np.exp(s)
    Phi = np.exp(-np.outer(x_grid, t)) * W_sqrt[:, None]
    g_w = g * W_sqrt
    U, _, _ = np.linalg.svd(Phi, full_matrices=False)
    return g_w - U @ (U.T @ g_w)


def _imag_solve_once(s_init, x_grid, g, s_lo, s_hi, weights=None):
    """One VarPro solve (scipy TRF)."""
    M = len(x_grid)
    W_sqrt = np.sqrt(weights) if weights is not None else np.ones(M)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = least_squares(
            _imag_varpro_residual, s_init,
            args=(x_grid, g, W_sqrt),
            method='trf',
            bounds=(np.full_like(s_init, s_lo), np.full_like(s_init, s_hi)),
            ftol=1e-14, xtol=1e-14, gtol=1e-14,
            max_nfev=200 * len(s_init),
        )
    s = res.x

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


def _imag_solve_at_R(N, Ri, omega_hat, s_init, lawson_iter=4):
    """Solve at a single R with Lawson IRLS."""
    M = max(200, 15 * N)
    x_grid = np.exp(np.linspace(0, np.log(Ri), M))
    g = _imag_target(x_grid, omega_hat)

    s_lo = -np.log(2.0 * Ri) - 1.0
    s_hi = np.log(max(5.0 * N, 10.0)) + 1.0
    s_init = np.clip(s_init, s_lo, s_hi)

    s, w = _imag_solve_once(s_init, x_grid, g, s_lo, s_hi)

    for k in range(lawson_iter):
        Phi = np.exp(-np.outer(x_grid, np.exp(s)))
        e = g - Phi @ w
        ae = np.abs(e)
        delta = max(1e-2 * np.max(ae), 1e-30)
        irls_w = 1.0 / np.maximum(ae, delta)
        irls_w /= np.sum(irls_w)
        s, w = _imag_solve_once(s, x_grid, g, s_lo, s_hi, weights=irls_w)

    return s, w


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

    Returns
    -------
    t : ndarray (N,)
    w : ndarray (N,)
    err : float
    """
    M_eval = max(1000, 40 * N)
    x_eval = np.exp(np.linspace(0, np.log(R), M_eval))
    g_eval = _imag_target(x_eval, omega_hat)

    s_lo = -np.log(2.0 * R) - 1.0
    s_hi = np.log(max(5.0 * N, 10.0)) + 1.0

    def _try_init(s0):
        R_start = max(2.0, min(np.exp(0.7 * N), R))
        R_sched = []
        r_ = R_start
        while r_ < R:
            R_sched.append(r_)
            r_ = min(r_ * 4.0, R)
        R_sched.append(R)

        s_ = s0.copy()
        for Ri in R_sched:
            s_, w_ = _imag_solve_at_R(N, Ri, omega_hat, s_, lawson_iter=lawson_iter)

        t_ = np.exp(s_)
        approx = np.exp(-np.outer(x_eval, t_)) @ w_
        err_ = np.max(np.abs(g_eval - approx))
        return s_, w_, err_

    s_hack = np.log(np.pi**2 * (np.arange(1, N + 1) - 0.5)
                    / (2.0 * np.log(4.0 * R)))
    best_s, best_w, best_err = _try_init(s_hack)

    s_unif = np.linspace(s_lo + 0.5, s_hi - 0.5, N)
    s2, w2 = _imag_solve_at_R(N, R, omega_hat, s_unif, lawson_iter=lawson_iter)
    t2 = np.exp(s2)
    err2 = np.max(np.abs(g_eval - np.exp(-np.outer(x_eval, t2)) @ w2))
    if err2 < best_err:
        best_s, best_w, best_err = s2, w2, err2

    t = np.exp(best_s)
    order = np.argsort(t)
    return t[order], best_w[order], best_err


def evaluate_noncrossing_imag(x, t, w):
    """Evaluate sum_l w_l exp(-t_l x)."""
    x = np.asarray(x)
    return np.exp(-np.outer(x, t)) @ w


def noncrossing_imag_grids(R, omega_hat, eps, N_start=2, N_max=60):
    """Find minimum N achieving error < eps.

    Returns t, w, N, err.
    """
    for N in range(N_start, N_max + 1):
        t, w, err = solve_noncrossing_imag(N, R, omega_hat)
        if err < eps:
            return t, w, N, err
    return t, w, N_max, err




# ================================================================
# CROSSING: sin-sum regularizations
# ================================================================

# --- VarPro-LM + LP solver ---

def _cr_varpro_lm(tau, u_grid, g, max_iter=120, tol=1e-14, weights=None,
                  tau_hi=None):
    """VarPro-LM for sin-basis crossing problem (Phase 2 polish)."""
    N = len(tau)
    M = len(u_grid)
    tau = tau.copy()

    W = np.sqrt(weights) if weights is not None else np.ones(M)
    g_w = g * W

    def _eval(tau_):
        tau_s = np.maximum(tau_, 1e-10)
        Phi = np.sin(np.outer(u_grid, tau_s)) * W[:, None]
        U, sig, Vt = np.linalg.svd(Phi, full_matrices=False)
        keep = sig > 1e-14 * max(sig[0], 1e-30)
        sig_inv = np.divide(1.0, sig, out=np.zeros_like(sig), where=keep)
        w_ = Vt.T @ (sig_inv * (U.T @ g_w))
        r_ = g_w - U @ (U.T @ g_w)
        return w_, r_, np.dot(r_, r_), U

    w_lin, r, cost, U = _eval(tau)
    mu = 1e-5
    stall = 0

    for it in range(max_iter):
        tau_s = np.maximum(tau, 1e-10)
        dPhi_w = u_grid[:, None] * np.cos(np.outer(u_grid, tau_s)) * (w_lin[None, :] * W[:, None])
        J = -(dPhi_w - U @ (U.T @ dPhi_w))

        JtJ = J.T @ J
        Jtr = J.T @ r
        diag_d = np.diag(JtJ).copy()
        diag_d[diag_d < 1e-20] = 1e-20

        try:
            dtau = np.linalg.solve(JtJ + mu * np.diag(diag_d), -Jtr)
        except np.linalg.LinAlgError:
            dtau = np.linalg.lstsq(JtJ + mu * np.diag(diag_d), -Jtr,
                                   rcond=None)[0]

        ub = tau_hi if tau_hi is not None else 1e30
        tau_new = np.sort(np.clip(tau + dtau, 1e-10, ub))
        w_new, r_new, cost_new, U_new = _eval(tau_new)

        if cost_new < cost:
            stall = 0 if cost - cost_new > 1e-14 * cost else stall + 1
            tau, w_lin, r, cost, U = tau_new, w_new, r_new, cost_new, U_new
            mu = max(mu * 0.3, 1e-15)
        else:
            mu = min(mu * 5.0, 1e8)
            stall += 1

        if np.linalg.norm(dtau) / max(np.linalg.norm(tau), 1e-30) < tol \
                or stall >= 10:
            break

    return tau, w_lin


def _cr_minimax_lp(Phi, g, method='highs-ipm'):
    """Solve the minimax LP:  min t  s.t.  |g - Phi @ w| <= t."""
    M, K = Phi.shape

    c = np.zeros(1 + 2 * K)
    c[0] = 1.0
    c[1:K + 1] = 1e-12
    c[K + 1:] = 1e-12

    A_ub = np.zeros((2 * M, 1 + 2 * K))
    A_ub[:M, 0] = -1.0
    A_ub[:M, 1:K + 1] = Phi
    A_ub[:M, K + 1:] = -Phi
    A_ub[M:, 0] = -1.0
    A_ub[M:, 1:K + 1] = -Phi
    A_ub[M:, K + 1:] = Phi
    b_ub = np.concatenate([g, -g])
    bounds = [(0, None)] * (1 + 2 * K)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method=method)

    if not result.success:
        return None, np.inf

    t_opt = result.x[0]
    w = result.x[1:K + 1] - result.x[K + 1:]
    return w, t_opt


def _cr_lp_backward_elim(N, A, G_func, tau_max_val):
    """LP with backward elimination to select exactly N frequencies."""
    eta = 3.0
    tau_upper = tau_max_val * 1.3
    K = max(30, int(np.ceil(eta * A * tau_upper / np.pi)) + 20)
    K = min(K, 500)

    M = max(600, 15 * N)
    u_grid = np.linspace(0, A, M)
    g = G_func(u_grid)

    tau_candidates = np.linspace(0.01, tau_upper, K)
    Phi_full = np.sin(np.outer(u_grid, tau_candidates))

    w_full, t_full = _cr_minimax_lp(Phi_full, g)
    if w_full is None:
        return None, None, None

    amp = np.abs(w_full)
    S_size = min(3 * N, K)
    S_idx = np.argsort(amp)[-S_size:]
    S_idx = np.sort(S_idx)

    while len(S_idx) > N:
        Phi_S = Phi_full[:, S_idx]
        w_S, _ = _cr_minimax_lp(Phi_S, g)
        if w_S is None:
            amp_S = np.abs(w_full[S_idx])
            keep = np.argsort(amp_S)[-N:]
            S_idx = np.sort(S_idx[keep])
            break
        drop = np.argmin(np.abs(w_S))
        S_idx = np.delete(S_idx, drop)

    Phi_N = Phi_full[:, S_idx]
    w_N, t_N = _cr_minimax_lp(Phi_N, g)
    if w_N is None:
        w_N = np.linalg.lstsq(Phi_N, g, rcond=None)[0]

    return tau_candidates[S_idx], w_N, u_grid


def _cr_final_lp_weights(tau, u_eval, g_eval):
    """Final minimax LP for weights with tau fixed."""
    Phi = np.sin(np.outer(u_eval, tau))
    w, t = _cr_minimax_lp(Phi, g_eval)
    if w is not None:
        return w, t
    return np.linalg.lstsq(Phi, g_eval, rcond=None)[0], np.inf


def solve_crossing(N, A, G_func, tau_max_val, lawson_iter=5):
    """Compute N-point minimax quadrature for crossing regularization.

    Architecture:
      1. LP on adaptive candidate grid -> backward elimination to N freqs
      2. VarPro-LM + Lawson polish to refine continuous frequencies
      3. Final minimax LP for optimal weights at the polished frequencies

    Returns
    -------
    tau, w, err
    """
    M_eval = max(1000, 30 * N)
    u_eval = np.linspace(0, A, M_eval)
    g_eval = G_func(u_eval)
    tau_hi = tau_max_val * 1.5

    def _eval_err(tau_, w_):
        return np.max(np.abs(g_eval - np.sin(np.outer(u_eval, tau_)) @ w_))

    def _polish(tau_init):
        M = max(500, 20 * N)
        u_grid = np.linspace(0, A, M)
        g = G_func(u_grid)

        tau = np.clip(tau_init.copy(), 1e-10, tau_hi)
        tau, w = _cr_varpro_lm(tau, u_grid, g, tau_hi=tau_hi)

        s_law = np.full(M, 1.0 / M)
        for k in range(lawson_iter):
            Phi = np.sin(np.outer(u_grid, np.maximum(tau, 1e-10)))
            ae = np.abs(g - Phi @ w)
            # Lawson's L-infinity update: weights grow where the error is
            # large.  The former 1/max(|e|, 1e-2 max|e|) is the L1 IRLS weight
            # and moves the polish away from the sup-norm optimum.
            s_law = np.maximum(s_law * ae / max(np.sum(s_law * ae), 1e-300),
                               1e-14)
            tau, w = _cr_varpro_lm(tau, u_grid, g, weights=s_law,
                                   tau_hi=tau_hi)

        tau = np.sort(np.maximum(tau, 1e-10))

        w_lp, _ = _cr_final_lp_weights(tau, u_eval, g_eval)
        err_lp = _eval_err(tau, w_lp)
        err_vp = _eval_err(tau, w)
        if err_lp < err_vp:
            w = w_lp

        return tau, w, min(err_lp, err_vp)

    best_tau, best_w, best_err = None, None, np.inf

    def _update(tau_, w_, err_):
        nonlocal best_tau, best_w, best_err
        if err_ < best_err:
            best_tau, best_w, best_err = tau_, w_, err_

    tau_lp, w_lp, _ = _cr_lp_backward_elim(N, A, G_func, tau_max_val)
    if tau_lp is not None:
        _update(*_polish(tau_lp))

    tau_support = np.linspace(0.1, tau_max_val * 1.1, N)
    _update(*_polish(tau_support))

    k = np.arange(1, N + 1)
    tau_cheb = tau_max_val * 0.5 * (1 - np.cos(np.pi * k / (N + 1)))
    _update(*_polish(tau_cheb))

    order = np.argsort(best_tau)
    return best_tau[order], best_w[order], best_err


def crossing_grids(A, eps, G_func, tau_max_func, eps_q=1e-3, N_max=500):
    """Find minimum N achieving error < eps for crossing regularization.

    Returns
    -------
    tau, w, N, err
    """
    tmx = tau_max_func(eps_q)
    N_est = max(2, int(np.ceil(A * tmx / np.pi)))
    N_lo = max(2, N_est - 5)

    for N in range(N_lo, N_max + 1):
        tau, w, err = solve_crossing(N, A, G_func, tmx,
                                     lawson_iter=5)
        if err < eps:
            return tau, w, N, err

    return tau, w, N_max, err


# --- Higher-level crossing builder (binary-search for xi_eff) ---

_CR_INTERCEPT = -0.93
_CR_SLOPE = -14.25
_CR_TAU_MAX = np.sqrt(2 * np.log(1e3))
















# ================================================================
# Physical rescaling helpers
# ================================================================






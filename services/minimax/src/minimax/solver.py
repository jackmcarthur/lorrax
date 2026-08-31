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


def predict_N_noncrossing(R, target_error):
    """Predict number of exponential-sum nodes for 1/x on [1, R]."""
    rate = _NC_A / np.log(R) + _NC_B
    N = (np.log(target_error) - _NC_LN_C) / (-rate)
    return max(int(np.ceil(N)), 2)


def error_estimate_noncrossing(N, R):
    """Estimate L-infinity error for N nodes at dynamic range R."""
    rate = _NC_A / np.log(R) + _NC_B
    return np.exp(_NC_LN_C - N * rate)


# --- VarPro + Lawson solver (used by noncrossing_grids and as warm-start) ---

def _nc_varpro_residual(s, x_grid, g, W_sqrt):
    """VarPro residual for non-crossing: (I - UU^T)(W * g)."""
    t = np.exp(s)
    Phi = np.exp(-np.outer(x_grid, t)) * W_sqrt[:, None]
    g_w = g * W_sqrt
    U, sig, Vt = np.linalg.svd(Phi, full_matrices=False)
    return g_w - U @ (U.T @ g_w)


def _nc_solve_once(s_init, x_grid, g, s_lo, s_hi, weights=None):
    """One VarPro solve for non-crossing (scipy TRF)."""
    M = len(x_grid)
    W_sqrt = np.sqrt(weights) if weights is not None else np.ones(M)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = least_squares(
            _nc_varpro_residual, s_init,
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


def _nc_solve_at_R(N, Ri, s_init, lawson_iter=4):
    """Solve the non-crossing problem at a single R, given an initial s."""
    M = max(200, 15 * N)
    x_grid = np.exp(np.linspace(0, np.log(Ri), M))
    g = 1.0 / x_grid

    s_lo = -np.log(2.0 * Ri) - 1.0
    s_hi = np.log(max(5.0 * N, 10.0)) + 1.0
    s_init = np.clip(s_init, s_lo, s_hi)

    s, w = _nc_solve_once(s_init, x_grid, g, s_lo, s_hi)

    for k in range(lawson_iter):
        Phi = np.exp(-np.outer(x_grid, np.exp(s)))
        e = g - Phi @ w
        ae = np.abs(e)
        delta = max(1e-2 * np.max(ae), 1e-30)
        irls_w = 1.0 / np.maximum(ae, delta)
        irls_w /= np.sum(irls_w)
        s, w = _nc_solve_once(s, x_grid, g, s_lo, s_hi, weights=irls_w)

    return s, w


def _nc_solve_varpro(N, R, lawson_iter=4):
    """Compute N-point minimax quadrature for 1/x on [1, R] via VarPro+Lawson.

    Uses continuation in R from R=2 upward, warm-starting each solve.

    Returns
    -------
    t : ndarray (N,)
    w : ndarray (N,)
    err : float
    """
    M_eval = max(1000, 40 * N)
    x_eval = np.exp(np.linspace(0, np.log(R), M_eval))

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
            s_, w_ = _nc_solve_at_R(N, Ri, s_, lawson_iter=lawson_iter)

        t_ = np.exp(s_)
        approx = np.exp(-np.outer(x_eval, t_)) @ w_
        err_ = np.max(np.abs(1.0 / x_eval - approx))
        return s_, w_, err_

    s_hack = np.log(np.pi**2 * (np.arange(1, N + 1) - 0.5) / (2.0 * np.log(4.0 * R)))
    best_s, best_w, best_err = _try_init(s_hack)

    s_unif = np.linspace(s_lo + 0.5, s_hi - 0.5, N)
    s2, w2 = _nc_solve_at_R(N, R, s_unif, lawson_iter=lawson_iter)
    t2 = np.exp(s2)
    err2 = np.max(np.abs(1.0 / x_eval - np.exp(-np.outer(x_eval, t2)) @ w2))
    if err2 < best_err:
        best_s, best_w, best_err = s2, w2, err2

    t = np.exp(best_s)
    order = np.argsort(t)
    return t[order], best_w[order], best_err


def noncrossing_grids(R, eps, N_start=2, N_max=60):
    """Find minimum N achieving error < eps on [1, R].

    Returns
    -------
    t, w, N, err
    """
    for N in range(N_start, N_max + 1):
        t, w, err = _nc_solve_varpro(N, R)
        if err < eps:
            return t, w, N, err
    return t, w, N_max, err


# --- Remez exchange solver (enhanced, standalone use) ---

def _nc_hack_init_s(N, R):
    return np.log(np.pi**2 * (np.arange(1, N + 1) - 0.5) / (2.0 * np.log(4.0 * R)))


def _nc_loguni_init_s(N, R):
    s_lo = -np.log(2.0 * R) - 1.0
    s_hi = np.log(max(5.0 * N, 10.0)) + 1.0
    return np.linspace(s_lo + 0.5, s_hi - 0.5, N)


def _nc_phi(x, s):
    t = np.exp(s)
    return np.exp(-np.outer(x, t)), t


def _nc_ls_weights(x, s):
    P, _ = _nc_phi(x, s)
    return np.linalg.lstsq(P, 1.0 / x, rcond=None)[0]


def _nc_err_curve(x, s, w):
    P, _ = _nc_phi(x, s)
    return 1.0 / x - P @ w


def _nc_select_alternating_extrema(x, e, m):
    a = np.abs(e)
    idx = [0]
    for i in range(1, len(e) - 1):
        if a[i] >= a[i - 1] and a[i] >= a[i + 1]:
            idx.append(i)
    idx.append(len(e) - 1)

    blocks = []
    cur = [idx[0]]
    cur_sign = 1 if e[idx[0]] >= 0 else -1
    for j in idx[1:]:
        sgn = 1 if e[j] >= 0 else -1
        if sgn == cur_sign:
            cur.append(j)
        else:
            blocks.append(max(cur, key=lambda k: abs(e[k])))
            cur = [j]
            cur_sign = sgn
    blocks.append(max(cur, key=lambda k: abs(e[k])))
    blocks = np.array(blocks, dtype=int)

    if len(blocks) < m:
        extra = np.linspace(0, len(x) - 1, m, dtype=int)
        blocks = np.unique(np.concatenate([blocks, extra]))

    if len(blocks) > m:
        best = None
        best_score = (-1.0, -1.0)
        for s0 in range(len(blocks) - m + 1):
            seg = blocks[s0:s0 + m]
            score = (float(np.min(np.abs(e[seg]))), float(np.sum(np.abs(e[seg]))))
            if score > best_score:
                best_score = score
                best = seg
        blocks = best

    signs = np.sign(e[blocks])
    signs[signs == 0] = 1.0
    first = signs[0]
    signs = np.array([first * ((-1.0) ** i) for i in range(len(blocks))], dtype=float)
    return x[blocks], signs


def _nc_newton_equioscillation(xr, signs, s0, w0, lam0, R, maxit=20):
    N = len(s0)
    s = np.array(s0, float)
    w = np.array(w0, float)
    lam = float(lam0)

    s_lo = -np.log(2.0 * R) - 2.0
    s_hi = np.log(max(8.0 * N, 10.0)) + 2.0
    y = 1.0 / xr

    for _ in range(maxit):
        t = np.exp(s)
        E = np.exp(-np.outer(xr, t))
        F = y - E @ w - signs * lam
        fn = float(np.max(np.abs(F)))
        if fn < 1e-13:
            break

        J = np.empty((len(xr), 2 * N + 1))
        J[:, :N] = (w * t)[None, :] * xr[:, None] * E
        J[:, N:2 * N] = -E
        J[:, 2 * N] = -signs

        try:
            dz = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dz = np.linalg.lstsq(J, -F, rcond=None)[0]

        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
            sn = np.clip(s + alpha * dz[:N], s_lo, s_hi)
            order = np.argsort(sn)
            sn = sn[order]
            wn = (w + alpha * dz[N:2 * N])[order]
            lamn = lam + alpha * dz[2 * N]

            tn = np.exp(sn)
            En = np.exp(-np.outer(xr, tn))
            Fn = y - En @ wn - signs * lamn
            if float(np.max(np.abs(Fn))) < fn:
                s, w, lam = sn, wn, lamn
                improved = True
                break

        if not improved:
            break

        if np.linalg.norm(dz) < 1e-13 * (1.0 + np.linalg.norm(np.concatenate([s, w, [lam]]))):
            break

    return s, w, lam


def _nc_remez_at_R(N, R, s_init, max_outer=6):
    Md = max(2000, 120 * N)
    x = np.exp(np.linspace(0.0, np.log(R), Md))

    s = np.sort(np.array(s_init, float))

    try:
        s2, _ = _nc_solve_at_R(N, R, s, lawson_iter=0)
        s = np.sort(s2)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        # Falling back to the unrefined initial nodes is a legitimate
        # recovery, but it silently DEGRADES the tau quadrature that every
        # chi0 build in the run is evaluated on -- and the resulting error
        # shows up as a physics discrepancy, never as a failure.  Say so.
        import warnings
        warnings.warn(
            f"minimax: node refinement at N={N}, R={R:.3e} failed "
            f"({type(exc).__name__}: {exc}); continuing from the unrefined "
            f"initial nodes.  The tau-quadrature error below is the number "
            f"to check.",
            RuntimeWarning, stacklevel=2)

    w = _nc_ls_weights(x, s)
    e = _nc_err_curve(x, s, w)
    err = float(np.max(np.abs(e)))
    best_s, best_w, best_err = s.copy(), w.copy(), err
    history = [best_err]

    prev_ref = None
    for _ in range(max_outer):
        xr, signs = _nc_select_alternating_extrema(x, e, 2 * N + 1)

        if prev_ref is not None and np.max(np.abs(np.log(xr) - np.log(prev_ref))) < 1e-9:
            break
        prev_ref = xr.copy()

        lam0 = float(np.median(signs * (1.0 / xr - np.exp(-np.outer(xr, np.exp(s))) @ w)))
        s2, w2, _ = _nc_newton_equioscillation(xr, signs, s, w, lam0, R, maxit=25)

        e2 = _nc_err_curve(x, s2, w2)
        err2 = float(np.max(np.abs(e2)))
        if err2 < best_err * 0.9999:
            s, w, e = s2, w2, e2
            best_s, best_w, best_err = s.copy(), w.copy(), err2
            history.append(best_err)
        else:
            break

    t = np.exp(best_s)
    order = np.argsort(t)
    return t[order], best_w[order], best_err, history


def solve_noncrossing(N, R):
    """Compute minimax quadrature for 1/x on [1, R] via Remez exchange.

    More accurate than the VarPro+Lawson solver used by noncrossing_grids,
    but slower.  Use this for standalone high-accuracy solves.

    Returns
    -------
    tau : ndarray (N,)
    w : ndarray (N,)
    err : float
    """
    sched = [2.0]
    while sched[-1] < R:
        sched.append(min(R, sched[-1] * 2.0))

    starts = [
        _nc_hack_init_s(N, 2.0),
        _nc_loguni_init_s(N, 2.0),
    ]

    best = None
    for s0 in starts:
        s = np.array(s0, float)
        t = None
        w = None
        err = np.inf
        for Ri in sched:
            t, w, err, _ = _nc_remez_at_R(N, Ri, s, max_outer=6)
            s = np.log(t)

        if best is None or err < best[2]:
            best = (t, w, err)

    return best


def evaluate_noncrossing(x, tau, w):
    """Evaluate sum_l w_l exp(-tau_l x)."""
    x = np.asarray(x)
    return np.exp(-np.outer(x, tau)) @ w


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


# ================================================================
# CROSSING: sin-sum regularizations
# ================================================================

# --- VarPro-LM + LP solver ---

def _cr_varpro_lm(tau, u_grid, g, max_iter=120, tol=1e-14, weights=None,
                  tau_hi=None):
    """VarPro-LM for sin-basis crossing problem (Phase 2 polish)."""
    M = len(u_grid)
    tau = tau.copy()

    W = np.sqrt(weights) if weights is not None else np.ones(M)
    g_w = g * W

    def _eval(tau_):
        tau_s = np.maximum(tau_, 1e-10)
        Phi = np.sin(np.outer(u_grid, tau_s)) * W[:, None]
        U, sig, Vt = np.linalg.svd(Phi, full_matrices=False)
        sig_inv = np.where(sig > 1e-14 * max(sig[0], 1e-30), 1.0 / sig, 0.0)
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

        for k in range(lawson_iter):
            Phi = np.sin(np.outer(u_grid, np.maximum(tau, 1e-10)))
            e = g - Phi @ w
            ae = np.abs(e)
            delta = max(1e-2 * np.max(ae), 1e-30)
            irls_w = 1.0 / np.maximum(ae, delta)
            irls_w /= np.sum(irls_w)
            tau, w = _cr_varpro_lm(tau, u_grid, g, weights=irls_w,
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


def predict_N_crossing(xi_eff_target, E_bw, target_error, a_eff_est=1.35):
    """Predict number of sine-sum nodes for a crossing window.

    Parameters
    ----------
    xi_eff_target : float
        Desired effective Lorentzian broadening (eV).
    E_bw : float
        Energy bandwidth of the crossing window (eV).
    target_error : float
        Desired L-infinity fit error.

    Returns
    -------
    N : int
    A_est : float
    """
    xi_0_est = xi_eff_target / a_eff_est
    A_est = E_bw / xi_0_est
    ratio = (np.log(target_error) - _CR_INTERCEPT) / _CR_SLOPE
    ratio = max(ratio, 0.15)
    N = int(np.ceil(ratio * A_est))
    return max(N, 5), A_est


def _cr_delta_from_sines(tau, w, A):
    term = (np.sin(tau * A) - tau * A * np.cos(tau * A)) / (tau ** 2)
    return A - np.dot(w, term)


def _cr_a_eff_from_delta(delta, A):
    f = lambda a: a * np.arctan(A / a) - delta
    lo, hi = 1e-14, 10 * A
    try:
        return brentq(f, lo, hi)
    except ValueError:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0: hi = mid
            else: lo = mid
        return 0.5 * (lo + hi)


def _cr_solve_1overx(N, A_dim, u_min=5.0):
    M = max(250, 8 * N)
    u = np.linspace(u_min, A_dim, M)
    g = 1.0 / u
    fit_len = A_dim - u_min

    best_err, best_zeta = np.inf, 1.0
    for zeta in np.linspace(0.01, 10, 30):
        tau = np.arange(1, N + 1) * np.pi / (fit_len + zeta)
        Phi = np.sin(np.outer(u, tau))
        w = np.linalg.lstsq(Phi, g, rcond=None)[0]
        err = np.max(np.abs(g - Phi @ w))
        if err < best_err:
            best_err, best_zeta = err, zeta

    tau = np.arange(1, N + 1) * np.pi / (fit_len + best_zeta)
    tau = np.clip(tau, 1e-10, _CR_TAU_MAX * 1.5)

    def _eval(t):
        Phi = np.sin(np.outer(u, np.maximum(t, 1e-10)))
        U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
        si = np.where(s > 1e-14 * max(s[0], 1e-30), 1 / s, 0)
        w = Vt.T @ (si * (U.T @ g))
        r = g - U @ (U.T @ g)
        return w, r, r @ r, U

    w, r, cost, UU = _eval(tau)
    mu = 1e-6
    for _ in range(100):
        J = np.empty((M, N))
        for n in range(N):
            col = u * np.cos(tau[n] * u) * w[n]
            J[:, n] = -(col - UU @ (UU.T @ col))
        JtJ = J.T @ J
        dd = np.diag(JtJ).copy()
        dd[dd < 1e-20] = 1e-20
        try:
            dt = np.linalg.solve(JtJ + mu * np.diag(dd), -J.T @ r)
        except np.linalg.LinAlgError:
            dt = np.linalg.lstsq(JtJ + mu * np.diag(dd), -J.T @ r, rcond=None)[0]
        tn = np.sort(np.clip(tau + dt, 1e-10, _CR_TAU_MAX * 1.5))
        wn, rn, cn, Un = _eval(tn)
        if cn < cost:
            tau, w, r, cost, UU = tn, wn, rn, cn, Un
            mu = max(mu * 0.3, 1e-16)
        else:
            mu = min(mu * 5, 1e10)
        if np.linalg.norm(dt) < 1e-14 * (np.linalg.norm(tau) + 1e-30):
            break

    u_e = np.linspace(u_min, A_dim, 5000)
    g_e = 1.0 / u_e
    Phi = np.sin(np.outer(u_e, tau))
    w_f = np.linalg.lstsq(Phi, g_e, rcond=None)[0]
    err = np.max(np.abs(g_e - Phi @ w_f))
    return tau, w_f, err


def build_crossing_quadrature(N, xi_eff_target, E_bw, tol=0.05, verbose=True):
    """Build sine quadrature for a GW crossing window.

    Fits 1/u on [u_min, A] with N sines, binary-searching A to hit
    the target effective Lorentzian width xi_eff.

    Returns
    -------
    tau : ndarray (N,)
    w : ndarray (N,)
    info : dict
        xi_0, xi_eff, a_eff, u_min, A_dim, fit_err, N_over_A
    """
    u_min = 5.0

    def _try(A_dim):
        if A_dim < u_min + 2 or N > 1.5 * A_dim:
            return None, None, None, None
        tau, w, err = _cr_solve_1overx(N, A_dim, u_min)
        delta = _cr_delta_from_sines(tau, w, A_dim)
        if delta <= 0 or delta > A_dim:
            return tau, w, err, None
        a = _cr_a_eff_from_delta(delta, A_dim)
        return tau, w, err, a

    if verbose:
        print(f"Target: xi_eff={xi_eff_target:.4f} eV, "
              f"E_bw={E_bw:.2f} eV, N={N}")

    A_lo = max(u_min + 3, N / 1.0)
    A_hi = max(N * 5, E_bw / xi_eff_target * 10)

    best_tau, best_w, best_err, best_a = None, None, np.inf, None
    best_A = None

    for iteration in range(30):
        A_mid = 0.5 * (A_lo + A_hi)
        tau, w, err, a = _try(A_mid)

        if a is None or a <= 0:
            A_hi = A_mid
            continue

        xi_0 = E_bw / A_mid
        xi_eff = a * xi_0

        if verbose and iteration < 6:
            print(f"  iter {iteration}: A={A_mid:.1f}, N/A={N/A_mid:.3f}, "
                  f"a_eff={a:.3f}, xi_eff={xi_eff:.4f}, err={err:.2e}")

        if abs(xi_eff - xi_eff_target) / xi_eff_target < tol:
            best_tau, best_w, best_err, best_a, best_A = tau, w, err, a, A_mid
            break

        if xi_eff > xi_eff_target:
            A_lo = A_mid
        else:
            A_hi = A_mid

        best_tau, best_w, best_err, best_a, best_A = tau, w, err, a, A_mid
    else:
        if verbose:
            print(f"  Warning: did not converge to target within {tol:.0%}")

    xi_0 = E_bw / best_A
    xi_eff = best_a * xi_0

    info = {
        'xi_0': xi_0,
        'xi_eff': xi_eff,
        'a_eff': best_a,
        'u_min': u_min,
        'A_dim': best_A,
        'fit_err': best_err,
        'xi_eff_target': xi_eff_target,
        'N_over_A': N / best_A,
    }

    if verbose:
        print(f"\nResult:")
        print(f"  A_dim  = {best_A:.1f}  (N/A = {N/best_A:.3f})")
        print(f"  xi_0   = {xi_0:.5f} eV")
        print(f"  xi_eff = {xi_eff:.5f} eV  "
              f"({xi_eff/xi_eff_target:.1%} of target)")
        print(f"  a_eff  = {best_a:.4f}")
        print(f"  fit error = {best_err:.2e}")

    return best_tau, best_w, info


def evaluate_crossing(x, tau, w, xi_0):
    """Evaluate F(x) = sum w_l sin(tau_l x / xi_0)."""
    u = np.asarray(x) / xi_0
    return np.sin(np.outer(u, tau)) @ w


# ================================================================
# Physical rescaling helpers
# ================================================================

def rescale_noncrossing(t, w, E_gap):
    """Rescale canonical [1,R] grids to physical units.

    tau_phys = t / E_gap,  W_phys = w / E_gap.
    """
    return t / E_gap, w / E_gap


def rescale_crossing(tau, w, xi):
    """Rescale crossing grids to physical units.

    t_phys = xi * tau,  W_phys = w / xi.
    """
    return xi * tau, w / xi


def rescale_noncrossing_imag(t, w, E_gap):
    """Rescale from [1, R] to physical units [E_gap, E_bw].

    Physical: sum w_phys exp(-t_phys E) ≈ E/(E^2+omega_p^2)
    where omega_p = omega_hat * E_gap.

    t_phys = t / E_gap,  w_phys = w / E_gap.
    """
    return t / E_gap, w / E_gap

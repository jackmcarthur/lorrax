"""
Minimax quadrature nodes for GW self-energy frequency integration.

Two approximation problems for energy denominators 1/x:

Non-crossing windows (x > 0, definite sign):
    1/x ≈ sum_l w_l exp(-t_l x)   on [1, R],  R = E_bw / E_gap
    Node count scales as O(ln R) for fixed accuracy.
    Error: eps ≈ 0.31 * exp[-N * (3.55/ln(R) + 0.68)]

Crossing windows (x changes sign, regularized):
    1/x ≈ sum_l w_l sin(tau_l x / xi_0)   on [-A, A] excluding |x| < x_min
    Node count scales as O(A) where A = E_bw / xi_0.
    Error: eps ≈ exp(-0.93 - 14.25 * N/A)

Usage:
    from minimax_nodes import (
        # Non-crossing
        solve_noncrossing, predict_N_noncrossing, error_estimate_noncrossing,
        evaluate_noncrossing,
        # Crossing
        build_crossing_quadrature, predict_N_crossing,
        evaluate_crossing,
    )
"""

import numpy as np
import importlib.util
import sys
from scipy.optimize import brentq


# ================================================================
# NON-CROSSING: 1/x on [1, R] with exponential sums
# ================================================================

# Empirical error fit (R^2 = 0.995 on eps in [1e-5, 1e-2]):
#   log(eps) = ln(0.31) - N * (3.55/ln(R) + 0.68)
_NC_LN_C = np.log(0.3112)
_NC_A = 3.5456
_NC_B = 0.6845


def predict_N_noncrossing(R, target_error):
    """
    Predict number of exponential-sum nodes for 1/x on [1, R].

    Parameters
    ----------
    R : float
        Dynamic range E_bw / E_gap.
    target_error : float
        Desired L-infinity error.

    Returns
    -------
    N : int
        Predicted number of nodes (rounded up).
    """
    rate = _NC_A / np.log(R) + _NC_B
    N = (np.log(target_error) - _NC_LN_C) / (-rate)
    return max(int(np.ceil(N)), 2)


def error_estimate_noncrossing(N, R):
    """Estimate L-infinity error for N nodes at dynamic range R."""
    rate = _NC_A / np.log(R) + _NC_B
    return np.exp(_NC_LN_C - N * rate)


def solve_noncrossing(N, R):
    """
    Compute minimax quadrature for 1/x on [1, R].

    Parameters
    ----------
    N : int
        Number of quadrature nodes.
    R : float
        Dynamic range (must be > 1).

    Returns
    -------
    tau : ndarray (N,)
        Exponents (sorted ascending).
    w : ndarray (N,)
        Weights.
    err : float
        L-infinity error: max_{x in [1,R]} |1/x - sum w_l exp(-t_l x)|.
    """
    t, w, err, _ = _nc_solve_with_history(N, R)
    return t, w, err


def evaluate_noncrossing(x, tau, w):
    """Evaluate sum_l w_l exp(-tau_l x)."""
    x = np.asarray(x)
    return np.exp(-np.outer(x, tau)) @ w


# ================================================================
# CROSSING: 1/x on [-A, A] with sine sums (learned regularization)
# ================================================================

# Empirical error fit (R^2 = 0.996):
#   log(eps) = -0.93 - 14.25 * (N/A)
_CR_INTERCEPT = -0.93
_CR_SLOPE = -14.25

TAU_MAX = np.sqrt(2 * np.log(1e3))


def predict_N_crossing(xi_eff_target, E_bw, target_error, a_eff_est=1.35):
    """
    Predict number of sine-sum nodes for a crossing window.

    Parameters
    ----------
    xi_eff_target : float
        Desired effective Lorentzian broadening (eV).
    E_bw : float
        Energy bandwidth of the crossing window (eV).
    target_error : float
        Desired L-infinity fit error.
    a_eff_est : float
        Estimated a_eff = xi_eff / xi_0 (default 1.35, typical for u_min=5).

    Returns
    -------
    N : int
        Predicted number of nodes (rounded up).
    A_est : float
        Estimated dimensionless bandwidth.
    """
    xi_0_est = xi_eff_target / a_eff_est
    A_est = E_bw / xi_0_est
    ratio = (np.log(target_error) - _CR_INTERCEPT) / _CR_SLOPE
    ratio = max(ratio, 0.15)
    N = int(np.ceil(ratio * A_est))
    return max(N, 5), A_est


def build_crossing_quadrature(N, xi_eff_target, E_bw, tol=0.05, verbose=True):
    """
    Build sine quadrature for a GW crossing window.

    Fits 1/u on [u_min, A] with N sines, binary-searching A to hit
    the target effective Lorentzian width xi_eff.

    Parameters
    ----------
    N : int
        Number of quadrature nodes.
    xi_eff_target : float
        Desired effective Lorentzian broadening (eV).
    E_bw : float
        Energy bandwidth of the crossing window (eV).
    tol : float
        Fractional tolerance on xi_eff (default 5%).
    verbose : bool
        Print progress.

    Returns
    -------
    tau : ndarray (N,)
        Frequencies in units of 1/xi_0.
        Physical: F(x) = sum w_l sin(tau_l * x / xi_0).
    w : ndarray (N,)
        Weights.
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
# NON-CROSSING INTERNALS
# ================================================================

_cached_minimax_module = None


def _nc_get_minimax_module():
    global _cached_minimax_module
    if _cached_minimax_module is None:
        from pathlib import Path
        path = str(Path(__file__).resolve().parent / "minimax.py")
        spec = importlib.util.spec_from_file_location("minimax_mod_nc", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["minimax_mod_nc"] = mod
        spec.loader.exec_module(mod)
        _cached_minimax_module = mod
    return _cached_minimax_module


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


def _nc_remez_at_R(N, R, s_init, minimax_module, max_outer=6):
    Md = max(2000, 120 * N)
    x = np.exp(np.linspace(0.0, np.log(R), Md))

    s = np.sort(np.array(s_init, float))

    try:
        s2, _ = minimax_module._nc_solve_at_R(N, R, s, lawson_iter=0)
        s = np.sort(s2)
    except Exception:
        pass

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


def _nc_solve_with_history(N, R):
    minimax_module = _nc_get_minimax_module()

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
        hist = []
        for Ri in sched:
            t, w, err, hist = _nc_remez_at_R(N, Ri, s, minimax_module, max_outer=6)
            s = np.log(t)

        if best is None or err < best[2]:
            best = (t, w, err, hist)

    return best


# ================================================================
# CROSSING INTERNALS
# ================================================================

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
    tau = np.clip(tau, 1e-10, TAU_MAX * 1.5)

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
        tn = np.sort(np.clip(tau + dt, 1e-10, TAU_MAX * 1.5))
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

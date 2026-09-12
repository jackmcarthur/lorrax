"""Levelled (minimax-optimal) exponential sums for ``1/x`` on ``[1, R]``.

``noncrossing_levelled(R, eps, N_max)`` returns the smallest ``N`` whose best
uniform ``N``-term error ``E_N(R)`` is at most ``eps``, the best rule for that
``N`` and its certified error, in tens to a few hundred milliseconds on one
core.  NumPy only.

Certificate.  A nonzero sum of ``M`` exponentials ``exp(-t x)`` with distinct
real ``t`` has at most ``M - 1`` real zeros.  If ``e = 1/x - s`` alternates in
sign at ``2N + 1`` points with ``|e| >= delta``, no ``N``-term sum has error
below ``delta`` (de la Vallee Poussin for this family).  ``certify_noncrossing``
returns that alternation minimum and the refined dense sup; when they agree
the rule is the minimax rule to that factor, and when the ``N - 1`` rule is
levelled above ``eps`` the returned ``N`` is minimal.

Algorithm.  Remez exchange.  At each fixed reference the levelled equations
``1/x_i = sum_k w_k exp(-t_k x_i) + (-1)^i E`` are solved by variable
projection: ``(w, E)`` by least squares, ``log t`` by Levenberg-Marquardt.
Tempting, and why not: the full Newton on ``(log t, w, E)`` that
``solver._nc_newton_equioscillation`` uses.  The basis trades a node shift
against a weight change; the Newton step is ``O(cond * E)`` in ``log t`` while
its quadratic error is ``O(step^2) >> E``, so no line-search step lowers the
residual (measured at ``R = 10, N = 7``: condition 3.7e6, residual stuck at
1.04e-8 against ``E = 2.4e-8``).  Projection removes those directions.

``N`` advances by one node below ``N = 6`` and by two above.  Each start is the
last two levelled rules extrapolated in ``N`` at fixed normalised position
(``log t`` over ``(i + 1/2)/N``, ``log x_ref`` over ``i/(2N)``); on certified
rules this predicts ``log t`` to a median 0.015 for one node and 0.073 for two,
against 0.084 and 0.183 for plain resampling.  Tempting, and why not: jumps of
``N/2`` or more (median error >= 0.35; the smallest node moves ~100x between
``N = 8`` and ``N = 16`` at large ``R``).  When ``N + 2`` passes, ``N + 1`` is
solved from the midpoint of the two rules.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["noncrossing_levelled", "certify_noncrossing"]

_ROUNDOFF = 1.0e-13          # |upper - lower| below this is double-precision noise (O(1) terms)
_LAW_A, _LAW_B = 3.5456, 0.6845


def _err(x, t, w, order=1):
    E = np.exp(-np.outer(x, t))
    out = [1.0 / x - E @ w]
    if order >= 1:
        out.append(-1.0 / x ** 2 + E @ (w * t))
    if order >= 2:
        out.append(2.0 / x ** 3 - E @ (w * t * t))
    return out


def _extrema(t, w, R):
    """End points plus the interior roots of e' (coarse log grid, safeguarded
    Newton on x e'(x) in log x)."""
    N = t.size
    M = max(400, 24 * (2 * N + 1))
    for attempt in range(4):
        x = np.geomspace(1.0, R, M)
        e1 = _err(x, t, w)[1]
        idx = np.nonzero(np.sign(e1[:-1]) * np.sign(e1[1:]) < 0)[0]
        if idx.size >= 2 * N - 1 or attempt == 3:
            break
        M *= 4
    lo, hi = np.log(x[idx]), np.log(x[idx + 1])
    slo = np.sign(e1[idx])
    u = 0.5 * (lo + hi)
    for _ in range(10):
        X = np.exp(u)
        _, d1, d2 = _err(X, t, w, order=2)
        g, gp = X * d1, X * d1 + X * X * d2
        same = np.sign(g) == slo
        lo, hi = np.where(same, u, lo), np.where(same, hi, u)
        un = u - g / np.where(gp != 0.0, gp, 1.0)
        u = np.where((un > lo) & (un < hi) & np.isfinite(un), un, 0.5 * (lo + hi))
    X = np.unique(np.concatenate([[1.0], np.exp(u), [float(R)]]))
    return X, _err(X, t, w, order=0)[0]


def _alternating_reference(ev, n_ref):
    keep = []
    for i, v in enumerate(ev):
        if keep and np.sign(v) == np.sign(ev[keep[-1]]):
            if abs(v) > abs(ev[keep[-1]]):
                keep[-1] = i
        else:
            keep.append(i)
    while len(keep) > n_ref:
        vals = np.abs(ev[keep])
        j = int(np.argmin(vals))
        if j in (0, len(keep) - 1):
            keep.pop(j)
        elif len(keep) - n_ref >= 2:
            if abs(ev[keep[j - 1]]) < abs(ev[keep[j + 1]]):
                del keep[j - 1:j + 1]
            else:
                del keep[j:j + 2]
        else:
            keep.pop(0 if abs(ev[keep[0]]) < abs(ev[keep[-1]]) else len(keep) - 1)
    return keep


def certify_noncrossing(tau, weights, R, M=None, iters=40):
    """``(lower, upper)`` for a rule on ``[1, R]``: the alternation minimum
    (a lower bound on the best error at this node count, 0 without ``2N+1``
    alternations) and the dense sup refined by bisection on e'."""
    t = np.asarray(tau, np.float64)
    w = np.asarray(weights, np.float64)
    M = M or max(4000, 200 * t.size)
    x = np.geomspace(1.0, float(R), M)
    e1 = _err(x, t, w)[1]
    idx = np.nonzero(np.sign(e1[:-1]) * np.sign(e1[1:]) < 0)[0]
    lo, hi = np.log(x[idx]), np.log(x[idx + 1])
    slo = np.sign(e1[idx])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        left = np.sign(_err(np.exp(mid), t, w)[1]) == slo
        lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
    X = np.unique(np.concatenate([[1.0], np.exp(0.5 * (lo + hi)), [float(R)]]))
    ev = _err(X, t, w, order=0)[0]
    ref = _alternating_reference(ev, 2 * t.size + 1)
    lower = float(np.min(np.abs(ev[ref]))) if len(ref) == 2 * t.size + 1 else 0.0
    return lower, float(np.max(np.abs(ev)))


def _level(xr, signs, s, R, rtol, mu=1e-6, maxit=40):
    N = s.size
    f = 1.0 / xr
    s_lo, s_hi = -math.log(R) - 12.0, math.log(1e6)

    def lin(s_):
        Ex = np.exp(-np.outer(xr, np.exp(s_)))
        B = np.empty((xr.size, N + 1))
        B[:, :N] = Ex
        B[:, N] = signs
        cs = np.sqrt(np.einsum("ij,ij->j", B, B))
        Q, Rr = np.linalg.qr(B / cs)
        try:
            c = np.linalg.solve(Rr, Q.T @ f)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(Rr, Q.T @ f, rcond=None)[0]
        p = c / cs
        return p, f - B @ p, Q, Ex

    p, r, Q, Ex = lin(s)
    cost = float(r @ r)
    for _ in range(maxit):
        if math.sqrt(cost) <= rtol * abs(p[N]):
            break
        t = np.exp(s)
        dB = -(xr[:, None] * Ex) * (t * p[:N])[None, :]
        J = -(dB - Q @ (Q.T @ dB))
        G = J.T @ J
        g = J.T @ r
        dg = np.maximum(np.diag(G), 1e-300)
        improved = False
        for _k in range(10):
            try:
                ds = np.linalg.solve(G + mu * np.diag(dg), -g)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            sn = np.clip(s + np.clip(ds, -1.0, 1.0), s_lo, s_hi)
            if N > 1 and np.min(np.diff(np.sort(sn))) <= 1e-9:
                mu *= 4.0
                continue
            pn, rn, Qn, Exn = lin(sn)
            cn = float(rn @ rn)
            if np.isfinite(cn) and cn < cost:
                s, p, r, Q, Ex, cost = sn, pn, rn, Qn, Exn, cn
                mu = max(mu / 5.0, 1e-15)
                improved = True
                break
            mu *= 4.0
        if not improved:
            break
    order = np.argsort(s)
    return s[order], p[:N][order], mu


def _exchange(s, R, xr, signs, maxit=20, gap_tol=1e-4):
    N = s.size
    best, mu, gap = None, 1e-6, 1.0
    for it in range(1, maxit + 1):
        rtol = max(1e-9, min(1e-2, 0.1 * gap))
        s, w, mu = _level(np.asarray(xr, float), np.asarray(signs, float), s, R, rtol, mu)
        X, ev = _extrema(np.exp(s), w, R)
        upper = float(np.max(np.abs(ev)))
        ref = _alternating_reference(ev, 2 * N + 1)
        full = len(ref) == 2 * N + 1
        lower = float(np.min(np.abs(ev[ref]))) if full else 0.0
        cand = dict(t=np.exp(s), w=w, lower=lower, upper=upper,
                    ref=(X[ref].copy(), np.sign(ev[ref]).copy()) if full else None)
        if best is None or (cand["lower"] > 0) > (best["lower"] > 0) or (
                (cand["lower"] > 0) == (best["lower"] > 0) and cand["upper"] < best["upper"]):
            best = cand
        if not full:
            break
        gap = upper / lower - 1.0
        if gap < gap_tol or upper - lower < _ROUNDOFF:
            break
        xr, signs = X[ref], np.sign(ev[ref])
    g = best["upper"] / best["lower"] - 1.0 if best["lower"] > 0 else np.inf
    best["levelled"] = bool(best["lower"] > 0 and (g < 1e-2 or best["upper"] - best["lower"] < _ROUNDOFF))
    return best


def _resample(v, n_new, centred):
    v = np.asarray(v, float)
    if not centred:
        return np.interp(np.linspace(0.0, 1.0, n_new), np.linspace(0.0, 1.0, v.size), v)
    n = v.size
    if n == 1:
        return v[0] + np.log(3.0) * np.linspace(-1.0, 1.0, n_new)
    xi = (np.arange(n) + 0.5) / n
    xn = (np.arange(n_new) + 0.5) / n_new
    out = np.interp(xn, xi, v)
    lo = (v[1] - v[0]) / (xi[1] - xi[0])
    hi = (v[-1] - v[-2]) / (xi[-1] - xi[-2])
    out = np.where(xn < xi[0], v[0] + lo * (xn - xi[0]), out)
    return np.where(xn > xi[-1], v[-1] + hi * (xn - xi[-1]), out)


def _start(a, b, Na, Nb, n_new, R):
    """Start for ``n_new`` nodes from levelled rules ``b`` (``Nb`` nodes) and,
    when given, ``a`` (``Na < Nb``): linear in N at fixed normalised position."""
    lt = _resample(np.log(np.sort(b["t"])), n_new, True)
    lx = _resample(np.log(b["ref"][0]), 2 * n_new + 1, False)
    if a is not None:
        lam = (n_new - Nb) / (Nb - Na)
        lt = (1 + lam) * lt - lam * _resample(np.log(np.sort(a["t"])), n_new, True)
        lx = (1 + lam) * lx - lam * _resample(np.log(a["ref"][0]), 2 * n_new + 1, False)
    lt = np.sort(lt)
    for i in range(1, lt.size):
        lt[i] = max(lt[i], lt[i - 1] + 1e-3)
    xr = np.exp(np.sort(lx))
    xr[0], xr[-1] = 1.0, float(R)
    return lt, xr, b["ref"][1][0] * (-1.0) ** np.arange(2 * n_new + 1)


def _prelevel(n_new, R, t0, rounds=6, lm_iters=15):
    """Robust start: VarPro Levenberg-Marquardt on log t with Lawson
    reweighting on a log grid, then an exchange from the curve's own extrema."""
    x = np.geomspace(1.0, R, max(800, 40 * n_new))
    f = 1.0 / x
    s = np.log(np.sort(np.asarray(t0, float)))
    lam = np.full(x.size, 1.0 / x.size)
    w = None
    for _ in range(rounds):
        sq = np.sqrt(lam)

        def solve(s_):
            A = np.exp(-np.outer(x, np.exp(s_))) * sq[:, None]
            cs = np.linalg.norm(A, axis=0)
            cs[cs == 0] = 1.0
            Q, Rr = np.linalg.qr(A / cs)
            ww = np.linalg.lstsq(Rr, Q.T @ (f * sq), rcond=None)[0] / cs
            return ww, f * sq - A @ ww, A, Q

        w, r, A, Q = solve(s)
        cost, mu = r @ r, 1e-3
        for _ in range(lm_iters):
            dA = -(x[:, None] * A) * (np.exp(s) * w)[None, :]
            J = -(dA - Q @ (Q.T @ dA))
            G, g = J.T @ J, J.T @ r
            dg = np.maximum(np.diag(G), 1e-300)
            ok = False
            for _k in range(8):
                try:
                    ds = np.linalg.solve(G + mu * np.diag(dg), -g)
                except np.linalg.LinAlgError:
                    mu *= 10.0
                    continue
                sn = np.clip(s + np.clip(ds, -2.0, 2.0), -math.log(R) - 12.0, math.log(1e6))
                wn, rn, An, Qn = solve(sn)
                if np.isfinite(rn @ rn) and rn @ rn < cost:
                    s, w, r, A, Q, cost, ok = sn, wn, rn, An, Qn, rn @ rn, True
                    mu = max(mu / 3.0, 1e-12)
                    break
                mu *= 5.0
            if not ok:
                break
        lam = lam * np.abs(f - np.exp(-np.outer(x, np.exp(s))) @ w)
        if not np.isfinite(lam.sum()) or lam.sum() <= 0:
            break
        lam /= lam.sum()
    t = np.exp(s)
    X, ev = _extrema(t, w, R)
    ref = _alternating_reference(ev, 2 * n_new + 1)
    if len(ref) < 2 * n_new + 1:
        return dict(t=t, w=w, lower=0.0, upper=float(np.max(np.abs(ev))), ref=None, levelled=False)
    return _exchange(s, R, X[ref], np.sign(ev[ref]))


def _step(sols, n_new, R):
    lev = [n for n in sorted(sols) if sols[n]["levelled"] and n < n_new]
    a = sols[lev[-2]] if len(lev) >= 2 else None
    b = sols[lev[-1]]
    s0, xr, signs = _start(a, b, lev[-2] if a is not None else None, lev[-1], n_new, R)
    new = _exchange(s0, R, xr, signs)
    if not new["levelled"]:
        alt = _exchange(s0, R, xr, -signs)
        if alt["levelled"] or alt["upper"] < new["upper"]:
            new = alt
    if not new["levelled"]:
        rob = _prelevel(n_new, R, np.exp(s0))
        if rob["levelled"] or rob["upper"] < new["upper"]:
            new = rob
    return new


def noncrossing_levelled(R, eps, N_max=64):
    """Smallest-``N`` levelled rule for ``1/x`` on ``[1, R]`` with sup <= eps.

    Returns ``(tau, weights, N, err)`` like the other family drivers;
    ``err`` is the refined dense sup of the returned rule.  When ``N_max`` does
    not reach ``eps`` the ``N_max`` rule is returned with its (larger) error,
    exactly as the VarPro ladder it replaced did, and the caller must
    compare."""
    R = max(float(R), 1.0 + 1.0e-9)
    eps = float(eps)
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(f"noncrossing_levelled: eps must be finite and positive; got {eps!r}")
    N_max = int(N_max)
    s0 = np.array([-0.5 * math.log(R)])
    xr = np.array([1.0, math.sqrt(R), R])
    sol = _exchange(s0, R, xr, np.array([1.0, -1.0, 1.0]))
    if not sol["levelled"]:
        sol = _exchange(s0, R, xr, np.array([-1.0, 1.0, -1.0]))
    sols = {1: sol}
    N = 1
    while sols[N]["upper"] > eps and N < N_max:
        lev = [n for n in sorted(sols) if sols[n]["levelled"]]
        if not lev:
            break
        if len(lev) >= 2:
            rate = math.log(sols[lev[-2]]["upper"] / sols[lev[-1]]["upper"]) / (lev[-1] - lev[-2])
        else:
            rate = _LAW_A / math.log(R) + _LAW_B
        need = math.log(sols[N]["upper"] / eps) / max(rate, 1e-3)
        k = min(1 if (need <= 1.0 or N < 6) else 2, N_max - N)
        new = _step(sols, N + k, R)
        if k == 2 and not new["levelled"]:
            # A two-node step can miss the basin where the smallest node moves
            # fastest (measured R ~ 7e3..3e4 at N = 8 -> 10 and 10 -> 12);
            # one node from the last levelled rule is the reliable move.
            k = 1
            new = _step(sols, N + 1, R)
        sols[N + k] = new
        if k == 2 and new["levelled"] and new["upper"] <= eps and sols[N]["levelled"]:
            s1, xr1, sg1 = _start(sols[N], new, N, N + 2, N + 1, R)
            mid = _exchange(s1, R, xr1, sg1)
            if mid["levelled"] and mid["upper"] <= eps:
                N, new = N + 1, mid
            else:
                N += 2
            sols[N] = new
            break
        N += k
    best = sols[N]
    _lower, upper = certify_noncrossing(best["t"], best["w"], R)
    return np.asarray(best["t"], np.float64), np.asarray(best["w"], np.float64), int(N), float(upper)

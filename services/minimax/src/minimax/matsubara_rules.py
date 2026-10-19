"""Finite-temperature bosonic Matsubara rules for KMS-bounded imaginary-time correlations.

The response of independent Fermi-Dirac particles at inverse temperature beta is
an imaginary-time correlation on [0, beta] whose every term is a bounded
exponential (Kubo-Martin-Schwinger):

    h_mn(tau) = f_m (1 - f_n) exp(-x tau),   x = e_n - e_m,   0 <= h <= 1,
    int_0^beta exp(i nu_k tau) h_mn(tau) dtau = (f_m - f_n) / (x - i nu_k),

with bosonic nu_k = 2 pi k / beta.  KMS maps a pair with x < 0 onto the mirrored
node beta - tau with |x| (f_m (1-f_n) exp(-x tau) = f_n (1-f_m) exp(-|x| (beta-tau))),
so one set of nodes tau_l in (0, beta/2] with complex weights W_kl, used as

    sum_l W_kl h(tau_l) + conj(W_kl) h(beta - tau_l),

integrates every pair of either sign once it satisfies, for y in [0, Delta],

    even: sum_l Re W_kl [e^{-y tau_l} + e^{-y (beta - tau_l)}] ~ y (1 - e^{-beta y}) / (y^2 + nu_k^2)
    odd:  sum_l Im W_kl [e^{-y tau_l} - e^{-y (beta - tau_l)}] ~ nu_k (1 - e^{-beta y}) / (y^2 + nu_k^2).

This is the finite-temperature Matsubara quadrature problem of Kaltak and
Kresse (Phys. Rev. B 101, 205145 (2020)) for the bosonic kernel.  The node set
here is NOT their nonlinear minimax optimum: it is a graded composite
Gauss-Legendre pool on [0, beta/2] compressed by a column-pivoted QR
(interpolative decomposition) of the basis, with least-squares weights per
frequency, and the smallest node count whose sup-norm error on a dense y grid
(maxima refined locally) meets the tolerance for every requested frequency.
Numbers in, numbers out; no GW code.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

#: Largest node count this rule builds before refusing.
MAX_NODES = 256
#: Roundoff ceiling: the rule refuses when eps_machine * amplification exceeds a
#: tenth of the tolerance (the weights would cancel below their own noise).
_ROUNDOFF_FRACTION = 0.1


def _targets(y, beta, nu):
    g = -np.expm1(-beta * y)
    if nu == 0.0:
        safe = np.where(y > 0.0, y, 1.0)
        return np.where(y > 0.0, g / safe, beta), np.zeros_like(y)
    den = y * y + nu * nu
    return y * g / den, nu * g / den


def _basis(y, beta, tau):
    a = np.exp(-np.outer(y, tau))
    b = np.exp(-np.outer(y, beta - tau))
    return a + b, a - b


def _chebyshev(lo, hi, n):
    j = np.arange(n)
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.pi * (j + 0.5) / n)


def _y_grid(beta, delta, n_lin, n_log, extra):
    parts = [_chebyshev(0.0, min(delta, 40.0 / beta), n_lin), [0.0, delta], extra]
    if delta > 1.0 / beta:
        parts.append(np.exp(_chebyshev(math.log(1.0 / beta), math.log(delta), n_log)))
    y = np.unique(np.clip(np.concatenate([np.ravel(p) for p in parts]), 0.0, delta))
    return y


def _pool(beta, delta, nu_max, order):
    """Graded composite Gauss-Legendre nodes on (0, beta/2]."""
    half = 0.5 * beta
    edges, h = [0.0], min(half, 1.0 / delta)
    while edges[-1] + h < half:
        edges.append(edges[-1] + h)
        h *= 2.0
    edges.append(half)
    if nu_max > 0.0:          # bound the Matsubara phase per panel to one period
        refined = [edges[0]]
        for a, b in zip(edges[:-1], edges[1:]):
            pieces = max(1, math.ceil((b - a) * nu_max / (2.0 * math.pi)))
            refined.extend(a + (b - a) * np.arange(1, pieces + 1) / pieces)
        edges = refined
    x, _ = leggauss(order)
    nodes = [a + (b - a) * (x + 1.0) / 2.0 for a, b in zip(edges[:-1], edges[1:])]
    return np.concatenate(nodes), len(edges) - 1


def _fit(C, S, targets, sel):
    """Least-squares weights on the selected columns, one per frequency and channel."""
    rows = []
    for (even, odd, s_even, s_odd) in targets:
        pair = []
        for M, f, s in ((C, even, s_even), (S, odd, s_odd)):
            if s == 0.0:
                pair.append(np.zeros(len(sel)))
                continue
            A = M[:, sel]
            norm = np.linalg.norm(A, axis=0)
            norm[norm == 0.0] = 1.0
            c, *_ = np.linalg.lstsq(A / norm, f / s, rcond=None)
            pair.append(c / norm * s)
        rows.append(pair[0] + 1j * pair[1])
    return np.asarray(rows)


def _sup_error(y, beta, tau, W, nus, scales, refine=True):
    """Relative sup errors (even, odd) per frequency on the grid, maxima refined between neighbours."""
    out = []
    for k, nu in enumerate(nus):
        s_even, s_odd = scales[k]

        def err(yy):
            C, S = _basis(yy, beta, tau)
            even, odd = _targets(yy, beta, nu)
            e_even = np.abs(C @ W[k].real - even) / s_even
            e_odd = (np.abs(S @ W[k].imag - odd) / s_odd if s_odd > 0.0
                     else np.abs(S @ W[k].imag))
            return e_even, e_odd

        e_even, e_odd = err(y)
        best = [float(e_even.max()), float(e_odd.max())]
        if refine:
            for channel, values in enumerate((e_even, e_odd)):
                for i in np.argsort(values)[-4:]:
                    lo, hi = y[max(i - 1, 0)], y[min(i + 1, y.size - 1)]
                    if hi > lo:
                        fine = err(np.linspace(lo, hi, 65))[channel]
                        best[channel] = max(best[channel], float(fine.max()))
        out.append(best)
    return np.asarray(out)


def matsubara_response_rule(beta_ry_inv, delta_max_ry, n_indices, *, rel_tol=1e-8):
    """Nodes and complex weights for chi(i nu_k) of a KMS-bounded tau correlation.

    Parameters
    ----------
    beta_ry_inv : float
        Inverse temperature beta = 1 / k_B T in 1/Ry (Fermi-Dirac width sigma = 1/beta).
    delta_max_ry : float
        Bound on |x| = |e_n - e_m| over every state pair the correlation contains, Ry.
    n_indices : sequence of int
        Nonnegative bosonic Matsubara indices k (nu_k = 2 pi k / beta).
    rel_tol : float
        Sup-norm error of each even and odd target over y in [0, delta_max],
        relative to that target's own sup over the same interval.

    Returns
    -------
    dict
        ``t`` [L] in (0, beta/2] (1/Ry), ``weights`` [n_nu, L] complex (1/Ry) used as
        ``sum_l W h(t_l) + conj(W) h(beta - t_l)``; ``nu_ry``; the measured relative
        errors, the roundoff amplification and a certificate; ``node_digest``.
        The consumer owns the correlation, its sign and its mirrored partner.
    """
    beta = float(beta_ry_inv)
    delta = float(delta_max_ry)
    idx = np.asarray(n_indices, dtype=np.int64)
    if not (np.isfinite(beta) and beta > 0.0):
        raise ValueError(f"matsubara rule: beta must be finite and positive; got {beta_ry_inv!r}")
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError(f"matsubara rule: delta_max must be finite and positive; got {delta_max_ry!r}")
    if idx.ndim != 1 or idx.size == 0 or np.any(idx < 0) or np.unique(idx).size != idx.size:
        raise ValueError("matsubara rule: n_indices must be distinct nonnegative integers")
    if not (np.isfinite(rel_tol) and 1e-13 <= rel_tol < 0.1):
        raise ValueError("matsubara rule tolerance must lie in [1e-13, .1)")
    nus = 2.0 * math.pi * idx.astype(np.float64) / beta
    extra = nus[nus <= delta]
    y_fit = _y_grid(beta, delta, 160, 320, extra)
    y_check = _y_grid(beta, delta, 1280, 2560, extra)

    targets, scales = [], []
    for nu in nus:
        even, odd = _targets(y_check, beta, nu)
        s_even, s_odd = float(np.abs(even).max()), float(np.abs(odd).max())
        scales.append((s_even, s_odd))
        fe, fo = _targets(y_fit, beta, nu)
        targets.append((fe, fo, s_even, s_odd))

    from scipy.linalg import qr

    for order in (24, 32, 48):
        pool, panels = _pool(beta, delta, float(nus.max()), order)
        C, S = _basis(y_fit, beta, pool)
        stacked = np.vstack([C, S])
        stacked = stacked / np.linalg.norm(stacked, axis=0)
        _, _, piv = qr(stacked, mode="economic", pivoting=True)
        n = 4
        while n <= min(MAX_NODES, pool.size):
            sel = np.sort(piv[:n])
            W = _fit(C, S, targets, sel)
            errors = _sup_error(y_check, beta, pool[sel], W, nus, scales)
            if errors.max() <= 0.5 * rel_tol:
                tau = pool[sel]
                amplification = [float(2.0 * max(np.abs(W[k].real).sum() / scales[k][0],
                                                  (np.abs(W[k].imag).sum() / scales[k][1]
                                                   if scales[k][1] > 0.0 else 0.0)))
                                 for k in range(nus.size)]
                if np.finfo(np.float64).eps * max(amplification) > _ROUNDOFF_FRACTION * rel_tol:
                    raise RuntimeError(
                        "matsubara rule: weights cancel below roundoff "
                        f"(amplification {max(amplification):.3e} at rel_tol {rel_tol:g})")
                certificate = dict(
                    status="PASS", rel_tol=rel_tol, beta_ry_inv=beta, delta_max_ry=delta,
                    n_indices=idx.tolist(), even_error=errors[:, 0].tolist(),
                    odd_error=errors[:, 1].tolist(), amplification=amplification,
                    norm="sup over y in [0, delta_max] relative to each target's own sup",
                    grid=dict(check_points=int(y_check.size), fit_points=int(y_fit.size),
                              refinement="65 points between the neighbours of the 4 largest samples"),
                    method=("graded composite Gauss-Legendre pool on (0, beta/2] "
                            f"(order {order}, {panels} panels, {pool.size} nodes), column-pivoted QR "
                            "selection, least-squares weights per frequency"),
                    kernel="bosonic KMS: int_0^beta e^{i nu tau} f_m(1-f_n) e^{-x tau} = (f_m-f_n)/(x - i nu)")
                result = dict(t=tau, weights=W, nu_ry=nus, n_indices=idx, beta_ry_inv=beta,
                              delta_max_ry=delta, node_count=int(tau.size), certificate=certificate)
                digest = hashlib.sha256()
                for key in ("t", "weights"):
                    array = np.ascontiguousarray(result[key])
                    digest.update(key.encode())
                    digest.update(str(array.shape).encode())
                    digest.update(array.tobytes())
                result["node_digest"] = digest.hexdigest()
                return result
            n += 2 if n < 32 else max(2, n // 8)
    raise RuntimeError(
        f"matsubara rule: no node set up to {MAX_NODES} nodes meets rel_tol {rel_tol:g} "
        f"(beta {beta:g}, delta_max {delta:g}, indices {idx.tolist()})")

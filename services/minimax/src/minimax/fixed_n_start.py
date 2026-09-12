"""How many nodes a uniform box rule needs, and where to put them.

``build_uniform_rule``'s reduction starts at the interpolatory rank and removes
nodes one at a time until its budget runs out, so the count it ships depends on
the wall clock: on the widest corpus boxes 120 s removed nothing at all and the
rule went out at the full rank.  The two functions here replace that search on
crossing boxes -- predict the count, place exactly that many nodes where
certified rules actually put them, and polish once.

Both are empirical, fitted on certified rules from the run below, and both are
only ever used as a STARTING POINT: nothing here is trusted, the boundary
certificate still decides, and a miss costs planning wall and nothing else.

``predict_nodes`` (crossing).  With m the short side and M the long side of the
box in ``im_lo`` units, a rule needs time support ``ln(c/eps)/im_lo`` to resolve
the peak and ``W T / 2 pi`` nodes to cover an effective width W.  W is the full
width ``2m + (M - m)`` only while the box is nearly symmetric; once one side is
long its nodes leave the real axis and damp it, so the far side saturates.
110 converged boxes, 51 geometry clusters, 5-fold clustered CV: median ratio
0.971, p95 1.176, max 1.296.  The sign-definite law is kept for callers that
want an estimate, but the builder does not use it -- see below.

``start_param`` (crossing).  Measured on 252 reduced certified rules: they end
at the amplitude floor rather than the horizon (last node at
``Re s rate / ln(1/eps) = 1.01``); ``|w| / spacing = 1`` in the interior, so the
node DENSITY carries the structure and the weights follow; the spacing is the
live-band Nyquist spacing times ``c(h)``, which rises 1 -> 2 from head to tail;
and ``Im s`` takes the sign that damps the wider real edge, pinned at the
off-ray cap wherever the cap narrows the live band, ramping from zero over the
first ``K_r`` nodes and falling to 0.46 cap over the last ``K_f``.  The
placement is the fixed point of "index -> Im profile -> live band -> density ->
index"; six sweeps converge it, in about 10 ms.

Evidence: runs/DEV/327_uniform_rule_fixed_n_2026-09-11 (lanes ``nlaw`` and
``struct``, and ``results/e7`` for the combination).  One production polish
certifies 18 of 21 reduced crossing boxes at the predicted count, 20 at 1.03x
and 21 at 1.06x, against 5/6/7 for the pivoted-QR start it replaces.

Not used on sign-definite boxes.  Their reduction already finishes in 1-4 s well
below the budget, and the fixed-N path measured +37 % nodes on the 16 corpus
sign-definite boxes for twice the wall: there is nothing to buy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["predict_nodes", "start_param"]

# ------------------------------------------------------- node-count law (lane NLAW)
# crossing, model X7_satfar_log
_C0 = 1.5147174154894905
_C1 = 1.0424620255224675
_C_EPS = 0.08635737098310855          # exp(-2.4490674198737836)
_U0 = 0.8718444576860356              # exp(-0.1371536310714858)
_C2 = 1.4285309222563163
# sign-definite, model S8_span_H
_A0 = 0.7487135042435832
_A1 = 0.11699531406225654
_A2 = 0.08245582803404355
_A3 = 0.17642124415065782
_A4 = 0.06153762000946991
# crossing box in the RELATIVE currency is unmeasured; the builder's own
# docstring measures +50 % nodes on the Na conduction box (76 -> 115)
_RELATIVE_CROSSING_PENALTY = 1.5
_TINY = 1.0e-3                        # the builder's floor on a vanishing side

# ------------------------------------------------------- placement (lane STRUCT, v1)
# c(h): local spacing in live-Nyquist units against h = Re s rate / ln(1/eps)
_CH = np.array([0.01, 0.035, 0.075, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05])
_C_SYM = np.array([0.66, 0.90, 0.95, 0.975, 1.00, 1.03, 1.37, 1.54, 1.80, 2.07, 2.03, 1.91, 1.90])
_C_ASYM = np.array([0.57, 1.14, 1.28, 1.37, 1.45, 1.55, 1.69, 1.86, 2.05, 2.04, 1.88, 1.81, 1.80])
_BETA = 0.25            # first node at this many local spacings
_KR_THIN = 1.2          # Im ramp length in nodes, / sqrt(N)
_KR_TALL = 0.8
_KR_TALL_B = 3.0
_KF = 0.45              # Im fall length in nodes, / sqrt(N)
_KF_B = 0.3
_V_END = 0.46           # Im / cap at the last node
_H_LAST = 1.01          # last node, in units of ln(1/eps)/rate
_WTOL = 0.02            # live-band tolerance for preferring the smaller |Im s|
_NSIG = 25
_SWEEPS = 6


def predict_nodes(box, eps, relative=None):
    """Nodes ``build_uniform_rule(box, eps, relative=...)`` should need.

    A central estimate: the median ratio to the true count is ~1, so a builder
    that wants one shot should ask for a few percent more (the 95 % level of
    ``N_true / N_pred`` is 1.18 crossing, 1.23 sign-definite).
    """
    re_lo, re_hi, im_lo, im_hi = (float(v) for v in box)
    if not (np.isfinite([re_lo, re_hi, im_lo, im_hi]).all() and re_lo <= re_hi
            and 0.0 < im_lo <= im_hi):
        raise ValueError(f"invalid support box {box!r}")
    if not (0.0 < float(eps) < 1.0):
        raise ValueError(f"invalid eps {eps!r}")
    lo, hi, height = re_lo / im_lo, re_hi / im_lo, im_hi / im_lo
    crossing = not (lo > 0.0 or hi < 0.0)
    if relative is None:
        relative = not crossing

    if crossing:
        m = max(min(hi, -lo), _TINY)
        big = max(max(hi, -lo), _TINY)
        spread = big - m
        support = max(np.log(_C_EPS / eps), 0.25)
        width = 2.0 * m + spread / (1.0 + spread / (_U0 * m)) + _C2 * np.log1p(spread / m)
        n = _C0 + _C1 * width * support / (2.0 * np.pi)
        if relative:
            n *= _RELATIVE_CROSSING_PENALTY
    else:
        gap = max(min(abs(lo), abs(hi)), _TINY)
        far = max(abs(lo), abs(hi))
        rc = np.hypot(far, height) / np.hypot(gap, 1.0)
        span = np.arctan2(height, gap) - np.arctan2(1.0, far)
        ln_eps = np.log(1.0 / eps)
        n = (_A0 + (_A1 + _A2 * span) * np.log(rc) * ln_eps + _A3 * ln_eps
             + _A4 * np.log(height) * ln_eps)
    return int(max(2, round(float(n))))


def _boundary(box, phase, n=200):
    re_lo, re_hi, im_lo, im_hi = (float(v) for v in box)
    x = np.linspace(re_lo, re_hi, n)
    y = np.geomspace(im_lo, max(im_hi, im_lo * 1.0001), 24)
    d = np.concatenate([x + 1j * im_lo, x + 1j * im_hi, re_lo + 1j * y, re_hi + 1j * y])
    return phase * d


def _live_band(tau, sigma, dp, lam, floor):
    """Re d' extent of the family members still alive at each node."""
    live = tau[:, None] * dp.imag[None, :] + sigma[:, None] * dp.real[None, :] <= lam
    re = np.where(live, dp.real[None, :], np.nan)
    with np.errstate(invalid="ignore"):
        width = np.nanmax(re, 1) - np.nanmin(re, 1)
    return np.maximum(np.nan_to_num(width), floor)


def start_param(box, eps, theta, horizon, im_lo_s, im_hi_s, n):
    """``n`` ray positions ``s`` for a crossing box: ``Re s`` in ``(0, horizon)``
    and ``Im s`` inside the off-ray caps ``[im_lo_s, im_hi_s]``, ordered.

    ``theta`` and ``horizon`` are the builder's chosen ray angle and ``S``.
    """
    n = int(n)
    re_lo, re_hi, im_lo, im_hi = (float(v) for v in box)
    eps, horizon = float(eps), float(horizon)
    lam = np.log(10.0 / eps)
    rate = lam / horizon
    tau_end = min(_H_LAST * np.log(1.0 / eps) / rate, 0.999 * horizon)
    dp = _boundary(box, np.exp(-1j * theta))
    floor = (dp.real.max() - dp.real.min()) / 50.0
    cap_lo, cap_hi = 0.999 * float(im_lo_s), 0.999 * float(im_hi_s)
    k_r = (_KR_THIN * np.sqrt(n) if im_hi / im_lo < 1.1
           else _KR_TALL * np.sqrt(n) + _KR_TALL_B)
    k_f = max(2.0, _KF * np.sqrt(n) + _KF_B)
    # asymmetry picks the spacing profile: z is the wide-edge damping at the cap
    asym = max(re_hi, -re_lo) / (re_hi - re_lo)
    z = 3.0 * asym / max(1.0 - asym, 1.0e-6) / lam
    mix = float(np.clip((z - 0.8) / 0.8, 0.0, 1.0))
    ctab = (1.0 - mix) * _C_SYM + mix * _C_ASYM

    tau = np.linspace(0.0, tau_end, 1200)
    c = np.interp(tau * rate / np.log(1.0 / eps), _CH, ctab)
    sig_grid = np.sort(np.concatenate([np.linspace(cap_lo, cap_hi, _NSIG), [0.0]]))
    # pointwise Im target: the smallest |Im s| within _WTOL of the narrowest band
    bands = np.stack([_live_band(tau, np.full(tau.size, g), dp, lam, floor)
                      for g in sig_grid], 1)
    within = bands <= (1.0 + _WTOL) * bands.min(1)[:, None]
    sig_target = sig_grid[np.argmin(np.where(within, np.abs(sig_grid)[None, :], np.inf), 1)]

    index = np.linspace(0.0, n - 1.0, tau.size)
    sigma = np.zeros_like(tau)
    for _sweep in range(_SWEEPS):
        prev = 0.0
        for i in range(tau.size):                       # ramp-limited Im profile
            step = ((abs(cap_lo) if sig_target[i] < 0 else cap_hi) / k_r
                    * max(index[i] - (index[i - 1] if i else 0.0), 0.0))
            prev = float(np.clip(sig_target[i], prev - step, prev + step)) if i else 0.0
            sigma[i] = prev
        sigma = sigma * (_V_END + (1.0 - _V_END) * np.sin(
            0.5 * np.pi * np.clip((n - 1.0 - index) / k_f, 0.0, 1.0)) ** 2)
        density = _live_band(tau, sigma, dp, lam, floor) / (2.0 * np.pi * c)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1])
                                               * np.diff(tau))])
        index = (n - 1.0 + _BETA) * cum / max(cum[-1], 1.0e-300) - _BETA
    tau_k = np.clip(np.interp(np.arange(n, dtype=float), index, tau),
                    1.0e-4 * tau_end, 0.999 * horizon)
    return tau_k + 1j * np.clip(np.interp(tau_k, tau, sigma), cap_lo, cap_hi)

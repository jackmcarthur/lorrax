"""Measure-independent generalized-Gauss time rules for ``1/d`` on a box.

The delivered Sigma executor needs, per product window, complex time nodes
``t_k`` and weights ``w_k`` with ``1/d ~= sum_k w_k exp(i t_k d)`` for every
denominator ``d = omega - z`` the window can produce.  Here ``Im d >= eta``
always and ``Re d`` lies in a known interval, so the only input is the
support box ``[re_lo, re_hi] x [im_lo, im_hi]``; the histogram of ``d``
inside the box never enters.  That makes the rule safe by construction:
its accuracy is a uniform bound on the box, not a mass-weighted average
that a spiky measure can defeat.

Construction (one recipe for crossing and sign-definite windows alike):

1. ``1/d = -i int_0^inf exp(i t d) dt`` is taken along the ray
   ``t = s exp(-i theta)``.  ``theta`` is scanned over the interval on
   which every ``exp(i t d)`` on the box still decays and the angle with
   the smallest numerical rank of the family ``{exp(i t(s) d)}`` wins.
   A symmetric crossing box gets ``theta ~ 0`` (real time, Lorentzian
   family); a sign-definite box rotates toward imaginary time (Laplace
   family, Braess-Hackbusch regime); a nearly sign-definite box lands in
   between.  No angle dial.
2. On the chosen ray the family is sampled on a Chebyshev grid, its SVD
   basis ``u_j(s)`` is truncated at ``eps/10`` and an interpolatory rule
   (pivoted QR on the basis) is formed.
3. The rule is reduced toward the Gauss count by removing one node at a
   time and re-solving the moment equations ``sum_k w_k u_j(s_k) = int u_j``
   with a Levenberg-Marquardt Gauss-Newton step (Bremer, Gimbutas and
   Rokhlin, SIAM J. Sci. Comput. 32, 2010).  A candidate is kept only while
   the rule's sup error on the box stays below ``eps`` (relative to the
   ``1/im_lo`` kernel peak) and its term-cancellation ratio below
   ``kappa_cap``.

The node count follows the band-limited theory: about
``0.5 * (B/eta) * ln(10/eps) / pi`` for a crossing box of real width
``B``, and ``O(log(re_hi/re_lo))`` for a sign-definite box.  Only ``eps``
and the box are inputs; ``eta`` is the box's ``im_lo``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.linalg import lstsq, svd
from scipy.linalg import qr as _pivoted_qr

__all__ = ["UniformRule", "build_uniform_rule", "box_samples", "rule_sup_error"]


# ----------------------------------------------------------------- support cloud
def _re_line(lo, hi, h, near):
    parts = []
    a, b = max(lo, -near), min(hi, near)
    if a < b:
        parts.append(np.arange(a, b + 0.5 * h, h))
    for sign, edge in ((1.0, hi), (-1.0, lo)):
        if sign * edge > near:
            n = int(1.5 * np.log(sign * edge / near) * near / h) + 8
            parts.append(sign * np.geomspace(near, sign * edge, n))
    re = np.concatenate(parts) if parts else np.array([lo, hi])
    return np.sort(np.unique(np.concatenate([re, [lo, hi]])))


def box_samples(re_lo, re_hi, im_lo, im_hi, per_unit=5.0, n_im=6, near=30.0):
    """Sample cloud on the box.  Along ``Re d`` the spacing is ``im/per_unit``
    within ``near*im_lo`` of zero (the rule's error oscillates on the scale
    ``2 pi im / ln(1/eps)``) and geometric beyond; each ``Im`` level gets
    its own density, log-spaced from ``im_lo`` to ``im_hi``."""
    im = np.geomspace(im_lo, max(im_hi, im_lo * 1.0001), n_im)
    out = []
    for v in im:
        h = min(v, 4.0 * im_lo) / per_unit
        out.append(_re_line(re_lo, re_hi, h, near * im_lo) + 1j * v)
    return np.concatenate(out)


def _legal_angles(d, margin=0.25):
    """Ray angles whose slowest family member still decays at a rate of at
    least ``margin * min(Im d)``:  ``|exp(i s e^{-i th} d)| =
    exp(-s (cos th Im d - sin th Re d))``."""
    th = np.deg2rad(np.arange(-89.0, 89.5, 1.0))
    rate = np.min(np.cos(th)[:, None] * d.imag[None, :]
                  - np.sin(th)[:, None] * d.real[None, :], axis=1)
    ok = rate >= margin * d.imag.min()
    return th[ok], rate[ok]


# ----------------------------------------------------------------- Chebyshev grid
def _cheb_grid(S, n):
    """Chebyshev-Lobatto points on ``[0, S]`` with Clenshaw-Curtis weights."""
    N = n - 1
    x = np.cos(np.pi * np.arange(n) / N)
    m = np.arange(1, N // 2 + 1)
    b = np.where(2 * m < N, 1.0, 0.5) / (4.0 * m * m - 1.0)
    acc = np.cos(2.0 * np.pi * np.outer(np.arange(N + 1), m) / N) @ b
    c = (2.0 / N) * (1.0 - 2.0 * acc)
    c[0] *= 0.5
    c[N] *= 0.5
    return 0.5 * S * (1.0 - x), 0.5 * S * c


def _grid_size(d, theta, S, eps=1e-5, points_per_half_wave=3.0, cap=6000):
    """Chebyshev points needed on ``[0, S]`` for the family on this ray.

    A member with oscillation frequency ``nu`` and decay rate ``lam`` is alive
    for ``min(S, ln(10/eps)/lam)`` and needs ``points_per_half_wave`` points per
    half wave over that lifetime (on a rotated ray the fast members are also
    the fast-decaying ones, so this is far below ``nu_max * S``).  The second
    term keeps the Chebyshev spacing near ``s = 0``, ``S (pi/N)^2 / 2``, at or
    below a half wave of the fastest member.
    """
    nu = np.abs(np.cos(theta) * d.real + np.sin(theta) * d.imag)
    lam = np.maximum(np.cos(theta) * d.imag - np.sin(theta) * d.real, 1e-300)
    life = np.minimum(S, np.log(10.0 / eps) / lam)
    interior = points_per_half_wave * np.max(nu * life) / np.pi
    near_zero = np.sqrt(points_per_half_wave * np.pi * S * np.max(nu) / 2.0)
    return int(min(max(interior, near_zero) + 100, cap))


def _family(d, theta, s):
    return np.exp(1j * d[:, None] * (s * np.exp(-1j * theta))[None, :])


def _choose_angle(d, eps, scan=(-85, -70, -55, -40, -20, 0, 20, 40, 55, 70, 85)):
    """Minimum eps-rank ray angle over the legal interval (thinned cloud)."""
    th_ok, rate_ok = _legal_angles(d)
    thin = d[::max(1, d.size // 500)]
    best = None
    for deg in scan:
        i = int(np.argmin(np.abs(th_ok - np.deg2rad(deg))))
        if abs(th_ok[i] - np.deg2rad(deg)) > np.deg2rad(0.6):
            continue
        S = np.log(10.0 / eps) / rate_ok[i]
        s, w = _cheb_grid(S, _grid_size(thin, th_ok[i], S, eps))
        sv = svd(_family(thin, th_ok[i], s) * np.sqrt(w)[None, :], compute_uv=False)
        r = int(np.sum(sv > eps * sv[0]))
        if best is None or r < best[0]:
            best = (r, th_ok[i], S, rate_ok[i])
    if best is None:
        i = len(th_ok) // 2
        best = (0, th_ok[i], np.log(10.0 / eps) / rate_ok[i], rate_ok[i])
    return best


# ----------------------------------------------------------------- ray family
class _RayFamily:
    """SVD basis of ``{exp(i t(s) d)}`` on one ray, in Chebyshev form."""

    def __init__(self, d, theta, S, eps):
        self.d, self.theta, self.S = d, theta, S
        self.phase = np.exp(-1j * theta)
        n_s = _grid_size(d, theta, S, eps)
        s, ws = _cheb_grid(S, n_s)
        M = _family(d, theta, s) * np.sqrt(ws)[None, :]
        _P, sig, Qh = svd(M, full_matrices=False)
        r = int(np.sum(sig > eps * sig[0]))
        self.r = r
        Ug = Qh[:r] / np.sqrt(ws)[None, :]                # u_j on the grid
        self.m = Ug @ ws                                  # moments int u_j ds
        N = n_s - 1
        V = np.cos(np.pi * np.outer(np.arange(N + 1), np.arange(N + 1)) / N)
        scale = np.full(N + 1, 2.0 / N)
        scale[0] = scale[N] = 1.0 / N
        wj = np.ones(N + 1)
        wj[0] = wj[N] = 0.5
        self.A = (Ug * wj[None, :]) @ V * scale[None, :]  # Chebyshev coefficients
        self.N = N
        self.s_grid = s

    def _T(self, s):
        x = 1.0 - 2.0 * s / self.S
        k = np.arange(self.N + 1)
        th = np.arccos(x)
        T = np.cos(np.outer(k, th))
        sinth = np.sin(th)
        sinth = np.where(np.abs(sinth) < 1e-12, 1e-12, sinth)
        dT = k[:, None] * np.sin(np.outer(k, th)) / sinth[None, :] * (-2.0 / self.S)
        return T, dT

    def U(self, s):
        return self.A @ self._T(s)[0]

    def U_dU(self, s):
        T, dT = self._T(s)
        return self.A @ T, self.A @ dT

    def interpolatory(self):
        Ug = self.U(self.s_grid.astype(complex))
        _, _, piv = _pivoted_qr(Ug, mode="economic", pivoting=True)
        s = np.sort(self.s_grid[piv[:self.r]]).astype(complex)
        w = lstsq(self.U(s), self.m, rcond=None)[0]
        return s, w

    def newton(self, s, w, im_lo, im_hi, steps=40):
        """Damped Gauss-Newton on the moment equations; complex nodes and
        weights (4n real unknowns against 2r real equations)."""
        F = self.U(s) @ w - self.m
        tol = 1e-12 * np.linalg.norm(self.m)
        lam = 1e-6
        for _ in range(steps):
            nF = np.linalg.norm(F)
            if nF < tol:
                break
            U, dU = self.U_dU(s)
            dU = dU * w[None, :]
            n = s.size
            J = np.concatenate([dU, U], 1)
            Jr = np.block([[J.real, -J.imag], [J.imag, J.real]])
            Fr = np.concatenate([F.real, F.imag])
            D = np.linalg.norm(Jr, axis=0)
            D = np.where(D > 0, D, 1.0)
            improved = False
            for _lm in range(8):
                Jd = Jr / D[None, :]
                p = lstsq(np.vstack([Jd, np.sqrt(lam) * np.eye(Jd.shape[1])]),
                          np.concatenate([-Fr, np.zeros(Jd.shape[1])]),
                          rcond=None)[0] / D
                ds = p[:n] + 1j * p[2 * n:3 * n]
                dw = p[n:2 * n] + 1j * p[3 * n:]
                s_new = s + ds
                s_new = (s_new.real.clip(1e-9 * self.S, self.S)
                         + 1j * s_new.imag.clip(im_lo, im_hi))
                w_new = w + dw
                F_new = self.U(s_new) @ w_new - self.m
                if np.linalg.norm(F_new) < nF:
                    s, w, F = s_new, w_new, F_new
                    lam = max(lam / 5.0, 1e-12)
                    improved = True
                    break
                lam *= 10.0
            if not improved:
                break
        return s, w

    def reduce(self, s, w, im_lo, im_hi, accepted, batch_frac=0.05, deadline=None):
        """Remove nodes while ``accepted(s, w)`` holds.  Candidates are ranked
        by the leave-one-out residual gain ``|w_k|^2 / [(U^H U)^-1]_kk``."""
        best = (s.copy(), w.copy())
        batch = max(1, int(batch_frac * s.size))
        fails = 0
        while s.size > 2:
            if deadline is not None and time.perf_counter() > deadline:
                break
            U = self.U(s)
            G = np.linalg.pinv(U.conj().T @ U)
            score = np.abs(w) ** 2 / np.maximum(np.real(np.diag(G)), 1e-300)
            order = np.argsort(score)
            drop, protected = [], set()
            skip = fails if batch == 1 else 0
            for k in order:
                if k in protected:
                    continue
                if skip > 0:
                    skip -= 1
                    continue
                drop.append(int(k))
                protected.update((k - 1, k, k + 1))
                if len(drop) >= batch:
                    break
            s_t = np.delete(s, drop)
            w_t = lstsq(self.U(s_t), self.m, rcond=None)[0]
            s_t, w_t = self.newton(s_t, w_t, im_lo, im_hi)
            if not accepted(s_t, w_t):
                fails += 1
                if batch == 1 and fails >= 6:
                    break
                batch = max(1, batch // 2)
                continue
            fails = 0
            s, w = s_t, w_t
            best = (s.copy(), w.copy())
        return best

    def to_rule(self, s, w):
        """Time nodes and weights of ``1/d ~= sum w_k exp(i t_k d)``."""
        return self.phase * s, -1j * self.phase * w


def rule_sup_error(times, weights, d):
    """``(max |Q(d) - 1/d| * min Im d, max kappa)`` on the cloud ``d``."""
    A = np.exp(1j * d[:, None] * times[None, :])
    Q = A @ weights
    err = np.abs(Q - 1.0 / d)
    kappa = np.abs(A * weights[None, :]).sum(1) / np.maximum(np.abs(Q), 1e-300)
    return float(err.max() * d.imag.min()), float(kappa.max())


@dataclass(frozen=True)
class UniformRule:
    times: np.ndarray
    weights: np.ndarray
    box: tuple
    eps: float
    theta_deg: float
    rank: int
    sup_error: float
    kappa_max: float
    seconds: float

    @property
    def node_count(self) -> int:
        return int(self.times.size)

    def one_line(self) -> str:
        return (f"uniform box rule: {self.node_count} nodes, ray {self.theta_deg:.0f} deg, "
                f"rank {self.rank}, sup {self.sup_error:.2e} (eps {self.eps:g}), "
                f"kappa {self.kappa_max:.3g}, {self.seconds:.1f} s")


def build_uniform_rule(box, eps, *, im_cap=3.0, kappa_cap=1.0e4, trunc=10.0,
                       reduce=True, time_budget=None):
    """Rule for ``1/d`` on ``box = (re_lo, re_hi, im_lo, im_hi)`` with
    ``Im d > 0``.  ``eps`` is the sup error relative to the ``1/im_lo``
    peak.  ``im_cap`` bounds ``|Im t| * (box half-width)`` so no family
    member grows by more than ``exp(im_cap)`` off the ray.  ``time_budget``
    (seconds) stops the Gauss reduction early and returns the best rule so
    far; the interpolatory rule is always available."""
    re_lo, re_hi, im_lo, im_hi = map(float, box)
    if not (np.isfinite([re_lo, re_hi, im_lo, im_hi]).all() and re_lo <= re_hi
            and 0.0 < im_lo <= im_hi):
        raise ValueError(f"invalid support box {box!r}")
    t0 = time.perf_counter()
    d = box_samples(re_lo, re_hi, im_lo, im_hi)
    _r0, theta, S, _rate = _choose_angle(d, eps)
    fam = _RayFamily(d, theta, S, eps / trunc)
    s, w = fam.interpolatory()
    Bp = max(re_hi, 1e-3 * im_lo)
    Bm = max(-re_lo, 1e-3 * im_lo)
    # Off-ray excursions are also capped so the Chebyshev basis stays finite:
    # T_k grows like exp(k |Im x|), x = 1 - 2 s / S.
    off_ray = 20.0 * S / max(fam.N, 1)
    im_range = (max(-im_cap / Bp, -off_ray), min(im_cap / Bm, off_ray))

    def accepted(s_, w_):
        e_, k_ = rule_sup_error(*fam.to_rule(s_, w_), d)
        return e_ <= eps and k_ <= kappa_cap

    if reduce:
        deadline = None if time_budget is None else t0 + float(time_budget)
        s, w = fam.reduce(s, w, *im_range, accepted, deadline=deadline)
    times, weights = fam.to_rule(s, w)
    sup, kappa = rule_sup_error(times, weights, d)
    return UniformRule(
        times=times, weights=weights, box=(re_lo, re_hi, im_lo, im_hi),
        eps=float(eps), theta_deg=float(np.rad2deg(theta)), rank=int(fam.r),
        sup_error=sup, kappa_max=kappa, seconds=time.perf_counter() - t0)

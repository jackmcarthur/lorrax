"""Measure-independent generalized-Gauss time rules for ``1/d`` on a box (v5 reducer).

Drop-in for ``minimax.uniform_rule``.  The ray choice, Chebyshev SVD basis
and interpolatory start are unchanged; the Gauss reduction now works on the
cloud residual itself (Bremer-Gimbutas-Rokhlin reduction driven by a
variable-projection Levenberg-Marquardt solve of ``sum_k w_k exp(i t_k d) -
1/d`` on the sample cloud), because the truncated SVD model is exact only on
the ray: its dropped tail grows like ``exp(B |Im t|)`` at nodes off the ray,
so the moment residual could read 1e-14 while the box error was 1e-3.

Per window the inputs are the support box ``[re_lo, re_hi] x [im_lo, im_hi]``
(``im_lo = eta``) and ``eps``, the sup error relative to the ``1/eta`` peak.

1. **Ray angle.** ``t = s exp(-i theta)``; ``theta`` is scanned over the
   interval where every ``exp(i t d)`` on the box decays and the angle with
   the smallest numerical rank wins.  Symmetric crossing boxes get real time,
   sign-definite boxes rotate toward imaginary time (the Laplace family).
2. **Start.** Interpolatory rule from a pivoted QR of the ray family's SVD
   basis (``r`` nodes at ``eps/10``).
3. **Reduction.** Nodes are removed one at a time (batches while far above
   the target).  Candidates are ranked by the leave-one-out residual gain
   ``|w_k|^2 / [(A^H A + mu^2)^-1]_kk``; the best ``K`` get a short
   variable-projection solve, the best ``keep`` of those are solved to
   acceptance, and the accepted one with the smallest residual is kept.
   Weights are eliminated by penalised least squares (Tikhonov ``mu``
   pinning the cancellation ratio), ``Im s`` is parametrised as
   ``c + h tanh(y)`` so the off-ray cap is built into the model.  A candidate
   is kept while the sup error on the box stays below ``eps`` and the
   term-cancellation ratio below ``kappa_cap``.  ``time_budget`` bounds the
   reduction; the interpolatory rule (about 1.3-1.5x the Gauss count) is
   ready after about a second.

Counts: about ``0.5 r`` for a crossing box with ``r ~ 1.9 (B/eta)
ln(10/eps)/13.8``; ``O(log(re_hi/re_lo))`` for a sign-definite box.
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


def _im_levels(theta):
    """Log-spaced ``Im d`` levels needed on a ray: on real time the family
    only decays in ``Im d``, six levels resolve it; on a rotated ray it also
    oscillates in ``Im d`` (phase ``sin(theta) s Im d``), and the members
    alive longest sit at the small-``|Re d|`` edge with up to ~20 rad of
    phase across the box, so about four times as many levels are needed."""
    return 6 if abs(np.sin(theta)) < 0.15 else 24


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
    """Chebyshev points needed on ``[0, S]`` for the family on this ray: each
    member needs ``points_per_half_wave`` points per half wave over its
    lifetime ``min(S, ln(10/eps)/decay)`` (on a rotated ray the fast members
    are also the fast-decaying ones), and the near-zero spacing
    ``S (pi/N)^2 / 2`` must stay below a half wave of the fastest member."""
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


# ----------------------------------------------------------------- ray family (start rule)
class _RayFamily:
    """SVD basis of ``{exp(i t(s) d)}`` on one ray; provides the interpolatory start."""

    def __init__(self, d, theta, S, eps):
        self.d, self.theta, self.S = d, theta, S
        self.phase = np.exp(-1j * theta)
        n_s = _grid_size(d, theta, S, eps)
        s, ws = _cheb_grid(S, n_s)
        M = _family(d, theta, s) * np.sqrt(ws)[None, :]
        _P, sig, Qh = svd(M, full_matrices=False)
        r = int(np.sum(sig > eps * sig[0]))
        self.r = r
        Ug = Qh[:r] / np.sqrt(ws)[None, :]
        self.m = Ug @ ws
        N = n_s - 1
        V = np.cos(np.pi * np.outer(np.arange(N + 1), np.arange(N + 1)) / N)
        scale = np.full(N + 1, 2.0 / N)
        scale[0] = scale[N] = 1.0 / N
        wj = np.ones(N + 1)
        wj[0] = wj[N] = 0.5
        self.A = (Ug * wj[None, :]) @ V * scale[None, :]
        self.N = N
        self.s_grid = s

    def _T(self, s):
        x = 1.0 - 2.0 * s / self.S
        k = np.arange(self.N + 1)
        th = np.arccos(x)
        return np.cos(np.outer(k, th))

    def U(self, s):
        return self.A @ self._T(s)

    def interpolatory(self):
        # The Lobatto grid contains s = 0; a node there is a zero time node,
        # which the executor refuses, so the end points are never offered.
        cand = self.s_grid[1:-1]
        _, _, piv = _pivoted_qr(self.U(cand.astype(complex)), mode="economic", pivoting=True)
        s = np.sort(cand[piv[:self.r]]).astype(complex)
        w = lstsq(self.U(s), self.m, rcond=None)[0]
        return s, w

    def to_rule(self, s, w):
        """Time nodes and weights of ``1/d ~= sum w_k exp(i t_k d)``."""
        return self.phase * s, -1j * self.phase * w


# ----------------------------------------------------------------- cloud solver (reduction)
class _CloudFit:
    """Nonlinear least squares of the rule against ``1/d`` on the cloud.

    Residual ``E_i = sum_k w_k exp(i d_i t_k) - 1/d_i`` scaled by ``im_lo`` so
    ``|E|`` is relative to the ``1/eta`` peak.  Nodes ``t = phase * s`` with
    ``Re s`` in ``[0, S]`` and ``Im s`` in ``[im_lo, im_hi]``; weights are
    eliminated by penalised least squares (variable projection with the
    Kaufman Jacobian).
    """

    def __init__(self, d, phase, S, im_lo, im_hi, eps, w_ref, alpha=0.3):
        self.d, self.phase, self.S, self.im_lo, self.im_hi = d, phase, S, im_lo, im_hi
        self.eps = eps
        self.scale = d.imag.min()
        self.b = self.scale / d
        self.nb = np.linalg.norm(self.b)
        self.idp = 1j * d * phase
        # Tikhonov weight penalty: |mu w| ~ alpha*eps*|b| for weights of the
        # reference size, so runaway cancelling weights cost more than the
        # acceptance residual while normal weights cost less.
        self.mu = alpha * eps * self.nb / max(np.linalg.norm(w_ref), 1e-300)
        self.c = 0.5 * (im_lo + im_hi)
        self.h = 0.5 * (im_hi - im_lo)

    def A(self, s):
        return np.exp(self.idp[:, None] * s[None, :]) * self.scale

    def ls(self, s):
        A = self.A(s)
        n = s.size
        Aa = np.concatenate([A, self.mu * np.eye(n, dtype=complex)], 0)
        Q, R = np.linalg.qr(Aa)
        ba = np.concatenate([self.b, np.zeros(n, complex)])
        c = Q.conj().T @ ba
        w = np.linalg.solve(R, c)
        return w, ba - Q @ c, Q, A

    def _y_of(self, s):
        u = np.clip((s.imag - self.c) / max(self.h, 1e-300), -1.0 + 1e-3, 1.0 - 1e-3)
        return np.arctanh(u)

    def _s_of(self, re, y):
        return re.clip(0.0, self.S) + 1j * (self.c + self.h * np.tanh(y))

    def newton(self, s, steps=30, tol=1e-14, check=None, chunk=10, stall=1e-3):
        """Variable-projection Levenberg-Marquardt.  One Gram eigendecomposition
        per step gives the step for every damping.  Every ``chunk`` steps the
        optional ``check(s, w, res)`` may stop the solve (rule accepted), and
        the solve stops when the residual fell by less than ``stall`` over the
        chunk.  Returns nodes, weights, relative residual."""
        s = s.real.clip(0.0, self.S) + 1j * s.imag.clip(self.im_lo, self.im_hi)
        y = self._y_of(s)
        s = self._s_of(s.real, y)
        w, F, Q, A = self.ls(s)
        n = s.size
        lam = 1e-3
        n_last = np.linalg.norm(F)
        for it in range(steps):
            nF = np.linalg.norm(F)
            if nF < tol * self.nb:
                break
            if it > 0 and it % chunk == 0:
                if check is not None and check(s, w, np.linalg.norm(F[:self.d.size]) / self.nb):
                    break
                if nF > (1.0 - stall) * n_last:
                    break
                n_last = nF
            Jc = np.concatenate([-(self.idp[:, None] * A) * w[None, :],
                                 np.zeros((n, n), complex)], 0)
            Jc -= Q @ (Q.conj().T @ Jc)
            dIm = self.h / np.cosh(y) ** 2
            Jr = np.block([[Jc.real, -Jc.imag * dIm[None, :]],
                           [Jc.imag, Jc.real * dIm[None, :]]])
            Fr = np.concatenate([F.real, F.imag])
            D = np.linalg.norm(Jr, axis=0)
            D = np.where(D > 0, D, 1.0)
            Jd = Jr / D[None, :]
            ev, V = np.linalg.eigh(Jd.T @ Jd)
            sj2 = np.maximum(ev, 0.0)
            g = V.T @ (Jd.T @ (-Fr))
            improved = False
            for _lm in range(12):
                p = (V @ (g / (sj2 + lam))) / D
                y_new = np.clip(y + np.clip(p[n:], -3.0, 3.0), -8.0, 8.0)
                s_new = self._s_of(s.real + p[:n], y_new)
                w_new, F_new, Q_new, A_new = self.ls(s_new)
                if np.linalg.norm(F_new) < nF:
                    s, w, F, Q, A, y = s_new, w_new, F_new, Q_new, A_new, y_new
                    lam = max(lam / 3.0, 1e-9)
                    improved = True
                    break
                lam *= 4.0
            if not improved:
                break
        return s, w, np.linalg.norm(F[:self.d.size]) / self.nb

    def loo_scores(self, s, w):
        """Residual increase when node k is dropped and the weights re-solved."""
        A = self.A(s)
        G = np.linalg.pinv(A.conj().T @ A + self.mu ** 2 * np.eye(s.size))
        return np.abs(w) ** 2 / np.maximum(np.real(np.diag(G)), 1e-300)

    def _solve_pick(self, starts, ok, nstep, keep, rank_steps=8):
        """Successive halving: every start gets a short solve, the ``keep``
        best continue to ``nstep`` steps with early exit on acceptance; the
        accepted solve with the smallest residual wins (or None)."""
        stop = lambda a, b, r: ok(a, b) and r <= self.eps
        short = []
        for s0 in starts:
            s_t, w_t, res = self.newton(s0, steps=rank_steps, chunk=100)
            short.append((res, s_t, w_t))
        short.sort(key=lambda z: z[0])
        found = None
        for res, s_t, w_t in short[:keep]:
            if not (ok(s_t, w_t) and res <= self.eps):
                s_t, w_t, res = self.newton(s_t, steps=nstep, check=stop)
            if ok(s_t, w_t) and (found is None or res < found[2]):
                found = (s_t, w_t, res)
        return found

    def reduce(self, s, ok, deadline, batch_frac=0.10, K=6, nstep=60, keep=2):
        """Gauss-type reduction with lookahead, bounded by ``deadline``.

        Far above the target the batch move drops ``batch`` well separated
        low-score nodes at once and solves to the optimum (an early exit at
        acceptance leaves a marginal state the single removals inherit); each
        failure halves the batch.  At batch 1 the ``K`` best leave-one-out
        candidates are tried by successive halving, then the next ``2K``;
        then the reduction stops."""
        s, w, _res = self.newton(s.astype(complex), steps=nstep)     # polish the start
        if not ok(s, w):
            return None                                             # caller keeps the start
        best = (s.copy(), w.copy())
        batch = max(1, int(batch_frac * s.size))
        while s.size > 2 and time.perf_counter() < deadline:
            order = np.argsort(self.loo_scores(s, w))
            if batch > 1:
                drop, protected = [], set()
                for k in order:
                    if k in protected:
                        continue
                    drop.append(int(k))
                    protected.update((k - 1, k, k + 1))
                    if len(drop) >= batch:
                        break
                s_t, w_t, _ = self.newton(np.delete(s, drop), steps=3 * nstep)
                if ok(s_t, w_t):
                    s, w = s_t, w_t
                    best = (s.copy(), w.copy())
                else:
                    batch //= 2
                continue
            found = None
            for lo, hi, ns in ((0, K, nstep), (K, 3 * K, 2 * nstep)):
                starts = [np.delete(s, k) for k in order[lo:hi]]
                if starts:
                    found = self._solve_pick(starts, ok, ns, keep)
                if found is not None:
                    break
            if found is None:
                break
            s, w = found[0], found[1]
            best = (s.copy(), w.copy())
        return best


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
    ``Im d > 0``.  ``eps`` is the sup error relative to the ``1/im_lo`` peak.
    ``im_cap`` bounds ``|Im t| * (box half-width)`` so no family member grows
    by more than ``exp(im_cap)`` off the ray.  ``time_budget`` (seconds)
    bounds the Gauss reduction and returns the best rule at the deadline; the
    interpolatory rule is always available."""
    re_lo, re_hi, im_lo, im_hi = map(float, box)
    if not (np.isfinite([re_lo, re_hi, im_lo, im_hi]).all() and re_lo <= re_hi
            and 0.0 < im_lo <= im_hi):
        raise ValueError(f"invalid support box {box!r}")
    t0 = time.perf_counter()
    d = box_samples(re_lo, re_hi, im_lo, im_hi)
    _r0, theta, S, _rate = _choose_angle(d, eps)
    n_im = _im_levels(theta)
    if n_im != 6:
        d = box_samples(re_lo, re_hi, im_lo, im_hi, n_im=n_im)
    # Acceptance is judged on a cloud finer than the fit cloud in both
    # directions, so a rule that is exact between fit samples only is
    # rejected (property test: 17 of 80 random rotated-ray boxes failed a
    # finer check before this).
    d_check = box_samples(re_lo, re_hi, im_lo, im_hi, per_unit=8.0, n_im=2 * n_im)
    fam = _RayFamily(d, theta, S, eps / trunc)
    s, w = fam.interpolatory()
    if reduce:
        Bp = max(re_hi, 1e-3 * im_lo)
        Bm = max(-re_lo, 1e-3 * im_lo)
        im_lo_s, im_hi_s = max(-im_cap / Bp, -0.3 * S), min(im_cap / Bm, 0.3 * S)
        # the cloud solver works in the executor's convention t = phase*s with
        # weights -i*phase*w: the sup test below uses exactly that map
        fit = _CloudFit(d, fam.phase, S, im_lo_s, im_hi_s, eps, w_ref=w)

        def ok(s_, w_):
            e_, k_ = _score_cloud(fit, s_, w_, d_check)
            return e_ <= eps and k_ <= kappa_cap

        deadline = t0 + (float(time_budget) if time_budget is not None else 1e30)
        red = fit.reduce(s, ok, deadline)
        if red is not None:
            s, w_fit = red
            times, weights = fam.phase * s, w_fit          # A w - b = scale (Q - 1/d): w is the rule weight
        else:
            times, weights = fam.to_rule(s, w)
    else:
        times, weights = fam.to_rule(s, w)
    sup, kappa = rule_sup_error(times, weights, d_check)
    return UniformRule(
        times=times, weights=weights, box=(re_lo, re_hi, im_lo, im_hi),
        eps=float(eps), theta_deg=float(np.rad2deg(theta)), rank=int(fam.r),
        sup_error=sup, kappa_max=kappa, seconds=time.perf_counter() - t0)


def _score_cloud(fit, s, w, d):
    """Sup error and kappa of the cloud-solver state ``(s, w)`` in rule form."""
    return rule_sup_error(fit.phase * s, w, d)

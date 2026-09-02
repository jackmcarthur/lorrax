"""Measure-independent generalized-Gauss time rules for ``1/d`` on a box.

One rule per product window.  The only inputs are the window's support box
``[re_lo, re_hi] x [im_lo, im_hi]`` in the denominator ``d = omega - z``
(``im_lo = eta``) and ``eps``, a sup bound on the box in the currency that
matches Sigma's error: the RELATIVE error ``|d| |Q - 1/d|`` on a
sign-definite box (its terms are far from resonance and large, semicore
states at ``|d| ~ 200 eta`` included), the error relative to the ``1/eta``
peak, ``eta |Q - 1/d|``, on a crossing box (where ``1/d`` is bounded only by
the peak and a relative bound at the edges would cost +50 % nodes for no
delivered gain).  See ``build_uniform_rule``.
Nothing about the spectral measure enters: the same box gives the same rule
on every deck, which is what makes the result reproducible and cacheable
(rules are keyed by ``(box, eps)``) and what removed the campaign's failure
mode of histogram-weighted fits that missed a low-mass state at E_F.

The identity being discretised is ``1/d = -i int_0^inf exp(i t d) dt``
(``Im d > 0``), so ``sum_k w_k exp(i t_k d)`` with the time nodes ``t_k`` is
one FFT convolution per node in the executor; nodes are the runtime
currency.  Boxes with ``Im d < 0`` are NOT handled here: the caller
conjugates (``times -> -conj(times)``, ``weights -> conj(weights)``), which
flips ``Im d`` and leaves the real corners alone.

1. **Ray angle.** ``t = s exp(-i theta)``; ``theta`` is scanned over the
   interval where every ``exp(i t d)`` on the box decays and the angle with
   the smallest numerical rank wins.  Symmetric crossing boxes get real time,
   sign-definite boxes rotate toward imaginary time (the Laplace family).
2. **Start.** Interpolatory rule from a pivoted QR of the ray family's SVD
   basis (``r`` nodes at ``eps/10``), ready after about a second.  In the
   relative currency the basis and the least squares carry the cloud's
   log-density (every decade of ``|d|`` counts once) and the start is
   polished by a few Lawson reweighting rounds toward the sup, because the
   L2 optimum leaves the near corner 3-7x above ``eps``.
3. **Reduction.** Bremer-Gimbutas-Rokhlin style: nodes are removed one at a
   time (batches while far above the target), the survivors re-solved by a
   variable-projection Levenberg-Marquardt on the CLOUD residual
   ``sum_k w_k exp(i t_k d) - 1/d``.  Candidates are ranked by the
   leave-one-out residual gain ``|w_k|^2 / [(A^H A + mu^2)^-1]_kk``; the best
   ``K`` get a short solve, the best ``keep`` of those are solved to
   acceptance, and the accepted one with the smallest residual is kept.
   Weights are eliminated by penalised least squares (Tikhonov ``mu`` pinning
   the cancellation ratio), ``Im s`` is parametrised as ``c + h tanh(y)`` so
   the off-ray cap is built into the model.  A candidate is kept while the
   sup error on a FINER check cloud stays below ``eps`` and the
   term-cancellation ratio below ``kappa_cap``.  ``time_budget`` bounds the
   reduction and returns the best accepted rule at the deadline.

Why the reduction works on the cloud and not on the SVD moments: the
truncated SVD model is exact only on the ray, and its dropped tail grows like
``exp(B |Im t|)`` at nodes off the ray, so a moment residual read 1e-14 while
the box error was 1e-3.  Every acceptance decision below is a sup norm on
sampled denominators for that reason.

Counts: about ``0.5 r`` for a crossing box with ``r ~ 1.9 (B/eta)
ln(10/eps)/13.8`` (``B`` the real width; ~0.67 r at a 15 s budget, ~0.5 r at
120 s); ``O(log(re_hi/re_lo))`` for a sign-definite box.  The count is set by
the box width in units of ``eta``, so nothing in this module can go below
the geometry of the window it is given.
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
    """Real-axis sample line: spacing ``h`` within ``near`` of zero, geometric
    beyond, both end points always present.

    The rule's error oscillates on the scale ``2 pi Im d / ln(1/eps)`` only
    near ``Re d = 0``; far out the family is smooth on the scale of ``|Re d|``
    itself, so geometric spacing (about 1.5 points per e-fold, plus 8) resolves
    it.  Tempting, and why not: a uniform line over the whole box.  The
    sign-definite tails on a metal reach ``Re d ~ -8.5 Ry`` at ``eta`` 0.018
    Ry; a uniform line at the near-zero density would be ~1e5 points per Im
    level and the solver's matrices would not fit in cache, for no gain."""
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
    """Log-spaced ``Im d`` levels needed on a ray.  On real time the family
    only decays in ``Im d`` and six levels resolve it.  On a rotated ray it
    also oscillates in ``Im d`` (phase ``sin(theta) s Im d``), and the members
    alive longest sit at the small-``|Re d|`` edge with up to ~20 rad of phase
    across the box, so about four times as many levels are needed.

    Tempting, and why not: six levels everywhere (it is four times cheaper).
    With six levels on rotated rays, 17 of 80 random boxes in the property
    test held ``eps`` on the fit cloud and failed it on a finer one -- the
    rule was exact between the sampled Im levels only."""
    return 6 if abs(np.sin(theta)) < 0.15 else 24


def box_samples(re_lo, re_hi, im_lo, im_hi, per_unit=5.0, n_im=6, near=30.0):
    """Sample cloud on the box: ``n_im`` log-spaced ``Im d`` levels from
    ``im_lo`` to ``im_hi``, each carrying its own real line.

    Along ``Re d`` the spacing is ``min(Im d, 4 im_lo) / per_unit`` within
    ``near * im_lo`` of zero and geometric beyond (``_re_line``).  Higher Im
    levels are smoother and get coarser lines, but never coarser than
    ``4 im_lo / per_unit``: on a rotated ray the fast oscillation is set by
    the ray's own frequency, not by the level's ``Im d``.  The fit cloud uses
    the defaults; acceptance uses ``per_unit = 8`` and twice the levels."""
    im = np.geomspace(im_lo, max(im_hi, im_lo * 1.0001), n_im)
    out = []
    for v in im:
        h = min(v, 4.0 * im_lo) / per_unit
        out.append(_re_line(re_lo, re_hi, h, near * im_lo) + 1j * v)
    return np.concatenate(out)


def _log_density_weights(d):
    """Per-sample weights ``sqrt(q_i)`` with ``q_i`` the spacing of ``log|Re d|``
    within each ``Im`` level (mean 1).  In the relative currency the basis
    truncation and the least-squares steps should count every decade of
    ``|d|`` equally (the natural measure for ``1/x`` on ``[a, b]``); the cloud
    itself is denser far out, and without this the truncated basis is all
    about the far region and the start misses the relative criterion at the
    near corner by 30x (measured: sup_rel 2.8e-3 at eps 1e-4 on the Na
    val:bulk box, worst point ``d = re_lo + i im_lo``)."""
    q = np.empty(d.size)
    for level in np.unique(d.imag):
        idx = np.nonzero(d.imag == level)[0]
        x = np.log(np.abs(d.real[idx]))
        order = np.argsort(x)
        g = np.gradient(x[order]) if idx.size > 1 else np.ones(1)
        q[idx[order]] = np.abs(g)
    q /= max(q.mean(), 1e-300)
    return np.sqrt(q)


def _legal_angles(d, margin=0.25):
    """Ray angles (1 degree steps) whose slowest family member still decays at
    a rate of at least ``margin * min(Im d)``:
    ``|exp(i s e^{-i th} d)| = exp(-s (cos th Im d - sin th Re d))``.

    The margin keeps the horizon ``S = ln(10/eps)/rate`` within four times the
    real-time horizon.  Tempting, and why not: allow every angle with a
    positive rate.  Near the limit ``rate -> 0`` the horizon and the Chebyshev
    grid grow without bound, and a rule on that ray amplifies executor noise
    by the same factor it gained in rank."""
    th = np.deg2rad(np.arange(-89.0, 89.5, 1.0))
    rate = np.min(np.cos(th)[:, None] * d.imag[None, :]
                  - np.sin(th)[:, None] * d.real[None, :], axis=1)
    ok = rate >= margin * d.imag.min()
    return th[ok], rate[ok]


# ----------------------------------------------------------------- Chebyshev grid
def _cheb_grid(S, n):
    """Chebyshev-Lobatto points on ``[0, S]`` with Clenshaw-Curtis weights.

    The family is compressed by an SVD in the L2 inner product on the ray,
    which these weights integrate exactly up to degree ``n - 1``; the Lobatto
    clustering near ``s = 0`` is what resolves the short-lived fast members
    for free.  Tempting, and why not: a uniform grid with trapezoid weights.
    Its near-zero spacing is ``S/n``, so the fastest members alias unless
    ``n`` is many thousands; the first reducer stalled on exactly that."""
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

    Each member oscillates at ``nu = |cos th Re d + sin th Im d|`` and decays
    at ``lam``; it needs ``points_per_half_wave`` points per half wave over its
    LIFETIME ``min(S, ln(10/eps)/lam)``, and the near-zero Lobatto spacing
    ``S (pi/N)^2 / 2`` must stay below a half wave of the fastest member.
    Three points per half wave is the measured floor: at two the SVD basis
    aliased and the reduction stalled.

    Tempting, and why not: size the grid by ``nu_max * S`` (the fastest member
    over the whole horizon).  On a rotated ray the fast members are the
    fast-decaying ones and are dead long before ``S``; sizing by ``nu_max * S``
    gave 6000-point grids and a 353 s start where the lifetime rule takes
    seconds for the same basis."""
    nu = np.abs(np.cos(theta) * d.real + np.sin(theta) * d.imag)
    lam = np.maximum(np.cos(theta) * d.imag - np.sin(theta) * d.real, 1e-300)
    life = np.minimum(S, np.log(10.0 / eps) / lam)
    interior = points_per_half_wave * np.max(nu * life) / np.pi
    near_zero = np.sqrt(points_per_half_wave * np.pi * S * np.max(nu) / 2.0)
    # The near-zero spacing must also resolve the shortest LIFETIME, not only
    # the fastest oscillation: on a rotated ray the far members of a wide
    # sign-definite box die within ``1/(|Re d| sin th)``, which in the
    # relative currency they must be integrated to ``eps`` over (in the
    # peak-relative currency they hardly count).  Measured on a box with
    # ``R ~ 9000`` at 85 deg: the oscillation rule gave 138 points, the first
    # interval was ten lifetimes wide, and the start rule was wrong
    # everywhere (relative error 0.6, kappa 2e4).
    near_decay = np.pi * np.sqrt(
        points_per_half_wave * S * np.max(lam) / (2.0 * np.log(10.0 / eps)))
    return int(min(max(interior, near_zero, near_decay) + 100, cap))


def _family(d, theta, s):
    """``exp(i t d)`` for ``t = s exp(-i theta)``: rows are cloud points,
    columns are ray positions."""
    return np.exp(1j * d[:, None] * (s * np.exp(-1j * theta))[None, :])


def _choose_angle(d, eps, rho, scan=(-85, -70, -55, -40, -20, 0, 20, 40, 55, 70, 85)):
    """Minimum ``eps``-rank ray angle over the legal interval (rank in the
    ``rho``-weighted norm, see ``build_uniform_rule``).

    Eleven candidate angles, each snapped to the nearest legal degree and
    skipped if none is within 0.6 degrees; the rank is measured on a thinned
    cloud (~500 points), which is within a node or two of the full cloud's.
    Crossing boxes land on real time, sign-definite ones at 55-70 degrees.
    If no scanned angle is legal (a very asymmetric box) the middle legal
    angle is used.

    Tempting, and why not: optimise the angle continuously or on a fine scan.
    The rank is a step function of the angle and flat around its minimum, so
    a finer search changes the count by a node while costing an SVD per
    angle; the coarse scan is about a second."""
    th_ok, rate_ok = _legal_angles(d)
    step = max(1, d.size // 500)
    thin, thin_rho = d[::step], rho[::step]
    best = None
    for deg in scan:
        i = int(np.argmin(np.abs(th_ok - np.deg2rad(deg))))
        if abs(th_ok[i] - np.deg2rad(deg)) > np.deg2rad(0.6):
            continue
        S = np.log(10.0 / eps) / rate_ok[i]
        s, w = _cheb_grid(S, _grid_size(thin, th_ok[i], S, eps))
        sv = svd(thin_rho[:, None] * _family(thin, th_ok[i], s) * np.sqrt(w)[None, :],
                 compute_uv=False)
        r = int(np.sum(sv > eps * sv[0]))
        if best is None or r < best[0]:
            best = (r, th_ok[i], S, rate_ok[i])
    if best is None:
        i = len(th_ok) // 2
        best = (0, th_ok[i], np.log(10.0 / eps) / rate_ok[i], rate_ok[i])
    return best


# ----------------------------------------------------------------- ray family (start rule)
class _RayFamily:
    """SVD basis of ``{exp(i t(s) d)}`` on one ray; provides the interpolatory
    start rule.

    ``M`` is the family on the Lobatto grid with the Clenshaw-Curtis weights
    folded in, so its right singular vectors are L2-orthonormal on the ray;
    ``r`` columns survive at the truncation ``eps`` the caller passes (which is
    ``eps/10``: the start must be an order more accurate than the acceptance
    or the reduction finds nothing to remove).  ``m`` holds the integrals of
    the basis functions over ``[0, S]``, which an interpolatory rule must
    reproduce.  ``A`` converts the basis on the grid to Chebyshev coefficients
    so ``U(s)`` can evaluate it at ANY ``s`` as a cosine series (exact for the
    degree-``N`` interpolant).

    Tempting, and why not: evaluate the basis off-grid by re-sampling the
    family and projecting.  That is a fresh SVD per evaluation and, worse, the
    projection is not the interpolant, so the pivoted QR below picked nodes
    the rule then could not reproduce."""

    def __init__(self, d, theta, S, eps, rho):
        self.d, self.theta, self.S = d, theta, S
        self.phase = np.exp(-1j * theta)
        n_s = _grid_size(d, theta, S, eps)
        s, ws = _cheb_grid(S, n_s)
        # rows weighted by rho: the basis is truncated in the same norm the
        # rule is accepted in (relative on a sign-definite box), otherwise
        # the start misses the criterion and the reducer has nothing to keep
        M = rho[:, None] * _family(d, theta, s) * np.sqrt(ws)[None, :]
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
        """The ``r`` basis functions at ray positions ``s`` (cosine series)."""
        return self.A @ self._T(s)

    def interpolatory(self):
        """``r`` nodes by pivoted QR on the interior Lobatto points, weights
        by least squares against the basis integrals.  About 1.3-1.5x the
        Gauss count; ready in about a second."""
        # The Lobatto grid contains s = 0; a node there is a zero time node,
        # which the executor refuses ("served quadrature has invalid or zero
        # time nodes"), so the end points are never offered.  Tempting, and
        # why not: s = 0 is often the best-conditioned pivot, and it was the
        # first thing the QR picked before this line existed.
        cand = self.s_grid[1:-1]
        _, _, piv = _pivoted_qr(self.U(cand.astype(complex)), mode="economic", pivoting=True)
        s = np.sort(cand[piv[:self.r]]).astype(complex)
        w = lstsq(self.U(s), self.m, rcond=None)[0]
        return s, w

    def to_rule(self, s, w):
        """Time nodes and weights of ``1/d ~= sum w_k exp(i t_k d)``: with
        ``t = phase s`` the identity ``1/d = -i int_0^inf exp(i t d) dt``
        becomes ``-i phase int exp(i phase s d) ds``, hence the weight map."""
        return self.phase * s, -1j * self.phase * w


# ----------------------------------------------------------------- cloud solver (reduction)
class _CloudFit:
    """Nonlinear least squares of the rule against ``1/d`` on the cloud.

    Residual ``E_i = sum_k w_k exp(i d_i t_k) - 1/d_i`` scaled by ``rho_i``:
    ``im_lo`` (relative to the ``1/eta`` peak) on a crossing box, ``|d_i|``
    (relative to the term's own size) on a sign-definite box -- the same
    currency ``eps`` is stated in, see ``build_uniform_rule``.
    Nodes ``t = phase * s`` with ``Re s`` in ``[0, S]`` and ``Im s`` in
    ``[im_lo, im_hi]`` (the caller's off-ray cap); weights are eliminated by
    penalised least squares (variable projection with the Kaufman Jacobian).
    The state ``(s, w)`` is in the executor's convention already, so
    ``fam.phase * s`` and ``w`` ARE the rule (no ``-i phase`` factor here).

    Tempting, and why not: optimise nodes and weights jointly.  Weights are
    linear and ill-conditioned (exponentially clustered columns); solving them
    exactly at every node step is what keeps the LM steps meaningful.
    """

    def __init__(self, d, phase, S, im_lo, im_hi, eps, w_ref, rho, alpha=0.3):
        self.d, self.phase, self.S, self.im_lo, self.im_hi = d, phase, S, im_lo, im_hi
        self.eps = eps
        self.scale = rho
        self.b = self.scale / d
        self.nb = np.linalg.norm(self.b)
        self.idp = 1j * d * phase
        # Tikhonov weight penalty: |mu w| ~ alpha*eps*|b| for weights of the
        # reference size, so runaway cancelling weights cost more than the
        # acceptance residual while normal weights cost less.  This is what
        # pins the cancellation ratio kappa: without it the solver happily
        # trades two nodes for one pair of 1e6-sized opposite weights that
        # the executor's noise floor then amplifies.  Tempting, and why not:
        # mu = 0 with a kappa check afterwards -- the solve converges to the
        # cancelling solution first and the check just refuses it.
        self.mu = alpha * eps * self.nb / max(np.linalg.norm(w_ref), 1e-300)
        self.c = 0.5 * (im_lo + im_hi)
        self.h = 0.5 * (im_hi - im_lo)

    def A(self, s):
        """Design matrix ``exp(i d t)`` with the rows in the rule's currency."""
        return np.exp(self.idp[:, None] * s[None, :]) * self.scale[:, None]

    def ls(self, s):
        """Penalised least-squares weights for nodes ``s``: QR of ``[A; mu I]``.
        Returns ``w``, the full residual (cloud rows then penalty rows), ``Q``
        and ``A``; the Jacobian below reuses ``Q``."""
        A = self.A(s)
        n = s.size
        Aa = np.concatenate([A, self.mu * np.eye(n, dtype=complex)], 0)
        Q, R = np.linalg.qr(Aa)
        ba = np.concatenate([self.b, np.zeros(n, complex)])
        c = Q.conj().T @ ba
        w = np.linalg.solve(R, c)
        return w, ba - Q @ c, Q, A

    def _y_of(self, s):
        # Im s = c + h tanh(y): the off-ray cap is part of the model, so no
        # step can leave the strip and no clipping is needed on Im.
        u = np.clip((s.imag - self.c) / max(self.h, 1e-300), -1.0 + 1e-3, 1.0 - 1e-3)
        return np.arctanh(u)

    def _s_of(self, re, y):
        return re.clip(0.0, self.S) + 1j * (self.c + self.h * np.tanh(y))

    def newton(self, s, steps=30, tol=1e-14, check=None, chunk=10, stall=1e-3):
        """Variable-projection Levenberg-Marquardt on the node positions.

        Kaufman Jacobian ``J = P_perp (dA/ds) w`` (the weight response is
        projected out through ``Q``), split into real and imaginary parts with
        the tanh chain rule on ``Im s``, columns scaled to unit norm.  One
        eigendecomposition of the Gram matrix per step gives the step for
        every damping ``lam`` at once; up to 12 dampings are tried, the first
        that lowers the residual is taken and ``lam`` relaxed by 3, else it is
        raised by 4.  Every ``chunk`` steps the optional ``check(s, w, res)``
        may stop the solve (rule accepted) and the solve stops when the
        residual fell by less than ``stall`` over the chunk.  Returns nodes,
        weights, relative residual.

        Tempting, and why not: a plain Gauss-Newton or lstsq step per
        damping.  Undamped steps move clustered nodes past each other and
        the residual explodes; a separate lstsq per damping costs 12 solves
        of a tall system per step where the eigendecomposition costs one Gram
        of size ``2n``."""
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
        """Residual increase when node ``k`` is dropped and the weights
        re-solved: ``|w_k|^2 / [(A^H A + mu^2)^-1]_kk``, exact for the linear
        (weights-only) re-solve.  Small score = cheap to remove.

        Tempting, and why not: rank by ``|w_k|``.  A small weight on a node the
        neighbours cannot cover is expensive to remove and a large weight in a
        dense cluster is cheap; the ``|w|``-ranked reducer stalled at ~0.9 r
        while this one reaches ~0.5 r on the same windows."""
        A = self.A(s)
        G = np.linalg.pinv(A.conj().T @ A + self.mu ** 2 * np.eye(s.size))
        return np.abs(w) ** 2 / np.maximum(np.real(np.diag(G)), 1e-300)

    def _solve_pick(self, starts, ok, nstep, keep, rank_steps=8):
        """Successive halving: every start gets a short solve (``rank_steps``),
        the ``keep`` best continue to ``nstep`` steps with early exit on
        acceptance; the accepted solve with the smallest residual wins (or
        None).  A full solve for every candidate would cost ``K`` times the
        budget per removal; the short solve ranks them well enough."""
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

        The start is first polished to the optimum for its node count; if
        even that is not accepted the caller keeps the interpolatory rule.
        Far above the target the batch move drops ``batch`` low-score nodes at
        once, never two neighbours (a gap two nodes wide cannot be closed by
        the survivors), and solves to the optimum: an early exit at
        acceptance would leave a marginal state the single removals inherit.
        Each failed batch halves it.  At batch 1 the ``K`` best leave-one-out
        candidates are tried by successive halving, then the next ``2K``;
        when neither yields an accepted rule the reduction stops.  ``best`` is
        always the last accepted state, so the deadline can fall anywhere.

        Tempting, and why not: single removals from the start (``r`` is ~300
        on a wide crossing box, and ~150 removals times ``K`` solves never
        finish in the budget), or a fixed batch (one failure at batch 30
        would end the batch phase 30 nodes early)."""
        s, w, _res = self.newton(s.astype(complex), steps=nstep)     # polish the start
        # A least-squares polish is L2-optimal, and on a sign-definite box in
        # the relative currency its sup sits at the near corner, 3-7x above
        # the L2 level (the corner is a small region in the log measure that
        # carries the slowest ray members).  Lawson's reweighting -- rows
        # re-weighted by their own residual and re-solved -- moves the L2
        # optimum toward the minimax one; a handful of tempered rounds is
        # enough to bring the corner under eps.  Tempting, and why not: a
        # larger trunc (eps/100) instead -- it costs 4-5 nodes and kappa and
        # still misses (measured 2.6e-4 at eps 1e-4 on the Na val:bulk box).
        # The start's acceptance is not subject to the deadline: the budget
        # bounds the REDUCTION, and a rule must exist at any budget (the
        # rounds are bounded by construction, ~30 s on an R ~ 1e4 box).
        for _round in range(6):
            if ok(s, w):
                break
            E = np.abs(self.A(s) @ w - self.b)
            factor = np.clip(np.sqrt(E / max(E.mean(), 1e-300)), 0.5, 2.0)
            self.scale = self.scale * factor
            self.scale *= self.nb / max(np.linalg.norm(self.scale / self.d), 1e-300)
            self.b = self.scale / self.d
            s, w, _res = self.newton(s, steps=nstep)
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


def rule_sup_error(times, weights, d, rho=None):
    """``(max rho |Q(d) - 1/d|, max kappa)`` on the cloud ``d``, where ``rho``
    is ``min Im d`` (error relative to the ``1/eta`` peak, the default) or
    ``|d|`` (relative error), and ``kappa = sum_k |w_k exp(i t_k d)| / |Q(d)|``
    is the term-cancellation ratio: the factor by which the executor's
    per-term noise is amplified in the sum.  Both are what the planner's
    gates read."""
    A = np.exp(1j * d[:, None] * times[None, :])
    Q = A @ weights
    err = np.abs(Q - 1.0 / d) * (d.imag.min() if rho is None else rho)
    kappa = np.abs(A * weights[None, :]).sum(1) / np.maximum(np.abs(Q), 1e-300)
    return float(err.max()), float(kappa.max())


@dataclass(frozen=True)
class UniformRule:
    """A finished rule: ``times``/``weights`` in the executor's convention
    (``1/d ~= sum weights * exp(i times * d)``), the box and ``eps`` it was
    built for (its cache key), the ray angle, the interpolatory rank ``r`` it
    started from, and the sup error and cancellation ratio measured on the
    check cloud."""
    times: np.ndarray
    weights: np.ndarray
    box: tuple
    eps: float
    relative: bool
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
                f"rank {self.rank}, sup {self.sup_error:.2e} (eps {self.eps:g}, "
                f"{'relative' if self.relative else 'peak-relative'}), "
                f"kappa {self.kappa_max:.3g}, {self.seconds:.1f} s")


def build_uniform_rule(box, eps, *, im_cap=3.0, kappa_cap=1.0e4, trunc=10.0,
                       reduce=True, time_budget=None, relative=None):
    """Rule for ``1/d`` on ``box = (re_lo, re_hi, im_lo, im_hi)`` with
    ``Im d > 0``.

    ``eps`` is a sup bound on the box in the rule's currency: on a
    sign-definite box (``re_lo > 0`` or ``re_hi < 0``) the RELATIVE error
    ``|d| |Q(d) - 1/d| <= eps``; on a crossing box the error relative to the
    ``1/im_lo`` peak, ``im_lo |Q(d) - 1/d| <= eps`` (``relative`` overrides
    the choice).  The distinction is physics, not taste: a term's error in
    Sigma scales with the term's own size ``1/|d|``.  A sign-definite tail
    carries states far from resonance whose terms are large (semicore
    states at ``|d| ~ 200 eta``), and a peak-relative sup of ``eps`` there is
    a ``200 eps`` relative error: measured 4 meV at the Na 2s state where the
    relative rule gives 0.1 meV.  Exponential sums for ``1/x`` on ``[a, b]``
    are uniform in relative error at ``O(log(b/a))`` terms anyway, so the
    relative criterion is free on those boxes.  On a crossing box ``1/d`` is
    bounded only by the peak, and the relative criterion would tighten the
    far edges by ``|d|/eta`` (up to 40x) for terms whose size the peak
    already dominates: measured +50 % nodes on the Na conduction crossing
    box (76 -> 115) for no delivered-error gain.

    ``im_cap`` bounds ``|Im t| * (box half-width)`` so no family member grows
    by more than ``exp(im_cap)`` off the ray (and never more than ``0.3 S``
    off it): the SVD basis is a basis of the ray only, and nodes that wander
    far off it buy accuracy on the fit cloud with growth the check cloud then
    catches.  ``trunc`` is the start rule's extra accuracy (``eps/trunc``).
    ``kappa_cap`` is the largest cancellation ratio accepted.  ``time_budget``
    (seconds, from the start of this call) bounds the Gauss reduction and
    returns the best accepted rule at the deadline; the interpolatory rule is
    always available after about a second, so the budget trades planning
    wall for node count and nothing else.  Never refuses a finite box.

    Tempting, and why not: judge acceptance on the fit cloud ``d`` itself
    (17 of 80 random boxes passed there and failed on a finer cloud), or skip
    the ``ok`` polish of the start (an unpolished start inherits the
    interpolatory weights, which are far from the least-squares optimum and
    cost the first batch removals)."""
    re_lo, re_hi, im_lo, im_hi = map(float, box)
    if not (np.isfinite([re_lo, re_hi, im_lo, im_hi]).all() and re_lo <= re_hi
            and 0.0 < im_lo <= im_hi):
        raise ValueError(f"invalid support box {box!r}")
    t0 = time.perf_counter()
    if relative is None:
        relative = re_lo > 0.0 or re_hi < 0.0
    # rho_of: the acceptance currency (sup); fit_of: the same currency with
    # the cloud's log-density folded in, for the basis and the least squares
    rho_of = (lambda x: np.abs(x)) if relative else (lambda x: np.full(x.size, im_lo))
    fit_of = ((lambda x: np.abs(x) * _log_density_weights(x)) if relative
              else rho_of)
    d = box_samples(re_lo, re_hi, im_lo, im_hi)
    _r0, theta, S, _rate = _choose_angle(d, eps, fit_of(d))
    n_im = _im_levels(theta)
    if n_im != 6:
        d = box_samples(re_lo, re_hi, im_lo, im_hi, n_im=n_im)
    # Acceptance is judged on a cloud finer than the fit cloud in both
    # directions, so a rule that is exact between fit samples only is
    # rejected (property test: 17 of 80 random rotated-ray boxes failed a
    # finer check before this).  Tempting, and why not: use the fit cloud,
    # or a coarser check (it is the largest matrix in the reduction loop):
    # the failures were all rotated-ray boxes whose rule was exact at the
    # sampled Im levels and off between them.
    d_check = box_samples(re_lo, re_hi, im_lo, im_hi, per_unit=8.0, n_im=2 * n_im)
    rho, rho_check = fit_of(d), rho_of(d_check)
    fam = _RayFamily(d, theta, S, eps / trunc, rho)
    s, w = fam.interpolatory()
    if reduce:
        Bp = max(re_hi, 1e-3 * im_lo)
        Bm = max(-re_lo, 1e-3 * im_lo)
        im_lo_s, im_hi_s = max(-im_cap / Bp, -0.3 * S), min(im_cap / Bm, 0.3 * S)

        def ok(s_, w_):
            e_, k_ = _score_cloud(fit, s_, w_, d_check, rho_check)
            return e_ <= eps and k_ <= kappa_cap

        deadline = t0 + (float(time_budget) if time_budget is not None else 1e30)
        # The start must be accepted before anything can be removed.  In the
        # relative currency a loose eps (1e-3) with the default eps/10 basis
        # can leave the near corner of a wide box (R ~ 500) above eps even
        # after the Lawson rounds; a basis one order tighter costs ~2 nodes
        # at the start and a few seconds, so escalate rather than refuse.
        red = None
        for extra in (1.0, 10.0, 100.0):
            if extra > 1.0:
                fam = _RayFamily(d, theta, S, eps / (trunc * extra), rho)
                s, w = fam.interpolatory()
            # the cloud solver works in the executor's convention t = phase*s
            # with weights -i*phase*w: the sup test uses exactly that map
            fit = _CloudFit(d, fam.phase, S, im_lo_s, im_hi_s, eps, w_ref=w, rho=rho)
            red = fit.reduce(s, ok, deadline)       # start acceptance ignores the deadline
            if red is not None:
                break
        if red is not None:
            s, w_fit = red
            times, weights = fam.phase * s, w_fit          # A w - b = scale (Q - 1/d): w is the rule weight
        else:
            times, weights = fam.to_rule(s, w)
    else:
        times, weights = fam.to_rule(s, w)
    sup, kappa = rule_sup_error(times, weights, d_check, rho_check)
    return UniformRule(
        times=times, weights=weights, box=(re_lo, re_hi, im_lo, im_hi),
        eps=float(eps), relative=bool(relative), theta_deg=float(np.rad2deg(theta)),
        rank=int(fam.r), sup_error=sup, kappa_max=kappa,
        seconds=time.perf_counter() - t0)


def _score_cloud(fit, s, w, d, rho):
    """Sup error and kappa of the cloud-solver state ``(s, w)`` in rule form."""
    return rule_sup_error(fit.phase * s, w, d, rho)

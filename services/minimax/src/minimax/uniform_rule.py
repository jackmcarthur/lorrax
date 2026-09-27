"""The Sigma box rule's record, its boundary certificate and its noise measure.

A rule for ``1/d`` on a denominator box ``[re_lo, re_hi] x [im_lo, im_hi]``
(``im_lo = eta > 0``) is a :class:`UniformRule`: times and weights in the
executor convention ``1/d ~= sum_k w_k exp(i t_k d)``, the box and ``eps``
it answers, and the certificate it passed.  The builder is
:func:`minimax.analytic_box.analytic_box_rule`; this module holds what the
builder, the planner and the rule table share:

- :class:`_BoundaryCloud`, the acceptance certificate: the error and the
  executor's noise mass are analytic/subharmonic on the closed box, so their
  maxima lie on its boundary, sampled at the rule's own horizon and refined
  by a bracketed golden-section search;
- :func:`rule_sup_error` and :func:`rule_roundoff_amplification`, the same
  two numbers on a caller's cloud (the planner's noise gate);
- the 16-thread BLAS pin every build runs under, and
  :func:`uniform_rule_solver_identity`, which keys the persistent rule table.

Currencies: the RELATIVE error ``|d| |Q - 1/d|`` on a sign-definite box, the
peak-relative ``eta |Q - 1/d|`` on a crossing box
(``docs/theory/sigma-quadrature-problem.md`` section 6).  Boxes with
``Im d < 0`` are the caller's: it conjugates (``times -> -conj(times)``,
``weights -> conj(weights)``).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

__all__ = [
    "UniformRule", "boundary_samples", "rule_roundoff_amplification",
    "rule_sup_error", "uniform_rule_solver_identity",
]


# ----------------------------------------------------------------- BLAS threads
#: The thread count every rule build runs its BLAS at, whatever the launch
#: environment. OpenBLAS sums in a thread-count dependent order, so the weight
#: solve's round-off, and with it a marginal certificate, would otherwise
#: follow the environment; the pin makes a rule a function of (box, eps) on
#: one machine class, which the rule table's memo needs. 16 is the physical
#: core count of a Perlmutter GPU rank (``runtime.default_blas_threads``).
_BLAS_THREADS = 16
_BLAS_CONTROLS = None


def _openblas_controls():
    """``(get, set)`` for each OpenBLAS in this process; empty if none.

    numpy's and scipy's wheels each bundle one (``scipy_openblas64_`` and
    ``scipy_openblas``), and the builder calls both. They are found by path
    in ``/proc/self/maps``, since threadpoolctl is not in the runtime, once
    per process: both are loaded by the imports above.
    """
    global _BLAS_CONTROLS
    if _BLAS_CONTROLS is None:
        import ctypes
        try:
            with open("/proc/self/maps", encoding="ascii") as maps:
                paths = sorted({line.split()[-1] for line in maps
                                if "openblas" in line.rsplit("/", 1)[-1].lower()})
        except OSError:
            paths = []
        controls = []
        for path in paths:
            lib = ctypes.CDLL(path)
            for suffix in ("64_", ""):
                get = getattr(lib, f"scipy_openblas_get_num_threads{suffix}", None)
                put = getattr(lib, f"scipy_openblas_set_num_threads{suffix}", None)
                if get is None or put is None:
                    get = getattr(lib, f"openblas_get_num_threads{suffix}", None)
                    put = getattr(lib, f"openblas_set_num_threads{suffix}", None)
                if get is not None and put is not None:
                    get.restype, put.argtypes, put.restype = ctypes.c_int, [ctypes.c_int], None
                    controls.append((get, put))
                    break
        _BLAS_CONTROLS = tuple(controls)
    return _BLAS_CONTROLS


@contextmanager
def _pinned_blas_threads():
    """Run the enclosed rule build at ``_BLAS_THREADS``, then restore."""
    controls = _openblas_controls()
    saved = [get() for get, _put in controls]
    for _get, put in controls:
        put(_BLAS_THREADS)
    try:
        yield
    finally:
        for (_get, put), count in zip(controls, saved):
            put(count)


# ----------------------------------------------------------------- boundary cloud
def _live_spacing(d, theta, S, eps, p, p_target):
    """Spacing that resolves every live family member at ``d``.

    ``exp(i t d)`` with ``t = s exp(-i theta)``, ``s <= S``, has modulus
    ``exp(-s lam(d))``, ``lam = cos(theta) Im d - sin(theta) Re d``, and is
    alive (above ``eps/10``) for ``s <= min(S, ln(10/eps)/lam)``.  Its local
    frequency along any direction is at most that horizon, so ``p`` points per
    half wave need ``pi / (p horizon)``; ``1/d`` varies on ``|d|``, hence the
    cap ``|d| / p_target``.  On real time the horizon is ``S`` along the whole
    bottom edge, so the spacing is uniform in ``Re d``; on a rotated ray it
    grows with ``|Re d|`` and the edge is sampled geometrically."""
    lam = np.cos(theta) * d.imag - np.sin(theta) * d.real
    horizon = np.minimum(S, np.log(10.0 / eps) / np.maximum(lam, 1e-300))
    return np.minimum(np.pi / (p * horizon), np.abs(d) / p_target)


def _edge_points(lo, hi, point, theta, S, eps, p, p_target):
    """Edge parameters in ``[lo, hi]``, ends included, spaced by
    ``_live_spacing``: the cumulative sample density is integrated on an
    auxiliary grid refined toward both ends and ``Re d = 0`` and inverted."""
    if not hi > lo:
        return np.array([lo])
    offs = np.geomspace(1e-7 * (hi - lo), hi - lo, 1200)
    aux = [np.linspace(lo, hi, 4001), lo + offs, hi - offs]
    if lo > 0.0:
        aux.append(np.geomspace(lo, hi, 4001))
    if lo < 0.0 < hi:
        aux += [offs[offs < hi], -offs[offs < -lo]]
    u = np.unique(np.clip(np.concatenate(aux), lo, hi))
    density = 1.0 / _live_spacing(point(u), theta, S, eps, p, p_target)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(u))])
    pts = np.interp(np.linspace(0.0, cum[-1], max(int(np.ceil(cum[-1])), 2) + 1), cum, u)
    pts[0], pts[-1] = lo, hi
    return pts


class _BoundaryCloud:
    """Samples on the four edges of a box, kept in order per edge.

    Why the boundary is enough: the accepted quantity is
    ``sup rho |Q(d) - 1/d|`` with ``rho = im_lo`` or ``rho = |d|``; both
    ``im_lo (Q - 1/d)`` and ``d Q(d) - 1`` are analytic on the closed box
    (``d = 0`` lies below it), so by the maximum modulus principle the sup is
    attained on the boundary.  The executor-noise mass
    ``rho sum_k |w_k exp(i t_k d)|`` has a subharmonic logarithm and peaks
    there too (checked on 300 cached rules).  ``sup`` refines the sampled
    local maxima by a bracketed golden-section search.

    Tempting, and why not: a cloud of interior levels, or a geometric far
    field.  On a thin box (``Im d`` in ``[eta, 1.01 eta]``, every real-pole
    window) the levels are copies of one line, and on real time the rule's
    error oscillates at the node horizon at every ``Re d``: a dense boundary
    audit found 100 of 2,603 level-certified rules above eps, up to 247x on
    wide boxes (runs/DEV/326_minimax_fit_review_2026-09-11)."""

    def __init__(self, box, theta, S, eps, *, p, p_target, top=True):
        re_lo, re_hi, im_lo, im_hi = box
        kw = dict(theta=theta, S=S, eps=eps, p=p, p_target=p_target)
        x = _edge_points(re_lo, re_hi, lambda v: v + 1j * im_lo, **kw)
        edges = [(x, x + 1j * im_lo)]
        if im_hi > im_lo:
            yl = _edge_points(im_lo, im_hi, lambda v: re_lo + 1j * v, **kw)
            yr = _edge_points(im_lo, im_hi, lambda v: re_hi + 1j * v, **kw)
            thin = yl.size <= 2 and yr.size <= 2
            if top or not thin:     # a box thinner than one spacing fits on its bottom edge
                xt = _edge_points(re_lo, re_hi, lambda v: v + 1j * im_hi, **kw)
                edges.append((xt, xt + 1j * im_hi))
            if not thin:
                edges += [(yl, re_lo + 1j * yl), (yr, re_hi + 1j * yr)]
        self.edges = edges
        self.d = np.concatenate([e[1] for e in edges])
        self.im_lo = im_lo

    def arc_weights(self, relative):
        """``sqrt`` of each sample's local arc length in ``|dd|`` (``|dd|/|d|``
        on a relative box), mean 1: the least squares approximates a boundary
        integral rather than the sampling density."""
        lengths = np.concatenate([np.abs(np.gradient(u)) for u, _ in self.edges])
        if relative:
            lengths = lengths / np.abs(self.d)
        return np.sqrt(lengths / lengths.mean())

    def _g(self, d, times, weights, relative, power=1):
        Q = _cexp(1j * d[:, None] * np.asarray(times)[None, :]) @ np.asarray(weights)
        if relative:
            return np.abs(d ** power * Q - 1.0)
        return self.im_lo ** power * np.abs(Q - 1.0 / d ** power)

    def sup(self, times, weights, relative, iters=20, power=1):
        """``(sup rho |Q - 1/d|, max kappa, max rho sum_k |w_k exp(i t_k d)|)``:
        every sampled local maximum of the error within 10% of the largest
        (edge ends included) is refined by a golden-section search on its
        bracket.  The third number is the executor's noise amplification in
        the certificate's currency.  ``power = 2`` certifies ``1/d**2`` in the
        same currency raised to the second power (the response rules' ds rows).

        Tempting, and why not: the vertex of the parabola through the three
        samples (no extra evaluations).  Near the corner of a relative box the
        error falls 4x within two samples, and the vertex misread the sup by
        -1.3% and +0.5% where the bracketed search reads the dense value
        (runs/DEV/326_minimax_fit_review_2026-09-11/tools/diag_refinement.py)."""
        T = _cexp(1j * self.d[:, None] * np.asarray(times)[None, :]) * np.asarray(weights)[None, :]
        Q = T.sum(1)
        g = (np.abs(self.d ** power * Q - 1.0) if relative
             else self.im_lo ** power * np.abs(Q - 1.0 / self.d ** power))
        term_mass = np.abs(T).sum(1)
        kappa = float((term_mass / np.maximum(np.abs(Q), 1e-300)).max())
        mass = float(((np.abs(self.d) if relative else self.im_lo) ** power * term_mass).max())
        best, start = float(g.max()), 0
        base, step, lo, hi = [], [], [], []
        for u, pts in self.edges:
            y = g[start:start + pts.size]
            start += pts.size
            if pts.size < 2:
                continue
            peak = (np.concatenate([[True], y[1:] >= y[:-1]])
                    & np.concatenate([y[:-1] >= y[1:], [True]]) & (y >= 0.9 * best))
            i = np.nonzero(peak)[0]
            horizontal = pts[0].imag == pts[-1].imag
            base.append(pts[i] - (u[i] if horizontal else 1j * u[i]))
            step.append(np.full(i.size, 1.0 if horizontal else 1.0j))
            lo.append(u[np.maximum(i - 1, 0)])
            hi.append(u[np.minimum(i + 1, pts.size - 1)])
        if sum(v.size for v in base):
            base, step, a, b = map(np.concatenate, (base, step, lo, hi))
            r = 0.5 * (np.sqrt(5.0) - 1.0)
            c, e = b - r * (b - a), a + r * (b - a)
            fc = self._g(base + step * c, times, weights, relative, power)
            fe = self._g(base + step * e, times, weights, relative, power)
            for _ in range(iters):
                left = fc > fe                   # the maximum lies in [a, e]
                a, b = np.where(left, a, c), np.where(left, e, b)
                x = np.where(left, b - r * (b - a), a + r * (b - a))
                fx = self._g(base + step * x, times, weights, relative, power)
                c, e, fc, fe = (np.where(left, x, e), np.where(left, c, x),
                                np.where(left, fx, fe), np.where(left, fc, fx))
            best = max(best, float(fc.max()), float(fe.max()))
        return best, kappa, mass


def boundary_samples(box, theta_deg, horizon, eps, p=6.0):
    """Boundary samples of ``box`` resolving a rule on the ray ``theta_deg``
    whose largest ``|t|`` is ``horizon``: the cloud on which the executor's
    noise mass (and the rule's error) attains its box maximum."""
    return _BoundaryCloud(tuple(map(float, box)), np.deg2rad(float(theta_deg)),
                          float(horizon), float(eps), p=p, p_target=8.0).d


def _cexp(z):
    """``exp(z)`` for complex ``z`` as ``exp(Re z) (cos Im z + i sin Im z)``.
    numpy's complex ``exp`` is scalar code (44 ns per element measured); the
    three real ufuncs are SIMD and about 3x faster.  Same values."""
    return np.exp(z.real) * (np.cos(z.imag) + 1j * np.sin(z.imag))


def rule_sup_error(times, weights, d, rho=None):
    """``(max rho |Q(d) - 1/d|, max kappa)`` on the cloud ``d``, where ``rho``
    is ``min Im d`` (error relative to the ``1/eta`` peak, the default) or
    ``|d|`` (relative error), and ``kappa = sum_k |w_k exp(i t_k d)| / |Q(d)|``
    is the term-cancellation ratio: the factor by which the executor's
    per-term noise is amplified in the sum.  Both are what the planner's
    gates read."""
    A = _cexp(1j * d[:, None] * times[None, :])
    Q = A @ weights
    err = np.abs(Q - 1.0 / d) * (d.imag.min() if rho is None else rho)
    kappa = np.abs(A * weights[None, :]).sum(1) / np.maximum(np.abs(Q), 1e-300)
    return float(err.max()), float(kappa.max())


def rule_roundoff_amplification(times, weights, d, rho):
    """Worst absolute term mass in the approximation-error currency.

    If each exponential term carries a relative runtime perturbation
    ``eps_runtime``, the error is bounded by

    ``eps_runtime * max_d rho(d) * sum_k |w_k exp(i t_k d)|``.

    ``rho=|d|`` is the sign-definite rule's relative-error currency, while
    ``rho=min(Im d)`` is the crossing rule's peak-relative currency.  The
    cancellation ratio returned by :func:`rule_sup_error` divides by
    ``|Q(d)|`` and is therefore not in the crossing rule's currency: at a
    far edge it grows like ``|d|/eta`` while ``eta*|Q(d)|`` shrinks by the
    reciprocal factor.
    """
    d = np.asarray(d, dtype=np.complex128)
    scale = np.asarray(rho, dtype=np.float64)
    if d.ndim != 1 or scale.ndim > 1 or (
            scale.ndim == 1 and scale.shape != d.shape):
        raise ValueError(
            "roundoff amplification needs d as a vector and rho as a "
            f"scalar or matching vector; got {d.shape} and {scale.shape}")
    A = _cexp(1j * d[:, None] * np.asarray(times)[None, :])
    mass = np.sum(np.abs(A * np.asarray(weights)[None, :]), axis=1)
    return float(np.max(scale * mass))


@dataclass(frozen=True)
class UniformRule:
    """A finished rule: ``times``/``weights`` in the executor's convention
    (``1/d ~= sum weights * exp(i times * d)``), the box and ``eps`` it was
    built for (its cache key), the family's ray angle, its degree (the count
    law's answer), and the sup error and cancellation ratio measured on the
    certificate cloud."""
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
        return (f"box rule: {self.node_count} nodes, ray {self.theta_deg:.0f} deg, "
                f"degree {self.rank}, sup {self.sup_error:.2e} (eps {self.eps:g}, "
                f"{'relative' if self.relative else 'peak-relative'}), "
                f"kappa {self.kappa_max:.3g}, {self.seconds:.1f} s")


_STATIC_IDENTITY = None


def uniform_rule_solver_identity():
    """What an :func:`minimax.analytic_box_rule` result depends on besides
    its arguments, as a JSON-ready dict.

    The builder reads no clock and pins its BLAS threads, so two calls with
    equal arguments return the same rule bit for bit when these fields are
    equal: the minimax sources (sha256 over every ``.py`` of the package),
    the numerics backend (:func:`minimax.cache.backend_tag`), the CPU model
    (OpenBLAS dispatches its kernels by it) and the pinned thread count.
    A persistent rule table keys on this dict,
    so an edited builder or another machine class opens a new namespace
    instead of being served another solver's rule.
    """
    global _STATIC_IDENTITY
    if _STATIC_IDENTITY is None:
        import hashlib
        from pathlib import Path

        from .cache import backend_tag
        package = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(package.rglob("*.py")):
            digest.update(str(path.relative_to(package)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
        cpu = "unknown"
        try:
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as info:
                cpu = next((line.split(":", 1)[1].strip() for line in info
                            if line.startswith("model name")), cpu)
        except OSError:
            pass
        _STATIC_IDENTITY = {
            "minimax_sources": digest.hexdigest(), "backend": backend_tag(),
            "cpu": cpu, "blas_threads": _BLAS_THREADS,
        }
    return dict(_STATIC_IDENTITY)

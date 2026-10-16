"""solvers/time_integrators.py — one-step integrators for y' = f(t, y).

No physics.  Every routine takes the right-hand side, or the pieces of a
split right-hand side, as callables on JAX arrays, and returns arrays of the
state's shape and sharding; nothing here knows what the state represents.
The families, with the equation each step integrates:

* **Dormand–Prince 5(4)** (``dp54_*``): y' = f(t, y), the tableau, embedded
  error estimate, step controller and quartic dense output of SciPy's
  ``RK45`` (``scipy/integrate/_ivp/rk.py``), so a run started from the same
  first step takes the same steps and samples the same interpolant.
* **Classical RK4** (:func:`rk4_step`): y' = f(t, y).
* **Exponential RK, Krogstad ETDRK4** (:func:`etdrk4_step`):
  y' = λ ∘ y + N(t, y) with λ elementwise (a diagonal linear part), the
  linear part integrated exactly through φ-functions
  (Hochbruck & Ostermann, Acta Numer. 19, 209 (2010), §2.3, the scheme of
  Krogstad, J. Comput. Phys. 203, 72 (2005)).
* **Magnus in the interaction picture of a free flow, on a deviation**
  (:func:`ip_magnus_step` with :func:`ip_cf4_flow` or
  :func:`ip_midpoint_flow`): the isospectral flow
  y' = -i[A0 + A(t) + B(t, y), y] with A0 constant and exactly integrable
  (``free(X, s) = e^{-isA0} X e^{isA0}``), A(t) cheap and known at every t,
  B(t, y) the expensive state-dependent part, and a stationary y_eq of A0.
  The state is the deviation x = y - y_eq and every exponential acts on the
  interaction-picture generator G_I(s) = e^{isA0} (A + B) e^{-isA0} only, so
  each term of an update has the size of x, not of y.
* **Composed Strang splitting** (:func:`compose_strang`):
  R(f1 h/2) M(f1 h) R((f1 + f2) h/2) … M(fn h) R(fn h/2) for a remainder
  flow R that the unitary step does not contain; ``(1,)`` is Strang (order
  2), :data:`TRIPLE_JUMP` is Yoshida's fourth-order composition.

Stepping loops live in :func:`dp54_advance` (adaptive, emits samples from
the dense output at caller-given times) and :func:`fixed_advance` (fixed
step, samples at arbitrary times from the method's dense output); both are
``lax`` loops with no host round trip.
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

__all__ = [
    "DP54_C", "DP54_A", "DP54_B", "DP54_E", "DP54_P",
    "DP54_SAFETY", "DP54_MIN_FACTOR", "DP54_MAX_FACTOR", "DP54_ERROR_EXPONENT",
    "Dp54State", "dp54_initial_state", "dp54_step", "dp54_error_norm", "dp54_dense", "dp54_advance",
    "rk4_step", "phi_functions", "EtdCoefficients", "etdrk4_coefficients", "etdrk4_step",
    "hermite_dense",
    "extrapolation_weights", "lagrange_weights",
    "CFM4_NODES", "CFM4_WEIGHTS", "CFM4_HISTORY", "MIDPOINT_NODES", "MIDPOINT_HISTORY",
    "CF4_QUADRATURE", "TRIPLE_JUMP",
    "ip_cf4_flow", "ip_midpoint_flow", "ip_magnus_step", "frozen_generator",
    "magnus_amplification", "magnus_stability_limit",
    "compose_strang", "fixed_advance",
]

# ---------------------------------------------------------------------------
# Dormand–Prince 5(4): SciPy RK45's tableau, controller and dense output
# ---------------------------------------------------------------------------

DP54_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0])
DP54_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1 / 5, 0.0, 0.0, 0.0, 0.0],
    [3 / 40, 9 / 40, 0.0, 0.0, 0.0],
    [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
])
DP54_B = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
DP54_E = np.array([-71 / 57600, 0.0, 71 / 16695, -71 / 1920, 17253 / 339200,
                   -22 / 525, 1 / 40])
DP54_P = np.array([
    [1.0, -8048581381 / 2820520608, 8663915743 / 2820520608,
     -12715105075 / 11282082432],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 131558114200 / 32700410799, -68118460800 / 10900136933,
     87487479700 / 32700410799],
    [0.0, -1754552775 / 470086768, 14199869525 / 1410260304,
     -10690763975 / 1880347072],
    [0.0, 127303824393 / 49829197408, -318862633887 / 49829197408,
     701980252875 / 199316789632],
    [0.0, -282668133 / 205662961, 2019193451 / 616988883,
     -1453857185 / 822651844],
    [0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
])
#: SciPy's step-size controller constants (``rk.py``: SAFETY, MIN_FACTOR,
#: MAX_FACTOR; the error exponent is -1/(error_estimator_order + 1)).
DP54_SAFETY = 0.9
DP54_MIN_FACTOR = 0.2
DP54_MAX_FACTOR = 10.0
DP54_ERROR_EXPONENT = -1.0 / 5.0


def _combine(coeffs, ks):
    """sum_j coeffs[j] * ks[j] over the nonzero coefficients, in j order."""
    out = None
    for c, k in zip(coeffs, ks):
        if float(c) == 0.0:
            continue
        term = float(c) * k
        out = term if out is None else out + term
    return out


def dp54_step(rhs: Callable, t, y, f, h):
    """One Dormand–Prince step of SciPy's ``rk_step``.

    Parameters
    ----------
    rhs : callable (t, y) -> dy/dt, same shape and sharding as ``y``.
    t, h : scalar arrays, the step start and signed step.
    y, f : the state and rhs(t, y) at the step start (FSAL value).

    Returns
    -------
    y_new, f_new, K
        The fifth-order solution, rhs(t + h, y_new), and the seven stage
        derivatives (K[6] = f_new), as SciPy's ``K`` rows.
    """
    ks = [f]
    for s in range(1, 6):
        dy = _combine(DP54_A[s, :s], ks) * h
        ks.append(rhs(t + DP54_C[s] * h, y + dy))
    y_new = y + h * _combine(DP54_B, ks)
    f_new = rhs(t + h, y_new)
    ks.append(f_new)
    return y_new, f_new, tuple(ks)


def dp54_error_norm(ks, h, y, y_new, atol, rtol, count):
    """SciPy's error norm ``rms(err / (atol + max(|y|, |y_new|) * rtol))``.

    ``err = h * sum_j E_j K_j``.  ``count`` is the number of LOGICAL
    entries: a zero pad (y = y_new = err = 0) adds nothing to the sum, so
    dividing by the logical count makes a padded state reproduce the
    unpadded norm exactly.  The reduction over a sharded state is one
    all-reduce.
    """
    err = _combine(DP54_E, ks) * h
    scale = atol + jnp.maximum(jnp.abs(y), jnp.abs(y_new)) * rtol
    return jnp.sqrt(jnp.sum(jnp.abs(err / scale) ** 2) / count)


def dp54_dense(ks, h, y_old, x):
    """SciPy's ``RkDenseOutput`` at fraction ``x = (t - t_old)/h`` of the step.

    y(t) = y_old + h * sum_j K_j * sum_q P[j, q] x^(q+1).
    """
    powers = x ** jnp.arange(1, 5, dtype=jnp.float64)
    weights = jnp.asarray(DP54_P) @ powers
    out = y_old
    for j, k in enumerate(ks):
        if not np.any(DP54_P[j]):
            continue
        out = out + (h * weights[j]) * k
    return out


class Dp54State(NamedTuple):
    """The SciPy solver state carried between steps and between calls.

    t : scalar, current time.   y, f : state and rhs(t, y).
    h_abs : the step size the controller proposes next.
    t_old, h_last, y_old, ks : the last accepted step's dense output (the
        interpolant on [t_old, t]); a fresh state has h_last = 0, so its
        interpolant is the constant y.
    n_accepted, n_rejected, n_rhs : int32 counters.
    failed : bool, the controller's step fell below SciPy's ``min_step``.
    """
    t: jax.Array
    y: jax.Array
    f: jax.Array
    h_abs: jax.Array
    t_old: jax.Array
    h_last: jax.Array
    y_old: jax.Array
    ks: tuple
    n_accepted: jax.Array
    n_rejected: jax.Array
    n_rhs: jax.Array
    failed: jax.Array


def dp54_initial_state(rhs: Callable, t0, y0, first_step) -> Dp54State:
    """SciPy's solver after ``__init__``: f0 = rhs(t0, y0), h_abs = first_step."""
    f0 = rhs(t0, y0)
    zeros = jnp.zeros_like(y0)
    return Dp54State(
        t=jnp.asarray(t0, jnp.float64), y=y0, f=f0,
        h_abs=jnp.asarray(first_step, jnp.float64),
        t_old=jnp.asarray(t0, jnp.float64), h_last=jnp.asarray(0.0, jnp.float64),
        y_old=y0, ks=(zeros,) * 7, n_accepted=jnp.int32(0),
        n_rejected=jnp.int32(0), n_rhs=jnp.int32(1), failed=jnp.asarray(False))


def _min_step(t):
    """``10 * |nextafter(t, +inf) - t|`` (SciPy's ``min_step``)."""
    return 10.0 * jnp.abs(jnp.nextafter(t, jnp.inf) - t)


def _dp54_accepted_step(rhs, state, t_bound, max_step, atol, rtol, count):
    """Advance ``state`` by one accepted step (SciPy ``RungeKutta._step_impl``)."""
    t, y, f = state.t, state.y, state.f
    min_step = _min_step(t)
    h_abs = jnp.where(state.h_abs > max_step, max_step,
                      jnp.where(state.h_abs < min_step, min_step, state.h_abs))

    def body(c):
        h_abs, rejected, _, n_rej, _, _, _, _ = c
        too_small = h_abs < min_step
        t_new = jnp.minimum(t + h_abs, t_bound)
        h = t_new - t
        y_new, f_new, ks = dp54_step(rhs, t, y, f, h)
        err = dp54_error_norm(ks, h, y, y_new, atol, rtol, count)
        ok = err < 1.0
        grow = jnp.where(err == 0.0, DP54_MAX_FACTOR,
                         jnp.minimum(DP54_MAX_FACTOR,
                                     DP54_SAFETY * err ** DP54_ERROR_EXPONENT))
        grow = jnp.where(rejected, jnp.minimum(1.0, grow), grow)
        shrink = jnp.maximum(DP54_MIN_FACTOR,
                             DP54_SAFETY * err ** DP54_ERROR_EXPONENT)
        h_next = jnp.abs(h) * jnp.where(ok, grow, shrink)
        n_rej = n_rej + jnp.where(ok | too_small, 0, 1).astype(jnp.int32)
        return (h_next, rejected | ~ok, ok | too_small, n_rej, too_small,
                (y_new, f_new), ks, (t_new, h))

    zeros = jnp.zeros_like(y)
    init = (h_abs, jnp.asarray(False), jnp.asarray(False), jnp.int32(0),
            jnp.asarray(False), (zeros, zeros), (zeros,) * 7,
            (t, jnp.zeros_like(t)))
    h_next, _, _, n_rej, too_small, (y_new, f_new), ks, (t_new, h) = \
        jax.lax.while_loop(lambda c: ~c[2], body, init)
    n_tries = n_rej + jnp.where(too_small, 0, 1).astype(jnp.int32)
    keep = lambda old, new: jnp.where(too_small, old, new)
    return Dp54State(
        t=keep(t, t_new), y=keep(y, y_new), f=keep(f, f_new), h_abs=h_next,
        t_old=keep(state.t_old, t), h_last=keep(state.h_last, h),
        y_old=keep(state.y_old, y),
        ks=tuple(keep(a, b) for a, b in zip(state.ks, ks)),
        n_accepted=state.n_accepted + jnp.where(too_small, 0, 1).astype(jnp.int32),
        n_rejected=state.n_rejected + n_rej,
        n_rhs=state.n_rhs + 6 * n_tries,
        failed=state.failed | too_small)


def _emit_due(i, buf, st: Dp54State, sample_times, sample_fn):
    """Sample every pending time <= st.t from the interpolant on [t_old, t]."""
    n = int(sample_times.shape[0])
    if n == 0:
        return i, buf
    h = st.h_last
    safe_h = jnp.where(h == 0.0, 1.0, h)

    def c(cc):
        j, _ = cc
        return (j < n) & (sample_times[jnp.minimum(j, n - 1)] <= st.t)

    def b(cc):
        j, bb = cc
        ts = sample_times[j]
        x = jnp.where(h == 0.0, 0.0, (ts - st.t_old) / safe_h)
        val = sample_fn(ts, dp54_dense(st.ks, h, st.y_old, x))
        bb = jax.tree.map(lambda a, v: jax.lax.dynamic_update_index_in_dim(
            a, v, j, axis=0), bb, val)
        return j + 1, bb

    return jax.lax.while_loop(c, b, (i, buf))


def dp54_advance(rhs: Callable, state: Dp54State, *, t_bound, max_step,
                 atol, rtol, count, sample_times, sample_fn: Callable,
                 run_to_bound: bool = False):
    """Step until every time in ``sample_times`` is sampled (SciPy ``solve_ivp``).

    After each accepted step from t_old to t_new, every pending sample time
    ``t_s <= t_new`` is evaluated from that step's dense output, as
    ``solve_ivp`` does with ``searchsorted(t_eval, t, side='right')``.
    Steps are never truncated at a sample time: ``t_bound`` (the run's final
    time) is the only bound the controller sees, and the last step's
    interpolant travels in the state, so splitting one run into several
    calls reproduces the uninterrupted SciPy run.

    Parameters
    ----------
    sample_times : (n,) float64 array, ascending, all >= state.t_old.
    sample_fn : callable (t, y) -> pytree of arrays (one sample).
    run_to_bound : after the last sample keep stepping to ``t_bound``.
    count : the logical entry count for :func:`dp54_error_norm`.

    Returns
    -------
    state, samples
        ``samples`` stacks ``sample_fn`` outputs on a new leading axis of
        length n.  A failed controller stops the loop with ``state.failed``
        set; rows it never reached are zero.
    """
    n = int(sample_times.shape[0])
    template = jax.eval_shape(sample_fn, state.t, state.y)
    buf0 = jax.tree.map(lambda s: jnp.zeros((n,) + s.shape, s.dtype), template)
    i0, buf0 = _emit_due(jnp.int32(0), buf0, state, sample_times, sample_fn)

    def cond(c):
        st, i, _ = c
        pending = (i < n) | (st.t < t_bound) if run_to_bound else (i < n)
        return pending & ~st.failed & (st.t < t_bound)

    def body(c):
        st, i, buf = c
        st = _dp54_accepted_step(rhs, st, t_bound, max_step, atol, rtol, count)
        i, buf = _emit_due(i, buf, st, sample_times, sample_fn)
        return st, i, buf

    st, _, samples = jax.lax.while_loop(cond, body, (state, i0, buf0))
    return st, samples


# ---------------------------------------------------------------------------
# Classical RK4
# ---------------------------------------------------------------------------

def rk4_step(rhs: Callable, t, y, h, k1=None):
    """Classical fourth-order Runge–Kutta step for y' = rhs(t, y).

    ``k1`` = rhs(t, y) when the caller already has it (first-same-as-last
    carry of the fixed-step driver); the arithmetic is the same either way.
    """
    k1 = rhs(t, y) if k1 is None else k1
    k2 = rhs(t + 0.5 * h, y + (0.5 * h) * k1)
    k3 = rhs(t + 0.5 * h, y + (0.5 * h) * k2)
    k4 = rhs(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Exponential integrator: Krogstad's ETDRK4 for y' = lam*y + N(t, y)
# ---------------------------------------------------------------------------

_TAYLOR_RADIUS = 0.5
_TAYLOR_TERMS = 24


def phi_functions(z):
    """phi_0..phi_3 of a complex array, elementwise.

    phi_0(z) = e^z, phi_{j+1}(z) = (phi_j(z) - 1/j!)/z.  For |z| < 1/2 the
    Taylor series sum_k z^k/(k+j)! is used (24 terms: truncation below
    1e-30), above it the recurrence (cancellation below ~5e-15 relative).
    """
    z = jnp.asarray(z)
    small = jnp.abs(z) < _TAYLOR_RADIUS
    zs = jnp.where(small, jnp.zeros_like(z), z)
    zd = jnp.where(small, jnp.ones_like(z), z)
    e = jnp.exp(zs)
    p1 = (e - 1.0) / zd
    p2 = (p1 - 1.0) / zd
    p3 = (p2 - 0.5) / zd
    zt = jnp.where(small, z, jnp.zeros_like(z))
    t1 = t2 = t3 = jnp.zeros_like(z)
    for k in range(_TAYLOR_TERMS - 1, -1, -1):
        t1 = t1 * zt + 1.0 / math.factorial(k + 1)
        t2 = t2 * zt + 1.0 / math.factorial(k + 2)
        t3 = t3 * zt + 1.0 / math.factorial(k + 3)
    return (jnp.exp(z), jnp.where(small, t1, p1), jnp.where(small, t2, p2),
            jnp.where(small, t3, p3))


class EtdCoefficients(NamedTuple):
    """Krogstad ETDRK4 weights for one step h, elementwise in lam.

    e_half, e_full : exp(h lam / 2), exp(h lam).
    a21 = phi1(z/2)/2 ; a31 = phi1(z/2)/2 - phi2(z/2) ; a32 = phi2(z/2)
    a41 = phi1(z) - 2 phi2(z) ; a43 = 2 phi2(z)
    b1 = phi1 - 3 phi2 + 4 phi3 ; b23 = 2 phi2 - 4 phi3 ; b4 = 4 phi3 - phi2
    with z = h lam (Hochbruck & Ostermann 2010, the Krogstad tableau).
    """
    e_half: jax.Array
    e_full: jax.Array
    a21: jax.Array
    a31: jax.Array
    a32: jax.Array
    a41: jax.Array
    a43: jax.Array
    b1: jax.Array
    b23: jax.Array
    b4: jax.Array


def etdrk4_coefficients(lam, h) -> EtdCoefficients:
    """Krogstad ETDRK4 coefficients for the diagonal linear part ``lam``."""
    z = h * lam
    e_h, p1_h, p2_h, _ = phi_functions(0.5 * z)
    e_f, p1, p2, p3 = phi_functions(z)
    return EtdCoefficients(
        e_half=e_h, e_full=e_f, a21=0.5 * p1_h, a31=0.5 * p1_h - p2_h,
        a32=p2_h, a41=p1 - 2.0 * p2, a43=2.0 * p2,
        b1=p1 - 3.0 * p2 + 4.0 * p3, b23=2.0 * p2 - 4.0 * p3,
        b4=4.0 * p3 - p2)


def etdrk4_step(nonlin: Callable, coef: EtdCoefficients, t, y, h, n1=None):
    """One Krogstad ETDRK4 step of y' = lam ∘ y + nonlin(t, y).

    ``coef`` = :func:`etdrk4_coefficients` (lam, h), broadcastable to y.
    Four evaluations of ``nonlin`` (three when the caller passes
    ``n1`` = nonlin(t, y)); exact for nonlin = const and stiff order 4 under
    the conditions of Hochbruck & Ostermann (2010).
    """
    n1 = nonlin(t, y) if n1 is None else n1
    a = coef.e_half * y + h * coef.a21 * n1
    n2 = nonlin(t + 0.5 * h, a)
    b = coef.e_half * y + h * (coef.a31 * n1 + coef.a32 * n2)
    n3 = nonlin(t + 0.5 * h, b)
    c = coef.e_full * y + h * (coef.a41 * n1 + coef.a43 * n3)
    n4 = nonlin(t + h, c)
    return coef.e_full * y + h * (coef.b1 * n1 + coef.b23 * (n2 + n3)
                                  + coef.b4 * n4)


def hermite_dense(lam, y0, n0, y1, n1, h, theta):
    """y(t + θh) for y' = lam ∘ y + n(t), cubic Hermite in the frame of ``lam``.

    u(s) = e^{-lam s} ∘ y(t + s) has u' = e^{-lam s} ∘ n, so the phases and
    decays of ``lam`` are carried exactly and only the slow amplitude is
    interpolated: u(θh) from u(0) = y0, u'(0) = n0, u(h) = e^{-lam h} y1,
    u'(h) = e^{-lam h} n1 (the four cubic Hermite basis polynomials), then
    y(t + θh) = e^{lam θh} ∘ u(θh).  Local error O(h^4 |u''''|), the global
    order of a fourth-order step; θ = 0 and θ = 1 return y0 and y1 to
    rounding.  ``n`` is the derivative minus ``lam ∘ y`` at each end.
    """
    back = jnp.exp(-lam * h)
    th = theta
    h00 = (2.0 * th - 3.0) * th * th + 1.0
    h10 = ((th - 2.0) * th + 1.0) * th
    h01 = (3.0 - 2.0 * th) * th * th
    h11 = (th - 1.0) * th * th
    u = h00 * y0 + (h * h10) * n0 + h01 * (back * y1) + (h * h11) * (back * n1)
    return jnp.exp(lam * (th * h)) * u


# ---------------------------------------------------------------------------
# Magnus in the interaction picture of a free flow, on a deviation
# ---------------------------------------------------------------------------
#
# The equation is y' = -i[A0 + A(t) + B(t, y), y] (+ a remainder flow, see
# compose_strang) with three caller callables:
#
#   free(X, s)          = e^{-isA0} X e^{isA0}, the exact free flow (A0 constant);
#   conj_dev(W, x, tau) = e^{-i tau W} (y_eq + x) e^{i tau W} - y_eq, the unitary
#                         conjugation of the full state written for the
#                         deviation x = y - y_eq, every term of the size of x
#                         (y_eq is stationary: free(y_eq, s) = y_eq);
#   generator(t)        = A(t) + B(t), Schrödinger picture.
#
# In the frame of the step start t, x_I(s) = free(x(t + s), -s) obeys
# x_I' = -i[G_I(s), y_eq + x_I] with G_I(s) = free(generator(t + s), -s): the
# exponentials see the coupling only, never A0, and A0's phases are exact.

#: Gauss–Legendre nodes c1 < c2 on [0, 1] and the CFM4 weights
#: alpha1 = (3 - 2 sqrt3)/12, alpha2 = (3 + 2 sqrt3)/12 (Blanes & Moan 2006;
#: Alvermann & Fehske, J. Comput. Phys. 230, 5930 (2011), CFET4:2).
CFM4_NODES = (0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0)
CFM4_WEIGHTS = ((3.0 - 2.0 * np.sqrt(3.0)) / 12.0,
                (3.0 + 2.0 * np.sqrt(3.0)) / 12.0)
#: The exponential midpoint rule's one node.
MIDPOINT_NODES = (0.5,)
#: Gauss–Legendre rule on [0, 1] for the moments of the interaction-picture
#: generator in :func:`ip_cf4_flow`: (nodes, weights).  With the two CFM4
#: nodes the moment form IS CFET4:2.
CF4_QUADRATURE = (CFM4_NODES, (0.5, 0.5))


def extrapolation_weights(past, target) -> np.ndarray:
    """Lagrange weights W with f(target_j) ~ sum_i W[j, i] f(past_i).

    ``past`` and ``target`` are offsets in units of the step, so the weights
    are constants of the scheme (independent of h).  With m past points the
    extrapolation error is O(h^m).
    """
    past = np.asarray(past, float)
    target = np.atleast_1d(np.asarray(target, float))
    w = np.ones((target.size, past.size))
    for i, xi in enumerate(past):
        for k, xk in enumerate(past):
            if k != i:
                w[:, i] *= (target - xk) / (xi - xk)
    return w


#: Midpoint predictor: B at the new midpoint (+1/2) from the last three
#: midpoints (-5/2, -3/2, -1/2), quadratic, O(h^3).
MIDPOINT_HISTORY = (-2.5, -1.5, -0.5)
#: CFM4 predictor: B at the new Gauss nodes from the last two steps' four
#: node values, cubic, O(h^4).
CFM4_HISTORY = (-2.0 + 0.5 - np.sqrt(3.0) / 6.0, -2.0 + 0.5 + np.sqrt(3.0) / 6.0,
                -1.0 + 0.5 - np.sqrt(3.0) / 6.0, -1.0 + 0.5 + np.sqrt(3.0) / 6.0)


def frozen_generator(field: Callable, free: Callable, t, h, nodes, values) -> Callable:
    """generator(t') = A(t') + B_frozen(t') with B frozen as a polynomial in time.

    ``values[i]`` is B at t + nodes[i] h in the interaction picture of t;
    B_frozen(t + s) = free(p(s), s) with p the Lagrange polynomial through
    the node values (constant for one node, linear for two).  Defined for
    every t', so a composition that steps outside [t, t + h] (the triple
    jump) still integrates one fixed smooth equation.
    """
    nodes = tuple(float(c) for c in nodes)

    def generator(ta):
        s = ta - t
        u = s / h
        p = None
        for i, ci in enumerate(nodes):
            li = 1.0
            for j, cj in enumerate(nodes):
                if j != i:
                    li = li * ((u - cj) / (ci - cj))
            p = li * values[i] if p is None else p + li * values[i]
        return field(ta) + free(p, s)
    return generator


def ip_cf4_flow(generator: Callable, free: Callable, conj_dev: Callable, t, x, h,
                quadrature=CF4_QUADRATURE):
    """Commutator-free fourth-order Magnus over [t, t + h], interaction picture.

    With the moments of G_I(s) = free(generator(t + s), -s),

        B0 = (1/h) ∫_0^h G_I(s) ds,   B1 = (12/h) ∫_0^h (s/h - 1/2) G_I(s) ds

    (``quadrature`` = Gauss–Legendre (nodes, weights) on [0, 1]),

        x_I(h) = Ψ(B0/2 + B1/6) Ψ(B0/2 - B1/6) x,   Ψ(W) x = conj_dev(W, x, h),

    the right exponential applied first (= exp(-ih(α2 G1 + α1 G2)) of
    CFET4:2 at the two Gauss nodes; Blanes, Casas & Ros, BIT 40, 434 (2000)
    for the moment form), and x(t + h) = free(x_I(h), h).  Fourth order;
    unitary in y = y_eq + x for any h; two calls of ``conj_dev``.
    """
    xq, wq = quadrature
    b0 = b1 = None
    for xi, wi in zip(xq, wq):
        s = float(xi) * h
        g = free(generator(t + s), -s)
        b0 = float(wi) * g if b0 is None else b0 + float(wi) * g
        c = 12.0 * float(wi) * (float(xi) - 0.5)
        if c != 0.0:
            b1 = c * g if b1 is None else b1 + c * g
    y = conj_dev(0.5 * b0 - b1 / 6.0, x, h)
    y = conj_dev(0.5 * b0 + b1 / 6.0, y, h)
    return free(y, h)


def ip_midpoint_flow(generator: Callable, free: Callable, conj_dev: Callable, t, x, h):
    """Exponential midpoint rule, interaction picture: x_I(h) = Ψ(G_I(h/2)) x.

    Second order; unitary in y = y_eq + x; one call of ``conj_dev``.
    """
    g = free(generator(t + 0.5 * h), -0.5 * h)
    return free(conj_dev(g, x, h), h)


def ip_magnus_step(field: Callable, state_part: Callable, inner: Callable,
                   free: Callable, t, x, h, guess, *, nodes, n_corr):
    """One predictor–corrector Magnus step on the deviation x = y - y_eq.

    B is frozen as the polynomial through its values at the ``nodes``
    (offsets in units of h; :data:`CFM4_NODES` or :data:`MIDPOINT_NODES`),
    held in the interaction picture of t.  ``guess`` predicts them.  Each
    of the ``n_corr`` corrections (may be traced) propagates x from t to
    every node with ``inner(generator, t, x, tau)`` over the frozen
    generator and evaluates B there (``state_part(t', x)``, the expensive
    call): len(nodes) evaluations per correction.  The step is
    ``inner`` over [t, t + h] with the final node values, so B is evaluated
    only at the nodes however ``inner`` is composed (a fourth-order
    composition of a remainder costs exponentials, not B).  A correction
    contracts the node error by O(h ||dB/dy||); a predictor of order m
    reaches the scheme's order after max(1, order - m) corrections.

    Returns (x(t + h), node values in the frame of t).
    """
    nodes = tuple(float(c) for c in nodes)

    def corr(_, g):
        gen = frozen_generator(field, free, t, h, nodes, g)
        return tuple(free(state_part(t + c * h, inner(gen, t, x, c * h)), -c * h)
                     for c in nodes)

    g = jax.lax.fori_loop(0, n_corr, corr, tuple(guess))
    return inner(frozen_generator(field, free, t, h, nodes, g), t, x, h), g


def _node_quadrature(nodes):
    """A[i, j] = ∫_0^{c_i} ℓ_j, b[j] = ∫_0^1 ℓ_j for the Lagrange basis ℓ_j on ``nodes``."""
    from numpy.polynomial import polynomial as npoly
    nodes = np.asarray(nodes, float)
    m = nodes.size
    a, b = np.zeros((m, m)), np.zeros(m)
    for j in range(m):
        c = np.array([1.0])
        for k in range(m):
            if k != j:
                c = npoly.polymul(c, np.array([-nodes[k], 1.0]) / (nodes[j] - nodes[k]))
        ic = npoly.polyint(c)
        b[j] = npoly.polyval(1.0, ic)
        a[:, j] = npoly.polyval(nodes, ic)
    return a, b


def magnus_amplification(nodes, history, n_corr: int, z) -> np.ndarray:
    """The linearized predictor–corrector Magnus step on u' = λu, z = hλ.

    For a coupling B linear in the deviation (linear response), every
    eigenmode of the coupling map evolves alone, B = λu up to a constant,
    and :func:`ip_magnus_step` reduces to the recurrence on
    s = (u_n, G of the ``history`` nodes), G = h B:

        G <- W G_hist                      (the predictor; W of extrapolation_weights)
        G <- z u_n 1 + z A G               (each of n_corr corrections)
        u_{n+1} = u_n + b·G,  history <- the newest len(history) node values,

    A, b the integrals of the Lagrange basis on the nodes (for the two Gauss
    nodes: the 2-stage Gauss collocation matrix, |eig A| = 1/(2 sqrt 3), so
    the corrections are its fixed-point iteration).  Returns the
    (1 + len(history))-square matrix of that map.
    """
    a, b = _node_quadrature(nodes)
    w = extrapolation_weights(history, list(nodes))
    m, nh = w.shape
    out = np.zeros((1 + nh, 1 + nh), complex)
    for col in range(1 + nh):
        s0 = np.zeros(1 + nh, complex)
        s0[col] = 1.0
        u, hist = s0[0], s0[1:]
        g = w @ hist
        for _ in range(int(n_corr)):
            g = z * u * np.ones(m) + z * (a @ g)
        out[:, col] = np.concatenate([[u + b @ g], np.concatenate([hist, g])[-nh:]])
    return out


def magnus_stability_limit(nodes, history, n_corr: int, *, growth: float = 1e-6,
                           y_cap: float = 8.0, dy: float = 1e-3) -> float:
    """Largest y with spectral radius of magnus_amplification(i y') <= 1 + growth for all y' <= y.

    The stability interval of the predictor–corrector step on the imaginary
    axis (Hermitian couplings: TDA, or a full BSE with real frequencies): a
    step h is stable for a coupling map of spectral radius σ when h σ <= y.
    ``growth`` is the largest per-step growth of any mode accepted (1e-6:
    1 % over 10^4 steps); scanned on a grid of spacing ``dy`` up to
    ``y_cap``.  Values (growth 1e-6, dy 1e-3), CFM4 nodes: cubic predictor
    (CFM4_HISTORY) 0.117 / 0.730 / 1.144 for 1 / 2 / 3 corrections, linear
    (the last step's two nodes) 0.625 / 1.203 / 0.231; the midpoint rule's
    quadratic predictor 0.126 at 2.  Not monotone in the corrections: the
    truncated fixed-point (Picard) iteration of a collocation is unstable on
    the imaginary axis at some truncation orders, as explicit RK of order 1
    and 2 are.  ``tests/tdagw/test_prop_integrators_cpu.py`` checks the
    recurrence against the growth of the real step.
    """
    for y in np.arange(dy, y_cap + 0.5 * dy, dy):
        rho = np.max(np.abs(np.linalg.eigvals(
            magnus_amplification(nodes, history, n_corr, 1j * y))))
        if rho > 1.0 + growth:
            return float(y - dy)
    return float(y_cap)


# ---------------------------------------------------------------------------
# Splitting of a remainder flow: Strang and its fourth-order composition
# ---------------------------------------------------------------------------

_CBRT2 = 2.0 ** (1.0 / 3.0)
#: Yoshida's triple jump (Phys. Lett. A 150, 262 (1990)): S4(h) =
#: S2(g1 h) S2(g0 h) S2(g1 h), g1 = 1/(2 - 2^(1/3)), g0 = 1 - 2 g1; fourth
#: order for any symmetric second-order S2.
TRIPLE_JUMP = (1.0 / (2.0 - _CBRT2), -_CBRT2 / (2.0 - _CBRT2), 1.0 / (2.0 - _CBRT2))


def compose_strang(core: Callable, remainder: Callable, t, y, h, fractions=(1.0,)):
    """∏_j S2(f_j h), S2(τ) at a = R(a + τ/2 -> a + τ) ∘ M(a -> a + τ) ∘ R(a -> a + τ/2).

    ``core(a, y, tau)`` and ``remainder(a, y, tau)`` advance y over
    [a, a + tau] (tau may be negative).  Adjacent remainder pieces are merged
    into one call over the joined interval, so the composition calls
    ``remainder`` len(fractions) + 1 times: R(f1 h/2) M(f1 h)
    R((f1 + f2) h/2) … M(fn h) R(fn h/2).  ``(1,)`` is Strang (second order
    in the splitting error [M, R]); :data:`TRIPLE_JUMP` cancels the h^3
    error of the symmetric S2 and is fourth order (Yoshida 1990), with
    fourth-order (or exact) core and remainder flows.
    """
    f = tuple(float(v) for v in fractions)
    y = remainder(t, y, 0.5 * f[0] * h)
    a = t
    for j, fj in enumerate(f):
        y = core(a, y, fj * h)
        r0 = a + 0.5 * fj * h
        a = a + fj * h
        tail = 0.5 * fj * h + (0.5 * f[j + 1] * h if j + 1 < len(f) else 0.0)
        y = remainder(r0, y, tail)
    return y


# ---------------------------------------------------------------------------
# Fixed-step loop with samples from the dense output
# ---------------------------------------------------------------------------

def fixed_advance(step: Callable, dense: Callable, state_of: Callable, carry, *,
                  t_start, h, step0, n_steps: int, sample_step, sample_theta,
                  sample_times, sample_fn: Callable):
    """Take ``n_steps`` fixed steps from step ``step0``, emitting every sample on the way.

    Step j covers [t_j, t_j + h] with t_j = t_start + j h from the integer
    counter (no rounding accumulates over a long run).
    ``step(t, carry, h, t_next) -> (carry, aux)`` with t_next = t_{j+1} from
    the same counter (a first-same-as-last derivative evaluated at t_next
    is bitwise the next step's own first stage);
    ``dense(t, carry_old, carry_new, aux, theta) -> state`` at t + θh;
    ``state_of(carry) -> state``; ``sample_fn(t, state) -> pytree``.

    Sample i lies in step ``sample_step[i]`` at fraction ``sample_theta[i]``
    in [0, 1) (ascending in time, all in [step0, step0 + n_steps)): θ = 0
    samples ``state_of`` at t_j before the step (bitwise the stepped state,
    at time t_j), θ > 0 the dense output after the step (at
    ``sample_times[i]``).  Returns (carry, samples), samples stacked on a new
    leading axis of length len(sample_times).
    """
    n = int(sample_times.shape[0])
    template = jax.eval_shape(sample_fn, jnp.asarray(t_start, jnp.float64),
                              state_of(carry))
    buf0 = jax.tree.map(lambda s: jnp.zeros((n,) + s.shape, s.dtype), template)
    last = max(n - 1, 0)

    def put(buf, i, val):
        return jax.tree.map(lambda a, v: jax.lax.dynamic_update_index_in_dim(
            a, v, i, axis=0), buf, val)

    def due(i, j):
        k = jnp.minimum(i, last)
        return (i < n) & (sample_step[k] == j)

    def boundary(i, buf, j, t, c):
        cond = lambda ib: due(ib[0], j) & (sample_theta[jnp.minimum(ib[0], last)] == 0.0)
        body = lambda ib: (ib[0] + 1, put(ib[1], ib[0], sample_fn(t, state_of(c))))
        return jax.lax.while_loop(cond, body, (i, buf))

    def interior(i, buf, j, t, c0, c1, aux):
        def body(ib):
            k, b = ib
            val = sample_fn(sample_times[k], dense(t, c0, c1, aux, sample_theta[k]))
            return k + 1, put(b, k, val)
        return jax.lax.while_loop(lambda ib: due(ib[0], j), body, (i, buf))

    def one(cib, _):
        c, i, buf, j = cib
        t = t_start + j.astype(jnp.float64) * h
        if n:
            i, buf = boundary(i, buf, j, t, c)
        c1, aux = step(t, c, h, t_start + (j + 1).astype(jnp.float64) * h)
        if n:
            i, buf = interior(i, buf, j, t, c, c1, aux)
        return (c1, i, buf, j + 1), None

    init = (carry, jnp.int32(0), buf0, jnp.asarray(step0, jnp.int64))
    (carry, _, buf, _), _ = jax.lax.scan(one, init, None, length=int(n_steps), unroll=1)
    return carry, buf

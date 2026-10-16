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
* **Magnus, unitary** (:func:`magnus_midpoint_step`, :func:`cfm4_step`):
  the isospectral flow y' = -i[H(t, y), y] with H = A(t) + B(t, y), where A is
  cheap and known at every t and B is the expensive state-dependent part.
  Each exponential is applied by the caller's ``conjugate(H, y, tau)`` =
  exp(-i tau H) y exp(+i tau H); the predictor extrapolates B from earlier
  steps so the step itself pays only the corrector evaluations of B.
* **Strang splitting** (:func:`strang_step`): S(h) = R(h/2) ∘ M(h) ∘ R(h/2)
  for a remainder flow R that the unitary step does not contain.

Stepping loops live in :func:`dp54_advance` (adaptive, emits samples from
the dense output at caller-given times) and :func:`fixed_advance` (fixed
step, a sample every ``n_sub`` steps); both are ``lax`` loops with no host
round trip.
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
    "extrapolation_weights", "MIDPOINT_HISTORY", "CFM4_HISTORY",
    "magnus_midpoint_step", "cfm4_step", "strang_step", "fixed_advance",
    "CFM4_NODES", "CFM4_WEIGHTS",
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

def rk4_step(rhs: Callable, t, y, h):
    """Classical fourth-order Runge–Kutta step for y' = rhs(t, y)."""
    k1 = rhs(t, y)
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


def etdrk4_step(nonlin: Callable, coef: EtdCoefficients, t, y, h):
    """One Krogstad ETDRK4 step of y' = lam ∘ y + nonlin(t, y).

    ``coef`` = :func:`etdrk4_coefficients` (lam, h), broadcastable to y.
    Four evaluations of ``nonlin``; exact for nonlin = const and stiff
    order 4 under the conditions of Hochbruck & Ostermann (2010).
    """
    n1 = nonlin(t, y)
    a = coef.e_half * y + h * coef.a21 * n1
    n2 = nonlin(t + 0.5 * h, a)
    b = coef.e_half * y + h * (coef.a31 * n1 + coef.a32 * n2)
    n3 = nonlin(t + 0.5 * h, b)
    c = coef.e_full * y + h * (coef.a41 * n1 + coef.a43 * n3)
    n4 = nonlin(t + h, c)
    return coef.e_full * y + h * (coef.b1 * n1 + coef.b23 * (n2 + n3)
                                  + coef.b4 * n4)


# ---------------------------------------------------------------------------
# Magnus integrators for y' = -i[A(t) + B(y), y]
# ---------------------------------------------------------------------------

#: Gauss–Legendre nodes c1 < c2 on [0, 1] and the CFM4 weights
#: alpha1 = (3 - 2 sqrt3)/12, alpha2 = (3 + 2 sqrt3)/12 (Blanes & Moan 2006;
#: Alvermann & Fehske, J. Comput. Phys. 230, 5930 (2011), CFET4:2).
CFM4_NODES = (0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0)
CFM4_WEIGHTS = ((3.0 - 2.0 * np.sqrt(3.0)) / 12.0,
                (3.0 + 2.0 * np.sqrt(3.0)) / 12.0)


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


def magnus_midpoint_step(explicit: Callable, state_part: Callable,
                         conjugate: Callable, t, y, h, b_guess, *, n_iter):
    """Exponential midpoint rule with a self-consistent midpoint Hamiltonian.

    y_{n+1} = exp(-i h H_m) y_n exp(+i h H_m),
    H_m = A(t + h/2) + B(t + h/2, (y_n + y_{n+1})/2),
    solved by ``n_iter`` fixed-point corrections (each one evaluation of B)
    from ``b_guess``, a prediction of B at the midpoint.  ``n_iter`` may be
    traced.  Second order; unitary for any ``n_iter``.  Returns
    (y_{n+1}, B_m); B_m feeds the next predictor.
    """
    a_mid = explicit(t + 0.5 * h)

    def corr(_, b):
        return state_part(t + 0.5 * h, 0.5 * (y + conjugate(a_mid + b, y, h)))

    b = jax.lax.fori_loop(0, n_iter, corr, b_guess)
    return conjugate(a_mid + b, y, h), b


def cfm4_step(explicit: Callable, state_part: Callable, conjugate: Callable,
              t, y, h, b_guess, *, n_corr):
    """Fourth-order commutator-free Magnus step (CFET4:2).

    y_{n+1} = V2 V1 y_n V1^H V2^H with
    V1 = exp(-i h (alpha2 H1 + alpha1 H2)) applied first,
    V2 = exp(-i h (alpha1 H1 + alpha2 H2)),
    H_i = A(t + c_i h) + B(t + c_i h, y(t + c_i h)) at the Gauss nodes c_i.

    ``b_guess = (B1, B2)`` predicts B at the two nodes.  Each of the
    ``n_corr`` corrections (may be traced) propagates y_n to both nodes by
    the same CFM4 formula on [t, t + c_i h] with B interpolated linearly
    through the current node values (A exact) and re-evaluates B there: two
    evaluations of B and four exponentials per correction, two exponentials
    for the step.  A correction contracts the node error by O(h ||dB/dy||),
    so a predictor of order m reaches the scheme's order 4 after
    max(1, 4 - m) corrections: at least one, because the returned node
    values feed the next predictor, and with none B is extrapolated from its
    own guesses and never sees y (the cubic history predictor then stays at
    error 0.27 under step halving in the unit test, where one correction
    gives order 4.5-5.5 and two give 3.95-3.99).  One correction is unstable
    on the CrI3 COHSEX kernel from 50 as, so the TD-aGW driver defaults to
    two.  Returns (y_{n+1}, (B1, B2)).
    """
    c1, c2 = CFM4_NODES
    al1, al2 = CFM4_WEIGHTS

    def to_node(c, b1, b2):
        # CFM4 over [t, t + c h], B linear through (c1 h, b1), (c2 h, b2)
        s1, s2 = c1 * c * h, c2 * c * h
        lin = lambda s: b1 + ((s - c1 * h) / ((c2 - c1) * h)) * (b2 - b1)
        h1 = explicit(t + s1) + lin(s1)
        h2 = explicit(t + s2) + lin(s2)
        y1 = conjugate(al2 * h1 + al1 * h2, y, c * h)
        return conjugate(al1 * h1 + al2 * h2, y1, c * h)

    def corr(_, b):
        b1, b2 = b
        return (state_part(t + c1 * h, to_node(c1, b1, b2)),
                state_part(t + c2 * h, to_node(c2, b1, b2)))

    b1, b2 = jax.lax.fori_loop(0, n_corr, corr, tuple(b_guess))
    h1 = explicit(t + c1 * h) + b1
    h2 = explicit(t + c2 * h) + b2
    y1 = conjugate(al2 * h1 + al1 * h2, y, h)
    return conjugate(al1 * h1 + al2 * h2, y1, h), (b1, b2)


def strang_step(core: Callable, remainder: Callable, t, y, h):
    """S(h) = R(t + h/2 -> t + h) ∘ M(t -> t + h) ∘ R(t -> t + h/2).

    ``core(t, y, h)`` and ``remainder(t, y, tau)`` advance y over [t, t+h]
    and [t, t+tau].  Second order in the splitting error [M, R].  ``core``
    may return (y, aux); the aux is passed through.
    """
    y = remainder(t, y, 0.5 * h)
    out = core(t, y, h)
    y, aux = out if isinstance(out, tuple) else (out, None)
    y = remainder(t + 0.5 * h, y, 0.5 * h)
    return (y, aux) if isinstance(out, tuple) else y


# ---------------------------------------------------------------------------
# Fixed-step loop with periodic samples
# ---------------------------------------------------------------------------

def fixed_advance(step: Callable, carry, *, t_start, h, n_sub: int,
                  n_samples: int, sample_fn: Callable, step0=0):
    """Sample, then take ``n_sub`` fixed steps, ``n_samples`` times.

    ``step(t, carry, h) -> carry``; ``sample_fn(t, carry) -> pytree``.
    Times are ``t_start + (step0 + i) * h`` from an integer step counter,
    so no rounding accumulates over a long run.  Returns
    (carry, samples) with samples stacked along a new leading axis; the
    carry is at time ``t_start + (step0 + n_samples * n_sub) * h``.
    """
    def one(carry_i, _):
        carry, i = carry_i
        t = t_start + i.astype(jnp.float64) * h
        out = sample_fn(t, carry)

        def sub(ci, _):
            c, j = ci
            return (step(t_start + j.astype(jnp.float64) * h, c, h), j + 1), None

        (carry, i), _ = jax.lax.scan(sub, (carry, i), None, length=int(n_sub),
                                     unroll=1)
        return (carry, i), out

    (carry, _), samples = jax.lax.scan(
        one, (carry, jnp.asarray(step0, jnp.int64)), None,
        length=int(n_samples), unroll=1)
    return carry, samples

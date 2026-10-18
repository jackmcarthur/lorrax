"""
Anderson and rCROP acceleration for the self-consistent loops.

These methods accelerate fixed-point iterations of the form:
    x^{(k+1)} = g(x^{(k)}) = x^{(k)} + f(x^{(k)})

where f(x) = 0 at the solution. In the context of GW self-consistency:
    x  ↔ Σ^{in}_{mnk}               (the iterate, kept in its own shape and layout)
    f(x) = Σ^{out}[x] - Σ^{in}[x]  (residual)

One method, host-driven (not jitted), because the residual function calls
pre-jitted pipelines: :func:`anderson_nojit` — Anderson type II, one map
evaluation per iteration; the SC loop (``gw.sc_iteration``) uses it.  (rCROP,
two evaluations per iteration, retired with its one caller, fixed-Sigma EQP2.)

References:
    Wan & Międlar, "On the Convergence of CROP-Anderson Acceleration Method"
"""

from __future__ import annotations

import os
from typing import Callable, NamedTuple

# Enable 64-bit precision before importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
from lxkit import ordered_reduce
import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------


class AccelerationResult(NamedTuple):
    """Result from an acceleration method."""

    x: jnp.ndarray  # Final iterate
    residual_norms: jnp.ndarray  # Residual norm at each iteration
    iterations: int  # Number of iterations performed
    converged: bool  # Whether tolerance was reached


class AndersonState(NamedTuple):
    """Everything :func:`anderson_nojit` carries from one evaluation to the next.

    ``X``/``F`` are the history stacks ``(m,) + shape`` in ring order
    (``head``, ``filled``) and ``(x, f)`` the newest pair; ``best`` is the
    evaluated pair of least residual, ``None`` when it lies in that window
    (the first minimum of ``res`` locates it); ``res`` holds every residual
    norm; ``fallback`` says whether the next step is the secant fallback.  The
    next iterate is a function of these alone, so a fresh process handed this
    state continues the same trajectory.
    """

    X: object
    F: object
    head: int
    filled: int
    x: object
    f: object
    best: tuple | None
    res: tuple
    fallback: bool


def _pin_entry(v, entry_sharding):
    """Pin one history entry to the operand layout (identity if unsharded).

    The one small device placement of anderson_nojit:
    an entry is carry-shaped (the SC Hamiltonian), already sharded like the
    operand, so this is a layout pin, not a host payload transfer.
    """
    return v if entry_sharding is None else jax.device_put(v, entry_sharding)


# -----------------------------------------------------------------------------
# Least-squares solvers
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# The accelerator the SC loop calls
# -----------------------------------------------------------------------------


def _rows_sq_norm(a):
    """Per row of a (k, *state) stack: sum |a|^2 over the state (a rank's partial)."""
    return jnp.sum(jnp.abs(a) ** 2, axis=tuple(range(1, a.ndim)))


def _rows_gram(a):
    ax = tuple(range(1, a.ndim))
    return jnp.tensordot(jnp.conj(a), a, axes=(ax, ax))


def _rows_dot(a, f):
    return jnp.tensordot(jnp.conj(a), f, axes=(tuple(range(1, a.ndim)), tuple(range(f.ndim))))


def _sq_norm(v):
    return jnp.sum(jnp.abs(v) ** 2)


def _solve_alpha_filtered(Fw: jnp.ndarray, cond_max: float = 1.0e12):
    """The CROP affine least squares with the Walker-Ni conditioning filter.

    ``Fw`` is ``(k+1,) + operand_shape``: history entries in chronological
    order (zero-norm slots invalid), newest last; ``alpha`` minimises
    ``||sum_i alpha_i Fw_i||`` subject to ``sum_i alpha_i = 1``, each
    difference normalised before the Gram is formed.  The OLDEST valid differences are dropped until the unit-column Gram
    has condition number <= ``cond_max`` (1e12 on the Gram = 1e6 on R, the
    DFTK / Walker-Ni cap).  One (k, k) Gram and one length-k reduction, as
    before; the reduced solves are on the host.  Returns (alpha, n_used).
    """
    k = Fw.shape[0] - 1
    ax = tuple(range(1, Fw.ndim))
    bshape = (k,) + (1,) * (Fw.ndim - 1)
    f_new = Fw[-1]
    F_hist = Fw[:k]
    # Every reduction over the (sharded) state runs in rank order (lxkit.ordered_reduce):
    # as jnp.sum/tensordot on the sharded history XLA added the ranks' partials with a
    # library all-reduce in device-topology order, so the mixing step, and every later
    # map, depended on the GCD placement (det lane 2026-10-16).
    hist_norms = np.sqrt(np.asarray(ordered_reduce(_rows_sq_norm, F_hist)))
    valid = hist_norms > 1e-14
    F_prev = jnp.where(jnp.asarray(valid).reshape(bshape),
                       F_hist - f_new[None], 0.0 + 0.0j)
    col = np.sqrt(np.asarray(ordered_reduce(_rows_sq_norm, F_prev)))
    scale = np.where(col > 0.0, col, 1.0)
    F_s = F_prev / jnp.asarray(scale).reshape(bshape)
    G = np.real(np.asarray(ordered_reduce(_rows_gram, F_s)))
    b = -np.real(np.asarray(ordered_reduce(_rows_dot, F_s, f_new)))
    use = [i for i in range(k) if valid[i] and col[i] > 0.0]
    while use:
        Gu = G[np.ix_(use, use)]
        w = np.linalg.eigvalsh(Gu)
        if w[0] > 0.0 and w[-1] / w[0] <= cond_max:
            break
        use = use[1:]                       # drop the oldest difference
    gamma = np.zeros(k)
    if use:
        Gu = G[np.ix_(use, use)] + 1e-12 * np.eye(len(use))
        gamma[use] = np.linalg.solve(Gu, b[use]) / scale[use]
    alpha = np.concatenate([gamma, [1.0 - gamma.sum()]])
    return jnp.asarray(alpha, dtype=Fw.dtype), len(use)


def anderson_nojit(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    m: int = 5,
    maxit: int = 100,
    tol: float = 1e-10,
    print_fn: Callable = None,
    entry_sharding=None,
    metric=None,
    resume: Callable[[], AndersonState] | None = None,
    on_eval: Callable[[AndersonState], None] | None = None,
) -> AccelerationResult:
    """Anderson type II (Pulay/DIIS): ONE map evaluation per iteration.

    ``on_eval`` receives the :class:`AndersonState` after every evaluation
    but the budget's last; ``resume`` loads one (a callable, so the caller
    keeps no reference to its buffers) and skips the evaluation of ``x0``,
    which then fixes only shape and layout: the run continues at the step
    that state implies, with ``maxit`` counting from the original first step.

    Every evaluated pair (x_i, f_i = G(x_i) - x_i) enters a history of the
    newest ``m + 1``; the next and only evaluation is at

        x_{k+1} = sum_i alpha_i (x_i + f_i),
        alpha = argmin || sum_i alpha_i f_i ||_metric,  sum_i alpha_i = 1,

    with beta = 1 (no damping), except the first step, x_1 = x_0 + f_0 / 2.
    On an affine map every later iterate is the same for any first beta != 0
    (the history spans the same affine set), while along a mode with Picard
    eigenvalue lambda the map-1 residual is |1 + lambda| / 2 of map 0's
    instead of |lambda|: 0.75 instead of 2.5 on the CrI3 bispinor mode.
    The step index is the global map index, so a resume from a map-0
    state takes the same half step.  This is CROP with real residuals (Wan &
    Miedlar 2024, "CROP-Anderson"); rCROP reaches the same iterates on an
    affine map at two evaluations per iteration, the second re-evaluating a
    residual the linear model already predicts.

    Two safeguards, neither costing a map evaluation nor carrying a tunable
    constant (the threshold is the literature's numerical default):

    * conditioning filter (Walker & Ni 2011; DFTK): the oldest differences
      are dropped until the unit-column Gram has cond <= 1e12 (cond(R) <=
      1e6), :func:`_solve_alpha_filtered`;
    * nonmonotone fallback (after Ouyang et al. 2023): an evaluation worse
      than every residual in the window means the multisecant model failed
      there, so the next point is the two-point secant between the best
      evaluated pair and the rejected one -- a free line search along the
      failed step; never twice in a row; the rejected pair stays in the
      history (it is valid secant data).

    There is deliberately NO restart on the map's discrete events: early QSGW
    maps grow the sampled Sigma grid on every call, and restarting there
    reduced the method to Picard steps, which diverge on an expansive map
    (CrI3 / Fe bispinor, 2026-09-24).  The jumps are far below the
    secant-model error at that stage, and stale pairs are down-weighted by
    the least squares itself.

    alpha is solved over the REAL numbers: the iterates are Hermitian
    matrices, a real vector space, and the Gram of Hermitian residuals is
    real.  THE ITERATE KEEPS ITS OWN SHAPE AND LAYOUT: the history is a stack on a leading, never-sharded
    axis at ``P(None, *entry_sharding.spec)``, the only collective is one
    (m+1, m+1) Gram plus a length-(m+1) reduction, and the update is an
    elementwise combination over the history axis.  Residency: 2(m+1)
    history entries.
    """
    shape = x0.shape
    dtype = x0.dtype

    if entry_sharding is None:
        stack_sharding = None
    else:
        from jax.sharding import NamedSharding, PartitionSpec as P
        stack_sharding = NamedSharding(
            entry_sharding.mesh, P(None, *entry_sharding.spec))

    def _entry(v):
        return _pin_entry(v, entry_sharding)

    def _zeros_hist():
        return jnp.zeros((m,) + shape, dtype=dtype, device=stack_sharding)

    def _weighted(v):
        return v if metric is None else v * metric

    def _norm(v):
        return float(np.sqrt(np.asarray(ordered_reduce(_sq_norm, _weighted(v)))))

    fresh = resume is None
    if fresh:
        Xhist, Fhist, head, filled = _zeros_hist(), _zeros_hist(), 0, 0
        x = _entry(x0)
        f = _entry(residual_fn(x))
        res_history, fallback, best = [_norm(f)], False, None
    else:
        Xhist, Fhist, head, filled, x, f, best, res_history, fallback = resume()
        res_history = list(res_history)
    resume = None

    def _best_in_window():
        return int(np.argmin(res_history)) >= len(res_history) - 1 - filled

    if best is None:  # in the window: the first minimum, as the loop keeps it
        i = int(np.argmin(res_history)) - (len(res_history) - 1 - filled)
        slot = (head - filled + i) % m
        best = (x, f) if i == filled else (Xhist[slot], Fhist[slot])
    best_res = min(res_history)
    window_res = res_history[-(m + 1):]

    def _state():
        return AndersonState(Xhist, Fhist, head, filled, x, f,
                             None if _best_in_window() else best,
                             tuple(res_history), fallback)

    if fresh:
        if res_history[0] <= tol:
            return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                                      iterations=0, converged=True)
        if on_eval is not None and maxit > 0:
            on_eval(_state())
    mask_shape = (m,) + (1,) * len(shape)

    for it in range(len(res_history) - 1, maxit):
        oldest_pos = (head - filled) % m
        X_ord = jnp.roll(Xhist, shift=-oldest_pos, axis=0)
        F_ord = jnp.roll(Fhist, shift=-oldest_pos, axis=0)
        mask_cols = (jnp.arange(m) < filled).reshape(mask_shape)
        X_ord = jnp.where(mask_cols, X_ord, 0.0 + 0.0j)
        F_ord = jnp.where(mask_cols, F_ord, 0.0 + 0.0j)
        Xw = jnp.concatenate([X_ord, x[None]], axis=0)
        Fw = jnp.concatenate([F_ord, f[None]], axis=0)
        alpha, n_used = _solve_alpha_filtered(_weighted(Fw))
        x_opt = jnp.tensordot(alpha, Xw, axes=(0, 0))
        f_opt = jnp.tensordot(alpha, Fw, axes=(0, 0))
        if fallback:
            a2, _ = _solve_alpha_filtered(_weighted(jnp.stack([best[1], f])))
            x_opt = a2[0] * best[0] + a2[1] * x
            f_opt = a2[0] * best[1] + a2[1] * f
        if print_fn is not None:
            print_fn(f"  Anderson step {it:02d}: window {n_used + 1}"
                     f"{', secant fallback to the best pair' if fallback else ''}")
        Xhist = Xhist.at[head].set(x)
        Fhist = Fhist.at[head].set(f)
        head = (head + 1) % m
        filled = min(filled + 1, m)
        Xw = Fw = X_ord = F_ord = None

        x = _entry(x_opt + (0.5 if it == 0 else 1.0) * f_opt)
        f = _entry(residual_fn(x))
        res = _norm(f)
        res_history.append(res)
        fallback = (not fallback) and res > max(window_res)
        window_res = (window_res + [res])[-(m + 1):]
        if res < best_res:
            best, best_res = (x, f), res
        if res <= tol:
            return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                                      iterations=it + 1, converged=True)
        if on_eval is not None and it + 1 < maxit:
            on_eval(_state())

    return AccelerationResult(x=x, residual_norms=jnp.array(res_history),
                              iterations=maxit, converged=False)

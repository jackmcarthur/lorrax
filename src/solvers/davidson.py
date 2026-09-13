"""Host convenience API for the single planned Davidson implementation.

Use ``plan_davidson`` directly when operator arrays must change without a new
trace, or when composing the whole solve inside another JIT. This interface
resolves a plan from the initial vector geometry and reports its final status.
"""
from __future__ import annotations

import inspect
import time

import jax
import jax.numpy as jnp
import numpy as np

from common.collectives import gather_to_host
from solvers.davidson_fixed import (
    BAD_INITIAL, BAD_TOLERANCE, NONFINITE, CONVERGED, STALLED, plan_davidson,
)

# The measured Si clustered-band default; a capacity is still explicit in the
# composable API. See docs/services/davidson.md for memory/conditioning policy.
DEFAULT_M_MAX_FACTOR = 10
TRACE_COUNTS: dict = {}
MATVEC_APPLICATIONS = [0]
LAST_RUN: dict = {}


def reset_instrumentation():
    TRACE_COUNTS.clear()
    MATVEC_APPLICATIONS[0] = 0
    LAST_RUN.clear()


def instrumentation_summary():
    return (f'[dav-instr] matvec applications (vectors): {MATVEC_APPLICATIONS[0]}\n'
            f'[dav-instr] solver traces: {sum(TRACE_COUNTS.values())}\n'
            '[dav-instr] history contains a final snapshot, not per-iteration host callbacks')


def _count_converged(conv, n_eig):
    return int(sum(bool(conv[i]) for i in range(n_eig)))


def _default_precond(r, eigenvalues):
    norms = jnp.sqrt(jnp.sum(jnp.abs(r)**2, axis=tuple(range(1, r.ndim))))
    return r/jnp.maximum(norms, 1e-30).reshape((r.shape[0],)+(1,)*(r.ndim-1))


def davidson(apply_H, *, n_eig, precond_fn=None, init_fn=None, X0=None,
             m_max=None, max_iter=100, tol=1e-8, stall_patience=20, verbose=True,
             data=None):
    """Lowest Hermitian eigenpairs, using fixed-capacity active-only iteration.

    ``X0`` has shape ``(n_eig, *vector_shape)``; vector axes retain their
    NamedSharding. Capacity defaults to ``10*n_eig`` and is at least twice
    the requested roots. A two- or three-argument preconditioner is accepted.
    The third argument is the current Ritz vectors. When ``data`` is given,
    both callbacks take that explicit pytree as their first argument; use this
    form for distributed arrays so none is captured as a compiled constant.

    Convergence requires every true residual below ``tol*max(1,abs(e))``.
    ``stall_patience`` stops after that many iterations without a 1% reduction
    of the largest residual (zero disables it). Stalling is not convergence.
    Eigenvalues return on the host; eigenvectors preserve distributed storage.
    ``LAST_RUN`` holds one final snapshot and an explicit termination status.
    """
    if X0 is None:
        if init_fn is None:
            raise ValueError('Provide either init_fn or X0')
        X0, _ = init_fn(apply_H if data is None else lambda v: apply_H(data, v), n_eig)
    initial = jnp.asarray(X0[:n_eig], dtype=jnp.complex128)
    capacity = max(int(m_max if m_max is not None else DEFAULT_M_MAX_FACTOR*n_eig), 2*n_eig)
    if precond_fn is None:
        precond_fn = _default_precond if data is None else lambda payload, r, e: _default_precond(r, e)
    # Inspect the callable without swallowing an internal TypeError during a
    # trace, which previously hid preconditioner defects as arity fallbacks.
    try:
        inspect.signature(precond_fn).bind(*((None,)* (3 if data is None else 4)))
        uses_x = True
    except TypeError:
        uses_x = False
    explicit_data = data is not None
    def precondition(payload, r, e, x):
        args = (payload, r, e) if explicit_data else (r, e)
        return precond_fn(*args, x) if uses_x else precond_fn(*args)
    def apply(payload, v):
        # Tally trace events here, not solver executions or shape visits.
        key = ('planned_H', v.shape)
        TRACE_COUNTS[key] = TRACE_COUNTS.get(key, 0)+1
        return apply_H(payload, v) if explicit_data else apply_H(v)
    sharding = initial.sharding if isinstance(initial.sharding, jax.sharding.NamedSharding) else None
    plan = plan_davidson(apply, precondition, n_eig=n_eig, capacity=capacity,
                         vector_shape=initial.shape[1:], vector_sharding=sharding)
    started = time.perf_counter()
    values, vectors, info = plan.solve(data if explicit_data else (), initial, tol, max_iter, stall_patience)
    values = gather_to_host(values)
    status, iterations, matvecs, restarts, active, residuals = jax.tree.map(gather_to_host, info)
    status, iterations, matvecs = int(status), int(iterations), int(matvecs)
    MATVEC_APPLICATIONS[0] = matvecs
    LAST_RUN.clear()
    LAST_RUN.update(status=status, history_kind='final_snapshot', iter=[iterations],
                    mv=[matvecs], m=[int(active)], eig=[values.copy()],
                    res=[np.asarray(residuals)], t=[time.perf_counter()-started],
                    restarts=int(restarts))
    if status == BAD_INITIAL:
        raise ValueError('davidson: initial subspace is rank deficient')
    if status == BAD_TOLERANCE:
        raise ValueError('davidson: tolerance must be finite and positive')
    if status == NONFINITE:
        raise FloatingPointError('davidson: nonfinite operator/preconditioner output')
    if verbose:
        count = _count_converged(residuals < tol*np.maximum(1., np.abs(values)), n_eig)
        label = 'Converged' if status == CONVERGED else 'STALLED' if status == STALLED else 'NOT CONVERGED'
        print(f'Davidson: {label}; {count}/{n_eig} roots, {iterations} iterations, '
              f'{matvecs} matvecs, {int(restarts)} restarts; max residual={np.max(residuals):.3e}', flush=True)
    return values, vectors

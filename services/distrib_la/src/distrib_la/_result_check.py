"""Checked dense results: no distributed library result leaves this service unchecked.

The distributed libraries fail silently: cuSOLVERMp's syevd returned wrong
eigenpairs with info 0 on a Hermitian matrix with a large exact-zero block
(Fe 4^3 bispinor CT metric, residual 6.8e-3; a synthetic one, residual 2.0e-2
at a min/max eigenvalue ratio of -2.7e-8, which no downstream Gram gate sees).
So every eigh takes its exact-zero rows out of the solver
(:func:`deflate_zero_rows`), and every distributed eigh and LU solve is checked
on its own output against fixed-seed random probes before it is returned
(:func:`checked_eigh`, :func:`checked_solve`), at O(n^2 k) beside the O(n^3)
solve. A failed check retries where the operands survive (another layout),
and otherwise refuses by name (GATE distrib_la_result_check: printed, the
result NaN-poisoned on every rank, raised by an eager call): a wrong result is
never returned. There is no dial.

Acceptance (:data:`ACCEPT`) is on backward errors, which a backward-stable
solver keeps near n * eps whatever the condition number:
  eigh   ||(A V - V diag(w)) X|| / (||A|| ||X||) and ||(V^H V - I) X|| / ||X||
  solve  ||W^H (A X - B)|| / (sqrt(k) (||A|| ||X|| + ||B||))
with X, W fixed-seed Gaussian probes (k = :data:`PROBES` columns, identical on
every rank) and Frobenius norms. 1e-8 is above the n * eps rounding of a
stable solve by 10^3 at n = 10^5 and below every failure measured by 10^5.
"""
from __future__ import annotations

import os
import sys
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

#: Accepted backward error of a checked result (module docstring).
ACCEPT = 1e-8
#: Probe columns per check.
PROBES = 8
#: The probe seed: every rank draws the same probes, so every rank reaches
#: the same verdict from the same replicated norms.
_SEED = 20261002


def deflate_zero_rows(eigh):
    """Wrap a Hermitian eigensolver so exact zero rows never reach it.

    Capacity padding and unselected columns reach eigensolvers as exact zero
    rows/columns, and a large zero block breaks them: the native solver
    returned nonfinite values (Na P16, 1460 of 2584 rows) and cuSOLVERMp wrong
    eigenpairs with info 0. Those rows are decoupled, so they are replaced by
    distinct diagonal sentinels below the Gershgorin bound of the rest, solved,
    and reported back as exact zero eigenvalues with their unit eigenvectors,
    in ascending order: the spectrum and vectors of the zero-padded matrix,
    without a zero block inside the solver. A matrix with no zero row passes
    through unchanged.
    """
    def solve(a):
        n = a.shape[-1]
        dead = jnp.all(a == 0, axis=-1)
        # Every live eigenvalue lies in [-bound, bound] (Gershgorin); the
        # sentinels sit at or below -(2 bound + 1), a relative gap of one bound.
        bound = jnp.max(jnp.sum(jnp.abs(a), axis=-1), axis=-1)
        sentinel = -(2 * bound + 1)[..., None] * (1 + jnp.arange(n, dtype=bound.dtype) / n)
        diagonal = jnp.where(dead, sentinel, 0).astype(a.dtype)
        values, vectors = eigh(a + diagonal[..., :, None] * jnp.eye(n, dtype=a.dtype))
        values = jnp.where(values < -(1.5 * bound[..., None] + 0.5), 0, values)
        order = jnp.argsort(values, axis=-1, stable=True)
        return (jnp.take_along_axis(values, order, axis=-1),
                jnp.take_along_axis(vectors, order[..., None, :], axis=-1))
    return solve


def shifted(eigh):
    """Solve A + s I, s = ||A||_F, and return the eigenpairs of A.

    The shift moves a large (near-)zero cluster away from the origin.
    cuSOLVERMp's silent failures on rank-deficient PSD responses (n 432, no
    zero row, every block size) and on the CT metric's zero block (n 2688,
    block 224) all pass shifted (probe residual <= 1e-15). The eigenvalues
    come back to within n * eps * (||A|| + s) of A's, the stable bound.
    """
    def solve(a):
        s = jnp.linalg.norm(a, axis=(-2, -1))
        values, vectors = eigh(a + s[..., None, None] * jnp.eye(a.shape[-1], dtype=a.dtype))
        return values - s[..., None], vectors
    return solve


#: The local (one-device) eigh of every native route: jnp.linalg.eigh, deflated.
native_eigh = deflate_zero_rows(jnp.linalg.eigh)


def probes(n, dtype, k=PROBES, salt=0):
    """Fixed-seed Gaussian [n, k] probe columns in ``dtype``, the same on every rank."""
    key = jax.random.fold_in(jax.random.key(_SEED), salt)
    real = jax.random.normal(key, (n, k), dtype=jnp.float64)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        imag = jax.random.normal(jax.random.fold_in(key, 1), (n, k), dtype=jnp.float64)
        return (real + 1j * imag).astype(dtype)
    return real.astype(dtype)


def _norm(a):
    return jnp.linalg.norm(a, axis=(-2, -1))


def eigh_errors(a, values, vectors):
    """Largest probe residual and orthogonality error of a (stack of) eigh result(s)."""
    x = probes(a.shape[-1], vectors.dtype)
    vx = vectors @ x
    residual = _norm(a @ vx - vectors @ (values[..., :, None] * x)) / (
        jnp.maximum(_norm(a), jnp.finfo(values.dtype).tiny) * _norm(x))
    orthogonality = _norm(jnp.conj(jnp.swapaxes(vectors, -1, -2)) @ vx - x) / _norm(x)
    finite = jnp.all(jnp.isfinite(values))
    return (jnp.where(finite, jnp.max(residual), jnp.inf),
            jnp.where(finite, jnp.max(orthogonality), jnp.inf))


def call_site():
    """The first frame outside distrib_la and JAX: the role the refusal names."""
    for frame in reversed(traceback.extract_stack()[:-1]):
        path = frame.filename.replace(os.sep, "/")
        if "/distrib_la/" in path or "/jax/" in path or "/jaxlib/" in path or path.startswith("<"):
            continue
        parts = path.split("/")
        return f"{'/'.join(parts[-2:])}:{frame.lineno} {frame.name}"
    return "unknown caller"


def _report_refusal(op, n, site, attempts, *errors):
    """Host side of a failed final check: the named refusal on rank 0's stderr.

    It does not raise: a host callback's exception is an unordered effect that
    each rank meets at a different point. The result is NaN-poisoned instead,
    identically on every rank, and a caller's finite-result gate refuses; an
    eager call refuses here by name (``refuse_if_poisoned``).
    """
    if jax.process_index() == 0:
        print(refusal_message(op, n, site, attempts, *errors), file=sys.stderr, flush=True)


def refusal_message(op, n, site, attempts, *errors):
    detail = ", ".join(f"{name} {float(value):.2e}" for name, value in
                       zip(("residual", "orthogonality"), errors))
    return (f"GATE distrib_la_result_check: {op} n={n} at {site}: "
            f"{detail or 'no attempt passed'} after {attempts} attempt(s) (accept {ACCEPT:.0e}); the "
            f"distributed solver returned a wrong result and no retry repaired it; result set to NaN")


def refuse_if_poisoned(result, op, n, site):
    """An eager call's refusal: raise by name when the checked result came back NaN-poisoned.

    The poison is all-NaN and identical on every rank (one replicated verdict),
    so every rank raises here together.
    """
    leaf = jax.tree.leaves(result)[0]
    if isinstance(leaf, jax.core.Tracer):
        return result
    local = np.asarray(leaf.addressable_data(0))
    if local.size and np.isnan(local).all():
        raise ValueError(refusal_message(op, n, site, "every"))
    return result


def _notice(op, n, site, what, *errors):
    if jax.process_index() == 0:
        detail = ", ".join(f"{name} {float(value):.2e}" for name, value in
                           zip(("residual", "orthogonality"), errors))
        print(f"distrib_la: {op} n={n} at {site} failed its result check ({detail}, "
              f"accept {ACCEPT:.0e}); {what}", file=sys.stderr, flush=True)


def _verdict(errors):
    return jnp.all(jnp.stack(errors) <= ACCEPT)


def checked(op, attempts, errors_of, operands, *, site, n=None):
    """Run ``attempts`` in order until one passes ``errors_of``; refuse if none does.

    Each attempt maps ``operands`` to a result; ``errors_of(result)`` returns
    replicated scalars. Later attempts run only when the earlier ones failed
    (``lax.cond``). A final failure prints GATE distrib_la_result_check on
    rank 0's stderr and poisons the result with NaN on every rank; an eager
    caller raises it (``refuse_if_poisoned``).
    """
    n = int(operands[0].shape[-1]) if n is None else int(n)

    def run(index, previous_errors):
        result = attempts[index](*operands)
        errors = errors_of(result)
        if index + 1 == len(attempts):
            def refuse(r):
                jax.debug.callback(partial(_report_refusal, op, n, site, len(attempts)), *errors)
                return jax.tree.map(lambda a: jnp.full_like(a, jnp.nan), r)
            return jax.lax.cond(_verdict(errors), lambda r: r, refuse, result)

        def again(_):
            jax.debug.callback(partial(_notice, op, n, site,
                                       f"attempt {index + 1} of {len(attempts)} failed; solving again"),
                               *errors)
            return run(index + 1, errors)
        return jax.lax.cond(_verdict(errors), lambda r: r, again, result)
    return run(0, None)


def checked_eigh(attempts, a, *, site):
    """A distributed eigh, checked: ``attempts`` are solvers of ``a`` in order of preference."""
    return checked("eigh", attempts, lambda r: eigh_errors(a, *r), (a,), site=site)


def matrix_sketch(a):
    """Before a factorization or solve that may consume ``a``: (A^H W, ||A||)."""
    w = probes(a.shape[-1], a.dtype, salt=1)
    return jnp.conj(jnp.swapaxes(a, -1, -2)) @ w, _norm(a)


def rhs_sketch(b):
    """Before a solve that may consume ``b``: (W^H B, ||B||)."""
    w = probes(b.shape[-2], b.dtype, salt=1)
    return jnp.conj(w.T) @ b, _norm(b)


def solve_errors(sketch, x):
    """||W^H (A X - B)|| / (sqrt(k) (||A|| ||X|| + ||B||)) from the pre-solve sketches, largest over the stack.

    ``sketch`` is ``matrix_sketch(A) + rhs_sketch(B)``; W^H A X = (A^H W)^H X.
    """
    ahw, norm_a, whb, norm_b = sketch
    projected = jnp.conj(jnp.swapaxes(ahw, -1, -2)) @ x - whb
    scale = np.sqrt(PROBES) * (norm_a * _norm(x) + norm_b)
    error = _norm(projected) / jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)
    return (jnp.where(jnp.all(jnp.isfinite(x)), jnp.max(error), jnp.inf),)

"""Planned block Davidson for one k-point on one CUDA device.

The complete iteration is one ``lax.while_loop`` with fixed-capacity storage.
Only active vectors enter projections, orthogonalization, reconstruction and
H applications. ``distrib_la`` owns the runtime-size native linear algebra;
its small host size synchronizations are part of this explicit local route.
The existing distributed/host Davidson interface remains separate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import NamedTuple

import jax
import jax.numpy as jnp

from distrib_la import plan_local_subspace
from solvers.davidson import _rank_whitener

RUNNING, CONVERGED, ITERATION_LIMIT, NO_DIRECTIONS, BAD_INITIAL, NONFINITE, BAD_TOLERANCE = range(7)


class DavidsonInfo(NamedTuple):
    status: jax.Array
    iterations: jax.Array
    matvecs: jax.Array
    restarts: jax.Array
    active_size: jax.Array
    residuals: jax.Array


class _State(NamedTuple):
    basis: jax.Array
    images: jax.Array
    projected: jax.Array
    size: jax.Array
    iterations: jax.Array
    status: jax.Array
    vectors: jax.Array
    values: jax.Array
    residuals: jax.Array
    matvecs: jax.Array
    restarts: jax.Array


def _gram(a, b):
    return jnp.einsum('i...,j...->ij', a.conj(), b)


def _take(a, start, size):
    return jax.lax.dynamic_slice_in_dim(a, start, size)


def _put(a, b, start):
    return jax.lax.dynamic_update_slice_in_dim(a, b, start, axis=0)


def _active_blocks(count, block, state, operation):
    """Full blocks followed by binary-sized tails; no padded H application."""
    count = jnp.asarray(count, jnp.int32)
    state = jax.lax.fori_loop(
        jnp.int32(0), count // block,
        lambda i, s: operation(i*block, block, s), state)
    offset = (count // block)*block
    for bit in reversed(range((block-1).bit_length())):
        width = 1 << bit
        use = count-offset >= width
        state = jax.lax.cond(use, lambda s: operation(offset, width, s), lambda s: s, state)
        offset += jnp.where(use, width, 0).astype(jnp.int32)
    return state


@dataclass(frozen=True)
class DavidsonPlan:
    """Static memory geometry and a composable jitted ``solve`` callable.

    ``solve(data, initial, tolerance, max_iterations)`` returns eigenvalues,
    row eigenvectors, and ``DavidsonInfo``. Operator/preconditioner data are
    explicit array pytrees, so a new same-shape Hamiltonian reuses compilation.
    Status CONVERGED means every true residual is below
    ``tolerance * max(1, abs(eigenvalue))``. Inspect status on every return.
    """
    n_eig: int
    capacity: int
    vector_shape: tuple[int, ...]
    subspace: object
    normalizer: object
    solve: object

    @property
    def workspace_specs(self):
        """Persistent arrays and native scratch known before lowering.

        Operator scratch, fusion temporaries, output buffers and backend
        handles are not estimated here. Compile with ``solve.lower(...).compile()``
        and inspect ``memory_analysis()`` for XLA's complete buffer schedule.
        """
        cap, b = self.capacity, self.n_eig
        return {
            'basis': jax.ShapeDtypeStruct((cap,)+self.vector_shape, jnp.complex128),
            'images': jax.ShapeDtypeStruct((cap,)+self.vector_shape, jnp.complex128),
            'projected': jax.ShapeDtypeStruct((cap, cap), jnp.complex128),
            'coefficients': jax.ShapeDtypeStruct((cap, b), jnp.complex128),
            **self.subspace.workspace_specs,
            **{'normalizer_'+k: v for k, v in self.normalizer.workspace_specs.items()},
        }


def _whiten(p, normalizer):
    coefficients, rank = _rank_whitener(_gram(p, p))
    rank = rank.astype(jnp.int32)
    def retained(p):
        result, _ = normalizer.reconstruct(
            p, p, coefficients, p.shape[0], p, columns=rank, compute_image=False)
        return normalizer.normalize(result, rank)
    return jax.lax.cond(rank > 0, retained, jnp.zeros_like, p), rank


def _make_step(apply_h, precondition, subspace, normalizer, block, capacity, data, tolerance, budget):
    """Only small correction blocks cross conditionals; capacity buffers alias."""
    def step(st):
        v, hv, h, m = st.basis, st.images, st.projected, st.size
        e, c = subspace.eigh(h, m)
        x, hx = subspace.reconstruct(v, hv, c, m, st.vectors)
        residual = hx-e.reshape((block,)+(1,)*(x.ndim-1))*x
        norms = jnp.sqrt(jnp.sum(jnp.abs(residual)**2, axis=tuple(range(1, x.ndim))))
        converged = jnp.all(norms < tolerance*jnp.maximum(1., jnp.abs(e)))
        status = jnp.where(converged, CONVERGED,
                           jnp.where(st.iterations+1 >= budget, ITERATION_LIMIT, RUNNING))
        finite = jnp.all(jnp.isfinite(e)) & jnp.all(jnp.isfinite(norms))
        status = jnp.where(finite, status, NONFINITE).astype(jnp.int32)
        restart = (status == RUNNING) & (m+block > capacity)

        def correction(_):
            p = precondition(data, residual, e, x)
            finite_correction = jnp.all(jnp.isfinite(p))
            p = jax.lax.cond(
                restart, lambda p: normalizer.orthogonalize(x, p, block),
                lambda p: subspace.orthogonalize(v, p, m), p)
            p, rank = _whiten(p, normalizer)
            hp = _active_blocks(rank, block, jnp.zeros_like(p),
                                lambda i, w, hp: _put(hp, apply_h(data, _take(p, i, w)), i))
            return p, hp, rank, finite_correction & jnp.all(jnp.isfinite(hp))

        p, hp, rank, finite_correction = jax.lax.cond(
            status == RUNNING, correction,
            lambda _: (jnp.zeros_like(x), jnp.zeros_like(x), jnp.int32(0), jnp.bool_(True)), None)
        # These operations alias the capacity buffers. Carrying them through
        # lax.cond generated full-capacity copies even in the identity branch.
        reset_count = jnp.where(restart, block, 0).astype(jnp.int32)
        base = jnp.where(restart, block, m).astype(jnp.int32)
        v, hv = subspace.store(v, hv, x, hx, 0, reset_count)
        h = subspace.project(v, hv, base, h, jnp.where(restart, 0, base), reset_count)
        v, hv = subspace.store(v, hv, p, hp, base, rank)
        h = subspace.project(v, hv, base+rank, h, base, rank)
        status = jnp.where((status == RUNNING) & (rank == 0), NO_DIRECTIONS, status)
        status = jnp.where(finite_correction, status, NONFINITE).astype(jnp.int32)
        return _State(v, hv, h, base+rank, st.iterations+1, status, x, e, norms,
                      st.matvecs+rank, st.restarts+restart.astype(jnp.int32))
    return step


def plan_local_davidson(apply_h, precondition, *, n_eig, capacity, vector_shape):
    """Plan a complex128 local solver before solver compilation.

    ``apply_h(data, vectors)`` preserves ``(rows, *vector_shape)`` and must
    accept full blocks and power-of-two tail widths. ``precondition(data,
    residuals, eigenvalues, vectors)`` returns a full correction block.
    Both must be JAX-traceable and free of inter-rank collectives: different
    k-points may stop after different numbers of iterations.

    ``capacity >= 2*n_eig`` is explicit; no iteration changes an array shape.
    There is no hidden default capacity or persistent compilation cache.
    This opt-in CUDA route does not replace distributed Davidson callers.
    """
    if not isinstance(n_eig, int) or not isinstance(capacity, int):
        raise TypeError('n_eig and capacity must be Python integers')
    vector_shape = tuple(vector_shape)
    if not vector_shape or any(not isinstance(n, int) or n < 1 for n in vector_shape):
        raise ValueError('vector_shape must contain positive Python integers')
    if not 1 <= n_eig <= prod(vector_shape) or capacity < 2*n_eig:
        raise ValueError('require 1 <= n_eig <= vector size and capacity >= 2*n_eig')
    subspace = plan_local_subspace(capacity=capacity, n_eig=n_eig)
    normalizer = plan_local_subspace(capacity=n_eig, n_eig=n_eig)
    block = n_eig

    def solve(data, initial, tolerance=1e-8, max_iterations=100):
        if initial.shape != (block,)+vector_shape or initial.dtype != jnp.complex128:
            raise ValueError('initial shape/dtype differs from the declared complex128 plan')
        if jnp.ndim(tolerance) != 0 or jnp.ndim(max_iterations) != 0:
            raise ValueError('tolerance and max_iterations must be scalars')
        if not jnp.issubdtype(jnp.asarray(max_iterations).dtype, jnp.integer):
            raise TypeError('max_iterations must be an integer')
        x, rank = _whiten(initial, normalizer)
        hx = apply_h(data, x)
        v = jnp.zeros((capacity,)+vector_shape, x.dtype)
        v, hv = subspace.store(v, jnp.zeros_like(v), x, hx, 0, block)
        h = subspace.project(v, hv, block, jnp.zeros((capacity, capacity), x.dtype), 0, block)
        status = jnp.where(rank != block, BAD_INITIAL,
                           jnp.where(max_iterations <= 0, ITERATION_LIMIT, RUNNING))
        status = jnp.where(jnp.isfinite(tolerance) & (tolerance > 0), status, BAD_TOLERANCE)
        status = jnp.where(jnp.all(jnp.isfinite(initial)) & jnp.all(jnp.isfinite(hx)), status, NONFINITE).astype(jnp.int32)
        state = _State(v, hv, h, jnp.int32(block), jnp.int32(0), status, x,
                       jnp.zeros(block), jnp.full(block, jnp.inf), jnp.int32(block), jnp.int32(0))
        result = jax.lax.while_loop(
            lambda st: (st.status == RUNNING) & (st.iterations < max_iterations),
            _make_step(apply_h, precondition, subspace, normalizer, block, capacity, data, tolerance, max_iterations), state)
        info = DavidsonInfo(result.status, result.iterations, result.matvecs,
                            result.restarts, result.size, result.residuals)
        return result.values, result.vectors, info

    solve = jax.jit(solve, in_shardings=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]))
    return DavidsonPlan(n_eig, capacity, vector_shape, subspace, normalizer, solve)

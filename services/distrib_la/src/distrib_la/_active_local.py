"""Exact interval products for the local planned GEMM backend.

The allocation shapes are fixed. A dynamic interval is decomposed into
power-of-two widths, each selecting a statically shaped dot through lax.switch.
Only disjoint, wholly active slices enter a dot. This avoids padding arithmetic
and works on CPU and GPU through their existing JAX dot lowering.
"""
import jax
import jax.numpy as jnp


def _interval_product(a, b, weights, lo, hi):
    """One common interval for rank-two or batched local operands."""
    k = a.shape[-1]
    zero = jnp.zeros((*a.shape[:-1], b.shape[-1]), a.dtype)

    def partial_product(_):
        def branch(width):
            def accumulate(state):
                start, result = state
                left = jax.lax.dynamic_slice_in_dim(a, start, width, axis=-1)
                right = jax.lax.dynamic_slice_in_dim(b, start, width, axis=-2)
                weight = jax.lax.dynamic_slice_in_dim(weights, start, width, axis=-1)
                return start + width, result + jnp.matmul(left * weight[..., None, :], right)
            return accumulate
        max_bit = min(k.bit_length() - 1, 8)
        branches = tuple(branch(1 << bit) for bit in range(max_bit + 1))

        def step(state):
            start, _ = state
            bit = jnp.minimum(jnp.int32(31) - jax.lax.clz(hi - start), max_bit)
            return jax.lax.switch(bit, branches, state)

        return jax.lax.while_loop(lambda state: state[0] < hi, step,
                                  (lo, zero))[1]

    return jax.lax.cond((lo == 0) & (hi == k),
                        lambda _: jnp.matmul(a * weights[..., None, :], b), partial_product, None)


def active_local_matmul(a, b, bounds, weights, c=None, *, alpha, beta):
    """Local tile contraction; no processor exchange or host callbacks.

    Bounds have shape (nq,2). Equal intervals retain a batched dot; mixed
    intervals scan parents into the original output shape. Invalid dynamic
    bounds produce an all-NaN output rather than silently clamping slices.
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    valid = jnp.all((lo >= 0) & (lo <= hi) & (hi <= a.shape[-1]))

    def contract(_):
        def per_parent(_):
            def step(_, operands):
                left, right, weight, limits = operands
                return None, _interval_product(left, right, weight, limits[0], limits[1])
            return jax.lax.scan(step, None, (a, b, weights, bounds), unroll=1)[1]
        same = jnp.all(bounds == bounds[0])
        result = jax.lax.cond(same,
            lambda _: _interval_product(a, b, weights, lo[0], hi[0]), per_parent, None)
        result = (alpha if a.dtype.kind == 'c' else alpha.real) * result
        if beta != 0:
            result = result + (beta if a.dtype.kind == 'c' else beta.real) * c
        return result

    return jax.lax.cond(valid, contract,
        lambda _: jnp.full((a.shape[0], a.shape[1], b.shape[-1]), jnp.nan, a.dtype), None)

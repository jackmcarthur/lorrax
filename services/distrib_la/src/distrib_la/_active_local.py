"""Exact interval products for the local planned GEMM backend and the XLA subspace route.

Two XLA routes compute ``a[q][:, lo:hi] diag(w[q]) b[q][lo:hi, :]`` per parent:

- ``"interval"`` (:func:`active_local_matmul`): a dynamic interval is decomposed
  into power-of-two widths, each selecting a statically shaped dot through
  ``lax.switch``; only disjoint, wholly active slices enter a dot, so no padding
  arithmetic is spent.  The price is device control flow: a parent ``scan``,
  a ``while_loop`` over the pieces and a ``cond`` per piece, every trip a
  host-driven step of the XLA GPU runtime.
- ``"masked"`` (:func:`masked_local_matmul`): ONE batched dot over the whole
  storage axis, inactive columns of ``a diag(w)`` and rows of ``b`` *selected*
  to zero first (``where``, never a multiply, so NaN/Inf in inactive storage
  never enters arithmetic).  No control flow; it spends the inactive columns'
  FLOPs and two operand-sized temporaries.  Equal to the interval route up to
  the dot's summation order (the added terms are exact zeros).

:data:`ACTIVE_LOCAL_ROUTE` picks one per leg.
"""
import jax
import jax.numpy as jnp


#: Active-range local GEMM implementation by ``lxkit`` leg key -- a backend
#: preference of this service (as ``subspace.SUBSPACE_ROUTE``), so it lives here
#: and not in ``lxkit.LEGS``.  ``"native"``: the ``lorrax_cublas_local_*``
#: handlers (``_active_local_cuda``); ``"interval"``: :func:`active_local_matmul`;
#: ``"masked"``: :func:`masked_local_matmul`.  A vendor with no row takes
#: ``"interval"``.  ROCM: ``"masked"`` -- on MI250X (ROCm 7.14.1, JAX 0.9.1)
#: every XLA ``while``/``cond`` trip costs ~36 us of host-driven launches and
#: copies (command buffers off or forced: no better), and the interval route's
#: ~64 x (3-6) trips per call made Sigma's tau windows 68 s vs 7.5 s on A100
#: (Fe 4x4x4, n_mu 312, P=4; GPU idle 77% of the trace).
ACTIVE_LOCAL_ROUTE = {"cpu": "interval", "CUDA": "native", "ROCM": "masked"}


def active_local_route(platform):
    """The active-range route for an FFI platform key (``resolve.mesh_platform``)."""
    return ACTIVE_LOCAL_ROUTE.get(platform, "interval")


def masked_local_matmul(a, b, bounds, weights, c=None, *, alpha, beta):
    """:func:`active_local_matmul`'s contract as one masked batched dot.

    ``a`` ``(nq,m,k)``, ``b`` ``(nq,k,n)``, ``weights`` ``(nq,k)``, int ``bounds``
    ``(nq,2)``.  Columns outside ``[lo, hi)`` are selected to zero in
    ``a diag(w)`` and in ``b`` before the dot; an empty interval gives zeros.
    Invalid dynamic bounds produce an all-NaN output, as the interval route.
    """
    k = a.shape[-1]
    lo, hi = bounds[:, 0], bounds[:, 1]
    valid = jnp.all((lo >= 0) & (lo <= hi) & (hi <= k))
    column = jnp.arange(k, dtype=bounds.dtype)
    live = (column >= lo[:, None]) & (column < hi[:, None])          # (nq, k)
    zero = jnp.zeros((), a.dtype)
    left = jnp.where(live[:, None, :], a * weights[:, None, :].astype(a.dtype), zero)
    right = jnp.where(live[:, :, None], b, zero)
    result = (alpha if a.dtype.kind == 'c' else alpha.real) * jnp.matmul(left, right)
    if beta != 0:
        result = result + (beta if a.dtype.kind == 'c' else beta.real) * c
    return jnp.where(valid, result, jnp.full((), jnp.nan, a.dtype))


def local_active_impl(platform):
    """The XLA implementation behind a non-native route (``"interval"`` | ``"masked"``)."""
    return masked_local_matmul if active_local_route(platform) == "masked" else active_local_matmul


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


def interval_rows(block, lo, hi, k, init):
    """Fold ``block(start, width, out)`` over the rows ``[lo, hi)`` of a k-row buffer.

    The interval is decomposed into power-of-two widths (at most 256), each a
    statically shaped branch of ``lax.switch``, so only wholly active rows are
    ever sliced: inactive rows (which may hold NaN) never enter arithmetic.
    The whole buffer (``lo == 0``, ``hi == k``) is one ``block(0, k, out)``.
    No host callback; ``lo``/``hi`` are traced scalars and must be valid.
    """
    max_bit = min(k.bit_length() - 1, 8)
    branches = tuple((lambda w: lambda s: (s[0] + w, block(s[0], w, s[1])))(1 << bit)
                     for bit in range(max_bit + 1))

    def step(state):
        bit = jnp.minimum(jnp.int32(31) - jax.lax.clz((hi - state[0]).astype(jnp.int32)), max_bit)
        return jax.lax.switch(bit, branches, state)

    def partial(out):
        return jax.lax.while_loop(lambda s: s[0] < hi, step, (jnp.asarray(lo, jnp.int32), out))[1]

    return jax.lax.cond((lo == 0) & (hi == k), lambda out: block(jnp.int32(0), k, out),
                        partial, init)

"""Planned, trace-safe N,N GEMM: resolve once, call inside a hot loop.

:func:`gemm_plan` fixes one shape, layout and pair of GEMM scalars eagerly
and returns a :class:`GemmPlan` whose call runs only static shape, dtype and
layout checks before its callable, so Green builds, ζ-fit projectors and Σ
projections can call it inside an outer ``jax.jit``/``lax.scan``.

* **Face layout** (``P(None,'x','y')`` on A, B and D): the product is
  :func:`distrib_la.panel_matmul`'s batched SUMMA on every platform.  Its
  local panel products run over each row's active interval through the
  local active-range GEMM (classic cuBLAS on CUDA, bounded XLA panels
  elsewhere).  No native context or communicator is created.
* **Axis layout** (:func:`local_gemm_plan`): each rank contracts its own tile
  with XLA's dot, optionally reducing over one mesh axis.
* **N,N only, one replicated leading batch** ``nq``: a spinor axis is
  flattened into m/k/n by the caller.

The staged ``batch_reshard`` route is refused (``backend='off'``): it holds
whole matrices per device, which a G/Σ-sized product must never do.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._shard_map import shard_map
from distrib_la.matmul import MATMUL_BACKEND_CHOICES, _mesh_shape, _zeros
from distrib_la.resolve import mesh_platform

__all__ = ["GemmPlan", "gemm_plan", "local_gemm_plan"]

_SUPPORTED_DTYPES = (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128))


def _as_extent(label: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"gemm_plan: {label} must be a positive integer, got {value!r}")
    try:
        out = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"gemm_plan: {label} must be a positive integer, "
            f"got {value!r}") from exc
    if out <= 0:
        raise ValueError(f"gemm_plan: {label} must be positive, got {out}")
    return int(out)


def _as_scalar(label: str, value) -> complex:
    try:
        return complex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"gemm_plan: {label} must be a real or complex scalar, "
            f"got {value!r}") from exc


def _validate_dtype(dtype) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        allowed = "|".join(str(x) for x in _SUPPORTED_DTYPES)
        raise TypeError(f"gemm_plan: dtype must be {allowed}; got {dtype}")


def _same_layout(have, want: NamedSharding) -> bool:
    return (getattr(have, "spec", None) == want.spec
            and getattr(have, "mesh", None) == want.mesh)


def _prepare_host_bounds(plan: "GemmPlan", lo, hi):
    """Validate eager bounds; return them expanded to int32 ``(nq, 2)`` and the
    immutable int64 attribute form the local CUDA prepared kernel takes."""
    try:
        indices = tuple(np.asarray(value) for value in (lo, hi))
    except Exception as exc:  # JAX tracers and non-array-like values refuse here.
        raise TypeError(
            "gemm_plan.prepare_active_range: bounds must be eager host integers") from exc
    if any(value.shape not in ((), (plan.nq,)) or
           not np.issubdtype(value.dtype, np.integer) for value in indices):
        raise TypeError(
            "gemm_plan.prepare_active_range: bounds must be integer scalars or shape(nq,)")
    try:
        raw = np.stack(np.broadcast_arrays(*indices), axis=-1).reshape(-1, 2)
    except ValueError as exc:
        raise TypeError(
            "gemm_plan.prepare_active_range: scalar and shape(nq,) bounds must broadcast") from exc
    if not np.all((raw[:, 0] >= 0) & (raw[:, 0] <= raw[:, 1]) &
                  (raw[:, 1] <= plan.k)):
        raise ValueError(
            "gemm_plan.prepare_active_range: require 0 <= lo <= hi <= K")
    if np.any(raw > np.iinfo(np.int32).max):
        raise ValueError(
            "gemm_plan.prepare_active_range: bounds must fit signed int32")
    expanded = np.broadcast_to(raw, (plan.nq, 2)).astype(np.int32, copy=True)
    active_bounds = np.asarray(raw, dtype=np.int64).reshape(-1).copy()
    expanded.setflags(write=False)
    active_bounds.setflags(write=False)
    return expanded, active_bounds


def _check_operand(plan: "GemmPlan", label: str, x, shape: tuple[int, int, int],
                   sharding: NamedSharding) -> None:
    xshape = getattr(x, "shape", None)
    if xshape is None or tuple(int(s) for s in xshape) != shape:
        raise ValueError(
            f"gemm_plan {label}: expected shape {shape}; got "
            f"{None if xshape is None else tuple(xshape)}")
    if jnp.dtype(x.dtype) != plan.dtype:
        raise TypeError(
            f"gemm_plan {label}: expected dtype {plan.dtype}; got {x.dtype}")
    # A tracer's layout is fixed by the outer jit boundary and reinforced
    # by this plan's own in_specs; a concrete operand must already obey
    # the contract — silently device_put'ing an (nq,m,k)-class array is
    # exactly the hidden reshard distrib_la.polar refuses for the same
    # reason (polar.py:125-134).
    if not isinstance(x, jax.core.Tracer):
        have = getattr(x, "sharding", None)
        if not _same_layout(have, sharding):
            raise ValueError(
                f"gemm_plan {label}: must already be sharded "
                f"{sharding.spec} on the plan's mesh; refusing an "
                f"implicit reshard of a {shape} array.  Got {have!r}.")


@dataclass(frozen=True)
class GemmPlan:
    """A resolved, trace-safe ``D[q] = alpha*A[q]@B[q] (+ beta*C[q])``.

    Construct with :func:`gemm_plan` or :func:`local_gemm_plan`.  ``backend``
    is ``"panel_matmul"`` (the face layout: :func:`distrib_la.panel_matmul`'s
    batched SUMMA) or ``"local"`` (the axis layout).  Calling the plan runs
    static shape/dtype/layout checks, then the plan's callable.
    """

    mesh: Mesh
    backend: str
    m: int
    k: int
    n: int
    nq: int
    dtype: object
    alpha: complex
    beta: complex
    in_sharding_a: NamedSharding
    in_sharding_b: NamedSharding
    out_sharding: NamedSharding
    #: No native context backs either backend (0); kept for the workspace queries.
    ctx_handle: int
    _fn_with_c: Callable = field(compare=False, hash=False)
    _fn_no_c: Callable | None = field(compare=False, hash=False)
    reduction_axis: str | None = None
    _active_fn_with_c: Callable | None = field(default=None, compare=False, hash=False)
    _active_fn_no_c: Callable | None = field(default=None, compare=False, hash=False)

    def describe(self) -> str:
        """One line for a run banner: what resolved, and to what shape."""
        px, py = _mesh_shape(self.mesh)
        backend = self.backend
        return (f"gemm_plan: {backend} N,N on {px}x{py}, "
                f"shape (nq={self.nq}, m={self.m}, k={self.k}, n={self.n}), "
                f"dtype={self.dtype}, alpha={self.alpha}, beta={self.beta}")

    def __call__(self, A, B, C=None, *, out=None):
        """Return ``D``.  Trace-safe: usable inside nested ``jit``/``scan``.

        ``C`` (accumulate; required when this plan's ``beta != 0``) and
        ``out`` (a live buffer DONATED purely for its storage when
        ``beta == 0`` — its content is ignored) are mutually exclusive.
        The local backend leaves `out` live and returns a fresh result.
        With neither, and ``beta == 0``, the zero addend is built inside
        the same compiled call — see the module docstring.

        ``out=`` is refused on a ``beta != 0`` plan.  Both ``C`` and
        ``out`` reach the identical compiled ``_fn_with_c`` — the PLAN's
        own ``beta`` (fixed at construction, not chosen per call) decides
        whether that buffer's content is mathematically live.  On a
        ``beta != 0`` plan, ``out=``'s "content is ignored, pure storage"
        contract would silently be false: whatever the buffer happened to
        hold gets scaled by ``beta`` and added into the result.  Refuse
        by name rather than let a caller who read only the general
        ``out=`` framing above get a silently wrong accumulate from stale
        buffer content; use ``C=`` on such a plan, where the accumulate
        semantics are explicit at the call site.
        """
        if C is not None and out is not None:
            raise ValueError("gemm_plan: pass C or out, not both")
        if out is not None and self.beta != 0:
            raise ValueError(
                "gemm_plan: out= is only a content-ignored donation on a "
                f"beta==0 plan (this plan's beta={self.beta}).  On a "
                "beta!=0 plan out='s buffer content would be scaled by "
                "beta and added into the result -- pass C= instead, "
                "where that accumulate is explicit at the call site.")
        _check_operand(self, "A", A, (self.nq, self.m, self.k),
                      self.in_sharding_a)
        _check_operand(self, "B", B, (self.nq, self.k, self.n),
                      self.in_sharding_b)
        c_or_out = C if C is not None else out
        if c_or_out is None:
            if self.beta != 0:
                raise ValueError(
                    "gemm_plan: C is required when beta != 0 "
                    f"(this plan's beta={self.beta})")
            if self._fn_no_c is None:
                raise AssertionError(
                    "gemm_plan: internal-zero kernel was not warmed for a "
                    "beta==0 plan; this is a construction bug, not a "
                    "caller error")
            return self._fn_no_c(A, B)
        _check_operand(self, "C/out", c_or_out, (self.nq, self.m, self.n),
                      self.out_sharding)
        return self._fn_with_c(A, B, c_or_out)

    def active_range(self, A, B, lo, hi, C=None, *, out=None, weights=None):
        """Contract exact intervals, optionally scaling columns by weights[nq,K].

        Construct with ``enable_active_range=True``. Full allocation shapes
        and output placement stay fixed; bounds are replicated integer scalars
        or arrays of shape (nq,). Each local product (the face backend's panel
        products, the local backend's tile) runs over the interval only: CUDA
        cuBLAS pointer views, or bounded JAX panels elsewhere. Neither route
        multiplies inactive tails.

        Require 0 <= lo <= hi <= K. Invalid Python integer bounds raise eagerly;
        invalid traced bounds raise in the native provider or produce an
        all-NaN result on the callback-free local backend. C/out have the same
        beta and donation semantics as ``__call__``.
        """
        if self._active_fn_with_c is None:
            raise ValueError("gemm_plan: active_range requires enable_active_range=True")
        if C is not None and out is not None:
            raise ValueError("gemm_plan.active_range: pass C or out, not both")
        if out is not None and self.beta != 0:
            raise ValueError("gemm_plan.active_range: out= requires beta==0")
        _check_operand(self, "A", A, (self.nq, self.m, self.k), self.in_sharding_a)
        _check_operand(self, "B", B, (self.nq, self.k, self.n), self.in_sharding_b)
        if isinstance(lo, int) and isinstance(hi, int) and not 0 <= lo <= hi <= self.k:
            raise ValueError("gemm_plan.active_range: require 0 <= lo <= hi <= K")
        indices = tuple(jnp.asarray(v) for v in (lo, hi))
        if any(v.shape not in ((), (self.nq,)) or not jnp.issubdtype(v.dtype, jnp.integer)
               for v in indices):
            raise TypeError("gemm_plan.active_range: bounds must be integer scalars or shape(nq,)")
        raw_bounds = jnp.stack(jnp.broadcast_arrays(*indices), axis=-1).reshape(-1, 2)
        # Preserve invalid wide integers as a refusal, rather than wrapping
        # (e.g. 2**32 to zero) while packing the native int32 operands.
        valid = jnp.all((raw_bounds >= 0) & (raw_bounds <= self.k))
        bounds = jnp.broadcast_to(jnp.where(valid, raw_bounds, -1).astype(jnp.int32),
                                  (self.nq, 2))
        if weights is not None:
            weights = jnp.asarray(weights)
            if self.dtype.kind != "c" and jnp.issubdtype(weights.dtype, jnp.complexfloating):
                raise TypeError("gemm_plan.active_range: complex weights require a complex plan")
            weights = weights.astype(self.dtype)
            if weights.shape != (self.nq, self.k):
                raise ValueError("gemm_plan.active_range: weights must have shape(nq,K)")
        if weights is None:
            weights = jnp.ones((self.nq, self.k), dtype=self.dtype)
        active_args = (A, B, bounds, weights)
        c = C if C is not None else out
        if c is None:
            if self._active_fn_no_c is None:
                raise ValueError("gemm_plan.active_range: C is required when beta != 0")
            return self._active_fn_no_c(*active_args)
        _check_operand(self, "C/out", c, (self.nq, self.m, self.n), self.out_sharding)
        return self._active_fn_with_c(*active_args, c)

    def prepare_active_range(self, lo, hi):
        """Capture eager bounds and return a trace-safe interval callable.

        The returned function has signature
        ``fn(A, B, C=None, *, out=None, weights=None)``. A local CUDA plan
        encodes the immutable bounds as FFI attributes, so repeated calls
        perform no device-to-host bounds transfer; the face backend and the
        CPU local plan close over the same constant bounds.  Prepared kernels
        compile on first use and allocate no warmup operands.
        """
        if self._active_fn_with_c is None:
            raise ValueError(
                "gemm_plan: prepare_active_range requires enable_active_range=True")
        expanded, active_bounds = _prepare_host_bounds(self, lo, hi)
        constant_bounds = jnp.asarray(expanded, dtype=jnp.int32)
        a_spec, b_spec, out_spec = (self.in_sharding_a.spec,
                                    self.in_sharding_b.spec,
                                    self.out_sharding.spec)
        if self.backend == "local" and mesh_platform(self.mesh) == "CUDA":
            from distrib_la._active_local_cuda import (
                prepared_active_local_cuda,
                require_prepared_active_local_cuda,
            )
            require_prepared_active_local_cuda()
            local_impl = partial(prepared_active_local_cuda, active_bounds=active_bounds,
                                 alpha=self.alpha, beta=self.beta)
            prepared_no_c = None
            if self.beta == 0:
                prepared_no_c = jax.jit(shard_map(
                    lambda a, b, weight: local_impl(a, b, weights=weight),
                    mesh=self.mesh, in_specs=(a_spec, b_spec, P()),
                    out_specs=out_spec, check_vma=False))
                prepared_with_c = lambda a, b, weight, _c: prepared_no_c(a, b, weight)
            else:
                prepared_with_c = jax.jit(shard_map(
                    lambda a, b, weight, c: local_impl(a, b, weights=weight, c=c),
                    mesh=self.mesh, in_specs=(a_spec, b_spec, P(), out_spec),
                    out_specs=out_spec, check_vma=False), donate_argnums=(3,))
        else:
            active_no_c, active_with_c = self._active_fn_no_c, self._active_fn_with_c
            prepared_no_c = (None if active_no_c is None else
                             lambda a, b, weight: active_no_c(a, b, constant_bounds, weight))
            prepared_with_c = lambda a, b, weight, c: active_with_c(a, b, constant_bounds, weight, c)

        def prepared(A, B, C=None, *, out=None, weights=None):
            if C is not None and out is not None:
                raise ValueError(
                    "gemm_plan prepared active_range: pass C or out, not both")
            if out is not None and self.beta != 0:
                raise ValueError(
                    "gemm_plan prepared active_range: out= requires beta==0")
            _check_operand(self, "A", A, (self.nq, self.m, self.k),
                           self.in_sharding_a)
            _check_operand(self, "B", B, (self.nq, self.k, self.n),
                           self.in_sharding_b)
            if weights is not None:
                weights = jnp.asarray(weights)
                if (self.dtype.kind != "c" and
                        jnp.issubdtype(weights.dtype, jnp.complexfloating)):
                    raise TypeError(
                        "gemm_plan prepared active_range: complex weights require a complex plan")
                weights = weights.astype(self.dtype)
                if weights.shape != (self.nq, self.k):
                    raise ValueError(
                        "gemm_plan prepared active_range: weights must have shape(nq,K)")
            else:
                weights = jnp.ones((self.nq, self.k), dtype=self.dtype)
            c = C if C is not None else out
            if c is None:
                if prepared_no_c is None:
                    raise ValueError(
                        "gemm_plan prepared active_range: C is required when beta != 0")
                return prepared_no_c(A, B, weights)
            _check_operand(self, "C/out", c, (self.nq, self.m, self.n),
                           self.out_sharding)
            return prepared_with_c(A, B, weights, c)

        return prepared


def _axis_matmul(a, b, c=None, *, alpha, beta, reduction_axis=None):
    """Contract local bands or centroid tiles with the requested centroid reduction."""
    scale = alpha if a.dtype.kind == "c" else alpha.real
    result = scale * jnp.matmul(a, b)
    if reduction_axis is not None:
        result = jax.lax.psum_scatter(result, reduction_axis,
            scatter_dimension=1 if reduction_axis == "x" else 2, tiled=True)
    if beta != 0:
        scale = beta if a.dtype.kind == "c" else beta.real
        result = result + scale * c
    return result


def local_gemm_plan(mesh: Mesh, *, m: int, k: int, n: int, nq: int,
                    dtype, alpha=1.0, beta=0.0, reduction_axis=None, out_spec=None,
                    enable_active_range=False, warmup: bool = True) -> GemmPlan:
    """Warm a local product on CPU/GPU while retaining output axis shards.

    ``enable_active_range`` supports replicated K and a two-axis output, using
    the same ``GemmPlan.active_range`` API as the distributed face backend.
    Reduction-axis and single-axis-output plans retain dense behavior only.
    """
    return _local_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype, alpha=alpha,
                       beta=beta, reduction_axis=reduction_axis,
                       out_spec=out_spec,
                       enable_active_range=enable_active_range,
                       warmup=warmup)


def _local_plan(mesh: Mesh, *, m, k, n, nq, dtype, alpha, beta,
                reduction_axis, out_spec, enable_active_range, warmup) -> GemmPlan:
    """:func:`local_gemm_plan`."""
    m, k, n, nq = (_as_extent(label, value) for label, value in
                   (("m", m), ("k", k), ("n", n), ("nq", nq)))
    dtype = jnp.dtype(dtype)
    _validate_dtype(dtype)
    alpha, beta = _as_scalar("alpha", alpha), _as_scalar("beta", beta)
    if dtype.kind != "c" and (alpha.imag or beta.imag):
        raise ValueError("local_gemm_plan: alpha/beta must be real for real dtype")
    px, py = _mesh_shape(mesh)
    out_spec = P(None, "x", "y") if out_spec is None else out_spec
    if out_spec not in (P(None, "x", "y"), P(None, "x", None), P(None, None, "y")):
        raise ValueError("local_gemm_plan: output must retain its centroid axis shards")
    for extent, axis in zip((m, n), out_spec[1:]):
        if axis is not None and extent % mesh.shape[axis]:
            raise ValueError("local_gemm_plan: output extent does not tile its mesh axis")
    a_spec, b_spec = P(None, out_spec[1], None), P(None, None, out_spec[2])
    if reduction_axis is not None and out_spec != P(None, "x", "y"):
        raise ValueError("local_gemm_plan: centroid reduction requires the two-axis output")
    if reduction_axis == "y":
        a_spec, b_spec = P(None, "x", "y"), P(None, "y", None)
    elif reduction_axis == "x":
        a_spec, b_spec = P(None, None, "x"), P(None, "x", "y")
    elif reduction_axis is not None:
        raise ValueError("local_gemm_plan: reduction_axis must be x, y or None")
    if reduction_axis is not None and k % mesh.shape[reduction_axis]:
        raise ValueError("local_gemm_plan: contraction extent must tile its reduction axis")
    if enable_active_range and (reduction_axis is not None or out_spec != P(None, "x", "y")):
        raise NotImplementedError("local_gemm_plan active_range requires replicated K and two-axis output")
    if enable_active_range and k > 2**31 - 1:
        raise ValueError("local_gemm_plan active_range storage K exceeds int32 bounds")
    active_impl = None
    if enable_active_range:
        if mesh_platform(mesh) == "CUDA":
            from distrib_la._active_local_cuda import (active_local_cuda,
                                                       require_active_local_cuda)
            require_active_local_cuda()
            active_impl = active_local_cuda
        else:
            from distrib_la._active_local import active_local_matmul
            active_impl = active_local_matmul
    a_sh, b_sh, out_sh = (NamedSharding(mesh, spec)
                           for spec in (a_spec, b_spec, out_spec))
    local = partial(_axis_matmul, alpha=alpha, beta=beta, reduction_axis=reduction_axis)
    with_c = jax.jit(shard_map(local, mesh=mesh,
        in_specs=(a_spec, b_spec, out_spec), out_specs=out_spec,
        check_vma=False), donate_argnums=(2,) if beta != 0 else ())
    if warmup:
        with_c(_zeros((nq, m, k), dtype, a_sh),
               _zeros((nq, k, n), dtype, b_sh),
               _zeros((nq, m, n), dtype, out_sh))
    no_c = None
    if beta == 0:
        no_c = jax.jit(shard_map(local, mesh=mesh,
            in_specs=(a_spec, b_spec), out_specs=out_spec, check_vma=False))
        if warmup:
            no_c(_zeros((nq, m, k), dtype, a_sh), _zeros((nq, k, n), dtype, b_sh))
    plan = GemmPlan(mesh=mesh, backend="local", m=m, k=k, n=n, nq=nq,
                    dtype=dtype, alpha=alpha, beta=beta,
                    in_sharding_a=a_sh, in_sharding_b=b_sh, out_sharding=out_sh,
                    ctx_handle=0, _fn_with_c=with_c, _fn_no_c=no_c, reduction_axis=reduction_axis)
    if not enable_active_range:
        return plan
    from dataclasses import replace
    active = partial(active_impl, alpha=alpha, beta=beta)
    # Exercise a genuine partial interval before returning the plan. Full
    # ranges deliberately bypass the active CUDA target and would not warm it.
    warm_hi = k - 1 if k > 1 else k
    if beta == 0:
        active_no_c = jax.jit(shard_map(active, mesh=mesh,
            in_specs=(a_spec, b_spec, P(), P()), out_specs=out_spec, check_vma=False))
        # Local out= leaves its storage live, just as the dense local plan does.
        def active_with_c(a, b, limits, weight, c):
            return active_no_c(a, b, limits, weight)
    else:
        active_no_c = None
        active_with_c = jax.jit(shard_map(active, mesh=mesh,
            in_specs=(a_spec, b_spec, P(), P(), out_spec), out_specs=out_spec,
            check_vma=False), donate_argnums=(4,))
    if warmup:
        a0, b0 = _zeros((nq, m, k), dtype, a_sh), _zeros((nq, k, n), dtype, b_sh)
        bounds = _zeros((nq, 2), jnp.int32, NamedSharding(mesh, P())).at[:, 1].set(warm_hi)
        weights = _zeros((nq, k), dtype, NamedSharding(mesh, P())) + 1
        if active_no_c is not None:
            jax.block_until_ready(active_no_c(a0, b0, bounds, weights))
        else:
            jax.block_until_ready(active_with_c(a0, b0, bounds, weights,
                _zeros((nq, m, n), dtype, out_sh)))
    return replace(plan, _active_fn_with_c=active_with_c,
                   _active_fn_no_c=active_no_c)


def _panel_plan(mesh: Mesh, *, m, k, n, nq, dtype, alpha, beta) -> GemmPlan:
    """The face plan: every product runs :func:`distrib_la.panel_matmul`.

    No native context, communicator or warmup: panel_matmul's batched SUMMA
    all-gathers bounded band panels and multiplies them into each rank's own
    output tile.  The panel budget is one output tile per rank, the bound a
    Green build reserves (``gw.greens_function_kernel.green_panel_bytes``)."""
    from distrib_la._panel_matmul import panel_matmul
    px, py = _mesh_shape(mesh)
    tile = int(dtype.itemsize) * nq * (m // px) * (n // py)
    real = dtype.kind != "c"
    a_scale = alpha.real if real else alpha
    b_scale = beta.real if real else beta

    def finish(d, c):
        if alpha != 1:
            d = jnp.asarray(a_scale, dtype) * d
        return d if c is None else d + jnp.asarray(b_scale, dtype) * c

    def product(a, b, bounds=None, weights=None):
        return panel_matmul(a, b, mesh=mesh, panel_bytes=tile, bounds=bounds, weights=weights)

    face = NamedSharding(mesh, P(None, "x", "y"))
    with_c = jax.jit(lambda a, b, c: finish(product(a, b), c))
    no_c = jax.jit(lambda a, b: finish(product(a, b), None)) if beta == 0 else None
    active_with_c = jax.jit(lambda a, b, bounds, w, c: finish(product(a, b, bounds, w), c))
    active_no_c = (jax.jit(lambda a, b, bounds, w: finish(product(a, b, bounds, w), None))
                   if beta == 0 else None)
    return GemmPlan(mesh=mesh, backend="panel_matmul", m=m, k=k, n=n, nq=nq, dtype=dtype,
                    alpha=alpha, beta=beta, in_sharding_a=face, in_sharding_b=face,
                    out_sharding=face, ctx_handle=0, _fn_with_c=with_c, _fn_no_c=no_c,
                    _active_fn_with_c=active_with_c, _active_fn_no_c=active_no_c)


def gemm_plan(
    mesh: Mesh,
    *,
    m: int,
    k: int,
    n: int,
    nq: int,
    dtype,
    backend: str = "auto",
    alpha=1.0,
    beta=0.0,
    layout="face",
    reduction_axis=None,
    out_spec=None,
    enable_active_range: bool = False,
    warmup: bool = True,
) -> GemmPlan:
    """Resolve one trace-safe N,N GEMM shape.

    ``D[q] = alpha * A[q] @ B[q] (+ beta * C[q])`` per q in ``range(nq)``;
    ``A`` ``(nq,m,k)``, ``B`` ``(nq,k,n)``, ``D``/``C`` ``(nq,m,n)``.

    ``layout="face"``: every operand at ``P(None,'x','y')``; ``m`` and ``n``
    tile the mesh and ``k`` tiles both axes.  The plan runs
    :func:`distrib_la.panel_matmul` on every platform (:func:`_panel_plan`).
    ``layout="axis"``: :func:`local_gemm_plan`.  ``backend`` is checked
    against ``distrib_la.MATMUL_BACKEND_CHOICES``; every name runs the face
    plan except ``'off'``, which refuses because
    the staged route materializes whole matrices per device.  ``warmup``
    reaches only the axis plans: the face plan compiles at its first call.
    ``alpha``/``beta`` are fixed per plan; ``beta=0`` also gives a call
    without ``C``.
    """
    if layout == "axis":
        return local_gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                               alpha=alpha, beta=beta, reduction_axis=reduction_axis, out_spec=out_spec,
                               enable_active_range=enable_active_range, warmup=warmup)
    if layout != "face":
        raise ValueError(f"gemm_plan: unknown psi layout {layout!r}")
    if out_spec is not None and out_spec != P(None, "x", "y"):
        raise ValueError("gemm_plan: single-axis output requires layout=axis")
    px, py = _mesh_shape(mesh)
    m, k, n, nq = (_as_extent(label, value) for label, value in
                   (("m", m), ("k", k), ("n", n), ("nq", nq)))
    dtype = jnp.dtype(dtype)
    _validate_dtype(dtype)
    alpha_c = _as_scalar("alpha", alpha)
    beta_c = _as_scalar("beta", beta)
    if dtype.kind != "c" and (alpha_c.imag or beta_c.imag):
        raise ValueError(
            f"gemm_plan: alpha/beta must be real for real dtype {dtype}")
    requested = str(backend).strip().lower()
    if requested == "off":
        raise ValueError(
            "gemm_plan: backend='off' has no provider, and this planned "
            "surface never selects batch_reshard — it materializes "
            "complete A/B/C/D on every device.  Use "
            "distrib_la.matmul(..., batched_route='batch_reshard') "
            "directly for that route.")
    if requested not in MATMUL_BACKEND_CHOICES + ("local", "panel_matmul"):
        raise ValueError(
            f"gemm_plan: backend must be one of "
            f"{'|'.join(MATMUL_BACKEND_CHOICES)}; got {backend!r}")
    if reduction_axis is not None:
        raise ValueError("gemm_plan: a centroid reduction requires layout=axis")
    for label, extent, divisor in (("m", m, px), ("k", k, px),
                                   ("k", k, py), ("n", n, py)):
        if extent % divisor:
            raise ValueError(
                f"gemm_plan: {label}={extent} does not tile the "
                f"{px}x{py} mesh (needs divisor {divisor})")
    if enable_active_range and k > 2**31 - 1:
        raise ValueError("gemm_plan active_range storage K exceeds int32 bounds")
    return _panel_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype, alpha=alpha_c, beta=beta_c)

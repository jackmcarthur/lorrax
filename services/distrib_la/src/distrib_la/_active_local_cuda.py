"""Exact active-range local GEMM through classic cuBLAS.

This module is imported only for the CUDA axis-layout plan.  Each invocation
operates on one process-local tile and issues no communication.  Runtime
bounds remain device operands; the FFI reads only their small metadata array
to the host and views the original row-major buffers without packing slices.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from distrib_la.loader import probe_target


_TARGET = "lorrax_cublas_local_active_range_gemm"
_PREPARED_TARGET = "lorrax_cublas_local_prepared_active_range_gemm"
_WORKSPACE_BYTES = 4 * 1024 * 1024


def require_active_local_cuda() -> None:
    """Load and capability-probe the CUDA handler before tracing a plan."""
    if jax.default_backend() != "gpu":
        raise ValueError("local active cuBLAS GEMM requires a CUDA backend")
    usable, reason = probe_target(_TARGET, "CUDA")
    if not usable:
        raise RuntimeError(
            "local active cuBLAS GEMM is unavailable: " + reason)


def require_prepared_active_local_cuda() -> None:
    """Load and capability-probe the prepared CUDA handler before tracing."""
    if jax.default_backend() != "gpu":
        raise ValueError("local prepared active cuBLAS GEMM requires a CUDA backend")
    usable, reason = probe_target(_PREPARED_TARGET, "CUDA")
    if not usable:
        raise RuntimeError(
            "local prepared active cuBLAS GEMM is unavailable: " + reason)


def _dense(a, b, weighted_a, c, *, alpha, beta):
    scale = alpha if a.dtype.kind == "c" else alpha.real
    result = scale * jnp.matmul(weighted_a, b)
    if beta != 0:
        beta_scale = beta if a.dtype.kind == "c" else beta.real
        result = result + beta_scale * c
    return result


def _native(a, b, bounds, c, *, alpha, beta):
    output = jax.ShapeDtypeStruct((a.shape[0], a.shape[1], b.shape[2]), a.dtype)
    workspace = jax.ShapeDtypeStruct((_WORKSPACE_BYTES,), jnp.uint8)
    call = jax.ffi.ffi_call(
        _TARGET,
        (output, workspace),
        # jax.ffi spells layouts major-to-minor. These are ordinary C-order
        # buffers; the handler performs the row/column-major reinterpretation.
        input_layouts=[(0, 1, 2), (0, 1, 2), (0, 1), (0, 1, 2)],
        output_layouts=[(0, 1, 2), (0,)],
        input_output_aliases={3: 0},
        vmap_method="sequential",
    )
    result, _ = call(
        a,
        b,
        bounds,
        c,
        nq=a.shape[0],
        m=a.shape[1],
        k=a.shape[2],
        n=b.shape[2],
        alpha_re=float(alpha.real),
        alpha_im=float(alpha.imag),
        beta_re=float(beta.real),
        beta_im=float(beta.imag),
    )
    return result


def _prepared_native(a, b, c, *, active_bounds, alpha, beta):
    output = jax.ShapeDtypeStruct((a.shape[0], a.shape[1], b.shape[2]), a.dtype)
    workspace = jax.ShapeDtypeStruct((_WORKSPACE_BYTES,), jnp.uint8)
    call = jax.ffi.ffi_call(
        _PREPARED_TARGET,
        (output, workspace),
        input_layouts=[(0, 1, 2), (0, 1, 2), (0, 1, 2)],
        output_layouts=[(0, 1, 2), (0,)],
        input_output_aliases={2: 0},
        vmap_method="sequential",
    )
    result, _ = call(
        a,
        b,
        c,
        nq=a.shape[0],
        m=a.shape[1],
        k=a.shape[2],
        n=b.shape[2],
        alpha_re=float(alpha.real),
        alpha_im=float(alpha.imag),
        beta_re=float(beta.real),
        beta_im=float(beta.imag),
        active_bounds=active_bounds,
    )
    return result


def active_local_cuda(a, b, bounds, weights, c=None, *, alpha, beta):
    """Compute weighted local interval products with fixed allocation shapes.

    ``a`` is ``(nq,m,K)``, ``b`` is ``(nq,K,n)``, ``bounds`` is
    ``(nq,2)``, and ``weights`` is ``(nq,K)``.  All are local values inside
    the plan's shard_map. Full bounds retain the original JAX dense product;
    valid partial bounds use the cuBLAS pointer-view handler; invalid dynamic
    bounds produce an all-NaN result without entering the native call.
    """
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError("local active cuBLAS GEMM operand geometry mismatch")
    if a.dtype != b.dtype or a.dtype not in (jnp.float64, jnp.complex128):
        raise TypeError("local active cuBLAS GEMM requires matching float64/complex128 operands")
    if bounds.shape != (a.shape[0], 2) or not jnp.issubdtype(bounds.dtype, jnp.integer):
        raise ValueError("local active cuBLAS GEMM bounds must have shape (nq,2)")
    if weights.shape != (a.shape[0], a.shape[2]) or weights.dtype != a.dtype:
        raise ValueError("local active cuBLAS GEMM weights must match (nq,K) and operand dtype")
    if c is None:
        if beta != 0:
            raise ValueError("local active cuBLAS GEMM requires C when beta != 0")
        c = jnp.zeros((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype)
    elif c.shape != (a.shape[0], a.shape[1], b.shape[2]) or c.dtype != a.dtype:
        raise ValueError("local active cuBLAS GEMM C geometry or dtype mismatch")

    weighted_a = a * weights[:, None, :]
    lo, hi = bounds[:, 0], bounds[:, 1]
    valid = jnp.all((lo >= 0) & (lo <= hi) & (hi <= a.shape[2]))
    full = jnp.all((lo == 0) & (hi == a.shape[2]))
    nan = lambda _: jnp.full(c.shape, jnp.nan, dtype=a.dtype)

    def valid_call(_):
        return jax.lax.cond(
            full,
            lambda _: _dense(a, b, weighted_a, c, alpha=alpha, beta=beta),
            lambda _: _native(weighted_a, b, bounds, c, alpha=alpha, beta=beta),
            operand=None,
        )

    return jax.lax.cond(valid, valid_call, nan, operand=None)


def prepared_active_local_cuda(
    a, b, weights, c=None, *, active_bounds, alpha, beta,
):
    """Compute one host-validated interval encoded in immutable FFI attrs."""
    if (a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or
            a.shape[2] != b.shape[1]):
        raise ValueError("local prepared active cuBLAS GEMM operand geometry mismatch")
    if a.dtype != b.dtype or a.dtype not in (jnp.float64, jnp.complex128):
        raise TypeError(
            "local prepared active cuBLAS GEMM requires matching float64/complex128 operands")
    if weights.shape != (a.shape[0], a.shape[2]) or weights.dtype != a.dtype:
        raise ValueError(
            "local prepared active cuBLAS GEMM weights must match (nq,K) and operand dtype")
    if c is None:
        if beta != 0:
            raise ValueError(
                "local prepared active cuBLAS GEMM requires C when beta != 0")
        c = jnp.zeros((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype)
    elif c.shape != (a.shape[0], a.shape[1], b.shape[2]) or c.dtype != a.dtype:
        raise ValueError(
            "local prepared active cuBLAS GEMM C geometry or dtype mismatch")

    weighted_a = a * weights[:, None, :]
    pairs = np.asarray(active_bounds).reshape(-1, 2)
    if np.all(pairs[:, 0] == 0) and np.all(pairs[:, 1] == a.shape[2]):
        return _dense(a, b, weighted_a, c, alpha=alpha, beta=beta)
    return _prepared_native(
        weighted_a, b, c, active_bounds=active_bounds,
        alpha=alpha, beta=beta)


__all__ = [
    "active_local_cuda",
    "prepared_active_local_cuda",
    "require_active_local_cuda",
    "require_prepared_active_local_cuda",
]

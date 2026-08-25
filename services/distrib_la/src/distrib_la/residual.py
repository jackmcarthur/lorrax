"""Normwise residuals for distributed dense linear algebra.

This module is the one owner of matrix Frobenius norms and direct-solve
backward errors at the :mod:`distrib_la` door.  It deliberately builds
``A @ X - B`` through :func:`distrib_la.matmul`: the provider already owns
the distributed GEMM and its communication, so a diagnostic must not grow a
second SUMMA, gather, or hand-written collective.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from distrib_la.matmul import matmul

__all__ = ["frobenius_norm", "solve_backward_error"]


@jax.jit
def frobenius_norm(A: jax.Array) -> jax.Array:
    """Return ``||A||_F`` over the last two axes.

    Any leading batch axes are preserved.  On a face-sharded global array,
    JAX lowers the reduction over the two matrix axes to the required mesh
    reduction; no matrix is gathered and this function spells no collective
    of its own.
    """
    if A.ndim < 2:
        raise ValueError(
            f"frobenius_norm expects at least two matrix axes; got {A.shape}")
    return jnp.linalg.norm(A, ord="fro", axis=(-2, -1))


def _validate_solve_operands(A, X, B) -> tuple[tuple[int, ...], int, int, int]:
    if A.ndim < 2 or X.ndim != A.ndim or B.ndim != A.ndim:
        raise ValueError(
            "solve_backward_error expects A, X, and B with the same rank "
            f">=2; got A={A.shape}, X={X.shape}, B={B.shape}")
    if A.dtype != X.dtype or A.dtype != B.dtype:
        raise TypeError(
            "solve_backward_error operand dtypes disagree: "
            f"A={A.dtype}, X={X.dtype}, B={B.dtype}")
    batch_shape = tuple(int(s) for s in A.shape[:-2])
    if (tuple(int(s) for s in X.shape[:-2]) != batch_shape
            or tuple(int(s) for s in B.shape[:-2]) != batch_shape):
        raise ValueError(
            "solve_backward_error leading batch shapes disagree: "
            f"A={A.shape[:-2]}, X={X.shape[:-2]}, B={B.shape[:-2]}")
    m, k = (int(s) for s in A.shape[-2:])
    kx, nrhs = (int(s) for s in X.shape[-2:])
    mb, nrhs_b = (int(s) for s in B.shape[-2:])
    if k != kx or m != mb or nrhs != nrhs_b:
        raise ValueError(
            "solve_backward_error needs A(...,m,k), X(...,k,nrhs), and "
            f"B(...,m,nrhs); got A={A.shape}, X={X.shape}, B={B.shape}")
    nbatch = math.prod(batch_shape) if batch_shape else 1
    if nbatch < 1:
        raise ValueError("solve_backward_error batch extents must be nonzero")
    return batch_shape, m, k, nrhs


def _backward_error_ratio(residual_norm, denominator):
    """Apply the exact-zero policy for a normwise backward error."""
    return jnp.where(
        denominator == 0,
        jnp.where(residual_norm == 0, 0.0, jnp.inf),
        residual_norm / denominator,
    )


def solve_backward_error(
    A: jax.Array,
    X: jax.Array,
    B: jax.Array,
    *,
    mesh: Mesh,
    backend: str = "auto",
    batched_route: str = "auto",
    norm_a: jax.Array | None = None,
) -> jax.Array:
    """Return the per-batch normwise backward error of ``A X = B``.

    The returned array has ``A.shape[:-2]`` and contains

    ``||A X - B||_F / (||A||_F ||X||_F + ||B||_F)``.

    Arbitrary leading batch axes are flattened only for the existing
    rank-3 :func:`distrib_la.matmul` call and restored on return.  They must
    be replicated; the last two matrix axes retain the service's face layout.

    ``B`` is consumed: it is passed as GEMM's ``C`` operand with ``beta=-1``
    and its storage may become the residual.  A caller whose solve already
    donated its RHS must preserve exactly one explicit copy for this call.
    ``norm_a`` may supply a previously computed per-batch Frobenius norm when
    the same operator is applied to many RHS blocks.

    A zero denominator is reported as zero only for an exact zero residual;
    otherwise it is infinity.  No tolerance, condition estimate, or pass/fail
    policy belongs to this numerical primitive.
    """
    batch_shape, m, k, nrhs = _validate_solve_operands(A, X, B)
    a_norm = frobenius_norm(A) if norm_a is None else jnp.asarray(norm_a)
    if tuple(int(s) for s in a_norm.shape) != batch_shape:
        raise ValueError(
            f"solve_backward_error norm_a shape {a_norm.shape} does not "
            f"match batch shape {batch_shape}")
    x_norm = frobenius_norm(X)
    b_norm = frobenius_norm(B)

    if batch_shape:
        nbatch = math.prod(batch_shape)
        A_mm = jnp.reshape(A, (nbatch, m, k))
        X_mm = jnp.reshape(X, (nbatch, k, nrhs))
        B_mm = jnp.reshape(B, (nbatch, m, nrhs))
    else:
        A_mm, X_mm, B_mm = A, X, B
    residual_mm = matmul(
        A_mm, X_mm, B_mm, mesh=mesh, beta=-1,
        backend=backend, batched_route=batched_route)
    residual = (jnp.reshape(residual_mm, batch_shape + (m, nrhs))
                if batch_shape else residual_mm)
    residual_norm = frobenius_norm(residual)
    denominator = a_norm * x_norm + b_norm
    return _backward_error_ratio(residual_norm, denominator)

"""The local (one-device) LU solve ``A X = B``: ``jnp.linalg.solve`` with a per-leg route.

Every distrib_la site that solves a general system whole on one device (the ``one_device``
size route and the ``batch_reshard`` local stage, :mod:`distrib_la._batch_reshard`) calls
:func:`solve_local`.  It is a plain function (no ``jit`` boundary).  A leg without a row,
and a matrix outside its row, call ``jnp.linalg.solve`` itself, bit for bit.

ROCM: :data:`LOCAL_LU_ROUTE`.  rocBLAS 5 (ROCm 7.14.1 wheels, MI250X) ``{z,d}trsm``, left
side, returns a WRONG solution with status 0 on its in-place blocked path (taken when the
triangle's side is a multiple of 128 and the RHS is wide) once a 128-block diagonal offset of
the triangle crosses 2^31 BYTES -- i.e. once ``(n - 128)(n + 1) itemsize >= 2^31``.  XLA's
TriangularSolve (``jnp.linalg.solve``, ``lax.linalg.triangular_solve``, ``lu_solve``) and
``rocsolver_{z,d}getrs`` (also ``_64``) all reach it: complex128 n = 11648 exact, 11776,
12288, 14336, 16384 wrong (relative residual 5e-3 .. 9e-3); float64 16384 exact, 16512,
20480, 24576 wrong; ``rocblas_ztrsm_64`` is wrong at the same sizes, so it is not the int32
API.  ``getrf`` is exact at every size measured (complex128 to 16384, float64 to 23200).
Evidence and harnesses: ROCSOLVER_LU_REPORT.md (pool logs ``lurc_*``).

The row's target (``src/ffi/cpp/rocsolver/lu_solve_ffi.cc``) factors with rocSOLVER getrf and
solves on a correctly blocked path: every rocBLAS call gets a sub-matrix whose byte span from
its own pointer is below 2^31 (the offsets are formed in the handler, in 64-bit).  The route
rule is the same bound on the operands: a matrix whose A or B spans MORE than 2^31 bytes (so
that a byte offset into it can reach 2^31) takes the target; at or below it no rocBLAS offset
can reach 2^31 and ``jnp.linalg.solve`` is exact (and unchanged: float64 n = 16384, exactly
2^31 bytes, is exact on the native path).  A routed solve whose target is not loadable refuses by GATE ``local_lu_target``
(no fallback to the defective path).  Gate: ``tests/test_rocsolver_lu_gate.py``.

CUDA has no row: jaxlib's cuSOLVER LU was measured exact through n = 16384 complex128
(``resolve.ONE_DEVICE_BY_ROOM``'s comment), so the defect above does not occur there.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["LOCAL_LU_ROUTE", "LocalLuRow", "local_lu_row", "solve_local"]


class LocalLuRow(NamedTuple):
    """A leg's native one-device LU solve and the matrices it takes."""
    target: str              #: XLA FFI target (distrib_la.loader's table for the leg)
    dtypes: tuple            #: dtype names the handler serves
    min_span_bytes: int      #: a matrix whose A or B spans more than this many bytes is taken


#: Leg key -> its one-device LU solve.  ROCM: the span rule of the module docstring (2^31).
LOCAL_LU_ROUTE = {"ROCM": LocalLuRow("lorrax_rocsolver_lu_solve", ("complex128", "float64"), 1 << 31)}


def _process_leg_key():
    from lxkit import device_vendor, leg_for_vendor
    leg = leg_for_vendor(device_vendor(jax.devices()[0]))
    return None if leg is None else leg.key


def local_lu_row(a_shape, b_shape, dtype, leg_key=None):
    """The :data:`LOCAL_LU_ROUTE` row that solves this system, or None (``jnp.linalg.solve``).

    ``a_shape`` [..., n, n], ``b_shape`` [..., n, nrhs] (same batch).  A shape rule: every
    rank computes the same answer, no exchange."""
    row = LOCAL_LU_ROUTE.get(_process_leg_key() if leg_key is None else leg_key)
    if row is None or np.dtype(dtype).name not in row.dtypes:
        return None
    if len(a_shape) < 2 or len(b_shape) != len(a_shape):
        return None
    n, nrhs, item = int(a_shape[-1]), int(b_shape[-1]), np.dtype(dtype).itemsize
    if n * max(n, nrhs) * item <= row.min_span_bytes:
        return None
    return row


def _require_target(target):
    from distrib_la.loader import probe_target
    leg = _process_leg_key()
    usable, reason = probe_target(target, leg)        # loads the leg and registers its targets
    if not usable:
        raise RuntimeError(f"GATE local_lu_target: got: {target} unusable on {leg}: {reason}; "
                           f"want: the leg built with src/ffi/cpp/rocsolver (LOCAL_LU_ROUTE row)")


def _native_solve(row, a, b):
    """One FFI call for the whole stack.  The handler reads the row-major ``a`` as A^T
    (column-major) and factors it in place; ``b`` goes in as B^T row-major (= B column-major)
    and comes back as X^T.  ``a``/``b`` alias the factor/solution (no extra n^2 buffer)."""
    _require_target(row.target)
    bt = jnp.swapaxes(b, -1, -2)
    _lu, xt, _info = jax.ffi.ffi_call(
        row.target,
        (jax.ShapeDtypeStruct(a.shape, a.dtype), jax.ShapeDtypeStruct(bt.shape, bt.dtype),
         jax.ShapeDtypeStruct(a.shape[:-2], jnp.int32)),
        input_output_aliases={0: 0, 1: 1},
        vmap_method="broadcast_all")(a, bt)
    return jnp.swapaxes(xt, -1, -2)


def solve_local(a, b):
    """``jnp.linalg.solve(a, b)`` for ``a`` [..., n, n] and ``b`` [..., n, nrhs], by the leg's
    route.  A singular matrix gives non-finite X on both routes (no info is read)."""
    row = local_lu_row(a.shape, b.shape, a.dtype)
    if row is None or b.dtype != a.dtype:
        return jnp.linalg.solve(a, b)
    return _native_solve(row, a, b)

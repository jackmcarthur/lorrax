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

Three entry points, each taking the caller's own computation as ``native`` (the default is the
call it replaces), so a system below the span is the caller's original program bit for bit:
:func:`solve_local` (``jnp.linalg.solve``), :func:`lu_solve_local` (``jax.scipy.linalg.lu_solve``
on jax's ``(lu, piv)``), :func:`tri_solve_local` (a LOWER triangle: ``L x = b``, ``L^T x = b``,
``L^H x = b``).  float32 / complex64 have the same defect at their own thresholds (float32:
n = 23168 exact, 23296 wrong; complex64 as float64), so the row serves all four dtypes.

CUDA has no row: jaxlib's cuSOLVER LU was measured exact through n = 16384 complex128
(``resolve.ONE_DEVICE_BY_ROOM``'s comment), so the defect above does not occur there.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["LOCAL_LU_ROUTE", "LocalLuRow", "local_lu_row", "routes", "solve_local",
           "lu_solve_local", "tri_solve_local"]


class LocalLuRow(NamedTuple):
    """A leg's native one-device LU / triangular solves and the systems they take."""
    target: str              #: XLA FFI target: factor + solve (distrib_la.loader's table)
    tri_target: str          #: XLA FFI target: apply given factors / a triangle
    dtypes: tuple            #: dtype names the handlers serve
    min_span_bytes: int      #: a matrix whose A or B spans more than this many bytes is taken


#: Leg key -> its one-device LU solve.  ROCM: the span rule of the module docstring (2^31).
LOCAL_LU_ROUTE = {"ROCM": LocalLuRow("lorrax_rocsolver_lu_solve", "lorrax_rocsolver_tri_solve",
                                     ("float32", "float64", "complex64", "complex128"), 1 << 31)}

#: tri_solve_local's handler modes (src/ffi/cpp/rocsolver/lu_solve_ffi.cc).
_MODE_LU, _MODE_L, _MODE_LT = 1, 2, 3


def _process_leg_key():
    from lxkit import device_vendor, leg_for_vendor
    leg = leg_for_vendor(device_vendor(jax.devices()[0]))
    return None if leg is None else leg.key


def local_lu_row(a_shape, b_shape, dtype, leg_key=None):
    """The :data:`LOCAL_LU_ROUTE` row that solves this system, or None (the caller's own call).

    ``a_shape`` [..., n, n], ``b_shape`` [..., n, nrhs] or [..., n].  A shape rule: every rank
    computes the same answer, no exchange."""
    row = LOCAL_LU_ROUTE.get(_process_leg_key() if leg_key is None else leg_key)
    if row is None or np.dtype(dtype).name not in row.dtypes or len(a_shape) < 2:
        return None
    n, item = int(a_shape[-1]), np.dtype(dtype).itemsize
    nrhs = 1 if len(b_shape) == len(a_shape) - 1 else int(b_shape[-1])
    if n * max(n, nrhs) * item <= row.min_span_bytes:
        return None
    return row


def routes(a_shape, b_shape, dtype, leg_key=None) -> bool:
    """True when :func:`local_lu_row` takes the system (for call sites that restructure it)."""
    return local_lu_row(a_shape, b_shape, dtype, leg_key) is not None


def _require_target(target):
    from distrib_la.loader import probe_target
    leg = _process_leg_key()
    usable, reason = probe_target(target, leg)        # loads the leg and registers its targets
    if not usable:
        raise RuntimeError(f"GATE local_lu_target: got: {target} unusable on {leg}: {reason}; "
                           f"want: the leg built with src/ffi/cpp/rocsolver (LOCAL_LU_ROUTE row)")


def _as_matrix_rhs(a, b):
    vec = b.ndim == a.ndim - 1
    return (b[..., None] if vec else b), vec


def _workspace_bytes(mode, n, nrhs, dtype):
    """Native workspace + mode-0 pivots, owned by an explicit FFI result.

    The query uses the execution handler's mapped provider and blocked call
    sequence. It may create a small handle, but allocates no matrix or queried
    scratch buffer. The native owner caches sizes by current device and shape.
    """
    import ctypes
    from distrib_la.loader import get_lib
    kinds = {"float32": 0, "float64": 1, "complex64": 2, "complex128": 3}
    lib = get_lib("ROCM")
    name = "lrx_rocsolver_lu_workspace_bytes"
    if not hasattr(lib, name):
        raise RuntimeError(f"GATE local_lu_workspace: native provider must export {name}; rebuild the whole native pair")
    query = getattr(lib, name)
    query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                      ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_size_t]
    query.restype = ctypes.c_int
    out, error = ctypes.c_uint64(), ctypes.create_string_buffer(512)
    status = query(mode, kinds[np.dtype(dtype).name], n, nrhs,
                   ctypes.byref(out), error, ctypes.sizeof(error))
    if status:
        raise RuntimeError(f"GATE local_lu_workspace: mode={mode}, n={n}, nrhs={nrhs}, dtype={dtype}: "
                           + error.value.decode("utf-8", errors="replace"))
    return int(out.value)


def _workspace_result(a, b, mode):
    size = _workspace_bytes(int(mode), int(a.shape[-1]), int(b.shape[-1]),
                            np.dtype(a.dtype).name)
    return jax.ShapeDtypeStruct((size,), jnp.uint8)


def _native_solve(row, a, b):
    """One FFI call for the whole stack.  The handler reads the row-major ``a`` as A^T
    (column-major) and factors it in place; ``b`` goes in as B^T row-major (= B column-major)
    and comes back as X^T.  ``a``/``b`` alias the factor/solution (no extra n^2 buffer)."""
    _require_target(row.target)
    b, vec = _as_matrix_rhs(a, b)
    bt = jnp.swapaxes(b, -1, -2)
    _lu, xt, _info, _workspace = jax.ffi.ffi_call(
        row.target,
        (jax.ShapeDtypeStruct(a.shape, a.dtype), jax.ShapeDtypeStruct(bt.shape, bt.dtype),
         jax.ShapeDtypeStruct(a.shape[:-2], jnp.int32), _workspace_result(a, b, 0)),
        input_output_aliases={0: 0, 1: 1},
        vmap_method="broadcast_all")(a, bt)
    x = jnp.swapaxes(xt, -1, -2)
    return x[..., 0] if vec else x


def _native_tri(row, a, b, mode, piv=None, conj=False):
    """Apply factors / a triangle held in ``a`` (read only) to ``b``: one FFI call.  ``conj``
    conjugates B before and X after (fused into the layout copies): L^H x = b is mode 3 on
    conj(b)."""
    _require_target(row.tri_target)
    b, vec = _as_matrix_rhs(a, b)
    bt = jnp.swapaxes(b, -1, -2)
    if conj:
        bt = jnp.conj(bt)
    pv = (jnp.zeros(a.shape[:-1], jnp.int32) if piv is None
          else (piv.astype(jnp.int32) + 1))             # rocSOLVER pivots are 1-based
    xt, _workspace = jax.ffi.ffi_call(
        row.tri_target, (jax.ShapeDtypeStruct(bt.shape, bt.dtype), _workspace_result(a, b, mode)),
        input_output_aliases={1: 0}, vmap_method="broadcast_all")(a, bt, pv, mode=np.int64(mode))
    if conj:
        xt = jnp.conj(xt)
    x = jnp.swapaxes(xt, -1, -2)
    return x[..., 0] if vec else x


def solve_local(a, b, native=None):
    """``jnp.linalg.solve(a, b)`` for ``a`` [..., n, n] and ``b`` [..., n, nrhs] (or [..., n]), by
    the leg's route; ``native(a, b)`` (default ``jnp.linalg.solve``) below it.  A singular matrix
    gives non-finite X on both routes (no info is read)."""
    row = local_lu_row(a.shape, b.shape, a.dtype)
    if row is None or b.dtype != a.dtype:
        return (native or jnp.linalg.solve)(a, b)
    return _native_solve(row, a, b)


def lu_solve_local(lu, piv, b, native=None):
    """``jax.scipy.linalg.lu_solve((lu, piv), b, trans=0)`` on jax's factors (``lax.linalg.lu``
    / ``lu_factor``: A[perm] = L U, ``piv`` 0-based), by the leg's route."""
    row = local_lu_row(lu.shape, b.shape, lu.dtype)
    if row is None or b.dtype != lu.dtype:
        if native is not None:
            return native(lu, piv, b)
        import jax.scipy.linalg as jsl
        return jsl.lu_solve((lu, piv), b, trans=0)
    return _native_tri(row, lu, b, _MODE_LU, piv=piv)


def tri_solve_local(l, b, *, transpose=False, conjugate=False, native=None):
    """Solve with the LOWER triangle of ``l``: ``L x = b`` (default), ``L^T x = b``
    (``transpose``) or ``L^H x = b`` (``transpose`` and ``conjugate``), by the leg's route;
    ``native(l, b)`` below it (default ``lax.linalg.triangular_solve`` with the same flags).
    ``conjugate`` without ``transpose`` is not served on the route (GATE)."""
    row = local_lu_row(l.shape, b.shape, l.dtype)
    if row is None or b.dtype != l.dtype:
        if native is not None:
            return native(l, b)
        bb, vec = _as_matrix_rhs(l, b)
        x = jax.lax.linalg.triangular_solve(l, bb, left_side=True, lower=True,
                                            transpose_a=transpose, conjugate_a=conjugate)
        return x[..., 0] if vec else x
    if conjugate and not transpose:
        raise ValueError("GATE local_lu_tri_flags: conj(L) x = b (no transpose) has no routed form; "
                         "want L, L^T or L^H")
    if not transpose:
        return _native_tri(row, l, b, _MODE_L)
    return _native_tri(row, l, b, _MODE_LT, conj=conjugate and jnp.iscomplexobj(l))

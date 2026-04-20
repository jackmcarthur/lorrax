"""``distributed_trsm`` — JAX FFI wrapper around ``slate::trsm``.

Solves one of:
    op(A) * X = alpha * B   (side == "L")
    X * op(A) = alpha * B   (side == "R")

A is an n×n triangular matrix; B has shape (n, m) for side="L" or
(m, n) for side="R".  Output X has the same shape as B.

Only side="L" is currently wired; the handler accepts side="R" but the
Python side hasn't been tested.  Covers what we need for Cholesky-based
GW solves.
"""
from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .cholesky import SlateLowerL
from .context import get_or_init_context

__all__ = ["distributed_trsm"]

_FFI_TARGET = "lorrax_slate_trsm"

_SIDE  = {"L": 0, "R": 1}
_UPLO  = {"L": 0, "U": 1}
_OP    = {"N": 0, "T": 1, "C": 2}
_DIAG  = {"N": 0, "U": 1}


def distributed_trsm(
    A,
    B: jax.Array,
    *,
    mesh: Mesh,
    side: Literal["L", "R"] = "L",
    uplo: Literal["L", "U"] = "L",
    op: Literal["N", "T", "C"] = "N",
    diag: Literal["N", "U"] = "N",
    alpha: complex | float = 1.0,
    block_size: int | None = None,
) -> jax.Array:
    """Distributed triangular solve.

    A may be either a plain JAX array (interpreted as a row-major lower
    or upper triangular matrix per ``uplo``) or a :class:`SlateLowerL`
    handle returned by :func:`distributed_cholesky`.  In the handle
    case, the underlying SLATE-format buffer is passed straight back to
    SLATE without any layout transform — best for chained
    cholesky -> trsm where we want to keep SLATE's tile layout
    consistent.
    """
    if isinstance(A, SlateLowerL):
        # Handle path: SlateLowerL holds SLATE's tile-layout factor.  We
        # remap the user-facing op to the SLATE convention that gives the
        # right answer in JAX row-major view.  The forward solve needs
        # ConjTrans (not just Trans) so the complex case is handled
        # correctly; for real f64 ConjTrans collapses to Trans, no harm.
        # Empirically:
        #
        #   user op='N' (forward solve, L @ X = B)
        #     -> SLATE side='R', uplo='L', op='C'   residual ~1e-16
        #   user op='C' (adjoint solve, L^H @ X = B)
        #     -> SLATE side='R', uplo='L', op='N'   residual ~1e-16
        #
        # side and uplo from the caller are ignored for the handle path
        # (the handle's layout pins them).
        if op == "N":
            side, uplo, op = "R", "L", "C"
        elif op in ("C", "T"):
            side, uplo, op = "R", "L", "N"
        else:
            raise ValueError(f"distributed_trsm(SlateLowerL, ...): "
                             f"op must be 'N' (forward) or 'C' (adjoint); "
                             f"got {op!r}")
        A_raw = A.raw
        if block_size is None:
            block_size = A.nb
        if mesh.axis_names != A.mesh.axis_names:
            raise ValueError(
                "trsm mesh axis names don't match the handle's mesh; "
                "pass the same mesh used for distributed_cholesky.")
    else:
        A_raw = A

    if A_raw.ndim != 2 or A_raw.shape[0] != A_raw.shape[1]:
        raise ValueError(f"distributed_trsm: expected square A; got {A_raw.shape}")
    if B.ndim != 2:
        raise ValueError(f"distributed_trsm: expected 2D B; got {B.shape}")
    if A_raw.dtype != B.dtype:
        raise ValueError(
            f"distributed_trsm: A.dtype {A_raw.dtype} != B.dtype {B.dtype}")
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    if p != q:
        raise ValueError(f"slate.trsm currently requires square mesh; got {p}x{q}.")
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}x{q} != jax.process_count()={jax.process_count()}")
    n = int(A_raw.shape[0])
    if side == "L":
        if B.shape[0] != n:
            raise ValueError(
                f"side='L' requires B.shape[0]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[1])
    else:
        if B.shape[1] != n:
            raise ValueError(
                f"side='R' requires B.shape[1]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[0])
    if n % p != 0 or m % q != 0:
        raise ValueError(
            f"n={n}, m={m} must be divisible by mesh axes ({p},{q})")

    get_lib()
    ctx_handle = get_or_init_context(mesh)

    nb = n // p if block_size is None else int(block_size)

    # Local output shape matches B's sharding.
    bshape_local = (B.shape[0] // p, B.shape[1] // q)
    X_local = jax.ShapeDtypeStruct(bshape_local, B.dtype)

    alpha_c = complex(alpha)
    attrs = dict(
        n=n, m=m, nb=nb,
        side=_SIDE[side], uplo=_UPLO[uplo], op=_OP[op], diag=_DIAG[diag],
        alpha_re=float(alpha_c.real),
        alpha_im=float(alpha_c.imag),
        ctx_handle=int(ctx_handle),
    )

    @partial(shard_map, mesh=mesh,
             in_specs=(P("x", "y"), P("x", "y")),
             out_specs=P("x", "y"),
             check_rep=False)
    def _call(local_A, local_B):
        return jax.ffi.ffi_call(_FFI_TARGET, X_local)(
            local_A, local_B, **attrs)

    return _call(A_raw, B)

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
from .context import get_or_init_context

__all__ = ["distributed_trsm"]

_FFI_TARGET = "lorrax_slate_trsm"

_SIDE  = {"L": 0, "R": 1}
_UPLO  = {"L": 0, "U": 1}
_OP    = {"N": 0, "T": 1, "C": 2}
_DIAG  = {"N": 0, "U": 1}


def distributed_trsm(
    A: jax.Array,
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
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_trsm: expected square A; got {A.shape}")
    if B.ndim != 2:
        raise ValueError(f"distributed_trsm: expected 2D B; got {B.shape}")
    if A.dtype != B.dtype:
        raise ValueError(
            f"distributed_trsm: A.dtype {A.dtype} != B.dtype {B.dtype}")
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
    n = int(A.shape[0])
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

    return _call(A, B)

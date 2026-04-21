"""``distributed_trsm`` — JAX FFI wrapper around ``slate::trsm``.

Solves one of:
    op(A) @ X = alpha * B   (side='L')
    X @ op(A) = alpha * B   (side='R')

Layout: same scheme as ``distributed_cholesky`` (per-rank local
transpose + C++ side rank remap).  See ``cholesky.py`` and
``src/ffi/slate/README.md`` for the design notes.

For the :class:`SlateLowerL` handle path the underlying buffer is
already in SLATE-tile col-major (``P('y','x')``-sharded), so A skips
the local transpose; ``side``/``uplo`` are pinned by the handle and
``op`` selects forward (``'N'``: ``L X = B``) vs adjoint (``'C'``:
``L^H X = B``).
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
from .context import get_or_init_context, validate_mesh

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

    A : SlateLowerL or square 2-D jax.Array sharded ``P('x','y')``.
    B : 2-D jax.Array sharded ``P('x','y')``.

    For the SlateLowerL handle path, side/uplo are pinned to the
    standard cholesky-factor convention: op='N' is forward solve
    (L @ X = B); op='C' is adjoint (L^H @ X = B).
    """
    p, q = validate_mesh(mesh)

    if isinstance(A, SlateLowerL):
        # Handle.raw is already in SLATE-tile col-major (P('y','x') sharded).
        if op not in ("N", "C"):
            raise ValueError(f"distributed_trsm(SlateLowerL, ...): "
                             f"op must be 'N' or 'C'; got {op!r}")
        side, uplo = "L", "L"
        n = A.n
        if block_size is None:
            block_size = A.nb
        if mesh.axis_names != A.mesh.axis_names:
            raise ValueError(
                "trsm mesh axis names don't match the handle's mesh.")
        A_is_handle = True
        A_arg = A.raw
    else:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"distributed_trsm: expected square A; got {A.shape}")
        n = int(A.shape[0])
        A_is_handle = False
        A_arg = A

    if B.ndim != 2:
        raise ValueError(f"distributed_trsm: expected 2D B; got {B.shape}")
    if A_arg.dtype != B.dtype:
        raise ValueError(
            f"distributed_trsm: A.dtype {A_arg.dtype} != B.dtype {B.dtype}")

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
    nb = n // max(p, q) if block_size is None else int(block_size)

    # Local transpose of B: each rank flips its (B0/p, B1/q) row-major
    # shard to (B1/q, B0/p) row-major (= original block in col-major
    # layout).  Local op only; no inter-rank comm.
    bshape_local_T = (B.shape[1] // q, B.shape[0] // p)
    X_local_T = jax.ShapeDtypeStruct(bshape_local_T, B.dtype)

    alpha_c = complex(alpha)
    attrs = dict(
        n=n, m=m, nb=nb,
        side=_SIDE[side], uplo=_UPLO[uplo], op=_OP[op], diag=_DIAG[diag],
        alpha_re=float(alpha_c.real),
        alpha_im=float(alpha_c.imag),
        ctx_handle=int(ctx_handle),
    )

    # When A is a handle, A_arg is already P('y','x') sharded with
    # col-major bytes; just feed it through.  Otherwise, local-transpose.
    if A_is_handle:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P("y", "x"), P("x", "y")),
                 out_specs=P("y", "x"), check_rep=False)
        def _trsm(local_A_handle, local_B):
            local_B_T = jnp.transpose(local_B, (1, 0))
            return jax.ffi.ffi_call(_FFI_TARGET, X_local_T)(
                local_A_handle, local_B_T, **attrs)
        X_T = _trsm(A_arg, B)
    else:
        @partial(shard_map, mesh=mesh,
                 in_specs=(P("x", "y"), P("x", "y")),
                 out_specs=P("y", "x"), check_rep=False)
        def _trsm(local_A, local_B):
            local_A_T = jnp.transpose(local_A, (1, 0))
            local_B_T = jnp.transpose(local_B, (1, 0))
            return jax.ffi.ffi_call(_FFI_TARGET, X_local_T)(
                local_A_T, local_B_T, **attrs)
        X_T = _trsm(A_arg, B)

    # Local-transpose X back to user's P('x','y') row-major.
    @partial(shard_map, mesh=mesh,
             in_specs=P("y", "x"), out_specs=P("x", "y"),
             check_rep=False)
    def _untranspose(local_X_T):
        return jnp.transpose(local_X_T, (1, 0))
    return _untranspose(X_T)

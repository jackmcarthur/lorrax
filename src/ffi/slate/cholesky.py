"""``distributed_cholesky`` — JAX FFI wrapper around ``slate::potrf``.

Shape contract:
    A : (n, n) Hermitian positive-definite, P('x','y') sharded on `mesh`.
    L : (n, n) lower-triangular factor of A = L L^H, same sharding.

Only ``Uplo::Lower`` is supported (SLATE's potrf).  dtype F64 or C128.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_context

__all__ = ["distributed_cholesky"]

_FFI_TARGET = "lorrax_slate_potrf"


def distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
    block_size: int | None = None,
) -> jax.Array:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_cholesky: expected square A; got {A.shape}")
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    if p != q:
        raise ValueError(f"SLATE potrf requires square mesh; got p={p}, q={q}.")
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}x{q} != jax.process_count()={jax.process_count()}")
    n = int(A.shape[0])
    if n % p != 0:
        raise ValueError(f"n={n} must be divisible by mesh axis size {p}")

    get_lib()
    ctx_handle = get_or_init_context(mesh)

    nb = n // p if block_size is None else int(block_size)
    L_local = jax.ShapeDtypeStruct((n // p, n // q), A.dtype)

    attrs = dict(n=n, nb=nb, ctx_handle=int(ctx_handle))

    @partial(shard_map, mesh=mesh,
             in_specs=P("x", "y"), out_specs=P("x", "y"),
             check_rep=False)
    def _call(local_A):
        return jax.ffi.ffi_call(_FFI_TARGET, L_local)(local_A, **attrs)

    return _call(A)

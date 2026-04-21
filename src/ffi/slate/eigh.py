"""``distributed_eigh`` — JAX FFI wrapper around ``slate::heev``.

Shape contract mirrors ``ffi.cusolvermp.distributed_eigh`` so call sites
can swap backends with a one-line import change.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_context, validate_mesh

__all__ = ["distributed_eigh"]

_FFI_TARGET = "lorrax_slate_eigh"


def distributed_eigh(
    A: jax.Array,
    *,
    mesh: Mesh,
    compute_evecs: bool = True,
    block_size: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition via SLATE's ``heev``.

    Parameters
    ----------
    A
        Square ``(n, n)`` array, dtype ``float64`` (symmetric) or
        ``complex128`` (Hermitian), sharded ``P('x','y')`` on ``mesh``.
    mesh
        2-D ``Mesh`` labelled ``('x','y')``.  Must be *square*
        (``mesh.shape['x'] == mesh.shape['y']``) — SLATE's heev rejects
        rectangular grids.
    compute_evecs
        If False, only eigenvalues are meaningful; ``Q`` is still returned
        but its contents are unspecified.
    block_size
        Override SLATE's tile size ``nb``.  Default ``n/p`` (one tile per
        rank = JAX's block sharding matches SLATE's tile grid exactly,
        eigenvectors come out in-place).  Smaller ``nb`` increases
        panel-factor parallelism at the cost of more MPI traffic; 256 is a
        common A100 default for large ``n``.

    Returns
    -------
    W
        ``(n,)`` real eigenvalues, ascending, replicated.  Dtype is the
        real-part of A's dtype (``float64`` for both F64 and C128).
    Q
        ``(n, n)`` eigenvectors, same dtype as A, sharded ``P('x','y')``.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_eigh: expected a square matrix, "
                         f"got {A.shape}")
    p, q = validate_mesh(mesh, require_square=True)
    n = int(A.shape[0])
    if n % p != 0:
        raise ValueError(
            f"distributed_eigh: n={n} must be divisible by mesh axis size {p}.")

    get_lib()
    ctx_handle = get_or_init_context(mesh)

    nb = n // p if block_size is None else int(block_size)

    local_rows = n // p
    local_cols = n // q
    # Real-part dtype for eigenvalues (float64 for both F64 and C128).
    w_dtype = jnp.float64 if A.dtype in (jnp.complex128, jnp.float64) \
              else jnp.float32
    W_local = jax.ShapeDtypeStruct((n,), w_dtype)               # replicated
    Q_local = jax.ShapeDtypeStruct((local_rows, local_cols), A.dtype)

    attrs = dict(
        n=n, nb=nb,
        ctx_handle=int(ctx_handle),
        compute_evecs=bool(compute_evecs),
    )

    @partial(shard_map,
             mesh=mesh,
             in_specs=P("x", "y"),
             out_specs=(P(), P("x", "y")),
             check_rep=False)
    def _call(local_A):
        return jax.ffi.ffi_call(
            _FFI_TARGET, (W_local, Q_local))(local_A, **attrs)

    W, Q = _call(A)
    return W, Q

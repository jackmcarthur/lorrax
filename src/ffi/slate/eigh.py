"""``distributed_eigh`` — JAX FFI wrapper around ``slate::heev``.

Shape contract mirrors ``ffi.cusolvermp.distributed_eigh`` so call sites
can swap backends with a one-line import change — with one upgrade: the
returned ``Q`` here is the TRUE eigenvector matrix (columns are
eigenvectors of A, ``A @ Q == Q @ diag(W)``), not the conj-transposed
raw buffer the cusolvermp wrapper returns.

Layout: same scheme as ``distributed_cholesky`` — per-rank local
transpose on the input (row-major JAX shard → col-major bytes for
SLATE) paired with the C++ comm rank-remap, and a local untranspose on
the returned eigenvector tiles.  See ``cholesky.py`` and
``src/ffi/slate/README.md``.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from .context import (ensure_registered, get_or_init_context, validate_mesh,
                      validate_tile_layout)

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
        Override SLATE's tile size ``nb``.  Default ``n/p``, which is the
        ONLY value for which JAX's block sharding coincides with SLATE's
        block-cyclic tile grid on a multi-rank mesh — overriding it there
        makes SLATE assemble a permuted global matrix (eigenvalues survive
        the symmetric permutation; eigenvectors do not).  Overriding is
        therefore rejected on multi-rank meshes; on a 1×1 mesh any ``nb``
        is layout-safe.

    Returns
    -------
    W
        ``(n,)`` real eigenvalues, ascending, replicated.  Dtype is the
        real-part of A's dtype (``float64`` for both F64 and C128).
    Q
        ``(n, n)`` eigenvectors of A (as columns), same dtype as A,
        sharded ``P('x','y')``.  ``A @ Q == Q @ diag(W)`` to machine
        precision.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_eigh: expected a square matrix, "
                         f"got {A.shape}")
    p, q = validate_mesh(mesh, require_square=True)
    n = int(A.shape[0])
    if n % p != 0:
        raise ValueError(
            f"distributed_eigh: n={n} must be divisible by mesh axis size {p}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    nb = n // p if block_size is None else int(block_size)
    validate_tile_layout(n, nb, p, q, what="distributed_eigh")

    local_rows = n // p
    local_cols = n // q
    # Real-part dtype for eigenvalues (float64 for both F64 and C128).
    w_dtype = jnp.float64 if A.dtype in (jnp.complex128, jnp.float64) \
              else jnp.float32
    W_local = jax.ShapeDtypeStruct((n,), w_dtype)               # replicated
    # Q comes back as SLATE col-major tiles = row-major bytes of the
    # per-rank block of Q^T; assembled at P('y','x') then locally
    # untransposed (square mesh: local shape is (n/q, n/p)).
    Q_local_T = jax.ShapeDtypeStruct((local_cols, local_rows), A.dtype)

    attrs = dict(
        n=n, nb=nb,
        ctx_handle=int(ctx_handle),
        compute_evecs=bool(compute_evecs),
    )

    @partial(shard_map,
             mesh=mesh,
             in_specs=P("x", "y"),
             out_specs=(P(), P("y", "x")),
             check_rep=False)
    def _call(local_A):
        # Local transpose: row-major shard bytes → col-major block, the
        # layout SLATE reads (pure local op, pairs with the C++ comm
        # rank-remap).  Same convention as distributed_cholesky.
        local_A_T = jnp.transpose(local_A, (1, 0))
        return jax.ffi.ffi_call(
            _FFI_TARGET, (W_local, Q_local_T))(local_A_T, **attrs)

    W, Q_T = _call(A)

    @partial(shard_map, mesh=mesh,
             in_specs=P("y", "x"), out_specs=P("x", "y"),
             check_rep=False)
    def _untranspose(local_Q_T):
        return jnp.transpose(local_Q_T, (1, 0))

    return W, _untranspose(Q_T)

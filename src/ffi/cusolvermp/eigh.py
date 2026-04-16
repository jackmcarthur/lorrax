"""``distributed_eigh`` — JAX FFI wrapper around cusolverMpSyevd.

Usage
-----
Start in a multi-process JAX program (``jax.distributed.initialize()``
already called; one process per GPU).  Build a 2-D mesh labelled ``('x','y')``.
Shard the matrix with ``NamedSharding(mesh, P('x','y'))``.  Call::

    from ffi.cusolvermp import distributed_eigh
    evals, Q = distributed_eigh(A, mesh=mesh)

Layout note
-----------
cuSOLVERMp interprets the input buffer as a column-major ScaLAPACK tile.
JAX stores arrays row-major.  For a Hermitian matrix A = A^H, passing the
row-major local shard to cuSOLVERMp is equivalent to handing it A^T; since
the eigenvalues of A^T equal those of A (and of conj(A) for Hermitian),
the returned eigenvalues are correct.  The returned eigenvectors correspond
to A interpreted *column-major*; for a row-major consumer they represent
the eigenvectors of A.conj().T = A, just with a transposed storage layout.

Only one rank-per-GPU + one tile-per-rank is supported in this first
iteration.  The block size is fixed to ``(n/p, n/q)``.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_context

__all__ = ["distributed_eigh"]


_DTYPE_TO_TARGET = {
    jnp.float64:    "lorrax_cusolvermp_eigh_f64",
    jnp.complex128: "lorrax_cusolvermp_eigh_c128",
}


def _ffi_fn_for(dtype: jnp.dtype):
    """Return a closure that runs the FFI call for the given dtype."""
    name = _DTYPE_TO_TARGET.get(jnp.dtype(dtype).type)
    if name is None:
        raise TypeError(
            f"distributed_eigh: dtype {dtype} not supported.  "
            f"Supported: float64, complex128.")

    eval_dtype = jnp.float64  # cuSOLVERMp returns real evs for Hermitian too

    def call(local_A, *, n, mb, nb, ctx_handle, compute_evecs):
        eval_type = jax.ShapeDtypeStruct((n,), eval_dtype)
        Q_type    = jax.ShapeDtypeStruct(local_A.shape, local_A.dtype)
        evals, Q = jax.ffi.ffi_call(name, (eval_type, Q_type))(
            local_A,
            n=n, mb=mb, nb=nb, ctx_handle=ctx_handle,
            compute_evecs=compute_evecs,
        )
        return evals, Q

    return call


def distributed_eigh(
    A: jax.Array,
    *,
    mesh: Mesh,
    compute_evecs: bool = True,
    col_major: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition via cuSOLVERMp.

    Parameters
    ----------
    A
        Square ``(n, n)`` array, dtype float64 (symmetric) or complex128
        (Hermitian).  Must be sharded ``P('x','y')`` over ``mesh``.
    mesh
        2-D ``Mesh`` labelled ``('x','y')``.  Shape must satisfy
        ``x*y == jax.process_count()`` and ``n % x == n % y == 0``.
    compute_evecs
        If False, return only eigenvalues (Q is still returned, but its
        contents are undefined; consumer should ignore).
    col_major
        Passed to cusolverMp grid construction.  Leave default.

    Returns
    -------
    evals
        ``(n,)`` float64, replicated ``P()``.
    Q
        ``(n, n)``, same dtype as A, sharded ``P('x','y')``.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_eigh expects a square matrix; got {A.shape}")
    n = int(A.shape[0])
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    if n % p != 0 or n % q != 0:
        raise ValueError(
            f"n={n} must divide evenly by mesh shape ({p},{q}) in this "
            f"iteration (single-tile-per-rank).")
    # cuSOLVERMp is documented to require the block size to divide the
    # matrix evenly and to be "reasonable" (typically 32-256).  We set
    # mb = nb = min(n/p, n/q, 32) so the block layout degenerates to JAX's
    # block sharding when n == 32 * max(p,q) — the common case.
    # TODO: support larger block sizes once the divide-by-zero inside
    # cusolverMpSyevd_bufferSize with mb = n/p is understood.
    mb = min(n // p, 32)
    nb = min(n // q, 32)
    if n % (mb * p) != 0 or n % (nb * q) != 0:
        raise ValueError(
            f"distributed_eigh: n={n} must be divisible by mb*p={mb*p} "
            f"and nb*q={nb*q}.  Current mb=nb={mb}.")

    # Ensure the shared library is loaded and targets registered.
    get_lib()

    # Build / fetch the cusolverMp context (collective — every process
    # participates in the NCCL bootstrap on first call).
    ctx_handle = get_or_init_context(mesh, col_major=col_major)

    impl = _ffi_fn_for(A.dtype)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=P("x", "y"),
        out_specs=(P(), P("x", "y")),
        check_rep=False,
    )
    def _call(local_A):
        return impl(local_A,
                    n=n, mb=mb, nb=nb,
                    ctx_handle=int(ctx_handle),
                    compute_evecs=compute_evecs)

    evals, Q = _call(A)
    return evals, Q

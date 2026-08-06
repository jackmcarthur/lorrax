"""``distributed_eigh`` — JAX FFI wrapper around cusolverMpSyevd.

This file is intentionally written flat (no factory, no callbacks): it is
the template to copy when adding the next distributed routine (e.g.
``distributed_cholesky``).  The three pieces a new routine has to decide
are spelled out in comments:

    1. FFI target name registered in ``common/ffi_loader.py``.
    2. Output shapes + partition specs (inside ``shard_map``).
    3. Attributes forwarded to the C++ handler.

Everything else — loading the ``.so``, initialising the NCCL/cuSOLVERMp
context, validating the mesh, dispatching by dtype — is shared and
handled by ``get_lib()`` and ``get_or_init_context(mesh)``.

Layout note
-----------
cuSOLVERMp interprets the input buffer as a column-major ScaLAPACK tile.
JAX stores arrays row-major.  For a Hermitian matrix A = A^H, passing the
row-major local shard to cuSOLVERMp is equivalent to handing it A^T; since
the eigenvalues of A^T equal those of A (and of conj(A) for Hermitian),
the returned eigenvalues are correct.

The eigenVECTORS inherit the same untransposed layout: the returned ``Q``
raw buffer satisfies ``Q.conj().T[:, i] = i-th eigenvector`` (for F64,
``Q.T[:, i]``) — verified on 1x1 and 2x2 meshes, pinned by
``tests/test_ffi_linalg_contract.py``.  Callers that need conventional
column eigenvectors must conj-transpose.  (``ffi.slate.distributed_eigh``
returns TRUE column eigenvectors — the two wrappers share shapes but NOT
the Q layout convention.)

Mesh restriction: cusolverMpSyevd requires square ScaLAPACK blocks
(mb == nb).  On a non-square mesh the natural one-tile-per-rank blocks
are rectangular and syevd DEADLOCKS inside a collective instead of
returning an error status (observed 4x1/1x4, 2026-07-10) — so this
wrapper rejects p != q up front.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from common.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_context

__all__ = ["distributed_eigh"]

# (1) FFI target name registered in common/ffi_loader.py.
_FFI_TARGET = "lorrax_cusolvermp_eigh"


def distributed_eigh(
    A: jax.Array,
    *,
    mesh: Mesh,
    compute_evecs: bool = True,
    block_size: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Distributed Hermitian eigendecomposition via cuSOLVERMp.

    Parameters
    ----------
    A
        Square ``(n, n)`` array, dtype ``float64`` (symmetric) or
        ``complex128`` (Hermitian), sharded ``P('x','y')`` on ``mesh``.
    mesh
        2-D ``Mesh`` labelled ``('x','y')``.  ``mesh.shape['x'] *
        mesh.shape['y']`` must equal ``jax.process_count()``, and ``n``
        must be divisible by both axis sizes.
    compute_evecs
        If False, only eigenvalues are meaningful (Q is still returned but
        its contents should be ignored).
    block_size
        Override the 2-D block-cyclic tile size.  Default ``n/p`` (one
        tile per rank — matches JAX's block sharding, eigenvectors come
        out in-place).  Smaller blocks speed up the solve at larger n
        (block=256 gives 2.4× at n=16k) but return eigenvectors in a
        block-cyclic permutation of the input basis.

    Returns
    -------
    W
        ``(n,)`` ``float64`` eigenvalues, ascending, replicated.
    Q
        ``(n, n)`` raw eigenvector buffer, same dtype as A, sharded
        ``P('x','y')``.  ``Q.conj().T`` holds the eigenvectors as
        columns (see the module Layout note).
    """
    # --- validate ---------------------------------------------------------
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"distributed_eigh: expected a square matrix, "
                         f"got {A.shape}")
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    if p * q != jax.process_count():
        raise ValueError(
            f"mesh {p}×{q} does not cover all "
            f"{jax.process_count()} JAX processes")
    if p != q:
        raise ValueError(
            f"distributed_eigh: mesh {p}x{q} is not square — "
            f"cusolverMpSyevd requires mb == nb and DEADLOCKS (no error "
            f"status) on the rectangular one-tile-per-rank blocks a "
            f"non-square mesh produces.  Use a square mesh.")
    n = int(A.shape[0])
    if n % p != 0 or n % q != 0:
        raise ValueError(
            f"n={n} must be divisible by mesh shape ({p},{q}).")

    # --- one-time setup per process (lazy, cached) ------------------------
    get_lib()                                   # load the .so, register targets
    ctx_handle = get_or_init_context(mesh)      # NCCL + cal_comm + cusolverMp

    mb = n // p if block_size is None else block_size
    nb = n // q if block_size is None else block_size

    # (2) Local output shapes + partition specs for shard_map.
    local_rows = n // p
    local_cols = n // q
    W_local = jax.ShapeDtypeStruct((n,), jnp.float64)          # replicated
    Q_local = jax.ShapeDtypeStruct((local_rows, local_cols), A.dtype)

    # (3) Attributes forwarded to the C++ handler.
    attrs = dict(n=n, mb=mb, nb=nb,
                 ctx_handle=int(ctx_handle),
                 compute_evecs=bool(compute_evecs))

    @partial(shard_map,
             mesh=mesh,
             in_specs=P("x", "y"),
             out_specs=(P(), P("x", "y")),
             check_vma=False)
    def _call(local_A):
        return jax.ffi.ffi_call(
            _FFI_TARGET, (W_local, Q_local))(local_A, **attrs)

    W, Q = _call(A)
    return W, Q

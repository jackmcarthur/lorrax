"""``batched_distributed_solve_lu`` — per-q ScaLAPACK ``pXgetrf`` + ``pXgetrs``.

Host-platform (JAX CPU backend) twin of
:func:`ffi.cusolvermp.batched_distributed_solve_lu` — the portable backend
for the ``distributed_lu`` axis.  Identical call contract:

    A : (Nq, N, N)     P(None, 'x', 'y')   (donated — factored in place)
    B : (Nq, N, NRHS)  P(None, 'x', 'y')
    X : same shape/sharding as B

ScaLAPACK comes from Cray LibSci, which ``liblorrax_ffi_host.so`` already
links for SLATE's BLAS — no extra dependency.  The BLACS grid is built on
the slate context's rank-remapped MPI comm ("C" grid order lands JAX shard
(mx, my) at grid (mx, my)); the wrapper reuses ``ffi.slate.context`` for
mesh validation, registration, and the per-(p, q) ctx cache.

Mesh restriction: ``pXgetrf`` requires SQUARE descriptor blocks
(MB_A == NB_A), so with the one-tile-per-rank layout only square and 1-D
meshes are layout-consistent (block g = N / max(Px, Py)) — same geometry
class as the slate single-matrix ops, but 1×q is fine here (no SLATE
stride assert).  Host-only: on a GPU mesh use ``distributed_lu =
cusolvermp`` instead.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..slate.batched import _mesh_key
from ..slate.context import (ensure_registered, get_or_init_context,
                             validate_mesh)

__all__ = ["batched_distributed_solve_lu"]

_SOLVE_LU_TARGET = "lorrax_scalapack_batched_solve_lu"

# jit(shard_map) cache per signature — see ffi/phdf5/ARCHITECTURE.md §2.4.
_JIT_CACHE: dict = {}


def batched_distributed_solve_lu(
    A: jax.Array,
    B: jax.Array,
    *,
    mesh: Mesh,
) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` via per-q ScaLAPACK ``getrf`` + ``getrs``.

    See the module docstring for the sharding contract.  ``A`` is donated
    (the LU factors scribble over its buffer); pivots stay internal.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"scalapack solve_lu: expected A of shape (Nq, N, N); got {A.shape}")
    if (B.ndim != 3 or int(B.shape[0]) != int(A.shape[0])
            or int(B.shape[1]) != int(A.shape[1])):
        raise ValueError(
            f"scalapack solve_lu: B must be (Nq, N, NRHS) with Nq, N "
            f"matching A; got A={A.shape}, B={B.shape}")
    if A.dtype != B.dtype:
        raise ValueError(f"A.dtype {A.dtype} != B.dtype {B.dtype}")

    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise ValueError(
            f"scalapack solve_lu is host-only but the mesh devices are "
            f"{plat!r}; use distributed_lu = cusolvermp on GPU meshes.")

    Px, Py = validate_mesh(mesh)
    if Px > 1 and Py > 1 and Px != Py:
        raise ValueError(
            f"scalapack solve_lu: mesh {Px}x{Py} unsupported — pXgetrf "
            f"needs square blocks (MB == NB), which the one-tile-per-rank "
            f"layout only gives on square or 1-D meshes.")

    nq   = int(A.shape[0])
    n    = int(A.shape[1])
    nrhs = int(B.shape[2])
    gmax = max(Px, Py)
    if n % gmax != 0:
        raise ValueError(f"N={n} must be divisible by max(Px,Py)={gmax}.")
    if nrhs % Py != 0:
        raise ValueError(f"NRHS={nrhs} must be divisible by Py={Py}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    g = n // gmax                      # square block (MB_A == NB_A)
    nb_b = nrhs // Py if Py > 1 else nrhs

    key = ("scalapack_lu", _mesh_key(mesh), A.dtype,
           nq, n, nrhs, g, nb_b, int(ctx_handle))
    jit_solve = _JIT_CACHE.get(key)
    if jit_solve is None:
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // Py, n // Px), B.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs, g=g, nb_b=nb_b,
                     ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_rep=False)
        def _solve(local_A, local_B):
            # Inner transpose: row-major shard → col-major local block
            # (same convention as every slate/cusolvermp wrapper).
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            X_T = jax.ffi.ffi_call(
                _SOLVE_LU_TARGET, X_local_T,
                input_output_aliases={1: 0},
            )(local_A_T, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_solve = jax.jit(_solve, donate_argnums=(0, 1))
        _JIT_CACHE[key] = jit_solve

    return jit_solve(A, B)

"""``batched_distributed_eigh`` — ScaLAPACK ``pzheevd`` / ``pdsyevd``.

The **permanent CPU backend for distributed Hermitian eigendecomposition**
(``ffi.linalg.resolve_backend('eigh', 'distributed', <cpu mesh>)``).  SLATE's
host ``heev`` SIGSEGVs deterministically on this stack — bug L-2, reproduced
down to a 1×1 mesh, single rank — while ScaLAPACK's routines on the SAME
library and MPI context are clean.  ScaLAPACK comes from whichever
implementation of the ScaLAPACK API the host library was linked against
(see ``ffi.scalapack``); ``liblorrax_ffi_host.so`` already links it for
``solve_lu``, so this costs no new dependency.

Layout contract
---------------
::

    A : (Nq, N, N)  P(None, 'x', 'y')     Hermitian (lower triangle read)
    W : (Nq, N)     REPLICATED float64    ascending eigenvalues
    Z : (Nq, N, N)  P(None, 'x', 'y')     eigenvectors as COLUMNS,
                                          A[q] @ Z[q] == Z[q] @ diag(W[q])

``A`` is not donated (``pXheevd`` destroys its input, so the handler stages
each q into a scratch block).

Mesh restriction: ``pXheevd`` requires SQUARE descriptor blocks
(``MB_A == NB_A``), so with the one-tile-per-rank layout only **square and
1-D** meshes are layout-consistent (block ``g = N / max(Px, Py)``) — the
same geometry class as :func:`ffi.scalapack.batched_distributed_solve_lu`,
and unlike SLATE there is no 1×q stride assert, so ``1×q`` is allowed too.
Host-only: on a GPU mesh use ``ffi.cusolvermp.distributed_eigh``.

GAUGE WARNING for callers: eigenVECTORS are not mesh-invariant.  Any
degenerate (or numerically near-degenerate) eigenvalue leaves an arbitrary
unitary rotation inside its subspace, and the block-cyclic reduction picks
a different representative on a different process grid.  Compare
gauge-INVARIANT combinations (``W``, ``Z f(W) Zᴴ``, a projector, a
pseudo-inverse) across meshes — never ``Z`` itself.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..slate.batched import _mesh_key
from ..slate.context import ensure_registered, get_or_init_context, validate_mesh

__all__ = ["batched_distributed_eigh", "distributed_eigh", "validate_eigh_mesh"]

_EIGH_TARGET = "lorrax_scalapack_eigh"

# jit(shard_map) cache per signature — see ffi/phdf5/ARCHITECTURE.md §2.4.
_JIT_CACHE: dict = {}


def validate_eigh_mesh(mesh: Mesh) -> tuple[int, int]:
    """``(Px, Py)`` for a mesh this backend can run on; raises otherwise.

    THE call-time geometry guard, mirrored by
    ``ffi.linalg.resolve._check_geometry`` so a resolved ``'scalapack'``
    is a promise the call keeps (the lesson of bug L-1).
    """
    Px, Py = validate_mesh(mesh)
    if Px > 1 and Py > 1 and Px != Py:
        raise ValueError(
            f"scalapack eigh: mesh {Px}x{Py} unsupported — pXheevd needs "
            f"square descriptor blocks (MB == NB), which the "
            f"one-tile-per-rank layout only gives on square or 1-D meshes.")
    return Px, Py


def batched_distributed_eigh(
    A: jax.Array, *, mesh: Mesh,
) -> Tuple[jax.Array, jax.Array]:
    """Eigendecompose every ``A[q]`` with ScaLAPACK's divide & conquer.

    One FFI call for the whole stack (the handler loops q internally,
    reusing one descriptor and one workspace), so ``Nq`` matrices cost one
    collective-serialisation round, not ``Nq``.

    See the module docstring for the sharding contract and the gauge
    warning.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"scalapack eigh: expected A of shape (Nq, N, N); got {A.shape}")

    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise ValueError(
            f"scalapack eigh is host-only but the mesh devices are {plat!r}; "
            f"use ffi.cusolvermp.distributed_eigh on GPU meshes.")

    Px, Py = validate_eigh_mesh(mesh)
    nq = int(A.shape[0])
    n = int(A.shape[1])
    gmax = max(Px, Py)
    if n % gmax != 0:
        raise ValueError(f"N={n} must be divisible by max(Px,Py)={gmax}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    g = n // gmax                      # square block (MB_A == NB_A)

    key = ("scalapack_eigh", _mesh_key(mesh), A.dtype, nq, n, g,
           int(ctx_handle))
    jit_eigh = _JIT_CACHE.get(key)
    if jit_eigh is None:
        W_out = jax.ShapeDtypeStruct((nq, n), jnp.float64)
        # Z comes back as ScaLAPACK col-major local blocks = row-major
        # bytes of the per-rank block of Zᵀ; assembled at P(None,'y','x')
        # then locally untransposed (same trick as ffi.slate.eigh).
        Z_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        attrs = dict(nq=nq, n=n, g=g, ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=P(None, "x", "y"),
                 out_specs=(P(), P(None, "y", "x")),
                 check_rep=False)
        def _call(local_A):
            # Inner transpose: row-major shard → col-major local block
            # (same convention as every slate/scalapack wrapper).
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            return jax.ffi.ffi_call(
                _EIGH_TARGET, (W_out, Z_local_T))(local_A_T, **attrs)

        @partial(shard_map, mesh=mesh,
                 in_specs=P(None, "y", "x"), out_specs=P(None, "x", "y"),
                 check_rep=False)
        def _untranspose(local_Z_T):
            return jnp.transpose(local_Z_T, (0, 2, 1))

        def _eigh(A_):
            W, Z_T = _call(A_)
            return W, _untranspose(Z_T)

        jit_eigh = jax.jit(_eigh)
        _JIT_CACHE[key] = jit_eigh

    return jit_eigh(A)


def distributed_eigh(A: jax.Array, *, mesh: Mesh) -> Tuple[jax.Array, jax.Array]:
    """Single-matrix form of :func:`batched_distributed_eigh`.

    ``A`` is ``(N, N)`` at ``P('x','y')``; returns ``(W, Z)`` with ``W``
    ``(N,)`` replicated and ``Z`` ``(N, N)`` at ``P('x','y')``.  This is the
    shape ``ffi.linalg.LinalgPlan.__call__`` uses (one tile spread
    over the mesh); the batched entry point is what the ζ-fit uses.
    """
    if A.ndim != 2:
        raise ValueError(
            f"scalapack distributed_eigh: expected a square matrix, "
            f"got {A.shape}")
    W, Z = batched_distributed_eigh(A[None], mesh=mesh)
    return W[0], Z[0]

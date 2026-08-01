"""``batched_distributed_solve_lu`` — per-q ScaLAPACK ``pXgetrf`` + ``pXgetrs``.

Host-platform (JAX CPU backend) twin of
:func:`ffi.cusolvermp.batched_distributed_solve_lu` — the portable backend
for the ``distributed_lu`` axis.  Identical call contract:

    A : (Nq, N, N)     P(None, 'x', 'y')   (donated — factored in place)
    B : (Nq, N, NRHS)  P(None, 'x', 'y')
    X : same shape/sharding as B

ScaLAPACK comes from whichever implementation of the ScaLAPACK API the
host library was linked against — MKL on Frontera, Cray LibSci on
Perlmutter, and the source is identical either way (see
``ffi.scalapack``).  No extra dependency.  The BLACS grid is built on
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

__all__ = ["batched_distributed_solve_lu",
           "batched_distributed_getrf",
           "batched_distributed_getrs"]

_SOLVE_LU_TARGET = "lorrax_scalapack_batched_solve_lu"
_GETRF_TARGET = "lorrax_scalapack_batched_getrf"
_GETRS_TARGET = "lorrax_scalapack_batched_getrs"

# jit(shard_map) cache per signature — see ffi/phdf5/ARCHITECTURE.md §2.4.
_JIT_CACHE: dict = {}


def _validate_lu_geometry(A, mesh: Mesh, *, what: str):
    """Shared guard ladder for the LU family (fused solve, getrf, getrs):
    host platform, registered square/1-D mesh, divisibility.  Returns
    ``(Px, Py, nq, n, g)``."""
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"scalapack {what}: expected A of shape (Nq, N, N); got {A.shape}")
    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise ValueError(
            f"scalapack {what} is host-only but the mesh devices are "
            f"{plat!r}; use distributed_lu = cusolvermp on GPU meshes.")
    Px, Py = validate_mesh(mesh)
    if Px > 1 and Py > 1 and Px != Py:
        raise ValueError(
            f"scalapack {what}: mesh {Px}x{Py} unsupported — pXgetrf "
            f"needs square blocks (MB == NB), which the one-tile-per-rank "
            f"layout only gives on square or 1-D meshes.")
    nq = int(A.shape[0])
    n = int(A.shape[1])
    gmax = max(Px, Py)
    if n % gmax != 0:
        raise ValueError(f"N={n} must be divisible by max(Px,Py)={gmax}.")
    return Px, Py, nq, n, n // gmax


def _ipiv_local_len(n: int, Px: int, g: int) -> int:
    """Per-rank ipiv extent: ``LOCr(M_A) + MB_A`` per the ScaLAPACK spec.
    On the square / 1-D meshes the wrapper allows, ``LOCr = n / Px``."""
    return n // max(Px, 1) + g


def batched_distributed_getrf(
    A: jax.Array,
    *,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """Factor ``A[q]`` via per-q ScaLAPACK ``pXgetrf``, ONCE — the hoisted
    transverse ζ factor stage.

    Same sharding contract as :func:`batched_distributed_solve_lu`'s A
    (``(Nq, N, N)`` at ``P(None,'x','y')``, donated — factored in place),
    but the factors are RETURNED instead of consumed:

    Returns
    -------
    LU : (Nq, N, N) at ``P(None, 'x', 'y')``
        The block-cyclic L/U factors; each rank's shard is its local
        block, exactly as ``pXgetrf`` left it.  Feed back VERBATIM into
        :func:`batched_distributed_getrs` — never reshard it (the values
        only mean anything in this layout on this grid).
    ipiv : (Nq, P·ipiv_len) int32 at ``P(None, ('x','y'))``
        Each rank's own ScaLAPACK ipiv rows (``ipiv_len = LOCr + MB``);
        opaque, never gathered, never interpreted host-side.

    ``pXgetrf`` on a given matrix is bit-identical whether or not the
    ``pXgetrs`` follows immediately (same descriptors, same grid), so
    getrf-once + getrs-per-r-chunk reproduces the fused
    ``batched_distributed_solve_lu`` to the bit — gated in
    ``tests/test_transverse_factor_hoist.py`` (srun leg).
    """
    Px, Py, nq, n, g = _validate_lu_geometry(A, mesh, what="getrf")
    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)
    ipiv_len = _ipiv_local_len(n, Px, g)

    key = ("scalapack_getrf", _mesh_key(mesh), A.dtype,
           nq, n, g, int(ctx_handle))
    jit_getrf = _JIT_CACHE.get(key)
    if jit_getrf is None:
        LU_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        ipiv_local = jax.ShapeDtypeStruct((nq, ipiv_len), jnp.int32)
        attrs = dict(nq=nq, n=n, g=g, ipiv_len=ipiv_len,
                     ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"),),
                 out_specs=(P(None, "x", "y"), P(None, ("x", "y"))),
                 check_rep=False)
        def _getrf(local_A):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            LU_T, ipiv = jax.ffi.ffi_call(
                _GETRF_TARGET, (LU_local_T, ipiv_local),
                input_output_aliases={0: 0},
            )(local_A_T, **attrs)
            return jnp.transpose(LU_T, (0, 2, 1)), ipiv

        jit_getrf = jax.jit(_getrf, donate_argnums=(0,))
        _JIT_CACHE[key] = jit_getrf

    return jit_getrf(A)


def batched_distributed_getrs(
    LU: jax.Array,
    ipiv: jax.Array,
    B: jax.Array,
    *,
    mesh: Mesh,
) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` from the factors of
    :func:`batched_distributed_getrf` via per-q ScaLAPACK ``pXgetrs`` —
    the per-r-chunk half of the hoisted transverse factor stage.

    ``LU``/``ipiv`` must arrive EXACTLY as getrf returned them (same
    mesh, same shardings); ``B`` is ``(Nq, N, NRHS)`` at
    ``P(None,'x','y')`` with ``NRHS % Py == 0`` (donated — solved in
    place).  Returns X in B's shape/sharding.
    """
    Px, Py, nq, n, g = _validate_lu_geometry(LU, mesh, what="getrs")
    if (B.ndim != 3 or int(B.shape[0]) != nq or int(B.shape[1]) != n):
        raise ValueError(
            f"scalapack getrs: B must be (Nq, N, NRHS) matching LU; "
            f"got LU={LU.shape}, B={B.shape}")
    if LU.dtype != B.dtype:
        raise ValueError(f"LU.dtype {LU.dtype} != B.dtype {B.dtype}")
    ipiv_len = _ipiv_local_len(n, Px, g)
    P_tot = Px * Py
    if (ipiv.ndim != 2 or int(ipiv.shape[0]) != nq
            or int(ipiv.shape[1]) != P_tot * ipiv_len
            or ipiv.dtype != jnp.int32):
        raise ValueError(
            f"scalapack getrs: ipiv must be (Nq, P*ipiv_len) int32 = "
            f"({nq}, {P_tot * ipiv_len}) as returned by getrf; got "
            f"{ipiv.shape} {ipiv.dtype}")
    nrhs = int(B.shape[2])
    if nrhs % Py != 0:
        raise ValueError(f"NRHS={nrhs} must be divisible by Py={Py}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)
    nb_b = nrhs // Py if Py > 1 else nrhs

    key = ("scalapack_getrs", _mesh_key(mesh), LU.dtype,
           nq, n, nrhs, g, nb_b, int(ctx_handle))
    jit_getrs = _JIT_CACHE.get(key)
    if jit_getrs is None:
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // Py, n // Px), B.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs, g=g, nb_b=nb_b,
                     ipiv_len=ipiv_len, ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, ("x", "y")),
                           P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_rep=False)
        def _getrs(local_LU, local_ipiv, local_B):
            local_LU_T = jnp.transpose(local_LU, (0, 2, 1))
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            X_T = jax.ffi.ffi_call(
                _GETRS_TARGET, X_local_T,
                input_output_aliases={2: 0},
            )(local_LU_T, local_ipiv, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_getrs = jax.jit(_getrs, donate_argnums=(2,))
        _JIT_CACHE[key] = jit_getrs

    return jit_getrs(LU, ipiv, B)


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

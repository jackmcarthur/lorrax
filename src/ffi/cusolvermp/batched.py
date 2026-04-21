"""``batched_distributed_{cholesky,potrs}`` — cuSOLVERMp batched solve.

Use case: a batch of ``Nbatch`` independent Hermitian PD matrices
``A_q`` of shape ``(N, N)``, each one too big for one GPU (sharded
across a 1-axis sub-group of ``Py`` ranks), and a batch of right-hand
sides ``B_q`` of shape ``(N, Mrhs)``. Want ``A_q X_q = B_q`` for all
``q`` in parallel across the mesh.

Sharding contract
-----------------
Inputs / outputs have shape ``(Nbatch, N, N)`` (cholesky) or
``(Nbatch, N, Mrhs)`` (RHS), sharded ``P('x', None, 'y')`` on a 2-D
``('x', 'y')`` mesh of shape ``(Px, Py)``. Batch is split along
``'x'``; each ``(N, N)`` and ``(N, Mrhs)`` is column-sharded across
``'y'``. Each X-row of the mesh is an independent NCCL sub-comm of
size ``Py``; cuSOLVERMp runs on a grid ``(1, Py)`` inside.

cuSOLVERMp has no native batched potrf / potrs — the "batching" is a
C++ for-loop over the per-rank batch dimension in the FFI handler.
Grid, handle, and matrix descriptors are built once per FFI call and
reused across the loop; only the slice pointer changes per iteration.

For the slate twin with the same shape contract (AMD-GPU fallback
path) see ``ffi.slate.batched``.

Early-init tip
--------------
First call to either function triggers a lazy ``get_or_init_subrow_context``
that collectively creates an NCCL sub-comm + cuSOLVERMp handle + grid
(~hundreds of ms).  To hoist that off the critical path of the first
batched op, call it eagerly in ``main()`` right after
``jax.distributed.initialize()``::

    from ffi.cusolvermp.context import get_or_init_subrow_context
    get_or_init_subrow_context(mesh)     # warm the sub-row context
    # ... other JAX compile work can now overlap with the NCCL bootstrap

Restrictions
------------
* Mesh must be 2-D with axes ``('x', 'y')``.
* ``Nbatch % Px == 0``; ``N % Py == 0``; ``Mrhs % Py == 0``.
* Dtypes: F64 / C128.
* One cuSOLVERMp handle per process → don't mix the world-wide
  ``distributed_eigh`` context and this sub-row context in the same
  session.
* ``Mrhs`` is baked into the compiled JIT.  Users who loop with the
  *same* chunk size pay zero recompile cost (jit-cache in this module).
  Users who genuinely vary the chunk size per call would want an FFI
  variant with ``mrhs`` as a runtime ``Buffer<S64>`` arg (phdf5 §2.3
  pattern); not implemented today since GWJAX-style callers pick one
  chunk size and stick with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_subrow_context

__all__ = [
    "CusolverMpBatchedLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_potrs",
]

_POTRF_TARGET = "lorrax_cusolvermp_batched_potrf"
_POTRS_TARGET = "lorrax_cusolvermp_batched_potrs"

# ---------------------------------------------------------------------------
# jit(shard_map(...)) cache.  `shard_map` in eager mode re-traces on every
# invocation (see src/ffi/phdf5/ARCHITECTURE.md §2.4 for the measurement on
# the phdf5 write path).  Wrapping once in `jax.jit` and caching the
# compiled function by signature eliminates the per-call retrace cost —
# matters when the user loops potrs over many RHS chunks with the same
# shape.  Key covers everything that affects the compiled HLO.
# ---------------------------------------------------------------------------
_JIT_CACHE: dict = {}

def _mesh_key(mesh: Mesh):
    return (tuple(mesh.axis_names), tuple(int(s) for s in mesh.shape.values()))


@dataclass(frozen=True)
class CusolverMpBatchedLowerL:
    """Batched-cholesky output handle.

    Attributes
    ----------
    raw : jax.Array
        Shape ``(Nbatch, N, N)`` sharded ``P('x', 'y', None)``. Bytes
        hold cuSOLVERMp's col-major L tiles per slice. JAX reading
        ``raw[q]`` directly sees ``L.T`` (or ``L.conj().T`` for complex);
        the ``batched_distributed_potrs`` handler knows the layout and
        feeds it straight back to cuSOLVERMp.
    mesh : Mesh
    n : int
        Per-slice side length.
    nb : int
        cuSOLVERMp tile block size along the column axis.
    nbatch : int
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    nb: int
    nbatch: int


def _validate_mesh(mesh: Mesh):
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(
            f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    Px = int(mesh.shape["x"])
    Py = int(mesh.shape["y"])
    if Px * Py != jax.process_count():
        raise ValueError(
            f"mesh {Px}x{Py} != jax.process_count() = {jax.process_count()}")
    return Px, Py


def batched_distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
    block_size: Optional[int] = None,
) -> CusolverMpBatchedLowerL:
    """Factor each ``A[q]`` = ``L[q] L[q]^H`` via ``cusolverMpPotrf``.

    Parameters
    ----------
    A : jax.Array
        Shape ``(Nbatch, N, N)``, dtype F64 or C128, sharded
        ``P('x', None, 'y')``.
    mesh : Mesh
    block_size : int, optional
        cuSOLVERMp tile block size. Default ``N // Py``.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"batched_distributed_cholesky: expected (Nbatch, N, N); "
            f"got {A.shape}")
    Px, Py = _validate_mesh(mesh)
    nbatch = int(A.shape[0])
    n      = int(A.shape[1])
    if nbatch % Px != 0:
        raise ValueError(
            f"Nbatch={nbatch} must be divisible by mesh 'x' axis size {Px}.")
    if n % Py != 0:
        raise ValueError(
            f"N={n} must be divisible by mesh 'y' axis size {Py}.")

    get_lib()
    ctx_handle = get_or_init_subrow_context(mesh)

    nb_batch_local = nbatch // Px
    # cuSOLVERMp expects square tiles (mb == nb).  Default: one tile per
    # rank in the column direction (nb = N/Py).  With grid (1, Py) and
    # mb == nb, each rank holds N/nb row-tiles stacked in a single
    # (N × N/Py) col-major strip.
    nb = (n // Py) if block_size is None else int(block_size)
    mb = nb

    key = ("potrf", _mesh_key(mesh), A.dtype,
           nbatch, n, mb, nb, int(ctx_handle))
    jit_potrf = _JIT_CACHE.get(key)
    if jit_potrf is None:
        # Local inner-dim transpose: (Nb_local, N, N/Py) row-major →
        # (Nb_local, N/Py, N) row-major ≡ (N, N/Py) col-major per slice,
        # which is what cuSOLVERMp's grid (1, Py) expects with lld=N.
        L_local_T = jax.ShapeDtypeStruct(
            (nb_batch_local, n // Py, n), A.dtype)
        attrs = dict(
            nbatch_local=nb_batch_local, n=n, mb=mb, nb=nb,
            ctx_handle=int(ctx_handle),
        )

        @partial(shard_map, mesh=mesh,
                 in_specs=P("x", None, "y"),
                 out_specs=P("x", "y", None),
                 check_rep=False)
        def _potrf(local_A):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            return jax.ffi.ffi_call(_POTRF_TARGET, L_local_T)(local_A_T, **attrs)

        jit_potrf = jax.jit(_potrf)
        _JIT_CACHE[key] = jit_potrf

    L_raw_T = jit_potrf(A)
    return CusolverMpBatchedLowerL(
        raw=L_raw_T, mesh=mesh, n=n, nb=nb, nbatch=nbatch)


def batched_distributed_potrs(
    L: CusolverMpBatchedLowerL,
    B: jax.Array,
    *,
    mesh: Mesh,
    block_size: Optional[int] = None,
) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` using ``L[q]`` from ``batched_distributed_cholesky``.

    One cuSOLVERMp ``potrs`` call per slice; equivalent to applying
    ``L^{-1}`` then ``L^{-H}`` in sequence.

    Parameters
    ----------
    L : CusolverMpBatchedLowerL
        Output of ``batched_distributed_cholesky``.
    B : jax.Array
        Shape ``(Nbatch, N, Mrhs)`` sharded ``P('x', None, 'y')``, same
        dtype as the factored L.
    mesh : Mesh
        Must match ``L.mesh`` axis names.

    Returns
    -------
    X : jax.Array
        Same shape / sharding / dtype as ``B``.
    """
    Px, Py = _validate_mesh(mesh)
    if mesh.axis_names != L.mesh.axis_names:
        raise ValueError("potrs mesh axis names don't match the handle's mesh.")
    if B.ndim != 3:
        raise ValueError(
            f"batched_distributed_potrs: expected 3D B; got {B.shape}")
    nbatch = L.nbatch
    n      = L.n
    if int(B.shape[0]) != nbatch:
        raise ValueError(
            f"batched_distributed_potrs: B.shape[0]={B.shape[0]} != "
            f"Nbatch={nbatch}")
    if int(B.shape[1]) != n:
        raise ValueError(
            f"batched_distributed_potrs: B.shape[1]={B.shape[1]} != N={n}")
    mrhs = int(B.shape[2])
    if mrhs % Py != 0:
        raise ValueError(
            f"Mrhs={mrhs} must be divisible by mesh 'y' axis size {Py}.")
    if L.raw.dtype != B.dtype:
        raise ValueError(
            f"L.dtype {L.raw.dtype} != B.dtype {B.dtype}")

    get_lib()
    ctx_handle = get_or_init_subrow_context(mesh)

    nb = L.nb if block_size is None else int(block_size)
    mb = nb

    nb_batch_local = nbatch // Px
    key = ("potrs", _mesh_key(mesh), B.dtype,
           nbatch, n, mrhs, mb, nb, int(ctx_handle))
    jit_potrs = _JIT_CACHE.get(key)
    if jit_potrs is None:
        X_local_T = jax.ShapeDtypeStruct(
            (nb_batch_local, mrhs // Py, n), B.dtype)
        attrs = dict(
            nbatch_local=nb_batch_local,
            n=n, mrhs=mrhs, mb=mb, nb=nb,
            ctx_handle=int(ctx_handle),
        )

        @partial(shard_map, mesh=mesh,
                 in_specs=(P("x", "y", None), P("x", None, "y")),
                 out_specs=P("x", None, "y"),
                 check_rep=False)
        def _potrs(local_L, local_B):
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            X_T = jax.ffi.ffi_call(_POTRS_TARGET, X_local_T)(
                local_L, local_B_T, **attrs)
            # Untranspose inside the shard_map → output has P('x',None,'y')
            # directly, saving a second shard_map pass.
            return jnp.transpose(X_T, (0, 2, 1))

        jit_potrs = jax.jit(_potrs)
        _JIT_CACHE[key] = jit_potrs

    return jit_potrs(L.raw, B)

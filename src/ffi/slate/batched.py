"""``batched_distributed_{cholesky,trsm}`` — 3-D batched SLATE wrappers.

Use case: a batch of ``Nbatch`` independent ``N × N`` Hermitian PD
matrices, where each one is too big for a single GPU so must be 2-D
sharded, and multiple matrices should run in parallel across the mesh.

Sharding contract
-----------------
Inputs / outputs have shape ``(Nbatch, N, N)`` (cholesky) or
``(Nbatch, N, Nrhs)`` (trsm RHS), sharded ``P('x', None, 'y')`` on a
2-D ``('x', 'y')`` mesh of shape ``(Px, Py)``.  The batch is split along
``'x'`` and the per-matrix distribution happens along ``'y'``.  Each
X-row of the mesh gets its own MPI sub-communicator of size ``Py``, and
all ranks in a row cooperatively process the ``Nbatch/Px`` matrices
local to that row.

SLATE itself has no native batched potrf/trsm, so the C++ handler is
literally a ``for`` loop over the per-rank batch dimension — no
communication is shared across batch iterations.  This beats
``lax.scan``-ing the single-matrix primitive because the sub-comm and
device context are set up once and reused.

For the cuSOLVERMp twin (written against a real batched kernel) see
``ffi.cusolvermp.batched_*`` (forthcoming).

Restrictions
------------
* Mesh must be 2-D with axes ``('x', 'y')``.
* ``Nbatch`` must be divisible by ``Px``; ``N`` (and ``Nrhs``) divisible
  by ``Py``.
* Dtypes: F64 / C128.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from .context import get_or_init_subrow_context, validate_mesh

__all__ = [
    "SlateBatchedLowerL",
    "batched_distributed_cholesky",
    "batched_distributed_trsm",
]

_POTRF_TARGET = "lorrax_slate_batched_potrf"
_TRSM_TARGET  = "lorrax_slate_batched_trsm"

_SIDE  = {"L": 0, "R": 1}
_UPLO  = {"L": 0, "U": 1}
_OP    = {"N": 0, "T": 1, "C": 2}
_DIAG  = {"N": 0, "U": 1}

# jit(shard_map(...)) cache per signature.  shard_map in eager mode
# re-traces per call; wrapping in jax.jit once and reusing eliminates
# that cost when the user loops the same-shape op (e.g. batched_trsm
# across many RHS chunks).  See src/ffi/phdf5/ARCHITECTURE.md §2.4.
_JIT_CACHE: dict = {}

def _mesh_key(mesh: Mesh):
    return (tuple(mesh.axis_names), tuple(int(s) for s in mesh.shape.values()))


@dataclass(frozen=True)
class SlateBatchedLowerL:
    """Batched-cholesky output handle.

    Attributes
    ----------
    raw : jax.Array
        Shape ``(Nbatch, N, N)``, sharded ``P('x', 'y', None)``.  Bytes
        hold SLATE's col-major L tiles per slice; JAX reading the inner
        ``(N, N)`` directly would give ``L.T`` (or ``L.conj().T``).
    mesh : Mesh
    n : int
        Per-slice side length.
    nb : int
    nbatch : int
    """
    raw: jax.Array
    mesh: Mesh
    n: int
    nb: int
    nbatch: int


def batched_distributed_cholesky(
    A: jax.Array,
    *,
    mesh: Mesh,
    block_size: Optional[int] = None,
) -> SlateBatchedLowerL:
    """Factor each ``A[q]`` = ``L[q] L[q]^H`` via ``slate::potrf``.

    Parameters
    ----------
    A : jax.Array
        Shape ``(Nbatch, N, N)``, dtype F64 or C128, sharded
        ``P('x', None, 'y')``.
    mesh : Mesh
    block_size : int, optional
        Per-matrix SLATE tile size.  Default ``N // Py``.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"batched_distributed_cholesky: expected (Nbatch, N, N); "
            f"got {A.shape}")
    Px, Py = validate_mesh(mesh)
    nbatch = int(A.shape[0])
    n      = int(A.shape[1])
    if nbatch % Px != 0:
        raise ValueError(
            f"batched_distributed_cholesky: Nbatch={nbatch} must be "
            f"divisible by mesh 'x' axis size {Px}.")
    if n % Py != 0:
        raise ValueError(
            f"batched_distributed_cholesky: N={n} must be divisible by "
            f"mesh 'y' axis size {Py}.")

    get_lib()
    ctx_handle = get_or_init_subrow_context(mesh)

    nb_batch_local = nbatch // Px
    nb = n // Py if block_size is None else int(block_size)

    key = ("potrf", _mesh_key(mesh), A.dtype,
           nbatch, n, nb, int(ctx_handle))
    jit_potrf = _JIT_CACHE.get(key)
    if jit_potrf is None:
        # Local transpose of the inner two dims: (Nb_local, N, N/Py) row-major
        # → (Nb_local, N/Py, N) row-major, which is (N, N/Py) col-major per
        # slice — the layout SLATE's fromDevices expects with p=1, q=Py.
        L_local_T = jax.ShapeDtypeStruct(
            (nb_batch_local, n // Py, n), A.dtype)
        attrs = dict(
            nbatch_local=nb_batch_local, n=n, nb=nb,
            ctx_handle=int(ctx_handle),
        )

        @partial(shard_map, mesh=mesh,
                 in_specs=P("x", None, "y"),
                 out_specs=P("x", "y", None),
                 check_rep=False)
        def _potrf(local_A):
            # local_A shape: (Nb_local, N, N/Py)
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            return jax.ffi.ffi_call(_POTRF_TARGET, L_local_T)(local_A_T, **attrs)

        jit_potrf = jax.jit(_potrf)
        _JIT_CACHE[key] = jit_potrf

    L_raw_T = jit_potrf(A)
    return SlateBatchedLowerL(
        raw=L_raw_T, mesh=mesh, n=n, nb=nb, nbatch=nbatch)


def batched_distributed_trsm(
    A,
    B: jax.Array,
    *,
    mesh: Mesh,
    side: Literal["L", "R"] = "L",
    uplo: Literal["L", "U"] = "L",
    op: Literal["N", "T", "C"] = "N",
    diag: Literal["N", "U"] = "N",
    alpha: complex | float = 1.0,
    block_size: Optional[int] = None,
) -> jax.Array:
    """Batched triangular solve: one ``slate::trsm`` per batch slice.

    ``A`` is a :class:`SlateBatchedLowerL` handle (preferred) or a plain
    batched ``(Nbatch, N, N)`` array sharded ``P('x', None, 'y')``.  For
    the handle path, ``side/uplo`` are pinned to ``'L'/'L'`` and
    ``op='N'`` / ``'C'`` selects forward / adjoint solve.

    ``B`` has shape ``(Nbatch, N, Nrhs)`` sharded ``P('x', None, 'y')``.
    Output has the same shape / sharding as ``B``.
    """
    Px, Py = validate_mesh(mesh)

    if isinstance(A, SlateBatchedLowerL):
        if op not in ("N", "C"):
            raise ValueError(
                f"batched_distributed_trsm(SlateBatchedLowerL, ...): "
                f"op must be 'N' or 'C'; got {op!r}")
        side, uplo = "L", "L"
        nbatch = A.nbatch
        n = A.n
        if block_size is None:
            block_size = A.nb
        if mesh.axis_names != A.mesh.axis_names:
            raise ValueError(
                "batched trsm mesh axis names don't match the handle's mesh.")
        A_is_handle = True
        A_arg = A.raw
    else:
        if A.ndim != 3 or A.shape[1] != A.shape[2]:
            raise ValueError(
                f"batched_distributed_trsm: expected (Nbatch, N, N) A; "
                f"got {A.shape}")
        nbatch = int(A.shape[0])
        n      = int(A.shape[1])
        A_is_handle = False
        A_arg = A

    if B.ndim != 3:
        raise ValueError(
            f"batched_distributed_trsm: expected 3D B; got {B.shape}")
    if int(B.shape[0]) != nbatch:
        raise ValueError(
            f"batched_distributed_trsm: B.shape[0]={B.shape[0]} != "
            f"Nbatch={nbatch}")
    if A_arg.dtype != B.dtype:
        raise ValueError(
            f"batched_distributed_trsm: A.dtype {A_arg.dtype} != "
            f"B.dtype {B.dtype}")

    if side == "L":
        if int(B.shape[1]) != n:
            raise ValueError(
                f"side='L' requires B.shape[1]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[2])
    else:
        if int(B.shape[2]) != n:
            raise ValueError(
                f"side='R' requires B.shape[2]==n={n}; got B.shape={B.shape}")
        m = int(B.shape[1])

    if nbatch % Px != 0:
        raise ValueError(
            f"Nbatch={nbatch} must be divisible by mesh 'x' axis {Px}.")
    if n % Py != 0 or m % Py != 0:
        raise ValueError(
            f"N={n}, M={m} must be divisible by mesh 'y' axis {Py}.")

    get_lib()
    ctx_handle = get_or_init_subrow_context(mesh)
    nbatch_local = nbatch // Px
    nb = n // Py if block_size is None else int(block_size)

    alpha_c = complex(alpha)
    key = ("trsm", _mesh_key(mesh), B.dtype, A_is_handle,
           nbatch, n, m, nb,
           _SIDE[side], _UPLO[uplo], _OP[op], _DIAG[diag],
           float(alpha_c.real), float(alpha_c.imag),
           int(ctx_handle))
    jit_trsm = _JIT_CACHE.get(key)
    if jit_trsm is None:
        # Expected per-slice: (B.shape[1], B.shape[2]/Py) row-major → transpose
        # inner two dims → (B.shape[2]/Py, B.shape[1]) row-major ≡ col-major.
        X_local_T_shape = (nbatch_local, B.shape[2] // Py, B.shape[1])
        X_local_T = jax.ShapeDtypeStruct(X_local_T_shape, B.dtype)

        attrs = dict(
            nbatch_local=nbatch_local,
            n=n, m=m, nb=nb,
            side=_SIDE[side], uplo=_UPLO[uplo],
            op=_OP[op], diag=_DIAG[diag],
            alpha_re=float(alpha_c.real),
            alpha_im=float(alpha_c.imag),
            ctx_handle=int(ctx_handle),
        )

        if A_is_handle:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P("x", "y", None), P("x", None, "y")),
                     out_specs=P("x", None, "y"),
                     check_rep=False)
            def _trsm(local_A_handle, local_B):
                local_B_T = jnp.transpose(local_B, (0, 2, 1))
                X_T = jax.ffi.ffi_call(_TRSM_TARGET, X_local_T)(
                    local_A_handle, local_B_T, **attrs)
                # Untranspose inside the shard_map → skip a second pass.
                return jnp.transpose(X_T, (0, 2, 1))
        else:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P("x", None, "y"), P("x", None, "y")),
                     out_specs=P("x", None, "y"),
                     check_rep=False)
            def _trsm(local_A, local_B):
                local_A_T = jnp.transpose(local_A, (0, 2, 1))
                local_B_T = jnp.transpose(local_B, (0, 2, 1))
                X_T = jax.ffi.ffi_call(_TRSM_TARGET, X_local_T)(
                    local_A_T, local_B_T, **attrs)
                return jnp.transpose(X_T, (0, 2, 1))

        jit_trsm = jax.jit(_trsm)
        _JIT_CACHE[key] = jit_trsm

    return jit_trsm(A_arg, B)

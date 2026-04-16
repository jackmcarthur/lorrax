"""``eigh_mg`` — JAX FFI wrapper around cusolverMgSyevd.

Single-process, multi-GPU Hermitian/symmetric eigensolve.  This is the
simpler counterpart to ``ffi.cusolvermp.distributed_eigh``:
- No NCCL / MPI / multi-process bootstrap.
- Runs from one Python process that sees all GPUs as local devices.
- cuSOLVERMg handles the column-tile distribution across the GPUs
  internally (enables peer access, scatters, solves, gathers).

The FFI handler takes the full (n, n) matrix on device 0 and returns
eigenvalues + eigenvectors on device 0.  The "4-GPU distribution" happens
inside ``cusolverMgSyevd``.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from ..common.ffi_loader import get_lib

__all__ = ["eigh_mg"]


def eigh_mg(
    A: jax.Array,
    *,
    tile_size: int = 32,
    max_gpus: int = 0,
    compute_evecs: bool = True,
) -> Tuple[jax.Array, jax.Array]:
    """Multi-GPU Hermitian/symmetric eigendecomposition via cuSOLVERMg.

    Parameters
    ----------
    A : jax.Array, shape (n, n), dtype float64
        Real symmetric matrix.  Must be on a GPU (XLA will place it on
        device 0 inside the FFI).  Complex (Hermitian) is not implemented
        in this first iteration; use ``ffi.cusolvermp.distributed_eigh``
        once that path is unblocked.
    tile_size : int
        Column tile size for block-cyclic distribution.  cuSOLVERMg's
        default is 256 for large matrices; use 32 for small test cases
        so the work spreads across all GPUs.
    max_gpus : int
        If positive, cap the number of GPUs used to this value.  Default
        0 = use all visible GPUs.
    compute_evecs : bool
        If False, return only eigenvalues (Q is populated but its contents
        should be ignored).

    Returns
    -------
    evals : (n,) float64, on the same device as A.  Ascending order.
    Q : (n, n) float64, on the same device as A.  Eigenvectors of A^T
        (column-major view of the row-major input).  For real-symmetric A
        these equal the eigenvectors of A by transposition.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"eigh_mg expects a square matrix; got {A.shape}")
    n = int(A.shape[0])
    if jnp.dtype(A.dtype) != jnp.dtype(jnp.float64):
        raise TypeError(
            f"eigh_mg supports float64 only for now; got {A.dtype}.  "
            f"Complex Hermitian is not yet implemented in the Mg path.")

    # Ensure the library is loaded and FFI targets registered.
    get_lib()

    eval_type = jax.ShapeDtypeStruct((n,), jnp.float64)
    Q_type    = jax.ShapeDtypeStruct((n, n), jnp.float64)

    def _call(A_):
        return jax.ffi.ffi_call(
            "lorrax_cusolvermg_eigh_f64",
            (eval_type, Q_type),
        )(A_,
          n=int(n),
          tile_size=int(tile_size),
          max_gpus=int(max_gpus),
          compute_evecs=bool(compute_evecs))

    evals, Q = jax.jit(_call)(A)
    return evals, Q

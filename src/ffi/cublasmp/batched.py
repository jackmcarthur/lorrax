"""The remaining LORRAX-private cuBLASMp fused W-solve wrapper.

Distributed GEMM moved to the standalone :func:`distrib_la.matmul` door.
``ffi.cublasmp`` re-exports that function under its historical name only for
source compatibility; this module no longer owns a second implementation.
"""
from __future__ import annotations

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from common.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from ..cusolvermp.context import get_or_init_context

__all__ = ["batched_fused_w_solve", "batched_fused_w_solve_jit"]

_W_SOLVE_TARGET = "lorrax_cublasmp_batched_w_solve"

_JIT_CACHE: dict = {}


def _mesh_key(mesh: Mesh):
    return (tuple(mesh.axis_names), tuple(int(s) for s in mesh.shape.values()))


def _validate_mesh(mesh: Mesh):
    if "x" not in mesh.axis_names or "y" not in mesh.axis_names:
        raise ValueError(f"mesh must have axes ('x','y'); got {mesh.axis_names}")
    Px = int(mesh.shape["x"])
    Py = int(mesh.shape["y"])
    if Px * Py != jax.process_count():
        raise ValueError(
            f"mesh {Px}x{Py} != jax.process_count() = {jax.process_count()}")
    return Px, Py


def batched_fused_w_solve(
    V: jax.Array,
    chi: jax.Array,
    pref: Union[float, complex],
    *,
    mesh: Mesh,
    stop_after_step: int = 0,
) -> jax.Array:
    """Fused distributed W-solve:  W = X (I − X^† pref·χ X)^{-1} X^†.

    Runs the entire symmetric Cholesky formulation of the Dyson solve in
    a single FFI: potrf(V) → 2 cuBLASMp gemms → identity−T kernel →
    potrf(H) → 2 cuBLASMp trsms → cuBLASMp gemm.  No JAX-level
    intermediates, no opportunity for XLA to reshard or rematerialize.

    Inputs
    ------
    V : (Nq, N, N)  P(None,'x','y')  — Hermitian PD Coulomb, DONATED (XLA
        reuses its buffer for the Cholesky factor X).
    chi : (Nq, N, N) P(None,'x','y') — Hermitian, typically χ ≼ 0 so
        H = I − X^† pref·χ X is Hermitian PD (Cholesky-factorisable).
    pref : complex scalar — multiplier on χ inside the solve.

    Output
    ------
    W : (Nq, N, N) P(None,'x','y') — same sharding as V/χ.
    """
    if V.ndim != 3 or V.shape[1] != V.shape[2]:
        raise ValueError(f"V must be (Nq, N, N); got {V.shape}")
    if chi.shape != V.shape or chi.dtype != V.dtype:
        raise ValueError(
            f"chi must match V in shape/dtype; got V={V.shape}/{V.dtype} "
            f"chi={chi.shape}/{chi.dtype}")
    jit_fn = batched_fused_w_solve_jit(
        dtype=V.dtype, nq=int(V.shape[0]), n=int(V.shape[1]),
        pref=pref, mesh=mesh, stop_after_step=stop_after_step)
    return jit_fn(V, chi)


def batched_fused_w_solve_jit(
    *,
    dtype,
    nq: int,
    n: int,
    pref: Union[float, complex],
    mesh: Mesh,
    stop_after_step: int = 0,
):
    """Return the cached ``jax.jit``-wrapped W-solve for the given
    (dtype, nq, n, pref, mesh, stop_after_step).  Builds it on cache
    miss.  Exposed so callers can AOT ``lower(V, chi).compile()`` the
    jit independently of invoking it — used by
    ``precompile_solve_w`` to split compile time from exec time in
    the end-of-run timing report.
    """
    Px, Py = _validate_mesh(mesh)
    if n % Px != 0 or n % Py != 0:
        raise ValueError(
            f"N={n} must be divisible by both Px={Px} and Py={Py}.")
    if Px != Py:
        # The fused solve runs cusolverMpPotrf internally, which needs
        # square ScaLAPACK blocks (mb == nb) — impossible with the
        # one-tile-per-rank layout on a non-square mesh; and cuBLASMp's
        # Matmul_bufferSize likewise rejects 1-D grids (status=3,
        # verified 4x1/1x4 2026-07-10).  1x1 and square meshes work.
        raise ValueError(
            f"batched_fused_w_solve: mesh {Px}x{Py} is not square — the "
            f"embedded cusolverMpPotrf / cublasMpMatmul require square "
            f"block-cyclic tiles.  Use a square mesh or the JAX_NATIVE "
            f"screening solver.")

    get_lib()
    ctx_handle = get_or_init_context(mesh, col_major=False)

    pref_c = complex(pref)
    stop_after_step = int(stop_after_step)
    key = ("w_solve", _mesh_key(mesh), dtype, nq, n,
           pref_c, int(ctx_handle), stop_after_step)
    jit_fn = _JIT_CACHE.get(key)
    if jit_fn is not None:
        return jit_fn

    W_local_T = jax.ShapeDtypeStruct(
        (nq, n // Py, n // Px), dtype)
    attrs = dict(
        nq=nq, n=n,
        pref_re=float(pref_c.real),
        pref_im=float(pref_c.imag),
        ctx_handle=int(ctx_handle),
        stop_after_step=stop_after_step,
    )

    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, "x", "y"), P(None, "x", "y")),
             out_specs=P(None, "x", "y"),
             check_vma=False)
    def _w_solve(local_V, local_chi):
        # Pre-transpose so bytes are col-major (N/Px × N/Py) per rank.
        V_T   = jnp.transpose(local_V,   (0, 2, 1))
        chi_T = jnp.transpose(local_chi, (0, 2, 1))
        W_T = jax.ffi.ffi_call(
            _W_SOLVE_TARGET, W_local_T,
            # Don't alias V → output: V is typically consumed by
            # downstream (e.g. sigma_coh needs the original v), and
            # the FFI's cudaMemcpyAsync V_in → V_work path handles
            # the non-aliased case correctly.
        )(V_T, chi_T, **attrs)
        return jnp.transpose(W_T, (0, 2, 1))

    jit_fn = jax.jit(_w_solve)
    _JIT_CACHE[key] = jit_fn
    return jit_fn

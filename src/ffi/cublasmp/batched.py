"""Per-q cuBLASMp distributed GEMM on the full (Px, Py) process mesh.

Computes ``C[q] = alpha * op(A[q]) @ op(B[q]) + beta * C[q]`` per batch
element, with each matrix 2D-sharded across the device mesh in the same
``P(None, 'x', 'y')`` layout used throughout the ISDF / W-solve pipeline.

No reshards in or out — A, B, C flow through their natural layouts.  Under
the hood the FFI handler calls ``cublasMpMatmul`` once per q, reusing a
single set of descriptors (same per-rank tile size, same grid).

Restrictions:
  * Mesh is 2-D with axes ('x', 'y').
  * All matrix dims divisible by Px and Py as appropriate (one tile per
    rank — simplification that matches our ISDF convention).
  * Dtypes F64 / C128.

Pre-transpose trick: JAX stores row-major, cuBLASMp reads col-major.  We
pre-transpose the inner two dims inside shard_map so the raw bytes are
interpretable as col-major with ``lld = local_rows``.  The ``transa`` /
``transb`` attrs then have standard BLAS semantics on top of that.
"""
from __future__ import annotations

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from ..common.ffi_loader import get_lib
from ..cusolvermp.context import get_or_init_context

__all__ = ["batched_distributed_gemm", "batched_fused_w_solve",
           "batched_fused_w_solve_jit"]

_GEMM_TARGET = "lorrax_cublasmp_batched_gemm"
_W_SOLVE_TARGET = "lorrax_cublasmp_batched_w_solve"

_OP_CODE = {"N": 0, "T": 1, "C": 2,
            "n": 0, "t": 1, "c": 2}

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


def batched_distributed_gemm(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    *,
    mesh: Mesh,
    alpha: Union[float, complex] = 1.0,
    beta:  Union[float, complex] = 0.0,
    transa: str = "N",
    transb: str = "N",
) -> jax.Array:
    """Compute ``D[q] = alpha * op(A[q]) @ op(B[q]) + beta * C[q]`` via cuBLASMp.

    Inputs
    ------
    A : (Nq, Arows, Acols) sharded ``P(None, 'x', 'y')``
    B : (Nq, Brows, Bcols) sharded ``P(None, 'x', 'y')``
    C : (Nq, m,     n)     sharded ``P(None, 'x', 'y')`` — passed even
        when beta=0 (cuBLASMp requires it); can share buffer with the
        intended output by donation.
    transa, transb : 'N' | 'T' | 'C' — no-op / transpose / conjugate-transpose

    Shapes: with op(X) applied, ``op(A)`` is ``(m, k)`` and ``op(B)`` is
    ``(k, n)``.  The FFI resolves ``Arows, Acols, Brows, Bcols`` from the
    transpose flags.
    """
    if A.ndim != 3 or B.ndim != 3 or C.ndim != 3:
        raise ValueError(
            f"batched_distributed_gemm: all inputs must be 3D "
            f"(Nq, rows, cols); got A={A.shape} B={B.shape} C={C.shape}")
    if A.shape[0] != B.shape[0] or A.shape[0] != C.shape[0]:
        raise ValueError(
            f"batch dims disagree: A={A.shape[0]} B={B.shape[0]} C={C.shape[0]}")
    if A.dtype != B.dtype or A.dtype != C.dtype:
        raise ValueError(f"dtypes disagree: {A.dtype}, {B.dtype}, {C.dtype}")
    transa = transa.upper()
    transb = transb.upper()
    if transa not in "NTC" or transb not in "NTC":
        raise ValueError(f"transa/transb must be N/T/C; got {transa}/{transb}")

    Px, Py = _validate_mesh(mesh)
    if Px != Py:
        # cublasMpMatmul_bufferSize fails with status=3 (INVALID_VALUE)
        # on 1-D process grids at every size/combo (verified 4x1 and
        # 1x4, 2026-07-10); 1x1 and square grids work.
        raise ValueError(
            f"batched_distributed_gemm: mesh {Px}x{Py} is not square — "
            f"cuBLASMp rejects non-square process grids in this layout.  "
            f"Use a square mesh.")
    if transb != "N" and Px * Py > 1:
        # cuBLASMp 0.5.1 cublasMpMatmul with op(B) != N on a multi-rank
        # grid is rank-DIVERGENT: some ranks return status=6
        # (INVALID_VALUE), the rest block in a collective — the job
        # deadlocks rather than erroring (observed 2x2, 2026-07-10).
        # op(A) in {N, T, C} with op(B) = N works on every mesh; callers
        # needing op(B) != N must pre-transpose B on the JAX side.
        raise ValueError(
            f"batched_distributed_gemm: transb={transb!r} on a "
            f"{Px}x{Py} grid is not supported by cuBLASMp (rank-divergent "
            f"INVALID_VALUE -> collective deadlock).  Pre-transpose B "
            f"instead (transb='N' is the only multi-rank-safe value).")
    nq = int(A.shape[0])
    # Resolve op(A) = (m, k) and op(B) = (k, n)
    A_rows, A_cols = int(A.shape[1]), int(A.shape[2])
    B_rows, B_cols = int(B.shape[1]), int(B.shape[2])
    if transa == "N":
        m, k_a = A_rows, A_cols
    else:
        m, k_a = A_cols, A_rows
    if transb == "N":
        k_b, n = B_rows, B_cols
    else:
        k_b, n = B_cols, B_rows
    if k_a != k_b:
        raise ValueError(
            f"contraction mismatch: op(A) k = {k_a}, op(B) k = {k_b}")
    k = k_a
    if int(C.shape[1]) != m or int(C.shape[2]) != n:
        raise ValueError(
            f"C shape {C.shape[1:]} != expected ({m}, {n}) for "
            f"op(A)@op(B) = ({m}, {k}) @ ({k}, {n})")
    if m % Px != 0 or n % Py != 0 or k % Py != 0 or k % Px != 0:
        raise ValueError(
            f"m/Px={m/Px}, k/Px={k/Px}, k/Py={k/Py}, n/Py={n/Py} must all "
            "be integer — one tile per rank requirement")

    get_lib()
    ctx_handle = get_or_init_context(mesh, col_major=False)

    # Per-rank leading dimensions (= local rows) for each matrix, in the
    # col-major view after the inner transpose.
    lld_A = A_rows // Px
    lld_B = B_rows // Px
    lld_C = m // Px
    mb_a = A_rows // Px
    nb_a = A_cols // Py
    mb_b = B_rows // Px
    nb_b = B_cols // Py
    mb_c = m // Px
    nb_c = n // Py

    alpha_c = complex(alpha)
    beta_c  = complex(beta)

    key = ("gemm", _mesh_key(mesh), A.dtype, A.shape, B.shape, C.shape,
           transa, transb, alpha_c, beta_c, int(ctx_handle))
    jit_gemm = _JIT_CACHE.get(key)
    if jit_gemm is None:
        C_local_T = jax.ShapeDtypeStruct(
            (nq, n // Py, m // Px), C.dtype)

        attrs = dict(
            nq=nq, m=m, n=n, k=k,
            mb_a=mb_a, nb_a=nb_a, mb_b=mb_b, nb_b=nb_b,
            mb_c=mb_c, nb_c=nb_c,
            lld_a=lld_A, lld_b=lld_B, lld_c=lld_C,
            transa=_OP_CODE[transa], transb=_OP_CODE[transb],
            alpha_re=float(alpha_c.real),
            alpha_im=float(alpha_c.imag),
            beta_re=float(beta_c.real),
            beta_im=float(beta_c.imag),
            ctx_handle=int(ctx_handle),
        )

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"),
                           P(None, "x", "y"),
                           P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_rep=False)
        def _gemm(local_A, local_B, local_C):
            # Pre-transpose inner dims (row-major → col-major bytes).
            A_T = jnp.transpose(local_A, (0, 2, 1))
            B_T = jnp.transpose(local_B, (0, 2, 1))
            C_T = jnp.transpose(local_C, (0, 2, 1))
            # The FFI writes D in place of C (aliased).
            D_T = jax.ffi.ffi_call(
                _GEMM_TARGET, C_local_T,
                input_output_aliases={2: 0},
            )(A_T, B_T, C_T, **attrs)
            return jnp.transpose(D_T, (0, 2, 1))

        jit_gemm = jax.jit(_gemm, donate_argnums=(2,))
        _JIT_CACHE[key] = jit_gemm

    return jit_gemm(A, B, C)


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
             check_rep=False)
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

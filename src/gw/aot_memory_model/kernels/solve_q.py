"""Batched triangular solve zeta_q = L_q^{-H} L_q^{-1} Z_q across all q.

Production entry: ``common.isdf_fitting._solve_all_at_once`` (inside
``solve_zeta_from_L_q``).  Solves (nq, μ, μ) · (nq, μ, B_r) → (nq, μ, B_r).

Input shapes:
    L_q   (nq, μ, μ)   on P(None, None, None)  -- replicated panels
    Z_col (nq, μ, B_r) on P(None, None, ('x','y'))
Output:
    zeta  (nq, μ, B_r) same sharding as Z_col

MEMORY_MODEL.md says: M_solve(B_q=1) ≈ base + 4·16·n_q·μ·B_r/P + M_L_q +
q_chunk·16·μ² replicated panels.  Interpretation: 4 concurrent Z-sized
buffers + q_chunk replicated L panels.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0

def _T_Zcol(sys, knobs, mesh):
    Br = knobs.get("chunk_r", sys.n_r or 1)
    return _B * sys.n_k * sys.n_rmu * Br / (mesh.p_x * mesh.p_y)

def _T_Lrep(sys, knobs, mesh):
    # Replicated L panel per batched q: lives once per q_chunk.
    qc = knobs.get("q_chunk", sys.n_k)
    return _B * qc * sys.n_rmu ** 2

def _T_Lfull_rep(sys, knobs, mesh):
    # Full (nq, μ, μ) L_q replicated across all devices inside the
    # ``P(None, None, None)`` shard_map — NOT sharded.  Per-device size
    # is the full logical size.
    return _B * sys.n_k * sys.n_rmu ** 2


@register_kernel
class SolveQKernel(AotKernel):
    name = "solve_q"
    SYSTEM_DIMS = ("n_k", "n_rmu", "n_r", "kgrid")
    KNOBS = ("chunk_r", "q_chunk")
    PRIMITIVES = {"Zcol": _T_Zcol, "Lrep": _T_Lrep, "Lfull_rep": _T_Lfull_rep}

    def build_specs(self, sys, knobs, mesh):
        nq, mu = sys.n_k, sys.n_rmu
        Br = knobs.get("chunk_r", sys.n_r)
        sh_L = NamedSharding(mesh, P(None, None, None))  # replicated
        sh_Z = NamedSharding(mesh, P(None, None, ("x", "y")))
        return (
            jax.ShapeDtypeStruct((nq, mu, mu), jnp.complex128, sharding=sh_L),
            jax.ShapeDtypeStruct((nq, mu, Br), jnp.complex128, sharding=sh_Z),
        )

    def build_callable(self, sys, knobs, mesh):
        mesh_xy = mesh
        L_rep_shard = NamedSharding(mesh_xy, P(None, None, None))

        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None, None), P(None, None, ("x", "y"))),
                 out_specs=P(None, None, ("x", "y")))
        def _sharded_cho_solve_batch(L_batch, Z_batch):
            def _solve_single(L, Z):
                y = jax.scipy.linalg.solve_triangular(L, Z, lower=True)
                return jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
            return jax.vmap(_solve_single)(L_batch, Z_batch)

        @jax.jit
        def _solve_all_at_once(L_q, Z_col):
            L_full_rep = jax.lax.with_sharding_constraint(L_q, L_rep_shard)
            return _sharded_cho_solve_batch(L_full_rep, Z_col)
        return _solve_all_at_once

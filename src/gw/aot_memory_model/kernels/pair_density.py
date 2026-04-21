"""Pair density: P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(ν).

Production entry: ``common.isdf_fitting._compute_P_traced``
Einsum:      'kmns,knsv->kmv'

Input shapes:
    psi_L (nk, n_rmu, nb, ns) on P(None, 'x', None, None)
    psi_R (nk, nb, ns, n_rmu) on P(None, None, None, 'y')
Output:
    P     (nk, n_rmu, n_rmu) on P(None, 'x', 'y')

Primitives per device:
    T_P    = 16·n_k·μ² / (p_x·p_y)     output
    T_psiL = 16·n_k·μ·n_b·n_s / p_x    input shard (μ on x)
    T_psiR = 16·n_k·μ·n_b·n_s / p_y    input shard (μ on y)

Expected fit: β[P]≈1 (single output buffer), β[psiL]≈1, β[psiR]≈1 (inputs).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0

def _T_P(sys, knobs, mesh):
    return _B * sys.n_k * sys.n_rmu ** 2 / (mesh.p_x * mesh.p_y)

def _T_psiL(sys, knobs, mesh):
    return _B * sys.n_k * sys.n_rmu * sys.n_b * sys.n_s / mesh.p_x

def _T_psiR(sys, knobs, mesh):
    return _B * sys.n_k * sys.n_rmu * sys.n_b * sys.n_s / mesh.p_y


@register_kernel
class PairDensityKernel(AotKernel):
    name = "pair_density_traced"
    SYSTEM_DIMS = ("n_k", "n_rmu", "n_b", "n_s", "kgrid")
    KNOBS = ()
    PRIMITIVES = {"P": _T_P, "psiL": _T_psiL, "psiR": _T_psiR}

    def build_specs(self, sys, knobs, mesh):
        nk, mu, nb, ns = sys.n_k, sys.n_rmu, sys.n_b, sys.n_s
        sL = NamedSharding(mesh, P(None, "x", None, None))
        sR = NamedSharding(mesh, P(None, None, None, "y"))
        return (
            jax.ShapeDtypeStruct((nk, mu, nb, ns), jnp.complex128, sharding=sL),
            jax.ShapeDtypeStruct((nk, nb, ns, mu), jnp.complex128, sharding=sR),
        )

    def build_callable(self, sys, knobs, mesh):
        xy_out = NamedSharding(mesh, P(None, "x", "y"))
        x1_4 = NamedSharding(mesh, P(None, "x", None, None))
        y3_4 = NamedSharding(mesh, P(None, None, None, "y"))

        @partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
        def _compute_P_traced(psi_L, psi_R):
            return jnp.einsum("kmns,knsv->kmv", psi_L, psi_R, optimize=True)
        return _compute_P_traced

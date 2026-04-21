"""Σ_kij: build G(R)·W(R)/√Nk in real space + reduce-scatter tail project.

Production entry: ``gw.ppm_sigma._get_sigma_kij_kernel`` (returns
``_sigma_kij_kernel``).  Inputs (axis order matches build_G einsum
``'ksxi,kij,kjty->ksxty'``):

    psi_coh_rmuT_X  (nk, n_s, μ_X, m_coh)   P(None, None, 'x', None)
    psi_coh_rmu_Y   (nk, m_coh, n_s, μ_Y)   P(None, None, None, 'y')
    psi_proj_rmu_X  (nk, n_proj, n_s, μ_X)  P(None, None, None, 'x')
    psi_proj_rmuT_Y (nk, n_s, μ_Y, n_proj)  P(None, None, 'y', None)
    Gij             (nk, m_coh, m_coh)       replicated
    W_q             (nq, μ, μ)               P(None, 'x', 'y')

Output (2-tuple):
    sigma_re, sigma_im  (nk, m_proj, n_proj)   each  P(None, 'x', 'y')

Key primitives:
    T_G_mid   = 16·n_k·n_s²·μ² / (p_x·p_y)   G_k / G_R (Gbuf-like)
    T_V_mid   = 16·n_k·μ² / (p_x·p_y)        V_R / W_q
    T_psi_L   = 16·n_k·m_coh·n_s·μ / p_x     psi_coh_rmuT_X shard
    T_psi_R   = 16·n_k·m_coh·n_s·μ / p_y     psi_coh_rmu_Y shard
    T_psi_proj = 16·n_k·n_proj·n_s·μ / P     average for proj shards

For this POC we model ``m_coh == n_proj == sys.n_b`` (they often are at
the PPM use-site).  Breaking that out would need more primitives.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0

def _T_Gmid(sys, knobs, mesh):
    return _B * sys.n_k * (sys.n_s ** 2) * sys.n_rmu ** 2 / (mesh.p_x * mesh.p_y)

def _T_Vmid(sys, knobs, mesh):
    return _B * sys.n_k * sys.n_rmu ** 2 / (mesh.p_x * mesh.p_y)

def _T_psi_X(sys, knobs, mesh):
    return _B * sys.n_k * sys.n_b * sys.n_s * sys.n_rmu / mesh.p_x

def _T_psi_Y(sys, knobs, mesh):
    return _B * sys.n_k * sys.n_b * sys.n_s * sys.n_rmu / mesh.p_y


@register_kernel
class SigmaKijKernel(AotKernel):
    name = "sigma_kij"
    SYSTEM_DIMS = ("n_k", "n_rmu", "n_s", "n_b", "kgrid")
    KNOBS = ()
    PRIMITIVES = {"Gmid": _T_Gmid, "Vmid": _T_Vmid,
                  "psi_X": _T_psi_X, "psi_Y": _T_psi_Y}

    def build_specs(self, sys, knobs, mesh):
        nk, mu, ns, nb = sys.n_k, sys.n_rmu, sys.n_s, sys.n_b
        s_xn = NamedSharding(mesh, P(None, None, "x", None))       # (nk, s, μ_X, i)
        s_yr = NamedSharding(mesh, P(None, None, None, "y"))       # (nk, j, s, μ_Y)
        s_projx = NamedSharding(mesh, P(None, None, None, "x"))    # (nk, m, s, μ_X)
        s_projy = NamedSharding(mesh, P(None, None, "y", None))    # (nk, s, μ_Y, n)
        s_xy = NamedSharding(mesh, P(None, "x", "y"))
        s_rep = NamedSharding(mesh, P())
        return (
            jax.ShapeDtypeStruct((nk, ns, mu, nb), jnp.complex128, sharding=s_xn),
            jax.ShapeDtypeStruct((nk, nb, ns, mu), jnp.complex128, sharding=s_yr),
            jax.ShapeDtypeStruct((nk, nb, ns, mu), jnp.complex128, sharding=s_projx),
            jax.ShapeDtypeStruct((nk, ns, mu, nb), jnp.complex128, sharding=s_projy),
            jax.ShapeDtypeStruct((nk, nb, nb), jnp.complex128, sharding=s_rep),
            jax.ShapeDtypeStruct((nk, mu, mu), jnp.complex128, sharding=s_xy),
        )

    def build_callable(self, sys, knobs, mesh):
        from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn
        from gw.greens_function_kernel import build_G as _build_G
        from gw.ppm_sigma import _make_project_ri_reduce_scatter

        kgrid = sys.kgrid
        nk_tot = sys.n_k
        _G_spec = P(None, None, None, None, "x", None, "y")
        _V_spec = P(None, None, None, "x", "y")
        _G_ifftn = make_flat_k_ifftn(mesh, kgrid, _G_spec, norm="ortho")
        _G_fftn  = make_flat_k_fftn( mesh, kgrid, _G_spec, norm="ortho")
        _V_ifftn = make_flat_k_ifftn(mesh, kgrid, _V_spec, norm="ortho")
        inv_sqrt_nk = -1.0 / np.sqrt(float(nk_tot))
        _project_ri = _make_project_ri_reduce_scatter(mesh)

        @jax.jit
        def _sigma_kij_kernel(
            psi_coh_rmuT_X, psi_coh_rmu_Y, psi_proj_rmu_X, psi_proj_rmuT_Y,
            Gij, W_q,
        ):
            G_k = _build_G(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij=Gij)
            G_R = _G_ifftn(G_k)
            V_R = _V_ifftn(W_q)[:, None, :, None, :]
            sigma_k = _G_fftn(G_R * V_R * inv_sqrt_nk)
            return _project_ri(psi_proj_rmu_X, sigma_k, psi_proj_rmuT_Y)
        return _sigma_kij_kernel

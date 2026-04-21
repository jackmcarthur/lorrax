"""CCT_LR: C_q(μ,ν) = FFT[ conj(IFFT(P_l(μ,ν))) ⊙ IFFT(P_r(μ,ν)) ].

Production entry: ``common.isdf_fitting.compute_CCT_from_left_right``
(the ``_compute_CCT_LR`` inner jit with ``donate_argnums=(0,1)``).

Input shapes (flat-k form, rank 3):
    P_l (nk, μ, μ) on P(None, 'x', 'y')   -- donated
    P_r (nk, μ, μ) on P(None, 'x', 'y')   -- donated
Output:
    C_q (nq, μ, μ) on P(None, 'x', 'y')

Primitives per device:
    T_Pq   = 16·n_k·μ² / (p_x·p_y)         each (P_l, P_r, Rt-intermediates, C_q)
    T_fft  = 16·n_k·μ² / (p_x·p_y)         cuFFT workspace lives here

Expected fit: β[Pq]≈2 (one temp + one live input under donation), β[fft]≈1.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0

def _T_Pq(sys, knobs, mesh):
    return _B * sys.n_k * sys.n_rmu ** 2 / (mesh.p_x * mesh.p_y)


@register_kernel
class CctLrKernel(AotKernel):
    name = "cct_lr"
    SYSTEM_DIMS = ("n_k", "n_rmu", "kgrid")
    KNOBS = ()
    PRIMITIVES = {"Pq": _T_Pq}

    def build_specs(self, sys, knobs, mesh):
        nk, mu = sys.n_k, sys.n_rmu
        sh = NamedSharding(mesh, P(None, "x", "y"))
        return (
            jax.ShapeDtypeStruct((nk, mu, mu), jnp.complex128, sharding=sh),
            jax.ShapeDtypeStruct((nk, mu, mu), jnp.complex128, sharding=sh),
        )

    def build_callable(self, sys, knobs, mesh):
        from common.fft_helpers import make_flat_k_ifftn, make_flat_k_fftn
        kgrid = sys.kgrid
        spec_3d = P(None, None, None, "x", "y")
        flat_xy = NamedSharding(mesh, P(None, "x", "y"))
        local_ifftn = make_flat_k_ifftn(mesh, kgrid, spec_3d, norm="forward")
        local_fftn  = make_flat_k_fftn( mesh, kgrid, spec_3d, norm="forward")

        @partial(jax.jit, in_shardings=(flat_xy, flat_xy), out_shardings=flat_xy,
                 donate_argnums=(0, 1))
        def _compute_CCT_LR(P_l, P_r):
            P_l_Rt = local_ifftn(P_l)
            P_r_Rt = local_ifftn(P_r)
            C_Rt = jnp.conj(P_l_Rt) * P_r_Rt
            return local_fftn(C_Rt)
        return _compute_CCT_LR

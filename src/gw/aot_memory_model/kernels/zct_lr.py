"""ZCT_LR: Z_q(μ,r) = FFT[ conj(IFFT(P_l(μ,r))) ⊙ IFFT(P_r(μ,r)) ].

r is the z-chunk (knob ``chunk_r``).  Production entry:
``common.isdf_fitting.compute_ZCT_from_left_right_zchunk``.

The production code splits this into two sub-jits so XLA can donate
rank-3 inputs through the IFFT.  For AOT modeling we mirror the same
split as a single outer callable (jax will re-jit the composite).

Input shapes (flat-k form, rank 3):
    P_l (nk, μ, B_r) on P(None, 'x', 'y')   -- donated
    P_r (nk, μ, B_r) on P(None, 'x', 'y')   -- donated
Output:
    Z_q (nq, μ, B_r) on P(None, 'x', 'y')

MEMORY_MODEL.md stage says: M_zct = base + 16·B_r·[(4·n_k + n_q)·μ/P]
interpreted as 4 pair-sized temps + 1 output.  AOT will measure whether
the '4' concurrency is real.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0

def _T_PrBr(sys, knobs, mesh):
    Br = knobs.get("chunk_r", sys.n_r or 1)
    return _B * sys.n_k * sys.n_rmu * Br / (mesh.p_x * mesh.p_y)


@register_kernel
class ZctLrKernel(AotKernel):
    name = "zct_lr"
    SYSTEM_DIMS = ("n_k", "n_rmu", "n_r", "kgrid")
    KNOBS = ("chunk_r",)
    PRIMITIVES = {"PrBr": _T_PrBr}

    def build_specs(self, sys, knobs, mesh):
        nk, mu = sys.n_k, sys.n_rmu
        Br = knobs.get("chunk_r", sys.n_r)
        sh = NamedSharding(mesh, P(None, "x", "y"))
        return (
            jax.ShapeDtypeStruct((nk, mu, Br), jnp.complex128, sharding=sh),
            jax.ShapeDtypeStruct((nk, mu, Br), jnp.complex128, sharding=sh),
        )

    def build_callable(self, sys, knobs, mesh):
        from common.fft_helpers import make_flat_k_ifftn, make_flat_k_fftn
        kgrid = sys.kgrid
        spec_3d = P(None, None, None, "x", "y")
        flat_xy = NamedSharding(mesh, P(None, "x", "y"))
        local_ifftn = make_flat_k_ifftn(mesh, kgrid, spec_3d, norm="forward")
        local_fftn  = make_flat_k_fftn( mesh, kgrid, spec_3d, norm="forward")

        @partial(jax.jit, in_shardings=flat_xy, out_shardings=flat_xy,
                 donate_argnums=(0,))
        def _left_ifft_conj(P_l):
            return jnp.conj(local_ifftn(P_l))

        @partial(jax.jit,
                 in_shardings=(flat_xy, flat_xy), out_shardings=flat_xy,
                 donate_argnums=(0, 1))
        def _right_ifft_mul_fft(P_r, P_l_Rt):
            P_r_Rt = local_ifftn(P_r)
            Z_Rt = P_l_Rt * P_r_Rt
            return local_fftn(Z_Rt)

        # Outer wrapper — AOT-lowered as a single program; JAX traces the
        # two sub-jits and inlines them.
        def _compute_ZCT_LR(P_l, P_r):
            P_l_Rt = _left_ifft_conj(P_l)
            return _right_ifft_mul_fft(P_r, P_l_Rt)

        return jax.jit(_compute_ZCT_LR, donate_argnums=(0, 1))

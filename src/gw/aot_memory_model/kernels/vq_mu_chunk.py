"""V_q single-μ-chunk kernel: FFT + weight + self-contract at one q.

Production inner: ``make_v_munu_chunked_kernel.fft_weight_contract_diag``
(used by ``compute_all_V_q_from_zeta_h5`` per μ-chunk).  Flow:

    ζ(r) ∈ C^{B_μ, n_r}  →  FFT over 3D grid  →  weight by √v(G)
                       →  contract  V[μ,ν] = Σ_G conj(ζ̃_μ) ζ̃_ν

Input shapes:
    zeta_mu_r : (μ_chunk, n_r)         replicated
    sqrt_v    : (n_r,)                 replicated
    phase     : (n_r,) complex         replicated

Output:
    V_ii      : (μ_chunk, μ_chunk)

MEMORY_MODEL.md says: ``μ_chunk ≤ available_bytes / (3 * 16 * n_r)``
interpreted as 3 concurrent (μ_chunk × n_r) buffers.  AOT will measure
whether the '3' concurrency is real.

This kernel is small-scale and runs single-GPU; we set p_x=p_y=1.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0

def _T_zeta(sys, knobs, mesh):
    mc = knobs.get("mu_chunk", sys.n_rmu)
    return _B * mc * sys.n_r

def _T_vphase(sys, knobs, mesh):
    return _B * sys.n_r

def _T_out(sys, knobs, mesh):
    mc = knobs.get("mu_chunk", sys.n_rmu)
    return _B * mc * mc


@register_kernel
class VqMuChunkKernel(AotKernel):
    name = "vq_mu_chunk"
    SYSTEM_DIMS = ("n_rmu", "n_r", "fft_grid")
    KNOBS = ("mu_chunk",)
    PRIMITIVES = {"zeta": _T_zeta, "vphase": _T_vphase, "out": _T_out}

    def build_specs(self, sys, knobs, mesh):
        mc = knobs.get("mu_chunk", sys.n_rmu)
        nr = sys.n_r
        rep = NamedSharding(mesh, P())
        return (
            jax.ShapeDtypeStruct((mc, nr), jnp.complex128, sharding=rep),
            jax.ShapeDtypeStruct((nr,), jnp.float64, sharding=rep),
            jax.ShapeDtypeStruct((nr,), jnp.complex128, sharding=rep),
        )

    def build_callable(self, sys, knobs, mesh):
        nx, ny, nz = sys.kgrid if sys.kgrid else (1, 1, 1)
        # Use n_r as the flat product; assume the AotKernel caller set
        # kgrid to fft_grid (we reuse the SysDims.kgrid field as the
        # 3-D FFT box here).
        fft_nx, fft_ny, fft_nz = nx, ny, nz
        n_G = fft_nx * fft_ny * fft_nz

        @jax.jit
        def _fft_weight_contract_diag(zeta_r, sqrt_v, phase):
            B_mu = zeta_r.shape[0]
            zeta_G = jnp.fft.fftn(
                zeta_r.reshape(B_mu, fft_nx, fft_ny, fft_nz) * phase.reshape(fft_nx, fft_ny, fft_nz),
                axes=(-3, -2, -1)).reshape(B_mu, n_G)
            zeta_weighted = zeta_G * sqrt_v.astype(jnp.complex128).reshape(1, -1)
            V_ii = jnp.einsum("mG,nG->mn", jnp.conj(zeta_weighted), zeta_weighted,
                              optimize=True)
            return V_ii
        return _fft_weight_contract_diag

"""3D bulk Coulomb: v(q+G) = 8π/|q+G|², no truncation.  This is the default."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from common import Meta
from .base import SysDim, sample_minibz_qpoints


class Bulk3D:
    sys_dim = SysDim.BULK_3D

    def v_qG(self, wfn, qvec_wrapped, comps_qG) -> jax.Array:
        bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
        comps = np.asarray(comps_qG, dtype=np.float64)
        qvec = np.asarray(qvec_wrapped, dtype=np.float64)
        G_cart = (comps + qvec) @ bvec  # (nG, 3)
        denom = np.sum(G_cart * G_cart, axis=1)
        denom_zero = denom < 1e-12
        denom_safe = np.where(denom_zero, 1.0, denom)
        v = 8.0 * np.pi / denom_safe
        v = v / float(wfn.cell_volume)
        v = np.where(denom_zero, 0.0, v)
        return jnp.asarray(v, dtype=jnp.complex128)

    def _vq_isotropic(self, qcart):
        denom = jnp.einsum("ij,ij->i", qcart, qcart)
        return 8.0 * jnp.pi / denom

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,
        epshead=None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
    ):
        batches = sample_minibz_qpoints(
            wfn, meta, nsamples=nsamples, method=method, qmc_reps=qmc_reps,
        )
        # vc0_mean: average v(q) across all sampled q-points, then mean over reps.
        means = [jnp.mean(self._vq_isotropic(rq)) for rq in batches]
        vc0_mean = jnp.mean(jnp.stack(means))

        if S_cart is not None:
            # Anisotropic screened: w0 = ⟨ v / (1 - v · qᵀSq) ⟩ on the same q's.
            S = jnp.asarray(S_cart, dtype=jnp.complex128)
            wmeans = []
            for rq in batches:
                vq = self._vq_isotropic(rq).astype(jnp.complex128)
                qSq = jnp.einsum('qi,ij,qj->q', rq, S, rq)
                wmeans.append(jnp.mean(vq / (1.0 - vq * qSq)))
            wcoul0 = jnp.mean(jnp.stack(wmeans))
            return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

        # Isotropic Ismail-Beigi gamma fallback (epshead-driven).  Kept for
        # back-compat with older runs that don't compute the dipole tensor.
        bvec = jnp.asarray(wfn.blat * wfn.bvec, dtype=jnp.float64)
        q0_crys = jnp.asarray((0.001, 0.0, 0.0), dtype=jnp.float64)
        q0_cart = q0_crys @ bvec
        q0sq = jnp.dot(q0_cart, q0_cart)
        vc_q0 = 8.0 * jnp.pi / q0sq
        eps_real = jnp.asarray(jnp.real(epshead), dtype=jnp.float64)
        gamma = (1.0 / eps_real - 1.0) / (q0sq * vc_q0)
        # Reuse the last batch's q-points for the screening-fallback estimator.
        rq_last = batches[-1]
        qsq = jnp.einsum("ij,ij->i", rq_last, rq_last)
        vq = 8.0 * jnp.pi / qsq
        wq = vq / (1.0 + vq * qsq * gamma)
        wcoul0 = jnp.mean(wq)
        return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

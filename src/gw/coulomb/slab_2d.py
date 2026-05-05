"""2D slab (Ismail-Beigi) Coulomb truncation along the c axis.

  v_2D(q+G) = (8π/|q+G|²) · (1 − exp(−zc·|q‖+G‖|) cos((qz+Gz)·zc)),  zc = π/b_z

The sampler in ``base.sample_minibz_qpoints`` already sets ``qz=0`` for
2D; this module just supplies the formula.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from common import Meta
from .base import SysDim, sample_minibz_qpoints


class Slab2D:
    sys_dim = SysDim.SLAB_2D

    def v_qG(self, wfn, qvec_wrapped, comps_qG) -> jax.Array:
        bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
        comps = np.asarray(comps_qG, dtype=np.float64)
        qvec = np.asarray(qvec_wrapped, dtype=np.float64)
        G_cart = (comps + qvec) @ bvec
        denom = np.sum(G_cart * G_cart, axis=1)
        denom_zero = denom < 1e-12

        zc = float(np.pi / bvec[2, 2])
        kxy = np.linalg.norm(G_cart[:, :2], axis=1)
        kz = G_cart[:, 2]
        f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kz * zc)

        denom_safe = np.where(denom_zero, 1.0, denom)
        v = (8.0 * np.pi / denom_safe) * f2d
        v = v / float(wfn.cell_volume)
        v = np.where(denom_zero, 0.0, v)
        return jnp.asarray(v, dtype=jnp.complex128)

    def _vq_2d(self, qcart, zc):
        # qcart already has qz=0 from sample_minibz_qpoints
        denom = jnp.einsum("ij,ij->i", qcart, qcart)
        base = 4.0 * jnp.pi / denom
        kxy = jnp.linalg.norm(qcart[:, :2], axis=1)
        # The "2 ·" pulls the 4π → 8π in front while the truncation factor
        # ``(1 − e^{-zc·kxy})`` runs against the physical (qz=0) shell.
        f2d = 2.0 * (1.0 - jnp.exp(-zc * kxy))
        return base * f2d

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,
        epshead=None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
    ):
        bvec = jnp.asarray(wfn.blat * wfn.bvec, dtype=jnp.float64)
        zc = jnp.pi / bvec[2, 2]

        batches = sample_minibz_qpoints(
            wfn, meta, nsamples=nsamples, method=method, qmc_reps=qmc_reps,
        )
        # Sobol path uses the per-rep formula with the *cosine* term included
        # (since rq.z is exactly zero by construction, cos(qz·zc) == 1, but
        # we mirror the historical 8π form for bit-identity with prior runs).
        # Uniform fallback uses the 4π form below.
        # In practice: Sobol uses the explicit formula in the historical
        # code; we replicate it.
        def _vq_sobol(rq):
            denom = jnp.einsum("ij,ij->i", rq, rq)
            base = 8.0 * jnp.pi / denom
            kxy = jnp.linalg.norm(rq[:, :2], axis=1)
            # rq[:, 2] is already 0 → cos(...) is 1 numerically, kept for
            # explicitness against prior bit-identical reference output.
            f2d = 1.0 - jnp.exp(-zc * kxy) * jnp.cos(rq[:, 2] * zc)
            return base * f2d

        means = [jnp.mean(_vq_sobol(rq)) for rq in batches]
        vc0_mean = jnp.mean(jnp.stack(means))

        if S_cart is not None:
            S = jnp.asarray(S_cart, dtype=jnp.complex128)
            wmeans = []
            for rq in batches:
                vq = _vq_sobol(rq).astype(jnp.complex128)
                qSq = jnp.einsum('qi,ij,qj->q', rq, S, rq)
                wmeans.append(jnp.mean(vq / (1.0 - vq * qSq)))
            wcoul0 = jnp.mean(jnp.stack(wmeans))
            return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

        # 2D Ismail-Beigi gamma fallback (epshead-driven, isotropic).
        q0_crys = jnp.asarray((0.001, 0.0, 0.0), dtype=jnp.float64)
        q0_cart = q0_crys @ bvec
        q0len = jnp.linalg.norm(q0_cart)
        vc_q0 = (1.0 - jnp.exp(-q0len * zc)) / jnp.where(
            q0len > 0, q0len * q0len, 1.0)
        eps_real = jnp.asarray(jnp.real(epshead), dtype=jnp.float64)
        gamma = (1.0 / eps_real - 1.0) / jnp.where(
            vc_q0 > 0, (q0len * q0len) * vc_q0, 1.0)
        # Build w(q) on the same Sobol points (qz=0 shell)
        rq_last = batches[-1]
        kxy = jnp.linalg.norm(rq_last[:, :2], axis=1)
        vc_q = (1.0 - jnp.exp(-kxy * zc)) / (kxy * kxy)
        wq = vc_q / (1.0 + vc_q * (kxy * kxy) * gamma)
        wcoul0 = 8.0 * jnp.pi * jnp.mean(wq)
        return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

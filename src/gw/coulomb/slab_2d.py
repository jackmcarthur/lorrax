"""2D slab (Ismail-Beigi) Coulomb truncation along the c axis — q→0 head.

  v_2D(q+G) = (8π/|q+G|²) · (1 − exp(−zc·|q‖+G‖|) cos((qz+Gz)·zc)),  zc = π/b_z

The sampler in ``base.sample_minibz_qpoints`` already sets ``qz=0`` for
2D; this module supplies the q→0 ``(vc0_mean, wcoul0)`` average only.  The
per-sphere ``v(q+G)`` builder is :func:`gw.compute_vcoul.compute_v_q_per_G`.
"""
from __future__ import annotations

import jax.numpy as jnp

from common import Meta
from .base import SysDim, sample_minibz_qpoints
from .kernel import TOL_MC_NAN, v_qG


class Slab2D:
    sys_dim = SysDim.SLAB_2D
    q0_units = "bare"

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,
        epshead=None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ):
        bvec = jnp.asarray(wfn.blat * wfn.bvec, dtype=jnp.float64)
        zc = jnp.pi / bvec[2, 2]

        # 2D head is a |Q| cusp, not a 1/q² pole → no analytic sphere term;
        # the flag only widens the Voronoi fold (nmax 1→3, BGW ncell=3).
        # Default (flag off) keeps nmax=1 → bit-identical.
        nmax = 3 if analytic_sphere else 1
        batches = sample_minibz_qpoints(
            wfn, meta, nsamples=nsamples, method=method, qmc_reps=qmc_reps,
            nmax=nmax,
        )
        # The slab kernel on MC draws, BARE units (see q0_average's contract),
        # with the MC NaN guard rather than the lattice-slot guard: a draw at
        # tiny |q| is a real sample of an integrable integrand, not the q=G=0
        # lattice slot.  rq[:, 2] is already 0 by construction, so the cosine
        # is 1 — kept in the shared formula for explicitness.
        def _vq_sobol(rq):
            return v_qG(rq, axis=1, sys_dim=2, channel="full", units="bare",
                        zc=zc, zero_tol=TOL_MC_NAN, xp=jnp)

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

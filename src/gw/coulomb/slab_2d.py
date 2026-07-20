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
from .base import (SysDim, sample_minibz_qpoints, minibz_average,
                   minibz_voronoi_batches, minibz_inscribed_sphere_r2)


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

    def v_head_minibz_avg(
        self, wfn, meta: Meta, shift_frac, *,
        alpha: float | None = None,
        kind: str = "slab",
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        n_coarse: int = 250_000,
    ) -> float:
        """Mini-BZ CELL AVERAGE of the slab Coulomb head at a FINITE shift.

        ``shift_frac`` — fractional ``Q + G*`` (the smallest-|Q+G|
        umklapp).  Returns ``<v_slab(Q+G*)>_mBZ`` in **bare** units (no
        ``1/celvol``).  In-plane pure adaptive MC — the 2D head is a ``|Q|``
        cusp, not a ``1/q²`` pole, so no inscribed-sphere split (BGW
        ``minibzaverage_2d``, ``minibzaverage.f90:97-186``).  ``kind`` picks
        the bare ``slab`` or the Gaussian ``slab_lr`` (b26p SR/LR) channel.

        This is the finite-q cell-average path §16.4 flagged missing for the
        2D slab (the stored ``V_qmunu`` body uses a POINT value); it shares
        the single-source :func:`base.minibz_average` with the 3D head and
        the BSE per-Q ``eval_vq`` head.
        """
        bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
        kgrid = (meta.nkx, meta.nky, meta.nkz)
        shift_cart = np.asarray(shift_frac, dtype=np.float64) @ bvec
        dq = minibz_voronoi_batches(
            bvec, kgrid, nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, nmax=3, is_2d=True)
        q0sph2 = minibz_inscribed_sphere_r2(bvec, kgrid, is_2d=True)
        return minibz_average(
            shift_cart, dq, kind=kind, celvol=float(wfn.cell_volume),
            n_kpts=int(meta.nkx * meta.nky * meta.nkz), q0sph2=q0sph2,
            alpha=alpha, zc=float(np.pi / bvec[2, 2]),
            analytic_sphere=False, adaptive=True, n_coarse=n_coarse)

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

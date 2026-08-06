"""3D bulk Coulomb: v(q+G) = 8π/|q+G|², no truncation.  This is the default."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from common import Meta
from .base import (SysDim, sample_minibz_qpoints, minibz_average,
                   minibz_inscribed_sphere_r2)
from .kernel import TOL_MC_NAN, v_qG


class Bulk3D:
    sys_dim = SysDim.BULK_3D
    q0_units = "bare"

    def _vq_isotropic(self, qcart):
        """8π/|q|² on MC draws (N, 3), BARE units — see q0_average's contract.

        ``TOL_MC_NAN`` (not ``TOL_QG_ZERO``): these are Monte-Carlo samples,
        and a draw at |q|² ~ 1e-13 is a legitimate sample of an integrable
        integrand carrying real weight.  The guard exists only to stop an
        exact 0/0.
        """
        return v_qG(qcart, axis=1, sys_dim=3, channel="full", units="bare",
                    zero_tol=TOL_MC_NAN, xp=jnp)

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,
        epshead=None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ):
        # ``analytic_sphere`` (head_minibz_average): add the analytic
        # Baldereschi-Tosatti sphere term to the q→0 head so vc0_mean is
        # seed-independent (the pure-Sobol mean has a few tiny δq → 8π/|δq|²
        # blow-ups that make it drift between seeds).  nmax 1→3 widens the
        # Voronoi fold (BGW ncell=3) for skewed cells.  Default False keeps
        # the historical pure-Sobol average bit-identical.
        nmax = 3 if analytic_sphere else 1
        batches = sample_minibz_qpoints(
            wfn, meta, nsamples=nsamples, method=method, qmc_reps=qmc_reps,
            nmax=nmax,
        )
        if analytic_sphere:
            bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
            kgrid = (meta.nkx, meta.nky, meta.nkz)
            q0sph2 = minibz_inscribed_sphere_r2(bvec, kgrid, is_2d=False)
            n_kpts = int(meta.nkx * meta.nky * meta.nkz)
            vc0_mean = jnp.asarray(minibz_average(
                np.zeros(3), [np.asarray(b) for b in batches],
                kind="bulk_3d", celvol=float(wfn.cell_volume), n_kpts=n_kpts,
                q0sph2=q0sph2, analytic_sphere=True), dtype=jnp.float64)
        else:
            # vc0_mean: average v(q) across all sampled q-points, mean over reps.
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

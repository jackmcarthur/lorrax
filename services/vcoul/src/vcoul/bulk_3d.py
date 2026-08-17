"""3D bulk Coulomb: v(q+G) = 8π/|q+G|², no truncation.  This is the default."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.base import SysDim, check_head_tensor_symmetry, v_qG_single
from vcoul.geometry import CoulombGeometry
from vcoul.minibz import (minibz_average, minibz_inscribed_sphere_r2,
                          sample_minibz_qpoints)

__all__ = ["Bulk3D"]


class Bulk3D:
    sys_dim = SysDim.BULK_3D

    def _v_bare_per_q(self, qf, gvec_q, *, bvec_f, fact,
                      bdot=None, fft_grid=None):
        """``8π/|q+G|² / Ω_cell``, no truncation.  See the base Protocol.

        Arithmetic order is the shipped production order (``v_reg * fact``,
        NOT ``v / cell_volume``) — the two differ in the last ulp and this
        path is bit-compared against the pre-port table.
        """
        del bdot, fft_grid
        qG_frac = qf[:, None] + gvec_q                        # (3, nG)
        qG_cart = bvec_f.T @ qG_frac                          # (3, nG)
        denom = np.sum(qG_cart * qG_cart, axis=0)             # (nG,)
        denom_zero = denom < 1e-12
        denom_safe = np.where(denom_zero, 1.0, denom)
        v_reg = 8.0 * np.pi / denom_safe
        v = np.where(denom_zero, 0.0, v_reg * fact)
        return v, denom

    def v_qG(self, geometry: CoulombGeometry, qvec_wrapped,
             comps_qG) -> jax.Array:
        return v_qG_single(self, geometry, qvec_wrapped, comps_qG)

    def _vq_isotropic(self, qcart):
        denom = jnp.einsum("ij,ij->i", qcart, qcart)
        return 8.0 * jnp.pi / denom

    def q0_average(
        self, geometry: CoulombGeometry, kgrid, *,
        S_cart=None,
        epshead=None,
        static_kappa2=None,
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
        nkx, nky, nkz = (int(s) for s in kgrid)
        nmax = 3 if analytic_sphere else 1
        batches = sample_minibz_qpoints(
            geometry, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, nmax=nmax, is_2d=False,
        )
        if analytic_sphere:
            bvec = np.asarray(geometry.bvec, dtype=np.float64)
            q0sph2 = minibz_inscribed_sphere_r2(
                bvec, (nkx, nky, nkz), is_2d=False)
            n_kpts = int(nkx * nky * nkz)
            vc0_mean = jnp.asarray(minibz_average(
                np.zeros(3), [np.asarray(b) for b in batches],
                kind="bulk_3d", celvol=float(geometry.cell_volume),
                n_kpts=n_kpts,
                q0sph2=q0sph2, analytic_sphere=True), dtype=jnp.float64)
        else:
            # vc0_mean: average v(q) across all sampled q-points, mean over reps.
            means = [jnp.mean(self._vq_isotropic(rq)) for rq in batches]
            vc0_mean = jnp.mean(jnp.stack(means))

        if static_kappa2 is not None:
            if S_cart is not None:
                raise ValueError(
                    "q0_average accepts either static_kappa2 or S_cart, not both")
            kappa2 = jnp.asarray(static_kappa2, dtype=jnp.complex128)
            if kappa2.ndim != 0 or float(jnp.real(kappa2)) <= 0.0:
                raise ValueError("static_kappa2 must be one positive scalar")
            wmeans = []
            for rq in batches:
                q2 = jnp.einsum("qi,qi->q", rq, rq)
                wmeans.append(jnp.mean(8.0 * jnp.pi / (q2 + kappa2)))
            wcoul0 = jnp.mean(jnp.stack(wmeans))
            return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

        if S_cart is not None:
            # Anisotropic screened: w0 = ⟨ v / (1 - v · qᵀSq) ⟩ on the same q's.
            S = jnp.asarray(S_cart, dtype=jnp.complex128)
            check_head_tensor_symmetry(S, "vcoul.Bulk3D.q0_average")
            wmeans = []
            for rq in batches:
                vq = self._vq_isotropic(rq).astype(jnp.complex128)
                qSq = jnp.einsum('qi,ij,qj->q', rq, S, rq)
                wmeans.append(jnp.mean(vq / (1.0 - vq * qSq)))
            wcoul0 = jnp.mean(jnp.stack(wmeans))
            return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

        # Isotropic Ismail-Beigi gamma fallback (epshead-driven).  Kept for
        # back-compat with older runs that don't compute the dipole tensor.
        bvec = jnp.asarray(geometry.bvec, dtype=jnp.float64)
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

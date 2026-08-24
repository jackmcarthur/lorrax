"""2D slab (Ismail-Beigi) Coulomb truncation along the c axis.

  v_2D(q+G) = (8π/|q+G|²) · (1 − exp(−zc·|q‖+G‖|) cos((qz+Gz)·zc)),  zc = π/b_z

:func:`~vcoul.minibz.sample_minibz_qpoints` already sets ``qz=0`` when
``is_2d``; this module just supplies the formula.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.base import SysDim, v_qG_single
from vcoul.geometry import CoulombGeometry
from vcoul.minibz import (_sample_q0_minibz_qpoints, minibz_average,
                          minibz_inscribed_sphere_r2,
                          minibz_transverse_head_avg, minibz_voronoi_batches)

__all__ = ["Slab2D"]


class Slab2D:
    sys_dim = SysDim.SLAB_2D

    def _v_bare_per_q(self, qf, gvec_q, *, bvec_f, fact,
                      bdot=None, fft_grid=None):
        """Ismail-Beigi slab truncation.  See the base Protocol.

        Arithmetic order is the shipped production order — ``v_reg * fact``
        rather than ``v / cell_volume``, and ``sqrt(x² + y²)`` rather than
        ``linalg.norm``; both differ from the old class spelling in the
        last ulp, and this path is bit-compared against the pre-port table.
        """
        del bdot, fft_grid
        zc = float(np.pi / float(bvec_f[2, 2]))
        qG_frac = qf[:, None] + gvec_q                        # (3, nG)
        qG_cart = bvec_f.T @ qG_frac                          # (3, nG)
        denom = np.sum(qG_cart * qG_cart, axis=0)             # (nG,)
        denom_zero = denom < 1e-12
        denom_safe = np.where(denom_zero, 1.0, denom)
        kxy = np.sqrt(qG_cart[0]**2 + qG_cart[1]**2)
        kz = qG_cart[2]
        f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kz * zc)
        v_reg = (8.0 * np.pi / denom_safe) * f2d
        v = np.where(denom_zero, 0.0, v_reg * fact)
        return v, denom

    def v_qG(self, geometry: CoulombGeometry, qvec_wrapped,
             comps_qG) -> jax.Array:
        return v_qG_single(self, geometry, qvec_wrapped, comps_qG)

    def v_head_minibz_avg(
        self, geometry: CoulombGeometry, kgrid, shift_frac, *,
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
        the single-source :func:`~vcoul.minibz.minibz_average` with the 3D
        head and the BSE per-Q ``eval_vq`` head.

        UNWIRED: zero callers as of the extraction.  It is the only route
        from the GW side to ``_minibz_kernel_bare(kind="slab"/"slab_lr")``
        — a capability awaiting a caller, not a duplicate.
        """
        nkx, nky, nkz = (int(s) for s in kgrid)
        bvec = np.asarray(geometry.bvec, dtype=np.float64)
        shift_cart = np.asarray(shift_frac, dtype=np.float64) @ bvec
        dq = minibz_voronoi_batches(
            bvec, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, nmax=3, is_2d=True)
        q0sph2 = minibz_inscribed_sphere_r2(bvec, (nkx, nky, nkz), is_2d=True)
        return minibz_average(
            shift_cart, dq, kind=kind, celvol=float(geometry.cell_volume),
            n_kpts=int(nkx * nky * nkz), q0sph2=q0sph2,
            alpha=alpha, zc=float(np.pi / bvec[2, 2]),
            analytic_sphere=False, adaptive=True, n_coarse=n_coarse)

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
        if static_kappa2 is not None:
            raise NotImplementedError(
                "static_kappa2 is the 3D Thomas-Fermi model; the 2D metallic "
                "q->0 form requires its separate |q| kernel")
        nkx, nky, nkz = (int(s) for s in kgrid)
        bvec = jnp.asarray(geometry.bvec, dtype=jnp.float64)
        zc = jnp.pi / bvec[2, 2]

        # 2D head is a |Q| cusp, not a 1/q² pole → no analytic sphere term;
        # the flag only widens the Voronoi fold (nmax 1→3, BGW ncell=3).
        # Default (flag off) keeps nmax=1 → bit-identical.
        batches = _sample_q0_minibz_qpoints(
            geometry, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, analytic_sphere=analytic_sphere, is_2d=True,
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

    def q0_average_transverse_tensor(
        self, geometry: CoulombGeometry, kgrid, *,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ) -> np.ndarray:
        """``T_ab = ⟨v_slab(q) t_ab(q̂)⟩_mBZ`` at q=Γ — the bare Coulomb-
        gauge transverse-projector head (bispinor TT), bare units (no
        ``1/celvol``).  Same draw as :meth:`q0_average` (``nmax`` 1↔3 on
        the same ``analytic_sphere`` flag, ``is_2d=True``), so a caller
        pinned to the CC ``vc0`` sampler gets the SAME sample points for
        the TT head — see :func:`~vcoul.minibz.minibz_transverse_head_avg`
        for the estimator and the physics it replaces (the missing q=Γ,
        G=0 slot of the bare TT tiles,
        ``docs/BISPINOR_DHFB_DESIGN.md`` §11).
        """
        nkx, nky, nkz = (int(s) for s in kgrid)
        bvec = np.asarray(geometry.bvec, dtype=np.float64)
        batches = _sample_q0_minibz_qpoints(
            geometry, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, analytic_sphere=analytic_sphere, is_2d=True)
        q0sph2 = minibz_inscribed_sphere_r2(bvec, (nkx, nky, nkz), is_2d=True)
        return minibz_transverse_head_avg(
            np.zeros(3), [np.asarray(b) for b in batches], kind="slab",
            celvol=float(geometry.cell_volume), n_kpts=int(nkx * nky * nkz),
            q0sph2=q0sph2, zc=float(np.pi / bvec[2, 2]),
            analytic_sphere=analytic_sphere, adaptive=True)

"""3D bulk Coulomb: v(q+G) = 8π/|q+G|², no truncation.  This is the default."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.base import SysDim, v_qG_single
from vcoul.geometry import CoulombGeometry
from vcoul.quadrature import gauss_legendre_interval
from vcoul.minibz import (_sample_q0_minibz_qpoints, minibz_average,
                          minibz_inscribed_sphere_r2,
                          minibz_transverse_head_avg,
                          _analytic_sphere_bare_head)

__all__ = ["Bulk3D"]


def _isotropic_vq(qcart):
    """Bare ``v(q) = 8 pi/|q|^2`` at every row of ``qcart (n,3)``."""
    denom = jnp.einsum("ij,ij->i", qcart, qcart)
    return 8.0 * jnp.pi / denom


def _screened_terms(rq, S, extra):
    """``v / (1 - v (q.S_z.q + chi_z(q)))`` at every point of ``rq`` for every row z: [Z, n].

    ``S`` [Z,3,3]; ``extra`` [Z,n] (the rows' ``chi_extra`` at these points) or None.
    """
    vq = _isotropic_vq(rq).astype(jnp.complex128)[None]
    qSq = jnp.einsum("qi,zij,qj->zq", rq, S, rq)
    if extra is not None:
        qSq = qSq + extra
    return vq / (1.0 - vq * qSq)


@jax.jit
def _screened_rows(rq, S, extra):
    """The rows' mean over one draw batch: [Z]. One program per batch serves all rows."""
    return jnp.mean(_screened_terms(rq, S, extra), axis=-1)


@jax.jit
def _screened_sums(rq, S, extra, live):
    """The rows' sum over one process's share of a draw batch, ``live`` points only: [Z]."""
    return jnp.sum(jnp.where(live, _screened_terms(rq, S, extra), 0.0), axis=-1)


def _interband_sphere_factor(S_carts):
    """Solid-angle mean of 1/(1 - 8π n.S.n); radial integral is exact.

    Polar Gauss--Legendre and periodic azimuth rules share the existing
    quadrature backend. Two successive refinements must agree. This is an
    observed angular convergence check, not an error bound for a singular
    dielectric tensor. Isotropic tensors are exact at every rule order.
    """
    S = np.asarray(S_carts, dtype=np.complex128)
    if S.ndim != 3 or S.shape[1:] != (3, 3) or not np.isfinite(S).all():
        raise ValueError('GATE bulk_q0_sphere_tensor: finite (n,3,3) required')
    # A real symmetric dielectric with both signs (or a zero eigenvalue)
    # has an angular pole, whether or not this rule samples its direction.
    # Antisymmetric Hall components do not enter n.S.n. Complex tensors
    # still need the observed refinement; this is not an exhaustive test.
    epsilon = np.eye(3)[None] - 4*np.pi*(S+S.transpose(0, 2, 1))
    for row in epsilon:
        if np.all(row.imag == 0):
            eig = np.linalg.eigvalsh(row.real)
            if eig[0] <= 0 <= eig[-1]:
                raise ValueError('GATE bulk_q0_sphere_singular: real angular pole')
    previous, good = None, 0
    for order in (16, 32, 64, 128, 256, 512):
        u, w = gauss_legendre_interval(order, -1.0, 1.0)
        phi = 2*np.pi*np.arange(2*order)/(2*order)
        radius = np.sqrt(np.maximum(0.0, 1-u*u))
        directions = np.stack(np.broadcast_arrays(
            radius[:, None]*np.cos(phi), radius[:, None]*np.sin(phi),
            u[:, None]), axis=-1).reshape(-1, 3)
        weights = np.repeat(w/(4*order), 2*order)
        denominator = 1-8*np.pi*np.einsum('qa,zab,qb->zq', directions, S, directions)
        if not np.isfinite(denominator).all() or np.any(denominator == 0):
            raise ValueError('GATE bulk_q0_sphere_singular: angular pole')
        value = np.sum(weights[None, :]/denominator, axis=-1)
        if previous is not None:
            passed = np.all(np.abs(value-previous) <= 1e-12 + 1e-10*np.abs(value))
            good = good+1 if passed else 0
            if good >= 2:
                return value
        previous = value
    raise ValueError('GATE bulk_q0_sphere_not_converged: angular refinement failed')


def _thomas_fermi_sphere_factor(radius, kappa):
    """Exact radial sphere integral divided by its bare counterpart."""
    x = complex(radius)/complex(kappa)
    if abs(x) < 1e-3:
        x2 = x*x
        return x2*(1/3-x2*(1/5-x2*(1/7-x2/9)))
    return 1-np.arctan(x)/x


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
        return _isotropic_vq(qcart)

    def q0_average(
        self, geometry: CoulombGeometry, kgrid, *,
        S_cart=None,
        epshead=None,
        static_kappa2=None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
        extra_chi=None,
    ):
        if analytic_sphere and extra_chi is not None:
            raise NotImplementedError(
                'GATE bulk_q0_sphere_extra_chi_unavailable: the finite-q '
                'response needs a matched radial sphere integral; use the '
                'raw draw with explicit refinement controls')
        # ``analytic_sphere`` (head_minibz_average): add the analytic
        # Baldereschi-Tosatti sphere term to the q→0 head so vc0_mean is
        # seed-independent (the pure-Sobol mean has a few tiny δq → 8π/|δq|²
        # blow-ups that make it drift between seeds).  nmax 1→3 widens the
        # Voronoi fold (BGW ncell=3) for skewed cells.  Default False keeps
        # the historical pure-Sobol average bit-identical.
        nkx, nky, nkz = (int(s) for s in kgrid)
        batches = _sample_q0_minibz_qpoints(
            geometry, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, analytic_sphere=analytic_sphere, is_2d=False,
        )
        vc0_mean = self._vc0_mean(geometry, (nkx, nky, nkz), batches,
                                  analytic_sphere)

        if extra_chi is not None and S_cart is None:
            raise ValueError("q0_average: extra_chi adds to q.S.q; pass S_cart")
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
                values = 8.0 * jnp.pi / (q2 + kappa2)
                if analytic_sphere:
                    r2 = minibz_inscribed_sphere_r2(geometry.bvec, (nkx, nky, nkz))
                    values = jnp.where(q2 > r2, values, 0.0)
                wmeans.append(jnp.mean(values))
            wcoul0 = jnp.mean(jnp.stack(wmeans))
            if analytic_sphere:
                sphere = _analytic_sphere_bare_head(r2, geometry.cell_volume, nkx*nky*nkz)
                wcoul0 = wcoul0 + sphere*_thomas_fermi_sphere_factor(
                    np.sqrt(r2), np.sqrt(complex(np.asarray(kappa2))))
            return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

        if S_cart is not None:
            rows = None if extra_chi is None else (lambda rq: extra_chi(rq)[None])
            if analytic_sphere:
                r2 = minibz_inscribed_sphere_r2(geometry.bvec, (nkx, nky, nkz))
                wcoul0 = self._screened_means(batches, [S_cart], rows, outside_sphere=r2)[0]
                sphere = _analytic_sphere_bare_head(r2, geometry.cell_volume, nkx*nky*nkz)
                wcoul0 = wcoul0 + sphere*_interband_sphere_factor([S_cart])[0]
            else:
                wcoul0 = self._screened_means(batches, [S_cart], rows)[0]
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
        if analytic_sphere:
            # The incumbent epshead convention is a homogeneous scalar
            # multiplier. Preserve it on the same corrected bare average.
            wcoul0 = vc0_mean/(1.0 + 8.0*jnp.pi*gamma)
            return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)
        # Reuse the last batch's q-points for the screening-fallback estimator.
        rq_last = batches[-1]
        qsq = jnp.einsum("ij,ij->i", rq_last, rq_last)
        vq = 8.0 * jnp.pi / qsq
        wq = vq / (1.0 + vq * qsq * gamma)
        wcoul0 = jnp.mean(wq)
        return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

    def _vc0_mean(self, geometry, kgrid, batches, analytic_sphere):
        """``<v(q)>_mBZ`` on one draw (Baldereschi sphere split when asked)."""
        if analytic_sphere:
            nkx, nky, nkz = kgrid
            bvec = np.asarray(geometry.bvec, dtype=np.float64)
            q0sph2 = minibz_inscribed_sphere_r2(
                bvec, (nkx, nky, nkz), is_2d=False)
            n_kpts = int(nkx * nky * nkz)
            return jnp.asarray(minibz_average(
                np.zeros(3), [np.asarray(b) for b in batches],
                kind="bulk_3d", celvol=float(geometry.cell_volume),
                n_kpts=n_kpts,
                q0sph2=q0sph2, analytic_sphere=True), dtype=jnp.float64)
        # vc0_mean: average v(q) across all sampled q-points, mean over reps.
        means = [jnp.mean(self._vq_isotropic(rq)) for rq in batches]
        return jnp.mean(jnp.stack(means))

    def _screened_means(self, batches, S_carts, extra_chi_rows, *, shared=False,
                        outside_sphere=None):
        """Anisotropic screened ``w0 = <v / (1 - v (q.S.q + chi_extra(q)))>`` of every row, mean of batch means: [Z].

        ``extra_chi_rows(rq)`` returns every row's ``chi_extra`` [Z, n] on a
        set of points, or is None. One program per set of points serves all
        rows. ``shared`` (every process calls, as the head rows' average does):
        each process evaluates its own share of each batch's points on its
        device and the shares' sums are gathered once, in process order, so
        each process gets the same result for 1/P of the work; the metal
        head's Fermi-surface Lindhard term (velocity atoms x 2^18 points x
        rows x draws) was 13 s per map, repeated on every rank, at Ni 20^3.
        """
        S = jnp.asarray(np.stack([np.asarray(S, np.complex128) for S in S_carts]))
        if not shared or jax.process_count() == 1:
            means = []
            for rq in batches:
                rq = jnp.asarray(rq)
                extra = None if extra_chi_rows is None else extra_chi_rows(rq)
                if outside_sphere is None:
                    means.append(_screened_rows(rq, S, extra))
                else:
                    live = jnp.einsum('qi,qi->q', rq, rq) > outside_sphere
                    means.append(_screened_sums(rq, S, extra, live)/rq.shape[0])
            return jnp.mean(jnp.stack(means), axis=0)
        from jax.experimental import multihost_utils
        rank, ranks = jax.process_index(), jax.process_count()
        sums, counts = [], []
        for rq in batches:
            # One share length on every rank (the programs must agree): the
            # last share is padded with a repeated point that the sum masks.
            n = int(rq.shape[0])
            m = -(-n // ranks)
            idx = rank * m + np.arange(m)
            share = jnp.take(jnp.asarray(rq), jnp.asarray(np.minimum(idx, n - 1)), axis=0)
            live = jnp.asarray(idx < n)
            if outside_sphere is not None:
                live = live & (jnp.einsum('qi,qi->q', share, share) > outside_sphere)
            sums.append(_screened_sums(
                share, S, None if extra_chi_rows is None else extra_chi_rows(share),
                live))
            counts.append(n)
        total = np.asarray(multihost_utils.process_allgather(
            np.asarray(jnp.stack(sums), dtype=np.complex128), tiled=False)).sum(axis=0)
        return jnp.asarray(np.mean(total / np.asarray(counts, np.float64)[:, None], axis=0))

    def q0_average_screened(
        self, geometry: CoulombGeometry, kgrid, *,
        S_carts,
        extra_chi_rows=None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ):
        """``q0_average`` at many screened tensors on one draw: ``(vc0_mean, [wcoul0])``.

        ``extra_chi_rows(rq)`` returns every row's ``chi_extra`` [Z, n] on one
        draw batch (None: no extra term). Each draw batch is one program for
        all rows (a metal head has ~20 rows per SC map), so row ``i`` equals
        ``q0_average(S_cart=S_carts[i], extra_chi=...)`` up to round-off (the
        same draw and terms, regrouped sums). The device copy of the draw
        (``qmc_reps x nsamples x 3`` float64, 63 MB at the defaults) and one
        batch's [Z, n] complex rows live only for this call.
        """
        if analytic_sphere and extra_chi_rows is not None:
            raise NotImplementedError(
                'GATE bulk_q0_sphere_extra_chi_unavailable: the finite-q '
                'response needs a matched radial sphere integral; use the '
                'raw draw with explicit refinement controls')
        nkx, nky, nkz = (int(s) for s in kgrid)
        batches = _sample_q0_minibz_qpoints(
            geometry, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, analytic_sphere=analytic_sphere, is_2d=False,
        )
        vc0_mean = self._vc0_mean(geometry, (nkx, nky, nkz), batches,
                                  analytic_sphere).astype(jnp.complex128)
        S_carts = list(S_carts)
        if analytic_sphere:
            r2 = minibz_inscribed_sphere_r2(geometry.bvec, (nkx, nky, nkz))
            wcoul0 = self._screened_means(batches, S_carts, extra_chi_rows,
                shared=True, outside_sphere=r2)
            sphere = _analytic_sphere_bare_head(r2, geometry.cell_volume, nkx*nky*nkz)
            wcoul0 = wcoul0 + sphere*_interband_sphere_factor(S_carts)
        else:
            wcoul0 = self._screened_means(batches, S_carts, extra_chi_rows, shared=True)
        return vc0_mean, list(np.asarray(wcoul0, dtype=np.complex128))

    def q0_average_transverse_tensor(
        self, geometry: CoulombGeometry, kgrid, *,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ) -> np.ndarray:
        """``T_ab = ⟨v(q) t_ab(q̂)⟩_mBZ`` at q=Γ — the bare transverse-
        projector head (bispinor TT), bare units.  See
        :meth:`Slab2D.q0_average_transverse_tensor` for the physics.

        ``analytic_sphere=False`` (the default, matching :meth:`q0_average`)
        is a pure-Sobol mean of a ``1/q²``-singular integrand and inherits
        the SAME infinite-variance estimator problem the scalar 3D head has
        (measured tail index ``alpha≈1.5 < 2``, so ``sigma/sqrt(N)`` is not
        a valid error bar).  Pass ``True`` for a 3D bulk production head — it adds the
        isotropic Baldereschi-Tosatti sphere term exactly as
        :meth:`q0_average` does for the scalar case.  The 2D slab sibling
        is unaffected (marginal ``alpha=2``); this caveat is 3D-only.
        """
        nkx, nky, nkz = (int(s) for s in kgrid)
        bvec = np.asarray(geometry.bvec, dtype=np.float64)
        batches = _sample_q0_minibz_qpoints(
            geometry, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, analytic_sphere=analytic_sphere, is_2d=False)
        q0sph2 = minibz_inscribed_sphere_r2(bvec, (nkx, nky, nkz), is_2d=False)
        return minibz_transverse_head_avg(
            np.zeros(3), [np.asarray(b) for b in batches], kind="bulk_3d",
            celvol=float(geometry.cell_volume), n_kpts=int(nkx * nky * nkz),
            q0sph2=q0sph2, analytic_sphere=analytic_sphere, adaptive=True)

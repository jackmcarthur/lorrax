"""3D bulk Coulomb: ``v(q+G) = 8π/|q+G|²``, no truncation.

The scalar q=0 head and the packed photon completion have one quadrature
owner: :func:`vcoul.minibz.bulk_minibz_photon_cubature`.  Production consumes
its exact Wigner--Seitz-polyhedron/Duffy ladder.  The former scrambled-Sobol
draw, including its optional analytic-sphere split, survives only as the
explicit ``sobol_debug`` rule so historical values remain reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.base import SysDim, v_qG_single
from vcoul.geometry import CoulombGeometry
from vcoul.minibz import (Q0_RULE_SOBOL_DEBUG, _announce_q0_rule,
                          _sample_q0_minibz_qpoints,
                          bulk_minibz_photon_cubature, minibz_average,
                          minibz_inscribed_sphere_r2)

__all__ = ["Bulk3D", "BulkQ0Certificate", "BULK_Q0_RULE_EXACT"]


#: The exact bulk q=0 rule.  The dimension-specific name prevents a caller
#: from claiming the slab polygon rule was applied to a three-dimensional
#: cell; both exact spellings dispatch to the same receipt family.
BULK_Q0_RULE_EXACT = "wigner_seitz_polyhedron"
_BULK_Q0_RULES = (BULK_Q0_RULE_EXACT, Q0_RULE_SOBOL_DEBUG)
_Q0_LADDER_RTOL = 1.0e-8
_Q0_LADDER_ATOL = 1.0e-12


@dataclass(frozen=True)
class BulkQ0Certificate:
    """Public certificate for one exact bulk q=0 cell average."""

    dimension: int
    method: str
    orders: tuple[int, ...]
    physical_counts: tuple[int, ...]
    polyhedron_faces: int
    final_error_ratio: float
    mean_v: complex
    mean_w: complex


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
        rule: str = BULK_Q0_RULE_EXACT,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
        certificate_fn=None,
    ):
        """Return exact bulk ``(<v>, <W>)`` over the Gamma mini-BZ cell.

        ``wigner_seitz_polyhedron`` is the sole production rule and consumes
        the same authenticated receipt as the packed photon completion.
        ``sobol_debug`` reproduces the superseded scrambled-Sobol estimator;
        its ``analytic_sphere`` branch is debug-only and must be named.
        """
        if rule not in _BULK_Q0_RULES:
            raise ValueError(
                "GATE bulk_q0_rule_unknown: the bulk q0 cell average has "
                f"one production rule and one debug rule; got rule={rule!r}, "
                f"want one of {_BULK_Q0_RULES!r}.  Fix: drop the argument "
                f"to take {BULK_Q0_RULE_EXACT!r}.  "
                "doc: docs/services/vcoul.md.")
        if rule == Q0_RULE_SOBOL_DEBUG:
            _announce_q0_rule(
                "[vcoul] bulk q0_average rule = sobol_debug (DEBUG): "
                f"scrambled-Sobol Voronoi draw, nsamples={nsamples}, "
                f"method={method!r}, qmc_reps={qmc_reps}, "
                f"analytic_sphere={bool(analytic_sphere)}.  This is NOT the "
                "production rule: its 1/q^2 estimator has infinite variance "
                "(measured tail index about 1.5).  Production is "
                f"rule={BULK_Q0_RULE_EXACT!r}.", warn=True)
            return self._q0_average_sobol_debug(
                geometry, kgrid, S_cart=S_cart, epshead=epshead,
                static_kappa2=static_kappa2, nsamples=nsamples,
                method=method, qmc_reps=qmc_reps,
                analytic_sphere=analytic_sphere)
        if analytic_sphere:
            raise ValueError(
                "GATE bulk_q0_debug_rule_required: "
                "head_minibz_average/analytic_sphere selects the superseded "
                "Baldereschi-Tosatti + Sobol estimator; got production "
                f"rule={BULK_Q0_RULE_EXACT!r} with analytic_sphere=True.  "
                "Fix: unset head_minibz_average for the exact Wigner-Seitz "
                f"rule, or explicitly request rule={Q0_RULE_SOBOL_DEBUG!r} "
                "for a DEBUG reproduction.  doc: docs/services/vcoul.md.")
        return self._q0_average_wigner_seitz(
            geometry, kgrid, S_cart=S_cart, epshead=epshead,
            static_kappa2=static_kappa2, certificate_fn=certificate_fn)

    def _q0_average_wigner_seitz(
        self, geometry, kgrid, *, S_cart, epshead, static_kappa2,
        certificate_fn=None,
    ):
        """Evaluate every scalar-head model on the exact receipt ladder."""
        if static_kappa2 is not None and S_cart is not None:
            raise ValueError(
                "q0_average accepts either static_kappa2 or S_cart, not both")
        base_kernel = self if type(self) is Bulk3D else Bulk3D()
        receipt = bulk_minibz_photon_cubature(
            base_kernel, geometry, tuple(int(value) for value in kgrid))

        S3 = None if S_cart is None else np.asarray(
            S_cart, dtype=np.complex128)
        if S3 is not None and S3.shape != (3, 3):
            raise ValueError(f"bulk S_cart must be (3,3); got {S3.shape}")
        kappa2 = None
        if static_kappa2 is not None:
            kappa2 = np.asarray(static_kappa2, dtype=np.complex128)
            if (kappa2.ndim != 0 or not np.isfinite(kappa2)
                    or float(np.real(kappa2)) <= 0.0):
                raise ValueError("static_kappa2 must be one positive scalar")
        if S3 is None and kappa2 is None and epshead is None:
            raise ValueError(
                "GATE bulk_q0_head_source: the bulk q0 screened head needs "
                "S_cart, static_kappa2, or epshead; all are None")

        gamma = None
        if S3 is None and kappa2 is None:
            bvec = np.asarray(geometry.bvec, dtype=np.float64)
            q0_cart = np.asarray((0.001, 0.0, 0.0)) @ bvec
            q0sq = float(q0_cart @ q0_cart)
            vc_q0 = 8.0 * np.pi / q0sq
            eps_real = float(np.real(np.asarray(epshead)))
            gamma = (1.0 / eps_real - 1.0) / (q0sq * vc_q0)

        ladder = []
        for chunk in receipt.chunks:
            n_valid = int(chunk.physical_count)
            weight = np.asarray(
                chunk.sample_weight[:n_valid], dtype=np.float64)
            q_cart = np.asarray(
                chunk.q_cart[:n_valid], dtype=np.float64)
            v = np.asarray(
                chunk.D_raw[:n_valid, 0, 0], dtype=np.float64)
            measure = float(np.sum(weight))
            if not np.isfinite(measure) or measure <= 0.0:
                raise ValueError(
                    "GATE bulk_q0_polyhedron_measure: cubature measure got "
                    f"{measure!r} at order={chunk.order}; want finite > 0")
            q2 = np.einsum("qi,qi->q", q_cart, q_cart, optimize=True)
            if kappa2 is not None:
                screened = 8.0 * np.pi / (q2 + kappa2)
            elif S3 is not None:
                qSq = np.einsum(
                    "qi,ij,qj->q", q_cart, S3, q_cart, optimize=True)
                screened = v / (1.0 - v * qSq)
            else:
                screened = v / (1.0 + v * q2 * gamma)
            ladder.append((
                int(chunk.order),
                complex(np.sum(weight * v) / measure),
                complex(np.sum(weight * screened) / measure),
            ))

        certificate = self._require_q0_ladder_converged(receipt, ladder)
        if certificate_fn is not None:
            certificate_fn(certificate)
        vc0_mean, wcoul0 = ladder[-1][1:]
        if not (np.isfinite(vc0_mean) and np.isfinite(wcoul0)):
            raise ValueError(
                "GATE bulk_q0_polyhedron_nonfinite: q0 cell average got "
                f"<v>={vc0_mean!r}, <W>={wcoul0!r}; want both finite")
        return (jnp.asarray(vc0_mean, dtype=jnp.complex128),
                jnp.asarray(wcoul0, dtype=jnp.complex128))

    @staticmethod
    def _require_q0_ladder_converged(receipt, ladder) -> BulkQ0Certificate:
        """Apply the packed completion's mixed absolute/relative discipline."""
        adjacent_ratios = []
        for index in range(1, len(ladder)):
            ratios = []
            for slot in (1, 2):
                previous, current = ladder[index - 1][slot], ladder[index][slot]
                delta = abs(current - previous)
                scale = max(abs(previous), abs(current))
                values = np.asarray((delta, scale), dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        "GATE bulk_q0_polyhedron_nonfinite: ladder "
                        f"diagnostics are nonfinite at orders "
                        f"{ladder[index - 1][0]}->{ladder[index][0]}")
                ratios.append(
                    delta / (_Q0_LADDER_ATOL + _Q0_LADDER_RTOL * scale))
            adjacent_ratios.append(max(ratios))
        if len(adjacent_ratios) != 2:
            raise ValueError(
                "GATE bulk_q0_polyhedron_nonfinite: the fixed three-order "
                "ladder must produce two adjacent-order error ratios")
        if adjacent_ratios[-1] > 1.0:
            raise ValueError(
                "GATE bulk_q0_polyhedron_not_converged: the final "
                f"{receipt.orders[-2]}->{receipt.orders[-1]} order pair "
                "missed the mixed absolute+relative budget: error_ratio="
                f"{adjacent_ratios[-1]:.3e} > 1")
        _announce_q0_rule(
            f"[vcoul] bulk q0_average rule = {BULK_Q0_RULE_EXACT} "
            f"(production): exact Wigner-Seitz polyhedron, "
            f"{len(receipt.polytope_faces)} faces, orders {receipt.orders}, "
            f"nodes {receipt.physical_counts}; <v>={ladder[-1][1].real:.9g} "
            f"a.u.; <W>={ladder[-1][2].real:.9g} a.u.; ladder "
            f"error_ratio={adjacent_ratios[-1]:.3e} (<= 1 required).  Same "
            "receipt the packed Gamma completion consumes.")
        return BulkQ0Certificate(
            dimension=3,
            method=BULK_Q0_RULE_EXACT,
            orders=tuple(int(value) for value in receipt.orders),
            physical_counts=tuple(
                int(value) for value in receipt.physical_counts),
            polyhedron_faces=len(receipt.polytope_faces),
            final_error_ratio=float(adjacent_ratios[-1]),
            mean_v=complex(ladder[-1][1]),
            mean_w=complex(ladder[-1][2]),
        )

    def _q0_average_sobol_debug(
        self, geometry, kgrid, *, S_cart, epshead, static_kappa2,
        nsamples, method, qmc_reps, analytic_sphere,
    ):
        """The byte-stable pre-migration bulk estimator, DEBUG only."""
        # ``analytic_sphere`` adds the analytic Baldereschi-Tosatti sphere
        # term and widens the Voronoi fold.  Both policies are retained here
        # exactly for measurement/reproduction, never selected implicitly.
        nkx, nky, nkz = (int(s) for s in kgrid)
        batches = _sample_q0_minibz_qpoints(
            geometry, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, analytic_sphere=analytic_sphere, is_2d=False,
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
            wmeans = []
            for rq in batches:
                vq = self._vq_isotropic(rq).astype(jnp.complex128)
                qSq = jnp.einsum('qi,ij,qj->q', rq, S, rq)
                wmeans.append(jnp.mean(vq / (1.0 - vq * qSq)))
            wcoul0 = jnp.mean(jnp.stack(wmeans))
            return vc0_mean.astype(jnp.complex128), wcoul0.astype(jnp.complex128)

        # Isotropic Ismail-Beigi gamma fallback (epshead-driven).  Kept for
        # back-compat with older runs that don't compute the dipole tensor.
        if epshead is None:
            raise ValueError(
                "bulk Sobol DEBUG rule needs S_cart, static_kappa2, or epshead")
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

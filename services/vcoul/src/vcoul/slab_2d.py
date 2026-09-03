"""2D slab (Ismail-Beigi) Coulomb truncation along the c axis.

  v_2D(q+G) = (8π/|q+G|²) · (1 − exp(−zc·|q‖+G‖|) cos((qz+Gz)·zc)),  zc = π/b_z

:func:`~vcoul.minibz.sample_minibz_qpoints` already sets ``qz=0`` when
``is_2d``; this module just supplies the formula.

THE q→0 CELL AVERAGE HAS ONE OWNER (2026-09-01).  ``⟨v⟩`` and
``⟨v/(1 − v qᵀSq)⟩`` at Γ are evaluated on the exact Wigner--Seitz
Duffy--Gauss ladder issued by
:func:`~vcoul.minibz.slab_minibz_photon_cubature` — the same provider
receipt, the same nodes and the same weights the packed bispinor Γ
completion (``gw.head_correction.complete_static_photon_q0``)
consumes, so the scalar route and the packed completion evaluate the same
integral with the same rule.  The historical scrambled-Sobol draw is the
named DEBUG rule ``sobol_debug``: it carries a ~0.1–0.2 % sampling error
on the ``|q|`` cusp that is deterministic per seed and therefore invisible
as noise (+5.72 meV per occupied state on MoS2 3×3, lane J claim 0586).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.base import SysDim, v_qG_single
from vcoul.geometry import CoulombGeometry
from vcoul.minibz import (_sample_q0_minibz_qpoints, minibz_average,
                          Q0_RULE_SOBOL_DEBUG, _announce_q0_rule,
                          minibz_inscribed_sphere_r2,
                          minibz_voronoi_batches,
                          slab_minibz_photon_cubature)

__all__ = ["Slab2D", "SlabQ0Certificate", "Q0_RULE_EXACT",
           "Q0_RULE_SOBOL_DEBUG"]

#: The production q→0 cell-average rule: the exact mini-lattice
#: Wigner--Seitz polygon, Γ-to-edge Duffy triangulation, fixed 16/24/32
#: Gauss--Legendre ladder.  ONE owner, shared with the packed completion.
Q0_RULE_EXACT = "wigner_seitz_polygon"
_Q0_RULES = (Q0_RULE_EXACT, Q0_RULE_SOBOL_DEBUG)

# Same mixed absolute+relative budget the packed completion applies to its
# own polygon ladder (``gw.head_correction``
# ``_STATIC_PHOTON_POLYGON_CONVERGENCE_{ATOL,RTOL}``); the two consumers
# assert convergence of the same provider ladder, so the budget is one
# number written twice rather than two policies.
_Q0_LADDER_RTOL = 1.0e-8
_Q0_LADDER_ATOL = 1.0e-12

@dataclass(frozen=True)
class SlabQ0Certificate:
    """Public certificate for one exact slab q=0 cell average."""

    dimension: int
    method: str
    orders: tuple[int, ...]
    physical_counts: tuple[int, ...]
    polygon_edges: int
    final_error_ratio: float
    mean_v: complex
    mean_w: complex | None


class Slab2D:
    sys_dim = SysDim.SLAB_2D

    @staticmethod
    def _truncation_half_height_from_bvec(bvec) -> float:
        """Return the incumbent ``zc = pi / b3z`` slab convention.

        The Ismail--Beigi kernel in this class is defined only for the
        repository's Cartesian orientation: ``b1,b2`` lie in ``xy`` and
        ``b3`` points along ``+z``.  A negative ``b3z`` is not another
        spelling of the same supported geometry; taking ``abs(b3z)`` would
        silently invent an orientation convention outside this owner.
        """
        bvec = np.asarray(bvec, dtype=np.float64)
        if bvec.shape != (3, 3) or not np.all(np.isfinite(bvec)):
            raise ValueError(
                "Slab2D requires finite (3,3) Cartesian reciprocal rows")
        in_plane_scale = max(
            float(np.linalg.norm(bvec[0, :2])),
            float(np.linalg.norm(bvec[1, :2])), 1.0)
        if max(abs(float(bvec[0, 2])), abs(float(bvec[1, 2]))) > (
                1.0e-12 * in_plane_scale):
            raise ValueError(
                "Slab2D requires b1 and b2 in the Cartesian xy plane")
        b3_scale = max(in_plane_scale, abs(float(bvec[2, 2])), 1.0)
        if (float(bvec[2, 2])
                <= 128.0 * np.finfo(np.float64).eps * b3_scale
                or np.linalg.norm(bvec[2, :2]) > 1.0e-12 * b3_scale):
            raise ValueError(
                "Slab2D requires a finite nonzero b3 directed along "
                "Cartesian +z; reversed or tilted slab orientation is "
                "unsupported")
        return float(np.pi / float(bvec[2, 2]))

    @classmethod
    def truncation_half_height(cls, geometry: CoulombGeometry) -> float:
        """Return the supported slab's Ismail--Beigi half-height."""
        return cls._truncation_half_height_from_bvec(geometry.bvec)

    def _v_bare_per_q(self, qf, gvec_q, *, bvec_f, fact,
                      bdot=None, fft_grid=None):
        """Ismail-Beigi slab truncation.  See the base Protocol.

        Arithmetic order is the shipped production order — ``v_reg * fact``
        rather than ``v / cell_volume``, and ``sqrt(x² + y²)`` rather than
        ``linalg.norm``; both differ from the old class spelling in the
        last ulp, and this path is bit-compared against the pre-port table.
        """
        del bdot, fft_grid
        zc = self._truncation_half_height_from_bvec(bvec_f)
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
        zc = self.truncation_half_height(geometry)
        shift_cart = np.asarray(shift_frac, dtype=np.float64) @ bvec
        dq = minibz_voronoi_batches(
            bvec, (nkx, nky, nkz), nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, nmax=3, is_2d=True)
        q0sph2 = minibz_inscribed_sphere_r2(bvec, (nkx, nky, nkz), is_2d=True)
        return minibz_average(
            shift_cart, dq, kind=kind, celvol=float(geometry.cell_volume),
            n_kpts=int(nkx * nky * nkz), q0sph2=q0sph2,
            alpha=alpha, zc=zc,
            analytic_sphere=False, adaptive=True, n_coarse=n_coarse)

    def q0_average(
        self, geometry: CoulombGeometry, kgrid, *,
        S_cart=None,
        epshead=None,
        static_kappa2=None,
        rule: str = Q0_RULE_EXACT,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
        certificate_fn=None,
    ):
        """``(⟨v⟩, ⟨v/(1 − v qᵀSq)⟩)`` over the Γ mini-BZ cell, bare units.

        ONE OWNER.  ``rule`` is a named selection with no silent
        alternative:

        * ``"wigner_seitz_polygon"`` (default, PRODUCTION) — the exact
          mini-lattice Wigner--Seitz polygon cubature issued by
          :func:`~vcoul.minibz.slab_minibz_photon_cubature`.  The scalar
          charge head and the packed bispinor Γ completion therefore
          evaluate the same integral on the same nodes and weights.  The
          16/24/32 ladder's last pair must converge under the mixed
          absolute+relative budget or the call REFUSES; the certificate is
          announced once per process.
        * ``"sobol_debug"`` — DEBUG/DIAGNOSTIC ONLY, the historical
          scrambled-Sobol Voronoi draw, kept so the superseded estimator
          stays reproducible.  ``nsamples``/``method``/``qmc_reps`` and
          ``analytic_sphere`` are ITS dials and have no meaning under the
          production rule, which is exact and takes no order knob.  It
          announces itself on stdout.

        ``analytic_sphere`` is the deck's ``head_minibz_average`` key.  For
        the slab it only ever widened the Sobol draw's Voronoi fold
        (``nmax`` 1→3); the exact rule integrates the Wigner--Seitz cell
        itself, so there is no fold to widen and the key is REFUSED rather
        than silently ignored.

        ``epshead`` (no ``S_cart``) keeps the historical Ismail--Beigi
        gamma model for ``wcoul0``, now evaluated on the same exact nodes.
        """
        if static_kappa2 is not None:
            raise NotImplementedError(
                "static_kappa2 is the 3D Thomas-Fermi model; the 2D metallic "
                "q->0 form requires its separate |q| kernel")
        if rule not in _Q0_RULES:
            raise ValueError(
                "GATE slab_q0_rule_unknown: the slab q->0 cell average has "
                f"one production rule and one debug rule; got rule={rule!r}, "
                f"want one of {_Q0_RULES!r}.  Fix: drop the argument to take "
                f"{Q0_RULE_EXACT!r}.  doc: docs/services/vcoul.md.")
        if rule == Q0_RULE_SOBOL_DEBUG:
            _announce_q0_rule(
                "[vcoul] q0_average rule = sobol_debug (DEBUG): scrambled-"
                f"Sobol Voronoi draw, nsamples={nsamples}, method={method!r}, "
                f"qmc_reps={qmc_reps}, analytic_sphere={bool(analytic_sphere)}."
                "  This is NOT the production rule; it carries a ~0.1-0.2 % "
                "sampling error on the |q| cusp (claim 0586).  Production is "
                f"rule={Q0_RULE_EXACT!r}.", warn=True)
            return self._q0_average_sobol_debug(
                geometry, kgrid, S_cart=S_cart, epshead=epshead,
                nsamples=nsamples, method=method, qmc_reps=qmc_reps,
                analytic_sphere=analytic_sphere)
        if analytic_sphere:
            raise ValueError(
                "GATE slab_q0_analytic_sphere_unavailable: "
                "head_minibz_average/analytic_sphere widened the Sobol "
                "draw's Voronoi fold; got analytic_sphere=True, want False. "
                f"The production rule {Q0_RULE_EXACT!r} integrates the exact "
                "Wigner-Seitz cell, so there is no fold to widen and no "
                "Baldereschi-Tosatti sphere in 2D (the slab head is a |q| "
                "cusp, not a 1/q^2 pole).  Fix: unset head_minibz_average on "
                f"a sys_dim=2 deck, or ask for rule={Q0_RULE_SOBOL_DEBUG!r} "
                "explicitly.  doc: docs/services/vcoul.md.")
        return self._q0_average_wigner_seitz(
            geometry, kgrid, S_cart=S_cart, epshead=epshead,
            certificate_fn=certificate_fn)

    def _q0_average_wigner_seitz(
        self, geometry: CoulombGeometry, kgrid, *, S_cart, epshead,
        certificate_fn=None,
    ):
        """The production rule: the provider's exact WS/Duffy ladder.

        Nothing is re-derived here.  The nodes ``q``, the normalized
        weights ``w`` and the kernel value ``v = D_raw[:, 0, 0]`` are read
        straight off the authenticated receipt the packed completion
        consumes, so ``⟨v⟩`` here and ``bare_D_mean[0, 0]`` there are the
        same reduction of the same numbers.
        """
        # ``slab_minibz_photon_cubature`` authenticates the EXACT service
        # kernel type (a lorrax (wfn, meta)-facing subclass is not it), so
        # the canonical kernel is constructed when ``self`` is a subclass.
        # ``_v_bare_per_q`` is inherited, so this is the same arithmetic.
        base_kernel = self if type(self) is Slab2D else Slab2D()
        receipt = slab_minibz_photon_cubature(
            base_kernel, geometry, tuple(int(s) for s in kgrid))
        S2 = None if S_cart is None else np.asarray(
            S_cart, dtype=np.complex128)
        ladder = []
        for chunk in receipt.chunks:
            n_valid = int(chunk.physical_count)
            weight = np.asarray(
                chunk.sample_weight[:n_valid], dtype=np.float64)
            q_cart = np.asarray(chunk.q_cart[:n_valid], dtype=np.float64)
            v = np.asarray(chunk.D_raw[:n_valid, 0, 0], dtype=np.float64)
            measure = float(np.sum(weight))
            if not np.isfinite(measure) or measure <= 0.0:
                raise ValueError(
                    "GATE slab_q0_polygon_measure: cubature_measure got: "
                    f"{measure!r} at order={chunk.order}; want: finite and "
                    "> 0; why: the slab q->0 cell average requires a "
                    "positive normalized polygon measure.")
            vc0 = complex(np.sum(weight * v) / measure)
            if S2 is None:
                wcoul0 = None
            else:
                qSq = np.einsum(
                    "qi,ij,qj->q", q_cart, S2, q_cart, optimize=True)
                wcoul0 = complex(
                    np.sum(weight * (v / (1.0 - v * qSq))) / measure)
            ladder.append((int(chunk.order), vc0, wcoul0))
        vc0_mean = ladder[-1][1]
        wcoul0 = ladder[-1][2]
        certificate = self._require_q0_ladder_converged(receipt, ladder)
        if certificate_fn is not None:
            certificate_fn(certificate)
        if wcoul0 is None:
            wcoul0 = self._q0_epshead_gamma_average(
                geometry, receipt, epshead)
        if not (np.isfinite(vc0_mean) and np.isfinite(wcoul0)):
            raise ValueError(
                "GATE slab_q0_polygon_nonfinite: q0_cell_average got: "
                f"<v>={vc0_mean!r}, <W>={wcoul0!r}; want: both finite; "
                "why: non-finite Coulomb heads cannot enter screening.")
        return (jnp.asarray(vc0_mean, dtype=jnp.complex128),
                jnp.asarray(wcoul0, dtype=jnp.complex128))

    @staticmethod
    def _require_q0_ladder_converged(receipt, ladder) -> SlabQ0Certificate:
        """Refuse an unconverged ladder; announce the certificate once."""
        ratios = []
        for index in range(1, len(ladder)):
            for slot in (1, 2):
                previous, current = ladder[index - 1][slot], ladder[index][slot]
                if previous is None or current is None:
                    continue
                delta = abs(current - previous)
                scale = max(abs(previous), abs(current))
                values = np.asarray(
                    (delta, scale, abs(current)), dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        "GATE slab_q0_polygon_nonfinite: ladder_diagnostics "
                        f"got: delta={delta!r}, scale={scale!r}, "
                        f"current_abs={abs(current)!r}; want: all finite; "
                        "why: the fixed cubature ladder needs a valid "
                        "convergence certificate.")
                ratios.append(
                    delta / (_Q0_LADDER_ATOL + _Q0_LADDER_RTOL * scale))
        if not ratios:
            raise ValueError(
                "GATE slab_q0_polygon_nonfinite: comparable_ladder_pairs "
                f"got: 0 from orders={receipt.orders!r}; want: at least one; "
                "why: the slab q->0 average needs a convergence comparison.")
        if ratios[-1] > 1.0:
            raise ValueError(
                "GATE slab_q0_polygon_not_converged: the final polygon "
                "Duffy--Gauss order pair "
                f"{receipt.orders[-2]}->{receipt.orders[-1]} did not converge "
                "the slab q->0 cell average under the mixed absolute+relative "
                f"budget: error_ratio={ratios[-1]:.3e} > 1 "
                f"(atol={_Q0_LADDER_ATOL:.1e}, rtol={_Q0_LADDER_RTOL:.1e}).  "
                "The provider ladder is fixed; refusing the head rather than "
                "accepting a caller dial.")
        _announce_q0_rule(
            f"[vcoul] q0_average rule = {Q0_RULE_EXACT} (production): exact "
            f"Wigner-Seitz polygon, {len(receipt.polytope_vertices)} edges, "
            f"orders {receipt.orders}, nodes {receipt.physical_counts}; "
            f"<v>={ladder[-1][1].real:.9g} a.u.; ladder error_ratio="
            f"{ratios[-1]:.3e} (<= 1 required).  Same receipt the packed "
            "Gamma completion consumes.")
        return SlabQ0Certificate(
            dimension=2,
            method=Q0_RULE_EXACT,
            orders=tuple(int(v) for v in receipt.orders),
            physical_counts=tuple(int(v) for v in receipt.physical_counts),
            polygon_edges=len(receipt.polytope_vertices),
            final_error_ratio=float(ratios[-1]),
            mean_v=complex(ladder[-1][1]),
            mean_w=(None if ladder[-1][2] is None
                    else complex(ladder[-1][2])),
        )

    @staticmethod
    def _q0_epshead_gamma_average(geometry, receipt, epshead):
        """Historical Ismail-Beigi gamma model, on the exact nodes."""
        if epshead is None:
            raise ValueError(
                "GATE slab_q0_head_source: the slab q->0 screened head needs "
                "either S_cart (the anisotropic q^2 tensor) or epshead (the "
                "Ismail-Beigi gamma model); both are None")
        bvec = np.asarray(geometry.bvec, dtype=np.float64)
        zc = float(receipt.slab_zc)
        q0_cart = np.asarray((0.001, 0.0, 0.0), dtype=np.float64) @ bvec
        q0len = float(np.linalg.norm(q0_cart))
        vc_q0 = (1.0 - np.exp(-q0len * zc)) / (q0len * q0len)
        eps_real = float(np.real(np.asarray(epshead)))
        gamma = (1.0 / eps_real - 1.0) / ((q0len * q0len) * vc_q0)
        chunk = receipt.chunks[-1]
        n_valid = int(chunk.physical_count)
        weight = np.asarray(chunk.sample_weight[:n_valid], dtype=np.float64)
        kxy = np.linalg.norm(
            np.asarray(chunk.q_cart[:n_valid, :2], dtype=np.float64), axis=1)
        vc_q = (1.0 - np.exp(-kxy * zc)) / (kxy * kxy)
        wq = vc_q / (1.0 + vc_q * (kxy * kxy) * gamma)
        return complex(
            8.0 * np.pi * float(np.sum(weight * wq) / np.sum(weight)))

    def _q0_average_sobol_debug(
        self, geometry: CoulombGeometry, kgrid, *,
        S_cart, epshead, nsamples: int, method: str, qmc_reps: int,
        analytic_sphere: bool,
    ):
        """DEBUG RULE ONLY — the superseded scrambled-Sobol Voronoi draw.

        Byte-for-byte the estimator that shipped before 2026-09-01, kept so
        the superseded numbers stay reproducible (claim 0586 measured it
        against the exact rule).  It has NO production caller; see the
        module docstring for why.
        """
        nkx, nky, nkz = (int(s) for s in kgrid)
        bvec = jnp.asarray(geometry.bvec, dtype=jnp.float64)
        self.truncation_half_height(geometry)  # orientation refusal
        zc = jnp.pi / bvec[2, 2]  # retain the incumbent device arithmetic

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

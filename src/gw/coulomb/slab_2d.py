"""COMPAT SHIM — 2D slab (Ismail-Beigi) Coulomb truncation along the c axis.

  v_2D(q+G) = (8π/|q+G|²) · (1 − exp(−zc·|q‖+G‖|) cos((qz+Gz)·zc)),  zc = π/b_z

The arithmetic moved to :class:`vcoul.Slab2D` (2026-08-07); this is the
``(wfn, meta)``-facing adapter.  ``_v_bare_per_q`` is INHERITED.

``v_head_minibz_avg`` is adapted too, and it is still UNWIRED (zero
callers): it is the finite-shift 2D cell average §16.4 flagged missing,
a capability awaiting a caller rather than a duplicate.
"""
from __future__ import annotations

import jax

from common import Meta
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import vcoul                                                # noqa: E402
from vcoul import CoulombGeometry                           # noqa: E402

__all__ = ["Slab2D"]


class Slab2D(vcoul.Slab2D):
    """``(wfn, meta)``-facing :class:`vcoul.Slab2D`."""

    def v_qG(self, wfn, qvec_wrapped, comps_qG) -> jax.Array:
        return super().v_qG(CoulombGeometry.from_wfn(wfn), qvec_wrapped,
                            comps_qG)

    def v_head_minibz_avg(
        self, wfn, meta: Meta, shift_frac, *,
        alpha: float | None = None,
        kind: str = "slab",
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        n_coarse: int = 250_000,
    ) -> float:
        return super().v_head_minibz_avg(
            CoulombGeometry.from_wfn(wfn),
            (meta.nkx, meta.nky, meta.nkz), shift_frac,
            alpha=alpha, kind=kind, nsamples=nsamples, method=method,
            qmc_reps=qmc_reps, n_coarse=n_coarse,
        )

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,
        epshead=None,
        static_kappa2=None,
        rule: str = vcoul.Q0_RULE_EXACT,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ):
        """``(wfn, meta)``-facing :meth:`vcoul.Slab2D.q0_average`.

        ``rule`` is forwarded verbatim; the service owns the selection and
        its refusals.  The default is the exact Wigner--Seitz polygon
        cubature, so ``nsamples``/``method``/``qmc_reps`` — which every
        deck-facing wrapper still threads through with their historical
        values — configure the named ``sobol_debug`` rule and nothing else.
        """
        return super().q0_average(
            CoulombGeometry.from_wfn(wfn),
            (meta.nkx, meta.nky, meta.nkz),
            S_cart=S_cart, epshead=epshead, static_kappa2=static_kappa2,
            rule=rule,
            nsamples=nsamples,
            method=method, qmc_reps=qmc_reps,
            analytic_sphere=analytic_sphere,
        )

"""COMPAT SHIM — 0D cell-box truncation (Wigner-Seitz real-space FFT).

The arithmetic moved to :class:`vcoul.Box0D` (2026-08-07); this is the
``(wfn, meta)``-facing adapter.  ``_v_bare_per_q`` — including the q≠0
REFUSAL and the "G=0 is not zeroed here" convention — is INHERITED from
the service class, not re-spelled.

Note the default arguments: ``nsamples`` / ``method`` / ``qmc_reps``
default to ``None`` here rather than to the 3D/2D values, because a box
head is finite from the FFT and never samples.  That was the shipped
signature and it is preserved exactly.
"""
from __future__ import annotations

import jax

from common import Meta
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import vcoul                                                # noqa: E402
from vcoul import CoulombGeometry                           # noqa: E402

__all__ = ["Box0D"]


class Box0D(vcoul.Box0D):
    """``(wfn, meta)``-facing :class:`vcoul.Box0D`."""

    def v_qG(self, wfn, qvec_wrapped, comps_qG) -> jax.Array:
        return super().v_qG(CoulombGeometry.from_wfn(wfn), qvec_wrapped,
                            comps_qG)

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,        # ignored (box truncation: head is finite already)
        epshead=None,       # ignored (no screening correction at the head)
        static_kappa2=None, # ignored (no metallic long-wave limit in 0D)
        nsamples=None,
        method=None,
        qmc_reps=None,
        analytic_sphere=False,  # ignored (box head is finite; no mini-BZ avg)
    ):
        """Box: V(q=0, G=0) is finite from the WS-truncated FFT.

        BGW convention: ``wcoul0 = vc0`` for box truncation.  Screening
        enters only through the body of the dielectric matrix; the head
        is left untouched.  See BGW Common/vcoul_generator.f90:717.
        """
        del meta        # a box is Gamma-only; the service takes no kgrid here
        return super().q0_average(
            CoulombGeometry.from_wfn(wfn),
            S_cart=S_cart, epshead=epshead, static_kappa2=static_kappa2,
            nsamples=nsamples,
            method=method, qmc_reps=qmc_reps,
            analytic_sphere=analytic_sphere,
        )

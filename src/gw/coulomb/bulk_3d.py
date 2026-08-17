"""COMPAT SHIM — 3D bulk Coulomb, ``v(q+G) = 8π/|q+G|²``, no truncation.

The arithmetic moved to :class:`vcoul.Bulk3D` (2026-08-07).  What is left
is a two-method ADAPTER: the service kernel takes a
:class:`vcoul.CoulombGeometry` and an explicit ``kgrid``; this subclass
takes the ``wfn`` loader and the lorrax ``common.Meta`` that every call
site in the tree still writes, and translates.

``_v_bare_per_q`` — the dimension's one arithmetic method — is INHERITED,
not re-spelled.  That is the point of subclassing rather than wrapping: a
copy here would be the tenth ``8π/|q+G|²`` in the tree and could drift
from the service's by an ulp with nothing to catch it.
"""
from __future__ import annotations

import jax

from common import Meta
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import vcoul                                                # noqa: E402
from vcoul import CoulombGeometry                           # noqa: E402

__all__ = ["Bulk3D"]


class Bulk3D(vcoul.Bulk3D):
    """``(wfn, meta)``-facing :class:`vcoul.Bulk3D`."""

    def v_qG(self, wfn, qvec_wrapped, comps_qG) -> jax.Array:
        return super().v_qG(CoulombGeometry.from_wfn(wfn), qvec_wrapped,
                            comps_qG)

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,
        epshead=None,
        static_kappa2=None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ):
        return super().q0_average(
            CoulombGeometry.from_wfn(wfn),
            (meta.nkx, meta.nky, meta.nkz),
            S_cart=S_cart, epshead=epshead, static_kappa2=static_kappa2,
            nsamples=nsamples,
            method=method, qmc_reps=qmc_reps,
            analytic_sphere=analytic_sphere,
        )

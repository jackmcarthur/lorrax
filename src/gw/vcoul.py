"""COMPAT SHIM — Coulomb q=0 mini-BZ averaging + Voronoi-cell point wrapping.

Both names moved to the ``vcoul`` service (2026-08-07):

  * :func:`wrap_points_to_voronoi` → :func:`vcoul.wrap_points_to_voronoi`,
    re-exported here verbatim (same jitted function object) because
    :mod:`bse.vq_interp` and :mod:`gw.compute_vcoul` still import it from
    this path.
  * :func:`compute_q0_averages` → the ``q0_average`` method on any
    :func:`vcoul.get_kernel` kernel.  What stays here is the DECK-FACING
    translation: ``wfn`` + ``common.Meta`` in, ``(vc0, wcoul0)`` out.
    ``gw.head_correction``'s HeadResolver is the caller, in both the
    epshead and the anisotropic-S(ω) flavour.
"""

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import jax.numpy as jnp                                     # noqa: E402

from common import Meta                                     # noqa: E402
from vcoul import wrap_points_to_voronoi                    # noqa: E402,F401

__all__ = ["wrap_points_to_voronoi", "compute_q0_averages"]


def compute_q0_averages(
	wfn,
	epshead,
	meta: Meta,
	S_cart: jnp.ndarray | None = None,
	nsamples: int = 2**18,
	method: str = "auto",
	qmc_reps: int = 10,
	analytic_sphere: bool = False,
):
	"""Compute q=0 averages (vc0_mean, wcoul0) for the system's dimensionality.

	Thin compatibility wrapper around the dimension-aware ``CoulombKernel``
	in :mod:`gw.coulomb`, which adapts :mod:`vcoul`'s.  ``analytic_sphere``
	(``head_minibz_average``) adds the Baldereschi-Tosatti analytic sphere
	term (3D) / widens the Voronoi fold (both dims) to the q→0 head; default
	False keeps the historical pure-Sobol average bit-identical.

	``method`` DEFAULTS TO ``"auto"`` since the extraction (it was
	``"sobol"``).  With scipy present — which is every production machine,
	and the ``sobol`` extra declares it — ``auto`` resolves to exactly the
	same scrambled-Sobol draw, so every number here is unchanged.  What
	changes is the machine WITHOUT scipy: it used to fall back to a uniform
	draw with ``nsamples`` silently raised to 2.5e6, and now it does the
	same thing while SAYING so once (``vcoul.minibz``'s announce-or-refuse
	gate).  A caller that wants the old silence back should not have it; a
	caller that wants a refusal instead passes ``method="sobol"``.
	"""
	from .coulomb import get_kernel
	return get_kernel(getattr(meta, 'sys_dim', None)).q0_average(
		wfn, meta,
		S_cart=S_cart, epshead=epshead,
		nsamples=nsamples, method=method, qmc_reps=qmc_reps,
		analytic_sphere=analytic_sphere,
	)

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
	static_kappa2: jnp.ndarray | None = None,
	nsamples: int = 2**18,
	method: str = "auto",
	qmc_reps: int = 10,
	analytic_sphere: bool = False,
	certificate_fn=None,
):
	"""Compute q=0 averages (vc0_mean, wcoul0) for the system's dimensionality.

	Thin compatibility wrapper around the dimension-aware ``CoulombKernel``
	in :mod:`gw.coulomb`, which adapts :mod:`vcoul`'s.  ``analytic_sphere``
	(``head_minibz_average``) now names a retired Sobol/Baldereschi debug
	policy and is refused by both exact production rules.  The historical
	estimator remains directly reachable only as the service's explicit
	``sobol_debug`` rule; this deck-facing wrapper never selects it.

	The slab and bulk kernels run the exact Wigner--Seitz polygon/polyhedron
	cubatures shared with their packed bispinor Gamma completions.  Neither
	takes a sample count, sequence, or replicate count in production;
	``nsamples``/``method``/``qmc_reps`` configure ``sobol_debug`` only.

	``method`` defaults to ``"auto"`` for compatibility, but affects only
	the quarantined debug estimator.  The exact production rules have no
	random sequence, seed, replicate count, or sample-count control.

	For a 3D metal at exactly zero frequency, ``static_kappa2`` selects the
	Thomas-Fermi order of limits, ``<8*pi/(q^2+kappa_TF^2)>``.  It is mutually
	exclusive with the finite-frequency ``S_cart`` representation.
	"""
	from .coulomb import get_kernel
	kernel = get_kernel(getattr(meta, 'sys_dim', None))
	kwargs = {}
	if int(getattr(meta, 'sys_dim', 3)) in (2, 3):
		kwargs["certificate_fn"] = certificate_fn
	return kernel.q0_average(
		wfn, meta,
		S_cart=S_cart, epshead=epshead, static_kappa2=static_kappa2,
		nsamples=nsamples, method=method, qmc_reps=qmc_reps,
		analytic_sphere=analytic_sphere,
		**kwargs,
	)

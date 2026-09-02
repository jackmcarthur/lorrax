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
	(``head_minibz_average``) adds the Baldereschi-Tosatti analytic sphere
	term (3D) / widens the Voronoi fold (both dims) to the q→0 head; default
	False keeps the historical pure-Sobol average bit-identical.

	THE SLAB IGNORES THE SAMPLER DIALS BY DESIGN (2026-09-01).  On a
	``sys_dim = 2`` deck ``q0_average`` runs the exact Wigner--Seitz
	polygon cubature — the single q→0 cell-average owner, shared with the
	packed bispinor Γ completion — which takes no sample count, no
	sequence and no replicate count; ``nsamples``/``method``/``qmc_reps``
	below configure that kernel's named ``sobol_debug`` rule, which this
	wrapper never selects.  ``analytic_sphere`` is REFUSED there rather
	than ignored, because it is a deck key (``head_minibz_average``).
	Everything in the paragraphs below is therefore about the 3D bulk and
	its Baldereschi--Tosatti sphere, whose rule is unchanged.

	``method`` DEFAULTS TO ``"auto"`` since the extraction (it was
	``"sobol"``).  With scipy present — which is every production machine,
	and the ``sobol`` extra declares it — ``auto`` resolves to exactly the
	same scrambled-Sobol draw, so every number here is unchanged.  What
	changes is the machine WITHOUT scipy: it used to fall back to a uniform
	draw with ``nsamples`` silently raised to 2.5e6, and now it does the
	same thing while SAYING so once (``vcoul.minibz``'s announce-or-refuse
	gate).  A caller that wants the old silence back should not have it; a
	caller that wants a refusal instead passes ``method="sobol"``.

	For a 3D metal at exactly zero frequency, ``static_kappa2`` selects the
	Thomas-Fermi order of limits, ``<8*pi/(q^2+kappa_TF^2)>``.  It is mutually
	exclusive with the finite-frequency ``S_cart`` representation.
	"""
	from .coulomb import get_kernel
	kernel = get_kernel(getattr(meta, 'sys_dim', None))
	kwargs = {}
	if int(getattr(meta, 'sys_dim', 3)) == 2:
		kwargs["certificate_fn"] = certificate_fn
	return kernel.q0_average(
		wfn, meta,
		S_cart=S_cart, epshead=epshead, static_kappa2=static_kappa2,
		nsamples=nsamples, method=method, qmc_reps=qmc_reps,
		analytic_sphere=analytic_sphere,
		**kwargs,
	)

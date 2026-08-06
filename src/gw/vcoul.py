"""Coulomb q=0 mini-BZ averaging + Voronoi-cell point wrapping.

Two small helpers kept after the legacy per-q ``V(q,G)`` builder and the unused
``compute_vcoul_comps_for_q`` / ``compute_wcoul0_with_S`` routines were removed:

  * :func:`wrap_points_to_voronoi` — mini-BZ QMC sample wrapping (used by
    :mod:`gw.coulomb.sampler` and :mod:`gw.coulomb.base`).
  * :func:`compute_q0_averages`    — q=0 ``(vc0, wcoul0)`` average, a thin wrapper
    over ``gw.coulomb.get_kernel(sys_dim).q0_average`` (used by
    :mod:`gw.head_correction`).

WHAT ``gw.coulomb`` ACTUALLY OWNS (this file used to say "the
dimension-aware CoulombKernel in gw.coulomb now owns that logic", which
was never true of ``v(q+G)``):

  * :mod:`gw.coulomb.kernel` — the ONE ``v(q+G)`` formula, evaluated by
    every builder in the tree.
  * :mod:`gw.coulomb.sampler` — the ONE mini-BZ Monte-Carlo cell average.
  * ``gw.coulomb.{bulk_3d,slab_2d,box_0d}`` — the q->0 ``(vc0, wcoul0)``
    head per dimensionality, and NOTHING else.

The per-q-sphere ``v(q+G)`` BUILDERS live outside this package:
:func:`gw.compute_vcoul.compute_v_q_per_G` (GW) and
:func:`bse.vq_interp.v_slab_on_set` (BSE).  They share the formula, not
the plumbing.
"""

import functools

import jax
import jax.numpy as jnp

from common import Meta


@functools.partial(jax.jit, static_argnames=('nmax',))
def wrap_points_to_voronoi(randcart, bvec, nmax: int = 1):
	"""
	Helper function to get test q-points for mini-BZ average with correct Voronoi cell.
	Rewritten to use JAX arrays.

	Wrapped in ``@jax.jit`` (with ``nmax`` static) so all the per-line
	primitives (meshgrid, stack, reshape, matmul, broadcast subtract,
	norm, argmin, gather, subtract) collapse into a single XLA module
	cached on (input shape × nmax).  Without the jit each call site
	emitted ~10 eager-pjit cache misses.
	"""
	randcart_j = jnp.asarray(randcart, dtype=jnp.float64)
	bvec_j = jnp.asarray(bvec, dtype=jnp.float64)

	grid = jnp.arange(-nmax, nmax + 1)
	shifts = jnp.stack(jnp.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
	candidate_shifts = shifts @ bvec_j  # (M, 3)

	diff = randcart_j[:, None, :] - candidate_shifts[None, :, :]  # (N, M, 3)
	dists = jnp.linalg.norm(diff, axis=2)  # (N, M)
	best_idx = jnp.argmin(dists, axis=1)  # (N,)
	wrapped = randcart_j - candidate_shifts[best_idx]
	return wrapped


def compute_q0_averages(
	wfn,
	epshead,
	meta: Meta,
	S_cart: jnp.ndarray | None = None,
	nsamples: int = 2**18,
	method: str = "sobol",
	qmc_reps: int = 10,
	analytic_sphere: bool = False,
):
	"""Compute q=0 averages (vc0_mean, wcoul0) for the system's dimensionality.

	Thin compatibility wrapper around the dimension-aware ``CoulombKernel``
	in :mod:`gw.coulomb`.  The branching logic that used to live here is
	now distributed across the per-dim kernel modules.  ``analytic_sphere``
	(``head_minibz_average``) adds the Baldereschi-Tosatti analytic sphere
	term (3D) / widens the Voronoi fold (both dims) to the q→0 head; default
	False keeps the historical pure-Sobol average bit-identical.

	Returns ``vc0`` in **BARE** units (no ``1/Omega_cell``) — the caller
	applies the volume factor at injection.  ``get_kernel`` asserts this
	(``CoulombKernel.q0_units``); it is not a docstring promise.
	"""
	from .coulomb import get_kernel
	return get_kernel(getattr(meta, 'sys_dim', None)).q0_average(
		wfn, meta,
		S_cart=S_cart, epshead=epshead,
		nsamples=nsamples, method=method, qmc_reps=qmc_reps,
		analytic_sphere=analytic_sphere,
	)

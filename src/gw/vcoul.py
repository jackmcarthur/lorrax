"""Coulomb q=0 mini-BZ averaging + Voronoi-cell point wrapping.

Two small helpers kept after the legacy per-q ``V(q,G)`` builder and the unused
``compute_vcoul_comps_for_q`` / ``compute_wcoul0_with_S`` routines were removed
(the dimension-aware ``CoulombKernel`` in :mod:`gw.coulomb` now owns that logic):

  * :func:`wrap_points_to_voronoi` — mini-BZ QMC sample wrapping (used by
    :mod:`gw.compute_vcoul`).
  * :func:`compute_q0_averages`    — q=0 ``(vc0, wcoul0)`` average, a thin wrapper
    over ``gw.coulomb.get_kernel(sys_dim).q0_average`` (used by
    :mod:`gw.head_correction`).
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
):
	"""Compute q=0 averages (vc0_mean, wcoul0) for the system's dimensionality.

	Thin compatibility wrapper around the dimension-aware ``CoulombKernel``
	in :mod:`gw.coulomb`.  The branching logic that used to live here is
	now distributed across the per-dim kernel modules.
	"""
	from .coulomb import get_kernel
	return get_kernel(getattr(meta, 'sys_dim', None)).q0_average(
		wfn, meta,
		S_cart=S_cart, epshead=epshead,
		nsamples=nsamples, method=method, qmc_reps=qmc_reps,
	)

"""Coulomb utilities: Voronoi-cell sampling and per-q V(q,G).

This module now supports Sobol QMC sampling for the q=0 averages by default
and keeps the V(q,G) head zero so head averages are injected explicitly later.
"""

import functools

import numpy as np
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


# def get_V_qG(wfn, sym, q0, epshead, sys_dim, meta: Meta, do_Dmunu=False, mesh_xy=None):
# 	# V(q,G,G') = 4\pi/|q+G|^2 \delta_{G,G'} with 2D truncation factor in 2D systems.
# 	# Refactored: host prep (I/O) + jitted kernel (pure JAX math), single output sharding.
# 	print(q0)

# 	# Polarizations: 1 (Coulomb) + optional 3 (Breit)
# 	npol = 4 if do_Dmunu else 1

# 	# Host-side prep of static data
# 	bvec = jnp.asarray(wfn.blat * wfn.bvec, dtype=jnp.float64)
# 	q0j = jnp.asarray(q0, dtype=jnp.float64)
# 	zc = jnp.pi / bvec[2, 2]
# 	if sys_dim == 3:
# 		raise NotImplementedError("3D system calculation not yet implemented")

# 	# Decide padded G dimension for Y-sharding
# 	G_dim = int(wfn.ngkmax)
# 	if mesh_xy is not None:
# 		try:
# 			grid_y = int(mesh_xy.devices.shape[1])
# 		except Exception:
# 			grid_y = int(np.array(mesh_xy.devices).shape[1])
# 		G_dim_padded = ((G_dim + grid_y - 1) // grid_y) * grid_y
# 	else:
# 		G_dim_padded = G_dim

# 	# Materialize per-q inputs on host
# 	nkpts = int(wfn.nkpts)
# 	gvecs_padded_np = np.zeros((nkpts, G_dim_padded, 3), dtype=np.float64)
# 	nG_per_q_np = np.zeros((nkpts,), dtype=np.int32)
# 	qvecs_all_np = np.zeros((nkpts, 3), dtype=np.float64)
# 	for iq in range(nkpts):
# 		gq = wfn.get_gvec_nk(iq).astype(np.float64)
# 		gcount = gq.shape[0]
# 		nG_per_q_np[iq] = gcount
# 		gvecs_padded_np[iq, :gcount, :] = gq
# 		qvec = np.asarray(wfn.kpoints[iq], dtype=np.float64)
# 		if iq == 0:
# 			qvec = np.asarray(q0, dtype=np.float64)
# 		qvecs_all_np[iq] = qvec
# 	gvecs_padded = jnp.asarray(gvecs_padded_np)
# 	nG_per_q = jnp.asarray(nG_per_q_np)
# 	qvecs_all = jnp.asarray(qvecs_all_np)

# 	@partial(jax.jit, static_argnames=("do_Dmunu", "npol"))
# 	def vq_kernel(gvecs_all, nG_all, qvecs_all, bvec, zc, do_Dmunu: bool, npol: int):
# 		# gvecs_all: (nk, G, 3), nG_all: (nk,), qvecs_all: (nk,3)
# 		def compute_for_q(g_G3, nG, qvec):
# 			# Broadcast qvec over G, compute G_cart
# 			G_cart = (g_G3 + qvec) @ bvec  # (G,3)
# 			mask = jnp.arange(g_G3.shape[0], dtype=jnp.int32) < nG
# 			mask_f = mask.astype(jnp.float64)
# 			denom = jnp.sum(G_cart * G_cart, axis=1)
# 			denom_safe = jnp.where(mask, denom, 1.0)
# 			v = 8.0 * jnp.pi / denom_safe
# 			kxy = jnp.linalg.norm(G_cart[:, :2], axis=1)
# 			kz = G_cart[:, 2]
# 			v = v * ((1.0 - jnp.exp(-zc * kxy) * jnp.cos(kz * zc)))
# 			v = v * mask_f
# 			# Assemble (npol,npol,G)
# 			Vq = jnp.zeros((npol, npol, g_G3.shape[0]), dtype=jnp.float64)
# 			Vq = Vq.at[0, 0, :].set(v)
# 			def add_breit(Vq):
# 				# Avoid divide-by-zero in padded tail
# 				norms = jnp.linalg.norm(G_cart, axis=1, keepdims=True)
# 				norms = jnp.where(norms == 0.0, 1.0, norms)
# 				unitG = G_cart / norms
# 				proj = jnp.eye(3)[None, :, :] - unitG[:, :, None] * unitG[:, None, :]  # (G,3,3)
# 				proj = proj.transpose(1, 2, 0) * mask_f  # (3,3,G)
# 				return Vq.at[1:4, 1:4, :].set(proj)
# 			Vq = jax.lax.cond(do_Dmunu, add_breit, lambda Vq: Vq, Vq)
# 			return Vq
# 		V_q = jax.vmap(compute_for_q, in_axes=(0, 0, 0))(gvecs_all, nG_all, qvecs_all)  # (nk,npol,npol,G)
# 		# Move nk to Coulomb index order (npol,npol,nk,G)
# 		return jnp.swapaxes(V_q, 0, 2).swapaxes(0, 1)

# 	V_qG = vq_kernel(gvecs_padded, nG_per_q, qvecs_all, bvec, zc, bool(do_Dmunu), int(npol))

# 	# mini-BZ Monte Carlo for V_q=0,G=0 and W_q=0 in a single jitted kernel
# 	@jax.jit
# 	def minibz_mc(key, bvec, nkx, nky, nkz, zc, q0len, epshead_real):
# 		randlims = bvec.T @ (jnp.diag(1.0 / jnp.asarray((nkx, nky, nkz))) @ jnp.linalg.inv(bvec.T))
# 		randvals = jax.random.uniform(key, (2_500_000, 3), dtype=jnp.float64)
# 		randcart = (bvec.T @ randvals.T).T
# 		wrapped_cart = wrap_points_to_voronoi(randcart, bvec, nmax=1)
# 		randqcart = (randlims @ wrapped_cart.T).T
# 		randqcart = randqcart.at[:, 2].set(0.0)
# 		# vc(0) truncated
# 		rand_vq = 4.0 * jnp.pi / jnp.einsum("ij,ij->i", randqcart, randqcart)
# 		kxy = jnp.linalg.norm(randqcart[:, :2], axis=1)
# 		rand_vq = rand_vq * 2.0 * (1.0 - jnp.exp(-jnp.pi / bvec[2, 2] * kxy) * jnp.cos(randqcart[:, 2] * jnp.pi / bvec[2, 2]))
# 		vc0_mean = jnp.mean(rand_vq)
# 		# W(0) via Ismail-Beigi gamma
# 		vc_qtozero = (1.0 - jnp.exp(-q0len * zc)) / (q0len * q0len)
# 		gamma = (1.0 / epshead_real - 1.0) / (q0len * q0len * vc_qtozero)
# 		vc_q = (1.0 - jnp.exp(-kxy * zc)) / (kxy * kxy)
# 		wq = vc_q / (1.0 + vc_q * (kxy * kxy) * gamma)
# 		wcoul0 = 8.0 * jnp.pi * jnp.mean(wq)
# 		return vc0_mean, wcoul0
# 	vc0_mean, wcoul0 = minibz_mc(jax.random.PRNGKey(0), bvec, meta.nkx, meta.nky, meta.nkz, zc, jnp.linalg.norm(q0j @ bvec), jnp.asarray(epshead.real, dtype=jnp.float64))
# 	print(f"V_q=0,G=G'=0 from miniBZ monte carlo: {float(vc0_mean):.4f}")
# 	V_qG = V_qG.at[0, 0, 0, 0].set(vc0_mean)

# 	# Overall physical scaling
# 	fact = jnp.float64(1.0 / wfn.cell_volume)
# 	V_qG = V_qG * fact
# 	wcoul0 = wcoul0 * fact

# 	# Apply a single Y sharding over the G axis, leveraging padding
# 	if mesh_xy is not None:
# 		V_qG = jax.lax.with_sharding_constraint(V_qG, NamedSharding(mesh_xy, P(None, None, None, 'y')))

# 	return V_qG.astype(jnp.complex128), wcoul0.astype(jnp.complex128)


# New: per-q helpers to avoid materializing all q at once

def compute_vcoul_comps_for_q(wfn, sym, meta: Meta, qvec_nonneg):
	"""Return vcoul_psiG_comps for a single q (array of shape (nG,3), int32)."""
	qvec = jnp.asarray(qvec_nonneg, dtype=jnp.float64)
	kgrid = jnp.asarray((meta.nkx, meta.nky, meta.nkz), dtype=jnp.float64)
	qvec_wrapped = jnp.divide(jnp.where(qvec > kgrid // 2, qvec - kgrid, qvec), kgrid)
	# Map to iq index using existing SymMaps API
	iq = sym.find_qpoint_index(qvec_wrapped, tol=1e-6)
	iq_cpu = int(iq) if not hasattr(iq, "get") else int(iq.get())
	iqbar = sym.irr_idx_k[iq_cpu]
	Sq = sym.sym_mats_k[sym.sym_idx_k[iq_cpu]]
	G_Sq = np.round(qvec_wrapped - Sq @ wfn.kpoints[iqbar]).astype(jnp.float64)
	vcoul_comps = np.einsum("ij,kj->ki", Sq.astype(jnp.int32), wfn.get_gvec_nk(iqbar)) - G_Sq[np.newaxis, :]
	return iq_cpu, iqbar, qvec_wrapped, vcoul_comps.astype(jnp.int32)


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


def compute_wcoul0_with_S(
	bvec: jnp.ndarray,
	nkx: int,
	nky: int,
	nkz: int,
	S_cart: jnp.ndarray,
	nsamples: int = 2_500_000,
) -> jnp.complex128:
	"""Average wcoul0 over Voronoi-cell q using small-q tensor S(ω).

	Computes wcoul0 = ⟨ v(q) · (1 - v(q) · q^T S q)^{-1} ⟩ over the Voronoi cell,
	with 2D truncation consistent with vcoul setup.
	"""
	bvec = jnp.asarray(bvec, dtype=jnp.float64)
	S = jnp.asarray(S_cart, dtype=jnp.complex128)
	zc = jnp.pi / bvec[2, 2]

	@jax.jit
	def _average(key):
		# Sample random q in Cartesian space covering Voronoi region of the mini-BZ cell
		randvals = jax.random.uniform(key, (nsamples, 3), dtype=jnp.float64)
		randcart = (bvec.T @ randvals.T).T
		wrapped_cart = wrap_points_to_voronoi(randcart, bvec, nmax=1)
		# Scale by grid to Voronoi cell of the Gamma-centered tile
		kgrid = jnp.asarray((nkx, nky, nkz), dtype=jnp.float64)
		randqcart = (bvec.T @ (jnp.diag(1.0 / kgrid) @ jnp.linalg.inv(bvec.T)) @ wrapped_cart.T).T
		# 2D truncation
		kxy = jnp.linalg.norm(randqcart[:, :2], axis=1)
		kz = randqcart[:, 2]
		vq = 4.0 * jnp.pi / jnp.einsum('ij,ij->i', randqcart, randqcart)
		vq = vq * 2.0 * (1.0 - jnp.exp(-zc * kxy) * jnp.cos(kz * zc))
		# q^T S q (complex)
		Sq = jnp.einsum('qi,ij,qj->q', randqcart, S, randqcart)
		w = vq.astype(jnp.complex128) / (1.0 - vq.astype(jnp.complex128) * Sq)
		return jnp.mean(w)

	return _average(jax.random.PRNGKey(0)).astype(jnp.complex128)

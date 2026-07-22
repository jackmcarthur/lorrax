"""ISDF core primitives: ψ + centroids -> ζ interpolation vectors.

Neutral array-in / array-out core of the ISDF fit — the composable phases
``c_q_from_psi_sm`` -> ``factor_c_q`` -> ``fit_one_rchunk`` (which fuses
``z_q_from_psi_sm`` + ``solve_zeta``) plus the q=0 Gram building blocks used
by centroid selection.  Depends only on ``common/`` (Meta, timing,
gamma_matrices, cholesky_2d, fft_helpers, wfn_transforms, psi_G_store) and
(func-local) ``ffi/`` (cusolvermp).  NO ``gw`` / LorraxConfig / h5 / V_q
packaging lives here — GW and BSE are consumers.
"""
import os
import time
from types import SimpleNamespace
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import io_callback as _io_callback

from common import Meta
from common import timing
from runtime.padding import pad_last_axis_to, round_up, solve_at_logical
from common.gamma_matrices import (
    gamma_perm_phase as _gamma_perm_phase_mu,
    gamma_double_contract,
)
from common.cholesky_2d import (
    cholesky_2d_batched,
    dense_to_tiles,
    tiles_to_dense,
)
from common.fft_helpers import (
    make_flat_k_ifftn,
    make_flat_k_fftn,
    compute_block_size_for_2d_cholesky,
)
from common.wfn_transforms import to_rchunk_inner


# ============================================================================
# Open-spin pair density: P_k,ab(μ, ν) = Σ_n ψ*_{n,k,a}(μ) ψ_{n,k,b}(ν)
# ============================================================================
#
# Single rank-5 pair-density path used by every channel (charge γ̃^0 = I_4
# AND transverse γ̃^i = α^i).  The (αβ) spin axes stay OPEN through the
# pair-density and IFFT steps; γ̃·γ̃ contraction happens at the post-IFFT
# reduction inside :func:`c_q_from_psi_sm` / :func:`z_q_from_psi_sm`
# (and at q=0 inside :func:`gram_q0_from_pair`).
#
# For γ̃^0 = γ̃^0 = I_4, ``gamma_double_contract`` short-circuits to the
# Σ_{αβ} P_l_conj·P_r Frobenius reduction (no gather, no phase mul) — see
# :func:`common.gamma_matrices.gamma_double_contract`.
#
# For γ̃^i, γ̃^j = α^l, α^l' it produces
#
#     C_q^{μ_L, ν_L}(μ,ν) = Σ_{k, αβα'β'} P*_{αβ}(μ,ν;k-q)
#                              · γ̃^{μ_L}_{αα'} γ̃^{ν_L}_{ββ'}
#                              · P_{α'β'}(μ,ν;k)
#
# The :func:`pair_density` standalone (rank-5 P_k_ab carrier) is kept
# for ``centroid.pivoted_cholesky``'s q=0 valence-conduction Gram —
# its caller can hold the full P_k_ab in HBM at the centroid-selection
# stage.  The hot CCT/ZCT path in :func:`fit_zeta_to_h5` never
# materialises P_k_ab globally; everything happens inside the
# monolithic shard_map kernels below.

# Compiled-kernel caches (keyed by shape/dtype/sharding). Module-level so the
# per-r-chunk loop reuses one traced kernel; do NOT rename `_fit_one_rchunk_cache`
# (gw_init clears it by name between runs). See ISDF_MOVE_PLAN.md STEP 2.
_pair_density_cache = {}  # pair-density kernel
_pair_pipeline_sm_cache = {}  # pair-density shard_map pipeline kernel


def pair_density(
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""Open-spin pair density P_k,ab(μ, col) = Σ_n ψ*_{n,k,a}(μ) ψ_{n,k,b}(col).

	Spin axes (a,b) are kept open; γ̃ is applied downstream at the C_q
	or Z_q post-IFFT reduction step.

	Inputs:
	    psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
	    psi_rcol_Y: (nk, nb, ns, n_col) with P(None, None, None, 'y')

	Output:
	    P_k_ab: (nk, ns, ns, n_rmu, n_col) with P(None, None, None, 'x', 'y')

	einsum: ``'kmna,knbr->kabmr'``.
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	cache_key = ('pair_density', id(mesh_xy), nk, n_rmu, nb, ns, n_col)

	if cache_key not in _pair_density_cache:
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		xy_out = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))

		@partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
		def _pair_density(psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			return jnp.einsum('kmna,knbr->kabmr', psi_L, psi_R, optimize=True)

		_pair_density_cache[cache_key] = _pair_density

	return _pair_density_cache[cache_key](psi_rmuT_X, psi_rcol_Y)


# Cache for ISDF pipeline jitted functions
_isdf_pipeline_cache = {}  # ISDF pipeline (z_q/c_q from psi_sm) kernel



# ============================================================================
# q=0 Gram matrix for pivoted-Cholesky centroid selection
# ============================================================================
# Drops the k→q FFT pair from the CCT formula: at q=0 the k-sum IS
# the answer.  Used by :mod:`centroid.pivoted_cholesky` to score
# candidate centroids on valence × conduction pair products.

def gram_q0_from_pair(
	P_v_k: jax.Array,
	P_c_k: jax.Array,
	k_weights: jax.Array,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	*,
	mesh_xy: Mesh,
) -> jax.Array:
	"""q=0 valence-conduction pair-product Gram from open-spin pair densities.

	Mathematically (q=0 special case of the CCT-over-k structure):

	    G(μ,ν) = Σ_k w_k · [Σ_{αβα'β'} γ̃^{μ_L}_{αα'} γ̃^{ν_L}_{ββ'}
	                          · P_v_{αβ}(μ,ν;k)*  · P_c_{α'β'}(μ,ν;k)]

	γ̃ identity short-circuit: pass ``gamma_L=None`` (and/or
	``gamma_R=None``) for charge / left-only / right-only sides.
	Both None → Σ_{αβ} P_v* · P_c, the historical pivoted-Cholesky
	candidate Gram in open-spin form.  γ̃^μ is monomial — each non-
	identity contraction is one ``jnp.take`` + element-wise phase
	multiply, not a 4×4 matmul.

	Used by :mod:`centroid.pivoted_cholesky`.

	Args:
		P_v_k: (nk, ns, ns, n_rmu, n_rmu) complex, valence open-spin pair
			density (output of :func:`pair_density` on the valence band
			window), sharded ``P(None, None, None, 'x', 'y')``.
		P_c_k: (nk, ns, ns, n_rmu, n_rmu) complex, conduction window,
			same layout.
		k_weights: (nk,) real, k-point weights (IBZ weights summing to 1,
			or 1/nk_tot for each full-BZ k-point).
		gamma_L, gamma_R: ``(perm, phase)`` tuples or ``None`` (=identity).
		mesh_xy: ('x','y') device mesh, same as the pair densities.

	Returns:
		G: (n_rmu, n_rmu) complex Hermitian PSD, sharded ``P('x','y')``.
	"""
	nk, ns1, ns2, n_rmu, _ = P_v_k.shape
	lhs_id = gamma_L is None
	rhs_id = gamma_R is None
	cache_key = ('gram_q0_from_pair', id(mesh_xy), nk, ns1, ns2, n_rmu,
	             lhs_id, rhs_id)

	if cache_key not in _isdf_pipeline_cache:
		in_spin = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P('x', 'y'))
		kw_rep = NamedSharding(mesh_xy, P())
		rep = NamedSharding(mesh_xy, P())

		_lhs_id = lhs_id
		_rhs_id = rhs_id

		@partial(jax.jit,
		         in_shardings=(in_spin, in_spin, kw_rep, rep, rep, rep, rep),
		         out_shardings=out_xy)
		def _gram_q0(P_v, P_c, kw, perm_L_, phase_L_, perm_R_, phase_R_):
			# γ̃-contracted spin reduction (5 → 3 rank, dropping (a,b)).
			# spin axes are (1, 2) on the (k, a, b, μ, ν) layout.
			prod = gamma_double_contract(
				jnp.conj(P_v), P_c,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(1, 2),
			)
			G = jnp.sum(kw[:, None, None] * prod, axis=0)
			# Symmetrize: q=0 Gram is Hermitian by construction; fp roundoff
			# can break it.  Cheap fix.
			G = 0.5 * (G + jnp.conj(G.T))
			return G

		_isdf_pipeline_cache[cache_key] = _gram_q0

	if lhs_id:
		perm_L = jnp.arange(ns1, dtype=jnp.int32)
		phase_L = jnp.ones(ns1, dtype=jnp.complex128)
	else:
		perm_L, phase_L = gamma_L
	if rhs_id:
		perm_R = jnp.arange(ns2, dtype=jnp.int32)
		phase_R = jnp.ones(ns2, dtype=jnp.complex128)
	else:
		perm_R, phase_R = gamma_R

	return _isdf_pipeline_cache[cache_key](
		P_v_k, P_c_k, k_weights, perm_L, phase_L, perm_R, phase_R)


# ============================================================================
# 2D Blocked Cholesky Solver - memory efficient for large n_rmu
# ============================================================================

# Cache for 2D Cholesky functions
_chol_2d_cache = {}


# ============================================================================
# Full zeta fitting pipeline with z-chunk loop and HDF5 output
# ============================================================================


# ============================================================================
# Monolithic shard_map pair pipelines — pair density + IFFT + γ̃·γ̃ + FFT
# fused into ONE shard_map.  Take wavefunction tensors directly
# (skipping the standalone rank-5 pair-density buffer); inside the
# shard_map region everything is local-per-rank and the FFTs run via
# direct ``jnp.fft.ifftn`` / ``jnp.fft.fftn`` calls (no nested
# ``make_flat_k_*`` helper — same approach as
# ``wfn_transforms.to_rchunk``).  No nested shard_maps, no helper
# boundary that could let XLA re-globalise the pair density.
#
# Why monolithic: a decomposed chain (pair density → IFFT → γ̃-contract
# → FFT each as its own jit) lands at 5 concurrent rank-5 pair-density
# slots in XLA's BufferAssignment (~4 GiB each at MoS2 3×3 bispinor:
# ``nk · ns² · n_rmu_local · col_local · 16``), pegging the kernel
# preallocated-temp peak at 21 GiB on a 28 GiB budget.  Folding into
# one shard_map drops the slot count to 3 (saving the standalone IFFT
# outputs + the gamma-contract intermediate), bringing the kernel peak
# to 13 GiB.
# ============================================================================


def c_q_from_psi_sm(
	psi_l_X: jax.Array,
	psi_l_Y: jax.Array,
	psi_r_X: jax.Array,
	psi_r_Y: jax.Array,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	*,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""C_q built from ψ directly inside one monolithic shard_map.

	Inputs:
	    psi_l_X, psi_r_X : (nk, n_rmu, nb, ns) sharded ``P(None, 'x', None, None)``
	    psi_l_Y, psi_r_Y : (nk, nb, ns, n_col) sharded ``P(None, None, None, 'y')``
	    gamma_L, gamma_R : ``(perm, phase)`` tuples or ``None`` (= γ̃^0 = I).
	    kgrid            : (nkx, nky, nkz).
	Output:
	    C_q              : (nq, n_rmu, n_col) sharded ``P(None, 'x', 'y')``.

	``n_col == n_rmu`` for CCT (square centroid).
	"""
	nkx, nky, nkz = kgrid
	nk = int(psi_l_X.shape[0])
	n_rmu = int(psi_l_X.shape[1])
	nb_l = int(psi_l_X.shape[2])
	nb_r = int(psi_r_X.shape[2])
	ns = int(psi_l_X.shape[3])
	n_col = int(psi_l_Y.shape[3])
	lhs_id = gamma_L is None
	rhs_id = gamma_R is None

	cache_key = ('c_q_from_psi_sm', id(mesh_xy), nk, n_rmu, n_col, ns,
	             nb_l, nb_r, nkx, nky, nkz, lhs_id, rhs_id)
	if cache_key not in _pair_pipeline_sm_cache:
		_lhs_id = lhs_id
		_rhs_id = rhs_id
		L_spec = P(None, 'x', None, None)
		R_spec = P(None, None, None, 'y')
		out_spec = P(None, 'x', 'y')

		@partial(shard_map, mesh=mesh_xy,
		         in_specs=(L_spec, R_spec, L_spec, R_spec,
		                   P(), P(), P(), P()),
		         out_specs=out_spec,
		         check_rep=False)
		def _local(psi_l_X_, psi_l_Y_, psi_r_X_, psi_r_Y_,
		           perm_L_, phase_L_, perm_R_, phase_R_):
			# Per-rank shapes (n_rmu_local on 'x', n_col_local on 'y',
			# bands replicated):
			#   psi_l_X_ : (nk, n_rmu_local, nb_l, ns)
			#   psi_l_Y_ : (nk, nb_l, ns, n_col_local)
			#
			# Einsum output order matters for memory: the gemm naturally
			# produces ``(k, ns_l · col_local, ns_r · μ_local)`` with the
			# bitcast factoring as ``(k, ns_l, col_local, μ_local, ns_r)``.
			# We pick output spec ``'karmb'`` (k, ns_l, col, μ, ns_r) so
			# the rank-7 reshape that feeds the IFFT is a pure bitcast —
			# no extra 4-GiB buffer for the rank-3 → rank-7 transpose
			# (which the natural ``'kabmr'`` order forced).  Verified
			# from HLO: dropped one of the three rank-5 lifetime slots.
			mu_loc = psi_l_X_.shape[1]
			col_loc = psi_l_Y_.shape[3]
			P_l = jnp.einsum(
				'kmna,knbr->karmb', psi_l_X_, psi_l_Y_, optimize=True)
			P_r = jnp.einsum(
				'kmna,knbr->karmb', psi_r_X_, psi_r_Y_, optimize=True)
			# Split k → (kx, ky, kz) — bitcast given the above layout.
			# Rank-7: (kx, ky, kz, ns_l, col, μ, ns_r).
			P_l_3d = P_l.reshape(nkx, nky, nkz, ns, col_loc, mu_loc, ns)
			del P_l
			P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0, 1, 2), norm='forward')
			P_l_R_conj = jnp.conj(P_l_R)
			del P_l_3d, P_l_R
			P_r_3d = P_r.reshape(nkx, nky, nkz, ns, col_loc, mu_loc, ns)
			del P_r
			P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0, 1, 2), norm='forward')
			del P_r_3d
			# Reduce over the spin axes (3=ns_l, 6=ns_r) of the rank-7
			# form.  Output rank-5: (kx, ky, kz, col, μ).
			C_R = gamma_double_contract(
				P_l_R_conj, P_r_R,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(3, 6),
			)
			del P_l_R_conj, P_r_R
			C_q_3d = jnp.fft.fftn(C_R, axes=(0, 1, 2), norm='forward')
			# Reshape back to (nk, col, μ); transpose final two axes
			# to satisfy out_spec ``P(None, 'x', 'y')`` for (nk, μ, col).
			# This transpose acts on the rank-3 reduced form (~16 MB),
			# not the rank-5 pair density.
			return jnp.transpose(
				C_q_3d.reshape(nkx * nky * nkz, col_loc, mu_loc),
				(0, 2, 1))

		@jax.jit
		def fn(psi_l_X_, psi_l_Y_, psi_r_X_, psi_r_Y_, pL, phL, pR, phR):
			return _local(psi_l_X_, psi_l_Y_, psi_r_X_, psi_r_Y_,
			              pL, phL, pR, phR)

		_pair_pipeline_sm_cache[cache_key] = fn

	if lhs_id:
		perm_L = jnp.arange(ns, dtype=jnp.int32)
		phase_L = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_L, phase_L = gamma_L
	if rhs_id:
		perm_R = jnp.arange(ns, dtype=jnp.int32)
		phase_R = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_R, phase_R = gamma_R

	return _pair_pipeline_sm_cache[cache_key](
		psi_l_X, psi_l_Y, psi_r_X, psi_r_Y,
		perm_L, phase_L, perm_R, phase_R)


def z_q_from_psi_sm(
	psi_l_X: jax.Array,
	psi_r_X: jax.Array,
	psi_G_store,
	*,
	band_chunk_ranges: tuple[tuple[int, int], ...],
	band_range_left: tuple[int, int],
	band_range_right: tuple[int, int],
	r_start_dyn,
	r_chunk_size: int,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""Z_q built from ψ via a streaming-scan pair density inside one shard_map.

	Round 6 redesign (`round5_unified_plan.md` §2.10 / §6.5).  Replaces
	the all-at-once einsum that consumed pre-computed ``psi_l_Y`` /
	``psi_r_Y`` with a ``lax.scan`` over band-chunks inside the
	``shard_map`` body.  Per iter:

	  1. ``io_callback`` pulls this rank's 1/P bands of bc ``i`` from
	     :class:`PsiGStore`'s host tile (band-flat-sharded over the
	     full ``('x','y')`` mesh).
	  2. :func:`common.wfn_transforms.to_rchunk_inner` does the local
	     IFFT + per-rank r-slab (``r0_local = r_start + axis_index('y')
	     * r_loc``).
	  3. ``lax.all_gather(axis_name=('x','y'), axis=1, tiled=True)``
	     aligns the band axis with ``psi_l_X`` / ``psi_r_X``'s
	     band-replicated layout.  IFFT-FIRST; gather-first would blow
	     the FFT box to ~80 GB / rank.
	  4. L/R per-bc band masks (mask approach — ``jnp.where`` on a
	     rank-local axis).
	  5. Two einsums into rank-5 carries ``(P_l_acc, P_r_acc)``.

	Post-scan: existing IFFT(k) → γ̃·γ̃ → FFT(k) tail (byte-identical
	to the pre-rewrite body — only the front of the body changes per
	plan §2.7).

	Inputs:
	    psi_l_X, psi_r_X    : ``(nk, n_rmu, nb_l, ns)`` /
	                          ``(nk, n_rmu, nb_r, ns)`` sharded
	                          ``P(None, 'x', None, None)`` (μ on 'x',
	                          bands replicated).
	    psi_G_store         : :class:`PsiGStore` — closure-captured;
	                          provides ``_slice_local_tile_bc`` /
	                          ``g_index`` / ``kvecs_frac``.  NOT a jit
	                          argument.
	    band_chunk_ranges   : tuple of (b_lo, b_hi) global band indices.
	    band_range_left     : (L_lo, L_hi) global L-window.  Must
	                          satisfy nb_l == L_hi - L_lo.
	    band_range_right    : (R_lo, R_hi) global R-window.
	    fft_grid            : (nx, ny, nz).
	    r_start_dyn         : int32 scalar — flat-r start of the chunk.
	    r_chunk_size        : static int — full r-chunk extent.  Per
	                          rank slab is ``r_chunk_size // p_y``.
	    gamma_L, gamma_R    : ``(perm, phase)`` tuples or ``None``
	                          (= γ̃^0 = I).
	    kgrid               : (nkx, nky, nkz).
	Output:
	    Z_q                 : (nq, n_rmu, n_zchunk) sharded
	                          ``P(None, 'x', 'y')``.
	"""
	from common.psi_G_store import _PSI_G_FLAT_SPEC  # noqa: F401  (sharding contract)

	fft_grid = tuple(int(s) for s in psi_G_store.meta.fft_grid)
	nkx, nky, nkz = kgrid
	nk = int(psi_l_X.shape[0])
	n_rmu = int(psi_l_X.shape[1])
	nb_l = int(psi_l_X.shape[2])
	nb_r = int(psi_r_X.shape[2])
	ns = int(psi_l_X.shape[3])
	n_zchunk = int(r_chunk_size)
	p_x = int(mesh_xy.shape['x'])
	p_y = int(mesh_xy.shape['y'])
	if n_zchunk % p_y != 0:
		raise ValueError(
			f"z_q_from_psi_sm: r_chunk_size={n_zchunk} not divisible by "
			f"p_y={p_y} (out_spec=P(None,'x','y') requires this).")
	r_loc = n_zchunk // p_y

	bcr = tuple((int(lo), int(hi)) for (lo, hi) in band_chunk_ranges)
	L_lo_g = int(band_range_left[0]); L_hi_g = int(band_range_left[1])
	R_lo_g = int(band_range_right[0]); R_hi_g = int(band_range_right[1])
	if L_hi_g - L_lo_g != nb_l:
		raise ValueError(
			f"z_q_from_psi_sm: psi_l_X.shape[2]={nb_l} != L_hi - L_lo "
			f"= {L_hi_g - L_lo_g}.")
	if R_hi_g - R_lo_g != nb_r:
		raise ValueError(
			f"z_q_from_psi_sm: psi_r_X.shape[2]={nb_r} != R_hi - R_lo "
			f"= {R_hi_g - R_lo_g}.")

	lhs_id = gamma_L is None
	rhs_id = gamma_R is None

	cache_key = (
		'z_q_from_psi_sm_streaming', id(mesh_xy), id(psi_G_store),
		nk, n_rmu, n_zchunk, ns, nb_l, nb_r, nkx, nky, nkz,
		lhs_id, rhs_id, bcr, (L_lo_g, L_hi_g), (R_lo_g, R_hi_g),
		tuple(int(s) for s in fft_grid),
	)
	if cache_key not in _pair_pipeline_sm_cache:
		_lhs_id = lhs_id
		_rhs_id = rhs_id
		_psi_G_store = psi_G_store
		fft_grid_t = tuple(int(s) for s in fft_grid)

		# Static per-bc tables: global band-axis offsets for the
		# bc-aligned scan body.  Each iter gathers ``bpd_max_global =
		# P · _bpd_max`` bands (the bc's full band width), so the L/R
		# masks index into a static-size axis.
		P_total = p_x * p_y
		bpd_max = int(_psi_G_store._bpd_max)
		bpd_max_global = bpd_max * P_total          # padded global bc band count
		n_bc = len(bcr)
		# Per-bc tables.  Built as np arrays here (NOT jnp.asarray) so
		# they enter the shard_map body via numpy → jnp lift inside the
		# Manual-mode body.  Closure-captured Auto-sharded jax.Arrays
		# inside a Manual-mode shard_map body trigger a mesh-context
		# mismatch (same issue as kvecs_frac in the earlier debug);
		# wrapping them as numpy constants and lifting inside the body
		# treats them as concrete constants from the body's perspective.
		_b_lo_global_np = np.asarray(
			[lo for (lo, _hi) in bcr], dtype=np.int32)
		_b_hi_global_np = np.asarray(
			[hi for (_lo, hi) in bcr], dtype=np.int32)
		# psi_l_X / psi_r_X are sized to their L/R window; per-bc slice
		# offset within them is `bc.lo - L_lo_g` / `bc.lo - R_lo_g`,
		# which can be NEGATIVE when bc starts below the window AND
		# the slice (offset, offset+bpd_max_global) can extend PAST
		# the window when bc spans the upper boundary or the final bc
		# is short.  We pre-pad psi_l_X / psi_r_X at BOTH ends with
		# zero rows so that the dynamic_slice never goes out of
		# bounds — which would otherwise trigger XLA's silent clamp
		# (start clamped to ``axis_size - slice_size``), producing
		# *physically wrong bands* that the L/R mask CANNOT recover
		# (the mask's index assumes the slice covers
		# ``[bc.lo, bc.lo+bpd_max_global)`` but the clamp returns
		# something else).  See round6_discussion.md:506 BLOCKER from
		# Agent 4.  Pad rows correspond to out-of-window global bands;
		# the L/R mask zeros their contribution to the einsum
		# (math-neutral) — same contract as the front-pad rows.
		front_pad_l = max(
			(max(0, L_lo_g - lo) for (lo, _hi) in bcr), default=0)
		front_pad_r = max(
			(max(0, R_lo_g - lo) for (lo, _hi) in bcr), default=0)
		# After front-pad, offset = bc.lo - L_lo_g + front_pad_l (always ≥ 0).
		_psi_l_X_bc_offset_np = np.asarray(
			[lo - L_lo_g + front_pad_l for (lo, _hi) in bcr],
			dtype=np.int32)
		_psi_r_X_bc_offset_np = np.asarray(
			[lo - R_lo_g + front_pad_r for (lo, _hi) in bcr],
			dtype=np.int32)
		# Back-pad: the slice (offset, offset+bpd_max_global) must fit
		# entirely within the padded array (front_pad + nb + back_pad).
		# The largest end-offset across all bcs determines the back-pad.
		_max_end_l = int(max(
			(off + bpd_max_global for off in _psi_l_X_bc_offset_np),
			default=0))
		_max_end_r = int(max(
			(off + bpd_max_global for off in _psi_r_X_bc_offset_np),
			default=0))
		back_pad_l = max(0, _max_end_l - (front_pad_l + nb_l))
		back_pad_r = max(0, _max_end_r - (front_pad_r + nb_r))
		ngkmax = int(_psi_G_store._per_rank_shape[3])

		# Slicer needs static `out_sds`; close over the padded shape.
		_per_rank_bc_shape = (nk, bpd_max, ns, ngkmax)
		_slicer_out_sds = jax.ShapeDtypeStruct(
			_per_rank_bc_shape, jnp.complex128)

		def _slicer_host(x_idx, y_idx, bc_idx):
			return _psi_G_store._slice_local_tile_bc(x_idx, y_idx, bc_idx)

		L_spec = P(None, 'x', None, None)
		out_spec = P(None, 'x', 'y')
		# g_index, kvecs_frac: replicated.  Pass through shard_map's
		# in_specs (NOT closure) so JAX sees Manual-mode-compatible
		# access inside the body (closure-captured Auto-sharded arrays
		# trip a mesh-context mismatch under multi-device shard_map).
		g_index_spec    = P(None, None, None, None)
		kvecs_frac_spec = P(None, None)

		@partial(shard_map, mesh=mesh_xy,
		         in_specs=(L_spec, L_spec, P(), P(), P(), P(), P(),
		                   g_index_spec, kvecs_frac_spec),
		         out_specs=out_spec,
		         check_rep=False)
		def _local(psi_l_X_, psi_r_X_, perm_L_, phase_L_,
		           perm_R_, phase_R_, r_start_,
		           g_index_dev, kvecs_frac_dev):
			# Per-rank shapes:
			#   psi_l_X_ : (nk, n_rmu_loc, nb_l, ns)   μ on 'x' (replicated bands)
			#   psi_r_X_ : (nk, n_rmu_loc, nb_r, ns)
			# Carry (rank-local, NO sharding annotation):
			#   P_l_acc  : (nk, ns, r_loc, mu_loc, ns)
			# r_loc = n_zchunk / p_y per §2.3 (out_spec='y' on n_zchunk).
			x_idx = jax.lax.axis_index('x')
			y_idx = jax.lax.axis_index('y')
			mu_loc = psi_l_X_.shape[1]

			# Lift per-bc static tables to jnp.array INSIDE the
			# Manual-mode body so they're treated as Manual-mode
			# concrete constants (vs Auto-sharded closure jax.Arrays
			# which trip a mesh-context mismatch).
			b_lo_global_arr = jnp.asarray(_b_lo_global_np)
			b_hi_global_arr = jnp.asarray(_b_hi_global_np)
			psi_l_X_bc_offset = jnp.asarray(_psi_l_X_bc_offset_np)
			psi_r_X_bc_offset = jnp.asarray(_psi_r_X_bc_offset_np)

			# Pre-pad psi_l_X_ / psi_r_X_ at the front with front_pad_*
			# zero bands so the per-bc offset is always non-negative
			# (handles the bc.lo < L_lo_g / R_lo_g case where the
			# X-side offset would otherwise be negative; bands at
			# position [0, front_pad) are zeros, masked-zero in the
			# einsum).  Static front_pad_* baked at trace.
			# Pad both ends.  Front-pad covers bc.lo < window_lo
			# (negative offset case); back-pad covers
			# bc.lo + bpd_max_global > window_lo + nb (out-of-bounds
			# end case — XLA's dynamic_slice would otherwise silently
			# clamp the start, producing wrong bands the L/R mask
			# can't recover; see round6_discussion.md:506 BLOCKER).
			if front_pad_l > 0 or back_pad_l > 0:
				psi_l_X_padded = jnp.pad(
					psi_l_X_,
					((0, 0), (0, 0), (front_pad_l, back_pad_l), (0, 0)))
			else:
				psi_l_X_padded = psi_l_X_
			if front_pad_r > 0 or back_pad_r > 0:
				psi_r_X_padded = jnp.pad(
					psi_r_X_,
					((0, 0), (0, 0), (front_pad_r, back_pad_r), (0, 0)))
			else:
				psi_r_X_padded = psi_r_X_

			# r-slab strategy: each y-rank ultimately owns ``r_loc``
			# positions of the r-chunk (out_spec=P(None, 'x', 'y') on
			# n_zchunk).  We CANNOT slice the r-axis per-rank BEFORE
			# the all_gather over bands — that would mix r-slabs from
			# different y-ranks at the same gathered band position
			# (r-incoherence at the einsum).  Instead: per rank
			# compute the FULL r-chunk in psi_Y_local, gather bands,
			# THEN slice the r-axis to this y-rank's per-rank slab.
			# Per-rank cost of full-r psi_Y_local is bigger by p_y vs
			# the r_loc version, but XLA's scan-internal allocator
			# aliases the per-iter slab across iters → single slot.
			r0_y_offset = y_idx * jnp.int32(r_loc)

			P_l_init = jnp.zeros(
				(nk, ns, r_loc, mu_loc, ns), dtype=jnp.complex128)
			P_r_init = jnp.zeros(
				(nk, ns, r_loc, mu_loc, ns), dtype=jnp.complex128)

			def body(carry, bc_idx):
				P_l_acc, P_r_acc = carry
				# (1) io_callback: this rank's 1/P bands of bc bc_idx.
				#     ordered=False per §3.4 (lax.scan(unroll=1) gives
				#     sequential per-rank execution at runtime; ordered
				#     is a perf knob, not correctness).
				psi_G_bc_local = _io_callback(
					_slicer_host, _slicer_out_sds,
					x_idx, y_idx, bc_idx,
					ordered=False)
				# (2) IFFT + FULL r-chunk slab (NOT per-rank r_loc — see
				#     "r-slab strategy" comment above).  Local FFT box
				#     per rank is c128[nk, bpd_max, ns, n_rtot] ·
				#     cuFFT_scratch; the full-r slab is c128[nk,
				#     bpd_max, ns, n_zchunk] per rank.  Both per-iter,
				#     aliased across iters by XLA's scan-internal
				#     allocator → single slot of each.
				psi_Y_bc_local_full_r = to_rchunk_inner(
					psi_G_bc_local, g_index_dev, fft_grid_t,
					r_start_, n_zchunk,
					kvecs_frac=kvecs_frac_dev, norm="ortho")
				# (3) all_gather across both mesh axes along the band
				#     axis.  IFFT-FIRST per §2.9 (gather-first → 80 GB
				#     FFT box, infeasible).  Output: (nk, P·bpd_max,
				#     ns, n_zchunk) on every rank.
				psi_Y_bc_full_r = jax.lax.all_gather(
					psi_Y_bc_local_full_r, axis_name=('x', 'y'),
					axis=1, tiled=True)
				# (3b) Slice the r-axis to THIS y-rank's r_loc slab.
				#      MUST happen AFTER gather so the band axis +
				#      r axis are coherent (each gathered band's r
				#      values come from the SAME source rank's full
				#      r-chunk computation).
				psi_Y_bc = jax.lax.dynamic_slice_in_dim(
					psi_Y_bc_full_r, r0_y_offset, r_loc, axis=3)
				# (4) L/R global-index masks (mask approach per §2.5).
				#     bc_valid handles short final bc (pad rows = zero).
				g_axis = (b_lo_global_arr[bc_idx]
				          + jnp.arange(bpd_max_global, dtype=jnp.int32))
				bc_valid = g_axis < b_hi_global_arr[bc_idx]
				l_mask = ((g_axis >= L_lo_g) & (g_axis < L_hi_g) & bc_valid)
				r_mask = ((g_axis >= R_lo_g) & (g_axis < R_hi_g) & bc_valid)
				psi_l_Y_bc = jnp.where(
					l_mask[None, :, None, None], psi_Y_bc, 0)
				psi_r_Y_bc = jnp.where(
					r_mask[None, :, None, None], psi_Y_bc, 0)
				# (5) psi_l_X / psi_r_X per-bc slice (band axis
				#     replicated → purely local).  Static slice length
				#     bpd_max_global so XLA can fold; offset traced.
				#     Slice from the FRONT-PADDED psi_*_X_padded so the
				#     offset is always non-negative even for bcs
				#     starting below the L/R window start.
				psi_l_X_bc = jax.lax.dynamic_slice_in_dim(
					psi_l_X_padded, psi_l_X_bc_offset[bc_idx],
					bpd_max_global, axis=2)
				psi_l_X_bc = jnp.where(
					l_mask[None, None, :, None], psi_l_X_bc, 0)
				psi_r_X_bc = jax.lax.dynamic_slice_in_dim(
					psi_r_X_padded, psi_r_X_bc_offset[bc_idx],
					bpd_max_global, axis=2)
				psi_r_X_bc = jnp.where(
					r_mask[None, None, :, None], psi_r_X_bc, 0)
				# (6) Two einsums into the carries (interleaved single
				#     scan per §2.10; with the small carry the γ̃
				#     contract dominates peak — serialization bought
				#     nothing).
				delta_P_l = jnp.einsum(
					'kmna,knbr->karmb',
					psi_l_X_bc, psi_l_Y_bc, optimize=True)
				delta_P_r = jnp.einsum(
					'kmna,knbr->karmb',
					psi_r_X_bc, psi_r_Y_bc, optimize=True)
				return (P_l_acc + delta_P_l, P_r_acc + delta_P_r), None

			# DO NOT unroll — the FFT-box and psi_G_bc aliasing depends
			# on per-iter sequential lifetime.  unroll=1 keeps the
			# WhileOp atomic and lets XLA's scan-internal allocator
			# reuse the slot (§3.5).
			(P_l, P_r), _ = jax.lax.scan(
				body, (P_l_init, P_r_init),
				jnp.arange(n_bc, dtype=jnp.int32))

			# Post-pair pipeline (byte-identical to today's tail per §2.7).
			P_l_3d = P_l.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
			del P_l
			P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0, 1, 2), norm='forward')
			P_l_R_conj = jnp.conj(P_l_R)
			del P_l_3d, P_l_R
			P_r_3d = P_r.reshape(nkx, nky, nkz, ns, r_loc, mu_loc, ns)
			del P_r
			P_r_R = jnp.fft.ifftn(P_r_3d, axes=(0, 1, 2), norm='forward')
			del P_r_3d
			Z_R = gamma_double_contract(
				P_l_R_conj, P_r_R,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(3, 6),
			)
			del P_l_R_conj, P_r_R
			Z_q_3d = jnp.fft.fftn(Z_R, axes=(0, 1, 2), norm='forward')
			return jnp.transpose(
				Z_q_3d.reshape(nkx * nky * nkz, r_loc, mu_loc),
				(0, 2, 1))

		@jax.jit
		def fn(psi_l_X_, psi_r_X_, pL, phL, pR, phR, r_start_,
		        g_index_, kvecs_frac_):
			return _local(psi_l_X_, psi_r_X_, pL, phL, pR, phR, r_start_,
			              g_index_, kvecs_frac_)

		_pair_pipeline_sm_cache[cache_key] = fn

	if lhs_id:
		perm_L = jnp.arange(ns, dtype=jnp.int32)
		phase_L = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_L, phase_L = gamma_L
	if rhs_id:
		perm_R = jnp.arange(ns, dtype=jnp.int32)
		phase_R = jnp.ones(ns, dtype=jnp.complex128)
	else:
		perm_R, phase_R = gamma_R

	r_start_arg = (jnp.int32(int(r_start_dyn))
	                if isinstance(r_start_dyn, (int, np.integer))
	                else r_start_dyn)
	return _pair_pipeline_sm_cache[cache_key](
		psi_l_X, psi_r_X, perm_L, phase_L, perm_R, phase_R, r_start_arg,
		psi_G_store.g_index, psi_G_store.kvecs_frac)


# Backward-compat shim removed — old z_q_from_psi_sm signature
# (consumed pre-computed psi_l_Y / psi_r_Y) is gone.  Call sites
# updated in `_make_fit_one_rchunk_kernel._kernel`.


def _identity_pad_block_diagonal(
    M: jax.Array,
    *,
    n_rmu_logical: int,
    mesh_xy: Mesh,
) -> jax.Array:
    """Add identity to the pad-block diagonal of a square N_μ² matrix.

    ``M`` has shape ``(nq, n_rmu, n_rmu)`` at PADDED μ extent with
    zero pad rows/cols (the Phase 3a contract: bilinear in zero-padded
    ψ ⇒ M's pad rows/cols are exact zeros).  This helper adds 1 to the
    diagonal entries in positions ``[n_rmu_logical, n_rmu)``, leaving
    the logical block exactly intact.  Result: ``M_id_pad =
    block_diag(M_log, I_pad)`` — block-diagonal with the input's
    logical block on top-left and identity on bottom-right.

    Why this matters — and the limits of the guarantee:  In EXACT
    arithmetic, Cholesky and LU on the identity-padded matrix produce
    factorisations whose logical block equals the factorisation of the
    un-padded logical-only matrix (the recursions never read across
    the zero off-diagonal pad blocks, and ``√1 = 1`` exactly), and the
    back-solve with zero-pad-row ``Z`` gives ``y_pad = 0`` with the
    logical solve unchanged.  In FLOATING POINT the guarantee is only
    approximate, because blocked/tiled implementations regroup partial
    sums when the matrix extent changes:

    * **Cholesky (charge channel): holds to ≤1e-7 rel** in practice
      (measured ζ_C 5.5e-8 under a pad-extent flip at fixed P; the
      well-conditioned PSD CCT does not amplify the regrouping noise).
    * **LU on the near-singular indefinite transverse CCT: does NOT
      hold.**  Shape-dependent LU roundoff is amplified O(1) in the
      near-null modes — each pad extent yields a different,
      per-extent-deterministic ζ_T, with catastrophic resonances at
      some extents (MoS2 668→672: Σ^B tile(2,2) −0.15 → −117.9 eV).
      See ``reports/device_invariance_2026-07-08/ROOT_CAUSE.md``.
      For this reason :func:`solve_zeta` slices the indefinite solve
      back to the LOGICAL extent — the identity pad added here is only
      a non-singularity safety net for the padded buffer, never the
      extent the transverse system is actually solved at.

    This is NOT ridge regularisation on C_q (which would corrupt the
    logical block).  The identity is added ONLY to the pad-block
    diagonal; the logical block is untouched.

    Output sharding is ``P(None, 'x', 'y')`` (n_rmu_padded is
    mesh-divisible by construction so single-axis sharding on each
    μ-dim works at any padded extent).  When ``n_rmu_logical ==
    n_rmu`` (no pad), the function is a no-op pass-through with the
    sharding constraint reapplied.
    """
    nq, n_rmu, n_rmu2 = M.shape
    if n_rmu != n_rmu2:
        raise ValueError(f"_identity_pad_block_diagonal expects square M; got {M.shape}")
    if n_rmu_logical > n_rmu:
        raise ValueError(
            f"n_rmu_logical={n_rmu_logical} > input extent {n_rmu}")
    sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    if n_rmu_logical == n_rmu:
        return jax.lax.with_sharding_constraint(M, sharding)
    # Build a (n_rmu, n_rmu) diagonal matrix whose only non-zeros sit
    # on positions ``[n_rmu_logical, n_rmu)`` of the diagonal.
    idx = jnp.arange(n_rmu)
    pad_diag_mask = (idx >= n_rmu_logical).astype(M.dtype)
    eye_pad = jnp.diag(pad_diag_mask)
    M_id_pad = M + eye_pad[None, :, :]
    return jax.lax.with_sharding_constraint(M_id_pad, sharding)


# Replication cap for the mesh-INVARIANT charge Cholesky.  When the whole
# CCT stack (nq, n_μ, n_μ) c128 fits under this many bytes on one device we
# factor it with a fully-replicated dense ``jnp.linalg.cholesky`` (exact,
# grid-agnostic); only genuinely large stacks fall back to the distributed
# cuSolverMp potrf.  4 GiB covers every current production fit (MoS2 6×6
# n_μ=1600 → 1.5 GiB, CrI3 6×6 80Ry n_μ≈1800 → ≤1.9 GiB, Si IBZ) and
# excludes only the full-BZ Si 4×4×4 60Ry stack (nq=64, n_μ=2400 → 24 GiB).
# See reports/gw_zeta_mesh_invariance_2026-07-20 for the drift this removes.
# Raise it with LORRAX_ZETA_REPLICATE_CAP_GIB when the stack is bigger but the
# device budget allows (full-BZ MoS2 12×12 n_μ=2412 is 13.4 GiB and NEEDS the
# rank truncation — see _resolve_solver_kind_charge).
_REPLICATED_CHOL_MAX_STACK_BYTES = int(
    float(os.environ.get("LORRAX_ZETA_REPLICATE_CAP_GIB", "4")) * 1024**3)


def _replicate_charge_ok(nq: int | None, n_rmu: int | None) -> bool:
    """True when the charge CCT stack ``(nq, n_μ, n_μ)`` c128 fits under the
    replication cap — the criterion for the mesh-invariant dense Cholesky
    over the grid-dependent distributed cuSolverMp potrf.

    Requires both ``nq`` and ``n_rmu`` (the ζ-fit caller passes them from
    ``C_q.shape[0]`` and ``meta.n_rmu``); ``None`` — direct callers that
    don't supply them — keeps the legacy distributed policy so nothing off
    the GW ζ-fit path changes behaviour.
    """
    if nq is None or n_rmu is None:
        return False
    return int(nq) * int(n_rmu) ** 2 * 16 <= _REPLICATED_CHOL_MAX_STACK_BYTES


def _resolve_solver_kind_charge(
    mesh_xy: Mesh, override: str = "auto",
    n_rmu: int | None = None, nq: int | None = None,
    charge_zeta_solve: str = "cholesky",
) -> str:
    """Pick the charge-channel ζ-fit solver: fully-replicated dense
    Cholesky (mesh-invariant, the default for fit-size tiles) vs the
    distributed cuSolverMp potrf+potrs vs the in-tree shard_map 2D-blocked
    Cholesky + per-q triangular solve.

    Default policy (2026-07-20): **replicated dense Cholesky** whenever the
    CCT stack fits on one device (:func:`_replicate_charge_ok`).  The
    distributed cuSolverMp potrf is block-cyclic — its partial-sum
    regrouping depends on the process grid ``(px, py)`` — so at large,
    mildly rank-deficient n_μ (MoS2 6×6, 1600 centroids) the factor drifts
    ~0.3% between a 2×2 and a 4×4 grid, and the GN-PPM pole construction
    amplifies that into tens-of-eV Σ_c garbage on non-16-GPU meshes.  The
    replicated ``jnp.linalg.cholesky`` runs on the whole matrix on every
    device (one dense potrf per q), so L_q is bit-identical across device
    counts and process grids.  This mirrors the eigh-backend policy in
    ``bse/vq_interp`` (native batched by default; FFI backends reserved for
    tiles too large to replicate).  See
    ``reports/gw_zeta_mesh_invariance_2026-07-20``.

    Above the replication cap the older policy applies: cuSolverMp on
    **true 2D meshes** (px≥2 AND py≥2) — it bundles the distributed
    Cholesky into one FFI call per q, vs the in-tree ``sharded_cholesky``'s
    many small NCCL all-reduces per panel — otherwise the in-tree sharded
    path.

    Override via cohsex.in ``distributed_cholesky``:
      ``off``        → force the in-tree sharded Cholesky.
      ``cusolvermp`` → force cuSolverMp (still falls back on 1D meshes;
                       legacy alias ``on``).
      ``slate``      → SLATE ``potrf`` — the portable (Frontier/Aurora)
                       backend.  EXPLICIT choice: fails loudly if the
                       FFI/library is absent or the mesh geometry is the
                       guarded 1×q case (SLATE stride assert; see
                       tests/test_ffi_linalg_contract.py) rather than
                       silently running a different backend.
      ``auto`` (default) → replicated dense for fit-size stacks, else
                       cuSolverMp on true 2D / sharded otherwise (neither
                       cuSolverMp nor slate is auto-picked below the cap).
    """
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    is_2d = (px >= 2 and py >= 2)

    if override == 'off':
        return 'sharded_cholesky'
    if override in ('on', 'cusolvermp'):
        return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'
    if override == 'slate':
        _require_slate_ffi()
        if px == 1 and py > 1:
            raise ValueError(
                f"distributed_cholesky=slate: 1×{py} meshes hit a SLATE "
                f"stride assert (guarded; see ffi/slate README).  Use a "
                f"{py}×1 or square mesh, or a different backend.")
        return 'slate_cholesky'

    # auto (or unrecognised) → default policy.  Fit-size stacks factor with
    # the mesh-invariant replicated dense factor; larger stacks keep the
    # distributed / sharded policy.  ``charge_zeta_solve == 'rank_truncate'``
    # (the production default) selects the rank-revealing eigh pseudo-inverse
    # on the replicated route — the only route it applies to (a full eigh
    # cannot be block-cyclic).  Above the cap we therefore CANNOT honour it,
    # and we refuse rather than downgrade: the 2026-07-21 full-BZ 12×12 fit
    # (13.4 GiB, just over the cap) silently fell back and returned ζ 4.5×
    # too large, rebuilding V_q to relF 16–32 instead of 1.8e-15.
    if _replicate_charge_ok(nq, n_rmu):
        return ('replicated_rank_truncate'
                if charge_zeta_solve == 'rank_truncate'
                else 'replicated_cholesky')
    if charge_zeta_solve == 'rank_truncate':
        need = int(nq) * int(n_rmu) ** 2 * 16 / 1024**3 if nq and n_rmu else 0.0
        raise ValueError(
            f"charge_zeta_solve='rank_truncate' needs the replicated route, "
            f"but the CCT stack (nq={nq}, n_mu={n_rmu}) is {need:.2f} GiB > "
            f"the {_REPLICATED_CHOL_MAX_STACK_BYTES / 1024**3:.2f} GiB cap.  "
            f"Set LORRAX_ZETA_REPLICATE_CAP_GIB={-(-need // 1) + 1:.0f} if the "
            f"device budget allows, or charge_zeta_solve='cholesky' to accept "
            f"the distributed factor (NOT rank-conditioned — verify V_q).")
    return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'


def _require_slate_ffi() -> None:
    """Raise with an actionable message unless the SLATE FFI is loadable.

    SLATE is an OPTIONAL dependency: nothing imports it unless the input
    file explicitly selects ``distributed_cholesky = slate``.
    """
    try:
        from ffi.common.ffi_loader import get_lib
        get_lib()
        from ffi.slate import distributed_cholesky  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "distributed_cholesky=slate requested but the SLATE FFI is "
            f"unavailable ({exc}).  Build it with "
            "src/ffi/slate/scripts/build_perlmutter.sh + "
            "src/ffi/common/cpp/build.sh (CUDA) or "
            "src/ffi/common/cpp/host/build_host.sh (CPU backend), or use "
            "distributed_cholesky = auto|off|cusolvermp."
        ) from exc


def _require_scalapack_ffi() -> None:
    """Raise with an actionable message unless the ScaLAPACK host FFI is
    loadable.  Host-only optional dependency (Cray LibSci via
    liblorrax_ffi_host.so): nothing imports it unless the input file
    explicitly selects ``distributed_lu = scalapack``.
    """
    try:
        from ffi.common.ffi_loader import get_lib
        get_lib("cpu")
        from ffi.scalapack import batched_distributed_solve_lu  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "distributed_lu=scalapack requested but the ScaLAPACK host FFI "
            f"is unavailable ({exc}).  Build it with "
            "src/ffi/common/cpp/host/build_host.sh (host-only backend — on "
            "GPU meshes use distributed_lu = auto|off|cusolvermp)."
        ) from exc


def _resolve_solver_kind_transverse(mesh_xy: Mesh, override: str = "auto") -> str:
    """Pick the transverse-channel ζ-fit solver: cuSolverMp distributed
    getrf+getrs vs the in-tree per-q ``jnp.linalg.solve`` + ridge.

    Default policy (2026-05-12): mirrors the charge-channel resolver —
    use cuSolverMp on **true 2D meshes** (px≥2 AND py≥2).  cuSolverMp
    0.7.2 fixes the earlier 2D-grid getrf/getrs correctness bug
    (validated end-to-end on MoS2 3×3 bispinor at 2×2 mesh; see
    ``src/ffi/cusolvermp/cpp/batched_solve_lu_ffi.cc`` for history).

    Tradeoff: small FFI setup overhead at MoS2 scale (n_rmu=656,
    2×2 mesh).  At CrI3 6×6 80 Ry (n_rmu≈1800, 4×4 mesh) the cuSolverMp
    path is the right tool.

    Override via cohsex.in ``distributed_lu``:
      ``off``        → force per-q ``jnp.linalg.solve``.
      ``cusolvermp`` → force cuSolverMp (still falls back on 1D meshes;
                       legacy alias ``on``).
      ``scalapack``  → ScaLAPACK ``pXgetrf``+``pXgetrs`` from Cray LibSci
                       — the host/CPU-backend backend (liblorrax_ffi_host).
                       EXPLICIT choice, never auto-picked; fails loudly if
                       the host FFI is absent, and requires a square or
                       1-D mesh (pXgetrf needs square blocks).
      ``auto`` (default) → cuSolverMp on true 2D, legacy otherwise.
      (No ``slate`` value: a SLATE getrf wrapper does not exist yet.)
    """
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    is_2d = (px >= 2 and py >= 2)

    if override == 'off':
        return 'lu'
    if override in ('on', 'cusolvermp'):
        return 'cusolvermp_lu' if is_2d else 'lu'
    if override == 'scalapack':
        plat = mesh_xy.devices.flat[0].platform
        if plat != 'cpu':
            # Defense-in-depth for direct callers — gw_config already
            # rejects scalapack on non-CPU backends at parse time.
            raise ValueError(
                f"distributed_lu=scalapack is host-only (Cray LibSci) but "
                f"the mesh devices are {plat!r}; use distributed_lu = "
                f"auto|off|cusolvermp on GPU meshes.")
        _require_scalapack_ffi()
        if px > 1 and py > 1 and px != py:
            raise ValueError(
                f"distributed_lu=scalapack: mesh {px}x{py} unsupported — "
                f"pXgetrf needs square descriptor blocks, which the "
                f"one-tile-per-rank layout only gives on square or 1-D "
                f"meshes.")
        return 'scalapack_lu'

    return 'cusolvermp_lu' if is_2d else 'lu'


def _resolve_solver_kind(
    mesh_xy: Mesh, vertex_mu_L: int, solver_kind: str,
    distributed_cholesky: str = "auto",
    distributed_lu: str = "auto",
    n_rmu: int | None = None,
    nq: int | None = None,
    charge_zeta_solve: str = "cholesky",
) -> str:
    """Single source of truth for the ``auto`` resolution.  Transverse
    channels (γ̃^i, μ_L≠0) take ``_resolve_solver_kind_transverse``;
    charge channel takes ``_resolve_solver_kind_charge``.

    ``n_rmu`` (logical centroid count) and ``nq`` (per-q factor batch =
    ``C_q.shape[0]``) let the charge resolver pick the mesh-invariant
    replicated dense factor for fit-size stacks; ``charge_zeta_solve``
    (``'rank_truncate'`` | ``'cholesky'``) then picks the rank-revealing
    eigh pseudo-inverse vs Cholesky on that route.  The ζ-fit caller passes
    all three (``isdf_fitting.fit_zeta_to_h5``).  A concrete ``solver_kind``
    is returned unchanged (so ``factor_c_q`` / ``solve_zeta`` re-resolving
    the already-resolved kind need not repeat them).
    """
    if solver_kind != 'auto':
        return solver_kind
    if int(vertex_mu_L) != 0:
        return _resolve_solver_kind_transverse(mesh_xy, distributed_lu)
    return _resolve_solver_kind_charge(
        mesh_xy, distributed_cholesky, n_rmu=n_rmu, nq=nq,
        charge_zeta_solve=charge_zeta_solve)


_replicated_chol_cache = {}  # replicated dense Cholesky kernel (keyed by shape)


def _factor_c_q_replicated(
    C_q: jax.Array, mesh_xy: Mesh, n_rmu_logical: int,
    zeta_ridge: float = 0.0,
    charge_zeta_solve: str = 'cholesky',
    zeta_rcond: float = 1e-8,
) -> jax.Array:
    """Dense, fully REPLICATED factor of the identity-padded charge CCT.

    Two selectable conditioners share this ONE replicated (mesh-invariant)
    seam — ``charge_zeta_solve`` picks which factor is returned:

    * ``'rank_truncate'`` (production default) — rank-revealing ``eigh``
      pseudo-inverse factor ``B`` with ``B Bᴴ = C⁺`` (see the WHY note
      inside).  The back-solve is a matmul ``ζ = B(BᴴZ)``.
    * ``'cholesky'`` — the historical lower-triangular Cholesky factor
      ``L`` with ``L Lᴴ = C+ridge``.  Back-solve is two triangular solves.
      Bit-identical to the pre-rank-truncation code (the frozen contract);
      it is the selectable ALTERNATIVE.

    Mesh-invariant by construction for BOTH: the factorisation runs on the
    fully-replicated LOGICAL block — one dense ``eigh`` / ``cholesky`` per q
    on whole tiles — so the factor is bit-identical across device counts and
    process grids, unlike the block-cyclic cuSolverMp potrf whose partial-sum
    regrouping depends on ``(px, py)``.  This is the single code path for the
    ``'replicated_cholesky'`` / ``'replicated_rank_truncate'`` auto picks
    (fit-size n_μ on any mesh) and every single-device / 1-D-degenerate mesh
    (where a dense factor is the only option).

    Cholesky ridge (two per-q scalar terms, so both mesh-invariant):

      ridge = [ 1e-14·|tr(C)|  +  zeta_ridge·|tr(C)|/n ] · I

    * The hard ``1e-14·|tr(C)|`` FLOOR is unchanged from the historical
      single-device path — it lifts the tiny negative eigenvalues that
      appear with more centroids than band pairs so ``potrf`` stays real.
      With ``zeta_ridge == 0`` (the default) the factor is bit-identical to
      that path (the frozen-golden contract).
    * ``zeta_ridge`` (a fraction of the mean diagonal tr(C)/n, default 0) is
      an OPT-IN Tikhonov term that CONDITIONS a near-singular CCT (n_μ
      over-complete for the pair-density rank).  ``rank_truncate`` (the
      default) is the PRINCIPLED cure that supersedes it — drop the near-null
      directions instead of shifting them — so the ridge stays 0 there.
      ``LORRAX_ZETA_RIDGE`` overrides ε for tuning.

    Factorise at the LOGICAL extent and re-embed identity in the pad block
    (√1 = 1 for L; B's pad block is likewise identity and is sliced away in
    the back-solve) — see :func:`_identity_pad_block_diagonal`.  The factor
    regroups partial sums when the matrix extent changes, so factorising at
    the logical (not padded) extent keeps the factor pad-extent-invariant
    (the fixed-P invariance gate).
    """
    import os as _os
    nq, n_rmu, _ = C_q.shape
    n_log = int(n_rmu_logical)
    ridge_extra = float(_os.environ.get("LORRAX_ZETA_RIDGE", str(zeta_ridge)))
    rcond = float(_os.environ.get("LORRAX_ZETA_RCOND", str(zeta_rcond)))
    mode = str(charge_zeta_solve)
    out_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    rep_sh = NamedSharding(mesh_xy, P())
    key = (id(mesh_xy), int(nq), int(n_rmu), n_log,
           float(ridge_extra), mode, float(rcond))
    if key not in _replicated_chol_cache:
        _re = ridge_extra
        _rc = rcond
        @partial(jax.jit, out_shardings=out_sh)
        def _fn(C):
            def _ridged_chol(C_log):
                # Replicate the logical block so every device factors the
                # WHOLE matrix — this is what makes the factor grid-agnostic
                # (the distributed potrf's block-cyclic accumulation is not).
                C_log = jax.lax.with_sharding_constraint(C_log, rep_sh)
                tr = jnp.abs(jnp.trace(C_log, axis1=-2, axis2=-1))
                # Floor (1e-14·|tr|, bit-identical to the historical path)
                # + opt-in conditioning term (ε·|tr|/n).  Per-q scalars.
                ridge_scalar = (1e-14 * tr + _re * tr / n_log)[:, None, None]
                ridge = ridge_scalar * jnp.eye(n_log, dtype=C_log.dtype)[None, :, :]
                return jnp.linalg.cholesky(C_log + ridge)

            def _rank_trunc_factor(C_log):
                # WHY THIS FEATURE EXISTS: the charge CCT near-singularizes when
                # n_μ over-completes the pair-density rank (κ~1e13); plain
                # Cholesky then amplifies ULP/mesh/nband roundoff into O(1) V_q
                # errors that GN-PPM magnifies to tens of eV.  Rank-truncation
                # DROPS eigenvalues < zeta_rcond·λ_max (the near-null
                # directions) → a conditioned, mesh-invariant ζ = C⁺Z.
                C_log = jax.lax.with_sharding_constraint(C_log, rep_sh)
                lam, V = jnp.linalg.eigh(C_log)      # Hermitian-SPD, λ ascending
                lam_max = lam[..., -1:]              # (nq,1) largest λ per q
                keep = lam > (_rc * lam_max)         # near-null cut
                # B = V·diag(1/√λ_kept) ⇒ B Bᴴ = Σ_{keep} vᵢvᵢᴴ/λᵢ = C⁺.
                # Double-``where`` keeps rsqrt off the dropped (tiny/≤0) modes.
                inv_sqrt = jnp.where(
                    keep, jax.lax.rsqrt(jnp.where(keep, lam, 1.0)), 0.0)
                return V * inv_sqrt[..., None, :].astype(V.dtype)

            factor_fn = (_rank_trunc_factor if mode == 'rank_truncate'
                         else _ridged_chol)
            F_log = solve_at_logical(
                factor_fn, n_log, (C,), pad_axes=(-2, -1))
            # Pad-block factor = identity (√1 = 1 for L; for B the pad block
            # is sliced off in the back-solve, so identity is just a
            # non-singular filler): re-embed via the shared helper (no-op
            # when n_log == n_rmu).
            return _identity_pad_block_diagonal(
                F_log, n_rmu_logical=n_log, mesh_xy=mesh_xy)
        _replicated_chol_cache[key] = _fn
    return _replicated_chol_cache[key](C_q)


# Largest REPLICATED q-batch handed to one dense factor call.  The factor is
# per-q independent (one eigh / cholesky per matrix), but its device workspace
# scales with the batch: measured ~0.30 GB per q at n_μ = 2416, so the full-BZ
# MoS2 12×12 stack (nq = 144) asks XLA for a single 42.55 GB allocation and
# dies on an 80 GB card, while the IBZ stack (nq = 74) fits.  Batching keeps the
# workspace bounded by nq_chunk instead of nq — the q axis is not a physics
# knob here, so the split is invisible to the result.
_REPLICATED_FACTOR_MAX_BATCH_BYTES = 4 * 1024**3


def _replicated_factor_q_chunk(nq: int, n_rmu: int) -> int:
    """q-batch size for :func:`factor_c_q_replicated_batched`."""
    per_q = max(1, int(n_rmu) ** 2 * 16)
    return max(1, min(int(nq), _REPLICATED_FACTOR_MAX_BATCH_BYTES // per_q))


def factor_c_q_replicated_batched(
    C_q: jax.Array, mesh_xy: Mesh, n_rmu_logical: int, **kw
) -> jax.Array:
    """:func:`_factor_c_q_replicated` over q in bounded batches.

    Per-q independent, so concatenating the batches reproduces the one-shot
    call; only the XLA workspace differs.  A single batch (every stack that
    already fitted) takes the identical code path it always did.
    """
    nq, n_rmu, _ = C_q.shape
    step = _replicated_factor_q_chunk(nq, n_rmu)
    if step >= nq:
        return _factor_c_q_replicated(C_q, mesh_xy, n_rmu_logical, **kw)
    parts = [_factor_c_q_replicated(C_q[q0:min(q0 + step, nq)], mesh_xy,
                                    n_rmu_logical, **kw)
             for q0 in range(0, nq, step)]
    return jax.device_put(jnp.concatenate(parts, axis=0),
                          NamedSharding(mesh_xy, P(None, 'x', 'y')))


def factor_c_q(
    C_q: jax.Array,
    mesh_xy: Mesh,
    block_size: int = None,
    vertex_mu_L: int = 0,
    n_rmu_logical: int | None = None,
    solver_kind: str = 'auto',
    zeta_ridge: float = 0.0,
    zeta_rcond: float = 1e-8,
) -> jax.Array:
    """
    Compute system-matrix L_q from CCT matrix.

    For ``vertex_mu_L == 0`` (standard spin-traced path) the CCT is
    Hermitian positive-definite (modulo numerical noise); we run the
    optimized 2D blocked Cholesky and return the lower-triangular
    factor.  Downstream :func:`solve_zeta` then does two
    triangular solves per-q.

    For ``vertex_mu_L != 0`` (transverse Lorentz channels γ̃^i, i∈{1,2,3})
    the CCT is Hermitian but **indefinite** — Cholesky NaNs and the LU
    fallback in :func:`solve_zeta` is required.  In this case
    we skip the factorization here and pass C_q through unchanged; the
    solve routine consumes it via ``jnp.linalg.solve`` on a per-q-batch
    basis (one LU per call, small enough that explicit
    ``lu_factor`` + ``lu_solve`` reuse buys nothing).

    Padded-input path (``n_rmu_logical < C_q.shape[-1]``):
    n_rmu may be padded to mesh divisibility at the boundary so the
    ``P(None, 'x', 'y')`` input sharding is admissible at any logical
    centroid count (e.g. n_rmu_logical = 661 prime → padded to 672 on
    a 4×4 mesh).  By the Phase 3a contract the trailing pad rows/cols
    of C_q are exact zeros (bilinear in zero-padded ψ).  We add
    identity ONLY to the pad-block diagonal in-place — turning C_q
    into a block-diagonal ``[C_log 0; 0 I_pad]`` matrix — and then
    run the same sharded Cholesky / LU path the divisible case uses.
    Cholesky of an identity-padded matrix produces a factor whose
    logical block matches the logical-only factor in exact
    arithmetic; in floating point the match is ≤1e-7 rel (blocked
    implementations regroup partial sums when the extent changes —
    see ``_identity_pad_block_diagonal``).  The pad-block factor is
    exactly identity; the back-solve's pad rows of ζ come out as zero
    (because Z's pad rows are zero by the same bilinear argument).
    For the indefinite transverse channels the exact-arithmetic
    guarantee FAILS in floating point (near-null-mode amplification —
    ROOT_CAUSE.md 2026-07-08), so ``solve_zeta`` slices that solve
    back to the logical extent.  On single-device meshes the dense
    Cholesky below also factorises at the logical extent and
    re-embeds, making the charge factor pad-extent-invariant at P=1
    (the fixed-P invariance gate).

    This is NOT ridge regularisation of C_q.  The logical block is
    untouched; identity is added ONLY to the pad-block diagonal.

    Output sharding is ``P(None, 'x', 'y')`` natively at the padded
    extent — no replication, no slice + embed gymnastics, the chol
    stays sharded across the mesh.

    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix at PADDED μ extent, sharded
            ``P(None, 'x', 'y')``.  ``n_rmu == n_rmu_padded`` (== ∏ p_a
            of the device mesh) so the existing 2D-blocked path
            applies.
        mesh_xy: 2D device mesh.
        block_size: Tile block size (auto if None).
        vertex_mu_L: Lorentz vertex index (0 = spin-traced PSD path,
            1/2/3 = transverse indefinite path).
        n_rmu_logical: Logical centroid count.  When given and
            strictly less than ``C_q.shape[-1]``, the pad-block
            diagonal is set to identity before factorisation.
            ``None`` (default) skips the identity-pad: input == output
            extent and the matrix is assumed to be PSD on its full
            extent (legacy mesh-divisible path).
        zeta_rcond: rank-truncation cutoff for the
            ``'replicated_rank_truncate'`` charge factor (drop
            eigenvalues < ``zeta_rcond·λ_max``).  Ignored by the Cholesky
            paths.  ``LORRAX_ZETA_RCOND`` overrides it for tuning.

    Returns:
        L_q: ``(nq, n_rmu, n_rmu)`` at PADDED extent, sharded
        ``P(None, 'x', 'y')``.  For ``vertex_mu_L == 0``: the Cholesky
        factor (block-diagonal ``[L_log 0; 0 I_pad]``) for the cholesky
        paths, or the rank-revealing pseudo-inverse factor ``B``
        (``B Bᴴ = C⁺``) for ``'replicated_rank_truncate'``; passthrough
        identity-padded CCT for ``vertex_mu_L ≠ 0``.
    """
    nq, n_rmu, n_rmu2 = C_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"
    if n_rmu_logical is None:
        n_rmu_logical = n_rmu
    if n_rmu_logical > n_rmu:
        raise ValueError(
            f"n_rmu_logical={n_rmu_logical} exceeds input extent {n_rmu}")

    # Pad-block identity in-place.  No-op when n_rmu_logical == n_rmu.
    # See ``_identity_pad_block_diagonal`` for why the logical block of
    # the resulting factorisation is bit-identical to a logical-only
    # factorisation.
    C_q = _identity_pad_block_diagonal(
        C_q, n_rmu_logical=n_rmu_logical, mesh_xy=mesh_xy)

    # Indefinite-CCT path: skip the Cholesky outright.  ``solve_zeta``
    # consumes C_q directly via the SVD pseudoinverse / pivoted-LU
    # branch.  After identity-pad the matrix is non-singular at the
    # padded extent; the indefinite logical block is untouched.
    if int(vertex_mu_L) != 0:
        return C_q

    solver_kind = _resolve_solver_kind(mesh_xy, vertex_mu_L=0, solver_kind=solver_kind)

    Pr = mesh_xy.shape['x']
    Pc = mesh_xy.shape['y']

    # Replicated dense factor — the mesh-INVARIANT charge factor.  Fires for
    # the 'replicated_cholesky' / 'replicated_rank_truncate' auto picks
    # (fit-size n_μ on any mesh) AND for any single-device / 1-D-degenerate
    # mesh, where a dense factor is the only option (the 2D-blocked shard_map
    # kernel needs a true 2-D mesh; cuSolverMp needs one process per device).
    # ONE code path — see _factor_c_q_replicated for why the factor is
    # grid-agnostic and for the rank_truncate vs cholesky choice.  On 1×1
    # meshes this also sidesteps a JAX 0.9 shard_map+scan carry-type failure
    # in the blocked kernel.
    if (solver_kind in ('replicated_cholesky', 'replicated_rank_truncate')
            or mesh_xy.devices.size == 1 or (Pr == 1 and Pc == 1)):
        _mode = ('rank_truncate' if solver_kind == 'replicated_rank_truncate'
                 else 'cholesky')
        return factor_c_q_replicated_batched(
            C_q, mesh_xy, n_rmu_logical, zeta_ridge=zeta_ridge,
            charge_zeta_solve=_mode, zeta_rcond=zeta_rcond)

    if solver_kind == 'cusolvermp_cholesky':
        # Distributed potrf on C_q at P(None,'x','y'); returns the raw
        # lower-triangular factor.  Downstream solve_zeta rebuilds the
        # CusolverMpBatchedLowerL handle and dispatches to potrs.
        from ffi.cusolvermp import batched_distributed_cholesky
        L_handle = batched_distributed_cholesky(C_q, mesh=mesh_xy)
        return L_handle.raw

    if solver_kind == 'slate_cholesky':
        # SLATE ``potrf`` — the portable (non-NVIDIA-capable library)
        # backend, explicit-request only (see _resolve_solver_kind_charge).
        # One whole-mesh block-cyclic factorization per q: SLATE's
        # *batched* API distributes the batch over the mesh 'x' axis
        # (needs nq % px == 0), which doesn't match this call site's
        # replicated-q layout — a per-q loop over nq ≲ tens of matrices
        # is the correct shape here.  ``to_jax_lower`` returns a
        # conventional row-major L at P('x','y'), so downstream
        # ``solve_zeta`` consumes it through the SAME triangular-solve
        # branch as 'sharded_cholesky' — no solve-side changes.
        # (Wiring slate::trsm for the back-solve is a perf follow-up.)
        # n_rmu here is the PADDED extent (divisible by px·py, hence by
        # each axis individually), so SLATE's divisibility contract
        # always holds.
        from ffi.slate import distributed_cholesky as _slate_potrf
        L_rows = [
            _slate_potrf(C_q[iq], mesh=mesh_xy).to_jax_lower()
            for iq in range(nq)
        ]
        return jax.lax.with_sharding_constraint(
            jnp.stack(L_rows, axis=0),
            NamedSharding(mesh_xy, P(None, 'x', 'y')))

    # 2D-blocked path: requires n_rmu divisible into mesh-friendly tiles.
    # The caller is expected to pass C_q at PADDED μ extent
    # (n_rmu_padded ≡ ∏ p_a is mesh-product divisible), in which case
    # this always succeeds.  ``n_rmu_logical`` only adjusts the
    # identity-pad above; it doesn't affect this code path.
    try:
        if block_size is None:
            block_size, J = compute_block_size_for_2d_cholesky(n_rmu, Pr, Pc)
        else:
            J = n_rmu // block_size
    except ValueError as exc:
        raise ValueError(
            f"factor_c_q: n_rmu={n_rmu} is not 2D-blocked-Cholesky "
            f"compatible with mesh {Pr}×{Pc} ({exc}). Pass C_q at "
            f"PADDED μ extent (round up to ∏ p_a = world_size) so "
            f"the 2D-blocked path applies."
        ) from exc

    # Get or build cached Cholesky function
    cache_key = ('chol_2d', id(mesh_xy), J, block_size)
    if cache_key not in _chol_2d_cache:
        _chol_2d_cache[cache_key] = cholesky_2d_batched(mesh_xy, J, block_size)
    chol_fn = _chol_2d_cache[cache_key]

    # Convert to tiles
    C_q_tiles = dense_to_tiles(C_q, block_size)
    tiles_shard = NamedSharding(mesh_xy, P(None, 'x', 'y', None, None))
    C_q_tiles = jax.lax.with_sharding_constraint(C_q_tiles, tiles_shard)

    # 2D blocked Cholesky
    L_q_tiles = chol_fn(C_q_tiles)

    # Convert back to dense, keep sharded
    L_q_dense = tiles_to_dense(L_q_tiles, block_size)
    L_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    L_q_dense = jax.lax.with_sharding_constraint(L_q_dense, L_shard)

    return L_q_dense


# Cache for solve function
_solve_cache = {}  # zeta-solve kernel


def _reshard_zeta_mu_X_r_Y_to_mu_XY(zeta: jax.Array, mesh_xy: Mesh) -> jax.Array:
    """Reshard (q_, μ_X, r_Y) → (q_, μ_XY, r_) for the cuSolverMp branches.

    Single mesh axis ``'y'`` moves from the r-axis to the μ-axis (where
    it joins ``'x'`` to form a flat tuple).  All other shardings stay.

    Downstream consumer ``accumulate_rchunk_to_gflat`` wants ζ
    μ-flat-sharded so the FFT box and gflat-accumulator both live at
    ``P(None, ('x','y'), None)``; landing ζ in that layout here means
    the FFT runs sharding-preserving (no further reshard, no
    replicated FFT box).

    Note on overhead: tried both ``@jax.jit(donate_argnums=(0,))``
    closure-wrapping (matching the ``_reshard_z`` pattern above) and a
    module-level decorator with ``static_argnums``.  Neither flipped
    XLA's ``is_sync`` flag on the emitted all-to-all from ``true`` to
    ``false``, and runtime cost was the same either way (~3 ms/call in
    the trace).  The bare ``with_sharding_constraint`` is the simplest
    form for the same emitted HLO; trace shows the reshard is not on
    the critical path at MoS2 3×3 scale.
    """
    return jax.lax.with_sharding_constraint(
        zeta, NamedSharding(mesh_xy, P(None, ('x', 'y'), None)))


def _reshard_zeta_r_XY_to_mu_XY(zeta: jax.Array, mesh_xy: Mesh) -> jax.Array:
    """Reshard (q_, μ_, r_XY) → (q_, μ_XY, r_) for the shard_map branch.

    The shard_map triangular-solve naturally lands ζ at
    ``P(None, None, ('x','y'))`` because the solve is parallelised over
    r-columns.  The downstream FFT wants μ-sharded.  Two mesh axes have
    to move on the (μ, r) data axes; SPMD's all-to-all planner only
    handles one mesh axis at a time, so we stage through the cuSolverMp
    intermediate ``P(None, 'x', 'y')`` to keep every step a single-axis
    all-to-all primitive ``(a_X, b) → (a, b_X)``:

      Step 1  (q_, μ_, r_XY) → (q_, μ_X, r_Y)   ['x' moves r → μ]
      Step 2  (q_, μ_X, r_Y) → (q_, μ_XY, r_)   ['y' moves r → μ]
    """
    zeta = jax.lax.with_sharding_constraint(
        zeta, NamedSharding(mesh_xy, P(None, 'x', 'y')))
    return jax.lax.with_sharding_constraint(
        zeta, NamedSharding(mesh_xy, P(None, ('x', 'y'), None)))


def solve_zeta(
    L_q: jax.Array,
    Z_q: jax.Array,
    mesh_xy: Mesh,
    q_chunk_size: int = 1,
    vertex_mu_L: int = 0,
    solver_kind: str = 'auto',
    cct_trace_per_q: jax.Array | None = None,
    n_rmu_logical: int | None = None,
) -> jax.Array:
    """
    Solve for zeta_q given pre-computed system matrix from
    :func:`factor_c_q`.

    For ``vertex_mu_L == 0`` ``L_q`` is the lower-triangular Cholesky
    factor of CCT and the inner solve is two triangular substitutions
    (``L y = Z`` then ``L^H ζ = y``).  This is the historical fast
    path — bit-identical to the previous implementation.

    For ``vertex_mu_L != 0`` ``L_q`` is the *unfactored* CCT^μ matrix.
    The transverse-channel CCT^μ is Hermitian but indefinite — γ̃^i ⊗
    γ̃^i has both signs of eigenvalue (eigenvalues of α^i are ±1), so
    Cholesky is invalid.  We solve via pivoted LU (``jnp.linalg.solve``)
    with a small diagonal ridge ``ε·|tr(L)|/n_rmu`` (ε = 1e-12) added
    to lift TRS-paired near-zero modes safely above the LU stability
    floor.  Bunch-Kaufman LDL^T would be the natural Hermitian-
    indefinite factorization but JAX doesn't expose it; pivoted LU is
    numerically equivalent for our purposes.

    Uses q-chunked all-gather strategy: gather B_q matrices at a time,
    then solve all B_q systems in parallel using vmap.

    Memory trade-off:
    - q_chunk_size=1: Minimum memory (one matrix replicated at a time)
    - q_chunk_size=nq: Maximum parallelism (all matrices replicated)

    Args:
        L_q: (nq, n_rmu, n_rmu) Cholesky factor (μ_L=0) or raw CCT
             (μ_L=1,2,3), sharded P(None, 'x', 'y')
        Z_q: (nq, n_rmu, n_zchunk) ZCT matrix, sharded P(None, 'x', 'y')
             or P(None, None, ('x','y')) if caller already resharded
        mesh_xy: 2D device mesh
        q_chunk_size: Number of q-points to solve simultaneously (default 1)
        vertex_mu_L: Lorentz vertex index — selects Cholesky-back-solve
                     vs jnp.linalg.solve.  Output sharding is identical
                     in both branches.
        solver_kind: 'auto' (default) defers to :func:`_resolve_solver_kind`;
                     explicit values are 'replicated_cholesky' (mesh-
                     invariant dense factor from :func:`_factor_c_q_replicated`;
                     back-solve shares the 'sharded_cholesky' per-q
                     triangular path — L is replicated, r-columns sharded,
                     so ζ is grid-agnostic), 'sharded_cholesky' (legacy 2D
                     blocked chol + per-q triangular solve), 'lu' (per-q
                     pivoted-LU for transverse channels),
                     'cusolvermp_cholesky' (distributed potrs via FFI),
                     'cusolvermp_lu' (distributed getrf+getrs via FFI
                     for the transverse channels), or
                     'replicated_rank_truncate' (charge rank-truncation:
                     ``L_q`` is the pseudo-inverse factor B, back-solve is
                     the matmul ζ = B(BᴴZ)).  'replicated_cholesky',
                     'sharded_cholesky' and 'replicated_rank_truncate' all
                     take the general shard_map back-solve branch below
                     (none matches the cuSolverMp/scalapack guards).
        n_rmu_logical: Logical centroid count.  When given and smaller
                     than the padded input extent, every per-q dense
                     solve (pivoted LU AND the per-q triangular
                     back-solve) is μ-SLICED to this extent before the
                     factorisation and the ζ pad rows are zero-filled
                     after.  This is load-bearing for device-count
                     invariance: solving the identity-padded system at
                     the padded extent makes ζ depend deterministically
                     on the pad extent (= on the device count), with
                     O(1) amplification in the near-null transverse
                     modes (reports/device_invariance_2026-07-08/
                     ROOT_CAUSE.md).  ``None`` keeps the padded extent
                     (back-compat for mesh-divisible callers).

    Returns:
        zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, ('x','y'), None)
                — μ-axis flat-sharded across the ('x','y') mesh product,
                r-axis replicated.  This is the layout the downstream
                G-flat FFT (``accumulate_rchunk_to_gflat``) wants:
                each rank owns a μ-slab over the full r-extent, so the
                per-rank cuFFT runs locally without resharding.
    """
    nq, n_rmu, _ = L_q.shape
    _, _, n_zchunk = Z_q.shape
    n_log = int(n_rmu_logical) if n_rmu_logical is not None else int(n_rmu)
    if n_log > n_rmu:
        raise ValueError(
            f"solve_zeta: n_rmu_logical={n_log} exceeds input extent {n_rmu}")
    mu_pad = n_rmu - n_log

    solver_kind = _resolve_solver_kind(mesh_xy, vertex_mu_L, solver_kind)

    if solver_kind in ('cusolvermp_lu', 'scalapack_lu') and mu_pad:
        Px_ = int(mesh_xy.shape['x'])
        Py_ = int(mesh_xy.shape['y'])
        if (n_log % Px_) or (n_log % Py_):
            # The indefinite solve MUST run at the logical extent (see
            # ``n_rmu_logical`` above), but the distributed block-cyclic
            # descriptors need n % Px == n % Py == 0.  Fall back to the
            # per-q replicated LU, which runs at any logical extent.
            import warnings
            warnings.warn(
                f"solve_zeta: n_rmu_logical={n_log} not divisible by the "
                f"{Px_}x{Py_} mesh axes; transverse LU falls back from "
                f"the distributed backend to the per-q jnp.linalg.solve "
                f"path so the solve can run at the logical extent.")
            solver_kind = 'lu'

    if solver_kind == 'cusolvermp_cholesky':
        # Distributed potrs: Z stays at P(None,'x','y'), no input reshard
        # and no all-gather of L.  Output reshards to P(None, ('x','y'),
        # None) so the downstream G-flat accumulator receives ζ
        # μ-flat-sharded (the FFT layout it actually wants — see
        # accumulate_rchunk_to_gflat / common.wfn_transforms).
        from ffi.cusolvermp import batched_distributed_potrs, CusolverMpBatchedLowerL
        Px = int(mesh_xy.shape['x'])
        Py = int(mesh_xy.shape['y'])
        # potrs requires Z's last dim divisible by Py (see pad_last_axis_to).
        Z_q, n_zchunk = pad_last_axis_to(Z_q, Py)
        needs_padding = int(Z_q.shape[-1]) != n_zchunk
        # Re-attach handle metadata (the raw array carries no shape/grid info).
        L_handle = CusolverMpBatchedLowerL(
            raw=L_q, mesh=mesh_xy, n=int(n_rmu),
            mb=int(n_rmu) // Px, nb=int(n_rmu) // Py, nbatch=int(nq),
        )
        zeta_xy = batched_distributed_potrs(L_handle, Z_q, mesh=mesh_xy)
        # Natural potrs output sharding: P(None, 'x', 'y') = (q_, μ_X, r_Y).
        # Target: P(None, ('x','y'), None) = (q_, μ_XY, r_).
        # Single mesh axis 'y' moves from r-axis to μ-axis (joining 'x'
        # there) — one all-to-all on 'y' between data axes 1 and 2.
        zeta_out = _reshard_zeta_mu_X_r_Y_to_mu_XY(zeta_xy, mesh_xy)
        if needs_padding:
            return zeta_out[:, :, :n_zchunk]
        return zeta_out

    if solver_kind in ('cusolvermp_lu', 'scalapack_lu'):
        # Distributed getrf+getrs for the transverse channels.  L_q here
        # is the *unfactored* CCT^μ (Hermitian indefinite) — factor_c_q
        # passes it through.  Same input sharding, output reshard, and
        # column padding pattern as the cholesky branch.  The two
        # backends share this branch verbatim — identical call contract;
        # scalapack is the host/CPU-backend twin (Cray LibSci).
        if solver_kind == 'scalapack_lu':
            from ffi.scalapack import batched_distributed_solve_lu
        else:
            from ffi.cusolvermp import batched_distributed_solve_lu
        Px = int(mesh_xy.shape['x'])
        Py = int(mesh_xy.shape['y'])
        # getrs descB requires NRHS % Py == 0 (see pad_last_axis_to).
        Z_q, n_zchunk = pad_last_axis_to(Z_q, Py)
        needs_padding = int(Z_q.shape[-1]) != n_zchunk

        def _dist_ridged_lu(L_log, Z_log):
            # The μ-slice to the LOGICAL extent (via solve_at_logical;
            # guarded above: n_log divides both mesh axes on this path)
            # is load-bearing: L/Z pad rows are exact zeros (+ identity
            # pad diag on L), so the sliced system IS the logical
            # system; solving at the padded extent instead changes ζ_T
            # wholesale — pad-shape LU roundoff is amplified O(1) in
            # the near-null transverse modes (ROOT_CAUSE.md 2026-07-08,
            # Manifestation 1).
            xy_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
            L_log = jax.lax.with_sharding_constraint(L_log, xy_shard)
            Z_log = jax.lax.with_sharding_constraint(Z_log, xy_shard)
            # Per-q ridge ε·|tr(L_log)|/n_log — same lift as the legacy
            # 'lu' branch, to keep TRS-paired near-zero modes above the
            # LU stability floor without perturbing well-conditioned
            # ones.  ``cct_trace_per_q`` is precomputed once per channel
            # by the caller (fit_zeta_to_h5) over the LOGICAL block —
            # computing it inline re-fires an all-reduce across the
            # (μ_X, ν_Y) sharding on every r-chunk (~17 s GPU stream at
            # MoS2 3×3 bispinor).  Both the trace and the denominator
            # must be LOGICAL quantities or the ridge (hence ζ) depends
            # on the pad extent.
            LU_RIDGE = 1e-12
            trace_per_q = (cct_trace_per_q if cct_trace_per_q is not None
                           else jnp.einsum('qii->q', L_log))
            ridge = (LU_RIDGE * jnp.abs(trace_per_q) / n_log)[:, None, None]
            eye_n = jnp.eye(n_log, dtype=L_log.dtype)[None, :, :]
            return batched_distributed_solve_lu(
                L_log + ridge * eye_n, Z_log, mesh=mesh_xy)

        zeta_xy = solve_at_logical(_dist_ridged_lu, n_log, (L_q,), Z_q)
        if mu_pad:
            zeta_xy = jax.lax.with_sharding_constraint(
                zeta_xy, NamedSharding(mesh_xy, P(None, 'x', 'y')))
        # Reshard rationale identical to the cholesky branch above.
        zeta_out = _reshard_zeta_mu_X_r_Y_to_mu_XY(zeta_xy, mesh_xy)
        if needs_padding:
            return zeta_out[:, :, :n_zchunk]
        return zeta_out

    # Compute padding needed for even sharding across all devices
    total_devices = mesh_xy.devices.size
    n_zchunk_padded = round_up(n_zchunk, total_devices)
    needs_padding = n_zchunk_padded != n_zchunk

    z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    L_rep_shard = NamedSharding(mesh_xy, P(None, None))
    L_batch_rep_shard = NamedSharding(mesh_xy, P(None, None, None))  # (B_q, n_rmu, n_rmu)
    q_batch = min(q_chunk_size, nq)
    nq_padded = round_up(nq, q_batch)

    # Dispatch on solver_kind (already resolved above) — independent of
    # vertex_mu_L so the rchunk kernel cache can collapse transverse
    # channels with the same solver_kind into a single compile.
    use_lu = (solver_kind == 'lu')
    # ``use_lu`` selects a pivoted-LU back-solve for transverse channels
    # (γ̃^i, i∈{1,2,3}).  CCT^μ for those channels is Hermitian but
    # indefinite (γ̃^i ⊗ γ̃^i has both signs of eigenvalue), so Cholesky
    # is invalid.  Bunch-Kaufman LDL^T would be the natural fit for
    # Hermitian indefinite, but JAX doesn't expose it; ``jnp.linalg.solve``
    # uses LU with partial pivoting which handles indefinite matrices
    # correctly as long as they aren't actually singular.  We keep a
    # small ridge ``LU_RIDGE·trace/n_rmu`` on the diagonal to lift any
    # near-zero modes from TRS-paired band cancellations safely above
    # the LU stability floor — small enough not to perturb the
    # well-conditioned modes.
    LU_RIDGE = 1e-12

    # ``use_rank_trunc`` selects the matmul back-solve for the charge
    # rank-truncation factor: ``L_q`` is then the pseudo-inverse factor B
    # (B Bᴴ = C⁺) from ``_factor_c_q_replicated``, so ζ = C⁺Z = B(BᴴZ) is
    # two matmuls, NOT a triangular solve (C⁺ is rank-deficient — its
    # inverse does not exist, so the tri-solve would be wrong).
    use_rank_trunc = (solver_kind == 'replicated_rank_truncate')

    # Cache key for solve function (includes q_chunk_size and padded size).
    # ``use_lu`` / ``use_rank_trunc`` partition the cache so the three
    # back-solve compiles don't collide on the same key.  ``n_log`` is
    # closure state of the kernels below (the slice extent), so it keys the
    # cache too.
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu, n_log,
                 n_zchunk_padded, q_chunk_size, bool(use_lu),
                 bool(use_rank_trunc))

    def _ridge_indef_solve(L: jax.Array, Z: jax.Array) -> jax.Array:
        """Solve (L + ε·tr(L)/n · I) · ζ = Z via pivoted LU at the
        LOGICAL μ extent (slice/zero-refill via ``solve_at_logical`` —
        load-bearing: LU at the identity-padded extent yields a
        different, per-extent-deterministic ζ_T, amplified O(1) in the
        near-null transverse modes; ROOT_CAUSE.md 2026-07-08).

        ε = ``LU_RIDGE`` (1e-12) on the logical trace/denominator: well
        below any physically meaningful eigenvalue but above the
        partial-pivoting floor, so LU stays stable on TRS-paired
        near-zero modes without perturbing the rest of the spectrum.
        """
        def _ridged_lu(L_log, Z_log):
            ridge = LU_RIDGE * jnp.abs(jnp.trace(L_log)) / n_log
            L_reg = L_log + ridge * jnp.eye(n_log, dtype=L.dtype)
            return jnp.linalg.solve(L_reg, Z_log)
        return solve_at_logical(_ridged_lu, n_log, (L,), Z)

    def _tri_solve_logical(L: jax.Array, Z: jax.Array) -> jax.Array:
        """Charge-channel two-triangular back-solve at the LOGICAL μ
        extent (same ``solve_at_logical`` rationale — the
        well-conditioned Cholesky back-solve only wobbles ≤1e-7 under a
        pad-extent change, but at fixed shape it is exactly
        pad-invariant, which the fixed-P invariance gate requires).
        L is the block-diag ``[L_log 0; 0 I]`` factor; its logical
        block is exactly the factor of the logical system."""
        def _chol_backsolve(L_log, Z_log):
            y = jax.scipy.linalg.solve_triangular(L_log, Z_log, lower=True)
            return jax.scipy.linalg.solve_triangular(
                L_log.conj().T, y, lower=False)
        return solve_at_logical(_chol_backsolve, n_log, (L,), Z)

    def _pinv_matmul_logical(B: jax.Array, Z: jax.Array) -> jax.Array:
        """Charge rank-truncation back-solve at the LOGICAL μ extent:
        ζ = C⁺Z = B(BᴴZ), two matmuls (B is the pseudo-inverse factor,
        B Bᴴ = C⁺).  ``solve_at_logical`` slices to the logical block
        (dropping the identity pad — never inverted, so no LU/tri
        amplification) and zero-refills ζ's pad rows."""
        def _mm(B_log, Z_log):
            return B_log @ (B_log.conj().T @ Z_log)
        return solve_at_logical(_mm, n_log, (B,), Z)

    if cache_key not in _solve_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            if use_rank_trunc:
                # Charge C⁺ pseudo-inverse factor: matmul back-solve.
                return _pinv_matmul_logical(L, Z_cols)
            if use_lu:
                # Indefinite CCT^μ: pivoted-LU back-solve with ridge.
                return _ridge_indef_solve(L, Z_cols)
            return _tri_solve_logical(L, Z_cols)

        # Vectorized solve for a batch of q-points
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None, None), P(None, None, ('x', 'y'))),
                 out_specs=P(None, None, ('x', 'y')))
        def _sharded_cho_solve_batch(L_batch: jax.Array, Z_batch: jax.Array) -> jax.Array:
            """Solve (B_q, n_rmu, n_rmu) @ (B_q, n_rmu, n_cols) -> (B_q, n_rmu, n_cols)"""
            if use_rank_trunc:
                # C⁺ factor matmul back-solve, per-q vmapped (same reshard
                # plan as the Cholesky/LU paths so the caller is agnostic).
                return jax.vmap(_pinv_matmul_logical)(L_batch, Z_batch)
            if use_lu:
                # ``jnp.linalg.solve`` is natively batched on the leading
                # axis and dispatches one LU factorization per q.  Same
                # vmap structure as the Cholesky path so reshard plans
                # match.  We vmap the ridge-add per-q so each LU sees
                # its own conditioning shift.
                return jax.vmap(_ridge_indef_solve)(L_batch, Z_batch)
            return jax.vmap(_tri_solve_logical)(L_batch, Z_batch)

        @partial(jax.jit, donate_argnums=(2,))
        def _solve_batch_and_update(L_batch_sharded, Z_batch_col, zeta_acc, q_start):
            """Solve one q-batch and update zeta_acc via dynamic_update_slice.
            donate_argnums=(2,) donates zeta_acc so XLA reuses its buffer."""
            L_rep = jax.lax.with_sharding_constraint(L_batch_sharded, L_batch_rep_shard)
            batch_result = _sharded_cho_solve_batch(L_rep, Z_batch_col)
            return jax.lax.dynamic_update_slice(zeta_acc, batch_result, (q_start, 0, 0))

        @jax.jit
        def _solve_all_at_once(L_q_sharded, Z_col):
            """Fast path: solve all q-points in a single batched call."""
            L_full_rep = jax.lax.with_sharding_constraint(L_q_sharded, L_batch_rep_shard)
            return _sharded_cho_solve_batch(L_full_rep, Z_col)

        _solve_cache[cache_key] = SimpleNamespace(
            solve_batch_and_update=_solve_batch_and_update,
            solve_all_at_once=_solve_all_at_once,
            sharded_cho_solve=_sharded_cho_solve,
            sharded_cho_solve_batch=_sharded_cho_solve_batch,
        )

    helpers = _solve_cache[cache_key]

    # Pad Z if needed (zeros on RHS → zero solution for those columns, harmless)
    if needs_padding:
        pad_width = n_zchunk_padded - n_zchunk
        Z_q = jnp.pad(Z_q, ((0, 0), (0, 0), (0, pad_width)), mode='constant')

    # Reshard Z once: P(None, 'x', 'y') → P(None, None, ('x','y'))
    # If the caller already resharded Z_q (to avoid keeping 3× m_zcol alive
    # across the solve loop), _reshard_z would redundantly copy the buffer.
    # Check sharding and skip if already correct.
    _target_sharding = z_col_shard
    _already_resharded = (hasattr(Z_q, 'sharding') and
                          getattr(Z_q.sharding, 'spec', None) == _target_sharding.spec)
    if _already_resharded:
        Z_col = Z_q
        del Z_q
    else:
        # Donation on Z_q is the key — the caller `del Z_q` immediately
        # after this block, so XLA can alias the input buffer for the
        # output (2× tile theoretical minimum).  Without donation SPMD
        # hits an Involuntary full rematerialization going from
        # P(None,'x','y') → P(None,None,('x','y')) because both mesh
        # axes need to re-shard at once.  Measured at Si 4×4×4 60Ry
        # (nq=64, μ=2400, B_r=12672, 2×2 mesh):
        #   direct, no donate: 31.14 GB/dev (temp 15.57 GB, Involuntary Remat)
        #   direct, donate:    15.57 GB/dev (temp 7.79 GB) -- 50% reduction
        # Two-step reshard via P('x',None,'y') intermediate.  Direct
        # P(None,'x','y') → P(None,None,('x','y')) moves both mesh axes
        # at once on the (μ, ν) data axes, which SPMD can't plan as a
        # single all-to-all and falls into Involuntary Full
        # Rematerialization (the full nq×μ×ν tensor materialised on
        # every device, ~16× per-device shard).  Staging through
        # P('x', None, 'y') parks 'x' on the leading nq axis so each
        # with_sharding_constraint moves only one mesh axis — two pure
        # all-to-alls, no all-gather inflation.  Same pattern as
        # w_isdf._get_w_solve_fn (V/χ reshard).  See lorrax_B commit
        # c0307a0 for the original Si 4×4×4 result (HLO peak
        # 68.94 → 29.94 GB on the same kernel compile).
        intermediate_shard = NamedSharding(mesh_xy, P('x', None, 'y'))
        @partial(jax.jit, donate_argnums=(0,))
        def _reshard_z(z):
            z = jax.lax.with_sharding_constraint(z, intermediate_shard)
            return jax.lax.with_sharding_constraint(z, _target_sharding)
        Z_col = _reshard_z(Z_q)
        # No-op when called inside an outer jit (tracer has no
        # block_until_ready and the outer jit syncs at its boundary).
        if not isinstance(Z_col, jax.core.Tracer):
            Z_col.block_until_ready()
        del Z_q

    # Fast path: solve all q-points at once
    if q_batch >= nq:
        result = helpers.solve_all_at_once(L_q, Z_col)
        del Z_col
        # Reshard r_XY → μ_XY so the downstream G-flat FFT runs
        # sharding-preserving (see _reshard_zeta_r_XY_to_mu_XY).
        result = _reshard_zeta_r_XY_to_mu_XY(result, mesh_xy)
        if needs_padding:
            return result[:, :, :n_zchunk]
        return result

    # Allocate output buffer
    zeta = jnp.zeros_like(Z_col)

    # Python loop with async dispatch — each call returns immediately.
    # The donation chain (output N = input N+1) ensures sequential GPU execution
    # without explicit block_until_ready. Python dispatch overhead: ~0.3ms × 8 = 2.4ms.
    # NOTE: scan(unroll=8) was attempted but OOMs — XLA pipelines adjacent unrolled
    # iterations, keeping 2× preallocated-temp alive (18.9 GB). scan without unroll
    # triggers SPMD replication of the sharded accumulator (88 GB OOM). fori_loop
    # has the same WhileOp issue. The Python loop is the only approach that gives
    # constant DUS offsets AND sequential memory reuse.
    for q0 in range(0, nq_padded, q_batch):
        q1 = q0 + q_batch
        zeta = helpers.solve_batch_and_update(L_q[q0:q1], Z_col[q0:q1], zeta, q0)

    del Z_col

    # Reshard r_XY → μ_XY so the downstream G-flat FFT runs
    # sharding-preserving (see _reshard_zeta_r_XY_to_mu_XY).
    zeta = _reshard_zeta_r_XY_to_mu_XY(zeta, mesh_xy)
    if needs_padding:
        return zeta[:, :, :n_zchunk]
    return zeta


# =============================================================================
# Jittable r-chunk body — the whole per-r-chunk iteration in one jit
# =============================================================================
#
# This is the hot kernel: load-phase FFT → reshard → streaming pair-density
# accumulate (both L and R) over all band-chunks → ZCT → reshard → solve,
# all fused into one jax.jit.  The Python driver only does persistent-state
# setup (centroids, L_q, G-space cache) and H5 I/O writes around the jit call.
#
# All "structural" configuration (band_chunk_ranges, band_range_left/right,
# actual_n_rchunk, q_chunk_size, kvecs_frac, mesh, meta) is closure state —
# the factory compiles a distinct jit per (hashable) tuple.  The typical
# r-chunk loop has exactly TWO compiled variants: the full-sized r-chunks
# and the last remainder.
#
# Dynamic inputs: the pre-loaded G-space tuple (one array per band-chunk),
# the centroid copies and L_q (persistent across the full fit), band_norms
# (normalised to jnp.ones(nb) when absent so the shape is uniform), and
# r_start as a scalar dynamic int.

_fit_one_rchunk_cache: dict = {}  # fused fit_one_rchunk kernel (cleared per-run by gw_init:703)


def _make_fit_one_rchunk_kernel(
    mesh_xy: Mesh,
    meta: Meta,
    band_chunk_ranges: tuple[tuple[int, int], ...],
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    band_range_full: tuple[int, int],
    actual_n_rchunk: int,
    q_chunk_size: int,
    kvecs_frac,
    psi_G_store,
    is_charge: bool = True,
    solver_kind: str = 'auto',
    q_irr_full_idx: np.ndarray | None = None,
):
    """Factory: returns a ``jax.jit``'d fit_one_rchunk callable closing
    over every piece of static structure + a :class:`PsiGStore` that
    supplies per-bc ψ(G) slices from host memory.

    Returned function signature::

        zeta_chunk = kernel(
            psi_l_rmuT_X_fit,
            psi_r_rmuT_X_fit,
            L_q,
            norms_l, norms_r,      # jax arrays; jnp.ones when no band_norms
            r_start_dyn,           # scalar int32
        )

    ψ(G) is NOT a jit argument.  The bc-loop fetches each chunk from
    the host-resident store via ``io_callback`` — only the currently
    active bc lives on device.  See :mod:`common.psi_G_store` for
    lifecycle + layout details.

    The inner body fully composes the load-phase FFT + reshard, the
    band-chunk pair-density streaming loop (Python-unrolled at trace),
    the ZCT, the Z_q→Z_col reshard, and the Cholesky solve.
    """
    nk_tot = meta.nk_tot
    # In-memory shapes throughout this kernel use the PADDED μ extent.
    # ψ enters at padded (Phase 3a's load_centroids contract); all
    # bilinear consumers / WSCs / inner-jit boundary checks see
    # n_rmu_padded == ∏ p_a (mesh-divisible by construction).  L_q
    # is also at padded extent (factor_c_q embeds the
    # logical-extent factor with identity in the pad block — pad rows
    # of zeta come out as zero, logical block byte-identical to a
    # pure-logical solve).  meta.n_rmu (logical) is used only at the
    # SlabIO valid_shape= seam in fit_zeta_to_h5 so on-disk
    # extent stays logical and round-trips across mesh sizes.
    n_rmu = meta.n_rmu_padded
    nspinor = meta.nspinor
    kgrid = meta.kgrid

    # Closure-side IBZ row indices (Phase B).  None → full-BZ solve
    # (back-compat for ``write_ibz_only=False``).  Otherwise a static
    # jax int32 array baked into the jit's HLO.
    if q_irr_full_idx is not None:
        q_irr_idx_j = jnp.asarray(
            np.asarray(q_irr_full_idx, dtype=np.int32))
    else:
        q_irr_idx_j = None

    # ψ enters at PADDED n_rmu.  All in-memory arrays here operate at
    # PADDED extent so the inner shard_map boundaries see divisible
    # shapes (n_rmu_padded ≡ ∏ p_a).  The Cholesky factor L_q is at
    # PADDED extent too (factor_c_q embeds the logical-extent factor
    # with identity in the pad block); the back-solve produces zeta
    # with zero pad rows, logical block byte-identical to a logical-
    # only solve.  zeta is returned at padded extent; the SlabIO write
    # uses valid_shape=meta.n_rmu so on-disk extent stays logical.

    def z_q_phase(
        psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
        norms_l, norms_r, r_start_dyn, gamma_perm, gamma_phase,
    ):
        # Pre-multiply by 1/norms so the pair-density einsum sees the
        # norm-scaled input without a per-bc divide inside the scan
        # (algebraically identical: einsum is linear in psi_X·psi_Y).
        psi_l_X_scaled = psi_l_rmuT_X_fit / norms_l[None, None, :, None]
        psi_r_X_scaled = psi_r_rmuT_X_fit / norms_r[None, None, :, None]
        gamma_mu = None if is_charge else (gamma_perm, gamma_phase)
        # Round 6 streaming pair density + IFFT + γ̃·γ̃ + FFT — single
        # shard_map; io_callback pulls per-bc ψ(G) from the host store
        # inside the lax.scan body.  Output Z_q at FULL-BZ q-shape.
        return z_q_from_psi_sm(
            psi_l_X_scaled, psi_r_X_scaled, psi_G_store,
            band_chunk_ranges=band_chunk_ranges,
            band_range_left=band_range_left,
            band_range_right=band_range_right,
            r_start_dyn=r_start_dyn,
            r_chunk_size=actual_n_rchunk,
            gamma_L=gamma_mu,
            gamma_R=gamma_mu,
            kgrid=kgrid,
            mesh_xy=mesh_xy,
        )

    def solve_phase(Z_q, L_q, cct_trace_per_q):
        # IBZ-only solve (Phase B): L_q + cct_trace come in PRE-SLICED
        # at IBZ rows (via ``symmetry_maps.slice_q_full_to_ibz``
        # upstream in ``fit_zeta_to_h5``).  Z_q is built at full BZ by
        # ``z_q_phase``, so it gets the IBZ slice here.  When
        # ``q_irr_full_idx`` is None (write_ibz_only=False), all three
        # arrays are full-BZ and the solve runs as before.
        if q_irr_idx_j is not None:
            Z_q_for_solve = Z_q[q_irr_idx_j]
        else:
            Z_q_for_solve = Z_q
        # ``n_rmu_logical=meta.n_rmu``: the per-q dense solves run at
        # the LOGICAL μ extent (ζ pad rows zero-filled after) so ζ is
        # independent of the pad extent / device count.  See
        # solve_zeta's n_rmu_logical docstring.
        return solve_zeta(
            L_q, Z_q_for_solve, mesh_xy, q_chunk_size,
            solver_kind=solver_kind,
            cct_trace_per_q=cct_trace_per_q,
            n_rmu_logical=int(meta.n_rmu))

    @jax.jit
    def _kernel(
        psi_l_rmuT_X_fit,
        psi_r_rmuT_X_fit,
        L_q,
        norms_l,
        norms_r,
        r_start_dyn,
        gamma_perm,
        gamma_phase,
        cct_trace_per_q,
    ):
        # Composed (z_q ∘ solve) under one ``@jax.jit`` — preserved for
        # the AOT memory model path which lowers a single callable.
        # Production ``fit_one_rchunk`` calls ``z_q_phase`` and
        # ``solve_phase`` directly with ``timing.section`` between, so
        # the per-r-chunk breakdown is host-visible.
        Z_q = z_q_phase(
            psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
            norms_l, norms_r, r_start_dyn, gamma_perm, gamma_phase)
        return solve_phase(Z_q, L_q, cct_trace_per_q)

    # Attach the un-fused phases so ``fit_one_rchunk`` can time them.
    _kernel.z_q_phase = z_q_phase
    _kernel.solve_phase = solve_phase
    return _kernel


def fit_one_rchunk(
    *,
    psi_G_store,
    psi_l_rmuT_X_fit,
    psi_r_rmuT_X_fit,
    L_q,
    norms_l,
    norms_r,
    r_start_dyn,
    mesh_xy: Mesh,
    meta: Meta,
    band_chunk_ranges: tuple[tuple[int, int], ...],
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    band_range_full: tuple[int, int],
    actual_n_rchunk: int,
    q_chunk_size: int,
    kvecs_frac: np.ndarray,
    vertex_mu_L: int = 0,
    solver_kind: str = 'auto',
    q_irr_full_idx: np.ndarray | None = None,
    cct_trace_per_q: jax.Array | None = None,
):
    """Entry point for the r-chunk body jit.  Caches one compiled kernel
    per distinct static configuration.

    ``psi_G_store`` is captured in the jit closure (not a jit arg) so
    the compiled kernel calls its ``_slice_local_tile_bc`` method via
    io_callback inside ``z_q_from_psi_sm``'s scan body.  The cache key
    includes ``id(psi_G_store)`` to avoid reusing a compile built against
    a different store.
    """
    # vertex_mu_L splits into two runtime/closure pieces: a structural
    # ``is_charge`` (Python bool — keys the cache, separates the chol
    # branch from the gamma-fold branch) and the gamma matrix
    # ``(perm, phase)`` (runtime jit args — μ=1/2/3 all reuse the same
    # compiled HLO since the values never appear as static constants).
    is_charge = (int(vertex_mu_L) == 0)
    gamma_perm, gamma_phase = _gamma_perm_phase_mu(vertex_mu_L)
    cache_key = (
        id(mesh_xy),
        actual_n_rchunk,
        tuple(tuple(b) for b in band_chunk_ranges),
        tuple(band_range_left), tuple(band_range_right),
        tuple(band_range_full),
        q_chunk_size,
        meta.n_rmu, meta.n_rmu_padded, meta.nk_tot, meta.nspinor,
        tuple(meta.fft_grid),
        hash(kvecs_frac.tobytes()),
        id(psi_G_store),
        bool(is_charge),
        str(solver_kind),
        (None if q_irr_full_idx is None
         else (int(q_irr_full_idx.shape[0]),
               hash(np.asarray(q_irr_full_idx,
                               dtype=np.int32).tobytes()))),
    )
    fn = _fit_one_rchunk_cache.get(cache_key)
    if fn is None:
        fn = _make_fit_one_rchunk_kernel(
            mesh_xy, meta,
            tuple(tuple(b) for b in band_chunk_ranges),
            tuple(band_range_left),
            tuple(band_range_right),
            tuple(band_range_full),
            actual_n_rchunk,
            q_chunk_size,
            kvecs_frac,
            psi_G_store,
            is_charge=bool(is_charge),
            solver_kind=str(solver_kind),
            q_irr_full_idx=q_irr_full_idx,
        )
        _fit_one_rchunk_cache[cache_key] = fn
    # cct_trace_per_q is None for the charge channel (Cholesky path
    # ignores it); pass a tiny placeholder so the jit signature is
    # uniform across channels.  Size it to match L_q's q-axis (IBZ
    # extent when ``q_irr_full_idx`` is set; full BZ otherwise) — XLA
    # would dead-arg-eliminate it anyway, but keep the spec honest.
    if cct_trace_per_q is None:
        cct_trace_per_q = jnp.zeros((int(L_q.shape[0]),),
                                    dtype=jnp.complex128)

    # Call z_q and solve phases separately so each can be wrapped in
    # ``timing.section`` for per-r-chunk breakdown.  ``block_until_ready``
    # at each phase boundary is the cost of separating them — a few
    # microseconds on small kernels, dominated by the per-phase work.
    _dbg = bool(os.environ.get("LORRAX_RCHUNK_DEBUG"))
    _t_z0 = time.perf_counter() if _dbg else 0.0
    with timing.section("zeta_fit.chunk.z_q_build"):
        Z_q = fn.z_q_phase(
            psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
            norms_l, norms_r, r_start_dyn,
            gamma_perm, gamma_phase)
        Z_q.block_until_ready()
    _t_z = (time.perf_counter() - _t_z0) if _dbg else 0.0
    _t_s0 = time.perf_counter() if _dbg else 0.0
    with timing.section("zeta_fit.chunk.solve"):
        zeta = fn.solve_phase(Z_q, L_q, cct_trace_per_q)
        zeta.block_until_ready()
    _t_s = (time.perf_counter() - _t_s0) if _dbg else 0.0
    if _dbg and jax.process_index() == 0:
        print(f"[rchunk_dbg]   z_q_build={_t_z*1000:.0f}ms "
              f"solve={_t_s*1000:.0f}ms", flush=True)
    return zeta


def _band_norms_slice(
    band_norms: np.ndarray | None, band_range: tuple[int, int], nb: int,
) -> jax.Array:
    """Slice + clamp the pseudobands weights to a ``(nb,)`` jax array.

    Divisor is ``max(1, w_n)``: low-weight pseudobands keep their
    sub-unit norm (DOS-preserving), high-weight ones are pulled back
    to unit (no dominance), zero-weight windows stay at 1.0 since the
    ``max(1, 0)=1`` floor avoids a divide-by-zero.  When
    ``band_norms`` is ``None`` (no pseudobands), returns ``jnp.ones``.
    """
    if band_norms is None:
        return jnp.ones((nb,), dtype=jnp.float64)
    lo, hi = band_range
    n_avail = max(0, min(int(band_norms.shape[0]) - lo, hi - lo))
    sliced = np.zeros((hi - lo,), dtype=np.float64)
    if n_avail > 0:
        sliced[:n_avail] = np.asarray(band_norms[lo:lo + n_avail], dtype=np.float64)
    # max(1, 0) = 1 → padded entries divide ψ by 1, leaving the zeroed
    # ψ at zero.  No divide-by-zero hazard for [n_avail:nb] tail.
    return jnp.maximum(jnp.asarray(sliced), 1.0)


# ψ(G) no longer enters the jit as a tuple of arguments.  It lives on
# host, sharded n_XY over bands in per-rank tiles — each rank owns one
# contiguous ``(nk, nb/P, ns, nx, ny, nz)`` numpy array — and the
# kernel body slices per-bc (and optionally per-k) subsets via
# :func:`common.psi_G_store.PsiGStore._slice_local_tile_bc`.  See that module
# for the two lifecycle modes (host_cache vs file_reread).

"""ISDF core primitives: ψ + centroids -> ζ interpolation vectors.

Neutral array-in / array-out core of the ISDF fit — the composable phases
``c_q_from_psi_sm`` -> ``factor_c_q`` -> ``fit_one_rchunk`` (which fuses
``z_q_from_psi_sm`` + ``solve_zeta``) plus the q=0 Gram building blocks used
by centroid selection.  Depends only on ``common/`` (Meta, timing,
gamma_matrices, cholesky_2d, fft_helpers, wfn_transforms, psi_G_store) and
(func-local) ``ffi/`` (cusolvermp).  NO ``gw`` / LorraxConfig / h5 / V_q
packaging lives here — GW and BSE are consumers.
"""
import math
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
from common.fft_helpers import compute_block_size_for_2d_cholesky
from common.wfn_transforms import to_rchunk_inner
# Distributed-linalg facade: mesh probing, guard resolution, and the ONE
# import seam for the FFI backend packages (cusolvermp / slate / scalapack).
from ffi.linalg import backend_module, mesh_is_cpu as _mesh_is_cpu, \
    plan as linalg_plan, resolve_backend as _resolve_linalg_backend


def host_rss_gb() -> float:
    """This process's resident set size in GB, from ``/proc/self/status``.

    The CPU backend returns ``None`` from ``device.memory_stats()``, so
    on a CPU mesh the ONLY faithful per-rank memory observable is the
    kernel's own RSS accounting.  Cheap (one small read, no JAX calls) —
    safe to sample inside the r-chunk loop.  Returns -1.0 where
    ``/proc`` is unavailable.
    """
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return -1.0


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
	symmetrize: bool = True,
) -> jax.Array:
	"""q=0 valence-conduction pair-product Gram from open-spin pair densities.

	``symmetrize=False`` skips the final Hermitian symmetrization (which
	requires a SQUARE G) — used by the column-blocked Gram build in
	:mod:`centroid.pivoted_cholesky`, which assembles rectangular column
	blocks and applies the identical 0.5·(G+G^H) once on the full matrix.

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
	             lhs_id, rhs_id, symmetrize)

	if cache_key not in _isdf_pipeline_cache:
		in_spin = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P('x', 'y'))
		kw_rep = NamedSharding(mesh_xy, P())
		rep = NamedSharding(mesh_xy, P())

		_lhs_id = lhs_id
		_rhs_id = rhs_id
		_symmetrize = symmetrize

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
			# can break it.  Cheap fix.  (Skipped for rectangular column
			# blocks — the blocked caller symmetrizes the assembled square.)
			if _symmetrize:
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
	  3. ``lax.all_to_all('y', split_axis=r, concat_axis=band,
	     tiled=True)`` then ``lax.all_gather('x', axis=1, tiled=True)``
	     aligns the band axis with ``psi_l_X`` / ``psi_r_X``'s
	     band-replicated layout WHILE scattering r onto 'y', so the
	     gathered slab is ``r_loc`` deep, not ``n_zchunk`` (p_y× less
	     memory; see the step-(3) comment for why this is the exact
	     movement and why band order is unchanged).  IFFT-FIRST;
	     gather-first would blow the FFT box to ~80 GB / rank.
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
		# Static per-bc Y-compaction gather table (device-invariance fix):
		# all_gather(tiled) over ('x','y') stacks P per-rank blocks of
		# `bpd_max` slots along the gathered band axis; rank r's block holds
		# this bc's global bands in its FIRST `bpd_per_bc` slots + zero pad.
		# The g_axis mask and the contiguous psi_*_X slice both assume a
		# CONTIGUOUS global band axis, so reorder the gathered slots to place
		# this bc's real bands contiguously at the front (out pos p -> src
		# slot).  Identity whenever bpd_per_bc == bpd_max (every full chunk
		# AND every chunk at P=1) -> no-op / byte-identical there.
		_y_compact_idx_np = np.zeros((n_bc, bpd_max_global), dtype=np.int32)
		for _bc in range(n_bc):
			_bpd = int(_psi_G_store._bpd_per_bc[_bc])
			if _bpd <= 0:
				continue  # all-pad bc: every slot masked downstream
			_nb_tot = _bpd * P_total  # == b_hi-b_lo (sharded load is P-divisible)
			# Precondition guard (device-invariance audit): each band-chunk
			# width MUST be world_size-divisible, else floor-div silently drops
			# bands and the populate tile assignment shape-mismatches at P>1.
			# ValueError, not assert: the fix is a user input key
			# (band_chunk_size), and an assert vanishes under `python -O`,
			# re-arming exactly the silent band-dropping this guards
			# (audit fix/zq 2026-07-28).
			if _nb_tot != (bcr[_bc][1] - bcr[_bc][0]) or _bpd > bpd_max:
				raise ValueError(
					f"z_q band chunk {_bc} width {bcr[_bc][1]-bcr[_bc][0]} is not a "
					f"multiple of world_size {P_total} (bpd_per_bc={_bpd}); set "
					f"band_chunk_size to a multiple of world_size")
			# out pos p -> src slot (strided real bands compacted to front);
			# tail (p >= _nb_tot) -> slot _bpd, a guaranteed-zero pad slot.
			_p = np.arange(bpd_max_global)
			_y_compact_idx_np[_bc] = np.where(
				_p < _nb_tot, (_p // _bpd) * bpd_max + (_p % _bpd), _bpd)
		# STATIC identity check.  When EVERY bc's compaction is the
		# identity permutation the ``jnp.take`` in the scan body is
		# mathematically a no-op -- but XLA cannot prove that from a
		# traced index array, so it materialises a SECOND copy of the
		# band-gathered FULL-r psi(r) slab:
		#     nk * bpd_max_global * ns * n_zchunk * 16 bytes
		# with NO mesh division on either axis (129 GB/rank at MoS2
		# 12x12, 160-band window, r_chunk = n_rtot = 174960 -- half of
		# the 271 GB single allocation that OOM'd job 7874236).
		# Eliding it is BIT-EXACT (take with an identity index is the
		# identity) and halves the Stage-C arena.  An all-pad bc leaves
		# its row at zeros (not identity) and correctly disables it.
		_y_compact_identity = bool(
			n_bc > 0
			and np.array_equal(
				_y_compact_idx_np,
				np.broadcast_to(
					np.arange(bpd_max_global, dtype=np.int32),
					(n_bc, bpd_max_global))))
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
			if not _y_compact_identity:
				y_compact_idx = jnp.asarray(_y_compact_idx_np)

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
			# n_zchunk).  A plain per-rank r-slice BEFORE the band gather
			# would mix r-slabs from different y-ranks at the same gathered
			# band position (r-incoherence at the einsum) — which is why
			# this used to gather bands at FULL r and slice afterwards.
			# It no longer does: step (3) below performs the band-gather and
			# the r-scatter as ONE all-to-all + all_gather, which is exactly
			# the required permutation and never materialises the
			# full-bands × full-r slab.  ``psi_Y_local`` (this rank's OWN
			# bands over full r) is still built per iter — unavoidable, since
			# other y-ranks need other r-blocks of those same bands — but it
			# is only ``bpd_max`` bands wide and XLA's scan-internal
			# allocator aliases it across iters → single slot.

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
				# (3) Band-gather AND r-scatter in one shot.
				#
				#     The old code did all_gather(('x','y')) over bands at the
				#     FULL r-chunk and only THEN sliced r — so every rank held
				#     nk · bpd_max_global · ns · n_zchunk · 16 bytes with NO mesh
				#     division on either axis (129 GB/rank at MoS2 12x12,
				#     band_chunk=160, cr=174960).  That single object is wall #0:
				#     it is what forced r_chunk down to <=81 chunks and capped the
				#     machine at n_mu ~ 4000 independently of everything else.
				#
				#     The r-slice CANNOT simply move before the gather (that mixes
				#     r-slabs from different y-ranks at the same gathered band --
				#     the r-incoherence the old comment warns about).  But the
				#     movement we actually want IS an all-to-all: rank (x,y) owns
				#     its own bands over ALL r and needs ALL bands over r-block y.
				#     Express it exactly:
				#       (3a) all_to_all on 'y'  — split r into p_y blocks, concat
				#            onto bands: (x,y) ships its y'-th r-block to (x,y')
				#            and receives every (x,y'')'s bands at r-block y.
				#       (3b) all_gather on 'x'  — the remaining band blocks, now
				#            already restricted to r-block y.
				#     Same NCCL/Gloo byte volume, pure data movement (BIT-EXACT),
				#     and the peak drops by p_y: 129 -> 12.9 GB/rank at run1 scale.
				#
				#     BAND ORDER IS PRESERVED, which is load-bearing (g_axis mask,
				#     y_compact_idx and the psi_*_X slice all assume it).  all_to_all
				#     concatenates sources in 'y' order, then all_gather concatenates
				#     in 'x' order, so block (x,y) lands at offset (x·p_y + y)·bpd_max
				#     — exactly where all_gather(('x','y'), tiled=True) put it (that
				#     flattens row-major with 'x' slowest).
				psi_Y_col = jax.lax.all_to_all(
					psi_Y_bc_local_full_r, 'y',
					split_axis=3, concat_axis=1, tiled=True)
				psi_Y_bc_full_r = jax.lax.all_gather(
					psi_Y_col, axis_name='x', axis=1, tiled=True)
				# Guard: the gathered band axis must be the full contiguous width
				# (P blocks x bpd_max).  A wrong strided/contiguous read surfaces
				# HERE as a loud shape mismatch, not silent P-dependent corruption.
				assert psi_Y_bc_full_r.shape[1] == bpd_max_global
				# (3a) Compact strided per-rank band blocks to a CONTIGUOUS
				#      global-band axis so the g_axis mask + psi_*_X slice align
				#      with the gathered Y band axis.  No-op when
				#      bpd_per_bc == bpd_max (full chunk / P=1).
				#      ELIDED when the permutation is the identity for every
				#      bc (``_y_compact_identity``): XLA cannot fold a
				#      traced-index take, so it would allocate a SECOND full
				#      band-gathered FULL-r slab (nk * bpd_max_global * ns *
				#      n_zchunk * 16, unsharded on both axes).
				if not _y_compact_identity:
					psi_Y_bc_full_r = jnp.take(
						psi_Y_bc_full_r, y_compact_idx[bc_idx], axis=1)
				# (3c) The r-slice is GONE: step (3a) already delivered this
				#      y-rank's r-block, coherently with the gathered band axis.
				assert psi_Y_bc_full_r.shape[3] == r_loc
				psi_Y_bc = psi_Y_bc_full_r
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


# ``_mesh_is_cpu`` (historical name, still exported for tests) is the
# facade's ``ffi.linalg.mesh_is_cpu`` — imported at the top of the module.


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


def _replicate_rank_truncate_ok(nq: int | None, n_rmu: int | None) -> bool:
    """True when the rank-truncating charge factor can run replicated.

    DIFFERENT CRITERION from :func:`_replicate_charge_ok`, deliberately.
    That one gates the *Cholesky* route on the whole ``(nq, μ, μ)`` stack,
    which is the right question there.  It is the WRONG question for
    ``rank_truncate``: :func:`factor_c_q_replicated_batched` already splits
    the q axis at its own ``_REPLICATED_FACTOR_MAX_BATCH_BYTES`` bound, so
    the replicated transient is ONE q-batch (≤ that bound, plus the eigh's
    own workspace) and is FLAT IN nq — it does not grow with the stack.
    Testing the stack made the resolver refuse fits that comfortably fit,
    e.g. MoS2 12×12 full-BZ (nq=144, μ=2412 → 13.4 GiB stack, but only
    ~4 GiB replicated at a time), and refusing means losing the §6a
    rank-truncation physics cure rather than losing memory.

    This can only make the production-default ``rank_truncate`` route
    REACHABLE where it previously raised; it never changes a route that
    resolves today, and it does not touch the ``cholesky`` branch at all.

    NOTE (the real μ ceiling on this route): memory is not what breaks
    here.  The factor is a dense whole-tile ``eigh`` per q (~5.5 h at
    μ=4k, ~86 h at μ=10k on 28 cores for the FULL nq sweep).  Since
    2026-08-01 the plan executes q-parallel above the fold threshold
    (:func:`_factor_c_q_replicated_qparallel` — per-rank cost
    ceil(nq/P)·μ³, bits unchanged), which divides those walls by
    min(P, nq) but cannot touch the SINGLE-q eigh: past ~4k centroids the
    route still needs a genuinely distributed eigh (SLATE/ScaLAPACK via
    ``ffi.linalg``; cuSOLVERMp is out on a rectangular mesh), not a
    bigger cap.
    """
    if nq is None or n_rmu is None:
        return False
    n = int(n_rmu)
    batch = _replicated_factor_q_chunk(int(nq), n)
    return batch * n * n * 16 <= max(_REPLICATED_CHOL_MAX_STACK_BYTES,
                                     _REPLICATED_FACTOR_MAX_BATCH_BYTES)


def _resolve_channel_ladder(
    mesh_xy: Mesh,
    override: str,
    *,
    kind_fallback: str,
    kind_cusolvermp: str,
    explicit: dict | None = None,
    auto_pre=None,
) -> str:
    """The mesh/CPU/backend decision ladder SHARED by the per-channel
    ζ-fit solver resolvers (:func:`_resolve_solver_kind_charge`,
    :func:`_resolve_solver_kind_transverse`) — written once so the two
    channels cannot drift.

    Ladder (identical for both channels):

      * ``override='off'``                 → ``kind_fallback``.
      * ``override`` in ``explicit``       → that handler decides (called
        with ``(px, py)``; owns its own FFI-availability / mesh-geometry
        checks and may raise).  Both channels route EXPLICIT
        ``'cusolvermp'`` (legacy alias ``'on'``) through the ffi.linalg
        facade — platform, compiled-capability, process-coverage and
        true-2D geometry guards — exactly like 'slate'/'scalapack'.  The
        old inline shortcut (``kind_cusolvermp if is_2d else
        kind_fallback``) silently demoted an explicit request on a 1-D
        mesh AND skipped every capability probe, so resolve could promise
        a handler the mesh/build couldn't run (doctrine 3 / quality
        pattern #6; audit fix/zq 2026-07-28).
      * auto (or unrecognised): ``auto_pre()`` first when given (the charge
        channel's replication-cap branch; returns a kind, raises, or
        returns ``None`` to fall through), then ``kind_cusolvermp`` on true
        2D non-CPU meshes (cuSOLVERMp is CUDA-only — never auto-picked on
        a CPU mesh), else ``kind_fallback``.
    """
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    is_2d = (px >= 2 and py >= 2)

    if override == 'off':
        return kind_fallback
    handler = (explicit or {}).get(override)
    if handler is not None:
        return handler(px, py)
    # auto (or unrecognised) → default policy.
    if auto_pre is not None:
        kind = auto_pre()
        if kind is not None:
            return kind
    if is_2d and not _mesh_is_cpu(mesh_xy):
        return kind_cusolvermp
    return kind_fallback


def _resolve_solver_kind_charge(
    mesh_xy: Mesh, override: str = "auto",
    n_rmu: int | None = None, nq: int | None = None,
    charge_zeta_solve: str = "cholesky",
    replicated_factor_used: bool = True,
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
      ``cusolvermp`` → force cuSolverMp (legacy alias ``on``).  EXPLICIT
                       choice via the ffi.linalg facade: refuses at
                       resolve time on a non-CUDA mesh, a build without
                       the compiled handler, or a 1-D mesh (block-cyclic
                       layout degenerates) — never a silent fallback
                       (doctrine 3; audit fix/zq 2026-07-28).
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
    def _slate(px: int, py: int) -> str:
        # Facade guard ladder (ffi.linalg.resolve): platform, compiled-
        # capability probe (a slate-less build fails HERE, at resolve time,
        # naming what IS available), process coverage, and the SLATE 1×q
        # stride-assert geometry guard.  This layer only maps the approved
        # backend to its charge-channel route string.
        _resolve_linalg_backend('cholesky', 'slate', mesh_xy)
        return 'slate_cholesky'

    def _cusolvermp(px: int, py: int) -> str:
        # EXPLICIT cusolvermp runs the same facade guard ladder as
        # 'slate': CUDA platform, compiled handler, process coverage,
        # true-2D geometry (a 1-D mesh REFUSES at resolve time instead
        # of silently returning the sharded fallback).  (audit fix/zq
        # 2026-07-28; doctrine 3 / quality pattern #6)
        _resolve_linalg_backend('cholesky', 'cusolvermp', mesh_xy)
        return 'cusolvermp_cholesky'

    # auto → default policy.  Fit-size stacks factor with the mesh-invariant
    # replicated dense factor; larger stacks keep the distributed / sharded
    # policy (fall through to the shared ladder).  ``charge_zeta_solve ==
    # 'rank_truncate'`` (the production default) selects the rank-revealing
    # eigh pseudo-inverse on the replicated route — the only route it applies
    # to (a full eigh cannot be block-cyclic).  Above the cap we therefore
    # CANNOT honour it, and we refuse rather than downgrade: the 2026-07-21
    # full-BZ 12×12 fit (13.4 GiB, just over the cap) silently fell back and
    # returned ζ 4.5× too large, rebuilding V_q to relF 16–32 instead of
    # 1.8e-15.  The replicated route is dense JAX with no FFI, so it is valid
    # on every backend including CPU -- and it is the only route carrying the
    # rank-truncation cure, so it must stay reachable there.
    def _auto_pre() -> str | None:
        if _replicate_charge_ok(nq, n_rmu):
            return ('replicated_rank_truncate'
                    if charge_zeta_solve == 'rank_truncate'
                    else 'replicated_cholesky')
        # Above the (whole-stack) Cholesky cap, rank_truncate gets its own,
        # correct criterion: the replicated transient is ONE q-batch, not the
        # stack (see _replicate_rank_truncate_ok).  Strictly widening — this
        # branch is only reached where the code raised before.
        if (charge_zeta_solve == 'rank_truncate'
                and _replicate_rank_truncate_ok(nq, n_rmu)):
            return 'replicated_rank_truncate'
        if charge_zeta_solve == 'rank_truncate' and not replicated_factor_used:
            # CAPACITY FIX (size campaign 2026-07-29, ladder notes R15.1).
            # ``distributed_zeta_solve='distributed'`` REPLACES this factor
            # wholesale with ``_factor_c_q_distributed_rank_truncate``, whose
            # layout contract never replicates an O(mu^2) object at all
            # (C_q/C+/V all P(None,'x','y'); only lambda (nq,mu) is
            # replicated).  The caller overrides ``_resolved_solver_kind`` to
            # 'distributed_rank_truncate' on the very next statement.  So
            # enforcing the REPLICATED capacity here refuses a run on the
            # size of a buffer that is never allocated -- it was capping mu at
            # sqrt(4 GiB / 16 B) = 16,384 for a route that does not use the
            # buffer.  Return the nominal kind and let the caller override.
            return 'replicated_rank_truncate'
        if charge_zeta_solve == 'rank_truncate':
            # REPORT THE QUANTITY THAT ACTUALLY FAILED (DLM campaign
            # 2026-07-29, jobs 7879700 / 7879689).  Reaching here means BOTH
            # gates above said no, and they test DIFFERENT things:
            #   _replicate_charge_ok      whole stack   nq * mu^2 * 16
            #   _replicate_rank_truncate_ok  one q-batch  batch * mu^2 * 16
            # The second is the weaker one, so IT is what binds, and the cap
            # that would clear it is the per-batch figure -- not the stack.
            # The old message quoted the stack and advised the stack-sized cap
            # (61 / 94 GiB at the two sizes measured), which over-states the
            # fix by ~10x: 6 / 10 GiB is what those runs actually needed.
            stack = (int(nq) * int(n_rmu) ** 2 * 16 / 1024**3
                     if nq and n_rmu else 0.0)
            batch = (_replicated_factor_q_chunk(int(nq), int(n_rmu))
                     if nq and n_rmu else 1)
            need = (batch * int(n_rmu) ** 2 * 16 / 1024**3
                    if nq and n_rmu else 0.0)
            # The exact mu ceiling this route carries, from the two 4 GiB caps:
            # batch collapses to 1 once one (mu, mu) c128 matrix exceeds
            # _REPLICATED_FACTOR_MAX_BATCH_BYTES, so the criterion reduces to
            # mu <= sqrt(max(cap, factor_cap) / 16).
            _cap = max(_REPLICATED_CHOL_MAX_STACK_BYTES,
                       _REPLICATED_FACTOR_MAX_BATCH_BYTES)
            mu_max = int(math.isqrt(_cap // 16))
            raise ValueError(
                f"charge_zeta_solve='rank_truncate' needs the replicated "
                f"factor, and the binding limit is ONE q-batch, not the "
                f"stack: batch={batch} x (n_mu={n_rmu})^2 x 16 B = "
                f"{need:.2f} GiB > the {_cap / 1024**3:.2f} GiB per-batch cap.  "
                f"(The whole CCT stack, nq={nq}, is {stack:.2f} GiB -- context "
                f"only; it is NOT what failed.)  On this route the replicated "
                f"factor is allocated one q-batch at a time, so the ceiling is "
                f"n_mu <= {mu_max}.  "
                f"Set LORRAX_ZETA_REPLICATE_CAP_GIB={-(-need // 1) + 1:.0f} to "
                f"clear it if the device budget allows -- but note the factor "
                f"is a dense whole-tile eigh per q (q-parallel over devices "
                f"above the fold threshold, so per-rank ceil(nq/P)*n_mu^3; "
                f"the ALL-RANKS execution measured 4712 s at n_mu=10015 on "
                f"64 ranks, so ~20 h at n_mu=24933 before the fold and still "
                f"hours-per-q after it): raising the "
                f"cap makes this RESOLVE, not finish.  For large n_mu use "
                f"distributed_zeta_solve='distributed' instead (ScaLAPACK "
                f"pzheevd, 236 s at the same size), or "
                f"charge_zeta_solve='cholesky' to accept the distributed "
                f"factor (NOT rank-conditioned — verify V_q).")
        return None

    return _resolve_channel_ladder(
        mesh_xy, override,
        kind_fallback='sharded_cholesky',
        kind_cusolvermp='cusolvermp_cholesky',
        explicit={'slate': _slate,
                  'cusolvermp': _cusolvermp, 'on': _cusolvermp},
        auto_pre=_auto_pre)


def _resolve_solver_kind_transverse(mesh_xy: Mesh, override: str = "auto",
                                    n_rmu_logical: int | None = None,
                                    transverse_zeta_solve: str = "ridge",
                                    ) -> str:
    """Pick the transverse-channel ζ-fit solver: cuSolverMp distributed
    getrf+getrs vs the in-tree per-q ``jnp.linalg.solve`` + ridge.

    ``transverse_zeta_solve`` (deck key, 2026-08-01) selects the SOLVE
    FAMILY first, before any backend ladder:

    * ``'ridge'`` (default) — the historical LU+ridge family below,
      byte-identical behaviour.
    * ``'rank_truncate'`` — per-q eigh pseudo-inverse of the indefinite
      transverse CCT with an |λ| cut (the charge channel's conditioning
      cure ported to the transverse channel; see
      ``_charge_factor_math``'s ``'transverse_rank_truncate'`` mode).
      Returns ``'transverse_rank_truncate'`` — the LOCAL plan (whole-tile
      replicated eigh, q-parallel at P>1, valid at ANY logical extent on
      ANY mesh).  Its DISTRIBUTED plan (pzheevd at the padded extent) is
      selected by ``distributed_zeta_solve = 'distributed'`` exactly like
      the charge channel — the ζ-fit caller overrides the kind to
      ``'distributed_transverse_rank_truncate'`` after resolving the
      tier.  ``distributed_lu`` names an LU backend this family does not
      run, so an EXPLICIT ``distributed_lu`` request combined with
      ``rank_truncate`` REFUSES here (promise contract) instead of
      silently ignoring one of the two keys.

    The rest of this docstring documents the RIDGE (LU) family.

    Default policy (2026-05-12): mirrors the charge-channel resolver —
    use cuSolverMp on **true 2D meshes** (px≥2 AND py≥2).  cuSolverMp
    0.7.2 fixes the earlier 2D-grid getrf/getrs correctness bug
    (validated end-to-end on MoS2 3×3 bispinor at 2×2 mesh; see
    ``src/ffi/cpp/cusolvermp/batched_solve_lu_ffi.cc`` for history).

    Tradeoff: small FFI setup overhead at MoS2 scale (n_rmu=656,
    2×2 mesh).  At CrI3 6×6 80 Ry (n_rmu≈1800, 4×4 mesh) the cuSolverMp
    path is the right tool.

    Override via cohsex.in ``distributed_lu``:
      ``off``        → force per-q ``jnp.linalg.solve``.
      ``cusolvermp`` → force cuSolverMp (legacy alias ``on``).  EXPLICIT
                       choice via the ffi.linalg facade: refuses at
                       resolve time on a non-CUDA mesh, a build without
                       the compiled handler, or a 1-D mesh — never a
                       silent fallback (doctrine 3; audit fix/zq
                       2026-07-28).
      ``scalapack``  → ScaLAPACK ``pXgetrf``+``pXgetrs`` from Cray LibSci
                       — the host/CPU-backend backend (liblorrax_ffi_host).
                       EXPLICIT choice, never auto-picked; fails loudly if
                       the host FFI is absent, and requires a square or
                       1-D mesh (pXgetrf needs square blocks).
      ``auto`` (default) → cuSolverMp on true 2D, legacy otherwise.
      (No ``slate`` value: a SLATE getrf wrapper does not exist yet.)

    ``n_rmu_logical`` (the LOGICAL transverse centroid count) activates
    the resolve-time divisibility contract for the two DISTRIBUTED
    backends: the indefinite solve must run at the logical μ extent
    (ROOT_CAUSE.md 2026-07-08 — pad-shape LU roundoff is amplified O(1)
    in the near-null transverse modes), and the block-cyclic descriptors
    need ``n_log % px == n_log % py == 0``.  When they don't divide:

      * EXPLICIT request (``cusolvermp``/``on``/``scalapack``) → raise
        HERE, at resolve time, naming the fix — the promise contract
        (quality pattern #6/#8; the same treatment the charge W solve
        got in the two-plan cleanup).  Before 2026-07-27 this demoted to
        the per-q replicated LU via a ``warnings.warn`` deep inside
        ``solve_zeta`` — the ledgered "silent replicated-LU fallback".
      * ``auto`` resolution → announce the demotion (rank-0 print) and
        return the per-q ``'lu'`` route.

    Callers that don't know ``n_rmu_logical`` (pass ``None``) keep the
    pure mesh/backend ladder; ``solve_zeta`` retains an announced
    call-time demotion as defense in depth for those.
    """
    fam = str(transverse_zeta_solve).strip().lower()
    if fam == 'rank_truncate':
        if override in ('on', 'cusolvermp', 'scalapack'):
            raise ValueError(
                f"transverse_zeta_solve='rank_truncate' selects the eigh "
                f"pseudo-inverse family, whose distributed plan is chosen "
                f"by distributed_zeta_solve='distributed' (pzheevd) — but "
                f"distributed_lu={override!r} explicitly requests an LU "
                f"backend the family does not run.  Leave distributed_lu "
                f"at 'auto'/'off', or set transverse_zeta_solve='ridge' "
                f"to use the LU family.")
        return 'transverse_rank_truncate'
    if fam != 'ridge':
        raise ValueError(
            f"transverse_zeta_solve={transverse_zeta_solve!r} invalid; "
            f"expected ridge / rank_truncate.")

    def _scalapack(px: int, py: int) -> str:
        # Facade guard ladder (ffi.linalg.resolve): host-only platform
        # (defense-in-depth — gw_config already rejects scalapack on
        # non-CPU backends at parse time), compiled-capability probe,
        # process coverage, and the square-or-1-D descriptor geometry.
        _resolve_linalg_backend('solve_lu', 'scalapack', mesh_xy)
        return 'scalapack_lu'

    def _cusolvermp(px: int, py: int) -> str:
        # EXPLICIT cusolvermp runs the same facade guard ladder as
        # 'scalapack': CUDA platform, compiled handler, process coverage,
        # true-2D geometry (a 1-D mesh REFUSES at resolve time instead of
        # silently returning the per-q fallback).  (audit fix/zq
        # 2026-07-28; doctrine 3 / quality pattern #6)
        _resolve_linalg_backend('solve_lu', 'cusolvermp', mesh_xy)
        return 'cusolvermp_lu'

    # auto: cuSolverMp on true 2D GPU meshes; the shared ladder's CPU-mesh
    # guard falls back to the CPU-safe in-tree per-q solve.
    kind = _resolve_channel_ladder(
        mesh_xy, override,
        kind_fallback='lu',
        kind_cusolvermp='cusolvermp_lu',
        explicit={'scalapack': _scalapack,
                  'cusolvermp': _cusolvermp, 'on': _cusolvermp})

    if kind in ('cusolvermp_lu', 'scalapack_lu') and n_rmu_logical is not None:
        px = int(mesh_xy.shape['x'])
        py = int(mesh_xy.shape['y'])
        n_log = int(n_rmu_logical)
        if (n_log % px) or (n_log % py):
            if override in ('on', 'cusolvermp', 'scalapack'):
                raise ValueError(
                    f"distributed_lu={override!r} was explicitly requested, "
                    f"but the transverse centroid count n_rmu_T={n_log} is "
                    f"not divisible by the {px}x{py} mesh axes.  The "
                    f"indefinite transverse solve must run at the LOGICAL "
                    f"extent (pad-extent LU roundoff is amplified O(1) in "
                    f"the near-null transverse modes) and the block-cyclic "
                    f"descriptors need n % px == n % py == 0.  Either pick "
                    f"a transverse centroid count divisible by both mesh "
                    f"axes, change the process mesh, set "
                    f"distributed_lu = off (per-q replicated "
                    f"jnp.linalg.solve, valid at any extent), or use "
                    f"transverse_zeta_solve = rank_truncate (its "
                    f"distributed plan runs pzheevd at the PADDED extent "
                    f"— divisible by construction — with exactly-inert "
                    f"pad modes, so any count fits any square mesh).")
            if jax.process_index() == 0:
                print(
                    f"  [solver resolve] transverse LU: auto resolved to "
                    f"{kind} but n_rmu_T={n_log} does not divide the "
                    f"{px}x{py} mesh axes (block-cyclic descriptor rule); "
                    f"demoting to the per-q replicated LU "
                    f"(distributed_lu-equivalent 'off') so the solve runs "
                    f"at the logical extent.  For a distributed transverse "
                    f"plan at ANY count use transverse_zeta_solve = "
                    f"rank_truncate + distributed_zeta_solve = distributed "
                    f"(pzheevd at the padded extent).", flush=True)
            return 'lu'
    return kind


def _resolve_solver_kind(
    mesh_xy: Mesh, vertex_mu_L: int, solver_kind: str,
    distributed_cholesky: str = "auto",
    distributed_lu: str = "auto",
    n_rmu: int | None = None,
    nq: int | None = None,
    charge_zeta_solve: str = "cholesky",
    replicated_factor_used: bool = True,
    transverse_zeta_solve: str = "ridge",
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
        return _resolve_solver_kind_transverse(
            mesh_xy, distributed_lu, n_rmu_logical=n_rmu,
            transverse_zeta_solve=transverse_zeta_solve)
    return _resolve_solver_kind_charge(
        mesh_xy, distributed_cholesky, n_rmu=n_rmu, nq=nq,
        charge_zeta_solve=charge_zeta_solve,
        replicated_factor_used=replicated_factor_used)


# Budget for the ζ back-solve's replicated-factor ALL-GATHER, i.e. the
# transient that ``_solve_all_at_once`` / ``_solve_batch_and_update`` put on
# every rank when they pull the (q_batch, μ, μ) factor to P(None,None,None).
# Deliberately SEPARATE from ``_REPLICATED_CHOL_MAX_STACK_BYTES``: that one
# gates whether the FACTORIZATION may be replicated (a physics-route
# decision, and production raises it to 16 GiB to keep rank_truncate
# reachable), this one gates only the gather GRANULARITY inside the
# back-solve, which is numerically free either way.
_ZETA_GATHER_MAX_BYTES = int(
    float(os.environ.get("LORRAX_ZETA_GATHER_CAP_GIB", "4")) * 1024 ** 3)


# ---------------------------------------------------------------------------
# DEPRECATED env overrides of input-file keys (scorecard AV; pattern #8).
#
# ``zeta_rcond`` / ``zeta_ridge`` are THE conditioning knobs of the μ ladder
# — physics policy, not machine capability — so their home is the input file,
# where they are parsed, validated, echoed into the run log and captured by
# ζ-fit provenance.  The env forms predate the keys and used to win SILENTLY
# (the env read had the key's value as its *fallback*), which is exactly the
# env-coupled-behavior failure class: a sweep run whose central parameter is
# not in its own input file.  No live harness depends on the env forms
# (audited 2026-07-27: the one historical user is the completed one-off
# run_B_c1998_rcond10/run72.sbatch).  They keep working this release, but
# the override is announced LOUDLY, once per process per variable.
_env_override_warned: set = set()


def _env_override_raw(env_name: str) -> str | None:
    """THE non-empty-env-wins rule of the deprecated env twins, in ONE
    place: the raw env string when it is set and non-blank (that value
    wins this release), else ``None`` (the input key is used).  Shared by
    the factor sites (:func:`_deprecated_env_float`) and the ζ-provenance
    record (:func:`deprecated_env_record` ←
    ``gw.gw_init._zeta_fit_provenance``) so the two can never drift
    (quality pattern #3; audit fix/zq 2026-07-28)."""
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return None
    return raw


def deprecated_env_record(env_name: str, key_value) -> str:
    """The string ζ-fit provenance records for a deprecated env-twin knob:
    the raw env string when the env form wins (the exact rule the factor
    sites apply, via :func:`_env_override_raw`), else ``repr(key_value)``.
    Byte-identical to the historical inline format in every case that
    ever produced a reusable ζ, so existing provenance stamps keep
    matching.  (audit fix/zq 2026-07-28)"""
    raw = _env_override_raw(env_name)
    return raw if raw is not None else repr(key_value)


# Canonical boolean env grammar — ONE parser, imported, not copied.
#
# This module used to carry its own ``_env_bool`` (same token set, no
# telemetry): a typo'd value (``LORRAX_ZETA_RANK_LOG=ture``) resolved
# silently to False, which for a default-ON knob silently turned the μ
# ladder's conditioning signal OFF.  ``gw.gw_config.env_bool`` has the
# identical vocabulary plus the once-per-(name,value) ``*** LORRAX
# SANITY`` announcement on unrecognised tokens.  The import direction is
# L1→L1 and safe: ``gw/__init__`` pulls only ``gw_config``, which is
# deliberately jax-free and imports nothing from ``isdf``.
# (P1.3 grammar unification, 2026-07-31; the drift gate is
# ``tests/test_env_grammar.py``, which scans this file as an OWNED file.)
from gw.gw_config import env_bool


def _deprecated_env_float(env_name: str, key_name: str, key_value) -> float:
    """Input key is the source of truth; a non-empty env var still overrides,
    but prints a deprecation notice on rank 0 (once per process).

    Empty/unset env → the key's value, exactly.  This also removes the old
    crash on ``LORRAX_ZETA_RCOND=""`` (``float('')``).
    """
    raw = _env_override_raw(env_name)
    if raw is None:
        return float(key_value)
    val = float(raw)
    if env_name not in _env_override_warned:
        _env_override_warned.add(env_name)
        if jax.process_index() == 0:
            print(f"  *** DEPRECATED env override: {env_name}={raw} overrides "
                  f"input key {key_name}={key_value!r}.  The input file is "
                  f"the record — put '{key_name} = {raw}' in the deck "
                  f"instead.  The env form still wins this release, but it "
                  f"is loud on purpose (env grants capability; it must not "
                  f"silently select policy).", flush=True)
    return val


def _resolve_zeta_gather(
    override: str = "auto",
    n_rmu: int | None = None,
    nq: int | None = None,
    *,
    mesh_xy: Mesh | None = None,
    vertex_mu_L: int = 0,
    charge_zeta_solve: str = "cholesky",
    transverse_zeta_solve: str = "ridge",
) -> str:
    """Resolve the ζ back-solve TIER — the input key
    ``distributed_zeta_solve``.

    Returns ``'replicated'``, ``'per_q'`` or ``'distributed'``.

    * ``replicated`` — today's path: the back-solve all-gathers the whole
      ``(q_batch, μ, μ)`` factor onto every rank, ``nq·μ²·16`` B per rank
      (18.9 GB at MoS2 12×12 / μ=1998 counting the logical-extent copies,
      and it is re-gathered on EVERY r-chunk).
    * ``per_q`` — gather ONE ``(μ, μ)`` tile at a time and loop q inside
      the r-chunk.  ``μ²·(1 + 1/p_y)·16`` B (75 MB at μ_pad=2048 on an 8×8
      mesh, 1.8 GB at μ=10k).  Same per-q arithmetic as the batched
      kernel; only the live gathered extent shrinks.  The slice is taken
      INSIDE a ``shard_map`` (``_per_q_block``) — written as a
      ``with_sharding_constraint`` on a traced-``q`` slice it read the
      same way but COMPILED to the full ``(nq, μ, μ)`` gather plus a
      dynamic_slice, which is worse than ``replicated`` and cost 12–40×
      the back-solve wall (scorecard Y.2; do not regress it).
    * ``distributed`` — the factor is NEVER gathered.  ``C_q`` is
      eigendecomposed distributed (ScaLAPACK ``pzheevd``), truncated on the
      replicated spectrum, and the truncated pseudo-inverse ``C⁺`` is kept
      2D-sharded; the back-solve is a stacked 2D-sharded GEMM ``C⁺ @ Z``.
      This is the ONLY tier whose eigh ITSELF divides by P — the other two
      run whole-tile dense ``eigh``s per q (q-parallel over devices above
      the replicated plan's fold threshold, so min(P, nq)-scaling since
      2026-08-01; redundant on every rank below it — ~5.5 h at μ=4k,
      ~86 h at μ=10k for the full sweep, /min(P, nq) with the fold).
      EXPLICIT opt-in only: ``auto`` never picks it, because it changes the
      arithmetic (block-cyclic eigh ⇒ a different, equally valid gauge) and
      so is not bit-identical to the other two.
    * ``auto`` (default) — ``replicated`` while the gather fits under
      :data:`_ZETA_GATHER_MAX_BYTES`, ``per_q`` above it.  At fixture scale
      (nq=9, μ_pad=64 ⇒ 0.6 MB) that is ``replicated``, i.e. bit-identical
      to the pre-feature path; at MoS2 12×12 / μ=2016 (9.4 GB) it is
      ``per_q``.

    ``distributed`` additionally REQUIRES (all checked here, at resolve
    time, so nothing fails minutes later inside an FFI call):

    * ``charge_zeta_solve = 'rank_truncate'`` — the tier IS distributed
      rank truncation, and the spectral cut is the charge channel's
      conditioning cure (ADVICE §6a); a plain distributed inverse would
      silently destroy the physics, so it is refused rather than offered;
    * a mesh the ScaLAPACK eigh backend accepts — host devices, one
      process per device, square or 1-D, ``μ_pad`` divisible by both axes
      (``ffi.linalg.resolve_backend('eigh', 'distributed', …)`` owns that
      ladder and raises with the failed guard named).

    On the TRANSVERSE channels (``vertex_mu_L != 0``) ``distributed``
    resolves to ``per_q``: the transverse CCT is Hermitian INDEFINITE, so
    no eigh-based rank truncation applies to it, and its distributed route
    is the already-2D-sharded ``pXgetrf``/``pXgetrs`` pair selected by a
    DIFFERENT key (``distributed_lu = scalapack``).  One key drives both
    channels, so raising here would kill a bispinor run in the transverse
    fit after the charge fit had succeeded.
    """
    tier = str(override or "auto").strip().lower()
    if tier == "distributed":
        if int(vertex_mu_L) != 0:
            if str(transverse_zeta_solve).strip().lower() == 'rank_truncate':
                # The transverse rank_truncate family (2026-08-01) HAS a
                # distributed plan: pzheevd at the padded extent, |λ|
                # cut, 2D-sharded C⁺ — the same guard ladder as the
                # charge tier (``n_rmu`` here is the PADDED extent, so
                # the divisibility guard always holds by construction).
                if mesh_xy is None:
                    raise ValueError(
                        "distributed_zeta_solve='distributed' needs the "
                        "device mesh to resolve its eigh backend; the "
                        "caller passed none.")
                _resolve_linalg_backend('eigh', 'distributed', mesh_xy,
                                        n=n_rmu)
                return tier
            # RIDGE family: ONE key drives both channels, so a bispinor
            # run must not die in the transverse fit after the charge fit
            # succeeded.  The transverse CCT is Hermitian INDEFINITE — no
            # eigh-based rank truncation applies to the LU family; its
            # distributed route is ``distributed_lu = scalapack``
            # (pXgetrf/pXgetrs, already 2D-sharded end to end, see
            # solve_zeta's 'scalapack_lu' branch), a different key.
            # Resolve to the tightest tier this key CAN offer here; the
            # caller's banner prints the request and the resolution side
            # by side, so it is visible, not silent.
            return "per_q"
        if str(charge_zeta_solve) != 'rank_truncate':
            raise ValueError(
                "distributed_zeta_solve='distributed' requires "
                f"charge_zeta_solve='rank_truncate'; got "
                f"{charge_zeta_solve!r}.  The tier's whole content is a "
                "DISTRIBUTED rank truncation: dropping the near-null "
                "directions is the charge channel's conditioning cure, and "
                "a plain distributed inverse without the spectral cut "
                "silently destroys the physics (ADVICE §6a).")
        if mesh_xy is None:
            raise ValueError(
                "distributed_zeta_solve='distributed' needs the device "
                "mesh to resolve its eigh backend; the caller passed none.")
        # Raises with the failed guard named (platform / compiled handler /
        # process coverage / geometry / divisibility).
        _resolve_linalg_backend('eigh', 'distributed', mesh_xy, n=n_rmu)
        return tier
    if tier in ("replicated", "per_q"):
        return tier
    if tier != "auto":
        raise ValueError(
            f"distributed_zeta_solve={override!r} invalid; expected "
            f"auto / replicated / per_q / distributed.")
    if nq is None or n_rmu is None:
        return "replicated"
    return ("replicated"
            if int(nq) * int(n_rmu) ** 2 * 16 <= _ZETA_GATHER_MAX_BYTES
            else "per_q")


_replicated_chol_cache = {}  # replicated dense Cholesky kernel (keyed by shape)


def _charge_factor_math(C_log, *, mode: str, n_log: int,
                        ridge_extra: float, rcond: float, rank_log: bool):
    """The per-q dense factor arithmetic — ONE kernel, shared bit-for-bit
    by the all-ranks (replicated) and q-parallel executions of the
    replicated plan (:func:`_factor_c_q_replicated`,
    :func:`_factor_c_q_replicated_qparallel`).

    ``C_log``: ``(nqb, n_log, n_log)`` whole LOGICAL tiles; the caller
    guarantees they are fully local / replicated per device.  Pure jnp with
    NO sharding ops, so the emitted per-q LAPACK calls are identical
    wherever it runs — the bit-identity contract of the q-parallel fold.
    ``mode`` selects the factor exactly as documented on
    :func:`_factor_c_q_replicated` (``'rank_truncate'`` | ``'cholesky'``,
    charge channel) plus ``'transverse_rank_truncate'`` (bispinor
    transverse channels, 2026-08-01): the SAME eigh rank truncation on the
    Hermitian INDEFINITE transverse CCT — the cut is on |λ| (both signs
    are physical there) and the return value is the EXPLICIT truncated
    pseudo-inverse C⁺ = Σ_{|λ|>τ·|λ|_max} vᵢvᵢᴴ/λᵢ, not a B with
    BBᴴ = C⁺ (no such Hermitian factor exists for an indefinite C⁺;
    explicit C⁺ also halves the per-r-chunk back-solve to ONE matmul —
    the same trade the distributed charge tier documents).
    """
    if mode == 'transverse_rank_truncate':
        # WHY THIS FEATURE EXISTS (mirror of the charge cure below, for
        # the indefinite transverse CCT): TRS in non-magnetic ground
        # states gives near-null transverse-current modes; the LU+ridge
        # family inverts THROUGH them (lifted only to the 1e-12·tr/n
        # ridge floor, κ~1e12), so ULP/mesh roundoff on those modes is
        # amplified O(1) into ζ_T.  Rank truncation DROPS |λ| <
        # τ·|λ|_max instead — κ_eff ≤ 1/τ by construction — and is the
        # basis-adequacy instrument for the transverse set (n_keep/q).
        lam, V = jnp.linalg.eigh(C_log)      # Hermitian INDEFINITE
        sig = jnp.abs(lam)
        sig_max = jnp.max(sig, axis=-1, keepdims=True)
        keep = sig > (rcond * sig_max)
        inv = jnp.where(keep, 1.0 / jnp.where(keep, lam, 1.0), 0.0)
        if rank_log:
            # Same conditioning signal as the charge route: n_keep/q is
            # the measured transverse basis adequacy; σ_min(kept)/σ_max
            # bound the achieved amplification κ_eff ≤ 1/τ.
            sig_keep_min = jnp.min(
                jnp.where(keep, sig, jnp.inf), axis=-1)
            n_keep = jnp.sum(keep, axis=-1)
            sig_drop_hi = jnp.max(
                jnp.where(keep, -jnp.inf, sig), axis=-1)
            jax.debug.print(
                "[zeta transverse rank_truncate] n_log={n} rcond={rc:.1e} "
                "n_keep/q={k} sig_max/q={mx} sig_min_kept/q={mn} "
                "kappa/q={kp} sdrop_hi/q={dh}",
                n=n_log, rc=rcond, k=n_keep,
                mx=sig_max[..., 0], mn=sig_keep_min,
                kp=sig_max[..., 0] / sig_keep_min,
                dh=sig_drop_hi,
                ordered=False)
        Vs = V * inv[..., None, :].astype(V.dtype)
        return Vs @ jnp.conj(jnp.swapaxes(V, -1, -2))
    if mode == 'rank_truncate':
        # WHY THIS FEATURE EXISTS: the charge CCT near-singularizes when
        # n_μ over-completes the pair-density rank (κ~1e13); plain
        # Cholesky then amplifies ULP/mesh/nband roundoff into O(1) V_q
        # errors that GN-PPM magnifies to tens of eV.  Rank-truncation
        # DROPS eigenvalues < zeta_rcond·λ_max (the near-null
        # directions) → a conditioned, mesh-invariant ζ = C⁺Z.
        lam, V = jnp.linalg.eigh(C_log)      # Hermitian-SPD, λ ascending
        lam_max = lam[..., -1:]              # (nqb,1) largest λ per q
        keep = lam > (rcond * lam_max)       # near-null cut
        # B = V·diag(1/√λ_kept) ⇒ B Bᴴ = Σ_{keep} vᵢvᵢᴴ/λᵢ = C⁺.
        # Double-``where`` keeps rsqrt off the dropped (tiny/≤0) modes.
        inv_sqrt = jnp.where(
            keep, jax.lax.rsqrt(jnp.where(keep, lam, 1.0)), 0.0)
        # OBSERVABILITY: the retained-mode count IS the conditioning
        # signal for this route — it is what tells you whether n_μ has
        # over-completed the pair-density rank (κ blow-up) and by how
        # much.  It lives inside the jit, so print it from there.
        # ``n_keep`` per q + the spectral span λ_max/λ_min(kept).
        # Silence with LORRAX_ZETA_RANK_LOG=0.
        #
        # THE CRITERION, stated: ``keep`` above is NOT a search for a
        # gap in λ — a real ISDF charge spectrum is smooth and has
        # none.  It is a CAP on how much C⁺ may amplify round-off:
        # κ_eff = λ_max/λ_min(kept) ≤ 1/zeta_rcond by construction.
        # ``common/rank_criterion`` carries the derivation, the three
        # standard alternatives (discrepancy principle / L-curve /
        # GCV) and the measurement that refutes each of them here.
        #
        # The three extra fields below are the ones a run needs in
        # order to be auditable without a sweep:
        #   kappa/q     achieved amplification — the invariant
        #   ldrop_hi/q  the LARGEST discarded λ, i.e. the top of the
        #               discarded band (paired with lam_min_kept it
        #               gives the whole cut, and shows there is no
        #               plateau at the cut — there never is)
        #   margin/q    fractional rank inflation from loosening
        #               rcond by 1e-4.  §R19 measured +41 % of rank
        #               costing 5000 eV, so a LARGE margin means the
        #               basis is over-complete and rcond must NOT be
        #               loosened on this run.
        if rank_log:
            lam_keep_min = jnp.min(
                jnp.where(keep, lam, jnp.inf), axis=-1)
            n_keep = jnp.sum(keep, axis=-1)
            lam_drop_hi = jnp.max(
                jnp.where(keep, -jnp.inf, lam), axis=-1)
            n_loose = jnp.sum(lam > (rcond * 1e-4 * lam_max), axis=-1)
            margin = (n_loose - n_keep) / jnp.maximum(n_keep, 1)
            jax.debug.print(
                "[zeta rank_truncate] n_log={n} rcond={rc:.1e} "
                "n_keep/q={k} lam_max/q={mx} lam_min_kept/q={mn} "
                "kappa/q={kp} ldrop_hi/q={dh} lam_min/q={lo} "
                "margin/q={mg}",
                n=n_log, rc=rcond,
                k=n_keep,
                mx=lam_max[..., 0], mn=lam_keep_min,
                kp=lam_max[..., 0] / lam_keep_min,
                dh=lam_drop_hi, lo=jnp.min(lam, axis=-1),
                mg=margin,
                ordered=False)
        return V * inv_sqrt[..., None, :].astype(V.dtype)
    tr = jnp.abs(jnp.trace(C_log, axis1=-2, axis2=-1))
    # Floor (1e-14·|tr|, bit-identical to the historical path)
    # + opt-in conditioning term (ε·|tr|/n).  Per-q scalars.
    ridge_scalar = (1e-14 * tr + ridge_extra * tr / n_log)[:, None, None]
    ridge = ridge_scalar * jnp.eye(n_log, dtype=C_log.dtype)[None, :, :]
    return jnp.linalg.cholesky(C_log + ridge)


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
      Tune ε via the ``zeta_ridge`` input key in the deck; the
      ``LORRAX_ZETA_RIDGE`` env form is a DEPRECATED twin (scorecard AV:
      still wins when set non-empty, but loudly — see
      :func:`_deprecated_env_float`) slated for removal.

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
    mode = str(charge_zeta_solve)
    # DEPRECATED env forms — the input keys are the record (scorecard AV).
    # The env twins are CHARGE-channel keys; the transverse tau
    # (transverse_zeta_rcond) deliberately has no env twin, so the charge
    # override must not bleed into the transverse mode.
    ridge_extra = _deprecated_env_float(
        "LORRAX_ZETA_RIDGE", "zeta_ridge", zeta_ridge)
    rcond = (float(zeta_rcond) if mode == 'transverse_rank_truncate'
             else _deprecated_env_float(
                 "LORRAX_ZETA_RCOND", "zeta_rcond", zeta_rcond))
    out_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    rep_sh = NamedSharding(mesh_xy, P())
    key = (id(mesh_xy), int(nq), int(n_rmu), n_log,
           float(ridge_extra), mode, float(rcond))
    if key not in _replicated_chol_cache:
        _re = ridge_extra
        _rc = rcond
        _rank_log = (mode in ('rank_truncate', 'transverse_rank_truncate')
                     and env_bool("LORRAX_ZETA_RANK_LOG", True))
        @partial(jax.jit, out_shardings=out_sh)
        def _fn(C):
            def _factor_log(C_log):
                # Replicate the logical block so every device factors the
                # WHOLE matrix — this is what makes the factor grid-agnostic
                # (the distributed potrf's block-cyclic accumulation is not).
                # The arithmetic itself lives in ``_charge_factor_math`` —
                # ONE traced kernel shared with the q-parallel execution so
                # the two schedules cannot drift (bit-identity contract).
                C_log = jax.lax.with_sharding_constraint(C_log, rep_sh)
                F = _charge_factor_math(
                    C_log, mode=mode, n_log=n_log, ridge_extra=_re,
                    rcond=_rc, rank_log=_rank_log)
                if mode == 'transverse_rank_truncate':
                    # This mode's factor ENDS in a GEMM (C⁺ = Vs Vᴴ).
                    # LAPACK factor outputs (cholesky/eigh) plus
                    # elementwise tails are replicated-identical under
                    # SPMD, but an unconstrained GEMM feeding the jit's
                    # P(None,'x','y') out_shardings gets partitioned —
                    # each device then runs a SHARD-shaped dot micro-
                    # kernel whose fma grouping differs from the whole-
                    # tile one, breaking the plan's mesh-invariance
                    # contract (caught by the unit gate, job 7885328:
                    # q-parallel legs exact, all-ranks multi-device legs
                    # drifted).  Pin the product replicated so every
                    # device computes the WHOLE tile — bit-identical to
                    # 1x1 and to the q-parallel whole-tile GEMM; the
                    # out_shardings shard afterwards is a local slice.
                    F = jax.lax.with_sharding_constraint(F, rep_sh)
                return F

            F_log = solve_at_logical(
                _factor_log, n_log, (C,), pad_axes=(-2, -1))
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

    P>1 SCHEDULE (2026-08-01): above :data:`_QPARALLEL_MIN_NQ_MU3` the
    same plan EXECUTES q-parallel (:func:`_factor_c_q_replicated_qparallel`
    — q's scattered over all devices, whole tiles per q, bits unchanged)
    instead of redundantly on every rank.  This is a fold INTO the
    replicated plan, deliberately not a third resolution — see the WHY on
    the q-parallel function.
    """
    nq, n_rmu, _ = C_q.shape
    if _qparallel_factor_ok(nq, int(n_rmu_logical), mesh_xy):
        _qparallel_announce(nq, n_rmu, int(n_rmu_logical), mesh_xy)
        return _factor_c_q_replicated_qparallel(
            C_q, mesh_xy, n_rmu_logical, **kw)
    step = _replicated_factor_q_chunk(nq, n_rmu)
    if step >= nq:
        return _factor_c_q_replicated(C_q, mesh_xy, n_rmu_logical, **kw)
    parts = [_factor_c_q_replicated(C_q[q0:min(q0 + step, nq)], mesh_xy,
                                    n_rmu_logical, **kw)
             for q0 in range(0, nq, step)]
    return jax.device_put(jnp.concatenate(parts, axis=0),
                          NamedSharding(mesh_xy, P(None, 'x', 'y')))


# ---------------------------------------------------------------------------
# q-PARALLEL EXECUTION of the replicated charge factor — a schedule, NOT a
# third plan.
#
# The replicated plan's contract is its OUTPUT: whole-tile dense per-q
# factors, bit-identical across process grids and device counts.  Nothing in
# that contract says every rank must COMPUTE every q — only that every q is
# factored as ONE dense whole-tile call.  Above the fold threshold the plan
# therefore scatters the q axis over all devices (the same q-parallel idiom
# as the W solve's LOCAL plan, ``gw/w_isdf._get_w_solve_fn_local`` /
# scorecard AN), each device factors its owned q's through the SAME traced
# kernel (``_charge_factor_math``), and the factors are resharded back to
# ``P(None, 'x', 'y')``.  Only data movement differs; the values are the
# same bits, so mesh-invariance survives by construction and the ζ-fit
# factor family keeps exactly TWO plans:
#     replicated  (mesh-invariant whole-tile factor; q-parallel at P>1)
#     distributed (2-D ScaLAPACK eigh — different gauge, explicit opt-in)
#
# MEASURED motivation (job 7884656, MoS2 4x4 b300, P=16 / 4x4 mesh,
# nq_ibz=10, mu_log=2979): zeta_fit.cholesky = 105.1 s — one dense eigh per
# q on EVERY rank — the dominant term of the 64%-of-GW-wall ζ-fit stage.
# q-parallel caps the per-rank factor count at ceil(nq/P).
# ---------------------------------------------------------------------------

# Fold threshold, in nq·μ_log³ units (the eigh work the all-ranks execution
# repeats on every rank).  Calibrated from job 7884656: 105.1 s at
# nq·μ³ = 10·2979³ ≈ 2.6e11 on a 28-thread CLX rank → ~4e-10 s/unit, so
# 5e9 ≈ 2 s of redundant per-rank factor work.  Below that, the fold's own
# costs (two staged all-to-all reshards of the (nq, μ, μ) stack — ~stack/P
# per rank each way — plus one extra jit compile) outweigh the saving: the
# fastloop mini-deck (nq=4, μ≈400 → ~2.6e8 ≈ 0.1 s) stays on the pure
# replicated execution, while every production fit from the b300 deck up
# folds.  A module CONSTANT on purpose — this is machine capability, not
# physics policy, and the AV audit retired policy env twins;
# LORRAX_ZETA_QPARALLEL (below) is the schedule escape hatch / test hook.
_QPARALLEL_MIN_NQ_MU3 = 5.0e9

_qparallel_factor_cache: dict = {}
_qparallel_announced: set = set()


def _qparallel_factor_ok(nq: int, n_rmu_logical: int, mesh_xy: Mesh) -> bool:
    """True when the replicated charge factor should EXECUTE q-parallel.

    ``LORRAX_ZETA_QPARALLEL``: unset/``auto`` → fold above
    :data:`_QPARALLEL_MIN_NQ_MU3` (needs >1 device and >1 q to scatter);
    ``0`` → never (the pre-fold all-ranks execution, kept as the A/B
    control); ``1`` → always (the bit-identity gate forces it at fixture
    size).  Either way the RESULT is the same bits — this knob selects an
    execution schedule, never a numerical route.
    """
    if int(mesh_xy.devices.size) <= 1:
        return False
    raw = os.environ.get("LORRAX_ZETA_QPARALLEL", "auto")
    raw = raw.strip().lower() if raw else "auto"
    if raw in ("", "auto"):
        return (int(nq) >= 2
                and float(nq) * float(n_rmu_logical) ** 3
                >= _QPARALLEL_MIN_NQ_MU3)
    return env_bool("LORRAX_ZETA_QPARALLEL", False)


def _qparallel_announce(nq: int, n_rmu: int, n_log: int,
                        mesh_xy: Mesh) -> None:
    """One line naming the schedule the factor actually runs (rank 0,
    deduplicated) — a fold that silently stopped engaging would otherwise
    be invisible until the 105-s stage reappeared."""
    if jax.process_index() != 0:
        return
    ndev = int(mesh_xy.devices.size)
    sig = (id(mesh_xy), int(nq), int(n_rmu), int(n_log))
    if sig in _qparallel_announced:
        return
    _qparallel_announced.add(sig)
    blk = -(-int(nq) // ndev)
    print(f"  [zeta factor] replicated plan, q-parallel execution: "
          f"nq={nq} scattered over {ndev} devices "
          f"(ceil(nq/P)={blk} whole ({n_rmu},{n_rmu}) tile(s)/device, "
          f"q-pad {round_up(int(nq), ndev) - int(nq)}); factors are "
          f"bit-identical to the all-ranks execution "
          f"(LORRAX_ZETA_QPARALLEL=0 restores it)", flush=True)
    if ndev > int(nq):
        # The fold SATURATES at P = nq: q is its only parallel axis, so
        # every rank past nq idles for the whole factor stage (measured:
        # 54/64 ranks idle for 53.7 s = 22.4% of GW wall at nq=10,
        # b600/P=64, job 7885316).  Auto stays on this plan — the
        # distributed tier is a different (equally valid) gauge and auto
        # never silently crosses that line — but the operator gets the
        # measured crossover: at P/nq = 6.4 the distributed tier ran the
        # factor 1.64x faster and GW wall 0.83x (job 7885323).
        print(f"  [zeta factor] NOTE: the q-parallel fold saturates at "
              f"P = nq — {ndev - int(nq)} of {ndev} ranks idle for this "
              f"stage (1 q/rank ceiling).  At this P/nq "
              f"({ndev / int(nq):.1f}) consider "
              f"distributed_zeta_solve = distributed (pzheevd, whole-mesh "
              f"P-scaling; measured factor 1.64x faster / GW wall 0.83x "
              f"at P/nq=6.4, jobs 7885316/7885323 — NOTE: a different "
              f"gauge, ~kappa*eps vs this plan, not bit-identical).",
              flush=True)


def _factor_c_q_replicated_qparallel(
    C_q: jax.Array, mesh_xy: Mesh, n_rmu_logical: int,
    zeta_ridge: float = 0.0,
    charge_zeta_solve: str = 'cholesky',
    zeta_rcond: float = 1e-8,
) -> jax.Array:
    """The replicated charge factor, EXECUTED q-parallel.

    WHY THIS IS A FOLD AND NOT A THIRD RESOLUTION: a plan in this family
    is a numerical contract — ``replicated`` = whole-tile dense factor,
    bit-identical across meshes and device counts; ``distributed`` =
    block-cyclic eigh, a different (equally valid) gauge, explicit opt-in.
    This path changes only WHICH device runs each per-q factorisation,
    never what is computed, so its output is the replicated plan's output
    to the bit and it carries no new resolver string, no new input key,
    and no new downstream contract.  (Precedent: the W-solve family's
    LOCAL plan is likewise q-parallel — scorecard AN.)

    Schedule: zero-pad the q axis to the device count, scatter q over the
    FLATTENED mesh (``P(('x','y'), None, None)``) through the measured
    single-axis staging (``P('x', None, 'y')`` — see gw/w_isdf's
    involuntary-remat note), factor each OWNED q as one whole-tile call
    into :func:`_charge_factor_math` (per-q ``fori_loop``: the XLA eigh
    workspace is bounded by ONE (μ, μ) tile, strictly tighter than
    :func:`_replicated_factor_q_chunk`'s batch bound), skip pad q's with a
    ``lax.cond`` (so the Cholesky branch never factors filler and the
    rank log prints no phantom q's), then stage the factors back to
    ``P(None, 'x', 'y')`` and re-embed the identity μ-pad block.

    BIT-IDENTITY to the all-ranks execution, claim by claim:

    * the factor is per-q independent — the q-batch split is already
      relied on (``factor_c_q_replicated_batched`` concatenates cap-sized
      batches) and XLA's batched LAPACK wrappers loop per matrix;
    * the reshards move exact byte copies (pure data movement);
    * the per-q arithmetic is the SAME traced kernel on the same whole
      logical tile (``_charge_factor_math``; the μ-slice/zero-refill is
      the same ``solve_at_logical``; the identity μ-pad re-embed is the
      same helper).

    Gate: ``tests/test_zeta_mesh_invariance.py::
    test_qparallel_execution_is_bit_identical_to_replicated`` (exact
    equality, both modes, non-dividing nq, padded μ).

    Observability delta, deliberate: the rank_truncate conditioning log
    prints per OWNED q from the owning process (the all-ranks execution
    printed every q from every process); fields are unchanged.
    """
    nq, n_rmu, _ = C_q.shape
    n_log = int(n_rmu_logical)
    mode = str(charge_zeta_solve)
    # DEPRECATED env forms — charge-channel only; the transverse tau has
    # no env twin (see _factor_c_q_replicated).
    ridge_extra = _deprecated_env_float(
        "LORRAX_ZETA_RIDGE", "zeta_ridge", zeta_ridge)
    rcond = (float(zeta_rcond) if mode == 'transverse_rank_truncate'
             else _deprecated_env_float(
                 "LORRAX_ZETA_RCOND", "zeta_rcond", zeta_rcond))
    rank_log = (mode in ('rank_truncate', 'transverse_rank_truncate')
                and env_bool("LORRAX_ZETA_RANK_LOG", True))
    ndev = int(mesh_xy.devices.size)
    py = int(mesh_xy.shape['y'])
    nq_pad = round_up(int(nq), ndev)
    out_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    q_sh = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    mid_sh = NamedSharding(mesh_xy, P('x', None, 'y'))

    key = (id(mesh_xy), int(nq), int(n_rmu), n_log,
           float(ridge_extra), mode, float(rcond), bool(rank_log))
    if key not in _qparallel_factor_cache:
        _re, _rc, _rl = ridge_extra, rcond, rank_log
        blk = nq_pad // ndev

        def _local_factor(C_loc):
            # C_loc: (blk, n_rmu, n_rmu) — whole tiles for the q's this
            # device owns (global q of local slot i = dev·blk + i, in the
            # ('x','y') row-major order the flattened q-shard uses).
            dev = jax.lax.axis_index('x') * py + jax.lax.axis_index('y')

            def _fact(C1):
                return solve_at_logical(
                    lambda Cl: _charge_factor_math(
                        Cl, mode=mode, n_log=n_log, ridge_extra=_re,
                        rcond=_rc, rank_log=_rl),
                    n_log, (C1,), pad_axes=(-2, -1))

            def _one(i, F_acc):
                C1 = jax.lax.dynamic_slice_in_dim(C_loc, i, 1, axis=0)
                F1 = jax.lax.cond(dev * blk + i < nq, _fact,
                                  jnp.zeros_like, C1)
                return jax.lax.dynamic_update_slice(F_acc, F1, (i, 0, 0))

            return jax.lax.fori_loop(0, blk, _one, jnp.zeros_like(C_loc))

        _sm = shard_map(_local_factor, mesh=mesh_xy,
                        in_specs=P(('x', 'y'), None, None),
                        out_specs=P(('x', 'y'), None, None),
                        check_rep=False)

        @partial(jax.jit, out_shardings=out_sh)
        def _fn(C):
            if nq_pad > nq:
                C = jnp.pad(C, ((0, nq_pad - nq), (0, 0), (0, 0)))
            # Single-axis staging both ways: the composite reshard makes
            # SPMD replicate-then-partition (the w_isdf reshard_mid
            # measurement); each stage moves ONE mesh axis.
            C = jax.lax.with_sharding_constraint(C, mid_sh)
            C = jax.lax.with_sharding_constraint(C, q_sh)
            F = _sm(C)
            F = F[:nq]
            F = jax.lax.with_sharding_constraint(F, mid_sh)
            # Pad-block factor = identity — same re-embed (and same final
            # P(None,'x','y') constraint) as the all-ranks execution;
            # no-op when n_log == n_rmu.
            return _identity_pad_block_diagonal(
                F, n_rmu_logical=n_log, mesh_xy=mesh_xy)

        _qparallel_factor_cache[key] = _fn
    return _qparallel_factor_cache[key](C_q)


# =============================================================================
# The hoisted TRANSVERSE factor stage (bispinor mu_L = 1, 2, 3)
# =============================================================================
#
# The transverse CCT is Hermitian INDEFINITE — no Cholesky, no eigh-based
# rank truncation.  Historically factor_c_q passed the (identity-padded)
# CCT through unfactored and solve_zeta re-ran the pivoted LU on EVERY
# r-chunk (on every rank on the local path; on the mesh but still per
# r-chunk under distributed_lu=scalapack): nq·mu_T³·n_rchunks redundant
# work, and the q-parallel charge fold could not apply because there was
# no factor stage to schedule.  The functions below hoist the factor so
# the transverse channels have the SAME two plans as the charge family:
#
# * LOCAL plan ('lu') — per-q pivoted LU on the whole ridged LOGICAL
#   tile, computed ONCE per channel (q-parallel over devices at P>1 under
#   the charge fold's policy), stored as (LU, perm) with the LU factors
#   identity-re-embedded at the padded extent so every downstream gather
#   tier (replicated / per_q) consumes them exactly like the CCT it
#   replaced.  BIT-IDENTICAL to the fused per-r-chunk solve:
#   jnp.linalg.solve(A, b) IS lax.linalg.lu(A) followed by
#   lax.linalg.lu_solve(lu, perm, b, 0) (jax _solve), and this stage runs
#   exactly those two ops with the factor cached between r-chunks.
#   Gate: tests/test_transverse_factor_hoist.py (exact equality).
# * DISTRIBUTED plan ('scalapack_lu', host mesh) — per-q ScaLAPACK
#   pXgetrf run ONCE per channel at the LOGICAL extent, factors kept 2-D
#   block-cyclic, per-rank ipiv threaded alongside; solve_zeta calls
#   pXgetrs per r-chunk.  getrf on the ridged logical tile is
#   bit-identical whether or not the getrs follows immediately (same
#   descriptors, same grid — the fused handler runs the same two calls
#   back to back), so this differs from the fused path only in WHEN the
#   factor work happens.
#
# 'cusolvermp_lu' (CUDA mesh) keeps the FUSED per-r-chunk getrf+getrs
# path for now: splitting its FFI handler is mechanical but cannot be
# validated on the Frontera CPU stage — factor_c_q returns the CCT
# passthrough with piv=None and solve_zeta dispatches as before.

# The per-q diagonal ridge for the indefinite transverse LU:
# eps·|tr(C_log)|/n_log lifts TRS-paired near-zero modes above the
# partial-pivoting stability floor without perturbing well-conditioned
# ones.  Module constant so the hoisted factor stage and solve_zeta's
# fused fallback paths cannot drift.
_TRANSVERSE_LU_RIDGE = 1e-12

_transverse_lu_cache: dict = {}      # hoisted local LU factor kernels
_transverse_scalapack_cache: dict = {}  # hoisted distributed getrf kernels


def _transverse_lu_math(C_log: jax.Array, n_log: int):
    """Per-q hoisted transverse LU arithmetic — ONE kernel shared by the
    all-ranks and q-parallel executions (bit-identity contract, same role
    as ``_charge_factor_math``).

    ``C_log``: one whole REPLICATED logical tile ``(n_log, n_log)``.
    Returns ``(lu, piv)`` with ``lu`` the packed L/U factors and ``piv``
    the int32 LAPACK pivots — the pair ``jnp.linalg.solve`` computes
    internally (``lax.linalg.lu``), so ``jax.scipy.linalg.lu_solve((lu,
    piv), Z)`` at solve time runs the identical
    ``lu_pivots_to_permutation`` + ``lax_linalg.lu_solve`` arithmetic and
    reproduces the fused ``jnp.linalg.solve(C_reg, Z)`` to the bit.  The
    ridge uses ``jnp.trace`` on the replicated tile — the same
    expression (same reduction order, same bits) the fused
    ``_ridge_indef_solve`` used.
    """
    ridge = _TRANSVERSE_LU_RIDGE * jnp.abs(jnp.trace(C_log)) / n_log
    C_reg = C_log + ridge * jnp.eye(n_log, dtype=C_log.dtype)
    lu, piv, _perm = jax.lax.linalg.lu(C_reg)
    return lu, piv.astype(jnp.int32)


def _embed_lu_padded(LU_log: jax.Array, n_rmu: int, n_log: int,
                     mesh_xy: Mesh) -> jax.Array:
    """Zero-embed per-q LOGICAL LU factors at the padded extent and set
    identity on the pad-block diagonal (shape/sharding uniformity only:
    the back-solve slices back to the logical block, so the pad content
    is never part of any solve — same contract as the charge factor's
    identity pad)."""
    if int(n_rmu) == int(n_log):
        return jax.lax.with_sharding_constraint(
            LU_log, NamedSharding(mesh_xy, P(None, 'x', 'y')))
    pad = int(n_rmu) - int(n_log)
    LU_pad = jnp.pad(LU_log, ((0, 0), (0, pad), (0, pad)))
    return _identity_pad_block_diagonal(
        LU_pad, n_rmu_logical=int(n_log), mesh_xy=mesh_xy)


def _factor_c_q_transverse_lu(
    C_q: jax.Array, mesh_xy: Mesh, n_rmu_logical: int,
) -> tuple[jax.Array, jax.Array]:
    """LOCAL-plan hoisted transverse factor: per-q pivoted LU of the
    ridged LOGICAL block, once per channel.

    Returns ``(LU_q, perm_q)``:

    * ``LU_q`` ``(nq, n_rmu, n_rmu)`` at PADDED extent, sharded
      ``P(None, 'x', 'y')`` — the packed L/U factors in the logical
      block, identity in the pad block.  Downstream gather tiers
      (replicated / per_q) consume it exactly like the CCT passthrough
      they used to gather: same shape, same sharding, same bytes moved.
    * ``perm_q`` ``(nq, n_log)`` int32, replicated — the LU permutation
      for ``lax.linalg.lu_solve``.

    Execution schedule mirrors the charge fold
    (:func:`_factor_c_q_replicated_qparallel`): q-parallel over the
    flattened mesh when :func:`_qparallel_factor_ok` says so (the factor
    is per-q independent; scatter/gather reshards are exact byte moves;
    the per-q arithmetic is the ONE shared kernel
    :func:`_transverse_lu_math`), all-ranks whole-tile execution
    otherwise.  Both produce the same bits.
    """
    nq, n_rmu, _ = C_q.shape
    n_log = int(n_rmu_logical)
    qparallel = _qparallel_factor_ok(nq, n_log, mesh_xy)
    out_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    if not qparallel:
        # All-ranks execution: replicate whole logical tiles per q-batch
        # (bounded exactly like the charge factor's batched path) and run
        # the shared kernel vmapped.  Every rank factors every q —
        # affordable because it now happens ONCE per channel, not per
        # r-chunk; per-q independent, so concatenating batches reproduces
        # the one-shot call.
        step = _replicated_factor_q_chunk(nq, n_log)
        key = (id(mesh_xy), int(n_rmu), n_log, 'batch')
        if key not in _transverse_lu_cache:
            @partial(jax.jit,
                     out_shardings=(out_sh, NamedSharding(mesh_xy, P(None, None))))
            def _fn(C):
                def _fact_log(C_log):
                    C_log = jax.lax.with_sharding_constraint(
                        C_log, NamedSharding(mesh_xy, P(None, None, None)))
                    return jax.vmap(
                        lambda C1: _transverse_lu_math(C1, n_log))(C_log)
                # Slice to logical, factor, re-embed (LU only; perm is
                # logical-extent by definition).
                LU_log, perm = _fact_log(C[:, :n_log, :n_log])
                return _embed_lu_padded(LU_log, int(n_rmu), n_log,
                                        mesh_xy), perm
            _transverse_lu_cache[key] = _fn
        _fn = _transverse_lu_cache[key]
        if step >= nq:
            return _fn(C_q)
        parts = [_fn(C_q[q0:min(q0 + step, nq)])
                 for q0 in range(0, nq, step)]
        LU_q = jax.device_put(
            jnp.concatenate([p[0] for p in parts], axis=0), out_sh)
        perm_q = jax.device_put(
            jnp.concatenate([p[1] for p in parts], axis=0),
            NamedSharding(mesh_xy, P(None, None)))
        return LU_q, perm_q
    key = (id(mesh_xy), int(nq), int(n_rmu), n_log, 'qpar')
    if key not in _transverse_lu_cache:
        _qparallel_announce_transverse(nq, n_rmu, n_log, mesh_xy)
        ndev = int(mesh_xy.devices.size)
        py = int(mesh_xy.shape['y'])
        nq_pad = round_up(int(nq), ndev)
        blk = nq_pad // ndev
        q_sh = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
        mid_sh = NamedSharding(mesh_xy, P('x', None, 'y'))

        def _local_factor(C_loc):
            # C_loc: (blk, n_log, n_log) whole logical tiles for the
            # q's this device owns (global q = dev·blk + i).
            dev = jax.lax.axis_index('x') * py + jax.lax.axis_index('y')

            def _fact(C1):
                lu1, perm1 = _transverse_lu_math(C1[0], n_log)
                return lu1[None], perm1[None]

            def _skip(C1):
                return (jnp.zeros_like(C1),
                        jnp.zeros((1, n_log), dtype=jnp.int32))

            def _one(i, accs):
                LU_acc, perm_acc = accs
                C1 = jax.lax.dynamic_slice_in_dim(C_loc, i, 1, axis=0)
                LU1, perm1 = jax.lax.cond(
                    dev * blk + i < nq, _fact, _skip, C1)
                return (jax.lax.dynamic_update_slice(
                            LU_acc, LU1, (i, 0, 0)),
                        jax.lax.dynamic_update_slice(
                            perm_acc, perm1, (i, 0)))

            return jax.lax.fori_loop(
                0, blk, _one,
                (jnp.zeros_like(C_loc),
                 jnp.zeros((blk, n_log), dtype=jnp.int32)))

        _sm = shard_map(_local_factor, mesh=mesh_xy,
                        in_specs=P(('x', 'y'), None, None),
                        out_specs=(P(('x', 'y'), None, None),
                                   P(('x', 'y'), None)),
                        check_rep=False)

        @partial(jax.jit,
                 out_shardings=(out_sh, NamedSharding(mesh_xy, P(None, None))))
        def _fn(C):
            C = C[:, :n_log, :n_log]
            if nq_pad > nq:
                C = jnp.pad(C, ((0, nq_pad - nq), (0, 0), (0, 0)))
            # Single-axis staging both ways (see the charge fold).
            C = jax.lax.with_sharding_constraint(C, mid_sh)
            C = jax.lax.with_sharding_constraint(C, q_sh)
            LU_log, perm = _sm(C)
            LU_log = LU_log[:nq]
            perm = perm[:nq]
            LU_log = jax.lax.with_sharding_constraint(LU_log, mid_sh)
            return _embed_lu_padded(LU_log, int(n_rmu), n_log,
                                    mesh_xy), perm
        _transverse_lu_cache[key] = _fn
    return _transverse_lu_cache[key](C_q)


_transverse_qparallel_announced: set = set()


def _qparallel_announce_transverse(nq: int, n_rmu: int, n_log: int,
                                   mesh_xy: Mesh) -> None:
    """Rank-0, deduplicated: name the schedule the hoisted transverse
    factor runs (mirror of ``_qparallel_announce``)."""
    if jax.process_index() != 0:
        return
    ndev = int(mesh_xy.devices.size)
    sig = ('T', id(mesh_xy), int(nq), int(n_rmu), int(n_log))
    if sig in _transverse_qparallel_announced:
        return
    _transverse_qparallel_announced.add(sig)
    blk = -(-int(nq) // ndev)
    print(f"  [zeta transverse factor] hoisted per-q LU, q-parallel "
          f"execution: nq={nq} scattered over {ndev} devices "
          f"(ceil(nq/P)={blk} whole ({n_log},{n_log}) tile(s)/device); "
          f"factors are bit-identical to the all-ranks execution "
          f"(LORRAX_ZETA_QPARALLEL=0 restores it)", flush=True)
    if ndev > int(nq):
        # Same P = nq saturation ceiling as the charge fold (q is the
        # only parallel axis; the b600/P=64 measurement, job 7885316,
        # applies shape-for-shape).  The distributed transverse routes:
        # ridge family -> distributed_lu = scalapack (needs mu_T
        # divisible by both mesh axes); rank_truncate family ->
        # distributed_zeta_solve = distributed (pzheevd at the padded
        # extent, any count).  Both are different gauges; auto never
        # promotes.
        print(f"  [zeta transverse factor] NOTE: the q-parallel fold "
              f"saturates at P = nq — {ndev - int(nq)} of {ndev} ranks "
              f"idle for this stage (1 q/rank ceiling; same shape as the "
              f"charge fold's measured ceiling, job 7885316).  "
              f"Distributed transverse plans: distributed_lu = scalapack "
              f"(ridge family) or transverse_zeta_solve = rank_truncate "
              f"+ distributed_zeta_solve = distributed (any centroid "
              f"count); both are a different gauge, not bit-identical.",
              flush=True)


def _factor_c_q_transverse_scalapack(
    C_q: jax.Array, mesh_xy: Mesh, n_rmu_logical: int,
) -> tuple[jax.Array, jax.Array]:
    """DISTRIBUTED-plan hoisted transverse factor: per-q ScaLAPACK
    ``pXgetrf`` on the ridged LOGICAL block, once per channel, factors
    kept 2-D block-cyclic.

    Returns ``(LU_q, ipiv_q)``:

    * ``LU_q`` ``(nq, n_log, n_log)`` at the LOGICAL extent (the resolve
      contract guarantees ``n_log % px == n_log % py == 0`` on this
      path), sharded ``P(None, 'x', 'y')`` — each rank's shard IS its
      block-cyclic local block (one-tile-per-rank layout).
    * ``ipiv_q`` ``(nq, P·ipiv_len)`` int32 sharded
      ``P(None, ('x','y'))`` — each rank's own ScaLAPACK ipiv rows,
      threaded back verbatim into ``pXgetrs`` per r-chunk.  Never
      gathered, never interpreted host-side.

    The ridge uses the SAME einsum expression over the sharded logical
    block that ``fit_zeta_to_h5`` fed the fused path as
    ``cct_trace_per_q`` — same reduction order, same bits, so the
    factored matrix is bit-identical to the one the fused
    ``batched_distributed_solve_lu`` factored every r-chunk.
    """
    nq, n_rmu, _ = C_q.shape
    n_log = int(n_rmu_logical)
    batched_distributed_getrf = backend_module(
        'scalapack').batched_distributed_getrf
    xy_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    key = (id(mesh_xy), int(nq), int(n_rmu), n_log)
    if key not in _transverse_scalapack_cache:
        @jax.jit
        def _prep(C):
            # Slice to the LOGICAL extent (load-bearing: pad-extent LU
            # roundoff is amplified O(1) in the near-null transverse
            # modes — ROOT_CAUSE.md 2026-07-08) and add the per-q ridge.
            C_log = jax.lax.with_sharding_constraint(
                C[:, :n_log, :n_log], xy_shard)
            trace_per_q = jnp.einsum('qii->q', C_log)
            ridge = (_TRANSVERSE_LU_RIDGE
                     * jnp.abs(trace_per_q) / n_log)[:, None, None]
            eye_n = jnp.eye(n_log, dtype=C.dtype)[None, :, :]
            return jax.lax.with_sharding_constraint(
                C_log + ridge * eye_n, xy_shard)
        _transverse_scalapack_cache[key] = _prep
    C_reg = _transverse_scalapack_cache[key](C_q)
    LU_q, ipiv_q = batched_distributed_getrf(C_reg, mesh=mesh_xy)
    return LU_q, ipiv_q


# =============================================================================
# The `distributed` ζ tier — 2D-sharded rank truncation and back-solve
# =============================================================================
#
# LAYOUT CONTRACT for everything in this section (nothing here ever
# replicates an O(μ²) object):
#
#     C_q, C⁺   (nq, μ, μ)  P(None, 'x', 'y')   rows on 'x', cols on 'y'
#     V         (nq, μ, μ)  P(None, 'x', 'y')   eigenvectors as COLUMNS
#     λ         (nq, μ)     replicated          ascending, IDENTICAL per rank
#     Z, ζ      (nq, μ, r)  P(None, 'x', 'y')   μ on 'x', r on 'y'
#
# Z arriving on 'x'/'y' rather than columns-on-the-FLAT-mesh is the whole
# reason this works.  Scorecard J.9 recorded the failure of the first
# attempt: with Z at P(None, None, ('x','y')) the ranks sharing a `y` index
# hold UNRELATED column blocks, so a psum over 'x' sums partial products
# built from different columns — NaNs, silently (the gate caught them only
# as float-count deficits in eqp).  A block-sharded (μ,μ) operator requires
# the ranks that share a column block to cooperate on the μ contraction, so
# this tier keeps Z in the layout it is BUILT in (P(None,'x','y')) and never
# does the `_reshard_z` two-step all-to-all at all.  Net communication is
# strictly LOWER than the replicated/per_q tiers — see the accounting in
# :func:`_distributed_pinv_apply`.

_dist_factor_cache: dict = {}   # distributed rank-truncate factor kernel
_dist_solve_cache: dict = {}    # distributed back-solve GEMM kernel


def _distributed_q_batch(nq: int, per_q_bytes: int) -> int:
    """q-batch size bounding the GEMM's gathered transient.

    Reuses :data:`_ZETA_GATHER_MAX_BYTES` (``LORRAX_ZETA_GATHER_CAP_GIB``,
    4 GiB) because it gates exactly the same thing here as it does for the
    other tiers: the live extent of the back-solve's gathered operands.
    """
    return max(1, min(int(nq), _ZETA_GATHER_MAX_BYTES // max(1, per_q_bytes)))


# --------------------------------------------------------------------------
# COLLECTIVE PAYLOAD CHUNKING  (scorecard AF)
#
# A memory budget is NOT a transport budget.  ``_ZETA_GATHER_MAX_BYTES``
# (4 GiB) bounds how much gathered data may be LIVE; it says nothing about
# how many bytes ONE `all_gather` / `psum_scatter` instruction hands to the
# interconnect in a single shot.  Those are different quantities, and only
# the second one is what a fabric actually has to survive.
#
# THIS IS TRANSPORT-AGNOSTIC, AND DELIBERATELY SO.  Nothing below is
# specific to a backend, a fabric or a machine: the tier still issues plain
# ``lax.all_gather`` / ``lax.psum_scatter``, just in bounded per-instruction
# payloads.  The identical code path runs unchanged on NCCL/CUDA, on any
# other XLA backend, and on any interconnect -- there is no transport probe,
# no per-fabric branch and no environment sniffing anywhere load-bearing.
# Bounded collectives are simply the robust regime everywhere; large
# single-shot ones are the fragile regime everywhere, and differ only in HOW
# they degrade (a slow tail, a retry storm, or an outright transport error).
# The default below is a transport-agnostic bound, NOT a tuning constant for
# any one cluster.
#
# It was CALIBRATED against the loudest available failure, which is the only
# reason a specific machine appears here at all.  MEASURED (scorecard AC.2,
# job 7876062, 72 nodes x 2 ranks, 12x12 mesh, nq=144, mu_pad=2448): the C+
# formation below issued ONE all_gather of 144*2448*204*16 = 1.15 GB and ONE
# psum_scatter of the same order, and the job died inside that instruction
# with MaxRSS 10.69 GB against an 85 GB budget -- i.e. every memory cap was
# satisfied and it still died.  The counter-evidence that FIXES the bound
# rather than merely lowering it: the sibling `per_q` tier ran HEALTHY on the
# identical 144 ranks (job 7876086) while issuing collectives of 0.104 GB.
# So ~100 MB per instruction is a measured-good payload and ~1.15 GB a
# measured-fatal one; the default sits just above the measured-good point.
#
# The chunking is done as a HOST-LEVEL loop over q-blocks -- one XLA
# execution per block -- rather than a loop inside one jit.  That is
# deliberate: XLA carries collective-combiner passes that merge adjacent
# same-axis collectives back into one instruction, so a loop inside the jit
# is a chunked-in-Python / fused-in-HLO non-fix.  Separate executions cannot
# be combined by construction, and the HLO dump gate proves the bound.
_DEFAULT_COLLECTIVE_CHUNK_MB = 128.0


def _collective_chunk_bytes() -> int:
    """Upper bound on ONE emitted collective's payload, in bytes.

    ``LORRAX_COLLECTIVE_CHUNK_MB`` (default 128 MB, see the note above).
    ``0`` or a negative value disables chunking entirely and restores the
    pre-AF single-shot behaviour — kept only so the failure can be
    reproduced on demand.
    """
    try:
        mb = float(os.environ.get("LORRAX_COLLECTIVE_CHUNK_MB",
                                  _DEFAULT_COLLECTIVE_CHUNK_MB))
    except ValueError:
        mb = _DEFAULT_COLLECTIVE_CHUNK_MB
    if mb <= 0:
        return 1 << 62                      # "no bound" — reproduction only
    return max(1, int(mb * (1 << 20)))


def _chunk_q(nq: int, per_q_collective_bytes: int) -> int:
    """Largest q-block whose LARGEST single collective fits the budget.

    ``per_q_collective_bytes`` must be the size of the BIGGEST collective
    the block emits per q — not the sum over collectives and not the live
    footprint.  The bound is per-instruction because that is what the
    transport sees.
    """
    return max(1, min(int(nq),
                      _collective_chunk_bytes()
                      // max(1, int(per_q_collective_bytes))))


_chunk_logged: set = set()


def _chunk_log(where: str, nq: int, qb: int, per_q_bytes: int) -> None:
    """One line per call site naming the emitted per-collective payload.

    On by default (``LORRAX_COLLECTIVE_CHUNK_LOG=0`` silences it): a tier
    that silently stopped chunking would otherwise be invisible until it
    took a 72-node job down again.  Deduplicated on the tuple, because the
    back-solve site is re-entered once per r-chunk (9–81 times).
    """
    # --- LOUD FLOOR (size campaign 2026-07-29, owner-approved) -------------
    # `_chunk_q` splits the q axis ONLY.  Once ONE q's collective exceeds the
    # budget its `max(1, ...)` floor returns q_block=1 and there is no
    # granularity left: the advertised bound is then simply ABANDONED and the
    # emitted payload grows as mu^2 unchecked.  That is a silent downgrade of a
    # bound this module advertises, which the project's rules forbid.
    #
    # Trigger arithmetic differs PER CALL SITE — do not quote one threshold:
    #   C+ formation (pinv):     per-q ~ mu^2 * 16 / Px
    #        -> breaches 128 MiB above mu ~ sqrt(budget*Px/16) = 8,192 (Px=8)
    #   C+ back-solve (GEMM):    per-q ~ (mu^2/Px + mu*r_chunk/Py) * 16,
    #        i.e. dominated by the mu*r_chunk term, LINEAR in mu
    #        -> breaches 128 MiB above mu ~ budget*Py/(16*r_chunk) = 1,456
    #           at r_chunk = 46,080, Py = 8.
    # MEASURED, and it corrected the first version of this comment: at the
    # campaign's SMALLEST deck (mu=2475) the pinv site sits at 12.5 MB/q
    # (fine, q_block=10) while the back-solve site already emits 230.0 MB/q
    # (1.7x over).  So the back-solve bound has been violated by essentially
    # EVERY run this project has ever made, not merely by mu > 8,192.
    # Larger measured back-solve payloads: 926 MB at mu=10015, 1386 MB at
    # 15007, 1773 MB at 24933 -- up to 13.2x the cap.
    # NOTE: no failure has ever been attributed to the violation; this is an
    # honesty fix, not a wall.  Deliberately announced BEFORE the
    # LORRAX_COLLECTIVE_CHUNK_LOG check — a routine-logging knob must not be
    # able to silence a bound violation — and with its own dedup key.
    _budget = _collective_chunk_bytes()
    if int(qb) <= 1 and int(per_q_bytes) > _budget and jax.process_index() == 0:
        _wsig = ("__floor__", where, int(per_q_bytes))
        if _wsig not in _chunk_logged:
            _chunk_logged.add(_wsig)
            print(f"  [collective chunk] WARNING {where}: cannot honour the "
                  f"payload bound — one q alone emits "
                  f"{per_q_bytes / 1e6:.1f} MB against a "
                  f"{_budget / 1e6:.1f} MB budget "
                  f"({per_q_bytes / max(1.0, _budget):.1f}x). q is the only "
                  f"split axis, so q_block is already 1 and the bound is "
                  f"ABANDONED, not enforced. The per-q payload grows with mu "
                  f"at this site (pinv ~ mu^2/Px; back-solve ~ mu*r_chunk/Py, "
                  f"so the back-solve bound is already unhonourable at the "
                  f"smallest production mu). No failure has been attributed to "
                  f"this; raise LORRAX_COLLECTIVE_CHUNK_MB to silence it "
                  f"honestly, or accept the larger payload.",
                  flush=True)
    if not env_bool("LORRAX_COLLECTIVE_CHUNK_LOG", True):
        return
    if jax.process_index() != 0:
        return
    sig = (where, int(nq), int(qb), int(per_q_bytes))
    if sig in _chunk_logged:
        return
    _chunk_logged.add(sig)
    nblk = (int(nq) + int(qb) - 1) // max(1, int(qb))
    print(f"  [collective chunk] {where}: nq={nq} q_block={qb} "
          f"({nblk} executions) max collective/exec = "
          f"{qb * per_q_bytes / 1e6:.1f} MB "
          f"(cap {_collective_chunk_bytes() / 1e6:.1f} MB, "
          f"unchunked would be {nq * per_q_bytes / 1e9:.3f} GB)",
          flush=True)


def _factor_c_q_distributed_rank_truncate(
    C_q: jax.Array, mesh_xy: Mesh, n_rmu_logical: int,
    zeta_rcond: float = 1e-8,
    indefinite: bool = False,
) -> jax.Array:
    """Truncated pseudo-inverse ``C⁺``, formed and kept 2D-SHARDED.

    Same physics as :func:`_factor_c_q_replicated`'s ``rank_truncate``
    branch — drop ``λ < rcond·λ_max``, then ``C⁺ = Σ_{keep} vᵢvᵢᴴ/λᵢ`` —
    with two structural differences:

    1. the ``eigh`` is DISTRIBUTED (ScaLAPACK ``pzheevd`` over the whole
       mesh), so the O(nq·μ³) factorisation finally divides by P instead of
       running redundantly on every rank;
    2. ``C⁺`` is returned EXPLICITLY (not as the factor ``B`` with
       ``BBᴴ = C⁺``).  Explicit costs one extra ``nq·μ³`` at fit time but
       halves the per-r-chunk back-solve: one GEMM ``C⁺Z`` instead of two
       (``B(BᴴZ)``), and the r-chunk loop runs 9–81 times.

    PADDED extent, deliberately.  The other charge routes factor at the
    LOGICAL extent and re-embed identity, because a blocked factorisation
    regroups partial sums when the extent changes.  ScaLAPACK's descriptors
    need ``n`` divisible by both mesh axes, which ``n_rmu_logical`` in
    general is not and ``n_rmu_padded`` always is — so this route factors
    the identity-padded block-diagonal ``[C_log 0; 0 I]``.  That is exact,
    not a compromise: the blocks do not mix, so ``C⁺``'s logical block is
    ``pinv(C_log)`` and its pad block is ``I`` or ``0`` depending on which
    side of the cut ``λ = 1`` lands; either way ζ's pad rows come out zero
    because Z's pad rows are exactly zero (the bilinear-in-zero-padded-ψ
    contract).  The *floating-point* consequence is that ζ from this tier
    agrees with the replicated tier to ~κ·ε rather than bit-exactly —
    which is already true of any block-cyclic eigh (different gauge), and
    is why the tier is explicit opt-in.

    ``λ`` is replicated by ScaLAPACK's own contract (``W`` is a global
    output computed on every process of the grid), so the truncation mask
    is computed LOCALLY and is identical on every rank by construction —
    no collective, and no chance of a rank-dependent cut.

    ``indefinite=True`` (2026-08-01) is the TRANSVERSE-channel mode
    (``transverse_zeta_solve='rank_truncate'`` +
    ``distributed_zeta_solve='distributed'``): the transverse CCT is
    Hermitian INDEFINITE, so (a) the cut is on ``|λ|`` (both signs are
    physical) and (b) the pad block is ZEROED instead of identity —
    ``[C_log 0; 0 0]`` — so the pad eigenvalues are exactly 0, are
    truncated for EVERY τ, and can never contaminate ``σ_max`` (an
    identity pad's λ=1 modes could win σ_max on a small-|λ| transverse
    spectrum; zeros cannot).  Zero rows/cols stay exact zeros through the
    Householder tridiagonalization and deflate exactly, so the pad modes
    are inert in the same block-diagonal sense the charge note above
    argues — and their ``inv=0`` removes them from C⁺ regardless.  THIS
    is what removes the transverse mesh-divisibility constraint: the
    eigh runs at the PADDED extent (divisible by both axes by
    construction of ``n_rmu_padded``), where the LU family had to refuse
    (pad-extent LU roundoff is amplified O(1) through the near-null
    modes that rank truncation removes).
    """
    nq, n_pad, _ = C_q.shape
    n_log = int(n_rmu_logical)
    if indefinite:
        # The input arrives identity-padded (factor_c_q's entry pad).
        # Restore exact-zero pad rows/cols INCLUDING the diagonal — a
        # local elementwise mask, no collective.
        rcond = float(zeta_rcond)   # transverse_zeta_rcond: no env twin
        if n_pad > n_log:
            xy_sh_in = NamedSharding(mesh_xy, P(None, 'x', 'y'))
            _row_log = (jnp.arange(n_pad) < n_log)
            C_q = jax.lax.with_sharding_constraint(
                C_q * (_row_log[None, :, None] & _row_log[None, None, :]),
                xy_sh_in)
    else:
        # DEPRECATED env form — the input key is the record (scorecard AV).
        rcond = _deprecated_env_float(
            "LORRAX_ZETA_RCOND", "zeta_rcond", zeta_rcond)
    rank_log = env_bool("LORRAX_ZETA_RANK_LOG", True)

    # ONE resolved plan, then one call.  ``'distributed'`` (not a hard-coded
    # 'scalapack') is deliberate and is the SAME name ``_resolve_zeta_gather``
    # approved: the platform default (ScaLAPACK on cpu, cuSOLVERMp on CUDA,
    # ``resolve._DISTRIBUTED_DEFAULT``) is then chosen in ONE place instead of
    # two that can disagree — naming scalapack here made a CUDA mesh pass the
    # tier's resolve guard and then hit ffi.scalapack's host-only check at
    # call time.  ``plan.batched`` uses ScaLAPACK's real batched entry point
    # (one descriptor + one workspace for the whole (nq, μ, μ) stack) and
    # falls back to a per-q loop for a backend that has none, so this call
    # site does not encode which is which.
    eigh_plan = linalg_plan('eigh', mesh_xy, backend='distributed',
                            n=int(n_pad))
    W, V = eigh_plan.batched(C_q)

    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    # The two collectives `_pinv_local` emits, per q:
    #   all_gather('x')   (μ/Px, μ/Py) -> (μ, μ/Py)   = μ²/Py · 16 B
    #   psum_scatter('y') (μ/Px, μ)    -> (μ/Px, μ/Py) = μ²/Px · 16 B
    # The BIGGER of the two sets the q-block (see `_chunk_q`).
    per_q_coll = max(n_pad * (n_pad // py), (n_pad // px) * n_pad) * 16
    qb = _chunk_q(nq, per_q_coll)
    _chunk_log('C+ formation (pinv)', nq, qb, per_q_coll)

    key = ('dist_rank_trunc', id(mesh_xy), int(nq), int(n_pad), n_log,
           float(rcond), bool(rank_log), int(qb), bool(indefinite))
    if key not in _dist_factor_cache:
        out_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))

        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, 'x', 'y'), P(None, None)),
                 out_specs=P(None, 'x', 'y'), check_rep=False)
        def _pinv_local(V_loc, inv_lam):
            # C⁺[i, j] = Σ_k V[i,k]·inv_k·conj(V[j,k]).
            #   V_loc     (nqb, μ/Px, μ/Py)  rows i on 'x', cols k on 'y'
            #   inv_lam   (nqb, μ)           replicated
            # My k-block is the 'y' slice of the replicated inv vector.
            ncol = V_loc.shape[2]
            y_i = jax.lax.axis_index('y')
            inv_my = jax.lax.dynamic_slice_in_dim(
                inv_lam, y_i * ncol, ncol, axis=1)          # (nqb, μ/Py)
            Vs = V_loc * inv_my[:, None, :].astype(V_loc.dtype)
            # The ONE transpose-class collective: rows j live on the mesh
            # ROW indexed by their block, so getting "all j, my k-block"
            # is an all-gather along 'x'.  μ²/Py per rank per q-batch.
            V_all_rows = jax.lax.all_gather(
                V_loc, 'x', axis=1, tiled=True)             # (nqb, μ, μ/Py)
            partial_ij = jnp.einsum('qik,qjk->qij', Vs, jnp.conj(V_all_rows))
            # Sum the k-blocks and land j on 'y' in one reduce-scatter —
            # never materialising the (μ/Px, μ) full-row product globally.
            return jax.lax.psum_scatter(
                partial_ij, 'y', scatter_dimension=2, tiled=True)

        @jax.jit
        def _masks(lam):
            if indefinite:
                # TRANSVERSE mode: cut on |λ| (Hermitian indefinite).
                # The pad modes are exactly 0 (zero-padded input), so
                # they never set σ_max and are dropped for every τ > 0
                # — no positional pad logic needed at all.
                sig = jnp.abs(lam)
                sig_max = jnp.max(sig, axis=-1, keepdims=True)
                keep = sig > (rcond * sig_max)
                inv = jnp.where(keep, 1.0 / jnp.where(keep, lam, 1.0), 0.0)
                if rank_log:
                    sig_keep_min = jnp.min(
                        jnp.where(keep, sig, jnp.inf), axis=-1)
                    n_keep = jnp.sum(keep, axis=-1)
                    jax.debug.print(
                        "[zeta transverse rank_truncate/distributed] "
                        "n_pad={n} rcond={rc:.1e} n_keep/q={k} "
                        "sig_max/q={mx} sig_min_kept/q={mn} kappa/q={kp} "
                        "sdrop_hi/q={dh}",
                        n=n_pad, rc=rcond, k=n_keep,
                        mx=sig_max[..., 0], mn=sig_keep_min,
                        kp=sig_max[..., 0] / sig_keep_min,
                        dh=jnp.max(jnp.where(keep, -jnp.inf, sig), axis=-1),
                        ordered=False)
                return inv
            # λ_max must be the LOGICAL block's, not the padded matrix's,
            # or the cut moves with the device count.  The padded matrix is
            # exactly block-diagonal [C_log 0; 0 I], so its spectrum is
            # spec(C_log) ∪ {1}×(n_pad−n_log).  Ascending order then makes
            # this exact: if λ_max > 1 the top mode belongs to C_log; if
            # λ_max ≤ 1 the (n_pad−n_log) pad ones ARE the top modes, so
            # C_log's largest sits at index n_log−1.  (n_pad == n_log makes
            # both branches the same element.)
            lam_max = jnp.where(lam[..., -1:] > 1.0,
                                lam[..., -1:],
                                lam[..., n_log - 1:n_log])
            keep = lam > (rcond * lam_max)
            inv = jnp.where(keep, 1.0 / jnp.where(keep, lam, 1.0), 0.0)
            if rank_log:
                # Same conditioning signal the replicated route prints —
                # n_keep/q is what tells you n_μ has over-completed the
                # pair-density rank — plus the same three audit fields (see
                # the replicated route's note and ``common/rank_criterion``):
                # the achieved amplification κ_eff = λ_max/λ_min(kept), which
                # the cut exists to bound at 1/rcond; the top of the discarded
                # band; and the margin to the §R19 cliff.
                # CAVEAT specific to this route: ``lam`` is the spectrum of
                # the PADDED matrix [C_log 0; 0 I], so the (n_pad − n_log)
                # eigenvalues exactly equal to 1 are pad, not physics.  They
                # are dropped whenever rcond·λ_max > 1 (always, at production
                # λ_max ~ 1e11 and rcond 1e-8), so n_keep is clean — but
                # ldrop_hi can be the pad value 1.0 rather than a physical λ.
                lam_keep_min = jnp.min(jnp.where(keep, lam, jnp.inf), axis=-1)
                n_keep = jnp.sum(keep, axis=-1)
                n_loose = jnp.sum(lam > (rcond * 1e-4 * lam_max), axis=-1)
                jax.debug.print(
                    "[zeta rank_truncate/distributed] n_pad={n} rcond={rc:.1e} "
                    "n_keep/q={k} lam_max/q={mx} lam_min_kept/q={mn} "
                    "kappa/q={kp} ldrop_hi/q={dh} margin/q={mg}",
                    n=n_pad, rc=rcond, k=n_keep,
                    mx=lam_max[..., 0], mn=lam_keep_min,
                    kp=lam_max[..., 0] / lam_keep_min,
                    dh=jnp.max(jnp.where(keep, -jnp.inf, lam), axis=-1),
                    mg=(n_loose - n_keep) / jnp.maximum(n_keep, 1),
                    ordered=False)
            return inv

        @partial(jax.jit, out_shardings=out_sh, donate_argnums=(2,))
        def _block(V_blk, inv_blk, acc, q0):
            return jax.lax.dynamic_update_slice(
                acc, _pinv_local(V_blk, inv_blk), (q0, 0, 0))

        # Zeros in the OUTPUT layout.  `V` already carries it, so this is a
        # local fill on every rank — no collective, no host round-trip.
        _zeros = jax.jit(jnp.zeros_like, out_shardings=out_sh)

        _dist_factor_cache[key] = (_masks, _block, _zeros)

    _masks, _block, _zeros = _dist_factor_cache[key]
    inv = _masks(W)
    C_pinv = _zeros(V)
    # Host-level q-block loop: ONE XLA execution per block, so the emitted
    # all_gather / psum_scatter payloads are bounded by construction and
    # cannot be re-combined by a compiler pass (see the AF note above).
    # At most two compiled shapes (full blocks + the remainder).
    for q0 in range(0, nq, qb):
        q1 = min(q0 + qb, nq)
        C_pinv = _block(V[q0:q1], inv[q0:q1], C_pinv, q0)
    return C_pinv


def _distributed_pinv_apply(
    C_pinv: jax.Array, Z_q: jax.Array, mesh_xy: Mesh, n_rmu_logical: int,
) -> jax.Array:
    """ζ = C⁺ Z as a stacked GEMM with BOTH operands 2D-sharded.

    ``out[q,i,j] = Σ_k C⁺[q,i,k]·Z[q,k,j]`` with ``C⁺`` at
    ``P(None,'x','y')`` (i on 'x', k on 'y') and ``Z`` at the same spec
    (k on 'x', j on 'y') — the classic 2-D block GEMM pairing.  Rank (x,y)
    all-gathers C⁺'s row-block along 'y' (full k for its own i rows) and
    Z's column-block along 'x' (full k for its own j columns), multiplies
    locally, and is done: no psum, and the output lands at
    ``P(None,'x','y')`` with no further movement.

    COMMUNICATION, honestly counted (per rank, per r-chunk, μ_pad=μ,
    r = r_chunk, mesh Px×Py):

        this tier   nq·(μ²/Px + μ·r/Py)·16 B   received
        replicated  nq·μ²·16 B                 received (the whole factor)
        per_q       nq·μ²·16 B                 received (same total, lower peak)

    At MoS2 12×12 (nq=144, μ=2016, r_chunk=11664, 12×12 mesh) that is
    5.3 GB/rank/r-chunk here against 9.4 GB/rank/r-chunk for the other two
    — 1.8× less traffic AND a 36.8 MB live transient per q instead of a
    65 MB gathered tile (replicated: 9.4 GB).  On top of that this tier
    does NOT run ``_reshard_z`` (two all-to-alls moving the whole
    ``nq·μ·r`` tensor) and skips the first leg of the output reshard,
    because Z is consumed in the layout it is built in.

    The q axis is batched to bound the gathered transient (see
    :func:`_distributed_q_batch`); the GEMM is per-q independent, so the
    batching is invisible to the result.
    """
    nq, n_pad, _ = C_pinv.shape
    n_zcols = int(Z_q.shape[2])
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    xy_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    n_log = int(n_rmu_logical)

    per_q_bytes = (n_pad * (n_pad // px) + n_pad * (n_zcols // py)) * 16
    # TWO bounds, and they answer different questions (scorecard AF):
    #   `_distributed_q_batch`  — how much gathered data may be LIVE
    #                             (LORRAX_ZETA_GATHER_CAP_GIB, 4 GiB).
    #   `_chunk_q`              — how many bytes ONE collective instruction
    #                             may hand to the transport in a single shot
    #                             (LORRAX_COLLECTIVE_CHUNK_MB, 128 MB).
    # The GEMM emits two gathers per q: C⁺'s row block over 'y'
    # (μ·μ/Px·16) and Z's column block over 'x' (μ·r/Py·16).  The second is
    # the larger at production r_chunk and is what sets the block.
    per_q_coll = max(n_pad * (n_pad // px), n_pad * (n_zcols // py)) * 16
    qb = min(_distributed_q_batch(nq, per_q_bytes),
             _chunk_q(nq, per_q_coll))
    _chunk_log('C+ back-solve (GEMM)', nq, qb, per_q_coll)

    # NOTE on the eager ``C_pinv[q0:q1]`` / ``Z_q[q0:q1]`` slices below:
    # the sibling per_q tier (``_solve_one_q_and_update``) slices INSIDE
    # its jit off a traced q, which gives one compiled shape for the whole
    # loop.  Here the slices are eager, so a non-dividing ``nq`` gives a
    # second compiled shape for the remainder block — bounded at two, and
    # deliberate: the q-batch is chosen from a BYTE budget, so making the
    # block shape uniform would mean padding nq and factoring q-blocks
    # that do not exist.  Two compiles + two transient slices per r-chunk
    # against ~nq of each on the traced-q form; at MoS2 12x12 (nq=144,
    # qb=116) that is 2 blocks per r-chunk.
    key = ('dist_pinv_apply', id(mesh_xy), int(nq), int(n_pad), n_log,
           n_zcols, int(qb))
    if key not in _dist_solve_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, 'x', 'y'), P(None, 'x', 'y')),
                 out_specs=P(None, 'x', 'y'), check_rep=False)
        def _gemm(A_loc, B_loc):
            A_row = jax.lax.all_gather(A_loc, 'y', axis=2, tiled=True)
            B_col = jax.lax.all_gather(B_loc, 'x', axis=1, tiled=True)
            return jnp.einsum('qik,qkj->qij', A_row, B_col)

        # Pad rows of ζ must be exactly zero (the contract every other
        # route gets from ``solve_at_logical``'s zero-refill).  Here they
        # are only ~1e-16 noise from C⁺'s inter-block coupling, so mask
        # them — a local elementwise op, no collective.
        row_keep = (jnp.arange(n_pad) < n_log)

        @partial(jax.jit, donate_argnums=(2,))
        def _block(A_blk, B_blk, zeta_acc, q0):
            out = _gemm(A_blk, B_blk)
            out = jnp.where(row_keep[None, :, None], out, 0)
            return jax.lax.dynamic_update_slice(zeta_acc, out, (q0, 0, 0))

        _dist_solve_cache[key] = _block

    apply_block = _dist_solve_cache[key]
    Z_q = jax.lax.with_sharding_constraint(Z_q, xy_sh)
    zeta = jnp.zeros_like(Z_q)
    # Python loop, not lax.scan: a scan over a sharded accumulator makes
    # SPMD replicate it (documented at solve_zeta's q-batch loop).  At most
    # two compiled shapes (full blocks + the remainder).
    for q0 in range(0, nq, qb):
        q1 = min(q0 + qb, nq)
        zeta = apply_block(C_pinv[q0:q1], Z_q[q0:q1], zeta, q0)
    return zeta


def factor_c_q(
    C_q: jax.Array,
    mesh_xy: Mesh,
    block_size: int = None,
    vertex_mu_L: int = 0,
    n_rmu_logical: int | None = None,
    solver_kind: str = 'auto',
    zeta_ridge: float = 0.0,
    zeta_rcond: float = 1e-8,
    transverse_zeta_rcond: float = 1e-10,
) -> jax.Array:
    """
    Compute system-matrix L_q from CCT matrix.

    For ``vertex_mu_L == 0`` (standard spin-traced path) the CCT is
    Hermitian positive-definite (modulo numerical noise); we run the
    optimized 2D blocked Cholesky and return the lower-triangular
    factor.  Downstream :func:`solve_zeta` then does two
    triangular solves per-q.

    For ``vertex_mu_L != 0`` (transverse Lorentz channels γ̃^i, i∈{1,2,3})
    the CCT is Hermitian but **indefinite** — Cholesky NaNs; the factor
    is a per-q pivoted LU with a stabilising ridge, HOISTED here (once
    per channel) since 2026-08-01.  The return value is a PAIR
    ``(factor, piv)``: the local plan stores ``(LU, perm)`` for
    ``lax.linalg.lu_solve`` (bit-identical to the fused per-r-chunk
    ``jnp.linalg.solve`` it replaced), the scalapack plan stores the
    block-cyclic ``pXgetrf`` factors + per-rank ``ipiv`` for ``pXgetrs``
    per r-chunk, and the cusolvermp plan still passes the CCT through
    (fused per-r-chunk getrf+getrs; hoist not yet ported to CUDA).

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
            paths.  Tune via the ``zeta_rcond`` input key in the deck;
            the ``LORRAX_ZETA_RCOND`` env form is a DEPRECATED twin
            (scorecard AV: still wins when set non-empty, but loudly)
            slated for removal.

    Returns:
        For ``vertex_mu_L == 0``: L_q ``(nq, n_rmu, n_rmu)`` at PADDED
        extent, sharded ``P(None, 'x', 'y')`` — the Cholesky factor
        (block-diagonal ``[L_log 0; 0 I_pad]``) for the cholesky paths,
        or the rank-revealing pseudo-inverse factor ``B``
        (``B Bᴴ = C⁺``) for ``'replicated_rank_truncate'``.
        For ``vertex_mu_L ≠ 0``: the PAIR ``(factor, piv)`` described
        above (piv is None only on the cusolvermp passthrough).
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

    # Indefinite-CCT path: no Cholesky.  TWO solve families since
    # 2026-08-01, both hoisted (factor ONCE per channel, applied per
    # r-chunk):
    #   ridge (LU) family — per-q pivoted LU + 1e-12 ridge (see the
    #   "hoisted TRANSVERSE factor stage" section above);
    #   rank_truncate family — per-q eigh pseudo-inverse with an |λ| cut
    #   (transverse_zeta_solve='rank_truncate'; the charge conditioning
    #   cure ported to the indefinite CCT).
    # Returns a (factor, piv) PAIR:
    #   'lu'           -> (LU embedded at padded extent, perm)  [local]
    #   'scalapack_lu' -> (block-cyclic LU at logical extent, ipiv)
    #   'cusolvermp_lu'-> (identity-padded CCT passthrough, None)
    #                     [fused per-r-chunk getrf+getrs kept on CUDA]
    #   'transverse_rank_truncate'             -> (C⁺ at padded extent
    #                     via the replicated scaffolding, None)
    #   'distributed_transverse_rank_truncate' -> (C⁺ kept 2D-sharded
    #                     at padded extent, None)
    if int(vertex_mu_L) != 0:
        t_kind = _resolve_solver_kind(
            mesh_xy, int(vertex_mu_L), solver_kind, n_rmu=n_rmu_logical)
        if t_kind == 'transverse_rank_truncate':
            # LOCAL plan of the rank_truncate family (2026-08-01):
            # explicit truncated pseudo-inverse C⁺ per q, through the
            # SAME replicated scaffolding as the charge factor (batched
            # + q-parallel fold, identity pad re-embed) — the schedule
            # contract is inherited, not duplicated.  piv slot is None:
            # the back-solve is one matmul, no permutation exists.
            return factor_c_q_replicated_batched(
                C_q, mesh_xy, n_rmu_logical, zeta_ridge=0.0,
                charge_zeta_solve='transverse_rank_truncate',
                zeta_rcond=float(transverse_zeta_rcond)), None
        if t_kind == 'distributed_transverse_rank_truncate':
            # DISTRIBUTED plan: pzheevd at the PADDED extent with
            # zeroed (exactly inert) pad modes — the charge distributed
            # machinery in indefinite mode.
            return _factor_c_q_distributed_rank_truncate(
                C_q, mesh_xy, n_rmu_logical,
                zeta_rcond=float(transverse_zeta_rcond),
                indefinite=True), None
        if t_kind == 'scalapack_lu':
            return _factor_c_q_transverse_scalapack(
                C_q, mesh_xy, n_rmu_logical)
        if t_kind == 'cusolvermp_lu':
            if jax.process_index() == 0:
                print("  [zeta transverse factor] cusolvermp_lu keeps the "
                      "fused per-r-chunk getrf+getrs (factor hoist not yet "
                      "ported to the CUDA backend)", flush=True)
            return C_q, None
        if t_kind != 'lu':
            raise ValueError(
                f"factor_c_q: unknown transverse solver_kind {t_kind!r}")
        return _factor_c_q_transverse_lu(C_q, mesh_xy, n_rmu_logical)

    solver_kind = _resolve_solver_kind(mesh_xy, vertex_mu_L=0, solver_kind=solver_kind)

    Pr = mesh_xy.shape['x']
    Pc = mesh_xy.shape['y']

    # `distributed` ζ tier: distributed eigh -> local identical truncation
    # -> 2D-sharded C⁺.  Checked FIRST because it is an explicit opt-in
    # (``distributed_zeta_solve = distributed``) and must not be swallowed
    # by the single-device / 1-D shortcut below — pzheevd runs on a 1×1
    # mesh too, and the route must stay the one the caller asked for so the
    # back-solve sees the operator it expects.
    if solver_kind == 'distributed_rank_truncate':
        return _factor_c_q_distributed_rank_truncate(
            C_q, mesh_xy, n_rmu_logical, zeta_rcond=zeta_rcond)

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
        L_handle = backend_module('cusolvermp').batched_distributed_cholesky(
            C_q, mesh=mesh_xy)
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
        _slate_potrf = backend_module('slate').distributed_cholesky
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


def _distributed_backsolve(Z_q: jax.Array, mesh_xy: Mesh, run) -> jax.Array:
    """RHS pad → distributed back-solve → output reshard → trim.

    THE shared frame for every ζ back-solve that keeps the factor
    distributed — cuSolverMp ``potrs``, the cuSolverMp/ScaLAPACK
    ``getrf``+``getrs`` pair, and the ``distributed`` tier's ``C⁺Z``
    GEMM.  Those three differ ONLY in ``run``; the three things around
    it are identical and used to be written out three times:

    1. **NRHS padding.**  Every block-cyclic descriptor (and the GEMM's
       ``'y'``-sharded column block) needs the last axis divisible by
       ``Py``.  ``pad_last_axis_to`` appends zero columns, which give
       exactly zero solution columns, so this is free of arithmetic
       consequences.
    2. **The output reshard.**  All three land ζ at ``P(None,'x','y')``
       = ``(q_, μ_X, r_Y)``; the downstream G-flat accumulator wants
       ``(q_, μ_XY, r_)`` so its FFT runs sharding-preserving.  That is
       ONE all-to-all on ``'y'`` (:func:`_reshard_zeta_mu_X_r_Y_to_mu_XY`)
       — half of what the replicated/per_q tiers pay, because their
       shard_map back-solve lands ζ column-sharded over the flat mesh.
    3. **The trim** back to the caller's logical column count.

    Keeping them here is not only de-duplication: FFI-adjacent
    resharding is where this code base has lost the most time (J.9's
    silent NaNs from a Z re-layout, T.4's per-r-chunk recompile of one),
    so there is exactly one copy to keep right.

    ``run`` takes the PADDED Z and returns ζ at ``P(None,'x','y')``.
    """
    Py = int(mesh_xy.shape['y'])
    Z_pad, n_cols = pad_last_axis_to(Z_q, Py)
    zeta_out = _reshard_zeta_mu_X_r_Y_to_mu_XY(run(Z_pad), mesh_xy)
    if int(Z_pad.shape[-1]) != n_cols:
        return zeta_out[:, :, :n_cols]
    return zeta_out


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
    zeta_gather: str = "replicated",
    lu_piv: jax.Array | None = None,
) -> jax.Array:
    """
    Solve for zeta_q given pre-computed system matrix from
    :func:`factor_c_q`.

    For ``vertex_mu_L == 0`` ``L_q`` is the lower-triangular Cholesky
    factor of CCT and the inner solve is two triangular substitutions
    (``L y = Z`` then ``L^H ζ = y``).  This is the historical fast
    path — bit-identical to the previous implementation.

    For ``vertex_mu_L != 0`` the transverse CCT^μ is Hermitian but
    indefinite — γ̃^i ⊗ γ̃^i has both signs of eigenvalue, so Cholesky is
    invalid and the factor is a pivoted LU with a small diagonal ridge
    ``ε·|tr(C_log)|/n_log`` (ε = :data:`_TRANSVERSE_LU_RIDGE`).  Since
    the 2026-08 hoist ``factor_c_q`` computes that LU ONCE per channel:
    ``L_q`` then carries the packed factors and ``lu_piv`` the
    permutation (local plan) / per-rank ScaLAPACK ipiv (scalapack plan),
    and this routine only APPLIES them per r-chunk
    (``lax.linalg.lu_solve`` — bit-identical to the fused
    ``jnp.linalg.solve`` — or ``pXgetrs``).  When ``lu_piv`` is None
    (cusolvermp plan, or a legacy caller passing the raw CCT) the fused
    per-r-chunk factor+solve paths below remain and behave exactly as
    before.  Bunch-Kaufman LDL^T would be the natural Hermitian-
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
                     the matmul ζ = B(BᴴZ)), or
                     'distributed_rank_truncate' (the
                     ``distributed_zeta_solve='distributed'`` tier: ``L_q``
                     is the truncated pseudo-inverse ``C⁺`` itself, kept
                     2D-sharded, and the back-solve is one stacked GEMM
                     with BOTH operands 2D-sharded — see
                     :func:`_distributed_pinv_apply`).  'replicated_cholesky',
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
            # Defense in depth ONLY: the config path refuses (explicit
            # request) or announces (auto) this at RESOLVE time inside
            # ``_resolve_solver_kind_transverse``; reaching this branch
            # means a caller passed an explicit distributed kind without
            # the divisibility precondition.  Announce via print, not
            # warnings.warn — warnings dedupe/capture made the original
            # demotion effectively silent in production logs (the
            # ledgered "silent replicated-LU fallback").
            if jax.process_index() == 0:
                print(
                    f"  [solve_zeta] n_rmu_logical={n_log} not divisible "
                    f"by the {Px_}x{Py_} mesh axes; transverse LU falls "
                    f"back from {solver_kind} to the per-q "
                    f"jnp.linalg.solve path so the solve can run at the "
                    f"logical extent.", flush=True)
            solver_kind = 'lu'

    if solver_kind in ('distributed_rank_truncate',
                       'distributed_transverse_rank_truncate'):
        # `distributed` tier: L_q IS the truncated pseudo-inverse C⁺, kept
        # 2D-sharded.  ζ = C⁺Z is one stacked GEMM with BOTH operands at
        # P(None,'x','y') — no factor gather, and no Z re-layout (Z is
        # consumed in the layout z_q_from_psi_sm builds it in, which is
        # exactly what scorecard J.9's flat-mesh column sharding made
        # impossible).  The transverse spelling (2026-08-01) is the SAME
        # back-solve on the transverse C⁺ (formed in indefinite mode);
        # it runs at the padded extent by design, so it must NOT enter
        # the logical-extent mu_pad guard above this dispatch.
        return _distributed_backsolve(
            Z_q, mesh_xy,
            lambda Z: _distributed_pinv_apply(L_q, Z, mesh_xy, n_log))

    if solver_kind == 'cusolvermp_cholesky':
        # Distributed potrs: Z stays at P(None,'x','y'), no input reshard
        # and no all-gather of L.
        _mp = backend_module('cusolvermp')
        Px = int(mesh_xy.shape['x'])
        Py = int(mesh_xy.shape['y'])
        # Re-attach handle metadata (the raw array carries no shape/grid info).
        L_handle = _mp.CusolverMpBatchedLowerL(
            raw=L_q, mesh=mesh_xy, n=int(n_rmu),
            mb=int(n_rmu) // Px, nb=int(n_rmu) // Py, nbatch=int(nq),
        )
        return _distributed_backsolve(
            Z_q, mesh_xy,
            lambda Z: _mp.batched_distributed_potrs(L_handle, Z, mesh=mesh_xy))

    if solver_kind == 'scalapack_lu' and lu_piv is not None:
        # HOISTED distributed plan: ``L_q`` carries the block-cyclic
        # ``pXgetrf`` factors at the LOGICAL extent (factored ONCE per
        # channel in :func:`_factor_c_q_transverse_scalapack`, ridge
        # baked in) and ``lu_piv`` the per-rank ipiv shards.  Only
        # ``pXgetrs`` runs per r-chunk — same descriptors, same grid,
        # same solve bits as the fused handler's back-to-back pair.
        batched_distributed_getrs = backend_module(
            'scalapack').batched_distributed_getrs
        xy_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        n_rows_pad = int(Z_q.shape[1])

        def _run_getrs(Z):
            # μ-slice the RHS to the LOGICAL extent and zero-refill ζ's
            # pad rows after — the solve_at_logical invariant, written
            # out because the extent comes from Z (the factor operand is
            # ALREADY logical, so the helper would read the wrong pad
            # extent off it).  Z pad rows are exact zeros by the Phase 3a
            # contract, so the sliced system IS the logical system.
            Z_log = jax.lax.with_sharding_constraint(
                Z[:, :n_log, :], xy_shard)
            X_log = batched_distributed_getrs(
                L_q, lu_piv, Z_log, mesh=mesh_xy)
            if n_rows_pad != n_log:
                X_log = jnp.pad(
                    X_log, ((0, 0), (0, n_rows_pad - n_log), (0, 0)))
            return jax.lax.with_sharding_constraint(X_log, xy_shard)

        return _distributed_backsolve(Z_q, mesh_xy, _run_getrs)

    if solver_kind in ('cusolvermp_lu', 'scalapack_lu'):
        # FUSED distributed getrf+getrs for the transverse channels
        # (cusolvermp always — factor hoist not yet ported to CUDA — and
        # scalapack only for legacy callers that passed the raw CCT).
        # L_q here is the *unfactored* CCT^μ (Hermitian indefinite) —
        # factor_c_q passes it through.  Same input sharding, output
        # reshard, and column padding pattern as the cholesky branch.
        # The two backends share this branch verbatim — identical call
        # contract; scalapack is the host/CPU-backend twin (Cray LibSci).
        if lu_piv is not None:
            raise ValueError(
                f"solve_zeta: lu_piv was passed with solver_kind="
                f"{solver_kind!r}, but the fused branch expects the raw "
                f"CCT (the {solver_kind} factor hoist does not exist).")
        batched_distributed_solve_lu = backend_module(
            'scalapack' if solver_kind == 'scalapack_lu' else 'cusolvermp'
        ).batched_distributed_solve_lu

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

        def _run_lu(Z):
            zeta_xy = solve_at_logical(_dist_ridged_lu, n_log, (L_q,), Z)
            if mu_pad:
                # solve_at_logical's zero-refill re-embeds at the padded
                # extent; pin the layout back before the output reshard.
                zeta_xy = jax.lax.with_sharding_constraint(
                    zeta_xy, NamedSharding(mesh_xy, P(None, 'x', 'y')))
            return zeta_xy

        return _distributed_backsolve(Z_q, mesh_xy, _run_lu)

    # Compute padding needed for even sharding across all devices
    total_devices = mesh_xy.devices.size
    n_zchunk_padded = round_up(n_zchunk, total_devices)
    needs_padding = n_zchunk_padded != n_zchunk

    z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    # Staging layout for the Z reshard (see ``_reshard_z``): parks 'x' on
    # the leading nq axis so each with_sharding_constraint moves exactly
    # one mesh axis.
    intermediate_shard = NamedSharding(mesh_xy, P('x', None, 'y'))
    L_rep_shard = NamedSharding(mesh_xy, P(None, None))
    L_batch_rep_shard = NamedSharding(mesh_xy, P(None, None, None))  # (B_q, n_rmu, n_rmu)
    # The layout ``L_q`` actually ARRIVES in (2-D over the mesh face); the
    # per_q tier consumes it directly instead of asking for a replica.
    L_batch_xy_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
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
    LU_RIDGE = _TRANSVERSE_LU_RIDGE

    # HOISTED transverse back-solve: factor_c_q already ran the pivoted
    # LU (once per channel) and handed us (LU factors, permutation).
    # This routine then only APPLIES lax.linalg.lu_solve per r-chunk —
    # the same call jnp.linalg.solve makes after its internal lu(), so
    # the bits match the fused path exactly.  ``lu_piv is None`` keeps
    # the fused per-r-chunk factor+solve (legacy callers, cusolvermp).
    hoisted_lu = bool(use_lu and lu_piv is not None)

    # ``use_rank_trunc`` selects the matmul back-solve for the charge
    # rank-truncation factor: ``L_q`` is then the pseudo-inverse factor B
    # (B Bᴴ = C⁺) from ``_factor_c_q_replicated``, so ζ = C⁺Z = B(BᴴZ) is
    # two matmuls, NOT a triangular solve (C⁺ is rank-deficient — its
    # inverse does not exist, so the tri-solve would be wrong).
    use_rank_trunc = (solver_kind == 'replicated_rank_truncate')

    # ``use_pinv_T`` selects the transverse rank-truncation back-solve
    # (2026-08-01): ``L_q`` is the EXPLICIT truncated pseudo-inverse C⁺
    # of the indefinite transverse CCT (no BBᴴ factor exists there), so
    # ζ = C⁺Z is ONE matmul at the logical extent.  Flows through the
    # same replicated/per_q gather tiers as every other whole-tile
    # factor.
    use_pinv_T = (solver_kind == 'transverse_rank_truncate')

    # ``zeta_gather`` selects the GATHER GRANULARITY of the replicated
    # factor, not the factorization — see :func:`_resolve_zeta_gather`.
    #   'replicated' : one all-gather of the whole (q_batch, μ, μ) stack
    #                  (today's path, and what ``_solve_all_at_once``
    #                  does at q_batch = nq).
    #   'per_q'      : one all-gather of a SINGLE (1, μ, μ) tile at a
    #                  time, looped over q.  Same arithmetic per q as the
    #                  batched kernel — only the live gathered extent
    #                  changes, from ``nq·μ²·16`` to ``μ²·(1+1/Py)·16``.
    #                  The gather is written INSIDE a shard_map so the
    #                  partitioner cannot hoist it back to the full stack;
    #                  see ``_per_q_block`` for the measurement that forced
    #                  that form (scorecard Y.2).
    per_q_gather = (str(zeta_gather).strip().lower() == "per_q")

    # Cache key for solve function (includes q_chunk_size and padded size).
    # ``use_lu`` / ``use_rank_trunc`` partition the cache so the three
    # back-solve compiles don't collide on the same key.  ``n_log`` is
    # closure state of the kernels below (the slice extent), so it keys the
    # cache too.
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu, n_log,
                 n_zchunk_padded, q_chunk_size, bool(use_lu),
                 bool(use_rank_trunc), bool(per_q_gather),
                 bool(hoisted_lu), bool(use_pinv_T))

    # Uniform piv operand for the kernels below: the real (nq, n_log)
    # permutation on the hoisted-LU path, a (nq, 1) placeholder (dead
    # operand, DCE'd by XLA) everywhere else — same idiom as the
    # cct_trace placeholder in fit_one_rchunk.
    piv_arr = (lu_piv if hoisted_lu
               else jnp.zeros((nq, 1), dtype=jnp.int32))
    piv_rep_shard = NamedSharding(mesh_xy, P(None, None))

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

    def _lu_apply_logical(LU: jax.Array, piv: jax.Array,
                          Z: jax.Array) -> jax.Array:
        """HOISTED transverse back-solve at the LOGICAL μ extent: apply
        the per-q ``(LU, piv)`` factor that ``factor_c_q`` computed once
        per channel.  ``jax.scipy.linalg.lu_solve((lu, piv), Z)`` runs
        ``lu_pivots_to_permutation`` + ``lax_linalg.lu_solve`` — exactly
        the arithmetic ``jnp.linalg.solve`` runs after its internal
        ``lu()`` — so the result is bit-identical to the fused
        ``_ridge_indef_solve`` path this replaces (the ridge is baked
        into the factor).  ``piv`` is built at the logical extent
        already; ``solve_at_logical`` slices LU/Z and zero-refills ζ's
        pad rows (gate: tests/test_transverse_factor_hoist.py)."""
        return solve_at_logical(
            lambda LU_log, Z_log: jax.scipy.linalg.lu_solve(
                (LU_log, piv), Z_log, trans=0),
            n_log, (LU,), Z)

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

    def _pinv_apply_T_logical(Cp: jax.Array, Z: jax.Array) -> jax.Array:
        """Transverse rank-truncation back-solve at the LOGICAL μ
        extent: ζ = C⁺Z, ONE matmul (``Cp`` is the explicit truncated
        pseudo-inverse of the indefinite transverse CCT).  Same
        slice/zero-refill contract as the other whole-tile bodies."""
        def _mm(Cp_log, Z_log):
            return Cp_log @ Z_log
        return solve_at_logical(_mm, n_log, (Cp,), Z)

    if cache_key not in _solve_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            if use_rank_trunc:
                # Charge C⁺ pseudo-inverse factor: matmul back-solve.
                return _pinv_matmul_logical(L, Z_cols)
            if use_pinv_T:
                # Transverse explicit C⁺: one-matmul back-solve.
                return _pinv_apply_T_logical(L, Z_cols)
            if use_lu:
                # Indefinite CCT^μ: pivoted-LU back-solve with ridge.
                return _ridge_indef_solve(L, Z_cols)
            return _tri_solve_logical(L, Z_cols)

        # Vectorized solve for a batch of q-points.  ``piv_batch`` is the
        # hoisted-LU permutation (replicated; placeholder + dead on every
        # other path — see ``piv_arr`` above).
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None, None), P(None, None, ('x', 'y')),
                           P(None, None)),
                 out_specs=P(None, None, ('x', 'y')))
        def _sharded_cho_solve_batch(L_batch: jax.Array, Z_batch: jax.Array,
                                     piv_batch: jax.Array) -> jax.Array:
            """Solve (B_q, n_rmu, n_rmu) @ (B_q, n_rmu, n_cols) -> (B_q, n_rmu, n_cols)"""
            if use_rank_trunc:
                # C⁺ factor matmul back-solve, per-q vmapped (same reshard
                # plan as the Cholesky/LU paths so the caller is agnostic).
                return jax.vmap(_pinv_matmul_logical)(L_batch, Z_batch)
            if use_pinv_T:
                # Transverse explicit C⁺, per-q vmapped one-matmul.
                return jax.vmap(_pinv_apply_T_logical)(L_batch, Z_batch)
            if hoisted_lu:
                # Apply the once-per-channel (LU, perm) factor.
                return jax.vmap(_lu_apply_logical)(
                    L_batch, piv_batch, Z_batch)
            if use_lu:
                # FUSED fallback (lu_piv=None): ``jnp.linalg.solve`` is
                # natively batched on the leading axis and dispatches one
                # LU factorization per q.  Same vmap structure as the
                # Cholesky path so reshard plans match.  We vmap the
                # ridge-add per-q so each LU sees its own conditioning
                # shift.
                return jax.vmap(_ridge_indef_solve)(L_batch, Z_batch)
            return jax.vmap(_tri_solve_logical)(L_batch, Z_batch)

        @partial(jax.jit, donate_argnums=(2,))
        def _solve_batch_and_update(L_batch_sharded, Z_batch_col, zeta_acc,
                                    q_start, piv_batch):
            """Solve one q-batch and update zeta_acc via dynamic_update_slice.
            donate_argnums=(2,) donates zeta_acc so XLA reuses its buffer."""
            L_rep = jax.lax.with_sharding_constraint(L_batch_sharded, L_batch_rep_shard)
            piv_rep = jax.lax.with_sharding_constraint(piv_batch, piv_rep_shard)
            batch_result = _sharded_cho_solve_batch(L_rep, Z_batch_col, piv_rep)
            return jax.lax.dynamic_update_slice(zeta_acc, batch_result, (q_start, 0, 0))

        @jax.jit
        def _solve_all_at_once(L_q_sharded, Z_col, piv):
            """Fast path: solve all q-points in a single batched call."""
            L_full_rep = jax.lax.with_sharding_constraint(L_q_sharded, L_batch_rep_shard)
            piv_rep = jax.lax.with_sharding_constraint(piv, piv_rep_shard)
            return _sharded_cho_solve_batch(L_full_rep, Z_col, piv_rep)

        # PER-Q tier.  The q-selection happens INSIDE a shard_map, where
        # the gather is a `lax.all_gather` on an already-sliced tile — so
        # the per-q extent is a STRUCTURAL property of the program, not a
        # request the partitioner is free to reorder.
        #
        # HISTORY — do not regress this (scorecard Y.2, measured on the
        # production deck).  The first implementation sliced a traced ``q``
        # out of the sharded stack and then asked for the tile replicated::
        #
        #     L_one = lax.dynamic_slice_in_dim(L_q_sharded, q, 1, axis=0)
        #     L_one_rep = lax.with_sharding_constraint(L_one, replicated)
        #
        # which reads as "gather one (μ,μ) tile".  XLA:CPU's SPMD
        # partitioner does NOT sink a q-axis ``dynamic_slice`` through the
        # μ-axis ``all-gather`` even though the two commute: it emitted the
        # WHOLE ``(nq, μ_pad, μ_pad)`` gather and applied the slice
        # afterwards.  Measured at 1998 centroids / μ_pad = 2048 / P = 64,
        # the buffer assignment charged ``jit(_solve_one_q_and_update)``
        # ``nq·μ_pad·(μ_pad + μ_pad/P_x)·16`` = 10.87 GB — i.e. the tier's
        # own gather was LARGER than the ``replicated`` gather it exists to
        # avoid (9.66 GB), and because the module runs once per q it moved
        # 144× that per r-chunk.  That is the whole of the 12–40× wall-clock
        # penalty Y.1 measured, and it is why T.5's "9.36 GB → 0.065 GB"
        # headline was wrong.
        #
        # Inside a shard_map there is nothing left to hoist: the local
        # slice is a local slice of the rank's OWN ``(nq, μ/Px, μ/Py)``
        # block, and the two ``all_gather``s that follow it are written on
        # a single-q operand.  Gathered bytes per execution are exactly
        # ``μ_pad·(μ_pad/Py)·16 + μ_pad²·16`` — 75 MB at μ_pad = 2048,
        # independent of nq.
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, 'x', 'y'),            # L_q  (nq, μ, μ)
                           P(None, None, ('x', 'y')),    # Z_col
                           P(None, None, ('x', 'y')),    # zeta_acc
                           P(),                          # q (replicated scalar)
                           P(None, None)),               # piv (replicated)
                 out_specs=P(None, None, ('x', 'y')),
                 check_rep=False)
        def _per_q_block(L_loc, Z_loc, zeta_loc, q, piv_loc):
            # L_loc: (nq, μ/Px, μ/Py) — this rank's 2-D block of the stack.
            L_one = jax.lax.dynamic_slice_in_dim(L_loc, q, 1, axis=0)
            # Rebuild EXACTLY the replicated (1, μ, μ) tile the batched
            # kernel would have seen: 'x' owns axis 1, 'y' owns axis 2, so
            # the tiled all_gathers concatenate in mesh-index order.
            L_row = jax.lax.all_gather(L_one, 'x', axis=1, tiled=True)
            L_tile = jax.lax.all_gather(L_row, 'y', axis=2, tiled=True)
            Z_one = jax.lax.dynamic_slice_in_dim(Z_loc, q, 1, axis=0)
            piv_one = jax.lax.dynamic_slice_in_dim(piv_loc, q, 1, axis=0)
            # Same back-solve bodies as ``_sharded_cho_solve_batch``
            # at batch 1 — identical shapes, identical operand values,
            # therefore bit-identical arithmetic.
            if use_rank_trunc:
                out = jax.vmap(_pinv_matmul_logical)(L_tile, Z_one)
            elif use_pinv_T:
                out = jax.vmap(_pinv_apply_T_logical)(L_tile, Z_one)
            elif hoisted_lu:
                out = jax.vmap(_lu_apply_logical)(L_tile, piv_one, Z_one)
            elif use_lu:
                out = jax.vmap(_ridge_indef_solve)(L_tile, Z_one)
            else:
                out = jax.vmap(_tri_solve_logical)(L_tile, Z_one)
            return jax.lax.dynamic_update_slice_in_dim(
                zeta_loc, out, q, axis=0)

        @partial(jax.jit, donate_argnums=(2,))
        def _solve_one_q_and_update(L_q_sharded, Z_col, zeta_acc, q, piv):
            """PER-Q tier: gather ONE ``(μ, μ)`` factor tile, solve that q,
            scatter into ``zeta_acc``.

            ``q`` is a traced argument, so every iteration shares one
            trace, one compile and one executable, and ``Z_col`` is never
            sliced eagerly (an eager slice would materialise ``nq`` extra
            ``(1, μ, r/P)`` device arrays per r-chunk).  ``donate_argnums``
            chains ``zeta_acc`` through the loop the same way
            ``_solve_batch_and_update`` does.
            """
            L_xy = jax.lax.with_sharding_constraint(
                L_q_sharded, L_batch_xy_shard)
            piv_rep = jax.lax.with_sharding_constraint(piv, piv_rep_shard)
            return _per_q_block(L_xy, Z_col, zeta_acc, jnp.asarray(q),
                                piv_rep)

        # Z reshard P(None,'x','y') → P(None,None,('x','y')), staged
        # through P('x',None,'y') so each step moves ONE mesh axis (see
        # the call site below for the Involuntary-Remat measurement that
        # forced the two-step form).
        #
        # MUST live in the cache with the other kernels.  It used to be
        # defined at the call site, i.e. a FRESH ``jax.jit`` object per
        # r-chunk — and a fresh wrapper is a fresh key for JAX's
        # trace/lower/compile caches (they key on the wrapped function's
        # identity), so every r-chunk retraced, relowered and
        # RECOMPILED this reshard.  At production scale that is the one
        # XLA compilation inside the r-chunk loop, and it is on the
        # r_chunk-sized tensor.
        @partial(jax.jit, donate_argnums=(0,))
        def _reshard_z(z):
            z = jax.lax.with_sharding_constraint(z, intermediate_shard)
            return jax.lax.with_sharding_constraint(z, z_col_shard)

        _solve_cache[cache_key] = SimpleNamespace(
            solve_batch_and_update=_solve_batch_and_update,
            solve_all_at_once=_solve_all_at_once,
            sharded_cho_solve=_sharded_cho_solve,
            sharded_cho_solve_batch=_sharded_cho_solve_batch,
            solve_one_q_and_update=_solve_one_q_and_update,
            reshard_z=_reshard_z,
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
        # w_isdf._get_w_solve_fn_local (V/χ reshard).  See lorrax_B commit
        # c0307a0 for the original Si 4×4×4 result (HLO peak
        # 68.94 → 29.94 GB on the same kernel compile).
        # The kernel itself is built ONCE per cache_key (above) — building
        # it here made every r-chunk recompile it.
        Z_col = helpers.reshard_z(Z_q)
        # No-op when called inside an outer jit (tracer has no
        # block_until_ready and the outer jit syncs at its boundary).
        if not isinstance(Z_col, jax.core.Tracer):
            Z_col.block_until_ready()
        del Z_q

    # PER-Q tier: one (μ, μ) gather at a time.  Sits BEFORE the
    # ``q_batch >= nq`` fast path because it deliberately overrides the
    # planner's q_chunk (which is a compute-batching choice, not a memory
    # one) — the whole point of this tier is that the gathered extent is
    # independent of both nq and q_chunk_size.
    if per_q_gather:
        zeta = jnp.zeros_like(Z_col)
        # Python loop, not lax.scan/fori: a scan over a q-sharded carry
        # makes SPMD replicate the accumulator (documented at the
        # q-batch loop below).  Same reason, same shape of fix.
        for q in range(nq):
            zeta = helpers.solve_one_q_and_update(L_q, Z_col, zeta, q,
                                                  piv_arr)
        del Z_col
        zeta = _reshard_zeta_r_XY_to_mu_XY(zeta, mesh_xy)
        if needs_padding:
            return zeta[:, :, :n_zchunk]
        return zeta

    # Fast path: solve all q-points at once
    if q_batch >= nq:
        result = helpers.solve_all_at_once(L_q, Z_col, piv_arr)
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
        zeta = helpers.solve_batch_and_update(L_q[q0:q1], Z_col[q0:q1], zeta,
                                              q0, piv_arr[q0:q1])

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
    zeta_gather: str = 'replicated',
    lu_hoisted: bool = False,
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

    def solve_phase(Z_q, L_q, cct_trace_per_q, lu_piv=None):
        # IBZ-only solve (Phase B): L_q + cct_trace + lu_piv come in
        # PRE-SLICED at IBZ rows (via ``symmetry_maps.slice_q_full_to_ibz``
        # upstream in ``fit_zeta_to_h5``; the factor stage runs after the
        # slice, so its piv q-axis is already IBZ).  Z_q is built at full
        # BZ by ``z_q_phase``, so it gets the IBZ slice here.  When
        # ``q_irr_full_idx`` is None (write_ibz_only=False), all
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
            n_rmu_logical=int(meta.n_rmu),
            zeta_gather=zeta_gather,
            lu_piv=lu_piv)

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
        lu_piv,
    ):
        # Composed (z_q ∘ solve) under one ``@jax.jit`` — preserved for
        # the AOT memory model path which lowers a single callable.
        # Production ``fit_one_rchunk`` calls ``z_q_phase`` and
        # ``solve_phase`` directly with ``timing.section`` between, so
        # the per-r-chunk breakdown is host-visible.  ``lu_piv`` is a
        # placeholder unless ``lu_hoisted`` (the closure static that keys
        # the cache) — the hoisted/fused distinction must be structural.
        Z_q = z_q_phase(
            psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
            norms_l, norms_r, r_start_dyn, gamma_perm, gamma_phase)
        return solve_phase(Z_q, L_q, cct_trace_per_q,
                           lu_piv if lu_hoisted else None)

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
    zeta_gather: str = 'replicated',
    lu_piv: jax.Array | None = None,
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
        str(zeta_gather),
        bool(lu_piv is not None),
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
            zeta_gather=str(zeta_gather),
            lu_hoisted=bool(lu_piv is not None),
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
    # Same knob, same grammar as gw/isdf_fitting.py:937.  This was a bare
    # presence test, so ``LORRAX_RCHUNK_DEBUG=0`` turned the debug path ON
    # here and OFF there — one knob, two answers, in the same r-chunk loop.
    _dbg = env_bool("LORRAX_RCHUNK_DEBUG", False)
    # Per-phase host-RSS deltas: on CPU the XLA arena is invisible to
    # ``memory_stats()``, so attributing the per-r-chunk anonymous ramp
    # to z_q_build vs solve needs the kernel's own accounting.
    _r0 = host_rss_gb() if _dbg else 0.0
    _t_z0 = time.perf_counter() if _dbg else 0.0
    with timing.section("zeta_fit.chunk.z_q_build"):
        Z_q = fn.z_q_phase(
            psi_l_rmuT_X_fit, psi_r_rmuT_X_fit,
            norms_l, norms_r, r_start_dyn,
            gamma_perm, gamma_phase)
        Z_q.block_until_ready()
    _t_z = (time.perf_counter() - _t_z0) if _dbg else 0.0
    _r1 = host_rss_gb() if _dbg else 0.0
    _t_s0 = time.perf_counter() if _dbg else 0.0
    with timing.section("zeta_fit.chunk.solve"):
        zeta = fn.solve_phase(Z_q, L_q, cct_trace_per_q, lu_piv)
        zeta.block_until_ready()
    _t_s = (time.perf_counter() - _t_s0) if _dbg else 0.0
    if _dbg and jax.process_index() == 0:
        _r2 = host_rss_gb()
        print(f"[rchunk_dbg]   z_q_build={_t_z*1000:.0f}ms "
              f"solve={_t_s*1000:.0f}ms "
              f"d_zq={_r1 - _r0:+.3f}GB d_solve={_r2 - _r1:+.3f}GB",
              flush=True)
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

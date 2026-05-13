import gc
import os
import queue
import threading
import time
from types import SimpleNamespace
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from . import Meta
from . import timing
from .gamma_matrices import (
    gamma_perm_phase as _gamma_perm_phase_mu,
    gamma_double_contract,
)

_MEM_PROFILE = bool(os.environ.get("LORRAX_MEM_PROFILE", ""))

def _mem_report(label):
    """Print per-device memory if LORRAX_MEM_PROFILE is set. Zero cost otherwise."""
    if not _MEM_PROFILE:
        return
    gc.collect()
    s = jax.local_devices()[0].memory_stats()
    u = s.get('bytes_in_use', 0) / 1e9
    p = s.get('peak_bytes_in_use', 0) / 1e9
    if jax.process_index() == 0:
        print(f'  [MEM {label}] used={u:.3f} peak={p:.3f} GB', flush=True)
from common import jax_profile
from .cholesky_2d import (
    cholesky_2d_batched,
    dense_to_tiles,
    tiles_to_dense,
)
from .fft_helpers import (
    make_flat_k_ifftn,
    make_flat_k_fftn,
    compute_block_size_for_2d_cholesky,
)
from .load_wfns import (
    read_Gvecs_to_devices,
    load_centroids_band_chunked,
)


# ============================================================================
# Open-spin pair density: P_k,ab(μ, ν) = Σ_n ψ*_{n,k,a}(μ) ψ_{n,k,b}(ν)
# ============================================================================
#
# Single rank-5 pair-density path used by every channel (charge γ̃^0 = I_4
# AND transverse γ̃^i = α^i).  The (αβ) spin axes stay OPEN through the
# pair-density and IFFT steps; γ̃·γ̃ contraction happens at the post-IFFT
# reduction inside :func:`c_q_from_pair` and :func:`z_q_from_pair`.
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
# Memory cost: rank-5 P is ns²=16× the historical spin-traced rank-3 P,
# but only at the (k, a, b, μ, ν) carrier — the band-chunk accumulator
# streams over n so peak memory is bounded.  The basis-quality
# improvement (each ψ_α*ψ_β interpolated cleanly rather than the
# post-cancellation Σ_α ψ_α*ψ_α) was worth the trade for the bispinor
# pipeline; we now use it for the charge channel too so both paths
# share one set of helpers.

_pair_density_cache = {}
_accum_pair_density_cache = {}
_pair_pipeline_sm_cache = {}


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


def accum_pair_density(
	P_accum: jax.Array,
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""Band-chunk streaming accumulator for :func:`pair_density`.

	    P_accum += Σ_n ψ*_{n,a}(μ) ψ_{n,b}(col)        (open spin (a,b))

	Input shardings mirror :func:`pair_density`; ``P_accum`` lives on
	``P(None, None, None, 'x', 'y')`` and is donated.
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	cache_key = ('pair_density_accum', id(mesh_xy), nk, n_rmu, nb, ns, n_col)
	if cache_key not in _accum_pair_density_cache:
		P_sharding = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		L_sharding = NamedSharding(mesh_xy, P(None, 'x', None, None))
		R_sharding = NamedSharding(mesh_xy, P(None, None, None, 'y'))

		@partial(jax.jit,
				 in_shardings=(P_sharding, L_sharding, R_sharding),
				 out_shardings=P_sharding,
				 donate_argnums=(0,))
		def _accum(P_in: jax.Array, psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			return P_in + jnp.einsum('kmna,knbr->kabmr', psi_L, psi_R, optimize=True)

		_accum_pair_density_cache[cache_key] = _accum
	return _accum_pair_density_cache[cache_key](P_accum, psi_rmuT_X, psi_rcol_Y)


# Cache for ISDF pipeline jitted functions
_isdf_pipeline_cache = {}



# ============================================================================
# Left/Right CCT and ZCT (matching gw_jax physics)
# ============================================================================
# gw_jax uses separate left and right wavefunctions:
#   - Left:  bands 0 to b3 (all occupied + sigma conduction)
#   - Right: bands 0 to b4 (all occupied + all conduction up to nband)
#
# CCT: C_q(μ,ν) = Σ_k exp(iq·k) [Σ_n,s ψ_l,n,k,s(μ) ψ*_l,n,k,s(ν)]* × [Σ_m,s ψ*_r,m,k,s(μ) ψ_r,m,k,s(ν)]
#    = FFT_k→q[ conj(P_l_R) ⊙ P_r_R ] where P_l_R = IFFT(P_l_k), P_r_R = IFFT(P_r_k)
#
# ZCT follows the same pattern for (μ, r) instead of (μ, ν).

def c_q_from_pair(
	P_l_k_ab: jax.Array,
	P_r_k_ab: jax.Array,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	*,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""C_q from open-spin pair densities with optional γ̃ insertions.

	    C_q^{μ_L, ν_L}(μ, ν)
	      = FFT_k→q[ Σ_{αβα'β'} γ̃^{μ_L}_{αα'} γ̃^{ν_L}_{ββ'}
	                 · conj(IFFT(P_l_{αβ}))(R; μ, ν)
	                 · IFFT(P_r_{α'β'})(R; μ, ν) ]

	γ̃ identity short-circuit: pass ``gamma_L=None`` (or ``gamma_R=None``)
	to mean γ̃_L = I_4 (or γ̃_R = I_4); the corresponding gather + phase
	multiply is skipped at JIT trace time.  Both None → charge channel,
	pure Σ_{αβ} P_l_conj·P_r reduction.

	γ̃^μ are monomial (one non-zero per row/column, value ∈ {±1, ±i}),
	so each non-identity spin contraction is one ``jnp.take`` +
	element-wise phase multiply, not a 4×4 matmul.

	Inputs:
	    P_l_k_ab, P_r_k_ab : (nk, ns, ns, n_rmu, n_col) c128, sharded
	                        ``P(None, None, None, 'x', 'y')``.
	    gamma_L, gamma_R   : ``(perm, phase)`` tuples or ``None``.
	                        Build perm/phase via
	                        :func:`common.gamma_matrices.gamma_perm_phase`.
	    kgrid              : (nkx, nky, nkz).
	    mesh_xy            : 2-D device mesh.

	Output:
	    C_q : (nq, n_rmu, n_col) c128, sharded ``P(None, 'x', 'y')``.
	"""
	nkx, nky, nkz = kgrid
	nk, ns1, ns2, n_rmu, n_col = P_l_k_ab.shape
	assert n_col == n_rmu, (
		f"CCT expects square centroid columns, got n_col={n_col}, n_rmu={n_rmu}"
	)
	assert P_r_k_ab.shape == P_l_k_ab.shape, (
		f"P_l/P_r shape mismatch: {P_l_k_ab.shape} vs {P_r_k_ab.shape}"
	)
	lhs_id = gamma_L is None
	rhs_id = gamma_R is None

	cache_key = ('c_q_from_pair', id(mesh_xy), nk, ns1, ns2, n_rmu,
	             nkx, nky, nkz, lhs_id, rhs_id)

	if cache_key not in _isdf_pipeline_cache:
		spin_spec = P(None, None, None, 'x', 'y')   # (nk, a, b, μ, μ)
		scalar_spec = P(None, 'x', 'y')             # (nk, μ, μ)
		spin_flat_shard = NamedSharding(mesh_xy, spin_spec)
		scalar_flat_shard = NamedSharding(mesh_xy, scalar_spec)
		# 3-D k-grid forms (the FFT helpers prepend (nkx,nky,nkz) → (nk,)).
		fft_spin_3d = P(None, None, None, None, None, 'x', 'y')
		fft_scalar_3d = P(None, None, None, 'x', 'y')
		local_ifftn_spin   = make_flat_k_ifftn(mesh_xy, kgrid, fft_spin_3d,   norm='forward')
		local_fftn_scalar  = make_flat_k_fftn( mesh_xy, kgrid, fft_scalar_3d, norm='forward')

		rep = NamedSharding(mesh_xy, P())

		@partial(jax.jit, in_shardings=spin_flat_shard, out_shardings=spin_flat_shard,
		         donate_argnums=(0,))
		def _ifft_conj(P_l: jax.Array) -> jax.Array:
			return jnp.conj(local_ifftn_spin(P_l))

		# Closed-over Python bools — compile-time branches in
		# gamma_double_contract via None-passthrough.
		_lhs_id = lhs_id
		_rhs_id = rhs_id

		@partial(jax.jit,
		         in_shardings=(spin_flat_shard, spin_flat_shard,
		                       rep, rep, rep, rep),
		         out_shardings=scalar_flat_shard,
		         donate_argnums=(0, 1))
		def _ifft_contract_fft(P_r, P_l_Rt_conj, perm_L_, phase_L_, perm_R_, phase_R_):
			P_r_Rt = local_ifftn_spin(P_r)
			C_Rt = gamma_double_contract(
				P_l_Rt_conj, P_r_Rt,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(1, 2),
			)
			return local_fftn_scalar(C_Rt)

		def _c_q_kernel(P_l, P_r, pL, phL, pR, phR):
			P_l_Rt_conj = _ifft_conj(P_l)
			return _ifft_contract_fft(P_r, P_l_Rt_conj, pL, phL, pR, phR)

		_isdf_pipeline_cache[cache_key] = _c_q_kernel

	# Identity-side perm/phase still passed (kernel ignores them via flags),
	# so the JIT signature stays uniform across charge / transverse paths.
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
		P_l_k_ab, P_r_k_ab, perm_L, phase_L, perm_R, phase_R)


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

	γ̃ identity short-circuit: same convention as :func:`c_q_from_pair` —
	pass ``gamma_L=None`` (and/or ``gamma_R=None``) for charge / left-only /
	right-only sides.  Both None → Σ_{αβ} P_v* · P_c, the historical
	pivoted-Cholesky candidate Gram in open-spin form.

	Compared to :func:`c_q_from_pair`, this drops the k→q FFT pair: at
	q=0 the k-sum IS the answer, no convolution is needed.  Used by
	:mod:`centroid.pivoted_cholesky`.

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


def z_q_from_pair(
	P_l_k_ab_muz: jax.Array,
	P_r_k_ab_muz: jax.Array,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	*,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""Z_q from open-spin pair densities on (μ, r-chunk) with optional γ̃.

	    Z_q^{μ_L, ν_L}(μ, r) =
	        FFT_k→q[ Σ_{αβα'β'} γ̃^{μ_L}_{αα'} γ̃^{ν_L}_{ββ'}
	                 · conj(IFFT(P_l_{αβ}))(R; μ, r)
	                 · IFFT(P_r_{α'β'})(R; μ, r) ]

	γ̃ identity short-circuit: same convention as :func:`c_q_from_pair`
	(``None`` = γ̃ = I_4 on that side).

	Inputs:
	    P_l_k_ab_muz, P_r_k_ab_muz : (nk, ns, ns, n_rmu, n_zchunk) c128
	                                 sharded ``P(None, None, None, 'x', 'y')``.
	    gamma_L, gamma_R           : ``(perm, phase)`` tuples or ``None``.
	    kgrid                      : (nkx, nky, nkz).
	    mesh_xy                    : 2-D device mesh.

	Output:
	    Z_q : (nq, n_rmu, n_zchunk) c128, sharded ``P(None, 'x', 'y')``.
	"""
	nkx, nky, nkz = kgrid
	nk, ns1, ns2, n_rmu, n_zchunk = P_l_k_ab_muz.shape
	assert nk == nkx * nky * nkz, (
		f"P_l_k_ab_muz flat-k dim {nk} does not match kgrid product {nkx*nky*nkz}"
	)
	assert P_r_k_ab_muz.shape == P_l_k_ab_muz.shape, (
		f"P_l/P_r shape mismatch: {P_l_k_ab_muz.shape} vs {P_r_k_ab_muz.shape}"
	)
	lhs_id = gamma_L is None
	rhs_id = gamma_R is None

	cache_key = ('z_q_from_pair', id(mesh_xy), nk, ns1, ns2, n_rmu, n_zchunk,
	             lhs_id, rhs_id)

	if cache_key not in _isdf_pipeline_cache:
		spin_spec = P(None, None, None, 'x', 'y')   # (nk, a, b, μ, z)
		scalar_spec = P(None, 'x', 'y')             # (nk, μ, z)
		spin_flat_shard = NamedSharding(mesh_xy, spin_spec)
		scalar_flat_shard = NamedSharding(mesh_xy, scalar_spec)
		spec_spin_3d = P(None, None, None, None, None, 'x', 'y')
		spec_scalar_3d = P(None, None, None, 'x', 'y')
		local_ifftn_spin   = make_flat_k_ifftn(mesh_xy, kgrid, spec_spin_3d,   norm='forward')
		local_fftn_scalar  = make_flat_k_fftn( mesh_xy, kgrid, spec_scalar_3d, norm='forward')
		rep = NamedSharding(mesh_xy, P())

		@partial(jax.jit, in_shardings=spin_flat_shard, out_shardings=spin_flat_shard,
		         donate_argnums=(0,))
		def _left_ifft_conj(P_l: jax.Array) -> jax.Array:
			return jnp.conj(local_ifftn_spin(P_l))

		_lhs_id = lhs_id
		_rhs_id = rhs_id

		@partial(jax.jit,
		         in_shardings=(spin_flat_shard, spin_flat_shard,
		                       rep, rep, rep, rep),
		         out_shardings=scalar_flat_shard,
		         donate_argnums=(0, 1))
		def _right_ifft_contract_fft(P_r, P_l_Rt_conj, perm_L_, phase_L_, perm_R_, phase_R_):
			P_r_Rt = local_ifftn_spin(P_r)
			Z_Rt = gamma_double_contract(
				P_l_Rt_conj, P_r_Rt,
				perm_L=None if _lhs_id else perm_L_,
				phase_L=None if _lhs_id else phase_L_,
				perm_R=None if _rhs_id else perm_R_,
				phase_R=None if _rhs_id else phase_R_,
				spin_axes=(1, 2),
			)
			return local_fftn_scalar(Z_Rt)

		def _z_q_kernel(P_l, P_r, pL, phL, pR, phR):
			P_l_Rt_conj = _left_ifft_conj(P_l)
			return _right_ifft_contract_fft(P_r, P_l_Rt_conj, pL, phL, pR, phR)

		_isdf_pipeline_cache[cache_key] = _z_q_kernel

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
		P_l_k_ab_muz, P_r_k_ab_muz, perm_L, phase_L, perm_R, phase_R)


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
# fused into ONE shard_map.  Default-on; override with
# ``LORRAX_PAIR_PIPELINE_SHARDMAP=0`` to fall back to the legacy chain
# (kept around because pivoted-Cholesky centroid selection still calls
# ``pair_density`` + ``c_q_from_pair`` directly outside this codepath).
# ============================================================================
#
# Legacy ``c_q_from_pair`` / ``z_q_from_pair`` chains three sub-jits
# (``_ifft_conj`` + ``_ifft_contract_fft``, with FFT helpers adding their
# own shard_map regions inside).  Between them XLA materialises a global
# rank-5 pair-density value, and for each pair-density-shaped value
# across the lifetime — pair-density einsum output, IFFT outputs (P_l_R,
# P_r_R), gamma-contract take intermediate — XLA's BufferAssignment
# lands at 5 concurrent rank-5 lifetime slots (~4 GiB each at MoS2 3×3
# bispinor: ``nk · ns² · n_rmu_local · col_local · 16``), pegging the
# kernel preallocated-temp peak at 21 GiB on a 28 GiB budget.
#
# These ``_sm`` variants take the wavefunction tensors directly
# (skipping the standalone rank-5 pair-density buffer) and fold the
# entire pair-density → IFFT → γ̃·γ̃ → FFT pipeline into one
# ``shard_map``.  Inside that region everything is local-per-rank and
# the FFTs run via direct ``jnp.fft.ifftn`` / ``jnp.fft.fftn`` calls
# (no nested ``make_flat_k_*`` helper — same approach as
# ``wfn_transforms.to_rchunk``).  No
# nested shard_maps, no helper boundary that could let XLA re-globalise
# the pair density.  Drops the slot count from 5 → 3 (the two saved are
# the standalone IFFT outputs + the gamma-contract intermediate),
# reducing the kernel peak from 21.45 GiB → 13.11 GiB on the MoS2
# bispinor reference (a ~40% drop) with comparable wall time.


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
	"""``c_q_from_pair`` from psi inputs directly, inside one shard_map.

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
	psi_l_Y: jax.Array,
	psi_r_X: jax.Array,
	psi_r_Y: jax.Array,
	gamma_L: tuple[jax.Array, jax.Array] | None = None,
	gamma_R: tuple[jax.Array, jax.Array] | None = None,
	*,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""``z_q_from_pair`` from psi inputs directly, inside one shard_map.

	Identical structure to :func:`c_q_from_psi_sm` but ``n_col`` is the
	r-chunk extent (not n_rmu), and the L / R bands may have different
	extents (``nb_l != nb_r`` in the general fit_one_rchunk case).
	"""
	nkx, nky, nkz = kgrid
	nk = int(psi_l_X.shape[0])
	n_rmu = int(psi_l_X.shape[1])
	nb_l = int(psi_l_X.shape[2])
	nb_r = int(psi_r_X.shape[2])
	ns = int(psi_l_X.shape[3])
	n_zchunk = int(psi_l_Y.shape[3])
	lhs_id = gamma_L is None
	rhs_id = gamma_R is None

	cache_key = ('z_q_from_psi_sm', id(mesh_xy), nk, n_rmu, n_zchunk, ns,
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
			# Einsum spec ``'karmb'`` matches cuBLAS's natural gemm
			# output factoring ``(k, ns_l, col, μ, ns_r)`` so the
			# rank-7 reshape is a bitcast.  See c_q_from_psi_sm._local
			# for full commentary.  ``col`` here is the r-chunk extent.
			mu_loc = psi_l_X_.shape[1]
			z_loc = psi_l_Y_.shape[3]
			P_l = jnp.einsum(
				'kmna,knbr->karmb', psi_l_X_, psi_l_Y_, optimize=True)
			P_r = jnp.einsum(
				'kmna,knbr->karmb', psi_r_X_, psi_r_Y_, optimize=True)
			P_l_3d = P_l.reshape(nkx, nky, nkz, ns, z_loc, mu_loc, ns)
			del P_l
			P_l_R = jnp.fft.ifftn(P_l_3d, axes=(0, 1, 2), norm='forward')
			P_l_R_conj = jnp.conj(P_l_R)
			del P_l_3d, P_l_R
			P_r_3d = P_r.reshape(nkx, nky, nkz, ns, z_loc, mu_loc, ns)
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
				Z_q_3d.reshape(nkx * nky * nkz, z_loc, mu_loc),
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

    Why this matters:  Cholesky and LU on the identity-padded matrix
    produce factorisations whose **logical block is bit-identical** to
    the factorisation of the un-padded logical-only matrix.  The
    standard recursion for ``L[i, j]`` (Cholesky) and the column-
    pivoting LU never read across the zero off-diagonal pad blocks,
    and ``√1 = 1`` exactly in IEEE 754 so the pad-block factor is
    exactly identity.  The downstream back-solve sees ``Z`` with zero
    pad rows (bilinear in zero-padded ψ ⇒ zero pad rows), and
    ``[L_log 0; 0 I][y_log; y_pad] = [Z_log; 0]`` gives
    ``y_pad = 0`` and ``y_log = L_log⁻¹ Z_log`` — logical solve
    unchanged.

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


def _resolve_solver_kind_charge(mesh_xy: Mesh) -> str:
    """Pick the charge-channel ζ-fit solver: cuSolverMp distributed
    potrf+potrs vs the in-tree shard_map 2D-blocked Cholesky + per-q
    triangular solve.

    Default policy (2026-05-12): use cuSolverMp on **true 2D meshes**
    (px≥2 AND py≥2); the in-tree ``sharded_cholesky`` runs many small
    NCCL all-reduces per panel which become the dominant GPU stream
    consumer at production scale (CrI3 6×6 80 Ry n_rmu≈1500 sees ~tens
    of seconds in the panel loop).  cuSolverMp bundles the whole
    distributed Cholesky into one FFI call per q.

    Tradeoff: at small scales (MoS2 3×3, n_rmu=640, 2×2 mesh) the FFI
    setup overhead exceeds the savings — measured cholesky 3.6 s
    (cuSolverMp) vs 1.3 s (sharded) on a 2×2 mesh.  Total wall
    difference ~0.5 s, within run-to-run noise.  We accept the small
    overhead for the larger-scale win.

    Override via env var:
      ``LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR=0`` → force sharded.
      ``LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR=1`` → force cuSolverMp
        (still falls back on 1D meshes).
    Unset → default policy (cuSolverMp on true 2D, sharded otherwise).
    """
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    is_2d = (px >= 2 and py >= 2)

    env = os.environ.get('LORRAX_USE_CUSOLVERMP_CHARGE_FACTOR')
    if env == '0':
        return 'sharded_cholesky'
    if env == '1':
        return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'

    # Unset → default policy.
    return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'


def _resolve_solver_kind_transverse(mesh_xy: Mesh) -> str:
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

    Override via env var:
      ``LORRAX_USE_CUSOLVERMP_LU=0`` → force per-q ``jnp.linalg.solve``.
      ``LORRAX_USE_CUSOLVERMP_LU=1`` → force cuSolverMp (still falls
        back on 1D meshes).
    Unset → default policy (cuSolverMp on true 2D, legacy otherwise).
    """
    px = int(mesh_xy.shape['x'])
    py = int(mesh_xy.shape['y'])
    is_2d = (px >= 2 and py >= 2)

    env = os.environ.get('LORRAX_USE_CUSOLVERMP_LU')
    if env == '0':
        return 'lu'
    if env == '1':
        return 'cusolvermp_lu' if is_2d else 'lu'

    return 'cusolvermp_lu' if is_2d else 'lu'


def _resolve_solver_kind(mesh_xy: Mesh, vertex_mu_L: int, solver_kind: str) -> str:
    """Single source of truth for the ``auto`` resolution.  Transverse
    channels (γ̃^i, μ_L≠0) take ``_resolve_solver_kind_transverse``;
    charge channel takes ``_resolve_solver_kind_charge``.
    """
    if solver_kind != 'auto':
        return solver_kind
    if int(vertex_mu_L) != 0:
        return _resolve_solver_kind_transverse(mesh_xy)
    return _resolve_solver_kind_charge(mesh_xy)


def factor_c_q(
    C_q: jax.Array,
    mesh_xy: Mesh,
    block_size: int = None,
    vertex_mu_L: int = 0,
    n_rmu_logical: int | None = None,
    solver_kind: str = 'auto',
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
    Cholesky / LU of an identity-padded matrix produces a factor
    whose logical block is bit-identical to the factor of the
    logical-only matrix (the recursion never reads across zero
    off-diagonal pad blocks, and ``√1 = 1`` exactly).  The pad-block
    factor is exactly identity; the back-solve's pad rows of ζ come
    out as zero (because Z's pad rows are zero by the same bilinear
    argument); logical block of ζ is unchanged.

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

    Returns:
        L_q: ``(nq, n_rmu, n_rmu)`` at PADDED extent, sharded
        ``P(None, 'x', 'y')``.  Cholesky factor for
        ``vertex_mu_L == 0`` (block-diagonal ``[L_log 0; 0 I_pad]``
        when n_rmu_logical < n_rmu); passthrough identity-padded CCT
        for ``vertex_mu_L ≠ 0``.
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

    if solver_kind == 'cusolvermp_cholesky':
        # Distributed potrf on C_q at P(None,'x','y'); returns the raw
        # lower-triangular factor.  Downstream solve_zeta rebuilds the
        # CusolverMpBatchedLowerL handle and dispatches to potrs.
        from ffi.cusolvermp import batched_distributed_cholesky
        L_handle = batched_distributed_cholesky(C_q, mesh=mesh_xy)
        return L_handle.raw

    Pr = mesh_xy.shape['x']
    Pc = mesh_xy.shape['y']

    # On 1x1 meshes, JAX 0.9 can fail in the shard_map+scan blocked kernel with:
    # "scan body function carry input and carry output must have equal types ...
    # varying manual axes do not match". Use dense batched Cholesky in this case.
    if mesh_xy.devices.size == 1 or (Pr == 1 and Pc == 1):
        # Regularize C_q: the pair density matrix can be numerically
        # rank-deficient (more centroids than band pairs), producing
        # tiny negative eigenvalues that break Cholesky. Add a small
        # ridge proportional to the trace to ensure positive definiteness.
        trace_per_q = jnp.trace(C_q, axis1=-2, axis2=-1)
        ridge = 1e-14 * jnp.abs(trace_per_q)[:, None, None] * jnp.eye(n_rmu)[None, :, :]
        C_q_reg = C_q + ridge
        L_q_dense = jnp.linalg.cholesky(C_q_reg)
        L_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        return jax.lax.with_sharding_constraint(L_q_dense, L_shard)

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
_solve_cache = {}


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
                     explicit values are 'sharded_cholesky' (legacy 2D
                     blocked chol + per-q triangular solve), 'lu' (per-q
                     pivoted-LU for transverse channels),
                     'cusolvermp_cholesky' (distributed potrs via FFI),
                     or 'cusolvermp_lu' (distributed getrf+getrs via FFI
                     for the transverse channels).

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

    solver_kind = _resolve_solver_kind(mesh_xy, vertex_mu_L, solver_kind)

    if solver_kind == 'cusolvermp_cholesky':
        # Distributed potrs: Z stays at P(None,'x','y'), no input reshard
        # and no all-gather of L.  Output reshards to P(None, ('x','y'),
        # None) so the downstream G-flat accumulator receives ζ
        # μ-flat-sharded (the FFT layout it actually wants — see
        # accumulate_rchunk_to_gflat / common.wfn_transforms).
        from ffi.cusolvermp import batched_distributed_potrs, CusolverMpBatchedLowerL
        Px = int(mesh_xy.shape['x'])
        Py = int(mesh_xy.shape['y'])
        # potrs requires Z's last dim divisible by Py; zero-pad columns
        # produce zero ζ columns and get trimmed on return.
        n_zchunk_padded = ((n_zchunk + Py - 1) // Py) * Py
        needs_padding = (n_zchunk_padded != n_zchunk)
        if needs_padding:
            Z_q = jnp.pad(Z_q, ((0, 0), (0, 0), (0, n_zchunk_padded - n_zchunk)),
                          mode='constant')
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

    if solver_kind == 'cusolvermp_lu':
        # Distributed getrf+getrs for the transverse channels.  L_q here
        # is the *unfactored* CCT^μ (Hermitian indefinite) — factor_c_q
        # passes it through.  Same input sharding, output reshard, and
        # column padding pattern as the cholesky branch.
        from ffi.cusolvermp import batched_distributed_solve_lu
        Px = int(mesh_xy.shape['x'])
        Py = int(mesh_xy.shape['y'])
        # getrs descB requires NRHS % Py == 0; zero-pad columns produce
        # zero ζ columns and get trimmed on return.
        n_zchunk_padded = ((n_zchunk + Py - 1) // Py) * Py
        needs_padding = (n_zchunk_padded != n_zchunk)
        if needs_padding:
            Z_q = jnp.pad(Z_q, ((0, 0), (0, 0), (0, n_zchunk_padded - n_zchunk)),
                          mode='constant')
        # Per-q ridge ε·|tr(L)|/n_rmu — same lift as the legacy 'lu'
        # branch, to keep TRS-paired near-zero modes above the LU
        # stability floor without perturbing well-conditioned ones.
        # ``cct_trace_per_q`` is precomputed once per channel by the
        # caller (fit_zeta_to_h5) — the trace doesn't change across
        # r-chunks since L_q is the per-channel CCT.  Computing it
        # inline here re-fires an all-reduce across the (μ_X, ν_Y)
        # sharding on every r-chunk: ~17 s GPU stream time on MoS2
        # 3×3 bispinor at our default chunk count.
        LU_RIDGE = 1e-12
        trace_per_q = (cct_trace_per_q if cct_trace_per_q is not None
                       else jnp.einsum('qii->q', L_q))
        ridge = (LU_RIDGE * jnp.abs(trace_per_q) / n_rmu)[:, None, None]
        eye_n = jnp.eye(n_rmu, dtype=L_q.dtype)[None, :, :]
        A_q = L_q + ridge * eye_n
        zeta_xy = batched_distributed_solve_lu(A_q, Z_q, mesh=mesh_xy)
        # Reshard rationale identical to the cholesky branch above.
        zeta_out = _reshard_zeta_mu_X_r_Y_to_mu_XY(zeta_xy, mesh_xy)
        if needs_padding:
            return zeta_out[:, :, :n_zchunk]
        return zeta_out

    # Compute padding needed for even sharding across all devices
    total_devices = mesh_xy.devices.size
    n_zchunk_padded = ((n_zchunk + total_devices - 1) // total_devices) * total_devices
    needs_padding = n_zchunk_padded != n_zchunk

    z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    L_rep_shard = NamedSharding(mesh_xy, P(None, None))
    L_batch_rep_shard = NamedSharding(mesh_xy, P(None, None, None))  # (B_q, n_rmu, n_rmu)
    q_batch = min(q_chunk_size, nq)
    nq_padded = ((nq + q_batch - 1) // q_batch) * q_batch

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

    # Cache key for solve function (includes q_chunk_size and padded size).
    # ``use_lu`` partitions the cache so the Cholesky and LU compiles
    # don't collide on the same key.
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu,
                 n_zchunk_padded, q_chunk_size, bool(use_lu))

    def _ridge_indef_solve(L: jax.Array, Z: jax.Array) -> jax.Array:
        """Solve (L + ε·tr(L)/n · I) · ζ = Z via pivoted LU.

        ε = ``LU_RIDGE`` (1e-12).  The shift sits well below any
        physically meaningful eigenvalue but well above the partial-
        pivoting floor, so LU stays stable on TRS-paired near-zero
        modes without perturbing the rest of the spectrum.
        """
        n = L.shape[-1]
        ridge = LU_RIDGE * jnp.abs(jnp.trace(L)) / n
        L_reg = L + ridge * jnp.eye(n, dtype=L.dtype)
        return jnp.linalg.solve(L_reg, Z)

    if cache_key not in _solve_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            if use_lu:
                # Indefinite CCT^μ: pivoted-LU back-solve with ridge.
                return _ridge_indef_solve(L, Z_cols)
            y = jax.scipy.linalg.solve_triangular(L, Z_cols, lower=True)
            zeta = jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
            return zeta

        # Vectorized solve for a batch of q-points
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None, None), P(None, None, ('x', 'y'))),
                 out_specs=P(None, None, ('x', 'y')))
        def _sharded_cho_solve_batch(L_batch: jax.Array, Z_batch: jax.Array) -> jax.Array:
            """Solve (B_q, n_rmu, n_rmu) @ (B_q, n_rmu, n_cols) -> (B_q, n_rmu, n_cols)"""
            if use_lu:
                # ``jnp.linalg.solve`` is natively batched on the leading
                # axis and dispatches one LU factorization per q.  Same
                # vmap structure as the Cholesky path so reshard plans
                # match.  We vmap the ridge-add per-q so each LU sees
                # its own conditioning shift.
                return jax.vmap(_ridge_indef_solve)(L_batch, Z_batch)
            def solve_single(L, Z):
                y = jax.scipy.linalg.solve_triangular(L, Z, lower=True)
                return jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
            return jax.vmap(solve_single)(L_batch, Z_batch)

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

_fit_one_rchunk_cache: dict = {}


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

    l_lo, l_hi = band_range_left
    r_lo, r_hi = band_range_right

    P_sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    L_slice_shard = NamedSharding(mesh_xy, P(None, 'x', None, None))

    # Classify each band-chunk against the L and R endpoint ranges at
    # trace time.  Status codes (``'skip' | 'direct' | 'pad'``) dispatch
    # the Python if-branch inside _kernel below so the jit body only
    # emits the ops that matter for this bc.
    #
    #   skip   — bc_range is entirely outside [rs, re); the contribution
    #            is dropped at trace (no ops emitted).
    #   direct — bc_range ⊆ [rs, re); straight slice of the centroid
    #            copy, no zero-padding needed.
    #   pad    — bc_range straddles an endpoint; slot the overlapping
    #            bands into a zero-padded bc_size-wide buffer so the
    #            pair-density einsum still hits the uniform jit cache.
    def _classify(rs, re, bc_lo, bc_hi):
        ol_lo = max(bc_lo, rs)
        ol_hi = min(bc_hi, re)
        if ol_hi <= ol_lo:
            return 'skip', None
        if ol_lo == bc_lo and ol_hi == bc_hi:
            return 'direct', (ol_lo - rs, ol_hi - rs)
        return 'pad', (ol_lo - rs, ol_hi - rs, ol_lo - bc_lo, ol_hi - bc_lo)

    bc_classify = [
        (
            bc_range,
            _classify(l_lo, l_hi, bc_range[0], bc_range[1]),
            _classify(r_lo, r_hi, bc_range[0], bc_range[1]),
        )
        for bc_range in band_chunk_ranges
    ]

    # Closure-side IBZ row indices (Phase B).  None → full-BZ solve
    # (back-compat for ``write_ibz_only=False``).  Otherwise a static
    # jax int32 array baked into the jit's HLO.
    if q_irr_full_idx is not None:
        q_irr_idx_j = jnp.asarray(
            np.asarray(q_irr_full_idx, dtype=np.int32))
    else:
        q_irr_idx_j = None

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
        # ψ enters at PADDED n_rmu (load_centroids_band_chunked output,
        # Phase 3a).  All in-memory arrays here — P_l, P_r, ψ_L_bc,
        # ψ_R_bc, Z_q — operate at PADDED extent so the inner
        # ``accumulate_pair_density_*`` / ``compute_ZCT`` /
        # ``solve_zeta`` jits' committed
        # ``in_shardings=P(None, 'x', 'y')`` boundaries see divisible
        # shapes (n_rmu_padded ≡ ∏ p_a is divisible by every relevant
        # mesh axis).  The Cholesky factor L_q comes in at the same
        # PADDED extent — ``factor_c_q`` runs the chol on
        # the LOGICAL block and embeds the factor into a padded matrix
        # with identity in the pad block (see ``_embed_logical_in_padded``);
        # the back-solve produces zeta with zero in pad rows, logical
        # block byte-identical to a pure-logical solve.  zeta is
        # returned at padded extent; the SlabIO write uses
        # ``valid_shape=meta.n_rmu`` so on-disk extent stays logical.

        # --- 1. Pair-density accumulators (one r-chunk wide) ---
        # Open-spin rank-5 accumulator for ALL channels (charge γ̃^0=I and
        # transverse γ̃^i=α^i).  Shape (nk, ns, ns, n_rmu, n_zchunk),
        # sharding P(None, None, None, 'x', 'y').  γ̃^μ_L (or identity for
        # charge) is applied at the Z_q post-IFFT contraction step inside
        # ``z_q_from_pair``.
        P_acc_shape = (nk_tot, nspinor, nspinor, n_rmu, actual_n_rchunk)
        P_acc_sharding = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))

        def _zero_P():
            return jax.lax.with_sharding_constraint(
                jnp.zeros(P_acc_shape, dtype=jnp.complex128),
                P_acc_sharding)
        P_l = _zero_P()
        P_r = _zero_P()

        # Pair-density bc-contribution — shared for the L and R sides.
        # ``cls`` is the _classify tuple for THIS side; ``psi_centroid``
        # is psi_l_rmuT_X_fit (for L) or psi_r_rmuT_X_fit (for R);
        # ``norms`` is the matching clamped weights.  Returns the
        # updated accumulator (P_l or P_r).
        def _accumulate(P_acc, cls, psi_centroid, norms, psi_bc_Y, bc_size):
            tag, payload = cls
            if tag == 'skip':
                return P_acc
            if tag == 'direct':
                lo, hi = payload
                psi_L_bc = psi_centroid[:, :, lo:hi, :]
                norm_slice = norms[lo:hi]
            else:  # 'pad' — straddles an endpoint
                cen_lo, cen_hi, dst_lo, dst_hi = payload
                psi_L_bc = jax.lax.with_sharding_constraint(
                    jnp.zeros((nk_tot, n_rmu, bc_size, nspinor),
                              dtype=jnp.complex128),
                    L_slice_shard)
                psi_L_bc = psi_L_bc.at[:, :, dst_lo:dst_hi, :].set(
                    psi_centroid[:, :, cen_lo:cen_hi, :])
                norm_slice = jnp.ones((bc_size,), dtype=jnp.float64
                                      ).at[dst_lo:dst_hi].set(
                    norms[cen_lo:cen_hi])
            psi_R_bc = psi_bc_Y / norm_slice[None, :, None, None]
            # γ̃^μ_L (charge: identity) deferred to z_q_from_pair below.
            return accum_pair_density(
                P_acc, psi_L_bc, psi_R_bc, mesh_xy)

        # --- 2. Stream band-chunks: fetch ψ(G) per-bc from host via
        #       io_callback, FFT + reshard, accumulate pair density.
        #       Each bc's ψ(G) is live only during its own iteration.
        # When ``LORRAX_PAIR_PIPELINE_SHARDMAP=1`` we skip the rank-5
        # P_l/P_r accumulator entirely: fetch all bcs, concatenate
        # ψ_Y along the band axis, slice into L / R band ranges, and
        # call ``z_q_from_psi_sm`` which fuses pair density + IFFT +
        # γ̃·γ̃ + FFT inside one monolithic shard_map.  The rank-5 pair
        # density never exists as a global XLA value, so the rank-3
        # fused-replicated buffer that pegs the kernel peak (5 slots ×
        # 4.16 GiB on MoS2 3×3 bispinor) cannot form.
        _use_pair_pipe_sm = (
            os.environ.get('LORRAX_PAIR_PIPELINE_SHARDMAP', '1') != '0')
        if _use_pair_pipe_sm:
            del P_l, P_r  # zeros no longer needed
            _b0 = int(band_range_full[0])
            _l_lo = int(band_range_left[0]) - _b0
            _l_hi = int(band_range_left[1]) - _b0
            _r_lo = int(band_range_right[0]) - _b0
            _r_hi = int(band_range_right[1]) - _b0
            psi_Y_parts = []
            for bc_idx, (bc_range, _l_cls, _r_cls) in enumerate(bc_classify):
                psi_Y_parts.append(psi_G_store.fetch_psi_rchunk(
                    bc_range, r_start_dyn, actual_n_rchunk))
            psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)
            del psi_Y_parts
            # Per-side band slices + norm divide.  norms_l / norms_r
            # are sized to the side's band range, so they apply 1:1 to
            # the corresponding slice of psi_Y_full.
            psi_l_Y_sm = (psi_Y_full[:, _l_lo:_l_hi, :, :]
                          / norms_l[None, :, None, None])
            psi_r_Y_sm = (psi_Y_full[:, _r_lo:_r_hi, :, :]
                          / norms_r[None, :, None, None])
            del psi_Y_full
            if is_charge:
                Z_q = z_q_from_psi_sm(
                    psi_l_rmuT_X_fit, psi_l_Y_sm,
                    psi_r_rmuT_X_fit, psi_r_Y_sm,
                    kgrid=kgrid, mesh_xy=mesh_xy)
            else:
                gamma_mu = (gamma_perm, gamma_phase)
                Z_q = z_q_from_psi_sm(
                    psi_l_rmuT_X_fit, psi_l_Y_sm,
                    psi_r_rmuT_X_fit, psi_r_Y_sm,
                    gamma_mu, gamma_mu,
                    kgrid=kgrid, mesh_xy=mesh_xy)
        else:
            for bc_idx, (bc_range, l_cls, r_cls) in enumerate(bc_classify):
                bc_size = bc_range[1] - bc_range[0]
                # ψ(r-chunk) directly via g_flat host cache + on-device
                # to_rchunk (FFT box never materialised as a persistent
                # buffer).  Bloch phase + kvecs lookup happen inside
                # ``fetch_psi_rchunk`` — caller's ``kvecs_frac`` arg is
                # no longer needed at this site.
                psi_bc_Y = psi_G_store.fetch_psi_rchunk(
                    bc_range, r_start_dyn, actual_n_rchunk)
                P_l = _accumulate(P_l, l_cls, psi_l_rmuT_X_fit,
                                  norms_l, psi_bc_Y, bc_size)
                P_r = _accumulate(P_r, r_cls, psi_r_rmuT_X_fit,
                                  norms_r, psi_bc_Y, bc_size)

            # 3. ZCT + solve.  Pass Z_q UN-RESHARDED — the inline
            # ``with_sharding_constraint`` here was inside this outer kernel
            # jit and tied XLA's hands on the two-step reshard (P('x',None,'y')
            # intermediate cancels with all-to-all chains across consumer
            # ops in the fused trace, forcing Involuntary Full Rematerialization
            # of the full (nq, μ, ν) tensor).  Letting solve_zeta's
            # sub-jit boundary handle the reshard (with its own two-step staging)
            # decouples the reshard scheduler from the kernel body and matches
            # the load_wfns separate-jit pattern.  Same pattern as lorrax_B
            # commit c0307a0.
            # γ̃^μ_L applied on BOTH P-sides at the post-IFFT contraction
            # step (γ̃·γ̃ reduce).  ``is_charge`` is a closure bool — Python
            # if resolved at trace time — so charge and transverse get
            # distinct HLO branches, but all three transverse channels share
            # one compile (gamma_perm/gamma_phase are runtime inputs, baked
            # nowhere into the HLO).
            if is_charge:
                Z_q = z_q_from_pair(P_l, P_r, kgrid=kgrid, mesh_xy=mesh_xy)
            else:
                gamma_mu = (gamma_perm, gamma_phase)
                Z_q = z_q_from_pair(P_l, P_r, gamma_mu, gamma_mu,
                                    kgrid=kgrid, mesh_xy=mesh_xy)
        # IBZ-only solve (Phase B of PLAN_zeta_g_flat_migration.md):
        # the FFT-built C_q and Z_q are naturally full-BZ, but the
        # triangular solve has no inter-q coupling, so we gather IBZ
        # rows of L_q and Z_q here and solve only those.  Output
        # ``zeta`` is (n_q_ibz, n_rmu, n_rchunk) — caller writes it
        # directly with no post-solve slice.
        #
        # When ``q_irr_full_idx`` is None (write_ibz_only=False
        # codepath; centroid orbit closure failed) the gather is a
        # no-op identity and the full-BZ solve runs as before.
        if q_irr_idx_j is not None:
            L_q_for_solve = L_q[q_irr_idx_j]
            Z_q_for_solve = Z_q[q_irr_idx_j]
            cct_trace_for_solve = (
                cct_trace_per_q[q_irr_idx_j]
                if cct_trace_per_q is not None else None)
        else:
            L_q_for_solve = L_q
            Z_q_for_solve = Z_q
            cct_trace_for_solve = cct_trace_per_q
        # ``solver_kind`` is resolved upstream (sharded_cholesky /
        # cusolvermp_cholesky for charge; lu / cusolvermp_lu for
        # transverse).  solve_zeta dispatches purely on solver_kind.
        # ``cct_trace_for_solve`` is precomputed once per channel in
        # fit_zeta_to_h5 (the all-reduce on L_q[q,i,i] is invariant
        # across r-chunks; computing it here would refire the
        # ~5–17 s of GPU stream time per channel that we measured).
        zeta = solve_zeta(
            L_q_for_solve, Z_q_for_solve, mesh_xy, q_chunk_size,
            solver_kind=solver_kind,
            cct_trace_per_q=cct_trace_for_solve)
        return zeta

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
    the compiled kernel calls its ``fetch_psi_G`` method via io_callback
    inside the bc-loop.  The cache key includes ``id(psi_G_store)`` to
    avoid reusing a compile built against a different store.
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
        meta.n_rmu, meta.nk_tot, meta.nspinor,
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
    # uniform across channels.
    if cct_trace_per_q is None:
        cct_trace_per_q = jnp.zeros((int(meta.nk_tot),),
                                    dtype=jnp.complex128)
    return fn(
        psi_l_rmuT_X_fit,
        psi_r_rmuT_X_fit,
        L_q,
        norms_l,
        norms_r,
        r_start_dyn,
        gamma_perm,
        gamma_phase,
        cct_trace_per_q,
    )


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
# :func:`common.psi_G_store.PsiGStore.fetch_psi_G`.  See that module
# for the two lifecycle modes (host_cache vs file_reread).


def fit_zeta_to_h5(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    mesh_xy: Mesh,
    chunk_r: int,
    output_file: str,
    psi_rmu_Y: jax.Array,
    psi_rmuT_X: jax.Array,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    k_chunk_size: int = 0,
    q_gather_size: int = 0,
    band_norms: np.ndarray | None = None,
    slab_io_backend=None,
    gspace_mode: str = "host_cache",
    vertex_mu_L: int = 0,
    solver_kind: str = 'auto',
    write_ibz_only: bool = True,
    zeta_cutoff_ry: float | None = None,
):
    """
    Full zeta fitting pipeline with r-chunk loop and HDF5 output.

    For ``vertex_mu_L == 0`` (default) this is the standard spin-traced
    path used by the charge-channel ISDF fit — bit-identical to the
    pre-bispinor implementation.  For ``vertex_mu_L ∈ {1, 2, 3}`` the
    pair-density helpers contract through the Lorentz vertex γ̃^{μ_L}
    instead of the identity, and ``factor_c_q`` /
    ``solve_zeta`` switch from Cholesky to LU because the
    transverse-channel CCT is indefinite.  See
    ``docs/BISPINOR_DHFB_DESIGN.md`` for the math.

    Workflow:
    1. Slice pre-loaded centroid wavefunctions into left/right halves.
    2. Compute C_q from left/right pair density via FFT.
    3. Compute L_q = chol(C_q) using 2D blocked algorithm.
    4. For each r-chunk:
       a. Compute psi_nk,a(r_chunk) via FFT
       b. Compute left/right pair densities at r-chunk
       c. Compute Z_q via ortho FFT with left/right cross-product
       d. Solve zeta_q = L^{-H}(L^{-1} Z_q) (q-chunked)
       e. Write zeta_q chunk to HDF5

    Args:
        wfn: WFNReader object
        sym: SymMaps object
        meta: Meta object with system info
        centroid_indices: ISDF centroid indices
        mesh_xy: 2D device mesh
        chunk_r: Number of flattened r-points per chunk
        output_file: Path to output HDF5 file
        psi_rmu_Y:  Centroid wavefunctions for the full [b0, b4) band range,
                    shape (nk, nb_full, ns, n_rmu), P(None, None, None, 'y'),
                    un-conjugated ψ.  Produced by
                    :func:`common.load_wfns.load_centroids_band_chunked`.
        psi_rmuT_X: Same centroid data transposed/sharded for the pair-density
                    kernel, shape (nk, n_rmu, nb_full, ns),
                    P(None, 'x', None, None), conjugated ψ*.
        band_chunk_size: Bands to process at once when FFTing wavefunctions (with global r)
        q_chunk_size: Q-points to solve C_q @ zeta_q = Z_q simultaneously
        bispinor: Whether to use bispinor wavefunctions
        gspace_mode: ``"host_cache"`` (default) loads all ψ(G) band-chunks
                     into host RAM once at startup and pulls per-bc shards
                     into the jit via io_callback.  ``"file_reread"`` drops
                     the host cache between r-chunks and re-reads via
                     phdf5 collective I/O.  In both modes the jit never
                     holds more than one bc's ψ(G) on device.
        band_range_left: (start, end) for left wfns. Default: (b0, b3)
        band_range_right: (start, end) for right wfns. Default: (b0, b4)

    Returns:
        peak_bytes:  GPU high-water mark (peak_bytes_in_use) during chunk loop

    The centroid wavefunctions are inputs, not outputs — the caller is
    expected to hold the single ``load_centroids_band_chunked`` result and
    reuse it for :func:`gw.wavefunction_bundle.build_wavefunctions` after
    the fit completes.
    """
    from gw.gw_config import SlabIOBackend
    if slab_io_backend is None:
        slab_io_backend = SlabIOBackend.H5PY_ALLGATHER
    use_ffi_io = (slab_io_backend is SlabIOBackend.PHDF5_FFI)
    import h5py

    nx, ny, nz = meta.fft_grid
    # Two μ extents flow through this function (see common/meta.py:38):
    # ``n_rmu`` is the LOGICAL centroid count from the centroid file;
    # ``n_rmu_padded`` rounds up to ``world_size = ∏ p_a`` so any
    # single- or product-axis sharding on the μ dim divides cleanly.
    # ψ is delivered at PADDED extent by ``load_centroids_band_chunked``
    # (Phase 3a) — pad rows zero — and stays there through the
    # in-memory pair-density / CCT chain.  The Cholesky in
    # ``factor_c_q`` slices internally to logical via the
    # ``n_rmu_logical=`` kwarg (Phase 3b-Cholesky) so the factorization
    # sees a non-singular matrix at its true extent.  zeta_q on disk
    # has logical extent (SlabIO ``valid_shape=`` clips the padded
    # output before write).
    n_rmu = meta.n_rmu                      # logical
    n_rmu_padded = meta.n_rmu_padded        # padded
    n_rtot = meta.n_rtot
    nk_tot = meta.nk_tot
    kgrid = meta.kgrid
    nqx, nqy, nqz = kgrid
    nq = nqx * nqy * nqz

    num_chunks = (n_rtot + chunk_r - 1) // chunk_r
    n_rchunk = chunk_r

    # Band ranges for left and right wavefunctions.
    # Defaults here are (b0,b3) and (b0,b4); gw_jax typically passes (b0,b3) and (b1,b4).
    if band_range_left is None:
        band_range_left = (meta.b_id_0, meta.b_id_3)
    if band_range_right is None:
        band_range_right = (meta.b_id_0, meta.b_id_4)

    # Full range for loading (max of left and right)
    band_range_full = (min(band_range_left[0], band_range_right[0]),
                       max(band_range_left[1], band_range_right[1]))

    nb_left = band_range_left[1] - band_range_left[0]
    nb_right = band_range_right[1] - band_range_right[0]
    nb_full = band_range_full[1] - band_range_full[0]

    print(f"\n  Zeta fitting: {num_chunks} r-chunks x {n_rchunk} r-points, "
          f"{nb_full} bands ({nb_left} left + {nb_right} right)")
    print(f"  Output: {output_file}")

    # ========== STEP 1: Slice pre-loaded centroid ψ into left/right halves ==========
    with timing.section("zeta_fit.slice_halves"):
        # Band range arithmetic — left/right are sub-ranges of [b0, b4).
        l_band_start = band_range_left[0] - band_range_full[0]
        l_band_end = l_band_start + nb_left
        r_band_start = band_range_right[0] - band_range_full[0]
        r_band_end = r_band_start + nb_right

        # Cheap views — the caller keeps the full arrays alive for the
        # post-fit wfn bundle build, so we don't need independent copies.
        psi_l_rmu_Y = psi_rmu_Y[:, l_band_start:l_band_end, :, :]
        psi_l_rmuT_X = psi_rmuT_X[:, :, l_band_start:l_band_end, :]
        psi_r_rmu_Y = psi_rmu_Y[:, r_band_start:r_band_end, :, :]
        psi_r_rmuT_X = psi_rmuT_X[:, :, r_band_start:r_band_end, :]

        print(f"  Left wfns:  {psi_l_rmu_Y.shape}")
        print(f"  Right wfns: {psi_r_rmu_Y.shape}")

        # Pseudobands: clamp weights to ``max(1, w_n)`` and apply them to
        # the centroid copies used for CCT.  When band_norms is None the
        # slices are jnp.ones → the *_fit aliases are identical to the
        # *_rmu_Y / *_rmuT_X copies.  See _band_norms_slice for the why.
        norms_l_jax = _band_norms_slice(band_norms, band_range_left, nb_left)
        norms_r_jax = _band_norms_slice(band_norms, band_range_right, nb_right)
        # psi shapes: Y=(nk, nb, ns, n_rmu), X=(nk, n_rmu, nb, ns)
        psi_l_rmu_Y_fit = psi_l_rmu_Y / norms_l_jax[None, :, None, None]
        psi_l_rmuT_X_fit = psi_l_rmuT_X / norms_l_jax[None, None, :, None]
        psi_r_rmu_Y_fit = psi_r_rmu_Y / norms_r_jax[None, :, None, None]
        psi_r_rmuT_X_fit = psi_r_rmuT_X / norms_r_jax[None, None, :, None]
        if band_norms is not None:
            n_weighted = int(np.sum(band_norms > 1.01))
            n_zero = int(np.sum(band_norms < 1e-10))
            print(f"  Pseudobands normalization: {n_weighted} weighted, "
                  f"{n_zero} zero-weight (skipped)")

    # ========== STEP 2: Compute CCT (C_q) from left/right pair densities ==========
    # γ̃^0 = I_4 → vertex_mu_L=0 is the standard spin-traced path.  For
    # vertex_mu_L ∈ {1,2,3} the γ̃^μ vertex is folded into both P_l and
    # P_r so C_q is the proper per-channel interpolation metric for the
    # Lorentz pair density.  CCT^μ for transverse channels is Hermitian
    # indefinite and rank-deficient: TRS in non-magnetic ground states
    # gives near-null transverse-current modes that would be amplified
    # by 10^4–10^6 if we naively LU-solved through them (the original
    # MoS2 σ^B blowup).  The robust solver in :func:`solve_zeta`
    # uses an SVD pseudoinverse with rcond cutoff to drop those null
    # modes instead of inverting through them — the unique min-norm LSQ
    # solution.
    # Force-eager-import gamma_matrices so its module-level
    # ``gammas_sparse = [_to_sparse(g) for g in gammas]`` (which calls
    # ``jnp.nonzero``) runs OUTSIDE any JIT trace; otherwise the first
    # reference comes from inside the per-chunk kernel jit and trips
    # a ConcretizationTypeError.
    if int(vertex_mu_L) != 0:
        from . import gamma_matrices as _gm  # noqa: F401  (warm import)

    with timing.section("zeta_fit.CCT"):
        # ψ inputs at PADDED n_rmu (Phase 3a's load_centroids contract).
        # Open-spin rank-5 P_l/P_r for ALL channels — γ̃·γ̃ reduction
        # happens inside ``c_q_from_pair`` (None = γ̃^0 = I_4 for charge,
        # (perm, phase) = γ̃^μ for transverse).  Output C_q is rank-3
        # (k, μ, ν), same as the historical scalar path.
        chan_label = ("charge γ̃^0=I" if vertex_mu_L == 0
                      else f"transverse γ̃^{vertex_mu_L}")
        _use_pair_pipe_sm = (
            os.environ.get('LORRAX_PAIR_PIPELINE_SHARDMAP', '1') != '0')
        if _use_pair_pipe_sm:
            # Monolithic shard_map path — see c_q_from_psi_sm docstring.
            # Skips the global rank-5 P_l/P_r materialisation that XLA
            # would otherwise reshape to rank-3 fused-replicated.
            print(f"  Computing C_q via shard_map pipeline (open-spin, {chan_label})")
            if vertex_mu_L == 0:
                C_q = c_q_from_psi_sm(
                    psi_l_rmuT_X_fit, psi_l_rmu_Y_fit,
                    psi_r_rmuT_X_fit, psi_r_rmu_Y_fit,
                    kgrid=kgrid, mesh_xy=mesh_xy)
            else:
                gamma_mu = _gamma_perm_phase_mu(vertex_mu_L)
                C_q = c_q_from_psi_sm(
                    psi_l_rmuT_X_fit, psi_l_rmu_Y_fit,
                    psi_r_rmuT_X_fit, psi_r_rmu_Y_fit,
                    gamma_mu, gamma_mu,
                    kgrid=kgrid, mesh_xy=mesh_xy)
        else:
            print(f"  Computing pair densities P_l, P_r (open-spin, {chan_label})")
            P_l_k = pair_density(psi_l_rmuT_X_fit, psi_l_rmu_Y_fit, mesh_xy)
            P_r_k = pair_density(psi_r_rmuT_X_fit, psi_r_rmu_Y_fit, mesh_xy)
            P_l_k.block_until_ready()
            P_r_k.block_until_ready()
            if vertex_mu_L == 0:
                C_q = c_q_from_pair(P_l_k, P_r_k, kgrid=kgrid, mesh_xy=mesh_xy)
            else:
                gamma_mu = _gamma_perm_phase_mu(vertex_mu_L)
                C_q = c_q_from_pair(P_l_k, P_r_k, gamma_mu, gamma_mu,
                                    kgrid=kgrid, mesh_xy=mesh_xy)
            # Free pair densities - only needed for C_q
            del P_l_k, P_r_k
        C_q.block_until_ready()
        # C_q: (nqx, nqy, nqz, n_rmu_padded, n_rmu_padded) with zero
        # pad rows/cols.

        # Flatten for Cholesky.  Reshape uses padded extent (the
        # in-memory shape); factor_c_q slices to logical
        # internally via ``n_rmu_logical=``.
        C_q_flat = C_q.reshape(nq, n_rmu_padded, n_rmu_padded)
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)

    # ========== STEP 3: Compute L_q from CCT ==========
    # μ_L=0 (charge): C_q is PSD → 2D-blocked Cholesky factor L_q.
    # μ_L=1,2,3 (transverse): C_q is Hermitian indefinite — skip the
    # factorization and pass the slice through; the per-chunk
    # solve_zeta dispatches to an SVD pseudoinverse with
    # rcond cutoff (drops null transverse-current modes that would
    # otherwise be amplified by 10^4–10^6).
    with timing.section("zeta_fit.cholesky"):
        # Resolve once so the banner reflects what actually runs and
        # downstream callees skip their own 'auto' fallback.
        _resolved_solver_kind = _resolve_solver_kind(
            mesh_xy, int(vertex_mu_L), solver_kind)
        if int(vertex_mu_L) == 0:
            print(f"  Computing L_q = chol(C_q)  [PSD, charge channel, "
                  f"path={_resolved_solver_kind}]")
        else:
            print(f"  Pass through C_q  [γ̃^{vertex_mu_L} indefinite — "
                  f"path={_resolved_solver_kind}]")
        L_q = factor_c_q(
            C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L),
            n_rmu_logical=n_rmu, solver_kind=_resolved_solver_kind)
        L_q.block_until_ready()
        print(f"  L_q: {L_q.shape}")

    # Pre-compute per-q trace of L_q ONCE per channel.  Only the
    # transverse (LU) path uses it (for the ridge ``ε·|tr(L)|/n_rmu``
    # before each per-q LU solve).  Computing inside solve_zeta means an
    # all-reduce across the (mu/p_x, mu/p_y) mesh sharding fires on every
    # r-chunk — 17 s of GPU stream time on MoS2 3×3 bispinor across 4
    # r-chunks × 3 transverse channels.  L_q (which is CCT for the LU
    # path) doesn't change across r-chunks, so the trace is invariant.
    if int(vertex_mu_L) != 0:
        with timing.section("zeta_fit.trace_L_q"):
            cct_trace_per_q = jnp.einsum('qii->q', L_q)
            cct_trace_per_q.block_until_ready()
    else:
        cct_trace_per_q = None

    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget
    del C_q, C_q_flat
    with timing.section("zeta_fit.gc_pre_chunk_loop"):
        gc.collect()
        jax.clear_caches()  # Clear JAX function caches that may hold array refs

    # ========== STEP 4a: q-IBZ reduction + header writes (rank 0) ==========
    # When ``write_ibz_only=True`` (default), ζ is written for IBZ q's
    # only.  V_q at the full BZ is recovered by the reader / V_q
    # orchestrator using sym data from ``mf_header`` (see report.md
    # §2.4).  The on-disk ``zeta_q`` leading axis is ``n_q_disk``
    # rather than ``n_q_full``; the chunk loop slices
    # ``zeta_chunk[q_irr_full_idx]`` before writing.
    #
    # When ``write_ibz_only=False`` (bispinor μ_L>0 caller for now,
    # until the bispinor V_q orchestrator gains IBZ support), the
    # full-BZ axis is preserved on disk for back-compatibility.
    #
    # Auto-fallback: if the centroid set isn't orbit-closed under the
    # WFN sym group (typical for ``kmeans_cli --no-orbit`` outputs),
    # the V_q unfold can't reconstruct full-BZ V_q from the IBZ
    # representation, so we keep the full-BZ axis on disk too.  The
    # closure check uses the same helper that the V_q consumer would.
    if write_ibz_only:
        try:
            from centroid.orbit_syms import (
                compute_centroid_sym_perm as _check_perm,
            )
            _cent_idx_for_check = np.asarray(
                jax.device_get(centroid_indices), dtype=np.int32)
            _ntran_check = int(np.asarray(sym.sym_matrices).shape[0])
            # ``sym.sym_matrices`` holds the spatial ops; the fractional
            # translations live on WFNReader (BGW WFN.h5 layout).
            _check_perm(
                _cent_idx_for_check,
                sym_matrices=np.asarray(sym.sym_matrices[:_ntran_check]),
                translations=np.asarray(wfn.translations[:_ntran_check]),
                fft_grid=np.asarray(meta.fft_grid, dtype=np.int32),
            )
        except RuntimeError as _exc:
            if jax.process_index() == 0:
                _first = (_exc.args[0].splitlines()[0]
                          if _exc.args else str(_exc))
                print(f"  q-IBZ reduction: centroid orbit closure failed "
                      f"— falling back to full-BZ on disk.  Reason: "
                      f"{_first}")
            write_ibz_only = False

    # BGW Brillouin-zone wrap used by the V_q kernel
    # (``_qvec_wrap`` at ``gw/v_q_tile.py:1204``): ``q > kgrid/2 → q
    # − kgrid``.  The writer must match so the per-q phase
    # ``exp(-2πi (q/kgrid)·r)`` baked into the G-flat output is the
    # convention the consumer expects.
    def _bgw_wrap_q(q_int_kgrid: np.ndarray) -> np.ndarray:
        kg = np.asarray(meta.kgrid, dtype=np.float64)
        q = np.asarray(q_int_kgrid, dtype=np.float64)
        return np.where(q > kg / 2, q - kg, q)

    if write_ibz_only:
        (q_irr_kgrid_int, _q_full_to_irr_idx,
         _q_full_to_irr_sym, q_irr_full_idx) = sym.find_irreducible_qpoints()
        n_q_disk = int(q_irr_full_idx.shape[0])
        # IBZ fractional q-vectors for the G-flat accumulator (Phase C1b).
        # BGW wrap THEN divide by kgrid so the writer's per-q phase
        # matches the V_q kernel's ``apply_bloch_phase`` convention.
        _kgrid_arr_for_qfrac = np.asarray(meta.kgrid, dtype=np.float64)
        q_irr_frac = (_bgw_wrap_q(q_irr_kgrid_int)
                       / _kgrid_arr_for_qfrac[None, :])
        print(f"  q-IBZ reduction: {n_q_disk} IBZ q-points / {nq} full-BZ "
              f"(disk shrink {nq / max(1, n_q_disk):.1f}×)")
    else:
        q_irr_full_idx = None
        q_irr_frac = None
        n_q_disk = nq
        print(f"  q axis on disk: full BZ ({nq} q-points) "
              f"(write_ibz_only=False or closure check failed)")

    # ---- Phase C: G-flat on-disk format toggle -----------------
    # When ``LORRAX_WRITE_G_FLAT_ZETA=1`` is set, the writer
    # accumulates each r-chunk's contribution into a persistent
    # G-flat buffer via ``common.wfn_transforms.accumulate_rchunk_to_gflat``
    # and writes the final tensor as ``zeta_q_G`` (shape
    # ``(n_q_disk, n_rmu, ngkmax)``).  The full r-space ζ_q is never
    # materialised on disk or as a persistent device buffer.  When
    # ``vcoul_cutoff_ry`` is provided we build the per-q WFN.h5-style
    # sphere ``{G : |q+G|² ≤ cutoff}``, pad to a uniform ``ngkmax``
    # with the sentinel Miller index ``(-nx/2, -ny/2, -nz/2)``, and
    # store both the coeffs and the per-q components on disk.  Without
    # a cutoff the writer falls back to the full flat-FFT axis
    # (n_G_sph = n_rtot) — slow disk path, kept for sanity checks.
    write_g_flat_zeta = bool(int(os.environ.get(
        'LORRAX_WRITE_G_FLAT_ZETA', '0')))
    if write_g_flat_zeta and q_irr_frac is None:
        # Full-BZ q-vectors with BGW wrap, then / kgrid — same convention
        # the V_q kernel's ``_zeta_disk_to_G`` consumed via
        # ``_qvec_wrap``.
        _kgrid_arr_for_qfrac = np.asarray(meta.kgrid, dtype=np.float64)
        q_irr_frac = (_bgw_wrap_q(sym.kvecs_asints)
                       / _kgrid_arr_for_qfrac[None, :])

    # Build the per-q WFN.h5-style sphere when a cutoff is available.
    # The output is host numpy; the writer threads ``sphere_idx_padded``
    # through ``accumulate_rchunk_to_gflat`` and stashes the components
    # / ngk / cutoff into the isdf_header below.  ``zeta_cutoff_ry``
    # — distinct from V_q's bare-Coulomb cutoff — defines the per-q
    # sphere on disk.  Caller (``gw_init.fit_zeta``) validates
    # ``zeta_cutoff_ry ≥ bare_coulomb_cutoff_ry`` so V_q has every G
    # it needs.
    _gflat_sphere_idx_padded = None      # (n_q_disk, ngkmax) int32
    _gflat_gvec_components = None        # (n_q_disk, 3, ngkmax) int32
    _gflat_ngk_per_q = None              # (n_q_disk,) int32
    _gflat_ngkmax = None
    if write_g_flat_zeta and zeta_cutoff_ry is not None \
            and int(meta.sys_dim) != 0:
        from common.coulomb_sphere import compute_per_q_bare_coulomb_components
        _bvec_for_sphere = np.asarray(
            wfn.blat * wfn.bvec, dtype=np.float64)
        _sphere_pkg = compute_per_q_bare_coulomb_components(
            fft_grid=meta.fft_grid,
            bvec=_bvec_for_sphere,
            q_irr_frac=q_irr_frac,
            vcoul_cutoff_ry=float(zeta_cutoff_ry),
            sys_dim=int(meta.sys_dim),
        )
        _gflat_sphere_idx_padded = _sphere_pkg["sphere_idx_padded"]
        _gflat_gvec_components = _sphere_pkg["gvec_components_padded"]
        _gflat_ngk_per_q = _sphere_pkg["ngk_per_q"]
        _gflat_ngkmax = int(_sphere_pkg["ngkmax"])
        if jax.process_index() == 0:
            print(
                f"  G-flat ζ sphere: ngkmax={_gflat_ngkmax}, "
                f"min ngk={int(_gflat_ngk_per_q.min())}, "
                f"max ngk={int(_gflat_ngk_per_q.max())} "
                f"({_gflat_ngkmax / float(n_rtot):.3%} of n_rtot)")

    # ``zeta_q.h5`` carries the BGW-style ``mf_header`` verbatim from
    # the source WFN so any downstream consumer (the new
    # :class:`file_io.zeta_reader.ZetaReader`, or anything else that
    # speaks the WFN.h5 header) sees the same crystal / k-grid / G-grid
    # / symmetry view.  ``isdf_header`` holds ζ-specific metadata only
    # — centroids in FFT-grid + fractional coords, density label,
    # ``vertex_mu_L``.  Everything sym-derivable (q-IBZ list, centroid
    # orbit permutation, G-sphere) is rebuilt at read time via
    # ``SymMaps`` + ``orbit_syms`` and is *not* stored.
    #
    # Sequence: rank 0 pre-stripes the file, writes both header groups
    # in mode='w' (truncate), closes.  Then SlabIO re-opens with
    # mode='a' so the headers survive and ``create_dataset('zeta_q')``
    # appends rather than truncates.
    from file_io.slab_io import SlabIO
    from file_io.mf_header import copy_mf_header
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    from file_io._slab_io_ffi import _lustre_prestripe

    _wfn_src_path = getattr(wfn, '_filename', None)
    if _wfn_src_path is None:
        raise ValueError(
            "fit_zeta_to_h5: wfn must expose '_filename' (the source "
            "WFN.h5 path) so mf_header can be copied verbatim into "
            "zeta_q.h5.")

    # Centroid FFT-grid indices for the isdf_header.  ``centroid_indices``
    # may be a jax.Array on device; pull to host as int32 (n_rmu, 3).
    _cent_idx_np = np.asarray(jax.device_get(centroid_indices),
                              dtype=np.int32)
    if _cent_idx_np.shape != (n_rmu, 3):
        raise ValueError(
            f"fit_zeta_to_h5: centroid_indices has shape "
            f"{_cent_idx_np.shape}, expected ({n_rmu}, 3).")
    _density_label = 'scalar' if int(vertex_mu_L) == 0 else 'current'
    _hdr_kwargs = dict(
        r_mu_fft_idx=_cent_idx_np,
        fft_grid=meta.fft_grid,
        density=_density_label,
        vertex_mu_L=int(vertex_mu_L),
        zeta_layout=('G_flat' if write_g_flat_zeta else 'r_space'),
    )
    if write_g_flat_zeta and _gflat_gvec_components is not None:
        _hdr_kwargs.update(
            gvec_components=_gflat_gvec_components,
            ngk_per_q=_gflat_ngk_per_q,
            zeta_cutoff_ry=float(zeta_cutoff_ry),
        )
    elif write_g_flat_zeta:
        raise ValueError(
            "G-flat ζ writer is enabled (LORRAX_WRITE_G_FLAT_ZETA=1) "
            "but no ζ sphere was built — pass zeta_cutoff_ry to "
            "fit_zeta_to_h5 (or unset the env var to fall back to "
            "the r-space layout).")
    _isdf_hdr = IsdfHeader.build(**_hdr_kwargs)

    with timing.section("zeta_fit.write_headers"):
        if jax.process_index() == 0:
            # Pre-stripe the file (delete + lfs setstripe).  Idempotent
            # no-op on non-Lustre filesystems.  Must happen before any
            # h5py create so the stripe layout survives ``H5Fcreate``.
            stripe_count = int(
                os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "16"))
            stripe_size = os.environ.get(
                "LORRAX_PHDF5_STRIPE_SIZE_FS", "4M")
            _lustre_prestripe(output_file, stripe_count=stripe_count,
                              stripe_size=stripe_size)
            # Create file with mf_header, then append isdf_header.
            copy_mf_header(_wfn_src_path, output_file, dst_mode='w')
            write_isdf_header(output_file, _isdf_hdr, mode='a')
        jax.experimental.multihost_utils.sync_global_devices(
            "zeta_fit_headers_written")

    # ========== STEP 4b: SlabIO appends zeta_q to the pre-created file ==========
    # zeta_q is stored flat-q: shape (nq, n_rmu, n_rtot) with
    # q_flat = qx*nqy*nqz + qy*nqz + qz.  Flat-q is the ongoing
    # convention across LORRAX; see file_io.slab_io docs.  Chunk by
    # single-q r-slice so per-q reads stay contiguous.
    #
    # Single SlabIO handle reused for both create_dataset and all
    # writes — avoids the ~900 ms cost of a second collective
    # H5Fopen/close pair (measured 2026-04-18 at MoS2 3x3).  The
    # allgather backend doesn't need a long-lived handle (rank 0 writes
    # from a Python worker using plain h5py) so we keep the old
    # create-then-reopen pattern for that path.
    #
    # mode='a' (not 'w') so the pre-written mf_header + isdf_header
    # are preserved.  SlabIO's FFI prestripe step is skipped on 'a'
    # — we already striped above.
    #
    # Dataset layout ``(nq, n_rtot, n_rmu)`` — NOT ``(nq, n_rmu, n_rtot)``.
    # Rationale: per-r-chunk writes span the full innermost axis (n_rmu)
    # under this layout, so each ``(q, r)`` row is contiguous on disk.
    # Under the old ``(nq, n_rmu, n_rtot)`` layout we'd write n_rchunk <
    # n_rtot on the innermost axis, producing 480K × 1920-B scattered
    # strips per rank per write (measured at 0.18 GB/s on Perlmutter
    # pscratch, 8× slower than contiguous).  Per-q reads (V_q) stay
    # contiguous under this layout too: a 6.6 M-element slab at
    # ``(q, 0, 0)`` is a single contiguous block.  Downstream V_q
    # transposes the returned array on GPU to match the kernel's
    # (n_rmu, n_rtot) expectation — ~50 µs per q, negligible.
    with timing.section("zeta_fit.open_file"):
        if write_g_flat_zeta:
            # G-flat layout: ``zeta_q_G`` dataset (n_q_disk, n_rmu, ngkmax)
            # — WFN.h5 ``wfns/coeffs`` style with a fixed ``ngkmax``
            # padded G axis.  Per-q components live in
            # ``isdf_header/gvec_components`` (already serialised by the
            # write_isdf_header call above).  Chunking: one row per q
            # × full μ × full ngkmax keeps per-q reads contiguous.
            _n_G_sph = (int(_gflat_ngkmax)
                         if _gflat_ngkmax is not None else n_rtot)
            if use_ffi_io:
                zeta_io = SlabIO(output_file, mode='a', mesh=mesh_xy,
                                 backend=slab_io_backend)
                zeta_io.create_dataset(
                    'zeta_q_G',
                    shape=(n_q_disk, n_rmu, _n_G_sph),
                    dtype=np.complex128,
                    chunks=(1, n_rmu, _n_G_sph),
                )
            else:
                with SlabIO(output_file, mode='a', mesh=mesh_xy,
                            backend=slab_io_backend) as _zeta_create_io:
                    _zeta_create_io.create_dataset(
                        'zeta_q_G',
                        shape=(n_q_disk, n_rmu, _n_G_sph),
                        dtype=np.complex128,
                        chunks=(1, n_rmu, _n_G_sph),
                    )
                zeta_io = None
        elif use_ffi_io:
            zeta_io = SlabIO(output_file, mode='a', mesh=mesh_xy,
                             backend=slab_io_backend)
            zeta_io.create_dataset(
                'zeta_q',
                shape=(n_q_disk, n_rtot, n_rmu),
                dtype=np.complex128,
                chunks=(1, n_rchunk, n_rmu),
            )
        else:
            with SlabIO(output_file, mode='a', mesh=mesh_xy,
                        backend=slab_io_backend) as _zeta_create_io:
                _zeta_create_io.create_dataset(
                    'zeta_q',
                    shape=(n_q_disk, n_rtot, n_rmu),
                    dtype=np.complex128,
                    chunks=(1, n_rchunk, n_rmu),
                )
            zeta_io = None

    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]

    # Env-var override: forces the slow path (re-read WFN.h5 per
    # r-chunk) even when memory would allow host caching.  Useful for
    # probing the scaling regime where wavefunctions don't fit in host
    # memory (multi-TB WFN.h5).  LORRAX_GSPACE_MODE=file_reread to
    # override the caller's default.
    _gspace_mode_override = os.environ.get("LORRAX_GSPACE_MODE")
    if _gspace_mode_override:
        gspace_mode = _gspace_mode_override

    # Uniform band chunks over [b_full_start, b_full_end]: N-1 of
    # size ``band_chunk_size`` plus one remainder chunk.  This gives
    # the read/FFT pipeline and the pair-density einsum exactly
    # TWO compile shapes, regardless of where the L/R endpoints fall.
    # Chunks that straddle an L/R endpoint get handled in the loop
    # below by padding the left-side ``psi_L_bc`` slice with zero
    # bands — the resulting einsum still runs at the uniform
    # ``bc_size``, so it hits the same JIT cache.
    _bfs, _bfe = band_range_full
    band_chunk_ranges = [
        (_bfs + i * band_chunk_size,
         min(_bfs + (i + 1) * band_chunk_size, _bfe))
        for i in range((_bfe - _bfs + band_chunk_size - 1) // band_chunk_size)
    ]

    # Build the host-resident ψ(G) store.  Both modes keep zero
    # persistent device residency — the jit fetches one bc at a time
    # via io_callback.  See :mod:`common.psi_G_store` for details.
    from common.psi_G_store import build_psi_G_store
    psi_G_store = build_psi_G_store(
        wfn=wfn, sym=sym, mesh_xy=mesh_xy, meta=meta,
        band_chunk_ranges=band_chunk_ranges,
        bispinor=bispinor,
        mode=gspace_mode,
    )

    # ========== STEP 6: Loop over chunks ==========
    # Wall-clock totals for the end-of-fit timing line.  ``t_fit_total``
    # covers the fused fit_one_rchunk jit (load + pair + ZCT + solve) —
    # finer-grained breakdown now lives inside the jit and is only
    # observable via xprof, not perf_counter.
    t_fit_total = 0.0
    t_write_total = 0.0
    t_chunk_start = time.perf_counter()

    # Per-chunk writes go through zeta_io (SlabIO).  On the allgather
    # backend we keep the old async-writer pattern: main thread does
    # the allgather, rank-0 background thread does the h5py hyperslab
    # writes — this hides the h5 latency behind the next chunk's GPU
    # compute.  On the FFI backend, writes are collective so all ranks
    # must enter in lock-step; the synchronous SlabIO.write_slab call
    # from the main thread is the right shape (H5Dwrite's host-block
    # doesn't stall the CUDA stream, so the next chunk's build still
    # overlaps).
    write_queue = queue.Queue()
    write_error = [None]

    def writer_worker():
        """Rank-0 background thread: dequeue + h5py hyperslab writes."""
        try:
            while True:
                item = write_queue.get()
                if item is None:
                    break
                zeta_data, r_start, r_end, chunk_id, q_start, q_end = item
                # zeta_data is a host numpy array of shape
                # (q_end-q_start, n_rmu_padded, r_end-r_start) in the
                # order the shard_map produced.  Dataset on disk is
                # (nq, n_rtot, n_rmu_logical) — swap the last two
                # axes at write time AND slice the μ axis to logical
                # extent so on-disk extent stays logical (Phase 3a/3b
                # contract: pad rows of zeta are zero; on-disk
                # round-trips across mesh sizes).
                with h5py.File(output_file, 'a') as f:
                    for i, q_flat in enumerate(range(q_start, q_end)):
                        # zeta_data[i] is (n_rmu_padded, r_chunk) →
                        # transpose to (r_chunk, n_rmu_padded) → slice
                        # to logical (n_rmu) on the last axis.
                        f['zeta_q'][q_flat, r_start:r_end, :] = (
                            zeta_data[i].T[:, :n_rmu])
                write_queue.task_done()
        except Exception as e:
            write_error[0] = e
            write_queue.task_done()

    writer_thread = None
    if not use_ffi_io and jax.process_index() == 0:
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        writer_thread.start()

    # Peak GPU memory tracker — reports the all-time high-water mark (peak_bytes_in_use).
    # This is the number that determines whether you OOM: it includes JIT caches and
    # prior-stage allocations, not just the chunk loop arrays.
    # GPU high-water tracker.  The JAX CUDA PJRT on this stack returns
    # None from ``memory_stats()``, so we fall back to a single
    # nvidia-smi sample at the end of the chunk loop.  Sampled once
    # (not per-chunk) because concurrent nvidia-smi from all 4 ranks
    # inside the Shifter container has been observed to hang on some
    # Perlmutter node types.
    _peak_bytes = 0
    def _track_peak():
        nonlocal _peak_bytes
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,nounits,noheader", "--id=0"],
                text=True, timeout=2).strip()
            mb = int(out.splitlines()[0])
            _peak_bytes = max(_peak_bytes, mb * (1024 ** 2))
        except Exception:
            pass  # leave _peak_bytes = 0; caller suppresses the print

    from common.progress import LoopProgress
    r_progress = LoopProgress(
        num_chunks, print, title="zeta fitting",
        item_name="r-chunk", max_updates=min(num_chunks, 20))

    # norms_l_jax / norms_r_jax were built in STEP 1 above — reuse them
    # as the uniform-shape (nb,) inputs to the fit_one_rchunk jit.

    # ---- G-flat accumulator (zero-init, μ-sharded) ----
    # Persistent buffer: (n_q_disk, n_rmu_padded, ngkmax) c128 with
    # μ sharded across ('x', 'y') so each rank holds n_rmu/p per q.
    # Donated to ``accumulate_rchunk_to_gflat`` each iter; in-place add.
    # When the per-q sphere isn't available (no vcoul_cutoff_ry) we
    # fall back to the full flat-FFT axis n_rtot — slow, kept for
    # smoke / sanity tests.
    gflat_acc = None
    _gflat_acc_n_G = None
    if write_g_flat_zeta:
        from common.wfn_transforms import accumulate_rchunk_to_gflat
        # μ allocated at PADDED extent so the ('x','y') sharding
        # divides cleanly (same pad-then-clip-on-write pattern the
        # r-space path uses; see ``meta.n_rmu_padded`` and the
        # SlabIO ``valid_shape=`` argument below).  Pad rows are zero
        # because the back-solve produces zeta_pad = 0 (L_q's pad
        # block is identity).
        _n_rmu_padded = int(meta.n_rmu_padded)
        _gflat_acc_n_G = (int(_gflat_ngkmax)
                           if _gflat_ngkmax is not None else n_rtot)
        _gflat_acc_sharding = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
        gflat_acc = jax.jit(
            lambda: jnp.zeros(
                (n_q_disk, _n_rmu_padded, _gflat_acc_n_G),
                dtype=jnp.complex128),
            out_shardings=_gflat_acc_sharding,
        )()
        # Flat-axis chunking inside ``accumulate_rchunk_to_gflat``.
        # The kernel runs inside a ``shard_map`` over ``('x','y')`` and
        # chunks the per-rank flat ``(n_q · n_mu_local)`` axis into
        # rows-per-scan-iteration of ``chunk_size``.  Memory bound:
        # ``chunk_size · n_rtot · 16 B`` for the per-iteration FFT box.
        #
        # Default ``None`` (one-shot) is fine when the full per-rank
        # box ``N · n_rtot · 16 B`` fits — MoS2 3×3 at 4 ranks: 1.1 GB.
        # For CrI3-class FFT grids set ``LORRAX_GFLAT_CHUNK_SIZE`` to
        # an integer; the kernel zero-pads N up to a multiple of the
        # chunk size so any value works (no divisibility constraint
        # on either n_q or n_mu_local).
        _env_cs = int(os.environ.get('LORRAX_GFLAT_CHUNK_SIZE', '0') or 0)
        _gflat_chunk_size = _env_cs if _env_cs > 0 else None
        if jax.process_index() == 0:
            _p_prod = int(jax.device_count())
            _n_mu_local = int(meta.n_rmu_padded) // _p_prod
            _N = n_q_disk * _n_mu_local
            _cs = _gflat_chunk_size or _N
            print(f"  G-flat ζ accumulator: N={_N} rows/rank "
                  f"(n_q={n_q_disk} × n_mu_local={_n_mu_local}); "
                  f"chunk_size={_cs} → "
                  f"per-iter FFT box {_cs * n_rtot * 16 / 1e9:.2f} GB/rank")
        # Numpy → replicated: avoid the ``jnp.asarray`` wrap that would
        # single-device-stage and turn device_put into an all-reduce.
        _q_irr_frac_dev = jax.device_put(
            np.asarray(q_irr_frac, dtype=np.float64),
            NamedSharding(mesh_xy, P(None, None)))

    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_chunks):
            r_start = chunk_idx * chunk_r
            r_end = min(r_start + chunk_r, n_rtot)
            actual_n_rchunk = r_end - r_start

            # file_reread mode: (re)build the host-side ψ(G) tiles
            # for this r-chunk.  host_cache mode: no-op.
            psi_G_store.begin_rchunk(r_start, r_end)

            t0 = time.perf_counter()
            try:
                with timing.section("zeta_fit.chunk.fit_one_rchunk"), \
                     jax_profile.step_annotation("chunk_fit", step_num=chunk_idx):
                    zeta_chunk = fit_one_rchunk(
                        psi_G_store=psi_G_store,
                        psi_l_rmuT_X_fit=psi_l_rmuT_X_fit,
                        psi_r_rmuT_X_fit=psi_r_rmuT_X_fit,
                        L_q=L_q,
                        norms_l=norms_l_jax,
                        norms_r=norms_r_jax,
                        r_start_dyn=jnp.asarray(r_start, dtype=jnp.int32),
                        mesh_xy=mesh_xy,
                        meta=meta,
                        band_chunk_ranges=band_chunk_ranges,
                        band_range_left=band_range_left,
                        band_range_right=band_range_right,
                        band_range_full=band_range_full,
                        actual_n_rchunk=actual_n_rchunk,
                        q_chunk_size=q_chunk_size,
                        kvecs_frac=kvecs_frac,
                        vertex_mu_L=int(vertex_mu_L),
                        solver_kind=_resolved_solver_kind,
                        q_irr_full_idx=q_irr_full_idx,   # Phase B: gather inside the kernel
                        cct_trace_per_q=cct_trace_per_q,
                    )
                    zeta_chunk.block_until_ready()
            finally:
                # MUST run after block_until_ready — under file_reread
                # the host tiles are freed here and any still-pending
                # io_callback would use-after-free.
                psi_G_store.end_rchunk()
            t_fit_total += time.perf_counter() - t0

            # 6e. IBZ-slice → allgather (or FFI) → HDF5 write.
            # ``zeta_chunk`` is computed at full BZ q (the FFT in
            # ``solve_zeta`` naturally outputs all q's).  We slice to
            # IBZ rows here so the disk image is IBZ-only.  The
            # compute side stays full-BZ until a future optimization
            # threads IBZ through the solve.
            #
            # The allgather replicates zeta slices: per-device output is
            # (q_gather, n_rmu, chunk_r) which at large chunk_r can be huge.
            # Chunking over q keeps each allgather under memory limits.
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                # Phase B: ``zeta_chunk`` is already IBZ-shape
                # (n_q_disk, n_rmu, n_rchunk) — the gather happens
                # inside ``fit_one_rchunk`` before the triangular
                # solve.  No post-solve slice needed.  In full-BZ
                # mode (q_irr_full_idx=None) the kernel returns
                # full-BZ shape and we still alias to keep the
                # downstream write path uniform.
                zeta_chunk_ibz = zeta_chunk
                del zeta_chunk

                # Phase C1b: G-flat accumulator branch.  Accumulate
                # this r-chunk's contribution into ``gflat_acc`` and
                # skip the per-chunk SlabIO write.  After the loop
                # the full accumulator is written once.
                if write_g_flat_zeta:
                    gflat_acc = accumulate_rchunk_to_gflat(
                        rchunk=zeta_chunk_ibz, gflat_acc=gflat_acc,
                        fft_grid=meta.fft_grid, r0=r_start,
                        sphere_idx=_gflat_sphere_idx_padded,
                        qvec_frac=_q_irr_frac_dev,
                        norm='backward',
                        chunk_size=_gflat_chunk_size,
                        mesh=mesh_xy,
                    )
                    del zeta_chunk_ibz
                    t_write_total += time.perf_counter() - t0
                    r_progress.step()
                    continue

                # Each allgather produces a FULLY REPLICATED output per device:
                # q_gather × n_rmu × chunk_r × 16 bytes, plus NCCL temp of same size.
                # Cap to keep replicated output + NCCL under available memory.
                _bytes_per_q_replicated = 2 * n_rmu * actual_n_rchunk * 16
                _safe_q_gather = max(1, min(n_q_disk,
                    int(10 * 1024**3 / max(1, _bytes_per_q_replicated))))
                if q_gather_size > 0:
                    _q_gather = min(n_q_disk, q_gather_size, _safe_q_gather)
                else:
                    _q_gather = _safe_q_gather

                if use_ffi_io:
                    # FFI path: zeta_chunk_ibz is (n_q_disk,
                    # n_rmu_padded, chunk_r), dataset is
                    # (n_q_disk, n_rtot, n_rmu) at LOGICAL extent.
                    # Transpose the last two axes so the slab matches
                    # disk layout, then SlabIO ``valid_shape=`` clips
                    # the trailing pad slots off axis -1 on write
                    # (zeta pad rows are zero by construction — pad
                    # block of L_q is identity, so the back-solve
                    # produces zeta_pad = 0).
                    # The transpose is a JAX metadata-only operation
                    # (shard axis 'chunk_r' / sharded → axis 1 of the
                    # post-transpose tensor; NamedSharding updates in
                    # place, no data motion).
                    zeta_chunk_write = zeta_chunk_ibz.transpose(0, 2, 1)
                    actual_q = int(zeta_chunk_write.shape[0])
                    zeta_io.write_slab(
                        'zeta_q', zeta_chunk_write,
                        offset=(0, r_start, 0),
                        global_shape=(n_q_disk, n_rtot, n_rmu),
                        valid_shape=(actual_q, actual_n_rchunk, n_rmu),
                    )
                else:
                    # Allgather path: gather once per q-chunk on every rank,
                    # queue the per-q hyperslab writes on rank 0's
                    # background thread so the h5py I/O is hidden behind
                    # the next chunk's GPU compute.  process_allgather is
                    # itself a blocking D2H+collective — there is no async
                    # allgather-to-host API in JAX today — so the main
                    # thread stall per chunk is roughly (cross-rank NCCL
                    # time) + (PCIe D2H).  First chunk eats an extra ~1 s
                    # of NCCL/XLA first-collective setup.
                    for _q0 in range(0, n_q_disk, _q_gather):
                        _q1 = min(_q0 + _q_gather, n_q_disk)
                        _slice = zeta_chunk_ibz[_q0:_q1]
                        _gathered = jax.experimental.multihost_utils.process_allgather(
                            _slice, tiled=False)
                        if jax.process_index() == 0:
                            _g = np.asarray(_gathered)
                            if _g.ndim == 4 and _g.shape[0] == 1:
                                _g = _g[0]
                            write_queue.put((_g, r_start, r_end, chunk_idx, _q0, _q1))
                        del _gathered, _slice

                # No per-chunk sync_global_devices here: the
                # allgather is itself a collective so all ranks are
                # already aligned at this point, and we want the main
                # thread free to start next chunk's GPU compute while
                # rank 0's writer thread flushes to disk.  Final
                # sync_global_devices("zeta_writes_complete") at the
                # bottom of this function serves as the rendezvous.
                del zeta_chunk_ibz
            t_write_total += time.perf_counter() - t0
            r_progress.step()


    t_chunks_total = time.perf_counter() - t_chunk_start
    r_progress.finish()
    # Sample GPU memory ONCE after the last chunk's jit settles.  The
    # allocator keeps the peak reservation so this reads close to the
    # all-time high water.
    _track_peak()

    # ---- Phase C: write the accumulated G-flat ζ_q ----
    # One collective write of the persistent ``(n_q_disk, n_rmu,
    # ngkmax)`` tensor to disk.  The r-space per-chunk write loop
    # already short-circuited at ``continue``, so this is the ONLY
    # write that happens when ``write_g_flat_zeta`` is True.
    if write_g_flat_zeta:
        with timing.section("zeta_fit.write_g_flat"):
            # Pad slot zero-fill (WFN.h5 ``coeffs = 0`` convention).
            # The per-q gather inside ``accumulate_rchunk_to_gflat`` read
            # the sentinel ``(-nx/2, -ny/2, -nz/2)`` flat-FFT slot into
            # every pad position; those values are physical (not zero)
            # so we mask them here.  Logical slots ``[..., :ngk[q]]``
            # carry the real coeffs and are untouched.
            if _gflat_ngk_per_q is not None:
                _ngk_dev = jax.device_put(
                    np.asarray(_gflat_ngk_per_q, dtype=np.int32),
                    NamedSharding(mesh_xy, P(None)))
                _g_axis = jnp.arange(int(gflat_acc.shape[-1]),
                                      dtype=jnp.int32)        # (ngkmax,)
                _mask = (_g_axis[None, None, :] < _ngk_dev[:, None, None])
                gflat_acc = jnp.where(
                    _mask, gflat_acc, jnp.zeros_like(gflat_acc))
            jax.block_until_ready(gflat_acc)
            _n_G_sph = int(gflat_acc.shape[-1])
            if use_ffi_io:
                # On-disk extent is LOGICAL n_rmu; in-memory buffer
                # is PADDED ``n_rmu_padded``.  SlabIO ``valid_shape=``
                # clips the trailing μ pad rows on write — same trick
                # the r-space writer uses (those rows are zero).
                zeta_io.write_slab(
                    'zeta_q_G', gflat_acc,
                    offset=(0, 0, 0),
                    global_shape=(n_q_disk, n_rmu, _n_G_sph),
                    valid_shape=(n_q_disk, n_rmu, _n_G_sph),
                )
            else:
                # allgather backend: same per-q allgather pattern as
                # the r-space write loop, but only once (not per
                # chunk).  The full tensor is at most a few GB
                # replicated; for CrI3 scale the FFI backend is
                # mandatory anyway.
                _gathered = jax.experimental.multihost_utils.process_allgather(
                    gflat_acc, tiled=False)
                if jax.process_index() == 0:
                    _g = np.asarray(_gathered)
                    if _g.ndim == 4 and _g.shape[0] == 1:
                        _g = _g[0]
                    import h5py as _h5
                    with _h5.File(output_file, 'a') as _f:
                        _f['zeta_q_G'][...] = _g
                del _gathered
        del gflat_acc

    # Drain the rank-0 writer queue (allgather backend only; FFI path
    # writes are already fully flushed by the synchronous SlabIO calls).
    if jax.process_index() == 0 and writer_thread is not None:
        write_queue.join()
        write_queue.put(None)
        writer_thread.join()
        if write_error[0] is not None:
            raise RuntimeError(f"Async writer failed: {write_error[0]}")

    # Close the SlabIO handle (FFI path only; allgather path never
    # opened one after STEP 4).
    with timing.section("zeta_fit.close_io"):
        if zeta_io is not None:
            zeta_io.close()

    with timing.section("zeta_fit.sync_global"):
        jax.experimental.multihost_utils.sync_global_devices("zeta_writes_complete")

    # Flip ``isdf_header/zeta_is_done`` to True now that every chunk
    # has drained to disk.  Restart paths key off this flag to decide
    # whether the on-disk ζ is trustable; flipping it here (after the
    # global sync above) guarantees every rank's writes are durable.
    if jax.process_index() == 0:
        from file_io.isdf_header import mark_zeta_done
        mark_zeta_done(output_file)

    # Free the host tiles (host_cache mode only; file_reread's tiles
    # are already empty after the final end_rchunk).  The phdf5 reader
    # itself is cached at module level and survives.
    psi_G_store.close()

    # Per-stage timing breakdown.  ``fit`` is the fused fit_one_rchunk jit;
    # ``H5`` is the allgather+write (or FFI write_slab).  Everything else
    # lives inside the jit — see xprof for the intra-jit breakdown.
    print(f"  Zeta output: {output_file}  shape: "
          f"(n_q_disk={n_q_disk} of {nqx}·{nqy}·{nqz}={nq} full-BZ, "
          f"n_rtot={n_rtot}, n_rmu={n_rmu})")
    print(f"  Timing ({num_chunks} r-chunks, {t_chunks_total:.1f}s total):")
    for label, t in [("fit", t_fit_total), ("H5", t_write_total)]:
        print(f"    {label:<6} {t:6.2f}s  {100*t/t_chunks_total:4.1f}%")

    # Return only peak-memory high-water mark; centroid wavefunctions
    # are not returned (see docstring — callers re-load them directly
    # via ``load_centroids_band_chunked``).
    return _peak_bytes

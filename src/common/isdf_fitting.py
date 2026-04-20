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
    get_psi_rchunk,
    iter_psi_rchunk_bandwise,
    load_centroids_band_chunked,
)


def load_gspace_for_bands(
    wfn, sym, meta, mesh_xy, band_range, bispinor,
    band_chunk_size: int = 16,
    band_chunk_ranges: list[tuple[int, int]] | None = None,
) -> list[tuple[jax.Array, tuple[int, int]]]:
    """Load G-space wavefunctions for all band chunks ONCE.

    Caches the expensive HDF5 read + scatter so it can be reused across
    multiple r-chunk iterations.  Returns a list of ``(psi_Gtot,
    band_range)`` tuples, one per band chunk.

    Pass ``band_chunk_ranges`` when the downstream streaming
    pair-density loop uses custom chunk boundaries (e.g. respecting
    left/right pair-density endpoints) — the cache keys must match
    the yielded ``bc_range`` sequence exactly.  When None, contiguous
    chunks of ``band_chunk_size`` are built from ``band_range``.
    """
    if band_chunk_ranges is None:
        b_start, b_end = band_range
        nb_total = b_end - b_start
        num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
        band_chunk_ranges = [
            (b_start + i * band_chunk_size,
             min(b_start + (i + 1) * band_chunk_size, b_end))
            for i in range(num_band_chunks)
        ]
    cached_gspace = []
    for bc_range in band_chunk_ranges:
        global_psi_Gtot, _ = read_Gvecs_to_devices(
            wfn, sym, bc_range, meta, bispinor, mesh_xy)
        cached_gspace.append((global_psi_Gtot, tuple(bc_range)))
    return cached_gspace


def _band_chunk_ranges_respecting_endpoints(
    band_full: tuple[int, int],
    endpoints: list[int],
    chunk_size: int,
) -> list[tuple[int, int]]:
    """Build contiguous band-chunk ranges that never straddle any of
    the given ``endpoints``.

    Given the full band range ``(fs, fe)`` and a list of internal
    breakpoints (typically the left/right pair-density endpoints
    ``{b_L_start, b_L_end, b_R_start, b_R_end}``), return a list of
    ``(bc_start, bc_end)`` chunks such that each chunk lies fully
    inside one "segment" — which means the chunk is entirely inside
    the left range, entirely inside the right range, inside both,
    or outside both.  Downstream code can then do a Python-level
    ``if bc_in_left:`` to skip the einsum for chunks outside each
    range, never materialising an out-of-range matmul.
    """
    fs, fe = band_full
    breakpoints = sorted(set([fs, fe] + [b for b in endpoints if fs < b < fe]))
    ranges: list[tuple[int, int]] = []
    for seg_start, seg_end in zip(breakpoints[:-1], breakpoints[1:]):
        pos = seg_start
        while pos < seg_end:
            nxt = min(pos + chunk_size, seg_end)
            ranges.append((pos, nxt))
            pos = nxt
    return ranges


# ============================================================================
# Pair density computation: P_k,ab(r_mu, r_nu) = sum_n psi*_nk,a(r_mu) * psi_nk,b(r_nu)
# ============================================================================

# Cache for pair density jitted functions
_compute_pair_density_cache = {}



# ============================================================================
# Spin-traced pair density (matching gw_jax treatment)
# ============================================================================
# gw_jax traces over spin for ISDF fitting:
#   P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(r_μ) × ψ_{n,k,s}(r_ν)
# This reduces lstsq error by fitting a lower-rank object.

def compute_pair_density_spin_traced(
	psi_rmuT_X: jax.Array,
	psi_rmu_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute spin-traced pair density P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(ν).

	This matches gw_jax spin treatment for ISDF fitting.

	Input shapes and shardings:
		psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
			- conj(psi_nk,s(r_mu)) with mu sharded on X
		psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
			- psi_nk,s(r_nu) with nu sharded on Y

	Output:
		P_k: (nk, n_rmu, n_rmu) with P(None, 'x', 'y')
			- P[k, mu, nu] = Σ_{n,s} psi*_nk,s(r_mu) * psi_nk,s(r_nu)
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	cache_key = ('spin_traced', id(mesh_xy), nk, n_rmu, nb, ns)

	if cache_key not in _compute_pair_density_cache:
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		xy_out = NamedSharding(mesh_xy, P(None, 'x', 'y'))

		@partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
		def _compute_P_traced(psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			"""
			psi_L: (nk, n_rmu, nb, ns) - conjugated, mu on X
			psi_R: (nk, nb, ns, n_rmu) - nu on Y

			Contract over band (n) and spin (s): 'kmns,knsv->kmv'
			"""
			return jnp.einsum('kmns,knsv->kmv', psi_L, psi_R, optimize=True)

		_compute_pair_density_cache[cache_key] = _compute_P_traced

	return _compute_pair_density_cache[cache_key](psi_rmuT_X, psi_rmu_Y)


def compute_pair_density_spin_matrix(
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute spin-resolved pair density P_k,ab(mu,col) = sum_n psi*_nka(mu) psi_nkb(col).

	Input:
		psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
		psi_rcol_Y: (nk, nb, ns, n_col) with P(None, None, None, 'y')

	Output:
		P_k_ab: (nk, ns, ns, n_rmu, n_col) with P(None, None, None, 'x', 'y')
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	cache_key = ('spin_matrix', id(mesh_xy), nk, n_rmu, nb, ns, n_col)

	if cache_key not in _compute_pair_density_cache:
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		xy_out = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))

		@partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
		def _compute_P_spin_matrix(psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			# Keep spin channels explicit: P[k,a,b,mu,col].
			return jnp.einsum('kmna,knbr->kabmr', psi_L, psi_R, optimize=True)

		_compute_pair_density_cache[cache_key] = _compute_P_spin_matrix

	return _compute_pair_density_cache[cache_key](psi_rmuT_X, psi_rcol_Y)


_accum_pair_density_cache = {}


def accumulate_pair_density_spin_traced(
	P_accum: jax.Array,
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""``P_accum + Σ_{n,s} ψ*_{n,s}(r_μ) ψ_{n,s}(r_col)`` in one JIT.

	Used by the band-chunk streaming pair-density path: each band
	chunk contributes its partial einsum to the running accumulator
	without materialising an intermediate — XLA can fuse the add
	into the einsum's output write.

	Shapes / shardings match :func:`compute_pair_density_spin_traced`
	on the two inputs; ``P_accum`` lives on the same sharding as the
	output (``P(None, 'x', 'y')``).
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	cache_key = ('spin_traced_accum', id(mesh_xy), nk, n_rmu, nb, ns, n_col)
	if cache_key not in _accum_pair_density_cache:
		P_sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		L_sharding = NamedSharding(mesh_xy, P(None, 'x', None, None))
		R_sharding = NamedSharding(mesh_xy, P(None, None, None, 'y'))

		@partial(jax.jit,
				 in_shardings=(P_sharding, L_sharding, R_sharding),
				 out_shardings=P_sharding,
				 donate_argnums=(0,))
		def _accum(P_in: jax.Array, psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			return P_in + jnp.einsum('kmns,knsv->kmv', psi_L, psi_R, optimize=True)

		_accum_pair_density_cache[cache_key] = _accum
	return _accum_pair_density_cache[cache_key](P_accum, psi_rmuT_X, psi_rcol_Y)


def accumulate_pair_density_spin_matrix(
	P_accum: jax.Array,
	psi_rmuT_X: jax.Array,
	psi_rcol_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""``P_accum + Σ_n ψ*_{n,a}(r_μ) ψ_{n,b}(r_col)`` (spin-resolved).

	Shapes / shardings mirror :func:`compute_pair_density_spin_matrix`
	on the inputs; ``P_accum`` holds the 2×2 spin matrix on
	``P(None, None, None, 'x', 'y')``.
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_col = psi_rcol_Y.shape
	cache_key = ('spin_matrix_accum', id(mesh_xy), nk, n_rmu, nb, ns, n_col)
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

def compute_CCT_from_left_right(
	P_l_k: jax.Array,
	P_r_k: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute CCT from separate left and right spin-traced pair densities.

	C_q(μ,ν) = FFT[ conj(IFFT(P_l)) ⊙ IFFT(P_r) ]

	This matches gw_jax physics where left and right have different band ranges.

	Args:
		P_l_k: (nk, n_rmu, n_rmu) left pair density, P(None, 'x', 'y')
		P_r_k: (nk, n_rmu, n_rmu) right pair density, P(None, 'x', 'y')
		kgrid: (nkx, nky, nkz) — the 3-D form only appears inside the FFT helper.
		mesh_xy: Device mesh

	Returns:
		C_q: (nq, n_rmu, n_rmu) flat-q, P(None, 'x', 'y').
	"""
	nkx, nky, nkz = kgrid
	nk, n_rmu, _ = P_l_k.shape

	cache_key = ('CCT_LR', id(mesh_xy), nk, n_rmu, nkx)

	if cache_key not in _isdf_pipeline_cache:
		flat_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		spec = P(None, None, None, 'x', 'y')
		local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, spec, norm='forward')
		local_fftn  = make_flat_k_fftn( mesh_xy, kgrid, spec, norm='forward')

		# Flat-k in and flat-q out (same (nk, μ, μ) shape and sharding on both
		# sides).  The 3-D k-grid reshape now lives inside ``local_{i,}fftn``,
		# so donation is safe: P_l and P_r are consumed by the two IFFTs and
		# the result preserves rank-3 end-to-end.
		@partial(jax.jit, in_shardings=(flat_xy, flat_xy), out_shardings=flat_xy,
		         donate_argnums=(0, 1))
		def _compute_CCT_LR(P_l: jax.Array, P_r: jax.Array) -> jax.Array:
			# norm='forward' for BOTH IFFT and FFT — convolution theorem with
			# unscaled IFFT (sum) + FFT/N matches gw_jax's direct k-sum:
			#   C_q = FFT(conj(IFFT(A)) ⊙ IFFT(B)) = Σ_k A*_k B_{k+q}.
			P_l_Rt = local_ifftn(P_l)
			P_r_Rt = local_ifftn(P_r)
			C_Rt = jnp.conj(P_l_Rt) * P_r_Rt
			return local_fftn(C_Rt)

		_isdf_pipeline_cache[cache_key] = _compute_CCT_LR

	return _isdf_pipeline_cache[cache_key](P_l_k, P_r_k)


def compute_CCT_from_left_right_spin_matrix(
	P_l_k_ab: jax.Array,
	P_r_k_ab: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute CCT from explicit spin-channel pair densities.

	C_q(mu,nu) = FFT[ sum_ab conj(IFFT(P_l_ab)) * IFFT(P_r_ab) ].
	For identical left/right windows, this reduces to Frobenius ||P_ab||^2.
	"""
	nkx, nky, nkz = kgrid
	nk, ns1, ns2, n_rmu, n_col = P_l_k_ab.shape
	assert n_col == n_rmu, f"CCT expects square centroid columns, got n_col={n_col}, n_rmu={n_rmu}"

	cache_key = ('CCT_LR_spin_matrix', id(mesh_xy), nk, ns1, ns2, n_rmu, nkx, nky, nkz)

	if cache_key not in _isdf_pipeline_cache:
		spin_spec = P(None, None, None, 'x', 'y')  # 5-axis: (nk, a, b, μ, μ)
		scalar_spec = P(None, 'x', 'y')            # 3-axis: (nk, μ, μ)
		spin_flat_shard = NamedSharding(mesh_xy, spin_spec)
		scalar_flat_shard = NamedSharding(mesh_xy, scalar_spec)
		# The 3D-form specs prepend (None, None, None) to the flat spec.
		fft_spin_3d = P(None, None, None, None, None, 'x', 'y')
		fft_scalar_3d = P(None, None, None, 'x', 'y')
		local_ifftn_spin   = make_flat_k_ifftn(mesh_xy, kgrid, fft_spin_3d,   norm='forward')
		local_fftn_scalar  = make_flat_k_fftn( mesh_xy, kgrid, fft_scalar_3d, norm='forward')

		# Split into two sub-jits so XLA can donate the rank-5 P_l/P_r into
		# their rank-5 IFFT outputs; the sum-over-spin rank drop (5→3)
		# happens in the outer jit which doesn't try to donate.
		@partial(jax.jit, in_shardings=spin_flat_shard, out_shardings=spin_flat_shard,
		         donate_argnums=(0,))
		def _ifft_conj(P_l: jax.Array) -> jax.Array:
			return jnp.conj(local_ifftn_spin(P_l))

		@partial(jax.jit,
		         in_shardings=(spin_flat_shard, spin_flat_shard),
		         out_shardings=scalar_flat_shard,
		         donate_argnums=(0, 1))
		def _ifft_contract_fft(P_r: jax.Array, P_l_Rt_conj: jax.Array) -> jax.Array:
			P_r_Rt = local_ifftn_spin(P_r)
			C_Rt = jnp.sum(P_l_Rt_conj * P_r_Rt, axis=(1, 2))
			return local_fftn_scalar(C_Rt)

		def _compute_CCT_LR_spin(P_l: jax.Array, P_r: jax.Array) -> jax.Array:
			P_l_Rt_conj = _ifft_conj(P_l)
			return _ifft_contract_fft(P_r, P_l_Rt_conj)

		_isdf_pipeline_cache[cache_key] = _compute_CCT_LR_spin

	return _isdf_pipeline_cache[cache_key](P_l_k_ab, P_r_k_ab)


def compute_gram_q0_from_left_right(
	P_v_k: jax.Array,
	P_c_k: jax.Array,
	k_weights: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Build the q=0 valence-conduction pair-product Gram matrix from two
	per-k pair densities produced by ``compute_pair_density_spin_traced``.

	Mathematically (q=0 special case of the CCT-over-k structure):

	    G_{ab} = Σ_k w_k · [Σ_v φ_{v,k}(r_a)  φ*_{v,k}(r_b)]
	                     · [Σ_c ψ*_{c,k}(r_a) ψ_{c,k}(r_b)]
	           = Σ_k w_k · conj(P_v_k(a,b)) · P_c_k(a,b)

	where both ``P_v_k`` and ``P_c_k`` follow the gw_jax convention

	    P_k(μ, ν) = Σ_{n,s} ψ*_{n,k,s}(μ) · ψ_{n,k,s}(ν)

	(the ``compute_pair_density_spin_traced`` output). The ``conj`` on
	``P_v_k`` flips its conjugation pattern to the valence-projector form
	φ(a)φ*(b); multiplying it elementwise by the conduction
	ψ*(a)ψ(b) yields the valence-conduction pair-product Gram used for
	pivoted-Cholesky candidate pruning. See
	``sandbox/pivoted_cholesky.md`` §1 for the full derivation.

	Compared to ``compute_CCT_from_left_right``, this drops the k→q FFT
	pair: at q=0 the k-sum IS the answer, no convolution is needed. For
	any q≠0 you want the CCT path, not this one.

	Args:
		P_v_k: (nk, n_rmu, n_rmu) complex, valence pair density,
			P(None, 'x', 'y'). The gw_jax-convention pair density — pass
			the output of ``compute_pair_density_spin_traced`` fed with
			the valence window.
		P_c_k: (nk, n_rmu, n_rmu) complex, conduction pair density, same
			layout. Same routine, conduction window.
		k_weights: (nk,) real, k-point weights (IBZ weights summing to 1,
			or 1/nk_tot for each full-BZ k-point — whatever convention
			was used when building P_v_k / P_c_k).
		mesh_xy: ('x','y') device mesh, same one used for the pair
			densities.

	Returns:
		G: (n_rmu, n_rmu) complex Hermitian PSD, sharded P('x','y') on
			the mesh.
	"""
	nk, n_rmu, _ = P_v_k.shape
	cache_key = ('gram_q0_LR', id(mesh_xy), nk, n_rmu)

	if cache_key not in _isdf_pipeline_cache:
		in_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P('x', 'y'))
		kw_rep = NamedSharding(mesh_xy, P())

		@partial(jax.jit,
		         in_shardings=(in_xy, in_xy, kw_rep),
		         out_shardings=out_xy)
		def _compute_gram_q0(P_v: jax.Array, P_c: jax.Array,
		                     kw: jax.Array) -> jax.Array:
			# Per-k product of (conj P_v) × P_c, weighted by kw[k], summed.
			# Broadcasting kw to (nk, 1, 1) matches the (nk, μ, ν) layout.
			prod = jnp.conj(P_v) * P_c
			G = jnp.sum(kw[:, None, None] * prod, axis=0)
			# Symmetrize: the q=0 Gram is Hermitian by construction, but
			# fp-roundoff + reduction-order noise can break it. The
			# pivoted-Cholesky select does its own diagonal clamp, but
			# symmetrizing here costs only O(n_rmu²) and keeps the select
			# on a bit-cleaner input.
			G = 0.5 * (G + jnp.conj(G.T))
			return G

		_isdf_pipeline_cache[cache_key] = _compute_gram_q0

	return _isdf_pipeline_cache[cache_key](P_v_k, P_c_k, k_weights)


def compute_ZCT_from_left_right_zchunk(
	P_l_k_muz: jax.Array,
	P_r_k_muz: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute ZCT from left and right pair densities, both at (μ, z-chunk).

	Z_q(μ,r) = FFT[ conj(IFFT(P_l(μ,r))) ⊙ IFFT(P_r(μ,r)) ]

	Args:
		P_l_k_muz: (nk, n_rmu, n_zchunk) left pair density, flat-k, P(None, 'x', 'y').
		P_r_k_muz: (nk, n_rmu, n_zchunk) right pair density, flat-k, P(None, 'x', 'y').
		kgrid: (nkx, nky, nkz) — 3-D form appears only inside the FFT helper.
		mesh_xy: Device mesh

	Returns:
		Z_q: (nq, n_rmu, n_zchunk) flat-q, P(None, 'x', 'y').
	"""
	nkx, nky, nkz = kgrid
	nk, n_rmu, n_zchunk = P_l_k_muz.shape
	assert nk == nkx * nky * nkz, (
		f"P_l_k_muz flat-k dim {nk} does not match kgrid product {nkx*nky*nkz}"
	)
	assert P_r_k_muz.shape == P_l_k_muz.shape, (
		f"P_l/P_r shape mismatch: {P_l_k_muz.shape} vs {P_r_k_muz.shape}"
	)

	cache_key = ('ZCT_LR', id(mesh_xy), nk, n_rmu, n_zchunk)

	if cache_key not in _isdf_pipeline_cache:
		flat_spec = P(None, 'x', 'y')
		flat_shard = NamedSharding(mesh_xy, flat_spec)
		spec_3d = P(None, None, None, 'x', 'y')
		local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid, spec_3d, norm='forward')
		local_fftn  = make_flat_k_fftn( mesh_xy, kgrid, spec_3d, norm='forward')

		@partial(jax.jit, in_shardings=flat_shard, out_shardings=flat_shard,
		         donate_argnums=(0,))
		def _left_ifft_conj(P_l: jax.Array) -> jax.Array:
			return jnp.conj(local_ifftn(P_l))

		@partial(jax.jit,
		         in_shardings=(flat_shard, flat_shard), out_shardings=flat_shard,
		         donate_argnums=(0, 1))
		def _right_ifft_mul_fft(P_r: jax.Array, P_l_Rt: jax.Array) -> jax.Array:
			P_r_Rt = local_ifftn(P_r)
			Z_Rt = P_l_Rt * P_r_Rt
			return local_fftn(Z_Rt)

		def _compute_ZCT_LR(P_l: jax.Array, P_r: jax.Array) -> jax.Array:
			P_l_Rt = _left_ifft_conj(P_l)
			return _right_ifft_mul_fft(P_r, P_l_Rt)

		_isdf_pipeline_cache[cache_key] = _compute_ZCT_LR

	return _isdf_pipeline_cache[cache_key](P_l_k_muz, P_r_k_muz)


def compute_ZCT_from_left_right_zchunk_spin_matrix(
	P_l_k_ab_muz: jax.Array,
	P_r_k_ab_muz: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute ZCT from explicit spin-channel pair densities on (mu, r_chunk).

	Z_q(mu,r) = FFT[ sum_ab conj(IFFT(P_l_ab(mu,r))) * IFFT(P_r_ab(mu,r)) ].
	"""
	nkx, nky, nkz = kgrid
	nk, n_s1, n_s2, n_rmu, n_zchunk = P_l_k_ab_muz.shape
	assert nk == nkx * nky * nkz, (
		f"P_l_k_ab_muz flat-k dim {nk} does not match kgrid product {nkx*nky*nkz}"
	)
	assert P_r_k_ab_muz.shape == P_l_k_ab_muz.shape, (
		f"P_l/P_r shape mismatch: {P_l_k_ab_muz.shape} vs {P_r_k_ab_muz.shape}"
	)

	cache_key = ('ZCT_LR_spin_matrix', id(mesh_xy), nk, n_s1, n_s2, n_rmu, n_zchunk)

	if cache_key not in _isdf_pipeline_cache:
		spin_spec = P(None, None, None, 'x', 'y')  # 5-axis: (nk, a, b, μ, z)
		scalar_spec = P(None, 'x', 'y')            # 3-axis: (nk, μ, z)
		spin_flat_shard = NamedSharding(mesh_xy, spin_spec)
		scalar_flat_shard = NamedSharding(mesh_xy, scalar_spec)
		spec_spin_3d = P(None, None, None, None, None, 'x', 'y')
		spec_scalar_3d = P(None, None, None, 'x', 'y')
		local_ifftn_spin   = make_flat_k_ifftn(mesh_xy, kgrid, spec_spin_3d,   norm='forward')
		local_fftn_scalar  = make_flat_k_fftn( mesh_xy, kgrid, spec_scalar_3d, norm='forward')

		@partial(jax.jit, in_shardings=spin_flat_shard, out_shardings=spin_flat_shard,
		         donate_argnums=(0,))
		def _left_ifft_conj(P_l: jax.Array) -> jax.Array:
			return jnp.conj(local_ifftn_spin(P_l))

		@partial(jax.jit,
		         in_shardings=(spin_flat_shard, spin_flat_shard),
		         out_shardings=scalar_flat_shard,
		         donate_argnums=(0, 1))
		def _right_ifft_contract_fft(P_r: jax.Array, P_l_Rt: jax.Array) -> jax.Array:
			P_r_Rt = local_ifftn_spin(P_r)
			# Sum over spin channels (axes 1, 2 on flat-k rank-5 tensor).
			Z_Rt = jnp.sum(P_l_Rt * P_r_Rt, axis=(1, 2))
			return local_fftn_scalar(Z_Rt)

		def _compute_ZCT_LR_spin(P_l: jax.Array, P_r: jax.Array) -> jax.Array:
			P_l_Rt = _left_ifft_conj(P_l)
			return _right_ifft_contract_fft(P_r, P_l_Rt)

		_isdf_pipeline_cache[cache_key] = _compute_ZCT_LR_spin

	return _isdf_pipeline_cache[cache_key](P_l_k_ab_muz, P_r_k_ab_muz)



# ============================================================================
# 2D Blocked Cholesky Solver - memory efficient for large n_rmu
# ============================================================================

# Cache for 2D Cholesky functions
_chol_2d_cache = {}


# ============================================================================
# Full zeta fitting pipeline with z-chunk loop and HDF5 output
# ============================================================================

def compute_L_q_from_CCT(
    C_q: jax.Array,
    mesh_xy: Mesh,
    block_size: int = None,
) -> jax.Array:
    """
    Compute Cholesky factor L_q from CCT matrix using 2D blocked algorithm.

    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix, sharded P(None, 'x', 'y')
        mesh_xy: 2D device mesh
        block_size: Tile block size (auto if None)

    Returns:
        L_q: (nq, n_rmu, n_rmu) Cholesky factor, sharded P(None, 'x', 'y')
    """
    nq, n_rmu, n_rmu2 = C_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"

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

    if block_size is None:
        block_size, J = compute_block_size_for_2d_cholesky(n_rmu, Pr, Pc)
    else:
        J = n_rmu // block_size

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


def solve_zeta_from_L_q(
    L_q: jax.Array,
    Z_q: jax.Array,
    mesh_xy: Mesh,
    q_chunk_size: int = 1,
) -> jax.Array:
    """
    Solve for zeta_q given pre-computed Cholesky factor L_q.

    Uses q-chunked all-gather strategy: gather B_q L matrices at a time,
    then solve all B_q systems in parallel using vmap.

    Memory trade-off:
    - q_chunk_size=1: Minimum memory (one L replicated at a time)
    - q_chunk_size=nq: Maximum parallelism (all L replicated)

    Args:
        L_q: (nq, n_rmu, n_rmu) Cholesky factor, sharded P(None, 'x', 'y')
        Z_q: (nq, n_rmu, n_zchunk) ZCT matrix, sharded P(None, 'x', 'y')
             or P(None, None, ('x','y')) if caller already resharded
        mesh_xy: 2D device mesh
        q_chunk_size: Number of q-points to solve simultaneously (default 1)

    Returns:
        zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, None, ('x','y'))
    """
    nq, n_rmu, _ = L_q.shape
    _, _, n_zchunk = Z_q.shape

    # Compute padding needed for even sharding across all devices
    total_devices = mesh_xy.devices.size
    n_zchunk_padded = ((n_zchunk + total_devices - 1) // total_devices) * total_devices
    needs_padding = n_zchunk_padded != n_zchunk

    z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    L_rep_shard = NamedSharding(mesh_xy, P(None, None))
    L_batch_rep_shard = NamedSharding(mesh_xy, P(None, None, None))  # (B_q, n_rmu, n_rmu)
    q_batch = min(q_chunk_size, nq)
    nq_padded = ((nq + q_batch - 1) // q_batch) * q_batch

    # Cache key for solve function (includes q_chunk_size and padded size)
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu, n_zchunk_padded, q_chunk_size)

    if cache_key not in _solve_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            y = jax.scipy.linalg.solve_triangular(L, Z_cols, lower=True)
            zeta = jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
            return zeta

        # Vectorized solve for a batch of q-points
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None, None), P(None, None, ('x', 'y'))),
                 out_specs=P(None, None, ('x', 'y')))
        def _sharded_cho_solve_batch(L_batch: jax.Array, Z_batch: jax.Array) -> jax.Array:
            """Solve (B_q, n_rmu, n_rmu) @ (B_q, n_rmu, n_cols) -> (B_q, n_rmu, n_cols)"""
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
        @jax.jit
        def _reshard_z(z):
            return jax.lax.with_sharding_constraint(z, _target_sharding)
        Z_col = _reshard_z(Z_q)
        Z_col.block_until_ready()
        del Z_q

    # Fast path: solve all q-points at once
    if q_batch >= nq:
        result = helpers.solve_all_at_once(L_q, Z_col)
        del Z_col
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

    if needs_padding:
        return zeta[:, :, :n_zchunk]
    return zeta


def fit_zeta_chunked_to_h5(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    mesh_xy: Mesh,
    chunk_r: int,
    output_file: str,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = True,
    use_gspace_cache: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    isdf_pair_mode: str = "spin_traced",
    k_chunk_size: int = 0,
    q_gather_size: int = 0,
    band_norms: np.ndarray | None = None,
    use_ffi_io: bool = False,
):
    """
    Full zeta fitting pipeline with r-chunk loop and HDF5 output.

    Workflow:
    1. Load wavefunctions (band-chunked FFT) for max range
    2. Slice to get left (0:b3) and right (0:b4) views
    3. Compute C_q from left/right pair density via FFT
    4. Compute L_q = chol(C_q) using 2D blocked algorithm
    5. For each r-chunk:
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
        band_chunk_size: Bands to process at once when FFTing wavefunctions (with global r)
        q_chunk_size: Q-points to solve C_q @ zeta_q = Z_q simultaneously
        bispinor: Whether to use bispinor wavefunctions
        use_gspace_cache: If True, cache G-space across r-chunks
        band_range_left: (start, end) for left wfns. Default: (b0, b3)
        band_range_right: (start, end) for right wfns. Default: (b0, b4)
        isdf_pair_mode: Pair-density mode for CCT/ZCT.
            "spin_traced": P(mu,col)=sum_{n,s} psi* psi (current default)
            "spin_matrix_frobenius": Keep spin channels P_ab and contract sum_ab after FFT

    Returns:
        psi_l_yr:    Left centroid wfns  (nk, nb_l, ns, n_rmu), Y-sharded
        psi_r_yr:    Right centroid wfns (nk, nb_r, ns, n_rmu), Y-sharded
        psi_l_xn:    Left centroid wfns  (nk, ns, n_rmu, nb_l), X-sharded
        psi_r_xn:    Right centroid wfns (nk, ns, n_rmu, nb_r), X-sharded
        peak_bytes:  GPU high-water mark (peak_bytes_in_use) during chunk loop
    """
    import h5py

    nx, ny, nz = meta.fft_grid
    isdf_pair_mode = str(isdf_pair_mode).strip().lower()
    if isdf_pair_mode not in ("spin_traced", "spin_matrix_frobenius"):
        raise ValueError(
            f"Unknown isdf_pair_mode={isdf_pair_mode!r}. "
            "Expected 'spin_traced' or 'spin_matrix_frobenius'."
        )
    n_rmu = meta.n_rmu
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
          f"{nb_full} bands ({nb_left} left + {nb_right} right), {isdf_pair_mode}")
    print(f"  Output: {output_file}")

    # ========== STEP 1: Load wavefunctions at centroids (band-chunked) ==========
    # Load full range, then slice for left and right
    with timing.section("zeta_fit.load_wfns"):
        psi_full_rmu_Y, psi_full_rmuT_X = load_centroids_band_chunked(
            wfn, sym, meta, centroid_indices, bispinor, mesh_xy, band_range_full,
            band_chunk_size=band_chunk_size
        )
        # psi_full shapes: (nk, nb_full, ns, n_rmu) and (nk, n_rmu, nb_full, ns)

        # Slice for left and right wavefunctions
        # Convert global band ranges to local indices in the full array
        # Left: bands [band_range_left[0], band_range_left[1]) relative to full[0]
        # Right: bands [band_range_right[0], band_range_right[1]) relative to full[0]
        l_band_start = band_range_left[0] - band_range_full[0]
        l_band_end = l_band_start + nb_left
        r_band_start = band_range_right[0] - band_range_full[0]
        r_band_end = r_band_start + nb_right

        # NOTE: JAX slices are views sharing memory with the parent array.
        # To allow garbage collection of the full array, we must make
        # the slices independent using jnp.asarray which forces a copy.
        psi_l_rmu_Y = jnp.asarray(psi_full_rmu_Y[:, l_band_start:l_band_end, :, :])
        psi_l_rmuT_X = jnp.asarray(psi_full_rmuT_X[:, :, l_band_start:l_band_end, :])
        psi_r_rmu_Y = jnp.asarray(psi_full_rmu_Y[:, r_band_start:r_band_end, :, :])
        psi_r_rmuT_X = jnp.asarray(psi_full_rmuT_X[:, :, r_band_start:r_band_end, :])

        # Now safe to delete full arrays - slices are independent copies
        del psi_full_rmu_Y, psi_full_rmuT_X

        print(f"  Left wfns:  {psi_l_rmu_Y.shape}")
        print(f"  Right wfns: {psi_r_rmu_Y.shape}")

        # For pseudobands: normalize wavefunctions for ISDF fitting.
        # Divisor is max(1, w_n) — we never amplify a stored state (so
        # low-weight pseudobands retain their sub-unit norm in the fit,
        # preserving DOS weighting), and high-weight pseudobands are
        # brought back down to unit so they don't dominate.  Zero-norm
        # pseudobands (empty/dropped windows) stay zero via the floor.
        if band_norms is not None:
            norms_l = jnp.asarray(band_norms[band_range_left[0]:band_range_left[1]])
            norms_r = jnp.asarray(band_norms[band_range_right[0]:band_range_right[1]])
            norms_l = jnp.maximum(norms_l, 1.0)
            norms_r = jnp.maximum(norms_r, 1.0)
            # psi shapes: Y=(nk, nb, ns, n_rmu), X=(nk, n_rmu, nb, ns)
            psi_l_rmu_Y_fit = psi_l_rmu_Y / norms_l[None, :, None, None]
            psi_l_rmuT_X_fit = psi_l_rmuT_X / norms_l[None, None, :, None]
            psi_r_rmu_Y_fit = psi_r_rmu_Y / norms_r[None, :, None, None]
            psi_r_rmuT_X_fit = psi_r_rmuT_X / norms_r[None, None, :, None]
            n_weighted = int(np.sum(band_norms > 1.01))
            n_zero = int(np.sum(band_norms < 1e-10))
            print(f"  Pseudobands normalization: {n_weighted} weighted, "
                  f"{n_zero} zero-weight (skipped)")
        else:
            psi_l_rmu_Y_fit = psi_l_rmu_Y
            psi_l_rmuT_X_fit = psi_l_rmuT_X
            psi_r_rmu_Y_fit = psi_r_rmu_Y
            psi_r_rmuT_X_fit = psi_r_rmuT_X

    # ========== STEP 2: Compute CCT (C_q) from left/right pair densities ==========
    # Uses normalized copies for fitting (equal-weight pair densities)
    with timing.section("zeta_fit.CCT"):
        print(f"  Computing pair densities P_l, P_r ({isdf_pair_mode})")
        if isdf_pair_mode == "spin_traced":
            P_l_k = compute_pair_density_spin_traced(psi_l_rmuT_X_fit, psi_l_rmu_Y_fit, mesh_xy)
            P_r_k = compute_pair_density_spin_traced(psi_r_rmuT_X_fit, psi_r_rmu_Y_fit, mesh_xy)
            P_l_k.block_until_ready()
            P_r_k.block_until_ready()
            C_q = compute_CCT_from_left_right(P_l_k, P_r_k, kgrid, mesh_xy)
        else:
            P_l_k = compute_pair_density_spin_matrix(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
            P_r_k = compute_pair_density_spin_matrix(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
            P_l_k.block_until_ready()
            P_r_k.block_until_ready()
            C_q = compute_CCT_from_left_right_spin_matrix(P_l_k, P_r_k, kgrid, mesh_xy)
        C_q.block_until_ready()
        # C_q: (nqx, nqy, nqz, n_rmu, n_rmu)

        # Free pair densities - only needed for C_q
        del P_l_k, P_r_k

        # Flatten for Cholesky
        C_q_flat = C_q.reshape(nq, n_rmu, n_rmu)
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)

    # ========== STEP 3: Compute L_q = chol(C_q) once ==========
    with timing.section("zeta_fit.cholesky"):
        print(f"  Computing L_q = chol(C_q)")
        L_q = compute_L_q_from_CCT(C_q_flat, mesh_xy)
        L_q.block_until_ready()
        print(f"  L_q: {L_q.shape}")

    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget
    del C_q, C_q_flat
    with timing.section("zeta_fit.gc_pre_chunk_loop"):
        gc.collect()
        jax.clear_caches()  # Clear JAX function caches that may hold array refs

    # ========== STEP 4: Create HDF5 file ==========
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
    from file_io.slab_io import SlabIO
    nq = nqx * nqy * nqz
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
        if use_ffi_io:
            zeta_io = SlabIO(output_file, mode='w', mesh=mesh_xy,
                             use_ffi_io=True)
            zeta_io.create_dataset(
                'zeta_q',
                shape=(nq, n_rtot, n_rmu),
                dtype=np.complex128,
                chunks=(1, n_rchunk, n_rmu),
            )
        else:
            with SlabIO(output_file, mode='w', mesh=mesh_xy,
                        use_ffi_io=False) as _zeta_create_io:
                _zeta_create_io.create_dataset(
                    'zeta_q',
                    shape=(nq, n_rtot, n_rmu),
                    dtype=np.complex128,
                    chunks=(1, n_rchunk, n_rmu),
                )
            zeta_io = None

    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]

    # Env-var override: forces the slow path (re-read WFN.h5 + re-FFT per
    # r-chunk) even when memory would allow caching.  Useful for probing
    # the scaling regime where wavefunctions don't fit in host memory
    # (multi-TB WFN.h5).  LORRAX_DISABLE_GSPACE_CACHE=1 to enable.
    if os.environ.get("LORRAX_DISABLE_GSPACE_CACHE") == "1":
        use_gspace_cache = False
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

    if use_gspace_cache:
        with timing.section("zeta_fit.cache_gspace"):
            print(f"  Caching G-space wavefunctions for r-chunk loop")
            cached_gspace = load_gspace_for_bands(
                wfn, sym, meta, mesh_xy, band_range_full, bispinor,
                band_chunk_ranges=band_chunk_ranges,
            )
            print(f"  G-space cache: {len(cached_gspace)} band chunks "
                  f"(chunk_size={band_chunk_size}, remainder="
                  f"{(_bfe - _bfs) % band_chunk_size or band_chunk_size})")
    else:
        cached_gspace = None
        print("  G-space cache: disabled")

    # ========== STEP 6: Loop over chunks ==========
    # Track timing for summary (manual perf_counter for detailed breakdown)
    t_load_total = 0.0
    t_pair_total = 0.0
    t_zct_total = 0.0
    t_solve_total = 0.0
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
                # zeta_data is already a host numpy array of shape
                # (q_end-q_start, n_rmu, r_end-r_start) in the order
                # the shard_map produced.  Dataset on disk is
                # (nq, n_rtot, n_rmu) — swap the last two axes at
                # write time.  h5py is happy with a non-contiguous
                # source (stride-swap view); it linearizes internally.
                with h5py.File(output_file, 'a') as f:
                    for i, q_flat in enumerate(range(q_start, q_end)):
                        # zeta_data[i] is (n_rmu, r_chunk) → transpose
                        # to (r_chunk, n_rmu) to match file layout.
                        f['zeta_q'][q_flat, r_start:r_end, :] = zeta_data[i].T
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
    _peak_bytes = 0
    def _track_peak():
        nonlocal _peak_bytes
        try:
            stats = jax.devices()[0].memory_stats()
            if stats:
                _peak_bytes = max(_peak_bytes, stats.get('peak_bytes_in_use', 0))
        except Exception:
            pass

    # Opt-in per-stage peak probe.  When ``LORRAX_MEM_PROBE=1`` is set
    # emits a rank-0 stderr line at each labeled point inside the
    # r-chunk loop: ``[memprobe] label=... chunk=i used=... peak=...``.
    # Zero cost when disabled.  Used to rebuild the memory model by
    # attributing observed peaks to specific chunk-loop stages.
    _memprobe_on = bool(os.environ.get('LORRAX_MEM_PROBE', '')) and jax.process_index() == 0
    _memprobe_chunk_idx = 0
    if _memprobe_on:
        import sys as _sys
        def _probe(label, **extra):
            try:
                s = jax.devices()[0].memory_stats() or {}
                used = s.get('bytes_in_use', 0) / 1e9
                peak = s.get('peak_bytes_in_use', 0) / 1e9
                tag = " ".join(f"{k}={v}" for k, v in extra.items())
                _sys.stderr.write(
                    f"[memprobe] chunk={_memprobe_chunk_idx} "
                    f"label={label:<18s} used={used:6.3f} GB peak={peak:6.3f} GB"
                    f"{' ' + tag if tag else ''}\n")
                _sys.stderr.flush()
            except Exception:
                pass
    else:
        def _probe(label, **extra):
            pass

    from common.progress import LoopProgress
    r_progress = LoopProgress(
        num_chunks, print, title="zeta fitting",
        item_name="r-chunk", max_updates=min(num_chunks, 20))

    # P-accumulator zero-allocator.  Built inside a JIT so XLA
    # allocates directly with the sharded layout — a plain
    # ``jnp.zeros`` at module scope would materialise a fully
    # replicated intermediate on every device (e.g. 19.7 GB/device
    # at Si 10×10×10 with n_rchunk ~2500) before
    # ``with_sharding_constraint`` got a chance to reshard it.
    # Hoisted out of the r-chunk loop and memoised per n_rchunk so
    # the N-1 equally sized r-chunks + the final remainder hit
    # exactly two compile-shape entries, with no new jit-wrapper
    # identity per r-chunk iter.
    ns = meta.nspinor
    P_sharding_traced = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    P_sharding_spin   = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
    if isdf_pair_mode == "spin_traced":
        _P_sharding = P_sharding_traced
        _P_shape_fn = lambda n_r: (nk_tot, n_rmu, n_r)
        _accum = accumulate_pair_density_spin_traced
    else:
        _P_sharding = P_sharding_spin
        _P_shape_fn = lambda n_r: (nk_tot, ns, ns, n_rmu, n_r)
        _accum = accumulate_pair_density_spin_matrix
    _P_zeros_cache: dict = {}

    def _zero_P(n_r: int):
        fn = _P_zeros_cache.get(n_r)
        if fn is None:
            shape = _P_shape_fn(n_r)
            fn = jax.jit(
                lambda: jnp.zeros(shape, dtype=jnp.complex128),
                out_shardings=_P_sharding)
            _P_zeros_cache[n_r] = fn
        return fn()

    _probe("before_chunk_loop")
    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_chunks):
            if _memprobe_on:
                _memprobe_chunk_idx = chunk_idx
            r_start = chunk_idx * chunk_r
            r_end = min(r_start + chunk_r, n_rtot)
            actual_n_rchunk = r_end - r_start
            _probe("chunk_start", r_start=r_start, r_end=r_end)

            # 6ab. Stream band chunks of the r-chunk wfns and
            # accumulate P_l / P_r incrementally.  At any moment only
            # one band chunk's r-chunk shard is live (instead of the
            # full-band-range tensor) — decouples the pair-density
            # peak from the band count.  Chunks are uniform size
            # ``band_chunk_size`` (N-1 of them) + one remainder, so
            # the read/FFT pipeline and the pair-density einsum see
            # exactly TWO compile shapes.  Chunks fully past an L/R
            # endpoint skip the corresponding einsum; the one chunk
            # that straddles an endpoint gets zero-padded on the L
            # side so its einsum still dispatches at ``bc_size``.
            P_l_k_mux = _zero_P(actual_n_rchunk)
            P_r_k_mux = _zero_P(actual_n_rchunk)

            # Per-chunk L/R slicing.  For a chunk fully inside an L or
            # R range, ``_slice_and_norm`` returns a direct view of the
            # centroid tensor — no extra allocation, no padding, so
            # the downstream einsum hits the same JIT cache entry as
            # the N-1 full-size chunks.  For the at-most-one-per-
            # endpoint straddle chunk, it zero-pads out-of-range
            # bands into a fresh tensor so the einsum shape is still
            # uniform ``bc_size``.  Compile cost: the hot einsum is
            # TWO shapes ({B, remainder}); the zero-pad op compiles
            # once per unique straddle geometry (≤ one per endpoint).
            L_slice_shard = NamedSharding(
                mesh_xy, P(None, 'x', None, None))

            def _slice_and_norm(psi_fit, norms_fit, range_abs,
                                bc_lo, bc_hi):
                """Return (psi_L_bc, norm_bc) for this band chunk, both
                zero/one-padded if the chunk straddles ``range_abs``.
                ``psi_L_bc`` has shape ``(nk, nrmu, bc_hi-bc_lo, ns)``;
                ``norm_bc`` has shape ``(bc_hi-bc_lo,)`` with 1.0
                outside the overlap so dividing ``psi_bc_Y`` by it
                leaves out-of-range bands unchanged (psi_L's zero
                entries kill their contribution in the einsum
                anyway).  Returns ``(None, None)`` when the chunk is
                entirely outside ``range_abs``.
                """
                rs, re = range_abs
                ol_lo = max(bc_lo, rs)
                ol_hi = min(bc_hi, re)
                if ol_hi <= ol_lo:
                    return None, None
                bc_size = bc_hi - bc_lo
                # Fast path: chunk fully inside range, direct slice.
                if ol_lo == bc_lo and ol_hi == bc_hi:
                    psi_L_bc = psi_fit[:, :, (bc_lo - rs):(bc_hi - rs), :]
                    norm_bc = (norms_fit[(bc_lo - rs):(bc_hi - rs)]
                               if norms_fit is not None else None)
                    return psi_L_bc, norm_bc
                # Straddle: zero-pad psi_L and one-pad norm to bc_size.
                ns = psi_fit.shape[-1]
                psi_L_bc = jnp.zeros(
                    (nk_tot, n_rmu, bc_size, ns),
                    dtype=jnp.complex128)
                psi_L_bc = jax.lax.with_sharding_constraint(
                    psi_L_bc, L_slice_shard)
                psi_L_bc = psi_L_bc.at[
                    :, :, (ol_lo - bc_lo):(ol_hi - bc_lo), :].set(
                    psi_fit[:, :, (ol_lo - rs):(ol_hi - rs), :])
                if norms_fit is not None:
                    norm_bc = jnp.ones((bc_size,), dtype=jnp.float64)
                    norm_bc = norm_bc.at[
                        (ol_lo - bc_lo):(ol_hi - bc_lo)].set(
                        norms_fit[(ol_lo - rs):(ol_hi - rs)])
                else:
                    norm_bc = None
                return psi_L_bc, norm_bc

            norms_l_used = norms_l if band_norms is not None else None
            norms_r_used = norms_r if band_norms is not None else None

            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.load_and_pair_density"):
                for bc_range, psi_bc_Y in iter_psi_rchunk_bandwise(
                    wfn, sym, meta, mesh_xy, band_range_full,
                    r_start, r_end, bispinor,
                    band_chunk_size=band_chunk_size,
                    k_chunk_size=k_chunk_size,
                    band_chunk_ranges=band_chunk_ranges,
                    cached_gspace=cached_gspace, kvecs_frac=kvecs_frac,
                ):
                    bc_lo, bc_hi = bc_range

                    # Left contribution
                    psi_L_bc, norm_bc = _slice_and_norm(
                        psi_l_rmuT_X_fit, norms_l_used,
                        band_range_left, bc_lo, bc_hi)
                    if psi_L_bc is not None:
                        psi_R_bc = (psi_bc_Y / norm_bc[None, :, None, None]
                                    if norm_bc is not None else psi_bc_Y)
                        P_l_k_mux = _accum(P_l_k_mux, psi_L_bc, psi_R_bc, mesh_xy)

                    # Right contribution
                    psi_L_bc, norm_bc = _slice_and_norm(
                        psi_r_rmuT_X_fit, norms_r_used,
                        band_range_right, bc_lo, bc_hi)
                    if psi_L_bc is not None:
                        psi_R_bc = (psi_bc_Y / norm_bc[None, :, None, None]
                                    if norm_bc is not None else psi_bc_Y)
                        P_r_k_mux = _accum(P_r_k_mux, psi_L_bc, psi_R_bc, mesh_xy)

                    del psi_bc_Y

                P_l_k_mux.block_until_ready()
                _track_peak()
                _probe("after_pair")
                # No block_until_ready on P_r — ZCT will consume it
                # asynchronously.  P_l's block is needed to bound
                # t_pair_total and to gate _track_peak.
            # Load and pair-density are fused into one streaming loop
            # now — accumulate the combined wall into t_pair_total.
            # t_load_total stays at 0 so the two-column "load vs pair"
            # breakdown below doesn't double-count; the end-of-run
            # table prints pair-density as the merged total.
            t_pair_total += time.perf_counter() - t0

            # 6c. Compute Z_q via left/right cross-product FFT.
            # P_l_k_mux / P_r_k_mux stay flat-k throughout — the 3-D k-grid
            # only appears inside the FFT helper.  The ZCT kernels return
            # flat-q (nq, μ, z-chunk) matching the input sharding.
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.ZCT"):
                if isdf_pair_mode == "spin_traced":
                    Z_q_flat = compute_ZCT_from_left_right_zchunk(
                        P_l_k_mux, P_r_k_mux, kgrid, mesh_xy
                    )
                else:
                    Z_q_flat = compute_ZCT_from_left_right_zchunk_spin_matrix(
                        P_l_k_mux, P_r_k_mux, kgrid, mesh_xy
                    )
                # P_l/P_r are consumed by the ZCT jits (donate_argnums);
                # no-op del for name-scope hygiene.
                del P_l_k_mux, P_r_k_mux

                Z_q_flat = jax.lax.with_sharding_constraint(Z_q_flat, flat_shard)
                # No block_until_ready — reshard will consume Z_q_flat asynchronously.
            t_zct_total += time.perf_counter() - t0

            # 6d. Reshard Z_q_flat → Z_col, then free Z_q_flat BEFORE the solve
            # q-loop.  If we pass Z_q_flat into solve_zeta_from_L_q, the caller's
            # reference survives during the entire solve loop, keeping 3× m_zcol
            # alive (Z_q_flat + Z_col + zeta) instead of 2× (Z_col + zeta).
            z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
            Z_col = jax.lax.with_sharding_constraint(Z_q_flat, z_col_shard)
            Z_col.block_until_ready()
            _track_peak()
            _probe("after_zct_reshard")
            del Z_q_flat
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.solve"), jax_profile.step_annotation("chunk_solve", step_num=chunk_idx):
                zeta_chunk = solve_zeta_from_L_q(L_q, Z_col, mesh_xy, q_chunk_size)
                zeta_chunk.block_until_ready()
                _track_peak()
                _probe("after_solve")
                del Z_col
            t_solve_total += time.perf_counter() - t0

            # 6e. Q-chunked allgather → host copy → async HDF5 write.
            # The allgather replicates zeta slices: per-device output is
            # (q_gather, n_rmu, chunk_r) which at large chunk_r can be huge.
            # Chunking over q keeps each allgather under memory limits.
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                # Each allgather produces a FULLY REPLICATED output per device:
                # q_gather × n_rmu × chunk_r × 16 bytes, plus NCCL temp of same size.
                # Cap to keep replicated output + NCCL under available memory.
                _bytes_per_q_replicated = 2 * n_rmu * actual_n_rchunk * 16
                _safe_q_gather = max(1, min(nq, int(10 * 1024**3 / max(1, _bytes_per_q_replicated))))
                if q_gather_size > 0:
                    _q_gather = min(nq, q_gather_size, _safe_q_gather)
                else:
                    _q_gather = _safe_q_gather

                if use_ffi_io:
                    # FFI path: zeta_chunk is (nq, n_rmu, chunk_r),
                    # dataset is (nq, n_rtot, n_rmu) — transpose the
                    # last two axes before writing so the slab matches
                    # the disk layout.  The transpose is a JAX
                    # metadata-only operation (shard axis 'chunk_r' /
                    # sharded → axis 1 of the post-transpose tensor;
                    # NamedSharding updates in place, no data motion).
                    # Per-rank write pattern goes from ~120 K small
                    # strips (old (nq, n_rmu, n_rtot) layout) to 1000
                    # fat contiguous strips (one per q, full n_rmu +
                    # rank's n_rchunk/4 rows).
                    zeta_chunk_write = zeta_chunk.transpose(0, 2, 1)
                    zeta_io.write_slab(
                        'zeta_q', zeta_chunk_write,
                        offset=(0, r_start, 0),
                        global_shape=(nq, n_rtot, n_rmu))
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
                    for _q0 in range(0, nq, _q_gather):
                        _q1 = min(_q0 + _q_gather, nq)
                        _slice = zeta_chunk[_q0:_q1]
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
                del zeta_chunk
            t_write_total += time.perf_counter() - t0
            _probe("after_h5_write_dispatch")
            r_progress.step()


    t_chunks_total = time.perf_counter() - t_chunk_start
    r_progress.finish()
    _probe("after_chunk_loop")

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

    # Free cached G-space now that chunk loop is done
    if cached_gspace is not None:
        del cached_gspace

    # Per-stage timing breakdown
    print(f"  Zeta output: {output_file}  shape: ({nqx},{nqy},{nqz},{n_rmu},{n_rtot})")
    print(f"  Timing ({num_chunks} r-chunks, {t_chunks_total:.1f}s total):")
    for label, t in [("load", t_load_total), ("pair", t_pair_total),
                     ("ZCT", t_zct_total), ("solve", t_solve_total), ("H5", t_write_total)]:
        print(f"    {label:<6} {t:6.2f}s  {100*t/t_chunks_total:4.1f}%")

    # Return left/right centroid wavefunctions + peak memory high-water mark.
    return psi_l_rmu_Y, psi_r_rmu_Y, psi_l_rmuT_X, psi_r_rmuT_X, _peak_bytes

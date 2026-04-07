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
    lim = s.get('bytes_limit', 0) / 1e9
    if jax.process_index() == 0:
        print(f'  [MEM isdf | {label}] used={u:.3f} peak={p:.3f} limit={lim:.1f} GB',
              flush=True)
from common import jax_profile
from .cholesky_2d import (
    cholesky_2d_batched,
    dense_to_tiles,
    tiles_to_dense,
)
from .fft_helpers import (
    make_sharded_ifftn_3d,
    make_sharded_fftn_3d,
    compute_block_size_for_2d_cholesky,
)
from .load_wfns import (
    read_Gvecs_to_devices,
    get_psi_rchunk_from_cached,
    get_psi_rchunk,
    load_centroids_band_chunked,
)


def load_gspace_for_bands(
    wfn, sym, meta, mesh_xy, band_range, bispinor,
    band_chunk_size: int = 16,
) -> list[tuple[jax.Array, tuple[int, int]]]:
    """Load G-space wavefunctions for all band chunks ONCE.

    Caches the expensive HDF5 read + scatter so it can be reused across
    multiple r-chunk iterations. Returns a list of (psi_Gtot, band_range)
    tuples, one per band chunk.
    """
    b_start, b_end = band_range
    nb_total = b_end - b_start
    num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
    cached_gspace = []
    for bc_idx in range(num_band_chunks):
        bc_start = b_start + bc_idx * band_chunk_size
        bc_end = min(bc_start + band_chunk_size, b_end)
        bc_range = (bc_start, bc_end)
        global_psi_Gtot, _ = read_Gvecs_to_devices(wfn, sym, bc_range, meta, bispinor, mesh_xy)
        cached_gspace.append((global_psi_Gtot, bc_range))
    return cached_gspace


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
		kgrid: (nkx, nky, nkz)
		mesh_xy: Device mesh

	Returns:
		C_q: (nqx, nqy, nqz, n_rmu, n_rmu) with P(None, None, None, 'x', 'y')
	"""
	nkx, nky, nkz = kgrid
	nk, n_rmu, _ = P_l_k.shape

	cache_key = ('CCT_LR', id(mesh_xy), nk, n_rmu, nkx)

	if cache_key not in _isdf_pipeline_cache:
		in_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		# Keep pair-density in (k, mu, nu) layout (good for einsum/matmul contiguity),
		# and FFT directly over the leading k-grid axes.
		fft_in = P(None, None, None, 'x', 'y')
		sharded_ifftn = make_sharded_ifftn_3d(mesh_xy, fft_in, fft_in, norm='forward', axes=(0, 1, 2))
		sharded_fftn = make_sharded_fftn_3d(mesh_xy, fft_in, fft_in, norm='forward', axes=(0, 1, 2))

		@partial(jax.jit, in_shardings=(in_xy, in_xy), out_shardings=out_xy,
		         static_argnames=('nkx', 'nky', 'nkz'))
		def _compute_CCT_LR(P_l: jax.Array, P_r: jax.Array,
		                    nkx: int, nky: int, nkz: int) -> jax.Array:
			# Reshape to 3D k-grid: (nk, μ, ν) -> (nkx, nky, nkz, μ, ν)
			P_l_3d = P_l.reshape(nkx, nky, nkz, n_rmu, n_rmu)
			P_r_3d = P_r.reshape(nkx, nky, nkz, n_rmu, n_rmu)

			# Use norm='forward' for BOTH IFFT and FFT to match direct k-sum.
			# With forward: IFFT is unscaled (sum), FFT divides by N.
			# Convolution theorem: C_q = FFT(IFFT(A)* ⊙ IFFT(B))
			# This gives C_q = Σ_k A*_k B_{k+q} (matches gw_jax direct sum)
			P_l_Rt = sharded_ifftn(P_l_3d)
			P_r_Rt = sharded_ifftn(P_r_3d)

			# Cross-product: C_R = conj(P_l_R) * P_r_R (element-wise)
			C_Rt = jnp.conj(P_l_Rt) * P_r_Rt

			# FFT to q-space
			C_qt = sharded_fftn(C_Rt)
			return C_qt

		_isdf_pipeline_cache[cache_key] = _compute_CCT_LR

	return _isdf_pipeline_cache[cache_key](P_l_k, P_r_k, nkx, nky, nkz)


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
		in_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		fft_spin = P(None, None, None, None, None, 'x', 'y')
		fft_scalar = P(None, None, None, 'x', 'y')
		sharded_ifftn_spin = make_sharded_ifftn_3d(mesh_xy, fft_spin, fft_spin, norm='forward', axes=(0, 1, 2))
		sharded_fftn_scalar = make_sharded_fftn_3d(mesh_xy, fft_scalar, fft_scalar, norm='forward', axes=(0, 1, 2))

		@partial(jax.jit, in_shardings=(in_xy, in_xy), out_shardings=out_xy,
		         static_argnames=('nkx', 'nky', 'nkz'))
		def _compute_CCT_LR_spin(P_l: jax.Array, P_r: jax.Array,
		                         nkx: int, nky: int, nkz: int) -> jax.Array:
			P_l_3d = P_l.reshape(nkx, nky, nkz, ns1, ns2, n_rmu, n_rmu)
			P_r_3d = P_r.reshape(nkx, nky, nkz, ns1, ns2, n_rmu, n_rmu)
			P_l_Rt = sharded_ifftn_spin(P_l_3d)
			P_r_Rt = sharded_ifftn_spin(P_r_3d)
			C_Rt = jnp.sum(jnp.conj(P_l_Rt) * P_r_Rt, axis=(3, 4))
			return sharded_fftn_scalar(C_Rt)

		_isdf_pipeline_cache[cache_key] = _compute_CCT_LR_spin

	return _isdf_pipeline_cache[cache_key](P_l_k_ab, P_r_k_ab, nkx, nky, nkz)


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
		P_l_k_muz: (nkx, nky, nkz, n_rmu, n_zchunk) left pair density in k-grid form
		          with P(None, None, None, 'x', 'y')
		P_r_k_muz: (nkx, nky, nkz, n_rmu, n_zchunk) right pair density in k-grid form
		          with P(None, None, None, 'x', 'y')
		kgrid: (nkx, nky, nkz)
		mesh_xy: Device mesh

	Returns:
		Z_q: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, 'x', 'y')
	"""
	nkx, nky, nkz = kgrid
	n_rmu, n_zchunk = P_l_k_muz.shape[3], P_l_k_muz.shape[4]
	assert P_l_k_muz.shape[:3] == (nkx, nky, nkz), (
		f"P_l_k_muz leading k-grid dims {P_l_k_muz.shape[:3]} do not match {kgrid}"
	)
	assert P_r_k_muz.shape[:3] == (nkx, nky, nkz), (
		f"P_r_k_muz leading k-grid dims {P_r_k_muz.shape[:3]} do not match {kgrid}"
	)

	cache_key = ('ZCT_LR', id(mesh_xy), nkx, nky, nkz, n_rmu, n_zchunk)

	if cache_key not in _isdf_pipeline_cache:
		out_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		# Keep pair-density in (k, mu, z) layout (good for einsum/matmul contiguity),
		# and FFT directly over the leading k-grid axes.
		fft_in = P(None, None, None, 'x', 'y')
		fft_shard = NamedSharding(mesh_xy, fft_in)
		sharded_ifftn = make_sharded_ifftn_3d(mesh_xy, fft_in, fft_in, norm='forward', axes=(0, 1, 2))
		sharded_fftn = make_sharded_fftn_3d(mesh_xy, fft_in, fft_in, norm='forward', axes=(0, 1, 2))

		@partial(jax.jit, in_shardings=fft_shard, out_shardings=fft_shard, donate_argnums=(0,))
		def _left_ifft_conj(P_l_3d: jax.Array) -> jax.Array:
			# Use norm='forward' to match direct k-sum convention.
			return jnp.conj(sharded_ifftn(P_l_3d))

		@partial(jax.jit, in_shardings=(fft_shard, fft_shard), out_shardings=out_xy, donate_argnums=(0,))
		def _right_ifft_mul_fft(P_r_3d: jax.Array, P_l_Rt: jax.Array) -> jax.Array:
			# Keep R-side intermediate internal to avoid materializing both Rt arrays at API boundary.
			P_r_Rt = sharded_ifftn(P_r_3d)
			Z_Rt = P_l_Rt * P_r_Rt
			return sharded_fftn(Z_Rt)

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
	assert P_l_k_ab_muz.shape[:3] == (nkx, nky, nkz), (
		f"P_l_k_ab_muz leading k-grid dims {P_l_k_ab_muz.shape[:3]} do not match {kgrid}"
	)
	assert P_r_k_ab_muz.shape[:3] == (nkx, nky, nkz), (
		f"P_r_k_ab_muz leading k-grid dims {P_r_k_ab_muz.shape[:3]} do not match {kgrid}"
	)
	n_s1, n_s2, n_rmu, n_zchunk = (
		P_l_k_ab_muz.shape[3],
		P_l_k_ab_muz.shape[4],
		P_l_k_ab_muz.shape[5],
		P_l_k_ab_muz.shape[6],
	)

	cache_key = ('ZCT_LR_spin_matrix', id(mesh_xy), nkx, nky, nkz, n_s1, n_s2, n_rmu, n_zchunk)

	if cache_key not in _isdf_pipeline_cache:
		fft_spin = P(None, None, None, None, None, 'x', 'y')
		fft_scalar = P(None, None, None, 'x', 'y')
		fft_shard_spin = NamedSharding(mesh_xy, fft_spin)
		out_xy = NamedSharding(mesh_xy, fft_scalar)
		sharded_ifftn_spin = make_sharded_ifftn_3d(mesh_xy, fft_spin, fft_spin, norm='forward', axes=(0, 1, 2))
		sharded_fftn_scalar = make_sharded_fftn_3d(mesh_xy, fft_scalar, fft_scalar, norm='forward', axes=(0, 1, 2))

		@partial(jax.jit, in_shardings=fft_shard_spin, out_shardings=fft_shard_spin, donate_argnums=(0,))
		def _left_ifft_conj(P_l_3d: jax.Array) -> jax.Array:
			return jnp.conj(sharded_ifftn_spin(P_l_3d))

		@partial(jax.jit, in_shardings=(fft_shard_spin, fft_shard_spin), out_shardings=out_xy, donate_argnums=(0,))
		def _right_ifft_contract_fft(P_r_3d: jax.Array, P_l_Rt: jax.Array) -> jax.Array:
			P_r_Rt = sharded_ifftn_spin(P_r_3d)
			Z_Rt = jnp.sum(P_l_Rt * P_r_Rt, axis=(3, 4))
			return sharded_fftn_scalar(Z_Rt)

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
    _mem_report(f"solve: before q-loop (q_batch={q_batch}, nq={nq})")

    for q0 in range(0, nq_padded, q_batch):
        q1 = q0 + q_batch
        zeta = helpers.solve_batch_and_update(L_q[q0:q1], Z_col[q0:q1], zeta, q0)

    zeta.block_until_ready()
    _mem_report(f"solve: after q-loop")
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
        psi_l_rmu_Y: Left centroid wfns (nk, nb_l, ns, n_rmu), Y-sharded
        psi_l_rmuT_X: Left conjugated wfns (nk, n_rmu, nb_l, ns), X-sharded
        psi_r_rmu_Y: Right centroid wfns (nk, nb_r, ns, n_rmu), Y-sharded
        psi_r_rmuT_X: Right conjugated wfns (nk, n_rmu, nb_r, ns), X-sharded
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

    print(f"\n{'='*60}")
    print(f"Zeta fitting: {num_chunks} r-chunks, {n_rchunk} points each")
    print(f"  Left bands:  {band_range_left} ({nb_left} bands)")
    print(f"  Right bands: {band_range_right} ({nb_right} bands)")
    print(f"  Full range:  {band_range_full} ({nb_full} bands)")
    print(f"  Pair mode:   {isdf_pair_mode}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    _mem_report("before ISDF fitting")

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
    _mem_report("after centroid load + slice")

    # ========== STEP 2: Compute CCT (C_q) from left/right pair densities ==========
    with timing.section("zeta_fit.CCT"):
        if isdf_pair_mode == "spin_traced":
            print("\nComputing spin-traced pair densities P_l and P_r...")
            P_l_k = compute_pair_density_spin_traced(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
            P_r_k = compute_pair_density_spin_traced(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
            P_l_k.block_until_ready()
            P_r_k.block_until_ready()
            print("Computing C_q from left/right cross-product...")
            C_q = compute_CCT_from_left_right(P_l_k, P_r_k, kgrid, mesh_xy)
        else:
            print("\nComputing spin-matrix pair densities P_l,ab and P_r,ab...")
            P_l_k = compute_pair_density_spin_matrix(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
            P_r_k = compute_pair_density_spin_matrix(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
            P_l_k.block_until_ready()
            P_r_k.block_until_ready()
            print("Computing C_q from spin-channel-contracted left/right cross-product...")
            C_q = compute_CCT_from_left_right_spin_matrix(P_l_k, P_r_k, kgrid, mesh_xy)
        C_q.block_until_ready()
        # C_q: (nqx, nqy, nqz, n_rmu, n_rmu)

        # Free pair densities - only needed for C_q
        del P_l_k, P_r_k

        # Flatten for Cholesky
        C_q_flat = C_q.reshape(nq, n_rmu, n_rmu)
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)

    _mem_report("after CCT")

    # ========== STEP 3: Compute L_q = chol(C_q) once ==========
    with timing.section("zeta_fit.cholesky"):
        print("\nComputing L_q = chol(C_q) using 2D blocked algorithm...")
        L_q = compute_L_q_from_CCT(C_q_flat, mesh_xy)
        L_q.block_until_ready()
        print(f"  L_q shape: {L_q.shape}")

    _mem_report("after cholesky (before del C_q)")

    # Free C_q to reclaim GPU memory before z-chunk loop
    del C_q, C_q_flat
    gc.collect()
    jax.clear_caches()

    _mem_report("after cholesky + del C_q")

    # ========== STEP 4: Create HDF5 file ==========
    # Only rank 0 creates the file structure
    if jax.process_index() == 0:
        with h5py.File(output_file, 'w') as f:
            # Create dataset for full zeta
            f.create_dataset(
                'zeta_q',
                shape=(nqx, nqy, nqz, n_rmu, n_rtot),
                dtype=np.complex128,
                chunks=(1, 1, 1, n_rmu, n_rchunk),  # Chunk by r-slice (contiguous in r!)
            )
            # Store metadata
            f.attrs['n_rmu'] = n_rmu
            f.attrs['n_rtot'] = n_rtot
            f.attrs['fft_grid'] = meta.fft_grid
            f.attrs['kgrid'] = kgrid
            f.attrs['chunk_mode'] = 'r'
            f.attrs['r_chunk_size'] = chunk_r
            f.attrs['num_r_chunks'] = num_chunks
            f.attrs['isdf_pair_mode'] = isdf_pair_mode

    # Synchronize before writing
    jax.experimental.multihost_utils.sync_global_devices("zeta_h5_create")

    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]

    if use_gspace_cache:
        with timing.section("zeta_fit.cache_gspace"):
            print("\nCaching G-space wavefunctions for r-chunk loop...")
            cached_gspace = load_gspace_for_bands(
                wfn, sym, meta, mesh_xy, band_range_full, bispinor, band_chunk_size
            )
            print(f"  Cached {len(cached_gspace)} band chunks (sharded across devices)")
    else:
        cached_gspace = None
        print("\nG-space caching DISABLED (too large for memory budget)")
        print("  Will reload from HDF5 each r-chunk (slower)")

    _mem_report("after G-cache load (before chunk loop)")

    # ========== STEP 6: Loop over chunks ==========
    # Track timing for summary (manual perf_counter for detailed breakdown)
    t_load_total = 0.0
    t_pair_total = 0.0
    t_zct_total = 0.0
    t_solve_total = 0.0
    t_write_total = 0.0
    t_chunk_start = time.perf_counter()

    # Setup async writer thread for overlapped I/O
    # This allows GPU computation to continue while HDF5 writes happen in background
    write_queue = queue.Queue()
    write_error = [None]  # Mutable container to capture errors from writer thread

    def writer_worker():
        """Background thread that processes HDF5 writes from the queue.

        R-chunking advantage: r-slices are contiguous in the flattened xyz
        index, enabling a single sequential HDF5 write per chunk.
        """
        try:
            while True:
                item = write_queue.get()
                if item is None:  # Poison pill signals shutdown
                    break
                # Support both full-q and partial-q (q-chunked gather) formats
                if len(item) == 6:
                    zeta_data, r_start, r_end, chunk_id, q_start, q_end = item
                else:
                    zeta_data, r_start, r_end, chunk_id = item
                    q_start, q_end = 0, zeta_data.shape[0]
                # zeta_data: (n_q_slice, n_rmu, actual_n_rchunk)
                # R-chunks are contiguous in r-space, so we can write directly!

                with h5py.File(output_file, 'a') as f:
                    for i, q_flat in enumerate(range(q_start, q_end)):
                        qx = q_flat // (nqy * nqz)
                        qy = (q_flat % (nqy * nqz)) // nqz
                        qz = q_flat % nqz
                        f['zeta_q'][qx, qy, qz, :, r_start:r_end] = zeta_data[i]
                write_queue.task_done()
        except Exception as e:
            write_error[0] = e
            write_queue.task_done()

    # Start writer thread (only on rank 0)
    writer_thread = None
    if jax.process_index() == 0:
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        writer_thread.start()

    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_chunks):
            r_start = chunk_idx * chunk_r
            r_end = min(r_start + chunk_r, n_rtot)
            actual_n_rchunk = r_end - r_start
            print(f"Chunk {chunk_idx+1}/{num_chunks}: r=[{r_start}:{r_end}]")

            _mem_report(f"chunk[{chunk_idx}]: start")

            # 6a. Get psi_nk,a(r_chunk) for FULL band range
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.load"):
                if cached_gspace is not None:
                    psi_chunk_Y = get_psi_rchunk_from_cached(
                        cached_gspace, meta, mesh_xy, band_range_full,
                        r_start, r_end, kvecs_frac,
                        band_chunk_size=band_chunk_size
                    )
                else:
                    psi_chunk_Y = get_psi_rchunk(
                        wfn, sym, meta, mesh_xy, band_range_full,
                        r_start, r_end, bispinor,
                        band_chunk_size=band_chunk_size,
                        k_chunk_size=k_chunk_size,
                    )
                psi_chunk_Y.block_until_ready()

                # Slice for left and right (same logic as centroids)
                psi_l_chunk_Y = psi_chunk_Y[:, l_band_start:l_band_end, :, :]
                psi_r_chunk_Y = psi_chunk_Y[:, r_band_start:r_band_end, :, :]
            t_load_total += time.perf_counter() - t0
            _mem_report(f"chunk[{chunk_idx}]: after load (6a)")

            # 6b. Compute left/right pair densities
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.pair_density"):
                if isdf_pair_mode == "spin_traced":
                    # Left: P_l_k(μ, r) = Σ_{n,s} ψ*_l(μ) ψ_l(r)
                    P_l_k_mux = compute_pair_density_spin_traced(
                        psi_l_rmuT_X, psi_l_chunk_Y, mesh_xy
                    )
                    # Right: P_r_k(μ, r) = Σ_{n,s} ψ*_r(μ) ψ_r(r)
                    P_r_k_mux = compute_pair_density_spin_traced(
                        psi_r_rmuT_X, psi_r_chunk_Y, mesh_xy
                    )
                else:
                    # Keep spin channels explicit and contract only after IFFT.
                    P_l_k_mux = compute_pair_density_spin_matrix(
                        psi_l_rmuT_X, psi_l_chunk_Y, mesh_xy
                    )
                    P_r_k_mux = compute_pair_density_spin_matrix(
                        psi_r_rmuT_X, psi_r_chunk_Y, mesh_xy
                    )
                P_l_k_mux.block_until_ready()
                # No block_until_ready on P_r — ZCT will consume it asynchronously.
                # P_l's block_until_ready (above) is needed for del psi_l_chunk_Y.
            t_pair_total += time.perf_counter() - t0

            _mem_report(f"chunk[{chunk_idx}]: after pair density (6b)")

            # Free psi_chunk arrays - we have P_k now
            del psi_chunk_Y, psi_l_chunk_Y, psi_r_chunk_Y

            # Reshape to explicit 3D k-grid before ZCT kernels so donation-compatible
            # stages see consistent input/output ranks and sharding.
            if isdf_pair_mode == "spin_traced":
                fft_chunk_shard = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
                P_l_k_mux = jax.lax.with_sharding_constraint(
                    P_l_k_mux.reshape(nqx, nqy, nqz, n_rmu, actual_n_rchunk),
                    fft_chunk_shard,
                )
                P_r_k_mux = jax.lax.with_sharding_constraint(
                    P_r_k_mux.reshape(nqx, nqy, nqz, n_rmu, actual_n_rchunk),
                    fft_chunk_shard,
                )
            else:
                n_s = meta.nspinor
                fft_chunk_spin_shard = NamedSharding(mesh_xy, P(None, None, None, None, None, 'x', 'y'))
                P_l_k_mux = jax.lax.with_sharding_constraint(
                    P_l_k_mux.reshape(nqx, nqy, nqz, n_s, n_s, n_rmu, actual_n_rchunk),
                    fft_chunk_spin_shard,
                )
                P_r_k_mux = jax.lax.with_sharding_constraint(
                    P_r_k_mux.reshape(nqx, nqy, nqz, n_s, n_s, n_rmu, actual_n_rchunk),
                    fft_chunk_spin_shard,
                )

            # 6c. Compute Z_q via left/right cross-product FFT
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.ZCT"):
                if isdf_pair_mode == "spin_traced":
                    Z_q = compute_ZCT_from_left_right_zchunk(
                        P_l_k_mux, P_r_k_mux, kgrid, mesh_xy
                    )
                else:
                    Z_q = compute_ZCT_from_left_right_zchunk_spin_matrix(
                        P_l_k_mux, P_r_k_mux, kgrid, mesh_xy
                    )
                # No block_until_ready on Z_q — reshape consumes it asynchronously.
                # P_l/P_r become unreferenced and will be freed by XLA when Z_q
                # computation completes (they're consumed by the ZCT JIT).
                del P_l_k_mux, P_r_k_mux

                Z_q_flat = Z_q.reshape(nq, n_rmu, actual_n_rchunk)
                del Z_q
                Z_q_flat = jax.lax.with_sharding_constraint(Z_q_flat, flat_shard)
            t_zct_total += time.perf_counter() - t0
            _mem_report(f"chunk[{chunk_idx}]: after ZCT (6c)")

            # 6d. Reshard Z_q_flat → Z_col, then free Z_q_flat BEFORE the solve
            # q-loop.  If we pass Z_q_flat into solve_zeta_from_L_q, the caller's
            # reference survives during the entire solve loop, keeping 3× m_zcol
            # alive (Z_q_flat + Z_col + zeta) instead of 2× (Z_col + zeta).
            z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
            Z_col = jax.lax.with_sharding_constraint(Z_q_flat, z_col_shard)
            Z_col.block_until_ready()
            del Z_q_flat
            _mem_report(f"chunk[{chunk_idx}]: after reshard Z (6d)")
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.solve"), jax_profile.step_annotation("chunk_solve", step_num=chunk_idx):
                zeta_chunk = solve_zeta_from_L_q(L_q, Z_col, mesh_xy, q_chunk_size)
                zeta_chunk.block_until_ready()
                del Z_col
            t_solve_total += time.perf_counter() - t0
            _mem_report(f"chunk[{chunk_idx}]: after solve (6e)")

            # 6e. Q-chunked allgather → host copy → async HDF5 write.
            # The allgather replicates zeta slices: per-device output is
            # (q_gather, n_rmu, chunk_r) which at large chunk_r can be huge.
            # Chunking over q keeps each allgather under memory limits.
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                # Each allgather produces a FULLY REPLICATED output per device:
                # q_gather × n_rmu × chunk_r × 16 bytes, plus NCCL temp of same size.
                # Cap to keep replicated output + NCCL under available memory.
                _bytes_per_q_replicated = 2 * n_rmu * actual_n_rchunk * 16  # output + NCCL
                _safe_q_gather = max(1, min(nq, int(10 * 1024**3 / max(1, _bytes_per_q_replicated))))
                if q_gather_size > 0:
                    _q_gather = min(nq, q_gather_size, _safe_q_gather)
                else:
                    _q_gather = _safe_q_gather

                for _q0 in range(0, nq, _q_gather):
                    _q1 = min(_q0 + _q_gather, nq)
                    _slice = zeta_chunk[_q0:_q1]
                    _gathered = jax.experimental.multihost_utils.process_allgather(_slice, tiled=False)
                    if jax.process_index() == 0:
                        _g = np.asarray(_gathered)
                        if _g.ndim == 4 and _g.shape[0] == 1:
                            _g = _g[0]
                        write_queue.put((_g, r_start, r_end, chunk_idx, _q0, _q1))
                    del _gathered, _slice

                jax.experimental.multihost_utils.sync_global_devices(f"zeta_chunk_{chunk_idx}")
                del zeta_chunk
            t_write_total += time.perf_counter() - t0
            _mem_report(f"chunk[{chunk_idx}]: after gather+write (6f)")


    t_chunks_total = time.perf_counter() - t_chunk_start

    # Wait for all async writes to complete
    if jax.process_index() == 0 and writer_thread is not None:
        write_queue.join()  # Wait for all queued writes
        write_queue.put(None)  # Poison pill to stop writer thread
        writer_thread.join()  # Wait for thread to exit

        # Check for errors from writer thread
        if write_error[0] is not None:
            raise RuntimeError(f"Async writer failed: {write_error[0]}")

    # Sync all processes after writes complete
    jax.experimental.multihost_utils.sync_global_devices("zeta_writes_complete")

    # Free cached G-space now that chunk loop is done
    if cached_gspace is not None:
        del cached_gspace
    gc.collect()

    _mem_report("ISDF fitting complete")

    # Print summary
    print()  # Clear the \r line
    print(f"\nWritten to {output_file}")
    print(f"{'='*60}")
    print(f"Zeta fitting complete!")
    print(f"  Shape: ({nqx}, {nqy}, {nqz}, {n_rmu}, {n_rtot})")
    print(f"{'='*60}")
    print(f"\nTiming Summary ({num_chunks} r-chunks):")
    print(f"  {'Phase':<20} {'Total':>10} {'Per-chunk':>12} {'%':>6}")
    print(f"  {'-'*50}")
    print(f"  {'Load chunk':<20} {t_load_total:>10.2f}s {t_load_total/num_chunks*1000:>10.1f}ms {100*t_load_total/t_chunks_total:>6.1f}%")
    print(f"  {'Pair density':<20} {t_pair_total:>10.2f}s {t_pair_total/num_chunks*1000:>10.1f}ms {100*t_pair_total/t_chunks_total:>6.1f}%")
    print(f"  {'ZCT (FFT pipeline)':<20} {t_zct_total:>10.2f}s {t_zct_total/num_chunks*1000:>10.1f}ms {100*t_zct_total/t_chunks_total:>6.1f}%")
    print(f"  {'Solve (L^-1 Z)':<20} {t_solve_total:>10.2f}s {t_solve_total/num_chunks*1000:>10.1f}ms {100*t_solve_total/t_chunks_total:>6.1f}%")
    print(f"  {'H5 write':<20} {t_write_total:>10.2f}s {t_write_total/num_chunks*1000:>10.1f}ms {100*t_write_total/t_chunks_total:>6.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'Chunk loop total':<20} {t_chunks_total:>10.2f}s {t_chunks_total/num_chunks*1000:>10.1f}ms")
    print(f"  {'Per r-point':<20} {'':<10} {t_chunks_total/n_rtot*1e6:>10.1f}us")

    # Return left and right centroid wavefunctions (persist for downstream use)
    # Note: full arrays were already deleted after slicing in STEP 1
    return psi_l_rmu_Y, psi_l_rmuT_X, psi_r_rmu_Y, psi_r_rmuT_X

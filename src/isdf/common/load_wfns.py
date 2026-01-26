import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

from . import Meta
from . import timing
from .cholesky_2d import (
    cholesky_2d_batched,
    dense_to_tiles,
    tiles_to_dense,
)


# ============================================================================
# shard_map based FFT - runs FFT independently on each device's local data
# See: https://docs.jax.dev/en/latest/notebooks/shard_map.html
# ============================================================================

def make_sharded_ifftn_3d(mesh: Mesh, in_spec: P, out_spec: P):
    """
    Create a 3D inverse FFT function that works on sharded arrays.
    
    Uses shard_map to run FFT independently on each device's local data.
    The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
    
    Args:
        mesh: The device mesh
        in_spec: PartitionSpec for input (e.g., P(None, ('x','y'), None, None, None, None))
        out_spec: PartitionSpec for output (same as in_spec for FFT)
    
    Returns:
        A function that performs 3D IFFT on sharded data
    """
    @jax.shard_map(mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)
    def _local_ifftn(x_local):
        # Each device runs FFT on its local shard independently
        return jnp.fft.ifftn(x_local, axes=(-3, -2, -1))
    
    return _local_ifftn


def get_enk_bandrange(wfn, sym, bandrange, sigma_bandrange, nspinor=2):
	"""Return band energies and per-band weights for a given band window.

	Args:
		wfn: WFNReader providing energies and Fermi level
		sym: SymMaps with mappings between irreducible and full k sets
		bandrange: tuple[int,int] inclusive-exclusive (start, end) bands to extract
		sigma_bandrange: tuple[int,int] band window used to compute weighting
		nspinor: Number of spinor components (2 for Pauli, 4 for bispinor)

	Returns:
		enk: jax.Array of shape (nk_full, nb)
		weights: jax.Array of shape (nk_full, nb * nspinor) with simple val/cond weights
	"""
	# Energies are stored on irreducible k; expand to full k using mapping
	nb = int(bandrange[1] - bandrange[0])
	en_irk = jnp.asarray(wfn.energies[0, :, bandrange[0] : bandrange[1]])
	# Arrange as (nk_full, nb) to keep nk as first dim for consistency
	enk = en_irk[sym.irk_to_k_map, :]

	# Build simple least-squares weights following sigma window heuristic
	sigma_start, sigma_end = int(sigma_bandrange[0]), int(sigma_bandrange[1])
	enk_sigma_start = max(sigma_start - int(bandrange[0]), 0)
	enk_sigma_end = min(sigma_end - int(bandrange[0]), nb)
	energies_sym = jnp.asarray(wfn.energies[0, :, :])  # (nk_sym, nband_total)
	energies_full = energies_sym[sym.irk_to_k_map, :]   # (nk_full, nband_total)
	energies_sigma = energies_full[:, sigma_start:sigma_end]
	E_min = jnp.min(energies_sigma)
	E_max = jnp.max(energies_sigma)
	# Determine valence vs conduction relative to Fermi level
	mask_val = enk <= wfn.efermi
	val_weights = 1.0 / jnp.sqrt(jnp.maximum(E_max - enk, 1e-12))
	cond_weights = 1.0 / jnp.sqrt(jnp.maximum(enk - E_min, 1e-12))
	weights_full = jnp.where(mask_val, val_weights, cond_weights)
	# Normalize and set sigma subwindow weights to 1.0
	wmax = jnp.max(weights_full)
	weights_full = jnp.where(wmax > 0, weights_full / wmax, weights_full)
	weights_full = weights_full.at[:, enk_sigma_start:enk_sigma_end].set(1.0)
	# Repeat weights for each spinor component (2 for Pauli, 4 for bispinor)
	return enk, jnp.repeat(weights_full, repeats=nspinor, axis=1)


def get_small_psi_component(gvecs, kvec, bvec, psi_G):
	# get alpha/2 (sigma dot (k+G)) psi_nk(G) for bispinor functionality (single k at a time).
	# Note: Not @jax.jit because ngk varies per k-point → recompilation overhead on GPU
	# possible improvements: do sigma dot v, v = p + [r,V_NL+Sigma], add the DKH4 contribution
	halfalpha = jnp.complex128(0.00364867628215)  # 1/2 * alpha
	sigmadotp = jnp.zeros((2, 2, gvecs.shape[0]), dtype=jnp.complex128)

	gvecsk_cart = jnp.matmul(gvecs + kvec, bvec)

	sigmadotp[0, 0, :] = gvecsk_cart[:, 2]
	sigmadotp[0, 1, :] = gvecsk_cart[:, 0] - 1j * gvecsk_cart[:, 1]
	sigmadotp[1, 0, :] = gvecsk_cart[:, 0] + 1j * gvecsk_cart[:, 1]
	sigmadotp[1, 1, :] = -gvecsk_cart[:, 2]

	return jnp.multiply(
		halfalpha, jnp.einsum("ijG,bjG->biG", sigmadotp, psi_G[:, 0:2, :])
	)


def read_Gvecs_to_devices(
	wfn, sym, bandrange, meta: Meta, bispinor: bool, mesh_xy: Mesh
):
	"""
	Non-jitted: load cnk(G) for all k-points and (padded) band shards into a global
	sharded G-space FFT box over a 2D mesh ['x','y'] along the band axis.
	Returns the global sharded array global_psi_Gtot and nb_actual.
	"""
	nb = bandrange[1] - bandrange[0]

	# 2D device mesh already provided (mesh_xy); derive grid dims
	devices_2d = mesh_xy.devices
	grid_x, grid_y = devices_2d.shape
	total_devices = grid_x * grid_y

	# Bands per shard and total padded bands
	bands_per_shard = (nb + total_devices - 1) // total_devices
	total_bands_padded = bands_per_shard * total_devices

	# Map local devices to their (x, y) coordinates in the mesh
	local_devices = list(jax.local_devices())
	local_coords = [tuple(np.argwhere(np.asarray(devices_2d) == d)[0]) for d in local_devices]
	local_flat_ids = [cx * grid_y + cy for (cx, cy) in local_coords]
	order = np.argsort(local_flat_ids)
	local_coords = [local_coords[i] for i in order]
	n_local_shards = len(local_coords)

	# Local buffer for all k-points and this process's band shards (G-space)
	psi_Gtot_local = np.zeros(
		(meta.nk_tot, n_local_shards * bands_per_shard, meta.nspinor, *meta.fft_grid),
		dtype=np.complex128,
	)

	def place_band_into_local(j: int) -> tuple[int, int] | None:
		global_shard = j // bands_per_shard
		shard_x, shard_y = divmod(global_shard, grid_y)
		try:
			local_slot = local_coords.index((shard_x, shard_y))
		except ValueError:
			return None
		offset = j % bands_per_shard
		return local_slot, offset

	# Pre-compute which bands this process owns (avoids checking every band per k-point)
	with timing.section("load_wfns.precompute_owned"):
		owned_band_indices = []  # global band indices
		local_band_indices = []  # where they go in local buffer
		for j in range(nb):
			placement = place_band_into_local(j)
			if placement is not None:
				local_slot, offset = placement
				owned_band_indices.append(bandrange[0] + j)
				local_band_indices.append(local_slot * bands_per_shard + offset)
		owned_band_indices = np.array(owned_band_indices, dtype=np.int64)
		local_band_indices = np.array(local_band_indices, dtype=np.int64)
		n_owned = len(owned_band_indices)

	# Load G-coefficients for all k-points
	# Phase 1: HDF5 reads (inherently serial) - collect all data
	with timing.section("load_wfns.k_loop"):
		# Pre-compute ngk and max for padding
		ngk_all = np.array([int(wfn.ngk[sym.irk_to_k_map[k]]) for k in range(sym.nk_tot)])
		max_ngk = int(ngk_all.max())
		
		# Pre-allocate padded arrays for batched scatter
		n_local_bands = n_local_shards * bands_per_shard
		psi_Gspace_all = np.zeros((sym.nk_tot, n_local_bands, meta.nspinor, max_ngk), dtype=np.complex128)
		gvecs_all = np.zeros((sym.nk_tot, max_ngk, 3), dtype=np.int32)
		
		for k_idx in range(sym.nk_tot):
			k_red = sym.irk_to_k_map[k_idx]
			gvecs_k_rot = np.asarray(sym.get_gvecs_kfull(wfn, k_idx))
			ngk = ngk_all[k_idx]
			
			# Store G-vectors (padded)
			gvecs_all[k_idx, :ngk, :] = gvecs_k_rot
			
			# Batch read and rotate all owned bands at once
			if n_owned > 0:
				cnk_batch = sym.get_cnk_fullzone_batch(wfn, owned_band_indices, k_idx)
				psi_Gspace_all[k_idx, local_band_indices, 0:meta.nspinor_wfnfile, :ngk] = cnk_batch
			
			# Expand to 4 components if requested
			if bispinor:
				psi_Gspace_local = psi_Gspace_all[k_idx, :, :, :ngk]
				psi_Gspace_all[k_idx, :, 2:4, :ngk] = np.asarray(get_small_psi_component(
					jnp.asarray(gvecs_k_rot),
					jnp.asarray(sym.unfolded_kpts[k_idx], dtype=jnp.float64),
					jnp.asarray(wfn.bvec, dtype=jnp.float64),
					jnp.asarray(psi_Gspace_local),
				))
	
	# Phase 2: Scatter to FFT box (NumPy advanced indexing)
	# Note: JAX scatter is slower due to immutability overhead in loop
	with timing.section("load_wfns.scatter"):
		for k_idx in range(sym.nk_tot):
			ngk = ngk_all[k_idx]
			gvecs_k = gvecs_all[k_idx, :ngk]
			psi_k = psi_Gspace_all[k_idx, :, :, :ngk]
			# Scatter: psi_Gtot_local[k, :, :, gx, gy, gz] = psi_k.T
			psi_Gtot_local[k_idx, :, :, gvecs_k[:, 0], gvecs_k[:, 1], gvecs_k[:, 2]] = np.transpose(psi_k, (2, 0, 1))

	# Promote local buffer to a global sharded JAX array over bands across both [x,y]
	with timing.section("load_wfns.make_global_array"):
		global_shape = (meta.nk_tot, total_bands_padded, meta.nspinor, *meta.fft_grid)
		band_sharding = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None, None, None))
		# Use device_put for faster host-to-device transfer (9x faster than jnp.asarray)
		psi_local_jax = jax.device_put(psi_Gtot_local)
		global_psi_Gtot = jax.make_array_from_process_local_data(
			band_sharding, psi_local_jax, global_shape
		)

	return global_psi_Gtot, nb


# Cache for jitted functions keyed by (mesh_id, fft_grid, nk_tot, nspinor)
_get_sharded_wfns_cache = {}


def get_sharded_wfns(
	global_psi_Gtot: jax.Array,
	sym,
	meta: Meta,
	centroid_indices,
	nb_actual: int,
	is_left: bool,
	mesh_xy: Mesh,
):
	"""
	Jitted: FFT -> apply phase -> normalize/trim -> flatten r -> reshard (Y-only) -> centroid gather ->
	build psi_rmu^T with X-only sharding. Returns (psi_rtot_Y, psi_rmu_Y, psi_rmuT_X).
	
	Uses function caching to avoid JIT recompilation on repeated calls.
	"""
	# Create cache key from hashable values
	cache_key = (
		id(mesh_xy),  # Mesh identity
		meta.fft_grid,  # Tuple of grid dims
		meta.nk_tot,
		meta.nspinor,
		meta.n_rtot,
		len(centroid_indices),  # Number of centroids
	)
	
	if cache_key not in _get_sharded_wfns_cache:
		# Create shardings and jitted function once per unique configuration
		xy2_6 = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None, None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))  # psi_rmuT sharded on mu (axis 1)
		null_4 = NamedSharding(mesh_xy, P(None, None, None, None))
		
		# Create sharded FFT using shard_map
		fft_spec = P(None, ('x', 'y'), None, None, None, None)
		sharded_ifftn = make_sharded_ifftn_3d(mesh_xy, fft_spec, fft_spec)
		
		# Pre-compute static values
		fft_grid = meta.fft_grid
		nk_tot = meta.nk_tot
		nspinor = meta.nspinor
		n_rtot = meta.n_rtot
		sqrt_n_rtot = jnp.sqrt(n_rtot)
		
		# Pre-compute phase grid (static)
		fx = jnp.arange(fft_grid[0], dtype=jnp.float64)[None, :, None, None] / fft_grid[0]
		fy = jnp.arange(fft_grid[1], dtype=jnp.float64)[None, None, :, None] / fft_grid[1]
		fz = jnp.arange(fft_grid[2], dtype=jnp.float64)[None, None, None, :] / fft_grid[2]
		
		# Pre-compute centroid linear indices
		centroids = jnp.asarray(centroid_indices, dtype=jnp.int32)
		ny, nz = fft_grid[1], fft_grid[2]
		centroid_lin = (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(jnp.int32)
		n_rmu = len(centroid_indices)

		@partial(jax.jit,
				 static_argnames=("nb_actual", "is_left"), 
				 in_shardings=(xy2_6, None), 
				 out_shardings=(y3_4, y3_4, x1_4))
		def _finalize(global_psi_Gtot: jax.Array, kpts: jax.Array, nb_actual: int, is_left: bool):
			# FFT to real space - shard_map runs FFT independently on each device
			psi_r = sharded_ifftn(global_psi_Gtot)

			# Apply Bloch phase exp(ik·r) using pre-computed grids
			phase_spatial = jnp.exp(
				2j * jnp.pi *
				(
					kpts[:, 0:1, None, None] * fx
					+ kpts[:, 1:2, None, None] * fy
					+ kpts[:, 2:3, None, None] * fz
				)
			)
			psi_r = psi_r * phase_spatial[:, None, None, :, :, :]

			# Conjugate (if left) and normalization
			psi_r = jnp.where(jnp.asarray(is_left), jnp.conj(psi_r), psi_r)
			psi_r = psi_r * sqrt_n_rtot

			# Trim bands to actual request
			psi_r = psi_r[:, :nb_actual]

			# Flatten spatial dims to rtot
			psi_rtot = psi_r.reshape(nk_tot, nb_actual, nspinor, -1)

			# Centroid gather using pre-computed linear indices
			# NOTE: No replication needed - JAX handles gather efficiently with
			# bands still sharded, avoiding a 250x memory spike.
			psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)

			# Conjugate-transpose to (nk, n_rmu, nb, nspinor)
			psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2).reshape(nk_tot, n_rmu, -1, nspinor))
			
			# Re-shard results: rtot and rmu to Y, rmuT to X
			psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, y3_4)
			psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, y3_4)
			psi_rmuT = jax.lax.with_sharding_constraint(psi_rmuT, x1_4)

			return psi_rtot, psi_rmu, psi_rmuT
		
		_get_sharded_wfns_cache[cache_key] = _finalize
	
	# Get cached function and call it
	_finalize = _get_sharded_wfns_cache[cache_key]
	kpts = jnp.asarray(sym.unfolded_kpts[:meta.nk_tot], dtype=jnp.float64)
	return _finalize(global_psi_Gtot, kpts, nb_actual, bool(is_left))


def load_psi_rtot_for_bandrange(
	wfn,
	sym,
	band_start: int,
	band_end: int,
	meta: Meta,
	bispinor: bool,
	mesh_xy: Mesh,
	chunk_size: int = 64,
):
	"""Load psi_nk(r) on the full FFT grid for a specific band range.
	
	Designed for future ZCT chunking: call repeatedly with different band ranges,
	accumulate contributions, then discard intermediate results to save memory.
	
	This function does NOT return centroid-sampled psi_rmu - use get_sharded_wfns
	for that (centroids should be loaded once for all bands and kept in memory).
	
	Args:
		wfn: WFNReader instance
		sym: SymMaps instance
		band_start: First band index (inclusive)
		band_end: Last band index (exclusive)
		meta: Meta instance with grid/dimension info
		bispinor: If True, compute 4-component spinors
		mesh_xy: JAX device mesh for sharding
		chunk_size: Bands per chunk (default 64, for memory tuning)
		
	Returns:
		psi_rtot: jax.Array of shape (nk, nb, nspinor, n_rtot)
	"""
	bandrange = (band_start, band_end)
	nb = band_end - band_start
	
	# Load G-space coefficients using the optimized batch path
	with timing.section("load_psi_rtot.read_Gvecs"):
		global_psi_Gtot, _ = read_Gvecs_to_devices(wfn, sym, bandrange, meta, bispinor, mesh_xy)
	
	# FFT to real space and apply phase
	xy2_6 = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None, None, None))
	y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
	
	# Create sharded FFT using shard_map
	fft_spec = P(None, ('x', 'y'), None, None, None, None)
	sharded_ifftn = make_sharded_ifftn_3d(mesh_xy, fft_spec, fft_spec)
	
	@partial(jax.jit, static_argnames=("nb",), in_shardings=(xy2_6,), out_shardings=y3_4)
	def _to_rtot(psi_Gtot: jax.Array, nb: int) -> jax.Array:
		# FFT to real space - shard_map runs FFT independently on each device
		psi_r = sharded_ifftn(psi_Gtot)
		# Apply Bloch phase exp(ik·r)
		fx = jnp.arange(meta.fft_grid[0], dtype=jnp.float64)[None, :, None, None] / meta.fft_grid[0]
		fy = jnp.arange(meta.fft_grid[1], dtype=jnp.float64)[None, None, :, None] / meta.fft_grid[1]
		fz = jnp.arange(meta.fft_grid[2], dtype=jnp.float64)[None, None, None, :] / meta.fft_grid[2]
		kpts = jnp.asarray(sym.unfolded_kpts, dtype=jnp.float64)[:psi_r.shape[0]]
		phase = jnp.exp(2j * jnp.pi * (
			kpts[:, 0:1, None, None] * fx +
			kpts[:, 1:2, None, None] * fy +
			kpts[:, 2:3, None, None] * fz
		))
		psi_r = psi_r * phase[:, None, None, :, :, :]
		psi_r = psi_r * jnp.sqrt(meta.n_rtot)
		psi_r = psi_r[:, :nb]
		# Flatten spatial dims: (nk, nb, nspinor, nx, ny, nz) -> (nk, nb, nspinor, n_rtot)
		return psi_r.reshape(meta.nk_tot, nb, meta.nspinor, -1)
	
	with timing.section("load_psi_rtot.fft_phase"):
		psi_rtot = _to_rtot(global_psi_Gtot, nb)
	
	return psi_rtot


# ============================================================================
# Pair density computation: P_k,ab(r_mu, r_nu) = sum_n psi*_nk,a(r_mu) * psi_nk,b(r_nu)
# ============================================================================

# Cache for pair density jitted functions
_compute_pair_density_cache = {}


def compute_pair_density_k(
	psi_rmuT_X: jax.Array,
	psi_rmu_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute pair density P_k,ab(r_mu, r_nu) = sum_n psi*_nk,a(r_mu) * psi_nk,b(r_nu).
	
	This function computes the pair density matrix for all k-points at once,
	with spin indices a,b explicitly tracked (unlike cohsex_jax which traces).
	
	The result is sharded with r_mu on X and r_nu on Y, enabling zero-communication
	contraction from the input wavefunctions.
	
	Input shapes and shardings:
		psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
			- This is conj(psi_nk,s(r_mu)) with mu sharded on X
		psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
			- This is psi_nk,s(r_nu) with nu sharded on Y
	
	Output:
		P_k_ab: (nk, ns, ns, n_rmu, n_rmu) with P(None, None, None, 'x', 'y')
			- P[k, a, b, mu, nu] = sum_n psi*_nk,a(r_mu) * psi_nk,b(r_nu)
			- mu sharded on X, nu sharded on Y (zero-comm from inputs)
	
	Note: This differs from cohsex_jax which computes P_mu,nu = sum_nks for the
	spin-traced case. Here we keep spin indices explicit for different treatment.
	"""
	# Cache key based on shapes and mesh
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	cache_key = (id(mesh_xy), nk, n_rmu, nb, ns)
	
	if cache_key not in _compute_pair_density_cache:
		# Define shardings
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		xy_out = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		
		@partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
		def _compute_P(psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			"""
			psi_L: (nk, n_rmu, nb, ns) - conjugated, mu on X
			psi_R: (nk, nb, ns, n_rmu) - nu on Y
			
			Einsum: P[k,a,b,mu,nu] = sum_n psi_L[k,mu,n,a] * psi_R[k,n,b,nu]
			       'kmna,knbv->kabmv'
			"""
			return jnp.einsum('kmna,knbv->kabmv', psi_L, psi_R, optimize=True)
		
		_compute_pair_density_cache[cache_key] = _compute_P
	
	_compute_P = _compute_pair_density_cache[cache_key]
	return _compute_P(psi_rmuT_X, psi_rmu_Y)


def compute_pair_density_k_zchunk(
	psi_rmuT_X: jax.Array,
	psi_zchunk_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute pair density P_k,ab(r_mu, r_zchunk) for ZCT accumulation.
	
	Same as compute_pair_density_k but with r_zchunk (a z-slice of r_tot) instead
	of r_nu (centroids). Used for iterating over z-chunks to build ZCT without
	storing the full psi_nk(r_tot).
	
	Input shapes and shardings:
		psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
			- conj(psi_nk,s(r_mu)) with mu sharded on X
		psi_zchunk_Y: (nk, nb, ns, n_zchunk) with P(None, None, None, 'y')
			- psi_nk,s(r_zchunk) with zchunk sharded on Y
	
	Output:
		P_k_ab_zchunk: (nk, ns, ns, n_rmu, n_zchunk) with P(None, None, None, 'x', 'y')
			- P[k, a, b, mu, r] = sum_n psi*_nk,a(r_mu) * psi_nk,b(r_zchunk)
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_zchunk = psi_zchunk_Y.shape
	
	# Different cache key due to different n_zchunk
	cache_key = ('zchunk', id(mesh_xy), nk, n_rmu, nb, ns, n_zchunk)
	
	if cache_key not in _compute_pair_density_cache:
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		xy_out = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		
		@partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
		def _compute_P_zchunk(psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			"""
			psi_L: (nk, n_rmu, nb, ns) - conjugated, mu on X
			psi_R: (nk, nb, ns, n_zchunk) - zchunk on Y
			
			Einsum: P[k,a,b,mu,r] = sum_n psi_L[k,mu,n,a] * psi_R[k,n,b,r]
			       'kmna,knbr->kabmr'
			"""
			return jnp.einsum('kmna,knbr->kabmr', psi_L, psi_R, optimize=True)
		
		_compute_pair_density_cache[cache_key] = _compute_P_zchunk
	
	_compute_P_zchunk = _compute_pair_density_cache[cache_key]
	return _compute_P_zchunk(psi_rmuT_X, psi_zchunk_Y)


# ============================================================================
# Pair density k-space <-> R-space transforms and CCT/ZCT accumulation
# ============================================================================

# Cache for ISDF pipeline jitted functions
_isdf_pipeline_cache = {}


def pair_density_k_to_R(
	P_k_ab: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Transform pair density from k-space to R-space via ortho-IFFT.
	
	P_k,ab(μ,ν) -> P_R,ab(μ,ν)
	
	This is analogous to the G_k -> G_R transform in cohsex_jax/w_isdf.
	The k-grid is reshaped to 3D and ortho-IFFT is applied along kx,ky,kz.
	
	Args:
		P_k_ab: (nk, ns, ns, n_rmu, n_col) with P(None, None, None, 'x', 'y')
			where n_col is either n_rmu (for CCT) or n_zchunk (for ZCT)
		kgrid: (nkx, nky, nkz) k-grid dimensions
		mesh_xy: Device mesh
	
	Returns:
		P_R_ab: (ns, ns, n_rmu, n_col, nkx, nky, nkz) with P(None, None, 'x', 'y', None, None, None)
			R-space pair density
	"""
	nkx, nky, nkz = kgrid
	nk, ns1, ns2, n_rmu, n_col = P_k_ab.shape
	
	cache_key = ('k_to_R', id(mesh_xy), nk, ns1, n_rmu, n_col, nkx)
	
	if cache_key not in _isdf_pipeline_cache:
		in_shard = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		out_shard = NamedSharding(mesh_xy, P(None, None, 'x', 'y', None, None, None))
		
		@partial(jax.jit, in_shardings=(in_shard,), out_shardings=out_shard,
		         static_argnames=('nkx', 'nky', 'nkz'))
		def _k_to_R(P_k: jax.Array, nkx: int, nky: int, nkz: int) -> jax.Array:
			# Reshape k to 3D grid: (nk, s1, s2, mu, col) -> (nkx, nky, nkz, s1, s2, mu, col)
			P_k = P_k.reshape(nkx, nky, nkz, ns1, ns2, n_rmu, n_col)
			# Move spatial indices last for FFT: (s1, s2, mu, col, nkx, nky, nkz)
			P_k = P_k.transpose(3, 4, 5, 6, 0, 1, 2)
			# Ortho IFFT along k-grid axes
			return jnp.fft.ifftn(P_k, axes=(-3, -2, -1), norm='ortho')
		
		_isdf_pipeline_cache[cache_key] = _k_to_R
	
	return _isdf_pipeline_cache[cache_key](P_k_ab, nkx, nky, nkz)


def pair_density_R_to_q(
	P_R_ab: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Transform pair density from R-space to q-space via ortho-FFT.
	
	P_R,ab(μ,col) -> P_q,ab(μ,col)
	
	Args:
		P_R_ab: (ns, ns, n_rmu, n_col, nkx, nky, nkz) with P(None, None, 'x', 'y', None, None, None)
		mesh_xy: Device mesh
	
	Returns:
		P_q_ab: (nqx, nqy, nqz, ns, ns, n_rmu, n_col) with P(None, None, None, None, None, 'x', 'y')
	"""
	ns1, ns2, n_rmu, n_col, nkx, nky, nkz = P_R_ab.shape
	
	cache_key = ('R_to_q', id(mesh_xy), ns1, n_rmu, n_col, nkx)
	
	if cache_key not in _isdf_pipeline_cache:
		in_shard = NamedSharding(mesh_xy, P(None, None, 'x', 'y', None, None, None))
		out_shard = NamedSharding(mesh_xy, P(None, None, None, None, None, 'x', 'y'))
		
		@partial(jax.jit, in_shardings=(in_shard,), out_shardings=out_shard)
		def _R_to_q(P_R: jax.Array) -> jax.Array:
			# Ortho FFT along R-grid axes (last 3)
			P_q = jnp.fft.fftn(P_R, axes=(-3, -2, -1), norm='ortho')
			# Reorder: (s1, s2, mu, col, qx, qy, qz) -> (qx, qy, qz, s1, s2, mu, col)
			return P_q.transpose(4, 5, 6, 0, 1, 2, 3)
		
		_isdf_pipeline_cache[cache_key] = _R_to_q
	
	return _isdf_pipeline_cache[cache_key](P_R_ab)


def compute_CCT_ZCT_from_pair_density(
	P_k_mumu: jax.Array,
	P_k_mu_zchunk: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array]:
	"""
	Compute CCT and ZCT matrices from pair densities using ortho FFT pipeline.
	
	For ISDF fitting of spin-traced pair products:
		target(r) = ψ*_{m,k-q,↑}(r)ψ_{n,k,↑}(r) + ψ*_{m,k-q,↓}(r)ψ_{n,k,↓}(r)
	
	The Galerkin condition (see docs/isdf_spin_galerkin_derivation.md) gives:
		C(μ,ν) = Σ_{k,s,s'} P*_{k-q,s's}(ν,μ) P_{k,ss'}(ν,μ)
	
	For q=0 (k-q = k), this simplifies to the Frobenius norm:
		C(μ,ν) = Σ_k ||P_k(μ,ν)||²_F = Σ_{k,a,b} |P_{k,ab}(μ,ν)|²
	
	PHYSICS NOTE:
	Even though we fit only the spin-diagonal sum (↑↑ + ↓↓), the Galerkin
	condition involves ALL FOUR spin combinations |P_ab|² because the band
	summation in the error metric couples spin channels via Σ_m ψ*_{m,s} ψ_{m,s'}.
	
	This is NOT equivalent to |P_↑↑ + P_↓↓|², which would miss the off-diagonal
	spin contributions |P_↑↓|² + |P_↓↑|² and incorrectly include cross-terms.
	
	Pipeline:
		1. P_k,ab(μ,ν) -> P_R,ab(μ,ν)  via ortho-IFFT
		2. C_R(μ,ν) = Σ_ab |P_R,ab(μ,ν)|²  (Frobenius norm squared)
		3. C_R -> C_q  via ortho-FFT
	
	Args:
		P_k_mumu: (nk, ns, ns, n_rmu, n_rmu) pair density at centroids
		P_k_mu_zchunk: (nk, ns, ns, n_rmu, n_zchunk) pair density for z-chunk
		kgrid: (nkx, nky, nkz)
		mesh_xy: Device mesh
	
	Returns:
		C_q: (nqx, nqy, nqz, n_rmu, n_rmu) CCT matrix for all q
		Z_q: (nqx, nqy, nqz, n_rmu, n_zchunk) ZCT matrix for all q
	"""
	nkx, nky, nkz = kgrid
	nk, ns1, ns2, n_rmu, _ = P_k_mumu.shape
	_, _, _, _, n_zchunk = P_k_mu_zchunk.shape
	
	cache_key = ('CCT_ZCT', id(mesh_xy), nk, ns1, n_rmu, n_zchunk, nkx)
	
	if cache_key not in _isdf_pipeline_cache:
		in_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		
		@partial(jax.jit, in_shardings=(in_xy, in_xy), out_shardings=(out_xy, out_xy),
		         static_argnames=('nkx', 'nky', 'nkz'))
		def _compute_CCT_ZCT(P_mumu: jax.Array, P_mu_z: jax.Array,
		                     nkx: int, nky: int, nkz: int) -> tuple[jax.Array, jax.Array]:
			# ---- CCT pathway ----
			# Input: (nk, ns, ns, n_rmu, n_rmu)
			# Reshape to expose k-dimensions: (nkx, nky, nkz, ns, ns, n_rmu, n_rmu)
			P_k_mumu = P_mumu.reshape(nkx, nky, nkz, ns1, ns2, n_rmu, n_rmu)
			
			# IFFT over k-dimensions directly - NO TRANSPOSE NEEDED
			P_R_mumu = jnp.fft.ifftn(P_k_mumu, axes=(0, 1, 2), norm='ortho')
			# P_R_mumu: (Rx, Ry, Rz, ns, ns, n_rmu, n_rmu)
			
			# Frobenius norm squared over spin (axes 3, 4)
			C_R = jnp.sum(jnp.abs(P_R_mumu) ** 2, axis=(3, 4))
			# C_R: (Rx, Ry, Rz, n_rmu, n_rmu)
			
			# FFT over R-dimensions directly - NO TRANSPOSE NEEDED
			C_q = jnp.fft.fftn(C_R, axes=(0, 1, 2), norm='ortho')
			# C_q: (qx, qy, qz, n_rmu, n_rmu) - already correct!
			
			# ---- ZCT pathway ----
			# Input: (nk, ns, ns, n_rmu, n_zchunk)
			P_k_muz = P_mu_z.reshape(nkx, nky, nkz, ns1, ns2, n_rmu, n_zchunk)
			
			# IFFT over k-dimensions directly - NO TRANSPOSE NEEDED
			P_R_muz = jnp.fft.ifftn(P_k_muz, axes=(0, 1, 2), norm='ortho')
			# P_R_muz: (Rx, Ry, Rz, ns, ns, n_rmu, n_zchunk)
			
			# Frobenius norm squared over spin (axes 3, 4)
			Z_R = jnp.sum(jnp.abs(P_R_muz) ** 2, axis=(3, 4))
			# Z_R: (Rx, Ry, Rz, n_rmu, n_zchunk)
			
			# FFT over R-dimensions directly - NO TRANSPOSE NEEDED
			Z_q = jnp.fft.fftn(Z_R, axes=(0, 1, 2), norm='ortho')
			# Z_q: (qx, qy, qz, n_rmu, n_zchunk) - already correct!
			
			return C_q, Z_q
		
		_isdf_pipeline_cache[cache_key] = _compute_CCT_ZCT
	
	return _isdf_pipeline_cache[cache_key](P_k_mumu, P_k_mu_zchunk, nkx, nky, nkz)


def compute_CCT_only(
	P_k_mumu: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute CCT matrix only (no ZCT) from pair density P_k(μ,μ).
	
	This is more memory-efficient than compute_CCT_ZCT_from_pair_density
	when you only need C_q for Cholesky factorization.
	
	Pipeline:
		1. P_k,ab(μ,ν) -> P_R,ab(μ,ν) via ortho-IFFT
		2. C_R(μ,ν) = Σ_ab |P_R,ab(μ,ν)|² (Frobenius norm squared)
		3. C_R -> C_q via ortho-FFT
	
	Args:
		P_k_mumu: (nk, ns, ns, n_rmu, n_rmu) pair density at centroids
		kgrid: (nkx, nky, nkz)
		mesh_xy: Device mesh
	
	Returns:
		C_q: (nqx, nqy, nqz, n_rmu, n_rmu) CCT matrix
	"""
	nkx, nky, nkz = kgrid
	nk, ns1, ns2, n_rmu, _ = P_k_mumu.shape
	
	cache_key = ('CCT_only', id(mesh_xy), nk, ns1, n_rmu, nkx)
	
	if cache_key not in _isdf_pipeline_cache:
		in_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		
		@partial(jax.jit, in_shardings=in_xy, out_shardings=out_xy,
		         static_argnames=('nkx', 'nky', 'nkz'))
		def _compute_CCT(P_mumu: jax.Array, nkx: int, nky: int, nkz: int) -> jax.Array:
			# Input: (nk, ns1, ns2, n_rmu, n_rmu)
			# Reshape to expose k-dimensions: (nkx, nky, nkz, ns1, ns2, n_rmu, n_rmu)
			P_k = P_mumu.reshape(nkx, nky, nkz, ns1, ns2, n_rmu, n_rmu)
			
			# IFFT over k-dimensions directly - NO TRANSPOSE NEEDED
			P_R = jnp.fft.ifftn(P_k, axes=(0, 1, 2), norm='ortho')
			# P_R: (Rx, Ry, Rz, ns1, ns2, n_rmu, n_rmu)
			
			# Frobenius norm squared over spin (axes 3, 4 are spin dimensions)
			C_R = jnp.sum(jnp.abs(P_R) ** 2, axis=(3, 4))
			# C_R: (Rx, Ry, Rz, n_rmu, n_rmu)
			
			# FFT over R-dimensions directly - NO TRANSPOSE NEEDED
			C_q = jnp.fft.fftn(C_R, axes=(0, 1, 2), norm='ortho')
			# C_q: (qx, qy, qz, n_rmu, n_rmu) - already in correct order!
			
			return C_q
		
		_isdf_pipeline_cache[cache_key] = _compute_CCT
	
	return _isdf_pipeline_cache[cache_key](P_k_mumu, nkx, nky, nkz)


# ============================================================================
# ISDF Zeta Fitting: Cholesky Solve with Optimized Sharding
# ============================================================================
#
# Strategy: "fori_loop + shard_map" (benchmarked as fastest for large n_rmu)
#
# Input shardings (from CCT/ZCT pipeline):
#   C_q: (nq, n_rmu, n_rmu) with P(None, 'x', 'y')  -- flattened q-grid
#   Z_q: (nq, n_rmu, n_zchunk) with P(None, 'x', 'y')
#
# Strategy:
#   1. Reshard Z_q from P(None, 'x', 'y') to P(None, None, ('x','y'))
#      -> Z_q(q, μ, rchunk_XY) for column-parallel solve
#      (Verified: this resharding does NOT trigger XLA rematerialization)
#
#   2. For each q in fori_loop:
#      a. Extract C_q[q] (μ_X, ν_Y) and all-gather to replicated (μ, ν)
#      b. Cholesky: L = chol(C_q[q]) -- redundant on each device but fast
#      c. shard_map triangular solve: zeta_q[q] = L^{-H}(L^{-1} Z_q[q])
#         - L is replicated, Z_q[q] has P(None, ('x','y')) on columns
#         - Solve is embarrassingly parallel over rchunk columns
#
# Output:
#   zeta_q: (nq, n_rmu, n_zchunk) with P(None, None, ('x','y'))
#
# Communication per q:
#   - C_q all-gather: n_rmu² × 16 bytes
#   - For n_rmu=10k: 1.6 GB per q (high bandwidth, one collective)
#
# Parallelism:
#   - Solve: each device handles n_zchunk/P columns independently
#   - FLOPs: n_rmu² × (n_zchunk/P) per device per q
#
# For very large n_rmu where replication is infeasible, consider
# custom blocked Cholesky (test_blocked_cholesky.py), but benchmarks
# show that Strategy B is 2x faster than q-resharding for n_rmu ≥ 128.
# ============================================================================

# Cache for solve functions
_zeta_solve_cache: dict = {}


def solve_zeta_q_from_CCT_ZCT(
    C_q: jax.Array,
    Z_q: jax.Array,
    mesh_xy: Mesh,
) -> jax.Array:
    """
    Solve for zeta_q,μ(r) given CCT and ZCT matrices.
    
    Implements: C_q · zeta_q = Z_q  =>  zeta_q = (L L^H)^{-1} Z_q
    where L = cholesky(C_q).
    
    Uses optimized "fori_loop + shard_map" strategy:
    - Input C_q, Z_q: sharded P(None, 'x', 'y') from CCT/ZCT pipeline
    - Z_q resharded to P(None, None, ('x','y')) for column-parallel solve
    - Each q: replicate C_q[q], do local Cholesky, shard_map triangular solve
    - Output zeta: P(None, None, ('x','y')) preserving column sharding
    
    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix with P(None, 'x', 'y')
             (flattened q-grid: nq = nqx * nqy * nqz)
        Z_q: (nq, n_rmu, n_zchunk) ZCT matrix with P(None, 'x', 'y')
        mesh_xy: Device mesh
    
    Returns:
        zeta_q: (nq, n_rmu, n_zchunk) interpolation vectors with P(None, None, ('x','y'))
    
    Notes:
        - For 3D q-grid, reshape output as (nqx, nqy, nqz, n_rmu, n_zchunk)
        - Communication: one all-gather of n_rmu² per q
        - Solve is embarrassingly parallel over rchunk columns
    """
    nq, n_rmu, n_rmu2 = C_q.shape
    _, _, n_zchunk = Z_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"
    
    cache_key = ('zeta_solve', id(mesh_xy), nq, n_rmu, n_zchunk)
    
    if cache_key not in _zeta_solve_cache:
        # Define shardings
        cct_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
        c_rep_shard = NamedSharding(mesh_xy, P(None, None))  # Replicated
        
        # shard_map for triangular solve with column-sharded Z
        # L is replicated (full matrix on each device)
        # Z_cols is local columns (n_rmu, n_zchunk/P)
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            """Column-parallel Cholesky solve: L L^H zeta = Z => zeta = L^{-H}(L^{-1} Z)"""
            # Forward: L y = Z
            y = jax.scipy.linalg.solve_triangular(L, Z_cols, lower=True)
            # Backward: L^H zeta = y
            zeta = jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
            return zeta
        
        @jax.jit
        def _solve_all_q(C_cct: jax.Array, Z_cct: jax.Array) -> jax.Array:
            """Solve zeta for all q using fori_loop over q-points."""
            # Reshard Z for column-parallel solve (no rematerialization)
            Z_col = jax.lax.with_sharding_constraint(Z_cct, z_col_shard)
            
            def body(q, zeta_all):
                # Extract and replicate C_q[q]
                C_single = C_cct[q]  # (μ_X, ν_Y)
                C_rep = jax.lax.with_sharding_constraint(C_single, c_rep_shard)
                
                # Cholesky factorization (redundant on each device but fast)
                L = jnp.linalg.cholesky(C_rep)
                
                # Extract Z_q[q] (already column-sharded)
                Z_single = Z_col[q]  # (μ, rchunk_XY)
                
                # Sharded triangular solve
                zeta_q = _sharded_cho_solve(L, Z_single)
                
                # Store result
                return zeta_all.at[q].set(zeta_q)
            
            # Initialize output with same sharding as Z_col
            zeta_init = jnp.zeros_like(Z_col)
            return jax.lax.fori_loop(0, nq, body, zeta_init)
        
        _zeta_solve_cache[cache_key] = _solve_all_q
    
    return _zeta_solve_cache[cache_key](C_q, Z_q)


def solve_zeta_q_3d(
    C_q_3d: jax.Array,
    Z_q_3d: jax.Array,
    mesh_xy: Mesh,
) -> jax.Array:
    """
    Solve for zeta with 3D q-grid indexing.
    
    Convenience wrapper that handles the q-grid reshaping.
    
    Args:
        C_q_3d: (nqx, nqy, nqz, n_rmu, n_rmu) with P(None, None, None, 'x', 'y')
        Z_q_3d: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, 'x', 'y')
        mesh_xy: Device mesh
    
    Returns:
        zeta_3d: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, None, ('x','y'))
    """
    nqx, nqy, nqz, n_rmu, n_rmu2 = C_q_3d.shape
    _, _, _, _, n_zchunk = Z_q_3d.shape
    nq = nqx * nqy * nqz
    
    # Flatten q-grid
    C_q_flat = C_q_3d.reshape(nq, n_rmu, n_rmu2)
    Z_q_flat = Z_q_3d.reshape(nq, n_rmu, n_zchunk)
    
    # Ensure correct input sharding
    flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)
    Z_q_flat = jax.lax.with_sharding_constraint(Z_q_flat, flat_shard)
    
    # Solve
    zeta_flat = solve_zeta_q_from_CCT_ZCT(C_q_flat, Z_q_flat, mesh_xy)
    
    # Reshape back to 3D q-grid
    return zeta_flat.reshape(nqx, nqy, nqz, n_rmu, n_zchunk)


# ============================================================================
# 2D Blocked Cholesky Solver - memory efficient for large n_rmu
# ============================================================================

# Cache for 2D cholesky function
_chol_2d_cache = {}


def solve_zeta_q_blocked_2d(
    C_q: jax.Array,
    Z_q: jax.Array,
    mesh_xy: Mesh,
    block_size: int = None,
) -> jax.Array:
    """
    Solve C_q · zeta_q = Z_q using 2D blocked Cholesky + q-by-q solve.
    
    Memory-efficient hybrid approach:
    1. 2D blocked Cholesky: C_q(μ_X, ν_Y) → L_q(μ_X, ν_Y) distributed
    2. Loop over q: all-gather L_q[q] to every device (one at a time)
    3. Triangular solve: with Z_q(μ, rchunk_XY) column-sharded (zero comm)
    
    Memory comparison for n_rmu=10k, nq=100, P=128:
        - Full replicate: nq × n_rmu² × 16B = 160 GB total, 1.25 GB/device
        - This approach:   n_rmu² × 16B = 1.6 GB (peak, during solve loop)
        - During Cholesky: n_rmu²/P × 16B ≈ 12 MB/device
    
    Performance:
        - Cholesky: O(J × log P) communication rounds, O(n³/P) compute
        - Solve: O(nq) all-gathers of n_rmu² each, O(nq × n_rmu² × n_zchunk/P) compute
        - Solve is zero-comm after all-gather (embarrassingly parallel over columns)
    
    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix, sharded P(None, 'x', 'y')
        Z_q: (nq, n_rmu, n_zchunk) ZCT matrix, sharded P(None, 'x', 'y')  
        mesh_xy: 2D device mesh with axes ('x', 'y')
        block_size: Tile block size for Cholesky. If None, auto-selects
                    based on n_rmu (default: n_rmu / lcm(Pr, Pc))
    
    Returns:
        zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, None, ('x','y'))
    
    Example:
        zeta = solve_zeta_q_blocked_2d(C_q, Z_q, mesh_xy)
    """
    nq, n_rmu, n_rmu2 = C_q.shape
    _, _, n_zchunk = Z_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"
    
    # Auto-select block size for 2D distribution
    Pr = mesh_xy.shape['x']
    Pc = mesh_xy.shape['y']
    
    if block_size is None:
        # Choose block size so J = n_rmu/b is divisible by both Pr and Pc
        # Target: J = lcm(Pr, Pc) or a small multiple
        import math
        target_J = math.lcm(Pr, Pc)
        if n_rmu % target_J == 0:
            block_size = n_rmu // target_J
        else:
            # Fall back: make b divide n_rmu and J divisible by Pr*Pc
            for j_mult in range(1, 10):
                J = target_J * j_mult
                if n_rmu % J == 0:
                    block_size = n_rmu // J
                    break
            else:
                # Last resort: simple block size
                block_size = max(1, n_rmu // (Pr * Pc))
    
    J = n_rmu // block_size
    
    # Validate
    assert n_rmu % block_size == 0, f"n_rmu={n_rmu} must divide block_size={block_size}"
    assert J % Pr == 0, f"J={J} must divide Pr={Pr}"
    assert J % Pc == 0, f"J={J} must divide Pc={Pc}"
    
    # Get or build cached Cholesky function
    cache_key = ('chol_2d', id(mesh_xy), J, block_size)
    if cache_key not in _chol_2d_cache:
        _chol_2d_cache[cache_key] = cholesky_2d_batched(mesh_xy, J, block_size)
    
    chol_fn = _chol_2d_cache[cache_key]
    
    # Convert C_q to tiles
    C_q_tiles = dense_to_tiles(C_q, block_size)  # (nq, J, J, b, b)
    
    # Apply sharding for 2D distribution
    tiles_shard = NamedSharding(mesh_xy, P(None, 'x', 'y', None, None))
    C_q_tiles = jax.lax.with_sharding_constraint(C_q_tiles, tiles_shard)
    
    # ========== STEP 1: 2D Blocked Cholesky ==========
    # L_q remains distributed as (nq, J_x, J_y, b, b) during factorization
    L_q_tiles = chol_fn(C_q_tiles)  # (nq, J, J, b, b), sharded P(None, 'x', 'y', None, None)
    
    # Convert to dense but keep sharded P(None, 'x', 'y')
    L_q_dense = tiles_to_dense(L_q_tiles, block_size)  # (nq, n_rmu, n_rmu)
    L_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    L_q_dense = jax.lax.with_sharding_constraint(L_q_dense, L_shard)
    
    # ========== STEP 2: Reshard Z for column-parallel solve ==========
    # Z_q: P(None, 'x', 'y') → P(None, None, ('x','y'))
    z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    Z_col = jax.lax.with_sharding_constraint(Z_q, z_col_shard)
    
    # ========== STEP 3: Loop over q, replicate L[q], solve ==========
    # Define shardings for the solve
    L_rep_shard = NamedSharding(mesh_xy, P(None, None))  # Replicated
    
    # shard_map for column-parallel triangular solve
    # L is replicated (full n_rmu × n_rmu on each device)
    # Z_cols is local columns (n_rmu, n_zchunk/P)
    @partial(shard_map, mesh=mesh_xy,
             in_specs=(P(None, None), P(None, ('x', 'y'))),
             out_specs=P(None, ('x', 'y')))
    def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
        """Column-parallel solve: L L^H zeta = Z => zeta = L^{-H}(L^{-1} Z)"""
        # Forward: L y = Z
        y = jax.scipy.linalg.solve_triangular(L, Z_cols, lower=True)
        # Backward: L^H zeta = y
        zeta = jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
        return zeta
    
    # Build the solve function with fori_loop over q
    @jax.jit
    def _solve_all_q(L_q_sharded: jax.Array, Z_col_sharded: jax.Array) -> jax.Array:
        """Solve zeta for all q using fori_loop, all-gathering L[q] one at a time."""
        
        def body(q, zeta_all):
            # Extract L[q] (still sharded as P('x', 'y'))
            L_single = L_q_sharded[q]  # (n_rmu_X, n_rmu_Y)
            
            # All-gather to replicate L[q] on every device
            L_rep = jax.lax.with_sharding_constraint(L_single, L_rep_shard)
            
            # Extract Z[q] (already column-sharded as P(None, ('x','y')))
            Z_single = Z_col_sharded[q]  # (n_rmu, n_zchunk_XY)
            
            # Column-parallel triangular solve (zero communication)
            zeta_q = _sharded_cho_solve(L_rep, Z_single)
            
            # Store result
            return zeta_all.at[q].set(zeta_q)
        
        # Initialize output with column sharding
        zeta_init = jnp.zeros_like(Z_col_sharded)
        return jax.lax.fori_loop(0, nq, body, zeta_init)
    
    zeta_q = _solve_all_q(L_q_dense, Z_col)
    
    # Output already has correct sharding P(None, None, ('x','y'))
    return zeta_q


def solve_zeta_q_blocked_2d_3d(
    C_q_3d: jax.Array,
    Z_q_3d: jax.Array,
    mesh_xy: Mesh,
    block_size: int = None,
) -> jax.Array:
    """
    Solve zeta with 3D q-grid using 2D blocked Cholesky.
    
    Convenience wrapper that handles q-grid reshaping.
    
    Args:
        C_q_3d: (nqx, nqy, nqz, n_rmu, n_rmu) with P(None, None, None, 'x', 'y')
        Z_q_3d: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, 'x', 'y')
        mesh_xy: 2D device mesh
        block_size: Tile block size (auto if None)
    
    Returns:
        zeta_3d: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, None, ('x','y'))
    """
    nqx, nqy, nqz, n_rmu, n_rmu2 = C_q_3d.shape
    _, _, _, _, n_zchunk = Z_q_3d.shape
    nq = nqx * nqy * nqz
    
    # Flatten q-grid
    C_q_flat = C_q_3d.reshape(nq, n_rmu, n_rmu2)
    Z_q_flat = Z_q_3d.reshape(nq, n_rmu, n_zchunk)
    
    # Ensure correct input sharding
    flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)
    Z_q_flat = jax.lax.with_sharding_constraint(Z_q_flat, flat_shard)
    
    # Solve using 2D blocked Cholesky
    zeta_flat = solve_zeta_q_blocked_2d(C_q_flat, Z_q_flat, mesh_xy, block_size)
    
    # Reshape back to 3D q-grid
    return zeta_flat.reshape(nqx, nqy, nqz, n_rmu, n_zchunk)


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
    
    This is the first step of zeta fitting - L_q is computed once and reused
    for all z-chunks.
    
    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix, sharded P(None, 'x', 'y')
        mesh_xy: 2D device mesh
        block_size: Tile block size (auto if None)
    
    Returns:
        L_q: (nq, n_rmu, n_rmu) Cholesky factor, sharded P(None, 'x', 'y')
    """
    import math
    
    nq, n_rmu, n_rmu2 = C_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"
    
    Pr = mesh_xy.shape['x']
    Pc = mesh_xy.shape['y']
    
    if block_size is None:
        target_J = math.lcm(Pr, Pc)
        if n_rmu % target_J == 0:
            block_size = n_rmu // target_J
        else:
            for j_mult in range(1, 10):
                J = target_J * j_mult
                if n_rmu % J == 0:
                    block_size = n_rmu // J
                    break
            else:
                block_size = max(1, n_rmu // (Pr * Pc))
    
    J = n_rmu // block_size
    
    assert n_rmu % block_size == 0, f"n_rmu={n_rmu} must divide block_size={block_size}"
    assert J % Pr == 0, f"J={J} must divide Pr={Pr}"
    assert J % Pc == 0, f"J={J} must divide Pc={Pc}"
    
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
        mesh_xy: 2D device mesh
        q_chunk_size: Number of q-points to solve simultaneously (default 1)
    
    Returns:
        zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, None, ('x','y'))
    """
    nq, n_rmu, _ = L_q.shape
    _, _, n_zchunk = Z_q.shape
    
    # Cache key for solve function (includes q_chunk_size)
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu, n_zchunk, q_chunk_size)
    
    if cache_key not in _solve_cache:
        # Reshard Z for column-parallel solve
        z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
        L_rep_shard = NamedSharding(mesh_xy, P(None, None))
        L_batch_rep_shard = NamedSharding(mesh_xy, P(None, None, None))  # (B_q, n_rmu, n_rmu)
        nq_c = nq
        q_chunk_c = min(q_chunk_size, nq)
        
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
        
        @jax.jit
        def _solve_all_q(L_q_sharded: jax.Array, Z_col_sharded: jax.Array) -> jax.Array:
            # Reshard Z for column-parallel solve
            Z_col = jax.lax.with_sharding_constraint(Z_col_sharded, z_col_shard)
            
            if q_chunk_c == 1:
                # Original q-by-q loop (minimum memory)
                def body(q, zeta_all):
                    L_single = L_q_sharded[q]
                    L_rep = jax.lax.with_sharding_constraint(L_single, L_rep_shard)
                    Z_single = Z_col[q]
                    zeta_q = _sharded_cho_solve(L_rep, Z_single)
                    return zeta_all.at[q].set(zeta_q)
                
                zeta_init = jnp.zeros_like(Z_col)
                return jax.lax.fori_loop(0, nq_c, body, zeta_init)
            else:
                # Q-chunked: gather B_q L matrices, solve in batch
                n_q_chunks = (nq_c + q_chunk_c - 1) // q_chunk_c
                
                def chunk_body(chunk_idx, zeta_all):
                    q_start = chunk_idx * q_chunk_c
                    q_end = jnp.minimum(q_start + q_chunk_c, nq_c)
                    actual_chunk = q_end - q_start
                    
                    # Gather chunk of L matrices (replicate on all devices)
                    L_chunk = jax.lax.dynamic_slice(
                        L_q_sharded, (q_start, 0, 0), (q_chunk_c, n_rmu, n_rmu)
                    )
                    L_chunk_rep = jax.lax.with_sharding_constraint(L_chunk, L_batch_rep_shard)
                    
                    # Get corresponding Z chunk
                    Z_chunk = jax.lax.dynamic_slice(
                        Z_col, (q_start, 0, 0), (q_chunk_c, n_rmu, Z_col.shape[2])
                    )
                    
                    # Batched solve
                    zeta_chunk = _sharded_cho_solve_batch(L_chunk_rep, Z_chunk)
                    
                    # Store results
                    return jax.lax.dynamic_update_slice(zeta_all, zeta_chunk, (q_start, 0, 0))
                
                zeta_init = jnp.zeros_like(Z_col)
                return jax.lax.fori_loop(0, n_q_chunks, chunk_body, zeta_init)
        
        _solve_cache[cache_key] = _solve_all_q
    
    return _solve_cache[cache_key](L_q, Z_q)


# Cache for ZCT computation function
_zct_cache = {}

def compute_ZCT_for_zchunk(
    P_k_mu_zchunk: jax.Array,
    kgrid: tuple[int, int, int],
    mesh_xy: Mesh,
) -> jax.Array:
    """
    Compute ZCT matrix for a single z-chunk.
    
    Pipeline: P_k,ab(μ,zchunk) -> P_R,ab -> Z_R = Σ_ab|P_R|² -> Z_q
    
    Args:
        P_k_mu_zchunk: (nk, ns, ns, n_rmu, n_zchunk) pair density
        kgrid: (nkx, nky, nkz)
        mesh_xy: Device mesh
    
    Returns:
        Z_q: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, 'x', 'y')
    """
    nkx, nky, nkz = kgrid
    nk, ns1, ns2, n_rmu, n_zchunk = P_k_mu_zchunk.shape
    
    # Cache key based on shapes
    cache_key = ('ZCT', id(mesh_xy), kgrid, ns1, ns2, n_rmu, n_zchunk)
    
    if cache_key not in _zct_cache:
        in_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
        out_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
        
        # Capture shape constants in closure
        ns1_c, ns2_c, n_rmu_c, n_zchunk_c = ns1, ns2, n_rmu, n_zchunk
        
        @partial(jax.jit, in_shardings=in_xy, out_shardings=out_xy,
                 static_argnames=('nkx', 'nky', 'nkz'))
        def _compute_ZCT(P_mu_z: jax.Array, nkx: int, nky: int, nkz: int) -> jax.Array:
            # Input: (nk, ns1, ns2, n_rmu, n_zchunk)
            # Reshape to expose k-dimensions: (nkx, nky, nkz, ns1, ns2, n_rmu, n_zchunk)
            P_k = P_mu_z.reshape(nkx, nky, nkz, ns1_c, ns2_c, n_rmu_c, n_zchunk_c)
            
            # IFFT over k-dimensions directly - NO TRANSPOSE NEEDED
            # JAX FFT supports arbitrary axes, avoiding a full copy
            P_R = jnp.fft.ifftn(P_k, axes=(0, 1, 2), norm='ortho')
            # P_R: (Rx, Ry, Rz, ns1, ns2, n_rmu, n_zchunk)
            
            # Frobenius norm squared over spin (axes 3, 4 are spin dimensions)
            Z_R = jnp.sum(jnp.abs(P_R) ** 2, axis=(3, 4))
            # Z_R: (Rx, Ry, Rz, n_rmu, n_zchunk)
            
            # FFT over R-dimensions directly - NO TRANSPOSE NEEDED
            Z_q = jnp.fft.fftn(Z_R, axes=(0, 1, 2), norm='ortho')
            # Z_q: (qx, qy, qz, n_rmu, n_zchunk) - already in correct order!
            
            return Z_q
        
        _zct_cache[cache_key] = _compute_ZCT
    
    return _zct_cache[cache_key](P_k_mu_zchunk, nkx, nky, nkz)


def fit_zeta_chunked_to_h5(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    mesh_xy: Mesh,
    z_chunk_size: int,
    output_file: str,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = True,
    use_gspace_cache: bool = True,
):
    """
    Full zeta fitting pipeline with z-chunk loop and HDF5 output.
    
    Workflow:
    1. Load wavefunctions (band-chunked FFT)
    2. Compute C_q from P_k(r_mu, r_mu) - once
    3. Compute L_q = chol(C_q) using 2D blocked algorithm - once
    4. For each z-chunk:
       a. Compute psi_nk,a(r_chunk) via FFT
       b. Compute P_k,ab(r_mu, r_chunk)
       c. Compute Z_q via ortho FFT pipeline
       d. Solve zeta_q = L^{-H}(L^{-1} Z_q) (q-chunked)
       e. Write zeta_q chunk to HDF5
    
    Args:
        wfn: WFNReader object
        sym: SymMaps object
        meta: Meta object with system info
        centroid_indices: ISDF centroid indices
        mesh_xy: 2D device mesh
        z_chunk_size: Number of z-slices per chunk
        output_file: Path to output HDF5 file
        band_chunk_size: Bands to process at once (memory control)
        q_chunk_size: Q-points to solve simultaneously (memory vs parallelism trade-off)
        bispinor: Whether to use bispinor wavefunctions
        use_gspace_cache: If True, cache G-space across z-chunks (faster but uses more memory).
                         If False, reload from HDF5 each z-chunk (slower but less memory).
    
    Returns:
        None (writes to HDF5)
    """
    import h5py
    
    nx, ny, nz = meta.fft_grid
    n_rmu = meta.n_rmu
    n_rtot = meta.n_rtot
    nk_tot = meta.nk_tot
    kgrid = meta.kgrid
    nqx, nqy, nqz = kgrid
    nq = nqx * nqy * nqz
    
    # Number of z-chunks
    num_z_chunks = (nz + z_chunk_size - 1) // z_chunk_size
    n_zchunk = nx * ny * z_chunk_size
    
    print(f"\n{'='*60}")
    print(f"Zeta fitting: {num_z_chunks} z-chunks, {n_zchunk} points each")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    # Band range for pair density (all occupied + sigma window)
    band_range = (meta.b_id_0, meta.b_id_3)
    
    # ========== STEP 1: Load wavefunctions at centroids (band-chunked) ==========
    with timing.section("zeta_fit.load_wfns"):
        psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
            wfn, sym, meta, centroid_indices, bispinor, mesh_xy, band_range,
            band_chunk_size=band_chunk_size
        )
    
    # ========== STEP 2: Compute CCT (C_q) from r_mu × r_mu ==========
    with timing.section("zeta_fit.CCT"):
        print("\nComputing pair density P_k(r_mu, r_mu)...")
        P_k_mumu = compute_pair_density_k(psi_rmuT_X, psi_rmu_Y, mesh_xy)
        P_k_mumu.block_until_ready()
        
        print("Computing C_q via ortho FFT pipeline...")
        C_q = compute_CCT_only(P_k_mumu, kgrid, mesh_xy)
        C_q.block_until_ready()
        # C_q: (nqx, nqy, nqz, n_rmu, n_rmu)
        
        # Free P_k_mumu immediately - we only needed it for C_q
        del P_k_mumu
        
        # Flatten for Cholesky
        C_q_flat = C_q.reshape(nq, n_rmu, n_rmu)
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)
    
    # ========== STEP 3: Compute L_q = chol(C_q) once ==========
    with timing.section("zeta_fit.cholesky"):
        print("\nComputing L_q = chol(C_q) using 2D blocked algorithm...")
        L_q = compute_L_q_from_CCT(C_q_flat, mesh_xy)
        L_q.block_until_ready()
        print(f"  L_q shape: {L_q.shape}")
    
    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget
    del C_q, C_q_flat
    # Force garbage collection and JAX device memory cleanup
    import gc
    gc.collect()
    jax.clear_caches()  # Clear JAX function caches that may hold array refs
    
    # ========== STEP 4: Create HDF5 file ==========
    # Only rank 0 creates the file structure
    if jax.process_index() == 0:
        with h5py.File(output_file, 'w') as f:
            # Create dataset for full zeta
            f.create_dataset(
                'zeta_q',
                shape=(nqx, nqy, nqz, n_rmu, n_rtot),
                dtype=np.complex128,
                chunks=(1, 1, 1, n_rmu, n_zchunk),  # Chunk by z-slice
            )
            # Store metadata
            f.attrs['n_rmu'] = n_rmu
            f.attrs['n_rtot'] = n_rtot
            f.attrs['fft_grid'] = meta.fft_grid
            f.attrs['kgrid'] = kgrid
            f.attrs['z_chunk_size'] = z_chunk_size
            f.attrs['num_z_chunks'] = num_z_chunks
    
    # Synchronize before writing
    jax.experimental.multihost_utils.sync_global_devices("zeta_h5_create")
    
    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each z-chunk. Memory cost: ~0.5-1 GB (fits within budget).
    # For large systems (many k-points), caching may be disabled to save memory.
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]
    
    if use_gspace_cache:
        with timing.section("zeta_fit.cache_gspace"):
            print("\nCaching G-space wavefunctions for z-chunk loop...")
            cached_gspace = load_gspace_for_bands(
                wfn, sym, meta, mesh_xy, band_range, bispinor, band_chunk_size
            )
            print(f"  Cached {len(cached_gspace)} band chunks (sharded across devices)")
    else:
        cached_gspace = None
        print("\nG-space caching DISABLED (too large for memory budget)")
        print("  Will reload from HDF5 each z-chunk (slower)")
    
    # ========== STEP 6: Loop over z-chunks ==========
    # Track timing for summary (manual perf_counter for detailed breakdown)
    t_load_total = 0.0
    t_pair_total = 0.0
    t_zct_total = 0.0
    t_solve_total = 0.0
    t_write_total = 0.0
    t_chunk_start = time.perf_counter()
    
    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_z_chunks):
            z_start = chunk_idx * z_chunk_size
            z_end = min(z_start + z_chunk_size, nz)
            actual_z_size = z_end - z_start
            actual_n_zchunk = nx * ny * actual_z_size
            
            r_start = chunk_idx * n_zchunk
            r_end = r_start + actual_n_zchunk
            
            print(f"Chunk {chunk_idx+1}/{num_z_chunks}: r=[{r_start}:{r_end}]", end="\r")
            
            # 6a. Get psi_nk,a(r_chunk)
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.load"):
                if cached_gspace is not None:
                    # Fast path: FFT only (G-space is cached)
                    psi_zchunk_Y = get_psi_zchunk_from_cached(
                        cached_gspace, meta, mesh_xy, band_range,
                        z_start, z_end, kvecs_frac,
                        band_chunk_size=band_chunk_size
                    )
                else:
                    # Slow path: reload from HDF5 each chunk
                    psi_zchunk_Y = get_psi_zchunk(
                        wfn, sym, meta, mesh_xy, band_range,
                        z_start, z_end, bispinor,
                        band_chunk_size=band_chunk_size
                    )
                psi_zchunk_Y.block_until_ready()
            t_load_total += time.perf_counter() - t0
            
            # 5b. Compute P_k,ab(r_mu, r_chunk)
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.pair_density"):
                P_k_mu_zchunk = compute_pair_density_k_zchunk(
                    psi_rmuT_X, psi_zchunk_Y, mesh_xy
                )
                P_k_mu_zchunk.block_until_ready()
            t_pair_total += time.perf_counter() - t0
            
            # Free psi_zchunk_Y immediately - we have P_k now
            del psi_zchunk_Y
            
            # 5c. Compute Z_q via ortho FFT pipeline
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.ZCT"):
                Z_q = compute_ZCT_for_zchunk(P_k_mu_zchunk, kgrid, mesh_xy)
                Z_q.block_until_ready()
                
                # Free P_k immediately - we have Z_q now
                del P_k_mu_zchunk
                
                Z_q_flat = Z_q.reshape(nq, n_rmu, actual_n_zchunk)
                Z_q_flat = jax.lax.with_sharding_constraint(Z_q_flat, flat_shard)
                Z_q_flat.block_until_ready()
                
                # Free Z_q (3D) - we have Z_q_flat now
                del Z_q
            t_zct_total += time.perf_counter() - t0
            
            # 5d. Solve zeta_q = L^{-H}(L^{-1} Z_q)
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.solve"):
                zeta_chunk = solve_zeta_from_L_q(L_q, Z_q_flat, mesh_xy, q_chunk_size)
                zeta_chunk.block_until_ready()
                
                # Free Z_q_flat - we have zeta now
                del Z_q_flat
            t_solve_total += time.perf_counter() - t0
            
            # 5e. Gather to host and write to HDF5
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                # Write q-by-q to avoid gathering full (nq × n_rmu × n_zchunk) to host
                # Each q gather is only (n_rmu × n_zchunk) ≈ 160 MB for large systems
                if jax.process_index() == 0:
                    with h5py.File(output_file, 'a') as f:
                        for q_flat in range(nq):
                            # Get one q-point at a time
                            zeta_q_single = np.asarray(zeta_chunk[q_flat])  # (n_rmu, n_zchunk)
                            # Convert flat q to 3D indices
                            qx = q_flat // (nqy * nqz)
                            qy = (q_flat % (nqy * nqz)) // nqz
                            qz = q_flat % nqz
                            f['zeta_q'][qx, qy, qz, :, r_start:r_end] = zeta_q_single
                
                # Synchronize
                jax.experimental.multihost_utils.sync_global_devices(f"zeta_chunk_{chunk_idx}")
            t_write_total += time.perf_counter() - t0
            
    
    t_chunks_total = time.perf_counter() - t_chunk_start
    
    # Free cached G-space now that chunk loop is done
    if cached_gspace is not None:
        del cached_gspace
    
    # Print summary
    print()  # Clear the \r line
    print(f"\nWritten to {output_file}")
    print(f"{'='*60}")
    print(f"Zeta fitting complete!")
    print(f"  Shape: ({nqx}, {nqy}, {nqz}, {n_rmu}, {n_rtot})")
    print(f"{'='*60}")
    print(f"\nTiming Summary ({num_z_chunks} z-chunks):")
    print(f"  {'Phase':<20} {'Total':>10} {'Per-chunk':>12} {'%':>6}")
    print(f"  {'-'*50}")
    print(f"  {'Load zchunk':<20} {t_load_total:>10.2f}s {t_load_total/num_z_chunks*1000:>10.1f}ms {100*t_load_total/t_chunks_total:>6.1f}%")
    print(f"  {'Pair density':<20} {t_pair_total:>10.2f}s {t_pair_total/num_z_chunks*1000:>10.1f}ms {100*t_pair_total/t_chunks_total:>6.1f}%")
    print(f"  {'ZCT (FFT pipeline)':<20} {t_zct_total:>10.2f}s {t_zct_total/num_z_chunks*1000:>10.1f}ms {100*t_zct_total/t_chunks_total:>6.1f}%")
    print(f"  {'Solve (L^-1 Z)':<20} {t_solve_total:>10.2f}s {t_solve_total/num_z_chunks*1000:>10.1f}ms {100*t_solve_total/t_chunks_total:>6.1f}%")
    print(f"  {'H5 write':<20} {t_write_total:>10.2f}s {t_write_total/num_z_chunks*1000:>10.1f}ms {100*t_write_total/t_chunks_total:>6.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'Chunk loop total':<20} {t_chunks_total:>10.2f}s {t_chunks_total/num_z_chunks*1000:>10.1f}ms")
    print(f"  {'Per r-point':<20} {'':<10} {t_chunks_total/n_rtot*1e6:>10.1f}μs")


def load_gspace_for_bands(
    wfn, sym, meta, mesh_xy, band_range, bispinor,
    band_chunk_size: int = 16,
) -> list[tuple[jax.Array, tuple[int, int]]]:
    """
    Load G-space wavefunctions for all band chunks ONCE.
    
    This caches the expensive HDF5 read + scatter operation so it can be
    reused across multiple z-chunk iterations. Memory cost is ~0.5-1 GB
    for typical systems (nk * nb * ns * fft_grid * 16 bytes).
    
    Args:
        wfn: WFNReader
        sym: SymMaps
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end) - total bands needed
        bispinor: Whether to use bispinor
        band_chunk_size: Bands to process at once
    
    Returns:
        List of (global_psi_Gtot, bc_range) for each band chunk
    """
    b_start, b_end = band_range
    nb_total = b_end - b_start
    num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
    
    cached_gspace = []
    for bc_idx in range(num_band_chunks):
        bc_start = b_start + bc_idx * band_chunk_size
        bc_end = min(bc_start + band_chunk_size, b_end)
        bc_range = (bc_start, bc_end)
        
        # Load G-space for this band chunk
        global_psi_Gtot, _ = read_Gvecs_to_devices(wfn, sym, bc_range, meta, bispinor, mesh_xy)
        cached_gspace.append((global_psi_Gtot, bc_range))
    
    return cached_gspace


def get_psi_zchunk_from_cached(
    cached_gspace: list[tuple[jax.Array, tuple[int, int]]],
    meta, mesh_xy, band_range, z_start, z_end, kvecs_frac,
    band_chunk_size: int = 16,
) -> jax.Array:
    """
    Extract z-chunk from pre-loaded G-space (FFT only, no HDF5 read).
    
    This is the fast path that reuses cached G-space across z-chunk iterations.
    
    Args:
        cached_gspace: Pre-loaded G-space from load_gspace_for_bands()
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end) - total bands needed
        z_start, z_end: Z-slice range
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        band_chunk_size: Bands to FFT at once
    
    Returns:
        psi_zchunk_Y: (nk, nb, ns, n_zchunk) with P(None, None, None, 'y')
    """
    nx, ny, nz = meta.fft_grid
    n_zchunk = nx * ny * (z_end - z_start)
    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    
    # Output sharding
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    
    # Allocate output array for all bands (z-chunk is small enough)
    psi_zchunk_all = jnp.zeros((nk_tot, nb_total, nspinor, n_zchunk), dtype=jnp.complex128)
    psi_zchunk_all = jax.lax.with_sharding_constraint(psi_zchunk_all, out_Y)
    
    # Process each cached band chunk - FFT only (no HDF5 read)
    for bc_idx, (global_psi_Gtot, bc_range) in enumerate(cached_gspace):
        nb_chunk = bc_range[1] - bc_range[0]
        
        # FFT and extract z-slice for this chunk
        psi_zchunk_chunk = get_sharded_wfns_zchunk_slice(
            global_psi_Gtot, meta, z_start, z_end, kvecs_frac, mesh_xy, bc_range
        )
        
        # Place into output array at correct band indices
        local_bc_start = bc_idx * band_chunk_size
        local_bc_end = local_bc_start + nb_chunk
        psi_zchunk_all = psi_zchunk_all.at[:, local_bc_start:local_bc_end, :, :].set(psi_zchunk_chunk)
        
        # Free FFT output only (keep G-space cached)
        del psi_zchunk_chunk
    
    return psi_zchunk_all


def get_psi_zchunk(
    wfn, sym, meta, mesh_xy, band_range, z_start, z_end, bispinor,
    band_chunk_size: int = 16,
) -> jax.Array:
    """
    Load and FFT wavefunctions for a specific z-chunk.
    
    NOTE: This function reloads G-space from HDF5 each call. For multiple
    z-chunks, use load_gspace_for_bands() + get_psi_zchunk_from_cached()
    to avoid redundant HDF5 reads.
    
    Uses band chunking to limit memory during FFT step:
    - Loop over band chunks
    - FFT each chunk to real-space (the memory bottleneck)
    - Extract z-slice and accumulate into output array
    
    The final psi_zchunk has all bands but only the z-slice, which is
    small enough to hold in memory for downstream pair density computation.
    
    Args:
        wfn: WFNReader
        sym: SymMaps
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end) - total bands needed
        z_start, z_end: Z-slice range
        bispinor: Whether to use bispinor
        band_chunk_size: Bands to FFT at once (memory control for FFT step)
    
    Returns:
        psi_zchunk_Y: (nk, nb, ns, n_zchunk) with P(None, None, None, 'y')
    """
    # Get k-vectors from sym (as fractions of kgrid)
    kgrid = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid[None, :]  # (nk, 3) in fractional coords
    
    # Load G-space and extract z-chunk (non-cached path)
    cached_gspace = load_gspace_for_bands(
        wfn, sym, meta, mesh_xy, band_range, bispinor, band_chunk_size
    )
    
    result = get_psi_zchunk_from_cached(
        cached_gspace, meta, mesh_xy, band_range, z_start, z_end, kvecs_frac,
        band_chunk_size
    )
    
    # Free cached G-space
    del cached_gspace
    
    return result


# Cache for zchunk extraction function
_zchunk_slice_cache = {}


def get_sharded_wfns_zchunk_slice(
    global_psi_Gtot: jax.Array,
    meta: Meta,
    z_start: int,
    z_end: int,
    kvecs_frac: np.ndarray,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
) -> jax.Array:
    """
    FFT wavefunctions and extract z-chunk via slicing (not gather).
    
    Uses z-axis slicing before flattening for better XLA optimization.
    This avoids the "involuntary full rematerialization" warning that
    occurs with gather-based resharding.
    
    Args:
        global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
        meta: Meta object
        z_start, z_end: Z-slice range [z_start, z_end)
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
    
    Returns:
        psi_zchunk_Y: (nk, nb, ns, n_zchunk) with P(None, None, None, 'y')
    """
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    fft_grid = meta.fft_grid
    nx, ny, nz = fft_grid
    z_chunk_size = z_end - z_start
    n_zchunk = nx * ny * z_chunk_size
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rtot = nx * ny * nz
    
    # Cache key - use hash of kvecs since it's constant for a given system
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = ('zchunk_slice', id(mesh_xy), nk_tot, nspinor, z_chunk_size, nx, ny, nz, kvecs_hash)
    
    if cache_key not in _zchunk_slice_cache:
        out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        
        sharded_ifftn = make_sharded_ifftn_3d(
            mesh_xy, 
            P(None, ('x', 'y'), None, None, None, None),
            P(None, ('x', 'y'), None, None, None, None)
        )
        
        # Intermediate sharding for staged reshard: gather bands over X first
        stage1_shard = NamedSharding(mesh_xy, P(None, 'y', None, None))
        
        # Pre-compute phase grids and kvecs ONCE in closure (not passed as args)
        fx_cached = jnp.arange(nx, dtype=jnp.float64)[None, :, None, None] / nx
        fy_cached = jnp.arange(ny, dtype=jnp.float64)[None, None, :, None] / ny
        fz_cached = jnp.arange(nz, dtype=jnp.float64)[None, None, None, :] / nz
        kvecs_cached = jnp.asarray(kvecs_frac)  # Cache kvecs in closure
        n_rtot_cached = n_rtot
        
        # z_chunk_size is static (from cache key), z_start is dynamic
        z_chunk_size_static = z_chunk_size
        
        @partial(jax.jit, static_argnames=('nb_static',))
        def _extract_zchunk_slice(psi_G, z_start_dyn, nb_static):
            # FFT to real space: (nk, nb_padded, ns, nx, ny, nz)
            psi_r = sharded_ifftn(psi_G)
            
            # Apply Bloch phase exp(ik·r) - use cached phase grids and kvecs from closure
            phase_spatial = jnp.exp(
                2j * jnp.pi * (
                    kvecs_cached[:, 0:1, None, None] * fx_cached
                    + kvecs_cached[:, 1:2, None, None] * fy_cached
                    + kvecs_cached[:, 2:3, None, None] * fz_cached
                )
            )
            psi_r = psi_r * phase_spatial[:, None, None, :, :, :]
            
            # Normalize
            psi_r = psi_r * jnp.sqrt(n_rtot_cached)
            
            # Trim bands
            psi_r = psi_r[:, :nb_static, :, :, :, :]
            
            # Slice z-axis with DYNAMIC start, STATIC size
            # psi_r: (nk, nb, ns, nx, ny, nz) -> (nk, nb, ns, nx, ny, z_chunk)
            psi_zslice = jax.lax.dynamic_slice(
                psi_r,
                (0, 0, 0, 0, 0, z_start_dyn),
                (nk_tot, nb_static, nspinor, nx, ny, z_chunk_size_static)
            )
            
            # Flatten spatial dims: (nk, nb, ns, nx*ny*z_chunk)
            psi_zchunk = psi_zslice.reshape(nk_tot, nb_static, nspinor, -1)
            
            # STAGED RESHARD to avoid XLA "involuntary full rematerialization":
            # Direct reshard from P(None, ('x','y'), None, None) → P(None, None, None, 'y')
            # causes XLA to replicate the entire array before repartitioning.
            # By breaking into two steps, XLA can use efficient collectives:
            #
            # Stage 1: P(None, ('x','y'), None, None) → P(None, 'y', None, None)
            #          all-gather bands over X axis only
            psi_zchunk = jax.lax.with_sharding_constraint(psi_zchunk, stage1_shard)
            
            # Stage 2: P(None, 'y', None, None) → P(None, None, None, 'y')
            #          all-gather bands over Y, then slice zchunk for Y position
            psi_zchunk = jax.lax.with_sharding_constraint(psi_zchunk, out_Y)
            
            return psi_zchunk
        
        _zchunk_slice_cache[cache_key] = _extract_zchunk_slice
    
    # Call cached function - psi_G and z_start are dynamic, nb is static per band chunk
    return _zchunk_slice_cache[cache_key](global_psi_Gtot, z_start, nb)


# ============================================================================
# Unified band-chunked FFT backend for centroid and z-chunk extraction
# ============================================================================

# Cache for centroid extraction function
_centroid_extract_cache = {}


def get_sharded_wfns_centroids(
    global_psi_Gtot: jax.Array,
    meta: Meta,
    centroid_indices: jax.Array,
    kvecs_frac: np.ndarray,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    """
    FFT wavefunctions and extract centroids for a single band chunk.
    
    This is the centroid-extraction counterpart to get_sharded_wfns_zchunk_slice.
    Both use the same caching and staging patterns for memory efficiency.
    
    Args:
        global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
        meta: Meta object
        centroid_indices: (n_rmu, 3) centroid grid coordinates
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
    
    Returns:
        psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
        psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
    """
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    fft_grid = meta.fft_grid
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rmu = len(centroid_indices)
    
    # Cache key
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = ('centroid_extract', id(mesh_xy), nk_tot, nspinor, n_rmu, nx, ny, nz, kvecs_hash)
    
    if cache_key not in _centroid_extract_cache:
        out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
        null_4 = NamedSharding(mesh_xy, P(None, None, None, None))
        stage1_shard = NamedSharding(mesh_xy, P(None, 'y', None, None))
        
        sharded_ifftn = make_sharded_ifftn_3d(
            mesh_xy,
            P(None, ('x', 'y'), None, None, None, None),
            P(None, ('x', 'y'), None, None, None, None)
        )
        
        # Pre-compute phase grids and kvecs in closure
        fx_cached = jnp.arange(nx, dtype=jnp.float64)[None, :, None, None] / nx
        fy_cached = jnp.arange(ny, dtype=jnp.float64)[None, None, :, None] / ny
        fz_cached = jnp.arange(nz, dtype=jnp.float64)[None, None, None, :] / nz
        kvecs_cached = jnp.asarray(kvecs_frac)
        n_rtot_cached = n_rtot
        
        # Pre-compute centroid linear indices
        centroids = jnp.asarray(centroid_indices, dtype=jnp.int32)
        centroid_lin = (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(jnp.int32)
        
        @partial(jax.jit, static_argnames=('nb_static',))
        def _extract_centroids(psi_G, nb_static):
            # FFT to real space
            psi_r = sharded_ifftn(psi_G)
            
            # Apply Bloch phase
            phase_spatial = jnp.exp(
                2j * jnp.pi * (
                    kvecs_cached[:, 0:1, None, None] * fx_cached
                    + kvecs_cached[:, 1:2, None, None] * fy_cached
                    + kvecs_cached[:, 2:3, None, None] * fz_cached
                )
            )
            psi_r = psi_r * phase_spatial[:, None, None, :, :, :]
            
            # Normalize and trim bands
            psi_r = psi_r * jnp.sqrt(n_rtot_cached)
            psi_r = psi_r[:, :nb_static, :, :, :, :]
            
            # Flatten spatial dims
            psi_rtot = psi_r.reshape(nk_tot, nb_static, nspinor, -1)
            
            # Gather centroids using pre-computed linear indices
            # NOTE: We do NOT replicate first - JAX handles the gather efficiently
            # with bands still sharded, avoiding a massive memory spike.
            # Old code used null_4 replication which required 250x more temp memory!
            psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)
            
            # Create psi_rmuT (conjugate transpose for left wfn in pair density)
            psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))  # (nk, n_rmu, nb, ns)
            
            # Apply output shardings
            psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, out_Y)
            psi_rmuT = jax.lax.with_sharding_constraint(psi_rmuT, out_X)
            
            return psi_rmu, psi_rmuT
        
        _centroid_extract_cache[cache_key] = _extract_centroids
    
    return _centroid_extract_cache[cache_key](global_psi_Gtot, nb)


def load_centroids_band_chunked(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    bispinor: bool,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
    band_chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """
    Load centroid-sampled wavefunctions using band chunking.
    
    Memory-safe version that loops over band chunks to avoid OOM
    when loading all bands at once for FFT.
    
    This is the unified band-chunked backend used by fit_zeta_chunked_to_h5.
    
    Args:
        wfn: WFNReader
        sym: SymMaps
        meta: Meta object
        centroid_indices: (n_rmu, 3) centroid grid coordinates
        bispinor: Whether to use bispinor
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
        band_chunk_size: Bands to FFT at once (memory control)
    
    Returns:
        psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
        psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
    """
    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    n_rmu = len(centroid_indices)
    
    # Get k-vectors
    kgrid = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid[None, :]
    
    # Output shardings
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
    
    # Allocate output arrays for all bands
    psi_rmu_all = jnp.zeros((nk_tot, nb_total, nspinor, n_rmu), dtype=jnp.complex128)
    psi_rmuT_all = jnp.zeros((nk_tot, n_rmu, nb_total, nspinor), dtype=jnp.complex128)
    psi_rmu_all = jax.lax.with_sharding_constraint(psi_rmu_all, out_Y)
    psi_rmuT_all = jax.lax.with_sharding_constraint(psi_rmuT_all, out_X)
    
    # Process bands in chunks
    num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
    
    for bc_idx in range(num_band_chunks):
        bc_start = b_start + bc_idx * band_chunk_size
        bc_end = min(bc_start + band_chunk_size, b_end)
        bc_range = (bc_start, bc_end)
        nb_chunk = bc_end - bc_start
        
        # Load G-space for this band chunk
        global_psi_Gtot, _ = read_Gvecs_to_devices(wfn, sym, bc_range, meta, bispinor, mesh_xy)
        
        # FFT and extract centroids for this chunk
        psi_rmu_chunk, psi_rmuT_chunk = get_sharded_wfns_centroids(
            global_psi_Gtot, meta, centroid_indices, kvecs_frac, mesh_xy, bc_range
        )
        
        # Place into output arrays
        local_bc_start = bc_idx * band_chunk_size
        local_bc_end = local_bc_start + nb_chunk
        psi_rmu_all = psi_rmu_all.at[:, local_bc_start:local_bc_end, :, :].set(psi_rmu_chunk)
        psi_rmuT_all = psi_rmuT_all.at[:, :, local_bc_start:local_bc_end, :].set(psi_rmuT_chunk)
        
        # Free memory
        del global_psi_Gtot, psi_rmu_chunk, psi_rmuT_chunk
    
    return psi_rmu_all, psi_rmuT_all


def read_Gvecs_and_get_sharded_wfns(wfn, sym, meta, centroid_indices, bispinor, mesh_xy, band_range):
    """
    Combined wavefunction loading: read G-space + FFT + extract centroids.
    
    WARNING: This loads all bands at once - can cause OOM for large systems.
    For memory-safe loading, use load_centroids_band_chunked instead.
    
    Returns:
        psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
        psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
        psi_rtot_Y: (nk, nb, ns, n_rtot) with P(None, None, None, 'y') - full grid
    """
    nb = band_range[1] - band_range[0]
    
    # Load to G-space (note argument order: wfn, sym, bandrange, meta, bispinor, mesh_xy)
    global_psi_Gtot, nb_actual = read_Gvecs_to_devices(wfn, sym, band_range, meta, bispinor, mesh_xy)
    
    # FFT + extract centroids (note: is_left=False for right wfn)
    psi_rtot_Y, psi_rmu_Y, psi_rmuT_X = get_sharded_wfns(
        global_psi_Gtot, sym, meta, centroid_indices, nb, False, mesh_xy
    )
    
    return psi_rmu_Y, psi_rmuT_X, psi_rtot_Y

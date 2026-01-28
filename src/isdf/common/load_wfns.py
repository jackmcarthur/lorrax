import gc
import math
import queue
import threading
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


def compute_block_size_for_2d_cholesky(n_rmu: int, Pr: int, Pc: int) -> tuple[int, int]:
    """
    Compute block size for 2D blocked Cholesky that satisfies distribution constraints.
    
    Constraints (fundamental to 2D blocked algorithms):
        - n_rmu % block_size == 0  (matrix divides into whole tiles)
        - J % Pr == 0              (tile rows distribute evenly on X-axis)
        - J % Pc == 0              (tile cols distribute evenly on Y-axis)
    
    Where J = n_rmu / block_size is the number of tiles per dimension.
    
    The simplest solution: J = lcm(Pr, Pc), giving block_size = n_rmu / J.
    If n_rmu doesn't divide evenly, we try multiples of lcm(Pr, Pc).
    
    Args:
        n_rmu: Matrix dimension (number of ISDF centroids)
        Pr: Number of devices on X-axis
        Pc: Number of devices on Y-axis
    
    Returns:
        (block_size, J) tuple
    
    Raises:
        ValueError: If no valid block size exists (n_rmu incompatible with mesh)
    """
    target_J = math.lcm(Pr, Pc)
    
    if n_rmu % target_J == 0:
        block_size = n_rmu // target_J
        return block_size, target_J
    
    # Try multiples of lcm(Pr, Pc)
    for j_mult in range(2, 20):
        J = target_J * j_mult
        if n_rmu % J == 0:
            block_size = n_rmu // J
            return block_size, J
    
    # Last resort: find any valid block size
    for b in range(n_rmu, 0, -1):
        if n_rmu % b == 0:
            J = n_rmu // b
            if J % Pr == 0 and J % Pc == 0:
                return b, J
    
    raise ValueError(
        f"No valid block size for n_rmu={n_rmu} with mesh {Pr}×{Pc}. "
        f"n_rmu should be divisible by lcm({Pr},{Pc})={target_J} or a multiple thereof."
    )


# ============================================================================
# shard_map based FFT - runs FFT independently on each device's local data
# See: https://docs.jax.dev/en/latest/notebooks/shard_map.html
# ============================================================================

def make_sharded_ifftn_3d(mesh: Mesh, in_spec: P, out_spec: P):
    """
    Uses shard_map to run IFFT independently on each device's local data.
    The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
    
    This works around an XLA bug where regular jnp.fft on sharded arrays
    triggers unnecessary all-gathers even when FFT axes are replicated.
    
    Args:
        mesh: The device mesh
        in_spec: PartitionSpec for input (e.g., P(None, ('x','y'), None, None, None, None))
        out_spec: PartitionSpec for output (same as in_spec for FFT)
    
    Returns:
        A function that performs 3D IFFT on sharded data
    """
    def _local_ifftn(x_local):
        # Each device runs FFT on its local shard independently
        return jnp.fft.ifftn(x_local, axes=(-3, -2, -1))
    
    return shard_map(_local_ifftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


def make_sharded_fftn_3d(mesh: Mesh, in_spec: P, out_spec: P):
    """
    Uses shard_map to run FFT independently on each device's local data.
    The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
    
    This works around an XLA bug where regular jnp.fft on sharded arrays
    triggers unnecessary all-gathers even when FFT axes are replicated.
    
    Args:
        mesh: The device mesh
        in_spec: PartitionSpec for input (e.g., P(None, None, None, 'x', 'y'))
        out_spec: PartitionSpec for output (same as in_spec for FFT)
    
    Returns:
        A function that performs 3D FFT on sharded data
    """
    def _local_fftn(x_local):
        # Each device runs FFT on its local shard independently
        return jnp.fft.fftn(x_local, axes=(-3, -2, -1))
    
    return shard_map(_local_fftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


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

			# Centroid gather on axis 3 (spatial) - this axis is replicated so gather is LOCAL
			# Bands stay sharded on axis 1 throughout, no all-gather needed
			psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)

			# Conjugate-transpose to (nk, n_rmu, nb, nspinor) for pair density
			psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))  # (nk, n_rmu, nb, ns)
			
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
# Spin-traced pair density (matching cohsex_jax treatment)
# ============================================================================
# cohsex_jax traces over spin for ISDF fitting:
#   P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(r_μ) × ψ_{n,k,s}(r_ν)
# This reduces lstsq error by fitting a lower-rank object.

def compute_pair_density_spin_traced(
	psi_rmuT_X: jax.Array,
	psi_rmu_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute spin-traced pair density P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(ν).
	
	This matches cohsex_jax spin treatment for ISDF fitting.
	
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


def compute_pair_density_spin_traced_zchunk(
	psi_rmuT_X: jax.Array,
	psi_zchunk_Y: jax.Array,
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Spin-traced pair density for z-chunk: P_k(μ,r) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(r).
	
	Input shapes and shardings:
		psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
		psi_zchunk_Y: (nk, nb, ns, n_zchunk) with P(None, None, None, 'y')
	
	Output:
		P_k_zchunk: (nk, n_rmu, n_zchunk) with P(None, 'x', 'y')
	"""
	nk, n_rmu, nb, ns = psi_rmuT_X.shape
	_, _, _, n_zchunk = psi_zchunk_Y.shape
	cache_key = ('spin_traced_zchunk', id(mesh_xy), nk, n_rmu, nb, ns, n_zchunk)
	
	if cache_key not in _compute_pair_density_cache:
		x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		xy_out = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		
		@partial(jax.jit, in_shardings=(x1_4, y3_4), out_shardings=xy_out)
		def _compute_P_traced_z(psi_L: jax.Array, psi_R: jax.Array) -> jax.Array:
			return jnp.einsum('kmns,knsr->kmr', psi_L, psi_R, optimize=True)
		
		_compute_pair_density_cache[cache_key] = _compute_P_traced_z
	
	return _compute_pair_density_cache[cache_key](psi_rmuT_X, psi_zchunk_Y)


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
	Compute CCT and ZCT matrices via ortho FFT: P_k -> P_R -> C_q/Z_q.
	Uses Frobenius norm over spin: C_R(μ,ν) = Σ_ab |P_R,ab(μ,ν)|²
	See docs/isdf_spin_galerkin_derivation.md for derivation.
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
			P_R_mumu = jnp.fft.ifftn(P_k_mumu, axes=(0, 1, 2), norm='ortho')
			# P_R_mumu: (Rx, Ry, Rz, ns, ns, n_rmu, n_rmu)
			
			# Frobenius norm squared over spin (axes 3, 4)
			C_R = jnp.sum(jnp.abs(P_R_mumu) ** 2, axis=(3, 4))
			C_q = jnp.fft.fftn(C_R, axes=(0, 1, 2), norm='ortho')
			# C_q: (qx, qy, qz, n_rmu, n_rmu) - already correct!
			
			# ---- ZCT pathway ----
			# Input: (nk, ns, ns, n_rmu, n_zchunk)
			P_k_muz = P_mu_z.reshape(nkx, nky, nkz, ns1, ns2, n_rmu, n_zchunk)
			P_R_muz = jnp.fft.ifftn(P_k_muz, axes=(0, 1, 2), norm='ortho')
			# P_R_muz: (Rx, Ry, Rz, ns, ns, n_rmu, n_zchunk)
			
			# Frobenius norm squared over spin (axes 3, 4)
			Z_R = jnp.sum(jnp.abs(P_R_muz) ** 2, axis=(3, 4))
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
	"""Compute CCT matrix C_q from P_k(μ,μ) via ortho FFT pipeline."""
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
			P_R = jnp.fft.ifftn(P_k, axes=(0, 1, 2), norm='ortho')
			# P_R: (Rx, Ry, Rz, ns1, ns2, n_rmu, n_rmu)
			
			# Frobenius norm squared over spin (axes 3, 4 are spin dimensions)
			C_R = jnp.sum(jnp.abs(P_R) ** 2, axis=(3, 4))
			C_q = jnp.fft.fftn(C_R, axes=(0, 1, 2), norm='ortho')
			# C_q: (qx, qy, qz, n_rmu, n_rmu) - already in correct order!
			
			return C_q
		
		_isdf_pipeline_cache[cache_key] = _compute_CCT
	
	return _isdf_pipeline_cache[cache_key](P_k_mumu, nkx, nky, nkz)


# ============================================================================
# Left/Right CCT and ZCT (matching cohsex_jax physics)
# ============================================================================
# cohsex_jax uses separate left and right wavefunctions:
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
	
	This matches cohsex_jax physics where left and right have different band ranges.
	
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
		
		# Create sharded FFT for k-space operations (FFT over first 3 axes, (μ,ν) sharded)
		# Input/output: (nkx, nky, nkz, μ, ν) with P(None, None, None, 'x', 'y')
		sharded_spec_5d = P(None, None, None, 'x', 'y')
		sharded_ifftn_k = make_sharded_ifftn_3d(mesh_xy, sharded_spec_5d, sharded_spec_5d)
		sharded_fftn_k = make_sharded_fftn_3d(mesh_xy, sharded_spec_5d, sharded_spec_5d)
		
		@partial(jax.jit, in_shardings=(in_xy, in_xy), out_shardings=out_xy,
		         static_argnames=('nkx', 'nky', 'nkz'))
		def _compute_CCT_LR(P_l: jax.Array, P_r: jax.Array,
		                    nkx: int, nky: int, nkz: int) -> jax.Array:
			# Reshape to 3D k-grid: (nk, μ, ν) -> (nkx, nky, nkz, μ, ν)
			P_l_3d = P_l.reshape(nkx, nky, nkz, n_rmu, n_rmu)
			P_r_3d = P_r.reshape(nkx, nky, nkz, n_rmu, n_rmu)
			
			# Use sharded FFT to avoid XLA rematerialization bug
			# k-dims are replicated, so each device can FFT independently
			# Note: norm='forward' gives unscaled IFFT, scaled FFT (matches direct k-sum)
			P_l_R = sharded_ifftn_k(P_l_3d) / nk  # Apply forward normalization
			P_r_R = sharded_ifftn_k(P_r_3d) / nk
			
			# Cross-product: C_R = conj(P_l_R) * P_r_R (element-wise)
			C_R = jnp.conj(P_l_R) * P_r_R
			
			# FFT to q-space
			C_q = sharded_fftn_k(C_R) / nk  # Apply forward normalization
			return C_q
		
		_isdf_pipeline_cache[cache_key] = _compute_CCT_LR
	
	return _isdf_pipeline_cache[cache_key](P_l_k, P_r_k, nkx, nky, nkz)


def compute_ZCT_from_left_right(
	P_l_k_mumu: jax.Array,
	P_r_k_muz: jax.Array,
	kgrid: tuple[int, int, int],
	mesh_xy: Mesh,
) -> jax.Array:
	"""
	Compute ZCT from separate left (at centroids) and right (at z-chunk) pair densities.
	
	Z_q(μ,r) = FFT[ conj(IFFT(P_l(μ,μ))) ⊙ IFFT(P_r(μ,r)) ]
	
	Wait - this is wrong! For ZCT we need:
	Z_q(μ,r) = Σ_k exp(iq·k) [Σ_n,s ψ_l,n,k,s(μ)* ψ_l,n,k,s(μ')]  ???
	
	Actually looking at cohsex_jax more carefully:
	P_l = psi_l_rmuT @ psi_l_rtot  -> (n_rmu, n_rtot)
	P_r = psi_r_rmuT @ psi_r_rtot  -> (n_rmu, n_rtot)
	ZCT += conj(P_l) * P_r
	
	So ZCT uses (μ, r) for BOTH left and right.
	
	Args:
		P_l_k_mumu: Not used - kept for signature compatibility
		P_r_k_muz: Actually we need P_l(μ,r) and P_r(μ,r) separately!
	
	This function needs redesign - see compute_ZCT_from_left_right_zchunk.
	"""
	raise NotImplementedError("Use compute_ZCT_from_left_right_zchunk instead")


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
		P_l_k_muz: (nk, n_rmu, n_zchunk) left pair density at z-chunk, P(None, 'x', 'y')
		P_r_k_muz: (nk, n_rmu, n_zchunk) right pair density at z-chunk, P(None, 'x', 'y')
		kgrid: (nkx, nky, nkz)
		mesh_xy: Device mesh
	
	Returns:
		Z_q: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, 'x', 'y')
	"""
	nkx, nky, nkz = kgrid
	nk, n_rmu, n_zchunk = P_l_k_muz.shape
	
	cache_key = ('ZCT_LR', id(mesh_xy), nk, n_rmu, n_zchunk, nkx)
	
	if cache_key not in _isdf_pipeline_cache:
		in_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
		out_xy = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
		
		# Create sharded FFT for k-space operations (FFT over first 3 axes, (μ,zchunk) sharded)
		# Input/output: (nkx, nky, nkz, μ, zchunk) with P(None, None, None, 'x', 'y')
		sharded_spec_5d = P(None, None, None, 'x', 'y')
		sharded_ifftn_k = make_sharded_ifftn_3d(mesh_xy, sharded_spec_5d, sharded_spec_5d)
		sharded_fftn_k = make_sharded_fftn_3d(mesh_xy, sharded_spec_5d, sharded_spec_5d)
		
		@partial(jax.jit, in_shardings=(in_xy, in_xy), out_shardings=out_xy,
		         static_argnames=('nkx', 'nky', 'nkz'))
		def _compute_ZCT_LR(P_l: jax.Array, P_r: jax.Array,
		                    nkx: int, nky: int, nkz: int) -> jax.Array:
			# Reshape to 3D k-grid
			P_l_3d = P_l.reshape(nkx, nky, nkz, n_rmu, n_zchunk)
			P_r_3d = P_r.reshape(nkx, nky, nkz, n_rmu, n_zchunk)
			
			# Use sharded FFT to avoid XLA rematerialization bug
			# k-dims are replicated, so each device can FFT independently
			# Note: norm='forward' gives unscaled IFFT, scaled FFT (matches direct k-sum)
			P_l_R = sharded_ifftn_k(P_l_3d) / nk  # Apply forward normalization
			P_r_R = sharded_ifftn_k(P_r_3d) / nk
			
			# Cross-product
			Z_R = jnp.conj(P_l_R) * P_r_R
			
			# FFT to q-space
			Z_q = sharded_fftn_k(Z_R) / nk  # Apply forward normalization
			return Z_q
		
		_isdf_pipeline_cache[cache_key] = _compute_ZCT_LR
	
	return _isdf_pipeline_cache[cache_key](P_l_k_muz, P_r_k_muz, nkx, nky, nkz)


# ============================================================================
# ISDF Zeta Fitting: Cholesky Solve with Optimized Sharding
# ============================================================================
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
    """Solve C_q · zeta_q = Z_q via Cholesky, with column-parallel triangular solve."""
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


# ============================================================================
# 2D Blocked Cholesky Solver - memory efficient for large n_rmu
# ============================================================================

# Caches for 2D Cholesky and solve functions
_chol_2d_cache = {}
_solve_fn_cache = {}


def _get_sharded_cho_solve(mesh_xy: Mesh):
    """Get or create cached shard_map function for column-parallel triangular solve."""
    cache_key = ('cho_solve', id(mesh_xy))
    if cache_key not in _solve_fn_cache:
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None), P(None, ('x', 'y'))),
                 out_specs=P(None, ('x', 'y')))
        def _sharded_cho_solve(L: jax.Array, Z_cols: jax.Array) -> jax.Array:
            """Column-parallel solve: L L^H zeta = Z => zeta = L^{-H}(L^{-1} Z)"""
            y = jax.scipy.linalg.solve_triangular(L, Z_cols, lower=True)
            zeta = jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
            return zeta
        _solve_fn_cache[cache_key] = _sharded_cho_solve
    return _solve_fn_cache[cache_key]


def _get_solve_all_q_fn(mesh_xy: Mesh, nq: int):
    """Get or create cached fori_loop solve function for all q-points."""
    cache_key = ('solve_all_q', id(mesh_xy), nq)
    if cache_key not in _solve_fn_cache:
        L_rep_shard = NamedSharding(mesh_xy, P(None, None))
        cho_solve = _get_sharded_cho_solve(mesh_xy)
        
        @jax.jit
        def _solve_all_q(L_q_sharded: jax.Array, Z_col_sharded: jax.Array) -> jax.Array:
            """Solve zeta for all q using fori_loop, all-gathering L[q] one at a time."""
            def body(q, zeta_all):
                L_single = L_q_sharded[q]
                L_rep = jax.lax.with_sharding_constraint(L_single, L_rep_shard)
                Z_single = Z_col_sharded[q]
                zeta_q = cho_solve(L_rep, Z_single)
                return zeta_all.at[q].set(zeta_q)
            
            zeta_init = jnp.zeros_like(Z_col_sharded)
            return jax.lax.fori_loop(0, nq, body, zeta_init)
        
        _solve_fn_cache[cache_key] = _solve_all_q
    return _solve_fn_cache[cache_key]


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
    
    Args:
        C_q: (nq, n_rmu, n_rmu) CCT matrix, sharded P(None, 'x', 'y')
        Z_q: (nq, n_rmu, n_zchunk) ZCT matrix, sharded P(None, 'x', 'y')  
        mesh_xy: 2D device mesh with axes ('x', 'y')
        block_size: Tile block size for Cholesky. If None, auto-selects
    
    Returns:
        zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, None, ('x','y'))
    """
    nq, n_rmu, n_rmu2 = C_q.shape
    _, _, n_zchunk = Z_q.shape
    assert n_rmu == n_rmu2, f"C_q must be square, got {n_rmu} x {n_rmu2}"
    
    Pr = mesh_xy.shape['x']
    Pc = mesh_xy.shape['y']
    
    if block_size is None:
        block_size, J = compute_block_size_for_2d_cholesky(n_rmu, Pr, Pc)
    else:
        J = n_rmu // block_size
    
    # Get or build cached Cholesky function
    cache_key = ('chol_2d', id(mesh_xy), J, block_size)
    if cache_key not in _chol_2d_cache:
        _chol_2d_cache[cache_key] = cholesky_2d_batched(mesh_xy, J, block_size)
    chol_fn = _chol_2d_cache[cache_key]
    
    # Convert C_q to tiles and apply sharding
    C_q_tiles = dense_to_tiles(C_q, block_size)
    tiles_shard = NamedSharding(mesh_xy, P(None, 'x', 'y', None, None))
    C_q_tiles = jax.lax.with_sharding_constraint(C_q_tiles, tiles_shard)
    
    # STEP 1: 2D Blocked Cholesky
    L_q_tiles = chol_fn(C_q_tiles)
    
    # Convert to dense, keep sharded P(None, 'x', 'y')
    L_q_dense = tiles_to_dense(L_q_tiles, block_size)
    L_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    L_q_dense = jax.lax.with_sharding_constraint(L_q_dense, L_shard)
    
    # STEP 2: Reshard Z for column-parallel solve
    z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    Z_col = jax.lax.with_sharding_constraint(Z_q, z_col_shard)
    
    # STEP 3: Solve using cached function
    solve_fn = _get_solve_all_q_fn(mesh_xy, nq)
    zeta_q = solve_fn(L_q_dense, Z_col)
    
    return zeta_q


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
    
    # Cache key for solve function (includes q_chunk_size and padded size)
    cache_key = ('solve_from_L', id(mesh_xy), nq, n_rmu, n_zchunk_padded, q_chunk_size)
    
    if cache_key not in _solve_cache:
        # Reshard Z for column-parallel solve
        z_col_shard = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
        L_rep_shard = NamedSharding(mesh_xy, P(None, None))
        L_batch_rep_shard = NamedSharding(mesh_xy, P(None, None, None))  # (B_q, n_rmu, n_rmu)
        nq_c = nq
        q_chunk_c = min(q_chunk_size, nq)
        n_zchunk_c = n_zchunk_padded  # Use padded size
        
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
    
    # Pad Z if needed (zeros on RHS → zero solution for those columns, harmless)
    if needs_padding:
        pad_width = n_zchunk_padded - n_zchunk
        Z_q_padded = jnp.pad(Z_q, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        result_padded = _solve_cache[cache_key](L_q, Z_q_padded)
        # Trim back to original size
        return result_padded[:, :, :n_zchunk]
    else:
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
    x_chunk_size: int,
    output_file: str,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = True,
    use_gspace_cache: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
):
    """
    Full zeta fitting pipeline with x-chunk loop and HDF5 output.
    
    Workflow:
    1. Load wavefunctions (band-chunked FFT) for max range
    2. Slice to get left (0:b3) and right (0:b4) views
    3. Compute C_q from spin-traced P_l × P_r via ortho FFT
    4. Compute L_q = chol(C_q) using 2D blocked algorithm
    5. For each x-chunk:
       a. Compute psi_nk,a(r_chunk) via FFT
       b. Compute spin-traced P_l and P_r at x-chunk
       c. Compute Z_q via ortho FFT with left/right cross-product
       d. Solve zeta_q = L^{-H}(L^{-1} Z_q) (q-chunked)
       e. Write zeta_q chunk to HDF5
    
    Args:
        wfn: WFNReader object
        sym: SymMaps object
        meta: Meta object with system info
        centroid_indices: ISDF centroid indices
        mesh_xy: 2D device mesh
        x_chunk_size: Number of x-slices per chunk
        output_file: Path to output HDF5 file
        band_chunk_size: Bands to process at once when FFTing wavefunctions (with global r)
        q_chunk_size: Q-points to solve C_q @ zeta_q = Z_q simultaneously
        bispinor: Whether to use bispinor wavefunctions
        use_gspace_cache: If True, cache G-space across x-chunks
        band_range_left: (start, end) for left wfns. Default: (b0, b3)
        band_range_right: (start, end) for right wfns. Default: (b0, b4)
    
    Returns:
        psi_l_rmu_Y: Left centroid wfns (nk, nb_l, ns, n_rmu), Y-sharded
        psi_l_rmuT_X: Left conjugated wfns (nk, n_rmu, nb_l, ns), X-sharded
        psi_r_rmu_Y: Right centroid wfns (nk, nb_r, ns, n_rmu), Y-sharded
        psi_r_rmuT_X: Right conjugated wfns (nk, n_rmu, nb_r, ns), X-sharded
    """
    import h5py
    
    nx, ny, nz = meta.fft_grid
    n_rmu = meta.n_rmu
    n_rtot = meta.n_rtot
    nk_tot = meta.nk_tot
    kgrid = meta.kgrid
    nqx, nqy, nqz = kgrid
    nq = nqx * nqy * nqz
    
    # Number of x-chunks (x is the slowest-varying index, so x-chunks are contiguous in r)
    num_x_chunks = (nx + x_chunk_size - 1) // x_chunk_size
    n_xchunk = x_chunk_size * ny * nz  # Points per full x-chunk (contiguous in r-space!)
    
    # Band ranges for left and right wavefunctions (cohsex_jax convention)
    # Left:  (b0, b3) = all occupied + sigma conduction
    # Right: (b0, b4) = all occupied + all conduction up to nband
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
    print(f"Zeta fitting: {num_x_chunks} x-chunks, {n_xchunk} points each")
    print(f"  Left bands:  {band_range_left} ({nb_left} bands)")
    print(f"  Right bands: {band_range_right} ({nb_right} bands)")
    print(f"  Full range:  {band_range_full} ({nb_full} bands)")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
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
    
    # ========== STEP 2: Compute CCT (C_q) from left/right pair densities ==========
    # Uses spin-traced pair density matching cohsex_jax physics
    with timing.section("zeta_fit.CCT"):
        print("\nComputing spin-traced pair densities P_l and P_r...")
        P_l_k = compute_pair_density_spin_traced(psi_l_rmuT_X, psi_l_rmu_Y, mesh_xy)
        P_r_k = compute_pair_density_spin_traced(psi_r_rmuT_X, psi_r_rmu_Y, mesh_xy)
        P_l_k.block_until_ready()
        P_r_k.block_until_ready()
        # P_l_k, P_r_k: (nk, n_rmu, n_rmu)
        
        print("Computing C_q from left/right cross-product...")
        C_q = compute_CCT_from_left_right(P_l_k, P_r_k, kgrid, mesh_xy)
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
        print("\nComputing L_q = chol(C_q) using 2D blocked algorithm...")
        L_q = compute_L_q_from_CCT(C_q_flat, mesh_xy)
        L_q.block_until_ready()
        print(f"  L_q shape: {L_q.shape}")
    
    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget
    del C_q, C_q_flat
    # Force garbage collection and JAX device memory cleanup
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
                chunks=(1, 1, 1, n_rmu, n_xchunk),  # Chunk by x-slice (contiguous in r!)
            )
            # Store metadata
            f.attrs['n_rmu'] = n_rmu
            f.attrs['n_rtot'] = n_rtot
            f.attrs['fft_grid'] = meta.fft_grid
            f.attrs['kgrid'] = kgrid
            f.attrs['x_chunk_size'] = x_chunk_size
            f.attrs['num_x_chunks'] = num_x_chunks
    
    # Synchronize before writing
    jax.experimental.multihost_utils.sync_global_devices("zeta_h5_create")
    
    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each x-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]
    
    if use_gspace_cache:
        with timing.section("zeta_fit.cache_gspace"):
            print("\nCaching G-space wavefunctions for x-chunk loop...")
            # Cache FULL band range (max of left and right)
            cached_gspace = load_gspace_for_bands(
                wfn, sym, meta, mesh_xy, band_range_full, bispinor, band_chunk_size
            )
            print(f"  Cached {len(cached_gspace)} band chunks (sharded across devices)")
    else:
        cached_gspace = None
        print("\nG-space caching DISABLED (too large for memory budget)")
        print("  Will reload from HDF5 each x-chunk (slower)")
    
    # ========== STEP 6: Loop over x-chunks ==========
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
        
        X-chunking advantage: With r = x*(ny*nz) + y*nz + z, an x-chunk with
        x in [x_start, x_end) maps to CONTIGUOUS r-indices [x_start*ny*nz, x_end*ny*nz).
        This enables a single sequential HDF5 write per chunk!
        """
        try:
            while True:
                item = write_queue.get()
                if item is None:  # Poison pill signals shutdown
                    break
                zeta_data, r_start, r_end, chunk_id = item
                # zeta_data: (nq, n_rmu, actual_n_xchunk) 
                # X-chunks are CONTIGUOUS in r-space, so we can write directly!
                
                with h5py.File(output_file, 'a') as f:
                    for q_flat in range(nq):
                        qx = q_flat // (nqy * nqz)
                        qy = (q_flat % (nqy * nqz)) // nqz
                        qz = q_flat % nqz
                        # Single contiguous write per q-point!
                        f['zeta_q'][qx, qy, qz, :, r_start:r_end] = zeta_data[q_flat]
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
        for chunk_idx in range(num_x_chunks):
            x_start = chunk_idx * x_chunk_size
            x_end = min(x_start + x_chunk_size, nx)
            actual_x_size = x_end - x_start
            actual_n_xchunk = actual_x_size * ny * nz
            
            # X-chunks are CONTIGUOUS in r-space: r = x*(ny*nz) + y*nz + z
            r_start = x_start * ny * nz
            r_end = x_end * ny * nz
            
            print(f"Chunk {chunk_idx+1}/{num_x_chunks}: x=[{x_start}:{x_end}], r=[{r_start}:{r_end}]")
            
            # 6a. Get psi_nk,a(r_chunk) for FULL band range
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.load"):
                if cached_gspace is not None:
                    # Fast path: FFT only (G-space is cached)
                    psi_xchunk_full_Y = get_psi_xchunk_from_cached(
                        cached_gspace, meta, mesh_xy, band_range_full,
                        x_start, x_end, kvecs_frac,
                        band_chunk_size=band_chunk_size
                    )
                else:
                    # Slow path: reload from HDF5 each chunk
                    psi_xchunk_full_Y = get_psi_xchunk(
                        wfn, sym, meta, mesh_xy, band_range_full,
                        x_start, x_end, bispinor,
                        band_chunk_size=band_chunk_size
                    )
                psi_xchunk_full_Y.block_until_ready()
                
                # Slice for left and right (same logic as centroids)
                psi_l_xchunk_Y = psi_xchunk_full_Y[:, l_band_start:l_band_end, :, :]
                psi_r_xchunk_Y = psi_xchunk_full_Y[:, r_band_start:r_band_end, :, :]
            t_load_total += time.perf_counter() - t0
            
            # 6b. Compute spin-traced pair densities for left and right
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.pair_density"):
                # Left: P_l_k(μ, r) = Σ_{n,s} ψ*_l(μ) ψ_l(r)
                P_l_k_mux = compute_pair_density_spin_traced_xchunk(
                    psi_l_rmuT_X, psi_l_xchunk_Y, mesh_xy
                )
                # Right: P_r_k(μ, r) = Σ_{n,s} ψ*_r(μ) ψ_r(r)
                P_r_k_mux = compute_pair_density_spin_traced_xchunk(
                    psi_r_rmuT_X, psi_r_xchunk_Y, mesh_xy
                )
                P_l_k_mux.block_until_ready()
                P_r_k_mux.block_until_ready()
            t_pair_total += time.perf_counter() - t0
            
            # Free psi_xchunk arrays - we have P_k now
            del psi_xchunk_full_Y, psi_l_xchunk_Y, psi_r_xchunk_Y
            
            # 6c. Compute Z_q via left/right cross-product FFT
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.ZCT"):
                Z_q = compute_ZCT_from_left_right_xchunk(
                    P_l_k_mux, P_r_k_mux, kgrid, mesh_xy
                )
                Z_q.block_until_ready()
                
                # Free P_k arrays - we have Z_q now
                del P_l_k_mux, P_r_k_mux
                
                Z_q_flat = Z_q.reshape(nq, n_rmu, actual_n_xchunk)
                Z_q_flat = jax.lax.with_sharding_constraint(Z_q_flat, flat_shard)
                Z_q_flat.block_until_ready()
                
                # Free Z_q (3D) - we have Z_q_flat now
                del Z_q
            t_zct_total += time.perf_counter() - t0
            
            # 6d. Solve zeta_q = L^{-H}(L^{-1} Z_q)
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.solve"):
                zeta_chunk = solve_zeta_from_L_q(L_q, Z_q_flat, mesh_xy, q_chunk_size)
                zeta_chunk.block_until_ready()
                
                # Free Z_q_flat - we have zeta now
                del Z_q_flat
            t_solve_total += time.perf_counter() - t0
            
            # 6e. Copy to host and queue async write to HDF5
            # GPU→CPU copy is fast; actual HDF5 write happens in background thread
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                if jax.process_index() == 0:
                    # Copy all q-points to CPU at once (faster than one-by-one)
                    zeta_cpu = np.asarray(zeta_chunk)  # (nq, n_rmu, n_xchunk) on CPU
                    
                    # Queue the write with r-range (x-chunks are contiguous in r-space!)
                    write_queue.put((zeta_cpu, r_start, r_end, chunk_idx))
                
                # Synchronize across devices (but not waiting for HDF5 write)
                jax.experimental.multihost_utils.sync_global_devices(f"zeta_chunk_{chunk_idx}")
            t_write_total += time.perf_counter() - t0
            
    
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
    
    # Print summary
    print()  # Clear the \r line
    print(f"\nWritten to {output_file}")
    print(f"{'='*60}")
    print(f"Zeta fitting complete!")
    print(f"  Shape: ({nqx}, {nqy}, {nqz}, {n_rmu}, {n_rtot})")
    print(f"{'='*60}")
    print(f"\nTiming Summary ({num_x_chunks} x-chunks):")
    print(f"  {'Phase':<20} {'Total':>10} {'Per-chunk':>12} {'%':>6}")
    print(f"  {'-'*50}")
    print(f"  {'Load xchunk':<20} {t_load_total:>10.2f}s {t_load_total/num_x_chunks*1000:>10.1f}ms {100*t_load_total/t_chunks_total:>6.1f}%")
    print(f"  {'Pair density':<20} {t_pair_total:>10.2f}s {t_pair_total/num_x_chunks*1000:>10.1f}ms {100*t_pair_total/t_chunks_total:>6.1f}%")
    print(f"  {'ZCT (FFT pipeline)':<20} {t_zct_total:>10.2f}s {t_zct_total/num_x_chunks*1000:>10.1f}ms {100*t_zct_total/t_chunks_total:>6.1f}%")
    print(f"  {'Solve (L^-1 Z)':<20} {t_solve_total:>10.2f}s {t_solve_total/num_x_chunks*1000:>10.1f}ms {100*t_solve_total/t_chunks_total:>6.1f}%")
    print(f"  {'H5 write':<20} {t_write_total:>10.2f}s {t_write_total/num_x_chunks*1000:>10.1f}ms {100*t_write_total/t_chunks_total:>6.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'Chunk loop total':<20} {t_chunks_total:>10.2f}s {t_chunks_total/num_x_chunks*1000:>10.1f}ms")
    print(f"  {'Per r-point':<20} {'':<10} {t_chunks_total/n_rtot*1e6:>10.1f}μs")
    
    # Return left and right centroid wavefunctions (persist for downstream use)
    # Note: full arrays were already deleted after slicing in STEP 1
    return psi_l_rmu_Y, psi_l_rmuT_X, psi_r_rmu_Y, psi_r_rmuT_X


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
# X-CHUNK EXTRACTION: Contiguous r-space chunking via x-axis slicing
# ============================================================================
# X-chunking advantage: With r = x*(ny*nz) + y*nz + z, an x-chunk with
# x in [x_start, x_end) maps to CONTIGUOUS r-indices [x_start*ny*nz, x_end*ny*nz).
# This enables single sequential HDF5 writes instead of strided writes.
# ============================================================================

# Cache for xchunk extraction function
_xchunk_slice_cache = {}


def get_sharded_wfns_xchunk_slice(
    global_psi_Gtot: jax.Array,
    meta: Meta,
    x_start: int,
    x_end: int,
    kvecs_frac: np.ndarray,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
) -> jax.Array:
    """
    FFT wavefunctions and extract x-chunk via slicing.
    
    X-chunking gives CONTIGUOUS r-indices: slicing x in [x_start, x_end)
    produces r-indices [x_start*ny*nz, x_end*ny*nz) which can be written
    to HDF5 in a single sequential operation.
    
    Args:
        global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
        meta: Meta object
        x_start, x_end: X-slice range [x_start, x_end)
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
    
    Returns:
        psi_xchunk_Y: (nk, nb, ns, n_xchunk) with P(None, None, None, 'y')
    """
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    fft_grid = meta.fft_grid
    nx, ny, nz = fft_grid
    x_chunk_size = x_end - x_start
    n_xchunk = x_chunk_size * ny * nz  # CONTIGUOUS in r-space!
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rtot = nx * ny * nz
    
    # Cache key - use hash of kvecs since it's constant for a given system
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = ('xchunk_slice', id(mesh_xy), nk_tot, nspinor, x_chunk_size, nx, ny, nz, kvecs_hash)
    
    if cache_key not in _xchunk_slice_cache:
        out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
        
        sharded_ifftn = make_sharded_ifftn_3d(
            mesh_xy, 
            P(None, ('x', 'y'), None, None, None, None),
            P(None, ('x', 'y'), None, None, None, None)
        )
        
        # Intermediate sharding for staged reshard
        stage1_shard = NamedSharding(mesh_xy, P(None, 'y', None, None))
        
        # Pre-compute phase grids and kvecs ONCE in closure
        fx_cached = jnp.arange(nx, dtype=jnp.float64)[None, :, None, None] / nx
        fy_cached = jnp.arange(ny, dtype=jnp.float64)[None, None, :, None] / ny
        fz_cached = jnp.arange(nz, dtype=jnp.float64)[None, None, None, :] / nz
        kvecs_cached = jnp.asarray(kvecs_frac)
        n_rtot_cached = n_rtot
        
        # x_chunk_size is static (from cache key), x_start is dynamic
        x_chunk_size_static = x_chunk_size
        
        @partial(jax.jit, static_argnames=('nb_static',))
        def _extract_xchunk_slice(psi_G, x_start_dyn, nb_static):
            # FFT to real space: (nk, nb_padded, ns, nx, ny, nz)
            psi_r = sharded_ifftn(psi_G)
            
            # Apply Bloch phase exp(ik·r)
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
            
            # Slice x-axis with DYNAMIC start, STATIC size
            # psi_r: (nk, nb, ns, nx, ny, nz) -> (nk, nb, ns, x_chunk, ny, nz)
            psi_xslice = jax.lax.dynamic_slice(
                psi_r,
                (0, 0, 0, x_start_dyn, 0, 0),
                (nk_tot, nb_static, nspinor, x_chunk_size_static, ny, nz)
            )
            
            # Flatten spatial dims: (nk, nb, ns, x_chunk*ny*nz)
            # This is CONTIGUOUS in r-space since r = x*ny*nz + y*nz + z
            psi_xchunk = psi_xslice.reshape(nk_tot, nb_static, nspinor, -1)
            
            # Staged reshard
            psi_xchunk = jax.lax.with_sharding_constraint(psi_xchunk, stage1_shard)
            psi_xchunk = jax.lax.with_sharding_constraint(psi_xchunk, out_Y)
            
            return psi_xchunk
        
        _xchunk_slice_cache[cache_key] = _extract_xchunk_slice
    
    return _xchunk_slice_cache[cache_key](global_psi_Gtot, x_start, nb)


def get_psi_xchunk_from_cached(
    cached_gspace: list[tuple[jax.Array, tuple[int, int]]],
    meta, mesh_xy, band_range, x_start, x_end, kvecs_frac,
    band_chunk_size: int = 16,
) -> jax.Array:
    """
    Extract x-chunk from pre-loaded G-space (FFT only, no HDF5 read).
    
    This is the fast path that reuses cached G-space across x-chunk iterations.
    X-chunks are contiguous in r-space, enabling efficient HDF5 writes.
    
    Args:
        cached_gspace: Pre-loaded G-space from load_gspace_for_bands()
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end) - total bands needed
        x_start, x_end: X-slice range
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        band_chunk_size: Bands to FFT at once
    
    Returns:
        psi_xchunk_Y: (nk, nb, ns, n_xchunk) with P(None, None, None, 'y')
    """
    nx, ny, nz = meta.fft_grid
    n_xchunk = (x_end - x_start) * ny * nz
    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    
    # Output sharding
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    
    # Allocate output array
    psi_xchunk_all = jnp.zeros((nk_tot, nb_total, nspinor, n_xchunk), dtype=jnp.complex128)
    psi_xchunk_all = jax.lax.with_sharding_constraint(psi_xchunk_all, out_Y)
    
    # Process each cached band chunk
    for bc_idx, (global_psi_Gtot, bc_range) in enumerate(cached_gspace):
        nb_chunk = bc_range[1] - bc_range[0]
        
        # FFT and extract x-slice
        psi_xchunk_chunk = get_sharded_wfns_xchunk_slice(
            global_psi_Gtot, meta, x_start, x_end, kvecs_frac, mesh_xy, bc_range
        )
        
        # Place into output array
        local_bc_start = bc_idx * band_chunk_size
        local_bc_end = local_bc_start + nb_chunk
        psi_xchunk_all = psi_xchunk_all.at[:, local_bc_start:local_bc_end, :, :].set(psi_xchunk_chunk)
        
        del psi_xchunk_chunk
    
    return psi_xchunk_all


def get_psi_xchunk(
    wfn, sym, meta, mesh_xy, band_range, x_start, x_end, bispinor,
    band_chunk_size: int = 16,
) -> jax.Array:
    """
    Load and FFT wavefunctions for a specific x-chunk.
    
    NOTE: This function reloads G-space from HDF5 each call. For multiple
    x-chunks, use load_gspace_for_bands() + get_psi_xchunk_from_cached()
    to avoid redundant HDF5 reads.
    
    Args:
        wfn: WFNReader
        sym: SymMaps
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end) - total bands needed
        x_start, x_end: X-slice range
        bispinor: Whether to use bispinor
        band_chunk_size: Bands to FFT at once
    
    Returns:
        psi_xchunk_Y: (nk, nb, ns, n_xchunk) with P(None, None, None, 'y')
    """
    kgrid = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid[None, :]
    
    cached_gspace = load_gspace_for_bands(
        wfn, sym, meta, mesh_xy, band_range, bispinor, band_chunk_size
    )
    
    result = get_psi_xchunk_from_cached(
        cached_gspace, meta, mesh_xy, band_range, x_start, x_end, kvecs_frac,
        band_chunk_size
    )
    
    del cached_gspace
    return result


# Aliases for x-chunk pair density and ZCT functions
# These are generic and work with any chunk shape (z-chunk or x-chunk)
compute_pair_density_spin_traced_xchunk = compute_pair_density_spin_traced_zchunk
compute_ZCT_from_left_right_xchunk = compute_ZCT_from_left_right_zchunk


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
        # out_X: transposed shape is (nk, n_rmu, nb, ns) for pair density einsum
        out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))
        null_4 = NamedSharding(mesh_xy, P(None, None, None, None))
        
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
            
            # Gather centroids on axis 3 (spatial) - this axis is replicated so gather is LOCAL
            # Bands stay sharded on axis 1 throughout, no all-gather of the large array
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
    # psi_rmu: (nk, nb, ns, n_rmu), psi_rmuT: (nk, n_rmu, nb, ns) for pair density
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
        # psi_rmu_chunk: (nk, nb_chunk, ns, n_rmu)
        # psi_rmuT_chunk: (nk, n_rmu, nb_chunk, ns)
        local_bc_start = bc_idx * band_chunk_size
        local_bc_end = local_bc_start + nb_chunk
        psi_rmu_all = psi_rmu_all.at[:, local_bc_start:local_bc_end, :, :].set(psi_rmu_chunk)
        psi_rmuT_all = psi_rmuT_all.at[:, :, local_bc_start:local_bc_end, :].set(psi_rmuT_chunk)
        
        # Free memory
        del global_psi_Gtot, psi_rmu_chunk, psi_rmuT_chunk
    
    return psi_rmu_all, psi_rmuT_all



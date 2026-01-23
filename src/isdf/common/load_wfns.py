import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from . import Meta
from . import timing


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

	# Load G-coefficients for all k-points using batched reads
	with timing.section("load_wfns.k_loop"):
		for k_idx in range(sym.nk_tot):
			k_red = sym.irk_to_k_map[k_idx]
			gvecs_k_rot = np.asarray(sym.get_gvecs_kfull(wfn, k_idx))
			ngk = int(wfn.ngk[k_red])
			psi_Gspace_local = np.zeros(
				(n_local_shards * bands_per_shard, meta.nspinor, ngk),
				dtype=np.complex128,
			)
			# Batch read and rotate all owned bands at once
			if n_owned > 0:
				cnk_batch = sym.get_cnk_fullzone_batch(wfn, owned_band_indices, k_idx)
				psi_Gspace_local[local_band_indices, 0:meta.nspinor_wfnfile, :] = cnk_batch
			# Expand to 4 components if requested: small component ≈ (α/2)(σ·p)ψ_large
			if bispinor:
				psi_Gspace_local[:, 2:4, :] = np.asarray(get_small_psi_component(
					jnp.asarray(gvecs_k_rot),
					jnp.asarray(sym.unfolded_kpts[k_idx], dtype=jnp.float64),
					jnp.asarray(wfn.bvec, dtype=jnp.float64),
					jnp.asarray(psi_Gspace_local),
				))
			# Scatter G-space coefficients into the FFT box for all local bands
			psi_Gtot_local[k_idx, :, :, gvecs_k_rot[:, 0], gvecs_k_rot[:, 1], gvecs_k_rot[:, 2]] = np.transpose(
				psi_Gspace_local, (2, 0, 1)
			)

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
				 out_shardings=(y3_4, y3_4, null_4))
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

			# Replicate before indexed gather to avoid multi-shard gather issues
			psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, null_4)

			# Centroid gather using pre-computed linear indices
			psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)

			# Conjugate-transpose to (nk, n_rmu, nb, nspinor)
			psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2).reshape(nk_tot, n_rmu, -1, nspinor))
			
			# Re-shard results to y-only
			psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, y3_4)
			psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, y3_4)

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

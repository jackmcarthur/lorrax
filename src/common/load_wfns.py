import gc
import math
import os
import queue
import threading
import time
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

from . import Meta
from . import timing
from common import jax_profile
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

def make_sharded_ifftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    Uses shard_map to run FFT independently on each device's local data.
    The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
    Args:
        mesh: The device mesh
        in_spec: PartitionSpec for input (e.g., P(None, ('x','y'), None, None, None, None))
        out_spec: PartitionSpec for output (same as in_spec for FFT)
    
    Returns:
        A function that performs 3D IFFT on sharded data
    """
    def _local_ifftn(x_local):
        # Each device runs FFT on its local shard independently
        return jnp.fft.ifftn(x_local, axes=axes, norm=norm)
    
    return shard_map(_local_ifftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)

def make_sharded_fftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    shard_map local FFT (forward).
    
    This is the forward-FFT counterpart to make_sharded_ifftn_3d.
    """
    def _local_fftn(x_local):
        return jnp.fft.fftn(x_local, axes=axes, norm=norm)
    
    return shard_map(_local_fftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


def load_kpoint_fftbox(wfn, sym, meta, k_idx, nb):
    """Load a single k-point's wavefunction into the FFT box on GPU.

    Returns jax array of shape (nb, nspinor, nx, ny, nz), ~0.55 GiB for 12x12.
    """
    nx, ny, nz = meta.fft_grid
    band_indices = np.arange(nb)
    gvecs_k = np.asarray(sym.get_gvecs_kfull(wfn, k_idx))
    cnk = sym.get_cnk_fullzone_batch(wfn, band_indices, k_idx)
    psi = np.zeros((nb, meta.nspinor, nx, ny, nz), dtype=np.complex128)
    psi[:, :meta.nspinor_wfnfile, gvecs_k[:, 0], gvecs_k[:, 1], gvecs_k[:, 2]] = cnk
    return jnp.asarray(psi)


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
	gvecsk_cart = jnp.matmul(gvecs + kvec, bvec)

	sigmadotp = jnp.array([
		[gvecsk_cart[:, 2], gvecsk_cart[:, 0] - 1j * gvecsk_cart[:, 1]],
		[gvecsk_cart[:, 0] + 1j * gvecsk_cart[:, 1], -gvecsk_cart[:, 2]],
	], dtype=jnp.complex128)

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
		
		# Pre-compute centroid linear indices (int64 for XLA compatibility)
		centroids = jnp.asarray(centroid_indices, dtype=jnp.int64)
		ny, nz = fft_grid[1], fft_grid[2]
		centroid_lin = (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(jnp.int64)
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
			
			# Step 1: Reshard from (b_XY, ns, n_rtot) to (b_X, ns, n_rtot) BEFORE gather
			# This simplifies XY sharding to just X on bands, preparing for gather
			psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, x1_4)

			# Centroid gather on axis 3 (spatial) - now with simpler X-sharding on bands
			psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)

			# Conjugate-transpose to (nk, n_rmu, nb, nspinor) for pair density
			psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))  # (nk, n_rmu, nb, ns)
			
			# Step 2: Apply final output shardings
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
            # Cache FULL band range (max of left and right)
            cached_gspace = load_gspace_for_bands(
                wfn, sym, meta, mesh_xy, band_range_full, bispinor, band_chunk_size
            )
            print(f"  Cached {len(cached_gspace)} band chunks (sharded across devices)")
    else:
        cached_gspace = None
        print("\nG-space caching DISABLED (too large for memory budget)")
        print("  Will reload from HDF5 each r-chunk (slower)")
    
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
            
            # 6a. Get psi_nk,a(r_chunk) for FULL band range
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.load"):
                if cached_gspace is not None:
                    # Fast path: FFT only (G-space is cached)
                    psi_chunk_Y = get_psi_rchunk_from_cached(
                        cached_gspace, meta, mesh_xy, band_range_full,
                        r_start, r_end, kvecs_frac,
                        band_chunk_size=band_chunk_size
                    )
                else:
                    # Slow path: reload from HDF5 each chunk
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
                del Z_q  # free 3D ref; reshape may share buffer
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
            del Z_q_flat
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.solve"), jax_profile.step_annotation("chunk_solve", step_num=chunk_idx):
                zeta_chunk = solve_zeta_from_L_q(L_q, Z_col, mesh_xy, q_chunk_size)
                # No block_until_ready — gather will consume zeta_chunk asynchronously.
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
                del zeta_chunk  # Free GPU memory before next chunk's FFT stage
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



# ============================================================================
# R-CHUNK EXTRACTION: Contiguous r-space chunking via flattened r-index
# ============================================================================
# R-chunking advantage: r in [r_start, r_end) is contiguous in r-space and can
# be written to HDF5 in a single sequential operation. This allows arbitrary
# chunk sizes by slicing along the flattened xyz index.
# ============================================================================

# Cache for rchunk extraction function
_rchunk_slice_cache = {}


def get_sharded_wfns_rchunk_slice(
    global_psi_Gtot: jax.Array,
    meta: Meta,
    r_start: int,
    r_end: int,
    kvecs_frac: np.ndarray,
    mesh_xy: Mesh,
    band_range: tuple[int, int],
) -> jax.Array:
    """
    FFT wavefunctions and extract r-chunk via flattened r-index slicing.
    
    R-chunking gives CONTIGUOUS r-indices: slicing r in [r_start, r_end)
    produces a contiguous block in the flattened xyz order and can be written
    to HDF5 in a single sequential operation.
    
    Args:
        global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
        meta: Meta object
        r_start, r_end: R-index range [r_start, r_end)
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        mesh_xy: Device mesh
        band_range: (b_start, b_end)
    
    Returns:
        psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')
    """
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    fft_grid = meta.fft_grid
    nx, ny, nz = fft_grid
    r_chunk_size = r_end - r_start
    b_start, b_end = band_range
    nb = b_end - b_start
    n_rtot = nx * ny * nz
    
    # Cache key - use hash of kvecs since it's constant for a given system
    kvecs_hash = hash(kvecs_frac.tobytes())
    cache_key = ('rchunk_slice', id(mesh_xy), nk_tot, nspinor, r_chunk_size, nx, ny, nz, kvecs_hash)
    
    if cache_key not in _rchunk_slice_cache:
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
        
        # r_chunk_size is static (from cache key), r_start is dynamic
        r_chunk_size_static = r_chunk_size
        
        band_shard = P(None, ('x', 'y'), None, None)

        @partial(jax.jit, static_argnames=('nb_static',))
        def _extract_rchunk_slice(psi_G, r_start_dyn, nb_static):
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

            # NOTE: Do NOT trim bands here. psi_r has padded band count
            # (divisible by mesh size) which shard_map requires. Trim after reshard.
            nb_padded = psi_r.shape[1]

            # Flatten spatial dims: (nk, nb_padded, ns, nx*ny*nz)
            psi_flat = psi_r.reshape(nk_tot, nb_padded, nspinor, n_rtot_cached)

            # Local r-slice via shard_map: each device slices its own band
            # shard WITHOUT triggering an all-gather of all 80 bands.
            def _local_rslice(psi_local, r_start_arr):
                return jax.lax.dynamic_slice_in_dim(
                    psi_local, r_start_arr[0], r_chunk_size_static, axis=3
                )

            psi_rchunk = shard_map(
                _local_rslice,
                mesh=mesh_xy,
                in_specs=(band_shard, P()),
                out_specs=band_shard,
            )(psi_flat, jnp.array([r_start_dyn]))
            # psi_rchunk: (nk, nb_padded_local, ns, r_chunk_size) per device — small!

            # Staged reshard: now all-gather bands on the SMALL r-chunk array
            psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, stage1_shard)
            psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, out_Y)

            # NOW trim to actual band count (after reshard, bands are replicated)
            psi_rchunk = psi_rchunk[:, :nb_static, :, :]

            return psi_rchunk
        
        _rchunk_slice_cache[cache_key] = _extract_rchunk_slice
    
    return _rchunk_slice_cache[cache_key](global_psi_Gtot, r_start, nb)


def get_psi_rchunk_from_cached(
    cached_gspace: list[tuple[jax.Array, tuple[int, int]]],
    meta, mesh_xy, band_range, r_start, r_end, kvecs_frac,
    band_chunk_size: int = 16,
) -> jax.Array:
    """
    Extract r-chunk from pre-loaded G-space (FFT only, no HDF5 read).
    
    This is the fast path that reuses cached G-space across r-chunk iterations.
    R-chunks are contiguous in r-space, enabling efficient HDF5 writes.
    
    Args:
        cached_gspace: Pre-loaded G-space from load_gspace_for_bands()
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end) - total bands needed
        r_start, r_end: R-index range
        kvecs_frac: (nk, 3) k-vectors in fractional coordinates
        band_chunk_size: Bands to FFT at once
    
    Returns:
        psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')
    """
    n_rchunk = r_end - r_start
    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    
    # Output sharding
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    
    # Allocate output array for all bands (r-chunk is small enough)
    psi_rchunk_all = jnp.zeros((nk_tot, nb_total, nspinor, n_rchunk), dtype=jnp.complex128)
    psi_rchunk_all = jax.lax.with_sharding_constraint(psi_rchunk_all, out_Y)
    
    # Process each cached band chunk - FFT only (no HDF5 read)
    for bc_idx, (global_psi_Gtot, bc_range) in enumerate(cached_gspace):
        nb_chunk = bc_range[1] - bc_range[0]
        
        # FFT and extract r-slice for this chunk
        psi_rchunk_chunk = get_sharded_wfns_rchunk_slice(
            global_psi_Gtot, meta, r_start, r_end, kvecs_frac, mesh_xy, bc_range
        )
        
        # Place into output array at correct band indices
        local_bc_start = bc_idx * band_chunk_size
        local_bc_end = local_bc_start + nb_chunk
        psi_rchunk_all = psi_rchunk_all.at[:, local_bc_start:local_bc_end, :, :].set(psi_rchunk_chunk)
        
        # Free FFT output only (keep G-space cached)
        del psi_rchunk_chunk
    
    return psi_rchunk_all


def get_psi_rchunk(
    wfn, sym, meta, mesh_xy, band_range, r_start, r_end, bispinor,
    band_chunk_size: int = 16,
    k_chunk_size: int = 0,
) -> jax.Array:
    """
    Load and FFT wavefunctions for a specific r-chunk.
    
    NOTE: This function reloads G-space from HDF5 each call. For multiple
    r-chunks, use load_gspace_for_bands() + get_psi_rchunk_from_cached()
    to avoid redundant HDF5 reads.
    
    Uses band chunking to limit memory during FFT step:
    - Loop over band chunks
    - FFT each chunk to real-space (the memory bottleneck)
    - Extract r-slice and accumulate into output array
    
    The final psi_rchunk has all bands but only the r-slice, which is
    small enough to hold in memory for downstream pair density computation.
    
    Args:
        wfn: WFNReader
        sym: SymMaps
        meta: Meta object
        mesh_xy: Device mesh
        band_range: (b_start, b_end) - total bands needed
        r_start, r_end: R-index range
        bispinor: Whether to use bispinor
        band_chunk_size: Bands to FFT at once (memory control for FFT step)
    
    Returns:
        psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')
    """
    kgrid = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid[None, :]
    b_start, b_end = band_range
    nb_total = b_end - b_start
    nk_tot = meta.nk_tot
    nspinor = meta.nspinor
    n_rchunk = r_end - r_start
    
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    
    psi_rchunk_all = jnp.zeros((nk_tot, nb_total, nspinor, n_rchunk), dtype=jnp.complex128)
    psi_rchunk_all = jax.lax.with_sharding_constraint(psi_rchunk_all, out_Y)
    
    # Determine effective k-chunk size
    nk_batch = nk_tot if k_chunk_size <= 0 else min(k_chunk_size, nk_tot)

    # Process bands in chunks (outer), k-points in batches (inner)
    num_band_chunks = (nb_total + band_chunk_size - 1) // band_chunk_size
    for bc_idx in range(num_band_chunks):
        bc_start = b_start + bc_idx * band_chunk_size
        bc_end = min(bc_start + band_chunk_size, b_end)
        bc_range = (bc_start, bc_end)

        # Load G-space for this band chunk (all k-points)
        global_psi_Gtot, _ = read_Gvecs_to_devices(wfn, sym, bc_range, meta, bispinor, mesh_xy)

        if nk_batch >= nk_tot:
            # No k-batching: FFT all k-points at once
            psi_rchunk_chunk = get_sharded_wfns_rchunk_slice(
                global_psi_Gtot, meta, r_start, r_end, kvecs_frac, mesh_xy, bc_range
            )
            local_bc_start = bc_idx * band_chunk_size
            local_bc_end = local_bc_start + (bc_end - bc_start)
            psi_rchunk_all = psi_rchunk_all.at[:, local_bc_start:local_bc_end, :, :].set(psi_rchunk_chunk)
            del psi_rchunk_chunk
        else:
            # K-batched path: FFT subsets of k-points to limit memory
            local_bc_start = bc_idx * band_chunk_size
            nb_chunk = bc_end - bc_start
            for k0 in range(0, nk_tot, nk_batch):
                k1 = min(k0 + nk_batch, nk_tot)
                sub_meta = SimpleNamespace(
                    nk_tot=k1 - k0, nspinor=nspinor, fft_grid=meta.fft_grid,
                    n_rtot=meta.n_rtot,
                )
                psi_k_chunk = get_sharded_wfns_rchunk_slice(
                    global_psi_Gtot[k0:k1], sub_meta, r_start, r_end,
                    kvecs_frac[k0:k1], mesh_xy, bc_range
                )
                psi_rchunk_all = psi_rchunk_all.at[k0:k1, local_bc_start:local_bc_start + nb_chunk, :, :].set(psi_k_chunk)
                del psi_k_chunk

        del global_psi_Gtot

    return psi_rchunk_all


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
    
    This is the centroid-extraction counterpart to get_sharded_wfns_rchunk_slice.
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
        
        # Pre-compute centroid linear indices (int64 for XLA compatibility)
        centroids = jnp.asarray(centroid_indices, dtype=jnp.int64)
        centroid_lin = (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(jnp.int64)
        
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
            # Keep bands sharded on (x,y); avoid resharding a huge (nk,nb,ns,n_rtot) tensor.
            intermediate_XY = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None))
            psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, intermediate_XY)
            
            # Centroid gather on axis 3 (spatial) while bands remain sharded on (x,y).
            psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)
            
            # Create psi_rmuT (conjugate transpose for left wfn in pair density)
            psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2))  # (nk, n_rmu, nb, ns)
            
            # Step 2: Apply final output shardings
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

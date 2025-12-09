import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from . import Meta


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

	# Load G-coefficients for all k-points at once into local buffer (still in G-space)
	for k_idx in range(sym.nk_tot):
		k_red = sym.irk_to_k_map[k_idx]
		gvecs_k_rot = np.asarray(sym.get_gvecs_kfull(wfn, k_idx))
		psi_Gspace_local = np.zeros(
			(n_local_shards * bands_per_shard, meta.nspinor, int(wfn.ngk[k_red])),
			dtype=np.complex128,
		)
		# Populate 2-component coefficients from file
		for j in range(nb):
			placement = place_band_into_local(j)
			if placement is None:
				continue
			local_slot, offset = placement
			local_band = local_slot * bands_per_shard + offset
			band_idx = bandrange[0] + j
			cnk = np.asarray(sym.get_cnk_fullzone(wfn, band_idx, k_idx))
			psi_Gspace_local[local_band, 0:meta.nspinor_wfnfile, :] = cnk
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
	global_shape = (meta.nk_tot, total_bands_padded, meta.nspinor, *meta.fft_grid)
	band_sharding = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None, None, None))
	global_psi_Gtot = jax.make_array_from_process_local_data(
		band_sharding, jnp.asarray(psi_Gtot_local), global_shape
	)

	return global_psi_Gtot, nb


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
	"""
	xy2_6 = NamedSharding(mesh_xy, P(None, ('x', 'y'), None, None, None, None))
	xy3_4 = NamedSharding(mesh_xy, P(None, None, None, ('x', 'y')))
	y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
	x1_4 = NamedSharding(mesh_xy, P(None, 'x', None, None))
	null_4 = NamedSharding(mesh_xy, P(None, None, None, None))

	@partial(jax.jit,
			 static_argnames=("nb_actual", "is_left"), 
			 in_shardings=(xy2_6), 
			 out_shardings=(y3_4, y3_4, null_4))
	def _finalize(global_psi_Gtot: jax.Array, nb_actual: int, is_left: bool):
		# FFT to real space
		psi_r = jnp.fft.ifftn(global_psi_Gtot, axes=(-3, -2, -1))

		# Vectorized exp(ikr)
		fx = jnp.arange(meta.fft_grid[0], dtype=jnp.float64)[None, :, None, None] / meta.fft_grid[0]
		fy = jnp.arange(meta.fft_grid[1], dtype=jnp.float64)[None, None, :, None] / meta.fft_grid[1]
		fz = jnp.arange(meta.fft_grid[2], dtype=jnp.float64)[None, None, None, :] / meta.fft_grid[2]
		kpts = jnp.asarray(sym.unfolded_kpts, dtype=jnp.float64)[: psi_r.shape[0]]
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
		psi_r = psi_r * jnp.sqrt(meta.n_rtot)

		# Trim bands to actual request
		psi_r = psi_r[:, :nb_actual]

		# Flatten spatial dims to rtot
		psi_rtot = psi_r.reshape(meta.nk_tot, nb_actual, meta.nspinor, -1)

		# Replicate before indexed gather to avoid multi-shard gather issues
		psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, null_4)

		# Centroid gather along r on replicated array
		centroids = jnp.asarray(centroid_indices, dtype=jnp.int32)
		ny = jnp.asarray(meta.fft_grid[1], dtype=jnp.int32)
		nz = jnp.asarray(meta.fft_grid[2], dtype=jnp.int32)
		centroid_lin = (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(jnp.int32)
		psi_rmu = jnp.take(psi_rtot, centroid_lin, axis=3)

		# Conjugate-transpose to (nk, n_rmu, nb*nspinor) and X-only reshard over rmu
		n_rmu = psi_rmu.shape[-1]
		psi_rmuT = jnp.conj(psi_rmu.transpose(0, 3, 1, 2).reshape(meta.nk_tot, n_rmu, -1, meta.nspinor))
		
		# Re-shard results to y-only to match out_shardings
		psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, y3_4)
		psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, y3_4)
		#psi_rmuT_X = jax.lax.with_sharding_constraint(psi_rmuT, null_4)

		return psi_rtot, psi_rmu, psi_rmuT

	return _finalize(global_psi_Gtot, nb_actual, bool(is_left))

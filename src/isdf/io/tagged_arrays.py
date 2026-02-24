"""Tagged arrays I/O for ISDF restart files.

This module handles reading/writing V_qmunu, psi arrays, and other ISDF 
data structures to HDF5 for restart capability.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import h5py
from jax.sharding import NamedSharding, PartitionSpec as P


def write_labeled_arrays_to_h5(
	filename,
	V_qmunu,
	psi_l,
	psi_r,
	enk_l=None,
	enk_r=None,
	S_qmunu=None,
	V0_noG0_munu=None,
	G0_mu_nu=None,
	W0_qmunu=None,
	init_W0: bool = False,
	band_edges=None,
):
	"""
	Write raw JAX/Numpy arrays to an HDF5 file for restart.
	Only rank 0 performs the write; arrays are gathered to host first.
	"""
	from jax.experimental import multihost_utils as _mh
	# Gather sharded arrays to process-local host copies
	def _to_host(a):
		if isinstance(a, (jax.Array,)):
			try:
				return _mh.process_allgather(a, tiled=True)
			except Exception:
				return jax.device_get(a)
		return a
	V_qmunu_h = _to_host(V_qmunu)
	psi_l_h = _to_host(psi_l)
	psi_r_h = _to_host(psi_r)
	S_qmunu_h = _to_host(S_qmunu) if S_qmunu is not None else None
	V0_noG0_h = _to_host(V0_noG0_munu) if V0_noG0_munu is not None else None
	G0_mu_nu_h = _to_host(G0_mu_nu) if G0_mu_nu is not None else None
	W0_qmunu_h = _to_host(W0_qmunu) if W0_qmunu is not None else None
	enk_l_h = getattr(enk_l, 'data', enk_l)
	enk_r_h = getattr(enk_r, 'data', enk_r)
	if enk_l_h is not None:
		enk_l_h = _to_host(enk_l_h)
	if enk_r_h is not None:
		enk_r_h = _to_host(enk_r_h)
	# Only rank 0 writes to disk
	if jax.process_index() == 0:
		with h5py.File(filename, "w") as f:
			f.create_dataset("V_qmunu", data=np.asarray(V_qmunu_h))
			if S_qmunu_h is not None:
				f.create_dataset("S_qmunu", data=np.asarray(S_qmunu_h))
			if V0_noG0_h is not None:
				f.create_dataset("V0_noG0_munu", data=np.asarray(V0_noG0_h))
			if G0_mu_nu_h is not None:
				f.create_dataset("G0_mu_nu", data=np.asarray(G0_mu_nu_h))
			if W0_qmunu_h is not None:
				dset = f.create_dataset("W0_qmunu", data=np.asarray(W0_qmunu_h))
				dset.attrs["W0_ready"] = True
			elif init_W0:
				v_shape = np.asarray(V_qmunu_h).shape
				v_dtype = np.asarray(V_qmunu_h).dtype
				fill = np.zeros((), dtype=v_dtype)
				dset = f.create_dataset("W0_qmunu", shape=v_shape, dtype=v_dtype, fillvalue=fill)
				dset.attrs["W0_ready"] = False
			f.create_dataset("psi_l", data=np.asarray(psi_l_h))
			f.create_dataset("psi_r", data=np.asarray(psi_r_h))
			if enk_l_h is not None:
				f.create_dataset("enk_l", data=np.asarray(enk_l_h))
			if enk_r_h is not None:
				f.create_dataset("enk_r", data=np.asarray(enk_r_h))
			if band_edges is not None:
				f.create_dataset("band_edges", data=np.asarray(band_edges, dtype=np.int64))


def write_w0_qmunu_to_h5(filename, W0_qmunu):
	"""Overwrite or append the W0_qmunu dataset in an existing restart file."""
	from jax.experimental import multihost_utils as _mh

	def _to_host(a):
		if isinstance(a, (jax.Array,)):
			try:
				return _mh.process_allgather(a, tiled=True)
			except Exception:
				return jax.device_get(a)
		return a

	W0_qmunu_h = _to_host(W0_qmunu)
	if jax.process_index() == 0:
		with h5py.File(filename, "a") as f:
			if "W0_qmunu" in f:
				del f["W0_qmunu"]
			dset = f.create_dataset("W0_qmunu", data=np.asarray(W0_qmunu_h))
			dset.attrs["W0_ready"] = True


def write_vw_nohead_to_h5(filename, V_qmunu_nohead=None, W0_qmunu_nohead=None):
	"""Append headless V_qmunu/W0_qmunu datasets to an existing restart file."""
	from jax.experimental import multihost_utils as _mh

	def _to_host(a):
		if isinstance(a, (jax.Array,)):
			try:
				return _mh.process_allgather(a, tiled=True)
			except Exception:
				return jax.device_get(a)
		return a

	V_nohead_h = _to_host(V_qmunu_nohead) if V_qmunu_nohead is not None else None
	W0_nohead_h = _to_host(W0_qmunu_nohead) if W0_qmunu_nohead is not None else None

	if jax.process_index() == 0:
		with h5py.File(filename, "a") as f:
			if V_nohead_h is not None:
				if "V_qmunu_nohead" in f:
					del f["V_qmunu_nohead"]
				f.create_dataset("V_qmunu_nohead", data=np.asarray(V_nohead_h))
			if W0_nohead_h is not None:
				if "W0_qmunu_nohead" in f:
					del f["W0_qmunu_nohead"]
				dset = f.create_dataset("W0_qmunu_nohead", data=np.asarray(W0_nohead_h))
				dset.attrs["W0_ready"] = True


def read_labeled_arrays_from_h5(filename):
	"""
	Read raw arrays from an HDF5 restart file and return JAX arrays:
	(V_qmunu, S_qmunu, psi_l, psi_r, enk_l, enk_r)
	"""
	with h5py.File(filename, "r") as f:
		V_qmunu = jnp.asarray(f["V_qmunu"][:])
		S_qmunu = jnp.asarray(f["S_qmunu"][:]) if "S_qmunu" in f else None
		V0_noG0_munu = jnp.asarray(f["V0_noG0_munu"][:]) if "V0_noG0_munu" in f else None
		G0_mu_nu = jnp.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None
		psi_l = jnp.asarray(f["psi_l"][:])
		psi_r = jnp.asarray(f["psi_r"][:])
		enk_l = jnp.asarray(f["enk_l"][:]) if "enk_l" in f else None
		enk_r = jnp.asarray(f["enk_r"][:]) if "enk_r" in f else None
	return V_qmunu, S_qmunu, psi_l, psi_r, enk_l, enk_r, V0_noG0_munu, G0_mu_nu


def load_labeled_arrays_from_h5(filename, mesh_xy):
	"""
	Load restart arrays and apply intended sharding, returning the same tuple
	shape as get_zeta_q_and_v_q_mu_nu:
	(V_qmunu, S_qmunu, psi_lT, psi_l, psi_r, psi_rT, enk_l, enk_r)
	"""
	V_qmunu, S_qmunu, psi_l_raw, psi_r_raw, enk_l, enk_r, V0_noG0_munu, G0_mu_nu = read_labeled_arrays_from_h5(filename)
	# Recreate shardings to match post-get_zeta layout
	x6y7_8 = NamedSharding(mesh_xy, P(None, None, None, None, None, None, 'x', 'y'))
	x3y4_5 = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
	y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y'))
	x3_4 = NamedSharding(mesh_xy, P(None, None, None, 'x'))
	y2_4 = NamedSharding(mesh_xy, P(None, None, 'y', None))
	x2_4 = NamedSharding(mesh_xy, P(None, None, 'x', None))
	V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, x6y7_8)
	if S_qmunu is not None:
		S_qmunu = jax.lax.with_sharding_constraint(S_qmunu, x3y4_5)
	if V0_noG0_munu is not None:
		V0_noG0_munu = jax.lax.with_sharding_constraint(V0_noG0_munu, NamedSharding(mesh_xy, P('x','y')))
	if G0_mu_nu is not None:
		# If it's a vector over rmu, shard along 'y'; if it's a matrix, shard (x,y)
		if G0_mu_nu.ndim == 1:
			G0_mu_nu = jax.lax.with_sharding_constraint(G0_mu_nu, NamedSharding(mesh_xy, P('y')))
		else:
			G0_mu_nu = jax.lax.with_sharding_constraint(G0_mu_nu, NamedSharding(mesh_xy, P('x','y')))
	psi_l = jax.lax.with_sharding_constraint(psi_l_raw, y3_4)
	psi_r = jax.lax.with_sharding_constraint(psi_r_raw, x3_4)
	psi_lT = jax.lax.with_sharding_constraint(psi_l.transpose(0, 2, 3, 1), x2_4)
	psi_rT = jax.lax.with_sharding_constraint(psi_r.transpose(0, 2, 3, 1), y2_4)
	return V_qmunu, S_qmunu, psi_lT, psi_l, psi_r, psi_rT, enk_l, enk_r, V0_noG0_munu, G0_mu_nu


def _mesh_coords_for_local_process(mesh_xy):
	devices_2d = np.array(mesh_xy.devices)
	local = list(jax.local_devices())
	local.sort(key=lambda d: d.id)
	target = local[0]
	coord = tuple(np.argwhere(devices_2d == target)[0])
	return coord


def save_restart_per_proc(prefix: str, V_qmunu, S_qmunu, psi_l, psi_r, enk_l, enk_r, meta, mesh_xy, V0_noG0_munu=None):
	"""Save per-process local shards to HDF5 files named by (x,y) mesh coords."""
	cx, cy = _mesh_coords_for_local_process(mesh_xy)
	rank = jax.process_index()
	fname = f"{prefix}.rank{rank}.x{cx}.y{cy}.h5"
	devices_2d = np.array(mesh_xy.devices)
	grid_x, grid_y = devices_2d.shape
	# Compute simple block slices for last-axis sharding
	def _block_slice(n, parts, idx):
		start = (n * idx) // parts
		end = (n * (idx + 1)) // parts
		return int(start), int(end)
	# V_qmunu last two dims shard as (x,y): (..., nrmu1, nrmu2)
	vx0, vx1 = _block_slice(int(V_qmunu.shape[-2]), grid_x, cx)
	vy0, vy1 = _block_slice(int(V_qmunu.shape[-1]), grid_y, cy)
	V_local = jax.device_get(V_qmunu[..., vx0:vx1, vy0:vy1])
	# V0_noG0_munu is (nrmu1, nrmu2), shard identically over (x,y)
	V0_local = None
	if V0_noG0_munu is not None:
		vx0_V0, vx1_V0 = _block_slice(int(V0_noG0_munu.shape[-2]), grid_x, cx)
		vy0_V0, vy1_V0 = _block_slice(int(V0_noG0_munu.shape[-1]), grid_y, cy)
		V0_local = jax.device_get(V0_noG0_munu[vx0_V0:vx1_V0, vy0_V0:vy1_V0])
	# S_qmunu shape (nkx,nky,nkz,nrmu1,nrmu2)
	S_local = None
	if S_qmunu is not None:
		sx0, sx1 = _block_slice(int(S_qmunu.shape[-2]), grid_x, cx)
		sy0, sy1 = _block_slice(int(S_qmunu.shape[-1]), grid_y, cy)
		S_local = jax.device_get(S_qmunu[..., sx0:sx1, sy0:sy1])
	# psi_l Y-sharded over last axis (nrmu)
	py0, py1 = _block_slice(int(psi_l.shape[-1]), grid_y, cy)
	psi_l_local = jax.device_get(psi_l[..., py0:py1])
	# psi_r X-sharded over last axis (nrmu)
	rx0, rx1 = _block_slice(int(psi_r.shape[-1]), grid_x, cx)
	psi_r_local = jax.device_get(psi_r[..., rx0:rx1])
	# Robust host conversion (JAX/CuPy/Numpy)
	def _to_np(a):
		try:
			return jax.device_get(a)
		except Exception:
			return a.get() if hasattr(a, 'get') else np.asarray(a)
	with h5py.File(fname, "w") as f:
		f.attrs['global_V_shape'] = np.array(V_qmunu.shape, dtype=np.int64)
		f.attrs['global_S_shape'] = np.array(S_qmunu.shape, dtype=np.int64) if S_qmunu is not None else np.array([-1], dtype=np.int64)
		f.attrs['global_psil_shape'] = np.array(psi_l.shape, dtype=np.int64)
		f.attrs['global_psir_shape'] = np.array(psi_r.shape, dtype=np.int64)
		if V0_noG0_munu is not None:
			f.attrs['global_V0_shape'] = np.array(V0_noG0_munu.shape, dtype=np.int64)
		f.attrs['grid_x'] = int(grid_x)
		f.attrs['grid_y'] = int(grid_y)
		f.attrs['coord_x'] = int(cx)
		f.attrs['coord_y'] = int(cy)
		f.create_dataset("V_local", data=V_local)
		if S_local is not None:
			f.create_dataset("S_local", data=S_local)
		if V0_local is not None:
			f.create_dataset("V0_noG0_local", data=V0_local)
		f.create_dataset("psi_l_local", data=psi_l_local)
		f.create_dataset("psi_r_local", data=psi_r_local)
		if enk_l is not None:
			f.create_dataset("enk_l", data=_to_np(getattr(enk_l, 'data', enk_l)))
		if enk_r is not None:
			f.create_dataset("enk_r", data=_to_np(getattr(enk_r, 'data', enk_r)))

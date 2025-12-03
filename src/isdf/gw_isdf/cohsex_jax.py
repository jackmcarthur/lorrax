# Standard Library imports
import os
# Force JAX to create four CPU devices before import
# os.environ['XLA_FLAGS'] = ' '.join(filter(None, [
# 	os.environ.get('XLA_FLAGS', ''),
# 	'--xla_cpu_multi_thread_eigen=true'
# ]))

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
import argparse
import configparser
import re

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial
from types import SimpleNamespace
from collections.abc import Iterable, Iterator
#jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_platform_name", "cpu")

# Initialize JAX distributed only when running multi-process
def _maybe_init_jax_distributed():
	proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
						 os.environ.get("JAX_NUM_PROCESSES",
						 os.environ.get("SLURM_NTASKS", "1"))))
	if proc_count > 1:
		coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
		if coord is None:
			host = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME") or "localhost"
			coord = f"{host}:12355"
		proc_id = int(os.environ.get("JAX_PROCESS_INDEX", os.environ.get("SLURM_PROCID", "0")))
		jax.distributed.initialize(coordinator_address=coord,
								   num_processes=proc_count,
								   process_id=proc_id)

# Global mesh for sharding across bands
_maybe_init_jax_distributed()
try:
	_default_devices = jax.devices()
except RuntimeError as exc:
	if "Unknown backend: 'gpu'" in str(exc):
		os.environ.pop("JAX_PLATFORM_NAME", None)
		os.environ["JAX_PLATFORMS"] = "cpu"
		_default_devices = jax.devices("cpu")
	else:
		raise
mesh_bands = Mesh(np.asarray(_default_devices), ("bands",))
from ..common.wfnreader import WFNReader
#from ..common.epsreader import EPSReader
from ..common import symmetry_maps
from ..common.load_wfns import read_Gvecs_to_devices, get_sharded_wfns, get_enk_bandrange
from .get_windows import get_window_info
from .w_isdf import get_chi0_jax, get_static_w_q_jax
from .vcoul import compute_q0_averages
from ..common.chi_from_dipole import read_dipole_h5, compute_S_omega
from ..common.epsreader import EPSReader
from .gw_file_io import (write_sigma_to_file, write_eqp_table, write_labeled_arrays_to_h5, 
                         read_labeled_arrays_from_h5, load_labeled_arrays_from_h5, 
                         save_restart_per_proc)
from .archive.jax_fixed_point_demo import crop_family_fixed_history_map
from ..common import Meta
from ..common.gamma_matrices import gammas_sparse
import isdf.common.timing as timing
import h5py
import builtins
import gc


# ================= Helper kernels (jitted) defined at module scope =================

@partial(jax.jit, donate_argnums=(0, 1))
def compute_CCT_ZCT_for_q(
	CCT_buf: jax.Array,
	ZCT_buf: jax.Array,
	psi_l_rmu: jax.Array,
	psi_r_rmu: jax.Array,
	psi_l_rtot: jax.Array,
	psi_r_rtot: jax.Array,
	psi_l_rmuT: jax.Array,
	psi_r_rmuT: jax.Array,
	k_l_indices: jax.Array,
	k_r_indices: jax.Array,
):
	"""Compute CCT and ZCT accumulators for a single q-point.

	Args are sharded arrays with shapes:
	- psi_*_rmu: (nk, nb, ns, n_rmu)
	- psi_*_rtot: (nk, nb, ns, n_rtot)
	- psi_*_rmuT: (nk, n_rmu, nb, ns)
	- k_*_indices: (n_pairs,)
	"""
	n_rmu = psi_l_rmu.shape[-1]
	n_rtot = psi_l_rtot.shape[-1]
	# Flatten band/spinor dimensions once to avoid per-iteration reshapes.
	psi_l_rmu_flat = psi_l_rmu.reshape(psi_l_rmu.shape[0], -1, n_rmu)
	psi_r_rmu_flat = psi_r_rmu.reshape(psi_r_rmu.shape[0], -1, n_rmu)
	psi_l_rtot_flat = psi_l_rtot.reshape(psi_l_rtot.shape[0], -1, n_rtot)
	psi_r_rtot_flat = psi_r_rtot.reshape(psi_r_rtot.shape[0], -1, n_rtot)
	psi_l_rmuT_flat = psi_l_rmuT.reshape(psi_l_rmuT.shape[0], n_rmu, -1)
	psi_r_rmuT_flat = psi_r_rmuT.reshape(psi_r_rmuT.shape[0], n_rmu, -1)

	# Zero reuse buffers (donated by caller) without allocating fresh storage.
	CCT_acc_init = CCT_buf * 0.0
	ZCT_acc_init = ZCT_buf * 0.0

	total_pairs = k_l_indices.shape[0]
	if total_pairs == 0:
		return CCT_acc_init, ZCT_acc_init

	def accumulate_k_pair(carry, i):
		CCT_acc, ZCT_acc = carry
		k_l = k_l_indices[i]
		k_r = k_r_indices[i]
		psi_l_rmu_k = psi_l_rmu_flat[k_l]
		psi_r_rmu_k = psi_r_rmu_flat[k_r]
		psi_l_rtot_k = psi_l_rtot_flat[k_l]
		psi_r_rtot_k = psi_r_rtot_flat[k_r]
		psi_l_rmuT_k = psi_l_rmuT_flat[k_l]
		psi_r_rmuT_k = psi_r_rmuT_flat[k_r]
		Pmu_l = psi_l_rmuT_k @ psi_l_rmu_k
		Pmu_r = psi_r_rmuT_k @ psi_r_rmu_k
		CCT_acc = CCT_acc + jnp.conj(Pmu_l) * Pmu_r
		P_l = psi_l_rmuT_k @ psi_l_rtot_k
		P_r = psi_r_rmuT_k @ psi_r_rtot_k
		ZCT_acc = ZCT_acc + jnp.conj(P_l) * P_r
		return (CCT_acc, ZCT_acc), None

	(CCT, ZCT), _ = jax.lax.scan(
		accumulate_k_pair,
		(CCT_acc_init, ZCT_acc_init),
		jnp.arange(total_pairs, dtype=jnp.int32),
	)
	return CCT, ZCT


def solve_zeta_cholesky(CCT: jax.Array, ZCT: jax.Array) -> jax.Array:
	"""Regularize CCT, chol factor, and solve for zeta_q (n_rmu, n_rtot)."""
	CCT = CCT + 1e-8 * jnp.mean(jnp.real(jnp.diag(CCT))) * jnp.eye(CCT.shape[0], dtype=CCT.dtype)
	CCT_cholesky = jax.scipy.linalg.cho_factor(CCT)
	return jax.scipy.linalg.cho_solve(CCT_cholesky, ZCT, overwrite_b=True)


@partial(jax.jit)
def compute_Sq_from_zeta(zeta_q: jax.Array) -> jax.Array:
	"""S_q = zeta^H zeta over rtot: (n_rmu, n_rtot) -> (n_rmu, n_rmu)."""
	return jnp.einsum('mu,nu->mn', jnp.conj(zeta_q), zeta_q, optimize=True)


def exp_ikr_fftbox(fft_nx: int, fft_ny: int, fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Return fractional coordinate grids for constructing exp(ik·r) on the FFT box."""
	fx = jnp.arange(fft_nx, dtype=jnp.float64)[None, :, None, None] / float(fft_nx)
	fy = jnp.arange(fft_ny, dtype=jnp.float64)[None, None, :, None] / float(fft_ny)
	fz = jnp.arange(fft_nz, dtype=jnp.float64)[None, None, None, :] / float(fft_nz)
	return fx, fy, fz


def fft_integer_axes(fft_nx: int, fft_ny: int, fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""Return integer FFT frequency grids in numpy.fft.fftfreq order."""
	gx = (jnp.fft.fftfreq(fft_nx) * fft_nx).astype(jnp.float64).reshape(fft_nx, 1, 1)
	gy = (jnp.fft.fftfreq(fft_ny) * fft_ny).astype(jnp.float64).reshape(1, fft_ny, 1)
	gz = (jnp.fft.fftfreq(fft_nz) * fft_nz).astype(jnp.float64).reshape(1, 1, fft_nz)
	return gx, gy, gz


def as_index_tuple(vec) -> tuple[int, ...]:
	"""Convert an integer vector into a Python index tuple."""
	return tuple(np.asarray(vec, dtype=np.int64))


def make_v_munu_kernel(
	fft_nx: int,
	fft_ny: int,
	fft_nz: int,
	nkx: int,
	nky: int,
	nkz: int,
	bvec: np.ndarray,
	cell_volume: float,
	sys_dim: int,
):
	"""Factory for a jitted kernel that computes v_{μν} for one q on the dense FFT grid."""
	if sys_dim != 2:
		raise NotImplementedError("make_v_munu_kernel currently supports sys_dim == 2 only")

	fx, fy, fz = exp_ikr_fftbox(fft_nx, fft_ny, fft_nz)
	gx, gy, gz = fft_integer_axes(fft_nx, fft_ny, fft_nz)
	gx_b, gy_b, gz_b = jnp.broadcast_arrays(gx, gy, gz)
	gstack_base = jnp.stack((gx_b, gy_b, gz_b), axis=-1)
	nkx = jnp.asarray(float(nkx))
	nky = jnp.asarray(float(nky))
	nkz = jnp.asarray(float(nkz))
	bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
	fact = jnp.float64(1.0 / cell_volume)
	zc = jnp.float64(np.pi / float(bvec[2, 2]))
	G_cart_base = jnp.einsum('...a,ab->...b', gstack_base, bvec_j, optimize=True)

	@partial(jax.jit)
	def kernel(
		zeta_q: jax.Array,
		qvec_wrapped: jax.Array,
	) -> tuple[jax.Array, jax.Array]:
		zeta_q_spatial = zeta_q.reshape(zeta_q.shape[0], fft_nx, fft_ny, fft_nz)
		phase = jnp.exp(-2j * jnp.pi * (qvec_wrapped[0]/nkx * fx + qvec_wrapped[1]/nky * fy + qvec_wrapped[2]/nkz * fz))
		zeta_qG = jnp.fft.fftn(zeta_q_spatial * phase, axes=(-3, -2, -1))  # unscaled.

		q_frac = jnp.asarray((
			qvec_wrapped[0] / nkx,
			qvec_wrapped[1] / nky,
			qvec_wrapped[2] / nkz,
		), dtype=jnp.float64)
		q_cart = jnp.einsum('a,ab->b', q_frac, bvec_j, optimize=True).reshape((1, 1, 1, 3))
		G_cart = G_cart_base + q_cart
		denom = jnp.sum(G_cart * G_cart, axis=-1)
		denom_zero = denom < 1e-12
		denom_safe = jnp.where(denom_zero, 1.0, denom)
		kxy = jnp.sqrt(G_cart[..., 0] * G_cart[..., 0] + G_cart[..., 1] * G_cart[..., 1])
		kz = G_cart[..., 2]
		f2d = 1.0 - jnp.exp(-zc * kxy) * jnp.cos(kz * zc)
		v_reg = (8.0 * jnp.pi / denom_safe) * f2d
		v_scaled = jnp.where(denom_zero, 0.0, v_reg * fact)
		sqrt_cube = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)

		weighted = zeta_qG * sqrt_cube[None, :, :, :]
		weighted_flat = weighted.reshape(weighted.shape[0], -1)
		v_munu = jnp.einsum('mG,nG->mn', jnp.conj(weighted_flat), weighted_flat, optimize=True)
		g0_mu = zeta_qG[:, 0, 0, 0]
		return v_munu, g0_mu

	return kernel


def compute_v_munu_from_zeta(
	zeta_q: jax.Array,
	qvec_wrapped: jax.Array,
	fft_nx: int,
	fft_ny: int,
	fft_nz: int,
	nkx: int,
	nky: int,
	nkz: int,
	bvec: np.ndarray,
	cell_volume: float,
	sys_dim: int,
) -> jax.Array:
	"""Reference helper that reuses the dense-grid kernel to obtain v_{μν}(q)."""
	kernel = make_v_munu_kernel(
		fft_nx,
		fft_ny,
		fft_nz,
		nkx,
		nky,
		nkz,
		bvec,
		cell_volume,
		sys_dim,
	)
	v_munu, _ = kernel(zeta_q, qvec_wrapped)
	return v_munu

def set_zeta_sharding(zeta_q: jax.Array, mesh_xy: Mesh | None):
	if mesh_xy is None:
		return zeta_q
	return jax.lax.with_sharding_constraint(zeta_q, NamedSharding(mesh_xy, P(('x','y'), None)))


def make_shardings(mesh_xy: Mesh) -> SimpleNamespace:
	"""Centralize all NamedSharding declarations used in this file."""
	return SimpleNamespace(
		# General 2D shardings
		xy_shard = NamedSharding(mesh_xy, P('x', 'y')),
		replicated_2 = NamedSharding(mesh_xy, P(None, None)),
		y_shard_vec = NamedSharding(mesh_xy, P('y')),
		xy0_2 = NamedSharding(mesh_xy, P(('x','y'), None)),
		x0y1_2 = NamedSharding(mesh_xy, P('x','y')),
		# Multi-dim array shardings
		x6y7_8 = NamedSharding(mesh_xy, P(None, None, None, None, None, None, 'x', 'y')),
		x3y4_5 = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y')),
		y3_4 = NamedSharding(mesh_xy, P(None, None, None, 'y')),
		x3_4 = NamedSharding(mesh_xy, P(None, None, None, 'x')),
		y2_4 = NamedSharding(mesh_xy, P(None, None, 'y', None)),
		x2_4 = NamedSharding(mesh_xy, P(None, None, 'x', None)),
		# Pipeline shardings
		XT_shard = NamedSharding(mesh_xy, P(None, None, 'x', None)),
		Y_shard = NamedSharding(mesh_xy, P(None, None, None, 'y')),
		X_shard = NamedSharding(mesh_xy, P(None, None, None, 'x')),
		YT_shard = NamedSharding(mesh_xy, P(None, None, 'y', None)),
		V_shard = NamedSharding(mesh_xy, P('x', 'y', None, None, None)),
		out_shard = NamedSharding(mesh_xy, P(None, None, None)),
	)


def _get_kvec_lookup(sym) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
	kvecs_np = getattr(sym, "_kvecs_asints_np", None)
	if kvecs_np is None:
		kvecs_np = np.asarray(sym.kvecs_asints, dtype=np.int32)
		setattr(sym, "_kvecs_asints_np", kvecs_np)
	lookup = getattr(sym, "_kvec_lookup", None)
	if lookup is None:
		lookup = {tuple(int(v) for v in vec): idx for idx, vec in enumerate(kvecs_np)}
		setattr(sym, "_kvec_lookup", lookup)
	return kvecs_np, lookup


def iter_qpoint_data(sym, meta: Meta) -> Iterator[SimpleNamespace]:
	kvecs_np, k_lookup = _get_kvec_lookup(sym)
	kgrid_np = meta.kgrid_np
	half_grid = kgrid_np // 2
	k_r_indices = jnp.arange(kvecs_np.shape[0], dtype=jnp.int32)
	for qvec_nonneg in np.ndindex(*meta.kgrid):
		qvec_nonneg_np = np.asarray(qvec_nonneg, dtype=np.int32)
		qvec_wrapped = np.where(qvec_nonneg_np > half_grid, qvec_nonneg_np - kgrid_np, qvec_nonneg_np)
		targets = (kvecs_np - qvec_wrapped) % kgrid_np
		k_l_np = np.fromiter((k_lookup[tuple(t)] for t in targets), dtype=np.int32, count=targets.shape[0])
		qvec_frac = qvec_wrapped.astype(np.float64) / kgrid_np
		iq = sym.find_qpoint_index(qvec_frac, tol=1e-6)
		iq_cpu = iq.get() if hasattr(iq, "get") else int(iq)
		yield SimpleNamespace(
			q_index=iq_cpu,
			q_nonneg=qvec_nonneg_np,
			q_wrapped=jnp.asarray(qvec_wrapped, dtype=jnp.int32),
			k_l_indices=jnp.asarray(k_l_np, dtype=jnp.int32),
			k_r_indices=k_r_indices,
		)


def _legacy_q_data_iter(preprocessed_q_data) -> Iterator[SimpleNamespace]:
	all_k_l_indices, all_k_r_indices, all_qvecs_wrapped, all_qvecs_nonneg, all_iq_indices = preprocessed_q_data[:5]
	for k_l, k_r, q_wrapped, q_nonneg, iq in zip(
		np.asarray(all_k_l_indices),
		np.asarray(all_k_r_indices),
		np.asarray(all_qvecs_wrapped),
		np.asarray(all_qvecs_nonneg),
		np.asarray(all_iq_indices),
	):
		yield SimpleNamespace(
			q_index=int(np.asarray(iq)),
			q_nonneg=np.asarray(q_nonneg, dtype=np.int32),
			q_wrapped=jnp.asarray(q_wrapped, dtype=jnp.int32),
			k_l_indices=jnp.asarray(k_l, dtype=jnp.int32),
			k_r_indices=jnp.asarray(k_r, dtype=jnp.int32),
		)


def build_q_coulomb_cache(
	wfn,
	sym,
	meta: Meta,
	do_Dmunu: bool,
	sys_dim: int,
	mesh_xy: Mesh | None = None,
) -> SimpleNamespace:
	"""Precompute q-grid Coulomb metadata reused inside the q-loop.

	Returns batched JAX arrays for the regular shapes so the q-loop can
	run with minimal host-device transfers.
	"""
	_ = do_Dmunu
	_ = sys_dim
	k_l_list: list[jax.Array] = []
	q_nonneg_list: list[jax.Array] = []
	q_wrapped_list: list[jax.Array] = []
	iq_indices: list[int] = []
	k_r_indices_ref: jax.Array | None = None

	for q_data in iter_qpoint_data(sym, meta):
		if k_r_indices_ref is None:
			k_r_indices_ref = q_data.k_r_indices
		k_l_list.append(q_data.k_l_indices)
		q_nonneg_list.append(jnp.asarray(q_data.q_nonneg, dtype=jnp.int32))
		q_wrapped_list.append(jnp.asarray(q_data.q_wrapped, dtype=jnp.int32))
		iq_indices.append(int(q_data.q_index))

	if not k_l_list:
		return SimpleNamespace(
			num_q=0,
			k_l_indices=jnp.zeros((0, 0), dtype=jnp.int32),
			k_r_indices=jnp.zeros((0,), dtype=jnp.int32),
			q_nonneg=jnp.zeros((0, 3), dtype=jnp.int32),
			q_wrapped=jnp.zeros((0, 3), dtype=jnp.int32),
			iq_indices=jnp.zeros((0,), dtype=jnp.int32),
		)

	k_l_stacked = jnp.stack(k_l_list, axis=0)
	q_nonneg_arr = jnp.stack(q_nonneg_list, axis=0)
	q_wrapped_arr = jnp.stack(q_wrapped_list, axis=0)
	iq_arr = jnp.asarray(iq_indices, dtype=jnp.int32)
	k_r_indices_arr = jnp.asarray(k_r_indices_ref, dtype=jnp.int32)

	if mesh_xy is not None:
		# Replicate metadata across the mesh to avoid repeated transfers.
		rep_q = NamedSharding(mesh_xy, P(None, None))
		q_nonneg_arr = jax.device_put(q_nonneg_arr, rep_q)
		q_wrapped_arr = jax.device_put(q_wrapped_arr, rep_q)
		rep_k = NamedSharding(mesh_xy, P(None, None))
		k_l_stacked = jax.device_put(k_l_stacked, rep_k)
		k_r_indices_arr = jax.device_put(k_r_indices_arr, NamedSharding(mesh_xy, P(None)))
		iq_arr = jax.device_put(iq_arr, NamedSharding(mesh_xy, P(None)))

	return SimpleNamespace(
		num_q=int(q_nonneg_arr.shape[0]),
		k_l_indices=k_l_stacked,
		k_r_indices=k_r_indices_arr,
		q_nonneg=q_nonneg_arr,
		q_wrapped=q_wrapped_arr,
		iq_indices=iq_arr,
	)


def determine_wcoul0(params, input_dir, wfn, sym, meta, print_fn):
	"""Resolve (v_c0, w_c0) head averages using user preference fallback order."""
	want_source = str(params.get("wcoul0_source", "epshead")).strip().lower()
	if want_source not in ("epshead", "s_tensor"):
		print_fn(f"Unknown wcoul0_source={want_source}; defaulting to 'epshead'")
		want_source = "epshead"

	eps0_path = os.path.join(input_dir, "eps0mat.h5")
	dipole_path = os.path.join(input_dir, "dipole.h5")

	def from_epshead():
		if not os.path.exists(eps0_path):
			return None
		try:
			eps0 = EPSReader(eps0_path)
			vc0_mean, wcoul0 = compute_q0_averages(
				wfn,
				jnp.asarray(eps0.epshead, dtype=jnp.complex128),
				meta,
				S_cart=None,
			)
			print_fn("Using eps0mat.h5 epshead-based wcoul0")
			return vc0_mean, wcoul0, "epshead"
		except Exception as exc:  # pragma: no cover - diagnostic path
			print_fn(f"epshead wcoul0 failed: {exc}")
			return None

	def from_s_tensor():
		if not os.path.exists(dipole_path):
			print_fn(f"dipole.h5 not found at {dipole_path}; cannot build S(0) wcoul0")
			return None
		try:
			dipole_cart, deltaE = read_dipole_h5(dipole_path)
			nk_tot = int(sym.nk_tot)
			nb = int(dipole_cart.shape[2])
			nelec = int(wfn.nelec)
			occ = np.zeros((nk_tot, nb), dtype=float)
			occ[:, :max(0, min(nelec, nb))] = 1.0
			f_nk = jnp.asarray(occ, dtype=jnp.float64)
			omegas = jnp.asarray([0.0], dtype=jnp.float64)
			S_cart_omega0 = compute_S_omega(
				dipole_cart,
				deltaE,
				f_nk,
				float(wfn.cell_volume),
				int(sym.nk_tot),
				int(wfn.nspin),
				int(wfn.nspinor),
				omegas,
				eta=0.0,
			)[0]
			vc0_mean, wcoul0 = compute_q0_averages(
				wfn,
				jnp.asarray(0.0, dtype=jnp.float64),
				meta,
				S_cart=S_cart_omega0,
			)
			print_fn("Using dipole.h5 S(0)-based wcoul0")
			return vc0_mean, wcoul0, "s_tensor"
		except Exception as exc:  # pragma: no cover - diagnostic path
			print_fn(f"S(0) wcoul0 failed: {exc}")
			return None

	source_order = [want_source] + [s for s in ("epshead", "s_tensor") if s != want_source]
	for source in source_order:
		result = from_epshead() if source == "epshead" else from_s_tensor()
		if result is not None:
			return result
	raise RuntimeError("Failed to determine wcoul0: neither eps0mat.h5 epshead nor dipole.h5 S(0) available")


def read_cohsex_input(filename: str) -> dict:
	"""Parse input file for the COHSEX driver, allowing a QE K_POINTS block.

	We extract the [cohsex] section using a substring to avoid configparser
	errors from non-INI blocks like K_POINTS. The K_POINTS {crystal_b} block
	is parsed manually and returned under 'kpoints_crystal_b'.
	"""
	with open(filename, 'r') as f:
		lines = f.readlines()

	# Locate [cohsex] section boundaries
	start = None
	for i, l in enumerate(lines):
		if l.strip().lower().startswith('[cohsex]'):
			start = i
			break
	if start is None:
		for i, l in enumerate(lines):
			if re.match(r"\s*\[.*\]", l):
				start = i
				break
	end = len(lines)
	# Locate optional K_POINTS block in full file first
	kp_idx = None
	for i, l in enumerate(lines):
		ls = l.strip().lower()
		if ls.startswith("k_points"):
			kp_idx = i
			break
	seg_count = 0
	kp_end = None
	if kp_idx is not None and kp_idx + 1 < len(lines):
		# count is on the next line; read exactly that many entries
		try:
			seg_count = int(lines[kp_idx + 1].strip().split()[0])
		except Exception:
			seg_count = 0
		kp_end = min(len(lines), kp_idx + 2 + max(seg_count, 0))
	if start is not None:
		for j in range(start + 1, len(lines)):
			if re.match(r"\s*\[.*\]", lines[j]):
				end = j
				break
		# Remove K_POINTS block from the ini text before feeding to configparser
		if kp_idx is not None and (start <= kp_idx < end):
			section_lines = lines[start:kp_idx] + lines[(kp_end if kp_end is not None else kp_idx+1):end]
			ini_text = ''.join(section_lines)
		else:
			ini_text = ''.join(lines[start:end])
		parser = configparser.ConfigParser()
		parser.read_string(ini_text)
		section = parser["cohsex"] if "cohsex" in parser else parser[parser.sections()[0]]
		getb = section.getboolean
		get = section.get
		geti = section.getint
		# ============================================================================
		# PARAMETER DEFINITIONS:
		# ============================================================================
		# restart:       If True, load V_qmunu/wavefunctions from taggedarrays.h5
		#                instead of rebuilding ISDF from scratch. Default=True.
		# x_only:        If True, compute bare exchange only (no screening).
		#                Cannot be True if do_screened=True. Default=False.
		# do_screened:   If True, build W from (1-Vχ)⁻¹V and use for Σ_x.
		#                If False, use bare Coulomb V. Default=True.
		# bispinor:      If True, use 2-component spinor wavefunctions (SOC).
		#                Default=False (scalar or collinear spin).
		# wcoul0_source: Method for q→0 head average: 'epshead' (from eps0mat.h5)
		#                or 's_tensor' (from dipole.h5). Default='s_tensor'.
		# self_consistent: If True, run fixed-point SCF loop. Default=False.
		# nval:          Number of valence bands in sigma window. These are the
		#                highest occupied bands: indices [nelec-nval, nelec).
		# ncond:         Number of conduction bands in sigma window. These are the
		#                lowest unoccupied bands: indices [nelec, nelec+ncond).
		# nband:         Total bands to load (for chi0, etc.). Usually > nval+ncond.
		# sys_dim:       Dimensionality: 2=2D (slab with truncated Coulomb),
		#                3=3D (bulk, not yet implemented). Default=2.
		# debug_hartree: If True, print diagnostic info for Hartree calculation.
		# ============================================================================
		params = {
			"restart": getb("restart", fallback=True),           # load from h5 vs rebuild
			"x_only": getb("x_only", fallback=False),            # bare exchange only
			"do_screened": getb("do_screened", fallback=True),   # use W instead of V
			"bispinor": getb("bispinor", fallback=False),        # 2-component spinors
			"wcoul0_source": get("wcoul0_source", fallback="s_tensor").strip().lower(),
			"wfn_file": get("wfn_file", fallback="WFN.h5"),
			"centroids_file": get("centroids_file", fallback="centroids_frac.txt"),
			"output_file": get("output_file", fallback="eqp0_noqsym.dat"),
			"self_consistent": getb("self_consistent", fallback=False),
			"kin_ion_file": get("kin_ion_file", fallback="kin_ion.h5"),
			"eqp_output_file": get("eqp_output_file", fallback="eqp.dat"),
			"nval": geti("nval", fallback=5),    # valence bands in sigma window
			"ncond": geti("ncond", fallback=5),  # conduction bands in sigma window
			"nband": geti("nband", fallback=100), # total bands for chi0/screening
			"sys_dim": geti("sys_dim", fallback=2),  # 2=slab, 3=bulk
			"debug_hartree": getb("debug_hartree", fallback=False),
		}
	else:
		# Fallback defaults if no section found
		params = {
			"restart": True,
			"x_only": False,
			"do_screened": True,
			"bispinor": False,
			"wcoul0_source": "s_tensor",
			"wfn_file": "WFN.h5",
			"centroids_file": "centroids_frac.txt",
			"output_file": "eqp0_noqsym.dat",
			"self_consistent": False,
			"kin_ion_file": "kin_ion.h5",
			"eqp_output_file": "eqp.dat",
			"nval": 5,
			"ncond": 5,
			"nband": 100,
			"sys_dim": 2,
			"debug_hartree": False,
		}

	# Parse optional QE-style K_POINTS block: take the number after it, read next that many lines
	if kp_idx is not None:
		j = kp_idx + 1
		try:
			nseg = int(lines[j].strip().split()[0])
		except Exception:
			nseg = 0
		segments = []
		for k in range(nseg):
			row_idx = j + 1 + k
			if row_idx >= len(lines):
				break
			row_full = lines[row_idx].rstrip('\n')
			label = None
			comment_split = None
			for marker in ('#', '!', ';'):
				if marker in row_full:
					comment_split = row_full.split(marker, 1)
					label = comment_split[1].strip() or None
					row = comment_split[0].strip()
					break
			if comment_split is None:
				row = row_full.strip()
			if not row:
				continue
			parts = row.split()
			if len(parts) < 3:
				continue
			x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
			npts = int(parts[3]) if len(parts) >= 4 else 1
			segments.append({"k": [x, y, z], "n": npts, "label": label})
		if segments:
			params["kpoints_crystal_b"] = {"segments": segments}
	return params

# The current implementation focuses on the static COHSEX limit.  Many of the
# routines below (e.g. chi0 and sigma construction) are written in a style that
# follows the complex time shredded propagator (CTSP) formulation so that we can
# later restore full frequency dependence and iterate towards self-consistency.


# return ranges of bands necessary for \sigma_{X,SX,COH}
def get_bandranges(nv, nc, nband, nelec):
	r"""Return ranges of bands necessary for \sigma_{X,SX,COH}"""
	nvrange = [int(nelec - nv), int(nelec)]
	ncrange = [int(nelec), int(nelec + nc)]
	nsigmarange = [int(nelec - nv), int(nelec + nc)]
	n_fullrange = [0, int(nband)]
	n_valrange = [0, int(nelec)]
	return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


# get_enk_bandrange moved to common.load_wfns and refactored to return plain JAX arrays

def get_zeta_q_and_v_q_mu_nu(
	wfn,
	sym,
	bandrange_l,
	bandrange_r,
	V_qG,
	meta: Meta,
	psi_l_rtot,
	psi_r_rtot,	
	psi_l_rmu,
	psi_r_rmu,
	psi_l_rmuT,
	psi_r_rmuT,
	*,  # make the rest keyword-only
	preprocessed_q_data=None,
	bispinor=False,
	mesh_xy=None,
):
	"""Find the interpolative separable density fitting representation."""
	# Get dimensions with padding for distributed computation
	nb_l = bandrange_l[1] - bandrange_l[0]
	nb_r = bandrange_r[1] - bandrange_r[0]
	total_devices = jax.process_count() * jax.local_device_count()
	bands_per_device_l = (nb_l + total_devices - 1) // total_devices
	bands_per_device_r = (nb_r + total_devices - 1) // total_devices
	nb_l_padded = total_devices * bands_per_device_l
	nb_r_padded = total_devices * bands_per_device_r

	# initialize output V_q,mu,nu array
	sys_dim = getattr(meta, "sys_dim", 2)
	# V_qG retained for API compatibility; per-q Coulomb data is built on demand below.
	# Create a JAX array for V_qmunu with 2D sharding over the last two dims (nrmu1,nrmu2)

	sh = make_shardings(mesh_xy) if mesh_xy is not None else None
	V_shape = (1, meta.npol, meta.npol, *meta.kgrid, meta.n_rmu, meta.n_rmu)
	S_shape = (*meta.kgrid, meta.n_rmu, meta.n_rmu)
	V_qmunu = jnp.zeros(V_shape, dtype=jnp.complex128)
	S_qmunu = jnp.zeros(S_shape, dtype=jnp.complex128)
	if sh is not None:
		V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, sh.x6y7_8)
		S_qmunu = jax.lax.with_sharding_constraint(S_qmunu, sh.x3y4_5)

	V_q_names = ["nfreq", "npol1", "npol2", "nkx", "nky", "nkz", "nrmu1", "nrmu2"]
	fft_nx, fft_ny, fft_nz = (int(dim) for dim in meta.fft_grid)

	fx_grid, fy_grid, fz_grid = exp_ikr_fftbox(fft_nx, fft_ny, fft_nz)

	# 3. Convert weights and other arrays to JAX
	kgrid = meta.kgrid_jax
	# Get band ranges and weights in JAX
	enk_l, weights_l = get_enk_bandrange(
		wfn, sym, bandrange_l, (bandrange_r[0], bandrange_l[1])
	)
	enk_r, weights_r = get_enk_bandrange(
		wfn, sym, bandrange_r, (bandrange_r[0], bandrange_l[1])
	)
	# Pad weights purely on device to match the padded band dimensions
	#pad_l = int((nb_l_padded * meta.nspinor) - int(weights_l.shape[1]))
	#pad_r = int((nb_r_padded * meta.nspinor) - int(weights_r.shape[1]))
	#weights_l = jnp.pad(weights_l, ((0, 0), (0, pad_l)), mode='constant') if pad_l > 0 else weights_l
	##weights_r = jnp.pad(weights_r, ((0, 0), (0, pad_r)), mode='constant') if pad_r > 0 else weights_r

	##########################################
	# Iterate q-point data on demand (optional legacy cache)
	##########################################
	if preprocessed_q_data is None:
		print("Iterating q-point data on demand (no preprocessing cache).")
		q_data_iter: Iterable[SimpleNamespace] = iter_qpoint_data(sym, meta)
	else:
		if isinstance(preprocessed_q_data, tuple):
			print("Using legacy preprocessed q-point tuple.")
			q_data_iter = _legacy_q_data_iter(preprocessed_q_data)
		else:
			q_data_iter = iter(preprocessed_q_data)

	bvec_cart = jnp.asarray(wfn.blat * wfn.bvec, dtype=jnp.float64)
	inv_cell_volume = jnp.float64(1.0 / float(wfn.cell_volume))
	zc = jnp.float64(np.pi / float((wfn.blat * wfn.bvec)[2, 2]))
	gx_fft, gy_fft, gz_fft = fft_integer_axes(fft_nx, fft_ny, fft_nz)
	gx_fft_b, gy_fft_b, gz_fft_b = jnp.broadcast_arrays(gx_fft, gy_fft, gz_fft)
	gstack_fft = jnp.stack((gx_fft_b, gy_fft_b, gz_fft_b), axis=-1)
	G_cart_base_fft = jnp.einsum('...a,ab->...b', gstack_fft, bvec_cart, optimize=True)
	G0_mu_nu = None
	# Main q-point loop
	for q_idx, q_data in enumerate(q_data_iter):
		k_l_indices = q_data.k_l_indices
		k_r_indices = q_data.k_r_indices
		qvec_nonneg = q_data.q_nonneg
		iq_cpu = q_data.q_index
		qvec = jnp.asarray(q_data.q_wrapped, dtype=jnp.float64)
		# CCT and ZCT for this q-point
		n_rmu = psi_l_rmu.shape[-1]
		n_rtot = psi_l_rtot.shape[-1]
		CCT, ZCT = compute_CCT_ZCT_for_q(
			jnp.zeros((n_rmu, n_rmu), dtype=jnp.complex128),
			jnp.zeros((n_rmu, n_rtot), dtype=jnp.complex128),
			psi_l_rmu,
			psi_r_rmu,
			psi_l_rtot,
			psi_r_rtot,
			psi_l_rmuT,
			psi_r_rmuT,
			k_l_indices,
			k_r_indices,
		)

		# Minimal sharding hint: keep rows on x and cols on y, matching wfn shardings
		CCT = jax.lax.with_sharding_constraint(CCT, sh.replicated_2)
		ZCT = jax.lax.with_sharding_constraint(ZCT, sh.xy_shard)

		# lstsq solve with optimal sharding (Y over longer rtot dimension)
		CCT = CCT + 1e-8 * jnp.mean(jnp.real(jnp.diag(CCT))) * jnp.eye(CCT.shape[0], dtype=CCT.dtype)
		CCT_cholesky = jax.scipy.linalg.cho_factor(CCT)
		# should make this parallel over xy in ZCT, CCT_cholesky replicated
		zeta_q = jax.scipy.linalg.cho_solve(CCT_cholesky, ZCT, overwrite_b=True)
		zeta_q = jax.lax.with_sharding_constraint(zeta_q, sh.xy0_2)
		S_q_local = compute_Sq_from_zeta(zeta_q)
		S_q_local = jax.lax.with_sharding_constraint(S_q_local, sh.x0y1_2)
		qx, qy, qz = as_index_tuple(qvec_nonneg)
		S_qmunu = S_qmunu.at[qx, qy, qz, :, :].set(S_q_local)

		# Reshape zeta_q: (n_rmu, n_rtot) → (n_rmu, nx, ny, nz)
		zeta_q_spatial = zeta_q.reshape(meta.n_rmu, fft_nx, fft_ny, fft_nz)
		# Phase removal and FFT
		kgrfloat = jnp.asarray(kgrid, dtype=jnp.float64)
		phase = jnp.exp(-2j * jnp.pi * (qvec[0] / kgrfloat[0] * fx_grid + qvec[1] / kgrfloat[1] * fy_grid + qvec[2] / kgrfloat[2] * fz_grid))
		zeta_q_spatial = zeta_q_spatial * phase
		zeta_qG = jnp.fft.fftn(zeta_q_spatial, axes=(-3, -2, -1))

		q_frac = jnp.asarray((
			qvec[0] / kgrfloat[0],
			qvec[1] / kgrfloat[1],
			qvec[2] / kgrfloat[2],
		), dtype=jnp.float64)
		q_cart = jnp.einsum('a,ab->b', q_frac, bvec_cart, optimize=True).reshape((1, 1, 1, 3))
		G_cart = G_cart_base_fft + q_cart
		denom = jnp.sum(G_cart * G_cart, axis=-1)
		denom_zero = denom < 1e-12
		denom_safe = jnp.where(denom_zero, 1.0, denom)
		kxy = jnp.sqrt(G_cart[..., 0] * G_cart[..., 0] + G_cart[..., 1] * G_cart[..., 1])
		kz = G_cart[..., 2]
		f2d = 1.0 - jnp.exp(-zc * kxy) * jnp.cos(kz * zc)
		v_reg = (8.0 * jnp.pi / denom_safe) * f2d
		v_scaled = jnp.where(denom_zero, 0.0, v_reg * inv_cell_volume)
		sqrt_cube = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)

		weighted = zeta_qG * sqrt_cube[None, :, :, :]
		weighted_flat = weighted.reshape(weighted.shape[0], -1)
		v_munu = jnp.einsum('mG,nG->mn', jnp.conj(weighted_flat), weighted_flat, optimize=True)
		g0_mu = zeta_qG[:, 0, 0, 0]

		#V_weighted = V_qfullG_masked[None, :] * zeta_qG_flat
		#V_qmunu_q = jnp.conj(zeta_qG_masked) @ V_weighted.T
		# Store result into sharded JAX array at this q-point
		qx, qy, qz = as_index_tuple(qvec_nonneg)
		V_qmunu = V_qmunu.at[0, 0, 0, qx, qy, qz, :, :].set(v_munu)
		if G0_mu_nu is None and jnp.all(qvec_nonneg == 0):
			G0_mu_nu = g0_mu
		print(f"qpoint {iq_cpu} done")

	if sh is not None:
		with mesh_xy:
			psil_Y = jax.device_put(psi_l_rmu, sh.y3_4)
			psilT_X = jax.lax.with_sharding_constraint(psil_Y.transpose(0, 2, 3, 1), sh.x2_4)
			psir_X = jax.device_put(psi_r_rmu, sh.x3_4)
			psirT_Y = jax.lax.with_sharding_constraint(psir_X.transpose(0, 2, 3, 1), sh.y2_4)
	else:
		psil_Y = psi_l_rmu
		psilT_X = psi_l_rmuT
		psir_X = psi_r_rmu
		psirT_Y = psi_r_rmuT

	return V_qmunu, S_qmunu, psilT_X, psil_Y, psir_X, psirT_Y, enk_l, enk_r # T indicates b at end.

# ================= JAX-sharded Sigma pipeline =================

def get_G_mu_nu_jax(psi_vTX, psi_vY):
	"""Pure: psi_* (nk, nb, nspinor, n_rmu) -> G_k (nk, nspinor, n_rmu, nspinor, n_rmu).
	Zero-comm contraction when left is X-sharded on rmu and right is Y-sharded on rmu.
	Einsum order: kxmb,kbyn->kxmyn (spin indices x,y kept separate from rmu m,n)."""
	# Contract over band m only. x y rmu, s,t spinor. believe this is optimal
	G_k = jnp.einsum('ksxm,kmty->ksxty', psi_vTX, jnp.conj(psi_vY), optimize=True)
	return G_k

def get_G_R_jax(G_k, nkx, nky, nkz):
	"""Pure: (nk, s1,rmu1,s2,rmu2) -> (s1,rmu1,s2,rmu2,nkx,nky,nkz)."""
	G_k = G_k.transpose(1, 2, 3, 4, 0)
	G_k = G_k.reshape(*G_k.shape[:4], nkx, nky, nkz)  # (s1,rmu1,s2,rmu2,nkx,nky,nkz)
	G_k = jax.lax.with_sharding_constraint(G_k, P(None, 'x', None, 'y', None, None, None))
	return jnp.fft.ifftn(G_k, axes=(4,5,6), norm='ortho')

def get_sigma_x_mu_nu_jax(G_R, V_mu_nu, nk_tot):
	"""Pure: G_R (s1,rmu1,s2,rmu2,nkx,nky,nkz), V_mu_nu (rmu1,rmu2,nkx,nky,nkz) -> sigma_k same shape as G_R."""
	V_R = V_mu_nu[None, :, None, :, :, :, :]
	V_R = jnp.array(V_R, copy=True)
	sigma_R = G_R * jnp.fft.ifftn(V_R, axes=(4,5,6), norm='ortho') * (-1.0 / jnp.sqrt(nk_tot)) # 1/sqrt(Nk) is the correct round-trip factor for a convolution with ortho ffts.
	sigma_R = jax.lax.with_sharding_constraint(sigma_R, P(None, 'x', None, 'y', None, None, None))
	return jnp.fft.fftn(sigma_R, axes=(4,5,6), norm='ortho')

def get_sigma_x_kij_jax(psi_sigX, psi_sigTY, sigma_k_munu):
	"""Pure: psi_* (nk, nb, ns, rmu), sigma_k_munu (s1,rmu1,s2,rmu2,nkx,nky,nkz) -> sigma_kij (nk, nb, nb)."""
	nkx, nky, nkz = sigma_k_munu.shape[-3:]
	nk = nkx * nky * nkz
	sigma_k = sigma_k_munu.transpose(4, 5, 6, 0, 1, 2, 3).reshape(nk, *sigma_k_munu.shape[:4]) # (nk,s1,rmu1,s2,rmu2)
	left = jnp.einsum('kmsx,ksxty->kmty', jnp.conj(psi_sigX), sigma_k, optimize=True)
	return jnp.einsum('kmty,ktyn->kmn', left, psi_sigTY, optimize=True)
# m,n bands, s,t spinor, x,y rmu

def compute_sigma_pipeline_jax(
	psi_l_rmuT_X,
	psi_l_rmu_Y,
	psi_r_rmu_X,
	psi_r_rmuT_Y,
	V_mu_nu,
	V0_munu,
	nkx: int,
	nky: int,
	nkz: int,
	nk_tot: int,
	nspinor: int,
	fft_vol_au: float,
):
	"""
	Pure JAX pipeline: compute exchange self-energy and Hartree matrix elements.
	
	Returns:
		sigma_kij: (nk, nb_sigma, nb_sigma) complex - exchange self-energy
		hartree_kmn: (nk, nb_sigma, nb_sigma) complex - Hartree matrix elements
	
	Wavefunctions:
		psi_l: valence bands only (for density ρ and Green's function G)
		       shape (nk, nval, nspinor, n_rmu) where nval = nelec - b1 (all occupied)
		psi_r: sigma window bands (for projecting to band basis)
		       shape (nk, n_sigma, nspinor, n_rmu) where n_sigma = nval + ncond
	
	EXCHANGE (Σ_x):
		G_μν(k) = Σ_occ ψ*_nk(r_μ) ψ_nk(r_ν)     [Green's function from valence]
		G_μν(R) = FFT[ G_μν(k) ]                  [to real-space lattice]
		Σ_μν(k) = (1/N_k) Σ_R G_μν(R) V_μν(R)    [exchange in ISDF basis]
		Σ_ij(k) = Σ_μν ψ*_i(r_μ) Σ_μν ψ_j(r_ν)   [project to sigma bands]
	
	HARTREE (V_H):
		ρ_μ = (1/N_k) Σ_k,n,s |ψ_nk(r_μ)|²       [density at centroids, from valence]
		[Vρ]_μ = Σ_ν V0_μν ρ_ν                    [Hartree potential at centroids]
		<m|V_H|n>_k = Σ_μ,s ψ*_mk(r_μ) [Vρ]_μ ψ_nk(r_μ)  [project to sigma bands]
	
	Key: V0_munu is V(q=0) with G=0 component EXCLUDED (to avoid divergence).
	     The G=0 piece is added back via the head correction in the main pipeline.
	"""
	# ========== EXCHANGE SELF-ENERGY ==========
	# G_μν(k): Green's function in ISDF basis, built from VALENCE bands only
	G_k = get_G_mu_nu_jax(psi_l_rmuT_X, psi_l_rmu_Y)  # (nkx,nky,nkz,spin,μ,spin,ν)
	# G_μν(R): FFT to real-space lattice vectors R
	G_R = get_G_R_jax(G_k, nkx, nky, nkz)
	# Σ_μν(k): exchange self-energy via ISDF, σ_μν = (1/Nk) Σ_R G_μν(R) V_μν(R)
	sigma_k_munu = get_sigma_x_mu_nu_jax(G_R, V_mu_nu, nk_tot)
	# Σ_ij(k): project to SIGMA WINDOW bands using psi_r
	sigma_kij = get_sigma_x_kij_jax(psi_r_rmu_X, psi_r_rmuT_Y, sigma_k_munu)

	# ========== HARTREE MATRIX ELEMENTS ==========
	# Step 1: Density at centroids from VALENCE bands (psi_l)
	# ρ_μ = (1/Nk) Σ_k,n,s |ψ_nk(r_μ)|²
	# psi_l_rmu_Y has shape (nk, nval, nspinor, n_rmu), contracted to (n_rmu,)
	rho_mu = jnp.einsum('knsx,knsx->x', jnp.conj(psi_l_rmu_Y), psi_l_rmu_Y, optimize=True)
	rho_mu = rho_mu * 1.0 / jnp.asarray(nk_tot, dtype=jnp.float64)  # BZ integration: 1/Nk
	
	# Step 2: Hartree potential at centroids
	# [Vρ]_μ = Σ_ν V0_μν ρ_ν
	# V0_munu is V(q=0) with G=0 excluded to avoid Coulomb divergence
	Vrho_mu = jnp.einsum('xy,y->x', V0_munu, rho_mu, optimize=True)
	
	# Step 3: Project to SIGMA WINDOW bands (psi_r)
	# <mk|V_H|nk> = Σ_μ,s ψ*_mk(r_μ) [Vρ]_μ ψ_nk(r_μ)
	# Note: This uses psi_r which includes both valence AND conduction in sigma window
	hartree_kmn = jnp.einsum('kmsx,x,knsx->kmn', jnp.conj(psi_r_rmu_X), Vrho_mu, psi_r_rmu_X, optimize=True)
	
	return sigma_kij, hartree_kmn


def load_kin_ion_submatrix(h5_path: str, band_start: int, band_stop: int) -> jax.Array:
	"""Load sub-block of the kin+ion Hamiltonian for the requested band slice."""
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")
	with h5py.File(h5_path, "r") as h5:
		if "kin_ion" not in h5:
			raise KeyError("Dataset 'kin_ion' missing from kin_ion file")
		kin_dset = h5["kin_ion"]
		nb_total = kin_dset.shape[1]
		if band_stop > nb_total:
			raise ValueError(
				f"Requested bands require {band_stop} states but kin_ion only has {nb_total}"
			)
		sub = kin_dset[:, band_start:band_stop, band_start:band_stop]
	return jnp.asarray(sub, dtype=jnp.complex128)



def summarize_hermitian_matrix(name: str, mats: np.ndarray, print_fn=print, warn_threshold: float = 1e-6):
	"""Emit diagnostics for a batch of Hermitian matrices shaped (nk, nb, nb)."""
	if mats.ndim != 3:
		print_fn(f"[{name}] unexpected shape {mats.shape}; expected (nk, nb, nb)")
		return
	herm_resid = np.max(np.abs(mats - np.conj(np.swapaxes(mats, -2, -1))))
	diag = np.diagonal(mats, axis1=-2, axis2=-1)
	diag_im = np.max(np.abs(np.imag(diag)))
	diag_min = float(np.min(np.real(diag)))
	diag_max = float(np.max(np.real(diag)))
	print_fn(f"[{name}] hermitian residual={herm_resid:.3e} max|Im diag|={diag_im:.3e} diag range=[{diag_min:.4f}, {diag_max:.4f}]")
	if name.lower().startswith("hartree") and diag_min < -warn_threshold:
		print_fn(f"[{name}] warning: min diagonal {diag_min:.4f} < 0.0 (negativity may signal improper G=0 handling)")

def preprocess_q_loops(wfn, sym, meta, mesh_xy=None):
	"""
	Compatibility helper that materializes the legacy q-point cache.

	Prefer :func:`iter_qpoint_data` plus on-demand evaluation. This function
	exists so downstream code that still expects the old tuple-of-arrays
	representation keeps working.
	"""
	print("Precomputing q-point mappings (compatibility path)...")
	q_entries = list(iter_qpoint_data(sym, meta))
	if not q_entries:
		return (
			jnp.zeros((0, 0), dtype=jnp.int32),
			jnp.zeros((0, 0), dtype=jnp.int32),
			jnp.zeros((0, 3), dtype=jnp.int32),
			jnp.zeros((0, 3), dtype=jnp.int32),
			jnp.zeros((0,), dtype=jnp.int32),
		)

	all_k_l_indices = jnp.stack([entry.k_l_indices for entry in q_entries])
	k_r_indices = q_entries[0].k_r_indices
	all_k_r_indices = jnp.repeat(k_r_indices[None, :], all_k_l_indices.shape[0], axis=0)
	all_qvecs_wrapped = jnp.stack([entry.q_wrapped for entry in q_entries])
	all_qvecs_nonneg = jnp.asarray([entry.q_nonneg for entry in q_entries], dtype=jnp.int32)
	all_iq_indices = jnp.asarray([entry.q_index for entry in q_entries], dtype=jnp.int32)
	print(f"Precomputed q/k mappings for {len(q_entries)} q-points")
	return (all_k_l_indices, all_k_r_indices, all_qvecs_wrapped, all_qvecs_nonneg, all_iq_indices)




def main(argv=None):	
	global sym
	argp = argparse.ArgumentParser(description="COHSEX self-energy driver")
	argp.add_argument(
		"-i",
		"--input",
		default="cohsex_test.in",
		help="Input file",
	)
	args = argp.parse_args(argv)
	
	# Gate prints to rank 0 during main
	_orig_print = builtins.print
	def _print0(*a, **k):
		if jax.process_index() == 0:
			_orig_print(*a, **k)
	builtins.print = _print0
 
	params = read_cohsex_input(args.input)
	current_backend = jax.default_backend()
	_print0(f"JAX backend in use: {current_backend}")
	_print0(jax.devices())
	# Resolve relative paths against the input file's directory
	input_dir = os.path.dirname(os.path.abspath(args.input))
	def _resolve_path(path: str) -> str:
		return path if os.path.isabs(path) else os.path.join(input_dir, path)
	params["wfn_file"] = _resolve_path(params["wfn_file"])
	params["centroids_file"] = _resolve_path(params["centroids_file"])
	params["output_file"] = _resolve_path(params["output_file"])
	params["kin_ion_file"] = _resolve_path(params["kin_ion_file"])
	params["eqp_output_file"] = _resolve_path(params["eqp_output_file"])
	nval = params["nval"]
	ncond = params["ncond"]
	nband = params["nband"]
	sys_dim = params["sys_dim"]  # 3 for 3D, 2 for 2D
	self_consistent = bool(params.get("self_consistent", False))

	ryd2ev = 13.6056980659

	global wfn
	wfn = WFNReader(params["wfn_file"])
	sym = symmetry_maps.SymMaps(wfn)
	
	# Load centroids
	centroids_frac = np.loadtxt(params["centroids_file"])
	_n_rmu = int(centroids_frac.shape[0])
	# Resolve tmp_dir and output path relative to input file directory
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	taggedarray_filename = os.path.join(tmp_dir, f"taggedarrays{_n_rmu}.h5")

	# Build centroid indices on host (NumPy), handle periodic boundary in-place, then promote to JAX later
	centroid_indices = np.round(centroids_frac * wfn.fft_grid).astype(int)
	for i in range(3):
		centroid_indices[centroid_indices[:, i] == wfn.fft_grid[i], i] = 0
	print("unique centroid indices:")
	print(np.unique(centroid_indices, axis=0).shape)
	print(f"fft grid: {wfn.fft_grid}, celvol: {wfn.cell_volume}")

	# windows for polarizability and sigma
	# Get window information
	epsq = 0.01
	#window_pairs = get_window_info(epsq, wfn)

	# ============================================================================
	# MAIN CONTROL FLAGS (from input file):
	# ============================================================================
	# restart:     If True, load ISDF vectors and V_qmunu from taggedarrays.h5
	#              instead of recomputing. Saves ~30 min for large systems.
	# x_only:      If True, use bare Coulomb V for exchange (no screening).
	#              Mutually exclusive with do_screened=True.
	# do_screened: If True, build screened Coulomb W = (1-Vχ)⁻¹V and use for Σ_x.
	#              If False, Σ_x uses bare V. Default=True for full GW/COHSEX.
	# bispinor:    If True, use 2-component spinor wavefunctions (for SOC).
	#              If False, scalar or collinear spin. Default=False.
	# ============================================================================
	restart = params["restart"]       # True: load from h5, False: rebuild ISDF
	x_only = params["x_only"]         # True: bare exchange only, no screening
	do_screened = params["do_screened"]  # True: use W, False: use V
	global bispinor
	bispinor = params["bispinor"]     # True: 2-component spinors (SOC)
	
	# ============================================================================
	# INTERNAL FEATURE TOGGLES (not exposed in input file):
	# ============================================================================
	# do_G0: If True, apply head corrections for q→0, G=0 divergence.
	#        This adds the cell-averaged 8π/q² to V_μν and W_μν via outer product
	#        of ζ(G=0) vectors. Essential for correct Hartree and long-range physics.
	# do_S:  If True, store the overlap matrix S_q = ⟨ζ_μ|ζ_ν⟩ (not currently used).
	# ============================================================================
	do_G0 = True   # Always True: head corrections are essential
	do_S = False   # Overlap matrix storage (disabled, not needed)
	if x_only and do_screened:  # x_only=bare exchange only, do_screened=use W instead of v
		raise ValueError("x_only and do_screened cannot both be True")

	meta = Meta.from_system(wfn, sym, nval, ncond, nband, _n_rmu, bispinor)
	meta.rank = jax.process_index()
	meta.n_proc = jax.process_count()
	meta.sys_dim = sys_dim
	meta.bispinor = bispinor
	fft_nx, fft_ny, fft_nz = (int(dim) for dim in meta.fft_grid)
	nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
	band = meta.band_ranges
	# ============================================================================
	# BAND EDGE DEFINITIONS (0-indexed):
	# ============================================================================
	#   b0 = 0                      : first band (absolute minimum)
	#   b1 = nelec - nval           : first "active" valence band (lowest in sigma window)
	#   b2 = nelec                  : VBM+1 = CBM = first conduction band
	#   b3 = nelec + ncond          : last "active" conduction band + 1 (end of sigma window)
	#   b4 = nband                  : highest band read (for chi0 sums, etc.)
	#
	# EXAMPLE: For nelec=26, nval=26, ncond=14, nband=80:
	#   b0=0, b1=0, b2=26, b3=40, b4=80
	#   Valence bands: 0..25 (indices b1..b2-1, all 26 included if nval=nelec)
	#   Sigma window:  0..39 (indices b1..b3-1, includes nval + ncond = 40 bands)
	#   Conduction:    26..39 (indices b2..b3-1)
	#
	# BAND RANGES (tuples for slicing):
	#   valence = (b1, b2)           : active valence for sigma (may exclude deep core)
	#   conduction = (b2, b3)        : active conduction for sigma
	#   sigma = (b1, b3)             : full sigma window (valence + conduction)
	#   occupied = (b0, b2)          : ALL occupied bands (for density, including core)
	#   val_plus_sigma = (b0, b3)    : bands 0..b3-1 (for left wfns in ISDF)
	#   cond_plus_sigma = (b1, b4)   : bands b1..b4-1 (for right wfns if screened)
	# ============================================================================
	b0, b1, b2, b3, b4 = meta.band_edges
	nvplussigrange = band.val_plus_sigma   # (b0, b3) = (0, nelec+ncond) for left wfns
	ncplussigrange = band.cond_plus_sigma  # (b1, b4) for right wfns when screened
	# rank-aware print helper
	def print0(*a, **k):
		if meta.rank == 0:
			print(*a, **k)
	print0(jax.devices())
	print0(jax.process_indices())
	shard_debug = True
	# Default mesh when not constructed (e.g., restart path)
	mesh_xy = None
	sh = None


	# Initialize timing system
	timing.reset()
	if not restart:
		with timing.section("cohsex_jax.wavefunction_setup") as timer_setup:
			####################################
			# 0.) Build a 2D mesh ['x','y'] up-front and materialize global G-space arrays
			####################################
			total_devices = jax.process_count() * jax.local_device_count()
			grid_x = int(np.sqrt(total_devices))
			while total_devices % grid_x != 0:
				grid_x -= 1
			grid_y = total_devices // grid_x
			devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
			mesh_xy = Mesh(devices_2d, ['x', 'y'])
			sh = make_shardings(mesh_xy)
			print0(f"Device mesh: (X={grid_x}, Y={grid_y})")

			# Left window: read G-vectors -> global G-space; then jitted sharded wfn build
			sigma_window = nvplussigrange
			brange_l = sigma_window
			global_psiG_l, nb_l = read_Gvecs_to_devices(wfn, sym, brange_l, meta, bispinor, mesh_xy)
			psi_l_rtot_Y, psi_l_rmu_Y, psi_l_rmuT_X = get_sharded_wfns(
				global_psiG_l, sym, meta, centroid_indices, nb_l, False, mesh_xy
			)
			# Ensure kernels finish then free the large G-space buffer to reduce peak memory
			psi_l_rtot_Y.block_until_ready(); psi_l_rmu_Y.block_until_ready(); psi_l_rmuT_X.block_until_ready()
			del global_psiG_l
			gc.collect()

			# Right window (nv+sigma or sigma-only)
			brange_r = sigma_window if x_only else ncplussigrange
			global_psiG_r, nb_r = read_Gvecs_to_devices(wfn, sym, brange_r, meta, bispinor, mesh_xy)
			psi_r_rtot_Y, psi_r_rmu_Y, psi_r_rmuT_X = get_sharded_wfns(
				global_psiG_r, sym, meta, centroid_indices, nb_r, False, mesh_xy
			)
			# Free the right G-space buffer as well
			psi_r_rtot_Y.block_until_ready(); psi_r_rmu_Y.block_until_ready(); psi_r_rmuT_X.block_until_ready()
			del global_psiG_r
			gc.collect()
			print0('wavefunction sharding complete')

		with timing.section("cohsex_jax.zeta_V_build") as timer_zeta:
			####################################
			# 2.) Explicit q-loop: build zeta_q,mu(r), S_q, and V_q,mu,nu
			####################################
			# Energies and weights for windows (kept as before, last entry is just sigma range)
			enk_l, weights_l = get_enk_bandrange(wfn, sym, brange_l, (b1,b3))
			enk_r, weights_r = get_enk_bandrange(wfn, sym, brange_r, (b1,b3))

			# Allocate sharded outputs
			V_qmunu = jnp.zeros((1, meta.npol, meta.npol, meta.nkx, meta.nky, meta.nkz, meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
			if sh is not None:
				V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, sh.x6y7_8)
			if do_S:
				S_qmunu = jnp.zeros((meta.nkx, meta.nky, meta.nkz, meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
				if sh is not None:
					S_qmunu = jax.lax.with_sharding_constraint(S_qmunu, sh.x3y4_5)
			else:
				S_qmunu = None
			v_q0_noG0_munu = jnp.zeros((meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
			if sh is not None:
				v_q0_noG0_munu = jax.lax.with_sharding_constraint(v_q0_noG0_munu, sh.xy_shard)
			G0_mu_nu = None
			# Reusable buffers for the expensive CCT/ZCT accumulation.
			CCT_buf = jnp.zeros((meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
			ZCT_buf = jnp.zeros((meta.n_rmu, psi_l_rtot_Y.shape[-1]), dtype=jnp.complex128)
			if sh is not None:
				CCT_buf = jax.lax.with_sharding_constraint(CCT_buf, sh.replicated_2)
				ZCT_buf = jax.lax.with_sharding_constraint(ZCT_buf, sh.xy_shard)

			# No local kernel definitions; use module-scope jitted helpers

			#################################################################################
			# Main q-point loop
			# zeta_q(r) is ephemeral inside the loop
			################################################################################
			bvec_cart = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
			v_munu_kernel = make_v_munu_kernel(
				fft_nx,
				fft_ny,
				fft_nz,
				nkx,
				nky,
				nkz,
				bvec_cart,
				float(wfn.cell_volume),
				sys_dim,
			)
			q_cache = build_q_coulomb_cache(wfn, sym, meta, do_Dmunu=bispinor, sys_dim=sys_dim, mesh_xy=mesh_xy)

			for q_idx in range(q_cache.num_q):
				k_l_indices = q_cache.k_l_indices[q_idx]
				k_r_indices = q_cache.k_r_indices
				qvec_nonneg = q_cache.q_nonneg[q_idx]
				iq_cpu = np.asarray(q_cache.iq_indices[q_idx]).item()
				qvec = jnp.asarray(q_cache.q_wrapped[q_idx], dtype=jnp.float64)

				CCT, ZCT = compute_CCT_ZCT_for_q(
					CCT_buf,
					ZCT_buf,
					psi_l_rmu_Y,
					psi_r_rmu_Y,
					psi_l_rtot_Y,
					psi_r_rtot_Y,
					psi_l_rmuT_X,
					psi_r_rmuT_X,
					k_l_indices,
					k_r_indices,
				)
				CCT_buf, ZCT_buf = CCT, ZCT

				zeta_q = solve_zeta_cholesky(CCT, ZCT)
				zeta_q = set_zeta_sharding(zeta_q, mesh_xy)

				qx, qy, qz = as_index_tuple(qvec_nonneg)
				if do_S:
					S_q_local = compute_Sq_from_zeta(zeta_q)
					if sh is not None:
						S_q_local = jax.lax.with_sharding_constraint(S_q_local, sh.x0y1_2)
					S_qmunu = S_qmunu.at[qx, qy, qz, :, :].set(S_q_local)

				v_munu, g0_mu = v_munu_kernel(zeta_q, qvec)
				V_qmunu = V_qmunu.at[0, 0, 0, qx, qy, qz, :, :].set(v_munu)

				if do_G0 and np.all(np.asarray(qvec_nonneg) == 0):
					G0_mu_nu = g0_mu

			print0(f"finished building zeta/V for {q_cache.num_q} q-points")

			# if do_G0 and G0_mu_nu is None:
			# 	q_nonneg_np = np.asarray(q_cache.q_nonneg)
			# 	q0_candidates = np.where(np.all(q_nonneg_np == 0, axis=1))[0]
			# 	if q0_candidates.size == 0:
			# 		raise ValueError('q=0 not found in Coulomb cache')
			# 	q0_idx = np.asarray(q0_candidates[0]).item()
			# 	k_l_q0 = q_cache.k_l_indices[q0_idx]
			# 	k_r_q0 = q_cache.k_r_indices
			# 	qvec_q0 = jnp.asarray(q_cache.q_wrapped[q0_idx], dtype=jnp.float64)
			# 	CCT0, ZCT0 = compute_CCT_ZCT_for_q(
			# 		CCT_buf,
			# 		ZCT_buf,
			# 		psi_l_rmu_Y,
			# 		psi_r_rmu_Y,
			# 		psi_l_rtot_Y,
			# 		psi_r_rtot_Y,
			# 		psi_l_rmuT_X,
			# 		psi_r_rmuT_X,
			# 		k_l_q0,
			# 		k_r_q0,
			# 	)
			# 	zeta_q0 = solve_zeta_cholesky(CCT0, ZCT0)
			# 	zeta_q0 = set_zeta_sharding(zeta_q0, mesh_xy)
			# 	_, G0_mu_nu = v_munu_kernel(zeta_q0, qvec_q0)

			# ============================================================================
			# V0_noG0 EXTRACTION: This is the q=0 Coulomb matrix WITH G=0 EXCLUDED.
			# ============================================================================
			# V_qmunu is indexed as [qx,qy,qz,qx',qy',qz',μ,ν] where q' handles folding.
			# At q=0, we extract V_qmunu[0,0,0,0,0,0,:,:] which is the μ×μ matrix
			# built with the G=0 term ZEROED in make_v_munu_kernel (see denom_zero mask).
			#
			# This V0_noG0 is used for HARTREE to avoid the 1/q² divergence.
			# The missing G=0 piece is NOT added back for Hartree (it's a constant
			# energy shift that cancels between electron-electron and electron-ion).
			# ============================================================================
			v_q0_noG0_munu = v_q0_noG0_munu.at[:,:].set(V_qmunu[0, 0, 0, 0, 0, 0])
			if not do_G0:  # do_G0=True normally; G0_mu_nu = ζ_μ(G=0) for head corrections
				G0_mu_nu = None

			# ============================================================================
			# NOTE ON ISDF HARTREE AND THE "HEAD-LIKE" STRUCTURE
			# ============================================================================
			# The ISDF Coulomb matrix V_{μν} = (1/Ω) Σ_{G≠0} v(G) ζ*_μ(G) ζ_ν(G) has a
			# large projection (~99%) onto the "head" direction ζ*_μ(0) ⊗ ζ_ν(0), even
			# with G=0 zeroed. This is NOT contamination - it's legitimate physics:
			#
			#   1. ISDF vectors ζ_μ(r) are smooth, so ζ(G) is concentrated at low G
			#   2. The Coulomb v(G) ∝ 1/G² is also peaked at low G
			#   3. So Σ_{G≠0} v(G) |ζ(G)|² naturally resembles |ζ(0)|² structure
			#
			# The G→ζ mapping is LOW-RANK and NOT INVERTIBLE. You cannot cleanly
			# separate "G=0 contribution" from "G≠0 contribution" in the μν basis
			# because the ζ basis mixes them together.
			#
			# CURRENT STATUS: There is a ~17 Ry constant offset in Hartree vs QE.
			# The exchange Σ_x using the same V_μν is correct, so the issue is
			# specific to the Hartree formula, not the V_μν construction.
			#
			# POSSIBLE CAUSES UNDER INVESTIGATION:
			#   - Missing overlap matrix S_μν in the Hartree contraction
			#   - Different integration measure for Hartree vs exchange
			#   - Gauge/reference energy difference with QE
			# ============================================================================

			# Sharded psi variants outside q-loop
			with mesh_xy:
				psil_Y = jax.device_put(psi_l_rmu_Y, sh.y3_4)
				psilT_X = jax.lax.with_sharding_constraint(psil_Y.transpose(0, 2, 3, 1), sh.x2_4)
				psir_X = jax.device_put(psi_r_rmu_Y, sh.x3_4)
				psirT_Y = jax.lax.with_sharding_constraint(psir_X.transpose(0, 2, 3, 1), sh.y2_4)

			# ============================================================================
			# WAVEFUNCTION ARRAYS AND BAND SLICING:
			# ============================================================================
			# psi_l: "left" wavefunctions for building density ρ and Green's function G
			#        Originally loaded as bands [b0, b3) = [0, nelec+ncond)
			#        Sliced to VALENCE ONLY [b0, b2) = [0, nelec) for ρ and G
			#
			# psi_r: "right" wavefunctions for projecting Σ to band basis
			#        Loaded as bands [b1, b4) for do_screened or [b0, b3) otherwise
			#        Used as-is for sigma window projection
			#
			# KEY: ρ_μ and G_μν use VALENCE (psi_l sliced), but <m|Σ|n> uses SIGMA
			#      WINDOW (psi_r), which includes both valence and conduction.
			# ============================================================================
			psi_l_full = psil_Y        # Full array: bands [b0, b3)
			psi_lT_full = psilT_X      # Transposed version
			
			# Slices for Hartree/exchange: valence only [b0, b2) = [0, nelec)
			valence_slice = slice(b0, b2)  # Indices 0..nelec-1
			nb_valence = int(b2 - b0)      # = nelec (number of valence bands)
			nb_sigma = int(b3 - b0)        # = nelec + ncond (sigma window size)
			
			# psi_l used for ρ and G: currently using full (should be valence_slice)
			# TODO: verify this is correct - commented slicing suggests debugging
			psi_l = psi_l_full   # [:, valence_slice, :, :] for valence only
			psi_lT = psi_lT_full # [:, :, :, valence_slice] for valence only
			# psi_r used for band projection: sigma window [b1, b3)
			psi_r = psir_X       # [:, :nb_sigma, :, :]
			psi_rT = psirT_Y     # [:, :, :, :nb_sigma]

			# Persist restart artifacts (store full-sized left/right; trim on load/use)
			write_labeled_arrays_to_h5(
				taggedarray_filename,
				V_qmunu,
				psi_l_full,
				psir_X,
				enk_l,
				enk_r,
				S_qmunu,
				V0_noG0_munu=v_q0_noG0_munu,
				G0_mu_nu=G0_mu_nu,
			)
			save_restart_per_proc(os.path.join(tmp_dir, "taggedarrays"), V_qmunu, S_qmunu, psi_l_full, psir_X, enk_l, enk_r, meta, mesh_xy, V0_noG0_munu=v_q0_noG0_munu)
			V_qmunu.block_until_ready()

	elif restart and not x_only: # TODO update for jax
		with timing.section("cohsex_jax.restart_load") as restart_timer:
			# Build mesh for sharding in restart path if missing
			if mesh_xy is None:
				total_devices = jax.process_count() * jax.local_device_count()
				grid_x = int(np.sqrt(total_devices))
				while total_devices % grid_x != 0:
					grid_x -= 1
				grid_y = total_devices // grid_x
				devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
				mesh_xy = Mesh(devices_2d, ['x', 'y'])
				print(f"Device mesh: (X={grid_x}, Y={grid_y}) [restart]")
			sh = make_shardings(mesh_xy)
			V_qmunu, S_qmunu, psi_lT, psi_l, psi_r, psi_rT, enk_l, enk_r, v_q0_noG0_munu, G0_mu_nu = load_labeled_arrays_from_h5(
				taggedarray_filename, mesh_xy
			)
			V_mu_nu = jnp.asarray(V_qmunu)[0, 0, 0]
	elif restart and x_only:
		with timing.section("cohsex_jax.restart_load_x_only") as restart_timer:
			# Same restart flow for X-only
			if mesh_xy is None:
				total_devices = jax.process_count() * jax.local_device_count()
				grid_x = int(np.sqrt(total_devices))
				while total_devices % grid_x != 0:
					grid_x -= 1
				grid_y = total_devices // grid_x
				devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
				mesh_xy = Mesh(devices_2d, ['x', 'y'])
				print(f"Device mesh: (X={grid_x}, Y={grid_y}) [restart]")
			sh = make_shardings(mesh_xy)
			V_qmunu, S_qmunu, psi_lT, psi_l, psi_r, psi_rT, enk_l, enk_r, v_q0_noG0_munu, G0_mu_nu = load_labeled_arrays_from_h5(
				taggedarray_filename, mesh_xy
			)
			V_mu_nu = jnp.asarray(V_qmunu)[0, 0, 0]

	# Optionally compute chi0 via JAX for screened interaction if requested
	if do_screened:
		with timing.section("cohsex_jax.chi0_W") as timer_chiw:
			# Ensure energies are plain arrays for JAX
			enk_v_arr = jnp.asarray(getattr(enk_l, 'data', enk_l))
			enk_c_arr = jnp.asarray(getattr(enk_r, 'data', enk_r))
			# Four wavefunction copies and shardings for low-comm G and Sigma construction:
			# psi_lT: (nk, ns, rmu, nb) XT_shard; psi_l: (nk, nb, ns, rmu) Y_shard
			# psi_r:  (nk, nb, ns, rmu) X_shard;  psi_rT: (nk, ns, rmu, nb) YT_shard
			# We reshard psi_rT to XT for chi0 so both G_v and G_c use {mu_X, nu_Y} without communication.
			window_pairs = get_window_info(epsq, wfn, nband_max=nband)
			chi0 = get_chi0_jax(psi_lT, psi_l, psi_r, psi_rT, enk_v_arr, enk_c_arr, window_pairs, meta, mesh_xy)
			# Compute static W under k_XY sharding (S_qmunu included but unused for now)
			W_q = get_static_w_q_jax(V_qmunu, chi0, None, meta, mesh_xy)
			W_q.block_until_ready()
			# Compute static W under k_XY sharding (S_qmunu included but unused for now)
			#W_q = get_static_w_q_jax(V_qmunu, chi0, S_qmunu, meta, mesh_xy)

		# ============================================================================
		# HEAD INJECTION: Add the q→0, G=0 Coulomb divergence correction
		# ============================================================================
		# At q=0, the Coulomb interaction v(q,G) = 4π/|q+G|² diverges as G→0.
		# We handle this by:
		#   1. BUILDING V_μν with G=0 ZEROED (done in make_v_munu_kernel)
		#   2. ADDING BACK the cell-averaged head: ⟨v(q→0,G=0)⟩ = vc0_mean
		#
		# The head is added as a rank-1 correction in the μν basis:
		#   V_μν ← V_μν + (vc0_mean / Ω) × ζ*_μ(0) ⊗ ζ_ν(0)
		#
		# where ζ_μ(0) = G0_mu_nu is the G=0 component of the ISDF vectors.
		#
		# IMPORTANT: This head correction is added to V_qmunu (for Σ_x) but NOT
		# to v_q0_noG0_munu (for Hartree). Hartree uses V0 with G=0 excluded because
		# the divergent piece cancels with the electron-ion interaction.
		# ============================================================================
		if do_G0:  # do_G0=True: apply head corrections
			vc0_mean, wcoul0, wcoul0_source = determine_wcoul0(params, input_dir, wfn, sym, meta, print0)
			print0(f"wcoul0 source: {wcoul0_source}")
			print0(f"wcoul0 value (atomic units): {wcoul0}")
			
			# outer_u = ζ*_μ(0) ⊗ ζ_ν(0), the "head" direction in μν space
			outer_u = (G0_mu_nu[:, None] * jnp.conj(G0_mu_nu)[None, :])
			# V_μν has units of (energy × volume), so divide by Ω
			vol_scale = jnp.asarray(1.0 / float(wfn.cell_volume), dtype=jnp.float64)
			
			# HEAD INJECTION FOR Σ_x: V_qmunu at q=0 gets the bare Coulomb head
			# V_Γ ← V_Γ + (vc0_mean / Ω) × ζ*(0) ⊗ ζ(0)
			V_qmunu = V_qmunu.at[0, 0, 0, 0, 0, 0, :, :].add((vc0_mean * vol_scale) * outer_u)
			
			# HEAD INJECTION FOR W (screened): W at q=0 gets the screened head wcoul0
			if do_screened:  # do_screened=True: using W instead of V for exchange
				W_q = W_q.at[0, 0, 0, 0, :, 0, :].add((wcoul0 * vol_scale) * outer_u)

	# ============================================================================
	# SELECT INTERACTION FOR Σ_x: bare V or screened W
	# ============================================================================
	# V_qmunu: bare Coulomb, shape [nfreq, npol1, npol2, nkx, nky, nkz, nrmu1, nrmu2]
	# W_q:     screened Coulomb, shape [nkx, nky, nkz, nfreq, nrmu, npol, nrmu]
	# We extract the static (freq=0) component and reshape to (nrmu1, nrmu2, nkx, nky, nkz)
	# ============================================================================
	V_mu_nu = jnp.asarray(V_qmunu)[0, 0, 0]  # (nkx,nky,nkz,nrmu1,nrmu2) from bare V
	if do_screened:  # do_screened=True: use W instead of V for exchange
		V_mu_nu = W_q[:,:,:,0,:,0,:]  # (nkx,nky,nkz,nrmu1,nrmu2) from screened W
	V_mu_nu = V_mu_nu.transpose(3, 4, 0, 1, 2)  # → (nrmu1,nrmu2,nkx,nky,nkz)

	# After W is computed, the trimmed views (psi_l, psi_lT, psi_r, psi_rT)
	# are already defined; preserve psi_l_full/psi_lT_full as b3-sized copies.
	valence_slice = slice(b0, b2)
	nb_sigma = int(b3 - b0)
	psi_l = psi_l[:, valence_slice, :, :]
	psi_lT = psi_lT[:, :, :, valence_slice]
	psi_r = psi_r[:, :nb_sigma, :, :]
	psi_rT = psi_rT[:, :, :, :nb_sigma]

	if sh is None and mesh_xy is not None:
		sh = make_shardings(mesh_xy)

	# Jitted pipeline with explicit shardings along rmu/rnu XY
	pipeline_jit = jax.jit(
		compute_sigma_pipeline_jax,
		static_argnames=('nkx', 'nky', 'nkz', 'nk_tot', 'nspinor', 'fft_vol_au'),
		in_shardings=(sh.XT_shard, sh.Y_shard, sh.X_shard, sh.YT_shard, sh.V_shard, sh.xy_shard),
		out_shardings=(sh.out_shard, sh.out_shard),
	)

	# ========== HARTREE DIAGNOSTIC ==========
	# Prints detailed info about V0, density, and Hartree matrix elements
	if params.get("debug_hartree", False):
		print0("\n" + "="*70)
		print0("HARTREE DIAGNOSTIC")
		print0("="*70)
		
		# Print band range info for context
		print0(f"\n  BAND RANGES (0-indexed):")
		print0(f"    b0={b0}, b1={b1}, b2={b2} (VBM+1=CBM), b3={b3}, b4={b4}")
		print0(f"    Valence in sigma:    bands {b1}..{b2-1} ({b2-b1} bands)")
		print0(f"    Conduction in sigma: bands {b2}..{b3-1} ({b3-b2} bands)")
		print0(f"    Total sigma window:  bands {b1}..{b3-1} ({b3-b1} bands)")
		print0(f"    N_electrons = {int(wfn.nelec)}")
		
		V0_np = np.asarray(v_q0_noG0_munu)
		V_q0_from_Vqmunu = np.asarray(V_qmunu[0, 0, 0, 0, 0, 0])
		print0(f"\n  V0 MATRIX (q=0, G=0 excluded):")
		print0(f"    ||V0_noG0|| = {np.linalg.norm(V0_np):.4f}")
		print0(f"    ||V_qmunu[q=0]|| = {np.linalg.norm(V_q0_from_Vqmunu):.4f}")
		print0(f"    ||V0 - Vq0|| = {np.linalg.norm(V0_np - V_q0_from_Vqmunu):.4e}")
		if do_G0 and G0_mu_nu is not None:
			g0 = np.asarray(G0_mu_nu)
			head_outer = np.outer(g0.conj(), g0)
			print0(f"    ||G0_mu_nu|| = {np.linalg.norm(g0):.4f}")
			print0(f"    ||head_outer|| = {np.linalg.norm(head_outer):.4f}")
			proj = np.abs(np.vdot(V0_np, head_outer)) / (np.linalg.norm(V0_np) * np.linalg.norm(head_outer) + 1e-12)
			print0(f"    V0 projection onto head: {proj:.6f}")
			if do_screened:
				print0(f"    vc0_mean = {float(vc0_mean.real):.4f}")
				head_mag = float(vc0_mean.real) / float(wfn.cell_volume) * np.linalg.norm(head_outer)
				print0(f"    Expected head contribution: {head_mag:.4f}")
		
		# Density diagnostic
		psi_l_np = np.asarray(psi_l)  # psi_l is sliced to valence: bands b0..b2-1
		n_bands_in_density = psi_l_np.shape[1]
		rho_test = np.einsum('knsx,knsx->x', np.conj(psi_l_np), psi_l_np).real / meta.nk_tot
		print0(f"\n  DENSITY (from psi_l, valence bands for Hartree):")
		print0(f"    Bands in density sum: {n_bands_in_density} (should be {b2-b0} = all occupied)")
		print0(f"    Sum(ρ_μ) = {np.sum(rho_test):.4f}")
		print0(f"    Mean(ρ_μ) = {np.mean(rho_test):.6f}")
		if do_G0 and G0_mu_nu is not None:
			rho_integral_proxy = np.sum(rho_test * np.abs(g0)**2)
			print0(f"    Σ_μ ρ_μ × |ζ_μ(0)|² = {rho_integral_proxy:.4f}")
		
		# Hartree potential at centroids
		Vrho_test = V0_np @ rho_test
		print0(f"\n  [V0 @ ρ] (Hartree potential at centroids):")
		print0(f"    Min = {Vrho_test.real.min():.4f}, Max = {Vrho_test.real.max():.4f}")
		print0(f"    # negative values: {np.sum(Vrho_test.real < 0)} / {len(Vrho_test)}")
		
		# Hartree matrix elements for valence and conduction separately
		print0(f"\n  V_H DIAGONAL (k=0):")
		psi_k0 = psi_l_np[0]  # This is psi_l at k=0, contains valence bands only
		n_val_bands = min(psi_k0.shape[0], b2 - b0)  # Number of valence bands
		
		# Valence bands (from psi_l)
		print0(f"    VALENCE (bands {b0}..{b0 + min(5, n_val_bands) - 1}):")
		for n in range(min(5, n_val_bands)):
			overlap = np.sum(np.abs(psi_k0[n])**2, axis=0)
			V_H_n = np.sum(overlap * Vrho_test).real
			print0(f"      band {b0+n}: {V_H_n:.4f} Ry")
		
		# Check if we have sigma bands (psi_r) to show conduction
		psi_r_np = np.asarray(psi_r)  # psi_r contains sigma window bands
		n_sigma_bands = psi_r_np.shape[1]
		print0(f"\n  PSI ARRAY SHAPES:")
		print0(f"    psi_l shape: {psi_l_np.shape} (bands in density)")
		print0(f"    psi_r shape: {psi_r_np.shape} (bands for projection)")
		print0(f"    n_sigma_bands in psi_r: {n_sigma_bands}")
		print0(f"    Sigma window should be: {b3-b1} bands (b1={b1} to b3={b3})")
		
		if n_sigma_bands > n_val_bands:
			print0(f"    CONDUCTION (first 5 cond bands in sigma window):")
			psi_r_k0 = psi_r_np[0]
			# psi_r starts at band b1, so conduction starts at index (b2-b1)
			cond_start_idx = b2 - b1 if b1 <= b2 else 0
			for i in range(min(5, n_sigma_bands - cond_start_idx)):
				n_idx = cond_start_idx + i
				if n_idx < n_sigma_bands:
					overlap = np.sum(np.abs(psi_r_k0[n_idx])**2, axis=0)
					V_H_n = np.sum(overlap * Vrho_test).real
					band_id = b1 + n_idx  # Actual band index
					in_sigma = b1 <= band_id < b3
					print0(f"      band {band_id}: {V_H_n:.4f} Ry {'(NEGATIVE!)' if V_H_n < 0 else ''} {'[IN SIGMA]' if in_sigma else '[OUTSIDE SIGMA]'}")
		
		# Scan ALL bands in psi_r and find which ones have negative Hartree
		print0(f"\n  NEGATIVE HARTREE SCAN (all bands in psi_r):")
		psi_r_k0 = psi_r_np[0]
		neg_in_sigma = []
		neg_outside_sigma = []
		for n_idx in range(n_sigma_bands):
			overlap = np.sum(np.abs(psi_r_k0[n_idx])**2, axis=0)
			V_H_n = np.sum(overlap * Vrho_test).real
			band_id = b1 + n_idx
			if V_H_n < 0:
				in_sigma = b1 <= band_id < b3
				if in_sigma:
					neg_in_sigma.append((band_id, V_H_n))
				else:
					neg_outside_sigma.append((band_id, V_H_n))
		
		print0(f"    # negative IN sigma window [b1={b1}, b3={b3}): {len(neg_in_sigma)}")
		for bid, vh in neg_in_sigma[:5]:
			print0(f"      band {bid}: {vh:.4f} Ry")
		if len(neg_in_sigma) > 5:
			print0(f"      ... and {len(neg_in_sigma)-5} more")
		
		print0(f"    # negative OUTSIDE sigma window (bands >= b3={b3}): {len(neg_outside_sigma)}")
		for bid, vh in neg_outside_sigma[:5]:
			print0(f"      band {bid}: {vh:.4f} Ry")
		if len(neg_outside_sigma) > 5:
			print0(f"      ... and {len(neg_outside_sigma)-5} more")
		
		print0("="*70 + "\n")
	# ========== END HARTREE DIAGNOSTIC ==========

	with mesh_xy:
			with timing.section("cohsex_jax.pipeline"):
				sigma_x_kbar_ij_jax, hartree_kbar_ij_jax = pipeline_jit(
					psi_lT,  # (nk, nb, ns, rmu) X-sharded over rmu (resharded by jit)
					psi_l,   # (nk, nb, ns, rmu) Y-sharded over rmu
					psi_r,   # (nk, ns, rmu, nb)  X-sharded over rmu (unused inside)
					psi_rT,  # (nk, nb, ns, rmu)  X-sharded over rmu (unused inside)
					V_mu_nu, # (rmu1, rmu2, nkx, nky, nkz) rmu1, rmu2 sharded over (x,y)
					v_q0_noG0_munu, # (rmu1, rmu2) sharded over (x,y)
					meta.nkx, meta.nky, meta.nkz, meta.nk_tot, meta.nspinor,
					float(wfn.cell_volume/np.prod(wfn.fft_grid)),
				)
				sigma_x_kbar_ij_jax.block_until_ready()
				hartree_kbar_ij_jax.block_until_ready()


	sigma_total_full = sigma_x_kbar_ij_jax #+ hartree_kbar_ij_jax
	sigma_full_shape = sigma_total_full.shape

	qp_band_start = int(b0)
	qp_band_stop = int(b3)
	kin_ion_path = params["kin_ion_file"]
	kin_ion_full = load_kin_ion_submatrix(kin_ion_path, qp_band_start, qp_band_stop)

	
	if self_consistent:
		# Fixed reference gauge from initial Hamiltonian H0 = kin_ion + initial sigma_x
		H0 = kin_ion_full + sigma_total_full
		H0 = 0.5 * (H0 + jnp.conj(jnp.swapaxes(H0, -1, -2)))
		_E0, U_fix0 = jax.vmap(jnp.linalg.eigh, in_axes=0)(H0)

		def sigma_iteration_step(sigma_full: jax.Array):
			# Build current Hamiltonian and diagonalize
			H_full = kin_ion_full + sigma_full
			H_full = 0.5 * (H_full + jnp.conj(jnp.swapaxes(H_full, -1, -2)))
			# Batched eigh over k
			evals_full, U_full_raw = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_full)
			def align_unitaries(U_ref, U_full):
				# M = U_ref^† U_full = W S V^†  => polar unitary Q = W V^†
				def one_k(Ur, U):
					M = jnp.conj(Ur).swapaxes(-1, -2) @ U
					W, S, Vh = jnp.linalg.svd(M, full_matrices=False)
					Q = W @ Vh  # unitary closest to M in Frobenius norm
					return U @ jnp.conj(Q).swapaxes(-1, -2)  # rotate U toward reference
				return jax.vmap(one_k)(U_ref, U_full)
			# Align U to fixed gauge U_fix0
			U_full = align_unitaries(U_fix0, U_full_raw)
			# Rotate left wavefunctions explicitly: psi' = U psi, psi_T' = psi_T @ U^T
			# Ensure we run under mesh context for inner constraints
			with mesh_xy:
				psi_full_rot = jnp.einsum('kij, kjab->kiab', U_full, psi_l_full, optimize=True)
				psiT_full_rot = jnp.einsum('kij, kabj->kabi', U_full, psi_lT_full, optimize=True)
				psi_val_rot = psi_full_rot[:, valence_slice, :, :]
				psi_valT_rot = psiT_full_rot[:, :, :, valence_slice]

				sigma_x_val, hartree_val = pipeline_jit(
				psi_valT_rot,
				psi_val_rot,
				psi_r,
				psi_rT,
				V_mu_nu,
				v_q0_noG0_munu,
				meta.nkx, meta.nky, meta.nkz, meta.nk_tot, meta.nspinor,
				float(wfn.cell_volume/np.prod(wfn.fft_grid)),
				)
			sigma_full_updated = sigma_x_val #+ hartree_val
			#sigma_full_updated = sigma_full.at[:, sigma_slice, sigma_slice].set(sigma_total_block_local)
			return sigma_full_updated, (
				sigma_x_val,
				hartree_val,
				evals_full,
				U_full,
				H_full,
			)

		def residual_map(vec: jax.Array) -> jax.Array:
			sigma_full = jnp.reshape(vec, sigma_full_shape)
			sigma_next, _ = sigma_iteration_step(sigma_full)
			return jnp.reshape(sigma_next - sigma_full, (-1,))


		sigma_vec0 = jnp.reshape(sigma_total_full, (-1,))
		sigma_vec_final, residual_hist, n_iters = crop_family_fixed_history_map(
			residual_map,
			sigma_vec0,
			m=2,
			maxit=40,
			tol=1e-5,
			real_residual=True,
		)
		# Print residual history on rank 0
		if meta.rank == 0:
			_hist = np.array(residual_hist)
			_nit = int(n_iters)
			print0(f"CROP residual history (iters={_nit}):")
			for i in range(_nit + 1):
				print0(f"  it {i:02d}: {_hist[i]:.6e}")
		sigma_total_full = jnp.reshape(sigma_vec_final, sigma_full_shape)
		sigma_total_full, (sigma_x_kbar_ij_jax, hartree_kbar_ij_jax, E_full, U_full, H_qp_mnk) = sigma_iteration_step(sigma_total_full)
		#sigma_total_block = sigma_total_full[:, sigma_slice, sigma_slice]
		if meta.rank == 0:
			final_res = float(residual_hist[int(n_iters)])
			print0(f"Self-consistent GW completed in {int(n_iters)} iterations; final residual={final_res:.3e}")
	else:
		# One-shot diagonalization consistent with the in-loop path
		H_qp_mnk = kin_ion_full + sigma_total_full
		H_qp_mnk = 0.5 * (H_qp_mnk + jnp.conj(jnp.swapaxes(H_qp_mnk, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_qp_mnk)
		#E_full = 
		#sigma_total_block = sigma_total_full[:, sigma_slice, sigma_slice]

	#sigma_total_val = sigma_total_full[:, valence_slice, valence_slice]

	energies_full_dft, _ = get_enk_bandrange(
		wfn,
		sym,
		(qp_band_start, qp_band_stop),
		(qp_band_start, qp_band_stop),
	)
	energies_dft_ev = jnp.asarray(energies_full_dft) * ryd2ev
	energies_qp_ev = E_full * ryd2ev

	sigma_x_full = sigma_x_kbar_ij_jax
	hartree_full = sigma_total_full - sigma_x_full # this isn't even right, delete later

	# Host copies for downstream I/O (keep qp eigenpairs for forthcoming steps)
	sigma_x_kbar_ij = np.array(sigma_x_full)
	hartree_kbar_ij = np.array(hartree_kbar_ij_jax)
	H_qp_mnk_host = np.array(H_qp_mnk)
	E_qp_mnk_host = np.array(E_full)
	U_qp_mnk_host = np.array(U_full)
	energies_dft_ev_host = np.array(energies_dft_ev)
	energies_qp_ev_host = np.array(energies_qp_ev)

	write_sigma_to_file(ryd2ev * sigma_x_kbar_ij, params["output_file"], hartree_kij_Ry=hartree_kbar_ij)
	#write_eqp_table(energies_dft_ev_host, energies_qp_ev_host, params["eqp_output_file"])
	write_eqp_table(energies_dft_ev_host, np.diagonal(H_qp_mnk_host, axis1=-2, axis2=-1), params["eqp_output_file"])
	
	# ============================================================================
	# MATRIX SUMMARY: Report diagnostics for key matrices
	# ============================================================================
	# sigma_x:  exchange self-energy, shape (nk, nb_sigma, nb_sigma)
	# hartree:  Hartree matrix elements (may include bands outside sigma window
	#           if psi_r has more bands than sigma window - see psi_r shape in debug)
	# H_qp:     quasiparticle Hamiltonian after SCF/diagonalization
	# ============================================================================
	if meta.rank == 0:
		summarize_hermitian_matrix("sigma_x", sigma_x_kbar_ij, print_fn=print0)
		# Hartree might include bands outside sigma window; report both full and sliced
		nb_sigma_window = int(b3 - b1)  # Expected sigma window size
		nb_hartree = hartree_kbar_ij.shape[1]
		if nb_hartree > nb_sigma_window:
			print0(f"[hartree] NOTE: matrix has {nb_hartree} bands but sigma window is {nb_sigma_window}")
			hartree_sigma_only = hartree_kbar_ij[:, :nb_sigma_window, :nb_sigma_window]
			summarize_hermitian_matrix("hartree (full)", hartree_kbar_ij, print_fn=print0)
			summarize_hermitian_matrix("hartree (sigma)", hartree_sigma_only, print_fn=print0)
		else:
			summarize_hermitian_matrix("hartree", hartree_kbar_ij, print_fn=print0)
		summarize_hermitian_matrix("H_qp", H_qp_mnk_host, print_fn=print0)
	# Timing report
	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing (seconds) ---")

	# Later stages of this project will iterate this workflow so that the COHSEX
	# potential feeds back into updated wavefunctions (self-consistent COHSEX)
	# and eventually into a full quasiparticle self-consistent GW cycle.
	return 0


if __name__ == "__main__":
	#jax.distributed.initialize()
	raise SystemExit(main())

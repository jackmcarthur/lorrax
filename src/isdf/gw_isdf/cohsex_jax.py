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
		# Prefer auto-detection (NERSC pattern): JAX reads SLURM env vars directly
		try:
			jax.distributed.initialize()
			return
		except Exception:
			pass
		# Fallback: explicit coordinator from SLURM_NODELIST (first node)
		coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
		if coord is None:
			import subprocess
			nodelist = os.environ.get("SLURM_NODELIST")
			if nodelist:
				try:
					result = subprocess.run(
						["scontrol", "show", "hostnames", nodelist],
						capture_output=True, text=True, check=True
					)
					first_host = result.stdout.strip().split("\n")[0]
					coord = f"{first_host}:12355"
				except Exception:
					pass
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
from ..io import (
    WFNReader, EPSReader,
    write_sigma_to_file, write_eqp_table,
    write_labeled_arrays_to_h5, write_w0_qmunu_to_h5, read_labeled_arrays_from_h5,
    load_labeled_arrays_from_h5, save_restart_per_proc,
    write_qp_rotations_h5, load_kin_ion_submatrix,
    load_centroids, resolve_input_paths,
)
from ..common import symmetry_maps
from ..common.load_wfns import (
    read_Gvecs_to_devices, get_sharded_wfns, get_enk_bandrange,
    fit_zeta_chunked_to_h5,
)
from .compute_vcoul import compute_all_V_q_from_zeta_h5
from .cohsex_init import compute_optimal_chunks
from .get_windows import get_window_info
from .w_isdf import get_chi0_jax, get_static_w_q_jax, get_w_omega_jax
from .vcoul import compute_q0_averages
from ..common.chi_from_dipole import read_dipole_h5, compute_S_omega
from .cohsex_init import get_effective_chunk_size, read_cohsex_input, get_bandranges
from ..mixing.acceleration import (
    rcrop_nojit, hermitian_to_upper_flat, upper_flat_to_hermitian
)
from ..common import Meta
from ..common import jax_profile
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
			# Source printed in finite-size corrections section
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
			# Source printed in finite-size corrections section
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


# The current implementation focuses on the static COHSEX limit.  Many of the
# routines below (e.g. chi0 and sigma construction) are written in a style that
# follows the complex time shredded propagator (CTSP) formulation so that we can
# later restore full frequency dependence and iterate towards self-consistency.


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
	# Get band ranges and weights in JAX (pass nspinor for bispinor support)
	enk_l, weights_l = get_enk_bandrange(
		wfn, sym, bandrange_l, (bandrange_r[0], bandrange_l[1]), nspinor=meta.nspinor
	)
	enk_r, weights_r = get_enk_bandrange(
		wfn, sym, bandrange_r, (bandrange_r[0], bandrange_l[1]), nspinor=meta.nspinor
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


def fit_zeta_and_compute_V_q_chunked(
	wfn,
	sym,
	meta: Meta,
	centroid_indices: jax.Array,
	mesh_xy: Mesh,
	output_dir: str,
	bispinor: bool = False,
	memory_budget_gb: float = 6.0,
	sys_dim: int = 2,
	x_chunk_size_override: int = 0,
):
	"""
	Chunked zeta fitting and V_q computation pipeline.
	
	This replaces the per-q-point zeta fitting in the main loop with a memory-efficient
	chunked approach that:
	1. Loads wavefunctions for full band range (b0 to b4)
	2. Slices for left (b0→b3) and right (b0→b4) with spin-traced pair density
	3. Fits zeta via z-chunked algorithm and writes to HDF5
	4. Reads zeta back and computes V_qmunu
	
	Physics note:
		Uses spin-traced pair density P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(ν)
		matching cohsex_jax convention for ISDF fitting. Different from keeping all
		spin combinations which would increase lstsq error.
	
	Args:
		wfn: WFNReader object
		sym: SymMaps object
		meta: Meta object with system info
		centroid_indices: ISDF centroid indices
		mesh_xy: 2D device mesh
		output_dir: Directory for zeta HDF5 file
		bispinor: Whether to use bispinor wavefunctions
		memory_budget_gb: Memory budget per device in GB
		sys_dim: System dimensionality (2 or 3)
		x_chunk_size_override: If > 0, use this explicit x-chunk size instead of auto-compute
	
	Returns:
		Dictionary with:
		- V_qmunu: (1, npol, npol, nkx, nky, nkz, n_rmu, n_rmu) Coulomb matrix
		- v_q0_noG0_munu: (n_rmu, n_rmu) V_q at q=0 with G=0 excluded
		- G0_mu_nu: (n_rmu,) ζ_μ(G=0) for head corrections
		- psi_l_rmu_Y: Left centroid wfns, Y-sharded
		- psi_l_rmuT_X: Left conjugated wfns, X-sharded  
		- psi_r_rmu_Y: Right centroid wfns, Y-sharded
		- psi_r_rmuT_X: Right conjugated wfns, X-sharded
		- zeta_h5_path: Path to zeta HDF5 file
	"""
	import os
	
	n_devices = jax.device_count()
	p_x = mesh_xy.devices.shape[0]
	p_y = mesh_xy.devices.shape[1]
	
	# Band ranges for left and right (cohsex_jax convention)
	# Left: nvplussigrange = (b0, b3) = all valence + sigma-window conduction
	# Right: ncplussigrange = (b1, b4) = sigma-window valence + all conduction
	b0, b1, b2, b3, b4 = meta.band_edges
	band_range_left = (b0, b3)
	band_range_right = (b1, b4)
	n_b_full = b4 - b0
	
	# Compute optimal chunk sizes
	nqx, nqy, nqz = meta.kgrid
	n_q = nqx * nqy * nqz
	
	chunks = compute_optimal_chunks(
		n_k=meta.nk_tot,
		n_b=n_b_full,
		n_s=meta.nspinor,
		n_rmu=meta.n_rmu,
		n_r=meta.n_rtot,
		n_q=n_q,
		fft_grid=meta.fft_grid,
		n_devices=n_devices,
		memory_budget_gb=memory_budget_gb,
		target_utilization=0.85,
		p_x=p_x,
		p_y=p_y,
		n_b_left=band_range_left[1] - band_range_left[0],
		n_b_right=band_range_right[1] - band_range_right[0],
		verbose=True,
	)
	
	band_chunk_size = chunks['band_chunk']
	x_chunk_size = chunks['x_chunk']
	q_chunk_size = chunks['q_chunk']
	use_gspace_cache = chunks.get('use_gspace_cache', True)
	mem_est = chunks.get('memory_estimate', {})
	
	# Override x_chunk_size if explicitly specified
	if x_chunk_size_override > 0:
		nx, ny, nz = meta.fft_grid
		x_chunk_size = min(x_chunk_size_override, nx)
		print(f"  NOTE: Using explicit x_chunk_size={x_chunk_size} (override)")
	
	# Output path for zeta
	zeta_h5_path = os.path.join(output_dir, "zeta_q.h5")
	
	print(f"\n  Chunked ISDF fitting:")
	print(f"    Band chunks: {band_chunk_size}")
	print(f"    X chunks:    {x_chunk_size} (contiguous r-space)")
	print(f"    Q chunks:    {q_chunk_size}")
	print(f"    G-space cache: {'enabled' if use_gspace_cache else 'disabled'}")
	print(f"    Zeta output: {zeta_h5_path}")
	
	# Step 1: Fit zeta and write to HDF5
	with timing.section("cohsex_jax.zeta_fit_chunked"):
		psi_l_rmu_Y, psi_l_rmuT_X, psi_r_rmu_Y, psi_r_rmuT_X = fit_zeta_chunked_to_h5(
			wfn, sym, meta, centroid_indices, mesh_xy,
			x_chunk_size, zeta_h5_path, band_chunk_size, q_chunk_size, bispinor,
			use_gspace_cache=use_gspace_cache,
			band_range_left=band_range_left,
			band_range_right=band_range_right,
		)
	
	# Step 2: Compute V_qmunu from zeta
	# Ensure filesystem is flushed before reading
	if jax.process_index() == 0:
		os.sync()
	jax.experimental.multihost_utils.sync_global_devices("zeta_flush")
	
	bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)
	cell_volume = float(wfn.cell_volume)
	
	# V_q memory: zeta_mu(G) + zeta_nu(G) + FFT workspace + V_q block
	# Available: full budget (no centroids needed, zeta read from H5)
	n_G = meta.n_rtot
	bytes_per_complex = 16
	# Use 90% of budget - zeta is streamed from H5, no centroids in memory
	m_budget_vcoul = memory_budget_gb * 1e9 * 0.90
	# Each mu needs: 2 × n_G × 16 (zeta_mu + zeta_nu for off-diag) + 1 × n_G × 16 (FFT workspace)
	m_per_mu = 3 * bytes_per_complex * n_G
	mu_chunk_vcoul = max(1, min(meta.n_rmu, int(m_budget_vcoul / m_per_mu)))
	q_batch_vcoul = 1
	n_q_total = n_q
	if mu_chunk_vcoul >= meta.n_rmu and n_q_total > 1:
		q_batch_vcoul = min(4, n_q_total)
	print(f"    V_q mu chunks: {mu_chunk_vcoul}")
	if q_batch_vcoul > 1:
		print(f"    V_q q batches: {q_batch_vcoul}")
	
		with timing.section("cohsex_jax.V_q_compute"):
			with h5py.File(zeta_h5_path, 'r') as zeta_h5:
				with mesh_xy:
					V_qmunu_raw, g0_mu_all = compute_all_V_q_from_zeta_h5(
						zeta_h5,
						kgrid=meta.kgrid,
						fft_grid=meta.fft_grid,
						bvec=bvec,
						cell_volume=cell_volume,
						mu_chunk_size=mu_chunk_vcoul,
						mesh_xy=mesh_xy,
						sys_dim=sys_dim,
						q_batch_size=q_batch_vcoul if mu_chunk_vcoul >= meta.n_rmu else None,
					)
	
	# Write G0 (ζ_μ(G=0) for each q) to the zeta HDF5 file for restart/reuse
	# g0_mu_all shape: (nqx, nqy, nqz, n_rmu)
	if jax.process_index() == 0:
		with h5py.File(zeta_h5_path, 'a') as f:
			g0_np = np.asarray(g0_mu_all)
			if 'g0_mu' in f:
				del f['g0_mu']  # Overwrite if exists
			f.create_dataset('g0_mu', data=g0_np)
			print(f"    G0 written to {zeta_h5_path} (shape: {g0_np.shape})")
	jax.experimental.multihost_utils.sync_global_devices("g0_write")
	
	# TODO: Compute and store S_qmunu = ⟨ζ_μ|ζ_ν⟩ overlap matrix during V_q computation.
	# This would be useful for Hartree potential and other applications.
	# S_q = zeta^H @ zeta, can be computed from zeta chunks as sum over q.
	
	# Reshape V_qmunu to match expected format: (1, npol, npol, nkx, nky, nkz, n_rmu, n_rmu)
	# Our V_qmunu_raw is (nkx, nky, nkz, n_rmu, n_rmu)
	V_qmunu = V_qmunu_raw[None, None, None, :, :, :, :, :]  # Add leading dims
	# Broadcast to (1, npol, npol, ...)
	V_qmunu = jnp.broadcast_to(V_qmunu, (1, meta.npol, meta.npol, nqx, nqy, nqz, meta.n_rmu, meta.n_rmu))
	V_qmunu = jnp.array(V_qmunu)  # Force copy to avoid broadcast issues
	
	# Extract v_q0 (q=0 with G=0 excluded) and G0 (ζ_μ at G=0)
	v_q0_noG0_munu = V_qmunu_raw[0, 0, 0, :, :]  # At q=(0,0,0)
	G0_mu_nu = g0_mu_all[0, 0, 0, :]  # ζ_μ(G=0) at q=0
	
	print(f"\n  V_q computed:")
	print(f"    Shape: {V_qmunu.shape}")
	print(f"    V_q=0 trace: {jnp.trace(v_q0_noG0_munu).real:.4f}")
	
	return {
		'V_qmunu': V_qmunu,
		'v_q0_noG0_munu': v_q0_noG0_munu,
		'G0_mu_nu': G0_mu_nu,
		'psi_l_rmu_Y': psi_l_rmu_Y,
		'psi_l_rmuT_X': psi_l_rmuT_X,
		'psi_r_rmu_Y': psi_r_rmu_Y,
		'psi_r_rmuT_X': psi_r_rmuT_X,
		'zeta_h5_path': zeta_h5_path,
	}


# ================= JAX-sharded Sigma pipeline =================

def get_G_mu_nu_jax(psi_vTX, psi_vY, Gij_static):
	"""Pure: psi_* (nk, nb, nspinor, n_rmu), Gij_static (nk,nb,nb) -> G_k (nk, nspinor, n_rmu, nspinor, n_rmu).
	
	Computes G_μν(k) = Σ_ij ψ*_ik(r_μ) G_ijk ψ_jk(r_ν)
	
	Zero-comm contraction when left is X-sharded on rmu and right is Y-sharded on rmu.
	Gij_static should be initialized as zeros with diagonal 0:nelec set to 1.0+0.j
	for the static COHSEX Green's function (identity on occupied states).
	"""
		
	# G[k,s,x,t,y] = Σ_ij ψ*_ik(s,x) G_ijk ψ_jk(t,y)
	# psi_vTX: (k, spinor, rmu, band) = 'ksxi'
	# Gij: (k, band_i, band_j) = 'kij'  
	# conj(psi_vY): (k, band, spinor, rmu) = 'kjty'
	G_k = jnp.einsum('ksxi,kij,kjty->ksxty', psi_vTX, Gij_static, jnp.conj(psi_vY), optimize=True)
	return G_k

def get_G_mu_nu_RI(psi_vTX, psi_vY):
	"""Pure: psi_* (nk, nb, nspinor, n_rmu) -> G_k (nk, nspinor, n_rmu, nspinor, n_rmu).
	
	Computes G_μν(k) = Σ_n ψ*_nk(r_μ) ψ_nk(r_ν) for ALL bands (no occupation weighting).
	
	This is the "resolution of identity" style sum used for the Coulomb hole term.
	Zero-comm contraction when left is X-sharded on rmu and right is Y-sharded on rmu.
	"""
	# G[k,s,x,t,y] = Σ_n ψ*_nk(s,x) ψ_nk(t,y)
	# psi_vTX: (k, spinor, rmu, band) = 'ksxn'
	# conj(psi_vY): (k, band, spinor, rmu) = 'knty'
	G_k = jnp.einsum('ksxn,knty->ksxty', psi_vTX, jnp.conj(psi_vY), optimize=True)
	return G_k

def get_G_R_jax(G_k, nkx, nky, nkz):
	"""Pure: (nk, s1,rmu1,s2,rmu2) -> (s1,rmu1,s2,rmu2,nkx,nky,nkz)."""
	G_k = G_k.transpose(1, 2, 3, 4, 0)
	G_k = G_k.reshape(*G_k.shape[:4], nkx, nky, nkz)  # (s1,rmu1,s2,rmu2,nkx,nky,nkz)
	G_k = jax.lax.with_sharding_constraint(G_k, P(None, 'x', None, 'y', None, None, None))
	return jnp.fft.ifftn(G_k, axes=(4,5,6), norm='ortho')

def get_sigma_static_mu_nu_jax(G_R, V_mu_nu, nk_tot, bispinor=False):
	"""Compute sigma in (s1,rmu1,s2,rmu2,nkx,nky,nkz) basis via convolution in real space.
	
	For nspinor=2: Σ_ab(μ,ν,R) = G_ab(μ,ν,R) * V(μ,ν,R)
	
	For nspinor=4 (bispinor): Uses γ⁰ Coulomb vertex:
		Σ_ab = γ⁰_aa γ⁰_bb G_ab V
	where γ⁰ = diag(1,1,-1,-1) in the Dirac representation.
	This gives sign_a × sign_b × G_ab × V where sign=[1,1,-1,-1].
	Large-large and small-small blocks get +1, cross terms get -1.
	
	Args:
		G_R: (nspinor, rmu1, nspinor, rmu2, nkx, nky, nkz) Green's function in real space
		V_mu_nu: (rmu1, rmu2, nkx, nky, nkz) Coulomb interaction
		nk_tot: Total number of k-points for normalization
		bispinor: If True, apply γ⁰ vertex factors for 4-component spinors
		
	Returns:
		sigma_k: Same shape as G_R, self-energy in k-space
	"""
	V_R = V_mu_nu[None, :, None, :, :, :, :]
	V_R = jnp.array(V_R, copy=True)
	
	# For bispinor case, apply γ⁰ vertex: Σ_ab = γ⁰_aa γ⁰_bb G_ab V
	# γ⁰ = diag(1,1,-1,-1), so sign_a * sign_b gives the prefactor
	if bispinor:
		# sign[a] * sign[b] for 4-component spinors: [1,1,-1,-1]
		gamma0_diag = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64)
		# Outer product gives sign_a * sign_b matrix of shape (4, 4)
		gamma0_vertex = gamma0_diag[:, None] * gamma0_diag[None, :]  # (4,4)
		# Broadcast to G_R shape: (s1, 1, s2, 1, 1, 1, 1)
		gamma0_vertex = gamma0_vertex[:, None, :, None, None, None, None]
		G_R = G_R * gamma0_vertex
	
	sigma_R = G_R * jnp.fft.ifftn(V_R, axes=(4,5,6), norm='ortho') * (-1.0 / jnp.sqrt(nk_tot))
	sigma_R = jax.lax.with_sharding_constraint(sigma_R, P(None, 'x', None, 'y', None, None, None))
	return jnp.fft.fftn(sigma_R, axes=(4,5,6), norm='ortho')

def get_sigma_static_kij_jax(psi_sigX, psi_sigTY, sigma_k_munu):
	"""Project self-energy from (spinor,rmu) basis to band basis.
	
	Computes: Σ_mn(k) = Σ_{s,t,μ,ν} ψ*_ms(k,μ) Σ_st(k,μ,ν) ψ_nt(k,ν)
	
	Works for both 2-component (Pauli) and 4-component (bispinor) wavefunctions.
	The spinor contraction sums over all spinor components (s,t indices).
	
	Args:
		psi_sigX: (nk, nb, nspinor, rmu) wavefunctions
		psi_sigTY: (nk, nspinor, rmu, nb) transposed wavefunctions
		sigma_k_munu: (nspinor, rmu1, nspinor, rmu2, nkx, nky, nkz) self-energy
		
	Returns:
		sigma_kij: (nk, nb, nb) band-space self-energy matrix
	"""
	nkx, nky, nkz = sigma_k_munu.shape[-3:]
	nk = nkx * nky * nkz
	sigma_k = sigma_k_munu.transpose(4, 5, 6, 0, 1, 2, 3).reshape(nk, *sigma_k_munu.shape[:4]) # (nk,s1,rmu1,s2,rmu2)
	left = jnp.einsum('kmsx,ksxty->kmty', jnp.conj(psi_sigX), sigma_k, optimize=True)
	return jnp.einsum('kmty,ktyn->kmn', left, psi_sigTY, optimize=True)


# ================= Hartree helper functions =================

def build_density_from_Gij(psi_rmu, Gij, nk_tot):
	"""Build charge density at centroids from wavefunctions and Green's function.
	
	ρ_μ = (1/Nk) Σ_k Σ_{ij} G_ij(k) ψ*_ik(r_μ) ψ_jk(r_μ)
	
	For diagonal Gij (initial), this reduces to Σ_n f_n |ψ_n|².
	For Gij = U @ diag(f) @ U† (self-consistent), this correctly computes
	the density from the QP Green's function in the DFT basis.
	
	Args:
		psi_rmu: (nk, nb, nspinor, n_rmu) wavefunctions at centroids
		Gij: (nk, nb, nb) Green's function matrix (FULL matrix, not just diagonal)
		nk_tot: Total number of k-points for BZ averaging
		
	Returns:
		rho_mu: (n_rmu,) density at centroids
	"""
	# psi_rmu: (nk, nb, ns, rmu)
	# Gij: (nk, ni, nj)
	# rho_μ = Σ_k Σ_{ij} Σ_s G_ij ψ*_is(μ) ψ_js(μ)
	# Contract: 'kij,kismu,kjsmu->mu' but need real part
	# Build ψ*_i ψ_j first: (nk, ni, nj, ns, rmu)
	psi_conj = jnp.conj(psi_rmu)  # (nk, nb, ns, rmu)
	# ψ*_i(μ) ψ_j(μ) summed over spinor
	psi_ij = jnp.einsum('kisx,kjsx->kijx', psi_conj, psi_rmu, optimize=True)  # (nk, ni, nj, rmu)
	# Contract with Gij and sum over k, i, j
	rho_mu = jnp.einsum('kij,kijx->x', Gij, psi_ij, optimize=True)
	rho_mu = jnp.real(rho_mu) / jnp.asarray(nk_tot, dtype=jnp.float64)
	return rho_mu


def build_hartree_potential(rho_mu, V0_munu):
	"""Build Hartree potential at centroids from density.
	
	[Vρ]_μ = Σ_ν V0_μν ρ_ν
	
	Args:
		rho_mu: (n_rmu,) density at centroids
		V0_munu: (n_rmu, n_rmu) bare Coulomb at q=0 (G=0 excluded)
		
	Returns:
		Vrho_mu: (n_rmu,) Hartree potential at centroids
	"""
	return jnp.einsum('xy,y->x', V0_munu, rho_mu, optimize=True)


def project_potential_to_bands(psi_rmu, Vrho_mu):
	"""Project local potential to band matrix elements.
	
	V_mn(k) = Σ_μ,s ψ*_mk(r_μ) V_μ ψ_nk(r_μ)
	
	Args:
		psi_rmu: (nk, nb, nspinor, n_rmu) wavefunctions at centroids
		Vrho_mu: (n_rmu,) potential at centroids
		
	Returns:
		V_kmn: (nk, nb, nb) potential matrix elements
	"""
	return jnp.einsum('kmsx,x,knsx->kmn', jnp.conj(psi_rmu), Vrho_mu, psi_rmu, optimize=True)

def compute_sigma_pipeline_jax(
	psi_l_rmuT_X,
	psi_l_rmu_Y,
	psi_coh_rmuT_X,
	psi_coh_rmu_Y,
	psi_proj_rmu_X,
	psi_proj_rmuT_Y,
	W_mu_nu,
	V_mu_nu,
	V0_munu,
	Gij_static,
	nkx: int,
	nky: int,
	nkz: int,
	nk_tot: int,
	nspinor: int,
	fft_vol_au: float,
	bispinor: bool = False,
):
	"""
	Pure JAX pipeline: compute static COHSEX self-energy components and Hartree.
	
	Returns:
		sigma_sx_kij: (nk, nb_sigma, nb_sigma) complex - screened exchange self-energy
		sigma_coh_kij: (nk, nb_sigma, nb_sigma) complex - Coulomb hole self-energy
		hartree_kmn: (nk, nb_sigma, nb_sigma) complex - Hartree matrix elements
	
	Wavefunctions:
		psi_l: sigma window bands (b0, b3) for SX Green's function + Hartree density
		       shape (nk, nb_sigma, nspinor, n_rmu)
		psi_coh: ALL bands (b0, b4) for COH resolution of identity
		         shape (nk, nband_full, nspinor, n_rmu)
		psi_proj: sigma window bands (b0, b3) for final projection <m|Σ|n>
		          shape (nk, nb_sigma, nspinor, n_rmu)
	
	Gij_static:
		Static Green's function matrix in band space, shape (nk, nb_sigma, nb_sigma).
		For COHSEX: zeros with diagonal 0:nelec set to 1.0+0.j (projector onto occupied).
		Must match psi_l band range.
	
	W_mu_nu:
		Screened Coulomb interaction, shape (nrmu1, nrmu2, nkx, nky, nkz).
		Same shardings as V_mu_nu.
	
	SCREENED EXCHANGE (Σ_sx):
		G_μν(k) = Σ_ij ψ*_ik(r_μ) G_ijk ψ_jk(r_ν)  [Green's function from Gij_static]
		G_μν(R) = FFT[ G_μν(k) ]                    [to real-space lattice]
		Σ_sx_μν(k) = (1/N_k) Σ_R G_μν(R) W_μν(R)   [screened exchange in ISDF basis]
		Σ_sx_ij(k) = Σ_μν ψ*_i(r_μ) Σ_μν ψ_j(r_ν)  [project to sigma bands]
	
	COULOMB HOLE (Σ_coh):
		G_RI_μν(k) = Σ_n ψ*_nk(r_μ) ψ_nk(r_ν)      [RI sum over ALL nband bands]
		G_RI_μν(R) = FFT[ G_RI_μν(k) ]              [to real-space lattice]
		Σ_coh_μν(k) = (1/N_k) Σ_R G_RI_μν(R) [V_μν(R) - W_μν(R)]
		Σ_coh_ij(k) = Σ_μν ψ*_i(r_μ) Σ_μν ψ_j(r_ν) [project to sigma bands]
	
	HARTREE (V_H):
		ρ_μ = (1/N_k) Σ_k,n,s f_n |ψ_nk(r_μ)|²   [density, weighted by Gij diagonal]
		[Vρ]_μ = Σ_ν V0_μν ρ_ν                    [Hartree potential at centroids]
		<m|V_H|n>_k = Σ_μ,s ψ*_mk(r_μ) [Vρ]_μ ψ_nk(r_μ)  [project to sigma bands]
	
	Key: V0_munu is V(q=0) with G=0 component EXCLUDED (to avoid divergence).
	     The G=0 piece is added back via the head correction in the main pipeline.
	"""
	# psi_l: sigma window (b0, b3) for SX Green's function + Hartree density
	# psi_coh: all bands (b0, b4) for COH resolution of identity
	# psi_proj: sigma window (b0, b3) for final projection <m|Σ|n>
	
	# ========== SCREENED EXCHANGE SELF-ENERGY (Σ_sx) ==========
	# G_μν(k): Green's function in ISDF basis, built via Gij_static projector
	# Gij_static is sized for psi_l bands (nb_sigma x nb_sigma)
	G_k = get_G_mu_nu_jax(psi_l_rmuT_X, psi_l_rmu_Y, Gij_static)  # (nk,spin,μ,spin,ν)
	# G_μν(R): FFT to real-space lattice vectors R
	G_R = get_G_R_jax(G_k, nkx, nky, nkz)
	# Σ_sx_μν(k): screened exchange via ISDF
	# For bispinor: applies γ⁰ vertex factors to Coulomb interaction
	sigma_sx_k_munu = get_sigma_static_mu_nu_jax(G_R, W_mu_nu, nk_tot, bispinor=bispinor)
	# Σ_sx_ij(k): project to SIGMA WINDOW bands using psi_proj
	sigma_sx_kij = get_sigma_static_kij_jax(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_sx_k_munu)

	# ========== COULOMB HOLE SELF-ENERGY (Σ_coh) ==========
	# G_RI_μν(k): resolution of identity sum over ALL bands (no occupation weighting)
	# Use psi_coh which has all bands (b0, b4)
	G_RI_k = get_G_mu_nu_RI(psi_coh_rmuT_X, psi_coh_rmu_Y)  # (nk,spin,μ,spin,ν)
	# G_RI_μν(R): FFT to real-space lattice vectors R
	G_RI_R = get_G_R_jax(G_RI_k, nkx, nky, nkz)
	# Σ_coh_μν(k): Coulomb hole via (W - V)
	# BerkeleyGW uses [ε⁻¹_{GG'} - δ_{GG'}] v(q+G') = (W - V), NOT (V - W)
	# See bgw_src/Sigma/mtxel_cor.f90 lines 604-607
	# 
	# IMPORTANT: CH has a factor of 1/2 that SX does not have!
	# See mtxel_cor.f90 line 1646: achtemp_loc = achtemp_loc + 0.5d0*aqsn_Ieps
	# vs line 1648: if (flag_occ) asxtemp_loc = asxtemp_loc - aqsn_Ieps
	W_minus_V = W_mu_nu - V_mu_nu
	# For bispinor: applies γ⁰ vertex factors to Coulomb interaction
	sigma_coh_k_munu = get_sigma_static_mu_nu_jax(G_RI_R, W_minus_V, nk_tot, bispinor=bispinor)
	# Σ_coh_ij(k): project to SIGMA WINDOW bands using psi_proj
	# Apply the 0.5 factor for COH and the overall minus sign
	sigma_coh_kij = -0.5 * get_sigma_static_kij_jax(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_coh_k_munu)

	# ========== HARTREE MATRIX ELEMENTS ==========
	# Uses Gij_static diagonal for occupation weights
	rho_mu = build_density_from_Gij(psi_l_rmu_Y, Gij_static, nk_tot)
	Vrho_mu = build_hartree_potential(rho_mu, V0_munu)
	hartree_kmn = project_potential_to_bands(psi_proj_rmu_X, Vrho_mu)
	
	return sigma_sx_kij, sigma_coh_kij, hartree_kmn


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
	
	# ========================================================================
	# INITIALIZATION - Build device mesh early for banner
	# ========================================================================
	current_backend = jax.default_backend()
	n_devices = len(jax.devices())
	n_procs = jax.process_count()
	device_names = jax.devices()[0].device_kind if n_devices > 0 else "unknown"
	
	# Construct device mesh early so we can report it in the banner
	total_devices = jax.process_count() * jax.local_device_count()
	grid_x = int(np.sqrt(total_devices))
	while total_devices % grid_x != 0:
		grid_x -= 1
	grid_y = total_devices // grid_x
	devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
	mesh_xy = Mesh(devices_2d, ['x', 'y'])
	
	_print0("")
	_print0("=" * 72)
	_print0("  COHSEX-JAX: Self-Energy Calculation")
	_print0("=" * 72)
	_print0(f"  Backend: {current_backend.upper():<8}  Devices: {n_devices}  Mesh: {grid_x}×{grid_y}  Processes: {n_procs}")
	_print0(f"  Device type: {device_names}")
	_print0("=" * 72)
	_print0("")
	
	# Resolve relative paths against the input file's directory
	input_dir = os.path.dirname(os.path.abspath(args.input))
	resolve_input_paths(params, input_dir)
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
	centroids_frac, centroid_indices, _n_rmu = load_centroids(params["centroids_file"], wfn.fft_grid)
	# Resolve tmp_dir and output path relative to input file directory
	tmp_dir = os.path.join(input_dir, "tmp")
	os.makedirs(tmp_dir, exist_ok=True)
	tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{_n_rmu}.h5")
	_print0(f"  ISDF basis: {_n_rmu} centroids")
	_print0(f"  FFT grid: {wfn.fft_grid[0]}×{wfn.fft_grid[1]}×{wfn.fft_grid[2]}   Cell volume: {wfn.cell_volume:.2f} a.u.³")
	_print0("")

	# windows for polarizability and sigma
	# Get window information
	epsq = 0.01
	#window_pairs = get_window_info(epsq, wfn)

	# ============================================================================
	# MAIN CONTROL FLAGS (from input file):
	# ============================================================================
	# restart:     If True, load ISDF vectors and V_qmunu from isdf_tensors_*.h5
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
	# use_chunked_isdf: If True, use the memory-efficient chunked ISDF fitting
	#        that writes zeta to HDF5 and computes V_q separately. Enables larger
	#        systems by chunking over z-axis and using spin-traced pair density.
	# ============================================================================
	do_G0 = True   # Always True: head corrections are essential
	do_S = False   # Overlap matrix storage (disabled, not needed)
	use_chunked_isdf = params.get("use_chunked_isdf", True)  # Read from input file
	
	# Memory budget for chunked ISDF (0 = auto-detect)
	memory_per_device_gb = params.get("memory_per_device_gb", 0.0)
	if memory_per_device_gb <= 0:
		from ..common.gpu_utils import get_device_memory_gb
		memory_per_device_gb = get_device_memory_gb()
		if jax.process_index() == 0:
			print(f"  Auto-detected memory budget: {memory_per_device_gb:.2f} GB/device")
	
	# Explicit x_chunk_size override (0 = auto-compute from memory budget)
	x_chunk_size_override = params.get("x_chunk_size", 0)
	
	if x_only and do_screened:  # x_only=bare exchange only, do_screened=use W instead of v
		raise ValueError("x_only and do_screened cannot both be True")

	meta = Meta.from_system(wfn, sym, nval, ncond, nband, _n_rmu, bispinor)
	meta.rank = jax.process_index()
	meta.n_proc = jax.process_count()
	meta.sys_dim = sys_dim
	meta.bispinor = bispinor
	meta.chunk_size = get_effective_chunk_size(params["chunk_size"])
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
	# Shardings from the pre-constructed mesh
	sh = make_shardings(mesh_xy)
	
	chunk_str = "disabled" if meta.chunk_size is None else str(meta.chunk_size)
	print0(f"  Band chunk size: {chunk_str}")

	# Initialize timing system
	timing.reset()
	if not restart:
		# ============================================================================
		# BAND RANGE DEFINITIONS
		# ============================================================================
		valence_range = (b0, b2)      # Valence bands only (for chi0)
		conduction_range = (b2, b4)   # Conduction bands only (for chi0)
		brange_l = nvplussigrange     # (b0, b3) for ISDF left
		brange_r = ncplussigrange     # (b1, b4) for ISDF right
		full_band_range = (b0, b4)    # All bands (for COH G_RI)
		
		# ============================================================================
		# CHUNKED ISDF PATH (memory-efficient, spin-traced pair density)
		# ============================================================================
		if use_chunked_isdf:
			print0("  Using CHUNKED ISDF fitting (memory-efficient)")
			
			# Fit zeta and compute V_q using chunked approach
			with mesh_xy:
				chunked_result = fit_zeta_and_compute_V_q_chunked(
					wfn, sym, meta, centroid_indices, mesh_xy,
					output_dir=tmp_dir,
					bispinor=bispinor,
					memory_budget_gb=memory_per_device_gb,
					sys_dim=sys_dim,
					x_chunk_size_override=x_chunk_size_override,
				)
			
			# Extract results
			V_qmunu = chunked_result['V_qmunu']
			v_q0_noG0_munu = chunked_result['v_q0_noG0_munu']
			G0_mu_nu = chunked_result['G0_mu_nu']
			psi_l_rmu_Y = chunked_result['psi_l_rmu_Y']
			psi_l_rmuT_X = chunked_result['psi_l_rmuT_X']
			psi_r_rmu_Y = chunked_result['psi_r_rmu_Y']
			psi_r_rmuT_X = chunked_result['psi_r_rmuT_X']
			
			# Still need to load psi_v, psi_c, psi_coh for chi0 and COH
			with timing.section("cohsex_jax.wavefunction_setup"):
				# Load VALENCE wavefunctions for chi0 (psi_v)
				global_psiG_v, nb_v = read_Gvecs_to_devices(wfn, sym, valence_range, meta, bispinor, mesh_xy)
				psi_v_rtot_Y, psi_v_rmu_Y, psi_v_rmuT_X = get_sharded_wfns(
					global_psiG_v, sym, meta, centroid_indices, nb_v, False, mesh_xy
				)
				psi_v_rtot_Y.block_until_ready(); psi_v_rmu_Y.block_until_ready(); psi_v_rmuT_X.block_until_ready()
				del global_psiG_v
				gc.collect()
				
				# Load CONDUCTION wavefunctions for chi0 (psi_c)
				global_psiG_c, nb_c = read_Gvecs_to_devices(wfn, sym, conduction_range, meta, bispinor, mesh_xy)
				psi_c_rtot_Y, psi_c_rmu_Y, psi_c_rmuT_X = get_sharded_wfns(
					global_psiG_c, sym, meta, centroid_indices, nb_c, False, mesh_xy
				)
				psi_c_rtot_Y.block_until_ready(); psi_c_rmu_Y.block_until_ready(); psi_c_rmuT_X.block_until_ready()
				del global_psiG_c
				gc.collect()
				
				# Load ALL bands for COH resolution of identity (psi_coh)
				global_psiG_coh, nb_coh = read_Gvecs_to_devices(wfn, sym, full_band_range, meta, bispinor, mesh_xy)
				psi_coh_rtot_Y, psi_coh_rmu_Y, psi_coh_rmuT_X = get_sharded_wfns(
					global_psiG_coh, sym, meta, centroid_indices, nb_coh, False, mesh_xy
				)
				psi_coh_rtot_Y.block_until_ready(); psi_coh_rmu_Y.block_until_ready(); psi_coh_rmuT_X.block_until_ready()
				del global_psiG_coh
				gc.collect()
				print0("  Wavefunction loading complete")
			
			# Get energies (still needed for downstream)
			enk_v, weights_v = get_enk_bandrange(wfn, sym, valence_range, (b1, b2), nspinor=meta.nspinor)
			enk_c, weights_c = get_enk_bandrange(wfn, sym, conduction_range, (b2, b4), nspinor=meta.nspinor)
			enk_l, weights_l = get_enk_bandrange(wfn, sym, brange_l, (b1, b3), nspinor=meta.nspinor)
			enk_r, weights_r = get_enk_bandrange(wfn, sym, brange_r, (b1, b3), nspinor=meta.nspinor)
			
			# Apply sharding (matching original code)
			with mesh_xy:
				# Valence wavefunctions for chi0 (psi_v)
				psiv_Y = jax.device_put(psi_v_rmu_Y, sh.y3_4)
				psivT_X = jax.lax.with_sharding_constraint(psiv_Y.transpose(0, 2, 3, 1), sh.x2_4)
				# Conduction wavefunctions for chi0 (psi_c)  
				psic_X = jax.device_put(psi_c_rmu_Y, sh.x3_4)
				psicT_Y = jax.lax.with_sharding_constraint(psic_X.transpose(0, 2, 3, 1), sh.y2_4)
				# ISDF left wavefunctions (psi_l) - Y-sharded for SX/Hartree
				psil_Y = jax.device_put(psi_l_rmu_Y, sh.y3_4)
				psilT_X = jax.lax.with_sharding_constraint(psil_Y.transpose(0, 2, 3, 1), sh.x2_4)
				# ISDF left wavefunctions - X-sharded for projection
				psil_X = jax.device_put(psi_l_rmu_Y, sh.x3_4)
				psilT_Y = jax.lax.with_sharding_constraint(psil_X.transpose(0, 2, 3, 1), sh.y2_4)
				# ISDF right wavefunctions (psi_r)
				psir_X = jax.device_put(psi_r_rmu_Y, sh.x3_4)
				psirT_Y = jax.lax.with_sharding_constraint(psir_X.transpose(0, 2, 3, 1), sh.y2_4)
				# COH resolution of identity (psi_coh)
				psicoh_Y = jax.device_put(psi_coh_rmu_Y, sh.y3_4)
				psicohT_X = jax.lax.with_sharding_constraint(psicoh_Y.transpose(0, 2, 3, 1), sh.x2_4)
			
			# S_qmunu not computed by chunked approach (do_S=False)
			S_qmunu = None
			
			# ============================================================================
			# WAVEFUNCTION ARRAYS (matching else block structure):
			# ============================================================================
			psi_v = psiv_Y       # Valence for chi0
			psi_vT = psivT_X     # Valence transposed for chi0
			psi_c = psic_X       # Conduction for chi0
			psi_cT = psicT_Y     # Conduction transposed for chi0
			psi_l = psil_Y       # ISDF left (Y-sharded for SX/Hartree)
			psi_lT = psilT_X     # ISDF left transposed (X-sharded)
			psi_l_proj = psil_X  # ISDF left (X-sharded for projection)
			psi_lT_proj = psilT_Y  # ISDF left transposed (Y-sharded for projection)
			psi_l_full = psi_l   # Alias for h5 saving
			psi_lT_full = psi_lT # Alias for h5 saving
			psi_r = psir_X       # ISDF right
			psi_rT = psirT_Y     # ISDF right transposed
			psi_coh = psicoh_Y   # COH G_RI (all bands)
			psi_cohT = psicohT_X # COH G_RI transposed
			
			nb_sigma = int(b3 - b0)  # = nelec + ncond (sigma window size)
			
			# V_mu_nu is V_qmunu without q indices (for screened interaction calculation)
			# Extract and transpose to match expected shape: (n_rmu, n_rmu, nkx, nky, nkz)
			V_mu_nu_raw = jnp.asarray(V_qmunu)[0, 0, 0]  # (nkx, nky, nkz, n_rmu, n_rmu)
			V_mu_nu = V_mu_nu_raw.transpose(3, 4, 0, 1, 2)  # → (n_rmu, n_rmu, nkx, nky, nkz)
			with mesh_xy:
				V_mu_nu = jax.lax.with_sharding_constraint(V_mu_nu, sh.V_shard)
			
			# Persist restart artifacts
			write_labeled_arrays_to_h5(
				tensors_filename,
				V_qmunu,
				psi_l_full,
				psir_X,
				enk_l,
				enk_r,
				S_qmunu,
				V0_noG0_munu=v_q0_noG0_munu,
				G0_mu_nu=G0_mu_nu,
				init_W0=True,
			)
			save_restart_per_proc(os.path.join(tmp_dir, "isdf_tensors"), V_qmunu, S_qmunu, psi_l_full, psir_X, enk_l, enk_r, meta, mesh_xy, V0_noG0_munu=v_q0_noG0_munu)
			V_qmunu.block_until_ready()
			print0("  Chunked ISDF path complete")
		
		# ============================================================================
		# ORIGINAL ISDF PATH (per-q-point fitting, keeps rtot in memory)
		# ============================================================================
		else:
			with timing.section("cohsex_jax.wavefunction_setup") as timer_setup:
				# ============================================================================
				# WAVEFUNCTION LOADING STRATEGY:
				# ============================================================================
				# For ISDF fitting (zeta construction):
				#   - psi_l: nvplussigrange (b0, b3) - all valence + sigma-window conduction
				#   - psi_r: ncplussigrange (b1, b4) - sigma-window valence + all conduction
				# For chi0/W calculation (screened interaction):
				#   - psi_v: VALENCE bands only (b0, b2) - used as left/right in chi0
				#   - psi_c: CONDUCTION bands only (b2, b4) - used as left/right in chi0
				# For COH G_RI (resolution of identity):
				#   - psi_coh: ALL bands (b0, b4) - needed for COH sum over all bands
				# For final sigma projection:
				#   - uses psi_l (b0, b3) which covers sigma window
				# ============================================================================
				
				# Load LEFT wavefunctions for ISDF fitting and sigma projection (psi_l)
				# Range: (b0, b3) = all valence + sigma-window conduction
				global_psiG_l, nb_l = read_Gvecs_to_devices(wfn, sym, brange_l, meta, bispinor, mesh_xy)
				psi_l_rtot_Y, psi_l_rmu_Y, psi_l_rmuT_X = get_sharded_wfns(
					global_psiG_l, sym, meta, centroid_indices, nb_l, False, mesh_xy
				)
				psi_l_rtot_Y.block_until_ready(); psi_l_rmu_Y.block_until_ready(); psi_l_rmuT_X.block_until_ready()
				del global_psiG_l
				gc.collect()
				
				# Load RIGHT wavefunctions for ISDF fitting (psi_r)
				# Range: (b1, b4) = sigma-window valence + all conduction
				global_psiG_r, nb_r = read_Gvecs_to_devices(wfn, sym, brange_r, meta, bispinor, mesh_xy)
				psi_r_rtot_Y, psi_r_rmu_Y, psi_r_rmuT_X = get_sharded_wfns(
					global_psiG_r, sym, meta, centroid_indices, nb_r, False, mesh_xy
				)
				psi_r_rtot_Y.block_until_ready(); psi_r_rmu_Y.block_until_ready(); psi_r_rmuT_X.block_until_ready()
				del global_psiG_r
				gc.collect()
				
				# Load VALENCE wavefunctions for chi0 (psi_v)
				global_psiG_v, nb_v = read_Gvecs_to_devices(wfn, sym, valence_range, meta, bispinor, mesh_xy)
				psi_v_rtot_Y, psi_v_rmu_Y, psi_v_rmuT_X = get_sharded_wfns(
					global_psiG_v, sym, meta, centroid_indices, nb_v, False, mesh_xy
				)
				psi_v_rtot_Y.block_until_ready(); psi_v_rmu_Y.block_until_ready(); psi_v_rmuT_X.block_until_ready()
				del global_psiG_v
				gc.collect()
				
				# Load CONDUCTION wavefunctions for chi0 (psi_c)
				global_psiG_c, nb_c = read_Gvecs_to_devices(wfn, sym, conduction_range, meta, bispinor, mesh_xy)
				psi_c_rtot_Y, psi_c_rmu_Y, psi_c_rmuT_X = get_sharded_wfns(
					global_psiG_c, sym, meta, centroid_indices, nb_c, False, mesh_xy
				)
				psi_c_rtot_Y.block_until_ready(); psi_c_rmu_Y.block_until_ready(); psi_c_rmuT_X.block_until_ready()
				del global_psiG_c
				gc.collect()
				
				# Load ALL bands for COH resolution of identity (psi_coh)
				global_psiG_coh, nb_coh = read_Gvecs_to_devices(wfn, sym, full_band_range, meta, bispinor, mesh_xy)
				psi_coh_rtot_Y, psi_coh_rmu_Y, psi_coh_rmuT_X = get_sharded_wfns(
					global_psiG_coh, sym, meta, centroid_indices, nb_coh, False, mesh_xy
				)
				psi_coh_rtot_Y.block_until_ready(); psi_coh_rmu_Y.block_until_ready(); psi_coh_rmuT_X.block_until_ready()
				del global_psiG_coh
				gc.collect()
				print0("  Wavefunction loading complete")

			with timing.section("cohsex_jax.zeta_V_build") as timer_zeta:
				####################################
				# 2.) Explicit q-loop: build zeta_q,mu(r), S_q, and V_q,mu,nu
				####################################
				# Energies for chi0: valence and conduction separately (pass nspinor for bispinor)
				enk_v, weights_v = get_enk_bandrange(wfn, sym, valence_range, (b1, b2), nspinor=meta.nspinor)
				enk_c, weights_c = get_enk_bandrange(wfn, sym, conduction_range, (b2, b4), nspinor=meta.nspinor)
				# Energies for ISDF left/right arrays
				enk_l, weights_l = get_enk_bandrange(wfn, sym, brange_l, (b1, b3), nspinor=meta.nspinor)
				enk_r, weights_r = get_enk_bandrange(wfn, sym, brange_r, (b1, b3), nspinor=meta.nspinor)

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

			print0(f"  ISDF fitting complete ({q_cache.num_q} q-points)")

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
				# Valence wavefunctions for chi0 (psi_v)
				psiv_Y = jax.device_put(psi_v_rmu_Y, sh.y3_4)
				psivT_X = jax.lax.with_sharding_constraint(psiv_Y.transpose(0, 2, 3, 1), sh.x2_4)
				# Conduction wavefunctions for chi0 (psi_c)  
				psic_X = jax.device_put(psi_c_rmu_Y, sh.x3_4)
				psicT_Y = jax.lax.with_sharding_constraint(psic_X.transpose(0, 2, 3, 1), sh.y2_4)
				# ISDF left wavefunctions (psi_l) - Y-sharded for SX/Hartree
				psil_Y = jax.device_put(psi_l_rmu_Y, sh.y3_4)
				psilT_X = jax.lax.with_sharding_constraint(psil_Y.transpose(0, 2, 3, 1), sh.x2_4)
				# ISDF left wavefunctions - X-sharded for projection
				psil_X = jax.device_put(psi_l_rmu_Y, sh.x3_4)
				psilT_Y = jax.lax.with_sharding_constraint(psil_X.transpose(0, 2, 3, 1), sh.y2_4)
				# ISDF right wavefunctions (psi_r): (b1, b4) for ISDF fitting
				psir_X = jax.device_put(psi_r_rmu_Y, sh.x3_4)
				psirT_Y = jax.lax.with_sharding_constraint(psir_X.transpose(0, 2, 3, 1), sh.y2_4)
				# COH resolution of identity (psi_coh): (b0, b4) = all bands
				psicoh_Y = jax.device_put(psi_coh_rmu_Y, sh.y3_4)
				psicohT_X = jax.lax.with_sharding_constraint(psicoh_Y.transpose(0, 2, 3, 1), sh.x2_4)

			# ============================================================================
			# WAVEFUNCTION ARRAYS:
			# ============================================================================
			# psi_v, psi_vT: VALENCE bands (b0, b2) for chi0/W calculation
			# psi_c, psi_cT: CONDUCTION bands (b2, b4) for chi0/W calculation
			# psi_l, psi_lT: (b0, b3) for ISDF fitting left (Y-sharded for SX/Hartree)
			# psi_l_proj, psi_lT_proj: same as psi_l but X-sharded for projection
			# psi_r, psi_rT: (b1, b4) for ISDF fitting right
			# psi_coh, psi_cohT: ALL bands (b0, b4) for COH G_RI resolution of identity
			# ============================================================================
			psi_v = psiv_Y       # Valence for chi0
			psi_vT = psivT_X     # Valence transposed for chi0
			psi_c = psic_X       # Conduction for chi0
			psi_cT = psicT_Y     # Conduction transposed for chi0
			psi_l = psil_Y       # ISDF left (Y-sharded for SX/Hartree)
			psi_lT = psilT_X     # ISDF left transposed (X-sharded)
			psi_l_proj = psil_X  # ISDF left (X-sharded for projection)
			psi_lT_proj = psilT_Y  # ISDF left transposed (Y-sharded for projection)
			psi_l_full = psi_l   # Alias for h5 saving
			psi_lT_full = psi_lT # Alias for h5 saving
			psi_r = psir_X       # ISDF right
			psi_rT = psirT_Y     # ISDF right transposed
			psi_coh = psicoh_Y   # COH G_RI (all bands)
			psi_cohT = psicohT_X # COH G_RI transposed
			
			nb_sigma = int(b3 - b0)  # = nelec + ncond (sigma window size)

			# Persist restart artifacts (store full-sized left/right; trim on load/use)
			write_labeled_arrays_to_h5(
				tensors_filename,
				V_qmunu,
				psi_l_full,
				psir_X,
				enk_l,
				enk_r,
				S_qmunu,
				V0_noG0_munu=v_q0_noG0_munu,
				G0_mu_nu=G0_mu_nu,
				init_W0=True,
			)
			save_restart_per_proc(os.path.join(tmp_dir, "isdf_tensors"), V_qmunu, S_qmunu, psi_l_full, psir_X, enk_l, enk_r, meta, mesh_xy, V0_noG0_munu=v_q0_noG0_munu)
			V_qmunu.block_until_ready()

	elif restart and not x_only:
		with timing.section("cohsex_jax.restart_load") as restart_timer:
			V_qmunu, S_qmunu, psi_lT, psi_l, psi_r, psi_rT, enk_l, enk_r, v_q0_noG0_munu, G0_mu_nu = load_labeled_arrays_from_h5(
				tensors_filename, mesh_xy
			)
			V_mu_nu = jnp.asarray(V_qmunu)[0, 0, 0]
			# Aliases for self-consistent loop
			psi_l_full = psi_l
			psi_lT_full = psi_lT
			# COH needs all bands - in restart mode, psi_l has all bands (old format)
			psi_coh = psi_l
			psi_cohT = psi_lT
			# Create X-sharded versions for projection
			with mesh_xy:
				psi_l_proj = jax.lax.with_sharding_constraint(psi_l, sh.x3_4)
				psi_lT_proj = jax.lax.with_sharding_constraint(
					psi_l_proj.transpose(0, 2, 3, 1), sh.y2_4
				)
			
		# If do_screened, load valence/conduction wavefunctions from WFN for chi0
		if do_screened:
			with timing.section("cohsex_jax.restart_load_chi0_wfns") as timer_chi_wfns:
				valence_range = (b0, b2)
				conduction_range = (b2, b4)
				
				# Load VALENCE wavefunctions for chi0
				global_psiG_v, nb_v = read_Gvecs_to_devices(wfn, sym, valence_range, meta, bispinor, mesh_xy)
				psi_v_rtot_Y, psi_v_rmu_Y, psi_v_rmuT_X = get_sharded_wfns(
					global_psiG_v, sym, meta, centroid_indices, nb_v, False, mesh_xy
				)
				del global_psiG_v
				gc.collect()
				
				# Load CONDUCTION wavefunctions for chi0
				global_psiG_c, nb_c = read_Gvecs_to_devices(wfn, sym, conduction_range, meta, bispinor, mesh_xy)
				psi_c_rtot_Y, psi_c_rmu_Y, psi_c_rmuT_X = get_sharded_wfns(
					global_psiG_c, sym, meta, centroid_indices, nb_c, False, mesh_xy
				)
				del global_psiG_c
				gc.collect()
				
				# Get energies for valence/conduction (pass nspinor for bispinor)
				enk_v, _ = get_enk_bandrange(wfn, sym, valence_range, (b1, b2), nspinor=meta.nspinor)
				enk_c, _ = get_enk_bandrange(wfn, sym, conduction_range, (b2, b4), nspinor=meta.nspinor)
				
				# Apply shardings for chi0
				with mesh_xy:
					psi_v = jax.device_put(psi_v_rmu_Y, sh.y3_4)
					psi_vT = jax.lax.with_sharding_constraint(psi_v.transpose(0, 2, 3, 1), sh.x2_4)
					psi_c = jax.device_put(psi_c_rmu_Y, sh.x3_4)
					psi_cT = jax.lax.with_sharding_constraint(psi_c.transpose(0, 2, 3, 1), sh.y2_4)
				print0(f"  Loaded χ₀ wavefunctions: {nb_v} valence, {nb_c} conduction bands")
	elif restart and x_only:
		with timing.section("cohsex_jax.restart_load_x_only") as restart_timer:
			V_qmunu, S_qmunu, psi_lT, psi_l, psi_r, psi_rT, enk_l, enk_r, v_q0_noG0_munu, G0_mu_nu = load_labeled_arrays_from_h5(
				tensors_filename, mesh_xy
			)
			V_mu_nu = jnp.asarray(V_qmunu)[0, 0, 0]
			# Aliases for self-consistent loop
			psi_l_full = psi_l
			psi_lT_full = psi_lT
			# COH needs all bands - in restart mode, psi_l has all bands (old format)
			psi_coh = psi_l
			psi_cohT = psi_lT
			# Create X-sharded versions for projection
			with mesh_xy:
				psi_l_proj = jax.lax.with_sharding_constraint(psi_l, sh.x3_4)
				psi_lT_proj = jax.lax.with_sharding_constraint(
					psi_l_proj.transpose(0, 2, 3, 1), sh.y2_4
				)

	# Optionally compute chi0 via JAX for screened interaction if requested
	if do_screened:
		with timing.section("cohsex_jax.chi0_W") as timer_chiw:
			with jax_profile.trace_section("chi0_W"):
				# Ensure energies are plain arrays for JAX
				enk_v_arr = jnp.asarray(getattr(enk_v, 'data', enk_v))
				enk_c_arr = jnp.asarray(getattr(enk_c, 'data', enk_c))
				
				# Compute optimal energy windows for CTSP quadrature
				window_pairs = get_window_info(epsq, wfn, nband_max=nband)
				
				# Check if debug_omega is set for dynamic W(ω) testing
				debug_omega = params.get("debug_omega", None)
				if debug_omega is not None:
					ryd2ev = 13.605693122994
					omega_ev = debug_omega * ryd2ev
					print0("")
					print0(f"  [DEBUG] Dynamic screening at ω = {omega_ev:.4f} eV ({debug_omega:.6f} Ry)")
					W_q, chi_omega = get_w_omega_jax(
						V_qmunu, psi_vT, psi_v, psi_c, psi_cT,
						enk_v_arr, enk_c_arr, window_pairs, debug_omega,
						meta, mesh_xy
					)
					W_q.block_until_ready()
					chi_max = float(jnp.max(jnp.abs(chi_omega)))
					print0(f"  [DEBUG] |χ(ω)|_max = {chi_max:.6e}")
				else:
					# Static case (ω = 0)
					chi0 = get_chi0_jax(psi_vT, psi_v, psi_c, psi_cT, enk_v_arr, enk_c_arr, window_pairs, meta, mesh_xy)
					W_q = get_static_w_q_jax(V_qmunu, chi0, None, meta, mesh_xy)
					W_q.block_until_ready()
					if os.path.exists(tensors_filename):
						W0_qmunu = W_q[..., 0, :, 0, :]
						W0_qmunu = W0_qmunu[None, None, None, :, :, :, :, :]
						write_w0_qmunu_to_h5(tensors_filename, W0_qmunu)

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
		# Check if G0_mu_nu is available (might be None for old restart files)
		if G0_mu_nu is None:
			print0("")
			print0("-" * 72)
			print0("  WARNING: G0_mu_nu not available (missing from restart file?)")
			print0("  Skipping head corrections - results may be inaccurate!")
			print0("  Re-run with restart=false to regenerate G0_mu_nu.")
			print0("-" * 72)
			do_G0 = False  # Skip head corrections this run
		else:
			vc0_mean, wcoul0, wcoul0_source = determine_wcoul0(params, input_dir, wfn, sym, meta, print0)
			
			# Print finite-size corrections
			print0("")
			print0("-" * 72)
			print0("  FINITE-SIZE CORRECTIONS")
			print0("-" * 72)
			print0(f"  Head source: {wcoul0_source}")
			vc0_real = float(vc0_mean.real) if hasattr(vc0_mean, 'real') else float(vc0_mean)
			print0(f"  v(q→0)  = {vc0_real:12.3f} a.u.  (bare Coulomb head)")
			if do_screened:
				wcoul0_real = float(wcoul0.real) if hasattr(wcoul0, 'real') else float(wcoul0)
				dW_real = wcoul0_real - vc0_real
				print0(f"  W(q→0)  = {wcoul0_real:12.3f} a.u.  (screened Coulomb head)")
				print0(f"  ΔW      = {dW_real:12.3f} a.u.  (screening correction)")
			
			# outer_u = ζ*_μ(0) ⊗ ζ_ν(0), the "head" direction in μν space
			# Must match V_μν construction: V = Σ_G v(G) ζ*_μ(G) ζ_ν(G)
			# So the head should have conjugate on the LEFT (μ index), not right (ν index)
			outer_u = (jnp.conj(G0_mu_nu)[:, None] * G0_mu_nu[None, :])
			vol_scale = jnp.asarray(1.0 / float(wfn.cell_volume), dtype=jnp.float64)
			
			# HEAD INJECTION FOR V: V_qmunu at q=0 gets the bare Coulomb head
			# V_Γ ← V_Γ + (vc0_mean / Ω) × ζ*(0) ⊗ ζ(0)
			# This is ALWAYS applied regardless of do_screened
			V_qmunu = V_qmunu.at[0, 0, 0, 0, 0, 0, :, :].add((vc0_mean * vol_scale) * outer_u)
			
			# HEAD INJECTION FOR W (screened): W at q=0 gets the screened head wcoul0
			# Only applied when do_screened=True since W_q only exists then
			if do_screened:
				W_q = W_q.at[0, 0, 0, 0, :, 0, :].add((wcoul0 * vol_scale) * outer_u)

	# ============================================================================
	# EXTRACT BARE V AND SCREENED W INTERACTIONS
	# ============================================================================
	# V_qmunu: bare Coulomb, shape [nfreq, npol1, npol2, nkx, nky, nkz, nrmu1, nrmu2]
	# W_q:     screened Coulomb, shape [nkx, nky, nkz, nfreq, nrmu, npol, nrmu]
	# We extract the static (freq=0) component and reshape to (nrmu1, nrmu2, nkx, nky, nkz)
	# ============================================================================
	# Skip V_mu_nu extraction if already set by chunked path (check shape)
	need_extract_V = False
	try:
		_ = V_mu_nu.shape  # Check if V_mu_nu exists
		if V_mu_nu.shape != (meta.n_rmu, meta.n_rmu, meta.nkx, meta.nky, meta.nkz):
			need_extract_V = True
	except NameError:
		need_extract_V = True
	
	if need_extract_V:
		V_mu_nu = jnp.asarray(V_qmunu)[0, 0, 0]  # (nkx,nky,nkz,nrmu1,nrmu2) from bare V
		V_mu_nu = V_mu_nu.transpose(3, 4, 0, 1, 2)  # → (nrmu1,nrmu2,nkx,nky,nkz)
	
	# Apply sharding to V_mu_nu and W_mu_nu
	if sh is None and mesh_xy is not None:
		sh = make_shardings(mesh_xy)
	
	with mesh_xy:
		V_mu_nu = jax.lax.with_sharding_constraint(V_mu_nu, sh.V_shard)
	
	if do_screened:  # do_screened=True: use W for exchange
		W_mu_nu = W_q[:,:,:,0,:,0,:]  # (nkx,nky,nkz,nrmu1,nrmu2) from screened W
		W_mu_nu = W_mu_nu.transpose(3, 4, 0, 1, 2)  # → (nrmu1,nrmu2,nkx,nky,nkz)
		with mesh_xy:
			W_mu_nu = jax.lax.with_sharding_constraint(W_mu_nu, sh.V_shard)
	else:
		W_mu_nu = V_mu_nu  # For bare exchange, W = V (already sharded)

	# All nband bands are passed to the pipeline for the COH resolution of identity.
	# The sigma window slicing (to nb_sigma bands) is done inside the pipeline
	# for the final projection step only.
	valence_slice = slice(b0, b2)
	# psi_l/psi_lT and psi_r/psi_rT: all nband bands (no slicing here)

	if sh is None and mesh_xy is not None:
		sh = make_shardings(mesh_xy)

	# Create Gij_static: (nk, nb_sigma, nb_sigma) with diagonal 0:nelec set to 1.0
	# This is the static COHSEX Green's function (projector onto occupied states)
	# Sized for psi_l which has bands (b0, b3) = sigma window
	nband_full = int(b4 - b0)  # All bands from 0 to nband (for COH G_RI)
	nb_sigma = int(b3 - b0)    # Sigma window size (for SX Green's function + projection)
	nk = meta.nk_tot
	
	# Gij_static sized for psi_l (sigma window), NOT psi_coh (all bands)
	Gij_static = jnp.zeros((nk, nb_sigma, nb_sigma), dtype=jnp.complex128)
	nelec = int(wfn.nelec)
	# Set diagonal elements for occupied bands to 1.0
	occ_diag = jnp.arange(min(nelec, nb_sigma))
	Gij_static = Gij_static.at[:, occ_diag, occ_diag].set(1.0 + 0.0j)
	# Replicate Gij_static across all devices
	Gij_shard = NamedSharding(mesh_xy, P(None, None, None))
	Gij_static = jax.device_put(Gij_static, Gij_shard)

	# Jitted pipeline with explicit shardings along rmu/rnu XY
	# New signature: psi_l (SX), psi_coh (COH), psi_proj (projection), W, V, V0, Gij
	# psi_l: (b0, b3) sigma window for SX Green's function + Hartree density
	# psi_coh: (b0, b4) all bands for COH resolution of identity
	# psi_proj: same as psi_l for final projection
	# Returns: (sigma_sx_kij, sigma_coh_kij, hartree_kmn)
	pipeline_jit = jax.jit(
		compute_sigma_pipeline_jax,
		static_argnames=('nkx', 'nky', 'nkz', 'nk_tot', 'nspinor', 'fft_vol_au', 'bispinor'),
		in_shardings=(
			sh.XT_shard, sh.Y_shard,     # psi_l for SX
			sh.XT_shard, sh.Y_shard,     # psi_coh for COH
			sh.X_shard, sh.YT_shard,     # psi_proj for projection
			sh.V_shard, sh.V_shard,      # W_mu_nu, V_mu_nu
			sh.xy_shard, Gij_shard,      # V0_munu, Gij_static
		),
		out_shardings=(sh.out_shard, sh.out_shard, sh.out_shard),
	)

	with mesh_xy:
			with timing.section("cohsex_jax.pipeline"):
				sigma_sx_kbar_ij_jax, sigma_coh_kbar_ij_jax, hartree_kbar_ij_jax = pipeline_jit(
					psi_lT,    # psi_l for SX (sigma window, b0..b3)
					psi_l,     
					psi_cohT,  # psi_coh for COH (all bands, b0..b4)
					psi_coh,
					psi_l_proj,   # psi_proj for projection (X-sharded)
					psi_lT_proj,  # psi_proj transposed (Y-sharded)
					W_mu_nu,   # (rmu1, rmu2, nkx, nky, nkz) screened Coulomb
					V_mu_nu,   # (rmu1, rmu2, nkx, nky, nkz) bare Coulomb
					v_q0_noG0_munu, # (rmu1, rmu2) sharded over (x,y)
					Gij_static, # (nk, nb_sigma, nb_sigma) replicated
					meta.nkx, meta.nky, meta.nkz, meta.nk_tot, meta.nspinor,
					float(wfn.cell_volume/np.prod(wfn.fft_grid)),
					bispinor,  # γ⁰ vertex for 4-component spinors
				)
				sigma_sx_kbar_ij_jax.block_until_ready()
				sigma_coh_kbar_ij_jax.block_until_ready()
				hartree_kbar_ij_jax.block_until_ready()


	# Initial Σ = SX + COH + Hartree (all three combined for self-consistency)
	sigma_total_full = sigma_sx_kbar_ij_jax + sigma_coh_kbar_ij_jax + hartree_kbar_ij_jax
	
	# Save INITIAL sigma for one-shot diagnostic (before self-consistency changes them)
	sigma_sx_initial = sigma_sx_kbar_ij_jax
	sigma_coh_initial = sigma_coh_kbar_ij_jax
	hartree_initial = hartree_kbar_ij_jax

	qp_band_start = int(b0)
	qp_band_stop = int(b3)
	kin_ion_path = params["kin_ion_file"]
	kin_ion_full = load_kin_ion_submatrix(kin_ion_path, qp_band_start, qp_band_stop)
	nelec = int(wfn.nelec)
	
	if self_consistent:
		# Self-consistent COHSEX: iterate until Σ converges
		# Key equations:
		#   H = H_DFT + Σ
		#   Diagonalize: H U = U ε
		#   G_ij = U @ diag(f) @ U†  where f = occupation (1 for occupied, 0 for empty)
		#   Compute new Σ_SX, Σ_COH, V_H using G_ij
		#   Σ_new = Σ_SX + Σ_COH + V_H
		# Wavefunctions stay fixed; rotation is encoded in G_ij.
		
		n_upper = nb_sigma * (nb_sigma + 1) // 2  # Upper triangle size per k-point
		
		def sigma_iteration_step(sigma_upper_flat: jax.Array) -> jax.Array:
			"""One iteration of self-consistent COHSEX.
			
			Takes/returns upper triangle of Σ (Hermitian optimization).
			"""
			# Restore full Hermitian matrix from upper triangle
			# Input is (nk * n_upper,), reshape to (nk, n_upper) then convert
			sigma_upper = sigma_upper_flat.reshape(nk, n_upper)
			sigma_full = upper_flat_to_hermitian(sigma_upper, nb_sigma)  # (nk, nb_sigma, nb_sigma)
			
			# Diagonalize H = H_DFT + Σ
			H_full = kin_ion_full + sigma_full
			H_full = 0.5 * (H_full + jnp.conj(jnp.swapaxes(H_full, -1, -2)))
			_, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_full)
			
			# Build G_ij = U @ diag(f) @ U† (projector onto occupied states)
			# f = [1,1,...,1,0,0,...,0] with nelec ones
			f_occ = (jnp.arange(nb_sigma) < nelec).astype(jnp.float64)
			Gij_new = jnp.einsum('kim,m,kjm->kij', U_full, f_occ, jnp.conj(U_full), optimize=True)
			
			# Compute new Σ using original wavefunctions but updated G_ij
			with mesh_xy:
				sigma_sx_new, sigma_coh_new, hartree_new = pipeline_jit(
					psi_lT, psi_l,      # psi_l for SX
					psi_cohT, psi_coh,  # psi_coh for COH (unchanged, uses G_RI)
					psi_l_proj, psi_lT_proj,  # psi_proj for projection (X/Y-sharded)
					W_mu_nu, V_mu_nu,
					v_q0_noG0_munu,
					Gij_new,            # Updated Green's function
					meta.nkx, meta.nky, meta.nkz, meta.nk_tot, meta.nspinor,
					float(wfn.cell_volume/np.prod(wfn.fft_grid)),
					bispinor,  # γ⁰ vertex for 4-component spinors
				)
			
			# Combine and extract upper triangle
			sigma_new = sigma_sx_new + sigma_coh_new + hartree_new
			return hermitian_to_upper_flat(sigma_new).flatten()
		
		def residual_fn(sigma_upper_flat: jax.Array) -> jax.Array:
			sigma_next = sigma_iteration_step(sigma_upper_flat)
			return sigma_next - sigma_upper_flat
		
		# Initial guess: upper triangle of Σ
		sigma0_upper = hermitian_to_upper_flat(sigma_total_full).flatten()
		
		# Run rCROP acceleration (Python loop to avoid XLA constant folding)
		result = rcrop_nojit(
			residual_fn,
			sigma0_upper,
			m=3,
			maxit=40,
			tol=1e-5,
			print_fn=print0 if meta.rank == 0 else None,
		)
		
		# Restore final Σ
		sigma_final = result.x.reshape(nk, n_upper)
		sigma_total_full = upper_flat_to_hermitian(sigma_final, nb_sigma)
		
		# Final diagonalization
		H_qp_mnk = kin_ion_full + sigma_total_full
		H_qp_mnk = 0.5 * (H_qp_mnk + jnp.conj(jnp.swapaxes(H_qp_mnk, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_qp_mnk)
		
		# Get final components for output
		f_occ = (jnp.arange(nb_sigma) < nelec).astype(jnp.float64)
		Gij_final = jnp.einsum('kim,m,kjm->kij', U_full, f_occ, jnp.conj(U_full), optimize=True)
		
		# Diagnostic: check Gij_final properties
		Gij_trace = jnp.real(jnp.trace(Gij_final[0]))
		Gij_diag = jnp.real(jnp.diagonal(Gij_final[0]))
		print(f"[Diagnostic] Gij_final trace at k=0: {float(Gij_trace):.4f} (should be {nelec})")
		print(f"[Diagnostic] Gij_final diag[:5] at k=0: {np.array(Gij_diag[:5])}")
		print(f"[Diagnostic] Gij_final diag sum at k=0: {float(jnp.sum(Gij_diag)):.4f}")
		
		# Check U unitarity: U† @ U should be identity
		UdagU = jnp.einsum('kim,kin->kmn', jnp.conj(U_full[0:1]), U_full[0:1])
		unitarity_err = jnp.max(jnp.abs(UdagU[0] - jnp.eye(nb_sigma)))
		print(f"[Diagnostic] U unitarity error at k=0: {float(unitarity_err):.2e} (should be ~0)")
		
		# Check eigenvector structure: U[i,m] = <i_DFT|m_QP>
		# For nearly-identity rotation, U[i,i] should be ~1
		U_diag = jnp.abs(jnp.diagonal(U_full[0]))
		print(f"[Diagnostic] |U| diagonal[:5] at k=0: {np.array(U_diag[:5])} (should be ~1 if no mixing)")
		print(f"[Diagnostic] |U| diagonal[25:30] at k=0: {np.array(U_diag[25:30])} (valence-cond boundary)")
		
		# First eigenvector (lowest QP state): which DFT bands contribute?
		U_col0_abs = jnp.abs(U_full[0, :, 0])  # |<i_DFT|0_QP>|
		top_contrib = jnp.argsort(U_col0_abs)[::-1][:5]  # Top 5 DFT contributors
		print(f"[Diagnostic] Lowest QP state: top DFT contributors = {np.array(top_contrib)}")
		print(f"[Diagnostic] Lowest QP state: their |U| values = {np.array(U_col0_abs[top_contrib])}")
		
		with mesh_xy:
			sigma_sx_kbar_ij_jax, sigma_coh_kbar_ij_jax, hartree_kbar_ij_jax = pipeline_jit(
				psi_lT, psi_l, psi_cohT, psi_coh, psi_l_proj, psi_lT_proj,
				W_mu_nu, V_mu_nu, v_q0_noG0_munu, Gij_final,
				meta.nkx, meta.nky, meta.nkz, meta.nk_tot, meta.nspinor,
				float(wfn.cell_volume/np.prod(wfn.fft_grid)),
			)
		
		# Diagnostic: check Hartree in DFT basis before rotation
		hartree_dft_diag = jnp.real(jnp.diagonal(hartree_kbar_ij_jax[0]))
		hartree_dft_trace = jnp.sum(hartree_dft_diag)
		print(f"[Diagnostic] Hartree DFT-basis diag[:5] (Ry): {np.array(hartree_dft_diag[:5])}")
		print(f"[Diagnostic] Hartree DFT-basis trace (Ry): {float(hartree_dft_trace):.4f}")
	else:
		# One-shot: diagonalize H = H_DFT + Σ
		H_qp_mnk = kin_ion_full + sigma_total_full
		H_qp_mnk = 0.5 * (H_qp_mnk + jnp.conj(jnp.swapaxes(H_qp_mnk, -1, -2)))
		E_full, U_full = jax.vmap(jnp.linalg.eigh, in_axes=0)(H_qp_mnk)

	energies_full_dft, _ = get_enk_bandrange(
		wfn, sym, (qp_band_start, qp_band_stop), (qp_band_start, qp_band_stop),
		nspinor=meta.nspinor,
	)
	energies_dft_ev = jnp.asarray(energies_full_dft) * ryd2ev
	energies_qp_ev = E_full * ryd2ev

	# Rotate sigma components to QP basis: Σ_QP = U† Σ U
	# U_full: (nk, nb_sigma, nb_sigma) eigenvector matrix from diagonalizing H_QP
	# U[k,i,m] = ⟨i_DFT|m_QP⟩ = component i of eigenvector m
	# σ_QP[m,n] = ⟨m_QP|σ|n_QP⟩ = Σ_{ij} conj(U[i,m]) σ[i,j] U[j,n]
	def rotate_to_qp_basis(sigma_dft, U):
		# sigma_dft: (nk, i, j) in DFT basis
		# U: (nk, i, m) where columns m are eigenvectors
		# sigma_qp = U† @ sigma @ U → output (nk, m, n) in QP basis
		return jnp.einsum('kim,kij,kjn->kmn', jnp.conj(U), sigma_dft, U, optimize=True)
	
	sigma_sx_qp = rotate_to_qp_basis(sigma_sx_kbar_ij_jax, U_full)
	sigma_coh_qp = rotate_to_qp_basis(sigma_coh_kbar_ij_jax, U_full)
	hartree_qp = rotate_to_qp_basis(hartree_kbar_ij_jax, U_full)
	
	# Diagnostic: check trace preservation after rotation
	if self_consistent:
		hartree_qp_diag = jnp.real(jnp.diagonal(hartree_qp[0]))
		hartree_qp_trace = jnp.sum(hartree_qp_diag)
		print(f"[Diagnostic] Hartree QP-basis diag[:5] (Ry): {np.array(hartree_qp_diag[:5])}")
		print(f"[Diagnostic] Hartree QP-basis trace (Ry): {float(hartree_qp_trace):.4f} (should match DFT-basis trace)")

	energies_dft_ev_host = np.array(energies_dft_ev)
	energies_qp_ev_host = np.array(energies_qp_ev)

	# Write PRE-self-consistency sigma to eqp0_noqsym (initial, one-shot values)
	sigma_sx_initial_host = np.array(sigma_sx_initial)
	sigma_coh_initial_host = np.array(sigma_coh_initial)
	hartree_initial_host = np.array(hartree_initial)
	write_sigma_to_file(
		ryd2ev * sigma_sx_initial_host, params["output_file"],
		sigma_coh_kij_eV=ryd2ev * sigma_coh_initial_host,
		hartree_kij_eV=ryd2ev * hartree_initial_host,
	)

	# Write POST-self-consistency sigma to eqp0_sc (rotated to QP basis)
	sc_output_file = None
	if self_consistent:
		sigma_sx_final_host = np.array(sigma_sx_qp)
		sigma_coh_final_host = np.array(sigma_coh_qp)
		hartree_final_host = np.array(hartree_qp)
		sc_output_file = params["output_file"].replace("eqp0", "eqp0_sc").replace(".dat", "_sc.dat")
		if sc_output_file == params["output_file"]:
			sc_output_file = params["output_file"].replace(".dat", "_sc.dat")
		write_sigma_to_file(
			ryd2ev * sigma_sx_final_host, sc_output_file,
			sigma_coh_kij_eV=ryd2ev * sigma_coh_final_host,
			hartree_kij_eV=ryd2ev * hartree_final_host,
		)

	# Write eqp1.dat-style output for self-consistent runs
	eqp1_written = None
	if self_consistent and meta.rank == 0:
		eqp1_path = os.path.join(input_dir, "eqp1.dat")
		# Generate full k-mesh in crystal coords
		nkx, nky, nkz = meta.nkx, meta.nky, meta.nkz
		
		# Compute one-shot diagonal: H_QP[n,n] = kin_ion[n,n] + sigma_total[n,n] (DFT basis)
		# Use INITIAL sigma (before self-consistency) for one-shot
		sigma_total_initial = sigma_sx_initial + sigma_coh_initial + hartree_initial
		H_oneshot_diag = np.array(jnp.real(jnp.diagonal(kin_ion_full, axis1=1, axis2=2) + 
		                                    jnp.diagonal(sigma_total_initial, axis1=1, axis2=2)))
		E_oneshot_ev = H_oneshot_diag * ryd2ev
		
		with open(eqp1_path, "w") as f:
			f.write("# kx ky kz nbands\n")
			f.write("# spin band E_DFT E_oneshot(DFT-basis) E_QP(eigh)\n")
			ik = 0
			for ikz in range(nkz):
				for iky in range(nky):
					for ikx in range(nkx):
						kx = ikx / nkx
						ky = iky / nky
						kz = ikz / nkz
						f.write(f"  {kx:.9f}  {ky:.9f}  {kz:.9f}      {nb_sigma}\n")
						for ib in range(nb_sigma):
							e_dft = float(energies_dft_ev_host[ik, ib])
							e_oneshot = float(E_oneshot_ev[ik, ib])
							e_qp = float(energies_qp_ev_host[ik, ib])
							f.write(f"       1       {ib+1}  {e_dft:14.9f}  {e_oneshot:14.9f}  {e_qp:14.9f}\n")
						ik += 1
		eqp1_written = eqp1_path
	else:
		eqp1_written = None

	# Write QP rotation matrices and eigenvalues to h5 file
	if meta.rank == 0:
		qp_rot_path = os.path.join(input_dir, "qp_wfn_rotations.h5")
		# Generate full k-mesh for output
		kpoints_full = np.array(sym.unfolded_kpts, dtype=np.float64)
		# Get reduced k-points and mapping for WFN.h5 lookup
		kpoints_reduced = np.array(wfn.kpoints, dtype=np.float64)
		kirr_to_kfull = np.array(sym.kirr_fullids, dtype=np.int32)
		# Convert eigenvalues from Rydberg to Hartree
		E_qp_hartree = np.array(E_full) / 2.0  # Ry -> Ha
		U_host = np.array(U_full)
		write_qp_rotations_h5(
			qp_rot_path,
			U_mnk=U_host,
			E_qp_nk=E_qp_hartree,
			band_start=qp_band_start,
			band_stop=qp_band_stop,
			kpoints_crys=kpoints_full,
			nkx=meta.nkx,
			nky=meta.nky,
			nkz=meta.nkz,
			kpoints_reduced=kpoints_reduced,
			kirr_to_kfull=kirr_to_kfull,
		)
	else:
		qp_rot_path = None

	# ========================================================================
	# OUTPUT FILES SUMMARY
	# ========================================================================
	print0("")
	print0("-" * 72)
	print0("  OUTPUT FILES")
	print0("-" * 72)
	print0(f"  Sigma matrix elements:  {params['output_file']}")
	if self_consistent:
		print0(f"  Sigma (SC rotated):     {sc_output_file}")
	if eqp1_written:
		print0(f"  QP energies (eqp1):     {eqp1_written}")
	if qp_rot_path:
		print0(f"  QP rotations (h5):      {qp_rot_path}")
	print0(f"  Restart arrays:         {tensors_filename}")
	print0("")

	if jax.process_index() == 0:
		timing.report(print_fn=print0, title="--- Timing ---")

	return 0


if __name__ == "__main__":
	#jax.distributed.initialize()
	raise SystemExit(main())

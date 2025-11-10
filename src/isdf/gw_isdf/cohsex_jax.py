# Standard Library imports
import os
# Force JAX to create four CPU devices before import
# os.environ['XLA_FLAGS'] = ' '.join(filter(None, [
# 	os.environ.get('XLA_FLAGS', ''),
# 	'--xla_cpu_multi_thread_eigen=true'
# ]))

os.environ.setdefault("JAX_ENABLE_X64", "1")
# Force CPU backend regardless of external environment
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import configparser
import re
import time
from contextlib import nullcontext

import numpy as np
import jax
import jax.numpy as jnp
import jax.profiler as jax_profiler
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
mesh_bands = Mesh(np.asarray(jax.devices()), ("bands",))
from ..common.wfnreader import WFNReader
#from ..common.epsreader import EPSReader
from ..common import symmetry_maps
from ..common.load_wfns import read_Gvecs_to_devices, get_sharded_wfns, get_enk_bandrange
from .get_windows import get_window_info
from .w_isdf import get_chi0_jax, get_static_w_q_jax
from .vcoul import compute_vcoul_comps_for_q, compute_V_qfullG_for_q, compute_q0_averages
from ..common.chi_from_dipole import read_dipole_h5, compute_S_omega
from ..common.epsreader import EPSReader
from .gw_file_io import (write_sigma_to_file, write_eqp_table, write_labeled_arrays_to_h5, 
                         read_labeled_arrays_from_h5, load_labeled_arrays_from_h5, 
                         save_restart_per_proc)
from .jax_fixed_point_demo import crop_family_fixed_history_map
from ..common import Meta
from ..common.gamma_matrices import gammas_sparse
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
	n_rmu = int(psi_l_rmu.shape[-1])
	n_rtot = int(psi_l_rtot.shape[-1])

	# Zero reuse buffers (donated by caller) without allocating fresh storage.
	CCT_buf = jnp.zeros_like(CCT_buf)
	ZCT_buf = jnp.zeros_like(ZCT_buf)

	def accumulate_k_pair(carry, i):
		CCT_acc, ZCT_acc = carry
		k_l = k_l_indices[i]
		k_r = k_r_indices[i]
		psi_l_rmu_k = psi_l_rmu[k_l].reshape(-1, n_rmu)
		psi_r_rmu_k = psi_r_rmu[k_r].reshape(-1, n_rmu)
		psi_l_rtot_k = psi_l_rtot[k_l].reshape(-1, n_rtot)
		psi_r_rtot_k = psi_r_rtot[k_r].reshape(-1, n_rtot)
		psi_l_rmuT_k = psi_l_rmuT[k_l].reshape(n_rmu, -1)
		psi_r_rmuT_k = psi_r_rmuT[k_r].reshape(n_rmu, -1)
		Pmu_l = jnp.einsum('ij,jk->ik', psi_l_rmuT_k, psi_l_rmu_k, optimize=True)
		Pmu_r = jnp.einsum('ij,jk->ik', psi_r_rmuT_k, psi_r_rmu_k, optimize=True)
		CCT_acc = CCT_acc + jnp.conj(Pmu_l) * Pmu_r
		P_l = jnp.einsum('ij,jk->ik', psi_l_rmuT_k, psi_l_rtot_k, optimize=True)
		P_r = jnp.einsum('ij,jk->ik', psi_r_rmuT_k, psi_r_rtot_k, optimize=True)
		ZCT_acc = ZCT_acc + jnp.conj(P_l) * P_r
		return (CCT_acc, ZCT_acc), None

	(CCT, ZCT), _ = jax.lax.scan(
		accumulate_k_pair,
		(CCT_buf, ZCT_buf),
		jnp.arange(k_l_indices.shape[0]),
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


def make_v_munu_kernel(fft_nx: int, fft_ny: int, fft_nz: int, nkx: int, nky: int, nkz: int):
	"""Factory for a jitted kernel that computes v_{mu,nu} for one q.

	Captures static FFT/k-grid sizes to keep the jitted function shape-stable.
	"""
	fx = jnp.arange(fft_nx)[None, :, None, None] / fft_nx
	fy = jnp.arange(fft_ny)[None, None, :, None] / fft_ny
	fz = jnp.arange(fft_nz)[None, None, None, :] / fft_nz
	nkx = jnp.asarray(nkx, dtype=jnp.float64)
	nky = jnp.asarray(nky, dtype=jnp.float64)
	nkz = jnp.asarray(nkz, dtype=jnp.float64)

	@partial(jax.jit)
	def kernel(zeta_q: jax.Array, qvec_wrapped: jax.Array, vcoul_comps: jax.Array, V_qfullG: jax.Array) -> jax.Array:
		zeta_q_spatial = zeta_q.reshape(zeta_q.shape[0], fft_nx, fft_ny, fft_nz)
		phase = jnp.exp(-2j * jnp.pi * (qvec_wrapped[0]/nkx * fx + qvec_wrapped[1]/nky * fy + qvec_wrapped[2]/nkz * fz))
		zeta_qG = jnp.fft.fftn(zeta_q_spatial * phase, axes=(-3, -2, -1)) # unscaled.
		zeta_qG_flat = zeta_qG[:, vcoul_comps[:, 0], vcoul_comps[:, 1], vcoul_comps[:, 2]]
		zeta_v = zeta_qG_flat * jnp.sqrt(V_qfullG)
		return jnp.einsum('mG,nG->mn', jnp.conj(zeta_v), zeta_v, optimize=True)

	return kernel


@partial(jax.jit, static_argnames=(
	"fft_nx", "fft_ny", "fft_nz", "nkx", "nky", "nkz"
))
def compute_v_munu_from_zeta(
	zeta_q: jax.Array,
	qvec_wrapped: jax.Array,
	vcoul_comps: jax.Array,
	sqrt_V_qfullG: jax.Array,
	fft_nx: int,
	fft_ny: int,
	fft_nz: int,
	nkx: int,
	nky: int,
	nkz: int,
) -> jax.Array:
	"""Compute v_{mu,nu}(q) from zeta_q(r) with explicit arguments.

	All required sizes and vectors are passed as arguments to avoid reliance on
	outer-scope state. Returns (n_rmu, n_rmu) for a single q. Expects sqrt(V_qfullG)
	to avoid recomputing the square root inside the loop.
	"""
	fx = jnp.arange(fft_nx)[None, :, None, None] / float(fft_nx)
	fy = jnp.arange(fft_ny)[None, None, :, None] / float(fft_ny)
	fz = jnp.arange(fft_nz)[None, None, None, :] / float(fft_nz)
	denx = jnp.asarray(float(nkx))
	deny = jnp.asarray(float(nky))
	denz = jnp.asarray(float(nkz))
	zeta_q_spatial = zeta_q.reshape(zeta_q.shape[0], fft_nx, fft_ny, fft_nz)
	phase = jnp.exp(-2j * jnp.pi * (
		qvec_wrapped[0] / denx * fx + qvec_wrapped[1] / deny * fy + qvec_wrapped[2] / denz * fz
	))
	zeta_qG = jnp.fft.fftn(zeta_q_spatial * phase, axes=(-3, -2, -1))
	zeta_qG_flat = zeta_qG[:, vcoul_comps[:, 0], vcoul_comps[:, 1], vcoul_comps[:, 2]]
	zeta_v = zeta_qG_flat * sqrt_V_qfullG
	return jnp.einsum('mG,nG->mn', jnp.conj(zeta_v), zeta_v, optimize=True)

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
	run with minimal host-device transfers. Ragged per-q data (vcoul
	components and sqrt(V)) are returned as tuples of device arrays.
	"""
	vcoul_comps_list: list[jax.Array] = []
	sqrt_V_list: list[jax.Array] = []
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
		_, _, qvec_wrapped_frac, vcoul_comps = compute_vcoul_comps_for_q(wfn, sym, meta, q_data.q_nonneg)
		V_qfullG = compute_V_qfullG_for_q(wfn, qvec_wrapped_frac, vcoul_comps, 0.0, do_Dmunu=do_Dmunu, sys_dim=sys_dim)
		vcoul_comps_list.append(jnp.asarray(vcoul_comps, dtype=jnp.int32))
		sqrt_V_list.append(jnp.sqrt(V_qfullG))

	if not k_l_list:
		return SimpleNamespace(
			num_q=0,
			k_l_indices=jnp.zeros((0, 0), dtype=jnp.int32),
			k_r_indices=jnp.zeros((0,), dtype=jnp.int32),
			q_nonneg=jnp.zeros((0, 3), dtype=jnp.int32),
			q_wrapped=jnp.zeros((0, 3), dtype=jnp.int32),
			iq_indices=jnp.zeros((0,), dtype=jnp.int32),
			vcoul_comps=tuple(),
			sqrt_V_qfullG=tuple(),
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
		vcoul_comps=tuple(vcoul_comps_list),
		sqrt_V_qfullG=tuple(sqrt_V_list),
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
			nb = int(wfn.nbands)
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
		params = {
			"restart": getb("restart", fallback=True),
			"x_only": getb("x_only", fallback=False),
			"do_screened": getb("do_screened", fallback=True),
			"bispinor": getb("bispinor", fallback=False),
			# Source for wcoul0 small-q head average: 'epshead' or 's_tensor'
			"wcoul0_source": get("wcoul0_source", fallback="s_tensor").strip().lower(),
			"wfn_file": get("wfn_file", fallback="WFN.h5"),
			"centroids_file": get("centroids_file", fallback="centroids_frac.txt"),
			"output_file": get("output_file", fallback="eqp0_noqsym.dat"),
			"self_consistent": getb("self_consistent", fallback=False),
			"kin_ion_file": get("kin_ion_file", fallback="kin_ion.h5"),
			"eqp_output_file": get("eqp_output_file", fallback="eqp.dat"),
			"nval": geti("nval", fallback=5),
			"ncond": geti("ncond", fallback=5),
			"nband": geti("nband", fallback=100),
			"sys_dim": geti("sys_dim", fallback=2),
			"profile_qloop": getb("profile_qloop", fallback=False),
			"profile_trace_dir": get("profile_trace_dir", fallback=None),
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
			"profile_qloop": False,
			"profile_trace_dir": None,
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

	##########################################
	# Clean idiomatic distributed computation
	##########################################
	@partial(jax.jit)
	def compute_CCT_ZCT_for_q(k_l_indices, k_r_indices, psi_l_rmu, psi_r_rmu, psi_l_rtot, psi_r_rtot, psi_l_rmuT, psi_r_rmuT):
		"""Exact match to original cohsex_isdf.py physics - direct accumulation"""
		# Derive sizes from array shapes
		n_rmu = psi_l_rmu.shape[-1]
		n_rtot = psi_l_rtot.shape[-1]
		def accumulate_k_pair(carry, i):
			CCT_acc, ZCT_acc = carry
			k_l, k_r = k_l_indices[i], k_r_indices[i]
			# Extract wavefunctions for this k-point pair
			psi_l_rmu_k = psi_l_rmu[k_l].reshape(-1, n_rmu)	  # (nb*nspinor, n_rmu)
			psi_r_rmu_k = psi_r_rmu[k_r].reshape(-1, n_rmu)	  # (nb*nspinor, n_rmu)
			psi_l_rtot_k = psi_l_rtot[k_l].reshape(-1, n_rtot)   # (nb*nspinor, n_rtot)
			psi_r_rtot_k = psi_r_rtot[k_r].reshape(-1, n_rtot)   # (nb*nspinor, n_rtot)
			psi_l_rmuT_k = psi_l_rmuT[k_l].reshape(n_rmu, -1)
			psi_r_rmuT_k = psi_r_rmuT[k_r].reshape(n_rmu, -1)
			
			Pmu_l = jnp.einsum('ij,jk->ik',psi_l_rmuT_k, psi_l_rmu_k, optimize=True)  # (n_rmu, n_rmu)
			Pmu_r = jnp.einsum('ij,jk->ik',psi_r_rmuT_k, psi_r_rmu_k, optimize=True)  # (n_rmu, n_rmu)
			CCT_acc = CCT_acc + jnp.conj(Pmu_l) * Pmu_r   # Direct accumulation!
			P_l = jnp.einsum('ij,jk->ik',psi_l_rmuT_k, psi_l_rtot_k, optimize=True)   # (n_rmu, n_rtot)
			P_r = jnp.einsum('ij,jk->ik',psi_r_rmuT_k, psi_r_rtot_k, optimize=True)   # (n_rmu, n_rtot)
			ZCT_acc = ZCT_acc + jnp.conj(P_l) * P_r	   # Direct accumulation!
			return (CCT_acc, ZCT_acc), None
		# Initialize accumulators
		CCT_init = jnp.zeros((n_rmu, n_rmu), dtype=jnp.complex128)
		ZCT_init = jnp.zeros((n_rmu, n_rtot), dtype=jnp.complex128)
		k_indices = jnp.arange(k_l_indices.shape[0], dtype=jnp.int32)
		(CCT, ZCT), _ = jax.lax.scan(accumulate_k_pair, (CCT_init, ZCT_init), k_indices)
		return CCT, ZCT

	# Main q-point loop
	for q_idx, q_data in enumerate(q_data_iter):
		k_l_indices = q_data.k_l_indices
		k_r_indices = q_data.k_r_indices
		qvec_nonneg = q_data.q_nonneg
		iq_cpu = q_data.q_index
		qvec = jnp.asarray(q_data.q_wrapped, dtype=jnp.float64)
		_, _, qvec_wrapped, vcoul_comps = compute_vcoul_comps_for_q(wfn, sym, meta, qvec_nonneg)
		V_qfullG = compute_V_qfullG_for_q(
			wfn,
			qvec_wrapped,
			vcoul_comps,
			0.0,
			do_Dmunu=bispinor,
			sys_dim=sys_dim,
		)
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
		try:
			if sh is not None:
				CCT = jax.lax.with_sharding_constraint(CCT, sh.replicated_2)
				ZCT = jax.lax.with_sharding_constraint(ZCT, sh.xy_shard)
			print("CCT sharding:", jax.sharding.get_array_sharding(CCT))
			print("ZCT sharding:", jax.sharding.get_array_sharding(ZCT))
		except Exception:
			pass
		# lstsq solve with optimal sharding (Y over longer rtot dimension)
		CCT = CCT + 1e-8 * jnp.mean(jnp.real(jnp.diag(CCT))) * jnp.eye(CCT.shape[0], dtype=CCT.dtype)
		CCT_cholesky = jax.scipy.linalg.cho_factor(CCT)
		# should make this parallel over xy in ZCT, CCT_cholesky replicated
		zeta_q = jax.scipy.linalg.cho_solve(CCT_cholesky, ZCT, overwrite_b=True)

		# Reshard to be sharded over ALL processors in rmu dimension (1D mesh)
		if sh is not None:
			zeta_q = jax.lax.with_sharding_constraint(zeta_q, sh.xy0_2)
		S_q_local = compute_Sq_from_zeta(zeta_q)
		if sh is not None:
			S_q_local = jax.lax.with_sharding_constraint(S_q_local, sh.x0y1_2)
		qx, qy, qz = (int(v) for v in qvec_nonneg)
		S_qmunu = S_qmunu.at[qx, qy, qz, :, :].set(S_q_local)

		# Reshape zeta_q: (n_rmu, n_rtot) → (n_rmu, nx, ny, nz)
		zeta_q_spatial = zeta_q.reshape(meta.n_rmu, *meta.fft_grid)
		# Phase removal and FFT
		fx = jnp.arange(meta.fft_grid[0])[None, :, None, None] / meta.fft_grid[0]
		fy = jnp.arange(meta.fft_grid[1])[None, None, :, None] / meta.fft_grid[1]
		fz = jnp.arange(meta.fft_grid[2])[None, None, None, :] / meta.fft_grid[2]
		kgrfloat = jnp.asarray(kgrid, dtype=jnp.float64)
		phase = jnp.exp(-2j * jnp.pi * (qvec[0] / kgrfloat[0] * fx + qvec[1] / kgrfloat[1] * fy + qvec[2] / kgrfloat[2] * fz))
		zeta_q_spatial = zeta_q_spatial * phase
		zeta_qG = jnp.fft.fftn(zeta_q_spatial, axes=(-3, -2, -1))
		zeta_qG_flat = zeta_qG[:, vcoul_comps[:, 0], vcoul_comps[:, 1], vcoul_comps[:, 2]]

		# Compute v[mu@X,nu@Y] = (zeta_qG .* sqrt(V))^H @ (zeta_qG .* sqrt(V)) with 2D sharding over (x,y)
		# This keeps G sharded along Y inside the jitted matmul
		if sh is not None:
			zeta_qG_flat = jax.lax.with_sharding_constraint(zeta_qG_flat, sh.xy_shard)

		@partial(
			jax.jit,
			in_shardings=(sh.xy_shard if sh is not None else None, sh.y_shard_vec if sh is not None else None),
			out_shardings=sh.xy_shard if sh is not None else None,
		)
		def v_munu_matmul(zflat, vmask):
			zeta_v = zflat * jnp.sqrt(vmask)
			return jnp.einsum('mG,nG->mn', jnp.conj(zeta_v), zeta_v, optimize=True)

		v_munu = v_munu_matmul(zeta_qG_flat, V_qfullG)

		#V_weighted = V_qfullG_masked[None, :] * zeta_qG_flat
		#V_qmunu_q = jnp.conj(zeta_qG_masked) @ V_weighted.T
		# Store result into sharded JAX array at this q-point
		qx, qy, qz = (int(v) for v in qvec_nonneg)
		V_qmunu = V_qmunu.at[0, 0, 0, qx, qy, qz, :, :].set(v_munu)
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
	"""Pure JAX pipeline: returns sigma_kij (nk, nb, nb).
	Uses psi_l for building G (valence-only) and psi_r for projection to bands."""
	G_k = get_G_mu_nu_jax(psi_l_rmuT_X, psi_l_rmu_Y)
	G_R = get_G_R_jax(G_k, nkx, nky, nkz)
	sigma_k_munu = get_sigma_x_mu_nu_jax(G_R, V_mu_nu, nk_tot)
	sigma_kij = get_sigma_x_kij_jax(psi_r_rmu_X, psi_r_rmuT_Y, sigma_k_munu)

	# Hartree from q=0 component
	# rho_mu = sum_{k,n,s} |psi_l(k,n,s,mu)|^2, shape (n_rmu,)
	# Density overlap per centroid, normalized by cell volume
	#rho_mu = jnp.sum(jnp.conj(psi_l_rmu_Y) * psi_l_rmu_Y, axis=(0,1,2))
	rho_mu = jnp.einsum('knsx,knsx->x', jnp.conj(psi_l_rmu_Y), psi_l_rmu_Y, optimize=True)
	rho_mu = rho_mu * 1.0 / jnp.asarray(nk_tot, dtype=jnp.float64) # bz integration factor
	# V0(mu,nu) = V_mu_nu at (q=0)
	# Vrho(mu) = V0(mu,nu) @ rho(nu); implicit psum over Y when lowered
	#Vrho_mu = jnp.matmul(V0_munu, rho_mu) 
	Vrho_mu =jnp.einsum('xy,y->x', V0_munu, rho_mu, optimize=True)
	# psi_overlap(k,m,n,mu) = sum_s psi*_mk(μ) psi_nk(μ) using right X-sharded copy
	#psi_overlap = jnp.einsum('kmsx,knsx->kmnx', jnp.conj(psi_r_rmu_X), psi_r_rmu_X, optimize=True)
	# Hartree matrix elements per k: sum_mu psi_overlap * Vrho_mu
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
	profile_qloop = bool(params.get("profile_qloop", False))
	profile_trace_dir = params.get("profile_trace_dir")

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

	# restart: if True, read interp. vectors and V_qmunu from file
	restart = params["restart"]
	x_only = params["x_only"]
	do_screened = params["do_screened"]
	global bispinor
	bispinor = params["bispinor"]
	if x_only and do_screened:
		raise ValueError("x_only and do_screened cannot both be True")

	meta = Meta.from_system(wfn, sym, nval, ncond, nband, _n_rmu, bispinor)
	meta.rank = jax.process_index()
	meta.n_proc = jax.process_count()
	meta.sys_dim = sys_dim
	meta.bispinor = bispinor
	band = meta.band_ranges
	b0, b1, b2, b3, b4 = meta.band_edges
	nvplussigrange = band.val_plus_sigma
	ncplussigrange = band.cond_plus_sigma
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


	# Initialize timing accumulators for both fresh and restart flows
	zeta_secs = 0.0
	chiw_secs = 0.0
	if not restart:
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

		####################################
		# 2.) Explicit q-loop: build zeta_q,mu(r), S_q, and V_q,mu,nu
		####################################
		_t_zeta_start = time.perf_counter()
		# Energies and weights for windows (kept as before, last entry is just sigma range)
		enk_l, weights_l = get_enk_bandrange(wfn, sym, brange_l, (b1,b3))
		enk_r, weights_r = get_enk_bandrange(wfn, sym, brange_r, (b1,b3))

		# Allocate sharded outputs
		V_qmunu = jnp.zeros((1, meta.npol, meta.npol, meta.nkx, meta.nky, meta.nkz, meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
		S_qmunu = jnp.zeros((meta.nkx, meta.nky, meta.nkz, meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
		V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, sh.x6y7_8)
		S_qmunu = jax.lax.with_sharding_constraint(S_qmunu, sh.x3y4_5)
		v_q0_noG0_munu = jnp.zeros((meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
		v_q0_noG0_munu = jax.lax.with_sharding_constraint(v_q0_noG0_munu, sh.xy_shard)
		G0_mu_nu = None
		# Reusable buffers for the expensive CCT/ZCT accumulation.
		CCT_buf = jnp.zeros((meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
		ZCT_buf = jnp.zeros((meta.n_rmu, psi_l_rtot_Y.shape[-1]), dtype=jnp.complex128)
		if mesh_xy is not None:
			CCT_buf = jax.lax.with_sharding_constraint(CCT_buf, sh.replicated_2)
			ZCT_buf = jax.lax.with_sharding_constraint(ZCT_buf, sh.xy_shard)

		# No local kernel definitions; use module-scope jitted helpers

		#################################################################################
		# Main q-point loop
		# zeta_q(r) is ephemeral inside the loop
		################################################################################
		if profile_trace_dir:
			profile_trace_dir = _resolve_path(profile_trace_dir)
			os.makedirs(profile_trace_dir, exist_ok=True)
		trace_context = jax_profiler.trace(profile_trace_dir, create_perfetto_link=False) if profile_trace_dir else nullcontext()
		with trace_context:
			q_cache = build_q_coulomb_cache(wfn, sym, meta, do_Dmunu=bispinor, sys_dim=sys_dim, mesh_xy=mesh_xy)
			q_profile_samples: list[tuple[int, float]] = []
			cct_samples: list[float] = []
			zeta_samples: list[float] = []
			vmunu_samples: list[float] = []
			for q_idx in range(q_cache.num_q):
				k_l_indices = q_cache.k_l_indices[q_idx]
				k_r_indices = q_cache.k_r_indices
				qvec_nonneg = q_cache.q_nonneg[q_idx]
				iq_cpu = int(q_cache.iq_indices[q_idx])
				qvec = jnp.asarray(q_cache.q_wrapped[q_idx], dtype=jnp.float64)
				vcoul_comps = q_cache.vcoul_comps[q_idx]
				sqrt_V_qfullG = q_cache.sqrt_V_qfullG[q_idx]
				_t_q_start = time.perf_counter() if profile_qloop else None
				annotation = (
					jax_profiler.StepTraceAnnotation("cohsex_q_iteration", q_index=iq_cpu, q_idx=int(q_idx))
					if profile_trace_dir else nullcontext()
				)
				with annotation:
					# Build CCT/ZCT then Cholesky solve for zeta_q
					_t_cct = time.perf_counter()
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
						k_r_indices
					)
					CCT_buf, ZCT_buf = CCT, ZCT
					if profile_qloop:
						CCT = CCT.block_until_ready()
						ZCT = ZCT.block_until_ready()
						cct_samples.append(time.perf_counter() - _t_cct)
						_t_zeta = time.perf_counter()
					else:
						_t_zeta = None
					zeta_q = solve_zeta_cholesky(CCT, ZCT)
					if profile_qloop:
						zeta_q = zeta_q.block_until_ready()
						zeta_samples.append(time.perf_counter() - _t_zeta)
					zeta_q = set_zeta_sharding(zeta_q, mesh_xy)
					# Compute S_q and write into S_qmunu
					qx = int(qvec_nonneg[0]); qy = int(qvec_nonneg[1]); qz = int(qvec_nonneg[2])
					# FFT to G and accumulate V_q,mu,nu using explicit-arg kernel
					_t_v = time.perf_counter()
					v_munu = compute_v_munu_from_zeta(
						zeta_q,
						qvec,
						vcoul_comps,
						sqrt_V_qfullG,
						int(meta.fft_grid[0]), int(meta.fft_grid[1]), int(meta.fft_grid[2]),
						int(meta.nkx), int(meta.nky), int(meta.nkz),
					)
					V_qmunu = V_qmunu.at[0, 0, 0, qx, qy, qz, :, :].set(v_munu)
					if profile_trace_dir or profile_qloop:
						v_munu = v_munu.block_until_ready()
						if profile_qloop:
							vmunu_samples.append(time.perf_counter() - _t_v)

				if profile_trace_dir and hasattr(jax_profiler, "step_end"):
					jax_profiler.step_end()

				if _t_q_start is not None:
					q_profile_samples.append((iq_cpu, time.perf_counter() - _t_q_start))

				# Capture the q=0 Coulomb (with G=0 removed) for Hartree, and the head vector u
				if int(qx) == 0 and int(qy) == 0 and int(qz) == 0:
					v_q0_noG0_munu = v_munu  # V_qfullG already has head zeroed; reuse v_munu
					# Head-direction vector u in ISDF index space (unnormalized):
					# Extract u = zeta_{mu}(G=0) using the SAME FFT convention as used in V/W build.
					# Build zeta_G and pick the G=0 index from vcoul_comps.
					fft_nx, fft_ny, fft_nz = int(meta.fft_grid[0]), int(meta.fft_grid[1]), int(meta.fft_grid[2])
					z_sp = zeta_q.reshape(zeta_q.shape[0], fft_nx, fft_ny, fft_nz)
					phase = jnp.ones((1, fft_nx, fft_ny, fft_nz), dtype=jnp.complex128)  # q=0
					z_G = jnp.fft.fftn(z_sp * phase, axes=(-3, -2, -1))
					# find index of G=(0,0,0) in vcoul_comps (use NumPy for robustness outside jit)
					vc_np = np.asarray(vcoul_comps)
					g0_mask_np = (vc_np[:, 0] == 0) & (vc_np[:, 1] == 0) & (vc_np[:, 2] == 0)
					if not np.any(g0_mask_np):
						g0_idx = int(np.argmin(np.sum(vc_np * vc_np, axis=1)))
					else:
						g0_idx = int(np.where(g0_mask_np)[0][0])
					G0_mu_nu = z_G[:, vc_np[g0_idx, 0], vc_np[g0_idx, 1], vc_np[g0_idx, 2]]


				print(f"qpoint {iq_cpu} done")

			if profile_qloop and meta.rank == 0:
				if q_profile_samples:
					elapsed = np.asarray([sample[1] for sample in q_profile_samples], dtype=np.float64)
					print0(f"q-loop timing: mean={elapsed.mean()*1e3:.2f} ms, max={elapsed.max()*1e3:.2f} ms over {len(elapsed)} q-points")
				if cct_samples:
					cct_arr = np.asarray(cct_samples, dtype=np.float64)
					print0(f"  CCT/ZCT build: mean={cct_arr.mean()*1e3:.2f} ms, max={cct_arr.max()*1e3:.2f} ms")
				if zeta_samples:
					zeta_arr = np.asarray(zeta_samples, dtype=np.float64)
					print0(f"  zeta solve: mean={zeta_arr.mean()*1e3:.2f} ms, max={zeta_arr.max()*1e3:.2f} ms")
				if vmunu_samples:
					vmunu_arr = np.asarray(vmunu_samples, dtype=np.float64)
					print0(f"  v_munu FFT build: mean={vmunu_arr.mean()*1e3:.2f} ms, max={vmunu_arr.max()*1e3:.2f} ms")

		# Sharded psi variants outside q-loop
		with mesh_xy:
			psil_Y = jax.device_put(psi_l_rmu_Y, sh.y3_4)
			psilT_X = jax.lax.with_sharding_constraint(psil_Y.transpose(0, 2, 3, 1), sh.x2_4)
			psir_X = jax.device_put(psi_r_rmu_Y, sh.x3_4)
			psirT_Y = jax.lax.with_sharding_constraint(psir_X.transpose(0, 2, 3, 1), sh.y2_4)

		# Preserve full-sized left wavefunctions (b3) for rotation in SCF loop
		psi_l_full = psil_Y
		psi_lT_full = psilT_X
		# Define slices and trimmed copies for the one-shot sigma pipeline
		valence_slice = slice(b0, b2)
		nb_valence = int(b2 - b0)
		nb_sigma = int(b3 - b0)
		psi_l = psi_l_full#[:, valence_slice, :, :]
		psi_lT = psi_lT_full#[:, :, :, valence_slice]
		psi_r = psir_X#[:, :nb_sigma, :, :]
		psi_rT = psirT_Y#[:, :, :, :nb_sigma]

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
		zeta_secs = time.perf_counter() - _t_zeta_start

	elif restart and not x_only: # TODO update for jax
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
		# Ensure energies are plain arrays for JAX
		enk_v_arr = jnp.asarray(getattr(enk_l, 'data', enk_l))
		enk_c_arr = jnp.asarray(getattr(enk_r, 'data', enk_r))
		# Four wavefunction copies and shardings for low-comm G and Sigma construction:
		# psi_lT: (nk, ns, rmu, nb) XT_shard; psi_l: (nk, nb, ns, rmu) Y_shard
		# psi_r:  (nk, nb, ns, rmu) X_shard;  psi_rT: (nk, ns, rmu, nb) YT_shard
		# We reshard psi_rT to XT for chi0 so both G_v and G_c use {mu_X, nu_Y} without communication.
		window_pairs = get_window_info(epsq, wfn, nband_max=nband)
		_t_chiw_start = time.perf_counter()
		chi0 = get_chi0_jax(psi_lT, psi_l, psi_r, psi_rT, enk_v_arr, enk_c_arr, window_pairs, meta, mesh_xy)
		# Compute static W under k_XY sharding (S_qmunu included but unused for now)
		W_q = get_static_w_q_jax(V_qmunu, chi0, None, meta, mesh_xy)
		W_q.block_until_ready()
		chiw_secs = time.perf_counter() - _t_chiw_start
		# Compute static W under k_XY sharding (S_qmunu included but unused for now)
		#W_q = get_static_w_q_jax(V_qmunu, chi0, S_qmunu, meta, mesh_xy)

 
	# Compute q=0 averages (after restart/loop) and inject head-averages
	vc0_mean, wcoul0, wcoul0_source = determine_wcoul0(params, input_dir, wfn, sym, meta, print0)
	print0(f"wcoul0 source: {wcoul0_source}")
	# Inject head-averages after building W/V
	# - Add wcoul0 * u u^† to W at q=0 (if screened)
	# - Add vcoul0 * u u^† to V at q=0 for Sigma_X

	outer_u = (G0_mu_nu[:, None] * jnp.conj(G0_mu_nu)[None, :])
	# Scale by 1/Volume to match V/W units used in μν-space (see compute_V_qfullG_for_q)
	vol_scale = jnp.asarray(1.0 / float(wfn.cell_volume), dtype=jnp.float64)
	# For Sigma_X, use the simple head injection as before:
	# V_Γ ← V_Γ + (vcoul0/Ω) · u u†
	V_qmunu = V_qmunu.at[0, 0, 0, 0, 0, 0, :, :].add((vc0_mean * vol_scale) * outer_u)
	# For screened W, apply the same simple head injection when enabled
	if do_screened:
		W_q = W_q.at[0, 0, 0, 0, :, 0, :].add((wcoul0 * vol_scale) * outer_u)

	# Prepare V_mu_nu (nrmu1,nrmu2,nkx,nky,nkz) from LabeledArray V_qmunu data
	# V_qmunu has axes [nfreq, npol1, npol2, nkx, nky, nkz, nrmu1, nrmu2]
	# We need V_mu_nu = V_qmunu[0,0,0,:,:,:, :, :].transpose(6->0,7->1,3->2,4->3,5->4)
	V_mu_nu = jnp.asarray(V_qmunu)[0, 0, 0]  # (nkx,nky,nkz,nrmu1,nrmu2)
	if do_screened:
		V_mu_nu = W_q[:,:,:,0,:,0,:]
	V_mu_nu = V_mu_nu.transpose(3, 4, 0, 1, 2)  # (nrmu1,nrmu2,nkx,nky,nkz)

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

	with mesh_xy:
		_t_pipe_start = time.perf_counter()
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
	pipe_secs = time.perf_counter() - _t_pipe_start
	#vhartree_k_mu = jnp.einsum('xy,y->x', v_q0_noG0_munu, rho_k_mu, optimize=True)
	#hartree_kmn = jnp.einsum('kmsx,x,knsx->kmn', jnp.conj(psi_r), vhartree_k_mu, psi_r, optimize=True)
	#rho_k_mu = jnp.transpose(rho_k_mu, (1,0)).reshape(-1,meta.nkx,meta.nky,meta.nkz)
	#rho_R_mu = jnp.fft.ifftn(rho_k_mu, axes=(3,2,1), norm='ortho')
	#print(rho_k_mu[:4])
	#print(vhartree_k_mu[:4])
	#print(hartree_kmn[:4])


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

	write_sigma_to_file(ryd2ev * sigma_x_kbar_ij, params["output_file"], hartree_kij=ryd2ev * hartree_kbar_ij)
	#write_eqp_table(energies_dft_ev_host, energies_qp_ev_host, params["eqp_output_file"])
	write_eqp_table(energies_dft_ev_host, np.diagonal(H_qp_mnk_host, axis1=-2, axis2=-1), params["eqp_output_file"])
	if meta.rank == 0:
		summarize_hermitian_matrix("sigma_x", sigma_x_kbar_ij, print_fn=print0)
		summarize_hermitian_matrix("hartree", hartree_kbar_ij, print_fn=print0)
		summarize_hermitian_matrix("H_qp", H_qp_mnk_host, print_fn=print0)
	# Timing report
	if jax.process_index() == 0:
		print0("--- Timing (seconds) ---")
		print0(f"zeta/V build: {zeta_secs:.3f}")
		if do_screened:
			print0(f"chi0 + W_q: {chiw_secs:.3f}")
		else:
			print0("chi0 + W_q: n/a")
		print0(f"pipeline_jit: {pipe_secs:.3f}")

	# Later stages of this project will iterate this workflow so that the COHSEX
	# potential feeds back into updated wavefunctions (self-consistent COHSEX)
	# and eventually into a full quasiparticle self-consistent GW cycle.
	return 0


if __name__ == "__main__":
	#jax.distributed.initialize()
	raise SystemExit(main())

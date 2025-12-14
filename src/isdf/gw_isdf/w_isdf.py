import os
from pathlib import Path
import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.experimental import compilation_cache as jax_compilation_cache
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import numpy as np

from ..common import Meta, jax_profile
from ..common.gamma_matrices import gammas_sparse
from ..common.gpu_utils import cp, xp
from ..common.wfnreader import WFNReader


_MESH_SHARD_REGISTRY: dict[int | str, dict[str, NamedSharding | None]] = {}
_COMPILATION_CACHE_READY = False
_ENERGY_TOL = 1e-6


def _ensure_compilation_cache():
	global _COMPILATION_CACHE_READY
	if _COMPILATION_CACHE_READY:
		return
	cache_dir = os.environ.get("ISDF_JAX_CACHE_DIR")
	if cache_dir is None:
		base_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
		cache_dir = os.path.join(base_cache, "isdf_jax_compilation")
	cache_path = Path(cache_dir).expanduser()
	cache_path.mkdir(parents=True, exist_ok=True)
	try:
		jax_compilation_cache.set_cache_dir(cache_path)
		_COMPILATION_CACHE_READY = True
	except Exception:
		# If compilation cache is not supported, we simply proceed without it.
		_COMPILATION_CACHE_READY = True


def _register_mesh_shardings(mesh_xy: Mesh | None) -> str | int:
	"""Ensure shardings for a mesh are cached and return the registry key."""
	if mesh_xy is None:
		key = "none"
		if key not in _MESH_SHARD_REGISTRY:
			_MESH_SHARD_REGISTRY[key] = {
				"xt": None,
				"y_shard": None,
				"x_shard": None,
				"yt": None,
				"y": None,
				"out": None,
				"chiR": None,
				"gv": None,
				"gc": None,
			}
		return key

	key = id(mesh_xy)
	if key not in _MESH_SHARD_REGISTRY:
		_MESH_SHARD_REGISTRY[key] = {
			"xt": NamedSharding(mesh_xy, P(None, None, 'x', None)),
			"y_shard": NamedSharding(mesh_xy, P(None, None, None, 'y')),
			"x_shard": NamedSharding(mesh_xy, P(None, None, None, 'x')),
			"yt": NamedSharding(mesh_xy, P(None, None, 'y', None)),
			"out": NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y')),
			"chiR": NamedSharding(mesh_xy, P('x', 'y', None, None, None)),
			"gv": NamedSharding(mesh_xy, P(None, None, 'x', None, 'y')),
			"gc": NamedSharding(mesh_xy, P(None, None, 'y', None, 'x')),
		}
	return key


def _band_slice_metadata(
	energies: np.ndarray,
	emin: float,
	emax: float,
	upper_inclusive: bool = True,
	lower_relaxed: bool = False,
):
	"""Return per-k slice metadata covering all bands within [emin, emax].

	Args:
		energies: Array of shape (nk, nb) with per-k band energies.
		emin/emax: Energy window bounds.
		upper_inclusive: If False, treat emax as an open boundary to avoid overlaps.
		lower_relaxed: Allow a small tolerance below emin (for the first window) to avoid
			losing states to numerical round-off.

	Returns:
		clipped_start (nk,): actual slice origin after enforcing fixed max span
		spans (nk,): span in bands between first and last included band (inclusive)
		mask_arr (nk,max_span): boolean mask aligned to the clipped slice
		max_span: global maximum span length (used for fixed slice length)
	"""
	if energies is None or energies.size == 0:
		nk = 0 if energies is None else energies.shape[0]
		zero = np.zeros(nk, dtype=np.int32)
		mask = np.zeros((nk, 1), dtype=bool)
		return zero, zero, mask, 0
	lower_bound = emin - (_ENERGY_TOL if lower_relaxed else 0.0)
	if upper_inclusive:
		upper_bound = emax + _ENERGY_TOL
		mask = (energies >= lower_bound) & (energies <= upper_bound)
	else:
		mask = (energies >= lower_bound) & (energies < emax)
	nk, nb = mask.shape
	starts = np.zeros(nk, dtype=np.int32)
	spans = np.zeros(nk, dtype=np.int32)
	slice_masks: list[np.ndarray] = []
	max_span = 0
	for k in range(nk):
		idx = np.nonzero(mask[k])[0]
		if idx.size == 0:
			slice_masks.append(np.zeros(0, dtype=bool))
			continue
		start = int(idx[0])
		end = int(idx[-1])
		span = end - start + 1
		starts[k] = start
		spans[k] = span
		slice_masks.append(mask[k, start:start + span])
		if span > max_span:
			max_span = span
	if max_span == 0:
		clipped_start = np.zeros_like(starts)
		return clipped_start, spans, np.zeros((nk, 1), dtype=bool), 0

	slice_cap = max(nb - max_span, 0)
	clipped_start = np.minimum(starts, slice_cap).astype(np.int32)
	offsets = (starts - clipped_start).astype(np.int32)
	mask_arr = np.zeros((nk, max_span), dtype=bool)
	for k, sl in enumerate(slice_masks):
		if sl.size:
			off = offsets[k]
			mask_arr[k, off:off + sl.size] = sl
	return clipped_start, spans, mask_arr, max_span


def _ensure_window_band_ranges(win, enk_v_host: np.ndarray, enk_c_host: np.ndarray):
	"""Populate per-window band metadata for efficient slicing."""
	if getattr(win, "_has_band_ranges", False):
		return
	val_upper = getattr(win.val_window, "upper_inclusive", True)
	val_lower_relaxed = getattr(win.val_window, "index", 0) == 0
	val_start, val_len, val_mask, max_val_len = _band_slice_metadata(
		enk_v_host,
		win.val_window.start_energy,
		win.val_window.end_energy,
		upper_inclusive=val_upper,
		lower_relaxed=val_lower_relaxed,
	)
	cond_upper = getattr(win.cond_window, "upper_inclusive", True)
	cond_lower_relaxed = getattr(win.cond_window, "index", 0) == 0
	cond_start, cond_len, cond_mask, max_cond_len = _band_slice_metadata(
		enk_c_host,
		win.cond_window.start_energy,
		win.cond_window.end_energy,
		upper_inclusive=cond_upper,
		lower_relaxed=cond_lower_relaxed,
	)
	win.val_band_start = val_start
	win.val_band_len = val_len
	win.val_band_mask = val_mask
	win.cond_band_start = cond_start
	win.cond_band_len = cond_len
	win.cond_band_mask = cond_mask
	win.max_val_len = max_val_len
	win.max_cond_len = max_cond_len
	win._has_band_ranges = True


def _slice_along_axis(arr: jax.Array, starts: jax.Array, max_len: int, axis: int):
	"""Slice each k-row of arr along the given axis using per-row start indices."""
	if max_len <= 0:
		raise ValueError("max_len must be positive for slicing")
	if axis == 0:
		raise ValueError("Axis 0 (k-dimension) is not sliceable via _slice_along_axis")
	axis_row = axis - 1  # remove the leading k-dimension
	max_len = int(max_len)
	axis_size = arr.shape[axis]
	if max_len > axis_size:
		raise ValueError(f"Requested slice length {max_len} exceeds axis {axis} size {axis_size}")
	max_start_allowed = max(axis_size - max_len, 0)
	starts = jnp.clip(starts, 0, max_start_allowed)

	def _slice_one(row, start_idx):
		return jax.lax.dynamic_slice_in_dim(row, start_idx, max_len, axis=axis_row)

	return jax.vmap(_slice_one, in_axes=(0, 0), out_axes=0)(arr, starts)


def _as_int_array(data: np.ndarray | jax.Array | None, nk: int) -> jax.Array:
	if data is None:
		return jnp.zeros((nk,), dtype=jnp.int32)
	return jnp.asarray(data, dtype=jnp.int32)


@lru_cache(maxsize=None)
def _get_chi_kernel(mesh_key, nkx: int, nky: int, nkz: int, max_val_len: int, max_cond_len: int):
	shards = _MESH_SHARD_REGISTRY[mesh_key]
	xt_shard = shards["xt"]
	y_shard = shards["y_shard"]
	x_shard = shards["x_shard"]
	yt_shard = shards["yt"]
	out_shard = shards["out"]
	chiR_shard = shards["chiR"]
	gv_shard = shards["gv"]
	gc_shard = shards["gc"]

	@partial(
		jax.jit,
		static_argnames=("nkx", "nky", "nkz", "max_val_len", "max_cond_len"),
		in_shardings=(
			xt_shard,
			y_shard,
			x_shard,
			yt_shard,
			None,
			None,
			None, None, None, None,
			None,
			None,
			None,
			None, None, None,
			None, None, None,
		),
		out_shardings=out_shard,
	)
	def _compute(
		psi_vTX: jax.Array,
		psi_vY: jax.Array,
		psi_cX: jax.Array,
		psi_cTY: jax.Array,
		enk_v: jax.Array,
		enk_c: jax.Array,
		vmin, vmax, cmin, cmax, tau_i, z_lm, w_i,
		val_start, val_len, val_mask,
		cond_start, cond_len, cond_mask,
		nkx: int, nky: int, nkz: int,
		max_val_len: int, max_cond_len: int,
	) -> jax.Array:
		nrmu_local = psi_vTX.shape[2]
		if max_val_len == 0 or max_cond_len == 0 or tau_i.shape[0] == 0:
			chi_empty = jnp.zeros((nkx, nky, nkz, 1, nrmu_local, 1, nrmu_local), dtype=jnp.complex128)
			chi_empty = jax.lax.with_sharding_constraint(chi_empty, out_shard)
			return chi_empty

		psi_vTX_win = _slice_along_axis(psi_vTX, val_start, max_val_len, axis=3)
		psi_vY_win = _slice_along_axis(psi_vY, val_start, max_val_len, axis=1)
		psi_cX_win = _slice_along_axis(psi_cX, cond_start, max_cond_len, axis=1)
		psi_cTY_win = _slice_along_axis(psi_cTY, cond_start, max_cond_len, axis=3)
		enk_v_win = _slice_along_axis(enk_v, val_start, max_val_len, axis=1)
		enk_c_win = _slice_along_axis(enk_c, cond_start, max_cond_len, axis=1)

		val_mask_f = val_mask.astype(psi_vTX_win.dtype)
		cond_mask_f = cond_mask.astype(psi_cX_win.dtype)
		psi_vTX_win = psi_vTX_win * val_mask_f[:, None, None, :]
		psi_vY_win = psi_vY_win * val_mask_f[:, :, None, None]
		psi_cX_win = psi_cX_win * cond_mask_f[:, :, None, None]
		psi_cTY_win = psi_cTY_win * cond_mask_f[:, None, None, :]
		val_mask_complex = val_mask.astype(jnp.complex128)
		cond_mask_complex = cond_mask.astype(jnp.complex128)

		quad_w = -2.0 * z_lm * w_i * jnp.exp(-(z_lm * (cmin - vmax) - 1.0) * tau_i)

		def _k_to_R(g_k: jax.Array, flip_sign: bool) -> jax.Array:
			"""Map G(k) -> G(±R) with the same layout used in the legacy kernel."""
			g_fft = g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
			return jax.lax.cond(
				flip_sign,
				lambda x: jnp.fft.fftn(x, axes=(-3, -2, -1), norm='ortho'),
				lambda x: jnp.fft.ifftn(x, axes=(-3, -2, -1), norm='ortho'),
				g_fft,
			)

		def tau_body(itau, chi_R_acc):
			tau = tau_i[itau]
			exp_v = jnp.exp(-z_lm * tau * (vmax - enk_v_win)) * val_mask_complex
			exp_c = jnp.exp(-z_lm * tau * (enk_c_win - cmin)) * cond_mask_complex
			w_v = exp_v.astype(jnp.complex128)
			w_c = exp_c.astype(jnp.complex128)
			Gv_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_vTX_win), w_v, psi_vY_win, optimize=True)
			Gc_k = jnp.einsum('ksxm,km,kmty->ksxty', jnp.conj(psi_cTY_win), w_c, psi_cX_win, optimize=True)
			Gv_k = jax.lax.with_sharding_constraint(Gv_k, gv_shard)
			Gc_k = jax.lax.with_sharding_constraint(Gc_k, gc_shard)
			Gv_R = _k_to_R(Gv_k, flip_sign=False)
			Gc_R = _k_to_R(Gc_k, flip_sign=True)
			chi_tau = jnp.einsum('ambnxyz, bnamxyz-> mnxyz', Gc_R, Gv_R, optimize=True)
			return chi_R_acc + quad_w[itau] * chi_tau

		chi_R = jnp.zeros((nrmu_local, nrmu_local, nkx, nky, nkz), dtype=jnp.complex128)
		chi_R = jax.lax.with_sharding_constraint(chi_R, chiR_shard)
		chi_R = jax.lax.fori_loop(0, tau_i.shape[0], tau_body, chi_R)
		chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm='ortho')
		chi_q = chi_q.transpose(2, 3, 4, 0, 1)
		chi_q = chi_q[:, :, :, None, :, None, :]
		chi_q = jax.lax.with_sharding_constraint(chi_q, out_shard)
		return chi_q

	return _compute


# The routines here construct chi^0 and the screened interaction W using the
# CTSP approach in the static limit.  Once the frequency grids are restored, the
# same machinery will let us tackle full dynamical GW.


def get_chi_lm_Yt_jax(
	psi_vTX: jax.Array,
	psi_vY: jax.Array,
	psi_cX: jax.Array,
	psi_cTY: jax.Array,
	enk_v: jax.Array,
	enk_c: jax.Array,
	win,
	meta: Meta,
	mesh_xy: Mesh | None = None,
):
	"""Compute chi_lm integrated over tau using only JAX arrays.

	Args:
		psi_vTX: (nk, ns, rmu, nb)  left valence, rmu sharded on X, band is fastest
		psi_vY:  (nk, nb, ns, rmu)  right valence, rmu sharded on Y, rmu is fastest
		psi_cX:  (nk, ns, rmu, nb)  left conduction, rmu sharded on X, band is fastest
		psi_cTY: (nk, nb, ns, rmu)  right conduction, rmu sharded on Y, rmu is fastest
		enk_v: (nk, nb_v)
		enk_c: (nk, nb_c)
		win: window object with attributes (tau_i, z_lm, w_i, val_window, cond_window)
		meta: Meta with nkx,nky,nkz
		mesh_xy: optional 2D device mesh for sharding (mu on x, nu on y)

	Returns:
		chi_q: (nkx, nky, nkz, npol1=1, nrmu1, npol2=1, nrmu2) complex128
	"""
	nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
	nspinor = psi_vTX.shape[1]
	nrmu = psi_vTX.shape[2]
	nk = psi_vTX.shape[0]
	max_val_len = int(getattr(win, "max_val_len", 0))
	max_cond_len = int(getattr(win, "max_cond_len", 0))
	val_band_start = _as_int_array(getattr(win, "val_band_start", None), nk)
	val_band_len = _as_int_array(getattr(win, "val_band_len", None), nk)
	val_band_mask = jnp.asarray(getattr(win, "val_band_mask", None), dtype=jnp.bool_)
	cond_band_start = _as_int_array(getattr(win, "cond_band_start", None), nk)
	cond_band_len = _as_int_array(getattr(win, "cond_band_len", None), nk)
	cond_band_mask = jnp.asarray(getattr(win, "cond_band_mask", None), dtype=jnp.bool_)

	# Extract window params
	vmin = jnp.asarray(win.val_window.start_energy)
	vmax = jnp.asarray(win.val_window.end_energy)
	cmin = jnp.asarray(win.cond_window.start_energy)
	cmax = jnp.asarray(win.cond_window.end_energy)
	tau_i = jnp.asarray(win.tau_i, dtype=jnp.float64)
	z_lm = jnp.asarray(win.z_lm, dtype=jnp.complex128)
	w_i = jnp.asarray(win.w_i, dtype=jnp.complex128)

	_ensure_compilation_cache()
	if mesh_xy is None:
		raise ValueError("chi kernel requires mesh_xy sharding")
	mesh_key = _register_mesh_shardings(mesh_xy)
	kernel = _get_chi_kernel(mesh_key, nkx, nky, nkz, max_val_len, max_cond_len)

	with jax_profile.annotation(f"chi0_kernel[v{max_val_len}_c{max_cond_len}]"):
		chi_q = kernel(
			psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
			vmin, vmax, cmin, cmax, tau_i, z_lm, w_i,
			val_band_start, val_band_len, val_band_mask,
			cond_band_start, cond_band_len, cond_band_mask,
			nkx, nky, nkz, max_val_len, max_cond_len,
		)
	chi_q = jax.lax.with_sharding_constraint(chi_q, NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y')))
	return chi_q


def get_chi0_jax(
	psi_vTX: jax.Array,
	psi_vY: jax.Array,
	psi_cX: jax.Array,
	psi_cTY: jax.Array,
	enk_v: jax.Array,
	enk_c: jax.Array,
	windows,
	meta: Meta,
	mesh_xy: Mesh | None = None,
):
	"""Sum chi_lm over windows using JAX arrays.

	Args:
		psi_vTX: (nk, ns, rmu, nb)  left valence, rmu sharded on X, band is fastest
		psi_vY:  (nk, nb, ns, rmu)  right valence, rmu sharded on Y, rmu is fastest
		psi_cX:  (nk, ns, rmu, nb)  left conduction, rmu sharded on X, band is fastest
		psi_cTY: (nk, nb, ns, rmu)  right conduction, rmu sharded on Y, rmu is fastest
		enk_v: (nk, nb_v)
		enk_c: (nk, nb_c)
		windows: iterable of window objects
		meta: Meta
		mesh_xy: optional mesh for sharding

	Returns:
		chi_q: (nkx,nky,nkz,npol1=1,nrmu1,npol2=1,nrmu2) complex128
	"""
	enk_v_host = None
	enk_c_host = None
	mask_logs = None
	if jax.process_index() == 0:
		try:
			ener_v0 = np.asarray(jax.device_get(enk_v[0]))
			ener_c0 = np.asarray(jax.device_get(enk_c[0]))
			def _mask_bar(mask: np.ndarray) -> str:
				return ''.join('X' if bool(b) else '_' for b in mask.tolist())
			mask_logs = []
			for w_i, win in enumerate(windows):
				try:
					vmin = float(win.val_window.start_energy)
					vmax = float(win.val_window.end_energy)
					cmin = float(win.cond_window.start_energy)
					cmax = float(win.cond_window.end_energy)
					val_mask = (ener_v0 >= vmin) & (ener_v0 <= vmax)
					cond_mask = (ener_c0 >= cmin) & (ener_c0 <= cmax)
					mask_logs.append((
						f"[win {w_i}] k=0 val  : " + _mask_bar(val_mask),
						f"[win {w_i}] k=0 cond : " + _mask_bar(cond_mask),
					))
				except Exception:
					mask_logs.append(None)
		except Exception:
			mask_logs = None
	for win in windows:
		if getattr(win, "_has_band_ranges", False):
			continue
		if enk_v_host is None:
			enk_v_host = np.asarray(jax.device_get(enk_v))
		if enk_c_host is None:
			enk_c_host = np.asarray(jax.device_get(enk_c))
		_ensure_window_band_ranges(win, enk_v_host, enk_c_host)

	# JIT per-window compute; windows typically small in count
	chi_sum = None
	for w_i, win in enumerate(windows):
		step_detail = f"v{getattr(win, 'max_val_len', 0)}_c{getattr(win, 'max_cond_len', 0)}"
		with jax_profile.step_annotation("chi0_window", step_num=w_i, detail=step_detail):
			if mask_logs and w_i < len(mask_logs) and mask_logs[w_i]:
				line_val, line_cond = mask_logs[w_i]
				print(line_val)
				print(line_cond)

			chi_win = get_chi_lm_Yt_jax(psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c, win, meta, mesh_xy)
			chi_sum = chi_win if chi_sum is None else (chi_sum + chi_win)
	return chi_sum




def get_static_w_q_jax(
	V_qmunu: jax.Array,
	chi_q: jax.Array,
	S_qmunu: jax.Array | None,
	meta: Meta,
	mesh_xy: Mesh | None = None,
):
	"""Compute static W_q using JAX under k_XY sharding inside a single jit.

	Inputs:
	- V_qmunu: (1, npol1=1, npol2=1, nkx, nky, nkz, nrmu, nrmu)
	- chi_q:   (nkx, nky, nkz, 1, nrmu, 1, nrmu)
	- S_qmunu: (nkx, nky, nkz, nrmu, nrmu) or None (whitening; required for overlap)

	Returns:
	- W_q: (nkx, nky, nkz, 1, nrmu, 1, nrmu) with mu_X,nu_Y sharding
	"""
	@partial(jax.jit, static_argnames=("nkx", "nky", "nkz"))
	def _compute(V_qmunu, chi_q, S_qmunu, nkx: int, nky: int, nkz: int, pref: float):
		# Whitening with overlap S via Cholesky inside jit:
		# S = R^H R, Vbar = R^{-H} V R^{-1}, Chibar = R^{-H} Chi R^{-1}
		# (I - Vbar Chibar) Wbar = Vbar, then W = R^H Wbar R
		# Extract and flatten k-grid → (nq, nrmu, nrmu)
		V_kmn = V_qmunu[0, 0, 0]  # (nkx,nky,nkz,nrmu,nrmu)
		# Flatten k-grid and align shardings to avoid rematerialization
		# Old direct reshape (kept for reference):
		# V_flat = V_kmn.reshape(nkx * nky * nkz, V_kmn.shape[-2], V_kmn.shape[-1])
		# chi_kmn = chi_q[:, :, :, 0, :, 0, :]  # (nkx,nky,nkz,nrmu,nrmu)
		# chi_flat = chi_kmn.reshape(nkx * nky * nkz, chi_kmn.shape[-2], chi_kmn.shape[-1])
		# S_flat = None if S_qmunu is None else S_qmunu.reshape(nkx * nky * nkz, chi_kmn.shape[-2], chi_kmn.shape[-1])
		
		nk_flat = nkx * nky * nkz
		# V: (nkx,nky,nkz, nrmu, nrmu) -> (nk, nrmu, nrmu)
		V_kmn = jax.lax.with_sharding_constraint(
			V_kmn, NamedSharding(mesh_xy, P('x', 'y', None, None, None))
		)
		V_flat = V_kmn.reshape(nk_flat, V_kmn.shape[-2], V_kmn.shape[-1])
		V_flat = jax.lax.with_sharding_constraint(
			V_flat, NamedSharding(mesh_xy, P(('x', 'y'), None, None))
		)
		# chi: (nkx,nky,nkz,nrmu,nrmu) -> (nk, nrmu, nrmu)
		chi_kmn = chi_q[:, :, :, 0, :, 0, :]
		chi_kmn = jax.lax.with_sharding_constraint(
			chi_kmn, NamedSharding(mesh_xy, P('x', 'y', None, None, None))
		)
		chi_flat = chi_kmn.reshape(nk_flat, chi_kmn.shape[-2], chi_kmn.shape[-1])
		chi_flat = jax.lax.with_sharding_constraint(
			chi_flat, NamedSharding(mesh_xy, P(('x', 'y'), None, None))
		)
		# Apply global prefactor to chi (passed from Python to avoid tracer->float issues)
		_pref = jnp.asarray(pref, dtype=chi_flat.dtype)
		chi_flat = _pref * chi_flat
		# S: optional (nkx,nky,nkz,nrmu,nrmu) -> (nk, nrmu, nrmu)
		if S_qmunu is None:
			S_flat = None
		else:
			S_kmn = jax.lax.with_sharding_constraint(
				S_qmunu, NamedSharding(mesh_xy, P('x', 'y', None, None, None))
			)
			S_flat = S_kmn.reshape(nk_flat, chi_kmn.shape[-2], chi_kmn.shape[-1])
			S_flat = jax.lax.with_sharding_constraint(
				S_flat, NamedSharding(mesh_xy, P(('x', 'y'), None, None))
			)
		# Reshard to k_XY (batch sharded across both mesh axes), mu/nu replicated within jit
		if mesh_xy is not None:
			kXY3 = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
			V_flat = jax.lax.with_sharding_constraint(V_flat, kXY3)
			chi_flat = jax.lax.with_sharding_constraint(chi_flat, kXY3)
			if S_flat is not None:
				S_flat = jax.lax.with_sharding_constraint(S_flat, kXY3)
		# Solve (I - (V S^{-1}) χ) W = V if S provided; else (I - V χ) W = V
		n = V_flat.shape[-1]
		if S_flat is not None:
			L = jnp.linalg.cholesky(S_flat)
			# Compute Y = V S^{-1} using two right solves via transpose tricks
			# First right solve by L^H: W1 = V L^{-H}
			W1_T = jsp_linalg.solve_triangular(
				L.conj().transpose(0, 2, 1), V_flat.transpose(0, 2, 1), lower=False
			)
			W1 = W1_T.transpose(0, 2, 1)
			# Then right solve by L: Y = W1 L^{-1}
			Y_T = jsp_linalg.solve_triangular(
				L.transpose(0, 2, 1), W1.transpose(0, 2, 1), lower=False
			)
			Y = Y_T.transpose(0, 2, 1)
			I = jnp.eye(n, dtype=V_flat.dtype)
			A = I - jnp.matmul(Y, chi_flat)
			def solve_one(Ak, Vk):
				lu, piv = jsp_linalg.lu_factor(Ak)
				return jsp_linalg.lu_solve((lu, piv), Vk)
			W_flat = jax.vmap(solve_one, in_axes=(0, 0))(A, V_flat)
		else:
			I = jnp.eye(n, dtype=V_flat.dtype)
			A = I - jnp.matmul(V_flat, chi_flat)
			def solve_one(Ak, Vk):
				lu, piv = jsp_linalg.lu_factor(Ak)
				return jsp_linalg.lu_solve((lu, piv), Vk)
			W_flat = jax.vmap(solve_one, in_axes=(0, 0))(A, V_flat)
		# Reshape back and apply mu_X,nu_Y sharding
		W_kmn = W_flat.reshape(nkx, nky, nkz, n, n)
		W_out = W_kmn[:, :, :, None, :, None, :]
		if mesh_xy is not None:
			W_out = jax.lax.with_sharding_constraint(W_out, NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y')))
		return W_out

	# Compute prefactor outside jit to avoid concretization errors inside the trace
	# Keep user's normalization choice:
	#   pref = 2.0 / (sqrt(Nk) * nspin * nspinor)
	# (if you want 4.0/(Nk*nspin*nspinor), change here and remove sqrt/2 logic)
	_nkx, _nky, _nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
	_Nk = max(1, _nkx * _nky * _nkz)
	_nspin = max(1, int(getattr(meta, 'nspin', 1)))
	_nspinor = max(1, int(getattr(meta, 'nspinor', 1)))
	pref = 2.0 / (math.sqrt(float(_Nk)) * float(_nspin) * float(_nspinor))

	return _compute(V_qmunu, chi_q, S_qmunu, _nkx, _nky, _nkz, pref)

import numpy as np
import math
from ..common.gpu_utils import cp, xp
from ..common.wfnreader import WFNReader
from ..common.gamma_matrices import gammas_sparse
from ..common import Meta
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

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

	# Extract window params
	vmin = jnp.asarray(win.val_window.start_energy)
	vmax = jnp.asarray(win.val_window.end_energy)
	cmin = jnp.asarray(win.cond_window.start_energy)
	cmax = jnp.asarray(win.cond_window.end_energy)
	tau_i = jnp.asarray(win.tau_i, dtype=jnp.float64)
	z_lm = jnp.asarray(win.z_lm, dtype=jnp.complex128)
	w_i = jnp.asarray(win.w_i, dtype=jnp.complex128)

	# Define explicit shardings to avoid expensive auto-SPMD when mesh is provided
	_xt_shard = _y_shard = _x_shard = _yt_shard = _out_shard = None
	_chiR_shard = None
	if mesh_xy is not None:
		_xt_shard = NamedSharding(mesh_xy, P(None, None, 'x', None))
		_y_shard  = NamedSharding(mesh_xy, P(None, None, None, 'y'))
		_x_shard  = NamedSharding(mesh_xy, P(None, None, None, 'x'))
		_yt_shard = NamedSharding(mesh_xy, P(None, None, 'y', None))
		_out_shard = NamedSharding(mesh_xy, P(None, None, None, None, 'x', None, 'y'))
		_chiR_shard = NamedSharding(mesh_xy, P('x', 'y', None, None, None))

	@partial(
		jax.jit,
		static_argnames=("nkx", "nky", "nkz"),
		in_shardings=(
			_xt_shard,  # psi_vTX
			_y_shard,   # psi_vY
			_x_shard,   # psi_cX
			_yt_shard,  # psi_cTY
			None,       # enk_v
			None,       # enk_c
			None, None, None, None,  # vmin,vmax,cmin,cmax
			None,       # tau_i
			None,       # z_lm
			None,       # w_i
		),
		out_shardings=_out_shard,
	)
	def _compute(
		psi_vTX: jax.Array,
		psi_vY: jax.Array,
		psi_cX: jax.Array,
		psi_cTY: jax.Array,
		enk_v: jax.Array,
		enk_c: jax.Array,
		vmin, vmax, cmin, cmax, tau_i, z_lm, w_i,
		nkx: int, nky: int, nkz: int,
	) -> jax.Array:
		# Masks for energy windows
		val_mask = (enk_v >= vmin) & (enk_v <= vmax)  # (nk, nb_v)
		cond_mask = (enk_c >= cmin) & (enk_c <= cmax) # (nk, nb_c)
		# Quadrature weights per tau (complex128)
		quad_w = -2.0 * z_lm * w_i * jnp.exp(-(z_lm * (cmin - vmax) - 1.0) * tau_i)

		@partial(jax.jit)
		def compute_G_from_psi(psiTX: jax.Array, psiY: jax.Array, weights: jax.Array) -> jax.Array:
			"""Zero-comm G build using sharded valence/conduction pairs.
			psiTX: (nk, s, x, m), psiY: (nk, m, t, y), weights: (nk, m)
			Returns G_k: (nk, s, x, t, y)."""
			w = weights.astype(jnp.complex128)
			return jnp.einsum('ksxm,km,kmty->ksxty', psiTX, w, jnp.conj(psiY), optimize=True)

		@partial(jax.jit)
		def k_to_R(G_k: jax.Array, flip_sign: bool) -> jax.Array:
			"""(nk, s, m, t, n) -> (s, m, t, n, nkx, nky, nkz).
			If flip_sign is True, compute G(-R) via FFT; otherwise G(R) via IFFT (both ortho).
			"""
			G = G_k.reshape(nkx, nky, nkz, *G_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
			G = jnp.array(G, copy=True)
			return jax.lax.cond(
				flip_sign,
				lambda X: jnp.fft.fftn(X, axes=(-3, -2, -1), norm='ortho'),
				lambda X: jnp.fft.ifftn(X, axes=(-3, -2, -1), norm='ortho'),
				G,
			)

		def tau_body(carry, itau):
			chi_R_acc = carry
			tau = tau_i[itau]
			# exponentials with masking
			exp_v = jnp.exp(-z_lm * tau * (vmax - enk_v)) * val_mask  # (nk, nb_v)
			exp_c = jnp.exp(-z_lm * tau * (enk_c - cmin)) * cond_mask  # (nk, nb_c)
			Gv_k_shard = NamedSharding(mesh_xy, P(None, None, 'x', None, 'y'))
			Gc_k_shard = NamedSharding(mesh_xy, P(None, None, 'y', None, 'x'))
			Gv_k = jax.lax.with_sharding_constraint(compute_G_from_psi(jnp.conj(psi_vTX), jnp.conj(psi_vY), exp_v), Gv_k_shard)
			Gc_k = jax.lax.with_sharding_constraint(compute_G_from_psi(jnp.conj(psi_cTY), jnp.conj(psi_cX), exp_c), Gc_k_shard) # constructed backwards! Y,X sharding. should still give chi with no comm
			Gv_R = k_to_R(Gv_k, flip_sign=False) #True. convolving G^v_k-q with G^c_k
			Gc_R = k_to_R(Gc_k, flip_sign=True)
			# Contract over spin indices (a,b)
			chi_tau = jnp.einsum('ambnxyz, bnamxyz-> mnxyz', Gc_R, Gv_R, optimize=True)
			chi_R_acc = chi_R_acc + quad_w[itau] * chi_tau
			return chi_R_acc, None

		chi_R0 = jnp.zeros((nrmu, nrmu, nkx, nky, nkz), dtype=jnp.complex128)
		if _chiR_shard is not None:
			chi_R0 = jax.lax.with_sharding_constraint(chi_R0, _chiR_shard)
		(chi_R, _) = jax.lax.scan(tau_body, chi_R0, jnp.arange(tau_i.shape[0]))
		chi_R = jnp.array(chi_R, copy=True)
		chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm='ortho')  # (nrmu1,nrmu2,nkx,nky,nkz)
		# Reorder and add npol dims → (nkx,nky,nkz,1,nrmu1,1,nrmu2)
		chi_q = chi_q.transpose(2, 3, 4, 0, 1)
		chi_q = chi_q[:, :, :, None, :, None, :]
		return chi_q

	chi_q = _compute(
		psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
		vmin, vmax, cmin, cmax, tau_i, z_lm, w_i,
		nkx, nky, nkz,
	)
	# Optional sharding on (nrmu1,nrmu2) as (x,y)
	if mesh_xy is not None:
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
	# JIT per-window compute; windows typically small in count
	chi_sum = None
	for w_i, win in enumerate(windows):
		# Rank-0 diagnostic: show which bands are included at k-point 0
		if jax.process_index() == 0:
			try:
				vmin = float(win.val_window.start_energy)
				vmax = float(win.val_window.end_energy)
				cmin = float(win.cond_window.start_energy)
				cmax = float(win.cond_window.end_energy)
				# Fetch k=0 energies to host for display
				ener_v0 = np.asarray(enk_v[0])
				ener_c0 = np.asarray(enk_c[0])
				val_mask = (ener_v0 >= vmin) & (ener_v0 <= vmax)
				cond_mask = (ener_c0 >= cmin) & (ener_c0 <= cmax)
				def _mask_bar(mask: np.ndarray) -> str:
					return ''.join('X' if bool(b) else '_' for b in mask.tolist())
				print(f"[window {w_i}] k=0 valence bands:  " + _mask_bar(val_mask))
				print(f"[window {w_i}] k=0 conduction bands:" + _mask_bar(cond_mask))
			except Exception:
				pass

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

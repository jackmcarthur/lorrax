import numpy as np
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
	return chi_sum * 0.1111111


# # do chi_lm,0(r,r',Yt) = \sum_ab Gc_lm,R(ra,r'b,Yt)Gv_lm,-R(r'b,ra,-Yt) (a,b=spin indices)
# # Now applies quadrature weights and returns integrated result
# def get_chi_lm_Yt(psi_v, psi_c, win, meta: Meta, wfn, xp):
# 	ntau = win.ntau
# 	nspinor = psi_v.psi.shape("nspinor")
# 	npol = 4 if nspinor == 4 else 1  # JM: check this
# 	nrmu = psi_v.psi.shape("nrmu")
# 	psi_v.psi.join("nspinor", "nrmu")
# 	psi_c.psi.join("nspinor", "nrmu")

# 	# Now G arrays only need to hold one tau at a time
# 	Gv_lm = LabeledArray(
# 		shape=(meta.nkx, meta.nky, meta.nkz, nspinor, nrmu, nspinor, nrmu),
# 		axes=("nkx", "nky", "nkz", "nspinor1", "nrmu1", "nspinor2", "nrmu2"),
# 	)
# 	Gc_lm = LabeledArray(
# 		shape=(meta.nkx, meta.nky, meta.nkz, nspinor, nrmu, nspinor, nrmu),
# 		axes=("nkx", "nky", "nkz", "nspinor1", "nrmu1", "nspinor2", "nrmu2"),
# 	)

# 	# Create integrated chi array (no tau dimension) and compute quadrature weights
# 	chi_lm_integrated = LabeledArray(
# 		shape=(npol, nrmu, npol, nrmu, meta.nkx, meta.nky, meta.nkz),
# 		axes=("npol1", "nrmu1", "npol2", "nrmu2", "nkx", "nky", "nkz"),
# 	)
# 	chi_lm_integrated.data[:] = 0.0  # Initialize to zero

# 	# Precompute quadrature weights: -2 z_lm w_i exp(-(z_lm (E_c - E_v) - 1) tau_i)
# 	quad_weights = xp.asarray(
# 		-2.0
# 		* win.z_lm
# 		* win.w_i
# 		* np.exp(
# 			-(
# 				win.z_lm * (win.cond_window.start_energy - win.val_window.end_energy)
# 				- 1.0
# 			)
# 			* win.tau_i
# 		),
# 		dtype=xp.complex128,
# 	)

# 	Gv_lm.join("nkx", "nky", "nkz")
# 	Gc_lm.join("nkx", "nky", "nkz")
# 	Gv_lm.join("nspinor1", "nrmu1")
# 	Gv_lm.join("nspinor2", "nrmu2")
# 	Gc_lm.join("nspinor1", "nrmu1")
# 	Gc_lm.join("nspinor2", "nrmu2")

# 	# Precompute masks and find maximum number of bands in any window
# 	val_mask_all = (psi_v.enk.data >= win.val_window.start_energy) & (
# 		psi_v.enk.data <= win.val_window.end_energy
# 	)
# 	cond_mask_all = (psi_c.enk.data >= win.cond_window.start_energy) & (
# 		psi_c.enk.data <= win.cond_window.end_energy
# 	)

# 	max_val_bands = int(xp.max(xp.sum(val_mask_all, axis=1)))
# 	max_cond_bands = int(xp.max(xp.sum(cond_mask_all, axis=1)))

# 	nk = psi_v.psi.shape("nk")
# 	norb = psi_v.psi.shape("nspinor*nrmu")

# 	# Allocate compressed arrays - only store bands within energy windows
# 	psi_v_masked = xp.zeros((nk, max_val_bands, norb), dtype=psi_v.psi.data.dtype)
# 	psi_v_conj = xp.zeros((nk, norb, max_val_bands), dtype=psi_v.psi.data.dtype)
# 	psi_c_masked = xp.zeros((nk, norb, max_cond_bands), dtype=psi_c.psi.data.dtype)
# 	psi_c_conj = xp.zeros((nk, max_cond_bands, norb), dtype=psi_c.psi.data.dtype)

# 	# Also compress the energy arrays to match
# 	enk_v_compressed = xp.zeros((nk, max_val_bands), dtype=psi_v.enk.data.dtype)
# 	enk_c_compressed = xp.zeros((nk, max_cond_bands), dtype=psi_c.enk.data.dtype)

# 	# Fill compressed arrays with only the bands within energy windows
# 	for ik in range(nk):
# 		val_indices = xp.where(val_mask_all[ik])[0]
# 		cond_indices = xp.where(cond_mask_all[ik])[0]

# 		if len(val_indices) > 0:
# 			psi_v_masked[ik, : len(val_indices)] = psi_v.psi.data[ik, val_indices]
# 			psi_v_conj[ik, :, : len(val_indices)] = xp.conj(
# 				psi_v.psi.data[ik, val_indices]
# 			).T
# 			enk_v_compressed[ik, : len(val_indices)] = psi_v.enk.data[ik, val_indices]

# 		if len(cond_indices) > 0:
# 			psi_c_conj[ik, : len(cond_indices)] = xp.conj(
# 				psi_c.psi.data[ik, cond_indices]
# 			)
# 			psi_c_masked[ik, :, : len(cond_indices)] = psi_c.psi.data[
# 				ik, cond_indices
# 			].T
# 			enk_c_compressed[ik, : len(cond_indices)] = psi_c.enk.data[ik, cond_indices]

# 	# Make arrays contiguous for optimal performance
# 	psi_v_masked = xp.ascontiguousarray(psi_v_masked)
# 	psi_v_conj = xp.ascontiguousarray(psi_v_conj)
# 	psi_c_masked = xp.ascontiguousarray(psi_c_masked)
# 	psi_c_conj = xp.ascontiguousarray(psi_c_conj)

# 	# Allocate temporary exponential arrays for each tau iteration
# 	exp_v_tmp = xp.zeros((nk, max_val_bands), dtype=xp.complex128)
# 	exp_c_tmp = xp.zeros((nk, max_cond_bands), dtype=xp.complex128)

# 	# Loop over tau values to save memory
# 	for itau in range(ntau):
# 		tau_val = win.tau_i[itau]

# 		# Compute exponentials directly from compressed energy arrays into temporary arrays
# 		exp_v_tmp[:] = xp.exp(
# 			-win.z_lm * tau_val * (win.val_window.end_energy - enk_v_compressed)
# 		)
# 		exp_c_tmp[:] = xp.exp(
# 			-win.z_lm * tau_val * (enk_c_compressed - win.cond_window.start_energy)
# 		)

# 		# Apply exponential to compressed arrays - note shapes:
# 		# psi_v_conj(nk,norb,max_val_bands), psi_c_conj(nk,max_cond_bands,norb)
# 		xp.multiply(
# 			psi_v_conj, exp_v_tmp[:, np.newaxis, :], out=psi_v_conj
# 		)  # Apply exponential
# 		xp.multiply(
# 			psi_c_conj, exp_c_tmp[:, :, np.newaxis], out=psi_c_conj
# 		)  # Apply exponential

# 		# Efficient batched matmuls without any transposes:
# 		# psi_v_conj(nk,norb,max_val_bands) @ psi_v_masked(nk,max_val_bands,norb) -> (nk,norb,norb)
# 		Gv_lm.data = xp.matmul(psi_v_conj, psi_v_masked)
# 		# psi_c_masked(nk,norb,max_cond_bands) @ psi_c_conj(nk,max_cond_bands,norb) -> (nk,norb,norb)
# 		Gc_lm.data = xp.matmul(psi_c_masked, psi_c_conj)

# 		# Remove exponential from arrays for reuse
# 		xp.divide(
# 			psi_v_conj, exp_v_tmp[:, np.newaxis, :], out=psi_v_conj
# 		)  # Remove exponential
# 		xp.divide(
# 			psi_c_conj, exp_c_tmp[:, :, np.newaxis], out=psi_c_conj
# 		)  # Remove exponential

# 		# No tau-specific arrays to free now
# 		if hasattr(xp, "get_default_memory_pool"):
# 			xp.get_default_memory_pool().free_all_blocks()

# 		# Transform to real space for this tau
# 		Gv_lm.unjoin("nkx", "nky", "nkz")
# 		Gv_lm = Gv_lm.kgrid_to_last()
# 		Gv_lm.ifft_kgrid()  # G_k -> G_R

# 		Gc_lm.unjoin("nkx", "nky", "nkz")
# 		Gc_lm = Gc_lm.kgrid_to_last()
# 		Gc_lm.ifft_kgrid()

# 		# flip Gv_R -> Gv_-R, keeping Gv_R=0 in the 0th index
# 		for ik in range(2, 5):
# 			Gv_lm.data = xp.flip(Gv_lm.data, axis=ik)
# 			Gv_lm.data = xp.roll(Gv_lm.data, 1, axis=ik)

# 		Gv_lm.unjoin("nspinor1", "nrmu1")
# 		Gv_lm.unjoin("nspinor2", "nrmu2")
# 		Gc_lm.unjoin("nspinor1", "nrmu1")
# 		Gc_lm.unjoin("nspinor2", "nrmu2")

# 		# Compute chi contribution for this tau and accumulate with quadrature weight
# 		current_weight = quad_weights[itau]

# 		if npol == 4:
# 			scratch = xp.empty_like(Gc_lm.slice_many({"nspinor1": 0, "nspinor2": 0}))

# 			for I, (rI, cI, vI) in enumerate(gammas_sparse):
# 				for J, (rJ, cJ, vJ) in enumerate(gammas_sparse):
# 					target = chi_lm_integrated.data[I, :, J, :, :, :, :]
# 					for p in range(len(vI)):
# 						a = int(rI[p])
# 						c = int(cI[p])
# 						gI = vI[p]
# 						for q in range(len(vJ)):
# 							b = int(rJ[q])
# 							d = int(cJ[q])
# 							gJ = vJ[q]
# 							xp.multiply(
# 								Gc_lm.slice_many({"nspinor1": a, "nspinor2": b}),
# 								Gv_lm.slice_many({"nspinor1": c, "nspinor2": d}),
# 								out=scratch,
# 							)
# 							# Apply quadrature weight and accumulate
# 							xp.add(
# 								target, current_weight * gI * gJ * scratch, out=target
# 							)
# 		else:
# 			# TODO: NOTE CHANGE! swapping r,r' means that a,b go from ab,ba to ab,ab. need to adjust bispinor case to account for this
# 			for a in range(nspinor):
# 				for b in range(nspinor):
# 					chi_contribution = xp.multiply(
# 						Gc_lm.slice_many({"nspinor1": a, "nspinor2": b}),
# 						Gv_lm.slice_many({"nspinor1": a, "nspinor2": b}),
# 					)
# 					# Apply quadrature weight and accumulate
# 					chi_lm_integrated.data[0, :, 0, :, :, :, :] += (
# 						current_weight * chi_contribution
# 					)

# 		# Prepare for next tau iteration - rejoin for k-space operations
# 		Gv_lm.join("nspinor1", "nrmu1")
# 		Gv_lm.join("nspinor2", "nrmu2")
# 		Gc_lm.join("nspinor1", "nrmu1")
# 		Gc_lm.join("nspinor2", "nrmu2")
# 		Gv_lm.join("nkx", "nky", "nkz")
# 		Gc_lm.join("nkx", "nky", "nkz")
# 		# TODO: this is wasteful data rearrangement because the data doesn't matter, only the axes
# 		Gc_lm = Gc_lm.transpose("nkx*nky*nkz", "nspinor1*nrmu1", "nspinor2*nrmu2")
# 		Gv_lm = Gv_lm.transpose("nkx*nky*nkz", "nspinor1*nrmu1", "nspinor2*nrmu2")

# 	# Clean up masks and reused arrays
# 	del val_mask_all, cond_mask_all, psi_v_masked, psi_v_conj, psi_c_masked, psi_c_conj
# 	del exp_v_tmp, exp_c_tmp, enk_v_compressed, enk_c_compressed
# 	if hasattr(xp, "get_default_memory_pool"):
# 		xp.get_default_memory_pool().free_all_blocks()

# 	psi_v.psi.unjoin("nspinor", "nrmu")
# 	psi_c.psi.unjoin("nspinor", "nrmu")

# 	# note it would be more efficient to only fft chi0 in get_chi0
# 	chi_lm_integrated.fft_kgrid()  # chi_R -> chi_q
# 	chi_out = chi_lm_integrated.transpose(
# 		"nkx", "nky", "nkz", "npol1", "nrmu1", "npol2", "nrmu2"
# 	)
# 	oneoverkgrid = xp.complex128(
# 		np.power(np.complex128(meta.nkx * meta.nky * meta.nkz), -0.5)
# 	)
# 	xp.multiply(chi_out.data, oneoverkgrid, out=chi_out.data)
# 	# xp.multiply(chi_out.data, 0.45, out=chi_out.data)
# 	print("one chi_lm element ", chi_out.data[0, 0, 0, 0, 0, 0, 0].item())
# 	return chi_out.data


# # sums contributions from all windows
# def get_chi0(psi_v, psi_c, windows, meta: Meta, wfn, xp):
# 	nspinor = psi_v.psi.shape("nspinor")
# 	npol = 4 if nspinor == 4 else 1
# 	nrmu = psi_v.psi.shape("nrmu")
# 	chi0 = LabeledArray(
# 		shape=(1, meta.nkx, meta.nky, meta.nkz, npol, nrmu, npol, nrmu),
# 		axes=("ntau", "nkx", "nky", "nkz", "npol1", "nrmu1", "npol2", "nrmu2"),
# 	)
# 	# chi0.join('nkx', 'nky', 'nkz')
# 	# chi0.join('nspinor1', 'nrmu1')
# 	# chi0.join('nspinor2', 'nrmu2')

# 	for win in windows:
# 		chi_lm_integrated = get_chi_lm_Yt(psi_v, psi_c, win, meta, wfn, xp)
# 		# Quadrature weights are now applied inside get_chi_lm_Yt, so just add the result
# 		xp.add(
# 			chi0.data[0, :, :, :, :, :, :, :],
# 			chi_lm_integrated,
# 			out=chi0.data[0, :, :, :, :, :, :, :],
# 		)

# 	chi = chi0.transpose(
# 		"nkx", "nky", "nkz", "ntau", "npol1", "nrmu1", "npol2", "nrmu2"
# 	)
# 	return chi


# def get_static_w_q(
# 	chi_q, Vq, meta: Meta, wfn, sym, xp, n_mult=10, block_f=1, bispinor=False
# ):
# 	# w_q(omega) = (1-v_q @ chi_q)^{-1} @ v_q
# 	# This implementation performs the CTSP matrix inversion in the static limit.
# 	# Once the frequency mesh is restored this routine will compute W(omega) on
# 	# the full imaginary-time grid.
# 	# if A = v_q @ chi_q, then (1-A)^{-1} = 1 + A + A^2 + A^3 + ... (iterative matrix inversion faster + more stable than direct)
# 	# A^N is done with blocked GEMMs along the frequency axis; since we currently do COHSEX we set block_q=1

# 	# if bispinor:

# 	# die because no chi_munu = gamma_mu gamma_nu G G yet
# 	#    raise ValueError("bispinor not implemented yet")
# 	npol_w = chi_q.shape("npol1")
# 	nrmu = chi_q.shape("nrmu1")
# 	print("one chi element: ", chi_q.data[0, 0, 0, 0, 0, 0, 0, 0].item())

# 	# does not matter if bispinor or not
# 	V_q = Vq.transpose("nfreq", "nkx", "nky", "nkz", "npol1", "nrmu1", "npol2", "nrmu2")
# 	V_q.join("nkx", "nky", "nkz")
# 	V_q.join("npol1", "nrmu1")
# 	V_q.join("npol2", "nrmu2")

# 	W_q = LabeledArray(
# 		shape=(meta.nkx, meta.nky, meta.nkz, 1, npol_w, nrmu, npol_w, nrmu),
# 		axes=("nkx", "nky", "nkz", "nfreq", "npol1", "nrmu1", "npol2", "nrmu2"),
# 	)
# 	W_q.join("nkx", "nky", "nkz")
# 	W_q.join("npol1", "nrmu1")
# 	W_q.join("npol2", "nrmu2")

# 	chi_q.join("nkx", "nky", "nkz")
# 	chi_q.join("npol1", "nrmu1")
# 	chi_q.join("npol2", "nrmu2")

# 	nk_tot, nfreq, N, _ = chi_q.data.shape

# 	# pick a block‐size along the frequency axis
# 	if block_f is None:
# 		# e.g. cap at 128 MB of scratch:
# 		max_bytes = 128 * 1024**2
# 		per_mat = 16 * N * N  # bytes per (N×N) complex128
# 		block_f = max(1, int(max_bytes // per_mat))
# 	block_f = min(block_f, nfreq)

# 	# allocate scratch buffers once
# 	A = xp.empty((block_f, N, N), dtype=xp.complex128)
# 	Wb = xp.empty((block_f, N, N), dtype=xp.complex128)
# 	P = xp.empty((block_f, N, N), dtype=xp.complex128)
# 	I = xp.eye(N, dtype=xp.complex128)[None, :, :]

# 	# loop over q‐points
# 	for iq in range(nk_tot):
# 		Vf = V_q.data[0, iq]  # shape = (N, N)
# 		ch = chi_q.data[iq]  # shape = (nfreq, N, N)
# 		# Wf = W_q.data[iq]    # shape = (nfreq, N, N)

# 		# chunk over freq‐axis
# 		# for f0 in range(0, nfreq, block_f):
# 		#     f1 = min(f0+block_f, nfreq)
# 		#     B  = f1 - f0

# 		#     cb = ch[f0:f1]      # (B, N, N)
# 		#     wb = Wb[:B]         # view into scratch
# 		#     a  = A[:B]

# 		#     # 1) A := Vb @ cb
# 		#     xp.matmul(Vf, cb, out=a)

# 		#     # 2) Wb := I + A
# 		#     wb[:] = I           # broadcast eye
# 		#     wb += a

# 		#     # 3) Build powers A^2 … A^(n_mult+1)
# 		#     # P = a.copy()        # P == A^1
# 		#     # for _ in range(n_mult):
# 		#     #     xp.matmul(P, cb, out=P)
# 		#     #     wb += P
# 		#     #     cb = a.copy() # chi array now contains vchi
# 		#     #     for _ in range(n_mult-1):
# 		#     #         xp.matmul(cb, a, out=P)
# 		#     #         wb += P
# 		#     #         cb = P.copy()
# 		#     #         #print('mtx norm P: ', xp.linalg.norm(P))
# 		#     #     # 4) Multiply by Vb → W = (1 - Vχ)^(-1) V
# 		#     #     xp.matmul(wb, Vf, out=Wf[f0:f1])

# 		#     # 5) write‐back
# 		#     #Wf[f0:f1] = wb

# 		W_q.data[iq] = xp.matmul(xp.linalg.inv(I - xp.matmul(Vf, ch)), Vf)

# 	W_q.unjoin("nkx", "nky", "nkz")
# 	W_q.kgrid_to_last()
# 	# W_q.ifft_kgrid() # W_q -> W_R
# 	W_q.unjoin("npol1", "nrmu1")
# 	W_q.unjoin("npol2", "nrmu2")
# 	# could do W_q -> W_R here but it's already done in the get_sigma function
# 	W = W_q.transpose("nfreq", "nkx", "nky", "nkz", "npol1", "nrmu1", "npol2", "nrmu2")

# 	V_q.unjoin("nkx", "nky", "nkz")

# 	V_q.unjoin("npol1", "nrmu1")
# 	V_q.unjoin("npol2", "nrmu2")

# 	return W


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
	def _compute(V_qmunu, chi_q, S_qmunu, nkx: int, nky: int, nkz: int):
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

	return _compute(V_qmunu, chi_q, S_qmunu, int(meta.nkx), int(meta.nky), int(meta.nkz))
